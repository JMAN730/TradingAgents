"""Claude Agent SDK provider: run LLM calls through a Claude Code subscription.

Auth: no ANTHROPIC_API_KEY needed. The Agent SDK spawns the bundled Claude
Code CLI, which resolves credentials in this order: ANTHROPIC_API_KEY ->
CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) -> stored subscription
OAuth from `claude /login`. A Pro/Max user who is logged into Claude Code
needs no configuration at all.

The SDK does not expose temperature/max_tokens/max_retries; those kwargs are
accepted and ignored so cross-provider config keys don't break construction.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def _sdk():
    """Import claude_agent_sdk lazily so the core install stays lean."""
    try:
        import claude_agent_sdk
    except ImportError as e:
        raise ImportError(
            "claude-agent-sdk is required for the claude_code provider. "
            "Install it with: pip install 'tradingagents[claude-code]' "
            "(or: pip install claude-agent-sdk). Then log in once with "
            "`claude /login` (Pro/Max subscription) or set CLAUDE_CODE_OAUTH_TOKEN."
        ) from e
    if claude_agent_sdk is None:  # monkeypatched-away in tests
        raise ImportError("claude-agent-sdk unavailable (install 'tradingagents[claude-code]')")
    return claude_agent_sdk


def _flatten_content(content: Any) -> str:
    """Normalize langchain message content (str or block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _split_messages(messages: list[BaseMessage]) -> tuple[str | None, str]:
    """Split langchain messages into (system_prompt, transcript_prompt).

    The Agent SDK takes a single prompt string plus a system prompt option,
    so conversation history is serialized as a labeled transcript.
    """
    system_parts: list[str] = []
    transcript: list[str] = []
    for m in messages:
        text = _flatten_content(m.content)
        if isinstance(m, SystemMessage):
            system_parts.append(text)
        elif isinstance(m, HumanMessage):
            transcript.append(f"User: {text}")
        elif isinstance(m, AIMessage):
            transcript.append(f"Assistant: {text}")
        elif isinstance(m, ToolMessage):
            transcript.append(f"Tool result: {text}")
        else:
            transcript.append(f"{m.type}: {text}")
    system = "\n\n".join(system_parts) or None
    prompt = "\n\n".join(transcript) or "Continue with the task described in the system prompt."
    return system, prompt


def _run_async(coro):
    """Run a coroutine from sync code, even when an event loop is running.

    LangGraph drives the graph synchronously, but callers may embed it in an
    async app; running the SDK coroutine on a worker thread covers both.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()
