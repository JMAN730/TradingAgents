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
import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from .base_client import BaseLLMClient


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


# Claude Code built-in tools; always disallowed so the trading graph's SDK
# calls can't touch the local filesystem, shell, or web.
#
# Hermeticity here is blocklist-based, not allowlist-based: permission_mode
# is "bypassPermissions" and any builtin tool NOT named below remains
# callable. This list must be kept in sync with Claude Code's built-in tool
# additions -- a new builtin ships enabled-by-default until added here.
_BUILTIN_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebFetch", "WebSearch", "NotebookEdit", "TodoWrite", "Task",
]

_AUTH_HINT = (
    "Claude Agent SDK call failed. If this is an auth problem, log in once "
    "with `claude /login` (Pro/Max subscription) or run `claude setup-token` "
    "and export CLAUDE_CODE_OAUTH_TOKEN. The `claude` CLI must be installed."
)


class ChatClaudeCode(BaseChatModel):
    """Langchain chat model backed by the Claude Agent SDK (subscription auth).

    temperature / max_retries / timeout are accepted for cross-provider config
    compatibility but the Agent SDK does not expose them; they are ignored.
    """

    model: str
    effort: str | None = None
    max_tool_turns: int = 12
    temperature: float | None = None   # ignored (not supported by Agent SDK)
    max_retries: int | None = None     # ignored
    timeout: float | None = None       # ignored

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    def _build_options(self, system_prompt: str | None, kwargs: dict):
        sdk = _sdk()
        opts: dict[str, Any] = {
            "system_prompt": system_prompt,
            "model": self.model,
            "setting_sources": [],
            "permission_mode": "bypassPermissions",
            "allowed_tools": [],
            "disallowed_tools": list(_BUILTIN_TOOLS),
            "max_turns": 1,
        }
        if self.effort:
            opts["effort"] = self.effort
        schema = kwargs.get("structured_schema")
        if schema is not None:
            opts["output_format"] = {"type": "json_schema", "schema": schema}
        lc_tools = kwargs.get("langchain_tools")
        if lc_tools:
            server, tool_names = _build_tool_server(lc_tools)
            opts["mcp_servers"] = {"toolkit": server}
            opts["allowed_tools"] = tool_names
            opts["max_turns"] = self.max_tool_turns
        return sdk.ClaudeAgentOptions(**opts)

    async def _acall_sdk(self, prompt: str, options) -> tuple[str, dict | None]:
        sdk = _sdk()
        text_parts: list[str] = []
        final_text = ""
        structured: dict | None = None
        subtype = "success"
        async for msg in sdk.query(prompt=prompt, options=options):
            if isinstance(msg, sdk.AssistantMessage):
                for block in msg.content:
                    if isinstance(block, sdk.TextBlock):
                        text_parts.append(block.text)
            elif isinstance(msg, sdk.ResultMessage):
                structured = getattr(msg, "structured_output", None)
                final_text = getattr(msg, "result", "") or ""
                subtype = getattr(msg, "subtype", "success")
        if subtype not in (None, "success"):
            raise RuntimeError(
                f"Claude Agent SDK run ended with subtype={subtype!r}; refusing to treat "
                "partial output as a final response. If this is a tool-loop turn-budget "
                "exhaustion (error_max_turns), consider raising max_tool_turns."
            )
        if not final_text:
            final_text = "\n".join(text_parts)
        return final_text, structured

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, prompt = _split_messages(messages)
        options = self._build_options(system_prompt, kwargs)
        try:
            final_text, structured = _run_async(self._acall_sdk(prompt, options))
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(f"{_AUTH_HINT} Underlying error: {e}") from e
        if kwargs.get("structured_schema") is not None and structured is not None:
            content = json.dumps(structured)
        else:
            content = final_text
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Route through the Agent SDK's json_schema output_format."""
        if include_raw:
            raise NotImplementedError(
                "include_raw=True is not supported by the claude_code provider"
            )
        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()

            def _parse(message: AIMessage):
                return schema.model_validate(json.loads(message.content))
        else:
            json_schema = schema

            def _parse(message: AIMessage):
                return json.loads(message.content)

        bound = self.bind(structured_schema=json_schema)
        return bound | RunnableLambda(_parse)

    def bind_tools(self, tools, **kwargs):
        """Expose langchain tools to the SDK's internal agent loop.

        The SDK executes the whole tool loop in one call, so responses carry
        no tool_calls and LangGraph's ToolNode cycle never fires — analysts
        receive the final report directly.
        """
        return self.bind(langchain_tools=list(tools))


def _tool_schema(lc_tool) -> dict:
    """Best-effort JSON schema for a langchain tool's arguments."""
    args_schema = getattr(lc_tool, "tool_call_schema", None) or getattr(lc_tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()
    return {"type": "object", "additionalProperties": True}


def _build_tool_server(lc_tools):
    """Wrap langchain tools in an in-process SDK MCP server."""
    sdk = _sdk()
    sdk_tools = []
    tool_names = []

    for lc_tool in lc_tools:
        @sdk.tool(lc_tool.name, lc_tool.description or lc_tool.name, _tool_schema(lc_tool))
        async def handler(args: dict, _lc_tool=lc_tool):
            try:
                result = await asyncio.to_thread(_lc_tool.invoke, args)
                return {"content": [{"type": "text", "text": str(result)}]}
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}

        sdk_tools.append(handler)
        tool_names.append(f"mcp__toolkit__{lc_tool.name}")

    server = sdk.create_sdk_mcp_server(name="toolkit", version="1.0.0", tools=sdk_tools)
    return server, tool_names


class ClaudeCodeClient(BaseLLMClient):
    """Provider client for Claude via a Claude Code subscription (Agent SDK).

    Keyless: auth comes from `claude /login` OAuth or CLAUDE_CODE_OAUTH_TOKEN.
    """

    provider = "claude_code"

    _ACCEPTED_KWARGS = ("effort", "max_tool_turns", "temperature", "max_retries", "timeout")

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        kwargs = {k: v for k, v in self.kwargs.items() if k in self._ACCEPTED_KWARGS}
        return ChatClaudeCode(model=self.model, **kwargs)

    def validate_model(self) -> bool:
        from .validators import validate_model

        return validate_model("claude_code", self.model)
