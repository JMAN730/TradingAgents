# Claude Code (Subscription) LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `claude_code` LLM provider that routes all TradingAgents LLM calls through the Claude Agent SDK, so a user with a Claude Pro/Max subscription (logged into Claude Code, no `ANTHROPIC_API_KEY`) can run the full trading graph.

**Architecture:** A new `ChatClaudeCode(BaseChatModel)` langchain chat model wraps `claude_agent_sdk.query()`. Text calls are one-shot (`max_turns=1`). Structured output maps to the SDK's `output_format={"type": "json_schema", ...}`. Tool-using analysts get their langchain tools bridged into an in-process SDK MCP server — the Agent SDK runs the tool loop internally and returns final text with **no** `tool_calls`, so LangGraph's ToolNode cycle simply never triggers (analysts already treat a no-tool-calls response as the final report). A thin `ClaudeCodeClient(BaseLLMClient)` plus registry/CLI wiring makes it a first-class provider.

**Tech Stack:** Python ≥3.10, `claude-agent-sdk` (optional extra), langchain-core ≥0.3.81, pytest.

## Global Constraints

- Python floor: `>=3.10` (pyproject.toml:10); ruff target `py310`.
- `claude-agent-sdk` must be an **optional** dependency (extra named `claude-code`), lazily imported with an actionable ImportError hint — same pattern as the `bedrock` extra (`bedrock_client.py:23-29`, pyproject `[project.optional-dependencies]`).
- Every test carries a registered marker; use `@pytest.mark.unit` (pytest runs with `--strict-markers`). Unit tests must NOT import the real `claude-agent-sdk` and must NOT hit the network — inject a fake module via `monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)`.
- Provider key string is exactly `"claude_code"` everywhere.
- The SDK does not expose `temperature`/`max_tokens`/`max_retries`; the client must silently accept-and-ignore them (cross-provider kwargs from `trading_graph._get_provider_kwargs` will pass `temperature` and `max_retries`).
- Hermetic SDK calls always set: `setting_sources=[]`, `permission_mode="bypassPermissions"`, built-in tools disallowed (see `_BUILTIN_TOOLS` in Task 2).
- Commit after every green task, conventional-commit style (`feat(llm_clients): ...`), body only when the why isn't obvious.

## Verified facts the plan relies on

- Auth precedence (code.claude.com/docs/en/authentication.md): `ANTHROPIC_API_KEY` → `apiKeyHelper` → `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`, Pro/Max) → stored subscription OAuth from `claude /login` (the default for Pro/Max users). The Agent SDK spawns the bundled Claude Code CLI, which resolves credentials in that order — **no env var needed for a logged-in user**.
- `claude_agent_sdk` API surface: `query(prompt, options) -> AsyncIterator[Message]`, `ClaudeAgentOptions(system_prompt, model, max_turns, allowed_tools, disallowed_tools, permission_mode, setting_sources, mcp_servers, output_format, effort, ...)`, message types `AssistantMessage` (with `content: list[TextBlock|...]`), `ResultMessage` (with `.result: str`, `.structured_output: dict | None`, `.subtype`), tool helpers `tool()` decorator + `create_sdk_mcp_server()`. MCP tool names are `mcp__{server}__{tool}`.
- Downstream contract (from repo): `get_llm()` result must support `.invoke(messages) -> AIMessage`, `.bind_tools(tools)`, `.with_structured_output(schema)`; ToolNode routing keys off `AIMessage.tool_calls` (`conditional_logic.py:14-50`); schema-only agents degrade gracefully on `NotImplementedError` but the market/news/fundamentals analysts call `bind_tools` unconditionally.

## File map

- Create: `tradingagents/llm_clients/claude_code_client.py` — `ChatClaudeCode`, `ClaudeCodeClient`, serialization + async helpers, tool bridge.
- Create: `tests/test_claude_code_provider.py` — all unit tests (one file, matching repo convention of one test file per provider).
- Create: `tests/test_claude_code_smoke.py` — skip-guarded integration smoke test.
- Modify: `tradingagents/llm_clients/factory.py` (new branch), `tradingagents/llm_clients/api_key_env.py` (`"claude_code": None`), `tradingagents/llm_clients/validators.py` (`_ANY_MODEL_PROVIDERS`), `tradingagents/llm_clients/model_catalog.py` (`MODEL_OPTIONS["claude_code"]`), `tradingagents/graph/trading_graph.py` (`_get_provider_kwargs` branch), `cli/utils.py` (provider table row), `pyproject.toml` (optional extra), `README.md`, `CHANGELOG.md`.

---

### Task 1: Module skeleton — message serialization + async runner (no SDK needed)

**Files:**
- Create: `tradingagents/llm_clients/claude_code_client.py`
- Create: `tests/test_claude_code_provider.py`

**Interfaces:**
- Consumes: `langchain_core.messages` types only.
- Produces: `_split_messages(messages) -> tuple[str | None, str]` (system_prompt, prompt) and `_run_async(coro) -> Any`. Later tasks call both. Also module-level `_sdk()` lazy importer raising `ImportError` with install hint.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_claude_code_provider.py
"""Unit tests for the claude_code provider (Claude Agent SDK adapter)."""
import asyncio
import sys
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

pytestmark = pytest.mark.unit


class TestSplitMessages:
    def test_system_separated_from_transcript(self):
        from tradingagents.llm_clients.claude_code_client import _split_messages

        system, prompt = _split_messages([
            SystemMessage(content="You are a trader."),
            HumanMessage(content="Analyze AAPL."),
            AIMessage(content="Looking at data."),
            HumanMessage(content="Continue."),
        ])
        assert system == "You are a trader."
        assert "User: Analyze AAPL." in prompt
        assert "Assistant: Looking at data." in prompt
        assert prompt.index("Analyze AAPL.") < prompt.index("Looking at data.")

    def test_multiple_system_messages_joined(self):
        from tradingagents.llm_clients.claude_code_client import _split_messages

        system, _ = _split_messages([
            SystemMessage(content="Rule one."),
            SystemMessage(content="Rule two."),
            HumanMessage(content="hi"),
        ])
        assert system == "Rule one.\n\nRule two."

    def test_block_list_content_flattened(self):
        from tradingagents.llm_clients.claude_code_client import _split_messages

        _, prompt = _split_messages([
            HumanMessage(content=[{"type": "text", "text": "part1"}, "part2"]),
        ])
        assert "part1" in prompt and "part2" in prompt

    def test_tool_message_labeled(self):
        from tradingagents.llm_clients.claude_code_client import _split_messages

        _, prompt = _split_messages([
            ToolMessage(content="42 rows", tool_call_id="t1"),
        ])
        assert "Tool result: 42 rows" in prompt

    def test_empty_transcript_gets_placeholder(self):
        from tradingagents.llm_clients.claude_code_client import _split_messages

        system, prompt = _split_messages([SystemMessage(content="sys only")])
        assert system == "sys only"
        assert prompt  # non-empty placeholder so SDK gets a prompt


class TestRunAsync:
    def test_runs_coroutine_outside_loop(self):
        from tradingagents.llm_clients.claude_code_client import _run_async

        async def coro():
            return 7

        assert _run_async(coro()) == 7

    def test_runs_coroutine_inside_running_loop(self):
        from tradingagents.llm_clients.claude_code_client import _run_async

        async def inner():
            return 9

        async def outer():
            # simulates being called from sync code within an active event loop
            return _run_async(inner())

        assert asyncio.run(outer()) == 9


class TestLazySdkImport:
    def test_import_error_has_install_hint(self, monkeypatch):
        from tradingagents.llm_clients import claude_code_client

        monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
        with pytest.raises(ImportError, match="claude-code"):
            claude_code_client._sdk()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_claude_code_provider.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'tradingagents.llm_clients.claude_code_client'`

- [ ] **Step 3: Write the implementation**

```python
# tradingagents/llm_clients/claude_code_client.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_claude_code_provider.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/claude_code_client.py tests/test_claude_code_provider.py
git commit -m "feat(llm_clients): scaffold claude_code provider serialization helpers"
```

---

### Task 2: `ChatClaudeCode` text-only generation path

**Files:**
- Modify: `tradingagents/llm_clients/claude_code_client.py`
- Modify: `tests/test_claude_code_provider.py`
- Modify: `pyproject.toml` (add optional extra)

**Interfaces:**
- Consumes: `_split_messages`, `_run_async`, `_sdk` from Task 1.
- Produces: class `ChatClaudeCode(BaseChatModel)` with pydantic fields `model: str`, `effort: str | None = None`, `max_tool_turns: int = 12`, plus ignored `temperature: float | None = None`, `max_retries: int | None = None`, `timeout: float | None = None`. Methods later tasks extend: `_build_options(self, system_prompt, kwargs) -> ClaudeAgentOptions`, `async _acall_sdk(self, prompt, options) -> tuple[str, dict | None]` (final_text, structured_output), `_generate(...)`. `_llm_type` = `"claude-code"`.

- [ ] **Step 1: Add a fake SDK factory + failing tests**

Append to `tests/test_claude_code_provider.py`:

```python
def make_fake_sdk(final_text="FAKE RESULT", structured=None, capture=None):
    """Build a fake claude_agent_sdk module. `capture` (dict) records call args."""
    fake = types.ModuleType("claude_agent_sdk")

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class ResultMessage:
        def __init__(self, result, structured_output=None, subtype="success"):
            self.result = result
            self.structured_output = structured_output
            self.subtype = subtype

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            if capture is not None:
                capture["options"] = kwargs

    async def query(prompt=None, options=None):
        if capture is not None:
            capture["prompt"] = prompt
        yield AssistantMessage([TextBlock("interim narration")])
        yield ResultMessage(final_text, structured_output=structured)

    def tool(name, description, schema):
        def deco(fn):
            fn._sdk_tool = (name, description, schema)
            return fn
        return deco

    def create_sdk_mcp_server(name, version, tools):
        server = types.SimpleNamespace(name=name, version=version, tools=tools)
        if capture is not None:
            capture["mcp_server"] = server
        return server

    fake.TextBlock = TextBlock
    fake.AssistantMessage = AssistantMessage
    fake.ResultMessage = ResultMessage
    fake.ClaudeAgentOptions = ClaudeAgentOptions
    fake.query = query
    fake.tool = tool
    fake.create_sdk_mcp_server = create_sdk_mcp_server
    return fake


class TestChatClaudeCodeText:
    def _model(self, monkeypatch, **fake_kwargs):
        capture = {}
        fake = make_fake_sdk(capture=capture, **fake_kwargs)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import ChatClaudeCode

        return ChatClaudeCode(model="sonnet"), capture

    def test_invoke_returns_ai_message_with_result_text(self, monkeypatch):
        llm, _ = self._model(monkeypatch, final_text="AAPL looks strong.")
        out = llm.invoke([HumanMessage(content="Analyze AAPL")])
        assert isinstance(out, AIMessage)
        assert out.content == "AAPL looks strong."
        assert out.tool_calls == []

    def test_options_are_hermetic_one_shot(self, monkeypatch):
        llm, capture = self._model(monkeypatch)
        llm.invoke([SystemMessage(content="Be terse."), HumanMessage(content="hi")])
        opts = capture["options"]
        assert opts["system_prompt"] == "Be terse."
        assert opts["model"] == "sonnet"
        assert opts["max_turns"] == 1
        assert opts["allowed_tools"] == []
        assert opts["setting_sources"] == []
        assert opts["permission_mode"] == "bypassPermissions"
        assert "Bash" in opts["disallowed_tools"]

    def test_prompt_contains_transcript(self, monkeypatch):
        llm, capture = self._model(monkeypatch)
        llm.invoke([HumanMessage(content="first"), AIMessage(content="second")])
        assert "User: first" in capture["prompt"]
        assert "Assistant: second" in capture["prompt"]

    def test_effort_forwarded_when_set(self, monkeypatch):
        capture = {}
        fake = make_fake_sdk(capture=capture)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import ChatClaudeCode

        ChatClaudeCode(model="opus", effort="high").invoke([HumanMessage(content="x")])
        assert capture["options"]["effort"] == "high"

    def test_effort_omitted_when_unset(self, monkeypatch):
        llm, capture = self._model(monkeypatch)
        llm.invoke([HumanMessage(content="x")])
        assert "effort" not in capture["options"]

    def test_temperature_and_max_retries_accepted_and_ignored(self, monkeypatch):
        capture = {}
        fake = make_fake_sdk(capture=capture)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import ChatClaudeCode

        llm = ChatClaudeCode(model="sonnet", temperature=0.3, max_retries=5)
        llm.invoke([HumanMessage(content="x")])
        assert "temperature" not in capture["options"]
        assert "max_retries" not in capture["options"]

    def test_falls_back_to_assistant_text_when_result_empty(self, monkeypatch):
        llm, _ = self._model(monkeypatch, final_text="")
        out = llm.invoke([HumanMessage(content="x")])
        assert out.content == "interim narration"

    def test_sdk_error_wrapped_with_login_hint(self, monkeypatch):
        capture = {}
        fake = make_fake_sdk(capture=capture)

        async def boom(prompt=None, options=None):
            raise RuntimeError("process exited")
            yield  # pragma: no cover - makes this an async generator

        fake.query = boom
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import ChatClaudeCode

        with pytest.raises(RuntimeError, match="claude /login"):
            ChatClaudeCode(model="sonnet").invoke([HumanMessage(content="x")])
```

- [ ] **Step 2: Run tests to verify the new class tests fail**

Run: `python -m pytest tests/test_claude_code_provider.py -v -k ChatClaudeCodeText`
Expected: FAIL with `ImportError: cannot import name 'ChatClaudeCode'`

- [ ] **Step 3: Implement `ChatClaudeCode`**

Append to `claude_code_client.py`:

```python
import json  # noqa: E402  (keep with other stdlib imports at top of file)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

# Claude Code built-in tools; always disallowed so the trading graph's SDK
# calls can't touch the local filesystem, shell, or web.
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
        async for msg in sdk.query(prompt=prompt, options=options):
            if isinstance(msg, sdk.AssistantMessage):
                for block in msg.content:
                    if isinstance(block, sdk.TextBlock):
                        text_parts.append(block.text)
            elif isinstance(msg, sdk.ResultMessage):
                structured = getattr(msg, "structured_output", None)
                final_text = getattr(msg, "result", "") or ""
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


def _build_tool_server(lc_tools):  # implemented in Task 4
    raise NotImplementedError("tool bridging arrives in Task 4")
```

Move the `import json` line up into the stdlib import block at the top of the file (shown inline here only for diff clarity).

- [ ] **Step 4: Add the optional extra to `pyproject.toml`**

In `[project.optional-dependencies]`, next to the existing `bedrock` extra, add (check the current released version first with `pip index versions claude-agent-sdk` and use it as the floor):

```toml
claude-code = ["claude-agent-sdk>=0.1.0"]
```

- [ ] **Step 5: Run the full test file**

Run: `python -m pytest tests/test_claude_code_provider.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add tradingagents/llm_clients/claude_code_client.py tests/test_claude_code_provider.py pyproject.toml
git commit -m "feat(llm_clients): ChatClaudeCode text generation via Claude Agent SDK"
```

---

### Task 3: Structured output (`with_structured_output`)

**Files:**
- Modify: `tradingagents/llm_clients/claude_code_client.py`
- Modify: `tests/test_claude_code_provider.py`

**Interfaces:**
- Consumes: `ChatClaudeCode._generate` structured branch from Task 2 (`structured_schema` kwarg → `output_format`, JSON content).
- Produces: `ChatClaudeCode.with_structured_output(schema, *, include_raw=False, **kwargs) -> Runnable` returning parsed pydantic instances (or dicts for raw-dict schemas). Consumed by `agents/utils/structured.py:bind_structured` (calls `llm.with_structured_output(schema)` then `.invoke(prompt)`).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_claude_code_provider.py`:

```python
from pydantic import BaseModel


class _Verdict(BaseModel):
    action: str
    confidence: float


class TestStructuredOutput:
    def _llm(self, monkeypatch, structured):
        capture = {}
        fake = make_fake_sdk(
            final_text=json.dumps(structured) if structured else "",
            structured=structured,
            capture=capture,
        )
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import ChatClaudeCode

        return ChatClaudeCode(model="sonnet"), capture

    def test_returns_pydantic_instance(self, monkeypatch):
        llm, _ = self._llm(monkeypatch, {"action": "BUY", "confidence": 0.8})
        structured_llm = llm.with_structured_output(_Verdict)
        out = structured_llm.invoke([HumanMessage(content="decide")])
        assert isinstance(out, _Verdict)
        assert out.action == "BUY"

    def test_schema_forwarded_as_output_format(self, monkeypatch):
        llm, capture = self._llm(monkeypatch, {"action": "HOLD", "confidence": 0.5})
        llm.with_structured_output(_Verdict).invoke([HumanMessage(content="x")])
        fmt = capture["options"]["output_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"] == _Verdict.model_json_schema()

    def test_dict_schema_returns_dict(self, monkeypatch):
        llm, _ = self._llm(monkeypatch, {"k": "v"})
        raw_schema = {"type": "object", "properties": {"k": {"type": "string"}}}
        out = llm.with_structured_output(raw_schema).invoke([HumanMessage(content="x")])
        assert out == {"k": "v"}
```

Add `import json` to the test file's imports.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_claude_code_provider.py -v -k Structured`
Expected: FAIL (BaseChatModel's default `with_structured_output` will try `bind_tools` or error)

- [ ] **Step 3: Implement**

Add to `ChatClaudeCode` (and `from langchain_core.runnables import RunnableLambda` to imports):

```python
    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Route through the Agent SDK's json_schema output_format."""
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
```

- [ ] **Step 4: Run the full test file**

Run: `python -m pytest tests/test_claude_code_provider.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/claude_code_client.py tests/test_claude_code_provider.py
git commit -m "feat(llm_clients): structured output for claude_code via SDK json_schema"
```

---

### Task 4: Tool bridge — `bind_tools` → in-process MCP server

**Files:**
- Modify: `tradingagents/llm_clients/claude_code_client.py`
- Modify: `tests/test_claude_code_provider.py`

**Interfaces:**
- Consumes: `_build_options` tool branch from Task 2 (`langchain_tools` kwarg).
- Produces: `ChatClaudeCode.bind_tools(tools, **kwargs) -> Runnable` and module function `_build_tool_server(lc_tools) -> tuple[server, list[str]]` (replaces the Task 2 stub). Tool names exposed as `mcp__toolkit__{tool.name}`. The SDK runs the tool loop internally; the returned `AIMessage` has empty `tool_calls`, which `conditional_logic.should_continue_*` treats as "analysis done" — the LangGraph ToolNode cycle is intentionally bypassed.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_claude_code_provider.py`:

```python
from langchain_core.tools import tool as lc_tool_decorator


@lc_tool_decorator
def get_price(symbol: str) -> str:
    """Get latest price for a symbol."""
    return f"{symbol}: 100.0"


class TestToolBridge:
    def _bound(self, monkeypatch):
        capture = {}
        fake = make_fake_sdk(final_text="report text", capture=capture)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import ChatClaudeCode

        llm = ChatClaudeCode(model="sonnet")
        return llm.bind_tools([get_price]), capture

    def test_tools_registered_as_mcp_server(self, monkeypatch):
        bound, capture = self._bound(monkeypatch)
        out = bound.invoke([HumanMessage(content="price of AAPL?")])
        opts = capture["options"]
        assert opts["allowed_tools"] == ["mcp__toolkit__get_price"]
        assert "toolkit" in opts["mcp_servers"]
        assert opts["max_turns"] > 1
        assert out.content == "report text"
        assert out.tool_calls == []

    def test_tool_handler_invokes_langchain_tool(self, monkeypatch):
        capture = {}
        fake = make_fake_sdk(capture=capture)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import _build_tool_server

        server, names = _build_tool_server([get_price])
        assert names == ["mcp__toolkit__get_price"]
        handler = server.tools[0]
        result = asyncio.run(handler({"symbol": "MSFT"}))
        assert result["content"][0]["text"] == "MSFT: 100.0"

    def test_tool_handler_reports_errors(self, monkeypatch):
        capture = {}
        fake = make_fake_sdk(capture=capture)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import _build_tool_server

        @lc_tool_decorator
        def broken(x: str) -> str:
            """Always fails."""
            raise ValueError("nope")

        server, _ = _build_tool_server([broken])
        result = asyncio.run(server.tools[0]({"x": "y"}))
        assert result.get("is_error") is True
        assert "nope" in result["content"][0]["text"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_claude_code_provider.py -v -k ToolBridge`
Expected: FAIL with `NotImplementedError: tool bridging arrives in Task 4`

- [ ] **Step 3: Implement**

Replace the Task 2 `_build_tool_server` stub and add `bind_tools`:

```python
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
```

Add to `ChatClaudeCode`:

```python
    def bind_tools(self, tools, **kwargs):
        """Expose langchain tools to the SDK's internal agent loop.

        The SDK executes the whole tool loop in one call, so responses carry
        no tool_calls and LangGraph's ToolNode cycle never fires — analysts
        receive the final report directly.
        """
        return self.bind(langchain_tools=list(tools))
```

- [ ] **Step 4: Run the full test file**

Run: `python -m pytest tests/test_claude_code_provider.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/claude_code_client.py tests/test_claude_code_provider.py
git commit -m "feat(llm_clients): bridge langchain tools into Agent SDK MCP loop"
```

---

### Task 5: `ClaudeCodeClient` + provider registration

**Files:**
- Modify: `tradingagents/llm_clients/claude_code_client.py`
- Modify: `tradingagents/llm_clients/factory.py` (if-ladder, before the OpenAI-compatible fallback)
- Modify: `tradingagents/llm_clients/api_key_env.py` (`PROVIDER_API_KEY_ENV`)
- Modify: `tradingagents/llm_clients/validators.py` (`_ANY_MODEL_PROVIDERS`)
- Modify: `tradingagents/llm_clients/model_catalog.py` (`MODEL_OPTIONS`)
- Modify: `tests/test_claude_code_provider.py`

**Interfaces:**
- Consumes: `ChatClaudeCode` from Tasks 2–4; `BaseLLMClient` (`base_client.py:28`); `validate_model(provider, model)` (`validators.py:20`).
- Produces: `ClaudeCodeClient(BaseLLMClient)` with `provider = "claude_code"`, `get_llm() -> ChatClaudeCode`, `validate_model() -> bool`. `create_llm_client("claude_code", ...)` returns it. `get_api_key_env("claude_code")` returns `None` (keyless — same as bedrock/ollama, so `cli/utils.ensure_api_key` skips the key prompt automatically).

- [ ] **Step 1: Write failing tests**

Append to `tests/test_claude_code_provider.py`:

```python
class TestClaudeCodeClientRegistration:
    def test_factory_returns_claude_code_client(self):
        from tradingagents.llm_clients.factory import create_llm_client

        client = create_llm_client("claude_code", "sonnet")
        assert type(client).__name__ == "ClaudeCodeClient"
        assert client.get_provider_name() == "claude_code"

    def test_api_key_env_is_keyless(self):
        from tradingagents.llm_clients.api_key_env import get_api_key_env

        assert get_api_key_env("claude_code") is None

    def test_any_model_id_validates(self):
        from tradingagents.llm_clients.validators import validate_model

        assert validate_model("claude_code", "sonnet") is True
        assert validate_model("claude_code", "claude-opus-4-8") is True
        assert validate_model("claude_code", "anything-at-all") is True

    def test_model_catalog_has_entries(self):
        from tradingagents.llm_clients.model_catalog import get_model_options

        quick = get_model_options("claude_code", "quick")
        deep = get_model_options("claude_code", "deep")
        assert any(model_id == "sonnet" for _, model_id in quick)
        assert any(model_id == "opus" for _, model_id in deep)

    def test_get_llm_builds_chat_claude_code(self, monkeypatch):
        fake = make_fake_sdk()
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.factory import create_llm_client

        client = create_llm_client(
            "claude_code", "opus", effort="high", temperature=0.2, max_retries=3
        )
        llm = client.get_llm()
        assert type(llm).__name__ == "ChatClaudeCode"
        assert llm.model == "opus"
        assert llm.effort == "high"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_claude_code_provider.py -v -k Registration`
Expected: FAIL with `ValueError: Unsupported LLM provider: claude_code`

- [ ] **Step 3: Implement registration**

`claude_code_client.py` — append:

```python
from .base_client import BaseLLMClient  # add to top-of-file imports


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
```

`factory.py` — add before the OpenAI-compatible fallback:

```python
    if provider_lower == "claude_code":
        from .claude_code_client import ClaudeCodeClient

        return ClaudeCodeClient(model, base_url, **kwargs)
```

`api_key_env.py` — add to `PROVIDER_API_KEY_ENV` (near bedrock, with comment):

```python
    # Claude Code subscription auth (Agent SDK): `claude /login` OAuth or
    # CLAUDE_CODE_OAUTH_TOKEN — no API key env var.
    "claude_code": None,
```

`validators.py` — add `"claude_code"` to `_ANY_MODEL_PROVIDERS` (SDK accepts aliases and any full model ID).

`model_catalog.py` — add to `MODEL_OPTIONS`:

```python
    "claude_code": {
        "quick": [
            ("Claude Haiku 4.5 — fastest", "haiku"),
            ("Claude Sonnet (latest) — balanced", "sonnet"),
        ],
        "deep": [
            ("Claude Sonnet (latest) — balanced", "sonnet"),
            ("Claude Opus (latest) — most capable", "opus"),
        ],
    },
```

- [ ] **Step 4: Run full suite (registration can break other provider tests)**

Run: `python -m pytest tests/ -v -x -q`
Expected: all PASS (watch `test_provider_registry.py`, `test_api_key_env.py`, `test_model_validation.py` — they iterate over the registries and may assert counts/membership; update those tables if they enumerate providers exhaustively)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/ tests/test_claude_code_provider.py
git commit -m "feat(llm_clients): register claude_code provider (factory, keys, catalog)"
```

---

### Task 6: Graph kwargs + CLI wiring

**Files:**
- Modify: `tradingagents/graph/trading_graph.py:153-186` (`_get_provider_kwargs`)
- Modify: `cli/utils.py:338-366` (`_llm_provider_table`)
- Modify: `tests/test_claude_code_provider.py`

**Interfaces:**
- Consumes: config keys `anthropic_effort` (`default_config.py:93` — reused for claude_code; no new config key), `temperature`, `llm_max_retries`.
- Produces: `_get_provider_kwargs()` yields `{"effort": ...}` for provider `"claude_code"` when `anthropic_effort` is set; CLI provider menu row `("Claude Code (Pro/Max subscription — no API key)", "claude_code", None)`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_claude_code_provider.py`:

```python
class TestGraphAndCliWiring:
    def test_provider_kwargs_include_effort(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = dict(DEFAULT_CONFIG)
        config["llm_provider"] = "claude_code"
        config["anthropic_effort"] = "high"
        kwargs = TradingAgentsGraph._get_provider_kwargs.__wrapped__(
            types.SimpleNamespace(config=config)
        ) if hasattr(TradingAgentsGraph._get_provider_kwargs, "__wrapped__") else (
            TradingAgentsGraph._get_provider_kwargs(types.SimpleNamespace(config=config))
        )
        assert kwargs.get("effort") == "high"

    def test_cli_provider_table_row(self):
        from cli.utils import _llm_provider_table

        keys = [row[1] for row in _llm_provider_table()]
        assert "claude_code" in keys
        row = next(r for r in _llm_provider_table() if r[1] == "claude_code")
        assert row[2] is None  # no backend URL
```

Note for implementer: `_get_provider_kwargs` is an instance method — if the `SimpleNamespace` trick doesn't match its actual signature, construct the minimal object the method needs (it only reads `self.config`) or refactor the test to match existing tests in `tests/test_temperature_config.py`, which already exercise `_get_provider_kwargs`; follow that file's established pattern instead of inventing one.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_claude_code_provider.py -v -k Wiring`
Expected: FAIL (no effort kwarg; no menu row)

- [ ] **Step 3: Implement**

`trading_graph.py` `_get_provider_kwargs` — add branch alongside the existing `anthropic` branch:

```python
        elif provider == "claude_code":
            if self.config.get("anthropic_effort") is not None:
                kwargs["effort"] = self.config["anthropic_effort"]
```

(Keep the cross-provider `temperature`/`max_retries` behavior untouched — `ClaudeCodeClient` accepts and ignores them.)

`cli/utils.py` `_llm_provider_table` — add row (near anthropic):

```python
        ("Claude Code (Pro/Max subscription — no API key)", "claude_code", None),
```

Check `cli/main.py:694-725` (provider-specific reasoning-knob prompts): mirror the `anthropic` effort prompt for `claude_code` by extending the existing condition to cover both providers (exact edit depends on current shape — extend `elif provider_lower == "anthropic":` to `elif provider_lower in ("anthropic", "claude_code"):`).

- [ ] **Step 4: Run full suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS (CLI tests in `test_cli_config_precedence.py` / `test_cli_env_skip.py` may enumerate the provider table — update expectations if they do)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/graph/trading_graph.py cli/ tests/test_claude_code_provider.py
git commit -m "feat(cli): expose claude_code provider in menu and graph kwargs"
```

---

### Task 7: Docs + integration smoke test

**Files:**
- Modify: `README.md` (provider setup section, ~lines 132-156 and provider list ~line 200)
- Modify: `CHANGELOG.md`
- Create: `tests/test_claude_code_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing docs; a skip-guarded end-to-end test.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_claude_code_smoke.py
"""End-to-end smoke test for the claude_code provider.

Requires: `pip install 'tradingagents[claude-code]'`, the `claude` CLI on
PATH, and a logged-in Claude Pro/Max subscription (`claude /login`) or
CLAUDE_CODE_OAUTH_TOKEN. Skipped otherwise.
"""
import shutil

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed"),
]


def test_one_shot_text_roundtrip():
    from langchain_core.messages import HumanMessage

    from tradingagents.llm_clients.factory import create_llm_client

    llm = create_llm_client("claude_code", "haiku").get_llm()
    out = llm.invoke([HumanMessage(content="Reply with exactly the word PONG.")])
    assert "PONG" in out.content.upper()


def test_structured_roundtrip():
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel

    from tradingagents.llm_clients.factory import create_llm_client

    class Answer(BaseModel):
        value: int

    llm = create_llm_client("claude_code", "haiku").get_llm()
    out = llm.with_structured_output(Answer).invoke(
        [HumanMessage(content="What is 2+2? Answer as JSON with key 'value'.")]
    )
    assert out.value == 4
```

- [ ] **Step 2: Run it (only if environment has CLI + login; otherwise confirm it skips)**

Run: `python -m pytest tests/test_claude_code_smoke.py -v`
Expected: PASS (with login) or SKIPPED (without) — never FAIL

- [ ] **Step 3: Write docs**

`README.md`: add a "Claude Code (subscription)" subsection to the provider setup block:

```markdown
### Claude Code (Pro/Max subscription — no API key)

Run TradingAgents on your Claude subscription instead of API billing:

1. `pip install "tradingagents[claude-code]"` (installs the Claude Agent SDK)
2. Install the Claude Code CLI and log in once: `claude /login`
   (or mint a long-lived token with `claude setup-token` and export it as
   `CLAUDE_CODE_OAUTH_TOKEN` — useful for servers/CI)
3. Pick **Claude Code** in the CLI provider menu, or set
   `TRADINGAGENTS_LLM_PROVIDER=claude_code`. Model IDs accept aliases
   (`haiku`, `sonnet`, `opus`) or full IDs (`claude-opus-4-8`).

Notes: usage counts against your subscription limits. `temperature` and
`max_tokens` are not configurable through the Agent SDK and are ignored for
this provider. Tool-using analysts run their tool loop inside the Agent SDK
rather than through LangGraph's ToolNode.
```

Add `claude_code` to the provider list, and a CHANGELOG entry under Unreleased:

```markdown
- feat: `claude_code` provider — run all agents on a Claude Pro/Max
  subscription via the Claude Agent SDK, no `ANTHROPIC_API_KEY` needed.
```

- [ ] **Step 4: Full suite + lint**

Run: `python -m pytest tests/ -q && ruff check .`
Expected: all PASS, no lint errors

- [ ] **Step 5: Commit**

```bash
git add README.md CHANGELOG.md tests/test_claude_code_smoke.py
git commit -m "docs: claude_code subscription provider setup + smoke test"
```

---

## Known risks for implementers

1. **`ClaudeAgentOptions` field drift.** `effort`, `output_format`, `setting_sources` were verified against docs (2026), but the installed SDK version may differ. If a field is rejected at construction, check `ClaudeAgentOptions.__init__` in the installed package and adapt (e.g. route unknown knobs through `extra_args` if present, or drop `effort`). The unit tests use a fake SDK, so only the smoke test catches this — run it if the environment has a login.
2. **`@tool` schema argument form.** The SDK accepts either a `{name: type}` dict or a full JSON schema. `_tool_schema()` emits full JSON schema; if the installed SDK version rejects it, convert to the simple-dict form.
3. **Existing registry tests.** `test_api_key_env.py`, `test_model_validation.py`, `test_provider_registry.py`, and CLI tests may enumerate providers exhaustively; adding `claude_code` can break equality assertions. Fix by extending their expected tables — do not weaken assertions.
4. **`_get_provider_kwargs` test shape.** Follow `tests/test_temperature_config.py`'s existing pattern for exercising `_get_provider_kwargs` rather than the SimpleNamespace sketch if they differ.
5. **Latency.** Each SDK call spawns a CLI subprocess (~1–2s overhead). Acceptable for this workload; do not add premature pooling/caching.
