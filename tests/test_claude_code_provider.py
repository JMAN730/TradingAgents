"""Unit tests for the claude_code provider (Claude Agent SDK adapter)."""
import asyncio
import json
import sys
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool_decorator
from pydantic import BaseModel

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


def make_fake_sdk(final_text="FAKE RESULT", structured=None, capture=None, subtype="success"):
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
        yield ResultMessage(final_text, structured_output=structured, subtype=subtype)

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
        assert "Write" in opts["disallowed_tools"]
        assert "WebFetch" in opts["disallowed_tools"]

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

    def test_error_subtype_raises_instead_of_partial_report(self, monkeypatch):
        capture = {}
        fake = make_fake_sdk(final_text="", subtype="error_max_turns", capture=capture)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        from tradingagents.llm_clients.claude_code_client import ChatClaudeCode

        with pytest.raises(RuntimeError, match="error_max_turns"):
            ChatClaudeCode(model="sonnet").invoke([HumanMessage(content="x")])


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

    def test_include_raw_rejected(self, monkeypatch):
        llm, _ = self._llm(monkeypatch, {"action": "BUY", "confidence": 0.8})
        with pytest.raises(NotImplementedError, match="include_raw"):
            llm.with_structured_output(_Verdict, include_raw=True)


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


class TestToolSchemaFallback:
    def test_object_without_schema_attrs_falls_back_to_open_object(self):
        from tradingagents.llm_clients.claude_code_client import _tool_schema

        lc_tool = types.SimpleNamespace(name="x", description="d")
        assert _tool_schema(lc_tool) == {"type": "object", "additionalProperties": True}


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


class TestGraphAndCliWiring:
    def test_provider_kwargs_include_effort(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # Follows the established pattern in tests/test_temperature_config.py
        # (TestProviderKwargsTemperature._kwargs_for): construct a bare instance
        # via __new__ and set only the `config` attribute the method reads,
        # rather than the brief's speculative SimpleNamespace/__wrapped__ sketch.
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = {"llm_provider": "claude_code", "anthropic_effort": "high"}
        kwargs = TradingAgentsGraph._get_provider_kwargs(graph)
        assert kwargs.get("effort") == "high"

    def test_cli_provider_table_row(self):
        from cli.utils import _llm_provider_table

        keys = [row[1] for row in _llm_provider_table()]
        assert "claude_code" in keys
        row = next(r for r in _llm_provider_table() if r[1] == "claude_code")
        assert row[2] is None  # no backend URL
