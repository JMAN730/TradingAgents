"""Unit tests for the claude_code provider (Claude Agent SDK adapter)."""
import asyncio
import sys

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
