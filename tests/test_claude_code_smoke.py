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
