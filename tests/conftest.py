"""
Test helpers: a scriptable fake model client, so the harness can be exercised
end to end without Ollama or an API key.
"""

import itertools
from collections.abc import Callable

import pytest

from harness.client import BaseClient, Response, ToolCall

_ids = itertools.count(1)


def text(t: str, tokens: int = 10) -> Response:
    return Response(text=t, tool_calls=[], input_tokens=tokens, output_tokens=tokens)


def call(name: str, tokens: int = 10, **inputs) -> Response:
    return Response(text="", tool_calls=[ToolCall(id=f"t{next(_ids)}", name=name, input=inputs)],
                    input_tokens=tokens, output_tokens=tokens)


def calls(*pairs: tuple[str, dict], tokens: int = 10) -> Response:
    return Response(text="", tool_calls=[ToolCall(id=f"t{next(_ids)}", name=n, input=i) for n, i in pairs],
                    input_tokens=tokens, output_tokens=tokens)


Handler = Callable[[list[dict], str, list[dict] | None], Response]


class FakeClient(BaseClient):
    """
    Every send() goes to `handler(messages, system, tools)`. Handlers usually
    look at the system prompt (planner, verifier, role) and at how many tool
    results are already in the history to decide what to do next.
    """

    def __init__(self, handler: Handler, model: str = "fake"):
        self.handler = handler
        self.model = model
        self.calls: list[dict] = []

    async def send(self, messages, system="", tools=None, max_tokens=0, on_text_chunk=None):
        self.calls.append({"messages": [dict(m) for m in messages], "system": system, "tools": tools})
        return self.handler(messages, system, tools)


def tool_results(messages: list[dict]) -> list[str]:
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            out += [b["content"] for b in m["content"] if b.get("type") == "tool_result"]
    return out


@pytest.fixture
def memory_path(tmp_path):
    return tmp_path / "memory.json"
