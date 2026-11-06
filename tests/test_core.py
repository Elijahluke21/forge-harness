"""Loop, registry, client parsing, text helpers, budgets."""

import os
from types import SimpleNamespace

import pytest

from conftest import FakeClient, call, calls, text, tool_results
from harness.budget import BudgetExceeded, RunBudget
from harness.client import AnthropicClient, OllamaClient, _make_tool_call
from harness.loop import INCOMPLETE_PREFIX, run
from harness.session import SessionStats
from harness.text import extract_json, strip_think
from harness.tools.registry import ToolRegistry


# ── text ──────────────────────────────────────────────────────────────────

def test_strip_think_removes_blocks_and_unterminated_tail():
    assert strip_think("<think>hmm</think>answer") == "answer"
    assert strip_think("a<think>x</think>b<think>y</think>c") == "abc"
    assert strip_think("ok<think>cut off mid") == "ok"


@pytest.mark.parametrize("raw", [
    '[{"a": 1}]',
    '```json\n[{"a": 1}]\n```',
    '<think>plan it</think>\n[{"a": 1}]',
    'Here is the plan:\n[{"a": 1}]\nHope that helps!',
])
def test_extract_json_tolerates_model_noise(raw):
    assert extract_json(raw) == [{"a": 1}]


def test_extract_json_raises_when_nothing_parses():
    with pytest.raises(ValueError):
        extract_json("no json here")


# ── client parsing ────────────────────────────────────────────────────────

def test_bad_tool_json_is_flagged_not_crashed():
    tc = _make_tool_call("", "bash", '{"command": "ls"')     # truncated JSON
    assert tc.error and tc.input == {} and tc.id.startswith("call_")
    assert _make_tool_call("x", "bash", "[1,2]").error        # not an object
    assert _make_tool_call("x", "bash", '{"command": "ls"}').input == {"command": "ls"}


def test_ollama_parse_completion_survives_bad_arguments():
    client = OllamaClient.__new__(OllamaClient)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[
            SimpleNamespace(id="c1", function=SimpleNamespace(name="bash", arguments="{oops")),
        ]))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
    )
    resp = client._parse_completion(completion)
    assert resp.tool_calls[0].error


async def test_anthropic_keeps_all_text_blocks():
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = "x"

    async def create(**kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="first "), SimpleNamespace(type="text", text="second")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
    client._api = SimpleNamespace(messages=SimpleNamespace(create=create))
    resp = await client.send(messages=[{"role": "user", "content": "hi"}])
    assert resp.text == "first second"


# ── loop ──────────────────────────────────────────────────────────────────

def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(description="echo", input_schema={"type": "object"})
    async def echo(msg: str = "") -> str:
        return f"echo:{msg}"
    return reg


async def test_loop_reports_malformed_tool_call_to_model():
    from harness.client import Response, ToolCall
    bad = Response("", [ToolCall("b1", "echo", {}, error="arguments were not valid JSON")], 1, 1)
    seq = iter([bad, text("done")])
    msgs = [{"role": "user", "content": "go"}]
    out = await run(msgs, FakeClient(lambda *a: next(seq)), _echo_registry())
    assert out == "done"
    assert tool_results(msgs)[0].startswith("[error] could not run 'echo'")


async def test_loop_strips_think_from_history_and_answer():
    seq = iter([text("<think>long reasoning</think>final")])
    msgs = [{"role": "user", "content": "go"}]
    out = await run(msgs, FakeClient(lambda *a: next(seq)), _echo_registry())
    assert out == "final"
    assert msgs[-1]["content"] == "final"


async def test_loop_iteration_limit_is_incomplete_not_error():
    client = FakeClient(lambda *a: call("echo", msg="again"))
    out = await run([{"role": "user", "content": "go"}], client, _echo_registry(), max_iterations=3)
    assert out.startswith(INCOMPLETE_PREFIX)


async def test_loop_enforces_token_budget():
    stats = SessionStats(budget=RunBudget(max_total_tokens=50))
    client = FakeClient(lambda *a: call("echo", tokens=20, msg="x"))
    with pytest.raises(BudgetExceeded) as exc:
        await run([{"role": "user", "content": "go"}], client, _echo_registry(),
                  session_stats=stats, max_iterations=50)
    assert exc.value.limit == "tokens"


async def test_budget_abort_inside_a_tool_unwinds_and_cancels_siblings():
    import asyncio
    reg = ToolRegistry()
    finished = []

    @reg.register(description="", input_schema={"type": "object"})
    async def boom() -> str:
        raise BudgetExceeded("spawns", 2, 1)

    @reg.register(description="", input_schema={"type": "object"})
    async def slow() -> str:
        await asyncio.sleep(5)
        finished.append(True)
        return "slow"

    client = FakeClient(lambda *a: calls(("boom", {}), ("slow", {})))
    with pytest.raises(BudgetExceeded):
        await run([{"role": "user", "content": "go"}], client, reg)
    assert not finished


# ── registry ──────────────────────────────────────────────────────────────

async def test_subset_keeps_permissions():
    reg = _echo_registry()
    asked = []

    async def deny(name, inputs):
        asked.append(name)
        return False
    reg.set_permissions(["echo"], deny)
    scoped = reg.subset(["echo"])
    assert (await scoped.execute("echo", {"msg": "x"})).startswith("[cancelled]")
    assert asked == ["echo"]


async def test_hooks_keep_path_and_report_status(tmp_path):
    out = tmp_path / "hook.txt"
    reg = _echo_registry()
    reg.set_hooks([{"on": "post", "tool": "*", "cmd": f'echo "$FORGE_TOOL $FORGE_RESULT_STATUS" > {out}; command -v sh >> {out}'}])
    await reg.execute("echo", {"msg": "x"})
    content = out.read_text()
    assert content.startswith("echo ok")
    assert "/sh" in content          # PATH survived, so commands resolve


async def test_non_dict_arguments_rejected():
    assert (await _echo_registry().execute("echo", ["x"])).startswith("[error]")
