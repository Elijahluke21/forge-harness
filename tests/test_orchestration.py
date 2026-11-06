"""Spawning, caching, plans, the executor, checks, and the autopilot end to end."""

import asyncio
import json

import pytest

from conftest import FakeClient, call, text, tool_results
from harness.agent import spawn_subagent
from harness.app import build_forge
from harness.autopilot import Autopilot, AutopilotConfig
from harness.budget import BudgetExceeded
from harness.checks import Check, load_checks, run_checks
from harness.config import ForgeConfig
from harness.executor import PlanError, Step, execute_plan, parse_plan
from harness.planning import PLAN_SYSTEM


def forge_with(handler, tmp_path, **cfg_overrides):
    cfg = ForgeConfig(memory_path=str(tmp_path / "mem.json"), **cfg_overrides)
    return build_forge(cfg, client=FakeClient(handler), workspace=str(tmp_path / "ws"))


@pytest.fixture(autouse=True)
def _ws(tmp_path):
    (tmp_path / "ws").mkdir()


# ── spawning + cache ─────────────────────────────────────────────────────

async def test_unknown_role_is_reported(tmp_path):
    f = forge_with(lambda *a: text("x"), tmp_path)
    r = await spawn_subagent(f.orchestrator, f.worker_registry, "t", role="wizard")
    assert r.status == "error" and "unknown role" in r.answer


async def test_reader_results_cached_until_workspace_changes(tmp_path):
    runs = []

    def handler(messages, system, tools):
        runs.append(system)
        return text("contents")
    f = forge_with(handler, tmp_path)
    agent, workers = f.orchestrator, f.worker_registry

    await spawn_subagent(agent, workers, "read a.txt", role="reader")
    r2 = await spawn_subagent(agent, workers, "read a.txt", role="reader")
    assert r2.cached and len(runs) == 1

    agent._tracking_callback("write_file", {}, "wrote")      # something changed
    r3 = await spawn_subagent(agent, workers, "read a.txt", role="reader")
    assert not r3.cached and len(runs) == 2


async def test_executor_results_never_cached(tmp_path):
    runs = []

    def handler(messages, system, tools):
        runs.append(1)
        return text("tests pass")
    f = forge_with(handler, tmp_path)
    for _ in range(2):
        await spawn_subagent(f.orchestrator, f.worker_registry, "run tests", role="executor")
    assert len(runs) == 2


async def test_incomplete_is_not_retried_but_error_is(tmp_path):
    seen = []

    def handler(messages, system, tools):
        seen.append(system)
        return text("[error] broke") if len(seen) == 1 else text("fixed it")
    f = forge_with(handler, tmp_path)
    r = await spawn_subagent(f.orchestrator, f.worker_registry, "do x", role="writer")
    assert r.status == "ok" and r.retried and "previous attempt" in seen[1]

    f2 = forge_with(lambda *a: call("read_file", path="x"), tmp_path, max_iterations=2)
    r2 = await spawn_subagent(f2.orchestrator, f2.worker_registry, "do y", role="reader")
    assert r2.status == "incomplete" and not r2.retried


async def test_verify_failure_sets_status(tmp_path):
    def handler(messages, system, tools):
        if "verification sub-agent" in system:
            return text("FAILED: file is empty")
        return text("wrote it")
    f = forge_with(handler, tmp_path)
    r = await spawn_subagent(f.orchestrator, f.worker_registry, "write x", role="writer", verify=True)
    assert r.status == "verify_failed" and "empty" in r.verification


async def test_spawn_budget_stops_the_tree(tmp_path):
    f = forge_with(lambda *a: text("ok"), tmp_path, budget_max_spawns=2)
    for _ in range(2):
        await spawn_subagent(f.orchestrator, f.worker_registry, "t", role="writer")
    with pytest.raises(BudgetExceeded):
        await spawn_subagent(f.orchestrator, f.worker_registry, "t", role="writer")


async def test_role_scoped_children_keep_permissions(tmp_path):
    asked = []

    async def deny(name, inputs):
        asked.append(name)
        return False

    state = {"n": 0}

    def handler(messages, system, tools):
        state["n"] += 1
        return call("write_file", path="a", content="b") if state["n"] == 1 else text("done")
    cfg = ForgeConfig(memory_path=str(tmp_path / "m.json"), require_confirm=["write_file"])
    f = build_forge(cfg, client=FakeClient(handler), workspace=str(tmp_path / "ws"), confirm_fn=deny)
    await spawn_subagent(f.orchestrator, f.worker_registry, "write", role="writer")
    assert asked == ["write_file"]


# ── plans ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plan,err", [
    ([], "non-empty"),
    ([{"id": "1", "task": "a", "role": "wizard"}], "unknown role"),
    ([{"id": "1", "task": "a"}, {"id": "1", "task": "b"}], "duplicate"),
    ([{"id": "1", "task": "a", "depends_on": ["9"]}], "unknown step"),
    ([{"id": "1", "task": "a", "depends_on": ["2"]}, {"id": "2", "task": "b", "depends_on": ["1"]}], "cycle"),
    ([{"id": "1", "task": ""}], "no task"),
])
def test_parse_plan_rejects_bad_plans(plan, err):
    with pytest.raises(PlanError, match=err):
        parse_plan(plan)


async def test_executor_respects_dependencies_and_parallelism(tmp_path):
    order, active, peak = [], [0], [0]

    async def spawn(task, context, role, verify):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.05)
        active[0] -= 1
        order.append(task)
        return "ok", f"did {task}"

    steps = parse_plan([
        {"id": "a", "task": "A"}, {"id": "b", "task": "B"},
        {"id": "c", "task": "C", "depends_on": ["a", "b"]},
    ])
    outs = await execute_plan(steps, spawn, tmp_path, max_parallel=3)
    assert order[-1] == "C" and peak[0] == 2
    assert all(o.status == "ok" for o in outs)
    assert (tmp_path / "step-c.md").exists()


async def test_executor_passes_dependency_summaries_not_raw_output(tmp_path):
    seen = {}

    async def spawn(task, context, role, verify):
        seen[task] = context
        return "ok", "x" * 5000 if task == "A" else "fine"

    steps = parse_plan([{"id": "a", "task": "A"}, {"id": "b", "task": "B", "depends_on": ["a"]}])
    await execute_plan(steps, spawn, tmp_path)
    assert "step-a.md" in seen["B"] and len(seen["B"]) < 1500


async def test_executor_skips_downstream_of_failure(tmp_path):
    async def spawn(task, context, role, verify):
        return ("error", "broke") if task == "A" else ("ok", "fine")

    steps = parse_plan([
        {"id": "a", "task": "A"}, {"id": "b", "task": "B", "depends_on": ["a"]},
        {"id": "c", "task": "C", "depends_on": ["b"]}, {"id": "d", "task": "D"},
    ])
    outs = {o.step.id: o.status for o in await execute_plan(steps, spawn, tmp_path)}
    assert outs == {"a": "error", "b": "skipped", "c": "skipped", "d": "ok"}


async def test_executor_abort_cancels_running_steps(tmp_path):
    finished = []

    async def spawn(task, context, role, verify):
        if task == "A":
            raise BudgetExceeded("tokens", 2, 1)
        await asyncio.sleep(2)
        finished.append(task)
        return "ok", ""

    steps = parse_plan([{"id": "a", "task": "A"}, {"id": "b", "task": "B"}])
    with pytest.raises(BudgetExceeded):
        await execute_plan(steps, spawn, tmp_path)
    assert finished == []


# ── checks ────────────────────────────────────────────────────────────────

async def test_checks_load_and_run(tmp_path):
    f = tmp_path / "checks.yaml"
    f.write_text("checks:\n  - id: yes\n    cmd: 'true'\n  - id: no\n    cmd: 'echo boom; exit 3'\n    description: must not fail\n  - cmd: 'sleep 5'\n    timeout: 1\n")
    checks = load_checks(f)
    report = await run_checks(checks, tmp_path)
    assert report.score() == "1/3" and not report.all_passed
    summary = report.failure_summary()
    assert "boom" in summary and "exit 3" in summary and "timed out" in summary and "must not fail" in summary


def test_checks_file_validation(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("checks: []\n")
    with pytest.raises(ValueError):
        load_checks(f)


# ── autopilot end to end ──────────────────────────────────────────────────

PLAN_ONE_WRITER = json.dumps([{"id": "1", "task": "create done.txt containing ok", "role": "writer",
                               "depends_on": [], "hint": ""}])


def builder_handler(messages, system, tools):
    """Planner returns a one-step plan; the writer writes done.txt then reports."""
    if system == PLAN_SYSTEM:
        return text(f"<think>simple</think>{PLAN_ONE_WRITER}")
    if not tool_results(messages):
        return call("write_file", path="done.txt", content="ok")
    return text("created done.txt")


async def test_autopilot_finishes_when_checks_pass(tmp_path):
    f = forge_with(builder_handler, tmp_path)
    pilot = Autopilot(f.orchestrator, f.worker_registry,
                      [Check("done", "test -f done.txt")], f.workspace.cwd)
    result = await pilot.run("make done.txt")
    assert result.status == "done" and result.rounds == 1
    assert (result.run_dir / "result.md").exists()
    assert (result.run_dir / "round-1" / "plan.json").exists()
    events = [json.loads(l)["event"] for l in (result.run_dir / "events.jsonl").read_text().splitlines()]
    assert events[0] == "run_start" and events[-1] == "run_end"


async def test_autopilot_does_nothing_when_already_done(tmp_path):
    (tmp_path / "ws" / "done.txt").write_text("ok")
    f = forge_with(lambda *a: pytest.fail("model should not be called"), tmp_path)
    result = await Autopilot(f.orchestrator, f.worker_registry,
                             [Check("done", "test -f done.txt")], f.workspace.cwd).run("x")
    assert result.status == "done" and result.rounds == 0


def lazy_handler(messages, system, tools):
    """Plans every round but the work never makes the check pass."""
    if system == PLAN_SYSTEM:
        return text(PLAN_ONE_WRITER)
    return text("I think it's done")


async def test_autopilot_stops_when_stalled(tmp_path):
    f = forge_with(lazy_handler, tmp_path)
    result = await Autopilot(f.orchestrator, f.worker_registry, [Check("never", "false")],
                             f.workspace.cwd, AutopilotConfig(max_rounds=20, stall_rounds=2)).run("x")
    assert result.status == "stalled"
    assert result.rounds == 2           # two rounds of work, neither moved a check
    assert (result.run_dir / "blocked.md").exists()


async def test_autopilot_escalates_once_before_giving_up(tmp_path):
    strong_calls = []

    def strong(messages, system, tools):
        strong_calls.append(1)
        return text(PLAN_ONE_WRITER)
    f = forge_with(lazy_handler, tmp_path)
    result = await Autopilot(f.orchestrator, f.worker_registry, [Check("never", "false")],
                             f.workspace.cwd, AutopilotConfig(max_rounds=20, stall_rounds=2),
                             escalation_client=FakeClient(strong)).run("x")
    assert result.status == "stalled" and strong_calls


async def test_autopilot_respects_budget(tmp_path):
    f = forge_with(lazy_handler, tmp_path, budget_max_spawns=2)
    result = await Autopilot(f.orchestrator, f.worker_registry, [Check("never", "false")],
                             f.workspace.cwd, AutopilotConfig(max_rounds=20, stall_rounds=50)).run("x")
    assert result.status == "budget" and "spawns" in result.reason


async def test_autopilot_ignores_edits_to_checks_file(tmp_path):
    """An agent rewriting checks.yaml mid-run must not change what 'done' means."""
    ws = tmp_path / "ws"
    (ws / "checks.yaml").write_text("checks:\n  - id: real\n    cmd: 'test -f done.txt'\n")

    def cheater(messages, system, tools):
        if system == PLAN_SYSTEM:
            return text(PLAN_ONE_WRITER)
        if not tool_results(messages):
            return call("write_file", path="checks.yaml", content="checks:\n  - cmd: 'true'\n")
        return text("all checks pass now")
    f = forge_with(cheater, tmp_path)
    checks = load_checks(ws / "checks.yaml")
    result = await Autopilot(f.orchestrator, f.worker_registry, checks, ws,
                             AutopilotConfig(max_rounds=3, stall_rounds=2)).run("x")
    assert result.status != "done"


async def test_autopilot_bad_plan_counts_as_no_progress(tmp_path):
    def handler(messages, system, tools):
        return text("not a plan at all")
    f = forge_with(handler, tmp_path)
    result = await Autopilot(f.orchestrator, f.worker_registry, [Check("never", "false")],
                             f.workspace.cwd, AutopilotConfig(max_rounds=10, stall_rounds=2)).run("x")
    assert result.status == "stalled"
    assert "planning failed" in "\n".join(result.history)


async def test_on_finish_notification(tmp_path):
    out = tmp_path / "notified.txt"
    f = forge_with(builder_handler, tmp_path)
    cfg = AutopilotConfig(on_finish=f'echo "$FORGE_STATUS|$FORGE_SUMMARY" > {out}')
    await Autopilot(f.orchestrator, f.worker_registry, [Check("done", "test -f done.txt")],
                    f.workspace.cwd, cfg).run("x")
    assert out.read_text().startswith("done|done: all 1 checks pass")
