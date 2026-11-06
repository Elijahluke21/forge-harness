"""
Plan executor (7.3).

`plan_task` produces a list of steps with dependencies. Before this module the
orchestrator *model* had to read that plan and decide which spawn_agent calls
to make, in what order, and when it was finished — exactly the judgment small
models get wrong. Here, code walks the dependency graph:

  - validate the plan (unique ids, known roles, existing deps, no cycles)
  - spawn every step whose dependencies are done, in parallel
  - pass each step the *summaries and file paths* of the steps it depends on,
    never their raw output
  - when a step fails, skip everything downstream of it
  - stop when no step can make progress

Every step's full answer is written to the run directory, so nothing has to
stay in a model's context between steps.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from harness.roles import ROLES

SUMMARY_CHARS = 600   # how much of a dependency's answer a dependent sees inline


class PlanError(ValueError):
    pass


@dataclass
class Step:
    id: str
    task: str
    role: str = "executor"
    depends_on: list[str] = field(default_factory=list)
    hint: str = ""
    verify: bool = False


@dataclass
class StepOutcome:
    step: Step
    status: str            # ok | error | incomplete | verify_failed | skipped
    answer: str = ""
    output_path: str = ""


def parse_plan(raw) -> list[Step]:
    """Turn model-produced plan JSON into validated Steps. Raises PlanError."""
    if isinstance(raw, dict) and isinstance(raw.get("steps"), list):
        raw = raw["steps"]
    if not isinstance(raw, list) or not raw:
        raise PlanError("plan must be a non-empty JSON array of steps")

    steps: list[Step] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise PlanError(f"step {i} is not an object")
        sid = str(item.get("id", i)).strip()
        task = str(item.get("task", "")).strip()
        if not task:
            raise PlanError(f"step {sid} has no task")
        if sid in seen:
            raise PlanError(f"duplicate step id {sid!r}")
        seen.add(sid)
        role = str(item.get("role", "executor")).strip().lower() or "executor"
        if role not in ROLES:
            raise PlanError(f"step {sid} has unknown role {role!r} (valid: {', '.join(ROLES)})")
        deps = item.get("depends_on") or []
        if not isinstance(deps, list):
            deps = [deps]
        steps.append(Step(
            id=sid, task=task, role=role,
            depends_on=[str(d).strip() for d in deps],
            hint=str(item.get("hint", "")).strip(),
            verify=bool(item.get("verify", False)),
        ))

    ids = {s.id for s in steps}
    for s in steps:
        missing = [d for d in s.depends_on if d not in ids]
        if missing:
            raise PlanError(f"step {s.id} depends on unknown step(s) {missing}")
        if s.id in s.depends_on:
            raise PlanError(f"step {s.id} depends on itself")
    _check_acyclic(steps)
    return steps


def _check_acyclic(steps: list[Step]) -> None:
    deps = {s.id: set(s.depends_on) for s in steps}
    done: set[str] = set()
    while len(done) < len(deps):
        ready = [sid for sid, d in deps.items() if sid not in done and d <= done]
        if not ready:
            stuck = sorted(set(deps) - done)
            raise PlanError(f"plan has a dependency cycle among steps {stuck}")
        done.update(ready)


# (task, context, role, verify) -> (status, answer)
SpawnFn = Callable[[str, str, str, bool], Awaitable[tuple[str, str]]]
EventFn = Callable[[str, dict], None]


async def execute_plan(
    steps: list[Step],
    spawn: SpawnFn,
    out_dir: Path,
    shared_context: str = "",
    max_parallel: int = 3,
    on_event: EventFn | None = None,
) -> list[StepOutcome]:
    """
    Run steps respecting dependencies. Returns one outcome per step, in plan
    order. Budget aborts (HarnessAbort) propagate after in-flight steps are
    cancelled.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, StepOutcome] = {}
    by_id = {s.id: s for s in steps}
    sem = asyncio.Semaphore(max(1, max_parallel))

    def emit(kind: str, **data) -> None:
        if on_event:
            on_event(kind, data)

    def context_for(step: Step) -> str:
        parts = [shared_context] if shared_context else []
        if step.hint:
            parts.append(f"Hint: {step.hint}")
        for dep_id in step.depends_on:
            dep = outcomes[dep_id]
            summary = dep.answer.strip()
            if len(summary) > SUMMARY_CHARS:
                summary = summary[:SUMMARY_CHARS] + "…"
            parts.append(
                f"Result of step {dep_id} ({dep.step.task[:80]}):\n{summary}\n"
                f"(full output: {dep.output_path})"
            )
        return "\n\n".join(parts)

    async def run_step(step: Step) -> StepOutcome:
        async with sem:
            emit("step_start", id=step.id, role=step.role, task=step.task)
            status, answer = await spawn(step.task, context_for(step), step.role, step.verify)
            path = out_dir / f"step-{_safe(step.id)}.md"
            path.write_text(
                f"# Step {step.id} [{step.role}] — {status}\n\n## Task\n{step.task}\n\n## Result\n{answer}\n",
                encoding="utf-8",
            )
            emit("step_end", id=step.id, status=status)
            return StepOutcome(step, status, answer, str(path))

    pending = {s.id for s in steps}
    running: dict[asyncio.Task, str] = {}
    try:
        while pending or running:
            # Skip anything downstream of a failure.
            for sid in sorted(pending):
                failed = [d for d in by_id[sid].depends_on
                          if d in outcomes and outcomes[d].status != "ok"]
                if failed:
                    pending.discard(sid)
                    outcomes[sid] = StepOutcome(by_id[sid], "skipped",
                                                f"skipped: dependency {failed[0]} did not succeed")
                    emit("step_skipped", id=sid, because=failed[0])

            ready = [sid for sid in sorted(pending)
                     if all(d in outcomes and outcomes[d].status == "ok" for d in by_id[sid].depends_on)]
            for sid in ready:
                pending.discard(sid)
                running[asyncio.create_task(run_step(by_id[sid]))] = sid

            if not running:
                break   # nothing runnable and nothing in flight

            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                sid = running.pop(t)
                outcomes[sid] = t.result()   # re-raises HarnessAbort here
    except BaseException:
        for t in running:
            t.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise

    for sid in pending:   # unreachable in a validated plan; defensive
        outcomes[sid] = StepOutcome(by_id[sid], "skipped", "skipped: never became runnable")
    return [outcomes[s.id] for s in steps]


def summarize(outcomes: list[StepOutcome]) -> str:
    lines = []
    for o in outcomes:
        first = o.answer.strip().splitlines()[0][:160] if o.answer.strip() else ""
        lines.append(f"- step {o.step.id} [{o.step.role}] {o.status}: {first}")
    return "\n".join(lines)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:40] or "x"


def plan_to_json(steps: list[Step]) -> str:
    return json.dumps([s.__dict__ for s in steps], indent=2)
