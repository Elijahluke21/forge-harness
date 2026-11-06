"""
Autopilot (7.1–7.4): run a goal unattended until the acceptance checks pass,
then stop. The model never decides when it is finished.

    round N:
      1. run the checks                     → all pass?  done
      2. progress watchdog                  → no new passing checks for
                                              `stall_rounds` rounds? escalate the
                                              planner once, then stop (stalled)
      3. plan against the failing checks    → validated step graph
      4. execute the plan in code           → sub-agents do the work
    any time: run budget spent              → stop (budget)

Everything lands in a run directory the person can read later:

    .forge/runs/<timestamp>/
      task.md          the frozen goal and checks (the spec)
      events.jsonl     every round, plan, step and check result
      round-N/         plan.json + one file per step's full output
      result.md        written on success
      blocked.md       written on any other outcome, with the reason

An optional `on_finish` shell command runs at the end with FORGE_STATUS,
FORGE_RUN_DIR and FORGE_SUMMARY in its environment — use it to push a
notification to your phone or watch.
"""

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from harness.agent import Agent, spawn_subagent
from harness.budget import BudgetExceeded
from harness.checks import Check, CheckReport, run_checks
from harness.client import BaseClient
from harness.executor import PlanError, execute_plan, plan_to_json, summarize
from harness.planning import generate_plan
from harness.tools.registry import ToolRegistry
from harness.tools.shell import spawn_shell


@dataclass
class AutopilotConfig:
    max_rounds: int = 8          # plan→execute cycles before giving up
    stall_rounds: int = 3        # rounds with no new passing check before stopping
    max_parallel: int = 3        # concurrent steps within a round
    on_finish: str = ""          # shell command run at the end (notifications)


@dataclass
class AutopilotResult:
    status: str                  # done | stalled | budget | max_rounds
    rounds: int
    run_dir: Path
    report: CheckReport | None
    reason: str = ""
    history: list[str] = field(default_factory=list)


EventFn = Callable[[str, dict], None]


class Autopilot:
    def __init__(
        self,
        orchestrator: Agent,
        worker_registry: ToolRegistry,
        checks: list[Check],
        workdir: str | Path,
        config: AutopilotConfig | None = None,
        run_root: str | Path | None = None,
        escalation_client: BaseClient | None = None,
        on_event: EventFn | None = None,
    ):
        if not checks:
            raise ValueError("autopilot needs at least one acceptance check")
        self.agent = orchestrator
        self.workers = worker_registry
        # Checks are loaded once and held in memory: an agent editing the
        # checks file mid-run cannot redefine what "done" means.
        self.checks = list(checks)
        self.workdir = Path(workdir).resolve()
        self.cfg = config or AutopilotConfig()
        self.run_root = Path(run_root) if run_root else self.workdir / ".forge" / "runs"
        self.escalation_client = escalation_client
        self.on_event = on_event
        self.run_dir: Path | None = None

    # ------------------------------------------------------------------

    def _emit(self, kind: str, **data) -> None:
        if self.run_dir:
            with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"),
                                    "event": kind, **data}, default=str) + "\n")
        if self.on_event:
            self.on_event(kind, data)

    def _write_spec(self, goal: str) -> None:
        lines = [f"# Goal\n\n{goal}\n", "# Done when every check passes\n"]
        for c in self.checks:
            desc = f" — {c.description}" if c.description else ""
            lines.append(f"- **{c.id}**{desc}\n  `{c.cmd}`")
        (self.run_dir / "task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _shared_context(self, goal: str) -> str:
        return (
            f"Overall goal: {goal}\n"
            f"Working directory: {self.workdir}\n"
            "Do only your assigned step. Do not edit acceptance checks or tests to make them pass "
            "unless your step explicitly says so. Report exactly what you changed (file paths)."
        )

    async def _spawn(self, task: str, context: str, role: str, verify: bool) -> tuple[str, str]:
        result = await spawn_subagent(self.agent, self.workers, task, context, role, verify)
        answer = result.answer
        if result.verification:
            answer += f"\n\n[verification] {result.verification}"
        return result.status, answer

    # ------------------------------------------------------------------

    async def run(self, goal: str) -> AutopilotResult:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.run_root / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_spec(goal)
        self._emit("run_start", goal=goal, checks=[c.id for c in self.checks])

        planner: BaseClient = self.agent.client
        escalated = False
        best = -1
        stalls = 0
        history: list[str] = []
        report: CheckReport | None = None
        rnd = 0
        status, reason = "max_rounds", f"checks still failing after {self.cfg.max_rounds} rounds"

        try:
            while True:
                if self.agent.session_stats:
                    self.agent.session_stats.check_budget()

                report = await run_checks(self.checks, self.workdir)
                self._emit("checks", round=rnd, score=report.score(),
                           passed=sorted(report.passed_ids),
                           failed=[r.check.id for r in report.failed])

                if report.all_passed:
                    status, reason = "done", f"all {len(self.checks)} checks pass"
                    break
                if rnd >= self.cfg.max_rounds:
                    break

                # ── Progress watchdog (7.4) ───────────────────────────
                score = len(report.passed)
                if score > best:
                    best, stalls = score, 0
                elif rnd > 0:
                    stalls += 1
                if stalls >= self.cfg.stall_rounds:
                    if self.escalation_client and not escalated:
                        escalated, stalls, planner = True, 0, self.escalation_client
                        self._emit("escalate", round=rnd, model=getattr(planner, "model", "?"))
                    else:
                        status = "stalled"
                        reason = (f"no new passing checks for {self.cfg.stall_rounds} rounds "
                                  f"(stuck at {report.score()})")
                        break

                rnd += 1
                round_dir = self.run_dir / f"round-{rnd}"
                round_dir.mkdir(exist_ok=True)

                # ── Plan against what is failing ──────────────────────
                plan_context = "\n\n".join(filter(None, [
                    f"Acceptance checks passing: {report.score()}.",
                    "Failing checks:\n" + report.failure_summary(),
                    ("What earlier rounds did:\n" + "\n".join(history[-3:])) if history else "",
                    f"Working directory: {self.workdir}",
                ]))
                objective = f"{goal}\n\nPlan only the work needed to make the failing checks pass."
                try:
                    steps = await generate_plan(planner, objective, plan_context, self.agent.session_stats)
                except PlanError as exc:
                    self._emit("plan_failed", round=rnd, error=str(exc))
                    history.append(f"round {rnd}: planning failed ({exc})")
                    continue   # counts as a round with no progress
                (round_dir / "plan.json").write_text(plan_to_json(steps), encoding="utf-8")
                self._emit("plan", round=rnd, steps=[{"id": s.id, "role": s.role, "task": s.task} for s in steps])

                # ── Execute in code (7.3) ─────────────────────────────
                outcomes = await execute_plan(
                    steps, self._spawn, round_dir,
                    shared_context=self._shared_context(goal),
                    max_parallel=self.cfg.max_parallel,
                    on_event=lambda kind, data, r=rnd: self._emit(kind, round=r, **data),
                )
                history.append(f"round {rnd}:\n{summarize(outcomes)}")

        except BudgetExceeded as exc:
            status, reason = "budget", str(exc)

        result = AutopilotResult(status, rnd, self.run_dir, report, reason, history)
        self._finish(goal, result)
        await self._notify(result)
        return result

    # ------------------------------------------------------------------

    def _finish(self, goal: str, result: AutopilotResult) -> None:
        stats = self.agent.session_stats.summary() if self.agent.session_stats else ""
        lines = [
            f"# {'Done' if result.status == 'done' else 'Blocked'}: {result.status}",
            "", f"**Goal:** {goal}", f"**Reason:** {result.reason}",
            f"**Rounds:** {result.rounds}",
        ]
        if result.report:
            lines.append(f"**Checks:** {result.report.score()} passing")
        if stats:
            lines.append(f"**Usage:** {stats}")
        if result.report and result.report.failed:
            lines += ["", "## Still failing", "", result.report.failure_summary()]
        if result.history:
            lines += ["", "## Rounds", "", *result.history]
        name = "result.md" if result.status == "done" else "blocked.md"
        (self.run_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._emit("run_end", status=result.status, reason=result.reason, rounds=result.rounds)

    async def _notify(self, result: AutopilotResult) -> None:
        if not self.cfg.on_finish:
            return
        env = {
            **os.environ,
            "FORGE_STATUS": result.status,
            "FORGE_RUN_DIR": str(result.run_dir),
            "FORGE_SUMMARY": f"{result.status}: {result.reason}"[:300],
        }
        try:
            proc = await spawn_shell(
                self.cfg.on_finish, env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
        except (asyncio.TimeoutError, OSError):
            pass


__all__ = ["Autopilot", "AutopilotConfig", "AutopilotResult"]
