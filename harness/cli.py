"""
Forge command line.

  forge                                   interactive REPL (as before)
  forge run "goal" --checks checks.yaml   unattended autopilot: works until every
                                          check passes, the run stalls, or the
                                          budget is spent
"""

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

from harness.app import build_forge
from harness.autopilot import Autopilot, AutopilotConfig
from harness.checks import load_checks
from harness.config import load_config

console = Console()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge", description="Forge agent harness")
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a goal unattended until its acceptance checks pass")
    run.add_argument("goal", help="what to build or fix (or @path to read it from a file)")
    run.add_argument("--checks", required=True, help="YAML file of acceptance checks")
    run.add_argument("--workspace", help="directory the agents work in (default: config or cwd)")
    run.add_argument("--max-rounds", type=int, help="plan→execute rounds before giving up")
    run.add_argument("--stall-rounds", type=int, help="rounds without progress before stopping")
    run.add_argument("--max-minutes", type=float, help="wall-clock budget for the whole run")
    run.add_argument("--max-tokens", type=int, help="token budget for the whole run")
    return p


def _on_event(kind: str, data: dict) -> None:
    if kind == "checks":
        style = "green" if not data["failed"] else "yellow"
        console.print(Text(f"◆ round {data['round']} checks {data['score']}"
                           + (f" — failing: {', '.join(data['failed'])}" if data["failed"] else ""), style=style))
    elif kind == "plan":
        console.print(Text(f"  plan: {len(data['steps'])} steps", style="cyan"))
        for s in data["steps"]:
            console.print(Text(f"    {s['id']} [{s['role']}] {s['task'][:90]}", style="dim"))
    elif kind == "step_end":
        style = "dim green" if data["status"] == "ok" else "red"
        console.print(Text(f"    ✓ step {data['id']}: {data['status']}", style=style))
    elif kind == "step_skipped":
        console.print(Text(f"    ↷ step {data['id']} skipped (dependency {data['because']} failed)", style="dim red"))
    elif kind in ("plan_failed", "escalate"):
        console.print(Text(f"  {kind}: {data}", style="bold yellow"))
    elif kind == "run_end":
        style = "bold green" if data["status"] == "done" else "bold red"
        console.print(Text(f"■ {data['status']}: {data['reason']}", style=style))


async def _run_autopilot(args) -> int:
    cfg = load_config()
    if args.max_rounds:
        cfg.max_rounds = args.max_rounds
    if args.stall_rounds:
        cfg.stall_rounds = args.stall_rounds
    if args.max_minutes:
        cfg.budget_max_minutes = args.max_minutes
    if args.max_tokens:
        cfg.budget_max_tokens = args.max_tokens

    goal = args.goal
    if goal.startswith("@"):
        goal = Path(goal[1:]).read_text(encoding="utf-8")

    checks = load_checks(args.checks)
    forge = build_forge(cfg, workspace=args.workspace)
    workdir = forge.workspace.cwd

    pilot = Autopilot(
        forge.orchestrator, forge.worker_registry, checks, workdir,
        config=AutopilotConfig(cfg.max_rounds, cfg.stall_rounds, cfg.max_parallel, cfg.on_finish),
        escalation_client=forge.escalation_client,
        on_event=_on_event,
    )
    console.print(Text(f"Forge autopilot · {cfg.model} · {workdir}", style="bold cyan"))
    result = await pilot.run(goal)
    console.print(Text(f"Run log: {result.run_dir}", style="dim"))
    if forge.orchestrator.session_stats:
        console.print(Text(forge.orchestrator.session_stats.summary(), style="dim"))
    return 0 if result.status == "done" else 1


async def _run_repl() -> None:
    from harness.ui.repl import confirm_tool, on_spawn, on_tool, repl

    cfg = load_config()
    forge = build_forge(cfg, on_tool_call=on_tool, on_spawn=on_spawn, confirm_fn=confirm_tool)
    await repl(forge.orchestrator, stream=cfg.stream)


def cli(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        sys.exit(asyncio.run(_run_autopilot(args)))
    asyncio.run(_run_repl())


if __name__ == "__main__":
    cli()
