"""
Acceptance checks (7.2).

"Done" is decided by code, not by the model. A checks file lists commands;
the run is complete when every one exits 0.

checks.yaml:

    checks:
      - id: builds
        cmd: xcodebuild -scheme MyApp -destination 'generic/platform=iOS Simulator' build
        timeout: 900
        description: The app compiles for the simulator
      - id: unit-tests
        cmd: swift test
        timeout: 600

`description` is shown to the planner so it knows what the check means; the
command's output tail is shown when it fails.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness.text import tail
from harness.tools.bash import run_shell

DEFAULT_CHECK_TIMEOUT = 300
FAILURE_TAIL_CHARS = 1_500


@dataclass
class Check:
    id: str
    cmd: str
    timeout: int = DEFAULT_CHECK_TIMEOUT
    description: str = ""


@dataclass
class CheckResult:
    check: Check
    passed: bool
    exit_code: int | None
    output: str


@dataclass
class CheckReport:
    results: list[CheckResult]

    @property
    def passed(self) -> list[CheckResult]:
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and not self.failed

    @property
    def passed_ids(self) -> frozenset[str]:
        return frozenset(r.check.id for r in self.passed)

    def score(self) -> str:
        return f"{len(self.passed)}/{len(self.results)}"

    def failure_summary(self, max_chars: int = FAILURE_TAIL_CHARS) -> str:
        """What the planner sees about failing checks: description + output tail."""
        parts = []
        for r in self.failed:
            code = "timed out" if r.exit_code is None else f"exit {r.exit_code}"
            desc = f" — {r.check.description}" if r.check.description else ""
            parts.append(
                f"### {r.check.id} FAILED ({code}){desc}\n$ {r.check.cmd}\n{tail(r.output.strip(), max_chars)}"
            )
        return "\n\n".join(parts)


def load_checks(path: str | Path) -> list[Check]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = data.get("checks", data) if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list under 'checks'")
    checks, seen = [], set()
    for i, item in enumerate(raw, 1):
        if isinstance(item, str):
            item = {"cmd": item}
        if not isinstance(item, dict) or not item.get("cmd"):
            raise ValueError(f"{path}: check {i} needs a 'cmd'")
        cid = str(item.get("id") or f"check-{i}")
        if cid in seen:
            raise ValueError(f"{path}: duplicate check id {cid!r}")
        seen.add(cid)
        checks.append(Check(
            id=cid,
            cmd=str(item["cmd"]),
            timeout=int(item.get("timeout", DEFAULT_CHECK_TIMEOUT)),
            description=str(item.get("description", "")),
        ))
    return checks


async def run_checks(checks: list[Check], cwd: str | Path, parallel: bool = False) -> CheckReport:
    """
    Run every check. Sequential by default: builds and tests usually share
    build directories and simulators, and parallel runs fight over them.
    """
    async def one(c: Check) -> CheckResult:
        code, out = await run_shell(c.cmd, c.timeout, cwd=str(cwd))
        return CheckResult(c, code == 0, code, out)

    if parallel:
        results = list(await asyncio.gather(*(one(c) for c in checks)))
    else:
        results = [await one(c) for c in checks]
    return CheckReport(results)
