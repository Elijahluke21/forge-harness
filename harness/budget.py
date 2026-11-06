"""
Run budgets (7.1).

Per-agent `max_iterations` alone cannot bound a recursive agent tree: every
agent may spawn children, every child may spawn more. A RunBudget is one shared
ceiling for the whole tree. When any limit is crossed, `check()` raises
BudgetExceeded, which unwinds the entire run instead of being swallowed as a
tool error.
"""

import time
from dataclasses import dataclass, field


class HarnessAbort(Exception):
    """Base for exceptions that must stop the whole run, never be retried."""


class BudgetExceeded(HarnessAbort):
    def __init__(self, limit: str, used: float, cap: float):
        self.limit, self.used, self.cap = limit, used, cap
        super().__init__(f"run budget exceeded: {limit} ({used:,.0f} / {cap:,.0f})")


@dataclass
class RunBudget:
    """0 (or None) disables a limit."""
    max_total_tokens: int = 0       # input + output across every agent
    max_spawns:       int = 0       # sub-agents created
    max_tool_calls:   int = 0
    max_wall_seconds: float = 0
    started_at:       float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def check(self, *, tokens: int, spawns: int, tool_calls: int) -> None:
        if self.max_total_tokens and tokens > self.max_total_tokens:
            raise BudgetExceeded("tokens", tokens, self.max_total_tokens)
        if self.max_spawns and spawns > self.max_spawns:
            raise BudgetExceeded("spawns", spawns, self.max_spawns)
        if self.max_tool_calls and tool_calls > self.max_tool_calls:
            raise BudgetExceeded("tool_calls", tool_calls, self.max_tool_calls)
        if self.max_wall_seconds and self.elapsed() > self.max_wall_seconds:
            raise BudgetExceeded("wall_seconds", self.elapsed(), self.max_wall_seconds)
