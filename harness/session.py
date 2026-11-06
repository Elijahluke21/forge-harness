"""
Session-scoped objects shared across the entire agent tree.

SessionStats  — aggregate token/tool/spawn counts for the full run,
                optionally enforcing a RunBudget
ResultCache   — deduplicates identical read-only sub-agent tasks, invalidated
                whenever anything in the tree mutates the workspace
"""

import hashlib
from dataclasses import dataclass, field

from harness.budget import RunBudget


@dataclass
class SessionStats:
    input_tokens:   int = 0
    output_tokens:  int = 0
    tool_calls:     int = 0
    agents_spawned: int = 0
    cache_hits:     int = 0
    retries:        int = 0
    budget:         RunBudget | None = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def check_budget(self) -> None:
        """Raise BudgetExceeded if the run is over any limit."""
        if self.budget:
            self.budget.check(
                tokens=self.total_tokens,
                spawns=self.agents_spawned,
                tool_calls=self.tool_calls,
            )

    def record_tokens(self, inp: int, out: int) -> None:
        self.input_tokens  += inp
        self.output_tokens += out

    def record_tool(self) -> None:
        self.tool_calls += 1

    def record_spawn(self) -> None:
        self.agents_spawned += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_retry(self) -> None:
        self.retries += 1

    def summary(self) -> str:
        parts = [
            f"tokens {self.input_tokens:,} in / {self.output_tokens:,} out",
            f"agents {self.agents_spawned}",
            f"tools {self.tool_calls}",
        ]
        if self.cache_hits:
            parts.append(f"cache hits {self.cache_hits}")
        if self.retries:
            parts.append(f"retries {self.retries}")
        if self.budget and self.budget.max_wall_seconds:
            parts.append(f"{self.budget.elapsed():.0f}s")
        return " │ ".join(parts)


# Tools that change the workspace. Any call to one of these makes every cached
# result potentially stale ("run the tests" must re-run after a fix).
MUTATING_TOOLS = frozenset({"bash", "write_file", "edit_file"})


class ResultCache:
    """
    In-memory cache for sub-agent results.

    Key: md5(generation + role + task + context). `generation` is bumped every
    time a mutating tool runs anywhere in the tree, so a result is only reused
    while the workspace is provably unchanged.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.generation = 0

    def invalidate(self) -> None:
        self.generation += 1
        self._store.clear()

    def _key(self, task: str, context: str, role: str) -> str:
        raw = f"{self.generation}|||{role}|||{task}|||{context}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, task: str, context: str = "", role: str = "") -> str | None:
        return self._store.get(self._key(task, context, role))

    def put(self, task: str, context: str, result: str, role: str = "") -> None:
        self._store[self._key(task, context, role)] = result

    @property
    def size(self) -> int:
        return len(self._store)
