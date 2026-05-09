"""
Session-scoped objects shared across the entire agent tree.

SessionStats  — aggregate token/tool/spawn counts for the full run
ResultCache   — deduplicates identical sub-agent tasks within a session
"""

import hashlib
from dataclasses import dataclass, field


@dataclass
class SessionStats:
    input_tokens:   int = 0
    output_tokens:  int = 0
    tool_calls:     int = 0
    agents_spawned: int = 0
    cache_hits:     int = 0
    retries:        int = 0

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
        return " │ ".join(parts)


class ResultCache:
    """
    In-memory cache for sub-agent results.
    Key: md5(task + context).  Cleared when the session ends.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def _key(self, task: str, context: str) -> str:
        return hashlib.md5(f"{task}|||{context}".encode()).hexdigest()

    def get(self, task: str, context: str = "") -> str | None:
        return self._store.get(self._key(task, context))

    def put(self, task: str, context: str, result: str) -> None:
        self._store[self._key(task, context)] = result

    @property
    def size(self) -> int:
        return len(self._store)
