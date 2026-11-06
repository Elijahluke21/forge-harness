import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path.home() / ".forge" / "memory.json"


@dataclass
class MemoryEntry:
    key: str
    content: str
    tags: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    updated: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryStore:
    """
    Persistent key/value memory backed by a JSON file.
    All agents in a session share one instance — this is the shared memory bus.
    Changes are written to disk immediately on every set/delete.
    """

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def set(self, key: str, content: str, tags: list[str] | None = None) -> None:
        slug = _slugify(key)
        existing = self._entries.get(slug)
        self._entries[slug] = MemoryEntry(
            key=slug,
            content=content,
            tags=tags or (existing.tags if existing else []),
            created=existing.created if existing else datetime.now().isoformat(),
            updated=datetime.now().isoformat(),
        )
        self._save()

    def get(self, key: str) -> MemoryEntry | None:
        return self._entries.get(_slugify(key))

    def delete(self, key: str) -> bool:
        slug = _slugify(key)
        if slug in self._entries:
            del self._entries[slug]
            self._save()
            return True
        return False

    def all(self) -> list[MemoryEntry]:
        return sorted(self._entries.values(), key=lambda e: e.updated, reverse=True)

    def search(self, query: str) -> list[MemoryEntry]:
        """Simple keyword search across key, content, and tags."""
        words = query.lower().split()
        results = []
        for entry in self._entries.values():
            haystack = f"{entry.key} {entry.content} {' '.join(entry.tags)}".lower()
            if all(w in haystack for w in words):
                results.append(entry)
        return sorted(results, key=lambda e: e.updated, reverse=True)

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    def relevant(self, query: str = "", limit: int = 8) -> list[MemoryEntry]:
        """
        Entries tagged "pinned" first, then entries ranked by word overlap with
        `query`, most recently updated breaking ties. No query = most recent.
        """
        pinned = [e for e in self.all() if "pinned" in e.tags]
        rest = [e for e in self.all() if "pinned" not in e.tags]
        words = _words(query)
        if words:
            scored = []
            for e in rest:
                hits = len(words & _words(f"{e.key} {e.content} {' '.join(e.tags)}"))
                if hits:
                    scored.append((hits, e.updated, e))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            rest = [e for _, _, e in scored]
        return (pinned + rest)[:limit]

    def format_for_prompt(self, query: str = "", limit: int = 8, max_chars: int = 2_000) -> str:
        """
        A bounded memory block for a system prompt. Dumping every entry into
        every agent's prompt grows without limit and crowds out the task on a
        small context window, so only the most relevant entries are included.
        """
        entries = self.relevant(query, limit)
        if not entries:
            return ""
        lines = ["## Forge Memory (relevant facts from previous sessions)"]
        used = len(lines[0])
        for e in entries:
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            line = f"- [{e.key}]{tag_str}: {e.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        if len(self) > len(lines) - 1:
            lines.append(f"({len(self)} memories total — use memory_search for others)")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps([asdict(e) for e in self._entries.values()], indent=2)
        # Atomic replace: a crash mid-write must not corrupt the whole store.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".memory-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data:
                e = MemoryEntry(**item)
                self._entries[e.key] = e
        except (json.JSONDecodeError, TypeError):
            # Keep the corrupt file for inspection instead of overwriting it
            # on the next save.
            backup = self.path.with_suffix(".corrupt.json")
            self.path.replace(backup)


def _slugify(key: str) -> str:
    return key.lower().strip().replace(" ", "_")


_STOP = frozenset("a an the and or of to in on for with is are be this that it as at by from".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if w not in _STOP and len(w) > 1}
