import json
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

    def format_for_prompt(self) -> str:
        entries = self.all()
        if not entries:
            return ""
        lines = ["## Forge Memory (persistent facts from previous sessions)"]
        for e in entries:
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"- [{e.key}]{tag_str}: {e.content}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(e) for e in self._entries.values()], indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data:
                e = MemoryEntry(**item)
                self._entries[e.key] = e
        except (json.JSONDecodeError, TypeError):
            pass  # corrupt file — start fresh


def _slugify(key: str) -> str:
    return key.lower().strip().replace(" ", "_")
