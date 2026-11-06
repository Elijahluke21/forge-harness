import json

from harness.memory.store import MemoryStore
from harness.tools.registry import ToolRegistry


def register(registry: ToolRegistry, store: MemoryStore, read_only: bool = False) -> None:
    """
    Register memory tools against a shared MemoryStore instance.

    read_only=True registers only read/search/list. Sub-agents get read-only
    memory: they read files and web pages, and text injected through those must
    not be able to plant "facts" that persist into every future session.
    """

    @registry.register(
        name="memory_read",
        description="Retrieve a specific memory entry by key.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The memory key to look up."},
            },
            "required": ["key"],
        },
    )
    async def memory_read(key: str) -> str:
        entry = store.get(key)
        if not entry:
            return f"[no memory found for key '{key}']"
        return json.dumps({
            "key": entry.key,
            "content": entry.content,
            "tags": entry.tags,
            "updated": entry.updated,
        }, indent=2)

    @registry.register(
        name="memory_search",
        description="Search memories by keyword. Returns all matching entries.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for."},
            },
            "required": ["query"],
        },
    )
    async def memory_search(query: str) -> str:
        results = store.search(query)
        if not results:
            return f"[no memories matched '{query}']"
        return "\n".join(f"[{e.key}]: {e.content}" for e in results)

    @registry.register(
        name="memory_list",
        description="List all memory keys and a short preview of each entry.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    async def memory_list() -> str:
        entries = store.all()
        if not entries:
            return "[memory is empty]"
        return "\n".join(f"[{e.key}] {e.content[:80]}" for e in entries)

    if read_only:
        return

        @registry.register(
            name="memory_write",
            description=(
                "Save a fact, preference, decision, or piece of context to persistent memory. "
                "Use this whenever you learn something worth remembering across sessions: "
                "user preferences, project facts, file locations, decisions made."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key":     {"type": "string", "description": "Short unique identifier (e.g. 'project_language', 'user_name')."},
                    "content": {"type": "string", "description": "The fact to remember."},
                    "tags":    {"type": "array",  "items": {"type": "string"}, "description": "Optional category tags."},
                },
                "required": ["key", "content"],
            },
        )
        async def memory_write(key: str, content: str, tags: list[str] | None = None) -> str:
            store.set(key, content, tags)
            return f"remembered: [{key}] {content}"

        @registry.register(
            name="memory_delete",
            description="Delete a memory entry by key.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
        )
        async def memory_delete(key: str) -> str:
            if store.delete(key):
                return f"deleted memory: {key}"
            return f"[no memory found for key '{key}']"
