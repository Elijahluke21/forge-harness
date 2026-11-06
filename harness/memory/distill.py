"""
Distillation: ask the model to extract memorable facts from a conversation
and write them into the memory store.

Called:
  - manually via /distill
  - at session end
"""
from harness.client import BaseClient
from harness.memory.store import MemoryStore
from harness.text import extract_json, strip_think


async def distill(
    messages: list[dict],
    client: BaseClient,
    store: MemoryStore,
) -> list[str]:
    """
    Extract facts from `messages` and persist them to `store`.
    Returns a list of keys that were written.
    """
    if not messages:
        return []

    # Build a compact transcript (skip tool_use/tool_result noise)
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str) and strip_think(content):
            lines.append(f"{role.upper()}: {strip_think(content)[:600]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = strip_think(block["text"])
                    if text:
                        lines.append(f"{role.upper()}: {text[:600]}")

    if not lines:
        return []

    transcript = "\n".join(lines)

    resp = await client.send(
        messages=[{"role": "user", "content": transcript}],
        system="""
You are a memory extraction assistant. Read the conversation and identify facts worth
remembering across sessions. Focus on:
- User name, role, preferences, or working style
- Project names, languages, frameworks, goals
- File paths, decisions made, constraints established
- Things the user explicitly asked to remember

Output ONLY a JSON array of objects. Each object: {"key": "slug_key", "content": "fact", "tags": ["tag1"]}
If nothing is worth remembering, output: []
""".strip(),
        max_tokens=600,
    )

    try:
        items = extract_json(resp.text)
    except ValueError:
        return []
    if not isinstance(items, list):
        return []

    written = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        content = str(item.get("content", "")).strip()
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        if key and content:
            store.set(key, content, tags)
            written.append(key)

    return written
