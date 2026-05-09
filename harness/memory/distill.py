"""
Distillation: ask the model to extract memorable facts from a conversation
and write them into the memory store.

Called:
  - manually via /distill
  - automatically when context compaction fires
  - at session end
"""
from harness.client import BaseClient
from harness.context.manager import ContextManager
from harness.memory.store import MemoryStore


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
        if isinstance(content, str) and content.strip():
            lines.append(f"{role.upper()}: {content[:600]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block["text"].strip()
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

    # Parse the JSON response
    text = resp.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.rstrip("`").strip()

    try:
        import json
        items = json.loads(text)
        if not isinstance(items, list):
            return []
    except (json.JSONDecodeError, ValueError):
        return []

    written = []
    for item in items:
        key = item.get("key", "").strip()
        content = item.get("content", "").strip()
        tags = item.get("tags", [])
        if key and content:
            store.set(key, content, tags)
            written.append(key)

    return written
