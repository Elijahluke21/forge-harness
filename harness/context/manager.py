import json
import tempfile
from pathlib import Path

from harness.client import BaseClient, Response

# Default context windows
CONTEXT_WINDOWS = {
    "qwen3.6:27b": 32_768,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-haiku-4-5": 200_000,
    "default": 32_768,
}

COMPACT_AT   = 0.85   # summarize when context hits this fraction
WARN_AT      = 0.70   # log a warning at this fraction
OFFLOAD_CHARS = 3_000  # tool results larger than this go to a temp file


class ContextManager:
    """
    Tracks token usage, offloads large tool results to disk, and compacts
    the message history via summarization when the context budget fills up.

    Uses actual token counts from API responses as the primary signal.
    No external tokenizer library needed.
    """

    def __init__(
        self,
        model: str = "default",
        context_window: int | None = None,
        compact_at: float = COMPACT_AT,
        offload_chars: int = OFFLOAD_CHARS,
        budget_fraction: float = 1.0,
    ):
        base_window = context_window or CONTEXT_WINDOWS.get(model, CONTEXT_WINDOWS["default"])
        self.context_window = int(base_window * budget_fraction)
        self.compact_at = compact_at
        self.offload_chars = offload_chars

        self.last_input_tokens: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.compactions: int = 0

        self._offload_dir = Path(tempfile.mkdtemp(prefix="forge_ctx_"))

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def record(self, response: Response) -> None:
        """Call after every API response to keep token counts current."""
        self.last_input_tokens = response.input_tokens
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens

    @property
    def usage_fraction(self) -> float:
        """Fraction of context window used (based on last API call)."""
        if self.context_window == 0:
            return 0.0
        return self.last_input_tokens / self.context_window

    @property
    def usage_summary(self) -> str:
        pct = int(self.usage_fraction * 100)
        k = self.last_input_tokens / 1000
        cap_k = self.context_window / 1000
        return f"{k:.1f}k/{cap_k:.0f}k ({pct}%)"

    # ------------------------------------------------------------------
    # Large result offloading  (3.3)
    # ------------------------------------------------------------------

    def offload_if_large(self, content: str) -> str:
        """
        If content exceeds the threshold, write it to a temp file and return
        a compact pointer the model can act on with read_file.
        """
        if len(content) <= self.offload_chars:
            return content

        fname = f"result_{abs(hash(content)) % 10 ** 9:09x}.txt"
        path = self._offload_dir / fname
        path.write_text(content, encoding="utf-8")

        preview = content[:400].rstrip()
        return (
            f"[Output offloaded to conserve context — {len(content):,} chars total]\n"
            f"File: {path}\n"
            f"Preview:\n{preview}\n"
            f"...\n"
            f"Use read_file with offset/limit to retrieve specific sections."
        )

    # ------------------------------------------------------------------
    # Compaction  (3.2 + 3.4)
    # ------------------------------------------------------------------

    async def maybe_compact(
        self,
        messages: list[dict],
        client: BaseClient,
        system: str = "",
    ) -> str | None:
        """
        Check budget; compact if over threshold.
        Returns a status string if compaction happened, None otherwise.
        """
        if self.usage_fraction < self.compact_at:
            return None
        return await self._compact(messages, client)

    async def _compact(self, messages: list[dict], client: BaseClient) -> str:
        """
        Summarize the middle of the message history.
        Keeps the first 2 messages (establishes task context) and
        the last 6 (immediate working context), summarizes everything between.
        """
        if len(messages) <= 8:
            return "context high but history too short to compact"

        head = messages[:2]
        tail = messages[-6:]
        mid  = messages[2:-6]

        transcript = self._messages_to_transcript(mid)

        resp = await client.send(
            messages=[{"role": "user", "content": transcript}],
            system=(
                "Summarize this conversation segment precisely. "
                "Preserve: all files created/modified (with paths), commands run, "
                "command outputs, decisions made, and any facts established. "
                "Be concise. Output the summary only — no preamble."
            ),
            max_tokens=700,
        )

        summary_block: dict = {
            "role": "user",
            "content": (
                f"[Context compacted — {len(mid)} messages → 1 summary]\n\n"
                f"{resp.text}"
            ),
        }

        messages.clear()
        messages.extend(head + [summary_block] + tail)
        self.compactions += 1
        return f"compacted {len(mid)} messages into summary (compaction #{self.compactions})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_transcript(messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content[:800]}")
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        lines.append(f"{role}: {block['text'][:800]}")
                    elif btype == "tool_use":
                        args = json.dumps(block.get("input", {}))[:300]
                        lines.append(f"{role} [tool: {block['name']}] {args}")
                    elif btype == "tool_result":
                        val = str(block.get("content", ""))[:400]
                        lines.append(f"TOOL_RESULT: {val}")
        return "\n".join(lines)
