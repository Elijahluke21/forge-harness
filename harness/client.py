"""
Client abstraction layer.

Internal message format is Anthropic-style throughout the harness.
Each backend translates to/from its own wire format here so the
loop and tools never need to know which provider is in use.

Anthropic tool_use block:
    {"type": "tool_use", "id": "...", "name": "...", "input": {...}}

Anthropic tool_result block (inside a user message):
    {"type": "tool_result", "tool_use_id": "...", "content": "..."}
"""

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

DEFAULT_MAX_TOKENS = 8096
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"
OLLAMA_DEFAULT_MODEL = "qwen3.6:27b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

TextChunkFn = Callable[[str], None]   # called with each streamed text chunk


# ---------------------------------------------------------------------------
# Normalised response
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Response:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseClient(ABC):
    @abstractmethod
    async def send(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_text_chunk: TextChunkFn | None = None,
    ) -> Response: ...

    def count_tokens(self, messages: list[dict], system: str = "") -> int:
        text = system + "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for msg in messages
            for part in (
                msg["content"]
                if isinstance(msg["content"], list)
                else [{"text": msg["content"]}]
            )
        )
        return len(text) // 4


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class AnthropicClient(BaseClient):
    def __init__(self, model: str = ANTHROPIC_DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._api = AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    async def send(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_text_chunk: TextChunkFn | None = None,
    ) -> Response:
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        if on_text_chunk:
            return await self._send_streaming(kwargs, on_text_chunk)

        msg = await self._api.messages.create(**kwargs)
        text, tool_calls = "", []
        for block in msg.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
        return Response(text=text, tool_calls=tool_calls,
                        input_tokens=msg.usage.input_tokens,
                        output_tokens=msg.usage.output_tokens)

    async def _send_streaming(self, kwargs: dict, on_text_chunk: TextChunkFn) -> Response:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        async with self._api.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "content_block_delta":
                    delta = event.delta
                    if getattr(delta, "type", "") == "text_delta":
                        chunk = delta.text
                        text_parts.append(chunk)
                        on_text_chunk(chunk)

            msg = await stream.get_final_message()
            for block in msg.content:
                if block.type == "tool_use":
                    tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return Response(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )


# ---------------------------------------------------------------------------
# Ollama backend  (OpenAI-compatible wire format)
# ---------------------------------------------------------------------------

class OllamaClient(BaseClient):
    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self._api = AsyncOpenAI(api_key="ollama", base_url=base_url)

    async def send(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_text_chunk: TextChunkFn | None = None,
    ) -> Response:
        oai_messages = self._to_oai_messages(messages, system)
        oai_tools = self._to_oai_tools(tools) if tools else None
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens, messages=oai_messages)
        if oai_tools:
            kwargs["tools"] = oai_tools

        if on_text_chunk:
            return await self._send_streaming(kwargs, on_text_chunk)

        completion = await self._api.chat.completions.create(**kwargs)
        return self._parse_completion(completion)

    async def _send_streaming(self, kwargs: dict, on_text_chunk: TextChunkFn) -> Response:
        text_parts: list[str] = []
        # Assemble tool call fragments: index → {id, name, arguments}
        tc_fragments: dict[int, dict] = {}
        prompt_tokens = 0
        completion_tokens = 0

        stream = await self._api.chat.completions.create(**kwargs, stream=True,
                                                          stream_options={"include_usage": True})
        async for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                text_parts.append(delta.content)
                on_text_chunk(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_fragments:
                        tc_fragments[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tc_fragments[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_fragments[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_fragments[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = []
        for idx in sorted(tc_fragments):
            frag = tc_fragments[idx]
            try:
                inp = json.loads(frag["arguments"]) if frag["arguments"] else {}
            except json.JSONDecodeError:
                inp = {}
            tool_calls.append(ToolCall(id=frag["id"], name=frag["name"], input=inp))

        return Response(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )

    def _parse_completion(self, completion) -> Response:
        choice = completion.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))
        usage = completion.usage
        return Response(
            text=text,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def _to_oai_messages(self, messages: list[dict], system: str) -> list[dict]:
        result = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            if role == "assistant":
                text_parts, tool_calls = [], []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"], "type": "function",
                            "function": {"name": block["name"],
                                         "arguments": json.dumps(block["input"])},
                        })
                oai_msg: dict = {"role": "assistant", "content": " ".join(text_parts) or None}
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)

            elif role == "user":
                for tr in [b for b in content if b.get("type") == "tool_result"]:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"] if isinstance(tr["content"], str)
                                   else json.dumps(tr["content"]),
                    })
                text_blocks = [b for b in content if b.get("type") == "text"]
                if text_blocks:
                    result.append({"role": "user",
                                   "content": " ".join(b["text"] for b in text_blocks)})

        return result

    def _to_oai_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }}
            for t in tools
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(backend: str = "ollama", **kwargs) -> BaseClient:
    if backend == "anthropic":
        return AnthropicClient(**kwargs)
    if backend == "ollama":
        return OllamaClient(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'anthropic' or 'ollama'.")
