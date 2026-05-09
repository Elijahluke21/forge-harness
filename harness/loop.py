import asyncio
from collections.abc import Awaitable, Callable

from harness.client import DEFAULT_MAX_TOKENS, BaseClient, TextChunkFn
from harness.tools.registry import ToolRegistry

DEFAULT_MAX_ITERATIONS = 20

ToolCallback = Callable[[str, dict, str], Awaitable[None] | None]


async def run(
    messages: list[dict],
    client: BaseClient,
    registry: ToolRegistry,
    system: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    on_tool_call: ToolCallback | None = None,
    context_manager=None,       # harness.context.manager.ContextManager | None
    session_stats=None,         # harness.session.SessionStats | None
    on_stream_chunk: TextChunkFn | None = None,   # None = no streaming
) -> str:
    """
    Run the agentic loop until the model stops requesting tool calls.
    Mutates `messages` in place. Returns the final text response.
    """
    for iteration in range(max_iterations):
        # Only stream on the final response turn (no tool calls pending).
        # For intermediate turns we use regular send so tool calls process cleanly.
        response = await client.send(
            messages=messages,
            system=system,
            tools=registry.schemas() or None,
            max_tokens=max_tokens,
            on_text_chunk=on_stream_chunk if on_stream_chunk else None,
        )

        # Track token usage
        if context_manager:
            context_manager.record(response)
        if session_stats:
            session_stats.record_tokens(response.input_tokens, response.output_tokens)

        if not response.has_tool_calls:
            messages.append({"role": "assistant", "content": response.text})
            return response.text

        # Append assistant turn (text + tool_use blocks)
        assistant_content: list[dict] = []
        if response.text:
            assistant_content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute all tool calls in parallel
        results: list[str] = list(await asyncio.gather(*[
            registry.execute(tc.name, tc.input)
            for tc in response.tool_calls
        ]))

        # Offload large results before they enter the context
        if context_manager:
            results = [context_manager.offload_if_large(r) for r in results]

        # Fire callbacks
        if on_tool_call:
            for tc, result in zip(response.tool_calls, results):
                ret = on_tool_call(tc.name, tc.input, result)
                if asyncio.iscoroutine(ret):
                    await ret

        # Append tool results
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                for tc, result in zip(response.tool_calls, results)
            ],
        })

        # Compact if context budget is running low
        if context_manager:
            status = await context_manager.maybe_compact(messages, client, system)
            if status:
                # Surface compaction event through the tool callback channel
                if on_tool_call:
                    ret = on_tool_call("__compact__", {}, status)
                    if asyncio.iscoroutine(ret):
                        await ret

    return "[error] max_iterations reached without a final response"
