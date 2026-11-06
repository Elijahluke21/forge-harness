import asyncio
from collections.abc import Awaitable, Callable

from harness.client import DEFAULT_MAX_TOKENS, BaseClient, TextChunkFn, ToolCall
from harness.text import strip_think
from harness.tools.registry import ToolRegistry

DEFAULT_MAX_ITERATIONS = 20

# Returned when an agent runs out of iterations. Deliberately NOT "[error]":
# an agent that ran out of turns should be reported, not blindly retried with
# the same budget (which just doubles the cost of a task that is too big).
INCOMPLETE_PREFIX = "[incomplete]"

ToolCallback = Callable[[str, dict, str], Awaitable[None] | None]


async def _execute_all(registry: ToolRegistry, calls: list[ToolCall]) -> list[str]:
    """
    Run tool calls in parallel. If one raises (a budget abort, or the run being
    cancelled), cancel the others instead of leaving them running unattended.
    """
    async def one(tc: ToolCall) -> str:
        if tc.error:
            return f"[error] could not run '{tc.name}': {tc.error}"
        return await registry.execute(tc.name, tc.input)

    tasks = [asyncio.create_task(one(tc)) for tc in calls]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
    Mutates `messages` in place. Returns the final text response with any
    <think> blocks removed.

    Raises BudgetExceeded (via session_stats.check_budget) when the shared run
    budget is spent — callers that want a string should catch HarnessAbort.
    """
    for _ in range(max_iterations):
        if session_stats:
            session_stats.check_budget()

        response = await client.send(
            messages=messages,
            system=system,
            tools=registry.schemas() or None,
            max_tokens=max_tokens,
            on_text_chunk=on_stream_chunk if on_stream_chunk else None,
        )

        if context_manager:
            context_manager.record(response)
        if session_stats:
            session_stats.record_tokens(response.input_tokens, response.output_tokens)

        # Reasoning text is useful while generating, but re-sending it on every
        # later turn burns context for nothing — keep only the visible answer.
        visible = strip_think(response.text)

        if not response.has_tool_calls:
            messages.append({"role": "assistant", "content": visible})
            return visible

        assistant_content: list[dict] = []
        if visible:
            assistant_content.append({"type": "text", "text": visible})
        for tc in response.tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        messages.append({"role": "assistant", "content": assistant_content})

        results = await _execute_all(registry, response.tool_calls)

        if context_manager:
            results = [context_manager.offload_if_large(r) for r in results]

        if on_tool_call:
            for tc, result in zip(response.tool_calls, results):
                ret = on_tool_call(tc.name, tc.input, result)
                if asyncio.iscoroutine(ret):
                    await ret

        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                for tc, result in zip(response.tool_calls, results)
            ],
        })

        if context_manager:
            status = await context_manager.maybe_compact(messages, client, system)
            if status and on_tool_call:
                ret = on_tool_call("__compact__", {}, status)
                if asyncio.iscoroutine(ret):
                    await ret

    return f"{INCOMPLETE_PREFIX} stopped after {max_iterations} iterations without a final answer"
