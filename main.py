import asyncio
from pathlib import Path

from harness.agent import Agent, make_spawn_tool
from harness.client import make_client
from harness.config import load_config
from harness.context.manager import ContextManager
from harness.memory import tools as memory_tools
from harness.memory.store import MemoryStore
from harness.planning import make_plan_tool
from harness.prompt import build_system_prompt
from harness.session import ResultCache, SessionStats
from harness.tools import bash, files
from harness.tools.registry import ToolRegistry
from harness.ui.repl import confirm_tool, on_tool, on_spawn, repl


def build_worker_registry(store: MemoryStore) -> ToolRegistry:
    registry = ToolRegistry()
    bash.register(registry)
    files.register(registry)
    memory_tools.register(registry, store)
    return registry


ORCHESTRATOR_SYSTEM = build_system_prompt(extra="\n".join([
    "## Orchestration strategy",
    "You are the orchestrator. For any non-trivial multi-step task:",
    "1. Call plan_task first to produce a structured execution plan.",
    "2. Execute the plan using spawn_agent — spawn independent steps in parallel.",
    "   - Use the `role` parameter to give sub-agents focused capabilities.",
    "   - Set `verify=true` for critical steps that need confirmation.",
    "3. Synthesize results and respond to the user.",
    "Reserve direct tool use (bash, files) for quick single-step actions only.",
    "Use memory_write to persist anything worth remembering across sessions.",
]))


async def main():
    cfg = load_config()

    client = make_client(backend=cfg.backend, model=cfg.model)
    store  = MemoryStore(Path(cfg.memory_path).expanduser())
    stats  = SessionStats()
    cache  = ResultCache()
    ctx    = ContextManager(
        model=cfg.model,
        context_window=cfg.context_window,
        compact_at=cfg.compact_at,
        budget_fraction=cfg.budget_fraction,
    )

    worker_registry = build_worker_registry(store)

    orchestrator = Agent(
        client=client,
        registry=worker_registry,   # placeholder — replaced below
        system=ORCHESTRATOR_SYSTEM,
        max_tokens=cfg.max_tokens,
        max_iterations=cfg.max_iterations,
        on_tool_call=on_tool,
        on_spawn=on_spawn,
        context_manager=ctx,
        memory_store=store,
        session_stats=stats,
        result_cache=cache,
    )

    orch_registry = ToolRegistry()
    bash.register(orch_registry)
    files.register(orch_registry)
    memory_tools.register(orch_registry, store)
    orch_registry.add(make_plan_tool(client, store))
    orch_registry.add(make_spawn_tool(orchestrator, worker_registry))

    # Permissions
    if cfg.require_confirm:
        orch_registry.set_permissions(cfg.require_confirm, confirm_tool)
        worker_registry.set_permissions(cfg.require_confirm, confirm_tool)

    orchestrator.registry = orch_registry

    await repl(orchestrator, stream=cfg.stream, hooks=cfg.hooks)


def cli():
    asyncio.run(main())


if __name__ == "__main__":
    cli()
