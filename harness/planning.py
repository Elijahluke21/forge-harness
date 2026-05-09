"""
plan_task tool — asks the model to decompose an objective into a structured
execution plan before any sub-agents are spawned.

Returns a JSON array of steps:
  [{"id": "1", "task": "...", "role": "executor", "depends_on": [], "hint": "..."}]

The orchestrator uses this to sequence spawn_agent calls intelligently.
Dynamic re-planning: the orchestrator can call plan_task again mid-execution
to revise the plan based on intermediate results.
"""

import json

from harness.client import BaseClient
from harness.memory.store import MemoryStore
from harness.tools.registry import Tool

PLAN_SYSTEM = """
You are a task decomposition expert.
Given an objective, output ONLY a valid JSON array of steps — no preamble, no explanation.

Each step must have:
  "id"         : unique string ("1", "2", ...)
  "task"       : clear, self-contained description of the work
  "role"       : one of: planner, reader, executor, writer, synthesizer, verifier
  "depends_on" : list of step ids this step waits for ([] if independent)
  "hint"       : one sentence of context to pass to the sub-agent

Prefer parallelism: steps with no shared dependencies should have empty depends_on.
Output JSON only. No markdown fences.
""".strip()


def make_plan_tool(client: BaseClient, memory_store: MemoryStore | None = None) -> Tool:

    async def plan_task(objective: str, store_as: str = "") -> str:
        """
        Generate a structured execution plan for an objective.
        Optionally saves the plan to memory under `store_as` key.
        """
        resp = await client.send(
            messages=[{"role": "user", "content": f"Objective: {objective}"}],
            system=PLAN_SYSTEM,
            max_tokens=1_000,
        )

        # Robustly strip markdown fences if the model adds them
        text = resp.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]).rstrip("`").strip()

        # Validate JSON
        try:
            steps = json.loads(text)
            if not isinstance(steps, list):
                return f"[error] plan_task: model returned non-list JSON: {text[:200]}"
        except json.JSONDecodeError as exc:
            return f"[error] plan_task: could not parse plan JSON: {exc}\nRaw: {text[:300]}"

        # Optionally persist to memory for later reference
        if store_as and memory_store:
            memory_store.set(
                store_as,
                json.dumps(steps, indent=2),
                tags=["plan"],
            )

        # Format for easy reading by the orchestrator
        lines = [f"Plan for: {objective}", ""]
        for step in steps:
            deps = f"  (after {', '.join(step.get('depends_on', []))})" if step.get("depends_on") else ""
            lines.append(
                f"Step {step['id']} [{step['role']}]{deps}: {step['task']}"
            )
            if step.get("hint"):
                lines.append(f"  → hint: {step['hint']}")
        lines += ["", "Full JSON:", json.dumps(steps, indent=2)]

        return "\n".join(lines)

    return Tool(
        name="plan_task",
        description=(
            "Decompose a complex objective into a structured, ordered execution plan. "
            "Call this BEFORE spawning sub-agents for any multi-step task. "
            "Returns a human-readable plan + JSON you can use to sequence spawn_agent calls. "
            "Independent steps can be spawned in parallel (same response). "
            "Call again mid-execution to revise the plan based on new information."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "The high-level goal to decompose into steps.",
                },
                "store_as": {
                    "type": "string",
                    "description": "Optional memory key to persist the plan for later reference.",
                },
            },
            "required": ["objective"],
        },
        executor=plan_task,
    )
