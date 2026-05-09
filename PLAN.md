# Forge Harness — Living Plan
# Philosophy: The orchestrator never does work directly. It decomposes, delegates, and synthesizes.

---

## Core Architecture Philosophy

```
User
 │
 ▼
Orchestrator Agent          ← tiny context, high-level reasoning only
 │  (spawns, synthesizes)
 ├──► Sub-agent A           ← fresh context, scoped task, specialized tools
 ├──► Sub-agent B           ← fresh context, scoped task, specialized tools
 └──► Sub-agent C           ← can spawn its own sub-agents (recursive)
          │
          └──► Sub-sub-agent
```

- Orchestrator primary tool: `spawn_agent`
- Orchestrator context: task graph + agent results only — never raw file contents or command output
- Sub-agents: do all heavy lifting, report structured results back
- Effective context: scales with number of sub-agents, not conversation length
- Sub-agents can be specialized by role: Planner, Reader, Executor, Synthesizer, Verifier

---

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Complete

---

## Phase 1 — Foundation
Goal: A working harness. The loop runs, tools execute, REPL accepts input.

- [x] 1.1  Project scaffold (structure, deps, pyproject.toml)
- [x] 1.2  Client abstraction layer — AnthropicClient + OllamaClient, normalized Response type
- [x] 1.3  Tool registry (register name + JSON schema + executor function)
- [x] 1.4  Agentic loop (send → parse tool_use → execute → append tool_result → repeat)
- [x] 1.5  Parallel tool execution (run independent tool_use blocks concurrently)
- [x] 1.6  Core tools: Bash, ReadFile, WriteFile
- [x] 1.7  Simple REPL (input → loop → print response)
- [x] 1.8  Basic system prompt builder

---

## Phase 2 — Sub-Agent System (Priority)
Goal: The orchestrator can spawn isolated child agents and receive structured results.
This is the primary context multiplication mechanism.

- [x] 2.1  Sub-agent class (own message history, own tool set, own system prompt)
- [x] 2.2  `spawn_agent` tool (orchestrator calls this to delegate a task)
- [x] 2.3  Sub-agent result serialization (structured return: result + summary + metadata)
- [x] 2.4  Recursive sub-agents (sub-agents can spawn their own children)
- [x] 2.5  Parallel sub-agent execution (spawn multiple agents concurrently)
- [x] 2.6  Sub-agent specialization via role presets (Planner, Reader, Executor, Synthesizer, Writer, Verifier)
- [x] 2.7  Sub-agent tool scoping (role presets filter tool access automatically)
- [x] 2.8  Sub-agent depth limit (MAX_DEPTH=4, enforced in make_spawn_tool)
- [x] 2.9  Agent tree visualization (depth-indented spawn events in REPL)

---

## Phase 3 — Context Management
Goal: The harness never hits a hard context limit. Context degrades gracefully and intelligently.

- [x] 3.1  Token counting (actual counts from API responses, updated every turn)
- [x] 3.2  Context budget monitor (bar in REPL, compaction at 80-85%)
- [x] 3.3  Large tool-result offloading (>3k chars → temp file + pointer)
- [x] 3.4  Rolling summarization (head+tail preserved, middle compacted via model)
- [x] 3.5  Conversation checkpointing (/save and /load commands)
- [ ] 3.6  Hierarchical context (summary layer + full detail on demand)
- [x] 3.7  Per-agent context budgets (orchestrator 90%, sub-agents 85%, compact at 80%)

---

## Phase 4 — Memory System
Goal: Knowledge persists across turns and across sessions.

- [x] 4.1  Session memory (model calls memory_write during conversation)
- [x] 4.2  Persistent memory (~/.forge/memory.json, survives restart)
- [x] 4.3  Memory injection (rebuilt into system prompt on every Agent.run())
- [x] 4.4  Memory distillation (/distill command + auto on /exit)
- [x] 4.5  Shared memory bus (all agents share one MemoryStore instance)

---

## Phase 5 — Advanced Orchestration
Goal: The orchestrator is genuinely intelligent about decomposition and delegation.

- [x] 5.1  Task planner tool (plan_task → structured JSON plan before acting)
- [x] 5.2  Dynamic re-planning (call plan_task again mid-execution to revise)
- [x] 5.3  Result verification (verify=True on spawn_agent auto-runs verifier)
- [x] 5.4  Retry and error recovery (auto-retry on [error] with failure context)
- [x] 5.5  Cross-agent token tracking (SessionStats shared across full tree)
- [x] 5.6  Result caching (ResultCache keyed by hash(task+context), session-scoped)

---

## Phase 6 — Polish
Goal: Production-quality UX and configurability.

- [x] 6.1  Streaming output (token-level, Qwen3 think-blocks rendered in dim italic)
- [x] 6.2  Rich terminal UI (agent tree, token bar, live prompt with agent/tool counts)
- [x] 6.3  Permission system (require_confirm list in config, interactive y/N prompt)
- [x] 6.4  Hooks (pre/post shell hooks per tool, configured in config.yaml)
- [x] 6.5  Config file (.forge.yaml / ~/.forge/config.yaml, all settings exposed)
- [x] 6.6  Conversation export (/export md | /export json)

---

## Deferred / Future
- Vector DB for semantic memory retrieval
- Web search tool
- Browser tool
- Multi-model routing (use cheaper model for simple sub-tasks)
- Persistent agent state between sessions

---

## File Structure

```
forge_harness/
├── harness/
│   ├── client.py           # Anthropic API wrapper
│   ├── loop.py             # Agentic loop
│   ├── agent.py            # Agent class (orchestrator + sub-agent share this)
│   ├── tools/
│   │   ├── registry.py     # Tool registry
│   │   ├── bash.py
│   │   ├── files.py
│   │   └── spawn_agent.py  # The sub-agent spawner tool
│   ├── context/
│   │   ├── manager.py      # Context window tracking + compaction
│   │   ├── summarizer.py   # Summarization strategies
│   │   └── checkpoint.py   # Save/load conversation state
│   ├── memory/
│   │   ├── session.py      # In-session memory
│   │   └── persistent.py   # Cross-session file memory
│   └── ui/
│       └── repl.py         # Terminal REPL
├── main.py
├── config.yaml
├── pyproject.toml
└── PLAN.md                 # This file
```
