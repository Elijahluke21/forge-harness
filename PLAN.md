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

## Phase 7 — Autonomy
Goal: Give it a goal and walk away. The run ends when code says it is done,
not when the model feels done, and it can never run forever.

- [x] 7.1  Run budget — one shared ceiling (tokens, spawns, tool calls, minutes) for the whole agent tree; BudgetExceeded unwinds the run and cancels in-flight work
- [x] 7.2  Acceptance checks — checks.yaml of commands; done = every one exits 0; loaded once so agents can't redefine "done" mid-run
- [x] 7.3  Code-driven plan executor — validated step graph (ids, roles, deps, cycles), parallel where independent, failures skip downstream steps, dependents get summaries + file paths instead of raw output
- [x] 7.4  Progress watchdog — no new passing check for N rounds → escalate planning to a stronger model once → stop with blocked.md
- [x] 7.5  `forge run "goal" --checks checks.yaml` — run folder with task.md, events.jsonl, per-round plans and step outputs, result.md/blocked.md; on_finish hook for phone/watch notifications
- [ ] 7.6  Real isolation — run each job in a container or separate OS user, plus a git worktree per run (workspace confinement today is a guard, not a sandbox)
- [ ] 7.7  Ship tool — trigger a fastlane lane in CI (keys stay in the CI secret store; agents never see them)
- [ ] 7.8  App templates — starter projects (SwiftUI + Supabase, fastlane, test targets, checks.yaml) so agents fill in features instead of inventing structure

## Review fixes (v0.2.0)
- ResultCache returned stale results after changes ("run the tests" stayed cached after a fix): only read-only roles are cached now, and any mutating tool call invalidates the cache
- Compaction could orphan a tool_use from its tool_result (API rejects the history); cut points now respect tool pairing and roles keep alternating
- Every memory entry was injected into every agent's prompt; now relevance-ranked, pinned-first, size-capped
- Sub-agents could write persistent memory (prompt-injection persistence); workers now get read-only memory
- Role-scoped sub-agents lost require_confirm (fresh registry); permissions and hooks now carry over
- Hooks replaced the whole environment (no PATH, so hook commands failed silently) and only ran for the orchestrator; now registry-level, async, env merged
- bash: 30s timeout killed builds and left child processes running; now process-group kill, configurable timeout up to 30 min, head+tail truncation
- Malformed tool-call JSON from local models crashed the run (non-streaming) or silently became `{}` (streaming); now reported back to the model as an error
- Anthropic client kept only the last text block of a response
- Qwen <think> blocks broke plan/distill JSON parsing and were re-sent in history every turn; now stripped
- Hitting max_iterations returned "[error]" and triggered a full retry with the same budget; now "[incomplete]", not retried
- Empty MemoryStore was falsy (`__len__`), so /memory and /distill said "no memory store" and distillation never ran on a fresh install
- /help ate "[path]" and "[md|json]" as Rich markup
- max_depth config was ignored; children ignored context_window config; unknown role names silently fell back to defaults
- New edit_file tool (exact snippet replace) — small models do far better than rewriting whole files
- main.py and cli.py duplicated all wiring; both now use harness/app.py
- Unknown config keys are reported instead of silently ignored

---

## Running unattended

```bash
pip install -e ".[dev]"
forge run "Add a settings screen with a dark-mode toggle" --checks checks.yaml --workspace ~/projects/my-app
```

`checks.yaml`:

```yaml
checks:
  - id: builds
    cmd: xcodebuild -scheme MyApp -destination 'generic/platform=iOS Simulator' build
    timeout: 900
    description: The app compiles
  - id: tests
    cmd: xcodebuild test -scheme MyApp -destination 'platform=iOS Simulator,name=iPhone 16'
    timeout: 1200
```

Exit code is 0 only when every check passes. Read `.forge/runs/<timestamp>/` for what happened.
Keep the tests the checks run somewhere agents can't quietly edit (a separate test target you own, or a read-only path) — otherwise "make the tests pass" can be satisfied by changing the tests.

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
forge-harness/
├── harness/
│   ├── app.py              # Builds orchestrator + registries from config
│   ├── cli.py              # `forge` (REPL) and `forge run` (autopilot)
│   ├── client.py           # Anthropic + Ollama clients, normalized Response
│   ├── loop.py             # Agentic loop
│   ├── agent.py            # Agent class + spawn_subagent / spawn_agent tool
│   ├── roles.py            # Role presets
│   ├── planning.py         # plan_task tool + generate_plan
│   ├── executor.py         # Plan validation + dependency-graph execution
│   ├── checks.py           # Acceptance checks
│   ├── autopilot.py        # Unattended run loop + watchdog
│   ├── budget.py           # RunBudget / BudgetExceeded
│   ├── session.py          # SessionStats, ResultCache
│   ├── text.py             # think-stripping, JSON extraction
│   ├── config.py
│   ├── prompt.py
│   ├── tools/              # registry, bash, files (+ edit_file), workspace
│   ├── context/            # manager (offload + compaction), checkpoint
│   ├── memory/             # store, tools, distill
│   └── ui/repl.py
├── tests/                  # pytest suite with a scripted fake model
├── main.py
├── config.yaml
└── PLAN.md
```
