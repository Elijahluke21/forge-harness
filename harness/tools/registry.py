import asyncio
import os
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from harness.budget import HarnessAbort


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    executor: Callable[..., Awaitable[str]]

    def to_api_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# Called before a tool executes: (name, inputs) → True to allow, False to deny
ConfirmFn = Callable[[str, dict], Awaitable[bool]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._require_confirm: set[str] = set()
        self._confirm_fn: ConfirmFn | None = None
        # Tool calls run in parallel; without a lock two y/N prompts interleave.
        self._confirm_lock = asyncio.Lock()
        self._hooks: list[dict] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, description: str, input_schema: dict, name: str | None = None):
        def decorator(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
            tool_name = name or fn.__name__
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                input_schema=input_schema,
                executor=fn,
            )
            return fn
        return decorator

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Permissions  (6.3)
    # ------------------------------------------------------------------

    def set_permissions(
        self,
        require_confirm: list[str],
        confirm_fn: ConfirmFn,
    ) -> None:
        """
        Mark tools that need user confirmation before execution.
        `confirm_fn(name, inputs)` must return True to allow, False to deny.
        """
        self._require_confirm = set(require_confirm)
        self._confirm_fn = confirm_fn

    # ------------------------------------------------------------------
    # Hooks  (6.4)
    # ------------------------------------------------------------------

    def set_hooks(self, hooks: list[dict]) -> None:
        """
        Shell commands run around tool calls. Each hook:
          {"on": "pre"|"post", "tool": "*"|"<name>", "cmd": "..."}
        The command gets FORGE_TOOL (and FORGE_RESULT_STATUS on post) in its env.
        """
        self._hooks = list(hooks or [])

    async def _run_hooks(self, timing: str, tool_name: str, extra_env: dict | None = None) -> None:
        for hook in self._hooks:
            if hook.get("on") != timing:
                continue
            if hook.get("tool", "*") not in ("*", tool_name):
                continue
            cmd = hook.get("cmd", "")
            if not cmd:
                continue
            # Merge with the real environment — replacing it drops PATH and the
            # hook silently fails to find any command.
            env = {**os.environ, "FORGE_TOOL": tool_name, **(extra_env or {})}
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd, env=env,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=int(hook.get("timeout", 30)))
            except (asyncio.TimeoutError, OSError):
                pass

    # ------------------------------------------------------------------
    # Lookup / introspection
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.to_api_dict() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def subset(self, names: list[str] | None) -> "ToolRegistry":
        """New registry with only `names` (None = all), sharing permissions."""
        scoped = ToolRegistry()
        for tool in self._tools.values():
            if names is None or tool.name in names:
                scoped.add(tool)
        scoped._require_confirm = self._require_confirm
        scoped._confirm_fn = self._confirm_fn
        scoped._confirm_lock = self._confirm_lock
        scoped._hooks = self._hooks
        return scoped

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, name: str, inputs: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"[error] unknown tool '{name}'. Available: {self.names()}"

        if not isinstance(inputs, dict):
            return f"[error] tool '{name}' expects a JSON object of arguments, got {type(inputs).__name__}"

        if name in self._require_confirm and self._confirm_fn is not None:
            async with self._confirm_lock:
                allowed = await self._confirm_fn(name, inputs)
            if not allowed:
                return f"[cancelled] user denied '{name}'"

        await self._run_hooks("pre", name)
        status = "ok"
        try:
            result = await tool.executor(**inputs)
            if isinstance(result, str) and result.lstrip().lower().startswith("[error]"):
                status = "error"
            return result
        except HarnessAbort:
            status = "abort"
            raise  # budget/abort signals must unwind the run, not become tool errors
        except TypeError as exc:
            status = "error"
            return f"[error] bad arguments for tool '{name}': {exc}"
        except Exception:
            status = "error"
            return f"[error] tool '{name}' raised an exception:\n{traceback.format_exc()}"
        finally:
            await self._run_hooks("post", name, {"FORGE_RESULT_STATUS": status})

