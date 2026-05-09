import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


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
    # Lookup / introspection
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.to_api_dict() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, name: str, inputs: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"[error] unknown tool '{name}'. Available: {self.names()}"

        # Permission check
        if name in self._require_confirm and self._confirm_fn is not None:
            allowed = await self._confirm_fn(name, inputs)
            if not allowed:
                return f"[cancelled] user denied '{name}'"

        try:
            return await tool.executor(**inputs)
        except TypeError as exc:
            return f"[error] bad arguments for tool '{name}': {exc}"
        except Exception:
            return f"[error] tool '{name}' raised an exception:\n{traceback.format_exc()}"
