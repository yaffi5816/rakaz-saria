"""Tool schemas for the model, and the dispatch back into shared code.

The schemas are generated from shared/tools.py rather than written again here.
Two copies of a tool definition is the defect this whole exercise is about: the
SDK version declares the same eight tools through @tool decorators, and if each
harness carried its own hand-written schema the comparison would be measuring my
typing rather than the two approaches.

Failures come back to the model as tool results marked is_error, not as
exceptions. A model that asks for a person who does not exist should be told
that in a sentence and allowed to try the right name; ending the run instead
turns a recoverable mistake into a failed scenario, and hides from the results
file the fact that the agent recovers at all.
"""

from __future__ import annotations

from shared import tools as shared_tools
from shared.tools import Context, ToolError
from API.harness import say

# The agent's whole surface. Named here as well as in the SDK's allowed_tools so
# both versions are restricted to the same eight.
ALLOWED = [tool.name for tool in shared_tools.TOOLS]


def schemas() -> list[dict]:
    """Render the shared registry as Anthropic tool definitions."""
    out = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in shared_tools.TOOLS
    ]
    say("tools.py", f"{len(out)} tool schemas sent: {', '.join(t['name'] for t in out)}")
    return out


def dispatch(ctx: Context, block) -> dict:
    """Run one tool_use block and build its tool_result.

    Input:
        ctx: the shared context holding the workspace, memory and calendar.
        block: the tool_use block from the model.

    Output: a tool_result dict, carrying is_error when the tool refused.
    """
    name = getattr(block, "name", None)
    args = getattr(block, "input", None) or {}
    try:
        text = shared_tools.run(ctx, name, args)
        say("tools.py", f"{name}({_brief(args)}) -> {_size(text)}")
        return {"type": "tool_result", "tool_use_id": block.id, "content": text}
    except ToolError as error:
        say("tools.py", f"{name} refused: {error}")
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(error),
            "is_error": True,
        }
    except Exception as error:  # noqa: BLE001 - the model gets the message, the run continues
        say("tools.py", f"{name} failed: {type(error).__name__}: {error}")
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"שגיאה בכלי {name}: {error}",
            "is_error": True,
        }


def _brief(args: dict) -> str:
    parts = []
    for key, value in list(args.items())[:3]:
        rendered = str(value)
        if len(rendered) > 40:
            rendered = rendered[:37] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _size(text: str) -> str:
    lines = text.count("\n") + 1 if text else 0
    return f"{lines} שורות, {len(text)} תווים"
