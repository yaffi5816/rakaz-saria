"""The same eight tools, declared for the SDK instead of for the raw API.

Every handler here forwards straight into shared/tools.py. Nothing in this file
decides anything about the domain - that is the condition that makes REPORT.md
a comparison of harnesses rather than of two different agents.

What the file does own is the shape the SDK expects: an async function taking a
single dict, returning {"content": [...]}, with failures marked is_error rather
than raised. That is the counterpart of API/harness/tools.py, which builds
Anthropic tool schemas and turns a ToolError into a tool_result block.

The wrappers are generated from the shared registry rather than written out
eight times. Hand-writing them would have been eight chances to let a
description drift away from the one the API version sends, and the whole
comparison rests on both versions offering the model the identical surface.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from shared import tools as shared_tools
from shared.tools import Context, ToolError

# The tool handlers run in this process, so they reach the workspace the same
# way the API version's do. SDK/agent.py puts the live context here before the
# first query; a tool that fires before that is a bug worth crashing on.
CONTEXT: Context | None = None


def use(context: Context) -> None:
    global CONTEXT
    CONTEXT = context


def _wrap(spec: shared_tools.Tool):
    """Turn one shared tool into an SDK tool."""

    @tool(spec.name, spec.description, spec.input_schema)
    async def handler(args: dict[str, Any], _name: str = spec.name) -> dict[str, Any]:
        # replaces: harness/tools.py  (dispatch into the shared registry)
        try:
            text = shared_tools.run(CONTEXT, _name, args or {})
            return {"content": [{"type": "text", "text": text}]}
        except ToolError as error:
            # replaces: harness/tools.py  (a refusal the model can recover from)
            return {"content": [{"type": "text", "text": str(error)}], "is_error": True}

    return handler


TOOLS = [_wrap(spec) for spec in shared_tools.TOOLS]

# The server's own name is not what the model sees. Tool names are built from
# the key this server is given in ClaudeAgentOptions.mcp_servers, so the names
# below come out as mcp__rakaz__<tool> because agent.py files it under "rakaz".
SERVER = create_sdk_mcp_server(name="rakaz", version="1.0.0", tools=TOOLS)

# What allowed_tools has to say, spelled the way the SDK names them.
NAMES = [f"mcp__rakaz__{spec.name}" for spec in shared_tools.TOOLS]
