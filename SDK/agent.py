"""The same agent, defined rather than built.

Every line below that stands in for one of the eight files in API/harness/
carries a `# replaces:` comment naming it, and shared/loc.py counts those
markers to produce the right-hand column of REPORT.md's size table.

Three of the eight have no marker at all, and that is the most interesting
result in the exercise rather than an omission:

  compaction.py - Claude Code compacts when the window fills. There is no
                  option to set, so there is no line to mark.
  caching.py    - the cache breakpoints are placed by the harness. Same.
  context.py    - and this one is not free. The API version chooses, before
                  each turn, which saved rules to put in front of the model.
                  Here there is no pre-turn hook to do that from, so the model
                  has to go and fetch them with recall. It usually does. "Usually"
                  is the difference, and it is measured rather than argued:
                  the convention check in results/sdk.json either passes or
                  does not.

What is deliberately not default: the system prompt is mine and not Claude
Code's, built-in tools are removed entirely, and strict_mcp_config keeps the
machine's own MCP configuration out of the run - otherwise this would be a
comparison between my harness and whatever happens to be installed.

`resume` is wired but the scenario never passes it. It is the SDK's answer to
history.py and belongs in the table as such; using it for session 2 would hand
that session the transcript of session 1, and then "the agent applied a rule it
was taught last week" would no longer be evidence that anything came off the
disk. Within a session the conversation is the client itself, which needs no
resume at all; across a process boundary - a real second day - this is the line
that would carry it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,

)

from SDK import tools as sdk_tools
from shared.client import build_client
from shared.hebcal import CalendarService
from shared.memory_store import MemoryStore
from shared.plan import DraftBook
from shared.tools import Context, system_prompt

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("RAKAZ_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("RAKAZ_EFFORT", "medium")


def options(resume: str | None = None) -> ClaudeAgentOptions:
    """The whole configuration of the SDK agent."""
    return ClaudeAgentOptions(
        model=MODEL,
        effort=EFFORT,
        system_prompt=system_prompt(),
        mcp_servers={"rakaz": sdk_tools.SERVER},  # replaces: harness/tools.py
        allowed_tools=sdk_tools.NAMES,            # replaces: harness/tools.py
        # allowed_tools only auto-approves; this is what actually narrows the
        # surface to the eight. Without it the model also holds Read, Bash,
        # Edit and the rest, which is a different agent from the API version's.
        tools=[],
        skills=[],
        permission_mode="dontAsk",
        strict_mcp_config=True,
        setting_sources=["project"],              # replaces: harness/memory.py
        cwd=str(ROOT),                            # replaces: harness/memory.py
        resume=resume,                            # replaces: harness/history.py
        max_turns=12,
        # Tool search is on by default and defers tool schemas until the model
        # asks for them. That is a real token saving and also a difference the
        # API version does not have, so it is turned off: both versions send
        # all eight schemas, and the comparison stays about the harness.
        env={"ENABLE_TOOL_SEARCH": "false"},
    )


@dataclass
class TurnResult:
    turn: int
    text: str = ""
    tools: list[str] = field(default_factory=list)
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    session_id: str | None = None


class Rakaz:
    """One SDK session. Turns run on one client, which is the conversation."""

    def __init__(self, session: str = "s1") -> None:
        self.session = session
        self.workspace = build_client()
        self.memory = MemoryStore()
        self.ctx = Context(
            client=self.workspace,
            memory=self.memory,
            calendar=CalendarService(),
            book=DraftBook(),
            session=session,
        )
        sdk_tools.use(self.ctx)
        self.client: ClaudeSDKClient | None = None
        self.session_id: str | None = None
        self._running_cost = 0.0

    async def __aenter__(self) -> "Rakaz":
        # replaces: harness/loop.py  (the client owns the conversation)
        self.client = ClaudeSDKClient(options=options())
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self.client is not None:
            await self.client.__aexit__(*exc)

    async def ask(self, number: int, text: str) -> TurnResult:
        """One turn. The SDK drives the tool loop; this only reads the result."""
        import time

        assert self.client is not None
        self.ctx.turn_text = text
        before = len(self.ctx.tool_calls)
        started = time.perf_counter()
        result = TurnResult(turn=number)

        await self.client.query(text)              # replaces: harness/loop.py
        async for message in self.client.receive_response():  # replaces: harness/loop.py
            if isinstance(message, SystemMessage) and message.subtype == "init":
                self.session_id = message.data.get("session_id") or self.session_id
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result.text += block.text
            elif isinstance(message, ResultMessage):
                usage = message.usage or {}        # replaces: harness/tokens.py
                result.input_tokens = usage.get("input_tokens", 0) or 0
                result.output_tokens = usage.get("output_tokens", 0) or 0
                result.cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
                result.cache_write_tokens = usage.get("cache_creation_input_tokens", 0) or 0
                result.model_calls = message.num_turns or 0
                result.session_id = message.session_id
                self.session_id = message.session_id or self.session_id
                # total_cost_usd is a running total for the whole session, not
                # this turn's cost, so the turn's figure is the difference.
                total = message.total_cost_usd or 0.0   # replaces: harness/tokens.py
                result.cost_usd = max(0.0, total - self._running_cost)
                self._running_cost = max(self._running_cost, total)

        result.seconds = time.perf_counter() - started
        result.tools = [name for name in self.ctx.tool_calls[before:]]
        return result
