"""The agent itself: the eight harness files, wired together.

Everything this file does is assembly. The model call lives here because it is
the one thing none of the eight files owns - loop.py drives it, tokens.py meters
it, caching.py shapes its prefix, but somebody has to construct it, and that is
the job of the thing that owns the session.

Request parameters worth naming, because each is a decision and not a default:

thinking is not passed at all. On Claude Opus 5.5 thinking cannot be switched
off - {"type": "disabled"} and budget_tokens are both rejected - and omitting
the parameter runs adaptive thinking, which is what an agent choosing between
eight tools wants. Depth is controlled with effort instead.

effort is pinned to medium, which is this model's default. Pinning the default
costs nothing in cache terms and makes the comparison honest: the SDK version
runs the same model at the same effort, and a number that differs between the
two runs differs because of the harness.

tool_choice is left alone. Forcing a tool is rejected on this model, and the
whole point of the first turn is that the agent decides for itself whether it
needs to look something up before drafting.
"""

from __future__ import annotations

import os

from API.harness import say
from API.harness import caching, compaction, context, loop, tools as harness_tools
from API.harness.history import History
from API.harness.memory import Memory
from API.harness.tokens import MODEL, Meter
from shared.client import build_client
from shared.hebcal import CalendarService
from shared.plan import DraftBook
from shared.tools import Context, system_prompt

MAX_TOKENS = 16000
EFFORT = os.environ.get("RAKAZ_EFFORT", "medium")


class Rakaz:
    """One session of the agent. A second session is a second instance."""

    def __init__(self, session: str = "s1", client_factory=None) -> None:
        self.session = session
        self.workspace = build_client()
        self.memory = Memory()
        self.calendar = CalendarService()
        self.history = History(session)
        self.meter = Meter()
        self.ctx = Context(
            client=self.workspace,
            memory=self.memory.store,
            calendar=self.calendar,
            book=DraftBook(),
            session=session,
        )
        self.system = caching.system_blocks(system_prompt(), self.memory.profile())
        self.tools = harness_tools.schemas()
        self.client = (client_factory or _default_client)()
        self.compactions = 0
        say("agent.py", f"session {session} ready, model {MODEL}, effort {EFFORT}")

    # ── the model call ───────────────────────────────────────────────

    def _call(self, messages: list[dict]):
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": EFFORT},
            system=self.system,
            tools=self.tools,
            messages=messages,
        )
        self.meter.record(response.usage)
        caching.report(response.usage)
        return response

    def _summarise(self, prompt: str) -> str:
        """The one model call that is not part of a turn: the compaction summary."""
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1000,
            output_config={"effort": EFFORT},
            messages=[{"role": "user", "content": prompt}],
        )
        self.meter.record(response.usage)
        return "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

    # ── one turn ─────────────────────────────────────────────────────

    def ask(self, number: int, text: str) -> dict:
        usage = self.meter.start_turn(number)
        before = len(self.ctx.tool_calls)
        self.ctx.turn_text = text

        messages = self.history.replay()
        compacted = compaction.maybe_compact(messages, self.history, self._summarise)
        if compacted is not messages:
            self.compactions += 1
        messages = compacted

        turn_message = context.build_turn(self.workspace, self.memory, text)
        self.history.append(turn_message)
        messages.append(turn_message)

        answer, cut_off = loop.run_turn(self._call, self.ctx, self.history, messages)
        usage.tools = self.ctx.tool_calls[before:]
        self.meter.end_turn()
        return {
            "text": answer,
            "cut_off": cut_off,
            "usage": usage.as_dict(),
            "context_chars": len(turn_message["content"]),
            "compactions": self.compactions,
        }


CREDENTIALS_HELP = """גרסת ה-API לא מצאה אישור לדבר עם המודל.

שתי דרכים, ושתיהן בטרמינל בלבד - לעולם לא בקוד ולא ב-git:

  1. מפתח מ-Claude Console (מה שהמשימה מבקשת):
         export ANTHROPIC_API_KEY=sk-ant-...        (bash)
         $env:ANTHROPIC_API_KEY = "sk-ant-..."      (PowerShell)

  2. פרופיל התחברות, אם מותקן אצלך ה-CLI של Anthropic:
         ant auth login
     ואז להריץ בלי להגדיר מפתח כלל.

גרסת ה-SDK לא צריכה אף אחת מהשתיים - היא רצה על ה-login של Claude Code."""


def _default_client():
    """Build the real API client, imported here so the suite never needs it."""
    import anthropic

    return anthropic.Anthropic()


def check_credentials() -> None:
    """Fail before anything is deleted, rather than after.

    Purpose: a run starts by wiping the workspace, the memory and the
    transcripts, and only then builds a client. Without this, a missing key
    costs the previous run's state and gives a traceback in exchange. The probe
    is messages.count_tokens, which is free and asks the server the one
    question that matters: does this machine have an identity.

    Also deliberately NOT a check that ANTHROPIC_API_KEY is set. A login
    profile is a perfectly good credential, and a guard that insists on the
    environment variable would refuse a setup that works.
    """
    try:
        _default_client().messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": "בדיקה"}]
        )
    except Exception as error:  # noqa: BLE001 - every auth failure reads the same here
        raise SystemExit(f"{CREDENTIALS_HELP}\n\n({type(error).__name__}: {error})") from error
