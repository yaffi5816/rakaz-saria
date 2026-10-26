"""A scripted stand-in for the model, so the suite needs no key and no network.

What this tests is the harness, not the model. The script below says "on turn 1
ask for these four tools with these arguments, then answer" - and the suite
checks that the loop stops on the right stop_reason, that all four tool results
come back in one user message, that the transcript on disk replays, that the
draft is not written until a commit arrives, and that the guard refuses the
commit that contradicts a decision.

Whether a real model would choose those calls is the scenario run's business,
and results/api.json answers it. This file answers the other question: given
those calls, does the plumbing do what it says.

Usage is invented but shaped like the real thing, so tokens.py is exercised and
the results file produced by a test run has the same fields as a measured one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FakeUsage:
    input_tokens: int = 900
    output_tokens: int = 120
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)


class Step:
    """One model reply: either a set of tool calls, or the final text."""

    def __init__(self, tools: list[tuple[str, Any]] | None = None, text: str = "") -> None:
        self.tools = tools or []
        self.text = text


class ScriptedMessages:
    def __init__(self, owner: "ScriptedClient") -> None:
        self.owner = owner

    def create(self, **kwargs) -> FakeResponse:
        self.owner.calls.append(kwargs)

        # The compaction summary is the one call that carries no tools.
        if "tools" not in kwargs:
            self.owner.summaries += 1
            return FakeResponse(
                content=[TextBlock(text="סיכום אוטומטי של השיחה עד כה (מדומה).")],
                stop_reason="end_turn",
                usage=FakeUsage(input_tokens=1200, output_tokens=90),
            )

        if not self.owner.steps:
            return FakeResponse(content=[TextBlock(text="(אין עוד תסריט)")],
                                stop_reason="end_turn")

        step = self.owner.steps.pop(0)
        if step.tools:
            blocks = []
            for name, args in step.tools:
                self.owner.counter += 1
                resolved = args(self.owner) if callable(args) else args
                blocks.append(ToolUseBlock(id=f"tu{self.owner.counter}", name=name,
                                           input=resolved))
            return FakeResponse(content=blocks, stop_reason="tool_use",
                                usage=FakeUsage(input_tokens=1500, output_tokens=200))
        return FakeResponse(content=[TextBlock(text=step.text)], stop_reason="end_turn")


class ScriptedClient:
    """Plays a fixed list of steps, one per model call."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.messages = ScriptedMessages(self)
        self.calls: list[dict] = []
        self.counter = 0
        self.summaries = 0


# ── the script for scenario.json ─────────────────────────────────────

MEETING_TASKS = [
    {"ref": "d1", "title": "הכנת סביבת בדיקות להזנה", "owners": ["דנה שפירא"],
     "due_date": "2026-11-26", "project_id": 2, "group_name": "הכנה"},
    {"ref": "d2", "title": "כתיבת נוהל עבודה בשטח", "owners": ["רחל כהן"],
     "due_date": "2026-11-21", "project_id": 2, "group_name": "שדה"},
    {"ref": "d3", "title": "מצגת סטטוס להנהלה", "owners": ["איתי גולן"],
     "due_date": "2026-12-05", "project_id": 2, "group_name": "הכנה"},
    {"ref": "d4", "title": "תיאום הדרכה מול לשכות המחוז", "owners": ["יוסי ברק"],
     "due_date": "2026-11-26", "project_id": 2, "group_name": "שדה"},
]

REASSIGNED = [dict(task, owners=["איתי גולן"]) if task["ref"] == "d2" else dict(task)
              for task in MEETING_TASKS]


def script() -> list[Step]:
    """The eight turns of scenario.json, as a model that behaves would play them."""
    return [
        # turn 1 - look, check the dates, draft, do not write
        Step(tools=[("search_workspace", {"kind": "projects", "query": "בריאות"}),
                    ("search_workspace", {"kind": "tasks", "project_id": 2, "limit": 8})]),
        Step(tools=[("calendar_check",
                     {"dates": ["2026-11-21", "2026-12-05", "2026-11-26"]})]),
        Step(tools=[("draft_plan", {"tasks": MEETING_TASKS,
                                    "links": [{"predecessor": "202", "successor": "d1"}]})]),
        Step(text="הכנתי טיוטה של ארבע משימות. שני תאריכים נפלו בשבת והוזזו. לא כתבתי כלום."),

        # turn 2 - check the load, redraft with a different owner
        Step(tools=[("workload", {"names": ["רחל כהן", "יוסי ברק", "איתי גולן"]})]),
        Step(tools=[("draft_plan", {"tasks": REASSIGNED,
                                    "links": [{"predecessor": "202", "successor": "d1"}]})]),
        Step(text="רחל עמוסה, העברתי את נוהל העבודה לאיתי."),

        # turn 3 - approval
        Step(tools=[("commit_plan", {"draft_id": "draft2"})]),
        Step(text="נכתב למערכת."),

        # turn 4 - a convention
        Step(tools=[("remember", {"kind": "convention", "topic": "בדיקת שאלון",
                                  "rule": "נועה לוי אחראית ואבי מזרחי שותף"})]),
        Step(text="נשמר."),

        # turn 5 - a management decision, with the guard that enforces it
        Step(tools=[("remember", {"kind": "decision", "topic": "שלב ב",
                                  "rule": "לא פותחים את משימות שלב ב' לפני ינואר 2027",
                                  "guard_date": "2027-01-01"})]),
        Step(text="נרשם."),

        # turn 6 - session 2: the convention applies without being asked
        Step(tools=[("recall", {"topic": "בדיקת שאלון"})]),
        Step(tools=[("draft_plan", {"tasks": [{
            "ref": "d1", "title": "בדיקת שאלון מודול שינה", "owners": ["נועה לוי"],
            "partners": ["אבי מזרחי"], "due_date": "2026-11-26", "project_id": 2,
            "group_name": "הכנה"}]})]),
        Step(tools=[("commit_plan", {"draft_id": "draft3"})]),
        Step(text="נוצרה משימה אחת, עם נועה אחראית ואבי שותף לפי הכלל השמור."),

        # turn 7 - the contradiction: drafted, refused on commit
        Step(tools=[("draft_plan", {"tasks": [
            {"ref": "d1", "title": "הכנת מדגם", "owners": ["רחל כהן"],
             "due_date": "2026-12-20", "project_id": 1},
            {"ref": "d2", "title": "הדרכת סוקרים", "owners": ["איתי גולן"],
             "due_date": "2026-12-20", "project_id": 1}]})]),
        Step(tools=[("commit_plan", {"draft_id": "draft4"})]),
        Step(text="לא כתבתי: ההחלטה מ-5 בנובמבר אוסרת לפתוח את שלב ב' לפני ינואר 2027."),

        # turn 8 - the risk question
        Step(tools=[("search_workspace", {"kind": "tasks", "project_id": 1, "late": True})]),
        Step(text="משימה 101 באיחור ומושכת אחריה את כל שרשרת השדה."),
    ]


def client_factory() -> Callable[[], ScriptedClient]:
    """One scripted client shared by both sessions.

    Each session builds its own agent, and a fresh client per session would
    rewind the script to turn 1. The agent instances are separate - which is
    the point of the memory checks - but the model they talk to is one.
    """
    shared = ScriptedClient(script())
    return lambda: shared
