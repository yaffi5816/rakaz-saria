"""The eight tools, defined once and consumed by both harnesses.

This module is the reason the comparison in REPORT.md means anything. The
agent's capabilities - what it can see, what it can write, what it refuses -
live here, in plain functions over the shared domain. The API harness wraps them
in Anthropic tool schemas and dispatches by name; the SDK version wraps the same
functions in @tool and hands them to an in-process MCP server. Neither owns a
single business rule, so what the two runs differ by is harness and nothing else.

Every handler answers in text, never in JSON. A JSON envelope around rows that
are already pipe-separated is braces and quotation marks the model pays for on
every turn the result stays in the transcript, and it carries no information the
column order in the system prompt has not already given.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Callable

from shared import plan as planning
from shared import transcript
from shared.hebcal import CalendarService
from shared.hebcal import render as render_calendar
from shared.memory_store import MemoryStore
from shared.memory_store import render as render_memory
from shared.plan import DraftBook
from shared.schema import PERSON_COLUMNS, PROJECT_COLUMNS, TASK_COLUMNS, today


class ToolError(Exception):
    """A failure the model should see and recover from, not a crash.

    Both harnesses turn this into a tool result marked as an error, so a wrong
    identifier or an unknown person comes back as a sentence the model can act
    on rather than ending the run.
    """


@dataclass
class Context:
    """Everything the handlers share for one run."""

    client: Any
    memory: MemoryStore
    calendar: CalendarService
    book: DraftBook
    session: str = "s1"
    # What the person said this turn. A recorded decision is about the request,
    # not only about the words that happen to survive into a task title, so the
    # guard check reads this alongside the drafted titles.
    turn_text: str = ""
    tool_calls: list[str] = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


# ── handlers ─────────────────────────────────────────────────────────


def search_workspace(ctx: Context, args: dict) -> str:
    kind = args.get("kind", "tasks")
    limit = int(args.get("limit") or 12)
    if kind == "people":
        rows = ctx.client.people(args.get("query"))
        return "\n".join(
            f"{p.id}|{p.username}|{p.name}|{p.department}" for p in rows[:limit]
        ) or "(אין תוצאות)"
    if kind == "projects":
        rows = ctx.client.projects(args.get("query"))
        return "\n".join(p.line() for p in rows[:limit]) or "(אין תוצאות)"
    rows = ctx.client.tasks(
        query=args.get("query"),
        project_id=args.get("project_id"),
        owner=args.get("owner"),
        status=args.get("status"),
        late=args.get("late"),
        limit=limit,
    )
    return "\n".join(t.line() for t in rows) or "(אין תוצאות)"


def workload(ctx: Context, args: dict) -> str:
    names = args.get("names") or []
    if not names:
        raise ToolError("צריך רשימת שמות")
    days = int(args.get("days") or 30)
    rows = ctx.client.workload(names, days)
    lines = []
    for row in rows:
        if not row.get("found"):
            lines.append(f"{row['name']}|לא נמצא")
            continue
        lines.append(f"{row['name']}|{row['open']}|{row['late']}|{row['soon']}")
    return "name|open|late|due_in_" + str(days) + "d\n" + "\n".join(lines)


def calendar_check(ctx: Context, args: dict) -> str:
    dates = args.get("dates") or []
    if not dates:
        raise ToolError("צריך רשימת תאריכים בפורמט YYYY-MM-DD")
    return render_calendar(ctx.calendar.check(dates))


def history_stats(ctx: Context, args: dict) -> str:
    keyword = (args.get("keyword") or "").strip()
    if not keyword:
        raise ToolError("צריך מילת מפתח, למשל: שאלון")
    rows = ctx.client.completed_like(keyword)
    if not rows:
        return f"{keyword}: אין משימות שהושלמו להשוות אליהן"
    slips = []
    for row in rows:
        due = datetime.date.fromisoformat(row.due_date)
        done = datetime.date.fromisoformat(row.completed_at[:10])
        slips.append((done - due).days)
    average = sum(slips) / len(slips)
    detail = ",".join(f"{row.id}:{slip:+d}" for row, slip in zip(rows, slips))
    return (
        f"{keyword}|{len(rows)} הושלמו|חריגה ממוצעת {average:+.1f} ימים|"
        f"טווח {min(slips):+d}..{max(slips):+d}\n{detail}"
    )


def draft_plan(ctx: Context, args: dict) -> str:
    tasks = args.get("tasks") or []
    if not tasks:
        raise ToolError("צריך לפחות משימה אחת")
    for task in tasks:
        if not task.get("title"):
            raise ToolError("לכל משימה צריך title")
    draft = planning.build(
        ctx.book, ctx.calendar, ctx.memory, ctx.client, tasks, args.get("links") or [],
        subject_extra=ctx.turn_text,
    )
    return planning.render(draft)


def commit_plan(ctx: Context, args: dict) -> str:
    draft_id = args.get("draft_id")
    if not draft_id:
        raise ToolError("צריך draft_id")
    approve = args.get("approve")
    if approve is None:
        raise ToolError(
            "צריך approve: רשימת ה-ref שהמשתמשת אישרה, או [\"all\"] אם אישרה את כולן. "
            "אל תאשר בשמה."
        )
    if isinstance(approve, str):
        approve = [approve]
    draft, problem, created = planning.commit(ctx.book, ctx.client, draft_id, approve)
    if draft is None:
        raise ToolError(problem)
    if not created:
        raise ToolError(
            problem + "\nכדי להתקדם: לשנות את הטיוטה, לבטל את ההחלטה במפורש, "
            "או לאשר משימות אחרות."
        )
    ids = ",".join(str(i) for i in created)
    still = [task.ref for task in draft.tasks if task.status == planning.PENDING]
    tail = f"\n{problem}" if problem else ""
    waiting = f"\nממתינות לאישור: {', '.join(still)}" if still else ""
    return f"נוצרו {len(created)} משימות: {ids}{waiting}{tail}"


def read_transcript(ctx: Context, args: dict) -> str:
    name = (args.get("file") or "").strip()
    if not name:
        return "תמלולים זמינים: " + (", ".join(transcript.available()) or "(אין)")
    try:
        text, report = transcript.read(name)
    except (FileNotFoundError, ValueError) as error:
        raise ToolError(str(error)) from error
    header = (
        f"{report['file']} | {report['cues']} קטעים | "
        f"דוברים: {', '.join(report['speakers']) or 'לא מזוהים'} | "
        f"{report['chars_in']}->{report['chars_out']} תווים"
    )
    return f"{header}\n---\n{text}"


def remember(ctx: Context, args: dict) -> str:
    kind = args.get("kind", "convention")
    topic = (args.get("topic") or "").strip()
    rule = (args.get("rule") or "").strip()
    if not topic or not rule:
        raise ToolError("צריך topic ו-rule")
    supersedes = args.get("supersedes")
    if supersedes:
        record = ctx.memory.release(supersedes, rule, session=ctx.session)
        if record is None:
            raise ToolError(f"אין רשומה בשם {supersedes}")
        return f"בוטלה {supersedes}, נרשמה {record['id']} (הרשומה המקורית נשמרה)"
    guard = None
    if args.get("guard_date"):
        guard = {"type": "not_before", "date": args["guard_date"]}
    record = ctx.memory.remember(kind, topic, rule, guard, session=ctx.session)
    guard_note = f" (guard: לא לפני {record['guard']['date']})" if record.get("guard") else ""
    return f"נשמר {record['id']}: {kind}/{topic}{guard_note}"


def recall(ctx: Context, args: dict) -> str:
    topic = (args.get("topic") or "").strip()
    records = ctx.memory.relevant(topic, limit=int(args.get("limit") or 5)) if topic \
        else ctx.memory.active()
    return render_memory(records)


# ── registry ─────────────────────────────────────────────────────────


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[Context, dict], str]


TOOLS: list[Tool] = [
    Tool(
        name="search_workspace",
        description=(
            "חיפוש במרחב העבודה. kind=tasks מחזיר שורות " + TASK_COLUMNS +
            ", kind=projects מחזיר " + PROJECT_COLUMNS +
            ", kind=people מחזיר " + PERSON_COLUMNS +
            ". התוצאות מוגבלות למה שהמשתמש הפועל רשאי לראות."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["tasks", "projects", "people"]},
                "query": {"type": "string", "description": "טקסט חופשי לחיפוש בשם"},
                "project_id": {"type": "integer"},
                "owner": {"type": "string", "description": "שם או שם משתמש של האחראי"},
                "status": {"type": "string", "enum": ["new", "in_progress", "stuck", "done"]},
                "late": {"type": "boolean", "description": "רק משימות שעבר תאריך היעד שלהן"},
                "limit": {"type": "integer"},
            },
            "required": ["kind"],
        },
        handler=search_workspace,
    ),
    Tool(
        name="workload",
        description="כמה משימות פתוחות, באיחור ולפני יעד קרוב יש לכל אדם. name|open|late|soon",
        input_schema={
            "type": "object",
            "properties": {
                "names": {"type": "array", "items": {"type": "string"}},
                "days": {"type": "integer", "description": "חלון הימים ל-soon, ברירת מחדל 30"},
            },
            "required": ["names"],
        },
        handler=workload,
    ),
    Tool(
        name="calendar_check",
        description=(
            "בדיקת תאריכים מול לוח השנה העברי. מחזיר date|weekday|ok או blocked|חלופה|הערה. "
            "שבת ויום טוב חוסמים תאריך יעד; חנוכה או חול המועד הם הערה בלבד."
        ),
        input_schema={
            "type": "object",
            "properties": {"dates": {"type": "array", "items": {"type": "string"}}},
            "required": ["dates"],
        },
        handler=calendar_check,
    ),
    Tool(
        name="history_stats",
        description="כמה זמן באמת לקחו בעבר משימות שבשמן מילת המפתח, לפי תאריך יעד מול השלמה בפועל.",
        input_schema={
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"],
        },
        handler=history_stats,
    ),
    Tool(
        name="draft_plan",
        description=(
            "הכנת טיוטת משימות. לא כותב כלום למערכת. מחזיר את הטיוטה עם מזהה, "
            "אחרי בדיקת תאריכים מול לוח השנה ומול החלטות שמורות. "
            "links מקשר בין משימות: predecessor ו-successor הם ref של משימה בטיוטה (d1, d2) "
            "או מזהה של משימה קיימת."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref": {"type": "string", "description": "d1, d2 ..."},
                            "title": {"type": "string"},
                            "owners": {"type": "array", "items": {"type": "string"}},
                            "partners": {"type": "array", "items": {"type": "string"}},
                            "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                            "project_id": {"type": "integer"},
                            "group_name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["title"],
                    },
                },
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "predecessor": {"type": "string"},
                            "successor": {"type": "string"},
                        },
                        "required": ["predecessor", "successor"],
                    },
                },
            },
            "required": ["tasks"],
        },
        handler=draft_plan,
    ),
    Tool(
        name="commit_plan",
        description=(
            "כתיבת המשימות שאושרו, אחת-אחת. approve הוא רשימת ה-ref שהמשתמשת אישרה "
            "במפורש בתור הזה, למשל [\"d1\",\"d3\"], או [\"all\"] אם אמרה לאשר הכל. "
            "משימה שלא ברשימה לא נכתבת ונשארת ממתינה - אפשר לאשר אותה בהמשך. "
            "משימה שסותרת החלטה שמורה תיחסם גם אם אושרה."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "draft_id": {"type": "string"},
                "approve": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ref של כל משימה שאושרה, או [\"all\"]",
                },
            },
            "required": ["draft_id", "approve"],
        },
        handler=commit_plan,
    ),
    Tool(
        name="read_transcript",
        description=(
            "קריאת תמלול ישיבה מקובץ (vtt/txt/md/srt), כמו שטימס מייצר אחרי הקלטה. "
            "בלי file מחזיר את רשימת התמלולים הזמינים. חותמות הזמן מוסרות."
        ),
        input_schema={
            "type": "object",
            "properties": {"file": {"type": "string", "description": "שם הקובץ"}},
        },
        handler=read_transcript,
    ),
    Tool(
        name="remember",
        description=(
            "שמירת כלל עבודה (kind=convention) או החלטה ניהולית (kind=decision). "
            "אם ההחלטה קובעת שלא מתחילים משהו לפני תאריך, חובה למלא guard_date "
            "בפורמט YYYY-MM-DD כדי שהמערכת תוכל לאכוף אותה. "
            "לביטול רשומה קיימת: supersedes עם המזהה שלה - הרשומה המקורית לא נמחקת."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["convention", "decision"]},
                "topic": {"type": "string", "description": "הנושא, למשל: בדיקת שאלון"},
                "rule": {"type": "string", "description": "הכלל במילים של המשתמש"},
                "guard_date": {"type": "string"},
                "supersedes": {"type": "string"},
            },
            "required": ["topic", "rule"],
        },
        handler=remember,
    ),
    Tool(
        name="recall",
        description="שליפת כללים והחלטות שמורים לפי נושא. id|kind|topic|rule|guard",
        input_schema={
            "type": "object",
            "properties": {"topic": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["topic"],
        },
        handler=recall,
    ),
]

BY_NAME = {tool.name: tool for tool in TOOLS}


def run(ctx: Context, name: str, args: dict) -> str:
    """Execute one tool by name, recording that it ran."""
    tool = BY_NAME.get(name)
    if tool is None:
        raise ToolError(f"אין כלי בשם {name}")
    ctx.tool_calls.append(name)
    return tool.handler(ctx, args or {})


def system_prompt() -> str:
    """The agent's instructions, built when asked rather than at import.

    A function and not a constant because it names today's date, and a constant
    would freeze whatever the clock said when the module happened to be
    imported - which on a run that pins RAKAZ_TODAY afterwards is the wrong day,
    silently, in the one place the agent trusts most.
    """
    return _SYSTEM_TEMPLATE.format(
        today=today().isoformat(),
        TASK_COLUMNS=TASK_COLUMNS,
        PROJECT_COLUMNS=PROJECT_COLUMNS,
        PERSON_COLUMNS=PERSON_COLUMNS,
    )


_SYSTEM_TEMPLATE = """אתה "רכז" - עוזר ניהולי של מנהלת אגף במערכת ניהול משימות ופרויקטים.
התפקיד שלך הוא להפוך דברים שנאמרו - סיכום ישיבה, מייל, בקשה בעל פה - למשימות
מסודרות במערכת, ולהתריע כשמה שמבקשים לא ריאלי או סותר החלטה קודמת.

היום: {today}

איך אתה עובד:
1. אם הפנייה מצביעה על תמלול ישיבה - קורא אותו קודם עם read_transcript.
2. מבין על מה מדובר: מחפש את הפרויקט, את האנשים ואת המשימות הקיימות.
3. בודק תאריכים מול לוח השנה לפני שאתה מציע אותם.
4. מכין טיוטה. אף פעם לא כותב למערכת בלי אישור מפורש.
5. האישור הוא לכל משימה בנפרד: commit_plan מקבל ב-approve רק את ה-ref שהמשתמשת
   אישרה בתור הזה. אם אישרה שלוש מתוך ארבע - תכתוב שלוש, ותגיד שהרביעית ממתינה.

הכלים מחזירים שורות מופרדות ב-| בלי שמות שדות. סדר העמודות:
  משימות:  {TASK_COLUMNS}
  פרויקטים: {PROJECT_COLUMNS}
  אנשים:   {PERSON_COLUMNS}
"late5" בעמודת התאריך פירושו חמישה ימי איחור.

כללים:
- כשהמשתמש מספר לך כלל עבודה או החלטה - תשמור אותו ב-remember מיד.
- החלטה שקובעת שלא מתחילים משהו לפני תאריך חייבת guard_date, אחרת היא לא תיאכף.
- לפני שאתה מתכנן משימה בנושא כלשהו - בדוק ב-recall אם יש עליו כלל או החלטה
  שמורים, והחל אותם גם אם לא ביקשו ממך.
- בקשה שסותרת החלטה שמורה: תגיד מה ההחלטה, מתי נרשמה, ואל תעקוף אותה.
- לעולם אל תאשר משימה בשם המשתמשת. "תכיני" זה לא "תכתבי".
- אתה רואה רק מה שהמשתמשת רשאית לראות. אם משהו לא נמצא, אמור זאת.
- תשובות קצרות, בעברית, בלי לחזור על מה שהכלי כבר הדפיס."""
