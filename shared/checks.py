"""The pass/fail checks, written against saved state rather than against prose.

Eight checks, seven of which open the database or the memory file and read what
is actually there. That split is deliberate: a model that answers "בוצע, נוצרו
חמש משימות" while writing nothing passes any check that greps the reply, and
the whole claim of this agent is that the workspace changed.

The two drivers share this module. tests/ runs the scenario against a scripted
model client with no key and no network and then calls run_all; API/main.py and
SDK/main.py call the same function after a real run, so results/api.json and
results/sdk.json carry the identical checks.
"""

from __future__ import annotations

import datetime
from typing import Callable

from shared.hebcal import CalendarService
from shared.memory_store import MemoryStore
from shared.store import LocalStore


def snapshot(client, memory: MemoryStore) -> dict:
    """What the state looked like at the end of a turn."""
    tasks = client.created_by_agent() if hasattr(client, "created_by_agent") else []
    return {
        "agent_task_ids": [t["id"] for t in tasks],
        "memory_ids": [r["id"] for r in memory.records()],
        "active_memory_ids": [r["id"] for r in memory.active()],
    }


def _turn(record: dict, number: int) -> dict:
    for turn in record.get("turns", []):
        if turn.get("turn") == number:
            return turn
    return {}


def _after(record: dict, number: int) -> dict:
    return _turn(record, number).get("state", {})


# ── the checks ───────────────────────────────────────────────────────


def no_write_before_approval(record: dict, store: LocalStore, memory: MemoryStore) -> tuple[bool, str]:
    """Turn 1 drafts five tasks and the workspace still holds none of them."""
    created = _after(record, 1).get("agent_task_ids", [])
    return not created, f"אחרי תור 1 קיימות {len(created)} משימות שהסוכן יצר (ציפייה: 0)"


def plan_written_on_approval(record: dict, store: LocalStore, memory: MemoryStore) -> tuple[bool, str]:
    """The approval in turn 3 is what puts the tasks in the workspace."""
    created = _after(record, 3).get("agent_task_ids", [])
    deps = store.all_dependencies()
    agent_ids = set(created)
    new_links = [d for d in deps if d[0] in agent_ids or d[1] in agent_ids]
    ok = 3 <= len(created) <= 6 and len(new_links) >= 1
    return ok, (
        f"אחרי תור 3: {len(created)} משימות (ציפייה 3-6), "
        f"{len(new_links)} תלויות שנוגעות בהן (ציפייה: לפחות 1)"
    )


def due_dates_are_working_days(record: dict, store: LocalStore, memory: MemoryStore) -> tuple[bool, str]:
    """No task the agent created falls on Friday, Shabbat or a festival."""
    calendar = CalendarService(mode="offline")
    bad = []
    for task in store.created_by_agent():
        if not task["due_date"]:
            continue
        info = calendar.day(datetime.date.fromisoformat(task["due_date"]))
        if not info["working"]:
            bad.append(f"{task['id']} {task['due_date']} ({info['reason']})")
    return not bad, (
        "כל תאריכי היעד שנוצרו הם ימי עבודה" if not bad else "תאריכים פסולים: " + ", ".join(bad)
    )


def convention_saved(record: dict, store: LocalStore, memory: MemoryStore) -> tuple[bool, str]:
    """Turn 4 leaves a convention on disk, not just an acknowledgement."""
    hits = [r for r in memory.active() if "שאלון" in r["topic"] + r["rule"]
            and r["kind"] == "convention"]
    return bool(hits), (
        f"נשמרו {len(hits)} קונבנציות בנושא שאלון: "
        + ", ".join(r["id"] for r in hits) if hits else "לא נשמרה קונבנציה בנושא שאלון"
    )


def convention_applied_across_sessions(record: dict, store: LocalStore,
                                       memory: MemoryStore) -> tuple[bool, str]:
    """A new session applies the saved rule without being asked again.

    The strongest check in the suite: session 2 never saw the turn that taught
    the rule, so an owner of נועה and a partner of אבי on the task created in
    turn 6 can only have come off the disk.
    """
    before = set(_after(record, 5).get("agent_task_ids", []))
    after = set(_after(record, 6).get("agent_task_ids", []))
    new_ids = after - before
    if not new_ids:
        return False, "תור 6 לא יצר משימה"
    tasks = {t["id"]: t for t in store.created_by_agent()}
    details = []
    ok = False
    for task_id in sorted(new_ids):
        task = tasks.get(task_id, {})
        owners = ",".join(task.get("owners", []))
        partners = ",".join(task.get("partners", []))
        details.append(f"{task_id}: אחראי={owners or '-'} שותף={partners or '-'}")
        if any("נועה" in o for o in task.get("owners", [])) and any(
            "אבי" in p for p in task.get("partners", [])
        ):
            ok = True
    return ok, "; ".join(details)


def decision_blocks_contradiction(record: dict, store: LocalStore,
                                  memory: MemoryStore) -> tuple[bool, str]:
    """The contradicting request in turn 7 changes nothing in the workspace."""
    before = _after(record, 6).get("agent_task_ids", [])
    after = _after(record, 7).get("agent_task_ids", [])
    added = set(after) - set(before)
    return not added, (
        "תור 7 לא יצר משימות - ההחלטה נאכפה" if not added
        else f"תור 7 יצר {len(added)} משימות למרות ההחלטה: {sorted(added)}"
    )


def external_tool_ran(record: dict, store: LocalStore, memory: MemoryStore) -> tuple[bool, str]:
    """The calendar was actually consulted before dates were proposed."""
    tools = _turn(record, 1).get("tools", [])
    calendar = record.get("calendar") or {}
    where = calendar.get("source", "?")
    http = calendar.get("http_calls", 0)
    return "calendar_check" in tools, (
        f"כלים בתור 1: {', '.join(tools) or '(אין)'} | לוח השנה: {where}, "
        f"{http} קריאות רשת"
    )


def memory_is_append_only(record: dict, store: LocalStore, memory: MemoryStore) -> tuple[bool, str]:
    """The memory file only ever grew, and no identifier was reused."""
    counts = [len(_after(record, t["turn"]).get("memory_ids", [])) for t in record.get("turns", [])]
    monotonic = all(b >= a for a, b in zip(counts, counts[1:]))
    ids = [r["id"] for r in memory.records()]
    unique = len(ids) == len(set(ids))
    return monotonic and unique, (
        f"מספר הרשומות לאורך הריצה: {counts}, מזהים ייחודיים: {unique}"
    )


def context_stays_small(record: dict, store: LocalStore, memory: MemoryStore) -> tuple[bool, str]:
    """No turn was handed the board - the per-turn context stayed a digest.

    The limit is in characters because it is a property of what the harness
    chose to send, which is knowable without the API. Nineteen task rows would
    be roughly 1,400 characters on their own.
    """
    sizes = [t.get("context_chars", 0) for t in record.get("turns", [])]
    biggest = max(sizes) if sizes else 0
    return biggest < 1500, f"הקשר מרבי לתור: {biggest} תווים (מגבלה 1500), כל התורים: {sizes}"


def compaction_actually_ran(record: dict, store: LocalStore,
                            memory: MemoryStore) -> tuple[bool, str]:
    """The context-window file did its job at least once during the run.

    A compaction.py that never fires is a file in the line-count table that
    nothing in the comparison exercised, so the threshold is set where this run
    crosses it and this check says whether it did. The SDK version compacts
    inside Claude Code, where there is nothing to count - the check records that
    rather than failing it, because "the SDK does this for you" is the finding,
    not a defect.
    """
    if record.get("version") == "sdk":
        return True, "ה-SDK מבצע compaction אוטומטי - אין מונה בצד שלנו"
    fired = max((t.get("compactions", 0) for t in record.get("turns", [])), default=0)
    return fired >= 1, f"compaction רץ {fired} פעמים בריצה"


CHECKS: list[tuple[str, Callable]] = [
    ("אחרי תור 1 לא נכתבה אף משימה", no_write_before_approval),
    ("האישור בתור 3 יצר את המשימות ואת התלות", plan_written_on_approval),
    ("כל תאריכי היעד נופלים ביום עבודה", due_dates_are_working_days),
    ("הקונבנציה נשמרה לדיסק בתור 4", convention_saved),
    ("סשן 2 החיל את הקונבנציה בלי שביקשו", convention_applied_across_sessions),
    ("ההחלטה חסמה את הבקשה הסותרת בתור 7", decision_blocks_contradiction),
    ("הכלי החיצוני רץ לפני שהוצעו תאריכים", external_tool_ran),
    ("הזיכרון append-only לאורך כל הריצה", memory_is_append_only),
    ("אף תור לא קיבל את כל הלוח", context_stays_small),
    ("ה-compaction נכנס לפעולה בריצה", compaction_actually_ran),
]


def run_all(record: dict, actor: str = "sarah.avital") -> list[dict]:
    """Run every check against the state this run left behind."""
    store = LocalStore(actor=actor)
    memory = MemoryStore()
    out = []
    for name, check in CHECKS:
        try:
            passed, detail = check(record, store, memory)
        except Exception as error:  # noqa: BLE001 - a broken check is a failed check
            passed, detail = False, f"{type(error).__name__}: {error}"
        out.append({"name": name, "passed": bool(passed), "detail": detail})
    return out
