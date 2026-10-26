"""Drafting work, and the gate between drafting it and writing it down.

Nothing this agent does reaches the workspace without passing through here
twice: once as a draft the person reads, and once as an explicit approval
naming the tasks to write. An agent that creates tasks the moment it
understands a sentence is an agent nobody lets near a real board.

Approval is per task, not per plan. A meeting produces four items and a
manager agrees with three of them; a single "approve" button makes her accept
the fourth to get the three, and the fourth is the one that was wrong. So the
unit of approval is the task: each carries its own status, a draft can be
committed more than once as items are agreed, and the ones never approved stay
pending forever and write nothing.

Blockers are per task for the same reason. A recorded decision that forbids one
item should not hold the other three hostage - and an approval naming a blocked
task is refused in code, not by the model behaving well. That is what makes the
contradiction check deterministic.
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.memory_store import STOPWORDS

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "state" / "drafts.json"

PENDING = "pending"
APPROVED = "approved"
BLOCKED = "blocked"


@dataclass
class DraftTask:
    ref: str
    title: str
    owners: list[str] = field(default_factory=list)
    partners: list[str] = field(default_factory=list)
    due_date: str | None = None
    project_id: int | None = None
    group_name: str | None = None
    description: str | None = None
    moved_from: str | None = None
    notes: list[str] = field(default_factory=list)
    blocker: str | None = None
    status: str = PENDING
    created_id: int | None = None


@dataclass
class Draft:
    id: str
    tasks: list[DraftTask]
    links: list[dict]
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""

    @property
    def blockers(self) -> list[str]:
        return [task.blocker for task in self.tasks if task.blocker]

    @property
    def committed(self) -> list[int]:
        return [task.created_id for task in self.tasks if task.created_id]

    def by_ref(self, ref: str) -> DraftTask | None:
        return next((task for task in self.tasks if task.ref == str(ref)), None)


def _topic_words(text: str) -> set[str]:
    return {w for w in re.split(r"\W+", text) if len(w) > 1 and w not in STOPWORDS}


def guard_hit(record: dict, task: DraftTask, subject_extra: str = "") -> str | None:
    """Decide whether a recorded decision forbids this drafted task.

    Input:
        record: an active memory record carrying a guard.
        task: one drafted task, with its title, description and date.
        subject_extra: what the person asked for this turn. A decision about
            "שלב ב" has to catch a request to open שלב ב even when the titles
            that came out of it say "הכנת מדגם" and nothing else.

    Output: the sentence to show the person, or None when the decision is
    unrelated to this task or the task respects it.
    """
    guard = record.get("guard") or {}
    if guard.get("type") != "not_before" or not guard.get("date"):
        return None
    subject = f"{task.title} {task.description or ''} {subject_extra}"
    topic_words = _topic_words(record.get("topic", ""))
    if not topic_words:
        return None
    # The topic's longest word has to appear, not just any word of it. Hebrew
    # topics carry one-letter particles ("שלב ב"), and matching on those would
    # make a decision about one phase apply to half the board.
    key = sorted(topic_words, key=lambda word: (-len(word), word))[0]
    if key not in _topic_words(subject):
        return None
    if not task.due_date or task.due_date >= guard["date"]:
        return None
    return (
        f"{record['id']}: \"{record['rule']}\" - והמשימה \"{task.title}\" "
        f"מתוארכת ל-{task.due_date}, לפני {guard['date']}"
    )


class DraftBook:
    """Drafts on disk, so a draft survives between turns and between processes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DRAFTS
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _all(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def put(self, draft: Draft) -> None:
        data = self._all()
        data[draft.id] = {
            "id": draft.id,
            "tasks": [asdict(task) for task in draft.tasks],
            "links": draft.links,
            "warnings": draft.warnings,
            "created_at": draft.created_at,
        }
        self._save(data)

    def get(self, draft_id: str) -> Draft | None:
        raw = self._all().get(draft_id)
        if raw is None:
            return None
        return Draft(
            id=raw["id"],
            tasks=[DraftTask(**task) for task in raw["tasks"]],
            links=raw.get("links", []),
            warnings=raw.get("warnings", []),
            created_at=raw.get("created_at", ""),
        )

    def next_id(self) -> str:
        return f"draft{len(self._all()) + 1}"


def build(book: DraftBook, calendar, memory, client, tasks: list[dict],
          links: list[dict] | None = None, subject_extra: str = "") -> Draft:
    """Turn requested work into a checked draft, without writing anything.

    Three kinds of finding come out, and they are deliberately not one list.
    A date on Shabbat or a festival is moved and reported as moved. A date on
    Hanukkah, or an owner who is already carrying late work, is a warning that
    changes nothing. A decision the task would break lands on that task as a
    blocker, and an approval naming it is refused.
    """
    draft = Draft(
        id=book.next_id(), tasks=[], links=list(links or []),
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )

    for index, raw in enumerate(tasks, start=1):
        task = DraftTask(
            ref=raw.get("ref") or f"d{index}",
            title=raw["title"],
            owners=list(raw.get("owners") or []),
            partners=list(raw.get("partners") or []),
            due_date=raw.get("due_date"),
            project_id=raw.get("project_id"),
            group_name=raw.get("group_name"),
            description=raw.get("description"),
        )

        if task.due_date:
            info = calendar.check([task.due_date])[0]
            if "error" in info:
                draft.warnings.append(f"{task.ref}: תאריך לא תקין {task.due_date}")
            elif not info["working"]:
                task.moved_from = task.due_date
                task.due_date = info["suggested"]
                task.notes.append(f"הוזז מ-{task.moved_from} ({info['reason']})")
                draft.warnings.append(
                    f"{task.ref}: {task.moved_from} נופל ב{info['reason']}, הוזז ל-{task.due_date}"
                )
            if task.due_date:
                after = calendar.check([task.due_date])[0]
                if after.get("note"):
                    task.notes.append(after["note"])
                    draft.warnings.append(f"{task.ref}: {task.due_date} נופל ב{after['note']}")

        for name in task.owners + task.partners:
            if not client.people(name):
                draft.warnings.append(f"{task.ref}: לא נמצא משתמש בשם {name}")
        if not task.owners:
            draft.warnings.append(f"{task.ref}: אין אחראי")

        for record in memory.guards():
            hit = guard_hit(record, task, subject_extra)
            if hit:
                task.blocker = hit
                task.status = BLOCKED

        draft.tasks.append(task)

    known = {task.ref for task in draft.tasks}
    for link in draft.links:
        for side in ("predecessor", "successor"):
            ref = str(link.get(side, ""))
            if ref not in known and not ref.isdigit():
                draft.warnings.append(f"תלות מפנה ל-{ref} שאינו במערכת ואינו בטיוטה")

    book.put(draft)
    return draft


def commit(book: DraftBook, client, draft_id: str,
           approve: list[str] | None = None) -> tuple[Draft | None, str, list[int]]:
    """Write the approved tasks of a draft, and only those.

    Input:
        approve: the refs the person agreed to, or ["all"] for every pending
            task. An empty list is a refusal to write anything, not a shorthand
            for everything.

    Output: the draft, a problem description, and the identifiers created by
    THIS call. The third value is separate because the first two cannot tell
    "three written, one link deferred" from "nothing written, all of it
    blocked" - both are a draft plus a non-empty string.
    """
    draft = book.get(draft_id)
    if draft is None:
        return None, f"אין טיוטה בשם {draft_id}", []

    wanted = [str(ref) for ref in (approve or [])]
    if not wanted:
        return draft, "לא צוינה אף משימה לאישור. commit_plan דורש רשימת ref מפורשת.", []
    if wanted == ["all"]:
        wanted = [task.ref for task in draft.tasks if task.status == PENDING]
        if not wanted:
            return draft, "אין משימות ממתינות לאישור בטיוטה הזאת.", []

    problems: list[str] = []
    to_write: list[DraftTask] = []
    for ref in wanted:
        task = draft.by_ref(ref)
        if task is None:
            problems.append(f"{ref}: אין משימה כזאת בטיוטה")
        elif task.created_id:
            problems.append(f"{ref}: כבר נכתבה כמשימה {task.created_id}")
        elif task.blocker:
            problems.append(f"{ref}: חסומה - {task.blocker}")
        else:
            to_write.append(task)

    if not to_write:
        return draft, "\n".join(problems) or "לא נכתב כלום.", []

    for task in to_write:
        task.created_id = client.create_task(
            title=task.title, project_id=task.project_id, owners=task.owners,
            partners=task.partners, due_date=task.due_date, group_name=task.group_name,
            description=task.description,
        )
        task.status = APPROVED

    def resolve(ref: str) -> int | None:
        ref = str(ref)
        task = draft.by_ref(ref)
        if task is not None:
            return task.created_id
        return int(ref) if ref.isdigit() else None

    for link in draft.links:
        pred, succ = resolve(link.get("predecessor")), resolve(link.get("successor"))
        if pred is None or succ is None:
            # Not an error: a link whose other end is still awaiting approval
            # is simply not due yet, and will be made when that task is agreed.
            problems.append(
                f"תלות {link.get('predecessor')}->{link.get('successor')} ממתינה: "
                "אחד הצדדים עוד לא אושר"
            )
            continue
        if link.get("made"):
            continue
        try:
            client.link(pred, succ)
            link["made"] = True
        except ValueError as error:
            problems.append(str(error))

    book.put(draft)
    return draft, "\n".join(problems), [task.created_id for task in to_write]


def render(draft: Draft) -> str:
    """ref|status|title|owners|partners|due|project/group, plus what to know."""
    lines = [f"טיוטה {draft.id} - אישור נפרד לכל משימה"]
    marks = {PENDING: "ממתינה", APPROVED: "נכתבה", BLOCKED: "חסומה"}
    for task in draft.tasks:
        state = marks.get(task.status, task.status)
        if task.created_id:
            state = f"נכתבה #{task.created_id}"
        lines.append("|".join([
            task.ref, state, task.title, ",".join(task.owners) or "-",
            ",".join(task.partners) or "-", task.due_date or "-",
            f"{task.project_id or '-'}/{task.group_name or '-'}",
        ]))
    for link in draft.links:
        made = " (נוצרה)" if link.get("made") else ""
        lines.append(f"תלות: {link.get('predecessor')} -> {link.get('successor')}{made}")
    for warning in draft.warnings:
        lines.append(f"שים לב: {warning}")
    for task in draft.tasks:
        if task.blocker:
            lines.append(f"חסום {task.ref}: {task.blocker}")

    pending = [task.ref for task in draft.tasks if task.status == PENDING]
    if pending:
        lines.append(
            f"לאישור: commit_plan עם draft_id={draft.id} ו-approve=[{', '.join(pending)}] "
            "- או רק חלקן. אל תאשר בשם המשתמשת."
        )
    return "\n".join(lines)
