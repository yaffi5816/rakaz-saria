"""Drafting work, and the gate between drafting it and writing it down.

Nothing this agent does reaches the workspace without passing through here
twice: once as a draft the person reads, and once as an explicit commit naming
that draft's identifier. An agent that creates tasks the moment it understands a
sentence is an agent nobody lets near a real board, and the two-phase shape is
also what makes the first check in the suite possible - after the turn that
drafts five tasks, the workspace still holds none.

The gate is not advice. A draft that contradicts a recorded decision carries a
blocker, and commit() refuses while a blocker is on it. The model is told about
the decision as well, and ought to raise it on its own, but the refusal does not
depend on that: it is a condition in code, and that is what makes the
contradiction check deterministic rather than a hope about model behaviour.
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


@dataclass
class Draft:
    id: str
    tasks: list[DraftTask]
    links: list[dict]
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    created_at: str = ""
    committed: list[int] = field(default_factory=list)


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
    if not task.due_date:
        return None
    if task.due_date >= guard["date"]:
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
        data[draft.id] = asdict(draft)
        self._save(data)

    def get(self, draft_id: str) -> Draft | None:
        raw = self._all().get(draft_id)
        if raw is None:
            return None
        raw = dict(raw)
        raw["tasks"] = [DraftTask(**t) for t in raw["tasks"]]
        return Draft(**raw)

    def next_id(self) -> str:
        return f"draft{len(self._all()) + 1}"


def build(book: DraftBook, calendar, memory, client, tasks: list[dict],
          links: list[dict] | None = None, subject_extra: str = "") -> Draft:
    """Turn requested work into a checked draft, without writing anything.

    Three kinds of finding come out, and they are deliberately not one list.
    A date on Shabbat or a festival is moved and reported as moved. A date on
    Hanukkah, or an owner who is already carrying late work, is a warning that
    changes nothing. A decision the plan would break is a blocker, and blockers
    are what commit() refuses on.
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
                draft.blockers.append(hit)

        draft.tasks.append(task)

    known = {t.ref for t in draft.tasks}
    for link in draft.links:
        for side in ("predecessor", "successor"):
            ref = str(link.get(side, ""))
            if ref not in known and not ref.isdigit():
                draft.warnings.append(f"תלות מפנה ל-{ref} שאינו במערכת ואינו בטיוטה")

    book.put(draft)
    return draft


def commit(book: DraftBook, client, draft_id: str) -> tuple[Draft | None, str, bool]:
    """Write a draft to the workspace, unless something says it must not be.

    Output: the draft, a problem description, and whether anything was written.

    The third value exists because the first two cannot tell two opposite
    outcomes apart. "Four tasks were created but one dependency failed" and
    "nothing was created, this draft was already committed" both come back as a
    draft plus a non-empty string, and a caller reading only those two reports
    the second as a success.
    """
    draft = book.get(draft_id)
    if draft is None:
        return None, f"אין טיוטה בשם {draft_id}", False
    if draft.committed:
        return draft, f"הטיוטה {draft_id} כבר נכתבה קודם: {draft.committed}", False
    if draft.blockers:
        return draft, "הטיוטה חסומה:\n" + "\n".join(draft.blockers), False

    created: dict[str, int] = {}
    for task in draft.tasks:
        new_id = client.create_task(
            title=task.title, project_id=task.project_id, owners=task.owners,
            partners=task.partners, due_date=task.due_date, group_name=task.group_name,
            description=task.description,
        )
        created[task.ref] = new_id

    def resolve(ref: str) -> int | None:
        ref = str(ref)
        if ref in created:
            return created[ref]
        return int(ref) if ref.isdigit() else None

    failures = []
    for link in draft.links:
        pred, succ = resolve(link.get("predecessor")), resolve(link.get("successor"))
        if pred is None or succ is None:
            failures.append(f"תלות {link} לא נוצרה: מזהה לא נפתר")
            continue
        try:
            client.link(pred, succ)
        except ValueError as error:
            failures.append(str(error))

    draft.committed = list(created.values())
    book.put(draft)
    return draft, "\n".join(failures), True


def render(draft: Draft) -> str:
    """ref|title|owners|partners|due|project/group - plus what to know about it."""
    lines = [f"טיוטה {draft.id}"]
    for task in draft.tasks:
        lines.append("|".join([
            task.ref, task.title, ",".join(task.owners) or "-",
            ",".join(task.partners) or "-", task.due_date or "-",
            f"{task.project_id or '-'}/{task.group_name or '-'}",
        ]))
    for link in draft.links:
        lines.append(f"תלות: {link.get('predecessor')} -> {link.get('successor')}")
    for warning in draft.warnings:
        lines.append(f"שים לב: {warning}")
    for blocker in draft.blockers:
        lines.append(f"חסום: {blocker}")
    if draft.blockers:
        lines.append("לא ניתן לאשר את הטיוטה כל עוד יש חסימה.")
    else:
        lines.append(f"לאישור: commit_plan עם draft_id={draft.id}")
    return "\n".join(lines)
