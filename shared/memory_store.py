"""What the agent carries between sessions: how this department works.

Two files, and the split matters.

memory/profile.md is prose the person wrote about their own unit, small enough
to sit in every turn's context. In the SDK version it is CLAUDE.md, read by
setting_sources=["project"] - one line replacing the half of harness/memory.py
that loads it.

memory/conventions.jsonl is the growing part: every rule and every management
decision the agent has been told, one JSON object per line, appended and never
rewritten. Releasing a decision writes a new record pointing at the old one, so
the file is also the audit trail of who decided what and when - which is the
whole reason a department would trust the thing.

A decision may carry a guard: a machine-readable version of the constraint, such
as {"type": "not_before", "date": "2027-01-01"}. The model fills it when it saves
the decision, and derive_guard below re-derives it from the Hebrew if the model
left it out. That is deliberate belt and braces: the guard is what stops a
contradicting plan in code rather than by good behaviour, so it must not depend
on the model remembering to produce it.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = ROOT / "memory"
CONVENTIONS = MEMORY_DIR / "conventions.jsonl"
PROFILE = MEMORY_DIR / "profile.md"

HEBREW_MONTHS = {
    "ינואר": 1, "פברואר": 2, "מרץ": 3, "מרס": 3, "אפריל": 4, "מאי": 5, "יוני": 6,
    "יולי": 7, "אוגוסט": 8, "ספטמבר": 9, "אוקטובר": 10, "נובמבר": 11, "דצמבר": 12,
}

STOPWORDS = {"של", "על", "את", "לא", "עם", "אל", "כל", "זה", "הוא", "היא", "אני", "יש", "או"}


def derive_guard(rule: str) -> dict | None:
    """Turn a Hebrew constraint into something a check can run against.

    Purpose: the contradiction the scenario exercises - asking for work that a
    recorded decision forbids - has to be caught by the tool, not by the model
    noticing. Without a structured form there is nothing to catch it with.

    Input:
        rule: the decision as the person said it, e.g. "לא לפני ינואר 2027".

    Output: a guard dict, or None when the sentence carries no date constraint.
    """
    text = rule.strip()
    # A negation somewhere and the word "לפני" somewhere. Trying to enumerate
    # the verb forms a person might use - פותחים, לפתוח, מתחילים, נוגעים - is a
    # list that is wrong the first time somebody says it differently; the model
    # fills guard_date itself in the normal case, and this is the safety net.
    if not (re.search(r"\b(לא|אין|אסור)\b", text) and "לפני" in text):
        return None
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        return {"type": "not_before", "date": iso.group(0)}
    numeric = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", text)
    if numeric:
        day, month, year = (int(g) for g in numeric.groups())
        return {"type": "not_before", "date": f"{year:04d}-{month:02d}-{day:02d}"}
    for name, month in HEBREW_MONTHS.items():
        if name in text:
            year = re.search(r"\b(20\d{2})\b", text)
            if year:
                return {"type": "not_before", "date": f"{int(year.group(1)):04d}-{month:02d}-01"}
    return None


class MemoryStore:
    """Append-only conventions, plus the profile prose."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or MEMORY_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "conventions.jsonl"
        self.profile_path = self.dir / "profile.md"

    # ── writing ──────────────────────────────────────────────────────

    def _append(self, record: dict) -> dict:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def remember(self, kind: str, topic: str, rule: str, guard: dict | None = None,
                 session: str = "") -> dict:
        records = self.records()
        record = {
            "id": f"m{len(records) + 1}",
            "kind": kind,
            "topic": topic.strip(),
            "rule": rule.strip(),
            "guard": guard or derive_guard(rule),
            "status": "active",
            "supersedes": None,
            "session": session,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        return self._append(record)

    def release(self, record_id: str, reason: str, session: str = "") -> dict | None:
        """Retire a record by appending its replacement, never by erasing it."""
        target = next((r for r in self.records() if r["id"] == record_id), None)
        if target is None:
            return None
        records = self.records()
        return self._append({
            "id": f"m{len(records) + 1}",
            "kind": target["kind"],
            "topic": target["topic"],
            "rule": f"בוטל: {target['rule']}",
            "guard": None,
            "status": "released",
            "supersedes": record_id,
            "session": session,
            "reason": reason,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    # ── reading ──────────────────────────────────────────────────────

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def active(self) -> list[dict]:
        records = self.records()
        retired = {r["supersedes"] for r in records if r.get("supersedes")}
        return [r for r in records if r["status"] == "active" and r["id"] not in retired]

    def guards(self) -> list[dict]:
        return [r for r in self.active() if r.get("guard")]

    def relevant(self, text: str, limit: int = 4) -> list[dict]:
        """The records worth spending context on for this turn, and no others.

        Purpose: this is the context engineering, in four lines. A department
        accumulates rules indefinitely; a turn is about one or two of them. The
        score is deliberately crude word overlap - the model can always ask for
        more with recall, and a cheap filter that is sometimes wrong costs less
        than a clever one that is sometimes slow.
        """
        words = {w for w in re.split(r"\W+", text) if len(w) > 2 and w not in STOPWORDS}
        scored = []
        for record in self.active():
            haystack = {
                w for w in re.split(r"\W+", f"{record['topic']} {record['rule']}")
                if len(w) > 2 and w not in STOPWORDS
            }
            overlap = len(words & haystack)
            # A decision outranks a convention at equal overlap: forgetting a
            # preference produces a clumsy plan, forgetting a decision produces
            # a plan that should never have been drafted.
            weight = overlap + (0.5 if record["kind"] == "decision" else 0)
            if overlap:
                scored.append((weight, record))
        scored.sort(key=lambda pair: -pair[0])
        return [record for _, record in scored[:limit]]

    def profile(self) -> str:
        if self.profile_path.exists():
            return self.profile_path.read_text(encoding="utf-8").strip()
        return ""


def render(records: list[dict]) -> str:
    """id|kind|topic|rule|guard - the compact form of a memory lookup."""
    if not records:
        return "(אין רשומות מתאימות)"
    lines = []
    for r in records:
        guard = ""
        if r.get("guard"):
            guard = f"{r['guard'].get('type')}:{r['guard'].get('date', '')}"
        lines.append("|".join([r["id"], r["kind"], r["topic"], r["rule"], guard or "-"]))
    return "\n".join(lines)
