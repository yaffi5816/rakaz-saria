"""Row shapes, the frozen clock, and the one-line renderings every tool answers in.

The renderings are the token budget of this agent. A task as the application's
API returns it is a JSON object carrying owners, partners, has_chat_messages,
has_files, subtask_count and three timestamps - roughly 400 tokens once it is
indented. A screen needs all of that. A model deciding who should own the next
task needs seven fields, and pays for the rest on every turn it stays in the
transcript.

So every tool here answers in pipe-separated rows with no keys, and the column
order is documented once in the system prompt instead of being repeated on every
row. That is the whole trick, and it is worth about an order of magnitude.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field

# The organisation's zone, named the way the task application names it. Lateness
# is a statement about a working day in Israel, not about an instant.
LOCAL_TIMEZONE = "Asia/Jerusalem"


def today() -> datetime.date:
    """Return the date the agent treats as today.

    Purpose: every measured run has to start from the same state, and "late by
    five days" is state. Reading the real clock would make the numbers in
    results/api.json drift from the ones in results/sdk.json whenever the two
    runs fall on different days, and would make the tests fail silently from one
    morning to the next.

    Input: nothing is passed. Reads RAKAZ_TODAY, which .env.example pins.

    Output: the frozen date, or the real local date when nothing is pinned.
    """
    pinned = os.environ.get("RAKAZ_TODAY", "").strip()
    if pinned:
        return datetime.date.fromisoformat(pinned)
    return datetime.datetime.now().date()


def short(value: str | datetime.date | None) -> str:
    """Render a date as dd-mm, which is how every row prints one."""
    if value is None:
        return "-"
    if isinstance(value, str):
        value = datetime.date.fromisoformat(value[:10])
    return value.strftime("%d-%m")


@dataclass(frozen=True)
class Person:
    id: int
    username: str
    name: str
    department: str


@dataclass
class ProjectRow:
    id: int
    name: str
    manager: str
    status: str
    start_date: str | None
    end_date: str | None
    budget_total: float | None
    paid: float = 0.0
    open_tasks: int = 0
    late_tasks: int = 0

    def line(self) -> str:
        """id|name|manager|status|start|end|balance|open/late"""
        balance = "-"
        if self.budget_total is not None:
            balance = f"{self.budget_total - self.paid:,.0f}"
        return "|".join(
            [
                str(self.id),
                self.name,
                self.manager,
                self.status,
                short(self.start_date),
                short(self.end_date),
                balance,
                f"{self.open_tasks}/{self.late_tasks}",
            ]
        )


@dataclass
class TaskRow:
    id: int
    title: str
    status: str
    due_date: str | None
    project_id: int | None
    project_name: str | None
    group_name: str | None
    owners: list[str] = field(default_factory=list)
    partners: list[str] = field(default_factory=list)
    completed_at: str | None = None

    @property
    def days_late(self) -> int:
        """How many days past its deadline an unfinished task is, else 0.

        Derived, never stored, for the same reason the application derives it:
        a stored "late" would erase whether the task was new or in progress.
        """
        if self.status == "done" or not self.due_date:
            return 0
        gap = (today() - datetime.date.fromisoformat(self.due_date)).days
        return gap if gap > 0 else 0

    def line(self) -> str:
        """id|title|owners|partners|status|due|project/group

        Owners and partners are comma-separated names, "-" when there are none.
        The due column carries the lateness with it ("21-10 late5") rather than
        in a column of its own, because a date and its lateness are read
        together and an empty eighth column on every on-time row is pure cost.
        """
        due = short(self.due_date)
        if self.days_late:
            due = f"{due} late{self.days_late}"
        where = "-"
        if self.project_id:
            where = str(self.project_id)
            if self.group_name:
                where = f"{where}/{self.group_name}"
        return "|".join(
            [
                str(self.id),
                self.title,
                ",".join(self.owners) or "-",
                ",".join(self.partners) or "-",
                self.status,
                due,
                where,
            ]
        )


# The column orders, stated once in the system prompt and never again on a row.
TASK_COLUMNS = "id|title|owners|partners|status|due|project/group"
PROJECT_COLUMNS = "id|name|manager|status|start|end|budget_left|open/late"
PERSON_COLUMNS = "id|username|name|department"
