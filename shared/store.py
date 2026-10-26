"""The seeded SQLite workspace: the agent's own copy of a task application.

Built from shared/fixtures/workspace.json the first time it is opened, and
deleted by reset() at the start of every measured run. The tables mirror the
task application's schema closely enough that HttpClient can answer the same
questions from the real thing without any tool knowing which one it reached.

Visibility is enforced here, in _visible_tasks and _visible_projects, and
nowhere else - the same shape the application uses, where every read starts from
the access rule and only the column list changes. The agent has no way around
it: a tool asking for "all late tasks" receives the acting person's late tasks.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from pathlib import Path

from shared.client import WorkspaceClient
from shared.schema import Person, ProjectRow, TaskRow, today

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "shared" / "fixtures" / "workspace.json"
DB_PATH = ROOT / "state" / "workspace.db"

# Tasks the agent creates start here, so a test can tell seeded work from work
# this run produced without carrying a list of seeded identifiers around.
FIRST_CREATED_ID = 1000

SCHEMA = """
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE users (
    id INTEGER PRIMARY KEY, ad_username TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL,
    email TEXT NOT NULL, department_id INTEGER NOT NULL, is_admin INTEGER NOT NULL DEFAULT 0,
    views_organisation INTEGER NOT NULL DEFAULT 0);
CREATE TABLE projects (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, department_id INTEGER, manager_id INTEGER NOT NULL,
    budget_total REAL, status TEXT NOT NULL, start_date TEXT, end_date TEXT);
CREATE TABLE task_groups (
    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, name TEXT NOT NULL, position INTEGER NOT NULL);
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, description TEXT, department_id INTEGER NOT NULL,
    project_id INTEGER, group_id INTEGER, status TEXT NOT NULL, start_date TEXT, due_date TEXT,
    completed_at TEXT, created_by INTEGER NOT NULL, created_at TEXT NOT NULL,
    created_by_agent INTEGER NOT NULL DEFAULT 0);
CREATE TABLE task_owners (task_id INTEGER NOT NULL, user_id INTEGER NOT NULL, PRIMARY KEY (task_id, user_id));
CREATE TABLE task_partners (task_id INTEGER NOT NULL, user_id INTEGER NOT NULL, PRIMARY KEY (task_id, user_id));
CREATE TABLE dependencies (predecessor_id INTEGER NOT NULL, successor_id INTEGER NOT NULL,
    PRIMARY KEY (predecessor_id, successor_id));
CREATE TABLE payments (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, amount REAL NOT NULL,
    paid_at TEXT, description TEXT);
"""


# Every connection opened this process. Windows refuses to delete a file that
# any handle still has open, so reset() has to close them rather than hope the
# garbage collector got there first - which it does in a single run and does
# not when two runs share a process, as the suite does.
_OPEN: list[sqlite3.Connection] = []


def reset() -> None:
    """Delete the workspace so the next open rebuilds it from the fixture.

    Purpose: "every run starts from a clean state" is a comparison rule, and
    without it the second run of a scenario measures a workspace the first run
    already wrote into.
    """
    while _OPEN:
        try:
            _OPEN.pop().close()
        except sqlite3.Error:
            pass
    if DB_PATH.exists():
        DB_PATH.unlink()


class LocalStore(WorkspaceClient):
    def __init__(self, actor: str = "sarah.avital") -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        fresh = not DB_PATH.exists()
        self.db = sqlite3.connect(DB_PATH)
        self.db.row_factory = sqlite3.Row
        _OPEN.append(self.db)
        if fresh:
            self._seed()
        row = self.db.execute(
            "SELECT id, department_id FROM users WHERE ad_username = ?", (actor,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown actor: {actor}")
        self.actor_id = row["id"]
        self.actor_department = row["department_id"]

    # ── seeding ──────────────────────────────────────────────────────

    def _seed(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.db.executescript(SCHEMA)
        for d in data["departments"]:
            self.db.execute("INSERT INTO departments VALUES (?, ?)", (d["id"], d["name"]))
        for u in data["users"]:
            self.db.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                (u["id"], u["ad_username"], u["display_name"], u["email"],
                 u["department_id"], int(u["is_admin"]), int(u["views_organisation"])),
            )
        for p in data["projects"]:
            self.db.execute(
                "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?)",
                (p["id"], p["name"], p["department_id"], p["manager_id"], p["budget_total"],
                 p["status"], p["start_date"], p["end_date"]),
            )
        for g in data["task_groups"]:
            self.db.execute(
                "INSERT INTO task_groups VALUES (?,?,?,?)",
                (g["id"], g["project_id"], g["name"], g["position"]),
            )
        for t in data["tasks"]:
            self.db.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (t["id"], t["title"], t.get("description"), t["department_id"], t["project_id"],
                 t["group_id"], t["status"], t["start_date"], t["due_date"], t["completed_at"],
                 t["created_by"], "2026-01-01T08:00:00+02:00"),
            )
            for uid in t["owners"]:
                self.db.execute("INSERT INTO task_owners VALUES (?,?)", (t["id"], uid))
            for uid in t["partners"]:
                self.db.execute("INSERT INTO task_partners VALUES (?,?)", (t["id"], uid))
        for pred, succ in data["dependencies"]:
            self.db.execute("INSERT INTO dependencies VALUES (?,?)", (pred, succ))
        for p in data["payments"]:
            self.db.execute(
                "INSERT INTO payments VALUES (?,?,?,?,?)",
                (p["id"], p["project_id"], p["amount"], p["paid_at"], p["description"]),
            )
        self.db.commit()

    # ── the access rule, in its single copy ──────────────────────────

    def _visible_tasks(self) -> str:
        """The SQL condition every task read starts from.

        A person sees their own department's tasks, and any task they are
        attached to as owner or partner in another one.
        """
        return (
            "(t.department_id = :dept"
            " OR EXISTS (SELECT 1 FROM task_owners o WHERE o.task_id = t.id AND o.user_id = :me)"
            " OR EXISTS (SELECT 1 FROM task_partners p WHERE p.task_id = t.id AND p.user_id = :me))"
        )

    def _params(self, **extra) -> dict:
        return {"me": self.actor_id, "dept": self.actor_department, **extra}

    # ── reads ────────────────────────────────────────────────────────

    def people(self, query: str | None = None) -> list[Person]:
        sql = ("SELECT u.id, u.ad_username, u.display_name, d.name AS dept FROM users u "
               "JOIN departments d ON d.id = u.department_id")
        args: list = []
        if query:
            sql += " WHERE u.display_name LIKE ? OR u.ad_username LIKE ? OR d.name LIKE ?"
            args = [f"%{query}%"] * 3
        rows = self.db.execute(sql + " ORDER BY u.id", args).fetchall()
        return [Person(r["id"], r["ad_username"], r["display_name"], r["dept"]) for r in rows]

    def _person_names(self, task_ids: list[int], table: str) -> dict[int, list[str]]:
        if not task_ids:
            return {}
        marks = ",".join("?" * len(task_ids))
        rows = self.db.execute(
            f"SELECT a.task_id, u.display_name FROM {table} a JOIN users u ON u.id = a.user_id "
            f"WHERE a.task_id IN ({marks}) ORDER BY u.id",
            task_ids,
        ).fetchall()
        out: dict[int, list[str]] = {}
        for r in rows:
            out.setdefault(r["task_id"], []).append(r["display_name"])
        return out

    def _rows_to_tasks(self, rows) -> list[TaskRow]:
        ids = [r["id"] for r in rows]
        owners = self._person_names(ids, "task_owners")
        partners = self._person_names(ids, "task_partners")
        return [
            TaskRow(
                id=r["id"], title=r["title"], status=r["status"], due_date=r["due_date"],
                project_id=r["project_id"], project_name=r["project_name"],
                group_name=r["group_name"], owners=owners.get(r["id"], []),
                partners=partners.get(r["id"], []), completed_at=r["completed_at"],
            )
            for r in rows
        ]

    _TASK_SELECT = (
        "SELECT t.*, p.name AS project_name, g.name AS group_name FROM tasks t "
        "LEFT JOIN projects p ON p.id = t.project_id "
        "LEFT JOIN task_groups g ON g.id = t.group_id"
    )

    def tasks(self, query=None, project_id=None, owner=None, status=None, late=None, limit=12):
        sql = f"{self._TASK_SELECT} WHERE {self._visible_tasks()}"
        params = self._params()
        if query:
            sql += " AND (t.title LIKE :q OR p.name LIKE :q)"
            params["q"] = f"%{query}%"
        if project_id:
            sql += " AND t.project_id = :pid"
            params["pid"] = project_id
        if status:
            sql += " AND t.status = :st"
            params["st"] = status
        if owner:
            sql += (" AND EXISTS (SELECT 1 FROM task_owners o JOIN users u ON u.id = o.user_id"
                    " WHERE o.task_id = t.id AND (u.ad_username = :own OR u.display_name LIKE :ownl))")
            params["own"] = owner
            params["ownl"] = f"%{owner}%"
        if late:
            sql += " AND t.status <> 'done' AND t.due_date IS NOT NULL AND t.due_date < :today"
            params["today"] = today().isoformat()
        # Open work first, then by deadline. Ordering by date alone puts the
        # oldest finished tasks at the top of every answer, which is both the
        # least useful rows and the ones the caller pays for.
        sql += (" ORDER BY (CASE WHEN t.status = 'done' THEN 1 ELSE 0 END), "
                "(t.due_date IS NULL), t.due_date, t.id LIMIT :lim")
        params["lim"] = limit
        return self._rows_to_tasks(self.db.execute(sql, params).fetchall())

    def task(self, task_id: int) -> TaskRow | None:
        rows = self.db.execute(
            f"{self._TASK_SELECT} WHERE t.id = :tid AND {self._visible_tasks()}",
            self._params(tid=task_id),
        ).fetchall()
        found = self._rows_to_tasks(rows)
        return found[0] if found else None

    def dependents(self, task_id: int) -> list[tuple[int, TaskRow]]:
        seen: dict[int, int] = {}
        frontier = [(task_id, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            rows = self.db.execute(
                "SELECT successor_id FROM dependencies WHERE predecessor_id = ?", (current,)
            ).fetchall()
            for r in rows:
                nxt = r["successor_id"]
                if nxt not in seen:
                    seen[nxt] = depth + 1
                    frontier.append((nxt, depth + 1))
        out = []
        for tid, depth in sorted(seen.items(), key=lambda kv: (kv[1], kv[0])):
            row = self.task(tid)
            if row:
                out.append((depth, row))
        return out

    def projects(self, query: str | None = None) -> list[ProjectRow]:
        sql = ("SELECT p.*, u.display_name AS manager FROM projects p "
               "JOIN users u ON u.id = p.manager_id "
               "WHERE (p.department_id = :dept OR p.manager_id = :me)")
        params = self._params()
        if query:
            sql += " AND p.name LIKE :q"
            params["q"] = f"%{query}%"
        rows = self.db.execute(sql + " ORDER BY p.id", params).fetchall()
        out = []
        for r in rows:
            paid = self.db.execute(
                "SELECT COALESCE(SUM(amount), 0) AS s FROM payments WHERE project_id = ?", (r["id"],)
            ).fetchone()["s"]
            counts = self.db.execute(
                "SELECT COUNT(*) AS open_n, "
                "SUM(CASE WHEN due_date IS NOT NULL AND due_date < ? THEN 1 ELSE 0 END) AS late_n "
                "FROM tasks WHERE project_id = ? AND status <> 'done'",
                (today().isoformat(), r["id"]),
            ).fetchone()
            out.append(
                ProjectRow(
                    id=r["id"], name=r["name"], manager=r["manager"], status=r["status"],
                    start_date=r["start_date"], end_date=r["end_date"],
                    budget_total=r["budget_total"], paid=paid,
                    open_tasks=counts["open_n"] or 0, late_tasks=counts["late_n"] or 0,
                )
            )
        return out

    def workload(self, usernames: list[str], days: int) -> list[dict]:
        out = []
        horizon = (today() + datetime.timedelta(days=days)).isoformat()
        for name in usernames:
            person = self.db.execute(
                "SELECT id, display_name FROM users WHERE ad_username = ? OR display_name LIKE ?",
                (name, f"%{name}%"),
            ).fetchone()
            if person is None:
                out.append({"name": name, "found": False})
                continue
            row = self.db.execute(
                "SELECT COUNT(*) AS open_n, "
                " SUM(CASE WHEN t.due_date IS NOT NULL AND t.due_date < :today THEN 1 ELSE 0 END) AS late_n, "
                " SUM(CASE WHEN t.due_date IS NOT NULL AND t.due_date BETWEEN :today AND :horizon "
                "          THEN 1 ELSE 0 END) AS soon_n "
                "FROM tasks t JOIN task_owners o ON o.task_id = t.id "
                "WHERE o.user_id = :uid AND t.status <> 'done'",
                {"uid": person["id"], "today": today().isoformat(), "horizon": horizon},
            ).fetchone()
            out.append({
                "name": person["display_name"], "found": True, "open": row["open_n"] or 0,
                "late": row["late_n"] or 0, "soon": row["soon_n"] or 0,
            })
        return out

    def completed_like(self, keyword: str) -> list[TaskRow]:
        rows = self.db.execute(
            f"{self._TASK_SELECT} WHERE t.status = 'done' AND t.completed_at IS NOT NULL "
            f"AND t.due_date IS NOT NULL AND t.title LIKE :q AND {self._visible_tasks()} "
            "ORDER BY t.completed_at",
            self._params(q=f"%{keyword}%"),
        ).fetchall()
        return self._rows_to_tasks(rows)

    # ── writes ───────────────────────────────────────────────────────

    def _resolve_user(self, name: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT id, department_id FROM users WHERE ad_username = ? OR display_name = ? "
            "OR display_name LIKE ?",
            (name, name, f"%{name}%"),
        ).fetchone()

    def create_task(self, title, project_id, owners, partners, due_date,
                    group_name=None, description=None) -> int:
        next_id = self.db.execute(
            "SELECT COALESCE(MAX(id), ?) + 1 AS n FROM tasks WHERE id >= ?",
            (FIRST_CREATED_ID - 1, FIRST_CREATED_ID),
        ).fetchone()["n"]
        department_id = self.actor_department
        group_id = None
        if project_id:
            project = self.db.execute(
                "SELECT department_id FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if project and project["department_id"]:
                department_id = project["department_id"]
            if group_name:
                group = self.db.execute(
                    "SELECT id FROM task_groups WHERE project_id = ? AND name = ?",
                    (project_id, group_name),
                ).fetchone()
                group_id = group["id"] if group else None
        self.db.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,'new',NULL,?,NULL,?,?,1)",
            (next_id, title, description, department_id, project_id, group_id, due_date,
             self.actor_id, datetime.datetime.now().isoformat(timespec="seconds")),
        )
        for name in owners:
            person = self._resolve_user(name)
            if person:
                self.db.execute("INSERT OR IGNORE INTO task_owners VALUES (?,?)", (next_id, person["id"]))
        for name in partners:
            person = self._resolve_user(name)
            if person:
                self.db.execute("INSERT OR IGNORE INTO task_partners VALUES (?,?)", (next_id, person["id"]))
        self.db.commit()
        return next_id

    # ── inspection, for the checks ───────────────────────────────────

    def created_by_agent(self) -> list[dict]:
        """Every task this agent wrote, outside the visibility rule.

        Used by the checks and by nothing the model can reach: a check asking
        "what did the run actually create" must see the rows as they are, not
        as the acting person sees them.
        """
        rows = self.db.execute(
            "SELECT id, title, project_id, due_date, created_at FROM tasks "
            "WHERE created_by_agent = 1 ORDER BY id"
        ).fetchall()
        out = []
        for row in rows:
            owners = self._person_names([row["id"]], "task_owners").get(row["id"], [])
            partners = self._person_names([row["id"]], "task_partners").get(row["id"], [])
            out.append({
                "id": row["id"], "title": row["title"], "project_id": row["project_id"],
                "due_date": row["due_date"], "owners": owners, "partners": partners,
            })
        return out

    def all_dependencies(self) -> list[tuple[int, int]]:
        return [
            (r["predecessor_id"], r["successor_id"])
            for r in self.db.execute(
                "SELECT predecessor_id, successor_id FROM dependencies "
                "ORDER BY predecessor_id, successor_id"
            ).fetchall()
        ]

    def link(self, predecessor_id: int, successor_id: int) -> None:
        if predecessor_id == successor_id:
            raise ValueError("משימה לא יכולה לחכות לעצמה")
        for _, row in self.dependents(successor_id):
            if row.id == predecessor_id:
                raise ValueError(
                    f"התלות {predecessor_id}->{successor_id} סוגרת מעגל: "
                    f"{successor_id} כבר מוביל ל-{predecessor_id}"
                )
        self.db.execute(
            "INSERT OR IGNORE INTO dependencies VALUES (?,?)", (predecessor_id, successor_id)
        )
        self.db.commit()
