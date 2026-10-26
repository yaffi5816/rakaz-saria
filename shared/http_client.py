"""The same workspace, reached over HTTP against a running task application.

Used only for the live demonstration: run the scenario with RAKAZ_WORKSPACE=http
and then open the tasks screen, where the rows the agent drafted are now sitting
in the right project with the right owners. No test touches this class, and no
measured run uses it, because a run has to start from a clean state and this one
writes into a real board.

Authentication is the application's own development path: the X-Dev-User header
naming the person to act as. Against anything but a development instance that
header does nothing, which is correct - the agent should never be able to pick
its own identity on a real deployment.

The routes follow the application as it stands: GET/POST /tasks,
GET /projects, GET /users, POST /projects/{id}/dependencies. Response field
names are read defensively, so a renamed optional field degrades a row rather
than ending a demonstration.
"""

from __future__ import annotations

import datetime

import httpx

from shared.client import WorkspaceClient
from shared.schema import Person, ProjectRow, TaskRow, today


class HttpClient(WorkspaceClient):
    def __init__(self, base_url: str, username: str) -> None:
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-Dev-User": username},
            timeout=20.0,
        )
        self.username = username
        self._people_cache: list[Person] | None = None

    # ── helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, **params) -> list[dict]:
        clean = {k: v for k, v in params.items() if v is not None}
        response = self.http.get(path, params=clean)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("items", payload.get("results", []))
        return payload

    def _names(self, people: list[dict]) -> list[str]:
        return [p.get("display_name") or p.get("name") or str(p.get("id")) for p in people]

    def _task(self, raw: dict) -> TaskRow:
        project = raw.get("project") or {}
        group = raw.get("group") or {}
        return TaskRow(
            id=raw["id"], title=raw.get("title", ""), status=raw.get("status", "new"),
            due_date=(raw.get("due_date") or None),
            project_id=raw.get("project_id") or project.get("id"),
            project_name=project.get("name"),
            group_name=group.get("name"),
            owners=self._names(raw.get("owners") or []),
            partners=self._names(raw.get("partners") or []),
            completed_at=raw.get("completed_at"),
        )

    def _user_id(self, name: str) -> int | None:
        for person in self.people():
            if name in (person.username, person.name) or name in person.name:
                return person.id
        return None

    # ── reads ────────────────────────────────────────────────────────

    def people(self, query: str | None = None) -> list[Person]:
        if self._people_cache is None:
            self._people_cache = [
                Person(
                    u["id"], u.get("ad_username", ""), u.get("display_name", ""),
                    (u.get("department") or {}).get("name", ""),
                )
                for u in self._get("/users")
            ]
        if not query:
            return self._people_cache
        return [p for p in self._people_cache if query in p.name or query in p.username
                or query in p.department]

    def projects(self, query: str | None = None) -> list[ProjectRow]:
        out = []
        for raw in self._get("/projects"):
            if query and query not in raw.get("name", ""):
                continue
            manager = raw.get("manager") or {}
            out.append(ProjectRow(
                id=raw["id"], name=raw.get("name", ""),
                manager=manager.get("display_name", ""), status=raw.get("status", ""),
                start_date=raw.get("start_date"), end_date=raw.get("end_date"),
                budget_total=raw.get("budget_total"),
                paid=float(raw.get("paid_total") or 0),
                open_tasks=int(raw.get("open_tasks") or 0),
                late_tasks=int(raw.get("late_tasks") or 0),
            ))
        return out

    def tasks(self, query=None, project_id=None, owner=None, status=None, late=None, limit=12):
        rows = self._get("/tasks", project_id=project_id, status=status,
                         late="true" if late else None, q=query, page_size=limit)
        tasks = [self._task(r) for r in rows][:limit]
        if owner:
            tasks = [t for t in tasks if any(owner in name for name in t.owners)]
        return tasks

    def task(self, task_id: int) -> TaskRow | None:
        try:
            response = self.http.get(f"/tasks/{task_id}")
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return self._task(response.json())

    def dependents(self, task_id: int) -> list[tuple[int, TaskRow]]:
        row = self.task(task_id)
        if row is None or not row.project_id:
            return []
        edges = self._get(f"/projects/{row.project_id}/dependencies")
        forward: dict[int, list[int]] = {}
        for edge in edges:
            forward.setdefault(edge["predecessor_id"], []).append(edge["successor_id"])
        seen: dict[int, int] = {}
        frontier = [(task_id, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            for nxt in forward.get(current, []):
                if nxt not in seen:
                    seen[nxt] = depth + 1
                    frontier.append((nxt, depth + 1))
        out = []
        for tid, depth in sorted(seen.items(), key=lambda kv: (kv[1], kv[0])):
            found = self.task(tid)
            if found:
                out.append((depth, found))
        return out

    def workload(self, usernames: list[str], days: int) -> list[dict]:
        horizon = (today() + datetime.timedelta(days=days)).isoformat()
        out = []
        for name in usernames:
            person = next((p for p in self.people() if name in (p.username, p.name)
                           or name in p.name), None)
            if person is None:
                out.append({"name": name, "found": False})
                continue
            rows = [self._task(r) for r in self._get("/tasks", page_size=200)]
            mine = [t for t in rows if person.name in t.owners and t.status != "done"]
            out.append({
                "name": person.name, "found": True, "open": len(mine),
                "late": sum(1 for t in mine if t.days_late),
                "soon": sum(1 for t in mine if t.due_date and
                            today().isoformat() <= t.due_date <= horizon),
            })
        return out

    def completed_like(self, keyword: str) -> list[TaskRow]:
        rows = [self._task(r) for r in self._get("/tasks", status="done", page_size=200)]
        return [t for t in rows if keyword in t.title and t.due_date and t.completed_at]

    # ── writes ───────────────────────────────────────────────────────

    def create_task(self, title, project_id, owners, partners, due_date,
                    group_name=None, description=None) -> int:
        body = {
            "title": title, "description": description, "project_id": project_id,
            "due_date": due_date, "status": "new",
            "owner_ids": [i for i in (self._user_id(n) for n in owners) if i],
            "partner_ids": [i for i in (self._user_id(n) for n in partners) if i],
        }
        response = self.http.post("/tasks", json={k: v for k, v in body.items() if v is not None})
        response.raise_for_status()
        return response.json()["id"]

    def link(self, predecessor_id: int, successor_id: int) -> None:
        row = self.task(successor_id)
        if row is None or not row.project_id:
            raise ValueError(f"אין פרויקט למשימה {successor_id}")
        response = self.http.post(
            f"/projects/{row.project_id}/dependencies",
            json={"predecessor_id": predecessor_id, "successor_id": successor_id},
        )
        if response.status_code == 409:
            raise ValueError(f"התלות {predecessor_id}->{successor_id} סוגרת מעגל")
        response.raise_for_status()
