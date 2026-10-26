"""The workspace the agent acts on, behind one interface with two implementations.

LocalStore is a seeded SQLite copy. It is what every test and every measured run
uses, because the comparison rules require each run to start from a clean state
and "clean" here means a database, not a JSON file: the agent creates tasks, and
a second run against a workspace still holding the first run's tasks would make
"exactly four tasks were created" fail for a reason that has nothing to do with
the harness being measured.

HttpClient talks to a running instance of the task application over HTTP, as any
other client would. It exists for the live demonstration - run the scenario, then
open the tasks screen and watch the rows appear - and is never used by the tests.

The split is the same one the task application itself uses between
TaskRepository and FakeTaskRepository: one interface, two implementations, and a
single place that decides which one is built.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from shared.schema import Person, ProjectRow, TaskRow


class WorkspaceClient(ABC):
    """Everything the agent's tools need from the workspace.

    Every read is already narrowed to what the acting person may see. The agent
    inherits the application's visibility rule rather than restating it: a
    caller sees their own department's work, plus anything they are personally
    attached to elsewhere. Nothing in the agent can widen that.
    """

    @abstractmethod
    def people(self, query: str | None = None) -> list[Person]: ...

    @abstractmethod
    def projects(self, query: str | None = None) -> list[ProjectRow]: ...

    @abstractmethod
    def tasks(
        self,
        query: str | None = None,
        project_id: int | None = None,
        owner: str | None = None,
        status: str | None = None,
        late: bool | None = None,
        limit: int = 12,
    ) -> list[TaskRow]: ...

    @abstractmethod
    def task(self, task_id: int) -> TaskRow | None: ...

    @abstractmethod
    def dependents(self, task_id: int) -> list[tuple[int, TaskRow]]:
        """Everything downstream of a task, with how many hops away it is."""

    @abstractmethod
    def workload(self, usernames: list[str], days: int) -> list[dict]: ...

    @abstractmethod
    def completed_like(self, keyword: str) -> list[TaskRow]: ...

    @abstractmethod
    def create_task(
        self,
        title: str,
        project_id: int | None,
        owners: list[str],
        partners: list[str],
        due_date: str | None,
        group_name: str | None = None,
        description: str | None = None,
    ) -> int: ...

    @abstractmethod
    def link(self, predecessor_id: int, successor_id: int) -> None: ...


def build_client() -> WorkspaceClient:
    """Build the workspace the environment asks for.

    The one place a concrete implementation is named, so switching the agent
    between its own seeded copy and a live application is an environment
    variable and not a code change.
    """
    kind = os.environ.get("RAKAZ_WORKSPACE", "local").strip().lower()
    actor = os.environ.get("RAKAZ_ACTOR", "sarah.avital").strip()
    if kind == "http":
        from shared.http_client import HttpClient

        return HttpClient(
            base_url=os.environ.get("RAKAZ_API_BASE", "http://localhost:8000"),
            username=os.environ.get("RAKAZ_API_USER", actor),
        )
    from shared.store import LocalStore

    return LocalStore(actor=actor)
