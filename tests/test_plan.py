"""Drafting, the calendar, and the gate that refuses a contradicting plan."""

from __future__ import annotations

import pytest

from shared import plan as planning
from shared.hebcal import CalendarService
from shared.memory_store import MemoryStore
from shared.plan import DraftBook
from shared.store import LocalStore


@pytest.fixture
def table(clean):
    return (
        DraftBook(),
        CalendarService(mode="offline"),
        MemoryStore(),
        LocalStore(actor="sarah.avital"),
    )


def test_a_date_on_shabbat_is_moved_to_the_next_working_day(table):
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"title": "כתיבת נוהל", "owners": ["רחל כהן"], "due_date": "2026-11-21"},
    ])
    assert draft.tasks[0].due_date == "2026-11-22"
    assert draft.tasks[0].moved_from == "2026-11-21"
    assert any("שבת" in warning for warning in draft.warnings)
    assert not draft.blockers, "שבת מזיזה תאריך, היא לא חוסמת תוכנית"


def test_a_festival_is_a_warning_and_not_a_move(table):
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"title": "מצגת", "owners": ["איתי גולן"], "due_date": "2026-12-07"},
    ])
    assert draft.tasks[0].due_date == "2026-12-07", "חנוכה הוא יום עבודה"
    assert any("חנוכה" in warning for warning in draft.warnings)


def test_a_decision_blocks_the_draft_and_the_commit(table):
    book, calendar, memory, client = table
    memory.remember("decision", "שלב ב", "לא פותחים את שלב ב לפני ינואר 2027",
                    {"type": "not_before", "date": "2027-01-01"})

    draft = planning.build(book, calendar, memory, client, [
        {"title": "הכנת מדגם", "owners": ["רחל כהן"], "due_date": "2026-12-20",
         "project_id": 1},
    ], subject_extra="תפתחי את משימות שלב ב של סקר כוח אדם")

    assert draft.blockers, "בקשה שסותרת החלטה חייבת להיחסם"
    before = len(client.created_by_agent())
    _, problem, created = planning.commit(book, client, draft.id, ["d1"])
    assert created == [], "אישור מפורש לא גובר על חסימה"
    assert "חסומה" in problem
    assert len(client.created_by_agent()) == before, "חסימה פירושה שלא נכתב כלום"


def test_an_unrelated_topic_is_not_caught_by_the_guard(table):
    book, calendar, memory, client = table
    memory.remember("decision", "שלב ב", "לא לפני ינואר 2027",
                    {"type": "not_before", "date": "2027-01-01"})
    draft = planning.build(book, calendar, memory, client, [
        {"title": "בדיקת שאלון מודול שינה", "owners": ["נועה לוי"],
         "due_date": "2026-11-26", "project_id": 2},
    ], subject_extra="צריך משימת בדיקת שאלון למודול שינה")
    assert not draft.blockers


def test_commit_writes_tasks_owners_and_links(table):
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"ref": "d1", "title": "סביבת בדיקות", "owners": ["דנה שפירא"],
         "due_date": "2026-11-26", "project_id": 2},
    ], links=[{"predecessor": "202", "successor": "d1"}])

    committed, problem, created = planning.commit(book, client, draft.id, ["d1"])
    assert created and problem == ""
    assert len(committed.committed) == 1
    created = client.created_by_agent()[0]
    assert created["owners"] == ["דנה שפירא"]
    assert (202, created["id"]) in client.all_dependencies()


def test_a_draft_cannot_be_committed_twice(table):
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"title": "משימה", "owners": ["רחל כהן"], "due_date": "2026-11-26"},
    ])
    planning.commit(book, client, draft.id, ["d1"])
    _, problem, created = planning.commit(book, client, draft.id, ["d1"])
    assert created == []
    assert "כבר נכתבה" in problem


def test_a_cycle_is_refused(table):
    _, _, _, client = table
    with pytest.raises(ValueError, match="מעגל"):
        client.link(105, 101)


def test_only_the_approved_tasks_are_written(table):
    """Three of four named, so three rows - and the fourth still waiting."""
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"ref": "d1", "title": "סביבת בדיקות", "owners": ["דנה שפירא"], "due_date": "2026-11-26"},
        {"ref": "d2", "title": "נוהל עבודה", "owners": ["איתי גולן"], "due_date": "2026-11-25"},
        {"ref": "d3", "title": "מצגת סטטוס", "owners": ["איתי גולן"], "due_date": "2026-12-03"},
        {"ref": "d4", "title": "תיאום הדרכה", "owners": ["יוסי ברק"], "due_date": "2026-11-26"},
    ])

    _, _, created = planning.commit(book, client, draft.id, ["d1", "d2", "d3"])
    assert len(created) == 3

    written = {task["title"] for task in client.created_by_agent()}
    assert "תיאום הדרכה" not in written

    later = book.get(draft.id)
    assert later.by_ref("d4").status == planning.PENDING
    assert later.by_ref("d1").created_id is not None


def test_an_empty_approval_writes_nothing(table):
    """Silence is not consent: commit with no refs is a refusal."""
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"title": "משימה", "owners": ["רחל כהן"], "due_date": "2026-11-26"},
    ])
    _, problem, created = planning.commit(book, client, draft.id, [])
    assert created == []
    assert "לא צוינה" in problem
    assert client.created_by_agent() == []


def test_a_link_waits_for_its_other_end(table):
    """A dependency is made when both ends exist, not before."""
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"ref": "d1", "title": "ראשונה", "owners": ["רחל כהן"], "due_date": "2026-11-25"},
        {"ref": "d2", "title": "שנייה", "owners": ["רחל כהן"], "due_date": "2026-11-26"},
    ], links=[{"predecessor": "d1", "successor": "d2"}])

    _, problem, created = planning.commit(book, client, draft.id, ["d2"])
    assert len(created) == 1
    assert "ממתינה" in problem
    assert client.all_dependencies() == [(101, 102), (102, 103), (103, 104), (104, 105)]

    _, _, more = planning.commit(book, client, draft.id, ["d1"])
    assert (more[0], created[0]) in client.all_dependencies()
