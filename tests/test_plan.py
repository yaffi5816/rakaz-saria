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
    _, problem = planning.commit(book, client, draft.id)
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

    committed, problem = planning.commit(book, client, draft.id)
    assert problem == ""
    assert len(committed.committed) == 1
    created = client.created_by_agent()[0]
    assert created["owners"] == ["דנה שפירא"]
    assert (202, created["id"]) in client.all_dependencies()


def test_a_draft_cannot_be_committed_twice(table):
    book, calendar, memory, client = table
    draft = planning.build(book, calendar, memory, client, [
        {"title": "משימה", "owners": ["רחל כהן"], "due_date": "2026-11-26"},
    ])
    planning.commit(book, client, draft.id)
    _, problem = planning.commit(book, client, draft.id)
    assert "כבר נכתבה" in problem


def test_a_cycle_is_refused(table):
    _, _, _, client = table
    with pytest.raises(ValueError, match="מעגל"):
        client.link(105, 101)
