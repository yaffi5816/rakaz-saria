"""The memory file: append-only, retrievable, and able to read Hebrew dates."""

from __future__ import annotations

from shared.memory_store import MemoryStore, derive_guard


def test_release_appends_and_keeps_the_original(clean):
    memory = MemoryStore()
    first = memory.remember("decision", "שלב ב", "לא פותחים לפני ינואר 2027")
    memory.remember("convention", "בדיקת שאלון", "נועה אחראית")
    memory.release(first["id"], "ההנהלה אישרה להקדים")

    records = memory.records()
    assert len(records) == 3, "ביטול חייב להוסיף שורה, לא למחוק"
    original = next(r for r in records if r["id"] == first["id"])
    assert original["rule"] == "לא פותחים לפני ינואר 2027"
    assert original["status"] == "active", "הרשומה המקורית לא משתנה"

    active_ids = [r["id"] for r in memory.active()]
    assert first["id"] not in active_ids, "אחרי ביטול היא כבר לא פעילה"
    assert len(active_ids) == 1


def test_only_relevant_records_come_back(clean):
    memory = MemoryStore()
    memory.remember("convention", "בדיקת שאלון", "נועה לוי אחראית ואבי מזרחי שותף")
    memory.remember("convention", "הזמנת ציוד", "דרך הרכש בלבד")
    memory.remember("decision", "שלב ב", "לא לפני ינואר 2027", {"type": "not_before",
                                                                "date": "2027-01-01"})

    hits = memory.relevant("צריך משימת בדיקת שאלון למודול שינה")
    assert [r["topic"] for r in hits] == ["בדיקת שאלון"]
    assert memory.relevant("מתי קונים מחשבים") == []


def test_a_decision_outranks_a_convention_on_a_tie(clean):
    memory = MemoryStore()
    memory.remember("convention", "שלב ב", "נכנס לרצועת שדה")
    memory.remember("decision", "שלב ב", "לא לפני ינואר 2027")
    hits = memory.relevant("שלב ב")
    assert hits[0]["kind"] == "decision"


def test_guards_are_derived_when_the_model_omits_them():
    assert derive_guard("לא פותחים את שלב ב' לפני ינואר 2027") == {
        "type": "not_before", "date": "2027-01-01"
    }
    assert derive_guard("אין לפתוח לפני 1.1.2027") == {
        "type": "not_before", "date": "2027-01-01"
    }
    assert derive_guard("נועה אחראית על בדיקות שאלון") is None


def test_guards_only_list_active_records(clean):
    memory = MemoryStore()
    decision = memory.remember("decision", "שלב ב", "לא לפני ינואר 2027")
    assert len(memory.guards()) == 1
    memory.release(decision["id"], "בוטל")
    assert memory.guards() == []
