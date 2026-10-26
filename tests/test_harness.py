"""The eight harness files, each tested on the one thing it owns."""

from __future__ import annotations

import pytest

from API.harness import caching, compaction, loop
from API.harness import tools as harness_tools
from API.harness.history import History
from API.harness.tokens import Meter, TurnUsage
from shared.hebcal import CalendarService
from shared.memory_store import MemoryStore
from shared.plan import DraftBook
from shared.store import LocalStore
from shared.tools import Context
from tests.fake_model import FakeResponse, FakeUsage, TextBlock, ToolUseBlock


@pytest.fixture
def ctx(clean):
    return Context(
        client=LocalStore(actor="sarah.avital"),
        memory=MemoryStore(),
        calendar=CalendarService(mode="offline"),
        book=DraftBook(),
    )


# ── caching.py ───────────────────────────────────────────────────────


def test_the_breakpoint_sits_on_the_last_system_block():
    blocks = caching.system_blocks("הוראות", "פרופיל האגף")
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0], "רק הבלוק האחרון נושא נקודת cache"
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_the_profile_is_frozen_into_the_prefix():
    """Two builds of the same session produce byte-identical system blocks."""
    assert caching.system_blocks("a", "b") == caching.system_blocks("a", "b")


# ── tools.py ─────────────────────────────────────────────────────────


def test_a_refusing_tool_comes_back_as_an_error_result(ctx):
    block = ToolUseBlock(id="tu1", name="workload", input={})
    result = harness_tools.dispatch(ctx, block)
    assert result["is_error"] is True
    assert result["tool_use_id"] == "tu1"
    assert "שמות" in result["content"]


def test_an_unknown_tool_does_not_end_the_run(ctx):
    result = harness_tools.dispatch(ctx, ToolUseBlock(id="tu2", name="nope", input={}))
    assert result["is_error"] is True


def test_every_shared_tool_is_offered():
    names = [schema["name"] for schema in harness_tools.schemas()]
    assert names == harness_tools.ALLOWED
    assert len(names) == 9


# ── loop.py ──────────────────────────────────────────────────────────


def test_all_tool_results_return_in_one_user_message(ctx, clean):
    history = History("loop-test")
    replies = [
        FakeResponse(
            content=[
                ToolUseBlock(id="a", name="search_workspace", input={"kind": "projects"}),
                ToolUseBlock(id="b", name="calendar_check", input={"dates": ["2026-11-21"]}),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[TextBlock(text="סיימתי")], stop_reason="end_turn"),
    ]
    messages: list[dict] = []
    text, cut_off = loop.run_turn(lambda _: replies.pop(0), ctx, history, messages)

    assert text == "סיימתי"
    assert cut_off is False
    results = [m for m in messages if m["role"] == "user"]
    assert len(results) == 1, "שתי קריאות כלים חוזרות בהודעה אחת"
    assert [b["tool_use_id"] for b in results[0]["content"]] == ["a", "b"]


def test_the_loop_stops_on_stop_reason_and_not_on_content(ctx, clean):
    history = History("loop-stop")
    replies = [FakeResponse(content=[TextBlock(text="")], stop_reason="end_turn")]
    text, cut_off = loop.run_turn(lambda _: replies.pop(0), ctx, history, [])
    assert text == "" and cut_off is False


# ── history.py ───────────────────────────────────────────────────────


def test_the_transcript_is_append_only_across_a_compaction(clean):
    history = History("history-test")
    history.append({"role": "user", "content": "ראשון"})
    history.append({"role": "assistant", "content": [{"type": "text", "text": "תשובה"}]})
    history.append_compaction("סיכום", replaced=2)
    history.append({"role": "user", "content": "אחרי"})

    raw = history.path.read_text(encoding="utf-8")
    assert "ראשון" in raw, "שורות ישנות נשארות בקובץ"

    replayed = history.replay()
    assert len(replayed) == 2
    assert replayed[0]["content"].startswith("סיכום מה שהיה")
    assert replayed[1]["content"] == "אחרי"


def test_a_thinking_block_keeps_its_signature_on_the_way_to_disk(clean):
    from API.harness.history import to_message

    class Thinking:
        type = "thinking"
        thinking = "..."
        signature = "sig-123"

    stored = to_message("assistant", [Thinking()])
    assert stored["content"][0] == {
        "type": "thinking", "thinking": "...", "signature": "sig-123"
    }


# ── compaction.py ────────────────────────────────────────────────────


def test_compaction_replaces_the_whole_history_and_replays_no_thinking(clean, monkeypatch):
    monkeypatch.setenv("RAKAZ_COMPACT_THRESHOLD", "10")
    history = History("compaction-test")
    messages = [
        {"role": "user", "content": "שאלה ארוכה " * 50},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x", "signature": "s"},
            {"type": "text", "text": "תשובה"},
        ]},
    ]
    out = compaction.maybe_compact(messages, history, lambda prompt: "סיכום קצר")

    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "סיכום קצר" in out[0]["content"]
    assert "thinking" not in str(out), "אחרי compaction לא משחזרים בלוקים של חשיבה"


def test_a_short_conversation_is_left_alone(clean, monkeypatch):
    monkeypatch.setenv("RAKAZ_COMPACT_THRESHOLD", "100000")
    history = History("compaction-skip")
    messages = [{"role": "user", "content": "קצר"}]
    assert compaction.maybe_compact(messages, history, lambda p: "לא אמור לרוץ") is messages


# ── tokens.py ────────────────────────────────────────────────────────


def test_the_four_token_classes_are_priced_apart():
    usage = TurnUsage(turn=1, input_tokens=1_000_000)
    assert usage.cost_usd == pytest.approx(4.00)
    cached = TurnUsage(turn=1, cache_read_tokens=1_000_000)
    assert cached.cost_usd == pytest.approx(0.20)
    out = TurnUsage(turn=1, output_tokens=1_000_000)
    assert out.cost_usd == pytest.approx(20.00)


def test_the_meter_folds_several_model_calls_into_one_turn():
    meter = Meter()
    meter.start_turn(1)
    meter.record(FakeUsage(input_tokens=100, output_tokens=10))
    meter.record(FakeUsage(input_tokens=200, output_tokens=20,
                           cache_read_input_tokens=500))
    turn = meter.end_turn()
    assert turn.model_calls == 2
    assert turn.input_tokens == 300
    assert turn.cache_read_tokens == 500
    assert meter.totals()["model_calls"] == 2
