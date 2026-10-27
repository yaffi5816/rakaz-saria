"""A recording as input: the routing, the cache, and the refusals.

No speech recognition runs here. Whichever engine is configured, the thing
worth testing is the same: that an audio file is recognised as one, sent to
the engine exactly once, written out as a transcript beside it, and then read
through the ordinary transcript path - and that an agent reading audio is told
it has no speaker names.

The engine itself is swapped for a fake. A suite that downloads a 1.5 GB model
and spends a minute of CPU to assert the plumbing works is a suite that stops
being run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from shared import audio, transcript
from shared.audio import Segment, Transcriber
from shared.hebcal import CalendarService
from shared.memory_store import MemoryStore
from shared.plan import DraftBook
from shared.store import LocalStore
from shared.tools import Context, ToolError, run

FIXTURE = Path(__file__).resolve().parent.parent / "shared" / "fixtures" / "meetings"


class FakeEngine(Transcriber):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, path: Path) -> list[Segment]:
        self.calls += 1
        return [
            Segment(0.0, 6.5, "דנה תכין סביבת בדיקות, שבועיים מרגע שהגיוס נסגר."),
            Segment(6.5, 11.0, "ואיתי מכין מצגת סטטוס להנהלה לקראת חמישה בדצמבר."),
        ]


@pytest.fixture
def recording(tmp_path, monkeypatch, clean):
    """A copy of the audio fixture somewhere the test may write beside it."""
    folder = tmp_path / "meetings"
    folder.mkdir()
    shutil.copy(FIXTURE / "sample-recording.wav", folder / "ישיבה.wav")
    monkeypatch.setattr(transcript, "ALLOWED_ROOTS", [folder])
    return folder / "ישיבה.wav"


@pytest.fixture
def ctx(clean):
    return Context(
        client=LocalStore(actor="sarah.avital"), memory=MemoryStore(),
        calendar=CalendarService(mode="offline"), book=DraftBook(),
    )


def test_an_audio_file_is_recognised_as_one():
    assert audio.is_audio("ישיבה.m4a")
    assert audio.is_audio(Path("C:/x/meeting.WAV"))
    assert not audio.is_audio("minutes.vtt")


def test_a_recording_is_transcribed_and_then_read(recording, ctx, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(audio, "build_transcriber", lambda: engine)

    out = run(ctx, "read_transcript", {"file": "ישיבה.wav"})

    assert engine.calls == 1
    assert "תומללה" in out and "fake" in out
    assert "דנה תכין סביבת בדיקות" in out
    assert "אין שמות דוברים" in out, "סוכן שקורא הקלטה צריך לדעת שאין ייחוס"
    assert audio.cache_path(recording).exists()


def test_the_second_read_costs_nothing(recording, ctx, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(audio, "build_transcriber", lambda: engine)

    run(ctx, "read_transcript", {"file": "ישיבה.wav"})
    second = run(ctx, "read_transcript", {"file": "ישיבה.wav"})

    assert engine.calls == 1, "תמלול שכבר נעשה לא נעשה שוב"
    assert "מהמטמון" in second
    assert "דנה תכין סביבת בדיקות" in second


def test_a_newer_recording_is_transcribed_again(recording, ctx, monkeypatch):
    """Replacing the file replaces the transcript - the cache is not a trap."""
    engine = FakeEngine()
    monkeypatch.setattr(audio, "build_transcriber", lambda: engine)
    run(ctx, "read_transcript", {"file": "ישיבה.wav"})

    cached = audio.cache_path(recording)
    import os
    os.utime(recording, (cached.stat().st_mtime + 10, cached.stat().st_mtime + 10))

    run(ctx, "read_transcript", {"file": "ישיבה.wav"})
    assert engine.calls == 2


def test_with_no_engine_the_refusal_says_what_to_do(recording, ctx, monkeypatch):
    monkeypatch.setenv("RAKAZ_STT", "off")
    with pytest.raises(ToolError) as refusal:
        run(ctx, "read_transcript", {"file": "ישיבה.wav"})
    message = str(refusal.value)
    assert "RAKAZ_STT=local" in message and "RAKAZ_STT=server" in message


def test_silence_is_reported_rather_than_invented(recording, ctx, monkeypatch):
    class Silent(Transcriber):
        name = "silent"

        def transcribe(self, path):
            return []

    monkeypatch.setattr(audio, "build_transcriber", lambda: Silent())
    with pytest.raises(ToolError, match="לא זוהה דיבור"):
        run(ctx, "read_transcript", {"file": "ישיבה.wav"})


def test_a_file_outside_the_meeting_folders_is_refused(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "ALLOWED_ROOTS", [tmp_path / "meetings"])
    stray = tmp_path / "secrets.wav"
    stray.write_bytes(b"\x00")
    with pytest.raises(ToolError):
        run(ctx, "read_transcript", {"file": str(stray)})


def test_the_vtt_written_from_audio_is_readable_as_a_transcript(recording, monkeypatch):
    engine = FakeEngine()
    path, report = audio.transcribe_to_vtt(recording, engine)

    assert report["engine"] == "fake" and report["segments"] == 2
    body = path.read_text(encoding="utf-8")
    assert body.startswith("WEBVTT")
    assert "00:00:06.500 --> 00:00:11.000" in body

    text, read_report = transcript.read(str(path))
    assert read_report["cues"] == 2
    assert read_report["speakers"] == [], "הקלטה לא מביאה שמות דוברים"
    assert "מצגת סטטוס" in text
