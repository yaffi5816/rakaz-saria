"""Turning a recording into a transcript, with the engine as a replaceable part.

A recording of an internal meeting is not the same kind of data as a task
title. The title is a line of text the agent sends to a model anyway; the
recording is every word five people said in a closed room, including the parts
that were not meant for the minutes. So the question is not "which speech API
is best" but "whose machine hears it", and this module is written so that
answer is a setting rather than a rewrite.

Two engines ship:

  local   faster-whisper, running in this process. Nothing leaves the machine.
          Slow on a CPU - roughly as long as the meeting itself on a laptop -
          and the model downloads once, which on a filtered connection is its
          own small adventure. This is the default, because the safe answer
          should be the one you get without configuring anything.

  server  an OpenAI-compatible /audio/transcriptions endpoint. That is the
          shape whisper.cpp's server, faster-whisper-server and an Azure
          OpenAI Whisper deployment all speak, so a transcription service the
          organisation runs on its own network needs a URL here and nothing
          else. Audio leaves this machine and reaches that host, and no
          further - which is the arrangement a security officer can actually
          approve.

Both write a .vtt beside the recording and never transcribe the same file
twice. That cache is what makes a second run of a scenario free, and it is
also how a demonstration stays fast: transcribe once in the morning, and the
recording behaves like a transcript for the rest of the day.

What neither engine does is tell speakers apart. Whisper returns what was
said, not who said it, so a transcript made from audio carries no names in
front of its lines, and the agent has to take attribution from the words
themselves - "דנה תכין" rather than from a label. Teams' own transcript does
carry the names, which is one more reason to prefer it when it exists.
"""

from __future__ import annotations

import datetime
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

AUDIO_SUFFIXES = {
    ".m4a", ".mp3", ".wav", ".ogg", ".opus", ".aac", ".amr", ".flac", ".wma", ".mp4", ".webm",
}

# Hebrew, stated rather than detected: a meeting that opens with thirty seconds
# of English small talk gets mis-detected, and then the whole hour comes back
# transliterated.
LANGUAGE = "he"


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber(ABC):
    name = "none"

    @abstractmethod
    def transcribe(self, path: Path) -> list[Segment]: ...


class LocalWhisper(Transcriber):
    """faster-whisper in this process. The recording stays on this machine."""

    name = "local"

    def __init__(self, size: str | None = None) -> None:
        # medium is the smallest size that is honestly usable for Hebrew. large-v3
        # is better and roughly three times slower; both are set from the
        # environment rather than argued about here.
        self.size = size or os.environ.get("RAKAZ_STT_MODEL", "medium")

    def transcribe(self, path: Path) -> list[Segment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "זיהוי דיבור מקומי דורש faster-whisper:\n"
                "    pip install -r requirements-audio.txt\n"
                "או להגדיר RAKAZ_STT=server ולכוון ל-RAKAZ_STT_URL."
            ) from error

        model = WhisperModel(self.size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(path), language=LANGUAGE, vad_filter=True)
        return [Segment(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]


class WhisperServer(Transcriber):
    """An OpenAI-compatible /audio/transcriptions endpoint on your own network."""

    name = "server"

    def __init__(self, url: str | None = None, key: str | None = None,
                 model: str | None = None) -> None:
        self.url = (url or os.environ.get("RAKAZ_STT_URL", "")).rstrip("/")
        self.key = key or os.environ.get("RAKAZ_STT_KEY", "")
        self.model = model or os.environ.get("RAKAZ_STT_MODEL", "whisper-1")
        if not self.url:
            raise RuntimeError("RAKAZ_STT=server דורש גם RAKAZ_STT_URL")

    def transcribe(self, path: Path) -> list[Segment]:
        import httpx

        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        with path.open("rb") as handle:
            response = httpx.post(
                f"{self.url}/audio/transcriptions",
                headers=headers,
                files={"file": (path.name, handle, "application/octet-stream")},
                data={"model": self.model, "language": LANGUAGE,
                      "response_format": "verbose_json"},
                timeout=900.0,
            )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("segments")
        if not rows:
            text = (payload.get("text") or "").strip()
            return [Segment(0.0, 0.0, text)] if text else []
        return [
            Segment(float(row.get("start", 0)), float(row.get("end", 0)),
                    (row.get("text") or "").strip())
            for row in rows
            if (row.get("text") or "").strip()
        ]


def build_transcriber() -> Transcriber | None:
    """The engine the environment asks for, or None when audio is turned off."""
    kind = os.environ.get("RAKAZ_STT", "local").strip().lower()
    if kind in ("", "off", "none", "false"):
        return None
    if kind == "server":
        return WhisperServer()
    return LocalWhisper()


def _stamp(seconds: float) -> str:
    moment = datetime.timedelta(seconds=max(0.0, seconds))
    hours, rest = divmod(int(moment.total_seconds()), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{int(moment.microseconds / 1000):03d}"


def to_vtt(segments: list[Segment], source: str, engine: str) -> str:
    """Render segments as WebVTT, the same shape Teams writes."""
    lines = ["WEBVTT", "", f"NOTE תומלל מתוך {source} על ידי {engine}", ""]
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{_stamp(segment.start)} --> {_stamp(segment.end)}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines)


def cache_path(audio: Path) -> Path:
    return audio.with_suffix(audio.suffix + ".vtt")


def is_audio(path: Path | str) -> bool:
    return Path(path).suffix.lower() in AUDIO_SUFFIXES


def transcribe_to_vtt(audio: Path, transcriber: Transcriber | None = None) -> tuple[Path, dict]:
    """Produce the transcript for a recording, reusing one made earlier.

    Input:
        audio: the recording, already resolved to a file we may read.
        transcriber: the engine, or None to build one from the environment.

    Output: the path of the .vtt, and a small report naming the engine, how
    many segments came out, and whether this call did the work or the cache
    did - which is what lets the agent say so instead of implying it listened.
    """
    cached = cache_path(audio)
    if cached.exists() and cached.stat().st_mtime >= audio.stat().st_mtime:
        return cached, {"engine": "cache", "segments": None, "fresh": False,
                        "source": audio.name}

    engine = transcriber or build_transcriber()
    if engine is None:
        raise RuntimeError(
            f"{audio.name} הוא הקלטה, וזיהוי דיבור כבוי (RAKAZ_STT=off).\n"
            "אפשרויות: להעלות תמלול במקום, או להדליק מנוע - "
            "RAKAZ_STT=local (מקומי) או RAKAZ_STT=server עם RAKAZ_STT_URL."
        )

    segments = engine.transcribe(audio)
    if not segments:
        raise RuntimeError(f"{audio.name}: לא זוהה דיבור בהקלטה.")

    cached.write_text(to_vtt(segments, audio.name, engine.name), encoding="utf-8")
    return cached, {"engine": engine.name, "segments": len(segments), "fresh": True,
                    "source": audio.name}
