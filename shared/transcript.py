"""Reading a meeting the way the office already produces it.

"Upload the recording" almost always means something cheaper than it sounds.
Teams and Zoom already transcribe every meeting they record, and what lands in
the folder afterwards is a .vtt or .docx of Hebrew text with speaker names on
it. So the agent does not need speech recognition to accept a meeting - it
needs to read a file, which is a tool that works offline, costs nothing, keeps
the audio inside the building, and is testable.

Speech to text stays possible and stays out of here. If a recording ever
arrives with no transcript beside it, the fix is one step in front of this
module - something that turns audio into the same text this function returns -
and nothing downstream changes. That is the whole reason this is a separate
file with a narrow output.

Timestamps and cue numbers are dropped on the way in. A VTT of a twenty-minute
meeting is a few thousand tokens of clock readings that say nothing about who
owes what to whom, and a model asked to plan from it pays for every one.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEETINGS = ROOT / "shared" / "fixtures" / "meetings"

# Where a transcript may be read from. Anything outside these is refused, so a
# path arriving in a tool call cannot be turned into a file read somewhere else
# on the machine.
ALLOWED_ROOTS = [MEETINGS, ROOT / "meetings"]

SUFFIXES = {".vtt", ".txt", ".md", ".srt"}

_CUE_NUMBER = re.compile(r"^\d+$")
_TIMING = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}\s*-->")
_SPEAKER = re.compile(r"^<v\s+([^>]+)>(.*)$")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def resolve(given: str, suffixes: set[str] | None = None) -> Path:
    """Turn a path from a tool call into a file we are willing to open.

    Input:
        given: whatever the model passed - a bare name, or a path.
        suffixes: what counts as readable. Defaults to the text transcripts;
            the audio path passes a wider set, because a recording is resolved
            here first and transcribed afterwards.

    Output: the resolved file.

    Raises FileNotFoundError when it does not exist, and ValueError when it
    sits outside the meeting folders or carries an extension this cannot read.
    """
    allowed = suffixes or SUFFIXES
    candidate = Path(given.strip().strip('"'))
    tries = [candidate] if candidate.is_absolute() else [root / candidate for root in ALLOWED_ROOTS]
    for path in tries:
        resolved = path.resolve()
        if not any(_within(resolved, root) for root in ALLOWED_ROOTS):
            continue
        if resolved.exists():
            if resolved.suffix.lower() not in allowed:
                raise ValueError(
                    f"{resolved.name}: אפשר לקרוא רק {', '.join(sorted(allowed))}."
                )
            return resolved
    raise FileNotFoundError(
        f"לא נמצא תמלול בשם {given}. הקבצים הזמינים: {', '.join(available()) or '(אין)'}"
    )


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def available(suffixes: set[str] | None = None) -> list[str]:
    allowed = suffixes or SUFFIXES
    names = []
    for root in ALLOWED_ROOTS:
        if root.exists():
            names += sorted(p.name for p in root.iterdir() if p.suffix.lower() in allowed)
    return names


def read(given: str) -> tuple[str, dict]:
    """Read a transcript and return it as plain speaker-labelled text.

    Output: the cleaned text, and a small report on what was dropped - which
    is what lets the agent say "קראתי תמלול של 9 קטעים" instead of pretending
    it heard the meeting.
    """
    path = resolve(given)
    raw = path.read_text(encoding="utf-8-sig")
    lines: list[str] = []
    speakers: list[str] = []
    cues = 0
    current: str | None = None

    for line in raw.splitlines():
        text = line.strip()
        if not text or text.upper().startswith("WEBVTT") or text.startswith("NOTE "):
            continue
        if _CUE_NUMBER.match(text) or _TIMING.match(text):
            if _TIMING.match(text):
                cues += 1
            continue
        speaker = _SPEAKER.match(text)
        if speaker:
            current = speaker.group(1).strip()
            if current not in speakers:
                speakers.append(current)
            text = speaker.group(2).strip()
        text = _TAG.sub("", text).strip()
        if not text:
            continue
        if current and (not lines or not lines[-1].startswith(f"{current}:")):
            lines.append(f"{current}: {text}")
        elif current:
            lines[-1] = f"{lines[-1]} {text}"
        else:
            lines.append(text)

    return "\n".join(lines), {
        "file": path.name,
        "cues": cues,
        "speakers": speakers,
        "chars_in": len(raw),
        "chars_out": sum(len(line) for line in lines) + len(lines),
    }
