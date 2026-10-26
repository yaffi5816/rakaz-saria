"""The conversation on disk: append-only, and replayed from the last compaction.

One file per session, one JSON object per line, opened in append mode and never
rewritten. That is a property worth paying for: a crash halfway through a turn
leaves a readable transcript, two processes writing the same session cannot lose
each other's turns, and the file is evidence of what the model was actually sent
rather than a reconstruction.

Compaction does not break the rule. When the conversation outgrows its budget a
marker line is appended carrying the summary, and replay() starts from the last
marker instead of the top. Nothing is deleted; the summarised turns are still in
the file, they simply stop being replayed.

Blocks are stored in their minimal form rather than through model_dump(), so a
replayed assistant turn carries exactly the fields the API expects - including
the signature on a thinking block, which is what makes the block valid when it
goes back.
"""

from __future__ import annotations

import json
from pathlib import Path

from API.harness import say

ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_DIR = ROOT / "state" / "history"


def _block_to_dict(block) -> dict:
    """Reduce one content block to the fields the API reads back."""
    if isinstance(block, dict):
        return block
    kind = getattr(block, "type", None)
    if kind == "text":
        return {"type": "text", "text": block.text}
    if kind == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    dumped = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)
    return dumped


def to_message(role: str, content) -> dict:
    if isinstance(content, str):
        return {"role": role, "content": content}
    return {"role": role, "content": [_block_to_dict(b) for b in content]}


class History:
    def __init__(self, session: str) -> None:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        self.session = session
        self.path = HISTORY_DIR / f"{session}.jsonl"

    def append(self, message: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "message", **message}, ensure_ascii=False) + "\n")

    def append_compaction(self, summary: str, replaced: int) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"kind": "compaction", "summary": summary, "replaced": replaced},
                    ensure_ascii=False,
                )
                + "\n"
            )
        say("history.py", f"compaction marker appended, {replaced} messages retired")

    def _lines(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def replay(self) -> list[dict]:
        """The messages to send this turn: everything after the last compaction.

        A compacted session starts from a single user message holding the
        summary. No assistant turn from before the marker comes back, and with
        it no thinking block from before the marker - which is what makes this
        shape of compaction safe on a model that binds thinking blocks to the
        conversation that produced them.
        """
        lines = self._lines()
        start = 0
        summary = None
        for index, line in enumerate(lines):
            if line.get("kind") == "compaction":
                start = index + 1
                summary = line["summary"]
        messages = [
            {"role": line["role"], "content": line["content"]}
            for line in lines[start:]
            if line.get("kind") == "message"
        ]
        if summary:
            messages = [
                {"role": "user", "content": f"סיכום מה שהיה עד כה בשיחה:\n{summary}"}
            ] + messages
        say(
            "history.py",
            f"replaying {len(messages)} messages from {self.path.name}"
            + (" (after compaction)" if summary else ""),
        )
        return messages

    def message_count(self) -> int:
        return sum(1 for line in self._lines() if line.get("kind") == "message")
