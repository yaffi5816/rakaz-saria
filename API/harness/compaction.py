"""What happens when the conversation outgrows its budget.

The shape of compaction here is deliberate and not the obvious one. The obvious
implementation keeps the last few turns verbatim and summarises everything
before them. On Claude Opus 5.5 that is a request that can be rejected: the
model binds each thinking block to the conversation that produced it, so
replaying a recent assistant turn whose prefix has just been rewritten
invalidates the block. The published guidance names the safe client-side shape
exactly - replace the whole history with a summary and replay no earlier
thinking - and that is what this does.

It is also the cheaper shape. A summary of the entire conversation is a few
hundred tokens; a summary plus three verbatim turns is a few thousand, and the
verbatim turns are the part that was about to be re-read anyway.

The threshold is low on purpose. A scenario of eight turns would never cross a
realistic 150K boundary, and a compaction file that never runs is a file that
cannot be compared against the SDK's automatic one - so RAKAZ_COMPACT_THRESHOLD
is set where the run will cross it, and results/api.json records when it fired.
"""

from __future__ import annotations

import json
import os

from API.harness import say

# Hebrew runs about 2.2 characters to the token on this tokenizer. Used only to
# decide when to compact; every number that reaches the results file comes from
# the API's own usage, never from this estimate.
CHARS_PER_TOKEN = 2.2


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            total += len(content)
        else:
            total += len(json.dumps(content, ensure_ascii=False))
    return int(total / CHARS_PER_TOKEN)


def threshold() -> int:
    return int(os.environ.get("RAKAZ_COMPACT_THRESHOLD", "2000"))


def transcript(messages: list[dict]) -> str:
    """Flatten the conversation into something a summariser can read."""
    lines = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            lines.append(f"{message['role']}: {content}")
            continue
        for block in content:
            kind = block.get("type")
            if kind == "text":
                lines.append(f"{message['role']}: {block['text']}")
            elif kind == "tool_use":
                lines.append(
                    f"{message['role']} קרא ל-{block['name']} עם "
                    f"{json.dumps(block.get('input', {}), ensure_ascii=False)[:300]}"
                )
            elif kind == "tool_result":
                body = block.get("content")
                if not isinstance(body, str):
                    body = json.dumps(body, ensure_ascii=False)
                lines.append(f"תוצאת כלי: {body[:400]}")
    return "\n".join(lines)


SUMMARY_PROMPT = """להלן תמליל של שיחה בין מנהלת אגף לעוזר ניהולי.
כתוב סיכום קצר בעברית שישמר במקום השיחה עצמה. חובה לכלול:
- מה סוכם ומה כבר נכתב למערכת, כולל מזהי משימות וטיוטות
- כללים והחלטות שנשמרו, עם המזהים שלהם
- מה עדיין פתוח
אל תכלול פרטים שאפשר לשלוף שוב מהמערכת. עד 180 מילים."""


def maybe_compact(messages: list[dict], history, summarise) -> list[dict]:
    """Compact the conversation when it crosses the threshold.

    Input:
        messages: what would otherwise be sent this turn.
        history: the append-only transcript, which records the marker.
        summarise: a callable taking a prompt and returning the summary text.
            Injected so the suite can exercise this file without a model.

    Output: the messages to send - unchanged, or a single summary message.
    """
    size = estimate_tokens(messages)
    limit = threshold()
    if size <= limit:
        say("compaction.py", f"~{size} tokens, under the {limit} threshold - no compaction")
        return messages
    say("compaction.py", f"~{size} tokens, over the {limit} threshold - compacting")
    summary = summarise(SUMMARY_PROMPT + "\n\n---\n" + transcript(messages))
    history.append_compaction(summary, replaced=len(messages))
    say("compaction.py", f"history replaced by a {len(summary)}-char summary")
    return [{"role": "user", "content": f"סיכום מה שהיה עד כה בשיחה:\n{summary}"}]
