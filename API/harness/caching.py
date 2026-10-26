"""Where the cache breakpoints go, and what must never move above them.

Caching on this API is a prefix match: the request is rendered as tools, then
system, then messages, and a single changed byte anywhere in that prefix
invalidates everything after it. So the decision is not "should we cache" - it
is "which bytes are we willing to freeze for the whole session".

What is frozen here: the tool schemas and the system prompt, including the
department profile. The profile is memory read from disk, which looks like
something that belongs in the per-turn context, and putting it there would be a
mistake twice over - it would be re-sent in full on every turn, and it would
move bytes that sit above the breakpoint, so every turn would also pay to write
the cache again.

What stays below the breakpoint: the conversation, and the per-turn context
block that carries the memories relevant to this turn. Those change constantly,
which is exactly why they are downstream of the marker.

One breakpoint, on the last system block, covers tools plus system. The
conversation itself is left unmarked: it grows by an appended turn each time, so
the automatic prefix reuse handles it, and a second marker would buy an extra
write for bytes that are already read back as part of the same prefix.

The same prefix discipline is what keeps the history valid on Claude Opus 5.5
for a second reason: the model binds its thinking blocks to the conversation
that produced them, so a system prompt edited mid-session would not merely cost
a cache write, it would reject the request.
"""

from __future__ import annotations

from API.harness import say


def system_blocks(system_prompt: str, profile: str) -> list[dict]:
    """Build the frozen system prefix, with the one cache breakpoint on its end.

    Input:
        system_prompt: the agent's instructions, identical on every turn.
        profile: memory/profile.md - the department's working style.

    Output: system blocks ready to pass to messages.create, the last one
    carrying cache_control.
    """
    blocks: list[dict] = [{"type": "text", "text": system_prompt}]
    if profile:
        blocks.append({"type": "text", "text": "רקע קבוע על האגף:\n" + profile})
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    say("caching.py", f"cache breakpoint on system block {len(blocks)} (frozen for the session)")
    return blocks


def report(usage) -> None:
    """Say what the cache did on one model call, so a miss is visible as it happens."""
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    if read:
        say("caching.py", f"cache hit: {read} tokens read, {written} written")
    elif written:
        say("caching.py", f"cache miss: wrote {written} tokens")
