"""What survives between sessions, and how little of it is spent per turn.

The store itself is shared code - the same append-only conventions file serves
both versions of the agent. What lives here is the harness half: loading the
profile once at startup so caching.py can freeze it into the system prefix, and
selecting the few records worth sending on a given turn.

That selection is the whole difference between an agent that remembers and an
agent that recites. A department accumulates rules indefinitely, and a turn is
about one or two of them; sending all of them every turn would be correct,
expensive, and would still bury the relevant one in the middle of a list.

The SDK version keeps the first half and loses the second: setting_sources
["project"] loads CLAUDE.md for free, and from then on the model retrieves
through the tools, with no equivalent place to decide what is injected
unasked. Both behaviours are defensible; only one of them is mine to tune.
"""

from __future__ import annotations

from shared.memory_store import MemoryStore
from shared.memory_store import render as render_records
from API.harness import say


class Memory:
    def __init__(self) -> None:
        self.store = MemoryStore()

    def profile(self) -> str:
        text = self.store.profile()
        say("memory.py", f"profile loaded: {len(text)} chars (frozen into the system prefix)")
        return text

    def for_turn(self, text: str, limit: int = 4) -> str:
        """The records this turn should see, chosen before the model asks.

        Input:
            text: what the person just said.
            limit: how many records may be injected unasked.

        Output: the compact rendering, or an empty string when nothing matches.
        """
        records = self.store.relevant(text, limit=limit)
        total = len(self.store.active())
        if not records:
            say("memory.py", f"0 of {total} records relevant to this turn - nothing injected")
            return ""
        say(
            "memory.py",
            f"{len(records)} of {total} records injected: "
            + ", ".join(f"{r['id']}({r['topic']})" for r in records),
        )
        return render_records(records)

    def active_count(self) -> int:
        return len(self.store.active())
