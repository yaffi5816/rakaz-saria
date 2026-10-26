"""The hand-written harness, one file per responsibility.

Every file here answers exactly one question:

    loop.py        when does the conversation stop?
    tools.py       how does a model request become a function call?
    history.py     what is kept, and where?
    memory.py      what does the agent know between sessions?
    context.py     what goes into this turn?
    compaction.py  what happens when the conversation outgrows the window?
    caching.py     which bytes are stable enough to cache?
    tokens.py      what did that cost?

The SDK version of this agent has none of these files. Each line there that
replaces one carries a `# replaces:` comment naming it, so the two columns of
the table in REPORT.md are produced by counting rather than by argument.

say() is here rather than in a ninth file because it is not a responsibility -
it is how the other eight announce themselves. Every runtime action prints the
file that performed it, which is what makes a scenario run readable as a
description of the harness rather than of the agent.
"""

from __future__ import annotations

import sys

# Everything printed this run, so results/api.json can carry the trace without
# the caller having to capture stdout.
TRACE: list[str] = []
QUIET = False


def say(module: str, message: str) -> None:
    """Print one runtime action, attributed to the file that performed it."""
    line = f"[harness/{module}] {message}"
    TRACE.append(line)
    if not QUIET:
        print(line, file=sys.stdout, flush=True)


def reset_trace() -> None:
    TRACE.clear()
