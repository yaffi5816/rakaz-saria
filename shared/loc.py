"""Counting the two harnesses, so REPORT.md's size column is measured.

The API side is counted per file: lines of code in API/harness/<name>.py,
excluding blanks and comment-only lines, since docstrings in this project run
long on purpose and counting them would say more about my writing than about
the harness.

The SDK side is counted by marker. Every line in SDK/ that does the job of a
harness file carries `# replaces: <name>`, and this counts those lines against
the file they replace. That is the honest comparison - not "the SDK is three
files", but "the eleven lines that stand in for the eight files, named one by
one".
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "API" / "harness"
SDK = ROOT / "SDK"

RESPONSIBILITIES = [
    "loop.py", "tools.py", "history.py", "memory.py",
    "context.py", "compaction.py", "caching.py", "tokens.py",
]


def _code_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    in_doc = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        triples = line.count('"""')
        if in_doc:
            if triples:
                in_doc = False
            continue
        if line.startswith('"""'):
            if triples == 1:
                in_doc = True
            continue
        if line.startswith("#"):
            continue
        count += 1
    return count


def api_lines() -> dict[str, int]:
    return {name: _code_lines(HARNESS / name) for name in RESPONSIBILITIES}


def sdk_lines() -> dict[str, int]:
    """Lines in SDK/ marked as replacing each harness file."""
    marker = re.compile(r"#\s*replaces:\s*([\w./]+)")
    counts = {name: 0 for name in RESPONSIBILITIES}
    for path in sorted(SDK.glob("*.py")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            found = marker.search(raw)
            if not found:
                continue
            name = found.group(1).split("/")[-1]
            if name in counts:
                counts[name] += 1
    return counts


def table() -> dict:
    api, sdk = api_lines(), sdk_lines()
    return {
        "api_harness": api,
        "api_total": sum(api.values()),
        "sdk_replacing_lines": sdk,
        "sdk_total": sum(sdk.values()),
        "shared_total": sum(
            _code_lines(p) for p in sorted((ROOT / "shared").glob("*.py"))
        ),
    }
