"""The SDK side: its configuration, and the markers the size table counts.

The marker test runs everywhere. The configuration test needs the SDK package,
and skips without it, because the suite's promise is that it runs offline with
no key - a test that fails on a machine where the 129 MB wheel has not
finished downloading would break that promise for no gain.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from shared import loc
from shared import tools as shared_tools

ROOT = Path(__file__).resolve().parent.parent


def test_every_harness_file_is_accounted_for():
    """Each of the eight is either replaced by marked lines, or deliberately not.

    Three have no replacement and that is the finding, not an oversight: the
    SDK does compaction and caching with no configuration at all, and offers
    no pre-turn hook where context.py's work could go.
    """
    table = loc.table()
    automatic = {"compaction.py", "caching.py", "context.py"}
    for name, lines in table["sdk_replacing_lines"].items():
        if name in automatic:
            assert lines == 0, f"{name} אמור להיות אוטומטי בצד ה-SDK"
        else:
            assert lines > 0, f"אין שורה מסומנת ב-SDK שמחליפה את {name}"
    assert table["api_total"] > 10 * table["sdk_total"]


def test_markers_name_real_files():
    marker = re.compile(r"#\s*replaces:\s*harness/([\w.]+)")
    found = set()
    for path in (ROOT / "SDK").glob("*.py"):
        found |= set(marker.findall(path.read_text(encoding="utf-8")))
    assert found, "אין סימוני replaces בכלל"
    for name in found:
        assert (ROOT / "API" / "harness" / name).exists(), f"{name} לא קיים"


try:  # noqa: SIM105
    import claude_agent_sdk  # noqa: F401

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

needs_sdk = pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk לא מותקן בסביבה הזאת")


@needs_sdk
def test_the_configuration_says_what_the_exercise_requires():
    from SDK.agent import EFFORT, MODEL, options

    config = options()
    assert config.strict_mcp_config is True
    assert config.setting_sources == ["project"]
    assert config.system_prompt.startswith('אתה "רכז"'), "system prompt משלי, לא של Claude Code"
    assert config.tools == [], "הכלים המובנים מוסרים - רשימה מצומצמת"
    assert config.model == MODEL and config.effort == EFFORT
    assert list(config.mcp_servers) == ["rakaz"]


@needs_sdk
def test_the_sdk_offers_exactly_the_shared_eight():
    from SDK import tools as sdk_tools

    expected = [f"mcp__rakaz__{spec.name}" for spec in shared_tools.TOOLS]
    assert sdk_tools.NAMES == expected
    assert len(sdk_tools.TOOLS) == 8


@needs_sdk
def test_a_refusing_tool_is_marked_is_error(clean):
    """The SDK side turns a ToolError into is_error, as the API side does."""
    from SDK import tools as sdk_tools
    from shared.hebcal import CalendarService
    from shared.memory_store import MemoryStore
    from shared.plan import DraftBook
    from shared.store import LocalStore
    from shared.tools import Context

    sdk_tools.use(Context(
        client=LocalStore(actor="sarah.avital"), memory=MemoryStore(),
        calendar=CalendarService(mode="offline"), book=DraftBook(),
    ))
    workload = next(t for t in sdk_tools.TOOLS if t.name == "workload")
    result = asyncio.run(workload.handler({}))
    assert result["is_error"] is True
