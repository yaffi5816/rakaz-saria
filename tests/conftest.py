"""Test-wide setup: frozen clock, offline calendar, seeded workspace, no key.

Every test runs against the bundled fixtures. The calendar never reaches the
network, the workspace is the seeded SQLite copy, and nothing here reads
ANTHROPIC_API_KEY - a suite that needs a key is a suite nobody runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["RAKAZ_TODAY"] = "2026-10-26"
os.environ["RAKAZ_HEBCAL"] = "offline"
os.environ["RAKAZ_WORKSPACE"] = "local"
os.environ["RAKAZ_ACTOR"] = "sarah.avital"
# The same threshold the measured runs use, so the suite exercises compaction
# exactly where results/api.json will.
os.environ["RAKAZ_COMPACT_THRESHOLD"] = "2000"
os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture
def clean(tmp_path, monkeypatch):
    """A workspace, memory and draft book that exist only for one test."""
    from shared import memory_store, plan, store

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "workspace.db")
    monkeypatch.setattr(memory_store, "MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(plan, "DRAFTS", tmp_path / "drafts.json")
    return tmp_path
