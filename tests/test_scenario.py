"""The whole scenario, end to end, against a scripted model.

This is the test that matters: it runs the same code path as
`python API/main.py --scenario scenario.json`, through all eight turns and both
sessions, and then runs the same checks that the measured run writes into
results/api.json. If the harness stops replaying the transcript, stops
enforcing the guard, or starts writing drafts without a commit, this fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.fake_model import client_factory

ROOT = Path(__file__).resolve().parent.parent


def run_scenario():
    from API import main

    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    return main.run(scenario, keep_state=False, client_factory=client_factory())


def test_every_check_passes():
    record = run_scenario()
    failed = [c for c in record["checks"] if not c["passed"]]
    assert not failed, "\n".join(f"{c['name']}: {c['detail']}" for c in failed)


def test_results_shape_is_complete():
    """Every turn carries what the exercise asks the results file to carry."""
    record = run_scenario()
    assert len(record["turns"]) == 8
    for turn in record["turns"]:
        for field in ("input_tokens", "cache_read_tokens", "cache_write_tokens",
                      "output_tokens", "model_calls", "tools", "cost_usd", "seconds"):
            assert field in turn, f"turn {turn['turn']} is missing {field}"
    assert record["lines_of_code"]["api_total"] > 0
    assert record["totals"]["model_calls"] >= 8


def test_sessions_do_not_share_a_transcript():
    """Session 2 starts empty, so what it knows came off the disk."""
    run_scenario()
    history = ROOT / "state" / "history"
    s1 = (history / "s1.jsonl").read_text(encoding="utf-8")
    s2 = (history / "s2.jsonl").read_text(encoding="utf-8")
    assert "מודול שינה" in s2
    assert "מודול שינה" not in s1
    assert "ישיבת היגוי" in s1
    assert "ישיבת היגוי" not in s2


def test_tools_requested_match_the_scenario():
    """The tools each turn expects are the tools that ran."""
    record = run_scenario()
    for turn in record["turns"]:
        for expected in turn["expected_tools"]:
            assert expected in turn["tools"], (
                f"turn {turn['turn']} expected {expected}, ran {turn['tools']}"
            )
