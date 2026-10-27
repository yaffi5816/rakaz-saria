"""Run the scenario against the hand-written harness and write results/api.json.

    python API/main.py --scenario scenario.json --out results/api.json

Every run starts from a clean state unless --keep-state says otherwise: the
seeded workspace, the conventions file, the drafts and the transcripts are all
rebuilt, because a second run against a workspace the first one wrote into
measures something else. The frozen date in .env keeps lateness identical
between runs on different days.

Each session gets its own instance of the agent. That is what makes the memory
checks mean anything: the session-2 instance is a new process-level object with
its own empty transcript, so a rule it applies came off the disk and not out of
the conversation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):  # Hebrew on a Windows console
    sys.stdout.reconfigure(encoding="utf-8")

from API import harness  # noqa: E402
from API.harness import say  # noqa: E402
from shared import checks, loc  # noqa: E402
from shared.memory_store import MemoryStore  # noqa: E402


def load_env() -> None:
    """Read .env if it is there. Never a key - those come from the terminal."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "ANTHROPIC_API_KEY":
            continue
        os.environ.setdefault(key.strip(), value.strip())


def clean_state() -> None:
    """Delete everything a previous run wrote, keeping the profile prose."""
    from shared.store import reset as reset_workspace

    reset_workspace()
    for path in [
        ROOT / "state" / "drafts.json",
        ROOT / "state" / "hebcal_cache.json",
        ROOT / "memory" / "conventions.jsonl",
    ]:
        if path.exists():
            path.unlink()
    history = ROOT / "state" / "history"
    if history.exists():
        for path in history.glob("*.jsonl"):
            path.unlink()
    say("main.py", "clean state: workspace, drafts, conventions and transcripts removed")


def run(scenario: dict, keep_state: bool, client_factory=None) -> dict:
    """Play the scenario. client_factory is injected by the suite only."""
    from API.agent import Rakaz

    if not keep_state:
        clean_state()

    started = time.time()
    turns: list[dict] = []
    memory = MemoryStore()
    # Accumulated across sessions. Session 1 goes to the network and session 2
    # reads the cache it left, so the last session's view of the calendar is
    # not the run's.
    calendar = {"source": "?", "http_calls": 0}

    for session in scenario["sessions"]:
        say("main.py", f"=== session {session['id']}: {session.get('title', '')} ===")
        agent = Rakaz(session=session["id"], client_factory=client_factory)
        for turn in session["turns"]:
            say("main.py", f"--- turn {turn['n']} ---")
            answer = agent.ask(turn["n"], turn["text"])
            turns.append({
                "turn": turn["n"],
                "session": session["id"],
                "request": turn["text"][:120],
                "answer": answer["text"],
                "cut_off": answer["cut_off"],
                "context_chars": answer["context_chars"],
                "compactions": answer["compactions"],
                "tools": answer["usage"]["tools"],
                "expected_tools": turn.get("expects", []),
                **{k: v for k, v in answer["usage"].items() if k != "tools"},
                "state": checks.snapshot(agent.workspace, memory),
            })
        calendar["http_calls"] += agent.calendar.calls
        if agent.calendar.calls or calendar["source"] == "?":
            calendar["source"] = agent.calendar.source
        say("main.py", f"session {session['id']} done, {agent.compactions} compactions, "
                       f"calendar {calendar['source']} ({calendar['http_calls']} http)")

    record = {
        "version": "api",
        "model": os.environ.get("RAKAZ_MODEL", "claude-opus-5-5"),
        "effort": os.environ.get("RAKAZ_EFFORT", "medium"),
        "scenario": scenario["name"],
        "seconds": round(time.time() - started, 2),
        "turns": turns,
        "calendar": calendar,
    }
    record["totals"] = _totals(turns)
    record["checks"] = checks.run_all(record, scenario.get("actor", "sarah.avital"))
    record["lines_of_code"] = loc.table()
    record["trace"] = harness.TRACE
    return record


def _totals(turns: list[dict]) -> dict:
    keys = ["model_calls", "input_tokens", "cache_read_tokens",
            "cache_write_tokens", "output_tokens"]
    totals = {key: sum(t.get(key, 0) for t in turns) for key in keys}
    totals["cost_usd"] = round(sum(t.get("cost_usd", 0) for t in turns), 6)
    totals["seconds"] = round(sum(t.get("seconds", 0) for t in turns), 2)
    billed = totals["input_tokens"] + totals["cache_read_tokens"] + totals["cache_write_tokens"]
    totals["cache_hit_rate"] = round(totals["cache_read_tokens"] / billed, 4) if billed else 0.0
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description="רכז - גרסת ה-API")
    parser.add_argument("--scenario", default="scenario.json")
    parser.add_argument("--out", default="results/api.json")
    parser.add_argument("--keep-state", action="store_true",
                        help="אל תאפס את המצב לפני הריצה (לניפוי בלבד)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--fake", action="store_true",
        help="הרצה על client מדומה: בלי מפתח, בלי רשת ובלי עלות. "
             "מריץ את כל ה-harness ואת כל הבדיקות, והטוקנים בקובץ התוצאות מומצאים.",
    )
    args = parser.parse_args()

    load_env()
    harness.QUIET = args.quiet
    scenario = json.loads((ROOT / args.scenario).read_text(encoding="utf-8"))
    if scenario.get("today"):
        os.environ["RAKAZ_TODAY"] = scenario["today"]

    factory = None
    if args.fake:
        os.environ.setdefault("RAKAZ_HEBCAL", "offline")
        from tests.fake_model import client_factory

        factory = client_factory()
        say("main.py", "DRY RUN: scripted model, offline calendar, invented token counts")

    if not args.fake:
        from API.agent import check_credentials

        check_credentials()

    record = run(scenario, keep_state=args.keep_state, client_factory=factory)
    record["dry_run"] = bool(args.fake)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    passed = sum(1 for c in record["checks"] if c["passed"])
    print()
    for check in record["checks"]:
        print(("  עבר   " if check["passed"] else "  נכשל ") + check["name"])
        print(f"         {check['detail']}")
    totals = record["totals"]
    print(
        f"\n{passed}/{len(record['checks'])} בדיקות עברו | "
        f"{totals['model_calls']} קריאות למודל | "
        f"${totals['cost_usd']:.4f} | cache {totals['cache_hit_rate']:.0%} | "
        f"{record['seconds']}s"
    )
    print(f"נכתב: {out}")
    return 0 if passed == len(record["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
