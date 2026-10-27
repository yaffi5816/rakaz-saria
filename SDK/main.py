"""Run the same scenario on the SDK version and write results/sdk.json.

    python SDK/main.py --scenario scenario.json --out results/sdk.json

Same scenario, same clean state, same checks as API/main.py, so the two result
files are comparable line for line.

One difference is forced on this side and is worth stating rather than hiding:
this version must run in a terminal where ANTHROPIC_API_KEY is not set, so the
run goes through the Claude Code login rather than billing a key. The guard
below refuses to start otherwise, because the failure it prevents is silent -
the run would work perfectly and the charge would land somewhere unintended.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from API.main import _totals, clean_state, load_env  # noqa: E402
from shared import checks, loc  # noqa: E402
from shared.memory_store import MemoryStore  # noqa: E402


def refuse_if_a_key_is_set() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY מוגדר בטרמינל הזה.\n"
            "גרסת ה-SDK אמורה לרוץ על ה-login שלך ב-Claude Code; עם מפתח מוגדר\n"
            "החיוב עובר למפתח במקום. הסירי אותו והריצי שוב:\n"
            "    unset ANTHROPIC_API_KEY        (bash)\n"
            "    Remove-Item Env:ANTHROPIC_API_KEY   (PowerShell)"
        )


async def run(scenario: dict, keep_state: bool) -> dict:
    from SDK.agent import EFFORT, MODEL, Rakaz

    if not keep_state:
        clean_state()

    started = time.time()
    turns: list[dict] = []
    memory = MemoryStore()
    calendar = {"source": "?", "http_calls": 0}

    for session in scenario["sessions"]:
        print(f"=== session {session['id']}: {session.get('title', '')} ===", flush=True)
        async with Rakaz(session=session["id"]) as agent:
            for turn in session["turns"]:
                print(f"--- turn {turn['n']} ---", flush=True)
                result = await agent.ask(turn["n"], turn["text"])
                print(f"    {', '.join(result.tools) or '(ללא כלים)'} | "
                      f"{result.model_calls} turns | ${result.cost_usd:.4f}", flush=True)
                turns.append({
                    "turn": result.turn,
                    "session": session["id"],
                    "request": turn["text"][:120],
                    "answer": result.text,
                    "cut_off": False,
                    # The SDK builds the prompt itself, so there is no per-turn
                    # context the way API/harness/context.py has one. Recorded
                    # as the length of what was actually sent - the bare text.
                    "context_chars": len(turn["text"]),
                    "tools": result.tools,
                    "expected_tools": turn.get("expects", []),
                    "model_calls": result.model_calls,
                    "input_tokens": result.input_tokens,
                    "cache_read_tokens": result.cache_read_tokens,
                    "cache_write_tokens": result.cache_write_tokens,
                    "output_tokens": result.output_tokens,
                    "cost_usd": round(result.cost_usd, 6),
                    "seconds": round(result.seconds, 2),
                    "sdk_session_id": result.session_id,
                    "state": checks.snapshot(agent.workspace, memory),
                })
            calendar["http_calls"] += agent.ctx.calendar.calls
            if agent.ctx.calendar.calls or calendar["source"] == "?":
                calendar["source"] = agent.ctx.calendar.source

    record = {
        "version": "sdk",
        "model": MODEL,
        "effort": EFFORT,
        "scenario": scenario["name"],
        "seconds": round(time.time() - started, 2),
        "turns": turns,
        "calendar": calendar,
    }
    record["totals"] = _totals(turns)
    record["checks"] = checks.run_all(record, scenario.get("actor", "sarah.avital"))
    record["lines_of_code"] = loc.table()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="רכז - גרסת ה-SDK")
    parser.add_argument("--scenario", default="scenario.json")
    parser.add_argument("--out", default="results/sdk.json")
    parser.add_argument("--keep-state", action="store_true")
    args = parser.parse_args()

    load_env()
    refuse_if_a_key_is_set()
    scenario = json.loads((ROOT / args.scenario).read_text(encoding="utf-8"))
    if scenario.get("today"):
        os.environ["RAKAZ_TODAY"] = scenario["today"]

    record = asyncio.run(run(scenario, keep_state=args.keep_state))

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
        f"{totals['model_calls']} תורים | ${totals['cost_usd']:.4f} | "
        f"cache {totals['cache_hit_rate']:.0%} | {record['seconds']}s"
    )
    print(f"נכתב: {out}")
    return 0 if passed == len(record["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
