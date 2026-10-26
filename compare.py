"""Fill REPORT.md's number tables from results/api.json and results/sdk.json.

    python compare.py            # print the tables
    python compare.py --write    # replace the block inside REPORT.md

Every figure in the report comes from here, so no number in it is typed by
hand. The block it rewrites is delimited by HTML comments, and everything
outside those markers - the analysis, which is the part worth reading - is left
alone.

A missing results file is reported as missing rather than filled with zeros: a
table of zeros looks like a measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = "<!-- numbers:start -->"
END = "<!-- numbers:end -->"

ROWS = [
    ("קריאות למודל", "model_calls", "{:,}"),
    ("טוקני קלט (לא מהמטמון)", "input_tokens", "{:,}"),
    ("טוקנים שנקראו מהמטמון", "cache_read_tokens", "{:,}"),
    ("טוקנים שנכתבו למטמון", "cache_write_tokens", "{:,}"),
    ("טוקני פלט", "output_tokens", "{:,}"),
    ("אחוז מטמון", "cache_hit_rate", "{:.0%}"),
    ("עלות", "cost_usd", "${:.4f}"),
    ("זמן (שניות)", "seconds", "{:.1f}"),
]


def load(name: str) -> dict | None:
    path = ROOT / "results" / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(record: dict | None, key: str, fmt: str) -> str:
    if record is None:
        return "—"
    value = record["totals"].get(key, 0)
    return fmt.format(value)


def headline(api: dict | None, sdk: dict | None) -> str:
    lines = ["| | API | SDK |", "|---|---:|---:|"]
    for label, key, fmt in ROWS:
        lines.append(f"| {label} | {_cell(api, key, fmt)} | {_cell(sdk, key, fmt)} |")
    passed = []
    for record in (api, sdk):
        if record is None:
            passed.append("—")
        else:
            ok = sum(1 for c in record["checks"] if c["passed"])
            passed.append(f"{ok}/{len(record['checks'])}")
    lines.append(f"| בדיקות שעברו | {passed[0]} | {passed[1]} |")
    return "\n".join(lines)


def per_turn(record: dict | None, title: str) -> str:
    if record is None:
        return f"**{title}** — אין קובץ תוצאות.\n"
    lines = [f"**{title}**", "", "| תור | קריאות | קלט | מטמון־קריאה | מטמון־כתיבה | פלט | עלות | שניות | כלים |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for turn in record["turns"]:
        lines.append(
            "| {turn} | {model_calls} | {input_tokens:,} | {cache_read_tokens:,} | "
            "{cache_write_tokens:,} | {output_tokens:,} | ${cost_usd:.4f} | "
            "{seconds:.1f} | {tools} |".format(
                tools=", ".join(turn.get("tools", [])) or "—",
                **{k: turn.get(k, 0) for k in
                   ("turn", "model_calls", "input_tokens", "cache_read_tokens",
                    "cache_write_tokens", "output_tokens", "cost_usd", "seconds")},
            )
        )
    return "\n".join(lines) + "\n"


def lines_of_code(api: dict | None) -> str:
    from shared import loc

    table = api["lines_of_code"] if api else loc.table()
    rows = ["| אחריות | API (שורות) | SDK (שורות שמחליפות אותה) |", "|---|---:|---:|"]
    for name in table["api_harness"]:
        rows.append(
            f"| `{name}` | {table['api_harness'][name]} | "
            f"{table['sdk_replacing_lines'].get(name, 0)} |"
        )
    rows.append(f"| **סך הכל** | **{table['api_total']}** | **{table['sdk_total']}** |")
    rows.append(f"| קוד דומיין משותף (`shared/`) | {table['shared_total']} | "
                f"{table['shared_total']} |")
    return "\n".join(rows)


def dry_run_note(api: dict | None, sdk: dict | None) -> str:
    notes = []
    for name, record in (("api", api), ("sdk", sdk)):
        if record and record.get("dry_run"):
            notes.append(name)
    if not notes:
        return ""
    return (
        f"\n> ⚠️ ריצת יבש ({', '.join(notes)}): המודל היה מדומה והטוקנים מומצאים. "
        "המספרים כאן מראים את צורת הדוח, לא מדידה.\n"
    )


def build() -> str:
    api, sdk = load("api.json"), load("sdk.json")
    parts = [
        START,
        "",
        "### סיכום הריצה",
        "",
        headline(api, sdk),
        dry_run_note(api, sdk),
        "",
        "### שורות קוד לכל אחריות",
        "",
        lines_of_code(api),
        "",
        "### פירוט לפי תור",
        "",
        per_turn(api, "API"),
        "",
        per_turn(sdk, "SDK"),
        END,
    ]
    return "\n".join(parts)


def main() -> int:
    block = build()
    report = ROOT / "REPORT.md"
    if "--write" not in sys.argv:
        print(block)
        return 0
    if not report.exists():
        print("אין REPORT.md")
        return 1
    text = report.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"לא נמצאו הסימונים {START} ו-{END} ב-REPORT.md")
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    report.write_text(head + block + tail, encoding="utf-8")
    print("REPORT.md עודכן")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
