"""The external tool: the Jewish calendar, from hebcal.com.

Why an agent planning work in an Israeli office needs it at all: a deadline is a
promise about a working day, and a date chosen by arithmetic lands on Shabbat
about a seventh of the time and inside a festival more often than anybody
remembers. "עד 21.11" sounds fine in a meeting and is a Saturday.

Two kinds of answer come back from one call, and the agent is told to treat them
differently. Shabbat and a yom tov make a date unusable, and the draft moves to
the next working day. Chol HaMoed and Hanukkah do not - they are working days
with half the office away - so those are a warning printed beside the date and
nothing more.

Offline by default in tests, through the bundled fixture, so the suite needs no
network and no key. A measured run goes to the network and says so, because
"the external tool really ran" is one of the checks.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "shared" / "fixtures" / "hebcal_2026_2027.json"
CACHE = ROOT / "state" / "hebcal_cache.json"
ENDPOINT = "https://www.hebcal.com/hebcal"

WEEKDAY_HE = {1: "שני", 2: "שלישי", 3: "רביעי", 4: "חמישי", 5: "שישי", 6: "שבת", 7: "ראשון"}

# Hebcal returns festival names fully pointed ("חֲנוּכָּה"). Every one of those
# marks is a separate code point and most of them are a token of their own, so
# a holiday name costs three or four times what it needs to. Stripped on the
# way in; nothing downstream reads them.
NIKUD = {chr(code) for code in range(0x0591, 0x05C8)} | {"׳", "״"}


def strip_nikud(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in NIKUD)
# Sunday to Thursday. Friday is a half day that no deadline should land on, and
# treating it as a working day is how a task becomes due at an hour when the
# office it belongs to has already gone home.
WORKING_WEEKDAYS = {7, 1, 2, 3, 4}


class CalendarService:
    """Holiday lookups, cached on disk and answerable with no network at all."""

    def __init__(self, mode: str | None = None) -> None:
        self.mode = (mode or os.environ.get("RAKAZ_HEBCAL", "auto")).strip().lower()
        self._days: dict[str, list[dict]] = {}
        self.source = "fixture"
        self.calls = 0
        # Years this instance has already settled, from the cache file or from
        # the network. The bundled fixture does NOT count: it covers 2026 and
        # 2027, so treating it as "we have that year" would mean a measured run
        # never actually reaches hebcal.com while the report claims it does.
        self._settled: set[int] = set()
        self._load()

    def _load(self) -> None:
        self._days = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.source = "fixture"
        if self.mode == "offline" or not CACHE.exists():
            return
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._days.update(cached)
        self.source = "cache"
        self._settled = {int(day[:4]) for day in cached if day[:4].isdigit()}

    def _fetch(self, year: int) -> None:
        """Pull one year from hebcal.com and fold it into the cache on disk."""
        import httpx

        params = {
            "v": "1", "cfg": "json", "maj": "on", "min": "on", "mod": "on", "nx": "off",
            "year": str(year), "month": "x", "ss": "off", "mf": "off", "c": "off",
            "geo": "none", "lg": "he",
        }
        payload = httpx.get(ENDPOINT, params=params, timeout=20.0).json()
        self.calls += 1

        # Built fresh and then merged over, rather than appended to. The fixture
        # already holds these days, and appending would give every festival two
        # entries - which nothing would notice until a date reported its holiday
        # twice.
        live: dict[str, list[dict]] = {}
        for item in payload.get("items", []):
            day = (item.get("date") or "")[:10]
            if day:
                live.setdefault(day, []).append({
                    "title": item.get("title"), "yomtov": bool(item.get("yomtov")),
                    "category": item.get("category"), "subcat": item.get("subcat"),
                })
        self._days.update(live)

        # Only fetched days reach the cache. Writing the merged dictionary would
        # put fixture days in there too, and the next run would read the cache,
        # conclude it already holds that year, and never call out again.
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        stored = {}
        if CACHE.exists():
            try:
                stored = json.loads(CACHE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = {}
        stored.update(live)
        CACHE.write_text(
            json.dumps(stored, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        self.source = "live"

    def _ensure(self, year: int) -> None:
        if self.mode == "offline" or year in self._settled:
            return
        # Marked settled either way: an unreachable calendar must not stop a
        # planning session, and it must not be retried on every date either.
        # The bundled fixture still answers, and self.source says so.
        self._settled.add(year)
        try:
            self._fetch(year)
        except Exception:
            pass

    def day(self, date: datetime.date) -> dict:
        """Report what one calendar day is: working or not, and why."""
        self._ensure(date.year)
        entries = self._days.get(date.isoformat(), [])
        yomtov = next((e for e in entries if e.get("yomtov")), None)
        others = [e for e in entries if not e.get("yomtov")]
        weekday = date.isoweekday()
        working = weekday in WORKING_WEEKDAYS and yomtov is None
        if yomtov:
            reason = strip_nikud(yomtov["title"])
        elif weekday == 6:
            reason = "שבת"
        elif weekday == 5:
            reason = "יום שישי"
        else:
            reason = ""
        return {
            "date": date.isoformat(),
            "weekday": WEEKDAY_HE[weekday],
            "working": working,
            "reason": reason,
            "note": strip_nikud(others[0]["title"]) if others else "",
        }

    def next_working_day(self, date: datetime.date) -> datetime.date:
        for step in range(1, 15):
            candidate = date + datetime.timedelta(days=step)
            if self.day(candidate)["working"]:
                return candidate
        return date

    def check(self, dates: list[str]) -> list[dict]:
        """Answer for several dates at once, which is how a draft asks."""
        out = []
        for raw in dates:
            try:
                parsed = datetime.date.fromisoformat(raw[:10])
            except ValueError:
                out.append({"date": raw, "error": "תאריך לא תקין"})
                continue
            info = self.day(parsed)
            if not info["working"]:
                info["suggested"] = self.next_working_day(parsed).isoformat()
            out.append(info)
        return out


def render(rows: list[dict]) -> str:
    """date|weekday|ok/blocked|suggested|note - the compact form of a check."""
    lines = []
    for r in rows:
        if "error" in r:
            lines.append(f"{r['date']}|-|error|-|{r['error']}")
            continue
        verdict = "ok" if r["working"] else f"blocked:{r['reason']}"
        lines.append(
            "|".join([
                r["date"][5:], r["weekday"], verdict,
                r.get("suggested", "-")[5:] if r.get("suggested") else "-",
                r["note"] or "-",
            ])
        )
    return "\n".join(lines)
