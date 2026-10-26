"""What a turn cost, in tokens and in dollars.

Four token classes, not one, because they are priced an order of magnitude
apart on this model: ordinary input at $4 per million, a cache write at $5, a
cache read at $0.20, output at $20. A harness that reports "input tokens" and
sums the three input classes into it cannot tell a cheap turn from an expensive
one - on Claude Opus 5.5 a cached turn and an uncached turn of identical size
differ by twenty times on the input side.

The prices are pinned here with the date they were read, because a number
copied into a report is worth nothing without one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from API.harness import say

MODEL = "claude-opus-5-5"

# Dollars per million tokens, Claude Opus 5.5, read 2026-10-26.
PRICES = {
    "input": 4.00,
    "cache_write_5m": 5.00,
    "cache_write_1h": 8.00,
    "cache_read": 0.20,
    "output": 20.00,
}


@dataclass
class TurnUsage:
    """One turn: possibly several model calls, one line in the results file."""

    turn: int
    model_calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    tools: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICES["input"]
            + self.cache_write_tokens * PRICES["cache_write_5m"]
            + self.cache_read_tokens * PRICES["cache_read"]
            + self.output_tokens * PRICES["output"]
        ) / 1_000_000

    @property
    def cache_hit_rate(self) -> float:
        """Share of input bytes served from cache rather than paid for in full."""
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return (self.cache_read_tokens / total) if total else 0.0

    def as_dict(self) -> dict:
        return {
            "turn": self.turn,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cost_usd": round(self.cost_usd, 6),
            "seconds": round(self.seconds, 2),
            "tools": self.tools,
        }


class Meter:
    """Accumulates usage across the turns of a run."""

    def __init__(self) -> None:
        self.turns: list[TurnUsage] = []
        self._current: TurnUsage | None = None
        self._started = 0.0

    def start_turn(self, number: int) -> TurnUsage:
        self._current = TurnUsage(turn=number)
        self._started = time.perf_counter()
        self.turns.append(self._current)
        return self._current

    def record(self, usage) -> None:
        """Fold one model response's usage into the current turn."""
        if self._current is None:
            return
        self._current.model_calls += 1
        self._current.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self._current.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self._current.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self._current.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def end_turn(self) -> TurnUsage | None:
        if self._current is None:
            return None
        self._current.seconds = time.perf_counter() - self._started
        say(
            "tokens.py",
            f"turn {self._current.turn}: {self._current.model_calls} model calls, "
            f"in={self._current.input_tokens} read={self._current.cache_read_tokens} "
            f"write={self._current.cache_write_tokens} out={self._current.output_tokens} "
            f"${self._current.cost_usd:.4f} ({self._current.seconds:.1f}s)",
        )
        done, self._current = self._current, None
        return done

    def totals(self) -> dict:
        return {
            "model_calls": sum(t.model_calls for t in self.turns),
            "input_tokens": sum(t.input_tokens for t in self.turns),
            "cache_read_tokens": sum(t.cache_read_tokens for t in self.turns),
            "cache_write_tokens": sum(t.cache_write_tokens for t in self.turns),
            "output_tokens": sum(t.output_tokens for t in self.turns),
            "cost_usd": round(sum(t.cost_usd for t in self.turns), 6),
            "seconds": round(sum(t.seconds for t in self.turns), 2),
            "cache_hit_rate": round(
                sum(t.cache_read_tokens for t in self.turns)
                / max(
                    1,
                    sum(
                        t.input_tokens + t.cache_read_tokens + t.cache_write_tokens
                        for t in self.turns
                    ),
                ),
                4,
            ),
        }
