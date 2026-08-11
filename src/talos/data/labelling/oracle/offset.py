"""Testing the one thing no document can settle: which timezone the clocks are in.

Suricata settles it, because pcap packet timestamps are absolute. An alert's
time is a fact about the capture, not about anybody's documentation. So:

    shift the schedule by each candidate offset, and see which one puts the
    alerts inside the attack windows.

If the declared offset wins clearly, the inference is empirically supported. If
a neighbour wins, the manifest is wrong by that many hours. If nothing wins, the
probe says so rather than inventing a preference — an inconclusive answer here
is worth far more than a confident one, because the whole point is that this
number has never been evidenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from talos.data.labelling.windows import Window, parse_offset

HOUR = 3600.0

# Plausible alternatives for a capture in Atlantic Canada, plus UTC itself for
# the "the box was never configured" case, which is the commonest way a testbed
# ends up with unexpected timestamps.
DEFAULT_CANDIDATES = ("+00:00", "-01:00", "-02:00", "-03:00", "-04:00", "-05:00")

# Below this many alerts the probe cannot distinguish anything and says so.
MIN_ALERTS = 20
# The winner must beat the runner-up by this factor to count as a clear answer.
MIN_MARGIN = 1.5


@dataclass(frozen=True)
class OffsetScore:
    offset: str
    shift_seconds: float
    alerts_inside: int

    @property
    def hours(self) -> float:
        return self.shift_seconds / HOUR


@dataclass(frozen=True)
class OffsetVerdict:
    declared: str
    scores: tuple[OffsetScore, ...]
    alerts_considered: int
    min_alerts: int = MIN_ALERTS
    min_margin: float = MIN_MARGIN

    @property
    def ranked(self) -> tuple[OffsetScore, ...]:
        return tuple(sorted(self.scores, key=lambda s: -s.alerts_inside))

    @property
    def best(self) -> OffsetScore | None:
        return self.ranked[0] if self.scores else None

    @property
    def conclusive(self) -> bool:
        """Enough alerts, and a clear winner over the runner-up."""
        if self.alerts_considered < self.min_alerts or len(self.ranked) < 2:
            return False
        best, second = self.ranked[0], self.ranked[1]
        if best.alerts_inside == 0:
            return False
        return best.alerts_inside >= self.min_margin * max(second.alerts_inside, 1)

    @property
    def agrees(self) -> bool | None:
        """True/False when conclusive, None when the probe cannot say."""
        if not self.conclusive:
            return None
        return self.best.offset == self.declared

    def summary(self) -> str:
        lines = [f"declared offset  {self.declared}",
                 f"alerts considered {self.alerts_considered:,}"]
        for s in self.ranked:
            mark = " <- declared" if s.offset == self.declared else ""
            lines.append(f"  {s.offset}  {s.alerts_inside:>8,} alerts inside a window{mark}")
        if not self.conclusive:
            lines.append("\nINCONCLUSIVE — too few alerts, or no offset clearly wins. "
                         "The declared value remains unevidenced.")
        elif self.agrees:
            lines.append(f"\nthe declared offset {self.declared} is empirically supported")
        else:
            # best.shift_seconds is how far every window moves if the winner is
            # right -- positive means later. Reporting the shift, not the
            # difference between two offsets, because the shift is the thing
            # someone has to act on.
            direction = "later" if self.best.shift_seconds > 0 else "earlier"
            lines.append(
                f"\n!! {self.best.offset} fits the capture, not {self.declared}: "
                f"every window belongs {abs(self.best.hours):.0f}h {direction}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "declared": self.declared,
            "alerts_considered": self.alerts_considered,
            "conclusive": self.conclusive,
            "agrees": self.agrees,
            "best": self.best.offset if self.best else None,
            "scores": [{"offset": s.offset, "alerts_inside": s.alerts_inside}
                       for s in self.ranked],
        }


def _seconds(offset: str) -> float:
    return parse_offset(offset).utcoffset(None).total_seconds()


class OffsetProbe:
    """Scores candidate UTC offsets against where the alerts actually landed."""

    def __init__(self, candidates: Sequence[str] = DEFAULT_CANDIDATES,
                 endpoint_constrained: bool = True):
        self.candidates = tuple(candidates)
        self.endpoint_constrained = endpoint_constrained

    @staticmethod
    def shift_for(declared: str, candidate: str) -> float:
        """How far windows move if the true offset is `candidate`.

        `window = naive_local - offset`, so swapping the offset moves every
        window by the difference — and by the same amount, which is exactly why
        the error is invisible in the output.
        """
        return _seconds(declared) - _seconds(candidate)

    def _touches(self, alert, window: Window) -> bool:
        if not self.endpoint_constrained:
            return True
        hosts = set(window.attackers or ()) | set(window.victims or ())
        if window.subnet_prefix:
            if (alert.src_ip.startswith(window.subnet_prefix)
                    or alert.dest_ip.startswith(window.subnet_prefix)):
                return True
        return bool(hosts) and (alert.src_ip in hosts or alert.dest_ip in hosts)

    def score(self, alerts: Iterable, windows: Sequence[Window],
              shift: float) -> int:
        """Alerts landing inside any window, with the schedule shifted."""
        inside = 0
        for alert in alerts:
            for w in windows:
                if (w.t0 + shift) <= alert.ts < (w.t1 + shift) and self._touches(alert, w):
                    inside += 1
                    break
        return inside

    def probe(self, alerts: Sequence, windows: Sequence[Window],
              declared: str) -> OffsetVerdict:
        scores = tuple(
            OffsetScore(candidate, self.shift_for(declared, candidate),
                        self.score(alerts, windows, self.shift_for(declared, candidate)))
            for candidate in self.candidates)
        return OffsetVerdict(declared=declared, scores=scores,
                             alerts_considered=len(alerts))
