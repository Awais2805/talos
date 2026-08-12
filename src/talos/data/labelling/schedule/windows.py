"""Time-and-endpoint regions — the shape attack rules and quarantine share.

A window says "between these instants, involving these hosts". A schedule entry
is one; a withdrawn span is one. They are the same type deliberately: the
predicate applied to a region must be *literally* the predicate applied to a
window, or the two drift and nothing says so.

A `Window` cannot be constructed invalid — see `__post_init__`. The two
conversions it depends on are `to_utc` and `extend_windows`, and both fail
silently when wrong, so the reasoning lives on them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

MINUTE = 60.0


class WindowError(Exception):
    pass


def parse_offset(offset: str) -> timezone:
    """`"-03:00"` -> a tzinfo. Rejects anything it cannot read rather than guessing."""
    text = str(offset).strip()
    sign = -1 if text.startswith("-") else 1
    body = text.lstrip("+-")
    try:
        hours, minutes = (int(part) for part in body.split(":"))
    except ValueError as exc:
        raise WindowError(
            f"{offset!r} is not a UTC offset; expected a form like '-03:00'") from exc
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def to_utc(date, hhmm: str, offset: str) -> float:
    """Local wall-clock (date, HH:MM) at a declared offset -> epoch seconds.

    Converted once, in code. Hand-computing UTC into a manifest invites
    arithmetic errors nothing downstream can see, and it cannot express
    CIC-DDoS-2019 at all: its two capture days carry different offsets, ADT then
    AST, either side of a DST change.

    `%H:%M` also enforces the minute granularity the whole extension rule rests
    on — a time carrying seconds fails here rather than being extended by a
    spurious 60 seconds later.

    `date` may arrive as a `datetime.date`: YAML parses an unquoted `2017-07-04`
    into one, and stringifying is cheaper than teaching every call site to care.
    """
    stamp = datetime.strptime(f"{date} {hhmm}", "%Y-%m-%d %H:%M")
    return stamp.replace(tzinfo=parse_offset(offset)).timestamp()


@dataclass(frozen=True)
class Window:
    """One time-and-endpoint region.

    `victims is None` means "any peer" and is materially different from an empty
    list: a subnet rule carries no `victim_ips`, and reading that absence as "no
    victim constraint" once made a rule match every flow in its window.
    """

    rid: int
    name: str
    t0: float
    t1: float                       # already minute-extended
    t1_raw: float                   # the schedule's own end, kept for audit
    attackers: tuple[str, ...] | None
    victims: tuple[str, ...] | None
    subnet_prefix: str | None       # "192.168.10." for a /24 — a cheap prefix test
    date: str
    start: str
    end: str

    def __post_init__(self):
        """Invalid windows are unconstructable, not merely detectable.

        These used to be a separate `validate()` a loader had to remember to
        call. Two of the three are silent when wrong: a window that ends before
        it starts matches nothing and reads as an attack that produced no
        traffic, and a subnet prefix missing its trailing dot makes a /24 rule
        swallow its neighbour -- `192.168.100.5` starts with `192.168.10`.
        """
        if self.t1 <= self.t0:
            raise WindowError(f"{self.label()}: end {self.end} is not after "
                              f"start {self.start}")
        if not (self.attackers or self.victims or self.subnet_prefix):
            raise WindowError(f"{self.label()}: no attacker, victim or subnet — "
                              f"this would match every flow in its window")
        if self.subnet_prefix and not self.subnet_prefix.endswith("."):
            raise WindowError(
                f"{self.label()}: subnet prefix {self.subnet_prefix!r} must end "
                f"with a dot, or it matches the neighbouring range too")

    @property
    def extended_by(self) -> float:
        """Seconds the minute-inclusive rule added. Zero where the clamp bound."""
        return self.t1 - self.t1_raw

    @property
    def victim_only(self) -> bool:
        """No attacker constraint: match on the victim alone.

        Named for why it exists — 2019's reflection attacks spoof their sources,
        so the attacker address carries no information — but it means exactly
        "this rule names no attacker", whatever the reason.
        """
        return self.attackers is None

    def constrains_endpoints(self) -> bool:
        return bool(self.attackers or self.victims or self.subnet_prefix)

    def label(self) -> str:
        return f"{self.name} {self.date} {self.start}-{self.end}"


def subnet_prefix(cidr: str | None) -> str | None:
    """A /24 as a string prefix. Only /24 is supported, and anything else says so."""
    if not cidr:
        return None
    net, _, bits = str(cidr).partition("/")
    if bits.strip() != "24":
        raise WindowError(
            f"{cidr!r}: only /24 victim subnets are supported, because matching is a "
            f"string prefix test. Widen the implementation before widening the manifest.")
    return net.rsplit(".", 1)[0] + "."


def ips(values) -> tuple[str, ...] | None:
    """An IP list as a tuple, or None for 'unconstrained'. Empty list is None."""
    return tuple(str(v) for v in values) if values else None


def extend_windows(windows: Sequence[Window], clamp_starts: Iterable[float]) -> list[Window]:
    """Apply the minute-inclusive rule, clamped at the next window's start.

    `clamp_starts` is passed in rather than derived from `windows` because
    quarantine regions must clamp against the *attack* schedule: a region that
    ran into an attack window would withdraw flows the schedule can account for.
    """
    starts = sorted(clamp_starts)
    out = []
    for w in windows:
        nxt = next((s for s in starts if s >= w.t1_raw), None)
        end = w.t1_raw + MINUTE if nxt is None else min(w.t1_raw + MINUTE, nxt)
        out.append(replace(w, t1=end))
    return out


def resolve_when(entry: dict, doc: dict) -> tuple[str, str]:
    """(date, utc_offset) for one entry, from whichever shape the manifest uses.

    2019 keys entries on `day` because its two capture days carry different
    offsets -- ADT then AST, either side of the 2018 DST change. Resolving that
    per entry is the only way the difference can be expressed at all.

    Lives here rather than on the manifest loader because quarantine regions
    need it too, and a shared resolver is what stops the two shapes being
    interpreted differently.
    """
    days = doc.get("days")
    if days is not None:
        key = entry.get("day")
        if key not in days:
            raise WindowError(f"{entry.get('name') or entry.get('reason')}: "
                              f"day {key!r} is not in the days: table")
        day = days[key]
        return str(day["date"]), day["utc_offset"]

    if "date" not in entry:
        raise WindowError(f"{entry.get('name') or entry.get('reason')}: "
                          f"no date, and no days: table")
    offset = doc.get("utc_offset")
    if offset is None:
        raise WindowError(f"{entry.get('name') or entry.get('reason')}: "
                          f"manifest declares no utc_offset")
    return str(entry["date"]), offset


def build_all(factory, entries) -> list:
    """Construct every window, reporting ALL the bad ones rather than the first.

    `__post_init__` fails on one window; a manifest with four mistakes should
    cost one round trip, not four.
    """
    built, errs, kind = [], [], WindowError
    for entry in entries:
        try:
            built.append(factory(entry))
        except WindowError as exc:
            # Keep the first error's type: a caller catching QuarantineError
            # should still catch a batch of quarantine problems.
            kind = type(exc) if not errs else kind
            errs.append(str(exc))
    if errs:
        raise kind("invalid windows:\n  " + "\n  ".join(errs))
    return built
