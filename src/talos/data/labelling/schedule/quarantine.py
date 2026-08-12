
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from talos.data.labelling.schedule.windows import (
    Window, WindowError, build_all, extend_windows, ips, resolve_when,
    subnet_prefix, to_utc,
)

# A reason is written into every quarantined row and grouped on downstream, so it
# is an identifier. The sentence belongs in `justification`.
REASON_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")


class QuarantineError(WindowError):
    """A malformed region. A WindowError so `build_all` batches it with the rest."""


@dataclass(frozen=True)
class QuarantineRegion(Window):
    """A window we have withdrawn, and why.

    Deliberately the same `Window` as an attack rule: the endpoint predicate
    applied to a region is then literally the same SQL applied to a window, and
    cannot drift from it.
    """

    reason: str = ""
    justification: str = ""

    def __post_init__(self):
        """A region is unconstructable without a reason and an argument.

        These were a separate `validate_declarations` the loader had to remember
        to call -- the same shape of mistake `Window` just stopped making. An
        undocumented region is a silent narrowing of the dataset, which is the
        thing quarantine exists to prevent, so it is refused at construction.
        """
        super().__post_init__()
        where = f"{self.reason or '(no reason)'} {self.date} {self.start}-{self.end}"
        if not self.reason or set(self.reason) - REASON_CHARS:
            raise QuarantineError(
                f"{where}: reason must be a lower-case identifier ([a-z0-9-_]); "
                f"the prose belongs in justification")
        if not self.justification.strip():
            raise QuarantineError(f"{where}: no justification")


class QuarantineLoader:
    """Turns an already-parsed `quarantine:` block into regions.

    Takes the parsed document rather than a dataset name, so the manifest is
    read once and the regions are guaranteed to come from the same parse as the
    rules they clamp against. Previously this re-read the file itself, and
    nothing tied the `clamp_starts` it was handed to the document it loaded.
    """

    def load(self, doc: dict, clamp_starts: Sequence[float]) -> tuple[QuarantineRegion, ...]:
        """Regions from a manifest document. Absent block -> empty, not an error.

        `clamp_starts` are the *attack* rule starts: a region must never reach
        into a window and withdraw flows the schedule can account for.
        """
        entries = doc.get("quarantine") or []
        if not entries:
            return ()

        regions = tuple(build_all(
            lambda pair: self._region(pair[0], pair[1], doc),
            list(enumerate(entries))))
        return tuple(extend_windows(regions, clamp_starts))

    def _region(self, rid: int, entry: dict, doc: dict) -> QuarantineRegion:
        reason = entry.get("reason") or ""
        try:
            date, offset = resolve_when(entry, doc)
            t0 = to_utc(date, entry["start"], offset)
            t1 = to_utc(date, entry["end"], offset)
        except (KeyError, ValueError, WindowError) as exc:
            raise QuarantineError(f"{reason}: {exc}") from exc

        return QuarantineRegion(
            rid=rid,
            name=reason,                 # the shared Window field; rows group on it
            t0=t0,
            t1=t1,
            t1_raw=t1,
            attackers=ips(entry.get("attacker_ips")),
            victims=ips(entry.get("victim_ips")),
            subnet_prefix=subnet_prefix(entry.get("victim_subnet")),
            date=date,
            start=entry["start"],
            end=entry["end"],
            reason=reason,
            justification=str(entry.get("justification") or "").strip(),
        )
