"""What this machine can spare, detected once and declared everywhere.

Talos is meant to run the same way on a laptop, a rented box and a workstation.
That works because the *code path* never changes — only how much of the machine
it is allowed to use. This module is where "how much" is decided.

Detected by default and overridable in `config.yml`, because a detected value is
a guess about a shared machine: another process may want the RAM, or the biggest
volume may be a network mount you would rather not spill onto.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

GB = 1_000_000_000

# Leave the OS and everything else a quarter of the machine. DuckDB's own
# default is 80%; 75% is a little kinder on a laptop that is also running a
# browser and an editor.
MEMORY_FRACTION = 0.75

# A thread with less than this to work in spends its time evicting rather than
# computing. This is the cap that stops the thrash.
MIN_MEMORY_PER_THREAD_MB = 512

# Spill is bounded twice: an absolute ceiling, and a share of the volume, so a
# runaway query cannot take the disk with it.
MAX_TEMP_ABSOLUTE_GB = 20
MAX_TEMP_VOLUME_FRACTION = 0.25

# Used when the platform will not say how much memory it has (Windows, or an
# unusual libc). Small enough to be safe, large enough to be useful.
FALLBACK_MEMORY_GB = 4


def total_memory_bytes() -> int | None:
    """Physical RAM, or None if this platform will not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _free_bytes(path: str | Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def roomiest_volume(extra: str | Path | None = None) -> Path:
    """Whichever candidate volume has the most free space.

    The lake's own directory is included when it is local, because spilling
    beside the data usually means spilling on the big disk rather than the one
    holding the operating system.
    """
    candidates = [os.environ.get("TMPDIR"), tempfile.gettempdir(), "/var/tmp", Path.cwd()]
    if extra:
        candidates.insert(0, extra)
    usable = [Path(c) for c in candidates if c and Path(c).exists()]
    return max(usable, key=_free_bytes) if usable else Path(tempfile.gettempdir())


@dataclass(frozen=True)
class ResourceProfile:
    """The share of a machine one Talos stage may use."""

    memory_limit: str
    threads: int
    temp_dir: str
    max_temp: str
    detected: bool = True
    note: str = ""

    # ------------------------------------------------------------- detection

    @classmethod
    def detect(cls, lake_root: str | Path | None = None) -> "ResourceProfile":
        memory = total_memory_bytes()
        note = "" if memory else (
            f"could not read physical memory; assuming {FALLBACK_MEMORY_GB} GB")
        usable_gb = max(1, int((memory or FALLBACK_MEMORY_GB * GB) * MEMORY_FRACTION / GB))

        # Never more threads than the memory can feed. This is the thrash cap.
        by_memory = max(1, usable_gb * 1024 // MIN_MEMORY_PER_THREAD_MB)
        threads = max(1, min(os.cpu_count() or 1, by_memory))

        volume = roomiest_volume(lake_root)
        spill = Path(volume) / "talos_duckdb"
        cap_gb = max(1, min(MAX_TEMP_ABSOLUTE_GB,
                            int(_free_bytes(volume) * MAX_TEMP_VOLUME_FRACTION / GB)))

        return cls(memory_limit=f"{usable_gb}GB", threads=threads,
                   temp_dir=str(spill), max_temp=f"{cap_gb}GB",
                   detected=True, note=note)

    @classmethod
    def from_config(cls, doc: dict | None,
                    lake_root: str | Path | None = None) -> "ResourceProfile":
        """Detected values, with anything declared in config.yml winning.

        A partial block is normal and useful: pin `temp_dir` because you know
        which disk is big, and let the rest follow the machine.
        """
        profile = cls.detect(lake_root)
        declared = {k: v for k, v in (doc or {}).items()
                    if k in ("memory_limit", "threads", "temp_dir", "max_temp")
                    and v not in (None, "auto")}
        if not declared:
            return profile
        if "threads" in declared:
            declared["threads"] = int(declared["threads"])
        if "temp_dir" in declared:
            declared["temp_dir"] = str(declared["temp_dir"])
        return replace(profile, detected=False, **declared)

    # --------------------------------------------------------------- helpers

    def ensure_temp_dir(self) -> str:
        """Create the spill directory. DuckDB will not make it for us."""
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        return self.temp_dir

    def override(self, **kwargs) -> "ResourceProfile":
        """A per-stage adjustment. Stages differ: `discover` reads footers only
        and needs nothing; labelling 2019 spills hard."""
        given = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **given, detected=False) if given else self

    def as_duck_kwargs(self) -> dict:
        return {"memory_limit": self.memory_limit, "threads": self.threads,
                "temp_dir": self.temp_dir, "max_temp": self.max_temp}

    def describe(self) -> str:
        how = "detected" if self.detected else "from config"
        lines = [f"resources ({how})",
                 f"  memory    {self.memory_limit}",
                 f"  threads   {self.threads}",
                 f"  spill     {self.temp_dir}  (capped {self.max_temp})"]
        if self.note:
            lines.append(f"  note      {self.note}")
        return "\n".join(lines)
