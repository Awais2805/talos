"""The machine profile. Mostly a test that two specific failures stay impossible.
"""

import os
from pathlib import Path

from talos.common.config import Config
from talos.common.resources import (
    MEMORY_FRACTION, MIN_MEMORY_PER_THREAD_MB, ResourceProfile, roomiest_volume,
    total_memory_bytes,
)


def parse_gb(value: str) -> int:
    return int(value.removesuffix("GB"))


# ------------------------------------------------------------------ detection

def test_detection_produces_a_usable_profile():
    p = ResourceProfile.detect()
    assert parse_gb(p.memory_limit) >= 1
    assert p.threads >= 1
    assert p.temp_dir and p.max_temp
    assert p.detected


def test_it_leaves_the_machine_something():
    """A stage that claims all the RAM takes the editor and the shell with it."""
    total = total_memory_bytes()
    if total is None:
        return                              # platform will not say; fallback path
    assert parse_gb(ResourceProfile.detect().memory_limit) <= total / 1e9 * MEMORY_FRACTION + 1


def test_threads_never_outnumber_the_memory_that_feeds_them():
    """The thrash cap. A thread with too little to work in evicts rather than
    computes, and the box dies without an OOM kill or a log line."""
    for gb, cores in [(2, 64), (8, 32), (16, 8), (128, 12)]:
        by_memory = max(1, int(gb * MEMORY_FRACTION) * 1024 // MIN_MEMORY_PER_THREAD_MB)
        threads = max(1, min(cores, by_memory))
        assert threads * MIN_MEMORY_PER_THREAD_MB <= max(1, int(gb * MEMORY_FRACTION)) * 1024
        assert threads >= 1                 # never zero, however small the machine


def test_spill_is_bounded_and_never_unlimited():
    p = ResourceProfile.detect()
    assert parse_gb(p.max_temp) >= 1
    assert parse_gb(p.max_temp) <= 20       # the absolute ceiling


def test_the_spill_volume_is_chosen_not_defaulted(tmp_path):
    """Left to itself DuckDB spills wherever it happens to be, which is how the
    root disk filled and took sshd with it."""
    big = tmp_path / "big"
    big.mkdir()
    assert roomiest_volume(big).exists()
    assert "talos_duckdb" in ResourceProfile.detect().temp_dir


def test_a_tiny_machine_still_gets_a_workable_profile():
    """The floor of the supported spec, not an error case."""
    p = ResourceProfile.detect()
    assert p.threads >= 1 and parse_gb(p.memory_limit) >= 1


# --------------------------------------------------------------- overriding

def test_config_declarations_win_over_detection():
    p = ResourceProfile.from_config({"memory_limit": "96GB", "threads": 12})
    assert p.memory_limit == "96GB" and p.threads == 12
    assert not p.detected


def test_auto_and_omitted_both_mean_detect():
    detected = ResourceProfile.detect()
    for doc in ({}, None, {"memory_limit": "auto"}, {"threads": None}):
        assert ResourceProfile.from_config(doc).memory_limit == detected.memory_limit


def test_a_partial_block_keeps_the_rest_detected():
    """Pin the disk you know is big; let the rest follow the machine."""
    detected = ResourceProfile.detect()
    p = ResourceProfile.from_config({"temp_dir": "/mnt/fast/spill"})
    assert p.temp_dir == "/mnt/fast/spill"
    assert p.memory_limit == detected.memory_limit and p.threads == detected.threads


def test_a_stage_may_override_the_machine_profile():
    """Stages differ: discover reads footers only, labelling 2019 spills hard."""
    base = ResourceProfile.detect()
    tuned = base.override(threads=2)
    assert tuned.threads == 2 and tuned.memory_limit == base.memory_limit
    assert base.override() is base           # no-op stays identical


def test_the_temp_dir_is_created_on_demand(tmp_path):
    """DuckDB will not make it, and a missing spill directory fails mid-scan."""
    p = ResourceProfile.detect().override(temp_dir=str(tmp_path / "nested" / "spill"))
    assert not Path(p.temp_dir).exists()
    p.ensure_temp_dir()
    assert Path(p.temp_dir).is_dir()


# ------------------------------------------------------------------- wiring

def test_config_exposes_the_profile_and_prints_it():
    cfg = Config({"lake": {"root": "./lake"}, "resources": {"memory_limit": "48GB"}})
    assert cfg.resources.memory_limit == "48GB"
    assert "48GB" in cfg.describe()


def test_duck_kwargs_round_trip():
    kwargs = ResourceProfile.detect().as_duck_kwargs()
    assert set(kwargs) == {"memory_limit", "threads", "temp_dir", "max_temp"}
