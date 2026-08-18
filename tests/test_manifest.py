"""The manifest layer: schedules in, joinable rules out.

Weighted towards the conversions that fail *silently*. A wrong UTC offset or a
dropped final minute does not raise — it produces a labelled table that looks
complete and is not, which is why the recovered implementation carried scar
tissue for both. Several cases below are those scars, written down.
"""

import textwrap

import pytest

from talos.data.labelling import TaxonomyError, TaxonomyMapper
from talos.data.labelling.schedule import (
    ManifestError, ManifestLoader, QuarantineError, QuarantineLoader, WindowError,
)
from talos.data.labelling.schedule.windows import (
    Window, WindowError, extend_windows, ips, parse_offset, subnet_prefix, to_utc,
)


def window(rid=0, name="w", t0=0.0, t1=60.0, attackers=None, victims=("v",),
           subnet=None, date="2017-07-04", start="09:20", end="10:20") -> Window:
    return Window(rid=rid, name=name, t0=t0, t1=t1, t1_raw=t1,
                  attackers=attackers, victims=victims, subnet_prefix=subnet,
                  date=date, start=start, end=end)


# ------------------------------------------------------------ time conversion

def test_to_utc_applies_the_declared_offset():
    """09:20 local at -03:00 is 12:20 UTC."""
    from datetime import datetime, timezone
    assert to_utc("2017-07-04", "09:20", "-03:00") == datetime(
        2017, 7, 4, 12, 20, tzinfo=timezone.utc).timestamp()


def test_the_same_wall_clock_at_two_offsets_is_an_hour_apart():
    """2019's two capture days sit either side of a DST change; this is the gap."""
    adt = to_utc("2018-11-03", "09:43", "-03:00")
    ast = to_utc("2018-11-03", "09:43", "-04:00")
    assert ast - adt == 3600


def test_a_date_object_from_yaml_is_accepted():
    """Unquoted `2017-07-04` in YAML parses to a date, not a string."""
    from datetime import date
    assert to_utc(date(2017, 7, 4), "09:20", "-03:00") == to_utc("2017-07-04", "09:20", "-03:00")


def test_an_unreadable_offset_is_refused_not_guessed():
    with pytest.raises(WindowError):
        parse_offset("Atlantic")


# --------------------------------------------------- minute-inclusive windows

def test_a_window_is_extended_through_the_end_of_its_final_minute():
    """'13:29-13:34' means through 13:34:59. Reading it as :00 drops a minute
    from every rule -- 2,508,616 flows when this was last wrong."""
    w = window(t0=0.0, t1=600.0)
    assert extend_windows([w], clamp_starts=[])[0].t1 == 660.0


def test_the_extension_is_clamped_at_the_next_window_start():
    """Back-to-back attacks must never overlap; the shared minute goes to the
    window that STARTS on it, because a window runs up to its successor."""
    first, second = window(rid=0, t0=0.0, t1=600.0), window(rid=1, t0=600.0, t1=900.0)
    out = extend_windows([first, second], clamp_starts=[0.0, 600.0])
    assert out[0].t1 == 600.0 and out[0].extended_by == 0.0     # clamp bound exactly
    assert out[1].t1 == 960.0 and out[1].extended_by == 60.0    # clamp did not bind


def test_the_unextended_end_is_kept_for_audit():
    out = extend_windows([window(t0=0.0, t1=600.0)], clamp_starts=[])[0]
    assert out.t1_raw == 600.0 and out.extended_by == 60.0


# -------------------------------------------------------------- endpoint spec

def test_absent_victims_is_not_the_same_as_an_empty_list():
    """A subnet rule carries no victim_ips. Reading that as 'no constraint' once
    made a rule match every flow in its window."""
    assert ips([]) is None and ips(None) is None
    assert ips(["192.168.10.50"]) == ("192.168.10.50",)


def test_a_window_with_no_endpoint_constraint_cannot_be_constructed():
    """Unconstructable, not merely detectable: a loader could forget to check."""
    with pytest.raises(WindowError, match="every flow"):
        window(attackers=None, victims=None, subnet=None)


def test_a_window_that_ends_before_it_starts_cannot_be_constructed():
    with pytest.raises(WindowError):
        window(t0=600.0, t1=0.0)


def test_a_subnet_prefix_without_its_trailing_dot_is_refused():
    """`192.168.100.5` starts with `192.168.10`, so a /24 rule missing the dot
    silently swallows the neighbouring range."""
    with pytest.raises(WindowError, match="end with a dot"):
        window(victims=None, subnet="192.168.10")


def test_every_bad_window_is_reported_not_just_the_first():
    """A manifest with four mistakes should cost one round trip, not four."""
    from talos.data.labelling.schedule.windows import build_all
    with pytest.raises(WindowError) as exc:
        build_all(lambda spec: window(**spec),
                  [{"t0": 600.0, "t1": 0.0}, {"attackers": None, "victims": None}])
    assert str(exc.value).count("\n") >= 2


def test_only_slash_24_subnets_are_accepted():
    """Matching is a string prefix test, so anything else would silently mismatch."""
    assert subnet_prefix("192.168.10.0/24") == "192.168.10."
    assert subnet_prefix(None) is None
    with pytest.raises(WindowError, match="only /24"):
        subnet_prefix("10.0.0.0/8")


# ------------------------------------------------------------------- taxonomy

def test_requires_payload_keys_on_the_raw_name_not_the_class():
    """DoS-Hulk and DoS-Slowloris are both `dos`, and only one needs payload.

    Slow-DoS holds sockets open by sending almost nothing, so a class-level rule
    would mark real attack traffic as mere attempts.
    """
    tax = TaxonomyMapper()
    assert tax.canonical("DoS-Hulk") == tax.canonical("DoS-Slowloris") == "dos"
    assert tax.requires_payload("DoS-Hulk")
    assert not tax.requires_payload("DoS-Slowloris")


def test_scans_and_floods_are_exempt_from_the_payload_test():
    """For these, an empty connection IS the mechanism."""
    tax = TaxonomyMapper()
    for name in ("PortScan", "SYN", "UDP", "Infiltration-PortScan"):
        assert not tax.requires_payload(name), name


def test_the_nine_canonical_classes_are_declared():
    assert set(TaxonomyMapper().classes) == {
        "benign", "brute_force", "dos", "ddos", "web_attack",
        "heartbleed", "infiltration", "botnet", "portscan"}


def test_an_unmapped_attack_name_raises_rather_than_defaulting():
    """A silent fallback to benign would delete an attack class."""
    with pytest.raises(TaxonomyError, match="Nonexistent-Attack"):
        TaxonomyMapper().canonical("Nonexistent-Attack")


def test_a_mapping_onto_an_undeclared_class_is_caught_at_load(tmp_path):
    p = tmp_path / "taxonomy.yaml"
    p.write_text(textwrap.dedent("""
        canonical_classes: {benign: x, dos: y}
        raw_to_canonical: {DoS-Hulk: dos, Weird-One: typo_class}
    """))
    with pytest.raises(TaxonomyError, match="typo_class"):
        TaxonomyMapper(p)


# ------------------------------------------------------- the real manifests

@pytest.mark.parametrize("dataset,n_rules", [
    ("cic-ids-2017", 18), ("cic-ids-2018", 22), ("cic-ddos-2019", 19)])
def test_every_manifest_loads_with_the_expected_rule_count(dataset, n_rules):
    assert len(ManifestLoader().load(dataset).rules) == n_rules


def test_2017_infiltration_portscan_is_a_subnet_rule():
    rules = {r.name: r for r in ManifestLoader().load("cic-ids-2017").rules}
    scan = rules["Infiltration-PortScan"]
    assert scan.subnet_prefix == "192.168.10."
    assert scan.victims is None                  # "all other clients"
    assert scan.canonical == "portscan"          # not infiltration -- see the manifest


def test_2017_ddos_carries_the_nat_alias():
    """Without 172.16.0.1 this rule matched zero flows while 94,614 sat in the window."""
    loit = next(r for r in ManifestLoader().load("cic-ids-2017").rules
                if r.name == "DDoS-LOIT")
    assert "172.16.0.1" in loit.attackers
    assert len(loit.attackers) == 4


def test_2019_rules_are_victim_only():
    """Reflection attacks spoof their sources, so the attacker carries no signal."""
    rules = ManifestLoader().load("cic-ddos-2019").rules
    assert all(r.victim_only for r in rules)
    assert all(r.victims for r in rules)


def test_2019_resolves_a_different_offset_per_capture_day():
    """Day 1 is ADT, day 2 is AST. A dataset-level constant cannot express this."""
    from datetime import datetime, timezone
    rules = {(r.name, r.date): r for r in ManifestLoader().load("cic-ddos-2019").rules}
    day1 = rules[("PortMap", "2018-11-03")]
    assert day1.t0 == datetime(2018, 11, 3, 12, 43, tzinfo=timezone.utc).timestamp()
    day2 = rules[("NTP", "2018-12-01")]
    assert day2.t0 == datetime(2018, 12, 1, 14, 35, tzinfo=timezone.utc).timestamp()


def test_2019_shared_minute_goes_to_the_window_that_starts_on_it():
    """WebDDoS ends 13:29 and SYN starts 13:29 -- the only touching pair in the lake.

    The manifest documents the resolution explicitly, so it is asserted here
    rather than left to a clamp nobody reads.
    """
    rules = [r for r in ManifestLoader().load("cic-ddos-2019").rules
             if r.name in ("WebDDoS", "SYN") and r.date == "2018-12-01"]
    web = next(r for r in rules if r.name == "WebDDoS")
    syn = next(r for r in rules if r.name == "SYN")
    assert web.extended_by == 0.0        # clamped to zero by SYN's start
    assert syn.extended_by == 60.0
    assert web.t1 == syn.t0              # they tile, they do not overlap


def test_an_unknown_dataset_says_what_is_available():
    with pytest.raises(ManifestError, match="cic-ids-2017"):
        ManifestLoader().load("cic-ids-2020")


# ----------------------------------------------------------------- quarantine

@pytest.mark.parametrize("dataset,n_regions", [
    ("cic-ids-2017", 0), ("cic-ids-2018", 4), ("cic-ddos-2019", 15)])
def test_quarantine_regions_load(dataset, n_regions):
    """Regions come back on the Manifest, from the same parse as the rules."""
    assert len(ManifestLoader().load(dataset).regions) == n_regions


def test_a_region_is_the_same_window_type_as_a_rule():
    """So the endpoint predicate applied to a region is literally the rule's."""
    region = ManifestLoader().load("cic-ids-2018").regions[0]
    assert isinstance(region, Window)
    assert region.reason == "unlabellable-second-stage"
    assert region.justification                        # prose, not empty


def test_regions_clamp_against_the_attack_schedule():
    """A region must never reach into a window and withdraw adjudicable flows.

    Clamped against the rules from the SAME parse -- the loader no longer takes
    starts a caller could have got from a different read of the file.
    """
    manifest = ManifestLoader().load("cic-ids-2018")
    starts = [r.t0 for r in manifest.rules]
    for region in manifest.regions:
        later = [s for s in starts if s >= region.t1_raw]
        if later:
            assert region.t1 <= min(later)


def test_a_region_without_a_justification_is_refused(tmp_path):
    """An undocumented region is a silent narrowing of the dataset."""
    (tmp_path / "toy.yaml").write_text(textwrap.dedent("""
        dataset: toy
        utc_offset: "-03:00"
        attacks:
          - {name: PortScan, date: 2017-07-07, start: "13:55", end: "14:36",
             attacker_ips: [10.0.0.1], victim_ips: [10.0.0.2]}
        quarantine:
          - {reason: no-reason-given, date: 2017-07-07, start: "15:00", end: "15:10",
             victim_ips: [10.0.0.2]}
    """))
    with pytest.raises(QuarantineError, match="justification"):
        ManifestLoader(tmp_path).load("toy")


def test_a_reason_must_be_an_identifier_not_prose(tmp_path):
    """It is written into every quarantined row and grouped on downstream."""
    (tmp_path / "toy.yaml").write_text(textwrap.dedent("""
        dataset: toy
        utc_offset: "-03:00"
        attacks:
          - {name: PortScan, date: 2017-07-07, start: "13:55", end: "14:36",
             attacker_ips: [10.0.0.1], victim_ips: [10.0.0.2]}
        quarantine:
          - {reason: "We could not tell what this was.", justification: "because",
             date: 2017-07-07, start: "15:00", end: "15:10", victim_ips: [10.0.0.2]}
    """))
    with pytest.raises(QuarantineError, match="identifier"):
        ManifestLoader(tmp_path).load("toy")
