"""The three properties that matter downstream: a fatal failure stops the run, a
non-fatal one does not, and a check that throws is reported rather than allowed
to hide the checks behind it.
"""

import inspect

import pytest

from talos.common.validation import (
    Check, CheckResult, GateFailure, ValidationGate,
)


def gate_with(*checks) -> ValidationGate:
    g = ValidationGate("labelling")
    for c in checks:
        g.register(c)
    return g


def test_all_passing_is_a_pass():
    g = gate_with(Check("uid unique", lambda ctx: True),
                  Check("no null labels", lambda ctx: True))
    result = g.run_all({})
    assert result.passed
    assert result.failures == ()
    ValidationGate.raise_on_fail(result)            # must not raise


def test_a_fatal_failure_raises_and_names_the_check():
    g = gate_with(Check("silent rules", lambda ctx: (False, "3 rules matched nothing")))
    result = g.run_all({})

    assert not result.passed
    with pytest.raises(GateFailure) as exc:
        ValidationGate.raise_on_fail(result)
    # Naming the offending check is what makes the refusal actionable.
    assert "silent rules" in str(exc.value)
    assert "3 rules matched nothing" in str(exc.value)
    assert exc.value.result is result


def test_a_non_fatal_failure_is_recorded_but_does_not_stop_the_run():
    """The benign-burst audit is evidence to look at, not grounds to refuse to write."""
    g = gate_with(Check("benign burst", lambda ctx: (False, "556,201 flows"), fatal=False))
    result = g.run_all({})

    assert result.passed                             # warnings do not fail a gate
    assert [w.name for w in result.warnings] == ["benign burst"]
    ValidationGate.raise_on_fail(result)             # must not raise


def test_a_check_that_raises_is_a_failed_check_not_a_crashed_gate():
    """One broken check must not hide every check after it."""
    def explodes(ctx):
        raise KeyError("conn")

    g = gate_with(Check("reads conn", explodes),
                  Check("runs anyway", lambda ctx: True))
    result = g.run_all({})

    assert len(result.results) == 2                  # the second one still ran
    assert result.results[1].passed
    failed = result.failures[0]
    assert failed.name == "reads conn"
    assert "KeyError" in failed.detail


def test_every_check_runs_even_after_one_fails():
    """A run takes hours; reporting one fault per run is not acceptable."""
    seen = []
    g = gate_with(Check("a", lambda ctx: seen.append("a") or False),
                  Check("b", lambda ctx: seen.append("b") or False),
                  Check("c", lambda ctx: seen.append("c") or True))
    result = g.run_all({})
    assert seen == ["a", "b", "c"]
    assert len(result.failures) == 2


def test_checks_receive_the_context():
    g = gate_with(Check("rule count", lambda ctx: (ctx["rules"] == 18, f"{ctx['rules']} rules")))
    result = g.run_all({"rules": 18})
    assert result.passed
    assert result.results[0].detail == "18 rules"


def test_a_check_may_return_a_full_result_with_evidence():
    def with_evidence(ctx):
        return CheckResult("rule breadth", False, "1 rule touches 9 hosts",
                           fatal=False, evidence={"rule_id": 12, "hosts": 9})

    result = gate_with(Check("rule breadth", with_evidence, fatal=False)).run_all({})
    assert result.warnings[0].evidence == {"rule_id": 12, "hosts": 9}


def test_decorator_form_registers():
    g = ValidationGate("conversion")

    @g.check("row count parity")
    def _parity(ctx):
        return ctx["log_rows"] == ctx["parquet_rows"]

    assert len(g) == 1
    assert g.run_all({"log_rows": 10, "parquet_rows": 10}).passed


def test_report_and_dict_expose_what_ran():
    g = gate_with(Check("ok one", lambda ctx: True),
                  Check("bad one", lambda ctx: (False, "nope")))
    result = g.run_all({})

    text = result.report()
    assert "ok one" in text and "bad one" in text and "nope" in text

    payload = result.to_dict()
    assert payload["passed"] is False
    assert {c["name"] for c in payload["checks"]} == {"ok one", "bad one"}


def test_raise_on_fail_has_no_bypass_parameter():
    """The rule, encoded. A gate that can be skipped is not a gate.

    This guards against the flag being added later, which is exactly when the
    check is most likely to be telling the truth.
    """
    params = list(inspect.signature(ValidationGate.raise_on_fail).parameters)
    assert params == ["result"]


def test_a_missing_context_key_is_a_harness_fault_not_a_finding():
    """A stage that renames a context key must not produce a check that appears
    to have found a fault in the data -- for a fatal check that aborts the run
    for entirely the wrong reason."""
    g = gate_with(Check("uid unique", lambda ctx: ctx["distinct_uids"] == 1,
                        requires=("distinct_uids",)))
    result = g.run_all({"total_flows": 10})          # key renamed away

    failed = result.failures[0]
    assert "harness fault" in failed.detail
    assert failed.evidence["missing_context"] == ["distinct_uids"]


def test_a_declared_key_that_is_present_runs_normally():
    g = gate_with(Check("uid unique", lambda ctx: ctx["distinct_uids"] == 10,
                        requires=("distinct_uids",)))
    assert g.run_all({"distinct_uids": 10}).passed
