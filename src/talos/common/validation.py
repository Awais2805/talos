"""The shared fail-fast harness. Every stage's checks run through this.

Validation is a service rather than asserts scattered through business logic for
two reasons. A check written inline is invisible to anything that wants to
report what was verified, so "which checks ran" becomes unanswerable. And
`assert` is removed by `python -O`, which makes it precisely the wrong
instrument for a correctness gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


class GateFailure(Exception):
    """Raised when a fatal check did not pass. Carries the result for reporting."""

    def __init__(self, message: str, result: "GateResult"):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    fatal: bool = True
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        mark = "ok  " if self.passed else ("FAIL" if self.fatal else "warn")
        return f"  {mark}  {self.name}" + (f" -- {self.detail}" if self.detail else "")


@dataclass(frozen=True)
class Check:
    """A named predicate over a stage's context.

    `fn` returns True, or False, or a `(passed, detail)` pair, or a full
    `CheckResult` when it wants to attach evidence. The loosest form is allowed
    because the cheapest possible check should be one line -- a harness that
    makes adding a check tedious ends up with checks written as asserts instead.
    """

    name: str
    fn: Callable[[Mapping[str, Any]], Any]
    fatal: bool = True
    #: Context keys this check reads. Declared so a stage that renames one gets
    #: "the harness did not supply X" rather than a check that appears to have
    #: found a fault in the data -- which, for a fatal check, aborts the run for
    #: entirely the wrong reason.
    requires: tuple[str, ...] = ()

    def run(self, ctx: Mapping[str, Any]) -> CheckResult:
        absent = [k for k in self.requires if k not in ctx]
        if absent:
            return CheckResult(
                self.name, False,
                f"could not run: the stage did not supply {', '.join(absent)}. "
                f"This is a harness fault, not a finding about the data.",
                self.fatal, {"missing_context": absent})
        try:
            outcome = self.fn(ctx)
        except Exception as exc:                    
            # A check that raises is a failed check, not a crashed gate. Letting
            # it propagate would hide every check after it, which is the worst
            # possible moment to lose visibility.
            return CheckResult(self.name, False, f"{type(exc).__name__}: {exc}",
                               self.fatal, {"exception": repr(exc)})

        if isinstance(outcome, CheckResult):
            return outcome
        if isinstance(outcome, tuple):
            passed, detail = outcome
            return CheckResult(self.name, bool(passed), str(detail), self.fatal)
        return CheckResult(self.name, bool(outcome), "", self.fatal)


@dataclass(frozen=True)
class GateResult:
    gate: str
    results: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if not r.passed and r.fatal)

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if not r.passed and not r.fatal)

    @property
    def passed(self) -> bool:
        """True when nothing fatal failed. Warnings do not make a gate fail."""
        return not self.failures

    def report(self) -> str:
        head = f"{self.gate}: {len(self.results)} check(s), " \
               f"{len(self.failures)} failed, {len(self.warnings)} warning(s)"
        return "\n".join([head, *(r.line() for r in self.results)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "checks": [
                {"name": r.name, "passed": r.passed, "fatal": r.fatal,
                 "detail": r.detail, "evidence": dict(r.evidence)}
                for r in self.results
            ],
        }


class ValidationGate:
    """A named set of checks, run together against one context."""

    def __init__(self, name: str):
        self.name = name
        self._checks: list[Check] = []

    def register(self, check: Check) -> "ValidationGate":
        self._checks.append(check)
        return self                                  # chainable, so a stage declares its
                                                     # gate in one expression

    def check(self, name: str, *, fatal: bool = True):
        """Decorator form: `@gate.check("uid is unique")`."""
        def wrap(fn: Callable[[Mapping[str, Any]], Any]):
            self.register(Check(name, fn, fatal))
            return fn
        return wrap

    def __len__(self) -> int:
        return len(self._checks)

    def run_all(self, ctx: Mapping[str, Any]) -> GateResult:
        """Run every check. All of them, even after one fails.

        Stopping at the first failure would report one problem per run, and a
        stage with three genuine faults would take three runs to reveal them --
        on a dataset where a run is measured in hours.
        """
        return GateResult(self.name, tuple(c.run(ctx) for c in self._checks))

    @staticmethod
    def raise_on_fail(result: GateResult) -> None:
        """Raise if anything fatal failed. Takes no other argument, ever."""
        if result.failures:
            names = ", ".join(r.name for r in result.failures)
            raise GateFailure(f"{result.gate}: {names}\n{result.report()}", result)
