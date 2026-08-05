"""验收器的元门禁——轴6规则6在本仓的落地：判据本身也要被验。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import accept

from physics_engine.engine_facets import (
    ACCEPTANCE_RECEIPT_FACET,
    ACCEPTANCE_RECEIPT_VERSION,
    ENGINE_REGISTRY,
)


def _result(code: int, elapsed: float = 1.0) -> accept.CommandResult:
    return accept.CommandResult(argv=("x",), returncode=code, elapsed_s=elapsed)


def test_timed_profiles_have_fixed_budgets():
    assert accept.BUDGETS == {"quick": 30.0, "full": 120.0}


def test_quick_commands_are_a_subset_of_full():
    assert set(accept.COMMANDS["quick"]) <= set(accept.COMMANDS["full"])


def test_all_green_within_budget_passes():
    verdicts = accept.classify(
        (_result(0),), budget_s=30.0, timing_mode="development", total_elapsed_s=5.0
    )
    assert verdicts == ("PASS", "PASS", "PASS")


def test_timeout_cannot_be_reported_as_pass():
    verdicts = accept.classify(
        (_result(accept.TIMEOUT_RETURNCODE),),
        budget_s=30.0, timing_mode="development", total_elapsed_s=31.0,
    )
    assert verdicts == ("FAIL", "NOT_COMPLETED", "FAIL")


def test_budget_overrun_without_timeout_still_fails_timing():
    verdicts = accept.classify(
        (_result(0, elapsed=40.0),),
        budget_s=30.0, timing_mode="development", total_elapsed_s=40.0,
    )
    assert verdicts == ("FAIL", "PASS", "FAIL")


def test_functional_mode_never_enforces_timing_but_functional_failures_still_fail():
    ok = accept.classify(
        (_result(0),), budget_s=30.0, timing_mode="functional", total_elapsed_s=999999.0
    )
    assert ok == ("PASS", "PASS", "NOT_ENFORCED")
    bad = accept.classify(
        (_result(1),), budget_s=30.0, timing_mode="functional", total_elapsed_s=1.0
    )
    assert bad == ("FAIL", "FAIL", "NOT_ENFORCED")


def test_zero_executed_commands_can_never_pass():
    verdicts = accept.classify(
        (), budget_s=30.0, timing_mode="development", total_elapsed_s=0.0
    )
    assert verdicts[0] == "BLOCKED"


def test_repository_change_blocks_an_otherwise_passing_acceptance():
    before = accept.RepositoryIdentity("abc", False, "f" * 64)
    after = accept.RepositoryIdentity("abc", True, "0" * 64)
    overall, functional, timing, changed = accept.classify_with_repository(
        ("PASS", "PASS", "PASS"), before, after
    )
    assert (overall, changed) == ("BLOCKED", True)
    assert (functional, timing) == ("PASS", "PASS")


def test_stable_repository_keeps_the_verdict():
    identity = accept.RepositoryIdentity("abc", False, "f" * 64)
    overall, _, _, changed = accept.classify_with_repository(
        ("PASS", "PASS", "PASS"), identity, identity
    )
    assert (overall, changed) == ("PASS", False)


def test_receipt_facet_is_registered_and_reader_compatible():
    ENGINE_REGISTRY.assert_reader_compatible(
        ACCEPTANCE_RECEIPT_FACET, ACCEPTANCE_RECEIPT_VERSION
    )


def test_receipt_facet_rejects_a_future_minor():
    with pytest.raises(Exception, match="untested facet minor"):
        ENGINE_REGISTRY.assert_reader_compatible(ACCEPTANCE_RECEIPT_FACET, "0.9")
