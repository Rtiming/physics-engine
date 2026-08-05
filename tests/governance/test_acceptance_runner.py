"""验收器的元门禁——轴6规则6在本仓的落地：判据本身也要被验。"""

import subprocess
import sys
import time
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


# ── T1新增判据的元门禁（轴6规则4/5与空档位；判据本身也要被验）────────


def test_unqualified_resource_never_evaluates_timing_but_leaves_functional_alone():
    """0014法则2：负载敏感只允许存在于计时裁决，功能结论一个字不改。"""

    overall, functional, timing = accept.classify(
        (_result(0, elapsed=999.0),),
        budget_s=30.0,
        timing_mode="development",
        total_elapsed_s=999.0,
        resource=accept.RESOURCE_UNQUALIFIED,
    )
    assert timing == "NOT_EVALUATED"  # 不是FAIL——超预算但机器不合格，不予评判
    assert functional == "PASS"
    assert overall == "PASS"


def test_unqualified_resource_cannot_rescue_a_functional_failure():
    """反向：资源不合格不是免死金牌，功能红照样红。"""

    overall, functional, _ = accept.classify(
        (_result(1),),
        budget_s=30.0,
        timing_mode="development",
        total_elapsed_s=1.0,
        resource=accept.RESOURCE_UNQUALIFIED,
    )
    assert (overall, functional) == ("FAIL", "FAIL")


def test_resource_verdicts_cover_qualified_unqualified_and_unknown():
    limit = accept.LOAD_PER_CPU_LIMIT
    assert accept.classify_resource(limit * 4 - 0.1, 4) == accept.RESOURCE_QUALIFIED
    assert accept.classify_resource(limit * 4 + 0.1, 4) == accept.RESOURCE_UNQUALIFIED
    assert accept.classify_resource(None, 4) == accept.RESOURCE_UNKNOWN
    assert accept.classify_resource(1.0, None) == accept.RESOURCE_UNKNOWN


def test_performance_is_not_evaluable_when_the_execution_tree_moved():
    """轴6规则5：性能回执绑执行树——树变了，上一轮的性能结论作废。"""

    verdict, reason = accept.classify_performance(
        execution_tree_stable=False,
        resource=accept.RESOURCE_QUALIFIED,
        functional="PASS",
    )
    assert verdict == accept.PERFORMANCE_NOT_EVALUATED
    assert reason


def test_performance_is_evaluable_only_on_a_clean_qualified_passing_run():
    assert accept.classify_performance(
        execution_tree_stable=True,
        resource=accept.RESOURCE_QUALIFIED,
        functional="PASS",
    ) == (accept.PERFORMANCE_EVALUABLE, "")
    for kwargs in (
        {"execution_tree_stable": True, "resource": accept.RESOURCE_UNKNOWN,
         "functional": "PASS"},
        {"execution_tree_stable": True, "resource": accept.RESOURCE_QUALIFIED,
         "functional": "FAIL"},
    ):
        verdict, reason = accept.classify_performance(**kwargs)
        assert verdict == accept.PERFORMANCE_NOT_EVALUATED
        assert reason


def test_a_declared_empty_tier_passes_but_an_undeclared_one_must_be_red():
    """空档位只对申报过的命令视同通过——marker笔误必须红，不许静默不跑。"""

    declared = accept.CommandResult(
        argv=("pytest", "-m", "batch"), returncode=accept.EMPTY_SELECTION_RETURNCODE,
        elapsed_s=0.1,
    )
    typo = accept.CommandResult(
        argv=("pytest", "-m", "btach"), returncode=accept.EMPTY_SELECTION_RETURNCODE,
        elapsed_s=0.1,
    )
    allowed = frozenset({("pytest", "-m", "batch")})
    assert accept.classify(
        (declared,), budget_s=30.0, timing_mode="development",
        total_elapsed_s=1.0, may_be_empty=allowed,
    )[:2] == ("PASS", "PASS")
    assert accept.classify(
        (typo,), budget_s=30.0, timing_mode="development",
        total_elapsed_s=1.0, may_be_empty=allowed,
    )[:2] == ("FAIL", "FAIL")


def test_only_the_batch_tier_is_allowed_to_be_empty():
    """交互级绝不许为空——它为空说明测试没被收集，是事故不是状态。"""

    assert accept.MAY_BE_EMPTY == frozenset(accept.BATCH_COMMANDS)
    assert not (accept.MAY_BE_EMPTY & set(accept.QUICK_COMMANDS))


def test_hard_timeout_is_a_liveness_guard_not_the_sla():
    """真正杀进程的线必须远高于预算，否则宿主负载会改变功能结论。"""

    assert accept.HARD_TIMEOUT_FACTOR > 1.0


def test_timeout_kills_the_whole_descendant_tree(tmp_path):
    """accept.py:152的洞：只杀直接子进程会让派生子进程的对拍脚本挂死。"""

    marker = tmp_path / "grandchild-still-alive"
    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', "
        f"\"import time; time.sleep(30); open(r'{marker}', 'w').write('x')\"])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    code = accept.run_command(
        (sys.executable, str(script)), cwd=tmp_path, timeout_s=1.0
    )
    assert code == accept.TIMEOUT_RETURNCODE
    time.sleep(2.0)
    assert not marker.exists(), "后代进程在超时后仍然活着——进程树没被杀干净"


def test_repository_identity_survives_untracked_symlinks_and_tracks_their_target(tmp_path):
    """未跟踪的符号链接不得让验收器崩——worktree里软链.venv就会踩到。

    语义：链接的内容是它的**目标字符串**。跟随链接读目标字节会把仓外的东西
    算进仓库身份；直接read_bytes()在目标是目录时抛IsADirectoryError。
    """

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    target_dir = tmp_path / "target-dir"
    target_dir.mkdir()
    link = tmp_path / "link-to-dir"
    link.symlink_to(target_dir)

    first = accept.repository_identity(tmp_path)  # 修复前这里抛IsADirectoryError

    link.unlink()
    other = tmp_path / "other-dir"
    other.mkdir()
    link.symlink_to(other)
    second = accept.repository_identity(tmp_path)
    assert first.working_tree_sha256 != second.working_tree_sha256, (
        "改了符号链接指向却没进指纹——那是一次真实的工作区变化"
    )
