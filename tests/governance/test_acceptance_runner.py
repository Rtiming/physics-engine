"""验收器的元门禁——轴6规则6在本仓的落地：判据本身也要被验。"""

import os
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


def _collects_nothing(argv: tuple[str, ...]) -> bool:
    """这条命令今天真的一条测试都收集不到吗？

    只加``--collect-only``，其余照抄验收器要跑的那条命令——**判的是同一条命令，
    不是一条长得像的**。实测0.3秒，够便宜到常驻交互级。
    """

    environment = dict(os.environ, PYTHONPATH=str(accept.ROOT / "src"))
    completed = subprocess.run(
        [*argv, "--collect-only"],
        cwd=accept.ROOT,
        capture_output=True,
        env=environment,
        check=False,
    )
    return completed.returncode == accept.EMPTY_SELECTION_RETURNCODE


def test_no_tier_is_exempt_today_because_no_tier_is_empty_today():
    """**理由消失的豁免必须自动失效**（plans/09第七节第2条，决策0056第五节）。

    这里原本正面断言"`batch`档的豁免必须在"，**而batch今天装着40条测试**——
    立豁免时它确实是空的，理由早已消失而豁免被钉死。后果不是理论上的：
    一个`batch` marker笔误会让那40条静默不跑，退出码5被当成"申报过的空档"，
    **验收照绿**。

    改法不是删掉就算：**判据从"名单必须等于某个常量"改成"名单里的每一档
    今天必须实测收集到0条"**。理由还在，豁免就还在；理由没了，当场红。
    """

    assert not (accept.MAY_BE_EMPTY & set(accept.QUICK_COMMANDS)), (
        "交互级绝不许为空——它为空说明测试没被收集，是事故不是状态"
    )
    for argv in sorted(accept.MAY_BE_EMPTY):
        assert _collects_nothing(argv), (
            f"{' '.join(argv)}今天收集得到测试，却还挂着「可以为空」的豁免——"
            "理由消失的豁免等于把marker笔误变成静默通过"
        )


def test_the_batch_tier_is_not_empty_which_is_why_its_exemption_was_removed():
    """**上一条的红分支**：如果`batch`还在豁免名单里，它现在就该红。

    没有这一条，上面那个循环在名单为空时是**零次迭代**——
    一条从没被执行过的分支，正是教训三里域隔离门的形态。
    这里正面把同一个判据跑在`batch`上，**实测它不为空**。
    """

    for argv in accept.BATCH_COMMANDS:
        assert not _collects_nothing(argv), (
            f"{' '.join(argv)}真的收集不到测试了——那本条与上一条都要重新裁"
        )


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


# ---------------------------------------------------------------------------
# 工具门：**2026-08-12由"可选位"升为必需**（决策0053第二节）
# 立本节前，这个行为**一条治理测试都没有**——一个没有门看着的门。
# ---------------------------------------------------------------------------


def test_required_tool_commands_are_all_armed_in_full():
    """工具命令必须**无条件**出现在full档里。

    旧行为是"文件在就上膛，不在就记一条`absent`然后照样PASS"。
    那在单人开发下是诚实的，**在多代理并行下是致命的**：
    一个代理漏带工具文件，全队的门变虚而验收仍报PASS。
    """

    commands, _ = accept.resolve_commands("full", accept.ROOT)
    for argv in accept.REQUIRED_TOOL_COMMANDS:
        assert argv in commands, f"必需工具没上膛：{argv}"


def test_the_capability_ledger_gate_is_one_of_the_required_tools():
    """计数门（决策0056）必须在必需工具里，且它的文件真的在。

    单独立一条而不是靠上一条的循环：上一条遍历的是元组本身，
    **元组里少一行它一样绿**——那正是"门认得自己写的东西"的老毛病。
    这一条把工具名写死在断言里，少一行当场红。
    """

    argv = (".venv/bin/python", "tools/check_capability_ledger.py")
    assert argv in accept.REQUIRED_TOOL_COMMANDS, (
        "计数门没挂进验收器——两个分子就又回到散文里了（0052第二节）"
    )
    assert (accept.ROOT / argv[-1]).is_file()


def test_required_tools_stay_out_of_quick():
    """quick档不挂工具门——它是交互级（30秒预算），工具门属批末。"""

    commands, absent = accept.resolve_commands("quick", accept.ROOT)
    for argv in accept.REQUIRED_TOOL_COMMANDS:
        assert argv not in commands
    assert absent == ()


def test_a_missing_tool_file_still_gets_armed_so_the_run_fails(tmp_path):
    """**必红**：工具文件不在时，命令仍然上膛（于是它会失败），并被记进第二个返回值。

    这条守的是升级本身：**旧代码在这里会把命令从列表里拿掉**，
    于是文件不在=少跑一条=照样PASS。新代码必须相反。
    """

    commands, absent = accept.resolve_commands("full", tmp_path)
    for argv in accept.REQUIRED_TOOL_COMMANDS:
        assert argv in commands, "文件不在也必须上膛，让它自己失败"
        assert argv in absent, "文件不在必须被点名"


def test_the_receipt_field_keeps_its_registered_byte_shape():
    """回执是登记过的面：`absent_optional_commands`这个键**不许改名或消失**。

    语义从"可以缺席的位"变成"哪几个必需工具不见了"，**字节形制不动**——
    改面要走面清册与版本号，而这次升级不值得破一个已发布的面。
    """

    _, absent = accept.resolve_commands("full", accept.ROOT)
    assert absent == (), "主仓四个工具应当齐全"
    assert isinstance([list(argv) for argv in absent], list)


def test_the_old_name_still_resolves_for_one_version():
    """改名要留一版缓冲（AGENTS.md「API两档」）。"""

    assert accept.OPTIONAL_FULL_COMMANDS is accept.REQUIRED_TOOL_COMMANDS
