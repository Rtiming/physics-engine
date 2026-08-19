"""计数门的必红矩阵——**每个判据分支各被注错验过一次**。

这份文件的组织方式来自plans/09教训三：

> **一条从没被必红用例走过的分支，等于一条没有门的分支。**
> 域隔离门有九条必红——**全部用绝对import**，而它对相对import完全失明。
> 门全绿不是因为它挡得住，是因为那条分支从没被执行过。

所以这里**不按规则组织，按判据分支组织**：`check_capability_ledger.py`里
每一处`raise LedgerError("码", …)`就是一条分支，每条分支各有一条红用例，
每条红用例的docstring写明**注错方式**。

两条元测试把这件事从"我保证覆盖了"变成"覆盖不到就红"：

* `test_every_branch_has_its_own_code`——两处分支不许共用一个错误码
  （共用会让一条红同时"覆盖"两条分支，而其中一条从没被执行）；
* `test_every_branch_code_has_a_red_case`——每个码都要有一条红用例走过。
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools" / "check_capability_ledger.py"
LEDGER = ROOT / "docs" / "capability_ledger.json"


def _load_checker():
    """按路径加载`tools/`下的脚本——它不是包的一部分，不能import。"""

    spec = importlib.util.spec_from_file_location("check_capability_ledger", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    #: 先进`sys.modules`再执行：模块里的`@dataclass`要回查自己的命名空间解析
    #: `int | None`这类延迟注解，没登记时`dataclasses`拿到None当场炸。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _ledger_dict() -> dict:
    """真清单的一份可改副本。**红用例都是从真清单注错出来的**，不是造一份假的——
    造假清单只能证明门认得假清单，证明不了它认得真清单被改坏。"""

    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _bit(data: dict, bit_id: str) -> dict:
    for scenario in data["scenarios"]:
        for bit in scenario["bits"]:
            if bit["id"] == bit_id:
                return bit
    for bit in data["peer_tier_c"]:
        if bit["id"] == bit_id:
            return bit
    raise AssertionError(f"清单里没有{bit_id}——红用例本身过期了")


def _written(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "capability_ledger.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _mutated(tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    data = _ledger_dict()
    mutate(data)
    return _written(tmp_path, data)


def _prose_tree(tmp_path: Path, readme: str, cases_readme: str | None = None) -> Path:
    """造一棵只含本门要读的那两份散文的树（0084第八节的散文对账）。"""

    (tmp_path / "cases").mkdir(exist_ok=True)
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "cases" / "README.md").write_text(
        cases_readme if cases_readme is not None else _GOOD_CASES_README, encoding="utf-8"
    )
    return tmp_path


#: 与仓里两份散文**当前真值**同步的最小复制品。分子一变这里要跟着改——
#: **而它跟着改的那一刻，正是散文对账门在起作用。**
_GOOD_README = (
    "| **主｜用户六场景端到端** | | **0/6** |\n"
    "| 从｜同行C档13条标准案例 | | **6/13** |\n"
    "能力位清单的当前机械计数是**13/42**\n"
)
_GOOD_CASES_README = (
    "| **主｜用户六场景端到端** | | **0/6** |\n"
    "| 从｜同行C档13条标准案例 | | **6/13** |\n"
    "主分母的逐位机械计数当前为**13/42**\n"
)


def _expect_red(code: str, path: Path, root: Path = ROOT):
    """跑门并断言它以**这个分支**红。码写死在调用点上，元测试按调用点数覆盖。"""

    with pytest.raises(checker.LedgerError) as excinfo:
        checker.check(path, root)
    assert excinfo.value.code == code, (
        f"期望分支{code}，实际{excinfo.value.code}：{excinfo.value}"
    )
    return excinfo.value


def _expect_red_prose(code: str, root: Path):
    """散文对账那一族的必红。

    **它不能走`_expect_red`**：散文对账不在`check()`里（那是一条有意的分工，
    见checker的`main()`），所以红要从`assert_prose_matches_counts`那一层取。
    元测试同时数这两个调用点——**两个helper，一张覆盖表**。
    """

    counts = checker.check(LEDGER, ROOT)
    with pytest.raises(checker.LedgerError) as excinfo:
        checker.assert_prose_matches_counts(counts, root)
    assert excinfo.value.code == code, (
        f"期望分支{code}，实际{excinfo.value.code}：{excinfo.value}"
    )
    return excinfo.value


# ---------------------------------------------------------------------------
# 散文对账的必红（0084第八节）：五条分支各一条
# ---------------------------------------------------------------------------


def test_prose_file_missing_is_red(tmp_path: Path):
    """注错：锚点表指着的散文文件根本不在。"""

    (tmp_path / "cases").mkdir()
    _expect_red_prose("PROSE_FILE_MISSING", tmp_path)


def test_prose_anchor_lost_is_red(tmp_path: Path):
    """注错：锚串被改短了。

    **这一条是五条里最要紧的**——锚串失效意味着这一处不再被看着，
    而它不会自己喊。与0084第七节那个"照着复制出来的第二个入口不会跟着修"同族。
    """

    _expect_red_prose(
        "PROSE_ANCHOR_LOST",
        _prose_tree(tmp_path, _GOOD_README.replace("主｜用户六场景端到端", "主｜六场景端到端")),
    )


def test_prose_anchor_ambiguous_is_red(tmp_path: Path):
    """注错：锚串命中两行——说不清在核对哪一处。"""

    doubled = "| **主｜用户六场景端到端** | | **0/6** |\n" + _GOOD_README
    _expect_red_prose("PROSE_ANCHOR_AMBIGUOUS", _prose_tree(tmp_path, doubled))


def test_prose_fraction_absent_is_red(tmp_path: Path):
    """注错：锚住的那一行根本没写这个分母的分数。"""

    _expect_red_prose(
        "PROSE_FRACTION_ABSENT",
        _prose_tree(tmp_path, _GOOD_README.replace("**0/6**", "见台账")),
    )


def test_prose_stale_is_red(tmp_path: Path):
    """注错：散文里的数陈旧了——**这是本门存在的理由本身**。"""

    _expect_red_prose(
        "PROSE_STALE", _prose_tree(tmp_path, _GOOD_README.replace("**0/6**", "**1/6**"))
    )


def test_the_prose_red_cases_are_red_for_the_right_reason(tmp_path: Path):
    """先证明那棵造出来的树在**没有植入**时是绿的。

    否则上面五条红得没有意义——它们可能全部红在"这棵树本来就不合格"上。
    """

    counts = checker.check(LEDGER, ROOT)
    checker.assert_prose_matches_counts(counts, _prose_tree(tmp_path, _GOOD_README))


# ---------------------------------------------------------------------------
# 绿分支：真清单必须过，而且两个分子必须是**算**出来的
# ---------------------------------------------------------------------------


def test_the_real_ledger_passes_and_counts_add_up():
    """绿分支：仓里那份清单过门，且逐场景的位数加起来等于主分母。"""

    counts = checker.check(LEDGER, ROOT)
    assert counts.main_total == sum(entry.total for entry in counts.scenarios)
    assert counts.main_done == sum(entry.done for entry in counts.scenarios)
    assert counts.peer_total == checker.PEER_TIER_C_COUNT
    assert counts.main_done > 0 and counts.peer_done > 0


def test_the_numerator_is_computed_and_not_written_down(tmp_path):
    """**这一条是0052第二节的要害**：分子必须从清单长出来。

    注错方式：把一个done的位改成todo（补上`missing`），
    再数一次——分子必须**恰好少1**。若哪天有人把数字写死，这条当场红。
    """

    before = checker.check(LEDGER, ROOT)

    def demote(data: dict) -> None:
        bit = _bit(data, "S3.1")
        assert bit["status"] == "done"
        bit["status"] = "todo"
        bit["evidence"] = []
        bit["missing"] = "注错用：把它改成未做"

    after = checker.check(_mutated(tmp_path, demote), ROOT)
    assert after.main_done == before.main_done - 1
    assert after.main_total == before.main_total, "分母不该跟着动"


def test_the_report_prints_both_denominators_in_the_ruled_shape():
    """0052要的那句话必须真的被打出来：主分母X/Y位、C档A/B条，并逐场景报第几位。"""

    counts = checker.check(LEDGER, ROOT)
    report = checker.format_report(counts)
    assert f"主分母 {counts.main_done}/{counts.main_total}位" in report
    assert f"C档 {counts.peer_done}/{counts.peer_total}条" in report
    for entry in counts.scenarios:
        assert f"第{entry.done}位／共{entry.total}位" in report


def test_the_cli_exits_zero_on_the_real_tree_and_nonzero_on_a_broken_ledger(tmp_path):
    """端到端：退出码必须分得开三种结局，否则它挂不上`accept.py`。"""

    assert checker.main(["--root", str(ROOT)]) == 0

    broken = _mutated(tmp_path, lambda data: _bit(data, "S3.1").update(evidence=[]))
    assert checker.main(["--root", str(ROOT), "--ledger", str(broken)]) == 1

    assert checker.main(["--root", str(ROOT), "--ledger", str(tmp_path / "nope.json")]) == 2


def test_the_checker_exits_nonzero_as_a_process(tmp_path):
    """进程级：`accept.py`看的是退出码，不是异常。"""

    result = subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(ROOT), "--ledger", str(tmp_path / "x.json")],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 2
    assert "空跑不是通过" in result.stderr


# ---------------------------------------------------------------------------
# 分支：加载与空跑
# ---------------------------------------------------------------------------


def test_a_missing_ledger_is_not_a_pass(tmp_path):
    """**必红**。注错方式：清单文件根本不在。**判不了不等于判过了。**"""

    _expect_red("LEDGER_MISSING", tmp_path / "absent.json")


def test_an_unreadable_ledger_is_not_a_pass(tmp_path):
    """**必红**。注错方式：写一段不是JSON的字节进去。"""

    path = tmp_path / "capability_ledger.json"
    path.write_text("{这不是JSON", encoding="utf-8")
    _expect_red("LEDGER_NOT_JSON", path)


def test_a_ledger_that_is_not_an_object_is_rejected(tmp_path):
    """**必红**。注错方式：顶层写成数组。"""

    _expect_red("NODE_NOT_OBJECT", _written(tmp_path, []))


def test_a_hand_written_total_cannot_get_into_the_ledger(tmp_path):
    """**必红，而且是这道门存在的理由**。

    注错方式：往顶层塞一个`done_count`。严格键集当场拒收——
    **分子在清单里没有落脚点，所以它只可能是算出来的。**
    """

    error = _expect_red(
        "UNKNOWN_KEY", _mutated(tmp_path, lambda data: data.update(done_count=11))
    )
    assert "总数字段" in error.message


def test_a_missing_required_key_is_rejected(tmp_path):
    """**必红**。注错方式：删掉场景的`dominant_domain`。"""

    _expect_red(
        "MISSING_KEY",
        _mutated(tmp_path, lambda data: data["scenarios"][0].pop("dominant_domain")),
    )


def test_an_unknown_schema_version_is_not_judged(tmp_path):
    """**必红**。注错方式：把schema改成没见过的版本——**不认识就不许判**。"""

    _expect_red(
        "SCHEMA_UNKNOWN",
        _mutated(tmp_path, lambda data: data.update(schema="capability_ledger/99")),
    )


def test_zero_scenarios_is_an_empty_run_not_a_pass(tmp_path):
    """**必红**。注错方式：把`scenarios`清空。清空之后什么都不违反，**照样不许绿**。"""

    _expect_red("EMPTY_SCENARIOS", _mutated(tmp_path, lambda data: data.update(scenarios=[])))


def test_zero_peer_entries_is_an_empty_run_not_a_pass(tmp_path):
    """**必红**（同一族的第二条分支）。注错方式：把`peer_tier_c`清空。"""

    _expect_red("EMPTY_PEER_TIER", _mutated(tmp_path, lambda data: data.update(peer_tier_c=[])))


def test_a_scenario_without_an_id_is_rejected(tmp_path):
    """**必红**。注错方式：把场景`id`改成空串。"""

    _expect_red("SCENARIO_ID_MISSING", _mutated(tmp_path, lambda data: _blank_scenario_id(data)))


def _blank_scenario_id(data: dict) -> None:
    data["scenarios"][0]["id"] = ""


def test_a_scenario_with_no_bits_is_rejected(tmp_path):
    """**必红**。注错方式：把某个场景的`bits`清空。

    拆不出能力位的场景**没法报"第几位／共几位"**，而那正是0052要的话。
    """

    _expect_red(
        "SCENARIO_EMPTY", _mutated(tmp_path, lambda data: data["scenarios"][0].update(bits=[]))
    )


def test_a_blank_reason_is_rejected(tmp_path):
    """**必红**。注错方式：把某位的`why`改成空白。

    这一位为什么是这个状态，是清单唯一能被人反驳的地方；空着就不可反驳。
    """

    _expect_red("TEXT_EMPTY", _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(why="   ")))


def test_a_bit_without_an_id_is_rejected(tmp_path):
    """**必红**。注错方式：把某位的`id`改成空串。"""

    _expect_red("BIT_ID_MISSING", _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(id="")))


def test_an_undeclared_status_is_rejected(tmp_path):
    """**必红**。注错方式：把状态写成`almost`——三态之外不许有第四态。"""

    _expect_red(
        "STATUS_UNKNOWN", _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(status="almost"))
    )


def test_the_physics_flag_cannot_be_left_vague(tmp_path):
    """**必红**。注错方式：把`exercises_physics`写成字符串`"yes"`。

    "穿不穿引擎的物理机械"是plans/05第二节那一刀，**不许留白**：
    一个说不清自己验的是公式还是引擎的位，会让"其中穿过物理机械的几位"变成假话。
    """

    _expect_red(
        "PHYSICS_FLAG_SHAPE",
        _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(exercises_physics="yes")),
    )


def test_evidence_must_be_an_array(tmp_path):
    """**必红**。注错方式：把`evidence`写成一个字符串而不是数组。"""

    _expect_red(
        "EVIDENCE_SHAPE",
        _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(evidence="case:norris_thin_strip")),
    )


def test_a_done_bit_may_not_still_list_what_is_missing(tmp_path):
    """**必红**。注错方式：给一个done的位加上`missing`。**还缺东西就不是done。**"""

    _expect_red(
        "MISSING_ON_DONE",
        _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(missing="其实还缺一块")),
    )


def test_a_partial_bit_must_say_where_it_is_partial(tmp_path):
    """**必红，这是0052第二节的原话**："partial必须写明partial在哪"。

    注错方式：删掉一个partial位的`missing`。
    """

    _expect_red(
        "MISSING_ABSENT", _mutated(tmp_path, lambda data: _bit(data, "S3.6").pop("missing"))
    )


def test_an_off_list_structural_gap_is_rejected(tmp_path):
    """**必红**。注错方式：把`shared_gap`写成plans/06四条之外的词。

    自由文本会让"多少位卡在同一堵墙上"这个数立刻失去意义。
    """

    _expect_red(
        "SHARED_GAP_UNKNOWN",
        _mutated(tmp_path, lambda data: _bit(data, "S1.2").update(shared_gap="厚度与体积")),
    )


def test_a_peer_entry_without_an_integer_index_is_rejected(tmp_path):
    """**必红**。注错方式：把C档某条的`index`写成字符串`"7"`。"""

    _expect_red(
        "PEER_INDEX_MISSING", _mutated(tmp_path, lambda data: _bit(data, "C7").update(index="7"))
    )


# ---------------------------------------------------------------------------
# 分支：结构位置（位号、场景集合、C档条号）
# ---------------------------------------------------------------------------


def test_dropping_a_scenario_changes_the_denominator_and_must_be_red(tmp_path):
    """**必红**。注错方式：删掉场景⑥。

    六个场景是用户给的正本，**分母不许被悄悄改小**——少一个场景，
    "端到端0/6"这句话就变成"0/5"，而没有人做过那个决定。
    """

    _expect_red(
        "SCENARIO_SET", _mutated(tmp_path, lambda data: data["scenarios"].pop())
    )


def test_a_bit_id_that_does_not_match_its_position_is_red(tmp_path):
    """**必红**。注错方式：把`S3.4`改名成`S3.9`。

    位号是**结构位置**不是标签：它必须住在自己声明的场景底下、并且连号。
    跳号会让"共几位"这个数变得可以商量。
    """

    _expect_red("BIT_ID_SHAPE", _mutated(tmp_path, lambda data: _bit(data, "S3.4").update(id="S3.9")))


def test_the_frozen_main_denominator_cannot_be_shrunk_by_merging_bits(tmp_path):
    """**必须红**。注错方式：删掉S1最后一位并让剩余位号仍然连续。

    旧门只守连号，因此把两项语义合并后删掉末位会静默把42改成41。
    0057冻结的是**每个场景的位数**，修改必须先走新的决策记录。
    """

    _expect_red(
        "FROZEN_MAIN_DENOMINATOR",
        _mutated(tmp_path, lambda data: data["scenarios"][0]["bits"].pop()),
    )


def test_the_peer_tier_denominator_cannot_be_quietly_resized(tmp_path):
    """**必红**。注错方式：把C档第4条的条号也改成13，于是1—13不再各一次。

    13是外部给定的（research/05第2.3节），**不是我们自己划的及格线**。
    """

    _expect_red("PEER_INDEX_SET", _mutated(tmp_path, lambda data: _bit(data, "C4").update(index=13)))


def test_a_peer_id_that_disagrees_with_its_index_is_red(tmp_path):
    """**必红**。注错方式：把第5条的`id`改成`C05`，与它的条号对不上。"""

    _expect_red("PEER_ID_SHAPE", _mutated(tmp_path, lambda data: _bit(data, "C5").update(id="C05")))


def test_two_bits_with_the_same_label_are_one_bit(tmp_path):
    """**必红**。注错方式：把`S1.2`的名字改成与`S1.1`一模一样。"""

    _expect_red("DUPLICATE_LABEL", _mutated(tmp_path, _copy_label))


def _copy_label(data: dict) -> None:
    _bit(data, "S1.2")["label"] = _bit(data, "S1.1")["label"]


# ---------------------------------------------------------------------------
# 分支：证据——**这道门的要害**
# ---------------------------------------------------------------------------


def test_a_done_bit_with_no_evidence_is_red(tmp_path):
    """**必红**。注错方式：把一个done位的证据清空。做到了就必须指得出来。"""

    _expect_red("EVIDENCE_ABSENT", _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(evidence=[])))


def test_a_todo_bit_carrying_evidence_is_red(tmp_path):
    """**必红**。注错方式：给一个todo位挂上证据。未做就是未做，别挂装饰。"""

    _expect_red(
        "EVIDENCE_ON_TODO",
        _mutated(tmp_path, lambda data: _bit(data, "S1.2").update(evidence=["case:norris_thin_strip"])),
    )


def test_an_unknown_evidence_kind_is_red(tmp_path):
    """**必红**。注错方式：拿一份文档当证据（`doc:…`）。

    文档不是证据——**清单要指向能跑的东西**，否则"可核对"就退回散文。
    """

    _expect_red(
        "EVIDENCE_UNKNOWN_KIND",
        _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(evidence=["doc:plans/05"])),
    )


def test_evidence_pointing_at_a_case_that_does_not_exist_is_red(tmp_path):
    """**必红**。注错方式：指向一个不存在的案例目录。"""

    _expect_red(
        "CASE_MISSING",
        _mutated(tmp_path, lambda data: _bit(data, "S1.1").update(evidence=["case:no_such_case"])),
    )


def test_evidence_pointing_at_a_test_file_that_does_not_exist_is_red(tmp_path):
    """**必红**。注错方式：指向一个不存在的测试文件。"""

    _expect_red(
        "TEST_FILE_MISSING",
        _mutated(
            tmp_path,
            lambda data: _bit(data, "S1.1").update(
                evidence=["test:tests/cases/test_no_such_file.py::test_x"]
            ),
        ),
    )


def test_a_test_name_that_is_only_mentioned_does_not_count_as_registered(tmp_path):
    """**必红，这条是plans/09教训二在本门上的落点**。

    注错方式：指向`test_a_name_that_is_only_mentioned_never_defined`——
    这个名字**确确实实出现在本文件里**（就在这一行的字符串里），
    但它从来没有被`def`定义过。**认得"被提到"的门会绿，认结构位置的门必须红。**

    本仓已经在这个形态上栽过四次：`peer_fcl_distance`挂在"在建"那句话里就算登记、
    `check_gap_register`第一版认子串于是整行删掉照样绿。
    """

    error = _expect_red(
        "TEST_FUNCTION_MISSING",
        _mutated(
            tmp_path,
            lambda data: _bit(data, "S1.1").update(
                evidence=[
                    "test:tests/governance/test_capability_ledger.py"
                    "::test_a_name_that_is_only_mentioned_never_defined"
                ]
            ),
        ),
    )
    assert "没有被定义" in error.message
    source = Path(__file__).read_text(encoding="utf-8")
    assert "test_a_name_that_is_only_mentioned_never_defined" in source, (
        "这条红用例的前提是那个名字真的出现在本文件里——否则它验的不是"
        "『出现过但没被定义』，而只是『不存在』，那与上一条重复"
    )


def test_a_defined_test_in_this_very_file_does_resolve(tmp_path):
    """绿分支（与上一条成对）：同一个文件里**真被定义**的函数，门必须认。

    没有这一条，上面那条红说明不了任何事——它可能只是"这个文件里什么都认不出"。
    """

    path = _mutated(
        tmp_path,
        lambda data: _bit(data, "S1.1").update(
            evidence=[
                "test:tests/governance/test_capability_ledger.py"
                "::test_the_real_ledger_passes_and_counts_add_up"
            ]
        ),
    )
    assert checker.check(path, ROOT).main_done > 0


def test_splitting_one_achievement_into_two_bits_is_red(tmp_path):
    """**必红**。注错方式：把`S3.1`的证据原样复制到`S3.2`上并报done。

    这是计数门最该防的那一手：**把一件已经做成的事拆成两位来抬分子**。
    """

    _expect_red("DUPLICATE_CLAIM", _mutated(tmp_path, _clone_claim))


def _clone_claim(data: dict) -> None:
    source = _bit(data, "S3.1")
    target = _bit(data, "S3.2")
    target["status"] = "done"
    target["evidence"] = list(source["evidence"])
    target.pop("missing", None)


def test_the_same_case_may_serve_both_denominators(tmp_path):
    """绿分支（与上一条成对，且它把规则的第一版打红过）：
    **一个案例同时服务主分母与C档是设计如此**，不是重复计数。

    `fts_instrument_line_shape`既是场景④的一位又是C档第11条——
    第一版的重复判据没有按分母分组，实测当场把这条正当的复用判红了。
    **规则本身错了，不是数据错了。**
    """

    counts = checker.check(LEDGER, ROOT)
    assert counts.main_done and counts.peer_done
    shared = _mutated(
        tmp_path,
        lambda data: _bit(data, "C13").update(
            status="done", evidence=list(_bit(data, "S4.1")["evidence"])
        )
        or _bit(data, "C13").pop("missing", None),
    )
    assert checker.check(shared, ROOT).peer_done == counts.peer_done + 1


# ---------------------------------------------------------------------------
# 元测试：把"每个分支都有红"从保证变成断言
# ---------------------------------------------------------------------------


def _branch_codes() -> list[str]:
    """checker里每一处`raise LedgerError("码", …)`——**认结构位置，不认字符串出现**。"""

    source = CHECKER.read_text(encoding="utf-8")
    return re.findall(r'raise Ledger(?:Error|Empty)\(\s*"([A-Z_]+)"', source)


def test_every_branch_has_its_own_code():
    """两处分支不许共用一个错误码。

    共用会让一条红用例同时"覆盖"两条分支，而其中一条从没被执行过——
    那正是教训三里域隔离门发生的事。**码就是分支的身份。**
    """

    codes = _branch_codes()
    assert codes, "一处raise都没扫到——这不是通过，是空跑"
    repeated = sorted({code for code in codes if codes.count(code) > 1})
    assert not repeated, f"这些码被两处以上的分支共用：{repeated}"


def test_every_branch_code_has_a_red_case():
    """每个分支码都要有一条红用例走过。

    判据落在**调用点的第一个实参**上（`_expect_red("码", …)`），
    不落在"这个码在文件里出现过"——同一条教训二的通则。
    """

    source = Path(__file__).read_text(encoding="utf-8")
    # 两个helper一张覆盖表：`_expect_red`走`check()`，`_expect_red_prose`走
    # `assert_prose_matches_counts`（散文对账不在`check()`里，见checker的`main()`）。
    covered = set(re.findall(r'_expect_red(?:_prose)?\(\s*\n?\s*"([A-Z_]+)"', source))
    uncovered = sorted(set(_branch_codes()) - covered)
    assert not uncovered, f"这些分支没有必红用例：{uncovered}"


def test_every_assertion_function_is_actually_run():
    """checker的每个`assert_*`都必须被`check()`串起来。

    一个写好了却没被调用的断言，与没写没有区别——本仓在别处见过同源的病
    （0039："绊线一旦长期不响就等于被拆了"）。
    """

    source = CHECKER.read_text(encoding="utf-8")
    defined = set(re.findall(r"^def (assert_\w+)\(", source, re.MULTILINE))
    assert defined, "一个断言函数都没扫到——这不是通过，是空跑"
    body = source.split("def check(", 1)[1]
    for name in sorted(defined):
        assert f"{name}(" in body, f"断言{name}没有被check()调用"


def test_the_gate_is_armed_in_the_acceptance_runner():
    """门必须真的挂上验收器，否则它是一份自娱自乐的脚本。"""

    sys.path.insert(0, str(ROOT / "tools"))
    import accept

    commands, _ = accept.resolve_commands("full", ROOT)
    assert (".venv/bin/python", "tools/check_capability_ledger.py") in commands
