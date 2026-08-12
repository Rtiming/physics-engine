"""案例页校验器的必红矩阵——**补的是plans/09第七节记的那三个洞**。

那三个洞（决策0056第六节逐条兑现）：

1. 判据表**全空单元格照过**——`| | | |`曾是一行合法的判据；
2. **删掉`oracle.json`等于关掉两条校验**——原代码是"有清单才校验"，
   于是删清单成了最省事的过门方式；
3. **案例只在散文里被提到就算登记**——判据是`名字 in 索引页全文`，
   而决策0049第六节记过这个形态：`peer_fcl_distance`长期挂在"在建"那句话里。

本文件按**分支**组织，不按规则组织（plans/09教训三）。每条红用例的docstring
写明注错方式；每条新分支各有一条红，且第一张表与第二张表那条分界另配一条绿——
**没有绿的红说明不了门认得对的东西**。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools" / "check_case_pages.py"

_PAGE = """# demo

## 一、物理/几何设定

半径 1.0 mm。

## 二、参考解出处

教科书闭式。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---|---|---|
| `d` | 1e-12 | 0 | 闭式，机器精度 |

## 四、已知失效清单

无。

## 五、档位与负载级

A档，interactive。

## 六、本案例不是什么

不是一个引擎案例。
"""


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_case_pages", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _case(tmp_path: Path, page: str = _PAGE, *, manifest: bool = True) -> Path:
    """一个刚好合格的案例目录。红用例都是从**合格的那一份**注错出来的。"""

    case_dir = tmp_path / "demo"
    case_dir.mkdir(exist_ok=True)
    (case_dir / "case.md").write_text(page, encoding="utf-8")
    if manifest:
        (case_dir / "criteria.json").write_text(
            json.dumps({"note": "判据正本"}, ensure_ascii=False), encoding="utf-8"
        )
    return case_dir


def _problems(case_dir: Path, indexed: set[str] | None = None) -> list[str]:
    return checker.check_case(case_dir, {"demo"} if indexed is None else indexed)


# ---------------------------------------------------------------------------
# 绿分支
# ---------------------------------------------------------------------------


def test_the_real_case_suite_passes():
    """绿分支：仓里23个案例今天全过——否则下面的红说明不了任何事。"""

    assert checker.main([str(ROOT / "cases")]) == 0


def test_a_minimal_well_formed_case_passes(tmp_path):
    """绿分支：脚手架本身合格。"""

    assert _problems(_case(tmp_path)) == []


# ---------------------------------------------------------------------------
# 洞一：判据表的空单元格
# ---------------------------------------------------------------------------


def test_an_empty_reason_cell_is_red(tmp_path):
    """**必红**。注错方式：把判据行的『理由』一格清空。

    **一张判据表的价值全在第三列**——容差没有理由就是拍脑袋，
    而半年后有人改内核时，能不能判断"这条红了说明什么"全靠它。
    """

    page = _PAGE.replace("| `d` | 1e-12 | 0 | 闭式，机器精度 |", "| `d` | 1e-12 | 0 |  |")
    problems = _problems(_case(tmp_path, page))
    assert any("『理由』是空的" in problem for problem in problems), problems


def test_an_empty_quantity_cell_is_red(tmp_path):
    """**必红**（同一族的第二条分支）。注错方式：把『量』那一格清空。"""

    page = _PAGE.replace("| `d` | 1e-12 | 0 | 闭式，机器精度 |", "|  | 1e-12 | 0 | 理由在 |")
    problems = _problems(_case(tmp_path, page))
    assert any("没写是哪个量" in problem for problem in problems), problems


def test_a_two_column_criteria_row_is_red(tmp_path):
    """**必红**。注错方式：判据行只写两格——判据表是量→rel/abs→理由三件套。"""

    page = _PAGE.replace("| `d` | 1e-12 | 0 | 闭式，机器精度 |", "| `d` | 1e-12 |")
    problems = _problems(_case(tmp_path, page))
    assert any("判据表要三列" in problem for problem in problems), problems


def test_a_criteria_table_without_data_rows_is_red(tmp_path):
    """**必红**。注错方式：只留表头与分隔行，一行判据都不写。"""

    page = _PAGE.replace("| `d` | 1e-12 | 0 | 闭式，机器精度 |\n", "")
    problems = _problems(_case(tmp_path, page))
    assert any("至少要有表头、分隔行与一行判据" in problem for problem in problems), problems


def test_a_criteria_table_without_a_reason_column_is_red(tmp_path):
    """**必红**。注错方式：表头去掉『理由』列。"""

    page = _PAGE.replace("| 量 | rel | abs | 理由 |", "| 量 | rel | abs | 备注 |")
    problems = _problems(_case(tmp_path, page))
    assert any("缺『理由』列" in problem for problem in problems), problems


def test_a_second_table_in_the_same_section_is_not_judged_as_criteria(tmp_path):
    """绿分支，**且它把本轮第一版打红过**。

    判据表那一节允许有第二张表：`generator_determinism`在那里放了一张实测偏差表、
    `mutual_inductance_coaxial`放了一张十二行的必红矩阵，**两张的列数与语义都不同**。
    第一版拿"本节所有竖线开头的行"当判据行，实测把这两个案例判红14条——
    **门在管不该它管的事**。现在只判本节的第一张表。
    """

    page = _PAGE.replace(
        "| `d` | 1e-12 | 0 | 闭式，机器精度 |\n",
        "| `d` | 1e-12 | 0 | 闭式，机器精度 |\n\n实测：\n\n| 量 | 偏差 |\n|---|---|\n| `d` | 0 |\n",
    )
    assert _problems(_case(tmp_path, page)) == []


# ---------------------------------------------------------------------------
# 洞二：删掉清单等于关掉校验
# ---------------------------------------------------------------------------


def test_deleting_the_manifest_is_not_a_way_through_the_gate(tmp_path):
    """**必红**。注错方式：把判据正本删掉。

    原代码是"有`oracle.json`才校验`case_id`与`load_tier`"——
    于是**删掉清单反而变干净**，两条校验一起静默关掉。
    """

    problems = _problems(_case(tmp_path, manifest=False))
    assert any("一份判据正本都没有" in problem for problem in problems), problems


def test_a_peer_style_criteria_file_counts_as_a_manifest(tmp_path):
    """绿分支：`peer_fcl_distance`那种`criteria.json`也是判据正本。

    它2700组对拍写不进案例页，正本只能在文件里。**只认`oracle.json`会误红它，
    而会误红的门最终会被加豁免然后拆掉。**
    """

    case_dir = _case(tmp_path, manifest=False)
    (case_dir / "criteria.json").write_text("{}", encoding="utf-8")
    assert _problems(case_dir) == []


# ---------------------------------------------------------------------------
# 洞三：被提到 ≠ 被登记
# ---------------------------------------------------------------------------


def test_a_case_only_mentioned_in_prose_is_not_registered(tmp_path):
    """**必红，这条是决策0049第六节那个形态的直接落点**。

    注错方式：索引页的正文里写一句"`demo`在建"，但表格里没有它的行。
    旧判据是`名字 in 索引页全文`——**那句话就够它绿**。
    """

    index = "# 索引\n\n`demo`还在建，先不进表。\n\n| 案例 | 判据 |\n|---|---|\n| `other` | x |\n"
    indexed = checker.indexed_case_names(index)
    assert "demo" not in indexed, "『被提到』不许进第一格"
    problems = _problems(_case(tmp_path), indexed)
    assert any("索引**表格**登记" in problem for problem in problems), problems


def test_a_case_in_the_first_cell_of_a_table_row_is_registered():
    """绿分支（与上一条成对）：写进表格第一格才叫登记。"""

    index = "| 案例 | 判据 |\n|---|---|\n| [`demo`](demo/case.md) | 闭式 |\n"
    assert "demo" in checker.indexed_case_names(index)


def test_a_name_in_a_later_cell_does_not_register_it():
    """**必红**（同一条判据的第二个分支）：名字出现在**第二格**也不算登记。

    注错方式：把案例名写进判据那一列。只认第一格是这条校验的全部要害——
    判法与`check_gap_register.py`同源（那道门第一版认子串，必红当场把它打红）。
    """

    index = "| 案例 | 判据 |\n|---|---|\n| [`other`](other/case.md) | 与`demo`对拍 |\n"
    names = checker.indexed_case_names(index)
    assert "other" in names
    assert "demo" not in names


# ---------------------------------------------------------------------------
# 既有分支：本轮没动，但此前一条测试都没有
# ---------------------------------------------------------------------------


def test_a_case_without_a_page_is_red(tmp_path):
    """**必红**。注错方式：目录里没有`case.md`。案例页是案例的一部分，不是附件。"""

    case_dir = tmp_path / "demo"
    case_dir.mkdir()
    assert any("缺 case.md" in problem for problem in _problems(case_dir))


def test_a_missing_required_field_is_red(tmp_path):
    """**必红**。注错方式：删掉『本案例不是什么』一节（Drake形制的负空间声明）。"""

    page = _PAGE.split("## 六、本案例不是什么")[0]
    assert any("缺必填字段" in problem for problem in _problems(_case(tmp_path, page)))


def test_an_empty_required_field_is_red(tmp_path):
    """**必红**。注错方式：留下标题、清空正文——**有标题不等于有内容**。"""

    page = _PAGE.replace("不是一个引擎案例。\n", "")
    assert any("是空的" in problem for problem in _problems(_case(tmp_path, page)))


def test_fields_out_of_order_are_red(tmp_path):
    """**必红**。注错方式：把『参考解出处』挪到『物理/几何设定』前面。"""

    page = _PAGE.replace(
        "## 一、物理/几何设定\n\n半径 1.0 mm。\n\n## 二、参考解出处\n\n教科书闭式。\n",
        "## 二、参考解出处\n\n教科书闭式。\n\n## 一、物理/几何设定\n\n半径 1.0 mm。\n",
    )
    assert any("先后顺序" in problem for problem in _problems(_case(tmp_path, page)))
