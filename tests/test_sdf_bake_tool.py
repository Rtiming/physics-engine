"""`tools/model/sdf_bake/`的判据——甲3（决策0085第四节）。

**本文件跑在主环境里，它一行pcu都不import。** 它判两样：

1. **落盘的那份报告说了什么**（`sphere_probe.report.json`进版本控制）。
   报告是**验证期证人**的产物——形制照`cases/peer_fcl_distance`的`criteria.json`：
   同行库缺席不构成本仓失败，但**已经量到的东西不许悄悄变**；
2. **身份边界**：内核不import本目录。这条与
   `tests/governance/test_model_tools_stay_out_of_the_kernel.py`同源，
   这里只加一条本轨自己的：甲2那个新模块（`contact/field.py`）也不许碰它。

pcu环境在时**再跑一遍活的**（缩小样本），比对报告里的数——
环境不在就按`peer_fcl`那条纪律显式skip并写明理由，**不静默通过**。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/model/sdf_bake"
REPORT = TOOL / "sphere_probe.report.json"
PEER_VENV = TOOL / ".venv/bin/python"


@pytest.fixture(scope="module")
def report() -> dict:
    assert REPORT.is_file(), (
        f"{REPORT}不在——它是进版本控制的证人，不是产物。"
        "没有它，'烘过一次并且量到了这些数'这句话就没有出处。"
    )
    return json.loads(REPORT.read_text(encoding="utf-8"))


# --------------------------- 一、报告说了什么 ---


def test_the_peer_package_is_the_one_the_decision_chose(report: dict) -> None:
    """0074第5.1节选的是pcu 0.34.0（MIT）。**装上了，不是"装不上"。**"""

    assert report["peer"]["package"] == "point-cloud-utils"
    assert report["peer"]["version"] == "0.34.0"
    assert report["peer"]["license"] == "MIT"
    #: pcu**还拉了scipy**——0074第5.1节的选型表没有这一条，作为GAP登记在
    #: `tools/model/sdf_bake/README.md`第五节。这里只钉住"numpy是传递依赖"这个事实。
    assert report["peer"]["numpy"].startswith("2.")


def test_the_baked_sphere_converges_toward_the_analytic_one(report: dict) -> None:
    """腿A：细分加密时，pcu烘出来的值向解析球收敛。

    实测（R = 10 mm、400个查询点、细分1→4）：

    | 细分 | 三角 | max偏差mm | mean偏差mm |
    |---|---|---|---|
    | 1 | 80 | 0.657306 | 0.437526 |
    | 2 | 320 | 0.176336 | 0.114140 |
    | 3 | 1280 | 0.043707 | 0.029281 |
    | 4 | 5120 | 0.013485 | 0.007646 |

    **阶不是干净的4.0000**：max那一列的比是3.7276 / 4.0345 / **3.2411**，
    mean那一列是3.8332 / 3.8980 / 3.8299。如实写下来而不是挑一档报——
    max范数在一层壳上对"哪个点最差"很敏感，而查询点的壳半径逐点在变。
    **这条门判的是"收敛"而不是"恰好二阶"**，判据窗口按实测取``[3.0, 4.5]``；
    要一个干净的二阶，去`tests/test_contact_field.py`那边看插值本身的阶
    （那里是4.0012 / 4.0002），**两处量的不是同一件事**：
    这里量的是三角化，那里量的是插值。
    """

    leg = report["leg_a_and_b_approximation_and_sign"]
    levels = leg["levels"]
    assert [row["subdivisions"] for row in levels] == [1, 2, 3, 4]
    assert [row["triangle_count"] for row in levels] == [80, 320, 1280, 5120]

    for row in levels:
        #: 内接多面体在球内，于是烘出来的距离**处处偏大**。
        #: 符号性质是"误差来自三角化而不是来自pcu"的独立证据。
        assert row["min_signed_deviation_mm"] > 0.0, row

    for name in ("max_abs_deviation_ratios", "mean_abs_deviation_ratios"):
        ratios = leg[name]
        assert len(ratios) == 3
        assert all(3.0 < ratio < 4.5 for ratio in ratios), (name, ratios)

    assert levels[-1]["max_abs_deviation_mm"] < 0.02
    assert levels[-1]["mean_abs_deviation_mm"] < 0.01


def test_the_sign_is_right_everywhere_outside_the_triangulation_band(report: dict) -> None:
    """腿B：符号逐点一致，**四档全部零失配**。

    "除去落在三角化误差带里的点"这个限定不是放水：带内的点真值就在零附近，
    那里"符号对不对"没有意义。带外的点一个都不许错。
    """

    for row in report["leg_a_and_b_approximation_and_sign"]["levels"]:
        assert row["sign_mismatch_outside_the_deviation_band"] == 0, row
        assert row["winding_sign_mismatch_outside_the_deviation_band"] == 0, row
        #: 缠绕数在球心≈1（不是文档说的"负"）——语义差异那一条的量化出口。
        assert 0.99 < row["winding_at_the_centre"] < 1.01, row


def test_the_dirty_mesh_leg_is_the_only_direct_witness_of_the_selection(
    report: dict,
) -> None:
    """腿C：**内核拒、pcu照样对**——这一条是0074第5.1节选型裁决的直接证人。

    实测（细分3、一个三角绕向反过来）：

    * 内核`mesh.mesh_mass_properties`**当场拒**（有向边出现两次）；
    * pcu的有符号距离与干净网格最大差**0.227034 mm**，
      而符号在389个判定点上**一个都没错**；
    * 把**整张**网格反向：有符号距离最大只变**3.86e-07 mm**、符号零失配，
      缠绕数整体翻号（与原值之和的最大绝对值1.52e-07）。

    最后一条是决定性的：**面法向投票法在那里必然全部判反**，
    而缠绕数法一个点都没动。"选pcu是因为网格脏"这句话到这里才有出处。
    """

    leg = report["leg_c_dirty_mesh"]
    assert leg["kernel_refuses_the_dirty_mesh"] is True
    assert "not a closed oriented manifold" in leg["kernel_message"]
    assert leg["decisive_point_count"] > 300
    assert leg["sign_mismatch_outside_the_deviation_band"] == 0
    assert leg["winding_sign_mismatch_outside_the_deviation_band"] == 0
    assert 0.2 < leg["max_abs_deviation_vs_clean_mesh_mm"] < 0.3

    flipped = leg["all_triangles_flipped"]
    assert flipped["sign_mismatch_outside_the_deviation_band"] == 0
    assert flipped["max_abs_signed_distance_change_mm"] < 1.0e-6
    #: **不是零**——写实测量，不写"逐位不变"。
    assert flipped["max_abs_signed_distance_change_mm"] > 0.0
    assert flipped["winding_at_the_centre"] < -0.99
    assert flipped["winding_max_abs_sum_with_clean"] < 1.0e-6


def test_the_two_semantic_differences_are_written_down(report: dict) -> None:
    """两条语义差异必须在报告里有名字。

    **同行库的文档不是金标，实测才是**——这条纪律`cases/peer_fcl_distance`
    立过一次（胶囊定义、四元数次序、三条算子的语义），本目录复用它。
    """

    semantics = report["semantics"]
    assert "triangle_soup_fast_winding_number" in semantics
    assert "signed_distance_to_mesh" in semantics
    assert "0.5" in semantics["triangle_soup_fast_winding_number"]
    assert "3.86e-07" in semantics["signed_distance_to_mesh"]


# --------------------------- 二、身份边界 ---


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_the_new_kernel_modules_import_nothing_from_the_tool() -> None:
    """本轨新增的两个内核模块**一行第三方都不import**。

    `tests/governance/test_model_tools_stay_out_of_the_kernel.py`挡的是`tools`前缀；
    这里多挡一层：`numpy`、`scipy`、`point_cloud_utils`三个名字也不许出现。
    **零运行时依赖是承诺**（AGENTS.md本仓纪律第一条），
    而承诺要有一道门看着它，不是靠记性。
    """

    forbidden = {"numpy", "scipy", "point_cloud_utils", "tools"}
    for relative in ("mesh.py", "contact/field.py"):
        path = ROOT / "src/physics_engine" / relative
        offences = forbidden & _imported_roots(path)
        assert not offences, f"{relative}导入了{sorted(offences)}"


def test_the_import_gate_would_go_red_on_a_planted_import() -> None:
    """必须红：这道门自己得能红。"""

    planted = ast.parse("import numpy as np\nfrom tools.model import sdf_bake\n")
    roots: set[str] = set()
    for node in ast.walk(planted):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert {"numpy", "scipy", "point_cloud_utils", "tools"} & roots == {"numpy", "tools"}


# --------------------------- 三、活的重跑（有环境才跑） ---


@pytest.mark.skipif(
    not PEER_VENV.is_file(),
    reason=(
        "烘焙环境不存在：未找到tools/model/sdf_bake/.venv。"
        "按tools/model/sdf_bake/README.md建环境后本条才会执行"
        "（同行库是验证期证人，缺席不构成本仓失败——但落盘的报告仍然被上面那几条判着）"
    ),
)
def test_rerunning_the_probe_reproduces_the_recorded_numbers(tmp_path: Path) -> None:
    """环境在的时候，**重跑一遍并与落盘的数对上**。

    缩小样本（120个点、细分2与3）以留在交互级；
    比的是"同一份代码在同一台机器上给同一串数"，容差1e-12相对——
    **不写逐字节**：pcu的fast winding number带层次近似，
    本仓没有验过它跨进程逐位可复现，所以不声称那件事。
    """

    destination = tmp_path / "probe.json"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    completed = subprocess.run(
        [
            str(PEER_VENV),
            str(TOOL / "bake_sphere_probe.py"),
            "--out",
            str(destination),
            "--samples",
            "400",
            "--levels",
            "2",
            "3",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    fresh = json.loads(destination.read_text(encoding="utf-8"))
    recorded = json.loads(REPORT.read_text(encoding="utf-8"))
    by_level = {
        row["subdivisions"]: row
        for row in recorded["leg_a_and_b_approximation_and_sign"]["levels"]
    }
    for row in fresh["leg_a_and_b_approximation_and_sign"]["levels"]:
        reference = by_level[row["subdivisions"]]
        for key in ("max_abs_deviation_mm", "mean_abs_deviation_mm"):
            assert row[key] == pytest.approx(reference[key], rel=1.0e-12), (
                row["subdivisions"],
                key,
            )


def test_the_skip_reason_names_what_is_missing() -> None:
    """**禁止静默skip**：上一条的skip理由必须点名缺什么、怎么补。

    这条门守的是"跳过的东西必须写明跳过什么、为什么、什么条件下解封"
    （案例页第四节那条纪律的测试侧对应物）。
    """

    marks = [
        mark
        for mark in test_rerunning_the_probe_reproduces_the_recorded_numbers.pytestmark
        if mark.name == "skipif"
    ]
    assert len(marks) == 1
    reason = marks[0].kwargs["reason"]
    assert "tools/model/sdf_bake/.venv" in reason
    assert "README" in reason


if __name__ == "__main__":  # pragma: no cover - 手跑时的便利
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
