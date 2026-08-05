"""案例`peer_fcl_distance`的conformance测试——同行库对拍（0015第二条）。

轴7规则3的三条纪律在本文件里的落法：

1. **调生产内核**：被验的是`physics_engine.collision`里那两个真在算的东西，
   由`validation/peer_fcl/run_comparison.py`在同行环境里调用，本测试不另写一份；
2. **容差从清单读**：全部判据值来自`cases/peer_fcl_distance/criteria.json`，
   本文件里**没有一个硬编码的容差数字**；且要求产物内嵌的那份criteria与仓内
   逐字节相同——包自足与仓内声明不一致时，判红；
3. **不复述公式**：本文件不算距离，只读产物、比判据。

**同行库缺席时skip并给理由**（不是fail）：本仓的accept绝不能因为一台机器没装
同行库而红——同行库是证人不是依赖。但skip必须说清是哪一步缺，否则"永远绿"
和"永远skip"在回执里长得一样。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from physics_engine.canonical import strict_loads
from physics_engine.engine_facets import ORACLE_MANIFEST_FACET
from physics_engine.run_package import read_verified_package

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "cases" / "peer_fcl_distance"
CRITERIA_PATH = CASE_DIR / "criteria.json"
RUNNER = ROOT / "validation" / "peer_fcl" / "run_comparison.py"
MANIFEST_NAME = "peer_comparison.manifest.json"

#: 同行环境的解释器。POSIX与Windows的venv布局不同，两边都试。
PEER_PYTHON_CANDIDATES = (
    ROOT / "validation" / ".venv" / "bin" / "python",
    ROOT / "validation" / ".venv" / "Scripts" / "python.exe",
)


def _criteria() -> dict:
    return json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))


def _peer_python() -> Path | None:
    return next((path for path in PEER_PYTHON_CANDIDATES if path.is_file()), None)


def _require_peer() -> Path:
    interpreter = _peer_python()
    if interpreter is None:
        pytest.skip(
            "同行库环境不存在：未找到validation/.venv。"
            "按validation/README.md建环境后本案例才会执行"
            "（同行库是验证期证人，缺席不构成本仓失败）"
        )
    if not RUNNER.is_file():  # pragma: no cover - 仓内文件缺失属结构性错误
        pytest.skip(f"对拍脚本缺失：{RUNNER.relative_to(ROOT)}")
    probe = subprocess.run(
        [str(interpreter), "-c", "import fcl, numpy"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if probe.returncode != 0:
        pytest.skip(
            "同行库不可用：validation/.venv里import fcl/numpy失败——"
            f"{probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else '无错误输出'}。"
            "按validation/README.md重建环境"
        )
    return interpreter


def _run_sweep(
    interpreter: Path, out_dir: Path, name: str, *, per_cell: int, fault: str
) -> tuple[int, dict[str, bytes]]:
    completed = subprocess.run(
        [
            str(interpreter),
            str(RUNNER),
            "--out-dir",
            str(out_dir),
            "--name",
            name,
            "--per-cell",
            str(per_cell),
            "--fault",
            fault,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    package = out_dir / name
    assert package.is_dir(), (
        f"对拍脚本没有产出run package（fault={fault}）：\n"
        f"stdout={completed.stdout}\nstderr={completed.stderr}"
    )
    contents = read_verified_package(
        package,
        manifest_name=MANIFEST_NAME,
        extract_declared_sha256s=lambda raw: tuple(
            strict_loads(raw)["payload_sha256"].values()
        ),
    )
    return completed.returncode, contents


def test_criteria_page_declares_a_reason_for_every_tolerance() -> None:
    """判据表必须逐条带理由（plans/02案例页六必填字段第③条）。

    这条不依赖同行库，因此在任何机器上都执行——它守的是"容差不许没有来历"。
    """

    criteria = _criteria()
    assert criteria["criteria"], "判据表不得为空"
    for item in criteria["criteria"]:
        assert item["id"], "判据必须有编号"
        assert item["quantity"], f"{item['id']}缺被判的量"
        assert item["rule"], f"{item['id']}缺判据规则"
        assert len(item.get("reason", "")) >= 20, f"{item['id']}的理由太短或缺失"
    assert len(criteria["must_be_red"].get("reason", "")) >= 20
    expected = criteria["expected_peer_operator"]
    assert len(expected) == 9, "三类形状 × 三个band，算子路由表必须九条齐"


@pytest.mark.batch
def test_peer_fcl_distance_conformance(tmp_path: Path) -> None:
    """同一批固定种子输入，本仓闭式解与FCL逐点对比，判据全过。"""

    interpreter = _require_peer()
    criteria = _criteria()
    returncode, contents = _run_sweep(
        interpreter,
        tmp_path,
        "clean",
        per_cell=criteria["sweep"]["per_cell"],
        fault="none",
    )

    manifest = strict_loads(contents[MANIFEST_NAME])
    verdict = strict_loads(contents["verdict.json"])
    summary = strict_loads(contents["summary.json"])

    # 产物内嵌的判据必须与仓内逐字节相同——否则"包自足"是假的。
    assert contents["criteria.json"] == _canonical_repo_criteria(), (
        "run package里的criteria.json与仓内声明不一致：产物不可作为独立证据"
    )

    # 溯源字段（spec/13第三节runtime_environment的自然延伸）。
    assert manifest["facet"] == ORACLE_MANIFEST_FACET
    assert manifest["case_id"] == "peer_fcl_distance"
    peer = manifest["peer_library"]
    for field in ("distribution", "version", "install_method", "license_spdx", "upstream"):
        assert peer.get(field), f"manifest的peer_library缺{field}——同行库溯源不完整"
    assert peer["bundled_binaries_sha256"], "同行库的二进制哈希必须入manifest（版本号钉不住编译产物）"
    environment = manifest["runtime_environment"]
    for field in ("python_version", "platform", "machine", "numpy_version", "blas"):
        assert environment.get(field), f"manifest的runtime_environment缺{field}"

    # 版本钉死：同行库漂了，路由表与结论都必须重新审，不许自动继承。
    assert peer["version"] == criteria["peer"]["pinned_version"], (
        f"同行库版本漂移：装的是{peer['version']}，criteria.json钉的是"
        f"{criteria['peer']['pinned_version']}。三条FCL算子的语义与精度是逐版本实测的，"
        "换版本必须重跑validation/README.md第四节的语义复核，再改criteria与案例页"
    )

    failed = [check for check in verdict["checks"] if not check["passed"]]
    assert not failed, f"对拍判据未过：{json.dumps(failed, ensure_ascii=False)}"
    assert verdict["overall_passed"] is True
    assert returncode == 0

    j1 = next(item for item in criteria["criteria"] if item["id"] == "J1")
    overall = summary["overall"]
    assert overall["count"] == criteria["sweep"]["expected_sample_count"]
    assert (
        overall["max_abs_deviation_mm"] <= j1["abs_tolerance_mm"]
        or overall["max_rel_deviation"] <= j1["rel_tolerance"]
    )
    assert overall["predicate_disagreements"] == 0
    assert overall["missing_peer_value"] == 0


@pytest.mark.batch
@pytest.mark.parametrize("fault", ["quaternion_order", "radius_sum", "capsule_half_length"])
def test_peer_comparison_must_be_red_under_injected_faults(tmp_path: Path, fault: str) -> None:
    """轴7规则6：门必须红过。

    三条注错都是**仓内自洽测试抓不到**的错——四元数次序错了，我们自己算的
    距离依旧自洽；只有另一个证人能指出来。它们红，才证明这道门在工作。
    """

    interpreter = _require_peer()
    criteria = _criteria()
    returncode, contents = _run_sweep(
        interpreter,
        tmp_path,
        fault,
        per_cell=criteria["sweep"]["fault_per_cell"],
        fault=fault,
    )
    verdict = strict_loads(contents["verdict.json"])
    j1 = next(check for check in verdict["checks"] if check["id"] == "J1")
    assert not j1["passed"], f"注错`{fault}`没能让J1红——这道门抓不到它要抓的错"
    assert j1["observed_max_abs_deviation_mm"] >= (
        criteria["must_be_red"]["min_expected_abs_deviation_mm"]
    )
    assert verdict["overall_passed"] is False
    assert returncode == 1


def _canonical_repo_criteria() -> bytes:
    """仓内criteria.json按对拍脚本同一条规范化路径产出的字节。"""

    from physics_engine.canonical import FTS_PROFILE, canonical_file_bytes

    return canonical_file_bytes(_criteria(), FTS_PROFILE)
