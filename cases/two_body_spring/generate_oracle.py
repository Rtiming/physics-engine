#!/usr/bin/env python3
"""两体弹簧的金标生成器——**独立解析路径，不调被验内核**。

三组判据，一组比一组"更验物理"：

1. **解析能量值**：均匀拉伸 `U = ½·(EA/l0)·δ²`；均匀重力 `U = −m·g·y`。
   这两条验的是**能量本身写对没有**——有限差分门验不了它。
2. **解析角频率**：两个自由节点+一根弹簧，约化质量 `μ = m1·m2/(m1+m2)`，
   `ω = sqrt(k/μ)`，其中 `k = EA/l0`。**单位换算在这里露头**：
   `k` 是N/mm、`μ` 是kg，而 `N/kg = m/s²`，状态是mm制，所以 `k/μ` 要乘1000才是 `1/s²`。
3. **动量守恒**：无外力，质心不动。

**本案例的来历值得记一笔**：写它的时候抓到了一个真bug——能量→加速度的接缝
漏了mm与m的换算，加速度小了1000倍。当时有限差分门（梯度对能量、Hessian对梯度）
**全绿**，因为换算因子不在能量里、FD看不见它。抓住它的就是下面第2组判据。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/two_body_spring"
ALGORITHM_VERSION = "1.0.0"

MASS_A_KG, MASS_B_KG = 0.3, 0.5
REST_LENGTH_MM, AXIAL_STIFFNESS_N = 10.0, 45.0
ELONGATION_MM = 0.05
GRAVITY_MM_S2 = -9806.65
#: N/kg = m/s²，而状态是mm制——这个1000是单位换算不是物理系数。
MM_PER_M = 1000.0


def main() -> int:
    k_n_mm = AXIAL_STIFFNESS_N / REST_LENGTH_MM
    reduced_kg = MASS_A_KG * MASS_B_KG / (MASS_A_KG + MASS_B_KG)
    omega_per_s = math.sqrt(k_n_mm / reduced_kg * MM_PER_M)
    oracles = [{
        "id": "oracle:two_body_spring/stretch_energy",
        "inputs": {"kind": "analytic_energy", "term": "axial_stretch",
                   "rest_length_mm": REST_LENGTH_MM,
                   "axial_stiffness_n": AXIAL_STIFFNESS_N,
                   "elongation_mm": ELONGATION_MM},
        "expected": {"energy_nmm": 0.5 * k_n_mm * ELONGATION_MM**2},
        "tolerances": {"energy_nmm": {
            "abs": 0.0, "rel": 2.0e-13,
            "reason": "oracle直接由δ算½·(EA/l0)·δ²；**被验内核走的是另一条路**——"
                      "先求|x_j−x_i|再减l0，而10.05−10.0是相消，"
                      "相对误差被放大约L/δ=200倍（机器eps 2.2e-16×200≈4.4e-14）。"
                      "实测相对偏差2.7e-14，rel=2e-13留约七倍余量。"
                      "**容差不是放宽到能过为止，是把两条路径的数值差异算出来**——"
                      "这个200倍就是相消放大因子，δ取得更小它会更大。"
                      "这一条验的是能量本身，不是它的导数，有限差分门验不了它。",
        }},
    }, {
        "id": "oracle:two_body_spring/gravity_energy",
        "inputs": {"kind": "analytic_energy", "term": "uniform_gravity",
                   "mass_kg": MASS_A_KG, "gravity_mm_s2": GRAVITY_MM_S2,
                   "height_mm": 7.25},
        "expected": {"energy_nmm": -MASS_A_KG * GRAVITY_MM_S2 * 7.25 / MM_PER_M},
        "tolerances": {"energy_nmm": {
            "abs": 0.0, "rel": 1.0e-15,
            "reason": "−m·g·y。**注意除以1000**：m·g是N·mm/s²·kg→需换算到N，"
                      "与加速度接缝上那个乘1000是同一条单位边界的两侧。",
        }},
    }, {
        "id": "oracle:two_body_spring/analytic_angular_frequency",
        "inputs": {"kind": "two_body_oscillation",
                   "mass_a_kg": MASS_A_KG, "mass_b_kg": MASS_B_KG,
                   "rest_length_mm": REST_LENGTH_MM,
                   "axial_stiffness_n": AXIAL_STIFFNESS_N,
                   "initial_elongation_mm": ELONGATION_MM,
                   "integrator": "velocity_verlet", "steps_per_period": 20000},
        "expected": {
            "omega_per_s": omega_per_s,
            "elongation_after_full_period_mm": ELONGATION_MM,
            "elongation_after_half_period_mm": -ELONGATION_MM,
            "centre_of_mass_drift_mm": 0.0,
        },
        "tolerances": {
            "omega_per_s": {"abs": 0.0, "rel": 1.0e-15,
                            "reason": "sqrt(k/μ·1000)，闭式"},
            "elongation_after_full_period_mm": {
                "abs": 1.0e-9, "rel": 0.0,
                "reason": "velocity Verlet每周期20000步，实测偏差1.0e-13mm；"
                          "abs=1e-9留四个量级余量。**这一条是那个1000倍单位bug的捕手**。",
            },
            "elongation_after_half_period_mm": {
                "abs": 1.0e-9, "rel": 0.0,
                "reason": "半周期必须反相。只看整周期分不出「频率对」与「频率差整数倍」。",
            },
            "centre_of_mass_drift_mm": {
                "abs": 1.0e-9, "rel": 0.0,
                "reason": "无外力则质心不动（动量守恒）。它验的是「两端受力等大反向」，"
                          "与频率正确与否正交——实测1.2e-14mm。",
            },
        },
    }]
    document = {
        "facet": "engine_oracle_manifest", "facet_version": "0.1",
        "case_id": "case/two_body_spring", "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/two_body_spring/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles, "arrays": {}, "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
