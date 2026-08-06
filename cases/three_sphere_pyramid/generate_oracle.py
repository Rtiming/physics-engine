#!/usr/bin/env python3
"""三球金字塔临界摩擦的金标生成器——**闭式解，独立于被验内核**。

三个等径球（半径``r``、质量``m``）：两个底球贴地并相切，球心``(±r, 0, r)``；
顶球卧在凹槽里，球心``(0, 0, r + √3·r)``。连心线与水平成60°。

**球-球接触无摩擦**（只沿连心线传力），**球-地接触有摩擦**，
**且转动不参与**（球被钉成质点+半径）。**两个前提缺一不可**：
对底球球心取矩得``M_y = −(√3/6)·W·r ≠ 0``，可转动的刚体球在本构型下不是平衡态。
（2026-08-06对抗审核补；此前只写了第一条，且写了一句未经验证的"静定"论断。）

顶球竖直平衡（两个对称接触，力沿连心线，竖直分量``√3/2``）：

    2·F·(√3/2) = W          →   F = W/√3

底球（受顶球反作用``F``、自重``W``、地面法向``N``与摩擦``T``）：

    竖直：N = W + F·(√3/2) = W + W/2 = 3W/2
    水平：T = F·(1/2)       = W/(2√3)

粘住条件``T ≤ μ·N``：

    W/(2√3) ≤ μ·(3W/2)      →   μ ≥ 1/(3√3) = √3/9 ≈ 0.19245008972987526

**``W``两边约掉——临界摩擦系数与质量、重力加速度、罚刚度全部无关，
只由几何决定。** 这是本案例判据强度的来源。

## 罚函数模型下这个比值**不是精确的**（与斜面案例的关键差别）

斜面上法向是固定的，穿透不改变力的分解，所以``N``与``T``精确。
**这里不同**：穿透改变**接触几何本身**（球心距变了，连心线角度跟着变），
于是``T/N``带一个``O(δ/R) = O(1/k)``的偏差。

实测（``r = 10 mm``、``m = 1.5 kg``）：

| ``k`` (N/mm) | ``T/N``相对偏差 | 前档/本档 |
|---|---|---|
| 2e5 | 5.664e-06 | — |
| 2e6 | 5.661e-07 | **10.00** |
| 2e7 | 5.394e-08 | **10.50** |
| 2e8 | 2.370e-08 | 2.28（已入浮点地板） |
| 2e9 | 2.554e-07 | 0.09（地板以下，开始变差） |

**前三档的比值≈10，即一阶收敛**——这条比"偏差很小"强得多：
它证明那个偏差**是模型的柔度而不是实现的错误**。
后两档的退化是相消：``k = 2e9``时穿透约4e-9 mm而心距20 mm，
相对量2e-10，双精度只剩约6位有效数字。**地板的位置是算得出来的，不是玄学。**

生成器只写闭式，**不调`solve_equilibrium`、不调任何接触项**。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/three_sphere_pyramid"
ALGORITHM_VERSION = "1.0.0"

RADIUS_MM = 10.0
MASS_KG = 1.5
GRAVITY_MM_S2 = 9810.0
WEIGHT_N = MASS_KG * GRAVITY_MM_S2 / 1000.0

CRITICAL_FRICTION = 1.0 / (3.0 * math.sqrt(3.0))
SPHERE_CONTACT_FORCE_N = WEIGHT_N / math.sqrt(3.0)
GROUND_NORMAL_N = 1.5 * WEIGHT_N
GROUND_TANGENTIAL_N = WEIGHT_N / (2.0 * math.sqrt(3.0))

#: 收敛阶要扫的刚度（只取一阶区间的三档；后两档在浮点地板下，见模块docstring）。
CONVERGENCE_STIFFNESSES = (2.0e5, 2.0e6, 2.0e7)
#: 力判据用的刚度与相对容差。2e7档实测偏差5.4e-08，取2e-7约3.7倍余量。
FORCE_STIFFNESS = 2.0e7
FORCE_RELATIVE_TOLERANCE = 2.0e-7

#: 判别扫描：``μ/μc``的两侧偏移。1e-5是实测能分开的量级（见案例页第三节）。
FRICTION_RATIOS = (0.99999, 1.00001)


def main() -> int:
    oracles = [
        {
            "id": "oracle:pyramid/critical_friction",
            "inputs": {
                "kind": "three_sphere_pyramid_critical_friction",
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
                "stiffness_n_per_mm": FORCE_STIFFNESS,
                "friction_ratios": list(FRICTION_RATIOS),
            },
            "expected": {
                "critical_friction": CRITICAL_FRICTION,
                "holds_above": True,
                "collapses_below": True,
            },
            "tolerances": {
                "critical_friction": {
                    "abs": 0.0, "rel": FORCE_RELATIVE_TOLERANCE,
                    "reason": "``μc = 1/(3√3)``是几何决定的闭式值，但罚函数柔度让实测的"
                              "``T/N``带``O(1/k)``偏差（穿透改变接触几何本身）。"
                              "``k = 2e7``档实测5.4e-08，取2e-7约3.7倍余量。"
                              "**这个容差是模型的柔度，不是实现的余量**——"
                              "收敛阶那条门专门验这一点",
                },
                "holds_above": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**定性判据零容差**（Chrono `utest_DEM_pyramid.cpp`的形制）："
                              "同一模型只改``μ``，一边必须撑住",
                },
                "collapses_below": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "同上，另一边必须塌。**两侧都断言才叫临界值**",
                },
            },
        },
        {
            "id": "oracle:pyramid/force_decomposition",
            "inputs": {
                "kind": "pyramid_static_decomposition",
                "stiffness_n_per_mm": FORCE_STIFFNESS,
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
            },
            "expected": {
                "sphere_contact_force_n": SPHERE_CONTACT_FORCE_N,
                "ground_normal_n": GROUND_NORMAL_N,
                "ground_tangential_n": GROUND_TANGENTIAL_N,
            },
            "tolerances": {
                "sphere_contact_force_n": {
                    "abs": 0.0, "rel": FORCE_RELATIVE_TOLERANCE,
                    "reason": "同``μc``那条：``O(1/k)``的柔度偏差",
                },
                "ground_normal_n": {
                    "abs": 0.0, "rel": FORCE_RELATIVE_TOLERANCE,
                    "reason": "``N = 3W/2``。**它比另外两个准得多**（实测7e-12量级）——"
                              "竖直平衡不依赖接触角，所以柔度不进这一条。"
                              "容差仍取统一值，理由是判据表的可读性优先于逐条压紧",
                },
                "ground_tangential_n": {
                    "abs": 0.0, "rel": FORCE_RELATIVE_TOLERANCE,
                    "reason": "``T = W/(2√3)``，经接触角进来，故带柔度偏差",
                },
            },
        },
        {
            "id": "oracle:pyramid/compliance_is_first_order",
            "inputs": {
                "kind": "penalty_compliance_convergence_order",
                "stiffnesses_n_per_mm": list(CONVERGENCE_STIFFNESSES),
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
            },
            "expected": {
                "deviation_ratio_low": 8.0,
                "deviation_ratio_high": 12.0,
                "deviations_shrink": True,
            },
            "tolerances": {
                "deviation_ratio_low": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "区间端点是**声明的判据**不是测出来的数：刚度每涨10倍偏差应降约10倍。"
                              "实测10.00与10.50，区间取[8,12]。**不写死为10**——"
                              "一阶是渐近性质，写死会让正确实现在别的构型上红（与"
                              "`harmonic_oscillator`那条「不写死为4」同源）",
                },
                "deviation_ratio_high": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "同上，区间上端",
                },
                "deviations_shrink": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**前置断言**：偏差必须逐档单调减小。"
                              "不减小时比值区间那条门是在拿噪声算阶",
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/three_sphere_pyramid",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/three_sphere_pyramid/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
