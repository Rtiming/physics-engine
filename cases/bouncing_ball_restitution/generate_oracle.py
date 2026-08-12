#!/usr/bin/env python3
"""瞬态罚接触的金标生成器——**闭式，不跑引擎**。

三条闭式全部来自同一个事实：**罚接触期间的运动是简谐半周期**。
无重力、无阻尼时，接触段满足`m·ẍ = −1000·k·x`（x是穿透量，mm），
故角频率`ω = sqrt(1000k/m)`，而：

* 接触时长 `t_c = π/ω` —— 半个周期，与入射速度**无关**；
* 最大穿透 `δ_max = v_in/ω` —— 简谐运动的振幅；
* 恢复系数 `e = 1` —— 保守力场，出射速率等于入射速率。

**那个1000是单位制的必然，不是凑的**：`k`是N/mm、`g`是mm，故力是N；
质量是kg，而`N/kg = m/s²`是米制，而状态是mm制。
详见`energies.EnergyRegistry.acceleration`的docstring——
那里记着一个静默1000倍真的发生过一次，且有限差分门对它完全失明。

## 判据为什么不是"δ_max = N/k"

**因为那是准静态的律，这里是瞬态。**

* 准静态：`δ = N/k`，`O(1/k)`；
* 瞬态冲击：`δ_max = v_in·sqrt(m/k) = v_in/ω`，`O(k^(−1/2))`。

plans/08第零节实测`k = 1e5`时准静态式差**1010倍**。
两条律各管各的域——**瞬态案例的判据照抄准静态那条，刚度提100倍时
判据会松100倍而不是10000倍**，于是一个错的实现能一路绿着走过去。

`tests/cases/test_bouncing_ball_restitution.py`里有一条必红专防这件事。

## 这份金标不是实测

按spec/08规则1，实测数据不作金标。上面三条是闭式解，
本文件**不import积分器也不跑接触**——它只把闭式算出来。
引擎跑出来的数与它对拍，是测试的事。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/bouncing_ball_restitution"
ALGORITHM_VERSION = "1.0.0"

#: 单位：``k``是N/mm、``m``是kg、``v_in``是mm/s、半径是mm。
STIFFNESS_N_PER_MM = 1.0e4
MASS_KG = 1.0e-3
INCIDENT_SPEED_MM_S = 500.0
RADIUS_MM = 0.0

#: 刚度扫描——**这一档存在的唯一目的是让"照抄准静态律"的实现红**。
#: 两条律的标度不同（``O(1/k)``对``O(k^(−1/2))``），
#: 单一刚度上两者可以靠调系数对上，跨刚度对不上。
STIFFNESS_SWEEP_N_PER_MM = (1.0e3, 1.0e4, 1.0e5)

#: 每次接触的步数档。plans/08实测2步时恢复系数1.1433（**大于1**）；
#: 本仓2026-08-12实测20/200/2000步的`|e−1|`是2.5e-4 / 2.5e-7 / <1e-12。
STEPS_PER_CONTACT_SWEEP = (20, 200, 2000)


def omega_rad_per_s(stiffness_n_per_mm: float, mass_kg: float) -> float:
    """``ω = sqrt(1000·k/m)``。那个1000是mm制与米制的接缝，见模块docstring。"""

    return math.sqrt(1000.0 * stiffness_n_per_mm / mass_kg)


def main() -> int:
    omega = omega_rad_per_s(STIFFNESS_N_PER_MM, MASS_KG)

    oracles = [
        {
            "id": "oracle:bounce/contact_duration",
            "inputs": {
                "kind": "contact_duration",
                "stiffness_n_per_mm": STIFFNESS_N_PER_MM,
                "mass_kg": MASS_KG,
                "incident_speed_mm_s": INCIDENT_SPEED_MM_S,
                "radius_mm": RADIUS_MM,
            },
            "expected": {
                "omega_rad_per_s": omega,
                "contact_duration_s": math.pi / omega,
            },
            "tolerances": {
                "omega_rad_per_s": {
                    "abs": 0.0, "rel": 1e-14,
                    "reason": "闭式对闭式，只留浮点噪声；sqrt与除法各一次舍入",
                },
                "contact_duration_s": {
                    "abs": 0.0, "rel": 1e-6,
                    "reason": "引擎逐步推进，接触起止落在步长格点上，"
                              "故分辨误差约1步/接触=1/N。N=200时5e-3——"
                              "**取1e-6是因为实测比值恒为1.000000**："
                              "接触时长与入射速度无关，起止对称，格点误差首阶相消",
                },
            },
        },
        {
            "id": "oracle:bounce/max_penetration",
            "inputs": {
                "kind": "max_penetration",
                "stiffness_n_per_mm": STIFFNESS_N_PER_MM,
                "mass_kg": MASS_KG,
                "incident_speed_mm_s": INCIDENT_SPEED_MM_S,
            },
            "expected": {
                "max_penetration_mm": INCIDENT_SPEED_MM_S / omega,
                "quasistatic_formula_is_wrong_here": True,
            },
            "tolerances": {
                "max_penetration_mm": {
                    "abs": 0.0, "rel": 5e-3,
                    "reason": "N=20时实测比值1.003097（3.1e-3），N=200时1.000031。"
                              "**容差按最粗档定**，留约1.6倍余量；"
                              "细档的收敛由下面那条阶数判据管",
                },
                "quasistatic_formula_is_wrong_here": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔前置：这条金标声明'δ = N/k'在本案例域外。"
                              "零容差——它不是数值量",
                },
            },
        },
        {
            "id": "oracle:bounce/restitution_undamped",
            "inputs": {
                "kind": "restitution",
                "stiffness_n_per_mm": STIFFNESS_N_PER_MM,
                "mass_kg": MASS_KG,
                "incident_speed_mm_s": INCIDENT_SPEED_MM_S,
                "damping": None,
            },
            "expected": {"restitution": 1.0},
            "tolerances": {
                "restitution": {
                    "abs": 3e-4, "rel": 0.0,
                    "reason": "保守力场理论值恰为1。**用abs不用rel**：期望值是1，"
                              "两者数值相同但abs表达的是'离1有多远'这个物理量。"
                              "3e-4按N=20实测（2.5e-4）留约1.2倍余量。"
                              "**注意误差是单侧的：实测e恒>1**——"
                              "积分误差往碰撞里喂能量，不往外抽",
                },
            },
        },
        {
            "id": "oracle:bounce/penetration_scales_as_inverse_sqrt_k",
            "inputs": {
                "kind": "stiffness_scaling",
                "stiffness_sweep_n_per_mm": list(STIFFNESS_SWEEP_N_PER_MM),
                "mass_kg": MASS_KG,
                "incident_speed_mm_s": INCIDENT_SPEED_MM_S,
            },
            "expected": {
                #: 刚度×10 → 瞬态穿透÷sqrt(10)；准静态律会预测÷10。
                "penetration_ratio_per_decade": math.sqrt(10.0),
                "quasistatic_would_predict": 10.0,
            },
            "tolerances": {
                "penetration_ratio_per_decade": {
                    "abs": 0.0, "rel": 1e-2,
                    "reason": "**这条是必红的判据本体**：两个预测差sqrt(10)≈3.16倍，"
                              "远在1%之外。容差取1%是为了让'照抄准静态律'必红，"
                              "同时容得下最粗档的离散误差（3.1e-3）",
                },
                "quasistatic_would_predict": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "记录用，不参与断言——写在这里是为了让读金标的人"
                              "看见错的那条律长什么样",
                },
            },
        },
        {
            "id": "oracle:bounce/restitution_converges_monotonically",
            "inputs": {
                "kind": "steps_per_contact_sweep",
                "steps_per_contact": list(STEPS_PER_CONTACT_SWEEP),
                "stiffness_n_per_mm": STIFFNESS_N_PER_MM,
                "mass_kg": MASS_KG,
                "incident_speed_mm_s": INCIDENT_SPEED_MM_S,
            },
            "expected": {
                "monotonically_decreasing": True,
                "all_errors_positive": True,
            },
            "tolerances": {
                "monotonically_decreasing": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "非数值量（单调性）零容差。判**相对优劣**不判绝对值，"
                              "与`harmonic_oscillator`的漂移排序同形制",
                },
                "all_errors_positive": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "前置断言，防三档全为0时单调性假通过"
                              "（MuJoCo `EXPECT_NE`形制，spec/12第6.2节）。"
                              "**同时它本身是个物理断言**：实测e恒大于1",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/bouncing_ball_restitution",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/bouncing_ball_restitution/generate_oracle.py",
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
