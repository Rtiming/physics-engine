#!/usr/bin/env python3
"""斜面滑动阈值的金标生成器——**闭式解，独立于被验内核**。

刚体块静置于倾角``θ``的斜面，库仑摩擦系数``μs``。沿面法向与切向分解重力：

    N = W·cos θ        T = W·sin θ        W = m·g

粘住的条件是切向驱动力不超过摩擦锥：``T ≤ μs·N``，即

    W·sin θ ≤ μs·W·cos θ   ⟺   tan θ ≤ μs   ⟺   θ ≤ θc = arctan(μs)

**阈值与质量、重力加速度、罚刚度全部无关**——``W``两边约掉了。
这一条是本案例判据强度的来源：它是一个**定性翻转**的阈值，
不是一个可以靠调参凑近的数值。

来源：research/05第2.3节C档接触族第8条（斜面滑动阈值与无滑滚球，
判据出处Chrono/Drake）。本案例只做前半条（滑动阈值）；
无滑滚球要转动自由度参与接触，属下一片。

生成器只写闭式解，**不调`solve_equilibrium`、不调任何接触项**。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/incline_slide_threshold"
ALGORITHM_VERSION = "1.0.0"

MASS_KG = 2.0
GRAVITY_MM_S2 = 9810.0
WEIGHT_N = MASS_KG * GRAVITY_MM_S2 / 1000.0
FRICTION_COEFFICIENT = 0.30
NORMAL_STIFFNESS_N_PER_MM = 5.0e4
TANGENTIAL_STIFFNESS_N_PER_MM = 3.0e4

#: 分解判据要扫的倾角（度）。取值跨越浅、临界附近与陡三段。
DECOMPOSITION_ANGLES_DEG = (5.0, 10.0, 16.699244, 25.0, 40.0)

#: 阈值两侧的偏移（度）。1e-7°对应``tan``的相对变化约6.3e-9，
#: 远大于实测的1e-16量级偏差——**判据分得开，不是踩在噪声上**。
THRESHOLD_OFFSETS_DEG = (-1.0e-4, -1.0e-7, 1.0e-7, 1.0e-4)


def main() -> int:
    threshold_rad = math.atan(FRICTION_COEFFICIENT)
    oracles = [
        {
            "id": "oracle:incline/threshold_angle",
            "inputs": {
                "kind": "coulomb_incline_threshold",
                "friction_coefficient": FRICTION_COEFFICIENT,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
                "normal_stiffness_n_per_mm": NORMAL_STIFFNESS_N_PER_MM,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
                "offsets_deg": list(THRESHOLD_OFFSETS_DEG),
            },
            "expected": {
                "threshold_angle_deg": math.degrees(threshold_rad),
                "sticks_below": True,
                "slips_above": True,
            },
            "tolerances": {
                "threshold_angle_deg": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "θc = arctan(μs)是闭式值，本仓只做一次atan与一次弧度转角度；"
                              "1e-12是两次浮点运算的余量，不是给实现留的空间",
                },
                "sticks_below": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**定性判据零容差**：阈值下方必须粘住。"
                              "布尔量没有'差不多'",
                },
                "slips_above": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "同上，反向。**两侧都断言才叫阈值**——"
                              "只断一侧的话，一个'永远粘住'的实现能过一半",
                },
            },
        },
        {
            "id": "oracle:incline/force_decomposition",
            "inputs": {
                "kind": "gravity_decomposition_on_slope",
                "angles_deg": list(DECOMPOSITION_ANGLES_DEG),
                "weight_n": WEIGHT_N,
                "normal_stiffness_n_per_mm": NORMAL_STIFFNESS_N_PER_MM,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
            },
            "expected": {
                "normal_force_n": [
                    WEIGHT_N * math.cos(math.radians(deg)) for deg in DECOMPOSITION_ANGLES_DEG
                ],
                "tangential_force_n": [
                    WEIGHT_N * math.sin(math.radians(deg)) for deg in DECOMPOSITION_ANGLES_DEG
                ],
            },
            "tolerances": {
                "normal_force_n": {
                    "abs": 0.0, "rel": 4.0e-16,
                    "reason": "罚函数模型的**法向力是精确的**：平衡时k·δ = N，k被约掉。"
                              "实测五档偏差≤2.2e-16（1 ulp），故取4e-16≈2 ulp。"
                              "**位置不在判据里**——那一项是O(1/k)的模型穿透，不是误差",
                },
                "tangential_force_n": {
                    "abs": 0.0, "rel": 4.0e-16,
                    "reason": "粘着弹簧同构：平衡时k_t·|Δ| = T，与k_t无关。实测≤3.3e-16",
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/incline_slide_threshold",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/incline_slide_threshold/generate_oracle.py",
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
