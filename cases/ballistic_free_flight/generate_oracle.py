#!/usr/bin/env python3
"""B1弹道自由飞行的金标生成器——**独立解析路径，不调被验内核**。

常加速度下三个积分器的误差都有闭式：velocity Verlet精确（只剩浮点噪声）；
半隐式Euler误差恰为 +a·T·h/2；显式Euler误差恰为 −a·T·h/2。

推导（一行，可独立核对）：设步数N、Nh=T。
半隐式先更新v再推x，故 x_N = x0 + Σ_{n=1..N}(v0+n·a·h)·h = x0 + v0·T + a·h²·N(N+1)/2，
减去精确解 x0+v0·T+a·T²/2 得 +a·T·h/2；显式用旧v推x，求和从n=0到N−1，
得 a·h²·N(N−1)/2，差为 −a·T·h/2。**同幅反号**——这是本案例最要紧的一条。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/ballistic_free_flight"
ALGORITHM_VERSION = "1.0.0"

#: 重力取标准值（mm/s²），初速与总时长选得让行程落在米级——
#: 量级太小会让"误差恰为a·T·h/2"被浮点噪声淹没。
ACCELERATION_MM_S2 = -9806.65
INITIAL_VELOCITY_MM_S = 1000.0
HORIZON_S = 1.0
STEPS = (1.0e-3, 5.0e-4, 2.5e-4)

_WHY = (
    "误差常数由闭式给出（推导见生成器docstring），被验内核只做加减乘；"
    "行程约4.9e3mm下双精度舍入~1e-12mm，判据取比值而非绝对量，"
    "1±1e-8留六个数量级余量。**判据必须带符号**：显式与半隐式同幅反号，"
    "只比绝对值的写法分不开这两个积分器。"
)


def main() -> int:
    exact = INITIAL_VELOCITY_MM_S * HORIZON_S + ACCELERATION_MM_S2 * HORIZON_S**2 / 2
    oracles = []
    for h in STEPS:
        predicted = ACCELERATION_MM_S2 * HORIZON_S * h / 2
        for name, ratio in (("symplectic_euler", 1.0), ("explicit_euler", -1.0)):
            oracles.append({
                "id": f"oracle:ballistic/{name}_h{h:g}",
                "inputs": {
                    "kind": "constant_acceleration_flight",
                    "integrator": name,
                    "acceleration_mm_s2": ACCELERATION_MM_S2,
                    "initial_position_mm": 0.0,
                    "initial_velocity_mm_s": INITIAL_VELOCITY_MM_S,
                    "horizon_s": HORIZON_S,
                    "dt_s": h,
                },
                "expected": {
                    "exact_position_mm": exact,
                    "error_over_predicted": ratio,
                    "predicted_error_mm": predicted,
                },
                "tolerances": {
                    "exact_position_mm": {"abs": 0.0, "rel": 1.0e-15,
                                          "reason": "解析终点位置，只受双精度舍入影响"},
                    "error_over_predicted": {"abs": 1.0e-8, "rel": 0.0, "reason": _WHY},
                    "predicted_error_mm": {"abs": 0.0, "rel": 1.0e-15,
                                           "reason": "a·T·h/2直接算出，无累加"},
                },
            })
        oracles.append({
            "id": f"oracle:ballistic/velocity_verlet_h{h:g}",
            "inputs": {
                "kind": "constant_acceleration_flight",
                "integrator": "velocity_verlet",
                "acceleration_mm_s2": ACCELERATION_MM_S2,
                "initial_position_mm": 0.0,
                "initial_velocity_mm_s": INITIAL_VELOCITY_MM_S,
                "horizon_s": HORIZON_S,
                "dt_s": h,
            },
            "expected": {"exact_position_mm": exact, "relative_error": 0.0},
            "tolerances": {
                "exact_position_mm": {"abs": 0.0, "rel": 1.0e-15,
                                      "reason": "解析终点位置"},
                "relative_error": {
                    "abs": 1.0e-12, "rel": 0.0,
                    "reason": "velocity Verlet对常加速度**精确**，剩下的只是浮点噪声"
                              "（实测2.4e-14/1.5e-14/2.9e-14）；abs=1e-12留约两个量级余量。",
                },
            },
        })
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/ballistic_free_flight",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/ballistic_free_flight/generate_oracle.py",
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
