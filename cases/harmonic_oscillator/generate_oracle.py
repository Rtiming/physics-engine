#!/usr/bin/env python3
"""B2谐振子的金标生成器——**独立解析路径**。

两件事：①收敛阶（velocity Verlet对cos(ωT)的误差随h减半降约4倍）；
②漂移排序（explicit > symplectic > verlet，严格不等号）。

**为什么阶数判据是区间不是"恰为4"**：4是渐近值，粗档还没完全进渐近区
（实测3.9985就是这件事）。把它写死会让一个正确的实现在粗档上红——
这与spec/12第4.3节"比收敛阶不比单点"是同一条纪律。

**为什么漂移判据先断非零**：三个积分器若都返回初值，排序断言会在全零输入上
假通过（MuJoCo `EXPECT_NE`形制，spec/12第6.2节写法2的堵法）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/harmonic_oscillator"
ALGORITHM_VERSION = "1.0.0"

OMEGA_PER_S = 2.0
HORIZON_S = 3.0
ORDER_STEPS = (0.02, 0.01, 0.005)
DRIFT_DT_S = 0.01
DRIFT_STEPS = 2000


def main() -> int:
    oracles = [{
        "id": "oracle:harmonic/analytic_position",
        "inputs": {
            "kind": "undamped_harmonic_oscillator",
            "omega_per_s": OMEGA_PER_S, "initial_position_mm": 1.0,
            "initial_velocity_mm_s": 0.0, "horizon_s": HORIZON_S,
        },
        "expected": {"position_mm": math.cos(OMEGA_PER_S * HORIZON_S)},
        "tolerances": {"position_mm": {
            "abs": 0.0, "rel": 1.0e-15,
            "reason": "解析解cos(ωT)，只受libm与双精度舍入影响",
        }},
    }, {
        "id": "oracle:harmonic/verlet_order_ratio",
        "inputs": {
            "kind": "convergence_order", "integrator": "velocity_verlet",
            "omega_per_s": OMEGA_PER_S, "horizon_s": HORIZON_S,
            "dt_s_ladder": list(ORDER_STEPS),
        },
        "expected": {"ratio_low": 3.9, "ratio_high": 4.1, "formal_order": 2},
        "tolerances": {
            "ratio_low": {"abs": 0.0, "rel": 0.0, "reason": "区间下界，零容差比较"},
            "ratio_high": {"abs": 0.0, "rel": 0.0, "reason": "区间上界，零容差比较"},
            "formal_order": {"abs": 0.0, "rel": 0.0, "reason": "整数，非数值量必须零容差"},
        },
    }, {
        "id": "oracle:harmonic/drift_ordering",
        "inputs": {
            "kind": "energy_drift_ordering", "omega_per_s": OMEGA_PER_S,
            "dt_s": DRIFT_DT_S, "steps": DRIFT_STEPS,
            "initial_position_mm": 1.0, "initial_velocity_mm_s": 0.0,
        },
        "expected": {
            "ordering": ["explicit_euler", "symplectic_euler", "velocity_verlet"],
            "all_nonzero": True,
        },
        "tolerances": {
            "ordering": {"abs": 0.0, "rel": 0.0,
                         "reason": "非数值量（顺序列表）必须零容差——判据是相对优劣，"
                                   "不随机器/编译器漂移（MuJoCo形制）"},
            "all_nonzero": {"abs": 0.0, "rel": 0.0,
                            "reason": "布尔前置断言，防三者全零时排序假通过"},
        },
    }]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/harmonic_oscillator",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/harmonic_oscillator/generate_oracle.py",
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
