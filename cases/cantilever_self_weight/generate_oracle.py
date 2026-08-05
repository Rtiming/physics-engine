#!/usr/bin/env python3
"""自重悬臂梁的金标生成器——**教科书闭式解，独立于被验内核**。

小挠度Euler-Bernoulli，均布载荷q、固支端在s=0、自由端在s=L：

    δ_tip = q·L⁴ / (8·EI)

这是同行案例复用的一条（research/05第2.3节：PyElastica与Gazzola 2018的
自重悬臂族；WDS也有自己的`test_gravity_cantilever.py`）。
**它验的是"能量本身写对没有"**——WDS把自重项的独立解析基准列为最高优先级，
理由是有限差分门只验雅可比与能量的一致，不验能量本身。

生成器只写闭式解与收敛判据，不调`solve_equilibrium`、不调任何能量项。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/cantilever_self_weight"
ALGORITHM_VERSION = "1.0.0"

LENGTH_MM = 1000.0
BENDING_STIFFNESS_NMM2 = 2.0e7
LOAD_PER_LENGTH_N_MM = 0.5
GRAVITY_MM_S2 = 9806.65
REFINEMENTS = (10, 20, 40, 80)
#: 绝对残差容差。**声明它是绝对的**（spec/12第4.3节要求不许含糊）：
#: 力的量级由总载荷q·L=500N定，1e-7N是它的2e-10——远低于任何物理意义的量级。
#: 实测残差地板随加密上升（n=10时2.4e-10N、n=160时5.6e-8N），因为弯曲刚度
#: 标度EI/h³按h⁻³增长——这正是spec/12第4.3节说的"绝对残差门在细化时会被污染"。
RESIDUAL_TOL_N = 1.0e-7


def main() -> int:
    tip_mm = LOAD_PER_LENGTH_N_MM * LENGTH_MM**4 / (8.0 * BENDING_STIFFNESS_NMM2)
    oracles = [{
        "id": "oracle:cantilever/tip_deflection_closed_form",
        "inputs": {
            "kind": "self_weight_cantilever", "length_mm": LENGTH_MM,
            "bending_stiffness_nmm2": BENDING_STIFFNESS_NMM2,
            "load_per_length_n_mm": LOAD_PER_LENGTH_N_MM,
            "gravity_mm_s2": GRAVITY_MM_S2,
            "refinements": list(REFINEMENTS),
            "residual_tol_n": RESIDUAL_TOL_N,
            "newton_iterations_bound": 3,
        },
        "expected": {
            "tip_deflection_mm": tip_mm,
            "error_ratio_low": 3.9,
            "error_ratio_high": 4.1,
            "all_refinements_converged": True,
            "newton_iterations_within_bound": True,
        },
        "tolerances": {
            "tip_deflection_mm": {"abs": 0.0, "rel": 1.0e-15,
                                  "reason": "qL⁴/(8EI)，闭式，四次幂一次除法"},
            "error_ratio_low": {"abs": 0.0, "rel": 0.0,
                                "reason": "区间下界，零容差比较（非数值判据）"},
            "error_ratio_high": {"abs": 0.0, "rel": 0.0,
                                 "reason": "区间上界。**不写死为4**——4是渐近值；"
                                           "本仓实测四档全为4.000，但把它写死会让"
                                           "任何合理实现在粗档上红（spec/12第4.3节：比阶不比单点）"},
            "all_refinements_converged": {"abs": 0.0, "rel": 0.0,
                                          "reason": "布尔前置断言：不收敛的解不许参与收敛阶比较——"
                                                    "那会拿一个没解出来的数去算阶"},
            "newton_iterations_within_bound": {"abs": 0.0, "rel": 0.0,
                                      "reason": "总能量是位置的二次型（线性弯曲+均匀重力），"
                                                "**牛顿一步即达精确解**；容许3步是给残差地板留的余量。"
                                                "迭代数变多是「能量不再是二次型」的信号，是有效判据不是装饰"},
        },
    }]
    document = {
        "facet": "engine_oracle_manifest", "facet_version": "0.1",
        "case_id": "case/cantilever_self_weight", "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/cantilever_self_weight/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles, "arrays": {}, "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
