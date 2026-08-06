#!/usr/bin/env python3
"""摩擦迟滞回线的金标生成器——**闭式解，独立于被验内核**。

弹性-理想塑性滑块（切向刚度``k_t``串联库仑滑块，限``T_max = μN``）：

    屈服位移        u_y = T_max / k_t
    力的饱和值      |T| ≤ T_max          （**上界，不是拟合值**）
    整循环耗散      W = 4·T_max·(u_max − u_y)

耗散那条的推导：稳态回线是平行四边形。卸载支从``+T_max``弹性降到``−T_max``
用掉``2u_y``，其余段力恒为``∓T_max``。上下支各积一次相减即得。

**路径相关性**（本案例判据强度最高的一条）：两条终点**位置相同**的路径，
若其中一条中途越过屈服面，两者的锚点与切向力**必须不同**。
取``u_final = 2.5·u_y``：

* 路径A ``0 → 3u_y → 2.5u_y``：去程滑移把锚点推到``2u_y``，回到``2.5u_y``时
  ``T = k_t·(2.5−2)u_y = 0.5·T_max``，**粘**；
* 路径B ``0 → 2.5u_y``：单调加载，锚点停在``1.5u_y``，``T = T_max``，**滑**。

**比值恰为1/2。** 这一条验的不是公式而是**形制**——
锚点若能从当前位形算出来，两条路径会给出同一个答案。

生成器只写这些闭式，**不调`solve_equilibrium`、不调任何接触项**。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/friction_hysteresis_loop"
ALGORITHM_VERSION = "1.0.0"

MASS_KG = 2.0
GRAVITY_MM_S2 = 9810.0
WEIGHT_N = MASS_KG * GRAVITY_MM_S2 / 1000.0
FRICTION_COEFFICIENT = 0.30
NORMAL_STIFFNESS_N_PER_MM = 5.0e4
TANGENTIAL_STIFFNESS_N_PER_MM = 3.0e4

CONE_LIMIT_N = FRICTION_COEFFICIENT * WEIGHT_N
YIELD_DISPLACEMENT_MM = CONE_LIMIT_N / TANGENTIAL_STIFFNESS_N_PER_MM

#: 回线幅值（以``u_y``为单位）与每段步数。
#: **步数取5的倍数是有理由的**：回线的拐点落在``u_max ∓ 2u_y``，
#: 对``u_max = 5u_y``而言是全程的0.2与0.8处，5的倍数让拐点**恰好落在步长边界上**。
#: 梯形积分在分段线性上本就精确，拐点对齐时误差因此恰为0——
#: 这不是"算得准"，是**判据的定义域刚好没有跨拐点的panel**。
#: 实测：K=333/337/999（不整除）给出1.35e-05/1.32e-05/1.00e-06的偏差。
LOOP_AMPLITUDES_IN_YIELD = (2.0, 5.0, 20.0)
STEPS_PER_LEG = 500

PATH_FINAL_IN_YIELD = 2.5
PATH_A_PEAK_IN_YIELD = 3.0


def main() -> int:
    oracles = [
        {
            "id": "oracle:friction/cone_saturation",
            "inputs": {
                "kind": "coulomb_slider_saturation",
                "friction_coefficient": FRICTION_COEFFICIENT,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
                "normal_stiffness_n_per_mm": NORMAL_STIFFNESS_N_PER_MM,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
                "amplitude_in_yield": 5.0,
                "steps_per_leg": STEPS_PER_LEG,
            },
            "expected": {
                "cone_limit_n": CONE_LIMIT_N,
                "yield_displacement_mm": YIELD_DISPLACEMENT_MM,
                "normal_force_n": WEIGHT_N,
            },
            "tolerances": {
                "cone_limit_n": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**零容差**：饱和值是return-map投影出来的上界，"
                              "``|T| = μN``是构造保证的等式，不是数值逼近的结果。"
                              "它若不逐位相等，说明投影写错了",
                },
                "yield_displacement_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``u_y = μN/k_t``是一次除法；1e-15是那一次运算的余量",
                },
                "normal_force_n": {
                    "abs": 0.0, "rel": 4.0e-16,
                    "reason": "拖拽全程法向不变（面水平、粘着项扣掉法向分量）。"
                              "**它若随切向位移漂，说明粘着弹簧漏了法向投影**",
                },
            },
        },
        {
            "id": "oracle:friction/loop_dissipation",
            "inputs": {
                "kind": "elastic_perfectly_plastic_loop_area",
                "amplitudes_in_yield": list(LOOP_AMPLITUDES_IN_YIELD),
                "steps_per_leg": STEPS_PER_LEG,
                "friction_coefficient": FRICTION_COEFFICIENT,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
            },
            "expected": {
                "dissipation_n_mm": [
                    4.0 * CONE_LIMIT_N * (ratio * YIELD_DISPLACEMENT_MM - YIELD_DISPLACEMENT_MM)
                    for ratio in LOOP_AMPLITUDES_IN_YIELD
                ],
            },
            "tolerances": {
                "dissipation_n_mm": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "回线分段线性、梯形积分在线性段上精确，且步数取5的倍数让"
                              "拐点恰落在步长边界，故实测偏差为0。**取1e-14而不是0**："
                              "零容差会把「某天有人改了步数」变成一次神秘的红，"
                              "而那本该是一条可读的、关于离散化的警告。"
                              "不对齐时的实测量级是1e-5（见生成器注释）",
                },
            },
        },
        {
            "id": "oracle:friction/path_dependence",
            "inputs": {
                "kind": "same_position_two_paths",
                "final_in_yield": PATH_FINAL_IN_YIELD,
                "path_a_peak_in_yield": PATH_A_PEAK_IN_YIELD,
                "steps_per_leg": 400,
                "friction_coefficient": FRICTION_COEFFICIENT,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
            },
            "expected": {
                "force_ratio_a_over_b": 0.5,
                "anchor_a_mm": (PATH_A_PEAK_IN_YIELD - 1.0) * YIELD_DISPLACEMENT_MM,
                "anchor_b_mm": (PATH_FINAL_IN_YIELD - 1.0) * YIELD_DISPLACEMENT_MM,
                "path_a_sticks": True,
                "path_b_slips": True,
            },
            "tolerances": {
                "force_ratio_a_over_b": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``0.5·T_max / T_max``——分子分母同源，比值是一次除法",
                },
                "anchor_a_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "滑移把锚点推到``(peak − u_y)``，随后粘着不再动它。"
                              "**锚点是被断言的量，不是中间变量**——"
                              "它若对不上，说明历史没有被正确写回状态",
                },
                "anchor_b_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "单调加载下锚点停在``(final − u_y)``",
                },
                "path_a_sticks": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**定性判据零容差**：回退之后必须落回锥内",
                },
                "path_b_slips": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "同上，反向。两条路径的判别相反才叫路径相关",
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/friction_hysteresis_loop",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/friction_hysteresis_loop/generate_oracle.py",
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
