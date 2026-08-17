#!/usr/bin/env python3
"""各向异性摩擦椭圆的金标生成器——**闭式解，独立于被验内核**。

本文件只用``math``。**不import任何`physics_engine.contact`的东西**：
金标若调了被验的映射，它验的就只剩"这段代码没变过"。

记``a = μ_∥·N``、``b = μ_⊥·N``为椭圆两个半轴（力的量纲，N），
``m̂``为滑移方向的单位矢量（面内，与纵向轴夹角``ψ``）。

## 一、支撑函数：关联流动下沿``m̂``稳态滑移的耗散/单位距离

最大耗散原理（Hill）说实际切向力在容许集上使``f·m̂``最大，
而凸集上"沿给定方向的最大投影"就是**支撑函数**：

    h(ψ) = √(a²cos²ψ + b²sin²ψ)

它是纯凸分析结论，不含任何离散化，故是第1档解析闭式。

## 二、椭圆半径：径向返回下同一条件的耗散/单位距离

径向返回把力钉在``f ∥ m̂``上，故它的耗散是椭圆沿``m̂``的**半径**：

    ρ(ψ) = 1/√(cos²ψ/a² + sin²ψ/b²)

## 三、两条的比与最高短缺

    ρ(ψ)/h(ψ) ≤ 1，等号只在两个主轴上成立。

令``t = tan²ψ``、``r = a/b``：``(ρ/h)² = r²(1+t)²/((1+r²t)(r²+t))``，
对``t``求导在``t = 1``（**ψ = 45°**）取极值，代回得

    ρ/h |_45° = 2ab/(a² + b²)      短缺 = (a − b)²/(a² + b²)

``μ_∥:μ_⊥ = 5:1``时短缺 = 16/26 = **8/13 = 0.6153846…**。

## 四、径向返回违反外法向的最大角

椭圆上参数点``f = (a cosθ, b sinθ)``处，外法向``∝ (cosθ/a, sinθ/b)``，
而径向返回的滑移增量``∝ f``。两者的叉积/点积给

    tan∠ = ½·sin2θ·(a² − b²)/(ab)   ⟹  sin∠|_max = (a² − b²)/(a² + b²)

（用了``(a²−b²)² + 4a²b² = (a²+b²)²``。）5:1时 = 24/26 = **12/13 = 0.9230769…**，
即**67.38°**。**关联流动在同一条扫描上必须给0。**

## 五、横向力被高估多少

45°处关联流动的力是支撑点``f = (a²m_∥, b²m_⊥)/h``，径向返回的力是``ρ·m̂``。
横向分量之比``= ρh/b²``，在45°处``ρh = ab``，故

    f_⊥^径向 / f_⊥^关联 |_45° = a/b = μ_∥/μ_⊥

**径向返回把横向摩擦力恰好放大``μ_∥:μ_⊥``倍**——这正是"系统性偏置横向落位"的形状。

## 六、主轴上的迟滞回线

沿纵向轴往复时椭圆退化成1维理想塑性（横向分量恒为零），故

    u_y = a/k_t          W = 4·a·(U − u_y)

**步数取500的理由与`friction_hysteresis_loop`同**：回线拐点落在``U ∓ 2u_y``，
对``U = 2/5/20 u_y``分别是整段的0.500/0.200/0.050处，500的倍数让拐点
**恰好落在步长边界**，梯形积分因此精确。实测不对齐时（K=250）偏差4.2e-05。

## 七、混合角回线**没有闭式**，这里也不编一个

混合角上卸载支的力沿``−m̂``走直线离开支撑点，**再屈服点不在**``−f_sup``**上**，
之后力沿椭圆弯着走到另一端——回线不是平行四边形。
所以本案例在混合角上判的是**两条独立记账必须相等**（外功``∮f·du``与
塑性功``Σf·Δu_slip``），以及**稳态圈逐位周期**。这两条都不需要闭式。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/anisotropic_friction_ellipse"
ALGORITHM_VERSION = "1.0.0"

#: 消费方WDS给的比值就是5:1（决策0068第一节）。
MU_ALONG = 0.50
MU_ACROSS = 0.10
NORMAL_FORCE_N = 4.0
TANGENTIAL_STIFFNESS_N_PER_MM = 3.0e4

SEMI_ALONG_N = MU_ALONG * NORMAL_FORCE_N
SEMI_ACROSS_N = MU_ACROSS * NORMAL_FORCE_N
YIELD_DISPLACEMENT_MM = SEMI_ALONG_N / TANGENTIAL_STIFFNESS_N_PER_MM

#: 耗散并排扫的角度（度）。**含45°是必须的**——峰值恰在那里，
#: 少了它这张表就只剩两个主轴上的平凡相等。
SWEEP_ANGLES_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
#: 找峰值用的扫描步长。1°的格点含45°，故峰值角**要么恰是45.0要么是别的格点值**，
#: 没有中间态——这就是它可以零容差的理由。
PEAK_SCAN_STEP_DEG = 1.0

#: 退化对拍的规模：360个方向×4个量级。**它是被断言的量**，
#: 因为"比了0次然后全绿"是本仓反复吃亏的那种空门。
DEGENERATE_ANGLE_COUNT = 360
DEGENERATE_MAGNITUDE_COUNT = 4

LOOP_AMPLITUDES_IN_YIELD = (2.0, 5.0, 20.0)
STEPS_PER_LEG = 500
MIXED_ANGLE_DEG = 45.0
MIXED_STEPS_PER_LEG = 2000
MIXED_AMPLITUDE_IN_YIELD = 20.0
STEADY_CYCLE_COUNT = 4


def support_n(angle_deg: float) -> float:
    """``h(ψ)``——关联流动下的稳态滑移耗散/单位距离。"""

    angle = math.radians(angle_deg)
    return math.sqrt(
        (SEMI_ALONG_N * math.cos(angle)) ** 2 + (SEMI_ACROSS_N * math.sin(angle)) ** 2
    )


def radius_n(angle_deg: float) -> float:
    """``ρ(ψ)``——径向返回下的同一个量。"""

    angle = math.radians(angle_deg)
    return 1.0 / math.sqrt(
        (math.cos(angle) / SEMI_ALONG_N) ** 2 + (math.sin(angle) / SEMI_ACROSS_N) ** 2
    )


def main() -> int:
    peak_shortfall = (SEMI_ALONG_N - SEMI_ACROSS_N) ** 2 / (
        SEMI_ALONG_N**2 + SEMI_ACROSS_N**2
    )
    max_flow_sine = (SEMI_ALONG_N**2 - SEMI_ACROSS_N**2) / (
        SEMI_ALONG_N**2 + SEMI_ACROSS_N**2
    )
    oracles = [
        {
            "id": "oracle:friction/degenerate_to_isotropic",
            "inputs": {
                "kind": "equal_coefficients_reduce_to_circle",
                "mu_along": MU_ALONG,
                "mu_across": MU_ALONG,
                "normal_force_n": NORMAL_FORCE_N,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
                "angle_count": DEGENERATE_ANGLE_COUNT,
                "magnitude_count": DEGENERATE_MAGNITUDE_COUNT,
            },
            "expected": {
                "compared_cases": DEGENERATE_ANGLE_COUNT * DEGENERATE_MAGNITUDE_COUNT,
                "bitwise_identical": True,
                "general_path_max_relative_gap": 0.0,
            },
            "tolerances": {
                "compared_cases": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**比了几次是被断言的量**。零容差，因为"
                              "「比了0次然后全绿」是空门的标准形状，"
                              "而这条门的全部价值在于它真的比过",
                },
                "bitwise_identical": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**定性判据零容差**：``μ_∥ == μ_⊥``时映射把参数原样转交"
                              "`coulomb_return_map`，故相同是**构造出来的**不是算出来的。"
                              "判的是IEEE-754位型（`float.hex()`），不是``==``——"
                              "``-0.0 == 0.0``为真而位型不同",
                },
                "general_path_max_relative_gap": {
                    "abs": 4.0e-16,
                    "rel": 0.0,
                    "reason": "**关掉转交、强走通用椭圆路径**时与圆的闭式的最大相对逐分量差。"
                              "这一条不许零容差：通用路径解的是标量方程，"
                              "退化时只能到舍入。**没有它，上一条的「逐位」就只是**"
                              "**转交本身的同义反复**——转交可以掩盖通用路径写错",
                },
            },
        },
        {
            "id": "oracle:friction/maximum_dissipation",
            "inputs": {
                "kind": "associated_flow_outward_normal",
                "mu_along": MU_ALONG,
                "mu_across": MU_ACROSS,
                "normal_force_n": NORMAL_FORCE_N,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
                "ellipse_parameter_step_deg": PEAK_SCAN_STEP_DEG,
                "trial_overshoot_factors": [1.1, 3.0, 100.0],
                "near_yield_overshoot_factor": 1.001,
            },
            "expected": {
                "max_flow_normal_sine": 0.0,
                "near_yield_flow_normal_sine": 0.0,
                "max_support_function_relative_gap": 0.0,
                "radial_return_max_flow_normal_sine": max_flow_sine,
            },
            "tolerances": {
                "max_flow_normal_sine": {
                    "abs": 1.0e-14,
                    "rel": 0.0,
                    "reason": "最大耗散原理的KKT条件：滑移增量必须沿屈服面外法向，"
                              "故正弦的真值是**0**。实测最坏4.9e-15（λ ≥ 1.1）",
                },
                "near_yield_flow_normal_sine": {
                    "abs": 1.0e-12,
                    "rel": 0.0,
                    "reason": "**同一条判据，刚屈服那一档单独列出来**，因为它的容差"
                              "不是求解精度而是**消去误差**：滑移量``(f_trial − f)/k_t``"
                              "在``λ → 1``时是两个几乎相等的数相减，方向误差按"
                              "``eps/(λ−1)``放大。实测λ=1.0001/1.001/1.01/1.1给"
                              "3.59e-12/4.19e-13/3.31e-14/4.86e-15——**四档各差一个数量级，"
                              "与该定律逐档吻合**。λ=1.001时取1e-12（实测的2.4倍）。"
                              "把它混进上一条会让上一条松100倍而看不出为什么",
                },
                "max_support_function_relative_gap": {
                    "abs": 1.0e-14,
                    "rel": 0.0,
                    "reason": "同一条原理的**另一种写法**：``f·m̂``必须等于支撑函数``h(m̂)``。"
                              "它与上一条各自独立——上一条判方向，这条判大小。"
                              "**它对方向误差是二阶不敏感的**（``h``的梯度恰是``f``，"
                              "故两边的一阶变化相消），所以实测在全部λ上平在1.5—1.9e-15，"
                              "不随``λ → 1``放大。**两条一起才把方向与大小都钉住**——"
                              "只留这一条会让67°的方向错也通过",
                },
                "radial_return_max_flow_normal_sine": {
                    "abs": 0.0,
                    "rel": 1.0e-14,
                    "reason": "**这是对照组、不是被验对象**：径向返回（= 度规变换后的径向返回）"
                              "在同一条扫描上的最大违反角，闭式``(a²−b²)/(a²+b²)``。"
                              "它必须**显著非零**，否则「两条映射不同」这句话本身没被验过",
                },
            },
        },
        {
            "id": "oracle:friction/mixed_angle_dissipation",
            "inputs": {
                "kind": "steady_slide_dissipation_per_unit_distance",
                "mu_along": MU_ALONG,
                "mu_across": MU_ACROSS,
                "normal_force_n": NORMAL_FORCE_N,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
                "angles_deg": list(SWEEP_ANGLES_DEG),
                "peak_scan_step_deg": PEAK_SCAN_STEP_DEG,
            },
            "expected": {
                "associative_dissipation_n": [support_n(a) for a in SWEEP_ANGLES_DEG],
                "radial_dissipation_n": [radius_n(a) for a in SWEEP_ANGLES_DEG],
                "peak_shortfall": peak_shortfall,
                "peak_shortfall_angle_deg": MIXED_ANGLE_DEG,
                "transverse_force_overstatement_at_peak": MU_ALONG / MU_ACROSS,
            },
            "tolerances": {
                "associative_dissipation_n": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "支撑函数是闭式；引擎侧是位移控制稳态滑移的实测力投影，"
                              "留1e-12给稳态迭代的收敛余量（实测1e-15量级）",
                },
                "radial_dissipation_n": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "同上，对照映射一侧",
                },
                "peak_shortfall": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "``(a−b)²/(a²+b²) = 8/13``。**这一条是本案例的中心数字**："
                              "WDS报的「混合角耗散最高短缺60%」在本仓量到61.538%",
                },
                "peak_shortfall_angle_deg": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差**：峰值角是从1°格点里argmax选出来的，"
                              "格点含45°，故它要么恰是45.0要么是另一个格点值——**没有中间态**。"
                              "峰值跑到别的角上就说明椭圆的朝向接错了",
                },
                "transverse_force_overstatement_at_peak": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "45°处径向返回的横向力比关联流动大``a/b = μ_∥/μ_⊥``倍。"
                              "**「系统性偏置横向落位」这句话的定量形状就是它**——"
                              "耗散短缺只说少了多少功，这条说方向错到哪去了",
                },
            },
        },
        {
            "id": "oracle:friction/isotropic_substitute_shortfall",
            "inputs": {
                "kind": "single_scalar_mu_stands_in_for_the_ellipse",
                "substitute_mu": MU_ALONG,
                "mu_along": MU_ALONG,
                "mu_across": MU_ACROSS,
                "normal_force_n": NORMAL_FORCE_N,
                "angles_deg": list(SWEEP_ANGLES_DEG),
            },
            "expected": {
                "isotropic_dissipation_n": SEMI_ALONG_N,
                "shortfall_vs_anisotropic": [
                    1.0 - support_n(a) / SEMI_ALONG_N for a in SWEEP_ANGLES_DEG
                ],
                "peak_shortfall": 1.0 - MU_ACROSS / MU_ALONG,
                "peak_shortfall_angle_deg": 90.0,
            },
            "tolerances": {
                "isotropic_dissipation_n": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "各向同性映射的稳态耗散与方向无关，恒为``μ·N``",
                },
                "shortfall_vs_anisotropic": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "**这是另一个口径，不要与`mixed_angle_dissipation`的短缺混起来**："
                              "这一条比的是「今天的引擎拿单个``μ_∥``顶上去」，"
                              "那一条比的是「椭圆上径向缩 vs 最近点」。"
                              "两个数不同、峰值位置也不同（这条在主轴、那条在45°）",
                },
                "peak_shortfall": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": "``1 − μ_⊥/μ_∥ = 0.8``，在纯横向滑移上取到",
                },
                "peak_shortfall_angle_deg": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**峰值在主轴上而不在混合角上**——这正是WDS那句"
                              "「混合角上短缺最高」指的不是这个口径的证据",
                },
            },
        },
        {
            "id": "oracle:friction/principal_axis_loop",
            "inputs": {
                "kind": "elastic_perfectly_plastic_loop_area_on_the_long_axis",
                "mu_along": MU_ALONG,
                "mu_across": MU_ACROSS,
                "normal_force_n": NORMAL_FORCE_N,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
                "amplitudes_in_yield": list(LOOP_AMPLITUDES_IN_YIELD),
                "steps_per_leg": STEPS_PER_LEG,
                "steady_cycle_index": 3,
            },
            "expected": {
                "yield_displacement_mm": YIELD_DISPLACEMENT_MM,
                "dissipation_n_mm": [
                    4.0
                    * SEMI_ALONG_N
                    * (ratio * YIELD_DISPLACEMENT_MM - YIELD_DISPLACEMENT_MM)
                    for ratio in LOOP_AMPLITUDES_IN_YIELD
                ],
                "max_across_force_n": 0.0,
            },
            "tolerances": {
                "yield_displacement_mm": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "``u_y = μ_∥N/k_t``是一次除法",
                },
                "dissipation_n_mm": {
                    "abs": 0.0,
                    "rel": 1.0e-13,
                    "reason": "主轴上椭圆退化成1维理想塑性，回线是平行四边形、"
                              "梯形积分在线性段上精确、拐点对齐步长边界，**离散误差因此为0**；"
                              "剩下的是1500项梯形累加的浮点求和误差，"
                              "实测2.0e-14（K=500）、2.2e-14（K=1000）、2.1e-15（K=2000）"
                              "——**不随K单调，正是求和舍入而不是离散化的指纹**。"
                              "取1e-13是实测的5倍余量。"
                              "拐点不对齐时（K=250）实测4.2e-05，比这高九个数量级",
                },
                "max_across_force_n": {
                    "abs": 1.0e-15,
                    "rel": 0.0,
                    "reason": "沿主轴加载全程横向力**实测恰为0.0**（试探力在``e_⊥``上的"
                              "投影是精确零，之后每一步都乘0）。"
                              "**它非零就说明面内轴接错了**——而那正是"
                              "`PenaltyAnnulusLimit`把朝向藏进坐标符号时"
                              "单元门抓不到的那一类错。留1e-15是为了让"
                              "「有人把装置转了个角度」表现为可读偏差而不是神秘的红",
                },
            },
        },
        {
            "id": "oracle:friction/mixed_angle_closed_loop",
            "inputs": {
                "kind": "steady_cycle_closes_and_two_accountings_agree",
                "mu_along": MU_ALONG,
                "mu_across": MU_ACROSS,
                "normal_force_n": NORMAL_FORCE_N,
                "tangential_stiffness_n_per_mm": TANGENTIAL_STIFFNESS_N_PER_MM,
                "angle_deg": MIXED_ANGLE_DEG,
                "amplitude_in_yield": MIXED_AMPLITUDE_IN_YIELD,
                "steps_per_leg": MIXED_STEPS_PER_LEG,
                "cycles": STEADY_CYCLE_COUNT,
            },
            "expected": {
                "steady_anchor_drift_mm": 0.0,
                "steady_force_drift_n": 0.0,
                "cycle_to_cycle_relative_gap": 0.0,
                "energy_balance_relative_gap": 0.0,
                "first_cycle_is_different": True,
            },
            "tolerances": {
                "steady_anchor_drift_mm": {
                    "abs": 1.0e-15,
                    "rel": 0.0,
                    "reason": "**回线闭合**：稳态圈走完一整圈，锚点必须回到原处。"
                              "实测恰为0（映射是确定的，状态一旦重复就永远重复）；"
                              "留1e-15是让「有人改了步数」表现为可读偏差而不是神秘的红",
                },
                "steady_force_drift_n": {
                    "abs": 1.0e-15,
                    "rel": 0.0,
                    "reason": "同上，力那一半。锚点回原处而力不回，说明返回映射不是"
                              "锚点的函数——那会让历史记账整个失效",
                },
                "cycle_to_cycle_relative_gap": {
                    "abs": 1.0e-15,
                    "rel": 0.0,
                    "reason": "第3圈与第4圈的耗散必须相同。**这条顺手补上了"
                              "`friction_hysteresis_loop`第六条已知失效**"
                              "（那里明写「多圈的稳态性没验」）",
                },
                "energy_balance_relative_gap": {
                    "abs": 3.0e-6,
                    "rel": 0.0,
                    "reason": "外功``∮f·du``与塑性功``Σf·Δu_slip``是**两条独立记账**，"
                              "闭合回线上必须相等。**混合角上回线不是平行四边形**"
                              "（卸载支的再屈服点不在``−f_sup``上，之后力沿椭圆弯着走），"
                              "故梯形积分留O(h²)残余：实测2000步给1.11e-06，"
                              "细化比3.98/3.99/3.99/4.00——**二阶是判出来的**。"
                              "3e-6 = 实测值的2.7倍余量",
                },
                "first_cycle_is_different": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**定性判据零容差、而且是反向的**：第1圈必须**不等于**稳态圈。"
                              "它若也相等，说明锚点从零出发那一段瞬态被抹平了——"
                              "那意味着历史没在起作用，整个案例就是空的",
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/anisotropic_friction_ellipse",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/anisotropic_friction_ellipse/generate_oracle.py",
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
