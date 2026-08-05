#!/usr/bin/env python3
"""刚体自由飞行的金标生成器——**独立解析路径，不调被验内核**。

本脚本一次`import physics_engine.rigidbody`都没有：惯量走教科书闭式（不走
`geometry`）、进动率走Kane的轴对称闭式、增长率走中间轴定理的线性化闭式、
常力矩走`ω = τ·t/I`。它们与被验代码**不共享任何一行**。

四条闭式与出处：

1. **无力矩轴对称刚体的体系进动率** `λ = ω3·(Ia − It)/It`。
   出处：Drake `multibody/benchmarks/free_body/free_body.h`（Kane & Levinson,
   *Dynamics: Theory and Applications*, 1983, §1.13/§3.1），经research/05第2.2节
   收录为本仓可复用的B档判据。**核对过形式**：Euler方程在`I1 = I2 = It`下给
   `ω̇1 = −λ·ω2`、`ω̇2 = +λ·ω1`、`ω̇3 = 0`，即`(ω1, ω2)`在体系里绕对称轴以
   角速率`+λ`匀速转——与research/05抄下来的`λ = ω3·(Ia−It)/It`逐字一致。
   **λ带符号**：扁体（`Ia > It`）为正、长体（`Ia < It`）为负，本案例两个都跑。

2. **同一运动在惯性系里的进动率** `ψ̇ = |L|/It`（对称轴绕**固定的**角动量
   矢量进动）。它与第1条是同一个运动的两种看法，但**只有它用到姿态四元数**——
   第1条只用到`ω`。四元数乘法次序写反时第1条照样绿、第2条当场红（第六节必红3）。

3. **中间轴定理的线性化增长率**
   `σ = Ω·sqrt((I3−I2)(I2−I1)/(I1·I3))`（`I1 < I2 < I3`，绕中间轴`I2`自转）。
   推导：绕轴2以`Ω`自转、加小扰动`δω1, δω3`，Euler方程线性化得
   `I1·δω̇1 = (I2−I3)·Ω·δω3`、`I3·δω̇3 = (I1−I2)·Ω·δω1`，
   消元得`δω̈1 = [(I3−I2)(I2−I1)/(I1·I3)]·Ω²·δω1`，右端系数为**正**故指数增长。
   绕最小轴与最大轴时同一推导给出**负**系数，即有界振荡——这就是
   Dzhanibekov现象的定量形式（research/05第2.2节：MuJoCo/Bullet/PhysX/Newton
   四家独立实现的那条）。

4. **稳定轴的振幅上界**（同一线性化的另一半）。绕最小轴自转、只扰动一个横向分量时，
   扰动矢量模的最大值与初值之比是闭式 `max(1, sqrt(I2(I2−I1)/(I3(I3−I1))))`；
   绕最大轴时是 `max(1, sqrt(I1(I3−I1)/(I2(I3−I2))))`。
   **它是一个数，不是"有界就行"**——把定性判据钉成闭式常数，这是本案例
   相对同行（MuJoCo的`dzhanibekov.xml`只看现象）加的一道。

**为什么阶数判据是区间不是"恰为16"**：16是RK4漂移随步长减半的渐近比值，
粗档没完全进渐近区（实测15.68—16.12）。写死会让正确实现在粗档上红——
与spec/12第4.3节"比收敛阶不比单点"同一条纪律，也与`harmonic_oscillator`
的`[3.9, 4.1]`同源。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/rigid_body_free_flight"
ALGORITHM_VERSION = "1.0.0"

#: N·mm ↔ kg·mm²/s²。生成器与被验内核各写各的——`rigidbody`从`energies`导入
#: `MM_PER_M`，这里是独立的一份字面量，两边对不上就是有一边错了。
NMM_PER_KG_MM2_PER_S2 = 1000.0

MASS_KG = 1.0
#: 非对称体：半边长(30, 20, 10)mm的实心盒，主惯量严格三档不同（I1 < I2 < I3）。
BOX_HALF_EXTENTS_MM = (30.0, 20.0, 10.0)
#: 扁圆柱（`Ia > It`，进动率为**正**）与长圆柱（`Ia < It`，为**负**）。
DISC_RADIUS_MM, DISC_HALF_WIDTH_MM = 40.0, 5.0
ROD_RADIUS_MM, ROD_HALF_WIDTH_MM = 5.0, 40.0

CONSERVATION_OMEGA_RAD_PER_S = (1.0, 2.0, 3.0)
CONSERVATION_HORIZON_S = 2.0
CONSERVATION_DT_LADDER_S = (8.0e-3, 4.0e-3, 2.0e-3)
DRIFT_ORDERING_DT_S = 5.0e-4

PRECESSION_OMEGA_RAD_PER_S = (0.6, 0.0, 4.0)
PRECESSION_HORIZON_S = 4.0
PRECESSION_DT_S = 2.0e-3

SPIN_RAD_PER_S = 5.0
LINEAR_PERTURBATION_RAD_PER_S = 1.0e-6
LINEAR_HORIZON_S = 3.0
LINEAR_WINDOW_S = (1.5, 3.0)
LINEAR_DT_S = 1.0e-3

FLIP_PERTURBATION_RAD_PER_S = 1.0e-2
FLIP_HORIZON_S = 12.0
FLIP_DT_S = 2.0e-3

TORQUE_NMM = 2.5
TORQUE_HORIZON_S = 1.0
TORQUE_DT_S = 1.0e-3

GRAVITY_MM_PER_S2 = -9806.65
LAUNCH_SPEED_MM_PER_S = 1000.0
FLIGHT_HORIZON_S = 1.0
FLIGHT_DT_S = 1.0e-3


def box_inertia(half_extents, mass):
    """实心长方体绕质心的主惯量（教科书闭式，**不走`geometry`**）。"""

    a, b, c = half_extents
    return [
        mass * (b * b + c * c) / 3.0,
        mass * (c * c + a * a) / 3.0,
        mass * (a * a + b * b) / 3.0,
    ]


def cylinder_inertia(radius, half_width, mass):
    """实心圆柱：横向`m(3r² + L²)/12`（`L = 2·half_width`）、轴向`m·r²/2`。"""

    transverse = mass * (3.0 * radius * radius + 4.0 * half_width * half_width) / 12.0
    return [transverse, transverse, mass * radius * radius / 2.0]


BOX = box_inertia(BOX_HALF_EXTENTS_MM, MASS_KG)
DISC = cylinder_inertia(DISC_RADIUS_MM, DISC_HALF_WIDTH_MM, MASS_KG)
ROD = cylinder_inertia(ROD_RADIUS_MM, ROD_HALF_WIDTH_MM, MASS_KG)


def precession_rates(inertia, omega):
    """（体系进动率λ，惯性系进动率ψ̇）。两者都带符号。"""

    transverse, axial = inertia[0], inertia[2]
    body = omega[2] * (axial - transverse) / transverse
    momentum = math.sqrt(
        sum(
            (inertia[axis] * omega[axis]) ** 2 for axis in range(3)
        )
    )
    return body, momentum / transverse


def _sorted_moments(inertia):
    return sorted(inertia)


def growth_rate(inertia, spin):
    low, mid, high = _sorted_moments(inertia)
    return spin * math.sqrt((high - mid) * (mid - low) / (low * high))


def amplification_bound_about_min_axis(inertia):
    low, mid, high = _sorted_moments(inertia)
    return max(1.0, math.sqrt(mid * (mid - low) / (high * (high - low))))


def amplification_bound_about_max_axis(inertia):
    low, mid, high = _sorted_moments(inertia)
    return max(1.0, math.sqrt(low * (high - low) / (mid * (high - mid))))


_INERTIA_WHY = (
    "教科书闭式（盒`m(b²+c²)/3`、圆柱`m(3r²+L²)/12`与`m·r²/2`）对"
    "`geometry.mass_properties`——两条独立路径，后者走的是圆角盒的Steiner展开"
    "在`r_f → 0`的退化与圆柱闭式。只受双精度舍入影响，rel 1e-15。"
    "**这一行是本案例唯一验惯量绝对量级的地方**：其余判据在无力矩下"
    "对`I → c·I`全盲（第四节第1条）。"
)

_PRECESSION_WHY = (
    "闭式进动率（Kane 1983，经Drake `free_body`）对相位展开测得的平均角速率。"
    "实测rel：扁体2.886e-11（体系）/1.537e-11（惯性系）、"
    "长体3.104e-11/4.843e-11。rel 1e-8留约200倍余量——余量给的是`atan2`跨平台的"
    "末位差与相位展开的累加，**不是给物理错留的**：一条真的错的进动率差100%不差1e-8。"
    "**判据必须带符号**：扁体为正、长体为负，写成绝对值就分不开"
    "`ω × (I·ω)`与`(I·ω) × ω`（第六节必红2）。"
)

_GROWTH_WHY = (
    "线性化增长率闭式对窗口[1.5, 3.0]s内的对数斜率。实测偏差4.51e-05且"
    "**换步长到第13位都不变**（dt=1e-3与5e-4给2.401813966348615/…559），"
    "因此这4.5e-5是有限振幅的物理修正而不是积分误差——扰动到窗口末端已长到"
    "初值的约2.6e2倍。rel 1e-3给它约22倍余量；而漏掉陀螺项给0、"
    "绕稳定轴给负数，都在1e-3之外好几个数量级。"
)

_AMPLIFICATION_WHY = (
    "同一线性化的振幅上界闭式。实测rel 1.596e-07（最小轴）与9.315e-08（最大轴）。"
    "**这个残差不是积分误差，是取max的采样误差**：峰值落在两个采样点之间，"
    "误差按`(ν·dt)²`走（dt减半到5e-4时降到8.5e-9/3.8e-9）。"
    "rel 1e-5因此留约60倍余量，且不削弱判据——中间轴同一口径下放大554倍，"
    "与1.15差两个半数量级。"
)

_BAND_WHY = (
    "16是RK4漂移随步长减半的**渐近**比值，粗档没完全进渐近区（实测15.68—16.12），"
    "写死为16会让正确实现在粗档上红（与`harmonic_oscillator`的[3.9,4.1]同源）。"
    "区间[15.0, 17.0]不算松：一阶实现给2、二阶给4，离下界还差三倍以上。"
)


def main() -> int:
    # 盒是三档不同的非对称体，进动闭式对它**不成立**，所以它不参与这两条。
    disc_body, disc_inertial = precession_rates(DISC, PRECESSION_OMEGA_RAD_PER_S)
    rod_body, rod_inertial = precession_rates(ROD, PRECESSION_OMEGA_RAD_PER_S)
    sigma = growth_rate(BOX, SPIN_RAD_PER_S)

    oracles = [{
        "id": "oracle:rigidbody/inertia_from_geometry",
        "inputs": {
            "kind": "principal_moments_from_analytic_primitives",
            "mass_kg": MASS_KG,
            "box_half_extents_mm": list(BOX_HALF_EXTENTS_MM),
            "disc_radius_mm": DISC_RADIUS_MM, "disc_half_width_mm": DISC_HALF_WIDTH_MM,
            "rod_radius_mm": ROD_RADIUS_MM, "rod_half_width_mm": ROD_HALF_WIDTH_MM,
        },
        "expected": {
            "box_diagonal_kg_mm2": BOX,
            "disc_diagonal_kg_mm2": DISC,
            "rod_diagonal_kg_mm2": ROD,
        },
        "tolerances": {
            name: {"abs": 0.0, "rel": 1.0e-15, "reason": _INERTIA_WHY}
            for name in ("box_diagonal_kg_mm2", "disc_diagonal_kg_mm2", "rod_diagonal_kg_mm2")
        },
    }, {
        "id": "oracle:rigidbody/conservation_order_rk4",
        "inputs": {
            "kind": "torque_free_conservation_order",
            "integrator": "rk4_rigid_body",
            "body": "box",
            "omega0_rad_per_s": list(CONSERVATION_OMEGA_RAD_PER_S),
            "horizon_s": CONSERVATION_HORIZON_S,
            "dt_s_ladder": list(CONSERVATION_DT_LADDER_S),
        },
        "expected": {
            "ratio_low": 15.0,
            "ratio_high": 17.0,
            "formal_order": 4,
            "momentum_ratios_within_band": True,
            "energy_ratios_within_band": True,
            "all_drifts_nonzero": True,
            "body_frame_momentum_variation_floor": 0.5,
            "body_frame_momentum_varies": True,
        },
        "tolerances": {
            "ratio_low": {"abs": 0.0, "rel": 0.0, "reason": _BAND_WHY},
            "ratio_high": {"abs": 0.0, "rel": 0.0, "reason": _BAND_WHY},
            "formal_order": {"abs": 0.0, "rel": 0.0,
                             "reason": "整数，非数值量必须零容差"},
            "momentum_ratios_within_band": {
                "abs": 0.0, "rel": 0.0,
                "reason": "布尔判据：**惯性系**角动量矢量的漂移比落在区间内。"
                          "守恒量必须写在惯性系里——体系里的`I·ω`本来就不守恒",
            },
            "energy_ratios_within_band": {
                "abs": 0.0, "rel": 0.0,
                "reason": "布尔判据：转动动能的漂移比落在同一区间内",
            },
            "all_drifts_nonzero": {
                "abs": 0.0, "rel": 0.0,
                "reason": "前置断言（spec/12第6.2节写法1的堵法）：漂移全为零时"
                          "任何比值判据都无从谈起，而`0/0`会以各种方式假通过",
            },
            "body_frame_momentum_variation_floor": {
                "abs": 0.0, "rel": 0.0,
                "reason": "地板值，零容差比较",
            },
            "body_frame_momentum_varies": {
                "abs": 0.0, "rel": 0.0,
                "reason": "**反向**前置断言：体系角动量`I·ω`必须变化超过`|L|`的一半"
                          "（实测0.9847）。没有它，惯性系守恒判据可能只是在陈述"
                          "一个恒等式——ω不动的实现照样全绿。这是本案例最要紧的一行",
            },
        },
    }, {
        "id": "oracle:rigidbody/drift_ordering",
        "inputs": {
            "kind": "conservation_drift_ordering",
            "body": "box",
            "omega0_rad_per_s": list(CONSERVATION_OMEGA_RAD_PER_S),
            "horizon_s": CONSERVATION_HORIZON_S,
            "dt_s": DRIFT_ORDERING_DT_S,
        },
        "expected": {
            "ordering": ["explicit_euler_rigid_body", "rk4_rigid_body"],
            "all_nonzero": True,
        },
        "tolerances": {
            "ordering": {"abs": 0.0, "rel": 0.0,
                         "reason": "非数值量（顺序列表）必须零容差。排序判据不受实现常数"
                                   "影响，换机器换编译器都成立（spec/12第6.2节写法2）"},
            "all_nonzero": {"abs": 0.0, "rel": 0.0,
                            "reason": "前置断言，防两者全零时`0 > 0`被写成`>=`而假通过"},
        },
    }, {
        "id": "oracle:rigidbody/axisymmetric_precession_disc",
        "inputs": {
            "kind": "torque_free_axisymmetric_precession",
            "body": "disc",
            "radius_mm": DISC_RADIUS_MM, "half_width_mm": DISC_HALF_WIDTH_MM,
            "omega0_rad_per_s": list(PRECESSION_OMEGA_RAD_PER_S),
            "horizon_s": PRECESSION_HORIZON_S, "dt_s": PRECESSION_DT_S,
        },
        "expected": {
            "body_precession_rate_per_s": disc_body,
            "inertial_precession_rate_per_s": disc_inertial,
        },
        "tolerances": {
            "body_precession_rate_per_s": {"abs": 0.0, "rel": 1.0e-8,
                                           "reason": _PRECESSION_WHY},
            "inertial_precession_rate_per_s": {"abs": 0.0, "rel": 1.0e-8,
                                               "reason": _PRECESSION_WHY},
        },
    }, {
        "id": "oracle:rigidbody/axisymmetric_precession_rod",
        "inputs": {
            "kind": "torque_free_axisymmetric_precession",
            "body": "rod",
            "radius_mm": ROD_RADIUS_MM, "half_width_mm": ROD_HALF_WIDTH_MM,
            "omega0_rad_per_s": list(PRECESSION_OMEGA_RAD_PER_S),
            "horizon_s": PRECESSION_HORIZON_S, "dt_s": PRECESSION_DT_S,
        },
        "expected": {
            "body_precession_rate_per_s": rod_body,
            "inertial_precession_rate_per_s": rod_inertial,
            "body_rate_is_negative": True,
        },
        "tolerances": {
            "body_precession_rate_per_s": {"abs": 0.0, "rel": 1.0e-8,
                                           "reason": _PRECESSION_WHY},
            "inertial_precession_rate_per_s": {"abs": 0.0, "rel": 1.0e-8,
                                               "reason": _PRECESSION_WHY},
            "body_rate_is_negative": {
                "abs": 0.0, "rel": 0.0,
                "reason": "长圆柱`Ia < It`故进动率为负，扁圆柱为正——"
                          "两个符号都跑，写成`abs()`的判据当场分不开叉乘次序",
            },
        },
    }, {
        "id": "oracle:rigidbody/dzhanibekov_growth_rate",
        "inputs": {
            "kind": "intermediate_axis_growth_rate",
            "body": "box", "spin_axis": 1, "spin_rad_per_s": SPIN_RAD_PER_S,
            "perturbation_rad_per_s": LINEAR_PERTURBATION_RAD_PER_S,
            "horizon_s": LINEAR_HORIZON_S, "dt_s": LINEAR_DT_S,
            "window_s": list(LINEAR_WINDOW_S),
        },
        "expected": {"growth_rate_per_s": sigma, "growth_rate_is_positive": True},
        "tolerances": {
            "growth_rate_per_s": {"abs": 0.0, "rel": 1.0e-3, "reason": _GROWTH_WHY},
            "growth_rate_is_positive": {
                "abs": 0.0, "rel": 0.0,
                "reason": "**符号就是中间轴定理本身**：正=不稳定。"
                          "陀螺项漏掉时增长率恰为0，这条与上一条一起把它挡下",
            },
        },
    }, {
        "id": "oracle:rigidbody/dzhanibekov_stable_axes",
        "inputs": {
            "kind": "stable_axis_amplification",
            "body": "box", "spin_rad_per_s": SPIN_RAD_PER_S,
            "perturbation_rad_per_s": LINEAR_PERTURBATION_RAD_PER_S,
            "horizon_s": LINEAR_HORIZON_S, "dt_s": LINEAR_DT_S,
        },
        "expected": {
            "min_axis_amplification": amplification_bound_about_min_axis(BOX),
            "max_axis_amplification": amplification_bound_about_max_axis(BOX),
        },
        "tolerances": {
            "min_axis_amplification": {"abs": 0.0, "rel": 1.0e-5,
                                       "reason": _AMPLIFICATION_WHY},
            "max_axis_amplification": {"abs": 0.0, "rel": 1.0e-5,
                                       "reason": _AMPLIFICATION_WHY},
        },
    }, {
        "id": "oracle:rigidbody/dzhanibekov_flips",
        "inputs": {
            "kind": "intermediate_axis_flip_count",
            "body": "box", "spin_rad_per_s": SPIN_RAD_PER_S,
            "perturbation_rad_per_s": FLIP_PERTURBATION_RAD_PER_S,
            "horizon_s": FLIP_HORIZON_S, "dt_s": FLIP_DT_S,
        },
        "expected": {
            "flips_about_intermediate_axis": 2,
            "flips_about_min_axis": 0,
            "flips_about_max_axis": 0,
            "amplification_ordering": ["intermediate", "max", "min"],
        },
        "tolerances": {
            name: {
                "abs": 0.0, "rel": 0.0,
                "reason": "翻转次数是**确定性整数**（自转分量的变号次数），"
                          "零容差。把Dzhanibekov这条定性判据钉成可比的整数，"
                          "正是它能进门的原因（决策0018：进门的是确定性量）",
            }
            for name in (
                "flips_about_intermediate_axis",
                "flips_about_min_axis",
                "flips_about_max_axis",
            )
        } | {
            "amplification_ordering": {
                "abs": 0.0, "rel": 0.0,
                "reason": "非数值量必须零容差。中间轴的扰动放大必须严格大于最大轴、"
                          "最大轴严格大于最小轴——排序形式不受实现常数影响",
            },
        },
    }, {
        "id": "oracle:rigidbody/constant_torque_ramp",
        "inputs": {
            "kind": "constant_torque_about_a_principal_axis",
            "body": "box", "torque_nmm": TORQUE_NMM, "torque_axis": 2,
            "horizon_s": TORQUE_HORIZON_S, "dt_s": TORQUE_DT_S,
        },
        "expected": {
            "omega_z_rad_per_s": (
                TORQUE_NMM * NMM_PER_KG_MM2_PER_S2 * TORQUE_HORIZON_S / BOX[2]
            ),
            "omega_x_rad_per_s": 0.0,
            "omega_y_rad_per_s": 0.0,
        },
        "tolerances": {
            "omega_z_rad_per_s": {
                "abs": 0.0, "rel": 1.0e-12,
                "reason": "`ω = τ·t/I`对沿主轴的常力矩是**精确**的（`ω × I·ω = 0`），"
                          "RK4对常导数也精确，剩下的只是浮点噪声（实测rel 1.4e-14）。"
                          "**本条是本案例唯一钉住惯量绝对量级的判据**："
                          "无力矩的四层对`I → c·I`全盲，1000倍的mm²/m²错在那里查不出来",
            },
            "omega_x_rad_per_s": {
                "abs": 1.0e-15, "rel": 0.0,
                "reason": "沿主轴的力矩不产生横向角速度；abs判据因为期望恰为0",
            },
            "omega_y_rad_per_s": {
                "abs": 1.0e-15, "rel": 0.0,
                "reason": "同上",
            },
        },
    }, {
        "id": "oracle:rigidbody/free_flight_translation",
        "inputs": {
            "kind": "free_body_translation_rotation_decoupling",
            "body": "box", "gravity_mm_per_s2": GRAVITY_MM_PER_S2,
            "launch_speed_mm_per_s": LAUNCH_SPEED_MM_PER_S,
            "horizon_s": FLIGHT_HORIZON_S, "dt_s": FLIGHT_DT_S,
            "spinning_omega_rad_per_s": list(CONSERVATION_OMEGA_RAD_PER_S),
        },
        "expected": {
            "position_y_mm": (
                LAUNCH_SPEED_MM_PER_S * FLIGHT_HORIZON_S
                + GRAVITY_MM_PER_S2 * FLIGHT_HORIZON_S ** 2 / 2.0
            ),
            "spin_does_not_move_the_centre_of_mass": True,
        },
        "tolerances": {
            "position_y_mm": {
                "abs": 0.0, "rel": 1.0e-12,
                "reason": "抛物线闭式`x0 + v0T + aT²/2`；RK4对常加速度精确，"
                          "剩的是浮点噪声（实测rel 1.3e-14）。与"
                          "`cases/ballistic_free_flight`同一条物理、同一个数——"
                          "13自由度的打包次序错了这条会立刻红",
            },
            "spin_does_not_move_the_centre_of_mass": {
                "abs": 0.0, "rel": 0.0,
                "reason": "自由刚体绕质心的转动与质心平动**严格解耦**，"
                          "两次运行的质心轨迹要求**逐位相同**（不是近似相同）。"
                          "它挡的是'转动那一半漏进了平动那一半'这类串块错",
            },
        },
    }, {
        "id": "oracle:rigidbody/quaternion_norm_drift",
        "inputs": {
            "kind": "quaternion_norm_before_renormalisation",
            "body": "box",
            "omega0_rad_per_s": list(CONSERVATION_OMEGA_RAD_PER_S),
            "horizon_s": CONSERVATION_HORIZON_S,
            "dt_s_ladder": list(CONSERVATION_DT_LADDER_S),
        },
        "expected": {
            "max_norm_deviation_ceiling": 1.0e-11,
            "deviation_within_ceiling": True,
            "deviation_nonzero": True,
            "renormalisation_count_equals_steps": True,
        },
        "tolerances": {
            "max_norm_deviation_ceiling": {"abs": 0.0, "rel": 0.0,
                                           "reason": "上限值，零容差比较"},
            "deviation_within_ceiling": {
                "abs": 0.0, "rel": 0.0,
                "reason": "量的是**归一化之前**的`||q| − 1|`。归一化之后它恒等于1，"
                          "断言那一侧是一条永远通过的断言（spec/12第6.2节点名的假通过）。"
                          "实测最大2.288e-12（最粗档），上限1e-11留约4倍余量，"
                          "并比模块失败关闭阈值1e-6低五个数量级",
            },
            "deviation_nonzero": {
                "abs": 0.0, "rel": 0.0,
                "reason": "前置断言：偏离恒为零说明四元数根本没被推进（比如`q̇`写成了0），"
                          "此时上限判据会以最漂亮的方式假通过",
            },
            "renormalisation_count_equals_steps": {
                "abs": 0.0, "rel": 0.0,
                "reason": "确定性整数：归一化次数必须等于步数。少一次就是某条路径"
                          "跳过了投回单位球，而那条路径上的姿态是错的",
            },
        },
    }]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/rigid_body_free_flight",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/rigid_body_free_flight/generate_oracle.py",
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
