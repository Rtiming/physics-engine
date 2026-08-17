#!/usr/bin/env python3
"""自由跨段张力对线速度阶跃的金标生成器——**闭式解，独立于被验内核**。

本文件只import``math``与`physics_engine.oracles`（清单写入器）。
**不import`transport`、不import`drives`、不import任何力学模块**（轴7规则3）。
下面每一条式子都是在这里从头写的，与被验实现没有共用一行代码。

## 一、稳态：张力是**力矩平衡**的结果，不是一条换算

放线盘上没有电机，磁粉离合器是**制动**。稳态时转速不变、跨段里材料一进一出
正好抵消，于是``ω = v/R``且

    T·R = M + c·ω                ⟹     T = M/R + c·v/R²

``c = 0``时它恰是``T = M/R``——`drives.SpoolTension.tension_n`那条换算
**是本式在零轴承阻力矩下的特例**，不是一条独立的定律。

跨段里的材料长度由张力反解：``ε = (L_geo − L_mat)/L_mat``、``T = EA·ε``给出

    L_mat = EA·L_geo / (EA + T)

跨段作为一根弹簧的刚度是它的负导数

    K = −dT/dL_mat = EA·L_geo / L_mat²

**注意``K ≠ EA/L_geo``**：两者差``(1+ε)²``，本工况``ε ≈ 3.3e-4``⟹差``6.7e-4``相对，
**比本案例阶跃判据要分辨的量大**。

## 二、线性化：一个二阶系统，而``ω_n``里有一个``1000``

    dδT/dt        = −K·δv_放线
    dδv_放线/dt   = (1000·R²/J)·δT − (1000·c/J)·δv_放线

那个``1000``是``1 N = 1 kg·m/s² = 1000 kg·mm/s²``：力矩用``N·mm``、
惯量用``kg·mm²``时``α = 1000·M/J``。消去``δv``：

    δT'' + (1000c/J)·δT' + (1000·R²·K/J)·δT = 0
    ω_n = sqrt(1000·R²·K/J)          ζ = (1000·c/J)/(2·ω_n)

**掉了这个1000，``ω_n``差``sqrt(1000) = 31.6``倍**（`two_body_spring`那条
"1000倍单位bug的捕手"是同一类门）。

## 三、阶跃响应**不是**教科书那条二阶阶跃

``t = 0``收线端速度从``v0``跳到``v0 + Δv``。张力**连续**（材料长度是状态，
不会瞬变），而速度差**当场跳**``−Δv``。于是初值是

    y(0) = 0            y'(0) = K·Δv/ΔT_ss = ω_n/(2ζ)

（``y``是归一化到``ΔT_ss = T*(v1) − T*(v0) = c·Δv/R²``的响应。）
那个斜率冲击等价于传递函数多一个零点。解仍是闭式：

    y(t) = 1 − e^{−ζω_n t}·[cos(ω_d t) − ((1−2ζ²)/(2ζ√(1−ζ²)))·sin(ω_d t)]

令``y' = 0``：``tan(ω_d t) = −√(1−ζ²)/ζ``，第一个正根

    t_p = (π − acos ζ) / ω_d                    ← 标准二阶是 π/ω_d
    相对超调 = exp(−ζ(π − acos ζ)/√(1−ζ²)) / (2ζ)

## 四、``exp(−ζΦ/√(1−ζ²))``在本仓第三次出现，而``ζ = 0.5``是一个盲区

`contact.restitution_from_damping_ratio`取``Φ = 2·acos ζ``、
`drives.step_response_overshoot`取``Φ = π``、本式取``Φ = π − acos ζ``并多一个
前因子``1/(2ζ)``。

``π − acos ζ = 2·acos ζ`` ⟺ ``acos ζ = π/3`` ⟺ **``ζ = 0.5``**，
而``1/(2ζ)``在``ζ = 0.5``处**恰好是1**。**两处退化撞在同一个点上**，
于是本式与恢复系数式在``ζ = 0.5``处给出同一个实数。
``ζ = 0.5``正是写测试时最顺手的那个值——本清单因此在**五个**``ζ``上都给值。

## 五、``ζ → 0``：相对超调发散，而绝对幅值有限

``c = 0``时``ΔT_ss = 0``（稳态张力与线速度无关），于是相对超调无界。
绝对幅值仍是闭式——无阻尼时``δT = (K·Δv/ω_n)·sin(ω_n t)``：

    幅值 = K·Δv/ω_n              峰值时刻 = (π/2)/ω_n     ← 四分之一周期

**这条是本案例存在的理由**：开环几乎不衰减，速度扰动引起的张力振荡
**自己不会停**。控制器要控的就是它。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/free_span_tension_step"
ALGORITHM_VERSION = "1.0.0"

#: ``1 N = 1 kg·m/s² = 1000 kg·mm/s²``。**本文件自己写一遍，不从引擎取**。
MM_PER_M = 1000.0

#: 带材：4 mm宽×0.1 mm厚、``E ≈ 150 GPa`` ⟹ ``EA = 60 kN``。
#: **假设输入**，与`cases/capstan_tension_ratio`取同一个数（0062第二节裁决2）。
AXIAL_STIFFNESS_N = 60000.0
#: 自由跨长度：R1（最后一只实体导轮，世界系里固定）到落位点。
#: **假设输入**——plans/14第二节量出工件包络``432×270×224 mm``，
#: 几百mm是对的量级，但**这一段的实际长度没有实测**，而且真机上它每个样点都在变
#: （plans/14第3.3节的三号缺口）。本案例把它取成常数。
GEOMETRIC_LENGTH_MM = 300.0
#: 放线盘半径。**假设输入**。
REEL_RADIUS_MM = 60.0
#: 放线盘转动惯量``kg·mm²``。**假设输入**——满卷/空卷差几倍，本案例取一个定值。
REEL_INERTIA_KG_MM2 = 5000.0
#: 磁粉离合器的制动力矩。取``T = M/R = 20 N``落在真机张力区间10—30 N的中段。
#: **假设输入**：POC-050L的L后缀线圈参数厂家资料没有（0062第二节裁决3）。
BRAKE_TORQUE_NMM = 1200.0
#: 线速度与阶跃幅度。**假设输入**——真机线速度没有实测记录。
LINE_SPEED_MM_S = 20.0
#: 10%阶跃，用来判超调与峰值时刻。
STEP_MM_S = 2.0
#: 1%阶跃，用来判收敛阶。**小阶跃是有意的**：闭式是线性化的，
#: 它与非线性真解的系统偏差``∝ Δv²``而响应``∝ Δv``，
#: 所以相对偏差``∝ Δv``——阶跃越小，那条floor越低、可判的``dt``档越多。
FINE_STEP_MM_S = 0.2

#: 轴承粘性阻力矩``N·mm``每``rad/s``。**三档全是假设输入**：
#: * ``0``——无阻尼极限，本模型里的**唯一**阻尼通道关掉；
#: * ``50``——真实量级的轴承/密封阻力矩，给出``ζ ≈ 0.0132``（几乎无阻尼）；
#: * ``1000``——**为了把阶跃响应放到良态区而调出来的值**，给出``ζ ≈ 0.263``。
#:   它比真实轴承大二十倍，写在这里是为了不让人把它当实测。
DAMPING_UNDAMPED = 0.0
DAMPING_NOMINAL = 50.0
DAMPING_STIFF = 1000.0

#: 收敛阶研究的四档步长与终点时刻。
CONVERGENCE_DT_S = (1.6e-5, 8.0e-6, 4.0e-6, 2.0e-6)
CONVERGENCE_HORIZON_S = 0.02

#: 盲区门要扫的``ζ``。**必须多于一个**，理由见文件docstring第四节。
DAMPING_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)


def steady_tension_n(damping: float, speed: float) -> float:
    """``T = M/R + c·v/R²``。"""

    return (
        BRAKE_TORQUE_NMM / REEL_RADIUS_MM
        + damping * speed / (REEL_RADIUS_MM * REEL_RADIUS_MM)
    )


def material_length_mm(tension: float) -> float:
    """``L_mat = EA·L_geo/(EA + T)``。"""

    return AXIAL_STIFFNESS_N * GEOMETRIC_LENGTH_MM / (AXIAL_STIFFNESS_N + tension)


def span_stiffness_n_per_mm(tension: float) -> float:
    """``K = EA·L_geo/L_mat²``。**不是``EA/L_geo``**。"""

    length = material_length_mm(tension)
    return AXIAL_STIFFNESS_N * GEOMETRIC_LENGTH_MM / (length * length)


def natural_frequency_rad_s(tension: float) -> float:
    """``ω_n = sqrt(1000·R²·K/J)``。"""

    return math.sqrt(
        MM_PER_M
        * REEL_RADIUS_MM
        * REEL_RADIUS_MM
        * span_stiffness_n_per_mm(tension)
        / REEL_INERTIA_KG_MM2
    )


def damping_ratio(damping: float, tension: float) -> float:
    """``ζ = (1000·c/J)/(2·ω_n)``。"""

    return (MM_PER_M * damping / REEL_INERTIA_KG_MM2) / (
        2.0 * natural_frequency_rad_s(tension)
    )


def overshoot(zeta: float) -> float:
    """``exp(−ζ(π − acos ζ)/√(1−ζ²)) / (2ζ)``。"""

    return math.exp(
        -zeta * (math.pi - math.acos(zeta)) / math.sqrt(1.0 - zeta * zeta)
    ) / (2.0 * zeta)


def peak_time_s(zeta: float, natural: float) -> float:
    """``t_p = (π − acos ζ)/(ω_n·√(1−ζ²))``。"""

    return (math.pi - math.acos(zeta)) / (natural * math.sqrt(1.0 - zeta * zeta))


def restitution_shape(zeta: float) -> float:
    """`contact.restitution_from_damping_ratio`那一支：``Φ = 2·acos ζ``，无前因子。

    **这里重写一遍而不是import**：判据要判的正是"这两条不是同一条"，
    而import它等于让被判的两方共用一条实现。
    """

    return math.exp(
        -zeta * 2.0 * math.acos(zeta) / math.sqrt(1.0 - zeta * zeta)
    )


def textbook_shape(zeta: float) -> float:
    """`drives.step_response_overshoot`那一支：``Φ = π``，无前因子。"""

    return math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))


def step_response(zeta: float, natural: float, t_s: float) -> float:
    """归一化阶跃响应``y(t)``（0 → 1，峰值``1 + 超调``）。"""

    damped = natural * math.sqrt(1.0 - zeta * zeta)
    coefficient = (1.0 - 2.0 * zeta * zeta) / (2.0 * zeta * math.sqrt(1.0 - zeta * zeta))
    return 1.0 - math.exp(-zeta * natural * t_s) * (
        math.cos(damped * t_s) - coefficient * math.sin(damped * t_s)
    )


def tension_at(damping: float, step: float, t_s: float) -> float:
    """阶跃后``t_s``时刻的张力闭式（线性化）。"""

    before = steady_tension_n(damping, LINE_SPEED_MM_S)
    after = steady_tension_n(damping, LINE_SPEED_MM_S + step)
    zeta = damping_ratio(damping, after)
    natural = natural_frequency_rad_s(after)
    return before + (after - before) * step_response(zeta, natural, t_s)


def _step_block(damping: float, step: float, label: str) -> dict:
    before = steady_tension_n(damping, LINE_SPEED_MM_S)
    after = steady_tension_n(damping, LINE_SPEED_MM_S + step)
    zeta = damping_ratio(damping, after)
    natural = natural_frequency_rad_s(after)
    relative = overshoot(zeta)
    return {
        "id": f"oracle:free_span/velocity_step_{label}",
        "inputs": {
            "kind": "second_order_velocity_step_with_a_zero",
            "bearing_damping_nmm_s": damping,
            "line_speed_mm_s": LINE_SPEED_MM_S,
            "step_mm_s": step,
        },
        "expected": {
            "tension_before_n": before,
            "tension_after_n": after,
            "steady_change_n": after - before,
            "span_stiffness_n_per_mm": span_stiffness_n_per_mm(after),
            "natural_frequency_rad_s": natural,
            "damping_ratio": zeta,
            "relative_overshoot": relative,
            "peak_time_s": peak_time_s(zeta, natural),
            "peak_excursion_n": (after - before) * (1.0 + relative),
        },
        "tolerances": {
            "tension_before_n": {
                "abs": 1.0e-10, "rel": 0.0,
                "reason": (
                    "**离散不动点＝连续不动点**（半隐式Euler的两条更新在稳态处"
                    "各自恰为零），所以这条判的是模型不是步长。判绝对不判相对："
                    "剩下的偏差只来自``T → L_mat → T``这一趟往返的浮点舍入，"
                    "2026-08-17实测四档张力最坏3.908e-12 N，取1e-10留25倍余量"
                ),
            },
            "tension_after_n": {
                "abs": 1.0e-10, "rel": 0.0, "reason": "同上",
            },
            "steady_change_n": {
                "abs": 1.0e-12, "rel": 0.0,
                "reason": "``c·Δv/R²``是纯算术，双精度往返",
            },
            "span_stiffness_n_per_mm": {
                "abs": 0.0, "rel": 1.0e-14,
                "reason": (
                    "纯代数。**它与``EA/L_geo``差``(1+ε)²``即6.7e-4相对**，"
                    "比本案例超调判据的容差大一倍以上——拿``EA/L_geo``当K"
                    "会把正确实现判红"
                ),
            },
            "natural_frequency_rad_s": {
                "abs": 0.0, "rel": 1.0e-14,
                "reason": (
                    "纯代数。**它是那个``1000``的捕手**：掉了单位换算"
                    "``ω_n``差``sqrt(1000) = 31.6``倍，容差1e-14拦得住"
                ),
            },
            "damping_ratio": {"abs": 0.0, "rel": 1.0e-14, "reason": "纯代数"},
            "relative_overshoot": {
                "abs": 0.0, "rel": 3.0e-4,
                "reason": (
                    "**收敛结果**：半隐式Euler的一阶误差。2026-08-17实测"
                    "``dt = 1e-6``时``c = 50``给9.03e-6、``c = 1000``给7.35e-5，"
                    "取1000档的4倍余量。**收敛阶另有一条门判**——"
                    "只判落点会被一个碰巧的常数骗过，只判阶会被系统偏移骗过"
                ),
            },
            "peak_time_s": {
                "abs": 0.0, "rel": 1.0e-4,
                "reason": (
                    "峰值时刻的数值分辨率**就是步长**：``dt = 1e-6``对"
                    "``t_p ≈ 4.2e-3``是2.4e-4相对的栅格。实测两档都是1.76e-5，"
                    "取1e-4——**这条容差由采样栅格定，不由模型精度定**"
                ),
            },
            "peak_excursion_n": {
                "abs": 0.0, "rel": 2.0e-4,
                "reason": (
                    "实测``c = 50``给8.8e-6、``c = 1000``给3.93e-5，取5倍余量。"
                    "**它与相对超调必须并判**：``ΔT_ss``在``c → 0``时趋零，"
                    "只判相对超调时一个把``ΔT_ss``算错的实现可以照样过"
                ),
            },
        },
    }


def main() -> int:
    undamped_tension = steady_tension_n(DAMPING_UNDAMPED, LINE_SPEED_MM_S)
    undamped_stiffness = span_stiffness_n_per_mm(undamped_tension)
    undamped_natural = natural_frequency_rad_s(undamped_tension)

    oracles = [
        {
            "id": "oracle:free_span/steady_state_torque_balance",
            "inputs": {
                "kind": "payout_reel_torque_balance",
                "brake_torque_nmm": BRAKE_TORQUE_NMM,
                "reel_radius_mm": REEL_RADIUS_MM,
                "line_speed_mm_s": LINE_SPEED_MM_S,
            },
            "expected": {
                "tension_zero_damping_n": steady_tension_n(
                    DAMPING_UNDAMPED, LINE_SPEED_MM_S
                ),
                "torque_over_radius_n": BRAKE_TORQUE_NMM / REEL_RADIUS_MM,
                "tension_nominal_damping_n": steady_tension_n(
                    DAMPING_NOMINAL, LINE_SPEED_MM_S
                ),
                "tension_stiff_damping_n": steady_tension_n(
                    DAMPING_STIFF, LINE_SPEED_MM_S
                ),
                "material_length_zero_damping_mm": material_length_mm(
                    steady_tension_n(DAMPING_UNDAMPED, LINE_SPEED_MM_S)
                ),
                "strain_zero_damping": (
                    steady_tension_n(DAMPING_UNDAMPED, LINE_SPEED_MM_S)
                    / AXIAL_STIFFNESS_N
                ),
                "fixed_point_drift_n": 0.0,
            },
            "tolerances": {
                "tension_zero_damping_n": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**零容差，判逐位相等**：``c = 0``时本式必须与"
                        "`drives.SpoolTension.tension_n`的``T = M/R``给出**同一个浮点数**。"
                        "这一条说的是旧模型不是错的、它是本模型的特例——"
                        "**这句话要么逐位成立，要么就不该说**"
                    ),
                },
                "torque_over_radius_n": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "同上，是被对拍的另一半",
                },
                "tension_nominal_damping_n": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "纯算术",
                },
                "tension_stiff_damping_n": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "纯算术",
                },
                "material_length_zero_damping_mm": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "纯代数反解",
                },
                "strain_zero_damping": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``ε = T/EA``只在**一阶**上成立（严格定义是"
                        "``ε = (L_geo−L_mat)/L_mat``，而``T = EA·ε``是恒等的），"
                        "这里是恒等式：由``T``反解``L_mat``再算``ε``必然回到``T/EA``"
                    ),
                },
                "fixed_point_drift_n": {
                    "abs": 1.0e-10, "rel": 0.0,
                    "reason": (
                        "**从闭式稳态起手推进，张力一步都不该动**："
                        "半隐式Euler的两条更新在稳态处各自恰为零，"
                        "**离散不动点与连续不动点逐字相同**。"
                        "2026-08-17实测20000步（0.2秒）漂移**恰为0.0**；"
                        "容差按``T → L_mat → T``那一趟往返的浮点舍入给"
                        "（四档张力最坏3.908e-12 N），取1e-10留25倍余量"
                    ),
                },
            },
        },
        _step_block(DAMPING_NOMINAL, STEP_MM_S, "nominal"),
        _step_block(DAMPING_STIFF, STEP_MM_S, "stiff"),
        {
            "id": "oracle:free_span/undamped_oscillation",
            "inputs": {
                "kind": "undamped_span_oscillator",
                "bearing_damping_nmm_s": DAMPING_UNDAMPED,
                "step_mm_s": STEP_MM_S,
            },
            "expected": {
                "span_stiffness_n_per_mm": undamped_stiffness,
                "natural_frequency_rad_s": undamped_natural,
                "amplitude_n": undamped_stiffness * STEP_MM_S / undamped_natural,
                "quarter_period_s": (math.pi / 2.0) / undamped_natural,
                "period_s": 2.0 * math.pi / undamped_natural,
            },
            "tolerances": {
                "span_stiffness_n_per_mm": {
                    "abs": 0.0, "rel": 1.0e-14, "reason": "纯代数",
                },
                "natural_frequency_rad_s": {
                    "abs": 0.0, "rel": 1.0e-14, "reason": "纯代数，含那个1000",
                },
                "amplitude_n": {
                    "abs": 0.0, "rel": 1.0e-4,
                    "reason": (
                        "``K·Δv/ω_n``。2026-08-17实测（``dt = 1e-6``）相对偏差1.17e-5，"
                        "取8倍余量。**这一条是``ζ → 0``时相对超调发散的那一半**："
                        "相对量无界而绝对幅值有限"
                    ),
                },
                "quarter_period_s": {
                    "abs": 0.0, "rel": 1.0e-4,
                    "reason": (
                        "无阻尼时峰值落在四分之一周期，实测偏差7.8e-6；"
                        "它同时是``t_p = (π − acos ζ)/ω_d``在``ζ → 0``的极限"
                    ),
                },
                "period_s": {"abs": 0.0, "rel": 1.0e-14, "reason": "纯代数"},
            },
        },
        {
            "id": "oracle:free_span/overshoot_family_blind_spot",
            "inputs": {
                "kind": "exp_minus_zeta_phi_family",
                "damping_ratios": list(DAMPING_SWEEP),
            },
            "expected": {
                **{f"span_{index}": overshoot(z) for index, z in enumerate(DAMPING_SWEEP)},
                **{
                    f"restitution_{index}": restitution_shape(z)
                    for index, z in enumerate(DAMPING_SWEEP)
                },
                **{
                    f"textbook_{index}": textbook_shape(z)
                    for index, z in enumerate(DAMPING_SWEEP)
                },
                "degenerate_damping_ratio": 0.5,
            },
            "tolerances": {
                **{
                    f"span_{index}": {
                        "abs": 0.0, "rel": 1.0e-14,
                        "reason": "闭式求值；三支必须各自独立算，不许互相import",
                    }
                    for index in range(len(DAMPING_SWEEP))
                },
                **{
                    f"restitution_{index}": {
                        "abs": 0.0, "rel": 1.0e-14, "reason": "同上",
                    }
                    for index in range(len(DAMPING_SWEEP))
                },
                **{
                    f"textbook_{index}": {
                        "abs": 0.0, "rel": 1.0e-14, "reason": "同上",
                    }
                    for index in range(len(DAMPING_SWEEP))
                },
                "degenerate_damping_ratio": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**零容差，因为它是解出来的不是试出来的**："
                        "``π − acos ζ = 2·acos ζ`` ⟺ ``acos ζ = π/3`` ⟺ ``ζ = 1/2``，"
                        "且``1/(2ζ)``在同一点恰为1。**两处退化撞在同一个点上**，"
                        "而``ζ = 0.5``正是写测试时最顺手的那个值"
                    ),
                },
            },
        },
        {
            "id": "oracle:free_span/time_step_convergence",
            "inputs": {
                "kind": "semi_implicit_euler_refinement",
                "bearing_damping_nmm_s": DAMPING_STIFF,
                "step_mm_s": FINE_STEP_MM_S,
                "horizon_s": CONVERGENCE_HORIZON_S,
                "dt_s": list(CONVERGENCE_DT_S),
            },
            "expected": {
                "tension_at_horizon_n": tension_at(
                    DAMPING_STIFF, FINE_STEP_MM_S, CONVERGENCE_HORIZON_S
                ),
                "coarse_tension_at_horizon_n": tension_at(
                    DAMPING_STIFF, STEP_MM_S, CONVERGENCE_HORIZON_S
                ),
                "order_ratio_low": 1.9,
                "order_ratio_high": 2.1,
            },
            "tolerances": {
                "tension_at_horizon_n": {
                    "abs": 0.0, "rel": 5.0e-6,
                    "reason": (
                        "最细一档``dt = 2e-6``实测绝对误差4.513e-6 N对张力25.6 N"
                        "是1.76e-7相对；取5e-6留28倍余量。"
                        "**这一条只判落点，阶另有一条门**"
                    ),
                },
                "coarse_tension_at_horizon_n": {
                    "abs": 0.0, "rel": 5.0e-6,
                    "reason": "同上，是floor门用的那一档（``Δv``大十倍）",
                },
                "order_ratio_low": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**区间不写死为2**——与`harmonic_oscillator`那条"
                        "'收敛比不写死为4'同源：一致才是收敛的证据。"
                        "2026-08-17实测``Δv = 0.2``四档给1.9991/1.9991/1.9987，"
                        "区间[1.9, 2.1]留约50倍偏离余量"
                    ),
                },
                "order_ratio_high": {"abs": 0.0, "rel": 0.0, "reason": "同上"},
            },
        },
        {
            "id": "oracle:free_span/material_feed_timeline",
            "inputs": {
                "kind": "piecewise_constant_feed_rate",
                "times_s": [0.0, 0.002, 0.010],
                "feed_length_mm": [0.0, 0.04, 0.216],
            },
            "expected": {
                "segment_speed_0_mm_s": (0.04 - 0.0) / (0.002 - 0.0),
                "segment_speed_1_mm_s": (0.216 - 0.04) / (0.010 - 0.002),
                "length_at_mid_first_segment_mm": 0.02,
            },
            "tolerances": {
                "segment_speed_0_mm_s": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "**这就是plans/14第3.2节说的那一步**：累计喂料长度的时间导数"
                        "＝线速度。段内恒速语义下它是一次除法，双精度往返"
                    ),
                },
                "segment_speed_1_mm_s": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "同上。两段差10%——那就是本案例判的那次阶跃",
                },
                "length_at_mid_first_segment_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "段内线性＝段内恒速的积分，两者必须自洽",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/free_span_tension_step",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/free_span_tension_step/generate_oracle.py",
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
