#!/usr/bin/env python3
"""张力闭环对线速度阶跃的金标生成器——**闭式结构＋机器精度求根，独立于被验内核**。

本文件只import``cmath``/``math``与`physics_engine.oracles`（清单写入器）。
**不import`tension_control`、不import`drives`、不import`transport`**（轴7规则3）。
下面每一条式子都是在这里从头写的，与被验实现没有共用一行代码。

## 一、被验的那条链，写成四个状态

    输运账     dL_mat/dt = v_放线 − v_收线
    带材弹性   T = EA·(L_geo − L_mat)/L_mat
    力矩平衡   J·dω/dt = T·R − M − c·ω
    离合器     τ·dM/dt = k_M·I − M
    控制器     I = I_前馈 − (Kp + Ki/s + Kd·s)·δT

前三条是`cases/free_span_tension_step`已经验过的那套（决策0066），
后两条是本案例新接进来的。**装配层不新增物理**，所以金标也不新增物理——
它新增的是"这五条串起来之后是什么"。

## 二、线性化：一个**四阶**系统，系数是手推的

记``a = 1000R²/J``、``b = 1000R/J``、``d = 1000c/J``、
``K = −dT/dL_mat = EA·L_geo/L_mat²``、``Ka = K·a``、``G = K·b·k_M``。
那三个``1000``是``1 N = 1000 kg·mm/s²``——**掉一个，``ω_n``差31.6倍**。

    dδT/dt      = −K·δv_放线 + K·δv_收线
    dδv_放线/dt = a·δT − b·δM − d·δv_放线
    δM(s)       = k_M·δI(s)/(1 + τs)
    δI(s)       = −(Kp + Ki/s + Kd·s)·δT(s)

消元（两边乘``s(1+τs)``）：

    D(s) = τ·s⁴ + (1 + dτ)·s³ + (d + Ka·τ + G·Kd)·s²
           + (Ka + G·Kp)·s + G·Ki

收线端速度阶跃``Δv``（``v_收线(s) = Δv/s``）下张力偏差的像函数

    δT(s) = Δv·K·(τ·s² + (1 + dτ)·s + d) / D(s)

**分子里那个``s``被阶跃的``1/s``约掉了**——这就是"张力连续而速度差当场跳"
那条初值冲击在闭环里的形态，与0066第四节同源。

## 三、时域解：留数展开，根用Durand-Kerner求

``D``的根互不相同时（本案例各档实测都是），

    δT(t) = Δv·K·Σᵢ [ (τrᵢ² + (1+dτ)rᵢ + d) / D'(rᵢ) ] · e^{rᵢt}

**这是精确解不是数值积分**：只有"求``D``的根"这一步是数值的，
而它做到机器精度（Durand-Kerner，收敛判据``1e-16``相对）。
本仓已有先例——`euler_buckling`的临界载荷同样是**特征方程求根而非抄表**。

``Ki = 0``时``D``的常数项**恰为零**，本文件把那个零根**精确地提出来**
而不是让求根器去逼近它：``e^{0·t} = 1``那一项正是开环残留的稳态偏移，
把它交给数值求根会在长时程上产生一个假的漂移。

## 四、判据取三个量，因为它们红了说明的事不一样

| 量 | 它守什么 |
|---|---|
| **峰值与峰值时刻** | 瞬态的幅度与相位。**控制器带宽够不够就看它** |
| **ISE ``∫₀^T δT²dt``** | 整段的能量。**"压下去了多少"是这个数，不是峰值** |
| **末端稳态偏移** | 直流。积分项在就恒为零，不在就是``c·Δv/R²`` |

只判峰值会被一个"把峰压低了却一直不衰减"的实现骗过；
只判ISE会被一个"峰高得离谱但衰减极快"的实现骗过；
只判稳态会连开环都过（开环也有一个确定的稳态）。**三条必须并判。**

ISE的闭式是留数的双重和

    ∫₀^T δTγdt = ΣᵢΣⱼ wᵢwⱼ·(e^{(rᵢ+rⱼ)T} − 1)/(rᵢ+rⱼ)

（``rᵢ+rⱼ = 0``那一项取``wᵢwⱼ·T``——开环的零根成对时正好落在这里。）

## 五、三个档位各自在回答什么

1. **``open``（全零增益）**——被控对象自己。它必须重现
   `cases/free_span_tension_step`那条独立算出来的峰值，
   **两条案例的闭式是各写各的，对上了才说明装配层没有偷偷改物理**。

   **对上了，但不是逐位**（2026-08-17实测**4.60e-7**相对，峰值时刻4.54e-7），
   而差在哪里是查得出来的：`free_span`把线性化参考点取在**阶跃后**的稳态
   （``v = 22``），本文件取在**阶跃前**的工作点（``v = 20``）。
   两点的``T*``差``ΔT_ss = 0.0278 N``，而``K ∝ (EA+T)²``⟹
   ``ΔK/K = 2·0.0278/60020 = 9.26e-7``；峰值``∝ K·Δv/ω_n ∝ √K``⟹
   ``4.63e-7``——**与实测的4.60e-7对上**。
   **"线性化在哪一点"是一个选择，不是一个细节**，这条门判的正是
   "两个选择差多少、差得对不对"。

   稳态偏移那一条**差2.85e-14相对**，病根不同：`free_span`算的是
   ``T*(22) − T*(20)``（两个20.28 N相减，**灾难性抵消**），
   本文件直接算``Δv·c/R²``。**同一个量的两种算法差14位，那是浮点不是物理**；
2. **``fast``（``τ = 0.5 ms``）**——**一次数值实验，不是一台机器**。
   任何磁粉离合器都没有这个响应。它存在的唯一理由是把
   "环写错了"与"执行器太慢了"分开：环对不对，在这一档上才判得出来；
3. **``nominal``（``τ = 50 ms``，POC-050L的假设值）**——真机那一档。
   本案例最要紧的一个结论就在这里：**闭环把扰动响应变**坏**了**。

## 六、``μ``与包角：传感器位置≠被控点

    T_落位点 = T_传感器 · exp(μθ)

``μ = 0.3``、``θ = 90°``给``exp(0.4712) = 1.601978``——落位点比设定高60.198%。
方向搞反时估计差``exp(2μθ) = 2.566332``，**是平方不是两倍**。
本文件把``exp(μθ)``与``exp(2μθ)``各算一次并**分别**入清单：
让判据能判"平方"这件事本身，而不是判一个凑出来的数。

## 七、全部输入是假设，产物永久``hypothesis_only``

带材``EA``、跨段长度、放线盘转动惯量与轴承阻力矩、离合器的电流-扭矩增益与
时滞、绞盘``μ``与包角——一条实测都没有，逐条出处见0062第二节裁决2。
"""

from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/closed_loop_tension_step"
ALGORITHM_VERSION = "1.0.0"

#: ``1 N = 1 kg·m/s² = 1000 kg·mm/s²``。**本文件自己写一遍，不从引擎取**。
MM_PER_M = 1000.0

#: 对象参数**逐条与`cases/free_span_tension_step`取同一个数**——
#: 两条案例判的是同一条链路的两半，参数一旦分叉就没法互相印证。
AXIAL_STIFFNESS_N = 60000.0
GEOMETRIC_LENGTH_MM = 300.0
REEL_RADIUS_MM = 60.0
REEL_INERTIA_KG_MM2 = 5000.0
BEARING_DAMPING_NMM_S = 50.0
BRAKE_TORQUE_NMM = 1200.0
LINE_SPEED_MM_S = 20.0
STEP_MM_S = 2.0

#: 磁粉离合器：POC-050基型样本外推（0062第二节裁决2）。
TORQUE_PER_AMPERE_NMM = 23256.0
RATED_TORQUE_NMM = 50000.0
#: **假设输入**，厂家资料没有。50 ms是`tests/test_drives.py`一直用的那个值。
LAG_NOMINAL_S = 0.05
#: **一次数值实验，不是一台机器**：0.5 ms比任何磁粉离合器都快。
#: 它存在的理由见文件docstring第五节。
LAG_FAST_S = 0.0005

#: 绞盘：`docs/plans/13`第六节量过的那一组。
FRICTION_COEFFICIENT = 0.3
WRAP_ANGLE_RAD = math.pi / 2.0

#: ISE的积分上限。取0.2 s是因为`free_span_tension_step`用它跑满了12.08个周期。
ISE_HORIZON_S = 0.2
#: 稳态误差的观测时刻。开环衰减时间常数是``1/(ζω_n) = 0.2 s``，
#: 取2.0 s ⟹ 残余振荡``e^{−10} = 4.5e-5``相对幅值，**比要判的偏移小两个数量级**。
STEADY_HORIZON_S = 2.0

#: 闭环设计：把振荡模态推到``ω_目标 = 2ω_n``、``ζ_目标 = 0.7``，
#: 积分增益取Routh界的0.3倍。**三个数都是设计选择不是物理**，
#: 写成显式常数是为了让"换一套增益"这件事只改这里。
BANDWIDTH_MULTIPLIER = 2.0
TARGET_DAMPING_RATIO = 0.7
INTEGRAL_GAIN_FRACTION = 0.3
#: 故意很差的控制器：纯比例，增益取好控制器的1%与0.1%（plans/15第1.1条）。
BAD_GAIN_FRACTIONS = (0.01, 0.001)


# ------------------------------------------------------------------ 工作点 ---


def steady_tension_n(speed_mm_s: float) -> float:
    """``T = M/R + c·v/R²``（0066第三节）。"""

    return BRAKE_TORQUE_NMM / REEL_RADIUS_MM + BEARING_DAMPING_NMM_S * speed_mm_s / (
        REEL_RADIUS_MM * REEL_RADIUS_MM
    )


def material_length_mm(tension_n: float) -> float:
    return AXIAL_STIFFNESS_N * GEOMETRIC_LENGTH_MM / (AXIAL_STIFFNESS_N + tension_n)


def span_stiffness_n_per_mm(tension_n: float) -> float:
    """``K = EA·L_geo/L_mat²``。**不是``EA/L_geo``**，两者差``(1+ε)²``。"""

    length = material_length_mm(tension_n)
    return AXIAL_STIFFNESS_N * GEOMETRIC_LENGTH_MM / (length * length)


SETPOINT_N = steady_tension_n(LINE_SPEED_MM_S)
STIFFNESS = span_stiffness_n_per_mm(SETPOINT_N)
ACCEL_PER_TENSION = MM_PER_M * REEL_RADIUS_MM * REEL_RADIUS_MM / REEL_INERTIA_KG_MM2
ACCEL_PER_TORQUE = MM_PER_M * REEL_RADIUS_MM / REEL_INERTIA_KG_MM2
VELOCITY_DAMPING = MM_PER_M * BEARING_DAMPING_NMM_S / REEL_INERTIA_KG_MM2
STIFFNESS_TIMES_A = STIFFNESS * ACCEL_PER_TENSION
LOOP_GAIN = STIFFNESS * ACCEL_PER_TORQUE * TORQUE_PER_AMPERE_NMM
NATURAL_FREQUENCY = math.sqrt(STIFFNESS_TIMES_A)
DAMPING_RATIO = VELOCITY_DAMPING / (2.0 * NATURAL_FREQUENCY)


# ------------------------------------------------------- 特征多项式与稳定界 ---


def characteristic_polynomial(
    lag_s: float, proportional: float, integral_gain: float, derivative: float
) -> tuple[float, ...]:
    """从``s⁴``到``s⁰``。文件docstring第二节的那一行。"""

    return (
        lag_s,
        1.0 + VELOCITY_DAMPING * lag_s,
        VELOCITY_DAMPING + STIFFNESS_TIMES_A * lag_s + LOOP_GAIN * derivative,
        STIFFNESS_TIMES_A + LOOP_GAIN * proportional,
        LOOP_GAIN * integral_gain,
    )


def integral_gain_limit(lag_s: float, proportional: float, derivative: float) -> float:
    """四阶Routh：``Ki_界 = (a₃a₂ − a₄a₁)·a₁/(a₃²·G)``。

    ``τ = 0``时``a₄ = 0``，退化成三次的``(d + G·Kd)(Ka + G·Kp)/G``。
    **两条是同一条**，清单里两边都给值让判据去对。
    """

    fourth, third, second, first, _ = characteristic_polynomial(
        lag_s, proportional, 0.0, derivative
    )
    routh = third * second - fourth * first
    if routh <= 0.0 or min(third, second, first) <= 0.0:
        raise ValueError(
            f"Routh第一列非正（a₃a₂−a₄a₁ = {routh!r}）：τ = {lag_s!r}下"
            "任何正的积分增益都不稳，'界'这个说法不成立"
        )
    return routh * first / (third * third * LOOP_GAIN)


def pid_design(lag_s: float) -> tuple[float, float, float]:
    """按``ω_目标``/``ζ_目标``反解``(Kp, Ki, Kd)``。

    ``Kp``定闭环频率（``s¹``系数 ＝ ``ω_目标²``）、
    ``Kd``定闭环阻尼（``s²``系数 ＝ ``2ζ_目标ω_目标``）、
    ``Ki``取Routh界的一个固定份额。**``Kd``是唯一能加阻尼的那一项**——
    ``Ki``只出现在常数项上，而它把Routh裕度吃掉。
    """

    target = BANDWIDTH_MULTIPLIER * NATURAL_FREQUENCY
    proportional = (target * target - STIFFNESS_TIMES_A) / LOOP_GAIN
    derivative = (
        2.0 * TARGET_DAMPING_RATIO * target - VELOCITY_DAMPING - STIFFNESS_TIMES_A * lag_s
    ) / LOOP_GAIN
    derivative = max(derivative, 0.0)
    integral = INTEGRAL_GAIN_FRACTION * integral_gain_limit(
        lag_s, proportional, derivative
    )
    return proportional, integral, derivative


# --------------------------------------------------------------- 根与留数 ---


def _polynomial_roots(coefficients: tuple[float, ...]) -> list[complex]:
    """Durand-Kerner。**确定性**：初值、迭代上限与收敛判据全是常数。"""

    trimmed = list(coefficients)
    while trimmed and trimmed[0] == 0.0:
        trimmed = trimmed[1:]
    degree = len(trimmed) - 1
    monic = [value / trimmed[0] for value in trimmed]
    scale = max(abs(value) for value in monic[1:]) if degree else 1.0
    guesses = [
        complex(0.4, 0.9) ** index * (1.0 + scale) ** (1.0 / max(degree, 1))
        for index in range(degree)
    ]

    def evaluate(point: complex) -> complex:
        accumulator = 0j
        for value in monic:
            accumulator = accumulator * point + value
        return accumulator

    for _ in range(5000):
        moved = 0.0
        updated = []
        for index, point in enumerate(guesses):
            denominator = 1.0 + 0j
            for other, neighbour in enumerate(guesses):
                if index != other:
                    denominator *= point - neighbour
            correction = evaluate(point) / denominator
            moved = max(moved, abs(correction))
            updated.append(point - correction)
        guesses = updated
        if moved < 1.0e-16 * (1.0 + max(abs(point) for point in guesses)):
            break
    return guesses


def modes(lag_s: float, proportional: float, integral_gain: float, derivative: float):
    """``[(根, 留数×Δv·K), …]``——时域解的全部内容。"""

    polynomial = list(
        characteristic_polynomial(lag_s, proportional, integral_gain, derivative)
    )
    #: ``Ki = 0``时常数项恰为零：**精确**地把``s``提出来（文件docstring第三节）。
    zero_roots = 0
    while len(polynomial) > 1 and polynomial[-1] == 0.0:
        polynomial = polynomial[:-1]
        zero_roots += 1
    roots = _polynomial_roots(tuple(polynomial)) + [0.0 + 0.0j] * zero_roots
    polynomial = polynomial + [0.0] * zero_roots
    degree = len(polynomial) - 1
    derivative_poly = [polynomial[i] * (degree - i) for i in range(degree)]

    result = []
    for root in roots:
        numerator = (
            lag_s * root * root + (1.0 + VELOCITY_DAMPING * lag_s) * root + VELOCITY_DAMPING
        )
        denominator = 0j
        for value in derivative_poly:
            denominator = denominator * root + value
        result.append((root, STEP_MM_S * STIFFNESS * numerator / denominator))
    return result


def response_n(spectrum, t_s: float) -> float:
    accumulator = 0j
    for root, weight in spectrum:
        accumulator += weight * cmath.exp(root * t_s)
    return accumulator.real


def peak_n(spectrum, horizon_s: float = 0.05) -> tuple[float, float]:
    """先粗扫定位再黄金分割精修——**峰值时刻是解出来的，不是采样点**。"""

    samples = 500000
    best, best_time = -math.inf, 0.0
    for index in range(samples + 1):
        t_s = index * horizon_s / samples
        value = response_n(spectrum, t_s)
        if value > best:
            best, best_time = value, t_s
    low = best_time - horizon_s / samples
    high = best_time + horizon_s / samples
    for _ in range(200):
        first = low + (high - low) / 3.0
        second = high - (high - low) / 3.0
        if response_n(spectrum, first) < response_n(spectrum, second):
            low = first
        else:
            high = second
    t_s = 0.5 * (low + high)
    return response_n(spectrum, t_s), t_s


def integral_squared_error(spectrum, horizon_s: float) -> float:
    """``ΣᵢΣⱼ wᵢwⱼ(e^{(rᵢ+rⱼ)T} − 1)/(rᵢ+rⱼ)``，零和那一项取``wᵢwⱼT``。"""

    accumulator = 0j
    for left_root, left_weight in spectrum:
        for right_root, right_weight in spectrum:
            total = left_root + right_root
            if total == 0:
                accumulator += left_weight * right_weight * horizon_s
            else:
                accumulator += (
                    left_weight
                    * right_weight
                    * (cmath.exp(total * horizon_s) - 1.0)
                    / total
                )
    return accumulator.real


def steady_offset_n(proportional: float, integral_gain: float) -> float:
    """终值定理：积分项在则恒为零，否则``Δv·K·d/(Ka + G·Kp)``。

    ``Kp = 0``时它退化成``Δv·c/R²``——**开环那条稳态偏移**，
    与`free_span_tension_step`的``steady_change_n``是同一个数。
    """

    if integral_gain != 0.0:
        return 0.0
    return (
        STEP_MM_S
        * STIFFNESS
        * VELOCITY_DAMPING
        / (STIFFNESS_TIMES_A + LOOP_GAIN * proportional)
    )


# ------------------------------------------------------------------ 清单 ---


def _band_block(band_id: str, lag_s: float, gains: tuple[float, float, float]) -> dict:
    proportional, integral_gain, derivative = gains
    spectrum = modes(lag_s, proportional, integral_gain, derivative)
    peak, peak_time = peak_n(spectrum)
    return {
        "id": f"oracle:closed_loop/{band_id}",
        "inputs": {
            "kind": "fourth_order_tension_loop_velocity_step",
            "clutch_lag_s": lag_s,
            "proportional_a_per_n": proportional,
            "integral_gain_a_per_n_s": integral_gain,
            "derivative_a_s_per_n": derivative,
            "step_mm_s": STEP_MM_S,
        },
        "expected": {
            "peak_excursion_n": peak,
            "peak_time_s": peak_time,
            "integral_squared_error_n2_s": integral_squared_error(
                spectrum, ISE_HORIZON_S
            ),
            "steady_offset_n": steady_offset_n(proportional, integral_gain),
            "dominant_real_part_per_s": max(root.real for root, _ in spectrum),
            #: **振荡模态**的实部——``|Im|``最大的那个根。
            #: ``dominant_real_part``在开环那档是**零根**（那是稳态偏移那一支），
            #: 于是它回答不了"振荡衰减多快"。这一条才回答。
            "oscillatory_real_part_per_s": max(
                spectrum, key=lambda item: abs(item[0].imag)
            )[0].real,
            "oscillatory_frequency_rad_s": abs(
                max(spectrum, key=lambda item: abs(item[0].imag))[0].imag
            ),
        },
        "tolerances": {
            "peak_excursion_n": {
                "abs": 0.0, "rel": 5.0e-3,
                "reason": (
                    "**收敛结果**：半隐式Euler的对象＋精确ZOH的离合器＋采样控制器，"
                    "组合误差2026-08-17实测``dt = 1e-6``时``open``档9.26e-6、"
                    "``nominal``档7.74e-6、``fast``档5.90e-4。取5e-3给最差那档"
                    "8倍余量。**``fast``档差两个数量级是因为它的闭环极点在470/s上**，"
                    "同一个``dt``对更快的动态就是更粗的网格"
                ),
            },
            "peak_time_s": {
                "abs": 0.0, "rel": 2.0e-3,
                "reason": (
                    "峰值时刻的数值分辨率**就是步长**：``dt = 1e-6``对``fast``档的"
                    "``t_p ≈ 1.44e-3``是6.9e-4相对的栅格。实测三档最坏2.5e-4"
                    "（``fast``档），取2e-3——**这条容差由采样栅格定，不由模型精度定**"
                ),
            },
            "integral_squared_error_n2_s": {
                "abs": 0.0, "rel": 5.0e-3,
                "reason": (
                    "实测``open``档8.07e-6、``nominal``档1.37e-5、``fast``档6.07e-4。"
                    "取5e-3。**它与峰值必须并判**：只判峰值会被一个"
                    "'把峰压低了却一直不衰减'的实现骗过——而那正是``nominal``档"
                    "干的事（峰值×1.07而ISE×1.39，两个数一起才说得清）"
                ),
            },
            "steady_offset_n": {
                "abs": 1.0e-6, "rel": 0.0,
                "reason": (
                    "**判绝对不判相对**：带积分项时它恰为零，相对容差对零无意义。"
                    "2026-08-17实测（``t = 2.0 s``、``dt = 1e-5``）好控制器给"
                    "1.60e-10、完全不控给2.7734e-2对闭式2.7778e-2（差4.34e-5，"
                    "那是``e^{−10} = 4.5e-5``的残余振荡不是偏差）。"
                    "取1e-6：**它比好控制器的实测大四个数量级，比坏控制器的偏移"
                    "小四个数量级**——两侧各留四个数量级，这条判据分得开"
                ),
            },
            "dominant_real_part_per_s": {
                "abs": 1.0e-12, "rel": 1.0e-10,
                "reason": (
                    "纯代数（Durand-Kerner到机器精度）。**开环那档它恰为零**"
                    "——那是稳态偏移那一支的零根，所以要判绝对不判相对。"
                    "本条不与推进对拍，它是闭式内部的自洽判据"
                ),
            },
            "oscillatory_real_part_per_s": {
                "abs": 0.0, "rel": 1.0e-10,
                "reason": (
                    "**振荡衰减率本身**：开环−5.000（时间常数0.2 s，"
                    "**比一次落位动作还长**）、``fast``档−470.709（2.1 ms）、"
                    "``nominal``档−4.579——**比开环还小**，那就是"
                    "'积分项吃掉阻尼'这句话的量"
                ),
            },
            "oscillatory_frequency_rad_s": {
                "abs": 0.0, "rel": 1.0e-10,
                "reason": (
                    "开环379.569（≈``ω_n``）。**它与衰减率必须并判**："
                    "只判衰减率时，一个把频率也改了的实现可以照样过"
                ),
            },
        },
    }


def main() -> int:
    fast_gains = pid_design(LAG_FAST_S)
    #: 额定档**只能用纯积分**：把``fast``那套设计放上去时Routh第一列为负
    #: （实测``a₃a₂ − a₄a₁ = −17997.17``），任何正的积分增益都不稳。
    #: 这条"失败关闭"本身是一条判据，见清单``bandwidth_verdict``。
    nominal_limit = integral_gain_limit(LAG_NOMINAL_S, 0.0, 0.0)
    nominal_gains = (0.0, INTEGRAL_GAIN_FRACTION * nominal_limit, 0.0)

    open_spectrum = modes(LAG_NOMINAL_S, 0.0, 0.0, 0.0)
    open_peak, _ = peak_n(open_spectrum)
    open_ise = integral_squared_error(open_spectrum, ISE_HORIZON_S)

    fast_spectrum = modes(LAG_FAST_S, *fast_gains)
    fast_peak, _ = peak_n(fast_spectrum)
    nominal_spectrum = modes(LAG_NOMINAL_S, *nominal_gains)
    nominal_peak, _ = peak_n(nominal_spectrum)

    fast_limit_pure_integral = integral_gain_limit(LAG_FAST_S, 0.0, 0.0)
    ratio = math.exp(FRICTION_COEFFICIENT * WRAP_ANGLE_RAD)

    bad_blocks = {}
    for fraction in BAD_GAIN_FRACTIONS:
        proportional = fraction * fast_gains[0]
        spectrum = modes(LAG_FAST_S, proportional, 0.0, 0.0)
        peak, _ = peak_n(spectrum)
        key = f"{fraction:g}".replace(".", "p")
        bad_blocks[f"bad_peak_excursion_n_{key}"] = peak
        bad_blocks[f"bad_steady_offset_n_{key}"] = steady_offset_n(proportional, 0.0)
        bad_blocks[f"bad_proportional_a_per_n_{key}"] = proportional
        #: **判据要对的是这一个，不是渐近值**：``t = 2 s``上振荡还没走完
        #: （比例增益把振荡实部从−5.000推到−3.954，``e^{σt} = 3.68e-4``），
        #: 而那不是偏差是没跑完的尾巴。闭式把尾巴一起给出来，判据就紧16—100倍。
        bad_blocks[f"bad_response_at_horizon_n_{key}"] = response_n(
            spectrum, STEADY_HORIZON_S
        )
        #: 顺手记下来：**纯比例增益也把阻尼吃掉了**（−5.000 → −3.954）。
        bad_blocks[f"bad_oscillatory_real_part_per_s_{key}"] = max(
            spectrum, key=lambda item: abs(item[0].imag)
        )[0].real

    uncontrolled_spectrum = modes(LAG_FAST_S, 0.0, 0.0, 0.0)

    oracles = [
        {
            "id": "oracle:closed_loop/plant_linearisation",
            "inputs": {
                "kind": "operating_point_and_channel_authority",
                "brake_torque_nmm": BRAKE_TORQUE_NMM,
                "line_speed_mm_s": LINE_SPEED_MM_S,
                "bearing_damping_nmm_s": BEARING_DAMPING_NMM_S,
            },
            "expected": {
                "setpoint_n": SETPOINT_N,
                "span_stiffness_n_per_mm": STIFFNESS,
                "natural_frequency_rad_s": NATURAL_FREQUENCY,
                "damping_ratio": DAMPING_RATIO,
                "loop_gain": LOOP_GAIN,
                #: 裁决A第1条：两个通道按各自额定量的1%算出来的张力变化。
                "brake_authority_n_per_percent": 0.01 * BRAKE_TORQUE_NMM / REEL_RADIUS_MM,
                "takeup_authority_n_per_percent": (
                    0.01 * BEARING_DAMPING_NMM_S * LINE_SPEED_MM_S
                    / (REEL_RADIUS_MM * REEL_RADIUS_MM)
                ),
                "authority_ratio": (
                    (0.01 * BRAKE_TORQUE_NMM / REEL_RADIUS_MM)
                    / (
                        0.01 * BEARING_DAMPING_NMM_S * LINE_SPEED_MM_S
                        / (REEL_RADIUS_MM * REEL_RADIUS_MM)
                    )
                ),
                #: ``c = 0``时收线速度通道的稳态权限**恰为零**——裁决A的硬理由。
                "takeup_authority_at_zero_damping_n_per_percent": 0.0,
                "clutch_pole_nominal_rad_s": 1.0 / LAG_NOMINAL_S,
                "clutch_pole_fast_rad_s": 1.0 / LAG_FAST_S,
                "nominal_bandwidth_shortfall": NATURAL_FREQUENCY * LAG_NOMINAL_S,
            },
            "tolerances": {
                "setpoint_n": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "纯算术；与`free_span_tension_step`的``T*``同一个数",
                },
                "span_stiffness_n_per_mm": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "纯代数。**它与``EA/L_geo``差``(1+ε)²``即6.7e-4相对**",
                },
                "natural_frequency_rad_s": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "纯代数；**那个1000的捕手**，掉了差31.6倍",
                },
                "damping_ratio": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "纯代数。0.01317——**开环几乎不衰减**就是这个数",
                },
                "loop_gain": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "``G = K·(1000R/J)·k_M``，纯代数",
                },
                "brake_authority_n_per_percent": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``0.01·M₀/R``。**裁决A第1条的被除数**",
                },
                "takeup_authority_n_per_percent": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``0.01·c·v/R²``。**裁决A第1条的除数**",
                },
                "authority_ratio": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": (
                        "72倍。**这是'控制器接哪一路'那条裁决的量**——"
                        "它同时说明为什么不能反过来接：收线速度要动3.6倍"
                        "才换来制动力矩5%所换来的张力"
                    ),
                },
                "takeup_authority_at_zero_damping_n_per_percent": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**零容差，判恒等于零**：``∂T/∂v = c/R²``在``c = 0``时"
                        "恰为零。裁决A最硬的那半条——收线速度通道的全部稳态权限"
                        "都建在``c``上，而``c``是本模型里唯一没有实测的阻尼参数"
                    ),
                },
                "clutch_pole_nominal_rad_s": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "``1/τ``",
                },
                "clutch_pole_fast_rad_s": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "同上",
                },
                "nominal_bandwidth_shortfall": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": (
                        "``ω_n·τ = 18.98``——**执行器比对象慢19倍**。"
                        "这一个无量纲数就是``nominal``档全部结论的来源"
                    ),
                },
            },
        },
        _band_block("open_loop_step", LAG_NOMINAL_S, (0.0, 0.0, 0.0)),
        _band_block("fast_band_step", LAG_FAST_S, fast_gains),
        _band_block("nominal_band_step", LAG_NOMINAL_S, nominal_gains),
        {
            "id": "oracle:closed_loop/suppression_verdict",
            "inputs": {
                "kind": "closed_over_open_ratios",
                "ise_horizon_s": ISE_HORIZON_S,
            },
            "expected": {
                "fast_peak_ratio": fast_peak / open_peak,
                "fast_ise_ratio": integral_squared_error(fast_spectrum, ISE_HORIZON_S)
                / open_ise,
                "nominal_peak_ratio": nominal_peak / open_peak,
                "nominal_ise_ratio": integral_squared_error(
                    nominal_spectrum, ISE_HORIZON_S
                )
                / open_ise,
                #: **额定档比1大**——闭环把它变坏了。零容差判的是这个不等号。
                "nominal_makes_it_worse": 1.0,
            },
            "tolerances": {
                "fast_peak_ratio": {
                    "abs": 0.0, "rel": 5.0e-3,
                    "reason": "两条闭式相除，容差随分子分母的推进对拍容差走",
                },
                "fast_ise_ratio": {
                    "abs": 0.0, "rel": 5.0e-3,
                    "reason": (
                        "**这就是'压下去了多少'那个数**：6.07e-3，即165倍。"
                        "峰值只压到0.318倍而ISE压到六百分之一——"
                        "**压下去的主要不是峰高，是它不再一直响**"
                    ),
                },
                "nominal_peak_ratio": {
                    "abs": 0.0, "rel": 5.0e-3,
                    "reason": "1.0727——**大于1**。闭环把峰值抬高了7.3%",
                },
                "nominal_ise_ratio": {
                    "abs": 0.0, "rel": 5.0e-3,
                    "reason": (
                        "1.3882——**大于1**。积分项把振荡模态的阻尼吃掉了"
                        "（闭式：开环极点−5.000±379.569j，额定闭环−4.579±365.636j）。"
                        "**这不是实现的毛病，是闭式预言的**"
                    ),
                },
                "nominal_makes_it_worse": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**零容差的是那个不等号**：两条比值都必须严格大于这个1。"
                        "本案例最要紧的结论——**在真机那一档上，"
                        "这条通道的闭环让扰动响应更坏**"
                    ),
                },
            },
        },
        {
            "id": "oracle:closed_loop/pluggability",
            "inputs": {
                "kind": "same_case_worse_controller",
                "clutch_lag_s": LAG_FAST_S,
                "bad_gain_fractions": list(BAD_GAIN_FRACTIONS),
                "steady_horizon_s": STEADY_HORIZON_S,
            },
            "expected": {
                "good_peak_excursion_n": fast_peak,
                "good_steady_offset_n": 0.0,
                "uncontrolled_steady_offset_n": steady_offset_n(0.0, 0.0),
                "uncontrolled_response_at_horizon_n": response_n(
                    uncontrolled_spectrum, STEADY_HORIZON_S
                ),
                **bad_blocks,
            },
            "tolerances": {
                "good_peak_excursion_n": {
                    "abs": 0.0, "rel": 5.0e-3, "reason": "同``fast``档峰值",
                },
                "good_steady_offset_n": {
                    "abs": 1.0e-6, "rel": 0.0,
                    "reason": "积分项在，终值定理给恒零；实测1.60e-10",
                },
                "uncontrolled_steady_offset_n": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``Δv·c/R² = 0.0277778``。它同时是"
                        "`free_span_tension_step`的``steady_change_n``——"
                        "**两条案例必须给同一个数**"
                    ),
                },
                "uncontrolled_response_at_horizon_n": {
                    "abs": 5.0e-5, "rel": 0.0,
                    "reason": (
                        "``t = 2 s``上的闭式值（含没走完的振荡尾巴）。"
                        "实测差4.15e-7，取5e-5留120倍余量"
                    ),
                },
                **{
                    key: {
                        "abs": 0.0, "rel": 5.0e-3,
                        "reason": (
                            "坏控制器（纯比例、增益极小）的峰值。"
                            "**它必须显著大于好控制器那一个**，那就是"
                            "'换得动'的判据本身"
                        ),
                    }
                    for key in bad_blocks
                    if key.startswith("bad_peak")
                },
                **{
                    key: {
                        "abs": 0.0, "rel": 1.0e-14,
                        "reason": (
                            "``Δv·K·d/(Ka + G·Kp)``，纯代数**渐近值**。"
                            "增益极小时它几乎就是完全不控那个数——"
                            "**这正是'增益极小'的含义**。"
                            "**推进对拍不判它判下一条**：``t = 2 s``上振荡还没走完"
                        ),
                    }
                    for key in bad_blocks
                    if key.startswith("bad_steady")
                },
                **{
                    key: {
                        "abs": 5.0e-5, "rel": 0.0,
                        "reason": (
                            "``t = 2 s``上的闭式值，**含没走完的振荡尾巴**。"
                            "实测1%档8.64e-6、0.1%档1.14e-6，取5e-5留6倍余量。"
                            "**判它而不判渐近值，判据紧16—100倍**："
                            "对渐近值1%档差1.39e-4，而那1.30e-4是尾巴不是偏差"
                        ),
                    }
                    for key in bad_blocks
                    if key.startswith("bad_response")
                },
                **{
                    key: {
                        "abs": 0.0, "rel": 1.0e-10,
                        "reason": (
                            "**纯比例增益也把阻尼吃掉了**：−5.000（不控）→"
                            "−4.895（0.1%）→−3.954（1%）。``Kp``抬高``s¹``系数"
                            "而``s²``系数不动 ⟹ ``ω``升而``2ζω``不变 ⟹ ``ζ``降。"
                            "本条记的是这条单调性"
                        ),
                    }
                    for key in bad_blocks
                    if key.startswith("bad_oscillatory")
                },
                **{
                    key: {"abs": 0.0, "rel": 1.0e-15, "reason": "好控制器增益的一个份额"}
                    for key in bad_blocks
                    if key.startswith("bad_proportional")
                },
            },
        },
        {
            "id": "oracle:closed_loop/stability_limit",
            "inputs": {
                "kind": "routh_hurwitz_integral_gain",
                "clutch_lag_s_fast": LAG_FAST_S,
                "clutch_lag_s_nominal": LAG_NOMINAL_S,
                "probe_fractions": [0.9, 1.1],
            },
            "expected": {
                "pure_integral_limit_fast": fast_limit_pure_integral,
                "pure_integral_limit_nominal": nominal_limit,
                #: ``τ = 0``的退化：四阶Routh必须给出三次那条``d·Ka/G``。
                "pure_integral_limit_zero_lag": integral_gain_limit(0.0, 0.0, 0.0),
                "cubic_routh_zero_lag": VELOCITY_DAMPING * STIFFNESS_TIMES_A / LOOP_GAIN,
                #: 它同时等于``2ζω_n·Ka/G``——**界正比于开环阻尼**。
                "twice_zeta_omega_form": (
                    2.0 * DAMPING_RATIO * NATURAL_FREQUENCY * STIFFNESS_TIMES_A / LOOP_GAIN
                ),
                "dominant_real_part_below": max(
                    root.real
                    for root, _ in modes(LAG_FAST_S, 0.0, 0.9 * fast_limit_pure_integral, 0.0)
                ),
                "dominant_real_part_above": max(
                    root.real
                    for root, _ in modes(LAG_FAST_S, 0.0, 1.1 * fast_limit_pure_integral, 0.0)
                ),
                #: 额定档用``fast``那套设计时Routh第一列的值——**负的**。
                "nominal_routh_with_fast_design": (
                    lambda coefficients: coefficients[1] * coefficients[2]
                    - coefficients[0] * coefficients[3]
                )(characteristic_polynomial(LAG_NOMINAL_S, fast_gains[0], 0.0, fast_gains[2])),
            },
            "tolerances": {
                "pure_integral_limit_fast": {
                    "abs": 0.0, "rel": 1.0e-14, "reason": "纯代数",
                },
                "pure_integral_limit_nominal": {
                    "abs": 0.0, "rel": 1.0e-14, "reason": "纯代数",
                },
                "pure_integral_limit_zero_lag": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**零容差，判逐位相等**：四阶Routh在``τ = 0``处必须"
                        "与三次那条``d·Ka/G``给出**同一个浮点数**。"
                        "2026-08-17实测``==``为真。"
                        "'两条是同一条'这句话要么逐位成立，要么就不该说"
                    ),
                },
                "cubic_routh_zero_lag": {
                    "abs": 0.0, "rel": 0.0, "reason": "同上，是被对拍的另一半",
                },
                "twice_zeta_omega_form": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``2ζω_n·Ka/G``。**它说的话比这个数要紧**："
                        "积分增益的界正比于开环阻尼，而开环阻尼是``ζ = 0.0132``"
                    ),
                },
                "dominant_real_part_below": {
                    "abs": 0.0, "rel": 1.0e-10,
                    "reason": "−0.50210，**负的**：0.9倍界上闭环收敛",
                },
                "dominant_real_part_above": {
                    "abs": 0.0, "rel": 1.0e-10,
                    "reason": (
                        "+0.50249，**正的**：1.1倍界上闭环发散。"
                        "**两侧几乎反对称**（−0.5021对+0.5025）说明界确实在中间"
                    ),
                },
                "nominal_routh_with_fast_design": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": (
                        "**−16525.92，负的**：额定离合器上用``fast``那套设计时"
                        "Routh第一列已经翻号，**任何正的积分增益都不稳**。"
                        "实现在这里必须失败关闭而不是返回一个负的界。"
                        "（把``Kd``也按额定档重算会更负，实测−17997.17——"
                        "**两个数都是负的，本条钉的是前一个**）"
                    ),
                },
            },
        },
        {
            "id": "oracle:closed_loop/capstan_at_the_laydown_point",
            "inputs": {
                "kind": "euler_eytelwein_observation_layer",
                "friction_coefficient": FRICTION_COEFFICIENT,
                "wrap_angle_rad": WRAP_ANGLE_RAD,
            },
            "expected": {
                "transfer_ratio": ratio,
                "laydown_excess_fraction": ratio - 1.0,
                "reversed_ratio": math.exp(
                    2.0 * FRICTION_COEFFICIENT * WRAP_ANGLE_RAD
                ),
                "reversed_excess_fraction": math.exp(
                    2.0 * FRICTION_COEFFICIENT * WRAP_ANGLE_RAD
                )
                - 1.0,
                "laydown_tension_at_setpoint_n": SETPOINT_N * ratio,
            },
            "tolerances": {
                "transfer_ratio": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``exp(μθ) = 1.601978``。**它就是plans/13第六节那个60.2%**，"
                        "而`cases/capstan_tension_ratio`验的是同一条式子的离散形"
                    ),
                },
                "laydown_excess_fraction": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "0.601978——闭环稳态下落位点比设定高60.198%",
                },
                "reversed_ratio": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``exp(2μθ) = 2.566332``。**判据要判的是'平方'这件事**，"
                        "所以本条与``transfer_ratio``分开给值：一个凑出来的2.566"
                        "过不了'它必须等于前一条的平方'那道门"
                    ),
                },
                "reversed_excess_fraction": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "1.566332——156.63%对60.20%，**这是平方的代价**",
                },
                "laydown_tension_at_setpoint_n": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "设定20.2778 N时落位点上是32.4845 N。"
                        "**这个数是本案例交给现场的那一个**"
                    ),
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/closed_loop_tension_step",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/closed_loop_tension_step/generate_oracle.py",
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
