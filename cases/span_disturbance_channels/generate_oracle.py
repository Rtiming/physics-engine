#!/usr/bin/env python3
"""两条扰动通道的金标生成器——**闭式解，独立于被验内核**。

本文件只import``math``与`physics_engine.oracles`（清单写入器）。
**不import`transport`、不import`laydown`、不import`disturbance`、
不import任何力学模块**（轴7规则3）。下面每一条式子都是在这里从头写的。

## 一、槽的解析形状：**用切向角当参数**

真实工件的中心线在GCW那边，本仓一份都没有（plans/14第四节）。
所以取一条解析曲线，而且**用切向角``α``当参数**——这一步是本案例的关键设计：

    κ(α) = κ0 / (1 + m·cos α)          ← 曲率沿路径变，``m = 0``退化成圆
    ds/dα = 1/κ = (1 + m·cos α)/κ0

于是弧长与位置**全部是初等函数**：

    s(α) = (α + m·sin α)/κ0
    x(α) = (sin α + m·(α/2 + sin 2α/4))/κ0
    y(α) = (−cos α + m·sin²α/2)/κ0
    t̂(α) = (cos α, sin α, 0)      n̂ = (0,0,1)      ŝ = n̂ × t̂ = (−sin α, cos α, 0)

**用``α``当参数的好处是"槽切向转多快"变成了一个可以直接拧的旋钮**：
机器人让``α(t) = α₀ + Ω·t``，落位点处槽切向就**恰好**以``Ω``匀速转。

## 二、1.3的机制：``σ' = Ω_tan/κ``，而退化档由此**必然**为零

落位点要待在那个固定的入带点上，所以这一瞬要放出的带材长度速率就是弧长速率

    σ'(t) = ds/dα · dα/dt = Ω/κ(α) = (Ω/κ0)·(1 + m·cos α(t))

**平面圆（``m = 0``）时它恰是常数``Ω/κ0``，一点扰动都没有**——
而这与切向转多快无关：``Ω``再大，只要``κ``是常数，送带率就是常数。
**扰动来自曲率在变，不是来自切向在转。**

``m ≠ 0``时它是一条正弦：均值``v₀ = Ω/κ0``、幅值``a = m·v₀``、频率**恰好是``Ω``**。

## 三、位姿：姿态选``Rz(−α)``、平移反解——闭合于是是**构造出来的**

抄`helix_laydown_closure`那条形制（0067）。姿态取绕工件轴转``−α(t)``时，
世界系里的槽切向

    Rz(−α)·(cos α, sin α, 0) = (1, 0, 0)          ← **恒定**

正是"带材无折角地续上槽"那个理想绕线条件（入射角理想值0，见0067第5.1节）。
平移由``x(t) = entry − Rz(−α)·C(σ(t))``反解，闭合因此逐位成立。

**顺带说明一件容易搞混的事**：世界系切向在理想绕线下是**不动**的，
所以"槽切向转多快"这个旋钮取的是**工件系**（材料意义上）的那一条，
它等于``κ·σ' = α' = Ω``。两条都在清单里，第二条的期望值是**零**。

## 四、张力对正弦速度扰动的稳态响应

线性化（记号同0066）：

    δT'' + 2ζω_n·δT' + ω_n²·δT = K·a·[2ζω_n·cos ωt − ω·sin ωt]

**右端是两项不是一项**：速度扰动同时经"稳态张力随线速度走"（``c·v/R²``那一路）
与"长度账的导数"（``ω``那一路）进来。相量解

    X = K·a·(2ζω_n + iω) / ((ω_n² − ω²) + 2iζω_n·ω)
    |X| = K·a·sqrt(4ζ²ω_n² + ω²) / sqrt((ω_n² − ω²)² + 4ζ²ω_n²ω²)

``ω → 0``时``|X|/a → 2ζK/ω_n``，而那**恰好**是稳态关系``dT/dv = c/R²``
（把``ζ``与``ω_n``的定义代进去，``K``整个消掉）。清单里有一条**零容差**门判这个退化。

判据要判的是**幅值**，所以起点直接取受迫解本身（``δT(0) = Re X``、
``δv_放线(0) = a + ω·Im X/K``），**不留瞬态**——理由与
`SpanTransportLoop.at_steady_state`那条"起点必须已经是闭式"同源。
本工况``ζ = 0.0132``，瞬态一旦起来要0.2秒才衰减掉，比整段窗口还长。

## 五、1.4：横向侵入让直线段变成折线，**这是几何不是模型**

    L_path(δ) = sqrt(a² + δ²) + sqrt(b² + δ²)        a + b = L
    p(δ)      = L_path − L                            ← 路径增量
    g(δ)      = δ/sqrt(a²+δ²) + δ/sqrt(b²+δ²)         ← 横向力的几何因子
    F         = T·g(δ)

小角度展开``p ≈ δ²·L/(2ab)``：**二次**，而且**在跨段中点最小**
（``ab``在``a = b``处最大）。清单里两条门判这两件事，
并把精确式与小角度式的差额列出来——**拿小角度式当实现会在δ=4 mm处差1.78e-4相对**。

材料长度是状态、不会瞬变，所以路径一跳张力**当场跳**

    ΔT₀ = EA·p/L_mat                                  ← 精确，不是线性化

此后按二阶系统振铃（初值``δT(0) = ΔT₀``、``δT'(0) = 0``）：

    δT(t) = ΔT₀·e^{−ζω_n t}·[cos ω_d t + (ζ/sqrt(1−ζ²))·sin ω_d t]

**这两条初值与0066那条速度阶跃恰好相反**（那里张力连续而速度差当场跳）。
矩形触碰＝两次阶跃的叠加，所以整条时间历程仍是闭式。

## 六、"撤掉控制器时不回落"那条判据的闭式对手

开环下唯一的阻尼通道是轴承。真实量级``c = 50``给``ζ = 0.0132``，
包络``exp(−ζω_n t)``每个阻尼周期只掉7.9%——**四个周期之后还剩71.8%**。
**那就是"不回落"**：没有控制器时这个尖峰不会自己消失，只是慢慢地振。
"有控制器时回落"那一半归轨道E（决策0070），本案例**没有做**。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/span_disturbance_channels"
ALGORITHM_VERSION = "1.0.0"

#: ``1 N = 1 kg·m/s² = 1000 kg·mm/s²``。**本文件自己写一遍，不从引擎取。**
MM_PER_M = 1000.0

# --- 跨段与放线端：与`cases/free_span_tension_step`**逐条同值**（0062第二节裁决2）---
AXIAL_STIFFNESS_N = 60000.0
GEOMETRIC_LENGTH_MM = 300.0
REEL_RADIUS_MM = 60.0
REEL_INERTIA_KG_MM2 = 5000.0
BRAKE_TORQUE_NMM = 1200.0
#: 轴承粘性阻力矩。**取真实量级那一档**（不是`free_span_tension_step`为了良态调出来的
#: 1000）：本案例要判的正是"开环不回落"，而那件事只有在真实阻尼下才有意义。
BEARING_DAMPING_NMM_S = 50.0

# --- 槽：切向角参数化的平面曲线 ---
#: 基准曲率半径。取60 mm与放线盘同量级；**假设输入**，真实工件最小曲率半径27.6 mm
#: （plans/14第2.2节），本案例的``m = 0.3``档给出42—78 mm，落在真实区间的上沿。
BASE_RADIUS_MM = 60.0
BASE_CURVATURE_PER_MM = 1.0 / BASE_RADIUS_MM
#: 曲率调制深度。``m = 0``是**平面圆退化档**；``m = 0.3``给κ在±30%之间摆。
CURVATURE_MODULATION = 0.3
#: 世界系里那个固定的入带点。``x = 1126.0``抄plans/14第3.3节记的Frame-A原点
#: （三只Ø120导轮共面于``x ≈ 1126``）；另外两个分量是**假设输入**。
ENTRY_POINT_MM = (1126.0, 0.0, 300.0)

# --- 1.3 扫描 ---
#: 落位点处槽切向的转速，rad/s。**这就是那条判据里的旋钮。**
#: 四档全是**假设输入**，且它们同时决定线速度``v₀ = Ω/κ0``（60、120、240、480 mm/s）——
#: **旋钮与工作点是连在一起的，这不是可以分开拧的两件事**，如实写在案例页。
TANGENT_TURN_RATES_RAD_S = (1.0, 2.0, 4.0, 8.0)
SWEEP_DT_S = 1.0e-4
#: 一个整周期多5%。**少于一个整周期时峰峰值系统性偏小**——
#: 实测0.55周期档在Ω=8上偏低10.9%，而那是窗口不是物理。
SWEEP_CYCLES = 1.05
#: 接线档（真的走`laydown`那一档）用的参数。
WIRING_RATE_RAD_S = 2.0
WIRING_ALPHA0_RAD = 0.4
WIRING_STATIONS = 256
WIRING_ALPHA_MAX_RAD = 2.0 * math.pi
WIRING_PROBE_S = 1.0e-4
WIRING_STEPS = 1500
WIRING_PROBE_TIMES_S = (0.02, 0.05, 0.10, 0.14)

# --- 1.4 触碰 ---
TOUCH_LINE_SPEED_MM_S = 20.0
TOUCH_STATION_MM = 150.0
TOUCH_OFFCENTRE_STATION_MM = 75.0
TOUCH_OFFSET_MM = 2.0
TOUCH_OFFSET_SWEEP_MM = (0.5, 1.0, 2.0, 4.0)
TOUCH_START_S = 0.002
TOUCH_END_S = 0.022
TOUCH_DT_S = 2.0e-6
TOUCH_STEPS = 50000
#: 振铃包络要判几个阻尼周期。**四个**：``exp(−ζω_n·4T_d)``还剩71.8%，
#: 那就是"不回落"这句话的量化形式。
RING_PERIODS = (1, 2, 3, 4)


# --------------------------------------------------------------- 槽的闭式 ---


def groove_arc_mm(alpha: float, modulation: float) -> float:
    """``s(α) = (α + m sin α)/κ0``。"""

    return (alpha + modulation * math.sin(alpha)) / BASE_CURVATURE_PER_MM


def groove_curvature_per_mm(alpha: float, modulation: float) -> float:
    """``κ(α) = κ0/(1 + m cos α)``。"""

    return BASE_CURVATURE_PER_MM / (1.0 + modulation * math.cos(alpha))


def required_feed_rate_mm_s(alpha: float, modulation: float, turn_rate: float) -> float:
    """``σ' = Ω/κ(α)``——**1.3那条机制**，直接由上一条取倒数乘``Ω``。"""

    return turn_rate / groove_curvature_per_mm(alpha, modulation)


# ------------------------------------------------------- 跨段与放线端的闭式 ---


def steady_tension_n(speed_mm_s: float) -> float:
    """``T = M/R + c·v/R²``（0066第三节）。"""

    return BRAKE_TORQUE_NMM / REEL_RADIUS_MM + BEARING_DAMPING_NMM_S * speed_mm_s / (
        REEL_RADIUS_MM * REEL_RADIUS_MM
    )


def material_length_mm(tension_n: float) -> float:
    """``L_mat = EA·L_geo/(EA + T)``。"""

    return AXIAL_STIFFNESS_N * GEOMETRIC_LENGTH_MM / (AXIAL_STIFFNESS_N + tension_n)


def span_stiffness_n_per_mm(tension_n: float) -> float:
    """``K = EA·L_geo/L_mat²``。**不是``EA/L_geo``。**"""

    length = material_length_mm(tension_n)
    return AXIAL_STIFFNESS_N * GEOMETRIC_LENGTH_MM / (length * length)


def natural_frequency_rad_s(tension_n: float) -> float:
    """``ω_n = sqrt(1000·R²·K/J)``。那个1000是`MM_PER_M`。"""

    return math.sqrt(
        MM_PER_M
        * REEL_RADIUS_MM
        * REEL_RADIUS_MM
        * span_stiffness_n_per_mm(tension_n)
        / REEL_INERTIA_KG_MM2
    )


def damping_ratio(tension_n: float) -> float:
    """``ζ = (1000·c/J)/(2ω_n)``。"""

    return (MM_PER_M * BEARING_DAMPING_NMM_S / REEL_INERTIA_KG_MM2) / (
        2.0 * natural_frequency_rad_s(tension_n)
    )


def forced_phasor(
    stiffness: float, amplitude: float, forcing: float, natural: float, ratio: float
) -> tuple[float, float]:
    """``X = K·a·(2ζω_n + iω)/((ω_n² − ω²) + 2iζω_n ω)``，返回``(Re X, Im X)``。"""

    num_re = stiffness * amplitude * 2.0 * ratio * natural
    num_im = stiffness * amplitude * forcing
    den_re = natural * natural - forcing * forcing
    den_im = 2.0 * ratio * natural * forcing
    den = den_re * den_re + den_im * den_im
    return ((num_re * den_re + num_im * den_im) / den, (num_im * den_re - num_re * den_im) / den)


def harmonic_amplitude_n(
    stiffness: float, amplitude: float, forcing: float, natural: float, ratio: float
) -> float:
    """``|X|``，**独立于上一条**用模长式子直接算（两条互为对拍）。"""

    return (
        stiffness
        * abs(amplitude)
        * math.sqrt(4.0 * ratio * ratio * natural * natural + forcing * forcing)
        / math.sqrt(
            (natural * natural - forcing * forcing) ** 2
            + 4.0 * ratio * ratio * natural * natural * forcing * forcing
        )
    )


# ---------------------------------------------------------- 触碰的几何闭式 ---


def path_excess_mm(offset_mm: float, station_mm: float) -> float:
    """``sqrt(a²+δ²) + sqrt(b²+δ²) − L``。"""

    first = station_mm
    second = GEOMETRIC_LENGTH_MM - station_mm
    return (
        math.sqrt(first * first + offset_mm * offset_mm)
        + math.sqrt(second * second + offset_mm * offset_mm)
        - GEOMETRIC_LENGTH_MM
    )


def small_angle_path_excess_mm(offset_mm: float, station_mm: float) -> float:
    """``δ²·L/(2ab)``——**小角度式，本案例判的是它与精确式的差**。"""

    first = station_mm
    second = GEOMETRIC_LENGTH_MM - station_mm
    return offset_mm * offset_mm * GEOMETRIC_LENGTH_MM / (2.0 * first * second)


def force_factor(offset_mm: float, station_mm: float) -> float:
    """``g(δ) = δ/sqrt(a²+δ²) + δ/sqrt(b²+δ²)``。"""

    first = station_mm
    second = GEOMETRIC_LENGTH_MM - station_mm
    return offset_mm / math.sqrt(first * first + offset_mm * offset_mm) + offset_mm / math.sqrt(
        second * second + offset_mm * offset_mm
    )


def ring(step_n: float, natural: float, ratio: float, elapsed_s: float) -> float:
    """``ΔT₀·e^{−ζω_n t}·[cos ω_d t + (ζ/sqrt(1−ζ²))·sin ω_d t]``。"""

    root = math.sqrt(1.0 - ratio * ratio)
    damped = natural * root
    return (
        step_n
        * math.exp(-ratio * natural * elapsed_s)
        * (math.cos(damped * elapsed_s) + (ratio / root) * math.sin(damped * elapsed_s))
    )


def touch_tension_n(t_s: float, steady: float, step_n: float, natural: float, ratio: float) -> float:
    """矩形触碰＝**两次阶跃的叠加**（线性系统，所以可以叠）。"""

    value = steady
    if t_s >= TOUCH_START_S:
        value += ring(step_n, natural, ratio, t_s - TOUCH_START_S)
    if t_s >= TOUCH_END_S:
        value -= ring(step_n, natural, ratio, t_s - TOUCH_END_S)
    return value


def _extremum(steady: float, step_n: float, natural: float, ratio: float) -> tuple[float, float]:
    """整段窗口上闭式的峰与谷。粗扫1e-6再在最优点附近细化到1e-9。"""

    horizon = TOUCH_STEPS * TOUCH_DT_S
    coarse = 1.0e-6
    best_t, worst_t = 0.0, 0.0
    best, worst = -math.inf, math.inf
    index = 0
    while True:
        t_s = index * coarse
        if t_s > horizon:
            break
        value = touch_tension_n(t_s, steady, step_n, natural, ratio)
        if value > best:
            best, best_t = value, t_s
        if value < worst:
            worst, worst_t = value, t_s
        index += 1
    for centre, sign in ((best_t, 1.0), (worst_t, -1.0)):
        fine = 1.0e-9
        offset = -1000
        while offset <= 1000:
            t_s = centre + offset * fine
            if 0.0 <= t_s <= horizon:
                value = touch_tension_n(t_s, steady, step_n, natural, ratio)
                if sign > 0.0 and value > best:
                    best = value
                if sign < 0.0 and value < worst:
                    worst = value
            offset += 1
    return (best, worst)


# ------------------------------------------------------------------ 清单 ---


def _sweep_block() -> dict:
    expected: dict[str, float] = {}
    tolerances: dict[str, dict] = {}
    previous = None
    for index, turn_rate in enumerate(TANGENT_TURN_RATES_RAD_S):
        speed = turn_rate / BASE_CURVATURE_PER_MM
        ripple = CURVATURE_MODULATION * speed
        steady = steady_tension_n(speed)
        stiffness = span_stiffness_n_per_mm(steady)
        natural = natural_frequency_rad_s(steady)
        ratio = damping_ratio(steady)
        real, imaginary = forced_phasor(stiffness, ripple, turn_rate, natural, ratio)
        amplitude = harmonic_amplitude_n(stiffness, ripple, turn_rate, natural, ratio)
        expected[f"line_speed_{index}_mm_s"] = speed
        expected[f"steady_tension_{index}_n"] = steady
        expected[f"natural_frequency_{index}_rad_s"] = natural
        expected[f"damping_ratio_{index}"] = ratio
        expected[f"initial_tension_{index}_n"] = steady + real
        expected[f"initial_payout_speed_{index}_mm_s"] = speed + ripple + turn_rate * imaginary / stiffness
        expected[f"tension_amplitude_{index}_n"] = amplitude
        expected[f"steps_{index}"] = float(int(SWEEP_CYCLES * 2.0 * math.pi / turn_rate / SWEEP_DT_S))
        if previous is not None:
            expected[f"amplitude_ratio_{index}"] = amplitude / previous
        previous = amplitude
        for name, reason in (
            (f"line_speed_{index}_mm_s", "``v₀ = Ω/κ0``，一次除法"),
            (f"steady_tension_{index}_n", "``M/R + c·v/R²``，纯算术"),
            (f"natural_frequency_{index}_rad_s", "纯代数，含那个1000"),
            (f"damping_ratio_{index}", "纯代数"),
            (
                f"initial_tension_{index}_n",
                "受迫解在``t = 0``的值。**起点取闭式受迫解而不是稳态**："
                "``ζ = 0.0132``下瞬态要0.2 s才衰减，比整段窗口还长",
            ),
            (
                f"initial_payout_speed_{index}_mm_s",
                "``v₀ + a + ω·Im X/K``。它与上一条一起把回路直接放在受迫轨道上",
            ),
        ):
            tolerances[name] = {"abs": 0.0, "rel": 1.0e-14, "reason": reason}
        tolerances[f"steps_{index}"] = {
            "abs": 0.0,
            "rel": 0.0,
            "reason": "步数是整数，零容差",
        }
        tolerances[f"tension_amplitude_{index}_n"] = {
            "abs": 0.0,
            "rel": 1.0e-3,
            "reason": (
                "**收敛结果**：半隐式Euler的一阶误差。2026-08-17实测四档"
                "−4.95e-6 / −1.92e-5 / −6.94e-5 / −2.11e-4（``dt = 1e-4``），"
                "取最坏档的约5倍余量。**窗口必须≥一个整周期**——"
                "0.55周期档在Ω=8上偏低10.9%，而那是窗口不是物理"
            ),
        }
        if index:
            tolerances[f"amplitude_ratio_{index}"] = {
                "abs": 0.0,
                "rel": 1.0e-14,
                "reason": (
                    "闭式之比，纯代数。**它必须>1**——那就是那条单调判据本体，"
                    "实测2.0295 / 2.1123 / 2.3785"
                ),
            }
    return {
        "id": "oracle:span_disturbance/arm_rate_sweep",
        "inputs": {
            "kind": "sinusoidal_takeup_from_groove_curvature",
            "tangent_turn_rates_rad_s": list(TANGENT_TURN_RATES_RAD_S),
            "base_curvature_per_mm": BASE_CURVATURE_PER_MM,
            "curvature_modulation": CURVATURE_MODULATION,
            "bearing_damping_nmm_s": BEARING_DAMPING_NMM_S,
            "dt_s": SWEEP_DT_S,
            "cycles": SWEEP_CYCLES,
        },
        "expected": expected,
        "tolerances": tolerances,
    }


def _touch_blocks() -> list[dict]:
    steady = steady_tension_n(TOUCH_LINE_SPEED_MM_S)
    length = material_length_mm(steady)
    natural = natural_frequency_rad_s(steady)
    ratio = damping_ratio(steady)
    excess = path_excess_mm(TOUCH_OFFSET_MM, TOUCH_STATION_MM)
    step_n = AXIAL_STIFFNESS_N * excess / length
    damped_period = 2.0 * math.pi / (natural * math.sqrt(1.0 - ratio * ratio))
    peak, trough = _extremum(steady, step_n, natural, ratio)

    geometry_expected: dict[str, float] = {
        "centre_excess_mm": excess,
        "centre_small_angle_excess_mm": small_angle_path_excess_mm(
            TOUCH_OFFSET_MM, TOUCH_STATION_MM
        ),
        "centre_force_factor": force_factor(TOUCH_OFFSET_MM, TOUCH_STATION_MM),
        "offcentre_excess_mm": path_excess_mm(TOUCH_OFFSET_MM, TOUCH_OFFCENTRE_STATION_MM),
        "centre_over_offcentre_excess": excess
        / path_excess_mm(TOUCH_OFFSET_MM, TOUCH_OFFCENTRE_STATION_MM),
        "small_angle_centre_over_offcentre": small_angle_path_excess_mm(
            TOUCH_OFFSET_MM, TOUCH_STATION_MM
        )
        / small_angle_path_excess_mm(TOUCH_OFFSET_MM, TOUCH_OFFCENTRE_STATION_MM),
    }
    geometry_tolerances: dict[str, dict] = {
        "centre_excess_mm": {
            "abs": 0.0,
            "rel": 1.0e-15,
            "reason": (
                "**几何恒等式**（两条斜边减一条底边），双精度往返。"
                "判到1e-15是有意的：小角度式在这一档差**4.444e-5相对**，"
                "拿小角度式当实现会被这条门当场抓住"
            ),
        },
        "centre_small_angle_excess_mm": {
            "abs": 0.0,
            "rel": 1.0e-15,
            "reason": "同上，它是被对比的那一半——**清单里同时给两个数才看得见差额**",
        },
        "centre_force_factor": {"abs": 0.0, "rel": 1.0e-15, "reason": "纯代数"},
        "offcentre_excess_mm": {"abs": 0.0, "rel": 1.0e-15, "reason": "纯代数"},
        "centre_over_offcentre_excess": {
            "abs": 0.0,
            "rel": 1.0e-15,
            "reason": (
                "**这个比值小于1**：跨段中点是路径增量**最小**的地方"
                "（``p ∝ L/(2ab)``而``ab``在中点最大）。"
                "'中间按下去最狠'是直觉，而直觉在这里是反的"
            ),
        },
        "small_angle_centre_over_offcentre": {
            "abs": 0.0,
            "rel": 1.0e-15,
            "reason": "小角度下这个比值恰是``a·b``之比，即``0.75``——两条并排才看得出差额",
        },
    }
    for index, offset in enumerate(TOUCH_OFFSET_SWEEP_MM):
        geometry_expected[f"excess_{index}_mm"] = path_excess_mm(offset, TOUCH_STATION_MM)
        geometry_tolerances[f"excess_{index}_mm"] = {
            "abs": 0.0,
            "rel": 1.0e-15,
            "reason": "四档一起给，判的是**二次律**：δ翻倍路径增量翻四倍",
        }
        if index:
            previous = path_excess_mm(TOUCH_OFFSET_SWEEP_MM[index - 1], TOUCH_STATION_MM)
            geometry_expected[f"excess_ratio_{index}"] = (
                path_excess_mm(offset, TOUCH_STATION_MM) / previous
            )
            geometry_tolerances[f"excess_ratio_{index}"] = {
                "abs": 0.0,
                "rel": 1.0e-15,
                "reason": (
                    "相邻两档之比。**它不写死为4**——精确式比小角度式略小，"
                    "所以比值实测3.99997/3.99987/3.99947，是一条真的会动的判据"
                ),
            }

    ring_expected: dict[str, float] = {"damped_period_s": damped_period}
    ring_tolerances: dict[str, dict] = {
        "damped_period_s": {"abs": 0.0, "rel": 1.0e-14, "reason": "纯代数"}
    }
    for periods in RING_PERIODS:
        ring_expected[f"envelope_after_{periods}_periods"] = math.exp(
            -ratio * natural * periods * damped_period
        )
        ring_tolerances[f"envelope_after_{periods}_periods"] = {
            "abs": 0.0,
            "rel": 1.0e-6,
            "reason": (
                "**这是'撤掉控制器时不回落'那条判据的闭式对手**。"
                "2026-08-17实测四档偏差2.04e-8 / 6.02e-9 / −7.58e-9 / −4.82e-8，"
                "取约20倍余量。四个周期之后还剩**71.8%**——**那就是不回落**"
            ),
        }

    return [
        {
            "id": "oracle:span_disturbance/touch_path_geometry",
            "inputs": {
                "kind": "transverse_intrusion_polyline",
                "span_length_mm": GEOMETRIC_LENGTH_MM,
                "station_from_guide_mm": TOUCH_STATION_MM,
                "offcentre_station_mm": TOUCH_OFFCENTRE_STATION_MM,
                "offset_sweep_mm": list(TOUCH_OFFSET_SWEEP_MM),
                "offset_mm": TOUCH_OFFSET_MM,
            },
            "expected": geometry_expected,
            "tolerances": geometry_tolerances,
        },
        {
            "id": "oracle:span_disturbance/touch_tension_spike",
            "inputs": {
                "kind": "rectangular_path_step_second_order",
                "line_speed_mm_s": TOUCH_LINE_SPEED_MM_S,
                "bearing_damping_nmm_s": BEARING_DAMPING_NMM_S,
                "start_time_s": TOUCH_START_S,
                "end_time_s": TOUCH_END_S,
                "dt_s": TOUCH_DT_S,
                "steps": TOUCH_STEPS,
            },
            "expected": {
                "steady_tension_n": steady,
                "material_length_mm": length,
                "span_stiffness_n_per_mm": span_stiffness_n_per_mm(steady),
                "natural_frequency_rad_s": natural,
                "damping_ratio": ratio,
                "onset_step_n": step_n,
                "peak_tension_n": peak,
                "trough_tension_n": trough,
                "touched_steps": float(int(round((TOUCH_END_S - TOUCH_START_S) / TOUCH_DT_S))),
            },
            "tolerances": {
                "steady_tension_n": {"abs": 0.0, "rel": 1.0e-15, "reason": "纯算术"},
                "material_length_mm": {"abs": 0.0, "rel": 1.0e-15, "reason": "纯代数反解"},
                "span_stiffness_n_per_mm": {"abs": 0.0, "rel": 1.0e-14, "reason": "纯代数"},
                "natural_frequency_rad_s": {"abs": 0.0, "rel": 1.0e-14, "reason": "纯代数"},
                "damping_ratio": {"abs": 0.0, "rel": 1.0e-14, "reason": "纯代数"},
                "onset_step_n": {
                    "abs": 0.0,
                    "rel": 1.0e-12,
                    "reason": (
                        "**这一条是精确的不是线性化的**：材料长度是状态、不会瞬变，"
                        "所以路径一跳张力当场跳``EA·p/L_mat``。"
                        "2026-08-17实测四档``dt``全部给**同一个**偏差1.70e-13相对，"
                        "**它不随步长动**——那正是'这是恒等式不是收敛结果'的指纹"
                    ),
                },
                "peak_tension_n": {
                    "abs": 0.0,
                    "rel": 1.0e-4,
                    "reason": (
                        "整段窗口的峰值。**它由触碰结束那一次反向阶跃与振铃的相位叠加定**，"
                        "不是触碰开始那一跳。实测−2.10e-5 / −2.65e-5 / −2.93e-5 / −3.06e-5"
                        "（``dt``四档），**它有floor不随``dt``趋零**——那是闭式自己的"
                        "线性化偏差（``K``随``L_mat``变约8.8e-5）。取3倍余量"
                    ),
                },
                "trough_tension_n": {
                    "abs": 0.0,
                    "rel": 5.0e-4,
                    "reason": (
                        "谷值14.15 N对稳态20.28 N是**掉30%**——尖峰不是只往上。"
                        "容差比峰值那条松五倍，因为**谷值的线性化floor更高**："
                        "``dt``四档实测9.80e-5 / 1.09e-4 / 1.14e-4，"
                        "同样不随``dt``趋零。**两条floor不一样高这件事要写出来**，"
                        "否则下一个人会以为容差是照着峰值那条抄的"
                    ),
                },
                "touched_steps": {"abs": 0.0, "rel": 0.0, "reason": "整数，零容差"},
            },
        },
        {
            "id": "oracle:span_disturbance/touch_ring_down",
            "inputs": {
                "kind": "open_loop_ring_envelope",
                "bearing_damping_nmm_s": BEARING_DAMPING_NMM_S,
                "periods": list(RING_PERIODS),
            },
            "expected": ring_expected,
            "tolerances": ring_tolerances,
        },
    ]


def main() -> int:
    circle_speed = WIRING_RATE_RAD_S / BASE_CURVATURE_PER_MM
    wiring_probes = {
        f"feed_rate_{index}_mm_s": required_feed_rate_mm_s(
            WIRING_ALPHA0_RAD + WIRING_RATE_RAD_S * t_s, CURVATURE_MODULATION, WIRING_RATE_RAD_S
        )
        for index, t_s in enumerate(WIRING_PROBE_TIMES_S)
    }
    quasi_static = steady_tension_n(TOUCH_LINE_SPEED_MM_S)
    quasi_stiffness = span_stiffness_n_per_mm(quasi_static)
    quasi_natural = natural_frequency_rad_s(quasi_static)
    quasi_ratio = damping_ratio(quasi_static)

    oracles = [
        {
            "id": "oracle:span_disturbance/tangent_turn_feed_rate",
            "inputs": {
                "kind": "feed_rate_is_turn_rate_over_curvature",
                "base_curvature_per_mm": BASE_CURVATURE_PER_MM,
                "curvature_modulation": CURVATURE_MODULATION,
                "tangent_turn_rate_rad_s": WIRING_RATE_RAD_S,
                "alpha_origin_rad": WIRING_ALPHA0_RAD,
                "probe_times_s": list(WIRING_PROBE_TIMES_S),
                "stations": WIRING_STATIONS,
                "rate_probe_step_s": WIRING_PROBE_S,
                "dt_s": SWEEP_DT_S,
                "steps": WIRING_STEPS,
            },
            "expected": {
                **wiring_probes,
                "mean_feed_rate_mm_s": circle_speed,
                "world_tangent_turn_rate_rad_s": 0.0,
                "groove_tangent_turn_rate_rad_s": WIRING_RATE_RAD_S,
                "arc_at_alpha_max_mm": groove_arc_mm(WIRING_ALPHA_MAX_RAD, CURVATURE_MODULATION),
                "curvature_at_alpha0_per_mm": groove_curvature_per_mm(
                    WIRING_ALPHA0_RAD, CURVATURE_MODULATION
                ),
            },
            "tolerances": {
                **{
                    name: {
                        "abs": 0.0,
                        "rel": 1.0e-6,
                        "reason": (
                            "``σ' = Ω/κ``对`laydown`在离散中心线上差分出来的所需送带率。"
                            "2026-08-17实测最坏**6.04e-8**相对（256站点/整圈、"
                            "``hermite_tangent``位置语义、中心差分步长1e-4），取约16倍余量。"
                            "**这一条判的是接线不是物理**：数对上了才说明"
                            "`laydown`的落位点几何真的接进了`transport`的收线端"
                        ),
                    }
                    for name in wiring_probes
                },
                "mean_feed_rate_mm_s": {
                    "abs": 0.0,
                    "rel": 1.0e-6,
                    "reason": "``Ω/κ0``。它同时是平面圆退化档的那个恒定送带率",
                },
                "world_tangent_turn_rate_rad_s": {
                    "abs": 1.0e-3,
                    "rel": 0.0,
                    "reason": (
                        "**期望值是零**：姿态取``Rz(−α)``时世界系槽切向恒为``(1,0,0)``，"
                        "那正是`laydown`把入射角理想值定在0的那个条件（0067第5.1节）。"
                        "实测4.66e-4 rad/s（对旋钮值2.0是2.3e-4相对），"
                        "残差来自帧插值的二阶截断。**判绝对不判相对：期望值是0**"
                    ),
                },
                "groove_tangent_turn_rate_rad_s": {
                    "abs": 0.0,
                    "rel": 1.0e-3,
                    "reason": (
                        "**旋钮本体**：工件系槽切向的转速必须等于``α' = Ω``。"
                        "实测2.000466对2.0是**2.33e-4**相对，"
                        "残差是``reorthonormalised_linear``帧语义的二阶截断"
                        "（0067第四节量过那一条是一阶对二阶）。取约4倍余量"
                    ),
                },
                "arc_at_alpha_max_mm": {"abs": 0.0, "rel": 1.0e-15, "reason": "纯代数"},
                "curvature_at_alpha0_per_mm": {"abs": 0.0, "rel": 1.0e-15, "reason": "纯代数"},
            },
        },
        {
            "id": "oracle:span_disturbance/planar_circle_degenerate",
            "inputs": {
                "kind": "constant_curvature_gives_no_disturbance",
                "base_curvature_per_mm": BASE_CURVATURE_PER_MM,
                "curvature_modulation": 0.0,
                "tangent_turn_rate_rad_s": WIRING_RATE_RAD_S,
                "dt_s": SWEEP_DT_S,
                "steps": 20000,
            },
            "expected": {
                "feed_rate_mm_s": circle_speed,
                "feed_rate_spread_mm_s": 0.0,
                "takeup_variance_mm_s": 0.0,
                "tension_excursion_zero_damping_n": 0.0,
                "tension_excursion_nominal_damping_n": 0.0,
            },
            "tolerances": {
                "feed_rate_mm_s": {
                    "abs": 0.0,
                    "rel": 1.0e-9,
                    "reason": (
                        "平面圆上``κ``恒定，``σ' = Ω/κ0``**恒定**。"
                        "经`laydown`的离散中心线走一趟实测散布1.63e-7 mm/s对120 mm/s，"
                        "即1.36e-9相对——取1e-9是把那条散布单独判在下一行"
                    ),
                },
                "feed_rate_spread_mm_s": {
                    "abs": 2.0e-6,
                    "rel": 0.0,
                    "reason": (
                        "**期望值是零**：解析上送带率恒定。经离散中心线实测散布"
                        "**1.63e-7 mm/s**（256站点/整圈），取约12倍余量。"
                        "**判绝对不判相对**——期望值是0"
                    ),
                },
                "takeup_variance_mm_s": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": (
                        "**零容差，而且这一条真的逐位成立**：平面圆上所需送带率是常数，"
                        "所以扰动的**输入**方差恰为零。它与下面两条分开列，"
                        "是因为'输入没有扰动'与'输出一个ulp都不动'是两件事——"
                        "**后者做不到，而我一开始把它写成了零容差**（见案例页第四节）"
                    ),
                },
                "tension_excursion_zero_damping_n": {
                    "abs": 2.0e-10,
                    "rel": 0.0,
                    "reason": (
                        "**期望值是零，而实测不是逐位零**。本条最初写的是零容差，"
                        "**实测当场否掉**：``c = 0``档20000步（``dt = 1e-4``，共2秒）"
                        "峰谷差**2.2752e-11 N**。病根不在扰动而在起点——"
                        "``at_steady_state``那一趟``T → L_mat → T``的往返有几个ulp的舍入"
                        "（0066实测四档最坏3.908e-12 N），于是``T·R − M ≠ 0``逐位，"
                        "而``c = 0``**没有任何阻尼**把它吃掉，20000步就把它积成了这个数。"
                        "取2e-10留约9倍余量。**判绝对不判相对：期望值是0**"
                    ),
                },
                "tension_excursion_nominal_damping_n": {
                    "abs": 1.0e-10,
                    "rel": 0.0,
                    "reason": (
                        "同一件事的有阻尼档，实测**1.1376e-11 N**——**比无阻尼档还小一半**，"
                        "因为轴承把那几个ulp的激励吃掉了一部分。取1e-10留约9倍余量。"
                        "**两档并排列，是为了让'零容差做不到'这件事有两个独立的数支撑**"
                    ),
                },
            },
        },
        {
            "id": "oracle:span_disturbance/quasi_static_limit",
            "inputs": {
                "kind": "zero_frequency_limit_is_the_steady_state_relation",
                "line_speed_mm_s": TOUCH_LINE_SPEED_MM_S,
                "bearing_damping_nmm_s": BEARING_DAMPING_NMM_S,
            },
            "expected": {
                "harmonic_gain_at_zero_n_per_mm_s": harmonic_amplitude_n(
                    quasi_stiffness, 1.0, 0.0, quasi_natural, quasi_ratio
                ),
                "two_zeta_k_over_omega_n": 2.0 * quasi_ratio * quasi_stiffness / quasi_natural,
                "damping_over_radius_squared": BEARING_DAMPING_NMM_S
                / (REEL_RADIUS_MM * REEL_RADIUS_MM),
            },
            "tolerances": {
                "harmonic_gain_at_zero_n_per_mm_s": {
                    "abs": 0.0,
                    "rel": 4.0e-16,
                    "reason": (
                        "``ω → 0``时正弦响应的增益必须退回稳态关系``dT/dv = c/R²``。"
                        "两条路径（模长式 对 ``2ζK/ω_n``）实测差**1 ulp**"
                        "（2.22e-16相对），取2倍。**这一条是`SpoolTension`那条"
                        "'旧模型是本模型的稳态特例'在频域的同一件事**"
                    ),
                },
                "two_zeta_k_over_omega_n": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": (
                        "**零容差，判逐位相等**：``2ζK/ω_n``把``ζ``与``ω_n``的定义代进去，"
                        "``K``整个消掉，剩下的恰是``c/R²``。2026-08-17实测两边都是"
                        "``0.013888888888888888``——**这句话要么逐位成立，要么就不该说**"
                    ),
                },
                "damping_over_radius_squared": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "同上，是被对拍的另一半",
                },
            },
        },
        _sweep_block(),
        *_touch_blocks(),
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/span_disturbance_channels",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/span_disturbance_channels/generate_oracle.py",
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
