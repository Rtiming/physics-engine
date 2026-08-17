"""张力闭环对线速度阶跃的conformance门（案例`cases/closed_loop_tension_step`）。

守plans/15第三节阶段一的1.2与1.5，决策0070。

## 这条案例存在的理由，一句话

`free_span_tension_step`证明了**扰动是真的**（一次10%线速度阶跃让张力摆1.05 N
且12.08个周期不衰减）。本案例把控制器接上去，回答**那个扰动压不压得住**——
而答案取决于一个无量纲数：``ω_n·τ = 18.98``，**执行器比对象慢19倍**。

## 三个档位，各自回答不同的问题

| 档 | ``τ`` | 控制器 | 它回答什么 |
|---|---|---|---|
| ``open`` | 50 ms | 全零增益 | 对象自己。**必须与`free_span`那条独立闭式对上** |
| ``fast`` | **0.5 ms** | PID | **环写对了没有**。这一档``τ``不是一台机器 |
| ``nominal`` | 50 ms | 纯积分 | 真机那一档。**闭环把扰动响应变坏了** |

``fast``档存在的理由要写清楚：没有它，"压不下去"这个结论分不清是
**环写错了**还是**执行器太慢了**。有了它，两件事各判各的。

## 判据取三个量，因为它们红了说明的事不一样

**峰值**守瞬态幅度、**ISE**守整段能量、**稳态偏移**守直流。
``nominal``档正是那个只判一个量会被骗过去的例子：
峰值×1.0727而ISE×1.3882，**两个数一起才说得清"闭环让它更坏"**。

## 与`free_span_tension_step`的对拍差4.6e-7，而差在哪里是查得出来的

两条案例的闭式各写各的。开环峰值实测差**4.60e-7**相对——
`free_span`把线性化参考点取在**阶跃后**（``v = 22``），本案例取在**阶跃前**
（``v = 20``）。``K ∝ (EA+T)²``、峰值``∝ √K``，推出4.63e-7，**与实测对上**。
**"线性化在哪一点"是一个选择不是一个细节**，本文件有一条门判这个差。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.drives import MagneticParticleClutch, PidController
from physics_engine.oracles import load_manifest
from physics_engine.tension_control import (
    CapstanSpan,
    ClosedTensionLoop,
    TensionControlError,
    closed_loop_characteristic_polynomial,
    integral_gain_stability_limit,
)
from physics_engine.transport import FreeSpan, PayoutReel

CASES = Path(__file__).resolve().parents[2] / "cases"
MANIFEST = load_manifest(CASES / "closed_loop_tension_step" / "oracle.json")
FREE_SPAN_MANIFEST = load_manifest(CASES / "free_span_tension_step" / "oracle.json")

#: 全部**假设输入**，逐条出处见案例页第一节与金标生成器。
AXIAL_STIFFNESS_N = 60000.0
GEOMETRIC_LENGTH_MM = 300.0
REEL_RADIUS_MM = 60.0
REEL_INERTIA_KG_MM2 = 5000.0
BEARING_DAMPING_NMM_S = 50.0
BRAKE_TORQUE_NMM = 1200.0
LINE_SPEED_MM_S = 20.0
STEP_MM_S = 2.0
TORQUE_PER_AMPERE_NMM = 23256.0
RATED_TORQUE_NMM = 50000.0

ISE_HORIZON_S = 0.2
PLANT_DT_S = 1.0e-6
#: 稳态误差要跑到振荡沉下去之后：开环时间常数0.2 s，取2.0 s ⟹ 残余``e^{−10}``。
#: 那一档换成``dt = 1e-5``——**稳态判的是不动点，而离散不动点＝连续不动点**
#: （0066第89行），所以粗十倍不影响它。
STEADY_DT_S = 1.0e-5
STEADY_HORIZON_S = 2.0

SPAN = FreeSpan(
    span_id="span/free",
    geometric_length_mm=GEOMETRIC_LENGTH_MM,
    axial_stiffness_n=AXIAL_STIFFNESS_N,
)
REEL = PayoutReel(
    reel_id="reel/payout",
    radius_mm=REEL_RADIUS_MM,
    inertia_kg_mm2=REEL_INERTIA_KG_MM2,
    bearing_damping_nmm_s=BEARING_DAMPING_NMM_S,
)


def _oracle(oracle_id: str):
    for entry in MANIFEST.oracles:
        if entry.id == oracle_id:
            return entry
    raise AssertionError(f"清单里没有{oracle_id}")


def _clutch(lag_s: float) -> MagneticParticleClutch:
    return MagneticParticleClutch(
        torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
        rated_torque_nmm=RATED_TORQUE_NMM,
        lag_s=lag_s,
    )


def _loop(
    *,
    lag_s: float,
    proportional: float,
    integral_gain: float,
    derivative: float,
    dt_s: float = PLANT_DT_S,
    decimation: int = 1,
    capstan: CapstanSpan | None = None,
) -> ClosedTensionLoop:
    """从闭式稳态起手——**判据要判的东西不该被起点的瞬态污染**。"""

    return ClosedTensionLoop.at_steady_state(
        span=SPAN,
        reel=REEL,
        clutch=_clutch(lag_s),
        controller=PidController(
            proportional=proportional,
            integral_gain=integral_gain,
            derivative=derivative,
            integral_limit=1.0e6,
        ),
        capstan=capstan,
        sensor=None,
        plant_dt_s=dt_s,
        control_decimation=decimation,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        line_speed_mm_s=LINE_SPEED_MM_S,
        delay_line=None,
        forbid_slack=True,
    )


def _metrics(samples, setpoint_n: float, dt_s: float) -> dict[str, float]:
    """峰值／峰值时刻／ISE。**三条并判**，理由见模块docstring。"""

    deviations = [sample.tension_n - setpoint_n for sample in samples]
    peak = max(deviations)
    index = deviations.index(peak)
    #: 梯形积分。``dt = 1e-6``对最快的闭环模态（744 rad/s）是7400点每周期，
    #: 积分误差``O(dt²)``远在判据分辨率之下。
    energy = 0.0
    for left, right in zip(deviations, deviations[1:], strict=False):
        energy += 0.5 * (left * left + right * right) * dt_s
    return {
        "peak": peak,
        "peak_time_s": samples[index].time_s,
        "ise": energy,
        "final": deviations[-1],
    }


@pytest.fixture(scope="module")
def bands():
    """三档各跑一次0.2秒，峰值／ISE／压制比三组门共用。"""

    runs = {}
    for label in ("open_loop_step", "fast_band_step", "nominal_band_step"):
        entry = _oracle(f"oracle:closed_loop/{label}")
        loop = _loop(
            lag_s=entry.inputs["clutch_lag_s"],
            proportional=entry.inputs["proportional_a_per_n"],
            integral_gain=entry.inputs["integral_gain_a_per_n_s"],
            derivative=entry.inputs["derivative_a_s_per_n"],
        )
        _, samples = loop.run(
            int(round(ISE_HORIZON_S / PLANT_DT_S)),
            takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S,
        )
        runs[label] = _metrics(samples, loop.setpoint_n, PLANT_DT_S)
    return runs


# ---------------------------------------------------------------------------
# 工作点与两条通道的权限——**裁决A的量**
# ---------------------------------------------------------------------------


def test_the_brake_channel_has_seventy_two_times_the_steady_authority():
    """**裁决A（控制器接制动力矩）的判据本体**。

    稳态是``T = M/R + c·v/R²``，于是

        ∂T/∂M = 1/R      ← **与``c``无关**
        ∂T/∂v = c/R²     ← **正比于``c``**，而``c``一条实测都没有

    按各自额定量的1%算：制动力矩给``0.2 N``、收线速度给``0.0027778 N``——
    **72倍，而且是整数72**（``(M₀/R)/(c·v/R²) = M₀·R/(c·v) = 1200·60/(50·20)``）。

    **最硬的半条是零容差那一行**：``c = 0``时收线速度通道的稳态权限**恰为零**。
    一条把全部权限建在唯一一个没有实测的参数上的回路，
    在那个参数为零时什么也不是。
    """

    entry = _oracle("oracle:closed_loop/plant_linearisation")
    brake = 0.01 * BRAKE_TORQUE_NMM / REEL_RADIUS_MM
    takeup = (
        0.01 * BEARING_DAMPING_NMM_S * LINE_SPEED_MM_S / (REEL_RADIUS_MM * REEL_RADIUS_MM)
    )
    assert brake == pytest.approx(entry.expected["brake_authority_n_per_percent"], rel=1e-15)
    assert takeup == pytest.approx(
        entry.expected["takeup_authority_n_per_percent"], rel=1e-15
    )
    assert brake / takeup == pytest.approx(entry.expected["authority_ratio"], rel=1e-14)

    #: **零轴承阻力矩时收线速度通道的权限恰为零**——零容差。
    zero_damping = 0.01 * 0.0 * LINE_SPEED_MM_S / (REEL_RADIUS_MM * REEL_RADIUS_MM)
    assert zero_damping == entry.expected["takeup_authority_at_zero_damping_n_per_percent"]
    assert zero_damping == 0.0, (
        "``c = 0``时收线速度对张力仍有稳态权限——那说明``T = M/R + c·v/R²``"
        "这条稳态被算错了，裁决A的理由也就不成立"
    )

    #: 代价同批判：执行器比对象慢多少。
    assert LINE_SPEED_MM_S  # 工作点在场，避免这条门看起来与工况无关
    shortfall = entry.expected["nominal_bandwidth_shortfall"]
    assert shortfall > 10.0, (
        f"额定离合器与跨段谐振的带宽比只有{shortfall!r} —— "
        "本案例'压不下去'那个结论建立在这个数上，它变了结论要重写"
    )


def test_the_closed_loop_characteristic_polynomial_matches_the_hand_derivation():
    """特征多项式的五个系数逐条对手推式（模块docstring第四节）。

    **这一条不跑推进**：它判的是闭式本身。红了说明消元那一步写错了，
    而那会让下面每一条对拍同时错——**先判分母，再判用分母算出来的东西**。
    """

    entry = _oracle("oracle:closed_loop/plant_linearisation")
    stiffness = entry.expected["span_stiffness_n_per_mm"]
    damping = 1000.0 * BEARING_DAMPING_NMM_S / REEL_INERTIA_KG_MM2
    stiffness_times_a = stiffness * (1000.0 * REEL_RADIUS_MM**2 / REEL_INERTIA_KG_MM2)
    loop_gain = (
        stiffness * (1000.0 * REEL_RADIUS_MM / REEL_INERTIA_KG_MM2) * TORQUE_PER_AMPERE_NMM
    )
    assert loop_gain == pytest.approx(entry.expected["loop_gain"], rel=1e-14)
    assert math.sqrt(stiffness_times_a) == pytest.approx(
        entry.expected["natural_frequency_rad_s"], rel=1e-14
    )

    lag, kp, ki, kd = 0.05, 0.003, 1.5, 2.0e-5
    coefficients = closed_loop_characteristic_polynomial(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
        bearing_damping_nmm_s=BEARING_DAMPING_NMM_S,
        torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
        clutch_lag_s=lag,
        proportional=kp,
        integral_gain=ki,
        derivative=kd,
    )
    expected = (
        lag,
        1.0 + damping * lag,
        damping + stiffness_times_a * lag + loop_gain * kd,
        stiffness_times_a + loop_gain * kp,
        loop_gain * ki,
    )
    for got, want in zip(coefficients, expected, strict=True):
        assert got == pytest.approx(want, rel=1e-15)

    #: ``Kd``只出现在``s²``系数上（那是``2ζω``那一项），``Ki``只出现在常数项上。
    #: **这就是"能加阻尼的是``Kd``不是``Ki``"的结构证明。**
    without_derivative = closed_loop_characteristic_polynomial(
        span_stiffness_n_per_mm=stiffness, radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2, bearing_damping_nmm_s=BEARING_DAMPING_NMM_S,
        torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM, clutch_lag_s=lag,
        proportional=kp, integral_gain=ki, derivative=0.0,
    )
    assert without_derivative[2] < coefficients[2]
    assert without_derivative[:2] == coefficients[:2]
    assert without_derivative[3:] == coefficients[3:]


# ---------------------------------------------------------------------------
# 三档阶跃响应对闭式
# ---------------------------------------------------------------------------


@pytest.mark.batch
@pytest.mark.parametrize(
    "label", ("open_loop_step", "fast_band_step", "nominal_band_step")
)
def test_the_band_step_response_matches_the_fourth_order_closed_form(bands, label):
    """峰值、峰值时刻、ISE**三条并判**，对四阶闭环的精确留数解。

    2026-08-17实测（``dt = 1e-6``、``Δv = 2 mm/s``）：

    | 档 | 峰值实测 | 峰值闭式 | 相对 | ISE相对 |
    |---|---|---|---|---|
    | ``open`` | 1.0604629 | 1.0604531 | 9.26e-6 | 8.07e-6 |
    | ``fast`` | 0.3376856 | 0.3374866 | 5.90e-4 | 6.07e-4 |
    | ``nominal`` | 1.1375140 | 1.1375052 | 7.74e-6 | 1.37e-5 |

    **``fast``档差两个数量级不是它更差**：那一档的闭环极点在470/s上，
    同一个``dt``对更快的动态就是更粗的网格。
    """

    entry = _oracle(f"oracle:closed_loop/{label}")
    measured = bands[label]
    assert measured["peak"] == pytest.approx(
        entry.expected["peak_excursion_n"],
        rel=entry.tolerances["peak_excursion_n"].rel_tol,
    )
    assert measured["peak_time_s"] == pytest.approx(
        entry.expected["peak_time_s"], rel=entry.tolerances["peak_time_s"].rel_tol
    )
    assert measured["ise"] == pytest.approx(
        entry.expected["integral_squared_error_n2_s"],
        rel=entry.tolerances["integral_squared_error_n2_s"].rel_tol,
    )


@pytest.mark.batch
def test_the_open_loop_band_reproduces_the_free_span_case_to_five_significant_digits(
    bands,
):
    """**跨案例对拍**：两条案例的闭式各写各的，开环那档必须对上。

    2026-08-17实测：峰值差**4.60e-7**相对、峰值时刻差**4.54e-7**。
    **不是逐位，而差在哪里是查得出来的**——`free_span`把线性化参考点取在
    阶跃后的稳态（``v = 22``），本案例取在阶跃前的工作点（``v = 20``）：

        ΔT_ss = 0.0278 N ⟹ ΔK/K = 2·0.0278/60020 = 9.26e-7
        峰值 ∝ K·Δv/ω_n ∝ √K        ⟹ 4.63e-7      ← **与实测4.60e-7对上**

    **这条门判的是那个差本身**：容差取1e-6（比推出来的4.63e-7大一倍多），
    差到1e-5说明两条案例开始各说各的了，差到1e-9说明有人把两边接成了同一条路。

    稳态偏移那一条差**2.85e-14**，病根不同：`free_span`算``T*(22) − T*(20)``
    （两个20.28 N相减，**灾难性抵消**），本案例直接算``Δv·c/R²``。
    """

    mine = _oracle("oracle:closed_loop/open_loop_step")
    theirs = None
    for entry in FREE_SPAN_MANIFEST.oracles:
        if entry.id == "oracle:free_span/velocity_step_nominal":
            theirs = entry
    assert theirs is not None, "`free_span_tension_step`的额定档清单不见了"

    gap = abs(
        theirs.expected["peak_excursion_n"] / mine.expected["peak_excursion_n"] - 1.0
    )
    assert gap < 1.0e-6, (
        f"两条案例的开环峰值差{gap!r} —— 大于线性化参考点那条解释能给的4.63e-7，"
        "说明差的不止是参考点"
    )
    assert gap > 1.0e-8, (
        f"两条案例的开环峰值只差{gap!r} —— **小得可疑**：两边的参考点确实不同，"
        "差到这个量级说明有人把两条闭式接成了同一条路，跨案例对拍就失去意义"
    )
    time_gap = abs(theirs.expected["peak_time_s"] / mine.expected["peak_time_s"] - 1.0)
    assert time_gap < 1.0e-6

    #: 稳态偏移：同一个量的两种算法，差的是浮点不是物理。
    offset_gap = abs(
        theirs.expected["steady_change_n"] / mine.expected["steady_offset_n"] - 1.0
    )
    assert offset_gap < 1.0e-12
    assert theirs.expected["steady_change_n"] != mine.expected["steady_offset_n"], (
        "两边逐位相同了——那说明其中一边改了算法，"
        "而'差2.85e-14是灾难性抵消'这个解释要重写"
    )

    #: 而实测的开环峰值必须同时落在两边的容差里。
    assert bands["open_loop_step"]["peak"] == pytest.approx(
        theirs.expected["peak_excursion_n"], rel=1.0e-5
    )


# ---------------------------------------------------------------------------
# 压制比：本案例最要紧的两个不等号
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_fast_band_suppresses_and_the_nominal_band_makes_it_worse(bands):
    """**闭环压下去了多少，以及在真机那一档上它压没压**。

    2026-08-17实测（闭式／实测并列）：

    | 档 | 峰值比 | ISE比 |
    |---|---|---|
    | ``fast`` | **0.31825** | **6.0709e-3**（165倍） |
    | ``nominal`` | **1.07266** | **1.38817** |

    ``fast``档：峰值压到0.318倍而ISE压到六百分之一——
    **压下去的主要不是峰高，是它不再一直响**（振荡实部从−5.000到−470.709，
    衰减时间常数从0.2 s到2.1 ms）。

    ``nominal``档：**两个数都大于1**。闭式早就说了为什么——积分项只出现在
    特征多项式的常数项上，它把Routh裕度吃掉，振荡模态的实部从−5.000
    退到−4.579。**这不是实现的毛病，是闭式预言的，而实测把它兑现了。**

    ``nominal``档那两个"大于1"是**零容差**判的（判的是不等号不是数值）。
    """

    entry = _oracle("oracle:closed_loop/suppression_verdict")
    open_metrics = bands["open_loop_step"]
    unity = entry.expected["nominal_makes_it_worse"]
    assert unity == 1.0

    fast_peak_ratio = bands["fast_band_step"]["peak"] / open_metrics["peak"]
    fast_ise_ratio = bands["fast_band_step"]["ise"] / open_metrics["ise"]
    assert fast_peak_ratio == pytest.approx(
        entry.expected["fast_peak_ratio"],
        rel=entry.tolerances["fast_peak_ratio"].rel_tol,
    )
    assert fast_ise_ratio == pytest.approx(
        entry.expected["fast_ise_ratio"], rel=entry.tolerances["fast_ise_ratio"].rel_tol
    )
    assert fast_peak_ratio < unity and fast_ise_ratio < unity, (
        "快档闭环没有把扰动压下去——那说明环接反了或增益符号错了"
    )
    #: **"显著"要判成一个数**，否则"压下去了"是一句可以随便说的话。
    assert fast_ise_ratio < 0.01, (
        f"快档ISE只压到{fast_ise_ratio!r}倍 —— 本案例说的'显著压下去'是"
        "两个数量级，压不到就不该那么说"
    )

    nominal_peak_ratio = bands["nominal_band_step"]["peak"] / open_metrics["peak"]
    nominal_ise_ratio = bands["nominal_band_step"]["ise"] / open_metrics["ise"]
    assert nominal_peak_ratio == pytest.approx(
        entry.expected["nominal_peak_ratio"],
        rel=entry.tolerances["nominal_peak_ratio"].rel_tol,
    )
    assert nominal_ise_ratio == pytest.approx(
        entry.expected["nominal_ise_ratio"],
        rel=entry.tolerances["nominal_ise_ratio"].rel_tol,
    )
    #: **零容差的不等号**：真机那一档上闭环让扰动响应更坏。
    assert nominal_peak_ratio > unity, (
        f"额定档峰值比{nominal_peak_ratio!r}不大于1 —— 本案例最要紧的那个结论"
        "（这条通道够不着跨段谐振）不成立了，决策0070第一节要重写"
    )
    assert nominal_ise_ratio > unity


# ---------------------------------------------------------------------------
# 1.1的验收：换得动才叫可插拔
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_a_deliberately_bad_controller_makes_both_the_overshoot_and_the_offset_worse(
    bands,
):
    """**plans/15第1.1条的验收本体**：同一算例，换一个故意很差的控制器。

    坏控制器＝**纯比例、增益极小**（好控制器``Kp``的1%与0.1%，``Ki = Kd = 0``）。

    2026-08-17实测（``fast``档、``Δv = 2 mm/s``）：

    | 控制器 | 峰值 | 稳态偏移（``t = 2 s``） |
    |---|---|---|
    | 好（PID） | **0.3376856** | **+1.60e-10** |
    | 坏（纯P 1%） | 1.0487479（**×3.11**） | +2.6830e-2 |
    | 坏（纯P 0.1%） | 1.0592700（**×3.14**） | +2.7693e-2 |
    | 完全不控 | 1.0604629 | +2.7734e-2 |

    **两条都变坏**：峰值变坏3.1倍、稳态偏移变坏**1.7e8**倍
    （从1.6e-10到2.77e-2）。

    坏控制器的稳态偏移有闭式：``Δv·K·d/(Ka + G·Kp)``。增益极小时它几乎就是
    完全不控那个数（``Δv·c/R² = 0.0277778``）——**这正是"增益极小"的含义**，
    也是这条判据要的：一个"能传进去不报错"的控制器骗不过它。

    **只断言"能传进去"不算可插拔**。本门的分辨力在于：它跑的是同一条算例、
    同一个对象、同一个执行器，**只换控制器**，而两个指标都必须朝坏的方向动。
    """

    entry = _oracle("oracle:closed_loop/pluggability")
    lag_s = entry.inputs["clutch_lag_s"]
    good_peak = bands["fast_band_step"]["peak"]
    assert good_peak == pytest.approx(
        entry.expected["good_peak_excursion_n"],
        rel=entry.tolerances["good_peak_excursion_n"].rel_tol,
    )

    #: 好控制器的稳态偏移：跑到振荡沉下去，必须落在零上。
    good_loop = _loop(
        lag_s=lag_s,
        proportional=_oracle("oracle:closed_loop/fast_band_step").inputs[
            "proportional_a_per_n"
        ],
        integral_gain=_oracle("oracle:closed_loop/fast_band_step").inputs[
            "integral_gain_a_per_n_s"
        ],
        derivative=_oracle("oracle:closed_loop/fast_band_step").inputs[
            "derivative_a_s_per_n"
        ],
        dt_s=STEADY_DT_S,
    )
    good_final, _ = good_loop.run(
        int(round(STEADY_HORIZON_S / STEADY_DT_S)),
        takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S,
    )
    good_offset = good_final.tension_n - good_loop.setpoint_n
    assert good_offset == pytest.approx(
        entry.expected["good_steady_offset_n"],
        abs=entry.tolerances["good_steady_offset_n"].abs_tol,
    )

    for fraction in entry.inputs["bad_gain_fractions"]:
        key = f"{fraction:g}".replace(".", "p")
        proportional = entry.expected[f"bad_proportional_a_per_n_{key}"]

        bad_loop = _loop(
            lag_s=lag_s, proportional=proportional, integral_gain=0.0, derivative=0.0
        )
        _, bad_samples = bad_loop.run(
            int(round(ISE_HORIZON_S / PLANT_DT_S)),
            takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S,
        )
        bad = _metrics(bad_samples, bad_loop.setpoint_n, PLANT_DT_S)
        assert bad["peak"] == pytest.approx(
            entry.expected[f"bad_peak_excursion_n_{key}"],
            rel=entry.tolerances[f"bad_peak_excursion_n_{key}"].rel_tol,
        )
        #: **超调变坏**——不是"没有变好"，是严格变坏，而且是三倍。
        assert bad["peak"] > 3.0 * good_peak, (
            f"增益{fraction}的纯比例控制器峰值{bad['peak']!r}没有比好控制器"
            f"{good_peak!r}坏三倍 —— 换控制器换不动，可插拔就是句空话"
        )

        bad_steady_loop = _loop(
            lag_s=lag_s,
            proportional=proportional,
            integral_gain=0.0,
            derivative=0.0,
            dt_s=STEADY_DT_S,
        )
        bad_final, _ = bad_steady_loop.run(
            int(round(STEADY_HORIZON_S / STEADY_DT_S)),
            takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S,
        )
        bad_offset = bad_final.tension_n - bad_steady_loop.setpoint_n
        #: **对拍的是``t = 2 s``上的闭式值，不是渐近值**：那一刻振荡还没走完
        #: （比例增益把振荡实部从−5.000推到−3.954，``e^{σt} = 3.68e-4``），
        #: 而那1.30e-4是尾巴不是偏差。判含尾巴的那一条，判据紧16—100倍：
        #: 实测1%档8.64e-6、0.1%档1.14e-6，对渐近值则是1.39e-4／2.01e-6。
        assert bad_offset == pytest.approx(
            entry.expected[f"bad_response_at_horizon_n_{key}"],
            abs=entry.tolerances[f"bad_response_at_horizon_n_{key}"].abs_tol,
        )
        #: 渐近值本身另判一条，容差按尾巴给——**两条各说各的事**：
        #: 上一条判"推进对不对"，这一条判"这个环的直流落在哪里"。
        closed_form = entry.expected[f"bad_steady_offset_n_{key}"]
        assert bad_offset == pytest.approx(closed_form, abs=2.0e-4)
        #: **稳态误差变坏**——好控制器那一个是零。
        assert abs(bad_offset) > 1.0e5 * abs(good_offset), (
            f"坏控制器的稳态偏移{bad_offset!r}没有比好控制器{good_offset!r}"
            "坏五个数量级 —— 积分项在不在这件事应该看得出来"
        )
        #: 而且它几乎就是完全不控那个数：**"增益极小"的含义**。
        assert bad_offset == pytest.approx(
            entry.expected["uncontrolled_steady_offset_n"], rel=0.05
        )


# ---------------------------------------------------------------------------
# 稳定界：闭式解出来的那条线，两侧行为定性相反
# ---------------------------------------------------------------------------


def test_the_routh_limit_degenerates_to_the_cubic_form_bit_for_bit():
    """``τ = 0``时四阶Routh与三次Routh给出**同一个浮点数**。

    四阶：``Ki_界 = (a₃a₂ − a₄a₁)·a₁/(a₃²·G)``；``a₄ = 0``时它是
    ``a₂a₁/(a₃²G)``，而``a₃ = 1``，于是``= a₂a₁/G = (d + G·Kd)(Ka + G·Kp)/G``。

    2026-08-17实测``==``为真（两边都是``0.025799793601651185``），
    并且它同时等于``2ζω_n·Ka/G``——**界正比于开环阻尼**，
    而开环阻尼是``ζ = 0.0132``那一档。

    **"两条是同一条"这句话要么逐位成立，要么就不该说**
    （与`free_span_tension_step`那条``T = M/R``同源）。
    """

    entry = _oracle("oracle:closed_loop/stability_limit")
    stiffness = _oracle("oracle:closed_loop/plant_linearisation").expected[
        "span_stiffness_n_per_mm"
    ]
    limit = integral_gain_stability_limit(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
        bearing_damping_nmm_s=BEARING_DAMPING_NMM_S,
        torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
        clutch_lag_s=0.0,
        proportional=0.0,
        derivative=0.0,
    )
    assert limit == entry.expected["pure_integral_limit_zero_lag"]
    assert limit == entry.expected["cubic_routh_zero_lag"], (
        "四阶Routh在``τ = 0``处没有退化到三次那条——两条不是同一条了"
    )
    assert limit == pytest.approx(entry.expected["twice_zeta_omega_form"], rel=1e-15)


def test_the_nominal_clutch_refuses_the_fast_band_design():
    """**必须红**：额定离合器上用``fast``那套设计，Routh第一列已经翻号。

    实测``a₃a₂ − a₄a₁ = −16525.92``——**负的**。此时"积分增益的界"这个说法
    本身不成立（任何正的``Ki``都不稳），实现**必须失败关闭**而不是
    返回一个负的界让调用方拿去用。

    这条与"额定档只能用纯积分"是同一件事的两面：**不是我们没调好，
    是那套设计在这个执行器上没有可行域**。
    """

    entry = _oracle("oracle:closed_loop/stability_limit")
    fast = _oracle("oracle:closed_loop/fast_band_step")
    stiffness = _oracle("oracle:closed_loop/plant_linearisation").expected[
        "span_stiffness_n_per_mm"
    ]
    coefficients = closed_loop_characteristic_polynomial(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
        bearing_damping_nmm_s=BEARING_DAMPING_NMM_S,
        torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
        clutch_lag_s=entry.inputs["clutch_lag_s_nominal"],
        proportional=fast.inputs["proportional_a_per_n"],
        integral_gain=0.0,
        derivative=fast.inputs["derivative_a_s_per_n"],
    )
    routh = coefficients[1] * coefficients[2] - coefficients[0] * coefficients[3]
    assert routh == pytest.approx(
        entry.expected["nominal_routh_with_fast_design"], rel=1e-14
    )
    assert routh < 0.0

    with pytest.raises(TensionControlError, match="Routh"):
        integral_gain_stability_limit(
            span_stiffness_n_per_mm=stiffness,
            radius_mm=REEL_RADIUS_MM,
            inertia_kg_mm2=REEL_INERTIA_KG_MM2,
            bearing_damping_nmm_s=BEARING_DAMPING_NMM_S,
            torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
            clutch_lag_s=entry.inputs["clutch_lag_s_nominal"],
            proportional=fast.inputs["proportional_a_per_n"],
            derivative=fast.inputs["derivative_a_s_per_n"],
        )


@pytest.mark.batch
def test_the_two_sides_of_the_stability_limit_behave_qualitatively_opposite():
    """**阈值两侧行为定性相反**（与`incline_slide_threshold`同形）。

    纯积分、``fast``档：界是``0.0265916``。闭式说0.9倍时主导实部
    ``−0.50210``、1.1倍时``+0.50249``——**几乎反对称，界确实在中间**。

    2026-08-17实测（``dt = 1e-5``、跑0.4 s、比中段窗口与末段窗口的幅值）：
    0.9倍给比值**小于1**（衰减）、1.1倍给比值**大于1**（发散）。

    **判的是不等号不是数值**：包络比值受窗口对齐影响，而"衰减还是发散"
    不受。同时判闭式实部的符号——**两条一起才说明界是这条式子给的，
    不是碰巧**。
    """

    entry = _oracle("oracle:closed_loop/stability_limit")
    limit = entry.expected["pure_integral_limit_fast"]
    lag_s = entry.inputs["clutch_lag_s_fast"]
    horizon_s, dt_s = 0.4, 1.0e-5
    window = 3000

    ratios = {}
    for fraction in entry.inputs["probe_fractions"]:
        loop = _loop(
            lag_s=lag_s,
            proportional=0.0,
            integral_gain=fraction * limit,
            derivative=0.0,
            dt_s=dt_s,
        )
        _, samples = loop.run(
            int(round(horizon_s / dt_s)), takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S
        )
        deviations = [s.tension_n - loop.setpoint_n for s in samples]
        middle = len(deviations) // 2
        head = max(abs(x) for x in deviations[middle : middle + window])
        tail = max(abs(x) for x in deviations[-window:])
        ratios[fraction] = tail / head

    assert ratios[0.9] < 1.0, (
        f"0.9倍界上包络比值{ratios[0.9]!r} ≥ 1 —— 界以下应当收敛"
    )
    assert ratios[1.1] > 1.0, (
        f"1.1倍界上包络比值{ratios[1.1]!r} ≤ 1 —— 界以上应当发散，"
        "这条界要么算错了要么根本不在那里"
    )
    assert entry.expected["dominant_real_part_below"] < 0.0
    assert entry.expected["dominant_real_part_above"] > 0.0
    #: 两侧几乎反对称——**这一条说明"界"是解出来的而不是试出来的**。
    assert abs(
        entry.expected["dominant_real_part_above"]
        / entry.expected["dominant_real_part_below"]
    ) == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# 1.5：传感器位置≠被控点
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_settled_loop_still_misses_the_laydown_point_by_the_capstan_ratio():
    """**plans/15第1.5条的第一半**：闭环把测到的量调准了，落位点仍差``exp(μθ)``。

    2026-08-17实测（``fast``档、``μ = 0.3``、90°包角、跑0.05 s到沉下来）：

        读数     20.277777778939075   设定 20.27777777777778   误差 +1.16e-9
        落位点   32.48454681953038    比设定高 **60.1978%**
        落位/读数 1.6019776512823565  ＝ exp(0.4712) **逐位**

    **这个误差不是控制器不好，是它看不见。** 再好的环也只能把它测到的量调准，
    而判据的分辨力正在这里：**必须先证明环真的调准了**（误差1e-9），
    才谈得上"剩下的60%是看不见的那一部分"。
    """

    entry = _oracle("oracle:closed_loop/capstan_at_the_laydown_point")
    fast = _oracle("oracle:closed_loop/fast_band_step")
    capstan = CapstanSpan(
        friction_coefficient=entry.inputs["friction_coefficient"],
        wrap_angle_rad=entry.inputs["wrap_angle_rad"],
        #: 真机构型：传感器在跨段（松弛端），落位点在导轮下游（张紧端）。
        sensor_on_tight_side=False,
    )
    loop = _loop(
        lag_s=fast.inputs["clutch_lag_s"],
        proportional=fast.inputs["proportional_a_per_n"],
        integral_gain=fast.inputs["integral_gain_a_per_n_s"],
        derivative=fast.inputs["derivative_a_s_per_n"],
        capstan=capstan,
    )
    _, samples = loop.run(
        int(round(0.05 / PLANT_DT_S)), takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S
    )
    settled = samples[-1]

    #: 先证明环调准了它测到的那个量——**没有这一条，下面那条没有意义**。
    assert settled.measured_n == pytest.approx(loop.setpoint_n, abs=1.0e-6), (
        "闭环还没把读数调到设定值上，'剩下的误差是看不见的'这句话就无从说起"
    )
    assert settled.laydown_tension_n / settled.measured_n == pytest.approx(
        entry.expected["transfer_ratio"], rel=1.0e-15
    )
    assert settled.laydown_tension_n / loop.setpoint_n - 1.0 == pytest.approx(
        entry.expected["laydown_excess_fraction"], rel=1.0e-6
    )
    assert settled.laydown_tension_n == pytest.approx(
        entry.expected["laydown_tension_at_setpoint_n"], rel=1.0e-6
    )


def test_getting_the_wrap_direction_backwards_costs_the_square():
    """**plans/15第1.5条的第二半**：方向搞反，误差按**平方**放大。

    ``capstan_transfer_ratio``返回的恒是张紧端比松弛端（``≥ 1``），
    乘还是除由走向定。搞反一次：

        对   T_落位/T_读数 = exp(μθ)   = 1.6019777   ⟹ +60.198%
        反   T_落位/估计   = exp(2μθ)  = 2.5663324   ⟹ **+156.633%**

    2026-08-17实测``真值/反向估计 = 2.5663323952081356``对
    ``exp(2μθ) = 2.566332395208135``——**差一个ulp**。

    **判据判的是"平方"这件事本身**：清单把``exp(μθ)``与``exp(2μθ)``
    分开给值，于是一个凑出来的2.566过不了"它必须等于前一条的平方"那道门。
    """

    entry = _oracle("oracle:closed_loop/capstan_at_the_laydown_point")
    kwargs = {
        "friction_coefficient": entry.inputs["friction_coefficient"],
        "wrap_angle_rad": entry.inputs["wrap_angle_rad"],
    }
    right = CapstanSpan(**kwargs, sensor_on_tight_side=False)
    wrong = CapstanSpan(**kwargs, sensor_on_tight_side=True)

    reading = 20.0
    truth = right.laydown_tension_n(reading)
    estimate = wrong.laydown_tension_n(reading)
    assert right.ratio == pytest.approx(entry.expected["transfer_ratio"], rel=1e-15)
    assert truth / estimate == pytest.approx(
        entry.expected["reversed_ratio"], rel=1.0e-15
    )
    #: **平方**：反向的比值必须逐位（一个ulp内）是正向比值的平方。
    assert truth / estimate == pytest.approx(right.ratio**2, rel=1.0e-15)
    assert truth / estimate - 1.0 == pytest.approx(
        entry.expected["reversed_excess_fraction"], rel=1.0e-15
    )
    #: 而误差**不是**两倍——把平方写成两倍是这条判据挡的那个错。
    assert abs(truth / estimate - 1.0) > 2.0 * abs(right.ratio - 1.0), (
        "反向误差没有超过正向误差的两倍 —— 那说明它被写成了线性放大而不是平方"
    )


# ---------------------------------------------------------------------------
# 裁决B：两条时钟
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_a_coarser_control_period_monotonically_degrades_the_same_design():
    """**裁决B的量**：同一套增益，控制周期越粗闭环越差，直到它撑不住。

    2026-08-17实测（``fast``档、``dt_推进 = 1e-6``、抽取比逐档加大）：

    | 抽取比 | 控制周期 | 峰值 | ISE |
    |---|---|---|---|
    | 1 | 1 µs | 0.3376856 | 2.9283e-4 |
    | 10 | 10 µs | 0.3394912 | 2.9450e-4 |
    | 100 | 100 µs | 0.3589867 | 3.1627e-4 |
    | 200 | 200 µs | 0.3833266 | 3.5618e-4 |
    | 500 | 500 µs | 0.4649745 | 9.9286e-4 |
    | **1000** | **1 ms** | **失败关闭** | 放线盘转速穿零 |

    **最后一行是这条门最要紧的一行**：1 ms控制周期上这套增益已经不稳，
    而它不是给出一个错的数——是被`transport.PayoutReel`那条``ω ≤ 0``
    **接住**。两条链路的失败关闭串起来了。

    ``decimation = 1``那一档的控制周期等于推进步长。**那不是一台机器的采样率**，
    是把采样-保持的离散误差压到判据分辨率之下的口径——闭式对拍用它。
    """

    fast = _oracle("oracle:closed_loop/fast_band_step")
    gains = {
        "lag_s": fast.inputs["clutch_lag_s"],
        "proportional": fast.inputs["proportional_a_per_n"],
        "integral_gain": fast.inputs["integral_gain_a_per_n_s"],
        "derivative": fast.inputs["derivative_a_s_per_n"],
    }
    peaks = []
    for decimation in (1, 100, 500):
        loop = _loop(**gains, decimation=decimation)
        assert loop.control_period_s == decimation * PLANT_DT_S
        _, samples = loop.run(
            int(round(0.05 / PLANT_DT_S)), takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S
        )
        ticks = sum(1 for sample in samples if sample.control_tick)
        assert ticks == len(samples) // decimation, (
            f"抽取比{decimation}下控制拍数{ticks}对不上样点数{len(samples)} —— "
            "零阶保持的边界与抽取比不一致"
        )
        peaks.append(_metrics(samples, loop.setpoint_n, PLANT_DT_S)["peak"])

    for finer, coarser in zip(peaks, peaks[1:], strict=False):
        assert coarser > finer, (
            f"控制周期变粗而峰值没有变大（{finer!r} → {coarser!r}）—— "
            "那说明抽取比没有真的生效，零阶保持是假的"
        )

    #: 1 ms控制周期：这套增益撑不住，而失败关闭在`transport`那一侧。
    from physics_engine.transport import TransportError

    with pytest.raises(TransportError, match="angular_velocity_rad_s"):
        loop = _loop(**gains, decimation=1000)
        loop.run(
            int(round(0.2 / PLANT_DT_S)), takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S
        )
