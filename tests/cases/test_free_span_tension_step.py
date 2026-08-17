"""自由跨段张力对线速度阶跃的conformance门（案例`cases/free_span_tension_step`）。

守plans/14第3.2节的二号缺口（线速度与输运），决策0066。

## 这条案例存在的理由，一句话

`drives.SpoolTension`把张力写成``T = M/R``——**那里面没有速度，
于是控制器没有任何东西可控**。本案例把张力变成
"放线端与收线端的长度不匹配经带材弹性生成的量"，
于是**收线端速度阶跃变成一个真的扰动**，而抑制它正是控制器的职责。

## 六条判据各自守什么、各自的精度从哪来

| 判据 | 实测精度 | 精度由什么定 |
|---|---|---|
| ``c = 0``时``T_ss``与`SpoolTension`**逐位相等** | 0（同一个float） | 恒等式：旧模型是本模型的零阻尼稳态 |
| 稳态起点在推进下不动 | 漂移**恰为0**（20000步） | 半隐式Euler的离散不动点＝连续不动点 |
| 阶跃超调与峰值时刻 | 9.03e-6 / 7.35e-5 | **半隐式Euler的一阶误差**，``dt``减半即减半 |
| 无阻尼包络12个周期不衰减 | 幅值比**0.99999998** | 辛格式在无阻尼极限下的幅值守恒 |
| 时间步收敛阶 | 比1.99906/1.99911/1.99868 | 同上，**一阶** |
| 方向 | 符号，零容差 | 结构 |

**前两条是恒等式，中间三条是收敛结果，最后一条是符号**。分开写是因为它们
红了说明的事完全不同：第一条红说明两个模型的关系断了，第二条红说明离散化
给稳态引入了偏差，中间三条红说明步长不够或线性化用错了参考点，
最后一条红说明**符号反了**——而符号反了的症状是指数发散，不是一个错的数。

## 峰值时刻那条容差由**采样栅格**定，不由模型精度定

``dt = 1e-6``对``t_p ≈ 4.2e-3``是2.4e-4相对的栅格。实测两档都落在1.76e-5，
比栅格还小是因为峰附近是二次的（离峰值半个格时函数值只差``O(dt²)``），
但**判据不能按1.76e-5给**：换一个``dt``它立刻退回栅格量级。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import restitution_from_damping_ratio
from physics_engine.drives import SpoolTension, step_response_overshoot
from physics_engine.motion import PauseInterval
from physics_engine.oracles import load_manifest
from physics_engine.transport import (
    FreeSpan,
    MaterialFeedTimeline,
    PayoutReel,
    SpanTransportLoop,
    span_damping_ratio,
    span_natural_frequency_rad_s,
    steady_state_tension_n,
    velocity_step_overshoot,
    velocity_step_peak_time_s,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "free_span_tension_step"
MANIFEST = load_manifest(CASE / "oracle.json")

#: 全部**假设输入**，逐条出处见案例页第一节与金标生成器。
AXIAL_STIFFNESS_N = 60000.0
GEOMETRIC_LENGTH_MM = 300.0
REEL_RADIUS_MM = 60.0
REEL_INERTIA_KG_MM2 = 5000.0
BRAKE_TORQUE_NMM = 1200.0
LINE_SPEED_MM_S = 20.0
STEP_MM_S = 2.0
FINE_STEP_MM_S = 0.2

DAMPING_UNDAMPED = 0.0
DAMPING_NOMINAL = 50.0
DAMPING_STIFF = 1000.0

CONVERGENCE_DT_S = (1.6e-5, 8.0e-6, 4.0e-6, 2.0e-6)
CONVERGENCE_HORIZON_S = 0.02

SPAN = FreeSpan(
    span_id="span/free",
    geometric_length_mm=GEOMETRIC_LENGTH_MM,
    axial_stiffness_n=AXIAL_STIFFNESS_N,
)


def _oracle(oracle_id: str):
    for entry in MANIFEST.oracles:
        if entry.id == oracle_id:
            return entry
    raise AssertionError(f"清单里没有{oracle_id}")


def _reel(damping: float) -> PayoutReel:
    return PayoutReel(
        reel_id="reel/payout",
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
        bearing_damping_nmm_s=damping,
    )


def _loop(damping: float, dt_s: float) -> SpanTransportLoop:
    """从闭式稳态起手——**判据要判的东西不该被起点的瞬态污染**。"""

    return SpanTransportLoop.at_steady_state(
        span=SPAN,
        reel=_reel(damping),
        dt_s=dt_s,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        line_speed_mm_s=LINE_SPEED_MM_S,
        forbid_slack=True,
    )


@pytest.fixture(scope="module")
def step_runs():
    """两档阻尼各跑一次阶跃，超调/峰值时刻/峰值幅值三条门共用。"""

    runs = {}
    for damping, label, horizon in (
        (DAMPING_NOMINAL, "nominal", 0.006),
        (DAMPING_STIFF, "stiff", 0.008),
    ):
        _, samples = _loop(damping, 1.0e-6).run(
            int(round(horizon / 1.0e-6)),
            brake_torque_nmm=BRAKE_TORQUE_NMM,
            takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S,
        )
        runs[label] = samples
    return runs


@pytest.fixture(scope="module")
def undamped_run():
    """``c = 0``跑满0.2秒（12.08个周期），包络门与幅值门共用。"""

    _, samples = _loop(DAMPING_UNDAMPED, 1.0e-6).run(
        200000,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S,
    )
    return samples


# ---------------------------------------------------------------------------
# 恒等式两条
# ---------------------------------------------------------------------------


def test_the_old_torque_over_radius_is_the_zero_damping_steady_state_bit_for_bit():
    """``c = 0``时``T_ss``与`drives.SpoolTension.tension_n`**逐位相等**。

    这一条不是装饰。0062给出的``T = M/R``今天仍在`drives`里，而本模块说
    **那不是一条定律，是一个稳态**。这句话要么逐位成立，要么就不该说——
    差一个ulp都意味着两处各自写了一遍同一件事而没有对过。

    实测两边都是``20.0``，``==``为真（不是`approx`）。

    **本门不判引擎积分**，所以不标`batch`：它是两条闭式之间的关系。
    """

    entry = _oracle("oracle:free_span/steady_state_torque_balance")
    transport_value = steady_state_tension_n(
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        radius_mm=REEL_RADIUS_MM,
        bearing_damping_nmm_s=DAMPING_UNDAMPED,
        line_speed_mm_s=LINE_SPEED_MM_S,
    )
    spool = SpoolTension(barrel_radius_mm=REEL_RADIUS_MM, tape_thickness_mm=0.1)
    drives_value = spool.tension_n(BRAKE_TORQUE_NMM, 0.0)

    assert transport_value == drives_value, (
        f"transport给{transport_value!r}而drives给{drives_value!r}——"
        "'旧模型是本模型的特例'这句话不成立了"
    )
    assert transport_value == entry.expected["tension_zero_damping_n"]
    assert drives_value == entry.expected["torque_over_radius_n"]

    #: 线速度不为零时两者**必须**分开：轴承阻力矩那一项是真的。
    with_speed = steady_state_tension_n(
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        radius_mm=REEL_RADIUS_MM,
        bearing_damping_nmm_s=DAMPING_NOMINAL,
        line_speed_mm_s=LINE_SPEED_MM_S,
    )
    assert with_speed > drives_value, (
        "有轴承阻力矩、有线速度时张力仍等于M/R——那一项被算丢了"
    )
    assert with_speed == pytest.approx(
        entry.expected["tension_nominal_damping_n"], rel=1e-15
    )


@pytest.mark.batch
def test_the_closed_form_steady_state_is_a_fixed_point_of_the_discrete_scheme():
    """从闭式稳态起手，推进20000步（0.2秒）**一点都不动**。

    半隐式Euler的两条更新在稳态处各自恰为零（``α = 0``且``ω·R = v_收线``），
    所以**离散不动点与连续不动点逐字相同**——这条判据判的是模型，不是步长。

    2026-08-17实测：起点与闭式差``3.20e-12 N``（那是``T → L_mat → T``往返的
    浮点舍入），而**整条推进的最大漂移恰为0.0**。
    判绝对不判相对，容差按往返舍入给（四档张力最坏3.908e-12，取1e-10）。
    """

    entry = _oracle("oracle:free_span/steady_state_torque_balance")
    limit = entry.tolerances["fixed_point_drift_n"].abs_tol
    exact = entry.expected["tension_zero_damping_n"]

    loop = _loop(DAMPING_UNDAMPED, 1.0e-5)
    assert loop.tension_n == pytest.approx(exact, abs=limit)
    assert loop.material_length_mm == pytest.approx(
        entry.expected["material_length_zero_damping_mm"], rel=1e-15
    )

    final, samples = loop.run(
        20000, brake_torque_nmm=BRAKE_TORQUE_NMM, takeup_speed_mm_s=LINE_SPEED_MM_S
    )
    base = samples[0].tension_n
    drift = max(abs(sample.tension_n - base) for sample in samples)
    assert drift <= limit, f"稳态起点自己漂了{drift!r} N —— 离散化给稳态引入了偏差"
    assert final.tension_n == pytest.approx(exact, abs=limit)

    #: 两档有阻尼的稳态起点同样要落在闭式上，否则"``c·v/R²``那一项"只是写着好看。
    for damping, key in (
        (DAMPING_NOMINAL, "tension_nominal_damping_n"),
        (DAMPING_STIFF, "tension_stiff_damping_n"),
    ):
        assert _loop(damping, 1.0e-5).tension_n == pytest.approx(
            entry.expected[key], abs=limit
        )


# ---------------------------------------------------------------------------
# 阶跃扰动：本轨道的存在理由
# ---------------------------------------------------------------------------


@pytest.mark.batch
@pytest.mark.parametrize("label", ("nominal", "stiff"))
def test_the_takeup_velocity_step_response_matches_the_second_order_closed_form(
    step_runs, label
):
    """收线端速度阶跃10%，张力的超调、峰值时刻与峰值幅值**三条并判**。

    闭式**不是**教科书那条二阶阶跃：张力连续而速度差当场跳``−Δv``，
    初值带一个斜率冲击``y'(0) = ω_n/(2ζ)``，等价于传递函数多一个零点。
    于是``t_p = (π − acos ζ)/ω_d``而不是``π/ω_d``、
    超调是``exp(−ζ(π−acos ζ)/√(1−ζ²))/(2ζ)``而不是``exp(−ζπ/√(1−ζ²))``。

    2026-08-17实测（``dt = 1e-6``）：

    | 档 | ζ | 超调闭式 | 超调实测 | 相对偏差 |
    |---|---|---|---|---|
    | nominal（``c = 50``） | 0.013172 | 37.176327 | 37.176663 | 9.03e-6 |
    | stiff（``c = 1000``） | 0.263409 | 1.1493664 | 1.1492819 | 7.35e-5 |

    **两档必须都判**：只在一个``ζ``上判，判不出这条闭式与
    `contact.restitution_from_damping_ratio`是两件事（见盲区那条门）。

    **三条量必须并判**：超调只看``ζ``；峰值时刻还看``ω_n``（把``J``与``K``
    同乘一个倍数，``ζ``不变而``ω_n``变）；峰值幅值还看``ΔT_ss``
    （``c → 0``时``ΔT_ss → 0``，只判相对超调时把它算错照样过）。
    """

    entry = _oracle(f"oracle:free_span/velocity_step_{label}")
    samples = step_runs[label]
    before = entry.expected["tension_before_n"]
    change = entry.expected["steady_change_n"]

    peak = max(samples, key=lambda sample: sample.tension_n)
    excursion = peak.tension_n - before
    overshoot = excursion / change - 1.0

    assert overshoot == pytest.approx(
        entry.expected["relative_overshoot"],
        rel=entry.tolerances["relative_overshoot"].rel_tol,
    )
    assert peak.time_s == pytest.approx(
        entry.expected["peak_time_s"], rel=entry.tolerances["peak_time_s"].rel_tol
    )
    assert excursion == pytest.approx(
        entry.expected["peak_excursion_n"],
        rel=entry.tolerances["peak_excursion_n"].rel_tol,
    )

    #: 闭式参数本身也对一遍——它们是上面三条的分母，错了上面三条会一起错。
    stiffness = SPAN.stiffness_n_per_mm(
        SPAN.material_length_for_tension_mm(entry.expected["tension_after_n"])
    )
    assert stiffness == pytest.approx(
        entry.expected["span_stiffness_n_per_mm"], rel=1e-14
    )
    natural = span_natural_frequency_rad_s(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
    )
    assert natural == pytest.approx(
        entry.expected["natural_frequency_rad_s"], rel=1e-14
    )
    ratio = span_damping_ratio(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
        bearing_damping_nmm_s=(
            DAMPING_NOMINAL if label == "nominal" else DAMPING_STIFF
        ),
    )
    assert ratio == pytest.approx(entry.expected["damping_ratio"], rel=1e-14)
    assert velocity_step_overshoot(ratio) == pytest.approx(
        entry.expected["relative_overshoot"], rel=1e-14
    )
    assert velocity_step_peak_time_s(
        natural_frequency_rad_s=natural, damping_ratio=ratio
    ) == pytest.approx(entry.expected["peak_time_s"], rel=1e-14)


def test_dropping_the_unit_conversion_would_move_the_natural_frequency_by_thirty_two():
    """**1000倍单位bug的捕手**（与`two_body_spring`那条同源）。

    ``ω_n = sqrt(1000·R²·K/J)``里的1000是``1 N = 1000 kg·mm/s²``。
    掉了它``ω_n``要小``sqrt(1000) = 31.62``倍，而``ζ``要大同样的倍数。
    本门直接算一遍"掉了会怎样"并断言它离闭式**远到不可能是噪声**。
    """

    entry = _oracle("oracle:free_span/velocity_step_nominal")
    stiffness = entry.expected["span_stiffness_n_per_mm"]
    correct = entry.expected["natural_frequency_rad_s"]
    without_units = math.sqrt(
        REEL_RADIUS_MM * REEL_RADIUS_MM * stiffness / REEL_INERTIA_KG_MM2
    )
    assert correct / without_units == pytest.approx(math.sqrt(1000.0), rel=1e-14)
    assert span_natural_frequency_rad_s(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
    ) == pytest.approx(correct, rel=1e-14)


@pytest.mark.batch
def test_the_open_loop_span_is_practically_undamped_which_is_why_a_controller_exists(
    undamped_run,
):
    """``c = 0``时速度阶跃引起的张力振荡**自己不会停**。

    这是本案例最要紧的一条。2026-08-17实测（``dt = 1e-6``，跑满0.2秒＝
    **12.08个周期**）：

    * 幅值``1.0544563 N``对闭式``K·Δv/ω_n = 1.0544439``，相对1.17e-5；
    * 首峰落在``4.138 ms``对四分之一周期``4.1380324 ms``，相对7.84e-6；
    * **末周期幅值/首周期幅值 = 0.99999998**——十二个周期一共衰减了2e-8。

    ``c = 0``是理想化的，但**真实量级的轴承阻力矩（``c = 50``）给``ζ = 0.0132``**，
    衰减时间常数``1/(ζω_n) = 0.2 s``——比一次落位动作还长。
    **张力扰动在这条链路上开环是不会自己消失的**，
    而`drives.SpoolTension`那个模型里它压根不存在。

    幅值``1.05 N``是20 N张力的**5.3%**，由一次10%的线速度阶跃引起。
    """

    entry = _oracle("oracle:free_span/undamped_oscillation")
    base = _oracle("oracle:free_span/steady_state_torque_balance").expected[
        "tension_zero_damping_n"
    ]
    peak = max(undamped_run, key=lambda sample: sample.tension_n)

    assert peak.tension_n - base == pytest.approx(
        entry.expected["amplitude_n"], rel=entry.tolerances["amplitude_n"].rel_tol
    )
    assert peak.time_s == pytest.approx(
        entry.expected["quarter_period_s"],
        rel=entry.tolerances["quarter_period_s"].rel_tol,
    )

    period = entry.expected["period_s"]
    horizon = undamped_run[-1].time_s
    assert horizon / period > 12.0, "跑得不够长，'不衰减'这句话没有分量"

    def amplitude(window) -> float:
        highs = max(sample.tension_n for sample in window)
        lows = min(sample.tension_n for sample in window)
        return (highs - lows) / 2.0

    head = amplitude([s for s in undamped_run if s.time_s < period])
    tail = amplitude([s for s in undamped_run if s.time_s > horizon - period])
    assert tail / head == pytest.approx(1.0, abs=1.0e-6), (
        f"无阻尼跨段的振荡衰减了：首周期{head!r}、末周期{tail!r}——"
        "辛格式在无阻尼极限下应当守住幅值，衰减说明格式或模型多了一项耗散"
    )


# ---------------------------------------------------------------------------
# 收敛阶与它的floor
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_time_step_error_is_first_order():
    """**步长减半，误差减半。**

    2026-08-17实测（``c = 1000``、``Δv = 0.2``、``t = 0.02 s``，
    ``dt = 1.6e-5 → 2e-6``）：误差``3.605e-5 / 1.803e-5 / 9.021e-6 / 4.513e-6``，
    三个比值**1.99906 / 1.99911 / 1.99868**。

    门判比值落在``[1.9, 2.1]``而**不写死为2**——与`harmonic_oscillator`
    那条"收敛比不写死为4"同源：**一致才是收敛的证据，具体值不是**。

    **半隐式Euler是一阶不是从`integrate.py`推出来的**，是在这里量出来的：
    这条链路把一个精确的代数张力律与一个显式转速更新串在一起，
    组合的阶要么量、要么就是猜。
    """

    entry = _oracle("oracle:free_span/time_step_convergence")
    exact = entry.expected["tension_at_horizon_n"]
    errors = []
    for dt in CONVERGENCE_DT_S:
        final, _ = _loop(DAMPING_STIFF, dt).run(
            int(round(CONVERGENCE_HORIZON_S / dt)),
            brake_torque_nmm=BRAKE_TORQUE_NMM,
            takeup_speed_mm_s=LINE_SPEED_MM_S + FINE_STEP_MM_S,
        )
        errors.append(abs(final.tension_n - exact))
        assert final.tension_n == pytest.approx(
            exact, rel=entry.tolerances["tension_at_horizon_n"].rel_tol
        )

    assert errors[0] > errors[1] > errors[2] > errors[3], f"误差没有单调下降：{errors}"
    low = entry.expected["order_ratio_low"]
    high = entry.expected["order_ratio_high"]
    for earlier, later in zip(errors, errors[1:], strict=False):
        assert low <= earlier / later <= high, (
            f"时间步误差不是一阶：比值{earlier / later!r}，实测序列{errors}"
        )


@pytest.mark.batch
def test_the_linearised_oracle_puts_a_floor_under_an_error_that_should_vanish():
    """**判据选错会怎样**：金标是**线性化**解，它与非线性真解的偏差是一条floor。

    这条与`capstan_tension_ratio`那条"拿``e^{μΔφ}``当逐节点判据"同形。
    线性化的系统偏差``∝ Δv²``而响应``∝ Δv``，所以**相对偏差``∝ Δv``**：
    阶跃越大floor越高，收敛阶那条门就会在越粗的``dt``上先撞上它然后停住，
    **而症状看起来像"收敛阶坏了"**。

    2026-08-17实测（同一批``dt``、同一个终点时刻）：

    | ``Δv`` | 比值 |
    |---|---|
    | 0.2 mm/s | 1.99906 / 1.99911 / 1.99868 |
    | 2.0 mm/s | 1.99652 / 1.99404 / **1.98863** |

    再往细里走（``Δv = 2``、``dt`` 到 2.5e-7）实测掉到**1.9168**——
    那一档没有进本门，因为80000步的一趟只为演示一件已经被本门证到的事。

    **本门不注错任何代码**，它量的是两条闭式之间的关系：
    同一批``dt``上，大阶跃的比值必须**逐档更差**。
    """

    entry = _oracle("oracle:free_span/time_step_convergence")
    series = {}
    for label, step, key in (
        ("fine", FINE_STEP_MM_S, "tension_at_horizon_n"),
        ("coarse", STEP_MM_S, "coarse_tension_at_horizon_n"),
    ):
        exact = entry.expected[key]
        errors = []
        for dt in CONVERGENCE_DT_S:
            final, _ = _loop(DAMPING_STIFF, dt).run(
                int(round(CONVERGENCE_HORIZON_S / dt)),
                brake_torque_nmm=BRAKE_TORQUE_NMM,
                takeup_speed_mm_s=LINE_SPEED_MM_S + step,
            )
            errors.append(abs(final.tension_n - exact))
        series[label] = [
            earlier / later for earlier, later in zip(errors, errors[1:], strict=False)
        ]

    for fine, coarse in zip(series["fine"], series["coarse"], strict=True):
        assert coarse < fine, (
            f"大阶跃的收敛比{coarse!r}没有比小阶跃的{fine!r}更差——"
            "那说明线性化floor这个解释是错的，容差表要重算"
        )
    assert series["coarse"][-1] < entry.expected["order_ratio_low"] + 0.1, (
        f"最细一档的大阶跃比值{series['coarse'][-1]!r}还没有开始掉下来，"
        "floor的量级判断要重写"
    )


# ---------------------------------------------------------------------------
# 方向门
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_which_end_speeds_up_when_the_tension_rises():
    """**方向门（引擎级）**：三个符号，零容差。

    1. **收线端加速 ⟹ 张力上升**（材料被抽走）；
    2. **收线端减速 ⟹ 张力下降**（材料堆在跨段里）；
    3. **制动力矩加大 ⟹ 张力上升**（放线盘被拉转得更慢）。

    第3条同时回答了"张力升高让哪一端慢下来"这个问法：
    **在本模型里没有一端会慢下来**。收线端是伺服给定的（CSP位置模式），
    张力改不动它；放线端**加速**——离合器是制动，带材是唯一的驱动源，
    张力大了盘就被拉得更快。真正慢下来的是**跨段材料的净流失速率**，
    而那不是任何一端的速度。

    2026-08-17实测（``c = 50``、``dt = 1e-6``、2000步）：
    收线端``±2 mm/s``给``+0.72620``/``−0.72618 N``（几乎反对称，
    差0.002%来自非线性）；制动力矩``±1%``给``+0.054588``/``−0.054588 N``。

    **本仓在`winding_line_endtoend`上把一个方向搞反过一次（比值0.5587）。**
    这道门守的就是那一类：符号反了的症状是指数发散，不是一个错的数。
    """

    base = None
    deltas = {}
    for label, takeup in (
        ("takeup_faster", LINE_SPEED_MM_S + STEP_MM_S),
        ("takeup_slower", LINE_SPEED_MM_S - STEP_MM_S),
    ):
        final, samples = _loop(DAMPING_NOMINAL, 1.0e-6).run(
            2000, brake_torque_nmm=BRAKE_TORQUE_NMM, takeup_speed_mm_s=takeup
        )
        base = samples[0].tension_n
        deltas[label] = final.tension_n - base

    assert deltas["takeup_faster"] > 0.0, (
        f"收线端加速而张力没有上升（Δ = {deltas['takeup_faster']!r}）——"
        "被抽走的材料应该让跨段绷紧，长度账里放线/收线对调了"
    )
    assert deltas["takeup_slower"] < 0.0, (
        f"收线端减速而张力没有下降（Δ = {deltas['takeup_slower']!r}）"
    )
    #: 一阶上必须反对称——不反对称说明有一项与速度差**同号**地混进来了。
    assert deltas["takeup_faster"] == pytest.approx(
        -deltas["takeup_slower"], rel=1.0e-3
    )

    for label, brake, sign in (
        ("brake_up", BRAKE_TORQUE_NMM * 1.01, 1.0),
        ("brake_down", BRAKE_TORQUE_NMM * 0.99, -1.0),
    ):
        final, samples = _loop(DAMPING_NOMINAL, 1.0e-6).run(
            2000, brake_torque_nmm=brake, takeup_speed_mm_s=LINE_SPEED_MM_S
        )
        delta = final.tension_n - samples[0].tension_n
        assert delta * sign > 0.0, (
            f"{label}：制动力矩变了{brake / BRAKE_TORQUE_NMM:.2%}而张力"
            f"变化{delta!r}，符号不对——离合器被写成了驱动"
        )


# ---------------------------------------------------------------------------
# 盲区：exp(−ζΦ/√(1−ζ²))这个形状在本仓的第三支
# ---------------------------------------------------------------------------


def test_the_overshoot_family_has_three_members_and_zeta_one_half_is_a_blind_spot():
    """``exp(−ζΦ/√(1−ζ²))``在本仓出现了三次，而``ζ = 0.5``让其中两支重合。

    | 出处 | ``Φ`` | 前因子 |
    |---|---|---|
    | `contact.restitution_from_damping_ratio` | ``2·acos ζ`` | 无 |
    | `drives.step_response_overshoot` | ``π`` | 无 |
    | `transport.velocity_step_overshoot` | ``π − acos ζ`` | ``1/(2ζ)`` |

    ``π − acos ζ = 2·acos ζ`` ⟺ ``acos ζ = π/3`` ⟺ **``ζ = 0.5``**，
    而``1/(2ζ)``在同一点**恰好是1**。**两处退化撞在同一个点上**：
    2026-08-17实测``ζ = 0.5``时本式给``0.29843605919227484``、
    恢复系数给``0.2984360591922749``——**差一个ulp，是同一个实数**。

    ``ζ = 0.5``正是写测试时最顺手的那个值。**只在它上面判，
    判不出这两件事是两件事**——本门因此在五个``ζ``上扫。

    这条与`drives.py`docstring里那条"实测当场否掉'逐位相同'的判断"是同一族，
    但方向相反：那次是"以为相同、实测不同"，这次是
    **"结构上不同、而在最常用的那个点上实测相同"**。后者更难看，
    因为它不会红——它只是让一道门变成没有内容的门。
    """

    entry = _oracle("oracle:free_span/overshoot_family_blind_spot")
    ratios = entry.inputs["damping_ratios"]
    degenerate = entry.expected["degenerate_damping_ratio"]
    assert degenerate in ratios, "盲区那个点必须在扫描里"

    for index, ratio in enumerate(ratios):
        assert velocity_step_overshoot(ratio) == pytest.approx(
            entry.expected[f"span_{index}"], rel=1e-14
        )
        assert restitution_from_damping_ratio(ratio) == pytest.approx(
            entry.expected[f"restitution_{index}"], rel=1e-14
        )
        assert step_response_overshoot(ratio) == pytest.approx(
            entry.expected[f"textbook_{index}"], rel=1e-14
        )

    #: 退化点：两支给同一个实数（实测差一个ulp）。
    assert velocity_step_overshoot(degenerate) == pytest.approx(
        restitution_from_damping_ratio(degenerate), rel=1e-15
    )
    #: 而在别的``ζ``上必须**明显**分开——"明显"取10%，实测最小间隔是
    #: ``ζ = 0.25``处的1.2490 vs 0.5063（差147%）。
    for ratio in ratios:
        if ratio == degenerate:
            continue
        span = velocity_step_overshoot(ratio)
        restitution = restitution_from_damping_ratio(ratio)
        assert abs(span / restitution - 1.0) > 0.1, (
            f"ζ = {ratio!r}处两支只差{abs(span / restitution - 1.0):.3%}——"
            "退化点不止一个，本门的结论要重写"
        )

    #: 与教科书那支在**任何**扫描点上都不许重合（它连退化点都没有）。
    for ratio in ratios:
        assert abs(velocity_step_overshoot(ratio) / step_response_overshoot(ratio) - 1.0) > 0.1
    #: 退化关系是解出来的：``acos(1/2) = π/3``，于是``π − π/3 = 2·(π/3)``。
    assert math.pi - math.acos(degenerate) == pytest.approx(
        2.0 * math.acos(degenerate), rel=1e-15
    )
    assert 1.0 / (2.0 * degenerate) == 1.0


# ---------------------------------------------------------------------------
# 上游接缝：喂料长度时间线 → 线速度 → 张力
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_feed_length_derivative_drives_the_span_and_reproduces_the_scalar_run():
    """**plans/14第3.2节那条接缝**：``material_feed_length_mm``的时间导数＝线速度。

    上游``wii_motion_timeline.v2``给的是累计喂料长度。段内恒速语义下，
    时间导数就是逐段的差商——本门先判那两个差商（20.0与22.0 mm/s，
    正好是本案例判的那次10%阶跃），再把整条时间线接到跨段上跑。

    2026-08-17实测：**时间线驱动与标量驱动逐位相同**
    （8000步全部样点``==``，末态张力``26.708364419146104``两边一个字节不差）。
    这不是巧合而是构造：段内恒速语义下``speed_mm_s``在每一步给出的
    就是那两个常数本身。

    **逐位相同是有意判的**：只判"接近"的话，一个把速率算成
    "样点处线性插值"的实现照样能过——而那是一条不同的语义，
    上游没有声明过它。
    """

    entry = _oracle("oracle:free_span/material_feed_timeline")
    timeline = MaterialFeedTimeline(
        source_id="transport/wii_feed",
        times_s=tuple(entry.inputs["times_s"]),
        feed_length_mm=tuple(entry.inputs["feed_length_mm"]),
        rate_semantics="piecewise_constant",
        extrapolation="reject",
        pauses=(),
    )
    speeds = timeline.segment_speeds_mm_s()
    assert speeds[0] == pytest.approx(entry.expected["segment_speed_0_mm_s"], rel=1e-15)
    assert speeds[1] == pytest.approx(entry.expected["segment_speed_1_mm_s"], rel=1e-15)
    assert timeline.length_mm(0.001) == pytest.approx(
        entry.expected["length_at_mid_first_segment_mm"], rel=1e-15
    )
    #: 样点处**右连续**：``t = 0.002``属于第二段。这是一条被声明的语义，
    #: 判它是为了让"段内恒速"这四个字有确定的含义。
    assert timeline.speed_mm_s(0.002) == speeds[1]
    assert timeline.speed_mm_s(0.0019999) == speeds[0]

    driven_final, driven = _loop(DAMPING_STIFF, 1.0e-6).run_timeline(
        timeline, brake_torque_nmm=BRAKE_TORQUE_NMM, steps=8000
    )
    first, head = _loop(DAMPING_STIFF, 1.0e-6).run(
        2000, brake_torque_nmm=BRAKE_TORQUE_NMM, takeup_speed_mm_s=speeds[0]
    )
    scalar_final, tail = first.run(
        6000, brake_torque_nmm=BRAKE_TORQUE_NMM, takeup_speed_mm_s=speeds[1]
    )
    assert driven_final.tension_n == scalar_final.tension_n, (
        f"时间线驱动给{driven_final.tension_n!r}而标量驱动给"
        f"{scalar_final.tension_n!r} —— 段内恒速语义没有被兑现"
    )
    assert all(
        left.tension_n == right.tension_n
        for left, right in zip(driven, head + tail, strict=True)
    )


def test_a_zero_motion_segment_needs_an_explicit_pause_to_be_readable():
    """零运动段**必须**被声明的暂停覆盖——否则它与丢数据在字节上没有区别。

    这条不判物理，判的是**上游契约怎么被读**。WII的``pause_intervals``
    存在的理由就是这个：零运动段没有可推断的时长。

    带暂停时逐段速率是``(20.0, 0.0, 29.333333333333332)``——
    中间那段恰为零，而**它之所以可读，是因为有一条显式时间戳说它是暂停**。
    """

    timeline = MaterialFeedTimeline(
        source_id="transport/paused",
        times_s=(0.0, 0.002, 0.004, 0.010),
        feed_length_mm=(0.0, 0.04, 0.04, 0.216),
        rate_semantics="piecewise_constant",
        extrapolation="reject",
        pauses=(
            PauseInterval(
                pause_id="pause/transfer",
                start_time_s=0.002,
                end_time_s=0.004,
                reason="工件换手，带材不走",
            ),
        ),
    )
    speeds = timeline.segment_speeds_mm_s()
    assert speeds[1] == 0.0
    assert speeds[0] == pytest.approx(20.0, rel=1e-15)
    assert speeds[2] == pytest.approx(0.176 / 0.006, rel=1e-12)
    assert timeline.speed_mm_s(0.003) == 0.0
    assert timeline.length_mm(0.003) == timeline.length_mm(0.002)
