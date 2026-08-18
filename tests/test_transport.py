"""线速度与输运的门（决策0066，plans/14第3.2节的二号缺口）。

## 本文件的重心是**那道方向门**

`cases/free_span_tension_step`判的是数对不对；本文件判的是**符号对不对**，
以及每一条失败关闭真的关得住。

方向门在真实对象上**永远造不出一次红**——``K > 0``、``R²/J > 0``都是结构性的。
所以判据本体抽成两个纯函数（`tension_feedback_gain_per_s2`与
`assert_span_transport_directions`），注错用例把反号的偏导、
把对调了放线/收线的速率函数直接喂进去。

**这不是形式主义**：本仓在`cases/winding_line_endtoend`上把一个方向搞反过一次
（张力比给出0.5587而不是1.79）。方向反了的症状是**指数发散**，
不是一个偏了几个百分点的数——所以它要么当场被抓住，要么整个模型是废的。
本文件有一条门把那个发散**跑出来**：对调两端之后，同一组参数下张力从20 N
**涨到1.7e5 N**（2026-08-17实测），而正确方向下它在20 N附近振荡。

## 松弛那条裁决在这里被两面判

`FreeSpan.tension_n`在``ε ≤ 0``时**取零并给出``is_slack``**，
而`SpanTransportLoop`按显式声明的``forbid_slack``决定容不容忍。
两面各一条门：原语必须给零（不许抛），回路必须能关（不许静默）。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.drives import SpoolTension
from physics_engine.motion import PauseInterval
from physics_engine.transport import (
    MM_PER_M,
    FreeSpan,
    MaterialFeedTimeline,
    PayoutReel,
    SpanTransportLoop,
    TransportError,
    assert_span_transport_directions,
    assert_tension_feedback_is_negative,
    span_damping_ratio,
    span_natural_frequency_rad_s,
    steady_state_tension_n,
    tension_feedback_gain_per_s2,
    velocity_step_overshoot,
    velocity_step_peak_time_s,
)

SPAN = FreeSpan(span_id="span/unit", geometric_length_mm=300.0, axial_stiffness_n=60000.0)
REEL = PayoutReel(
    reel_id="reel/unit",
    radius_mm=60.0,
    inertia_kg_mm2=5000.0,
    bearing_damping_nmm_s=50.0,
)
BRAKE_NMM = 1200.0
LINE_SPEED = 20.0


def _timeline(**overrides) -> MaterialFeedTimeline:
    kwargs = {
        "source_id": "transport/unit",
        "times_s": (0.0, 0.002, 0.010),
        "feed_length_mm": (0.0, 0.04, 0.216),
        "rate_semantics": "piecewise_constant",
        "extrapolation": "reject",
        "pauses": (),
    }
    kwargs.update(overrides)
    return MaterialFeedTimeline(**kwargs)


# ---------------------------------------------------------------------------
# 方向门：本轨道的存在理由之一
# ---------------------------------------------------------------------------


def test_the_loop_gain_of_the_real_objects_is_negative():
    """真实对象上回路增益为负——负反馈。

    ``∂Ṫ/∂v_放线 = −K < 0``（多送让跨段松），
    ``∂v̇_放线/∂T = 1000·R²/J > 0``（离合器是**制动**，张力大盘转得快）。
    乘积为负。
    """

    length = SPAN.material_length_for_tension_mm(20.0)
    gain = tension_feedback_gain_per_s2(
        tension_rate_per_payout_speed_n_per_mm=-SPAN.stiffness_n_per_mm(length),
        payout_acceleration_per_tension_mm_s2_per_n=(
            REEL.payout_acceleration_per_tension_mm_s2_per_n()
        ),
    )
    assert gain < 0.0
    assert_tension_feedback_is_negative(gain)
    #: 增益的量值就是``−ω_n²``（连续系统的刚度项）——这条自洽把两条闭式钉在一起。
    natural = span_natural_frequency_rad_s(
        span_stiffness_n_per_mm=SPAN.stiffness_n_per_mm(length),
        radius_mm=REEL.radius_mm,
        inertia_kg_mm2=REEL.inertia_kg_mm2,
    )
    assert gain == pytest.approx(-(natural**2), rel=1e-14)


@pytest.mark.parametrize(
    ("first", "second", "what"),
    [
        (+200.0, +43.2, "∂Ṫ/∂v_放线写成了正的（多送反而绷紧）"),
        (-200.0, -43.2, "∂v̇_放线/∂T写成了负的（离合器被写成了驱动）"),
        (0.0, +43.2, "∂Ṫ/∂v_放线是零（张力与长度账脱钩，回到T=M/R）"),
    ],
)
def test_must_be_red_a_flipped_partial_makes_the_loop_positive_feedback(
    first, second, what
):
    """**必红**：任一偏导反号，回路增益就不再是负的。

    第三档（增益为零）同样必红：那正是`drives.SpoolTension`那个模型——
    张力与长度账脱钩，**扰动进不来也出不去**。零反馈与正反馈都不是负反馈，
    判据写的是``< 0``而不是``!= 正``，就是为了把这一档一起挡住。
    """

    gain = tension_feedback_gain_per_s2(
        tension_rate_per_payout_speed_n_per_mm=first,
        payout_acceleration_per_tension_mm_s2_per_n=second,
    )
    with pytest.raises(TransportError, match="正反馈"):
        assert_tension_feedback_is_negative(gain)
    assert what  # 参数名进报错行，便于定位


def test_the_span_direction_gate_passes_on_the_real_span():
    """三个符号在真`FreeSpan`上全对：等速恰为零、多送下降、多收上升。"""

    length = SPAN.material_length_for_tension_mm(20.0)
    assert_span_transport_directions(
        SPAN.tension_rate_n_per_s,
        material_length_mm=length,
        line_speed_mm_s=LINE_SPEED,
        probe_mm_s=1.0,
    )
    assert SPAN.tension_rate_n_per_s(
        material_length_mm=length,
        payout_speed_mm_s=LINE_SPEED,
        takeup_speed_mm_s=LINE_SPEED,
    ) == 0.0


def test_must_be_red_swapping_the_two_ends_flips_both_signs():
    """**必红**：把放线/收线对调着传进去，方向门当场抓住。

    这就是`winding_line_endtoend`那次（比值0.5587）的形态：
    **两端搞反不会报错，它给出一个看起来正常、方向相反的数。**
    """

    length = SPAN.material_length_for_tension_mm(20.0)

    def swapped(*, material_length_mm, payout_speed_mm_s, takeup_speed_mm_s):
        return SPAN.tension_rate_n_per_s(
            material_length_mm=material_length_mm,
            payout_speed_mm_s=takeup_speed_mm_s,
            takeup_speed_mm_s=payout_speed_mm_s,
        )

    with pytest.raises(TransportError, match="放线端多送时"):
        assert_span_transport_directions(
            swapped,
            material_length_mm=length,
            line_speed_mm_s=LINE_SPEED,
            probe_mm_s=1.0,
        )


def test_must_be_red_a_rate_that_ignores_the_takeup_end_is_caught():
    """**必红**：只看放线端的速率函数（收线端被算漏）。

    这一档在"等速"那条上就红了——一进一出没有抵消。
    **它比对调更隐蔽**：对调至少还是反对称的，漏一项连零点都没有。
    """

    length = SPAN.material_length_for_tension_mm(20.0)

    def half(*, material_length_mm, payout_speed_mm_s, takeup_speed_mm_s):
        return -SPAN.stiffness_n_per_mm(material_length_mm) * payout_speed_mm_s

    with pytest.raises(TransportError, match="不是恒等的0"):
        assert_span_transport_directions(
            half, material_length_mm=length, line_speed_mm_s=LINE_SPEED, probe_mm_s=1.0
        )


def test_must_be_red_the_swapped_model_runs_away_instead_of_oscillating():
    """**必红，而且是跑出来的**：两端对调之后系统**单调跑飞**。

    先说一句写这条门时被实测纠正的判断：**我先写下的是"指数发散到1.7e5 N"，
    而它是错的**。对调之后的反馈确实是正的，但收线端加速这个扰动把它推向的是
    **另一侧**——张力单调下降、放线盘被制动器越拖越慢，
    2026-08-17实测在第**7595**步（``t = 7.595 ms``）转速穿过零，
    被`PayoutReel`的``ω ≤ 0``那条失败关闭接住，此时张力已经从20 N掉到**8.593 N**。

    正确方向下，同一组参数跑60000步（60 ms）张力**有界**地在
    ``[19.281, 21.373]``之间振荡（``T*(22 mm/s) = 20.306``、
    闭式幅值``K·Δv/ω_n = 1.054``），一次都没有失败关闭。

    | | 单调？ | 60 ms内的张力 | 结局 |
    |---|---|---|---|
    | 正确 | 否（振荡） | 19.281 — 21.373 N | 跑完 |
    | 对调 | **是（单调降）** | 8.593 — 20.000 N | 7.595 ms时``ω ≤ 0``关闭 |

    **这一条是那道方向门有没有内容的证据**：一道判符号的门，
    若反过来的那一侧照样跑得好好的，说明符号根本不要紧。
    """

    length = SPAN.material_length_for_tension_mm(20.0)
    dt = 1.0e-6

    def integrate(*, swap: bool, steps: int):
        """返回``(停在第几步或None, 最低张力, 最高张力, 是否单调不增)``。"""

        material = length
        omega = LINE_SPEED / REEL.radius_mm
        takeup = LINE_SPEED * 1.1
        low = high = previous = SPAN.tension_n(material)
        monotone = True
        for index in range(steps):
            tension = SPAN.tension_n(material)
            low, high = min(low, tension), max(high, tension)
            monotone = monotone and tension <= previous
            previous = tension
            try:
                acceleration = REEL.angular_acceleration_rad_s2(
                    tension_n=tension,
                    brake_torque_nmm=BRAKE_NMM,
                    angular_velocity_rad_s=omega,
                )
            except TransportError:
                return index, low, high, monotone
            omega += dt * acceleration
            payout = omega * REEL.radius_mm
            material += dt * ((takeup - payout) if swap else (payout - takeup))
        return None, low, high, monotone

    stopped, low, high, monotone = integrate(swap=False, steps=60000)
    assert stopped is None, f"正确方向下第{stopped}步就失败关闭了"
    assert 18.5 < low < high < 22.5, f"正确方向下张力跑出了有界区间：[{low!r}, {high!r}]"
    assert not monotone, "正确方向下张力单调——那不是一个振荡，负反馈没有闭合"

    stopped, low, high, monotone = integrate(swap=True, steps=20000)
    assert stopped is not None, (
        "两端对调之后照样跑完了20000步 —— 若反过来的那一侧不会跑飞，"
        "方向门就没有内容"
    )
    assert stopped < 10000, f"对调之后拖到第{stopped}步才失控，跑飞得不够干脆"
    assert monotone, "对调之后张力不是单调的 —— 那说明它还在振荡而不是跑飞"
    assert low < 10.0, f"对调之后张力只掉到{low!r} N"


# ---------------------------------------------------------------------------
# 自由跨与松弛裁决
# ---------------------------------------------------------------------------


def test_the_tension_law_and_its_inverse_are_the_same_equation():
    """``T → L_mat → T``往返。2026-08-17实测四档最坏偏差3.908e-12 N。"""

    for tension in (0.0, 10.0, 20.0, 40.0, 100.0):
        length = SPAN.material_length_for_tension_mm(tension)
        assert SPAN.tension_n(length) == pytest.approx(tension, abs=1.0e-10)
    assert SPAN.material_length_for_tension_mm(0.0) == SPAN.geometric_length_mm


def test_slack_gives_zero_tension_and_says_so():
    """``ε ≤ 0``时**原语取零并给出``is_slack``**，不抛。

    裁决理由（模块docstring）：松弛是这个模型里的**可达状态**，
    对一个可达状态失败关闭等于让模型在它本该描述的那一刻炸掉；
    而静默取零会让"``L_geo``已经不是跨段长度"这件事看不见。
    **两个可读的事实，比一个异常或一个零更有用。**
    """

    slack = SPAN.geometric_length_mm * 1.001
    assert SPAN.strain(slack) < 0.0
    assert SPAN.is_slack(slack) is True
    assert SPAN.tension_n(slack) == 0.0
    #: 恰好等长是**边界**：应变恰为零，仍算松弛（``ε ≤ 0``），张力为零。
    assert SPAN.is_slack(SPAN.geometric_length_mm) is True
    assert SPAN.tension_n(SPAN.geometric_length_mm) == 0.0
    taut = SPAN.geometric_length_mm * 0.999
    assert SPAN.is_slack(taut) is False
    assert SPAN.tension_n(taut) > 0.0


def test_must_be_red_the_loop_closes_on_slack_when_it_was_told_to():
    """**必红**：``forbid_slack=True``时进入松弛当场关闭。

    构造：制动力矩调到很小，放线盘几乎不受阻，被带材拉着越转越快，
    材料堆进跨段——两步之内就松了。
    """

    loop = SpanTransportLoop(
        span=SPAN,
        reel=REEL,
        dt_s=1.0e-3,
        material_length_mm=SPAN.geometric_length_mm * 1.0001,
        angular_velocity_rad_s=LINE_SPEED / REEL.radius_mm,
        forbid_slack=True,
    )
    with pytest.raises(TransportError, match="跨段已经松了"):
        loop.step(brake_torque_nmm=1.0, takeup_speed_mm_s=LINE_SPEED)


def test_the_loop_tolerates_slack_when_it_was_told_to():
    """反面：``forbid_slack=False``时松弛照走，张力为零。

    **两面都要有门**：只判"关得住"，一个把``forbid_slack``读反的实现
    会在正常工况下炸而门是绿的。
    """

    loop = SpanTransportLoop(
        span=SPAN,
        reel=REEL,
        dt_s=1.0e-3,
        material_length_mm=SPAN.geometric_length_mm * 1.0001,
        angular_velocity_rad_s=LINE_SPEED / REEL.radius_mm,
        forbid_slack=False,
    )
    _, sample = loop.step(brake_torque_nmm=1.0, takeup_speed_mm_s=LINE_SPEED)
    assert sample.tension_n == 0.0
    assert sample.strain < 0.0


def test_must_be_red_forbid_slack_has_no_default_and_must_be_a_bool():
    """``forbid_slack``必须显式给、且必须是`bool`。

    与`TensionLoop.measurement_transfer`"没有默认值是有意的"同源：
    **把它做成默认值，等于让这条边界默默消失**。
    """

    with pytest.raises(TypeError):
        SpanTransportLoop(  # type: ignore[call-arg]
            span=SPAN,
            reel=REEL,
            dt_s=1.0e-4,
            material_length_mm=299.9,
            angular_velocity_rad_s=0.33,
        )
    with pytest.raises(TransportError, match="explicit bool"):
        SpanTransportLoop(
            span=SPAN,
            reel=REEL,
            dt_s=1.0e-4,
            material_length_mm=299.9,
            angular_velocity_rad_s=0.33,
            forbid_slack=1,  # type: ignore[arg-type]
        )


def test_the_span_stiffness_is_not_ea_over_the_geometric_length():
    """``K = EA·L_geo/L_mat²`` **不是** ``EA/L_geo``——差``(1+ε)²``。

    本工况``ε = 3.33e-4``⟹相对差``6.67e-4``，**比本案例超调判据的容差
    （3e-4）还大一倍**。拿``EA/L_geo``当K会把一个正确的实现判红。
    """

    tension = 20.0
    length = SPAN.material_length_for_tension_mm(tension)
    exact = SPAN.stiffness_n_per_mm(length)
    naive = SPAN.axial_stiffness_n / SPAN.geometric_length_mm
    strain = tension / SPAN.axial_stiffness_n
    assert exact / naive == pytest.approx((1.0 + strain) ** 2, rel=1e-12)
    assert exact / naive - 1.0 == pytest.approx(6.67e-4, rel=0.02)


# ---------------------------------------------------------------------------
# 放线盘
# ---------------------------------------------------------------------------


def test_the_brake_is_a_brake_not_a_motor():
    """张力升高 ⟹ 放线盘**加速**；制动力矩升高 ⟹ 放线盘**减速**。

    "张力升高让哪一端慢下来"在本模型里**没有答案**：收线端是伺服给定，
    放线端加速。慢下来的是跨段材料的净流失速率，那不是任何一端的速度。
    """

    omega = LINE_SPEED / REEL.radius_mm
    base = REEL.angular_acceleration_rad_s2(
        tension_n=20.0, brake_torque_nmm=BRAKE_NMM, angular_velocity_rad_s=omega
    )
    higher_tension = REEL.angular_acceleration_rad_s2(
        tension_n=21.0, brake_torque_nmm=BRAKE_NMM, angular_velocity_rad_s=omega
    )
    higher_brake = REEL.angular_acceleration_rad_s2(
        tension_n=20.0, brake_torque_nmm=BRAKE_NMM * 1.05, angular_velocity_rad_s=omega
    )
    assert higher_tension > base, "张力大了盘却没有转得更快——离合器被写成了驱动"
    assert higher_brake < base, "制动力矩大了盘却没有转得更慢"
    #: 单位：``α = 1000·(T·R − M − c·ω)/J``。
    assert base == pytest.approx(
        MM_PER_M
        * (20.0 * REEL.radius_mm - BRAKE_NMM - REEL.bearing_damping_nmm_s * omega)
        / REEL.inertia_kg_mm2,
        rel=1e-14,
    )


def test_must_be_red_a_stopped_or_reversing_reel_is_closed_out():
    """``ω ≤ 0``失败关闭——静摩擦保持段要一套return-map，本模块不假装能算。"""

    for omega in (0.0, -1.0e-12, -5.0):
        with pytest.raises(TransportError, match="静摩擦保持段"):
            REEL.angular_acceleration_rad_s2(
                tension_n=20.0, brake_torque_nmm=BRAKE_NMM, angular_velocity_rad_s=omega
            )


def test_must_be_red_a_negative_brake_torque_is_a_motor():
    with pytest.raises(TransportError, match="那不是磁粉离合器"):
        REEL.angular_acceleration_rad_s2(
            tension_n=20.0, brake_torque_nmm=-1.0, angular_velocity_rad_s=0.33
        )


def test_must_be_red_a_negative_bearing_damping_pumps_energy_in():
    with pytest.raises(TransportError, match="灌能量"):
        PayoutReel(
            reel_id="reel/bad",
            radius_mm=60.0,
            inertia_kg_mm2=5000.0,
            bearing_damping_nmm_s=-1.0,
        )


# ---------------------------------------------------------------------------
# 闭式与它的定义域
# ---------------------------------------------------------------------------


def test_the_zero_damping_steady_state_is_bit_for_bit_the_old_torque_over_radius():
    """``c = 0`` ⟹ 与`drives.SpoolTension.tension_n`**同一个浮点数**。"""

    spool = SpoolTension(barrel_radius_mm=REEL.radius_mm, tape_thickness_mm=0.1)
    for torque in (600.0, 1200.0, 1800.0):
        assert steady_state_tension_n(
            brake_torque_nmm=torque,
            radius_mm=REEL.radius_mm,
            bearing_damping_nmm_s=0.0,
            line_speed_mm_s=LINE_SPEED,
        ) == spool.tension_n(torque, 0.0)


def test_the_drives_feedforward_degenerates_to_the_old_torque_conversion():
    """`SpoolTension.brake_torque_for_tension_nmm`在``c = 0``时与`torque_nmm`逐位相同。

    有线速度、有轴承阻力矩时它**必须**给出更小的制动力矩——
    轴承已经替制动器出了``c·v/R``那一份。
    """

    spool = SpoolTension(barrel_radius_mm=REEL.radius_mm, tape_thickness_mm=0.1)
    for turns in (0.0, 12.0, 250.0):
        assert spool.brake_torque_for_tension_nmm(
            20.0, bearing_damping_nmm_s=0.0, line_speed_mm_s=LINE_SPEED, turns=turns
        ) == spool.torque_nmm(20.0, turns)
        with_bearing = spool.brake_torque_for_tension_nmm(
            20.0, bearing_damping_nmm_s=50.0, line_speed_mm_s=LINE_SPEED, turns=turns
        )
        assert with_bearing < spool.torque_nmm(20.0, turns)
        assert spool.torque_nmm(20.0, turns) - with_bearing == pytest.approx(
            50.0 * LINE_SPEED / spool.radius_mm(turns), rel=1e-14
        )


def test_must_be_red_a_bearing_that_already_holds_the_tension_is_closed_out():
    """轴承阻力矩本身超过所需时，制动力矩会是负的——失败关闭。"""

    spool = SpoolTension(barrel_radius_mm=60.0, tape_thickness_mm=0.1)
    with pytest.raises(Exception, match="关到底也太紧"):
        spool.brake_torque_for_tension_nmm(
            1.0, bearing_damping_nmm_s=5000.0, line_speed_mm_s=100.0
        )


def test_the_overshoot_closed_form_rejects_the_two_ends_of_its_domain():
    """``ζ = 0``相对超调无界、``ζ ≥ 1``没有峰值——两端都不是定义域。"""

    for ratio in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(TransportError):
            velocity_step_overshoot(ratio)
    assert velocity_step_overshoot(0.5) == pytest.approx(0.29843605919227484, rel=1e-14)


def test_the_peak_time_reduces_to_a_quarter_period_as_damping_vanishes():
    """``ζ → 0``时``t_p → (π/2)/ω_n``——无阻尼时``sin(ω_n t)``的第一个峰。"""

    natural = 379.6
    assert velocity_step_peak_time_s(
        natural_frequency_rad_s=natural, damping_ratio=0.0
    ) == pytest.approx((math.pi / 2.0) / natural, rel=1e-14)
    #: 标准二阶（无零点）是``π/ω_d``——本式在``ζ → 0``时恰是它的**一半**。
    assert velocity_step_peak_time_s(
        natural_frequency_rad_s=natural, damping_ratio=1.0e-9
    ) == pytest.approx(0.5 * math.pi / natural, rel=1e-8)


def test_the_damping_ratio_is_zero_exactly_when_the_bearing_is_ideal():
    """唯一的阻尼通道是轴承：``c = 0`` ⟹ ``ζ = 0``（无阻尼振子）。"""

    assert span_damping_ratio(
        span_stiffness_n_per_mm=200.0,
        radius_mm=60.0,
        inertia_kg_mm2=5000.0,
        bearing_damping_nmm_s=0.0,
    ) == 0.0
    #: 且``ζ``对``c``**严格单调**——这条比某一个具体值更难被错误实现凑对。
    ratios = [
        span_damping_ratio(
            span_stiffness_n_per_mm=200.0,
            radius_mm=60.0,
            inertia_kg_mm2=5000.0,
            bearing_damping_nmm_s=damping,
        )
        for damping in (10.0, 50.0, 200.0, 1000.0)
    ]
    assert ratios == sorted(ratios)
    assert ratios[-1] / ratios[0] == pytest.approx(100.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 上游时间线：形制门
# ---------------------------------------------------------------------------


def test_the_timeline_derivative_is_the_line_speed():
    timeline = _timeline()
    assert timeline.segment_speeds_mm_s() == pytest.approx((20.0, 22.0), rel=1e-15)
    assert timeline.horizon_s() == 0.010
    assert timeline.speed_mm_s(0.0) == pytest.approx(20.0, rel=1e-15)
    #: 样点处**右连续**：``t = 0.002``是第二段的起点。
    assert timeline.speed_mm_s(0.002) == pytest.approx(22.0, rel=1e-15)
    #: 终点**左连续**：没有第三段了，取最后一段的速率。
    assert timeline.speed_mm_s(0.010) == pytest.approx(22.0, rel=1e-15)
    assert timeline.length_mm(0.010) == 0.216
    assert timeline.length_mm(0.0) == 0.0


@pytest.mark.parametrize(
    ("overrides", "pattern", "what"),
    [
        (
            {"feed_length_mm": (0.0, 0.04, 0.03)},
            "monotone",
            "喂料长度回退＝带材被吸回去",
        ),
        (
            {"times_s": (0.001, 0.002, 0.010)},
            "run_start",
            "时间轴不从run_start=0起，两仓的0各说各的",
        ),
        (
            {"times_s": (0.0, 0.002, 0.002)},
            "strictly increasing",
            "时刻不严格递增，段长为零",
        ),
        (
            {"feed_length_mm": (0.0, 0.04)},
            "各说各的样点数",
            "两列长度不一致",
        ),
        (
            {"times_s": (0.0,), "feed_length_mm": (0.0,)},
            "至少要两个样点",
            "一个样点定义不出速率",
        ),
        (
            {"rate_semantics": "spline"},
            "rate_semantics",
            "没被声明过的速率语义",
        ),
        (
            {"extrapolation": "extend"},
            "extrapolation",
            "没被声明过的外推语义",
        ),
        (
            {"source_id": "feed/unit"},
            "transport/",
            "命名空间前缀不对",
        ),
    ],
)
def test_must_be_red_the_timeline_closes_on_a_malformed_declaration(
    overrides, pattern, what
):
    """**必红**：八种坏声明逐条失败关闭。"""

    with pytest.raises(TransportError, match=pattern):
        _timeline(**overrides)
    assert what


def test_must_be_red_a_zero_motion_segment_without_a_declared_pause():
    """**必红，本文件判据强度最高的形制门**：零增量段没有声明的暂停。

    **零运动段没有可推断的时长**——一段长度不变的采样与一段丢失的数据
    在字节上没有区别。上游WII为此专门发``pause_intervals``，
    而"上游保证过"与"本仓验过"不是同一件事。
    """

    with pytest.raises(TransportError, match="零增量"):
        _timeline(
            times_s=(0.0, 0.002, 0.004, 0.010),
            feed_length_mm=(0.0, 0.04, 0.04, 0.216),
        )
    #: 声明了就过。
    timeline = _timeline(
        times_s=(0.0, 0.002, 0.004, 0.010),
        feed_length_mm=(0.0, 0.04, 0.04, 0.216),
        pauses=(
            PauseInterval(
                pause_id="pause/transfer",
                start_time_s=0.002,
                end_time_s=0.004,
                reason="工件换手",
            ),
        ),
    )
    assert timeline.segment_speeds_mm_s()[1] == 0.0


def test_must_be_red_a_declared_pause_during_which_material_keeps_moving():
    """**必红，反方向**：声明为暂停、材料却还在走。

    只判正方向的话，一个把``pause_intervals``当装饰的上游可以随便贴，
    而下游会照着那条声明去跳过一段真在走的材料。
    """

    with pytest.raises(TransportError, match="自相矛盾"):
        _timeline(
            pauses=(
                PauseInterval(
                    pause_id="pause/bogus",
                    start_time_s=0.0,
                    end_time_s=0.002,
                    reason="声明得很像真的",
                ),
            )
        )


def test_must_be_red_a_pause_that_starts_mid_segment():
    """**必红**：暂停端点没有落在样点时刻上。

    段内恒速语义下，一个从段中间开始的暂停让这一段的速率无从定义——
    "这一段平均20 mm/s，其中后半段是停的"，那前半段是多少？没人知道。
    """

    with pytest.raises(TransportError, match="没有落在样点时刻上"):
        _timeline(
            times_s=(0.0, 0.002, 0.004, 0.010),
            feed_length_mm=(0.0, 0.04, 0.04, 0.216),
            pauses=(
                PauseInterval(
                    pause_id="pause/mid",
                    start_time_s=0.001,
                    end_time_s=0.004,
                    reason="从段中间开始",
                ),
            ),
        )


def test_must_be_red_extrapolation_reject_really_rejects():
    """``extrapolation='reject'``时样点之外必须拒；``'clamp'``时夹到边界。"""

    timeline = _timeline()
    for outside in (-1.0e-6, 0.0101):
        with pytest.raises(TransportError, match="reject"):
            timeline.speed_mm_s(outside)
    clamped = _timeline(extrapolation="clamp")
    assert clamped.speed_mm_s(-1.0) == pytest.approx(20.0, rel=1e-15)
    assert clamped.length_mm(99.0) == 0.216


# ---------------------------------------------------------------------------
# 推进器的形制门
# ---------------------------------------------------------------------------


def test_the_samples_are_head_of_step_snapshots():
    """样点是**步首**快照：第一条样点的时刻是0、状态是构造时的状态。"""

    loop = SpanTransportLoop.at_steady_state(
        span=SPAN, reel=REEL, dt_s=1.0e-5, brake_torque_nmm=BRAKE_NMM,
        line_speed_mm_s=LINE_SPEED, forbid_slack=True,
    )
    final, samples = loop.run(5, brake_torque_nmm=BRAKE_NMM, takeup_speed_mm_s=LINE_SPEED)
    assert samples[0].time_s == 0.0
    assert samples[0].material_length_mm == loop.material_length_mm
    assert samples[0].tension_n == loop.tension_n
    assert samples[0].payout_speed_mm_s == pytest.approx(LINE_SPEED, rel=1e-14)
    assert samples[-1].time_s == pytest.approx(4.0e-5, rel=1e-14)
    assert final.step_index == 5
    #: 原对象没有被就地改——历史一律显式传递（与`PidController`同源）。
    assert loop.step_index == 0


def test_must_be_red_the_steady_state_constructor_needs_a_turning_reel():
    with pytest.raises(TransportError, match="ω > 0"):
        SpanTransportLoop.at_steady_state(
            span=SPAN, reel=REEL, dt_s=1.0e-5, brake_torque_nmm=BRAKE_NMM,
            line_speed_mm_s=0.0, forbid_slack=True,
        )


def test_must_be_red_running_the_span_dry_is_closed_out():
    """材料被抽空（``L_mat ≤ 0``）失败关闭——那不是一个可以继续算的状态。"""

    loop = SpanTransportLoop(
        span=SPAN, reel=REEL, dt_s=1.0,
        material_length_mm=299.9, angular_velocity_rad_s=0.333,
        forbid_slack=False,
    )
    with pytest.raises(TransportError, match="材料被抽空"):
        loop.step(brake_torque_nmm=BRAKE_NMM, takeup_speed_mm_s=1.0e6)


def test_must_be_red_run_rejects_a_nonsense_step_count():
    loop = SpanTransportLoop.at_steady_state(
        span=SPAN, reel=REEL, dt_s=1.0e-5, brake_torque_nmm=BRAKE_NMM,
        line_speed_mm_s=LINE_SPEED, forbid_slack=True,
    )
    for steps in (0, -1, True, 2.0):
        with pytest.raises(TransportError, match="positive int"):
            loop.run(steps, brake_torque_nmm=BRAKE_NMM, takeup_speed_mm_s=LINE_SPEED)


def test_the_timeline_driven_run_needs_a_real_timeline():
    loop = SpanTransportLoop.at_steady_state(
        span=SPAN, reel=REEL, dt_s=1.0e-5, brake_torque_nmm=BRAKE_NMM,
        line_speed_mm_s=LINE_SPEED, forbid_slack=True,
    )
    with pytest.raises(TransportError, match="MaterialFeedTimeline"):
        loop.run_timeline(object(), brake_torque_nmm=BRAKE_NMM, steps=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("geometric_length_mm", 0.0, "positive"),
        ("geometric_length_mm", float("inf"), "positive"),
        ("axial_stiffness_n", -1.0, "positive"),
        ("span_id", "free", "span/"),
    ],
)
def test_must_be_red_the_span_closes_on_bad_declarations(field, value, pattern):
    kwargs = {
        "span_id": "span/unit",
        "geometric_length_mm": 300.0,
        "axial_stiffness_n": 60000.0,
    }
    kwargs[field] = value
    with pytest.raises(TransportError, match=pattern):
        FreeSpan(**kwargs)


def test_the_path_excess_enters_the_strain_numerator_and_nothing_else():
    """横向侵入让路径变长——**它进的是应变的分子，不是长度账的导数**。

    两条通道在推进器里各走各的：``path_excess_mm``改``L_path``（当场改张力），
    ``takeup_speed_mm_s``改``dL_mat/dt``（改的是张力的**导数**）。
    本门判两件事：路径增量当场把张力抬到``EA·(L_path − L_mat)/L_mat``，
    而**转速一点没动**（它是状态，不会因为路径变长而瞬变）。
    """

    loop = SpanTransportLoop.at_steady_state(
        span=SPAN, reel=REEL, dt_s=1.0e-6, brake_torque_nmm=BRAKE_NMM,
        line_speed_mm_s=LINE_SPEED, forbid_slack=True,
    )
    excess = 0.02
    _, plain = loop.step(brake_torque_nmm=BRAKE_NMM, takeup_speed_mm_s=LINE_SPEED)
    _, pushed = loop.step(
        brake_torque_nmm=BRAKE_NMM, takeup_speed_mm_s=LINE_SPEED, path_excess_mm=excess
    )
    assert pushed.path_excess_mm == excess
    assert plain.path_excess_mm == 0.0
    assert pushed.angular_velocity_rad_s == plain.angular_velocity_rad_s
    lift = SPAN.axial_stiffness_n * excess / loop.material_length_mm
    assert pushed.tension_n - plain.tension_n == pytest.approx(lift, rel=1.0e-12)


def test_must_be_red_the_span_refuses_a_negative_path_excess():
    """两个端点是世界系常量 ⟹ 路径只会**变长**（决策0071第二节）。"""

    loop = SpanTransportLoop.at_steady_state(
        span=SPAN, reel=REEL, dt_s=1.0e-6, brake_torque_nmm=BRAKE_NMM,
        line_speed_mm_s=LINE_SPEED, forbid_slack=True,
    )
    with pytest.raises(TransportError, match="path_excess_mm"):
        loop.step(
            brake_torque_nmm=BRAKE_NMM, takeup_speed_mm_s=LINE_SPEED, path_excess_mm=-1.0e-9
        )
    #: 零增量必须**逐位**退回不受扰的跨长——否则"没人碰它"这件事会带上一个偏差。
    assert SPAN.strain(299.9, path_excess_mm=0.0) == SPAN.strain(299.9)
    assert SPAN.stiffness_n_per_mm(299.9, path_excess_mm=0.0) == SPAN.stiffness_n_per_mm(299.9)


def test_the_span_stiffness_follows_the_disturbed_path_not_the_nominal_span():
    """路径被顶长之后，**跨段作为弹簧的刚度也跟着变**：``K = EA·L_path/L_mat²``。

    拿不受扰的``L_geo``当分子在``p/L_geo``这一档只差1e-4相对，
    **而它正好落在触碰那条判据要分辨的量级上**——所以这一条单独判。
    """

    material = 299.9
    excess = 0.03
    plain = SPAN.stiffness_n_per_mm(material)
    pushed = SPAN.stiffness_n_per_mm(material, path_excess_mm=excess)
    assert pushed > plain
    assert pushed == pytest.approx(
        SPAN.axial_stiffness_n * (SPAN.geometric_length_mm + excess) / (material * material),
        rel=1.0e-15,
    )
    assert pushed / plain == pytest.approx(
        1.0 + excess / SPAN.geometric_length_mm, rel=1.0e-15
    )


# ── 快路的前置条件（注错第二轮N1/N2/N5抓到的三个洞，decisions/0083第四节） ──


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (True, "must be a real number"),
        (False, "must be a real number"),
        ("20.0", "must be a real number"),
        (None, "must be a real number"),
        (float("nan"), "takeup_speed_mm_s must be finite"),
        (float("inf"), "takeup_speed_mm_s must be finite"),
        (float("-inf"), "takeup_speed_mm_s must be finite"),
    ],
)
def test_the_step_rejects_a_takeup_speed_that_is_not_a_finite_float(bad, message):
    """必红：`_require_finite`的类型门与有限性门，各自都要有东西钉着。

    **这条是`_require_finite`加快路之后补的**（decisions/0083第四节）。
    快路的前提是"类型已经确定是`float`"，而在补这条门之前，
    把类型门整个放开（``if True:``）或把有限性判去掉，**全套门一条都不红**——
    注错第二轮N1、N2实测各自119条全过。

    `bool`要单独列：``isinstance(True, int)``为真，**一个只判"是不是数"的
    实现会把`True`当成1.0放行**，而那正是本仓在别处反复钉的那类错。
    """

    loop = SpanTransportLoop.at_steady_state(
        span=SPAN,
        reel=REEL,
        dt_s=1.0e-4,
        brake_torque_nmm=1200.0,
        line_speed_mm_s=LINE_SPEED,
        forbid_slack=True,
    )
    #: **判到消息**，不只判"炸了"：`NaN`一路流下去也会在
    #: "材料长度推到nan"那条护栏上炸，于是一个把有限性判去掉的实现
    #: 在只判`TransportError`的门下照样绿（注错第二轮N2实测135条全过）。
    #: 判据要钉的是**哪一道关挡住了它**。
    with pytest.raises(TransportError, match=message):
        loop.step(brake_torque_nmm=1200.0, takeup_speed_mm_s=bad)


def test_slack_tolerance_survives_the_step_and_is_not_re_decided_each_frame():
    """必红：`forbid_slack`必须**被带过步**，不能在推进时被重新拍一遍。

    上面那条`test_the_loop_tolerates_slack_when_it_was_told_to`只走一步，
    判的是**步首**样点——于是一个在推进时把``forbid_slack``写死成`True`的实现
    在它下面是绿的（注错第二轮N5实测119条全过）。
    松弛是**跨步持续**的状态，所以门也必须跨步。
    """

    loop = SpanTransportLoop(
        span=SPAN,
        reel=REEL,
        dt_s=1.0e-3,
        material_length_mm=SPAN.geometric_length_mm * 1.0001,
        angular_velocity_rad_s=LINE_SPEED / REEL.radius_mm,
        forbid_slack=False,
    )
    tensions = []
    for _ in range(5):
        #: 收线端停住（``takeup = 0``）⟹ 材料只进不出、松弛只会更深。
        #: 取``2·LINE_SPEED``反而会把材料抽回绷紧，第3步就退出松弛（实测张力2.13 N）。
        loop, sample = loop.step(brake_torque_nmm=1.0, takeup_speed_mm_s=0.0)
        tensions.append(sample.tension_n)
    assert loop.forbid_slack is False, "推进把这条声明弄丢了"
    assert tensions == [0.0] * 5, f"松弛期张力不是零：{tensions!r}"
    #: 步长与声明也必须原样带过去——它们不是"每步重新决定"的东西。
    assert loop.dt_s == 1.0e-3
    assert loop.step_index == 5
