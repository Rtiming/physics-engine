"""扰动通道的门（决策0071，plans/15第三节阶段一的1.3与1.4）。

## 本文件的重心是**形制**，不是数

数在`cases/span_disturbance_channels`里判。本文件判的是三件别处判不了的：

1. **两条通道不许互相折算。** 路径增量``p``进的是应变的分子，收线端速度进的是
   长度账的导数。把``p``当成"一步内多收``p/dt``"会得到一个差``1/(1+ε)``的答案——
   本工况``ε ≈ 3.4e-4``，那个差**比本案例任何一条判据的容差都大**。
   有一条门把两条路各跑一步并断言它们**不相等**，且比值恰是``L_mat/L_path``；
2. **方向不许从符号里推。** `PenaltyAnnulusLimit`把法兰朝向编码在坐标符号里，
   单元门永远抓不到、端到端跑一次才炸（`winding_line_endtoend`案例页记着）。
   所以``offset_mm ≤ 0``当场关闭、``push_direction``必须是单位向量且与跨段正交，
   三条各一个必红；
3. **接错线要当场关闭。** `transport.FreeSpan`与`laydown.FreeSpanGeometry`
   拿着两条不同的跨段时，两边照样都跑得完，而算出来的张力没有任何东西说得清
   是哪一条跨段的。

## 那条不可达分支为什么也判

`_touch_window_indices`里"触碰窗口不连续"这一支，在``rectangular_hold``下
**构造上到不了**。0067第六节记过本仓的教训："**一条被别的门顺手拦下的必红，
说明不了这条门在起作用**"，而一条永远走不到的分支同样不是门。
所以本文件直接喂给它一串时刻乱序的样点——判的是这条分支真的在，
而不是它在今天这个剖面下用不上。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.disturbance import (
    AnalyticTakeup,
    ArmLaydownTakeup,
    ConstantTakeup,
    DisturbanceError,
    TransverseTouch,
    harmonic_tension_amplitude_n,
    path_step_tension_ring_n,
    ring_envelope_ratio,
    run_disturbed_span,
    tangent_turn_feed_rate_mm_s,
)
from physics_engine.disturbance import _touch_window_indices as _window
from physics_engine.laydown import FreeSpanGeometry
from physics_engine.transport import (
    FreeSpan,
    PayoutReel,
    SpanTransportLoop,
    SpanTransportSample,
    TransportError,
)

SPAN_LENGTH_MM = 300.0
ENTRY_MM = (1126.0, 0.0, 300.0)
GUIDE_MM = (1126.0 + SPAN_LENGTH_MM, 0.0, 300.0)

GEOMETRY = FreeSpanGeometry(
    span_id="span/free", guide_exit_mm=GUIDE_MM, entry_point_mm=ENTRY_MM
)
SPAN = FreeSpan(
    span_id="span/free", geometric_length_mm=SPAN_LENGTH_MM, axial_stiffness_n=60000.0
)
REEL = PayoutReel(
    reel_id="reel/payout", radius_mm=60.0, inertia_kg_mm2=5000.0, bearing_damping_nmm_s=50.0
)


def touch(**overrides) -> TransverseTouch:
    fields = {
        "touch_id": "touch/hand",
        "geometry": GEOMETRY,
        "profile": "rectangular_hold",
        "station_from_guide_mm": 150.0,
        "push_direction": (0.0, 0.0, 1.0),
        "start_time_s": 0.002,
        "end_time_s": 0.022,
        "offset_mm": 2.0,
    }
    fields.update(overrides)
    return TransverseTouch(**fields)


def loop_at(speed_mm_s: float, dt_s: float) -> SpanTransportLoop:
    return SpanTransportLoop.at_steady_state(
        span=SPAN,
        reel=REEL,
        dt_s=dt_s,
        brake_torque_nmm=1200.0,
        line_speed_mm_s=speed_mm_s,
        forbid_slack=True,
    )


# ------------------------------------------------------- 两条通道不许互相折算 ---


def test_a_path_excess_is_not_the_same_thing_as_an_extra_takeup_of_the_same_length():
    """**本文件最要紧的一条**：同样长的一段材料，两条通道给出的张力不一样。

    路径增量``p``进的是应变的分子：``δT = EA·p/L_mat``；
    "一步内多收``p``"进的是长度账：``δT = K·p = EA·L_path·p/L_mat²``。
    两者之比恰是``L_mat/L_path = 1/(1+ε)``——本工况差**3.4e-4相对**，
    而那**比本案例任何一条判据的容差都大**。

    折算掉这个因子在``ε → 0``时看不出来，而带材的工作应变正好是这个量级。
    """

    dt_s = 1.0e-6
    excess_mm = 0.02
    base = loop_at(20.0, dt_s)
    material = base.material_length_mm
    by_path = SPAN.tension_n(material, path_excess_mm=excess_mm) - SPAN.tension_n(material)
    by_length = SPAN.tension_n(material - excess_mm) - SPAN.tension_n(material)
    assert by_path != by_length
    #: 精确比值是``(L_mat − p)/L_geo``（两条都是代数恒等式，不是近似）。
    assert by_path / by_length == pytest.approx(
        (material - excess_mm) / SPAN.geometric_length_mm, rel=1.0e-12
    )
    #: ``p → 0``的极限是``L_mat/L_geo = 1/(1+ε)``——差额**相对3.4e-4**，
    #: 不是一个可以四舍五入掉的量。
    strain = SPAN.strain(material)
    assert material / SPAN.geometric_length_mm == pytest.approx(
        1.0 / (1.0 + strain), rel=1.0e-14
    )
    assert abs(material / SPAN.geometric_length_mm - 1.0) == pytest.approx(3.4e-4, rel=0.1)


def test_a_negative_path_excess_is_refused_because_the_two_ends_are_fixed():
    """跨段两端是世界系常量，所以路径只会**变长**。

    负增量意味着有人拿它当"跨长可以缩"的旋钮，而那正是plans/14第3.3节
    订正掉的那条错误因果（"臂动⟹跨段变长⟹张力变"）。
    """

    with pytest.raises(TransportError, match="path_excess_mm"):
        SPAN.strain(299.9, path_excess_mm=-0.01)
    with pytest.raises(TransportError, match="path_excess_mm"):
        SPAN.path_length_mm(-1.0)
    #: 零是合法的，而且``hypot(a, 0) = a``让它**逐位**回到不受扰的跨长。
    assert SPAN.path_length_mm(0.0) == SPAN.geometric_length_mm


# ------------------------------------------------------------- 接线那道门 ---


class _FakeModel:
    span = GEOMETRY


def test_the_two_modules_must_agree_on_which_free_span_they_are_holding():
    """`transport`的跨长与`laydown`两个端点之间的距离对不上⟹当场关闭。

    **这不是精度门是接线门**：接错了两边照样跑得完。
    """

    other = FreeSpan(
        span_id="span/other", geometric_length_mm=301.0, axial_stiffness_n=60000.0
    )
    model = _laydown_model()
    with pytest.raises(DisturbanceError, match="不是同一条自由跨段"):
        ArmLaydownTakeup(channel_id="takeup/arm", laydown=model, span=other)
    #: 对得上就放行。
    ArmLaydownTakeup(channel_id="takeup/arm", laydown=model, span=SPAN)


def test_the_laydown_channel_refuses_anything_that_is_not_a_laydown_model():
    with pytest.raises(DisturbanceError, match="LaydownModel"):
        ArmLaydownTakeup(channel_id="takeup/arm", laydown=_FakeModel(), span=SPAN)


def _laydown_model(account_rate_mm_s: float = 120.0):
    """一条最小的解析槽 + 反解位姿，只为把接线门跑通（数在案例里判）。

    ``account_rate_mm_s``让**送带账**可以故意与位姿不自洽——0067裁过
    "闭合的两条来源一般不自洽"，而只有在它们真的不一样的时候，
    "接的是哪一条"这件事才判得出来。
    """

    from physics_engine.laydown import (
        ArcRateProbe,
        CenterlineSemantics,
        FeedAccount,
        GrooveCenterline,
        GrooveStation,
        LaydownModel,
    )
    from physics_engine.motion import AnalyticPose, Pose

    radius = 60.0
    stations = []
    for index in range(65):
        alpha = 2.0 * math.pi * index / 64
        stations.append(
            GrooveStation(
                arc_length_mm=radius * alpha,
                position_mm=(radius * math.sin(alpha), -radius * math.cos(alpha), 0.0),
                tangent=(math.cos(alpha), math.sin(alpha), 0.0),
                width_direction=(-math.sin(alpha), math.cos(alpha), 0.0),
                surface_normal=(0.0, 0.0, 1.0),
            )
        )
    centerline = GrooveCenterline(
        centerline_id="groove/unit",
        stations=tuple(stations),
        semantics=CenterlineSemantics(
            position_interpolation="hermite_tangent",
            frame_interpolation="reorthonormalised_linear",
            topology="open",
            out_of_range="reject",
            nearest_refinement_iterations=2,
        ),
        length_unit="mm",
    )

    def pose_fn(t_s: float) -> Pose:
        alpha = 0.4 + 2.0 * t_s
        point = (radius * math.sin(alpha), -radius * math.cos(alpha), 0.0)
        cos_a, sin_a = math.cos(-alpha), math.sin(-alpha)
        rotated = (
            point[0] * cos_a - point[1] * sin_a,
            point[0] * sin_a + point[1] * cos_a,
            0.0,
        )
        return Pose(
            translation_mm=tuple(ENTRY_MM[axis] - rotated[axis] for axis in range(3)),
            rotation_xyzw=(0.0, 0.0, math.sin(-0.5 * alpha), math.cos(-0.5 * alpha)),
        )

    return LaydownModel(
        model_id="laydown/unit",
        motion=AnalyticPose(
            source_id="motion/unit",
            pose_fn=pose_fn,
            declared_horizon_s=1.0,
            extrapolation="reject",
            replayable=True,
            replay_probe_times_s=(0.0, 0.5),
        ),
        centerline=centerline,
        span=GEOMETRY,
        feed=FeedAccount(
            account_id="feed/unit",
            length_fn=lambda t_s: account_rate_mm_s * t_s,
            probe_times_s=(0.0, 0.5),
        ),
        arc_origin_mm=60.0 * 0.4,
        rate_probe=ArcRateProbe(scheme="central", step_s=1.0e-4),
    )


def test_the_arm_channel_reports_the_feed_rate_gap_instead_of_fixing_it():
    """位姿要的速率与送带账给的速率之差是**要报的量不是要用的量**（0067第二节）。"""

    channel = ArmLaydownTakeup(channel_id="takeup/arm", laydown=_laydown_model(), span=SPAN)
    #: 本夹具的圆槽 + 匀速转 ⟹ 两条速率都是120 mm/s。
    #: 这里只用64站点（**单元门要快**），所以容差按那一档给：
    #: 实测1.55e-7相对，案例里那条256站点的门判到1e-6。
    assert channel.takeup_speed_mm_s(0.1) == pytest.approx(120.0, rel=1.0e-6)
    assert abs(channel.feed_rate_gap_mm_s(0.1)) < 1.0e-4
    assert channel.groove_tangent_turn_rate_rad_s(0.1) == pytest.approx(2.0, rel=1.0e-3)
    #: 理想绕线下世界系切向不动——**这一条正是"槽切向转多快取工件系"那个选择的理由**。
    #: 64站点档实测1.3e-3 rad/s，对旋钮值2.0是6.5e-4相对；案例那条256站点的门判1e-3绝对。
    assert abs(channel.world_tangent_turn_rate_rad_s(0.1)) < 2.0e-3


def test_the_arm_channel_takes_the_pose_side_when_the_two_sources_disagree():
    """**闭合的两条来源不自洽时，收线端接的必须是位姿那一条。**

    "跨段里的材料被消耗得多快"是一件**物理**：线圈转过去了材料就被带走了，
    送带账同不同意都一样。送带账是**上游的声明**，两者之差是闭合残差在速率上的
    那一面——0067裁的是"两条各算一次、把差额报出来，不挑哪一条当成对的"，
    而本层要用的那一条**是位姿定的**。

    本门把送带账故意调快10%。闭合恒成立的算例上**这条门是空的**
    （两条速率一样，接哪一条都对）——那正是2026-08-17注错验证抓出来的一个空门。
    """

    honest = ArmLaydownTakeup(
        channel_id="takeup/arm", laydown=_laydown_model(120.0), span=SPAN
    )
    lying = ArmLaydownTakeup(
        channel_id="takeup/arm", laydown=_laydown_model(132.0), span=SPAN
    )
    #: 位姿没变 ⟹ 所需送带率一个字都不该动。
    assert lying.takeup_speed_mm_s(0.1) == honest.takeup_speed_mm_s(0.1)
    assert lying.takeup_speed_mm_s(0.1) == pytest.approx(120.0, rel=1.0e-6)
    #: 而差额被**报出来**：位姿要120、账给132，缺口−12 mm/s。
    assert lying.feed_rate_gap_mm_s(0.1) == pytest.approx(-12.0, rel=1.0e-5)
    assert abs(honest.feed_rate_gap_mm_s(0.1)) < 1.0e-4


# ------------------------------------------- 触碰：方向、量值、作用点、起止 ---


def test_the_offset_carries_no_direction_so_a_non_positive_one_is_refused():
    """**这一条是`PenaltyAnnulusLimit`那次教训的直接产物。**

    方向由``push_direction``携带，量值不兼职表示方向。负位移与零位移都关闭。
    """

    for bad in (-2.0, 0.0):
        with pytest.raises(DisturbanceError, match="必须为正"):
            touch(offset_mm=bad)


def test_the_push_direction_must_be_a_unit_vector_and_perpendicular_to_the_span():
    with pytest.raises(DisturbanceError, match="不是单位向量"):
        touch(push_direction=(0.0, 0.0, 1.4))
    with pytest.raises(DisturbanceError, match="不正交"):
        touch(push_direction=(1.0, 0.0, 0.0))
    #: 只要正交且单位，哪个横向都行——**方向是一个真的自由度不是一个符号**。
    assert touch(push_direction=(0.0, 1.0, 0.0)).path_excess_mm(0.01) == pytest.approx(
        touch().path_excess_mm(0.01), rel=1.0e-15
    )


def test_the_contact_point_moves_with_the_declared_direction():
    """反向推 ⟹ 接触点落在跨段的另一侧。**位置由方向字段定，不由任何符号推。**"""

    up = touch().contact_point_mm(0.01)
    down = touch(push_direction=(0.0, 0.0, -1.0)).contact_point_mm(0.01)
    assert up[2] - down[2] == pytest.approx(4.0, rel=1.0e-12)
    #: 沿跨段的坐标两者相同：方向只改横向那一份。
    assert up[0] == pytest.approx(down[0], rel=1.0e-15)
    #: 窗口之外接触点落在**没有被按开的**跨段上。
    assert touch().contact_point_mm(0.0)[2] == pytest.approx(ENTRY_MM[2], rel=1.0e-15)


def test_the_station_must_be_strictly_inside_the_span():
    for bad in (0.0, SPAN_LENGTH_MM, -1.0, SPAN_LENGTH_MM + 1.0):
        with pytest.raises(DisturbanceError, match="不在跨段"):
            touch(station_from_guide_mm=bad)


def test_the_window_is_half_open_and_must_have_positive_length():
    subject = touch()
    assert not subject.is_active(0.002 - 1.0e-12)
    assert subject.is_active(0.002)
    assert subject.is_active(0.022 - 1.0e-12)
    #: **右端开**：闭右端会让最后一步既在窗口内又在窗口外，而冲量恒等式
    #: 是按窗口逐步求和的。
    assert not subject.is_active(0.022)
    for start, end in ((0.01, 0.01), (0.02, 0.01)):
        with pytest.raises(DisturbanceError, match="不是正长度"):
            touch(start_time_s=start, end_time_s=end)
    with pytest.raises(DisturbanceError, match="start_time_s"):
        touch(start_time_s=-1.0, end_time_s=1.0)


def test_the_time_profile_has_no_default_and_only_one_declared_member():
    with pytest.raises(DisturbanceError, match="profile must be one of"):
        touch(profile="half_sine")


def test_the_path_excess_is_the_exact_polyline_not_the_small_angle_form():
    """精确式与小角度式在δ=4 mm处差**1.777e-4相对**——拿后者当实现会被判红。"""

    subject = touch(offset_mm=4.0)
    exact = subject.path_excess_for_offset_mm(4.0)
    small = 4.0 * 4.0 * SPAN_LENGTH_MM / (2.0 * 150.0 * 150.0)
    assert exact < small
    assert exact / small - 1.0 == pytest.approx(-1.7771e-4, rel=1.0e-3)
    #: 窗口之外恰为零（``hypot(a, 0) = a``逐位成立）。
    assert subject.path_excess_mm(0.0) == 0.0
    assert subject.force_geometry_factor(0.0) == 0.0


def test_the_middle_of_the_span_is_the_least_sensitive_place_to_push():
    """``p ∝ L/(2ab)``而``ab``在中点最大 ⟹ **中点是路径增量最小的地方**。

    "中间按下去最狠"是直觉，而直觉在这里是反的。
    """

    centre = touch(station_from_guide_mm=150.0).path_excess_for_offset_mm(2.0)
    offcentre = touch(station_from_guide_mm=75.0).path_excess_for_offset_mm(2.0)
    assert centre < offcentre
    assert centre / offcentre == pytest.approx(0.75, rel=1.0e-3)


def test_the_transverse_force_is_an_output_of_the_tension_not_an_input():
    subject = touch()
    factor = subject.force_geometry_factor(0.01)
    assert subject.transverse_force_n(0.01, tension_n=20.0) == pytest.approx(
        20.0 * factor, rel=1.0e-15
    )
    #: 窗口之外没有手指，也就没有力。
    assert subject.transverse_force_n(0.0, tension_n=20.0) == 0.0


# ----------------------------------------------------------- 收线端速度源 ---


def test_an_impure_takeup_is_falsified_at_construction():
    counter = {"n": 0}

    def impure(_t_s: float) -> float:
        counter["n"] += 1
        return float(counter["n"])

    with pytest.raises(DisturbanceError, match="不是纯函数"):
        AnalyticTakeup(channel_id="takeup/impure", speed_fn=impure, probe_times_s=(0.0,))
    with pytest.raises(DisturbanceError, match="至少要给一个探针时刻"):
        AnalyticTakeup(channel_id="takeup/none", speed_fn=lambda t: 1.0, probe_times_s=())


def test_the_driver_refuses_a_bare_callable_as_a_takeup_source():
    """收线端速度是1.3那条扰动通道的**类型**，不许随手传一个裸函数。"""

    with pytest.raises(DisturbanceError, match="takeup_speed_mm_s"):
        run_disturbed_span(
            loop_at(20.0, 1.0e-6),
            steps=1,
            brake_torque_nmm=1200.0,
            takeup=lambda t_s: 20.0,
        )


def test_the_driver_refuses_a_non_loop_and_a_non_touch():
    source = ConstantTakeup(channel_id="takeup/const", speed_mm_s=20.0)
    with pytest.raises(DisturbanceError, match="SpanTransportLoop"):
        run_disturbed_span(object(), steps=1, brake_torque_nmm=1200.0, takeup=source)
    with pytest.raises(DisturbanceError, match="TransverseTouch"):
        run_disturbed_span(
            loop_at(20.0, 1.0e-6),
            steps=1,
            brake_torque_nmm=1200.0,
            takeup=source,
            touch=object(),
        )
    with pytest.raises(DisturbanceError, match="positive int"):
        run_disturbed_span(
            loop_at(20.0, 1.0e-6), steps=0, brake_torque_nmm=1200.0, takeup=source
        )


def test_a_non_contiguous_touch_window_is_refused():
    """构造上到不了的分支也要被走过一次——0067第六节那条教训。"""

    def sample(time_s: float) -> SpanTransportSample:
        return SpanTransportSample(
            time_s=time_s,
            material_length_mm=299.9,
            tension_n=20.0,
            strain=3.0e-4,
            angular_velocity_rad_s=1.0 / 3.0,
            payout_speed_mm_s=20.0,
            takeup_speed_mm_s=20.0,
            brake_torque_nmm=1200.0,
        )

    scrambled = (sample(0.003), sample(0.0), sample(0.004))
    with pytest.raises(DisturbanceError, match="不连续"):
        _window(scrambled, touch())
    #: 顺序正常时它给出``[首, 末+1)``。
    assert _window((sample(0.0), sample(0.003), sample(0.03)), touch()) == (1, 2)
    #: 一步都没碰到时给出空窗口而不是报错——**没碰就是没碰**。
    assert _window((sample(0.0), sample(0.001)), touch()) == (0, 0)


# ------------------------------------------------------------ 闭式的定义域 ---


def test_the_mechanism_formula_refuses_a_straight_segment():
    assert tangent_turn_feed_rate_mm_s(
        tangent_turn_rate_rad_s=2.0, curvature_per_mm=1.0 / 60.0
    ) == pytest.approx(120.0, rel=1.0e-15)
    with pytest.raises(DisturbanceError, match="必须为正"):
        tangent_turn_feed_rate_mm_s(tangent_turn_rate_rad_s=2.0, curvature_per_mm=0.0)


def test_the_harmonic_amplitude_is_linear_in_the_ripple_and_rejects_bad_damping():
    common = {
        "span_stiffness_n_per_mm": 200.0,
        "forcing_rad_s": 4.0,
        "natural_frequency_rad_s": 380.0,
        "damping_ratio": 0.0132,
    }
    one = harmonic_tension_amplitude_n(takeup_amplitude_mm_s=1.0, **common)
    ten = harmonic_tension_amplitude_n(takeup_amplitude_mm_s=10.0, **common)
    #: **线性**——单调判据的全部理由就在这一条上。
    assert ten / one == pytest.approx(10.0, rel=1.0e-14)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(DisturbanceError, match="damping ratio"):
            harmonic_tension_amplitude_n(takeup_amplitude_mm_s=1.0, **{**common, "damping_ratio": bad})


def test_the_ring_starts_at_the_step_and_the_envelope_starts_at_one():
    assert path_step_tension_ring_n(
        step_tension_n=5.0, natural_frequency_rad_s=380.0, damping_ratio=0.0132, time_s=0.0
    ) == 5.0
    assert (
        ring_envelope_ratio(
            natural_frequency_rad_s=380.0, damping_ratio=0.0132, elapsed_s=0.0
        )
        == 1.0
    )
    with pytest.raises(DisturbanceError, match="不能为负"):
        ring_envelope_ratio(
            natural_frequency_rad_s=380.0, damping_ratio=0.0132, elapsed_s=-1.0
        )


def test_the_ring_is_not_the_velocity_step_response():
    """**同一个二阶系统，两条完全不同的响应。**

    路径阶跃：张力当场跳、速度连续 ⟹ ``δT(0) = ΔT₀``、``δT'(0) = 0``；
    速度阶跃（0066）：张力连续、速度差当场跳 ⟹ 初值带一个斜率冲击。
    两条的峰值时刻差``acos ζ``、超调差一个``1/(2ζ)``——**套错了不会报错，只会错**。
    """

    from physics_engine.transport import velocity_step_peak_time_s

    natural, ratio = 380.0, 0.2
    #: 路径阶跃的峰值就在``t = 0``（此后单调包络下降），
    #: 而速度阶跃的峰值在``(π − acos ζ)/ω_d``——两者不可能是同一条。
    peak = velocity_step_peak_time_s(natural_frequency_rad_s=natural, damping_ratio=ratio)
    assert peak > 0.0
    at_zero = path_step_tension_ring_n(
        step_tension_n=1.0, natural_frequency_rad_s=natural, damping_ratio=ratio, time_s=0.0
    )
    later = max(
        path_step_tension_ring_n(
            step_tension_n=1.0,
            natural_frequency_rad_s=natural,
            damping_ratio=ratio,
            time_s=0.001 * index,
        )
        for index in range(1, 40)
    )
    assert later <= at_zero * (1.0 + 1.0e-3)
