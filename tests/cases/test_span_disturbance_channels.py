"""conformance：臂动与人手触碰两条扰动通道（`cases/span_disturbance_channels`）。

守plans/15第三节阶段一的1.3与1.4，决策0071。

## 这条案例回答的是"**扰动从哪儿进来**"

0066把张力做成了"放线端与收线端速度差经带材弹性生成的量"，
于是控制器第一次有东西可控。但那一轮只能造一个**收线端速度阶跃**当扰动，
而真机上没有人去阶跃收线端——真机上的扰动有两个：

1. **臂动**。落位点必须待在那个世界系固定的入带点上，所以这一瞬要放多少带材
   由线圈的运动定：``σ' = Ω_tan/κ``。**槽的曲率沿路径一变，同样一条匀速转切向的
   臂运动就把送带率变成了一条会动的曲线**——那就是扰动；
2. **人手突然碰一下带材**。两个端点仍固定，被顶开的是中间：直线段变折线，
   **折线比直线长**，路径一长张力当场跳。

## ``L_geo``到底变不变——这一条本案例正面判掉

0066第九、十节把跨段几何长度取常数登记成欠账，措辞是"真机上跨长逐样点变"。
**那句措辞是错的**（决策0071第二节）：plans/14第3.3节订正后的场景里，
自由跨的两个端点都是世界系固定的，两点之间的直线段长度**是常数**。

所以本案例的两条通道**形制上就是分开的**：臂动走收线端速度、
触碰走路径增量，而且有一条门判它们**不许互相折算**
（差一个``L_mat/L_path``即3.4e-4相对，在`tests/test_disturbance.py`里）。

## 那个必须有的退化档

**平面圆 + 匀速转切向 ⟹ 送带率恒定 ⟹ 扰动恒为零。**
"恒为零"在**输入**那一侧是逐位成立的（所需送带率是常数，方差恰为0）；
在**输出**那一侧**做不到逐位**——本条最初写的就是零容差，**实测当场否掉**：
``c = 0``档20000步累出2.2752e-11 N、``c = 50``档1.1376e-11 N，
病根是``at_steady_state``那一趟``T → L_mat → T``往返的几个ulp。
**两档并排列，是为了让"零容差做不到"这件事有两个独立的数支撑**（案例页第三节）。

## 本案例做不到的那一半

"触碰→尖峰→**回落**"只做到前两段。开环下尖峰**不回落**：包络按
``exp(−ζω_n t)``走，真实量级轴承给``ζ = 0.0132``，四个阻尼周期之后还剩**71.8%**。
"有控制器时一个周期之内被压掉"那一半归轨道E（决策0070），
本案例**一条都没有做**，如实登记在案例页第四节。
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from physics_engine.disturbance import (
    AnalyticTakeup,
    ArmLaydownTakeup,
    ConstantTakeup,
    TransverseTouch,
    harmonic_tension_amplitude_n,
    path_step_tension_ring_n,
    ring_envelope_ratio,
    run_disturbed_span,
    tangent_turn_feed_rate_mm_s,
)
from physics_engine.laydown import (
    ArcRateProbe,
    CenterlineSemantics,
    FeedAccount,
    FreeSpanGeometry,
    GrooveCenterline,
    GrooveStation,
    LaydownModel,
)
from physics_engine.motion import AnalyticPose, Pose
from physics_engine.oracles import load_manifest
from physics_engine.transport import (
    FreeSpan,
    PayoutReel,
    SpanTransportLoop,
    span_damping_ratio,
    span_natural_frequency_rad_s,
    steady_state_tension_n,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "span_disturbance_channels"
MANIFEST = load_manifest(CASE / "oracle.json")

#: 全部**假设输入**，与`cases/free_span_tension_step`逐条同值（轴承阻力矩取真实量级50）。
AXIAL_STIFFNESS_N = 60000.0
GEOMETRIC_LENGTH_MM = 300.0
REEL_RADIUS_MM = 60.0
REEL_INERTIA_KG_MM2 = 5000.0
BRAKE_TORQUE_NMM = 1200.0
BEARING_DAMPING_NMM_S = 50.0

BASE_RADIUS_MM = 60.0
BASE_CURVATURE_PER_MM = 1.0 / BASE_RADIUS_MM
CURVATURE_MODULATION = 0.3
ENTRY_POINT_MM = (1126.0, 0.0, 300.0)
GUIDE_EXIT_MM = (1126.0 + GEOMETRIC_LENGTH_MM, 0.0, 300.0)

TANGENT_TURN_RATES_RAD_S = (1.0, 2.0, 4.0, 8.0)
SWEEP_DT_S = 1.0e-4
WIRING_RATE_RAD_S = 2.0
WIRING_ALPHA0_RAD = 0.4
WIRING_STATIONS = 256
WIRING_ALPHA_MAX_RAD = 2.0 * math.pi
WIRING_PROBE_S = 1.0e-4
WIRING_STEPS = 1500
WIRING_PROBE_TIMES_S = (0.02, 0.05, 0.10, 0.14)

TOUCH_LINE_SPEED_MM_S = 20.0
TOUCH_STATION_MM = 150.0
TOUCH_OFFCENTRE_STATION_MM = 75.0
TOUCH_OFFSET_MM = 2.0
TOUCH_OFFSET_SWEEP_MM = (0.5, 1.0, 2.0, 4.0)
TOUCH_START_S = 0.002
TOUCH_END_S = 0.022
TOUCH_DT_S = 2.0e-6
TOUCH_STEPS = 50000
RING_PERIODS = (1, 2, 3, 4)

SPAN = FreeSpan(
    span_id="span/free",
    geometric_length_mm=GEOMETRIC_LENGTH_MM,
    axial_stiffness_n=AXIAL_STIFFNESS_N,
)
GEOMETRY = FreeSpanGeometry(
    span_id="span/free", guide_exit_mm=GUIDE_EXIT_MM, entry_point_mm=ENTRY_POINT_MM
)


def _oracle(oracle_id: str):
    for entry in MANIFEST.oracles:
        if entry.id == oracle_id:
            return entry
    raise AssertionError(f"清单里没有{oracle_id}")


def _reel(damping_nmm_s: float = BEARING_DAMPING_NMM_S) -> PayoutReel:
    return PayoutReel(
        reel_id="reel/payout",
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
        bearing_damping_nmm_s=damping_nmm_s,
    )


def _modal(speed_mm_s: float, damping_nmm_s: float = BEARING_DAMPING_NMM_S):
    """稳态张力、材料长度、跨段刚度、``ω_n``、``ζ``——全部走引擎的闭式入口。"""

    tension = steady_state_tension_n(
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        radius_mm=REEL_RADIUS_MM,
        bearing_damping_nmm_s=damping_nmm_s,
        line_speed_mm_s=speed_mm_s,
    )
    material = SPAN.material_length_for_tension_mm(tension)
    stiffness = SPAN.stiffness_n_per_mm(material)
    natural = span_natural_frequency_rad_s(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
    )
    ratio = span_damping_ratio(
        span_stiffness_n_per_mm=stiffness,
        radius_mm=REEL_RADIUS_MM,
        inertia_kg_mm2=REEL_INERTIA_KG_MM2,
        bearing_damping_nmm_s=damping_nmm_s,
    )
    return tension, material, stiffness, natural, ratio


# ---------------------------------------------------------------------------
# 解析件：用切向角当参数的平面槽，以及"姿态选定、平移反解"的位姿
# ---------------------------------------------------------------------------


def groove_arc_mm(alpha: float, modulation: float) -> float:
    return (alpha + modulation * math.sin(alpha)) / BASE_CURVATURE_PER_MM


def groove_position_mm(alpha: float, modulation: float):
    return (
        (math.sin(alpha) + modulation * (0.5 * alpha + 0.25 * math.sin(2.0 * alpha)))
        / BASE_CURVATURE_PER_MM,
        (-math.cos(alpha) + modulation * 0.5 * math.sin(alpha) ** 2) / BASE_CURVATURE_PER_MM,
        0.0,
    )


def groove_frame(alpha: float):
    """``(t, s, n)``。约定抄GCW：``s = n × t``、右手系（plans/14第二节）。"""

    return (
        (math.cos(alpha), math.sin(alpha), 0.0),
        (-math.sin(alpha), math.cos(alpha), 0.0),
        (0.0, 0.0, 1.0),
    )


def analytic_feed_rate_mm_s(t_s: float, turn_rate: float, alpha0: float, modulation: float):
    """``σ' = Ω/κ(α)``——**机制本体**，与`disturbance`那条纯函数互为对拍。"""

    alpha = alpha0 + turn_rate * t_s
    return turn_rate * (1.0 + modulation * math.cos(alpha)) / BASE_CURVATURE_PER_MM


def centerline(modulation: float, stations: int = WIRING_STATIONS) -> GrooveCenterline:
    rows = []
    for index in range(stations + 1):
        alpha = WIRING_ALPHA_MAX_RAD * index / stations
        tangent, width, normal = groove_frame(alpha)
        rows.append(
            GrooveStation(
                arc_length_mm=groove_arc_mm(alpha, modulation),
                position_mm=groove_position_mm(alpha, modulation),
                tangent=tangent,
                width_direction=width,
                surface_normal=normal,
            )
        )
    return GrooveCenterline(
        centerline_id=f"groove/tangent_parametrised_{stations}",
        stations=tuple(rows),
        semantics=CenterlineSemantics(
            position_interpolation="hermite_tangent",
            frame_interpolation="reorthonormalised_linear",
            topology="open",
            out_of_range="reject",
            nearest_refinement_iterations=2,
        ),
        length_unit="mm",
    )


def arm_pose(turn_rate: float, alpha0: float, modulation: float, horizon_s: float):
    """姿态取``Rz(−α)``、平移反解 ⟹ **闭合是构造出来的**（形制抄0067）。

    ``Rz(−α)``下世界系槽切向恒为``(1,0,0)``，正是入射角理想值0那个条件。
    """

    def pose_fn(t_s: float) -> Pose:
        alpha = alpha0 + turn_rate * t_s
        point = groove_position_mm(alpha, modulation)
        cos_a, sin_a = math.cos(-alpha), math.sin(-alpha)
        rotated = (
            point[0] * cos_a - point[1] * sin_a,
            point[0] * sin_a + point[1] * cos_a,
            point[2],
        )
        return Pose(
            translation_mm=tuple(ENTRY_POINT_MM[axis] - rotated[axis] for axis in range(3)),
            rotation_xyzw=(0.0, 0.0, math.sin(-0.5 * alpha), math.cos(-0.5 * alpha)),
        )

    return AnalyticPose(
        source_id="motion/arm_tangent_sweep",
        pose_fn=pose_fn,
        declared_horizon_s=horizon_s,
        extrapolation="reject",
        replayable=True,
        replay_probe_times_s=(0.0, 0.5 * horizon_s),
    )


def laydown_model(turn_rate: float, alpha0: float, modulation: float, horizon_s: float):
    return LaydownModel(
        model_id="laydown/arm_tangent_sweep",
        motion=arm_pose(turn_rate, alpha0, modulation, horizon_s),
        centerline=centerline(modulation),
        span=GEOMETRY,
        feed=FeedAccount(
            account_id="feed/arm_tangent_sweep",
            length_fn=lambda t_s: groove_arc_mm(alpha0 + turn_rate * t_s, modulation)
            - groove_arc_mm(alpha0, modulation),
            probe_times_s=(0.0, 0.5 * horizon_s),
        ),
        arc_origin_mm=groove_arc_mm(alpha0, modulation),
        rate_probe=ArcRateProbe(scheme="central", step_s=WIRING_PROBE_S),
    )


def steady_loop(speed_mm_s: float, dt_s: float, *, damping_nmm_s=BEARING_DAMPING_NMM_S,
                step_index: int = 0) -> SpanTransportLoop:
    loop = SpanTransportLoop.at_steady_state(
        span=SPAN,
        reel=_reel(damping_nmm_s),
        dt_s=dt_s,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        line_speed_mm_s=speed_mm_s,
        forbid_slack=True,
    )
    #: 中心差分的速率探针要取``t ± h``，所以受`laydown`驱动的run**不能从``t = 0``起步**
    #: ——`laydown`那一层明写着"本层不夹到端点"。这里从第一步起算。
    return dataclasses.replace(loop, step_index=step_index) if step_index else loop


THE_TOUCH = TransverseTouch(
    touch_id="touch/hand",
    geometry=GEOMETRY,
    profile="rectangular_hold",
    station_from_guide_mm=TOUCH_STATION_MM,
    push_direction=(0.0, 0.0, 1.0),
    start_time_s=TOUCH_START_S,
    end_time_s=TOUCH_END_S,
    offset_mm=TOUCH_OFFSET_MM,
)


@pytest.fixture(scope="module")
def touch_run():
    """一次触碰跑满0.1秒：尖峰门、振铃门、三本账各一条门共用。"""

    return run_disturbed_span(
        steady_loop(TOUCH_LINE_SPEED_MM_S, TOUCH_DT_S),
        steps=TOUCH_STEPS,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        takeup=ConstantTakeup(channel_id="takeup/servo", speed_mm_s=TOUCH_LINE_SPEED_MM_S),
        touch=THE_TOUCH,
    )


# ---------------------------------------------------------------------------
# 1.3 机制与接线
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_required_feed_rate_is_the_tangent_turn_rate_over_the_curvature():
    """**1.3那条机制**：``σ' = Ω_tan/κ``，而且它真的从`laydown`里走出来。

    金标是解析槽的``Ω/κ(α)``（`generate_oracle.py`不import任何力学模块）。
    被验的是`laydown`在**离散**中心线上、经最近点搜索与中心差分算出来的
    所需送带率——两者对上了才说明落位点几何真的接进了跨段输运。

    顺带判两条"槽切向转多快"：**工件系**那一条必须等于旋钮``Ω``（实测2.33e-4相对），
    **世界系**那一条期望是**零**（理想绕线下带材无折角地续上槽，实测4.66e-4 rad/s）。
    """

    entry = _oracle("oracle:span_disturbance/tangent_turn_feed_rate")
    model = laydown_model(WIRING_RATE_RAD_S, WIRING_ALPHA0_RAD, CURVATURE_MODULATION, 1.0)
    channel = ArmLaydownTakeup(channel_id="takeup/arm", laydown=model, span=SPAN)

    for index, t_s in enumerate(WIRING_PROBE_TIMES_S):
        entry.check(f"feed_rate_{index}_mm_s", channel.takeup_speed_mm_s(t_s))
        #: 同一个数由那条纯函数独立算一遍——**机制与实现互为对拍**。
        alpha = WIRING_ALPHA0_RAD + WIRING_RATE_RAD_S * t_s
        curvature = BASE_CURVATURE_PER_MM / (1.0 + CURVATURE_MODULATION * math.cos(alpha))
        assert tangent_turn_feed_rate_mm_s(
            tangent_turn_rate_rad_s=WIRING_RATE_RAD_S, curvature_per_mm=curvature
        ) == pytest.approx(entry.expected[f"feed_rate_{index}_mm_s"], rel=1.0e-14)

    entry.check(
        "mean_feed_rate_mm_s",
        tangent_turn_feed_rate_mm_s(
            tangent_turn_rate_rad_s=WIRING_RATE_RAD_S, curvature_per_mm=BASE_CURVATURE_PER_MM
        ),
    )
    entry.check(
        "groove_tangent_turn_rate_rad_s", channel.groove_tangent_turn_rate_rad_s(0.1)
    )
    entry.check("world_tangent_turn_rate_rad_s", channel.world_tangent_turn_rate_rad_s(0.1))
    entry.check("arc_at_alpha_max_mm", model.centerline.total_arc_length_mm())
    entry.check(
        "curvature_at_alpha0_per_mm",
        BASE_CURVATURE_PER_MM / (1.0 + CURVATURE_MODULATION * math.cos(WIRING_ALPHA0_RAD)),
    )

    #: 闭合是构造出来的，所以残差必须是机器零——**接线接错了这一条先红**。
    point = model.at(0.1)
    assert point.closure.magnitude_mm < 1.0e-6
    assert abs(point.closure.arc_gap_mm) < 1.0e-6
    assert abs(point.incidence_angle_rad) < 1.0e-4


@pytest.mark.batch
def test_the_laydown_geometry_layer_really_drives_the_span_tension():
    """**接线的端到端那一条**：同一段臂运动，一路走`laydown`、一路走解析速率，
    两条张力时间历程必须重合。

    2026-08-17实测1500步（``dt = 1e-4``）最大偏差**8.71e-06 N**，
    而同一窗口里由臂动引起的张力摆幅是**17.54 N**——**信噪比六个数量级**。
    这条门红了只有两种可能：落位点几何算错了，或者它根本没接上收线端。
    """

    model = laydown_model(WIRING_RATE_RAD_S, WIRING_ALPHA0_RAD, CURVATURE_MODULATION, 1.0)
    channel = ArmLaydownTakeup(channel_id="takeup/arm", laydown=model, span=SPAN)
    speed = WIRING_RATE_RAD_S / BASE_CURVATURE_PER_MM

    _, by_laydown, _ = run_disturbed_span(
        steady_loop(speed, SWEEP_DT_S, step_index=1),
        steps=WIRING_STEPS,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        takeup=channel,
    )
    _, by_analytic, _ = run_disturbed_span(
        steady_loop(speed, SWEEP_DT_S, step_index=1),
        steps=WIRING_STEPS,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        takeup=AnalyticTakeup(
            channel_id="takeup/analytic",
            speed_fn=lambda t_s: analytic_feed_rate_mm_s(
                t_s, WIRING_RATE_RAD_S, WIRING_ALPHA0_RAD, CURVATURE_MODULATION
            ),
            probe_times_s=(0.0, 0.05),
        ),
    )
    worst = max(
        abs(left.tension_n - right.tension_n)
        for left, right in zip(by_laydown, by_analytic, strict=True)
    )
    excursion = max(abs(sample.tension_n - by_analytic[0].tension_n) for sample in by_analytic)
    assert excursion > 5.0, (
        f"整段窗口里张力只摆了{excursion!r} N —— 臂动没有变成扰动，这条通道是空的"
    )
    assert worst < 1.0e-4, (
        f"走`laydown`与走解析速率的张力差{worst!r} N，"
        "落位点几何与跨段输运之间接错了"
    )


@pytest.mark.batch
def test_the_planar_circle_is_the_degenerate_case_with_no_disturbance_at_all():
    """**必须有的退化档**：平面圆 + 匀速转切向 ⟹ 送带率恒定 ⟹ 扰动恒为零。

    三层一起判：

    1. 解析上送带率恰是``Ω/κ0``，经`laydown`的离散中心线走一趟散布**1.63e-7 mm/s**；
       而恒速收线端那一侧的方差**逐位为零**（零容差）——扰动的**输入**真的没有；
    2. ``c = 0``时推进20000步实测**2.2752e-11 N**。这一条最初写的是零容差，
       **实测当场否掉**：病根不在扰动而在起点，``at_steady_state``那一趟
       ``T → L_mat → T``往返有几个ulp（0066实测最坏3.908e-12 N），
       而``c = 0``**没有任何阻尼**把它吃掉；
    3. ``c = 50``时实测**1.1376e-11 N**——**比无阻尼档还小一半**，轴承吃掉了一部分。
    """

    entry = _oracle("oracle:span_disturbance/planar_circle_degenerate")
    speed = WIRING_RATE_RAD_S / BASE_CURVATURE_PER_MM
    model = laydown_model(WIRING_RATE_RAD_S, WIRING_ALPHA0_RAD, 0.0, 1.0)
    channel = ArmLaydownTakeup(channel_id="takeup/arm", laydown=model, span=SPAN)
    rates = [channel.takeup_speed_mm_s(0.001 * index) for index in range(1, 201)]
    entry.check("feed_rate_mm_s", sum(rates) / len(rates))
    entry.check("feed_rate_spread_mm_s", max(rates) - min(rates))

    source = ConstantTakeup(channel_id="takeup/servo", speed_mm_s=speed)
    #: 扰动的**输入**方差逐位为零——这一条是那个退化档里唯一真的能判零容差的。
    probes = [source.takeup_speed_mm_s(0.001 * index) for index in range(200)]
    entry.check("takeup_variance_mm_s", max(probes) - min(probes))
    for damping, quantity in (
        (0.0, "tension_excursion_zero_damping_n"),
        (BEARING_DAMPING_NMM_S, "tension_excursion_nominal_damping_n"),
    ):
        _, _, ledger = run_disturbed_span(
            steady_loop(speed, SWEEP_DT_S, damping_nmm_s=damping),
            steps=20000,
            brake_torque_nmm=BRAKE_TORQUE_NMM,
            takeup=source,
        )
        entry.check(quantity, ledger.peak_tension_n - ledger.trough_tension_n)


def test_the_zero_frequency_limit_of_the_harmonic_response_is_the_steady_state_relation():
    """``ω → 0``时正弦响应的增益必须**逐位**退回``dT/dv = c/R²``。

    这与0066那条"``c = 0``时``T_ss``与`SpoolTension`逐位相等"是同一件事的频域版本：
    一个新式子说自己在某个极限下退化成旧式子，**那句话要么逐位成立，要么就不该说**。

    2026-08-17实测``2ζK/ω_n``与``c/R²``都是``0.013888888888888888``（逐位相同），
    而模长式在``ω = 0``给``0.013888888888888886``——差**1 ulp**。
    **本门不判引擎积分**，所以不标`batch`。
    """

    entry = _oracle("oracle:span_disturbance/quasi_static_limit")
    _, _, stiffness, natural, ratio = _modal(TOUCH_LINE_SPEED_MM_S)
    limit = 2.0 * ratio * stiffness / natural
    quotient = BEARING_DAMPING_NMM_S / (REEL_RADIUS_MM * REEL_RADIUS_MM)
    assert limit == quotient, (
        f"``2ζK/ω_n``给{limit!r}而``c/R²``给{quotient!r}——"
        "'正弦响应在零频退回稳态关系'这句话不成立了"
    )
    entry.check("two_zeta_k_over_omega_n", limit)
    entry.check("damping_over_radius_squared", quotient)
    entry.check(
        "harmonic_gain_at_zero_n_per_mm_s",
        harmonic_tension_amplitude_n(
            span_stiffness_n_per_mm=stiffness,
            takeup_amplitude_mm_s=1.0,
            forcing_rad_s=0.0,
            natural_frequency_rad_s=natural,
            damping_ratio=ratio,
        ),
    )


@pytest.mark.batch
def test_the_tension_disturbance_amplitude_is_monotone_in_the_tangent_turn_rate():
    """**1.3那条单调判据**：槽切向转得越快，张力扰动幅值越大。

    四档``Ω``（1/2/4/8 rad/s）各跑一个整周期多5%。起点直接取**闭式受迫解**
    （``δT(0) = Re X``、``δv_放线(0) = a + ω·Im X/K``）——本工况``ζ = 0.0132``，
    瞬态一旦起来要0.2 s才衰减，比最短那一档的整个窗口还长。

    **窗口必须≥一个整周期**：0.55周期档在``Ω = 8``上把幅值判低10.9%，
    而那是窗口不是物理。这一条写进案例页第四节。

    实测四档偏差−4.95e-6 / −1.92e-5 / −6.94e-5 / −2.11e-4，
    相邻幅值之比2.0295 / 2.1123 / 2.3785——**严格递增**。
    """

    entry = _oracle("oracle:span_disturbance/arm_rate_sweep")
    previous = None
    for index, turn_rate in enumerate(TANGENT_TURN_RATES_RAD_S):
        speed = turn_rate / BASE_CURVATURE_PER_MM
        tension, _, stiffness, natural, ratio = _modal(speed)
        entry.check(f"line_speed_{index}_mm_s", speed)
        entry.check(f"steady_tension_{index}_n", tension)
        entry.check(f"natural_frequency_{index}_rad_s", natural)
        entry.check(f"damping_ratio_{index}", ratio)

        start_tension = entry.expected[f"initial_tension_{index}_n"]
        start_speed = entry.expected[f"initial_payout_speed_{index}_mm_s"]
        loop = SpanTransportLoop(
            span=SPAN,
            reel=_reel(),
            dt_s=SWEEP_DT_S,
            material_length_mm=SPAN.material_length_for_tension_mm(start_tension),
            angular_velocity_rad_s=start_speed / REEL_RADIUS_MM,
            forbid_slack=True,
        )
        steps = int(entry.expected[f"steps_{index}"])
        _, _, ledger = run_disturbed_span(
            loop,
            steps=steps,
            brake_torque_nmm=BRAKE_TORQUE_NMM,
            takeup=AnalyticTakeup(
                channel_id="takeup/analytic",
                speed_fn=lambda t_s, rate=turn_rate: analytic_feed_rate_mm_s(
                    t_s, rate, 0.0, CURVATURE_MODULATION
                ),
                probe_times_s=(0.0,),
            ),
        )
        measured = (ledger.peak_tension_n - ledger.trough_tension_n) / 2.0
        entry.check(f"tension_amplitude_{index}_n", measured)
        #: 闭式那一侧独立算一遍——引擎的纯函数与清单里冻结的数必须一致。
        closed = harmonic_tension_amplitude_n(
            span_stiffness_n_per_mm=stiffness,
            takeup_amplitude_mm_s=CURVATURE_MODULATION * speed,
            forcing_rad_s=turn_rate,
            natural_frequency_rad_s=natural,
            damping_ratio=ratio,
        )
        assert closed == pytest.approx(
            entry.expected[f"tension_amplitude_{index}_n"], rel=1.0e-14
        )
        if previous is not None:
            #: **单调判据本体判的是实测值**（清单里那个比值是闭式的，
            #: 它带的是1e-14容差，喂实测值进去只会把积分误差判成金标不符）。
            assert measured > previous[0], (
                f"Ω = {turn_rate!r}的扰动幅值{measured!r} N没有超过上一档{previous[0]!r} N"
                "——单调判据不成立"
            )
            entry.check(f"amplitude_ratio_{index}", closed / previous[1])
            assert measured / previous[0] > 1.0
        previous = (measured, closed)
        #: 长度账与角冲量账全程有账（触碰那条另有一门判横向冲量）。
        assert abs(ledger.material_length_residual_mm) < 1.0e-9
        assert abs(ledger.angular_impulse_residual_nmm_s) < 1.0e-9


# ---------------------------------------------------------------------------
# 1.4 人手触碰
# ---------------------------------------------------------------------------


def test_the_touch_path_geometry_is_the_exact_polyline_and_the_middle_is_the_softest_spot():
    """路径增量是**几何恒等式**，不是模型：``sqrt(a²+δ²)+sqrt(b²+δ²)−L``。

    三条判据：**二次律**（δ翻倍增量翻四倍，实测比3.99997/3.99987/3.99947，
    **不写死为4**）、**小角度式的差额**（δ=2 mm处4.444e-5、δ=4 mm处1.777e-4，
    拿小角度式当实现会被1e-15的容差当场抓住）、
    **中点是最软的地方**（``p ∝ L/(2ab)``而``ab``在中点最大——直觉在这里是反的）。

    **本门不判引擎积分**，所以不标`batch`。
    """

    entry = _oracle("oracle:span_disturbance/touch_path_geometry")
    subject = THE_TOUCH
    entry.check("centre_excess_mm", subject.path_excess_for_offset_mm(TOUCH_OFFSET_MM))
    entry.check("centre_force_factor", subject.force_geometry_factor(0.01))
    entry.check(
        "centre_small_angle_excess_mm",
        TOUCH_OFFSET_MM
        * TOUCH_OFFSET_MM
        * GEOMETRIC_LENGTH_MM
        / (2.0 * TOUCH_STATION_MM * (GEOMETRIC_LENGTH_MM - TOUCH_STATION_MM)),
    )
    offcentre = dataclasses.replace(subject, station_from_guide_mm=TOUCH_OFFCENTRE_STATION_MM)
    entry.check("offcentre_excess_mm", offcentre.path_excess_for_offset_mm(TOUCH_OFFSET_MM))
    entry.check(
        "centre_over_offcentre_excess",
        subject.path_excess_for_offset_mm(TOUCH_OFFSET_MM)
        / offcentre.path_excess_for_offset_mm(TOUCH_OFFSET_MM),
    )
    entry.check(
        "small_angle_centre_over_offcentre",
        (TOUCH_OFFCENTRE_STATION_MM * (GEOMETRIC_LENGTH_MM - TOUCH_OFFCENTRE_STATION_MM))
        / (TOUCH_STATION_MM * (GEOMETRIC_LENGTH_MM - TOUCH_STATION_MM)),
    )
    assert entry.expected["centre_over_offcentre_excess"] < 1.0

    previous = None
    for index, offset in enumerate(TOUCH_OFFSET_SWEEP_MM):
        value = subject.path_excess_for_offset_mm(offset)
        entry.check(f"excess_{index}_mm", value)
        if previous is not None:
            entry.check(f"excess_ratio_{index}", value / previous)
            #: **比值不写死为4**：精确式比小角度式略小，所以它是一条真的会动的判据。
            assert value / previous < 4.0
        previous = value


@pytest.mark.batch
def test_a_hand_touch_makes_the_tension_spike_and_the_onset_is_an_identity(touch_run):
    """触碰 ⟹ 尖峰。而**触碰开始那一跳是恒等式不是收敛结果**。

    材料长度是状态、不会瞬变，所以路径一跳张力当场跳``EA·p/L_mat``——
    2026-08-17在``dt``四档（4e-6→5e-7）上实测**全部给同一个偏差1.70e-13相对**，
    **它不随步长动**，那正是"这是恒等式"的指纹。

    整段窗口的**峰值**则不是那一跳：它由触碰结束那一次反向阶跃与振铃的相位叠加定，
    实测比开始那一跳还高10.2%。谷值掉到14.15 N（稳态20.28 N的**70%**）——
    **尖峰不是只往上**。
    """

    entry = _oracle("oracle:span_disturbance/touch_tension_spike")
    _, samples, ledger = touch_run
    tension, material, stiffness, natural, ratio = _modal(TOUCH_LINE_SPEED_MM_S)
    entry.check("steady_tension_n", tension)
    entry.check("material_length_mm", material)
    entry.check("span_stiffness_n_per_mm", stiffness)
    entry.check("natural_frequency_rad_s", natural)
    entry.check("damping_ratio", ratio)

    onset = next(sample for sample in samples if THE_TOUCH.is_active(sample.time_s))
    entry.check("onset_step_n", onset.tension_n - tension)
    entry.check("peak_tension_n", ledger.peak_tension_n)
    entry.check("trough_tension_n", ledger.trough_tension_n)
    entry.check("touched_steps", float(ledger.touched_steps))

    #: 触碰窗口内逐点对闭式振铃（两次阶跃的叠加，窗口内只有第一次）。
    step_n = entry.expected["onset_step_n"]
    worst = 0.0
    for sample in samples:
        if not THE_TOUCH.is_active(sample.time_s):
            continue
        want = tension + path_step_tension_ring_n(
            step_tension_n=step_n,
            natural_frequency_rad_s=natural,
            damping_ratio=ratio,
            time_s=sample.time_s - TOUCH_START_S,
        )
        worst = max(worst, abs(sample.tension_n - want))
    #: 实测``dt = 2e-6``给3.79e-4相对（一阶：4e-6档7.55e-4、1e-6档1.94e-4）。
    assert worst / step_n < 1.0e-3, f"窗口内逐点偏差{worst / step_n!r}相对"

    #: 手指感觉到的力：稳态张力乘几何因子，量级半牛顿——**一根手指按得住**。
    force = THE_TOUCH.transverse_force_n(0.01, tension_n=onset.tension_n)
    assert 0.5 < force < 1.0


@pytest.mark.batch
def test_the_open_loop_spike_does_not_fall_back(touch_run):
    """**这道门的分辨力就在这一条**：撤掉控制器，尖峰**不回落**。

    开环下唯一的阻尼通道是轴承。真实量级``c = 50``给``ζ = 0.0132``，
    包络``exp(−ζω_n t)``每个阻尼周期只掉7.9%——**四个周期（66 ms）之后还剩71.8%**。
    实测四档包络比对闭式偏差2.04e-8 / 6.02e-9 / −7.58e-9 / −4.82e-8。

    **有控制器时它应该在一个周期之内被压掉，而那一半本案例没有做**
    （控制器归轨道E，决策0070）。没有这一条反向门，"触碰→尖峰→回落"
    那句话证明不了任何东西：一个把扰动整个丢掉的实现照样能给出一条平的曲线。
    """

    entry = _oracle("oracle:span_disturbance/touch_ring_down")
    _, samples, _ = touch_run
    _, _, _, natural, ratio = _modal(TOUCH_LINE_SPEED_MM_S)
    period = 2.0 * math.pi / (natural * math.sqrt(1.0 - ratio * ratio))
    entry.check("damped_period_s", period)

    after = [sample for sample in samples if sample.time_s >= TOUCH_END_S]

    def half_range(index: int) -> float:
        window = [
            sample.tension_n
            for sample in after
            if index * period <= sample.time_s - TOUCH_END_S < (index + 1) * period
        ]
        return (max(window) - min(window)) / 2.0

    first = half_range(0)
    assert first > 5.0, f"触碰结束后第一个周期只摆了{first!r} N——扰动根本没进去"
    for periods in RING_PERIODS:
        measured = half_range(periods) / first
        entry.check(f"envelope_after_{periods}_periods", measured)
        assert measured == pytest.approx(
            ring_envelope_ratio(
                natural_frequency_rad_s=natural, damping_ratio=ratio, elapsed_s=periods * period
            ),
            rel=1.0e-6,
        )
    #: **不回落**这句话的量化形式：四个周期之后还剩七成。
    assert half_range(4) / first > 0.7


@pytest.mark.batch
def test_the_transverse_impulse_and_the_material_length_ledger_agree(touch_run):
    """**全程有账**：横向冲量与材料长度账之间有一条**恒等式**，不是"看着差不多"。

        ∫F dt = (g/R)·[(J/1000 − c·dt)·Δω + M·N·dt + (c/R)·(ΔL_mat + dt·Σv_收线)]

    ``−c·dt``那一项是**半隐式的指纹**：长度账用步末转速、力矩账用步首转速，
    两者差一个``dt·Δω``。写成``J/1000``（即丢掉它）在``dt → 0``时看不出来，
    而它恰恰是"把半隐式误当显式"的捕手。

    2026-08-17实测残差**8.29e-16 N·s**（冲量本身0.011149 N·s），相对7.44e-14；
    四档``dt``全部在同一量级——**恒等式，不是收敛结果**。
    """

    _, _, ledger = touch_run
    assert ledger.transverse_impulse_n_s > 0.0
    assert abs(ledger.transverse_impulse_residual_n_s) / ledger.transverse_impulse_n_s < 1.0e-12
    #: 另外两本账：材料长度（状态之差 对 逐步流量）与角冲量（动量之差 对 力矩积分）。
    assert abs(ledger.material_length_residual_mm) < 1.0e-9
    assert abs(ledger.angular_impulse_residual_nmm_s) < 1.0e-9
    assert ledger.steps == TOUCH_STEPS
    assert ledger.duration_s == pytest.approx(TOUCH_STEPS * TOUCH_DT_S, rel=1.0e-12)


@pytest.mark.batch
def test_must_be_red_a_small_angle_path_excess_would_fail_the_geometry_gate(touch_run):
    """**捕手门**：把路径增量写成小角度式``δ²L/(2ab)``会被判红。

    这不是假设——两者在δ=2 mm处差4.444e-5相对，而判据容差是1e-15。
    把它跑成一次真的失败：拿小角度值当"实现给出的数"喂进清单比对，必须抛。
    """

    from physics_engine.oracles import OracleError

    entry = _oracle("oracle:span_disturbance/touch_path_geometry")
    small = (
        TOUCH_OFFSET_MM
        * TOUCH_OFFSET_MM
        * GEOMETRIC_LENGTH_MM
        / (2.0 * TOUCH_STATION_MM * (GEOMETRIC_LENGTH_MM - TOUCH_STATION_MM))
    )
    with pytest.raises(OracleError):
        entry.check("centre_excess_mm", small)

    #: 同一条捕手的第二面：把张力尖峰算成``K·p``（走长度账那条通道）
    #: 而不是``EA·p/L_mat``，差``L_path/L_mat``即3.4e-4相对——也必须红。
    spike_entry = _oracle("oracle:span_disturbance/touch_tension_spike")
    _, material, stiffness, _, _ = _modal(TOUCH_LINE_SPEED_MM_S)
    with pytest.raises(OracleError):
        spike_entry.check(
            "onset_step_n", stiffness * THE_TOUCH.path_excess_for_offset_mm(TOUCH_OFFSET_MM)
        )
