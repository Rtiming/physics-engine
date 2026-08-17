"""conformance：落位点几何与闭合残差（`cases/helix_laydown_closure`）。

**这一条回答的是"这一瞬落位的是槽上哪一点、它的帧朝哪、要放多少带材"，
以及"位姿时间线与送带账对不对得上"。**

plans/14第3.3节订正过一次场景：张力机不动、入带点是**世界系里固定的一个点**、
在途自由段是空间里**一条不动的线段**。**于是臂动扰动张力的通道不是跨段长度，
是落位点的几何**——槽的切向在转、槽面法线在转，入射角与所需送带率跟着变。

## 闭合条件是本案例的核心

落位点必须同时在槽上（弧长坐标由送带账定）**且**在那个固定入带点上（由位姿定）。
**这两条一般不自洽**，三个位置约束配一个未知量。本案例判的是引擎有没有
**两条各算一次并把差额按方向拆开**，而不是偷偷把其中一条当成对的。

两个病人的残差模长可以完全一样，而处置完全相反：

* 送带账偏0.7 mm ⟹ 弧长坐标差**恰为**0.7、残差几乎全在沿槽分量。**多放带材就好**；
* 位姿把线圈举偏1.5 mm ⟹ 弧长坐标差**恰为零**、残差全在横向、且不可约。
  **多放少放带材都修不掉。**

## 金标为什么可以全闭式

真实中心线在GCW那边，本仓没有。所以取解析螺旋线当中心线、
**姿态随便选而平移反解**当机器人——闭合于是是构造出来的，
落位点的每个量都有闭式。推导见`generate_oracle.py`。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.laydown import (
    ArcRateProbe,
    CenterlineSemantics,
    ClosureTolerance,
    FeedAccount,
    FreeSpanGeometry,
    GrooveCenterline,
    GrooveStation,
    LaydownError,
    LaydownModel,
    assert_closure,
)
from physics_engine.motion import AnalyticPose, Pose
from physics_engine.oracles import load_manifest

CASE = Path(__file__).resolve().parents[2] / "cases" / "helix_laydown_closure"
MANIFEST = load_manifest(CASE / "oracle.json")

RADIUS_MM = 60.0
PITCH_MM_PER_RAD = 25.0
HELIX_SCALE_MM = 65.0
SPIN_RAD_S = 0.4
FEED_RATE_MM_S = 52.0
START_THETA_RAD = 0.6
HORIZON_S = 4.0
ENTRY_POINT_MM = (1126.0, 0.0, 300.0)
SPAN_LENGTH_MM = 200.0
FEED_DRIFT_MM = 0.7
POSE_LIFT_MM = 1.5
SAMPLE_TIMES_S = (0.5, 1.5, 2.5, 3.5)

CIRCLE_RADIUS_MM = 60.0
CIRCLE_FEED_RATE_MM_S = 24.0
CIRCLE_HORIZON_S = 20.0

#: 生产档的中心线密度与语义。**收敛阶另有一条门去扫**，这里只固定一个够密的档。
STATIONS_PER_TURN = 512
RATE_STEP_S = 1.0e-3


def _oracle(oracle_id: str):
    for entry in MANIFEST.oracles:
        if entry.id == oracle_id:
            return entry
    raise AssertionError(f"清单里没有{oracle_id}")


# ---------------------------------------------------------------------------
# 解析件：螺旋线、它的帧、以及那条"平移反解"的位姿
# ---------------------------------------------------------------------------


def helix_position(theta: float, pitch: float = PITCH_MM_PER_RAD, radius: float = RADIUS_MM):
    return (radius * math.cos(theta), radius * math.sin(theta), pitch * theta)


def helix_frame(theta: float, pitch: float = PITCH_MM_PER_RAD, radius: float = RADIUS_MM):
    """``(t, s, n)``。约定抄GCW：``s = n × t``、右手系（plans/14第二节）。"""

    scale = math.sqrt(radius * radius + pitch * pitch)
    tangent = (-radius * math.sin(theta) / scale, radius * math.cos(theta) / scale,
               pitch / scale)
    normal = (math.cos(theta), math.sin(theta), 0.0)
    width = (pitch * math.sin(theta) / scale, -pitch * math.cos(theta) / scale,
             radius / scale)
    return tangent, width, normal


def centerline(
    *,
    stations_per_turn: int = STATIONS_PER_TURN,
    pitch: float = PITCH_MM_PER_RAD,
    radius: float = RADIUS_MM,
    position_interpolation: str = "hermite_tangent",
    frame_interpolation: str = "reorthonormalised_linear",
    closed: bool = False,
    refinement: int = 2,
) -> GrooveCenterline:
    scale = math.sqrt(radius * radius + pitch * pitch)
    stations = []
    for index in range(stations_per_turn + 1):
        theta = 2.0 * math.pi * index / stations_per_turn
        tangent, width, normal = helix_frame(theta, pitch, radius)
        stations.append(
            GrooveStation(
                arc_length_mm=scale * theta,
                position_mm=helix_position(theta, pitch, radius),
                tangent=tangent,
                width_direction=width,
                surface_normal=normal,
            )
        )
    if closed:
        #: 闭合要求末站点**逐位**重复首站点——真实CSV首末差一个采样步，
        #: 补上那一站是声明者的事（`laydown`那条门就为此立着）。
        stations[-1] = GrooveStation(
            arc_length_mm=stations[-1].arc_length_mm,
            position_mm=stations[0].position_mm,
            tangent=stations[0].tangent,
            width_direction=stations[0].width_direction,
            surface_normal=stations[0].surface_normal,
        )
    return GrooveCenterline(
        centerline_id=f"groove/analytic{stations_per_turn}",
        stations=tuple(stations),
        semantics=CenterlineSemantics(
            position_interpolation=position_interpolation,
            frame_interpolation=frame_interpolation,
            topology="closed" if closed else "open",
            out_of_range="wrap" if closed else "reject",
            nearest_refinement_iterations=refinement,
        ),
        length_unit="mm",
    )


def screw_source(
    *,
    pitch: float = PITCH_MM_PER_RAD,
    radius: float = RADIUS_MM,
    feed_rate: float = FEED_RATE_MM_S,
    horizon_s: float = HORIZON_S,
    entry=ENTRY_POINT_MM,
) -> AnalyticPose:
    """**姿态随便选、平移反解**——闭合因此是构造出来的，不是碰巧成立的。"""

    scale = math.sqrt(radius * radius + pitch * pitch)

    def pose_fn(t_s: float) -> Pose:
        theta = START_THETA_RAD + feed_rate * t_s / scale
        angle = -SPIN_RAD_S * t_s
        point = helix_position(theta, pitch, radius)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = (point[0] * cos_a - point[1] * sin_a,
                   point[0] * sin_a + point[1] * cos_a, point[2])
        return Pose(
            translation_mm=tuple(entry[axis] - rotated[axis] for axis in range(3)),
            rotation_xyzw=(0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)),
        )

    return AnalyticPose(
        source_id="motion/analytic_screw",
        pose_fn=pose_fn,
        declared_horizon_s=horizon_s,
        extrapolation="reject",
        replayable=True,
        replay_probe_times_s=(0.0, 0.5 * horizon_s),
    )


def world_triad(t_s: float, pitch: float = PITCH_MM_PER_RAD, radius: float = RADIUS_MM,
                feed_rate: float = FEED_RATE_MM_S):
    """闭式：世界系三标架只随``χ = θ(t) − ω t``转。"""

    scale = math.sqrt(radius * radius + pitch * pitch)
    theta = START_THETA_RAD + feed_rate * t_s / scale
    return helix_frame(theta - SPIN_RAD_S * t_s, pitch, radius)


def build(
    *,
    groove: GrooveCenterline | None = None,
    pitch: float = PITCH_MM_PER_RAD,
    radius: float = RADIUS_MM,
    feed_rate: float = FEED_RATE_MM_S,
    horizon_s: float = HORIZON_S,
    arc_drift_mm: float = 0.0,
    lift_mm: float = 0.0,
    scheme: str = "central",
) -> LaydownModel:
    scale = math.sqrt(radius * radius + pitch * pitch)
    groove = centerline(pitch=pitch, radius=radius) if groove is None else groove
    tangent, _, normal = world_triad(0.0, pitch, radius, feed_rate)
    entry = tuple(ENTRY_POINT_MM[axis] + lift_mm * normal[axis] for axis in range(3))
    span = FreeSpanGeometry(
        span_id="span/tension_stand",
        guide_exit_mm=tuple(entry[axis] + SPAN_LENGTH_MM * tangent[axis] for axis in range(3)),
        entry_point_mm=entry,
    )
    return LaydownModel(
        model_id="laydown/analytic_case",
        motion=screw_source(pitch=pitch, radius=radius, feed_rate=feed_rate,
                            horizon_s=horizon_s),
        centerline=groove,
        span=span,
        feed=FeedAccount(
            account_id="feed/constant_rate",
            length_fn=lambda t_s: feed_rate * t_s,
            probe_times_s=(0.0, 0.25 * horizon_s, 0.5 * horizon_s, horizon_s),
        ),
        arc_origin_mm=scale * START_THETA_RAD + arc_drift_mm,
        rate_probe=ArcRateProbe(scheme=scheme, step_s=RATE_STEP_S),
    )


# ---------------------------------------------------------------------------
# 闭式自洽（不碰引擎，毫秒级）
# ---------------------------------------------------------------------------


def test_the_helix_kinematics_are_closed_form_and_self_consistent():
    """``a = 65``是精确整数、``κ = R/a²``、``τ = p/a²``，且``κ² + τ² = 1/a²``。

    最后那条恒等式是闭式自洽的证据：螺旋线上``sqrt(κ² + τ²)``就是``1/a``,
    **它不是本页凑出来的**。
    """

    entry = _oracle("oracle:laydown/helix_kinematics")
    scale = entry.expected["helix_scale_mm"]
    assert scale == pytest.approx(math.hypot(RADIUS_MM, PITCH_MM_PER_RAD), rel=1e-15)
    assert scale == 65.0, "60-25-65是精确勾股，闭式因此干净"
    curvature = entry.expected["curvature_per_mm"]
    torsion = entry.expected["torsion_per_mm"]
    assert math.hypot(curvature, torsion) == pytest.approx(1.0 / scale, rel=1e-14)
    assert entry.expected["arc_per_turn_mm"] == pytest.approx(2.0 * math.pi * scale, rel=1e-15)
    #: 折成°/mm与plans/14第2.2节的实测区间比一次：**量级对得上，但这是设定不是实测**。
    assert 0.3 < math.degrees(torsion) < 0.4
    #: 三标架转速不为零是刻意的——为零就退化成测不出东西的算例。
    assert entry.expected["triad_spin_rad_s"] == pytest.approx(
        FEED_RATE_MM_S / scale - SPIN_RAD_S, rel=1e-15
    )
    assert entry.expected["triad_spin_rad_s"] > 0.0


def test_the_two_closure_failures_have_the_same_magnitude_but_opposite_causes():
    """闭式自洽：两个病人的残差模长同量级（0.70与1.50 mm），**而分解完全相反**。

    这条门判的是**判据本身有区分力**：如果一个实现只报模长，
    这两行数字看起来只是"一个大一点一个小一点"。
    """

    drifted = _oracle("oracle:laydown/closure_from_a_drifted_feed_account").expected
    lifted = _oracle("oracle:laydown/closure_from_a_lifted_pose").expected
    #: 送带账偏：几乎全在沿槽，弧长差恰为挪动量。
    assert drifted["along_tangent_mm"] / drifted["magnitude_mm"] > 0.99
    assert drifted["transverse_mm"] / drifted["magnitude_mm"] < 0.01
    assert drifted["arc_gap_mm"] == FEED_DRIFT_MM
    #: 位姿举偏：全在横向，弧长差恰为零。
    assert lifted["transverse_mm"] == lifted["magnitude_mm"] == POSE_LIFT_MM
    assert lifted["along_tangent_mm"] == 0.0
    assert lifted["arc_gap_mm"] == 0.0
    #: 只看模长分不开——这正是本案例存在的理由。
    assert 0.4 < drifted["magnitude_mm"] / lifted["magnitude_mm"] < 0.6


def test_the_incidence_closed_form_starts_at_zero_and_grows_monotonically():
    """闭式：``cos θ_inc = (R² cos φ + p²)/a²``，``φ = 0``给0且单调增。

    ``φ = 0``那一点是**符号约定的锚点**：理想落位处入射角为零。
    取反的话它会是``π``（`tests/test_laydown.py`那条必红守着这一点）。
    """

    entry = _oracle("oracle:laydown/incidence_over_time")
    angles = [entry.expected[f"incidence_rad_at_t{t:g}".replace(".", "p")]
              for t in SAMPLE_TIMES_S]
    for earlier, later in zip(angles, angles[1:], strict=False):
        assert later > earlier, f"入射角没有随时间单调增：{angles}"
    scale_squared = HELIX_SCALE_MM**2
    for time_s, angle in zip(SAMPLE_TIMES_S, angles, strict=True):
        phi = (FEED_RATE_MM_S / HELIX_SCALE_MM - SPIN_RAD_S) * time_s
        expected = math.acos(
            (RADIUS_MM**2 * math.cos(phi) + PITCH_MM_PER_RAD**2) / scale_squared
        )
        assert angle == pytest.approx(expected, rel=1e-9), f"t={time_s}"
    #: 两个分量合成回总角（小角展开之外要靠球面关系，这里直接判余弦）。
    for time_s, angle in zip(SAMPLE_TIMES_S, angles, strict=True):
        key = f"{time_s:g}".replace(".", "p")
        in_plane = entry.expected[f"in_plane_rad_at_t{key}"]
        out_of_plane = entry.expected[f"out_of_plane_rad_at_t{key}"]
        assert math.cos(angle) == pytest.approx(
            math.cos(out_of_plane) * math.cos(in_plane), rel=1e-12
        )


# ---------------------------------------------------------------------------
# 引擎对闭式
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_engine_reproduces_the_incidence_angle_over_time():
    """引擎侧的落位点入射角对闭式。**这是"槽的切向在转"那句话的可判形式。**"""

    entry = _oracle("oracle:laydown/incidence_over_time")
    model = build()
    for time_s in SAMPLE_TIMES_S:
        key = f"{time_s:g}".replace(".", "p")
        point = model.at(time_s)
        for name, measured in (
            ("incidence", point.incidence_angle_rad),
            ("in_plane", point.incidence_in_plane_rad),
            ("out_of_plane", point.incidence_out_of_plane_rad),
        ):
            expected = entry.expected[f"{name}_rad_at_t{key}"]
            tolerance = entry.tolerances[f"{name}_rad_at_t{key}"]
            assert measured == pytest.approx(
                expected, rel=tolerance.rel_tol, abs=tolerance.abs_tol
            ), f"t={time_s} {name}"
    #: 入射角确实在变——不变的话上面那些相等断言全是在验一个常数。
    spread = max(model.at(t).incidence_angle_rad for t in SAMPLE_TIMES_S) - min(
        model.at(t).incidence_angle_rad for t in SAMPLE_TIMES_S
    )
    assert spread > 1.0, f"入射角几乎没动（跨度{spread!r} rad），这一档退化了"


@pytest.mark.batch
def test_the_engine_reproduces_the_required_feed_rate_from_the_pose_alone():
    """``dσ_pose/dt``**只由位姿算**，对闭式``u = 2aω``。

    引擎侧的路径是：把固定入带点变回工件系 → 中心线上找最近点 → 差分。
    **送带账一个字节都没有参与**——所以这个数与送带账给的速率对得上与否，
    本身就是一条独立信息（下一条门判它）。
    """

    entry = _oracle("oracle:laydown/helix_kinematics")
    tolerance = entry.tolerances["required_feed_rate_mm_s"]
    model = build()
    worst = 0.0
    for time_s in SAMPLE_TIMES_S:
        point = model.at(time_s)
        assert point.required_feed_rate_mm_s == pytest.approx(
            entry.expected["required_feed_rate_mm_s"],
            rel=tolerance.rel_tol,
            abs=tolerance.abs_tol,
        )
        assert point.accounted_feed_rate_mm_s == pytest.approx(FEED_RATE_MM_S, rel=1e-9)
        assert abs(point.feed_rate_gap_mm_s()) < 1.0e-6
        worst = max(worst, abs(point.required_feed_rate_mm_s - FEED_RATE_MM_S))
    assert worst < 1.0e-6, f"实测最大偏差{worst!r} mm/s"


@pytest.mark.batch
def test_a_drifted_feed_account_shows_up_along_the_groove_and_nowhere_else():
    """**病因一**：送带账偏0.7 mm。残差几乎全在沿槽，弧长差**恰为**0.7。

    闭式三分量都与``θ``无关（螺旋线齐次），所以本条同时判"沿弧长走一段
    残差不变"——变了说明取错了帧。
    """

    entry = _oracle("oracle:laydown/closure_from_a_drifted_feed_account")
    model = build(arc_drift_mm=FEED_DRIFT_MM)
    for time_s in SAMPLE_TIMES_S:
        residual = model.at(time_s).closure
        for name, measured in (
            ("magnitude_mm", residual.magnitude_mm),
            ("along_tangent_mm", residual.along_tangent_mm),
            ("across_normal_mm", residual.across_normal_mm),
            ("transverse_mm", residual.transverse_mm),
            ("arc_gap_mm", residual.arc_gap_mm),
        ):
            tolerance = entry.tolerances[name]
            assert measured == pytest.approx(
                entry.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
            ), f"t={time_s} {name}"
        #: **位姿没有任何问题**：入带点仍然落在曲线上。
        assert residual.pose_only_offset_mm < 1.0e-6


@pytest.mark.batch
def test_the_third_order_width_component_needs_four_times_the_sampling_to_show_up():
    """``across_width``是三阶小量（4.80e-06 mm），**生产档根本判不了它**。

    残差本身0.7 mm，乘上帧的角误差（512站点/匝约2e-06 rad）就已经与它同量级——
    实测512档相对偏差**96%**。加密到2048站点/匝才落到6.0e-02。

    **这一条不藏起来而是单列一门**，理由是：一个"跑得通"的实现完全可能把这一路
    算错整整一个量级而所有别的判据都绿。已知失效清单第2条登记了它。

    收敛实测（256/512/1024/2048/4096站点每匝的相对偏差）：
    3.64 / 0.96 / 0.24 / 0.060 / 0.015——**二阶**，与帧插值同阶。
    """

    entry = _oracle("oracle:laydown/closure_from_a_drifted_feed_account")
    tolerance = entry.tolerances["across_width_mm"]
    dense = build(groove=centerline(stations_per_turn=4 * STATIONS_PER_TURN),
                  arc_drift_mm=FEED_DRIFT_MM)
    for time_s in SAMPLE_TIMES_S:
        assert dense.at(time_s).closure.across_width_mm == pytest.approx(
            entry.expected["across_width_mm"], rel=tolerance.rel_tol
        ), f"t={time_s}"
    #: 生产档确实判不了——**写出来比让读者以为它被验过诚实**。
    coarse = build(arc_drift_mm=FEED_DRIFT_MM)
    worst = max(
        abs(coarse.at(t).closure.across_width_mm / entry.expected["across_width_mm"] - 1.0)
        for t in SAMPLE_TIMES_S
    )
    assert worst > 0.5, f"512站点/匝居然判得了这一路（偏差{worst!r}）——那要回来改本页"


@pytest.mark.batch
def test_a_lifted_pose_shows_up_as_transverse_only_and_the_feed_account_is_innocent():
    """**病因二，本案例最要紧的一条**：位姿把线圈举偏1.5 mm。

    弧长坐标差**恰为零**——送带账一点问题都没有。残差全部落在横向，
    而且不可约偏距也是1.5 mm：**多放少放带材都修不掉它**。

    报一个非零的弧长差，就是把账算到了无辜的一方；而"调送带账把残差压下去"
    正是一个只看模长的实现最自然会做的事。
    """

    entry = _oracle("oracle:laydown/closure_from_a_lifted_pose")
    point = build(lift_mm=POSE_LIFT_MM, scheme="forward").at(0.0)
    residual = point.closure
    for name, measured in (
        ("magnitude_mm", residual.magnitude_mm),
        ("transverse_mm", residual.transverse_mm),
        ("across_normal_mm", residual.across_normal_mm),
        ("pose_only_offset_mm", residual.pose_only_offset_mm),
        ("along_tangent_mm", residual.along_tangent_mm),
        ("arc_gap_mm", residual.arc_gap_mm),
    ):
        tolerance = entry.tolerances[name]
        assert measured == pytest.approx(
            entry.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        ), name
    assert point.incidence_angle_rad == pytest.approx(
        entry.expected["incidence_rad"],
        abs=entry.tolerances["incidence_rad"].abs_tol,
    )
    #: 与病因一的分水岭：弧长差差了五个数量级以上。
    drifted = build(arc_drift_mm=FEED_DRIFT_MM).at(1.5).closure
    assert abs(drifted.arc_gap_mm) / max(abs(residual.arc_gap_mm), 1.0e-18) > 1.0e4


@pytest.mark.batch
def test_the_closure_gate_is_the_only_thing_that_notices_the_two_inputs_disagree():
    """闭合门：对得上的时间线过，两个病人各被拒一次。

    今天本仓**没有任何别的东西**看得见"位姿与送带账各说各话"这件事——
    两条输入各自都合法，只有把它们放在一起才看得出来。
    """

    tolerance = ClosureTolerance(
        position_abs_mm=1.0e-3,
        arc_abs_mm=1.0e-3,
        reason=(
            "闭式构造下闭合恒成立，残差只剩离散误差；512站点/匝Hermite档实测"
            "3.5e-09 mm，1e-3是其三十万倍余量，留给更粗的中心线"
        ),
    )
    assert_closure(build().track(SAMPLE_TIMES_S), tolerance, run_label="case/aligned")
    with pytest.raises(LaydownError, match="闭合残差"):
        assert_closure(
            build(arc_drift_mm=FEED_DRIFT_MM).track(SAMPLE_TIMES_S),
            tolerance,
            run_label="case/drifted_feed",
        )
    with pytest.raises(LaydownError, match="闭合残差"):
        assert_closure(
            build(lift_mm=POSE_LIFT_MM, scheme="forward").track([0.0]),
            tolerance,
            run_label="case/lifted_pose",
        )


# ---------------------------------------------------------------------------
# 平面圆退化档：任何实现都该算对
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_plane_circle_degenerates_exactly_as_it_should():
    """**退化对照**：``p = 0``、``u = Rω``⟹ 挠率为零、三标架恒定、入射角恒为零。

    **这一档任何实现都该算对，算不对说明连基本盘都没接上。**
    它同时是闭合拓扑唯一的金标：本档走了1.27匝，弧长坐标**跨过接缝回绕**，
    而回绕算错会让弧长差直接差出一整匝（377 mm）。
    """

    entry = _oracle("oracle:laydown/plane_circle_degenerate")
    groove = centerline(pitch=0.0, radius=CIRCLE_RADIUS_MM, closed=True)
    assert groove.total_arc_length_mm() == pytest.approx(
        entry.expected["arc_per_turn_mm"], rel=1e-12
    )
    model = build(groove=groove, pitch=0.0, radius=CIRCLE_RADIUS_MM,
                  feed_rate=CIRCLE_FEED_RATE_MM_S, horizon_s=CIRCLE_HORIZON_S)
    laps = CIRCLE_FEED_RATE_MM_S * CIRCLE_HORIZON_S / entry.expected["arc_per_turn_mm"]
    assert laps > 1.0, f"只走了{laps!r}匝——回绕那一条根本没被走到"

    tangent0, width0, normal0 = world_triad(0.0, 0.0, CIRCLE_RADIUS_MM,
                                            CIRCLE_FEED_RATE_MM_S)
    worst_incidence = worst_rate = worst_closure = worst_triad = 0.0
    seen = 0
    for index in range(41):
        time_s = 0.05 + (CIRCLE_HORIZON_S - 0.1) * index / 40.0
        point = model.at(time_s)
        seen += 1
        worst_incidence = max(worst_incidence, abs(point.incidence_angle_rad))
        worst_rate = max(
            worst_rate, abs(point.required_feed_rate_mm_s - CIRCLE_FEED_RATE_MM_S)
        )
        worst_closure = max(worst_closure, point.closure.magnitude_mm)
        for measured, reference in ((point.tangent, tangent0),
                                    (point.width_direction, width0),
                                    (point.surface_normal, normal0)):
            worst_triad = max(
                worst_triad, max(abs(measured[axis] - reference[axis]) for axis in range(3))
            )
        #: **挠率为零的可判形式**：带宽方向恒为``(0, 0, 1)``，它不绕切向转。
        assert point.width_direction[2] == pytest.approx(1.0, abs=1.0e-6)
    assert seen == 41, "零执行绝不pass"
    assert worst_incidence < entry.tolerances["incidence_rad"].abs_tol
    assert worst_rate < entry.tolerances["required_feed_rate_mm_s"].abs_tol
    assert worst_closure < 1.0e-5
    assert worst_triad < 1.0e-5, f"世界系三标架漂了{worst_triad!r}——这一档它该是常数"


@pytest.mark.batch
def test_the_helix_and_the_circle_differ_in_exactly_the_declared_way():
    """两档的分水岭是``triad_spin_rad_s``：螺旋线档0.4、平面圆档0。

    **只有平面圆档过而螺旋线档不过，说明实现只接上了绕线盘那个旧模型**
    （plans/14第一节订正掉的那个）。所以两档要并排判。
    """

    helix = _oracle("oracle:laydown/helix_kinematics").expected
    circle = _oracle("oracle:laydown/plane_circle_degenerate").expected
    assert circle["triad_spin_rad_s"] == 0.0
    assert helix["triad_spin_rad_s"] > 0.1
    assert circle["torsion_per_mm"] == 0.0
    assert helix["torsion_per_mm"] > 0.0


# ---------------------------------------------------------------------------
# 离散化：阶要量出来，不写死
# ---------------------------------------------------------------------------


#: 收敛扫的采样时刻数。**9个太少**：采样误差是振荡的，9个点上的最大值本身在跳，
#: 实测``hold_station``的比值在9点采样下给出1.690（真值≈2）。33个点上稳定在2.01—2.07。
CONVERGENCE_SAMPLES = 33


def _worst(groove: GrooveCenterline) -> tuple[float, float, float]:
    """``(闭合残差, 弧长坐标偏差, 入射角偏差)``在采样时刻上的最大值。"""

    model = build(groove=groove)
    scale = HELIX_SCALE_MM
    closure = arc = incidence = 0.0
    for index in range(CONVERGENCE_SAMPLES):
        time_s = 0.2 + 3.6 * index / (CONVERGENCE_SAMPLES - 1)
        point = model.at(time_s)
        true_theta = START_THETA_RAD + FEED_RATE_MM_S * time_s / scale
        closure = max(closure, point.closure.magnitude_mm)
        arc = max(arc, abs(point.pose_arc_length_mm - scale * true_theta))
        phi = (FEED_RATE_MM_S / scale - SPIN_RAD_S) * time_s
        true_angle = math.acos(
            (RADIUS_MM**2 * math.cos(phi) + PITCH_MM_PER_RAD**2) / (scale * scale)
        )
        incidence = max(incidence, abs(point.incidence_angle_rad - true_angle))
    return (closure, arc, incidence)


@pytest.mark.batch
def test_the_convergence_orders_are_measured_and_the_two_semantics_are_two_orders_apart():
    """**收敛阶实测，不写死为某个数**（与`harmonic_oscillator`那条同源）。

    2026-08-17实测（螺旋线、33个时刻的最大偏差，站点/匝翻倍一档）：

    | 语义 | 量 | 64→128 | 128→256 | 256→512 | 阶 |
    |---|---|---:|---:|---:|---|
    | `chord_linear` | 闭合残差 | 4.00 | 4.30 | 3.87 | **二阶** |
    | `chord_linear` | 弧长坐标 | 7.99 | 9.01 | 7.35 | **三阶** |
    | `hermite_tangent` | 闭合残差 | 16.02 | 18.47 | 14.97 | **四阶** |
    | `hermite_tangent` | 弧长坐标 | 33.02 | 31.15 | 30.76 | **五阶** |

    绝对值（闭合残差，mm）：`chord_linear` 7.23e-02 → 1.09e-03；
    `hermite_tangent` 1.45e-05 → 3.28e-09。

    弧长坐标比位置**高一阶**是最近点条件``(C − p)·C' = 0``的驻点性质：
    在极小点附近，参数的一阶偏差只带来距离的二阶变化。
    **这一条是量出来才知道的，不是先验的。**

    比值不干净（4.30、18.47那两档）是因为采样误差**振荡**——
    9个时刻的采样下``hold_station``甚至给出过1.690。所以门判**区间**：
    `chord_linear` `[3.6, 4.6]`、`hermite_tangent` `[13.5, 20.0]`。
    两个区间**不相交**，所以"声明哪一条语义"是一个有代价的选择。
    """

    for interpolation, bounds in (
        ("chord_linear", (3.6, 4.6)),
        ("hermite_tangent", (13.5, 20.0)),
    ):
        errors = [
            _worst(centerline(stations_per_turn=count, position_interpolation=interpolation))[0]
            for count in (64, 128, 256, 512)
        ]
        assert errors[0] > errors[1] > errors[2] > errors[3], (
            f"{interpolation}的闭合残差没有单调下降：{errors}"
        )
        for earlier, later in zip(errors, errors[1:], strict=False):
            assert bounds[0] <= earlier / later <= bounds[1], (
                f"{interpolation}的收敛比{earlier / later!r}落在{bounds}之外：{errors}"
            )
    #: 两条语义在同一密度下差了四个数量级——**这才是"要声明"的代价**。
    coarse_chord = _worst(centerline(stations_per_turn=512,
                                     position_interpolation="chord_linear"))[0]
    coarse_hermite = _worst(centerline(stations_per_turn=512,
                                       position_interpolation="hermite_tangent"))[0]
    assert coarse_chord / coarse_hermite > 1.0e4


@pytest.mark.batch
def test_the_frame_semantics_are_one_order_apart_and_that_is_why_they_must_be_declared():
    """帧插值的两条语义：``hold_station``一阶、``reorthonormalised_linear``二阶。

    2026-08-17实测（入射角偏差，33个时刻的最大值，站点/匝翻倍）：

    | 语义 | 64→128 | 128→256 | 256→512 | 512档绝对值 |
    |---|---:|---:|---:|---:|
    | `hold_station` | 2.01 | 2.04 | 2.07 | 1.06e-02 rad |
    | `reorthonormalised_linear` | 3.73 | 4.25 | 3.70 | 2.16e-06 rad |

    后者的比值**不干净**，因为采样误差是**振荡**的——它取决于采样点落在段内哪里。
    所以门判``[1.8, 2.3]``与``[3.2, 4.8]``这两个**不相交**的区间，
    而不是判"恰好是2"与"恰好是4"。

    非平面槽的整匝帧扭转跨236°—657°（plans/14第2.2节），
    **零阶保持在粗采样下直接把这段扭转丢掉**——这就是为什么这条语义
    不许由库来挑一个默认值。
    """

    orders = {}
    for interpolation, bounds in (
        ("hold_station", (1.8, 2.3)),
        ("reorthonormalised_linear", (3.2, 4.8)),
    ):
        errors = [
            _worst(centerline(stations_per_turn=count, frame_interpolation=interpolation))[2]
            for count in (64, 128, 256, 512)
        ]
        for earlier, later in zip(errors, errors[1:], strict=False):
            assert bounds[0] <= earlier / later <= bounds[1], (
                f"{interpolation}的收敛比{earlier / later!r}落在{bounds}之外：{errors}"
            )
        orders[interpolation] = errors
    #: 同一个采样密度下，二阶那条至少好两个数量级——**这才是"要声明"的代价**。
    assert orders["hold_station"][-1] / orders["reorthonormalised_linear"][-1] > 100.0
