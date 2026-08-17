"""落位点几何层的单元门与必红矩阵（`src/physics_engine/laydown.py`，决策0067）。

本文件按**分支**组织而不是按规则组织（plans/09教训三）：每一条失败关闭各配一条
"必须红"用例，且每条红用例的docstring写明**注错方式**——注错方式写不出来的门，
说明它根本没有一个"错"的形态可挡。

三处绿分支单独立着，因为**没有绿的红说明不了门认得对的东西**：
帧约定的正例、闭合曲线接缝的正例、两条语义各自给出不同答案的正例。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.laydown import (
    ARC_OVER_CHORD_CEILING,
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
    rotate_by_quaternion,
)
from physics_engine.motion import AnalyticPose, Pose
from physics_engine.rigidbody import rotate_body_to_world, rotate_world_to_body

# ---------------------------------------------------------------------------
# 脚手架：一条解析螺旋线 + 一条解析位姿。红用例都是从**合格的那一份**注错出来的。
# ---------------------------------------------------------------------------

RADIUS_MM = 60.0
PITCH_MM_PER_RAD = 25.0
#: ``sqrt(60² + 25²) = 65``恰好是整数（5×勾股数(12, 5, 13)），闭式因此干净。
HELIX_SCALE_MM = 65.0
SPIN_RAD_S = 0.4
FEED_RATE_MM_S = 52.0
START_THETA_RAD = 0.6
ENTRY_POINT_MM = (1126.0, 0.0, 300.0)
SPAN_LENGTH_MM = 200.0
HORIZON_S = 4.0


def helix_position(theta: float, pitch: float = PITCH_MM_PER_RAD) -> tuple[float, float, float]:
    return (RADIUS_MM * math.cos(theta), RADIUS_MM * math.sin(theta), pitch * theta)


def helix_frame(theta: float, pitch: float = PITCH_MM_PER_RAD):
    scale = math.sqrt(RADIUS_MM * RADIUS_MM + pitch * pitch)
    tangent = (-RADIUS_MM * math.sin(theta) / scale, RADIUS_MM * math.cos(theta) / scale,
               pitch / scale)
    normal = (math.cos(theta), math.sin(theta), 0.0)
    width = (pitch * math.sin(theta) / scale, -pitch * math.cos(theta) / scale,
             RADIUS_MM / scale)
    return tangent, width, normal


def semantics(**overrides) -> CenterlineSemantics:
    fields = {
        "position_interpolation": "hermite_tangent",
        "frame_interpolation": "reorthonormalised_linear",
        "topology": "open",
        "out_of_range": "reject",
        "nearest_refinement_iterations": 2,
    }
    fields.update(overrides)
    return CenterlineSemantics(**fields)


def helix_stations(count: int = 129, pitch: float = PITCH_MM_PER_RAD) -> tuple[GrooveStation, ...]:
    scale = math.sqrt(RADIUS_MM * RADIUS_MM + pitch * pitch)
    stations = []
    for index in range(count):
        theta = 2.0 * math.pi * index / (count - 1)
        tangent, width, normal = helix_frame(theta, pitch)
        stations.append(
            GrooveStation(
                arc_length_mm=scale * theta,
                position_mm=helix_position(theta, pitch),
                tangent=tangent,
                width_direction=width,
                surface_normal=normal,
            )
        )
    return tuple(stations)


def helix_centerline(**overrides) -> GrooveCenterline:
    return GrooveCenterline(
        centerline_id="groove/helix_probe",
        stations=helix_stations(),
        semantics=semantics(**overrides),
        length_unit="mm",
    )


def screw_pose(**overrides) -> AnalyticPose:
    """位姿：绕世界z轴匀速转，平移由"把``σ(t)``那一点送到入带点"**反解**出来。

    这样闭合是**构造出来的**而不是碰巧成立的：机器人六个自由度里，
    姿态可以随便选，平移三个正好把落位点钉在固定入带点上。
    """

    def pose_fn(t_s: float) -> Pose:
        theta = START_THETA_RAD + FEED_RATE_MM_S * t_s / HELIX_SCALE_MM
        angle = -SPIN_RAD_S * t_s
        point = helix_position(theta)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = (point[0] * cos_a - point[1] * sin_a,
                   point[0] * sin_a + point[1] * cos_a, point[2])
        return Pose(
            translation_mm=tuple(ENTRY_POINT_MM[axis] - rotated[axis] for axis in range(3)),
            rotation_xyzw=(0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
        )

    fields = {
        "source_id": "motion/helix_screw",
        "pose_fn": pose_fn,
        "declared_horizon_s": HORIZON_S,
        "extrapolation": "reject",
        "replayable": True,
        "replay_probe_times_s": (0.0, 1.0, 2.0),
    }
    fields.update(overrides)
    return AnalyticPose(**fields)


def world_triad_at(t_s: float):
    """闭式：世界系三标架只随``χ = θ(t) − ω t``转。"""

    theta = START_THETA_RAD + FEED_RATE_MM_S * t_s / HELIX_SCALE_MM
    return helix_frame(theta - SPIN_RAD_S * t_s)


def free_span(entry=ENTRY_POINT_MM, direction=None) -> FreeSpanGeometry:
    direction = world_triad_at(0.0)[0] if direction is None else direction
    return FreeSpanGeometry(
        span_id="span/tension_stand",
        guide_exit_mm=tuple(entry[axis] + SPAN_LENGTH_MM * direction[axis] for axis in range(3)),
        entry_point_mm=entry,
    )


def linear_feed(rate: float = FEED_RATE_MM_S) -> FeedAccount:
    return FeedAccount(
        account_id="feed/constant_rate",
        length_fn=lambda t_s: rate * t_s,
        probe_times_s=(0.0, 1.0, 2.0, 3.0, 4.0),
    )


def model(**overrides) -> LaydownModel:
    fields = {
        "model_id": "laydown/helix_probe",
        "motion": screw_pose(),
        "centerline": helix_centerline(),
        "span": free_span(),
        "feed": linear_feed(),
        "arc_origin_mm": HELIX_SCALE_MM * START_THETA_RAD,
        "rate_probe": ArcRateProbe(scheme="central", step_s=1.0e-3),
    }
    fields.update(overrides)
    return LaydownModel(**fields)


def circle_stations(count: int = 257, radius: float = RADIUS_MM) -> tuple[GrooveStation, ...]:
    """平面圆：``p = 0``。**闭合**，末站点逐位重复首站点。"""

    stations = []
    for index in range(count):
        theta = 2.0 * math.pi * index / (count - 1)
        stations.append(
            GrooveStation(
                arc_length_mm=radius * theta,
                position_mm=(radius * math.cos(theta), radius * math.sin(theta), 0.0),
                tangent=(-math.sin(theta), math.cos(theta), 0.0),
                width_direction=(0.0, 0.0, 1.0),
                surface_normal=(math.cos(theta), math.sin(theta), 0.0),
            )
        )
    stations[-1] = GrooveStation(
        arc_length_mm=2.0 * math.pi * radius,
        position_mm=stations[0].position_mm,
        tangent=stations[0].tangent,
        width_direction=stations[0].width_direction,
        surface_normal=stations[0].surface_normal,
    )
    return tuple(stations)


def closed_circle(**overrides) -> GrooveCenterline:
    return GrooveCenterline(
        centerline_id="groove/plane_circle",
        stations=circle_stations(),
        semantics=semantics(topology="closed", out_of_range="wrap", **overrides),
        length_unit="mm",
    )


# ---------------------------------------------------------------------------
# 绿分支：门认得对的东西
# ---------------------------------------------------------------------------


def test_the_frame_convention_is_the_one_gcw_publishes():
    """绿分支：``width_direction = cross(surface_normal, tangent)``、``(t, s, n)``右手系。

    plans/14第二节把GCW的``centerline.meta.json``那条约定钉死过一次，本模块照抄。
    右手性由``t × s = n``单独再判一次——只判``s = n × t``挡不住整体镜像。
    """

    for theta in (0.0, 0.6, 2.5, 5.9):
        tangent, width, normal = helix_frame(theta)
        station = GrooveStation(
            arc_length_mm=HELIX_SCALE_MM * theta,
            position_mm=helix_position(theta),
            tangent=tangent,
            width_direction=width,
            surface_normal=normal,
        )
        cross = (
            station.tangent[1] * station.width_direction[2]
            - station.tangent[2] * station.width_direction[1],
            station.tangent[2] * station.width_direction[0]
            - station.tangent[0] * station.width_direction[2],
            station.tangent[0] * station.width_direction[1]
            - station.tangent[1] * station.width_direction[0],
        )
        assert cross == pytest.approx(station.surface_normal, abs=1e-12)


def test_the_quaternion_convention_is_bit_for_bit_the_rigidbody_one():
    """绿分支，**而且是本模块最重要的一条对拍**：抄来的公式必须逐位对上原件。

    `laydown`没有import `rigidbody`（那会连带拉进五个模块，抬高冷启动预算），
    而是把`attitude_matrix`那九个表达式抄了一份。**抄一份公式而不对拍它，
    就是本仓最怕的"第二套四元数约定"**——两套约定在数值上不报任何错，
    只是让一个坐标系悄悄差一个转置。
    """

    quaternions = [
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, math.sin(0.37), math.cos(0.37)),
        (0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214),
    ]
    vectors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (3.0, -7.0, 11.5)]
    checked = 0
    for quaternion in quaternions:
        for vector in vectors:
            assert rotate_by_quaternion(quaternion, vector) == rotate_body_to_world(
                quaternion, vector
            )
            checked += 1
    assert checked == 9, "对拍没有真的跑起来——零执行绝不pass"
    #: 反向那条也要对，否则"世界→工件"是另一套约定。
    from physics_engine.laydown import _inverse_rotate

    for quaternion in quaternions:
        for vector in vectors:
            assert _inverse_rotate(quaternion, vector) == rotate_world_to_body(
                quaternion, vector
            )


def test_sampling_at_a_station_returns_that_station():
    """绿分支：插值在站点上必须退化成站点本身，否则收敛阶量的是别的东西。"""

    centerline = helix_centerline()
    for station in (centerline.stations[0], centerline.stations[17], centerline.stations[-1]):
        sample = centerline.sample_at(station.arc_length_mm)
        assert sample.position_mm == pytest.approx(station.position_mm, abs=1e-9)
        assert sample.tangent == pytest.approx(station.tangent, abs=1e-12)
        assert sample.width_direction == pytest.approx(station.width_direction, abs=1e-12)
        assert sample.surface_normal == pytest.approx(station.surface_normal, abs=1e-12)


def test_a_closed_centerline_wraps_the_arc_coordinate_and_unwraps_differences():
    """绿分支：闭合曲线上``L + 3``就是``3``，而跨接缝的差不许差出一整匝。"""

    centerline = closed_circle()
    total = centerline.total_arc_length_mm()
    assert centerline.resolve_arc_length_mm(total + 3.0) == pytest.approx(3.0, abs=1e-9)
    assert centerline.resolve_arc_length_mm(-2.0) == pytest.approx(total - 2.0, abs=1e-9)
    #: 跨接缝：从``L − 1``走到``1``是**前进2 mm**，不是后退``L − 2`` mm。
    assert centerline.arc_difference_mm(1.0, total - 1.0) == pytest.approx(2.0, abs=1e-9)
    #: 开曲线不解卷——它没有接缝可跨。
    assert helix_centerline().arc_difference_mm(1.0, 400.0) == pytest.approx(-399.0)


def test_the_free_span_length_cannot_depend_on_time():
    """绿分支，**订正plans/14第3.3节第一版那个错**：跨段长度是常数。

    第一版把自由跨写成"一端固定R1、一端随臂运动"，于是"臂动⟹跨段变长⟹张力变"
    这条**错误的因果**看起来完全自洽。两个端点冻成常量之后，
    `length_mm()`连``t``都收不到——那条路在类型上就走不通了。
    """

    span = free_span()
    lengths = {span.length_mm() for _ in range(5)}
    assert lengths == {span.length_mm()}
    assert span.length_mm() == pytest.approx(SPAN_LENGTH_MM, rel=1e-12)
    with pytest.raises(TypeError):
        span.length_mm(1.0)  # type: ignore[call-arg]


def test_incidence_is_zero_when_the_tape_continues_the_groove_without_a_kink():
    """绿分支：自由段方向与落位点切向重合时入射角为0。**符号约定的锚点。**

    自由段"材料坐标增大"的方向是**从入带点指向导轮**（模块文档那一段）。
    取反的话理想落位角会是``π``，而一个理想值在``π``的量，
    它的小量展开、容差、在张力式子里的位置全部要跟着变号。

    ``t = 0``上用单边差分：中心差分要取``t = −h``，而那在时间线之外
    （`motion`的``extrapolation='reject'``会当场拒，本模块不替它夹）。

    自由段方向取的是**解析**切向，而落位点切向来自**离散**中心线，
    所以"0"的实测底是采样误差：129/257/1025站点每匝各给
    ``7.42e-05`` / ``2.64e-05`` / ``1.02e-06`` rad。**单点上的比值不是干净的4**
    （2.81然后25.9）——采样误差是**振荡**的，它取决于这一点落在段内哪里。
    所以本条只判"单调下降且到得了机器噪声附近"，
    **收敛阶要用多时刻的最大值去量**，那在`tests/cases/`那条门里。
    写在这里比让读者以为单点比值就是阶诚实。
    """

    probe = ArcRateProbe(scheme="forward", step_s=1.0e-3)
    angles = []
    for count in (129, 257, 1025):
        centerline = GrooveCenterline(
            centerline_id=f"groove/kink{count}",
            stations=helix_stations(count),
            semantics=semantics(),
            length_unit="mm",
        )
        point = model(centerline=centerline, rate_probe=probe).at(0.0)
        assert point.incidence_angle_rad == pytest.approx(
            math.hypot(point.incidence_in_plane_rad, point.incidence_out_of_plane_rad),
            rel=1e-6,
        )
        angles.append(point.incidence_angle_rad)
    assert angles[0] < 1.0e-4, f"粗档就已经不像0了：{angles}"
    assert angles[0] > angles[1] > angles[2], f"入射角没有随采样单调下降：{angles}"
    assert angles[2] < 5.0e-6, f"最密一档还没到噪声附近：{angles}"


def test_reversing_the_free_span_puts_the_ideal_at_pi_not_at_zero():
    """**符号约定的必红面**：把自由段两端对调，理想落位角从0跳到``π``。

    注错方式：把``guide_exit_mm``与``entry_point_mm``互换。
    这一条不判"哪个更对"，它判的是**换了之后确实是另一个量**——
    如果两种取法给同一个数，说明入射角根本没有用到自由段的方向。
    """

    span = free_span()
    reversed_span = FreeSpanGeometry(
        span_id="span/reversed",
        guide_exit_mm=span.entry_point_mm,
        entry_point_mm=span.guide_exit_mm,
    )
    forward = model().at(1.0).incidence_angle_rad
    backward = model(span=reversed_span).at(1.0).incidence_angle_rad
    assert backward == pytest.approx(math.pi - forward, abs=1e-9)
    assert abs(backward - forward) > 1.0, "两种取法给了同一个数——入射角没用到自由段方向"


def test_the_two_declared_frame_semantics_give_different_answers():
    """绿分支：**语义选择是有代价的，所以库不许替调用方猜**。

    同一份站点、同一个时刻，``hold_station``与``reorthonormalised_linear``
    给出的入射角差多少，本条只判"确实不同且差得看得见"——
    差多少、各自几阶，由`tests/cases/test_helix_laydown_closure.py`量。
    """

    coarse = tuple(helix_stations(33))
    held = GrooveCenterline(
        centerline_id="groove/coarse_hold",
        stations=coarse,
        semantics=semantics(frame_interpolation="hold_station"),
        length_unit="mm",
    )
    blended = GrooveCenterline(
        centerline_id="groove/coarse_blend",
        stations=coarse,
        semantics=semantics(frame_interpolation="reorthonormalised_linear"),
        length_unit="mm",
    )
    held_angle = model(centerline=held).at(1.3).incidence_angle_rad
    blended_angle = model(centerline=blended).at(1.3).incidence_angle_rad
    assert abs(held_angle - blended_angle) > 1.0e-3, (
        f"两条帧语义给了几乎相同的答案（{held_angle!r} vs {blended_angle!r}）——"
        "那说明frame_interpolation这个白名单没有载荷，不该让调用方声明它"
    )


def test_the_declared_refinement_count_is_load_bearing():
    """绿分支：``nearest_refinement_iterations=0``在Hermite下把最近点搜索退回弦投影。

    实测（N=64、螺旋线、五档时刻的最大弧长坐标偏差）：
    0步给``4.20e-04`` mm、1步给``1.50e-07`` mm——**差2800倍**，
    而1步之后不再变（牛顿在这条题上一步到位）。
    所以这个数是**真参数**不是装饰，白名单式的失败关闭在这里是对的。
    """

    coarse = helix_stations(65)
    errors = {}
    for iterations in (0, 1, 4):
        centerline = GrooveCenterline(
            centerline_id=f"groove/refine{iterations}",
            stations=coarse,
            semantics=semantics(nearest_refinement_iterations=iterations),
            length_unit="mm",
        )
        worst = 0.0
        for t_s in (0.5, 1.0, 1.7, 2.5, 3.3):
            true_arc = HELIX_SCALE_MM * START_THETA_RAD + FEED_RATE_MM_S * t_s
            measured = model(centerline=centerline).pose_arc_length_mm(t_s)[0]
            worst = max(worst, abs(measured - true_arc))
        errors[iterations] = worst
    assert errors[0] / errors[1] > 100.0, f"细化步数没有起作用：{errors}"
    assert errors[1] == pytest.approx(errors[4], rel=1e-6), f"一步之后还在动：{errors}"


def test_the_residual_separates_what_the_feed_account_can_fix_from_what_it_cannot():
    """绿分支，**这是本轨道的核心判据**：残差按方向拆开之后指向不同的病因。

    两个病人各造一个（数值见`cases/helix_laydown_closure/case.md`第三节）：

    1. **送带账偏0.7 mm**——弧长坐标差恰为0.7、残差几乎全在沿槽分量、
       位姿的不可约偏距**仍然是机器零**。多放0.7 mm带材就对上了；
    2. **位姿把线圈举偏1.5 mm**（入带点沿槽面法向偏1.5 mm）——弧长坐标差是
       机器零（**送带账一点问题都没有**）、残差全在横向、
       且不可约偏距等于1.5 mm。**多放少放带材都修不掉它。**

    一个只报"残差2.3 mm"的实现分不清这两件事，而它们的处置完全相反。

    本条用的是**离散**中心线（129个站点/匝），所以"机器零"那几项的实测底
    是``1.3e-06`` mm量级而不是``1e-16``——那是采样误差不是实现误差，
    收敛阶由`tests/cases/test_helix_laydown_closure.py`量。**先量后写。**
    """

    offset_mm = 0.7
    shifted = model(arc_origin_mm=HELIX_SCALE_MM * START_THETA_RAD + offset_mm).at(1.5)
    assert shifted.closure.arc_gap_mm == pytest.approx(offset_mm, abs=1e-6)
    assert abs(shifted.closure.along_tangent_mm) == pytest.approx(0.699988, abs=1e-5)
    assert shifted.closure.transverse_mm == pytest.approx(0.003479, abs=1e-5)
    assert shifted.closure.pose_only_offset_mm < 1.0e-6, "送带账偏了不该让曲线离开入带点"

    lifted_mm = 1.5
    normal = world_triad_at(0.0)[2]
    lifted_entry = tuple(
        ENTRY_POINT_MM[axis] + lifted_mm * normal[axis] for axis in range(3)
    )
    lifted = model(
        span=free_span(entry=lifted_entry),
        rate_probe=ArcRateProbe(scheme="forward", step_s=1.0e-3),
    ).at(0.0)
    assert abs(lifted.closure.arc_gap_mm) < 1.0e-5, "位姿举偏了不该被算成送带账的账"
    assert abs(lifted.closure.along_tangent_mm) < 1.0e-5
    assert lifted.closure.transverse_mm == pytest.approx(lifted_mm, abs=1e-5)
    assert lifted.closure.pose_only_offset_mm == pytest.approx(lifted_mm, abs=1e-5)
    #: 而"举偏"与"送带账偏"确实不是同一个形态：前者弧长差是采样噪声量级，
    #: 后者是0.7 mm——**两者差五个数量级**，门因此分得开。
    assert abs(shifted.closure.arc_gap_mm) / max(abs(lifted.closure.arc_gap_mm), 1e-18) > 1.0e4


def test_is_replayable_only_forwards_the_pose_source():
    """绿分支：可重放性只转发位姿来源的声明，**不把送带账那次探针包装成证明**。"""

    assert model().is_replayable() is True
    counter = {"n": 0}

    def drifting(t_s: float) -> Pose:
        counter["n"] += 1
        return Pose(
            translation_mm=(float(counter["n"]), 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

    source = AnalyticPose(
        source_id="motion/not_replayable",
        pose_fn=drifting,
        declared_horizon_s=HORIZON_S,
        extrapolation="reject",
        replayable=False,
        replay_probe_times_s=(),
    )
    assert model(motion=source).is_replayable() is False


# ---------------------------------------------------------------------------
# 必须红：五条中心线语义
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "position_interpolation",
        "frame_interpolation",
        "topology",
        "out_of_range",
    ],
)
def test_must_be_red_a_missing_semantic_is_not_defaulted(field: str):
    """**必红**。注错方式：把某一条语义留成``None``（"以后再说"）。

    库替他挑一个"合理默认"的后果是确定的：两个调用方拿同一份站点算出不同的
    落位点、不同的入射角、不同的物理，而**两边都以为自己是对的**。
    """

    with pytest.raises(LaydownError, match="must be declared explicitly"):
        semantics(**{field: None})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_interpolation", "cubic_spline"),
        ("frame_interpolation", "slerp"),
        ("topology", "periodic"),
        ("out_of_range", "extrapolate"),
    ],
)
def test_must_be_red_an_unlisted_semantic_value_is_refused(field: str, value: str):
    """**必红**。注错方式：写一个听起来很合理、但白名单里没有的取值。

    "加一个取值要改laydown.py并补一条测试"——不许在调用方就地放行。
    """

    with pytest.raises(LaydownError, match="must be one of"):
        semantics(**{field: value})


def test_must_be_red_wrap_on_an_open_centerline():
    """**必红**。注错方式：开曲线声明``out_of_range='wrap'``。

    开曲线上``σ = L + 1``是**出界**；闭曲线上它是**站点1 mm之后**。
    静默按后者读，差的是一整匝材料。
    """

    with pytest.raises(LaydownError, match="没有.绕回来.这回事"):
        semantics(topology="open", out_of_range="wrap")


def test_must_be_red_a_closed_centerline_that_does_not_wrap():
    """**必红**（上一条的反面）。注错方式：闭曲线声明``out_of_range='reject'``。"""

    with pytest.raises(LaydownError, match="必须模总长回绕"):
        semantics(topology="closed", out_of_range="reject")


def test_must_be_red_a_negative_refinement_count():
    """**必红**。注错方式：细化步数写成负数。"""

    with pytest.raises(LaydownError, match="不能为负"):
        semantics(nearest_refinement_iterations=-1)


def test_must_be_red_a_bool_refinement_count():
    """**必红**。注错方式：细化步数写``True``——它在Python里是``int``的子类。

    与`motion`那条"``replayable``必须是真``bool``"同源：**回避一个问题
    与回答它，在字节上要能分开**。
    """

    with pytest.raises(LaydownError, match="must be a real int"):
        semantics(nearest_refinement_iterations=True)


# ---------------------------------------------------------------------------
# 必须红：站点帧
# ---------------------------------------------------------------------------


def test_must_be_red_the_width_direction_is_the_other_cross_product():
    """**必红**。注错方式：把``s = n × t``写成``t × n``（整体反号）。

    这一条挡的是WDS `test_gravity_cantilever.py`记过的那族失效——
    参考帧取错让挠度差1600倍而**不报任何错**。取错``n``与``s``在数值上
    同样不报错，只是把带宽方向与法向对调。
    """

    tangent, width, normal = helix_frame(0.6)
    with pytest.raises(LaydownError, match="帧不满足"):
        GrooveStation(
            arc_length_mm=0.0,
            position_mm=helix_position(0.6),
            tangent=tangent,
            width_direction=tuple(-value for value in width),
            surface_normal=normal,
        )


def test_must_be_red_a_non_orthogonal_frame():
    """**必红，而且这一条是注错验证自己抓出来的**。

    第一版的注错方式是"法向朝切向倒0.3"，实测**关掉正交那条门测试照样绿**——
    因为``s = n × t``那条门先把它拦下了，**正交这条门是空的**。

    真正只有正交门抓得住的形态：法向只倒``ε = 1e-5``，带宽方向取
    ``n × t``（**不归一**）。此时``|n × t| = cos ε = 1 − 5e-11``，
    在单位容差``1e-9``之内；``s − n × t``逐位为零。于是前两条门全过，
    只有``|t · n| = sin ε = 1e-5``越过``1e-9``。

    **一条被别的门顺手拦下的必红，说明不了这条门在起作用。**
    """

    tangent, _, normal = helix_frame(0.6)
    epsilon = 1.0e-5
    tilted_raw = tuple(normal[axis] + epsilon * tangent[axis] for axis in range(3))
    scale = math.sqrt(sum(value * value for value in tilted_raw))
    tilted = tuple(value / scale for value in tilted_raw)
    implied_width = (
        tilted[1] * tangent[2] - tilted[2] * tangent[1],
        tilted[2] * tangent[0] - tilted[0] * tangent[2],
        tilted[0] * tangent[1] - tilted[1] * tangent[0],
    )
    #: 前两条门确实过得去——否则这条红又说明不了是哪条门红的。
    assert abs(math.sqrt(sum(v * v for v in implied_width)) - 1.0) < 1.0e-9
    with pytest.raises(LaydownError, match="不正交"):
        GrooveStation(
            arc_length_mm=0.0,
            position_mm=helix_position(0.6),
            tangent=tangent,
            width_direction=implied_width,
            surface_normal=tilted,
        )


def test_must_be_red_a_frame_tilted_far_enough_is_caught_by_the_cross_product_gate():
    """**必红**（上一条的姊妹）：明显歪掉的帧由``s = n × t``那条门抓。

    注错方式：法向朝切向倒0.3。两条门各管一段——写成两条用例，
    是因为**它们红的不是同一条判据**。
    """

    tangent, width, normal = helix_frame(0.6)
    tilted = tuple(
        (normal[axis] + 0.3 * tangent[axis]) / math.sqrt(1.0 + 0.09) for axis in range(3)
    )
    with pytest.raises(LaydownError, match="帧不满足"):
        GrooveStation(
            arc_length_mm=0.0,
            position_mm=helix_position(0.6),
            tangent=tangent,
            width_direction=width,
            surface_normal=tilted,
        )


def test_must_be_red_a_non_unit_frame_vector():
    """**必红**。注错方式：切向不归一（比如直接拿两站点之差当切向）。"""

    tangent, width, normal = helix_frame(0.6)
    with pytest.raises(LaydownError, match="must be a unit vector"):
        GrooveStation(
            arc_length_mm=0.0,
            position_mm=helix_position(0.6),
            tangent=tuple(2.0 * value for value in tangent),
            width_direction=width,
            surface_normal=normal,
        )


# ---------------------------------------------------------------------------
# 必须红：站点表的结构
# ---------------------------------------------------------------------------


def test_must_be_red_arc_lengths_that_do_not_increase():
    """**必红**。注错方式：把两个站点的弧长坐标对调。"""

    stations = list(helix_stations(33))
    stations[5], stations[6] = stations[6], stations[5]
    with pytest.raises(LaydownError, match="必须严格递增"):
        GrooveCenterline(
            centerline_id="groove/unsorted",
            stations=tuple(stations),
            semantics=semantics(),
            length_unit="mm",
        )


def test_must_be_red_arc_lengths_that_do_not_start_at_zero():
    """**必红**。注错方式：整表弧长坐标平移一个常数。"""

    stations = tuple(
        GrooveStation(
            arc_length_mm=station.arc_length_mm + 7.0,
            position_mm=station.position_mm,
            tangent=station.tangent,
            width_direction=station.width_direction,
            surface_normal=station.surface_normal,
        )
        for station in helix_stations(33)
    )
    with pytest.raises(LaydownError, match="必须从0起"):
        GrooveCenterline(
            centerline_id="groove/shifted",
            stations=stations,
            semantics=semantics(),
            length_unit="mm",
        )


def test_must_be_red_an_arc_column_that_is_shorter_than_the_chord():
    """**必红，而且是最常来的一位客人**：把**序号**当弧长坐标传进来。

    注错方式：弧长坐标写成``0, 1, 2, …``。**弧长永远不小于弦长**，
    这一条不是精度问题——它当场说明那一列根本不是弧长。
    """

    stations = tuple(
        GrooveStation(
            arc_length_mm=float(index),
            position_mm=station.position_mm,
            tangent=station.tangent,
            width_direction=station.width_direction,
            surface_normal=station.surface_normal,
        )
        for index, station in enumerate(helix_stations(33))
    )
    with pytest.raises(LaydownError, match="小于弦长"):
        GrooveCenterline(
            centerline_id="groove/index_as_arc",
            stations=stations,
            semantics=semantics(),
            length_unit="mm",
        )


def test_must_be_red_a_sampling_too_coarse_to_resolve_the_curve():
    """**必红**。注错方式：一整匝只采5个站点（每段转72°）。

    ``弧/弦 = θ/(2 sin(θ/2))``：72°给1.069，越过上限
    ``ARC_OVER_CHORD_CEILING``。真实工件采样步2 mm、最小曲率半径27.6 mm，
    每段转4.2°、比值1.0002——**这条门离真实数据有三个数量级余量**。
    """

    with pytest.raises(LaydownError, match="超过上限"):
        GrooveCenterline(
            centerline_id="groove/too_coarse",
            stations=helix_stations(6),
            semantics=semantics(),
            length_unit="mm",
        )
    #: 成对的绿：再密一档就该过（否则这条门在管不该它管的事）。
    assert ARC_OVER_CHORD_CEILING > 1.0
    GrooveCenterline(
        centerline_id="groove/just_dense_enough",
        stations=helix_stations(7),
        semantics=semantics(),
        length_unit="mm",
    )


def test_must_be_red_a_tangent_column_that_points_backwards():
    """**必红**。注错方式：整表切向反号（读CSV时把行序读反的典型后果）。

    整表反号在数值上不报任何错，只是把``s = n × t``整个镜像掉——
    于是带宽方向反了、法向还是对的，而那正好是"不报错的错"。
    """

    stations = tuple(
        GrooveStation(
            arc_length_mm=station.arc_length_mm,
            position_mm=station.position_mm,
            tangent=tuple(-value for value in station.tangent),
            width_direction=tuple(-value for value in station.width_direction),
            surface_normal=station.surface_normal,
        )
        for station in helix_stations(33)
    )
    with pytest.raises(LaydownError, match="切向与前进方向反了"):
        GrooveCenterline(
            centerline_id="groove/reversed",
            stations=stations,
            semantics=semantics(),
            length_unit="mm",
        )


def test_must_be_red_a_closed_centerline_without_the_seam_station():
    """**必红**。注错方式：声明``closed``却不补那个重复首站点的末站点。

    真实CSV就是这样（plans/14第二节记的"首末间隙2.00 mm＝一个采样步"），
    而**没有任何东西能告诉库那一步有多长**——补上闭合站点是声明者的事。
    """

    with pytest.raises(LaydownError, match="逐位重复首站点"):
        GrooveCenterline(
            centerline_id="groove/unclosed",
            stations=helix_stations(33),
            semantics=semantics(topology="closed", out_of_range="wrap"),
            length_unit="mm",
        )


def test_must_be_red_a_frame_jump_at_the_seam():
    """**必红**。注错方式：闭合站点位置对上了，但帧绕切向转了180°。

    位置闭合而帧不闭合，绕过接缝的入射角会跳一个台阶——
    而那个台阶在张力上是一次阶跃载荷。
    """

    stations = list(circle_stations())
    last = stations[-1]
    stations[-1] = GrooveStation(
        arc_length_mm=last.arc_length_mm,
        position_mm=last.position_mm,
        tangent=last.tangent,
        width_direction=tuple(-value for value in last.width_direction),
        surface_normal=tuple(-value for value in last.surface_normal),
    )
    with pytest.raises(LaydownError, match="首末站点.*不一致"):
        GrooveCenterline(
            centerline_id="groove/seam_jump",
            stations=tuple(stations),
            semantics=semantics(topology="closed", out_of_range="wrap"),
            length_unit="mm",
        )


def test_must_be_red_a_centerline_declared_in_metres():
    """**必红**。注错方式：``length_unit='m'``。

    以米声明的中心线会整整差1000倍。**本仓已经栽过两次这个bug**
    （`motion.POSE_TRANSLATION_UNIT`那条常量就是为此立的），
    所以这里失败关闭而不是替调用方换算。
    """

    with pytest.raises(LaydownError, match="length_unit must be one of"):
        GrooveCenterline(
            centerline_id="groove/metres",
            stations=helix_stations(33),
            semantics=semantics(),
            length_unit="m",
        )


def test_must_be_red_two_stations_are_not_a_curve():
    """**必红**。注错方式：只给两个站点。最近点搜索会退化成一条线段。"""

    with pytest.raises(LaydownError, match="至少要三个站点"):
        GrooveCenterline(
            centerline_id="groove/segment",
            stations=helix_stations(33)[:2],
            semantics=semantics(),
            length_unit="mm",
        )


def test_must_be_red_an_arc_coordinate_outside_an_open_centerline():
    """**必红**。注错方式：在开曲线上问一个负的弧长坐标。"""

    with pytest.raises(LaydownError, match="out_of_range='reject'"):
        helix_centerline().sample_at(-1.0)
    #: 成对的绿：声明``clamp_to_end``之后同一个输入应当被夹到端点而不是炸。
    clamped = GrooveCenterline(
        centerline_id="groove/clamped",
        stations=helix_stations(33),
        semantics=semantics(out_of_range="clamp_to_end"),
        length_unit="mm",
    )
    assert clamped.sample_at(-1.0).arc_length_mm == 0.0


# ---------------------------------------------------------------------------
# 必须红：送带账、自由跨段、速率探针
# ---------------------------------------------------------------------------


def test_must_be_red_a_feed_account_that_goes_backwards():
    """**必红**。注错方式：把**放线速度**当累计长度传进来（先增后减的那种）。

    带材不会被吸回张力机。这条门抓的正是"两个量名字都叫feed"这一类。
    """

    with pytest.raises(LaydownError, match="单调非减"):
        FeedAccount(
            account_id="feed/backwards",
            length_fn=lambda t_s: 50.0 * math.sin(t_s),
            probe_times_s=(0.0, 1.0, 2.0, 3.0),
        )


def test_must_be_red_a_feed_account_that_is_not_a_pure_function():
    """**必红**。注错方式：让``length_fn``去读一个计数器（读时钟同理）。

    与`motion.AnalyticPose._probe_determinism`同一条纪律：**库证明不了
    一个函数是纯的**，所以这里做的是证伪——同一个时刻求两次值，不同即拒。
    """

    counter = {"n": 0.0}

    def creeping(t_s: float) -> float:
        counter["n"] += 1.0
        return 50.0 * t_s + counter["n"]

    with pytest.raises(LaydownError, match="不是纯函数"):
        FeedAccount(
            account_id="feed/impure", length_fn=creeping, probe_times_s=(0.0, 1.0)
        )


def test_must_be_red_a_feed_account_with_no_probe_times():
    """**必红**。注错方式：探针留空。一条没有配证伪尝试的单调声明就是冒充。"""

    with pytest.raises(LaydownError, match="至少要给一个探针时刻"):
        FeedAccount(account_id="feed/unprobed", length_fn=lambda t_s: 50.0 * t_s,
                    probe_times_s=())


def test_must_be_red_probe_times_out_of_order():
    """**必红**。注错方式：探针时刻乱序——乱序的探针判不了单调。"""

    with pytest.raises(LaydownError, match="必须严格递增"):
        FeedAccount(
            account_id="feed/unsorted",
            length_fn=lambda t_s: 50.0 * t_s,
            probe_times_s=(0.0, 2.0, 1.0),
        )


def test_must_be_red_a_free_span_with_no_direction():
    """**必红**。注错方式：出带点与入带点写成同一个点。入射角整个建立在这个方向上。"""

    with pytest.raises(LaydownError, match="自由段没有方向"):
        FreeSpanGeometry(
            span_id="span/degenerate",
            guide_exit_mm=ENTRY_POINT_MM,
            entry_point_mm=ENTRY_POINT_MM,
        )


def test_must_be_red_a_rate_probe_without_a_positive_step():
    """**必红**。注错方式：差分步长写0（或负数）。"""

    with pytest.raises(LaydownError, match="step_s必须为正"):
        ArcRateProbe(scheme="central", step_s=0.0)


def test_must_be_red_an_unlisted_rate_scheme():
    """**必红**。注错方式：写一个白名单外的差分格式。"""

    with pytest.raises(LaydownError, match="must be one of"):
        ArcRateProbe(scheme="richardson", step_s=1.0e-3)


def test_must_be_red_a_central_difference_that_walks_off_the_timeline():
    """**必红**。注错方式：在``t = 0``用中心差分。

    **本层不夹到端点**：夹过之后的差商不是那个导数，而是把两倍步长的分母
    配上一倍步长的分子。端点上要么换单边格式，要么把步长调小。
    """

    with pytest.raises(LaydownError, match="落在时间线"):
        model().at(0.0).required_feed_rate_mm_s  # noqa: B018
    #: 成对的绿：换单边格式之后同一个时刻应当算得出来。
    forward = model(rate_probe=ArcRateProbe(scheme="forward", step_s=1.0e-3)).at(0.0)
    assert forward.required_feed_rate_mm_s == pytest.approx(FEED_RATE_MM_S, rel=1e-4)


# ---------------------------------------------------------------------------
# 必须红：闭合门
# ---------------------------------------------------------------------------


def test_must_be_red_a_closure_tolerance_without_a_reason():
    """**必红**。注错方式：容差写了数、理由留空。

    没有理由的容差与"调到能过为止"在字节上没有区别（GROMACS式成对容差的本仓纪律）。
    """

    with pytest.raises(LaydownError, match="必须说明它凭什么是这个数"):
        ClosureTolerance(position_abs_mm=0.1, arc_abs_mm=0.1, reason="   ")


def test_must_be_red_a_negative_closure_tolerance():
    """**必红**。注错方式：容差写成负数——那不是"更严"，那是永远红。"""

    with pytest.raises(LaydownError, match="不能为负"):
        ClosureTolerance(position_abs_mm=-1.0, arc_abs_mm=0.1, reason="试图更严")


def test_must_be_red_closure_gate_catches_a_feed_account_that_drifted():
    """**必红**。注错方式：把送带账的原点挪0.7 mm——闭合门必须当场红。

    这是本轨道最要紧的一条门：位姿与送带账各说各话时，
    今天本仓**没有任何东西看得见它**。
    """

    tolerance = ClosureTolerance(
        position_abs_mm=1.0e-3,
        arc_abs_mm=1.0e-3,
        reason="闭式构造下闭合恒成立，1e-3 mm是离散误差的十倍余量",
    )
    drifted = model(arc_origin_mm=HELIX_SCALE_MM * START_THETA_RAD + 0.7)
    with pytest.raises(LaydownError, match="闭合残差"):
        assert_closure(drifted.track([1.0, 2.0]), tolerance, run_label="run/drifted")
    #: 成对的绿：不挪原点的那一份必须过，否则这条红说明不了任何事。
    assert_closure(model().track([1.0, 2.0]), tolerance, run_label="run/aligned")


def test_must_be_red_closure_gate_catches_an_arc_gap_even_when_positions_agree():
    """**必红**（上一条的第二个分支）：位置容差放宽也要被弧长那一条拦下。

    注错方式：把``position_abs_mm``放到1 mm，只留``arc_abs_mm``守着。
    两条判据各挡一半——只留位置那条，"沿槽偏了一点点但曲线几乎重合"会溜过去。
    """

    loose = ClosureTolerance(
        position_abs_mm=1.0,
        arc_abs_mm=1.0e-3,
        reason="故意只让弧长那一条起作用，判两条判据不是一条的两个写法",
    )
    drifted = model(arc_origin_mm=HELIX_SCALE_MM * START_THETA_RAD + 0.7)
    with pytest.raises(LaydownError, match="弧长坐标差"):
        assert_closure(drifted.track([1.0]), loose, run_label="run/arc_only")


def test_must_be_red_the_closure_gate_refuses_an_empty_run():
    """**必红**。注错方式：把空序列喂给闭合门。**零执行绝不pass。**

    一条"一个落位点都没有"的运行，与一条"每个落位点都合格"的运行，
    在门的返回值上没有区别——除非门自己拒绝前者。
    """

    tolerance = ClosureTolerance(
        position_abs_mm=1.0, arc_abs_mm=1.0, reason="本条只判门会不会拒空"
    )
    with pytest.raises(LaydownError, match="零执行绝不pass"):
        assert_closure([], tolerance, run_label="run/empty")


def test_must_be_red_a_model_whose_motion_is_not_a_motion_source():
    """**必红**。注错方式：把一个只有``pose_at``的对象当位姿来源传进来。

    `MotionSource`是三个方法，缺一就答不上"我可不可重放"。
    """

    class HalfSource:
        def pose_at(self, t_s: float) -> Pose:
            return Pose(translation_mm=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    with pytest.raises(LaydownError, match="不是MotionSource"):
        model(motion=HalfSource())


def test_must_be_red_a_model_id_without_a_namespace():
    """**必红**。注错方式：``model_id``不带命名空间前缀（轴2的稳定ID约定）。"""

    with pytest.raises(LaydownError, match="must be namespaced"):
        model(model_id="helix_probe")
