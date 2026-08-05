"""运动来源层的门——spec/10第二节。

形制照``tests/test_sensors.py``：每条校验一条"必须红"。**外加反证**——
把那条校验换成空操作（等价于写成``if False``），同一个声明当场变绿，
以此排除"红是别的规则顺手红的"这种假通过。反证一律用``monkeypatch``,
它在用例结束时自动还原，不会漏到别的用例上。
"""

from __future__ import annotations

import math
import random

import pytest

from physics_engine import motion
from physics_engine.motion import (
    EXTRAPOLATIONS,
    PAUSE_HOLDS,
    QUATERNION_NORM_ABS_TOL,
    ROTATION_ARCS,
    ROTATION_INTERPOLATIONS,
    SMALL_ANGLE_ONE_MINUS_COS,
    TRANSLATION_INTERPOLATIONS,
    AnalyticPose,
    InterpolationSemantics,
    MotionError,
    MotionSource,
    PauseInterval,
    Pose,
    PoseSample,
    SampledPoseTimeline,
    assert_replayable_for_fingerprint,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _q(angle_deg: float, axis: tuple[float, float, float] = (0.0, 0.0, 1.0)):
    half = math.radians(angle_deg) / 2.0
    norm = math.sqrt(sum(value * value for value in axis))
    unit = tuple(value / norm for value in axis)
    return (*(value * math.sin(half) for value in unit), math.cos(half))


def _semantics(**overrides) -> InterpolationSemantics:
    fields = {
        "translation_interpolation": "linear",
        "rotation_interpolation": "slerp",
        "rotation_arc": "shortest",
        "pause_hold": "hold_interval_start",
        "extrapolation": "reject",
    }
    fields.update(overrides)
    return InterpolationSemantics(**fields)


def _timeline(semantics: InterpolationSemantics | None = None, pauses=None):
    """WII形制的一条位姿时间线：三个样点、一段显式暂停。"""

    return SampledPoseTimeline(
        source_id="motion/wii_layup_pass1",
        samples=(
            PoseSample(0.0, Pose((0.0, 0.0, 0.0), IDENTITY)),
            PoseSample(1.0, Pose((100.0, 0.0, 0.0), _q(90.0))),
            PoseSample(2.0, Pose((100.0, 50.0, 0.0), _q(90.0))),
        ),
        semantics=_semantics() if semantics is None else semantics,
        translation_unit="mm",
        pauses=(
            (PauseInterval("pause/tool_change", 1.0, 1.5, "换刀，机器人保持不动"),)
            if pauses is None
            else pauses
        ),
    )


# --------------------------------------------------------------- 正常路径 ---


def test_a_wii_shaped_timeline_is_accepted_and_answers_the_three_methods():
    timeline = _timeline()
    assert timeline.horizon_s() == 2.0
    assert timeline.is_replayable() is True
    assert timeline.pose_at(0.5).translation_mm == (50.0, 0.0, 0.0)


def test_sample_times_come_back_byte_identical():
    """``pose_at(样点时刻)``必须逐字节等于那个样点。

    ``a + (b − a)·1.0``**不保证**等于``b``，所以两个端点都不许走插值——
    差几个ULP在指纹上是看得见的。
    """

    timeline = _timeline(semantics=_semantics(pause_hold="interpolate_through"))
    for sample in timeline.samples:
        assert timeline.pose_at(sample.time_s) == sample.pose


def test_the_declared_pause_semantics_actually_change_the_answer():
    """同一份样点 + 两种暂停语义 = 两个不同的位姿。**这就是不许库来猜的理由。**"""

    held = _timeline().pose_at(1.25)
    through = _timeline(semantics=_semantics(pause_hold="interpolate_through")).pose_at(1.25)
    assert held.translation_mm == (100.0, 0.0, 0.0)
    assert through.translation_mm == (100.0, 12.5, 0.0)
    assert held != through


def test_the_declared_translation_semantics_actually_change_the_answer():
    linear = _timeline().pose_at(0.5)
    holding = _timeline(semantics=_semantics(translation_interpolation="hold_previous"))
    assert linear.translation_mm == (50.0, 0.0, 0.0)
    assert holding.pose_at(0.5).translation_mm == (0.0, 0.0, 0.0)


def test_slerp_and_nlerp_are_measurably_different_at_a_real_angle():
    """90°一段上slerp与nlerp给出不同的中间姿态——所以它必须被声明，不能被默认。

    **取u=0.25而不是0.5**：半程处nlerp就是两端的归一化平均，与slerp**恰好重合**,
    在那一点上验这条差别永远是假绿的（第一版测试正是这样写的，实测差1.1e-16）。
    """

    with_slerp = _timeline().pose_at(0.25).rotation_xyzw
    with_nlerp = _timeline(semantics=_semantics(rotation_interpolation="nlerp"))
    nlerp_rotation = with_nlerp.pose_at(0.25).rotation_xyzw
    difference = max(
        abs(a - b) for a, b in zip(with_slerp, nlerp_rotation, strict=True)
    )
    # slerp在四分之一处给22.5°；nlerp给的不是22.5°，差得远超单位四元数容差。
    assert difference > 1.0e-3
    assert math.isclose(
        2.0 * math.degrees(math.acos(min(1.0, abs(with_slerp[3])))), 22.5, abs_tol=1e-9
    )
    slerp_angle = 2.0 * math.degrees(math.acos(min(1.0, abs(with_slerp[3]))))
    nlerp_angle = 2.0 * math.degrees(math.acos(min(1.0, abs(nlerp_rotation[3]))))
    assert abs(slerp_angle - nlerp_angle) > 0.1  # 度：肉眼可见的物理差别


def test_the_declared_arc_actually_change_the_answer():
    """``q``与``−q``是同一个旋转，插值却走相反的弧——WII的规范符号归一化正会撞上这个。"""

    samples = (
        PoseSample(0.0, Pose((0.0, 0.0, 0.0), IDENTITY)),
        PoseSample(1.0, Pose((0.0, 0.0, 0.0), tuple(-c for c in _q(120.0)))),
    )
    shortest = SampledPoseTimeline(
        "motion/arc_probe", samples, _semantics(), "mm", ()
    ).pose_at(0.5)
    declared = SampledPoseTimeline(
        "motion/arc_probe", samples, _semantics(rotation_arc="as_declared"), "mm", ()
    ).pose_at(0.5)
    # 短弧走+60°，照声明走的是另一条弧（−120°的一半）。两者的z分量符号相反。
    assert shortest.rotation_xyzw[2] * declared.rotation_xyzw[2] < 0.0


def test_clamping_extrapolation_holds_the_endpoints():
    timeline = _timeline(semantics=_semantics(extrapolation="clamp_to_endpoint"))
    assert timeline.pose_at(-5.0) == timeline.samples[0].pose
    assert timeline.pose_at(99.0) == timeline.samples[-1].pose


def test_an_analytic_trajectory_is_replayable_and_needs_only_one_semantic():
    """解析轨迹没有样点，所以五条里只有``extrapolation``对它有意义。"""

    source = AnalyticPose(
        source_id="motion/spool_rotation",
        pose_fn=lambda t_s: Pose((0.0, 0.0, 0.0), _q(36.0 * t_s)),
        declared_horizon_s=10.0,
        extrapolation="reject",
        replayable=True,
        replay_probe_times_s=(0.0, 3.3, 10.0),
    )
    assert source.horizon_s() == 10.0
    assert source.is_replayable() is True
    assert source.pose_at(0.0).rotation_xyzw == IDENTITY
    assert source.pose_at(5.0) == source.pose_at(5.0)


def test_both_implementations_satisfy_the_spec_protocol():
    assert isinstance(_timeline(), MotionSource)
    assert isinstance(
        AnalyticPose(
            "motion/still", lambda t_s: Pose((0.0, 0.0, 0.0), IDENTITY), 1.0, "reject",
            True, (0.0, 1.0),
        ),
        MotionSource,
    )


# ------------------------------------------------- 必须红：五条插值语义 -----


@pytest.mark.parametrize(
    ("field", "allowed"),
    [
        ("translation_interpolation", TRANSLATION_INTERPOLATIONS),
        ("rotation_interpolation", ROTATION_INTERPOLATIONS),
        ("rotation_arc", ROTATION_ARCS),
        ("pause_hold", PAUSE_HOLDS),
        ("extrapolation", EXTRAPOLATIONS),
    ],
)
def test_a_missing_interpolation_semantic_must_be_rejected(field, allowed):
    """缺一即拒。``None``不是"用默认值"，是"这一条还没想清楚"。"""

    with pytest.raises(MotionError, match="must be declared explicitly"):
        _semantics(**{field: None})
    assert allowed  # 白名单不得为空，否则上面的红是假的


@pytest.mark.parametrize(
    "field",
    [
        "translation_interpolation",
        "rotation_interpolation",
        "rotation_arc",
        "pause_hold",
        "extrapolation",
    ],
)
def test_an_unknown_interpolation_semantic_must_be_rejected(field):
    with pytest.raises(MotionError, match="must be one of"):
        _semantics(**{field: "whatever_seems_reasonable"})


def test_counterproof_the_semantics_whitelist_is_load_bearing(monkeypatch):
    """反证：把白名单校验换成空操作，五条全缺的声明当场变绿。"""

    monkeypatch.setattr(motion, "_require_declared_choice", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion, "_require_arc_matches_rotation", lambda *args: None)
    green = InterpolationSemantics(None, None, None, None, None)
    assert green.translation_interpolation is None


def test_an_arc_that_contradicts_the_rotation_mode_must_be_rejected():
    with pytest.raises(MotionError, match="不存在插值弧"):
        _semantics(rotation_interpolation="hold_previous", rotation_arc="shortest")
    with pytest.raises(MotionError, match="不能是'not_applicable'"):
        _semantics(rotation_interpolation="slerp", rotation_arc="not_applicable")


def test_counterproof_the_arc_compatibility_rule_is_load_bearing(monkeypatch):
    monkeypatch.setattr(motion, "_require_arc_matches_rotation", lambda *args: None)
    green = _semantics(rotation_interpolation="hold_previous", rotation_arc="shortest")
    assert green.rotation_arc == "shortest"


# ----------------------------------------------------- 必须红：单位与时间 ---


@pytest.mark.parametrize("unit", ["m", "M", "metre", "", None])
def test_a_timeline_declared_in_anything_but_millimetres_must_be_rejected(unit):
    """本仓已经栽过两次1000倍单位bug，所以这里失败关闭而不是替声明者换算。"""

    with pytest.raises(MotionError, match="translation_unit"):
        SampledPoseTimeline(
            "motion/metric", _timeline().samples, _semantics(), unit, ()
        )


def test_counterproof_the_unit_gate_is_load_bearing(monkeypatch):
    monkeypatch.setattr(motion, "_require_translation_unit", lambda unit, source_id: unit)
    green = SampledPoseTimeline("motion/metric", _timeline().samples, _semantics(), "m", ())
    assert green.translation_unit == "m"


def test_sample_times_must_start_at_run_start_zero():
    """与WII的``time_s must start at run_start=0``同一个约定，否则0起算是谁的0说不清。"""

    with pytest.raises(MotionError, match="start at run_start=0"):
        SampledPoseTimeline(
            "motion/late",
            (
                PoseSample(0.5, Pose((0.0, 0.0, 0.0), IDENTITY)),
                PoseSample(1.0, Pose((1.0, 0.0, 0.0), IDENTITY)),
            ),
            _semantics(),
            "mm",
            (),
        )


@pytest.mark.parametrize("second_time", [0.0, -1.0, 1.0])
def test_sample_times_must_be_strictly_increasing(second_time):
    samples = (
        PoseSample(0.0, Pose((0.0, 0.0, 0.0), IDENTITY)),
        PoseSample(1.0, Pose((1.0, 0.0, 0.0), IDENTITY)),
        PoseSample(second_time, Pose((2.0, 0.0, 0.0), IDENTITY)),
    )
    with pytest.raises(MotionError, match="strictly increasing|start at run_start"):
        SampledPoseTimeline("motion/backwards", samples, _semantics(), "mm", ())


def test_a_one_sample_timeline_must_be_rejected():
    with pytest.raises(MotionError, match="at least two samples"):
        SampledPoseTimeline(
            "motion/single",
            (PoseSample(0.0, Pose((0.0, 0.0, 0.0), IDENTITY)),),
            _semantics(),
            "mm",
            (),
        )


def test_counterproof_the_sample_time_rules_are_load_bearing(monkeypatch):
    """反证：换掉时间校验，倒着走的样点当场变绿——红确实是那一条红的。"""

    monkeypatch.setattr(motion, "_require_sample_times", lambda samples, source_id: None)
    green = SampledPoseTimeline(
        "motion/backwards",
        (
            PoseSample(9.0, Pose((0.0, 0.0, 0.0), IDENTITY)),
            PoseSample(1.0, Pose((1.0, 0.0, 0.0), IDENTITY)),
        ),
        _semantics(),
        "mm",
        (),
    )
    assert green.horizon_s() == 1.0


# --------------------------------------------------------- 必须红：暂停 -----


def test_overlapping_pauses_must_be_rejected():
    pauses = (
        PauseInterval("pause/a", 0.2, 0.9, "第一次停"),
        PauseInterval("pause/b", 0.5, 1.2, "第二次停"),
    )
    with pytest.raises(MotionError, match="ordered and disjoint"):
        _timeline(pauses=pauses)


def test_a_pause_past_the_horizon_must_be_rejected():
    with pytest.raises(MotionError, match="past the timeline horizon"):
        _timeline(pauses=(PauseInterval("pause/late", 1.0, 5.0, "停到时间线之外"),))


def test_counterproof_the_pause_range_rule_is_load_bearing(monkeypatch):
    monkeypatch.setattr(motion, "_require_pauses_in_range", lambda *args: None)
    green = _timeline(pauses=(PauseInterval("pause/late", 1.0, 5.0, "停到时间线之外"),))
    assert green.pauses[0].end_time_s == 5.0


def test_a_pause_without_a_reason_must_be_rejected():
    """一段没有理由的暂停与一段丢失的数据在字节上没有区别。"""

    with pytest.raises(MotionError, match="must say why"):
        PauseInterval("pause/mystery", 0.1, 0.2, "   ")


@pytest.mark.parametrize(("start", "end"), [(-0.1, 0.5), (0.5, 0.5), (0.5, 0.4)])
def test_a_pause_needs_zero_le_start_lt_end(start, end):
    with pytest.raises(MotionError, match="0 <= start < end"):
        PauseInterval("pause/bad", start, end, "理由在，区间不在")


# ------------------------------------------------------- 必须红：四元数 -----


@pytest.mark.parametrize(
    "rotation",
    [(0.0, 0.0, 0.0, 0.5), (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
)
def test_a_non_unit_quaternion_must_be_rejected(rotation):
    with pytest.raises(MotionError, match="unit quaternion|4-tuple"):
        Pose((0.0, 0.0, 0.0), rotation)


def test_counterproof_the_unit_quaternion_rule_is_load_bearing(monkeypatch):
    monkeypatch.setattr(motion, "_require_unit_quaternion", lambda rotation, what: rotation)
    green = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.5))
    assert green.rotation_xyzw == (0.0, 0.0, 0.0, 0.5)


def test_the_unit_quaternion_tolerance_matches_shapes():
    """两处判"这是不是单位四元数"必须给同一个答案，否则不一致最难查。"""

    from physics_engine import shapes

    body_source = shapes.PosedBody.__post_init__.__code__.co_consts
    assert QUATERNION_NORM_ABS_TOL in body_source, (
        "shapes.PosedBody的单位四元数容差与motion.QUATERNION_NORM_ABS_TOL不一致了"
    )


#: 反号的一对四元数：**同一个姿态**，但中间那条弧是绕某个说不出来的轴转一整圈。
#: 注意``dot ≈ −1``不是"旋转差180°"——180°对应``dot ≈ 0``（第一版测试写错了这一点,
#: 用``_q(180)``去构造反号，实测``dot=6.1e-17``，门当然不响）。
ANTIPODAL_SAMPLES = (
    PoseSample(0.0, Pose((0.0, 0.0, 0.0), IDENTITY)),
    PoseSample(1.0, Pose((0.0, 0.0, 0.0), tuple(-value for value in IDENTITY))),
)


def test_antipodal_samples_under_as_declared_must_be_rejected():
    """四元数反号时slerp的转轴不唯一——这不是数值问题，是声明问题。"""

    timeline = SampledPoseTimeline(
        "motion/flip", ANTIPODAL_SAMPLES, _semantics(rotation_arc="as_declared"), "mm", ()
    )
    with pytest.raises(MotionError, match="转轴不唯一"):
        timeline.pose_at(0.5)


def test_the_antipodal_rejection_also_covers_nlerp():
    """nlerp在反号一对上会把两个四元数加成零向量——同一条门必须先拦下它。"""

    timeline = SampledPoseTimeline(
        "motion/flip",
        ANTIPODAL_SAMPLES,
        _semantics(rotation_interpolation="nlerp", rotation_arc="as_declared"),
        "mm",
        (),
    )
    with pytest.raises(MotionError, match="转轴不唯一"):
        timeline.pose_at(0.5)


def test_the_shortest_arc_never_reaches_the_antipodal_branch():
    """``shortest``翻符号后``dot >= 0``，所以上一条的拒绝分支它到不了。

    反号的一对在短弧口径下就是**同一个姿态**，整段插值恒等于那个姿态。
    """

    timeline = SampledPoseTimeline("motion/flip", ANTIPODAL_SAMPLES, _semantics(), "mm", ())
    for u in (0.25, 0.5, 0.75):
        assert timeline.pose_at(u).rotation_xyzw == IDENTITY


def test_a_180_degree_step_is_not_antipodal_and_interpolates_fine():
    """定量分清两件事：180°旋转给``dot ≈ 0``，反号才给``dot ≈ −1``。"""

    quarter = SampledPoseTimeline(
        "motion/half_turn",
        (
            PoseSample(0.0, Pose((0.0, 0.0, 0.0), IDENTITY)),
            PoseSample(1.0, Pose((0.0, 0.0, 0.0), _q(180.0))),
        ),
        _semantics(rotation_arc="as_declared"),
        "mm",
        (),
    ).pose_at(0.5)
    assert math.isclose(abs(_q(180.0)[3]), 0.0, abs_tol=1e-15)  # dot with identity ≈ 0
    assert math.isclose(
        2.0 * math.degrees(math.acos(min(1.0, abs(quarter.rotation_xyzw[3])))),
        90.0,
        abs_tol=1e-9,
    )


# ------------------------------------------------------- 必须红：外推 -------


@pytest.mark.parametrize("t_s", [-0.001, 2.001, 1.0e6])
def test_extrapolation_reject_actually_rejects(t_s):
    with pytest.raises(MotionError, match="extrapolation='reject'"):
        _timeline().pose_at(t_s)


@pytest.mark.parametrize("t_s", [float("nan"), float("inf"), None, "0.5"])
def test_a_nonsense_query_time_must_be_rejected(t_s):
    with pytest.raises(MotionError, match="t_s"):
        _timeline().pose_at(t_s)


# ------------------------------- 必须红：可重放声明与轴3规则5的联动门 -----


def test_an_impure_pose_function_claiming_replayability_must_be_rejected():
    """读随机数的轨迹声称可重放，构造期双求值当场证伪。"""

    generator = random.Random(0)
    with pytest.raises(MotionError, match="not a pure function|不是纯函数"):
        AnalyticPose(
            "motion/noisy",
            lambda t_s: Pose((generator.random(), 0.0, 0.0), IDENTITY),
            1.0,
            "reject",
            True,
            (0.5,),
        )


def test_a_clock_reading_pose_function_is_caught_the_same_way():
    counter = {"n": 0}

    def drifting(t_s: float) -> Pose:
        counter["n"] += 1
        return Pose((float(counter["n"]), 0.0, 0.0), IDENTITY)

    with pytest.raises(MotionError, match="不是纯函数"):
        AnalyticPose("motion/drift", drifting, 1.0, "reject", True, (0.0,))


def test_counterproof_the_determinism_probe_is_load_bearing(monkeypatch):
    """反证：不跑双求值，同一个读随机数的轨迹当场变绿并自称可重放。"""

    monkeypatch.setattr(AnalyticPose, "_probe_determinism", lambda self: None)
    generator = random.Random(0)
    green = AnalyticPose(
        "motion/noisy",
        lambda t_s: Pose((generator.random(), 0.0, 0.0), IDENTITY),
        1.0,
        "reject",
        True,
        (0.5,),
    )
    assert green.is_replayable() is True
    assert green.pose_at(0.5) != green.pose_at(0.5)


def test_claiming_replayability_without_a_probe_time_must_be_rejected():
    """一条没有配证伪尝试的可重放声明就是冒充。"""

    with pytest.raises(MotionError, match="at least one probe time"):
        AnalyticPose(
            "motion/unchecked",
            lambda t_s: Pose((0.0, 0.0, 0.0), IDENTITY),
            1.0,
            "reject",
            True,
            (),
        )


def test_a_non_replayable_source_must_not_enter_a_fingerprinted_run():
    """**spec/10第二节 × 轴3规则5的联动门。**"""

    live = AnalyticPose(
        "motion/teleop",
        lambda t_s: Pose((0.0, 0.0, 0.0), IDENTITY),
        1.0,
        "reject",
        False,
        (),
    )
    assert live.is_replayable() is False
    with pytest.raises(MotionError, match="must not claim a reproduction fingerprint"):
        assert_replayable_for_fingerprint([_timeline(), live], run_label="run/nightly")


def test_counterproof_the_fingerprint_gate_is_load_bearing(monkeypatch):
    """反证：把门换成空操作，同一份不可重放的来源当场进了要指纹的运行。"""

    monkeypatch.setattr(
        motion, "assert_replayable_for_fingerprint", lambda sources, *, run_label: None
    )
    live = AnalyticPose(
        "motion/teleop", lambda t_s: Pose((0.0, 0.0, 0.0), IDENTITY), 1.0, "reject",
        False, (),
    )
    motion.assert_replayable_for_fingerprint([live], run_label="run/nightly")


def test_a_source_that_dodges_the_question_is_treated_as_not_replayable():
    """返回``1``或``"yes"``是在回避，不是在回答——按失败关闭处理。"""

    class Dodger:
        source_id = "motion/dodger"

        def pose_at(self, t_s):
            return Pose((0.0, 0.0, 0.0), IDENTITY)

        def horizon_s(self):
            return 1.0

        def is_replayable(self):
            return 1

    with pytest.raises(MotionError, match="not a bool"):
        assert_replayable_for_fingerprint([Dodger()], run_label="run/nightly")


def test_something_that_is_not_a_motion_source_at_all_is_rejected():
    with pytest.raises(MotionError, match="is not a MotionSource"):
        assert_replayable_for_fingerprint(["motion/a_string"], run_label="run/nightly")


@pytest.mark.parametrize("label", ["", "   ", None, 7])
def test_the_fingerprint_gate_needs_a_run_label(label):
    with pytest.raises(MotionError, match="run_label"):
        assert_replayable_for_fingerprint([_timeline()], run_label=label)


@pytest.mark.parametrize("replayable", [1, 0, "yes", None])
def test_an_analytic_source_must_declare_replayability_as_a_real_bool(replayable):
    with pytest.raises(MotionError, match="must be a real bool"):
        AnalyticPose(
            "motion/vague", lambda t_s: Pose((0.0, 0.0, 0.0), IDENTITY), 1.0, "reject",
            replayable, (0.0,),
        )


def test_a_pose_function_that_does_not_return_a_pose_must_be_rejected():
    with pytest.raises(MotionError, match="not a Pose"):
        AnalyticPose("motion/wrong", lambda t_s: (0.0, 0.0, 0.0), 1.0, "reject", True, (0.0,))


@pytest.mark.parametrize("horizon", [0.0, -1.0, float("inf"), float("nan")])
def test_an_analytic_horizon_must_be_positive_and_finite(horizon):
    with pytest.raises(MotionError, match="horizon"):
        AnalyticPose(
            "motion/bad_horizon", lambda t_s: Pose((0.0, 0.0, 0.0), IDENTITY), horizon,
            "reject", True, (0.0,),
        )


def test_a_probe_time_outside_the_horizon_must_be_rejected():
    with pytest.raises(MotionError, match="outside"):
        AnalyticPose(
            "motion/bad_probe", lambda t_s: Pose((0.0, 0.0, 0.0), IDENTITY), 1.0,
            "reject", True, (2.0,),
        )


@pytest.mark.parametrize(
    "identifier", ["wii_layup", "timeline/wii", "motion/a/b", 7, None]
)
def test_a_source_id_must_be_namespaced(identifier):
    with pytest.raises(MotionError, match="source_id"):
        SampledPoseTimeline(identifier, _timeline().samples, _semantics(), "mm", ())


def test_the_semantics_argument_is_type_checked():
    with pytest.raises(MotionError, match="must be an InterpolationSemantics"):
        SampledPoseTimeline("motion/typo", _timeline().samples, "linear", "mm", ())


# ------------------------------------------------- 容差本身要被验（轴6规则6）


def _max_component_deviation(theta_rad: float, seed: int = 11) -> float:
    """同一对四元数上slerp与nlerp的最大分量分歧。"""

    generator = random.Random(seed)
    semantics_slerp = _semantics()
    semantics_nlerp = _semantics(rotation_interpolation="nlerp")
    worst = 0.0
    for _ in range(200):
        axis = tuple(generator.gauss(0.0, 1.0) for _ in range(3))
        start = tuple(generator.gauss(0.0, 1.0) for _ in range(4))
        norm = math.sqrt(sum(value * value for value in start))
        left = tuple(value / norm for value in start)
        delta = _q(math.degrees(theta_rad), axis)
        x0, y0, z0, w0 = left
        x1, y1, z1, w1 = delta
        raw = (
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
        )
        raw_norm = math.sqrt(sum(value * value for value in raw))
        right = tuple(value / raw_norm for value in raw)
        for u in (0.25, 0.5, 0.75):
            a = motion._interpolate_rotation(left, right, u, semantics_slerp)
            b = motion._interpolate_rotation(left, right, u, semantics_nlerp)
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b, strict=True)))
    return worst


def test_the_small_angle_threshold_is_where_slerp_and_nlerp_stop_being_distinguishable():
    """**容差是算出来的**：阈值处两条路的分歧必须落在单位四元数容差之下，
    而十倍于阈值角处必须已经越过它——否则这个阈值要么白设要么设窄了。"""

    threshold_rad = math.acos(1.0 - SMALL_ANGLE_ONE_MINUS_COS)
    assert _max_component_deviation(threshold_rad) < QUATERNION_NORM_ABS_TOL
    assert _max_component_deviation(10.0 * threshold_rad) > QUATERNION_NORM_ABS_TOL


# ------------------------------------------------------------- 范围与预算 ---


def test_the_module_stops_where_spec_10_is_still_a_draft():
    """本模块只做运动学。求解器、状态、力一概不在这里——这条门守的是范围。"""

    for name in ("apply", "step", "solve", "force_n", "energy_j"):
        assert not hasattr(SampledPoseTimeline, name)
        assert not hasattr(AnalyticPose, name)
    assert motion.__all__ == [
        "ACCEPTED_TRANSLATION_UNITS",
        "EXTRAPOLATIONS",
        "MILLIMETRES_PER_METRE",
        "PAUSE_HOLDS",
        "POSE_TRANSLATION_UNIT",
        "QUATERNION_NORM_ABS_TOL",
        "ROTATION_ARCS",
        "ROTATION_INTERPOLATIONS",
        "SMALL_ANGLE_ONE_MINUS_COS",
        "TRANSLATION_INTERPOLATIONS",
        "AnalyticPose",
        "InterpolationSemantics",
        "MotionError",
        "MotionSource",
        "PauseInterval",
        "Pose",
        "PoseSample",
        "SampledPoseTimeline",
        "assert_replayable_for_fingerprint",
    ]


def test_motion_does_not_enter_the_top_level_re_export():
    """预算纪律：实验档模块不进顶层``__init__``（那是eager import成本）。"""

    import physics_engine

    assert not hasattr(physics_engine, "SampledPoseTimeline")
    assert "SampledPoseTimeline" not in physics_engine.__all__


def test_motion_does_not_import_any_physics_domain():
    """域隔离门的模块级复述：位姿来源不是状态（spec/12第2.3节），所以够不着力学。"""

    source = (motion.__file__ or "")
    assert source.endswith("motion.py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in ("physics_engine.state", "physics_engine.energies", "physics_engine.solve"):
        assert forbidden not in text
