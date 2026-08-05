"""`SceneAssembly`的门——spec/10第一节的装配容器（决策0045）。

本轨没有物理判据可用（装配层不算任何东西），所以门只能建在两件事上：
**失败关闭**与**确定性**。两类门各自的纪律：

* **每条校验都要红过**，且要红在自己那一条上。本文件的每个必红用例都
  **只犯一个错**——别的字段一律合法，这样"红了"才证明得了是那一条红的，
  不是别的规则顺手拦下的（`tests/governance/test_domain_isolation.py`同款纪律）；
* **确定性要逐字节验**，而且要**跨进程**验：同一进程里集合的迭代次序是同一个
  哈希种子下的次序，进程内比两次抓不到"次序随`PYTHONHASHSEED`变"这类隐患。
  这条直接采`cases/generator_determinism`对`modelgen`的做法。
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from physics_engine.actuators import ActuatorDeclaration, CommandChannel
from physics_engine.collision import BroadPhaseCollisionQuery
from physics_engine.motion import AnalyticPose, MotionError, Pose, SampledPoseTimeline
from physics_engine.scene import (
    AssembledBody,
    ContactPair,
    FinalizedScene,
    SceneAssembly,
    SceneError,
)
from physics_engine.shapes import CollisionShape, PosedBody, SimBody, Sphere

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SCRIPT = ROOT / "examples/winding_line_scene.py"


# ------------------------------------------------------------------ 素材 ---


def body(name: str, *, translation=(0.0, 0.0, 0.0), radius_mm: float = 5.0) -> PosedBody:
    return PosedBody(
        body=SimBody(
            body_id=name,
            collision=CollisionShape(shape=Sphere(radius_mm=radius_mm), direction="fitted"),
        ),
        translation_mm=translation,
    )


def spin(source_id: str = "motion/spin", *, x_mm: float = 0.0) -> AnalyticPose:
    """一条合法的解析轨迹：原地不动，姿态恒等。纯函数，可重放。"""

    return AnalyticPose(
        source_id=source_id,
        pose_fn=lambda t_s: Pose(
            translation_mm=(x_mm + t_s, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)
        ),
        declared_horizon_s=10.0,
        extrapolation="reject",
        replayable=True,
        replay_probe_times_s=(0.0, 10.0),
    )


def clutch(actuator_id: str = "actuator/tension") -> ActuatorDeclaration:
    return ActuatorDeclaration(
        actuator_id=actuator_id,
        kind="magnetic_particle_clutch",
        channels=(
            CommandChannel(
                channel_id="command/clutch_torque",
                quantity_id="clutch_torque_nmm",
                dimension=1,
                lower=(0.0,),
                upper=(4000.0,),
            ),
        ),
        delay_s=0.02,
        zero_delay_rationale=None,
    )


def two_body_assembly() -> SceneAssembly:
    """一份**处处合法**的两体装配，必红用例在它上面各犯一个错。"""

    assembly = SceneAssembly("scene/two_body")
    assembly.declare_body(body("body/a"))
    assembly.declare_body(body("body/b", translation=(20.0, 0.0, 0.0)))
    return assembly


# ------------------------------------------------------------ 装配成不成立 ---


def test_a_minimal_assembly_finalizes_and_keeps_declaration_order():
    assembly = two_body_assembly()
    assembly.declare_body(body("body/c", translation=(40.0, 0.0, 0.0)))
    assembly.declare_contact_between("body/b", "body/c")
    assembly.declare_contact_between("body/a", "body/b")
    assembly.declare_allowed_pair("body/a", "body/c")

    scene = assembly.finalize()
    assert isinstance(scene, FinalizedScene)
    assert scene.body_ids == ("body/a", "body/b", "body/c")
    # 次序即声明次序——不是排序后的次序。
    assert scene.contact_pairs == (
        ContactPair(body_a="body/b", body_b="body/c"),
        ContactPair(body_a="body/a", body_b="body/b"),
    )
    assert scene.allowed_pairs == frozenset({frozenset({"body/a", "body/c"})})


def test_declaration_order_of_bodies_and_pairs_does_not_matter():
    """先声明接触对、后声明体，与反过来，必须是同一个场景。

    这正是"组装期只登记，finalize统一校验"要买到的东西：
    交叉引用一律留到finalize解，声明的次序因此不决定装配成不成立。
    """

    forward = SceneAssembly("scene/order")
    forward.declare_body(body("body/a"))
    forward.declare_body(body("body/b"))
    forward.declare_contact_between("body/a", "body/b")

    backward = SceneAssembly("scene/order")
    backward.declare_contact_between("body/a", "body/b")
    backward.declare_body(body("body/a"))
    backward.declare_body(body("body/b"))

    assert forward.finalize().assembly_manifest_bytes() == (
        backward.finalize().assembly_manifest_bytes()
    )


def test_a_body_carries_its_motion_source_and_actuator():
    assembly = two_body_assembly()
    assembly.declare_motion_source("body/a", spin())
    assembly.declare_actuator("body/b", clutch())
    scene = assembly.finalize()

    first, second = scene.bodies
    assert isinstance(first, AssembledBody)
    assert first.motion_source is not None and first.actuator is None
    assert second.motion_source is None
    assert second.actuator is not None and second.actuator.actuator_id == "actuator/tension"


def test_a_sampled_timeline_is_accepted_as_a_motion_source():
    """`MotionSource`是Protocol，两个实现都必须进得来（不是只认`AnalyticPose`）。"""

    from physics_engine.motion import InterpolationSemantics, PoseSample

    timeline = SampledPoseTimeline(
        source_id="motion/sampled",
        samples=(
            PoseSample(
                time_s=0.0,
                pose=Pose(translation_mm=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            ),
            PoseSample(
                time_s=2.0,
                pose=Pose(translation_mm=(10.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            ),
        ),
        semantics=InterpolationSemantics(
            translation_interpolation="linear",
            rotation_interpolation="hold_previous",
            rotation_arc="not_applicable",
            pause_hold="hold_interval_start",
            extrapolation="reject",
        ),
        translation_unit="mm",
        pauses=(),
    )
    assembly = SceneAssembly("scene/sampled")
    assembly.declare_body(body("body/a"))
    assembly.declare_motion_source("body/a", timeline)
    scene = assembly.finalize()
    assert scene.posed_bodies_at(1.0)[0].translation_mm == (5.0, 0.0, 0.0)


# ------------------------------------------------------------ 位姿与时刻 ---


def test_a_static_body_returns_the_very_object_it_was_declared_with():
    """静态体的位姿逐字节是声明时那个——不经任何插值、不经任何浮点运算。"""

    declared = body("body/a", translation=(3.0, 0.0, 0.0))
    assembly = SceneAssembly("scene/static")
    assembly.declare_body(declared)
    scene = assembly.finalize()
    assert scene.posed_bodies_at(7.0)[0] is declared


def test_a_driven_body_takes_its_pose_from_the_motion_source():
    assembly = SceneAssembly("scene/driven")
    assembly.declare_body(body("body/a"))
    assembly.declare_motion_source("body/a", spin())
    scene = assembly.finalize()
    assert scene.posed_bodies_at(4.0)[0].translation_mm == (4.0, 0.0, 0.0)
    # 体本身（几何、id）不随时刻变，变的只有位姿。
    assert scene.posed_bodies_at(4.0)[0].body is scene.bodies[0].posed.body


def test_a_time_past_the_horizon_fails_closed_through_the_source():
    """越界不由装配层发明处理——运动源自己的`extrapolation`说了算。"""

    assembly = SceneAssembly("scene/horizon")
    assembly.declare_body(body("body/a"))
    assembly.declare_motion_source("body/a", spin())
    scene = assembly.finalize()
    with pytest.raises(MotionError, match="outside"):
        scene.posed_bodies_at(11.0)


# ------------------------------------------------- 六条必红（逐条只犯一个错）---


def test_red_duplicate_body_id():
    assembly = two_body_assembly()
    assembly.declare_body(body("body/a", translation=(60.0, 0.0, 0.0)))
    with pytest.raises(SceneError, match=r"duplicate body_id in assembly.*body/a"):
        assembly.finalize()


def test_red_contact_pair_points_at_a_body_that_does_not_exist():
    assembly = two_body_assembly()
    assembly.declare_contact_between("body/a", "body/ghost")
    with pytest.raises(SceneError, match=r"contact pair.*unknown bodies.*body/ghost"):
        assembly.finalize()


def test_red_motion_source_bound_to_a_body_that_does_not_exist():
    assembly = two_body_assembly()
    assembly.declare_motion_source("body/ghost", spin())
    with pytest.raises(SceneError, match=r"motion source bound to unknown body.*body/ghost"):
        assembly.finalize()


@pytest.mark.parametrize(
    ("call", "label"),
    [
        (lambda a: a.declare_body(body("body/late")), "declare_body"),
        (lambda a: a.declare_motion_source("body/a", spin()), "declare_motion_source"),
        (lambda a: a.declare_actuator("body/a", clutch()), "declare_actuator"),
        (lambda a: a.declare_contact_between("body/a", "body/b"), "declare_contact_between"),
        (lambda a: a.declare_allowed_pair("body/a", "body/b"), "declare_allowed_pair"),
    ],
)
def test_red_any_declaration_after_finalize(call, label):
    """finalize后场景不可变——**五个入口一个都不许开**，不是只堵住最常用的那个。"""

    assembly = two_body_assembly()
    assembly.finalize()
    with pytest.raises(SceneError, match=rf"finalized; {label} is no longer allowed"):
        call(assembly)


@pytest.mark.parametrize("second", [("body/a", "body/b"), ("body/b", "body/a")])
def test_red_the_same_contact_pair_declared_twice(second):
    """`(A, B)`与`(B, A)`是同一对——反着写一遍同样要红，否则去重就是假的。"""

    assembly = two_body_assembly()
    assembly.declare_contact_between("body/a", "body/b")
    assembly.declare_contact_between(*second)
    with pytest.raises(SceneError, match="is declared twice"):
        assembly.finalize()


def test_red_allowed_pair_contradicts_a_declared_contact_pair():
    """同一对既要算接触、又允许重叠不报——两条声明对同一对给出相反的处置。"""

    assembly = two_body_assembly()
    assembly.declare_contact_between("body/a", "body/b")
    assembly.declare_allowed_pair("body/b", "body/a")
    with pytest.raises(
        SceneError, match="declared both.*as a contact pair and as an allowed pair"
    ):
        assembly.finalize()


# --------------------------------------------- 另外八条（同样逐条只犯一个错）---


def test_red_actuator_bound_to_a_body_that_does_not_exist():
    assembly = two_body_assembly()
    assembly.declare_actuator("body/ghost", clutch())
    with pytest.raises(SceneError, match=r"actuator.*bound to unknown body.*body/ghost"):
        assembly.finalize()


def test_red_two_motion_sources_on_one_body():
    assembly = two_body_assembly()
    assembly.declare_motion_source("body/a", spin("motion/first"))
    assembly.declare_motion_source("body/a", spin("motion/second"))
    with pytest.raises(SceneError, match="already has a motion source"):
        assembly.finalize()


def test_red_two_actuators_on_one_body():
    assembly = two_body_assembly()
    assembly.declare_actuator("body/a", clutch("actuator/first"))
    assembly.declare_actuator("body/a", clutch("actuator/second"))
    with pytest.raises(SceneError, match="already has actuator"):
        assembly.finalize()


def test_red_contact_declared_between_a_body_and_itself():
    assembly = two_body_assembly()
    assembly.declare_contact_between("body/a", "body/a")
    with pytest.raises(SceneError, match="contact declared between .* and itself"):
        assembly.finalize()


def test_red_allowed_pair_points_at_a_body_that_does_not_exist():
    assembly = two_body_assembly()
    assembly.declare_allowed_pair("body/a", "body/ghost")
    with pytest.raises(SceneError, match=r"allowed pair references unknown bodies.*ghost"):
        assembly.finalize()


def test_red_a_body_with_both_a_static_pose_and_a_motion_source():
    """位姿有两个来源，本层不替调用方挑一个（也不发明复合约定）。"""

    assembly = SceneAssembly("scene/two_poses")
    assembly.declare_body(body("body/a", translation=(11.0, 0.0, 0.0)))
    assembly.declare_motion_source("body/a", spin())
    with pytest.raises(SceneError, match="both a nonidentity static pose"):
        assembly.finalize()


def test_red_an_empty_assembly():
    with pytest.raises(SceneError, match="needs at least one body"):
        SceneAssembly("scene/empty").finalize()


def test_red_a_scene_id_without_a_namespace():
    with pytest.raises(SceneError, match="must be namespaced"):
        SceneAssembly("winding_line")


@pytest.mark.parametrize(
    ("call", "pattern"),
    [
        (lambda a: a.declare_body("body/a"), "expects a PosedBody"),
        (lambda a: a.declare_motion_source("body/a", object()), "must implement MotionSource"),
        (lambda a: a.declare_actuator("body/a", object()), "must be an ActuatorDeclaration"),
        (lambda a: a.declare_contact_between("body/a", 7), "must be a nonempty string"),
        (lambda a: a.declare_allowed_pair("", "body/b"), "must be a nonempty string"),
    ],
)
def test_red_types_are_checked_at_declaration_time(call, pattern):
    """交叉引用留到finalize，**类型**不留——一个不是`PosedBody`的东西
    在登记那一刻就说得清是错的，留到finalize只会让错误离现场更远。"""

    with pytest.raises(SceneError, match=pattern):
        call(two_body_assembly())


# ------------------------------------------------------------ 不可变性 ---


def test_the_assembly_is_open_before_finalize_and_shut_after():
    assembly = two_body_assembly()
    assert assembly.is_finalized is False
    assembly.declare_contact_between("body/a", "body/b")  # 之前：受理
    assembly.finalize()
    assert assembly.is_finalized is True


def test_a_refused_late_declaration_does_not_leak_into_the_finalized_scene():
    """炸了之后，已产出的场景必须一个字节都没变——半截成功比直接失败更糟。"""

    assembly = two_body_assembly()
    scene = assembly.finalize()
    before = scene.assembly_manifest_bytes()
    with pytest.raises(SceneError):
        assembly.declare_body(body("body/late"))
    with pytest.raises(SceneError):
        assembly.declare_contact_between("body/a", "body/b")
    assert scene.assembly_manifest_bytes() == before
    assert scene.body_ids == ("body/a", "body/b")


@pytest.mark.parametrize("field", ["scene_id", "bodies", "contact_pairs", "allowed_pairs"])
def test_the_finalized_scene_is_frozen(field):
    scene = two_body_assembly().finalize()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(scene, field, None)


def test_the_finalized_scene_hands_no_mutable_collections_out():
    """元组与frozenset，不是list与set——拿到场景的人改不动它。"""

    scene = two_body_assembly().finalize()
    assert isinstance(scene.bodies, tuple)
    assert isinstance(scene.contact_pairs, tuple)
    assert isinstance(scene.allowed_pairs, frozenset)


def test_finalize_twice_gives_two_equivalent_scenes():
    """finalize是纯读取，封死的是declare那一侧——重复调用不该给出不同的场景。"""

    assembly = two_body_assembly()
    assembly.declare_contact_between("body/a", "body/b")
    first, second = assembly.finalize(), assembly.finalize()
    assert first.assembly_manifest_bytes() == second.assembly_manifest_bytes()
    assert first.contact_pairs == second.contact_pairs


# -------------------------------------------------------------- 确定性 ---


def test_two_independent_builds_agree_byte_for_byte():
    def build() -> bytes:
        assembly = SceneAssembly("scene/determinism")
        for name in ("body/z", "body/a", "body/m"):
            assembly.declare_body(body(name))
        assembly.declare_contact_between("body/z", "body/m")
        assembly.declare_contact_between("body/a", "body/z")
        assembly.declare_allowed_pair("body/a", "body/m")
        return assembly.finalize().assembly_manifest_bytes()

    assert build() == build()


def test_the_contact_list_is_not_quietly_sorted():
    """确定性不等于"排过序"——本层承诺的是**次序即声明次序**。

    没有这条断言，把清单排一遍序也能让上一条测试绿，而那是另一份契约。
    """

    assembly = SceneAssembly("scene/unsorted")
    for name in ("body/z", "body/a"):
        assembly.declare_body(body(name))
    assembly.declare_contact_between("body/z", "body/a")
    scene = assembly.finalize()
    assert scene.contact_pairs[0].body_a == "body/z"
    assert b'"contact_pairs":[["body/z","body/a"]]' in scene.assembly_manifest_bytes()


def _run_example(hash_seed: str) -> str:
    environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
    environment.pop("PYTHONPATH", None)  # 脚本自己把仓库树插进sys.path
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE_SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
        cwd=str(ROOT),
    )
    return completed.stdout


def test_the_scenario_six_example_is_deterministic_across_hash_seeds():
    """跨进程、换`PYTHONHASHSEED`，装配的产出逐字节相同。

    进程内比两次抓不到集合迭代次序这类隐患（同一进程里哈希种子是同一个），
    所以这条必须起子进程——`cases/generator_determinism`对`modelgen`同款做法。
    """

    assert _run_example("0") == _run_example("12345")


# ---------------------------------------------------- 场景⑥（plans/04第九节）---


@pytest.fixture(scope="module")
def winding_line() -> FinalizedScene:
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        from winding_line_scene import build_winding_line
    finally:
        sys.path.pop(0)
    return build_winding_line()


def test_scenario_six_assembles(winding_line):
    """plans/04第九节那条链路：放线盘→两导向轮→张力轮→收线盘，装得出来。"""

    assert winding_line.scene_id == "scene/winding_line"
    assert len(winding_line.bodies) == 13
    assert len(winding_line.contact_pairs) == 10
    assert len(winding_line.allowed_pairs) == 4


def test_scenario_six_puts_the_motion_sources_and_the_actuator_where_the_page_says(
    winding_line,
):
    driven = {body.body_id for body in winding_line.bodies if body.motion_source is not None}
    assert driven == {
        "body/payoff_spool_barrel",
        "body/payoff_spool_flange_low",
        "body/payoff_spool_flange_high",
        "body/takeup_spool_barrel",
        "body/takeup_spool_flange_low",
        "body/takeup_spool_flange_high",
    }
    actuated = {body.body_id for body in winding_line.bodies if body.actuator is not None}
    assert actuated == {"body/tension_roller_face"}
    assert all(
        body.motion_source.is_replayable()
        for body in winding_line.bodies
        if body.motion_source is not None
    )


def test_scenario_six_declares_the_flange_rub(winding_line):
    """蹭边那两对必须在清单里——它是本场景的存在理由之一（plans/04第九节）。"""

    declared = {frozenset((pair.body_a, pair.body_b)) for pair in winding_line.contact_pairs}
    assert frozenset({"body/strip_span_3", "body/takeup_spool_flange_low"}) in declared
    assert frozenset({"body/strip_span_3", "body/takeup_spool_flange_high"}) in declared


def test_scenario_six_takeup_traverse_actually_moves_the_spool(winding_line):
    """横动偏移经`MotionSource`喂进来，装配层表达得了——两个时刻的z不同。"""

    def z_of(t_s: float) -> float:
        for posed in winding_line.posed_bodies_at(t_s):
            if posed.body.body_id == "body/takeup_spool_barrel":
                return posed.translation_mm[2]
        raise AssertionError("收线盘筒不在场景里")

    assert z_of(0.0) == 0.0
    assert z_of(2.0) == pytest.approx(8.0)
    assert z_of(6.0) == pytest.approx(-8.0)


def test_scenario_six_feeds_the_existing_collision_query(winding_line):
    """装配的两个产出**就是**`BroadPhaseCollisionQuery`的两个入参。

    这条门守的是"沿用既有概念不另造一套"那句话有没有兑现：
    `posed_bodies_at(t)`与`allowed_pairs`直接喂进去，构造不炸、查询能跑，
    且白名单里的对一个事件都不报。
    """

    query = BroadPhaseCollisionQuery(
        winding_line.posed_bodies_at(0.0), allowed_pairs=winding_line.allowed_pairs
    )
    events = query.check_state()
    reported = {frozenset((event.body_a, event.body_b)) for event in events}
    assert not (reported & winding_line.allowed_pairs)


def test_the_example_script_runs_and_says_what_it_did():
    stdout = _run_example("0")
    assert "finalize()通过，装配成立" in stdout
    assert "接触对 10对" in stdout
    assert "body/strip_span_3        ↔ body/takeup_spool_flange_high" in stdout
    # 诚实边界必须留在输出里：跑得出东西 ≠ 算得出接触。
    assert "本脚本不算任何接触" in stdout
