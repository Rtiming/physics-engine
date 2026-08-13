"""声明候选→动态窄相→接触响应的整合门（决策0058）。"""

from __future__ import annotations

import pytest

from physics_engine.collision import BroadPhaseCollisionQuery, CollisionQueryResult
from physics_engine.contact import (
    LinearNormalDashpot,
    PenaltySphereContact,
    build_contact_layout,
)
from physics_engine.contact_pipeline import (
    ContactPipelineError,
    DetectedSphereContactDissipation,
    DetectedSphereContactPotential,
    SphereContactPipeline,
    SphereNodeBinding,
)
from physics_engine.energies import EnergyContext
from physics_engine.scene import SceneAssembly
from physics_engine.shapes import CollisionShape, PosedBody, SimBody, Sphere
from physics_engine.state import State, StateField, StateLayout


def _scene_and_pipeline() -> tuple[SphereContactPipeline, StateLayout]:
    assembly = SceneAssembly("scene/contact_pipeline_test")
    for name in ("a", "b", "c"):
        assembly.declare_body(
            PosedBody(
                SimBody(
                    body_id=f"body/{name}",
                    collision=CollisionShape(Sphere(radius_mm=1.0), "fitted"),
                )
            )
        )
    assembly.declare_contact_between("body/a", "body/b")
    assembly.declare_contact_between("body/b", "body/c")
    scene = assembly.finalize()
    pipeline = SphereContactPipeline(
        scene=scene,
        bindings=(
            SphereNodeBinding("body/a", 0),
            SphereNodeBinding("body/b", 1),
            SphereNodeBinding("body/c", 2),
        ),
        stiffness_n_per_mm=10.0,
        damping_n_s_per_mm=0.01,
    )
    layout = StateLayout(
        layout_id="layout/contact_pipeline_test",
        fields=tuple(
            StateField(f"node{node}_{axis}_mm", 1)
            for node in range(3)
            for axis in ("x", "y", "z")
        ),
    )
    return pipeline, layout


def _state(layout: StateLayout, xs: tuple[float, float, float]) -> State:
    return State(
        layout=layout,
        vector=tuple(value for x in xs for value in (x, 0.0, 0.0)),
    )


def test_active_set_changes_but_candidate_identity_and_slot_mapping_do_not():
    """必红：活动集变化只能改值，不能改变候选次序或历史槽身份。"""

    pipeline, layout = _scene_and_pipeline()
    first = pipeline.evaluate(_state(layout, (0.0, 1.5, 8.0)))
    second = pipeline.evaluate(_state(layout, (0.0, 5.0, 6.5)))
    assert [item.pair_id for item in first.active_contacts] == [
        pipeline.candidate_ids[0]
    ]
    assert [item.pair_id for item in second.active_contacts] == [
        pipeline.candidate_ids[1]
    ]
    declarations = pipeline.contact_declarations()
    contact_layout = build_contact_layout(
        layout_id="layout/contact_pipeline_with_history",
        node_count=3,
        declarations=declarations,
    )
    assert tuple(slot.pair_id for slot in contact_layout.slots) == pipeline.candidate_ids
    assert contact_layout.slot_of(pipeline.candidate_ids[0]).base == 9
    assert contact_layout.slot_of(pipeline.candidate_ids[1]).base == 14


def test_only_narrow_phase_active_pairs_enter_potential_and_dissipation_response():
    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    context = EnergyContext(
        context_id="context/contact_pipeline_test",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    potential = DetectedSphereContactPotential(pipeline)
    dissipation = DetectedSphereContactDissipation(pipeline)
    assert potential.energy(state, context) == pytest.approx(1.25)
    gradient = potential.gradient(state, context)
    assert gradient[0] == pytest.approx(5.0)
    assert gradient[3] == pytest.approx(-5.0)
    assert gradient[6] == 0.0

    force, power = dissipation.force_and_power(
        state,
        (1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        context,
    )
    assert force[0] == pytest.approx(-0.02)
    assert force[3] == pytest.approx(0.02)
    assert force[6] == 0.0
    assert power == pytest.approx(0.04)


def test_detected_potential_preserves_the_direct_candidate_term_bytes():
    """迁移对拍：未活动候选被检测裁掉后，活动罚势的数值字节仍与旧项相同。"""

    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    context = EnergyContext(
        context_id="context/contact_pipeline_potential_parity",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    direct = PenaltySphereContact(
        pairs=((0, 1, 2.0, 10.0), (1, 2, 2.0, 10.0))
    ).quantities(state, context, need_gradient=True, need_hessian=True)
    detected = DetectedSphereContactPotential(pipeline).quantities(
        state,
        context,
        need_gradient=True,
        need_hessian=True,
    )
    assert detected == direct


def test_detected_potential_does_not_revalidate_pipeline_owned_pairs(monkeypatch):
    """必须红：装配期已验证的活动对不得在每次能量请求时重验。"""

    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    context = EnergyContext(
        context_id="context/contact_pipeline_prevalidated_potential",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    potential = DetectedSphereContactPotential(pipeline)
    pipeline.evaluate(state)

    def forbidden_validation(_term):
        raise AssertionError("validated dynamic pairs must not be revalidated")

    monkeypatch.setattr(PenaltySphereContact, "__post_init__", forbidden_validation)
    assert potential.energy(state, context) == pytest.approx(1.25)


def test_detected_dashpot_does_not_revalidate_pipeline_owned_parts(monkeypatch):
    """必须红：固定平面只验一次，活动球对沿可信pipeline快路径装配。"""

    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    context = EnergyContext(
        context_id="context/contact_pipeline_prevalidated_dashpot",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    dissipation = DetectedSphereContactDissipation(
        pipeline,
        fixed_planes=((2, (0.0, 0.0, -0.75), (0.0, 0.0, 1.0), 10.0, 0.02, 1.0),),
    )
    pipeline.evaluate(state)

    def forbidden_validation(_term):
        raise AssertionError("validated dynamic parts must not be revalidated")

    monkeypatch.setattr(LinearNormalDashpot, "__post_init__", forbidden_validation)
    force, power = dissipation.force_and_power(
        state,
        (1.0, 0.0, -0.5, -1.0, 0.0, 0.25, 0.0, 0.0, -0.25),
        context,
    )
    assert force[0] == pytest.approx(-0.02)
    assert power > 0.0


def test_equal_position_frame_is_detected_once_across_potential_and_dissipation(
    monkeypatch,
):
    """必须红：同一位置帧的势能与耗散不得把完整碰撞查询重复跑两次。"""

    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    context = EnergyContext(
        context_id="context/contact_pipeline_shared_frame",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    original = BroadPhaseCollisionQuery.check_state_with_stats
    calls = 0

    def counted(query):
        nonlocal calls
        calls += 1
        return original(query)

    monkeypatch.setattr(
        "physics_engine.contact_pipeline.BroadPhaseCollisionQuery.check_state_with_stats",
        counted,
    )
    DetectedSphereContactPotential(pipeline).energy(state, context)
    DetectedSphereContactDissipation(pipeline).force_and_power(
        state,
        (0.0,) * len(state.vector),
        context,
    )
    assert calls == 1


def test_fixed_plane_and_detected_sphere_damping_preserve_the_combined_term_bytes():
    """必须红：迁移时拆开同一耗散项会改变浮点加法树，不许只报近似相等。"""

    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    velocity = (1.0, 0.0, -0.5, -1.0, 0.0, 0.25, 0.0, 0.0, -0.25)
    context = EnergyContext(
        context_id="context/contact_pipeline_combined_dashpot",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    fixed_planes = tuple(
        (node, (0.0, 0.0, -0.75), (0.0, 0.0, 1.0), 10.0, 0.02, 1.0)
        for node in range(3)
    )
    direct = LinearNormalDashpot(
        planes=fixed_planes,
        sphere_pairs=(
            (0, 1, 2.0, 10.0, 0.01),
            (1, 2, 2.0, 10.0, 0.01),
        ),
    ).force_and_power(state, velocity, context)
    detected = DetectedSphereContactDissipation(
        pipeline,
        fixed_planes=fixed_planes,
    ).force_and_power(state, velocity, context)
    assert detected == direct


def test_collision_output_really_controls_response(monkeypatch):
    """必红：重叠几何若绕过检测直进响应，这条会失败。"""

    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    context = EnergyContext(
        context_id="context/contact_pipeline_detection_gate",
        node_masses_kg=(1.0, 1.0, 1.0),
    )

    def report_no_active_contact(_query):
        return CollisionQueryResult(
            events=(),
            candidate_pair_count=2,
            broad_phase_overlap_count=0,
            narrow_phase_check_count=0,
        )

    monkeypatch.setattr(
        "physics_engine.contact_pipeline.BroadPhaseCollisionQuery.check_state_with_stats",
        report_no_active_contact,
    )
    potential = DetectedSphereContactPotential(pipeline)
    assert potential.energy(state, context) == 0.0
    assert potential.gradient(state, context) == (0.0,) * 9


@pytest.mark.parametrize("ball_count", (10, 20, 64))
def test_candidate_and_active_counts_are_deterministic_at_declared_scales(ball_count):
    """必须红：规模证据用确定性计数，不把某台机器的墙钟当功能判据。"""

    assembly = SceneAssembly(f"scene/contact_scale_{ball_count}")
    body_ids = tuple(f"body/ball_{node:02d}" for node in range(ball_count))
    for body_id in body_ids:
        assembly.declare_body(
            PosedBody(
                SimBody(
                    body_id=body_id,
                    collision=CollisionShape(Sphere(radius_mm=1.0), "fitted"),
                )
            )
        )
    for left in range(ball_count):
        for right in range(left + 1, ball_count):
            assembly.declare_contact_between(body_ids[left], body_ids[right])
    pipeline = SphereContactPipeline(
        scene=assembly.finalize(),
        bindings=tuple(
            SphereNodeBinding(body_id=body_id, node_index=node)
            for node, body_id in enumerate(body_ids)
        ),
        stiffness_n_per_mm=10.0,
        damping_n_s_per_mm=0.01,
    )
    layout = StateLayout(
        layout_id=f"layout/contact_scale_{ball_count}",
        fields=tuple(
            StateField(f"node{node}_{axis}_mm", 1)
            for node in range(ball_count)
            for axis in ("x", "y", "z")
        ),
    )
    state = State(
        layout=layout,
        vector=tuple(
            value
            for node in range(ball_count)
            for value in (1.5 * node, 0.0, 0.0)
        ),
    )

    evaluation = pipeline.evaluate(state)
    assert evaluation.query.candidate_pair_count == ball_count * (ball_count - 1) // 2
    assert evaluation.query.broad_phase_overlap_count == ball_count - 1
    assert evaluation.query.narrow_phase_check_count == ball_count - 1
    assert len(evaluation.active_contacts) == ball_count - 1


def test_pipeline_rejects_a_broad_phase_only_event(monkeypatch):
    """必红：没有穿透深度的broad命中不得冒充接触响应。"""

    pipeline, layout = _scene_and_pipeline()
    state = _state(layout, (0.0, 1.5, 8.0))
    from physics_engine.collision import CollisionEvent

    aabb = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    def report_broad_only(_query):
        return CollisionQueryResult(
            events=(
                CollisionEvent(
                    body_a="body/a",
                    body_b="body/b",
                    confidence="broad_phase",
                    penetration_mm=None,
                    aabb_a_mm=aabb,
                    aabb_b_mm=aabb,
                ),
            ),
            candidate_pair_count=2,
            broad_phase_overlap_count=1,
            narrow_phase_check_count=0,
        )

    monkeypatch.setattr(
        "physics_engine.contact_pipeline.BroadPhaseCollisionQuery.check_state_with_stats",
        report_broad_only,
    )
    with pytest.raises(ContactPipelineError, match="narrow_phase"):
        pipeline.evaluate(state)


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        ((SphereNodeBinding("body/a", 0), SphereNodeBinding("body/b", 0), SphereNodeBinding("body/c", 2)), "node"),
        ((SphereNodeBinding("body/a", 0), SphereNodeBinding("body/b", 1)), "body/c"),
    ],
)
def test_invalid_body_to_node_bindings_fail_closed(bindings, message):
    pipeline, _ = _scene_and_pipeline()
    with pytest.raises(ContactPipelineError, match=message):
        SphereContactPipeline(
            scene=pipeline.scene,
            bindings=bindings,
            stiffness_n_per_mm=10.0,
            damping_n_s_per_mm=0.01,
        )
