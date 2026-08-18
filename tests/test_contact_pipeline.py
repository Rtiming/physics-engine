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


# ── 相对接触刚度的那条步长上限（plans/16的M6，同行依据research/17第五节） ──


def test_the_step_bound_uses_the_same_omega0_as_the_dashpot_derivation():
    """必红：本模块的``ω0``与`contact/damping.py`的**逐字节相同**，`ζ`与速率≤1 ULP。

    两处算的是同一个物理量，只是方向相反：``linear_dashpot_parameters``从恢复系数
    派生阻尼，``contact_stiffness_step_bound``从已有的`(k, c, m_eff)`反读阻尼比。
    **两个式子各写一份就会漂**——本仓已经因为"同一个量两处各算各的"红过
    （`restitution_from_damping_ratio`与`step_response_overshoot`的Φ之争，
    见`contact/damping.py`文件头）。

    **判据分两档，因为这里确实是两件事**（spec/13第一节义务2）：

    * ``ω0 = √(1000·k/m_eff)``两边输入逐位相同、式子逐字相同 → **逐字节对拍**
      （`float.hex()`）。实测12个恢复系数上**全部相同**；
    * ``ζ``是**往返**出来的：那边由ζ算`c`、这边由`c`反算ζ，中间过了一次浮点乘除。
      往返不是逐位运算，所以判**声明容差**。实测e从0.999到0.001共12点，
      ζ与``stability_rate``的偏差**最大1 ULP**（相对1.85e-16与1.78e-16），
      判据取`rel=4e-16`（约2 ULP），留一倍余量。
      **不是放宽到能过——是把往返的舍入算出来再留余量**（0024第三节的先例）。

    欠阻尼与过阻尼两支都要判——**过阻尼那支正是最容易被省掉的那一支**
    （省掉它等于宣称"加阻尼总是更稳"，而实测最快模态`(ζ+√(ζ²−1))·ω0`随ζ单调增）。
    e=0.1及更小时ζ已越过1，上面的取值覆盖了两侧。
    """

    from physics_engine.contact import linear_dashpot_parameters
    from physics_engine.contact_pipeline import contact_stiffness_step_bound

    crossed_critical = False
    for restitution in (0.999, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001):
        peer = linear_dashpot_parameters(
            stiffness_n_per_mm=250.0, effective_mass_kg=0.4, restitution=restitution
        )
        mine = contact_stiffness_step_bound(
            stiffness_n_per_mm=peer.stiffness_n_per_mm,
            effective_mass_kg=peer.effective_mass_kg,
            damping_n_s_per_mm=peer.damping_n_s_per_mm,
        )
        crossed_critical = crossed_critical or peer.damping_ratio > 1.0
        assert mine.omega0_rad_per_s.hex() == peer.omega0_rad_per_s.hex(), (
            f"e={restitution}: ω0 不是逐字节相同 {mine.omega0_rad_per_s.hex()} "
            f"vs {peer.omega0_rad_per_s.hex()}"
        )
        assert mine.damping_ratio == pytest.approx(peer.damping_ratio, rel=4.0e-16), (
            f"e={restitution}: ζ 往返超出声明容差"
        )
        assert mine.stability_rate_per_s == pytest.approx(
            peer.stability_rate_per_s, rel=4.0e-16
        ), f"e={restitution}: stability_rate 超出声明容差"
    assert crossed_critical, "取值必须跨过临界阻尼，否则过阻尼那一支没被判到"


def test_more_damping_is_not_more_stable_past_critical():
    """必红：过阻尼那一支被省掉的话，这条会绿——所以它必须在。

    ``stability_rate = (ζ+√(ζ²−1))·ω0``在ζ大于1时单调增，于是**步长上限单调减**。
    "加阻尼总是更稳"是错的，这条把它钉成一个数。
    """

    from physics_engine.contact_pipeline import contact_stiffness_step_bound

    bounds = [
        contact_stiffness_step_bound(
            stiffness_n_per_mm=250.0,
            effective_mass_kg=0.4,
            damping_n_s_per_mm=damping,
        )
        for damping in (0.2, 2.0, 20.0, 200.0)
    ]
    assert bounds[0].damping_ratio < 1.0 < bounds[-1].damping_ratio
    for tighter, looser in zip(bounds[1:], bounds[:-1], strict=True):
        assert tighter.step_bound_s <= looser.step_bound_s, (
            f"阻尼加大反而放宽了上限：{tighter} vs {looser}"
        )
    assert bounds[-1].step_bound_s < bounds[0].step_bound_s


def test_the_two_step_bounds_are_independent_and_the_tighter_one_governs():
    """必红：只声明转动模态那一条会漏掉刚度模态——这正是research/17第五节抓到的错。

    两侧各判一次，外加相等那一支。**只报数字不报出处不行**：
    `rigidbody`看着有一条`step_bound`、其实只挡了一半物理，就是因为没有人报出处。
    """

    from physics_engine.contact_pipeline import (
        contact_stiffness_step_bound,
        governing_step_bound,
    )

    stiff = contact_stiffness_step_bound(
        stiffness_n_per_mm=250.0, effective_mass_kg=0.4, damping_n_s_per_mm=0.2
    )
    #: 转动模态上限取`h 小于 2.785/|ω|`，|ω|=10 rad/s（本仓案例的量级）——0.2785秒。
    rotational = 2.785 / 10.0
    verdict = governing_step_bound(
        contact_stiffness_bound_s=stiff.step_bound_s,
        rotational_mode_bound_s=rotational,
    )
    assert verdict.governed_by == "contact_stiffness"
    assert verdict.step_bound_s == stiff.step_bound_s
    #: 差多少：这个比值就是"只看转动那条"会超出的倍数。
    assert rotational / stiff.step_bound_s > 50.0, (
        f"两条上限之比 {rotational / stiff.step_bound_s} —— "
        "若不足以拉开，这条门就证明不了'漏掉刚度模态是真会出事的'"
    )

    #: 反向：极软的接触＋极快的自转，转动那条更紧。
    soft = contact_stiffness_step_bound(
        stiffness_n_per_mm=1.0e-4, effective_mass_kg=10.0, damping_n_s_per_mm=1.0e-6
    )
    fast_spin = governing_step_bound(
        contact_stiffness_bound_s=soft.step_bound_s,
        rotational_mode_bound_s=2.785 / 5000.0,
    )
    assert fast_spin.governed_by == "rotational_mode"
    assert fast_spin.step_bound_s == 2.785 / 5000.0

    tie = governing_step_bound(
        contact_stiffness_bound_s=0.25, rotational_mode_bound_s=0.25
    )
    assert tie.governed_by == "both"
    assert tie.step_bound_s == 0.25


def test_the_step_bound_scans_the_candidate_pool_not_the_active_set():
    """必红：按活动集算上限的实现在"此刻没接触"时会给出无穷宽的步长。

    定步长显式积分器在撞上的那一帧才发现步长太大已经晚了——那一步已经跨过去了。
    本条把三球拉开到一个接触都没有，再要一条上限：**它必须仍然是有限的、
    并且由最轻的那一对定**（约化质量最小 → ω0最大 → 上限最紧）。
    """

    from physics_engine.contact_pipeline import contact_stiffness_step_bound

    pipeline, layout = _scene_and_pipeline()
    apart = _state(layout, (0.0, 100.0, 200.0))
    assert pipeline.evaluate(apart).active_contacts == (), "本条要的是空活动集"

    #: a—b 用重的一对、b—c 用轻的一对（节点2最轻）。
    context = EnergyContext(
        context_id="context/step_bound",
        node_masses_kg=(1.0, 1.0, 0.01),
        gravity_mm_s2=(0.0, 0.0, 0.0),
    )
    bound = pipeline.stiffness_step_bound(context)
    assert 0.0 < bound.step_bound_s < float("inf")
    assert bound.governing_pair_id == pipeline.candidate_ids[1], (
        "定上限的必须是b—c那一对（约化质量最小）"
    )
    lightest = 1.0 * 0.01 / (1.0 + 0.01)
    expected = contact_stiffness_step_bound(
        stiffness_n_per_mm=pipeline.stiffness_n_per_mm,
        effective_mass_kg=lightest,
        damping_n_s_per_mm=pipeline.damping_n_s_per_mm,
    )
    assert bound.step_bound_s.hex() == expected.step_bound_s.hex()


def test_the_stability_radius_is_the_same_number_rigidbody_already_declared():
    """必红：注错第4轮抓到的洞——把2.785改成2.0，上面所有门都绿。

    那个常量当时**没有任何东西钉着**：它是"积分器的实轴稳定区半径"，
    而`rigidbody`的两条`step_bound`声明里已经各写着一个。两处各写一份就会漂，
    而漂了之后**上限只是变松，不会变红**——最坏的一类错。

    这条从`rigidbody`的声明**字符串里把数字抠出来**再对拍，
    所以它同时钉住"两条上限的分子是同一个数"这件事本身。
    """

    import re

    from physics_engine.contact_pipeline import (
        EXPLICIT_EULER_STABILITY_RADIUS,
        RK4_STABILITY_RADIUS,
    )
    from physics_engine.rigidbody import EXPLICIT_EULER_BODY, RK4_BODY

    def declared_radius(text: str) -> float:
        match = re.search(r"h\s*<\s*([0-9.]+)\s*/\s*\|ω\|_max", text)
        assert match is not None, f"这条声明里读不出稳定区半径：{text!r}"
        return float(match.group(1))

    assert declared_radius(RK4_BODY.declaration.step_bound) == RK4_STABILITY_RADIUS
    assert (
        declared_radius(EXPLICIT_EULER_BODY.declaration.step_bound)
        == EXPLICIT_EULER_STABILITY_RADIUS
    )
    #: 顺带钉住"分子属于积分器"：RK4的稳定区比显式Euler宽，这是它们的定义差别。
    assert RK4_STABILITY_RADIUS > EXPLICIT_EULER_STABILITY_RADIUS


def test_the_step_bound_declaration_names_the_other_bound_and_says_independent():
    """必红：声明被悄悄改成"我们有一条步长上限"就等于回到了research/17抓到的那个错。

    门只判**这句话里有没有那两件事**：另一条上限的出处、以及"独立/取更紧"。
    与公开面清册门同源——**只判有没有被说出来，不判说得好不好**。
    """

    from physics_engine.contact_pipeline import CONTACT_STIFFNESS_STEP_BOUND

    assert "rigidbody" in CONTACT_STIFFNESS_STEP_BOUND
    assert "转动模态" in CONTACT_STIFFNESS_STEP_BOUND
    assert "独立" in CONTACT_STIFFNESS_STEP_BOUND
    assert "更紧" in CONTACT_STIFFNESS_STEP_BOUND


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"stiffness_n_per_mm": 0.0}, "positive finite stiffness"),
        ({"stiffness_n_per_mm": float("inf")}, "positive finite stiffness"),
        ({"effective_mass_kg": -1.0}, "positive finite effective mass"),
        ({"damping_n_s_per_mm": -1.0e-9}, "finite nonnegative damping"),
        ({"damping_n_s_per_mm": float("nan")}, "finite nonnegative damping"),
        ({"stability_radius": 0.0}, "positive finite stability radius"),
    ],
)
def test_the_step_bound_fails_closed_on_unusable_inputs(override, message):
    """必红：一条算错的上限比没有上限糟得多——它会让人以为步长被挡住了。"""

    from physics_engine.contact_pipeline import contact_stiffness_step_bound

    base = {
        "stiffness_n_per_mm": 250.0,
        "effective_mass_kg": 0.4,
        "damping_n_s_per_mm": 0.2,
    }
    with pytest.raises(ContactPipelineError, match=message):
        contact_stiffness_step_bound(**{**base, **override})


def test_the_governing_bound_fails_closed_on_unusable_inputs():
    from physics_engine.contact_pipeline import governing_step_bound

    with pytest.raises(ContactPipelineError, match="contact stiffness step bound"):
        governing_step_bound(contact_stiffness_bound_s=0.0, rotational_mode_bound_s=1.0)
    with pytest.raises(ContactPipelineError, match="rotational mode step bound"):
        governing_step_bound(
            contact_stiffness_bound_s=1.0, rotational_mode_bound_s=float("inf")
        )
