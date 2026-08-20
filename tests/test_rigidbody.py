"""刚体层的门（spec/12第四节 × 决策0043）。每条校验附"必须红"。

本文件管的是**模块契约**：五项出生声明、布局指纹、惯量张量的入口校验、
四元数约定与本仓既有约定一致、加速档逐字节、观察者不改结果。
**物理判据不在这里**，在`tests/cases/test_rigid_body_free_flight.py`——
两者的分工与`test_integrate.py`对`cases/ballistic_free_flight`一样。
"""

from __future__ import annotations

import math

import pytest

from physics_engine import energies, rigidbody
from physics_engine.canonical import FTS_PROFILE, canonical_sha256
from physics_engine.geometry import GeometryError
from physics_engine.integrate import NumpyOps, PurePythonOps
from physics_engine.rigidbody import (
    EXPLICIT_EULER_BODY,
    QUATERNION_NORM_STEP_ABS_TOL,
    RIGID_BODY_INTEGRATORS,
    RIGID_BODY_LAYOUT,
    RK4_BODY,
    RigidBodyError,
    RigidBodyInertia,
    attitude_matrix,
    cross,
    integrate_free_flight,
    make_state,
    quaternion_multiply,
    rigid_body_state_derivative,
    rotate_body_to_world,
    rotate_world_to_body,
)
from physics_engine.shapes import (
    Capsule,
    CollisionShape,
    FiniteCylinder,
    MeshAsset,
    PosedBody,
    RoundedBox,
    SimBody,
    Sphere,
)
from physics_engine.state import State, StateError, StateField, StateLayout

BOX = RoundedBox(half_extents_mm=(30.0, 20.0, 10.0), fillet_radius_mm=0.0)
ASYMMETRIC = RigidBodyInertia.from_shape(BOX, mass_kg=1.0)


# ---------------------------------------------------------------------------
# 状态字段的无量纲声明（轴2规则5在`StateField`上的落地）
# ---------------------------------------------------------------------------


def test_a_field_without_a_unit_and_without_a_dimensionless_declaration_is_rejected():
    with pytest.raises(StateError, match="unit suffix"):
        StateField(name="attitude_xyzw", width=4)


def test_a_field_cannot_claim_dimensionless_while_carrying_a_unit_suffix():
    """两个方向都堵：留空装有是轴2规则5禁止的形状，带着单位说自己无量纲同样是。"""

    with pytest.raises(StateError, match="dimensionless"):
        StateField(name="position_mm", width=3, is_dimensionless=True)


def test_the_dimensionless_flag_does_not_enter_the_packing_fingerprint():
    """新标志**不进**`to_document`，所以既有布局的指纹逐字节不变（0001三前提第三条）。

    证明方式是手写一份旧口径的文档再比，不是抄一个魔法哈希串——
    魔法串只能说明"和上次一样"，说明不了"和旧口径一样"。
    """

    layout = StateLayout(
        layout_id="layout/point_mass_1d",
        fields=(StateField("position_mm", 1), StateField("velocity_mm_s", 1)),
    )
    legacy = {
        "layout_id": "layout/point_mass_1d",
        "dof_count": 2,
        "fields": [
            {"name": "position_mm", "width": 1, "is_history": False},
            {"name": "velocity_mm_s", "width": 1, "is_history": False},
        ],
    }
    assert layout.to_document() == legacy
    assert layout.fingerprint() == canonical_sha256(legacy, FTS_PROFILE)


# ---------------------------------------------------------------------------
# 布局：打包次序是形制的一部分，指纹是进函数的门
# ---------------------------------------------------------------------------


def test_the_rigid_body_layout_is_pinned():
    """13个自由度、四块、这个次序。改它是破坏性变更，指纹在这里被钉住。"""

    assert RIGID_BODY_LAYOUT.dof_count == 13
    assert [field.name for field in RIGID_BODY_LAYOUT.fields] == [
        "centre_of_mass_position_mm",
        "centre_of_mass_velocity_mm_per_s",
        "angular_velocity_body_rad_per_s",
        "attitude_body_to_world_xyzw",
    ]
    assert RIGID_BODY_LAYOUT.fingerprint() == (
        "f05b733fc2140e455b58debbfad82f7794e1bcf17362d55ab204b86447ae92ad"
    )
    assert RIGID_BODY_LAYOUT.history_fields() == ()


def test_must_be_red_a_layout_with_the_blocks_reordered_is_a_different_contract():
    """次序换了`dof_count`一样、向量长度一样——**只看长度的检查发现不了**，指纹发现得了。"""

    fields = RIGID_BODY_LAYOUT.fields
    swapped = StateLayout(
        layout_id=RIGID_BODY_LAYOUT.layout_id,
        fields=(fields[1], fields[0], fields[2], fields[3]),
    )
    assert swapped.dof_count == RIGID_BODY_LAYOUT.dof_count
    assert swapped.fingerprint() != RIGID_BODY_LAYOUT.fingerprint()
    state = State(layout=swapped, vector=(0.0,) * 12 + (1.0,))
    with pytest.raises(RigidBodyError, match="打包次序"):
        integrate_free_flight(
            RK4_BODY, state=state, inertia=ASYMMETRIC, dt_s=1.0e-3, steps=1
        )


def test_an_equivalent_layout_object_still_passes_the_fingerprint_fallback():
    """同一对象走热路径，但从严格reader重建的等价布局仍按指纹放行。"""

    clone = StateLayout(
        layout_id=RIGID_BODY_LAYOUT.layout_id,
        fields=RIGID_BODY_LAYOUT.fields,
        node_dof_count=RIGID_BODY_LAYOUT.node_dof_count,
    )
    assert clone is not RIGID_BODY_LAYOUT
    state = State(clone, make_state(angular_velocity_rad_per_s=(1.0, 2.0, 3.0)).vector)
    assert rigidbody.angular_velocity_body_rad_per_s(state) == (1.0, 2.0, 3.0)


def test_the_integrator_reads_block_offsets_from_the_layout_not_from_literals():
    """偏移量必须**由布局导出**。写死`y[9:13]`这类字面量时，布局一改它不跟着改，
    而`fingerprint()`会照常变、门会照常绿——因为门比的是布局，不是积分器里那个数。

    本条两面都验：导出的区间确实等于布局算出来的，且模块里**没有任何带整数字面量
    的切片**。用AST扫而不是正则扫——第一版写成正则，被自己的文档字符串里那句
    "曾经写过`y[9:13]`"打红了。判据本身也要被验，这就是它被验的方式。
    """

    import ast
    from pathlib import Path

    assert rigidbody._ATTITUDE_SLICE == slice(
        RIGID_BODY_LAYOUT.offset_of("attitude_body_to_world_xyzw"),
        RIGID_BODY_LAYOUT.dof_count,
    )
    assert rigidbody._OMEGA_SLICE == slice(6, 9)
    assert rigidbody._ASSEMBLY == (0, 1, 2, 3)
    tree = ast.parse(Path(rigidbody.__file__).read_text(encoding="utf-8"))
    naked = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Slice)
        and any(
            isinstance(bound, ast.Constant) and isinstance(bound.value, int)
            for bound in (node.lower, node.upper)
        )
    ]
    assert not naked, f"刚体模块里出现了写死的向量切片：{naked}——偏移量要走`_slice_of`"


def test_make_state_rejects_a_non_unit_quaternion():
    with pytest.raises(RigidBodyError, match="unit quaternion"):
        make_state(attitude_xyzw=(0.0, 0.0, 0.0, 1.1))


@pytest.mark.parametrize(
    "kwargs", [{"position_mm": (0.0, 0.0)}, {"attitude_xyzw": (0.0, 0.0, 1.0)}]
)
def test_make_state_rejects_blocks_of_the_wrong_width(kwargs):
    with pytest.raises(RigidBodyError, match="components"):
        make_state(**kwargs)


# ---------------------------------------------------------------------------
# 五项出生声明（spec/12第4.2节）
# ---------------------------------------------------------------------------


def test_every_rigid_body_integrator_carries_all_five_declarations():
    for name, integrator in RIGID_BODY_INTEGRATORS.items():
        declaration = integrator.declaration
        assert declaration.name == name
        for field_name in (
            "scope_excludes", "measured_order", "step_bound",
            "dissipation_accounting", "failure_ladder",
        ):
            assert str(getattr(declaration, field_name)).strip(), (
                f"{name}缺声明{field_name}——五项缺一不得进仓"
            )


def test_no_rigid_body_integrator_claims_to_be_production_ready():
    """两个都是`False`，且有这条门守着（形制照0019第五节）。

    理由不是谦虚：RK4**不是辛的**，长时间守恒量单调漂移；显式Euler反耗散。
    接触/刚性问题上这两个都不该用，而接触正是场景③的另一半。
    """

    assert not any(
        integrator.declaration.production_ready
        for integrator in RIGID_BODY_INTEGRATORS.values()
    )


def test_the_declared_orders_and_stability_classes_are_what_the_case_measured():
    assert RK4_BODY.declaration.formal_order == 4
    assert EXPLICIT_EULER_BODY.declaration.formal_order == 1
    assert RK4_BODY.declaration.stability == "explicit_conditional"
    assert "不是辛的" in RK4_BODY.declaration.scope_excludes, (
        "RK4非辛这件事必须留在适用域声明里——它是长时间守恒结论的边界"
    )
    assert "QUATERNION_NORM_STEP_ABS_TOL" in RK4_BODY.declaration.failure_ladder


# ---------------------------------------------------------------------------
# 四元数：与本仓既有约定是同一套
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quaternion",
    [
        (0.0, 0.0, 0.0, 1.0),
        (0.5, 0.5, 0.5, 0.5),
        (0.18257418583505536, 0.3651483716701107, 0.5477225575051661, 0.7302967433402214),
    ],
)
def test_the_attitude_matrix_is_bit_for_bit_the_one_shapes_already_uses(quaternion):
    """引擎里只许有一套四元数约定。第二套一出现就是静默的坐标系错，
    而`peer_fcl_distance`已经把"四元数次序错"实测成一条必红（1.508e+2 mm偏差）。"""

    body = PosedBody(
        body=SimBody(
            body_id="body/probe",
            collision=CollisionShape(shape=Sphere(radius_mm=1.0), direction="fitted"),
        ),
        rotation_xyzw=quaternion,
    )
    for vector in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.3, -0.7, 2.1)):
        assert rotate_body_to_world(quaternion, vector) == body.rotate_local_mm(vector)
    assert attitude_matrix(quaternion)[0][0] == pytest.approx(
        1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2), rel=0.0, abs=0.0
    )


def test_rotating_out_and_back_returns_the_same_vector():
    quaternion = (0.5, 0.5, 0.5, 0.5)
    vector = (0.3, -0.7, 2.1)
    round_trip = rotate_world_to_body(quaternion, rotate_body_to_world(quaternion, vector))
    for original, returned in zip(vector, round_trip, strict=True):
        assert returned == pytest.approx(original, rel=1.0e-15, abs=1.0e-15)


def test_quaternion_multiplication_does_not_commute():
    """不可交换正是`q̇ = ½q⊗ω`与`½ω⊗q`是两个物理的原因。"""

    left, right = (0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.0)
    assert quaternion_multiply(left, right) != quaternion_multiply(right, left)
    identity = (0.0, 0.0, 0.0, 1.0)
    assert quaternion_multiply(left, identity) == left


def test_cross_product_order_flips_the_sign():
    left, right = (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)
    forward, backward = cross(left, right), cross(right, left)
    assert forward == tuple(-value for value in backward)


# ---------------------------------------------------------------------------
# 惯量张量：直接取geometry，入口三条校验
# ---------------------------------------------------------------------------


def test_inertia_comes_straight_from_geometry():
    """本层不重推惯量。同一个形状、同一个质量，两边逐位相同。"""

    from physics_engine.geometry import mass_properties

    properties = mass_properties(BOX, mass_kg=1.0)
    assert ASYMMETRIC.inertia_body_kg_mm2 == properties.inertia_about_centroid_kg_mm2
    assert ASYMMETRIC.mass_kg == properties.mass_kg


def test_from_shape_inherits_geometrys_fail_closed_cases():
    """带法兰的圆柱与网格资产在`geometry`那一层就失败关闭，本层不开后门。"""

    with pytest.raises(GeometryError, match="flange"):
        RigidBodyInertia.from_shape(
            FiniteCylinder(radius_mm=10.0, half_width_mm=5.0, flange_outer_radius_mm=14.0),
            mass_kg=1.0,
        )
    with pytest.raises(GeometryError, match="mass model"):
        RigidBodyInertia.from_shape(
            MeshAsset(
                path_relative="assets/probe.stl", sha256="0" * 64, units="mm",
                usage="collision", convexity="exact_convex",
                aabb_min_mm=(-1.0, -1.0, -1.0), aabb_max_mm=(1.0, 1.0, 1.0),
            ),
            mass_kg=1.0,
        )


@pytest.mark.parametrize(
    ("tensor", "pattern"),
    [
        (((1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "not symmetric"),
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), "positive"),
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 10.0)), "triangle inequality"),
    ],
)
def test_must_be_red_a_tensor_no_mass_distribution_can_produce_is_rejected(tensor, pattern):
    """三角不等式那一条最有用：`diag(1, 1, 10)`随手写出来很自然，
    但**没有任何质量分布**给得出它——`I₁ + I₂ ≥ I₃`是形状约束不是建议。"""

    with pytest.raises(RigidBodyError, match=pattern):
        RigidBodyInertia(mass_kg=1.0, inertia_body_kg_mm2=tensor)


def test_principal_moments_survive_a_rotation_of_the_tensor():
    """主惯量是**不变量**。把对角张量转到一般轴上，三个特征值必须原样回来——
    这条同时验闭式特征值解在非对角输入上不退化。"""

    quaternion = (0.5, 0.5, 0.5, 0.5)
    rows = attitude_matrix(quaternion)
    diagonal = ASYMMETRIC.inertia_body_kg_mm2
    rotated = tuple(
        tuple(
            sum(rows[i][k] * diagonal[k][k] * rows[j][k] for k in range(3))
            for j in range(3)
        )
        for i in range(3)
    )
    moments = RigidBodyInertia(
        mass_kg=1.0, inertia_body_kg_mm2=rotated
    ).principal_moments_kg_mm2()
    expected = sorted((diagonal[0][0], diagonal[1][1], diagonal[2][2]), reverse=True)
    for measured, want in zip(moments, expected, strict=True):
        assert measured == pytest.approx(want, rel=1.0e-12)


def test_solving_with_the_inertia_tensor_inverts_applying_it():
    omega = (0.3, -1.2, 2.5)
    round_trip = ASYMMETRIC.solve(ASYMMETRIC.apply(omega))
    for measured, want in zip(round_trip, omega, strict=True):
        assert measured == pytest.approx(want, rel=1.0e-13)


def test_a_capsule_degenerating_to_a_sphere_gives_an_isotropic_tensor():
    """退化即球——`geometry`那一层的判据在本层照样成立，因为本层没有第二条路。"""

    capsule = Capsule(
        point_a_mm=(0.0, 0.0, 0.0), point_b_mm=(0.0, 0.0, 0.0), radius_mm=7.0
    )
    inertia = RigidBodyInertia.from_shape(capsule, mass_kg=2.0)
    moments = inertia.principal_moments_kg_mm2()
    assert moments[0] == pytest.approx(moments[2], rel=1.0e-14)


# ---------------------------------------------------------------------------
# 单位边界：只有一个1000
# ---------------------------------------------------------------------------


def test_the_unit_conversion_constant_is_imported_not_re_typed():
    """`N·mm ↔ kg·mm²/s²`的1000从`energies`导入。同一个换算有两份字面量，
    迟早只改一份——本仓的重力项已经栽过一次（`energies`第102行的注释）。"""

    assert rigidbody.MM_PER_M is energies.MM_PER_M
    source = (
        __import__("pathlib").Path(rigidbody.__file__).read_text(encoding="utf-8")
    )
    assert "1000.0" not in source, "刚体模块里出现了裸的1000.0——单位换算必须走命名常量"


def test_the_public_single_body_derivative_is_the_euler_equation_formula_source():
    """P3-M3的26维积分只许复用这一份单体公式，不许抄第二份。"""

    inertia = RigidBodyInertia(
        mass_kg=2.0,
        inertia_body_kg_mm2=(
            (2.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 4.0),
        ),
    )
    state = make_state(
        position_mm=(7.0, 8.0, 9.0),
        velocity_mm_per_s=(1.0, -2.0, 3.0),
        angular_velocity_rad_per_s=(0.5, -1.0, 2.0),
    )
    force = (1.0, -2.0, 3.0)
    torque = (4.0, 5.0, -6.0)
    derivative = rigid_body_state_derivative(
        state.vector,
        inertia=inertia,
        force_world_n=force,
        torque_body_nmm=torque,
    )

    angular_momentum = (1.0, -3.0, 8.0)
    gyroscopic = cross((0.5, -1.0, 2.0), angular_momentum)
    expected_angular = (
        (torque[0] * energies.MM_PER_M - gyroscopic[0]) / 2.0,
        (torque[1] * energies.MM_PER_M - gyroscopic[1]) / 3.0,
        (torque[2] * energies.MM_PER_M - gyroscopic[2]) / 4.0,
    )
    assert derivative == (
        1.0,
        -2.0,
        3.0,
        500.0,
        -1000.0,
        1500.0,
        *expected_angular,
        0.25,
        -0.5,
        1.0,
        0.0,
    )


def test_must_be_red_the_public_derivative_rejects_a_non_rigid_body_vector():
    with pytest.raises(RigidBodyError, match="13"):
        rigid_body_state_derivative(
            (0.0,) * 12,
            inertia=ASYMMETRIC,
            force_world_n=(0.0, 0.0, 0.0),
            torque_body_nmm=(0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# 推进：加速档逐字节、观察者不改结果、输入校验
# ---------------------------------------------------------------------------


def _fly(ops=None, observer=None, steps=200):
    return integrate_free_flight(
        RK4_BODY,
        state=make_state(
            velocity_mm_per_s=(10.0, -3.0, 0.5), angular_velocity_rad_per_s=(1.0, 2.0, 3.0)
        ),
        inertia=ASYMMETRIC,
        dt_s=2.0e-3,
        steps=steps,
        force_world_n=lambda _y, _t: (0.0, -9.80665, 0.0),
        ops=ops,
        observer=observer,
    )


def test_pure_python_and_accel_backends_are_bitwise_identical():
    """0016甲案的进仓门。逐字节而不是容差——两个后端跑的是**同一串运算、同一个次序**
    （一份公式源按`VectorOps`求值），逐位相同是构造保证的（决策0019第2.1节的论证）。"""

    numpy = pytest.importorskip(
        "numpy", reason="加速档未安装（pip install -e '.[accel]'）——核心零依赖，这是可选档"
    )
    assert numpy is not None
    pure, _ = _fly(ops=PurePythonOps())
    accel, _ = _fly(ops=NumpyOps())
    assert pure.vector == accel.vector


def test_the_observer_cannot_change_the_result():
    """观察者是诊断口不是积分的一部分。它一旦影响结果，案例测的就不是内核了。"""

    seen: list[int] = []
    plain, plain_report = _fly()
    watched, watched_report = _fly(observer=lambda index, t, state: seen.append(index))
    assert plain.vector == watched.vector
    assert plain_report == watched_report
    assert seen == list(range(200))


@pytest.mark.parametrize(("dt_s", "steps"), [(0.0, 1), (-1.0e-3, 1), (1.0e-3, -1)])
def test_step_size_and_count_are_validated(dt_s, steps):
    with pytest.raises(RigidBodyError):
        integrate_free_flight(
            RK4_BODY, state=make_state(), inertia=ASYMMETRIC, dt_s=dt_s, steps=steps
        )


def test_zero_steps_returns_the_state_untouched():
    state = make_state(angular_velocity_rad_per_s=(1.0, 2.0, 3.0))
    final, report = integrate_free_flight(
        RK4_BODY, state=state, inertia=ASYMMETRIC, dt_s=1.0e-3, steps=0
    )
    assert final.vector == state.vector
    assert report.steps == 0 and report.renormalisations == 0


def test_a_body_at_rest_stays_at_rest():
    """无力、无力矩、零初速——十三个自由度一个都不许动。
    这条挡的是"某一块被误当成另一块"的串块错，比任何守恒量判据都直接。"""

    state = make_state(position_mm=(1.0, 2.0, 3.0))
    final, _ = integrate_free_flight(
        RK4_BODY, state=state, inertia=ASYMMETRIC, dt_s=1.0e-3, steps=50
    )
    assert final.vector == state.vector


def test_must_be_red_the_quaternion_norm_guard_fires_before_renormalisation():
    """护栏守在归一化**之前**那一侧。归一化之后`|q|`恒为1，
    在那一侧写断言是一条永远通过的断言（spec/12第6.2节点名的假通过）。"""

    with pytest.raises(RigidBodyError, match="before renormalisation"):
        integrate_free_flight(
            EXPLICIT_EULER_BODY,
            state=make_state(angular_velocity_rad_per_s=(1.0, 2.0, 3.0)),
            inertia=ASYMMETRIC,
            dt_s=2.0e-3,
            steps=1,
        )
    # 同一个积分器在足够小的步长上通过——护栏挡的是步长，不是积分器本身。
    _final, report = integrate_free_flight(
        EXPLICIT_EULER_BODY,
        state=make_state(angular_velocity_rad_per_s=(1.0, 2.0, 3.0)),
        inertia=ASYMMETRIC,
        dt_s=1.0e-4,
        steps=10,
    )
    assert 0.0 < report.max_norm_deviation <= QUATERNION_NORM_STEP_ABS_TOL


def test_a_zero_quaternion_is_not_a_rotation():
    with pytest.raises(RigidBodyError, match="collapsed to zero"):
        rigidbody.normalise_quaternion((0.0, 0.0, 0.0, 0.0))


def test_a_singular_inertia_tensor_fails_closed():
    """奇异张量没有`I⁻¹`。它过不了三角不等式那一关，所以这里直接构造对象来验`solve`。"""

    inertia = RigidBodyInertia.__new__(RigidBodyInertia)
    object.__setattr__(inertia, "mass_kg", 1.0)
    object.__setattr__(
        inertia,
        "inertia_body_kg_mm2",
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
    )
    with pytest.raises(RigidBodyError, match="singular"):
        inertia.solve((1.0, 1.0, 1.0))


def test_a_sphere_spins_about_any_axis_without_precessing():
    """各向同性体的`ω × (I·ω) = 0`，所以`ω`常数、姿态匀速绕固定轴转。
    这是Euler方程的一个**可独立核对的退化**：球没有进动。"""

    inertia = RigidBodyInertia.from_shape(Sphere(radius_mm=12.0), mass_kg=1.5)
    omega0 = (1.0, 2.0, 3.0)
    final, _ = integrate_free_flight(
        RK4_BODY,
        state=make_state(angular_velocity_rad_per_s=omega0),
        inertia=inertia,
        dt_s=1.0e-3,
        steps=1000,
    )
    for measured, want in zip(final.block("angular_velocity_body_rad_per_s"), omega0, strict=True):
        assert measured == pytest.approx(want, rel=1.0e-13)
    # 姿态转过的角度 = |ω|·T，可闭式核对。
    quaternion = final.block("attitude_body_to_world_xyzw")
    angle = 2.0 * math.acos(min(1.0, abs(quaternion[3])))
    speed = math.sqrt(sum(value * value for value in omega0))
    turns = speed * 1.0 / (2.0 * math.pi)
    residual = turns - math.floor(turns)
    expected = 2.0 * math.pi * residual
    expected = min(expected, 2.0 * math.pi - expected)
    assert angle == pytest.approx(expected, rel=1.0e-10)
