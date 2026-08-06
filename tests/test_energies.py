"""能量项协议的门（spec/12第三节）。

物理判据不在这里——那些在`cases/two_body_spring`的清单里（轴7规则3）。
本文件守**协议契约**：四方法齐备、融合路径与单独调逐字节相同、
注册表次序即声明次序、失败关闭。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.energies import (
    AxialStretch,
    DiscreteElasticBending,
    EnergyContext,
    EnergyError,
    EnergyRegistry,
    UniformGravity,
)
from physics_engine.state import State, StateField, StateLayout


def _layout(nodes: int) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/n{nodes}",
        fields=tuple(
            field
            for index in range(nodes)
            for field in (
                StateField(f"node{index}_x_mm", 1),
                StateField(f"node{index}_y_mm", 1),
                StateField(f"node{index}_z_mm", 1),
            )
        ),
    )


CONTEXT = EnergyContext(
    context_id="context/t", node_masses_kg=(0.4, 0.7),
    gravity_mm_s2=(0.0, -9.80665, 0.0),
)
STATE = State(layout=_layout(2), vector=(0.0, 0.0, 0.0, 10.5, 1.0, 0.0))
STRETCH = AxialStretch(edges=((0, 1, 10.0, 2000.0),))
REGISTRY = EnergyRegistry(terms=(UniformGravity(), STRETCH))


def test_context_id_must_be_namespaced():
    with pytest.raises(EnergyError, match="context_id"):
        EnergyContext(context_id="t", node_masses_kg=(1.0,))


@pytest.mark.parametrize("mass", [0.0, -1.0, float("inf")])
def test_node_masses_must_be_positive_and_finite(mass):
    with pytest.raises(EnergyError, match="masses"):
        EnergyContext(context_id="context/t", node_masses_kg=(mass,))


def test_an_energy_term_needs_at_least_one_edge():
    with pytest.raises(EnergyError, match="at least one edge"):
        AxialStretch(edges=())


@pytest.mark.parametrize(
    ("edges", "match"),
    [
        (((0, 0, 10.0, 5.0),), "itself"),
        (((0, 1, 0.0, 5.0),), "rest length"),
        (((0, 1, 10.0, -5.0),), "axial stiffness"),
    ],
)
def test_edge_declarations_fail_closed(edges, match):
    with pytest.raises(EnergyError, match=match):
        AxialStretch(edges=edges)


def test_a_zero_length_edge_is_refused_rather_than_guessed():
    """长度为零时方向未定义、能量在此不可微——报错而不是返回一个方向。"""

    degenerate = State(layout=_layout(2), vector=(1.0, 2.0, 3.0, 1.0, 2.0, 3.0))
    with pytest.raises(EnergyError, match="zero length"):
        STRETCH.energy(degenerate, CONTEXT)


def test_the_registry_refuses_duplicate_term_names():
    with pytest.raises(EnergyError, match="duplicate"):
        EnergyRegistry(terms=(UniformGravity(), UniformGravity()))


def test_an_empty_registry_is_refused():
    with pytest.raises(EnergyError, match="at least one enabled term"):
        EnergyRegistry(terms=())


def test_registry_order_is_the_declared_order_not_a_sorted_one():
    """浮点加法不结合——次序是形制，不是"字典恰好这么排"。"""

    forward = EnergyRegistry(terms=(UniformGravity(), STRETCH))
    reverse = EnergyRegistry(terms=(STRETCH, UniformGravity()))
    assert forward.order == ("uniform_gravity", "axial_stretch")
    assert reverse.order == ("axial_stretch", "uniform_gravity")


def test_the_fused_path_energy_is_bitwise_identical_to_calling_energy_alone():
    """spec/12第3.1节的承重条款。WDS为守这条专门保留了零阶读值通道。"""

    for term in REGISTRY.terms:
        fused, _, _ = term.quantities(
            STATE, CONTEXT, need_gradient=True, need_hessian=True
        )
        assert fused == term.energy(STATE, CONTEXT), (
            f"{term.name}: 融合路径的能量值与单独调不逐位相同"
        )


def test_the_registry_total_is_bitwise_the_sum_in_declared_order():
    total, _, _ = REGISTRY.total(STATE, CONTEXT)
    expected = 0.0
    for term in REGISTRY.terms:
        expected += term.energy(STATE, CONTEXT)
    assert total == expected


def test_quantities_returns_none_for_what_was_not_asked_for():
    """按需计算是`quantities`存在的理由（75%时间在装配），不是可选优化。"""

    energy, gradient, hessian = STRETCH.quantities(
        STATE, CONTEXT, need_gradient=False, need_hessian=False
    )
    assert isinstance(energy, float) and gradient is None and hessian is None


def test_gravity_hessian_is_exactly_zero_and_gradient_is_constant():
    """平凡项也要守协议——而且正因为它平凡，它是"能量本身写错"的第一道门。"""

    gravity = UniformGravity()
    hessian = gravity.hessian(STATE, CONTEXT)
    assert all(value == 0.0 for row in hessian for value in row)
    moved = STATE.with_vector((5.0, 5.0, 5.0, 15.5, 6.0, 5.0))
    assert gravity.gradient(STATE, CONTEXT) == gravity.gradient(moved, CONTEXT)


def test_the_stretch_hessian_is_symmetric():
    hessian = STRETCH.hessian(STATE, CONTEXT)
    size = len(hessian)
    assert all(
        hessian[row][column] == pytest.approx(hessian[column][row], abs=1e-12)
        for row in range(size)
        for column in range(size)
    )


# ── 接缝的门：能量→力→加速度。**这是漏掉重力单位换算时缺席的那道门** ──


def test_free_fall_acceleration_equals_g_and_does_not_depend_on_mass():
    """自由落体：经`acceleration()`桥出来的加速度必须**恰好等于g**。

    这条门在`UniformGravity`的能量单位写错时会红，而有限差分门不会——
    因为换算因子出现在能量**之后**那一步。它一度缺席，缺席期间
    `two_body_spring`的重力能判据靠"把g除以1000传进去"通过，
    **两个错误互相抵消**。抵消掉的错误比暴露的错误危险得多。

    第二半（与质量无关）是等效原理：重力加速度不该记得物体多重。
    """

    layout = _layout(1)
    gravity = 9806.65
    accelerations = []
    for mass in (0.001, 0.37, 99.0):
        context = EnergyContext(
            context_id="context/free_fall", node_masses_kg=(mass,),
            gravity_mm_s2=(0.0, -gravity, 0.0),
        )
        registry = EnergyRegistry(terms=(UniformGravity(),))
        acceleration = registry.acceleration(context, layout)
        accelerations.append(acceleration((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0))

    for value in accelerations:
        assert value == pytest.approx((0.0, -gravity, 0.0), abs=1e-9)
    assert len(set(accelerations)) == 1, (
        f"自由落体加速度随质量变了——等效原理被破坏：{accelerations}"
    )


def test_gravity_energy_is_in_newton_millimetres_not_kilogram_millimetres_squared():
    """量纲门：重力能与拉伸能必须同量纲，否则两者相加是没有意义的数。

    取一个数值上可手算的构型：质量1kg、g=1000mm/s²、高度1mm →
    `m·g·y = 1000 kg·mm²/s² = 1 N·mm`。若漏掉除以MM_PER_M，这里会得到1000。
    """

    context = EnergyContext(
        context_id="context/dimension", node_masses_kg=(1.0,),
        gravity_mm_s2=(0.0, -1000.0, 0.0),
    )
    state = State(layout=_layout(1), vector=(0.0, 1.0, 0.0))
    assert UniformGravity().energy(state, context) == pytest.approx(1.0, rel=1e-15)


# ── LinearBending 与求解器 ──


def test_bending_energy_is_zero_on_a_straight_chain_and_positive_when_bent():
    from physics_engine.energies import LinearBending, clamped_chain_bending_stencils

    nodes = 5
    step = 10.0
    term = LinearBending(stencils=clamped_chain_bending_stencils(nodes, step, 1.0e6))
    context = EnergyContext(context_id="context/b", node_masses_kg=(1.0,) * nodes)
    straight = State(
        layout=_layout(nodes),
        vector=tuple(v for i in range(nodes) for v in (i * step, 0.0, 0.0)),
    )
    assert term.energy(straight, context) == 0.0
    bent = straight.with_vector(
        tuple(v for i in range(nodes) for v in (i * step, 0.5 * i * i, 0.0))
    )
    assert term.energy(bent, context) > 0.0


def test_bending_energy_is_invariant_under_rigid_translation():
    """刚体平移不产生弯曲能——模板系数和为零就是为了这件事，这里把它验出来。"""

    from physics_engine.energies import LinearBending, clamped_chain_bending_stencils

    nodes, step = 6, 8.0
    term = LinearBending(stencils=clamped_chain_bending_stencils(nodes, step, 3.0e5))
    context = EnergyContext(context_id="context/b", node_masses_kg=(1.0,) * nodes)
    bent = State(
        layout=_layout(nodes),
        vector=tuple(v for i in range(nodes) for v in (i * step, 0.3 * i * i, 0.1 * i)),
    )
    shifted = bent.with_vector(
        tuple(
            value + (7.0, -3.0, 11.0)[index % 3] for index, value in enumerate(bent.vector)
        )
    )
    assert term.energy(shifted, context) == pytest.approx(
        term.energy(bent, context), rel=1e-12
    )


def test_the_bending_hessian_is_constant_because_the_energy_is_quadratic():
    from physics_engine.energies import LinearBending, clamped_chain_bending_stencils

    nodes, step = 5, 10.0
    term = LinearBending(stencils=clamped_chain_bending_stencils(nodes, step, 2.0e6))
    context = EnergyContext(context_id="context/b", node_masses_kg=(1.0,) * nodes)
    first = State(
        layout=_layout(nodes),
        vector=tuple(v for i in range(nodes) for v in (i * step, 0.0, 0.0)),
    )
    second = first.with_vector(
        tuple(v for i in range(nodes) for v in (i * step, 2.0 * i * i, -0.7 * i))
    )
    assert term.hessian(first, context) == term.hessian(second, context)


def test_the_solver_fails_closed_when_a_degree_of_freedom_is_unconstrained():
    """欠约束的自由度让Hessian奇异——报错，而不是返回一个垃圾解。"""

    from physics_engine.solve import SolveError, solve_equilibrium

    layout = _layout(2)
    context = EnergyContext(context_id="context/free", node_masses_kg=(1.0, 1.0),
                            gravity_mm_s2=(0.0, -9806.65, 0.0))
    registry = EnergyRegistry(terms=(UniformGravity(),))  # 只有重力，无任何刚度
    with pytest.raises(SolveError, match="singular"):
        solve_equilibrium(
            registry, context, layout, (0.0,) * 6,
            fixed_indices=frozenset({0, 1, 2}), residual_tol_n=1.0e-9,
        )


def test_the_solver_refuses_a_fully_fixed_problem():
    from physics_engine.solve import SolveError, solve_equilibrium

    layout = _layout(1)
    context = EnergyContext(context_id="context/all_fixed", node_masses_kg=(1.0,))
    registry = EnergyRegistry(terms=(UniformGravity(),))
    with pytest.raises(SolveError, match="没有要解的东西"):
        solve_equilibrium(
            registry, context, layout, (0.0, 0.0, 0.0),
            fixed_indices=frozenset({0, 1, 2}), residual_tol_n=1.0e-9,
        )


def test_sparse_hessian_matches_the_dense_one_bitwise():
    """稀疏读法与稠密路径**逐位**相同——累加次序逐字复刻，不是"数值上接近"。

    这条门守的是spec/13第一节义务2：声称"结果不变"的优化，附逐字节对拍。
    第一版融合就在这里丢过末位（把`k*(d_a·d_b)`写成`(k*d_a)*d_b`，
    浮点乘法不结合），被抓住了。
    """

    from physics_engine.energies import LinearBending, clamped_chain_bending_stencils

    nodes, step = 7, 6.5
    layout = _layout(nodes)
    context = EnergyContext(
        context_id="context/sparse", node_masses_kg=tuple(0.2 + 0.07 * i for i in range(nodes)),
        gravity_mm_s2=(0.13, -9806.65, -0.41),
    )
    registry = EnergyRegistry(terms=(
        UniformGravity(),
        AxialStretch(edges=tuple((i, i + 1, step, 900.0 + i) for i in range(nodes - 1))),
        LinearBending(stencils=clamped_chain_bending_stencils(nodes, step, 4.4e5)),
    ))
    state = State(
        layout=layout,
        vector=tuple(
            v for i in range(nodes)
            for v in (i * step + 0.02 * i * i, 0.41 * i - 0.03 * i * i, 0.09 * i)
        ),
    )
    _, _, dense = registry.total(state, context, need_gradient=True, need_hessian=True)
    assert dense is not None
    sparse = registry.hessian_entries(state, context)
    size = len(state.vector)
    for row in range(size):
        for column in range(size):
            assert dense[row][column] == sparse.get((row, column), 0.0), (
                f"({row},{column}) 稠密{dense[row][column]!r} 稀疏{sparse.get((row, column), 0.0)!r}"
            )


# ── DiscreteElasticBending：几何精确弯曲（T5第三块） ──


def _bent_chain(nodes: int = 6, step: float = 3.0):
    """一条**真弯**且不在任何对称位置上的链——协议门要在一般构型上验。"""

    vector = []
    for index in range(nodes):
        vector.extend((
            index * step + 0.11 * index - 0.07 * index * index,
            0.43 * index - 0.09 * index * index,
            0.17 * index + 0.05 * index * index,
        ))
    return State(layout=_layout(nodes), vector=tuple(vector))


BENT = _bent_chain()
BENDING = DiscreteElasticBending(
    vertices=tuple((i - 1, i, i + 1, 1234.5, 3.0) for i in range(1, 5))
)
BENT_CONTEXT = EnergyContext(context_id="context/bent", node_masses_kg=(0.5,) * 6)


@pytest.mark.parametrize(
    ("vertices", "match"),
    [
        ((), "at least one vertex"),
        (((0, 1, 0, 1.0, 1.0),), "three distinct nodes"),
        (((0, 1, 2, 0.0, 1.0),), "bending stiffness"),
        (((0, 1, 2, 1.0, -1.0),), "voronoi"),
    ],
)
def test_bending_vertex_declarations_fail_closed(vertices, match):
    with pytest.raises(EnergyError, match=match):
        DiscreteElasticBending(vertices=vertices)


def test_a_folded_vertex_is_refused_rather_than_returning_a_huge_number():
    """θ→π是``κ = 2·tan(θ/2)``**自身**的奇点——失败关闭，不返回一个大数。

    这是`AxialStretch`零长度边那条的同类：模型在此不可微，
    返回一个有限的大数会让求解器带着一个假的能量继续往前走。
    """

    folded = State(
        layout=_layout(3),
        vector=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    term = DiscreteElasticBending(vertices=((0, 1, 2, 100.0, 1.0),))
    with pytest.raises(EnergyError, match="folded back"):
        term.energy(folded, BENT_CONTEXT)


def test_a_zero_length_edge_at_a_bending_vertex_is_refused():
    degenerate = State(
        layout=_layout(3),
        vector=(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    term = DiscreteElasticBending(vertices=((0, 1, 2, 100.0, 1.0),))
    with pytest.raises(EnergyError, match="zero-length edge"):
        term.energy(degenerate, BENT_CONTEXT)


@pytest.mark.parametrize(
    ("turn_degrees", "curvature_squared", "chord_squared"),
    [(90.0, 4.0, 2.0), (120.0, 12.0, 3.0)],
)
def test_a_known_turning_angle_stores_the_exact_tangent_half_angle_energy(
    turn_degrees, curvature_squared, chord_squared
):
    """**手算的解析判据，也是"这真是几何精确"的那条门。**

    转角θ已知时``κ = 2·tan(θ/2)``是一个手算得出的数：
    θ=90°给κ²=4、θ=120°给κ²=12（``2·tan60° = 2√3``）。
    能量``U = (EI/(2ℓ))·κ²``于是是闭式，与实现无关。

    **小挠度弯曲用的是弦长而不是转角**：``|x₀ − 2x₁ + x₂|²/h² = 4·sin²(θ/2)``，
    在θ=90°给2、θ=120°给3——分别只有几何精确值的1/2与1/4。
    两者在θ→0时同阶（都趋于θ²），差别正是本项交付的东西。
    所以这条门用一个手算数**同时**验了公式对不对、和它是不是那个小挠度近似。

    容差rel 1e-15：构型由``cos``/``sin``给出，转角本身带一次三角函数舍入。
    """

    turn = math.radians(turn_degrees)
    step, stiffness, voronoi = 2.0, 500.0, 7.0
    state = State(
        layout=_layout(3),
        vector=(
            0.0, 0.0, 0.0,
            step, 0.0, 0.0,
            step + step * math.cos(turn), step * math.sin(turn), 0.0,
        ),
    )
    term = DiscreteElasticBending(vertices=((0, 1, 2, stiffness, voronoi),))
    energy = term.energy(state, BENT_CONTEXT)
    assert energy == pytest.approx(
        0.5 * stiffness / voronoi * curvature_squared, rel=1.0e-15
    )
    assert energy != pytest.approx(
        0.5 * stiffness / voronoi * chord_squared, rel=1.0e-3
    ), "能量落在了弦长（小挠度）值上——那不是2·tan(θ/2)"


def test_the_geometrically_exact_energy_is_invariant_under_rigid_rotation():
    """能量只依赖形状，**不依赖任何参考方向**。

    `LinearBending`的固支模板带一个固定的仿射偏置`2h·t̂`，刚体转动会改变它的能量；
    本项里不该有任何这样的方向漏进来。（内部三点模板`|Δ²x|²`本身也是转动不变的，
    所以这条门**不是**几何精确与小挠度的分界——那条分界是上面那个手算角度门。）

    容差rel 1e-12：转动矩阵的元素带三角函数舍入，能量经``cos θ``进来，
    不是逐位相同的量。实测偏差9.0e-16，判据留了三个数量级余量。
    """

    angle = 0.7
    cosine, sine = math.cos(angle), math.sin(angle)
    turned = BENT.with_vector(tuple(
        value
        for index in range(len(BENT.vector) // 3)
        for value in (
            cosine * BENT.vector[3 * index] - sine * BENT.vector[3 * index + 1] + 11.0,
            sine * BENT.vector[3 * index] + cosine * BENT.vector[3 * index + 1] - 5.0,
            BENT.vector[3 * index + 2] + 2.0,
        )
    ))
    assert BENDING.energy(turned, BENT_CONTEXT) == pytest.approx(
        BENDING.energy(BENT, BENT_CONTEXT), rel=1.0e-12
    ), "几何精确弯曲能在刚体转动下变了——那它就不是几何精确的"


def test_the_independent_scalar_path_agrees_with_the_production_one():
    """spec/12第3.2节：一个能量项必须带一条**与生产路径不共享代码**的求值路径。

    生产路径走DER的分母形式``4·|e_a × e_b|² / (|e_a||e_b| + e_a·e_b)²``
    （近直链时无相消）；这里的独立路径先用``atan2``求出转角θ、再算``2·tan(θ/2)``。
    两条路线的浮点运算序列完全不同，所以判据是容差不是逐位。

    容差rel 1e-13：``atan2``与``tan``各自带一次超越函数舍入。实测偏差0.0
    （在本构型上恰好相同），余量给的是构型而不是这一个数。
    """

    independent = 0.0
    for left, middle, right, stiffness, voronoi in BENDING.vertices:
        first = [BENT.vector[3 * middle + a] - BENT.vector[3 * left + a] for a in range(3)]
        second = [BENT.vector[3 * right + a] - BENT.vector[3 * middle + a] for a in range(3)]
        cross = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        angle = math.atan2(
            math.sqrt(sum(v * v for v in cross)),
            sum(a * b for a, b in zip(first, second, strict=True)),
        )
        curvature = 2.0 * math.tan(0.5 * angle)
        independent += 0.5 * stiffness / voronoi * curvature * curvature
    assert BENDING.energy(BENT, BENT_CONTEXT) == pytest.approx(independent, rel=1.0e-13)


def test_the_bending_gradient_and_hessian_match_central_differences():
    """有限差分门。**它绿不代表物理对**（spec/12第6.1节）——

    验物理的是`cases/large_deflection_cantilever`的椭圆积分闭式。
    本仓已有两个活标本：0024的1000倍单位bug与0027的求积权，两次FD都全绿。
    这道门只回答一件事：梯度与Hessian是不是我写的那个`energy`的导数。

    步长1e-6·max(1,|x|)：中心差分的截断误差O(h²)≈1e-12、舍入误差O(eps/h)≈2e-10，
    两者在此量级平衡。判据rel 1e-6，实测梯度7.2e-10、Hessian(偏差/量级)3.3e-10。
    """

    gradient = BENDING.gradient(BENT, BENT_CONTEXT)
    hessian = BENDING.hessian(BENT, BENT_CONTEXT)
    magnitude = max(abs(value) for row in hessian for value in row)
    size = len(BENT.vector)
    for index in range(size):
        step = 1.0e-6 * max(1.0, abs(BENT.vector[index]))
        up = list(BENT.vector)
        up[index] += step
        down = list(BENT.vector)
        down[index] -= step
        state_up = BENT.with_vector(tuple(up))
        state_down = BENT.with_vector(tuple(down))
        difference = (
            BENDING.energy(state_up, BENT_CONTEXT)
            - BENDING.energy(state_down, BENT_CONTEXT)
        ) / (2.0 * step)
        assert difference == pytest.approx(gradient[index], rel=1.0e-6), (
            f"自由度{index}的解析梯度与中心差分不符"
        )
        gradient_up = BENDING.gradient(state_up, BENT_CONTEXT)
        gradient_down = BENDING.gradient(state_down, BENT_CONTEXT)
        for row in range(size):
            column = (gradient_up[row] - gradient_down[row]) / (2.0 * step)
            assert abs(column - hessian[row][index]) <= 1.0e-6 * magnitude, (
                f"Hessian({row},{index})与中心差分不符"
            )


def test_the_bending_hessian_is_symmetric_and_not_constant():
    """对称是二阶导的必然；**不是常量**是它与`LinearBending`的分界。

    `LinearBending`的Hessian恒为常量（能量是二次型，那里有一条门验它）。
    本项的Hessian随构型变——若它变成了常量，说明几何非线性被抹掉了。
    """

    hessian = BENDING.hessian(BENT, BENT_CONTEXT)
    magnitude = max(abs(value) for row in hessian for value in row)
    size = len(hessian)
    for row in range(size):
        for column in range(size):
            assert abs(hessian[row][column] - hessian[column][row]) <= 1.0e-12 * magnitude
    straighter = BENT.with_vector(tuple(
        value * (0.35 if index % 3 else 1.0) for index, value in enumerate(BENT.vector)
    ))
    assert BENDING.hessian(straighter, BENT_CONTEXT) != hessian, (
        "几何精确弯曲的Hessian成了常量——那它就退化成了二次型"
    )


def test_the_fused_bending_path_is_bitwise_identical_to_the_separate_calls():
    """spec/12第3.1节的承重条款，在本项上逐位守住——**三条一起**。

    0028里第一版融合就是在这里丢过末位（`k*(d_a·d_b)`写成`(k*d_a)*d_b`，
    浮点乘法不结合）。本项的做法是让`energy`/`gradient`/`hessian`/`quantities`
    共用同一个`_vertex_blocks`，**没有第二个求值表达式**。
    """

    energy, gradient, hessian = BENDING.quantities(
        BENT, BENT_CONTEXT, need_gradient=True, need_hessian=True
    )
    assert energy == BENDING.energy(BENT, BENT_CONTEXT)
    assert gradient == BENDING.gradient(BENT, BENT_CONTEXT)
    assert hessian == BENDING.hessian(BENT, BENT_CONTEXT)
    sparse: dict[tuple[int, int], float] = {}
    for row, column, value in BENDING.hessian_entries(BENT, BENT_CONTEXT):
        sparse[(row, column)] = sparse.get((row, column), 0.0) + value
    size = len(BENT.vector)
    assert hessian is not None
    for row in range(size):
        for column in range(size):
            assert hessian[row][column] == sparse.get((row, column), 0.0)


def test_the_bending_curvature_kernel_is_evaluated_once_per_vertex_when_fused():
    """融合是承重条款，不是性能糖：**曲率核每顶点恰好求值一次**。

    分开调`energy`/`gradient`/`hessian`会把它跑三遍（0028实测边核恰为
    1.0×/2.0×/3.0×边数）。这条是决策0018定的门形态——进门的是确定性整数。
    """

    calls = []
    original = DiscreteElasticBending._vertex_terms

    def counted(self, state, vertex):
        calls.append(vertex)
        return original(self, state, vertex)

    DiscreteElasticBending._vertex_terms = counted
    try:
        BENDING.quantities(BENT, BENT_CONTEXT, need_gradient=True, need_hessian=True)
    finally:
        DiscreteElasticBending._vertex_terms = original
    assert len(calls) == len(BENDING.vertices), (
        f"融合路径求值了{len(calls)}次曲率核，顶点只有{len(BENDING.vertices)}个——融合被做回去了"
    )


def test_the_clamp_vertex_carries_one_and_a_half_voronoi_lengths():
    """`3h/2`是推出来的，不是配出来的——这条门守住那个系数本身。

    它的物理后果（退回`h`则收敛阶从二阶掉到一阶）由
    `tests/cases/test_large_deflection_cantilever.py`守着；
    这里守的是"生成器确实按推导发系数"。
    """

    from physics_engine.energies import clamped_chain_bending_vertices

    vertices = clamped_chain_bending_vertices(6, 4.0, 250.0)
    assert vertices[0] == (0, 1, 2, 250.0, 6.0)
    assert all(vertex[4] == 4.0 for vertex in vertices[1:])
    assert len(vertices) == 4


def test_gravity_contributes_no_hessian_entries_at_all():
    """重力Hessian恒零——稀疏读法下它一个项都不该产生（稠密路径为它建了整个n²零矩阵）。"""

    context = EnergyContext(
        context_id="context/g", node_masses_kg=(1.0, 2.0),
        gravity_mm_s2=(0.0, -9806.65, 0.0),
    )
    state = State(layout=_layout(2), vector=(0.0, 0.0, 0.0, 3.0, 4.0, 5.0))
    assert UniformGravity().hessian_entries(state, context) == ()


def test_the_residual_tolerance_has_no_default_and_must_be_declared():
    """绝对残差的可达地板随规模上升——一个"看起来能用、一放大就不收敛"的默认值
    比没有默认值糟糕得多（决策0030第十节第1条）。

    实测：同一个悬臂，10段时残差地板2.4e-10 N、160段时5.6e-8 N，
    因为弯曲刚度标度`EI/h³`按`h⁻³`增长。旧默认值1e-9在40段以上够不到，
    会跑满50次迭代不收敛而调用方读不出为什么。
    """

    import inspect

    from physics_engine.solve import solve_equilibrium

    parameter = inspect.signature(solve_equilibrium).parameters["residual_tol_n"]
    assert parameter.default is inspect.Parameter.empty, (
        "residual_tol_n 不许有默认值——那会把「这个容差对我的载荷尺度合不合适」"
        "这个必须由调用方回答的问题伪装成库的实现细节"
    )


# ── 带状求解：换数值路径，走**声明容差**对拍（spec/13第一节义务2第二档） ──


def test_banded_matches_dense_within_a_declared_tolerance():
    """带状LU与稠密高斯-约当**不是同一条数值路径**，因此判据是容差不是逐字节。

    **第一版试图做成逐字节，那是错的**：给高斯-约当加带宽限制会得到错答案
    （实测收敛比从4.0掉到0.01），因为高斯-约当上下都消、会把带状结构破坏掉。
    带状只对LU+回代成立，而LU+回代与高斯-约当的运算次序本就不同。

    容差怎么算出来的：两条路径都是`O(m)`量级的累加，条件数由弯曲刚度标度
    `EI/h³`定。实测端点挠度偏差 n=40时1.6e-8mm、n=80时9.3e-8mm、
    n=160与320上**恰为0**；挠度量级3125mm，故相对偏差≤3e-11。
    判据取`rel=1e-9`，留约30倍余量。**不是放宽到能过——是把两条路径的
    数值差异算出来再留余量**（决策0024第三节的先例）。
    """

    from physics_engine import solve as solve_module
    from physics_engine.energies import LinearBending, clamped_chain_bending_stencils
    from physics_engine.solve import solve_equilibrium

    length, stiffness, load, gravity = 1000.0, 2.0e7, 0.5, 9806.65
    segments = 40
    nodes = segments + 1
    step = length / segments
    layout = StateLayout(
        layout_id="layout/banded_parity",
        fields=tuple(
            field
            for index in range(nodes)
            for field in (
                StateField(f"node{index}_x_mm", 1),
                StateField(f"node{index}_y_mm", 1),
                StateField(f"node{index}_z_mm", 1),
            )
        ),
    )
    masses = tuple(
        load * step * (0.5 if i in (0, nodes - 1) else 1.0) / gravity * 1000.0
        for i in range(nodes)
    )
    context = EnergyContext(
        context_id="context/banded_parity", node_masses_kg=masses,
        gravity_mm_s2=(0.0, -gravity, 0.0),
    )
    registry = EnergyRegistry(terms=(
        UniformGravity(),
        LinearBending(stencils=clamped_chain_bending_stencils(nodes, step, stiffness)),
    ))
    initial = tuple(v for i in range(nodes) for v in (i * step, 0.0, 0.0))
    fixed = frozenset(
        {3 * i for i in range(nodes)} | {3 * i + 2 for i in range(nodes)} | {1}
    )

    def run():
        return solve_equilibrium(
            registry, context, layout, initial,
            fixed_indices=fixed, residual_tol_n=1.0e-7,
        ).state.vector[3 * (nodes - 1) + 1]

    banded = run()
    original = solve_module._solve_banded
    solve_module._solve_banded = lambda m, r, b: original.__globals__["_solve_dense"](m, r)
    try:
        dense = run()
    finally:
        solve_module._solve_banded = original
    assert abs(banded - dense) <= 1.0e-9 * abs(dense), (
        f"带状 {banded!r} 与稠密 {dense!r} 的偏差超出声明容差 rel=1e-9"
    )


def test_the_bandwidth_of_a_chain_hessian_is_small_which_is_why_banded_pays():
    """带状之所以值得，是因为链式Hessian的半带宽**与规模无关**。

    实测：自重悬臂在40/160/320段上半带宽恒为**2**——弯曲模板只耦合相邻三个节点。
    于是`O(m³)`降到`O(m·b²)`，m=320时理论比11378×、实测端到端**47×**
    （16.9秒→0.36秒）。这条断言守的是"带宽不随规模长"，
    它一旦不成立，带状路径的收益就没了，判据也该重估。
    """

    from physics_engine.energies import LinearBending, clamped_chain_bending_stencils
    from physics_engine.solve import bandwidth_of

    bandwidths = []
    for segments in (40, 160):
        nodes = segments + 1
        step = 1000.0 / segments
        layout = StateLayout(
            layout_id=f"layout/band{segments}",
            fields=tuple(
                field
                for index in range(nodes)
                for field in (
                    StateField(f"node{index}_x_mm", 1),
                    StateField(f"node{index}_y_mm", 1),
                    StateField(f"node{index}_z_mm", 1),
                )
            ),
        )
        context = EnergyContext(
            context_id="context/band", node_masses_kg=(1.0,) * nodes,
            gravity_mm_s2=(0.0, -9806.65, 0.0),
        )
        registry = EnergyRegistry(terms=(
            UniformGravity(),
            LinearBending(stencils=clamped_chain_bending_stencils(nodes, step, 2.0e7)),
        ))
        state = State(
            layout=layout,
            vector=tuple(v for i in range(nodes) for v in (i * step, 0.0, 0.0)),
        )
        fixed = {3 * i for i in range(nodes)} | {3 * i + 2 for i in range(nodes)} | {1}
        free = [i for i in range(3 * nodes) if i not in fixed]
        position = {g: k for k, g in enumerate(free)}
        bandwidths.append(bandwidth_of(registry.hessian_entries(state, context), position))
    assert bandwidths == [2, 2], f"链式Hessian的半带宽不再是常数：{bandwidths}"


# ── PointLoad：外载项。**符号是它的全部内容**（decisions/0046） ──


def test_point_load_declarations_fail_closed():
    from physics_engine.energies import PointLoad

    with pytest.raises(EnergyError, match="at least one load"):
        PointLoad(loads=())
    with pytest.raises(EnergyError, match="nonnegative int"):
        PointLoad(loads=((-1, (1.0, 0.0, 0.0)),))
    with pytest.raises(EnergyError, match="two point loads"):
        PointLoad(loads=((0, (1.0, 0.0, 0.0)), (0, (2.0, 0.0, 0.0))))
    with pytest.raises(EnergyError, match="finite 3-vector"):
        PointLoad(loads=((0, (1.0, 0.0)),))
    with pytest.raises(EnergyError, match="finite 3-vector"):
        PointLoad(loads=((0, (float("inf"), 0.0, 0.0)),))


def test_a_point_load_on_a_node_the_state_does_not_have_fails_closed():
    """越界不是"切片给个短元组然后继续算"——那会静默算出一个别的能量。"""

    from physics_engine.energies import PointLoad

    term = PointLoad(loads=((5, (1.0, 0.0, 0.0)),))
    with pytest.raises(EnergyError, match="only has 2 nodes"):
        term.energy(STATE, CONTEXT)


def test_point_load_energy_is_the_hand_computed_negative_work():
    """**手算数**：`U = −F·x`。这是唯一能抓住"只活在`energy()`里的符号错"的门。

    decisions/0029第六节记着：求解器只用梯度与Hessian，一个只活在`energy()`里的
    错误可以从闭式案例底下整个溜过去。本条与量纲门、有限差分门三条并存，
    分别守表达式本身、单位、以及"梯度是不是这个能量的导数"。

    取 F = (2, −3, 5) N、x = (7, 11, 13) mm：
    `U = −(2·7 + (−3)·11 + 5·13) = −(14 − 33 + 65) = −46 N·mm`。
    **符号写成 `+F·x` 会得到 +46**，量纲写成除以1000会得到−0.046。
    """

    from physics_engine.energies import PointLoad

    context = EnergyContext(context_id="context/pl", node_masses_kg=(1.0,))
    state = State(layout=_layout(1), vector=(7.0, 11.0, 13.0))
    term = PointLoad(loads=((0, (2.0, -3.0, 5.0)),))
    assert term.energy(state, context) == -46.0
    assert term.gradient(state, context) == (-2.0, 3.0, -5.0)


def test_point_load_energy_is_in_newton_millimetres_with_no_unit_conversion():
    """量纲门：1 N的力把节点移到1 mm处，能量恰为−1 N·mm。

    **这里没有MM_PER_M，而`UniformGravity`那里有**——因为力已经是N、位置是mm，
    `N × mm`直接就是本仓的能量单位。照抄相邻代码除以1000，这里会得到−0.001。
    本仓已经因单位吃过两次亏（0024的静默1000倍、`two_body_spring`那次
    两个错误互相抵消），所以这条门与自由体门一起立在这里。
    """

    from physics_engine.energies import PointLoad

    context = EnergyContext(context_id="context/pl", node_masses_kg=(1.0,))
    state = State(layout=_layout(1), vector=(1.0, 0.0, 0.0))
    assert PointLoad(loads=((0, (1.0, 0.0, 0.0)),)).energy(state, context) == -1.0


def test_free_body_acceleration_under_a_point_load_is_exactly_force_over_mass():
    """自由体门：单个自由质点在`PointLoad`下的加速度恰为`F/m`。

    `a = F/m`是**米制**（N/kg = m/s²），状态是mm制，所以经`acceleration()`
    桥出来的值必须是`F/m × 1000` mm/s²。判据写成`a·m == F·1000`——
    **它对每一个质量都要成立**，那是牛顿第二定律而不是某个质量下的巧合。
    """

    from physics_engine.energies import MM_PER_M, PointLoad

    layout = _layout(1)
    force = 3.0
    products = []
    for mass in (0.002, 1.0, 57.5):
        context = EnergyContext(
            context_id="context/free_body", node_masses_kg=(mass,)
        )
        registry = EnergyRegistry(terms=(PointLoad(loads=((0, (force, 0.0, 0.0)),)),))
        acceleration = registry.acceleration(context, layout)
        value = acceleration((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)
        assert value[1] == 0.0 and value[2] == 0.0
        products.append(value[0] * mass)
    assert all(
        product == pytest.approx(force * MM_PER_M, rel=1e-15) for product in products
    ), f"a·m 不等于 F（牛顿第二定律不成立）：{products}"


def test_the_fused_point_load_path_is_bitwise_identical_to_the_separate_calls():
    """spec/12第3.1节的承重条款——`quantities`与单独调必须逐位相同。"""

    from physics_engine.energies import PointLoad

    context = EnergyContext(context_id="context/pl", node_masses_kg=(0.4, 0.7))
    state = State(layout=_layout(2), vector=(0.1, -2.7, 3.9, 10.5, 1.0, -0.25))
    term = PointLoad(loads=((0, (1.5, -2.25, 0.125)), (1, (-0.5, 7.0, 3.25))))
    fused, gradient, hessian = term.quantities(
        state, context, need_gradient=True, need_hessian=True
    )
    assert fused == term.energy(state, context)
    assert gradient == term.gradient(state, context)
    assert hessian == term.hessian(state, context)
    assert term.hessian_entries(state, context) == ()
    assert all(value == 0.0 for row in hessian for value in row)


def test_the_point_load_gradient_matches_central_differences_of_its_own_energy():
    """把`energy`与`gradient`绑在一起：符号只在其中一处翻转，这条立刻红。

    （`energy`与`gradient`**同时**翻转它抓不住——那一档由手算数门守。）
    """

    from physics_engine.energies import PointLoad

    context = EnergyContext(context_id="context/pl", node_masses_kg=(1.0,))
    base = (7.0, 11.0, 13.0)
    term = PointLoad(loads=((0, (2.0, -3.0, 5.0)),))
    analytic = term.gradient(State(layout=_layout(1), vector=base), context)
    for axis in range(3):
        step = 1.0e-6 * max(1.0, abs(base[axis]))
        shifted = []
        for sign in (1.0, -1.0):
            vector = list(base)
            vector[axis] += sign * step
            shifted.append(term.energy(State(layout=_layout(1), vector=tuple(vector)), context))
        numeric = (shifted[0] - shifted[1]) / (2.0 * step)
        assert numeric == pytest.approx(analytic[axis], rel=1e-9, abs=1e-12), (
            f"轴{axis}上梯度与能量的中心差分不符：{numeric!r} 对 {analytic[axis]!r}"
        )


def test_a_registry_of_only_point_loads_is_singular_and_fails_closed():
    """**零Hessian的陷阱**：只有线性势能时切线刚度全零，求解必须失败关闭。

    返回一个垃圾解比报错糟糕得多——`_solve_dense`与`_solve_banded`的奇异判据
    守着这一条，本门是它的正向断言（`PointLoad`是本仓第二个零Hessian项，
    第一个是`UniformGravity`，两者都不该能单独撑起一个平衡问题）。
    """

    from physics_engine.energies import PointLoad
    from physics_engine.solve import SolveError, solve_equilibrium

    for nodes in (2, 8):  # 小规模走稠密、大规模走带状，两条路径都要红
        layout = _layout(nodes)
        context = EnergyContext(context_id="context/pl", node_masses_kg=(1.0,) * nodes)
        registry = EnergyRegistry(terms=(
            PointLoad(loads=tuple((i, (1.0, 0.0, 0.0)) for i in range(nodes))),
        ))
        with pytest.raises(SolveError, match="singular"):
            solve_equilibrium(
                registry, context, layout, (0.0,) * (3 * nodes),
                residual_tol_n=1.0e-9,
            )


# ── 切线刚度的正定性：平衡态是极小还是鞍点（decisions/0046） ──


def test_the_stability_check_separates_a_minimum_from_a_saddle():
    """一根受压的三节点链：小载荷下正定，大载荷下不正定。

    这是最小的屈曲：两端钉住x与y、中间节点的y自由，弯曲刚度撑着它，
    轴向压力经`AxialStretch`的横向项`(k·ε/L)(I − d⊗d)`（**压缩时为负**）
    抵消它。判据是符号翻转，不是某个数。
    """

    from physics_engine.energies import DiscreteElasticBending, PointLoad
    from physics_engine.solve import (
        SolveError,
        solve_equilibrium,
        tangent_stiffness_is_positive_definite,
    )

    nodes, step, stiffness_ei, stiffness_ea = 3, 10.0, 100.0, 1.0e5
    layout = _layout(nodes)
    context = EnergyContext(context_id="context/buckle", node_masses_kg=(1.0,) * nodes)
    fixed_prebuckle = frozenset({0, 1, 2, 4, 5, 7, 8})
    fixed_stability = frozenset({0, 1, 2, 5, 7, 8})
    verdicts = []
    for load in (0.1, 100.0):
        registry = EnergyRegistry(terms=(
            PointLoad(loads=((nodes - 1, (-load, 0.0, 0.0)),)),
            AxialStretch(edges=tuple(
                (i, i + 1, step, stiffness_ea) for i in range(nodes - 1))),
            DiscreteElasticBending(vertices=((0, 1, 2, stiffness_ei, step),)),
        ))
        straight = tuple(v for i in range(nodes) for v in (i * step, 0.0, 0.0))
        solved = solve_equilibrium(
            registry, context, layout, straight, fixed_indices=fixed_prebuckle,
            residual_tol_n=1.0e-8, max_iterations=10,
        )
        assert solved.converged, solved.reason
        verdicts.append(tangent_stiffness_is_positive_definite(
            registry, context, solved.state, fixed_indices=fixed_stability))
    assert verdicts == [True, False], (
        f"稳定性判别没有随载荷翻转：{verdicts}——压缩时几何刚度必须变负"
    )
    with pytest.raises(SolveError, match="every degree of freedom is fixed"):
        tangent_stiffness_is_positive_definite(
            EnergyRegistry(terms=(UniformGravity(),)), context,
            State(layout=layout, vector=(0.0,) * 9),
            fixed_indices=frozenset(range(9)),
        )


# ---------------------------------------------------------------------------
# 节点块边界（决策0050的前置：状态向量不再"整条都是节点"）
# ---------------------------------------------------------------------------


def _with_anchor_slot(layout: StateLayout) -> StateLayout:
    """在节点块之后挂一个接触锚点槽（决策0050的形制：按声明的接触对分槽）。"""

    return StateLayout(
        layout_id=layout.layout_id + ".anchor",
        fields=layout.fields
        + tuple(
            StateField(f"contact0_anchor_{axis}_mm", 1, is_history=True)
            for axis in "xyz"
        ),
    )


def test_gravity_ignores_everything_past_the_node_block():
    """**本轮修的那个静默缺陷的正向门。**

    改之前`UniformGravity`按``len(state.vector) // 3``数节点，于是节点块后面
    挂一个锚点槽就多出一个"节点"。实测后果：两节点(z=1,2)的重力能29.43
    变成98.1（**3.3倍**），**而梯度长度仍然等于布局长度，求解器察觉不到**。
    """

    from physics_engine.energies import UniformGravity

    layout = _layout(2)
    context = EnergyContext(
        context_id="context/anchor-probe",
        node_masses_kg=(1.0, 1.0),
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )
    nodes_only = State(layout=layout, vector=(0.0, 0.0, 1.0, 10.0, 0.0, 2.0))
    term = UniformGravity()
    baseline = term.energy(nodes_only, context)

    with_anchor = State(
        layout=_with_anchor_slot(layout), vector=nodes_only.vector + (0.0, 0.0, 7.0)
    )
    assert term.energy(with_anchor, context) == baseline, "锚点槽改变了重力能"

    gradient = term.gradient(with_anchor, context)
    assert len(gradient) == 9, "梯度长度必须仍然等于布局长度"
    assert gradient[6:] == (0.0, 0.0, 0.0), (
        f"重力落到了锚点自由度上：{gradient[6:]}——"
        "求解器会把摩擦锚点当成有质量的点往下拉"
    )

    fused_energy, fused_gradient, _ = term.quantities(
        with_anchor, context, need_gradient=True, need_hessian=False
    )
    assert fused_energy == baseline, "融合路径没跟上（spec/12第3.1节要求两者逐字节相同）"
    assert fused_gradient == gradient


def test_the_historical_wrong_answer_is_now_red():
    """把当年那个数钉死：**98.1必须不再出现**。

    只断言"等于29.43"不够——一个把锚点算成半个节点的实现也能不等于98.1。
    两条一起断言，改坏的方向才被夹住。
    """

    from physics_engine.energies import UniformGravity

    layout = _with_anchor_slot(_layout(2))
    context = EnergyContext(
        context_id="context/historical",
        node_masses_kg=(1.0, 1.0),
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )
    state = State(layout=layout, vector=(0.0, 0.0, 1.0, 10.0, 0.0, 2.0, 0.0, 0.0, 7.0))
    energy = UniformGravity().energy(state, context)
    assert energy == pytest.approx(29.43), energy
    assert energy != pytest.approx(98.1), "锚点又被当成节点了"


def test_node_count_fails_closed_when_the_mass_table_outruns_the_vector():
    """质量表比向量还长——改之前这里报的是裸``IndexError``，不是能读的错。"""

    from physics_engine.energies import resolve_node_count

    context = EnergyContext(
        context_id="context/too-many-masses", node_masses_kg=(1.0, 1.0, 1.0)
    )
    state = State(layout=_layout(2), vector=(0.0,) * 6)
    with pytest.raises(EnergyError, match="node masses"):
        resolve_node_count(state, context)


def test_registry_rejects_a_term_that_indexes_past_the_node_block():
    """**这是本组最要紧的一条**：越界索引会落进锚点槽，而那里长度对得上。

    两个项各验一次——载荷点名越界节点、边连到越界节点。
    """

    from physics_engine.energies import PointLoad

    layout = _with_anchor_slot(_layout(2))
    context = EnergyContext(
        context_id="context/oob", node_masses_kg=(1.0, 1.0), gravity_mm_s2=(0.0, 0.0, 0.0)
    )
    state = State(layout=layout, vector=(0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 2.0, 3.0))

    load_past_the_end = EnergyRegistry(terms=(PointLoad(loads=((2, (1.0, 0.0, 0.0)),)),))
    with pytest.raises(EnergyError, match="indexes node 2"):
        load_past_the_end.total(state, context)

    edge_past_the_end = EnergyRegistry(
        terms=(AxialStretch(edges=((0, 2, 10.0, 1000.0),)),)
    )
    with pytest.raises(EnergyError, match="indexes node 2"):
        edge_past_the_end.total(state, context)
    with pytest.raises(EnergyError, match="indexes node 2"):
        edge_past_the_end.hessian_entries(state, context)


def test_terms_that_stay_inside_the_node_block_are_untouched():
    """反向：合法索引不许被这道门误伤（一道会误红的门最终会被拆掉）。"""

    from physics_engine.energies import PointLoad

    layout = _with_anchor_slot(_layout(2))
    context = EnergyContext(
        context_id="context/inbounds", node_masses_kg=(1.0, 1.0), gravity_mm_s2=(0.0, 0.0, 0.0)
    )
    state = State(layout=layout, vector=(0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 2.0, 3.0))
    registry = EnergyRegistry(
        terms=(PointLoad(loads=((1, (5.0, 0.0, 0.0)),)), UniformGravity())
    )
    energy, gradient, _ = registry.total(state, context, need_gradient=True)
    assert energy == pytest.approx(-50.0)
    assert gradient[6:] == (0.0, 0.0, 0.0), "锚点自由度上不许出现任何力"


# ---------------------------------------------------------------------------
# 能量→加速度的桥：节点块之外不是动力学自由度（决策0050）
# ---------------------------------------------------------------------------


def test_the_acceleration_bridge_survives_auxiliary_state():
    """**桥此前也踩了"整条向量都是节点"那个假定，实测抛裸`IndexError`。**

    能量项那一侧在本轮早些时候修过（`resolve_node_count`），
    但这条桥没有跟上——**同一个假定有两处实现，修了一处不等于修了那件事**。

    修法不是"给锚点也编个质量"，是**节点块之外一律返回0.0**：
    锚点是历史，由return-map推进，把它当成有质量的点去积分
    等于给摩擦锚点编造一个惯性。
    """

    from physics_engine.energies import UniformGravity

    layout = _with_anchor_slot(_layout(2))
    context = EnergyContext(
        context_id="context/bridge",
        node_masses_kg=(2.0, 3.0),
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )
    registry = EnergyRegistry(terms=(UniformGravity(),))
    acceleration = registry.acceleration(context, layout)

    vector = (0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 0.0, 0.0, 7.0)
    result = acceleration(vector, (0.0,) * 9, 0.0)

    assert len(result) == 9
    # 自由落体：两个节点都恰好是g，**与质量无关**（2kg与3kg给同一个数）
    assert result[2] == pytest.approx(-9810.0, rel=1e-15)
    assert result[5] == pytest.approx(-9810.0, rel=1e-15)
    assert result[6:] == (0.0, 0.0, 0.0), (
        f"锚点自由度上出现了加速度：{result[6:]}——积分器会把摩擦锚点当成有质量的点推走"
    )


def test_the_bridge_does_not_lean_on_the_gradient_being_zero_there():
    """**别把正确性寄托在"梯度恰好是零"上。**

    今天能量项都不碰节点块之外（`node_index_bound`那道门守着），
    所以那一段的梯度确实是零。但桥返回0.0是**独立的一条保证**——
    两条纪律各守一半，任一条被改坏时另一条还在。

    这里直接构造一个"锚点上有梯度"的情形来验桥自己那一半。
    """

    from physics_engine.energies import UniformGravity

    layout = _with_anchor_slot(_layout(1))
    context = EnergyContext(
        context_id="context/bridge-independent",
        node_masses_kg=(1.0,),
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )

    class _TouchesTheAnchor:
        """一个**违规**的项：往锚点自由度上写梯度。只在本门里存在。"""

        name = "rogue"

        def node_index_bound(self) -> int:
            return 0

        def energy(self, state, context):
            return 0.0

        def gradient(self, state, context):
            values = [0.0] * len(state.vector)
            values[-1] = 1234.0
            return tuple(values)

        def hessian(self, state, context):
            size = len(state.vector)
            return tuple(tuple(0.0 for _ in range(size)) for _ in range(size))

        def hessian_entries(self, state, context):
            return ()

        def quantities(self, state, context, *, need_gradient, need_hessian):
            return (
                0.0,
                self.gradient(state, context) if need_gradient else None,
                self.hessian(state, context) if need_hessian else None,
            )

    registry = EnergyRegistry(terms=(UniformGravity(), _TouchesTheAnchor()))
    acceleration = registry.acceleration(context, layout)
    result = acceleration((0.0,) * 6, (0.0,) * 6, 0.0)
    assert result[3:] == (0.0, 0.0, 0.0), (
        f"桥把锚点上的梯度变成了加速度：{result[3:]}——它该自己挡住，不该靠别人"
    )
