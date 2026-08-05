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
            registry, context, layout, (0.0,) * 6, fixed_indices=frozenset({0, 1, 2})
        )


def test_the_solver_refuses_a_fully_fixed_problem():
    from physics_engine.solve import SolveError, solve_equilibrium

    layout = _layout(1)
    context = EnergyContext(context_id="context/all_fixed", node_masses_kg=(1.0,))
    registry = EnergyRegistry(terms=(UniformGravity(),))
    with pytest.raises(SolveError, match="没有要解的东西"):
        solve_equilibrium(
            registry, context, layout, (0.0, 0.0, 0.0),
            fixed_indices=frozenset({0, 1, 2}),
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
