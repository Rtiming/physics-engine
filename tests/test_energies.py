"""能量项协议的门（spec/12第三节）。

物理判据不在这里——那些在`cases/two_body_spring`的清单里（轴7规则3）。
本文件守**协议契约**：四方法齐备、融合路径与单独调逐字节相同、
注册表次序即声明次序、失败关闭。
"""

from __future__ import annotations

import pytest

from physics_engine.energies import (
    AxialStretch,
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
