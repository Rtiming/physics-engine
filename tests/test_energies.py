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
