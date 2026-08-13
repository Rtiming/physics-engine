"""线性法向dashpot、全阻尼恢复系数分支与耗散力的门（阶段2）。"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import (
    ContactError,
    LinearNormalDashpot,
    damping_ratio_from_restitution,
    linear_dashpot_parameters,
    restitution_from_damping_ratio,
)
from physics_engine.energies import DISSIPATION, EnergyContext
from physics_engine.state import State, StateField, StateLayout


def _layout(nodes: int) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/damping_n{nodes}",
        fields=tuple(
            field
            for node in range(nodes)
            for field in (
                StateField(f"node{node}_x_mm", 1),
                StateField(f"node{node}_y_mm", 1),
                StateField(f"node{node}_z_mm", 1),
            )
        ),
    )


def test_force_zero_restitution_covers_under_critical_and_over_damping():
    """0057裁决后的整条定义域：ζ=1不是下界，只是分支连接点。"""

    assert restitution_from_damping_ratio(0.0) == 1.0
    assert restitution_from_damping_ratio(1.0) == pytest.approx(math.exp(-2.0), rel=1e-15)
    assert restitution_from_damping_ratio(1.9453461844077076) == pytest.approx(
        0.05, rel=2e-13
    )
    assert restitution_from_damping_ratio(5.0) < 0.01


@pytest.mark.parametrize("target", [0.99, 0.8, math.exp(-2.0), 0.1, 0.05, 0.01])
def test_restitution_inverse_round_trips_across_the_critical_branch(target):
    ratio = damping_ratio_from_restitution(target)
    assert restitution_from_damping_ratio(ratio) == pytest.approx(target, rel=2e-13)


@pytest.mark.parametrize("target", [0.0, -0.1, 1.0001, float("nan"), float("inf")])
def test_restitution_target_fails_closed_when_no_finite_dashpot_can_represent_it(target):
    """必须红：e=0只在ζ→∞极限到达，不能伪装成有限参数。"""

    with pytest.raises(ContactError, match="restitution"):
        damping_ratio_from_restitution(target)


def test_dashpot_parameters_keep_the_engine_mm_unit_conversion_explicit():
    target = restitution_from_damping_ratio(0.2)
    parameters = linear_dashpot_parameters(
        stiffness_n_per_mm=100.0,
        effective_mass_kg=0.1,
        restitution=target,
    )

    omega0 = math.sqrt(1000.0 * 100.0 / 0.1)
    expected_c = 2.0 * 0.2 * 0.1 * omega0 / 1000.0
    assert parameters.damping_ratio == pytest.approx(0.2, rel=2e-13)
    assert parameters.omega0_rad_per_s == pytest.approx(omega0, rel=1e-15)
    assert parameters.damping_n_s_per_mm == pytest.approx(expected_c, rel=2e-13)


def test_overdamped_stability_uses_the_fast_root_not_omega0():
    """必须红：ζ=5若仍把ω0交给顾问，会把稳定界放宽约9.9倍。"""

    target = restitution_from_damping_ratio(5.0)
    parameters = linear_dashpot_parameters(
        stiffness_n_per_mm=10.0, effective_mass_kg=1.0, restitution=target
    )
    factor = 5.0 + math.sqrt(24.0)
    assert parameters.stability_rate_per_s == pytest.approx(
        factor * parameters.omega0_rad_per_s, rel=2e-13
    )
    assert parameters.stability_rate_per_s > 9.8 * parameters.omega0_rad_per_s


def test_plane_dashpot_truncates_at_zero_total_force_and_reports_actual_power():
    """出射过快时dashpot最多抵消弹簧，绝不把接触变成拉力。"""

    layout = _layout(1)
    state = State(layout=layout, vector=(0.0, 0.0, -0.1))
    context = EnergyContext(context_id="context/dashpot", node_masses_kg=(1.0,))
    dashpot = LinearNormalDashpot(
        planes=((
            0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            1000.0, 2.0, 0.0,
        ),)
    )

    force, power = dashpot.force_and_power(state, (0.0, 0.0, 100.0), context)
    spring_force = 1000.0 * 0.1
    assert dashpot.kind == DISSIPATION
    assert force == (0.0, 0.0, -spring_force)
    assert spring_force + force[2] == 0.0
    assert power == spring_force * 100.0


def test_sphere_dashpot_is_equal_and_opposite_and_its_power_is_nonnegative():
    layout = _layout(2)
    state = State(layout=layout, vector=(0.0, 0.0, 0.0, 1.8, 0.0, 0.0))
    context = EnergyContext(context_id="context/sphere_dashpot", node_masses_kg=(1.0, 1.0))
    dashpot = LinearNormalDashpot(
        sphere_pairs=((0, 1, 2.0, 1000.0, 3.0),)
    )

    force, power = dashpot.force_and_power(
        state, (1.0, 0.0, 0.0, -1.0, 0.0, 0.0), context
    )
    assert force[:3] == pytest.approx((-6.0, 0.0, 0.0), rel=1e-15)
    assert force[3:] == pytest.approx((6.0, 0.0, 0.0), rel=1e-15)
    assert tuple(force[i] + force[i + 3] for i in range(3)) == (0.0, 0.0, 0.0)
    assert power == pytest.approx(12.0, rel=1e-15)


def test_dashpot_declarations_fail_closed():
    with pytest.raises(ContactError, match="at least one"):
        LinearNormalDashpot()
    with pytest.raises(ContactError, match="damping"):
        LinearNormalDashpot(
            planes=((0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.0, 0.0),)
        )
