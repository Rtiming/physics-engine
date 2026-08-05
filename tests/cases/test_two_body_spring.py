"""`case/two_body_spring`的conformance门（轴7规则3）。

**本案例是spec/12第6.1节"有限差分验不了物理"的活标本。** 写这一块时，
能量→加速度的接缝漏了mm与m的换算（加速度小了1000倍），而有限差分门
（梯度对能量、Hessian对梯度）**全绿**——因为换算因子不在能量里，FD看不见它。
抓住它的是这里的解析角频率门。所以这两类门必须并存，任何一类都不能替代另一类。
"""

from __future__ import annotations

import math
from pathlib import Path

from physics_engine.energies import (
    MM_PER_M,
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    UniformGravity,
)
from physics_engine.integrate import VELOCITY_VERLET, integrate
from physics_engine.oracles import load_manifest
from physics_engine.state import State, StateField, StateLayout

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/two_body_spring/oracle.json", root=ROOT)
BY_ID = {entry.id: entry for entry in MANIFEST.oracles}


def _layout(nodes: int) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/two_body_spring_n{nodes}",
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


def test_stretch_energy_matches_the_closed_form():
    """验的是**能量本身**，不是它的导数。"""

    entry = BY_ID["oracle:two_body_spring/stretch_energy"]
    rest_mm = entry.inputs["rest_length_mm"]
    elongation = entry.inputs["elongation_mm"]
    term = AxialStretch(edges=((0, 1, rest_mm, entry.inputs["axial_stiffness_n"]),))
    state = State(
        layout=_layout(2),
        vector=(0.0, 0.0, 0.0, rest_mm + elongation, 0.0, 0.0),
    )
    context = EnergyContext(context_id="context/spring", node_masses_kg=(1.0, 1.0))
    entry.check_all({"energy_nmm": term.energy(state, context)})


def test_gravity_energy_matches_the_closed_form():
    """注意：这条一度靠**把g除以1000传进去**才通过——两个错误互相抵消。
    `UniformGravity`的能量单位修正后抵消消失，它当场红了，这才是它该有的样子。"""

    entry = BY_ID["oracle:two_body_spring/gravity_energy"]
    height = entry.inputs["height_mm"]
    context = EnergyContext(
        context_id="context/gravity",
        node_masses_kg=(entry.inputs["mass_kg"],),
        gravity_mm_s2=(0.0, entry.inputs["gravity_mm_s2"], 0.0),
    )
    state = State(layout=_layout(1), vector=(0.0, height, 0.0))
    entry.check_all({"energy_nmm": UniformGravity().energy(state, context)})


def test_two_body_oscillation_reproduces_the_analytic_frequency():
    """**这条门是那个1000倍单位bug的捕手。**"""

    entry = BY_ID["oracle:two_body_spring/analytic_angular_frequency"]
    mass_a, mass_b = entry.inputs["mass_a_kg"], entry.inputs["mass_b_kg"]
    rest_mm = entry.inputs["rest_length_mm"]
    stiffness = entry.inputs["axial_stiffness_n"]
    elongation = entry.inputs["initial_elongation_mm"]

    stiffness_n_mm = stiffness / rest_mm
    reduced_kg = mass_a * mass_b / (mass_a + mass_b)
    omega_per_s = math.sqrt(stiffness_n_mm / reduced_kg * MM_PER_M)

    layout = _layout(2)
    context = EnergyContext(context_id="context/spring", node_masses_kg=(mass_a, mass_b))
    registry = EnergyRegistry(terms=(AxialStretch(edges=((0, 1, rest_mm, stiffness),)),))
    acceleration = registry.acceleration(context, layout)

    x0 = (0.0, 0.0, 0.0, rest_mm + elongation, 0.0, 0.0)
    steps_per_period = entry.inputs["steps_per_period"]
    dt_s = (2.0 * math.pi / omega_per_s) / steps_per_period

    def elongation_after(steps):
        x, _, _ = integrate(
            VELOCITY_VERLET, x0=x0, v0=(0.0,) * 6, dt_s=dt_s, steps=steps,
            acceleration=acceleration,
        )
        return x, (x[3] - x[0]) - rest_mm

    full_state, full = elongation_after(steps_per_period)
    _, half = elongation_after(steps_per_period // 2)
    centre_before = (mass_a * x0[0] + mass_b * x0[3]) / (mass_a + mass_b)
    centre_after = (mass_a * full_state[0] + mass_b * full_state[3]) / (mass_a + mass_b)

    entry.check_all({
        "omega_per_s": omega_per_s,
        "elongation_after_full_period_mm": full,
        "elongation_after_half_period_mm": half,
        "centre_of_mass_drift_mm": centre_after - centre_before,
    })


def test_finite_difference_agrees_with_the_analytic_gradient_and_hessian():
    """有限差分门。**它绿不代表物理对**——上面那条频率门才管物理。"""

    layout = _layout(3)
    context = EnergyContext(
        context_id="context/fd", node_masses_kg=(0.4, 0.7, 0.5),
        gravity_mm_s2=(0.0, -9.80665, 0.0),
    )
    registry = EnergyRegistry(terms=(
        UniformGravity(),
        AxialStretch(edges=((0, 1, 10.0, 2000.0), (1, 2, 12.0, 1500.0))),
    ))
    vector = (1.0, 2.0, -3.0, 11.5, 1.0, 2.0, 20.0, -4.0, 5.5)
    state = State(layout=layout, vector=vector)
    _, gradient, hessian = registry.total(
        state, context, need_gradient=True, need_hessian=True
    )
    assert gradient is not None and hessian is not None

    step = 1.0e-6
    for index in range(len(vector)):
        plus, minus = list(vector), list(vector)
        plus[index] += step
        minus[index] -= step
        energy_plus, gradient_plus, _ = registry.total(
            State(layout=layout, vector=tuple(plus)), context, need_gradient=True
        )
        energy_minus, gradient_minus, _ = registry.total(
            State(layout=layout, vector=tuple(minus)), context, need_gradient=True
        )
        assert gradient_plus is not None and gradient_minus is not None
        numerical = (energy_plus - energy_minus) / (2 * step)
        assert abs(numerical - gradient[index]) <= 1e-6 * max(abs(gradient[index]), 1.0)
        for column in range(len(vector)):
            numerical_second = (
                gradient_plus[column] - gradient_minus[column]
            ) / (2 * step)
            assert abs(numerical_second - hessian[index][column]) <= 1e-5 * max(
                abs(hessian[index][column]), 1.0
            )
