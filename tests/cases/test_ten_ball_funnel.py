"""十球平面漏斗的系统组合门（阶段3最小诚实规模）。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from physics_engine.contact import (
    LinearNormalDashpot,
    PenaltyNormalContact,
    PenaltySphereContact,
    linear_dashpot_parameters,
)
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
from physics_engine.integrate import (
    VELOCITY_VERLET_DAMPED,
    advise_step,
    integrate_with_dissipation,
)
from physics_engine.state import State, StateField, StateLayout

pytestmark = pytest.mark.batch

CASE = Path(__file__).resolve().parents[2] / "cases" / "ten_ball_funnel"
DOCUMENT = json.loads((CASE / "criteria.json").read_text(encoding="utf-8"))
PARAMETERS = DOCUMENT["parameters"]
CRITERIA = DOCUMENT["criteria"]


@pytest.fixture(scope="module")
def run() -> dict:
    ball_count = PARAMETERS["ball_count"]
    radius = PARAMETERS["radius_mm"]
    mass = PARAMETERS["mass_kg"]
    stiffness = PARAMETERS["stiffness_n_per_mm"]
    half_width = PARAMETERS["bottom_half_width_mm"]
    slope = PARAMETERS["wall_slope_dx_per_dz"]
    normalizer = math.sqrt(1.0 + slope * slope)
    plane_specs = (
        ((-half_width, 0.0, 0.0), (1.0 / normalizer, 0.0, slope / normalizer)),
        ((half_width, 0.0, 0.0), (-1.0 / normalizer, 0.0, slope / normalizer)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    layout = StateLayout(
        layout_id="layout/ten_ball_funnel",
        fields=tuple(
            field
            for node in range(ball_count)
            for field in (
                StateField(f"node{node}_x_mm", 1),
                StateField(f"node{node}_y_mm", 1),
                StateField(f"node{node}_z_mm", 1),
            )
        ),
    )
    plane_entries = tuple(
        (node, point, normal, stiffness, radius)
        for node in range(ball_count)
        for point, normal in plane_specs
    )
    sphere_pairs = tuple(
        (left, right, 2.0 * radius, stiffness)
        for left in range(ball_count)
        for right in range(left + 1, ball_count)
    )
    wall = linear_dashpot_parameters(
        stiffness_n_per_mm=stiffness,
        effective_mass_kg=mass,
        restitution=PARAMETERS["wall_restitution"],
    )
    sphere = linear_dashpot_parameters(
        stiffness_n_per_mm=stiffness,
        effective_mass_kg=mass / 2.0,
        restitution=PARAMETERS["sphere_restitution"],
    )
    plane_contact = PenaltyNormalContact(planes=plane_entries)
    sphere_contact = PenaltySphereContact(pairs=sphere_pairs)
    dashpot = LinearNormalDashpot(
        planes=tuple(
            (node, point, normal, stiffness, wall.damping_n_s_per_mm, radius)
            for node in range(ball_count)
            for point, normal in plane_specs
        ),
        sphere_pairs=tuple(
            (left, right, 2.0 * radius, stiffness, sphere.damping_n_s_per_mm)
            for left in range(ball_count)
            for right in range(left + 1, ball_count)
        ),
    )
    context = EnergyContext(
        context_id="context/ten_ball_funnel",
        node_masses_kg=(mass,) * ball_count,
        gravity_mm_s2=(0.0, 0.0, -PARAMETERS["gravity_mm_s2"]),
    )
    registry = EnergyRegistry(
        terms=(UniformGravity(), plane_contact, sphere_contact, dashpot)
    )
    initial_x = tuple(value for point in PARAMETERS["initial_centres_mm"] for value in point)
    initial_v = (0.0,) * len(initial_x)
    initial_state = State(layout=layout, vector=initial_x)
    initial_potential = registry.total(initial_state, context)[0]
    coefficient = VELOCITY_VERLET_DAMPED.declaration.oscillatory_step_coefficient
    wall_step = advise_step(
        wall.stability_rate_per_s,
        oscillatory_step_coefficient=coefficient,
        steps_per_contact=PARAMETERS["steps_per_contact"],
        contact_duration_s=wall.contact_duration_s,
    ).advised_step_s
    sphere_step = advise_step(
        sphere.stability_rate_per_s,
        oscillatory_step_coefficient=coefficient,
        steps_per_contact=PARAMETERS["steps_per_contact"],
        contact_duration_s=sphere.contact_duration_s,
    ).advised_step_s
    dt = min(wall_step, sphere_step)
    steps = math.ceil(PARAMETERS["duration_s"] / dt)

    started = time.perf_counter()
    result = integrate_with_dissipation(
        VELOCITY_VERLET_DAMPED,
        x0=initial_x,
        v0=initial_v,
        dt_s=dt,
        steps=steps,
        acceleration=registry.acceleration(context, layout),
        dissipation_rate=registry.dissipation_rate(context, layout),
    )
    elapsed = time.perf_counter() - started
    final_state = State(layout=layout, vector=result.x)
    final_potential = registry.total(final_state, context)[0]
    speeds = tuple(
        math.sqrt(sum(result.v[3 * node + axis] ** 2 for axis in range(3)))
        for node in range(ball_count)
    )
    final_kinetic = sum(0.5 * mass * speed * speed / 1000.0 for speed in speeds)
    plane_forces = plane_contact.normal_force_n(final_state)
    sphere_forces = sphere_contact.contact_force_n(final_state)
    plane_gaps = tuple(
        PenaltyNormalContact._gap_mm(result.x, node, point, normal, radius)
        for node in range(ball_count)
        for point, normal in plane_specs
    )
    sphere_gaps = tuple(
        PenaltySphereContact._pair_state(result.x, pair)[0] for pair in sphere_pairs
    )
    residual = (
        initial_potential
        - final_potential
        - final_kinetic
        - result.dissipated_energy_nmm
    )
    return {
        "ball_count": ball_count,
        "dof": len(result.x),
        "final_center_of_mass_z_mm": sum(
            result.x[3 * node + 2] for node in range(ball_count)
        ) / ball_count,
        "final_rms_speed_mm_s": math.sqrt(sum(speed * speed for speed in speeds) / ball_count),
        "max_plane_penetration_mm": max(0.0, -min(plane_gaps)),
        "max_sphere_penetration_mm": max(0.0, -min(sphere_gaps)),
        "active_wall_contacts": sum(force > 0.0 for force in plane_forces),
        "active_sphere_contacts": sum(force > 0.0 for force in sphere_forces),
        "dissipated_energy_nmm": result.dissipated_energy_nmm,
        "relative_energy_balance_residual": abs(residual) / result.dissipated_energy_nmm,
        "out_of_plane_abs_mm": max(
            abs(result.x[3 * node + 1]) for node in range(ball_count)
        ),
        "elapsed_s": elapsed,
        "steps": steps,
    }


def test_the_case_really_contains_ten_independently_moving_balls(run):
    """必须红：少一球或把十球压成一个质心都不算兑现10—20球裁决。"""

    assert run["ball_count"] == 10
    assert run["dof"] == 3 * run["ball_count"]


def test_the_batch_enters_the_funnel_and_settles_at_the_declared_scale(run):
    assert run["final_center_of_mass_z_mm"] <= CRITERIA["final_center_of_mass_z_mm_max"]
    assert run["final_rms_speed_mm_s"] <= CRITERIA["final_rms_speed_mm_s_max"]
    assert run["max_plane_penetration_mm"] <= CRITERIA["max_plane_penetration_mm"]
    assert run["max_sphere_penetration_mm"] <= CRITERIA["max_sphere_penetration_mm"]


def test_wall_and_ball_ball_response_are_both_exercised(run):
    """必须红：十条互不相碰的独立落体不能冒充多体落料。"""

    assert run["active_wall_contacts"] >= CRITERIA["active_wall_contacts_min"]
    assert run["active_sphere_contacts"] >= CRITERIA["active_sphere_contacts_min"]


def test_physical_dissipation_is_positive_and_the_energy_ledger_is_honest(run):
    assert run["dissipated_energy_nmm"] >= CRITERIA["dissipated_energy_nmm_min"]
    assert (
        run["relative_energy_balance_residual"]
        <= CRITERIA["relative_energy_balance_residual_max"]
    )


def test_the_declared_two_dimensional_subspace_is_invariant(run):
    assert run["out_of_plane_abs_mm"] <= CRITERIA["out_of_plane_abs_mm_max"]
