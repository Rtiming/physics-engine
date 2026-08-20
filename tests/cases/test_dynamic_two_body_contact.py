"""P3-M3：两dynamic刚体的显式候选、作用反作用、力矩与耦合RK4。"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.dynamic_contact import (
    DynamicNormalContactLaw,
    DynamicSpherePairRuntime,
    integrate_dynamic_sphere_pair,
)
from physics_engine.oracles import load_manifest
from physics_engine.rigidbody import centre_of_mass_velocity_mm_per_s
from tools.bench_dynamic_contact import (
    BODY_A,
    BODY_B,
    build_dynamic_sphere_pair_fixture,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "dynamic_two_body_contact"
ORACLES = {entry.id: entry for entry in load_manifest(CASE / "oracle.json").oracles}


def _assert_value(entry, name: str, actual) -> None:
    tolerance = entry.tolerances[name]
    assert actual == pytest.approx(
        entry.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
    )


def test_eccentric_contact_wrenches_match_the_independent_geometry_oracle():
    entry = ORACLES["oracle:dynamic-contact/eccentric-wrench"]
    prepared = build_dynamic_sphere_pair_fixture(
        centroid_y_mm=entry.inputs["centroid_y_mm"]
    )
    runtime = DynamicSpherePairRuntime(
        prepared,
        DynamicNormalContactLaw(
            entry.inputs["normal_stiffness_n_per_mm"],
            entry.inputs["normal_damping_n_s_per_mm"],
        ),
    )
    response = runtime.evaluate(prepared.initial_dynamic_states(), time_s=0.0)
    a, b = response.wrench(BODY_A), response.wrench(BODY_B)
    actual = {
        "penetration_mm": response.penetration_mm,
        "normal_ab": response.normal_ab,
        "witness_a_mm": a.contact_point_world_mm,
        "witness_b_mm": b.contact_point_world_mm,
        "force_a_world_n": a.force_world_n,
        "force_b_world_n": b.force_world_n,
        "lever_a_world_mm": a.lever_world_mm,
        "lever_b_world_mm": b.lever_world_mm,
        "torque_a_body_nmm": a.torque_body_nmm,
        "torque_b_body_nmm": b.torque_body_nmm,
        "total_force_world_n": response.total_force_world_n,
        "total_torque_about_origin_world_nmm": (
            response.total_torque_about_origin_world_nmm
        ),
    }
    for name, value in actual.items():
        _assert_value(entry, name, value)


def test_aligned_release_matches_the_independent_quarter_period_oracle():
    entry = ORACLES["oracle:dynamic-contact/aligned-quarter-period"]
    prepared = build_dynamic_sphere_pair_fixture(centroid_y_mm=0.0)
    runtime = DynamicSpherePairRuntime(
        prepared,
        DynamicNormalContactLaw(entry.inputs["normal_stiffness_n_per_mm"], 0.0),
    )
    result = integrate_dynamic_sphere_pair(
        runtime,
        states=prepared.initial_dynamic_states(),
        dt_s=entry.inputs["dt_s"],
        steps=entry.inputs["steps"],
    )
    states = result.as_mapping()
    posed = prepared.posed_bodies_at_time(result.final_time_s, dynamic_states=states)
    velocities = {
        body_id: centre_of_mass_velocity_mm_per_s(states[body_id])
        for body_id in (BODY_A, BODY_B)
    }
    total_momentum = tuple(
        velocities[BODY_A][axis] + velocities[BODY_B][axis] for axis in range(3)
    )
    kinetic_nmm = sum(
        0.5 * sum(value * value for value in velocities[body_id]) / 1000.0
        for body_id in (BODY_A, BODY_B)
    )
    actual = {
        "final_centre_a_x_mm": posed[0].translation_mm[0],
        "final_centre_b_x_mm": posed[1].translation_mm[0],
        "final_velocity_a_x_mm_per_s": velocities[BODY_A][0],
        "final_velocity_b_x_mm_per_s": velocities[BODY_B][0],
        "total_linear_momentum_kg_mm_per_s": total_momentum,
        "final_kinetic_energy_nmm": kinetic_nmm,
        "derivative_evaluations": result.diagnostics.derivative_evaluations,
        "renormalisations_per_body": result.diagnostics.renormalisations[0][1],
    }
    for name, value in actual.items():
        _assert_value(entry, name, value)
    assert result.diagnostics.renormalisations[1][1] == entry.expected[
        "renormalisations_per_body"
    ]
    assert result.diagnostics.dissipated_energy_nmm == 0.0
    assert runtime.body_ids == (BODY_A, BODY_B)
