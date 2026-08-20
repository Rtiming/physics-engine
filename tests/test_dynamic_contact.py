"""P3-M3：模型场景中两个dynamic刚体的显式候选接触与耦合推进。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from physics_engine.dynamic_contact import (
    DynamicContactError,
    DynamicNormalContactLaw,
    DynamicSpherePairRuntime,
    integrate_dynamic_sphere_pair,
)
from physics_engine.rigidbody import (
    angular_momentum_world_kg_mm2_per_s,
    centre_of_mass_position_mm,
    centre_of_mass_velocity_mm_per_s,
    cross,
)
from physics_engine.shapes import RoundedBox
from tools.bench_dynamic_contact import (
    BODY_A,
    BODY_B,
    build_dynamic_sphere_pair_fixture,
)


def _prepared_pair(*, centroid_y_mm: float, shape_b=None):
    return build_dynamic_sphere_pair_fixture(
        centroid_y_mm=centroid_y_mm,
        shape_b=shape_b,
    )


def _total_linear_momentum(states, prepared):
    return tuple(
        sum(
            prepared.body_runtime(body_id).dynamic_runtime.inertia.mass_kg
            * centre_of_mass_velocity_mm_per_s(states[body_id])[axis]
            for body_id in (BODY_A, BODY_B)
        )
        for axis in range(3)
    )


def _total_angular_momentum(states, prepared):
    total = [0.0, 0.0, 0.0]
    for body_id in (BODY_A, BODY_B):
        runtime = prepared.body_runtime(body_id).dynamic_runtime
        state = states[body_id]
        mass = runtime.inertia.mass_kg
        orbital = cross(
            centre_of_mass_position_mm(state),
            tuple(mass * value for value in centre_of_mass_velocity_mm_per_s(state)),
        )
        spin = angular_momentum_world_kg_mm2_per_s(runtime.inertia, state)
        for axis in range(3):
            total[axis] += orbital[axis] + spin[axis]
    return tuple(total)


def _with_velocity(state, velocity):
    offset = state.layout.offset_of("centre_of_mass_velocity_mm_per_s")
    vector = list(state.vector)
    vector[offset : offset + 3] = velocity
    return state.with_vector(tuple(vector))


def test_eccentric_com_contact_uses_collision_witnesses_and_returns_both_wrenches():
    prepared = _prepared_pair(centroid_y_mm=2.0)
    runtime = DynamicSpherePairRuntime(
        prepared,
        DynamicNormalContactLaw(
            normal_stiffness_n_per_mm=100.0,
            normal_damping_n_s_per_mm=0.0,
        ),
    )
    states = prepared.initial_dynamic_states()
    response = runtime.evaluate(states, time_s=0.0)

    assert response.active
    assert response.penetration_mm == pytest.approx(1.0, rel=0.0, abs=1.0e-15)
    assert response.normal_ab == (-1.0, 0.0, 0.0)
    assert response.normal_force_n == pytest.approx(100.0)
    assert response.query.candidate_pair_count == 1
    assert response.query.narrow_phase_check_count == 1
    a = response.wrench(BODY_A)
    b = response.wrench(BODY_B)
    assert a.force_world_n == pytest.approx((-100.0, 0.0, 0.0))
    assert b.force_world_n == pytest.approx((100.0, 0.0, 0.0))
    assert a.lever_world_mm == pytest.approx((10.0, -2.0, 0.0))
    assert b.lever_world_mm == pytest.approx((-10.0, -2.0, 0.0))
    assert a.torque_body_nmm == pytest.approx((0.0, 0.0, -200.0))
    assert b.torque_body_nmm == pytest.approx((0.0, 0.0, 200.0))
    assert response.total_force_world_n == pytest.approx((0.0, 0.0, 0.0), abs=0.0)
    assert response.total_torque_about_origin_world_nmm == pytest.approx(
        (0.0, 0.0, 0.0), abs=1.0e-13
    )


def test_eccentric_pair_integration_preserves_total_linear_and_angular_momentum():
    prepared = _prepared_pair(centroid_y_mm=2.0)
    runtime = DynamicSpherePairRuntime(prepared, DynamicNormalContactLaw(100.0, 0.0))
    initial = prepared.initial_dynamic_states()
    result = integrate_dynamic_sphere_pair(
        runtime,
        states=initial,
        dt_s=2.0e-6,
        steps=100,
    )
    final = result.as_mapping()
    assert _total_linear_momentum(final, prepared) == pytest.approx(
        _total_linear_momentum(initial, prepared), abs=1.0e-12
    )
    assert _total_angular_momentum(final, prepared) == pytest.approx(
        _total_angular_momentum(initial, prepared), abs=2.0e-9
    )
    assert all(value <= 1.0e-12 for _, value in result.diagnostics.max_norm_deviation)


def test_compression_dashpot_reports_force_power_and_integrated_dissipation():
    prepared = _prepared_pair(centroid_y_mm=0.0)
    runtime = DynamicSpherePairRuntime(prepared, DynamicNormalContactLaw(100.0, 0.5))
    states = prepared.initial_dynamic_states()
    moving = {
        BODY_A: _with_velocity(states[BODY_A], (10.0, 0.0, 0.0)),
        BODY_B: _with_velocity(states[BODY_B], (-10.0, 0.0, 0.0)),
    }
    response = runtime.evaluate(moving, time_s=0.0)
    assert response.relative_normal_speed_mm_per_s == pytest.approx(-20.0)
    assert response.normal_spring_force_n == pytest.approx(100.0)
    assert response.normal_damping_force_n == pytest.approx(10.0)
    assert response.normal_force_n == pytest.approx(110.0)
    assert response.dissipation_power_nmm_per_s == pytest.approx(200.0)
    result = integrate_dynamic_sphere_pair(
        runtime, states=moving, dt_s=1.0e-6, steps=1
    )
    assert result.diagnostics.dissipated_energy_nmm > 0.0
    final = result.as_mapping()
    final_response = runtime.evaluate(final, time_s=result.final_time_s)
    final_kinetic_nmm = sum(
        0.5
        * sum(value * value for value in centre_of_mass_velocity_mm_per_s(final[body_id]))
        / 1000.0
        for body_id in (BODY_A, BODY_B)
    )
    final_mechanical_nmm = (
        final_kinetic_nmm
        + 0.5
        * runtime.law.normal_stiffness_n_per_mm
        * final_response.penetration_mm**2
    )
    initial_mechanical_nmm = 50.0 + 0.1
    assert initial_mechanical_nmm - final_mechanical_nmm == pytest.approx(
        result.diagnostics.dissipated_energy_nmm,
        rel=1.0e-9,
        abs=1.0e-12,
    )


def test_must_be_red_the_pair_runtime_refuses_a_non_sphere_geometry():
    prepared = _prepared_pair(
        centroid_y_mm=0.0,
        shape_b=RoundedBox((10.0, 10.0, 10.0), 0.0),
    )
    with pytest.raises(DynamicContactError, match="Sphere"):
        DynamicSpherePairRuntime(prepared, DynamicNormalContactLaw(100.0, 0.0))


def test_must_be_red_a_missing_dynamic_state_cannot_fall_back_to_the_scene_pose():
    prepared = _prepared_pair(centroid_y_mm=0.0)
    runtime = DynamicSpherePairRuntime(prepared, DynamicNormalContactLaw(100.0, 0.0))
    states = prepared.initial_dynamic_states()
    with pytest.raises(DynamicContactError, match="missing dynamic states"):
        runtime.evaluate({BODY_A: states[BODY_A]}, time_s=0.0)


def test_must_be_red_inconsistent_collision_witnesses_cannot_enter_response(monkeypatch):
    prepared = _prepared_pair(centroid_y_mm=0.0)
    runtime = DynamicSpherePairRuntime(prepared, DynamicNormalContactLaw(100.0, 0.0))
    states = prepared.initial_dynamic_states()
    real = prepared.collision_query_at_time(
        0.0, dynamic_states=states
    ).check_state_with_stats()
    event = real.events[0]
    assert event.witness_a_mm is not None
    bad = replace(
        real,
        events=(
            replace(
                event,
                witness_a_mm=(
                    event.witness_a_mm[0],
                    event.witness_a_mm[1] + 0.25,
                    event.witness_a_mm[2],
                ),
            ),
        ),
    )

    class _BadQuery:
        def check_state_with_stats(self):
            return bad

    def _bad_query(_self, _time_s, *, dynamic_states):
        assert dynamic_states == states
        return _BadQuery()

    monkeypatch.setattr(type(prepared), "collision_query_at_time", _bad_query)
    with pytest.raises(DynamicContactError, match="violate their identity"):
        runtime.evaluate(states, time_s=0.0)


def test_must_be_red_a_nonunit_collision_normal_cannot_enter_response(monkeypatch):
    prepared = _prepared_pair(centroid_y_mm=0.0)
    runtime = DynamicSpherePairRuntime(prepared, DynamicNormalContactLaw(100.0, 0.0))
    states = prepared.initial_dynamic_states()
    real = prepared.collision_query_at_time(
        0.0, dynamic_states=states
    ).check_state_with_stats()
    event = real.events[0]
    assert event.witness_b_mm is not None
    bad_normal = (-2.0, 0.0, 0.0)
    bad = replace(
        real,
        events=(
            replace(
                event,
                normal_ab=bad_normal,
                witness_a_mm=(1.5, 0.0, 0.0),
            ),
        ),
    )

    class _BadQuery:
        def check_state_with_stats(self):
            return bad

    def _bad_query(_self, _time_s, *, dynamic_states):
        assert dynamic_states == states
        return _BadQuery()

    monkeypatch.setattr(type(prepared), "collision_query_at_time", _bad_query)
    with pytest.raises(DynamicContactError, match="unit length"):
        runtime.evaluate(states, time_s=0.0)


def test_must_be_red_a_step_at_or_above_the_contact_bound_is_rejected():
    prepared = _prepared_pair(centroid_y_mm=0.0)
    runtime = DynamicSpherePairRuntime(prepared, DynamicNormalContactLaw(100.0, 0.0))
    bound = runtime.contact_step_bound_s()
    with pytest.raises(DynamicContactError, match="contact step bound"):
        integrate_dynamic_sphere_pair(
            runtime,
            states=prepared.initial_dynamic_states(),
            dt_s=bound,
            steps=1,
        )
