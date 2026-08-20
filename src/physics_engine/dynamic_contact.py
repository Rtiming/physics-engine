"""Two dynamic rigid bodies coupled through one detected sphere contact.

P3-M3 keeps three authorities separate:

* :mod:`model_scene` owns model resources, dynamic state identities and geometry poses;
* :mod:`collision` owns the declared candidate query, normal and witness points;
* :mod:`rigidbody` owns each body's Euler equation and quaternion kinematics.

This module only assembles the equal-and-opposite normal wrenches and advances the
combined 26-dimensional state.  It deliberately does not call two single-body
integrators in sequence: RK stages must evaluate both bodies at the same coupled state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from physics_engine.collision import CollisionQueryError, CollisionQueryResult
from physics_engine.contact_pipeline import (
    EXPLICIT_EULER_STABILITY_RADIUS,
    RK4_STABILITY_RADIUS,
    contact_stiffness_step_bound,
)
from physics_engine.integrate import VectorOps, default_ops
from physics_engine.model_scene import ModelSceneError, PreparedModelScene
from physics_engine.planned_motion import MotionParameterization
from physics_engine.rigidbody import (
    EXPLICIT_EULER_BODY,
    QUATERNION_NORM_STEP_ABS_TOL,
    RIGID_BODY_LAYOUT,
    RK4_BODY,
    RigidBodyError,
    RigidBodyInertia,
    RigidBodyIntegrator,
    attitude_xyzw,
    centre_of_mass_position_mm,
    centre_of_mass_velocity_mm_per_s,
    cross,
    normalise_quaternion,
    rigid_body_state_derivative,
    rotate_body_to_world,
    rotate_world_to_body,
)
from physics_engine.shapes import GeneratedShape, Sphere
from physics_engine.state import State

Vector3 = tuple[float, float, float]
ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)
WITNESS_IDENTITY_ABS_TOL_MM = 1.0e-9


class DynamicContactError(ValueError):
    """The model scene, contact event or coupled state is not physically closed."""


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[axis] + right[axis] for axis in range(3))  # type: ignore[return-value]


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[axis] - right[axis] for axis in range(3))  # type: ignore[return-value]


def _scale(vector: Vector3, factor: float) -> Vector3:
    return tuple(factor * vector[axis] for axis in range(3))  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(left[axis] * right[axis] for axis in range(3))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


@dataclass(frozen=True)
class DynamicNormalContactLaw:
    """A normal penalty spring and compression-only linear dashpot."""

    normal_stiffness_n_per_mm: float
    normal_damping_n_s_per_mm: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.normal_stiffness_n_per_mm)
            or self.normal_stiffness_n_per_mm <= 0.0
        ):
            raise DynamicContactError(
                "normal_stiffness_n_per_mm must be positive and finite"
            )
        if (
            not math.isfinite(self.normal_damping_n_s_per_mm)
            or self.normal_damping_n_s_per_mm < 0.0
        ):
            raise DynamicContactError(
                "normal_damping_n_s_per_mm must be finite and nonnegative"
            )


@dataclass(frozen=True)
class DynamicBodyWrench:
    """One body's force and torque from the same detected contact event."""

    body_id: str
    centre_of_mass_world_mm: Vector3
    contact_point_world_mm: Vector3 | None
    lever_world_mm: Vector3 | None
    force_world_n: Vector3
    torque_world_nmm: Vector3
    torque_body_nmm: Vector3


@dataclass(frozen=True)
class DynamicPairContactEvaluation:
    """One candidate query plus the two equal-and-opposite body wrenches."""

    body_ids: tuple[str, str]
    active: bool
    penetration_mm: float
    normal_ab: Vector3 | None
    relative_normal_speed_mm_per_s: float
    normal_spring_force_n: float
    normal_damping_force_n: float
    normal_force_n: float
    dissipation_power_nmm_per_s: float
    wrenches: tuple[DynamicBodyWrench, DynamicBodyWrench]
    query: CollisionQueryResult

    def wrench(self, body_id: str) -> DynamicBodyWrench:
        for wrench in self.wrenches:
            if wrench.body_id == body_id:
                return wrench
        raise DynamicContactError(f"contact evaluation has no body {body_id!r}")

    @property
    def total_force_world_n(self) -> Vector3:
        return _add(self.wrenches[0].force_world_n, self.wrenches[1].force_world_n)

    @property
    def total_torque_about_origin_world_nmm(self) -> Vector3:
        total = ZERO_VECTOR
        for wrench in self.wrenches:
            if wrench.contact_point_world_mm is not None:
                total = _add(
                    total,
                    cross(wrench.contact_point_world_mm, wrench.force_world_n),
                )
        return total


@dataclass(frozen=True)
class DynamicPairIntegrationDiagnostics:
    steps: int
    derivative_evaluations: int
    active_derivative_evaluations: int
    dissipated_energy_nmm: float
    max_norm_deviation: tuple[tuple[str, float], ...]
    renormalisations: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class DynamicPairIntegrationResult:
    final_states: tuple[tuple[str, State], tuple[str, State]]
    final_time_s: float
    diagnostics: DynamicPairIntegrationDiagnostics

    def as_mapping(self) -> dict[str, State]:
        return dict(self.final_states)

    def state(self, body_id: str) -> State:
        for candidate, state in self.final_states:
            if candidate == body_id:
                return state
        raise DynamicContactError(f"integration result has no body {body_id!r}")


@dataclass(frozen=True)
class DynamicSpherePairRuntime:
    """Exactly two dynamic sphere geometries and one explicit contact candidate."""

    prepared_scene: PreparedModelScene
    law: DynamicNormalContactLaw
    _body_ids: tuple[str, str] = field(init=False, repr=False)
    _inertias: tuple[RigidBodyInertia, RigidBodyInertia] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_scene, PreparedModelScene):
            raise DynamicContactError("prepared_scene must be PreparedModelScene")
        if not isinstance(self.law, DynamicNormalContactLaw):
            raise DynamicContactError("law must be DynamicNormalContactLaw")
        if self.prepared_scene.parameterization is not MotionParameterization.TIME_S:
            raise DynamicContactError("dynamic contact requires physical time_s")
        dynamic = tuple(
            runtime
            for runtime in self.prepared_scene.physics_bodies
            if runtime.dynamic_runtime is not None
        )
        if len(dynamic) != 2:
            raise DynamicContactError(
                f"first dynamic contact slice requires exactly two dynamic bodies, got {len(dynamic)}"
            )
        pairs = self.prepared_scene.scene.contact_pairs
        if len(pairs) != 1:
            raise DynamicContactError(
                f"first dynamic contact slice requires exactly one candidate pair, got {len(pairs)}"
            )
        body_ids = (pairs[0].body_a, pairs[0].body_b)
        if set(body_ids) != {runtime.body_id for runtime in dynamic}:
            raise DynamicContactError(
                "the explicit candidate pair must contain the two dynamic bodies"
            )
        assembled = {
            body.body_id: body for body in self.prepared_scene.scene.bodies
        }
        for body_id in body_ids:
            shape = assembled[body_id].posed.body.collision.shape
            if isinstance(shape, GeneratedShape):
                shape = shape.shape
            if not isinstance(shape, Sphere):
                raise DynamicContactError(
                    f"{body_id}: P3-M3 first slice only supports Sphere geometry, "
                    f"got {type(shape).__name__}"
                )
        inertias = tuple(
            self.prepared_scene.body_runtime(body_id).dynamic_runtime.inertia
            for body_id in body_ids
        )
        object.__setattr__(self, "_body_ids", body_ids)
        object.__setattr__(self, "_inertias", inertias)

    @property
    def body_ids(self) -> tuple[str, str]:
        return self._body_ids

    def contact_step_bound_s(
        self, integrator: RigidBodyIntegrator = RK4_BODY
    ) -> float:
        if integrator.declaration.name == RK4_BODY.declaration.name:
            stability_radius = RK4_STABILITY_RADIUS
        elif integrator.declaration.name == EXPLICIT_EULER_BODY.declaration.name:
            stability_radius = EXPLICIT_EULER_STABILITY_RADIUS
        else:
            raise DynamicContactError(
                f"no contact stability radius for {integrator.declaration.name!r}"
            )
        mass_a, mass_b = (inertia.mass_kg for inertia in self._inertias)
        effective_mass = mass_a * mass_b / (mass_a + mass_b)
        return contact_stiffness_step_bound(
            stiffness_n_per_mm=self.law.normal_stiffness_n_per_mm,
            effective_mass_kg=effective_mass,
            damping_n_s_per_mm=self.law.normal_damping_n_s_per_mm,
            stability_radius=stability_radius,
            pair_id=f"{self._body_ids[0]}--{self._body_ids[1]}",
        ).step_bound_s

    def evaluate(
        self, states: Mapping[str, State], *, time_s: float
    ) -> DynamicPairContactEvaluation:
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
            raise DynamicContactError("time_s must be numeric")
        time_value = float(time_s)
        if not math.isfinite(time_value):
            raise DynamicContactError("time_s must be finite")
        try:
            query = self.prepared_scene.collision_query_at_time(
                time_value, dynamic_states=states
            ).check_state_with_stats()
        except (CollisionQueryError, ModelSceneError, RigidBodyError, ValueError) as error:
            raise DynamicContactError(f"dynamic contact query failed: {error}") from error
        if query.candidate_pair_count != 1:
            raise DynamicContactError(
                "collision query changed the single explicit candidate identity"
            )
        state_a, state_b = (states[body_id] for body_id in self._body_ids)
        centre_a = centre_of_mass_position_mm(state_a)
        centre_b = centre_of_mass_position_mm(state_b)
        if not query.events:
            return self._inactive_evaluation(query, centre_a, centre_b)
        if len(query.events) != 1:
            raise DynamicContactError(
                f"single candidate query returned {len(query.events)} events"
            )
        event = query.events[0]
        if (event.body_a, event.body_b) != self._body_ids:
            raise DynamicContactError("collision event changed the declared pair order")
        if event.confidence != "narrow_phase" or event.penetration_mm is None:
            raise DynamicContactError(
                "dynamic response requires an exact narrow_phase penetration"
            )
        if (
            event.normal_ab is None
            or event.witness_a_mm is None
            or event.witness_b_mm is None
        ):
            raise DynamicContactError(
                "dynamic response requires normal and both collision witness points"
            )
        normal = event.normal_ab
        penetration = event.penetration_mm
        if not math.isfinite(penetration) or penetration <= 0.0:
            raise DynamicContactError(
                f"dynamic response requires positive finite penetration, got {penetration!r}"
            )
        if not all(
            math.isfinite(value)
            for vector in (normal, event.witness_a_mm, event.witness_b_mm)
            for value in vector
        ):
            raise DynamicContactError("collision normal and witnesses must be finite")
        normal_length = _norm(normal)
        if not math.isclose(normal_length, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise DynamicContactError(
                f"collision normal must be unit length, got {normal_length!r}"
            )
        separation = -penetration
        witness_residual = _sub(
            event.witness_a_mm,
            _add(event.witness_b_mm, _scale(normal, separation)),
        )
        if _norm(witness_residual) > WITNESS_IDENTITY_ABS_TOL_MM:
            raise DynamicContactError(
                "collision normal, witnesses and penetration violate their identity"
            )

        lever_a = _sub(event.witness_a_mm, centre_a)
        lever_b = _sub(event.witness_b_mm, centre_b)
        velocity_a = self._contact_velocity(state_a, lever_a)
        velocity_b = self._contact_velocity(state_b, lever_b)
        normal_speed = _dot(_sub(velocity_a, velocity_b), normal)
        spring_magnitude = self.law.normal_stiffness_n_per_mm * penetration
        damping_magnitude = 0.0
        if normal_speed < 0.0:
            damping_magnitude = -self.law.normal_damping_n_s_per_mm * normal_speed
        magnitude = max(0.0, spring_magnitude + damping_magnitude)
        force_a = _scale(normal, magnitude)
        force_b = _scale(force_a, -1.0)
        wrench_a = self._wrench(
            self._body_ids[0], state_a, event.witness_a_mm, lever_a, force_a
        )
        wrench_b = self._wrench(
            self._body_ids[1], state_b, event.witness_b_mm, lever_b, force_b
        )
        return DynamicPairContactEvaluation(
            body_ids=self._body_ids,
            active=True,
            penetration_mm=penetration,
            normal_ab=normal,
            relative_normal_speed_mm_per_s=normal_speed,
            normal_spring_force_n=spring_magnitude,
            normal_damping_force_n=damping_magnitude,
            normal_force_n=magnitude,
            dissipation_power_nmm_per_s=damping_magnitude * max(-normal_speed, 0.0),
            wrenches=(wrench_a, wrench_b),
            query=query,
        )

    def _inactive_evaluation(
        self, query: CollisionQueryResult, centre_a: Vector3, centre_b: Vector3
    ) -> DynamicPairContactEvaluation:
        wrenches = tuple(
            DynamicBodyWrench(
                body_id=body_id,
                centre_of_mass_world_mm=centre,
                contact_point_world_mm=None,
                lever_world_mm=None,
                force_world_n=ZERO_VECTOR,
                torque_world_nmm=ZERO_VECTOR,
                torque_body_nmm=ZERO_VECTOR,
            )
            for body_id, centre in zip(self._body_ids, (centre_a, centre_b), strict=True)
        )
        return DynamicPairContactEvaluation(
            body_ids=self._body_ids,
            active=False,
            penetration_mm=0.0,
            normal_ab=None,
            relative_normal_speed_mm_per_s=0.0,
            normal_spring_force_n=0.0,
            normal_damping_force_n=0.0,
            normal_force_n=0.0,
            dissipation_power_nmm_per_s=0.0,
            wrenches=wrenches,  # type: ignore[arg-type]
            query=query,
        )

    @staticmethod
    def _contact_velocity(state: State, lever_world_mm: Vector3) -> Vector3:
        omega_world = rotate_body_to_world(
            attitude_xyzw(state), state.block("angular_velocity_body_rad_per_s")
        )
        return _add(
            centre_of_mass_velocity_mm_per_s(state),
            cross(omega_world, lever_world_mm),
        )

    @staticmethod
    def _wrench(
        body_id: str,
        state: State,
        contact_point_world_mm: Vector3,
        lever_world_mm: Vector3,
        force_world_n: Vector3,
    ) -> DynamicBodyWrench:
        torque_world = cross(lever_world_mm, force_world_n)
        return DynamicBodyWrench(
            body_id=body_id,
            centre_of_mass_world_mm=centre_of_mass_position_mm(state),
            contact_point_world_mm=contact_point_world_mm,
            lever_world_mm=lever_world_mm,
            force_world_n=force_world_n,
            torque_world_nmm=torque_world,
            torque_body_nmm=rotate_world_to_body(attitude_xyzw(state), torque_world),
        )


def _normalise_body_vector(vector: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
    state = State(layout=RIGID_BODY_LAYOUT, vector=vector)
    quaternion = attitude_xyzw(state)
    norm = math.sqrt(sum(value * value for value in quaternion))
    deviation = abs(norm - 1.0)
    if deviation > QUATERNION_NORM_STEP_ABS_TOL:
        raise DynamicContactError(
            f"coupled rigid-body quaternion drift {deviation!r} exceeds "
            f"{QUATERNION_NORM_STEP_ABS_TOL!r} before renormalisation"
        )
    normalised = normalise_quaternion(quaternion)
    offset = RIGID_BODY_LAYOUT.offset_of("attitude_body_to_world_xyzw")
    return (vector[:offset] + normalised, deviation)


def integrate_dynamic_sphere_pair(
    runtime: DynamicSpherePairRuntime,
    *,
    states: Mapping[str, State],
    dt_s: float,
    steps: int,
    integrator: RigidBodyIntegrator = RK4_BODY,
    t0_s: float = 0.0,
    ops: VectorOps | None = None,
) -> DynamicPairIntegrationResult:
    """Advance both bodies in one coupled RK system and renormalise each quaternion."""

    if not isinstance(runtime, DynamicSpherePairRuntime):
        raise DynamicContactError("runtime must be DynamicSpherePairRuntime")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise DynamicContactError("steps must be a nonnegative integer")
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)):
        raise DynamicContactError("dt_s must be numeric")
    step_size = float(dt_s)
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise DynamicContactError("dt_s must be positive and finite")
    if isinstance(t0_s, bool) or not isinstance(t0_s, (int, float)):
        raise DynamicContactError("t0_s must be numeric")
    time_s = float(t0_s)
    if not math.isfinite(time_s):
        raise DynamicContactError("t0_s must be finite")
    contact_bound = runtime.contact_step_bound_s(integrator)
    if step_size >= contact_bound:
        raise DynamicContactError(
            f"dt_s {step_size!r} violates contact step bound h < {contact_bound!r}"
        )

    runtime.evaluate(states, time_s=time_s)
    body_a, body_b = runtime.body_ids
    state_a, state_b = states[body_a], states[body_b]
    width = RIGID_BODY_LAYOUT.dof_count
    vector = state_a.vector + state_b.vector
    backend = ops or default_ops()
    derivative_evaluations = 0
    active_evaluations = 0
    stage_dissipation_powers: list[float] = []

    def derivative(combined: tuple[float, ...], at_time_s: float) -> tuple[float, ...]:
        nonlocal derivative_evaluations, active_evaluations
        candidate_a = State(RIGID_BODY_LAYOUT, combined[:width])
        candidate_b = State(RIGID_BODY_LAYOUT, combined[width:])
        evaluation = runtime.evaluate(
            {body_a: candidate_a, body_b: candidate_b}, time_s=at_time_s
        )
        derivative_evaluations += 1
        if evaluation.active:
            active_evaluations += 1
        stage_dissipation_powers.append(evaluation.dissipation_power_nmm_per_s)
        wrench_a = evaluation.wrench(body_a)
        wrench_b = evaluation.wrench(body_b)
        return rigid_body_state_derivative(
            candidate_a.vector,
            inertia=runtime._inertias[0],
            force_world_n=wrench_a.force_world_n,
            torque_body_nmm=wrench_a.torque_body_nmm,
        ) + rigid_body_state_derivative(
            candidate_b.vector,
            inertia=runtime._inertias[1],
            force_world_n=wrench_b.force_world_n,
            torque_body_nmm=wrench_b.torque_body_nmm,
        )

    max_deviation = {body_a: 0.0, body_b: 0.0}
    renormalisations = {body_a: 0, body_b: 0}
    dissipated_energy = 0.0
    for _ in range(steps):
        stage_dissipation_powers.clear()
        vector = integrator.step(vector, time_s, step_size, derivative, backend)
        if integrator.declaration.name == RK4_BODY.declaration.name:
            if len(stage_dissipation_powers) != 4:
                raise DynamicContactError("RK4 contact step did not evaluate four stages")
            dissipated_energy += step_size / 6.0 * (
                stage_dissipation_powers[0]
                + 2.0 * stage_dissipation_powers[1]
                + 2.0 * stage_dissipation_powers[2]
                + stage_dissipation_powers[3]
            )
        elif integrator.declaration.name == EXPLICIT_EULER_BODY.declaration.name:
            if len(stage_dissipation_powers) != 1:
                raise DynamicContactError("Euler contact step did not evaluate one stage")
            dissipated_energy += step_size * stage_dissipation_powers[0]
        else:  # pragma: no cover - `contact_step_bound_s` already rejects this branch
            raise DynamicContactError("unsupported coupled rigid-body integrator")
        normalised_a, deviation_a = _normalise_body_vector(vector[:width])
        normalised_b, deviation_b = _normalise_body_vector(vector[width:])
        vector = normalised_a + normalised_b
        max_deviation[body_a] = max(max_deviation[body_a], deviation_a)
        max_deviation[body_b] = max(max_deviation[body_b], deviation_b)
        renormalisations[body_a] += 1
        renormalisations[body_b] += 1
        time_s += step_size

    final = (
        (body_a, State(RIGID_BODY_LAYOUT, vector[:width])),
        (body_b, State(RIGID_BODY_LAYOUT, vector[width:])),
    )
    return DynamicPairIntegrationResult(
        final_states=final,
        final_time_s=time_s,
        diagnostics=DynamicPairIntegrationDiagnostics(
            steps=steps,
            derivative_evaluations=derivative_evaluations,
            active_derivative_evaluations=active_evaluations,
            dissipated_energy_nmm=dissipated_energy,
            max_norm_deviation=tuple(max_deviation.items()),
            renormalisations=tuple(renormalisations.items()),
        ),
    )


__all__ = [
    "WITNESS_IDENTITY_ABS_TOL_MM",
    "DynamicBodyWrench",
    "DynamicContactError",
    "DynamicNormalContactLaw",
    "DynamicPairContactEvaluation",
    "DynamicPairIntegrationDiagnostics",
    "DynamicPairIntegrationResult",
    "DynamicSpherePairRuntime",
    "integrate_dynamic_sphere_pair",
]
