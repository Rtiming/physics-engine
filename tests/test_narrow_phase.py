"""narrow phase第一片的门：手算穿透值、假阳性消除、诚实降级、位姿旋转。"""

from __future__ import annotations

import math

import pytest

from physics_engine.collision import (
    BroadPhaseCollisionQuery,
    segment_segment_distance_mm,
)
from physics_engine.shapes import (
    Capsule,
    CollisionShape,
    FiniteCylinder,
    PosedBody,
    SimBody,
    Sphere,
)


def _body(name: str, shape, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return PosedBody(
        SimBody(body_id=f"body/{name}", collision=CollisionShape(shape, "fitted")),
        translation_mm=translation,
        rotation_xyzw=rotation,
    )


def test_two_spheres_overlapping_report_exact_penetration():
    a = _body("a", Sphere(radius_mm=10.0))
    b = _body("b", Sphere(radius_mm=10.0), translation=(15.0, 0.0, 0.0))
    events = BroadPhaseCollisionQuery((a, b)).check_state()
    assert len(events) == 1
    assert events[0].confidence == "narrow_phase"
    assert events[0].penetration_mm == pytest.approx(5.0, abs=1e-12)


def test_broad_hit_but_narrow_clear_is_eliminated():
    # AABB相交（各轴距离8<10）但欧氏距离11.31>10：broad的假阳性必须被吃掉。
    a = _body("a", Sphere(radius_mm=5.0))
    b = _body("b", Sphere(radius_mm=5.0), translation=(8.0, 8.0, 0.0))
    assert BroadPhaseCollisionQuery((a, b)).check_state() == ()


def test_sphere_capsule_hand_value():
    capsule = _body("cap", Capsule((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), radius_mm=3.0))
    sphere = _body("sph", Sphere(radius_mm=4.0), translation=(10.0, 5.0, 0.0))
    events = BroadPhaseCollisionQuery((capsule, sphere)).check_state()
    assert events[0].confidence == "narrow_phase"
    assert events[0].penetration_mm == pytest.approx(2.0, abs=1e-12)  # 5-(3+4)=-2


def test_rotated_capsule_uses_world_endpoints():
    # 胶囊沿x，绕z转90°后沿y；球放在y轴上正好压进0.5。
    half = math.sqrt(0.5)
    capsule = _body(
        "cap", Capsule((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), radius_mm=2.0),
        rotation=(0.0, 0.0, half, half),
    )
    sphere = _body("sph", Sphere(radius_mm=1.0), translation=(0.0, 10.0, 2.5))
    events = BroadPhaseCollisionQuery((capsule, sphere)).check_state()
    assert events[0].confidence == "narrow_phase"
    assert events[0].penetration_mm == pytest.approx(0.5, abs=1e-9)


def test_unsupported_pair_stays_broad_phase_honestly():
    sphere = _body("sph", Sphere(radius_mm=30.0))
    roller = _body("rol", FiniteCylinder(radius_mm=45.0, half_width_mm=9.0),
                   translation=(40.0, 0.0, 0.0))
    events = BroadPhaseCollisionQuery((sphere, roller)).check_state()
    assert events[0].confidence == "broad_phase"
    assert events[0].penetration_mm is None


def test_parallel_segments_distance():
    assert segment_segment_distance_mm(
        (0, 0, 0), (10, 0, 0), (0, 3, 0), (10, 3, 0)
    ) == pytest.approx(3.0, abs=1e-12)


def test_crossing_segments_distance():
    assert segment_segment_distance_mm(
        (-5, 0, 1), (5, 0, 1), (0, -5, -1), (0, 5, -1)
    ) == pytest.approx(2.0, abs=1e-12)
