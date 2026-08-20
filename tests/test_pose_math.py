"""位姿组合必须只有一套明确的parent-from-child语义。"""

from __future__ import annotations

import math

import pytest

from physics_engine.motion import Pose
from physics_engine.pose_math import IDENTITY_POSE, compose_pose
from physics_engine.shapes import CollisionShape, PosedBody, SimBody, Sphere


def test_pose_composition_applies_the_child_translation_in_the_parent_axes():
    half = math.sqrt(0.5)
    root_from_component = Pose((10.0, 0.0, 0.0), (0.0, 0.0, half, half))
    component_from_asset = Pose((2.0, 0.0, 0.0), (0.0, 0.0, half, half))

    root_from_asset = compose_pose(root_from_component, component_from_asset)

    assert root_from_asset.translation_mm == pytest.approx((10.0, 2.0, 0.0))
    assert root_from_asset.rotation_xyzw == pytest.approx((0.0, 0.0, 1.0, 0.0))


def test_composed_pose_matches_two_explicit_point_transforms():
    half = math.sqrt(0.5)
    root_from_component = Pose((10.0, 0.0, 0.0), (0.0, 0.0, half, half))
    component_from_asset = Pose((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    point_asset = (3.0, 0.0, 0.0)

    def posed(pose: Pose) -> PosedBody:
        return PosedBody(
            body=SimBody(
                body_id="body/probe",
                collision=CollisionShape(Sphere(1.0), "fitted"),
            ),
            translation_mm=pose.translation_mm,
            rotation_xyzw=pose.rotation_xyzw,
        )

    point_component = posed(component_from_asset).transform_point_mm(point_asset)
    expected = posed(root_from_component).transform_point_mm(point_component)
    actual = posed(compose_pose(root_from_component, component_from_asset)).transform_point_mm(
        point_asset
    )
    assert actual == pytest.approx(expected)


def test_identity_pose_is_a_two_sided_identity():
    pose = Pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert compose_pose(IDENTITY_POSE, pose) == pose
    assert compose_pose(pose, IDENTITY_POSE) == pose
