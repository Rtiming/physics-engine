"""P3-M0 conformance：通用模型—运动—物理绑定对独立oracle。"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.materials import EvidenceRef
from physics_engine.model_physics import (
    BodyBehavior,
    GeometrySource,
    ModelPhysicsRelation,
    PhysicsBodyBinding,
    PhysicsModelMotionInput,
    VirtualFrameBinding,
)
from physics_engine.model_snapshot import (
    AssetRole,
    ModelAssetRef,
    ModelComponent,
    ModelSnapshot,
)
from physics_engine.motion import InterpolationSemantics, Pose
from physics_engine.oracles import load_manifest
from physics_engine.planned_motion import (
    MotionParameterization,
    MotionSourceArtifact,
    MotionStateCoordinate,
    MotionTrack,
    PlannedMotion,
    PlannedMotionSample,
    TrackPose,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "model_motion_physics_binding"
ORACLE = load_manifest(CASE / "oracle.json").oracles[0]


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/model-motion-case-synthetic",
        method="Synthetic P3-M0 ownership and interpolation case.",
    )


def _asset(asset_id, role, sha, path):
    return ModelAssetRef(
        asset_id,
        role,
        path,
        sha,
        path.rsplit(".", 1)[-1],
        "mm",
        "frame/root",
        Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


def _package() -> PhysicsModelMotionInput:
    identity = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    components = (
        ModelComponent(
            "model-component/tension-machine",
            "tension_machine",
            None,
            identity,
            _asset("asset/tm-v", AssetRole.VISUAL, "a" * 64, "assets/tm.glb"),
            _asset("asset/tm-c", AssetRole.COLLISION, "b" * 64, "assets/tm.stl"),
        ),
        ModelComponent(
            "model-component/workpiece",
            "workpiece",
            None,
            identity,
            _asset("asset/wp-v", AssetRole.VISUAL, "b" * 64, "assets/wp.glb"),
            _asset("asset/wp-c", AssetRole.COLLISION, "c" * 64, "assets/wp.stl"),
        ),
        ModelComponent(
            "model-component/robot-display",
            "robot_display",
            None,
            identity,
            _asset("asset/r-v", AssetRole.VISUAL, "c" * 64, "assets/r.glb"),
            None,
        ),
    )
    model = ModelSnapshot.create(
        model_id="model/p3-m0", root_frame_id="frame/root",
        producer_id="producer/test-fixture", source_manifest_sha256="d" * 64,
        components=components,
    )
    semantics = InterpolationSemantics(
        "linear", "slerp", "shortest", "hold_interval_start", "reject"
    )
    tracks = (
        MotionTrack("motion-track/workpiece", "model-component/workpiece", "frame/workpiece", semantics),
        MotionTrack("motion-track/process", None, "frame/process", semantics),
        MotionTrack(
            "motion-track/robot-display",
            "model-component/robot-display",
            "frame/robot-display",
            semantics,
        ),
    )
    inputs = ORACLE.inputs
    rotations = (
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    samples = tuple(
        PlannedMotionSample(
            sample_id=f"motion-sample/{index:06d}",
            coordinate=inputs["sample_coordinates_s"][index],
            path_progress=float(index),
            track_poses=(
                TrackPose(
                    "motion-track/workpiece",
                    Pose(
                        (inputs["workpiece_translation_x_mm"][index], 0.0, 0.0),
                        rotations[index],
                    ),
                ),
                TrackPose(
                    "motion-track/process",
                    Pose(
                        (inputs["process_translation_x_mm"][index], 0.0, 0.0),
                        rotations[index],
                    ),
                ),
                TrackPose(
                    "motion-track/robot-display",
                    Pose((20.0 + 2.0 * index, 0.0, 0.0), rotations[index]),
                ),
            ),
            source_state_values=(0.0, 0.0) if index == 0 else (5.0, 360.0),
            material_feed_length_mm=inputs["material_feed_length_mm"][index],
            stage_id="laydown",
        )
        for index in range(2)
    )
    motion = PlannedMotion.create(
        motion_id="motion/p3-m0", producer_id="producer/wii-adapter-fixture",
        root_frame_id="frame/root", parameterization=MotionParameterization.TIME_S,
        coordinate_unit="s",
        source_artifacts=(
            MotionSourceArtifact("selected_plan", "artifact/p3-m0-plan", "e" * 64),
        ),
        tracks=tracks,
        state_coordinates=(
            MotionStateCoordinate("state-coordinate/a1", "deg"),
            MotionStateCoordinate("state-coordinate/e1", "deg"),
        ),
        samples=samples,
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/p3-m0",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(
            PhysicsBodyBinding(
                "body/tension-machine", "model-component/tension-machine",
                BodyBehavior.STATIC, GeometrySource.COLLISION_ASSET,
                None, None, None, None,
            ),
            PhysicsBodyBinding(
                "body/workpiece", "model-component/workpiece",
                BodyBehavior.KINEMATIC, GeometrySource.COLLISION_ASSET,
                "motion-track/workpiece", None, None, None,
            ),
        ),
        virtual_frame_bindings=(
            VirtualFrameBinding("virtual-frame/process", "process_frame", "motion-track/process"),
        ),
        excluded_component_ids=("model-component/robot-display",),
        excluded_motion_track_ids=("motion-track/robot-display",),
    )
    return PhysicsModelMotionInput.create(
        input_id="physics-input/p3-m0", model=model, motion=motion,
        relation=relation, evidence=_evidence(),
    )


def test_ownership_and_midpoint_match_the_independent_oracle():
    package = _package()
    midpoint = package.motion.as_time_source("motion-track/workpiece").pose_at(0.5)
    actual = {
        "physical_body_ids": [item.body_id for item in package.relation.body_bindings],
        "body_behaviors": [item.behavior.value for item in package.relation.body_bindings],
        "excluded_component_ids": list(package.relation.excluded_component_ids),
        "excluded_motion_track_ids": list(
            package.relation.excluded_motion_track_ids
        ),
        "virtual_frame_roles": [item.role for item in package.relation.virtual_frame_bindings],
        "midpoint_workpiece_translation_mm": list(midpoint.translation_mm),
        "midpoint_workpiece_rotation_xyzw": list(midpoint.rotation_xyzw),
        "final_source_state_values": list(package.motion.samples[-1].source_state_values),
        "final_material_feed_length_mm": package.motion.samples[-1].material_feed_length_mm,
    }
    for name, value in actual.items():
        tolerance = ORACLE.tolerances[name]
        expected = ORACLE.expected[name]
        if isinstance(expected, list) and expected and isinstance(expected[0], str):
            assert value == expected
        else:
            assert value == pytest.approx(
                expected, rel=tolerance.rel_tol, abs=tolerance.abs_tol
            )


def test_visual_only_robot_is_present_in_model_but_absent_from_virtual_physics():
    package = _package()
    assert package.model.component("model-component/robot-display").visual_asset is not None
    assert package.model.component("model-component/robot-display").collision_asset is None
    assert "model-component/robot-display" in package.relation.excluded_component_ids
    assert package.motion.track("motion-track/robot-display").component_id == (
        "model-component/robot-display"
    )
    assert "motion-track/robot-display" in package.relation.excluded_motion_track_ids
    assert all(
        item.component_id != "model-component/robot-display"
        for item in package.relation.body_bindings
    )
