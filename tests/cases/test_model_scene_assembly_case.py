"""P3-M1案例：模型输入经模块化装配进入Scene与CollisionQuery。"""

from __future__ import annotations

import hashlib
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
from physics_engine.model_scene import SceneInteractionPlan, assemble_model_physics_scene
from physics_engine.model_snapshot import AssetRole, ModelAssetRef, ModelComponent, ModelSnapshot
from physics_engine.motion import InterpolationSemantics, Pose
from physics_engine.oracles import load_manifest
from physics_engine.planned_motion import (
    MotionParameterization,
    MotionSourceArtifact,
    MotionTrack,
    PlannedMotion,
    PlannedMotionSample,
    TrackPose,
)
from physics_engine.pose_math import IDENTITY_POSE
from physics_engine.scene_resources import (
    CollisionAssetLoadSpec,
    SceneResourceCatalog,
    load_collision_asset,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "model_scene_assembly"
ORACLE = load_manifest(CASE / "oracle.json").oracles[0]


def _semantics() -> InterpolationSemantics:
    return InterpolationSemantics(
        "linear", "hold_previous", "not_applicable", "hold_interval_start", "reject"
    )


def _asset(asset_id: str, path_relative: str, frame_id: str, offset_x: float):
    path = CASE / path_relative
    return ModelAssetRef(
        asset_id=asset_id,
        role=AssetRole.COLLISION,
        path_relative=path_relative,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        format="asset",
        units="mm",
        frame_id=frame_id,
        component_from_asset=Pose(
            (offset_x, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw
        ),
    )


def _prepared():
    tension = _asset(
        "asset/p3-m1-tension", "assets/tension-machine.collision.asset", "frame/tension", 0.0
    )
    workpiece = _asset(
        "asset/p3-m1-workpiece", "assets/workpiece.collision.asset", "frame/workpiece", 1.0
    )
    robot_visual = ModelAssetRef(
        "asset/p3-m1-robot-visual",
        AssetRole.VISUAL,
        "assets/robot.glb",
        "d" * 64,
        "glb",
        "mm",
        "frame/robot",
        IDENTITY_POSE,
    )
    model = ModelSnapshot.create(
        model_id="model/p3-m1",
        root_frame_id="frame/root",
        producer_id="producer/test-fixture",
        source_manifest_sha256="e" * 64,
        components=(
            ModelComponent(
                "model-component/tension-machine", "frame/tension-machine",
                "tension_machine", None,
                IDENTITY_POSE, None, tension,
            ),
            ModelComponent(
                "model-component/workpiece", "frame/workpiece", "workpiece", None,
                IDENTITY_POSE, None, workpiece,
            ),
            ModelComponent(
                "model-component/robot-display", "frame/robot",
                "robot_display", None,
                IDENTITY_POSE, robot_visual, None,
            ),
        ),
    )
    tracks = (
        MotionTrack(
            "motion-track/workpiece", "model-component/workpiece", "frame/workpiece",
            _semantics(),
        ),
        MotionTrack("motion-track/process", None, "frame/process", _semantics()),
        MotionTrack(
            "motion-track/robot-display", "model-component/robot-display", "frame/robot",
            _semantics(),
        ),
    )
    inputs = ORACLE.inputs
    samples = tuple(
        PlannedMotionSample(
            sample_id=f"motion-sample/p3-m1-{index}",
            coordinate=inputs["sample_times_s"][index],
            path_progress=float(index),
            track_poses=(
                TrackPose(
                    "motion-track/workpiece",
                    Pose(
                        (inputs["workpiece_component_x_mm"][index], 0.0, 0.0),
                        IDENTITY_POSE.rotation_xyzw,
                    ),
                ),
                TrackPose(
                    "motion-track/process",
                    Pose(
                        (inputs["process_frame_x_mm"][index], 0.0, 0.0),
                        IDENTITY_POSE.rotation_xyzw,
                    ),
                ),
                TrackPose(
                    "motion-track/robot-display",
                    Pose((50.0 + index, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw),
                ),
            ),
            source_state_values=(),
            material_feed_length_mm=100.0 * index,
            stage_id="laydown",
        )
        for index in range(2)
    )
    motion = PlannedMotion.create(
        motion_id="motion/p3-m1",
        producer_id="producer/test-fixture",
        root_frame_id="frame/root",
        parameterization=MotionParameterization.TIME_S,
        coordinate_unit="s",
        source_artifacts=(
            MotionSourceArtifact("selected_plan", "artifact/p3-m1-plan", "f" * 64),
        ),
        tracks=tracks,
        state_coordinates=(),
        samples=samples,
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/p3-m1",
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
    package = PhysicsModelMotionInput.create(
        input_id="physics-input/p3-m1",
        model=model,
        motion=motion,
        relation=relation,
        evidence=EvidenceRef(
            grade="estimated",
            evidence_id="evidence/p3-m1-synthetic",
            method="Synthetic scene assembly case.",
        ),
    )
    resources = SceneResourceCatalog(
        collision_assets=(
            load_collision_asset(
                CASE, tension,
                CollisionAssetLoadSpec(
                    tension.asset_id, "fitted", "nonconvex_declared",
                    (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0),
                ),
            ),
            load_collision_asset(
                CASE, workpiece,
                CollisionAssetLoadSpec(
                    workpiece.asset_id, "fitted", "nonconvex_declared",
                    (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
                ),
            ),
        ),
        analytic_shapes=(),
    )
    prepared = assemble_model_physics_scene(
        package,
        resources,
        SceneInteractionPlan(
            "scene/p3-m1",
            (("body/tension-machine", "body/workpiece"),),
            (),
        ),
    )
    return package, prepared


def test_scene_motion_virtual_frame_and_collision_match_independent_oracle():
    package, prepared = _prepared()
    expected = ORACLE.expected
    body_x = [
        next(
            body.translation_mm[0]
            for body in prepared.posed_bodies_at_time(time_s)
            if body.body.body_id == "body/workpiece"
        )
        for time_s in (0.0, 0.5, 1.0)
    ]
    results = [
        prepared.collision_query_at_time(time_s).check_state_with_stats()
        for time_s in (0.0, 1.0)
    ]
    actual = {
        "body_ids": list(prepared.scene.body_ids),
        "excluded_component_ids": list(prepared.excluded_component_ids),
        "excluded_motion_track_ids": list(prepared.excluded_motion_track_ids),
        "workpiece_asset_x_mm": body_x,
        "process_frame_midpoint_x_mm": prepared.virtual_frame_pose_at_time(
            "virtual-frame/process", 0.5
        ).translation_mm[0],
        "candidate_pair_count": [item.candidate_pair_count for item in results],
        "event_count": [len(item.events) for item in results],
        "event_confidence": results[-1].events[0].confidence,
        "qualification": package.qualification,
    }
    for name, value in actual.items():
        tolerance = ORACLE.tolerances[name]
        target = expected[name]
        if isinstance(target, (str, list)) and (
            isinstance(target, str) or not target or isinstance(target[0], str)
        ):
            assert value == target
        else:
            assert value == pytest.approx(
                target, rel=tolerance.rel_tol, abs=tolerance.abs_tol
            )


def test_case_assets_are_the_exact_bytes_named_by_the_oracle():
    package, _ = _prepared()
    by_id = {
        component.component_id: component.collision_asset
        for component in package.model.components
    }
    assert by_id["model-component/tension-machine"].sha256 == (
        ORACLE.inputs["tension_asset_sha256"]
    )
    assert by_id["model-component/workpiece"].sha256 == (
        ORACLE.inputs["workpiece_asset_sha256"]
    )
