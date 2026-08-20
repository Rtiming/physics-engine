"""P3.1：模型—运动—物理输入装配成可查询Scene。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from physics_engine.geometry import MassProperties
from physics_engine.materials import EvidenceRef, MaterialProperty, MaterialRecord
from physics_engine.model_physics import (
    BodyBehavior,
    DynamicBodyInitialState,
    DynamicStateFrame,
    GeometrySource,
    ModelPhysicsRelation,
    PhysicsBodyBinding,
    PhysicsModelMotionInput,
    VirtualFrameBinding,
    load_physics_model_motion_input,
)
from physics_engine.model_scene import (
    ModelSceneError,
    SceneInteractionPlan,
    assemble_model_physics_scene,
    reference_component_poses,
)
from physics_engine.model_snapshot import (
    AssetRole,
    ModelAssetRef,
    ModelComponent,
    ModelSnapshot,
)
from physics_engine.motion import InterpolationSemantics, Pose
from physics_engine.planned_motion import (
    MotionParameterization,
    MotionSourceArtifact,
    MotionStateCoordinate,
    MotionTrack,
    PlannedMotion,
    PlannedMotionSample,
    TrackPose,
)
from physics_engine.pose_math import IDENTITY_POSE
from physics_engine.scene_resources import (
    CollisionAssetLoadSpec,
    MassPropertiesRecord,
    SceneResourceCatalog,
    load_collision_asset,
)


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/model-scene-synthetic",
        method="Synthetic P3.1 scene assembly fixture.",
    )


def _semantics() -> InterpolationSemantics:
    return InterpolationSemantics(
        translation_interpolation="linear",
        rotation_interpolation="hold_previous",
        rotation_arc="not_applicable",
        pause_hold="hold_interval_start",
        extrapolation="reject",
    )


def _write_asset(root: Path, relative: str, payload: bytes) -> ModelAssetRef:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ModelAssetRef(
        asset_id=f"asset/{path.stem.replace('.', '-')}",
        role=AssetRole.COLLISION,
        path_relative=relative,
        sha256=hashlib.sha256(payload).hexdigest(),
        format="asset",
        units="mm",
        frame_id=f"frame/{path.stem.replace('.', '-')}",
        component_from_asset=(
            Pose((1.0, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw)
            if "workpiece" in relative
            else IDENTITY_POSE
        ),
    )


def _fixture(
    root: Path, *, parameterization: MotionParameterization = MotionParameterization.TIME_S
) -> tuple[PhysicsModelMotionInput, SceneResourceCatalog, SceneInteractionPlan]:
    tension_asset = _write_asset(
        root, "assets/tension-machine.collision.asset", b"tension-machine\n"
    )
    workpiece_asset = _write_asset(
        root, "assets/workpiece.collision.asset", b"workpiece\n"
    )
    robot_visual = ModelAssetRef(
        asset_id="asset/robot-visual",
        role=AssetRole.VISUAL,
        path_relative="assets/robot.glb",
        sha256="a" * 64,
        format="glb",
        units="mm",
        frame_id="frame/robot-visual",
        component_from_asset=IDENTITY_POSE,
    )
    model = ModelSnapshot.create(
        model_id="model/model-scene-synthetic",
        root_frame_id="frame/root",
        producer_id="producer/test-fixture",
        source_manifest_sha256="b" * 64,
        components=(
            ModelComponent(
                "model-component/tension-machine",
                "frame/tension-machine",
                "tension_machine",
                None,
                IDENTITY_POSE,
                None,
                tension_asset,
            ),
            ModelComponent(
                "model-component/workpiece",
                "frame/workpiece",
                "workpiece",
                None,
                Pose((30.0, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw),
                None,
                workpiece_asset,
            ),
            ModelComponent(
                "model-component/robot-display",
                "frame/robot-display",
                "robot_display",
                None,
                IDENTITY_POSE,
                robot_visual,
                None,
            ),
        ),
    )
    tracks = (
        MotionTrack(
            "motion-track/workpiece",
            "model-component/workpiece",
            "frame/workpiece",
            _semantics(),
        ),
        MotionTrack(
            "motion-track/process",
            None,
            "frame/process",
            _semantics(),
        ),
        MotionTrack(
            "motion-track/robot-display",
            "model-component/robot-display",
            "frame/robot-display",
            _semantics(),
        ),
    )
    samples = tuple(
        PlannedMotionSample(
            sample_id=f"motion-sample/{index:06d}",
            coordinate=float(index),
            path_progress=float(index),
            track_poses=(
                TrackPose(
                    "motion-track/workpiece",
                    Pose((20.0 * (1 - index), 0.0, 0.0), IDENTITY_POSE.rotation_xyzw),
                ),
                TrackPose(
                    "motion-track/process",
                    Pose((10.0 + 2.0 * index, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw),
                ),
                TrackPose(
                    "motion-track/robot-display",
                    Pose((50.0 + 2.0 * index, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw),
                ),
            ),
            source_state_values=(5.0 * index,),
            material_feed_length_mm=100.0 * index,
            stage_id="laydown",
        )
        for index in range(2)
    )
    motion = PlannedMotion.create(
        motion_id="motion/model-scene-synthetic",
        producer_id="producer/wii-adapter-fixture",
        root_frame_id="frame/root",
        parameterization=parameterization,
        coordinate_unit=(
            "s" if parameterization is MotionParameterization.TIME_S else "1"
        ),
        source_artifacts=(
            MotionSourceArtifact("selected_plan", "artifact/model-scene-plan", "c" * 64),
        ),
        tracks=tracks,
        state_coordinates=(MotionStateCoordinate("state-coordinate/a1", "deg"),),
        samples=samples,
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/model-scene-synthetic",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(
            PhysicsBodyBinding(
                "body/tension-machine",
                "model-component/tension-machine",
                BodyBehavior.STATIC,
                GeometrySource.COLLISION_ASSET,
                None,
                None,
                None,
                None,
            ),
            PhysicsBodyBinding(
                "body/workpiece",
                "model-component/workpiece",
                BodyBehavior.KINEMATIC,
                GeometrySource.COLLISION_ASSET,
                "motion-track/workpiece",
                None,
                None,
                None,
            ),
        ),
        virtual_frame_bindings=(
            VirtualFrameBinding(
                "virtual-frame/process", "process_frame", "motion-track/process"
            ),
        ),
        excluded_component_ids=("model-component/robot-display",),
        excluded_motion_track_ids=("motion-track/robot-display",),
    )
    package = PhysicsModelMotionInput.create(
        input_id="physics-input/model-scene-synthetic",
        model=model,
        motion=motion,
        relation=relation,
        evidence=_evidence(),
    )
    resources = SceneResourceCatalog(
        collision_assets=(
            load_collision_asset(
                root,
                tension_asset,
                CollisionAssetLoadSpec(
                    tension_asset.asset_id,
                    "fitted",
                    "nonconvex_declared",
                    (-2.0, -2.0, -2.0),
                    (2.0, 2.0, 2.0),
                ),
            ),
            load_collision_asset(
                root,
                workpiece_asset,
                CollisionAssetLoadSpec(
                    workpiece_asset.asset_id,
                    "fitted",
                    "nonconvex_declared",
                    (-1.0, -1.0, -1.0),
                    (1.0, 1.0, 1.0),
                ),
            ),
        ),
        analytic_shapes=(),
    )
    interactions = SceneInteractionPlan(
        scene_id="scene/model-scene-synthetic",
        contact_pairs=(("body/tension-machine", "body/workpiece"),),
        allowed_pairs=(),
    )
    return package, resources, interactions


def _body_x(posed_bodies, body_id: str) -> float:
    return next(
        posed.translation_mm[0]
        for posed in posed_bodies
        if posed.body.body_id == body_id
    )


def test_time_plan_assembles_static_kinematic_virtual_and_excluded_layers(tmp_path: Path):
    package, resources, interactions = _fixture(tmp_path)
    prepared = assemble_model_physics_scene(package, resources, interactions)

    assert prepared.scene.body_ids == ("body/tension-machine", "body/workpiece")
    assert prepared.body_runtime("body/workpiece").behavior is BodyBehavior.KINEMATIC
    assert prepared.excluded_component_ids == ("model-component/robot-display",)
    assert prepared.excluded_motion_track_ids == ("motion-track/robot-display",)
    assert _body_x(prepared.posed_bodies_at_time(0.0), "body/workpiece") == 21.0
    assert _body_x(prepared.posed_bodies_at_time(1.0), "body/workpiece") == 1.0
    assert prepared.virtual_frame_pose_at_time("virtual-frame/process", 0.5).translation_mm == (
        11.0,
        0.0,
        0.0,
    )


def test_scene_contact_candidates_reach_the_existing_collision_query(tmp_path: Path):
    package, resources, interactions = _fixture(tmp_path)
    prepared = assemble_model_physics_scene(package, resources, interactions)

    before = prepared.collision_query_at_time(0.0).check_state_with_stats()
    after = prepared.collision_query_at_time(1.0).check_state_with_stats()
    assert before.candidate_pair_count == after.candidate_pair_count == 1
    assert before.events == ()
    assert len(after.events) == 1
    assert after.events[0].confidence == "broad_phase"


def test_planning_scale_has_its_own_runtime_and_never_masquerades_as_time(tmp_path: Path):
    package, resources, interactions = _fixture(
        tmp_path, parameterization=MotionParameterization.PLANNING_SCALE
    )
    prepared = assemble_model_physics_scene(package, resources, interactions)

    assert all(body.motion_source is None for body in prepared.scene.bodies)
    assert _body_x(
        prepared.posed_bodies_at_planning_scale(0.5), "body/workpiece"
    ) == pytest.approx(11.0)
    assert prepared.virtual_frame_pose_at_planning_scale(
        "virtual-frame/process", 0.5
    ).translation_mm == pytest.approx((11.0, 0.0, 0.0))
    with pytest.raises(ModelSceneError, match="not physical time"):
        prepared.posed_bodies_at_time(0.5)


def test_red_time_plan_cannot_be_sampled_through_the_planning_scale_api(tmp_path: Path):
    package, resources, interactions = _fixture(tmp_path)
    prepared = assemble_model_physics_scene(package, resources, interactions)
    with pytest.raises(ModelSceneError, match="not planning_scale"):
        prepared.posed_bodies_at_planning_scale(0.5)


def test_red_missing_collision_resource_is_rejected_with_body_identity(tmp_path: Path):
    package, resources, interactions = _fixture(tmp_path)
    incomplete = SceneResourceCatalog(
        collision_assets=(resources.collision_assets[0],), analytic_shapes=()
    )
    with pytest.raises(ModelSceneError, match="body/workpiece.*no collision asset"):
        assemble_model_physics_scene(package, incomplete, interactions)


def _dynamic_fixture(
    tmp_path: Path,
    *,
    parameterization: MotionParameterization = MotionParameterization.TIME_S,
):
    package, resources, interactions = _fixture(
        tmp_path, parameterization=parameterization
    )
    retained_tracks = package.motion.tracks[1:]
    retained_ids = {track.track_id for track in retained_tracks}
    samples = tuple(
        replace(
            sample,
            track_poses=tuple(
                pose for pose in sample.track_poses if pose.track_id in retained_ids
            ),
        )
        for sample in package.motion.samples
    )
    motion = PlannedMotion.create(
        motion_id=package.motion.motion_id,
        producer_id=package.motion.producer_id,
        root_frame_id=package.motion.root_frame_id,
        parameterization=package.motion.parameterization,
        coordinate_unit=package.motion.coordinate_unit,
        source_artifacts=package.motion.source_artifacts,
        tracks=retained_tracks,
        state_coordinates=package.motion.state_coordinates,
        samples=samples,
    )
    static_binding = package.relation.body_bindings[0]
    material = MaterialRecord(
        material_id="material/workpiece",
        applicable_domains=("mechanics",),
        properties=(
            MaterialProperty(
                "density_kg_m3",
                1000.0,
                ("mechanics",),
                EvidenceRef(
                    "estimated",
                    "evidence/model-scene-dynamic-material",
                    "Synthetic dynamic scene fixture.",
                ),
            ),
        ),
    ).sealed()
    component = package.model.component("model-component/workpiece")
    assert component.collision_asset is not None
    mass = MassPropertiesRecord.create(
        mass_properties_id="mass-properties/workpiece",
        geometry_resource_id=component.collision_asset.asset_id,
        expressed_in_frame_id=component.collision_asset.frame_id,
        properties=MassProperties(
            volume_mm3=8.0,
            centroid_mm=(2.0, 0.0, 0.0),
            mass_kg=1.0,
            inertia_about_centroid_kg_mm2=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 2.0),
            ),
        ),
        evidence=EvidenceRef(
            "estimated",
            "evidence/model-scene-dynamic-mass",
            "Synthetic dynamic scene mass properties.",
        ),
    )
    dynamic_binding = PhysicsBodyBinding(
        "body/workpiece",
        "model-component/workpiece",
        BodyBehavior.DYNAMIC,
        GeometrySource.COLLISION_ASSET,
        None,
        None,
        "material/workpiece",
        "mass-properties/workpiece",
        DynamicBodyInitialState(
            state_frame=DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES,
            centre_of_mass_velocity_mm_per_s=(0.0, 0.0, 0.0),
            angular_velocity_body_rad_per_s=(0.0, 0.0, math.pi),
        ),
        material.content_sha256,
        mass.content_sha256,
    )
    relation = ModelPhysicsRelation.create(
        relation_id=package.relation.relation_id,
        model_snapshot_sha256=package.model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(static_binding, dynamic_binding),
        virtual_frame_bindings=package.relation.virtual_frame_bindings,
        excluded_component_ids=package.relation.excluded_component_ids,
        excluded_motion_track_ids=package.relation.excluded_motion_track_ids,
    )
    dynamic_package = PhysicsModelMotionInput.create(
        input_id=package.input_id,
        model=package.model,
        motion=motion,
        relation=relation,
        evidence=package.evidence,
    )
    resources = replace(
        resources,
        materials=(material,),
        mass_property_records=(mass,),
    )
    return dynamic_package, resources, interactions


def test_dynamic_body_uses_explicit_com_state_to_drive_scene_geometry(tmp_path: Path):
    package, resources, interactions = _dynamic_fixture(tmp_path)
    prepared = assemble_model_physics_scene(package, resources, interactions)
    states = prepared.initial_dynamic_states()
    assert states["body/workpiece"].block("centre_of_mass_position_mm") == (
        33.0,
        0.0,
        0.0,
    )
    assert prepared.body_runtime("body/workpiece").dynamic_runtime is not None
    with pytest.raises(ModelSceneError, match="dynamic states are required"):
        prepared.posed_bodies_at_time(0.0)
    posed = prepared.posed_bodies_at_time(0.0, dynamic_states=states)
    assert _body_x(posed, "body/workpiece") == 31.0


def test_dynamic_initial_state_survives_the_strict_outer_package_round_trip(tmp_path: Path):
    package, _, _ = _dynamic_fixture(tmp_path)
    payload = json.dumps(package.to_document()).encode()
    loaded = load_physics_model_motion_input(payload)
    binding = loaded.relation.body_bindings[1]
    assert binding.dynamic_initial_state == package.relation.body_bindings[1].dynamic_initial_state


def test_red_dynamic_state_mapping_must_be_exact(tmp_path: Path):
    package, resources, interactions = _dynamic_fixture(tmp_path)
    prepared = assemble_model_physics_scene(package, resources, interactions)
    with pytest.raises(ModelSceneError, match="missing dynamic states"):
        prepared.posed_bodies_at_time(0.0, dynamic_states={})
    states = prepared.initial_dynamic_states()
    states["body/ghost"] = states["body/workpiece"]
    with pytest.raises(ModelSceneError, match="unknown dynamic states"):
        prepared.posed_bodies_at_time(0.0, dynamic_states=states)
    with pytest.raises(ModelSceneError, match="must be a mapping"):
        prepared.posed_bodies_at_time(0.0, dynamic_states=[])  # type: ignore[arg-type]
    with pytest.raises(ModelSceneError, match="keys must be nonempty body IDs"):
        prepared.posed_bodies_at_time(
            0.0,
            dynamic_states={7: states["body/workpiece"]},  # type: ignore[dict-item]
        )


def test_red_dynamic_body_requires_physical_time_parameterization(tmp_path: Path):
    package, resources, interactions = _dynamic_fixture(
        tmp_path, parameterization=MotionParameterization.PLANNING_SCALE
    )

    with pytest.raises(ModelSceneError, match="dynamic.*physical time"):
        assemble_model_physics_scene(package, resources, interactions)


def test_reference_component_hierarchy_is_composed_in_model_order(tmp_path: Path):
    package, _, _ = _fixture(tmp_path)
    parent = package.model.components[0]
    child = ModelComponent(
        component_id="model-component/child",
        frame_id="frame/child",
        semantic_role="guide",
        parent_component_id=parent.component_id,
        parent_from_component=Pose(
            (3.0, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw
        ),
        visual_asset=None,
        collision_asset=None,
    )
    model = ModelSnapshot.create(
        model_id=package.model.model_id,
        root_frame_id=package.model.root_frame_id,
        producer_id=package.model.producer_id,
        source_manifest_sha256=package.model.source_manifest_sha256,
        components=(replace(parent, parent_from_component=Pose(
            (2.0, 0.0, 0.0), IDENTITY_POSE.rotation_xyzw
        )), child),
    )

    poses = dict(reference_component_poses(model))
    assert poses[parent.component_id].translation_mm == (2.0, 0.0, 0.0)
    assert poses[child.component_id].translation_mm == (5.0, 0.0, 0.0)
