"""`tools.bench`的P3.1固定装配语料；构造不混入计时。"""

from __future__ import annotations

from dataclasses import replace

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
)
from physics_engine.model_scene import SceneInteractionPlan
from physics_engine.model_snapshot import AssetRole, ModelAssetRef, ModelComponent, ModelSnapshot
from physics_engine.motion import InterpolationSemantics, Pose
from physics_engine.planned_motion import (
    MotionParameterization,
    MotionSourceArtifact,
    MotionTrack,
    PlannedMotion,
    PlannedMotionSample,
    TrackPose,
)
from physics_engine.scene_resources import (
    LoadedCollisionAsset,
    MassPropertiesRecord,
    SceneResourceCatalog,
)
from physics_engine.shapes import CollisionShape, MeshAsset

IDENTITY = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _asset(name: str, sha256: str) -> ModelAssetRef:
    return ModelAssetRef(
        asset_id=f"asset/bench-{name}",
        role=AssetRole.COLLISION,
        path_relative=f"assets/{name}.collision.asset",
        sha256=sha256,
        format="asset",
        units="mm",
        frame_id=f"frame/bench-{name}",
        component_from_asset=IDENTITY,
    )


def _loaded(asset: ModelAssetRef, half_extent_mm: float) -> LoadedCollisionAsset:
    return LoadedCollisionAsset(
        asset=asset,
        collision=CollisionShape(
            MeshAsset(
                path_relative=asset.path_relative,
                sha256=asset.sha256,
                units="mm",
                usage="collision",
                convexity="nonconvex_declared",
                aabb_min_mm=(-half_extent_mm,) * 3,
                aabb_max_mm=(half_extent_mm,) * 3,
            ),
            "fitted",
        ),
        byte_length=32,
    )


def build_model_scene_benchmark_fixture() -> tuple[
    PhysicsModelMotionInput, SceneResourceCatalog, SceneInteractionPlan
]:
    """返回固定两体、两track、两样点语料。"""

    tension = _asset("tension", "a" * 64)
    workpiece = _asset("workpiece", "b" * 64)
    model = ModelSnapshot.create(
        model_id="model/bench-scene",
        root_frame_id="frame/bench-root",
        producer_id="producer/bench",
        source_manifest_sha256="c" * 64,
        components=(
            ModelComponent(
                "model-component/bench-tension",
                "frame/bench-tension",
                "tension_machine",
                None,
                IDENTITY,
                None,
                tension,
            ),
            ModelComponent(
                "model-component/bench-workpiece",
                "frame/bench-workpiece",
                "workpiece",
                None,
                IDENTITY,
                None,
                workpiece,
            ),
        ),
    )
    semantics = InterpolationSemantics(
        "linear", "hold_previous", "not_applicable", "hold_interval_start", "reject"
    )
    tracks = (
        MotionTrack(
            "motion-track/bench-workpiece",
            "model-component/bench-workpiece",
            "frame/bench-workpiece",
            semantics,
        ),
        MotionTrack(
            "motion-track/bench-process", None, "frame/bench-process", semantics
        ),
    )
    samples = tuple(
        PlannedMotionSample(
            sample_id=f"motion-sample/bench-scene-{index}",
            coordinate=float(index),
            path_progress=float(index),
            track_poses=(
                TrackPose(
                    "motion-track/bench-workpiece",
                    Pose((float(index), 0.0, 0.0), IDENTITY.rotation_xyzw),
                ),
                TrackPose(
                    "motion-track/bench-process",
                    Pose((10.0 + index, 0.0, 0.0), IDENTITY.rotation_xyzw),
                ),
            ),
            source_state_values=(),
            material_feed_length_mm=float(index),
            stage_id="laydown",
        )
        for index in range(2)
    )
    motion = PlannedMotion.create(
        motion_id="motion/bench-scene",
        producer_id="producer/bench",
        root_frame_id="frame/bench-root",
        parameterization=MotionParameterization.TIME_S,
        coordinate_unit="s",
        source_artifacts=(
            MotionSourceArtifact("selected_plan", "artifact/bench-scene-plan", "d" * 64),
        ),
        tracks=tracks,
        state_coordinates=(),
        samples=samples,
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/bench-scene",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(
            PhysicsBodyBinding(
                "body/bench-tension",
                "model-component/bench-tension",
                BodyBehavior.STATIC,
                GeometrySource.COLLISION_ASSET,
                None,
                None,
                None,
                None,
            ),
            PhysicsBodyBinding(
                "body/bench-workpiece",
                "model-component/bench-workpiece",
                BodyBehavior.KINEMATIC,
                GeometrySource.COLLISION_ASSET,
                "motion-track/bench-workpiece",
                None,
                None,
                None,
            ),
        ),
        virtual_frame_bindings=(
            VirtualFrameBinding(
                "virtual-frame/bench-process",
                "process_frame",
                "motion-track/bench-process",
            ),
        ),
        excluded_component_ids=(),
        excluded_motion_track_ids=(),
    )
    package = PhysicsModelMotionInput.create(
        input_id="physics-input/bench-scene",
        model=model,
        motion=motion,
        relation=relation,
        evidence=EvidenceRef(
            grade="estimated",
            evidence_id="evidence/bench-scene",
            method="Synthetic benchmark fixture.",
        ),
    )
    resources = SceneResourceCatalog(
        collision_assets=(_loaded(tension, 2.0), _loaded(workpiece, 1.0)),
        analytic_shapes=(),
    )
    interactions = SceneInteractionPlan(
        scene_id="scene/bench-model-scene",
        contact_pairs=(("body/bench-tension", "body/bench-workpiece"),),
        allowed_pairs=(),
    )
    return package, resources, interactions


def build_dynamic_model_scene_benchmark_fixture() -> tuple[
    PhysicsModelMotionInput, SceneResourceCatalog, SceneInteractionPlan
]:
    """由同一两体语料构造一静一动、只保留虚拟frame时间线的dynamic场景。"""

    package, resources, interactions = build_model_scene_benchmark_fixture()
    process_track = package.motion.tracks[1]
    samples = tuple(
        replace(
            sample,
            track_poses=tuple(
                pose
                for pose in sample.track_poses
                if pose.track_id == process_track.track_id
            ),
        )
        for sample in package.motion.samples
    )
    motion = PlannedMotion.create(
        motion_id="motion/bench-dynamic-scene",
        producer_id=package.motion.producer_id,
        root_frame_id=package.motion.root_frame_id,
        parameterization=MotionParameterization.TIME_S,
        coordinate_unit="s",
        source_artifacts=package.motion.source_artifacts,
        tracks=(process_track,),
        state_coordinates=(),
        samples=samples,
    )
    static_binding = package.relation.body_bindings[0]
    material = MaterialRecord(
        material_id="material/bench-dynamic",
        applicable_domains=("mechanics",),
        properties=(
            MaterialProperty(
                "density_kg_m3",
                1000.0,
                ("mechanics",),
                EvidenceRef(
                    "estimated",
                    "evidence/bench-dynamic-material",
                    "Synthetic benchmark material.",
                ),
            ),
        ),
    ).sealed()
    component = package.model.component("model-component/bench-workpiece")
    assert component.collision_asset is not None
    mass = MassPropertiesRecord.create(
        mass_properties_id="mass-properties/bench-dynamic",
        geometry_resource_id=component.collision_asset.asset_id,
        expressed_in_frame_id=component.collision_asset.frame_id,
        properties=MassProperties(
            8.0,
            (2.0, 0.0, 0.0),
            1.0,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)),
        ),
        evidence=EvidenceRef(
            "estimated",
            "evidence/bench-dynamic-mass",
            "Synthetic benchmark mass properties.",
        ),
    )
    dynamic_binding = PhysicsBodyBinding(
        "body/bench-workpiece",
        "model-component/bench-workpiece",
        BodyBehavior.DYNAMIC,
        GeometrySource.COLLISION_ASSET,
        None,
        None,
        "material/bench-dynamic",
        "mass-properties/bench-dynamic",
        DynamicBodyInitialState(
            DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 3.141592653589793),
        ),
        material.content_sha256,
        mass.content_sha256,
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/bench-dynamic-scene",
        model_snapshot_sha256=package.model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(static_binding, dynamic_binding),
        virtual_frame_bindings=package.relation.virtual_frame_bindings,
        excluded_component_ids=(),
        excluded_motion_track_ids=(),
    )
    dynamic_package = PhysicsModelMotionInput.create(
        input_id="physics-input/bench-dynamic-scene",
        model=package.model,
        motion=motion,
        relation=relation,
        evidence=package.evidence,
    )
    dynamic_resources = replace(
        resources,
        materials=(material,),
        mass_property_records=(mass,),
    )
    return dynamic_package, dynamic_resources, interactions


__all__ = [
    "build_dynamic_model_scene_benchmark_fixture",
    "build_model_scene_benchmark_fixture",
]
