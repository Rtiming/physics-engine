"""P3-M2：dynamic COM状态、惯量与Scene几何位姿闭环。"""

from __future__ import annotations

import hashlib
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
from physics_engine.rigidbody import RK4_BODY, integrate_free_flight
from physics_engine.scene_resources import (
    CollisionAssetLoadSpec,
    MassPropertiesRecord,
    SceneResourceCatalog,
    load_collision_asset,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "dynamic_model_scene_free_flight"
ORACLE = load_manifest(CASE / "oracle.json").oracles[0]


def _prepared():
    inputs = ORACLE.inputs
    path = CASE / "assets" / "workpiece.collision.asset"
    asset = ModelAssetRef(
        asset_id="asset/p3-m2-workpiece",
        role=AssetRole.COLLISION,
        path_relative="assets/workpiece.collision.asset",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        format="asset",
        units="mm",
        frame_id="frame/p3-m2-geometry",
        component_from_asset=Pose(
            (inputs["geometry_origin_x_mm"], 0.0, 0.0),
            IDENTITY_POSE.rotation_xyzw,
        ),
    )
    model = ModelSnapshot.create(
        model_id="model/p3-m2",
        root_frame_id="frame/root",
        producer_id="producer/test-fixture",
        source_manifest_sha256="a" * 64,
        components=(
            ModelComponent(
                component_id="model-component/workpiece",
                frame_id="frame/workpiece",
                semantic_role="workpiece",
                parent_component_id=None,
                parent_from_component=IDENTITY_POSE,
                visual_asset=None,
                collision_asset=asset,
            ),
        ),
    )
    semantics = InterpolationSemantics(
        "hold_previous",
        "hold_previous",
        "not_applicable",
        "hold_interval_start",
        "reject",
    )
    track = MotionTrack(
        "motion-track/process", None, "frame/process", semantics
    )
    samples = tuple(
        PlannedMotionSample(
            sample_id=f"motion-sample/p3-m2-{index}",
            coordinate=index * inputs["dt_s"] * inputs["steps"],
            path_progress=float(index),
            track_poses=(TrackPose("motion-track/process", IDENTITY_POSE),),
            source_state_values=(),
            material_feed_length_mm=0.0,
            stage_id="free_flight",
        )
        for index in range(2)
    )
    motion = PlannedMotion.create(
        motion_id="motion/p3-m2",
        producer_id="producer/test-fixture",
        root_frame_id="frame/root",
        parameterization=MotionParameterization.TIME_S,
        coordinate_unit="s",
        source_artifacts=(
            MotionSourceArtifact("initial_state", "artifact/p3-m2-state", "b" * 64),
        ),
        tracks=(track,),
        state_coordinates=(),
        samples=samples,
    )
    material = MaterialRecord(
        material_id="material/p3-m2",
        applicable_domains=("mechanics",),
        properties=(
            MaterialProperty(
                "density_kg_m3",
                1000.0,
                ("mechanics",),
                EvidenceRef(
                    "estimated",
                    "evidence/p3-m2-material",
                    "Synthetic dynamic free-flight material.",
                ),
            ),
        ),
    ).sealed()
    mass = MassPropertiesRecord.create(
        mass_properties_id="mass-properties/p3-m2",
        geometry_resource_id=asset.asset_id,
        expressed_in_frame_id=asset.frame_id,
        properties=MassProperties(
            volume_mm3=8.0,
            centroid_mm=(inputs["centroid_in_geometry_x_mm"], 0.0, 0.0),
            mass_kg=1.0,
            inertia_about_centroid_kg_mm2=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 2.0),
            ),
        ),
        evidence=EvidenceRef(
            "estimated",
            "evidence/p3-m2-mass",
            "Synthetic dynamic free-flight mass properties.",
        ),
    )
    binding = PhysicsBodyBinding(
        body_id="body/workpiece",
        component_id="model-component/workpiece",
        behavior=BodyBehavior.DYNAMIC,
        geometry_source=GeometrySource.COLLISION_ASSET,
        motion_track_id=None,
        analytic_shape_id=None,
        material_record_id="material/p3-m2",
        mass_properties_id="mass-properties/p3-m2",
        dynamic_initial_state=DynamicBodyInitialState(
            DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, inputs["angular_velocity_z_rad_per_s"]),
        ),
        material_record_sha256=material.content_sha256,
        mass_properties_sha256=mass.content_sha256,
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/p3-m2",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(binding,),
        virtual_frame_bindings=(
            VirtualFrameBinding(
                "virtual-frame/process", "process_frame", "motion-track/process"
            ),
        ),
        excluded_component_ids=(),
        excluded_motion_track_ids=(),
    )
    package = PhysicsModelMotionInput.create(
        input_id="physics-input/p3-m2",
        model=model,
        motion=motion,
        relation=relation,
        evidence=EvidenceRef(
            "estimated", "evidence/p3-m2", "Synthetic dynamic free-flight case."
        ),
    )
    resources = SceneResourceCatalog(
        collision_assets=(
            load_collision_asset(
                CASE,
                asset,
                CollisionAssetLoadSpec(
                    asset.asset_id,
                    "fitted",
                    "nonconvex_declared",
                    (-1.0, -1.0, -1.0),
                    (1.0, 1.0, 1.0),
                ),
            ),
        ),
        analytic_shapes=(),
        materials=(material,),
        mass_property_records=(mass,),
    )
    prepared = assemble_model_physics_scene(
        package,
        resources,
        SceneInteractionPlan("scene/p3-m2", (), ()),
    )
    return package, prepared


def test_dynamic_geometry_orbits_com_under_free_flight_against_closed_form():
    package, prepared = _prepared()
    expected = ORACLE.expected
    runtime = prepared.body_runtime("body/workpiece").dynamic_runtime
    assert runtime is not None
    initial = runtime.initial_state
    initial_posed = prepared.posed_bodies_at_time(
        0.0, dynamic_states={"body/workpiece": initial}
    )[0]
    final, diagnostics = integrate_free_flight(
        RK4_BODY,
        state=initial,
        inertia=runtime.inertia,
        dt_s=ORACLE.inputs["dt_s"],
        steps=ORACLE.inputs["steps"],
    )
    final_posed = prepared.posed_bodies_at_time(
        0.5, dynamic_states={"body/workpiece": final}
    )[0]
    actual = {
        "initial_com_position_mm": list(initial.block("centre_of_mass_position_mm")),
        "initial_geometry_origin_mm": list(initial_posed.translation_mm),
        "final_com_position_mm": list(final.block("centre_of_mass_position_mm")),
        "final_attitude_xyzw": list(final.block("attitude_body_to_world_xyzw")),
        "final_geometry_origin_mm": list(final_posed.translation_mm),
        "renormalisations": diagnostics.renormalisations,
        "qualification": package.qualification,
    }
    for name, value in actual.items():
        tolerance = ORACLE.tolerances[name]
        target = expected[name]
        if isinstance(target, str):
            assert value == target
        else:
            assert value == pytest.approx(
                target, rel=tolerance.rel_tol, abs=tolerance.abs_tol
            )


def test_dynamic_case_asset_and_candidate_scope_are_explicit():
    package, prepared = _prepared()
    asset = package.model.components[0].collision_asset
    assert asset is not None and asset.sha256 == ORACLE.inputs["asset_sha256"]
    states = prepared.initial_dynamic_states()
    result = prepared.collision_query_at_time(
        0.0, dynamic_states=states
    ).check_state_with_stats()
    assert result.candidate_pair_count == 0
    assert result.events == ()
