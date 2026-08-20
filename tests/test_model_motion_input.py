"""模型快照、运动计划与模型—虚拟物理绑定合同。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from physics_engine.canonical import canonical_sha256
from physics_engine.materials import EvidenceRef
from physics_engine.model_physics import (
    PHYSICS_MODEL_MOTION_CANONICAL_PROFILE,
    BodyBehavior,
    DynamicBodyInitialState,
    DynamicStateFrame,
    GeometrySource,
    ModelPhysicsError,
    ModelPhysicsRelation,
    PhysicsBodyBinding,
    PhysicsModelMotionInput,
    VirtualFrameBinding,
    load_physics_model_motion_input,
)
from physics_engine.model_snapshot import (
    AssetRole,
    ModelAssetRef,
    ModelComponent,
    ModelSnapshot,
)
from physics_engine.motion import InterpolationSemantics, MotionError, Pose
from physics_engine.planned_motion import (
    MotionParameterization,
    MotionSourceArtifact,
    MotionStateCoordinate,
    MotionTrack,
    PlannedMotion,
    PlannedMotionSample,
    TrackPose,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _estimated() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/model-motion-synthetic",
        method="Synthetic model and planned motion for contract validation.",
    )


def _measured() -> EvidenceRef:
    return EvidenceRef(
        grade="measured",
        evidence_id="evidence/model-motion-measured-parts",
        method="Measured model dimensions; the motion remains a plan.",
        source_sha256="f" * 64,
    )


def _asset(asset_id: str, role: AssetRole, sha256: str, path: str) -> ModelAssetRef:
    return ModelAssetRef(
        asset_id=asset_id,
        role=role,
        path_relative=path,
        sha256=sha256,
        format="stl" if path.endswith(".stl") else "glb",
        units="mm",
        frame_id="frame/model-root",
        component_from_asset=Pose(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        ),
    )


def _model() -> ModelSnapshot:
    identity = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    return ModelSnapshot.create(
        model_id="model/winding-cell-synthetic",
        root_frame_id="frame/model-root",
        producer_id="producer/test-fixture",
        source_manifest_sha256="d" * 64,
        components=(
            ModelComponent(
                component_id="model-component/tension-machine",
                frame_id="frame/tension-machine",
                semantic_role="tension_machine",
                parent_component_id=None,
                parent_from_component=identity,
                visual_asset=_asset(
                    "asset/tension-machine-visual",
                    AssetRole.VISUAL,
                    _SHA_A,
                    "assets/tension-machine.glb",
                ),
                collision_asset=_asset(
                    "asset/tension-machine-collision",
                    AssetRole.COLLISION,
                    _SHA_B,
                    "assets/tension-machine.collision.stl",
                ),
            ),
            ModelComponent(
                component_id="model-component/workpiece",
                frame_id="frame/workpiece",
                semantic_role="workpiece",
                parent_component_id=None,
                parent_from_component=identity,
                visual_asset=_asset(
                    "asset/workpiece-visual",
                    AssetRole.VISUAL,
                    _SHA_B,
                    "assets/workpiece.glb",
                ),
                collision_asset=_asset(
                    "asset/workpiece-collision",
                    AssetRole.COLLISION,
                    _SHA_C,
                    "assets/workpiece.collision.stl",
                ),
            ),
            ModelComponent(
                component_id="model-component/robot-display",
                frame_id="frame/robot-display",
                semantic_role="robot_display",
                parent_component_id=None,
                parent_from_component=identity,
                visual_asset=_asset(
                    "asset/robot-visual",
                    AssetRole.VISUAL,
                    _SHA_C,
                    "assets/robot.glb",
                ),
                collision_asset=None,
            ),
        ),
    )


def _semantics() -> InterpolationSemantics:
    return InterpolationSemantics(
        translation_interpolation="linear",
        rotation_interpolation="slerp",
        rotation_arc="shortest",
        pause_hold="hold_interval_start",
        extrapolation="reject",
    )


def _motion(*, parameterization=MotionParameterization.TIME_S) -> PlannedMotion:
    coordinate_unit = "s" if parameterization is MotionParameterization.TIME_S else "1"
    return PlannedMotion.create(
        motion_id="motion/winding-plan-synthetic",
        producer_id="producer/wii-adapter-fixture",
        root_frame_id="frame/model-root",
        parameterization=parameterization,
        coordinate_unit=coordinate_unit,
        source_artifacts=(
            MotionSourceArtifact(
                "selected_plan", "artifact/wii-plan-fixture", "e" * 64
            ),
        ),
        tracks=(
            MotionTrack(
                track_id="motion-track/workpiece",
                component_id="model-component/workpiece",
                frame_id="frame/workpiece",
                interpolation=_semantics(),
            ),
            MotionTrack(
                track_id="motion-track/process-frame",
                component_id=None,
                frame_id="frame/process",
                interpolation=_semantics(),
            ),
            MotionTrack(
                track_id="motion-track/robot-display",
                component_id="model-component/robot-display",
                frame_id="frame/robot-display",
                interpolation=_semantics(),
            ),
        ),
        state_coordinates=(
            MotionStateCoordinate("state-coordinate/a1", "deg"),
            MotionStateCoordinate("state-coordinate/e1", "deg"),
        ),
        samples=(
            PlannedMotionSample(
                sample_id="motion-sample/000000",
                coordinate=0.0,
                path_progress=0.0,
                track_poses=(
                    TrackPose(
                        "motion-track/workpiece",
                        Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                    ),
                    TrackPose(
                        "motion-track/process-frame",
                        Pose((10.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                    ),
                    TrackPose(
                        "motion-track/robot-display",
                        Pose((20.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                    ),
                ),
                source_state_values=(0.0, 0.0),
                material_feed_length_mm=0.0,
                stage_id="laydown",
            ),
            PlannedMotionSample(
                sample_id="motion-sample/000001",
                coordinate=1.0,
                path_progress=1.0,
                track_poses=(
                    TrackPose(
                        "motion-track/workpiece",
                        Pose((2.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
                    ),
                    TrackPose(
                        "motion-track/process-frame",
                        Pose((12.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
                    ),
                    TrackPose(
                        "motion-track/robot-display",
                        Pose((22.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
                    ),
                ),
                source_state_values=(5.0, 360.0),
                material_feed_length_mm=100.0,
                stage_id="laydown",
            ),
        ),
    )


def _relation(model: ModelSnapshot, motion: PlannedMotion) -> ModelPhysicsRelation:
    return ModelPhysicsRelation.create(
        relation_id="model-physics/winding-cell-synthetic",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(
            PhysicsBodyBinding(
                body_id="body/tension-machine",
                component_id="model-component/tension-machine",
                behavior=BodyBehavior.STATIC,
                geometry_source=GeometrySource.COLLISION_ASSET,
                motion_track_id=None,
                analytic_shape_id=None,
                material_record_id=None,
                mass_properties_id=None,
            ),
            PhysicsBodyBinding(
                body_id="body/workpiece",
                component_id="model-component/workpiece",
                behavior=BodyBehavior.KINEMATIC,
                geometry_source=GeometrySource.COLLISION_ASSET,
                motion_track_id="motion-track/workpiece",
                analytic_shape_id=None,
                material_record_id=None,
                mass_properties_id=None,
            ),
        ),
        virtual_frame_bindings=(
            VirtualFrameBinding(
                virtual_frame_id="virtual-frame/process",
                role="process_frame",
                motion_track_id="motion-track/process-frame",
            ),
        ),
        excluded_component_ids=("model-component/robot-display",),
        excluded_motion_track_ids=("motion-track/robot-display",),
    )


def _package(*, parameterization=MotionParameterization.TIME_S):
    model = _model()
    motion = _motion(parameterization=parameterization)
    relation = _relation(model, motion)
    return PhysicsModelMotionInput.create(
        input_id="physics-input/winding-cell-synthetic",
        model=model,
        motion=motion,
        relation=relation,
        evidence=_estimated(),
    )


def test_model_motion_and_physics_relation_round_trip_strictly():
    package = _package()
    payload = json.dumps(package.to_document(), ensure_ascii=False).encode()
    assert load_physics_model_motion_input(payload) == package
    assert package.qualification == "hypothesis_only"
    assert "gcw" not in json.dumps(package.to_document(), sort_keys=True).lower()


def test_measured_parts_cannot_promote_planned_motion_to_calibrated_physics():
    package = _package()
    package = PhysicsModelMotionInput.create(
        input_id=package.input_id,
        model=package.model,
        motion=package.motion,
        relation=package.relation,
        evidence=_measured(),
    )
    assert package.qualification == "hypothesis_only"


def test_time_parameterized_track_becomes_a_replayable_motion_source():
    package = _package()
    timeline = package.motion.as_time_source("motion-track/workpiece")
    pose = timeline.pose_at(0.5)
    assert pose.translation_mm == pytest.approx((1.0, 0.0, 0.0))
    assert pose.rotation_xyzw == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))
    assert timeline.is_replayable() is True


def test_planning_scale_preserves_states_but_cannot_masquerade_as_seconds():
    package = _package(parameterization=MotionParameterization.PLANNING_SCALE)
    assert package.motion.samples[-1].source_state_values == (5.0, 360.0)
    assert package.motion.samples[-1].material_feed_length_mm == 100.0
    with pytest.raises(MotionError, match="physical time"):
        package.motion.as_time_source("motion-track/workpiece")


def test_every_model_component_and_motion_track_is_explicitly_accounted_for():
    package = _package()
    assert {binding.component_id for binding in package.relation.body_bindings} == {
        "model-component/tension-machine",
        "model-component/workpiece",
    }
    assert package.relation.excluded_component_ids == ("model-component/robot-display",)
    assert package.relation.excluded_motion_track_ids == (
        "motion-track/robot-display",
    )
    assert package.relation.virtual_frame_bindings[0].role == "process_frame"


# --- 必须红 ---------------------------------------------------------------


def test_red_visual_asset_cannot_silently_become_collision_geometry():
    model = _model()
    motion = _motion()
    bad_component = ModelComponent(
        component_id="model-component/visual-only-physical",
        frame_id="frame/visual-only-physical",
        semantic_role="workpiece",
        parent_component_id=None,
        parent_from_component=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        visual_asset=_asset(
            "asset/visual-only",
            AssetRole.VISUAL,
            _SHA_A,
            "assets/visual-only.glb",
        ),
        collision_asset=None,
    )
    bad_model = ModelSnapshot.create(
        model_id=model.model_id,
        root_frame_id=model.root_frame_id,
        producer_id=model.producer_id,
        source_manifest_sha256=model.source_manifest_sha256,
        components=(*model.components, bad_component),
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/visual-fallback",
        model_snapshot_sha256=bad_model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(
            PhysicsBodyBinding(
                body_id="body/visual-fallback",
                component_id=bad_component.component_id,
                behavior=BodyBehavior.STATIC,
                geometry_source=GeometrySource.COLLISION_ASSET,
                motion_track_id=None,
                analytic_shape_id=None,
                material_record_id=None,
                mass_properties_id=None,
            ),
        ),
        virtual_frame_bindings=(),
        excluded_component_ids=tuple(
            component.component_id for component in model.components
        ),
        excluded_motion_track_ids=tuple(track.track_id for track in motion.tracks),
    )
    with pytest.raises(ModelPhysicsError, match="collision asset"):
        PhysicsModelMotionInput.create(
            input_id="physics-input/visual-fallback",
            model=bad_model,
            motion=motion,
            relation=relation,
            evidence=_estimated(),
        )


def test_red_kinematic_and_dynamic_state_ownership_cannot_be_mixed():
    with pytest.raises(ModelPhysicsError, match="dynamic"):
        PhysicsBodyBinding(
            body_id="body/double-owned",
            component_id="model-component/workpiece",
            behavior=BodyBehavior.DYNAMIC,
            geometry_source=GeometrySource.COLLISION_ASSET,
            motion_track_id="motion-track/workpiece",
            analytic_shape_id=None,
            material_record_id="material/steel",
            mass_properties_id="mass-properties/workpiece",
            material_record_sha256="a" * 64,
            mass_properties_sha256="b" * 64,
        )


def test_dynamic_body_requires_an_explicit_com_state_frame_and_initial_rates():
    initial = DynamicBodyInitialState(
        state_frame=DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES,
        centre_of_mass_velocity_mm_per_s=(1.0, 2.0, 3.0),
        angular_velocity_body_rad_per_s=(0.1, 0.2, 0.3),
    )
    binding = PhysicsBodyBinding(
        body_id="body/dynamic-workpiece",
        component_id="model-component/workpiece",
        behavior=BodyBehavior.DYNAMIC,
        geometry_source=GeometrySource.COLLISION_ASSET,
        motion_track_id=None,
        analytic_shape_id=None,
        material_record_id="material/steel",
        mass_properties_id="mass-properties/workpiece",
        dynamic_initial_state=initial,
        material_record_sha256="a" * 64,
        mass_properties_sha256="b" * 64,
    )
    assert binding.dynamic_initial_state == initial


def test_red_dynamic_body_without_initial_state_is_rejected():
    with pytest.raises(ModelPhysicsError, match="dynamic body requires.*initial state"):
        PhysicsBodyBinding(
            body_id="body/dynamic-workpiece",
            component_id="model-component/workpiece",
            behavior=BodyBehavior.DYNAMIC,
            geometry_source=GeometrySource.COLLISION_ASSET,
            motion_track_id=None,
            analytic_shape_id=None,
            material_record_id="material/steel",
            mass_properties_id="mass-properties/workpiece",
            material_record_sha256="a" * 64,
            mass_properties_sha256="b" * 64,
        )


def test_red_static_or_kinematic_body_cannot_carry_dynamic_initial_state():
    initial = DynamicBodyInitialState(
        state_frame=DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES,
        centre_of_mass_velocity_mm_per_s=(0.0, 0.0, 0.0),
        angular_velocity_body_rad_per_s=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ModelPhysicsError, match="only valid for a dynamic"):
        PhysicsBodyBinding(
            body_id="body/static-with-state",
            component_id="model-component/workpiece",
            behavior=BodyBehavior.STATIC,
            geometry_source=GeometrySource.COLLISION_ASSET,
            motion_track_id=None,
            analytic_shape_id=None,
            material_record_id=None,
            mass_properties_id=None,
            dynamic_initial_state=initial,
        )


def test_red_unaccounted_model_component_is_rejected():
    model = _model()
    motion = _motion()
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/unaccounted",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(_relation(model, motion).body_bindings[0],),
        virtual_frame_bindings=_relation(model, motion).virtual_frame_bindings,
        excluded_component_ids=("model-component/robot-display",),
        excluded_motion_track_ids=("motion-track/robot-display",),
    )
    with pytest.raises(ModelPhysicsError, match="unaccounted model components"):
        PhysicsModelMotionInput.create(
            input_id="physics-input/unaccounted",
            model=model,
            motion=motion,
            relation=relation,
            evidence=_estimated(),
        )


def test_red_unaccounted_motion_track_is_rejected():
    model = _model()
    motion = _motion()
    base = _relation(model, motion)
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/unaccounted-track",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=base.body_bindings,
        virtual_frame_bindings=(),
        excluded_component_ids=base.excluded_component_ids,
        excluded_motion_track_ids=base.excluded_motion_track_ids,
    )
    with pytest.raises(ModelPhysicsError, match="unaccounted motion tracks"):
        PhysicsModelMotionInput.create(
            input_id="physics-input/unaccounted-track",
            model=model,
            motion=motion,
            relation=relation,
            evidence=_estimated(),
        )


def test_red_component_motion_track_frame_mismatch_is_rejected():
    model = _model()
    motion = _motion()
    components = tuple(
        replace(component, frame_id="frame/workpiece-other")
        if component.component_id == "model-component/workpiece"
        else component
        for component in model.components
    )
    bad_model = ModelSnapshot.create(
        model_id=model.model_id,
        root_frame_id=model.root_frame_id,
        producer_id=model.producer_id,
        source_manifest_sha256=model.source_manifest_sha256,
        components=components,
    )
    relation = _relation(bad_model, motion)
    with pytest.raises(ModelPhysicsError, match="frame.*differs from component frame"):
        PhysicsModelMotionInput.create(
            input_id="physics-input/frame-mismatch",
            model=bad_model,
            motion=motion,
            relation=relation,
            evidence=_estimated(),
        )


def test_red_motion_track_cannot_be_both_physics_owned_and_excluded():
    model = _model()
    motion = _motion()
    base = _relation(model, motion)
    with pytest.raises(ModelPhysicsError, match="both virtual-physics-owned and excluded"):
        ModelPhysicsRelation.create(
            relation_id="model-physics/double-owned-track",
            model_snapshot_sha256=model.content_sha256,
            motion_plan_sha256=motion.content_sha256,
            body_bindings=base.body_bindings,
            virtual_frame_bindings=base.virtual_frame_bindings,
            excluded_component_ids=base.excluded_component_ids,
            excluded_motion_track_ids=(
                "motion-track/workpiece",
                "motion-track/robot-display",
            ),
        )


def test_red_tampered_binding_is_rejected_even_after_rehashing_the_outer_document():
    package = _package()
    document = package.to_document()
    document["relation"]["excluded_component_ids"] = []
    relation_input = dict(document["relation"])
    relation_input.pop("content_sha256")
    document["relation"]["content_sha256"] = canonical_sha256(
        relation_input, PHYSICS_MODEL_MOTION_CANONICAL_PROFILE
    )
    outer = dict(document)
    outer.pop("content_sha256")
    document["content_sha256"] = canonical_sha256(
        outer, PHYSICS_MODEL_MOTION_CANONICAL_PROFILE
    )
    with pytest.raises(ModelPhysicsError, match="unaccounted model components"):
        load_physics_model_motion_input(json.dumps(document).encode())
