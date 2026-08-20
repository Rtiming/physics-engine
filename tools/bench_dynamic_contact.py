"""Fixed P3-M3 model-scene fixture; construction is excluded from timed integration."""

from __future__ import annotations

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
from physics_engine.model_scene import (
    PreparedModelScene,
    SceneInteractionPlan,
    assemble_model_physics_scene,
)
from physics_engine.model_snapshot import ModelComponent, ModelSnapshot
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
    AnalyticCollisionRecord,
    MassPropertiesRecord,
    SceneResourceCatalog,
)
from physics_engine.shapes import CollisionShape, Shape, Sphere

IDENTITY = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
BODY_A = "body/dynamic-a"
BODY_B = "body/dynamic-b"


def _evidence(name: str) -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id=f"evidence/dynamic-contact-{name}",
        method="Synthetic P3-M3 two-body contact fixture.",
    )


def _material(name: str) -> MaterialRecord:
    return MaterialRecord(
        material_id=f"material/{name}",
        applicable_domains=("mechanics",),
        properties=(
            MaterialProperty(
                "density_kg_m3",
                1000.0,
                ("mechanics",),
                _evidence(f"{name}-material"),
            ),
        ),
    ).sealed()


def build_dynamic_sphere_pair_fixture(
    *,
    centroid_y_mm: float,
    shape_b: Shape | None = None,
) -> PreparedModelScene:
    """Build two one-kilogram analytic bodies with one explicit candidate pair."""

    components = tuple(
        ModelComponent(
            component_id=f"model-component/dynamic-{name}",
            frame_id=f"frame/dynamic-{name}",
            semantic_role="workpiece",
            parent_component_id=None,
            parent_from_component=Pose((x_mm, 0.0, 0.0), IDENTITY.rotation_xyzw),
            visual_asset=None,
            collision_asset=None,
        )
        for name, x_mm in (("a", -9.5), ("b", 9.5))
    )
    model = ModelSnapshot.create(
        model_id="model/dynamic-contact-pair",
        root_frame_id="frame/root",
        producer_id="producer/test-fixture",
        source_manifest_sha256="a" * 64,
        components=components,
    )
    semantics = InterpolationSemantics(
        "linear", "hold_previous", "not_applicable", "hold_interval_start", "reject"
    )
    process_track = MotionTrack(
        "motion-track/process", None, "frame/process", semantics
    )
    motion = PlannedMotion.create(
        motion_id="motion/dynamic-contact-pair",
        producer_id="producer/test-fixture",
        root_frame_id="frame/root",
        parameterization=MotionParameterization.TIME_S,
        coordinate_unit="s",
        source_artifacts=(
            MotionSourceArtifact("selected_plan", "artifact/dynamic-contact", "b" * 64),
        ),
        tracks=(process_track,),
        state_coordinates=(),
        samples=tuple(
            PlannedMotionSample(
                sample_id=f"motion-sample/dynamic-contact-{index}",
                coordinate=float(index),
                path_progress=float(index),
                track_poses=(TrackPose(process_track.track_id, IDENTITY),),
                source_state_values=(),
                material_feed_length_mm=0.0,
                stage_id="contact",
            )
            for index in range(2)
        ),
    )
    materials = (_material("dynamic-a"), _material("dynamic-b"))
    masses = tuple(
        MassPropertiesRecord.create(
            mass_properties_id=f"mass-properties/dynamic-{name}",
            geometry_resource_id=f"shape/dynamic-{name}",
            expressed_in_frame_id=f"frame/shape-dynamic-{name}",
            properties=MassProperties(
                volume_mm3=4000.0,
                centroid_mm=(0.0, centroid_y_mm, 0.0),
                mass_kg=1.0,
                inertia_about_centroid_kg_mm2=(
                    (50.0, 0.0, 0.0),
                    (0.0, 50.0, 0.0),
                    (0.0, 0.0, 50.0),
                ),
            ),
            evidence=_evidence(f"{name}-mass"),
        )
        for name in ("a", "b")
    )
    bindings = tuple(
        PhysicsBodyBinding(
            body_id=f"body/dynamic-{name}",
            component_id=f"model-component/dynamic-{name}",
            behavior=BodyBehavior.DYNAMIC,
            geometry_source=GeometrySource.ANALYTIC_SHAPE,
            motion_track_id=None,
            analytic_shape_id=f"shape/dynamic-{name}",
            material_record_id=materials[index].material_id,
            mass_properties_id=masses[index].mass_properties_id,
            dynamic_initial_state=DynamicBodyInitialState(
                state_frame=DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES,
                centre_of_mass_velocity_mm_per_s=(0.0, 0.0, 0.0),
                angular_velocity_body_rad_per_s=(0.0, 0.0, 0.0),
            ),
            material_record_sha256=materials[index].content_sha256,
            mass_properties_sha256=masses[index].content_sha256,
        )
        for index, name in enumerate(("a", "b"))
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/dynamic-contact-pair",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=bindings,
        virtual_frame_bindings=(
            VirtualFrameBinding(
                "virtual-frame/process", "process_frame", process_track.track_id
            ),
        ),
        excluded_component_ids=(),
        excluded_motion_track_ids=(),
    )
    package = PhysicsModelMotionInput.create(
        input_id="physics-input/dynamic-contact-pair",
        model=model,
        motion=motion,
        relation=relation,
        evidence=_evidence("package"),
    )
    shapes = (
        AnalyticCollisionRecord(
            "shape/dynamic-a",
            "frame/shape-dynamic-a",
            CollisionShape(Sphere(10.0), "fitted"),
            IDENTITY,
        ),
        AnalyticCollisionRecord(
            "shape/dynamic-b",
            "frame/shape-dynamic-b",
            CollisionShape(shape_b or Sphere(10.0), "fitted"),
            IDENTITY,
        ),
    )
    resources = SceneResourceCatalog(
        collision_assets=(),
        analytic_shapes=shapes,
        materials=materials,
        mass_property_records=masses,
    )
    return assemble_model_physics_scene(
        package,
        resources,
        SceneInteractionPlan(
            scene_id="scene/dynamic-contact-pair",
            contact_pairs=((BODY_A, BODY_B),),
            allowed_pairs=(),
        ),
    )


__all__ = [
    "BODY_A",
    "BODY_B",
    "build_dynamic_sphere_pair_fixture",
]
