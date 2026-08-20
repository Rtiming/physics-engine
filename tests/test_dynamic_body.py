"""dynamic状态frame、质心和几何位姿必须形成可逆闭环。"""

from __future__ import annotations

import math

import pytest

from physics_engine.dynamic_body import DynamicBodyError, prepare_dynamic_body_runtime
from physics_engine.geometry import MassProperties
from physics_engine.materials import EvidenceRef, MaterialProperty, MaterialRecord
from physics_engine.model_physics import (
    BodyBehavior,
    DynamicBodyInitialState,
    DynamicStateFrame,
    GeometrySource,
    PhysicsBodyBinding,
)
from physics_engine.motion import Pose
from physics_engine.rigidbody import make_state
from physics_engine.scene_resources import MassPropertiesRecord
from physics_engine.shapes import CollisionShape, SimBody, Sphere
from physics_engine.state import State, StateField, StateLayout

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _binding() -> PhysicsBodyBinding:
    material = _material()
    mass = _mass_record()
    return PhysicsBodyBinding(
        body_id="body/workpiece",
        component_id="model-component/workpiece",
        behavior=BodyBehavior.DYNAMIC,
        geometry_source=GeometrySource.COLLISION_ASSET,
        motion_track_id=None,
        analytic_shape_id=None,
        material_record_id="material/workpiece",
        mass_properties_id="mass-properties/workpiece",
        dynamic_initial_state=DynamicBodyInitialState(
            state_frame=DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES,
            centre_of_mass_velocity_mm_per_s=(1.0, 2.0, 3.0),
            angular_velocity_body_rad_per_s=(0.0, 0.0, math.pi),
        ),
        material_record_sha256=material.content_sha256,
        mass_properties_sha256=mass.content_sha256,
    )


def _material() -> MaterialRecord:
    return MaterialRecord(
        material_id="material/workpiece",
        applicable_domains=("mechanics",),
        properties=(
            MaterialProperty(
                name="density_kg_m3",
                value=1000.0,
                domains=("mechanics",),
                evidence=EvidenceRef(
                    grade="estimated",
                    evidence_id="evidence/dynamic-body-material",
                    method="Synthetic dynamic-body fixture.",
                ),
            ),
        ),
    ).sealed()


def _mass_record() -> MassPropertiesRecord:
    return MassPropertiesRecord.create(
        mass_properties_id="mass-properties/workpiece",
        geometry_resource_id="asset/workpiece-collision",
        expressed_in_frame_id="frame/workpiece-asset",
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
            "evidence/dynamic-body-mass",
            "Synthetic dynamic-body mass properties.",
        ),
    )


def _runtime():
    return prepare_dynamic_body_runtime(
        binding=_binding(),
        root_from_geometry=Pose((10.0, 0.0, 0.0), IDENTITY),
        geometry_resource_id="asset/workpiece-collision",
        geometry_frame_id="frame/workpiece-asset",
        material=_material(),
        mass_record=_mass_record(),
    )


def _body() -> SimBody:
    return SimBody(
        body_id="body/workpiece",
        collision=CollisionShape(Sphere(1.0), "fitted"),
        mass_kg=1.0,
    )


def test_initial_geometry_pose_maps_to_com_state_and_back_exactly():
    runtime = _runtime()
    assert runtime.initial_state.block("centre_of_mass_position_mm") == (
        12.0,
        0.0,
        0.0,
    )
    assert runtime.initial_state.block("centre_of_mass_velocity_mm_per_s") == (
        1.0,
        2.0,
        3.0,
    )
    assert runtime.initial_state.block("angular_velocity_body_rad_per_s") == (
        0.0,
        0.0,
        math.pi,
    )
    posed = runtime.posed_geometry(runtime.initial_state, _body())
    assert posed.translation_mm == (10.0, 0.0, 0.0)
    assert posed.rotation_xyzw == IDENTITY


def test_geometry_origin_rotates_about_the_com_not_about_its_own_origin():
    runtime = _runtime()
    state = make_state(
        position_mm=(12.0, 0.0, 0.0),
        attitude_xyzw=(0.0, 0.0, 1.0, 0.0),
    )
    posed = runtime.posed_geometry(state, _body())
    assert posed.translation_mm == pytest.approx((14.0, 0.0, 0.0))
    assert posed.rotation_xyzw == (0.0, 0.0, 1.0, 0.0)


def test_runtime_uses_mass_and_inertia_about_com_in_geometry_axes():
    runtime = _runtime()
    assert runtime.inertia.mass_kg == 1.0
    assert runtime.inertia.inertia_body_kg_mm2 == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 2.0),
    )
    assert runtime.material.material_id == "material/workpiece"


def test_red_mass_properties_must_name_the_same_geometry_resource_and_frame():
    with pytest.raises(DynamicBodyError, match="geometry resource"):
        prepare_dynamic_body_runtime(
            binding=_binding(),
            root_from_geometry=Pose((10.0, 0.0, 0.0), IDENTITY),
            geometry_resource_id="asset/another",
            geometry_frame_id="frame/workpiece-asset",
            material=_material(),
            mass_record=_mass_record(),
        )
    with pytest.raises(DynamicBodyError, match="geometry frame"):
        prepare_dynamic_body_runtime(
            binding=_binding(),
            root_from_geometry=Pose((10.0, 0.0, 0.0), IDENTITY),
            geometry_resource_id="asset/workpiece-collision",
            geometry_frame_id="frame/another",
            material=_material(),
            mass_record=_mass_record(),
        )


def test_red_material_and_mass_ids_must_match_the_binding():
    wrong_material = MaterialRecord(
        material_id="material/another",
        applicable_domains=_material().applicable_domains,
        properties=_material().properties,
    ).sealed()
    with pytest.raises(DynamicBodyError, match="material"):
        prepare_dynamic_body_runtime(
            binding=_binding(),
            root_from_geometry=Pose((10.0, 0.0, 0.0), IDENTITY),
            geometry_resource_id="asset/workpiece-collision",
            geometry_frame_id="frame/workpiece-asset",
            material=wrong_material,
            mass_record=_mass_record(),
        )


def test_red_same_material_id_with_changed_content_is_rejected():
    changed = MaterialRecord(
        material_id="material/workpiece",
        applicable_domains=("mechanics",),
        properties=(
            MaterialProperty(
                "density_kg_m3",
                900.0,
                ("mechanics",),
                EvidenceRef(
                    "estimated",
                    "evidence/dynamic-body-material-changed",
                    "Deliberately changed material content.",
                ),
            ),
        ),
    ).sealed()
    with pytest.raises(DynamicBodyError, match="material sha256"):
        prepare_dynamic_body_runtime(
            binding=_binding(),
            root_from_geometry=Pose((10.0, 0.0, 0.0), IDENTITY),
            geometry_resource_id="asset/workpiece-collision",
            geometry_frame_id="frame/workpiece-asset",
            material=changed,
            mass_record=_mass_record(),
        )


def test_red_state_with_another_layout_cannot_drive_dynamic_geometry():
    runtime = _runtime()
    wrong = State(
        layout=StateLayout(
            layout_id="layout/wrong",
            fields=(StateField("position_mm", 3),),
        ),
        vector=(0.0, 0.0, 0.0),
    )
    with pytest.raises(DynamicBodyError, match="layout"):
        runtime.posed_geometry(wrong, _body())
