"""dynamic体的geometry frame、质心状态与惯量闭环。

冻结约定只有一条：``RIGID_BODY_LAYOUT``的原点是质心，body轴与geometry resource frame
平行；``MassProperties.centroid_mm``和绕质心惯量均在该geometry frame表达。由此几何初始
位姿可唯一推导COM状态，COM状态也可唯一反算几何位姿。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.materials import MaterialRecord
from physics_engine.model_physics import (
    BodyBehavior,
    DynamicStateFrame,
    PhysicsBodyBinding,
)
from physics_engine.motion import Pose
from physics_engine.pose_math import compose_pose
from physics_engine.rigidbody import (
    RigidBodyError,
    RigidBodyInertia,
    attitude_xyzw,
    centre_of_mass_position_mm,
    make_state,
)
from physics_engine.scene_resources import MassPropertiesRecord
from physics_engine.shapes import PosedBody, ShapeError, SimBody
from physics_engine.state import State

IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


class DynamicBodyError(ValueError):
    """dynamic资源、状态frame、质心或几何位姿没有闭合。"""


def _finite_vector(value: object, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise DynamicBodyError(f"{name} must be a numeric 3-tuple")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(item) for item in result):
        raise DynamicBodyError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DynamicBodyRuntime:
    """一个dynamic物理体的初态、惯量和几何相对COM变换。"""

    binding: PhysicsBodyBinding
    geometry_resource_id: str
    geometry_frame_id: str
    material: MaterialRecord
    mass_record: MassPropertiesRecord
    inertia: RigidBodyInertia
    geometry_from_body: Pose
    body_from_geometry: Pose
    initial_state: State

    @property
    def body_id(self) -> str:
        return self.binding.body_id

    def posed_geometry(self, state: State, body: SimBody) -> PosedBody:
        """从COM状态反算geometry resource frame的世界位姿。"""

        if not isinstance(state, State):
            raise DynamicBodyError("dynamic pose requires a State")
        if not isinstance(body, SimBody) or body.body_id != self.body_id:
            raise DynamicBodyError(
                f"dynamic geometry body must be {self.body_id!r}, got {body!r}"
            )
        try:
            root_from_body = Pose(
                centre_of_mass_position_mm(state), attitude_xyzw(state)
            )
            root_from_geometry = compose_pose(root_from_body, self.body_from_geometry)
            return PosedBody(
                body=body,
                translation_mm=root_from_geometry.translation_mm,
                rotation_xyzw=root_from_geometry.rotation_xyzw,
            )
        except (RigidBodyError, ShapeError, ValueError) as error:
            raise DynamicBodyError(
                f"dynamic body {self.body_id} state/layout cannot drive geometry: {error}"
            ) from error


def prepare_dynamic_body_runtime(
    *,
    binding: PhysicsBodyBinding,
    root_from_geometry: Pose,
    geometry_resource_id: str,
    geometry_frame_id: str,
    material: MaterialRecord,
    mass_record: MassPropertiesRecord,
) -> DynamicBodyRuntime:
    """由模型参考几何位姿和质量属性构造质心初态。"""

    if not isinstance(binding, PhysicsBodyBinding) or binding.behavior is not BodyBehavior.DYNAMIC:
        raise DynamicBodyError("binding must describe a dynamic body")
    if binding.dynamic_initial_state is None:
        raise DynamicBodyError("dynamic binding has no initial state")
    if (
        binding.dynamic_initial_state.state_frame
        is not DynamicStateFrame.CENTRE_OF_MASS_GEOMETRY_AXES
    ):
        raise DynamicBodyError("unsupported dynamic state frame")
    if not isinstance(root_from_geometry, Pose):
        raise DynamicBodyError("root_from_geometry must be a Pose")
    if not isinstance(material, MaterialRecord) or material.content_sha256 is None:
        raise DynamicBodyError("dynamic material must be a sealed MaterialRecord")
    if material.material_id != binding.material_record_id:
        raise DynamicBodyError(
            f"dynamic material {material.material_id!r} differs from binding "
            f"{binding.material_record_id!r}"
        )
    if material.content_sha256 != binding.material_record_sha256:
        raise DynamicBodyError(
            f"dynamic material sha256 {material.content_sha256!r} differs from binding "
            f"{binding.material_record_sha256!r}"
        )
    if "mechanics" not in material.applicable_domains:
        raise DynamicBodyError("dynamic material must declare mechanics domain")
    if not isinstance(mass_record, MassPropertiesRecord):
        raise DynamicBodyError("mass_record must be MassPropertiesRecord")
    if mass_record.mass_properties_id != binding.mass_properties_id:
        raise DynamicBodyError(
            f"mass properties {mass_record.mass_properties_id!r} differ from binding "
            f"{binding.mass_properties_id!r}"
        )
    if mass_record.content_sha256 != binding.mass_properties_sha256:
        raise DynamicBodyError(
            f"mass properties sha256 {mass_record.content_sha256!r} differs from binding "
            f"{binding.mass_properties_sha256!r}"
        )
    if mass_record.geometry_resource_id != geometry_resource_id:
        raise DynamicBodyError(
            f"mass properties name geometry resource "
            f"{mass_record.geometry_resource_id!r}, got {geometry_resource_id!r}"
        )
    if mass_record.expressed_in_frame_id != geometry_frame_id:
        raise DynamicBodyError(
            f"mass properties geometry frame {mass_record.expressed_in_frame_id!r} "
            f"differs from {geometry_frame_id!r}"
        )

    properties = mass_record.properties
    if not math.isfinite(properties.volume_mm3) or properties.volume_mm3 <= 0.0:
        raise DynamicBodyError("mass properties volume_mm3 must be positive and finite")
    centroid = _finite_vector(properties.centroid_mm, "centroid_mm")
    try:
        inertia = RigidBodyInertia(
            mass_kg=properties.mass_kg,
            inertia_body_kg_mm2=properties.inertia_about_centroid_kg_mm2,
        )
    except RigidBodyError as error:
        raise DynamicBodyError(f"invalid inertia about centroid: {error}") from error

    geometry_from_body = Pose(centroid, IDENTITY_ROTATION)
    body_from_geometry = Pose(
        (-centroid[0], -centroid[1], -centroid[2]), IDENTITY_ROTATION
    )
    root_from_body = compose_pose(root_from_geometry, geometry_from_body)
    initial = binding.dynamic_initial_state
    state = make_state(
        position_mm=root_from_body.translation_mm,
        velocity_mm_per_s=initial.centre_of_mass_velocity_mm_per_s,
        angular_velocity_rad_per_s=initial.angular_velocity_body_rad_per_s,
        attitude_xyzw=root_from_body.rotation_xyzw,
    )
    return DynamicBodyRuntime(
        binding=binding,
        geometry_resource_id=geometry_resource_id,
        geometry_frame_id=geometry_frame_id,
        material=material,
        mass_record=mass_record,
        inertia=inertia,
        geometry_from_body=geometry_from_body,
        body_from_geometry=body_from_geometry,
        initial_state=state,
    )


__all__ = [
    "DynamicBodyError",
    "DynamicBodyRuntime",
    "prepare_dynamic_body_runtime",
]
