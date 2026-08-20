"""模型、规划运动与虚拟物理体之间的显式关系。

这是physics-engine自己的上游合同，不是WII或GCW格式的镜像。适配器负责把外部模型和规划
归一化为本合同；内核只看模型快照、规划运动和三类关系：static/kinematic/dynamic物理体、
虚拟工艺frame、明确排除的显示/无关组件。

一个组件不能既由运动计划驱动又由求解器积分。visual资产不能静默替代collision资产。
每个模型组件和每条motion track都必须被绑定或明确排除，防止“文件存在但物理根本没读”。

dynamic还必须声明COM+geometry轴state frame、世界系初始线速度、体系初始角速度，
并以ID+SHA锁定材料和质量属性。position/attitude不重复存，由模型参考几何位姿与质心推导。
"""

from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any

from physics_engine.canonical import WDS_PROFILE, canonical_sha256, strict_loads
from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    PHYSICS_MODEL_MOTION_INPUT_FACET,
    PHYSICS_MODEL_MOTION_INPUT_VERSION,
)
from physics_engine.identity import IdentityError, parse_namespace_id
from physics_engine.materials import EvidenceRef
from physics_engine.model_snapshot import (
    ModelSnapshot,
    ModelSnapshotError,
    model_snapshot_from_document,
)
from physics_engine.planned_motion import (
    PlannedMotion,
    PlannedMotionError,
    planned_motion_from_document,
)

PHYSICS_MODEL_MOTION_CANONICAL_PROFILE = WDS_PROFILE


class ModelPhysicsError(ValueError):
    """模型、运动、虚拟物理所有权或字节闭包错误。"""


class BodyBehavior(enum.StrEnum):
    STATIC = "static"
    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"


class GeometrySource(enum.StrEnum):
    COLLISION_ASSET = "collision_asset"
    ANALYTIC_SHAPE = "analytic_shape"


class DynamicStateFrame(enum.StrEnum):
    """dynamic状态的原点与轴约定；当前只冻结一条。"""

    CENTRE_OF_MASS_GEOMETRY_AXES = "centre_of_mass_geometry_axes"


def _require_namespace(value: object, namespace: str, name: str) -> str:
    if not isinstance(value, str):
        raise ModelPhysicsError(f"{name} must be a string: {value!r}")
    try:
        parsed, _ = parse_namespace_id(value)
    except IdentityError as error:
        raise ModelPhysicsError(f"{name} is not a valid namespace id: {error}") from error
    if parsed != namespace:
        raise ModelPhysicsError(f"{name} must live in {namespace!r}: {value!r}")
    return value


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelPhysicsError(f"{name} must be a nonempty string")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
        raise ModelPhysicsError(f"{name} must use lowercase identifier characters: {value!r}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelPhysicsError(f"{name} must be 64 lowercase hex characters")
    return value


def _require_vector3(value: object, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ModelPhysicsError(f"{name} must be a numeric 3-tuple")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(item) for item in result):
        raise ModelPhysicsError(f"{name} must be finite")
    return result


def _parse_evidence(value: object) -> EvidenceRef:
    expected = {"grade", "evidence_id", "method", "source_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ModelPhysicsError("evidence fields differ from the contract")
    try:
        return EvidenceRef(
            grade=value["grade"],
            evidence_id=value["evidence_id"],
            method=value["method"],
            source_sha256=value["source_sha256"],
        )
    except ValueError as error:
        raise ModelPhysicsError(f"invalid evidence: {error}") from error


@dataclass(frozen=True)
class DynamicBodyInitialState:
    """dynamic体的初始速率与状态frame声明。

    position和attitude不重复存：它们由模型参考位姿、几何安装变换和质量属性质心
    唯一推导。线速度在root/world轴表达，角速度在geometry/body轴表达，与
    ``rigidbody.RIGID_BODY_LAYOUT``逐字段一致。
    """

    state_frame: DynamicStateFrame
    centre_of_mass_velocity_mm_per_s: tuple[float, float, float]
    angular_velocity_body_rad_per_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.state_frame, DynamicStateFrame):
            raise ModelPhysicsError("state_frame must be DynamicStateFrame")
        _require_vector3(
            self.centre_of_mass_velocity_mm_per_s,
            "centre_of_mass_velocity_mm_per_s",
        )
        _require_vector3(
            self.angular_velocity_body_rad_per_s,
            "angular_velocity_body_rad_per_s",
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "state_frame": self.state_frame.value,
            "centre_of_mass_velocity_mm_per_s": list(
                self.centre_of_mass_velocity_mm_per_s
            ),
            "angular_velocity_body_rad_per_s": list(
                self.angular_velocity_body_rad_per_s
            ),
        }


@dataclass(frozen=True)
class PhysicsBodyBinding:
    body_id: str
    component_id: str
    behavior: BodyBehavior
    geometry_source: GeometrySource
    motion_track_id: str | None
    analytic_shape_id: str | None
    material_record_id: str | None
    mass_properties_id: str | None
    dynamic_initial_state: DynamicBodyInitialState | None = None
    material_record_sha256: str | None = None
    mass_properties_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_namespace(self.body_id, "body", "body_id")
        _require_namespace(self.component_id, "model-component", "component_id")
        if not isinstance(self.behavior, BodyBehavior):
            raise ModelPhysicsError("behavior must be BodyBehavior")
        if not isinstance(self.geometry_source, GeometrySource):
            raise ModelPhysicsError("geometry_source must be GeometrySource")
        if self.motion_track_id is not None:
            _require_namespace(self.motion_track_id, "motion-track", "motion_track_id")
        if self.analytic_shape_id is not None:
            _require_namespace(self.analytic_shape_id, "shape", "analytic_shape_id")
        if self.material_record_id is not None:
            _require_namespace(self.material_record_id, "material", "material_record_id")
        if self.mass_properties_id is not None:
            _require_namespace(
                self.mass_properties_id, "mass-properties", "mass_properties_id"
            )
        if (self.material_record_id is None) != (self.material_record_sha256 is None):
            raise ModelPhysicsError(
                "material_record_id and material_record_sha256 must be present together"
            )
        if (self.mass_properties_id is None) != (self.mass_properties_sha256 is None):
            raise ModelPhysicsError(
                "mass_properties_id and mass_properties_sha256 must be present together"
            )
        if self.material_record_sha256 is not None:
            _require_sha256(self.material_record_sha256, "material_record_sha256")
        if self.mass_properties_sha256 is not None:
            _require_sha256(self.mass_properties_sha256, "mass_properties_sha256")
        if self.behavior is BodyBehavior.KINEMATIC and self.motion_track_id is None:
            raise ModelPhysicsError("kinematic body requires a motion track")
        if self.behavior is BodyBehavior.DYNAMIC and self.motion_track_id is not None:
            raise ModelPhysicsError(
                "dynamic body cannot also have a motion track; state would have two owners"
            )
        if self.behavior is BodyBehavior.STATIC and self.motion_track_id is not None:
            raise ModelPhysicsError("static body cannot have a motion track")
        if self.behavior is BodyBehavior.DYNAMIC and (
            self.material_record_id is None or self.mass_properties_id is None
        ):
            raise ModelPhysicsError(
                "dynamic body requires material_record_id and mass_properties_id"
            )
        if self.behavior is BodyBehavior.DYNAMIC and self.dynamic_initial_state is None:
            raise ModelPhysicsError("dynamic body requires an explicit initial state")
        if self.behavior is not BodyBehavior.DYNAMIC and self.dynamic_initial_state is not None:
            raise ModelPhysicsError("dynamic_initial_state is only valid for a dynamic body")
        if self.dynamic_initial_state is not None and not isinstance(
            self.dynamic_initial_state, DynamicBodyInitialState
        ):
            raise ModelPhysicsError(
                "dynamic_initial_state must be DynamicBodyInitialState or None"
            )
        if self.geometry_source is GeometrySource.COLLISION_ASSET:
            if self.analytic_shape_id is not None:
                raise ModelPhysicsError(
                    "collision_asset geometry must not also name analytic_shape_id"
                )
        elif self.analytic_shape_id is None:
            raise ModelPhysicsError("analytic_shape geometry requires analytic_shape_id")

    def to_document(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "component_id": self.component_id,
            "behavior": self.behavior.value,
            "geometry_source": self.geometry_source.value,
            "motion_track_id": self.motion_track_id,
            "analytic_shape_id": self.analytic_shape_id,
            "material_record_id": self.material_record_id,
            "mass_properties_id": self.mass_properties_id,
            "dynamic_initial_state": (
                None
                if self.dynamic_initial_state is None
                else self.dynamic_initial_state.to_document()
            ),
            "material_record_sha256": self.material_record_sha256,
            "mass_properties_sha256": self.mass_properties_sha256,
        }


@dataclass(frozen=True)
class VirtualFrameBinding:
    virtual_frame_id: str
    role: str
    motion_track_id: str

    def __post_init__(self) -> None:
        _require_namespace(self.virtual_frame_id, "virtual-frame", "virtual_frame_id")
        _require_identifier(self.role, "role")
        _require_namespace(self.motion_track_id, "motion-track", "motion_track_id")

    def to_document(self) -> dict[str, str]:
        return {
            "virtual_frame_id": self.virtual_frame_id,
            "role": self.role,
            "motion_track_id": self.motion_track_id,
        }


@dataclass(frozen=True)
class ModelPhysicsRelation:
    relation_id: str
    model_snapshot_sha256: str
    motion_plan_sha256: str
    body_bindings: tuple[PhysicsBodyBinding, ...]
    virtual_frame_bindings: tuple[VirtualFrameBinding, ...]
    excluded_component_ids: tuple[str, ...]
    excluded_motion_track_ids: tuple[str, ...]
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_namespace(self.relation_id, "model-physics", "relation_id")
        _require_sha256(self.model_snapshot_sha256, "model_snapshot_sha256")
        _require_sha256(self.motion_plan_sha256, "motion_plan_sha256")
        if not all(isinstance(item, PhysicsBodyBinding) for item in self.body_bindings):
            raise ModelPhysicsError("body_bindings must contain PhysicsBodyBinding values")
        if not all(
            isinstance(item, VirtualFrameBinding) for item in self.virtual_frame_bindings
        ):
            raise ModelPhysicsError(
                "virtual_frame_bindings must contain VirtualFrameBinding values"
            )
        body_ids = tuple(item.body_id for item in self.body_bindings)
        component_ids = tuple(item.component_id for item in self.body_bindings)
        virtual_ids = tuple(item.virtual_frame_id for item in self.virtual_frame_bindings)
        for label, values in (
            ("body IDs", body_ids),
            ("physical component IDs", component_ids),
            ("virtual frame IDs", virtual_ids),
            ("excluded component IDs", self.excluded_component_ids),
            ("excluded motion track IDs", self.excluded_motion_track_ids),
        ):
            if len(set(values)) != len(values):
                raise ModelPhysicsError(f"{label} must be unique")
        for component_id in self.excluded_component_ids:
            _require_namespace(component_id, "model-component", "excluded_component_id")
        for track_id in self.excluded_motion_track_ids:
            _require_namespace(track_id, "motion-track", "excluded_motion_track_id")
        if set(component_ids) & set(self.excluded_component_ids):
            raise ModelPhysicsError("a model component cannot be both physical and excluded")
        used_tracks = [
            binding.motion_track_id
            for binding in self.body_bindings
            if binding.motion_track_id is not None
        ] + [binding.motion_track_id for binding in self.virtual_frame_bindings]
        if len(set(used_tracks)) != len(used_tracks):
            raise ModelPhysicsError("a motion track may have only one virtual-physics owner")
        if set(used_tracks) & set(self.excluded_motion_track_ids):
            raise ModelPhysicsError(
                "a motion track cannot be both virtual-physics-owned and excluded"
            )
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
            if self.content_sha256 != self.content_address():
                raise ModelPhysicsError("model-physics relation content_sha256 does not match")

    @classmethod
    def create(
        cls,
        *,
        relation_id: str,
        model_snapshot_sha256: str,
        motion_plan_sha256: str,
        body_bindings: tuple[PhysicsBodyBinding, ...],
        virtual_frame_bindings: tuple[VirtualFrameBinding, ...],
        excluded_component_ids: tuple[str, ...],
        excluded_motion_track_ids: tuple[str, ...],
    ) -> ModelPhysicsRelation:
        return cls(
            relation_id=relation_id,
            model_snapshot_sha256=model_snapshot_sha256,
            motion_plan_sha256=motion_plan_sha256,
            body_bindings=body_bindings,
            virtual_frame_bindings=virtual_frame_bindings,
            excluded_component_ids=excluded_component_ids,
            excluded_motion_track_ids=excluded_motion_track_ids,
        ).sealed()

    def to_document(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "model_snapshot_sha256": self.model_snapshot_sha256,
            "motion_plan_sha256": self.motion_plan_sha256,
            "body_bindings": [item.to_document() for item in self.body_bindings],
            "virtual_frame_bindings": [
                item.to_document() for item in self.virtual_frame_bindings
            ],
            "excluded_component_ids": list(self.excluded_component_ids),
            "excluded_motion_track_ids": list(self.excluded_motion_track_ids),
            "content_sha256": self.content_sha256,
        }

    def content_address(self) -> str:
        document = self.to_document()
        document.pop("content_sha256")
        return canonical_sha256(document, PHYSICS_MODEL_MOTION_CANONICAL_PROFILE)

    def sealed(self) -> ModelPhysicsRelation:
        return replace(self, content_sha256=self.content_address())


@dataclass(frozen=True)
class PhysicsModelMotionInput:
    input_id: str
    model: ModelSnapshot
    motion: PlannedMotion
    relation: ModelPhysicsRelation
    evidence: EvidenceRef
    facet_version: str = PHYSICS_MODEL_MOTION_INPUT_VERSION
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        ENGINE_REGISTRY.assert_reader_compatible(
            PHYSICS_MODEL_MOTION_INPUT_FACET, self.facet_version
        )
        _require_namespace(self.input_id, "physics-input", "input_id")
        if not isinstance(self.model, ModelSnapshot) or self.model.content_sha256 is None:
            raise ModelPhysicsError("model must be a sealed ModelSnapshot")
        if not isinstance(self.motion, PlannedMotion) or self.motion.content_sha256 is None:
            raise ModelPhysicsError("motion must be a sealed PlannedMotion")
        if not isinstance(self.relation, ModelPhysicsRelation) or self.relation.content_sha256 is None:
            raise ModelPhysicsError("relation must be a sealed ModelPhysicsRelation")
        if not isinstance(self.evidence, EvidenceRef):
            raise ModelPhysicsError("evidence must be an EvidenceRef")
        self._validate_relationships()
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
            if self.content_sha256 != self.content_address():
                raise ModelPhysicsError("physics input content_sha256 does not match")

    @property
    def qualification(self) -> str:
        # 这个facet承载的是PlannedMotion。即使模型尺寸或adapter变换已有测量证据，
        # 规划轨迹本身也不是实际运动测量，不能借外层EvidenceRef升级为calibrated_model。
        return "hypothesis_only"

    def _validate_relationships(self) -> None:
        if self.model.root_frame_id != self.motion.root_frame_id:
            raise ModelPhysicsError("model and motion root_frame_id differ")
        if self.relation.model_snapshot_sha256 != self.model.content_sha256:
            raise ModelPhysicsError("relation references another model snapshot")
        if self.relation.motion_plan_sha256 != self.motion.content_sha256:
            raise ModelPhysicsError("relation references another motion plan")
        model_components = {component.component_id: component for component in self.model.components}
        physical_components = {binding.component_id for binding in self.relation.body_bindings}
        excluded_components = set(self.relation.excluded_component_ids)
        unknown_components = (physical_components | excluded_components) - set(model_components)
        if unknown_components:
            raise ModelPhysicsError(f"relation names unknown model components: {sorted(unknown_components)}")
        unaccounted_components = set(model_components) - physical_components - excluded_components
        if unaccounted_components:
            raise ModelPhysicsError(
                f"unaccounted model components: {sorted(unaccounted_components)}"
            )
        motion_tracks = {track.track_id: track for track in self.motion.tracks}
        for track in self.motion.tracks:
            if track.component_id is None:
                continue
            component = model_components.get(track.component_id)
            if component is None:
                raise ModelPhysicsError(
                    f"motion track {track.track_id} names unknown component "
                    f"{track.component_id}"
                )
            if track.frame_id != component.frame_id:
                raise ModelPhysicsError(
                    f"motion track {track.track_id} frame {track.frame_id!r} differs from "
                    f"component frame {component.frame_id!r}"
                )
        excluded_tracks = set(self.relation.excluded_motion_track_ids)
        unknown_tracks = excluded_tracks - set(motion_tracks)
        if unknown_tracks:
            raise ModelPhysicsError(
                f"relation excludes unknown motion tracks: {sorted(unknown_tracks)}"
            )
        for track_id in excluded_tracks:
            component_id = motion_tracks[track_id].component_id
            if component_id is not None and component_id not in excluded_components:
                raise ModelPhysicsError(
                    f"excluded motion track {track_id} belongs to physical component "
                    f"{component_id}"
                )
        used_tracks: set[str] = set()
        for binding in self.relation.body_bindings:
            component = model_components[binding.component_id]
            if binding.geometry_source is GeometrySource.COLLISION_ASSET:
                if component.collision_asset is None:
                    raise ModelPhysicsError(
                        f"component {component.component_id} has no collision asset; "
                        "visual asset fallback is forbidden"
                    )
            if binding.motion_track_id is not None:
                track = motion_tracks.get(binding.motion_track_id)
                if track is None:
                    raise ModelPhysicsError(
                        f"body {binding.body_id} names unknown motion track"
                    )
                if track.component_id != binding.component_id:
                    raise ModelPhysicsError(
                        f"motion track {track.track_id} belongs to {track.component_id!r}, "
                        f"not {binding.component_id!r}"
                    )
                used_tracks.add(track.track_id)
        for binding in self.relation.virtual_frame_bindings:
            track = motion_tracks.get(binding.motion_track_id)
            if track is None:
                raise ModelPhysicsError(
                    f"virtual frame {binding.virtual_frame_id} names unknown motion track"
                )
            if track.component_id is not None:
                raise ModelPhysicsError(
                    f"virtual frame track {track.track_id} is already owned by model component "
                    f"{track.component_id}"
                )
            used_tracks.add(track.track_id)
        unaccounted_tracks = set(motion_tracks) - used_tracks - excluded_tracks
        if unaccounted_tracks:
            raise ModelPhysicsError(f"unaccounted motion tracks: {sorted(unaccounted_tracks)}")

    @classmethod
    def create(
        cls,
        *,
        input_id: str,
        model: ModelSnapshot,
        motion: PlannedMotion,
        relation: ModelPhysicsRelation,
        evidence: EvidenceRef,
    ) -> PhysicsModelMotionInput:
        return cls(
            input_id=input_id,
            model=model,
            motion=motion,
            relation=relation,
            evidence=evidence,
        ).sealed()

    def to_document(self) -> dict[str, Any]:
        return {
            "facet": PHYSICS_MODEL_MOTION_INPUT_FACET,
            "facet_version": self.facet_version,
            "input_id": self.input_id,
            "qualification": self.qualification,
            "model": self.model.to_document(),
            "motion": self.motion.to_document(),
            "relation": self.relation.to_document(),
            "evidence": self.evidence.to_document(),
            "content_sha256": self.content_sha256,
        }

    def content_address(self) -> str:
        document = self.to_document()
        document.pop("content_sha256")
        return canonical_sha256(document, PHYSICS_MODEL_MOTION_CANONICAL_PROFILE)

    def sealed(self) -> PhysicsModelMotionInput:
        return replace(self, content_sha256=self.content_address())


def _binding_from_document(value: object, index: int) -> PhysicsBodyBinding:
    expected = {
        "body_id",
        "component_id",
        "behavior",
        "geometry_source",
        "motion_track_id",
        "analytic_shape_id",
        "material_record_id",
        "mass_properties_id",
        "dynamic_initial_state",
        "material_record_sha256",
        "mass_properties_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ModelPhysicsError(f"body_binding[{index}] fields differ from the contract")
    try:
        behavior = BodyBehavior(value["behavior"])
        geometry = GeometrySource(value["geometry_source"])
    except (TypeError, ValueError) as error:
        raise ModelPhysicsError(f"body_binding[{index}] enum is invalid") from error
    raw_initial = value["dynamic_initial_state"]
    initial = None
    if raw_initial is not None:
        expected_initial = {
            "state_frame",
            "centre_of_mass_velocity_mm_per_s",
            "angular_velocity_body_rad_per_s",
        }
        if not isinstance(raw_initial, dict) or set(raw_initial) != expected_initial:
            raise ModelPhysicsError(
                f"body_binding[{index}].dynamic_initial_state is invalid"
            )
        linear = raw_initial["centre_of_mass_velocity_mm_per_s"]
        angular = raw_initial["angular_velocity_body_rad_per_s"]
        if not isinstance(linear, list) or not isinstance(angular, list):
            raise ModelPhysicsError(
                f"body_binding[{index}] dynamic velocity fields must be arrays"
            )
        try:
            frame = DynamicStateFrame(raw_initial["state_frame"])
        except (TypeError, ValueError) as error:
            raise ModelPhysicsError(
                f"body_binding[{index}] dynamic state_frame is invalid"
            ) from error
        initial = DynamicBodyInitialState(
            state_frame=frame,
            centre_of_mass_velocity_mm_per_s=tuple(linear),  # type: ignore[arg-type]
            angular_velocity_body_rad_per_s=tuple(angular),  # type: ignore[arg-type]
        )
    return PhysicsBodyBinding(
        body_id=value["body_id"],
        component_id=value["component_id"],
        behavior=behavior,
        geometry_source=geometry,
        motion_track_id=value["motion_track_id"],
        analytic_shape_id=value["analytic_shape_id"],
        material_record_id=value["material_record_id"],
        mass_properties_id=value["mass_properties_id"],
        dynamic_initial_state=initial,
        material_record_sha256=value["material_record_sha256"],
        mass_properties_sha256=value["mass_properties_sha256"],
    )


def _relation_from_document(value: object) -> ModelPhysicsRelation:
    expected = {
        "relation_id",
        "model_snapshot_sha256",
        "motion_plan_sha256",
        "body_bindings",
        "virtual_frame_bindings",
        "excluded_component_ids",
        "excluded_motion_track_ids",
        "content_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ModelPhysicsError("model-physics relation fields differ from the contract")
    raw_bodies = value["body_bindings"]
    raw_frames = value["virtual_frame_bindings"]
    excluded = value["excluded_component_ids"]
    excluded_tracks = value["excluded_motion_track_ids"]
    if not isinstance(raw_bodies, list) or not isinstance(raw_frames, list):
        raise ModelPhysicsError("relation bindings must be arrays")
    if not isinstance(excluded, list):
        raise ModelPhysicsError("excluded_component_ids must be an array")
    if not isinstance(excluded_tracks, list):
        raise ModelPhysicsError("excluded_motion_track_ids must be an array")
    frames = []
    for index, raw in enumerate(raw_frames):
        if not isinstance(raw, dict) or set(raw) != {
            "virtual_frame_id",
            "role",
            "motion_track_id",
        }:
            raise ModelPhysicsError(f"virtual_frame_binding[{index}] is invalid")
        frames.append(
            VirtualFrameBinding(
                virtual_frame_id=raw["virtual_frame_id"],
                role=raw["role"],
                motion_track_id=raw["motion_track_id"],
            )
        )
    relation = ModelPhysicsRelation(
        relation_id=value["relation_id"],
        model_snapshot_sha256=value["model_snapshot_sha256"],
        motion_plan_sha256=value["motion_plan_sha256"],
        body_bindings=tuple(
            _binding_from_document(raw, index) for index, raw in enumerate(raw_bodies)
        ),
        virtual_frame_bindings=tuple(frames),
        excluded_component_ids=tuple(excluded),
        excluded_motion_track_ids=tuple(excluded_tracks),
        content_sha256=value["content_sha256"],
    )
    if relation.to_document() != value:
        raise ModelPhysicsError("model-physics relation is not canonical")
    return relation


def load_physics_model_motion_input(
    payload: bytes, *, expected_file_sha256: str | None = None
) -> PhysicsModelMotionInput:
    """严格读取并重验模型、运动、关系、外层哈希和所有权闭包。"""

    if expected_file_sha256 is not None:
        _require_sha256(expected_file_sha256, "expected_file_sha256")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_file_sha256:
            raise ModelPhysicsError(
                f"locked input bytes changed: expected {expected_file_sha256}, got {actual}"
            )
    document = strict_loads(payload)
    expected = {
        "facet",
        "facet_version",
        "input_id",
        "qualification",
        "model",
        "motion",
        "relation",
        "evidence",
        "content_sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ModelPhysicsError("physics model-motion input fields differ from the contract")
    if document["facet"] != PHYSICS_MODEL_MOTION_INPUT_FACET:
        raise ModelPhysicsError(f"facet must be {PHYSICS_MODEL_MOTION_INPUT_FACET!r}")
    if not isinstance(document["facet_version"], str):
        raise ModelPhysicsError("facet_version must be a string")
    ENGINE_REGISTRY.assert_reader_compatible(
        PHYSICS_MODEL_MOTION_INPUT_FACET, document["facet_version"]
    )
    content_sha256 = _require_sha256(document["content_sha256"], "content_sha256")
    address_input = dict(document)
    address_input.pop("content_sha256")
    if canonical_sha256(address_input, PHYSICS_MODEL_MOTION_CANONICAL_PROFILE) != content_sha256:
        raise ModelPhysicsError("physics input content_sha256 does not match")
    try:
        model = model_snapshot_from_document(document["model"])
        motion = planned_motion_from_document(document["motion"])
    except (ModelSnapshotError, PlannedMotionError) as error:
        raise ModelPhysicsError(f"nested input is invalid: {error}") from error
    package = PhysicsModelMotionInput(
        input_id=document["input_id"],
        model=model,
        motion=motion,
        relation=_relation_from_document(document["relation"]),
        evidence=_parse_evidence(document["evidence"]),
        facet_version=document["facet_version"],
        content_sha256=content_sha256,
    )
    if package.to_document() != document:
        raise ModelPhysicsError("physics model-motion input is not canonical")
    if document["qualification"] != package.qualification:
        raise ModelPhysicsError("planned model-motion input must remain hypothesis_only")
    return package


__all__ = [
    "BodyBehavior",
    "DynamicBodyInitialState",
    "DynamicStateFrame",
    "GeometrySource",
    "ModelPhysicsError",
    "ModelPhysicsRelation",
    "PHYSICS_MODEL_MOTION_CANONICAL_PROFILE",
    "PhysicsBodyBinding",
    "PhysicsModelMotionInput",
    "VirtualFrameBinding",
    "load_physics_model_motion_input",
]
