"""上游无关的模型快照——模型从哪里来，不改变physics-engine怎样读它。

模型快照只保存身份、层级、参考位姿以及显式visual/collision资产角色。GCW、WII、CAD
软件或人工导出都可以成为producer，但任何producer名称都不进入求解语义。visual资产永远
不会被静默回退为collision资产；真正怎样成为虚拟物理体由``model_physics``另行绑定。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

from physics_engine.canonical import WDS_PROFILE, canonical_sha256
from physics_engine.identity import IdentityError, parse_namespace_id
from physics_engine.motion import Pose

MODEL_SNAPSHOT_CANONICAL_PROFILE = WDS_PROFILE


class ModelSnapshotError(ValueError):
    """模型快照身份、资产角色、层级或字节闭包错误。"""


class AssetRole(enum.StrEnum):
    VISUAL = "visual"
    COLLISION = "collision"


def _require_namespace(value: object, namespace: str, name: str) -> str:
    if not isinstance(value, str):
        raise ModelSnapshotError(f"{name} must be a string: {value!r}")
    try:
        parsed, _ = parse_namespace_id(value)
    except IdentityError as error:
        raise ModelSnapshotError(f"{name} is not a valid namespace id: {error}") from error
    if parsed != namespace:
        raise ModelSnapshotError(f"{name} must live in {namespace!r}: {value!r}")
    return value


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelSnapshotError(f"{name} must be a nonempty string")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
        raise ModelSnapshotError(f"{name} must use lowercase identifier characters: {value!r}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelSnapshotError(f"{name} must be 64 lowercase hex characters")
    return value


def _require_relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ModelSnapshotError(f"{name} must be a portable relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelSnapshotError(f"{name} must not be absolute or escape its package: {value!r}")
    return value


def pose_to_document(pose: Pose) -> dict[str, list[float]]:
    if not isinstance(pose, Pose):
        raise ModelSnapshotError(f"expected Pose, got {pose!r}")
    return {
        "translation_parent_from_child_mm": list(pose.translation_mm),
        "rotation_parent_from_child_xyzw": list(pose.rotation_xyzw),
    }


def pose_from_document(value: object, name: str) -> Pose:
    if not isinstance(value, dict) or set(value) != {
        "translation_parent_from_child_mm",
        "rotation_parent_from_child_xyzw",
    }:
        raise ModelSnapshotError(f"{name} must be one exact rigid transform object")
    translation = value["translation_parent_from_child_mm"]
    rotation = value["rotation_parent_from_child_xyzw"]
    if not isinstance(translation, list) or len(translation) != 3:
        raise ModelSnapshotError(f"{name}.translation must have three values")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ModelSnapshotError(f"{name}.rotation must have four values")
    try:
        return Pose(tuple(translation), tuple(rotation))  # type: ignore[arg-type]
    except ValueError as error:
        raise ModelSnapshotError(f"invalid {name}: {error}") from error


@dataclass(frozen=True)
class ModelAssetRef:
    asset_id: str
    role: AssetRole
    path_relative: str
    sha256: str
    format: str
    units: str
    frame_id: str
    component_from_asset: Pose

    def __post_init__(self) -> None:
        _require_namespace(self.asset_id, "asset", "asset_id")
        if not isinstance(self.role, AssetRole):
            raise ModelSnapshotError(f"role must be an AssetRole: {self.role!r}")
        _require_relative_path(self.path_relative, "path_relative")
        _require_sha256(self.sha256, "sha256")
        _require_identifier(self.format, "format")
        if self.units != "mm":
            raise ModelSnapshotError(
                f"model snapshot assets must be normalized to mm, got {self.units!r}"
            )
        _require_namespace(self.frame_id, "frame", "frame_id")
        if not isinstance(self.component_from_asset, Pose):
            raise ModelSnapshotError("component_from_asset must be a Pose")

    def to_document(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "role": self.role.value,
            "path_relative": self.path_relative,
            "sha256": self.sha256,
            "format": self.format,
            "units": self.units,
            "frame_id": self.frame_id,
            "component_from_asset": pose_to_document(self.component_from_asset),
        }


@dataclass(frozen=True)
class ModelComponent:
    component_id: str
    frame_id: str
    semantic_role: str
    parent_component_id: str | None
    parent_from_component: Pose
    visual_asset: ModelAssetRef | None
    collision_asset: ModelAssetRef | None

    def __post_init__(self) -> None:
        _require_namespace(self.component_id, "model-component", "component_id")
        _require_namespace(self.frame_id, "frame", "frame_id")
        _require_identifier(self.semantic_role, "semantic_role")
        if self.parent_component_id is not None:
            _require_namespace(
                self.parent_component_id, "model-component", "parent_component_id"
            )
            if self.parent_component_id == self.component_id:
                raise ModelSnapshotError("a model component cannot parent itself")
        if not isinstance(self.parent_from_component, Pose):
            raise ModelSnapshotError("parent_from_component must be a Pose")
        for name, asset, role in (
            ("visual_asset", self.visual_asset, AssetRole.VISUAL),
            ("collision_asset", self.collision_asset, AssetRole.COLLISION),
        ):
            if asset is not None and not isinstance(asset, ModelAssetRef):
                raise ModelSnapshotError(f"{name} must be a ModelAssetRef or None")
            if asset is not None and asset.role is not role:
                raise ModelSnapshotError(
                    f"{name} must declare role={role.value!r}, got {asset.role.value!r}"
                )

    def to_document(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "frame_id": self.frame_id,
            "semantic_role": self.semantic_role,
            "parent_component_id": self.parent_component_id,
            "parent_from_component": pose_to_document(self.parent_from_component),
            "visual_asset": (
                None if self.visual_asset is None else self.visual_asset.to_document()
            ),
            "collision_asset": (
                None if self.collision_asset is None else self.collision_asset.to_document()
            ),
        }


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    root_frame_id: str
    producer_id: str
    source_manifest_sha256: str
    components: tuple[ModelComponent, ...]
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_namespace(self.model_id, "model", "model_id")
        _require_namespace(self.root_frame_id, "frame", "root_frame_id")
        _require_namespace(self.producer_id, "producer", "producer_id")
        _require_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        if not self.components or not all(
            isinstance(component, ModelComponent) for component in self.components
        ):
            raise ModelSnapshotError("components must contain ModelComponent values")
        component_ids = tuple(component.component_id for component in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ModelSnapshotError("model component IDs must be unique")
        component_frames = tuple(component.frame_id for component in self.components)
        if len(set(component_frames)) != len(component_frames):
            raise ModelSnapshotError("model component frame IDs must be unique")
        known = set(component_ids)
        for component in self.components:
            if (
                component.parent_component_id is not None
                and component.parent_component_id not in known
            ):
                raise ModelSnapshotError(
                    f"component {component.component_id!r} names unknown parent "
                    f"{component.parent_component_id!r}"
                )
        self._assert_acyclic()
        asset_ids = [
            asset.asset_id
            for component in self.components
            for asset in (component.visual_asset, component.collision_asset)
            if asset is not None
        ]
        if len(set(asset_ids)) != len(asset_ids):
            raise ModelSnapshotError("asset IDs must be unique across the model snapshot")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
            if self.content_sha256 != self.content_address():
                raise ModelSnapshotError("model snapshot content_sha256 does not match")

    def _assert_acyclic(self) -> None:
        parents = {
            component.component_id: component.parent_component_id
            for component in self.components
        }
        for component_id in parents:
            seen: set[str] = set()
            current: str | None = component_id
            while current is not None:
                if current in seen:
                    raise ModelSnapshotError(f"model component hierarchy has a cycle at {current}")
                seen.add(current)
                current = parents[current]

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        root_frame_id: str,
        producer_id: str,
        source_manifest_sha256: str,
        components: tuple[ModelComponent, ...],
    ) -> ModelSnapshot:
        return cls(
            model_id=model_id,
            root_frame_id=root_frame_id,
            producer_id=producer_id,
            source_manifest_sha256=source_manifest_sha256,
            components=components,
        ).sealed()

    def component(self, component_id: str) -> ModelComponent:
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise ModelSnapshotError(f"unknown model component: {component_id}")

    def to_document(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "root_frame_id": self.root_frame_id,
            "producer_id": self.producer_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "components": [component.to_document() for component in self.components],
            "content_sha256": self.content_sha256,
        }

    def content_address(self) -> str:
        document = self.to_document()
        document.pop("content_sha256")
        return canonical_sha256(document, MODEL_SNAPSHOT_CANONICAL_PROFILE)

    def sealed(self) -> ModelSnapshot:
        return replace(self, content_sha256=self.content_address())


def _asset_from_document(value: object, name: str) -> ModelAssetRef | None:
    if value is None:
        return None
    expected = {
        "asset_id",
        "role",
        "path_relative",
        "sha256",
        "format",
        "units",
        "frame_id",
        "component_from_asset",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ModelSnapshotError(f"{name} must be an exact asset reference object")
    try:
        role = AssetRole(value["role"])
    except (TypeError, ValueError) as error:
        raise ModelSnapshotError(f"{name}.role is invalid") from error
    return ModelAssetRef(
        asset_id=value["asset_id"],
        role=role,
        path_relative=value["path_relative"],
        sha256=value["sha256"],
        format=value["format"],
        units=value["units"],
        frame_id=value["frame_id"],
        component_from_asset=pose_from_document(
            value["component_from_asset"], f"{name}.component_from_asset"
        ),
    )


def model_snapshot_from_document(value: object) -> ModelSnapshot:
    expected = {
        "model_id",
        "root_frame_id",
        "producer_id",
        "source_manifest_sha256",
        "components",
        "content_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ModelSnapshotError("model snapshot fields differ from the contract")
    raw_components = value["components"]
    if not isinstance(raw_components, list):
        raise ModelSnapshotError("model snapshot components must be an array")
    components = []
    component_keys = {
        "component_id",
        "frame_id",
        "semantic_role",
        "parent_component_id",
        "parent_from_component",
        "visual_asset",
        "collision_asset",
    }
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, dict) or set(raw) != component_keys:
            raise ModelSnapshotError(f"component[{index}] fields differ from the contract")
        components.append(
            ModelComponent(
                component_id=raw["component_id"],
                frame_id=raw["frame_id"],
                semantic_role=raw["semantic_role"],
                parent_component_id=raw["parent_component_id"],
                parent_from_component=pose_from_document(
                    raw["parent_from_component"], f"component[{index}].pose"
                ),
                visual_asset=_asset_from_document(
                    raw["visual_asset"], f"component[{index}].visual_asset"
                ),
                collision_asset=_asset_from_document(
                    raw["collision_asset"], f"component[{index}].collision_asset"
                ),
            )
        )
    snapshot = ModelSnapshot(
        model_id=value["model_id"],
        root_frame_id=value["root_frame_id"],
        producer_id=value["producer_id"],
        source_manifest_sha256=value["source_manifest_sha256"],
        components=tuple(components),
        content_sha256=value["content_sha256"],
    )
    if snapshot.to_document() != value:
        raise ModelSnapshotError("model snapshot is not canonical")
    return snapshot


__all__ = [
    "AssetRole",
    "MODEL_SNAPSHOT_CANONICAL_PROFILE",
    "ModelAssetRef",
    "ModelComponent",
    "ModelSnapshot",
    "ModelSnapshotError",
    "model_snapshot_from_document",
    "pose_from_document",
    "pose_to_document",
]
