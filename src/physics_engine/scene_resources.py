"""模型资产引用到Scene碰撞形状的严格资源边界。

本模块只做资源身份和字节复读，不解析STL/OBJ，也不计算AABB或凸性。调用方必须显式
提供``CollisionAssetLoadSpec``；本模块重新读取包内文件、核SHA，再把声明与
``shapes.MeshAsset``钉在同一个记录里。这样“文件存在”“形状声明存在”和“二者确实
属于同一资产”是三道独立门。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from physics_engine.identity import IdentityError, parse_namespace_id
from physics_engine.model_snapshot import AssetRole, ModelAssetRef
from physics_engine.motion import Pose
from physics_engine.shapes import (
    COLLISION_DIRECTIONS,
    MESH_CONVEXITIES,
    CollisionShape,
    MeshAsset,
    ShapeError,
    Vector3,
)


class SceneResourceError(ValueError):
    """资产字节、形状记录或资源目录不闭合。"""


def _require_namespace(value: object, namespace: str, name: str) -> str:
    if not isinstance(value, str):
        raise SceneResourceError(f"{name} must be a string")
    try:
        actual, _ = parse_namespace_id(value)
    except IdentityError as error:
        raise SceneResourceError(f"{name} is not a valid namespace id: {error}") from error
    if actual != namespace:
        raise SceneResourceError(f"{name} must live in {namespace!r}: {value!r}")
    return value


def _require_vector(value: object, name: str) -> Vector3:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        or not all(math.isfinite(float(item)) for item in value)
    ):
        raise SceneResourceError(f"{name} must be a finite 3-tuple")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class CollisionAssetLoadSpec:
    """不能从网格文件名猜出的碰撞表示声明。"""

    asset_id: str
    direction: str
    convexity: str
    aabb_min_mm: Vector3
    aabb_max_mm: Vector3

    def __post_init__(self) -> None:
        _require_namespace(self.asset_id, "asset", "asset_id")
        if self.direction not in COLLISION_DIRECTIONS:
            raise SceneResourceError(
                f"direction must be one of {list(COLLISION_DIRECTIONS)}"
            )
        if self.convexity not in MESH_CONVEXITIES:
            raise SceneResourceError(
                f"convexity must be one of {list(MESH_CONVEXITIES)}"
            )
        low = _require_vector(self.aabb_min_mm, "aabb_min_mm")
        high = _require_vector(self.aabb_max_mm, "aabb_max_mm")
        if any(left >= right for left, right in zip(low, high, strict=True)):
            raise SceneResourceError("aabb_min_mm must be strictly below aabb_max_mm")


@dataclass(frozen=True)
class LoadedCollisionAsset:
    """已经复读源字节，并与一个碰撞形状声明闭合的资产。"""

    asset: ModelAssetRef
    collision: CollisionShape
    byte_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.asset, ModelAssetRef):
            raise SceneResourceError("asset must be a ModelAssetRef")
        if self.asset.role is not AssetRole.COLLISION:
            raise SceneResourceError("loaded collision asset requires collision role")
        if not isinstance(self.collision, CollisionShape) or not isinstance(
            self.collision.shape, MeshAsset
        ):
            raise SceneResourceError("loaded collision asset must resolve to MeshAsset")
        mesh = self.collision.shape
        if (
            mesh.path_relative != self.asset.path_relative
            or mesh.sha256 != self.asset.sha256
            or mesh.units != self.asset.units
            or mesh.usage != "collision"
        ):
            raise SceneResourceError("MeshAsset identity differs from ModelAssetRef")
        if isinstance(self.byte_length, bool) or not isinstance(self.byte_length, int):
            raise SceneResourceError("byte_length must be an integer")
        if self.byte_length <= 0:
            raise SceneResourceError("collision asset bytes must not be empty")


@dataclass(frozen=True)
class AnalyticCollisionRecord:
    """一个组件局部frame下的解析碰撞形状。"""

    shape_id: str
    collision: CollisionShape
    component_from_shape: Pose

    def __post_init__(self) -> None:
        _require_namespace(self.shape_id, "shape", "shape_id")
        if not isinstance(self.collision, CollisionShape):
            raise SceneResourceError("collision must be a CollisionShape")
        if isinstance(self.collision.shape, MeshAsset):
            raise SceneResourceError("analytic shape record cannot contain a MeshAsset")
        if not isinstance(self.component_from_shape, Pose):
            raise SceneResourceError("component_from_shape must be a Pose")


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            length += len(chunk)
    return digest.hexdigest(), length


def load_collision_asset(
    package_root: Path,
    asset: ModelAssetRef,
    specification: CollisionAssetLoadSpec,
) -> LoadedCollisionAsset:
    """从包根复读一份碰撞资产，并把源身份与AABB/凸性声明闭合。"""

    if not isinstance(asset, ModelAssetRef):
        raise SceneResourceError("asset must be a ModelAssetRef")
    if asset.role is not AssetRole.COLLISION:
        raise SceneResourceError("asset loader requires collision role")
    if not isinstance(specification, CollisionAssetLoadSpec):
        raise SceneResourceError("specification must be CollisionAssetLoadSpec")
    if specification.asset_id != asset.asset_id:
        raise SceneResourceError(
            f"specification {specification.asset_id!r} names another asset, "
            f"expected {asset.asset_id!r}"
        )
    root = Path(package_root).resolve()
    if not root.is_dir():
        raise SceneResourceError(f"package_root is not a directory: {package_root}")
    path = (root / asset.path_relative).resolve()
    if not path.is_relative_to(root):
        raise SceneResourceError(
            f"asset path escapes package_root after resolving links: {asset.path_relative!r}"
        )
    if not path.is_file():
        raise SceneResourceError(f"collision asset is not a file: {asset.path_relative!r}")
    actual_sha256, byte_length = _file_sha256(path)
    if actual_sha256 != asset.sha256:
        raise SceneResourceError(
            f"collision asset sha256 differs: expected {asset.sha256}, got {actual_sha256}"
        )
    try:
        mesh = MeshAsset(
            path_relative=asset.path_relative,
            sha256=actual_sha256,
            units=asset.units,
            usage="collision",
            convexity=specification.convexity,  # type: ignore[arg-type]
            aabb_min_mm=specification.aabb_min_mm,
            aabb_max_mm=specification.aabb_max_mm,
        )
        collision = CollisionShape(
            shape=mesh, direction=specification.direction  # type: ignore[arg-type]
        )
    except ShapeError as error:
        raise SceneResourceError(f"invalid collision shape record: {error}") from error
    return LoadedCollisionAsset(
        asset=asset, collision=collision, byte_length=byte_length
    )


@dataclass(frozen=True)
class SceneResourceCatalog:
    """装配前已经闭合的碰撞资产与解析形状目录。"""

    collision_assets: tuple[LoadedCollisionAsset, ...]
    analytic_shapes: tuple[AnalyticCollisionRecord, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(item, LoadedCollisionAsset) for item in self.collision_assets):
            raise SceneResourceError("collision_assets contains another type")
        if not all(isinstance(item, AnalyticCollisionRecord) for item in self.analytic_shapes):
            raise SceneResourceError("analytic_shapes contains another type")
        for label, identifiers in (
            ("collision asset", tuple(item.asset.asset_id for item in self.collision_assets)),
            ("analytic shape", tuple(item.shape_id for item in self.analytic_shapes)),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise SceneResourceError(f"duplicate {label} id in resource catalog")

    def collision_asset(self, asset_id: str) -> LoadedCollisionAsset:
        for record in self.collision_assets:
            if record.asset.asset_id == asset_id:
                return record
        raise SceneResourceError(f"resource catalog has no collision asset {asset_id!r}")

    def analytic_shape(self, shape_id: str) -> AnalyticCollisionRecord:
        for record in self.analytic_shapes:
            if record.shape_id == shape_id:
                return record
        raise SceneResourceError(f"resource catalog has no analytic shape {shape_id!r}")


__all__ = [
    "AnalyticCollisionRecord",
    "CollisionAssetLoadSpec",
    "LoadedCollisionAsset",
    "SceneResourceCatalog",
    "SceneResourceError",
    "load_collision_asset",
]
