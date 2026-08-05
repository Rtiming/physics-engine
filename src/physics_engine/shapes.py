"""模拟形状声明层——spec/11的首个实现（v0）。

**只做声明与校验，不做求解**：碰撞算法归求解器（spec/11第三节），
本模块给每个形状的唯一"算"是保守AABB（供broad phase）。
四条"必须红"（spec/11规则7）全部落在构造校验里：

1. 外观形不得进接触查询——usage="visual"的资产装进碰撞声明即拒；
2. 网格/SDF资产必须带单位与SHA-256；
3. 碰撞形必须声明保守方向（envelope/fitted），缺省禁止；
4. 参数化生成器必须带`algorithm:`前缀的身份。

规则1的另一半（碰撞形缺省用外观形禁止）由结构保证：`SimBody.collision`
是必填项，没有"从visual推导"的路径存在。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


class ShapeError(ValueError):
    """一切形状声明的失败关闭。"""


Vector3 = tuple[float, float, float]
Aabb = tuple[Vector3, Vector3]


def _require_vector(value: Vector3, name: str) -> None:
    if len(value) != 3 or not all(math.isfinite(component) for component in value):
        raise ShapeError(f"{name} must be a finite 3-vector")


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ShapeError(f"{name} must be positive and finite")


#: `Literal`只在类型检查期成立；跨边界字节（场景文件）走的是运行时这条路，
#: 所以取值集合要在运行时校验一次。缺省与非法值走同一个门。
MESH_UNITS: tuple[str, ...] = ("mm", "m")
MESH_USAGES: tuple[str, ...] = ("visual", "collision")
MESH_CONVEXITIES: tuple[str, ...] = ("convex_hull", "exact_convex", "nonconvex_declared")
COLLISION_DIRECTIONS: tuple[str, ...] = ("envelope", "fitted")


def _require_choice(value: object, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ShapeError(f"{name} must be one of {list(allowed)}: {value!r}")


@dataclass(frozen=True)
class Sphere:
    radius_mm: float

    def __post_init__(self) -> None:
        _require_positive(self.radius_mm, "radius_mm")

    def local_aabb_mm(self) -> Aabb:
        r = self.radius_mm
        return ((-r, -r, -r), (r, r, r))


@dataclass(frozen=True)
class Capsule:
    point_a_mm: Vector3
    point_b_mm: Vector3
    radius_mm: float

    def __post_init__(self) -> None:
        _require_vector(self.point_a_mm, "point_a_mm")
        _require_vector(self.point_b_mm, "point_b_mm")
        _require_positive(self.radius_mm, "radius_mm")

    def local_aabb_mm(self) -> Aabb:
        r = self.radius_mm
        low = tuple(min(a, b) - r for a, b in zip(self.point_a_mm, self.point_b_mm, strict=True))
        high = tuple(max(a, b) + r for a, b in zip(self.point_a_mm, self.point_b_mm, strict=True))
        return (low, high)  # type: ignore[return-value]


@dataclass(frozen=True)
class RoundedBox:
    half_extents_mm: Vector3
    fillet_radius_mm: float

    def __post_init__(self) -> None:
        _require_vector(self.half_extents_mm, "half_extents_mm")
        if any(extent <= 0.0 for extent in self.half_extents_mm):
            raise ShapeError("half_extents_mm must be positive")
        if not math.isfinite(self.fillet_radius_mm) or self.fillet_radius_mm < 0.0:
            raise ShapeError("fillet_radius_mm must be nonnegative")

    def local_aabb_mm(self) -> Aabb:
        f = self.fillet_radius_mm
        low = tuple(-(extent + f) for extent in self.half_extents_mm)
        high = tuple(extent + f for extent in self.half_extents_mm)
        return (low, high)  # type: ignore[return-value]


@dataclass(frozen=True)
class FiniteCylinder:
    """有限宽圆柱（轴向z），法兰外径可选——WDS导轮/带盘的声明形。"""

    radius_mm: float
    half_width_mm: float
    flange_outer_radius_mm: float | None = None

    def __post_init__(self) -> None:
        _require_positive(self.radius_mm, "radius_mm")
        _require_positive(self.half_width_mm, "half_width_mm")
        if self.flange_outer_radius_mm is not None:
            _require_positive(self.flange_outer_radius_mm, "flange_outer_radius_mm")
            if self.flange_outer_radius_mm < self.radius_mm:
                raise ShapeError("flange_outer_radius_mm must be >= radius_mm")

    def local_aabb_mm(self) -> Aabb:
        r = self.flange_outer_radius_mm or self.radius_mm
        w = self.half_width_mm
        return ((-r, -r, -w), (r, r, w))


@dataclass(frozen=True)
class MeshAsset:
    """网格资产声明：内容寻址+单位+用途+凸性语义+**声明的**AABB。

    引擎不解析网格字节，所以AABB由声明携带（消费方生成资产时算好）——
    这与MuJoCo"mesh碰撞默认凸包"的暗坑相反：凸性必须显式声明。
    """

    path_relative: str
    sha256: str
    units: Literal["mm", "m"]
    usage: Literal["visual", "collision"]
    convexity: Literal["convex_hull", "exact_convex", "nonconvex_declared"]
    aabb_min_mm: Vector3
    aabb_max_mm: Vector3

    def __post_init__(self) -> None:
        if not self.path_relative or "\\" in self.path_relative or self.path_relative.startswith("/"):
            raise ShapeError("path_relative must be a nonempty relative path")
        _require_choice(self.units, MESH_UNITS, "units")
        _require_choice(self.usage, MESH_USAGES, "usage")
        _require_choice(self.convexity, MESH_CONVEXITIES, "convexity")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ShapeError("sha256 must be 64 lowercase hex characters")
        _require_vector(self.aabb_min_mm, "aabb_min_mm")
        _require_vector(self.aabb_max_mm, "aabb_max_mm")
        if any(low >= high for low, high in zip(self.aabb_min_mm, self.aabb_max_mm, strict=True)):
            raise ShapeError("aabb_min_mm must be strictly below aabb_max_mm")

    def local_aabb_mm(self) -> Aabb:
        return (self.aabb_min_mm, self.aabb_max_mm)


@dataclass(frozen=True)
class GeneratedShape:
    """参数化生成器产出的形状引用：生成器身份+参数+产出的具体形。"""

    algorithm_id: str
    algorithm_version: str
    parameters: tuple[tuple[str, float], ...]
    shape: Sphere | Capsule | RoundedBox | FiniteCylinder

    def __post_init__(self) -> None:
        if not self.algorithm_id.startswith("algorithm:"):
            raise ShapeError("generator identity must carry the 'algorithm:' prefix")
        if not self.algorithm_version:
            raise ShapeError("generator requires a version")

    def local_aabb_mm(self) -> Aabb:
        return self.shape.local_aabb_mm()


Shape = Sphere | Capsule | RoundedBox | FiniteCylinder | MeshAsset | GeneratedShape


@dataclass(frozen=True)
class CollisionShape:
    """碰撞形声明：**保守方向必填**（spec/11规则5）。"""

    shape: Shape
    direction: Literal["envelope", "fitted"]

    def __post_init__(self) -> None:
        _require_choice(self.direction, COLLISION_DIRECTIONS, "collision direction")
        if isinstance(self.shape, MeshAsset) and self.shape.usage != "collision":
            raise ShapeError(
                "a visual asset must never enter collision declarations (spec/11 rule 1)"
            )


@dataclass(frozen=True)
class VisualShape:
    shape: Shape


@dataclass(frozen=True)
class SimBody:
    """仿真体：碰撞形必填；外观缺省复用碰撞形（单行道，反向不存在）。"""

    body_id: str
    collision: CollisionShape
    visual: VisualShape | None = None
    mass_kg: float | None = None

    def __post_init__(self) -> None:
        if "/" not in self.body_id:
            raise ShapeError("body_id must be namespaced like 'body/spool'")
        if self.mass_kg is not None:
            _require_positive(self.mass_kg, "mass_kg")


@dataclass(frozen=True)
class PosedBody:
    """位姿+体。位姿来自MotionSource或静态声明，不属于形状。"""

    body: SimBody
    translation_mm: Vector3 = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        _require_vector(self.translation_mm, "translation_mm")
        norm = math.sqrt(sum(component * component for component in self.rotation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ShapeError("rotation_xyzw must be a unit quaternion")

    def rotate_local_mm(self, point: Vector3) -> Vector3:
        """局部向量→世界向量（只转不平移）。"""

        x, y, z, w = self.rotation_xyzw
        rows = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        return tuple(
            sum(row[i] * point[i] for i in range(3)) for row in rows
        )  # type: ignore[return-value]

    def transform_point_mm(self, point: Vector3) -> Vector3:
        rotated = self.rotate_local_mm(point)
        return tuple(
            rotated[axis] + self.translation_mm[axis] for axis in range(3)
        )  # type: ignore[return-value]

    def world_aabb_mm(self) -> Aabb:
        """旋转后取八角点包盒——保守，供broad phase。"""

        (lx, ly, lz), (hx, hy, hz) = self.body.collision.shape.local_aabb_mm()
        x, y, z, w = self.rotation_xyzw
        rows = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        corners = [
            (cx, cy, cz)
            for cx in (lx, hx)
            for cy in (ly, hy)
            for cz in (lz, hz)
        ]
        world = [
            tuple(
                sum(row[i] * corner[i] for i in range(3)) + self.translation_mm[axis]
                for axis, row in enumerate(rows)
            )
            for corner in corners
        ]
        low = tuple(min(point[axis] for point in world) for axis in range(3))
        high = tuple(max(point[axis] for point in world) for axis in range(3))
        return (low, high)  # type: ignore[return-value]


__all__ = [
    "COLLISION_DIRECTIONS",
    "MESH_CONVEXITIES",
    "MESH_UNITS",
    "MESH_USAGES",
    "Aabb",
    "Capsule",
    "CollisionShape",
    "FiniteCylinder",
    "GeneratedShape",
    "MeshAsset",
    "PosedBody",
    "RoundedBox",
    "ShapeError",
    "SimBody",
    "Sphere",
    "VisualShape",
]
