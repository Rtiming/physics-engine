"""碰撞查询——spec/10 `CollisionQuery`：broad phase + narrow phase。

事件带可信度是接口的承重字段：broad phase命中≠真的撞。

## 三档可信度，各自的含义逐字

* ``"narrow_phase"``——**精确到舍入**。球/胶囊族走线段-线段闭式；
  球型探针对解析原语（球/胶囊/圆角盒/无法兰圆柱）走精确有符号距离；
  轴对齐的盒对盒走逐轴闭式。这一档的``penetration_mm``是真的穿透深度；
* ``"sampled_field"``——**从一个有限分辨率的采样场里读出来的**（决策0090第五节）。
  数带一个**已声明的偏差估计**（``NarrowPhaseResult.estimated_bias_mm``），
  主项来自0085那条``S = φ + (h²/6)·∇²φ``。**它不叫`narrow_phase`是有意的**：
  那一档在本仓的既有语义里是"精确到舍入"，把带O(h²)误差的数塞进去就是冒充；
* ``"broad_phase"``——**不知道**。``penetration_mm=None``，事件照报（不漏报）、
  深度留空（不冒充）。旋转盒对盒、胶囊对盒、圆柱对圆柱、带法兰的圆柱、
  没有配场的`MeshAsset`，全部落在这一档。

**broad命中但narrow判分离的对不再报事件**（假阳性消除）——这一条对三档一致。

## 失败关闭的两个口径（决策0090第六节）

**直查API没有"不知道"这个返回值**，所以答不出来一律`CollisionQueryError`；
**场景查询有**（就是``broad_phase`` + ``None``），所以它降级而不是抛。
把场景查询也改成抛会破掉"不漏报"这条唯一硬承诺：一个查不了的对会让整场炸掉。

## 距离场走协议不走import（决策0090第三节）

`collision`在基座`modelgen`圈、`contact.field`在力学域圈，而域隔离门写着
"基座不依赖任何上层"。于是本模块只声明`SignedDistanceSource`这个结构化协议，
`contact.field.SignedDistanceField`**恰好逐条满足**，鸭子类型直接可用，
**一行import都不需要**。装配由上层做——那本来就是上层的活。

装配期校验（PyElastica形制，spec/10场景装配第3条）：构造时逐体校验、
白名单按对声明（WII相邻连杆忽略的同型），配错当场炸，不拖到查询时。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from physics_engine.shapes import (
    Aabb,
    Capsule,
    FiniteCylinder,
    GeneratedShape,
    MeshAsset,
    PosedBody,
    RoundedBox,
    Shape,
    ShapeError,
    Sphere,
    Vector3,
)


class CollisionQueryError(ValueError):
    """窄相查询的失败关闭。

    **与`ShapeError`分开**：那一个是"这个声明本身非法"，这一个是
    "声明合法，但这道查询本模块答不出来"。调用方对两者的反应不同——
    前者要改场景，后者要么换算法要么接受降级。
    """



def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else (min(value, 1.0))


def segment_segment_distance_mm(
    p1: Vector3, q1: Vector3, p2: Vector3, q2: Vector3
) -> float:
    """两线段最近距离（Ericson《Real-Time Collision Detection》标准算法）。"""

    d1, d2, r = _sub(q1, p1), _sub(q2, p2), _sub(p1, p2)
    a, e, f = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    if a <= 1e-12 and e <= 1e-12:
        return _dot(r, r) ** 0.5
    if a <= 1e-12:
        s, t = 0.0, _clamp(f / e)
    else:
        c = _dot(d1, r)
        if e <= 1e-12:
            t, s = 0.0, _clamp(-c / a)
        else:
            b = _dot(d1, d2)
            denominator = a * e - b * b
            s = _clamp((b * f - c * e) / denominator) if denominator > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clamp((b - c) / a)
    closest1 = (p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s)
    closest2 = (p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t)
    gap = _sub(closest1, closest2)
    return _dot(gap, gap) ** 0.5


def _as_world_segment(posed: PosedBody) -> tuple[Vector3, Vector3, float] | None:
    """球/胶囊族→世界系(端点a, 端点b, 半径)；其余族返回None。"""

    shape = posed.body.collision.shape
    if isinstance(shape, GeneratedShape):
        shape = shape.shape
    if isinstance(shape, Sphere):
        # 球的局部原点就是球心，姿态不改变它；直接取平移还避免通用四元数路径。
        centre = posed.translation_mm
        return (centre, centre, shape.radius_mm)
    if isinstance(shape, Capsule):
        return (
            posed.transform_point_mm(shape.point_a_mm),
            posed.transform_point_mm(shape.point_b_mm),
            shape.radius_mm,
        )
    return None


def _segment_pair_separation_mm(
    segment_a: tuple[Vector3, Vector3, float], segment_b: tuple[Vector3, Vector3, float]
) -> float:
    """球/胶囊族的分离量。**这一行是既有窄相的全部**，不许改写。

    `check_state_with_stats`与`narrow_phase_separation_mm`都走它，
    于是两条调用面**逐位相同**是结构保证的，不靠对拍去发现它漂了。
    """

    return segment_segment_distance_mm(
        segment_a[0], segment_a[1], segment_b[0], segment_b[1]
    ) - (segment_a[2] + segment_b[2])


# --------------------------------------------------------------------------
# 解析原语的精确有符号距离（决策0090第四节）
#
# 四条全是标准闭式，**体内体外都精确**。它们与`contact/field.py`那三条解析SDF
# 是同一件事的两个方向：那边把闭式烘成场当插值的金标，这边直接拿闭式当查询。
# 两处各写一份不是重复——**那边的三条是"光滑到可以量插值阶"的特意选形**
# （无限长圆柱、不带节点半径），这边的四条要与`shapes.py`的声明**逐字段对上**。
# --------------------------------------------------------------------------


def _sphere_signed_distance_mm(point_mm: Vector3, radius_mm: float) -> float:
    """``|x| − R``，球心在局部原点。"""

    return math.sqrt(sum(component * component for component in point_mm)) - radius_mm


def _capsule_signed_distance_mm(
    point_mm: Vector3, point_a_mm: Vector3, point_b_mm: Vector3, radius_mm: float
) -> float:
    """``dist(x, 线段ab) − r``。线段退化成点时自动落到球那一支。"""

    axis = _sub(point_b_mm, point_a_mm)
    offset = _sub(point_mm, point_a_mm)
    length_squared = _dot(axis, axis)
    t = 0.0 if length_squared <= 1e-12 else _clamp(_dot(offset, axis) / length_squared)
    closest = (
        offset[0] - axis[0] * t,
        offset[1] - axis[1] * t,
        offset[2] - axis[2] * t,
    )
    return math.sqrt(_dot(closest, closest)) - radius_mm


def _rounded_box_signed_distance_mm(
    point_mm: Vector3, half_extents_mm: Vector3, fillet_radius_mm: float
) -> float:
    """圆角盒的精确SDF：``|max(q,0)| + min(max q_i, 0) − f``，``q_i = |x_i| − h_i``。

    圆角盒＝核心盒⊕半径``f``的球，而**对凸集``K``，``K⊕B_f``的有符号距离
    恰好是``K``的减去``f``**（体外体内都成立）。所以圆角只是末尾一次减法。
    """

    q = tuple(abs(point_mm[axis]) - half_extents_mm[axis] for axis in range(3))
    outside = math.sqrt(sum(max(component, 0.0) ** 2 for component in q))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return outside + inside - fillet_radius_mm


def _finite_cylinder_signed_distance_mm(
    point_mm: Vector3, radius_mm: float, half_width_mm: float
) -> float:
    """有限宽圆柱（轴向z）的精确SDF。棱上只有C⁰——**值仍然精确**，那是两回事。"""

    radial = math.sqrt(point_mm[0] * point_mm[0] + point_mm[1] * point_mm[1]) - radius_mm
    axial = abs(point_mm[2]) - half_width_mm
    if radial > 0.0 or axial > 0.0:
        return math.sqrt(max(radial, 0.0) ** 2 + max(axial, 0.0) ** 2)
    return max(radial, axial)


def _unwrap(shape: Shape) -> Shape:
    return shape.shape if isinstance(shape, GeneratedShape) else shape


def shape_signed_distance_mm(shape: Shape, local_point_mm: Vector3) -> float:
    """形状（在**自己的局部系**里）到一点的精确有符号距离，mm。

    ``< 0``在体内、``> 0``在体外、``= 0``在面上。四种解析原语精确到舍入。

    **两条失败关闭**（决策0090第四节）：

    * `MeshAsset`——内核不读网格字节，它只带路径＋SHA-256＋声明的AABB，
      **一个顶点都没有**。这不是"还没实现"，是形制上够不着；
    * 带法兰的`FiniteCylinder`——``flange_outer_radius_mm``**没有实体语义**，
      全仓只用来把AABB撑大。两端各一片还是整段？厚度多少？没有任何声明说得出来。
      按无法兰算会**系统性少报穿透**（真实导轮法兰比槽底大15 mm），而少报是静默的那一半。
    """

    resolved = _unwrap(shape)
    if isinstance(resolved, Sphere):
        return _sphere_signed_distance_mm(local_point_mm, resolved.radius_mm)
    if isinstance(resolved, Capsule):
        return _capsule_signed_distance_mm(
            local_point_mm, resolved.point_a_mm, resolved.point_b_mm, resolved.radius_mm
        )
    if isinstance(resolved, RoundedBox):
        return _rounded_box_signed_distance_mm(
            local_point_mm, resolved.half_extents_mm, resolved.fillet_radius_mm
        )
    if isinstance(resolved, FiniteCylinder):
        if resolved.flange_outer_radius_mm is not None:
            raise CollisionQueryError(
                "a flanged FiniteCylinder has no declared solid semantics — "
                "flange_outer_radius_mm only widens the AABB today (决策0090第4.1节). "
                "按无法兰算会系统性少报穿透，而少报是静默的那一半"
            )
        return _finite_cylinder_signed_distance_mm(
            local_point_mm, resolved.radius_mm, resolved.half_width_mm
        )
    if isinstance(resolved, MeshAsset):
        raise CollisionQueryError(
            "a MeshAsset carries no geometry in this repo — only a path, a sha256 and a "
            "declared AABB (内核不读网格字节，决策0074第二节). "
            "网格那一侧要由外部烘好的距离场经SignedDistanceSource协议进来"
        )
    raise CollisionQueryError(f"no exact signed distance for shape {resolved!r}")


def _rotation_rows(
    rotation_xyzw: tuple[float, float, float, float],
) -> tuple[Vector3, Vector3, Vector3]:
    """四元数→旋转矩阵的三行。

    与`shapes.PosedBody.rotate_local_mm`同一个式子写两遍，是因为本模块要的是
    **转置**那一支（世界→局部），而`shapes.py`本轨不碰（决策0090第八节）。
    """

    x, y, z, w = rotation_xyzw
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def world_to_local_mm(posed: PosedBody, world_point_mm: Vector3) -> Vector3:
    """世界点→体的局部系。``R^T·(x − t)``。"""

    rows = _rotation_rows(posed.rotation_xyzw)
    delta = _sub(world_point_mm, posed.translation_mm)
    return tuple(
        sum(rows[row][axis] * delta[row] for row in range(3)) for axis in range(3)
    )  # type: ignore[return-value]


def posed_signed_distance_mm(posed: PosedBody, world_point_mm: Vector3) -> float:
    """位姿好的体到一个世界点的精确有符号距离，mm。失败关闭同`shape_signed_distance_mm`。"""

    return shape_signed_distance_mm(
        posed.body.collision.shape, world_to_local_mm(posed, world_point_mm)
    )


# --------------------------------------------------------------------------
# 半空间：查询侧的目标，**不是`shapes.py`的形状**（决策0090第四节第3条）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HalfSpace:
    """半空间``(x − p)·n ≥ 0``为体外。

    **它进不了`shapes.py`，理由是硬的**：半空间没有有限AABB，
    于是`PosedBody.world_aabb_mm`对它无定义，broad phase也就没法收它。
    所以它只出现在查询侧，走`half_space_separation_mm`这条独立调用面。
    """

    point_mm: Vector3
    unit_normal: Vector3

    def __post_init__(self) -> None:
        for name, value in (("point_mm", self.point_mm), ("unit_normal", self.unit_normal)):
            if len(value) != 3 or not all(math.isfinite(item) for item in value):
                raise CollisionQueryError(f"{name} must be a finite 3-vector: {value!r}")
        norm = math.sqrt(_dot(self.unit_normal, self.unit_normal))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise CollisionQueryError(f"unit_normal must be a unit vector: {norm!r}")

    def signed_distance_mm(self, point_mm: Vector3) -> float:
        """``(x − p)·n``。仿射，所以处处精确。"""

        return _dot(_sub(point_mm, self.point_mm), self.unit_normal)


def _lowest_support_mm(shape: Shape, direction: Vector3) -> float:
    """``min_{q ∈ K} q·d``，``K``是局部系里的形状、``|d| = 1``。

    支撑函数的负半支。四种原语全是闭式——半空间那条判据要的正是它：
    体到平面的最小有符号距离 = ``(t − p)·n + min_q q·(R^T n)``。
    """

    resolved = _unwrap(shape)
    if isinstance(resolved, Sphere):
        return -resolved.radius_mm
    if isinstance(resolved, Capsule):
        return (
            min(_dot(resolved.point_a_mm, direction), _dot(resolved.point_b_mm, direction))
            - resolved.radius_mm
        )
    if isinstance(resolved, RoundedBox):
        return (
            -sum(
                resolved.half_extents_mm[axis] * abs(direction[axis]) for axis in range(3)
            )
            - resolved.fillet_radius_mm
        )
    if isinstance(resolved, FiniteCylinder):
        if resolved.flange_outer_radius_mm is not None:
            raise CollisionQueryError(
                "a flanged FiniteCylinder has no declared solid semantics "
                "(决策0090第4.1节)"
            )
        radial = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        return -(
            abs(direction[2]) * resolved.half_width_mm + radial * resolved.radius_mm
        )
    if isinstance(resolved, MeshAsset):
        raise CollisionQueryError(
            "a MeshAsset carries no geometry in this repo (内核不读网格字节)"
        )
    raise CollisionQueryError(f"no support function for shape {resolved!r}")


def half_space_separation_mm(posed: PosedBody, half_space: HalfSpace) -> float:
    """体到半空间的**精确**分离量，mm。``< 0``是穿透深度。

    走支撑函数：``min_x (x − p)·n``在凸体上就是``−h_K(−n)``，四种原语全是闭式。
    **不迭代、不采样。** 答不出来（网格、带法兰的圆柱）当场抛。
    """

    rows = _rotation_rows(posed.rotation_xyzw)
    normal = half_space.unit_normal
    local_direction = tuple(
        sum(rows[row][axis] * normal[row] for row in range(3)) for axis in range(3)
    )
    offset = _dot(_sub(posed.translation_mm, half_space.point_mm), normal)
    return offset + _lowest_support_mm(posed.body.collision.shape, local_direction)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 距离场：协议，不是import（决策0090第三节）
# --------------------------------------------------------------------------


class SignedDistanceSource(Protocol):
    """可查询的有符号距离场。**`contact.field.SignedDistanceField`逐条满足它。**

    四样是承重的：``spacing_mm``（偏差估计要它）、``contains_stencil``
    （先问再算，窄带外不猜）、``value_mm``、``hessian_per_mm``（偏差主项要迹）。

    **场按体的局部系烘。** 于是体的位姿动了场跟着动，而距离是刚体运动不变量、
    Hessian的迹是旋转不变量——**偏差估计因此与位姿无关**，这不是巧合，是选迹的理由之一。
    """

    spacing_mm: float

    def contains_stencil(self, point_mm: Vector3) -> bool: ...

    def value_mm(self, point_mm: Vector3) -> float: ...

    def hessian_per_mm(self, point_mm: Vector3) -> tuple[Vector3, Vector3, Vector3]: ...


_SOURCE_MEMBERS: tuple[str, ...] = (
    "spacing_mm",
    "contains_stencil",
    "value_mm",
    "hessian_per_mm",
)


@dataclass(frozen=True)
class NarrowPhaseResult:
    """一次窄相查询的结果。**精度声明是字段不是注释。**

    * ``separation_mm``——``< 0``时其绝对值即穿透深度；
    * ``estimated_bias_mm``——``None``表示精确到舍入；非``None``时是
      ``(h²/6)·tr(∇²φ)``，0085那条``S = φ + (h²/6)·∇²φ + O(h⁴)``的直接读数。
      **它是主项估计不是误差上界**：``O(h⁴)``余项没有被它覆盖；
    * ``resolution_mm``——场的节点间距；``None``表示这条路上没有分辨率这回事。
    """

    separation_mm: float
    confidence: Literal["narrow_phase", "sampled_field"]
    estimated_bias_mm: float | None = None
    resolution_mm: float | None = None


def field_separation_mm(
    field: SignedDistanceSource, posed: PosedBody, probe_centre_mm: Vector3, radius_mm: float
) -> NarrowPhaseResult:
    """球型探针对一个烘好的场的分离量，带偏差估计。

    **只接球型探针**（点＋半径），理由在决策0090第2.3节第一行：
    ``φ(c) − r``读成穿透深度**只对球成立**（球的Minkowski和还是球）；
    一般形状取``min φ``给的是面到面的距离，不是最小平移穿透深度。
    非球探针在这里当场拒，而不是取几个采样点的min——**那个min会静默漏接触**。

    窄带外**失败关闭**（0085第四节）：稀疏块表里"远在体外"与"深在体内"
    长得一模一样，外推等于在"不接触"与"接触力很大"之间猜一个。
    """

    local = world_to_local_mm(posed, probe_centre_mm)
    if not field.contains_stencil(local):
        raise CollisionQueryError(
            f"probe at {probe_centre_mm!r} (local {local!r}) falls outside the baked "
            "narrow band — 窄带外失败关闭而不是外推（0085第四节）：那里'远在体外'"
            "与'深在体内'的信号一模一样"
        )
    value = field.value_mm(local)
    curvature = field.hessian_per_mm(local)
    spacing = field.spacing_mm
    trace = curvature[0][0] + curvature[1][1] + curvature[2][2]
    return NarrowPhaseResult(
        separation_mm=value - radius_mm,
        confidence="sampled_field",
        estimated_bias_mm=spacing * spacing * trace / 6.0,
        resolution_mm=spacing,
    )


# --------------------------------------------------------------------------
# 形对形窄相的路由
# --------------------------------------------------------------------------


def _as_world_ball(posed: PosedBody) -> tuple[Vector3, float] | None:
    """球族→世界系(球心, 半径)；其余族返回``None``。

    **胶囊不是球**：它的Minkowski和不是球，``φ(c) − r``那条读法对它不成立。
    """

    shape = _unwrap(posed.body.collision.shape)
    if isinstance(shape, Sphere):
        return (posed.translation_mm, shape.radius_mm)
    return None


def _axis_aligned_boxes(
    posed_a: PosedBody, posed_b: PosedBody
) -> tuple[RoundedBox, RoundedBox] | None:
    """两个都是**姿态为单位四元数**的圆角盒时返回它们，否则``None``。

    旋转盒对旋转盒没有闭式，而本模块不做迭代——**退回`broad_phase`比给个近似诚实**。
    """

    identity = (0.0, 0.0, 0.0, 1.0)
    if posed_a.rotation_xyzw != identity or posed_b.rotation_xyzw != identity:
        return None
    shape_a = _unwrap(posed_a.body.collision.shape)
    shape_b = _unwrap(posed_b.body.collision.shape)
    if isinstance(shape_a, RoundedBox) and isinstance(shape_b, RoundedBox):
        return (shape_a, shape_b)
    return None


def _axis_aligned_box_separation_mm(
    posed_a: PosedBody, box_a: RoundedBox, posed_b: PosedBody, box_b: RoundedBox
) -> float:
    """轴对齐圆角盒对的**精确**分离量。

    逐轴间隙``g_i``；有一轴分开就是``sqrt(Σ max(g_i,0)²)``，全轴重叠就是``max_i g_i``
    （＝最小重叠的负值，也就是最小平移穿透深度）。两个圆角在末尾各减一次，
    理由与`_rounded_box_signed_distance_mm`同：``(A⊕B_fa)``与``(B⊕B_fb)``之间的
    有符号距离＝核心盒之间的减去``fa + fb``。
    """

    gaps = []
    for axis in range(3):
        centre_gap = abs(posed_b.translation_mm[axis] - posed_a.translation_mm[axis])
        gaps.append(
            centre_gap - box_a.half_extents_mm[axis] - box_b.half_extents_mm[axis]
        )
    fillets = box_a.fillet_radius_mm + box_b.fillet_radius_mm
    if any(gap > 0.0 for gap in gaps):
        return math.sqrt(sum(max(gap, 0.0) ** 2 for gap in gaps)) - fillets
    return max(gaps) - fillets


def _try_narrow_phase(
    posed_a: PosedBody,
    posed_b: PosedBody,
    fields: dict[str, SignedDistanceSource],
) -> NarrowPhaseResult | None:
    """球/胶囊族**之外**的窄相路由。答不出来返回``None``（由调用方决定抛还是降级）。

    次序是承重的：**球/胶囊族那条路不走这里**，它原样留在
    `check_state_with_stats`的最前面，于是"逐位不变"是结构保证的。
    """

    field_a = fields.get(posed_a.body.body_id)
    field_b = fields.get(posed_b.body.body_id)
    if field_a is not None and field_b is not None:
        return None  # 两个都是场：谁当探针没有答案，不猜
    if field_a is not None or field_b is not None:
        field_body, field, probe_body = (
            (posed_a, field_a, posed_b) if field_a is not None else (posed_b, field_b, posed_a)
        )
        ball = _as_world_ball(probe_body)
        if ball is None:
            return None
        return field_separation_mm(field, field_body, ball[0], ball[1])  # type: ignore[arg-type]

    for probe, target in ((posed_a, posed_b), (posed_b, posed_a)):
        ball = _as_world_ball(probe)
        if ball is None:
            continue
        if _as_world_ball(target) is not None:
            return None  # 球对球走既有那条路，不在这里重算
        try:
            distance = posed_signed_distance_mm(target, ball[0])
        except CollisionQueryError:
            return None
        return NarrowPhaseResult(separation_mm=distance - ball[1], confidence="narrow_phase")

    boxes = _axis_aligned_boxes(posed_a, posed_b)
    if boxes is not None:
        return NarrowPhaseResult(
            separation_mm=_axis_aligned_box_separation_mm(
                posed_a, boxes[0], posed_b, boxes[1]
            ),
            confidence="narrow_phase",
        )
    return None


def narrow_phase_separation_mm(
    posed_a: PosedBody,
    posed_b: PosedBody,
    *,
    distance_fields: dict[str, SignedDistanceSource] | None = None,
) -> NarrowPhaseResult:
    """两个位姿好的体之间的窄相分离量。**答不出来当场抛。**

    这是直查面：它**没有"不知道"这个返回值**，所以降级在这里是不允许的
    （决策0090第六节）。要"不知道"就走`BroadPhaseCollisionQuery`。
    """

    fields = dict(distance_fields or {})
    segment_a = _as_world_segment(posed_a)
    segment_b = _as_world_segment(posed_b)
    if (
        segment_a is not None
        and segment_b is not None
        and posed_a.body.body_id not in fields
        and posed_b.body.body_id not in fields
    ):
        return NarrowPhaseResult(
            separation_mm=_segment_pair_separation_mm(segment_a, segment_b),
            confidence="narrow_phase",
        )
    result = _try_narrow_phase(posed_a, posed_b, fields)
    if result is None:
        raise CollisionQueryError(
            f"no exact narrow phase for the pair "
            f"({posed_a.body.body_id!r}, {posed_b.body.body_id!r}): "
            "旋转盒对盒、胶囊对盒、圆柱对圆柱、带法兰的圆柱、没有配场的MeshAsset "
            "都落在这一档（决策0090第七节GAP第二行）。"
            "要一个'不知道'而不是异常，走BroadPhaseCollisionQuery"
        )
    return result


@dataclass(frozen=True)
class CollisionEvent:
    body_a: str
    body_b: str
    confidence: Literal["broad_phase", "narrow_phase", "sampled_field"]
    penetration_mm: float | None
    aabb_a_mm: Aabb
    aabb_b_mm: Aabb
    #: 场那条路的偏差主项估计，mm。**``None``表示这个数精确到舍入。**
    estimated_bias_mm: float | None = None
    #: 场的节点间距，mm。``None``表示这条路上没有分辨率这回事。
    resolution_mm: float | None = None


@dataclass(frozen=True)
class CollisionQueryResult:
    """一次查询的事件与**确定性工作量计数**。

    墙钟不进门；候选对数、AABB重叠数和窄相调用数是跨机器稳定的整数，
    可以回答“检测有没有真的裁掉响应工作”，而不把宿主负载当功能结果。
    """

    events: tuple[CollisionEvent, ...]
    candidate_pair_count: int
    broad_phase_overlap_count: int
    narrow_phase_check_count: int


def _overlaps(a: Aabb, b: Aabb) -> bool:
    (al, ah), (bl, bh) = a, b
    return all(al[axis] <= bh[axis] and bl[axis] <= ah[axis] for axis in range(3))


class BroadPhaseCollisionQuery:
    """AABB broad phase；可选候选池按声明次序查询。

    ``candidate_pairs=None``逐字保留原来的全体两两语义。显式候选池用于
    ``FinalizedScene.contact_pairs``：候选身份在装配期冻结，活动与否每步重算。
    """

    def __init__(
        self,
        bodies: tuple[PosedBody, ...],
        *,
        allowed_pairs: frozenset[frozenset[str]] = frozenset(),
        candidate_pairs: tuple[tuple[str, str], ...] | None = None,
        distance_fields: dict[str, SignedDistanceSource] | None = None,
    ) -> None:
        identifiers = [posed.body.body_id for posed in bodies]
        if len(set(identifiers)) != len(identifiers):
            raise ShapeError("duplicate body_id in collision scene")
        known = set(identifiers)
        self._fields: dict[str, SignedDistanceSource] = {}
        for body_id, source in (distance_fields or {}).items():
            if body_id not in known:
                raise ShapeError(
                    f"distance field references unknown body: {body_id!r}; "
                    f"known bodies are {sorted(known)}"
                )
            missing = [name for name in _SOURCE_MEMBERS if not hasattr(source, name)]
            if missing:
                raise ShapeError(
                    f"distance field for {body_id!r} is not a SignedDistanceSource: "
                    f"missing {missing}（决策0090第三节：基座定协议、物理域给实现）"
                )
            self._fields[body_id] = source
        for pair in allowed_pairs:
            if len(pair) != 2 or not pair <= known:
                raise ShapeError(f"allowed pair references unknown bodies: {sorted(pair)}")
        self._bodies = bodies
        if candidate_pairs is None:
            self._candidate_pairs = tuple(
                (name_a, name_b)
                for index, name_a in enumerate(identifiers)
                for name_b in identifiers[index + 1 :]
                if frozenset((name_a, name_b)) not in allowed_pairs
            )
        else:
            checked: list[tuple[str, str]] = []
            seen: set[frozenset[str]] = set()
            for pair in candidate_pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ShapeError(
                        f"candidate pair must be a two-item tuple of body ids: {pair!r}"
                    )
                body_a, body_b = pair
                if any(
                    not isinstance(body_id, str) or not body_id
                    for body_id in (body_a, body_b)
                ):
                    raise ShapeError(
                        "candidate pair body ids must be nonempty strings: "
                        f"{pair!r}"
                    )
                if body_a == body_b:
                    raise ShapeError(
                        f"candidate pair must name two distinct bodies: {body_a!r}"
                    )
                members = frozenset((body_a, body_b))
                unknown = sorted(members - known)
                if unknown:
                    raise ShapeError(
                        f"candidate pair references unknown bodies: {unknown}; "
                        f"known bodies are {sorted(known)}"
                    )
                if members in seen:
                    raise ShapeError(
                        f"candidate pair ({body_a!r}, {body_b!r}) is declared twice"
                    )
                if members in allowed_pairs:
                    raise ShapeError(
                        f"pair ({body_a!r}, {body_b!r}) is both a candidate and an "
                        "allowed pair — one asks for a query and the other suppresses it"
                    )
                seen.add(members)
                checked.append((body_a, body_b))
            self._candidate_pairs = tuple(checked)

    @property
    def candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        """真正会被检查的候选池；次序即声明次序。"""

        return self._candidate_pairs

    def check_state(self) -> tuple[CollisionEvent, ...]:
        return self.check_state_with_stats().events

    def check_state_with_stats(self) -> CollisionQueryResult:
        """查询并返回事件以及候选/broad/narrow三段确定性计数。"""

        posed = {body.body.body_id: body for body in self._bodies}
        boxes = {name: body.world_aabb_mm() for name, body in posed.items()}
        segments: dict[str, tuple[Vector3, Vector3, float] | None] = {}
        events: list[CollisionEvent] = []
        broad_phase_overlaps = 0
        narrow_phase_checks = 0
        for name_a, name_b in self._candidate_pairs:
            box_a, box_b = boxes[name_a], boxes[name_b]
            if not _overlaps(box_a, box_b):
                continue
            broad_phase_overlaps += 1
            if name_a not in segments:
                segments[name_a] = _as_world_segment(posed[name_a])
            if name_b not in segments:
                segments[name_b] = _as_world_segment(posed[name_b])
            segment_a = segments[name_a]
            segment_b = segments[name_b]
            has_field = name_a in self._fields or name_b in self._fields
            if segment_a is not None and segment_b is not None and not has_field:
                narrow_phase_checks += 1
                separation = _segment_pair_separation_mm(segment_a, segment_b)
                if separation >= 0.0:
                    continue  # broad命中但narrow判分离：假阳性，不报
                events.append(
                    CollisionEvent(
                        body_a=name_a,
                        body_b=name_b,
                        confidence="narrow_phase",
                        penetration_mm=-separation,
                        aabb_a_mm=box_a,
                        aabb_b_mm=box_b,
                    )
                )
                continue
            #: 球/胶囊族之外的路由。**答不出来降级而不是抛**（决策0090第六节）：
            #: 场景查询有"不知道"这个返回值，让一个查不了的对炸掉整场
            #: 会破掉"不漏报"这条唯一硬承诺。窄带外那一条同理接住。
            try:
                extended = _try_narrow_phase(posed[name_a], posed[name_b], self._fields)
            except CollisionQueryError:
                extended = None
            if extended is not None:
                narrow_phase_checks += 1
                if extended.separation_mm >= 0.0:
                    continue
                events.append(
                    CollisionEvent(
                        body_a=name_a,
                        body_b=name_b,
                        confidence=extended.confidence,
                        penetration_mm=-extended.separation_mm,
                        aabb_a_mm=box_a,
                        aabb_b_mm=box_b,
                        estimated_bias_mm=extended.estimated_bias_mm,
                        resolution_mm=extended.resolution_mm,
                    )
                )
                continue
            events.append(
                CollisionEvent(
                    body_a=name_a,
                    body_b=name_b,
                    confidence="broad_phase",
                    penetration_mm=None,
                    aabb_a_mm=box_a,
                    aabb_b_mm=box_b,
                )
            )
        return CollisionQueryResult(
            events=tuple(events),
            candidate_pair_count=len(self._candidate_pairs),
            broad_phase_overlap_count=broad_phase_overlaps,
            narrow_phase_check_count=narrow_phase_checks,
        )


__all__ = [
    "BroadPhaseCollisionQuery",
    "CollisionEvent",
    "CollisionQueryError",
    "CollisionQueryResult",
    "HalfSpace",
    "NarrowPhaseResult",
    "SignedDistanceSource",
    "field_separation_mm",
    "half_space_separation_mm",
    "narrow_phase_separation_mm",
    "posed_signed_distance_mm",
    "segment_segment_distance_mm",
    "shape_signed_distance_mm",
    "world_to_local_mm",
]
