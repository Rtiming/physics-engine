"""解析原语的质量属性与rounded-core SDF——spec/12第二节、spec/11规则2的首个实现。

**只做几何量，不做力学**：本模块给体积、质心、绕质心的惯量张量，
以及`φ(x) = φ_core(x) − r_f`（spec/11规则2点名、至今无实现）的符号距离。
不解方程、不推进时间——那是spec/12第四节的事。
惯量是力学域接进来第一个撞上的缺口（plans/02第一节资产圈）：
`shapes.py`至今只有可选的`mass_kg`，没有惯量张量。

三条边界写在最前面，因为它们都是spec/14第五节说的"静默1000倍"级风险：

1. **惯量张量绕质心、在形状的局部轴下表达**，单位kg·mm²。参考点写进字段名
   （`inertia_about_centroid_kg_mm2`），换参考点必须显式调
   :func:`shift_inertia_kg_mm2`（平行轴定理），没有隐式路径。
2. **长度mm、质量kg、密度kg/m³**。`_kg_m3`是spec/14第五节点名的两制通用SI
   复合后缀；由它到mm制只用到mm³↔m³这个**纯长度量纲**换算（1e-9），
   正是spec/14第五节登记过的那一条，不是本模块自己猜的因子。
3. **不猜没声明的几何**：带法兰的`FiniteCylinder`没有法兰轴向尺寸字段、
   `MeshAsset`的网格字节引擎从不读——两者一律失败关闭，不拿AABB冒充质量分布
   （AGENTS.md诚实可信度条款：不知道就说不知道）。

数值形态按0016：纯标准库、无状态纯函数、返回值是显式数组（元组），
没有可变几何对象，加速档要接就是把同一批闭式解换个数值命名空间算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.shapes import (
    Capsule,
    FiniteCylinder,
    GeneratedShape,
    MeshAsset,
    RoundedBox,
    Shape,
    Sphere,
    Vector3,
)


class GeometryError(ValueError):
    """几何量的失败关闭。

    与`ShapeError`分开是有意的：**形状声明合法不等于质量属性可算**。
    带法兰的圆柱是一条完全合法的碰撞形声明，但它的质量分布没被声明。
    """


Matrix3 = tuple[Vector3, Vector3, Vector3]

#: mm³/m³。spec/14第五节的换算表只登记纯长度量纲，mm³↔m³正在其中。
MM3_PER_M3 = 1.0e9


@dataclass(frozen=True)
class MassProperties:
    """一个形状的质量属性。**惯量绕质心、在局部轴下、单位kg·mm²。**"""

    volume_mm3: float
    centroid_mm: Vector3
    mass_kg: float
    inertia_about_centroid_kg_mm2: Matrix3


def _require_positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise GeometryError(f"{name} must be positive and finite: {value!r}")
    return float(value)


def _require_finite_point(point: Vector3, name: str) -> Vector3:
    if len(point) != 3 or not all(math.isfinite(component) for component in point):
        raise GeometryError(f"{name} must be a finite 3-vector: {point!r}")
    return (float(point[0]), float(point[1]), float(point[2]))


def _unwrap(shape: Shape) -> Shape:
    """生成器只是身份包装（spec/11规则2），质量属性看它产出的那个形。"""

    return shape.shape if isinstance(shape, GeneratedShape) else shape


def _reject_unsupported(shape: Shape) -> None:
    if isinstance(shape, MeshAsset):
        raise GeometryError(
            "mesh assets carry no mass model: the engine never reads mesh bytes "
            "and the declared AABB is an envelope, not a mass distribution"
        )
    if isinstance(shape, FiniteCylinder) and shape.flange_outer_radius_mm is not None:
        raise GeometryError(
            "a flanged FiniteCylinder has no declared flange width, so its volume "
            "is not determined by the declaration; declare the flange as its own "
            "shape or drop flange_outer_radius_mm before asking for mass properties"
        )
    if not isinstance(shape, Sphere | Capsule | RoundedBox | FiniteCylinder):
        raise GeometryError(f"no analytic mass model for {type(shape).__name__}")


def _diagonal(xx: float, yy: float, zz: float) -> Matrix3:
    return ((xx, 0.0, 0.0), (0.0, yy, 0.0), (0.0, 0.0, zz))


def _axisymmetric(axial: float, transverse: float, axis: Vector3) -> Matrix3:
    """轴对称体的张量：`I = transverse·E + (axial − transverse)·n⊗n`。

    对`v ∥ n`给`axial`、对`v ⊥ n`给`transverse`——这是把"沿轴/垂轴两个数"
    抬回任意方向局部轴的唯一正确写法，不是绕轴排列对角元。
    """

    delta = axial - transverse
    return tuple(  # type: ignore[return-value]
        tuple(
            (transverse if i == j else 0.0) + delta * axis[i] * axis[j] for j in range(3)
        )
        for i in range(3)
    )


def _capsule_axis(shape: Capsule) -> tuple[float, Vector3]:
    """返回（轴长L，单位轴向n）。L=0时轴向任取——退化成球，张量与n无关。"""

    delta = tuple(b - a for a, b in zip(shape.point_a_mm, shape.point_b_mm, strict=True))
    length = math.sqrt(sum(component * component for component in delta))
    if length <= 0.0:
        return 0.0, (0.0, 0.0, 1.0)
    return length, (delta[0] / length, delta[1] / length, delta[2] / length)


def _sphere_shape_mm5(radius: float, volume: float) -> Matrix3:
    value = 0.4 * volume * radius * radius
    return _diagonal(value, value, value)


def _capsule_shape_mm5(shape: Capsule) -> Matrix3:
    """圆柱段 + 两个半球帽，各自绕**胶囊质心**，再按轴向抬回局部轴。

    半球绕自身质心的横向矩是`(83/320)·m·r²`（`(2/5)m r²`减去质心偏移`3r/8`
    的平行轴项），帽心到胶囊质心的距离是`L/2 + 3r/8`。
    L→0时两式合并回`(2/5)m r²`——退化即球，这条在测试里是硬判据。
    """

    length, axis = _capsule_axis(shape)
    radius = shape.radius_mm
    cylinder = math.pi * radius * radius * length
    caps = 4.0 * math.pi * radius**3 / 3.0

    axial = 0.5 * cylinder * radius * radius + 0.4 * caps * radius * radius
    offset = 0.5 * length + 0.375 * radius
    transverse = (
        cylinder * (3.0 * radius * radius + length * length) / 12.0
        + (83.0 / 320.0) * caps * radius * radius
        + caps * offset * offset
    )
    return _axisymmetric(axial, transverse, axis)


def _cylinder_shape_mm5(shape: FiniteCylinder) -> Matrix3:
    radius = shape.radius_mm
    half_width = shape.half_width_mm
    volume = 2.0 * math.pi * radius * radius * half_width
    axial = 0.5 * volume * radius * radius
    transverse = volume * (3.0 * radius * radius + 4.0 * half_width * half_width) / 12.0
    return _diagonal(transverse, transverse, axial)


def _rounded_box_second_moments(shape: RoundedBox) -> tuple[float, Vector3]:
    """圆角盒 = 盒⊕球（spec/11规则2的`φ_core − r_f`正是这件事）。

    按互不重叠的四族积起来：核心盒 + 6个面板 + 12个四分之一圆柱 + 8个角八分球。
    返回（体积，`(∫x²dV, ∫y²dV, ∫z²dV)`）——惯量由`Ixx = Jy + Jz`得到，
    这样三个方向只写一遍公式，写错一处三条门会同时红。

    体积那一支是Steiner公式`V + S·r + πr²·Σ边长 + (4/3)πr³`的展开，
    可独立核对；`r→0`退化成盒、`h→0`退化成球、`hx,hy→0`退化成胶囊，
    三条退化在测试里都是判据。
    """

    hx, hy, hz = shape.half_extents_mm
    r = shape.fillet_radius_mm
    pi = math.pi

    volume = 0.0
    jx = jy = jz = 0.0

    # 核心盒
    core = 8.0 * hx * hy * hz
    volume += core
    jx += core * hx * hx / 3.0
    jy += core * hy * hy / 3.0
    jz += core * hz * hz / 3.0

    # 6个面板（成对处理：±方向的x_c²相同）
    for half, other_a, other_b, index in (
        (hx, hy, hz, 0),
        (hy, hz, hx, 1),
        (hz, hx, hy, 2),
    ):
        pair = 8.0 * r * other_a * other_b
        volume += pair
        along = pair * ((half + 0.5 * r) ** 2 + r * r / 12.0)
        across_a = pair * other_a * other_a / 3.0
        across_b = pair * other_b * other_b / 3.0
        contribution = [0.0, 0.0, 0.0]
        contribution[index] = along
        contribution[(index + 1) % 3] = across_a
        contribution[(index + 2) % 3] = across_b
        jx += contribution[0]
        jy += contribution[1]
        jz += contribution[2]

    # 12个四分之一圆柱：沿每个轴各4根，长度2·该轴半长
    quarter_area = 0.25 * pi * r * r
    for half, other_a, other_b, index in (
        (hx, hy, hz, 0),
        (hy, hz, hx, 1),
        (hz, hx, hy, 2),
    ):
        volume += 4.0 * quarter_area * 2.0 * half
        along = 4.0 * quarter_area * 2.0 * half**3 / 3.0
        across_a = 8.0 * half * (
            other_a * other_a * quarter_area
            + 2.0 * other_a * r**3 / 3.0
            + pi * r**4 / 16.0
        )
        across_b = 8.0 * half * (
            other_b * other_b * quarter_area
            + 2.0 * other_b * r**3 / 3.0
            + pi * r**4 / 16.0
        )
        contribution = [0.0, 0.0, 0.0]
        contribution[index] = along
        contribution[(index + 1) % 3] = across_a
        contribution[(index + 2) % 3] = across_b
        jx += contribution[0]
        jy += contribution[1]
        jz += contribution[2]

    # 8个角八分球（合起来正好一个整球）
    octant = pi * r**3 / 6.0
    volume += 8.0 * octant
    for half, target in ((hx, "x"), (hy, "y"), (hz, "z")):
        moment = 8.0 * (half * half * octant + 2.0 * half * pi * r**4 / 16.0 + pi * r**5 / 30.0)
        if target == "x":
            jx += moment
        elif target == "y":
            jy += moment
        else:
            jz += moment

    return volume, (jx, jy, jz)


def volume_mm3(shape: Shape) -> float:
    """解析体积。圆角盒走Steiner展开，其余是教科书闭式解。"""

    inner = _unwrap(shape)
    _reject_unsupported(inner)
    if isinstance(inner, Sphere):
        return 4.0 * math.pi * inner.radius_mm**3 / 3.0
    if isinstance(inner, Capsule):
        length, _ = _capsule_axis(inner)
        radius = inner.radius_mm
        return math.pi * radius * radius * length + 4.0 * math.pi * radius**3 / 3.0
    if isinstance(inner, FiniteCylinder):
        return 2.0 * math.pi * inner.radius_mm**2 * inner.half_width_mm
    volume, _ = _rounded_box_second_moments(inner)  # type: ignore[arg-type]
    return volume


def centroid_mm(shape: Shape) -> Vector3:
    """局部坐标下的质心。三种原语在原点，胶囊在两端点中点。"""

    inner = _unwrap(shape)
    _reject_unsupported(inner)
    if isinstance(inner, Capsule):
        return tuple(  # type: ignore[return-value]
            0.5 * (a + b)
            for a, b in zip(inner.point_a_mm, inner.point_b_mm, strict=True)
        )
    return (0.0, 0.0, 0.0)


def _inertia_shape_mm5(shape: Shape) -> Matrix3:
    """单位密度下绕质心的惯量（mm⁵）。质量在外面一次性乘进去。"""

    if isinstance(shape, Sphere):
        return _sphere_shape_mm5(shape.radius_mm, volume_mm3(shape))
    if isinstance(shape, Capsule):
        return _capsule_shape_mm5(shape)
    if isinstance(shape, FiniteCylinder):
        return _cylinder_shape_mm5(shape)
    _volume, (jx, jy, jz) = _rounded_box_second_moments(shape)  # type: ignore[arg-type]
    return _diagonal(jy + jz, jz + jx, jx + jy)


def mass_properties(
    shape: Shape,
    *,
    density_kg_m3: float | None = None,
    mass_kg: float | None = None,
) -> MassProperties:
    """体积/质心/绕质心惯量。密度与质量**恰给一个**，给两个或都不给即拒。

    两个都收是因为两条真实来路都存在：材料记录带`density_kg_m3`（spec/14），
    而`SimBody.mass_kg`是直接称出来的。两个同时给会产生一个**没人会去核对的
    隐含体积**，所以按失败关闭处理，不做"以质量为准"这种静默取舍。
    """

    inner = _unwrap(shape)
    _reject_unsupported(inner)
    if (density_kg_m3 is None) == (mass_kg is None):
        raise GeometryError("give exactly one of density_kg_m3 or mass_kg")

    volume = volume_mm3(inner)
    if density_kg_m3 is not None:
        density = _require_positive_finite(density_kg_m3, "density_kg_m3")
        mass = density * volume / MM3_PER_M3
    else:
        mass = _require_positive_finite(mass_kg, "mass_kg")  # type: ignore[arg-type]

    scale = mass / volume
    shape_mm5 = _inertia_shape_mm5(inner)
    inertia = tuple(tuple(scale * value for value in row) for row in shape_mm5)
    return MassProperties(
        volume_mm3=volume,
        centroid_mm=centroid_mm(inner),
        mass_kg=mass,
        inertia_about_centroid_kg_mm2=inertia,  # type: ignore[arg-type]
    )


def shift_inertia_kg_mm2(
    inertia_about_centroid_kg_mm2: Matrix3,
    mass_kg: float,
    offset_mm: Vector3,
) -> Matrix3:
    """平行轴定理：`I_P = I_cm + m(|d|²E − d⊗d)`。

    `offset_mm`是质心与新参考点之间的位移。**方向无关**——`|d|²`与`d⊗d`
    对`d`都是偶的，所以传`d`与传`−d`结果相同，这里没有可以搞反的符号。
    反过来（从某点搬回质心）要传负号是错的，用往返一次即可自检。
    """

    mass = _require_positive_finite(mass_kg, "mass_kg")
    offset = _require_finite_point(offset_mm, "offset_mm")
    squared = sum(component * component for component in offset)
    return tuple(  # type: ignore[return-value]
        tuple(
            inertia_about_centroid_kg_mm2[i][j]
            + mass * ((squared if i == j else 0.0) - offset[i] * offset[j])
            for j in range(3)
        )
        for i in range(3)
    )


def _point_segment_distance_mm(point: Vector3, start: Vector3, end: Vector3) -> float:
    edge = tuple(b - a for a, b in zip(start, end, strict=True))
    squared = sum(component * component for component in edge)
    if squared <= 0.0:
        return math.sqrt(sum((p - a) ** 2 for p, a in zip(point, start, strict=True)))
    t = sum(
        (p - a) * e for p, a, e in zip(point, start, edge, strict=True)
    ) / squared
    t = min(1.0, max(0.0, t))
    return math.sqrt(
        sum((p - (a + t * e)) ** 2 for p, a, e in zip(point, start, edge, strict=True))
    )


def sdf_mm(shape: Shape, point_mm: Vector3) -> float:
    """局部坐标下的rounded-core符号距离：`φ(x) = φ_core(x) − r_f`（spec/11规则2）。

    四种原语都是**精确**符号距离（不是保守下界）：球的核是一个点、
    胶囊的核是线段、圆角盒的核是盒、有限圆柱的核是它自己（`r_f = 0`）。
    体内为负、体外为正、体表为零。
    """

    inner = _unwrap(shape)
    _reject_unsupported(inner)
    point = _require_finite_point(point_mm, "point_mm")

    if isinstance(inner, Sphere):
        return math.sqrt(sum(c * c for c in point)) - inner.radius_mm
    if isinstance(inner, Capsule):
        core = _point_segment_distance_mm(point, inner.point_a_mm, inner.point_b_mm)
        return core - inner.radius_mm
    if isinstance(inner, RoundedBox):
        q = tuple(
            abs(c) - h for c, h in zip(point, inner.half_extents_mm, strict=True)
        )
        outside = math.sqrt(sum(max(component, 0.0) ** 2 for component in q))
        return outside + min(max(q[0], q[1], q[2]), 0.0) - inner.fillet_radius_mm
    radial = math.hypot(point[0], point[1]) - inner.radius_mm  # type: ignore[union-attr]
    axial = abs(point[2]) - inner.half_width_mm  # type: ignore[union-attr]
    outside = math.hypot(max(radial, 0.0), max(axial, 0.0))
    return outside + min(max(radial, axial), 0.0)


__all__ = [
    "MM3_PER_M3",
    "GeometryError",
    "MassProperties",
    "Matrix3",
    "centroid_mm",
    "mass_properties",
    "sdf_mm",
    "shift_inertia_kg_mm2",
    "volume_mm3",
]
