"""三角网格的质量属性——散度定理的四面体法（决策0074阶段四第一片、能力位S3.3）。

`geometry.py`给的是**解析原语**的质量属性（球、胶囊、圆角盒、有限圆柱四种闭式解）。
本模块给的是另一头：**一堆三角形**的体积、质心、绕质心惯量张量。
两者的关系是判据关系——本模块的验收方式就是把解析原语三角化后与`geometry.py`对拍，
误差随三角化加密按阶收敛。**两条独立的腿，谁也不是谁的实现。**

## 形制：多面体上这不是数值积分，是**闭式解**

对每个三角形``(a, b, c)``取以原点为顶的四面体``(0, a, b, c)``，
带符号地累加它的体积与二阶矩。散度定理保证：闭合定向网格上这些带符号量
**恰好**加成多面体的真值——不是逼近，是恒等式（Mirtich 1996那一族的做法）。

于是本模块的误差只有两处来源，**两处都不是积分误差**：

1. **三角化误差**：多面体不是球。这一项随三角化加密按阶收敛，是可测的；
2. **浮点求和误差**：远离原点的网格上求和会灾难性抵消。本模块用
   "先平移到参考点、算完再搬回"压住它（见`mesh_covariance_mm5`的``about_mm``）。

## 单位口径（照`geometry.py`，一个字不改）

长度mm、质量kg、密度kg/m³、惯量kg·mm²，绕**质心**、在网格自身的坐标轴下表达。
换参考点必须显式调`geometry.shift_inertia_kg_mm2`，没有隐式路径。
密度是入参：``mass = density_kg_m3 · volume_mm3 / MM3_PER_M3``，
那个1e-9是spec/14第五节登记过的纯长度量纲换算，不是本模块自己猜的因子。

## 失败关闭的三条（AGENTS.md诚实可信度条款：不知道就说不知道）

1. **网格不闭合就没有体积可言**。散度定理要的是闭合定向流形；开边界、
   非流形边（一条边被三个以上三角共用）、朝向自相矛盾（同一条**有向**边出现两次），
   三种都由`mesh_edge_defects`判出来，`mesh_mass_properties`见到即拒。
   **不拿一个算得出来的数冒充答案。**
2. **朝向反了体积变号**，于是`mesh_mass_properties`见到非正体积即拒并点名朝向。
   带符号体积是本模块唯一的朝向证人，所以`mesh_volume_mm3`**故意不取绝对值**。
3. **零面积三角**拒收。它对体积的贡献恒为0，所以数值上无害——
   拒它是因为它是坏网格的证据，不是因为它算错了。
   **近退化的薄三角不拒**：它们的贡献是对的，而"多薄算薄"本模块给不出有根据的阈值，
   给一个就是冒充。这条限制写在这里，不写在实现里。

## 脏网格走的不是这条路

0074第二节第3条实测：真实件有非流形边、无开放边界（多实体布尔没焊接的形态）。
**那种网格进不了本模块**，它们走的是SDF那条路（`contact/field.py`＋`tools/model/sdf_bake/`）——
广义缠绕数用体积判据定内外，不需要闭合流形。本模块与那条路是**两道题**，
本模块的失败关闭正是把这条分界钉在明处。

## 三角化器为什么住在这里

内核`dependencies = []`且**从不读网格字节**（`shapes.MeshAsset`那条纪律）。
于是仓内唯一能存在的三角网格是**算出来的**。三个三角化器（球/盒/圆柱）
是本模块唯一的语料来源，也是它验收判据的输入——没有它们，甲1一条判据都验不了。
它们产的都是**含原点的星形体**，于是朝向逐面按"面法向背离原点"定，
**不依赖体积的符号**——所以第2条那道朝向门不是自证的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.geometry import (
    MM3_PER_M3,
    GeometryError,
    MassProperties,
    Matrix3,
    shift_inertia_kg_mm2,
)
from physics_engine.shapes import FiniteCylinder, RoundedBox, Sphere, Vector3


class MeshError(GeometryError):
    """三角网格质量属性的失败关闭。

    继承`GeometryError`是有意的：调用方按"几何量算不出来"这一类接住即可，
    不必知道这个几何是解析原语还是一堆三角形。
    """


#: 规范四面体``(0, e1, e2, e3)``上的``∫x⊗x dV``的展开常数。
#: ``C_can = (1/120)·[[2,1,1],[1,2,1],[1,1,2]] = (1/120)·(E + 1)``，
#: 于是``J·C_can·Jᵀ = (1/120)·(a⊗a + b⊗b + c⊗c + s⊗s)``，``s = a+b+c``。
#: **把矩阵乘法化成四个外积**——这是本模块内层循环里唯一的代数化简，
#: 它让每个三角形只做常数次乘加，没有3×3矩阵对象。
_COVARIANCE_DENOMINATOR = 120.0


@dataclass(frozen=True)
class TriangleMesh:
    """一张三角网格：顶点表（mm）+ 三角形的顶点下标表。

    **朝向约定：三角形按右手法则给出的法向指向体外**（与本仓所有SDF
    "体外为正"的符号约定同源）。朝向反了不是一个可以容忍的输入，
    是一个必须被判出来的错误——见模块docstring第2条。

    形制按0016：不可变、纯数字、没有可变几何对象。
    """

    vertices_mm: tuple[Vector3, ...]
    triangles: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.vertices_mm) < 4:
            raise MeshError(
                f"a closed triangle mesh needs at least 4 vertices: {len(self.vertices_mm)}"
            )
        if len(self.triangles) < 4:
            raise MeshError(
                f"a closed triangle mesh needs at least 4 triangles: {len(self.triangles)}"
            )
        for index, vertex in enumerate(self.vertices_mm):
            if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
                raise MeshError(f"vertex {index} must be a finite 3-vector: {vertex!r}")
        count = len(self.vertices_mm)
        for face, triangle in enumerate(self.triangles):
            if len(triangle) != 3:
                raise MeshError(f"triangle {face} must have 3 indices: {triangle!r}")
            for index in triangle:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise MeshError(
                        f"triangle {face} index must be an int: {index!r}"
                    )
                if not 0 <= index < count:
                    raise MeshError(
                        f"triangle {face} index {index} outside [0, {count})"
                    )
            if len(set(triangle)) != 3:
                raise MeshError(
                    f"triangle {face} repeats a vertex index: {triangle!r} — "
                    "退化三角失败关闭，不拿一个恒为零的贡献冒充一个面"
                )

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    def translated(self, offset_mm: Vector3) -> TriangleMesh:
        """整体平移。平移轴定理那条恒等式的输入就是它。"""

        if len(offset_mm) != 3 or not all(math.isfinite(v) for v in offset_mm):
            raise MeshError(f"offset_mm must be a finite 3-vector: {offset_mm!r}")
        return TriangleMesh(
            vertices_mm=tuple(
                (
                    vertex[0] + offset_mm[0],
                    vertex[1] + offset_mm[1],
                    vertex[2] + offset_mm[2],
                )
                for vertex in self.vertices_mm
            ),
            triangles=self.triangles,
        )

    def flipped(self) -> TriangleMesh:
        """把每个三角形的朝向反过来。**朝向门的必红输入就是它。**"""

        return TriangleMesh(
            vertices_mm=self.vertices_mm,
            triangles=tuple((c, b, a) for a, b, c in self.triangles),
        )


def mesh_edge_defects(mesh: TriangleMesh) -> tuple[str, ...]:
    """闭合定向性的三类缺陷，空元组表示干净。

    判法只有一条：**每条无向边恰好被两个三角形共用，且这两次的方向相反。**
    等价说法是"每条有向边恰好出现一次"，本函数按后者数。三类缺陷各自的形态：

    * **开放边界**：有向边``(i, j)``出现，反向``(j, i)``没出现；
    * **非流形边**：同一条无向边被三个以上三角共用（0074实测：真实件有6条与10条）；
    * **朝向自相矛盾**：同一条**有向**边出现两次——两个相邻三角绕向相反。

    **三类都让散度定理失去前提**，所以它们是同一道门而不是三道。
    返回的是字符串清单而不是布尔，因为"网格哪里坏了"是调用方唯一能拿去修的东西。
    """

    seen: dict[tuple[int, int], int] = {}
    for a, b, c in mesh.triangles:
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1

    defects: list[str] = []
    for (i, j), times in sorted(seen.items()):
        if times != 1:
            defects.append(
                f"directed edge ({i}, {j}) appears {times} times — "
                "两个相邻三角绕向相反，或一条边被三个以上三角共用"
            )
    for (i, j), _times in sorted(seen.items()):
        if (j, i) not in seen:
            defects.append(
                f"edge ({i}, {j}) has no opposite — 开放边界，网格不闭合"
            )
    return tuple(defects)


def _require_closed(mesh: TriangleMesh) -> None:
    defects = mesh_edge_defects(mesh)
    if defects:
        head = "; ".join(defects[:3])
        more = f"（另有{len(defects) - 3}条）" if len(defects) > 3 else ""
        raise MeshError(
            "mesh is not a closed oriented manifold, so the divergence theorem "
            f"has no premise here: {head}{more} —— "
            "脏网格走的是SDF那条路（contact/field.py），不是本模块"
        )


def _require_nondegenerate(mesh: TriangleMesh) -> None:
    """零面积三角失败关闭。**只判恰好为零，不判'多薄算薄'。**

    薄三角的贡献是对的；"多薄算薄"本模块给不出有根据的阈值，
    给一个就是冒充（AGENTS.md诚实可信度条款）。
    """

    vertices = mesh.vertices_mm
    for face, (ia, ib, ic) in enumerate(mesh.triangles):
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        if cross == (0.0, 0.0, 0.0):
            raise MeshError(
                f"triangle {face} has zero area (vertices {ia}, {ib}, {ic} are "
                "collinear or coincident) — 它对体积的贡献恒为0，"
                "拒它是因为它是坏网格的证据，不是因为它算错了"
            )


@dataclass(frozen=True)
class MeshMoments:
    """网格关于某个参考点的原始矩。**这是本模块唯一真正做积分的东西。**

    ``reference_mm``是参考点；三个量都是相对它算的：

    * ``volume_mm3``：带符号体积（**平移不变**，参考点只影响求和的条件数）；
    * ``first_moment_mm4``：``∫(x − p)dV``，质心由``p + m1/V``得到；
    * ``covariance_mm5``：``∫(x − p)⊗(x − p)dV``。

    惯量由``I = tr(C)·E − C``得到——**三个方向只写一遍公式**
    （`geometry._rounded_box_second_moments`同一条理由：写错一处会同时红三条门）。
    """

    reference_mm: Vector3
    volume_mm3: float
    first_moment_mm4: Vector3
    covariance_mm5: Matrix3


def mesh_moments(mesh: TriangleMesh, *, about_mm: Vector3 | None = None) -> MeshMoments:
    """关于``about_mm``的带符号体积、一阶矩、二阶矩。**不校验闭合性**。

    ``about_mm``缺省取顶点坐标的算术平均。**这个缺省不是审美，是条件数**：
    远离原点的网格上，四面体的带符号体积是一串正负交替的大数，
    它们的和是一个小数——不先平移，有效位就在求和里掉光。
    显式传``(0.0, 0.0, 0.0)``可以拿到"关于原点"的原始矩，
    平移轴定理那条恒等式验的正是这两条路给出同一个答案。

    **不校验闭合性**是有意的：`mesh_edge_defects`是独立的一道门，
    而原始矩本身对任何一堆三角形都有定义（只是不再等于任何体的质量属性）。
    对外的入口`mesh_mass_properties`把两者串起来。
    """

    if about_mm is None:
        count = float(len(mesh.vertices_mm))
        about = (
            sum(vertex[0] for vertex in mesh.vertices_mm) / count,
            sum(vertex[1] for vertex in mesh.vertices_mm) / count,
            sum(vertex[2] for vertex in mesh.vertices_mm) / count,
        )
    else:
        if len(about_mm) != 3 or not all(math.isfinite(v) for v in about_mm):
            raise MeshError(f"about_mm must be a finite 3-vector: {about_mm!r}")
        about = (float(about_mm[0]), float(about_mm[1]), float(about_mm[2]))

    px, py, pz = about
    vertices = mesh.vertices_mm

    six_volume = 0.0
    m1x = m1y = m1z = 0.0
    #: 对称，只累加上三角六个分量——**内层循环里不建任何对象**（0083那一轮
    #: 实测：热点从来不在申报的那条路上，而每三角9次元组构造是真的会出现在profile里）。
    cxx = cyy = czz = cxy = cxz = cyz = 0.0

    for ia, ib, ic in mesh.triangles:
        va, vb, vc = vertices[ia], vertices[ib], vertices[ic]
        ax, ay, az = va[0] - px, va[1] - py, va[2] - pz
        bx, by, bz = vb[0] - px, vb[1] - py, vb[2] - pz
        cx, cy, cz = vc[0] - px, vc[1] - py, vc[2] - pz

        #: ``det[a b c] = a·(b×c)``，等于四面体``(0,a,b,c)``带符号体积的6倍。
        det = (
            ax * (by * cz - bz * cy)
            - ay * (bx * cz - bz * cx)
            + az * (bx * cy - by * cx)
        )
        six_volume += det

        sx, sy, sz = ax + bx + cx, ay + by + cy, az + bz + cz
        #: ``∫x dV = det·s/24``（四面体质心在``s/4``、体积``det/6``）。
        m1x += det * sx
        m1y += det * sy
        m1z += det * sz

        #: ``∫x⊗x dV = (det/120)·(a⊗a + b⊗b + c⊗c + s⊗s)``——模块顶部那条化简。
        cxx += det * (ax * ax + bx * bx + cx * cx + sx * sx)
        cyy += det * (ay * ay + by * by + cy * cy + sy * sy)
        czz += det * (az * az + bz * bz + cz * cz + sz * sz)
        cxy += det * (ax * ay + bx * by + cx * cy + sx * sy)
        cxz += det * (ax * az + bx * bz + cx * cz + sx * sz)
        cyz += det * (ay * az + by * bz + cy * cz + sy * sz)

    volume = six_volume / 6.0
    first = (m1x / 24.0, m1y / 24.0, m1z / 24.0)
    scale = 1.0 / _COVARIANCE_DENOMINATOR
    covariance = (
        (cxx * scale, cxy * scale, cxz * scale),
        (cxy * scale, cyy * scale, cyz * scale),
        (cxz * scale, cyz * scale, czz * scale),
    )
    return MeshMoments(
        reference_mm=about,
        volume_mm3=volume,
        first_moment_mm4=first,
        covariance_mm5=covariance,  # type: ignore[arg-type]
    )


def mesh_volume_mm3(mesh: TriangleMesh) -> float:
    """**带符号**体积。取绝对值是错的——符号是本模块唯一的朝向证人。

    外法向朝外给正，朝里给负。`mesh_mass_properties`见到非正即拒。
    """

    _require_nondegenerate(mesh)
    _require_closed(mesh)
    return mesh_moments(mesh).volume_mm3


def mesh_centroid_mm(mesh: TriangleMesh) -> Vector3:
    """体质心（不是顶点平均，也不是面积加权）。"""

    _require_nondegenerate(mesh)
    _require_closed(mesh)
    moments = mesh_moments(mesh)
    volume = moments.volume_mm3
    if volume <= 0.0:
        raise MeshError(_orientation_message(volume))
    return (
        moments.reference_mm[0] + moments.first_moment_mm4[0] / volume,
        moments.reference_mm[1] + moments.first_moment_mm4[1] / volume,
        moments.reference_mm[2] + moments.first_moment_mm4[2] / volume,
    )


def _orientation_message(volume: float) -> str:
    if volume == 0.0:
        return (
            "mesh encloses zero signed volume — 要么是一张退化到一个面的网格，"
            "要么正反两半互相抵消，两种都不是一个可以称质量的体"
        )
    return (
        f"mesh encloses negative signed volume ({volume!r} mm³): the triangles wind "
        "inward. 右手法则给出的法向必须指向体外——反了不是可以容忍的输入，"
        "是必须被判出来的错误（把每个三角的两个下标对换即可）"
    )


def _inertia_from_covariance(covariance: Matrix3) -> Matrix3:
    """``I = tr(C)·E − C``。三个方向只写一遍。"""

    trace = covariance[0][0] + covariance[1][1] + covariance[2][2]
    return tuple(  # type: ignore[return-value]
        tuple(
            (trace if i == j else 0.0) - covariance[i][j] for j in range(3)
        )
        for i in range(3)
    )


def mesh_mass_properties(
    mesh: TriangleMesh,
    *,
    density_kg_m3: float | None = None,
    mass_kg: float | None = None,
) -> MassProperties:
    """体积/质心/绕质心惯量。**密度与质量恰给一个**——口径逐字照`geometry.mass_properties`。

    返回的正是`geometry.MassProperties`，所以解析原语与三角网格的结果可以
    **同一个类型直接对拍**——那是甲1全部验收判据的形状。

    四道失败关闭按顺序：退化三角 → 闭合定向性 → 朝向（体积符号）→ 密度/质量二选一。
    """

    _require_nondegenerate(mesh)
    _require_closed(mesh)
    if (density_kg_m3 is None) == (mass_kg is None):
        raise MeshError("give exactly one of density_kg_m3 or mass_kg")

    moments = mesh_moments(mesh)
    volume = moments.volume_mm3
    if volume <= 0.0:
        raise MeshError(_orientation_message(volume))

    offset = (
        moments.first_moment_mm4[0] / volume,
        moments.first_moment_mm4[1] / volume,
        moments.first_moment_mm4[2] / volume,
    )
    centroid = (
        moments.reference_mm[0] + offset[0],
        moments.reference_mm[1] + offset[1],
        moments.reference_mm[2] + offset[2],
    )

    #: 把二阶矩从参考点搬到质心：``C_c = C_p − V·d⊗d``，``d = c − p``。
    #: **这是矩的平行轴定理，不是惯量的**——惯量那条在
    #: `geometry.shift_inertia_kg_mm2`，两条互不替代。
    centred = tuple(
        tuple(
            moments.covariance_mm5[i][j] - volume * offset[i] * offset[j]
            for j in range(3)
        )
        for i in range(3)
    )
    shape_mm5 = _inertia_from_covariance(centred)  # type: ignore[arg-type]

    if density_kg_m3 is not None:
        if not math.isfinite(density_kg_m3) or density_kg_m3 <= 0.0:
            raise MeshError(f"density_kg_m3 must be positive and finite: {density_kg_m3!r}")
        mass = density_kg_m3 * volume / MM3_PER_M3
    else:
        if not math.isfinite(mass_kg) or mass_kg <= 0.0:  # type: ignore[arg-type]
            raise MeshError(f"mass_kg must be positive and finite: {mass_kg!r}")
        mass = float(mass_kg)  # type: ignore[arg-type]

    scale = mass / volume
    inertia = tuple(tuple(scale * value for value in row) for row in shape_mm5)
    return MassProperties(
        volume_mm3=volume,
        centroid_mm=centroid,
        mass_kg=mass,
        inertia_about_centroid_kg_mm2=inertia,  # type: ignore[arg-type]
    )


def mesh_inertia_about_origin_kg_mm2(
    mesh: TriangleMesh,
    *,
    density_kg_m3: float | None = None,
    mass_kg: float | None = None,
) -> Matrix3:
    """绕**坐标原点**的惯量，由关于原点的原始矩**直接**算出。

    它与"绕质心的惯量再用`geometry.shift_inertia_kg_mm2`搬到原点"是
    **两条独立的路**：这一条不做重心化、直接吃``about_mm = (0,0,0)``的二阶矩。
    平移轴定理那条验收判据判的正是两条路给出同一个答案——
    **一条路自己跟自己对拍不算恒等式**。
    """

    properties = mesh_mass_properties(
        mesh, density_kg_m3=density_kg_m3, mass_kg=mass_kg
    )
    moments = mesh_moments(mesh, about_mm=(0.0, 0.0, 0.0))
    scale = properties.mass_kg / moments.volume_mm3
    return tuple(  # type: ignore[return-value]
        tuple(scale * value for value in row)
        for row in _inertia_from_covariance(moments.covariance_mm5)
    )


# --------------------------------------------------------------------------
# 三角化器：仓内唯一的三角网格语料来源（理由见模块docstring末节）
# --------------------------------------------------------------------------


def _orient_outward_about_origin(
    vertices: tuple[Vector3, ...], triangles: list[tuple[int, int, int]]
) -> tuple[tuple[int, int, int], ...]:
    """逐面按"面法向背离原点"定朝向。**只对含原点的星形体成立。**

    本模块三个三角化器产的都是含原点的凸体，所以这条判据够用。
    它**不看体积的符号**——所以`mesh_volume_mm3`那道朝向门不是自证的：
    生成器与判据用的是两条不相干的判法。
    """

    oriented: list[tuple[int, int, int]] = []
    for ia, ib, ic in triangles:
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        centre = (
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        )
        outward = sum(n * m for n, m in zip(normal, centre, strict=True))
        oriented.append((ia, ib, ic) if outward > 0.0 else (ia, ic, ib))
    return tuple(oriented)


def _split_edge(
    points: list[Vector3],
    midpoints: dict[tuple[int, int], int],
    i: int,
    j: int,
    radius: float,
) -> int:
    """边中点投影到球面，按无序边缓存——**共享边只投一次**。

    缓存不只是省时间：两个相邻三角各自算一次中点会得到**两个下标**，
    于是网格从"闭合"退化成"两片贴在一起"，`mesh_edge_defects`当场红。
    """

    key = (i, j) if i < j else (j, i)
    hit = midpoints.get(key)
    if hit is not None:
        return hit
    a, b = points[key[0]], points[key[1]]
    mx, my, mz = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])
    norm = math.sqrt(mx * mx + my * my + mz * mz)
    index = len(points)
    points.append((radius * mx / norm, radius * my / norm, radius * mz / norm))
    midpoints[key] = index
    return index


def triangulate_sphere(shape: Sphere, *, subdivisions: int) -> TriangleMesh:
    """内接**细分二十面体**（icosphere）。``subdivisions``每加一，边长减半。

    每加一层三角数×4、边长×½，于是"分辨率加倍"是精确的2的幂——
    收敛比落在4上时，那个4是货真价实的二阶而不是拟合出来的。

    **内接**（顶点在球面上、面在球内）故体积偏小，误差恒为负号；
    这条符号性质本身是一条判据（数值噪声不会只往一边偏）。
    """

    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int):
        raise MeshError(f"subdivisions must be an int: {subdivisions!r}")
    if not 0 <= subdivisions <= 6:
        raise MeshError(
            f"subdivisions must be in [0, 6]: {subdivisions!r} — "
            "上界不是算法限制，是纯Python的墙钟（第6层已是81920个三角）"
        )
    radius = shape.radius_mm
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    seeds = [
        (-1.0, phi, 0.0), (1.0, phi, 0.0), (-1.0, -phi, 0.0), (1.0, -phi, 0.0),
        (0.0, -1.0, phi), (0.0, 1.0, phi), (0.0, -1.0, -phi), (0.0, 1.0, -phi),
        (phi, 0.0, -1.0), (phi, 0.0, 1.0), (-phi, 0.0, -1.0), (-phi, 0.0, 1.0),
    ]
    faces: list[tuple[int, int, int]] = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    def _project(point: Vector3) -> Vector3:
        norm = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        return (
            radius * point[0] / norm,
            radius * point[1] / norm,
            radius * point[2] / norm,
        )

    points: list[Vector3] = [_project(seed) for seed in seeds]
    for _level in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}
        nxt: list[tuple[int, int, int]] = []
        for ia, ib, ic in faces:
            iab = _split_edge(points, midpoints, ia, ib, radius)
            ibc = _split_edge(points, midpoints, ib, ic, radius)
            ica = _split_edge(points, midpoints, ic, ia, radius)
            nxt.extend(
                ((ia, iab, ica), (ib, ibc, iab), (ic, ica, ibc), (iab, ibc, ica))
            )
        faces = nxt

    vertices = tuple(points)
    return TriangleMesh(
        vertices_mm=vertices,
        triangles=_orient_outward_about_origin(vertices, faces),
    )


def triangulate_box(shape: RoundedBox) -> TriangleMesh:
    """轴对齐盒的12个三角。**要求``fillet_radius_mm == 0``**。

    圆角盒的三角化没有"细分即收敛"的自然参数（圆角面要另一套采样），
    而本模块只需要一个**精确可三角化**的体来分辨"三角化误差"与"求和误差"——
    盒子给的是前者恒为零的那一档。圆角盒失败关闭，不拿一个方盒冒充它。
    """

    if shape.fillet_radius_mm != 0.0:
        raise MeshError(
            f"triangulate_box needs a sharp box (fillet_radius_mm = {shape.fillet_radius_mm!r}): "
            "圆角面要另一套采样，本函数不拿方盒冒充圆角盒"
        )
    hx, hy, hz = shape.half_extents_mm
    vertices: tuple[Vector3, ...] = (
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    )
    quads = (
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    )
    faces = [
        triangle
        for a, b, c, d in quads
        for triangle in ((a, b, c), (a, c, d))
    ]
    return TriangleMesh(
        vertices_mm=vertices,
        triangles=_orient_outward_about_origin(vertices, faces),
    )


def triangulate_cylinder(shape: FiniteCylinder, *, segments: int) -> TriangleMesh:
    """z轴向的圆柱：内接正``segments``棱柱 + 两个端面扇形。

    **端面是精确的**（平的），全部三角化误差都在侧面那一圈——
    于是加密``segments``时误差按``O(1/segments²)``降，与球那条同阶但来源不同。
    法兰字段（`flange_outer_radius_mm`）不为``None``时失败关闭，
    与`geometry._reject_unsupported`同一条理由：法兰的轴向尺寸没有被声明。
    """

    if shape.flange_outer_radius_mm is not None:
        raise MeshError(
            "a flanged FiniteCylinder has no declared flange width — 同geometry那条，"
            "不拿AABB冒充质量分布"
        )
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 3:
        raise MeshError(f"segments must be an int >= 3: {segments!r}")

    radius = shape.radius_mm
    half = shape.half_width_mm
    vertices: list[Vector3] = []
    for k in range(segments):
        angle = 2.0 * math.pi * k / segments
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        vertices.append((x, y, -half))
        vertices.append((x, y, half))
    bottom_hub = len(vertices)
    vertices.append((0.0, 0.0, -half))
    top_hub = len(vertices)
    vertices.append((0.0, 0.0, half))

    faces: list[tuple[int, int, int]] = []
    for k in range(segments):
        b0, t0 = 2 * k, 2 * k + 1
        b1, t1 = 2 * ((k + 1) % segments), 2 * ((k + 1) % segments) + 1
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))
        faces.append((bottom_hub, b0, b1))
        faces.append((top_hub, t0, t1))

    frozen = tuple(vertices)
    return TriangleMesh(
        vertices_mm=frozen,
        triangles=_orient_outward_about_origin(frozen, faces),
    )


__all__ = [
    "MeshError",
    "MeshMoments",
    "TriangleMesh",
    "mesh_centroid_mm",
    "mesh_edge_defects",
    "mesh_inertia_about_origin_kg_mm2",
    "mesh_mass_properties",
    "mesh_moments",
    "mesh_volume_mm3",
    "shift_inertia_kg_mm2",
    "triangulate_box",
    "triangulate_cylinder",
    "triangulate_sphere",
]
