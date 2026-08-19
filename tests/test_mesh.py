"""`physics_engine.mesh`的判据——甲1（决策0085第二节、0074阶段四）。

这份文件的结构是**每道门配一个必红**（AGENTS.md"每个门要红过"）：
门在这里是"某个错误的输入/错误的实现会被判出来"，
必红是"把那个错误真的种进去，看它红"。上一轮11轮注错抓到过空门，
所以本文件里**没有一条断言是只验正例的**。

三条腿分别验的东西不同，不要混：

* **盒**：三角化误差**恒为零**（盒真的就是12个三角），所以它验的是**代数**——
  四面体二阶矩那个式子写错一点，盒就当场对不上闭式解；
* **球/圆柱**：三角化误差不为零，所以它们验的是**阶**——实测二阶（比→4.0000）；
* **平移**：平移轴定理是**恒等式**，两条独立的路（重心化的矩、原点的矩）必须给同一个数。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.geometry import mass_properties, shift_inertia_kg_mm2
from physics_engine.mesh import (
    MeshError,
    TriangleMesh,
    mesh_centroid_mm,
    mesh_edge_defects,
    mesh_inertia_about_origin_kg_mm2,
    mesh_mass_properties,
    mesh_moments,
    mesh_volume_mm3,
    triangulate_box,
    triangulate_cylinder,
    triangulate_sphere,
)
from physics_engine.shapes import FiniteCylinder, RoundedBox, Sphere

DENSITY_KG_M3 = 2700.0
SPHERE = Sphere(radius_mm=7.0)
BOX = RoundedBox(half_extents_mm=(3.0, 5.0, 11.0), fillet_radius_mm=0.0)
CYLINDER = FiniteCylinder(radius_mm=4.0, half_width_mm=9.0)


def _matrix_max_abs_difference(left, right) -> float:
    return max(
        abs(left[i][j] - right[i][j]) for i in range(3) for j in range(3)
    )


def _ratios(errors: list[float]) -> list[float]:
    return [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]


# --------------------------- 腿一：盒——三角化误差恒为零，于是它验代数 ---


def test_the_box_reproduces_the_closed_form_bit_for_bit() -> None:
    """**逐字节**（`float.hex()`）。盒真的就是12个三角，没有三角化误差可言。

    这条不是"精度好"，是"这两条路算的是同一个数"——
    于是它成了四面体二阶矩那条代数的**唯一无噪声证人**：
    式子写错一点，这里当场红，而球/圆柱那两条门会把它当成三角化误差咽下去。
    """

    closed = mass_properties(BOX, density_kg_m3=DENSITY_KG_M3)
    measured = mesh_mass_properties(triangulate_box(BOX), density_kg_m3=DENSITY_KG_M3)

    assert measured.volume_mm3.hex() == closed.volume_mm3.hex()
    assert measured.mass_kg.hex() == closed.mass_kg.hex()
    for i in range(3):
        for j in range(3):
            assert (
                measured.inertia_about_centroid_kg_mm2[i][j].hex()
                == closed.inertia_about_centroid_kg_mm2[i][j].hex()
            ), (i, j)


def test_the_box_gate_would_go_red_on_the_lumped_centroid_approximation() -> None:
    """必须红：把四面体二阶矩换成"质量集中在三角形心"这个常见错法。

    那个错法梯度上看着很像（体积与质心都对），**只有二阶矩错**——
    正是上一条门存在的理由。这里把它种进去，看盒那条逐字节门红。
    """

    mesh = triangulate_box(BOX)
    closed = mass_properties(BOX, density_kg_m3=DENSITY_KG_M3)
    moments = mesh_moments(mesh, about_mm=(0.0, 0.0, 0.0))
    volume = moments.volume_mm3

    #: 集中近似：``∫x⊗x dV ≈ Σ V_tet·(s/4)⊗(s/4)``，丢掉四面体自身的展布。
    lumped = [[0.0] * 3 for _ in range(3)]
    for ia, ib, ic in mesh.triangles:
        a, b, c = (mesh.vertices_mm[k] for k in (ia, ib, ic))
        det = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        centre = tuple(0.25 * (a[k] + b[k] + c[k]) for k in range(3))
        for i in range(3):
            for j in range(3):
                lumped[i][j] += (det / 6.0) * centre[i] * centre[j]

    trace = lumped[0][0] + lumped[1][1] + lumped[2][2]
    scale = closed.mass_kg / volume
    planted = tuple(
        tuple(
            scale * ((trace if i == j else 0.0) - lumped[i][j]) for j in range(3)
        )
        for i in range(3)
    )
    deviation = _matrix_max_abs_difference(
        planted, closed.inertia_about_centroid_kg_mm2
    )
    #: 实测偏差是**同数量级的错**，不是末位噪声——集中近似丢的是主项。
    assert deviation > 0.1 * closed.inertia_about_centroid_kg_mm2[0][0]


# --------------------------- 腿二：球/圆柱——它们验阶 ---


def test_the_sphere_volume_converges_at_second_order() -> None:
    """内接icosphere，细分每加一层边长减半，**实测收敛比→4.0000**（二阶）。

    另判**误差恒为负**：内接体积必然偏小。数值噪声不会只往一边偏，
    所以这条符号性质是"误差来自三角化而不是来自求和"的独立证据。
    """

    closed = mass_properties(SPHERE, density_kg_m3=DENSITY_KG_M3)
    signed = []
    for level in range(2, 6):
        measured = mesh_mass_properties(
            triangulate_sphere(SPHERE, subdivisions=level),
            density_kg_m3=DENSITY_KG_M3,
        )
        signed.append(measured.volume_mm3 - closed.volume_mm3)

    assert all(value < 0.0 for value in signed), signed
    ratios = _ratios([abs(value) for value in signed])
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    #: 先断误差确实降到一个小数，否则上面那串比值可以由三个大数凑出来。
    assert abs(signed[-1]) / closed.volume_mm3 < 1.0e-3


def test_the_sphere_inertia_converges_at_second_order() -> None:
    """惯量张量与体积同阶（实测比**3.9736 / 3.9934**）。**体积对不等于惯量对。**

    起档比体积那条**晚一级**（从细分3起，不是从2起），理由是实测：
    细分2→3的惯量比是**3.8962**，还没进渐近区。
    这不是把门调松，是把它挪到该在的地方——**同一个窗口``[3.9, 4.1]``一个字没动**，
    而`test_the_order_gate_is_tight_enough_to_reject_the_preasymptotic_levels`
    正是拿这条窗口去红粗档的。惯量比体积晚进渐近区是意料之中：
    二阶矩对表面偏差的加权是``r²``，粗档上那个权把偏差放得更开。
    """

    closed = mass_properties(SPHERE, density_kg_m3=DENSITY_KG_M3)
    errors = []
    for level in range(3, 6):
        measured = mesh_mass_properties(
            triangulate_sphere(SPHERE, subdivisions=level),
            density_kg_m3=DENSITY_KG_M3,
        )
        errors.append(
            _matrix_max_abs_difference(
                measured.inertia_about_centroid_kg_mm2,
                closed.inertia_about_centroid_kg_mm2,
            )
        )

    ratios = _ratios(errors)
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    assert errors[-1] / closed.inertia_about_centroid_kg_mm2[0][0] < 2.0e-3


def test_the_cylinder_converges_at_second_order() -> None:
    """圆柱：端面精确、误差全在侧面那一圈，实测比→4.0000。"""

    closed = mass_properties(CYLINDER, density_kg_m3=DENSITY_KG_M3)
    errors = []
    for segments in (16, 32, 64, 128):
        measured = mesh_mass_properties(
            triangulate_cylinder(CYLINDER, segments=segments),
            density_kg_m3=DENSITY_KG_M3,
        )
        errors.append(abs(measured.volume_mm3 - closed.volume_mm3))

    ratios = _ratios(errors)
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    assert errors[-1] / closed.volume_mm3 < 1.0e-3


def test_the_order_gate_is_tight_enough_to_reject_the_preasymptotic_levels() -> None:
    """必须红：同一条``[3.9, 4.1]``窗口，套在**还没进渐近区**的粗档上必须落空。

    实测细分0→1→2那两个比是**3.1177与3.7396**。这条门证明那个窗口
    不是一个宽到什么都能过的区间——"收敛比恒4.0000"这句话有分辨力。
    """

    closed = mass_properties(SPHERE, density_kg_m3=DENSITY_KG_M3)
    errors = [
        abs(
            mesh_mass_properties(
                triangulate_sphere(SPHERE, subdivisions=level),
                density_kg_m3=DENSITY_KG_M3,
            ).volume_mm3
            - closed.volume_mm3
        )
        for level in range(3)
    ]
    ratios = _ratios(errors)
    assert not all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    assert ratios[0] == pytest.approx(3.1177, rel=1.0e-3)
    assert ratios[1] == pytest.approx(3.7396, rel=1.0e-3)


# --------------------------- 腿三：平移轴定理是恒等式 ---


def test_the_inertia_about_the_centroid_is_translation_invariant() -> None:
    """整体平移不改绕质心的惯量。实测偏差报到位（相对1e-13量级）。"""

    mesh = triangulate_sphere(SPHERE, subdivisions=3)
    here = mesh_mass_properties(mesh, density_kg_m3=DENSITY_KG_M3)
    there = mesh_mass_properties(
        mesh.translated((137.0, -49.0, 811.0)), density_kg_m3=DENSITY_KG_M3
    )

    reference = here.inertia_about_centroid_kg_mm2[0][0]
    deviation = _matrix_max_abs_difference(
        here.inertia_about_centroid_kg_mm2, there.inertia_about_centroid_kg_mm2
    )
    assert deviation / reference < 1.0e-12, deviation / reference
    assert there.volume_mm3 == pytest.approx(here.volume_mm3, rel=1.0e-13)
    for axis, offset in enumerate((137.0, -49.0, 811.0)):
        assert there.centroid_mm[axis] == pytest.approx(
            here.centroid_mm[axis] + offset, abs=1.0e-9
        )


def test_the_parallel_axis_theorem_closes_between_two_independent_paths() -> None:
    """``I_P = I_cm + m(|d|²E − d⊗d)``，两条**不共用中间量**的路对上。

    左边：`mesh_inertia_about_origin_kg_mm2`，吃的是关于**原点**的原始矩，
    **不做重心化**；右边：绕质心的惯量再用`geometry.shift_inertia_kg_mm2`搬过去。
    一条路自己跟自己对拍不算恒等式，所以这里必须是两条。
    """

    mesh = triangulate_sphere(SPHERE, subdivisions=3).translated((23.0, -7.0, 41.0))
    direct = mesh_inertia_about_origin_kg_mm2(mesh, density_kg_m3=DENSITY_KG_M3)
    properties = mesh_mass_properties(mesh, density_kg_m3=DENSITY_KG_M3)
    shifted = shift_inertia_kg_mm2(
        properties.inertia_about_centroid_kg_mm2,
        properties.mass_kg,
        properties.centroid_mm,
    )

    reference = max(abs(shifted[i][i]) for i in range(3))
    deviation = _matrix_max_abs_difference(direct, shifted)
    assert deviation / reference < 1.0e-12, deviation / reference


def test_the_parallel_axis_gate_would_go_red_on_the_wrong_sign() -> None:
    """必须红：平行轴项写成``−m(|d|²E − d⊗d)``（搬反方向）必须被判出来。

    `geometry.shift_inertia_kg_mm2`的docstring明写"反过来传负号是错的"——
    这条门把那句话变成一个会红的用例。
    """

    mesh = triangulate_sphere(SPHERE, subdivisions=3).translated((23.0, -7.0, 41.0))
    direct = mesh_inertia_about_origin_kg_mm2(mesh, density_kg_m3=DENSITY_KG_M3)
    properties = mesh_mass_properties(mesh, density_kg_m3=DENSITY_KG_M3)
    centroid = properties.centroid_mm
    squared = sum(component * component for component in centroid)
    wrong = tuple(
        tuple(
            properties.inertia_about_centroid_kg_mm2[i][j]
            - properties.mass_kg * ((squared if i == j else 0.0) - centroid[i] * centroid[j])
            for j in range(3)
        )
        for i in range(3)
    )
    reference = max(abs(direct[i][i]) for i in range(3))
    assert _matrix_max_abs_difference(direct, wrong) / reference > 1.0


# --------------------------- 门：朝向 ---


def test_flipping_the_winding_flips_the_sign_of_the_signed_volume() -> None:
    """朝向反了体积变号——**这是本模块唯一的朝向证人**，所以它必须精确变号。"""

    mesh = triangulate_sphere(SPHERE, subdivisions=2)
    forward = mesh_volume_mm3(mesh)
    backward = mesh_volume_mm3(mesh.flipped())
    assert forward > 0.0
    assert backward < 0.0
    assert backward == pytest.approx(-forward, rel=1.0e-14)


def test_a_flipped_mesh_is_refused_by_mass_properties() -> None:
    """必须红：朝向反了不许算出质量属性来。**取绝对值就是冒充。**"""

    mesh = triangulate_sphere(SPHERE, subdivisions=2).flipped()
    with pytest.raises(MeshError, match="negative signed volume"):
        mesh_mass_properties(mesh, density_kg_m3=DENSITY_KG_M3)
    with pytest.raises(MeshError, match="negative signed volume"):
        mesh_centroid_mm(mesh)


# --------------------------- 门：退化三角 ---


def test_a_repeated_vertex_index_is_refused_at_construction() -> None:
    """必须红：``(3, 3, 5)``这种三角在构造时就拒。"""

    mesh = triangulate_box(BOX)
    broken = list(mesh.triangles)
    broken[0] = (broken[0][0], broken[0][0], broken[0][2])
    with pytest.raises(MeshError, match="repeats a vertex index"):
        TriangleMesh(vertices_mm=mesh.vertices_mm, triangles=tuple(broken))


def test_a_zero_area_triangle_is_refused() -> None:
    """必须红：三个**不同下标**但共线的顶点，构造过得去、质量属性必须拒。"""

    mesh = triangulate_box(BOX)
    vertices = (*mesh.vertices_mm, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    base = len(mesh.vertices_mm)
    broken = TriangleMesh(
        vertices_mm=vertices,
        triangles=(*mesh.triangles, (base, base + 1, base + 2)),
    )
    with pytest.raises(MeshError, match="zero area"):
        mesh_mass_properties(broken, density_kg_m3=DENSITY_KG_M3)


def test_the_degeneracy_gate_says_what_it_does_not_judge() -> None:
    """**薄三角不拒**——这条不是遗漏，是模块docstring里明写的边界。

    "多薄算薄"本模块给不出有根据的阈值，给一个就是冒充。
    这条门把那句话钉成一个会红的用例：把盒的一个顶点挪到几乎共线，
    面积是``1e-12``量级但不为零，于是**必须过**。
    """

    hx, hy, hz = BOX.half_extents_mm
    vertices = (
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0e-13, 0.0),
    )
    sliver = TriangleMesh(
        vertices_mm=vertices, triangles=(*triangulate_box(BOX).triangles, (8, 9, 10))
    )
    #: 薄三角过了退化门，但它让网格不再闭合——于是红在**下一道门**上。
    #: 两道门各判各的，这正是它们分开写的理由。
    with pytest.raises(MeshError, match="not a closed oriented manifold"):
        mesh_mass_properties(sliver, density_kg_m3=DENSITY_KG_M3)


# --------------------------- 门：闭合定向性 ---


def test_a_clean_mesh_reports_no_edge_defects() -> None:
    """门不许恒红：三个三角化器产的网格必须是干净的。"""

    for mesh in (
        triangulate_box(BOX),
        triangulate_sphere(SPHERE, subdivisions=2),
        triangulate_cylinder(CYLINDER, segments=13),
    ):
        assert mesh_edge_defects(mesh) == ()


def test_an_open_mesh_is_refused() -> None:
    """必须红：删掉一个三角就是开放边界，散度定理当场失去前提。"""

    mesh = triangulate_box(BOX)
    holed = TriangleMesh(vertices_mm=mesh.vertices_mm, triangles=mesh.triangles[1:])
    defects = mesh_edge_defects(holed)
    assert any("开放边界" in defect for defect in defects), defects
    with pytest.raises(MeshError, match="not a closed oriented manifold"):
        mesh_mass_properties(holed, density_kg_m3=DENSITY_KG_M3)


def test_a_non_manifold_edge_is_refused() -> None:
    """必须红：一条边被三个三角共用——0074实测真实件有6条与10条这种边。"""

    mesh = triangulate_box(BOX)
    extra = (*mesh.vertices_mm, (0.0, 0.0, 40.0))
    apex = len(mesh.vertices_mm)
    first = mesh.triangles[0]
    fin = TriangleMesh(
        vertices_mm=extra,
        triangles=(*mesh.triangles, (first[0], first[1], apex)),
    )
    defects = mesh_edge_defects(fin)
    assert defects
    with pytest.raises(MeshError, match="not a closed oriented manifold"):
        mesh_mass_properties(fin, density_kg_m3=DENSITY_KG_M3)


def test_one_inconsistently_wound_triangle_is_refused() -> None:
    """必须红：只把**一个**三角反过来。

    体积照样算得出来（只差那一片的贡献），**这正是它危险的地方**——
    没有这道门，一张绕向不一致的网格会安静地给出一个偏了几个百分点的数。
    """

    mesh = triangulate_box(BOX)
    broken = list(mesh.triangles)
    a, b, c = broken[0]
    broken[0] = (a, c, b)
    twisted = TriangleMesh(vertices_mm=mesh.vertices_mm, triangles=tuple(broken))
    assert mesh_edge_defects(twisted)
    with pytest.raises(MeshError, match="not a closed oriented manifold"):
        mesh_mass_properties(twisted, density_kg_m3=DENSITY_KG_M3)


# --------------------------- 门：密度是入参、单位口径 ---


def test_mass_is_density_times_volume_bit_for_bit() -> None:
    """``mass = density · volume / 1e9``，逐字节。那个1e-9是mm³↔m³。"""

    mesh = triangulate_box(BOX)
    measured = mesh_mass_properties(mesh, density_kg_m3=DENSITY_KG_M3)
    expected = DENSITY_KG_M3 * measured.volume_mm3 / 1.0e9
    assert measured.mass_kg.hex() == expected.hex()


def test_giving_a_mass_directly_bypasses_the_density_but_not_the_volume() -> None:
    """质量直接给时体积不变、惯量按``mass/volume``线性缩放。"""

    mesh = triangulate_box(BOX)
    by_density = mesh_mass_properties(mesh, density_kg_m3=DENSITY_KG_M3)
    by_mass = mesh_mass_properties(mesh, mass_kg=by_density.mass_kg)
    assert by_mass.volume_mm3.hex() == by_density.volume_mm3.hex()
    assert (
        _matrix_max_abs_difference(
            by_mass.inertia_about_centroid_kg_mm2,
            by_density.inertia_about_centroid_kg_mm2,
        )
        / by_density.inertia_about_centroid_kg_mm2[0][0]
        < 1.0e-15
    )


def test_both_or_neither_of_density_and_mass_is_refused() -> None:
    """必须红：两个都给会产生一个没人会去核对的隐含体积（照`geometry`那条）。"""

    mesh = triangulate_box(BOX)
    with pytest.raises(MeshError, match="exactly one"):
        mesh_mass_properties(mesh)
    with pytest.raises(MeshError, match="exactly one"):
        mesh_mass_properties(mesh, density_kg_m3=1.0, mass_kg=1.0)
    with pytest.raises(MeshError, match="density_kg_m3 must be positive"):
        mesh_mass_properties(mesh, density_kg_m3=0.0)


# --------------------------- 三角化器自身的边界 ---


def test_the_triangulators_refuse_what_they_cannot_represent() -> None:
    """圆角盒与带法兰圆柱失败关闭——不拿方盒/AABB冒充。"""

    with pytest.raises(MeshError, match="sharp box"):
        triangulate_box(RoundedBox(half_extents_mm=(1.0, 1.0, 1.0), fillet_radius_mm=0.2))
    with pytest.raises(MeshError, match="flange"):
        triangulate_cylinder(
            FiniteCylinder(radius_mm=4.0, half_width_mm=9.0, flange_outer_radius_mm=6.0),
            segments=8,
        )
    with pytest.raises(MeshError, match="segments"):
        triangulate_cylinder(CYLINDER, segments=2)
    with pytest.raises(MeshError, match="subdivisions"):
        triangulate_sphere(SPHERE, subdivisions=7)


def test_the_orientation_of_the_generators_does_not_come_from_the_volume_sign() -> None:
    """生成器定朝向的判法与判据的判法**不相干**——所以朝向门不是自证的。

    生成器逐面用"面法向背离原点"，判据用带符号体积。
    这条门把那句话钉住：拿一个**不含原点**的平移网格，
    生成器已经定好的朝向依然给正体积。
    """

    mesh = triangulate_sphere(SPHERE, subdivisions=1).translated((500.0, 0.0, 0.0))
    assert mesh_volume_mm3(mesh) > 0.0
    assert all(
        math.hypot(vertex[0], vertex[1]) > SPHERE.radius_mm for vertex in mesh.vertices_mm
    )
