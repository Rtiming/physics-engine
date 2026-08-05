"""几何量的门——判据一律是**解析闭式解**，不是"跟上一版一样"。

判据强度按plans/02第四节：解析闭式解 > 跨实现对拍 > 自洽性。本文件三层都有：

* **闭式解**（最强）：球`I = 2mr²/5`、实心圆柱轴向`I = mR²/2`、
  盒`I = m(a²+b²)/12`、胶囊体积`πr²L + 4πr³/3`、圆角盒体积走Steiner展开。
  Steiner那条在实现里是**按四族分块积**出来的，在测试里是**整式**——
  两条推导路径不同，不是spec/08规则3禁止的"在测试里复述oracle公式"。
* **跨实现对拍**：圆角盒的三条退化极限（`r→0`是盒、`h→0`是球、
  `hx,hy→0`是胶囊）分别落到另外两个原语的独立代码路径上；
  再加一条以`sdf_mm`为独立oracle的数值求积——它是唯一能验到
  "三个半长与圆角都非零"的**一般**圆角盒的判据，前面三条退化极限
  每条都会杀掉一到两族分块。
* **自洽门**：平行轴定理（把盒劈成两半再合回来）、惯量张量对称正定、
  轴对称体的三个主不变量与摆放方向无关。

每条判据都实测过"必须红"（决策0022第五节记录了逐条的破法与输出）。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.geometry import (
    GeometryError,
    centroid_mm,
    mass_properties,
    sdf_mm,
    shift_inertia_kg_mm2,
    volume_mm3,
)
from physics_engine.shapes import (
    Capsule,
    FiniteCylinder,
    GeneratedShape,
    MeshAsset,
    RoundedBox,
    Sphere,
)

MESH = MeshAsset(
    path_relative="assets/link.stl",
    sha256="0" * 64,
    units="mm",
    usage="collision",
    convexity="convex_hull",
    aabb_min_mm=(-1.0, -1.0, -1.0),
    aabb_max_mm=(1.0, 1.0, 1.0),
)


def _relative(got: float, expected: float) -> float:
    return abs(got - expected) / abs(expected)


def _invariants(matrix) -> tuple[float, float, float]:
    """三个主不变量：迹、二阶主子式和、行列式。三者相同即谱相同。"""

    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    minors = (
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        + matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]
        + matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]
    )
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return trace, minors, determinant


# ---------------------------------------------------------------- 闭式解


def test_sphere_matches_the_closed_form():
    """`V = 4πr³/3`、`I = (2/5)mr²`各向同性、质心在原点。"""

    properties = mass_properties(Sphere(radius_mm=3.0), mass_kg=2.0)
    assert _relative(properties.volume_mm3, 4.0 * math.pi * 27.0 / 3.0) < 1e-15
    assert properties.centroid_mm == (0.0, 0.0, 0.0)
    expected = 0.4 * 2.0 * 9.0
    for axis in range(3):
        assert _relative(properties.inertia_about_centroid_kg_mm2[axis][axis], expected) < 1e-14


def test_density_path_matches_the_mass_path():
    """密度走的是mm³↔m³这一条纯长度量纲换算（spec/14第五节登记过的）。"""

    sphere = Sphere(radius_mm=100.0)
    volume_m3 = 4.0 * math.pi * 100.0**3 / 3.0 / 1.0e9
    properties = mass_properties(sphere, density_kg_m3=1000.0)
    assert _relative(properties.mass_kg, 1000.0 * volume_m3) < 1e-14
    by_mass = mass_properties(sphere, mass_kg=properties.mass_kg)
    assert properties.inertia_about_centroid_kg_mm2 == by_mass.inertia_about_centroid_kg_mm2


def test_solid_cylinder_matches_the_closed_form():
    """`V = 2πR²W`、轴向`I = mR²/2`、横向`I = m(3R² + 4W²)/12`。"""

    properties = mass_properties(FiniteCylinder(radius_mm=2.0, half_width_mm=5.0), mass_kg=1.0)
    inertia = properties.inertia_about_centroid_kg_mm2
    assert _relative(properties.volume_mm3, 2.0 * math.pi * 4.0 * 5.0) < 1e-15
    assert _relative(inertia[2][2], 1.0 * 4.0 / 2.0) < 1e-14
    transverse = 1.0 * (3.0 * 4.0 + 4.0 * 25.0) / 12.0
    assert _relative(inertia[0][0], transverse) < 1e-14
    assert _relative(inertia[1][1], transverse) < 1e-14


def test_box_matches_the_closed_form():
    """圆角为零就是盒：`V = abc`、`I = m(a² + b²)/12`。"""

    box = RoundedBox(half_extents_mm=(2.0, 3.0, 4.0), fillet_radius_mm=0.0)
    properties = mass_properties(box, mass_kg=6.0)
    inertia = properties.inertia_about_centroid_kg_mm2
    a, b, c = 4.0, 6.0, 8.0
    assert _relative(properties.volume_mm3, a * b * c) < 1e-15
    assert _relative(inertia[0][0], 6.0 * (b * b + c * c) / 12.0) < 1e-14
    assert _relative(inertia[1][1], 6.0 * (c * c + a * a) / 12.0) < 1e-14
    assert _relative(inertia[2][2], 6.0 * (a * a + b * b) / 12.0) < 1e-14


def test_capsule_volume_matches_the_closed_form():
    """圆柱段加两个半球：`V = πr²L + 4πr³/3`，质心在两端点中点。"""

    capsule = Capsule(point_a_mm=(1.0, 2.0, 3.0), point_b_mm=(1.0, 2.0, 11.0), radius_mm=1.5)
    volume = volume_mm3(capsule)
    expected = math.pi * 1.5**2 * 8.0 + 4.0 * math.pi * 1.5**3 / 3.0
    assert _relative(volume, expected) < 1e-15
    assert centroid_mm(capsule) == (1.0, 2.0, 7.0)


def test_degenerate_capsule_is_exactly_a_sphere():
    """两端点重合的胶囊必须退回球的闭式解——半球那两项合并的硬判据。"""

    point = (1.0, -2.0, 0.5)
    properties = mass_properties(Capsule(point, point, radius_mm=3.0), mass_kg=2.0)
    assert properties.centroid_mm == point
    assert _relative(properties.volume_mm3, 4.0 * math.pi * 27.0 / 3.0) < 1e-15
    for axis in range(3):
        assert _relative(
            properties.inertia_about_centroid_kg_mm2[axis][axis], 0.4 * 2.0 * 9.0
        ) < 1e-14


def test_rounded_box_volume_matches_the_steiner_expansion():
    """盒⊕球的体积整式：`abc + S·r + πr²·Σ边长 + 4πr³/3`。

    实现是按核心盒/面板/四分之一圆柱/角八分球四族**分块**积的，
    这里写的是Steiner**整式**——两条推导路径不同，不是复述实现。
    """

    hx, hy, hz, r = 2.0, 3.0, 4.0, 1.0
    box = RoundedBox(half_extents_mm=(hx, hy, hz), fillet_radius_mm=r)
    surface = 8.0 * (hx * hy + hy * hz + hz * hx)
    edges = 2.0 * (hx + hy + hz)
    expected = (
        8.0 * hx * hy * hz
        + surface * r
        + math.pi * r * r * edges
        + 4.0 * math.pi * r**3 / 3.0
    )
    assert _relative(volume_mm3(box), expected) < 1e-14


def test_rounded_box_degenerates_to_a_sphere():
    """半长趋零、圆角留着：盒⊕球就是球，`I = (2/5)mr²`。"""

    tiny = 1.0e-7
    properties = mass_properties(
        RoundedBox(half_extents_mm=(tiny, tiny, tiny), fillet_radius_mm=3.0), mass_kg=2.0
    )
    assert _relative(properties.volume_mm3, 4.0 * math.pi * 27.0 / 3.0) < 1e-6
    for axis in range(3):
        assert _relative(
            properties.inertia_about_centroid_kg_mm2[axis][axis], 0.4 * 2.0 * 9.0
        ) < 1e-6


def test_rounded_box_degenerates_to_a_capsule():
    """两个半长趋零：盒⊕球就是胶囊——**跨原语对拍**，两条独立代码路径。

    容差1e-6由摄动量`hx = hy = 1e-9 mm`定：极限本身是精确的，
    差的只是那点没抹干净的半长。
    """

    half_length, radius = 5.0, 2.0
    box = RoundedBox(half_extents_mm=(1.0e-9, 1.0e-9, half_length), fillet_radius_mm=radius)
    capsule = Capsule((0.0, 0.0, -half_length), (0.0, 0.0, half_length), radius)
    from_box = mass_properties(box, mass_kg=3.0)
    from_capsule = mass_properties(capsule, mass_kg=3.0)
    assert _relative(from_box.volume_mm3, from_capsule.volume_mm3) < 1e-6
    for axis in range(3):
        assert _relative(
            from_box.inertia_about_centroid_kg_mm2[axis][axis],
            from_capsule.inertia_about_centroid_kg_mm2[axis][axis],
        ) < 1e-6


def test_generated_shape_delegates_to_the_shape_it_produced():
    """生成器是身份包装（spec/11规则2），质量属性看它产出的那个形。"""

    inner = Sphere(radius_mm=3.0)
    wrapped = GeneratedShape(
        algorithm_id="algorithm:sphere_seed",
        algorithm_version="1",
        parameters=(("radius_mm", 3.0),),
        shape=inner,
    )
    assert mass_properties(wrapped, mass_kg=2.0) == mass_properties(inner, mass_kg=2.0)


# ---------------------------------------------------------------- 自洽门


SHAPE_BATTERY = (
    Sphere(radius_mm=2.5),
    Capsule((0.0, 0.0, -4.0), (0.0, 0.0, 4.0), 1.5),
    Capsule((-1.0, -2.0, -3.0), (4.0, 1.0, 2.0), 0.75),
    RoundedBox(half_extents_mm=(2.0, 3.0, 4.0), fillet_radius_mm=0.0),
    RoundedBox(half_extents_mm=(2.0, 3.0, 4.0), fillet_radius_mm=1.0),
    FiniteCylinder(radius_mm=2.0, half_width_mm=5.0),
)


@pytest.mark.parametrize("shape", SHAPE_BATTERY)
def test_inertia_tensor_is_symmetric_and_positive_definite(shape):
    """惯量张量是实对称正定的——这是它作为二阶矩的定义性质，不是选择。

    正定按Sylvester判据（三个顺序主子式全正），不用特征值求解器：
    判据本身要经得起查，塞一个自己写的求解器进来等于多一个可疑环节。
    """

    inertia = mass_properties(shape, mass_kg=1.0).inertia_about_centroid_kg_mm2
    scale = max(abs(inertia[i][j]) for i in range(3) for j in range(3))
    for i in range(3):
        for j in range(3):
            assert abs(inertia[i][j] - inertia[j][i]) <= 1e-12 * scale

    trace, minors, determinant = _invariants(inertia)
    assert inertia[0][0] > 0.0
    assert inertia[0][0] * inertia[1][1] - inertia[0][1] * inertia[1][0] > 0.0
    assert determinant > 0.0
    assert trace > 0.0 and minors > 0.0


def test_axisymmetric_inertia_does_not_depend_on_how_the_body_is_laid_out():
    """同一根胶囊摆成z轴与摆成体对角线，三个主不变量必须一致。

    迹/二阶主子式和/行列式三者相同等价于谱相同——这样验旋转不变性
    不需要引入特征值求解器。
    """

    half_length, radius = 4.0, 1.5
    along_z = Capsule((0.0, 0.0, -half_length), (0.0, 0.0, half_length), radius)
    unit = 1.0 / math.sqrt(3.0)
    diagonal = Capsule(
        tuple(-half_length * unit for _ in range(3)),
        tuple(half_length * unit for _ in range(3)),
        radius,
    )
    reference = _invariants(mass_properties(along_z, mass_kg=2.0).inertia_about_centroid_kg_mm2)
    rotated = _invariants(mass_properties(diagonal, mass_kg=2.0).inertia_about_centroid_kg_mm2)
    for got, expected in zip(rotated, reference, strict=True):
        assert _relative(got, expected) < 1e-12


def test_parallel_axis_theorem_reassembles_a_split_box():
    """把盒沿x劈成两半、各自平移回去再相加，必须等于整块盒的张量。

    这是平行轴定理的自洽门：两个半块的质心不在整块的质心上，
    `m(|d|²E − d⊗d)`那一项要是漏了或者符号反了，这条当场红。
    """

    hx, hy, hz, mass = 2.0, 3.0, 4.0, 6.0
    whole = mass_properties(
        RoundedBox(half_extents_mm=(hx, hy, hz), fillet_radius_mm=0.0), mass_kg=mass
    ).inertia_about_centroid_kg_mm2
    half = mass_properties(
        RoundedBox(half_extents_mm=(0.5 * hx, hy, hz), fillet_radius_mm=0.0),
        mass_kg=0.5 * mass,
    ).inertia_about_centroid_kg_mm2
    shifted = shift_inertia_kg_mm2(half, 0.5 * mass, (0.5 * hx, 0.0, 0.0))
    for i in range(3):
        for j in range(3):
            assert abs(2.0 * shifted[i][j] - whole[i][j]) <= 1e-12 * max(whole[i][i], 1.0)


def test_parallel_axis_shift_matches_the_closed_form_and_ignores_the_offset_sign():
    """球平移`d`后：沿平移方向不变，另两轴各加`md²`。

    顺带钉死"没有符号可以搞反"：`|d|²`与`d⊗d`对`d`都是偶的。
    """

    mass, radius, distance = 2.0, 3.0, 7.0
    centroidal = mass_properties(Sphere(radius_mm=radius), mass_kg=mass)
    base = 0.4 * mass * radius * radius
    shifted = shift_inertia_kg_mm2(
        centroidal.inertia_about_centroid_kg_mm2, mass, (distance, 0.0, 0.0)
    )
    assert _relative(shifted[0][0], base) < 1e-14
    assert _relative(shifted[1][1], base + mass * distance**2) < 1e-14
    assert _relative(shifted[2][2], base + mass * distance**2) < 1e-14
    mirrored = shift_inertia_kg_mm2(
        centroidal.inertia_about_centroid_kg_mm2, mass, (-distance, 0.0, 0.0)
    )
    assert mirrored == shifted


# ---------------------------------------------------------------- SDF与数值对拍


def test_sdf_is_the_exact_signed_distance():
    """`φ(x) = φ_core(x) − r_f`：体内负、体表零、体外正，且值就是距离。"""

    assert sdf_mm(Sphere(radius_mm=3.0), (5.0, 0.0, 0.0)) == pytest.approx(2.0, abs=1e-15)
    assert sdf_mm(Sphere(radius_mm=3.0), (3.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-15)
    assert sdf_mm(Sphere(radius_mm=3.0), (0.0, 0.0, 0.0)) == pytest.approx(-3.0, abs=1e-15)

    capsule = Capsule((0.0, 0.0, -4.0), (0.0, 0.0, 4.0), 1.5)
    assert sdf_mm(capsule, (4.5, 0.0, 0.0)) == pytest.approx(3.0, abs=1e-15)
    assert sdf_mm(capsule, (0.0, 0.0, 9.5)) == pytest.approx(4.0, abs=1e-15)

    box = RoundedBox(half_extents_mm=(2.0, 3.0, 4.0), fillet_radius_mm=1.0)
    assert sdf_mm(box, (0.0, 0.0, 0.0)) == pytest.approx(-3.0, abs=1e-15)
    assert sdf_mm(box, (2.0, 3.0, 4.0)) == pytest.approx(-1.0, abs=1e-15)
    assert sdf_mm(box, (5.0, 3.0, 4.0)) == pytest.approx(2.0, abs=1e-15)

    cylinder = FiniteCylinder(radius_mm=2.0, half_width_mm=5.0)
    assert sdf_mm(cylinder, (4.0, 0.0, 0.0)) == pytest.approx(2.0, abs=1e-15)
    assert sdf_mm(cylinder, (0.0, 0.0, 7.0)) == pytest.approx(2.0, abs=1e-15)
    assert sdf_mm(cylinder, (0.0, 0.0, 0.0)) == pytest.approx(-2.0, abs=1e-15)


def test_quadrature_over_the_sdf_confirms_the_general_rounded_box():
    """以`sdf_mm`为独立oracle数值求积，验一般圆角盒（三半长与圆角全非零）。

    这条是唯一能验到面板族的判据：`r→0`杀掉面板/棱/角，`h→0`只剩角，
    `hx,hy→0`只剩棱与角。闭式解那一路和求积这一路只共用形状声明。

    容差1e-2：把精确SDF按`0.5 − φ/Δ`折成体积分数后，光滑面上是O(Δ²)、
    棱角处退回O(Δ)，32³格实测最差3.5e-3，留约3倍余量。
    """

    box = RoundedBox(half_extents_mm=(2.0, 3.0, 4.0), fillet_radius_mm=1.0)
    exact = mass_properties(box, mass_kg=1.0)

    cells = 32
    low = (-4.0, -5.0, -6.0)
    high = (4.0, 5.0, 6.0)
    step = [(high[axis] - low[axis]) / cells for axis in range(3)]
    width = (step[0] * step[1] * step[2]) ** (1.0 / 3.0)
    cell_volume = step[0] * step[1] * step[2]

    volume = 0.0
    moments = [0.0, 0.0, 0.0]
    for i in range(cells):
        x = low[0] + (i + 0.5) * step[0]
        for j in range(cells):
            y = low[1] + (j + 0.5) * step[1]
            for k in range(cells):
                z = low[2] + (k + 0.5) * step[2]
                fraction = 0.5 - sdf_mm(box, (x, y, z)) / width
                fraction = min(1.0, max(0.0, fraction))
                if fraction == 0.0:
                    continue
                weight = cell_volume * fraction
                volume += weight
                moments[0] += weight * x * x
                moments[1] += weight * y * y
                moments[2] += weight * z * z

    assert _relative(volume, exact.volume_mm3) < 1e-2
    diagonal = (
        moments[1] + moments[2],
        moments[2] + moments[0],
        moments[0] + moments[1],
    )
    for axis in range(3):
        got = diagonal[axis] / volume
        assert _relative(got, exact.inertia_about_centroid_kg_mm2[axis][axis]) < 1e-2


# ---------------------------------------------------------------- 失败关闭


def test_mass_and_density_together_are_rejected():
    """两个都给会产生一个没人会去核对的隐含体积——不做"以质量为准"的静默取舍。"""

    with pytest.raises(GeometryError, match="exactly one"):
        mass_properties(Sphere(radius_mm=1.0), density_kg_m3=1000.0, mass_kg=1.0)


def test_neither_mass_nor_density_is_rejected():
    with pytest.raises(GeometryError, match="exactly one"):
        mass_properties(Sphere(radius_mm=1.0))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_density_is_rejected(bad):
    with pytest.raises(GeometryError, match="density_kg_m3"):
        mass_properties(Sphere(radius_mm=1.0), density_kg_m3=bad)


def test_mesh_assets_have_no_mass_model():
    """引擎从不读网格字节（shapes.py的既有边界），声明的AABB是包络不是质量分布。"""

    with pytest.raises(GeometryError, match="no mass model"):
        mass_properties(MESH, mass_kg=1.0)
    with pytest.raises(GeometryError, match="no mass model"):
        sdf_mm(MESH, (0.0, 0.0, 0.0))


def test_flanged_cylinder_has_no_declared_flange_width():
    """法兰只声明了外径、没声明轴向尺寸——体积不由声明决定，一律拒。

    这是spec/11形状词汇的一处真实缺口（决策0022第六节登记）。
    拿AABB当法兰会把导轮的质量算成一个实心大盘，错得没有告警。
    """

    flanged = FiniteCylinder(radius_mm=2.0, half_width_mm=5.0, flange_outer_radius_mm=6.0)
    with pytest.raises(GeometryError, match="flange"):
        mass_properties(flanged, mass_kg=1.0)
    with pytest.raises(GeometryError, match="flange"):
        volume_mm3(flanged)
    with pytest.raises(GeometryError, match="flange"):
        sdf_mm(flanged, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("bad", [(0.0, 0.0), (float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)])
def test_nonfinite_query_points_are_rejected(bad):
    with pytest.raises(GeometryError, match="point_mm"):
        sdf_mm(Sphere(radius_mm=1.0), bad)


def test_parallel_axis_shift_rejects_a_nonfinite_offset():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(GeometryError, match="offset_mm"):
        shift_inertia_kg_mm2(identity, 1.0, (float("nan"), 0.0, 0.0))
    with pytest.raises(GeometryError, match="mass_kg"):
        shift_inertia_kg_mm2(identity, -1.0, (1.0, 0.0, 0.0))
