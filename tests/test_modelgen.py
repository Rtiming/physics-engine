"""`modelgen`的校验门与形制——每条失败关闭都有一条"必须红"（AGENTS.md本仓纪律）。

案例`case/generator_determinism`验的是**产物**（确定性、齐次性、质量属性）；
本文件验的是**拒收**：参数不自洽时生成器必须炸，而不是取个合理默认。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.geometry import GeometryError, mass_properties
from physics_engine.modelgen import (
    FORMER_ALGORITHM_ID,
    ROLLER_ALGORITHM_ID,
    SPOOL_ALGORITHM_ID,
    GeneratedPart,
    ModelGenError,
    declaration_bytes,
    declaration_document,
    generate_former,
    generate_roller,
    generate_spool,
)
from physics_engine.shapes import (
    Capsule,
    FiniteCylinder,
    GeneratedShape,
    MeshAsset,
    RoundedBox,
    Sphere,
)

SPOOL = {
    "characteristic_length_mm": 256.0,
    "barrel_radius_ratio": 0.25,
    "barrel_width_ratio": 0.375,
    "flange_outer_radius_ratio": 0.4375,
    "flange_width_ratio": 0.0625,
    "wound_layers": 8,
    "layer_thickness_ratio": 0.001953125,
}
FORMER = {
    "characteristic_length_mm": 512.0,
    "skeleton_ratios": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.25, 0.0)),
    "root_radius_ratio": 0.0625,
    "tip_radius_ratio": 0.03125,
}


# --------------------------------------------------------------------------
# 三个生成器的正向形制
# --------------------------------------------------------------------------

def test_every_product_carries_the_generator_identity_and_the_whole_call():
    """spec/11规则2：生成器身份+版本+**参数记录**，三样都要进产出。"""

    for parts, algorithm_id in (
        (generate_spool(**SPOOL), SPOOL_ALGORITHM_ID),
        (generate_roller(characteristic_length_mm=128.0, radius_ratio=0.375,
                         face_width_ratio=0.125), ROLLER_ALGORITHM_ID),
        (generate_former(**FORMER), FORMER_ALGORITHM_ID),
    ):
        stamps = {part.shape.parameters for part in parts}
        assert len(stamps) == 1, "同一次调用的各件必须盖同一份参数表"
        for part in parts:
            assert isinstance(part.shape, GeneratedShape)
            assert part.shape.algorithm_id == algorithm_id
            assert part.shape.algorithm_version
            names = [name for name, _ in part.shape.parameters]
            assert names == sorted(set(names), key=names.index), "参数名不得重复"
            assert "characteristic_length_mm" in names


def test_the_spool_never_declares_a_flanged_cylinder():
    """法兰走"独立第二个形"那条路（spec/11第二之二节）——产物里没有法兰字段。

    这条不是风格偏好：带法兰的`FiniteCylinder`在`geometry`那头是失败关闭的，
    产了它就等于产了一个永远算不出质量属性的形。下半段把那个失败演示出来。
    """

    parts = generate_spool(**SPOOL)
    assert [part.part_id for part in parts] == ["barrel", "flange_low", "flange_high"]
    for part in parts:
        assert isinstance(part.shape.shape, FiniteCylinder)
        assert part.shape.shape.flange_outer_radius_mm is None
        mass_properties(part.shape, density_kg_m3=7850.0)  # 三件都算得出

    # 若当初选了"给FiniteCylinder加法兰外径"那条路，产物会长成这样——
    with pytest.raises(GeometryError, match="flange"):
        mass_properties(
            FiniteCylinder(radius_mm=68.0, half_width_mm=48.0, flange_outer_radius_mm=112.0),
            density_kg_m3=7850.0,
        )


def test_the_flanged_decomposition_is_an_exact_partition_of_the_real_spool():
    """筒(R_b,W) + 两片盘(R_f,w)贴在筒两端外侧 = 真实带盘实体，且三件互不重叠。

    互不重叠由轴向区间给出：筒占`|z| ≤ W/2`，两片盘占`W/2 ≤ |z| ≤ W/2 + w`。
    体积因此可直接相加——这是"分解是精确的，不是近似"的可执行形式。
    """

    barrel, low, high = generate_spool(**SPOOL)
    barrel_half = barrel.shape.shape.half_width_mm
    flange_half = low.shape.shape.half_width_mm
    assert low.offset_mm == (0.0, 0.0, -(barrel_half + flange_half))
    assert high.offset_mm == (0.0, 0.0, barrel_half + flange_half)
    assert low.shape.shape.radius_mm >= barrel.shape.shape.radius_mm


def test_the_wound_barrel_radius_grows_with_the_layer_count():
    """WDS `WindingSurface.effective_radius_mm`的形制：半径随已卷层数生长。"""

    empty = generate_spool(**{**SPOOL, "wound_layers": 0})
    wound = generate_spool(**SPOOL)
    assert empty[0].shape.shape.radius_mm == 64.0
    assert wound[0].shape.shape.radius_mm == 68.0


def test_the_former_tapers_from_root_to_tip_along_the_chain():
    """case2 `fo_h_el → fo_h_tip`锥度：根粗梢细，逐段递减。"""

    parts = generate_former(**FORMER)
    radii = [part.shape.shape.radius_mm for part in parts]
    assert radii == [28.0, 20.0]
    assert all(a > b for a, b in zip(radii, radii[1:], strict=False))


def test_the_former_chains_the_skeleton_points_end_to_end():
    parts = generate_former(**FORMER)
    assert [part.part_id for part in parts] == ["link_0", "link_1"]
    assert parts[0].shape.shape.point_b_mm == parts[1].shape.shape.point_a_mm
    assert all(part.offset_mm == (0.0, 0.0, 0.0) for part in parts), (
        "胶囊自带端点坐标，偏移必须是零向量——否则同一个点被算两次"
    )


# --------------------------------------------------------------------------
# 必须红：参数不自洽一律失败关闭
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"characteristic_length_mm": 0.0}, "characteristic_length_mm must be positive"),
        ({"characteristic_length_mm": -1.0}, "characteristic_length_mm must be positive"),
        ({"characteristic_length_mm": math.nan}, "must be finite"),
        ({"characteristic_length_mm": math.inf}, "must be finite"),
        ({"characteristic_length_mm": "256"}, "must be a real number"),
        ({"barrel_radius_ratio": 0.0}, "barrel_radius_ratio must be positive"),
        ({"barrel_width_ratio": -0.5}, "barrel_width_ratio must be positive"),
        ({"layer_thickness_ratio": -1.0}, "layer_thickness_ratio must be nonnegative"),
        ({"wound_layers": -1}, "wound_layers must be nonnegative"),
        ({"wound_layers": 2.0}, "wound_layers must be an integer"),
        ({"wound_layers": True}, "wound_layers must be an integer"),
        ({"layer_thickness_ratio": 0.0}, "requires a positive layer_thickness_ratio"),
        ({"flange_width_ratio": None}, "must be set together"),
        ({"flange_outer_radius_ratio": None}, "must be set together"),
        ({"flange_outer_radius_ratio": 0.26}, "overflowed its flange"),
        ({"flange_outer_radius_ratio": -0.5}, "flange_outer_radius_ratio must be positive"),
        ({"flange_width_ratio": 0.0}, "flange_width_ratio must be positive"),
    ],
)
def test_the_spool_fails_closed_on_incoherent_parameters(override, pattern):
    with pytest.raises(ModelGenError, match=pattern):
        generate_spool(**{**SPOOL, **override})


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"radius_ratio": 0.0}, "radius_ratio must be positive"),
        ({"face_width_ratio": -0.1}, "face_width_ratio must be positive"),
        ({"characteristic_length_mm": 0.0}, "characteristic_length_mm must be positive"),
    ],
)
def test_the_roller_fails_closed_on_incoherent_parameters(override, pattern):
    base = {"characteristic_length_mm": 128.0, "radius_ratio": 0.375, "face_width_ratio": 0.125}
    with pytest.raises(ModelGenError, match=pattern):
        generate_roller(**{**base, **override})


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"skeleton_ratios": ((0.0, 0.0, 0.0),)}, "at least 2 points"),
        ({"skeleton_ratios": ()}, "at least 2 points"),
        ({"skeleton_ratios": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}, "coincide"),
        (
            {"skeleton_ratios": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, 0.0))},
            "skeleton points 1 and 2 coincide",
        ),
        ({"skeleton_ratios": ((0.0, 0.0), (0.5, 0.0))}, "exactly 3 components"),
        ({"skeleton_ratios": ((0.0, 0.0, 0.0), 5.0)}, "must be a 3-sequence"),
        ({"skeleton_ratios": "abc"}, "must be a sequence"),
        ({"skeleton_ratios": ((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0))}, "must be finite"),
        ({"root_radius_ratio": 0.0}, "root_radius_ratio must be positive"),
        ({"tip_radius_ratio": -0.1}, "tip_radius_ratio must be positive"),
    ],
)
def test_the_former_fails_closed_on_incoherent_skeletons(override, pattern):
    with pytest.raises(ModelGenError, match=pattern):
        generate_former(**{**FORMER, **override})


def test_a_coincident_pair_is_rejected_even_when_it_is_written_with_a_signed_zero():
    """`-0.0`与`0.0`是同一个点，退化判定必须在归一之后做。"""

    with pytest.raises(ModelGenError, match="coincide"):
        generate_former(**{**FORMER,
                          "skeleton_ratios": ((0.0, 0.0, 0.0), (-0.0, -0.0, -0.0))})


@pytest.mark.parametrize(
    ("part_id", "offset", "pattern"),
    [
        ("", (0.0, 0.0, 0.0), "nonempty local name"),
        ("body/spool", (0.0, 0.0, 0.0), "without '/'"),
        ("barrel", (0.0, 0.0), "offset_mm must be a 3-vector"),
    ],
)
def test_generated_part_fails_closed_on_bad_identity_or_offset(part_id, offset, pattern):
    shape = GeneratedShape(
        algorithm_id=SPOOL_ALGORITHM_ID, algorithm_version="1.0.0",
        parameters=(), shape=Sphere(radius_mm=1.0),
    )
    with pytest.raises(ModelGenError, match=pattern):
        GeneratedPart(part_id=part_id, offset_mm=offset, shape=shape)


def test_the_declaration_form_refuses_a_shape_it_cannot_write():
    """指纹只认四种解析原语；网格资产没有声明形，写不出来就炸而不是静默跳过。"""

    mesh = MeshAsset(
        path_relative="assets/tetra.stl", sha256="0" * 64, units="mm",
        usage="collision", convexity="convex_hull",
        aabb_min_mm=(-1.0, -1.0, -1.0), aabb_max_mm=(1.0, 1.0, 1.0),
    )
    part = GeneratedPart(
        part_id="mesh", offset_mm=(0.0, 0.0, 0.0),
        shape=GeneratedShape(
            algorithm_id=SPOOL_ALGORITHM_ID, algorithm_version="1.0.0",
            parameters=(), shape=mesh,  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(ModelGenError, match="no declaration form"):
        declaration_document([part])


def test_the_declaration_writes_the_flange_field_so_it_cannot_collide():
    """带法兰与不带法兰的同尺寸圆柱**不得**给出同一份字节。"""

    def wrap(shape):
        return [GeneratedPart(
            part_id="c", offset_mm=(0.0, 0.0, 0.0),
            shape=GeneratedShape(
                algorithm_id=SPOOL_ALGORITHM_ID, algorithm_version="1.0.0",
                parameters=(), shape=shape,
            ),
        )]

    plain = FiniteCylinder(radius_mm=68.0, half_width_mm=48.0)
    flanged = FiniteCylinder(radius_mm=68.0, half_width_mm=48.0, flange_outer_radius_mm=112.0)
    assert declaration_bytes(wrap(plain)) != declaration_bytes(wrap(flanged))


def test_the_declaration_distinguishes_the_four_analytic_primitives():
    """四种原语各有自己的`kind`与字段，指纹不得互撞。"""

    shapes = (
        Sphere(radius_mm=1.0),
        Capsule(point_a_mm=(0.0, 0.0, 0.0), point_b_mm=(1.0, 0.0, 0.0), radius_mm=1.0),
        FiniteCylinder(radius_mm=1.0, half_width_mm=1.0),
        RoundedBox(half_extents_mm=(1.0, 1.0, 1.0), fillet_radius_mm=1.0),
    )
    fingerprints = {
        declaration_bytes([GeneratedPart(
            part_id="c", offset_mm=(0.0, 0.0, 0.0),
            shape=GeneratedShape(
                algorithm_id=SPOOL_ALGORITHM_ID, algorithm_version="1.0.0",
                parameters=(), shape=shape,
            ),
        )])
        for shape in shapes
    }
    assert len(fingerprints) == len(shapes)
