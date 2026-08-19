"""有截面线圈（一束细丝）的协议门。案例判据在`tests/cases/`，本文件不重复它们。

**两条必须红**：

1. `test_the_midpoint_grid_gate_is_red_on_a_half_cell_offset`
   ——截面网格若写成``i/n``而不是``(i+0.5)/n``，整块截面偏半格，
   细丝极限的收敛阶从2掉到1；
2. `test_the_self_bundle_gate_is_red_without_the_refusal`
   ——一束丝与自己配对时含"同一根丝与自己"那一项，必须拒跑。

另外守一条**建模假设**：截面上电流密度均匀（每根丝等份），
它是一个**选择**不是一个数值参数，所以要有一条门看着它别被悄悄改成别的加权。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.electromagnetics import ElectromagneticsError
from physics_engine.electromagnetics import bundle as bundle_module
from physics_engine.electromagnetics.bundle import (
    SECTION_DIVISIONS_MIN,
    RectangularSectionCoil,
    bundle_mutual_inductance_h,
    section_filaments,
)
from physics_engine.electromagnetics.neumann import (
    PlacedCircularLoop,
    neumann_mutual_inductance_h,
)

COIL_A = (0.050, 0.000, 0.010, 0.008)
COIL_B = (0.030, 0.040, 0.006, 0.005)
SEGMENTS = 24


def _coil(spec, grid, scale=1.0, turns=1):
    radius, axial, radial_extent, axial_extent = spec
    return RectangularSectionCoil(
        mean_radius_m=radius,
        centre_m=(0.0, 0.0, axial),
        normal=(0.0, 0.0, 1.0),
        radial_extent_m=radial_extent * scale,
        axial_extent_m=axial_extent * scale,
        radial_filaments=grid,
        axial_filaments=grid,
        turns=turns,
    )


def _offset_filaments(coil, *, half_cell):
    """截面网格的对照实现：``half_cell=False``时写成``i/n``（整块偏半格）。"""

    shift = 0.5 if half_cell else 0.0
    filaments = []
    for radial_index in range(coil.radial_filaments):
        radius = coil.mean_radius_m + coil.radial_extent_m * (
            (radial_index + shift) / coil.radial_filaments - 0.5
        )
        for axial_index in range(coil.axial_filaments):
            offset = coil.axial_extent_m * (
                (axial_index + shift) / coil.axial_filaments - 0.5
            )
            filaments.append(
                PlacedCircularLoop(
                    radius_m=radius,
                    centre_m=(
                        coil.centre_m[0] + offset * coil.normal[0],
                        coil.centre_m[1] + offset * coil.normal[1],
                        coil.centre_m[2] + offset * coil.normal[2],
                    ),
                    normal=coil.normal,
                )
            )
    return tuple(filaments)


def _bundle_with(filaments_a, filaments_b, turns_a=1, turns_b=1):
    values = [
        neumann_mutual_inductance_h(
            filament_a, filament_b, segments_a=SEGMENTS, segments_b=SEGMENTS
        )
        for filament_a in filaments_a
        for filament_b in filaments_b
    ]
    return (turns_a * turns_b) * (
        math.fsum(values) / (len(filaments_a) * len(filaments_b))
    )


def _limit_orders(half_cell):
    deviations = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        coil_a = _coil(COIL_A, 2, scale)
        coil_b = _coil(COIL_B, 2, scale)
        bundled = _bundle_with(
            _offset_filaments(coil_a, half_cell=half_cell),
            _offset_filaments(coil_b, half_cell=half_cell),
        )
        filament = neumann_mutual_inductance_h(
            coil_a.centre_filament(),
            coil_b.centre_filament(),
            segments_a=SEGMENTS,
            segments_b=SEGMENTS,
        )
        deviations.append(abs(bundled - filament) / abs(filament))
    return [
        math.log2(deviations[index] / deviations[index + 1])
        for index in range(len(deviations) - 1)
    ]


# ---------------------------------------------------------------------------
# 公开面与采样
# ---------------------------------------------------------------------------


def test_every_exported_name_exists_and_the_list_is_sorted():
    for name in bundle_module.__all__:
        assert hasattr(bundle_module, name), f"__all__列了不存在的名字{name!r}"
    assert bundle_module.__all__ == sorted(bundle_module.__all__)
    assert SECTION_DIVISIONS_MIN == 1


def test_the_section_grid_is_centred_on_the_mean_radius():
    """截面样本对平均半径与圆心**对称**——细丝极限的二阶精度全靠这一条。"""

    coil = _coil(COIL_A, 4)
    filaments = section_filaments(coil)
    assert len(filaments) == coil.filament_count() == 16
    radii = sorted({filament.radius_m for filament in filaments})
    offsets = sorted({filament.centre_m[2] for filament in filaments})
    assert len(radii) == 4 and len(offsets) == 4
    assert abs(math.fsum(radii) / 4 - coil.mean_radius_m) < 1e-17
    assert abs(math.fsum(offsets) / 4 - coil.centre_m[2]) < 1e-17
    # 最外与最内两圈落在截面边界内半格处
    half_cell = coil.radial_extent_m / (2 * coil.radial_filaments)
    assert abs(radii[0] - (coil.mean_radius_m - coil.radial_extent_m / 2 + half_cell)) < 1e-17
    assert abs(radii[-1] - (coil.mean_radius_m + coil.radial_extent_m / 2 - half_cell)) < 1e-17
    # 每根丝的匝数都是1——匝数是整束的属性，不在这里乘
    assert all(filament.turns == 1 for filament in filaments)


def test_the_current_density_assumption_is_a_plain_average():
    """**建模假设有门看着**：截面上电流密度均匀 = 丝对值的**算术平均**。

    换成任何别的加权（例如按半径加权）都会红。它不是数值参数，
    所以改它必须是一次显式的决策，而不是一次"顺手调一下"。
    """

    coil_a = _coil(COIL_A, 2)
    coil_b = _coil(COIL_B, 3)
    measured = bundle_mutual_inductance_h(
        coil_a, coil_b, segments_a=SEGMENTS, segments_b=SEGMENTS
    )
    expected = _bundle_with(section_filaments(coil_a), section_filaments(coil_b))
    assert measured == expected


def test_the_turns_are_a_factor_not_a_subdivision():
    """匝数与截面细分是两个完全无关的数——把一个当另一个用，本条当场红。"""

    single = bundle_mutual_inductance_h(
        _coil(COIL_A, 3), _coil(COIL_B, 2), segments_a=SEGMENTS, segments_b=SEGMENTS
    )
    with_turns = bundle_mutual_inductance_h(
        _coil(COIL_A, 3, turns=5),
        _coil(COIL_B, 2, turns=7),
        segments_a=SEGMENTS,
        segments_b=SEGMENTS,
    )
    assert with_turns == 35 * single
    # 细分加倍而匝数不变：值只在收敛容差内变，**不是倍数关系**
    finer = bundle_mutual_inductance_h(
        _coil(COIL_A, 6), _coil(COIL_B, 4), segments_a=SEGMENTS, segments_b=SEGMENTS
    )
    assert abs(finer - single) / abs(single) < 1.0e-3


# ---------------------------------------------------------------------------
# 两条必须红
# ---------------------------------------------------------------------------


def test_the_midpoint_grid_gate_is_red_on_a_half_cell_offset():
    """**必须红**：截面网格写成`i/n`时整块偏半格，细丝极限的阶从2掉到1。

    偏半格之后样本的平均半径是`R − Δr/(2n)`，一阶项不再相消——
    **而算出来的数仍然完全正常**（有限、量级对、随截面缩小仍然收敛到细丝），
    只有收敛阶变了。这正是"阶要被判据钉住"的理由。
    """

    correct = _limit_orders(half_cell=True)
    shifted = _limit_orders(half_cell=False)
    assert all(1.9 <= order <= 2.1 for order in correct), correct
    assert all(0.9 <= order <= 1.1 for order in shifted), shifted


def test_the_self_bundle_gate_is_red_without_the_refusal():
    """**必须红**：一束丝与自己的互感含"同一根丝与自己"那一项，必须拒跑。

    有截面之后自感**确实有限**（GMD一类方法就是为此存在的），
    但那一项要的是导线**内部**的自感，与截面上的电流分布直接相关——
    回到本模块明写不做的那一半。**给一个静默的数比拒跑坏得多。**
    """

    coil = _coil(COIL_A, 2)
    with pytest.raises(ElectromagneticsError):
        bundle_mutual_inductance_h(coil, coil, segments_a=SEGMENTS, segments_b=SEGMENTS)
    # 两束不同的丝仍然算得出来——**门要有两侧**
    assert math.isfinite(
        bundle_mutual_inductance_h(
            coil, _coil(COIL_B, 2), segments_a=SEGMENTS, segments_b=SEGMENTS
        )
    )


# ---------------------------------------------------------------------------
# 单位边界与失败关闭
# ---------------------------------------------------------------------------


def test_the_millimetre_entry_converts_every_length_and_leaves_the_normal_alone():
    coil = RectangularSectionCoil.from_millimetres(
        mean_radius_mm=50.0,
        centre_mm=(10.0, -20.0, 80.0),
        normal=(0.0, 0.0, 3.0),
        radial_extent_mm=10.0,
        axial_extent_mm=8.0,
        radial_filaments=2,
        axial_filaments=2,
    )
    assert coil.mean_radius_m == 0.05
    assert coil.centre_m == (0.01, -0.02, 0.08)
    assert coil.radial_extent_m == 0.01
    assert coil.axial_extent_m == 0.008
    assert coil.normal == (0.0, 0.0, 1.0)


def test_a_zero_section_is_legal_and_reproduces_the_centre_filament():
    coil_a = _coil(COIL_A, 3, 0.0)
    coil_b = _coil(COIL_B, 3, 0.0)
    assert bundle_mutual_inductance_h(
        coil_a, coil_b, segments_a=SEGMENTS, segments_b=SEGMENTS
    ) == neumann_mutual_inductance_h(
        coil_a.centre_filament(),
        coil_b.centre_filament(),
        segments_a=SEGMENTS,
        segments_b=SEGMENTS,
    )


@pytest.mark.parametrize(
    "call",
    [
        # 截面跨过轴心：最内侧那圈丝半径≤0
        lambda: RectangularSectionCoil(
            mean_radius_m=0.005,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            radial_extent_m=0.010,
            axial_extent_m=0.001,
            radial_filaments=2,
            axial_filaments=2,
        ),
        lambda: RectangularSectionCoil(
            mean_radius_m=0.05,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            radial_extent_m=-0.001,
            axial_extent_m=0.001,
            radial_filaments=2,
            axial_filaments=2,
        ),
        lambda: RectangularSectionCoil(
            mean_radius_m=0.05,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            radial_extent_m=0.001,
            axial_extent_m=float("nan"),
            radial_filaments=2,
            axial_filaments=2,
        ),
        lambda: RectangularSectionCoil(
            mean_radius_m=0.05,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            radial_extent_m=0.001,
            axial_extent_m=0.001,
            radial_filaments=0,
            axial_filaments=2,
        ),
        lambda: RectangularSectionCoil(
            mean_radius_m=0.05,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            radial_extent_m=0.001,
            axial_extent_m=0.001,
            radial_filaments=2,
            axial_filaments=2.0,
        ),
        # 位形与匝数的校验由PlacedCircularLoop那一份正本给出
        lambda: RectangularSectionCoil(
            mean_radius_m=-0.05,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            radial_extent_m=0.001,
            axial_extent_m=0.001,
            radial_filaments=2,
            axial_filaments=2,
        ),
        lambda: RectangularSectionCoil(
            mean_radius_m=0.05,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 0.0),
            radial_extent_m=0.001,
            axial_extent_m=0.001,
            radial_filaments=2,
            axial_filaments=2,
        ),
        lambda: RectangularSectionCoil(
            mean_radius_m=0.05,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            radial_extent_m=0.001,
            axial_extent_m=0.001,
            radial_filaments=2,
            axial_filaments=2,
            turns=0,
        ),
        lambda: section_filaments("不是线圈"),
        lambda: _coil(COIL_A, 2).scaled_section(-1.0),
    ],
)
def test_declaration_violations_fail_closed(call):
    with pytest.raises(ElectromagneticsError):
        call()
