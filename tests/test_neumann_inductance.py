"""一般位形互感的协议门（案例判据在`tests/cases/`，本文件不重复它们）。

分工照0024／0031／0042的先例：**案例验物理，本文件验协议**——
公开面、失败关闭、单位边界、刚体不变性，以及**五条必须红的守门测试**：

1. `test_the_chord_discretisation_gate_is_red_at_second_order`
   ——把"折线弦离散只有二阶"钉成一条会红的断言；
2. `test_the_naive_accumulation_gate_is_red_on_reciprocity`
   ——把"朴素累加换掉`math.fsum`就没有逐位互易"钉住
   （**顺带一条实测**：CPython 3.12起内置`sum()`用了Neumaier补偿求和，
   与`fsum`在本模块实测的五组构型上逐位相同，所以这条门必须写在
   `total += 项`那种真正的朴素累加上，写在`sum()`上会**假绿**——
   写这条门的时候就是这么撞上的）；
3. `test_the_role_mix_gate_is_red_on_reciprocity`
   ——把"两条回路的角色写反"钉住；
4. `test_the_resolution_gate_is_red_on_an_unresolved_pair`
   ——把"未分辨构型会给出一个看起来完全正常的错数"钉住；
5. `test_the_self_inductance_gate_is_red_without_the_refusal`
   ——把"离散化的自感不会报错、只会给一个随分段数变化的数"钉住。

**边界不是免责声明，是要有门守着的。**
"""

from __future__ import annotations

import math

import pytest

from physics_engine.electromagnetics import (
    CircularLoop,
    ElectromagneticsError,
    coaxial_mutual_inductance_h,
    dipole_mutual_inductance_h,
    mutual_inductance_h,
)
from physics_engine.electromagnetics import neumann as neumann_module
from physics_engine.electromagnetics.neumann import (
    NEUMANN_PREFACTOR_H_PER_M,
    NORMAL_NORM_MIN,
    RESOLUTION_RATIO_CALIBRATION,
    RESOLUTION_RATIO_MIN,
    SEGMENTS_MIN,
    PlacedCircularLoop,
    dipole_mutual_inductance_general_h,
    filament_resolution_ratio,
    filament_samples,
    neumann_condition_number,
    neumann_mutual_inductance_h,
)

# ---------------------------------------------------------------------------
# 测试自用的构件：**故意与被验实现不共用代码**的求和器与几何工具
# ---------------------------------------------------------------------------


def _coaxial_pair(radius_a, radius_b, separation):
    return (
        PlacedCircularLoop.from_coaxial(
            CircularLoop(radius_m=radius_a, axial_position_m=0.0)
        ),
        PlacedCircularLoop.from_coaxial(
            CircularLoop(radius_m=radius_b, axial_position_m=separation)
        ),
    )


def _double_sum(samples_a, samples_b) -> list[float]:
    terms = []
    for position_a, tangent_a in samples_a:
        for position_b, tangent_b in samples_b:
            delta = tuple(position_a[axis] - position_b[axis] for axis in range(3))
            dot = sum(tangent_a[axis] * tangent_b[axis] for axis in range(3))
            terms.append(dot / math.sqrt(sum(value * value for value in delta)))
    return terms


def _ungated_mutual_h(loop_a, loop_b, segments_a, segments_b, *, accurate=True):
    """**绕开分辨门与`math.fsum`的一份对照实现**，只在本文件里用。

    它存在的唯一理由是让"没有那道门会怎样""不用fsum会怎样"成为可测量的数，
    而不是两句断言。它**不进`src/`**——生产路径上不许有绕过门的入口。

    ``accurate=False``走的是``total += 项``这种**真正的朴素累加**，
    **不是`sum()`**：CPython 3.12起`sum()`对浮点已改用Neumaier补偿求和，
    实测与`fsum`逐位相同，拿它当反例会假绿。
    """

    terms = _double_sum(
        filament_samples(loop_a, segments_a), filament_samples(loop_b, segments_b)
    )
    if accurate:
        total = math.fsum(terms)
    else:
        total = 0.0
        for value in terms:
            total += value
    arc_a = loop_a.segment_arc_length_m(segments_a)
    arc_b = loop_b.segment_arc_length_m(segments_b)
    return (loop_a.turns * loop_b.turns) * (
        NEUMANN_PREFACTOR_H_PER_M * (arc_a * arc_b) * total
    )


def _chord_mutual_h(loop_a, loop_b, segments_a, segments_b):
    """**折线弦（多边形细丝）**离散的对照实现——收敛只有代数二阶。"""

    def segments(loop, count):
        u, v = loop.plane_frame()
        centre = loop.centre_m
        radius = loop.radius_m
        vertices = [
            tuple(
                centre[axis]
                + radius
                * (
                    u[axis] * math.cos(2.0 * math.pi * index / count)
                    + v[axis] * math.sin(2.0 * math.pi * index / count)
                )
                for axis in range(3)
            )
            for index in range(count)
        ]
        out = []
        for index in range(count):
            start = vertices[index]
            end = vertices[(index + 1) % count]
            out.append(
                (
                    tuple(end[axis] - start[axis] for axis in range(3)),
                    tuple(0.5 * (start[axis] + end[axis]) for axis in range(3)),
                )
            )
        return out

    terms = []
    for element_a, midpoint_a in segments(loop_a, segments_a):
        for element_b, midpoint_b in segments(loop_b, segments_b):
            delta = tuple(midpoint_a[axis] - midpoint_b[axis] for axis in range(3))
            dot = sum(element_a[axis] * element_b[axis] for axis in range(3))
            terms.append(dot / math.sqrt(sum(value * value for value in delta)))
    return NEUMANN_PREFACTOR_H_PER_M * math.fsum(terms)


def _rotate(vector, axis, angle):
    """Rodrigues旋转，测试自用。"""

    norm = math.sqrt(sum(value * value for value in axis))
    unit = tuple(value / norm for value in axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    cross = (
        unit[1] * vector[2] - unit[2] * vector[1],
        unit[2] * vector[0] - unit[0] * vector[2],
        unit[0] * vector[1] - unit[1] * vector[0],
    )
    dot = sum(unit[axis] * vector[axis] for axis in range(3))
    return tuple(
        vector[axis] * cosine + cross[axis] * sine + unit[axis] * dot * (1.0 - cosine)
        for axis in range(3)
    )


# ---------------------------------------------------------------------------
# 公开面
# ---------------------------------------------------------------------------


def test_every_exported_name_exists_and_the_list_is_sorted():
    for name in neumann_module.__all__:
        assert hasattr(neumann_module, name), f"__all__列了不存在的名字{name!r}"
    assert neumann_module.__all__ == sorted(neumann_module.__all__)
    assert "PlacedCircularLoop" in neumann_module.__all__


def test_the_calibration_table_is_ordered_and_brackets_the_gate():
    """标定表必须单调、且门的位置真的落在表上——**表不是注释，是被验的**。"""

    ratios = [entry[0] for entry in RESOLUTION_RATIO_CALIBRATION]
    errors = [entry[1] for entry in RESOLUTION_RATIO_CALIBRATION]
    assert ratios == sorted(ratios)
    assert errors == sorted(errors, reverse=True), "分辨比越大误差必须越小"
    assert RESOLUTION_RATIO_MIN in ratios, "门开在表上没有的位置，就没人知道它对应多少误差"
    assert SEGMENTS_MIN == 3


# ---------------------------------------------------------------------------
# 退化到共轴闭式：**本模块最强的一条自洽门**
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("radius_a", "radius_b", "separation", "segments", "bound"),
    [
        (0.050, 0.050, 0.030, 64, 1.0e-14),
        (0.100, 0.020, 0.050, 32, 1.0e-14),
        (0.010, 0.010, 0.200, 16, 1.0e-13),
        (0.030, 0.030, 0.010, 96, 1.0e-13),
    ],
)
def test_the_general_quadrature_degenerates_to_the_maxwell_closed_form(
    radius_a, radius_b, separation, segments, bound
):
    """一般位形的双重求积对`inductance.coaxial_mutual_inductance_h`（AGM闭式）。

    **两条路除了物理之外没有任何共同之处**：一条是完全椭圆积分的AGM闭式，
    一条是二维求积。案例那一侧对的是仓外独立参考路径；
    本条对的是**仓内已被案例钉住的那条闭式**，两条判据互为补充。
    """

    loop_a, loop_b = _coaxial_pair(radius_a, radius_b, separation)
    measured = neumann_mutual_inductance_h(
        loop_a, loop_b, segments_a=segments, segments_b=segments
    )
    expected = coaxial_mutual_inductance_h(
        radius_a_m=radius_a, radius_b_m=radius_b, axial_separation_m=separation
    )
    assert abs(measured - expected) / abs(expected) < bound


def test_the_turns_factor_matches_the_coaxial_loop_api_bit_for_bit():
    """带匝数时与`inductance.mutual_inductance_h`的匝数因子**同一种结合次序**。"""

    turns_a, turns_b = 3, 7
    loop_a = CircularLoop(radius_m=0.050, axial_position_m=0.0, turns=turns_a)
    loop_b = CircularLoop(radius_m=0.030, axial_position_m=0.040, turns=turns_b)
    placed_a = PlacedCircularLoop.from_coaxial(loop_a)
    placed_b = PlacedCircularLoop.from_coaxial(loop_b)
    single = neumann_mutual_inductance_h(
        PlacedCircularLoop.from_coaxial(
            CircularLoop(radius_m=0.050, axial_position_m=0.0)
        ),
        PlacedCircularLoop.from_coaxial(
            CircularLoop(radius_m=0.030, axial_position_m=0.040)
        ),
        segments_a=64,
        segments_b=64,
    )
    with_turns = neumann_mutual_inductance_h(
        placed_a, placed_b, segments_a=64, segments_b=64
    )
    assert with_turns == (turns_a * turns_b) * single
    # 与共轴闭式那一侧的匝数处理一致（值不同源，只比相对差）
    closed_form = mutual_inductance_h(loop_a, loop_b)
    assert abs(with_turns - closed_form) / abs(closed_form) < 1.0e-14


# ---------------------------------------------------------------------------
# 五条必须红
# ---------------------------------------------------------------------------


def test_the_chord_discretisation_gate_is_red_at_second_order():
    """**必须红**：折线弦（多边形）离散的收敛只有代数二阶，而本实现是几何收敛。

    同一构型、同一批分段数，两者的相对误差差六个数量级。
    案例第二层的三条判据（包络／下限／带宽）各自都拦得住它——
    本条把那句话变成本文件里可直接读到的数。
    """

    loop_a, loop_b = _coaxial_pair(0.030, 0.030, 0.010)
    exact = coaxial_mutual_inductance_h(
        radius_a_m=0.030, radius_b_m=0.030, axial_separation_m=0.010
    )
    counts = (40, 48, 56, 64, 80)
    production = [
        abs(neumann_mutual_inductance_h(loop_a, loop_b, segments_a=n, segments_b=n) - exact)
        / exact
        for n in counts
    ]
    chord = [
        abs(_chord_mutual_h(loop_a, loop_b, n, n) - exact) / exact for n in counts
    ]

    def ratios(errors):
        return [
            math.log2(errors[index] / errors[index + 1])
            * 8.0
            / (counts[index + 1] - counts[index])
            for index in range(len(counts) - 1)
        ]

    production_ratios = ratios(production)
    chord_ratios = ratios(chord)
    # 本实现：每加8段降15倍以上，且比值几乎不变（几何收敛的指纹）
    assert min(production_ratios) > 3.5
    assert max(production_ratios) / min(production_ratios) < 1.1
    # 折线弦：二阶，比值一路衰减
    assert max(chord_ratios) < 0.6
    assert max(chord_ratios) / min(chord_ratios) > 1.5
    # 在最细一档上，两者差六个数量级
    assert chord[-1] / production[-1] > 1.0e6


def test_the_naive_accumulation_gate_is_red_on_reciprocity():
    """**必须红**：把`math.fsum`换成朴素累加，逐位互易当场消失。

    零容差的互易判据靠的是`fsum`**精确求和后只舍一次**因而对项的置换不变；
    朴素累加按枚举次序加，两个方向的次序不同、结果就不同（实测相对差1.5e-15）。
    """

    loop_a = PlacedCircularLoop(
        radius_m=0.050, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    loop_b = PlacedCircularLoop(
        radius_m=0.021, centre_m=(0.013, -0.007, 0.044), normal=(0.3, -0.5, 0.81)
    )
    forward = _ungated_mutual_h(loop_a, loop_b, 96, 61, accurate=False)
    reverse = _ungated_mutual_h(loop_b, loop_a, 61, 96, accurate=False)
    assert forward != reverse, "**必须红**：朴素累加那一版竟然逐位互易？这条门就白写了"
    # 而生产路径在同一构型上逐位相等
    assert neumann_mutual_inductance_h(
        loop_a, loop_b, segments_a=96, segments_b=61
    ) == neumann_mutual_inductance_h(loop_b, loop_a, segments_a=61, segments_b=96)


def test_the_role_mix_gate_is_red_on_reciprocity():
    """**必须红**：把两条回路的角色写反（分段数只用第一条的）会破坏互易。

    这一类错保持量纲、量级与远场退化阶——**只有互易性没了**。
    """

    loop_a = PlacedCircularLoop(
        radius_m=0.050, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    loop_b = PlacedCircularLoop(
        radius_m=0.021, centre_m=(0.013, -0.007, 0.044), normal=(0.3, -0.5, 0.81)
    )
    forward = _ungated_mutual_h(loop_a, loop_b, 96, 96)
    reverse = _ungated_mutual_h(loop_b, loop_a, 61, 61)
    assert forward != reverse
    # 量级完全正常——所以除了互易没有别的门看得出它
    assert abs(forward - reverse) / abs(forward) < 1.0e-6


def test_the_resolution_gate_is_red_on_an_unresolved_pair():
    """**必须红**：分辨比<2时，绕开门算出来的数**有限、量级对、符号对，但是错的**。"""

    loop_a, loop_b = _coaxial_pair(0.050, 0.050, 0.020)
    exact = coaxial_mutual_inductance_h(
        radius_a_m=0.050, radius_b_m=0.050, axial_separation_m=0.020
    )
    ratio = filament_resolution_ratio(loop_a, loop_b, segments_a=24, segments_b=24)
    assert ratio < RESOLUTION_RATIO_MIN
    ungated = _ungated_mutual_h(loop_a, loop_b, 24, 24)
    deviation = abs(ungated - exact) / exact
    assert 1.0e-5 < deviation < 1.0e-3, f"未分辨档的偏差实测应在1e-4量级，得到{deviation!r}"
    assert math.isfinite(ungated) and ungated > 0.0
    with pytest.raises(ElectromagneticsError, match="分辨比"):
        neumann_mutual_inductance_h(loop_a, loop_b, segments_a=24, segments_b=24)


def test_the_self_inductance_gate_is_red_without_the_refusal():
    """**必须红**：丝状回路的自感对数发散，而离散之后它**只是一个随N变化的数**。

    绕开门以后，同一条回路在N=48／96／192上给出三个都"看起来正常"的自感值，
    而且随N单调增长——**这正是不许静默返回的理由**。
    """

    loop = PlacedCircularLoop(
        radius_m=0.050, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    values = []
    for count in (48, 96, 192):
        samples = filament_samples(loop, count)
        terms = [
            term
            for index_a, (position_a, tangent_a) in enumerate(samples)
            for index_b, (position_b, tangent_b) in enumerate(samples)
            if index_a != index_b
            for term in [
                sum(tangent_a[axis] * tangent_b[axis] for axis in range(3))
                / math.sqrt(
                    sum((position_a[axis] - position_b[axis]) ** 2 for axis in range(3))
                )
            ]
        ]
        arc = loop.segment_arc_length_m(count)
        values.append(NEUMANN_PREFACTOR_H_PER_M * arc * arc * math.fsum(terms))
    assert all(math.isfinite(value) and value > 0.0 for value in values)
    assert values[0] < values[1] < values[2], "自感的离散值随N单调发散——这就是那条奇异性"
    # 三个数都在同一量级，**没有任何东西看得出它是发散的**
    assert values[2] / values[0] < 2.0
    with pytest.raises(ElectromagneticsError):
        neumann_mutual_inductance_h(loop, loop, segments_a=64, segments_b=64)


# ---------------------------------------------------------------------------
# 不变性与符号
# ---------------------------------------------------------------------------


def test_the_result_is_invariant_under_a_rigid_motion():
    """整体平移+旋转不改互感——**这同时钉住"结果与局部标架无关"**。

    旋转会换掉两条回路各自的平面标架（种子轴按法向最小分量选），
    所以本条不是一句物理套话：它真的换掉了采样的相位。
    """

    loop_a = PlacedCircularLoop(
        radius_m=0.050, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    loop_b = PlacedCircularLoop(
        radius_m=0.021, centre_m=(0.013, -0.007, 0.044), normal=(0.3, -0.5, 0.81)
    )
    reference = neumann_mutual_inductance_h(loop_a, loop_b, segments_a=64, segments_b=64)
    axis = (0.37, -0.82, 0.44)
    shift = (1.5, -2.25, 0.75)
    moved_a = PlacedCircularLoop(
        radius_m=loop_a.radius_m,
        centre_m=tuple(
            _rotate(loop_a.centre_m, axis, 0.9)[index] + shift[index] for index in range(3)
        ),
        normal=_rotate(loop_a.normal, axis, 0.9),
    )
    moved_b = PlacedCircularLoop(
        radius_m=loop_b.radius_m,
        centre_m=tuple(
            _rotate(loop_b.centre_m, axis, 0.9)[index] + shift[index] for index in range(3)
        ),
        normal=_rotate(loop_b.normal, axis, 0.9),
    )
    moved = neumann_mutual_inductance_h(moved_a, moved_b, segments_a=64, segments_b=64)
    assert abs(moved - reference) / abs(reference) < 1.0e-13


def test_the_sign_is_physical_not_cosmetic():
    """共面并排的两条同向回路互感为**负**，正交摆放的互感为**零**。"""

    loop_a = PlacedCircularLoop(
        radius_m=0.010, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    coplanar = PlacedCircularLoop(
        radius_m=0.020, centre_m=(0.200, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    negative = neumann_mutual_inductance_h(
        loop_a, coplanar, segments_a=64, segments_b=64
    )
    assert negative < 0.0
    assert dipole_mutual_inductance_general_h(loop_a, coplanar) < 0.0

    perpendicular = PlacedCircularLoop(
        radius_m=0.030, centre_m=(0.0, 0.0, 0.070), normal=(1.0, 0.0, 0.0)
    )
    big = PlacedCircularLoop(
        radius_m=0.050, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    null = neumann_mutual_inductance_h(big, perpendicular, segments_a=64, segments_b=64)
    assert abs(null) < 1.0e-24


def test_the_two_dipole_formulas_agree_on_a_coaxial_placement():
    """一般位形的偶极式在共轴时退化成`inductance.dipole_mutual_inductance_h`。

    **不是逐位相同**：两侧分组不同（一侧`mu0 pi r1^2 r2^2/(2 d^3)`、
    一侧`(mu0/4pi) A1 A2 [3-1]/d^3`），实测差1 ulp。
    """

    loop_a = PlacedCircularLoop(
        radius_m=0.010, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    loop_b = PlacedCircularLoop(
        radius_m=0.020, centre_m=(0.0, 0.0, 0.500), normal=(0.0, 0.0, 1.0)
    )
    general = dipole_mutual_inductance_general_h(loop_a, loop_b)
    coaxial = dipole_mutual_inductance_h(
        radius_a_m=0.010, radius_b_m=0.020, axial_separation_m=0.500
    )
    assert abs(general - coaxial) / abs(coaxial) < 1.0e-15


def test_the_condition_number_grows_with_distance_and_bounds_the_accuracy():
    """条件数随距离平方增长——**远场判据的精度由相消而不是由求积决定**。"""

    numbers = []
    for separation in (0.2, 0.4, 0.8, 1.6):
        loop_a = PlacedCircularLoop(
            radius_m=0.010, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        )
        loop_b = PlacedCircularLoop(
            radius_m=0.020, centre_m=(0.0, 0.0, separation), normal=(0.0, 0.0, 1.0)
        )
        numbers.append(
            neumann_condition_number(loop_a, loop_b, segments_a=32, segments_b=32)
        )
    assert numbers == sorted(numbers)
    for index in range(len(numbers) - 1):
        assert 3.5 < numbers[index + 1] / numbers[index] < 4.5, "距离翻倍，条件数应约翻四倍"


def test_the_condition_number_refuses_on_a_zero_sum_configuration():
    """零磁通构型上条件数按定义要除以零——拒跑，不返回inf。"""

    big = PlacedCircularLoop(
        radius_m=0.050, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    perpendicular = PlacedCircularLoop(
        radius_m=0.030, centre_m=(0.0, 0.0, 0.070), normal=(1.0, 0.0, 0.0)
    )
    # 双重和恰为0的构型要靠分段数凑；这里只要求"要么给出正数、要么失败关闭"，
    # 不许返回inf或nan——**这条判的是行为不是数值**。
    try:
        value = neumann_condition_number(
            big, perpendicular, segments_a=64, segments_b=64
        )
    except ElectromagneticsError:
        return
    assert math.isfinite(value) and value > 0.0


# ---------------------------------------------------------------------------
# 采样与几何
# ---------------------------------------------------------------------------


def test_the_samples_are_unit_tangents_on_the_circle():
    loop = PlacedCircularLoop(
        radius_m=0.037, centre_m=(0.1, -0.2, 0.3), normal=(0.3, -0.5, 0.81)
    )
    samples = filament_samples(loop, 37)
    assert len(samples) == 37
    for position, tangent in samples:
        offset = tuple(position[axis] - loop.centre_m[axis] for axis in range(3))
        assert abs(math.sqrt(sum(value * value for value in offset)) - loop.radius_m) < 1e-15
        assert abs(math.sqrt(sum(value * value for value in tangent)) - 1.0) < 1e-15
        # 切向在回路平面内：与法向正交
        assert abs(sum(tangent[axis] * loop.normal[axis] for axis in range(3))) < 1e-15
        # 位置向量的面外分量为零
        assert abs(sum(offset[axis] * loop.normal[axis] for axis in range(3))) < 1e-16


def test_the_plane_frame_is_right_handed():
    """``u × v = n``——标架的手性决定切向的绕行方向，绕反了互感符号就反了。"""

    for normal in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, -0.5, 0.81), (-0.6, 0.6, 0.5)):
        loop = PlacedCircularLoop(
            radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=normal
        )
        u, v = loop.plane_frame()
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        for axis in range(3):
            assert abs(cross[axis] - loop.normal[axis]) < 1e-15


def test_the_arc_length_and_area_are_what_the_names_say():
    loop = PlacedCircularLoop(
        radius_m=0.05, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), turns=4
    )
    assert loop.segment_arc_length_m(64) == 2.0 * math.pi * 0.05 / 64
    assert loop.magnetic_area_m2() == 4 * math.pi * 0.05 * 0.05


# ---------------------------------------------------------------------------
# 单位边界与失败关闭
# ---------------------------------------------------------------------------


def test_the_millimetre_entry_converts_lengths_and_leaves_the_normal_alone():
    """mm入口换算半径与圆心，**不换算法向**（它是方向，没有长度单位）。"""

    loop = PlacedCircularLoop.from_millimetres(
        radius_mm=50.0, centre_mm=(10.0, -20.0, 80.0), normal=(0.0, 0.0, 2.0)
    )
    assert loop.radius_m == 0.05
    assert loop.centre_m == (0.01, -0.02, 0.08)
    assert loop.normal == (0.0, 0.0, 1.0)


def test_passing_millimetres_as_metres_inflates_the_inductance():
    """把mm当米传进来不会报任何错——只会把互感放大一千倍。"""

    correct_a, correct_b = _coaxial_pair(0.050, 0.050, 0.030)
    wrong_a = PlacedCircularLoop(
        radius_m=50.0, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    wrong_b = PlacedCircularLoop(
        radius_m=50.0, centre_m=(0.0, 0.0, 30.0), normal=(0.0, 0.0, 1.0)
    )
    correct = neumann_mutual_inductance_h(
        correct_a, correct_b, segments_a=64, segments_b=64
    )
    wrong = neumann_mutual_inductance_h(wrong_a, wrong_b, segments_a=64, segments_b=64)
    assert abs(wrong / correct - 1000.0) < 1.0e-9


def test_the_normal_is_stored_normalised():
    loop = PlacedCircularLoop(
        radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 7.0)
    )
    assert loop.normal == (0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: PlacedCircularLoop(
            radius_m=0.0, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        ),
        lambda: PlacedCircularLoop(
            radius_m=-1.0, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        ),
        lambda: PlacedCircularLoop(
            radius_m=float("nan"), centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01, centre_m=(0.0, 0.0), normal=(0.0, 0.0, 1.0)
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01, centre_m="000", normal=(0.0, 0.0, 1.0)
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01, centre_m=(0.0, 0.0, float("inf")), normal=(0.0, 0.0, 1.0)
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 0.0)
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 0.5 * NORMAL_NORM_MIN),
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), turns=0
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), turns=True
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), turns=2.0
        ),
        lambda: PlacedCircularLoop(
            radius_m=0.01,
            centre_m=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            current_a=float("nan"),
        ),
        lambda: PlacedCircularLoop.from_coaxial("不是回路"),
        lambda: filament_samples(
            PlacedCircularLoop(
                radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
            ),
            2,
        ),
        lambda: filament_samples(
            PlacedCircularLoop(
                radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
            ),
            64.0,
        ),
    ],
)
def test_declaration_violations_fail_closed(call):
    with pytest.raises(ElectromagneticsError):
        call()


def test_the_dipole_approximation_refuses_concentric_loops():
    loop = PlacedCircularLoop(
        radius_m=0.01, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    other = PlacedCircularLoop(
        radius_m=0.02, centre_m=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)
    )
    with pytest.raises(ElectromagneticsError, match="偶极近似"):
        dipole_mutual_inductance_general_h(loop, other)


def test_the_current_does_not_enter_the_mutual_inductance():
    """互感是纯几何量：改电流不改M（**一位小数都不改**）。"""

    loop_a, loop_b = _coaxial_pair(0.050, 0.030, 0.040)
    charged_a = PlacedCircularLoop(
        radius_m=loop_a.radius_m,
        centre_m=loop_a.centre_m,
        normal=loop_a.normal,
        current_a=-1234.5,
    )
    assert neumann_mutual_inductance_h(
        loop_a, loop_b, segments_a=64, segments_b=64
    ) == neumann_mutual_inductance_h(charged_a, loop_b, segments_a=64, segments_b=64)
