"""`optics/interference.py`的协议门（案例判据在`tests/cases/`，本文件不重复它们）。

分工照`tests/test_optics.py`立的先例：**案例验物理，本文件验协议**——
公开面、失败关闭、单位边界的活标本、申报精度、以及**那些没有门能抓的错法
在这里被如实写下来**（本仓禁止把已知的洞藏起来）。
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

import physics_engine
from physics_engine import optics
from physics_engine.optics import (
    FULL_COHERENCE,
    MICHELSON_OPD_PER_MIRROR_DISPLACEMENT,
    PHASE_ACCURACY_RAD_PER_FRINGE_ORDER,
    OpticsError,
    fringe_order,
    fringe_visibility,
    michelson_path_difference_m,
    phase_difference_rad,
    two_beam_intensity,
    two_beam_max_intensity,
    two_beam_mean_intensity,
    two_beam_min_intensity,
    unapodised_first_zero_per_m,
    young_exact_path_difference_m,
    young_fringe_spacing_m,
    young_paraxial_path_difference_m,
    young_paraxial_relative_deviation,
)

WAVELENGTH_M = 632.8e-9
SLIT_SEPARATION_M = 0.25e-3
SCREEN_DISTANCE_M = 1.20


# --- 公开面 ---------------------------------------------------------------


def test_interference_names_are_exported_and_stay_out_of_the_package_facade():
    """新公开名只进子包`__all__`，不进`physics_engine.__all__`（域隔离门③的另一面）。"""

    for name in (
        "FULL_COHERENCE",
        "MICHELSON_OPD_PER_MIRROR_DISPLACEMENT",
        "PHASE_ACCURACY_RAD_PER_FRINGE_ORDER",
        "two_beam_intensity",
        "fringe_visibility",
        "young_fringe_spacing_m",
        "michelson_path_difference_m",
    ):
        assert name in optics.__all__, f"{name!r}没进子包的__all__"
        assert hasattr(optics, name)
    assert not hasattr(physics_engine, "two_beam_intensity")
    assert not set(optics.__all__) & set(physics_engine.__all__)


# --- 失败关闭 -------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: two_beam_intensity(
            intensity_a=-1.0, intensity_b=1.0, phase_difference_rad=0.0
        ),
        lambda: two_beam_intensity(
            intensity_a=1.0, intensity_b=float("nan"), phase_difference_rad=0.0
        ),
        lambda: two_beam_intensity(
            intensity_a=1.0, intensity_b=1.0, phase_difference_rad=float("inf")
        ),
        lambda: two_beam_intensity(
            intensity_a=1.0,
            intensity_b=1.0,
            phase_difference_rad=0.0,
            coherence_modulus=1.0000001,
        ),
        lambda: two_beam_intensity(
            intensity_a=1.0,
            intensity_b=1.0,
            phase_difference_rad=0.0,
            coherence_modulus=-1e-16,
        ),
        lambda: fringe_visibility(intensity_a=0.0, intensity_b=0.0),
        lambda: phase_difference_rad(path_difference_m=1.0, wavelength_m=0.0),
        lambda: phase_difference_rad(path_difference_m=float("inf"), wavelength_m=1e-6),
        lambda: fringe_order(path_difference_m=1.0, wavelength_m=-1e-6),
        lambda: young_fringe_spacing_m(
            wavelength_m=WAVELENGTH_M, slit_separation_m=0.0, screen_distance_m=1.0
        ),
        lambda: young_fringe_spacing_m(
            wavelength_m=WAVELENGTH_M, slit_separation_m=1e-3, screen_distance_m=-1.0
        ),
        lambda: young_exact_path_difference_m(
            screen_position_m=float("nan"),
            slit_separation_m=1e-3,
            screen_distance_m=1.0,
        ),
        lambda: michelson_path_difference_m(mirror_displacement_m=float("nan")),
    ],
)
def test_domain_violations_fail_closed(call):
    with pytest.raises(OpticsError):
        call()


def test_a_coherence_modulus_above_one_is_refused_by_name():
    """`|gamma| > 1`不是“参数偏大”，是这个数不是一个相干度（Cauchy-Schwarz）。"""

    with pytest.raises(OpticsError, match="Cauchy-Schwarz"):
        fringe_visibility(intensity_a=1.0, intensity_b=1.0, coherence_modulus=1.5)


# --- 相干性：`V = |gamma|`是`|gamma|`的操作定义 ----------------------------


@pytest.mark.parametrize("modulus", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_visibility_equals_the_coherence_modulus_for_equal_beams(modulus):
    assert fringe_visibility(
        intensity_a=2.0, intensity_b=2.0, coherence_modulus=modulus
    ) == pytest.approx(modulus, rel=1e-15, abs=0.0)


def test_visibility_factorises_into_intensity_mismatch_times_coherence():
    """`V`是两个因子之积：只验等强构型时，把|gamma|当成整个可见度的实现照样绿。"""

    mismatch = fringe_visibility(intensity_a=3.0, intensity_b=0.75)
    both = fringe_visibility(
        intensity_a=3.0, intensity_b=0.75, coherence_modulus=0.4
    )
    assert both == pytest.approx(mismatch * 0.4, rel=1e-15)


def test_full_coherence_is_the_declared_default():
    assert FULL_COHERENCE == 1.0
    assert two_beam_intensity(
        intensity_a=3.0, intensity_b=0.75, phase_difference_rad=0.7
    ) == two_beam_intensity(
        intensity_a=3.0,
        intensity_b=0.75,
        phase_difference_rad=0.7,
        coherence_modulus=FULL_COHERENCE,
    )


# --- 能量：平均强度与相干度无关 -------------------------------------------


@pytest.mark.parametrize("modulus", [0.0, 0.37, 1.0])
def test_the_declared_mean_intensity_does_not_depend_on_coherence(modulus):
    """干涉重新分配能量不创造能量——平均值与`|gamma|`无关，逐位相同。"""

    mean = two_beam_mean_intensity(intensity_a=3.0, intensity_b=0.75)
    assert mean == 3.75
    top = two_beam_max_intensity(
        intensity_a=3.0, intensity_b=0.75, coherence_modulus=modulus
    )
    bottom = two_beam_min_intensity(
        intensity_a=3.0, intensity_b=0.75, coherence_modulus=modulus
    )
    assert 0.5 * (top + bottom) == pytest.approx(mean, rel=1e-15)


# --- 相消：不相消的暗纹式 -------------------------------------------------


def test_the_naive_dark_fringe_formula_can_return_a_negative_intensity():
    """朴素式``I1 + I2 - 2 sqrt(I1 I2)``在近等强时会算出**负强度**。

    负强度不是精度问题，是算出了一个不存在的东西。内核那条恒等变形
    ``(sqrt(I1) - sqrt(I2))^2 + 2 sqrt(I1 I2)(1 - |gamma|)``两项都非负，
    不可能为负——这条门守的就是那个写法不许被“简化”回去。
    """

    first, second = 6.887858857269796, 6.887858857237471
    assert first + second - 2.0 * math.sqrt(first * second) < 0.0
    assert two_beam_min_intensity(intensity_a=first, intensity_b=second) >= 0.0

    for step in range(1, 64):
        other = math.nextafter(first, math.inf)
        for _ in range(step):
            other = math.nextafter(other, math.inf)
        assert two_beam_min_intensity(intensity_a=first, intensity_b=other) >= 0.0


def test_the_exact_path_difference_beats_the_definition_written_literally():
    """精确光程差的定义式在float64上直接相减会丢有效位，恒等变形不会。

    近轴时两个根号都约等于`L = 1.2 m`，而它们的差只有约6.33e-7 m——
    直接相减把约6位有效数字丢在减法里（实测两条路差8.3e-11相对）。
    **这条门量的正是那6位。**
    """

    position = 3.03744e-3
    half = 0.5 * SLIT_SEPARATION_M
    literal = math.sqrt((position + half) ** 2 + SCREEN_DISTANCE_M**2) - math.sqrt(
        (position - half) ** 2 + SCREEN_DISTANCE_M**2
    )
    identity = young_exact_path_difference_m(
        screen_position_m=position,
        slit_separation_m=SLIT_SEPARATION_M,
        screen_distance_m=SCREEN_DISTANCE_M,
    )
    lost = abs(literal - identity) / identity
    assert lost > 1e-11, (
        f"定义式与恒等变形只差{lost!r}——若这条不成立，说明相消的标本失效了"
    )
    assert lost < 1e-8, f"差{lost!r}太大，不像相消像写错了公式"


# --- 申报精度：随条纹级次线性退化 -----------------------------------------


def _exact_reduced_phase(path_difference_m: float, wavelength_m: float) -> float:
    """精确约化的相位（`Fraction`），**与`tests/cases/`那份生成器不共用代码**。"""

    ratio = Fraction(path_difference_m) / Fraction(wavelength_m)
    return 2.0 * math.pi * float(ratio - (ratio.numerator // ratio.denominator))


def test_the_declared_phase_accuracy_per_fringe_order_is_a_bound_and_is_not_loose():
    """**申报的精度要被验，不是被声称**（0031第3.2节立的形制）。

    两个方向都断言：`PHASE_ACCURACY_RAD_PER_FRINGE_ORDER * N`必须是上界，
    也不许比实测松一个数量级以上——写松了的申报会让案例容差跟着虚胖。
    """

    worst_ratio = 0.0
    worst_at = 0.0
    for index in range(1, 400):
        path_difference = index * 2.5e-5
        order = fringe_order(
            path_difference_m=path_difference, wavelength_m=WAVELENGTH_M
        )
        reference = _exact_reduced_phase(path_difference, WAVELENGTH_M)
        computed = phase_difference_rad(
            path_difference_m=path_difference, wavelength_m=WAVELENGTH_M
        )
        slope = abs(math.sin(reference))
        if slope < 1e-3:
            continue
        implied = abs(math.cos(computed) - math.cos(reference)) / slope
        ratio = implied / (PHASE_ACCURACY_RAD_PER_FRINGE_ORDER * order)
        if ratio > worst_ratio:
            worst_ratio, worst_at = ratio, order
    assert worst_ratio <= 1.0, (
        f"申报系数{PHASE_ACCURACY_RAD_PER_FRINGE_ORDER!r}不是上界："
        f"级次{worst_at!r}处实测/申报 = {worst_ratio!r}"
    )
    assert worst_ratio > 0.1, (
        f"实测最坏只有申报的{worst_ratio!r}倍——申报写松了一个数量级以上，"
        "按实测收紧（案例容差逐字用这个系数乘条纹级次）"
    )


# --- 单位边界的活标本 -----------------------------------------------------


def test_swapping_the_screen_distance_and_the_slit_separation_changes_the_answer():
    """`dx = lambda L / d`里L与d颠倒：量纲仍是米、不报任何错，差``(L/d)^2``倍。

    所以三个参数都是**带单位角色的关键字**，不是位置参数。
    """

    right = young_fringe_spacing_m(
        wavelength_m=WAVELENGTH_M,
        slit_separation_m=SLIT_SEPARATION_M,
        screen_distance_m=SCREEN_DISTANCE_M,
    )
    wrong = young_fringe_spacing_m(
        wavelength_m=WAVELENGTH_M,
        slit_separation_m=SCREEN_DISTANCE_M,
        screen_distance_m=SLIT_SEPARATION_M,
    )
    assert right / wrong == pytest.approx(
        (SCREEN_DISTANCE_M / SLIT_SEPARATION_M) ** 2, rel=1e-12
    )


def test_the_michelson_factor_two_and_the_fts_scan_length_agree():
    """要扫到最大光程差`L`，动镜只需走`L/2`——这条把两个模块的口径钉在一起。

    `fts.max_opd_m`是**光程差**，迈克尔逊的`2 d`把动镜位移折成光程差。
    把动镜位移直接当光程差用是2倍的条纹级次错，而2倍“看起来完全合理”。
    """

    assert MICHELSON_OPD_PER_MIRROR_DISPLACEMENT == 2.0
    max_opd = 0.05
    assert michelson_path_difference_m(mirror_displacement_m=max_opd / 2.0) == max_opd
    assert unapodised_first_zero_per_m(max_opd) == pytest.approx(
        1.0 / (2.0 * max_opd), rel=1e-15
    )


def test_degrees_passed_as_radians_cannot_be_detected_by_this_module():
    """**已知的洞，如实写下来**：调用方把度数传进`phase_difference_rad`参数位，
    本模块无法察觉——它只是一个有限浮点数。

    唯一的防线是参数名带`_rad`。本条不是在验一个功能，是在把这个洞钉在测试里，
    免得半年后有人以为“这里有门守着”。决策0044第五节把它列为
    **本轨道唯一一条没有任何门能抓的错法**。
    """

    in_radians = two_beam_intensity(
        intensity_a=3.0, intensity_b=0.75, phase_difference_rad=math.pi
    )
    in_degrees = two_beam_intensity(
        intensity_a=3.0, intensity_b=0.75, phase_difference_rad=180.0
    )
    assert in_radians != in_degrees
    assert math.isfinite(in_degrees), "传度数不会报错——这正是问题所在"


# --- 傍轴适用性 -----------------------------------------------------------


def test_the_paraxial_path_difference_always_overestimates_the_exact_one():
    """傍轴式偏大：精确光程差比它小，且偏差随x单调增。符号错了这条当场红。"""

    spacing = young_fringe_spacing_m(
        wavelength_m=WAVELENGTH_M,
        slit_separation_m=SLIT_SEPARATION_M,
        screen_distance_m=SCREEN_DISTANCE_M,
    )
    previous = -1.0
    for multiplier in (0.25, 1.0, 5.0, 20.0, 40.0):
        position = multiplier * spacing
        exact = young_exact_path_difference_m(
            screen_position_m=position,
            slit_separation_m=SLIT_SEPARATION_M,
            screen_distance_m=SCREEN_DISTANCE_M,
        )
        paraxial = young_paraxial_path_difference_m(
            screen_position_m=position,
            slit_separation_m=SLIT_SEPARATION_M,
            screen_distance_m=SCREEN_DISTANCE_M,
        )
        assert 0.0 < exact < paraxial, f"x={position!r}处精确光程差没有小于傍轴式"
        deviation = young_paraxial_relative_deviation(
            screen_position_m=position,
            slit_separation_m=SLIT_SEPARATION_M,
            screen_distance_m=SCREEN_DISTANCE_M,
        )
        assert deviation > previous, "偏差没有随x单调增"
        previous = deviation


def test_the_pattern_is_even_and_periodic_in_the_phase():
    """图样对相位是偶函数且以2pi为周期——把cos写成sin这条当场红。"""

    for index in range(1, 33):
        phase = math.pi * index / 16.0
        value = two_beam_intensity(
            intensity_a=3.0, intensity_b=0.75, phase_difference_rad=phase
        )
        assert two_beam_intensity(
            intensity_a=3.0, intensity_b=0.75, phase_difference_rad=-phase
        ) == value
        shifted = two_beam_intensity(
            intensity_a=3.0,
            intensity_b=0.75,
            phase_difference_rad=phase + 2.0 * math.pi,
        )
        assert shifted == pytest.approx(value, abs=1e-14)
