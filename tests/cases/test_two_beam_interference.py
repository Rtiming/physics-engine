"""`case/two_beam_interference`的conformance门（轴7规则3）。

**引擎第一次算干涉**：双光束余弦定律`I = I1 + I2 + 2 sqrt(I1 I2) |gamma| cos(dphi)`，
两种光程差机制（杨氏双缝的`d x / L`、迈克尔逊的`2 d`）落到同一条定律上。

四层判据各一条测试，**每层独立**：定律本身、条纹几何、能量守恒、高条纹级次
与FTS的桥。判据数全部来自清单；本文件不复述任何公式（轴7规则4）。
"""

from __future__ import annotations

import math
from pathlib import Path

from physics_engine.optics import (
    PHASE_ACCURACY_RAD_PER_FRINGE_ORDER,
    fringe_order,
    fringe_visibility,
    michelson_path_difference_m,
    phase_difference_rad,
    spectroscopic_wavenumber_per_m,
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
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/two_beam_interference/oracle.json", root=ROOT)
LAW = MANIFEST.oracle("oracle:interference/two_beam_law")
YOUNG = MANIFEST.oracle("oracle:interference/young_double_slit")
ENERGY = MANIFEST.oracle("oracle:interference/energy_conservation")
BRIDGE = MANIFEST.oracle("oracle:interference/michelson_and_fts_bridge")


def test_the_two_beam_law_its_extremes_and_its_degenerate_limits():
    """第一层：余弦定律、可见度（含不等强与部分相干）、两条退化极限。

    对拍的是60位`Decimal`上的Taylor级数余弦——**金标不继承libm**。
    """

    phases = LAW.inputs["phase_samples_rad"]
    bright, faint = LAW.inputs["intensity_a"], LAW.inputs["intensity_b"]
    equal = LAW.inputs["equal_intensity"]
    partial = LAW.inputs["partial_coherence"]
    weak_a, weak_b = LAW.inputs["faint_intensity_a"], LAW.inputs["faint_intensity_b"]
    near_a, near_b = LAW.inputs["near_equal_a"], LAW.inputs["near_equal_b"]
    naive_a, naive_b = LAW.inputs["negative_naive_a"], LAW.inputs["negative_naive_b"]

    weak_max = two_beam_max_intensity(intensity_a=weak_a, intensity_b=weak_b)
    weak_min = two_beam_min_intensity(intensity_a=weak_a, intensity_b=weak_b)

    LAW.check_all({
        "intensity_samples": [
            two_beam_intensity(
                intensity_a=bright, intensity_b=faint, phase_difference_rad=phase
            )
            for phase in phases
        ],
        "intensity_samples_partial_coherence": [
            two_beam_intensity(
                intensity_a=equal,
                intensity_b=equal,
                phase_difference_rad=phase,
                coherence_modulus=partial,
            )
            for phase in phases
        ],
        "max_intensity": two_beam_max_intensity(intensity_a=bright, intensity_b=faint),
        "min_intensity": two_beam_min_intensity(intensity_a=bright, intensity_b=faint),
        "max_intensity_partial": two_beam_max_intensity(
            intensity_a=equal, intensity_b=equal, coherence_modulus=partial
        ),
        "min_intensity_partial": two_beam_min_intensity(
            intensity_a=equal, intensity_b=equal, coherence_modulus=partial
        ),
        "visibility_unequal": fringe_visibility(intensity_a=bright, intensity_b=faint),
        "visibility_equal": fringe_visibility(intensity_a=equal, intensity_b=equal),
        "visibility_partial_equal": fringe_visibility(
            intensity_a=equal, intensity_b=equal, coherence_modulus=partial
        ),
        "visibility_partial_unequal": fringe_visibility(
            intensity_a=bright, intensity_b=faint, coherence_modulus=partial
        ),
        "visibility_faint_closed_form": fringe_visibility(
            intensity_a=weak_a, intensity_b=weak_b
        ),
        "visibility_faint_from_extremes": (weak_max - weak_min) / (weak_max + weak_min),
        "min_intensity_near_equal": two_beam_min_intensity(
            intensity_a=near_a, intensity_b=near_b
        ),
        "intensity_at_pi_near_equal": two_beam_intensity(
            intensity_a=near_a, intensity_b=near_b, phase_difference_rad=math.pi
        ),
        "min_intensity_negative_naive_case": two_beam_min_intensity(
            intensity_a=naive_a, intensity_b=naive_b
        ),
        "intensity_with_second_beam_absent": [
            two_beam_intensity(
                intensity_a=bright, intensity_b=0.0, phase_difference_rad=phase
            )
            for phase in phases
        ],
        "intensity_with_zero_coherence": [
            two_beam_intensity(
                intensity_a=equal,
                intensity_b=equal,
                phase_difference_rad=phase,
                coherence_modulus=0.0,
            )
            for phase in phases
        ],
        "visibility_with_second_beam_absent": fringe_visibility(
            intensity_a=bright, intensity_b=0.0
        ),
        "visibility_with_zero_coherence": fringe_visibility(
            intensity_a=equal, intensity_b=equal, coherence_modulus=0.0
        ),
    })


def test_young_fringe_spacing_against_the_exact_two_point_source_geometry():
    """第二层：条纹间距闭式、傍轴与精确光程差、**傍轴适用性的可计算判据**。

    第一亮纹的位置由本测试在float64上**二分内核的精确光程差**求出，
    与生成器在60位上二分它自己的定义式求出的位置对拍。
    """

    separation = YOUNG.inputs["slit_separation_m"]
    distance = YOUNG.inputs["screen_distance_m"]
    wavelength = YOUNG.inputs["wavelength_m"]
    positions = YOUNG.inputs["screen_positions_m"]
    deviation_positions = YOUNG.inputs["deviation_positions_m"]
    bright, faint = YOUNG.inputs["intensity_a"], YOUNG.inputs["intensity_b"]

    spacing = young_fringe_spacing_m(
        wavelength_m=wavelength,
        slit_separation_m=separation,
        screen_distance_m=distance,
    )

    def exact(position: float) -> float:
        return young_exact_path_difference_m(
            screen_position_m=position,
            slit_separation_m=separation,
            screen_distance_m=distance,
        )

    def paraxial(position: float) -> float:
        return young_paraxial_path_difference_m(
            screen_position_m=position,
            slit_separation_m=separation,
            screen_distance_m=distance,
        )

    low, high = YOUNG.inputs["bisection_bracket_m"]
    for _ in range(YOUNG.inputs["bisection_steps"]):
        middle = 0.5 * (low + high)
        if exact(middle) < wavelength:
            low = middle
        else:
            high = middle
    first_bright = 0.5 * (low + high)

    measured = [1.0 - exact(position) / paraxial(position) for position in deviation_positions]
    estimate = [
        young_paraxial_relative_deviation(
            screen_position_m=position,
            slit_separation_m=separation,
            screen_distance_m=distance,
        )
        for position in deviation_positions
    ]

    YOUNG.check_all({
        "fringe_spacing_m": spacing,
        "paraxial_path_difference_m": [paraxial(position) for position in positions],
        "exact_path_difference_m": [exact(position) for position in positions],
        "screen_intensity_samples": [
            two_beam_intensity(
                intensity_a=bright,
                intensity_b=faint,
                phase_difference_rad=phase_difference_rad(
                    path_difference_m=paraxial(position), wavelength_m=wavelength
                ),
            )
            for position in positions
        ],
        "paraxial_relative_deviation_measured": measured,
        "paraxial_relative_deviation_estimate": estimate,
        "deviation_estimate_over_measured": [
            one / other for one, other in zip(estimate, measured, strict=True)
        ],
        "first_bright_fringe_relative_shift": first_bright / spacing - 1.0,
    })


def test_the_fringe_average_returns_the_incoherent_sum():
    """第三层：**能量守恒**——干涉重新分配能量，不创造能量。

    相位平均那条与条纹几何**无关**（独立门）；空间平均那条同时用到条纹间距
    与整条相位链（耦合门），两者分开申报。
    """

    count = ENERGY.inputs["sample_count"]
    spacing = ENERGY.inputs["fringe_spacing_m"]
    separation = ENERGY.inputs["slit_separation_m"]
    distance = ENERGY.inputs["screen_distance_m"]
    wavelength = ENERGY.inputs["wavelength_m"]
    bright, faint = ENERGY.inputs["intensity_a"], ENERGY.inputs["intensity_b"]

    phase_averages = []
    declared = []
    for first, second, coherence in ENERGY.inputs["configurations"]:
        phase_averages.append(
            math.fsum(
                two_beam_intensity(
                    intensity_a=first,
                    intensity_b=second,
                    phase_difference_rad=2.0 * math.pi * index / count,
                    coherence_modulus=coherence,
                )
                for index in range(count)
            )
            / count
        )
        declared.append(two_beam_mean_intensity(intensity_a=first, intensity_b=second))

    screen_average = (
        math.fsum(
            two_beam_intensity(
                intensity_a=bright,
                intensity_b=faint,
                phase_difference_rad=phase_difference_rad(
                    path_difference_m=young_paraxial_path_difference_m(
                        screen_position_m=spacing * index / count,
                        slit_separation_m=separation,
                        screen_distance_m=distance,
                    ),
                    wavelength_m=wavelength,
                ),
            )
            for index in range(count)
        )
        / count
    )

    ENERGY.check_all({
        "phase_average_intensity": phase_averages,
        "declared_mean_intensity": declared,
        "screen_average_over_one_fringe": screen_average,
    })


def test_michelson_at_high_fringe_order_and_the_bridge_to_the_fts_line_shape():
    """第四层：另一种光程差机制（`2 d`）+ **与本子包已有FTS的桥**。

    桥的内容：ILS首零`1/(2L)`按定义是分辨极限，它的物理意义是——波数差一个
    `1/(2L)`的两束光在整个扫描程`2L`上相对相位恰好滑过一个整条纹。
    **本条不引入任何FFT**，只用两条余弦的相位差。
    """

    wavelength = BRIDGE.inputs["wavelength_m"]
    displacement = BRIDGE.inputs["mirror_displacement_m"]
    max_opd = BRIDGE.inputs["fts_max_opd_m"]
    bright, faint = BRIDGE.inputs["intensity_a"], BRIDGE.inputs["intensity_b"]

    path_difference = michelson_path_difference_m(mirror_displacement_m=displacement)
    phase = phase_difference_rad(
        path_difference_m=path_difference, wavelength_m=wavelength
    )

    first_zero = unapodised_first_zero_per_m(max_opd)
    shifted_wavelength = 1.0 / (spectroscopic_wavenumber_per_m(wavelength) + first_zero)
    swing = 2.0 * max_opd

    BRIDGE.check_all({
        "michelson_path_difference_m": path_difference,
        "michelson_fringe_order": fringe_order(
            path_difference_m=path_difference, wavelength_m=wavelength
        ),
        "michelson_phase_rad": phase,
        "michelson_intensity": two_beam_intensity(
            intensity_a=bright, intensity_b=faint, phase_difference_rad=phase
        ),
        "ils_first_zero_per_m": first_zero,
        "shifted_wavelength_m": shifted_wavelength,
        "fringe_slip_over_full_scan_rad": phase_difference_rad(
            path_difference_m=swing, wavelength_m=shifted_wavelength
        )
        - phase_difference_rad(path_difference_m=swing, wavelength_m=wavelength),
        "declared_phase_accuracy_matches_manifest": (
            PHASE_ACCURACY_RAD_PER_FRINGE_ORDER
            == BRIDGE.inputs["phase_accuracy_rad_per_fringe_order"]
        ),
    })


def test_the_pattern_stays_inside_its_own_extremes_and_never_goes_negative():
    """三条不靠清单的结构判据——它们抓清单抓不到的那一类错。

    清单冻结的是**点**：一个把余弦写成正弦的实现在若干采样点上可能碰巧对，
    但它过不了"图样被自己的极值夹住"这一条；一个把`2 sqrt(I1 I2)`写成
    `2 (I1 + I2)`的实现会算出负强度，而负强度是物理上不存在的东西。
    """

    bright, faint = LAW.inputs["intensity_a"], LAW.inputs["intensity_b"]
    top = two_beam_max_intensity(intensity_a=bright, intensity_b=faint)
    bottom = two_beam_min_intensity(intensity_a=bright, intensity_b=faint)
    for index in range(257):
        phase = 2.0 * math.pi * index / 256.0
        value = two_beam_intensity(
            intensity_a=bright, intensity_b=faint, phase_difference_rad=phase
        )
        assert bottom - 1.0e-15 <= value <= top + 1.0e-15, (
            f"dphi={phase!r}处强度{value!r}跑出了[{bottom!r}, {top!r}]"
        )
        mirrored = two_beam_intensity(
            intensity_a=bright, intensity_b=faint, phase_difference_rad=-phase
        )
        assert mirrored == value, f"图样在dphi={phase!r}处不是偶函数"

    naive_a, naive_b = LAW.inputs["negative_naive_a"], LAW.inputs["negative_naive_b"]
    assert (naive_a + naive_b - 2.0 * math.sqrt(naive_a * naive_b)) < 0.0, (
        "这一对强度本该让朴素式算出负强度——它不再成立说明清单里的标本失效了，"
        "换一对（决策0044第四节记了怎么找）"
    )
    assert two_beam_min_intensity(intensity_a=naive_a, intensity_b=naive_b) >= 0.0, (
        "不相消的那条路返回了负强度——恒等变形的两项本该都非负"
    )
