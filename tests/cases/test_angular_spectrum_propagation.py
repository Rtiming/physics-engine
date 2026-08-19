"""`case/angular_spectrum_propagation`的conformance门（轴7规则3）。

**角谱那一侧的第一条冻结金标**。`cases/double_slit_propagated`冻的是
夫琅禾费那一支（单次FFT、傍轴）；角谱的平面波本征函数、倏逝波衰减、
``z -> 0``逐位、半群四条到今天为止只活在`tests/test_optics_propagation.py`里，
而本仓的分档是``tests/``守实现、**案例冻判据**——capability_ledger的S4.5
因此一直是`partial`（0086第八节自己登记的GAP）。本案例把它冻下来。

四条oracle里**有两条是傍轴形制给不出的**：

* `plane_wave_eigenvalues`：本征值里的根号是精确的，傍轴把它展到二阶；
* `evanescent_power_loss`：倏逝分量按``exp(-k z sqrt(q-1))``衰减，
  于是含倏逝内容的场传过去**掉能量**；而傍轴传递函数的模恒为1，
  同一个构型上给**恰好1.0**。这一条只是一个数，验它不需要复述任何公式。

判据数全部来自清单；本文件不复述任何**正确的**公式（轴7规则4）。
两条必红里出现的傍轴表达式是**故意错的那一个**——它不是oracle的公式，
是oracle要把自己与之分开的那个东西。
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import pytest

from physics_engine.optics.errors import OpticsError
from physics_engine.optics.field import (
    ComplexField2D,
    complex_from_components,
    complex_to_components,
    fft2,
    ifft2,
    signed_frequency_indices,
)
from physics_engine.optics.propagation import (
    incident_power,
    propagate_angular_spectrum,
    rectangular_aperture,
    spatial_coordinates_m,
    transfer_function_max_distance_m,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/angular_spectrum_propagation/oracle.json", root=ROOT)
EIGENVALUES = MANIFEST.oracle("oracle:angular_spectrum/plane_wave_eigenvalues")
EVANESCENT = MANIFEST.oracle("oracle:angular_spectrum/evanescent_power_loss")
HOLE = MANIFEST.oracle("oracle:angular_spectrum/rectangular_hole_field")
SELF_CONSISTENCY = MANIFEST.oracle("oracle:angular_spectrum/self_consistency")


def _plane_wave(bin_index, *, count, pitch):
    """第`bin_index`格的平面波``exp(2 pi i f x)``，`f`按有符号频率序号取。"""

    coordinates = spatial_coordinates_m(count, pitch)
    frequency = signed_frequency_indices(count)[bin_index] / (count * pitch)
    return ComplexField2D.from_function(
        1,
        count,
        lambda row, column: cmath.exp(
            complex(0.0, 2.0 * math.pi * frequency * coordinates[column])
        ),
    )


def _eigenvalues(bins):
    """把每个bin的平面波传过去，读出**本征值**``U_out / U_in``。

    平面波是角谱传播的本征函数，所以这个比值在整条线上是**同一个复数**——
    本函数顺带断言这件事（取最大偏差，它必须停在浮点地板上），
    再把那个复数交给清单去判。比值不是常数的话，被验对象根本不是角谱。
    """

    setup = EIGENVALUES.inputs
    values = []
    for bin_index in bins:
        wave = _plane_wave(bin_index, count=setup["count"], pitch=setup["pitch_m"])
        result = propagate_angular_spectrum(
            wave,
            wavelength_m=setup["wavelength_m"],
            distance_m=setup["distance_m"],
            pitch_x_m=setup["pitch_m"],
            pitch_y_m=setup["pitch_m"],
        )
        ratios = [
            got / want
            for got, want in zip(result.field.rows[0], wave.rows[0], strict=True)
        ]
        spread = max(abs(ratio - ratios[0]) for ratio in ratios)
        assert spread <= 1.0e-13, f"bin={bin_index}：本征值在线上不是常数（散布{spread!r}）"
        values.append(ratios[0])
    return values


def _rectangular_hole():
    """两个方向都有界的矩孔——**用引擎的构件**，与生成器的几何算法各走各的。"""

    setup = HOLE.inputs
    pitch = setup["pitch_m"]
    half_x = ((setup["aperture_samples_x"] - 1) / 2.0 + setup["edge_offset_in_pitches"]) * pitch
    half_y = ((setup["aperture_samples_y"] - 1) / 2.0 + setup["edge_offset_in_pitches"]) * pitch
    return rectangular_aperture(
        row_count=setup["count"],
        column_count=setup["count"],
        pitch_x_m=pitch,
        pitch_y_m=pitch,
        half_width_x_m=half_x,
        half_width_y_m=half_y,
        centre_x_m=0.5 * pitch,
        centre_y_m=0.5 * pitch,
    )


def _propagated_hole(distance_m=None):
    setup = HOLE.inputs
    mask = _rectangular_hole()
    return mask, propagate_angular_spectrum(
        mask,
        wavelength_m=setup["wavelength_m"],
        distance_m=setup["distance_m"] if distance_m is None else distance_m,
        pitch_x_m=setup["pitch_m"],
        pitch_y_m=setup["pitch_m"],
    )


def _evanescent_slit():
    setup = EVANESCENT.inputs
    pitch = setup["pitch_m"]
    half = ((setup["slit_samples"] - 1) / 2.0 + setup["edge_offset_in_pitches"]) * pitch
    return rectangular_aperture(
        row_count=setup["rows"],
        column_count=setup["count"],
        pitch_x_m=pitch,
        pitch_y_m=pitch,
        half_width_x_m=half,
        half_width_y_m=0.75 * pitch,
        centre_x_m=0.5 * pitch,
    )


def test_a_plane_wave_only_picks_up_the_exact_transfer_phase():
    """平面波本征函数：传播分量与倏逝分量各四个bin，本征值逐点对清单。

    **这是唯一能把"精确角谱"与"傍轴近似"分开的一类判据**——
    半群、能量守恒、``z -> 0``退化三条对两者一视同仁。
    """

    EIGENVALUES.check_all(
        {
            "propagating_eigenvalue_components": [
                complex_to_components(value)
                for value in _eigenvalues(EIGENVALUES.inputs["propagating_bins"])
            ],
            "evanescent_eigenvalue_components": [
                complex_to_components(value)
                for value in _eigenvalues(EIGENVALUES.inputs["evanescent_bins"])
            ],
            "paraxial_eigenvalue_gap": _paraxial_gaps(),
        }
    )


def _paraxial_gaps():
    """**必红那一半**：冻结的本征值与**傍轴**本征值之差的模，逐bin。

    本函数里那条``exp(i k z (1 - q/2))``是**故意错的公式**，不是oracle的公式——
    oracle冻的是精确值，本函数只是把"错的那一个离它多远"量出来。
    精确值直接从清单读，**不重算**：重算就成了拿判据去验判据。
    """

    setup = EIGENVALUES.inputs
    wavenumber = 2.0 * math.pi / setup["wavelength_m"]
    frozen = (
        EIGENVALUES.expected["propagating_eigenvalue_components"]
        + EIGENVALUES.expected["evanescent_eigenvalue_components"]
    )
    bins = list(setup["propagating_bins"]) + list(setup["evanescent_bins"])
    gaps = []
    for components, bin_index in zip(frozen, bins, strict=True):
        frequency = signed_frequency_indices(setup["count"])[bin_index] / (
            setup["count"] * setup["pitch_m"]
        )
        reduced = (setup["wavelength_m"] * frequency) ** 2
        paraxial = cmath.exp(
            complex(0.0, wavenumber * setup["distance_m"] * (1.0 - 0.5 * reduced))
        )
        gaps.append(abs(complex_from_components(components) - paraxial))
    return gaps


def test_the_evanescent_content_really_costs_power_and_a_paraxial_form_cannot_see_it():
    """倏逝功率损失——**本案例最硬的一条角谱独有判据，而且它只是一个数**。

    亚波长网格上4采样宽的缝：32格里只有11格是传播的，其余21格倏逝。
    精确角谱把它们按``exp(-k z sqrt(q-1))``压下去，于是能量比 < 1；
    傍轴传递函数的模恒为1，同一个构型上给**恰好1.0**。
    必红与判据写在同一条测试里，因为它们是同一句话的两半。
    """

    setup = EVANESCENT.inputs
    mask = _evanescent_slit()
    assert sum(1 for value in mask.rows[0] if abs(value) > 0.5) == setup["slit_samples"]
    result = propagate_angular_spectrum(
        mask,
        wavelength_m=setup["wavelength_m"],
        distance_m=setup["distance_m"],
        pitch_x_m=setup["pitch_m"],
        pitch_y_m=setup["pitch_m"],
    )
    before = incident_power(mask, pitch_x_m=setup["pitch_m"], pitch_y_m=setup["pitch_m"])

    #: 必红：把精确根号换成傍轴展开，同一条路再走一遍。
    #: ``exp(i k z (1 - q/2))``是**故意错的公式**（傍轴），它的模逐bin等于1。
    wavenumber = 2.0 * math.pi / setup["wavelength_m"]
    spectrum = fft2(mask)
    frequencies = [
        index / (setup["count"] * setup["pitch_m"])
        for index in signed_frequency_indices(setup["count"])
    ]
    paraxial = ifft2(
        ComplexField2D(
            tuple(
                tuple(
                    value
                    * cmath.exp(
                        complex(
                            0.0,
                            wavenumber
                            * setup["distance_m"]
                            * (1.0 - 0.5 * (setup["wavelength_m"] * frequencies[column]) ** 2),
                        )
                    )
                    for column, value in enumerate(row)
                )
                for row in spectrum.rows
            )
        )
    )
    paraxial_power = (
        sum(sum(row) for row in paraxial.intensity_rows())
        * setup["pitch_m"]
        * setup["pitch_m"]
    )

    EVANESCENT.check_all(
        {
            "power_ratio_after_propagation": result.total_power() / before,
            "paraxial_power_ratio_would_be": paraxial_power / before,
        }
    )


def test_the_rectangular_hole_near_field_agrees_with_a_direct_two_dimensional_summation():
    """MFT-FFT等价在**角谱**那一支上的兑现，外加矩孔的三个整数。

    金标那一侧是二维直接求和（无行列分解、无位反转、无旋转因子表）；
    被验侧是基2蝶形。冻的采样点跨四个象限。

    三个整数（亮采样32、亮行4、亮列8）锁的是"这真的是个**孔**"——
    亮行4 ≠ 亮列8 ≠ 网格32，仓里此前每一处``half_width_y_m = 1e9 * pitch``
    的写法在这三个数上当场露。
    """

    mask, screen = _propagated_hole()
    lit = [
        (row, column)
        for row in range(mask.row_count)
        for column in range(mask.column_count)
        if abs(mask.at(row, column)) > 0.5
    ]
    HOLE.check_all(
        {
            "field_components": [
                complex_to_components(screen.field.at(row, column))
                for row, column in HOLE.inputs["samples"]
            ],
            "aperture_sample_count": float(len(lit)),
            "aperture_lit_row_count": float(len({row for row, _ in lit})),
            "aperture_lit_column_count": float(len({column for _, column in lit})),
        }
    )


def test_the_transfer_function_form_is_self_consistent():
    """``z -> 0``退化、半群、能量守恒——三条恒等式，容差全部从误差模型来。

    第一条是**零容差**的：``z = 0``时传递函数逐位等于1，
    结果与``ifft2(fft2(U0))``是同一串浮点数。
    """

    setup = SELF_CONSISTENCY.inputs
    mask, screen = _propagated_hole()
    zero = propagate_angular_spectrum(
        mask,
        wavelength_m=setup["wavelength_m"],
        distance_m=0.0,
        pitch_x_m=setup["pitch_m"],
        pitch_y_m=setup["pitch_m"],
    )
    pure = ifft2(fft2(mask))
    half = propagate_angular_spectrum(
        mask,
        wavelength_m=setup["wavelength_m"],
        distance_m=setup["distance_m"] / 2.0,
        pitch_x_m=setup["pitch_m"],
        pitch_y_m=setup["pitch_m"],
    )
    twice = propagate_angular_spectrum(
        half.field,
        wavelength_m=setup["wavelength_m"],
        distance_m=setup["distance_m"] / 2.0,
        pitch_x_m=setup["pitch_m"],
        pitch_y_m=setup["pitch_m"],
    )
    before = incident_power(mask, pitch_x_m=setup["pitch_m"], pitch_y_m=setup["pitch_m"])
    SELF_CONSISTENCY.check_all(
        {
            "zero_distance_deviation_from_the_pure_round_trip": max(
                abs(got - want)
                for got, want in zip(zero.field.values(), pure.values(), strict=True)
            ),
            "zero_distance_deviation_from_the_incident_field": max(
                abs(got - want)
                for got, want in zip(zero.field.values(), mask.values(), strict=True)
            ),
            "semigroup_max_deviation": max(
                abs(one - two)
                for one, two in zip(screen.field.values(), twice.field.values(), strict=True)
            ),
            "power_ratio_after_propagation": screen.total_power() / before,
        }
    )


def test_the_zero_distance_result_is_bit_for_bit_the_pure_round_trip():
    """一条不靠清单的结构判据：``z = 0``不是"很接近"，是**同一串浮点数**。

    清单那条只判最大偏差是0.0；本条逐个比`float.hex()`。
    两者的分别在于：一个把``+0.0``写成``-0.0``的实现在偏差上照样是0。
    """

    setup = SELF_CONSISTENCY.inputs
    mask = _rectangular_hole()
    zero = propagate_angular_spectrum(
        mask,
        wavelength_m=setup["wavelength_m"],
        distance_m=0.0,
        pitch_x_m=setup["pitch_m"],
        pitch_y_m=setup["pitch_m"],
    )
    pure = ifft2(fft2(mask))
    for got, want in zip(zero.field.values(), pure.values(), strict=True):
        assert got.real.hex() == want.real.hex()
        assert got.imag.hex() == want.imag.hex()


def test_the_case_setup_really_sits_inside_the_declared_domain_and_the_engine_refuses_outside():
    """本案例的两个构型都在角谱的适用域``0 <= z <= z_c``之内，越界拒答。

    它守的是**案例自己的设定**：如果哪天有人把距离调过`z_c`，
    上面那些冻结值就不再是"角谱在适用域内的答案"，而是混叠区里的一堆数。
    """

    setup = SELF_CONSISTENCY.inputs
    limit = transfer_function_max_distance_m(
        count=setup["count"], pitch_m=setup["pitch_m"], wavelength_m=setup["wavelength_m"]
    )
    assert limit == setup["transfer_function_max_distance_m"]
    assert 0.0 < setup["distance_m"] < limit

    mask = _rectangular_hole()
    with pytest.raises(OpticsError, match="采样不足"):
        propagate_angular_spectrum(
            mask,
            wavelength_m=setup["wavelength_m"],
            distance_m=limit * 1.0001,
            pitch_x_m=setup["pitch_m"],
            pitch_y_m=setup["pitch_m"],
        )
