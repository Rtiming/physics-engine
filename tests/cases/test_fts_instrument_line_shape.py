"""`case/fts_instrument_line_shape`的conformance门（轴7规则3）。

无切趾ILS是sinc、半高全宽`1.20671/(2 OPD_max)`、Norton-Beer三组各自
`sum(Ci) = 1`——research/05第2.3节光学族的另外两条判据。

判据数全部来自清单；本文件不复述任何公式（轴7规则4）。
"""

from __future__ import annotations

from pathlib import Path

from physics_engine.optics import (
    UNAPODISED_FWHM_IN_SINC_UNITS,
    norton_beer_coefficients,
    norton_beer_throughput,
    norton_beer_window,
    unapodised_first_zero_per_m,
    unapodised_fwhm_per_m,
    unapodised_line_shape,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/fts_instrument_line_shape/oracle.json", root=ROOT)
BOXCAR = MANIFEST.oracle("oracle:fts/unapodised_line_shape")
APODISATION = MANIFEST.oracle("oracle:fts/norton_beer_apodisation")


def test_unapodised_line_shape_is_a_sinc_of_the_declared_width():
    """半高全宽的常量对独立求根，半高点上ILS必须恰为0.5。"""

    length = BOXCAR.inputs["max_opd_m"]
    fwhm = unapodised_fwhm_per_m(length)
    gap = abs(UNAPODISED_FWHM_IN_SINC_UNITS - BOXCAR.inputs["textbook_fwhm_in_sinc_units"])
    BOXCAR.check_all({
        "fwhm_in_sinc_units": UNAPODISED_FWHM_IN_SINC_UNITS,
        "fwhm_matches_textbook_quote": gap <= BOXCAR.inputs["textbook_half_step"],
        "first_zero_per_m": unapodised_first_zero_per_m(length),
        "fwhm_per_m": fwhm,
        "line_shape_at_peak": unapodised_line_shape(0.0, max_opd_m=length),
        "line_shape_at_first_zero": unapodised_line_shape(
            unapodised_first_zero_per_m(length), max_opd_m=length
        ),
        "line_shape_at_plus_half_fwhm": unapodised_line_shape(fwhm / 2.0, max_opd_m=length),
        "line_shape_at_minus_half_fwhm": unapodised_line_shape(-fwhm / 2.0, max_opd_m=length),
    })


def test_norton_beer_windows_sum_to_one_and_cost_throughput_in_order():
    """三组系数各自求和为一；通量代价弱>中>强（先断非退化再断严格不等号）。"""

    strengths = APODISATION.inputs["strengths"]
    length = APODISATION.inputs["max_opd_m"]
    beyond = APODISATION.inputs["beyond_scan_factor"] * length
    throughput = [norton_beer_throughput(name) for name in strengths]

    # 排序断言的假通过口子是三者全零（`0 > 0 > 0`按浮点比较为假，
    # 但只要有人把它写成`>=`就全过）。堵法：先断每个都落在(0,1)开区间。
    for name, value in zip(strengths, throughput, strict=True):
        assert 0.0 < value < 1.0, (
            f"{name}的通量代价{value!r}不在(0,1)内——排序断言在退化输入上会假通过"
        )
    assert throughput[0] > throughput[1] > throughput[2], (
        f"通量代价没有按弱>中>强排序：{list(zip(strengths, throughput, strict=True))}"
    )

    minima = [_scan_minimum(name, length) for name in strengths]
    APODISATION.check_all({
        "coefficient_sums": [sum(norton_beer_coefficients(name)) for name in strengths],
        "window_at_zero_opd": [
            norton_beer_window(0.0, strength=name, max_opd_m=length) for name in strengths
        ],
        "window_at_scan_end": [
            norton_beer_window(length, strength=name, max_opd_m=length) for name in strengths
        ],
        "window_beyond_scan": [
            norton_beer_window(beyond, strength=name, max_opd_m=length) for name in strengths
        ],
        "throughput": throughput,
        "throughput_strictly_decreasing": throughput[0] > throughput[1] > throughput[2],
        "window_minimum_reduced_opd": [position for position, _ in minima],
        "window_minimum_value": [value for _, value in minima],
    })


def _scan_minimum(strength: str, length: float) -> tuple[float, float]:
    """在等距网格上取窗的最小值与它的位置（归一化到`Delta/L`）。"""

    points = APODISATION.inputs["minimum_scan_points"]
    value, index = min(
        (
            norton_beer_window(
                length * step / (points - 1), strength=strength, max_opd_m=length
            ),
            step,
        )
        for step in range(points)
    )
    return index / (points - 1), value


def test_the_apodisation_windows_are_even_and_bounded_by_the_zero_opd_peak():
    """窗偶对称、全程落在`(0, A(0)]`内——清单抓不到的那一类错。

    一个把`(1 - x^2)`写成`(1 - x)`的实现在`A(0)`与`A(L)`两点上**全对**
    （0处都是sum(Ci)、L处都是C0），只有中间那段会露出来：它不再偶对称。

    **这里不断言单调**：weak与medium的窗在扫描端点附近真的会回升
    （C1为负），单调断言在第一版里当场红——那不是bug是这两组已发表系数的性质。
    把它钉成数的是清单里的`window_minimum_*`两条。
    """

    length = APODISATION.inputs["max_opd_m"]
    for name in APODISATION.inputs["strengths"]:
        peak = norton_beer_window(0.0, strength=name, max_opd_m=length)
        for index in range(1, 65):
            position = length * index / 64.0
            value = norton_beer_window(position, strength=name, max_opd_m=length)
            assert 0.0 < value <= peak, (
                f"{name}的窗在{position!r} m处取{value!r}，越出了(0, {peak!r}]"
            )
            assert value == norton_beer_window(-position, strength=name, max_opd_m=length), (
                f"{name}的窗在{position!r} m处不对称"
            )
