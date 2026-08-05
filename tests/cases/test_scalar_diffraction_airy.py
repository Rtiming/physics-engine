"""`case/scalar_diffraction_airy`的conformance门（轴7规则3）。

**引擎第一次算一条光学闭式解**：圆孔远场`E(x) = 2 J1(x)/x`，首零
`x = 3.8317059702`（research/05第2.3节光学族的第一条判据）。

判据数全部来自清单；本文件不复述任何公式（轴7规则4）。
"""

from __future__ import annotations

from pathlib import Path

from physics_engine.optics import (
    AIRY_FIRST_ZERO_X,
    airy_amplitude,
    airy_argument,
    airy_first_minimum_half_angle_rad,
    airy_intensity,
    bessel_j1,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/scalar_diffraction_airy/oracle.json", root=ROOT)
TABLE = MANIFEST.oracle("oracle:airy/bessel_j1_reference_table")
UNITS = MANIFEST.oracle("oracle:airy/first_zero_and_units")


def test_bessel_j1_matches_an_independent_integral_evaluation():
    """级数+渐近展开 对 贝塞尔积分的周期梯形求值——两条无共用代码的路。"""

    arguments = TABLE.inputs["arguments"]
    TABLE.check_all({
        "j1_values": [bessel_j1(x) for x in arguments],
        "airy_amplitude_values": [airy_amplitude(x) for x in arguments],
    })


def test_first_zero_and_the_unit_round_trip():
    """首零、轴上峰值、以及角度↔空间频率↔艾里自变量的往返。"""

    low, high = UNITS.inputs["bisection_bracket"]
    value_low = airy_amplitude(low)
    for _ in range(100):
        middle = 0.5 * (low + high)
        if value_low * airy_amplitude(middle) <= 0.0:
            high = middle
        else:
            low, value_low = middle, airy_amplitude(middle)
    first_zero = 0.5 * (low + high)

    half_angle = airy_first_minimum_half_angle_rad(
        wavelength_m=UNITS.inputs["wavelength_m"],
        aperture_diameter_m=UNITS.inputs["aperture_diameter_m"],
    )
    UNITS.check_all({
        "first_zero_x": first_zero,
        "constant_matches_literature_quote":
            AIRY_FIRST_ZERO_X == UNITS.inputs["literature_first_zero"],
        "amplitude_at_axis": airy_amplitude(0.0),
        "amplitude_at_literature_zero": airy_amplitude(AIRY_FIRST_ZERO_X),
        "intensity_at_literature_zero": airy_intensity(AIRY_FIRST_ZERO_X),
        "first_minimum_half_angle_rad": half_angle,
        "argument_at_first_minimum": airy_argument(
            half_angle_rad=half_angle,
            aperture_radius_m=UNITS.inputs["aperture_diameter_m"] / 2.0,
            wavelength_m=UNITS.inputs["wavelength_m"],
        ),
    })


def test_the_central_lobe_is_the_maximum_and_the_pattern_is_even():
    """主瓣是全局极大且图样偶对称——两条不靠清单的结构判据。

    它们抓的是清单抓不到的那一类错：一个把`J1`写成`J0`的实现在首零判据上会红，
    但一个把自变量取了绝对值**之后又平方**的实现在首零判据上照样绿。
    """

    assert airy_intensity(0.0) == 1.0
    step = AIRY_FIRST_ZERO_X / 64.0
    for index in range(1, 65):
        x = index * step
        assert airy_intensity(x) < airy_intensity(x - step), (
            f"主瓣内光强在x={x!r}处不再单调下降——首零之前不该有极值"
        )
        assert airy_amplitude(-x) == airy_amplitude(x), f"图样在x={x!r}处不对称"
