"""`case/double_slit_propagated`的conformance门（轴7规则3）。

**引擎第一次把一个孔径的衍射图样真的传出去算**，而不是代入一条闭式
（能力位S4.6的定义原文："一般孔径的衍射……由传播算出而非闭式代入"）。
同时它把research/05第2.3节光学族第13条的三件套一次跑齐：
MFT-FFT等价、通量守恒、往返可逆。

判据数全部来自清单；本文件不复述任何公式（轴7规则4）。
"""

from __future__ import annotations

from pathlib import Path

from physics_engine.optics.field import ComplexField2D, complex_to_components, fft2, ifft2
from physics_engine.optics.propagation import (
    incident_power,
    paraxial_sine_of_angle,
    propagate_fraunhofer,
    rectangular_aperture,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/double_slit_propagated/oracle.json", root=ROOT)
TABLE = MANIFEST.oracle("oracle:double_slit/relative_amplitude_table")
FRINGES = MANIFEST.oracle("oracle:double_slit/fringes_and_missing_order")
SELF_CONSISTENCY = MANIFEST.oracle("oracle:double_slit/flux_and_reversibility")

SETUP = TABLE.inputs


def _double_slit() -> ComplexField2D:
    """两条缝的振幅掩模——**用引擎的孔径构件**，与生成器的几何算法各走各的。"""

    half_width = (
        (SETUP["slit_samples"] - 1) / 2.0 + SETUP["edge_offset_in_pitches"]
    ) * SETUP["pitch_m"]
    masks = []
    for sign in (-1.0, 1.0):
        centre = (0.5 + sign * SETUP["separation_samples"] / 2.0) * SETUP["pitch_m"]
        masks.append(
            rectangular_aperture(
                row_count=SETUP["rows"],
                column_count=SETUP["columns"],
                pitch_x_m=SETUP["pitch_m"],
                pitch_y_m=SETUP["pitch_m"],
                half_width_x_m=half_width,
                half_width_y_m=1.0e9 * SETUP["pitch_m"],
                centre_x_m=centre,
            )
        )
    left, right = masks
    return ComplexField2D(
        tuple(
            tuple(a + b for a, b in zip(row_a, row_b, strict=True))
            for row_a, row_b in zip(left.rows, right.rows, strict=True)
        )
    )


def _propagated():
    mask = _double_slit()
    aperture_half = (
        SETUP["separation_samples"] / 2.0 + SETUP["slit_samples"] / 2.0
    ) * SETUP["pitch_m"]
    return mask, propagate_fraunhofer(
        mask,
        wavelength_m=SETUP["wavelength_m"],
        distance_m=SETUP["screen_distance_m"],
        pitch_x_m=SETUP["pitch_m"],
        pitch_y_m=SETUP["pitch_m"],
        aperture_half_width_m=aperture_half,
    )


def test_the_fast_transform_agrees_with_a_direct_summation_over_the_aperture():
    """MFT-FFT等价：基2蝶形 对 定义式直接求和，复振幅比逐点对拍。

    冻结的是**复数**，落盘形制是决策0086裁的``[实部, 虚部]``二元组——
    实部虚部各自吃同一条容差，报错时路径带`[0]`/`[1]`。
    """

    mask, screen = _propagated()
    row = screen.field.rows[0]
    on_axis = row[0]
    TABLE.check_all(
        {
            "relative_amplitude_components": [
                complex_to_components(row[index] / on_axis) for index in TABLE.inputs["bins"]
            ],
            "aperture_sample_count": float(
                sum(1 for value in mask.rows[0] if abs(value) > 0.5)
            ),
        }
    )


def test_the_fringes_and_the_missing_order():
    """条纹极大落在``m lambda / d``，第8级被单缝包络的首零压掉。"""

    _, screen = _propagated()
    row = screen.field.rows[0]
    intensity = screen.intensity_rows()[0]
    coordinates = screen.coordinates_x_m()
    distance = SETUP["screen_distance_m"]
    step = FRINGES.inputs["fringe_bin_step"]
    missing_bin = FRINGES.inputs["missing_order_bin"]
    on_axis = row[0]
    peak = intensity[0]
    FRINGES.check_all(
        {
            "fringe_sines": [
                paraxial_sine_of_angle(coordinates[order * step], distance)
                for order in FRINGES.inputs["orders"]
            ],
            "normalised_intensity": [
                abs(row[order * step] / on_axis) ** 2
                for order in FRINGES.inputs["orders"]
            ],
            "missing_order_intensity": intensity[missing_bin] / peak,
            "envelope_first_zero_sine": paraxial_sine_of_angle(
                coordinates[missing_bin], distance
            ),
        }
    )


def test_the_transform_layer_conserves_flux_and_is_reversible():
    """第13条三件套的另两件。通量守恒是前因子唯一的捕手。"""

    mask, screen = _propagated()
    before = incident_power(
        mask, pitch_x_m=SETUP["pitch_m"], pitch_y_m=SETUP["pitch_m"]
    )
    recovered = ifft2(fft2(mask))
    SELF_CONSISTENCY.check_all(
        {
            "power_ratio_after_propagation": screen.total_power() / before,
            "round_trip_max_deviation": max(
                abs(got - want)
                for got, want in zip(recovered.values(), mask.values(), strict=True)
            ),
        }
    )


def test_the_fringe_maxima_really_are_local_maxima():
    """一条不靠清单的结构判据：前6级条纹必须是局部极大。

    它抓的是清单抓不到的那一类错——一个把整幅图样平移了半个条纹的实现，
    在"某些bin上的强度值"这类判据上可以照样绿（因为余弦是偶的），
    但那些bin就不再是极大了。
    """

    _, screen = _propagated()
    intensity = screen.intensity_rows()[0]
    step = FRINGES.inputs["fringe_bin_step"]
    for order in FRINGES.inputs["orders"][1:]:
        index = order * step
        assert intensity[index] > intensity[index - 1], f"第{order}级左侧不是上坡"
        assert intensity[index] > intensity[index + 1], f"第{order}级右侧不是下坡"
