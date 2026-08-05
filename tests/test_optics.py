"""光学域第一块的协议门（案例判据在`tests/cases/`，本文件不重复它们）。

分工照decisions/0024的先例：**案例验物理，本文件验协议**——
公开面、失败关闭、单位边界的对称性、以及与材料记录那道域间接口。
"""

from __future__ import annotations

import math

import pytest

import physics_engine
from physics_engine import optics
from physics_engine.materials import EvidenceRef, MaterialProperty, MaterialRecord
from physics_engine.optics import (
    AIRY_FIRST_ZERO_TRUNCATION,
    AIRY_FIRST_ZERO_X,
    J1_ABSOLUTE_ACCURACY,
    J1_TESTED_ARGUMENT_MAX,
    NORTON_BEER_COEFFICIENTS,
    NORTON_BEER_STRENGTHS,
    NORTON_BEER_UNIT_SUM_TOLERANCE,
    OpticsError,
    airy_amplitude,
    airy_argument,
    airy_first_minimum_half_angle_rad,
    angular_wavenumber_rad_per_m,
    bessel_j1,
    normalised_sinc,
    norton_beer_window,
    optics_evidence_grade,
    optics_parameters,
    require_optics_parameter,
    spectroscopic_wavenumber_per_m,
    unapodised_line_shape,
)

SOURCE_SHA = "a" * 64


def _evidence(grade: str = "measured", name: str = "field") -> EvidenceRef:
    return EvidenceRef(
        grade=grade,
        evidence_id=f"evidence/optics-fixture-{name}",
        method="固定装置：本文件自造的记录，只为验域间接口，不代表任何真实材料。",
        source_sha256=None if grade in {"estimated", "unset"} else SOURCE_SHA,
    )


def _record(length_unit: str = "m") -> MaterialRecord:
    thickness = "thickness_m" if length_unit == "m" else "thickness_mm"
    return MaterialRecord(
        material_id="material/fixture__flat__nominal__estimated__v001",
        applicable_domains=("mechanics", "optics"),
        length_unit=length_unit,
        dimensionless=frozenset({"specular_reflectance", "refractive_index"}),
        properties=(
            MaterialProperty(
                name=thickness, value=1.0e-3 if length_unit == "m" else 1.0,
                domains=("mechanics", "optics"), evidence=_evidence(name="thickness"),
            ),
            MaterialProperty(
                name="refractive_index", value=1.4585,
                domains=("optics",), evidence=_evidence("derived", "index"),
            ),
            MaterialProperty(
                name="specular_reflectance", value=None,
                domains=("optics",),
                evidence=EvidenceRef(
                    grade="unset",
                    evidence_id="evidence/optics-fixture-reflectance-unmeasured",
                    method="未测量：本装置故意留一个unset字段，用来验拒跑那条路。",
                    source_sha256=None,
                ),
            ),
        ),
    )


# --- 公开面 ---------------------------------------------------------------


def test_every_exported_name_exists():
    for name in optics.__all__:
        assert hasattr(optics, name), f"__all__里的{name!r}在子包上不存在"
    assert optics.__all__ == sorted(optics.__all__), "__all__未排序"


def test_optics_names_stay_out_of_the_package_facade():
    """本仓硬纪律：新公开名只进子包自己的`__all__`，不进`physics_engine.__all__`。

    这条与域隔离门的门③是同一件事的两面——那边禁的是import边，
    这边禁的是名字。两边都验，因为绕过一边的手法绕不过另一边。
    """

    leaked = sorted(set(optics.__all__) & set(physics_engine.__all__))
    assert not leaked, f"光学公开名漏进了包门面：{leaked}"
    assert not hasattr(physics_engine, "airy_amplitude")


# --- 贝塞尔与艾里 ---------------------------------------------------------


def test_bessel_j1_is_odd_and_vanishes_at_the_origin():
    assert bessel_j1(0.0) == 0.0
    for x in (0.25, 3.0, 11.5, 12.5, 30.0):
        assert bessel_j1(-x) == -bessel_j1(x), f"J1在x={x!r}处不是奇函数"


def test_airy_amplitude_is_even_and_unity_on_axis():
    assert airy_amplitude(0.0) == 1.0
    for x in (0.5, 3.0, 13.0):
        assert airy_amplitude(-x) == airy_amplitude(x)


def test_the_declared_truncation_really_bounds_the_gap_to_the_true_first_zero():
    """`AIRY_FIRST_ZERO_TRUNCATION`不是装饰：案例页的容差推导逐字用它。

    两个方向都断言：它必须是上界（否则容差推导站不住），
    也不许比实差大一倍以上（写松了的上界会让容差跟着虚胖）。
    """

    true_first_zero = 3.8317059702075123156  # A&S表9.5的全位数
    gap = abs(true_first_zero - AIRY_FIRST_ZERO_X)
    assert gap <= AIRY_FIRST_ZERO_TRUNCATION, f"申报的截断量不是上界：实差{gap!r}"
    assert gap > 0.5 * AIRY_FIRST_ZERO_TRUNCATION, (
        f"申报的截断量{AIRY_FIRST_ZERO_TRUNCATION!r}比实差{gap!r}大一倍以上——写松了"
    )


def _bessel_integral_reference(x: float, nodes: int = 1024) -> float:
    """`J1`的独立参考：贝塞尔积分的周期梯形求值。

    与`cases/scalar_diffraction_airy/generate_oracle.py`同一条路，
    但**本文件不import那个生成器**——它是案例的资产，测试不该跨案例取用。
    """

    if x == 0.0:
        return 0.0
    return math.fsum(
        math.cos(t - x * math.sin(t))
        for t in (math.pi * (index + 0.5) / nodes for index in range(nodes))
    ) / nodes


@pytest.mark.batch
def test_the_declared_j1_accuracy_holds_across_the_whole_tested_range():
    """**申报的精度要被验，不是被声称**：`[0, 60]`上逐点对独立参考。

    案例只冻结16个采样点，那16点选得再好也覆盖不了整段；本条把整段扫一遍。
    走`batch`档是因为它要2e6次三角函数求值（交互级<1s装不下），
    不是因为它可选——`accept.py full`跑它。
    """

    worst = 0.0
    worst_at = 0.0
    steps = int(J1_TESTED_ARGUMENT_MAX / 0.031) + 1
    for index in range(steps):
        x = index * 0.031
        error = abs(bessel_j1(x) - _bessel_integral_reference(x))
        if error > worst:
            worst, worst_at = error, x
    assert worst <= J1_ABSOLUTE_ACCURACY, (
        f"申报绝对精度{J1_ABSOLUTE_ACCURACY!r}不成立：x={worst_at!r}处偏差{worst!r}"
    )
    assert worst > 0.1 * J1_ABSOLUTE_ACCURACY, (
        f"实测最坏偏差{worst!r}比申报值小一个数量级以上——申报写松了，"
        f"按实测（{worst!r}，x={worst_at!r}）收紧"
    )


def test_the_two_wavenumber_conventions_differ_by_exactly_two_pi():
    """谱学波数与角波数差一个2pi——这道边界写成两个函数就是为了它。"""

    wavelength = 632.8e-9
    ratio = angular_wavenumber_rad_per_m(wavelength) / spectroscopic_wavenumber_per_m(
        wavelength
    )
    assert ratio == pytest.approx(2.0 * math.pi, rel=1e-15)


def test_a_diameter_passed_as_a_radius_moves_the_first_zero_by_a_factor_of_two():
    """半径/直径这道边界的活标本：传错了图样宽一倍，而且不报任何错。

    所以`airy_argument`的签名写`aperture_radius_m`、
    `airy_first_minimum_half_angle_rad`的签名写`aperture_diameter_m`——
    让调用方不必记住吃的是哪一个。
    """

    wavelength, diameter = 632.8e-9, 1.0e-3
    angle = airy_first_minimum_half_angle_rad(
        wavelength_m=wavelength, aperture_diameter_m=diameter
    )
    right = airy_argument(
        half_angle_rad=angle, aperture_radius_m=diameter / 2.0, wavelength_m=wavelength
    )
    wrong = airy_argument(
        half_angle_rad=angle, aperture_radius_m=diameter, wavelength_m=wavelength
    )
    assert wrong / right == pytest.approx(2.0, rel=1e-12)


def test_a_wide_wavelength_to_diameter_ratio_fails_closed():
    """没有首个暗环时拒答，不返回nan也不夹到pi/2。"""

    with pytest.raises(OpticsError, match="没有首个暗环"):
        airy_first_minimum_half_angle_rad(
            wavelength_m=1.0e-3, aperture_diameter_m=1.0e-3
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: bessel_j1(float("nan")),
        lambda: airy_amplitude(float("inf")),
        lambda: airy_argument(half_angle_rad=0.1, aperture_radius_m=0.0, wavelength_m=1e-6),
        lambda: airy_argument(half_angle_rad=0.1, aperture_radius_m=1e-3, wavelength_m=-1.0),
        lambda: angular_wavenumber_rad_per_m(0.0),
        lambda: normalised_sinc(float("inf")),
        lambda: unapodised_line_shape(1.0, max_opd_m=0.0),
        lambda: norton_beer_window(0.0, strength="brutal", max_opd_m=0.05),
        lambda: norton_beer_window(0.0, strength="weak", max_opd_m=-1.0),
    ],
)
def test_domain_violations_fail_closed(call):
    with pytest.raises(OpticsError):
        call()


# --- FTS ------------------------------------------------------------------


def test_normalised_sinc_is_not_the_unnormalised_one():
    """归一化sinc的首零在1；未归一化的`sin(z)/z`首零在pi。差一个pi的那道边界。"""

    assert normalised_sinc(0.0) == 1.0
    assert abs(normalised_sinc(1.0)) < 1e-16
    assert math.sin(1.0) / 1.0 > 0.8, "若这条不成立，两种约定就分不开了"
    assert normalised_sinc(-0.37) == normalised_sinc(0.37)


def test_every_norton_beer_set_sums_to_one():
    """`sum(Ci) = 1`是一条门，且它守的是**整张表**不只是今天这三组。"""

    assert set(NORTON_BEER_COEFFICIENTS) == set(NORTON_BEER_STRENGTHS)
    for name, coefficients in NORTON_BEER_COEFFICIENTS.items():
        residual = abs(math.fsum(coefficients) - 1.0)
        assert residual <= NORTON_BEER_UNIT_SUM_TOLERANCE, (
            f"{name}组系数和偏离1达{residual!r}——切趾窗在零光程差处必须取1，"
            "否则整条谱线被整体缩放"
        )
        assert residual == 0.0, (
            f"{name}组的实测残差不再是0（{residual!r}）——"
            "三组已发表系数在float64上相加正好落回1.0，变了说明系数被动过"
        )


def test_the_unit_sum_gate_must_be_red_on_a_perturbed_set():
    """必须红：把一位系数改到超出容差，`sum(Ci)=1`那条门要炸。"""

    broken = (0.384093, -0.087577, 0.703484 + 1.0e-5, 0.0)
    residual = abs(math.fsum(broken) - 1.0)
    assert residual > NORTON_BEER_UNIT_SUM_TOLERANCE, (
        f"扰动1e-5后残差只有{residual!r}，门抓不住——容差写松了"
    )


def test_the_window_is_zero_outside_the_scan():
    length = 0.05
    for name in NORTON_BEER_STRENGTHS:
        assert norton_beer_window(length * 1.0000001, strength=name, max_opd_m=length) == 0.0
        assert norton_beer_window(-length * 2.0, strength=name, max_opd_m=length) == 0.0


# --- 域间接口：材料记录 ----------------------------------------------------


def test_optics_reads_the_material_record_through_the_domain_accessor():
    """光学域按`properties_for_domain("optics")`取字段，不另造材料通道。"""

    values = optics_parameters(_record())
    assert set(values) == {"thickness_m", "refractive_index", "specular_reflectance"}
    assert values["refractive_index"] == 1.4585
    assert values["specular_reflectance"] is None
    assert require_optics_parameter(_record(), "refractive_index") == 1.4585


def test_a_millimetre_record_fails_closed_until_it_is_converted():
    """毫米制记录直接读即拒——静默的1000倍是spec/14第五节钉的那个坑。"""

    millimetre = _record("mm")
    with pytest.raises(Exception, match="converted_to"):
        optics_parameters(millimetre)
    values = optics_parameters(millimetre.converted_to("m"))
    assert values["thickness_m"] == pytest.approx(1.0e-3, rel=1e-15)


def test_an_unset_parameter_refuses_to_be_used():
    """未测量就没有值，不是零——拒跑，并把证据说明带进失败信息。"""

    with pytest.raises(OpticsError, match="未测量"):
        require_optics_parameter(_record(), "specular_reflectance")


def test_a_missing_parameter_names_what_the_record_does_carry():
    with pytest.raises(OpticsError, match="没有字段"):
        require_optics_parameter(_record(), "extinction_coefficient")


def test_the_evidence_grade_is_reported_per_domain():
    """记录整体的可信度没有意义，只有光学域那批字段的可信度有意义。"""

    assert optics_evidence_grade(_record()) == "unset"
