"""`case/mutual_inductance_coaxial`的conformance门（轴7规则3）。

**引擎第三个物理域第一次给出一个可对拍的数**：同轴圆环互感的Maxwell闭式
`M = μ0·√(r1·r2)·[(2/k − k)·K(k) − (2/k)·E(k)]`，对Neumann双回路线积分。

判据数全部来自清单；本文件不复述任何公式（轴7规则4）——尤其**不在这里
写一遍互感闭式**，那样测的就成了"抄两遍抄得一样"。
"""

from __future__ import annotations

from pathlib import Path

from physics_engine.electromagnetics import (
    CircularLoop,
    coaxial_modulus,
    coaxial_mutual_inductance_h,
    complete_elliptic_e,
    complete_elliptic_e_of_parameter,
    complete_elliptic_k,
    complete_elliptic_k_of_parameter,
    dipole_mutual_inductance_h,
    flux_linkage_wb,
    metres_from_millimetres,
    millimetres_from_metres,
    mutual_inductance_h,
    vacuum_permeability_relative_deviation_from_legacy,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/mutual_inductance_coaxial/oracle.json", root=ROOT)
CLOSED_FORM = MANIFEST.oracle("oracle:mutual_inductance_coaxial/maxwell_closed_form")
ELLIPTIC = MANIFEST.oracle("oracle:mutual_inductance_coaxial/elliptic_integrals")
RECIPROCITY = MANIFEST.oracle("oracle:mutual_inductance_coaxial/reciprocity")
FAR_FIELD = MANIFEST.oracle("oracle:mutual_inductance_coaxial/far_field_dipole")
TURNS = MANIFEST.oracle("oracle:mutual_inductance_coaxial/turns")
UNITS = MANIFEST.oracle("oracle:mutual_inductance_coaxial/unit_boundary")


def _inductance(config: list[float]) -> float:
    radius_a, radius_b, separation = config
    return coaxial_mutual_inductance_h(
        radius_a_m=radius_a, radius_b_m=radius_b, axial_separation_m=separation
    )


def test_closed_form_matches_an_independent_neumann_quadrature():
    """第一层：Maxwell闭式（AGM）对Neumann周期梯形积分——两条无共用代码的路。"""

    configurations = CLOSED_FORM.inputs["configurations_r1_r2_d_m"]
    CLOSED_FORM.check_all({
        "moduli": [
            coaxial_modulus(
                radius_a_m=config[0], radius_b_m=config[1], axial_separation_m=config[2]
            )
            for config in configurations
        ],
        "mutual_inductances_h": [_inductance(config) for config in configurations],
    })


def test_elliptic_integrals_match_carlson_and_the_two_conventions_stay_apart():
    """第二层：AGM对Carlson对称形式，外加模/参数两种约定的换算。

    **这一层不是第一层的附属品**：本仓的互感不经过`K − E`（见`elliptic.py`
    第三节的无相消改写），所以把两个公开的椭圆积分函数对调，
    第一层一个字都不会红——只有本层抓得住。
    """

    moduli = ELLIPTIC.inputs["moduli"]
    probe = ELLIPTIC.inputs["convention_probe_modulus"]
    ELLIPTIC.check_all({
        "complete_elliptic_k": [complete_elliptic_k(k) for k in moduli],
        "complete_elliptic_e": [complete_elliptic_e(k) for k in moduli],
        "parameter_convention_agrees":
            complete_elliptic_k_of_parameter(probe * probe) == complete_elliptic_k(probe)
            and complete_elliptic_e_of_parameter(probe * probe)
            == complete_elliptic_e(probe),
        "parameter_convention_is_not_the_same_as_modulus":
            complete_elliptic_k_of_parameter(probe) != complete_elliptic_k(probe)
            and complete_elliptic_e_of_parameter(probe) != complete_elliptic_e(probe),
    })


def test_mutual_inductance_is_reciprocal_bit_for_bit():
    """第三层：`M(1→2) == M(2→1)`——独立于闭式的自洽门，零容差。"""

    configurations = RECIPROCITY.inputs["configurations_r1_r2_d_m"]
    turns_a, turns_b = RECIPROCITY.inputs["loop_turns"]
    scalar_worst = 0.0
    loop_worst = 0.0
    for radius_a, radius_b, separation in configurations:
        forward = coaxial_mutual_inductance_h(
            radius_a_m=radius_a, radius_b_m=radius_b, axial_separation_m=separation
        )
        reverse = coaxial_mutual_inductance_h(
            radius_a_m=radius_b, radius_b_m=radius_a, axial_separation_m=separation
        )
        scalar_worst = max(scalar_worst, abs(forward - reverse))
        loop_a = CircularLoop(radius_m=radius_a, axial_position_m=0.0, turns=turns_a)
        loop_b = CircularLoop(
            radius_m=radius_b, axial_position_m=separation, turns=turns_b
        )
        loop_worst = max(
            loop_worst,
            abs(mutual_inductance_h(loop_a, loop_b) - mutual_inductance_h(loop_b, loop_a)),
        )
    RECIPROCITY.check_all({
        "scalar_reciprocity_max_abs_difference_h": scalar_worst,
        "loop_reciprocity_max_abs_difference_h": loop_worst,
    })


def test_far_field_degenerates_to_the_magnetic_dipole_at_second_order():
    """第四层：`d ≫ r`退化到偶极子，且偏差按`(r/d)²`收敛——阶是测出来的。"""

    import math

    radius_a, radius_b = FAR_FIELD.inputs["radii_m"]
    separations = FAR_FIELD.inputs["separations_m"]
    deviations = [
        coaxial_mutual_inductance_h(
            radius_a_m=radius_a, radius_b_m=radius_b, axial_separation_m=separation
        )
        / dipole_mutual_inductance_h(
            radius_a_m=radius_a, radius_b_m=radius_b, axial_separation_m=separation
        )
        - 1.0
        for separation in separations
    ]
    FAR_FIELD.check_all({
        "dipole_ratio_deviations": deviations,
        "dipole_convergence_orders": [
            math.log2(deviations[index] / deviations[index + 1])
            for index in range(len(deviations) - 1)
        ],
    })


def test_turns_multiply_the_single_turn_value_exactly():
    """第五层：N匝互感恰是单匝的`N1·N2`倍，且因子的结合次序不许漂。"""

    radius_a, radius_b, separation = TURNS.inputs["configuration_r1_r2_d_m"]
    pairs = TURNS.inputs["turns_pairs"]
    single_turn = coaxial_mutual_inductance_h(
        radius_a_m=radius_a, radius_b_m=radius_b, axial_separation_m=separation
    )
    inductances = []
    exact = True
    for turns_a, turns_b in pairs:
        loop_a = CircularLoop(radius_m=radius_a, axial_position_m=0.0, turns=turns_a)
        loop_b = CircularLoop(
            radius_m=radius_b, axial_position_m=separation, turns=turns_b
        )
        value = mutual_inductance_h(loop_a, loop_b)
        inductances.append(value)
        exact = exact and value == (turns_a * turns_b) * single_turn
    TURNS.check_all({
        "mutual_inductances_h": inductances,
        "turns_factor_is_bit_exact": exact,
    })


def test_the_millimetre_to_metre_boundary_is_explicit_and_reversible():
    """第六层：mm↔m的往返、方向、以及μ₀不是按定义精确的那条证据分级。"""

    radius_mm = UNITS.inputs["radius_mm"]
    separation_mm = UNITS.inputs["separation_mm"]
    from_mm = (
        CircularLoop.from_millimetres(radius_mm=radius_mm, axial_position_mm=0.0),
        CircularLoop.from_millimetres(
            radius_mm=radius_mm, axial_position_mm=separation_mm
        ),
    )
    in_metres = (
        CircularLoop(radius_m=radius_mm / 1.0e3, axial_position_m=0.0),
        CircularLoop(radius_m=radius_mm / 1.0e3, axial_position_m=separation_mm / 1.0e3),
    )
    turns_a, turns_b = UNITS.inputs["flux_linkage_turns"]
    UNITS.check_all({
        "millimetre_round_trip": [
            millimetres_from_metres(metres_from_millimetres(value))
            for value in UNITS.inputs["round_trip_millimetres"]
        ],
        "mutual_inductance_from_millimetres_h": mutual_inductance_h(*from_mm),
        "millimetre_and_metre_declarations_agree":
            mutual_inductance_h(*from_mm) == mutual_inductance_h(*in_metres),
        "mu0_relative_deviation_from_legacy":
            vacuum_permeability_relative_deviation_from_legacy(),
        "flux_linkages_wb": [
            flux_linkage_wb(
                source=CircularLoop(
                    radius_m=0.05,
                    axial_position_m=0.0,
                    turns=turns_a,
                    current_a=current,
                ),
                target=CircularLoop(
                    radius_m=0.05, axial_position_m=0.02, turns=turns_b
                ),
            )
            for current in UNITS.inputs["flux_linkage_currents_a"]
        ],
    })
