"""`case/mutual_inductance_general`的conformance门（轴7规则3）。

一般位形（倾斜／偏心／非共轴）两条圆形回路的互感：Neumann双回路线积分的
中点切元求积，对**三条独立参考路径**——共轴那一支的约化Neumann单重积分、
一般位形的Biot-Savart圆盘磁通、远场的偶极-偶极式。

判据数全部来自清单；本文件不复述任何公式（轴7规则4）——
**尤其不在这里写一遍Neumann双重和**，那样测的就成了"抄两遍抄得一样"。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.electromagnetics import CircularLoop, ElectromagneticsError
from physics_engine.electromagnetics.neumann import (
    PlacedCircularLoop,
    dipole_mutual_inductance_general_h,
    filament_resolution_ratio,
    neumann_mutual_inductance_h,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/mutual_inductance_general/oracle.json", root=ROOT)
COAXIAL = MANIFEST.oracle("oracle:mutual_inductance_general/coaxial_degeneration")
CONVERGENCE = MANIFEST.oracle("oracle:mutual_inductance_general/geometric_convergence")
GENERAL = MANIFEST.oracle("oracle:mutual_inductance_general/general_position")
RECIPROCITY = MANIFEST.oracle("oracle:mutual_inductance_general/reciprocity")
FAR_FIELD = MANIFEST.oracle("oracle:mutual_inductance_general/far_field_dipole")
SINGULARITY = MANIFEST.oracle("oracle:mutual_inductance_general/singularity_and_resolution")
UNITS = MANIFEST.oracle("oracle:mutual_inductance_general/unit_boundary")


def _coaxial_pair(
    radius_a: float, radius_b: float, separation: float
) -> tuple[PlacedCircularLoop, PlacedCircularLoop]:
    """两条共轴回路，**经`from_coaxial`搬进一般位形**——这条接口本身也在被验。"""

    return (
        PlacedCircularLoop.from_coaxial(
            CircularLoop(radius_m=radius_a, axial_position_m=0.0)
        ),
        PlacedCircularLoop.from_coaxial(
            CircularLoop(radius_m=radius_b, axial_position_m=separation)
        ),
    )


def test_the_general_quadrature_degenerates_to_the_coaxial_closed_form():
    """第一层：一般位形的双重求积在共轴构型上对上约化Neumann单重积分。

    **这是本案例最强的一条金标**：共轴是一般位形的特例，而共轴那一支
    有解析约化（并另有Maxwell闭式在生成器里交叉验证过）。
    """

    segments = COAXIAL.inputs["segments"]
    values = []
    for radius_a, radius_b, separation in COAXIAL.inputs["configurations_r1_r2_d_m"]:
        loop_a, loop_b = _coaxial_pair(radius_a, radius_b, separation)
        values.append(
            neumann_mutual_inductance_h(
                loop_a, loop_b, segments_a=segments, segments_b=segments
            )
        )
    COAXIAL.check_all({"mutual_inductances_h": values})


def test_the_quadrature_converges_geometrically_not_at_a_fixed_order():
    """第二层：加密时误差按**几何**下降——阶不是2，也不是任何一个固定的数。

    判两件事：① 逐档误差落在声明的包络内；② 每加密8段的``log2``误差比
    既超过下限、又不随加密而衰减。**第二件是形态判据**——
    代数阶方法的比值会一路掉下去（折线弦离散实测0.53→0.26）。
    """

    radius_a, radius_b, separation = CONVERGENCE.inputs["configuration_r1_r2_d_m"]
    exact = CONVERGENCE.inputs["exact_mutual_inductance_h"]
    counts = CONVERGENCE.inputs["segment_counts"]
    envelope = CONVERGENCE.inputs["relative_error_envelope"]
    floor = CONVERGENCE.inputs["log2_ratio_floor_per_eight_segments"]
    band_max = CONVERGENCE.inputs["log2_ratio_band_max"]

    loop_a, loop_b = _coaxial_pair(radius_a, radius_b, separation)
    errors = [
        abs(
            neumann_mutual_inductance_h(
                loop_a, loop_b, segments_a=count, segments_b=count
            )
            - exact
        )
        / abs(exact)
        for count in counts
    ]
    ratios = [
        math.log2(errors[index] / errors[index + 1])
        * 8.0
        / (counts[index + 1] - counts[index])
        for index in range(len(counts) - 1)
    ]
    CONVERGENCE.check_all({
        "relative_errors_fall_under_the_envelope": all(
            error <= bound for error, bound in zip(errors, envelope, strict=True)
        ),
        "log2_ratios_exceed_the_floor": all(ratio >= floor for ratio in ratios),
        "log2_ratios_stay_in_a_band": max(ratios) / min(ratios) <= band_max,
    })


def test_general_placements_match_an_independent_flux_integral():
    """第三层：倾斜／偏心／非共轴对**Biot-Savart场的圆盘磁通**。

    两个不同的物理表述：Neumann积的是两条线，参考路径积的是一条线加一张面。
    附带一条**零磁通**构型：正交摆放时M按对称性精确为零，四组分段数都要给0。
    """

    segments = GENERAL.inputs["segments"]
    values = []
    for entry in GENERAL.inputs["configurations"]:
        loop_a = PlacedCircularLoop(
            radius_m=entry["radius_a_m"],
            centre_m=tuple(entry["centre_a_m"]),
            normal=tuple(entry["normal_a"]),
        )
        loop_b = PlacedCircularLoop(
            radius_m=entry["radius_b_m"],
            centre_m=tuple(entry["centre_b_m"]),
            normal=tuple(entry["normal_b"]),
        )
        values.append(
            neumann_mutual_inductance_h(
                loop_a, loop_b, segments_a=segments, segments_b=segments
            )
        )

    perpendicular = GENERAL.inputs["perpendicular_configuration"]
    null_a = PlacedCircularLoop(
        radius_m=perpendicular["radius_a_m"],
        centre_m=tuple(perpendicular["centre_a_m"]),
        normal=tuple(perpendicular["normal_a"]),
    )
    null_b = PlacedCircularLoop(
        radius_m=perpendicular["radius_b_m"],
        centre_m=tuple(perpendicular["centre_b_m"]),
        normal=tuple(perpendicular["normal_b"]),
    )
    nulls = [
        neumann_mutual_inductance_h(
            null_a, null_b, segments_a=pair[0], segments_b=pair[1]
        )
        for pair in GENERAL.inputs["perpendicular_segment_pairs"]
    ]
    GENERAL.check_all({
        "mutual_inductances_h": values,
        "perpendicular_mutual_inductances_h": nulls,
    })


def test_the_general_mutual_inductance_is_reciprocal_bit_for_bit():
    """第四层：`M(a,b) == M(b,a)`**逐位**，且分段数跟着交换——独立于任何参考路径。"""

    worst = 0.0
    for entry in RECIPROCITY.inputs["configurations"]:
        loop_a = PlacedCircularLoop(
            radius_m=entry["radius_a_m"],
            centre_m=tuple(entry["centre_a_m"]),
            normal=tuple(entry["normal_a"]),
            turns=entry["turns_a"],
        )
        loop_b = PlacedCircularLoop(
            radius_m=entry["radius_b_m"],
            centre_m=tuple(entry["centre_b_m"]),
            normal=tuple(entry["normal_b"]),
            turns=entry["turns_b"],
        )
        forward = neumann_mutual_inductance_h(
            loop_a,
            loop_b,
            segments_a=entry["segments_a"],
            segments_b=entry["segments_b"],
        )
        reverse = neumann_mutual_inductance_h(
            loop_b,
            loop_a,
            segments_a=entry["segments_b"],
            segments_b=entry["segments_a"],
        )
        worst = max(worst, abs(forward - reverse))
    RECIPROCITY.check_all({"reciprocity_max_abs_difference_h": worst})


def test_the_far_field_degenerates_to_the_dipole_limit_in_three_orientations():
    """第五层：`d ≫ R`退化到偶极-偶极式，偏差按`(R/d)²`收敛。

    三族取向：共轴（方括号=+2）、共面并排（方括号=−1，**M为负**）、
    倾斜60度。**阶是测出来的**，十二个数逐个钉住，不断言成2。
    """

    radius_a, radius_b = FAR_FIELD.inputs["radii_m"]
    separations = FAR_FIELD.inputs["separations_m"]
    segments = FAR_FIELD.inputs["segments"]
    dipoles: list[float] = []
    deviations: list[float] = []
    orders: list[float] = []
    for family in FAR_FIELD.inputs["families"]:
        normal_b = tuple(family["normal_b"])
        family_deviations = []
        for separation in separations:
            if family["name"] == "coaxial":
                centre_b = (0.0, 0.0, separation)
            elif family["name"] == "coplanar":
                centre_b = (separation, 0.0, 0.0)
            else:
                centre_b = (0.3 * separation, 0.0, separation)
            loop_a = PlacedCircularLoop(
                radius_m=radius_a, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
            )
            loop_b = PlacedCircularLoop(
                radius_m=radius_b, centre_m=centre_b, normal=normal_b
            )
            measured = neumann_mutual_inductance_h(
                loop_a, loop_b, segments_a=segments, segments_b=segments
            )
            dipole = dipole_mutual_inductance_general_h(loop_a, loop_b)
            dipoles.append(dipole)
            family_deviations.append(measured / dipole - 1.0)
        deviations.extend(family_deviations)
        orders.extend(
            math.log2(family_deviations[index] / family_deviations[index + 1])
            for index in range(len(family_deviations) - 1)
        )
    FAR_FIELD.check_all({
        "dipole_mutual_inductances_h": dipoles,
        "dipole_ratio_deviations": deviations,
        "dipole_convergence_orders": orders,
    })


def test_the_self_inductance_singularity_fails_closed_and_the_gate_has_two_sides():
    """第六层：自感与未分辨构型**拒跑**，已分辨构型**接受**，分辨比对闭式。"""

    radius_a, radius_b, separation = SINGULARITY.inputs["probe_r1_r2_d_m"]
    resolved = SINGULARITY.inputs["probe_segments"]
    refused = SINGULARITY.inputs["refusal_segments"]
    loop_a, loop_b = _coaxial_pair(radius_a, radius_b, separation)

    self_refuses = False
    try:
        neumann_mutual_inductance_h(
            loop_a, loop_a, segments_a=resolved, segments_b=resolved
        )
    except ElectromagneticsError:
        self_refuses = True

    unresolved_refuses = False
    try:
        neumann_mutual_inductance_h(
            loop_a, loop_b, segments_a=refused, segments_b=refused
        )
    except ElectromagneticsError:
        unresolved_refuses = True

    accepted = math.isfinite(
        neumann_mutual_inductance_h(
            loop_a, loop_b, segments_a=resolved, segments_b=resolved
        )
    )
    SINGULARITY.check_all({
        "resolution_ratio": filament_resolution_ratio(
            loop_a, loop_b, segments_a=resolved, segments_b=resolved
        ),
        "refusal_resolution_ratio": filament_resolution_ratio(
            loop_a, loop_b, segments_a=refused, segments_b=refused
        ),
        "self_pairing_refuses": self_refuses,
        "unresolved_pairing_refuses": unresolved_refuses,
        "resolved_pairing_is_accepted": accepted,
    })


def test_the_millimetre_declaration_agrees_with_the_metre_one():
    """第七层：mm制声明与米制声明**逐位相同**，且值对上米制的参考路径。"""

    segments = UNITS.inputs["segments"]
    radius_a_mm = UNITS.inputs["radius_a_mm"]
    radius_b_mm = UNITS.inputs["radius_b_mm"]
    centre_b_mm = UNITS.inputs["centre_b_mm"]
    normal_b = tuple(UNITS.inputs["normal_b"])

    from_mm_a = PlacedCircularLoop.from_millimetres(
        radius_mm=radius_a_mm, centre_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    from_mm_b = PlacedCircularLoop.from_millimetres(
        radius_mm=radius_b_mm, centre_mm=tuple(centre_b_mm), normal=normal_b
    )
    from_metres_a = PlacedCircularLoop(
        radius_m=radius_a_mm / 1.0e3, centre_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)
    )
    from_metres_b = PlacedCircularLoop(
        radius_m=radius_b_mm / 1.0e3,
        centre_m=tuple(value / 1.0e3 for value in centre_b_mm),
        normal=normal_b,
    )
    by_millimetres = neumann_mutual_inductance_h(
        from_mm_a, from_mm_b, segments_a=segments, segments_b=segments
    )
    by_metres = neumann_mutual_inductance_h(
        from_metres_a, from_metres_b, segments_a=segments, segments_b=segments
    )
    UNITS.check_all({
        "mutual_inductance_from_millimetres_h": by_millimetres,
        "millimetre_and_metre_declarations_agree": by_millimetres == by_metres,
    })


def test_the_manifest_generator_is_the_one_that_produced_it():
    """清单与生成它的脚本必须同批变（轴7规则2的执行体）。"""

    MANIFEST.verify_generator(ROOT)


@pytest.mark.parametrize(
    "oracle",
    [COAXIAL, CONVERGENCE, GENERAL, RECIPROCITY, FAR_FIELD, SINGULARITY, UNITS],
)
def test_every_expected_quantity_carries_a_reason(oracle):
    """**判据本身被验**：每一个量的容差都要有理由，且理由不许是空话。"""

    for quantity in oracle.expected:
        reason = oracle.tolerance(quantity).reason
        assert len(reason) > 40, f"{oracle.id}:{quantity}的理由太短：{reason!r}"
