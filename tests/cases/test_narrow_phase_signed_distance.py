"""`case/narrow_phase_signed_distance`的conformance门（轴7规则3）。

本文件**不含任何闭式**：它只把构型摆好、调窄相、把结果交给清单比对。
判据与容差理由全在`cases/narrow_phase_signed_distance/oracle.json`里，
那份清单由同目录的`generate_oracle.py`产出（一行不import`collision`）。

**必红那一族不在这里**，在`tests/test_narrow_phase_shapes.py`：
注错用例判的是"门有没有牙齿"，而本文件判的是"内核对不对"，两件事分开放。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.collision import (
    HalfSpace,
    field_separation_mm,
    half_space_separation_mm,
    narrow_phase_separation_mm,
)
from physics_engine.contact.field import sample_narrow_band
from physics_engine.oracles import load_manifest
from physics_engine.shapes import (
    CollisionShape,
    FiniteCylinder,
    PosedBody,
    RoundedBox,
    SimBody,
    Sphere,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(
    ROOT / "cases/narrow_phase_signed_distance/oracle.json", root=ROOT
)


def _body(name, shape, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return PosedBody(
        SimBody(body_id=f"body/{name}", collision=CollisionShape(shape, "fitted")),
        translation_mm=translation,
        rotation_xyzw=rotation,
    )


def _key(spacing_mm: float) -> str:
    return "h_" + f"{spacing_mm:g}".replace(".", "p")


# --------------------------------------------------------------------------
# 一、解析五族
# --------------------------------------------------------------------------


def test_analytic_pairs_match_the_manifest():
    case = MANIFEST.oracle("oracle:narrow_phase_signed_distance/analytic_pairs")
    radii = case.inputs["sphere_radii_mm"]
    box_a = tuple(case.inputs["box_a_half_extents_mm"])
    box_b = tuple(case.inputs["box_b_half_extents_mm"])
    cylinder = FiniteCylinder(
        radius_mm=case.inputs["cylinder_radius_mm"],
        half_width_mm=case.inputs["cylinder_half_width_mm"],
    )
    half = case.inputs["rotated_box_half_extent_mm"]
    turned = _body(
        "turned",
        RoundedBox(half_extents_mm=(half, half, half), fillet_radius_mm=0.0),
        rotation=(0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)),
    )
    plane = HalfSpace(point_mm=(0.0, 0.0, 1.0), unit_normal=(0.0, 0.0, 1.0))

    def spheres(gap_mm: float) -> float:
        return narrow_phase_separation_mm(
            _body("a", Sphere(radius_mm=radii[0])),
            _body("b", Sphere(radius_mm=radii[1]), (gap_mm, 0.0, 0.0)),
        ).separation_mm

    def to_plane(centre_z_mm: float) -> float:
        return half_space_separation_mm(
            _body("p", Sphere(radius_mm=2.0), (0.0, 0.0, centre_z_mm)), plane
        )

    def boxes(offset_x_mm: float) -> float:
        return narrow_phase_separation_mm(
            _body("ba", RoundedBox(half_extents_mm=box_a, fillet_radius_mm=0.0)),
            _body(
                "bb",
                RoundedBox(half_extents_mm=box_b, fillet_radius_mm=0.0),
                (offset_x_mm, 0.0, 0.0),
            ),
        ).separation_mm

    def to_cylinder(position) -> float:
        return narrow_phase_separation_mm(
            _body("ball", Sphere(radius_mm=1.0), position), _body("cyl", cylinder)
        ).separation_mm

    def to_turned_box(centre_x_mm: float) -> float:
        return narrow_phase_separation_mm(
            _body("ball", Sphere(radius_mm=2.0), (centre_x_mm, 0.0, 0.0)), turned
        ).separation_mm

    case.check_all(
        {
            "sphere_sphere_separated_mm": spheres(12.0),
            "sphere_sphere_penetrating_mm": spheres(5.0),
            "sphere_half_space_separated_mm": to_plane(4.0),
            "sphere_half_space_penetrating_mm": to_plane(2.0),
            "box_box_separated_mm": boxes(20.0),
            "box_box_penetrating_mm": boxes(12.0),
            "sphere_cylinder_side_mm": to_cylinder((12.0, 0.0, 0.0)),
            "sphere_cylinder_end_mm": to_cylinder((0.0, 0.0, 6.0)),
            "sphere_cylinder_inside_mm": to_cylinder((0.0, 0.0, 0.0)),
            "sphere_rotated_box_outside_mm": to_turned_box(18.0),
            "sphere_rotated_box_inside_mm": to_turned_box(14.0),
        }
    )


# --------------------------------------------------------------------------
# 二、场那一族：三种曲率各一条
# --------------------------------------------------------------------------


def _cube_field(distance_mm, *, extent_mm, spacing_mm, half_height_mm=None, band_mm=None):
    height = extent_mm if half_height_mm is None else half_height_mm
    band = band_mm if band_mm is not None else max(4.0, 2.0 * spacing_mm * math.sqrt(3.0) + 1.0)
    return sample_narrow_band(
        distance_mm,
        origin_mm=(-extent_mm, -extent_mm, -height),
        spacing_mm=spacing_mm,
        node_counts=(
            int(2.0 * extent_mm / spacing_mm) + 1,
            int(2.0 * extent_mm / spacing_mm) + 1,
            int(2.0 * height / spacing_mm) + 1,
        ),
        band_mm=band,
    )


def test_the_plane_field_is_exact_and_declares_zero_bias():
    case = MANIFEST.oracle("oracle:narrow_phase_signed_distance/plane_field")
    probe_z = case.inputs["probe_z_mm"]
    obstacle = _body("plane", Sphere(radius_mm=1.0))
    measured: dict[str, float] = {}
    for spacing in case.inputs["spacings_mm"]:
        field = _cube_field(lambda point: point[2], extent_mm=6.0, spacing_mm=spacing)
        result = field_separation_mm(field, obstacle, (0.3, -1.1, probe_z), 0.0)
        assert result.confidence == "sampled_field"
        assert result.resolution_mm == spacing
        measured[f"separation_{_key(spacing)}_mm"] = result.separation_mm
        measured[f"bias_estimate_{_key(spacing)}_mm"] = result.estimated_bias_mm
    case.check_all(measured)


def test_the_convex_sphere_field_bias_matches_the_leading_term():
    case = MANIFEST.oracle(
        "oracle:narrow_phase_signed_distance/convex_sphere_field_bias"
    )
    radius = case.inputs["obstacle_radius_mm"]
    rho = case.inputs["probe_rho_mm"]
    obstacle = _body("obs", Sphere(radius_mm=radius))
    measured: dict[str, float] = {}
    for spacing in case.inputs["spacings_mm"]:
        field = _cube_field(
            lambda point: math.sqrt(
                point[0] * point[0] + point[1] * point[1] + point[2] * point[2]
            )
            - radius,
            extent_mm=radius + 4.0,
            spacing_mm=spacing,
            band_mm=max(3.5, 2.0 * spacing * math.sqrt(3.0) + 1.0),
        )
        result = field_separation_mm(field, obstacle, (rho, 0.0, 0.0), 0.0)
        #: **方向**这一半不进清单（它是符号不是数），但它是本格的物理内容：
        #: 凸障碍上场偏松，物体沉得更深。
        assert result.estimated_bias_mm > 0.0
        assert result.separation_mm > rho - radius
        measured[f"bias_estimate_{_key(spacing)}_mm"] = result.estimated_bias_mm
    case.check_all(measured)


def test_the_concave_bore_field_bias_flips_sign_and_matches_its_own_leading_term():
    case = MANIFEST.oracle(
        "oracle:narrow_phase_signed_distance/concave_bore_field_bias"
    )
    bore_radius = case.inputs["bore_radius_mm"]
    rho = case.inputs["probe_rho_mm"]
    obstacle = _body("bore", Sphere(radius_mm=1.0))
    measured: dict[str, float] = {}
    for spacing in case.inputs["spacings_mm"]:
        field = _cube_field(
            lambda point: bore_radius - math.hypot(point[0], point[1]),
            extent_mm=bore_radius + 4.0,
            spacing_mm=spacing,
            half_height_mm=3.0,
            band_mm=max(3.0, 2.0 * spacing * math.sqrt(3.0) + 0.5),
        )
        result = field_separation_mm(field, obstacle, (rho, 0.0, 0.0), 0.0)
        #: 与上一格**方向相反**：凹面上场偏紧，物体被挡在更外面。
        assert result.estimated_bias_mm < 0.0
        assert result.separation_mm < bore_radius - rho
        measured[f"bias_estimate_{_key(spacing)}_mm"] = result.estimated_bias_mm
    case.check_all(measured)


def test_the_nonconvex_torus_field_error_is_second_order():
    case = MANIFEST.oracle(
        "oracle:narrow_phase_signed_distance/nonconvex_torus_order"
    )
    major = case.inputs["major_radius_mm"]
    minor = case.inputs["minor_radius_mm"]

    def torus(point):
        return math.hypot(math.hypot(point[0], point[1]) - major, point[2]) - minor

    probes = ((17.0, 0.0, 0.0), (0.0, 16.2, 1.3), (9.5, 0.0, 0.0), (12.0, 0.0, 5.2))
    obstacle = _body("torus", Sphere(radius_mm=1.0))
    errors = []
    for spacing in case.inputs["spacings_mm"]:
        field = _cube_field(
            torus,
            extent_mm=major + minor + 4.0,
            spacing_mm=spacing,
            half_height_mm=8.0,
            band_mm=max(3.0, 2.0 * spacing * math.sqrt(3.0) + 0.5),
        )
        worst = 0.0
        for probe in probes:
            result = field_separation_mm(field, obstacle, probe, 0.0)
            assert result.confidence == "sampled_field"
            worst = max(worst, abs(result.separation_mm - torus(probe)))
        errors.append(worst)
    case.check_all(
        {
            "order_ratio_coarse": errors[0] / errors[1],
            "order_ratio_fine": errors[1] / errors[2],
        }
    )


def test_the_manifest_covers_every_oracle_this_file_exercises():
    """清单里不许有本文件从没撞过的格子——**没被撞过的判据不是判据**。"""

    exercised = {
        "oracle:narrow_phase_signed_distance/analytic_pairs",
        "oracle:narrow_phase_signed_distance/plane_field",
        "oracle:narrow_phase_signed_distance/convex_sphere_field_bias",
        "oracle:narrow_phase_signed_distance/concave_bore_field_bias",
        "oracle:narrow_phase_signed_distance/nonconvex_torus_order",
    }
    assert {case.id for case in MANIFEST.oracles} == exercised


@pytest.mark.parametrize(
    "oracle_id", [case.id for case in MANIFEST.oracles], ids=lambda value: value.split("/")[-1]
)
def test_every_expected_quantity_carries_a_tolerance_with_a_reason(oracle_id):
    """每条`expected`都要有容差且容差要有理由——判据表第三列的执行面。"""

    case = MANIFEST.oracle(oracle_id)
    for quantity in case.expected:
        tolerance = case.tolerances[quantity]
        assert tolerance.reason.strip(), f"{oracle_id}/{quantity}的容差没有理由"
