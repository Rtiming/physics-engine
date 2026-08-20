"""T-M0 conformance：测力轮矢量合力对独立静力金标。"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.oracles import load_manifest
from physics_engine.tension_measurement import (
    equal_tension_resultant_force_n,
    web_force_on_roll_n,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "tension_measuring_roll_resultant"
MANIFEST = load_manifest(CASE / "oracle.json")
ORACLE = MANIFEST.oracles[0]


def test_vector_force_and_closed_form_magnitude_match_independent_oracle():
    expected_components = tuple(
        ORACLE.expected[name]
        for name in ("web_force_x_n", "web_force_y_n", "web_force_z_n")
    )
    component_tolerances = tuple(
        ORACLE.tolerances[name]
        for name in ("web_force_x_n", "web_force_y_n", "web_force_z_n")
    )
    magnitude_tolerance = ORACLE.tolerances["resultant_force_n"]
    for index, outgoing in enumerate(ORACLE.inputs["outgoing_tangent_xyz"]):
        force = web_force_on_roll_n(
            incoming_tension_n=ORACLE.inputs["incoming_tension_n"],
            outgoing_tension_n=ORACLE.inputs["outgoing_tension_n"],
            incoming_tangent_xyz=tuple(ORACLE.inputs["incoming_tangent_xyz"]),
            outgoing_tangent_xyz=tuple(outgoing),
        )
        for axis in range(3):
            tolerance = component_tolerances[axis]
            assert force[axis] == pytest.approx(
                expected_components[axis][index],
                rel=tolerance.rel_tol,
                abs=tolerance.abs_tol,
            )
        magnitude = math.sqrt(sum(component * component for component in force))
        assert magnitude == pytest.approx(
            ORACLE.expected["resultant_force_n"][index],
            rel=magnitude_tolerance.rel_tol,
            abs=magnitude_tolerance.abs_tol,
        )
        assert equal_tension_resultant_force_n(
            tension_n=ORACLE.inputs["incoming_tension_n"],
            wrap_angle_rad=math.radians(ORACLE.inputs["wrap_angles_deg"][index]),
        ) == pytest.approx(
            ORACLE.expected["resultant_force_n"][index],
            rel=magnitude_tolerance.rel_tol,
            abs=magnitude_tolerance.abs_tol,
        )


def test_mirroring_the_route_only_flips_the_lateral_force_sign():
    positive = web_force_on_roll_n(
        incoming_tension_n=17.0,
        outgoing_tension_n=17.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
    )
    negative = web_force_on_roll_n(
        incoming_tension_n=17.0,
        outgoing_tension_n=17.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, -1.0, 0.0),
    )
    assert positive[0] == negative[0]
    assert positive[1] == -negative[1]
    assert positive[2] == negative[2] == 0.0
