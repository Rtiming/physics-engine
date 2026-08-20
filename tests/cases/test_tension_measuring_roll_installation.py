"""T-M1 conformance：敏感轴、tare与支承分层对独立静力金标。"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.materials import EvidenceRef
from physics_engine.oracles import load_manifest
from physics_engine.tension_measurement import MeasuringRoll

CASE = Path(__file__).resolve().parents[2] / "cases" / "tension_measuring_roll_installation"
MANIFEST = load_manifest(CASE / "oracle.json")


def _estimated() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/tension-roll-installation-synthetic",
        method="Synthetic installation geometry for an independent statics case.",
    )


def test_installation_layers_match_the_independent_statics_oracles():
    for oracle in MANIFEST.oracles:
        roll = MeasuringRoll(
            measurement_id=f"measurement/{oracle.id.rsplit('/', 1)[-1]}",
            sensor_axis_xyz=tuple(oracle.inputs["sensor_axis_xyz"]),
            tare_force_n_xyz=tuple(oracle.inputs["tare_force_n_xyz"]),
            support_shares=tuple(oracle.inputs["support_shares"]),
            evidence=_estimated(),
        )
        sample = roll.measure(
            incoming_tension_n=oracle.inputs["incoming_tension_n"],
            outgoing_tension_n=oracle.inputs["outgoing_tension_n"],
            incoming_tangent_xyz=tuple(oracle.inputs["incoming_tangent_xyz"]),
            outgoing_tangent_xyz=tuple(oracle.inputs["outgoing_tangent_xyz"]),
        )
        for name in (
            "gross_axis_force_n",
            "tare_axis_force_n",
            "net_axis_force_n",
            "support_gross_forces_n",
            "support_tare_forces_n",
            "support_net_forces_n",
        ):
            tolerance = oracle.tolerances[name]
            assert getattr(sample, name) == pytest.approx(
                oracle.expected[name],
                rel=tolerance.rel_tol,
                abs=tolerance.abs_tol,
            ), f"{oracle.id}: {name}"
        assert sum(sample.support_gross_forces_n) == pytest.approx(
            sample.gross_axis_force_n, rel=4.0e-16, abs=8.0e-15
        )
        assert sum(sample.support_tare_forces_n) == pytest.approx(
            sample.tare_axis_force_n, rel=4.0e-16, abs=4.0e-15
        )
        assert sum(sample.support_net_forces_n) == pytest.approx(
            sample.net_axis_force_n, rel=4.0e-16, abs=8.0e-15
        )


def test_tare_does_not_change_the_net_web_force_on_the_same_axis():
    aligned = next(
        oracle for oracle in MANIFEST.oracles if oracle.id.endswith("single_aligned_no_tare")
    )
    tared = next(
        oracle for oracle in MANIFEST.oracles if oracle.id.endswith("double_aligned_with_tare")
    )
    assert aligned.expected["net_axis_force_n"] == pytest.approx(
        tared.expected["net_axis_force_n"], rel=4.0e-16, abs=8.0e-15
    )
    assert aligned.expected["gross_axis_force_n"] != tared.expected["gross_axis_force_n"]
