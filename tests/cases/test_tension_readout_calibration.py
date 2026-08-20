"""T-M2 conformance：标定、电气tare位置与gross过载对独立oracle。"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.drives import TensionSensor
from physics_engine.materials import EvidenceRef
from physics_engine.oracles import load_manifest
from physics_engine.tension_readout import (
    CalibrationDirection,
    CalibrationPurpose,
    LinearSpanCalibration,
    TareMode,
    TensionCalibrationPoint,
    TensionReadout,
)

CASE = Path(__file__).resolve().parents[2] / "cases" / "tension_readout_calibration"
MANIFEST = load_manifest(CASE / "oracle.json")
FIT = MANIFEST.oracles[0]
ANALOG = MANIFEST.oracles[1]
DIGITAL = MANIFEST.oracles[2]
OVERLOAD = MANIFEST.oracles[3]


def _estimated() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/t-m2-case-synthetic",
        method="Synthetic T-M2 independent-oracle case.",
    )


def _point(document) -> TensionCalibrationPoint:
    return TensionCalibrationPoint(
        point_id=document["point_id"],
        sensor_force_n=document["sensor_force_n"],
        reference_span_tension_n=document["reference_span_tension_n"],
        direction=CalibrationDirection(document["direction"]),
        purpose=CalibrationPurpose(document["purpose"]),
    )


def _calibration() -> LinearSpanCalibration:
    return LinearSpanCalibration.fit(
        calibration_id="calibration/t-m2-case",
        points=tuple(_point(point) for point in FIT.inputs["points"]),
        evidence=_estimated(),
        uncertainty_n=None,
    )


def test_fit_and_holdout_match_the_independent_least_squares_oracle():
    calibration = _calibration()
    holdout = calibration.evaluate_holdout(_point(FIT.inputs["holdout"]))
    actual = {
        "slope_span_per_sensor": calibration.slope_span_per_sensor,
        "intercept_n": calibration.intercept_n,
        "fit_rms_error_n": calibration.fit_rms_error_n,
        "fit_max_abs_error_n": calibration.fit_max_abs_error_n,
        "hysteresis_max_n": calibration.hysteresis_max_n,
        "holdout_predicted_span_tension_n": holdout.predicted_span_tension_n,
        "holdout_error_n": holdout.error_n,
    }
    for name, value in actual.items():
        tolerance = FIT.tolerances[name]
        assert value == pytest.approx(
            FIT.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        )
    assert holdout.point_id not in calibration.fit_point_ids


def test_both_tare_locations_match_the_independent_electrical_oracle():
    for oracle in (ANALOG, DIGITAL):
        inputs = oracle.inputs
        transducer = TensionSensor(
            inputs["full_scale_force_n"],
            inputs["output_at_full_scale_mv"],
            inputs["adc_bits"],
        )
        sample = TensionReadout(
            readout_id=f"readout/t-m2-case-{inputs['tare_mode']}",
            transducer=transducer,
            calibration=_calibration(),
            tare_mode=TareMode(inputs["tare_mode"]),
            evidence=_estimated(),
        ).measure(
            gross_axis_force_n=inputs["gross_axis_force_n"],
            tare_axis_force_n=inputs["tare_axis_force_n"],
        )
        for name, expected_value in oracle.expected.items():
            tolerance = oracle.tolerances[name]
            actual = getattr(sample, name)
            assert actual == pytest.approx(
                expected_value,
                rel=tolerance.rel_tol,
                abs=tolerance.abs_tol,
            ), f"{inputs['tare_mode']}: {name}"
        if inputs["tare_mode"] == "analog_pre_adc":
            assert sample.digitized_gross_axis_force_n is None
            assert sample.digitized_tare_axis_force_n is None


def test_gross_overload_cannot_be_erased_by_analog_tare():
    sample = TensionReadout(
        readout_id="readout/t-m2-case-overload",
        transducer=TensionSensor(100.0, 20.0, 12),
        calibration=_calibration(),
        tare_mode=TareMode(OVERLOAD.inputs["tare_mode"]),
        evidence=_estimated(),
    ).measure(
        gross_axis_force_n=OVERLOAD.inputs["gross_axis_force_n"],
        tare_axis_force_n=OVERLOAD.inputs["tare_axis_force_n"],
    )
    assert sample.net_axis_force_n == OVERLOAD.expected["net_axis_force_n"]
    assert sample.is_gross_saturated is OVERLOAD.expected["is_gross_saturated"]
    assert (
        sample.is_zeroed_path_saturated
        is OVERLOAD.expected["is_zeroed_path_saturated"]
    )
