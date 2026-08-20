"""T-M2张力电气读出：raw/tare、标定、量化、采样与时延。"""

from __future__ import annotations

import json
import math

import pytest

from physics_engine.canonical import canonical_sha256
from physics_engine.drives import TensionSensor
from physics_engine.materials import EvidenceRef
from physics_engine.tension_measurement import MeasuringRoll
from physics_engine.tension_readout import (
    TENSION_READOUT_CANONICAL_PROFILE,
    CalibrationDirection,
    CalibrationPurpose,
    LinearSpanCalibration,
    ReadoutError,
    TareMode,
    TensionCalibrationPoint,
    TensionReadout,
    TensionReadoutChannel,
    load_tension_readout_sample,
)

_SOURCE_SHA = "d" * 64


def _estimated() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/tension-readout-synthetic",
        method="Synthetic calibration and electrical topology for T-M2 tests.",
    )


def _calibrated_evidence() -> EvidenceRef:
    return EvidenceRef(
        grade="calibrated",
        evidence_id="evidence/tension-readout-reference-machine",
        method="Reference-force calibration with traceable source bytes.",
        source_sha256=_SOURCE_SHA,
    )


def _points(
    *, gain_sensor_per_span: float, offset_sensor_n: float = 0.0
) -> tuple[TensionCalibrationPoint, ...]:
    points = []
    for index, reference in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        for direction in (
            CalibrationDirection.INCREASING,
            CalibrationDirection.DECREASING,
        ):
            points.append(
                TensionCalibrationPoint(
                    point_id=f"calibration-point/level-{index}-{direction.value}",
                    sensor_force_n=gain_sensor_per_span * reference + offset_sensor_n,
                    reference_span_tension_n=reference,
                    direction=direction,
                    purpose=CalibrationPurpose.FIT,
                )
            )
    return tuple(points)


def _identity_calibration() -> LinearSpanCalibration:
    return LinearSpanCalibration.fit(
        calibration_id="calibration/t-m2-identity",
        points=_points(gain_sensor_per_span=1.0),
        evidence=_estimated(),
        uncertainty_n=None,
    )


def _roll() -> MeasuringRoll:
    root_half = math.sqrt(0.5)
    return MeasuringRoll(
        measurement_id="measurement/t-m2-roll",
        sensor_axis_xyz=(-root_half, root_half, 0.0),
        tare_force_n_xyz=(-5.0 * root_half, 5.0 * root_half, 0.0),
        support_shares=(0.5, 0.5),
        evidence=_estimated(),
    )


def _readout(*, tare_mode: TareMode, calibration=None) -> TensionReadout:
    return TensionReadout(
        readout_id=f"readout/t-m2-{tare_mode.value}",
        transducer=TensionSensor(100.0, 20.0, 12),
        calibration=calibration or _identity_calibration(),
        tare_mode=tare_mode,
        evidence=_estimated(),
    )


def test_five_level_forward_reverse_fit_and_holdout_are_separate():
    calibration = LinearSpanCalibration.fit(
        calibration_id="calibration/t-m2-affine",
        points=_points(gain_sensor_per_span=1.5, offset_sensor_n=0.3),
        evidence=_calibrated_evidence(),
        uncertainty_n=0.04,
    )
    assert calibration.slope_span_per_sensor == pytest.approx(2.0 / 3.0, rel=2e-15)
    assert calibration.intercept_n == pytest.approx(-0.2, abs=3e-15)
    assert calibration.fit_rms_error_n < 4e-15
    assert calibration.reference_levels_n == (0.0, 10.0, 20.0, 30.0, 40.0)
    assert len(calibration.fit_point_ids) == 10
    assert calibration.qualification == "calibrated_model"

    holdout = TensionCalibrationPoint(
        point_id="calibration-point/holdout-25",
        sensor_force_n=37.8,
        reference_span_tension_n=25.0,
        direction=CalibrationDirection.INCREASING,
        purpose=CalibrationPurpose.HOLDOUT,
    )
    check = calibration.evaluate_holdout(holdout)
    assert check.predicted_span_tension_n == pytest.approx(25.0, abs=4e-15)
    assert check.error_n == pytest.approx(0.0, abs=4e-15)
    assert holdout.point_id not in calibration.fit_point_ids


def test_tare_location_changes_quantization_but_not_the_declared_continuous_difference():
    analog = _readout(tare_mode=TareMode.ANALOG_PRE_ADC).measure(
        gross_axis_force_n=60.0, tare_axis_force_n=10.0
    )
    digital = _readout(tare_mode=TareMode.DIGITAL_POST_ADC).measure(
        gross_axis_force_n=60.0, tare_axis_force_n=10.0
    )

    assert analog.raw_bridge_output_mv == pytest.approx(12.0)
    assert analog.tare_bridge_output_mv == pytest.approx(2.0)
    assert analog.zeroed_bridge_output_mv == pytest.approx(10.0)
    assert digital.zeroed_bridge_output_mv == pytest.approx(10.0)
    assert analog.digitized_gross_axis_force_n is None
    assert analog.digitized_tare_axis_force_n is None
    assert analog.digitized_net_axis_force_n == TensionSensor(100.0, 20.0, 12).read_n(
        50.0
    )
    assert digital.digitized_gross_axis_force_n == TensionSensor(
        100.0, 20.0, 12
    ).read_n(60.0)
    assert digital.digitized_tare_axis_force_n == TensionSensor(
        100.0, 20.0, 12
    ).read_n(10.0)
    assert digital.digitized_net_axis_force_n == pytest.approx(
        digital.digitized_gross_axis_force_n - digital.digitized_tare_axis_force_n
    )
    assert analog.displayed_span_tension_n == analog.digitized_net_axis_force_n
    assert digital.displayed_span_tension_n == digital.digitized_net_axis_force_n


def test_tare_cannot_hide_a_physical_gross_overload():
    sample = _readout(tare_mode=TareMode.ANALOG_PRE_ADC).measure(
        gross_axis_force_n=110.0, tare_axis_force_n=20.0
    )
    assert sample.net_axis_force_n == 90.0
    assert sample.is_gross_saturated is True
    assert sample.is_zeroed_path_saturated is False
    assert sample.qualification == "hypothesis_only"


def test_readout_sample_round_trips_and_recomputes_derived_fields():
    sample = _readout(tare_mode=TareMode.DIGITAL_POST_ADC).measure(
        gross_axis_force_n=60.0, tare_axis_force_n=10.0
    ).sealed()
    payload = json.dumps(sample.to_document(), ensure_ascii=False).encode()
    assert load_tension_readout_sample(payload) == sample

    tampered = sample.to_document()
    tampered["displayed_span_tension_n"] += 1.0
    address_input = dict(tampered)
    address_input.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(
        address_input, TENSION_READOUT_CANONICAL_PROFILE
    )
    with pytest.raises(ReadoutError, match="derived readout fields"):
        load_tension_readout_sample(json.dumps(tampered).encode())


def test_sampling_delay_and_zero_order_hold_are_exact_in_sample_ticks():
    calibration = LinearSpanCalibration.fit(
        calibration_id="calibration/t-m2-wrap-gain",
        points=_points(gain_sensor_per_span=math.sqrt(2.0)),
        evidence=_estimated(),
        uncertainty_n=None,
    )
    channel = TensionReadoutChannel.at_steady_state(
        channel_id="measurement-channel/t-m2-delay",
        roll=_roll(),
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
        readout=_readout(
            tare_mode=TareMode.ANALOG_PRE_ADC, calibration=calibration
        ),
        plant_dt_s=0.001,
        sample_decimation=2,
        delay_samples=2,
        span_tension_n=10.0,
    )
    assert channel.sample_period_s == 0.002
    assert channel.delay_s == 0.004

    outputs = []
    for _ in range(6):
        channel, output = channel.advance(span_tension_n=20.0)
        outputs.append(output)
    assert [item.sample_tick for item in outputs] == [True, False, True, False, True, False]
    assert [item.displayed_span_tension_n for item in outputs[:4]] == pytest.approx(
        [10.0] * 4, abs=0.02
    )
    assert outputs[4].displayed_span_tension_n == pytest.approx(20.0, abs=0.02)
    assert outputs[5].displayed_span_tension_n == outputs[4].displayed_span_tension_n


# --- 必须红 ---------------------------------------------------------------


def test_red_fit_rejects_holdout_contamination():
    contaminated = list(_points(gain_sensor_per_span=1.0))
    contaminated.append(
        TensionCalibrationPoint(
            point_id="calibration-point/holdout-contamination",
            sensor_force_n=25.0,
            reference_span_tension_n=25.0,
            direction=CalibrationDirection.INCREASING,
            purpose=CalibrationPurpose.HOLDOUT,
        )
    )
    with pytest.raises(ReadoutError, match="holdout"):
        LinearSpanCalibration.fit(
            calibration_id="calibration/contaminated",
            points=tuple(contaminated),
            evidence=_estimated(),
            uncertainty_n=None,
        )


def test_red_fit_requires_five_levels_in_both_directions():
    with pytest.raises(ReadoutError, match="five distinct"):
        LinearSpanCalibration.fit(
            calibration_id="calibration/too-few",
            points=_points(gain_sensor_per_span=1.0)[:8],
            evidence=_estimated(),
            uncertainty_n=None,
        )


@pytest.mark.parametrize(
    ("sample_decimation", "delay_samples"),
    ((0, 1), (1.5, 1), (1, -1), (1, 0.5)),
)
def test_red_sampling_clock_requires_explicit_nonnegative_integer_counts(
    sample_decimation, delay_samples
):
    with pytest.raises(ReadoutError):
        TensionReadoutChannel.at_steady_state(
            channel_id="measurement-channel/bad-clock",
            roll=_roll(),
            incoming_tangent_xyz=(1.0, 0.0, 0.0),
            outgoing_tangent_xyz=(0.0, 1.0, 0.0),
            readout=_readout(tare_mode=TareMode.ANALOG_PRE_ADC),
            plant_dt_s=0.001,
            sample_decimation=sample_decimation,
            delay_samples=delay_samples,
            span_tension_n=10.0,
        )
