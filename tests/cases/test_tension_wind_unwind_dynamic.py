"""T-M3 conformance：动态收放线与完整测量通道对独立离散oracle。"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.drives import MagneticParticleClutch, PidController, TensionSensor
from physics_engine.materials import EvidenceRef
from physics_engine.oracles import load_manifest
from physics_engine.tension_control import ClosedTensionLoop
from physics_engine.tension_measurement import MeasuringRoll
from physics_engine.tension_readout import (
    CalibrationDirection,
    CalibrationPurpose,
    LinearSpanCalibration,
    TareMode,
    TensionCalibrationPoint,
    TensionReadout,
    TensionReadoutChannel,
)
from physics_engine.transport import FreeSpan, PayoutReel, steady_state_tension_n

CASE = Path(__file__).resolve().parents[2] / "cases" / "tension_wind_unwind_dynamic"
MANIFEST = load_manifest(CASE / "oracle.json")
DYNAMIC = MANIFEST.oracles[0]
RADIUS = MANIFEST.oracles[1]


def _estimated() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/t-m3-case-synthetic",
        method="Synthetic T-M3 case with independent discrete oracle.",
    )


def _calibration(gain: float) -> LinearSpanCalibration:
    points = []
    for index, tension in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        for direction in (
            CalibrationDirection.INCREASING,
            CalibrationDirection.DECREASING,
        ):
            points.append(
                TensionCalibrationPoint(
                    point_id=f"calibration-point/t-m3-case-{index}-{direction.value}",
                    sensor_force_n=gain * tension,
                    reference_span_tension_n=tension,
                    direction=direction,
                    purpose=CalibrationPurpose.FIT,
                )
            )
    return LinearSpanCalibration.fit(
        calibration_id="calibration/t-m3-case",
        points=tuple(points),
        evidence=_estimated(),
        uncertainty_n=None,
    )


def _channel(inputs, *, radius_mm: float) -> TensionReadoutChannel:
    gain = inputs["roll_gain_sensor_per_span"]
    root_half = math.sqrt(0.5)
    tare = inputs["tare_axis_force_n"]
    roll = MeasuringRoll(
        measurement_id="measurement/t-m3-case-roll",
        sensor_axis_xyz=(-root_half, root_half, 0.0),
        tare_force_n_xyz=(-tare * root_half, tare * root_half, 0.0),
        support_shares=(0.5, 0.5),
        evidence=_estimated(),
    )
    readout = TensionReadout(
        readout_id="readout/t-m3-case",
        transducer=TensionSensor(
            inputs["adc_full_scale_n"], 20.0, inputs["adc_bits"]
        ),
        calibration=_calibration(gain),
        tare_mode=TareMode.ANALOG_PRE_ADC,
        evidence=_estimated(),
    )
    setpoint = steady_state_tension_n(
        brake_torque_nmm=inputs["brake_torque_nmm"],
        radius_mm=radius_mm,
        bearing_damping_nmm_s=inputs["bearing_damping_nmm_s"],
        line_speed_mm_s=inputs["line_speed_mm_s"],
    )
    return TensionReadoutChannel.at_steady_state(
        channel_id="measurement-channel/t-m3-case",
        roll=roll,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
        readout=readout,
        plant_dt_s=inputs["dt_s"],
        sample_decimation=inputs["control_decimation"],
        delay_samples=inputs["delay_samples"],
        span_tension_n=setpoint,
    )


def _run(controller_document) -> dict[str, float]:
    inputs = DYNAMIC.inputs
    radius = inputs["radius_mm"]
    loop = ClosedTensionLoop.at_steady_state(
        span=FreeSpan(
            "span/t-m3-case",
            inputs["geometric_length_mm"],
            inputs["axial_stiffness_n"],
        ),
        reel=PayoutReel(
            "reel/t-m3-case",
            radius,
            inputs["inertia_kg_mm2"],
            inputs["bearing_damping_nmm_s"],
        ),
        clutch=MagneticParticleClutch(
            inputs["torque_per_ampere_nmm"],
            inputs["rated_torque_nmm"],
            inputs["clutch_lag_s"],
        ),
        controller=PidController(
            controller_document["proportional"],
            controller_document["integral_gain"],
            controller_document["derivative"],
            controller_document["integral_limit"],
        ),
        capstan=None,
        sensor=None,
        plant_dt_s=inputs["dt_s"],
        control_decimation=inputs["control_decimation"],
        brake_torque_nmm=inputs["brake_torque_nmm"],
        line_speed_mm_s=inputs["line_speed_mm_s"],
        delay_line=None,
        forbid_slack=True,
        measurement_channel=_channel(inputs, radius_mm=radius),
    )
    setpoint = loop.setpoint_n
    peak = 0.0
    measured_peak = 0.0
    ise = 0.0
    final = None
    for _ in range(inputs["steps"]):
        loop, sample = loop.step(takeup_speed_mm_s=inputs["takeup_step_mm_s"])
        true_error = sample.tension_n - setpoint
        measured_error = sample.measured_n - setpoint
        peak = max(peak, abs(true_error))
        measured_peak = max(measured_peak, abs(measured_error))
        ise += true_error * true_error * inputs["dt_s"]
        final = sample
    assert final is not None
    return {
        "peak_true_excursion_n": peak,
        "peak_measured_excursion_n": measured_peak,
        "integral_squared_error_n2_s": ise,
        "final_true_error_n": final.tension_n - setpoint,
        "final_measured_error_n": final.measured_n - setpoint,
    }


@pytest.mark.batch
def test_dynamic_open_good_and_bad_loops_match_the_independent_discrete_oracle():
    results = [_run(controller) for controller in DYNAMIC.inputs["controllers"]]
    for quantity in results[0]:
        tolerance = DYNAMIC.tolerances[quantity]
        assert [result[quantity] for result in results] == pytest.approx(
            DYNAMIC.expected[quantity],
            rel=tolerance.rel_tol,
            abs=tolerance.abs_tol,
        )
    open_result = results[2]
    ratios = {
        "good_peak_ratio_to_open": results[0]["peak_true_excursion_n"]
        / open_result["peak_true_excursion_n"],
        "good_ise_ratio_to_open": results[0]["integral_squared_error_n2_s"]
        / open_result["integral_squared_error_n2_s"],
        "bad_peak_ratio_to_open": results[1]["peak_true_excursion_n"]
        / open_result["peak_true_excursion_n"],
    }
    for name, value in ratios.items():
        tolerance = DYNAMIC.tolerances[name]
        assert value == pytest.approx(
            DYNAMIC.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        )
    assert ratios["good_peak_ratio_to_open"] < 0.7
    assert ratios["good_ise_ratio_to_open"] < 0.2
    assert ratios["bad_peak_ratio_to_open"] > 0.9


def test_sampling_delay_and_radius_change_are_explicit_and_match_the_oracle():
    inputs = DYNAMIC.inputs
    channel = _channel(inputs, radius_mm=inputs["radius_mm"])
    assert channel.sample_period_s == inputs["sample_period_s"]
    assert channel.delay_s == inputs["measurement_delay_s"]

    steady = []
    displayed = []
    for radius in RADIUS.inputs["radii_mm"]:
        setpoint = steady_state_tension_n(
            brake_torque_nmm=RADIUS.inputs["brake_torque_nmm"],
            radius_mm=radius,
            bearing_damping_nmm_s=RADIUS.inputs["bearing_damping_nmm_s"],
            line_speed_mm_s=RADIUS.inputs["line_speed_mm_s"],
        )
        current = _channel(inputs, radius_mm=radius)
        steady.append(setpoint)
        displayed.append(current.held.electrical.displayed_span_tension_n)
    for name, actual in (
        ("steady_span_tensions_n", steady),
        ("displayed_span_tensions_n", displayed),
    ):
        tolerance = RADIUS.tolerances[name]
        assert actual == pytest.approx(
            RADIUS.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        )
    assert steady[1] < steady[0]
