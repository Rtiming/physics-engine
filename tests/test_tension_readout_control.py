"""T-M3接线：ClosedTensionLoop可选读取测力轮/标定/采样通道。"""

from __future__ import annotations

import math

import pytest

from physics_engine.drives import MagneticParticleClutch, PidController, TensionSensor
from physics_engine.materials import EvidenceRef
from physics_engine.tension_control import ClosedTensionLoop, TensionControlError
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


def _estimated() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/t-m3-synthetic",
        method="Synthetic T-M3 line and measurement configuration.",
    )


def _calibration(gain_sensor_per_span: float) -> LinearSpanCalibration:
    points = []
    for index, tension in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        for direction in (
            CalibrationDirection.INCREASING,
            CalibrationDirection.DECREASING,
        ):
            points.append(
                TensionCalibrationPoint(
                    point_id=f"calibration-point/t-m3-{index}-{direction.value}",
                    sensor_force_n=gain_sensor_per_span * tension,
                    reference_span_tension_n=tension,
                    direction=direction,
                    purpose=CalibrationPurpose.FIT,
                )
            )
    return LinearSpanCalibration.fit(
        calibration_id="calibration/t-m3-wrap",
        points=tuple(points),
        evidence=_estimated(),
        uncertainty_n=None,
    )


def _channel(*, plant_dt_s: float, delay_samples: int = 0):
    root_half = math.sqrt(0.5)
    gain = math.sqrt(2.0)
    setpoint = steady_state_tension_n(
        brake_torque_nmm=1200.0,
        radius_mm=60.0,
        bearing_damping_nmm_s=50.0,
        line_speed_mm_s=20.0,
    )
    roll = MeasuringRoll(
        measurement_id="measurement/t-m3-roll",
        sensor_axis_xyz=(-root_half, root_half, 0.0),
        tare_force_n_xyz=(-5.0 * root_half, 5.0 * root_half, 0.0),
        support_shares=(0.5, 0.5),
        evidence=_estimated(),
    )
    readout = TensionReadout(
        readout_id="readout/t-m3",
        transducer=TensionSensor(100.0, 20.0, 16),
        calibration=_calibration(gain),
        tare_mode=TareMode.ANALOG_PRE_ADC,
        evidence=_estimated(),
    )
    return TensionReadoutChannel.at_steady_state(
        channel_id="measurement-channel/t-m3",
        roll=roll,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
        readout=readout,
        plant_dt_s=plant_dt_s,
        sample_decimation=10,
        delay_samples=delay_samples,
        span_tension_n=setpoint,
    )


def _loop(*, measurement_channel=None, sensor=None, plant_dt_s=1.0e-5):
    return ClosedTensionLoop.at_steady_state(
        span=FreeSpan("span/t-m3", 300.0, 60000.0),
        reel=PayoutReel("reel/t-m3", 60.0, 5000.0, 50.0),
        clutch=MagneticParticleClutch(23256.0, 50000.0, 0.0005),
        controller=PidController(0.0, 0.0, 0.0, 1.0),
        capstan=None,
        sensor=sensor,
        plant_dt_s=plant_dt_s,
        control_decimation=10,
        brake_torque_nmm=1200.0,
        line_speed_mm_s=20.0,
        delay_line=None,
        forbid_slack=True,
        measurement_channel=measurement_channel,
    )


def test_closed_loop_reads_the_optional_measurement_channel_and_advances_its_clock():
    channel = _channel(plant_dt_s=1.0e-5)
    loop = _loop(measurement_channel=channel)
    next_loop, sample = loop.step(takeup_speed_mm_s=20.0)
    assert sample.tension_n == pytest.approx(loop.setpoint_n, rel=2e-15)
    assert sample.measured_n == pytest.approx(loop.setpoint_n, abs=0.002)
    assert sample.measured_n != sample.tension_n
    assert next_loop.measurement_channel is not None
    assert next_loop.measurement_channel.step_index == 1


def test_legacy_sensor_path_remains_available_when_the_new_channel_is_absent():
    sensor = TensionSensor(49.033, 20.0, 12)
    loop = _loop(sensor=sensor)
    _, sample = loop.step(takeup_speed_mm_s=20.0)
    assert sample.measured_n == sensor.read_n(sample.tension_n)
    assert loop.measurement_channel is None


def test_red_sensor_and_measurement_channel_cannot_both_claim_the_same_observation():
    with pytest.raises(TensionControlError, match="both"):
        _loop(
            sensor=TensionSensor(49.033, 20.0, 12),
            measurement_channel=_channel(plant_dt_s=1.0e-5),
        )


def test_red_measurement_and_plant_clocks_must_be_identical():
    with pytest.raises(TensionControlError, match="plant_dt_s"):
        _loop(measurement_channel=_channel(plant_dt_s=2.0e-5), plant_dt_s=1.0e-5)
