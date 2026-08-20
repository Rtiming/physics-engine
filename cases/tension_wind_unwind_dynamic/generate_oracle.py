#!/usr/bin/env python3
"""T-M3金标：独立离散收放线、测力轮、ADC、采样时延与好/坏控制器。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

DT_S = 1.0e-5
STEPS = 10_000
CONTROL_DECIMATION = 10
DELAY_SAMPLES = 5
EA_N = 60_000.0
L_GEO_MM = 300.0
RADIUS_MM = 60.0
INERTIA_KG_MM2 = 5_000.0
DAMPING_NMM_S = 50.0
BRAKE_NMM = 1_200.0
LINE_SPEED_MM_S = 20.0
TAKEUP_STEP_MM_S = 22.0
TORQUE_PER_AMP_NMM = 23_256.0
RATED_TORQUE_NMM = 50_000.0
CLUTCH_LAG_S = 0.0005
ADC_FULL_SCALE_N = 100.0
ADC_BITS = 16
ROLL_GAIN = math.sqrt(2.0)
TARE_AXIS_N = 5.0

CONTROLLERS = (
    (
        "good",
        0.007739938080495356,
        2.3909045882870474,
        1.7561293801847282e-05,
    ),
    ("bad", 0.00007739938080495356, 0.0, 0.0),
    ("open", 0.0, 0.0, 0.0),
)


def steady_tension(radius_mm: float) -> float:
    return BRAKE_NMM / radius_mm + DAMPING_NMM_S * LINE_SPEED_MM_S / radius_mm**2


def material_length(tension_n: float) -> float:
    return EA_N * L_GEO_MM / (EA_N + tension_n)


def tension(length_mm: float) -> float:
    strain = (L_GEO_MM - length_mm) / length_mm
    return EA_N * strain if strain > 0.0 else 0.0


def digitized_display(span_tension_n: float) -> float:
    sensor_force = ROLL_GAIN * span_tension_n
    step = ADC_FULL_SCALE_N / ((1 << ADC_BITS) - 1)
    digitized = round(max(0.0, min(ADC_FULL_SCALE_N, sensor_force)) / step) * step
    return digitized / ROLL_GAIN


def simulate(proportional: float, integral_gain: float, derivative: float) -> dict:
    setpoint = steady_tension(RADIUS_MM)
    length = material_length(setpoint)
    omega = LINE_SPEED_MM_S / RADIUS_MM
    torque = BRAKE_NMM
    feedforward = BRAKE_NMM / TORQUE_PER_AMP_NMM
    current = feedforward
    integral = 0.0
    previous_error = None
    initial_reading = digitized_display(setpoint)
    pending = [initial_reading] * DELAY_SAMPLES
    held = initial_reading
    peak = 0.0
    measured_peak = 0.0
    ise = 0.0
    final_true = setpoint
    final_measured = initial_reading

    for step_index in range(STEPS):
        true_tension = tension(length)
        if step_index % CONTROL_DECIMATION == 0:
            instantaneous = digitized_display(true_tension)
            held = pending.pop(0)
            pending.append(instantaneous)
            error = setpoint - held
            control_dt = CONTROL_DECIMATION * DT_S
            integral = max(-1.0e6, min(1.0e6, integral + error * control_dt))
            rate = (
                0.0
                if previous_error is None
                else (error - previous_error) / control_dt
            )
            command = proportional * error + integral_gain * integral + derivative * rate
            current = feedforward + command
            previous_error = error

        error_true = true_tension - setpoint
        error_measured = held - setpoint
        peak = max(peak, abs(error_true))
        measured_peak = max(measured_peak, abs(error_measured))
        ise += error_true * error_true * DT_S
        final_true = true_tension
        final_measured = held

        acceleration = 1000.0 * (
            true_tension * RADIUS_MM - torque - DAMPING_NMM_S * omega
        ) / INERTIA_KG_MM2
        omega += DT_S * acceleration
        if omega <= 0.0:
            raise RuntimeError("independent oracle left the declared positive-speed domain")
        length += DT_S * (omega * RADIUS_MM - TAKEUP_STEP_MM_S)
        target = max(
            -RATED_TORQUE_NMM,
            min(RATED_TORQUE_NMM, TORQUE_PER_AMP_NMM * current),
        )
        torque = target + (torque - target) * math.exp(-DT_S / CLUTCH_LAG_S)

    return {
        "peak_true_excursion_n": peak,
        "peak_measured_excursion_n": measured_peak,
        "integral_squared_error_n2_s": ise,
        "final_true_error_n": final_true - setpoint,
        "final_measured_error_n": final_measured - setpoint,
    }


def main() -> int:
    results = [simulate(kp, ki, kd) for _, kp, ki, kd in CONTROLLERS]
    open_result = results[2]
    expected = {
        quantity: [result[quantity] for result in results]
        for quantity in results[0]
    }
    expected.update(
        {
            "good_peak_ratio_to_open": results[0]["peak_true_excursion_n"]
            / open_result["peak_true_excursion_n"],
            "good_ise_ratio_to_open": results[0]["integral_squared_error_n2_s"]
            / open_result["integral_squared_error_n2_s"],
            "bad_peak_ratio_to_open": results[1]["peak_true_excursion_n"]
            / open_result["peak_true_excursion_n"],
        }
    )

    radii = (60.0, 80.0)
    radius_steady = [steady_tension(radius) for radius in radii]
    radius_displayed = [digitized_display(value) for value in radius_steady]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/tension_wind_unwind_dynamic",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": "algorithm:oracle/tension_wind_unwind_dynamic",
            "algorithm_version": "1.0.0",
            "path_relative": "cases/tension_wind_unwind_dynamic/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:tension-wind-unwind/dynamic-step",
                "inputs": {
                    "dt_s": DT_S,
                    "steps": STEPS,
                    "control_decimation": CONTROL_DECIMATION,
                    "delay_samples": DELAY_SAMPLES,
                    "sample_period_s": CONTROL_DECIMATION * DT_S,
                    "measurement_delay_s": DELAY_SAMPLES
                    * CONTROL_DECIMATION
                    * DT_S,
                    "axial_stiffness_n": EA_N,
                    "geometric_length_mm": L_GEO_MM,
                    "radius_mm": RADIUS_MM,
                    "inertia_kg_mm2": INERTIA_KG_MM2,
                    "bearing_damping_nmm_s": DAMPING_NMM_S,
                    "brake_torque_nmm": BRAKE_NMM,
                    "line_speed_mm_s": LINE_SPEED_MM_S,
                    "takeup_step_mm_s": TAKEUP_STEP_MM_S,
                    "torque_per_ampere_nmm": TORQUE_PER_AMP_NMM,
                    "rated_torque_nmm": RATED_TORQUE_NMM,
                    "clutch_lag_s": CLUTCH_LAG_S,
                    "adc_full_scale_n": ADC_FULL_SCALE_N,
                    "adc_bits": ADC_BITS,
                    "roll_gain_sensor_per_span": ROLL_GAIN,
                    "tare_axis_force_n": TARE_AXIS_N,
                    "controllers": [
                        {
                            "name": name,
                            "proportional": kp,
                            "integral_gain": ki,
                            "derivative": kd,
                            "integral_limit": 1.0e6,
                        }
                        for name, kp, ki, kd in CONTROLLERS
                    ],
                },
                "expected": expected,
                "tolerances": {
                    name: {
                        "abs": 2.0e-8,
                        "rel": 2.0e-7,
                        "reason": "独立标量离散推进；余量覆盖标定斜率与求和末位，不覆盖一步错序。",
                    }
                    for name in expected
                },
            },
            {
                "id": "oracle:tension-wind-unwind/radius-change",
                "inputs": {
                    "radii_mm": list(radii),
                    "brake_torque_nmm": BRAKE_NMM,
                    "bearing_damping_nmm_s": DAMPING_NMM_S,
                    "line_speed_mm_s": LINE_SPEED_MM_S,
                },
                "expected": {
                    "steady_span_tensions_n": radius_steady,
                    "displayed_span_tensions_n": radius_displayed,
                },
                "tolerances": {
                    "steady_span_tensions_n": {
                        "abs": 0.0,
                        "rel": 2.0e-15,
                        "reason": "独立稳态力矩平衡；半径增大必须降低同扭矩张力。",
                    },
                    "displayed_span_tensions_n": {
                        "abs": 3.0e-14,
                        "rel": 3.0e-15,
                        "reason": "16位ADC就近量化后除以测力轮增益。",
                    },
                },
            },
        ],
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote 2 oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
