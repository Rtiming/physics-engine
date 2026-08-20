#!/usr/bin/env python3
"""T-M2金标：五级正反程线性标定、holdout与两种tare位置。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/tension_readout_calibration"
ALGORITHM_VERSION = "1.0.0"
FULL_SCALE_N = 100.0
FULL_SCALE_MV = 20.0
ADC_BITS = 12
COUNTS = (1 << ADC_BITS) - 1
STEP_N = FULL_SCALE_N / COUNTS


def quantize(force_n: float) -> float:
    clamped = max(0.0, min(FULL_SCALE_N, force_n))
    return round(clamped / STEP_N) * STEP_N


def main() -> int:
    points = []
    for index, reference in enumerate((0.0, 10.0, 20.0, 30.0, 40.0)):
        for direction, hysteresis in (("increasing", 0.03), ("decreasing", -0.03)):
            points.append(
                {
                    "point_id": f"calibration-point/t-m2-{index}-{direction}",
                    "sensor_force_n": 1.5 * reference + 0.3 + hysteresis,
                    "reference_span_tension_n": reference,
                    "direction": direction,
                    "purpose": "fit",
                }
            )
    xs = [point["sensor_force_n"] for point in points]
    ys = [point["reference_span_tension_n"] for point in points]
    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    slope = math.fsum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    ) / math.fsum((x - mean_x) ** 2 for x in xs)
    intercept = mean_y - slope * mean_x
    residuals = [slope * x + intercept - y for x, y in zip(xs, ys, strict=True)]
    rms = math.sqrt(math.fsum(value * value for value in residuals) / len(residuals))
    maximum = max(abs(value) for value in residuals)
    hysteresis_max = abs(0.06 * slope)
    holdout_sensor = 1.5 * 25.0 + 0.3 + 0.015
    holdout_predicted = slope * holdout_sensor + intercept

    modes = []
    for mode in ("analog_pre_adc", "digital_post_adc"):
        gross = 60.0
        tare = 10.0
        net = gross - tare
        raw_mv = min(FULL_SCALE_N, gross) / FULL_SCALE_N * FULL_SCALE_MV
        tare_mv = min(FULL_SCALE_N, tare) / FULL_SCALE_N * FULL_SCALE_MV
        if mode == "analog_pre_adc":
            digitized_gross = None
            digitized_tare = None
            digitized_net = quantize(net)
            zeroed_mv = net / FULL_SCALE_N * FULL_SCALE_MV
        else:
            digitized_gross = quantize(gross)
            digitized_tare = quantize(tare)
            digitized_net = digitized_gross - digitized_tare
            zeroed_mv = raw_mv - tare_mv
        modes.append(
            {
                "tare_mode": mode,
                "raw_bridge_output_mv": raw_mv,
                "tare_bridge_output_mv": tare_mv,
                "zeroed_bridge_output_mv": zeroed_mv,
                "digitized_gross_axis_force_n": digitized_gross,
                "digitized_tare_axis_force_n": digitized_tare,
                "digitized_net_axis_force_n": digitized_net,
                "displayed_span_tension_n": slope * digitized_net + intercept,
            }
        )

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/tension_readout_calibration",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/tension_readout_calibration/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:tension-readout/five-level-bidirectional-fit",
                "inputs": {
                    "points": points,
                    "holdout": {
                        "point_id": "calibration-point/t-m2-holdout-25",
                        "sensor_force_n": holdout_sensor,
                        "reference_span_tension_n": 25.0,
                        "direction": "increasing",
                        "purpose": "holdout",
                    },
                },
                "expected": {
                    "slope_span_per_sensor": slope,
                    "intercept_n": intercept,
                    "fit_rms_error_n": rms,
                    "fit_max_abs_error_n": maximum,
                    "hysteresis_max_n": hysteresis_max,
                    "holdout_predicted_span_tension_n": holdout_predicted,
                    "holdout_error_n": holdout_predicted - 25.0,
                },
                "tolerances": {
                    name: {
                        "abs": 8.0e-15,
                        "rel": 8.0e-15,
                        "reason": "独立普通最小二乘与逐级正反程差；余量覆盖求和末位。",
                    }
                    for name in (
                        "slope_span_per_sensor",
                        "intercept_n",
                        "fit_rms_error_n",
                        "fit_max_abs_error_n",
                        "hysteresis_max_n",
                        "holdout_predicted_span_tension_n",
                        "holdout_error_n",
                    )
                },
            },
            *[
                {
                    "id": f"oracle:tension-readout/{mode['tare_mode']}",
                    "inputs": {
                        "full_scale_force_n": FULL_SCALE_N,
                        "output_at_full_scale_mv": FULL_SCALE_MV,
                        "adc_bits": ADC_BITS,
                        "gross_axis_force_n": 60.0,
                        "tare_axis_force_n": 10.0,
                        "tare_mode": mode["tare_mode"],
                    },
                    "expected": {
                        name: value
                        for name, value in mode.items()
                        if name != "tare_mode" and value is not None
                    },
                    "tolerances": {
                        name: {
                            "abs": 2.0e-14,
                            "rel": 2.0e-15,
                            "reason": "线性mV、就近量化和一次标定换算；半台阶错误远大于本余量。",
                        }
                        for name, value in mode.items()
                        if name != "tare_mode" and value is not None
                    },
                }
                for mode in modes
            ],
            {
                "id": "oracle:tension-readout/gross-overload-survives-tare",
                "inputs": {
                    "gross_axis_force_n": 110.0,
                    "tare_axis_force_n": 20.0,
                    "tare_mode": "analog_pre_adc",
                },
                "expected": {
                    "net_axis_force_n": 90.0,
                    "is_gross_saturated": True,
                    "is_zeroed_path_saturated": False,
                },
                "tolerances": {
                    "net_axis_force_n": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "110−20是精确整数；tare不能清掉物理gross过载。",
                    },
                    "is_gross_saturated": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "布尔分支逐位；gross载荷独立于tare后的net力。",
                    },
                    "is_zeroed_path_saturated": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "布尔分支逐位；analog清零路径只看90N net力。",
                    },
                },
            },
        ],
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote 4 oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
