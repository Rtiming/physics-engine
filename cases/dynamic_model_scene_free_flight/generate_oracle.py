#!/usr/bin/env python3
"""P3-M2金标：偏心geometry绕COM的自由刚体运动。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402


def main() -> int:
    half = math.sqrt(0.5)
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/dynamic_model_scene_free_flight",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": "algorithm:oracle/dynamic_model_scene_free_flight",
            "algorithm_version": "1.0.0",
            "path_relative": "cases/dynamic_model_scene_free_flight/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:model-scene/dynamic-com-orbit",
                "inputs": {
                    "geometry_origin_x_mm": 10.0,
                    "centroid_in_geometry_x_mm": 2.0,
                    "angular_velocity_z_rad_per_s": math.pi,
                    "dt_s": 0.001,
                    "steps": 500,
                    "asset_sha256": file_sha256(
                        HERE / "assets" / "workpiece.collision.asset"
                    ),
                },
                "expected": {
                    "initial_com_position_mm": [12.0, 0.0, 0.0],
                    "initial_geometry_origin_mm": [10.0, 0.0, 0.0],
                    "final_com_position_mm": [12.0, 0.0, 0.0],
                    "final_attitude_xyzw": [0.0, 0.0, half, half],
                    "final_geometry_origin_mm": [12.0, -2.0, 0.0],
                    "renormalisations": 500,
                    "qualification": "hypothesis_only",
                },
                "tolerances": {
                    "initial_com_position_mm": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "geometry原点10mm加geometry frame内质心2mm。",
                    },
                    "initial_geometry_origin_mm": {
                        "abs": 2.0e-15,
                        "rel": 2.0e-15,
                        "reason": "COM状态反算必须回到初始geometry位姿。",
                    },
                    "final_com_position_mm": {
                        "abs": 2.0e-12,
                        "rel": 2.0e-12,
                        "reason": "无外力时COM保持不动。",
                    },
                    "final_attitude_xyzw": {
                        "abs": 2.0e-11,
                        "rel": 2.0e-11,
                        "reason": "主轴恒角速度πrad/s推进0.5s，绕z转90°。",
                    },
                    "final_geometry_origin_mm": {
                        "abs": 5.0e-11,
                        "rel": 5.0e-11,
                        "reason": "geometry原点相对COM的(-2,0,0)旋转到(0,-2,0)。",
                    },
                    "renormalisations": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "每个RK4步归一化一次。",
                    },
                    "qualification": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "全部输入为合成估计。",
                    },
                },
            }
        ],
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote 1 oracle, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
