#!/usr/bin/env python3
"""P3-M0金标：模型组件、规划track与虚拟物理所有权的独立小场景。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402


def main() -> int:
    root_half = math.sqrt(0.5)
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/model_motion_physics_binding",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": "algorithm:oracle/model_motion_physics_binding",
            "algorithm_version": "1.0.0",
            "path_relative": "cases/model_motion_physics_binding/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:model-motion/ownership-and-midpoint",
                "inputs": {
                    "sample_coordinates_s": [0.0, 1.0],
                    "workpiece_translation_x_mm": [0.0, 2.0],
                    "process_translation_x_mm": [10.0, 12.0],
                    "rotation_z_deg": [0.0, 180.0],
                    "material_feed_length_mm": [0.0, 100.0],
                },
                "expected": {
                    "physical_body_ids": ["body/tension-machine", "body/workpiece"],
                    "body_behaviors": ["static", "kinematic"],
                    "excluded_component_ids": ["model-component/robot-display"],
                    "excluded_motion_track_ids": ["motion-track/robot-display"],
                    "virtual_frame_roles": ["process_frame"],
                    "midpoint_workpiece_translation_mm": [1.0, 0.0, 0.0],
                    "midpoint_workpiece_rotation_xyzw": [0.0, 0.0, root_half, root_half],
                    "final_source_state_values": [5.0, 360.0],
                    "final_material_feed_length_mm": 100.0,
                },
                "tolerances": {
                    "physical_body_ids": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "物理体身份逐字；显示组件不得静默进入物理。",
                    },
                    "body_behaviors": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "static/kinematic所有权逐字，不能靠运行时猜。",
                    },
                    "excluded_component_ids": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "未进入物理的模型组件必须显式列出。",
                    },
                    "excluded_motion_track_ids": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "显示运动保留在计划中，但必须显式排除在虚拟物理之外。",
                    },
                    "virtual_frame_roles": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "process frame是虚拟frame，不冒充刚体。",
                    },
                    "midpoint_workpiece_translation_mm": {
                        "abs": 2.0e-15,
                        "rel": 2.0e-15,
                        "reason": "两端0/2mm线性插值，中点为1mm。",
                    },
                    "midpoint_workpiece_rotation_xyzw": {
                        "abs": 2.0e-15,
                        "rel": 2.0e-15,
                        "reason": "0→180°短弧SLERP中点为绕z轴90°。",
                    },
                    "final_source_state_values": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "上游A1/E1状态只保留，不被物理接口改写。",
                    },
                    "final_material_feed_length_mm": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "累计送带是独立运动状态，末值精确100mm。",
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
