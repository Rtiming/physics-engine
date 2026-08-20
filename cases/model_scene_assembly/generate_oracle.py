#!/usr/bin/env python3
"""P3-M1金标：模型输入到Scene、运动、虚拟frame和碰撞候选。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402


def main() -> int:
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/model_scene_assembly",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": "algorithm:oracle/model_scene_assembly",
            "algorithm_version": "1.0.0",
            "path_relative": "cases/model_scene_assembly/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:model-scene/time-and-collision",
                "inputs": {
                    "sample_times_s": [0.0, 1.0],
                    "workpiece_component_x_mm": [20.0, 0.0],
                    "component_from_asset_x_mm": 1.0,
                    "process_frame_x_mm": [10.0, 12.0],
                    "tension_asset_sha256": file_sha256(
                        HERE / "assets" / "tension-machine.collision.asset"
                    ),
                    "workpiece_asset_sha256": file_sha256(
                        HERE / "assets" / "workpiece.collision.asset"
                    ),
                },
                "expected": {
                    "body_ids": ["body/tension-machine", "body/workpiece"],
                    "excluded_component_ids": ["model-component/robot-display"],
                    "excluded_motion_track_ids": ["motion-track/robot-display"],
                    "workpiece_asset_x_mm": [21.0, 11.0, 1.0],
                    "process_frame_midpoint_x_mm": 11.0,
                    "candidate_pair_count": [1, 1],
                    "event_count": [0, 1],
                    "event_confidence": "broad_phase",
                    "qualification": "hypothesis_only",
                },
                "tolerances": {
                    "body_ids": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "只有物理组件进入Scene，显示组件保持明确排除。",
                    },
                    "excluded_component_ids": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "模型排除身份逐字。",
                    },
                    "excluded_motion_track_ids": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "显示运动排除身份逐字。",
                    },
                    "workpiece_asset_x_mm": {
                        "abs": 2.0e-15,
                        "rel": 2.0e-15,
                        "reason": "组件位姿线性插值后再显式后乘1mm资产安装偏置。",
                    },
                    "process_frame_midpoint_x_mm": {
                        "abs": 2.0e-15,
                        "rel": 2.0e-15,
                        "reason": "10mm到12mm线性插值中点。",
                    },
                    "candidate_pair_count": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "接触候选按声明固定为一对，不自动全连接。",
                    },
                    "event_count": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "0s分离，1s的AABB重叠。",
                    },
                    "event_confidence": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "fixture只声明AABB，不能冒充网格窄相。",
                    },
                    "qualification": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "合成规划场景永久是假设级。",
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
