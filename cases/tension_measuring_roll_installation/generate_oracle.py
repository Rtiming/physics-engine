#!/usr/bin/env python3
"""T-M1金标：敏感轴、tare与显式支承分配的独立刚体静力。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/tension_measuring_roll_installation"
ALGORITHM_VERSION = "1.0.0"


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def add(left, right):
    return [a + b for a, b in zip(left, right, strict=True)]


def scale(factor, vector):
    return [factor * value for value in vector]


def main() -> int:
    root_half = math.sqrt(0.5)
    resultant_axis = [-root_half, root_half, 0.0]
    transverse_axis = [root_half, root_half, 0.0]
    web_force = [-10.0, 10.0, 0.0]
    configurations = (
        {
            "id": "single_aligned_no_tare",
            "sensor_axis_xyz": resultant_axis,
            "tare_force_n_xyz": [0.0, 0.0, 0.0],
            "support_shares": [1.0],
        },
        {
            "id": "double_aligned_with_tare",
            "sensor_axis_xyz": resultant_axis,
            "tare_force_n_xyz": scale(5.0, resultant_axis),
            "support_shares": [0.5, 0.5],
        },
        {
            "id": "explicit_asymmetric_support_and_thirty_degree_axis",
            "sensor_axis_xyz": add(
                scale(math.cos(math.pi / 6.0), resultant_axis),
                scale(math.sin(math.pi / 6.0), transverse_axis),
            ),
            "tare_force_n_xyz": add(
                scale(5.0, resultant_axis), scale(2.0, transverse_axis)
            ),
            "support_shares": [0.3, 0.7],
        },
    )
    oracles = []
    for config in configurations:
        axis = config["sensor_axis_xyz"]
        tare = config["tare_force_n_xyz"]
        gross_force = add(web_force, tare)
        gross_axis = dot(gross_force, axis)
        tare_axis = dot(tare, axis)
        net_axis = gross_axis - tare_axis
        shares = config["support_shares"]
        oracles.append(
            {
                "id": f"oracle:tension-measuring-roll/{config['id']}",
                "inputs": {
                    "incoming_tension_n": 10.0,
                    "outgoing_tension_n": 10.0,
                    "incoming_tangent_xyz": [1.0, 0.0, 0.0],
                    "outgoing_tangent_xyz": [0.0, 1.0, 0.0],
                    "sensor_axis_xyz": axis,
                    "tare_force_n_xyz": tare,
                    "support_shares": shares,
                },
                "expected": {
                    "gross_axis_force_n": gross_axis,
                    "tare_axis_force_n": tare_axis,
                    "net_axis_force_n": net_axis,
                    "support_gross_forces_n": [gross_axis * share for share in shares],
                    "support_tare_forces_n": [tare_axis * share for share in shares],
                    "support_net_forces_n": [net_axis * share for share in shares],
                },
                "tolerances": {
                    "gross_axis_force_n": {
                        "abs": 8.0e-15,
                        "rel": 4.0e-16,
                        "reason": "刚体矢量和后一次轴投影；给约4 ulp覆盖第三个非轴对齐构型。",
                    },
                    "tare_axis_force_n": {
                        "abs": 4.0e-15,
                        "rel": 4.0e-16,
                        "reason": "tare单独投影，不能从gross倒猜，否则门对共同错误失明。",
                    },
                    "net_axis_force_n": {
                        "abs": 8.0e-15,
                        "rel": 4.0e-16,
                        "reason": "net=gross-tare；角度改变读数但tare后仍只保留web轴向分量。",
                    },
                    "support_gross_forces_n": {
                        "abs": 8.0e-15,
                        "rel": 4.0e-16,
                        "reason": "显式支承份额乘gross；不由实现猜单/双支承。",
                    },
                    "support_tare_forces_n": {
                        "abs": 4.0e-15,
                        "rel": 4.0e-16,
                        "reason": "同一支承份额必须作用于tare层。",
                    },
                    "support_net_forces_n": {
                        "abs": 8.0e-15,
                        "rel": 4.0e-16,
                        "reason": "各支承net之和必须回到总net，支承分配不能改变总测量力。",
                    },
                },
            }
        )
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/tension_measuring_roll_installation",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/tension_measuring_roll_installation/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
