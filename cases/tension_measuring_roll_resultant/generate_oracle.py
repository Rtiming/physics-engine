#!/usr/bin/env python3
"""T-M0金标：测力轮两侧等张力的矢量合力，独立于被验模块。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/tension_measuring_roll_resultant"
ALGORITHM_VERSION = "1.0.0"
TENSION_N = 17.0

#: 输入方向作为金标输入落盘；60°与90°的非平凡分量能抓符号/补角错误。
CONFIGURATIONS = (
    (0.0, (1.0, 0.0, 0.0)),
    (60.0, (0.5, math.sqrt(3.0) / 2.0, 0.0)),
    (90.0, (0.0, 1.0, 0.0)),
    (180.0, (-1.0, 0.0, 0.0)),
)


def main() -> int:
    force_x = []
    force_y = []
    force_z = []
    magnitudes = []
    outgoing = []
    angles = []
    for degrees, tangent in CONFIGURATIONS:
        # 独立静力式：上游段对轮取-T*t_in，下游段取+T*t_out。
        vector = (TENSION_N * (tangent[0] - 1.0), TENSION_N * tangent[1], 0.0)
        force_x.append(vector[0])
        force_y.append(vector[1])
        force_z.append(vector[2])
        # 第二条独立路径：只用包角闭式，不从上面的vector取模。
        magnitudes.append(2.0 * TENSION_N * math.sin(math.radians(degrees) / 2.0))
        outgoing.append(list(tangent))
        angles.append(degrees)

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/tension_measuring_roll_resultant",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/tension_measuring_roll_resultant/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:tension-measuring-roll/equal-tension-resultant",
                "inputs": {
                    "incoming_tension_n": TENSION_N,
                    "outgoing_tension_n": TENSION_N,
                    "incoming_tangent_xyz": [1.0, 0.0, 0.0],
                    "outgoing_tangent_xyz": outgoing,
                    "wrap_angles_deg": angles,
                },
                "expected": {
                    "web_force_x_n": force_x,
                    "web_force_y_n": force_y,
                    "web_force_z_n": force_z,
                    "resultant_force_n": magnitudes,
                },
                "tolerances": {
                    "web_force_x_n": {
                        "abs": 4.0e-15,
                        "rel": 0.0,
                        "reason": "直接矢量和只有一次乘加；4e-15约覆盖17N量级的2 ulp。",
                    },
                    "web_force_y_n": {
                        "abs": 4.0e-15,
                        "rel": 0.0,
                        "reason": "同上；判绝对误差可覆盖0分量而不在零点除法。",
                    },
                    "web_force_z_n": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "全部输入在xy平面，z分量按构造必须逐位为0.0。",
                    },
                    "resultant_force_n": {
                        "abs": 4.0e-15,
                        "rel": 4.0e-16,
                        "reason": "2*T*sin(beta/2)独立闭式；余量约2 ulp，不给符号或补角错误留空间。",
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
