#!/usr/bin/env python3
"""生成`cases/rotated_aabb/oracle.json`——旋转AABB对Arvo闭式解。

金标来源（轴7规则1）：Arvo《Graphics Gems》(1990) "Transforming Axis-Aligned
Bounding Boxes"的**中心-半边长分解**：

    c = (lo + hi)/2      h = (hi − lo)/2
    c' = R·c + t         h'_i = Σ_j |R_ij|·h_j
    世界盒 = (c' − h', c' + h')

**易错点**：局部AABB不居中时不能直接对`hi`套这式子（胶囊沿+x放时局部盒是
(−3,−3,−3)..(23,3,3)，中心在x=10）。本案例第一条就是非居中盒，专治这个错。

独立性（轴7规则4）：本脚本自带四元数→矩阵与Arvo求和，不import被验的
`shapes.py`；局部AABB是**手写的字面量**（推导见`case.md`），连"局部盒怎么算"
这一段都不与内核共享——所以圆角半径漏计、法兰外径漏计也逃不掉。

用法：`PYTHONPATH=src .venv/bin/python cases/rotated_aabb/generate_oracle.py`
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import (  # noqa: E402
    ORACLE_MANIFEST_FACET,
    ORACLE_MANIFEST_VERSION,
    file_sha256,
    write_manifest,
)

ALGORITHM_ID = "algorithm:oracle/rotated_aabb"
ALGORITHM_VERSION = "1.0.0"

_LOCAL_REASON = (
    "局部盒由形状参数的加减与min/max得到，量级50mm下双精度精确可表示；"
    "abs=1e-12mm是对『没有额外误差源』的声明，不写==0是不想把求值次序冻进契约。"
)
_WORLD_REASON = (
    "内核枚举八角点后取min/max，Arvo走中心-半边长求和——两条路径的加法次序不同，"
    "量级50mm下双精度误差~1e-14mm；abs=1e-9mm留五个量级余量。"
    "rel=0：判据是绝对包盒坐标，写成相对量在坐标接近0的轴上会退化成==0。"
)


def _quaternion(axis, angle_rad):
    """轴角→单位四元数(xyzw)。分量写进清单时已归一化，位姿层的单位检查即过。"""

    norm = math.sqrt(sum(component * component for component in axis))
    unit = [component / norm for component in axis]
    half = angle_rad / 2.0
    sine = math.sin(half)
    return (unit[0] * sine, unit[1] * sine, unit[2] * sine, math.cos(half))


def _rows(quaternion):
    x, y, z, w = quaternion
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _arvo_world_aabb(local_min, local_max, quaternion, translation):
    rows = _rows(quaternion)
    centre = [(local_min[i] + local_max[i]) / 2.0 for i in range(3)]
    half = [(local_max[i] - local_min[i]) / 2.0 for i in range(3)]
    world_centre = [
        sum(rows[axis][j] * centre[j] for j in range(3)) + translation[axis] for axis in range(3)
    ]
    world_half = [sum(abs(rows[axis][j]) * half[j] for j in range(3)) for axis in range(3)]
    return (
        [world_centre[axis] - world_half[axis] for axis in range(3)],
        [world_centre[axis] + world_half[axis] for axis in range(3)],
    )


#: 每条：(名字, 形状声明, 手写局部盒, 四元数, 平移, 它专治哪个错)
#: 局部盒的推导逐条写在`case.md`第一节，这里只放结果。
ORACLES = (
    (
        "capsule_offcentre_general_rotation",
        {
            "kind": "capsule",
            "point_a_mm": [0.0, 0.0, 0.0],
            "point_b_mm": [20.0, 0.0, 0.0],
            "radius_mm": 3.0,
        },
        (-3.0, -3.0, -3.0),
        (23.0, 3.0, 3.0),
        _quaternion((1.0, 2.0, 3.0), 0.7),
        (10.0, -20.0, 30.0),
        "非居中局部盒 + 一般轴旋转 + 非零平移：直接对hi套Arvo式、矩阵转置、平移漏加三错皆红",
    ),
    (
        "rounded_box_fillet_counted",
        {"kind": "rounded_box", "half_extents_mm": [30.0, 10.0, 5.0], "fillet_radius_mm": 2.0},
        (-32.0, -12.0, -7.0),
        (32.0, 12.0, 7.0),
        _quaternion((0.0, 0.0, 1.0), math.pi / 6.0),
        (5.0, 6.0, 7.0),
        "圆角半径必须计入局部盒：漏计则局部盒判据先红",
    ),
    (
        "sphere_translation_only",
        {"kind": "sphere", "radius_mm": 7.0},
        (-7.0, -7.0, -7.0),
        (7.0, 7.0, 7.0),
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 2.0, 3.0),
        "单位四元数+非零平移：平移漏加当场红，且旋转无关项不掩盖它",
    ),
    (
        "finite_cylinder_flange_counted",
        {
            "kind": "finite_cylinder",
            "radius_mm": 45.0,
            "half_width_mm": 9.0,
            "flange_outer_radius_mm": 50.0,
        },
        (-50.0, -50.0, -9.0),
        (50.0, 50.0, 9.0),
        _quaternion((2.0, -1.0, 0.5), 1.1),
        (-40.0, 15.0, 0.0),
        "法兰外径必须取代基圆半径进局部盒；扁盒被一般旋转后三轴半边长全变",
    ),
    (
        "capsule_quaternion_order_trap",
        {
            "kind": "capsule",
            "point_a_mm": [0.0, -4.0, 0.0],
            "point_b_mm": [30.0, -4.0, 0.0],
            "radius_mm": 2.5,
        },
        (-2.5, -6.5, -2.5),
        (32.5, -1.5, 2.5),
        (0.6, 0.0, 0.0, 0.8),
        (0.0, 0.0, 0.0),
        "xyzw按wxyz读会把绕x的旋转读成绕z：细长胶囊下两个包盒差出十几毫米",
    ),
)


def main() -> int:
    oracles = []
    for name, shape, local_min, local_max, quaternion, translation, purpose in ORACLES:
        norm = math.sqrt(sum(component * component for component in quaternion))
        if abs(norm - 1.0) > 1.0e-12:
            raise SystemExit(f"{name}: 四元数不是单位阵 {norm}")
        world_min, world_max = _arvo_world_aabb(local_min, local_max, quaternion, translation)
        oracles.append(
            {
                "id": f"oracle:rotated_aabb/{name}",
                "inputs": {
                    "shape": shape,
                    "rotation_xyzw": list(quaternion),
                    "translation_mm": list(translation),
                    "guards_against": purpose,
                },
                "expected": {
                    "local_aabb_min_mm": list(local_min),
                    "local_aabb_max_mm": list(local_max),
                    "world_aabb_min_mm": world_min,
                    "world_aabb_max_mm": world_max,
                },
                "tolerances": {
                    "local_aabb_min_mm": {"abs": 1.0e-12, "rel": 0.0, "reason": _LOCAL_REASON},
                    "local_aabb_max_mm": {"abs": 1.0e-12, "rel": 0.0, "reason": _LOCAL_REASON},
                    "world_aabb_min_mm": {"abs": 1.0e-9, "rel": 0.0, "reason": _WORLD_REASON},
                    "world_aabb_max_mm": {"abs": 1.0e-9, "rel": 0.0, "reason": _WORLD_REASON},
                },
            }
        )
    document = {
        "facet": ORACLE_MANIFEST_FACET,
        "facet_version": ORACLE_MANIFEST_VERSION,
        "case_id": "case/rotated_aabb",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/rotated_aabb/generate_oracle.py",
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
