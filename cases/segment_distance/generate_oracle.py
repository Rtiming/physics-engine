#!/usr/bin/env python3
"""生成`cases/segment_distance/oracle.json`——球/胶囊解析距离与线段退化分支。

金标来源（轴7规则1）：**手推闭式解**，每条的推导写在下面的注释与`case.md`里，
值全部是可精确表示的有理数/整数（5、13、7、3、5、10、2），不是"跑一遍记下来"。

独立性（轴7规则4）：脚本内的`_independent_distance`是**凸一维黄金分割搜索**
——把"两线段最近距离"化成"点到线段距离的一维凸最小化"，与被验内核
（Ericson的钳制-重算闭式）不是同一条路径。它不产生金标，只在写盘前
交叉验证手推值，防的是手算笔误。

用法：`PYTHONPATH=src .venv/bin/python cases/segment_distance/generate_oracle.py`
改了本脚本就必须重跑——清单里钉着本脚本的SHA-256，不重跑读侧当场红。
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

ALGORITHM_ID = "algorithm:oracle/segment_distance"
ALGORITHM_VERSION = "1.0.0"

#: 交叉验证用的容差：黄金分割搜索的距离值收敛到~1e-13，取1e-9留三个量级余量。
CROSSCHECK_ABS = 1.0e-9


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _point_segment_distance_sq(point, p, q):
    direction = _sub(q, p)
    length_sq = _dot(direction, direction)
    if length_sq == 0.0:
        offset = _sub(point, p)
        return _dot(offset, offset)
    t = _dot(_sub(point, p), direction) / length_sq
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    foot = tuple(p[i] + direction[i] * t for i in range(3))
    offset = _sub(point, foot)
    return _dot(offset, offset)


def _independent_distance(p1, q1, p2, q2) -> float:
    """独立实现：g(s)=点(p1+s·d1)到线段2的距离²，在[0,1]上凸，黄金分割求极小。"""

    d1 = _sub(q1, p1)

    def g(s: float) -> float:
        point = tuple(p1[i] + d1[i] * s for i in range(3))
        return _point_segment_distance_sq(point, p2, q2)

    lo, hi = 0.0, 1.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        m1 = hi - ratio * (hi - lo)
        m2 = lo + ratio * (hi - lo)
        if g(m1) <= g(m2):
            hi = m2
        else:
            lo = m1
    return math.sqrt(min(g(lo), g(hi), g(0.5 * (lo + hi))))


def _quaternion_rows(q):
    """四元数(xyzw)→旋转矩阵行。生成器自带一份，不借用被验内核的那份。"""

    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _world_point(point, translation, rotation):
    rows = _quaternion_rows(rotation)
    return tuple(_dot(rows[axis], point) + translation[axis] for axis in range(3))


def _tolerance(abs_tol: float, rel_tol: float, reason: str) -> dict:
    return {"abs": abs_tol, "rel": rel_tol, "reason": reason}


_DISTANCE_REASON = (
    "闭式解与被验内核都只做加减乘与一次开方，量级10mm下双精度舍入~1e-15mm；"
    "abs=1e-12mm留三个量级余量。**不写==0**：逐位相等会把求和次序也冻结成契约，"
    "那是实现细节不是物理。rel=0是因为判据是绝对几何间隙，不随量级缩放。"
)

#: 线段-线段的七条路径。`branch`字段记的是被验内核`collision.py`里走到的那一条，
#: 由一次line-trace实测确认（见case.md第四节），不是推测。
SEGMENT_ORACLES = (
    # a≤1e-12 且 e≤1e-12：两段都退化为点，直接返回|r|。r=(0,0,0)−(3,4,0)，|r|=5。
    (
        "branch_point_point",
        "两段都退化为点（a≤1e-12 且 e≤1e-12）",
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 0.0)),
        5.0,
    ),
    # a≤1e-12 且 e>1e-12：仅段1退化为点，走 s,t = 0, clamp(f/e)。
    # 点(0,12,5)到x轴线段：f=d2·r=20·10=200、e=400 → t=0.5 → 最近点(0,0,0)；
    # 距离=√(12²+5²)=13。
    (
        "branch_segment1_degenerate",
        "仅段1退化为点（a≤1e-12 且 e>1e-12）",
        ((0.0, 12.0, 5.0), (0.0, 12.0, 5.0), (-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        13.0,
    ),
    # e≤1e-12 且 a>1e-12：仅段2退化为点，走 t,s = 0, clamp(−c/a) 这条独立路径。
    # c=d1·r=20·(−10)=−200、a=400 → s=0.5 → 最近点(0,0,0)；到(0,0,7)距离=7。
    (
        "branch_segment2_degenerate",
        "仅段2退化为点（e≤1e-12 且 a>1e-12）",
        ((-10.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 7.0), (0.0, 0.0, 7.0)),
        7.0,
    ),
    # denominator=ae−b²≤1e-12：平行。a=e=b=100 → denominator=0 → s=0；
    # f=d2·r=0 → t=0，最近点对(0,0,0)与(0,3,0)，距离=3。
    (
        "branch_parallel",
        "两段平行（denominator=ae−b²≤1e-12）",
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 3.0, 0.0), (10.0, 3.0, 0.0)),
        3.0,
    ),
    # t<0钳制：b=0、f=d2·r=8·(−3)=−24、e=64 → t=−0.375<0 → t=0、s=clamp(−c/a)=0.5，
    # 最近点对(5,0,0)与(5,3,4)，距离=√(3²+4²)=5。
    (
        "branch_t_below_zero",
        "t<0被钳制到0（denominator>1e-12）",
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 3.0, 4.0), (5.0, 11.0, 4.0)),
        5.0,
    ),
    # t>1钳制：f=d2·r=8·16=128、e=64 → t=2>1 → t=1、s=clamp((b−c)/a)=0.5，
    # 最近点对(5,0,0)与(5,−8,6)，距离=√(8²+6²)=10。
    (
        "branch_t_above_one",
        "t>1被钳制到1（denominator>1e-12）",
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, -16.0, 6.0), (5.0, -8.0, 6.0)),
        10.0,
    ),
    # 无退化的一般路径（s、t都落在开区间内）：两段正交交错，z向间隔2。
    (
        "interior_crossing",
        "无退化的一般路径（s、t都不被钳制）",
        ((-5.0, 0.0, 1.0), (5.0, 0.0, 1.0), (0.0, -5.0, -1.0), (0.0, 5.0, -1.0)),
        2.0,
    ),
)


def _segment_entry(name, branch, points, distance_mm):
    p1, q1, p2, q2 = points
    measured = _independent_distance(p1, q1, p2, q2)
    if abs(measured - distance_mm) > CROSSCHECK_ABS:
        raise SystemExit(f"{name}: 手推值{distance_mm}与独立实现{measured}不符")
    return {
        "id": f"oracle:segment_distance/{name}",
        "inputs": {
            "kind": "segment_pair",
            "kernel_branch": branch,
            "p1_mm": list(p1),
            "q1_mm": list(q1),
            "p2_mm": list(p2),
            "q2_mm": list(q2),
        },
        "expected": {"distance_mm": distance_mm},
        "tolerances": {"distance_mm": _tolerance(1.0e-12, 0.0, _DISTANCE_REASON)},
    }


def _sphere(radius):
    return {"kind": "sphere", "radius_mm": radius}


def _capsule(a, b, radius):
    return {"kind": "capsule", "point_a_mm": list(a), "point_b_mm": list(b), "radius_mm": radius}


def _body(body_id, shape, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return {
        "body_id": body_id,
        "shape": shape,
        "translation_mm": list(translation),
        "rotation_xyzw": list(rotation),
    }


_HALF = math.sqrt(0.5)

#: 体对的四条：`d = segdist − (r1+r2)`、`penetration_mm = −d`。
#: 手推值——两球心距15、半径和20 → 侵入5；球心到胶囊轴距5、半径和7 → 侵入2；
#: 胶囊绕z转90°后轴沿y，球心到轴距2.5、半径和3 → 侵入0.5；
#: 第四条是broad假阳性：AABB各轴重叠而球心距8√2≈11.31>半径和10 → 不报事件。
BODY_ORACLES = (
    (
        "two_spheres_overlap",
        (_body("body/a", _sphere(10.0)), _body("body/b", _sphere(10.0), (15.0, 0.0, 0.0))),
        {"event_count": 1, "confidence": "narrow_phase", "penetration_mm": 5.0},
    ),
    (
        "sphere_capsule",
        (
            _body("body/cap", _capsule((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), 3.0)),
            _body("body/sph", _sphere(4.0), (10.0, 5.0, 0.0)),
        ),
        {"event_count": 1, "confidence": "narrow_phase", "penetration_mm": 2.0},
    ),
    (
        "rotated_capsule",
        (
            _body(
                "body/cap",
                _capsule((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), 2.0),
                rotation=(0.0, 0.0, _HALF, _HALF),
            ),
            _body("body/sph", _sphere(1.0), (0.0, 10.0, 2.5)),
        ),
        {"event_count": 1, "confidence": "narrow_phase", "penetration_mm": 0.5},
    ),
    (
        "broad_hit_narrow_clear",
        (_body("body/a", _sphere(5.0)), _body("body/b", _sphere(5.0), (8.0, 8.0, 0.0))),
        {"event_count": 0, "segment_distance_mm": math.sqrt(128.0)},
    ),
)


def _segment_of(body):
    """体→世界系(端点a, 端点b, 半径)。生成器自带，不走被验内核。"""

    shape = body["shape"]
    translation = body["translation_mm"]
    rotation = body["rotation_xyzw"]
    if shape["kind"] == "sphere":
        centre = _world_point((0.0, 0.0, 0.0), translation, rotation)
        return centre, centre, shape["radius_mm"]
    return (
        _world_point(shape["point_a_mm"], translation, rotation),
        _world_point(shape["point_b_mm"], translation, rotation),
        shape["radius_mm"],
    )


def _body_entry(name, bodies, expected):
    a, b = (_segment_of(body) for body in bodies)
    distance = _independent_distance(a[0], a[1], b[0], b[1])
    separation = distance - (a[2] + b[2])
    if "penetration_mm" in expected:
        if abs(-separation - expected["penetration_mm"]) > CROSSCHECK_ABS:
            raise SystemExit(f"{name}: 手推侵入值与独立实现{-separation}不符")
    if "segment_distance_mm" in expected:
        if abs(distance - expected["segment_distance_mm"]) > CROSSCHECK_ABS:
            raise SystemExit(f"{name}: 手推距离与独立实现{distance}不符")
        if separation <= 0.0:
            raise SystemExit(f"{name}: 该条应当是broad假阳性，独立实现却判相交")
    tolerances = {
        "event_count": _tolerance(0.0, 0.0, "事件条数是确定性整数，零容差；多报漏报都是行为回退。"),
        "confidence": _tolerance(0.0, 0.0, "可信度是枚举字符串，逐位相等；降级冒充narrow是诚实性缺陷。"),
        "penetration_mm": _tolerance(1.0e-12, 0.0, _DISTANCE_REASON),
        "segment_distance_mm": _tolerance(1.0e-12, 0.0, _DISTANCE_REASON),
    }
    return {
        "id": f"oracle:segment_distance/{name}",
        "inputs": {"kind": "body_pair", "bodies": list(bodies)},
        "expected": expected,
        "tolerances": {key: tolerances[key] for key in expected},
    }


def main() -> int:
    oracles = [_segment_entry(*entry) for entry in SEGMENT_ORACLES]
    oracles.extend(_body_entry(*entry) for entry in BODY_ORACLES)
    document = {
        "facet": ORACLE_MANIFEST_FACET,
        "facet_version": ORACLE_MANIFEST_VERSION,
        "case_id": "case/segment_distance",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/segment_distance/generate_oracle.py",
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
