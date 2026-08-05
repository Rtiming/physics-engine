#!/usr/bin/env python3
"""生成`cases/broadphase_superset/`的语料与清单——broad ⊇ narrow超集不变量。

判据是**可证命题**不是拟合数：世界AABB是形状的保守外包，所以

    separation_mm < 0  ⟹  两个世界AABB相交

反例数必须严格为0。换SAP/BVH那天，这是唯一不能破的承诺。

**只对球/胶囊族成立**——不是因为AABB对别的族不保守（它对所有族都保守），
而是因为别的族本仓根本算不出`separation_mm`（narrow phase第一片只覆盖
球/胶囊）。含finite_cylinder/rounded_box/mesh的对只有broad结论，进不了这条判据。
另见`case.md`第六节：世界AABB**不是**SE(3)不变量，"整场景旋转后事件集合相同"
那种判据是错的，本案例不写。

独立性（轴7规则4）：语料的分类计数由本脚本自带的Arvo包盒与凸一维搜索距离
算出，不import`shapes.py`/`collision.py`。

判定边界的稳健性：采样时拒绝任何"离判定边界太近"的配置
（|separation| ≤ 1e-6mm 或 任一轴的AABB间隙 ≤ 1e-6mm），
这样冻结的计数不会因为两条实现路径~1e-14的浮点差异而翻面。

用法：`PYTHONPATH=src .venv/bin/python cases/broadphase_superset/generate_oracle.py`
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.canonical import canonical_file_bytes  # noqa: E402
from physics_engine.oracles import (  # noqa: E402
    MANIFEST_PROFILE,
    ORACLE_MANIFEST_FACET,
    ORACLE_MANIFEST_VERSION,
    array_logical_sha256,
    file_sha256,
    flatten_values,
    write_manifest,
)

ALGORITHM_ID = "algorithm:oracle/broadphase_superset"
ALGORITHM_VERSION = "1.0.0"

#: 固定种子（清单里也记一份）。CPython的`random.Random`是MT19937，
#: 同种子跨平台跨版本同序列——语料因此可重生成、可对拍。
SEED = 20260805
PAIR_COUNT = 120

#: 判定边界排斥半径：比两条实现路径的浮点差（~1e-14mm）大八个量级。
BOUNDARY_MARGIN_MM = 1.0e-6

#: 每个体15个数：kind(0=球,1=胶囊)、半径、局部端点a、局部端点b、平移、四元数。
BODY_LAYOUT = (
    "kind", "radius_mm",
    "ax_mm", "ay_mm", "az_mm", "bx_mm", "by_mm", "bz_mm",
    "tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw",
)
LAYOUT = tuple(f"a.{name}" for name in BODY_LAYOUT) + tuple(f"b.{name}" for name in BODY_LAYOUT)


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
    """凸一维黄金分割搜索（与`cases/segment_distance/`同一独立实现）。"""

    d1 = _sub(q1, p1)

    def g(s: float) -> float:
        point = tuple(p1[i] + d1[i] * s for i in range(3))
        return _point_segment_distance_sq(point, p2, q2)

    lo, hi = 0.0, 1.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(160):
        m1 = hi - ratio * (hi - lo)
        m2 = lo + ratio * (hi - lo)
        if g(m1) <= g(m2):
            hi = m2
        else:
            lo = m1
    return math.sqrt(min(g(lo), g(hi), g(0.5 * (lo + hi))))


def _rows(quaternion):
    x, y, z, w = quaternion
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _world_point(point, translation, quaternion):
    rows = _rows(quaternion)
    return tuple(_dot(rows[axis], point) + translation[axis] for axis in range(3))


def _world_aabb(local_min, local_max, translation, quaternion):
    """Arvo中心-半边长分解（与`cases/rotated_aabb/`同一独立实现）。"""

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


def _body_geometry(body):
    """一行15个数→(局部盒, 世界端点, 半径)。球的两端点重合。"""

    kind, radius = body[0], body[1]
    point_a = tuple(body[2:5]) if kind else (0.0, 0.0, 0.0)
    point_b = tuple(body[5:8]) if kind else (0.0, 0.0, 0.0)
    translation, quaternion = tuple(body[8:11]), tuple(body[11:15])
    local_min = tuple(min(point_a[i], point_b[i]) - radius for i in range(3))
    local_max = tuple(max(point_a[i], point_b[i]) + radius for i in range(3))
    return (
        _world_aabb(local_min, local_max, translation, quaternion),
        (
            _world_point(point_a, translation, quaternion),
            _world_point(point_b, translation, quaternion),
        ),
        radius,
    )


def _random_unit_quaternion(rng):
    while True:
        raw = [rng.gauss(0.0, 1.0) for _ in range(4)]
        norm = math.sqrt(sum(component * component for component in raw))
        if norm > 1.0e-6:
            return [component / norm for component in raw]


def _random_body(rng, centre_span):
    kind = float(rng.randint(0, 1))
    radius = round(rng.uniform(2.0, 12.0), 6)
    if kind:
        point_a = [round(rng.uniform(-15.0, 15.0), 6) for _ in range(3)]
        point_b = [round(rng.uniform(-15.0, 15.0), 6) for _ in range(3)]
    else:
        point_a = [0.0, 0.0, 0.0]
        point_b = [0.0, 0.0, 0.0]
    translation = [round(rng.uniform(-centre_span, centre_span), 6) for _ in range(3)]
    return [kind, radius, *point_a, *point_b, *translation, *_random_unit_quaternion(rng)]


def _axis_margins(box_a, box_b):
    (a_low, a_high), (b_low, b_high) = box_a, box_b
    return [
        margin
        for axis in range(3)
        for margin in (b_high[axis] - a_low[axis], a_high[axis] - b_low[axis])
    ]


def _classify(row):
    box_a, segment_a, radius_a = _body_geometry(row[: len(BODY_LAYOUT)])
    box_b, segment_b, radius_b = _body_geometry(row[len(BODY_LAYOUT) :])
    distance = _independent_distance(segment_a[0], segment_a[1], segment_b[0], segment_b[1])
    separation = distance - (radius_a + radius_b)
    margins = _axis_margins(box_a, box_b)
    overlaps = all(margin >= 0.0 for margin in margins)
    return separation, overlaps, min(abs(margin) for margin in margins)


def _sample_rows():
    rng = random.Random(SEED)
    rows, rejected = [], 0
    while len(rows) < PAIR_COUNT:
        row = _random_body(rng, 6.0) + _random_body(rng, 22.0)
        separation, _overlaps, margin = _classify(row)
        if abs(separation) <= BOUNDARY_MARGIN_MM or margin <= BOUNDARY_MARGIN_MM:
            rejected += 1
            continue
        rows.append(row)
    return rows, rejected


def main() -> int:
    rows, rejected = _sample_rows()
    negative, overlapping, counterexamples = 0, 0, 0
    for row in rows:
        separation, overlaps, _margin = _classify(row)
        negative += int(separation < 0.0)
        overlapping += int(overlaps)
        counterexamples += int(separation < 0.0 and not overlaps)
    if counterexamples:
        raise SystemExit(f"独立实现自己就找到了{counterexamples}个反例——先查独立实现")
    broad_only = overlapping - negative
    if broad_only <= 0:
        raise SystemExit("语料里没有一个broad假阳性，超集判据会退化成平凡命题")

    samples = {"dtype": "float64", "layout": list(LAYOUT), "values": rows}
    raw = canonical_file_bytes(samples, MANIFEST_PROFILE)
    (HERE / "samples.json").write_bytes(raw)
    flat = flatten_values(rows)

    reason_integer = (
        "分类计数是确定性整数，零容差。采样时已拒绝一切离判定边界1e-6mm以内的配置，"
        "所以两条实现路径~1e-14mm的浮点差不可能让任何一对翻面——计数因此可以钉死。"
    )
    document = {
        "facet": ORACLE_MANIFEST_FACET,
        "facet_version": ORACLE_MANIFEST_VERSION,
        "case_id": "case/broadphase_superset",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/broadphase_superset/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:broadphase_superset/sphere_capsule_population",
                "inputs": {
                    "seed": SEED,
                    "rng": "python:random.Random(MT19937)",
                    "pair_count": PAIR_COUNT,
                    "rejected_near_boundary": rejected,
                    "boundary_margin_mm": BOUNDARY_MARGIN_MM,
                    "samples_array": "samples",
                    "families": ["sphere", "capsule"],
                },
                "expected": {
                    "pair_count": PAIR_COUNT,
                    "negative_separation_pairs": negative,
                    "aabb_overlapping_pairs": overlapping,
                    "broad_only_pairs": broad_only,
                    "narrow_phase_events": negative,
                    "superset_counterexamples": 0,
                },
                "tolerances": {
                    "pair_count": {"abs": 0.0, "rel": 0.0, "reason": reason_integer},
                    "negative_separation_pairs": {"abs": 0.0, "rel": 0.0, "reason": reason_integer},
                    "aabb_overlapping_pairs": {"abs": 0.0, "rel": 0.0, "reason": reason_integer},
                    "broad_only_pairs": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": (
                            "必须>0才证明语料里真有broad假阳性；否则超集判据平凡成立，"
                            "门看着绿其实什么都没守。零容差同上。"
                        ),
                    },
                    "narrow_phase_events": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": (
                            "查询报出的事件条数必须恰好等于separation<0的对数——"
                            "多一条是假阳性没吃干净，少一条是漏报。零容差同上。"
                        ),
                    },
                    "superset_counterexamples": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": (
                            "这是可证命题（AABB是保守外包）不是拟合数，反例数只能是0；"
                            "任何非零都意味着broad漏报，而漏报是碰撞查询对外唯一的硬承诺。"
                        ),
                    },
                },
            }
        ],
        "arrays": {
            "samples": {
                "path_relative": "cases/broadphase_superset/samples.json",
                "dtype": "float64",
                "count": len(flat),
                "logical_sha256": array_logical_sha256(flat, dtype="float64"),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"pairs={PAIR_COUNT} rejected={rejected} negative={negative} "
        f"overlapping={overlapping} broad_only={broad_only} "
        f"samples={len(raw)}B manifest={len(written)}B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
