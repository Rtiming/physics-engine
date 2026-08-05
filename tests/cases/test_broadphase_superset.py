"""`case/broadphase_superset`的conformance门（轴7规则3）。

守的是碰撞查询对外唯一的硬承诺：**broad phase不许漏报**——
`separation_mm < 0 ⟹ 两个世界AABB相交`，反例数严格为0。

语料从`samples.json`读，读之前先过raw级+语义级双哈希（`load_array`做的）；
分类计数与容差从清单读。测试自己不判"应该是多少"，只负责数数。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.collision import BroadPhaseCollisionQuery, segment_segment_distance_mm
from physics_engine.oracles import load_manifest
from physics_engine.shapes import Capsule, CollisionShape, PosedBody, SimBody, Sphere

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/broadphase_superset/oracle.json", root=ROOT)
CASE = MANIFEST.oracle("oracle:broadphase_superset/sphere_capsule_population")
SAMPLES = MANIFEST.load_array(CASE.inputs["samples_array"], ROOT)
STRIDE = len(SAMPLES["layout"]) // 2


def _posed(body_id: str, fields: list[float]) -> PosedBody:
    kind, radius = fields[0], fields[1]
    if kind:
        shape = Capsule(
            point_a_mm=tuple(fields[2:5]), point_b_mm=tuple(fields[5:8]), radius_mm=radius
        )
    else:
        shape = Sphere(radius_mm=radius)
    return PosedBody(
        SimBody(body_id=body_id, collision=CollisionShape(shape, "envelope")),
        translation_mm=tuple(fields[8:11]),
        rotation_xyzw=tuple(fields[11:15]),
    )


def _world_segment(posed: PosedBody):
    shape = posed.body.collision.shape
    if isinstance(shape, Sphere):
        centre = posed.transform_point_mm((0.0, 0.0, 0.0))
        return centre, centre, shape.radius_mm
    return (
        posed.transform_point_mm(shape.point_a_mm),
        posed.transform_point_mm(shape.point_b_mm),
        shape.radius_mm,
    )


def _overlaps(box_a, box_b) -> bool:
    (a_low, a_high), (b_low, b_high) = box_a, box_b
    return all(a_low[axis] <= b_high[axis] and b_low[axis] <= a_high[axis] for axis in range(3))


@pytest.fixture(scope="module")
def population() -> dict[str, int]:
    counters = {
        "pair_count": 0,
        "negative_separation_pairs": 0,
        "aabb_overlapping_pairs": 0,
        "broad_only_pairs": 0,
        "narrow_phase_events": 0,
        "superset_counterexamples": 0,
    }
    for row in SAMPLES["values"]:
        first = _posed("body/a", row[:STRIDE])
        second = _posed("body/b", row[STRIDE:])
        segment_a, segment_b = _world_segment(first), _world_segment(second)
        separation = segment_segment_distance_mm(
            segment_a[0], segment_a[1], segment_b[0], segment_b[1]
        ) - (segment_a[2] + segment_b[2])
        overlapping = _overlaps(first.world_aabb_mm(), second.world_aabb_mm())
        events = BroadPhaseCollisionQuery((first, second)).check_state()
        counters["pair_count"] += 1
        counters["negative_separation_pairs"] += int(separation < 0.0)
        counters["aabb_overlapping_pairs"] += int(overlapping)
        counters["broad_only_pairs"] += int(overlapping and separation >= 0.0)
        counters["narrow_phase_events"] += sum(
            1 for event in events if event.confidence == "narrow_phase"
        )
        counters["superset_counterexamples"] += int(separation < 0.0 and not overlapping)
    return counters


def test_population_matches_the_frozen_oracle(population):
    CASE.check_all(population)


def test_array_digest_covers_every_sampled_number():
    """语料的双哈希必须真的盖住全部样本，不是盖了个空壳。"""

    digest = MANIFEST.array("samples")
    assert digest.count == len(SAMPLES["values"]) * len(SAMPLES["layout"])
    assert digest.count == CASE.expected["pair_count"] * len(SAMPLES["layout"])
