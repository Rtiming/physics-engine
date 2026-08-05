"""`case/segment_distance`的conformance门（轴7规则3）。

**本文件里没有一个判据数**：期望值与容差全部从`cases/segment_distance/oracle.json`
读，测试只负责把输入喂给生产内核、把算出来的量交给清单比对。原先写死在
`test_narrow_phase.py`里的手算值（5.0/2.0/0.5/3.0/2.0）已按规则3搬进清单。

清单在模块导入期加载并连带校验生成器SHA与重生成留痕——清单被改过、
生成器被改过而没重生成，收集阶段就红，不会拖到某条断言里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.collision import BroadPhaseCollisionQuery, segment_segment_distance_mm
from physics_engine.oracles import load_manifest
from physics_engine.shapes import Capsule, CollisionShape, PosedBody, SimBody, Sphere

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/segment_distance/oracle.json", root=ROOT)

SEGMENT_CASES = [case for case in MANIFEST.oracles if case.inputs["kind"] == "segment_pair"]
BODY_CASES = [case for case in MANIFEST.oracles if case.inputs["kind"] == "body_pair"]


def _shape(declaration: dict):
    if declaration["kind"] == "sphere":
        return Sphere(radius_mm=declaration["radius_mm"])
    return Capsule(
        point_a_mm=tuple(declaration["point_a_mm"]),
        point_b_mm=tuple(declaration["point_b_mm"]),
        radius_mm=declaration["radius_mm"],
    )


def _posed(entry: dict) -> PosedBody:
    return PosedBody(
        SimBody(
            body_id=entry["body_id"],
            collision=CollisionShape(_shape(entry["shape"]), "fitted"),
        ),
        translation_mm=tuple(entry["translation_mm"]),
        rotation_xyzw=tuple(entry["rotation_xyzw"]),
    )


def _world_segment(posed: PosedBody) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """体→世界系线段端点。**只是把输入摆到内核门口**，不参与判据。"""

    shape = posed.body.collision.shape
    if isinstance(shape, Sphere):
        centre = posed.transform_point_mm((0.0, 0.0, 0.0))
        return centre, centre
    return (
        posed.transform_point_mm(shape.point_a_mm),
        posed.transform_point_mm(shape.point_b_mm),
    )


@pytest.mark.parametrize("case", SEGMENT_CASES, ids=[case.id for case in SEGMENT_CASES])
def test_segment_pair_matches_the_frozen_oracle(case):
    distance = segment_segment_distance_mm(
        tuple(case.inputs["p1_mm"]),
        tuple(case.inputs["q1_mm"]),
        tuple(case.inputs["p2_mm"]),
        tuple(case.inputs["q2_mm"]),
    )
    case.check_all({"distance_mm": distance})


@pytest.mark.parametrize("case", BODY_CASES, ids=[case.id for case in BODY_CASES])
def test_body_pair_matches_the_frozen_oracle(case):
    bodies = tuple(_posed(entry) for entry in case.inputs["bodies"])
    events = BroadPhaseCollisionQuery(bodies).check_state()
    measured: dict[str, object] = {"event_count": len(events)}
    if "confidence" in case.expected:
        measured["confidence"] = events[0].confidence
    if "penetration_mm" in case.expected:
        measured["penetration_mm"] = events[0].penetration_mm
    if "segment_distance_mm" in case.expected:
        (a0, a1), (b0, b1) = (_world_segment(body) for body in bodies)
        measured["segment_distance_mm"] = segment_segment_distance_mm(a0, a1, b0, b1)
    case.check_all(measured)


def test_every_degenerate_branch_carries_its_own_oracle():
    """五条退化分支+一条一般路径，一条都不许少（少了就是案例被悄悄削薄）。"""

    branches = {case.inputs["kernel_branch"] for case in SEGMENT_CASES}
    assert len(branches) == len(SEGMENT_CASES) == 7
