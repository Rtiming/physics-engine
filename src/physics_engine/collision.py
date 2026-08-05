"""碰撞查询——spec/10 `CollisionQuery`：broad phase + narrow phase第一片。

事件带可信度是接口的承重字段：broad phase命中≠真的撞。narrow phase
第一片只覆盖**球/胶囊族**（含GeneratedShape包裹的同族）：线段-线段最近
距离是闭式的、手算可验。这一族的对给`confidence="narrow_phase"`与精确
`penetration_mm`，且**broad命中但narrow判分离的对不再报事件**（假阳性
消除）。其余形状对诚实保留`confidence="broad_phase"`、`penetration_mm=None`
——圆柱/盒/网格的narrow phase等下一片，不冒充。

装配期校验（PyElastica形制，spec/10场景装配第3条）：构造时逐体校验、
白名单按对声明（WII相邻连杆忽略的同型），配错当场炸，不拖到查询时。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physics_engine.shapes import (
    Aabb,
    Capsule,
    GeneratedShape,
    PosedBody,
    ShapeError,
    Sphere,
    Vector3,
)


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def segment_segment_distance_mm(
    p1: Vector3, q1: Vector3, p2: Vector3, q2: Vector3
) -> float:
    """两线段最近距离（Ericson《Real-Time Collision Detection》标准算法）。"""

    d1, d2, r = _sub(q1, p1), _sub(q2, p2), _sub(p1, p2)
    a, e, f = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    if a <= 1e-12 and e <= 1e-12:
        return _dot(r, r) ** 0.5
    if a <= 1e-12:
        s, t = 0.0, _clamp(f / e)
    else:
        c = _dot(d1, r)
        if e <= 1e-12:
            t, s = 0.0, _clamp(-c / a)
        else:
            b = _dot(d1, d2)
            denominator = a * e - b * b
            s = _clamp((b * f - c * e) / denominator) if denominator > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clamp((b - c) / a)
    closest1 = (p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s)
    closest2 = (p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t)
    gap = _sub(closest1, closest2)
    return _dot(gap, gap) ** 0.5


def _as_world_segment(posed: PosedBody) -> tuple[Vector3, Vector3, float] | None:
    """球/胶囊族→世界系(端点a, 端点b, 半径)；其余族返回None。"""

    shape = posed.body.collision.shape
    if isinstance(shape, GeneratedShape):
        shape = shape.shape
    if isinstance(shape, Sphere):
        centre = posed.transform_point_mm((0.0, 0.0, 0.0))
        return (centre, centre, shape.radius_mm)
    if isinstance(shape, Capsule):
        return (
            posed.transform_point_mm(shape.point_a_mm),
            posed.transform_point_mm(shape.point_b_mm),
            shape.radius_mm,
        )
    return None


@dataclass(frozen=True)
class CollisionEvent:
    body_a: str
    body_b: str
    confidence: Literal["broad_phase", "narrow_phase"]
    penetration_mm: float | None
    aabb_a_mm: Aabb
    aabb_b_mm: Aabb


def _overlaps(a: Aabb, b: Aabb) -> bool:
    (al, ah), (bl, bh) = a, b
    return all(al[axis] <= bh[axis] and bl[axis] <= ah[axis] for axis in range(3))


class BroadPhaseCollisionQuery:
    """AABB两两相交的broad phase。O(n²)，本估算的场景是几十个体，够用。"""

    def __init__(
        self,
        bodies: tuple[PosedBody, ...],
        *,
        allowed_pairs: frozenset[frozenset[str]] = frozenset(),
    ) -> None:
        identifiers = [posed.body.body_id for posed in bodies]
        if len(set(identifiers)) != len(identifiers):
            raise ShapeError("duplicate body_id in collision scene")
        for pair in allowed_pairs:
            if len(pair) != 2 or not pair <= set(identifiers):
                raise ShapeError(f"allowed pair references unknown bodies: {sorted(pair)}")
        self._bodies = bodies
        self._allowed = allowed_pairs

    def check_state(self) -> tuple[CollisionEvent, ...]:
        boxes = [(posed.body.body_id, posed.world_aabb_mm()) for posed in self._bodies]
        events = []
        for index, (name_a, box_a) in enumerate(boxes):
            for offset, (name_b, box_b) in enumerate(boxes[index + 1 :]):
                if frozenset((name_a, name_b)) in self._allowed:
                    continue
                if not _overlaps(box_a, box_b):
                    continue
                segment_a = _as_world_segment(self._bodies[index])
                segment_b = _as_world_segment(self._bodies[index + 1 + offset])
                if segment_a is not None and segment_b is not None:
                    distance = segment_segment_distance_mm(
                        segment_a[0], segment_a[1], segment_b[0], segment_b[1]
                    )
                    separation = distance - (segment_a[2] + segment_b[2])
                    if separation >= 0.0:
                        continue  # broad命中但narrow判分离：假阳性，不报
                    events.append(
                        CollisionEvent(
                            body_a=name_a,
                            body_b=name_b,
                            confidence="narrow_phase",
                            penetration_mm=-separation,
                            aabb_a_mm=box_a,
                            aabb_b_mm=box_b,
                        )
                    )
                else:
                    events.append(
                        CollisionEvent(
                            body_a=name_a,
                            body_b=name_b,
                            confidence="broad_phase",
                            penetration_mm=None,
                            aabb_a_mm=box_a,
                            aabb_b_mm=box_b,
                        )
                    )
        return tuple(events)


__all__ = ["BroadPhaseCollisionQuery", "CollisionEvent", "segment_segment_distance_mm"]
