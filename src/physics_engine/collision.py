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
    return 0.0 if value < 0.0 else (min(value, 1.0))


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
        # 球的局部原点就是球心，姿态不改变它；直接取平移还避免通用四元数路径。
        centre = posed.translation_mm
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


@dataclass(frozen=True)
class CollisionQueryResult:
    """一次查询的事件与**确定性工作量计数**。

    墙钟不进门；候选对数、AABB重叠数和窄相调用数是跨机器稳定的整数，
    可以回答“检测有没有真的裁掉响应工作”，而不把宿主负载当功能结果。
    """

    events: tuple[CollisionEvent, ...]
    candidate_pair_count: int
    broad_phase_overlap_count: int
    narrow_phase_check_count: int


def _overlaps(a: Aabb, b: Aabb) -> bool:
    (al, ah), (bl, bh) = a, b
    return all(al[axis] <= bh[axis] and bl[axis] <= ah[axis] for axis in range(3))


class BroadPhaseCollisionQuery:
    """AABB broad phase；可选候选池按声明次序查询。

    ``candidate_pairs=None``逐字保留原来的全体两两语义。显式候选池用于
    ``FinalizedScene.contact_pairs``：候选身份在装配期冻结，活动与否每步重算。
    """

    def __init__(
        self,
        bodies: tuple[PosedBody, ...],
        *,
        allowed_pairs: frozenset[frozenset[str]] = frozenset(),
        candidate_pairs: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        identifiers = [posed.body.body_id for posed in bodies]
        if len(set(identifiers)) != len(identifiers):
            raise ShapeError("duplicate body_id in collision scene")
        known = set(identifiers)
        for pair in allowed_pairs:
            if len(pair) != 2 or not pair <= known:
                raise ShapeError(f"allowed pair references unknown bodies: {sorted(pair)}")
        self._bodies = bodies
        if candidate_pairs is None:
            self._candidate_pairs = tuple(
                (name_a, name_b)
                for index, name_a in enumerate(identifiers)
                for name_b in identifiers[index + 1 :]
                if frozenset((name_a, name_b)) not in allowed_pairs
            )
        else:
            checked: list[tuple[str, str]] = []
            seen: set[frozenset[str]] = set()
            for pair in candidate_pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ShapeError(
                        f"candidate pair must be a two-item tuple of body ids: {pair!r}"
                    )
                body_a, body_b = pair
                if any(
                    not isinstance(body_id, str) or not body_id
                    for body_id in (body_a, body_b)
                ):
                    raise ShapeError(
                        "candidate pair body ids must be nonempty strings: "
                        f"{pair!r}"
                    )
                if body_a == body_b:
                    raise ShapeError(
                        f"candidate pair must name two distinct bodies: {body_a!r}"
                    )
                members = frozenset((body_a, body_b))
                unknown = sorted(members - known)
                if unknown:
                    raise ShapeError(
                        f"candidate pair references unknown bodies: {unknown}; "
                        f"known bodies are {sorted(known)}"
                    )
                if members in seen:
                    raise ShapeError(
                        f"candidate pair ({body_a!r}, {body_b!r}) is declared twice"
                    )
                if members in allowed_pairs:
                    raise ShapeError(
                        f"pair ({body_a!r}, {body_b!r}) is both a candidate and an "
                        "allowed pair — one asks for a query and the other suppresses it"
                    )
                seen.add(members)
                checked.append((body_a, body_b))
            self._candidate_pairs = tuple(checked)

    @property
    def candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        """真正会被检查的候选池；次序即声明次序。"""

        return self._candidate_pairs

    def check_state(self) -> tuple[CollisionEvent, ...]:
        return self.check_state_with_stats().events

    def check_state_with_stats(self) -> CollisionQueryResult:
        """查询并返回事件以及候选/broad/narrow三段确定性计数。"""

        posed = {body.body.body_id: body for body in self._bodies}
        boxes = {name: body.world_aabb_mm() for name, body in posed.items()}
        segments: dict[str, tuple[Vector3, Vector3, float] | None] = {}
        events: list[CollisionEvent] = []
        broad_phase_overlaps = 0
        narrow_phase_checks = 0
        for name_a, name_b in self._candidate_pairs:
            box_a, box_b = boxes[name_a], boxes[name_b]
            if not _overlaps(box_a, box_b):
                continue
            broad_phase_overlaps += 1
            if name_a not in segments:
                segments[name_a] = _as_world_segment(posed[name_a])
            if name_b not in segments:
                segments[name_b] = _as_world_segment(posed[name_b])
            segment_a = segments[name_a]
            segment_b = segments[name_b]
            if segment_a is not None and segment_b is not None:
                narrow_phase_checks += 1
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
        return CollisionQueryResult(
            events=tuple(events),
            candidate_pair_count=len(self._candidate_pairs),
            broad_phase_overlap_count=broad_phase_overlaps,
            narrow_phase_check_count=narrow_phase_checks,
        )


__all__ = [
    "BroadPhaseCollisionQuery",
    "CollisionEvent",
    "CollisionQueryResult",
    "segment_segment_distance_mm",
]
