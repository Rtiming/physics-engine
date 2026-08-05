"""碰撞查询——spec/10 `CollisionQuery`的首个实现：**只到broad phase**。

事件带可信度是接口的承重字段：broad phase命中≠真的撞，把两者混报
就是消费方一整天在防的那个形状（WDS design/24 §6.4原文）。本实现
所有事件`confidence="broad_phase"`、`penetration_mm=None`——narrow
phase落地前，谁也不许从这里读出"侵入多深"。

装配期校验（PyElastica形制，spec/10场景装配第3条）：构造时逐体校验、
白名单按对声明（WII相邻连杆忽略的同型），配错当场炸，不拖到查询时。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physics_engine.shapes import Aabb, PosedBody, ShapeError


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
            for name_b, box_b in boxes[index + 1 :]:
                if frozenset((name_a, name_b)) in self._allowed:
                    continue
                if _overlaps(box_a, box_b):
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


__all__ = ["BroadPhaseCollisionQuery", "CollisionEvent"]
