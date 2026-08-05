"""形状声明层与broad-phase碰撞的门——spec/11四条红例全在此，附真实参数示例场景。"""

from __future__ import annotations

import pytest

from physics_engine.collision import BroadPhaseCollisionQuery
from physics_engine.shapes import (
    CollisionShape,
    FiniteCylinder,
    GeneratedShape,
    MeshAsset,
    PosedBody,
    RoundedBox,
    ShapeError,
    SimBody,
    Sphere,
)

_LINK3_SHA = "50556eb4e7df6dc27fb2361710fa3419a938ccd8ef6f84ea822d937ab2eb973d"


def _collision_mesh(usage: str = "collision") -> MeshAsset:
    return MeshAsset(
        path_relative="robot-links/collision/link_03_j3.collision.stl",
        sha256=_LINK3_SHA,
        units="mm",
        usage=usage,  # type: ignore[arg-type]
        convexity="nonconvex_declared",
        aabb_min_mm=(-400.0, -300.0, -300.0),
        aabb_max_mm=(400.0, 300.0, 300.0),
    )


# --- spec/11 规则7的四条"必须红" -----------------------------------------


def test_red_1_visual_asset_cannot_enter_collision():
    with pytest.raises(ShapeError, match="visual asset must never enter collision"):
        CollisionShape(shape=_collision_mesh(usage="visual"), direction="fitted")


def test_red_2_asset_without_hash_or_units_is_rejected():
    with pytest.raises(ShapeError, match="sha256"):
        MeshAsset(
            path_relative="x.stl", sha256="short", units="mm", usage="collision",
            convexity="convex_hull", aabb_min_mm=(-1, -1, -1), aabb_max_mm=(1, 1, 1),
        )


def test_red_3_conservativeness_direction_is_mandatory():
    with pytest.raises(TypeError):
        CollisionShape(shape=Sphere(radius_mm=1.0))  # type: ignore[call-arg]


def test_red_4_generator_without_algorithm_identity_is_rejected():
    with pytest.raises(ShapeError, match="algorithm:"):
        GeneratedShape(
            algorithm_id="spool_builder",
            algorithm_version="1.0",
            parameters=(("radius_mm", 50.0),),
            shape=FiniteCylinder(radius_mm=50.0, half_width_mm=9.0),
        )


# --- 结构与装配 -----------------------------------------------------------


def test_collision_shape_is_required_and_visual_defaults_forward_only():
    body = SimBody(
        body_id="body/spool",
        collision=CollisionShape(
            shape=FiniteCylinder(radius_mm=50.0, half_width_mm=10.0),
            direction="fitted",
        ),
    )
    assert body.visual is None  # 外观缺省=复用碰撞形；不存在反向路径


def test_assembly_rejects_duplicate_ids_and_unknown_allowed_pairs():
    spool = PosedBody(
        SimBody(
            body_id="body/spool",
            collision=CollisionShape(
                shape=FiniteCylinder(radius_mm=50.0, half_width_mm=10.0),
                direction="fitted",
            ),
        )
    )
    with pytest.raises(ShapeError, match="duplicate body_id"):
        BroadPhaseCollisionQuery((spool, spool))
    with pytest.raises(ShapeError, match="unknown bodies"):
        BroadPhaseCollisionQuery(
            (spool,), allowed_pairs=frozenset({frozenset({"body/spool", "body/ghost"})})
        )


# --- 真实参数示例场景（WDS碰撞预演的对象集雏形） ---------------------------


def _cell() -> tuple[PosedBody, ...]:
    """R1导轮（槽底R45/法兰R60/半宽9，roller_flange_profile.v1实值）、
    带盘（R50，今晚spool场景实值）、KUKA第三连杆（真SHA网格）、
    圆角盒工装。摆位让带盘与R1的包盒相接，其余分开。
    """

    roller_r1 = PosedBody(
        SimBody(
            body_id="body/roller_r1",
            collision=CollisionShape(
                shape=FiniteCylinder(
                    radius_mm=45.0, half_width_mm=9.0, flange_outer_radius_mm=60.0
                ),
                direction="fitted",
            ),
        ),
        translation_mm=(0.0, 0.0, 0.0),
    )
    spool = PosedBody(
        SimBody(
            body_id="body/spool",
            collision=CollisionShape(
                shape=FiniteCylinder(radius_mm=50.0, half_width_mm=10.0),
                direction="fitted",
            ),
        ),
        translation_mm=(100.0, 0.0, 0.0),  # 60+50=110 > 100：包盒相接
    )
    link3 = PosedBody(
        SimBody(body_id="body/kuka_link3", collision=CollisionShape(_collision_mesh(), "envelope")),
        translation_mm=(1500.0, 0.0, 0.0),
    )
    fixture = PosedBody(
        SimBody(
            body_id="body/fixture",
            collision=CollisionShape(
                RoundedBox(half_extents_mm=(80.0, 80.0, 40.0), fillet_radius_mm=5.0),
                "envelope",
            ),
        ),
        translation_mm=(-500.0, 0.0, 0.0),
    )
    return (roller_r1, spool, link3, fixture)


def test_cell_broad_phase_finds_exactly_the_touching_pair():
    events = BroadPhaseCollisionQuery(_cell()).check_state()
    assert [(e.body_a, e.body_b) for e in events] == [("body/roller_r1", "body/spool")]
    event = events[0]
    assert event.confidence == "broad_phase"
    assert event.penetration_mm is None  # narrow phase之前谁也读不出侵入深度


def test_allowed_pair_suppresses_the_declared_contact():
    allowed = frozenset({frozenset({"body/roller_r1", "body/spool"})})
    events = BroadPhaseCollisionQuery(_cell(), allowed_pairs=allowed).check_state()
    assert events == ()


def test_rotated_body_keeps_a_conservative_world_aabb():
    import math

    half = math.sqrt(0.5)
    rotated = PosedBody(
        SimBody(
            body_id="body/box",
            collision=CollisionShape(
                RoundedBox(half_extents_mm=(10.0, 10.0, 10.0), fillet_radius_mm=0.0),
                "fitted",
            ),
        ),
        rotation_xyzw=(0.0, 0.0, half, half),  # 绕z转90°
    )
    (low, high) = rotated.world_aabb_mm()
    assert low[0] == pytest.approx(-10.0 * math.sqrt(2), rel=1e-9) or low[0] <= -10.0
    assert high[0] >= 10.0
