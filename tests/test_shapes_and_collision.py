"""形状声明层与broad-phase碰撞的门——spec/11四条红例全在此，附真实参数示例场景。"""

from __future__ import annotations

import pytest

from physics_engine.collision import BroadPhaseCollisionQuery, CollisionQueryResult
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


def test_identity_pose_aabb_avoids_the_general_rotation_path(monkeypatch):
    """必须红：单位位姿不得为8个角点重复做通用旋转求和。"""

    import physics_engine.shapes as shapes_module

    posed = PosedBody(
        SimBody(
            body_id="body/identity_sphere",
            collision=CollisionShape(Sphere(radius_mm=2.0), "fitted"),
        ),
        translation_mm=(3.0, -4.0, 5.0),
    )
    builtin_sum = sum
    calls = 0

    def counted_sum(values, start=0):
        nonlocal calls
        calls += 1
        return builtin_sum(values, start)

    monkeypatch.setattr(shapes_module, "sum", counted_sum, raising=False)
    assert posed.world_aabb_mm() == ((1.0, -6.0, 3.0), (5.0, -2.0, 7.0))
    assert calls == 0


def _posed_sphere(body_id: str, x_mm: float) -> PosedBody:
    return PosedBody(
        SimBody(
            body_id=body_id,
            collision=CollisionShape(Sphere(radius_mm=1.0), "fitted"),
        ),
        translation_mm=(x_mm, 0.0, 0.0),
    )


def test_sphere_narrow_phase_uses_translation_as_the_exact_centre(monkeypatch):
    """必须红：球心与姿态无关，不得为零局部点调用通用位姿变换。"""

    bodies = (_posed_sphere("body/a", 0.0), _posed_sphere("body/b", 1.5))

    def forbidden_transform(*_args, **_kwargs):
        raise AssertionError("sphere centre must come directly from translation_mm")

    monkeypatch.setattr(PosedBody, "transform_point_mm", forbidden_transform)
    result = BroadPhaseCollisionQuery(bodies).check_state_with_stats()
    assert result.events[0].penetration_mm == pytest.approx(0.5)


def test_declared_candidates_preserve_order_and_only_those_pairs_are_queried():
    """必红：活跃对必须来自显式候选池，不能偷偷退回全体两两。"""

    bodies = (
        _posed_sphere("body/a", 0.0),
        _posed_sphere("body/b", 0.5),
        _posed_sphere("body/c", 1.0),
    )
    query = BroadPhaseCollisionQuery(
        bodies,
        candidate_pairs=(("body/c", "body/a"), ("body/a", "body/b")),
    )
    result = query.check_state_with_stats()
    assert isinstance(result, CollisionQueryResult)
    assert [(event.body_a, event.body_b) for event in result.events] == [
        ("body/c", "body/a"),
        ("body/a", "body/b"),
    ]
    assert result.candidate_pair_count == 2
    assert result.broad_phase_overlap_count == 2
    assert result.narrow_phase_check_count == 2


def test_narrow_geometry_is_prepared_once_per_body(monkeypatch):
    """必须红：一个体参与多对候选时，世界系窄相几何每帧只准备一次。"""

    import physics_engine.collision as collision_module

    bodies = (
        _posed_sphere("body/a", 0.0),
        _posed_sphere("body/b", 0.5),
        _posed_sphere("body/c", 1.0),
    )
    original = collision_module._as_world_segment
    calls: list[str] = []

    def counted(posed):
        calls.append(posed.body.body_id)
        return original(posed)

    monkeypatch.setattr(collision_module, "_as_world_segment", counted)
    result = BroadPhaseCollisionQuery(bodies).check_state_with_stats()
    assert len(result.events) == 3
    assert calls == ["body/a", "body/b", "body/c"]


def test_broad_phase_culled_bodies_do_not_prepare_narrow_geometry(monkeypatch):
    """必须红：没有AABB重叠时，稀疏场景不得白做窄相几何准备。"""

    import physics_engine.collision as collision_module

    bodies = (_posed_sphere("body/a", 0.0), _posed_sphere("body/b", 10.0))

    def forbidden(_posed):
        raise AssertionError("broad-phase miss must not prepare narrow geometry")

    monkeypatch.setattr(collision_module, "_as_world_segment", forbidden)
    result = BroadPhaseCollisionQuery(bodies).check_state_with_stats()
    assert result.events == ()
    assert result.broad_phase_overlap_count == 0
    assert result.narrow_phase_check_count == 0


def test_every_overlapping_declared_sphere_pair_reaches_the_narrow_phase():
    """必红：声明池里的真重叠对漏掉一个，动态响应就会静默少一股力。"""

    bodies = (
        _posed_sphere("body/a", 0.0),
        _posed_sphere("body/b", 1.5),
        _posed_sphere("body/c", 8.0),
    )
    result = BroadPhaseCollisionQuery(
        bodies,
        candidate_pairs=(("body/a", "body/b"), ("body/b", "body/c")),
    ).check_state_with_stats()
    assert [(event.body_a, event.body_b) for event in result.events] == [
        ("body/a", "body/b")
    ]
    assert result.candidate_pair_count == 2
    assert result.broad_phase_overlap_count == 1
    assert result.narrow_phase_check_count == 1
    assert result.events[0].penetration_mm == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("candidate_pairs", "allowed_pairs", "message"),
    [
        (((1, "body/ghost"),), frozenset(), "nonempty string"),
        ((("body/a", "body/ghost"),), frozenset(), "unknown bodies"),
        ((("body/a", "body/a"),), frozenset(), "distinct bodies"),
        (
            (("body/a", "body/b"), ("body/b", "body/a")),
            frozenset(),
            "declared twice",
        ),
        (
            (("body/a", "body/b"),),
            frozenset({frozenset(("body/a", "body/b"))}),
            "both a candidate and an allowed pair",
        ),
    ],
)
def test_invalid_candidate_pools_fail_closed(candidate_pairs, allowed_pairs, message):
    """五条构造期必红：坏ID、未知、自对、反序重复、与allowed矛盾。"""

    bodies = (_posed_sphere("body/a", 0.0), _posed_sphere("body/b", 1.0))
    with pytest.raises(ShapeError, match=message):
        BroadPhaseCollisionQuery(
            bodies,
            candidate_pairs=candidate_pairs,
            allowed_pairs=allowed_pairs,
        )
