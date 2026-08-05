"""装配期失败关闭的门（plans/02 T0b补齐的四条缺口）。

每条都是"必须红"用例（AGENTS.md代码三前提之外的仓内纪律）：这些输入在
0.4.0上全部被**放行**——重复body_id让`validate`报valid并以0退出、
`check-collisions`抛栈回溯以1退出（而1的语义是"有候选"）；缺`direction`
与`direction="banana"`都能加载成功；`allowed_pairs`可以点名不存在的体。
本文件锁住修复：非法场景两个子命令一律退出码2，且失败发生在**加载期**
（spec/10第3条：finalize统一校验、配错当场炸），不拖到查询构造期。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physics_engine import cli
from physics_engine.scene import SceneError, load_scene
from physics_engine.shapes import CollisionShape, MeshAsset, ShapeError, Sphere

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/collision_preview_cell.scene.json"

_SPHERE = {"kind": "sphere", "radius_mm": 10.0}


def _scene(bodies: list[dict], allowed_pairs: list | None = None) -> bytes:
    return json.dumps(
        {
            "contract_type": "physics_scene",
            "contract_version": "1.0.0",
            "scene_id": "scene/red_example",
            "extensions": [],
            "bodies": bodies,
            "allowed_pairs": allowed_pairs or [],
        }
    ).encode("utf-8")


def _body(body_id: str, collision: dict | None = None) -> dict:
    return {
        "body_id": body_id,
        "collision": collision or {"direction": "fitted", "shape": dict(_SPHERE)},
    }


# ── 场景加载期的四条门 ────────────────────────────────────────────────


def test_duplicate_body_id_fails_at_load_time():
    payload = _scene([_body("body/a"), _body("body/a")])
    with pytest.raises(SceneError, match="duplicate body_id"):
        load_scene(payload)


def test_allowed_pair_naming_an_unknown_body_fails_closed():
    payload = _scene([_body("body/a"), _body("body/b")], [["body/a", "body/nope"]])
    with pytest.raises(SceneError, match="unknown bodies"):
        load_scene(payload)


def test_missing_collision_direction_fails_closed():
    """spec/11规则5：保守方向缺省禁止——对写场景文件的人同样成立。"""

    payload = _scene([_body("body/a", {"shape": dict(_SPHERE)})])
    with pytest.raises(SceneError, match="collision direction"):
        load_scene(payload)


def test_illegal_collision_direction_value_fails_closed():
    payload = _scene([_body("body/a", {"direction": "banana", "shape": dict(_SPHERE)})])
    with pytest.raises(SceneError, match="collision direction"):
        load_scene(payload)


# ── 形状层的Literal取值在运行时校验 ──────────────────────────────────

_MESH = {
    "path_relative": "a/b.stl",
    "sha256": "a" * 64,
    "units": "mm",
    "usage": "collision",
    "convexity": "nonconvex_declared",
    "aabb_min_mm": (-1.0, -1.0, -1.0),
    "aabb_max_mm": (1.0, 1.0, 1.0),
}


@pytest.mark.parametrize(
    ("field", "value"),
    [("units", "furlong"), ("usage", "decoration"), ("convexity", "banana")],
)
def test_mesh_asset_rejects_illegal_literal_values(field, value):
    fields = dict(_MESH) | {field: value}
    with pytest.raises(ShapeError, match=field):
        MeshAsset(**fields)


def test_collision_shape_rejects_absent_direction():
    with pytest.raises(ShapeError, match="collision direction"):
        CollisionShape(shape=Sphere(radius_mm=1.0), direction=None)


# ── CLI退出码：非法场景一律2，两个子命令口径一致 ────────────────────


@pytest.mark.parametrize(
    "bodies_and_pairs",
    [
        ([_body("body/a"), _body("body/a")], []),
        ([_body("body/a"), _body("body/b")], [["body/a", "body/nope"]]),
        ([_body("body/a", {"shape": dict(_SPHERE)})], []),
        ([_body("body/a", {"direction": "banana", "shape": dict(_SPHERE)})], []),
    ],
)
def test_cli_exits_two_on_every_illegal_scene(tmp_path, bodies_and_pairs):
    bodies, pairs = bodies_and_pairs
    path = tmp_path / "red.scene.json"
    path.write_bytes(_scene(bodies, pairs))
    assert cli.main(["validate", str(path)]) == 2
    assert cli.main(["check-collisions", str(path)]) == 2


def test_events_document_declares_a_registered_facet():
    """自吃药：落盘的形制必须先在面清册登记（轴1规则1）。

    这条曾被本仓自己破了近两个版本——`collision_events.json`从0.3.0起
    就在落盘一个未登记的面名。门在此，破了就红。
    """

    from physics_engine.engine_facets import ENGINE_REGISTRY

    scene = load_scene(EXAMPLE.read_bytes())
    document = cli._events_document(scene, ())
    ENGINE_REGISTRY.assert_reader_compatible(document["facet"], document["facet_version"])


def test_legal_scene_still_passes_both_commands():
    """门只拒非法输入——合法示例场景的行为不得被这批校验改变。"""

    assert cli.main(["validate", str(EXAMPLE)]) == 0
    assert cli.main(["check-collisions", str(EXAMPLE)]) == 1  # 有候选=1
