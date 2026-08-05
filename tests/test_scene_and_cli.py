"""场景文件与pe-scene入口的门：严格加载红例、扩展声明加载、端到端CLI。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physics_engine import cli
from physics_engine.run_package import read_verified_package
from physics_engine.canonical import strict_loads
from physics_engine.scene import SceneError, load_scene

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/collision_preview_cell.scene.json"


def _document() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _payload(document: dict) -> bytes:
    return json.dumps(document).encode("utf-8")


def test_example_scene_loads_and_finds_the_touching_pair():
    scene = load_scene(EXAMPLE.read_bytes())
    assert scene.scene_id == "scene/collision_preview_cell"
    assert len(scene.posed_bodies) == 4


def test_unknown_top_key_is_rejected():
    document = _document()
    document["gravity"] = [0, 0, -9.81]
    with pytest.raises(SceneError, match="unknown top-level keys"):
        load_scene(_payload(document))


def test_unknown_shape_kind_names_the_extension_path():
    document = _document()
    document["bodies"][0]["collision"]["shape"] = {"kind": "torus", "radius_mm": 1.0}
    with pytest.raises(SceneError, match="declared in the scene's 'extensions'"):
        load_scene(_payload(document))


def test_wrong_contract_version_fails_closed():
    document = _document()
    document["contract_version"] = "2.0.0"
    with pytest.raises(Exception, match="unsupported facet major"):
        load_scene(_payload(document))


def test_declared_extension_registers_a_new_kind(tmp_path, monkeypatch):
    module = tmp_path / "torus_ext.py"
    module.write_text(
        "from physics_engine.scene import register_shape_kind\n"
        "from physics_engine.shapes import Sphere\n"
        "register_shape_kind('torus_proxy', lambda radius_mm: Sphere(radius_mm=radius_mm))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    document = _document()
    document["extensions"] = ["torus_ext"]
    document["bodies"][0]["collision"]["shape"] = {"kind": "torus_proxy", "radius_mm": 45.0}
    scene = load_scene(_payload(document))
    assert scene.posed_bodies[0].body.body_id == "body/roller_r1"


def test_missing_extension_module_fails_closed():
    document = _document()
    document["extensions"] = ["no_such_extension_module"]
    with pytest.raises(SceneError, match="not importable"):
        load_scene(_payload(document))


def test_cli_validate_and_check_collisions_roundtrip(tmp_path, capsys):
    assert cli.main(["validate", str(EXAMPLE)]) == 0
    out_dir = tmp_path / "runs"
    code = cli.main(
        ["check-collisions", str(EXAMPLE), "--out-dir", str(out_dir), "--run-name", "t1"]
    )
    assert code == 1  # 有候选=1（linter惯例）
    printed = capsys.readouterr().out
    assert "body/roller_r1 <-> body/spool" in printed
    contents = read_verified_package(
        out_dir / "t1",
        manifest_name="manifest.json",
        extract_declared_sha256s=lambda raw: tuple(strict_loads(raw)["files"].values()),
    )
    events = strict_loads(contents["collision_events.json"])
    assert events["events"][0]["confidence"] == "broad_phase"
    assert events["events"][0]["penetration_mm"] is None


def test_cli_reports_zero_when_pairs_are_declared(tmp_path):
    document = _document()
    document["allowed_pairs"] = [["body/roller_r1", "body/spool"]]
    scene_path = tmp_path / "quiet.scene.json"
    scene_path.write_text(json.dumps(document), encoding="utf-8")
    assert cli.main(["check-collisions", str(scene_path)]) == 0


def test_cli_invalid_input_exits_two(tmp_path):
    bad = tmp_path / "bad.scene.json"
    bad.write_text("{nope", encoding="utf-8")
    assert cli.main(["validate", str(bad)]) == 2
