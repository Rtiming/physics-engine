"""`pe-scene`——引擎的数据层命令行入口（一条命令跑一个场景文件）。

形制学Gazebo（`gz sim world.sdf`）与linter惯例的退出码：

* `pe-scene validate scene.json` —— 0=合法，2=非法输入；
* `pe-scene check-collisions scene.json [--out-dir DIR] [--run-name NAME]`
  —— 0=无broad-phase候选，1=有候选，2=非法输入。

给`--out-dir`时产物走run package原子发布（轴4/5），并**立即用对外同一个
加载器复读验真**（轴5规则1，引擎继续吃自己的药）。产物两件：
`scene-resolved.json`（解析后的场景规范字节）与`collision_events.json`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from physics_engine.canonical import canonical_file_bytes, strict_loads
from physics_engine.collision import BroadPhaseCollisionQuery
from physics_engine.engine_facets import PHYSICS_SCENE_FACET, PHYSICS_SCENE_VERSION
from physics_engine.run_package import publish_package, read_verified_package
from physics_engine.scene import SCENE_CANONICAL_PROFILE, SceneError, load_scene


def _load(path: Path):
    payload = path.read_bytes()
    scene = load_scene(payload)
    document = strict_loads(payload)
    return scene, document


def _events_document(scene, events) -> dict:
    return {
        "facet": "physics_scene_collision_events",
        "scene_id": scene.scene_id,
        "scene_sha256": scene.source_sha256,
        "events": [
            {
                "body_a": event.body_a,
                "body_b": event.body_b,
                "confidence": event.confidence,
                "penetration_mm": event.penetration_mm,
            }
            for event in events
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pe-scene", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="严格校验一个场景文件")
    validate.add_argument("scene", type=Path)
    check = commands.add_parser("check-collisions", help="broad-phase碰撞候选")
    check.add_argument("scene", type=Path)
    check.add_argument("--out-dir", type=Path, default=None)
    check.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args(argv)

    try:
        scene, document = _load(args.scene)
    except (SceneError, OSError, ValueError) as error:
        print(f"invalid scene: {error}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(
            f"valid: {scene.scene_id} bodies={len(scene.posed_bodies)} "
            f"sha256={scene.source_sha256[:12]}…"
        )
        return 0

    events = BroadPhaseCollisionQuery(
        scene.posed_bodies, allowed_pairs=scene.allowed_pairs
    ).check_state()
    for event in events:
        print(f"{event.confidence}: {event.body_a} <-> {event.body_b}")
    print(f"{len(events)} broad-phase candidate(s) in {scene.scene_id}")

    if args.out_dir is not None:
        run_name = args.run_name or f"collision-check-{scene.source_sha256[:12]}"
        payload = {
            "scene-resolved.json": canonical_file_bytes(document, SCENE_CANONICAL_PROFILE),
            "collision_events.json": canonical_file_bytes(
                _events_document(scene, events), SCENE_CANONICAL_PROFILE
            ),
        }

        def manifest_builder(digests: dict[str, str]) -> bytes:
            return canonical_file_bytes(
                {
                    "facet": PHYSICS_SCENE_FACET,
                    "facet_version": PHYSICS_SCENE_VERSION,
                    "scene_id": scene.scene_id,
                    "scene_sha256": scene.source_sha256,
                    "files": digests,
                },
                SCENE_CANONICAL_PROFILE,
            )

        args.out_dir.mkdir(parents=True, exist_ok=True)
        root = publish_package(
            args.out_dir, run_name, payload,
            manifest_name="manifest.json", manifest_builder=manifest_builder,
        )
        read_verified_package(
            root,
            manifest_name="manifest.json",
            extract_declared_sha256s=lambda raw: tuple(
                strict_loads(raw)["files"].values()
            ),
        )
        print(f"published+reread: {root}")

    return 1 if events else 0


if __name__ == "__main__":
    raise SystemExit(main())
