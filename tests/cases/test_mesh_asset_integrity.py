"""`case/mesh_asset_integrity`的conformance门（轴7规则3+规则6）。

这是本批唯一**今天就必红**的门：引擎至今没有任何校验能发现"声明的SHA对得上、
声明的包盒却与资产真值在某轴上完全不相交"（decisions/0017第四条的真实缺陷）。
两条必红语料`red_wrong_aabb`/`red_wrong_sha`就在仓里，红不红是断言不是承诺。

二进制STL解析放在案例侧（本文件），**不进`src/`**——引擎当前的设计是
"不解析网格字节、包盒由声明携带"（`shapes.py`的MeshAsset文档）。要不要把
解析升进产品面是B档的裁决，不是这条案例顺手做掉的事，见`case.md`第六节。
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from physics_engine.oracles import file_sha256, load_manifest
from physics_engine.scene import load_scene
from physics_engine.shapes import MeshAsset

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/mesh_asset_integrity/oracle.json", root=ROOT)

_HEADER_BYTES = 80
_COUNT_BYTES = 4
_TRIANGLE_BYTES = 50


def parse_binary_stl(payload: bytes) -> tuple[int, list[tuple[float, float, float]]]:
    """二进制STL→(三角形数, 顶点表)。失败关闭，不猜格式。"""

    if len(payload) < _HEADER_BYTES + _COUNT_BYTES:
        raise ValueError("binary STL is shorter than its header")
    if payload[:5].lower() == b"solid":
        raise ValueError("this looks like an ASCII STL; the case only handles the binary form")
    (count,) = struct.unpack_from("<I", payload, _HEADER_BYTES)
    expected = _HEADER_BYTES + _COUNT_BYTES + _TRIANGLE_BYTES * count
    if len(payload) != expected:
        raise ValueError(f"binary STL declares {count} triangles but is {len(payload)} bytes")
    vertices: list[tuple[float, float, float]] = []
    for index in range(count):
        base = _HEADER_BYTES + _COUNT_BYTES + _TRIANGLE_BYTES * index + 12  # 跳过法向
        for corner in range(3):
            vertices.append(struct.unpack_from("<3f", payload, base + 12 * corner))
    return count, vertices


class GateNotApplicable(Exception):
    """门拒绝下判决（而不是判红）——`fitted`语义不适用于包络保守性判据。"""


def judge_scene(scene_path: Path, root: Path) -> dict:
    """本案例建的那道门：资产SHA + 包络保守性，逐轴报违规。"""

    scene = load_scene(scene_path.read_bytes())
    collision = scene.posed_bodies[0].body.collision
    asset = collision.shape
    if not isinstance(asset, MeshAsset):
        raise GateNotApplicable("本判据只针对网格资产声明")
    if collision.direction != "envelope":
        raise GateNotApplicable(
            f"direction={collision.direction!r}：贴合形不承诺包住资产，"
            "包络保守性判据对它不成立，也不许套用"
        )
    if asset.units != "mm":
        raise GateNotApplicable(
            f"units={asset.units!r}：本仓无量纲换算代码（plans/02第一节identity缺口），"
            "换算一旦静默做错就是1000倍——宁可拒判"
        )
    raw = (root / asset.path_relative).read_bytes()
    count, vertices = parse_binary_stl(raw)
    true_min = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    true_max = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    violated: list[str] = []
    for axis, name in enumerate("xyz"):
        if not asset.aabb_min_mm[axis] <= true_min[axis]:
            violated.append(f"min_{name}")
        if not asset.aabb_max_mm[axis] >= true_max[axis]:
            violated.append(f"max_{name}")
    return {
        "sha256_matches": hashlib.sha256(raw).hexdigest() == asset.sha256,
        "envelope_encloses_asset": not violated,
        "violated_axes": violated,
        "true_aabb_min_mm": list(true_min),
        "true_aabb_max_mm": list(true_max),
        "triangle_count": count,
        "asset_bytes": len(raw),
    }


@pytest.mark.parametrize("case", MANIFEST.oracles, ids=[case.id for case in MANIFEST.oracles])
def test_scene_verdict_matches_the_frozen_oracle(case):
    scene_path = ROOT / case.inputs["scene_path_relative"]
    assert file_sha256(scene_path) == case.inputs["scene_sha256"], (
        "语料场景文件被改过而清单没重生成——金标与它的输入必须同批变（轴7规则5）"
    )
    case.check_all(judge_scene(scene_path, ROOT))


def test_the_two_red_corpora_really_are_red():
    """规则6要求每道门有『它必须红』的输入。这里把红的**理由**也钉住：
    一条只错SHA、一条只错包盒——两条判据各自独立可失效，不是一起绿一起红。"""

    verdicts = {
        case.id.rsplit("/", 1)[-1]: case.expected
        for case in MANIFEST.oracles
    }
    assert verdicts["red_wrong_sha"]["sha256_matches"] is False
    assert verdicts["red_wrong_sha"]["envelope_encloses_asset"] is True
    assert verdicts["red_wrong_aabb"]["sha256_matches"] is True
    assert verdicts["red_wrong_aabb"]["envelope_encloses_asset"] is False
    assert verdicts["tetra_envelope"]["sha256_matches"] is True
    assert verdicts["tetra_envelope"]["envelope_encloses_asset"] is True


def test_gate_refuses_to_judge_a_fitted_declaration():
    """`fitted`语义不同：贴合形可以比资产小。门必须**拒判**而不是判红。"""

    scene_path = ROOT / "cases/mesh_asset_integrity/fitted_not_judged.scene.json"
    with pytest.raises(GateNotApplicable, match="fitted"):
        judge_scene(scene_path, ROOT)


def test_asset_generator_is_in_the_repository_and_unchanged():
    """Chrono形制：生成金标的输入卡/脚本一起入库，且钉SHA。"""

    case = MANIFEST.oracles[0]
    generator = ROOT / case.inputs["asset_generator_path_relative"]
    assert generator.is_file()
    assert file_sha256(generator) == case.inputs["asset_generator_sha256"]
