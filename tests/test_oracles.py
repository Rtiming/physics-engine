"""oracle清单面自己的门——每一条都是一个"它必须红"的输入（轴7规则6）。

判据本身也要被验（AGENTS.md本仓纪律）：清单面的价值全在**拒收**上，
所以这里逐条构造被污染的清单，断言加载器当场炸。基准清单取仓内真件
`cases/segment_distance/oracle.json`，改一处、补回自指哈希、看它红。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from physics_engine.facets import FacetError
from physics_engine.oracles import (
    LOAD_TIERS,
    MANIFEST_PROFILE,
    OracleError,
    Tolerance,
    array_logical_sha256,
    file_sha256,
    flatten_values,
    load_manifest,
    manifest_self_sha256,
    parse_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "cases/segment_distance/oracle.json"
POPULATION = ROOT / "cases/broadphase_superset/oracle.json"


@pytest.fixture(scope="module")
def document() -> dict:
    return json.loads(REAL.read_text(encoding="utf-8"))


def _rewrite(tmp_path: Path, document: dict, *, refresh: bool = True) -> Path:
    payload = copy.deepcopy(document)
    if refresh:
        payload["manifest_self_sha256"] = manifest_self_sha256(payload)
    path = tmp_path / "oracle.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _expect_red(tmp_path: Path, document: dict, mutate, match: str, *, refresh: bool = True):
    payload = copy.deepcopy(document)
    mutate(payload)
    path = _rewrite(tmp_path, payload, refresh=refresh)
    with pytest.raises(OracleError, match=match):
        load_manifest(path, root=ROOT)


# ── 基准：真件必须能加载 ──────────────────────────────────────────────


def test_the_repository_manifests_load(document):
    manifest = load_manifest(REAL, root=ROOT)
    assert manifest.case_id == "case/segment_distance"
    assert manifest.load_tier in LOAD_TIERS
    assert manifest.regenerated_by is None
    assert len(manifest.oracles) == len(document["oracles"])


def test_manifest_identity_survives_reindentation(tmp_path, document):
    """身份是**规范字节**的哈希，不是文件字节——所以排版可以改给人读。

    这不是宽松，是分工：文件字节的完整性由git与评审负责，
    内容身份由规范化负责（轴3规则2）。
    """

    path = tmp_path / "reindented.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=7), encoding="utf-8")
    assert load_manifest(path, root=ROOT).self_sha256 == document["manifest_self_sha256"]


# ── 自指哈希 ─────────────────────────────────────────────────────────


def test_a_tampered_self_hash_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc.update({"manifest_self_sha256": "0" * 64}),
        "self hash mismatch",
        refresh=False,
    )


def test_editing_a_golden_value_without_regenerating_is_rejected(tmp_path, document):
    """规则5的执行体：手改一个expected值而不重生成→自指哈希对不上→红。"""

    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["oracles"][0]["expected"].update({"distance_mm": 4.9}),
        "self hash mismatch",
        refresh=False,
    )


# ── 面与顶层结构 ─────────────────────────────────────────────────────


def test_unknown_top_level_key_is_rejected(tmp_path, document):
    _expect_red(tmp_path, document, lambda doc: doc.update({"extra": 1}), "unknown manifest keys")


def test_missing_top_level_key_is_rejected(tmp_path, document):
    _expect_red(tmp_path, document, lambda doc: doc.pop("arrays"), "missing required keys")


def test_wrong_facet_name_is_rejected(tmp_path, document):
    _expect_red(tmp_path, document, lambda doc: doc.update({"facet": "physics_scene"}), "facet must")


def test_untested_facet_minor_is_rejected(tmp_path, document):
    payload = copy.deepcopy(document)
    payload["facet_version"] = "0.9"
    path = _rewrite(tmp_path, payload)
    with pytest.raises(FacetError, match="untested facet minor"):
        load_manifest(path, root=ROOT)


def test_case_id_must_be_namespaced(tmp_path, document):
    _expect_red(tmp_path, document, lambda doc: doc.update({"case_id": "segment"}), "case_id")


def test_load_tier_must_be_declared(tmp_path, document):
    _expect_red(tmp_path, document, lambda doc: doc.update({"load_tier": "whenever"}), "load_tier")


# ── 判据表（expected × tolerances） ──────────────────────────────────


def test_an_expected_quantity_without_a_tolerance_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["oracles"][0]["expected"].update({"orphan_mm": 1.0}),
        "every expected quantity needs exactly one tolerance",
    )


def test_a_tolerance_without_an_expected_quantity_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["oracles"][0]["tolerances"].update(
            {"ghost_mm": {"abs": 1.0, "rel": 0.0, "reason": "无主容差"}}
        ),
        "every expected quantity needs exactly one tolerance",
    )


def test_a_tolerance_without_a_reason_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["oracles"][0]["tolerances"]["distance_mm"].update({"reason": "   "}),
        "needs a reason",
    )


def test_a_negative_tolerance_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["oracles"][0]["tolerances"]["distance_mm"].update({"abs": -1.0}),
        "must be >= 0",
    )


def test_a_string_quantity_with_a_nonzero_tolerance_is_rejected(tmp_path, document):
    def mutate(doc):
        oracle = doc["oracles"][0]
        oracle["expected"] = {"confidence": "narrow_phase"}
        oracle["tolerances"] = {"confidence": {"abs": 1e-9, "rel": 0.0, "reason": "错的"}}

    _expect_red(tmp_path, document, mutate, "compared bit-for-bit")


def test_duplicate_oracle_ids_are_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["oracles"].append(copy.deepcopy(doc["oracles"][0])),
        "duplicate oracle id",
    )


def test_oracle_id_must_be_namespaced(tmp_path, document):
    _expect_red(
        tmp_path, document, lambda doc: doc["oracles"][0].update({"id": "plain"}), "oracle:"
    )


# ── 生成器身份与重生成留痕 ───────────────────────────────────────────


def test_a_changed_generator_script_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["generator"].update({"sha256": "f" * 64}),
        "generator script changed",
    )


def test_a_generator_without_the_algorithm_prefix_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc["generator"].update({"algorithm_id": "oracle/segment"}),
        "algorithm:",
    )


def test_regeneration_must_point_at_an_existing_decision_record(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc.update({"regenerated_by": "docs/decisions/9999_不存在.md"}),
        "decision record is missing",
    )


def test_regeneration_outside_the_decision_folder_is_rejected(tmp_path, document):
    _expect_red(
        tmp_path,
        document,
        lambda doc: doc.update({"regenerated_by": "README.md"}),
        "must point into docs/decisions/",
    )


# ── 数组双哈希 ───────────────────────────────────────────────────────


def test_array_raw_hash_catches_a_single_flipped_byte(tmp_path):
    manifest = load_manifest(POPULATION, root=ROOT)
    digest = manifest.array("samples")
    payload = bytearray((ROOT / digest.path_relative).read_bytes())
    payload[payload.index(b"1")] = ord("2")
    (tmp_path / "cases/broadphase_superset").mkdir(parents=True)
    (tmp_path / digest.path_relative).write_bytes(bytes(payload))
    with pytest.raises(OracleError, match="raw bytes changed"):
        manifest.load_array("samples", tmp_path)


def test_array_logical_hash_catches_a_re_serialised_but_altered_value(tmp_path):
    """raw级抓不到的那一半：重新序列化后raw必然变，所以这里直接验语义级本身。"""

    manifest = load_manifest(POPULATION, root=ROOT)
    digest = manifest.array("samples")
    values = list(flatten_values(json.loads((ROOT / digest.path_relative).read_text())["values"]))
    digest.verify_values(values)
    values[7] += 1.0e-9
    with pytest.raises(OracleError, match="logical hash changed"):
        digest.verify_values(values)


def test_array_count_mismatch_is_rejected():
    manifest = load_manifest(POPULATION, root=ROOT)
    with pytest.raises(OracleError, match="manifest declares"):
        manifest.array("samples").verify_values([1.0, 2.0])


def test_logical_hash_separates_dtypes():
    assert array_logical_sha256([1.0], dtype="float64") != array_logical_sha256(
        [1], dtype="int64"
    )


def test_flatten_values_fails_closed_on_non_numbers():
    with pytest.raises(OracleError):
        flatten_values([1.0, True])
    with pytest.raises(OracleError):
        flatten_values([1.0, "2"])


# ── 容差比较器 ───────────────────────────────────────────────────────


def test_tolerance_pairs_absolute_and_relative_terms():
    tolerance = Tolerance(abs_tol=1.0e-12, rel_tol=1.0e-9, reason="测试用")
    assert tolerance.holds(1000.0, 1000.0 + 9.0e-7)
    assert not tolerance.holds(1000.0, 1000.0 + 2.0e-6)
    assert tolerance.exceeded_by(1.0, 1.0) < 0.0


def test_check_rejects_a_value_outside_the_declared_tolerance():
    case = load_manifest(REAL, root=ROOT).oracles[0]
    quantity = next(iter(case.expected))
    case.check(quantity, case.expected[quantity])
    with pytest.raises(OracleError, match="exceeds abs"):
        case.check(quantity, case.expected[quantity] + 1.0e-6)


def test_check_all_rejects_a_missing_measurement():
    case = load_manifest(REAL, root=ROOT).oracles[0]
    with pytest.raises(OracleError, match="no measurement supplied"):
        case.check_all({})


def test_check_rejects_an_unknown_quantity():
    case = load_manifest(REAL, root=ROOT).oracles[0]
    with pytest.raises(OracleError, match="not an expected quantity"):
        case.check("nothing_mm", 1.0)


def test_vector_quantities_compare_component_wise():
    case = load_manifest(ROOT / "cases/rotated_aabb/oracle.json", root=ROOT).oracles[0]
    good = list(case.expected["world_aabb_min_mm"])
    case.check("world_aabb_min_mm", good)
    with pytest.raises(OracleError, match=r"world_aabb_min_mm\[1\]"):
        case.check("world_aabb_min_mm", [good[0], good[1] + 1.0, good[2]])
    with pytest.raises(OracleError, match="components"):
        case.check("world_aabb_min_mm", good[:2])


# ── 写侧 ─────────────────────────────────────────────────────────────


def test_write_manifest_refuses_to_emit_something_the_loader_would_reject(tmp_path):
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/smoke",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": "algorithm:oracle/smoke",
            "algorithm_version": "1.0.0",
            "path_relative": "cases/segment_distance/generate_oracle.py",
            "sha256": file_sha256(ROOT / "cases/segment_distance/generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:smoke/one",
                "inputs": {},
                "expected": {"value": 1.0},
                "tolerances": {"value": {"abs": 0.0, "rel": 0.0, "reason": "冒烟"}},
            }
        ],
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(tmp_path / "oracle.json", document, root=ROOT)
    assert written.endswith(b"\n")
    assert load_manifest(tmp_path / "oracle.json", root=ROOT).case_id == "case/smoke"

    broken = dict(document)
    broken["oracles"] = [dict(document["oracles"][0], tolerances={})]
    with pytest.raises(OracleError):
        write_manifest(tmp_path / "broken.json", broken, root=ROOT)
    assert not (tmp_path / "broken.json").exists()


def test_manifest_profile_is_declared_not_implied():
    """轴3规则2：规范化参数必须显式声明。清单面的声明就是这个对象。"""

    assert MANIFEST_PROFILE.ensure_ascii is False
    assert MANIFEST_PROFILE.encoding == "utf-8"


def test_parse_manifest_rejects_a_non_object():
    with pytest.raises(OracleError, match="must be a JSON object"):
        parse_manifest([1, 2, 3])
