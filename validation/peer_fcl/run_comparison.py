"""跑一次同行库对拍，产物走run package（轴4/5）落盘。

**只在`validation/.venv`里跑**：本文件import`fcl`与`numpy`，那两个包永不进主环境。
用法（路径全部从本文件位置算，无写死的用户目录）：

    validation/.venv/bin/python validation/peer_fcl/run_comparison.py \\
        --out-dir work/peer_fcl --name run-001 [--per-cell 300] [--fault quaternion_order]

产物目录里四个载荷 + 一个manifest：

- `samples.json`   逐点原始记录（两侧的**全部**返回值，不止判据用的那一列）
- `summary.json`   按(类,band)聚合
- `criteria.json`  判据声明的逐字副本（包自足：不必回仓才能复读结论）
- `verdict.json`   逐条判据的通过与否 + 实测值

manifest带`peer_library`与`runtime_environment`两块——同行库名称/版本/安装方式/
平台/许可证，以及Python/numpy/BLAS与二进制哈希（spec/13第三节的自然延伸）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
for _extra in (_REPO_ROOT / "src", _REPO_ROOT / "validation"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import fcl  # noqa: E402
import numpy  # noqa: E402
from peer_fcl.harness import (  # noqa: E402
    FAULTS,
    KNOWN_FCL_DEFECTS,
    compare,
    generate_samples,
    summarise,
)

from physics_engine.canonical import FTS_PROFILE, canonical_file_bytes  # noqa: E402
from physics_engine.engine_facets import (  # noqa: E402
    ORACLE_MANIFEST_FACET,
    ORACLE_MANIFEST_VERSION,
)
from physics_engine.run_package import publish_package  # noqa: E402

CASE_ID = "peer_fcl_distance"
CRITERIA_PATH = _REPO_ROOT / "cases" / CASE_ID / "criteria.json"
MANIFEST_NAME = "peer_comparison.manifest.json"


def _load_criteria() -> dict:
    return json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))


def _peer_library_block() -> dict:
    """同行库的身份三件套：名称/版本/安装方式 + 平台 + 许可证 + 二进制哈希。

    二进制哈希不是装饰：wheel名相同而编译产物不同（同行改了构建、或本地
    自己编了一份）是真实可能，只钉版本号钉不住。
    """

    metadata = importlib_metadata.metadata("python-fcl")
    package_dir = Path(fcl.__file__).resolve().parent
    binaries = {}
    for pattern in ("*.so", "*.pyd", "*.dylib", "*.dll"):
        for path in sorted(package_dir.rglob(pattern)):
            binaries[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "distribution": metadata["Name"],
        "version": importlib_metadata.version("python-fcl"),
        "license": metadata.get("License") or metadata.get("License-Expression"),
        "license_spdx": "BSD-3-Clause",
        "upstream": "https://github.com/berkeleyautomation/python-fcl",
        "install_method": "pip install python-fcl（PyPI预编译wheel，未本地编译）",
        "index_url": "https://pypi.org/simple",
        "install_target": "validation/.venv（独立环境，永不进主环境与dependencies）",
        "bundled_binaries_sha256": binaries,
        "role": "验证期oracle与对比证人；不是依赖，不被src/physics_engine import",
    }


def _runtime_environment_block() -> dict:
    try:
        blas = numpy.__config__.show(mode="dicts")["Build Dependencies"]["blas"]
        blas_name = f"{blas.get('name')}@{blas.get('version')}"
    except Exception:  # pragma: no cover - numpy配置形制随版本变
        blas_name = "unknown"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "numpy_version": numpy.__version__,
        "blas": blas_name,
        "determinism_class": "deterministic_cpu_f64",
    }


def _evaluate(criteria: dict, rows: list[dict], summary: dict) -> dict:
    """逐条判据求值。判据的数**只从criteria.json读**（轴7规则3）。"""

    by_id = {item["id"]: item for item in criteria["criteria"]}
    checks: list[dict] = []

    j1 = by_id["J1"]
    abs_tol, rel_tol = j1["abs_tolerance_mm"], j1["rel_tolerance"]
    offenders = [
        row["index"]
        for row in rows
        if row["abs_deviation_mm"] is None
        or (row["abs_deviation_mm"] > abs_tol and row["rel_deviation"] > rel_tol)
    ]
    checks.append(
        {
            "id": "J1",
            "passed": not offenders,
            "observed_max_abs_deviation_mm": summary["overall"]["max_abs_deviation_mm"],
            "observed_max_rel_deviation": summary["overall"]["max_rel_deviation"],
            "offending_sample_count": len(offenders),
            "offending_sample_indices": offenders[:20],
        }
    )

    disagreements = summary["overall"]["predicate_disagreements"]
    checks.append(
        {"id": "J2", "passed": disagreements == 0, "observed_disagreements": disagreements}
    )

    expected = criteria["expected_peer_operator"]
    observed = {key: cell["peer_operator"] for key, cell in summary["cells"].items()}
    checks.append(
        {
            "id": "J3",
            "passed": observed == expected,
            "observed_peer_operator": observed,
            "expected_peer_operator": expected,
        }
    )

    want = criteria["sweep"]["expected_sample_count"]
    got = summary["overall"]["count"]
    missing = summary["overall"]["missing_peer_value"]
    checks.append(
        {
            "id": "J4",
            "passed": got == want and missing == 0,
            "observed_count": got,
            "expected_count": want,
            "missing_peer_value": missing,
        }
    )
    return {"checks": checks, "overall_passed": all(check["passed"] for check in checks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="physics-engine × FCL 距离对拍")
    parser.add_argument("--out-dir", required=True, help="run package的父目录")
    parser.add_argument("--name", required=True, help="run package目录名（已存在则拒绝覆盖）")
    parser.add_argument("--per-cell", type=int, default=None, help="每格样本数，缺省读criteria")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，缺省读criteria")
    parser.add_argument(
        "--fault", default="none", choices=sorted(FAULTS), help="注错模式（必须红演示用）"
    )
    args = parser.parse_args(argv)

    criteria = _load_criteria()
    seed = args.seed if args.seed is not None else criteria["sweep"]["seed"]
    per_cell = args.per_cell if args.per_cell is not None else criteria["sweep"]["per_cell"]

    samples = generate_samples(seed, per_cell)
    rows = compare(fcl, numpy, samples, args.fault)
    summary = summarise(rows)
    verdict = _evaluate(criteria, rows, summary)
    verdict["fault"] = args.fault
    verdict["fault_description"] = FAULTS[args.fault]
    verdict["note_on_peer_defects"] = KNOWN_FCL_DEFECTS

    payload = {
        "samples.json": canonical_file_bytes({"rows": rows}, FTS_PROFILE),
        "summary.json": canonical_file_bytes(summary, FTS_PROFILE),
        "criteria.json": canonical_file_bytes(criteria, FTS_PROFILE),
        "verdict.json": canonical_file_bytes(verdict, FTS_PROFILE),
    }

    def build_manifest(digests: dict[str, str]) -> bytes:
        document = {
            "facet": ORACLE_MANIFEST_FACET,
            "facet_version": ORACLE_MANIFEST_VERSION,
            "case_id": CASE_ID,
            "kind": "peer_library_cross_check",
            "generator": {
                "algorithm_id": criteria["generator_algorithm_id"],
                "algorithm_version": criteria["generator_version"],
                "source_relative": "validation/peer_fcl/run_comparison.py",
            },
            "sweep": {
                "seed": seed,
                "per_cell": per_cell,
                "sample_count": len(samples),
                "fault": args.fault,
            },
            "subject_under_test": [
                "physics_engine.collision.segment_segment_distance_mm",
                "physics_engine.collision.BroadPhaseCollisionQuery",
            ],
            "peer_library": _peer_library_block(),
            "runtime_environment": _runtime_environment_block(),
            "overall_passed": verdict["overall_passed"],
            "payload_sha256": dict(sorted(digests.items())),
        }
        return canonical_file_bytes(document, FTS_PROFILE)

    parent = Path(args.out_dir).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = publish_package(
        parent,
        args.name,
        payload,
        manifest_name=MANIFEST_NAME,
        manifest_builder=build_manifest,
    )
    print(json.dumps({"package": str(final), "overall_passed": verdict["overall_passed"]}))
    return 0 if verdict["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
