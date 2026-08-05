"""run package装配与语义复读的门，含红例与消费方全量语义回放。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from physics_engine.canonical import FTS_PROFILE, canonical_file_bytes, strict_loads
from physics_engine.provenance import ProvenanceError
from physics_engine.run_package import (
    assert_lifecycle_fail_closed,
    publish_package,
    read_verified_package,
)


def _manifest_builder(digests: dict[str, str]) -> bytes:
    return canonical_file_bytes({"files": digests}, FTS_PROFILE)


def _extract(manifest: bytes) -> tuple[str, ...]:
    return tuple(strict_loads(manifest)["files"].values())


def _publish(tmp_path: Path) -> Path:
    return publish_package(
        tmp_path,
        "run-001",
        {"a.json": b'{"a":1}', "b.bin": b"\x00\x01"},
        manifest_name="manifest.json",
        manifest_builder=_manifest_builder,
    )


def test_publish_then_reread_roundtrips(tmp_path: Path):
    root = _publish(tmp_path)
    contents = read_verified_package(
        root, manifest_name="manifest.json", extract_declared_sha256s=_extract
    )
    assert contents["a.json"] == b'{"a":1}'
    assert not list(tmp_path.glob(".run-001.partial-*")), "临时区必须消失"


def test_publish_refuses_existing_destination(tmp_path: Path):
    _publish(tmp_path)
    with pytest.raises(ProvenanceError, match="already exists"):
        _publish(tmp_path)


def test_tampered_payload_fails_hash_closure(tmp_path: Path):
    root = _publish(tmp_path)
    (root / "a.json").write_bytes(b'{"a":2}')
    with pytest.raises(ProvenanceError, match="hash closure"):
        read_verified_package(
            root, manifest_name="manifest.json", extract_declared_sha256s=_extract
        )


def test_extra_file_fails_exact_closure(tmp_path: Path):
    root = _publish(tmp_path)
    (root / "extra.json").write_bytes(b"{}")
    with pytest.raises(ProvenanceError, match="closure mismatch"):
        read_verified_package(
            root, manifest_name="manifest.json", extract_declared_sha256s=_extract
        )


def test_missing_manifest_is_rejected(tmp_path: Path):
    root = _publish(tmp_path)
    (root / "manifest.json").unlink()
    with pytest.raises(ProvenanceError, match="no manifest"):
        read_verified_package(
            root, manifest_name="manifest.json", extract_declared_sha256s=_extract
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(lifecycle_status="queued", solver_status="not_evaluated",
              has_result=True, has_diagnostic=False, failure_stage=None,
              has_started=False, has_finished=False), "no outputs"),
        (dict(lifecycle_status="running", solver_status="converged",
              has_result=False, has_diagnostic=False, failure_stage=None,
              has_started=True, has_finished=False), "must not claim"),
        (dict(lifecycle_status="completed", solver_status="converged",
              has_result=False, has_diagnostic=False, failure_stage=None,
              has_started=True, has_finished=True), "exactly a result"),
        (dict(lifecycle_status="completed", solver_status="numerical_failure",
              has_result=True, has_diagnostic=True, failure_stage="solve",
              has_started=True, has_finished=True), "exactly a solve diagnostic"),
        (dict(lifecycle_status="failed", solver_status="not_evaluated",
              has_result=True, has_diagnostic=False, failure_stage="build",
              has_started=True, has_finished=True), "must not publish"),
    ],
)
def test_lifecycle_red_cases(kwargs, match):
    with pytest.raises(ProvenanceError, match=match):
        assert_lifecycle_fail_closed(**kwargs)


def test_lifecycle_green_paths():
    assert_lifecycle_fail_closed(
        lifecycle_status="completed", solver_status="converged",
        has_result=True, has_diagnostic=False, failure_stage=None,
        has_started=True, has_finished=True,
    )
    assert_lifecycle_fail_closed(
        lifecycle_status="completed", solver_status="numerical_failure",
        has_result=False, has_diagnostic=True, failure_stage="solve",
        has_started=True, has_finished=True,
    )


REPLAY_OUTPUT = os.environ.get("PE_REPLAY_OUTPUT_TREE")


def _wds_declared(manifest_bytes: bytes) -> tuple[str, ...]:
    document = json.loads(manifest_bytes)
    declared = [
        document["source_case_artifact"]["sha256"],
        document["resolved_case_artifact"]["sha256"],
        document["dependencies_artifact"]["sha256"],
    ]
    declared += [item["sha256"] for item in document.get("result_artifacts", [])]
    failure = document.get("failure") or {}
    declared += [item["sha256"] for item in failure.get("diagnostic_artifacts", [])]
    return tuple(declared)


@pytest.mark.skipif(
    not REPLAY_OUTPUT, reason="set PE_REPLAY_OUTPUT_TREE to a consumer output tree"
)
def test_semantic_replay_of_consumer_run_directories():
    """M-E2下半落地门：消费方全部已发布运行目录**含语义**逐例一致——

    精确闭包、哈希闭包、生命周期失败关闭三层全过；身份级语义（run_id、
    插件绑定）仍归消费方reader，本层不越权。
    """

    roots = sorted(
        manifest.parent
        for manifest in Path(REPLAY_OUTPUT).rglob("run_manifest.json")
    )
    assert roots, f"no run directories under {REPLAY_OUTPUT}"
    for run_dir in roots:
        contents = read_verified_package(
            run_dir,
            manifest_name="run_manifest.json",
            extract_declared_sha256s=_wds_declared,
        )
        document = json.loads(contents["run_manifest.json"])
        failure = document.get("failure") or {}
        assert_lifecycle_fail_closed(
            lifecycle_status=document["lifecycle_status"],
            solver_status=document["solver_status"],
            has_result=bool(document.get("result_artifacts")),
            has_diagnostic=bool(failure.get("diagnostic_artifacts")),
            failure_stage=failure.get("stage"),
            has_started=document.get("started_at_utc") is not None,
            has_finished=document.get("finished_at_utc") is not None,
        )
