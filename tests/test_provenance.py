"""溯源机械层的门，含全部"必须红"用例与WDS真实运行目录回放。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from physics_engine import provenance
from physics_engine.provenance import (
    ProvenanceError,
    directory_signature_snapshot,
    read_protected_file,
    rename_directory_noreplace,
    verified_bytes_snapshot,
    write_durable_exclusive,
)


def test_durable_write_refuses_to_overwrite(tmp_path: Path):
    target = tmp_path / "a.json"
    write_durable_exclusive(target, b"{}")
    with pytest.raises(FileExistsError):
        write_durable_exclusive(target, b"{}")


def test_noreplace_rename_moves_and_refuses_existing(tmp_path: Path):
    source = tmp_path / "staging"
    source.mkdir()
    (source / "f").write_bytes(b"x")
    destination = tmp_path / "final"
    rename_directory_noreplace(source, destination)
    assert (destination / "f").read_bytes() == b"x"

    second = tmp_path / "staging2"
    second.mkdir()
    empty_competitor = tmp_path / "final2"
    empty_competitor.mkdir()  # 哪怕是空目录也绝不覆盖
    with pytest.raises(ProvenanceError, match="already exists"):
        rename_directory_noreplace(second, empty_competitor)


def test_protected_read_refuses_symlink(tmp_path: Path):
    real = tmp_path / "real.json"
    real.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ProvenanceError, match="symlink refused"):
        read_protected_file(link)


def test_snapshot_refuses_subdirectories_and_symlinks(tmp_path: Path):
    (tmp_path / "ok.json").write_bytes(b"{}")
    (tmp_path / "nested").mkdir()
    with pytest.raises(ProvenanceError, match="non-regular entry"):
        directory_signature_snapshot(tmp_path)


def test_verified_snapshot_roundtrips(tmp_path: Path):
    (tmp_path / "a.json").write_bytes(b'{"a":1}')
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    contents = verified_bytes_snapshot(tmp_path)
    assert contents == {"a.json": b'{"a":1}', "b.bin": b"\x00\x01"}


def test_mid_verify_mutation_is_rejected(tmp_path: Path, monkeypatch):
    (tmp_path / "a.json").write_bytes(b'{"a":1}')
    target = tmp_path / "b.json"
    target.write_bytes(b'{"b":2}')
    target_before = provenance.file_signature(target.stat())
    original = provenance.read_protected_file
    state = {"fired": False}

    def mutate_then_read(path: Path) -> bytes:
        payload = original(path)
        if not state["fired"]:
            state["fired"] = True
            metadata = target.stat()
            os.utime(
                target,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 2_000_000_000),
            )
            assert provenance.file_signature(target.stat()) != target_before
        return payload

    monkeypatch.setattr(provenance, "read_protected_file", mutate_then_read)
    with pytest.raises(ProvenanceError, match="changed while being verified"):
        verified_bytes_snapshot(tmp_path)


REPLAY_ROOT = os.environ.get("PE_REPLAY_CASE_RUNS")


@pytest.mark.skipif(
    not REPLAY_ROOT, reason="set PE_REPLAY_CASE_RUNS to a consumer case-runs tree"
)
def test_replay_consumer_run_directories_and_reject_tamper(tmp_path: Path):
    """落地门：消费方已发布运行目录逐例机械复读通过；篡改副本必须拒。

    语义半边（manifest绑定）归run package装配；本门只对拍机械层判定——
    消费方严格复读器接受过的目录，本层必须也接受；改一个字节的副本，
    本层必须拒（经快照比对）。
    """

    roots = sorted(
        manifest.parent
        for manifest in Path(REPLAY_ROOT).glob("*/*/run_manifest.json")
    )
    assert roots, f"no run directories under {REPLAY_ROOT}"
    for run_dir in roots:
        contents = verified_bytes_snapshot(run_dir)
        assert "run_manifest.json" in contents

    victim = roots[0]
    copy = tmp_path / "tampered"
    shutil.copytree(victim, copy)
    target = copy / "run_manifest.json"
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    baseline = verified_bytes_snapshot(copy)
    target.write_bytes(bytes(payload))
    tampered = verified_bytes_snapshot(copy)
    assert tampered["run_manifest.json"] != baseline["run_manifest.json"], (
        "篡改必须反映在字节快照中（语义层哈希校验将在装配半边判拒）"
    )
