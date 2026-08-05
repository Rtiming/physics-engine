"""run package装配与语义复读——轴4/轴5的**语义半边**参考实现（M-E2下半）。

建立在`provenance`机械层之上，补三件语义事：

1. **发布装配**（轴5 `atomic_publish`档）：临时区耐久写全部载荷→
   逐文件哈希复验→manifest**最后**落定（partial+replace）→目录fsync→
   no-replace改名进终态→父目录fsync。manifest的出现即声称"集合完整"。
2. **语义复读**：机械快照之上验**精确闭包**（载荷文件数=声明数）与
   **哈希闭包**（盘上字节哈希的多重集=声明多重集）。本层刻意
   **schema无关**——声明哈希由调用方的抽取器从manifest字节里取，
   引擎不预设任何一家的manifest形制。
3. **生命周期失败关闭**（轴4规则4参考语义）：排队/运行中不得声称结果；
   完成态按求解状态分叉，数值失败是一等结果（有诊断、禁结果）；
   失败分阶段约束求解状态。

角色/身份级校验（谁的run_id、哪个插件）仍归消费方contract——
那一层是各仓的面，不该被引擎抢走。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from physics_engine.provenance import (
    ProvenanceError,
    fsync_directory,
    rename_directory_noreplace,
    verified_bytes_snapshot,
    write_durable_exclusive,
)


def publish_package(
    parent: Path,
    name: str,
    payload: dict[str, bytes],
    *,
    manifest_name: str,
    manifest_builder: Callable[[dict[str, str]], bytes],
) -> Path:
    """原子发布：manifest最后写，目标已存在（哪怕空目录）绝不覆盖。

    ``manifest_builder``拿到``{载荷文件名: sha256}``后返回manifest字节——
    manifest的形制归调用方，落盘纪律归本函数。
    """

    if manifest_name in payload:
        raise ProvenanceError("manifest must not appear in the payload mapping")
    if not payload:
        raise ProvenanceError("a package requires at least one payload file")
    final = parent / name
    if final.exists():
        raise ProvenanceError(f"package destination already exists: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.partial-", dir=parent))
    try:
        digests: dict[str, str] = {}
        for filename, content in payload.items():
            target = staging / filename
            write_durable_exclusive(target, content)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != hashlib.sha256(content).hexdigest():
                raise ProvenanceError(f"post-write hash drift: {filename}")
            digests[filename] = digest
        manifest_bytes = manifest_builder(dict(digests))
        partial = staging / (manifest_name + ".partial")
        write_durable_exclusive(partial, manifest_bytes)
        os.replace(partial, staging / manifest_name)
        fsync_directory(staging)
        rename_directory_noreplace(staging, final)
        fsync_directory(parent)
        return final
    except BaseException:
        for leftover in staging.glob("*") if staging.exists() else ():
            leftover.unlink(missing_ok=True)
        if staging.exists():
            staging.rmdir()
        raise


def read_verified_package(
    root: Path,
    *,
    manifest_name: str,
    extract_declared_sha256s: Callable[[bytes], tuple[str, ...]],
) -> dict[str, bytes]:
    """机械快照+语义闭包：文件数与哈希多重集都必须与manifest声明精确相等。"""

    contents = verified_bytes_snapshot(root)
    if manifest_name not in contents:
        raise ProvenanceError(f"package has no manifest {manifest_name!r}: {root}")
    declared = extract_declared_sha256s(contents[manifest_name])
    if not declared:
        raise ProvenanceError(f"manifest declares no payload files: {root}")
    payload_names = sorted(name for name in contents if name != manifest_name)
    if len(payload_names) != len(declared):
        raise ProvenanceError(
            f"package closure mismatch: {len(payload_names)} files on disk, "
            f"{len(declared)} declared ({root})"
        )
    actual = Counter(
        hashlib.sha256(contents[name]).hexdigest() for name in payload_names
    )
    if actual != Counter(declared):
        raise ProvenanceError(f"package hash closure mismatch: {root}")
    return contents


_TERMINAL = frozenset({"completed", "failed", "cancelled"})


def assert_lifecycle_fail_closed(
    *,
    lifecycle_status: str,
    solver_status: str,
    has_result: bool,
    has_diagnostic: bool,
    failure_stage: str | None,
    has_started: bool,
    has_finished: bool,
) -> None:
    """轴4规则4的参考语义（蒸馏自WDS `case_runtime`的失败关闭验证器）。"""

    def refuse(reason: str) -> None:
        raise ProvenanceError(f"lifecycle fail-closed violation ({lifecycle_status}): {reason}")

    if lifecycle_status == "queued":
        if has_started or has_finished or solver_status != "not_evaluated":
            refuse("queued run must claim nothing")
        if has_result or has_diagnostic or failure_stage is not None:
            refuse("queued run must carry no outputs")
    elif lifecycle_status == "running":
        if not has_started or has_finished or solver_status != "not_evaluated":
            refuse("running run must not claim results")
        if has_result or has_diagnostic or failure_stage is not None:
            refuse("running run must carry no outputs")
    elif lifecycle_status == "completed":
        if not (has_started and has_finished):
            refuse("completed run requires both timestamps")
        if solver_status == "converged":
            if not has_result or has_diagnostic or failure_stage is not None:
                refuse("converged completion requires exactly a result")
        elif solver_status == "numerical_failure":
            if has_result or not has_diagnostic or failure_stage != "solve":
                refuse("numerical failure requires exactly a solve diagnostic")
        else:
            refuse("completed run must have an evaluated solver status")
    elif lifecycle_status == "failed":
        if failure_stage not in {"build", "solve", "publish"}:
            refuse("failed run requires a non-cancel failure stage")
        if has_result:
            refuse("failed run must not publish a result")
    elif lifecycle_status == "cancelled":
        if solver_status != "not_evaluated" or failure_stage != "cancel" or has_result:
            refuse("cancelled run claims work it did not do")
    else:
        refuse("unknown lifecycle status")
    if lifecycle_status in _TERMINAL and not has_finished:
        refuse("terminal run requires a finish timestamp")


__all__ = [
    "assert_lifecycle_fail_closed",
    "publish_package",
    "read_verified_package",
]
