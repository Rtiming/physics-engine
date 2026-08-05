"""溯源原语——轴5（spec/06）的**机械半边**参考实现。

形制取自WDS `storage/case_runs.py`与`case_run_reader.py`（本仓消费方，
同主同权，蒸馏不涉许可问题）：

* 写侧：O_EXCL耐久写+文件/目录fsync+**no-replace目录改名**
  （macOS `renamex_np`、Linux `renameat2`，其它平台**失败关闭不静默降级**）；
* 读侧：O_NOFOLLOW+**四点签名**保护读（读前/打开/读后/最终四次stat逐一相等，
  防符号链接调包与读中替换）、目录初末签名快照。

**语义半边**（manifest绑定、哈希自洽、生命周期）不在本模块——它随
run package装配（M-E2后半）实现；本模块只保证"拿到的字节确实是盘上
那一刻的字节，且观察窗口内没人动过"。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import stat as stat_module
import sys
from pathlib import Path


class ProvenanceError(OSError):
    """一切溯源机械层的失败关闭。"""


def write_durable_exclusive(path: Path, content: bytes) -> None:
    """O_EXCL写入+fsync：目标已存在即失败，绝不覆盖。"""

    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    """目录项落盘。Windows无目录fsync语义，按平台直接返回。"""

    if sys.platform.startswith("win"):
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    """目录级no-replace改名：目标已存在（哪怕空目录）绝不覆盖。

    平台原语：macOS ``renamex_np(RENAME_EXCL)``、Linux
    ``renameat2(RENAME_NOREPLACE)``、Windows ``os.rename``（目标存在即错）。
    其它平台没有原子no-replace原语，**失败关闭**——静默退化成可覆盖改名
    正是轴5禁止的形状。
    """

    src = os.fspath(source).encode()
    dst = os.fspath(destination).encode()
    if sys.platform == "darwin":
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(src, dst, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
        )
        result = rename(-100, src, -100, dst, 0x00000001)  # AT_FDCWD, RENAME_NOREPLACE
    elif sys.platform.startswith("win"):
        if destination.exists():
            raise ProvenanceError(f"destination already exists: {destination}")
        os.rename(source, destination)
        return
    else:
        raise ProvenanceError(
            f"atomic no-replace directory rename is unavailable on {sys.platform}; "
            "failing closed instead of silently degrading"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in (17, 39, 66):  # EEXIST, ENOTEMPTY(darwin 66/linux 39)
            raise ProvenanceError(f"destination already exists: {destination}")
        raise ProvenanceError(
            f"no-replace rename failed: {os.strerror(error)} ({source} -> {destination})"
        )


FileSignature = tuple[int, int, int, int, int]


def file_signature(metadata: os.stat_result) -> FileSignature:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_protected_file(path: Path) -> bytes:
    """O_NOFOLLOW四点签名保护读：读前/打开/读后/最终逐一相等，否则拒收。"""

    try:
        before = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ProvenanceError(f"cannot stat {path}: {error}") from error
    if not stat_module.S_ISREG(before.st_mode):
        raise ProvenanceError(f"not a regular file (symlink refused): {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if file_signature(opened) != file_signature(before):
            raise ProvenanceError(f"file changed before reading: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if file_signature(after) != file_signature(opened):
            raise ProvenanceError(f"file changed while being read: {path}")
    finally:
        os.close(descriptor)
    final = os.stat(path, follow_symlinks=False)
    if file_signature(final) != file_signature(after):
        raise ProvenanceError(f"file changed after reading: {path}")
    return b"".join(chunks)


def directory_signature_snapshot(root: Path) -> tuple[tuple[int, int], dict[str, FileSignature]]:
    """根身份(dev,ino)+全部直接常规文件的签名。非常规条目一律拒收。"""

    root_stat = os.stat(root, follow_symlinks=False)
    if not stat_module.S_ISDIR(root_stat.st_mode):
        raise ProvenanceError(f"not a direct directory (symlink refused): {root}")
    signatures: dict[str, FileSignature] = {}
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        metadata = os.stat(entry.path, follow_symlinks=False)
        if not stat_module.S_ISREG(metadata.st_mode):
            raise ProvenanceError(f"non-regular entry in package: {entry.path}")
        signatures[entry.name] = file_signature(metadata)
    return (root_stat.st_dev, root_stat.st_ino), signatures


def verified_bytes_snapshot(root: Path) -> dict[str, bytes]:
    """机械严格复读：返回"观察窗口内无漂移"的字节快照，否则拒收。

    语义校验（manifest绑定、哈希）由调用方在这份字节上做——本函数保证的
    只是字节与盘上那一刻一致。诚实边界同WDS reader：不声称能阻止复读
    结束后的外部写者。
    """

    root_identity, initial = directory_signature_snapshot(root)
    contents = {name: read_protected_file(root / name) for name in initial}
    final_identity, final = directory_signature_snapshot(root)
    if final != initial:
        raise ProvenanceError(f"package changed while being verified: {root}")
    if final_identity != root_identity:
        raise ProvenanceError(f"package directory was replaced while being verified: {root}")
    return contents


__all__ = [
    "FileSignature",
    "ProvenanceError",
    "directory_signature_snapshot",
    "file_signature",
    "fsync_directory",
    "read_protected_file",
    "rename_directory_noreplace",
    "verified_bytes_snapshot",
    "write_durable_exclusive",
]
