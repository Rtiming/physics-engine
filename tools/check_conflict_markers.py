#!/usr/bin/env python3
"""合并冲突标记检查器：扫全部受版本控制的文本文件，见即红。

    python tools/check_conflict_markers.py          # 扫全仓
    python tools/check_conflict_markers.py a.md b.py  # 只扫指定文件（pre-commit用）

**为什么是`tools/`里的独立脚本而不只是一条测试**：这道门要在**两个时机**上膛——
`pre-commit`（提交时）与`accept.py`（批末）。提交时的那次不能依赖pytest与`.venv`：
钩子在裸`python`下跑，且要跨Windows/macOS（本仓两台开发机平级）。
所以判据落在这里、**只用标准库**，`tests/governance/test_no_conflict_markers.py`
import它并补必红用例——**一个判据，两个时机，不许有第二份实现**。

起因见决策0049第二节：`main`曾带着12行未解冲突标记（`CHANGELOG.md`三处嵌套9行、
`cases/README.md` 3行），而`accept.py` full全绿。冲突标记**不会让任何东西红**——
在`.py`里是语法错当场就炸，**而那次撞上的两个文件都是`.md`**。

## 判据形制：`=======`要条件化，其余三个无条件

`<<<<<<<`、`|||||||`（diff3的共同祖先段）、`>>>>>>>`——**行首见即红**。

`=======`**只有当同一文件里还存在上面任意一个时才算**。理由：它在Markdown里是
合法字节——setext式一级标题的下划线就是一串`=`，长度恰好7个时与冲突分隔线
逐字节相同。而git写出的冲突**永远是成对的**，没有"只有分隔线的冲突"。
**一道会对合法文档误红的门，最终会被加豁免、然后被拆掉。**

本文件的字节里**一个七连标记都没有**：标记由``"<" * 7``一类的式子在运行期拼出。
否则这道门会在自己身上红，于是只能给自己开豁免，而**带豁免的门是没有门的**。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 七连标记按运行期拼，理由见模块docstring末段。
OURS = "<" * 7
BASE = "|" * 7
THEIRS = ">" * 7
SEPARATOR = "=" * 7

#: 无条件标记：行首出现即为冲突，合法文本里没有第二种解释。
#: git写的形态是"标记+空格+标签"，手工mangle过的可能只剩裸标记，两种都收。
UNCONDITIONAL = (OURS, BASE, THEIRS)


def _is_unconditional_marker(line: str) -> bool:
    stripped = line.rstrip("\r\n")
    return any(
        stripped == marker or stripped.startswith(marker + " ") for marker in UNCONDITIONAL
    )


def find_conflict_markers(text: str) -> list[tuple[int, str]]:
    """返回``[(行号从1起, 该行内容), ...]``。判据见模块docstring第二节。

    ``=======``的条件化在这里实现：先扫一遍无条件标记，**只有扫到了**
    才把裸分隔线也算进去。单文件两趟，因为"这份文本里有没有别的标记"
    是分隔线能否定罪的前提。
    """
    lines = text.splitlines()
    hits = [
        (number, line)
        for number, line in enumerate(lines, start=1)
        if _is_unconditional_marker(line)
    ]
    if not hits:
        return []
    hits.extend(
        (number, line)
        for number, line in enumerate(lines, start=1)
        if line.rstrip("\r\n") == SEPARATOR
    )
    return sorted(hits)


def tracked_files() -> list[Path]:
    """受版本控制的文件清单，走``git ls-files -z``。

    用``-z``而不是按行切：本仓文件名大量是中文，默认``core.quotepath``
    会把它们转义成八进制形态，按行切再去掉引号是在**猜**原文件名。

    git不可用时**不是跳过而是失败**：一道"因为拿不到清单所以通过"的门，
    与没有这道门等价，且更坏——它会在报告里显示为绿。
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    )
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    return [ROOT / name for name in names]


def scan(paths: list[Path]) -> tuple[dict[Path, list[tuple[int, str]]], int]:
    """扫给定文件，返回``(命中表, 实扫文件数)``。

    非UTF-8的文件（二进制资产）跳过并**不计入实扫数**——
    实扫数是"零执行绝不pass"那条门的覆盖面证据，掺进跳过的文件就不再是证据。
    """
    hits: dict[Path, list[tuple[int, str]]] = {}
    scanned = 0
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        found = find_conflict_markers(text)
        if found:
            hits[path] = found
    return hits, scanned


def main(argv: list[str]) -> int:
    paths = [Path(name).resolve() for name in argv] if argv else tracked_files()
    hits, scanned = scan(paths)
    if not argv and scanned == 0:
        print("check-conflict-markers: 扫了0个文件——这是空跑不是通过", file=sys.stderr)
        return 2
    for path, found in sorted(hits.items()):
        for number, line in found:
            try:
                shown = path.relative_to(ROOT)
            except ValueError:
                shown = path
            print(f"{shown}:{number}: {line.rstrip()}", file=sys.stderr)
    if hits:
        print(
            f"check-conflict-markers: {len(hits)}个文件带未解合并冲突标记"
            "——**保留双方也要真的删掉标记**",
            file=sys.stderr,
        )
        return 1
    print(f"check-conflict-markers: {scanned} scanned, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
