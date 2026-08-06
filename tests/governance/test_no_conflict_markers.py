"""合并冲突标记门——2026-08-05那次"六个绿灯但主干带着未解冲突"的执行体。

## 为什么有这道门

决策0048第六节把并行波次的冲突形态记成"**五次，形态全部相同（保留双方即可）**"，
并据此认为那一波已经收口。**实测不是**：收口之后核对主干，
`CHANGELOG.md`带着**三处嵌套**的未解冲突、`cases/README.md`带着一处，
连同`<<<<<<<`/`=======`/`>>>>>>>`一起被提交进了`main`，
而`accept.py` full当时是**全绿**的。

这不是"漏解了两个"。这是**没有任何一道门在看这件事**：

* `.pre-commit-config.yaml`里挂着`rtime-project-check`，但一、
  `.git/hooks/pre-commit`当时并未安装，二、
  **那个校验器本身也不查冲突标记**——它查的是路径可移植性。两层都不成立；
* `accept.py`两档跑的是功能测试与案例页校验，
  **没有任何一条断言看过"仓里的字节是不是一份合法的文本"**；
* 而冲突标记恰好**不会让任何东西红**：Markdown照样渲染，
  `.py`里才会语法错——**而这次撞上的两个文件都是`.md`**。

**所以这道门必须扫全部受版本控制的文本文件，不能只扫`.py`。**
这与`tests/perf/test_budgets.py`那个`glob`只扫顶层的洞是**同一类错误**：
门的覆盖面比仓的生长面窄，于是仓往门看不见的地方长。

## 判据形制：`=======`要条件化，其余三个无条件

朴素写法是三个标记一律见即红。**但`=======`在Markdown里是合法字节**——
setext式一级标题的下划线就是一串`=`，长度恰好7个的时候与冲突分隔线逐字节相同。
一道会对合法文档误红的门，最终会被加豁免、被加`# noqa`，然后被拆掉。

因此本门的规则是：

* `<<<<<<<`、`|||||||`（diff3的共同祖先段）、`>>>>>>>`——**行首见即红**，
  它们在任何合法文本里都没有第二种解释；
* `=======`——**只有当同一文件里还存在上面任意一个时才算**。
  理由：git写出的冲突**永远是成对的**，没有"只有分隔线的冲突"。
  这一条把setext标题的误报降到零，且**不放过任何一个真冲突**。

## 本文件为什么不含标记字面量

它要扫的东西正是它自己要写下的东西。若把标记按字面写进源码，
这道门会在自己身上红——于是只能给自己开豁免，
而**带豁免的门是没有门的**（豁免会被后来的人复制到别处）。
所以标记一律由`"<" * 7`一类的式子在运行期拼出来，
本文件的字节里**一个七连标记都没有**。

## 必红清单（AGENTS.md"每个门要红过"）

1. `test_detector_catches_the_nested_shape_that_actually_got_in`——
   拿**2026-08-05真实撞上的那个三层嵌套形态**喂进去必须被抓；
2. `test_bare_separator_alone_is_not_a_conflict`与
   `test_bare_separator_counts_when_paired`——**判据本身被验**：
   同一串`=======`，无伴时不算、有伴时算；
3. `test_scan_is_not_empty`与`test_scan_covers_the_two_files_that_were_hit`——
   **零执行绝不pass**（`accept.py`第89行同源）。一道扫了0个文件的门是空跑，
   而空跑的绿与真绿在报告里长得一模一样。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: 七连标记按运行期拼，理由见模块docstring末段（本文件不许含标记字面量）。
_OURS = "<" * 7
_BASE = "|" * 7
_THEIRS = ">" * 7
_SEPARATOR = "=" * 7

#: 无条件标记：行首出现即为冲突，合法文本里没有第二种解释。
#: git写的形态是"标记+空格+标签"（`<<<<<<< HEAD`），但手工mangle过的冲突
#: 可能只剩裸标记，故两种都收。
_UNCONDITIONAL = (_OURS, _BASE, _THEIRS)


def _is_unconditional_marker(line: str) -> bool:
    stripped = line.rstrip("\r\n")
    return any(
        stripped == marker or stripped.startswith(marker + " ")
        for marker in _UNCONDITIONAL
    )


def find_conflict_markers(text: str) -> list[tuple[int, str]]:
    """返回``[(行号从1起, 该行内容), ...]``。判据见模块docstring第二节。

    ``=======``的条件化在这里实现：先扫一遍无条件标记，
    **只有扫到了**才把裸分隔线也算进去。单文件两趟，
    因为"这份文本里有没有别的标记"是分隔线能否定罪的前提。
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
        if line.rstrip("\r\n") == _SEPARATOR
    )
    return sorted(hits)


def _tracked_files() -> list[Path]:
    """受版本控制的文件清单，走``git ls-files -z``。

    用``-z``而不是按行切：本仓文件名大量是中文，
    默认``core.quotepath``会把它们转义成``"\\346\\226\\207"``形态，
    按行切再去掉引号是在**猜**原文件名。``-z``给的是原始字节，不用猜。

    git不可用时**不是跳过而是失败**：一道"因为拿不到清单所以通过"的门，
    与没有这道门等价，且更坏——它会在报告里显示为绿。
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    return [ROOT / name for name in names]


def _scan_tracked_text() -> tuple[dict[Path, list[tuple[int, str]]], int]:
    """扫全部受版本控制的文本文件，返回``(命中表, 实扫文件数)``。

    非UTF-8的文件（二进制资产，如``.stl``）跳过并**不计入实扫数**——
    实扫数是给"零执行绝不pass"那条门用的，掺进跳过的文件就不再是覆盖面证据。
    """
    hits: dict[Path, list[tuple[int, str]]] = {}
    scanned = 0
    for path in _tracked_files():
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


# ---------------------------------------------------------------------------
# 门本体
# ---------------------------------------------------------------------------


def test_no_conflict_markers_in_tracked_files() -> None:
    """仓里不许有任何未解合并冲突。"""
    hits, _ = _scan_tracked_text()
    report = "\n".join(
        f"  {path.relative_to(ROOT)}:{number}: {line.rstrip()}"
        for path, found in sorted(hits.items())
        for number, line in found
    )
    assert not hits, (
        "受版本控制的文件里有未解合并冲突标记——**保留双方也要真的删掉标记**：\n"
        f"{report}"
    )


def test_scan_is_not_empty() -> None:
    """零执行绝不pass：扫了0个文件的绿是空跑，不是通过。"""
    _, scanned = _scan_tracked_text()
    assert scanned > 100, (
        f"只扫到{scanned}个文本文件，本仓受版本控制的文本文件远多于此——"
        "门的覆盖面塌了（八成是`git ls-files`在别的工作目录里跑）"
    )


def test_scan_covers_the_two_files_that_were_hit() -> None:
    """覆盖面回归：2026-08-05真正带标记进仓的是这两个**Markdown**文件。

    它们钉在这里是因为那次事故的根因之一就是"门只看`.py`"。
    这条断言让"把扫描面缩回源码"这个改动当场红。
    """
    scanned_paths = {
        path
        for path in _tracked_files()
        if path.is_file() and path.suffix in {".md"}
    }
    for name in ("CHANGELOG.md", "cases/README.md"):
        assert ROOT / name in scanned_paths, f"{name}不在扫描面里"


# ---------------------------------------------------------------------------
# 必红：判据本身要被验
# ---------------------------------------------------------------------------


def test_detector_catches_the_nested_shape_that_actually_got_in() -> None:
    """2026-08-05进仓的真实形态：三层嵌套，六条轨道往同一锚点各插一节。

    这个形态之所以要单独钉一条，是因为它**不是**教科书里那种
    "一个`<<<`一个`===`一个`>>>`"——三条`<<<`连着出现、
    分隔线与收尾标记散在几百行之外。按行判的检测器对它没有难度，
    但**按配对判**的检测器会在这里出错，所以它是选型的守门人。
    """
    text = "\n".join(
        [
            f"{_OURS} HEAD",
            f"{_OURS} HEAD",
            f"{_OURS} HEAD",
            "### T9 光学干涉",
            "",
            f"{_SEPARATOR}",
            "### T8 刚体（rigidbody）",
            f"{_THEIRS} worktree-agent-a419f1d9242e5236f",
        ]
    )
    found = find_conflict_markers(text)
    numbers = [number for number, _ in found]
    assert numbers == [1, 2, 3, 6, 8], f"漏抓或多抓：{found}"


def test_detector_catches_diff3_base_marker() -> None:
    """diff3风格会多写一段共同祖先，标记是`|||||||`——它同样是未解冲突。"""
    text = "\n".join(
        [
            f"{_OURS} ours",
            "a",
            f"{_BASE} base",
            "b",
            f"{_SEPARATOR}",
            "c",
            f"{_THEIRS} theirs",
        ]
    )
    assert [number for number, _ in find_conflict_markers(text)] == [1, 3, 5, 7]


def test_bare_separator_alone_is_not_a_conflict() -> None:
    """判据本身被验（反向）：Markdown的setext一级标题下划线不许误红。

    这条是本门最容易被写错的地方。误红一次，
    下一个人加的不是修正而是豁免——**门就是这样被拆掉的**。
    """
    text = "\n".join(["物理引擎", _SEPARATOR, "", "正文。"])
    assert find_conflict_markers(text) == []


def test_bare_separator_counts_when_paired() -> None:
    """同一串`=======`，有伴时必须算——否则条件化就成了漏网的借口。"""
    text = "\n".join([f"{_OURS} HEAD", "我们这边", _SEPARATOR, "他们那边", f"{_THEIRS} 分支"])
    assert [number for number, _ in find_conflict_markers(text)] == [1, 3, 5]


@pytest.mark.parametrize(
    "line",
    [
        ">>> import physics_engine",  # doctest提示符是三连，不是七连
        "<<< 三连也不算",
        "======",  # 六个
        "========",  # 八个
        "  " + "<" * 7 + " HEAD",  # 缩进过的不是git写的
    ],
)
def test_near_misses_do_not_trip_the_gate(line: str) -> None:
    """边界：只有**行首的七连**算数。差一个字符、差一层缩进都不算。"""
    assert find_conflict_markers("\n".join(["前文", line, "后文"])) == []
