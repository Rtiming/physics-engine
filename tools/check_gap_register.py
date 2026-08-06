#!/usr/bin/env python3
"""缺口清册的防过期门：登记了欠账的决策记录必须被汇总进`docs/plans/07`。

    python tools/check_gap_register.py

## 为什么需要它

欠账在本仓的形态是散在决策记录里的"**触发条件：……**"。
实测（2026-08-05）：**13份决策记录 + 5个源码文件**各自挂着触发条件，
全仓53处，**而没有任何一处能回答"我们一共欠了多少"**。

散着的欠账各自都写得很清楚，**合起来没人看得见**。
而没有汇总，"核对一遍"这个动作就无从做起——
同一天连着抓到的三个错全都属于"以为已经核对过"的那一类。

## 这道门判什么

**只判"有没有被汇总"，不判内容对不对。**

1. 每份含"触发条件"的决策记录，其编号必须**作为第六节表格的一行**出现；
2. 清册那张表引用的决策记录必须真的存在（防止改名后留下幽灵条目）。

### 第1条为什么要认表格行，不能认子串

**第一版认的是子串，必红当场把它打红了**——把第六节里0042那一行整行删掉，
门**照样绿**，因为"0042"在别处以"同0042"出现过。

这正是决策0049第六节记过的那个形态：
`peer_fcl_distance`落地后长期挂在"在建"那句话里，
案例页校验器只要求案例名"出现在页上"，于是**门认得"被提到"，认不得"被登记"**。

**我在写这道门时犯了同一个错，而必红用例把它抓住了**——
这条经历本身就是"每个门要红过"存在的理由：
**一道没被注错验过的门，你不知道它在验什么。**

所以判据改成：编号必须是**表格行的第一个单元格**。
"被提到"进不了第一个单元格，"被登记"才进得去。

内容的正确性靠人读，而**人只有在东西被汇总到一页上时才读得动**。
一道试图判断"这条欠账描述得对不对"的门会立刻变成自然语言处理，
然后被关掉——**门要挑它判得准的事情判**。

## 明确挡不住的

**源码里的触发条件**（`energies.py`、`modelgen.py`、`loops.py`、
`optics/`两处）。源码里的登记形态自由（写在docstring、`#:`注释、行内注释里），
硬扫会误红，**而会误红的门最终会被加豁免然后拆掉**。
清册第七节登记了这一条，触发条件：源码里的触发条件超过10处时给它也上门。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "docs" / "decisions"
REGISTER = ROOT / "docs" / "plans" / "07_缺口清册_20260805.md"

#: 决策记录里"我登记了一条欠账"的标记。选它是因为它是本仓既有的、
#: 用得最一致的那个词（53处），不是为这道门新发明的约定。
DEBT_MARKER = "触发条件"


def decisions_with_debt() -> list[Path]:
    return sorted(
        path
        for path in DECISIONS.glob("*.md")
        if DEBT_MARKER in path.read_text(encoding="utf-8")
    )


def decision_number(path: Path) -> str:
    """``0042_电磁域…md`` → ``0042``。清册按编号引用，不抄整个文件名。"""

    return path.name.split("_", 1)[0]


def registered_rows(register_text: str) -> set[str]:
    """清册第六节表格里**作为第一个单元格**出现的决策编号。

    只认第一个单元格是这道门的全部要害，理由见模块docstring——
    子串匹配分不出"被提到"与"被登记"。

    一行可以在第一格里写多个编号（``0032 / 0034``那种合并登记），
    所以第一格按非数字字符切开再取四位数字。
    """

    numbers: set[str] = set()
    for line in register_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = stripped.split("|")
        if len(cells) < 3:
            continue
        first = cells[1].strip().strip("`*")
        for token in "".join(ch if ch.isdigit() else " " for ch in first).split():
            if len(token) == 4:
                numbers.add(token)
    return numbers


def main() -> int:
    if not REGISTER.is_file():
        print(f"缺口清册不存在：{REGISTER.relative_to(ROOT)}", file=sys.stderr)
        return 2

    register_text = REGISTER.read_text(encoding="utf-8")
    registered = decisions_with_debt()
    if not registered:
        print(
            "扫到0份含触发条件的决策记录——这不是通过，是空跑"
            f"（{DECISIONS.relative_to(ROOT)}下一份都没扫到？）",
            file=sys.stderr,
        )
        return 2

    rows = registered_rows(register_text)
    if not rows:
        print("清册第六节一行表格都没解析出来——这不是通过，是空跑", file=sys.stderr)
        return 2

    missing = [path for path in registered if decision_number(path) not in rows]
    if missing:
        for path in missing:
            print(
                f"决策{decision_number(path)}登记了欠账但没进缺口清册第六节的表："
                f"{path.name}",
                file=sys.stderr,
            )
        print(
            f"—— 把它补成{REGISTER.relative_to(ROOT)}第六节表格的一行"
            "（**写在正文里提一句不算登记**）。",
            file=sys.stderr,
        )
        return 1

    known = {decision_number(path) for path in DECISIONS.glob("*.md")}
    ghosts = sorted(number for number in rows if number not in known)
    if ghosts:
        print(f"清册的表引用了不存在的决策记录：{ghosts}", file=sys.stderr)
        return 1

    print(f"gap register: {len(registered)} decisions with debt, all registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
