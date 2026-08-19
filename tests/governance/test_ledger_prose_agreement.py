"""散文里的分数与清单算出来的数必须相等（决策0084第八节）。

## 它兑现的是哪一条登记

0056登记过一条："本次已人工对齐README、案例索引与plans/11为`12/42、6/13、0/6`，
但**自动门仍未落**；**下一次任一分子变化前**，先让门从清单计算值核对这些固定结构位置。"

基础设施批次四条轨都会动分子，**触发条件到了**。

## 为什么是"固定结构位置"而不是全文扫

全文扫会当场误红两处，而两处的形态不同、都不是缺陷：

1. **历史值**。README那一行里`6/13`之后紧跟着"2026-08-18曾报7/13，当天被否掉并退回"
   ——那个`7/13`是一段历史，不是一个待更新的读数；
2. **逐场景位数串**。`cases/README.md`写着`7/5/10/7/6/7`（0057冻结的每场景分母），
   里面的`7/6`不是"六个场景做完了七个"。

**一道会把历史记录判成陈旧的门会被关掉，然后这一条就白立了。**
所以`PROSE_ANCHORS`逐条写死"哪份文件、哪一行（按行内锚串认）、哪个分母"。

## 必红矩阵不在本文件

五条分支的必红用例住在`test_capability_ledger.py`——那份文件有两条元测试
（"每条分支各有自己的码""每个码都要有一条红用例走过"），
**把必红矩阵拆成两份就等于让那两条元测试各看半张表**。
本文件只留两件它管得住的：真仓对账，以及分数提取器自己的边界。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.check_capability_ledger import (  # noqa: E402
    PROSE_ANCHORS,
    Counts,
    assert_prose_matches_counts,
    check,
    first_fraction_with,
)

LEDGER = ROOT / "docs" / "capability_ledger.json"


@pytest.fixture(scope="module")
def counts() -> Counts:
    return check(LEDGER, ROOT)


def test_the_repository_prose_agrees_with_the_ledger(counts: Counts):
    assert_prose_matches_counts(counts, ROOT)


def test_the_anchor_table_is_not_empty():
    """**空表全绿**是这一族门最常见的失效方式（本仓已撞过：可移植性校验在worktree里扫0个文件）。"""

    assert len(PROSE_ANCHORS) >= 6, f"锚点表只有{len(PROSE_ANCHORS)}条，本门在空跑"


# ---------------------------------------------------------------------------
# 分数提取器自己的界：`7/5/10/7/6/7`里的`7/6`不许被读成"6个里做了7个"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "denominator", "expected"),
    [
        ("| **主｜用户六场景端到端** | … | **0/6** |", 6, 0),
        ("每场景位数`7/5/10/7/6/7`已由0057冻结", 6, None),
        ("每场景位数`7/5/10/7/6/7`已由0057冻结；主分母**12/42**", 42, 12),
        ("**6/13**（2026-08-18曾报7/13，当天退回）", 13, 6),
        ("这一行没有分数", 42, None),
    ],
)
def test_the_fraction_reader_respects_its_boundaries(line, denominator, expected):
    """第二条与第三条是同一串输入的两问：**串里的`7/6`不算，串外的`12/42`算。**

    第四条钉住"取该行第一个"——历史值写在真值之后是本仓三处散文的现状。
    """

    assert first_fraction_with(line, denominator) == expected


# ---------------------------------------------------------------------------
# 四条必红
# ---------------------------------------------------------------------------
