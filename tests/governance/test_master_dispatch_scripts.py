"""两个外派入口之间的一致性门（决策0084第七节，2026-08-18立）。

## 为什么要有这道门：一个缺陷修好了，它的复制品不会跟着修

`tools/master/`下有两个入口，第二个（`run_on_master.sh`，提交40ea0c2）
是**照着第一个（`run_accept_on_master.sh`）写出来的**——它自己的文件头写着
"不是新发明一套同步，是同一套的第二个入口"。

于是每一次修都只修了一边，而且**两个方向都发生过**：

| 缺陷 | 修在哪 | 另一个入口 |
|---|---|---|
| bundle只送`HEAD`，远端没有`main`这个ref | 5288bb5修了`run_accept` | `run_on_master`原样带着，**2026-08-18才补** |
| 并发作业互删检出目录（实测丢过一次结果） | 5fe49ab修了`run_on_master` | `run_accept`原样带着，**2026-08-18才补** |

两次的后果都不是"报个错"，是本仓反复记的那一类：
第一个给出**一份读起来像失败、实际全绿的回执**，
第二个给出**结果直接丢失而日志里没有任何痕迹**。

**这道门不判脚本写得对不对，它判两个入口有没有各自带上那些用事故换来的形制选择。**

## 三条判据各自的出处

1. **bundle必须带`main`**——`check_worktree_env.py`第④条要解析基线分支；
2. **落点必须带运行号**——同一SHA上的并发作业不许共用一个检出目录或一个`/tmp`名字；
3. **不许出现`bash -lc "…"`那种嵌套引号的作业体**——914cb69（嵌套heredoc把`srun`那行拆坏）
   与18bcfab（单引号把`$PATH`写死成字面量、外部程序全找不到、作业静默挂死19分钟）
   是同一个病的两次发作：**多一层引号就多一次静默出错的机会**。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MASTER_DIR = ROOT / "tools/master"

#: 两个入口。**新增第三个入口时它自动进这张表**——用glob而不是写死名字，
#: 正是因为本门要防的就是"照着写了一个新的，而它没带上那些形制选择"。
ENTRIES = sorted(MASTER_DIR.glob("*.sh"))


def test_there_are_at_least_two_entries_to_compare():
    """判据本身也要被验：一张空表会让下面三条参数化门**一条都不跑**而全绿。"""

    assert len(ENTRIES) >= 2, f"外派入口少于两个，本门失去意义：{[p.name for p in ENTRIES]}"


def bundles_the_baseline_ref(source: str) -> bool:
    """`git bundle create <文件> ... main ...`——打包里必须有`main`这个ref。"""

    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "git bundle create" not in stripped:
            continue
        return re.search(r"\bmain\b", stripped.split("git bundle create", 1)[1]) is not None
    return False


def lands_under_a_run_tag(source: str) -> bool:
    """检出目录与`/tmp`落点都要带运行号。"""

    if "RUN_TAG=" not in source:
        return False
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.match(r'DIR="\$HOME/\$REMOTE_DIR/(.+)"$', stripped)
        if match and "$RUN_TAG" not in match.group(1):
            return False
    # `/tmp`上的每一个以SHA命名的落点都要带运行号，否则并发时互相删。
    for hit in re.findall(r"/tmp/[A-Za-z0-9_.\-$]+", source):
        if "$SHORT" in hit and "$RUN_TAG" not in hit:
            return False
    return True


def has_no_nested_quote_job_body(source: str) -> bool:
    """不许把作业体塞进`bash -lc "…"`。注释里提到它是允许的（那是在讲教训）。"""

    for line in source.splitlines():
        if line.strip().startswith("#"):
            continue
        if "bash -lc" in line:
            return False
    return True


def installs_parallel_test_dependency(source: str) -> bool:
    """master共享venv必须与``pyproject.toml``的并行验收依赖一致。"""

    return "pytest-xdist>=3.6,<4" in "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_every_master_entry_bundles_the_baseline_ref(entry: Path):
    assert bundles_the_baseline_ref(entry.read_text(encoding="utf-8")), (
        f"{entry.name}的bundle没带`main`——远端克隆出来没有基线分支，"
        "`check_worktree_env.py`第④条解析不出来就红，"
        "于是回执**读起来像失败、实际全绿**（提交5288bb5那次的形态）。"
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_every_master_entry_lands_under_a_run_tag(entry: Path):
    assert lands_under_a_run_tag(entry.read_text(encoding="utf-8")), (
        f"{entry.name}的落点没带运行号——同一SHA上并发发两个作业时，"
        "后发的那个会把先发的那个连同正在跑的进程一起删掉，"
        "**结果直接丢失而日志里没有任何痕迹**（提交5fe49ab那次的形态）。"
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_no_master_entry_uses_a_nested_quote_job_body(entry: Path):
    assert has_no_nested_quote_job_body(entry.read_text(encoding="utf-8")), (
        f"{entry.name}把作业体塞进了`bash -lc \"…\"`——多一层引号就多一次静默出错的机会。"
        "18bcfab那次是单引号把`$PATH`写死成字面量，外部程序全找不到，"
        "作业**静默挂死19分钟**。作业体要逐字写成一个文件。"
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_every_master_entry_installs_the_parallel_test_dependency(entry: Path):
    assert installs_parallel_test_dependency(entry.read_text(encoding="utf-8")), (
        f"{entry.name}没装pytest-xdist——本机门会并行、master却在收集前直接失败"
    )


# ---------------------------------------------------------------------------
# 必红：三条判据各喂一份植入的坏脚本
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("planted", "checker"),
    [
        ('git bundle create "$STAGE/pe.bundle" HEAD 2>/dev/null\n', bundles_the_baseline_ref),
        ('echo no bundle line at all\n', bundles_the_baseline_ref),
        ('DIR="$HOME/$REMOTE_DIR/$SHORT"\nRUN_TAG="x"\n', lands_under_a_run_tag),
        ('DIR="$HOME/$REMOTE_DIR/$SHORT-$RUN_TAG"\n', lands_under_a_run_tag),
        ('RUN_TAG="x"\nDIR="$HOME/$REMOTE_DIR/$SHORT-$RUN_TAG"\n'
         'push "$STAGE/x" "$HOST:/tmp/pe-accept-$SHORT.bundle"\n', lands_under_a_run_tag),
        ('srun bash -lc "cd x && $CMD"\n', has_no_nested_quote_job_body),
        ("pip install pytest ruff numpy\n", installs_parallel_test_dependency),
    ],
)
def test_the_three_checks_are_not_empty_gates(planted: str, checker):
    """**必须红**：每条判据都要有一份它必须判否的输入。

    第二条`bundles_the_baseline_ref`喂的是"整个bundle行都不见了"——
    那时函数走到末尾返回`False`，**这一条在验那个末尾的返回**。
    第四条`lands_under_a_run_tag`喂的是"目录带了运行号但`RUN_TAG=`没定义"，
    **在验那个前置检查**。
    """

    assert checker(planted) is False, f"这份植入的脚本应当被判否，但没有：{planted!r}"
