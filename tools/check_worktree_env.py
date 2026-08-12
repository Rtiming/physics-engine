#!/usr/bin/env python3
"""并行开工自检：**证明这棵树验的是它自己的代码**。

    .venv/bin/python tools/check_worktree_env.py

## 为什么需要它

2026-08-12实测到一条机理：共享`.venv`里editable安装的`.pth`内容是
**主仓`src`的绝对路径**。于是站在worktree里跑

    .venv/bin/python -c "import physics_engine; print(physics_engine.__file__)"

打出来的是**主仓的包**，不是这棵树的。当时`.claude/worktrees/`下17个副本里
**14个**带着这条软链。

含义不是"少测了几条"，而是：

> **一个代理可以在副本里跑完整套门、全绿、交差，而它验的是别人的代码。**

`AGENTS.md`把这条写成过警告（"不建符号链接就会测到主仓的代码"），
但警告不是断言——**没有断言的纪律等于没有纪律**。本文件把它变成可执行的。

同一族的第二条：`tools/rtime-project-check.py`把路径含`.claude`的目录当跳过目录，
而worktree就住在`.claude/worktrees/`下，于是它在副本里**扫0个文件仍然报"通过"**
（主仓同一命令扫280个）。plans/09第六节第4条记着这条前科未修。

**"0错误/0文件"不是证据，是空跑。** 这也是本文件第二条断言。

## 四条断言

| # | 断言 | 挡住的事故 |
|---|---|---|
| 1 | `import physics_engine`解析到**本树**的`src/` | 代理验的是主仓的代码 |
| 2 | 可移植性校验扫到的文件数 **> 0** | 空跑冒充通过 |
| 3 | pytest收集到的用例数 **> 0** | conftest炸了/marker笔误导致整档静默不跑 |
| 4 | HEAD**不落后**基线分支 | 简报里点名的文件在起点上根本不存在（2026-08-05实测落后52个提交） |

第1条是要害。**其余三条是它的同族**——全都属于"绿了，但绿的不是你以为的那棵树"。

## 明确挡不住的

- **本树代码本身对不对**：那是`accept.py`的事，不是本文件的事；
- **合并之后的树**：0041第五节的纪律仍然成立——worktree里的绿不算数，
  合并后主仓再跑一遍才算数。本文件让"副本里的绿"变得**可信**，
  不让它变得**充分**。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 可移植性校验自报的扫描量。形如"扫描根: /...  (查了 280 个文件, git 仓库 1 个)"。
SCANNED_FILES = re.compile(r"查了\s*(\d+)\s*个文件")

#: pytest收集量。形如"992 tests collected in 0.13s"或"992 tests collected"。
COLLECTED_TESTS = re.compile(r"(\d+)\s+tests?\s+collected")

#: 收集量的下限。取1而不是取一个大数：本文件判的是**空跑**，不是覆盖率。
#: 判覆盖率要另立门，而一道兼判两件事的门，两件都判不准。
MIN_COLLECTED = 1


class CheckFailed(Exception):
    """一条断言没过。消息直接给使用者看，所以写成可照做的话。"""


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, check=False, timeout=300
    )


def assert_package_resolves_to_this_tree(root: Path) -> str:
    """断言1（要害）：``import physics_engine``必须解析到**本树**的``src/``。

    用子进程而不是直接import：本文件自己可能是被主仓的解释器跑的，
    而我们要问的是"**这棵树的开发环境**会import到谁"，不是"我现在import到谁"。
    """

    result = _run(
        [sys.executable, "-c", "import physics_engine, sys; sys.stdout.write(physics_engine.__file__)"],
        cwd=root,
    )
    if result.returncode != 0:
        raise CheckFailed(
            "import physics_engine 失败：\n"
            f"{result.stderr.strip()}\n"
            "—— 在这棵树里建`.venv`符号链接，并 export PYTHONPATH=\"$PWD/src\"。"
        )

    resolved = Path(result.stdout.strip()).resolve()
    expected_src = (root / "src").resolve()
    if expected_src not in resolved.parents:
        raise CheckFailed(
            f"import physics_engine 解析到了**别的树**：\n"
            f"  实际  {resolved}\n"
            f"  应在  {expected_src}/ 之下\n"
            "—— 这正是2026-08-12测到的那条：共享`.venv`的editable `.pth`写的是\n"
            "   主仓`src`的绝对路径。**在这棵树里 export PYTHONPATH=\"$PWD/src\"**，\n"
            "   否则你跑出来的绿验的是主仓的代码，不是你自己的。"
        )
    return str(resolved)


def assert_portability_check_is_not_empty(root: Path) -> int:
    """断言2：可移植性校验扫到的文件数必须>0。

    它在`.claude/worktrees/`下会扫0个文件**并且报"通过"**——
    那不是通过，是空跑。
    """

    checker = root / "tools" / "rtime-project-check.py"
    if not checker.is_file():
        raise CheckFailed(f"可移植性校验器不在：{checker}")

    result = _run([sys.executable, str(checker), ".", "--strict", "--no-git"], cwd=root)
    combined = result.stdout + result.stderr
    match = SCANNED_FILES.search(combined)
    if match is None:
        raise CheckFailed(
            "可移植性校验没报出扫描量——本门无法判断它是不是空跑。\n"
            f"原始输出：\n{combined.strip()[:800]}"
        )

    scanned = int(match.group(1))
    if scanned <= 0:
        raise CheckFailed(
            "可移植性校验**扫了0个文件**却不报错——这不是通过，是空跑。\n"
            "—— 这棵树多半住在`.claude/`下，而那是校验器的跳过目录。\n"
            "   把树挪到`.claude`之外，或者验收改在主仓合入后跑。"
        )
    return scanned


def assert_tests_are_collectable(root: Path) -> int:
    """断言3：pytest必须收集到用例。

    `accept.py`已经用退出码5挡"一条都没收集到"，但它只在**跑的时候**挡。
    开工自检要在**开工前**就把这条问出来——一个炸了的`conftest`
    会让代理干半天才发现测试从来没跑过。
    """

    result = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=root)
    combined = result.stdout + result.stderr

    #: pytest的"一条都没收集到"退出码。`accept.py`用同一个常数——
    #: 那边把它当"空档位"（申报过的档可以空），**这边一律当红**：
    #: 开工自检问的是"这棵树能不能测"，而一棵测不了的树没有"申报过的空档"这回事。
    if result.returncode == 5:
        raise CheckFailed(
            "pytest**一条用例都没收集到**（退出码5）——这棵树测不了任何东西。\n"
            "—— 检查`tests/`在不在、`testpaths`对不对、`conftest.py`是不是没导入成。"
        )

    match = COLLECTED_TESTS.search(combined)
    if match is None:
        raise CheckFailed(
            "pytest没报出收集量——多半是collect阶段就炸了。\n"
            f"原始输出：\n{combined.strip()[-1200:]}"
        )

    collected = int(match.group(1))
    if collected < MIN_COLLECTED:
        raise CheckFailed(
            f"pytest只收集到{collected}条用例（下限{MIN_COLLECTED}）——这是空跑。"
        )
    return collected


def assert_head_is_not_behind(root: Path, base: str) -> str:
    """断言4：HEAD不落后基线分支。

    2026-08-05实测：那一波的worktree起点**落后52个提交**，
    简报里点名的文件在起点上根本不存在，四条轨道各自`reset --hard`才开工。
    """

    verify = _run(["git", "rev-parse", "--verify", f"{base}^{{commit}}"], cwd=root)
    if verify.returncode != 0:
        raise CheckFailed(
            f"基线分支`{base}`在这棵树里解析不出来——本门无法判断起点。\n"
            f"用 --base 指定真实基线，或先把`{base}`取到本地。"
        )

    behind = _run(["git", "rev-list", "--count", f"HEAD..{base}"], cwd=root)
    if behind.returncode != 0:
        raise CheckFailed(f"git rev-list 失败：{behind.stderr.strip()}")

    count = int(behind.stdout.strip() or "0")
    if count > 0:
        raise CheckFailed(
            f"HEAD**落后`{base}` {count}个提交**。\n"
            "—— 简报里点名的文件可能在这个起点上根本不存在（2026-08-05实测落后52个）。\n"
            f"   先 git reset --hard {base} 再开工。"
        )

    head = _run(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    return head.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--base", default="main", help="基线分支名（默认main）")
    parser.add_argument("--root", default=None, help="被检查的树根（默认本文件所在仓）")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else ROOT

    checks = (
        ("包身份", lambda: f"import解析到 {assert_package_resolves_to_this_tree(root)}"),
        ("可移植性校验非空跑", lambda: f"扫了{assert_portability_check_is_not_empty(root)}个文件"),
        ("测试可收集", lambda: f"收集到{assert_tests_are_collectable(root)}条"),
        ("起点不落后", lambda: f"HEAD={assert_head_is_not_behind(root, args.base)}，不落后{args.base}"),
    )

    failures: list[tuple[str, str]] = []
    for name, run in checks:
        try:
            print(f"  [ok  ] {name}：{run()}")
        except CheckFailed as exc:
            failures.append((name, str(exc)))
            print(f"  [FAIL] {name}")

    if failures:
        print(file=sys.stderr)
        for name, message in failures:
            print(f"—— {name}", file=sys.stderr)
            print(message, file=sys.stderr)
            print(file=sys.stderr)
        print(
            f"worktree env: {len(failures)}/{len(checks)} 条断言未过——"
            "**这棵树跑出来的绿不可信，不要开工**。",
            file=sys.stderr,
        )
        return 1

    print(f"worktree env: {len(checks)}/{len(checks)} 条断言全过（树根 {root}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
