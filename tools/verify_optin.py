#!/usr/bin/env python3
"""选择进入档的统一入口——**把散在三个环境变量里的那条通道一次跑完**。

## 它补的是哪个洞

本仓有三条"指了才跑、不指明示skip"的通道（决策0073：真实资产永不进仓）：

| 变量 | 指向什么 | 谁在用 |
|---|---|---|
| ``PE_REAL_CENTERLINE_CSV`` | GCW导出的``centerline.csv``或它们的一个目录 | ``tests/cases/test_real_centerline_invariants.py``、``tests/test_model_tools.py`` |
| ``PE_REPLAY_CASE_RUNS`` | 消费方已发布的run目录树 | ``tests/test_provenance.py`` |
| ``PE_REPLAY_OUTPUT_TREE`` | 消费方的output树 | ``tests/test_run_package.py`` |

**2026-08-18之前没有任何一条命令能把它们一起跑起来**，于是这条通道的整体状态
从来没有被看过一眼——而`accept full`只会报"skipped"，**skip看起来和pass一样绿**。

同一天还实测到一个直接后果：`PE_REAL_CENTERLINE_CSV`在两侧有**互不兼容的两套约定**，
指目录一侧硬错、指单文件另一侧硬错，**不存在一个值能让这条通道全部跑过**。
那个缺陷之所以能活着，正是因为没有人把这条通道整体跑过一次。

## 三条纪律

1. **空跑不是通过**：一个变量都没解析到时**失败关闭**（返回码2），不报绿；
2. **逐条报"跑了还是跳了、以及为什么"**：skip不是失败，但它必须**被看见**；
3. **不写死任何用户主目录/盘符**（rtime-project范式）：路径只从环境变量或``--search-root``来。

用法::

    python tools/verify_optin.py                      # 只用已有的环境变量
    python tools/verify_optin.py --search-root <路径>  # 在这棵树下自动找语料
    python tools/verify_optin.py --require-all        # 有任何一条没解析到就红
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Channel:
    """一条选择进入的通道。"""

    #: 环境变量名。
    env: str
    #: 这条通道跑哪些测试文件。
    tests: tuple[str, ...]
    #: ``--search-root``下按这个glob找候选。
    discover_glob: str
    #: **候选还要过这一关**：它自己底下能递归找到这个glob才算数。
    #:
    #: 光按名字匹配是不够的，2026-08-18实测撞上：``**/output``在这台机器上
    #: 按字母序第一个命中的是**另一个项目的操作台输出目录**，
    #: 里面一个``run_manifest.json``都没有，于是测试红在"no run directories under …"上——
    #: **报错说的是语料不对，而真因是发现器挑错了树**。
    #: 一个只匹配名字的发现器会把"没找到"变成"找错了"，而后者更难查。
    validate_glob: str
    why: str


CHANNELS: tuple[Channel, ...] = (
    Channel(
        env="PE_REAL_CENTERLINE_CSV",
        tests=(
            "tests/cases/test_real_centerline_invariants.py",
            "tests/test_model_tools.py",
        ),
        discover_glob="**/handoff_runs",
        validate_glob="**/centerline.meta.json",
        why="真实工件的中心线导出（22列规范格式＋同目录的centerline.meta.json）",
    ),
    Channel(
        env="PE_REPLAY_CASE_RUNS",
        tests=("tests/test_provenance.py",),
        discover_glob="**/output/case-runs",
        validate_glob="*/*/run_manifest.json",
        why="消费方已发布的run目录树，用来验机械复读与篡改拒收",
    ),
    Channel(
        env="PE_REPLAY_OUTPUT_TREE",
        tests=("tests/test_run_package.py",),
        discover_glob="**/output",
        validate_glob="**/run_manifest.json",
        why="消费方的output树，用来验run package的装配与manifest绑定",
    ),
)


def discover(channel: Channel, search_root: Path) -> Path | None:
    """在``search_root``下找这条通道要的语料。

    两步：按名字找候选，**再逐个验它底下真有那种东西**，取第一个过关的。
    排序是为了可复现——同一棵树上两次跑必须选中同一个。
    """

    for candidate in sorted(search_root.glob(channel.discover_glob)):
        if not candidate.is_dir():
            continue
        if next(candidate.glob(channel.validate_glob), None) is not None:
            return candidate
    return None


def resolve(search_root: Path | None) -> dict[str, tuple[str | None, str]]:
    """逐条解析：已有环境变量优先，其次自动找。返回``{env: (值或None, 来源)}``。"""

    resolved: dict[str, tuple[str | None, str]] = {}
    for channel in CHANNELS:
        existing = os.environ.get(channel.env)
        if existing:
            resolved[channel.env] = (existing, "环境变量")
            continue
        if search_root is None:
            resolved[channel.env] = (None, "未设且未给--search-root")
            continue
        hit = discover(channel, search_root)
        resolved[channel.env] = (
            (str(hit), f"在{search_root}下找到") if hit else (None, f"在{search_root}下没找到")
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="选择进入档的统一入口")
    parser.add_argument("--search-root", default=None, help="在这棵树下自动找语料")
    parser.add_argument(
        "--require-all", action="store_true",
        help="有任何一条通道没解析到就判红（默认只要求至少一条）",
    )
    args = parser.parse_args(argv)

    search_root = Path(args.search_root).resolve() if args.search_root else None
    if search_root is not None and not search_root.is_dir():
        print(f"--search-root不是目录：{search_root}", file=sys.stderr)
        return 2

    resolved = resolve(search_root)
    env = dict(os.environ)
    tests: list[str] = []
    print("选择进入档，逐条：")
    for channel in CHANNELS:
        value, source = resolved[channel.env]
        if value:
            env[channel.env] = value
            tests.extend(channel.tests)
            print(f"  [跑] {channel.env}\n       = {value}\n       （{source}）{channel.why}")
        else:
            print(f"  [跳] {channel.env} —— {source}。{channel.why}")

    live = sum(1 for value, _ in resolved.values() if value)
    if live == 0:
        print(
            "\n**一条通道都没解析到——空跑不是通过。**\n"
            "给`--search-root`，或者自己设那三个环境变量。",
            file=sys.stderr,
        )
        return 2
    if args.require_all and live < len(CHANNELS):
        print(f"\n--require-all：{len(CHANNELS)}条里只解析到{live}条。", file=sys.stderr)
        return 1

    ordered = sorted(set(tests))
    print(f"\n跑：pytest {' '.join(ordered)}\n")
    completed = subprocess.run(
        [".venv/bin/python", "-m", "pytest", *ordered, "-q", "-rs"],
        cwd=ROOT, env=env, check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
