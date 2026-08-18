#!/usr/bin/env python3
"""热点画像——**M6的第一步，也是"改代码"的前置条件**。

本仓性能条款第一句是"优化先profile"（`AGENTS.md`性能条款节、spec/13第一节义务1：
"没有热点数据的优化提案不受理"）。`tools/bench.py`量的是**声明过的那几条预算**；
本脚本量的是**没被声明过的那部分**——`accept full`里真正花掉墙钟的是谁。

两者不重复：`bench.py`回答"申报的预算还守得住吗"，本脚本回答"下一刀该切哪"。

## 为什么用cProfile而不是采样器

cProfile是**确定性插桩**：调用次数一次就准，不受负载影响。墙钟占比会被插桩开销
拉偏（纯Python的小函数被放大），所以本脚本**同时报`ncalls`**——
调用次数是跨机器、跨负载都稳定的那一半证据，占比只作排序用。

## 用法

    .venv/bin/python tools/profile_hotspots.py suite --marker "not batch and not serverclass"
    .venv/bin/python tools/profile_hotspots.py suite --marker batch
    .venv/bin/python tools/profile_hotspots.py durations --marker batch

**必须在安静机器上跑**（本机Mac负载常年5—20，同一功能面实测量到93.7/121.5/225三个数）。
入口见`tools/master/run_profile_on_master.sh`。
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
#: 仓库根也要进path——`tests/test_bench_measurement.py`等会`from tools import bench`，
#: 而进程内跑pytest时`sys.path[0]`是本脚本所在的`tools/`而不是仓库根（实测：
#: 第一次上master跑当场2个收集错误`No module named 'tools'`）。
sys.path.insert(0, str(ROOT))


def _run_pytest_under_profile(
    marker: str, top: int, sort: str, callers: tuple[str, ...]
) -> int:
    import pytest

    argv = ["tests", "-q", "-p", "no:cacheprovider", "-m", marker]
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    code = pytest.main(argv)
    profiler.disable()
    elapsed = time.perf_counter() - started

    print(f"\n=== profile: pytest -m {marker!r} ===")
    print(f"墙钟（带插桩，比裸跑慢）：{elapsed:.1f} s；pytest退出码 {int(code)}")
    for key in ("tottime", "cumtime"):
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats(key).print_stats(top)
        print(f"\n--- 按 {key} 排序 前{top} ---")
        print(stream.getvalue())
    # 只看引擎自己的代码——标准库与pytest的行不是我们能改的那部分。
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(sort).print_stats("physics_engine", top)
    print(f"\n--- 只看 physics_engine，按 {sort} 排序 前{top} ---")
    print(stream.getvalue())
    # 归因：一个热点函数本身改不动时，**要改的是谁在调它**。
    for pattern in callers:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.print_callers(pattern)
        print(f"\n--- 谁在调 {pattern!r} ---")
        print(stream.getvalue()[:12000])
    return 0


def _run_durations(marker: str, top: int) -> int:
    argv = [
        sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
        "-m", marker, f"--durations={top}",
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    completed = subprocess.run(argv, cwd=ROOT, env=env, check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("suite", "durations"))
    parser.add_argument("--marker", default="not batch and not serverclass")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sort", default="tottime")
    parser.add_argument(
        "--callers",
        action="append",
        default=[],
        help="额外打印「谁在调它」的函数名正则；可重复。",
    )
    args = parser.parse_args(argv)
    if args.mode == "suite":
        return _run_pytest_under_profile(
            args.marker, args.top, args.sort, tuple(args.callers)
        )
    return _run_durations(args.marker, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
