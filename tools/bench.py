#!/usr/bin/env python3
"""受控本机性能测量——**本脚本不是回退测试**。

照Drake `multibody/benchmarking/README.md`的形制明写这句：墙钟基准不作为
性能回退门。理由是实测的（research/05第四节）：共享runner的墙钟变异系数
CV=2.66%，配2%阈值假阳率45%，压到1%假阳需要约7%的门——等于测不到小回退。
MuJoCo的CI干脆不跑benchmark，Drake在README里明写其基准不承担回退检测。

**本仓的分工**：墙钟走本脚本，产报告进``work/bench/``（untracked）供人看；
真正进门的是``tests/perf/test_budgets.py``里的**确定性量**（字节数、模块数），
它们跨平台逐位稳定，不受负载影响——这正好绕开0014法则2禁止的那件事
（功能路径永不因宿主负载改变结果）。

用法：``.venv/bin/python tools/bench.py [--repeat N]``
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.canonical import FTS_PROFILE, canonical_file_bytes
from physics_engine.engine_facets import PERF_BASELINE_FACET, PERF_BASELINE_VERSION

EXAMPLE_SCENE = ROOT / "examples/collision_preview_cell.scene.json"


def _time_subprocess(argv: list[str], repeat: int) -> dict[str, float]:
    """冷进程计时：每次都是新解释器，取中位数与最小值。

    报最小值是有意的——最小值最接近"无干扰时的真实成本"，中位数受负载影响。
    两个都报，读的人自己判断。
    """

    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
        samples.append(time.perf_counter() - started)
    return {
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
        "repeat": repeat,
    }


def _load_average() -> float | None:
    try:
        return round(os.getloadavg()[0], 3)
    except (OSError, AttributeError):
        return None


def measure(repeat: int) -> dict:
    python = str(ROOT / ".venv/bin/python")
    wheels = sorted((ROOT / "dist").glob("*.whl")) if (ROOT / "dist").is_dir() else []
    source_bytes = sum(
        path.stat().st_size for path in sorted((ROOT / "src/physics_engine").glob("*.py"))
    )
    return {
        "facet": PERF_BASELINE_FACET,
        "facet_version": PERF_BASELINE_VERSION,
        "kind": "measurement_report",
        "load_average_1m_at_start": _load_average(),
        "wall_clock": {
            "import_physics_engine": _time_subprocess(
                [python, "-c", "import physics_engine"], repeat
            ),
            "pe_scene_validate": _time_subprocess(
                [python, "-m", "physics_engine.cli", "validate", str(EXAMPLE_SCENE)], repeat
            ),
            "pe_scene_check_collisions": _time_subprocess(
                [python, "-m", "physics_engine.cli", "check-collisions", str(EXAMPLE_SCENE)],
                repeat,
            ),
        },
        "deterministic": {
            "source_bytes": source_bytes,
            "wheel_bytes": wheels[-1].stat().st_size if wheels else None,
            "wheel_name": wheels[-1].name if wheels else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)

    report = measure(args.repeat)
    out_dir = ROOT / "work/bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latest.json"
    out_path.write_bytes(canonical_file_bytes(report, FTS_PROFILE))

    print("本报告不是回退测试（Drake形制）——墙钟只供人看，进门的是确定性量。")
    for name, stats in report["wall_clock"].items():
        print(f"  {name}: median={stats['median_s']}s min={stats['min_s']}s")
    determ = report["deterministic"]
    print(f"  source_bytes={determ['source_bytes']} wheel_bytes={determ['wheel_bytes']}")
    print(f"  load_average_1m={report['load_average_1m_at_start']}")
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
