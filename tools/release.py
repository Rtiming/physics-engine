#!/usr/bin/env python3
"""一键发版：验收→构建→入wheelhouse→打tag。快速演进期的低摩擦发版链路。

    .venv/bin/python tools/release.py            # 全流程
    .venv/bin/python tools/release.py --dry-run  # 只检查不落盘

流程（decisions/0010）：
1. 仓库必须干净（未提交改动拒发——发出去的wheel必须能对回一个commit）；
2. 版本一致性：pyproject == __init__.__version__，且wheelhouse里没有同版本
   （**版本不可覆盖**：改了字节必须跳版本，与轴1同律）；
3. `tools/accept.py full`必须PASS；
4. `uv build`出wheel+sdist，算SHA-256；
5. wheel拷入`~/wheelhouse`并提交推送（提交信息含版本与哈希）；
6. 引擎仓打tag `v<版本>`并推送。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = Path.home() / "wheelhouse"


def run(argv: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(argv)}\n{completed.stdout}{completed.stderr}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if run(["git", "status", "--porcelain"]).strip():
        raise SystemExit("工作区不干净：发版前先提交（wheel必须能对回一个commit）")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"$', pyproject, re.M).group(1)
    init_text = (ROOT / "src/physics_engine/__init__.py").read_text(encoding="utf-8")
    init_version = re.search(r'^__version__ = "([^"]+)"$', init_text, re.M).group(1)
    if version != init_version:
        raise SystemExit(f"版本不一致：pyproject {version} vs __init__ {init_version}")

    if not WHEELHOUSE.is_dir() or not (WHEELHOUSE / ".git").exists():
        raise SystemExit(f"wheelhouse不存在：git clone ts-orangepi:wheelhouse.git {WHEELHOUSE}")
    wheel_name = f"physics_engine-{version}-py3-none-any.whl"
    if (WHEELHOUSE / wheel_name).exists():
        raise SystemExit(f"版本{version}已在wheelhouse——版本不可覆盖，改了字节必须跳版本")

    print(f"[release] accept full for v{version} ...")
    accept = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/accept.py"), "full"], cwd=ROOT
    )
    if accept.returncode != 0:
        raise SystemExit("accept full未PASS，拒发")

    if args.dry_run:
        print(f"[release] dry-run OK：v{version}可发")
        return 0

    dist = ROOT / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    run(["uv", "build"])
    wheel = dist / wheel_name
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()

    shutil.copy2(wheel, WHEELHOUSE / wheel_name)
    run(["git", "add", wheel_name], cwd=WHEELHOUSE)
    run(
        ["git", "commit", "-q", "-m", f"physics-engine {version} sha256={digest}"],
        cwd=WHEELHOUSE,
    )
    branch = run(["git", "branch", "--show-current"], cwd=WHEELHOUSE).strip() or "main"
    run(["git", "push", "-u", "origin", branch], cwd=WHEELHOUSE)

    run(["git", "tag", f"v{version}"])
    run(["git", "push", "origin", f"v{version}"])
    print(f"[release] v{version} 已入wheelhouse并打tag，sha256={digest}")
    print(f"[release] 消费方：uv add 'physics-engine=={version}' --find-links ~/wheelhouse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
