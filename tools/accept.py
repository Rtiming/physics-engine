#!/usr/bin/env python3
"""本仓的验收器——轴6（spec/07）在引擎仓自身的落地。

quick 30秒 / full 120秒；功能、计时、仓库稳定三轴正交；超时绝不pass；
验收期间仓库身份变了→BLOCKED；零执行命令→BLOCKED。裁决逻辑全部是
纯函数，governance元测试直接测它们——"判据本身也要被验"。

回执写到``work/acceptance/<profile>-latest.json``（untracked、被ignore），
盖``engine_acceptance_receipt``面的名与版本（本仓自己的面清册）。

用法：``.venv/bin/python tools/accept.py quick|full [--timing-mode development|functional]``
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.canonical import FTS_PROFILE, canonical_file_bytes
from physics_engine.engine_facets import (
    ACCEPTANCE_RECEIPT_FACET,
    ACCEPTANCE_RECEIPT_VERSION,
)

#: 双档预算，轴6规则1冻结值。改这里必须走决策记录。
BUDGETS: dict[str, float] = {"quick": 30.0, "full": 120.0}

#: 各档的命令集。套件尚小，quick与full同集；套件长大后quick必须保持
#: full的子集（governance测试守着这层关系）。
COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "quick": (
        (".venv/bin/python", "-m", "ruff", "check", "src", "tests", "tools"),
        (".venv/bin/python", "-m", "pytest", "tests", "-q"),
    ),
    "full": (
        (".venv/bin/python", "-m", "ruff", "check", "src", "tests", "tools"),
        (".venv/bin/python", "-m", "pytest", "tests", "-q"),
    ),
}

TIMEOUT_RETURNCODE = 124


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    elapsed_s: float


@dataclass(frozen=True)
class RepositoryIdentity:
    revision: str
    dirty: bool
    working_tree_sha256: str


def repository_identity(root: Path) -> RepositoryIdentity:
    """revision+dirty+内容指纹（tracked diff与untracked内容都进指纹）。"""

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=root, capture_output=True, check=True
    ).stdout
    digest.update(diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root, capture_output=True, check=True,
    ).stdout
    for raw in sorted(part for part in untracked.split(b"\0") if part):
        digest.update(b"\0" + raw + b"\0")
        digest.update((root / raw.decode("utf-8")).read_bytes())
    return RepositoryIdentity(
        revision=revision,
        dirty=bool(status.strip()),
        working_tree_sha256=digest.hexdigest(),
    )


def classify(
    results: tuple[CommandResult, ...],
    *,
    budget_s: float,
    timing_mode: str,
    total_elapsed_s: float,
) -> tuple[str, str, str]:
    """纯函数裁决：返回``(overall, functional, timing)``。

    超时（returncode 124）→functional=NOT_COMPLETED且（development下）
    timing=FAIL，双重挡死pass；零执行命令→BLOCKED。
    """

    if not results:
        return ("BLOCKED", "NOT_EVALUATED", "NOT_EVALUATED")
    codes = [result.returncode for result in results]
    if any(code == TIMEOUT_RETURNCODE for code in codes):
        functional = "NOT_COMPLETED"
    elif all(code == 0 for code in codes):
        functional = "PASS"
    else:
        functional = "FAIL"
    if timing_mode == "functional":
        timing = "NOT_ENFORCED"
    else:
        timed_out = any(code == TIMEOUT_RETURNCODE for code in codes)
        timing = "PASS" if (total_elapsed_s <= budget_s and not timed_out) else "FAIL"
    overall = "PASS" if (functional == "PASS" and timing != "FAIL") else "FAIL"
    return (overall, functional, timing)


def classify_with_repository(
    verdicts: tuple[str, str, str],
    before: RepositoryIdentity,
    after: RepositoryIdentity,
) -> tuple[str, str, str, bool]:
    """仓库身份变了→整体BLOCKED，功能计时结论原样保留供人读。"""

    changed = before != after
    overall, functional, timing = verdicts
    if changed:
        overall = "BLOCKED"
    return (overall, functional, timing, changed)


def run_profile(profile: str, timing_mode: str) -> int:
    budget = BUDGETS[profile]
    before = repository_identity(ROOT)
    started = time.perf_counter()
    deadline = started + budget if timing_mode == "development" else None
    results: list[CommandResult] = []
    for argv in COMMANDS[profile]:
        command_started = time.perf_counter()
        remaining = None if deadline is None else max(deadline - command_started, 0.0)
        if remaining is not None and remaining == 0.0:
            results.append(CommandResult(argv, TIMEOUT_RETURNCODE, 0.0))
            continue
        try:
            completed = subprocess.run(argv, cwd=ROOT, timeout=remaining)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = TIMEOUT_RETURNCODE
        results.append(
            CommandResult(argv, code, time.perf_counter() - command_started)
        )
    total_elapsed = time.perf_counter() - started
    after = repository_identity(ROOT)
    verdicts = classify(
        tuple(results), budget_s=budget, timing_mode=timing_mode,
        total_elapsed_s=total_elapsed,
    )
    overall, functional, timing, repo_changed = classify_with_repository(
        verdicts, before, after
    )
    receipt = {
        "facet": ACCEPTANCE_RECEIPT_FACET,
        "facet_version": ACCEPTANCE_RECEIPT_VERSION,
        "profile": profile,
        "timing_mode": timing_mode,
        "budget_s": budget,
        "elapsed_s": round(total_elapsed, 3),
        "overall": overall,
        "functional": functional,
        "timing": timing,
        "repository_stable": not repo_changed,
        "repository_before": before.__dict__,
        "repository_after": after.__dict__,
        "commands": [
            {"argv": list(r.argv), "returncode": r.returncode, "elapsed_s": round(r.elapsed_s, 3)}
            for r in results
        ],
    }
    out_dir = ROOT / "work" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{profile}-latest.json"
    out_path.write_bytes(canonical_file_bytes(receipt, FTS_PROFILE))
    print(
        f"{profile}: overall={overall} functional={functional} timing={timing} "
        f"elapsed={total_elapsed:.1f}s/{budget:.0f}s repo_stable={not repo_changed}"
    )
    print(f"receipt: {out_path}")
    return 0 if overall == "PASS" else (2 if overall == "BLOCKED" else 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=sorted(BUDGETS))
    parser.add_argument(
        "--timing-mode", choices=("development", "functional"), default="development"
    )
    args = parser.parse_args(argv)
    return run_profile(args.profile, args.timing_mode)


if __name__ == "__main__":
    raise SystemExit(main())
