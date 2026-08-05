#!/usr/bin/env python3
"""本仓的验收器——轴6（spec/07）在引擎仓自身的落地。

quick 30秒 / full 120秒；功能、计时、资源资格、仓库稳定四轴正交；超时绝不pass；
验收期间仓库身份变了→BLOCKED；零执行命令→BLOCKED。裁决逻辑全部是纯函数，
governance元测试直接测它们——"判据本身也要被验"。

**双档分家**（plans/02第二节T1）：quick只跑交互级（无marker的测试，spec/13零之二
第一级）；full = quick的全部命令 + 本机批级（`batch` marker）+ 案例页校验位。
`quick ⊆ full`由governance元测试守着。

**墙钟不是性能门**（research/05第四节）：共享runner墙钟CV=2.66%，配2%阈值假阳率
45%——墙钟做门测不到小回退。本验收器的计时轴只裁决**开发吞吐**（30/120预算，
spec/07规则1明写"预算只裁决开发吞吐，不裁决物理正确性"）；性能预算的墙钟测量走
`tools/bench.py`产报告，进门的是`tests/perf/`里的确定性整数量。

**争用韧性**（decisions/0014法则2）：负载敏感只允许存在于计时裁决——资源不合格时
计时判`NOT_EVALUATED`而非FAIL，功能结论一个字不改；实际杀进程的超时是
`HARD_TIMEOUT_FACTOR`倍预算的**活性护栏**，不是SLA。

回执写到``work/acceptance/<profile>-latest.json``（untracked、被ignore），
盖``engine_acceptance_receipt``面的名与版本（本仓自己的面清册）。

用法：``.venv/bin/python tools/accept.py quick|full [--timing-mode development|functional]``
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
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

#: 交互级命令（spec/13零之二第一级）：quick与full都跑。
#: pytest的``-m``表达式是双档分家的执行体——无marker=交互级。
QUICK_COMMANDS: tuple[tuple[str, ...], ...] = (
    (".venv/bin/python", "-m", "ruff", "check", "src", "tests", "tools"),
    (
        ".venv/bin/python", "-m", "pytest", "tests", "-q",
        "-m", "not batch and not serverclass",
    ),
)

#: 本机批级命令（<2分钟，与120秒全档预算同源）：只有full跑。
#: 服务器级（`serverclass`）两档都不跑——它按定义不在本机跑，走独立入口。
BATCH_COMMANDS: tuple[tuple[str, ...], ...] = (
    (".venv/bin/python", "-m", "pytest", "tests", "-q", "-m", "batch"),
)

#: 各档的命令集。full是quick的**严格超集**（governance测试守着这层关系）。
COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "quick": QUICK_COMMANDS,
    "full": QUICK_COMMANDS + BATCH_COMMANDS,
}

#: 案例页校验位（plans/02第四节"配``tools/check_case_pages.py``，缺一即红，进accept"）。
#: 该工具属T3轨道，本轨道只留位不占文件：**文件在就自动上膛**，不在就在回执里
#: 如实记一条``absent_optional_commands``——不假装绿，也不冒充红。
OPTIONAL_FULL_COMMANDS: tuple[tuple[str, ...], ...] = (
    (".venv/bin/python", "tools/check_case_pages.py"),
)

TIMEOUT_RETURNCODE = 124

#: pytest的"一条测试都没收集到"退出码。它既不是通过也不是失败——
#: 是**空档位**：本机批级今天还没有案例套件，`-m batch`选不中任何东西。
EMPTY_SELECTION_RETURNCODE = 5

#: 允许为空的命令白名单。**只有申报过的档位可以空**——交互级绝不许为空，
#: 那说明marker写错或测试没被收集，必须红（否则一个marker笔误就能让整档
#: 静默不跑而验收照绿，这正是"零执行命令→BLOCKED"要挡的那类事）。
MAY_BE_EMPTY: frozenset[tuple[str, ...]] = frozenset(BATCH_COMMANDS)

#: 活性护栏倍数：真正杀进程的超时=预算×本倍数，**不是SLA**。
#: 直接拿30/120当杀进程线，会让宿主负载改变**功能**结论（0014法则2禁止）；
#: 拿4倍当线，则只有真挂死才会被杀，而"慢"由计时轴单独裁决。
HARD_TIMEOUT_FACTOR = 4.0

#: 杀进程树时先SIGTERM、等这么久再SIGKILL。
KILL_GRACE_S = 2.0

#: 每核1分钟负载上限：超过即判资源不合格（计时不予评估，功能照跑不误）。
LOAD_PER_CPU_LIMIT = 1.5

RESOURCE_QUALIFIED = "QUALIFIED"
RESOURCE_UNQUALIFIED = "UNQUALIFIED"
RESOURCE_UNKNOWN = "UNKNOWN"

PERFORMANCE_EVALUABLE = "EVALUABLE"
PERFORMANCE_NOT_EVALUATED = "NOT_EVALUATED"

#: 执行树声明（轴6规则5）：受验字节树=代码+构建声明+被测输入+基准与预算件+
#: 基准工具自身。金标件（`benchmarks/`）与被测输入（`examples/`）都在树内——
#: 换了金标或换了输入还沿用上一轮的性能结论，是这条规则要挡的事。
EXECUTION_TREE_GLOBS: tuple[str, ...] = (
    "src/physics_engine/**/*.py",
    "pyproject.toml",
    "benchmarks/*.json",
    "examples/*.json",
    "tools/accept.py",
    "tools/bench.py",
    "tests/perf/*.py",
)

#: 字节码与缓存不进执行树（规则5明写排除）。
EXECUTION_TREE_EXCLUDED_PARTS = frozenset({"__pycache__"})
EXECUTION_TREE_EXCLUDED_SUFFIXES = (".pyc", ".pyo")


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
    #: 轴6规则5的执行树哈希。默认空串使既有三元构造保持可用（元测试沿用）。
    execution_tree_sha256: str = ""


@dataclass(frozen=True)
class ResourceObservation:
    """一次负载观测。``load_average_1m``在取不到时为None——不知道就说不知道。"""

    verdict: str
    cpu_count: int | None = None
    load_average_1m: float | None = None
    limit_per_cpu: float = LOAD_PER_CPU_LIMIT


@dataclass
class _Killed:
    """``run_command``内部用的可变标记，供元测试断言走了哪条路。"""

    killed_process_group: bool = False
    signals: list[str] = field(default_factory=list)


def execution_tree_digest(root: Path, globs: tuple[str, ...] = EXECUTION_TREE_GLOBS) -> str:
    """轴6规则5的执行树哈希。

    排序路径 + NUL分隔 + **内容长度前缀** + 内容字节，单流SHA-256。
    长度前缀是对规则原文的收紧：只用NUL分隔时，内容里含NUL的文件可以构造出
    两棵不同的树同哈希；加长度前缀后分帧无歧义。排除``__pycache__``与字节码。
    """

    paths: set[Path] = set()
    for pattern in globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if EXECUTION_TREE_EXCLUDED_PARTS & set(path.parts):
                continue
            if path.suffix in EXECUTION_TREE_EXCLUDED_SUFFIXES:
                continue
            paths.add(path)
    digest = hashlib.sha256()
    for path in sorted(paths):
        payload = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def classify_resource(load_average_1m: float | None, cpu_count: int | None) -> str:
    """轴6规则4的纯函数：负载资格判定。

    取不到负载（如Windows无``getloadavg``）→``UNKNOWN``，**按不合格处理计时**。
    冒充"合格"去发一个测不准的计时裁决，比承认不知道更糟。
    """

    if load_average_1m is None or not cpu_count:
        return RESOURCE_UNKNOWN
    if load_average_1m > LOAD_PER_CPU_LIMIT * cpu_count:
        return RESOURCE_UNQUALIFIED
    return RESOURCE_QUALIFIED


def observe_resources() -> ResourceObservation:
    """取一次负载观测（不纯，裁决部分在``classify_resource``里）。"""

    try:
        load_average_1m: float | None = os.getloadavg()[0]
    except (OSError, AttributeError):
        load_average_1m = None
    cpu_count = os.cpu_count()
    return ResourceObservation(
        verdict=classify_resource(load_average_1m, cpu_count),
        cpu_count=cpu_count,
        load_average_1m=load_average_1m,
    )


def resolve_commands(
    profile: str, root: Path
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """返回``(要跑的命令, 缺席的可选命令)``。可选位只在full档考虑。"""

    commands = list(COMMANDS[profile])
    absent: list[tuple[str, ...]] = []
    if profile == "full":
        for argv in OPTIONAL_FULL_COMMANDS:
            if (root / argv[-1]).is_file():
                commands.append(argv)
            else:
                absent.append(argv)
    return tuple(commands), tuple(absent)


def _kill_process_tree(process: subprocess.Popen, marker: _Killed) -> None:
    """杀掉整棵后代进程树，而不只是直接子进程。"""

    if os.name == "posix":
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            process.wait()
            return
        marker.killed_process_group = True
        for signal_number, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
            try:
                os.killpg(pgid, signal_number)
                marker.signals.append(name)
            except (ProcessLookupError, PermissionError):
                break
            if signal_number == signal.SIGTERM:
                try:
                    process.wait(timeout=KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    continue
        process.wait()
        # 组长退出不代表孙进程退出：补一发SIGKILL收尾（组已空则ProcessLookupError）。
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return
    marker.killed_process_group = True
    marker.signals.append("taskkill")
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(process.pid)],
        capture_output=True,
        check=False,
    )
    process.wait()


def run_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout_s: float | None,
    kill_process_tree: bool = True,
    _marker: _Killed | None = None,
) -> int:
    """跑一条命令；超时返回``TIMEOUT_RETURNCODE``并杀掉**整棵后代进程树**。

    ``subprocess.run(timeout=)``只终止直接子进程——它派生的孙进程被init收养后
    照跑不误。案例套件与同行库对拍必然派生子进程（`pytest`本身就在派生），
    这个洞会把"超时"变成"验收器退出了、机器上还挂着一堆进程"。

    做法：POSIX下``start_new_session=True``把子进程放进独立进程组，超时时
    ``os.killpg``整组；Windows下``taskkill /T /F``。``kill_process_tree=False``
    保留旧的裸``subprocess.run``路径，**只给元测试用来证明那个洞真的存在**。
    """

    marker = _marker if _marker is not None else _Killed()
    if not kill_process_tree:
        try:
            return subprocess.run(argv, cwd=cwd, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            return TIMEOUT_RETURNCODE
    if os.name == "posix":
        popen_kwargs: dict[str, object] = {"start_new_session": True}
    else:  # pragma: no cover - Windows分支在本机无法执行
        popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen(argv, cwd=cwd, **popen_kwargs)  # type: ignore[arg-type]
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process, marker)
        return TIMEOUT_RETURNCODE


def repository_identity(root: Path) -> RepositoryIdentity:
    """revision+dirty+内容指纹（tracked diff与untracked内容都进指纹）+执行树哈希。"""

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
        execution_tree_sha256=execution_tree_digest(root),
    )


def classify(
    results: tuple[CommandResult, ...],
    *,
    budget_s: float,
    timing_mode: str,
    total_elapsed_s: float,
    resource: str = RESOURCE_QUALIFIED,
    may_be_empty: frozenset[tuple[str, ...]] = MAY_BE_EMPTY,
) -> tuple[str, str, str]:
    """纯函数裁决：返回``(overall, functional, timing)``。

    超时（returncode 124）→functional=NOT_COMPLETED且（development下）
    timing=FAIL，双重挡死pass；零执行命令→BLOCKED。

    轴6规则4：资源不合格/不可知→timing=NOT_EVALUATED（**不是FAIL**），
    功能结论一个字不改，overall只跟着功能走。

    空档位（returncode 5）只对``may_be_empty``里申报过的命令视同通过；
    未申报的命令选不中测试仍是FAIL。
    """

    if not results:
        return ("BLOCKED", "NOT_EVALUATED", "NOT_EVALUATED")
    codes = [
        0
        if (result.returncode == EMPTY_SELECTION_RETURNCODE and result.argv in may_be_empty)
        else result.returncode
        for result in results
    ]
    timed_out = any(code == TIMEOUT_RETURNCODE for code in codes)
    if timed_out:
        functional = "NOT_COMPLETED"
    elif all(code == 0 for code in codes):
        functional = "PASS"
    else:
        functional = "FAIL"
    if timing_mode == "functional":
        timing = "NOT_ENFORCED"
    elif resource != RESOURCE_QUALIFIED:
        timing = "NOT_EVALUATED"
    else:
        timing = "PASS" if (total_elapsed_s <= budget_s and not timed_out) else "FAIL"
    overall = "PASS" if (functional == "PASS" and timing != "FAIL") else "FAIL"
    return (overall, functional, timing)


def classify_with_repository(
    verdicts: tuple[str, str, str],
    before: RepositoryIdentity,
    after: RepositoryIdentity,
) -> tuple[str, str, str, bool]:
    """仓库身份变了→整体BLOCKED，功能计时结论原样保留供人读。

    身份三元组已含执行树哈希，所以**受验树在验收期间被改**同样判BLOCKED
    （轴6规则5"测量前后比对，不稳→整体fail"在本仓的落地形式）。
    """

    changed = before != after
    overall, functional, timing = verdicts
    if changed:
        overall = "BLOCKED"
    return (overall, functional, timing, changed)


def classify_performance(
    *, execution_tree_stable: bool, resource: str, functional: str
) -> tuple[str, str]:
    """轴6规则5：执行树、资源资格、功能verdict**三者同绿性能才可裁决**。

    返回``(verdict, reason)``；不可裁决时给出机器可读的理由，绝不静默降级。
    注意本函数只回答"能不能裁决性能"，**不产生任何墙钟阈值判决**——
    墙钟走``tools/bench.py``的报告（research/05第四节）。
    """

    if not execution_tree_stable:
        return (PERFORMANCE_NOT_EVALUATED, "execution_tree_unstable")
    if resource != RESOURCE_QUALIFIED:
        return (PERFORMANCE_NOT_EVALUATED, f"resource_{resource.lower()}")
    if functional != "PASS":
        return (PERFORMANCE_NOT_EVALUATED, "functional_not_pass")
    return (PERFORMANCE_EVALUABLE, "")


def run_profile(profile: str, timing_mode: str) -> int:
    budget = BUDGETS[profile]
    commands, absent = resolve_commands(profile, ROOT)
    before = repository_identity(ROOT)
    resources = observe_resources()
    started = time.perf_counter()
    hard_deadline = (
        None if timing_mode == "functional" else started + budget * HARD_TIMEOUT_FACTOR
    )
    results: list[CommandResult] = []
    for argv in commands:
        command_started = time.perf_counter()
        remaining = None if hard_deadline is None else max(hard_deadline - command_started, 0.0)
        if remaining is not None and remaining == 0.0:
            results.append(CommandResult(argv, TIMEOUT_RETURNCODE, 0.0))
            continue
        code = run_command(argv, cwd=ROOT, timeout_s=remaining)
        results.append(
            CommandResult(argv, code, time.perf_counter() - command_started)
        )
    total_elapsed = time.perf_counter() - started
    after = repository_identity(ROOT)
    verdicts = classify(
        tuple(results), budget_s=budget, timing_mode=timing_mode,
        total_elapsed_s=total_elapsed, resource=resources.verdict,
    )
    overall, functional, timing, repo_changed = classify_with_repository(
        verdicts, before, after
    )
    performance, performance_reason = classify_performance(
        execution_tree_stable=before.execution_tree_sha256 == after.execution_tree_sha256,
        resource=resources.verdict,
        functional=functional,
    )
    receipt = {
        "facet": ACCEPTANCE_RECEIPT_FACET,
        "facet_version": ACCEPTANCE_RECEIPT_VERSION,
        "profile": profile,
        "timing_mode": timing_mode,
        "budget_s": budget,
        "hard_timeout_s": round(budget * HARD_TIMEOUT_FACTOR, 3),
        "elapsed_s": round(total_elapsed, 3),
        "overall": overall,
        "functional": functional,
        "timing": timing,
        "performance": performance,
        "performance_reason": performance_reason,
        "resource": resources.verdict,
        "resource_cpu_count": resources.cpu_count,
        "resource_load_average_1m": (
            None if resources.load_average_1m is None else round(resources.load_average_1m, 3)
        ),
        "resource_limit_per_cpu": resources.limit_per_cpu,
        "repository_stable": not repo_changed,
        "repository_before": before.__dict__,
        "repository_after": after.__dict__,
        "commands": [
            {"argv": list(r.argv), "returncode": r.returncode, "elapsed_s": round(r.elapsed_s, 3)}
            for r in results
        ],
        "absent_optional_commands": [list(argv) for argv in absent],
        "empty_selection_commands": [
            list(r.argv)
            for r in results
            if r.returncode == EMPTY_SELECTION_RETURNCODE and r.argv in MAY_BE_EMPTY
        ],
    }
    out_dir = ROOT / "work" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{profile}-latest.json"
    out_path.write_bytes(canonical_file_bytes(receipt, FTS_PROFILE))
    print(
        f"{profile}: overall={overall} functional={functional} timing={timing} "
        f"perf={performance}{f'({performance_reason})' if performance_reason else ''} "
        f"resource={resources.verdict} "
        f"elapsed={total_elapsed:.1f}s/{budget:.0f}s repo_stable={not repo_changed}"
    )
    for argv in absent:
        print(f"absent optional slot (not armed yet): {' '.join(argv)}")
    for entry in receipt["empty_selection_commands"]:
        print(f"declared-empty tier (no tests carry this marker yet): {' '.join(entry)}")
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
