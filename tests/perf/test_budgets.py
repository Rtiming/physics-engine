"""性能预算的门——**只放确定性量，不放墙钟**。

spec/13第零节声明了延迟与体积预算，但在T1之前它们没有一条是可执行断言，
"显著回退=破坏性变更"是空话。本文件补上，形态按decisions/0018：

* **墙钟不进门**（research/05第四节：共享runner CV=2.66%，2%阈值假阳率45%）
  ——墙钟走`tools/bench.py`产报告；
* **进门的是跨平台逐位稳定的确定性量**：源码字节数（wheel体积的代理）、
  eager import模块数（冷启动成本的结构代理）、运行时依赖数（永远为0）。

这三条都不受宿主负载影响，因此可以放心进功能路径——0014法则2禁止的是
"负载改变功能结论"，确定性量不会。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/engine_budgets.baseline.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_baseline_declares_a_registered_facet(baseline):
    """基线自己也要盖面（轴1规则1，本仓自吃药）。"""

    from physics_engine.engine_facets import ENGINE_REGISTRY

    ENGINE_REGISTRY.assert_reader_compatible(baseline["facet"], baseline["facet_version"])


def test_source_bytes_stay_within_the_declared_ceiling(baseline):
    """wheel体积的确定性代理。破了要么优化，要么走决策记录改预算——不许默认破了不算破。"""

    gate = baseline["deterministic_gates"]["source_bytes"]
    measured = sum(
        path.stat().st_size for path in sorted((ROOT / "src/physics_engine").glob("*.py"))
    )
    assert measured <= gate["ceiling"], (
        f"源码字节 {measured} 超出声明上限 {gate['ceiling']}——"
        "这是spec/13预算的回退。优化，或走决策记录抬上限并在"
        "source_bytes_ceiling_history里记一行。"
    )


def test_raising_the_source_ceiling_must_be_ledgered(baseline):
    """上限不是"引擎必须保持小"的承诺——0015已裁决要把物理搬进来，源码注定大涨。

    门守的是**抬上限必须留痕**：现行上限必须等于历史末行，且每一行都带
    决策记录与理由。想悄悄抬一下过不去——那正是固定上限用久了会退化成的样子。
    """

    gate = baseline["deterministic_gates"]["source_bytes"]
    history = baseline["source_bytes_ceiling_history"]
    assert history, "上限历史不得为空"
    assert gate["ceiling"] == history[-1]["ceiling"], (
        f"现行上限 {gate['ceiling']} 与历史末行 {history[-1]['ceiling']} 不符——"
        "抬上限必须同批在source_bytes_ceiling_history里记一行。"
    )
    for entry in history:
        assert entry.get("decision"), f"上限变更缺决策记录：{entry}"
        assert entry.get("reason"), f"上限变更缺理由：{entry}"
        assert (ROOT / entry["decision"]).exists(), f"决策记录不存在：{entry['decision']}"


def test_eager_import_surface_does_not_grow_past_the_ceiling(baseline):
    """新模块进顶层eager import会同时抬高模块数与import冷启动延迟。"""

    gate = baseline["deterministic_gates"]["eager_import_modules"]
    code = (
        "import sys;before=set(sys.modules);"
        "import physics_engine;print(len(set(sys.modules)-before))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    measured = int(proc.stdout.strip())
    assert measured <= gate["ceiling"], (
        f"eager import模块数 {measured} 超出上限 {gate['ceiling']}——"
        "新模块请走惰性导入，不要进顶层__init__。"
    )


def test_runtime_dependencies_remain_empty():
    """0014零设施承诺 + 0016：NumPy只能在可选加速档，永不进dependencies。"""

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in text, (
        "运行时依赖不再为空——这破了0014的零设施承诺，必须走决策记录（0016取甲案："
        "核心零依赖，NumPy进optional-dependencies的加速档）。"
    )


def test_wheel_budget_is_recorded_per_version_not_a_single_hard_number(baseline):
    """0018的结论：30KB是对30940字节的口语化取整，改成随版本记账的曲线。

    门在这里守的是**记账义务**：发了新版本就必须有对应的一行，
    否则wheel会在没人看见的情况下长大。
    """

    history = baseline["wheel_budget_history"]
    assert history, "wheel预算历史不得为空"
    recorded = {entry["version"] for entry in history}
    from physics_engine import __version__

    assert __version__ in recorded, (
        f"当前版本 {__version__} 没有wheel体积记账——发版前补一行到"
        "benchmarks/engine_budgets.baseline.json的wheel_budget_history。"
    )


def test_the_fused_path_evaluates_each_edge_kernel_exactly_once():
    """确定性整数门：**融合的回归会在这里红**（decisions/0026第八节点名的那条）。

    `quantities`若做回"分别调energy/gradient/hessian"，边核求值会变成边数的
    3.0×——这个倍数是实测出来的，不是估的。整数计数跨平台逐位稳定，
    正是决策0018说的"进门的是确定性量、墙钟只产报告"那一类。
    """

    from physics_engine import energies
    from physics_engine.energies import (
        AxialStretch,
        EnergyContext,
        EnergyRegistry,
        UniformGravity,
    )
    from physics_engine.state import State, StateField, StateLayout

    nodes, step = 9, 4.0
    layout = StateLayout(
        layout_id="layout/counter",
        fields=tuple(
            field
            for index in range(nodes)
            for field in (
                StateField(f"node{index}_x_mm", 1),
                StateField(f"node{index}_y_mm", 1),
                StateField(f"node{index}_z_mm", 1),
            )
        ),
    )
    edges = tuple((i, i + 1, step, 700.0) for i in range(nodes - 1))
    registry = EnergyRegistry(terms=(UniformGravity(), AxialStretch(edges=edges)))
    context = EnergyContext(
        context_id="context/counter", node_masses_kg=(0.4,) * nodes,
        gravity_mm_s2=(0.0, -9806.65, 0.0),
    )
    state = State(
        layout=layout,
        vector=tuple(v for i in range(nodes) for v in (i * step, 0.1 * i * i, 0.0)),
    )

    calls = 0
    original = AxialStretch._edge_energy

    def counting(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    AxialStretch._edge_energy = counting
    try:
        registry.total(state, context, need_gradient=True, need_hessian=True)
    finally:
        AxialStretch._edge_energy = original
    assert calls == len(edges), (
        f"边核求值{calls}次，边数{len(edges)}——融合被做回去了。"
        "分开调energy/gradient/hessian会是3倍边数（decisions/0026实测）"
    )
    assert energies is not None
