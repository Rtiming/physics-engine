"""`case/cantilever_self_weight`的conformance门（轴7规则3）。

**这是引擎第一次把一个教科书闭式解算对**：能量项 → 装配 → 牛顿求解 →
自重悬臂梁的端点挠度 `δ = qL⁴/(8EI)`，二阶收敛。

判据数全部来自清单。
"""

from __future__ import annotations

from pathlib import Path

from physics_engine.energies import (
    EnergyContext,
    EnergyRegistry,
    LinearBending,
    UniformGravity,
    clamped_chain_bending_stencils,
)
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium
from physics_engine.state import StateField, StateLayout

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/cantilever_self_weight/oracle.json", root=ROOT)
ENTRY = MANIFEST.oracles[0]


def _layout(node_count: int, name: str) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/{name}",
        fields=tuple(
            field
            for index in range(node_count)
            for field in (
                StateField(f"node{index}_x_mm", 1),
                StateField(f"node{index}_y_mm", 1),
                StateField(f"node{index}_z_mm", 1),
            )
        ),
    )


def _solve(segments: int):
    inputs = ENTRY.inputs
    length = inputs["length_mm"]
    stiffness = inputs["bending_stiffness_nmm2"]
    load = inputs["load_per_length_n_mm"]
    gravity = inputs["gravity_mm_s2"]

    nodes = segments + 1
    step_mm = length / segments
    layout = _layout(nodes, f"cantilever_n{segments}")

    # 节点载荷按梯形权（端点半权）；由力反推质量，因为重力项吃的是质量。
    masses = tuple(
        load * step_mm * (0.5 if index in (0, nodes - 1) else 1.0) / gravity * 1000.0
        for index in range(nodes)
    )
    context = EnergyContext(
        context_id="context/cantilever", node_masses_kg=masses,
        gravity_mm_s2=(0.0, -gravity, 0.0),
    )
    registry = EnergyRegistry(terms=(
        UniformGravity(),
        LinearBending(stencils=clamped_chain_bending_stencils(nodes, step_mm, stiffness)),
    ))
    initial = tuple(
        value for index in range(nodes) for value in (index * step_mm, 0.0, 0.0)
    )
    # 固支：节点0的y钉住（位置），且全链的x与z钉住——本案例是平面小挠度问题，
    # 轴向与面外自由度不参与。斜率条件由弯曲模板的幽灵节点承担，不靠再钉一个节点。
    fixed = frozenset(
        {3 * index for index in range(nodes)}
        | {3 * index + 2 for index in range(nodes)}
        | {1}
    )
    return solve_equilibrium(
        registry, context, layout, initial,
        fixed_indices=fixed, residual_tol_n=inputs["residual_tol_n"],
    ), nodes


def test_tip_deflection_converges_to_the_closed_form_at_second_order():
    truth = ENTRY.expected["tip_deflection_mm"]
    errors = []
    converged = []
    iterations = []
    for segments in ENTRY.inputs["refinements"]:
        result, nodes = _solve(segments)
        converged.append(result.converged)
        iterations.append(result.iterations)
        tip = abs(result.state.vector[3 * (nodes - 1) + 1])
        errors.append(abs(tip - truth))

    assert all(error > 0.0 for error in errors), (
        f"误差全为零——离散误差被别的东西吞了，收敛阶无从谈起：{errors}"
    )
    ratios = [a / b for a, b in zip(errors, errors[1:], strict=False)]
    low = ENTRY.expected["error_ratio_low"]
    high = ENTRY.expected["error_ratio_high"]
    for ratio in ratios:
        assert low <= ratio <= high, (
            f"收敛比{ratio!r}落在[{low}, {high}]之外——实测阶偏离二阶。逐档误差{errors}"
        )
    ENTRY.check_all({
        "tip_deflection_mm": truth,
        "error_ratio_low": low,
        "error_ratio_high": high,
        "all_refinements_converged": all(converged),
        "newton_iterations_within_bound":
            max(iterations) <= ENTRY.inputs["newton_iterations_bound"],
    })


def test_newton_reaches_the_exact_solution_in_one_or_two_steps():
    """总能量是位置的二次型，牛顿一步即达精确解——迭代数本身是有效判据。"""

    result, _ = _solve(ENTRY.inputs["refinements"][0])
    assert result.converged
    assert result.iterations <= ENTRY.inputs["newton_iterations_bound"]
    assert result.backtracks == 0, (
        f"二次能量上不该有任何回溯，实测{result.backtracks}次——"
        "回溯出现意味着牛顿方向不是下降方向，能量可能不再是二次型"
    )
