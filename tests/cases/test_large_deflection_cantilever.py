"""`case/large_deflection_cantilever`的conformance门（轴7规则3）。

**引擎第一次把一个几何非线性的闭式解算对**：几何精确弯曲（DER的
``κ = 2·tan(θ/2)``）→ 载荷分步 → 牛顿求解 → 端点位置对Bisshopp-Drucker 1945的
椭圆积分闭式，二阶收敛。

判据数全部来自清单；**本文件不复述闭式**（轴7规则：不得在测试里复述oracle公式）。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.energies import (
    AxialStretch,
    DiscreteElasticBending,
    EnergyContext,
    EnergyRegistry,
    PointLoad,
    clamped_chain_bending_vertices,
)
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium
from physics_engine.state import StateField, StateLayout

#: **本机批级**（`local_batch`），与案例页第五节和清单的`load_tier`一一对应
#: （pyproject：marker必须与案例页申报的级别一致）。`peer_fcl_distance`已经带着
#: 这个marker，但它在没有同行库环境时整体skip——所以本案例是batch档里
#: **第一个无条件真跑**的。它慢的原因是可申报的：四档网格 × 四个载荷步 ×
#: 每步约48次牛顿迭代，每次迭代一个稠密LU。
#: **不是可以靠调参省掉的慢**，是几何非线性的价钱。
pytestmark = pytest.mark.batch

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/large_deflection_cantilever/oracle.json", root=ROOT)
ENTRY = MANIFEST.oracles[0]
#: 牛顿最大迭代数。**比清单里的判据上界宽**：判据是"迭代数不该突增"，
#: 而这个是"给求解器多少机会"。两者混成一个数，门就分不清"变难了"与"跑不完"。
SOLVER_ITERATION_CAP = 90


def _layout(node_count: int) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/large_deflection_cantilever_n{node_count}",
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


#: 解过的网格缓存。求解是本案例的全部开销（最细档四个载荷步各约48次牛顿迭代），
#: 而两条门要看同一批解——**重解一遍不会多验出任何东西，只会把案例推出负载级**。
_SOLVED: dict[tuple[int, bool], tuple[tuple, int]] = {}


def _solve(segments: int, *, clamp_correction: bool = True):
    key = (segments, clamp_correction)
    if key not in _SOLVED:
        _SOLVED[key] = _solve_uncached(segments, clamp_correction=clamp_correction)
    return _SOLVED[key]


def _solve_uncached(segments: int, *, clamp_correction: bool = True):
    """解一档网格，返回``(逐载荷步的结果, 节点数)``。

    固支：**钉住节点0与节点1的全部分量**——钉住第一条边就同时钉住了位置与切向，
    不需要额外的斜率约束。面外（z）全钉：本案例是平面问题。

    载荷分步是**案例显式做的**，不是求解器偷偷做的（求解器申报过它没有载荷步
    生长，decisions/0027第四节）。从直链一步加到β=3牛顿不收敛。

    **端载荷由`PointLoad`直接施加**（decisions/0046）。此前它是用端部集中质量
    加重力凑出来的，非端点节点因此要带1e-15 kg的寄生质量——那个hack
    随`PointLoad`落地一并清除，`UniformGravity`退出本案例。
    质量因此不参与本案例的任何能量项，`EnergyContext`要求为正故取1.0 kg。
    **载荷分步现在改在能量项里做**：`PointLoad`是frozen的，所以每一步重建注册表；
    此前是靠改上下文里的集中质量做的，那本身就是那个hack的一部分。
    """

    inputs = ENTRY.inputs
    length = inputs["length_mm"]
    nodes = segments + 1
    step_mm = length / segments
    vertices = clamped_chain_bending_vertices(
        nodes, step_mm, inputs["bending_stiffness_nmm2"]
    )
    if not clamp_correction:
        # 必须红用来的那一档：固支顶点的Voronoi长度退回h。
        vertices = tuple(
            (left, middle, right, stiffness, step_mm)
            for left, middle, right, stiffness, _ in vertices
        )
    stretch = AxialStretch(edges=tuple(
        (index, index + 1, step_mm, inputs["axial_stiffness_n"])
        for index in range(segments)
    ))
    bending = DiscreteElasticBending(vertices=vertices)
    layout = _layout(nodes)
    vector = tuple(
        value for index in range(nodes) for value in (index * step_mm, 0.0, 0.0)
    )
    fixed = frozenset(
        {0, 1, 2, 3, 4, 5} | {3 * index + 2 for index in range(nodes)}
    )
    context = EnergyContext(
        context_id="context/large_deflection_cantilever",
        node_masses_kg=(1.0,) * nodes,
    )
    load_steps = inputs["load_steps"]
    results = []
    for step in range(1, load_steps + 1):
        force = -inputs["tip_force_n"] * step / load_steps
        registry = EnergyRegistry(terms=(
            PointLoad(loads=((nodes - 1, (0.0, force, 0.0)),)),
            stretch,
            bending,
        ))
        result = solve_equilibrium(
            registry, context, layout, vector, fixed_indices=fixed,
            residual_tol_n=inputs["residual_tol_n"],
            max_iterations=SOLVER_ITERATION_CAP,
        )
        results.append(result)
        vector = result.state.vector
        if not result.converged:
            break
    return results, nodes


def _tip_error(results, nodes: int) -> float:
    """端点位置对闭式的相对偏差（二维2范数，形制同WDS的B1判据）。"""

    length = ENTRY.inputs["length_mm"]
    vector = results[-1].state.vector
    x = vector[3 * (nodes - 1)] / length
    y = -vector[3 * (nodes - 1) + 1] / length
    truth_x = ENTRY.expected["tip_x_over_length"]
    truth_y = ENTRY.expected["tip_y_over_length"]
    return math.hypot(x - truth_x, y - truth_y) / math.hypot(truth_x, truth_y)


def test_the_tip_converges_to_the_elliptic_integral_closed_form_at_second_order():
    errors = []
    converged = []
    iterations = []
    for segments in ENTRY.inputs["refinements"]:
        results, nodes = _solve(segments)
        converged.append(all(result.converged for result in results))
        iterations.append(max(result.iterations for result in results))
        errors.append(_tip_error(results, nodes))

    assert all(converged), (
        f"有档次没收敛——**不收敛的解不许参与收敛阶比较**：{converged}，逐档误差{errors}"
    )
    assert all(error > 0.0 for error in errors), (
        f"误差全为零——离散误差被别的东西吞了，收敛阶无从谈起：{errors}"
    )
    low = ENTRY.expected["error_ratio_low"]
    high = ENTRY.expected["error_ratio_high"]
    ratios = [a / b for a, b in zip(errors, errors[1:], strict=False)]
    for ratio in ratios:
        assert low <= ratio <= high, (
            f"收敛比{ratio!r}落在[{low}, {high}]之外——实测阶偏离二阶。逐档误差{errors}；"
            "按decisions/0027与0029的教训，**先查固支顶点的Voronoi长度**（应为3h/2），"
            "再怀疑边界条件的阶次"
        )
    ENTRY.check_all({
        "tip_x_over_length": ENTRY.expected["tip_x_over_length"],
        "tip_y_over_length": ENTRY.expected["tip_y_over_length"],
        "tip_angle_rad": ENTRY.expected["tip_angle_rad"],
        "tip_relative_error_max": ENTRY.expected["tip_relative_error_max"],
        "error_ratio_low": low,
        "error_ratio_high": high,
        "linear_theory_relative_error": ENTRY.expected["linear_theory_relative_error"],
        "geometric_nonlinearity_margin": ENTRY.expected["geometric_nonlinearity_margin"],
        "all_refinements_converged": all(converged),
        "newton_iterations_within_bound":
            max(iterations) <= ENTRY.inputs["newton_iterations_bound"],
    })
    assert errors[-1] < ENTRY.expected["tip_relative_error_max"], (
        f"最细档误差{errors[-1]!r}超过上界{ENTRY.expected['tip_relative_error_max']!r}"
    )


def test_the_geometrically_exact_term_beats_small_deflection_theory_by_two_orders():
    """**这条门证明本案例真的在考几何非线性。**

    小挠度理论在同一载荷下给出的端点相对偏差是清单里的
    `linear_theory_relative_error`（实测49.1%）。几何精确项的偏差必须比它小
    至少`geometric_nonlinearity_margin`倍。若不然，这个案例就退化成一个
    "小挠度也能过"的案例——那样它验的就不是本块交付的东西了。
    """

    results, nodes = _solve(ENTRY.inputs["refinements"][-1])
    assert all(result.converged for result in results)
    error = _tip_error(results, nodes)
    margin = ENTRY.expected["linear_theory_relative_error"] / error
    assert margin >= ENTRY.expected["geometric_nonlinearity_margin"], (
        f"几何精确项只比小挠度理论好{margin!r}倍，未达"
        f"{ENTRY.expected['geometric_nonlinearity_margin']!r}倍——本案例失去了区分力"
    )


def test_dropping_the_clamp_voronoi_correction_costs_an_order():
    """**正向的必须红**：把固支顶点的Voronoi长度从3h/2退回h，收敛阶掉到一阶。

    3h/2不是配出来的数——它是把被钉死的第一条边所吞掉的那半格柔度做静态凝聚
    的结果（推导见`clamped_chain_bending_vertices`）。这条门把"退回h会掉到一阶"
    钉成永久断言，所以后人改动那个系数时不会只看到案例变红而不知为什么。

    实测：退回h时收敛比2.0582 / 2.0323（一阶），取3h/2时3.9612 / 4.0512（二阶）。
    加密阶梯在这里只取三档——它证的是"阶掉了"，不需要最细那一档的代价。
    """

    errors = []
    for segments in ENTRY.inputs["refinements"][:3]:
        results, nodes = _solve(segments, clamp_correction=False)
        assert all(result.converged for result in results), (
            f"n={segments}没收敛，这条门就无从判定阶次"
        )
        errors.append(_tip_error(results, nodes))
    ratios = [a / b for a, b in zip(errors, errors[1:], strict=False)]
    for ratio in ratios:
        assert 1.9 <= ratio <= 2.2, (
            f"退回h后的收敛比{ratio!r}不在一阶区间[1.9, 2.2]——逐档误差{errors}。"
            "要么3h/2那条推导变了，要么另有一处误差盖过了它"
        )
    assert min(errors) > ENTRY.expected["tip_relative_error_max"], (
        f"退回h后最细档误差{min(errors)!r}仍在判据上界之内——"
        "那说明上界太松，分不开二阶与一阶"
    )
