"""`case/euler_buckling`的conformance门（轴7规则3）。

**引擎第一次算对一个失稳载荷**：轴向压载（`PointLoad`）→ 预屈曲直链平衡态
→ 切线刚度的正定性 → 对载荷二分 → 临界载荷对`Fc = π²EI/(bL)²`，三种边界条件。

判据数全部来自清单；**本文件不复述闭式**（轴7规则：不得在测试里复述oracle公式）。
本文件里出现的``cos``/``sin``全部属于**试验形状**（Rayleigh-Ritz的测试函数），
不是判据——判据是"哪个形状给出最低的临界载荷"，那个答案在清单里。
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
)
from physics_engine.oracles import load_manifest
from physics_engine.solve import (
    solve_equilibrium,
    tangent_stiffness_is_positive_definite,
)
from physics_engine.state import State, StateField, StateLayout

#: **本机批级**（`local_batch`），与案例页第五节和清单的`load_tier`一一对应。
#: 慢的原因是可申报的：三种边界条件 × 四档网格 × 40次载荷二分，
#: 每次二分一个预屈曲牛顿解加一次带状Cholesky。
pytestmark = pytest.mark.batch

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/euler_buckling/oracle.json", root=ROOT)
ENTRY = MANIFEST.oracles[0]

BOUNDARY_CONDITIONS = ("pinned_pinned", "fixed_free", "fixed_fixed")


def _layout(node_count: int) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/euler_buckling_n{node_count}",
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


def _column(segments: int, condition: str, force_n: float, *, clamp_correction: bool = True):
    """建一档柱子。返回``(注册表, 上下文, 布局, 直链向量, 预屈曲钉法, 稳定性钉法)``。

    离散：等分链，轴向由`AxialStretch`承担、弯曲由`DiscreteElasticBending`承担、
    轴向压载由`PointLoad`施加在最后一个节点上、方向``−x``。
    面外（z）全钉：本案例是平面问题。

    **边界条件就是钉法**（这一层不猜，`solve_equilibrium`也不猜）：

    * 两端铰支：节点0的x与y钉住、节点n的y钉住（x自由，载荷在那里）；
    * 一端固支一端自由：节点0的x与y、节点1的y钉住——**钉住第一条边的方向就是固支**，
      而它的**长度仍自由**（柱子必须能轴向缩短）；
    * 两端固支：两端各钉两个节点的y，受载端的x自由（标准的"两端固支、一端可轴向滑动"）。

    固支顶点的Voronoi长度取``clamp_voronoi_factor × h``（decisions/0027、0029）。
    """

    inputs = ENTRY.inputs
    length = inputs["length_mm"]
    nodes = segments + 1
    step_mm = length / segments
    factor = inputs["clamp_voronoi_factor"] if clamp_correction else 1.0
    clamped_vertices = set()
    if condition in ("fixed_free", "fixed_fixed"):
        clamped_vertices.add(1)
    if condition == "fixed_fixed":
        clamped_vertices.add(segments - 1)
    vertices = tuple(
        (
            index - 1, index, index + 1, inputs["bending_stiffness_nmm2"],
            step_mm * (factor if index in clamped_vertices else 1.0),
        )
        for index in range(1, segments)
    )
    registry = EnergyRegistry(terms=(
        PointLoad(loads=((segments, (-force_n, 0.0, 0.0)),)),
        AxialStretch(edges=tuple(
            (index, index + 1, step_mm, inputs["axial_stiffness_n"])
            for index in range(segments)
        )),
        DiscreteElasticBending(vertices=vertices),
    ))
    layout = _layout(nodes)
    straight = tuple(
        value for index in range(nodes) for value in (index * step_mm, 0.0, 0.0)
    )
    out_of_plane = {3 * index + 2 for index in range(nodes)}
    if condition == "pinned_pinned":
        transverse = {1, 3 * segments + 1}
    elif condition == "fixed_free":
        transverse = {1, 4}
    else:
        transverse = {1, 4, 3 * (segments - 1) + 1, 3 * segments + 1}
    context = EnergyContext(
        context_id="context/euler_buckling",
        #: 质量**不参与本案例的任何能量项**（无重力项）；`EnergyContext`要求为正。
        node_masses_kg=(1.0,) * nodes,
    )
    every_transverse = {3 * index + 1 for index in range(nodes)}
    return (
        registry, context, layout, straight,
        frozenset(out_of_plane | every_transverse | {0}),
        frozenset(out_of_plane | transverse | {0}),
    )


def _prebuckle(segments: int, condition: str, force_n: float, *, clamp_correction: bool = True):
    """预屈曲的**直链**平衡态：把所有横向自由度钉住，只解轴向。

    这不是回避：直链在任何载荷下都**恰好**是平衡态（直链上弯曲梯度为零、
    轴向力沿x），钉住横向只是把这个已知事实用上，换来一个恒正定的一维问题。
    失稳发生在**放开横向之后**的切线刚度上，那一步在`_is_stable`里。
    """

    registry, context, layout, straight, pre_fixed, stability_fixed = _column(
        segments, condition, force_n, clamp_correction=clamp_correction
    )
    result = solve_equilibrium(
        registry, context, layout, straight, fixed_indices=pre_fixed,
        residual_tol_n=ENTRY.inputs["residual_tol_n"],
        max_iterations=ENTRY.inputs["newton_iterations_bound"],
    )
    return result, registry, context, layout, stability_fixed


def _is_stable(segments: int, condition: str, force_n: float, *, clamp_correction: bool = True) -> bool:
    result, registry, context, _layout_, stability_fixed = _prebuckle(
        segments, condition, force_n, clamp_correction=clamp_correction
    )
    assert result.converged, (
        f"预屈曲直链解没收敛（{condition}, n={segments}, F={force_n!r}）：{result.reason}"
    )
    return tangent_stiffness_is_positive_definite(
        registry, context, result.state, fixed_indices=stability_fixed
    )


_CRITICAL: dict[tuple[int, str, bool], float] = {}


def _critical(segments: int, condition: str, *, clamp_correction: bool = True) -> float:
    """离散临界载荷：对"切线刚度是否正定"二分。

    **这条路径不需要牛顿法穿过分岔点**——它问的是"直链在这个载荷下还是不是极小"，
    而不是"屈曲后的构型长什么样"。案例页第四节记着直接穿越会发生什么。
    """

    key = (segments, condition, clamp_correction)
    if key in _CRITICAL:
        return _CRITICAL[key]
    reference = ENTRY.expected[f"critical_load_{condition}_n"]
    low = ENTRY.inputs["bracket_low_factor"] * reference
    high = ENTRY.inputs["bracket_high_factor"] * reference
    assert _is_stable(segments, condition, low, clamp_correction=clamp_correction), (
        f"括号下端{low!r}已经不稳定——二分的前提不成立"
    )
    assert not _is_stable(segments, condition, high, clamp_correction=clamp_correction), (
        f"括号上端{high!r}仍然稳定——二分的前提不成立"
    )
    for _ in range(ENTRY.inputs["bisection_iterations"]):
        middle = 0.5 * (low + high)
        if _is_stable(segments, condition, middle, clamp_correction=clamp_correction):
            low = middle
        else:
            high = middle
    _CRITICAL[key] = 0.5 * (low + high)
    return _CRITICAL[key]


def _effective_length_factor(critical_load_n: float) -> float:
    """由实测临界载荷反解有效长度因子：``b = π·sqrt(EI/Fc)/L``。

    这是`Fc = π²EI/(bL)²`的**代数反解**，不是第二个判据——它让案例直接读出
    "两端铰支1、一端固支一端自由2、两端固支0.5"这三个数本身。
    """

    inputs = ENTRY.inputs
    return (
        math.pi * math.sqrt(inputs["bending_stiffness_nmm2"] / critical_load_n)
        / inputs["length_mm"]
    )


# ── 不可伸长扰动：试验形状（Rayleigh-Ritz的测试函数，不是判据） ──

#: 链节转角``φ_i``（i=0..n−1，取在链节中点）。前四个都对中点反对称，
#: 于是``Σ sin φ_i = 0``**逐项抵消**，远端精确回到y=0——两端铰支的容许场。
#: `cantilever`取``φ_0 = 0``，是一端固支一端自由的容许场。
_SHAPES = {
    "half_wave": lambda index, n: math.cos(math.pi * (index + 0.5) / n),
    "full_wave": lambda index, n: math.cos(2.0 * math.pi * (index + 0.5) / n),
    "parabola": lambda index, n: 1.0 - 2.0 * (index + 0.5) / n,
    "tent": lambda index, n: 1.0 if 2 * index < n else -1.0,
    "cantilever": lambda index, n: math.sin(math.pi * index / (2.0 * n)),
}


def _perturb(vector: tuple[float, ...], segments: int, shape: str, amplitude: float):
    """保长扰动：逐链节转过``φ_i``，**每一段的长度逐段不变**。

    保长是这条门的全部要害：轴向能量因此不变（不再有几何刚度那一路），
    失稳全部来自`PointLoad`随端点缩进而释放的功。
    """

    lengths = [vector[3 * (i + 1)] - vector[3 * i] for i in range(segments)]
    x, y = vector[0], vector[1]
    out = [x, y, 0.0]
    for index in range(segments):
        angle = amplitude * _SHAPES[shape](index, segments)
        x += lengths[index] * math.cos(angle)
        y += lengths[index] * math.sin(angle)
        out.extend((x, y, 0.0))
    return tuple(out)


def _delta_energy(segments: int, condition: str, force_n: float, shape: str) -> float:
    """保长扰动前后的**总能量差**。正=直链是极小，负=直链已经不是极小。"""

    result, registry, context, layout, stability_fixed = _prebuckle(
        segments, condition, force_n)
    assert result.converged, result.reason
    bent_vector = _perturb(
        result.state.vector, segments, shape, ENTRY.inputs["perturbation_amplitude_rad"]
    )
    # **扰动必须是这个边界条件的容许场**：被钉住的自由度一个都不许动。
    # 这条断言不是形式主义——必红实测里"模型悄悄换了边界条件"那一档，
    # 正是靠它才被本函数抓住（否则一个越界的试验场会给出一个看似合理的符号）。
    for index in sorted(stability_fixed):
        assert abs(bent_vector[index] - result.state.vector[index]) <= 1.0e-12, (
            f"{condition}: 试验形状{shape!r}动了被钉住的自由度{index}"
            f"（{bent_vector[index]!r} 对 {result.state.vector[index]!r}）——"
            "它不是这个边界条件的容许场，能量差判据在这里没有意义"
        )
    bent = State(layout=layout, vector=bent_vector)
    straight_energy, _, _ = registry.total(result.state, context)
    bent_energy, _, _ = registry.total(bent, context)
    return bent_energy - straight_energy


def _rayleigh_critical(segments: int, condition: str, shape: str) -> float:
    """试验形状``shape``给出的临界载荷：对``ΔU``的符号二分。

    变分原理：**任何**容许试验形状给出的临界载荷都不低于真值，
    等号只在它就是真模态时成立。所以这个数同时是"模态对不对"的度量。
    """

    reference = ENTRY.expected[f"critical_load_{condition}_n"]
    low = ENTRY.inputs["bracket_low_factor"] * reference
    high = 30.0 * reference
    for _ in range(ENTRY.inputs["bisection_iterations"]):
        middle = 0.5 * (low + high)
        if _delta_energy(segments, condition, middle, shape) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


_MEASURED: dict[str, object] = {}


def _measurements() -> dict[str, object]:
    """一次算齐全部判据量。**解过的都缓存**——重解不会多验出任何东西。"""

    if _MEASURED:
        return _MEASURED
    inputs = ENTRY.inputs
    finest = inputs["value_refinement"]
    mode_n = inputs["mode_refinement"]
    loads = {
        condition: _critical(finest, condition) for condition in BOUNDARY_CONDITIONS
    }
    ladder = [_critical(n, "pinned_pinned") for n in inputs["refinements"]]

    # 预屈曲缩短率：在连续解的临界载荷处解一次直链。
    reference = ENTRY.expected["critical_load_pinned_pinned_n"]
    result, _registry, _context, _layout_, _fixed = _prebuckle(
        mode_n, "pinned_pinned", reference
    )
    assert result.converged, result.reason
    span = result.state.vector[3 * mode_n] - result.state.vector[0]
    shortening = (inputs["length_mm"] - span) / inputs["length_mm"]

    # 定性层：保长扰动在0.7Fc处升能、在1.3Fc处降能。
    rises = []
    falls = []
    for condition, shape in (("pinned_pinned", "half_wave"), ("fixed_free", "cantilever")):
        reference_load = ENTRY.expected[f"critical_load_{condition}_n"]
        rises.append(_delta_energy(
            mode_n, condition, inputs["subcritical_load_factor"] * reference_load, shape) > 0.0)
        falls.append(_delta_energy(
            mode_n, condition, inputs["supercritical_load_factor"] * reference_load, shape) < 0.0)

    # 直链在超临界载荷下**仍然是收敛解**，但不再正定。
    stability = {}
    for name, factor in (("below", inputs["subcritical_load_factor"]),
                         ("above", inputs["supercritical_load_factor"])):
        load = factor * ENTRY.expected["critical_load_pinned_pinned_n"]
        registry, context, layout, straight, _pre, stability_fixed = _column(
            mode_n, "pinned_pinned", load)
        solved = solve_equilibrium(
            registry, context, layout, straight, fixed_indices=stability_fixed,
            residual_tol_n=inputs["residual_tol_n"],
            max_iterations=inputs["newton_iterations_bound"],
        )
        still_straight = all(
            solved.state.vector[3 * index + 1] == 0.0 for index in range(mode_n + 1)
        )
        positive = tangent_stiffness_is_positive_definite(
            registry, context, solved.state, fixed_indices=stability_fixed)
        stability[name] = (solved.converged, still_straight, positive)

    _MEASURED.update({
        "critical_load_pinned_pinned_n": loads["pinned_pinned"],
        "critical_load_fixed_free_n": loads["fixed_free"],
        "critical_load_fixed_fixed_n": loads["fixed_fixed"],
        "effective_length_factor_pinned_pinned": _effective_length_factor(loads["pinned_pinned"]),
        "effective_length_factor_fixed_free": _effective_length_factor(loads["fixed_free"]),
        "effective_length_factor_fixed_fixed": _effective_length_factor(loads["fixed_fixed"]),
        "discrete_critical_loads_pinned_pinned": ladder,
        "prebuckling_relative_shortening": shortening,
        "second_mode_ratio": (
            _rayleigh_critical(mode_n, "pinned_pinned", "full_wave")
            / _critical(mode_n, "pinned_pinned")
        ),
        "subcritical_energy_rises": all(rises),
        "supercritical_energy_falls": all(falls),
        "converged_and_stable_below_critical": stability["below"] == (True, True, True),
        "converged_but_unstable_above_critical": stability["above"] == (True, True, False),
        # 以下四条是**声明的界与区间**，不是被测量——判据在各自的测试里正向断言。
        "convergence_ratio_low": ENTRY.expected["convergence_ratio_low"],
        "convergence_ratio_high": ENTRY.expected["convergence_ratio_high"],
        "clamp_correction_error_advantage_min":
            ENTRY.expected["clamp_correction_error_advantage_min"],
        "clamp_uncorrected_ratio_low": ENTRY.expected["clamp_uncorrected_ratio_low"],
        "clamp_uncorrected_ratio_high": ENTRY.expected["clamp_uncorrected_ratio_high"],
        "first_mode_rayleigh_gap_max": ENTRY.expected["first_mode_rayleigh_gap_max"],
        "trial_shape_excess_min": ENTRY.expected["trial_shape_excess_min"],
    })
    return _MEASURED


def test_the_critical_load_matches_the_euler_closed_form_for_three_boundary_conditions():
    """**C档第3条的正身**：`Fc = π²EI/(bL)²`，b=1 / 2 / 0.5三种边界条件。

    b不是抄来的——生成器对每种边界条件的**特征方程求根**，``b = π/u``
    （两端固支那条还比较了另一支根``tan v = v``，因为"b用错"最常见的形态
    就是拿了大的那一支）。文献值Timoshenko & Gere 1961只作交叉验证。
    """

    ENTRY.check_all(_measurements())


def test_the_effective_length_factor_is_read_out_and_not_assumed():
    """三个b值必须**互不相同且各就各位**——一条防"三个都算成同一个"的门。

    如果柱子的边界条件没有真的进模型（比如钉法写错让三种退化成同一种），
    上一条门里三个Fc会同时偏，而这一条会立刻看出b挤在一起。
    """

    measured = _measurements()
    factors = [
        measured[f"effective_length_factor_{condition}"]
        for condition in BOUNDARY_CONDITIONS
    ]
    assert factors[1] > factors[0] > factors[2], (
        f"有效长度因子的**排序**不对（应当是 固支-自由 > 铰支-铰支 > 固支-固支）：{factors}"
    )
    assert factors[1] / factors[0] > 1.9 and factors[0] / factors[2] > 1.9, (
        f"三个b挤在一起——边界条件很可能没有真的进模型：{factors}"
    )


def test_the_critical_load_converges_to_the_continuum_at_second_order():
    """网格收敛：两端铰支的临界载荷对连续解**二阶**收敛。

    实测三档比值3.9916 / 4.0033 / 4.0226。判据写区间不写死为4
    （spec/12第4.3节：比阶不比单点）。
    """

    reference = ENTRY.expected["critical_load_pinned_pinned_n"]
    ladder = [_critical(n, "pinned_pinned") for n in ENTRY.inputs["refinements"]]
    errors = [abs(value - reference) / reference for value in ladder]
    assert all(error > 0.0 for error in errors), (
        f"误差全为零——离散误差被别的东西吞了，收敛阶无从谈起：{errors}"
    )
    low = ENTRY.expected["convergence_ratio_low"]
    high = ENTRY.expected["convergence_ratio_high"]
    for ratio in (a / b for a, b in zip(errors, errors[1:], strict=False)):
        assert low <= ratio <= high, (
            f"收敛比{ratio!r}落在[{low}, {high}]之外——逐档误差{errors}。"
            "按decisions/0027、0029、0046的教训，**先查固支顶点的Voronoi长度**，"
            "再怀疑边界条件的阶次；也要查轴向可伸长性给的那个不随n下降的常数偏置"
        )


def test_dropping_the_clamp_voronoi_correction_costs_an_order():
    """**正向的必须红**：固支顶点的Voronoi长度退回``h``，收敛阶掉到一阶。

    这是decisions/0027与0029那条教训的**第三例，而且换了一个物理问题**：
    前两次验的是静挠度，这次验的是**特征值**。同一个``3h/2``在两类问题上
    同时把阶次从一阶抬到二阶——那不像配出来的数。

    实测（一端固支一端自由）：退回``h``给+1.0551e-01 / +5.1371e-02 / +2.5342e-02
    （比值2.0539 / 2.0271，干净一阶）；取``3h/2``给−8.3796e-04 / −3.6007e-04 /
    −1.0900e-04，**每一档都小两个数量级**。
    """

    ladder = ENTRY.inputs["refinements"][:3]
    reference = ENTRY.expected["critical_load_fixed_free_n"]
    corrected = [
        abs(_critical(n, "fixed_free") - reference) / reference for n in ladder
    ]
    uncorrected = [
        abs(_critical(n, "fixed_free", clamp_correction=False) - reference) / reference
        for n in ladder
    ]
    low = ENTRY.expected["clamp_uncorrected_ratio_low"]
    high = ENTRY.expected["clamp_uncorrected_ratio_high"]
    for ratio in (a / b for a, b in zip(uncorrected, uncorrected[1:], strict=False)):
        assert low <= ratio <= high, (
            f"退回h后的收敛比{ratio!r}不在一阶区间[{low}, {high}]——逐档误差{uncorrected}。"
            "要么3h/2那条推导变了，要么另有一处误差盖过了它"
        )
    advantage = ENTRY.expected["clamp_correction_error_advantage_min"]
    for index, (bad, good) in enumerate(zip(uncorrected, corrected, strict=True)):
        assert bad / good >= advantage, (
            f"n={ladder[index]}档上3h/2只比h好{bad / good!r}倍，未达{advantage}倍——"
            "那说明这个案例分不开一阶与二阶"
        )


def test_the_discrete_chain_eigenvalue_is_hit_within_the_extensibility_offset():
    """**比连续解紧100倍的那条门**：实测值必须命中**离散链的精确特征值**。

    离散链的刚度矩阵是``tridiag(−1, 2, −1)``，特征值有闭式，于是离散误差
    在这条门里根本不出现，剩下的只有引擎与那个模型的差别——即轴向可伸长性。
    实测四档相对偏差全部落在``+Fc/EA``附近，扣掉它以后残差 ≤ 1.9e-10。
    """

    measured = _measurements()
    ENTRY.check("discrete_critical_loads_pinned_pinned",
                measured["discrete_critical_loads_pinned_pinned"])
    expected = ENTRY.expected["discrete_critical_loads_pinned_pinned"]
    offsets = [
        actual / want - 1.0
        for actual, want in zip(measured["discrete_critical_loads_pinned_pinned"],
                                expected, strict=True)
    ]
    assert all(offset > 0.0 for offset in offsets), (
        f"实测值落在离散闭式**之下**：{offsets}。可伸长的杆只会**抬高**临界载荷"
        "（预压缩使每段变短、弯曲刚度EI/h³随之变大），落在下面说明另有一处错"
    )


def test_a_transverse_perturbation_decays_below_the_critical_load_and_grows_above():
    """定性转变：**保长**横向扰动在0.7Fc处升能、在1.3Fc处降能。

    **这条门是本案例唯一验``energy()``表达式的门。** 扰动逐段保长，
    所以轴向能量不变；弯曲能一定上升；能否降能全看`PointLoad`随端点缩进
    释放的功——``E = −F·x``的符号写反，这条立刻红，而临界载荷那几条照绿
    （求解器只用梯度与Hessian，decisions/0029第六节）。

    **它也不依赖闭式值的准确性**：断言的是能量差的**符号翻转**，
    闭式只用来挑采样点，且0.7/1.3两侧各留30%余量。
    """

    inputs = ENTRY.inputs
    mode_n = inputs["mode_refinement"]
    for condition, shape in (("pinned_pinned", "half_wave"), ("fixed_free", "cantilever")):
        reference = ENTRY.expected[f"critical_load_{condition}_n"]
        below = _delta_energy(
            mode_n, condition, inputs["subcritical_load_factor"] * reference, shape)
        above = _delta_energy(
            mode_n, condition, inputs["supercritical_load_factor"] * reference, shape)
        assert below > 0.0, (
            f"{condition}: 亚临界({inputs['subcritical_load_factor']}Fc)处扰动没有升能"
            f"（ΔU={below!r}）——直链在这里本该还是极小"
        )
        assert above < 0.0, (
            f"{condition}: 超临界({inputs['supercritical_load_factor']}Fc)处扰动没有降能"
            f"（ΔU={above!r}）——**先查`PointLoad`的能量符号**：``E = −F·x``，"
            "外力做功是负势能"
        )


def test_the_first_buckling_mode_is_the_sine_half_wave():
    """模态层：正弦半波给出最低的临界载荷，其余试验形状都严格更高。

    变分原理保证任何容许试验形状给出的临界载荷 ≥ 真值，所以这条门有两半：

    * 正弦半波与实测离散Fc的相对间隙必须**极小**（实测5.9e-7）——它就是模态；
    * 其余形状必须**明显更高**（实测抛物线+16.0%、尖顶折线+712%、全波+298%），
      且全波那一档还要命中离散链的**二阶模态比**（连续极限4=2²）。
    """

    inputs = ENTRY.inputs
    mode_n = inputs["mode_refinement"]
    discrete = _critical(mode_n, "pinned_pinned")
    rayleigh = {
        shape: _rayleigh_critical(mode_n, "pinned_pinned", shape)
        for shape in ("half_wave", "full_wave", "parabola", "tent")
    }
    for shape, value in rayleigh.items():
        assert value >= discrete, (
            f"试验形状{shape!r}给出的临界载荷{value!r}低于实测离散Fc{discrete!r}——"
            "变分原理不允许这件事，所以红的是二分或者装配，不是这个形状"
        )
    gap = rayleigh["half_wave"] / discrete - 1.0
    assert gap <= ENTRY.expected["first_mode_rayleigh_gap_max"], (
        f"正弦半波与真模态的间隙{gap!r}超过上界——一阶屈曲模态不是正弦半波了"
    )
    excess = ENTRY.expected["trial_shape_excess_min"]
    for shape in ("full_wave", "parabola", "tent"):
        assert rayleigh[shape] / discrete - 1.0 >= excess, (
            f"试验形状{shape!r}只比一阶模态高{rayleigh[shape] / discrete - 1.0!r}，"
            f"未达{excess}——本案例失去了分辨模态的能力"
        )
    ENTRY.check("second_mode_ratio", rayleigh["full_wave"] / discrete)


def test_newton_converges_to_the_straight_saddle_above_the_critical_load():
    """**``converged=True``不是稳定性。**

    在1.3Fc处从**精确的直链**出发，牛顿一步收敛、残差达标、返回的解逐位仍然是
    直的——而切线刚度**不正定**。求解器没有错：``∇U = 0``确实成立，
    它只是落在鞍点上。分辨这件事的是`tangent_stiffness_is_positive_definite`，
    在它之前本仓**没有任何办法查这一条**（`solve.py`的适用域申报因此一直不可验证）。
    """

    measured = _measurements()
    assert measured["converged_and_stable_below_critical"], (
        "亚临界处直链要么没收敛、要么被判成不正定——那说明判别本身坏了"
    )
    assert measured["converged_but_unstable_above_critical"], (
        "超临界处直链**应当**是「收敛且不正定」。若它变成不收敛，"
        "说明预屈曲那一步被污染；若它变成正定，说明几何刚度没有进Hessian"
    )
