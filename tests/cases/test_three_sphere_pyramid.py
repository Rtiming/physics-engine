"""conformance：三球金字塔临界摩擦（`cases/three_sphere_pyramid`）。

判据值一律从清单读，**不在本文件复述闭式**（轴7规则3）。

**引擎第一个多体接触案例**：球-球接触两端都是自由度，
切线刚度里第一次出现跨节点的耦合块。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    PenaltyNormalContact,
    PenaltySphereContact,
    TangentialStickSpring,
    build_contact_layout,
    coulomb_return_map,
)
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium

CASE = Path(__file__).resolve().parents[2] / "cases" / "three_sphere_pyramid"
MANIFEST = load_manifest(CASE / "oracle.json")

UP = (0.0, 0.0, 1.0)
#: 延拓的起点刚度。高刚度直接起步会撞"接触恰好为零"的奇异（已知失效清单第2条）。
CONTINUATION_START_N_PER_MM = 2.0e5

#: 残差地板的估计系数。地板来自``length − 2r``的**相消**：
#: 心距约``2r``而穿透只有``δ``，两者相减后剩下的有效位数由``2r·ε``决定，
#: 乘上刚度就是力的残差地板 ``≈ k·2r·ε``。
#:
#: **它与质量无关**——这一点是起草时踩出来的：固定容差在``m = 0.05 kg``下
#: 永远不收敛（残差1.557e-09卡在1e-9的容差外），而同一个容差在``m = 1.5 kg``
#: 下轻松通过。**不是求解器变差了，是力小了30倍而地板没动。**
#: 取4倍余量。
RESIDUAL_FLOOR_FACTOR = 4.0
MACHINE_EPSILON = 2.220446049250313e-16


def _oracle(identifier: str):
    for oracle in MANIFEST.oracles:
        if oracle.id == identifier:
            return oracle
    raise AssertionError(f"清单里没有{identifier}")


def _solve_pyramid(
    stiffness: float, *, radius_mm: float, mass_kg: float, gravity_mm_s2: float
):
    """解金字塔的静平衡，返回``(球间力, 地面法向, 地面切向矢量, 刚度)``。

    **走延拓**：从`CONTINUATION_START_N_PER_MM`逐个数量级升到目标刚度，
    每档拿上一档的解当初值。理由见案例页已知失效清单第2、3条——
    直接从刚性几何起步在高刚度下会撞奇异，而残差地板随刚度涨，
    固定容差会永远不收敛。
    """

    contact_layout = build_contact_layout(
        layout_id="layout/three_sphere_pyramid",
        node_count=3,
        declarations=(ContactDeclaration("left_ground"), ContactDeclaration("right_ground")),
    )
    context = EnergyContext(
        context_id="context/three_sphere_pyramid",
        node_masses_kg=(mass_kg,) * 3,
        gravity_mm_s2=(0.0, 0.0, -gravity_mm_s2),
    )
    contact_layout.assert_matches_context(context)

    weight_n = mass_kg * gravity_mm_s2 / 1000.0
    height = radius_mm * math.sqrt(3.0)
    fixed = frozenset({1, 4, 7} | set(range(9, contact_layout.layout.dof_count)))
    vector = contact_layout.initial_vector(
        (
            -radius_mm, 0.0, radius_mm - weight_n / CONTINUATION_START_N_PER_MM,
            radius_mm, 0.0, radius_mm - weight_n / CONTINUATION_START_N_PER_MM,
            0.0, 0.0, radius_mm + height - 1.0e-3,
        )
    )

    ladder = []
    step = CONTINUATION_START_N_PER_MM
    while step < stiffness * (1.0 - 1.0e-12):
        ladder.append(step)
        step *= 10.0
    ladder.append(stiffness)

    ground = spheres = stick = result = None
    for level in ladder:
        ground = PenaltyNormalContact(
            planes=(
                (0, (0.0, 0.0, 0.0), UP, level, radius_mm),
                (1, (0.0, 0.0, 0.0), UP, level, radius_mm),
            )
        )
        spheres = PenaltySphereContact(
            pairs=((0, 2, 2.0 * radius_mm, level), (1, 2, 2.0 * radius_mm, level))
        )
        stick = TangentialStickSpring(
            springs=(
                (0, (-radius_mm, 0.0, radius_mm), UP, level),
                (1, (radius_mm, 0.0, radius_mm), UP, level),
            )
        )
        registry = EnergyRegistry(terms=(UniformGravity(), ground, spheres, stick))
        result = solve_equilibrium(
            registry, context, contact_layout.layout, vector,
            fixed_indices=fixed,
            # 容差按地板估计给（``k·2r·ε``），不是拍一个数——见模块常量的注释
            residual_tol_n=RESIDUAL_FLOOR_FACTOR
            * level
            * 2.0
            * radius_mm
            * MACHINE_EPSILON,
            max_iterations=200,
        )
        assert result.converged, f"k={level:g}: {result.reason}"
        vector = result.state.vector

    return (
        spheres.contact_force_n(result.state)[0],
        ground.normal_force_n(result.state)[0],
        stick.tangential_force_n(result.state)[0],
    )


def test_the_static_decomposition_matches_the_closed_form():
    """``F = W/√3``、``N = 3W/2``、``T = W/(2√3)``。"""

    oracle = _oracle("oracle:pyramid/force_decomposition")
    force, normal, tangential = _solve_pyramid(
        oracle.inputs["stiffness_n_per_mm"],
        radius_mm=oracle.inputs["radius_mm"],
        mass_kg=oracle.inputs["mass_kg"],
        gravity_mm_s2=oracle.inputs["gravity_mm_s2"],
    )
    for actual, key in (
        (force, "sphere_contact_force_n"),
        (normal, "ground_normal_n"),
        (abs(tangential[0]), "ground_tangential_n"),
    ):
        tolerance = oracle.tolerances[key]
        assert actual == pytest.approx(
            oracle.expected[key], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        ), key


def test_the_pyramid_holds_above_the_critical_friction_and_collapses_below():
    """**本案例的主判据**（Chrono形制）：同一模型只改``μ``，两侧行为定性相反。"""

    oracle = _oracle("oracle:pyramid/critical_friction")
    critical = oracle.expected["critical_friction"]
    stiffness = oracle.inputs["stiffness_n_per_mm"]
    _, normal, tangential = _solve_pyramid(
        stiffness,
        radius_mm=oracle.inputs["radius_mm"],
        mass_kg=oracle.inputs["mass_kg"],
        gravity_mm_s2=oracle.inputs["gravity_mm_s2"],
    )

    tolerance = oracle.tolerances["critical_friction"]
    assert abs(tangential[0]) / normal == pytest.approx(
        critical, rel=tolerance.rel_tol, abs=tolerance.abs_tol
    ), "实测的T/N没有落在闭式μc上"

    verdicts = []
    for ratio in oracle.inputs["friction_ratios"]:
        outcome = coulomb_return_map(
            trial_force_n=tangential,
            normal_force_n=normal,
            friction_coefficient=critical * ratio,
            tangential_stiffness_n_per_mm=stiffness,
        )
        verdicts.append(outcome.is_stick)

    below = [
        verdict
        for ratio, verdict in zip(oracle.inputs["friction_ratios"], verdicts, strict=True)
        if ratio < 1.0
    ]
    above = [
        verdict
        for ratio, verdict in zip(oracle.inputs["friction_ratios"], verdicts, strict=True)
        if ratio > 1.0
    ]
    assert (not any(below)) is oracle.expected["collapses_below"], (
        f"μ低于临界时金字塔没塌：{verdicts}"
    )
    assert all(above) is oracle.expected["holds_above"], (
        f"μ高于临界时金字塔没撑住：{verdicts}"
    )


def test_the_penalty_compliance_error_is_first_order_in_the_stiffness():
    """**这条门证明那个偏差是模型的柔度，不是实现的错误。**

    斜面上法向固定、力精确到1 ulp；这里穿透改变接触几何本身，
    所以``T/N``带``O(1/k)``偏差。**刚度涨10倍偏差降约10倍**即一阶。

    区间取``[8, 12]``而不写死为10——一阶是渐近性质
    （与`harmonic_oscillator`那条"不写死为4"同源）。
    """

    oracle = _oracle("oracle:pyramid/compliance_is_first_order")
    critical = _oracle("oracle:pyramid/critical_friction").expected["critical_friction"]

    deviations = []
    for stiffness in oracle.inputs["stiffnesses_n_per_mm"]:
        _, normal, tangential = _solve_pyramid(
            stiffness,
            radius_mm=oracle.inputs["radius_mm"],
            mass_kg=oracle.inputs["mass_kg"],
            gravity_mm_s2=oracle.inputs["gravity_mm_s2"],
        )
        deviations.append(abs(abs(tangential[0]) / normal / critical - 1.0))

    shrinks = all(
        deviations[index] < deviations[index - 1] for index in range(1, len(deviations))
    )
    assert shrinks is oracle.expected["deviations_shrink"], (
        f"偏差没有逐档减小：{deviations}——比值区间那条门会在拿噪声算阶"
    )

    ratios = [
        deviations[index - 1] / deviations[index] for index in range(1, len(deviations))
    ]
    for ratio in ratios:
        assert oracle.expected["deviation_ratio_low"] <= ratio <= oracle.expected[
            "deviation_ratio_high"
        ], f"收敛阶不是一阶：比值{ratios}"


def test_the_critical_friction_does_not_depend_on_mass():
    """``W``在``T ≤ μN``两边约掉——**质量改100倍，临界值一动不动**。

    这一条与斜面那条同源，但在这里更要紧：金字塔的``μc``是**纯几何**的，
    若它随质量漂，说明某处把重力混进了几何。
    """

    oracle = _oracle("oracle:pyramid/critical_friction")
    critical = oracle.expected["critical_friction"]
    tolerance = oracle.tolerances["critical_friction"]

    for mass_kg in (0.05, 1.5, 5.0):
        _, normal, tangential = _solve_pyramid(
            oracle.inputs["stiffness_n_per_mm"],
            radius_mm=oracle.inputs["radius_mm"],
            mass_kg=mass_kg,
            gravity_mm_s2=oracle.inputs["gravity_mm_s2"],
        )
        assert abs(tangential[0]) / normal == pytest.approx(
            critical, rel=tolerance.rel_tol, abs=tolerance.abs_tol
        ), f"m={mass_kg}kg下临界值漂了"


def test_the_sphere_contact_couples_two_free_nodes():
    """**多体接触与单体接触的分界**：Hessian里必须有跨节点的耦合块。

    半空间接触的Hessian只在单个节点的3×3块里；球-球接触两端都是自由度，
    所以``(i, j)``块非零。**这一条是本案例"第一个多体接触"这句话的执行面。**
    """

    oracle = _oracle("oracle:pyramid/force_decomposition")
    radius = oracle.inputs["radius_mm"]
    contact_layout = build_contact_layout(
        layout_id="layout/coupling-probe",
        node_count=2,
        declarations=(ContactDeclaration("pair"),),
    )
    context = EnergyContext(
        context_id="context/coupling-probe",
        node_masses_kg=(1.0, 1.0),
        gravity_mm_s2=(0.0, 0.0, 0.0),
    )
    term = PenaltySphereContact(pairs=((0, 1, 2.0 * radius, 1.0e4),))
    from physics_engine.state import State

    state = State(
        layout=contact_layout.layout,
        vector=contact_layout.initial_vector(
            (0.0, 0.0, 0.0, 2.0 * radius - 0.5, 0.0, 0.0)
        ),
    )
    hessian = term.hessian(state, context)
    assert hessian[0][3] != 0.0, "球-球接触没有产生跨节点耦合块"
    assert hessian[0][3] == pytest.approx(-hessian[0][0], rel=1e-15), (
        "耦合块不等于对角块的相反数——牛顿第三定律在Hessian上的形状"
    )

    half_space = PenaltyNormalContact(
        planes=((0, (0.0, 0.0, 0.0), UP, 1.0e4, 0.0),)
    )
    half_space_hessian = half_space.hessian(state, context)
    assert all(
        half_space_hessian[row][column] == 0.0
        for row in range(3)
        for column in range(3, 6)
    ), "半空间接触不该有跨节点耦合——固定面不是自由度"
