"""conformance：斜面滑动阈值（`cases/incline_slide_threshold`）。

判据值一律从清单读，**不在本文件复述闭式**（轴7规则3）。

本文件里被验的路径是**完整的一条**：
``state`` → ``energies``（重力 + 法向罚 + 粘着弹簧）→ ``solve``（牛顿+回溯+LU），
再由`coulomb_return_map`判粘/滑。这是`cases/README.md`第一节之二那张表里
第一条**带接触**的整条路案例。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    PenaltyNormalContact,
    TangentialStickSpring,
    build_contact_layout,
    coulomb_return_map,
)
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium

CASE = Path(__file__).resolve().parents[2] / "cases" / "incline_slide_threshold"
MANIFEST = load_manifest(CASE / "oracle.json")


def _oracle(identifier: str):
    for oracle in MANIFEST.oracles:
        if oracle.id == identifier:
            return oracle
    raise AssertionError(f"清单里没有{identifier}")


def _solve_on_slope(
    theta_rad: float,
    *,
    mass_kg: float,
    gravity_mm_s2: float,
    normal_stiffness: float,
    tangential_stiffness: float,
) -> tuple[float, tuple[float, float, float]]:
    """在倾角``θ``的面上解静平衡，返回``(N, T矢量)``。

    **三个自由度全部放开。** 只放开``z``会给出``W/cos θ``而不是``W·cos θ``——
    差``1/cos²θ``，因为那已经不是斜面问题了（案例页第一节）。
    """

    normal = (math.sin(theta_rad), 0.0, math.cos(theta_rad))
    weight_n = mass_kg * gravity_mm_s2 / 1000.0

    contact_layout = build_contact_layout(
        layout_id="layout/incline_slide_threshold",
        node_count=1,
        declarations=(ContactDeclaration("block_slope"),),
    )
    context = EnergyContext(
        context_id="context/incline_slide_threshold",
        node_masses_kg=(mass_kg,),
        gravity_mm_s2=(0.0, 0.0, -gravity_mm_s2),
    )
    contact_layout.assert_matches_context(context)

    normal_term = PenaltyNormalContact(
        planes=((0, (0.0, 0.0, 0.0), normal, normal_stiffness),)
    )
    stick_term = TangentialStickSpring(
        springs=((0, (0.0, 0.0, 0.0), normal, tangential_stiffness),)
    )
    registry = EnergyRegistry(terms=(UniformGravity(), normal_term, stick_term))

    # 初值刻意压在面内：分离态起步切线刚度奇异（已知失效清单第4条）
    depth = weight_n * math.cos(theta_rad) / normal_stiffness
    start = contact_layout.initial_vector(
        tuple(-0.5 * depth * component for component in normal)
    )
    result = solve_equilibrium(
        registry,
        context,
        contact_layout.layout,
        start,
        fixed_indices=frozenset(range(3, contact_layout.layout.dof_count)),
        residual_tol_n=1.0e-12,
        max_iterations=80,
    )
    assert result.converged, result.reason
    return (
        normal_term.normal_force_n(result.state)[0],
        stick_term.tangential_force_n(result.state)[0],
    )


def test_gravity_decomposes_into_the_closed_form_normal_and_tangential_forces():
    """``N = W cos θ``、``T = W sin θ``，五档倾角。

    **判力不判位置**：罚函数的穿透是模型自带的``O(1/k)``，不是误差。
    """

    oracle = _oracle("oracle:incline/force_decomposition")
    angles = oracle.inputs["angles_deg"]
    expected_normal = oracle.expected["normal_force_n"]
    expected_tangential = oracle.expected["tangential_force_n"]
    normal_tolerance = oracle.tolerances["normal_force_n"]
    tangential_tolerance = oracle.tolerances["tangential_force_n"]

    for index, degrees in enumerate(angles):
        normal, tangential_vector = _solve_on_slope(
            math.radians(degrees),
            mass_kg=2.0,
            gravity_mm_s2=9810.0,
            normal_stiffness=oracle.inputs["normal_stiffness_n_per_mm"],
            tangential_stiffness=oracle.inputs["tangential_stiffness_n_per_mm"],
        )
        tangential = math.sqrt(sum(value * value for value in tangential_vector))
        assert normal == pytest.approx(
            expected_normal[index], rel=normal_tolerance.rel_tol, abs=normal_tolerance.abs_tol
        ), f"{degrees}°的法向力"
        assert tangential == pytest.approx(
            expected_tangential[index],
            rel=tangential_tolerance.rel_tol,
            abs=tangential_tolerance.abs_tol,
        ), f"{degrees}°的切向力"


def test_the_stick_slip_threshold_is_arctan_mu():
    """**本案例的主判据**：阈值两侧行为定性相反。

    只断一侧的话，一个"永远粘住"的实现能过一半——所以两侧都断。
    """

    oracle = _oracle("oracle:incline/threshold_angle")
    mu = oracle.inputs["friction_coefficient"]
    threshold_deg = oracle.expected["threshold_angle_deg"]
    tolerance = oracle.tolerances["threshold_angle_deg"]

    assert math.degrees(math.atan(mu)) == pytest.approx(
        threshold_deg, rel=tolerance.rel_tol, abs=tolerance.abs_tol
    )

    verdicts = []
    for offset in oracle.inputs["offsets_deg"]:
        normal, tangential_vector = _solve_on_slope(
            math.radians(threshold_deg + offset),
            mass_kg=oracle.inputs["mass_kg"],
            gravity_mm_s2=oracle.inputs["gravity_mm_s2"],
            normal_stiffness=oracle.inputs["normal_stiffness_n_per_mm"],
            tangential_stiffness=oracle.inputs["tangential_stiffness_n_per_mm"],
        )
        outcome = coulomb_return_map(
            trial_force_n=tangential_vector,
            normal_force_n=normal,
            friction_coefficient=mu,
            tangential_stiffness_n_per_mm=oracle.inputs["tangential_stiffness_n_per_mm"],
        )
        verdicts.append(outcome.is_stick)

    below = [verdict for offset, verdict in zip(oracle.inputs["offsets_deg"], verdicts, strict=True) if offset < 0]
    above = [verdict for offset, verdict in zip(oracle.inputs["offsets_deg"], verdicts, strict=True) if offset > 0]
    assert all(below) is oracle.expected["sticks_below"], f"阈值下方没粘住：{verdicts}"
    assert (not any(above)) is oracle.expected["slips_above"], f"阈值上方没滑动：{verdicts}"


def test_the_threshold_does_not_move_when_the_penalty_stiffnesses_change():
    """**判据与罚刚度无关，这本身就是一道门。**

    ``θc = arctan(μs)``里没有``k``——两个刚度各改一个数量级，
    阈值两侧的判别必须一个字都不变。若它变了，说明某处把模型参数当成了物理。
    """

    oracle = _oracle("oracle:incline/threshold_angle")
    mu = oracle.inputs["friction_coefficient"]
    threshold_deg = oracle.expected["threshold_angle_deg"]

    for normal_stiffness, tangential_stiffness in ((5.0e3, 3.0e5), (5.0e5, 3.0e3)):
        verdicts = []
        for offset in (-1.0e-4, 1.0e-4):
            normal, tangential_vector = _solve_on_slope(
                math.radians(threshold_deg + offset),
                mass_kg=oracle.inputs["mass_kg"],
                gravity_mm_s2=oracle.inputs["gravity_mm_s2"],
                normal_stiffness=normal_stiffness,
                tangential_stiffness=tangential_stiffness,
            )
            verdicts.append(
                coulomb_return_map(
                    trial_force_n=tangential_vector,
                    normal_force_n=normal,
                    friction_coefficient=mu,
                    tangential_stiffness_n_per_mm=tangential_stiffness,
                ).is_stick
            )
        assert verdicts == [True, False], (
            f"k_n={normal_stiffness:g}, k_t={tangential_stiffness:g}下阈值动了：{verdicts}"
        )


def test_the_threshold_does_not_move_when_the_mass_changes():
    """``W``在``T ≤ μN``两边约掉——质量翻倍，阈值一动不动。"""

    oracle = _oracle("oracle:incline/threshold_angle")
    mu = oracle.inputs["friction_coefficient"]
    threshold_deg = oracle.expected["threshold_angle_deg"]

    for mass_kg in (0.5, 2.0, 50.0):
        verdicts = []
        for offset in (-1.0e-4, 1.0e-4):
            normal, tangential_vector = _solve_on_slope(
                math.radians(threshold_deg + offset),
                mass_kg=mass_kg,
                gravity_mm_s2=oracle.inputs["gravity_mm_s2"],
                normal_stiffness=oracle.inputs["normal_stiffness_n_per_mm"],
                tangential_stiffness=oracle.inputs["tangential_stiffness_n_per_mm"],
            )
            verdicts.append(
                coulomb_return_map(
                    trial_force_n=tangential_vector,
                    normal_force_n=normal,
                    friction_coefficient=mu,
                    tangential_stiffness_n_per_mm=oracle.inputs["tangential_stiffness_n_per_mm"],
                ).is_stick
            )
        assert verdicts == [True, False], f"m={mass_kg}kg下阈值动了：{verdicts}"


def test_the_return_map_lands_exactly_on_the_cone():
    """滑移之后，用**新锚点**重算的试探力必须恰好落在锥面上。

    这是return-map的定义，也是它唯一的自洽门：投影完还超出锥面，
    说明锚点挪的距离不对——而那个错误在阈值判别里**看不出来**
    （判别只看"超没超"，不看"投影到哪"）。
    """

    normal_force = 10.0
    mu = 0.25
    stiffness = 1000.0
    trial = (5.0, 0.0, 0.0)  # 远超 μN = 2.5
    outcome = coulomb_return_map(
        trial_force_n=trial,
        normal_force_n=normal_force,
        friction_coefficient=mu,
        tangential_stiffness_n_per_mm=stiffness,
    )
    assert not outcome.is_stick
    magnitude = math.sqrt(sum(value * value for value in outcome.tangential_force_n))
    assert magnitude == pytest.approx(mu * normal_force, rel=1e-15)

    # 锚点挪过之后，同一位形的试探力恰好等于锥面值
    slip = math.sqrt(sum(value * value for value in outcome.anchor_correction_mm))
    trial_magnitude = math.sqrt(sum(value * value for value in trial))
    assert trial_magnitude - stiffness * slip == pytest.approx(mu * normal_force, rel=1e-15)
