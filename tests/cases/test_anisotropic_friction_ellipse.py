"""conformance：各向异性摩擦椭圆（`cases/anisotropic_friction_ellipse`）。

判据值一律从清单读，**不在本文件复述闭式**（轴7规则3）。

本案例判的是`anisotropic_return_map`这一条**切向本构**，
装置是位移控制、法向力由声明给定——**不走`solve_equilibrium`**。
理由与登记见案例页第四节第1条。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import (
    REGIME_SLIP,
    REGIME_STICK,
    FrictionEllipse,
    FrictionOutcome,
    anisotropic_return_map,
)
from physics_engine.oracles import load_manifest

CASE = Path(__file__).resolve().parents[2] / "cases" / "anisotropic_friction_ellipse"
MANIFEST = load_manifest(CASE / "oracle.json")

NORMAL = (0.0, 0.0, 1.0)
ALONG = (1.0, 0.0, 0.0)

pytestmark = pytest.mark.batch


def _oracle(identifier: str):
    for oracle in MANIFEST.oracles:
        if oracle.id == identifier:
            return oracle
    raise AssertionError(f"清单里没有{identifier}")


def _ellipse(mu_along: float, mu_across: float) -> FrictionEllipse:
    return FrictionEllipse(
        mu_along=mu_along,
        mu_across=mu_across,
        along_direction=ALONG,
        normal=NORMAL,
    )


def _radial_return_onto_ellipse(
    *,
    trial_force_n: tuple[float, float, float],
    semi_along: float,
    semi_across: float,
    stiffness: float,
) -> FrictionOutcome:
    """**对照组：被证伪的那条映射。** 沿``f_trial``方向缩到椭圆上。

    它同时是"把椭圆拉成圆、圆上径向返回、再变回来"那条捷径——
    展开之后是同一个纯标量缩放（`anisotropic_return_map`的docstring第二节，
    逐分量实测差8.88e-16）。本文件把它写出来，是因为
    **"两条映射不同"这句话必须被量过，否则整个案例只是在验自己**。
    """

    along, across = trial_force_n[0], trial_force_n[1]
    quadratic = (along / semi_along) ** 2 + (across / semi_across) ** 2
    if quadratic <= 1.0:
        return FrictionOutcome(trial_force_n, REGIME_STICK, (0.0, 0.0, 0.0))
    scale = 1.0 / math.sqrt(quadratic)
    force = tuple(component * scale for component in trial_force_n)
    correction = tuple(
        (trial_force_n[axis] - force[axis]) / stiffness for axis in range(3)
    )
    return FrictionOutcome(force, REGIME_SLIP, correction)


class _SteadySlide:
    """位移控制的稳态滑移：沿固定方向逐步推，直到力**逐位不再变**。

    稳态下锚点与位移同速前进，故滑移方向恒等于施加方向；
    此时``f·m̂``就是单位滑移距离的耗散——**关联流动给支撑函数，径向返回给椭圆半径**。

    ``settled_at``是"力连续两步逐位相同"的第一步。**它是被断言的量**：
    没有它，"稳态"就只是一个形容词。
    """

    #: 每步施加的位移，以``u_y = μ_∥N/k_t``为单位。取10：实测最晚33步逐位稳定
    #: （1倍要177步、50倍要22步）。定点与步长无关，故大步只是省时间不改答案。
    STEP_IN_YIELD = 10.0
    STEPS = 120

    def __init__(self, *, normal_force_n: float, stiffness: float) -> None:
        self.normal_force_n = normal_force_n
        self.stiffness = stiffness

    def run(self, angle_deg: float, mapper) -> tuple[tuple[float, float, float], int]:
        angle = math.radians(angle_deg)
        direction = (math.cos(angle), math.sin(angle), 0.0)
        yield_displacement = self.normal_force_n / self.stiffness
        step = self.STEP_IN_YIELD * yield_displacement
        anchor = [0.0, 0.0, 0.0]
        position = [0.0, 0.0, 0.0]
        previous: tuple[float, float, float] | None = None
        force: tuple[float, float, float] = (0.0, 0.0, 0.0)
        settled_at = -1
        for index in range(1, self.STEPS + 1):
            for axis in range(3):
                position[axis] += step * direction[axis]
            trial = tuple(
                self.stiffness * (position[axis] - anchor[axis]) for axis in range(3)
            )
            outcome = mapper(trial)
            for axis in range(3):
                anchor[axis] += outcome.anchor_correction_mm[axis]
            force = outcome.tangential_force_n
            if previous is not None and force == previous and settled_at < 0:
                settled_at = index
            previous = force
        return force, settled_at

    def dissipation(self, angle_deg: float, mapper) -> float:
        force, settled_at = self.run(angle_deg, mapper)
        assert settled_at > 0, f"{angle_deg}°上力从未停止变化——稳态没到，判据是空的"
        angle = math.radians(angle_deg)
        return force[0] * math.cos(angle) + force[1] * math.sin(angle)


def _cycle_loop(
    *,
    angle_deg: float,
    amplitude_mm: float,
    steps_per_leg: int,
    cycles: int,
    ellipse: FrictionEllipse,
    normal_force_n: float,
    stiffness: float,
) -> list[dict]:
    """位移控制往复``0 → +U → −U → 0``，逐圈交出两条独立记账与闭合残差。

    * **外功**``∮f·du``：梯形，用步前步后力的均值；
    * **塑性功**``Σf·Δu_slip``：只用锚点修正，与位移增量无关。

    闭合回线上两者必须相等——它们各自算错的概率不相关，**这才叫两条记账**。
    """

    angle = math.radians(angle_deg)
    direction = (math.cos(angle), math.sin(angle), 0.0)
    anchor = (0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)
    force = (0.0, 0.0, 0.0)
    records: list[dict] = []

    for _ in range(cycles):
        anchor_at_start = anchor
        force_at_start = force
        external = 0.0
        plastic = 0.0
        peak_across = 0.0
        for start, end in (
            (0.0, amplitude_mm),
            (amplitude_mm, -amplitude_mm),
            (-amplitude_mm, 0.0),
        ):
            for step in range(1, steps_per_leg + 1):
                target = start + (end - start) * step / steps_per_leg
                new_position = tuple(target * direction[axis] for axis in range(3))
                increment = tuple(
                    new_position[axis] - position[axis] for axis in range(3)
                )
                trial = tuple(
                    stiffness * (new_position[axis] - anchor[axis]) for axis in range(3)
                )
                outcome = anisotropic_return_map(
                    trial_force_n=trial,
                    normal_force_n=normal_force_n,
                    ellipse=ellipse,
                    tangential_stiffness_n_per_mm=stiffness,
                )
                new_force = outcome.tangential_force_n
                external += sum(
                    0.5 * (force[axis] + new_force[axis]) * increment[axis]
                    for axis in range(3)
                )
                plastic += sum(
                    new_force[axis] * outcome.anchor_correction_mm[axis]
                    for axis in range(3)
                )
                anchor = tuple(
                    anchor[axis] + outcome.anchor_correction_mm[axis] for axis in range(3)
                )
                position = new_position
                force = new_force
                peak_across = max(peak_across, abs(force[1]))
        records.append(
            {
                "external_n_mm": external,
                "plastic_n_mm": plastic,
                "anchor_drift_mm": max(
                    abs(anchor[axis] - anchor_at_start[axis]) for axis in range(3)
                ),
                "force_drift_n": max(
                    abs(force[axis] - force_at_start[axis]) for axis in range(3)
                ),
                "peak_across_force_n": peak_across,
            }
        )
    return records


def test_equal_coefficients_reduce_to_the_isotropic_map_bit_for_bit():
    """``μ_∥ == μ_⊥``时**逐位**等于`coulomb_return_map`，且通用路径自己也退化。

    两条一起才叫"退化被验过"：转交保证逐位，通用路径保证转交没有掩盖错误。
    """

    from physics_engine.contact import coulomb_return_map
    from physics_engine.contact.friction import _plastic_multiplier

    oracle = _oracle("oracle:friction/degenerate_to_isotropic")
    mu = oracle.inputs["mu_along"]
    normal_force = oracle.inputs["normal_force_n"]
    stiffness = oracle.inputs["tangential_stiffness_n_per_mm"]
    angle_count = oracle.inputs["angle_count"]
    magnitudes = (0.1, 1.2, 5.0, 1.0e6)
    assert len(magnitudes) == oracle.inputs["magnitude_count"]

    ellipse = _ellipse(mu, mu)
    compared = 0
    identical = True
    for index in range(angle_count):
        angle = 2.0 * math.pi * index / angle_count
        for magnitude in magnitudes:
            trial = (magnitude * math.cos(angle), magnitude * math.sin(angle), 0.0)
            elliptic = anisotropic_return_map(
                trial_force_n=trial,
                normal_force_n=normal_force,
                ellipse=ellipse,
                tangential_stiffness_n_per_mm=stiffness,
            )
            circular = coulomb_return_map(
                trial_force_n=trial,
                normal_force_n=normal_force,
                friction_coefficient=mu,
                tangential_stiffness_n_per_mm=stiffness,
            )
            compared += 1
            #: **逐位判的是IEEE-754位型**，不是``==``——``-0.0 == 0.0``为真而位型不同。
            if _bits(elliptic) != _bits(circular):
                identical = False

    #: 关掉转交：直接调通用路径的标量求解，与圆的闭式并排。
    semi = mu * normal_force
    worst_gap = 0.0
    for index in range(angle_count):
        angle = 2.0 * math.pi * index / angle_count
        for factor in (1.3, 5.0, 1.0e3):
            along = factor * semi * math.cos(angle)
            across = factor * semi * math.sin(angle)
            eta = _plastic_multiplier(
                along=along, across=across, semi_along=semi, semi_across=semi
            )
            general = (
                along * semi * semi / (semi * semi + eta),
                across * semi * semi / (semi * semi + eta),
            )
            magnitude = math.hypot(along, across)
            closed = (along * semi / magnitude, across * semi / magnitude)
            worst_gap = max(
                worst_gap,
                max(abs(general[axis] - closed[axis]) for axis in range(2)) / semi,
            )

    oracle.check_all(
        {
            "compared_cases": compared,
            "bitwise_identical": identical,
            "general_path_max_relative_gap": worst_gap,
        }
    )


def _bits(outcome: FrictionOutcome) -> tuple[str, ...]:
    """把一次return-map的全部浮点输出摊成IEEE-754位型串。"""

    return (
        *(value.hex() for value in outcome.tangential_force_n),
        outcome.regime.hex(),
        *(value.hex() for value in outcome.anchor_correction_mm),
    )


def test_the_slip_increment_lies_along_the_outward_normal():
    """最大耗散原理：滑移增量沿屈服面外法向，且``f·m̂``等于支撑函数。

    对照组是径向返回——它在同一条扫描上必须**显著违反**，
    否则"两条映射不同"这句话本身没被验过。
    """

    oracle = _oracle("oracle:friction/maximum_dissipation")
    mu_along = oracle.inputs["mu_along"]
    mu_across = oracle.inputs["mu_across"]
    normal_force = oracle.inputs["normal_force_n"]
    stiffness = oracle.inputs["tangential_stiffness_n_per_mm"]
    step_deg = oracle.inputs["ellipse_parameter_step_deg"]
    factors = oracle.inputs["trial_overshoot_factors"]
    near_yield = oracle.inputs["near_yield_overshoot_factor"]

    ellipse = _ellipse(mu_along, mu_across)
    semi_along = mu_along * normal_force
    semi_across = mu_across * normal_force

    worst_sine = 0.0
    worst_near_yield_sine = 0.0
    worst_support_gap = 0.0
    worst_radial_sine = 0.0
    samples = 0
    parameter = 0.0
    while parameter < 360.0:
        theta = math.radians(parameter)
        #: 按**椭圆参数**造试探力（不是按力的方位角）：这样径向返回恰好落在
        #: 参数为``theta``的那一点上，而外法向违反角的闭式正是用它表达的。
        base = (semi_along * math.cos(theta), semi_across * math.sin(theta), 0.0)
        for factor in (*factors, near_yield):
            trial = tuple(factor * value for value in base)
            outcome = anisotropic_return_map(
                trial_force_n=trial,
                normal_force_n=normal_force,
                ellipse=ellipse,
                tangential_stiffness_n_per_mm=stiffness,
            )
            assert outcome.regime == REGIME_SLIP
            sine = _flow_normal_sine(outcome, semi_along, semi_across)
            if factor == near_yield:
                worst_near_yield_sine = max(worst_near_yield_sine, sine)
            else:
                worst_sine = max(worst_sine, sine)
            worst_support_gap = max(
                worst_support_gap,
                _support_gap(outcome, semi_along, semi_across),
            )
            radial = _radial_return_onto_ellipse(
                trial_force_n=trial,
                semi_along=semi_along,
                semi_across=semi_across,
                stiffness=stiffness,
            )
            worst_radial_sine = max(
                worst_radial_sine,
                _flow_normal_sine(radial, semi_along, semi_across),
            )
            samples += 1
        parameter += step_deg

    assert samples == 360 * (len(factors) + 1), f"只扫了{samples}个样本——扫描本身缩水了"
    oracle.check_all(
        {
            "max_flow_normal_sine": worst_sine,
            "near_yield_flow_normal_sine": worst_near_yield_sine,
            "max_support_function_relative_gap": worst_support_gap,
            "radial_return_max_flow_normal_sine": worst_radial_sine,
        }
    )


def _flow_normal_sine(
    outcome: FrictionOutcome, semi_along: float, semi_across: float
) -> float:
    """``|sin|``：滑移增量与屈服面外法向``(f_∥/a², f_⊥/b²)``之间的夹角正弦。"""

    slip = outcome.anchor_correction_mm
    force = outcome.tangential_force_n
    gradient = (
        force[0] / (semi_along * semi_along),
        force[1] / (semi_across * semi_across),
    )
    slip_norm = math.hypot(slip[0], slip[1])
    gradient_norm = math.hypot(gradient[0], gradient[1])
    if slip_norm == 0.0 or gradient_norm == 0.0:
        return 0.0
    cross = slip[0] * gradient[1] - slip[1] * gradient[0]
    return abs(cross) / (slip_norm * gradient_norm)


def _support_gap(
    outcome: FrictionOutcome, semi_along: float, semi_across: float
) -> float:
    """``|f·m̂ − h(m̂)| / h(m̂)``：最大耗散原理的大小那一半。"""

    slip = outcome.anchor_correction_mm
    force = outcome.tangential_force_n
    norm = math.hypot(slip[0], slip[1])
    if norm == 0.0:
        return 0.0
    direction = (slip[0] / norm, slip[1] / norm)
    projected = force[0] * direction[0] + force[1] * direction[1]
    support = math.hypot(semi_along * direction[0], semi_across * direction[1])
    return abs(projected - support) / support


def test_the_mixed_angle_dissipation_shortfall_peaks_at_forty_five_degrees():
    """椭圆上径向返回 vs 最近点返回：**混合角上短缺，主轴上相等**。"""

    oracle = _oracle("oracle:friction/mixed_angle_dissipation")
    mu_along = oracle.inputs["mu_along"]
    mu_across = oracle.inputs["mu_across"]
    normal_force = oracle.inputs["normal_force_n"]
    stiffness = oracle.inputs["tangential_stiffness_n_per_mm"]
    angles = oracle.inputs["angles_deg"]
    scan_step = oracle.inputs["peak_scan_step_deg"]

    ellipse = _ellipse(mu_along, mu_across)
    semi_along = mu_along * normal_force
    semi_across = mu_across * normal_force
    slide = _SteadySlide(normal_force_n=normal_force, stiffness=stiffness)

    def associative(trial):
        return anisotropic_return_map(
            trial_force_n=trial,
            normal_force_n=normal_force,
            ellipse=ellipse,
            tangential_stiffness_n_per_mm=stiffness,
        )

    def radial(trial):
        return _radial_return_onto_ellipse(
            trial_force_n=trial,
            semi_along=semi_along,
            semi_across=semi_across,
            stiffness=stiffness,
        )

    associative_dissipation = [slide.dissipation(angle, associative) for angle in angles]
    radial_dissipation = [slide.dissipation(angle, radial) for angle in angles]

    peak_shortfall = -1.0
    peak_angle = -1.0
    scanned = 0
    angle = 0.0
    while angle <= 90.0 + 0.5 * scan_step:
        shortfall = 1.0 - slide.dissipation(angle, radial) / slide.dissipation(
            angle, associative
        )
        if shortfall > peak_shortfall:
            peak_shortfall = shortfall
            peak_angle = angle
        scanned += 1
        angle += scan_step
    assert scanned == 91, f"峰值扫描只走了{scanned}个格点——格点里可能已经没有45°了"

    #: 横向力被高估多少：同一个45°稳态上两条映射的横向分量之比。
    peak_force, _ = slide.run(peak_angle, associative)
    peak_radial_force, _ = slide.run(peak_angle, radial)
    overstatement = peak_radial_force[1] / peak_force[1]

    oracle.check_all(
        {
            "associative_dissipation_n": associative_dissipation,
            "radial_dissipation_n": radial_dissipation,
            "peak_shortfall": peak_shortfall,
            "peak_shortfall_angle_deg": peak_angle,
            "transverse_force_overstatement_at_peak": overstatement,
        }
    )


def test_a_single_scalar_mu_shorts_the_dissipation_on_the_other_axis():
    """另一个口径：拿单个``μ_∥``顶替椭圆，短缺峰值在**主轴**上而不在混合角上。

    这一条存在的理由是**不许把两个口径混起来报**：
    "混合角上最高60%"说的不是这个口径（这个口径的峰值是80%、在90°）。
    """

    oracle = _oracle("oracle:friction/isotropic_substitute_shortfall")
    substitute = oracle.inputs["substitute_mu"]
    mu_along = oracle.inputs["mu_along"]
    mu_across = oracle.inputs["mu_across"]
    normal_force = oracle.inputs["normal_force_n"]
    angles = oracle.inputs["angles_deg"]
    stiffness = 3.0e4

    anisotropic = _ellipse(mu_along, mu_across)
    isotropic = _ellipse(substitute, substitute)
    slide = _SteadySlide(normal_force_n=normal_force, stiffness=stiffness)

    def with_ellipse(ellipse):
        def mapper(trial):
            return anisotropic_return_map(
                trial_force_n=trial,
                normal_force_n=normal_force,
                ellipse=ellipse,
                tangential_stiffness_n_per_mm=stiffness,
            )

        return mapper

    isotropic_values = [
        slide.dissipation(angle, with_ellipse(isotropic)) for angle in angles
    ]
    anisotropic_values = [
        slide.dissipation(angle, with_ellipse(anisotropic)) for angle in angles
    ]
    #: 各向同性那条必须与方向无关——先判这件事，再拿它当分母。
    assert max(isotropic_values) - min(isotropic_values) <= 1.0e-15, (
        f"各向同性的稳态耗散随方向变了（{min(isotropic_values)}—{max(isotropic_values)}）"
        "——那说明转交路径把方向带进去了"
    )
    shortfalls = [
        1.0 - anisotropic_values[index] / isotropic_values[index]
        for index in range(len(angles))
    ]
    peak = max(shortfalls)
    peak_angle = angles[shortfalls.index(peak)]

    oracle.check_all(
        {
            "isotropic_dissipation_n": isotropic_values[0],
            "shortfall_vs_anisotropic": shortfalls,
            "peak_shortfall": peak,
            "peak_shortfall_angle_deg": peak_angle,
        }
    )


def test_the_loop_on_the_long_axis_matches_one_dimensional_ideal_plasticity():
    """沿主轴往复时椭圆退化成1维理想塑性，回线面积对闭式，且横向力恒为机器零。"""

    oracle = _oracle("oracle:friction/principal_axis_loop")
    mu_along = oracle.inputs["mu_along"]
    mu_across = oracle.inputs["mu_across"]
    normal_force = oracle.inputs["normal_force_n"]
    stiffness = oracle.inputs["tangential_stiffness_n_per_mm"]
    amplitudes = oracle.inputs["amplitudes_in_yield"]
    steps = oracle.inputs["steps_per_leg"]
    steady_index = oracle.inputs["steady_cycle_index"]

    ellipse = _ellipse(mu_along, mu_across)
    yield_displacement = mu_along * normal_force / stiffness

    dissipation = []
    worst_across = 0.0
    for ratio in amplitudes:
        records = _cycle_loop(
            angle_deg=0.0,
            amplitude_mm=ratio * yield_displacement,
            steps_per_leg=steps,
            cycles=steady_index,
            ellipse=ellipse,
            normal_force_n=normal_force,
            stiffness=stiffness,
        )
        dissipation.append(records[steady_index - 1]["external_n_mm"])
        worst_across = max(
            worst_across, max(record["peak_across_force_n"] for record in records)
        )

    oracle.check_all(
        {
            "yield_displacement_mm": yield_displacement,
            "dissipation_n_mm": dissipation,
            "max_across_force_n": worst_across,
        }
    )


def test_the_mixed_angle_loop_closes_and_both_accountings_agree():
    """45°往复：稳态圈**逐位闭合**，外功与塑性功两条独立记账相等。

    顺手补上`friction_hysteresis_loop`第六条已知失效——那里明写
    "多圈的稳态性（第二圈与第一圈是否逐位相同）没验"。
    """

    oracle = _oracle("oracle:friction/mixed_angle_closed_loop")
    mu_along = oracle.inputs["mu_along"]
    mu_across = oracle.inputs["mu_across"]
    normal_force = oracle.inputs["normal_force_n"]
    stiffness = oracle.inputs["tangential_stiffness_n_per_mm"]
    angle = oracle.inputs["angle_deg"]
    amplitude_ratio = oracle.inputs["amplitude_in_yield"]
    steps = oracle.inputs["steps_per_leg"]
    cycles = oracle.inputs["cycles"]

    ellipse = _ellipse(mu_along, mu_across)
    yield_displacement = mu_along * normal_force / stiffness
    records = _cycle_loop(
        angle_deg=angle,
        amplitude_mm=amplitude_ratio * yield_displacement,
        steps_per_leg=steps,
        cycles=cycles,
        ellipse=ellipse,
        normal_force_n=normal_force,
        stiffness=stiffness,
    )

    last = records[-1]
    previous = records[-2]
    oracle.check_all(
        {
            "steady_anchor_drift_mm": last["anchor_drift_mm"],
            "steady_force_drift_n": last["force_drift_n"],
            "cycle_to_cycle_relative_gap": abs(
                last["external_n_mm"] - previous["external_n_mm"]
            )
            / abs(last["external_n_mm"]),
            "energy_balance_relative_gap": abs(
                last["external_n_mm"] - last["plastic_n_mm"]
            )
            / abs(last["plastic_n_mm"]),
            "first_cycle_is_different": records[0]["external_n_mm"]
            != last["external_n_mm"],
        }
    )
