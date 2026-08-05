"""`case/rigid_body_free_flight`的conformance门（轴7规则3）。

**本文件里没有一个判据数**：期望与容差全部从清单读，测试只把输入喂给
`rigidbody`、把算出的量交给清单比对。四层判据：

1. **守恒量**——无力矩下**惯性系**角动量矢量与转动动能的漂移，以及它随步长的阶；
2. **轴对称闭式**——体系进动率`λ = ω3(Ia−It)/It`（带符号，扁体正长体负）
   与惯性系进动率`ψ̇ = |L|/It`；
3. **中间轴定理**——线性化增长率闭式、稳定轴的振幅上界闭式、翻转次数整数；
4. **四元数范数**——归一化**之前**的偏离有界。

第六节的必红矩阵是本文件的另一半：四条典型错法逐条注入，逐条记下**哪些门红了、
哪些门瞎**。"哪条错法没被任何门抓住"写在案例页第四节。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine import rigidbody
from physics_engine.oracles import load_manifest
from physics_engine.rigidbody import (
    EXPLICIT_EULER_BODY,
    RIGID_BODY_INTEGRATORS,
    RK4_BODY,
    RigidBodyInertia,
)
from physics_engine.shapes import FiniteCylinder, RoundedBox

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/rigid_body_free_flight/oracle.json", root=ROOT)
BY_ID = {entry.id: entry for entry in MANIFEST.oracles}

pytestmark = pytest.mark.batch

MASS_KG = 1.0
_OMEGA_BLOCK = "angular_velocity_body_rad_per_s"
_ATTITUDE_BLOCK = "attitude_body_to_world_xyzw"
_POSITION_BLOCK = "centre_of_mass_position_mm"


def _box(half_extents) -> RigidBodyInertia:
    return RigidBodyInertia.from_shape(
        RoundedBox(half_extents_mm=tuple(half_extents), fillet_radius_mm=0.0),
        mass_kg=MASS_KG,
    )


def _cylinder(radius_mm, half_width_mm) -> RigidBodyInertia:
    return RigidBodyInertia.from_shape(
        FiniteCylinder(radius_mm=radius_mm, half_width_mm=half_width_mm),
        mass_kg=MASS_KG,
    )


def _diagonal(inertia: RigidBodyInertia) -> list[float]:
    return [inertia.inertia_body_kg_mm2[axis][axis] for axis in range(3)]


def _run(
    inertia,
    omega0,
    horizon_s,
    dt_s,
    *,
    integrator=RK4_BODY,
    observer=None,
    torque=None,
    force=None,
    velocity=(0.0, 0.0, 0.0),
):
    state = rigidbody.make_state(
        velocity_mm_per_s=velocity, angular_velocity_rad_per_s=omega0
    )
    return rigidbody.integrate_free_flight(
        integrator,
        state=state,
        inertia=inertia,
        dt_s=dt_s,
        steps=int(round(horizon_s / dt_s)),
        observer=observer,
        torque_body_nmm=torque,
        force_world_n=force,
    )


def _norm(vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _conservation_drifts(inertia, omega0, horizon_s, dt_s, integrator=RK4_BODY):
    """返回(角动量相对漂移, 动能相对漂移, 体系角动量相对变化量, 范数偏离, 归一化次数)。

    **三个漂移都是全程最大值**，不是首末差——只比首末两步看不见中间冲上去又掉回来
    （spec/12第6.2节写法3的堵法）。
    """

    state = rigidbody.make_state(angular_velocity_rad_per_s=omega0)
    reference = rigidbody.angular_momentum_world_kg_mm2_per_s(inertia, state)
    reference_norm = _norm(reference)
    initial_energy = rigidbody.rotational_kinetic_energy_nmm(inertia, state)
    body_reference = rigidbody.angular_momentum_body_kg_mm2_per_s(inertia, state)
    assert reference_norm > 0.0 and initial_energy > 0.0, (
        "参照量为零——任何相对漂移判据在这里都会假通过（spec/12第6.2节写法1的堵法）"
    )
    worst = {"momentum": 0.0, "energy": 0.0, "body": 0.0}

    def observe(_index, _t, current):
        momentum = rigidbody.angular_momentum_world_kg_mm2_per_s(inertia, current)
        energy = rigidbody.rotational_kinetic_energy_nmm(inertia, current)
        body = rigidbody.angular_momentum_body_kg_mm2_per_s(inertia, current)
        worst["momentum"] = max(
            worst["momentum"],
            _norm([a - b for a, b in zip(momentum, reference, strict=True)])
            / reference_norm,
        )
        worst["energy"] = max(worst["energy"], abs(energy - initial_energy) / initial_energy)
        worst["body"] = max(
            worst["body"],
            _norm([a - b for a, b in zip(body, body_reference, strict=True)])
            / reference_norm,
        )

    _final, diagnostics = _run(
        inertia, omega0, horizon_s, dt_s, integrator=integrator, observer=observe
    )
    return (
        worst["momentum"], worst["energy"], worst["body"],
        diagnostics.max_norm_deviation, diagnostics.renormalisations,
    )


def _unwrapped_rate(samples, initial_phase, horizon_s) -> float:
    """相位展开后的平均角速率。**带符号**——判据分不分得开叉乘次序全靠它。"""

    total = 0.0
    previous = initial_phase
    for phase in samples:
        delta = phase - previous
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        total += delta
        previous = phase
    return total / horizon_s


def _precession_rates(inertia, omega0, horizon_s, dt_s):
    """（体系进动率，惯性系进动率）。前者只看`ω`，后者必须用姿态四元数。"""

    state = rigidbody.make_state(angular_velocity_rad_per_s=omega0)
    momentum = rigidbody.angular_momentum_world_kg_mm2_per_s(inertia, state)
    axis_z = [value / _norm(momentum) for value in momentum]
    seed = (1.0, 0.0, 0.0) if abs(axis_z[0]) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(seed[i] * axis_z[i] for i in range(3))
    axis_x = [seed[i] - axis_z[i] * projection for i in range(3)]
    axis_x = [value / _norm(axis_x) for value in axis_x]
    axis_y = rigidbody.cross(axis_z, axis_x)

    body_phases: list[float] = []
    inertial_phases: list[float] = []

    def observe(_index, _t, current):
        omega = current.block(_OMEGA_BLOCK)
        body_phases.append(math.atan2(omega[1], omega[0]))
        symmetry_axis = rigidbody.rotate_body_to_world(
            current.block(_ATTITUDE_BLOCK), (0.0, 0.0, 1.0)
        )
        inertial_phases.append(
            math.atan2(
                sum(a * b for a, b in zip(symmetry_axis, axis_y, strict=True)),
                sum(a * b for a, b in zip(symmetry_axis, axis_x, strict=True)),
            )
        )

    _run(inertia, omega0, horizon_s, dt_s, observer=observe)
    return (
        _unwrapped_rate(body_phases, math.atan2(omega0[1], omega0[0]), horizon_s),
        _unwrapped_rate(inertial_phases, math.atan2(axis_y[2], axis_x[2]), horizon_s),
    )


def _perturbation_history(inertia, spin_axis, spin, delta, horizon_s, dt_s):
    """绕`spin_axis`自转+一个横向小扰动，逐步记（时刻，横向扰动模，自转分量）。"""

    omega0 = [0.0, 0.0, 0.0]
    omega0[spin_axis] = spin
    others = [axis for axis in range(3) if axis != spin_axis]
    omega0[others[0]] = delta
    history: list[tuple[float, float, float]] = []

    def observe(_index, t, current):
        omega = current.block(_OMEGA_BLOCK)
        history.append(
            (t, math.sqrt(sum(omega[axis] ** 2 for axis in others)), omega[spin_axis])
        )

    _run(inertia, tuple(omega0), horizon_s, dt_s, observer=observe)
    return history


def _log_slope(history, window) -> float:
    """窗口两端的对数斜率。**两点式而不是从0起算**：线性化解是`cosh(σt)`,
    从`t=0`起算会带一个`−ln2/T`的系统偏置，两点式把它消掉。"""

    start, end = (
        min(history, key=lambda row: abs(row[0] - target)) for target in window
    )
    return math.log(end[1] / start[1]) / (end[0] - start[0])


def _sign_changes(history) -> int:
    """自转分量的变号次数——Dzhanibekov翻转的**确定性整数**形式。"""

    changes = 0
    previous = 1.0
    for _t, _magnitude, spin in history:
        sign = 1.0 if spin > 0.0 else -1.0
        if sign != previous:
            changes += 1
            previous = sign
    return changes


# ---------------------------------------------------------------------------
# 判据一：惯量取自geometry，且与教科书闭式对得上
# ---------------------------------------------------------------------------


def test_inertia_comes_from_geometry_and_matches_the_textbook_closed_form():
    entry = BY_ID["oracle:rigidbody/inertia_from_geometry"]
    inputs = entry.inputs
    entry.check_all({
        "box_diagonal_kg_mm2": _diagonal(_box(inputs["box_half_extents_mm"])),
        "disc_diagonal_kg_mm2": _diagonal(
            _cylinder(inputs["disc_radius_mm"], inputs["disc_half_width_mm"])
        ),
        "rod_diagonal_kg_mm2": _diagonal(
            _cylinder(inputs["rod_radius_mm"], inputs["rod_half_width_mm"])
        ),
    })


# ---------------------------------------------------------------------------
# 判据二：守恒量与它随步长的阶
# ---------------------------------------------------------------------------


def test_torque_free_conservation_drift_falls_at_fourth_order():
    entry = BY_ID["oracle:rigidbody/conservation_order_rk4"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    momentum, energy, body = [], [], []
    for dt_s in inputs["dt_s_ladder"]:
        drift_l, drift_e, drift_body, _deviation, _count = _conservation_drifts(
            inertia, inputs["omega0_rad_per_s"], inputs["horizon_s"], dt_s
        )
        momentum.append(drift_l)
        energy.append(drift_e)
        body.append(drift_body)

    low, high = entry.expected["ratio_low"], entry.expected["ratio_high"]
    floor = entry.expected["body_frame_momentum_variation_floor"]
    momentum_ratios = [a / b for a, b in zip(momentum, momentum[1:], strict=False)]
    energy_ratios = [a / b for a, b in zip(energy, energy[1:], strict=False)]
    entry.check_all({
        "ratio_low": low,
        "ratio_high": high,
        "formal_order": RK4_BODY.declaration.formal_order,
        "momentum_ratios_within_band": all(low <= r <= high for r in momentum_ratios),
        "energy_ratios_within_band": all(low <= r <= high for r in energy_ratios),
        "all_drifts_nonzero": all(value > 0.0 for value in momentum + energy),
        "body_frame_momentum_variation_floor": floor,
        "body_frame_momentum_varies": min(body) > floor,
    })
    assert all(low <= ratio <= high for ratio in momentum_ratios), (
        f"角动量漂移比 {momentum_ratios} 落在[{low}, {high}]之外；逐档漂移 {momentum}"
    )
    assert all(low <= ratio <= high for ratio in energy_ratios), (
        f"动能漂移比 {energy_ratios} 落在[{low}, {high}]之外；逐档漂移 {energy}"
    )
    assert min(body) > floor, (
        f"体系角动量`I·ω`几乎没变（{body}）——惯性系守恒判据可能只是在陈述恒等式"
    )


def test_drift_ordering_holds_and_neither_integrator_is_silently_frozen():
    entry = BY_ID["oracle:rigidbody/drift_ordering"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    drifts = {}
    for name in entry.expected["ordering"]:
        drift_l, _e, _b, _d, _c = _conservation_drifts(
            inertia, inputs["omega0_rad_per_s"], inputs["horizon_s"], inputs["dt_s"],
            integrator=RIGID_BODY_INTEGRATORS[name],
        )
        drifts[name] = drift_l
    entry.check_all({
        "ordering": entry.expected["ordering"],
        "all_nonzero": all(value != 0.0 for value in drifts.values()),
    })
    ordered = [drifts[name] for name in entry.expected["ordering"]]
    assert ordered == sorted(ordered, reverse=True) and len(set(ordered)) == len(ordered), (
        f"漂移排序不成立：{drifts}"
    )


# ---------------------------------------------------------------------------
# 判据三：轴对称闭式（体系与惯性系两条，都带符号）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "oracle_id",
    [
        "oracle:rigidbody/axisymmetric_precession_disc",
        "oracle:rigidbody/axisymmetric_precession_rod",
    ],
)
def test_axisymmetric_precession_matches_the_closed_form(oracle_id):
    entry = BY_ID[oracle_id]
    inputs = entry.inputs
    inertia = _cylinder(inputs["radius_mm"], inputs["half_width_mm"])
    body_rate, inertial_rate = _precession_rates(
        inertia, inputs["omega0_rad_per_s"], inputs["horizon_s"], inputs["dt_s"]
    )
    measured = {
        "body_precession_rate_per_s": body_rate,
        "inertial_precession_rate_per_s": inertial_rate,
    }
    if "body_rate_is_negative" in entry.expected:
        measured["body_rate_is_negative"] = body_rate < 0.0
    entry.check_all(measured)


# ---------------------------------------------------------------------------
# 判据四：中间轴定理
# ---------------------------------------------------------------------------


def test_intermediate_axis_perturbation_grows_at_the_closed_form_rate():
    entry = BY_ID["oracle:rigidbody/dzhanibekov_growth_rate"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    history = _perturbation_history(
        inertia, inputs["spin_axis"], inputs["spin_rad_per_s"],
        inputs["perturbation_rad_per_s"], inputs["horizon_s"], inputs["dt_s"],
    )
    rate = _log_slope(history, inputs["window_s"])
    entry.check_all({
        "growth_rate_per_s": rate,
        "growth_rate_is_positive": rate > 0.0,
    })


def test_stable_axis_perturbations_stay_under_the_closed_form_bound():
    entry = BY_ID["oracle:rigidbody/dzhanibekov_stable_axes"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    delta = inputs["perturbation_rad_per_s"]
    amplifications = {}
    for label, axis in (("min", 0), ("max", 2)):
        history = _perturbation_history(
            inertia, axis, inputs["spin_rad_per_s"], delta,
            inputs["horizon_s"], inputs["dt_s"],
        )
        amplifications[label] = max(row[1] for row in history) / delta
    entry.check_all({
        "min_axis_amplification": amplifications["min"],
        "max_axis_amplification": amplifications["max"],
    })


def test_only_the_intermediate_axis_flips():
    entry = BY_ID["oracle:rigidbody/dzhanibekov_flips"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    delta = inputs["perturbation_rad_per_s"]
    flips, amplification = {}, {}
    for label, axis in (("min", 0), ("intermediate", 1), ("max", 2)):
        history = _perturbation_history(
            inertia, axis, inputs["spin_rad_per_s"], delta,
            inputs["horizon_s"], inputs["dt_s"],
        )
        flips[label] = _sign_changes(history)
        amplification[label] = max(row[1] for row in history) / delta
    ordering = sorted(amplification, key=lambda label: amplification[label], reverse=True)
    entry.check_all({
        "flips_about_intermediate_axis": flips["intermediate"],
        "flips_about_min_axis": flips["min"],
        "flips_about_max_axis": flips["max"],
        "amplification_ordering": ordering,
    })
    assert len(set(amplification.values())) == 3, (
        f"三个轴的放大量出现并列，排序判据失去意义：{amplification}"
    )


# ---------------------------------------------------------------------------
# 判据五：常力矩闭式（本案例唯一钉住惯量绝对量级的一条）
# ---------------------------------------------------------------------------


def test_constant_torque_about_a_principal_axis_is_an_exact_ramp():
    entry = BY_ID["oracle:rigidbody/constant_torque_ramp"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    axis, magnitude = inputs["torque_axis"], inputs["torque_nmm"]
    final, _diagnostics = _run(
        inertia, (0.0, 0.0, 0.0), inputs["horizon_s"], inputs["dt_s"],
        torque=lambda _y, _t: tuple(
            magnitude if index == axis else 0.0 for index in range(3)
        ),
    )
    omega = final.block(_OMEGA_BLOCK)
    entry.check_all({
        "omega_x_rad_per_s": omega[0],
        "omega_y_rad_per_s": omega[1],
        "omega_z_rad_per_s": omega[2],
    })


# ---------------------------------------------------------------------------
# 判据六：平动那一半与转动严格解耦
# ---------------------------------------------------------------------------


def test_free_flight_translation_is_the_ballistic_parabola_and_spin_does_not_touch_it():
    entry = BY_ID["oracle:rigidbody/free_flight_translation"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    weight = inertia.mass_kg * inputs["gravity_mm_per_s2"] / 1000.0

    def fly(omega0):
        final, _diagnostics = _run(
            inertia, omega0, inputs["horizon_s"], inputs["dt_s"],
            velocity=(0.0, inputs["launch_speed_mm_per_s"], 0.0),
            force=lambda _y, _t: (0.0, weight, 0.0),
        )
        return final.block(_POSITION_BLOCK)

    still = fly((0.0, 0.0, 0.0))
    spinning = fly(tuple(inputs["spinning_omega_rad_per_s"]))
    entry.check_all({
        "position_y_mm": spinning[1],
        "spin_does_not_move_the_centre_of_mass": still == spinning,
    })


# ---------------------------------------------------------------------------
# 判据七：四元数范数（门守在归一化**之前**那一侧）
# ---------------------------------------------------------------------------


def test_quaternion_norm_drift_before_renormalisation_is_bounded():
    entry = BY_ID["oracle:rigidbody/quaternion_norm_drift"]
    inputs = entry.inputs
    inertia = _box(BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"])
    deviations, counts = [], []
    for dt_s in inputs["dt_s_ladder"]:
        _l, _e, _b, deviation, count = _conservation_drifts(
            inertia, inputs["omega0_rad_per_s"], inputs["horizon_s"], dt_s
        )
        deviations.append(deviation)
        counts.append((count, int(round(inputs["horizon_s"] / dt_s))))
    ceiling = entry.expected["max_norm_deviation_ceiling"]
    entry.check_all({
        "max_norm_deviation_ceiling": ceiling,
        "deviation_within_ceiling": all(value <= ceiling for value in deviations),
        "deviation_nonzero": all(value > 0.0 for value in deviations),
        "renormalisation_count_equals_steps": all(a == b for a, b in counts),
    })
    assert max(deviations) <= ceiling, f"归一化前的范数偏离超上限：{deviations}"


# ---------------------------------------------------------------------------
# 必须红：四条典型错法的门矩阵（轴7规则6）
# ---------------------------------------------------------------------------

#: 门矩阵用的轻量参数——与上面的判据同一批物理，步数压到够判定为止。
_FAULT_HORIZON_S = 1.0
_FAULT_DT_S = 2.0e-3
_FAULT_SPIN_HORIZON_S = 3.0
_FAULT_SPIN_DT_S = 1.0e-3


def _gate_verdicts() -> dict[str, bool]:
    """把六道门跑一遍，返回`门名 → 是否通过`。注错后哪几个变False就是它被谁抓住。"""

    box_inputs = BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs
    box = _box(box_inputs["box_half_extents_mm"])
    disc_entry = BY_ID["oracle:rigidbody/axisymmetric_precession_disc"]
    disc = _cylinder(disc_entry.inputs["radius_mm"], disc_entry.inputs["half_width_mm"])
    growth_entry = BY_ID["oracle:rigidbody/dzhanibekov_growth_rate"]
    torque_entry = BY_ID["oracle:rigidbody/constant_torque_ramp"]

    momentum, energy, _body, deviation, _count = _conservation_drifts(
        box, (1.0, 2.0, 3.0), _FAULT_HORIZON_S, _FAULT_DT_S
    )
    body_rate, inertial_rate = _precession_rates(
        disc, disc_entry.inputs["omega0_rad_per_s"], 2.0, _FAULT_DT_S
    )
    history = _perturbation_history(
        box, growth_entry.inputs["spin_axis"], growth_entry.inputs["spin_rad_per_s"],
        growth_entry.inputs["perturbation_rad_per_s"],
        _FAULT_SPIN_HORIZON_S, _FAULT_SPIN_DT_S,
    )
    rate = _log_slope(history, growth_entry.inputs["window_s"])
    axis, magnitude = torque_entry.inputs["torque_axis"], torque_entry.inputs["torque_nmm"]
    final, _diagnostics = _run(
        box, (0.0, 0.0, 0.0), torque_entry.inputs["horizon_s"],
        torque_entry.inputs["dt_s"],
        torque=lambda _y, _t: tuple(
            magnitude if index == axis else 0.0 for index in range(3)
        ),
    )
    return {
        "inertial_angular_momentum": momentum < 1.0e-6,
        "rotational_kinetic_energy": energy < 1.0e-6,
        "body_frame_precession": abs(
            body_rate / disc_entry.expected["body_precession_rate_per_s"] - 1.0
        ) < 1.0e-6,
        "inertial_frame_precession": abs(
            inertial_rate / disc_entry.expected["inertial_precession_rate_per_s"] - 1.0
        ) < 1.0e-6,
        "intermediate_axis_growth": abs(
            rate / growth_entry.expected["growth_rate_per_s"] - 1.0
        ) < 1.0e-2,
        "quaternion_norm": 0.0 < deviation <= 1.0e-11,
        "constant_torque_ramp": abs(
            final.block(_OMEGA_BLOCK)[2] / torque_entry.expected["omega_z_rad_per_s"] - 1.0
        ) < 1.0e-9,
    }


def _inject(monkeypatch, fault: str) -> None:
    if fault == "drop_gyroscopic_term":
        # `ω × (I·ω)`整项漏掉——最常见的一条，PhysX与Havok据Erez 2015就是这样。
        monkeypatch.setattr(rigidbody, "cross", lambda left, right: (0.0, 0.0, 0.0))
    elif fault == "cross_product_order":
        original = rigidbody.cross
        monkeypatch.setattr(rigidbody, "cross", lambda left, right: original(right, left))
    elif fault == "quaternion_order":
        original = rigidbody.quaternion_multiply
        monkeypatch.setattr(
            rigidbody, "quaternion_multiply", lambda left, right: original(right, left)
        )
    elif fault == "body_inertial_frame_swap":
        monkeypatch.setattr(
            rigidbody, "rotate_body_to_world", rigidbody.rotate_world_to_body
        )
    elif fault == "uniform_inertia_scaling":
        original = RigidBodyInertia.from_shape.__func__

        def scaled(cls, shape, **kwargs):
            base = original(cls, shape, **kwargs)
            return cls(
                mass_kg=base.mass_kg,
                inertia_body_kg_mm2=tuple(
                    tuple(value * 1.0e6 for value in row)
                    for row in base.inertia_body_kg_mm2
                ),
            )

        monkeypatch.setattr(RigidBodyInertia, "from_shape", classmethod(scaled))
    else:  # pragma: no cover - 参数化列表以外的名字进不来
        raise AssertionError(fault)


#: 逐条注错**实测**的红门集合。空集合=这条错法没被任何门抓住（第五条就是）。
#: 这张表是案例页第六节的正本，改判据必须同批改它。
FAULT_MATRIX: dict[str, frozenset[str]] = {
    "drop_gyroscopic_term": frozenset({
        "inertial_angular_momentum", "body_frame_precession",
        "inertial_frame_precession", "intermediate_axis_growth",
    }),
    "cross_product_order": frozenset({
        "inertial_angular_momentum", "body_frame_precession",
        "inertial_frame_precession",
    }),
    "quaternion_order": frozenset({
        "inertial_angular_momentum", "inertial_frame_precession",
    }),
    "body_inertial_frame_swap": frozenset({
        "inertial_angular_momentum", "inertial_frame_precession",
    }),
    "uniform_inertia_scaling": frozenset({"constant_torque_ramp"}),
}


def test_the_gate_matrix_is_green_before_any_fault_is_injected():
    """基线：不注错时七道门全绿。没有这一条，下面的"变红了"什么也不说明。"""

    verdicts = _gate_verdicts()
    failed = sorted(name for name, passed in verdicts.items() if not passed)
    assert not failed, f"未注错时就有门是红的：{failed}"


@pytest.mark.parametrize("fault", sorted(FAULT_MATRIX))
def test_each_typical_fault_turns_exactly_the_recorded_gates_red(monkeypatch, fault):
    """轴7规则6：门必须红过。**并且要记下哪些门是瞎的。**

    本仓的教训（决策0029）：有限差分门验不了物理，已有两个活标本。守恒量门有同类
    风险——`I → c·I`在无力矩下让每一条守恒量、每一个进动率、每一个增长率**逐位不变**,
    因为运动方程`ω̇ = I⁻¹(τ − ω × I·ω)`在`τ = 0`时对`I`是零次齐次的。
    所以`uniform_inertia_scaling`这一行只有常力矩那道门是红的，而它是本案例
    唯一带力矩的判据——**这条盲区是主动找出来的，不是被撞出来的**。
    """

    _inject(monkeypatch, fault)
    verdicts = _gate_verdicts()
    red = frozenset(name for name, passed in verdicts.items() if not passed)
    assert red == FAULT_MATRIX[fault], (
        f"注错`{fault}`的红门集合与登记不符：实测{sorted(red)}，"
        f"登记{sorted(FAULT_MATRIX[fault])}。**这张表是案例页第六节的正本**，"
        "判据变了就要同批改它，不许让它慢慢与事实脱节"
    )
    assert red, f"注错`{fault}`一条门都没红——这条错法今天没有任何门抓得住"


def test_the_conservation_gates_are_blind_to_a_uniform_scaling_of_the_inertia_tensor():
    """把盲区写成一条**正面**断言：无力矩轨迹对`I → c·I`不变。

    不是"我们没测到差别"，是"它按定义不该有差别"——`ω̇ = −I⁻¹(ω × I·ω)`里
    `I⁻¹`降一次、`I`升一次，`c`整体约掉。所以任何只跑无力矩的判据组
    （包括同行的Drake `free_body`与MuJoCo `dzhanibekov.xml`）都**证明不了**
    惯量的绝对量级对不对，而mm²↔m²恰好就是一个1e6的整体缩放。
    """

    inertia = _box(
        BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"]
    )
    scaled = RigidBodyInertia(
        mass_kg=inertia.mass_kg,
        inertia_body_kg_mm2=tuple(
            tuple(value * 1.0e6 for value in row) for row in inertia.inertia_body_kg_mm2
        ),
    )
    plain, _ = _run(inertia, (1.0, 2.0, 3.0), 1.0, 2.0e-3)
    blown_up, _ = _run(scaled, (1.0, 2.0, 3.0), 1.0, 2.0e-3)
    for a, b in zip(plain.vector, blown_up.vector, strict=True):
        assert abs(a - b) <= 1.0e-12 * max(1.0, abs(a)), (
            "无力矩轨迹对惯量整体缩放竟然敏感——那本条盲区的推理就错了，先改推理"
        )


def test_explicit_euler_is_rejected_by_the_quaternion_norm_guard_at_the_conservation_step():
    """护栏必须真的拦过东西：显式Euler在守恒判据的步长上第一步就被挡下。

    这不是把它排除在生产之外的**理由**（理由是反耗散），而是那条护栏**红过**的
    证据——一条从没红过的护栏与没有护栏是一回事。
    """

    inertia = _box(
        BY_ID["oracle:rigidbody/inertia_from_geometry"].inputs["box_half_extents_mm"]
    )
    with pytest.raises(rigidbody.RigidBodyError, match="before renormalisation"):
        _run(
            inertia, (1.0, 2.0, 3.0), 2.0, 2.0e-3, integrator=EXPLICIT_EULER_BODY
        )
