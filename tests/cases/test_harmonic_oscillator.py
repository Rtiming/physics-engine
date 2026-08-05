"""`case/harmonic_oscillator`的conformance门（轴7规则3）。

判据数全部来自清单。两条被特别写明的纪律：

* **收敛比是区间不是"恰为4"**——4是渐近值，粗档还没完全进渐近区（实测3.9985）；
* **漂移排序先各自断非零**——三个积分器若都返回初值，排序断言会在全零输入上假通过。
"""

from __future__ import annotations

import math
from pathlib import Path

from physics_engine.integrate import EXPLICIT_EULER, INTEGRATORS, VELOCITY_VERLET, integrate
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/harmonic_oscillator/oracle.json", root=ROOT)
BY_ID = {entry.id: entry for entry in MANIFEST.oracles}


def _run(integrator, omega_per_s, dt_s, steps, x0=1.0, v0=0.0):
    def acceleration(x, v, t):
        return (-omega_per_s * omega_per_s * x[0],)

    position, velocity, _ = integrate(
        integrator, x0=(x0,), v0=(v0,), dt_s=dt_s, steps=steps,
        acceleration=acceleration,
    )
    return position[0], velocity[0]


def test_analytic_position_matches_the_manifest():
    entry = BY_ID["oracle:harmonic/analytic_position"]
    omega = entry.inputs["omega_per_s"]
    entry.check_all({"position_mm": math.cos(omega * entry.inputs["horizon_s"])})


def test_velocity_verlet_converges_at_second_order():
    entry = BY_ID["oracle:harmonic/verlet_order_ratio"]
    omega, horizon = entry.inputs["omega_per_s"], entry.inputs["horizon_s"]
    truth = math.cos(omega * horizon)
    errors = []
    for dt_s in entry.inputs["dt_s_ladder"]:
        position, _ = _run(VELOCITY_VERLET, omega, dt_s, int(round(horizon / dt_s)))
        errors.append(abs(position - truth))
    assert all(error > 0.0 for error in errors), (
        f"误差全为零——离散误差被别的东西吞了，收敛阶无从谈起：{errors}"
    )
    low, high = entry.expected["ratio_low"], entry.expected["ratio_high"]
    ratios = [a / b for a, b in zip(errors, errors[1:], strict=False)]
    for ratio in ratios:
        assert low <= ratio <= high, (
            f"收敛比 {ratio!r} 落在区间[{low}, {high}]之外——实测阶偏离形式二阶。"
            f"逐档误差 {errors}"
        )
    assert VELOCITY_VERLET.declaration.formal_order == entry.expected["formal_order"]


def test_energy_drift_ordering_holds_and_no_integrator_is_silently_frozen():
    entry = BY_ID["oracle:harmonic/drift_ordering"]
    omega = entry.inputs["omega_per_s"]
    dt_s, steps = entry.inputs["dt_s"], entry.inputs["steps"]
    initial_energy = 0.5 * omega * omega * entry.inputs["initial_position_mm"] ** 2
    drifts = {}
    for name in entry.expected["ordering"]:
        position, velocity = _run(
            INTEGRATORS[name], omega, dt_s, steps,
            x0=entry.inputs["initial_position_mm"],
            v0=entry.inputs["initial_velocity_mm_s"],
        )
        energy = 0.5 * velocity * velocity + 0.5 * omega * omega * position * position
        drifts[name] = abs(energy - initial_energy) / initial_energy

    # 前置断言（MuJoCo `EXPECT_NE`形制）：三者全零时排序会假通过。
    entry.check_all({
        "ordering": entry.expected["ordering"],
        "all_nonzero": all(drift != 0.0 for drift in drifts.values()),
    })
    ordered = [drifts[name] for name in entry.expected["ordering"]]
    assert ordered == sorted(ordered, reverse=True) and len(set(ordered)) == len(ordered), (
        f"漂移排序不成立：{drifts}"
    )


def test_explicit_euler_energy_grows_which_is_why_it_is_not_production_ready():
    """显式Euler是反耗散的——这条是它`production_ready=False`的物理理由，不是风格判断。"""

    omega = 2.0
    position, velocity = _run(EXPLICIT_EULER, omega, 0.01, 2000)
    initial_energy = 0.5 * omega * omega
    energy = 0.5 * velocity * velocity + 0.5 * omega * omega * position * position
    assert energy > initial_energy
    assert EXPLICIT_EULER.declaration.production_ready is False
