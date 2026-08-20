"""`cases/rolling_ball_incline`的门——plans/16的M2与M3。

**本文件判的三件事，一件都不能少：**

1. `a = (5/7)g·sinθ`（M2）；
2. 滑与不滑的分界`tanθ ≤ (7/2)μ`两侧各判一次（M3）；
3. **"无滑"是被判的，不是被假定的**——判据是`a/(αR)`这个比值：
   滚动时恰为1，滑动时是一个**可算的、不等于1的数**`k(tanθ/μ − 1)`。

第3条是0081第三节第4条设计的那条布尔判据。**没有它，前两条可以在一个
"永远按滚动公式算"的错误实现上一起绿**。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact_dynamics import sphere_plane_callbacks, sphere_plane_contact
from physics_engine.oracles import load_manifest
from physics_engine.rigidbody import (
    RK4_BODY,
    RigidBodyInertia,
    integrate_free_flight,
    make_state,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "cases/rolling_ball_incline/oracle.json"
pytestmark = pytest.mark.batch


@pytest.fixture(scope="module")
def oracles():
    manifest = load_manifest(MANIFEST)
    return {entry.id: entry for entry in manifest.oracles}


def _roll(friction: float, inputs: dict) -> dict:
    """跑一次，交出判据要读的四个量。**每个量都由状态算出来，没有一个是声明的。**"""

    radius = inputs["radius_mm"]
    mass = inputs["mass_kg"]
    gravity = inputs["gravity_mm_per_s2"]
    theta = math.radians(inputs["incline_deg"])
    dt = inputs["dt_s"]
    steps = int(inputs["steps"])

    normal = (math.sin(theta), 0.0, math.cos(theta))
    weight_n = mass * gravity / 1000.0
    inertia_scalar = 0.4 * mass * radius * radius
    inertia = RigidBodyInertia(
        mass_kg=mass,
        inertia_body_kg_mm2=(
            (inertia_scalar, 0.0, 0.0),
            (0.0, inertia_scalar, 0.0),
            (0.0, 0.0, inertia_scalar),
        ),
    )
    #: 初始高度扣掉静态压缩量，免得第一帧就是一次撞击。
    squash = weight_n * math.cos(theta) / inputs["normal_stiffness_n_per_mm"]
    state = make_state(position_mm=tuple((radius - squash) * c for c in normal))
    force, torque = sphere_plane_callbacks(
        radius_mm=radius,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=normal,
        normal_stiffness_n_per_mm=inputs["normal_stiffness_n_per_mm"],
        tangential_stiffness_n_per_mm=inputs["tangential_stiffness_n_s_per_mm"],
        friction_coefficient=friction,
        gravity_world_n=(0.0, 0.0, -weight_n),
    )
    final, _ = integrate_free_flight(
        RK4_BODY,
        state=state,
        inertia=inertia,
        dt_s=dt,
        steps=steps,
        force_world_n=force,
        torque_body_nmm=torque,
    )
    elapsed = dt * steps
    velocity = final.block("centre_of_mass_velocity_mm_per_s")
    omega = final.block("angular_velocity_body_rad_per_s")
    downhill = (math.cos(theta), 0.0, -math.sin(theta))
    acceleration = sum(velocity[i] * downhill[i] for i in range(3)) / elapsed
    angular = math.sqrt(sum(w * w for w in omega)) / elapsed
    response = sphere_plane_contact(
        final.vector,
        radius_mm=radius,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=normal,
        normal_stiffness_n_per_mm=inputs["normal_stiffness_n_per_mm"],
        tangential_stiffness_n_per_mm=inputs["tangential_stiffness_n_s_per_mm"],
        friction_coefficient=friction,
    )
    return {
        "acceleration_mm_per_s2": acceleration,
        "angular_acceleration_rad_per_s2": angular,
        "acceleration_over_alpha_radius": acceleration / (angular * radius),
        "sliding_flag": 1.0 if response.sliding else 0.0,
        "required_friction_n": 0.4 / 1.4 * (mass * gravity / 1000.0) * math.sin(theta),
        "response": response,
    }


def _check(entry, measured: dict) -> None:
    for quantity, expected in entry.expected.items():
        tolerance = entry.tolerances[quantity]
        got = measured[quantity]
        if tolerance.rel_tol == 0.0 and tolerance.abs_tol == 0.0:
            assert got == expected, f"{quantity}: {got!r} != {expected!r}（零容差）"
        else:
            limit = tolerance.rel_tol * abs(expected) + tolerance.abs_tol
            assert abs(got - expected) <= limit, (
                f"{quantity}: {got!r} 偏离 {expected!r} 超出 {limit!r}"
            )


@pytest.fixture(scope="module")
def branch_results(oracles):
    """同一轮验收内的两个确定性分支各算一次。

    旧测试重复算滚动支3次、滑动支2次；每次都是同一份已冻结
    manifest上的20000步纯函数积分。本fixture只活在当前pytest worker内，
    引擎字节、输入或算法一变仍会在本轮重算，不复用跨提交绿灯。
    """

    return {
        "rolling": _roll(
            oracles["oracle:rolling_ball/rolling_branch"].inputs["friction_coefficient"],
            oracles["oracle:rolling_ball/rolling_branch"].inputs,
        ),
        "sliding": _roll(
            oracles["oracle:rolling_ball/sliding_branch"].inputs["friction_coefficient"],
            oracles["oracle:rolling_ball/sliding_branch"].inputs,
        ),
    }


def test_the_exact_coefficients_are_five_sevenths_and_two_sevenths_and_seven_halves(oracles):
    """三条系数是精确有理数——**零容差**，它们是代数恒等式不是测量值。

    这条门守的是**生成器自己**：若有人把`SOLID_SPHERE_K`从实心球的`2/5`
    改成球壳的`2/3`，这里的三个数会一起变，而门会当场红。
    """

    from fractions import Fraction

    entry = oracles["oracle:rolling_ball/exact_coefficients"]
    k = Fraction(
        int(entry.inputs["inertia_coefficient_num"]),
        int(entry.inputs["inertia_coefficient_den"]),
    )
    measured = {
        "rolling_coefficient": float(1 / (1 + k)),
        "required_friction_coefficient": float(k / (1 + k)),
        "no_slip_limit_coefficient": float((1 + k) / k),
        "critical_friction_at_incline": math.tan(math.radians(20.0)) / float((1 + k) / k),
    }
    _check(entry, measured)
    assert measured["rolling_coefficient"] == float(Fraction(5, 7))
    assert measured["no_slip_limit_coefficient"] == 3.5


def test_the_rolling_branch_matches_five_sevenths_g_sin_theta(oracles, branch_results):
    """M2：`μ > μc`时球无滑滚下坡，`a = (5/7)g·sinθ`。

    **`a/(αR)`必须为1**——那是"无滑"这件事本身，不是它的推论。
    """

    entry = oracles["oracle:rolling_ball/rolling_branch"]
    measured = branch_results["rolling"]
    _check(entry, measured)
    assert measured["sliding_flag"] == 0.0


def test_the_sliding_branch_matches_the_saturated_closed_form(oracles, branch_results):
    """M3：`μ < μc`时球滑着滚，`a = g(sinθ − μcosθ)`、`α = 5μg cosθ/(2R)`。

    **要害是第三个量**：`a/(αR) = k(tanθ/μ − 1) ≈ 2.512`，**它不等于1**。
    """

    entry = oracles["oracle:rolling_ball/sliding_branch"]
    measured = branch_results["sliding"]
    _check(entry, measured)
    assert measured["sliding_flag"] == 1.0


def test_the_two_branches_are_told_apart_by_the_ratio_not_by_the_formula(branch_results):
    """**必红那一半**：两支的`a/(αR)`必须显著不同，否则那条判据没有分辨力。

    若某个实现"永远按滚动公式算"，两支的比值会**都是1**，
    而上面两条门里的`a`还可能各自碰巧落在容差内。**这条把那扇门关上。**
    """

    rolling = branch_results["rolling"]
    sliding = branch_results["sliding"]
    assert rolling["acceleration_over_alpha_radius"] < 1.01
    assert sliding["acceleration_over_alpha_radius"] > 2.0, (
        "滑动支的a/(αR)没有明显离开1 —— 那这条判据分不出两支"
    )
    assert rolling["sliding_flag"] != sliding["sliding_flag"]


def test_the_normal_penalty_makes_exactly_zero_torque_on_a_sphere(branch_results):
    """球上法向罚力对质心的力矩**逐位为零**——`r ∥ n̂`，结构性的不是近似的。

    这条不判物理，判**实现里有没有多出一项**。
    """

    measured = branch_results["rolling"]
    for component in measured["response"].normal_torque_body_nmm:
        assert component.hex() == (0.0).hex(), (
            f"法向力产生了力矩{component!r} —— 球的杆臂沿n̂、法向力也沿n̂，"
            "它们的叉积必须是逐位的零"
        )
