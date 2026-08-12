"""瞬态罚接触的判据门（阶段1，决策0052）。

**这条案例第一次让"瞬态接触"从既成事实变成被声明的能力。**
在它之前，罚接触+显式积分跑得通是plans/08在调研里顺手测出来的，
仓里**没有一条判据守着它**。

判据全在`cases/bouncing_ball_restitution/oracle.json`里，本文件不复述公式
（轴7规则3）——它只驱动引擎、取数、跟金标比。

**驱动的是真路径**：`state`→`energies`（`PenaltyNormalContact`）
→`acceleration`桥→`integrate`。不是在测试里手写一个弹簧。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from physics_engine.contact import PenaltyNormalContact
from physics_engine.energies import EnergyContext, EnergyRegistry
from physics_engine.integrate import INTEGRATORS, advise_step, integrate
from physics_engine.state import StateField, StateLayout

CASE_DIR = Path(__file__).resolve().parents[2] / "cases" / "bouncing_ball_restitution"
VERLET = INTEGRATORS["velocity_verlet"]
COEFFICIENT = VERLET.declaration.oscillatory_step_coefficient


@pytest.fixture(scope="module")
def oracle() -> dict:
    document = json.loads((CASE_DIR / "oracle.json").read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in document["oracles"]}


def _tol(entry: dict, key: str) -> dict:
    return entry["tolerances"][key]


def _bounce(
    *,
    stiffness_n_per_mm: float,
    mass_kg: float,
    incident_speed_mm_s: float,
    steps_per_contact: int,
    radius_mm: float = 0.0,
) -> dict:
    """跑一次撞击，返回`(接触时长, 最大穿透, 恢复系数)`。

    无重力：接触段是纯简谐半周期，三条闭式因此**精确**。
    带重力的完整弹跳（总时间级数）属阶段2，见案例页第六节。
    """

    layout = StateLayout(
        layout_id="layout/bouncing_ball_restitution",
        fields=(
            StateField("node0_x_mm", 1),
            StateField("node0_y_mm", 1),
            StateField("node0_z_mm", 1),
        ),
    )
    registry = EnergyRegistry(
        terms=(
            PenaltyNormalContact(
                planes=((0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), stiffness_n_per_mm, radius_mm),),
            ),
        )
    )
    context = EnergyContext(
        context_id="context/bouncing_ball_restitution",
        node_masses_kg=(mass_kg,),
        gravity_mm_s2=(0.0, 0.0, 0.0),
    )
    acceleration = registry.acceleration(context, layout)

    omega = math.sqrt(1000.0 * stiffness_n_per_mm / mass_kg)
    step_s = advise_step(
        omega,
        oscillatory_step_coefficient=COEFFICIENT,
        steps_per_contact=steps_per_contact,
    ).advised_step_s

    x, v, t = (0.0, 0.0, 0.0), (0.0, 0.0, -incident_speed_mm_s), 0.0
    deepest_mm = 0.0
    #: 上限按"至多两倍接触时长"给，不给魔数——接触不结束就是缺陷不是慢。
    max_steps = 4 * steps_per_contact
    for _ in range(max_steps):
        x, v, t = integrate(
            VERLET, x0=x, v0=v, dt_s=step_s, steps=1, acceleration=acceleration, t0_s=t
        )
        gap_mm = x[2] - radius_mm
        deepest_mm = min(deepest_mm, gap_mm)
        if gap_mm >= 0.0:
            return {
                "contact_duration_s": t,
                "max_penetration_mm": -deepest_mm,
                "restitution": abs(v[2]) / incident_speed_mm_s,
                "omega_rad_per_s": omega,
            }
    raise AssertionError(
        f"接触在{max_steps}步内没有结束——罚接触下存在临界入射速度，"
        "低于它接触不结束（research/13第六节）。这条断言就是那件事的兜底"
    )


def test_contact_duration_matches_the_half_period(oracle):
    """判据A1：`t_c = π/ω`。**与入射速度无关**——那是简谐半周期的性质。"""

    entry = oracle["oracle:bounce/contact_duration"]
    inputs, expected = entry["inputs"], entry["expected"]
    measured = _bounce(
        stiffness_n_per_mm=inputs["stiffness_n_per_mm"],
        mass_kg=inputs["mass_kg"],
        incident_speed_mm_s=inputs["incident_speed_mm_s"],
        steps_per_contact=200,
    )
    assert measured["omega_rad_per_s"] == pytest.approx(
        expected["omega_rad_per_s"], rel=_tol(entry, "omega_rad_per_s")["rel"]
    )
    assert measured["contact_duration_s"] == pytest.approx(
        expected["contact_duration_s"], rel=_tol(entry, "contact_duration_s")["rel"]
    )


def test_max_penetration_matches_the_transient_amplitude(oracle):
    """判据A2：`δ_max = v_in/ω`。"""

    entry = oracle["oracle:bounce/max_penetration"]
    inputs, expected = entry["inputs"], entry["expected"]
    measured = _bounce(
        stiffness_n_per_mm=inputs["stiffness_n_per_mm"],
        mass_kg=inputs["mass_kg"],
        incident_speed_mm_s=inputs["incident_speed_mm_s"],
        steps_per_contact=20,
    )
    assert measured["max_penetration_mm"] == pytest.approx(
        expected["max_penetration_mm"], rel=_tol(entry, "max_penetration_mm")["rel"]
    )


def test_undamped_restitution_is_one(oracle):
    """判据A3：无阻尼`e = 1`。保守力场，出射速率等于入射速率。"""

    entry = oracle["oracle:bounce/restitution_undamped"]
    inputs, expected = entry["inputs"], entry["expected"]
    measured = _bounce(
        stiffness_n_per_mm=inputs["stiffness_n_per_mm"],
        mass_kg=inputs["mass_kg"],
        incident_speed_mm_s=inputs["incident_speed_mm_s"],
        steps_per_contact=20,
    )
    assert measured["restitution"] == pytest.approx(
        expected["restitution"], abs=_tol(entry, "restitution")["abs"]
    )


def test_restitution_error_shrinks_monotonically_with_steps_per_contact(oracle):
    """判据A4：步数越多越准，且**误差恒为正**。

    正号本身是物理：**积分误差往碰撞里喂能量，不往外抽**。
    plans/08实测2步/接触时`e = 1.1433`——大于1。
    """

    entry = oracle["oracle:bounce/restitution_converges_monotonically"]
    inputs = entry["inputs"]
    errors = [
        _bounce(
            stiffness_n_per_mm=inputs["stiffness_n_per_mm"],
            mass_kg=inputs["mass_kg"],
            incident_speed_mm_s=inputs["incident_speed_mm_s"],
            steps_per_contact=n,
        )["restitution"]
        - 1.0
        for n in inputs["steps_per_contact"]
    ]
    assert all(error >= 0.0 for error in errors), (
        f"恢复系数误差出现负号：{errors}——"
        "无阻尼保守力场下`e < 1`意味着能量被凭空拿走了，那是缺陷不是精度"
    )
    assert errors[0] > 0.0, "最粗档误差为0，下面的单调性断言会假通过"
    assert errors == sorted(errors, reverse=True), f"误差没有随步数单调下降：{errors}"


# ---------------------------------------------------------------------------
# 必红（决策0052第五节点名的那一条）
# ---------------------------------------------------------------------------


def test_the_quasistatic_penetration_law_is_wrong_here_and_stiffness_proves_it(oracle):
    """**必红本体**：把`δ_max`判据写成准静态的`N/k`，刚度换一档必须红。

    两条律的**标度不同**：

    * 准静态 `δ = N/k` → `O(1/k)`，刚度×10则穿透÷**10**；
    * 瞬态 `δ_max = v_in/ω` → `O(k^(−1/2))`，刚度×10则穿透÷**sqrt(10)≈3.16**。

    单一刚度上两者可以靠调系数对上，**跨刚度对不上**——差3.16倍。

    这条防的是plans/08第零节记的那次口径混用：
    `PenaltyNormalContact`的docstring此前把`δ = N/k`写成无条件成立，
    **而瞬态冲击下k=1e5时它差1010倍**。

    本用例同时充当那条判据的必红：如果有人把判据换成准静态律，
    这里的比值会从3.16变成10，断言当场红。
    """

    entry = oracle["oracle:bounce/penetration_scales_as_inverse_sqrt_k"]
    inputs, expected = entry["inputs"], entry["expected"]
    sweep = inputs["stiffness_sweep_n_per_mm"]

    penetrations = [
        _bounce(
            stiffness_n_per_mm=k,
            mass_kg=inputs["mass_kg"],
            incident_speed_mm_s=inputs["incident_speed_mm_s"],
            steps_per_contact=200,
        )["max_penetration_mm"]
        for k in sweep
    ]
    #: `strict=False`是对的：相邻配对本来就短一个。写出来是因为
    #: 默认值在这里恰好正确，而"恰好正确"不该靠默认值沉默地成立。
    ratios = [
        previous / current
        for previous, current in zip(penetrations, penetrations[1:], strict=False)
    ]

    for ratio in ratios:
        assert ratio == pytest.approx(
            expected["penetration_ratio_per_decade"],
            rel=_tol(entry, "penetration_ratio_per_decade")["rel"],
        ), (
            f"每十倍刚度的穿透比是{ratio:.4f}，瞬态律要求"
            f"{expected['penetration_ratio_per_decade']:.4f}——"
            f"准静态律会预测{expected['quasistatic_would_predict']}。"
            "**这两个数差3.16倍，不是精度问题是用错了律**"
        )
        assert abs(ratio - expected["quasistatic_would_predict"]) > 1.0, (
            "实测比值落在准静态律那个数上了——要么模型变了，要么这条必红失效了"
        )


def test_the_advisor_is_what_picks_the_step_here():
    """结构断言：本案例的步长**由步长顾问算**，不是硬编码的魔数。

    案例里写死一个步长，等于把"这个步长为什么够"这个问题藏起来。
    顾问给的`stability_margin = π/(2N)`，与plans/08那张实测表逐行对上。
    """

    advice = advise_step(
        math.sqrt(1000.0 * 1.0e4 / 1.0e-3),
        oscillatory_step_coefficient=COEFFICIENT,
        steps_per_contact=200,
    )
    assert advice.binding == "contact_resolution"
    assert advice.stability_margin == pytest.approx(math.pi / 400.0, rel=1e-14)
