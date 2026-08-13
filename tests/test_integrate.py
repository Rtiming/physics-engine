"""积分器接口与出生声明的门（spec/12第4.2节）。

物理判据不在这里——那些在`cases/ballistic_free_flight`与`cases/harmonic_oscillator`
的清单里（轴7规则3：不得在测试里复述oracle公式）。本文件只守**接口契约**：
五项声明缺一不得进仓、显式积分器必须给步长上界、失败关闭的输入校验。
"""

from __future__ import annotations

import pytest

from physics_engine.integrate import (
    EXPLICIT_EULER,
    INTEGRATORS,
    SYMPLECTIC_EULER,
    VELOCITY_VERLET,
    VELOCITY_VERLET_DAMPED,
    DissipativeIntegrationResult,
    IntegrateError,
    IntegratorDeclaration,
    PurePythonOps,
    default_ops,
    integrate,
    integrate_with_dissipation,
)

_COMPLETE = {
    "name": "toy",
    "scope_excludes": "不管接触",
    "formal_order": 1,
    "measured_order": "1",
    "stability": "symplectic",
    "step_bound": "h < 2/ω_max",
    "dissipation_accounting": "无",
    "failure_ladder": "无",
    "production_ready": False,
}


@pytest.mark.parametrize(
    "missing",
    ["scope_excludes", "measured_order", "step_bound", "dissipation_accounting",
     "failure_ladder"],
)
def test_each_of_the_five_birth_declarations_is_mandatory(missing):
    """spec/12第4.2节：五项声明缺一不得进仓。空字符串等同于缺。"""

    fields = dict(_COMPLETE) | {missing: "   "}
    with pytest.raises(IntegrateError, match=missing):
        IntegratorDeclaration(**fields)


def test_a_conditionally_stable_integrator_must_state_its_step_bound():
    """显式积分器不写步长上界就是埋雷——写"无"也不行。"""

    fields = dict(_COMPLETE) | {"stability": "explicit_conditional", "step_bound": "无"}
    with pytest.raises(IntegrateError, match="step bound"):
        IntegratorDeclaration(**fields)


def test_every_shipped_integrator_carries_all_five_declarations():
    for name, integrator in INTEGRATORS.items():
        declaration = integrator.declaration
        assert declaration.name == name
        for field_name in ("scope_excludes", "measured_order", "step_bound",
                           "dissipation_accounting", "failure_ladder"):
            assert str(getattr(declaration, field_name)).strip(), (
                f"{name} 的 {field_name} 是空的"
            )


def test_no_integrator_claims_production_readiness_yet():
    """三个都是显式/辛族，无接触无隐式——今天没有一个够生产用，声明里必须这么说。"""

    assert not any(
        integrator.declaration.production_ready for integrator in INTEGRATORS.values()
    )


def test_the_two_euler_variants_differ_only_in_update_order():
    """两者的全部差别就是"先推位置还是先更新速度"，声明里的实测阶反映这一点。"""

    assert EXPLICIT_EULER.declaration.stability == "explicit_conditional"
    assert SYMPLECTIC_EULER.declaration.stability == "symplectic"
    assert VELOCITY_VERLET.declaration.formal_order == 2


def test_the_velocity_dependent_integrator_measures_second_order_under_damping():
    """阶段2主门：速度相关力下误差缩放必须趋近4，而不是只在声明里写“二阶”。"""

    damping_ratio = 0.2
    omega = 3.0
    duration = 1.0
    damped_omega = omega * math.sqrt(1.0 - damping_ratio * damping_ratio)
    exact_x = math.exp(-damping_ratio * omega * duration) * (
        math.cos(damped_omega * duration)
        + damping_ratio / math.sqrt(1.0 - damping_ratio * damping_ratio)
        * math.sin(damped_omega * duration)
    )

    errors = []
    for steps in (50, 100, 200, 400):
        x, _, _ = integrate(
            VELOCITY_VERLET_DAMPED,
            x0=(1.0,),
            v0=(0.0,),
            dt_s=duration / steps,
            steps=steps,
            acceleration=lambda x, v, t: (
                -omega * omega * x[0] - 2.0 * damping_ratio * omega * v[0],
            ),
        )
        errors.append(abs(x[0] - exact_x))

    ratios = [
        coarse / fine for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    ]
    assert all(3.8 < ratio < 4.3 for ratio in ratios), (
        f"速度相关力下没有实测到二阶：errors={errors}, ratios={ratios}"
    )


def test_the_damped_integrator_does_not_replace_the_old_undamped_route():
    """必须红：新积分器只能新增；把``velocity_verlet``映射偷偷换掉会立即红。"""

    assert INTEGRATORS["velocity_verlet"] is VELOCITY_VERLET
    assert INTEGRATORS["velocity_verlet_damped"] is VELOCITY_VERLET_DAMPED
    assert VELOCITY_VERLET_DAMPED is not VELOCITY_VERLET


def test_the_global_damped_stability_coefficient_is_measured_at_critical_damping():
    """ζ=1是全阻尼范围最紧点：``h·ω0=1``内稳、越过即爆。"""

    coefficient = VELOCITY_VERLET_DAMPED.declaration.oscillatory_step_coefficient
    assert coefficient == 1.0

    def final_norm(h_omega: float) -> float:
        x, v, _ = integrate(
            VELOCITY_VERLET_DAMPED,
            x0=(1.0,),
            v0=(0.0,),
            dt_s=h_omega,
            steps=1000,
            acceleration=lambda x, v, t: (-x[0] - 2.0 * v[0],),
        )
        return x[0] * x[0] + v[0] * v[0]

    assert final_norm(0.995 * coefficient) < 1.0
    assert final_norm(1.005 * coefficient) > 1.0e6


def test_physical_dissipation_is_integrated_and_returned_without_changing_integrate():
    """耗散记账是新返回形制；旧``integrate``三元组保持不变。"""

    result = integrate_with_dissipation(
        VELOCITY_VERLET_DAMPED,
        x0=(0.0,),
        v0=(2.0,),
        dt_s=0.01,
        steps=10,
        acceleration=lambda x, v, t: (0.0,),
        dissipation_rate=lambda x, v, t: 3.0,
    )
    assert isinstance(result, DissipativeIntegrationResult)
    assert result.x == pytest.approx((0.2,), rel=1e-15)
    assert result.v == (2.0,)
    assert result.t_s == pytest.approx(0.1, rel=1e-15)
    assert result.dissipated_energy_nmm == pytest.approx(0.3, rel=1e-15)
    assert len(integrate(
        VELOCITY_VERLET, x0=(0.0,), v0=(0.0,), dt_s=0.1, steps=1,
        acceleration=lambda x, v, t: (0.0,),
    )) == 3


@pytest.mark.parametrize("bad_rate", [-1.0, float("nan"), float("inf")])
def test_dissipation_accounting_fails_closed_on_nonphysical_rates(bad_rate):
    """必须红：负耗散或非有限耗散不能被静默累计成一份貌似正常的账。"""

    with pytest.raises(IntegrateError, match="dissipation rate"):
        integrate_with_dissipation(
            VELOCITY_VERLET_DAMPED,
            x0=(0.0,), v0=(0.0,), dt_s=0.1, steps=1,
            acceleration=lambda x, v, t: (0.0,),
            dissipation_rate=lambda x, v, t: bad_rate,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dt_s": 0.0}, "dt_s must be positive"),
        ({"dt_s": -1.0}, "dt_s must be positive"),
        ({"steps": -1}, "steps must be nonnegative"),
        ({"v0": (0.0, 0.0)}, "same length"),
    ],
)
def test_integrate_fails_closed_on_bad_arguments(kwargs, match):
    call = {"x0": (0.0,), "v0": (0.0,), "dt_s": 0.01, "steps": 1,
            "acceleration": lambda x, v, t: (0.0,)}
    call.update(kwargs)
    with pytest.raises(IntegrateError, match=match):
        integrate(VELOCITY_VERLET, **call)


def test_zero_steps_returns_the_initial_state_unchanged():
    x, v, t = integrate(
        VELOCITY_VERLET, x0=(1.0,), v0=(2.0,), dt_s=0.01, steps=0,
        acceleration=lambda x, v, t: (0.0,), t0_s=5.0,
    )
    assert (x, v, t) == ((1.0,), (2.0,), 5.0)


def test_the_default_backend_is_pure_python_and_never_requires_numpy():
    """0014零设施承诺：没有NumPy的机器上全部公开操作必须可用。"""

    assert isinstance(default_ops(), PurePythonOps)
    x, _, _ = integrate(
        VELOCITY_VERLET, x0=(0.0,), v0=(0.0,), dt_s=0.1, steps=1,
        acceleration=lambda x, v, t: (2.0,),
    )
    # ½·a·h² = 0.01；写approx而不是写死0.01——`0.5*0.1*0.1*2`的双精度结果是
    # 0.010000000000000002，把朴素十进制值当期望是**测试写错**不是内核错。
    assert x[0] == pytest.approx(0.01, rel=1e-12)


# ---------------------------------------------------------------------------
# 步长顾问（决策0052第五节）——把`step_bound`那个字符串变成可计算的数
# ---------------------------------------------------------------------------

import math  # noqa: E402

from physics_engine.integrate import (  # noqa: E402
    DEFAULT_STEPS_PER_CONTACT,
    MIN_MEANINGFUL_STEPS_PER_CONTACT,
    PRODUCTION_READY_CONDITIONS,
    advise_step,
)

VERLET_COEFFICIENT = VELOCITY_VERLET.declaration.oscillatory_step_coefficient


def _oscillator_energy_ratio(integrator, h_omega: float, cycles: int = 40) -> float:
    """线性振子跑`cycles`周期的能量比。`ω=1`故`h = h·ω`。

    这是`oscillatory_step_coefficient`那几个数的**生产者**——
    声明里写的数必须能被这个函数重新测出来，否则它就只是个断言。
    """

    omega = 1.0
    steps = max(int(round(cycles * 2 * math.pi / h_omega)), 1)
    x, v, _ = integrate(
        integrator,
        x0=(1.0,),
        v0=(0.0,),
        dt_s=h_omega,
        steps=steps,
        acceleration=lambda x, v, t: (-omega * omega * x[0],),
    )
    return (0.5 * v[0] ** 2 + 0.5 * omega * omega * x[0] ** 2) / 0.5


@pytest.mark.parametrize("integrator", [SYMPLECTIC_EULER, VELOCITY_VERLET])
def test_the_declared_coefficient_is_where_the_oscillator_actually_breaks(integrator):
    """声明的系数**必须是可证伪的**：在它上面有界，越过它就发散。

    这条守的不是顾问，是**声明的诚实度**。一个写在docstring里的数
    没有生产者就会漂——`source_bytes`台账漂过0.81%，`accept full`的记账
    漂过43倍，两次都是"没有门看着的数字被文档当成现在时引用"。
    """

    coefficient = integrator.declaration.oscillatory_step_coefficient
    assert coefficient is not None

    inside = _oscillator_energy_ratio(integrator, coefficient * 0.995)
    outside = _oscillator_energy_ratio(integrator, coefficient * 1.005)
    assert inside < 1e3, f"声明的界内侧就已经发散了：能量比{inside:.3e}"
    assert outside > 1e6, f"越过声明的界却没有发散：能量比{outside:.3e}——这个系数偏小"


def test_explicit_euler_has_no_usable_step_bound_and_says_so():
    """**实测把一条声明打脸了**：`explicit_euler`在任何步长下都发散。

    它此前那格写的是"h < 2/ω_max（线性振子）"，读起来像个可用的界。
    2/ω是**实轴**稳定区半径，而线性振子的特征值在**虚轴**上，
    放大因子恒为`sqrt(1+h²ω²) > 1`。

    2026-08-12实测：`h·ω=0.1`跑40周期能量已涨到**7.2e10倍**。
    """

    assert EXPLICIT_EULER.declaration.oscillatory_step_coefficient is None
    assert _oscillator_energy_ratio(EXPLICIT_EULER, 0.1) > 1e6, (
        "如果它在h·ω=0.1时不发散了，说明积分器被改过——回来重定这条声明"
    )


def test_advise_step_matches_the_closed_form():
    """两个界都是闭式，顾问不许算错。"""

    omega = 3162.277660168379  # sqrt(1e4 / 1e-3)
    advice = advise_step(omega, oscillatory_step_coefficient=VERLET_COEFFICIENT)

    assert advice.stability_bound_s == pytest.approx(VERLET_COEFFICIENT / omega, rel=1e-15)
    assert advice.contact_resolution_bound_s == pytest.approx(
        math.pi / (DEFAULT_STEPS_PER_CONTACT * omega), rel=1e-15
    )
    assert advice.advised_step_s == min(
        advice.stability_bound_s, advice.contact_resolution_bound_s
    )


def test_contact_resolution_is_the_binding_bound_not_stability():
    """**这是整条阶段1的要害**：管事的是分辨界，不是稳定界。

    两者的失败方式完全不同——撞稳定界是**爆掉**（看得见），
    撞分辨界是**静默地算出一个错的恢复系数**（看不见）。
    plans/08实测：在声明的稳定界内侧0.785倍处，恢复系数已经错**14.3%**。
    """

    advice = advise_step(1000.0, oscillatory_step_coefficient=VERLET_COEFFICIENT)
    assert advice.binding == "contact_resolution"
    assert advice.contact_resolution_bound_s < advice.stability_bound_s


def test_damped_contact_can_supply_its_actual_force_zero_duration():
    """阻尼接触不再假装持续π/ω；分辨界必须按三段闭式给出的真实时长。"""

    advice = advise_step(
        1000.0,
        oscillatory_step_coefficient=1.0,
        steps_per_contact=20,
        contact_duration_s=0.004,
    )
    assert advice.contact_duration_s == 0.004
    assert advice.contact_resolution_bound_s == pytest.approx(0.004 / 20, rel=1e-15)
    assert advice.advised_step_s == advice.contact_resolution_bound_s


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_advisor_fails_closed_on_bad_contact_duration(duration):
    with pytest.raises(IntegrateError, match="contact_duration_s"):
        advise_step(
            1000.0,
            oscillatory_step_coefficient=1.0,
            contact_duration_s=duration,
        )


def test_stability_margin_is_pi_over_two_n():
    """`建议/稳定界 = π/(2N)`——这条闭式让plans/08那张实测表对得上。

    N=2 → 0.785（plans/08那一行实测恢复系数1.1433，错14.3%）；
    N=20 → 0.0785（实测0.99984）；N=40 → 0.0393（实测0.99999）。
    """

    for n in (4, 20, 40):
        advice = advise_step(
            500.0, oscillatory_step_coefficient=2.0, steps_per_contact=n
        )
        assert advice.stability_margin == pytest.approx(math.pi / (2 * n), rel=1e-14)


def test_advisor_refuses_an_integrator_with_no_usable_bound():
    """**必红**：`explicit_euler`没有可用的界，顾问必须拒绝而不是给个数。

    给个数才是最坏的——使用者会照着用。
    """

    with pytest.raises(IntegrateError, match="没有可用的步长上界"):
        advise_step(
            100.0,
            oscillatory_step_coefficient=EXPLICIT_EULER.declaration.oscillatory_step_coefficient,
        )


@pytest.mark.parametrize("omega", [0.0, -1.0, float("nan"), float("inf")])
def test_advisor_fails_closed_on_bad_omega(omega):
    """**必红**：零频/负频/非有限频不是"不限步长"，是没有定义。

    这条是顾问唯一能挡的——`ω_max`由调用方自己算，**算错了顾问看不出来**
    （0052第五节如实登记的代价）。
    """

    with pytest.raises(IntegrateError):
        advise_step(omega, oscillatory_step_coefficient=2.0)


def test_advisor_fails_closed_below_the_meaningful_step_count():
    """**必红**：步数太少不只是不准，是**定性错**。

    plans/08实测2步/接触时恢复系数是1.1433——**大于1**，
    积分误差把能量喂进了碰撞。一个能算出`e > 1`的配置不该被建议出来。
    """

    with pytest.raises(IntegrateError, match="定性错"):
        advise_step(
            100.0,
            oscillatory_step_coefficient=2.0,
            steps_per_contact=MIN_MEANINGFUL_STEPS_PER_CONTACT - 1,
        )


def test_integrate_module_still_imports_nothing_from_the_package():
    """**结构断言（0052第五节的裁决前提）**：`integrate.py`包内import为0。

    这条独立性意味着积分器可以被单独拿走用。步长顾问放进本模块的**唯一条件**
    就是它只吃纯数字——顾问不该是破掉这条的那个。
    """

    from pathlib import Path

    import physics_engine.integrate as integrate_module

    source = Path(integrate_module.__file__).read_text(encoding="utf-8")
    offenders = [
        line
        for line in source.splitlines()
        if line.startswith(("from physics_engine", "import physics_engine", "from ."))
    ]
    assert offenders == [], f"integrate.py开始import包内东西了：{offenders}"


def test_production_ready_conditions_are_written_down_and_nothing_is_flipped_yet():
    """`production_ready`翻转条件必须写死（0052第六节），而这一轮**不翻**。

    不定条件，它就是个永远翻不了的死字段——0039写过同源的规矩：
    "绊线一旦长期不响就等于被拆了"。
    """

    assert len(PRODUCTION_READY_CONDITIONS) == 4
    for integrator in INTEGRATORS.values():
        assert integrator.declaration.production_ready is False


def test_a_nonpositive_coefficient_is_rejected_at_declaration_time():
    """**必红**：系数是`h·ω`的上界，必须为正；"没有可用的界"要写None。"""

    with pytest.raises(IntegrateError, match="必须为正"):
        IntegratorDeclaration(
            name="bad",
            scope_excludes="x",
            formal_order=1,
            measured_order="1",
            stability="symplectic",
            step_bound="h < 2/ω",
            dissipation_accounting="none",
            failure_ladder="none",
            production_ready=False,
            oscillatory_step_coefficient=0.0,
        )
