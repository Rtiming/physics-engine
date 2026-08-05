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
    IntegrateError,
    IntegratorDeclaration,
    PurePythonOps,
    default_ops,
    integrate,
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
