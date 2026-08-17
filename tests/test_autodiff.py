"""共享jet的门（决策0064第4.1节把它从`section_beam`提升上来）。

**数值正确性不在这里重复验**：`tests/test_section_beam.py`那三串WDS对拍常数
（曲率、能量、十一分量梯度）在提升前后逐字节相同，那就是搬运没改数学的判据。
本文件只守提升本身带进来的三件新事：

* 失败关闭的类型变了（`AutodiffError`），且它仍是`ValueError`的子类——
  `solve_equilibrium`线搜索那句`except (ValueError, ZeroDivisionError)`还接得住；
* 两个阶次的**值通道逐字节一致**（0阶、1阶、2阶算同一个数）；
* 二阶量对得上中心差分（新调用方`physics_engine.rod`要靠它）。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.autodiff import (
    AutodiffError,
    Jet1,
    Jet2,
    ad_cos,
    ad_norm,
    ad_sin,
    ad_sqrt,
)


def _expression(values):
    """一个既含开方、又含三角、还含除法的表达式——三条一元链式法则都被走到。"""

    x, y, z = values
    return ad_sqrt(x * x + y * y + 1.0) * ad_cos(z) / (2.0 + ad_sin(x)) + x * y * z


def test_the_value_channel_is_bytewise_identical_across_the_three_orders() -> None:
    """0/1/2阶必须算出**同一个**浮点数——否则"梯度是这个能量的梯度"就不成立。"""

    point = (0.37, -1.21, 0.83)
    zeroth = _expression(point)
    first = _expression(
        tuple(Jet1.variable(value, index, 3) for index, value in enumerate(point))
    )
    second = _expression(
        tuple(Jet2.variable(value, index, 3) for index, value in enumerate(point))
    )
    assert first.value == zeroth
    assert second.value == zeroth


def test_the_gradient_and_hessian_match_central_differences() -> None:
    """二阶链式法则的中心差分门。**它验的是导数，不验那个能量对不对**。"""

    point = (0.37, -1.21, 0.83)
    jets = _expression(
        tuple(Jet2.variable(value, index, 3) for index, value in enumerate(point))
    )
    step = 1.0e-5
    for index in range(3):
        plus = list(point)
        minus = list(point)
        plus[index] += step
        minus[index] -= step
        numeric = (_expression(tuple(plus)) - _expression(tuple(minus))) / (2.0 * step)
        assert abs(numeric - jets.gradient[index]) <= 1.0e-8 * max(1.0, abs(numeric))
    for row in range(3):
        for column in range(3):
            shifted = list(point)
            shifted[row] += step
            forward = _expression(
                tuple(Jet1.variable(v, i, 3) for i, v in enumerate(shifted))
            ).gradient[column]
            shifted[row] -= 2.0 * step
            backward = _expression(
                tuple(Jet1.variable(v, i, 3) for i, v in enumerate(shifted))
            ).gradient[column]
            numeric = (forward - backward) / (2.0 * step)
            assert abs(numeric - jets.hessian[row][column]) <= 1.0e-7 * max(1.0, abs(numeric))


def test_the_hessian_is_symmetric() -> None:
    point = (0.37, -1.21, 0.83)
    jets = _expression(
        tuple(Jet2.variable(value, index, 3) for index, value in enumerate(point))
    )
    for row in range(3):
        for column in range(row):
            assert jets.hessian[row][column] == jets.hessian[column][row]


def test_failures_are_autodiff_errors_and_still_value_errors() -> None:
    """类型变了要说清楚：`AutodiffError`仍是`ValueError`，求解器的回溯照样接得住。"""

    assert issubclass(AutodiffError, ValueError)
    with pytest.raises(AutodiffError, match="non-positive"):
        ad_norm((0.0, 0.0, 0.0))
    with pytest.raises(AutodiffError, match="division by zero"):
        Jet2.constant(0.0, 2).reciprocal()
    with pytest.raises(AutodiffError, match="division by zero"):
        Jet1.constant(0.0, 2).reciprocal()
    with pytest.raises(AutodiffError, match="inconsistent widths"):
        Jet2.constant(1.0, 2) * Jet2.constant(1.0, 3)
    with pytest.raises(AutodiffError, match="inconsistent widths"):
        Jet1.constant(1.0, 2) * Jet1.constant(1.0, 3)


def test_a_plain_float_still_goes_through_the_wrappers() -> None:
    """0阶入口不许被jet化——`energy()`那条路要的就是裸浮点。"""

    assert ad_sqrt(4.0) == 2.0
    assert ad_sin(0.5) == math.sin(0.5)
    assert ad_cos(0.5) == math.cos(0.5)
