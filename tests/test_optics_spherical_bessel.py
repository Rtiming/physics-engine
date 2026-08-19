"""`optics/spherical_bessel.py`的门——精度申报、朗斯基恒等式、必红（决策0091）。

三类判据：

1. **对外部参照**：50位`decimal`的上升级数（交错级数在高精度下逐项精确，
   与被验的Miller向下递推没有任何共用代码）。申报``|Δj_n| <= 1e-15``，
   **两个方向都断**——申报值不成立要红，申报值比实测松20倍以上也要红
   （写松的申报会让下游容差跟着虚胖，`bessel.py`的`J1`那条门立的实践）；
2. **不需要任何参照的自洽门**：朗斯基（cross-product）恒等式
   ``psi_n chi_{n+1} - psi_{n+1} chi_n = 1``。右边是精确的1，与`n`、`x`都无关；
3. **算法自身的可疑处被正面量出来**：Miller递推的起点抬高10阶再算一遍
   （`start_lift`），要求两次的差在申报精度内。
   **不设这条门的话`SPHERICAL_MILLER_START_MARGIN`就只是一个被相信的数。**

必红一条，而且它是**教科书算法**：Bohren & Huffman的`BHMIE`对``psi_n``
用向上递推，而``j_n``是那条递推的极小解——本文件把它构造出来，
实测``x = 1e-3``、``n = 3``上**答案比真值大六万倍**。
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

import pytest

from physics_engine.optics.errors import OpticsError
from physics_engine.optics.spherical_bessel import (
    SPHERICAL_J_ABSOLUTE_ACCURACY,
    SPHERICAL_TESTED_ARGUMENT_MAX,
    SPHERICAL_TESTED_ORDER_MAX,
    SPHERICAL_WRONSKIAN_RESIDUAL,
    riccati_bessel_chi_array,
    riccati_bessel_psi_array,
    riccati_bessel_xi_array,
    spherical_bessel_j,
    spherical_bessel_j_array,
    spherical_bessel_y,
    spherical_bessel_y_array,
    spherical_hankel_h1,
    wronskian_residual,
)

#: 高精度参照的位数。50位远超`float`的16位，于是参照本身的误差在这条判据里
#: 不参与——**判据自己的参照可能比被验对象更差**（0086第5.2节的教训）。
REFERENCE_DIGITS = 50

#: 覆盖的自变量。**故意含1e-3这种小宗量**（相消最凶的地方）、
#: `pi`的整数倍附近（``j_0 = sin x / x``过零、归一化要换锚点）、
#: 以及申报上界60。
ARGUMENTS: tuple[float, ...] = (
    1.0e-3, 1.0e-2, 0.1, 0.5, 1.0, 2.0, math.pi, 5.0, 10.0, 20.0,
    2.0 * math.pi * 5.0, 40.0, SPHERICAL_TESTED_ARGUMENT_MAX,
)

#: 申报值不许比实测松过头。实测最坏1.1102e-16、申报1e-15，比值0.111；
#: 取0.05即"申报不得超过实测的20倍"。
DECLARATION_TIGHTNESS = 0.05

#: 向上递推（`BHMIE`那条）在小宗量上必须坏成什么样才算这条必红成立。
#: 实测``x = 1e-3``、``n = 3``上相对误差**5.952e+04**，取1e3。
UPWARD_RECURRENCE_MINIMUM_DAMAGE = 1.0e3


getcontext().prec = REFERENCE_DIGITS


def _reference_j(order: int, x: float) -> float:
    """``j_n(x)``的50位参照：上升级数

    ``j_n(x) = x^n/(2n+1)!! * sum_k (-x^2/2)^k / (k! prod_{i=1..k}(2n+2i+1))``。

    它与被验实现**没有任何共用代码**：这里既没有递推也没有归一化，
    只有一条逐项精确的交错级数在50位上求和。
    """

    argument = Decimal(repr(x))
    half_square = argument * argument / 2
    double_factorial = Decimal(1)
    for value in range(1, 2 * order + 2, 2):
        double_factorial *= value
    total = Decimal(1)
    term = Decimal(1)
    for index in range(1, 500):
        term *= -half_square / (Decimal(index) * Decimal(2 * order + 2 * index + 1))
        total += term
        if abs(term) < Decimal(10) ** -(REFERENCE_DIGITS - 5) * (abs(total) + 1):
            break
    return float(argument**order / double_factorial * total)


def _textbook_upward_psi(order_max: int, x: float) -> list[float]:
    """`BHMIE`那条向上递推——**故意错的那一个**，只给必红用。

    ``psi_0 = sin x``、``psi_1 = sin x / x - cos x``、
    ``psi_n = (2n-1)/x psi_{n-1} - psi_{n-2}``。
    """

    values = [math.sin(x)]
    if order_max >= 1:
        values.append(math.sin(x) / x - math.cos(x))
    for order in range(1, order_max):
        values.append((2 * order + 1) / x * values[order] - values[order - 1])
    return values


# --- 判据一｜对50位参照的精度申报 -----------------------------------------


def test_the_downward_recurrence_matches_a_fifty_digit_reference():
    """``|j_n^算 - j_n^真| <= SPHERICAL_J_ABSOLUTE_ACCURACY``，两个方向都断。

    **判绝对不判相对**：``j_n``有零点，零点邻域的相对误差没有意义
    （与`bessel.py`的`J1`同一条理由）。
    """

    worst = 0.0
    worst_at = None
    for argument in ARGUMENTS:
        computed = spherical_bessel_j_array(SPHERICAL_TESTED_ORDER_MAX, argument)
        for order in range(SPHERICAL_TESTED_ORDER_MAX + 1):
            deviation = abs(computed[order] - _reference_j(order, argument))
            if deviation > worst:
                worst, worst_at = deviation, (order, argument)
    assert worst <= SPHERICAL_J_ABSOLUTE_ACCURACY, (
        f"最坏绝对偏差{worst!r}超过申报{SPHERICAL_J_ABSOLUTE_ACCURACY!r}，落在{worst_at!r}"
    )
    assert worst >= DECLARATION_TIGHTNESS * SPHERICAL_J_ABSOLUTE_ACCURACY, (
        f"实测{worst!r}比申报{SPHERICAL_J_ABSOLUTE_ACCURACY!r}松了20倍以上——"
        "**写松的申报会让下游容差跟着虚胖**，该收紧申报而不是留着"
    )


def test_must_be_red_the_textbook_upward_recurrence_loses_the_small_argument():
    """必须红：`BHMIE`那条向上递推。**这是教科书算法，不是一个假想的错误。**

    ``j_n``是那条递推的**极小解**，向上递推对极小解从来就是不稳的。
    实测``x = 1e-3``、``n = 3``上相对误差**5.952e+04**——
    答案比真值大六万倍，而``n = 3``恰恰是Mie在那个尺度参数上的截断阶。
    """

    argument = 1.0e-3
    order = 3
    reference = _reference_j(order, argument) * argument
    upward = _textbook_upward_psi(order, argument)[order]
    downward = riccati_bessel_psi_array(order, argument)[order]
    upward_error = abs(upward - reference) / abs(reference)
    downward_error = abs(downward - reference) / abs(reference)
    assert upward_error > UPWARD_RECURRENCE_MINIMUM_DAMAGE, (
        f"向上递推在x={argument!r}、n={order}上的相对误差只有{upward_error!r}——"
        "本条必红据以成立的那个事实没有出现，判据要重查"
    )
    assert downward_error <= 1.0e-14, f"向下递推自己的相对误差{downward_error!r}"


# --- 判据二｜不需要任何参照的自洽门 ---------------------------------------


def test_the_wronskian_cross_product_identity_holds_for_every_order():
    """``psi_n chi_{n+1} - psi_{n+1} chi_n = 1``——右边是**精确的1**。

    它同时套住``j``与``y``两串，而且不依赖任何外部数值表。
    **它还是`chi_n`符号的捕手**：把``chi_n = -x y_n``的负号丢掉，
    这条恒等式变成`-1`，而所有的模长、所有的强度都一个字不变。
    """

    worst = 0.0
    for argument in ARGUMENTS:
        for order in range(SPHERICAL_TESTED_ORDER_MAX):
            worst = max(worst, abs(wronskian_residual(order, argument)))
    assert worst <= SPHERICAL_WRONSKIAN_RESIDUAL, f"朗斯基残差{worst!r}"
    assert worst >= DECLARATION_TIGHTNESS * SPHERICAL_WRONSKIAN_RESIDUAL


def test_must_be_red_flipping_the_chi_sign_flips_the_wronskian():
    """必须红：``chi_n``的符号。翻了之后恒等式给`-1`而不是`+1`。

    本条不改`src/`，而是直接算"如果符号反了会是多少"——
    符号反了``xi_n``就是``h_n^(2)``（内行波），
    **散射截面照样是正的、图样照样合理，一个字都不报**。
    """

    argument = 5.0
    psi = riccati_bessel_psi_array(6, argument)
    chi = riccati_bessel_chi_array(6, argument)
    for order in range(5):
        flipped = psi[order] * (-chi[order + 1]) - psi[order + 1] * (-chi[order])
        assert abs(flipped - 1.0) > 1.0, f"n={order}：符号翻了居然还满足恒等式"
        assert abs(flipped + 1.0) <= SPHERICAL_WRONSKIAN_RESIDUAL


# --- 判据三｜递推起点够不够高，是被验的不是被相信的 -----------------------


def test_the_miller_start_is_high_enough_because_lifting_it_changes_nothing():
    """把向下递推的起点抬高10阶再算一遍，两次的差必须在申报精度内。

    **不设这条门的话`SPHERICAL_MILLER_START_MARGIN`就只是一个被相信的数。**
    同一条形态在`mie.py`的对数导数起点上**打红过一次真缺陷**
    （Bohren & Huffman那条``+15``，见`tests/test_optics_mie.py`）。
    """

    worst = 0.0
    for argument in ARGUMENTS:
        base = spherical_bessel_j_array(SPHERICAL_TESTED_ORDER_MAX, argument)
        lifted = spherical_bessel_j_array(
            SPHERICAL_TESTED_ORDER_MAX, argument, start_lift=10
        )
        worst = max(
            worst, max(abs(a - b) for a, b in zip(base, lifted, strict=True))
        )
    assert worst <= SPHERICAL_J_ABSOLUTE_ACCURACY, f"抬高起点后差了{worst!r}"


# --- 出口与失败关闭 -------------------------------------------------------


def test_the_hankel_and_riccati_exits_agree_with_the_two_arrays():
    """``xi_n = psi_n - i chi_n``与``h_n^(1) = j_n + i y_n``两个出口互相一致。

    ``xi``对``psi``/``chi``是**零容差**（同一串数的两种包装）。

    **但单阶出口对整串出口不是逐位相同的，这条被正面断言在这里**：
    `spherical_bessel_j(n, x)`按``order_max = n``起一条自己的向下递推，
    起点因此与`spherical_bessel_j_array(8, x)`那一条不同——
    实测``n = 2``、``x = 3``上差**最后一位**
    （`0x1.31ce072d32c2ap-2` 对 `0x1.31ce072d32c2bp-2`）。
    这不是缺陷，是Miller递推的性质；**它必须被写下来**，
    因为本仓有"消费方采纳后既有产物指纹逐字节不变"这条硬门（0001第三条），
    而混用两个出口的调用方会看到指纹变。
    ``y``那一串没有这个问题（向上递推与`order_max`无关，实测逐位相同）。
    """

    argument = 3.0
    top = 8
    psi = riccati_bessel_psi_array(top, argument)
    chi = riccati_bessel_chi_array(top, argument)
    xi = riccati_bessel_xi_array(top, argument)
    j = spherical_bessel_j_array(top, argument)
    y = spherical_bessel_y_array(top, argument)
    single_differs = False
    for order in range(top + 1):
        assert xi[order].real.hex() == psi[order].hex()
        assert xi[order].imag.hex() == (-chi[order]).hex()
        single = spherical_bessel_j(order, argument)
        assert abs(single - j[order]) <= SPHERICAL_J_ABSOLUTE_ACCURACY
        single_differs = single_differs or single.hex() != j[order].hex()
        assert spherical_bessel_y(order, argument).hex() == y[order].hex()
        hankel = spherical_hankel_h1(order, argument)
        #: ``h^(1)``的实部同样按``order_max = order``起一条自己的递推，
        #: 所以与整串出口一样只到申报精度；虚部走的是与`order_max`无关的
        #: 向上递推，**逐位相同**。这两者的分别正是上面那段docstring说的事。
        assert abs(hankel.real - j[order]) <= SPHERICAL_J_ABSOLUTE_ACCURACY
        assert hankel.imag.hex() == y[order].hex()
    assert single_differs, (
        "单阶出口与整串出口在这个构型上居然逐位相同——"
        "那说明起点规则变了，本条docstring里那段警告要重写"
    )


@pytest.mark.parametrize("bad", (0.0, -1.0, float("inf"), float("nan")))
def test_a_non_positive_argument_fails_closed(bad):
    with pytest.raises(OpticsError, match="有限正数"):
        spherical_bessel_j_array(3, bad)


def test_a_negative_order_fails_closed():
    with pytest.raises(OpticsError, match="非负整数"):
        spherical_bessel_j_array(-1, 1.0)


def test_the_normalisation_switches_anchor_where_the_first_seed_vanishes():
    """``x = pi``处``j_0 = sin x / x``恰好过零，归一化必须换到``j_1``。

    **这一格不测的话，那条"取模较大者"的规则就只是一句docstring。**
    实测``j_0(pi)``的量级是1e-17（`sin(pi)`的浮点残值），
    而``j_1(pi) = -cos(pi)/pi = 0.3183``。
    """

    values = spherical_bessel_j_array(4, math.pi)
    assert abs(values[0]) < 1.0e-15
    assert abs(values[1] - 1.0 / math.pi) <= 1.0e-15
    for order in range(5):
        assert abs(values[order] - _reference_j(order, math.pi)) <= (
            SPHERICAL_J_ABSOLUTE_ACCURACY
        )
