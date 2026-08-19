"""球贝塞尔与Riccati-Bessel函数——Mie严格级数解的地基（决策0091）。

`bessel.py`只有一阶柱贝塞尔`J1`（艾里斑要的那一个）。Mie要的是**球**贝塞尔
``j_n``、``y_n``与它们的Riccati形式``psi_n = x j_n``、``chi_n = -x y_n``、
``xi_n = x h_n^(1) = psi_n - i chi_n``，而且要**一整串阶**（`n = 0..n_max`）。
零运行时依赖（AGENTS.md本仓纪律）：只用`math`/`cmath`。

## 一、``psi_n``为什么**不能**照抄教科书的向上递推（本模块最要紧的一段）

Bohren & Huffman的`BHMIE`（以及照抄它的大多数实现）对``psi_n``用
**向上**递推``psi_n = (2n-1)/x psi_{n-1} - psi_{n-2}``，种子是
``psi_0 = sin x``、``psi_1 = sin x / x - cos x``。

**这条路在小宗量上是灾难，而且不报任何错**：

* ``psi_1 = sin(x)/x - cos(x)``本身就是两个都趋于1的数相减。``x = 1e-3``时
  真值是``x^2/3 = 3.33e-7``，两个被减数都是``~1``——**相减掉掉了约10位**；
* 更糟的是递推自己：``psi_2 = 3/x psi_1 - psi_0``在``x = 1e-3``时是
  ``1.0e-3 - 1.0e-3``，真值``x^3/15 = 6.7e-11``——**又掉6位**。
  逐阶继续，误差按**寄生解**``chi_n ~ (2n-1)!!/x^n``放大；
* ``j_n``是这条递推的**极小解**（minimal solution），向上递推对极小解
  从来就是不稳的——这是数值分析的教科书结论，不是本仓的发现。
  本模块只是**不假装它在Mie的取阶范围里够用**。

实测（`tests/test_optics_spherical_bessel.py`的必红门，对50位`decimal`参照，
逐条是**相对**误差）：

| ``x`` | ``n`` | 向上递推 | 本模块向下递推 |
|---|---|---|---|
| 1e-3 | 1 | 1.134e-10 | **0**（逐位相同） |
| 1e-3 | 2 | 1.700e-03 | **0** |
| 1e-3 | 3 | **5.952e+04** | **0** |
| 1e-2 | 3 | 7.547e-04 | 1.357e-16 |
| 1e-1 | 3 | 4.743e-08 | 2.225e-16 |
| 1 | 3 | 7.569e-14 | 1.926e-16 |

``x = 1e-3``、``n = 3``那一格**答案比真值大六万倍**——而``n = 3``
恰恰就是Mie在那个尺度参数上的截断阶（`mie.mie_max_order(1e-3) = 3`）。
它对``Q_sca``的实际影响另有一条门去量（那几阶的贡献本来就小），
**但"这个函数在这里返回的是噪声"这件事必须写下来而不是靠"反正影响不大"糊过去**。

## 二、本模块走的那条：Miller向下递推 + 三角种子归一

``j_n``是极小解 ⟹ **向下**递推对它是稳的（误差按寄生解衰减而不是增长）。

1. 从``n_start = n_max + margin``起，取任意种子``j_{start+1} = 0``、``j_{start} = 1``；
2. 向下递推``j_{n-1} = (2n+1)/x j_n - j_{n+1}``到``n = 0``，途中值超过
   `_RESCALE_CEILING`就整体缩放一次（小`x`上不缩放会溢出）；
3. 用**闭式**``j_0 = sin x / x``或``j_1 = sin x / x^2 - cos x / x``归一，
   **取两者中真值模较大的那一个**。这一步是本模块唯一可能相消的地方，
   而它被这条"取较大者"的规则挡住了：
   * 小``x``：``|j_0| -> 1``而``|j_1| -> x/3``，选``j_0``（``sin x / x``无相消）；
   * ``x = m pi``附近：``j_0 = 0``而``j_1 = -cos x / x``模最大，选``j_1``；
   * 两者**不可能同时为零**（朗斯基行列式非零），所以这条规则总有出口。

``y_n``是**极大解**（dominant），向上递推对它是稳的，本模块照做：
``y_0 = -cos x / x``、``y_1 = -cos x / x^2 - sin x / x``，
``y_{n+1} = (2n+1)/x y_n - y_{n-1}``。

## 三、精度怎么申报，以及一条不需要参照的自洽门

**申报**：``|j_n^算 - j_n^真| <= SPHERICAL_J_ABSOLUTE_ACCURACY``，
``n``取遍``[0, SPHERICAL_TESTED_ORDER_MAX]``、``x``取遍
``(0, SPHERICAL_TESTED_ARGUMENT_MAX]``。对照是**50位`decimal`的上升级数**
（交错级数在高精度下逐项精确，与本模块的向下递推没有任何共用代码）。
**申报的是绝对误差不是相对误差**，理由与`bessel.py`的`J1`同源：
``j_n``有零点，零点邻域的相对误差没有意义。

**外加一条不需要任何参照的自洽门**——朗斯基（cross-product）恒等式

    psi_n(x) chi_{n+1}(x) - psi_{n+1}(x) chi_n(x) = 1

右边是**精确的1**，与``n``和``x``都无关。它同时套住``j``与``y``两串，
而且不依赖任何外部数值表：**一条判据自己的参照可能比被验对象更差
（0086第5.2节），恒等式没有这个问题**。

## 四、域外与失败关闭

``x <= 0``、非有限、阶为负 → 当场炸。
``x``大于`SPHERICAL_TESTED_ARGUMENT_MAX`时**不拒答**（向下递推在大``x``上
只会更稳），但那是"未测"不是"成立"——精度申报只覆盖测过的范围，
调用方要更大的``x``请自己先测。**这一条与`bessel.py`的``|x| > 60``未测同一条纪律。**
"""

from __future__ import annotations

import math

from physics_engine.optics.errors import OpticsError

#: Miller向下递推的起点比要的最高阶高多少。**它不是拍的**：
#: 起点每高一阶，种子误差在到达``n_max``时就多衰减一个``(2n+1)/x``因子，
#: 于是"起点够不够高"是可以**验**的——`_spherical_j_array`本身不验，
#: 但`tests/test_optics_spherical_bessel.py`有一条门把起点抬高10阶再算一遍，
#: 要求两次结果的差在申报精度之内。改这个数要重跑那条门。
SPHERICAL_MILLER_START_MARGIN: int = 25

#: 起点还要至少是``sqrt(_MILLER_ACCURACY * n_max)``那么多阶
#: （大阶上固定的margin不够——衰减速率随``n/x``变慢）。取40是
#: Numerical Recipes对柱贝塞尔用的同一个数，本模块把它当**起点**而不是结论：
#: 上一条注释里那条"抬高10阶再算一遍"的门守着它。
_MILLER_ACCURACY: float = 40.0

#: 向下递推途中值超过它就整体缩放一次。小``x``上不缩放会溢出到`inf`：
#: ``x = 1e-3``、起点第28阶时，未归一的值比第0阶小``(2n+1)!!/x^n ~ 1e100``倍，
#: 于是从起点往下推的过程中值会涨到那个量级。
_RESCALE_CEILING: float = 1.0e250

#: 申报精度：``[0, SPHERICAL_TESTED_ORDER_MAX] x (0, SPHERICAL_TESTED_ARGUMENT_MAX]``
#: 上``j_n``的最大绝对误差上界。判据引用它，不许在测试里私改。
SPHERICAL_J_ABSOLUTE_ACCURACY: float = 1.0e-15

#: 申报精度成立的阶范围上界。超出即"未测"，不是"成立"。
SPHERICAL_TESTED_ORDER_MAX: int = 40

#: 申报精度成立的自变量范围上界。超出即"未测"，不是"成立"。
SPHERICAL_TESTED_ARGUMENT_MAX: float = 60.0

#: 朗斯基恒等式``psi_n chi_{n+1} - psi_{n+1} chi_n = 1``的申报残差上界。
#: 它是**相对**的（右边恒等于1，所以绝对与相对同一个数）。
#: 实测最坏**1.7764e-15**（`n`取遍0—39、`x`取13档到60），取1e-14即余量5.6倍。
SPHERICAL_WRONSKIAN_RESIDUAL: float = 1.0e-14


def _checked_argument(x: float) -> float:
    value = float(x)
    if not math.isfinite(value) or value <= 0.0:
        raise OpticsError(
            f"球贝塞尔的自变量必须是有限正数：{x!r}。"
            "``x = 0``不是可去奇点问题而是归一化没有出口"
            "（``j_0 = sin x / x``在那里要取极限，``y_n``则真的发散）"
        )
    return value


def _checked_order(order_max: int) -> int:
    value = int(order_max)
    if value < 0:
        raise OpticsError(f"最高阶必须是非负整数：{order_max!r}")
    return value


def _miller_start_order(order_max: int, x: float) -> int:
    """向下递推的起点阶。**要同时盖住"比n_max高margin"与"比x高"两件事**。"""

    floor_order = max(order_max, int(math.ceil(x)))
    return floor_order + max(
        SPHERICAL_MILLER_START_MARGIN,
        int(math.ceil(math.sqrt(_MILLER_ACCURACY * max(floor_order, 1)))),
    )


def spherical_bessel_j_array(order_max: int, x: float, *, start_lift: int = 0) -> tuple[float, ...]:
    """``j_0(x) .. j_{order_max}(x)``，Miller向下递推 + 三角种子归一。

    `start_lift`把递推起点再抬高若干阶——**它只给判据用**：
    `tests/test_optics_spherical_bessel.py`用它把起点抬高再算一遍，
    要求两次的差落在申报精度内。**那是"起点够不够高"这件事的门**，
    不设这个参数的话`SPHERICAL_MILLER_START_MARGIN`就只是一个被相信的数。
    """

    top = _checked_order(order_max)
    value = _checked_argument(x)
    lift = _checked_order(start_lift)
    start = _miller_start_order(top, value) + lift

    stored = [0.0] * (top + 1)
    upper = 0.0  # j_{n+1}（未归一）
    current = 1.0  # j_n（未归一，种子任意）
    if start <= top:
        stored[start] = current
    for order in range(start, 0, -1):
        lower = (2 * order + 1) / value * current - upper
        upper = current
        current = lower
        if order - 1 <= top:
            stored[order - 1] = current
        if abs(current) > _RESCALE_CEILING:
            #: 整体缩放一次。归一化在最后一步，所以缩放因子取多少都不影响结果——
            #: 它只是把中间值拉回浮点的可表示范围。
            upper /= _RESCALE_CEILING
            current /= _RESCALE_CEILING
            for index in range(top + 1):
                stored[index] /= _RESCALE_CEILING

    sine = math.sin(value)
    cosine = math.cos(value)
    true_j0 = sine / value
    true_j1 = sine / (value * value) - cosine / value
    #: **取真值模较大的那一个归一**——这是本模块唯一可能相消的一步，
    #: 而这条规则挡住了它（模块docstring第二节逐条写了三种情形）。
    if abs(true_j0) >= abs(true_j1):
        reference, index = true_j0, 0
    else:
        reference, index = true_j1, 1
    if index > top:
        #: 只要``j_0``或``j_1``（``order_max = 0``且归一化要用``j_1``）时，
        #: 未归一数组不够长；补算那一阶。
        stored = stored + [0.0]
        stored[1] = upper if start >= 1 else 0.0
    anchor = stored[index]
    if anchor == 0.0:
        raise OpticsError(
            f"向下递推在x={x!r}上给出的归一化锚点是0——这不该发生，"
            "请把这个输入连同`SPHERICAL_MILLER_START_MARGIN`一起报上来"
        )
    scale = reference / anchor
    return tuple(stored[order] * scale for order in range(top + 1))


def spherical_bessel_y_array(order_max: int, x: float) -> tuple[float, ...]:
    """``y_0(x) .. y_{order_max}(x)``，向上递推（``y``是**极大解**，向上是稳的）。"""

    top = _checked_order(order_max)
    value = _checked_argument(x)
    sine = math.sin(value)
    cosine = math.cos(value)
    values = [-cosine / value]
    if top >= 1:
        values.append(-cosine / (value * value) - sine / value)
    for order in range(1, top):
        values.append((2 * order + 1) / value * values[order] - values[order - 1])
    return tuple(values)


def riccati_bessel_psi_array(order_max: int, x: float) -> tuple[float, ...]:
    """``psi_n(x) = x j_n(x)``。Mie系数的分子分母都吃它。"""

    value = _checked_argument(x)
    return tuple(value * item for item in spherical_bessel_j_array(order_max, value))


def riccati_bessel_chi_array(order_max: int, x: float) -> tuple[float, ...]:
    """``chi_n(x) = -x y_n(x)``。**符号写在这里而不是在调用点**——

    Bohren & Huffman与Abramowitz & Stegun在这个符号上不一致，
    而符号反了``xi_n``就变成``h_n^(2)``（内行波而不是外行波），
    散射截面照样是正的、图样照样"合理"，**一个字都不报**。
    """

    value = _checked_argument(x)
    return tuple(-value * item for item in spherical_bessel_y_array(order_max, value))


def riccati_bessel_xi_array(order_max: int, x: float) -> tuple[complex, ...]:
    """``xi_n(x) = x h_n^(1)(x) = psi_n(x) - i chi_n(x)``（**外行球面波**）。

    ``h_n^(1) = j_n + i y_n``，配的是``exp(-i omega t)``的时间约定
    （`MIE_TIME_CONVENTION`，见`mie.py`）。这一对必须同时翻，
    翻错了吸收会变成增益——而`mie.py`有一条门正面守着"吸收效率非负"。
    """

    psi = riccati_bessel_psi_array(order_max, x)
    chi = riccati_bessel_chi_array(order_max, x)
    return tuple(complex(p, -c) for p, c in zip(psi, chi, strict=True))


def riccati_psi_logarithmic_derivative(
    order_max: int, z: complex, *, start_lift: int = 0
) -> tuple[complex, ...]:
    """``D_n(z) = psi_n'(z) / psi_n(z)``，`n = 0..order_max`，**复宗量**。

    Mie的内场只通过这个对数导数进入系数，**所以复球贝塞尔函数本身
    一个都不用算**——这是Bohren & Huffman那条算法里唯一值得照抄的部分。

    走向下递推``D_{n-1} = n/z - 1/(D_n + n/z)``，起点``D_{n_start} = 0``。
    向下对`D`是稳的（与``j_n``同源：``D``是``psi``的对数导数，而``psi``是极小解），
    **但"起点取多高"不是随便的**：起点必须比转折点``n ~ |z|``高出足够多，
    否则误差还没衰减完就到了要用的阶。

    **起点用的是与`spherical_bessel_j_array`同一条规则**
    （`_miller_start_order`，即``max(阶, |z|)``再加
    ``max(SPHERICAL_MILLER_START_MARGIN, sqrt(40 * 那个数))``），
    **不是**Bohren & Huffman的`BHMIE`那条``+ 15``。

    **那条``+15``是被本仓的判据打红的一条真缺陷**（决策0091第五节）：
    `mie.py`的`logarithmic_derivative_lift`把起点抬高再算一遍，实测
    ``x = 500``、``m = 1.33``（无吸收）上``+15``给``Q_ext = 2.031189119014``、
    收敛值是``2.030373894631``——**相对差4.0e-4**，
    而那是一个自称"严格级数解"的实现，**一个字都不报**。
    ``x = 50``、``m = 1.5``上同样的形态是6.7e-8。
    """

    top = _checked_order(order_max)
    value = complex(z)
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise OpticsError(f"对数导数的宗量必须有限：{z!r}")
    if value == 0:
        raise OpticsError("对数导数的宗量不能是0")
    start = _miller_start_order(top, abs(value)) + _checked_order(start_lift)
    values = [complex(0.0, 0.0)] * (start + 1)
    for order in range(start, 0, -1):
        ratio = order / value
        values[order - 1] = ratio - 1.0 / (values[order] + ratio)
    return tuple(values[: top + 1])


def wronskian_residual(order: int, x: float) -> float:
    """``psi_n chi_{n+1} - psi_{n+1} chi_n - 1``——**一条不需要任何参照的自洽门**。

    右边恒等于1，与``n``和``x``都无关。判据引用
    `SPHERICAL_WRONSKIAN_RESIDUAL`。本函数放在`src/`而不是测试里，
    是因为它是这两串函数**自带的**可自检性质，下游（例如换平台之后）
    要复核精度时用得上——与`bessel.py`把精度申报写进模块是同一条实践。
    """

    top = _checked_order(order) + 1
    psi = riccati_bessel_psi_array(top, x)
    chi = riccati_bessel_chi_array(top, x)
    index = int(order)
    return psi[index] * chi[index + 1] - psi[index + 1] * chi[index] - 1.0


def spherical_hankel_h1(order: int, x: float) -> complex:
    """``h_n^(1)(x) = j_n(x) + i y_n(x)``。单个阶的取用口。"""

    top = _checked_order(order)
    value = _checked_argument(x)
    j = spherical_bessel_j_array(top, value)
    y = spherical_bessel_y_array(top, value)
    return complex(j[top], y[top])


def spherical_bessel_j(order: int, x: float) -> float:
    """单个``j_n(x)``。要一整串请用`spherical_bessel_j_array`。

    **两条注意**：

    1. 本函数每次调用都会重跑整条向下递推，在循环里逐阶调它是``O(n^2)``；
    2. **它与整串出口不是逐位相同的**。向下递推的起点按`order_max`定，
       所以``spherical_bessel_j(2, 3.0)``与``spherical_bessel_j_array(8, 3.0)[2]``
       走的是两条起点不同的递推——实测差**最后一位**。两者都在申报精度内，
       但**混用两个出口的调用方会看到指纹变**（0001第三条那道硬门）。
       ``y``那一串没有这个问题（向上递推与`order_max`无关）。
       `tests/test_optics_spherical_bessel.py`把这件事正面断言着，
       免得下一个人以为它是逐位相同的。
    """

    return spherical_bessel_j_array(order, x)[int(order)]


def spherical_bessel_y(order: int, x: float) -> float:
    """单个``y_n(x)``。"""

    return spherical_bessel_y_array(order, x)[int(order)]


__all__ = [
    "SPHERICAL_J_ABSOLUTE_ACCURACY",
    "SPHERICAL_MILLER_START_MARGIN",
    "SPHERICAL_TESTED_ARGUMENT_MAX",
    "SPHERICAL_TESTED_ORDER_MAX",
    "SPHERICAL_WRONSKIAN_RESIDUAL",
    "riccati_bessel_chi_array",
    "riccati_bessel_psi_array",
    "riccati_bessel_xi_array",
    "riccati_psi_logarithmic_derivative",
    "spherical_bessel_j",
    "spherical_bessel_j_array",
    "spherical_bessel_y",
    "spherical_bessel_y_array",
    "spherical_hankel_h1",
    "wronskian_residual",
]
