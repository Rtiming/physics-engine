"""均匀球的Mie散射——**严格级数解，不是近似**（决策0091）。

`diffraction.py`是标量衍射的闭式（圆孔艾里斑），`propagation.py`是标量场传播。
两者都不含"散射"这件事：能力位S4.7的`why`原文写着plans/05对场景④的判词是
"**散射零**"。本模块补的就是那个零。

Mie解是Maxwell方程组对**均匀各向同性球**在平面波照明下的**严格解**
（不是小球近似、不是几何光学近似、不是任何一种微扰）。它的形态是一条
按球矢量波函数展开的无穷级数；数值上唯一的近似是**在第``n_max``阶截断**，
而那一条必须显式声明并**失败关闭**——见第三节。

零运行时依赖（AGENTS.md本仓纪律）：只用`math`/`cmath`与本域的
`spherical_bessel`/`errors`。

## 一、两个必须写下来的约定（反了不报错）

### 1.1 时间约定与折射率的虚部符号

本模块取``exp(-i omega t)``（`MIE_TIME_CONVENTION`），于是外行球面波是
``h_n^(1)``，而**吸收介质的折射率虚部为正**：``m = n + i k``、``k >= 0``
（`MIE_REFRACTIVE_INDEX_CONVENTION`）。

翻错会怎样：``k < 0``时级数照样收敛、散射效率照样是正数、图样照样光滑——
**但吸收效率变成负的**，即那个球在发光。所以本模块对``Im(m) < 0``
**当场失败关闭**并在报错里点名时间约定，而不是让一个"增益球"静静地算下去。
（另有一条门正面守着``Q_abs >= 0``，见`MIE_ABSORPTION_FLOOR`。）

### 1.2 尺度参数吃的是**半径**不是直径

``x = 2 pi a / lambda``，`a`是**半径**（`size_parameter`由调用方给，
`size_parameter_from_radius`替他算）。传直径进来所有效率都会错，
而且错得很像另一个球的答案——与`diffraction.airy_argument`吃半径同一条实践。

## 二、算法：为什么内场只算对数导数

系数（Bohren & Huffman式4.53）

    a_n = [(D_n(mx)/m + n/x) psi_n(x) - psi_{n-1}(x)]
        / [(D_n(mx)/m + n/x) xi_n(x)  - xi_{n-1}(x)]
    b_n = [(m D_n(mx) + n/x) psi_n(x) - psi_{n-1}(x)]
        / [(m D_n(mx) + n/x) xi_n(x)  - xi_{n-1}(x)]

**球内那一侧只通过对数导数``D_n(mx) = psi_n'(mx)/psi_n(mx)``进入**，
所以复宗量的球贝塞尔函数一个都不用算——`spherical_bessel`那边用
向下递推``D_{n-1} = n/z - 1/(D_n + n/z)``直接给`D`。
这是`BHMIE`那条算法里唯一值得照抄的部分。

**不值得照抄的那部分见`spherical_bessel`的模块docstring第一节**：
它对``psi_n(x)``用向上递推，而``j_n``是那条递推的极小解——
小宗量上实测掉到不能看。本模块的``psi``走Miller向下递推。

效率（同书式4.61—4.62）

    Q_ext = (2/x^2) sum (2n+1) Re(a_n + b_n)
    Q_sca = (2/x^2) sum (2n+1) (|a_n|^2 + |b_n|^2)
    Q_abs = Q_ext - Q_sca

**``Q_ext``与``Q_sca``是两条互相独立的式子**（一条取实部、一条取模平方），
所以"无吸收时两者相等"是一条**真的**判据而不是恒等变形——
它等价于每一阶的``|a_n|^2 = Re(a_n)``（系数落在幺正圆上），
`tests/test_optics_mie.py`把这两个形态都断了。
``Q_abs``按定义是两者之差，**所以"消光=散射+吸收"本身不是判据**，
本模块不假装它是。

## 三、截断：``n_max``怎么定，以及**不收敛就失败关闭**

``n_max = ceil(x + 4 x^(1/3) + 2)``（`mie_max_order`，Wiscombe判据，
`BHMIE`用的同一条）。这条判据是**经验的**，所以本模块不止步于此：

* 算完之后量**末项对两条求和各自的相对贡献**（`series_tail_ratio`）。
  它超过`MIE_SERIES_TAIL_TOLERANCE`即**当场炸**——
  "这个构型我算不收敛"与"我给你一个截短了的答案"是两件事
  （与`propagation.py`拒绝在混叠区给数同一条纪律）；
* ``x``与``|m| x``都有**申报过的测试范围**（`MIE_TESTED_SIZE_PARAMETER_MAX`
  等），越界**拒答**而不是外推。域外是"未测"不是"成立"。

**为什么不给个"自动加阶直到收敛"的循环**：那会把一个可以被判据看见的
失败变成一个安静的重试。今天没有任何构型需要它；真需要时调用方可以
显式传`order_count`（本模块也会照样量末项）。

## 参考

Bohren & Huffman,《Absorption and Scattering of Light by Small Particles》
第4.4节（球矢量波函数展开与系数`a_n`/`b_n`，式4.53）、
第4.4.2节（效率因子，式4.61—4.62）、第5.1节（附录`BHMIE`）；
Wiscombe 1980, Appl. Opt. 19, 1505（级数截断判据）；
Rayleigh散射的小球极限见`rayleigh_scattering_efficiency`的docstring。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.optics.errors import OpticsError
from physics_engine.optics.spherical_bessel import (
    riccati_bessel_xi_array,
    riccati_psi_logarithmic_derivative,
)

#: 时间约定。**写成常量而不是一句注释**：它决定折射率虚部的符号，
#: 而符号反了会得到一个"增益球"（吸收效率为负）且级数照样收敛。
MIE_TIME_CONVENTION: str = "exp(-i*omega*t)"

#: 折射率约定：``m = n + i k``、``k >= 0``表示吸收（与上一条配套）。
MIE_REFRACTIVE_INDEX_CONVENTION: str = "m = n + i*k, k >= 0 absorbing"

#: 尺度参数的申报测试范围。**越界拒答而不是外推**——大`x`上级数项数按`x`长，
#: 而"项数够不够"这件事只在测过的范围里被验过。
MIE_TESTED_SIZE_PARAMETER_MIN: float = 1.0e-6
MIE_TESTED_SIZE_PARAMETER_MAX: float = 1.0e3

#: 折射率模的申报测试范围上界。``|m| x``决定对数导数向下递推的起点，
#: 也决定球内的相位圈数。
MIE_TESTED_REFRACTIVE_INDEX_MODULUS_MAX: float = 1.0e1

#: 级数末项对求和的相对贡献上界。**超过它即当场炸**（第三节）。
#:
#: **这个数是扫出来的不是拍的**：`x`取16档（1e-6到1e3）× `m`取8种
#: （含无吸收、强吸收、`Re m < 1`、`|m| = 9`）共128个构型上，
#: 末项占比实测最坏**1.6963e-10**，取1e-8即余量59倍。
#:
#: 同一次扫描还量了**真正的截断误差**（截到``n_max``对截到``n_max + 10``的
#: 相对差）：实测最坏**1.2287e-10**，且**逐个构型都不超过同一构型的末项占比**。
#: 即"末项占比"在这张扫描表上确实是截断误差的上界——
#: **但那是实测的经验关系不是定理**，如实记在这里
#: （`tests/test_optics_mie.py`有一条门把这个关系重新量一遍）。
MIE_SERIES_TAIL_TOLERANCE: float = 1.0e-8

#: 吸收效率的下限（相对`Q_ext`）。``Q_abs = Q_ext - Q_sca``在无吸收介质上
#: 应当是浮点地板量级的小数，**负得超过这条即失败关闭**——
#: 那是时间约定翻了的形态（球在发光），不是数值噪声。
MIE_ABSORPTION_FLOOR: float = -1.0e-10

#: 几何光学极限下的消光效率（Extinction Paradox）。**它是2不是1**：
#: 球挡住的几何截面之外，还有等量的能量被衍射到前向小角内。
#: `tests/test_optics_mie.py`有一条门把``Q_ext``沿`x`推上去看它趋近这个数。
MIE_GEOMETRIC_OPTICS_EXTINCTION: float = 2.0


def size_parameter_from_radius(*, radius_m: float, wavelength_m: float) -> float:
    """``x = 2 pi a / lambda``。吃的是**半径**（第1.2节）。"""

    radius = float(radius_m)
    wavelength = float(wavelength_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise OpticsError(f"球半径必须是有限正数（米）：{radius_m!r}")
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise OpticsError(f"波长必须是有限正数（米）：{wavelength_m!r}")
    return 2.0 * math.pi * radius / wavelength


def mie_max_order(size_parameter: float) -> int:
    """``n_max = ceil(x + 4 x^(1/3) + 2)``（Wiscombe 1980）。

    **它是经验判据不是定理**，所以本模块在算完之后还要量末项
    （`MieEfficiencies.series_tail_ratio`）——见模块docstring第三节。
    """

    value = _checked_size_parameter(size_parameter)
    return int(math.ceil(value + 4.0 * value ** (1.0 / 3.0) + 2.0))


def _checked_size_parameter(size_parameter: float) -> float:
    value = float(size_parameter)
    if not math.isfinite(value) or value <= 0.0:
        raise OpticsError(f"尺度参数必须是有限正数：{size_parameter!r}")
    if not (MIE_TESTED_SIZE_PARAMETER_MIN <= value <= MIE_TESTED_SIZE_PARAMETER_MAX):
        raise OpticsError(
            f"尺度参数x={value!r}在申报的测试范围"
            f"[{MIE_TESTED_SIZE_PARAMETER_MIN!r}, {MIE_TESTED_SIZE_PARAMETER_MAX!r}]之外——"
            "**域外是「未测」不是「成立」**，本模块拒答而不是外推。"
            "要用更大的球请先按spec/13的诚实条款把精度测出来再放宽这两个常量"
        )
    return value


def _checked_refractive_index(refractive_index: complex) -> complex:
    value = complex(refractive_index)
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise OpticsError(f"折射率必须有限：{refractive_index!r}")
    if value.real <= 0.0:
        raise OpticsError(f"折射率实部必须为正：{refractive_index!r}")
    if value.imag < 0.0:
        raise OpticsError(
            f"折射率虚部为负（{refractive_index!r}）：本模块的时间约定是"
            f"{MIE_TIME_CONVENTION}，于是吸收对应``m = n + i k``且``k >= 0``"
            f"（{MIE_REFRACTIVE_INDEX_CONVENTION}）。"
            "虚部取负得到的是一个**发光的球**——级数照样收敛、散射效率照样为正、"
            "只有吸收效率是负的。**失败关闭而不是替调用方翻符号**："
            "翻符号等于替他决定他用的是哪一套约定"
        )
    if abs(value) > MIE_TESTED_REFRACTIVE_INDEX_MODULUS_MAX:
        raise OpticsError(
            f"折射率模{abs(value)!r}超过申报的测试上界"
            f"{MIE_TESTED_REFRACTIVE_INDEX_MODULUS_MAX!r}——域外是「未测」不是「成立」"
        )
    return value


@dataclass(frozen=True)
class MieCoefficients:
    """``a_n``与``b_n``，`n = 1..order_count`（下标0对应`n = 1`）。

    它是公开面，因为**幺正性判据要逐阶看**：无吸收介质上每一阶都满足
    ``|a_n|^2 = Re(a_n)``（系数落在以1/2为心、1/2为半径的圆上），
    而那条判据比"总的消光等于总的散射"更硬——后者可以靠两阶的误差
    互相抵消而蒙混过去。
    """

    a: tuple[complex, ...]
    b: tuple[complex, ...]
    size_parameter: float
    refractive_index: complex
    order_count: int


@dataclass(frozen=True)
class MieEfficiencies:
    """无量纲效率因子（截面除以几何截面``pi a^2``）。

    `series_tail_ratio`跟着数走，因为"这个答案收敛了没有"是它的一部分——
    与`PropagatedField.method`跟着数走同一条实践（0086第四节）。
    """

    extinction: float
    scattering: float
    absorption: float
    size_parameter: float
    refractive_index: complex
    order_count: int
    series_tail_ratio: float


def mie_coefficients(
    *,
    size_parameter: float,
    refractive_index: complex,
    order_count: int | None = None,
    logarithmic_derivative_lift: int = 0,
) -> MieCoefficients:
    """严格级数的系数``a_n``、``b_n``（Bohren & Huffman式4.53）。

    `logarithmic_derivative_lift`把对数导数向下递推的起点再抬高若干阶——
    **它只给判据用**：`tests/test_optics_mie.py`用它把起点抬高再算一遍，
    要求两次逐阶一致。不设这个参数的话"起点+15够不够"就只是一个被相信的数。
    """

    x = _checked_size_parameter(size_parameter)
    m = _checked_refractive_index(refractive_index)
    top = mie_max_order(x) if order_count is None else int(order_count)
    if top < 1:
        raise OpticsError(f"截断阶必须至少是1：{order_count!r}")

    #: ``psi``与``xi``：``xi_n = psi_n - i chi_n``，两者由同一条Riccati出口取，
    #: 于是"``psi``与``chi``用的是同一个``x``"这件事不可能在调用点错开。
    xi = riccati_bessel_xi_array(top, x)
    psi = tuple(value.real for value in xi)
    derivative = riccati_psi_logarithmic_derivative(
        top, m * x, start_lift=logarithmic_derivative_lift
    )

    a_values: list[complex] = []
    b_values: list[complex] = []
    for order in range(1, top + 1):
        ratio = order / x
        common_a = derivative[order] / m + ratio
        common_b = m * derivative[order] + ratio
        a_values.append(
            (common_a * psi[order] - psi[order - 1])
            / (common_a * xi[order] - xi[order - 1])
        )
        b_values.append(
            (common_b * psi[order] - psi[order - 1])
            / (common_b * xi[order] - xi[order - 1])
        )
    return MieCoefficients(
        a=tuple(a_values),
        b=tuple(b_values),
        size_parameter=x,
        refractive_index=m,
        order_count=top,
    )


def mie_efficiencies(
    *,
    size_parameter: float,
    refractive_index: complex,
    order_count: int | None = None,
    logarithmic_derivative_lift: int = 0,
) -> MieEfficiencies:
    """消光／散射／吸收效率。级数不收敛即**失败关闭**（模块docstring第三节）。"""

    coefficients = mie_coefficients(
        size_parameter=size_parameter,
        refractive_index=refractive_index,
        order_count=order_count,
        logarithmic_derivative_lift=logarithmic_derivative_lift,
    )
    x = coefficients.size_parameter
    extinction_sum = 0.0
    scattering_sum = 0.0
    last_extinction = 0.0
    last_scattering = 0.0
    for index, (a_value, b_value) in enumerate(
        zip(coefficients.a, coefficients.b, strict=True)
    ):
        weight = 2 * (index + 1) + 1
        last_extinction = weight * (a_value.real + b_value.real)
        last_scattering = weight * (
            a_value.real * a_value.real
            + a_value.imag * a_value.imag
            + b_value.real * b_value.real
            + b_value.imag * b_value.imag
        )
        extinction_sum += last_extinction
        scattering_sum += last_scattering

    factor = 2.0 / (x * x)
    extinction = factor * extinction_sum
    scattering = factor * scattering_sum
    tail = max(
        abs(last_extinction) / abs(extinction_sum) if extinction_sum != 0.0 else 0.0,
        last_scattering / scattering_sum if scattering_sum != 0.0 else 0.0,
    )
    if not math.isfinite(tail) or tail > MIE_SERIES_TAIL_TOLERANCE:
        raise OpticsError(
            f"Mie级数在x={x!r}、m={coefficients.refractive_index!r}上截到第"
            f"{coefficients.order_count}阶时末项仍占{tail!r}，"
            f"超过申报的收敛判据{MIE_SERIES_TAIL_TOLERANCE!r}——"
            "**不收敛就失败关闭**：一个在不收敛区静默给数的Mie是冒充"
        )
    absorption = extinction - scattering
    if absorption < MIE_ABSORPTION_FLOOR * abs(extinction):
        raise OpticsError(
            f"吸收效率{absorption!r}显著为负（消光{extinction!r}、散射{scattering!r}）——"
            f"这是时间约定翻了的形态。本模块取{MIE_TIME_CONVENTION}，"
            f"{MIE_REFRACTIVE_INDEX_CONVENTION}"
        )
    return MieEfficiencies(
        extinction=extinction,
        scattering=scattering,
        absorption=absorption,
        size_parameter=x,
        refractive_index=coefficients.refractive_index,
        order_count=coefficients.order_count,
        series_tail_ratio=tail,
    )


def clausius_mossotti_factor(refractive_index: complex) -> complex:
    """``(m^2 - 1)/(m^2 + 2)``——小球极限里唯一含材料的那一团。

    名字取自静电学里同一团（Clausius-Mossotti／Lorentz-Lorenz）：
    小球的Mie极限**就是**一个静电偶极子在振荡场里的辐射，
    这团因子在两边逐字相同。写成有名字的函数是因为散射与吸收
    两条小球闭式都吃它，而它们吃的是**同一个**——各写一遍会各自漂。
    """

    m = _checked_refractive_index(refractive_index)
    squared = m * m
    return (squared - 1.0) / (squared + 2.0)


def rayleigh_scattering_efficiency(
    *, size_parameter: float, refractive_index: complex
) -> float:
    """瑞利（小球）极限的散射效率``Q_sca = (8/3) x^4 |(m^2-1)/(m^2+2)|^2``。

    **它是一条独立的自洽门，不是Mie级数的重排**：这条闭式来自
    "把球当成一个静电极化率为``4 pi a^3 (m^2-1)/(m^2+2)``的点偶极子"
    这条完全不同的推导（Bohren & Huffman第5.1节），
    里面既没有球贝塞尔函数也没有任何递推。

    **``x^4``就是``1/lambda^4``**：固定半径与折射率时``x = 2 pi a / lambda``，
    于是散射**截面**``C_sca = Q_sca pi a^2 ∝ lambda^-4``——天空为什么是蓝的。
    `tests/test_optics_mie.py`把这两件事分成两条门：一条只用**严格级数**
    验``C_sca lambda^4``是常数（不碰本函数），另一条才把级数与本函数对拍。
    """

    x = _checked_size_parameter(size_parameter)
    factor = clausius_mossotti_factor(refractive_index)
    return 8.0 / 3.0 * x**4 * (factor.real * factor.real + factor.imag * factor.imag)


def rayleigh_absorption_efficiency(
    *, size_parameter: float, refractive_index: complex
) -> float:
    """瑞利极限的吸收效率``Q_abs = 4 x Im[(m^2-1)/(m^2+2)]``。

    **它随``x``只是一次方**，而散射是四次方——于是小球在吸收介质里
    "吸收远大于散射"，那正是瑞利区里判两者谁主导的那条线。
    无吸收介质（`m`实）上本函数恒为0。
    """

    x = _checked_size_parameter(size_parameter)
    return 4.0 * x * clausius_mossotti_factor(refractive_index).imag


def mie_unitarity_residual(coefficients: MieCoefficients) -> float:
    """无吸收介质上每一阶都满足``|a_n|^2 = Re(a_n)``——本函数给最大残差。

    **这条比"总消光等于总散射"更硬**：后者可以靠两阶的误差互相抵消
    而蒙混过去，逐阶的幺正性不行。几何上它说的是"系数落在以1/2为心、
    1/2为半径的圆上"，即无损球只能给散射波一个**相移**。

    `m`带虚部时本函数**没有意义**（那时残差正是吸收），所以它拒答——
    "这个构型下判据不成立"与"判据成立且等于某个数"是两件事。
    """

    if coefficients.refractive_index.imag != 0.0:
        raise OpticsError(
            f"幺正性判据只对无吸收介质成立，收到m={coefficients.refractive_index!r}："
            "有吸收时``|a_n|^2 < Re(a_n)``，那个差正是被吸收掉的部分，"
            "**不是残差**"
        )
    worst = 0.0
    for value in coefficients.a + coefficients.b:
        worst = max(worst, abs(value.real * value.real + value.imag * value.imag - value.real))
    return worst


__all__ = [
    "MIE_ABSORPTION_FLOOR",
    "MIE_GEOMETRIC_OPTICS_EXTINCTION",
    "MIE_REFRACTIVE_INDEX_CONVENTION",
    "MIE_SERIES_TAIL_TOLERANCE",
    "MIE_TESTED_REFRACTIVE_INDEX_MODULUS_MAX",
    "MIE_TESTED_SIZE_PARAMETER_MAX",
    "MIE_TESTED_SIZE_PARAMETER_MIN",
    "MIE_TIME_CONVENTION",
    "MieCoefficients",
    "MieEfficiencies",
    "clausius_mossotti_factor",
    "mie_coefficients",
    "mie_efficiencies",
    "mie_max_order",
    "mie_unitarity_residual",
    "rayleigh_absorption_efficiency",
    "rayleigh_scattering_efficiency",
    "size_parameter_from_radius",
]
