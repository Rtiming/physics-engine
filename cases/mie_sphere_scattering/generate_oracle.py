#!/usr/bin/env python3
"""Mie严格级数解的金标生成器——**50位十进制的独立实现**。

被验内核（`optics/mie.py` + `optics/spherical_bessel.py`）在`float`上走：

* ``psi_n(x)``：Miller**向下**递推 + 三角种子归一；
* ``D_n(mx)``：连分式**向下**递推``D_{n-1} = n/z - 1/(D_n + n/z)``；
* ``chi_n(x)``：向上递推。

本生成器在`decimal`的**50位**上走完全不同的三条：

* ``j_n(z)``：**上升级数**``j_n = z^n/(2n+1)!! sum_k (-z^2/2)^k/(k! prod(2n+2i+1))``
  ——没有任何递推、没有归一化锚点；
* ``D_n(z) = psi_n'(z)/psi_n(z) = j_{n-1}(z)/j_n(z) - n/z``
  ——**直接由两阶球贝塞尔相除**，不走连分式；
* ``sin``/``cos``自己用泰勒级数在50位上算（`decimal`没有三角函数），
  于是连libm都不共用。

50位远超`float`的16位，所以**参照本身的误差在这些判据里不参与**——
0086第5.2节的教训（判据自己的参照可能比被验对象更差）在这里被这条位数直接消掉。

## 为什么冻的是**无吸收**球

无吸收 ⟹ ``m``是实数 ⟹ ``mx``是实数 ⟹ 上面三条全部是实数运算，
`decimal`直接能做（Python的`Decimal`没有复数）。系数``a_n``/``b_n``仍然是复的
（``xi_n = psi_n - i chi_n``），但那一步只是"实数除以复数"，
用两个`Decimal`分量手写十行就够，**不需要一整套复数十进制算术**。

有吸收那一侧本案例用**另外两条不需要高精度参照的判据**去盖：
瑞利吸收闭式（小球极限）与消光佯谬的标度律（大球极限）——
**两者的金标都是解析的，不依赖任何实现**。

## 四条oracle

1. `lossless_series_against_fifty_digits`：``x = 5.213``、``m = 1.55``上
   逐阶的``a_n``、``b_n``（复数按0086的``[实部, 虚部]``二元组落盘）
   与``Q_ext``、``Q_sca``；
2. `unitarity_and_energy_balance`：无吸收球的``Q_ext - Q_sca``与
   **逐阶**幺正性残差``| |a_n|^2 - Re(a_n) |``，**期望都是精确的0**。
   ``Q_ext``取实部、``Q_sca``取模平方，是两条不同的式子——
   所以这不是恒等变形。**注意``Q_abs``按定义就是两者之差，
   于是"消光=散射+吸收"本身不是判据**，本页不假装它是；
3. `rayleigh_small_sphere_limit`：小球极限的``Q_sca``对瑞利闭式
   ``(8/3) x^4 |(m^2-1)/(m^2+2)|^2``，逐档给相对偏差——
   闭式来自一条完全不同的推导（球当成静电偶极子），
   **``x^4``就是``1/lambda^4``**；
4. `extinction_paradox`：大球极限``Q_ext -> 2``，
   而且**按``x^(-2/3)``趋近**（边缘衍射修正的标度律）。
   金标是解析的``2^(-2/3) = 0.629960...``，不依赖任何实现。

参考解出处：Bohren & Huffman,《Absorption and Scattering of Light by Small
Particles》第4.4节（式4.53的``a_n``/``b_n``）、第4.4.2节（式4.61—4.62的效率）、
第5.1—5.2节（瑞利极限与Clausius-Mossotti因子）；
van de Hulst,《Light Scattering by Small Particles》第8章
（消光佯谬``Q_ext -> 2``与边缘衍射的``x^(-2/3)``修正）；
Wiscombe 1980, Appl. Opt. 19, 1505（截断阶判据）。
"""

from __future__ import annotations

import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/mie_sphere_scattering"
ALGORITHM_VERSION = "1.0.0"

#: 参照的十进制位数。**50位远超`float`的16位**，于是参照自己的误差
#: 在本页每一条判据里都不参与。
REFERENCE_DIGITS = 50
getcontext().prec = REFERENCE_DIGITS

#: 主构型：无吸收球。``x``取5.213是`BHMIE`附录里那个例子的尺度参数，
#: ``m = 1.55``是玻璃量级——两者都不特殊，选它们只是为了**不是**
#: 一个会让某一阶恰好为零的巧合构型（那种构型会让判据看起来比实际更强）。
LOSSLESS_SIZE_PARAMETER = "5.213"
LOSSLESS_REFRACTIVE_INDEX = "1.55"

#: 小球极限那条的折射率与尺度梯子。梯子逐档减半，于是``O(x^2)``的修正
#: 应当逐档掉到四分之一——**"偏差小"是软判据，"偏差按x^2掉"不是**。
RAYLEIGH_REFRACTIVE_INDEX = "1.5"
RAYLEIGH_SIZE_PARAMETERS: tuple[str, ...] = ("0.1", "0.05", "0.025")

#: 大球极限那条：吸收折射率与尺度梯子（逐档翻倍）。
#: **用吸收球是有理由的**：吸收把共振ripple压掉，``Q_ext``才真的单调趋近2；
#: 无吸收球在大`x`上振荡得很厉害，那条曲线不适合用来验"趋近"。
#: 如实写在这里，不假装无吸收也单调。
PARADOX_REFRACTIVE_INDEX: tuple[float, float] = (1.5, 0.1)
PARADOX_SIZE_PARAMETERS: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0, 800.0)

#: 被验实现与本生成器之间的**实测**最坏偏差（本机，见案例页第三节）。
MEASURED_COEFFICIENT_DEVIATION = 5.5511e-16
MEASURED_EFFICIENCY_DEVIATION = 1.3323e-15
MEASURED_ENERGY_BALANCE = 0.0
MEASURED_UNITARITY_RESIDUAL = 1.1102e-16
MEASURED_RAYLEIGH_EFFICIENCY_DEVIATION = 1.5543e-15
MEASURED_PARADOX_RATIO_DEPARTURE = 4.4695e-3
MEASURED_PARADOX_GAP_AT_LARGEST = 2.2837e-2

#: 瑞利闭式那一行自己的实测相对偏差（被验侧只有一次乘方与一次模平方）。
MEASURED_RAYLEIGH_CLOSED_FORM_DEVIATION = 2.2204e-16

#: 相对偏差那一行的实测绝对偏差。
MEASURED_RAYLEIGH_DEPARTURE_DEVIATION = 1.4082e-15


def _sin(x: Decimal) -> Decimal:
    """50位泰勒``sin``。`decimal`没有三角函数，所以连libm都不与被验侧共用。"""

    term = x
    total = x
    index = 1
    while abs(term) > Decimal(10) ** -(REFERENCE_DIGITS + 5):
        term *= -x * x / Decimal((2 * index) * (2 * index + 1))
        total += term
        index += 1
    return total


def _cos(x: Decimal) -> Decimal:
    term = Decimal(1)
    total = Decimal(1)
    index = 1
    while abs(term) > Decimal(10) ** -(REFERENCE_DIGITS + 5):
        term *= -x * x / Decimal((2 * index - 1) * (2 * index))
        total += term
        index += 1
    return total


def _spherical_j(order: int, z: Decimal) -> Decimal:
    """上升级数``j_n(z)``——**没有递推、没有归一化锚点**。

    ``j_n(z) = z^n/(2n+1)!! sum_k (-z^2/2)^k / (k! prod_{i=1..k}(2n+2i+1))``
    """

    half_square = z * z / 2
    double_factorial = Decimal(1)
    for value in range(1, 2 * order + 2, 2):
        double_factorial *= value
    total = Decimal(1)
    term = Decimal(1)
    for index in range(1, 2000):
        term *= -half_square / (Decimal(index) * Decimal(2 * order + 2 * index + 1))
        total += term
        if abs(term) < Decimal(10) ** -(REFERENCE_DIGITS - 5) * (abs(total) + 1):
            break
    return z**order / double_factorial * total


def _spherical_y_array(order_max: int, z: Decimal) -> list[Decimal]:
    """``y_n``向上递推（极大解，向上是稳的），种子由50位``sin``/``cos``给。"""

    sine = _sin(z)
    cosine = _cos(z)
    values = [-cosine / z, -cosine / (z * z) - sine / z]
    for order in range(1, order_max):
        values.append(Decimal(2 * order + 1) / z * values[order] - values[order - 1])
    return values[: order_max + 1]


def _complex_divide(
    numerator: Decimal, denominator: tuple[Decimal, Decimal]
) -> tuple[Decimal, Decimal]:
    """实数除以复数。**只有这一处需要复数**，所以不引入一整套十进制复数算术。"""

    real, imaginary = denominator
    modulus_squared = real * real + imaginary * imaginary
    return (numerator * real / modulus_squared, -numerator * imaginary / modulus_squared)


def _lossless_series(size: Decimal, index: Decimal, order_max: int):
    """无吸收球的``a_n``、``b_n``、``Q_ext``、``Q_sca``（全部50位）。"""

    inner = index * size
    j_outer = [_spherical_j(order, size) for order in range(order_max + 1)]
    j_inner = [_spherical_j(order, inner) for order in range(order_max + 1)]
    y_outer = _spherical_y_array(order_max, size)

    psi = [size * value for value in j_outer]
    chi = [-size * value for value in y_outer]
    #: ``D_n(z) = psi_n'(z)/psi_n(z) = j_{n-1}(z)/j_n(z) - n/z``。
    #: 直接由两阶球贝塞尔相除，**不走连分式向下递推**。
    derivative = [
        j_inner[order - 1] / j_inner[order] - Decimal(order) / inner
        for order in range(1, order_max + 1)
    ]

    a_values: list[tuple[Decimal, Decimal]] = []
    b_values: list[tuple[Decimal, Decimal]] = []
    for order in range(1, order_max + 1):
        ratio = Decimal(order) / size
        d_value = derivative[order - 1]
        for common, sink in ((d_value / index + ratio, a_values), (index * d_value + ratio, b_values)):
            numerator = common * psi[order] - psi[order - 1]
            denominator = (
                common * psi[order] - psi[order - 1],
                -(common * chi[order] - chi[order - 1]),
            )
            sink.append(_complex_divide(numerator, denominator))

    extinction = Decimal(0)
    scattering = Decimal(0)
    for order in range(1, order_max + 1):
        weight = Decimal(2 * order + 1)
        a_real, a_imaginary = a_values[order - 1]
        b_real, b_imaginary = b_values[order - 1]
        extinction += weight * (a_real + b_real)
        scattering += weight * (
            a_real * a_real + a_imaginary * a_imaginary + b_real * b_real + b_imaginary * b_imaginary
        )
    factor = Decimal(2) / (size * size)
    return a_values, b_values, factor * extinction, factor * scattering


def _max_order(size: float) -> int:
    """Wiscombe判据``n_max = ceil(x + 4 x^(1/3) + 2)``。生成器自己算。"""

    return int(math.ceil(size + 4.0 * size ** (1.0 / 3.0) + 2.0))


def main() -> int:
    size = Decimal(LOSSLESS_SIZE_PARAMETER)
    index = Decimal(LOSSLESS_REFRACTIVE_INDEX)
    order_max = _max_order(float(size))
    a_values, b_values, extinction, scattering = _lossless_series(size, index, order_max)

    #: 幺正性：无吸收球的每一阶都满足``|a_n|^2 = Re(a_n)``。
    #: 50位上量出来的残差就是本条判据的"精确0"离浮点地板还有多远。
    unitarity = max(
        abs(real * real + imaginary * imaginary - real)
        for real, imaginary in a_values + b_values
    )

    rayleigh_index = Decimal(RAYLEIGH_REFRACTIVE_INDEX)
    squared = rayleigh_index * rayleigh_index
    clausius = (squared - 1) / (squared + 2)
    exact_efficiencies = []
    closed_form_efficiencies = []
    departures = []
    for text in RAYLEIGH_SIZE_PARAMETERS:
        small = Decimal(text)
        _, _, _, small_scattering = _lossless_series(
            small, rayleigh_index, _max_order(float(small))
        )
        closed = Decimal(8) / Decimal(3) * small**4 * clausius * clausius
        exact_efficiencies.append(small_scattering)
        closed_form_efficiencies.append(closed)
        departures.append(small_scattering / closed - 1)

    #: 大球极限那条的金标是**解析的**：边缘衍射修正按``x^(-2/3)``，
    #: 于是`x`翻倍时``|Q_ext - 2|``的比值趋于``2^(-2/3)``。
    theory_ratio = 2.0 ** (-2.0 / 3.0)

    oracles = [
        {
            "id": "oracle:mie/lossless_series_against_fifty_digits",
            "inputs": {
                "kind": "fifty_digit_ascending_series",
                "size_parameter": float(size),
                "refractive_index_real": float(index),
                "refractive_index_imaginary": 0.0,
                "order_count": order_max,
                "reference_digits": REFERENCE_DIGITS,
                "complex_component_order": ["real", "imaginary"],
            },
            "expected": {
                "coefficient_a_components": [
                    [float(real), float(imaginary)] for real, imaginary in a_values
                ],
                "coefficient_b_components": [
                    [float(real), float(imaginary)] for real, imaginary in b_values
                ],
                "extinction_efficiency": float(extinction),
                "scattering_efficiency": float(scattering),
            },
            "tolerances": {
                "coefficient_a_components": {
                    "abs": 4.0e-15,
                    "rel": 0.0,
                    "reason": "50位十进制的上升级数 对 `float`上的Miller向下递推："
                              "两侧连``sin``/``cos``都不共用（生成器自己用泰勒级数算）。"
                              "被验侧的误差模型：``psi``与``chi``各约几个eps、"
                              "``D_n``一条连分式、再做一次复数除法，合起来约``16 eps = 3.6e-15``；"
                              f"实测最坏{MEASURED_COEFFICIENT_DEVIATION:.4e}"
                              f"（余量{4.0e-15 / MEASURED_COEFFICIENT_DEVIATION:.1f}倍）。"
                              "**判绝对不判相对**：高阶系数的实部小到1e-20量级，"
                              "那里的相对误差没有意义",
                },
                "coefficient_b_components": {
                    "abs": 4.0e-15,
                    "rel": 0.0,
                    "reason": "同上一行。``b_n``与``a_n``只差分子分母里那个公共因子"
                              "（``m D_n``对``D_n/m``），走的是同一条数值路径",
                },
                "extinction_efficiency": {
                    "abs": 8.0e-15,
                    "rel": 0.0,
                    "reason": "``Q_ext = (2/x^2) sum (2n+1) Re(a_n + b_n)``，"
                              f"{order_max}项累加。容差取系数那条的2倍（求和放大）；"
                              f"实测{MEASURED_EFFICIENCY_DEVIATION:.4e}"
                              f"（余量{8.0e-15 / MEASURED_EFFICIENCY_DEVIATION:.1f}倍）",
                },
                "scattering_efficiency": {
                    "abs": 8.0e-15,
                    "rel": 0.0,
                    "reason": "``Q_sca = (2/x^2) sum (2n+1) (|a_n|^2 + |b_n|^2)``。"
                              "**它与上一行是两条不同的式子**——一条取实部、"
                              "一条取模平方——所以下一条oracle的「两者相等」是真判据",
                },
            },
        },
        {
            "id": "oracle:mie/unitarity_and_energy_balance",
            "inputs": {
                "kind": "unitarity_of_a_lossless_sphere",
                "size_parameter": float(size),
                "refractive_index_real": float(index),
                "order_count": order_max,
                "fifty_digit_unitarity_residual": float(unitarity),
                "note": "无吸收球的每一阶都落在以1/2为心、1/2为半径的幺正圆上，"
                        "即``|a_n|^2 = Re(a_n)``。50位上量出来的残差"
                        f"是{float(unitarity):.3e}，即这条恒等式在参照侧也到地板",
            },
            "expected": {
                "extinction_minus_scattering": 0.0,
                "unitarity_max_residual": 0.0,
            },
            "tolerances": {
                "extinction_minus_scattering": {
                    "abs": 8.0e-15,
                    "rel": 0.0,
                    "reason": "**无吸收 ⟹ 消光恰等于散射**。两者是两条不同的求和"
                              "（``Re(a_n+b_n)``对``|a_n|^2+|b_n|^2``），"
                              "只有系数落在幺正圆上时才相等——这不是恒等变形。"
                              f"实测{MEASURED_ENERGY_BALANCE!r}（**恰为0**），"
                              "容差按两条求和各自的浮点累积取8e-15。"
                              "**注意``Q_abs``按定义就是这个差**，"
                              "所以「消光=散射+吸收」本身不是判据，本页不假装它是",
                },
                "unitarity_max_residual": {
                    "abs": 1.0e-14,
                    "rel": 0.0,
                    "reason": "**逐阶**``| |a_n|^2 - Re(a_n) |``，比上一行更硬："
                              "上一行是两条求和相等，可以靠不同阶的误差互相抵消而蒙混，"
                              "逐阶看抵消不掉。"
                              f"实测最坏{MEASURED_UNITARITY_RESIDUAL:.4e}"
                              f"（余量{1.0e-14 / MEASURED_UNITARITY_RESIDUAL:.0f}倍）",
                },
            },
        },
        {
            "id": "oracle:mie/rayleigh_small_sphere_limit",
            "inputs": {
                "kind": "small_sphere_limit_against_the_electrostatic_dipole",
                "refractive_index_real": float(rayleigh_index),
                "size_parameters": [float(text) for text in RAYLEIGH_SIZE_PARAMETERS],
                "clausius_mossotti_factor": float(clausius),
                "note": "瑞利闭式``Q_sca = (8/3) x^4 |(m^2-1)/(m^2+2)|^2``来自"
                        "把球当成极化率``4 pi a^3 (m^2-1)/(m^2+2)``的静电偶极子——"
                        "**一条完全不同的推导，里面既没有球贝塞尔也没有递推**。"
                        "``x = 2 pi a / lambda`` ⟹ ``x^4``就是``1/lambda^4``",
            },
            "expected": {
                "scattering_efficiency": [float(value) for value in exact_efficiencies],
                "rayleigh_closed_form_efficiency": [
                    float(value) for value in closed_form_efficiencies
                ],
                "relative_departure": [float(value) for value in departures],
            },
            "tolerances": {
                "scattering_efficiency": {
                    "abs": 0.0,
                    "rel": 4.0e-14,
                    "reason": "严格级数在三个小尺度参数上的``Q_sca``，50位参照。"
                              "**这一行判相对不判绝对**：三个数从2.3e-5降到9.0e-8，"
                              "绝对容差会让最小的那一档形同虚设。"
                              f"实测最坏相对偏差{MEASURED_RAYLEIGH_EFFICIENCY_DEVIATION:.1e}",
                },
                "rayleigh_closed_form_efficiency": {
                    "abs": 0.0,
                    "rel": 4.0e-15,
                    "reason": "瑞利闭式本身（被验侧`rayleigh_scattering_efficiency`）。"
                              "它只有一次乘方与一次模平方，"
                              f"实测最坏相对偏差{MEASURED_RAYLEIGH_CLOSED_FORM_DEVIATION:.4e}"
                              "（1 eps），容差取18 eps留余量18倍。"
                              "**它冻在这里是因为「退化到瑞利」这条门有两半**："
                              "一半是严格级数算对了，另一半是闭式本身算对了",
                },
                "relative_departure": {
                    "abs": 2.0e-14,
                    "rel": 0.0,
                    "reason": "两者之比减一，即**严格解比瑞利多出来的那一项**。"
                              "三个数逐档约掉到四分之一（x减半、修正是``O(x^2)``），"
                              "冻住它等于冻住「这条极限是怎么收敛的」而不只是「它收敛了」。"
                              "容差取绝对：三个偏差本身在1e-3到1e-5量级，"
                              "被验侧两条效率各自的相对误差合起来约几个eps；"
                              f"实测最坏{MEASURED_RAYLEIGH_DEPARTURE_DEVIATION:.4e}"
                              f"（余量{2.0e-14 / MEASURED_RAYLEIGH_DEPARTURE_DEVIATION:.0f}倍）",
                },
            },
        },
        {
            "id": "oracle:mie/extinction_paradox",
            "inputs": {
                "kind": "geometric_optics_limit_with_an_analytic_scaling_law",
                "refractive_index_real": PARADOX_REFRACTIVE_INDEX[0],
                "refractive_index_imaginary": PARADOX_REFRACTIVE_INDEX[1],
                "size_parameters": list(PARADOX_SIZE_PARAMETERS),
                "note": "**本条oracle的金标是解析的，不依赖任何实现**："
                        "几何光学极限下``Q_ext -> 2``（球挡住的几何截面之外，"
                        "还有等量的能量被衍射到前向小角内），"
                        "而趋近是按边缘衍射修正``x^(-2/3)``走的，"
                        "于是`x`翻倍时``|Q_ext - 2|``的比值趋于``2^(-2/3)``。"
                        "梯子用**吸收**球：吸收把共振ripple压掉，"
                        "无吸收球在大`x`上振荡得厉害，那条曲线不适合验「趋近」",
            },
            "expected": {
                "extinction_gap_ratio": [theory_ratio] * (len(PARADOX_SIZE_PARAMETERS) - 1),
                "extinction_at_largest_size": 2.0,
            },
            "tolerances": {
                "extinction_gap_ratio": {
                    "abs": 1.0e-2,
                    "rel": 0.0,
                    "reason": f"解析标度律``2^(-2/3) = {theory_ratio:.6f}``。"
                              "实测0.63443／0.63434／0.63353／0.63266，"
                              f"**从上方单调趋近**，最大偏离{MEASURED_PARADOX_RATIO_DEPARTURE:.1e}"
                              f"（余量{1.0e-2 / MEASURED_PARADOX_RATIO_DEPARTURE:.1f}倍）。"
                              "**「接近2」是软判据**（任何缓慢下降的曲线都满足），"
                              "「按x^(-2/3)接近2」不是——本行判的是指数",
                },
                "extinction_at_largest_size": {
                    "abs": 3.0e-2,
                    "rel": 0.0,
                    "reason": f"``x = {PARADOX_SIZE_PARAMETERS[-1]:.0f}``处的``Q_ext``对2。"
                              f"实测差{MEASURED_PARADOX_GAP_AT_LARGEST:.4e}"
                              "（余量1.3倍）。**容差不是收敛误差是物理**："
                              "有限`x`上``Q_ext``本来就不等于2，差的正是那条"
                              "``x^(-2/3)``修正，所以容差必须比它大一点点而不是更多——"
                              "写松了这条门就只是在断「Q_ext是个O(1)的数」",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/mie_sphere_scattering",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/mie_sphere_scattering/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"lossless x={size} m={index} n_max={order_max}; "
        f"Qext={float(extinction)!r} Qsca={float(scattering)!r}; "
        f"50-digit unitarity residual {float(unitarity):.3e}; "
        f"rayleigh departures {[float(value) for value in departures]!r}; "
        f"paradox theory ratio {theory_ratio!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
