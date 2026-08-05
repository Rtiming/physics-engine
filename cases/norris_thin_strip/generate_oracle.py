#!/usr/bin/env python3
"""Norris薄带的金标生成器——**独立数值路径，不import被验内核**。

被验内核用float64闭式（arctan、log1p、幂级数）。本生成器走的是
**50位十进制定点算术**：`decimal.Decimal`的`sqrt`/`ln`是标准库自带的
正确舍入实现，`arctan`本文件自己用**半角约化＋Taylor**写
（`atan(t) = 2·atan(t/(1+sqrt(1+t²)))`反复用到`|t| < 0.05`再展开），
`π`用Machin公式`16·atan(1/5) − 4·atan(1/239)`从同一个`atan`得出。

这条路与被验实现的关系是：**同一条闭式，两套完全不同的算术**。
它抓得住的是float64上的相消、分支切换、写法退化——不抓"公式本身抄错了"。
公式本身由**电流守恒`∫K dx = I`**（另一条oracle）与**两个零容差极限**去守，
这三条合起来才构成判据，单看哪一条都不够。案例页第六节写明了这个边界。

参考解出处（案例页第二节有完整交代）：
W. T. Norris, *J. Phys. D* **3** (1970) 489；片电流分布的采信写法见
Pardo等arXiv:1410.0772；交流损耗两式见Grilli等arXiv:1306.6251式46、47。
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

ALGORITHM_ID = "algorithm:oracle/norris_thin_strip"
ALGORITHM_VERSION = "1.0.0"

#: 参考算术的位数。float64是约16位十进制，50位留34位余量——
#: 参考本身的误差比被验对象小20个数量级，不参与容差账。
REFERENCE_PRECISION = 50

getcontext().prec = REFERENCE_PRECISION + 10
ONE = Decimal(1)
TWO = Decimal(2)

#: 代表性REBCO带材构型（4 mm宽、1 μm超导层、Jc = 3e10 A/m² ⟹ Ic = 120 A）。
#: **这是案例的输入不是材料记录**：`Jc`是材料属性，但轴2的单位后缀基础集里
#: 没有安培，材料记录今天装不下它（决策0047第四节）。
WIDTH_M = 4.0e-3
LAYER_THICKNESS_M = 1.0e-6
CRITICAL_CURRENT_DENSITY_A_PER_M2 = 3.0e10
CRITICAL_SHEET_CURRENT_A_PER_M = CRITICAL_CURRENT_DENSITY_A_PER_M2 * LAYER_THICKNESS_M
CRITICAL_CURRENT_A = WIDTH_M * CRITICAL_SHEET_CURRENT_A_PER_M

#: 片电流分布采样的电流比。跨两个数量级：`i = 0.01`时磁通前沿几乎贴着带边
#: （`b/a = 0.99995`），是相消最凶的一档；`i = 0.99`时几乎全穿透。
PROFILE_RATIOS: tuple[str, ...] = ("0.01", "0.1", "0.25", "0.5", "0.75", "0.9", "0.99")

#: 采样点在未穿透区内的相对位置`|x|/b`。0.999那一档是**故意贴着磁通前沿取的**：
#: 那里`b² − x²`相消最凶，只在好算的地方采样是这类案例最容易犯的错。
PROFILE_FRACTIONS: tuple[str, ...] = ("0", "0.25", "0.5", "0.75", "0.9", "0.99", "0.999")

#: 电流守恒扫描的电流比（research/12实测的0.1到0.99那一段）。
CONSERVATION_RATIOS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

#: 代换式求积的Simpson区间数（`x = b·sin t`后被积函数解析，256段即到7.8e-11）。
ARCSINE_NODES = 256

#: 直接在`x`上做复合Simpson的区间数与电流比——**这一条是为了把
#: research/12那两个数复现出来并冻住**：4e5点时`i = 0.1`给1.3e-7、`i = 0.99`给5e-10。
DIRECT_NODES = 400000
DIRECT_RATIOS: tuple[float, ...] = (0.1, 0.99)

#: 损耗采样的电流比。0.6999/0.7/0.7001三点**跨过实现的级数↔log1p切换点**，
#: 冻住切换处的连续性；1.0是两式的闭式端点（薄带`2(2ln2−1)`、椭圆恰为1）。
LOSS_RATIOS: tuple[str, ...] = (
    "0.000001", "0.0001", "0.001", "0.01", "0.1", "0.3", "0.5",
    "0.6999", "0.7", "0.7001", "0.9", "0.99", "1",
)

#: 小幅值渐近的判据点，以及量幂次用的十倍程两端。
ASYMPTOTIC_RATIO = "0.001"
DECADE_RATIOS: tuple[str, str] = ("0.0001", "0.001")

#: 量纲损耗的判据点（`Ic = 120 A`、幅值60 A）。
LOSS_AMPLITUDE_A = 60.0


# --------------------------------------------------------------------------
# 50位算术的三个原语：atan、pi、以及由它们组成的闭式
# --------------------------------------------------------------------------
def decimal_atan(value: Decimal) -> Decimal:
    """``arctan``的50位求值：半角约化到`|t| < 0.05`后Taylor展开。

    约化用`atan(t) = 2·atan(t / (1 + sqrt(1 + t²)))`——它对任意大的`t`都收敛，
    而Taylor级数只在`|t| < 1`收敛且靠近1时慢得没法用。
    """

    if value < 0:
        return -decimal_atan(-value)
    halvings = 0
    while value > Decimal("0.05"):
        value = value / (ONE + (ONE + value * value).sqrt())
        halvings += 1
    total = Decimal(0)
    term = value
    square = value * value
    index = 0
    floor = Decimal(10) ** (-(getcontext().prec + 5))
    while True:
        contribution = term / (2 * index + 1)
        total += -contribution if index % 2 else contribution
        if abs(contribution) < floor:
            break
        term *= square
        index += 1
    return total * (TWO**halvings)


#: Machin公式的`π`——与`decimal_atan`同源，参考路径里不出现`math.pi`。
PI = 16 * decimal_atan(ONE / 5) - 4 * decimal_atan(ONE / 239)


def decimal_flux_front(half_width: Decimal, ratio: Decimal) -> Decimal:
    """``b = a·sqrt(1 − i²)``的50位值。"""

    return half_width * ((ONE - ratio) * (ONE + ratio)).sqrt()


def decimal_sheet_current(
    position: Decimal, half_width: Decimal, ratio: Decimal, sheet: Decimal
) -> Decimal:
    """``K(x)``的50位值。**按原始写法算**（`sqrt((a²−b²)/(b²−x²))`）：

    50位下相消不构成威胁，所以参考路径**故意不用**被验实现那条
    `(a·i)/sqrt((b−|x|)(b+|x|))`的重写——两边连代数形式都不同，
    才能抓住重写本身写错了的情形。
    """

    front = decimal_flux_front(half_width, ratio)
    if abs(position) >= front:
        return sheet
    numerator = half_width * half_width - front * front
    denominator = front * front - position * position
    return (TWO * sheet / PI) * decimal_atan((numerator / denominator).sqrt())


def decimal_strip_loss(ratio: Decimal) -> Decimal:
    """``2·[(1+i)ln(1+i) + (1−i)ln(1−i) − i²]``的50位值。

    50位下`i = 1e-3`处的相消只吃掉约6位，剩44位——比float64的整个位宽还宽，
    所以参考路径**直接按原式算**，不走级数。这正是它与被验实现独立的地方：
    被验实现在这一段必须走级数才不塌，参考不必。
    """

    if ratio == 0:
        return Decimal(0)
    if ratio == ONE:
        return 2 * ((ONE + ratio) * (ONE + ratio).ln() - ratio * ratio)
    return 2 * (
        (ONE + ratio) * (ONE + ratio).ln()
        + (ONE - ratio) * (ONE - ratio).ln()
        - ratio * ratio
    )


def decimal_ellipse_loss(ratio: Decimal) -> Decimal:
    """``(2−i)·i + 2(1−i)ln(1−i)``的50位值。"""

    if ratio == 0:
        return Decimal(0)
    if ratio == ONE:
        return (2 - ratio) * ratio
    return (2 - ratio) * ratio + 2 * (ONE - ratio) * (ONE - ratio).ln()


def decimal_dimensional_loss(normalised: Decimal, critical_current: Decimal) -> Decimal:
    """``Ql = μ0·Ic²·F(i)/(2π)``的50位值，`μ0 = 4π×10⁻⁷ H/m`（2019前的定义值）。"""

    permeability = 4 * PI * Decimal(10) ** -7
    return permeability * critical_current * critical_current * normalised / (2 * PI)


def main() -> int:
    half_width = Decimal(WIDTH_M) / 2
    sheet = Decimal(CRITICAL_SHEET_CURRENT_A_PER_M)
    critical_current = Decimal(WIDTH_M) * sheet

    # ---- oracle 1：片电流分布 -------------------------------------------
    positions: list[float] = []
    profile: list[float] = []
    for text in PROFILE_RATIOS:
        ratio = Decimal(text)
        front = decimal_flux_front(half_width, ratio)
        for fraction in PROFILE_FRACTIONS:
            # 采样点先落成float64（它是**输入**，两边必须逐位同一个数），
            # 再以`Decimal(float)`精确回读进参考路径。
            position = float(front * Decimal(fraction))
            positions.append(position)
            profile.append(
                float(decimal_sheet_current(Decimal(position), half_width, ratio, sheet))
            )

    # 饱和支：磁通前沿与带边之间取中点，以及带边本身。两处都恰为`Kc`，零容差。
    saturated_positions: list[float] = []
    for text in PROFILE_RATIOS:
        front = decimal_flux_front(half_width, Decimal(text))
        saturated_positions.append(float((front + half_width) / 2))
        saturated_positions.append(float(half_width))
    saturated = [CRITICAL_SHEET_CURRENT_A_PER_M] * len(saturated_positions)

    # ---- oracle 2：电流守恒 ---------------------------------------------
    conservation = [ratio * CRITICAL_CURRENT_A for ratio in CONSERVATION_RATIOS]
    direct = [ratio * CRITICAL_CURRENT_A for ratio in DIRECT_RATIOS]

    # ---- oracle 3：磁通前沿与它的两个极限 --------------------------------
    fronts = [
        float(decimal_flux_front(half_width, Decimal(text))) for text in PROFILE_RATIOS
    ]

    # ---- oracle 4：交流损耗 ---------------------------------------------
    strip = [float(decimal_strip_loss(Decimal(text))) for text in LOSS_RATIOS]
    ellipse = [float(decimal_ellipse_loss(Decimal(text))) for text in LOSS_RATIOS]

    asymptotic = Decimal(ASYMPTOTIC_RATIO)
    strip_asymptotic = decimal_strip_loss(asymptotic)
    ellipse_asymptotic = decimal_ellipse_loss(asymptotic)
    strip_power_law = asymptotic**4 / 3
    ellipse_power_law = asymptotic**3 / 3
    # **expected是幂律本身（比值1、幂次4与3），不是闭式的实测值**：
    # 这两条判据要问的是"闭式趋不趋于这条幂律"，把实测值冻成expected就变成了
    # "闭式还是不是上次那个数"，渐近这件事一个字也没验到（第一版正是这么写的，
    # 余量算出来是无穷大——那是判据失效的信号）。容差承载渐近的下一阶。
    strip_ratio_measured = float(strip_asymptotic / strip_power_law)
    ellipse_ratio_measured = float(ellipse_asymptotic / ellipse_power_law)

    low, high = (Decimal(text) for text in DECADE_RATIOS)
    strip_slope = float(
        (decimal_strip_loss(high) / decimal_strip_loss(low)).ln() / (high / low).ln()
    )
    ellipse_slope = float(
        (decimal_ellipse_loss(high) / decimal_ellipse_loss(low)).ln() / (high / low).ln()
    )

    amplitude_ratio = Decimal(LOSS_AMPLITUDE_A) / critical_current
    strip_dimensional = float(
        decimal_dimensional_loss(decimal_strip_loss(amplitude_ratio), critical_current)
    )
    ellipse_dimensional = float(
        decimal_dimensional_loss(decimal_ellipse_loss(amplitude_ratio), critical_current)
    )

    oracles = [
        {
            "id": "oracle:norris/sheet_current_profile",
            "inputs": {
                "kind": "norris_thin_strip_conformal_map",
                "width_m": WIDTH_M,
                "critical_sheet_current_a_per_m": CRITICAL_SHEET_CURRENT_A_PER_M,
                "critical_current_a": CRITICAL_CURRENT_A,
                "current_ratios": [float(text) for text in PROFILE_RATIOS],
                "position_fractions": [float(text) for text in PROFILE_FRACTIONS],
                "positions_m": positions,
                "saturated_positions_m": saturated_positions,
                "reference_precision": REFERENCE_PRECISION,
            },
            "expected": {
                "sheet_current_values_a_per_m": profile,
                "saturated_branch_values_a_per_m": saturated,
            },
            "tolerances": {
                "sheet_current_values_a_per_m": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "被验实现申报的相对精度SHEET_CURRENT_RELATIVE_ACCURACY。"
                              "实测最坏4.48e-14（202 eps）落在i=0.01、|x|/b=0.999——"
                              "那里b²−x²相消最凶且b几乎贴着带边。取1e-12是它的22倍。"
                              "**判相对不判绝对**：K跨两个数量级（中心Kc/3量级、"
                              "近前沿处趋近Kc），一个绝对容差在两端一定有一端没意义。"
                              "注意这个数是**重写后**的：按原式(a²−b²)/(b²−x²)直接算"
                              "同样样点最坏1.07e-12，24倍差距，1e-12会当场红",
                },
                "saturated_branch_values_a_per_m": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "b ≤ |x| ≤ w/2上闭式恰等于Kc，实现直接返回入参不做任何运算——"
                              "零容差。这一条同时锁住两支的**条件方向**："
                              "把`|x| >= b`写成`|x| <= b`时中心区会返回Kc、"
                              "饱和区会去算arctan（自变量为负数的sqrt，当场炸）",
                },
            },
        },
        {
            "id": "oracle:norris/current_conservation",
            "inputs": {
                "kind": "integral_identity_of_the_closed_form",
                "width_m": WIDTH_M,
                "critical_sheet_current_a_per_m": CRITICAL_SHEET_CURRENT_A_PER_M,
                "critical_current_a": CRITICAL_CURRENT_A,
                "current_ratios": list(CONSERVATION_RATIOS),
                "arcsine_simpson_nodes": ARCSINE_NODES,
                "direct_current_ratios": list(DIRECT_RATIOS),
                "direct_simpson_nodes": DIRECT_NODES,
            },
            "expected": {
                "arcsine_quadrature_currents_a": conservation,
                "direct_quadrature_currents_a": direct,
            },
            "tolerances": {
                "arcsine_quadrature_currents_a": {
                    "abs": 0.0, "rel": 1.0e-9,
                    "reason": "∫K dx必须恰等于I。本条走x = b·sin(t)代换后的复合Simpson："
                              "代换把端点的竖直斜率吃掉，被积函数K(b·sin t)·b·cos t"
                              "在t∈[0, π/2]上解析，Simpson恢复四阶收敛。"
                              "256段实测最坏7.84e-11（i=0.1），取1e-9是它的12.8倍。"
                              "**这条容差是求积法的属性不是物理的属性**——"
                              "下一条同一个恒等式在直接求积下要松三个数量级",
                },
                "direct_quadrature_currents_a": {
                    "abs": 0.0, "rel": 1.0e-6,
                    "reason": "同一个恒等式，直接在x上做复合Simpson。"
                              "arctan在|x|→b处斜率竖直（dK/dx→∞），"
                              "Simpson的四阶收敛在奇异端点上退化，"
                              "I/Ic越小磁通前沿越贴带边、退化越重。"
                              "4e5点实测：i=0.1给1.280e-7、i=0.99给4.986e-10——"
                              "**与research/12记的1.3e-7与5e-10逐位复现**。"
                              "取1e-6是最坏值的7.8倍。这一条冻的不是精度是**病态**："
                              "它在这里就是为了让'为什么另一条能到1e-9'有个对照",
                },
            },
        },
        {
            "id": "oracle:norris/flux_front",
            "inputs": {
                "kind": "critical_state_penetration_front",
                "width_m": WIDTH_M,
                "critical_current_a": CRITICAL_CURRENT_A,
                "current_ratios": [float(text) for text in PROFILE_RATIOS],
            },
            "expected": {
                "flux_front_positions_m": fronts,
                "front_at_zero_current_m": 0.5 * WIDTH_M,
                "front_at_critical_current_m": 0.0,
            },
            "tolerances": {
                "flux_front_positions_m": {
                    "abs": 0.0, "rel": 2.5e-15,
                    "reason": "b = (w/2)·sqrt(1−i²)。误差不是均匀的1eps："
                              "`1 − i`在i→1时相消，i=0.99处`1−i`本身的相对误差约9e-16"
                              "（0.99的表示误差相对0.01被放大100倍），sqrt折半后约4.5e-16。"
                              "**灵敏度写得出来**：dlnb/dlni = −i²/(1−i²)，i=0.99处49.25。"
                              "实测逐点0（i≤0.1）到5.76e-16（i=0.99），取2.5e-15是最坏的4.3倍。"
                              "第一版按'各1eps'拍了5e-16，i=0.99那一点当场红——"
                              "**红得对，它测出了这条式在i→1处不是1eps而是50eps**",
                },
                "front_at_zero_current_m": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "I = 0时sqrt(1−0) = 1恰好，b = w/2**逐位精确**——"
                              "电流全挤在两条边、中间完全没穿透。零容差。"
                              "**警告**：这条极限与下一条对'sqrt漏写'那个错法一条都抓不住"
                              "（b = (w/2)(1−i²)在两端给出同样的值），"
                              "抓它的是电流守恒。详见决策0047第三节的必红矩阵",
                },
                "front_at_critical_current_m": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "I = Ic时1−i² = 0恰好，b = 0**逐位精确**——完全穿透。"
                              "零容差。expected是0.0，任何非零结果都超出",
                },
            },
        },
        {
            "id": "oracle:norris/ac_loss",
            "inputs": {
                "kind": "norris_hysteresis_loss_per_cycle",
                "current_ratios": [float(text) for text in LOSS_RATIOS],
                "asymptotic_ratio": float(ASYMPTOTIC_RATIO),
                "decade_ratios": [float(text) for text in DECADE_RATIOS],
                "critical_current_a": CRITICAL_CURRENT_A,
                "loss_amplitude_a": LOSS_AMPLITUDE_A,
                "reference_precision": REFERENCE_PRECISION,
            },
            "expected": {
                "strip_normalised_losses": strip,
                "ellipse_normalised_losses": ellipse,
                "strip_power_law_ratio": 1.0,
                "ellipse_power_law_ratio": 1.0,
                "strip_decade_slope": 4.0,
                "ellipse_decade_slope": 3.0,
                "strip_loss_j_per_m_per_cycle": strip_dimensional,
                "ellipse_loss_j_per_m_per_cycle": ellipse_dimensional,
            },
            "tolerances": {
                "strip_normalised_losses": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "被验实现申报的相对精度NORRIS_LOSS_RELATIVE_ACCURACY。"
                              "实测最坏2.96e-15（13.3 eps，i=0.709——刚过切换点的log1p分支），取3.4倍。"
                              "**这条容差挡的是相消**：把log1p换成log(1±i)时"
                              "i=1e-3处相对偏差6.6e-4——比容差大11个数量级，当场红。"
                              "research/12记的'3.331e-13'正是那条塌掉的写法给出的数",
                },
                "ellipse_normalised_losses": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "同上。椭圆式的相消比薄带式轻（结果~i³/3而中间项~i，"
                              "相消因子6/i²而非6/i³），全区间实测最坏1.12e-15（5.0 eps，i=0.658）",
                },
                "strip_power_law_ratio": {
                    "abs": 0.0, "rel": 1.0e-6,
                    "reason": "小幅值渐近`→ i⁴/3`：测量量是闭式值除以i⁴/3，"
                              "**expected是幂律断言的那个1不是闭式的实测值**。"
                              "i=1e-3实测1+4.000002e-07。"
                              "**下一阶是O(i²)相对修正**，系数由级数给死："
                              "次项2i⁶/15比首项i⁴/3是0.4·i²，i=1e-3处4.0e-07，"
                              "与实测吻合到六位。取1e-6是它的2.5倍。"
                              "**容差是算出来的**：换i=1e-2它会涨到4e-5",
                },
                "ellipse_power_law_ratio": {
                    "abs": 0.0, "rel": 1.0e-3,
                    "reason": "小幅值渐近`→ i³/3`，**expected同样是那个1**。"
                              "i=1e-3实测1+5.003002e-04。"
                              "**下一阶是O(i)相对修正不是O(i²)**："
                              "次项i⁴/6比首项i³/3是i/2，i=1e-3处5.0e-04。"
                              "research/12的判据表把两条渐近的理由都写成"
                              "'下一阶是O(i²)相对修正'——对薄带成立、对椭圆不成立，"
                              "**两条容差因此必须差三个数量级**（本轮订正，见决策0047）。"
                              "取1e-3是实测的2.0倍：这是三条里余量最薄的一条，"
                              "薄是因为它测的是物理不是数值",
                },
                "strip_decade_slope": {
                    "abs": 0.0, "rel": 1.0e-7,
                    "reason": "十倍程双对数斜率ln(F(1e-3)/F(1e-4))/ln(10)，"
                              "**expected是幂次4本身**。它与比值判据不重复："
                              "比值验的是系数1/3，斜率验的是幂次——"
                              "把i⁴/3写成i⁴/4的实现比值红而斜率绿。"
                              "偏离量**算得出来**：lnF = 4·ln i + ln(1/3) + 0.4·i²，"
                              "斜率 = 4 + 0.4·(1e-6 − 1e-8)/ln10 = 4 + 1.7198e-7，"
                              "实测4.000000171980673，逐位吻合。"
                              "相对偏离4.2995e-8，取1e-7是它的2.3倍",
                },
                "ellipse_decade_slope": {
                    "abs": 0.0, "rel": 2.0e-4,
                    "reason": "同上，**expected是幂次3**。"
                              "**3与4的差别是这两条式的招牌**（Norris 1970的原始发现）："
                              "一条案例同时验形状与幂次。偏离量同样算得出来："
                              "lnF ≈ 3·ln i + ln(1/3) + i/2，"
                              "斜率 = 3 + (1e-3 − 1e-4)/(2·ln10) = 3 + 1.9543e-4，"
                              "实测3.0001955077981703（差的1e-8是再下一阶）。"
                              "相对偏离6.5169e-5，取2e-4是它的3.1倍。"
                              "**这条容差比薄带那条松2000倍，同一个原因**："
                              "椭圆的下一阶是O(i)、薄带是O(i²)",
                },
                "strip_loss_j_per_m_per_cycle": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "**单位边界**：无量纲的2π·Ql/(μ0·Ic²)乘回μ0·Ic²/(2π)才是焦耳。"
                              "Ic = 120 A、幅值60 A。μ0取2019年SI重新定义**之前**的"
                              "定义值4π×10⁻⁷（Norris推导用的就是它）；"
                              "CODATA实测值与它差约1.3e-10，**换过去这条会红**，"
                              "红得对：换物理常数是一次决定不是优化。"
                              "**容差不是按这一点的实测取的**（i=0.5落在级数支，实测恰为0），"
                              "而是随归一化量的申报精度1e-14——换一个幅值就会走log1p支，"
                              "按这一点实测掐紧等于给别的幅值埋雷",
                },
                "ellipse_loss_j_per_m_per_cycle": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "同上（实测1.66e-16）。与薄带那条并列不是重复："
                              "它验的是量纲包装对两条归一化式都接对了，"
                              "而不是只对其中一条接对",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/norris_thin_strip",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/norris_thin_strip/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"Ic = {CRITICAL_CURRENT_A} A, Kc = {CRITICAL_SHEET_CURRENT_A_PER_M} A/m; "
        f"strip power-law ratio {strip_ratio_measured!r} "
        f"(deviation {strip_ratio_measured - 1.0:.6e}), "
        f"ellipse {ellipse_ratio_measured!r} "
        f"(deviation {ellipse_ratio_measured - 1.0:.6e}); "
        f"slopes {strip_slope!r} / {ellipse_slope!r}; "
        f"pi check {float(PI) == math.pi}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
