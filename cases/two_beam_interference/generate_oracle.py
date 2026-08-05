#!/usr/bin/env python3
"""双光束干涉的金标生成器——**独立算法，不调被验内核**。

被验内核走float64：``dphi = (2 pi / lambda) * OPD``、``cos``取libm、
精确光程差走代数恒等式``2 x d / (sqrt(A) + sqrt(B))``。
本生成器**三条路全部换掉**：

1. **相位精确约化**：``OPD / lambda``用`Fraction`在有理数上精确算（两个float64
   本来就是精确有理数），取小数部分再乘2pi。内核在条纹级次`N`上损失
   ``2 pi N eps``量级的相位，本生成器不损失——**这个差就是案例要量的东西**；
2. **余弦用`Decimal`的Taylor级数**（60位），不调libm。金标因此不继承平台实现；
3. **精确光程差按定义式**``sqrt(A) - sqrt(B)``在`Decimal`60位上直接相减。
   60位下这条“朴素相减”仍准到50位，而内核那条恒等变形是另一条代数路——
   **两条路一致是判据，不是巧合**。

条纹位置由生成器**二分它自己的精确光程差**求出，不用任何闭式。

金标里的四层（与案例页的四层判据一一对应）：

1. 双光束定律本身：强度、极值、可见度（含不等强与部分相干）、退化极限；
2. 杨氏双缝：条纹间距、精确/傍轴光程差、傍轴适用性的**可计算判据**；
3. **能量守恒**：条纹的相位平均与空间平均都必须回到``I1 + I2``；
4. 迈克尔逊 + **与本子包已有FTS的桥**：ILS首零``1/(2L)``恰好是“整个扫描程
   `2L`上相对相位滑过一个整条纹”的那个波数间隔。

生成器不import `physics_engine.optics`，只用标准库与`physics_engine.oracles`。

参考解出处：Born & Wolf《Principles of Optics》第7.2节（双光束干涉与可见度）、
第10.3节（部分相干，等强时``V = |gamma|``）；Hecht《Optics》第9.3节
（杨氏双缝``dx = lambda L / d``）与第9.4节（迈克尔逊``OPD = 2 d``）。
"""

from __future__ import annotations

import math
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/two_beam_interference"
ALGORITHM_VERSION = "1.0.0"

#: 参考算术的位数。float64是16位十进制，60位留出44位余量——
#: 精确光程差的朴素相减在近轴丢约10位，仍剩50位，远在float64地板之下。
REFERENCE_DIGITS = 60
getcontext().prec = REFERENCE_DIGITS

#: pi的60位值。这是数学常数不是物理常数，直接引用；
#: `main`里有一条自校验断言它折成float64后与`math.pi`逐位相同。
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")

#: HeNe 632.8 nm——与`scalar_diffraction_airy`同一条谱线，故意的：
#: 两个光学案例共用同一个波长，任何“波长在某处被改了”的错在两边同时现形。
WAVELENGTH_M = 632.8e-9

#: 杨氏双缝构型：缝间距0.25 mm、屏距1.20 m（本科实验室的典型值，条纹3.0 mm）。
SLIT_SEPARATION_M = 0.25e-3
SCREEN_DISTANCE_M = 1.20

#: 屏上取样位置，以条纹间距为单位。含0（中央）、1/4（一般相位）、1/2（暗纹）、
#: 1（亮纹）、3/2（暗纹）、5（第五级）。**不只在“好算的地方”取样**。
POSITION_MULTIPLIERS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 5.0)

#: 傍轴偏差的取样位置，另加一个**故意不傍轴**的40倍点（x/L约0.1、偏差5e-3），
#: 用来看首阶估计式自己什么时候开始失准。x=0处偏差是0/0，故不取。
DEVIATION_MULTIPLIERS: tuple[float, ...] = (0.25, 1.0, 5.0, 40.0)

#: 相位取样：0、pi/3、pi/2、2pi/3、pi、4pi/3、3pi/2、2pi。
#: 含两个极值点（0与pi）与两个cos为负的点。
PHASE_SAMPLES: tuple[float, ...] = (
    0.0,
    math.pi / 3.0,
    math.pi / 2.0,
    2.0 * math.pi / 3.0,
    math.pi,
    4.0 * math.pi / 3.0,
    3.0 * math.pi / 2.0,
    2.0 * math.pi,
)

#: 主构型：不等强（比4:1）。**故意不取`I1 = I2 = 1`**——那个构型下
#: ``2 sqrt(I1 I2)``与``2 I1 I2``数值相同，把根号写掉的错抓不住。
INTENSITY_A = 3.0
INTENSITY_B = 0.75

#: 等强构型（取2.0不取1.0，同上理由：`2*sqrt(2*2)=4`而`2*(2*2)=8`）。
EQUAL_INTENSITY = 2.0

#: 部分相干的模。0.4是任取的一个非平凡值；等强时`V`必须恰好等于它。
PARTIAL_COHERENCE = 0.4

#: 极不等强构型：`V = 2e-6`。极值相减那条路在这里丢约6位有效数字，
#: 放大因子恰是`1/V`——案例用它量那条路与闭式的差。
FAINT_INTENSITY_A = 1.0
FAINT_INTENSITY_B = 1.0e-12

#: 近等强构型：暗纹相消的活标本。放大因子``sqrt(I2)/|sqrt(I1)-sqrt(I2)|``约2.0e7。
NEAR_EQUAL_A = 1.0
NEAR_EQUAL_B = 1.0 + 1.0e-7

#: 朴素式``I1 + I2 - 2 sqrt(I1 I2)``**返回负强度**的一对实测值
#: （40万对随机近等强扫描里12.5%命中，见决策0044第四节）。
NEGATIVE_NAIVE_A = 6.887858857269796
NEGATIVE_NAIVE_B = 6.887858857237471

#: 能量守恒的取样点数。720点整周期；余弦在整周期上的和解析地恰为零，
#: 所以残差全部是浮点的。
AVERAGE_SAMPLES = 720

#: 迈克尔逊：动镜走5 mm → 光程差10 mm → 条纹级次15802.78。
#: 取这么大是**故意的**：相位精度随级次线性退化，级次小的构型量不出这件事。
MIRROR_DISPLACEMENT_M = 5.0e-3

#: FTS桥的最大光程差（单边），与`fts_instrument_line_shape`案例同量级。
FTS_MAX_OPD_M = 0.05

#: 相位精度系数，与`optics/interference.py`的
#: `PHASE_ACCURACY_RAD_PER_FRINGE_ORDER`同值。生成器**不import内核**，
#: 所以这里是一份拷贝——清单里有一条零容差布尔判据守着两者不许漂。
PHASE_ACCURACY_RAD_PER_FRINGE_ORDER = 2.1e-15

#: 二分精确第一亮纹的区间（以傍轴条纹间距为单位）与次数。
BISECTION_BRACKET: tuple[float, float] = (0.5, 1.5)
BISECTION_STEPS = 200


def to_decimal(value: Fraction) -> Decimal:
    """精确有理数 → 60位`Decimal`（`Decimal()`不吃`Fraction`）。"""

    return Decimal(value.numerator) / Decimal(value.denominator)


def decimal_cos(x: Decimal) -> Decimal:
    """`cos`的Taylor级数（60位）——**不调libm**，金标不继承平台实现。

    自变量已被调用方精确约化到``[0, 2 pi)``，级数在这个区间上约50项到地板。
    """

    term = Decimal(1)
    total = Decimal(1)
    square = x * x
    floor = Decimal(10) ** (-(REFERENCE_DIGITS + 5))
    for index in range(1, 400):
        term = -term * square / Decimal((2 * index - 1) * (2 * index))
        total += term
        if abs(term) < floor:
            return total
    raise SystemExit("decimal_cos未收敛，不落盘")


def reduced_phase(path_difference_m: float, wavelength_m: float) -> Decimal:
    """``dphi = 2 pi OPD / lambda``**精确约化**到``[0, 2 pi)``。

    两个float64本来就是精确有理数，所以``OPD / lambda``用`Fraction`是**精确**的；
    取小数部分之后再乘pi，条纹级次一点都不参与舍入。被验内核走
    ``(2 pi / lambda) * OPD``，相位误差随级次线性增长——案例量的就是这个差。
    """

    ratio = Fraction(path_difference_m) / Fraction(wavelength_m)
    fractional = ratio - (ratio.numerator // ratio.denominator)
    return 2 * PI * to_decimal(fractional)


def decimal_intensity(
    intensity_a: float, intensity_b: float, phase: Decimal, coherence: float = 1.0
) -> Decimal:
    """``I1 + I2 + 2 sqrt(I1 I2) |gamma| cos(dphi)``在60位上算。"""

    first, second = Decimal(intensity_a), Decimal(intensity_b)
    cross = 2 * (first * second).sqrt() * Decimal(coherence)
    return first + second + cross * decimal_cos(phase)


def decimal_extremes(
    intensity_a: float, intensity_b: float, coherence: float
) -> tuple[Decimal, Decimal]:
    """亮纹与暗纹（60位）。暗纹按**定义式**算，不走内核那条恒等变形。"""

    first, second = Decimal(intensity_a), Decimal(intensity_b)
    cross = 2 * (first * second).sqrt() * Decimal(coherence)
    return first + second + cross, first + second - cross


def decimal_visibility(
    intensity_a: float, intensity_b: float, coherence: float
) -> Decimal:
    """可见度走**极值相减**那条路（60位下无相消），内核走闭式——两条路。"""

    top, bottom = decimal_extremes(intensity_a, intensity_b, coherence)
    return (top - bottom) / (top + bottom)


def paraxial_path_difference(position_m: float) -> float:
    """傍轴光程差``d x / L``，在**精确有理数**上算再折成float64。"""

    return float(
        Fraction(SLIT_SEPARATION_M) * Fraction(position_m) / Fraction(SCREEN_DISTANCE_M)
    )


def decimal_exact_path_difference(position_m: float) -> Decimal:
    """两点源精确光程差，按**定义式**直接相减（60位下仍准到50位）。

    被验内核走代数恒等式``2 x d / (sqrt(A) + sqrt(B))``——两条不同的代数路。
    """

    position = Decimal(position_m)
    half = Decimal(SLIT_SEPARATION_M) / 2
    distance = Decimal(SCREEN_DISTANCE_M)
    far = ((position + half) ** 2 + distance**2).sqrt()
    near = ((position - half) ** 2 + distance**2).sqrt()
    return far - near


def bisect_first_bright_fringe(low_m: float, high_m: float) -> Decimal:
    """二分**生成器自己的**精确光程差，求``OPD = lambda``的位置。不用任何闭式。"""

    target = Decimal(WAVELENGTH_M)
    low, high = Decimal(low_m), Decimal(high_m)
    if (decimal_exact_path_difference(float(low)) - target) * (
        decimal_exact_path_difference(float(high)) - target
    ) > 0:
        raise SystemExit("二分区间两端同号，第一亮纹不在区间内，不落盘")
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        if decimal_exact_path_difference(float(middle)) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def main() -> int:
    if float(PI) != math.pi:
        raise SystemExit("60位pi折成float64后与math.pi不逐位相同，不落盘")

    # --- 第一层：双光束定律本身 ------------------------------------------
    phases = [Decimal(value) for value in PHASE_SAMPLES]
    intensity_samples = [
        float(decimal_intensity(INTENSITY_A, INTENSITY_B, phase)) for phase in phases
    ]
    partial_samples = [
        float(
            decimal_intensity(EQUAL_INTENSITY, EQUAL_INTENSITY, phase, PARTIAL_COHERENCE)
        )
        for phase in phases
    ]
    absent_samples = [
        float(decimal_intensity(INTENSITY_A, 0.0, phase)) for phase in phases
    ]
    incoherent_samples = [
        float(decimal_intensity(EQUAL_INTENSITY, EQUAL_INTENSITY, phase, 0.0))
        for phase in phases
    ]

    max_main, min_main = decimal_extremes(INTENSITY_A, INTENSITY_B, 1.0)
    max_partial, min_partial = decimal_extremes(
        EQUAL_INTENSITY, EQUAL_INTENSITY, PARTIAL_COHERENCE
    )
    _, min_near = decimal_extremes(NEAR_EQUAL_A, NEAR_EQUAL_B, 1.0)
    _, min_negative_case = decimal_extremes(NEGATIVE_NAIVE_A, NEGATIVE_NAIVE_B, 1.0)
    faint_visibility = decimal_visibility(FAINT_INTENSITY_A, FAINT_INTENSITY_B, 1.0)
    near_equal_amplification = float(
        Decimal(NEAR_EQUAL_B).sqrt()
        / abs(Decimal(NEAR_EQUAL_A).sqrt() - Decimal(NEAR_EQUAL_B).sqrt())
    )

    # --- 第二层：杨氏双缝 ------------------------------------------------
    spacing = (
        Fraction(WAVELENGTH_M) * Fraction(SCREEN_DISTANCE_M) / Fraction(SLIT_SEPARATION_M)
    )
    spacing_m = float(spacing)
    positions = [float(Fraction(value) * spacing) for value in POSITION_MULTIPLIERS]
    paraxial = [paraxial_path_difference(position) for position in positions]
    exact = [float(decimal_exact_path_difference(position)) for position in positions]
    screen_intensities = [
        float(
            decimal_intensity(
                INTENSITY_A,
                INTENSITY_B,
                reduced_phase(paraxial_path_difference(position), WAVELENGTH_M),
            )
        )
        for position in positions
    ]

    deviation_positions = [
        float(Fraction(value) * spacing) for value in DEVIATION_MULTIPLIERS
    ]
    deviation_measured: list[float] = []
    deviation_estimate: list[float] = []
    deviation_ratio: list[float] = []
    for position in deviation_positions:
        near = Decimal(paraxial_path_difference(position))
        measured = 1 - decimal_exact_path_difference(position) / near
        estimate = (Decimal(position) ** 2 + (Decimal(SLIT_SEPARATION_M) / 2) ** 2) / (
            2 * Decimal(SCREEN_DISTANCE_M) ** 2
        )
        deviation_measured.append(float(measured))
        deviation_estimate.append(float(estimate))
        deviation_ratio.append(float(estimate / measured))

    bracket = [value * spacing_m for value in BISECTION_BRACKET]
    first_bright = bisect_first_bright_fringe(bracket[0], bracket[1])
    fringe_position_shift = float(first_bright / Decimal(spacing_m) - 1)

    # --- 第三层：能量守恒 -------------------------------------------------
    energy_configurations = (
        (INTENSITY_A, INTENSITY_B, 1.0),
        (EQUAL_INTENSITY, EQUAL_INTENSITY, 1.0),
        (EQUAL_INTENSITY, EQUAL_INTENSITY, PARTIAL_COHERENCE),
        (INTENSITY_A, 0.0, 1.0),
    )
    phase_averages: list[float] = []
    for first, second, coherence in energy_configurations:
        total = Decimal(0)
        for index in range(AVERAGE_SAMPLES):
            phase = 2 * PI * Decimal(index) / Decimal(AVERAGE_SAMPLES)
            total += decimal_intensity(first, second, phase, coherence)
        phase_averages.append(float(total / Decimal(AVERAGE_SAMPLES)))
    declared_means = [float(Decimal(a) + Decimal(b)) for a, b, _ in energy_configurations]

    screen_total = Decimal(0)
    for index in range(AVERAGE_SAMPLES):
        position = spacing_m * index / AVERAGE_SAMPLES
        screen_total += decimal_intensity(
            INTENSITY_A,
            INTENSITY_B,
            reduced_phase(paraxial_path_difference(position), WAVELENGTH_M),
        )
    screen_average = float(screen_total / Decimal(AVERAGE_SAMPLES))

    # --- 第四层：迈克尔逊与FTS桥 -----------------------------------------
    michelson_opd = float(2 * Decimal(MIRROR_DISPLACEMENT_M))
    michelson_order = float(Fraction(michelson_opd) / Fraction(WAVELENGTH_M))
    michelson_phase = float(2 * PI * to_decimal(Fraction(michelson_opd) / Fraction(WAVELENGTH_M)))
    michelson_intensity = float(
        decimal_intensity(
            INTENSITY_A, INTENSITY_B, reduced_phase(michelson_opd, WAVELENGTH_M)
        )
    )

    first_zero = float(1 / (2 * Decimal(FTS_MAX_OPD_M)))
    shifted_wavelength = float(1 / (1 / Decimal(WAVELENGTH_M) + Decimal(first_zero)))
    swing = float(2 * Decimal(FTS_MAX_OPD_M))
    swing_order = float(Fraction(swing) / Fraction(WAVELENGTH_M))
    exact_slip = 2 * PI * (
        to_decimal(Fraction(swing) / Fraction(shifted_wavelength))
        - to_decimal(Fraction(swing) / Fraction(WAVELENGTH_M))
    )
    slip_residual = float(exact_slip - 2 * PI)

    # 解析界，案例容差逐字用它们算（**容差是算出来的**）。
    phase_bound = PHASE_ACCURACY_RAD_PER_FRINGE_ORDER * michelson_order
    intensity_bound = 2.0 * math.sqrt(INTENSITY_A * INTENSITY_B) * phase_bound
    slip_bound = 2.0 * PHASE_ACCURACY_RAD_PER_FRINGE_ORDER * swing_order
    screen_bound = (
        2.0
        * math.sqrt(INTENSITY_A * INTENSITY_B)
        * PHASE_ACCURACY_RAD_PER_FRINGE_ORDER
        * max(POSITION_MULTIPLIERS)
    )

    oracles = [
        {
            "id": "oracle:interference/two_beam_law",
            "inputs": {
                "kind": "two_beam_cosine_law_in_60_digit_decimal",
                "phase_samples_rad": list(PHASE_SAMPLES),
                "intensity_a": INTENSITY_A,
                "intensity_b": INTENSITY_B,
                "equal_intensity": EQUAL_INTENSITY,
                "partial_coherence": PARTIAL_COHERENCE,
                "faint_intensity_a": FAINT_INTENSITY_A,
                "faint_intensity_b": FAINT_INTENSITY_B,
                "near_equal_a": NEAR_EQUAL_A,
                "near_equal_b": NEAR_EQUAL_B,
                "negative_naive_a": NEGATIVE_NAIVE_A,
                "negative_naive_b": NEGATIVE_NAIVE_B,
                "reference_digits": REFERENCE_DIGITS,
            },
            "expected": {
                "intensity_samples": intensity_samples,
                "intensity_samples_partial_coherence": partial_samples,
                "max_intensity": float(max_main),
                "min_intensity": float(min_main),
                "max_intensity_partial": float(max_partial),
                "min_intensity_partial": float(min_partial),
                "visibility_unequal": float(
                    decimal_visibility(INTENSITY_A, INTENSITY_B, 1.0)
                ),
                "visibility_equal": float(
                    decimal_visibility(EQUAL_INTENSITY, EQUAL_INTENSITY, 1.0)
                ),
                "visibility_partial_equal": float(
                    decimal_visibility(EQUAL_INTENSITY, EQUAL_INTENSITY, PARTIAL_COHERENCE)
                ),
                "visibility_partial_unequal": float(
                    decimal_visibility(INTENSITY_A, INTENSITY_B, PARTIAL_COHERENCE)
                ),
                "visibility_faint_closed_form": float(faint_visibility),
                "visibility_faint_from_extremes": float(faint_visibility),
                "min_intensity_near_equal": float(min_near),
                "intensity_at_pi_near_equal": float(min_near),
                "min_intensity_negative_naive_case": float(min_negative_case),
                "intensity_with_second_beam_absent": absent_samples,
                "intensity_with_zero_coherence": incoherent_samples,
                "visibility_with_second_beam_absent": 0.0,
                "visibility_with_zero_coherence": 0.0,
            },
            "tolerances": {
                "intensity_samples": {
                    "abs": 3.0e-15,
                    "rel": 0.0,
                    "reason": "条纹级次为0（相位直接给），误差是交叉项的三次舍入"
                    "3u*2sqrt(I1I2) = 1.0e-15加上末次加法的u*|I| <= 7.5e-16，"
                    "界1.75e-15，取3e-15是它的1.7倍（实测恰为0）。"
                    "**判绝对不判相对**：dphi=pi处强度只有0.75而两个加数各3.75，"
                    "相对判据在那里量的是相消不是实现",
                },
                "intensity_samples_partial_coherence": {
                    "abs": 3.0e-15,
                    "rel": 0.0,
                    "reason": "同上，另加|gamma|一次乘法舍入。等强2.0、|gamma|=0.4，"
                    "交叉项1.6、峰值5.6，界3u*1.6 + u*5.6 = 1.15e-15。"
                    "**实测8.88e-16恰是ulp(5.6)——一位末位，一点也不多**；"
                    "第一版容差写1e-15只余1.1倍，是按感觉写的不是按界写的，已按界改成3e-15",
                },
                "max_intensity": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "三项皆正不相消。I1=3、I2=0.75时I1*I2=2.25、sqrt=1.5"
                    "**在float64上精确**，实测逐位相同；留1e-15（约4.5eps）"
                    "是给将来换构型的余量，不是给本构型放水",
                },
                "min_intensity": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "内核走不相消的等价式(sqrt(I1)-sqrt(I2))^2 + "
                    "2sqrt(I1I2)(1-|gamma|)。本构型真值0.75，内核实测0.7499999999999999"
                    "（低1 ulp——**恒等变形在I1、I2相差远时反而多一次舍入**，"
                    "如实记在这里）。1e-15容得下这1 ulp；该式的价值在近等强构型，"
                    "见min_intensity_near_equal",
                },
                "max_intensity_partial": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "等强2.0、|gamma|=0.4：4 + 4*0.4 = 5.6，两次舍入",
                },
                "min_intensity_partial": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "等强时(sqrt(I1)-sqrt(I2))^2恰为0，剩2sqrt(I1I2)(1-|gamma|)"
                    "= 4*0.6 = 2.4。1-0.4按Sterbenz引理精确",
                },
                "visibility_unequal": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "闭式2sqrt(I1I2)/(I1+I2) = 3.0/3.75 = 0.8，一次除法",
                },
                "visibility_equal": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差**：等强完全相干时2sqrt(I*I)与I+I逐位相同，"
                    "商精确为1.0。这是一条能写成等号的物理主张，不留余量",
                },
                "visibility_partial_equal": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "等强时V = |gamma| = 0.4——**这就是|gamma|的操作定义**，"
                    "也是相干性那半边的判据：它红了说明|gamma|没有按定义进公式",
                },
                "visibility_partial_unequal": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "V = 0.8 * 0.4 = 0.32：可见度是**两个因子之积**"
                    "（强度失配与相干度）。与上一条合起来才把|gamma|与强度比分开——"
                    "只验等强时，把|gamma|错当成整个可见度的实现照样绿",
                },
                "visibility_faint_closed_form": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "I2/I1 = 1e-12时V = 2e-6。闭式路径无相消："
                    "sqrt(I1I2)与I1+I2都不是小差，一次除法",
                },
                "visibility_faint_from_extremes": {
                    "abs": 0.0,
                    "rel": 1.0e-9,
                    "reason": "**同一个量的另一条路**：(Imax-Imin)/(Imax+Imin)。"
                    "Imax与Imin的相对差只有2V = 4e-6，相减的**放大因子恰是1/V = 5e5**，"
                    "解析界2u/V = 1.1e-10，取1e-9是它的9倍。"
                    "**两条路容差差六个数量级，差的就是那个放大因子**——"
                    "所以fringe_visibility走闭式不走极值相减",
                },
                "min_intensity_near_equal": {
                    "abs": 0.0,
                    "rel": 5.0e-9,
                    "reason": "I1=1、I2=1+1e-7的暗纹。**这个量本身病态**："
                    f"放大因子sqrt(I2)/|sqrt(I1)-sqrt(I2)| = {near_equal_amplification:.4e}，"
                    "sqrt(I2)自己的半ulp经它放大后相对误差界2*amp*u = 4.4e-9。"
                    "内核走不相消的等价式，实测1.15e-9（界的1/3.8）。"
                    "**容差是从放大因子算出来的**：换一个更近的构型它会更大",
                },
                "intensity_at_pi_near_equal": {
                    "abs": 5.0e-16,
                    "rel": 0.0,
                    "reason": "**同一个物理量的通用路径**：two_beam_intensity(dphi=pi)走"
                    "I1+I2-2sqrt(I1I2)，两个约等于2的数相减。绝对误差界2u(I1+I2)=4.4e-16，"
                    "实测1.65e-16；但真值只有2.5e-15，**相对误差因此到6.6%**。"
                    "与上一条对照：同一个数，一条判相对5e-9、一条只能判绝对5e-16——"
                    "差的不是精度是写法。这条**不许改成相对判据**，改了就是在量相消",
                },
                "min_intensity_negative_naive_case": {
                    "abs": 0.0,
                    "rel": 2.0e-4,
                    "reason": "这一对I1、I2上朴素式I1+I2-2sqrt(I1I2)**返回-1.78e-15**"
                    "（负强度——物理上不存在的东西，不是精度问题）。真值只有3.79e-23，"
                    "**放大因子sqrt(I2)/|sqrt(I1)-sqrt(I2)| = 4.3e11**，"
                    "连内核那条不相消的路都只剩4位有效数字（界2*amp*u = 9.5e-5，"
                    "实测6.55e-5）。取2e-4是界的2.1倍。"
                    "**这条判相对**：判绝对的话5e-16比真值大七个数量级，"
                    "那样的判据只是在说“结果接近0”，一点判别力都没有",
                },
                "intensity_with_second_beam_absent": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**退化极限，零容差**：I2=0时sqrt(I1*0)在float64上精确为0，"
                    "交叉项精确为0，I(dphi)对一切相位精确等于I1。"
                    "没有第二束就没有条纹——这条不许有余量",
                },
                "intensity_with_zero_coherence": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**退化极限，零容差**：|gamma|=0时交叉项被乘成精确的0，"
                    "两束按强度直接相加（非相干叠加）。"
                    "与上一条是两个不同的退化机制落到同一个结论，各验各的",
                },
                "visibility_with_second_beam_absent": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "零容差：I2=0 → V精确为0。V>0而I2=0自相矛盾，不留余量",
                },
                "visibility_with_zero_coherence": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "零容差：|gamma|=0 → V精确为0。非相干光不产生条纹",
                },
            },
        },
        {
            "id": "oracle:interference/young_double_slit",
            "inputs": {
                "kind": "young_fringes_against_exact_two_point_source_geometry",
                "wavelength_m": WAVELENGTH_M,
                "slit_separation_m": SLIT_SEPARATION_M,
                "screen_distance_m": SCREEN_DISTANCE_M,
                "position_multipliers": list(POSITION_MULTIPLIERS),
                "screen_positions_m": positions,
                "deviation_multipliers": list(DEVIATION_MULTIPLIERS),
                "deviation_positions_m": deviation_positions,
                "intensity_a": INTENSITY_A,
                "intensity_b": INTENSITY_B,
                "bisection_bracket_m": bracket,
                "bisection_steps": BISECTION_STEPS,
            },
            "expected": {
                "fringe_spacing_m": spacing_m,
                "paraxial_path_difference_m": paraxial,
                "exact_path_difference_m": exact,
                "screen_intensity_samples": screen_intensities,
                "paraxial_relative_deviation_measured": deviation_measured,
                "paraxial_relative_deviation_estimate": deviation_estimate,
                "deviation_estimate_over_measured": deviation_ratio,
                "first_bright_fringe_relative_shift": fringe_position_shift,
            },
            "tolerances": {
                "fringe_spacing_m": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "dx = lambda L / d，一乘一除两次舍入（约2.2e-16）。"
                    "参考值在**精确有理数**上算再折成float64。"
                    "L与d写颠倒得到1.3e-9 m而不是3.0e-3 m，这条当场红6个数量级",
                },
                "paraxial_path_difference_m": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "OPD = d x / L，两次舍入。参考走精确有理数。"
                    "x=0处两侧都精确为0，rel判据在零值上自动退化成等号——"
                    "**中央条纹的光程差恰为零**，正该是等号",
                },
                "exact_path_difference_m": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "**两条代数路对拍**：内核走2xd/(sqrt(A)+sqrt(B))（不相消），"
                    "参考走60位Decimal上的定义式sqrt(A)-sqrt(B)。"
                    "内核那条只有四次舍入，1e-15约4.5eps。"
                    "**定义式在float64上直接相减会丢约6位**（L=1.2 m而OPD只有6.33e-7 m，"
                    "实测两条路差8.3e-11相对；越近轴丢得越多），"
                    "所以内核不许那么写——这条判据同时守着那件事",
                },
                "screen_intensity_samples": {
                    "abs": 5.0e-14,
                    "rel": 0.0,
                    "reason": "屏上位置→傍轴光程差→相位→强度四步。最远取样在第5级，"
                    f"相位误差界2.1e-15*5 = 1.05e-14 rad，乘交叉项3.0得{screen_bound:.2e}。"
                    "取5e-14≈解析界。**这一条同时是“相位差里漏了2pi/lambda”的捕手**："
                    "漏了之后暗纹位置上会算出亮纹强度，差3个数量级",
                },
                "paraxial_relative_deviation_measured": {
                    "abs": 0.0,
                    "rel": 2.0e-9,
                    "reason": "1 - OPD_exact/OPD_paraxial：两个相对差只有dev的数相减，"
                    "**放大因子就是1/dev**。四个取样点的dev是2.06e-7 / 3.21e-6 / "
                    "8.01e-5 / 5.09e-3，**最坏点是最靠里的x=0.25dx**（1/dev = 4.86e6，"
                    "解析界3u/dev = 1.62e-9，实测2.37e-10）。**同一条容差由最坏点定**，"
                    "取2e-9是界的1.2倍。注意最坏点不是最外面那个而是最里面那个——"
                    "越近轴、傍轴近似越准，这个比值就越难算准",
                },
                "paraxial_relative_deviation_estimate": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "闭式(x^2 + d^2/4)/(2 L^2)，四次舍入。参考走60位Decimal",
                },
                "deviation_estimate_over_measured": {
                    "abs": 0.0,
                    "rel": 2.0e-9,
                    "reason": "**这一条才是“适用条件”本身**：首阶估计除以实测偏差。"
                    "四点实测1.0000003 / 1.0000048 / 1.0001201 / 1.0076819——"
                    "**首阶展开在偏差5.09e-3（x=40dx、x/L约0.10）时自己差了7.7e-3，"
                    "这就是它的适用边界，写成一个能被断言的数而不是一句“近轴时成立”**。"
                    "容差随measured那条（同一个1/dev放大因子，最坏点同为x=0.25dx）",
                },
                "first_bright_fringe_relative_shift": {
                    "abs": 0.0,
                    "rel": 5.0e-10,
                    "reason": "**二分精确几何求出的第一亮纹位置**相对傍轴dx的偏移，"
                    "参考由生成器在60位上二分它自己的精确光程差得到（200次）。"
                    "测量侧在float64上二分内核的精确光程差；偏移量本身约+3.209e-6"
                    "（精确条纹比傍轴预测**更远**，与偏差同号）。两项误差："
                    "①float64二分的位置分辨率1 ulp(3e-3) = 4.3e-19 m，"
                    "折成相对偏移4.4e-11；②精确光程差自身2.2e-16的相对误差经斜率"
                    "dOPD/dx = 2.08e-4折算成6.7e-19 m，折成6.9e-11。界约1.2e-10，"
                    "取5e-10是它的4.2倍（实测4.7e-11）。"
                    "**这一条是条纹间距闭式的独立佐证**：闭式说dx，"
                    "精确几何说dx(1+3.209e-6)，两者的差正好是傍轴偏差本身",
                },
            },
        },
        {
            "id": "oracle:interference/energy_conservation",
            "inputs": {
                "kind": "fringe_average_returns_the_incoherent_sum",
                "sample_count": AVERAGE_SAMPLES,
                "configurations": [list(config) for config in energy_configurations],
                "fringe_spacing_m": spacing_m,
                "slit_separation_m": SLIT_SEPARATION_M,
                "screen_distance_m": SCREEN_DISTANCE_M,
                "wavelength_m": WAVELENGTH_M,
                "intensity_a": INTENSITY_A,
                "intensity_b": INTENSITY_B,
            },
            "expected": {
                "phase_average_intensity": phase_averages,
                "declared_mean_intensity": declared_means,
                "screen_average_over_one_fringe": screen_average,
            },
            "tolerances": {
                "phase_average_intensity": {
                    "abs": 3.0e-15,
                    "rel": 0.0,
                    "reason": "**干涉重新分配能量，不创造能量**：余弦在整周期上的平均"
                    "解析地恰为零，所以条纹的相位平均必须回到I1+I2，"
                    "**与|gamma|、与条纹间距、与光程差机制全都无关**——"
                    "这是一条独立于任何条纹公式的自洽门。720点整周期，"
                    "残差是每点的舍入被平均之后剩下的，最坏情形（各点同号）就是单点的界"
                    "1.75e-15，取3e-15是它的1.7倍。**四个构型实测残差都恰为0**——"
                    "余弦在整周期上的抵消在float64里也是精确的。"
                    "四个构型：不等强/等强/部分相干/单束，各一条",
                },
                "declared_mean_intensity": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "零容差：two_beam_mean_intensity就是I1+I2一次加法，"
                    "必须与参考逐位相同。它是上一条的**申报值**——"
                    "平均回到的那个数不许是另算的",
                },
                "screen_average_over_one_fringe": {
                    "abs": 5.0e-14,
                    "rel": 0.0,
                    "reason": "**空间平均**：屏上一个条纹间距内720点等间距取样的平均。"
                    "与相位平均那条不同，**这一条是耦合的**——它同时用到条纹间距"
                    "与整条相位链，条纹间距算错就不再是整周期平均、余弦不再抵消。"
                    "如实标注它不独立；独立的那条是phase_average_intensity。"
                    "容差按相位链的舍入取（同screen_intensity_samples）",
                },
            },
        },
        {
            "id": "oracle:interference/michelson_and_fts_bridge",
            "inputs": {
                "kind": "michelson_high_fringe_order_and_the_fts_resolution_bridge",
                "mirror_displacement_m": MIRROR_DISPLACEMENT_M,
                "wavelength_m": WAVELENGTH_M,
                "fts_max_opd_m": FTS_MAX_OPD_M,
                "intensity_a": INTENSITY_A,
                "intensity_b": INTENSITY_B,
                "phase_accuracy_rad_per_fringe_order": PHASE_ACCURACY_RAD_PER_FRINGE_ORDER,
            },
            "expected": {
                "michelson_path_difference_m": michelson_opd,
                "michelson_fringe_order": michelson_order,
                "michelson_phase_rad": michelson_phase,
                "michelson_intensity": michelson_intensity,
                "ils_first_zero_per_m": first_zero,
                "shifted_wavelength_m": shifted_wavelength,
                "fringe_slip_over_full_scan_rad": float(2 * PI),
                "declared_phase_accuracy_matches_manifest": True,
            },
            "tolerances": {
                "michelson_path_difference_m": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "零容差：动镜位移乘2在二进制上是精确的指数搬移。"
                    "**这个2是“光去一趟回一趟”**，与fts.DOUBLE_SIDED_OPD_FACTOR那个"
                    "“干涉图录在-L..+L上”的2不是同一件事，数值相同纯属巧合。"
                    "写成1就是2倍的条纹级次错，这条当场红",
                },
                "michelson_fringe_order": {
                    "abs": 0.0,
                    "rel": 2.3e-16,
                    "reason": "N = OPD/lambda一次除法、正确舍入，1 ulp约2.2e-16。"
                    "N = 15802.78——**它同时是下面两条的相消放大因子**",
                },
                "michelson_phase_rad": {
                    "abs": 3.4e-11,
                    "rel": 0.0,
                    "reason": f"dphi = 2 pi N约{michelson_phase:.0f} rad。内核走"
                    "(2pi/lambda)*OPD，三次舍入使相对误差3u，绝对误差因此约"
                    f"3u*2pi*N = 2.1e-15*N = {phase_bound:.3e} rad——"
                    "**相位精度随条纹级次线性退化，放大因子就是级次本身**。"
                    "参考侧用Fraction把OPD/lambda精确约化到[0,1)再乘2pi，一点不损失。"
                    "取3.4e-11≈解析界。**判绝对**：相对判据在这里量的是级次不是实现",
                },
                "michelson_intensity": {
                    "abs": 1.0e-10,
                    "rel": 0.0,
                    "reason": f"相位误差经交叉项放大：2sqrt(I1I2)*{phase_bound:.3e} = "
                    f"{intensity_bound:.3e}。取1e-10≈解析界，实测1.15e-11（界的1/8.6）。"
                    "**这不是“实现不够好”**：这个量在float64上就只有这么准，"
                    "级次再升一个数量级它还会更差——所以级次是公开的（fringe_order）",
                },
                "ils_first_zero_per_m": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "取自本子包已有的fts.unapodised_first_zero_per_m："
                    "1/(2L) = 10 per m。**本案例不另写一份**——"
                    "两份等价实现是末位漂移的温床（本仓有过k*(d_a·d_b)那条教训）",
                },
                "shifted_wavelength_m": {
                    "abs": 0.0,
                    "rel": 1.0e-15,
                    "reason": "lambda2 = 1/(1/lambda1 + dsigma0)，两次取倒一次加法。"
                    "**这里的波数必须是谱学波数sigma = 1/lambda不是角波数**，"
                    "差2pi则整条桥塌掉——两个换算函数并存就是为这一刻",
                },
                "fringe_slip_over_full_scan_rad": {
                    "abs": 8.0e-10,
                    "rel": 0.0,
                    "reason": "**与FTS的桥**：ILS首零1/(2L)按定义是分辨极限，"
                    "它的物理内容是——波数差一个1/(2L)的两束光，在整个扫描程2L上"
                    "相对相位**恰好滑过一个整条纹2pi**。期望值因此就写2pi。"
                    f"两项偏差：①级次{swing_order:.0f}上两个大相位相减，解析界"
                    f"2*2.1e-15*N = {slip_bound:.3e}（实测2.7e-13——两次舍入高度相关，"
                    "**容差按界取不按运气取**）；②sigma→lambda→sigma的往返把dsigma"
                    f"挪了1.7e-12相对，使精确值比2pi低{abs(slip_residual):.3e}。"
                    "取8e-10覆盖①的界。**本条不引入任何FFT**：0031把变换层声明为"
                    "下一块，这里只用两条余弦的相位差",
                },
                "declared_phase_accuracy_matches_manifest": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "布尔零容差：内核常量PHASE_ACCURACY_RAD_PER_FRINGE_ORDER"
                    "必须等于清单inputs里那个数。生成器不import内核，"
                    "所以那个系数在两边各存了一份——**这条门就是不许它们漂**。"
                    "上面三条容差全是拿它乘条纹级次算出来的，它漂了容差就成了空话",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/two_beam_interference",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/two_beam_interference/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes\n"
        f"  spacing            {spacing_m!r} m\n"
        f"  michelson order    {michelson_order:.6f}\n"
        f"  phase bound        {phase_bound:.4e} rad\n"
        f"  intensity bound    {intensity_bound:.4e}\n"
        f"  screen bound       {screen_bound:.4e}\n"
        f"  slip bound         {slip_bound:.4e}, residual {slip_residual:.4e}\n"
        f"  near-equal amp     {near_equal_amplification:.4e}\n"
        f"  first bright shift {fringe_position_shift:.6e}\n"
        f"  deviation ratios   {[format(value, '.7f') for value in deviation_ratio]}\n"
        f"  deviations         {[format(value, '.6e') for value in deviation_measured]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
