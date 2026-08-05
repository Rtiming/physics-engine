"""双光束干涉的闭式解：一条余弦定律，两种光程差机制。

一句话：两束光叠加，强度不是`I1 + I2`而是

    I(dphi) = I1 + I2 + 2 sqrt(I1 I2) |gamma| cos(dphi)

那个多出来的交叉项就是干涉。**它把能量重新分配，不创造能量**——
条纹的空间平均仍然恰好是`I1 + I2`（`two_beam_mean_intensity`），
这条守恒是独立于任何条纹公式的自洽门，本模块把它写成可断言的量。

## 本模块给的四样

1. **双光束强度与它的两个极值**：`two_beam_intensity`、
   `two_beam_max_intensity`、`two_beam_min_intensity`（**极小值走
   不相消的等价式**，见下"相消"一节）；
2. **可见度**`V = (Imax - Imin) / (Imax + Imin) = 2 sqrt(I1 I2) |gamma| / (I1 + I2)`
   （`fringe_visibility`，走**闭式**不走极值相减）；
3. **杨氏双缝**：傍轴条纹间距``dx = lambda L / d``、傍轴光程差``d x / L``、
   **精确光程差**（两点源的真几何），以及傍轴近似的**相对偏差闭式**
   ——"适用条件"在本仓不是一句文字，是一个能算出来的数；
4. **迈克尔逊**：动镜位移`d`给出光程差`2 d`。它与双缝的价值正在于
   **用完全不同的光程差机制产生同一条`cos(dphi)`定律**——
   两条几何、一条物理。

## 相干性：做`|gamma|`，不做`gamma`

`coherence_modulus`是**复相干度的模**`|gamma|`（0到1，越界失败关闭——
`|gamma| > 1`违反Cauchy-Schwarz，不是"参数偏大"是"这不是一个相干度"）。
等强度时`V = |gamma|`，这是`|gamma|`的操作定义。

**不做的是`gamma`的相位**`arg(gamma)`（它把整个条纹图样平移）。
理由不是"来不及"：`arg(gamma)`要么由光源谱经Wiener-Khinchin算出来
（那是傅里叶变换，0031已把变换层声明为下一块），要么由消费方直接测量后传进来
——今天没有任何消费方需要它。按本仓三前提第二条，不为想象中的消费方预支。
准单色、谱型对称的光源上`arg(gamma) = 0`，本模块覆盖的正是这一档。
**触发条件**：某个消费方要算非对称谱或要用相干度做定位时重开。

## 相消：本模块两处，各有一个可以算出来的放大因子

1. **`dphi`接近`pi`且`I1 ≈ I2`时的极小值**。`I1 + I2 - 2 sqrt(I1 I2)`
   是两个几乎相等的正数相减，真值``(sqrt(I1) - sqrt(I2))^2``小到什么程度，
   相对误差就烂到什么程度。放大因子是``sqrt(I2) / |sqrt(I1) - sqrt(I2)|``：
   `I1 = 1`、`I2 = 1 + 1e-7`时它是**2.0e7**，朴素写法的相对误差因此到18%量级
   （绝对误差只有4e-16——**所以这个量必须判绝对或换写法，判相对毫无意义**，
   与`J1`在零点附近那条同理，0031第四节）。
   `two_beam_min_intensity`因此走恒等变形

       Imin = (sqrt(I1) - sqrt(I2))^2 + 2 sqrt(I1 I2) (1 - |gamma|)

   **两项都非负，一次相消都没有**（且`|gamma| >= 0.5`时`1 - |gamma|`
   按Sterbenz引理精确）。通用的`two_beam_intensity(dphi=pi)`仍然会相消，
   这是它作为通用式的代价，写在案例的已知失效清单里而不是藏起来。
2. **相位随条纹级次线性变差**。``dphi = 2 pi N``，`N = OPD / lambda`是条纹级次；
   float64上`dphi`的绝对误差约``PHASE_ACCURACY_RAD_PER_FRINGE_ORDER * N``。
   迈克尔逊动镜走5mm（OPD 10mm、HeNe）时`N = 15803`，相位误差3.3e-11 rad——
   **不是实现差，是这个量本身在float64上就只有这么准**。案例的容差逐字用它算。

## 单位边界

* **相位是弧度**。参数名`phase_difference_rad`就是那道边界。
  **调用方传进来一个度数，本模块无法察觉**——这是本模块唯一一个
  没有任何门能抓的错法，如实写在这里而不是假装不存在；
* **波长→相位只经`angular_wavenumber_rad_per_m`**（`diffraction.py`那一个）。
  本模块不自己写`2 * math.pi / wavelength`：谱学波数与角波数差一个2pi
  且不报错（0031第3.3节），那道边界在本子包里只许有**一处**实现；
* **迈克尔逊的2与FTS的2不是同一个2**。这里的
  `MICHELSON_OPD_PER_MIRROR_DISPLACEMENT`是"光去一趟回一趟"，
  `fts.DOUBLE_SIDED_OPD_FACTOR`是"干涉图录在`-L..+L`上"。
  两者今天数值相同**纯属巧合**，合并成一个常量就等于把两条物理并成一条。

## 明确不做的（负空间声明，Drake形制）

* **不做场传播、不做FFT**。二维复数场与角谱/菲涅耳传播仍是0031声明的"下一块"；
  本模块只有闭式代数，一个数组都没有；
* **不做多光束**。法布里-珀罗的Airy函数（`1/(1 + F sin^2)`）不在这里，
  它是多光束不是双光束；
* **不做`arg(gamma)`、不做时间相干长度、不做谱型**（见上"相干性"一节）；
* **不做偏振**。两束正交偏振不干涉（Fresnel-Arago），本模块默认同偏振态且不校验
  ——`|gamma|`可以吸收这一项，但那是消费方的申报不是本模块的计算；
* **不做等厚/等倾干涉**。薄膜的半波损失要折射率与入射角，是另一块；
  第二个构型取迈克尔逊，因为它与本子包已有的FTS共用同一个光程差轴（见案例）；
* **不做衍射**。双缝的**包络**（单缝衍射因子）不在这里，本模块只给干涉因子。
  真实双缝图样是两者之积——这条边界写在案例的"不是什么"里。
"""

from __future__ import annotations

import math

from physics_engine.optics.diffraction import angular_wavenumber_rad_per_m
from physics_engine.optics.errors import OpticsError

#: 完全相干`|gamma| = 1`。写成常量而不是字面量1.0：默认值是一条**物理声明**
#: （"本次计算假定两束完全相干"），不是一个凑数的缺省。
FULL_COHERENCE: float = 1.0

#: 迈克尔逊动镜位移`d`→光程差`2 d`：光去一趟、回一趟。
#: **与`fts.DOUBLE_SIDED_OPD_FACTOR`不是同一个2**（那个是干涉图录在`-L..+L`上）。
#: 两者今天数值相同纯属巧合；合并常量等于把两条物理并成一条，正是0031第3.3节
#: 那张表要防的事。
MICHELSON_OPD_PER_MIRROR_DISPLACEMENT: float = 2.0

#: 相位绝对误差随条纹级次`N = OPD / lambda`线性增长的系数（弧度/级）。
#: 由``dphi = fl(2 pi / lambda) * OPD``的三次舍入得到：``3 u * 2 pi = 2.09e-15``
#: （`u = 2^-53 = 1.11e-16`是float64的半ulp）。
#: **案例容差逐字用它乘以N**，所以`tests/test_optics_interference.py`有一条门
#: 两个方向都断：它必须是上界，也不许比实测松一个数量级。
PHASE_ACCURACY_RAD_PER_FRINGE_ORDER: float = 2.1e-15


def _require_intensity(value: float, name: str) -> float:
    """强度必须是有限非负数。负强度不是"参数偏小"，是没有这个东西。"""

    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise OpticsError(f"{name}必须是有限非负数（强度）：{value!r}")
    return number


def _require_coherence(value: float) -> float:
    """复相干度的模必须落在`[0, 1]`。

    上界不是工程约定是Cauchy-Schwarz：`|gamma| > 1`意味着互相干函数超过了
    两束各自的自相干，那不是"相干度偏大"，是这个数不是一个相干度。
    """

    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise OpticsError(
            f"复相干度的模必须落在[0, 1]（|gamma| > 1违反Cauchy-Schwarz）：{value!r}"
        )
    return number


def _require_phase(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise OpticsError(f"相位差必须有限（**弧度**）：{value!r}")
    return number


def _require_positive_length(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise OpticsError(f"{name}必须是有限正数（米）：{value!r}")
    return number


def _require_finite_length(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise OpticsError(f"{name}必须有限（米）：{value!r}")
    return number


def two_beam_intensity(
    *,
    intensity_a: float,
    intensity_b: float,
    phase_difference_rad: float,
    coherence_modulus: float = FULL_COHERENCE,
) -> float:
    """双光束叠加强度``I1 + I2 + 2 sqrt(I1 I2) |gamma| cos(dphi)``。

    `phase_difference_rad`是**弧度**。交叉项里的``2 sqrt(I1 I2)``是振幅相加的
    结果（``|A1 + A2|^2``的交叉项），**不是**``2 I1 I2``——两者在
    `I1 = I2 = 1`时数值相同，所以只用等强度归一化构型验这个模块的实现，
    抓不住把根号写掉这个错（案例因此必须带不等强度构型）。

    `dphi`接近`pi`且`I1 ≈ I2`时本式相消，见模块docstring"相消"第1条；
    要极小值本身请用`two_beam_min_intensity`。
    """

    first = _require_intensity(intensity_a, "intensity_a")
    second = _require_intensity(intensity_b, "intensity_b")
    coherence = _require_coherence(coherence_modulus)
    phase = _require_phase(phase_difference_rad)
    cross = 2.0 * math.sqrt(first * second) * coherence
    return first + second + cross * math.cos(phase)


def two_beam_mean_intensity(*, intensity_a: float, intensity_b: float) -> float:
    """条纹的空间平均强度``I1 + I2``——**干涉重新分配能量，不创造能量**。

    余弦项在整周期上的平均恰为零，所以平均强度与`|gamma|`、与条纹间距、
    与光程差机制**全都无关**。这条因此是一条独立于任何条纹公式的自洽门：
    条纹间距算错、相位少乘一个2pi，条纹图样的数值平均都会离开这个值。
    """

    return _require_intensity(intensity_a, "intensity_a") + _require_intensity(
        intensity_b, "intensity_b"
    )


def two_beam_max_intensity(
    *,
    intensity_a: float,
    intensity_b: float,
    coherence_modulus: float = FULL_COHERENCE,
) -> float:
    """亮纹``I1 + I2 + 2 sqrt(I1 I2) |gamma|``（`dphi = 0`）。三项皆正，不相消。"""

    first = _require_intensity(intensity_a, "intensity_a")
    second = _require_intensity(intensity_b, "intensity_b")
    coherence = _require_coherence(coherence_modulus)
    return first + second + 2.0 * math.sqrt(first * second) * coherence


def two_beam_min_intensity(
    *,
    intensity_a: float,
    intensity_b: float,
    coherence_modulus: float = FULL_COHERENCE,
) -> float:
    """暗纹``I1 + I2 - 2 sqrt(I1 I2) |gamma|``（`dphi = pi`），**走不相消的等价式**

        Imin = (sqrt(I1) - sqrt(I2))^2 + 2 sqrt(I1 I2) (1 - |gamma|)

    两项都非负，一次相消都没有；`|gamma| >= 0.5`时``1 - |gamma|``按Sterbenz引理
    还是精确的。朴素写法在`I1 ≈ I2`时的相对误差可以到10%量级
    （放大因子``sqrt(I2) / |sqrt(I1) - sqrt(I2)|``，模块docstring"相消"第1条），
    **并且可能返回一个负强度**——那不是精度问题，那是算出了一个不存在的东西。
    本式的返回值永远`>= 0`。
    """

    first = _require_intensity(intensity_a, "intensity_a")
    second = _require_intensity(intensity_b, "intensity_b")
    coherence = _require_coherence(coherence_modulus)
    gap = math.sqrt(first) - math.sqrt(second)
    return gap * gap + 2.0 * math.sqrt(first * second) * (1.0 - coherence)


def fringe_visibility(
    *,
    intensity_a: float,
    intensity_b: float,
    coherence_modulus: float = FULL_COHERENCE,
) -> float:
    """条纹可见度``V = (Imax - Imin)/(Imax + Imin) = 2 sqrt(I1 I2) |gamma| / (I1 + I2)``。

    **走右边的闭式，不走左边的极值相减**：`I1 >> I2`时`Imax ≈ Imin`，
    极值相减的放大因子是``1 / V``（`V = 2e-6`时丢6位有效数字）。
    左边那条路在案例里作为**独立对拍**出现，容差按`1/V`算出来——
    两条路的差不是噪声，是那个放大因子本身。

    等强度时`V = |gamma|`：这就是`|gamma|`的操作定义。
    两束强度之一为零时`V = 0`（没有第二束就没有条纹）。
    """

    first = _require_intensity(intensity_a, "intensity_a")
    second = _require_intensity(intensity_b, "intensity_b")
    coherence = _require_coherence(coherence_modulus)
    total = first + second
    if total == 0.0:
        raise OpticsError("两束强度都为零时可见度没有定义（0/0），拒答")
    return 2.0 * math.sqrt(first * second) * coherence / total


def fringe_order(*, path_difference_m: float, wavelength_m: float) -> float:
    """条纹级次``N = OPD / lambda``（无量纲，可正可负）。

    它同时是**相位精度的放大因子**：`dphi = 2 pi N`的绝对误差约
    ``PHASE_ACCURACY_RAD_PER_FRINGE_ORDER * |N|``。案例容差逐字用它算，
    所以这个量是公开的——"这次算得有多准"必须能被调用方问出来。
    """

    difference = _require_finite_length(path_difference_m, "path_difference_m")
    return difference / _require_positive_length(wavelength_m, "wavelength_m")


def phase_difference_rad(*, path_difference_m: float, wavelength_m: float) -> float:
    """光程差（米）→相位差``dphi = 2 pi OPD / lambda``（弧度）。

    实现取``angular_wavenumber_rad_per_m(lambda) * OPD``——**先算角波数再乘光程差**，
    不是``2 pi * (OPD / lambda)``。两者代数相同、末位不同，本仓有过
    ``k*(d_a·d_b)``写成``(k*d_a)*d_b``的末位漂移教训，所以运算次序在这里被钉死。
    2pi的唯一来源是`diffraction.py`的角波数换算，本模块不自己写除法。

    精度随条纹级次线性退化，见`fringe_order`。
    """

    difference = _require_finite_length(path_difference_m, "path_difference_m")
    return angular_wavenumber_rad_per_m(wavelength_m) * difference


def young_fringe_spacing_m(
    *, wavelength_m: float, slit_separation_m: float, screen_distance_m: float
) -> float:
    """杨氏双缝的**傍轴**条纹间距``dx = lambda L / d``。

    `d`是缝间距、`L`是缝到屏的距离——**分子是L分母是d**。两者颠倒后量纲仍然
    是米、数值仍然"看起来像个条纹间距"，不报任何错：这就是本函数用三个
    带单位角色的关键字参数而不是位置参数的理由。

    傍轴：只在``(x^2 + d^2/4) / (2 L^2) << 1``的近轴区成立，且**条纹严格等距
    也只在这一近似下成立**（精确几何下条纹随x外移而变疏）。这个偏差有闭式，
    见`young_paraxial_relative_deviation`——"适用条件"在本仓是一个能算的数。
    """

    wavelength = _require_positive_length(wavelength_m, "wavelength_m")
    separation = _require_positive_length(slit_separation_m, "slit_separation_m")
    distance = _require_positive_length(screen_distance_m, "screen_distance_m")
    return wavelength * distance / separation


def young_paraxial_path_difference_m(
    *, screen_position_m: float, slit_separation_m: float, screen_distance_m: float
) -> float:
    """杨氏双缝的**傍轴**光程差``OPD = d x / L``。

    `x`是屏上离中央亮纹的横向位置（米，可正可负）。适用条件同
    `young_fringe_spacing_m`；精确值见`young_exact_path_difference_m`。
    """

    position = _require_finite_length(screen_position_m, "screen_position_m")
    separation = _require_positive_length(slit_separation_m, "slit_separation_m")
    distance = _require_positive_length(screen_distance_m, "screen_distance_m")
    return separation * position / distance


def young_exact_path_difference_m(
    *, screen_position_m: float, slit_separation_m: float, screen_distance_m: float
) -> float:
    """两点源到屏上一点的**精确**光程差，**不相消**地算。

    定义式是两个几乎相等的平方根之差

        OPD = sqrt((x + d/2)^2 + L^2) - sqrt((x - d/2)^2 + L^2)

    近轴时两个根号都约等于`L`，直接相减把有效位丢在减法里：`L = 1.2 m`而
    `OPD`只有6.33e-7 m时丢约6位（实测两条路差8.3e-11相对），越近轴丢得越多。
    本实现走恒等变形

        OPD = 2 x d / (sqrt(A) + sqrt(B))

    分子``A - B = 2 x d``是**精确**的代数化简（不是近似），分母两项同号相加，
    整条路一次相消都没有。这与`two_beam_min_intensity`是同一手法。
    """

    position = _require_finite_length(screen_position_m, "screen_position_m")
    separation = _require_positive_length(slit_separation_m, "slit_separation_m")
    distance = _require_positive_length(screen_distance_m, "screen_distance_m")
    half = 0.5 * separation
    near = position - half
    far = position + half
    root_far = math.sqrt(far * far + distance * distance)
    root_near = math.sqrt(near * near + distance * distance)
    return 2.0 * position * separation / (root_far + root_near)


def young_paraxial_relative_deviation(
    *, screen_position_m: float, slit_separation_m: float, screen_distance_m: float
) -> float:
    """傍轴光程差相对精确值的**首阶**相对偏差``(x^2 + d^2/4) / (2 L^2)``。

    由``OPD_exact / OPD_paraxial = 2 L / (sqrt(A) + sqrt(B))``展开到二阶得到，
    符号为正表示**傍轴式偏大**（精确光程差比它小）。

    这是"傍轴适用吗"的可计算判据：要`dx = lambda L / d`准到`1e-6`，
    就把本函数压到`1e-6`以下。**本仓不替调用方裁决多小算小**——
    给数，不给一个拍出来的阈值。首阶展开本身在偏差大到`1e-2`量级时失准，
    那时应直接用`young_exact_path_difference_m`。
    """

    position = _require_finite_length(screen_position_m, "screen_position_m")
    separation = _require_positive_length(slit_separation_m, "slit_separation_m")
    distance = _require_positive_length(screen_distance_m, "screen_distance_m")
    half = 0.5 * separation
    return (position * position + half * half) / (2.0 * distance * distance)


def michelson_path_difference_m(*, mirror_displacement_m: float) -> float:
    """迈克尔逊干涉仪：动镜位移`d`→光程差``2 d``（光去一趟回一趟）。

    这个2是`MICHELSON_OPD_PER_MIRROR_DISPLACEMENT`，
    **不是**`fts.DOUBLE_SIDED_OPD_FACTOR`那个2（那个是干涉图录在`-L..+L`上）。
    两者今天数值相同纯属巧合。

    本函数存在的全部理由就是那个2：把动镜位移直接当光程差用是**2倍的条纹级次错**，
    而2倍在干涉图上"看起来完全合理"。FTS的`max_opd_m`是**光程差**，
    所以要扫到`L`的最大光程差，动镜只需走`L / 2`。
    """

    displacement = _require_finite_length(mirror_displacement_m, "mirror_displacement_m")
    return MICHELSON_OPD_PER_MIRROR_DISPLACEMENT * displacement


__all__ = [
    "FULL_COHERENCE",
    "MICHELSON_OPD_PER_MIRROR_DISPLACEMENT",
    "PHASE_ACCURACY_RAD_PER_FRINGE_ORDER",
    "fringe_order",
    "fringe_visibility",
    "michelson_path_difference_m",
    "phase_difference_rad",
    "two_beam_intensity",
    "two_beam_max_intensity",
    "two_beam_mean_intensity",
    "two_beam_min_intensity",
    "young_exact_path_difference_m",
    "young_fringe_spacing_m",
    "young_paraxial_path_difference_m",
    "young_paraxial_relative_deviation",
]
