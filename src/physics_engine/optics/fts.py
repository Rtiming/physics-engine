"""傅里叶变换光谱仪的仪器线型与切趾（一维闭式，零依赖）。

FTS永远不会把一条谱线报成一条线。扫描停在某个最大光程差`L`，
分析者再乘一个切趾窗`w`，回来的东西是那条线与一个固定核的卷积——
**仪器线型**（ILS）。分辨率、边瓣、去卷积除的是什么，问的都是这个核。

## 本模块给的三样

1. **无切趾ILS的闭式**：``ILS(dsigma) = sinc(2 L dsigma)``（归一化sinc），
   首零在``1 / (2 L)``，半高全宽``1.2067091288 / (2 L)``；
2. **Norton-Beer三组切趾窗**：``A(x) = sum_i C_i (1 - x^2)^i``，``x = Delta / L``；
3. **切趾的通量代价**：切趾窗与boxcar的积分之比（闭式），
   即切趾换来的边瓣压低要付的信噪比。

**不做的**：带切趾的ILS本身（那要球贝塞尔与它的小自变量级数，另一块）、
自切趾（有限孔径把条纹对比度吃掉的那一项，FTS一侧已有实现）、
相位误差、采样与混叠。见`__init__.py`的"明确不做的"。

## 单位边界

* **波数是谱学波数`sigma = 1/lambda`，单位周/米**（`per_m`），
  不是角波数`k = 2 pi / lambda`。两者差一个2pi，见`diffraction.py`；
* **`max_opd_m`是单边最大光程差`L`**。干涉图录在`-L..+L`上，
  所以一切分辨率公式里的分母是``2 L``而不是`L`
  （`DOUBLE_SIDED_OPD_FACTOR`）。把`L`与`2L`弄反是**2倍的分辨率错**，
  而2倍在光谱上"看起来完全合理"——这正是它危险的地方；
* **sinc取归一化定义**``sinc(z) = sin(pi z) / (pi z)``，首零在`z = 1`。
  未归一化的``sin(z)/z``首零在`z = pi`，两者差一个pi
  （`normalised_sinc`的名字就是这道边界）。

## 参考解出处

无切趾ILS与`1.20671`：任何FTS教科书都有；本仓的独立oracle是对
``sin(pi z) / (pi z) = 1/2``直接求根，不引用实测数。
消费方fts-digital-twin的`assessment/instrument_line_shape.py`用同一约定
（`sinc(2 L dsigma)`、首零`1/(2L)`、FWHM`1.2067/(2L)`），本模块与它对得上。

Norton-Beer系数取Norton & Beer 1976（JOSA 66, 259）及其1977勘误
（JOSA 67, 419）的标准三组，与Naylor & Tahic 2007（JOSA A 24, 3644）表1
所列的NB weak/medium/strong同值。**本仓没有重新推导这三组系数**——
`sum(Ci) = 1`是它们的**必要**自洽条件（切趾窗在零光程差处必须取1，
那一点承载全部通量；`sum != 1`会把整条谱线整体缩放），
不是这三组数值正确性的证明。这条限制写在案例页的已知失效清单里。
"""

from __future__ import annotations

import math

from physics_engine.optics.errors import OpticsError

#: 干涉图录在`-L..+L`上，因此有效孔径是``2 L``：一切"1/(2L)"里的那个2。
#: **写成常量而不是字面量**：把单边`L`当成全程`2L`是2倍的分辨率错，
#: 且2倍在光谱上看起来完全合理——裸的2在代码里辨认不出是哪一个。
DOUBLE_SIDED_OPD_FACTOR: float = 2.0

#: 无切趾ILS的半高全宽，以sinc自变量为单位：``sinc(z) = 1/2``的两个根之差。
#: 由``sin(pi z) / (pi z) = 1/2``求根得到（`pi z = 1.8954942670339812`），
#: 教科书四舍五入成1.20671，本常量保留全部有效位——
#: 判据用它算，不用那个被截过的1.20671。
UNAPODISED_FWHM_IN_SINC_UNITS: float = 1.2067091288032286

#: 教科书写法的位数。判据里"本实现与教科书值一致"比到这个精度为止。
UNAPODISED_FWHM_TEXTBOOK: float = 1.20671

#: Norton-Beer三组系数（C0..C3），``A(x) = sum_i C_i (1 - x^2)^i``。
#: 三组各自``sum(Ci) = 1``——见模块docstring关于它是必要条件而非证明的说明。
NORTON_BEER_COEFFICIENTS: dict[str, tuple[float, float, float, float]] = {
    "weak": (0.384093, -0.087577, 0.703484, 0.0),
    "medium": (0.152442, -0.136176, 0.983734, 0.0),
    "strong": (0.045335, 0.0, 0.554883, 0.399782),
}

#: 切趾强度的申报次序：弱→中→强。排序判据按它比。
NORTON_BEER_STRENGTHS: tuple[str, ...] = ("weak", "medium", "strong")

#: ``sum(Ci) = 1``这条门的容差。实测三组残差都是**恰好0**（十进制系数在
#: 二进制下相加正好落回1.0），留1e-6是给将来可能引入的其它系数组
#: （文献里另有四阶与更多组），不是给这三组放水。
NORTON_BEER_UNIT_SUM_TOLERANCE: float = 1.0e-6

#: ``int_{-1}^{1} (1 - x^2)^i dx / int_{-1}^{1} 1 dx = 2^(2i) (i!)^2 / (2i+1)!``。
#: 闭式，逐项手算可核：1、2/3、8/15、16/35。切趾的通量代价由它加权得到。
_ORDER_INTEGRAL_RATIO: tuple[float, float, float, float] = (
    1.0,
    2.0 / 3.0,
    8.0 / 15.0,
    16.0 / 35.0,
)


def normalised_sinc(z: float) -> float:
    """归一化sinc``sin(pi z) / (pi z)``，`sinc(0) = 1`，首零在`z = 1`。

    **不是**``sin(z) / z``（那个的首零在`z = pi`）。名字里的`normalised`
    就是这道边界；FTS一侧与NumPy的`sinc`都取这个约定。
    """

    value = float(z)
    if not math.isfinite(value):
        raise OpticsError(f"normalised_sinc需要有限自变量：{z!r}")
    if value == 0.0:
        return 1.0
    argument = math.pi * value
    return math.sin(argument) / argument


def _require_max_opd(max_opd_m: float) -> float:
    value = float(max_opd_m)
    if not math.isfinite(value) or value <= 0.0:
        raise OpticsError(f"最大光程差必须是有限正数（米，单边）：{max_opd_m!r}")
    return value


def unapodised_line_shape(wavenumber_offset_per_m: float, *, max_opd_m: float) -> float:
    """无切趾仪器线型``ILS(dsigma) = sinc(2 L dsigma)``，峰值归一到1。

    `wavenumber_offset_per_m`是相对线心的**谱学**波数偏移（周/米）。
    """

    length = _require_max_opd(max_opd_m)
    offset = float(wavenumber_offset_per_m)
    if not math.isfinite(offset):
        raise OpticsError(f"波数偏移必须有限（周/米）：{wavenumber_offset_per_m!r}")
    return normalised_sinc(DOUBLE_SIDED_OPD_FACTOR * length * offset)


def unapodised_first_zero_per_m(max_opd_m: float) -> float:
    """无切趾ILS的首零位置``1 / (2 L)``（周/米）。"""

    return 1.0 / (DOUBLE_SIDED_OPD_FACTOR * _require_max_opd(max_opd_m))


def unapodised_fwhm_per_m(max_opd_m: float) -> float:
    """无切趾ILS的半高全宽``1.2067091288 / (2 L)``（周/米）。"""

    return UNAPODISED_FWHM_IN_SINC_UNITS / (
        DOUBLE_SIDED_OPD_FACTOR * _require_max_opd(max_opd_m)
    )


def norton_beer_coefficients(strength: str) -> tuple[float, float, float, float]:
    """一组Norton-Beer系数，或按已实现的组名失败关闭。"""

    try:
        return NORTON_BEER_COEFFICIENTS[strength]
    except KeyError:
        known = ", ".join(NORTON_BEER_STRENGTHS)
        raise OpticsError(
            f"未知的Norton-Beer切趾强度{strength!r}；已实现：{known}"
        ) from None


def norton_beer_window(opd_m: float, *, strength: str, max_opd_m: float) -> float:
    """Norton-Beer切趾窗``A(Delta) = sum_i C_i (1 - (Delta/L)^2)^i``。

    扫描区间之外（``|Delta| > L``）取0——那里没有数据，窗不是"延拓"是"没有"。
    `A(0) = sum(Ci) = 1`（零光程差点承载全部通量，不许被缩放）；
    `A(L) = C0`（窗在扫描端点上的残值，弱组0.384、中组0.152、强组0.045，
    **这个残值越小切趾越强**）。
    """

    length = _require_max_opd(max_opd_m)
    coefficients = norton_beer_coefficients(strength)
    position = float(opd_m)
    if not math.isfinite(position):
        raise OpticsError(f"光程差必须有限（米）：{opd_m!r}")
    if abs(position) > length:
        return 0.0
    reduced = position / length
    base = 1.0 - reduced * reduced
    total = 0.0
    power = 1.0
    for coefficient in coefficients:
        total += coefficient * power
        power *= base
    return total


def norton_beer_throughput(strength: str) -> float:
    """切趾的通量代价：``int A / int boxcar``（闭式）。

    切趾压边瓣要付的账。``int_{-L}^{L} A dDelta / (2 L)``
    ``= sum_i C_i 2^(2i) (i!)^2 / (2i+1)!``。
    实测弱0.7009 > 中0.5863 > 强0.5240——**越强的切趾扔掉越多信号**，
    这个排序本身是一条可断言的物理判据。
    """

    coefficients = norton_beer_coefficients(strength)
    return math.fsum(
        coefficient * ratio
        for coefficient, ratio in zip(coefficients, _ORDER_INTEGRAL_RATIO, strict=True)
    )


__all__ = [
    "DOUBLE_SIDED_OPD_FACTOR",
    "NORTON_BEER_COEFFICIENTS",
    "NORTON_BEER_STRENGTHS",
    "NORTON_BEER_UNIT_SUM_TOLERANCE",
    "UNAPODISED_FWHM_IN_SINC_UNITS",
    "UNAPODISED_FWHM_TEXTBOOK",
    "normalised_sinc",
    "norton_beer_coefficients",
    "norton_beer_throughput",
    "norton_beer_window",
    "unapodised_first_zero_per_m",
    "unapodised_fwhm_per_m",
    "unapodised_line_shape",
]
