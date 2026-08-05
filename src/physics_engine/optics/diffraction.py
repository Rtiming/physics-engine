"""标量衍射的闭式解：圆孔远场艾里斑。

一句话：直径`D`的圆孔在夫琅禾费区把一个点源摊成
``E(x) = 2 J1(x) / x``，首零在``x = 3.8317059702``，
折成角度就是那个人人会背的``sin(theta_1) = 1.2197 lambda / D``。

**本模块只做闭式解与它的单位边界，不做场传播。** 二维FFT场传播（角谱、菲涅耳）
要数组与FFT，按0016属可选加速档，是下一块的事；本块把它列在"明确不做的"里。

## 单位边界（本模块存在的一半理由）

光学侧最大的坑是**同一个"波数"有两套，差一个2pi**：

* **谱学波数** `sigma = 1 / lambda`，单位"周/米"（`per_m`）。FTS一侧全用它；
* **角波数** `k = 2 pi / lambda`，单位"弧度/米"（`rad_per_m`）。衍射一侧全用它。

艾里斑的自变量``x = k a sin(theta)``吃的是**角波数**与**半径**。
把sigma当成k、或把直径当成半径，都会给出一个"看起来很像"的图样——
差2倍或2pi倍，而且**不报任何错**。0024那条1000倍单位bug的教训在这里的形态就是它。
所以本模块的每一处换算都写成有名字的常量或有名字的函数：
`RADIANS_PER_CYCLE`、`spatial_frequency_per_m`、`angular_wavenumber_rad_per_m`、
`airy_argument`。**不要在调用点自己乘2pi。**

长度制：本域按米（`OPTICS_LENGTH_UNIT`，见`parameters.py`）。
毫米制的材料记录要先显式`converted_to("m")`。

## 参考解出处

Born & Wolf,《Principles of Optics》第8.5.2节（圆孔的夫琅禾费衍射）；
首零`j_{1,1} = 3.8317059702`取Abramowitz & Stegun表9.5（`J1`的第一个正零点，
真值3.8317059702075123156，本仓引用值截到1e-10）。
`E(x) = 2 J1(x)/x`的形式与首零值也是research/05第2.3节光学族列出的那一条判据。
"""

from __future__ import annotations

import math

from physics_engine.optics.bessel import bessel_j1
from physics_engine.optics.errors import OpticsError

#: 谱学波数（周/米）与角波数（弧度/米）之间的换算：``k = RADIANS_PER_CYCLE * sigma``。
#: **写成常量而不是字面量2pi**：一个裸的``2*math.pi``在半年后没人认得出它是
#: "周→弧度"还是某个几何因子，而这两者错了都不会报错（0024立的实践）。
RADIANS_PER_CYCLE: float = 2.0 * math.pi

#: `J1`的第一个正零点（A&S表9.5）。艾里斑的首个暗环落在这里。
#: 引用值截到1e-10；真值3.8317059702075123156，截断量7.5e-12是判据容差的来源之一。
AIRY_FIRST_ZERO_X: float = 3.8317059702

#: 引用值相对真值的截断量的**上界**（真值3.8317059702075123156，实差7.5123e-12）。
#: 案例`scalar_diffraction_airy`的容差推导逐字用它——**判据容差从它算出来，
#: 不是拍出来的**——所以它必须真的是个上界，`tests/test_optics.py`有一条门守着。
AIRY_FIRST_ZERO_TRUNCATION: float = 7.6e-12

#: ``sin(theta_1) = AIRY_FIRST_MINIMUM_DIAMETER_FACTOR * lambda / D``里的那个1.22。
#: **它吃的是直径D不是半径**——名字里带`DIAMETER`就是为了这个。
#: 由首零除以pi算出来（``x = pi D sin(theta) / lambda``），不另写一个1.22，
#: 免得两个数各自漂。
AIRY_FIRST_MINIMUM_DIAMETER_FACTOR: float = AIRY_FIRST_ZERO_X / math.pi


def angular_wavenumber_rad_per_m(wavelength_m: float) -> float:
    """波长（米）→ 角波数`k = 2 pi / lambda`（弧度/米）。"""

    value = float(wavelength_m)
    if not math.isfinite(value) or value <= 0.0:
        raise OpticsError(f"波长必须是有限正数（米）：{wavelength_m!r}")
    return RADIANS_PER_CYCLE / value


def spectroscopic_wavenumber_per_m(wavelength_m: float) -> float:
    """波长（米）→ 谱学波数`sigma = 1 / lambda`（周/米）。

    与`angular_wavenumber_rad_per_m`**差一个`RADIANS_PER_CYCLE`**。
    两个函数并存不是冗余，是让调用点必须说清自己要哪一个。
    """

    value = float(wavelength_m)
    if not math.isfinite(value) or value <= 0.0:
        raise OpticsError(f"波长必须是有限正数（米）：{wavelength_m!r}")
    return 1.0 / value


def spatial_frequency_per_m(*, half_angle_rad: float, wavelength_m: float) -> float:
    """角度 → 横向空间频率`f = sin(theta) / lambda`（周/米）。

    这是"角度"与"空间频率"之间那道边界。远场里衍射图样是孔径函数的傅里叶变换，
    **变换的自变量是空间频率不是角度**；把theta本身当自变量在小角下几乎对，
    在大角下静默地错——所以这道换算有名字。
    """

    angle = float(half_angle_rad)
    if not math.isfinite(angle):
        raise OpticsError(f"角度必须有限（弧度）：{half_angle_rad!r}")
    return math.sin(angle) * spectroscopic_wavenumber_per_m(wavelength_m)


def airy_argument(
    *, half_angle_rad: float, aperture_radius_m: float, wavelength_m: float
) -> float:
    """艾里斑的无量纲自变量``x = k a sin(theta) = 2 pi a sin(theta) / lambda``。

    吃的是**半径**`a`。传直径进来会让首零跑到``x/2``处，图样宽一倍且不报错——
    这就是本函数存在的理由：调用方不必记住吃的是半径还是直径，签名替他记。
    """

    radius = float(aperture_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise OpticsError(f"孔径半径必须是有限正数（米）：{aperture_radius_m!r}")
    frequency = spatial_frequency_per_m(
        half_angle_rad=half_angle_rad, wavelength_m=wavelength_m
    )
    return RADIANS_PER_CYCLE * radius * frequency


def airy_amplitude(x: float) -> float:
    """归一化远场振幅``E(x) = 2 J1(x) / x``，`E(0) = 1`。

    偶函数。`x = 0`是可去奇点，直接给1.0——不是为了躲除零，
    是因为极限本来就精确等于1，写成除法反而会给出一个带舍入的1。
    """

    value = float(x)
    if not math.isfinite(value):
        raise OpticsError(f"airy_amplitude需要有限自变量：{x!r}")
    if value == 0.0:
        return 1.0
    return 2.0 * bessel_j1(value) / value


def airy_intensity(x: float) -> float:
    """归一化远场光强``I(x) = E(x)^2``，峰值1。"""

    amplitude = airy_amplitude(x)
    return amplitude * amplitude


def airy_first_minimum_half_angle_rad(
    *, wavelength_m: float, aperture_diameter_m: float
) -> float:
    """首个暗环的半角``theta_1 = arcsin(1.2197 lambda / D)``（`D`是**直径**）。

    ``lambda / D``大到让正弦值超过1时**没有首个暗环**（整个半空间都在主瓣里）。
    那时失败关闭，而不是返回一个`nan`或悄悄夹到pi/2——
    "这个构型下判据不成立"与"判据成立且等于pi/2"是两件事。
    """

    diameter = float(aperture_diameter_m)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise OpticsError(f"孔径直径必须是有限正数（米）：{aperture_diameter_m!r}")
    wavelength = float(wavelength_m)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise OpticsError(f"波长必须是有限正数（米）：{wavelength_m!r}")
    sine = AIRY_FIRST_MINIMUM_DIAMETER_FACTOR * wavelength / diameter
    if sine > 1.0:
        raise OpticsError(
            f"lambda/D = {wavelength / diameter!r}时"
            f"sin(theta_1) = {sine!r} > 1：该构型没有首个暗环，"
            "主瓣覆盖整个半空间——夫琅禾费远场的角谱在这里已经不成立"
        )
    return math.asin(sine)


__all__ = [
    "AIRY_FIRST_MINIMUM_DIAMETER_FACTOR",
    "AIRY_FIRST_ZERO_TRUNCATION",
    "AIRY_FIRST_ZERO_X",
    "RADIANS_PER_CYCLE",
    "airy_amplitude",
    "airy_argument",
    "airy_first_minimum_half_angle_rad",
    "airy_intensity",
    "angular_wavenumber_rad_per_m",
    "spatial_frequency_per_m",
    "spectroscopic_wavenumber_per_m",
]
