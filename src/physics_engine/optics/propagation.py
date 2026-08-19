"""标量场传播：角谱、菲涅耳、夫琅禾费——0031列进负空间的那一块（决策0086）。

`diffraction.py`只有**闭式解**（圆孔远场的`E(x)=2J1(x)/x`），
它算不了"一般孔径"。本模块把场真的传过去：孔径掩模→变换→观察面，
**由传播算出而不是闭式代入**。案例`cases/double_slit_propagated`是它的第一个消费方。

零运行时依赖：只用`math`/`cmath`与本域的`field`/`diffraction`/`errors`。

## 三个形制与它们**互补的**适用域（本模块最要紧的一段）

| 形制 | 近似 | 输出网格 | 适用域 |
|---|---|---|---|
| `propagate_angular_spectrum` | 无（标量精确） | 与入射面**同间距** | ``0 <= z <= z_c`` |
| `propagate_fresnel` | 傍轴 | ``lambda z / (N dx)``，**随z张开** | ``z >= z_c`` |
| `propagate_fraunhofer` | 傍轴＋丢掉孔径二次相位 | 同上 | ``z >= z_c``且孔径边缘相位``<= FRAUNHOFER_MAX_EDGE_PHASE_RAD`` |

**两个域的边界是同一个数**``z_c = N dx^2 / lambda``（`transfer_function_max_distance_m`），
两侧各自的推导：

* 角谱是**传递函数形制**：相位``2 pi z sqrt(1/lambda^2 - f^2)``的局部频率
  在``f = 1/(2dx)``处约为``lambda z /(2 dx)``，它必须落在窗口半宽``N dx / 2``之内，
  否则传递函数自己在频域被采样不足、图样从窗口另一侧折回来。
  整理即``z <= N dx^2 / lambda``；
* 菲涅耳单次FFT形制是**冲激响应形制**：入射面上要乘一个啁啾
  ``exp(i pi x^2/(lambda z))``，它的局部频率在``x = N dx / 2``处是
  ``N dx /(2 lambda z)``，必须落在采样带宽半宽``1/(2 dx)``之内，
  整理即``z >= N dx^2 / lambda``。**同一个数，反向的不等号。**

**两轴不等长时两个域之间会张开一条缝，本模块不掩盖它**：角谱要**两轴都**没混叠，
所以上界取``min``；菲涅耳同理，下界取``max``。方阵上两者相等、无缝交接；
而``256 x 8``这种扁网格上上界是``8 dy^2/lambda``、下界是``256 dx^2/lambda``，
中间隔了32倍——**那一段两个形制都拒答**。这不是实现的漏，是那个网格本身
在那段距离上没有能力表示结果（y方向的窗口只有8个采样宽，衍射早就跑出窗口了）。
要么把行数加上去，要么承认那段距离算不了。

越界**失败关闭**，不给数。理由是本仓的诚实条款：
**一个在混叠区静默给数的传播器是冒充**——它给出的图样光滑、对称、看起来完全合理，
而且错得没有任何迹象。这与`airy_first_minimum_half_angle_rad`在``lambda/D``过大时
拒答是同一条纪律（0031第3.3节）。

行数或列数为1的轴**不参与判据**：单个采样点没有频率内容可混叠
（`signed_frequency_indices(1)`只有一个0）。一行N列的场因此就是一维问题，
本模块照做，不另开一维入口。

## 坐标次序：**全程FFT次序**，index 0 是原点

入射面与观察面的第`n`个采样点的坐标是``signed_frequency_indices(N)[n] * pitch``——
即``0, dx, 2dx, ..., -(N/2)dx, ..., -dx``。**不做fftshift**，因为每一次shift
都是一次可能忘掉的对称操作，而忘掉之后菲涅耳啁啾会以窗口边缘为中心、
图样会整体错半屏——**且不报错**。`spatial_coordinates_m`是唯一的坐标出口，
孔径掩模与观察面读数都要经它。

## 单位与符号

* 角波数经`diffraction.angular_wavenumber_rad_per_m`取，**不在本模块自己乘2pi**
  （0031第3.3节那张表立的规矩）；
* 传播因子取``exp(+i k z ...)``，与`field.FORWARD_TRANSFORM_SIGN = -1`配套。
  这一对符号必须同时翻，翻错了传播方向反过来而**强度图样一模一样**——
  `tests/test_optics_propagation.py`有一条门用半群性质加单向性抓它。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from physics_engine.optics.diffraction import angular_wavenumber_rad_per_m
from physics_engine.optics.errors import OpticsError
from physics_engine.optics.field import (
    ComplexField2D,
    fft2,
    ifft2,
    signed_frequency_indices,
)

#: 三个形制的名字。写成常量是因为它随`PropagatedField`跨函数传，
#: 而"这块数据是用哪个近似算出来的"是判据的一部分：
#: 拿夫琅禾费的结果去验菲涅耳数不小的构型，是把近似的失效当成了实现的错。
ANGULAR_SPECTRUM_METHOD: str = "angular_spectrum"
FRESNEL_METHOD: str = "fresnel_single_fft"
FRAUNHOFER_METHOD: str = "fraunhofer_single_fft"

#: 夫琅禾费近似丢掉的是孔径面上的二次相位``pi a^2 /(lambda z)``（`a`是孔径半宽）。
#: 申报的门槛是**相位**不是菲涅耳数，因为被丢掉的就是那个相位。
#: 0.1弧度**不是拍出来的**：`tests/test_optics_propagation.py`把同一个圆孔
#: 分别用`propagate_fresnel`（保留该相位）与`propagate_fraunhofer`（丢掉它）
#: 算一遍，量归一化强度图样的最大差——实测
#: 边缘相位0.02→**4.80e-6**、0.05→**3.00e-5**、0.10→**1.20e-4**，
#: **随相位平方增长**（0.02到0.10是5倍相位、25倍误差，实测25.0倍）。
#: 取0.1即申报"丢掉这一项带来的强度偏差不超过1.2e-4"。
#: 菲涅耳数上界由它折算：``N_F <= 0.1/pi = 0.0318``（`fraunhofer_max_fresnel_number`）。
FRAUNHOFER_MAX_EDGE_PHASE_RAD: float = 0.1

#: 孔径边界落在采样点上时的判定裕度（以采样间距为单位）。
#:
#: **这条常量来自一次实测的失败关闭，不是防御式编程**：第一版的单缝掩模写
#: ``half = (M/2 - 0.5) * dx``，边界恰好落在第``M/2``个采样上，于是"这个点在孔内
#: 还是孔外"由浮点最后一位决定——同一段代码在``dx = 10e-6``时给出8个采样、
#: 在``dx = 2.5e-6``时给出31个（期望32）。后果不是精度差一点：**采样缝宽差一个
#: 整格**，夫琅禾费零点因此整体挪位，而图样看起来仍然是漂亮的sinc平方
#: （实测最大偏差从1.4e-3跳到2.7e-2，且"收敛阶"变成负数）。
#:
#: 处理是**失败关闭**而不是给个容差偷偷把它算进去："这个采样点在不在孔内"
#: 在几何上确实没有答案，替调用方猜一个就是冒充。
APERTURE_EDGE_AMBIGUITY_TOLERANCE: float = 1.0e-9


def _positive_finite(value: float, what: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise OpticsError(f"{what}必须是有限正数：{value!r}")
    return number


def _nonnegative_finite(value: float, what: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise OpticsError(f"{what}必须是有限非负数：{value!r}")
    return number


def _reject_edge_on_a_sample(
    boundary_m: float, pitch_m: float, count_limit: float, what: str
) -> None:
    """孔径边界落在采样点上（到`APERTURE_EDGE_AMBIGUITY_TOLERANCE`以内）即失败关闭。"""

    ratio = boundary_m / pitch_m
    if abs(ratio) > count_limit:
        #: 边界在采样窗口之外，那里没有采样点，谈不上归谁。
        return
    if abs(ratio - round(ratio)) <= APERTURE_EDGE_AMBIGUITY_TOLERANCE:
        raise OpticsError(
            f"{what}落在第{round(ratio)}个采样点上（边界/间距={ratio!r}）："
            "那个采样点在孔内还是孔外由浮点最后一位决定，"
            "**采样孔宽会整差一格**、衍射零点随之挪位而图样照样漂亮。"
            "把边界移开半格（例如加减0.25个采样间距）再调——"
            "引擎不替你猜这个点归谁"
        )


def spatial_coordinates_m(count: int, pitch_m: float) -> tuple[float, ...]:
    """采样点坐标（米），**FFT次序**：``0, d, 2d, ..., -(N/2)d, ..., -d``。

    孔径掩模与观察面读数都要经这个出口。自己在调用点写``(n - N/2) * d``
    会得到**居中次序**，与本模块的变换差一次fftshift——图样错半屏且不报错。
    """

    pitch = _positive_finite(pitch_m, "采样间距")
    return tuple(index * pitch for index in signed_frequency_indices(count))


def transfer_function_max_distance_m(
    *, count: int, pitch_m: float, wavelength_m: float
) -> float:
    """``z_c = N dx^2 / lambda``——角谱的上界，同时是菲涅耳单次FFT的下界。

    推导见模块docstring。两个形制的适用域在这里**无缝且不重叠**地交接。
    """

    if count <= 0:
        raise OpticsError(f"采样点数必须是正整数：{count!r}")
    pitch = _positive_finite(pitch_m, "采样间距")
    wavelength = _positive_finite(wavelength_m, "波长")
    return count * pitch * pitch / wavelength


def fresnel_number(
    *, aperture_half_width_m: float, wavelength_m: float, distance_m: float
) -> float:
    """菲涅耳数``N_F = a^2 /(lambda z)``。`a`是孔径**半宽**（圆孔即半径）。

    名字里带`half_width`与`airy_argument`吃半径是同一条实践：
    半宽写成全宽会让菲涅耳数差4倍，而4倍的菲涅耳数不会报任何错。
    """

    half_width = _positive_finite(aperture_half_width_m, "孔径半宽")
    wavelength = _positive_finite(wavelength_m, "波长")
    distance = _positive_finite(distance_m, "传播距离")
    return half_width * half_width / (wavelength * distance)


def fraunhofer_max_fresnel_number() -> float:
    """夫琅禾费近似成立的菲涅耳数上界，由申报的边缘相位折算。"""

    return FRAUNHOFER_MAX_EDGE_PHASE_RAD / math.pi


@dataclass(frozen=True)
class PropagatedField:
    """观察面上的场：数**加上**它的单位与出身。

    `ComplexField2D`故意不带单位（见`field.py`）；一旦传播完，
    "间距多少、传了多远、哪个波长、用的哪个近似"就必须跟着数走——
    否则下一个函数只能猜，而猜错不报错。
    """

    field: ComplexField2D
    pitch_x_m: float
    pitch_y_m: float
    distance_m: float
    wavelength_m: float
    method: str

    def coordinates_x_m(self) -> tuple[float, ...]:
        return spatial_coordinates_m(self.field.column_count, self.pitch_x_m)

    def coordinates_y_m(self) -> tuple[float, ...]:
        return spatial_coordinates_m(self.field.row_count, self.pitch_y_m)

    def intensity_rows(self) -> tuple[tuple[float, ...], ...]:
        return self.field.intensity_rows()

    def total_power(self) -> float:
        """``sum |U|^2 dx dy``——带单位的能量。

        它是Parseval恒等式在**带单位形制上**的兑现：菲涅耳前因子
        ``dx dy /(lambda z)``与输出间距``lambda z /(N dx)``两者互相抵消，
        总能量恰好守恒。前因子写错（漏了`dx dy`、漏了`1/(lambda z)`、
        或把`i`丢了）在强度图样的**形状**上完全看不出来，只有这条能抓。
        """

        return (
            sum(sum(row) for row in self.field.intensity_rows())
            * self.pitch_x_m
            * self.pitch_y_m
        )


def incident_power(
    field: ComplexField2D, *, pitch_x_m: float, pitch_y_m: float
) -> float:
    """入射面上的``sum |U0|^2 dx dy``，与`PropagatedField.total_power`同口径。"""

    pitch_x = _positive_finite(pitch_x_m, "x间距")
    pitch_y = _positive_finite(pitch_y_m, "y间距")
    return sum(sum(row) for row in field.intensity_rows()) * pitch_x * pitch_y


def _sampling_limits(
    field: ComplexField2D, *, pitch_x_m: float, pitch_y_m: float, wavelength_m: float
) -> tuple[float, float]:
    """``(角谱的上界, 菲涅耳的下界)``。行/列为1的轴不参与（无频率内容可混叠）。"""

    limits = []
    if field.column_count > 1:
        limits.append(
            transfer_function_max_distance_m(
                count=field.column_count, pitch_m=pitch_x_m, wavelength_m=wavelength_m
            )
        )
    if field.row_count > 1:
        limits.append(
            transfer_function_max_distance_m(
                count=field.row_count, pitch_m=pitch_y_m, wavelength_m=wavelength_m
            )
        )
    if not limits:
        return (math.inf, 0.0)
    return (min(limits), max(limits))


def propagate_angular_spectrum(
    field: ComplexField2D,
    *,
    wavelength_m: float,
    distance_m: float,
    pitch_x_m: float,
    pitch_y_m: float,
) -> PropagatedField:
    """角谱传播（标量精确，不做傍轴近似）。输出网格与入射面**同间距**。

    ``U(z) = IFFT2{ FFT2{U0} * exp(i k z sqrt(1 - (lambda fx)^2 - (lambda fy)^2)) }``。
    根号变虚的那些分量是**倏逝波**，按``exp(-k z sqrt(...))``衰减
    ——不是丢掉、也不是让它涨。`z = 0`时传递函数**逐位等于1**，
    于是结果与``ifft2(fft2(U0))``逐位相同（这条被一条零容差的门守着）。

    适用域``0 <= z <= z_c``，越界失败关闭（模块docstring第二节）。
    **不做反向传播**（`z < 0`）：倏逝分量会按``exp(+k|z| sqrt(...))``爆炸，
    在浮点上那不是"病态"是"没有意义"。
    """

    wavelength = _positive_finite(wavelength_m, "波长")
    distance = _nonnegative_finite(distance_m, "传播距离")
    pitch_x = _positive_finite(pitch_x_m, "x间距")
    pitch_y = _positive_finite(pitch_y_m, "y间距")
    maximum, _ = _sampling_limits(
        field, pitch_x_m=pitch_x, pitch_y_m=pitch_y, wavelength_m=wavelength
    )
    if distance > maximum:
        raise OpticsError(
            f"角谱在z={distance!r}米处已经采样不足：上界z_c={maximum!r}米"
            f"（= N dx^2 / lambda）。这一段该用propagate_fresnel，"
            "本函数**不在混叠区给数**——那样给出的图样光滑、对称、且错得没有迹象"
        )

    wavenumber = angular_wavenumber_rad_per_m(wavelength)
    frequencies_x = [
        index / (field.column_count * pitch_x)
        for index in signed_frequency_indices(field.column_count)
    ]
    frequencies_y = [
        index / (field.row_count * pitch_y)
        for index in signed_frequency_indices(field.row_count)
    ]
    spectrum = fft2(field)
    filtered = []
    for row_index, frequency_y in enumerate(frequencies_y):
        y_term = (wavelength * frequency_y) ** 2
        row = spectrum.rows[row_index]
        new_row = []
        for column_index, frequency_x in enumerate(frequencies_x):
            square = 1.0 - (wavelength * frequency_x) ** 2 - y_term
            if square >= 0.0:
                factor = cmath.exp(complex(0.0, wavenumber * distance * math.sqrt(square)))
            else:
                factor = complex(math.exp(-wavenumber * distance * math.sqrt(-square)), 0.0)
            new_row.append(row[column_index] * factor)
        filtered.append(tuple(new_row))
    return PropagatedField(
        field=ifft2(ComplexField2D(tuple(filtered))),
        pitch_x_m=pitch_x,
        pitch_y_m=pitch_y,
        distance_m=distance,
        wavelength_m=wavelength,
        method=ANGULAR_SPECTRUM_METHOD,
    )


def _single_fft_propagation(
    field: ComplexField2D,
    *,
    wavelength_m: float,
    distance_m: float,
    pitch_x_m: float,
    pitch_y_m: float,
    apply_aperture_chirp: bool,
    method: str,
) -> PropagatedField:
    """菲涅耳／夫琅禾费共用的单次FFT路径。两者只差入射面那一个啁啾。"""

    wavelength = _positive_finite(wavelength_m, "波长")
    distance = _positive_finite(distance_m, "传播距离")
    pitch_x = _positive_finite(pitch_x_m, "x间距")
    pitch_y = _positive_finite(pitch_y_m, "y间距")
    _, minimum = _sampling_limits(
        field, pitch_x_m=pitch_x, pitch_y_m=pitch_y, wavelength_m=wavelength
    )
    if distance < minimum:
        raise OpticsError(
            f"单次FFT形制在z={distance!r}米处采样不足：下界z_c={minimum!r}米"
            f"（= N dx^2 / lambda）。这一段该用propagate_angular_spectrum。"
            "两个形制的适用域以同一个z_c互补，中间没有缝也没有重叠"
        )

    wavenumber = angular_wavenumber_rad_per_m(wavelength)
    reduced = wavelength * distance
    source_x = spatial_coordinates_m(field.column_count, pitch_x)
    source_y = spatial_coordinates_m(field.row_count, pitch_y)

    if apply_aperture_chirp:
        chirp_x = [cmath.exp(complex(0.0, math.pi * x * x / reduced)) for x in source_x]
        chirp_y = [cmath.exp(complex(0.0, math.pi * y * y / reduced)) for y in source_y]
        prepared = ComplexField2D(
            tuple(
                tuple(
                    value * chirp_y[row_index] * chirp_x[column_index]
                    for column_index, value in enumerate(row)
                )
                for row_index, row in enumerate(field.rows)
            )
        )
    else:
        prepared = field

    #: 观察面间距：``lambda z /(N dx)``。它**随z张开**——这正是单次FFT形制
    #: 与传递函数形制的分水岭，也是它没有半群性质的原因（两段z的输出网格不同）。
    observation_pitch_x = reduced / (field.column_count * pitch_x)
    observation_pitch_y = reduced / (field.row_count * pitch_y)
    observation_x = spatial_coordinates_m(field.column_count, observation_pitch_x)
    observation_y = spatial_coordinates_m(field.row_count, observation_pitch_y)

    #: 前因子``exp(ikz)/(i lambda z) * dx dy``。`dx dy`是把和变成积分的那一步，
    #: `1/(i lambda z)`是菲涅耳衍射积分自带的——两者一起决定总能量守恒。
    prefactor = (
        cmath.exp(complex(0.0, wavenumber * distance))
        / complex(0.0, reduced)
        * (pitch_x * pitch_y)
    )
    tail_x = [cmath.exp(complex(0.0, math.pi * x * x / reduced)) for x in observation_x]
    tail_y = [cmath.exp(complex(0.0, math.pi * y * y / reduced)) for y in observation_y]

    spectrum = fft2(prepared)
    rows = tuple(
        tuple(
            value * prefactor * tail_y[row_index] * tail_x[column_index]
            for column_index, value in enumerate(row)
        )
        for row_index, row in enumerate(spectrum.rows)
    )
    return PropagatedField(
        field=ComplexField2D(rows),
        pitch_x_m=observation_pitch_x,
        pitch_y_m=observation_pitch_y,
        distance_m=distance,
        wavelength_m=wavelength,
        method=method,
    )


def propagate_fresnel(
    field: ComplexField2D,
    *,
    wavelength_m: float,
    distance_m: float,
    pitch_x_m: float,
    pitch_y_m: float,
) -> PropagatedField:
    """菲涅耳传播（傍轴），单次FFT形制。观察面间距``lambda z /(N dx)``。

    ``U(x2) = exp(ikz)/(i lambda z) exp(i pi x2^2/(lambda z))
    FFT{ U0(x1) exp(i pi x1^2/(lambda z)) } dx dy``。

    **没有半群性质**：传两次`z/2`与传一次`z`的输出落在**不同的网格**上
    （间距分别是``lambda z/(2 N dx)``与``lambda z/(N dx)``），
    两者本来就不该逐点比。半群要验请用`propagate_angular_spectrum`。
    这是形制的性质不是实现的缺陷，写在这里以免被当成缺口。

    适用域``z >= z_c``，越界失败关闭。
    """

    return _single_fft_propagation(
        field,
        wavelength_m=wavelength_m,
        distance_m=distance_m,
        pitch_x_m=pitch_x_m,
        pitch_y_m=pitch_y_m,
        apply_aperture_chirp=True,
        method=FRESNEL_METHOD,
    )


def propagate_fraunhofer(
    field: ComplexField2D,
    *,
    wavelength_m: float,
    distance_m: float,
    pitch_x_m: float,
    pitch_y_m: float,
    aperture_half_width_m: float,
) -> PropagatedField:
    """夫琅禾费远场——菲涅耳丢掉入射面二次相位之后的极限。

    `aperture_half_width_m`是**必填**而不是可选：夫琅禾费成不成立由孔径尺度决定，
    而场本身不知道自己的孔径有多大。要调用方说出来，等于逼他面对
    "我这个构型真的在远场吗"这个问题——引擎不代判，但也不许不判
    （与`airy_first_minimum_half_angle_rad`拒答同源）。

    边缘相位``pi a^2/(lambda z)``超过`FRAUNHOFER_MAX_EDGE_PHASE_RAD`即失败关闭，
    并在信息里报出该用`propagate_fresnel`。
    """

    number = fresnel_number(
        aperture_half_width_m=aperture_half_width_m,
        wavelength_m=wavelength_m,
        distance_m=distance_m,
    )
    edge_phase = math.pi * number
    if edge_phase > FRAUNHOFER_MAX_EDGE_PHASE_RAD:
        raise OpticsError(
            f"菲涅耳数{number!r}对应孔径边缘相位{edge_phase!r}弧度，"
            f"超过夫琅禾费近似申报的上界{FRAUNHOFER_MAX_EDGE_PHASE_RAD!r}——"
            "这个构型不在远场，请用propagate_fresnel（它保留那个二次相位）"
        )
    return _single_fft_propagation(
        field,
        wavelength_m=wavelength_m,
        distance_m=distance_m,
        pitch_x_m=pitch_x_m,
        pitch_y_m=pitch_y_m,
        apply_aperture_chirp=False,
        method=FRAUNHOFER_METHOD,
    )


def rectangular_aperture(
    *,
    row_count: int,
    column_count: int,
    pitch_x_m: float,
    pitch_y_m: float,
    half_width_x_m: float,
    half_width_y_m: float,
    centre_x_m: float = 0.0,
    centre_y_m: float = 0.0,
) -> ComplexField2D:
    """矩孔（单缝就是它的一维极限）的振幅掩模，坐标经`spatial_coordinates_m`。

    边界取**闭区间**``|x - x0| <= a``：半宽恰好落在采样点上时那一点**算在孔内**。
    这一条必须写死，因为它决定缝宽是`M`还是`M+1`个采样——差一个采样点，
    夫琅禾费零点的位置就整体挪一格，而图样看起来照样漂亮。

    **中心可以不落在采样点上**，而且常常必须不落：本模块的坐标是
    ``0, +d, ..., -d``，任何以采样点为中心的对称孔覆盖的采样数**必然是奇数**
    （``2p+1``）。要让缝宽恰好是2的幂（于是夫琅禾费零点落在整数bin上、
    可以零容差地判），中心就得偏半格——``centre_x_m = pitch/2``。
    这不是技巧，是采样与对称性的一条硬事实，写在这里以免下一个人以为
    "居中"总是对的。
    """

    half_x = _positive_finite(half_width_x_m, "x半宽")
    half_y = _positive_finite(half_width_y_m, "y半宽")
    origin_x = float(centre_x_m)
    origin_y = float(centre_y_m)
    if not (math.isfinite(origin_x) and math.isfinite(origin_y)):
        raise OpticsError(f"孔径中心必须有限：{(centre_x_m, centre_y_m)!r}")
    xs = spatial_coordinates_m(column_count, pitch_x_m)
    ys = spatial_coordinates_m(row_count, pitch_y_m)
    for sign in (1.0, -1.0):
        _reject_edge_on_a_sample(
            origin_x + sign * half_x, pitch_x_m, column_count / 2.0, "矩孔的x边界"
        )
        _reject_edge_on_a_sample(
            origin_y + sign * half_y, pitch_y_m, row_count / 2.0, "矩孔的y边界"
        )
    return ComplexField2D.from_function(
        row_count,
        column_count,
        lambda row, column: complex(
            1.0
            if abs(xs[column] - origin_x) <= half_x and abs(ys[row] - origin_y) <= half_y
            else 0.0,
            0.0,
        ),
    )


def circular_aperture(
    *,
    row_count: int,
    column_count: int,
    pitch_x_m: float,
    pitch_y_m: float,
    radius_m: float,
) -> ComplexField2D:
    """圆孔的振幅掩模。边界同样取闭区间``r <= a``。

    圆孔在笛卡尔网格上是**阶梯边**，这是本模块与`diffraction.py`那条艾里闭式
    对比时误差的主要来源（收敛阶实测见`tests/test_optics_propagation.py`）。

    半径落在轴上采样点的那一格同样失败关闭（理由见
    `APERTURE_EDGE_AMBIGUITY_TOLERANCE`）。圆边界上还可能有别的采样点恰好
    落在圆上（勾股数），**那些没有被检**：它们改的是``O(1)``个采样而不是
    整条边，量级是``O(1/R^2)``——如实记在这里，不假装检全了。
    """

    radius = _positive_finite(radius_m, "孔径半径")
    _reject_edge_on_a_sample(radius, pitch_x_m, column_count / 2.0, "圆孔半径在x轴上")
    _reject_edge_on_a_sample(radius, pitch_y_m, row_count / 2.0, "圆孔半径在y轴上")
    xs = spatial_coordinates_m(column_count, pitch_x_m)
    ys = spatial_coordinates_m(row_count, pitch_y_m)
    squared = radius * radius
    return ComplexField2D.from_function(
        row_count,
        column_count,
        lambda row, column: complex(
            1.0 if xs[column] ** 2 + ys[row] ** 2 <= squared else 0.0, 0.0
        ),
    )


def paraxial_sine_of_angle(coordinate_m: float, distance_m: float) -> float:
    """观察面坐标 → ``sin(theta) ~ x / z``（**傍轴**）。

    单次FFT形制本来就是傍轴的，所以这里用``x/z``而不是``x/sqrt(x^2+z^2)``——
    用后者会给出一个"更精确"的角度，但它与产生这个图样的近似不自洽，
    对拍闭式时反而多出一项与实现无关的偏差。**参照要与被验对象同阶**。
    """

    distance = _positive_finite(distance_m, "传播距离")
    return float(coordinate_m) / distance


__all__ = [
    "ANGULAR_SPECTRUM_METHOD",
    "FRAUNHOFER_MAX_EDGE_PHASE_RAD",
    "FRAUNHOFER_METHOD",
    "FRESNEL_METHOD",
    "PropagatedField",
    "circular_aperture",
    "fraunhofer_max_fresnel_number",
    "fresnel_number",
    "incident_power",
    "paraxial_sine_of_angle",
    "propagate_angular_spectrum",
    "propagate_fraunhofer",
    "propagate_fresnel",
    "rectangular_aperture",
    "spatial_coordinates_m",
    "transfer_function_max_distance_m",
]
