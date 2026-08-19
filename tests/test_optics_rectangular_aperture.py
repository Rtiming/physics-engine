"""矩孔（**两个方向都有界**）的传播判据——能力位S4.6缺的那一格（决策0091）。

S4.6的label是"一般孔径的衍射（单缝／**双缝**／**矩孔**，由传播算出而非闭式代入）"。
0086把单缝与双缝做齐了，**矩孔那一格没有被传播验过**——这条是主代理合入时
逐格核出来的，轨道自己没报（0089第四节记的"同一形状的第三次"）。

**缺的到底是什么，一句话**：仓里**每一处**`rectangular_aperture`的调用
`half_width_y_m`都取``1.0e9 * pitch``，即y方向占满窗口。那**是缝不是孔**：
图样是一维``sinc^2(x)``，y方向一个采样都没被限制，
于是"二维可分离"这件事在**衍射图样**上从来没有被验过。
`optics/field.py`那边的`fft2`可分离性门验的是**变换**可分离，
不是**图样**是两条`sinc^2`的乘积——两者是两件事：
变换可分离对**任何**输入都成立（包括圆孔），而图样可分离只对可分离孔径成立。
本文件的`test_the_pattern_is_separable_and_a_circular_aperture_proves_the_gate_can_tell`
用圆孔正面证明了这条门分得开。

四条判据，每条附必红：

1. **孔真的是孔**：亮采样恰好``Mx * My``个、亮行恰好`My`、亮列恰好`Mx`
   （零容差，全是整数）。必红是仓里那个``1.0e9 * pitch``的写法——它亮满所有行；
2. **主极大与两个方向各自的前三个零点**：主极大在bin (0,0)且是全局最大；
   x零点在``bin_x = m * COLS / Mx``、y零点在``bin_y = m * ROWS / My``，
   归一化强度**恰为0.0**（缝宽取2的幂⟹蝶形末级相位差恰为pi，IEEE精确取负）。
   必红是把两个方向的半宽对调；
3. **图样可分离**：``I(r,c)/I(0,0) == (I(r,0)/I(0,0)) * (I(0,c)/I(0,0))``。
   必红是圆孔（不可分离）；
   外加剖面对**连续**``sinc^2(x) sinc^2(y)``的偏差与它的收敛阶；
4. **单缝是矩孔的极限**：把y半宽长到盖住整个窗口，掩模与仓里那个
   ``1.0e9 * pitch``的写法**逐位相同**，而离轴行的总强度**恰为0**。
   更硬的一条：离轴行占的功率份额有闭式``ROWS/My - 1``（零容差），
   `My = ROWS`时它恰好是0——**"缝是孔的极限"因此是一个可以判的数，不是一句话**。
   必红是有界的`My`——它的份额恰好不是0。

容差全部算出来或实测出来，逐条写在常量的注释里（0024第三节）。
"""

from __future__ import annotations

import math
import sys

import pytest

from physics_engine.optics.errors import OpticsError
from physics_engine.optics.fts import normalised_sinc
from physics_engine.optics.propagation import (
    circular_aperture,
    paraxial_sine_of_angle,
    propagate_fraunhofer,
    rectangular_aperture,
)

EPS = sys.float_info.epsilon

#: HeNe，与`cases/scalar_diffraction_airy`、`cases/double_slit_propagated`同一个。
WAVELENGTH_M = 632.8e-9

#: 主网格：64行×128列。**两个方向都要有真的频率内容**，
#: 所以行数不能取1、也不能像仓里既有的单缝那样让孔在y方向占满窗口。
GRID_ROWS = 64
GRID_COLUMNS = 128
GRID_PITCH_M = 10.0e-6
SCREEN_DISTANCE_M = 2.0

#: 孔的两个方向各占几个采样。**都取2的幂**，理由与单缝那条相同：
#: 夫琅禾费零点落在``bin = m N / M``上，`M`是2的幂时那正好是整数bin，
#: 可以**零容差**地判。两个方向取**不同**的数（8与4）是有意的——
#: 取成一样的话，行列搞混的实现在方形孔上完全看不出来。
APERTURE_SAMPLES_X = 8
APERTURE_SAMPLES_Y = 4

#: 半宽偏四分之一格：边界不许落在采样点上（`APERTURE_EDGE_AMBIGUITY_TOLERANCE`，
#: 0086第5.1节抓到的真缺陷）。偏多少不改**采样孔宽**，只是让"这个点归谁"没有歧义。
EDGE_OFFSET_IN_PITCHES = 0.25

#: 零点处的归一化强度**申报为恰好0**（与单缝那条同源，不是"很小"是"恰好"）。
APERTURE_ZERO_INTENSITY_FLOOR = 0.0

#: 图样可分离性的残差上界：``16 * eps``。
#: 实测最坏**7.7716e-16 = 3.50 eps**（64×128、Mx=8、My=4，全部8192个点），
#: 取16即余量4.6倍。
#:
#: **它为什么可以这么紧**：可分离孔径的二维变换恰好是两条一维变换的外积，
#: 而`fft2`是先行后列——两侧算的是同一串乘法，只差求和次序。
#: 不可分离的孔径（圆孔）在同一条式子上实测**3.5852e-2**，
#: 比上界（``16 eps = 3.55e-15``）大**1.0e13**倍——这条门分得开"可分离"与"不可分离"，
#: 而这正是"矩孔"与"圆孔"的分别。
PATTERN_SEPARABILITY_FACTOR = 16.0
PATTERN_SEPARABILITY_MEASURED = 7.7716e-16
CIRCULAR_SEPARABILITY_MEASURED = 3.5852e-2

#: 剖面对**连续**``sinc^2(x) sinc^2(y)``的偏差申报（Mx=8、My=4时）：**2.8192e-2**。
#: 它不是实现误差，是**离散孔径（Dirichlet核）与连续孔径（sinc）的差**，
#: 由较粗的那个方向（My=4）主导。固定物理孔宽（80微米×40微米）加密网格实测：
#:
#: | 网格 | Mx×My | 最大偏差 |
#: |---|---|---|
#: | 64×128 | 8×4 | 2.8192e-2 |
#: | 128×256 | 16×8 | 7.9721e-3 |
#: | 256×512 | 32×16 | 2.1429e-3 |
#: | 512×1024 | 64×32 | 5.5719e-4 |
#:
#: 阶 **1.822 → 1.895 → 1.943**，干净地趋向``O(h^2)``。
#: 本文件的门只跑前三档（第四档单次约1.2秒，会把本文件推出交互级）。
#: **注意第二档7.9721e-3与单缝那条门的M=8实测逐位同一个数**——
#: 那不是巧合：Mx=16、My=8时y方向是较粗的一维，偏差由它单独定。
SINC_PRODUCT_DEVIATION_AT_BASE_GRID = 3.0e-2
SINC_PRODUCT_MEASURED_AT_BASE_GRID = 2.8192e-2

#: 收敛阶的下界。三档实测1.822与1.895，取1.7留余量；
#: **不写死为2**——本仓在`harmonic_oscillator`上立的实践（收敛比落在区间里）。
SINC_PRODUCT_MIN_ORDER = 1.7

#: 单缝极限那条：`My = ROWS`时离轴行的强度**恰为0**（零容差）。
SLIT_LIMIT_OFF_AXIS_POWER = 0.0

#: 单缝极限用的窄网格（8行）。行数取8而不是64，是因为y方向占满窗口时
#: 孔的y半宽就是窗口半宽，夫琅禾费的远场门槛按``hypot(ax, ay)``算——
#: 64行×10微米的窗口半宽320微米在z=2米处边缘相位1.02弧度，**越界拒答**。
#: 8行时是7.9e-3弧度，在申报上界0.1之内。**这不是把网格挑软，
#: 是那条门本来就在管这件事**。
SLIT_LIMIT_ROWS = 8


# --- 构件 -----------------------------------------------------------------


def _rectangular_hole(
    samples_x,
    samples_y,
    *,
    rows=GRID_ROWS,
    columns=GRID_COLUMNS,
    pitch=GRID_PITCH_M,
):
    """两个方向都有界的矩孔。返回``(掩模, x半宽, y半宽)``。

    两个方向的中心都偏半格：以采样点为中心的对称孔覆盖的采样数**必然是奇数**，
    而我们要2的幂（`propagation.rectangular_aperture`的docstring写了这条）。
    """

    half_x = ((samples_x - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * pitch
    half_y = ((samples_y - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * pitch
    mask = rectangular_aperture(
        row_count=rows,
        column_count=columns,
        pitch_x_m=pitch,
        pitch_y_m=pitch,
        half_width_x_m=half_x,
        half_width_y_m=half_y,
        centre_x_m=0.5 * pitch,
        centre_y_m=0.5 * pitch,
    )
    return mask, half_x, half_y


def _screen(mask, half_x, half_y, *, pitch=GRID_PITCH_M):
    """夫琅禾费传播。远场门槛吃的是**孔角**到轴的距离``hypot(ax, ay)``——

    被丢掉的二次相位是``pi (x^2 + y^2)/(lambda z)``，它在孔的**角**上最大。
    只报x半宽（仓里既有的单缝就是这么调的，因为它们的y半宽更小）
    在两个方向都有界时是**报小了**，那会让越界的构型混进来。
    """

    return propagate_fraunhofer(
        mask,
        wavelength_m=WAVELENGTH_M,
        distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=pitch,
        pitch_y_m=pitch,
        aperture_half_width_m=math.hypot(half_x, half_y),
    )


def _lit_rows(mask):
    return {row for row, values in enumerate(mask.rows) if any(abs(v) > 0.5 for v in values)}


def _lit_columns(mask):
    return {
        column
        for column in range(mask.column_count)
        if any(abs(row[column]) > 0.5 for row in mask.rows)
    }


def _lit_samples(mask):
    return sum(1 for row in mask.rows for value in row if abs(value) > 0.5)


# --- 判据一｜孔真的是孔（不是缝） -----------------------------------------


def test_a_hole_is_bounded_in_both_directions_and_the_counts_are_exact():
    """亮采样``Mx * My``、亮行`My`、亮列`Mx`——**三个整数，零容差**。

    这一条锁的是**采样孔宽**，与单缝那条`aperture_sample_count`同源：
    边界落在采样点上时这些数会整差一格，而图样照样是漂亮的sinc平方
    （0086第5.1节实测踩过：期望32个采样、实得31个）。
    """

    mask, _, _ = _rectangular_hole(APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y)
    assert _lit_samples(mask) == APERTURE_SAMPLES_X * APERTURE_SAMPLES_Y
    assert len(_lit_rows(mask)) == APERTURE_SAMPLES_Y
    assert len(_lit_columns(mask)) == APERTURE_SAMPLES_X
    #: 孔在两个方向都**没有**占满窗口——这正是"孔"与"缝"的分别。
    assert len(_lit_rows(mask)) < GRID_ROWS
    assert len(_lit_columns(mask)) < GRID_COLUMNS


def test_must_be_red_the_repository_wide_slit_spelling_lights_every_row():
    """必须红：仓里**每一处**``half_width_y_m = 1.0e9 * pitch``的写法。

    它亮满所有行——**那是缝不是孔**。本条正面把那个写法构造出来并断言
    上一条判据在它身上不成立，于是"S4.6的label里三件只做了两件"
    这句话在仓里有一个**可执行的**证据，而不是一句台账里的话。
    """

    half_x = ((APERTURE_SAMPLES_X - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M
    slit = rectangular_aperture(
        row_count=GRID_ROWS,
        column_count=GRID_COLUMNS,
        pitch_x_m=GRID_PITCH_M,
        pitch_y_m=GRID_PITCH_M,
        half_width_x_m=half_x,
        half_width_y_m=1.0e9 * GRID_PITCH_M,
        centre_x_m=0.5 * GRID_PITCH_M,
    )
    assert len(_lit_rows(slit)) == GRID_ROWS, "这个写法没有亮满所有行，那本条必红失效了"
    assert len(_lit_rows(slit)) != APERTURE_SAMPLES_Y
    assert _lit_samples(slit) == APERTURE_SAMPLES_X * GRID_ROWS


# --- 判据二｜主极大与两个方向各自的零点 -----------------------------------


def test_the_main_maximum_sits_on_the_axis_and_is_the_global_maximum():
    mask, half_x, half_y = _rectangular_hole(APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y)
    intensity = _screen(mask, half_x, half_y).intensity_rows()
    peak = intensity[0][0]
    assert peak == max(max(row) for row in intensity)
    assert peak > 0.0


@pytest.mark.parametrize("order", (1, 2, 3))
def test_the_zeros_of_both_directions_are_exactly_zero(order):
    """x零点在``bin = m COLS / Mx``、y零点在``bin = m ROWS / My``，强度**恰为0**。

    "恰为0"不是"很小"：两个方向的采样孔宽都是2的幂、零点bin都是``N/M``的
    整数倍，蝶形最后一级配对的两项相位差恰为pi（精确取负），IEEE下逐位抵消。
    这条一旦不再成立，说明变换的结构变了——**那正是该红的时候**。
    """

    mask, half_x, half_y = _rectangular_hole(APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y)
    intensity = _screen(mask, half_x, half_y).intensity_rows()
    peak = intensity[0][0]
    bin_x = order * GRID_COLUMNS // APERTURE_SAMPLES_X
    bin_y = order * GRID_ROWS // APERTURE_SAMPLES_Y
    assert intensity[0][bin_x] / peak == APERTURE_ZERO_INTENSITY_FLOOR
    assert intensity[bin_y][0] / peak == APERTURE_ZERO_INTENSITY_FLOOR


def test_each_direction_puts_its_first_zero_on_its_own_closed_form_angle():
    """``sin(theta) = m lambda / w``，`w`是**那个方向的**孔宽。

    **x能验到m=3、y只能验到m=1**，这不是偷懒：本模块的坐标是FFT次序，
    奈奎斯特那一格归到**负**侧（`signed_frequency_indices`的口径）。
    y方向零点间隔是``ROWS/My = 16``格，正侧只到bin 31，
    所以``m=2``那一格恰是奈奎斯特（bin 32）、``m=3``（bin 48）已经折成了``m=-1``。
    本条把这件事**正面断言**出来而不是绕开：bin 32的坐标必须是**负**的、
    模长必须是``2 lambda / w_y``。绕开它就等于把一条会静默错的约定藏起来。
    """

    mask, half_x, half_y = _rectangular_hole(APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y)
    screen = _screen(mask, half_x, half_y)
    width_x = APERTURE_SAMPLES_X * GRID_PITCH_M
    width_y = APERTURE_SAMPLES_Y * GRID_PITCH_M
    xs = screen.coordinates_x_m()
    ys = screen.coordinates_y_m()

    #: 观察面坐标是``k lambda z /(N d)``，在``k = m N / M``处约掉z与N，
    #: **恒等地**落回``m lambda /(M d)``。理论偏差为0，容差只留2 eps给乘除次序。
    for order in (1, 2, 3):
        bin_x = order * GRID_COLUMNS // APERTURE_SAMPLES_X
        sine_x = paraxial_sine_of_angle(xs[bin_x], SCREEN_DISTANCE_M)
        assert abs(sine_x - order * WAVELENGTH_M / width_x) <= 2.0 * EPS

    bin_y = GRID_ROWS // APERTURE_SAMPLES_Y
    sine_y = paraxial_sine_of_angle(ys[bin_y], SCREEN_DISTANCE_M)
    assert abs(sine_y - WAVELENGTH_M / width_y) <= 2.0 * EPS

    nyquist = 2 * GRID_ROWS // APERTURE_SAMPLES_Y
    assert nyquist == GRID_ROWS // 2
    sine_nyquist = paraxial_sine_of_angle(ys[nyquist], SCREEN_DISTANCE_M)
    assert sine_nyquist < 0.0, "奈奎斯特那一格没有归到负侧，坐标出口的约定变了"
    assert abs(abs(sine_nyquist) - 2.0 * WAVELENGTH_M / width_y) <= 2.0 * EPS


def test_must_be_red_swapping_the_two_half_widths_moves_both_sets_of_zeros():
    """必须红：把两个方向的半宽对调（``Mx=4, My=8``）。

    对调之后零点间隔在x方向从``COLS/8 = 16``变成``COLS/4 = 32``、
    在y方向从``ROWS/4 = 16``变成``ROWS/8 = 8``。
    本条不去问"原来的bin还是不是0"（有些bin两边**都**是0，那种判据抓不住对调），
    而是问**两组零点集合真的不同**：
    x的bin 16对真孔是0、对调之后不是；y的bin 8对真孔不是0、对调之后是。
    一个把行列搞混的实现正是这个形态，而它在**方形**孔上完全看不出来——
    这就是本文件的两个方向取8与4而不是取一样的理由。
    """

    true_mask, true_x, true_y = _rectangular_hole(APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y)
    swapped_mask, swap_x, swap_y = _rectangular_hole(APERTURE_SAMPLES_Y, APERTURE_SAMPLES_X)
    true_intensity = _screen(true_mask, true_x, true_y).intensity_rows()
    swapped_intensity = _screen(swapped_mask, swap_x, swap_y).intensity_rows()
    true_peak = true_intensity[0][0]
    swapped_peak = swapped_intensity[0][0]

    bin_x = GRID_COLUMNS // APERTURE_SAMPLES_X
    assert true_intensity[0][bin_x] / true_peak == APERTURE_ZERO_INTENSITY_FLOOR
    assert swapped_intensity[0][bin_x] / swapped_peak > 1.0e-3, "对调之后x零点居然没挪"

    bin_y = GRID_ROWS // APERTURE_SAMPLES_X
    assert true_intensity[bin_y][0] / true_peak > 1.0e-3
    assert swapped_intensity[bin_y][0] / swapped_peak == APERTURE_ZERO_INTENSITY_FLOOR


# --- 判据三｜图样可分离，以及它对连续sinc平方乘积的偏差 -------------------


def test_the_pattern_is_separable_and_a_circular_aperture_proves_the_gate_can_tell():
    """``I(r,c)/I(0,0) = (I(r,0)/I(0,0)) (I(0,c)/I(0,0))``——**图样**的可分离性。

    与`optics/field.py`那条`fft2`可分离性门**不是同一件事**：
    那一条说的是"先行后列与先列后行给同一个答案"，对**任何**输入都成立；
    这一条说的是"图样是两条一维图样的乘积"，**只对可分离孔径成立**。
    圆孔在同一条式子上实测3.5852e-2，比上界大1.0e13倍——
    必红与判据写在同一条测试里，因为它们是同一句话的两半。
    """

    mask, half_x, half_y = _rectangular_hole(APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y)
    intensity = _screen(mask, half_x, half_y).intensity_rows()
    peak = intensity[0][0]
    bound = PATTERN_SEPARABILITY_FACTOR * EPS
    worst = max(
        abs(intensity[row][column] / peak - (intensity[row][0] / peak) * (intensity[0][column] / peak))
        for row in range(GRID_ROWS)
        for column in range(GRID_COLUMNS)
    )
    assert worst <= bound, f"矩孔图样的可分离性残差{worst!r}超出{bound!r}"

    #: 必红：圆孔不可分离。用同一条式子、同一个网格、同一个传播器。
    radius = (6.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M
    disc = circular_aperture(
        row_count=GRID_ROWS,
        column_count=GRID_COLUMNS,
        pitch_x_m=GRID_PITCH_M,
        pitch_y_m=GRID_PITCH_M,
        radius_m=radius,
    )
    disc_intensity = _screen(disc, radius, radius).intensity_rows()
    disc_peak = disc_intensity[0][0]
    disc_worst = max(
        abs(
            disc_intensity[row][column] / disc_peak
            - (disc_intensity[row][0] / disc_peak) * (disc_intensity[0][column] / disc_peak)
        )
        for row in range(GRID_ROWS)
        for column in range(GRID_COLUMNS)
    )
    assert disc_worst > bound, "这条门分不开可分离与不可分离的孔径，判据要重写"


def _sinc_product_deviation(samples_x, samples_y, rows, columns, pitch):
    """图样对连续``sinc^2(x) sinc^2(y)``的最大偏差（归一化强度）。"""

    mask, half_x, half_y = _rectangular_hole(
        samples_x, samples_y, rows=rows, columns=columns, pitch=pitch
    )
    screen = _screen(mask, half_x, half_y, pitch=pitch)
    intensity = screen.intensity_rows()
    peak = intensity[0][0]
    width_x = samples_x * pitch
    width_y = samples_y * pitch
    xs = screen.coordinates_x_m()
    ys = screen.coordinates_y_m()
    worst = 0.0
    for row in range(rows):
        sine_y = paraxial_sine_of_angle(ys[row], SCREEN_DISTANCE_M)
        envelope_y = normalised_sinc(width_y * sine_y / WAVELENGTH_M)
        for column in range(columns):
            sine_x = paraxial_sine_of_angle(xs[column], SCREEN_DISTANCE_M)
            want = (normalised_sinc(width_x * sine_x / WAVELENGTH_M) * envelope_y) ** 2
            worst = max(worst, abs(intensity[row][column] / peak - want))
    return worst


def test_the_profile_follows_the_product_of_two_continuous_sinc_squared():
    """整幅图样对``sinc^2(x) sinc^2(y)``的偏差在申报上界内。

    偏差由**较粗的那个方向**（My=4）主导，是Dirichlet核与sinc的差，
    不是实现误差。上界3e-2是实测2.8192e-2的1.06倍。
    """

    worst = _sinc_product_deviation(
        APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y, GRID_ROWS, GRID_COLUMNS, GRID_PITCH_M
    )
    assert worst <= SINC_PRODUCT_DEVIATION_AT_BASE_GRID, f"剖面偏差{worst!r}"
    #: 申报不许虚胖：实测比申报小一个数量级以上的话，下游容差会跟着虚胖。
    assert worst >= 0.1 * SINC_PRODUCT_DEVIATION_AT_BASE_GRID


def test_the_two_dimensional_pattern_converges_to_the_continuous_product_at_second_order():
    """固定物理孔径（80微米×40微米）加密网格，偏差按``O(h^2)``降。

    **两个方向一起加密**，所以它验的是二维的收敛而不是把一维的结论抄一遍。
    实测2.8192e-2 → 7.9721e-3 → 2.1429e-3，阶1.822与1.895。
    只跑前三档：第四档（512×1024）单次约1.2秒，会把本文件推出交互级。
    """

    deviations = [
        _sinc_product_deviation(
            APERTURE_SAMPLES_X * factor,
            APERTURE_SAMPLES_Y * factor,
            GRID_ROWS * factor,
            GRID_COLUMNS * factor,
            GRID_PITCH_M / factor,
        )
        for factor in (1, 2, 4)
    ]
    orders = [
        math.log(coarse / fine, 2.0)
        for coarse, fine in zip(deviations[:-1], deviations[1:], strict=True)
    ]
    assert all(order >= SINC_PRODUCT_MIN_ORDER for order in orders), (
        f"二维收敛阶{orders!r}低于申报下界{SINC_PRODUCT_MIN_ORDER}，实测偏差{deviations!r}"
    )


def test_must_be_red_dropping_the_y_envelope_leaves_an_order_one_error():
    """必须红：把参照写成只有``sinc^2(x)``（**忘掉y那一半**）。

    这正是"缝当成孔"在判据一侧的形态：一维参照对一维图样（占满窗口的缝）
    是对的，对真的矩孔差``O(1)``。实测最大偏差**1.0000**
    （y方向包络的第一个零点上，真图样是0而一维参照仍给1）。
    """

    mask, half_x, half_y = _rectangular_hole(APERTURE_SAMPLES_X, APERTURE_SAMPLES_Y)
    screen = _screen(mask, half_x, half_y)
    intensity = screen.intensity_rows()
    peak = intensity[0][0]
    width_x = APERTURE_SAMPLES_X * GRID_PITCH_M
    xs = screen.coordinates_x_m()
    worst = max(
        abs(
            intensity[row][column] / peak
            - normalised_sinc(
                width_x * paraxial_sine_of_angle(xs[column], SCREEN_DISTANCE_M) / WAVELENGTH_M
            )
            ** 2
        )
        for row in range(GRID_ROWS)
        for column in range(GRID_COLUMNS)
    )
    assert worst > SINC_PRODUCT_DEVIATION_AT_BASE_GRID, (
        "丢掉y方向的包络居然还在申报上界内——那说明这个构型的y方向没有被限制，"
        "即它又是一条缝而不是孔"
    )


# --- 判据四｜单缝是矩孔的极限（一个可以判的数） ---------------------------


def _off_axis_power_fraction(samples_y, *, rows=SLIT_LIMIT_ROWS):
    """离轴行占的功率份额``sum_{r>0} I / sum_{r=0} I``。

    闭式是``rows / My - 1``：y方向亮`My`个采样的常值列，
    其长度`rows`的DFT在bin 0上是``My``、全谱平方和是``rows * My``
    （Parseval），于是离轴份额恰是``rows/My - 1``。
    `My = rows`时它**恰好是0**——那就是"单缝是矩孔的极限"这句话的数。
    """

    half_x = ((APERTURE_SAMPLES_X - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M
    if samples_y >= rows:
        half_y = (rows / 2.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M
        centre_y = 0.0
    else:
        half_y = ((samples_y - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M
        centre_y = 0.5 * GRID_PITCH_M
    mask = rectangular_aperture(
        row_count=rows,
        column_count=GRID_COLUMNS,
        pitch_x_m=GRID_PITCH_M,
        pitch_y_m=GRID_PITCH_M,
        half_width_x_m=half_x,
        half_width_y_m=half_y,
        centre_x_m=0.5 * GRID_PITCH_M,
        centre_y_m=centre_y,
    )
    screen = propagate_fraunhofer(
        mask,
        wavelength_m=WAVELENGTH_M,
        distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M,
        pitch_y_m=GRID_PITCH_M,
        aperture_half_width_m=math.hypot(half_x, half_y),
    )
    intensity = screen.intensity_rows()
    return mask, sum(sum(row) for row in intensity[1:]) / sum(intensity[0])


def test_growing_the_hole_to_fill_the_window_is_bit_for_bit_the_slit_spelling():
    """y半宽长到盖住窗口时，掩模与仓里那个``1.0e9 * pitch``的写法**逐位相同**。

    这条是**零容差**的（`float.hex()`逐字节）：单缝不是矩孔的近似，
    是它在``My -> ROWS``上的极限，而这两个写法在浮点上是同一串数。
    """

    half_x = ((APERTURE_SAMPLES_X - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M
    grown = rectangular_aperture(
        row_count=SLIT_LIMIT_ROWS,
        column_count=GRID_COLUMNS,
        pitch_x_m=GRID_PITCH_M,
        pitch_y_m=GRID_PITCH_M,
        half_width_x_m=half_x,
        half_width_y_m=(SLIT_LIMIT_ROWS / 2.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M,
        centre_x_m=0.5 * GRID_PITCH_M,
    )
    legacy = rectangular_aperture(
        row_count=SLIT_LIMIT_ROWS,
        column_count=GRID_COLUMNS,
        pitch_x_m=GRID_PITCH_M,
        pitch_y_m=GRID_PITCH_M,
        half_width_x_m=half_x,
        half_width_y_m=1.0e9 * GRID_PITCH_M,
        centre_x_m=0.5 * GRID_PITCH_M,
    )
    for got, want in zip(grown.values(), legacy.values(), strict=True):
        assert got.real.hex() == want.real.hex()
        assert got.imag.hex() == want.imag.hex()


def test_the_slit_is_the_limit_of_the_hole_and_that_is_a_number_not_a_sentence():
    """``My = ROWS``时离轴行的功率份额**恰为0**；有界的`My`恰为``ROWS/My - 1``。

    必红与判据在同一条测试里：如果有界的`My`也给0，
    那说明y方向根本没被限制——即它又是一条缝。
    """

    _, filled = _off_axis_power_fraction(SLIT_LIMIT_ROWS)
    assert filled == SLIT_LIMIT_OFF_AXIS_POWER

    for samples_y in (2, 4):
        _, fraction = _off_axis_power_fraction(samples_y)
        expected = SLIT_LIMIT_ROWS / samples_y - 1.0
        #: **零容差**：实测My=2给`3.0`、My=4给`1.0`，与闭式逐位相同。
        #: 这不是乐观——两侧都是2的幂上的Parseval配平，IEEE下没有余数。
        assert fraction == expected, (
            f"My={samples_y}：离轴份额{fraction!r}对不上闭式{expected!r}"
        )
        assert fraction != SLIT_LIMIT_OFF_AXIS_POWER


# --- 边界：孔径构件自己的失败关闭在y方向上也要在 ---------------------------


def test_the_edge_ambiguity_gate_also_watches_the_y_direction():
    """0086那条"边界落在采样点上失败关闭"在y方向上同样有效。

    仓里既有的调用全部让y方向占满窗口（半宽``1e9 * pitch``远在窗口外，
    那一条判据**直接return**），所以**y方向那一支从来没有被走到过**。
    这里正面走一次。
    """

    half_x = ((APERTURE_SAMPLES_X - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * GRID_PITCH_M
    with pytest.raises(OpticsError, match="矩孔的y边界"):
        rectangular_aperture(
            row_count=GRID_ROWS,
            column_count=GRID_COLUMNS,
            pitch_x_m=GRID_PITCH_M,
            pitch_y_m=GRID_PITCH_M,
            half_width_x_m=half_x,
            half_width_y_m=2.0 * GRID_PITCH_M,
            centre_x_m=0.5 * GRID_PITCH_M,
        )
