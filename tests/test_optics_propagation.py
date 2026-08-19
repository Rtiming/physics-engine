"""`optics/propagation.py`的物理门——六条判据加它们各自的必须红。

plans/17轨乙给的六条，逐条在这里：

1. **单缝夫琅禾费`sinc^2`**：主极大与前三个零点位置；
2. **双缝条纹间距**闭式，外加缺级；
3. **菲涅耳数极限退化到艾里斑**（`diffraction.py`那条现成闭式是金标）；
4. **`z -> 0`退化回入射场**；
5. **半群**：传两次`z/2`等于一次`z`；
6. **采样判据**：角谱与菲涅耳各自的适用域显式声明，越界失败关闭。

另加三条本文件自己认为更值钱的：

* **平面波本征函数**——角谱的传递函数逐点对闭式，**它是唯一能把"精确角谱"
  与"傍轴近似"分开的判据**（半群、能量、退化三条对两者一视同仁）；
* **能量守恒**——菲涅耳前因子``dx dy/(i lambda z)``写错时图样形状完全不变，
  只有总能量会露馅；
* **孔径边界落在采样点上要失败关闭**——这条是本轨道抓到的**真缺陷**，
  第一版实测在``dx=2.5um``上把32个采样的缝算成31个，
  "收敛阶"随之变成负数（-4.22）而图样照样是漂亮的sinc平方。

容差全部算出来或实测出来，逐条写在常量的注释里。
"""

from __future__ import annotations

import cmath
import math
import sys

import pytest

import physics_engine
from physics_engine.optics.diffraction import (
    airy_argument,
    airy_intensity,
    angular_wavenumber_rad_per_m,
)
from physics_engine.optics.errors import OpticsError
from physics_engine.optics.field import ComplexField2D, fft2, ifft2
from physics_engine.optics.fts import normalised_sinc
from physics_engine.optics.propagation import (
    APERTURE_EDGE_AMBIGUITY_TOLERANCE,
    FRAUNHOFER_MAX_EDGE_PHASE_RAD,
    circular_aperture,
    fraunhofer_max_fresnel_number,
    fresnel_number,
    incident_power,
    paraxial_sine_of_angle,
    propagate_angular_spectrum,
    propagate_fraunhofer,
    propagate_fresnel,
    rectangular_aperture,
    spatial_coordinates_m,
    transfer_function_max_distance_m,
)

EPS = sys.float_info.epsilon

#: HeNe。与`cases/scalar_diffraction_airy`同一个波长，方便两条案例互相印证。
WAVELENGTH_M = 632.8e-9

#: 单缝／双缝共用的网格：256列×8行、10微米间距、观察距离2米。
#: 行数取8而不是1，是为了让二维路径真的被走到（一维退化路径盖不住行列混淆）；
#: 缝在y方向占满窗口，于是谱只在``fy = 0``那一行——**实测其余7行的强度和恰为0**
#: （常值列的长度8 DFT在非零bin上成对相消，是精确的）。
GRID_COLUMNS = 256
GRID_ROWS = 8
GRID_PITCH_M = 10.0e-6
SCREEN_DISTANCE_M = 2.0

#: 缝宽以**采样数**计。取2的幂是有理由的：夫琅禾费零点落在
#: ``bin = m N / M``上，`M`是2的幂时那正好是整数bin，可以**零容差**地判。
SLIT_SAMPLES = 8

#: 双缝：每缝4个采样、中心距32个采样。``P / M = 8``于是**第8级缺级**
#: （包络零点恰好压在条纹极大上）——这是双缝案例里最难蒙对的一条判据。
DOUBLE_SLIT_SAMPLES = 4
DOUBLE_SLIT_SEPARATION_SAMPLES = 32

#: 边界不许落在采样点上（见`APERTURE_EDGE_AMBIGUITY_TOLERANCE`），
#: 所以半宽一律偏四分之一格。偏多少不影响**采样缝宽**（仍是M个采样），
#: 只是把"这个点归谁"这个问题变得没有歧义。
EDGE_OFFSET_IN_PITCHES = 0.25

#: 夫琅禾费零点处的归一化强度**申报为恰好0**。
#: 不是"很小"是"恰好"：缝宽`M`是2的幂、零点bin是``N/M``的整数倍时，
#: 蝶形最后一级配对的两项相位差恰为pi（即精确取负），IEEE下逐位抵消。
#: 实测三个零点全部返回`0.0`。这条一旦不再成立，说明变换的结构变了——
#: **那正是该红的时候**。
SLIT_ZERO_INTENSITY_FLOOR = 0.0

#: 剖面对连续`sinc^2`的偏差申报：`M=8`时**7.98e-3**。
#: 它不是实现误差，是**Dirichlet核与sinc的差**（离散孔径 对 连续孔径）：
#: 固定物理缝宽80微米加密网格，实测
#: M=8→7.9721e-3、16→1.4349e-3、32→3.3335e-4、64→8.1863e-5，
#: 阶2.474→2.106→**2.026**，即``O(h^2)``。取1e-2是M=8实测的1.25倍。
SINC_PROFILE_DEVIATION_AT_EIGHT_SAMPLES = 1.0e-2
SINC_PROFILE_MEASURED_AT_EIGHT_SAMPLES = 7.9721e-3

#: 平面波本征函数的容差：``tol = 16 * N * eps``。实测最坏比值3.09
#: （N=64、8个不同的空间频率bin），取16是5.2倍。
#: 比值随bin增大而增大，因为传递函数相位``k z sqrt(...)``本身在增大。
PLANE_WAVE_FACTOR = 16.0
PLANE_WAVE_MEASURED_RATIO = 3.09

#: 半群残差的容差：``tol = 64 * eps * 峰值振幅``。实测比值2.14。
#:
#: 这个数**依赖平台的三角函数实现**：传递函数相位在本例里是``k z ~ 1.0e5``弧度，
#: 若`libm`的幅角约化只做简单取模，``cos``的误差会放大到``eps * 1e5 ~ 2e-11``，
#: 这条门会当场红。本机（macOS/arm64/CPython 3.13）的`libm`做的是精确约化，
#: 所以残差停在``eps``量级。**如实写在这里**：它红了先查平台，不要先放宽。
SEMIGROUP_FACTOR = 64.0
SEMIGROUP_MEASURED_RATIO = 2.14

#: `z=0`退化：与入射场的最大偏差``<= 4 * eps * max|U0|``。实测4.48e-16 = 2.02 eps。
ZERO_DISTANCE_FACTOR = 4.0

#: 能量守恒的相对容差。实测：角谱**恰为0**、菲涅耳3.33e-16。取1e-14。
POWER_CONSERVATION_TOLERANCE = 1.0e-14

#: 圆孔阶梯边导致的归一化强度偏差上界（对`diffraction.py`的艾里闭式）。
#:
#: **它不随网格干净地降**，如实记：固定物理孔径（半径320微米）加密网格，
#: 实测R=16→1.5101e-3、R=32→1.4691e-3、R=64→1.9796e-4，
#: 三点的"阶"是0.040与2.892。原因是笛卡尔格上圆内格点数的涨落
#: （Gauss圆问题那一类），**误差是振荡的不是单调的**。
#: 所以本文件只断两件事：每个分辨率都在申报上界内、且加密到R=64时真的更好。
#: 假装它是``O(h^2)``会得到一条随机红的门。
AIRY_STAIRCASE_DEVIATION = 3.0e-3

#: 夫琅禾费在申报门槛处丢掉的那一项的实测大小（归一化强度的最大差）。
FRAUNHOFER_EDGE_PHASE_MEASURED_DEVIATION = 1.2013e-4


# --- 构件 -----------------------------------------------------------------


def _slit(samples, *, centre_offset_samples=0.0, columns=GRID_COLUMNS,
          rows=GRID_ROWS, pitch=GRID_PITCH_M):
    """`samples`个采样宽的缝，y方向占满窗口。中心偏半格以取到**偶数**个采样。"""

    half = ((samples - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES) * pitch
    centre = (0.5 + centre_offset_samples) * pitch
    return rectangular_aperture(
        row_count=rows,
        column_count=columns,
        pitch_x_m=pitch,
        pitch_y_m=pitch,
        half_width_x_m=half,
        half_width_y_m=1.0e9 * pitch,
        centre_x_m=centre,
    ), half


def _lit_samples(mask):
    return sum(1 for value in mask.rows[0] if abs(value) > 0.5)


def _union(left, right):
    """两块不重叠的0/1掩模求并。不重叠由调用点断言（最大值仍是1）。"""

    return ComplexField2D(
        tuple(
            tuple(a + b for a, b in zip(row_a, row_b, strict=True))
            for row_a, row_b in zip(left.rows, right.rows, strict=True)
        )
    )


def _plane_wave(bin_index, *, count, pitch):
    coordinates = spatial_coordinates_m(count, pitch)
    frequency = bin_index / (count * pitch)
    return ComplexField2D.from_function(
        1,
        count,
        lambda row, column: cmath.exp(
            complex(0.0, 2.0 * math.pi * frequency * coordinates[column])
        ),
    ), frequency


# --- 判据六｜采样判据与两个形制的互补适用域 -------------------------------


def test_the_two_methods_share_one_boundary_and_it_is_the_declared_formula():
    """**方阵上**两个适用域在同一个`z_c`上无缝交接：一个到此为止，另一个从此开始。"""

    count, pitch = 64, GRID_PITCH_M
    limit = transfer_function_max_distance_m(
        count=count, pitch_m=pitch, wavelength_m=WAVELENGTH_M
    )
    assert limit == count * pitch**2 / WAVELENGTH_M
    mask = circular_aperture(
        row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
        radius_m=(8.0 + EDGE_OFFSET_IN_PITCHES) * pitch,
    )
    kwargs = {"wavelength_m": WAVELENGTH_M, "pitch_x_m": pitch, "pitch_y_m": pitch}
    propagate_angular_spectrum(mask, distance_m=limit, **kwargs)
    propagate_fresnel(mask, distance_m=limit, **kwargs)
    with pytest.raises(OpticsError, match="角谱在z="):
        propagate_angular_spectrum(mask, distance_m=limit * 1.001, **kwargs)
    with pytest.raises(OpticsError, match="单次FFT形制在z="):
        propagate_fresnel(mask, distance_m=limit * 0.999, **kwargs)


def test_an_anisotropic_grid_opens_a_gap_and_both_methods_refuse_inside_it():
    """两轴不等长时上界取`min`、下界取`max`，中间那一段**两个形制都拒答**。

    这不是实现的漏：``256 x 8``的网格在y方向只有8个采样宽的窗口，
    传到那段距离上衍射早就跑出窗口了。**如实拒答比给一个漂亮的错图样好**。
    实测这条缝是``[1.264e-3, 4.046e-2]``米，32倍宽。
    """

    mask, _ = _slit(SLIT_SAMPLES)
    upper = transfer_function_max_distance_m(
        count=GRID_ROWS, pitch_m=GRID_PITCH_M, wavelength_m=WAVELENGTH_M
    )
    lower = transfer_function_max_distance_m(
        count=GRID_COLUMNS, pitch_m=GRID_PITCH_M, wavelength_m=WAVELENGTH_M
    )
    assert upper < lower, "这个网格不是扁的，本条门测不到东西"
    inside_the_gap = math.sqrt(upper * lower)
    kwargs = {"wavelength_m": WAVELENGTH_M, "pitch_x_m": GRID_PITCH_M,
              "pitch_y_m": GRID_PITCH_M}
    with pytest.raises(OpticsError, match="角谱在z="):
        propagate_angular_spectrum(mask, distance_m=inside_the_gap, **kwargs)
    with pytest.raises(OpticsError, match="单次FFT形制在z="):
        propagate_fresnel(mask, distance_m=inside_the_gap, **kwargs)
    #: 缝的两侧各自可用。
    propagate_angular_spectrum(mask, distance_m=upper, **kwargs)
    propagate_fresnel(mask, distance_m=lower, **kwargs)


def test_the_refusal_says_which_method_to_use_instead():
    """失败关闭要**指路**，否则下一个人只会把判据删掉。"""

    mask, _ = _slit(SLIT_SAMPLES)
    limit = transfer_function_max_distance_m(
        count=GRID_COLUMNS, pitch_m=GRID_PITCH_M, wavelength_m=WAVELENGTH_M
    )
    with pytest.raises(OpticsError) as caught:
        propagate_angular_spectrum(
            mask, wavelength_m=WAVELENGTH_M, distance_m=10.0 * limit,
            pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        )
    assert "propagate_fresnel" in str(caught.value)
    with pytest.raises(OpticsError) as caught:
        propagate_fresnel(
            mask, wavelength_m=WAVELENGTH_M, distance_m=0.1 * limit,
            pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        )
    assert "propagate_angular_spectrum" in str(caught.value)


def test_an_axis_with_a_single_sample_does_not_constrain_anything():
    """一行N列的场就是一维问题：y轴只有一个采样，没有频率内容可混叠。"""

    single = ComplexField2D.from_function(1, 64, lambda row, column: complex(1.0, 0.0))
    limit = transfer_function_max_distance_m(
        count=64, pitch_m=GRID_PITCH_M, wavelength_m=WAVELENGTH_M
    )
    propagate_angular_spectrum(
        single, wavelength_m=WAVELENGTH_M, distance_m=limit,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
    )
    #: 若y轴也进了判据，下界会被那条``1 * dy^2 / lambda``压成极小的数，
    #: 菲涅耳就会在几乎任何z上都"可用"——这条门守的正是那个漏。
    with pytest.raises(OpticsError, match="单次FFT形制在z="):
        propagate_fresnel(
            single, wavelength_m=WAVELENGTH_M, distance_m=limit * 0.5,
            pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        )


@pytest.mark.parametrize("bad", (-1.0e-6, -1.0))
def test_backward_propagation_fails_closed(bad):
    mask, _ = _slit(SLIT_SAMPLES)
    with pytest.raises(OpticsError, match="传播距离必须是有限非负数"):
        propagate_angular_spectrum(
            mask, wavelength_m=WAVELENGTH_M, distance_m=bad,
            pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        )


def test_the_single_fft_forms_refuse_zero_distance():
    """观察面间距是``lambda z /(N dx)``，`z=0`时它是0——那不是一个网格。"""

    mask, half = _slit(SLIT_SAMPLES)
    with pytest.raises(OpticsError, match="传播距离必须是有限正数"):
        propagate_fresnel(
            mask, wavelength_m=WAVELENGTH_M, distance_m=0.0,
            pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        )
    with pytest.raises(OpticsError, match="传播距离必须是有限正数"):
        propagate_fraunhofer(
            mask, wavelength_m=WAVELENGTH_M, distance_m=0.0,
            pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
            aperture_half_width_m=half,
        )


# --- 角谱：平面波本征函数与倏逝波 -----------------------------------------


@pytest.mark.parametrize("bin_index", (0, 1, 3, 7, 15, 31, -1, -5))
def test_a_plane_wave_only_picks_up_the_closed_form_phase(bin_index):
    """角谱的本征函数判据——**唯一能把精确角谱与傍轴近似分开的一条**。"""

    count, pitch = 64, 5.0e-6
    distance = 0.5 * transfer_function_max_distance_m(
        count=count, pitch_m=pitch, wavelength_m=WAVELENGTH_M
    )
    wave, frequency = _plane_wave(bin_index, count=count, pitch=pitch)
    result = propagate_angular_spectrum(
        wave, wavelength_m=WAVELENGTH_M, distance_m=distance,
        pitch_x_m=pitch, pitch_y_m=pitch,
    )
    wavenumber = angular_wavenumber_rad_per_m(WAVELENGTH_M)
    expected = cmath.exp(
        complex(
            0.0,
            wavenumber * distance * math.sqrt(1.0 - (WAVELENGTH_M * frequency) ** 2),
        )
    )
    bound = PLANE_WAVE_FACTOR * count * EPS
    worst = max(
        abs(got - want * expected)
        for got, want in zip(result.field.rows[0], wave.rows[0], strict=True)
    )
    assert worst <= bound, f"bin={bin_index}：平面波相位残差{worst!r}超出{bound!r}"


def test_must_be_red_the_paraxial_transfer_function_fails_the_plane_wave_gate():
    """把``sqrt(1-q)``换成傍轴的``1-q/2``——半群、能量、退化三条门都抓不到它。

    实测相位差：bin=15处1.22e-3弧度、bin=31处2.22e-2弧度，
    而本条门的上界是``16*64*eps = 2.3e-13``。差十个量级。
    """

    count, pitch = 64, 5.0e-6
    distance = 0.5 * transfer_function_max_distance_m(
        count=count, pitch_m=pitch, wavelength_m=WAVELENGTH_M
    )
    wavenumber = angular_wavenumber_rad_per_m(WAVELENGTH_M)
    for bin_index in (15, 31):
        _, frequency = _plane_wave(bin_index, count=count, pitch=pitch)
        reduced = (WAVELENGTH_M * frequency) ** 2
        exact = wavenumber * distance * math.sqrt(1.0 - reduced)
        paraxial = wavenumber * distance * (1.0 - 0.5 * reduced)
        gap = abs(cmath.exp(complex(0.0, exact)) - cmath.exp(complex(0.0, paraxial)))
        assert gap > PLANE_WAVE_FACTOR * count * EPS, (
            f"bin={bin_index}：傍轴与精确差{gap!r}，居然进得了本条门的容差——"
            "那说明这条门分不开两者，判据要重写"
        )


@pytest.mark.parametrize("bin_index", (6, 8, 12, 16))
def test_evanescent_components_decay_by_the_closed_form_factor(bin_index):
    """``(lambda f)^2 > 1``的分量按``exp(-k z sqrt(q-1))``衰减——不丢也不涨。"""

    count, pitch = 32, 1.0e-7
    distance = 0.4 * transfer_function_max_distance_m(
        count=count, pitch_m=pitch, wavelength_m=WAVELENGTH_M
    )
    wave, frequency = _plane_wave(bin_index, count=count, pitch=pitch)
    reduced = (WAVELENGTH_M * frequency) ** 2
    assert reduced > 1.0, "这个bin不是倏逝的，判据选错了采样间距"
    wavenumber = angular_wavenumber_rad_per_m(WAVELENGTH_M)
    expected = math.exp(-wavenumber * distance * math.sqrt(reduced - 1.0))
    assert 0.0 < expected < 1.0
    result = propagate_angular_spectrum(
        wave, wavelength_m=WAVELENGTH_M, distance_m=distance,
        pitch_x_m=pitch, pitch_y_m=pitch,
    )
    worst = max(
        abs(got - want * expected)
        for got, want in zip(result.field.rows[0], wave.rows[0], strict=True)
    )
    assert worst <= PLANE_WAVE_FACTOR * count * EPS, f"倏逝衰减残差{worst!r}"


# --- 判据四｜z -> 0退化回入射场 -------------------------------------------


def test_zero_distance_is_bit_for_bit_the_pure_round_trip():
    """`z=0`时传递函数逐位等于1，于是结果与``ifft2(fft2(U0))``**逐位相同**。

    这条是**零容差**的：不是"很接近"，是同一串浮点数。
    传递函数里多算一步（例如把``exp(0)``写成``1+i*0``之外的什么）当场露。
    """

    mask, _ = _slit(SLIT_SAMPLES)
    result = propagate_angular_spectrum(
        mask, wavelength_m=WAVELENGTH_M, distance_m=0.0,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
    )
    pure = ifft2(fft2(mask))
    for got, want in zip(result.field.values(), pure.values(), strict=True):
        assert got.real.hex() == want.real.hex()
        assert got.imag.hex() == want.imag.hex()


def test_zero_distance_returns_the_incident_field_to_machine_precision():
    mask, _ = _slit(SLIT_SAMPLES)
    result = propagate_angular_spectrum(
        mask, wavelength_m=WAVELENGTH_M, distance_m=0.0,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
    )
    scale = max(abs(value) for value in mask.values())
    worst = max(
        abs(got - want)
        for got, want in zip(result.field.values(), mask.values(), strict=True)
    )
    assert worst <= ZERO_DISTANCE_FACTOR * EPS * scale, f"z=0残差{worst!r}"


# --- 判据五｜半群 ---------------------------------------------------------


@pytest.mark.parametrize("fraction", (0.5, 0.9))
def test_two_half_steps_equal_one_whole_step(fraction):
    count, pitch = 128, GRID_PITCH_M
    distance = fraction * transfer_function_max_distance_m(
        count=count, pitch_m=pitch, wavelength_m=WAVELENGTH_M
    )
    mask = circular_aperture(
        row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
        radius_m=(16.0 + EDGE_OFFSET_IN_PITCHES) * pitch,
    )
    kwargs = {"wavelength_m": WAVELENGTH_M, "pitch_x_m": pitch, "pitch_y_m": pitch}
    once = propagate_angular_spectrum(mask, distance_m=distance, **kwargs)
    first = propagate_angular_spectrum(mask, distance_m=distance / 2.0, **kwargs)
    twice = propagate_angular_spectrum(first.field, distance_m=distance / 2.0, **kwargs)
    peak = max(abs(value) for value in once.field.values())
    bound = SEMIGROUP_FACTOR * EPS * peak
    worst = max(
        abs(a - b)
        for a, b in zip(once.field.values(), twice.field.values(), strict=True)
    )
    assert worst <= bound, (
        f"z={distance!r}：半群残差{worst!r}超出{bound!r}。"
        "先查平台的三角函数幅角约化（本门的注释写了为什么）"
    )


def test_must_be_red_a_propagator_that_mishandles_the_distance_breaks_the_semigroup():
    """必须红：把两段各`z/2`换成两段各`z`（"忘了折半"最常见的形态）。"""

    count, pitch = 128, GRID_PITCH_M
    distance = 0.5 * transfer_function_max_distance_m(
        count=count, pitch_m=pitch, wavelength_m=WAVELENGTH_M
    )
    mask = circular_aperture(
        row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
        radius_m=(16.0 + EDGE_OFFSET_IN_PITCHES) * pitch,
    )
    kwargs = {"wavelength_m": WAVELENGTH_M, "pitch_x_m": pitch, "pitch_y_m": pitch}
    once = propagate_angular_spectrum(mask, distance_m=distance, **kwargs)
    wrong = propagate_angular_spectrum(
        propagate_angular_spectrum(mask, distance_m=distance, **kwargs).field,
        distance_m=distance,
        **kwargs,
    )
    peak = max(abs(value) for value in once.field.values())
    worst = max(
        abs(a - b)
        for a, b in zip(once.field.values(), wrong.field.values(), strict=True)
    )
    assert worst > SEMIGROUP_FACTOR * EPS * peak, "半群门抓不住走错距离的传播器"


# --- 能量守恒（前因子的唯一捕手） -----------------------------------------


def test_the_angular_spectrum_conserves_power():
    mask, _ = _slit(SLIT_SAMPLES)
    limit = transfer_function_max_distance_m(
        count=GRID_ROWS, pitch_m=GRID_PITCH_M, wavelength_m=WAVELENGTH_M
    )
    result = propagate_angular_spectrum(
        mask, wavelength_m=WAVELENGTH_M, distance_m=0.5 * limit,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
    )
    before = incident_power(mask, pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M)
    assert abs(result.total_power() / before - 1.0) <= POWER_CONSERVATION_TOLERANCE


def test_the_fresnel_prefactor_conserves_power_and_that_is_the_only_gate_that_sees_it():
    """前因子``exp(ikz)/(i lambda z) * dx dy``写错，图样形状**一点不变**。

    这条门与"剖面对sinc平方"不重复：后者比的是归一化强度，
    任何常数因子都被除掉了。
    """

    mask, half = _slit(SLIT_SAMPLES)
    result = propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M, aperture_half_width_m=half,
    )
    before = incident_power(mask, pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M)
    assert abs(result.total_power() / before - 1.0) <= POWER_CONSERVATION_TOLERANCE
    #: 必须红：漏掉``dx dy``的前因子——能量差``(dx dy)^2``倍，图样一模一样。
    dropped = result.total_power() / (GRID_PITCH_M * GRID_PITCH_M) ** 2
    assert abs(dropped / before - 1.0) > POWER_CONSERVATION_TOLERANCE


# --- 判据一｜单缝夫琅禾费 -------------------------------------------------


def test_the_single_slit_main_maximum_sits_on_the_axis():
    mask, half = _slit(SLIT_SAMPLES)
    assert _lit_samples(mask) == SLIT_SAMPLES
    result = propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M, aperture_half_width_m=half,
    )
    intensity = result.intensity_rows()
    #: 缝在y方向占满窗口 → 谱只在fy=0那一行，其余行**恰为0**。
    assert sum(sum(row) for row in intensity[1:]) == 0.0
    row = intensity[0]
    assert row[0] == max(row), "主极大不在轴上"
    assert row[0] > 0.0


@pytest.mark.parametrize("order", (1, 2, 3))
def test_the_first_three_zeros_land_exactly_where_the_closed_form_says(order):
    """前三个零点：位置由``sin(theta) = m lambda / w``给定，强度**恰为0**。"""

    mask, half = _slit(SLIT_SAMPLES)
    result = propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M, aperture_half_width_m=half,
    )
    row = result.intensity_rows()[0]
    coordinates = result.coordinates_x_m()
    index = order * GRID_COLUMNS // SLIT_SAMPLES
    assert row[index] / row[0] <= SLIT_ZERO_INTENSITY_FLOOR, (
        f"第{order}个零点的归一化强度{row[index] / row[0]!r}不是恰好0"
    )
    #: **采样缝宽是``M dx``**，不是几何半宽的两倍——两者可以差半格以内，
    #: 闭式要用前者（离散孔径的傅里叶变换对应的就是它）。
    width = SLIT_SAMPLES * GRID_PITCH_M
    sine = paraxial_sine_of_angle(coordinates[index], SCREEN_DISTANCE_M)
    closed_form = order * WAVELENGTH_M / width
    assert abs(sine - closed_form) <= 4.0 * EPS * abs(closed_form), (
        f"第{order}个零点的sin(theta)={sine!r}，闭式{closed_form!r}"
    )


def test_the_profile_follows_the_continuous_sinc_squared():
    mask, half = _slit(SLIT_SAMPLES)
    result = propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M, aperture_half_width_m=half,
    )
    row = result.intensity_rows()[0]
    coordinates = result.coordinates_x_m()
    width = SLIT_SAMPLES * GRID_PITCH_M
    worst = 0.0
    for index in range(1, GRID_COLUMNS // 2):
        sine = paraxial_sine_of_angle(coordinates[index], SCREEN_DISTANCE_M)
        reduced = width * sine / WAVELENGTH_M
        if abs(reduced) > 4.0:
            continue
        worst = max(worst, abs(row[index] / row[0] - normalised_sinc(reduced) ** 2))
    assert worst <= SINC_PROFILE_DEVIATION_AT_EIGHT_SAMPLES, f"剖面偏差{worst!r}"
    assert worst >= 0.5 * SINC_PROFILE_MEASURED_AT_EIGHT_SAMPLES, (
        f"偏差{worst!r}比申报的实测值小一半以上——"
        "要么实现变好了（该重测申报），要么这条门没真的比到东西"
    )


def test_the_discrete_pattern_converges_to_the_continuous_sinc_at_second_order():
    """固定物理缝宽加密网格：Dirichlet核 → sinc，实测阶趋于2。"""

    width = SLIT_SAMPLES * GRID_PITCH_M
    deviations = []
    for samples, columns in ((8, 256), (16, 512), (32, 1024), (64, 2048)):
        pitch = width / samples
        mask, half = _slit(samples, columns=columns, pitch=pitch)
        assert _lit_samples(mask) == samples, "采样缝宽不是期望值——边界又落到采样点上了"
        result = propagate_fraunhofer(
            mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
            pitch_x_m=pitch, pitch_y_m=pitch, aperture_half_width_m=half,
        )
        row = result.intensity_rows()[0]
        coordinates = result.coordinates_x_m()
        worst = 0.0
        for index in range(1, columns // 2):
            sine = paraxial_sine_of_angle(coordinates[index], SCREEN_DISTANCE_M)
            reduced = width * sine / WAVELENGTH_M
            if abs(reduced) > 4.0:
                continue
            worst = max(worst, abs(row[index] / row[0] - normalised_sinc(reduced) ** 2))
        deviations.append(worst)
    pairs = list(zip(deviations[:-1], deviations[1:], strict=True))
    orders = [math.log2(before / after) for before, after in pairs]
    assert all(after < before for before, after in pairs), (
        f"加密网格没让偏差单调下降：{deviations!r}"
    )
    assert 1.8 <= orders[-1] <= 2.2, f"最细两档的阶{orders[-1]!r}不在[1.8, 2.2]（实测2.026）"


# --- 抓到的真缺陷：孔径边界落在采样点上 -----------------------------------


def test_an_aperture_edge_that_lands_on_a_sample_fails_closed():
    """本轨道抓到的真缺陷：边界落在采样点上时"这个点归谁"由浮点最后一位决定。

    第一版写``half = (M/2 - 0.5) * dx``，在``dx = 10e-6``上给出8个采样、
    在``dx = 2.5e-6``上给出31个（期望32）。后果是采样缝宽整差一格、
    衍射零点挪位，而"收敛阶"实测从2.106变成**-4.222**。
    """

    with pytest.raises(OpticsError, match="落在第"):
        rectangular_aperture(
            row_count=8, column_count=64, pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
            half_width_x_m=4.0 * GRID_PITCH_M, half_width_y_m=1.0e9 * GRID_PITCH_M,
        )
    with pytest.raises(OpticsError, match="圆孔半径"):
        circular_aperture(
            row_count=64, column_count=64, pitch_x_m=GRID_PITCH_M,
            pitch_y_m=GRID_PITCH_M, radius_m=16.0 * GRID_PITCH_M,
        )
    #: 偏开四分之一格就没有歧义了，且采样数是确定的。
    mask = rectangular_aperture(
        row_count=8, column_count=64, pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        half_width_x_m=4.25 * GRID_PITCH_M, half_width_y_m=1.0e9 * GRID_PITCH_M,
    )
    assert _lit_samples(mask) == 9


def test_the_ambiguity_tolerance_is_a_band_not_an_equality():
    """判据是"落在容差带内"而不是"恰好相等"——浮点上"恰好"几乎从不成立。"""

    off_by_a_hair = (4.0 + 0.5 * APERTURE_EDGE_AMBIGUITY_TOLERANCE) * GRID_PITCH_M
    with pytest.raises(OpticsError, match="落在第"):
        rectangular_aperture(
            row_count=8, column_count=64, pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
            half_width_x_m=off_by_a_hair, half_width_y_m=1.0e9 * GRID_PITCH_M,
        )


# --- 判据二｜双缝 ---------------------------------------------------------


def _double_slit():
    offset = DOUBLE_SLIT_SEPARATION_SAMPLES / 2.0
    left, half = _slit(DOUBLE_SLIT_SAMPLES, centre_offset_samples=-offset)
    right, _ = _slit(DOUBLE_SLIT_SAMPLES, centre_offset_samples=offset)
    mask = _union(left, right)
    assert max(abs(value) for value in mask.values()) == 1.0, "两缝重叠了"
    assert _lit_samples(mask) == 2 * DOUBLE_SLIT_SAMPLES
    aperture_half = (
        DOUBLE_SLIT_SEPARATION_SAMPLES / 2.0 + DOUBLE_SLIT_SAMPLES / 2.0
    ) * GRID_PITCH_M
    return mask, aperture_half


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5, 6))
def test_the_double_slit_fringe_spacing_matches_the_closed_form(order):
    """条纹极大在``sin(theta) = m lambda / d``，`d`是**中心距**不是缝宽。"""

    mask, aperture_half = _double_slit()
    result = propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        aperture_half_width_m=aperture_half,
    )
    row = result.intensity_rows()[0]
    coordinates = result.coordinates_x_m()
    index = order * GRID_COLUMNS // DOUBLE_SLIT_SEPARATION_SAMPLES
    assert row[index] > row[index - 1] and row[index] > row[index + 1], (
        f"第{order}级条纹极大不在bin {index}上"
    )
    separation = DOUBLE_SLIT_SEPARATION_SAMPLES * GRID_PITCH_M
    sine = paraxial_sine_of_angle(coordinates[index], SCREEN_DISTANCE_M)
    closed_form = order * WAVELENGTH_M / separation
    assert abs(sine - closed_form) <= 4.0 * EPS * abs(closed_form)


def test_the_missing_order_is_really_missing():
    """`d/w = 8` → 第8级被单缝包络的第一个零点压掉。**实测恰为0**。

    这条最难蒙对：一个把中心距与缝宽搞反、或把包络漏掉的实现，
    在前六级条纹上全绿，只有这里露。
    """

    mask, aperture_half = _double_slit()
    result = propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M,
        aperture_half_width_m=aperture_half,
    )
    row = result.intensity_rows()[0]
    missing = DOUBLE_SLIT_SEPARATION_SAMPLES // DOUBLE_SLIT_SAMPLES
    index = missing * GRID_COLUMNS // DOUBLE_SLIT_SEPARATION_SAMPLES
    assert index == GRID_COLUMNS // DOUBLE_SLIT_SAMPLES, "缺级不在包络零点上，推导错了"
    assert row[index] / row[0] <= SLIT_ZERO_INTENSITY_FLOOR


# --- 判据三｜菲涅耳数极限退化到艾里斑 -------------------------------------


def _airy_deviation(result, radius, distance):
    intensity = result.intensity_rows()
    coordinates = result.coordinates_x_m()
    peak = intensity[0][0]
    worst = 0.0
    for index in range(1, len(coordinates) // 2):
        sine = paraxial_sine_of_angle(coordinates[index], distance)
        argument = airy_argument(
            half_angle_rad=math.asin(sine),
            aperture_radius_m=radius,
            wavelength_m=WAVELENGTH_M,
        )
        if argument > 12.0:
            continue
        worst = max(worst, abs(intensity[0][index] / peak - airy_intensity(argument)))
    return worst


def test_the_fresnel_pattern_falls_onto_the_airy_closed_form_as_the_fresnel_number_drops():
    """本条的金标是`optics/diffraction.py`**今天就有**的那条闭式。

    实测（256x256、半径322.5微米）：
    N_F=0.329→1.3354e-2、0.0822→1.3592e-3、0.0205→1.4627e-3、0.0051→1.4691e-3。
    最后三档挤在同一个数上，因为那已经不是菲涅耳项而是**圆孔阶梯边**的地板。
    """

    count, pitch = 256, GRID_PITCH_M
    radius = (32.0 + EDGE_OFFSET_IN_PITCHES) * pitch
    mask = circular_aperture(
        row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
        radius_m=radius,
    )
    near = propagate_fresnel(
        mask, wavelength_m=WAVELENGTH_M, distance_m=0.5,
        pitch_x_m=pitch, pitch_y_m=pitch,
    )
    far = propagate_fresnel(
        mask, wavelength_m=WAVELENGTH_M, distance_m=32.0,
        pitch_x_m=pitch, pitch_y_m=pitch,
    )
    near_deviation = _airy_deviation(near, radius, 0.5)
    far_deviation = _airy_deviation(far, radius, 32.0)
    assert far_deviation <= AIRY_STAIRCASE_DEVIATION, f"远场偏差{far_deviation!r}"
    assert near_deviation >= 5.0 * far_deviation, (
        f"菲涅耳数0.33处的偏差{near_deviation!r}没有明显大于远场的{far_deviation!r}"
        "——那说明这条门没在量菲涅耳项"
    )


def test_the_fraunhofer_deviation_does_not_depend_on_the_distance():
    """夫琅禾费丢掉了唯一含`z`的近似项，所以它对艾里的偏差**与z无关**。

    这条把"阶梯边的地板"与"菲涅耳项"彻底分开：地板是网格的，不是距离的。
    实测三个距离上都是1.4696e-3。
    """

    count, pitch = 256, GRID_PITCH_M
    radius = (32.0 + EDGE_OFFSET_IN_PITCHES) * pitch
    mask = circular_aperture(
        row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
        radius_m=radius,
    )
    deviations = []
    for distance in (10.0, 32.0):
        result = propagate_fraunhofer(
            mask, wavelength_m=WAVELENGTH_M, distance_m=distance,
            pitch_x_m=pitch, pitch_y_m=pitch, aperture_half_width_m=radius,
        )
        deviations.append(_airy_deviation(result, radius, distance))
    assert abs(deviations[0] - deviations[1]) <= 1.0e-12 * deviations[0], (
        f"两个距离上的偏差不同：{deviations!r}——夫琅禾费不该含z"
    )
    assert deviations[0] <= AIRY_STAIRCASE_DEVIATION


def test_refining_the_grid_lowers_the_staircase_floor_but_not_monotonically():
    """如实记：圆孔的阶梯边误差是**振荡**的，不是``O(h^2)``。

    固定物理孔径（半径320微米）加密网格，实测
    R=16→1.5101e-3、R=32→1.4691e-3、R=64→1.9796e-4，"阶"是0.040与2.892。
    原因是笛卡尔格上圆内格点数的涨落（Gauss圆问题那一类）。
    所以这条门只断"都在上界内"与"最细的一档真的更好"。
    """

    deviations = []
    for count, samples in ((128, 16), (256, 32), (512, 64)):
        pitch = 320.0e-6 / samples
        radius = (samples + EDGE_OFFSET_IN_PITCHES) * pitch
        mask = circular_aperture(
            row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
            radius_m=radius,
        )
        result = propagate_fraunhofer(
            mask, wavelength_m=WAVELENGTH_M, distance_m=32.0,
            pitch_x_m=pitch, pitch_y_m=pitch, aperture_half_width_m=radius,
        )
        deviations.append(_airy_deviation(result, radius, 32.0))
    assert max(deviations) <= AIRY_STAIRCASE_DEVIATION, f"{deviations!r}"
    assert deviations[-1] < deviations[0] / 4.0, (
        f"加密到R=64没有把地板压下去：{deviations!r}"
    )


# --- 夫琅禾费的适用域 -----------------------------------------------------


def test_the_fraunhofer_form_refuses_a_configuration_that_is_not_in_the_far_field():
    count, pitch = 256, GRID_PITCH_M
    radius = (32.0 + EDGE_OFFSET_IN_PITCHES) * pitch
    mask = circular_aperture(
        row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
        radius_m=radius,
    )
    #: 边缘相位恰好在门槛上 → 放行；两倍门槛 → 拒答。
    at_bound = radius * radius / (WAVELENGTH_M * fraunhofer_max_fresnel_number())
    propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=at_bound,
        pitch_x_m=pitch, pitch_y_m=pitch, aperture_half_width_m=radius,
    )
    with pytest.raises(OpticsError, match="不在远场"):
        propagate_fraunhofer(
            mask, wavelength_m=WAVELENGTH_M, distance_m=at_bound / 2.0,
            pitch_x_m=pitch, pitch_y_m=pitch, aperture_half_width_m=radius,
        )
    assert fraunhofer_max_fresnel_number() == FRAUNHOFER_MAX_EDGE_PHASE_RAD / math.pi


def test_the_declared_edge_phase_bound_is_measured_not_guessed():
    """门槛处**丢掉的那一项有多大**：实测归一化强度最大差1.2013e-4，
    且随边缘相位**平方**增长（0.02→4.80e-6、0.05→3.00e-5、0.10→1.20e-4）。
    """

    count, pitch = 256, GRID_PITCH_M
    radius = (32.0 + EDGE_OFFSET_IN_PITCHES) * pitch
    mask = circular_aperture(
        row_count=count, column_count=count, pitch_x_m=pitch, pitch_y_m=pitch,
        radius_m=radius,
    )
    #: 门槛距离一律从`at_bound`按倍数取，不各自反算——反算会在最后一位上
    #: 把边缘相位顶到`0.10000000000000002`，于是夫琅禾费当场拒答（实测踩过）。
    at_bound = radius * radius / (WAVELENGTH_M * fraunhofer_max_fresnel_number())
    measured = {}
    for phase, distance in ((0.02, 5.0 * at_bound), (0.1, at_bound)):
        fresnel = propagate_fresnel(
            mask, wavelength_m=WAVELENGTH_M, distance_m=distance,
            pitch_x_m=pitch, pitch_y_m=pitch,
        )
        fraunhofer = propagate_fraunhofer(
            mask, wavelength_m=WAVELENGTH_M, distance_m=distance,
            pitch_x_m=pitch, pitch_y_m=pitch, aperture_half_width_m=radius,
        )
        left = fresnel.intensity_rows()[0]
        right = fraunhofer.intensity_rows()[0]
        measured[phase] = max(
            abs(left[index] / left[0] - right[index] / right[0])
            for index in range(count // 2)
        )
    assert measured[0.1] <= 1.5 * FRAUNHOFER_EDGE_PHASE_MEASURED_DEVIATION
    assert measured[0.1] >= 0.5 * FRAUNHOFER_EDGE_PHASE_MEASURED_DEVIATION
    ratio = measured[0.1] / measured[0.02]
    assert 20.0 <= ratio <= 30.0, (
        f"相位放大5倍时误差放大了{ratio!r}倍，不是平方律的25倍——"
        "那说明门槛的推导（丢掉的是二次相位）不成立"
    )


def test_the_fresnel_number_helper_takes_a_half_width():
    """半宽写成全宽差4倍——名字里带`half_width`就是为了这个。"""

    number = fresnel_number(
        aperture_half_width_m=1.0e-3, wavelength_m=WAVELENGTH_M, distance_m=1.0
    )
    doubled = fresnel_number(
        aperture_half_width_m=2.0e-3, wavelength_m=WAVELENGTH_M, distance_m=1.0
    )
    assert abs(doubled / number - 4.0) <= 8.0 * EPS * 4.0


# --- 坐标出口与公开面 -----------------------------------------------------


def test_the_coordinate_exit_is_in_fft_order_not_centred_order():
    coordinates = spatial_coordinates_m(8, 2.0)
    assert coordinates == (0.0, 2.0, 4.0, 6.0, -8.0, -6.0, -4.0, -2.0)
    with pytest.raises(OpticsError, match="必须是2的幂"):
        spatial_coordinates_m(6, 1.0)


def test_the_propagated_field_carries_its_units():
    mask, half = _slit(SLIT_SAMPLES)
    result = propagate_fraunhofer(
        mask, wavelength_m=WAVELENGTH_M, distance_m=SCREEN_DISTANCE_M,
        pitch_x_m=GRID_PITCH_M, pitch_y_m=GRID_PITCH_M, aperture_half_width_m=half,
    )
    expected_pitch = (
        WAVELENGTH_M * SCREEN_DISTANCE_M / (GRID_COLUMNS * GRID_PITCH_M)
    )
    assert result.pitch_x_m == expected_pitch
    assert result.distance_m == SCREEN_DISTANCE_M
    assert result.wavelength_m == WAVELENGTH_M
    assert result.method == "fraunhofer_single_fft"
    assert len(result.coordinates_x_m()) == GRID_COLUMNS
    assert len(result.coordinates_y_m()) == GRID_ROWS


def test_the_new_names_stay_out_of_the_package_facade():
    from physics_engine.optics import propagation as module

    for name in module.__all__:
        assert hasattr(module, name), f"__all__里的{name!r}不存在"
    assert module.__all__ == sorted(module.__all__), "__all__未排序"
    leaked = sorted(set(module.__all__) & set(physics_engine.__all__))
    assert not leaked, f"传播的公开名漏进了包门面：{leaked}"
