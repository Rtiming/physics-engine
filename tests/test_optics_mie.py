"""`optics/mie.py`的物理门——均匀球的**严格级数解**（决策0091，能力位S4.7）。

四条**互相独立**的判据，外加每条的必红：

1. **幺正性／能量守恒**：无吸收介质上``Q_ext = Q_sca``，
   而且**逐阶**``|a_n|^2 = Re(a_n)``（系数落在幺正圆上）。
   ``Q_ext``与``Q_sca``是两条不同的式子（一条取实部、一条取模平方），
   所以这是一条真判据；``Q_abs``按定义是两者之差，
   **"消光=散射+吸收"本身不是判据**，本文件不假装它是；
2. **小球极限退化到瑞利**，分成两条**互不依赖**的门：
   * 只用**严格级数**验``C_sca ∝ 1/lambda^4``（不碰任何瑞利闭式）；
   * 严格级数 对 瑞利闭式，**并给收敛阶**（实测``O(x^2)``）；
3. **大球极限（Extinction Paradox）**：``Q_ext -> 2``。
   本文件不止断"接近2"，**还断它是按``x^(-2/3)``趋近的**——
   相邻档的比值必须落在``2^(-2/3) = 0.63``附近。
   "接近2"随便一个数值噪声都能满足，"按x^(-2/3)接近2"不能；
4. **截断判据显式声明并失败关闭**：末项占比超过申报即当场炸；
   且那条判据自己被验（末项占比要真的盖住``n_max``对``n_max+10``的差）。

外加本轨抓到的**一条真缺陷**（决策0091第五节）：Bohren & Huffman的
`BHMIE`把对数导数向下递推的起点取成``max(n_max, |mx|) + 15``，
在``x = 500``、``m = 1.33``（无吸收）上给出的``Q_ext``与收敛值
**差4.0e-4**，而它自称是严格级数解。本文件把那条起点规则构造出来当必红。
"""

from __future__ import annotations

import math
import sys

import pytest

from physics_engine.optics.errors import OpticsError
from physics_engine.optics.mie import (
    MIE_GEOMETRIC_OPTICS_EXTINCTION,
    MIE_SERIES_TAIL_TOLERANCE,
    MIE_TESTED_REFRACTIVE_INDEX_MODULUS_MAX,
    MIE_TESTED_SIZE_PARAMETER_MAX,
    MIE_TESTED_SIZE_PARAMETER_MIN,
    clausius_mossotti_factor,
    mie_coefficients,
    mie_efficiencies,
    mie_max_order,
    mie_unitarity_residual,
    rayleigh_absorption_efficiency,
    rayleigh_scattering_efficiency,
    size_parameter_from_radius,
)

EPS = sys.float_info.epsilon

#: 无吸收构型：``Q_ext``与``Q_sca``必须相等。实测最坏相对差**1.281e-16**
#: （六个构型，其中四个**恰为0**），取32 eps = 7.1e-15即余量55倍。
LOSSLESS_ENERGY_TOLERANCE = 32.0 * EPS
LOSSLESS_ENERGY_MEASURED = 1.281e-16

#: 逐阶幺正性``| |a_n|^2 - Re(a_n) |``的上界。实测最坏**2.220e-16**，取1e-14。
UNITARITY_TOLERANCE = 1.0e-14
UNITARITY_MEASURED = 2.220e-16

#: 小球``1/lambda^4``那条门：固定半径、五个波长上``C_sca lambda^4``的相对散布。
#: **它不是0，也不该是0**——严格级数比瑞利多出``O(x^2)``的修正。
#: 实测：半径20nm→4.8101e-3、10nm→1.4003e-3、5nm→3.6187e-4、2.5nm→9.1197e-5，
#: 阶**1.780 → 1.952 → 1.988**，干净地趋向``O(x^2)``。
#: 门取"半径10nm处散布 <= 2e-3"外加"半径减半散布至少降到0.35倍"。
INVERSE_FOURTH_POWER_SPREAD_AT_TEN_NANOMETRES = 2.0e-3
INVERSE_FOURTH_POWER_MIN_DECAY = 0.35

#: 严格级数对瑞利闭式的收敛阶下界。实测1.665 → 1.932 → 1.984 → 1.996。
#: **不写死为2**（`harmonic_oscillator`立的实践：收敛比落在区间里）。
RAYLEIGH_MIN_ORDER = 1.85

#: 大球极限：``|Q_ext - 2|``相邻档的比值。理论值``2^(-2/3) = 0.62996``
#: （边缘衍射修正按``x^(-2/3)``）。实测0.63443／0.63434／0.63353／0.63266，
#: **从上方单调趋近**。窗口取``[0.60, 0.66]``。
EXTINCTION_PARADOX_RATIO_LOW = 0.60
EXTINCTION_PARADOX_RATIO_HIGH = 0.66
EXTINCTION_PARADOX_THEORY_RATIO = 2.0 ** (-2.0 / 3.0)

#: 大球极限那条门用的尺度参数梯子与吸收折射率（吸收把ripple压掉，
#: 于是``Q_ext``真的单调趋近2；无吸收的球有很强的振荡，那条曲线
#: **不适合用来验"趋近"**——如实记，不挑软柿子也不假装无吸收也单调）。
EXTINCTION_LADDER: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0, 800.0)
ABSORBING_INDEX = complex(1.5, 0.1)

#: 末项占比 对 真正的截断误差（``n_max``对``n_max + 10``）。
#: 实测128个构型上末项占比最坏1.6963e-10、截断误差最坏1.2287e-10，
#: 且**逐个构型截断误差都不超过同一构型的末项占比**。
TRUNCATION_PROBE_EXTRA_ORDERS = 10


def _lossless_configurations():
    return (
        (0.01, complex(1.5, 0.0)),
        (0.5, complex(1.33, 0.0)),
        (1.0, complex(1.5, 0.0)),
        (5.213, complex(1.55, 0.0)),
        (20.0, complex(1.33, 0.0)),
        (100.0, complex(1.2, 0.0)),
    )


# --- 判据一｜幺正性与能量守恒 ---------------------------------------------


def test_a_lossless_sphere_has_extinction_equal_to_scattering():
    """无吸收 ⟹ ``Q_ext = Q_sca``。**两条不同的式子给同一个数**。

    ``Q_ext``取``Re(a_n + b_n)``、``Q_sca``取``|a_n|^2 + |b_n|^2``，
    它们只有在系数落在幺正圆上时才相等——所以这不是恒等变形。
    """

    worst = 0.0
    for size, index in _lossless_configurations():
        result = mie_efficiencies(size_parameter=size, refractive_index=index)
        worst = max(worst, abs(result.extinction - result.scattering) / result.extinction)
        assert result.absorption <= LOSSLESS_ENERGY_TOLERANCE * result.extinction
    assert worst <= LOSSLESS_ENERGY_TOLERANCE, f"无吸收球的消光与散射差了{worst!r}"


def test_every_coefficient_of_a_lossless_sphere_sits_on_the_unitarity_circle():
    """**逐阶**``|a_n|^2 = Re(a_n)``——比上一条更硬。

    上一条是两条求和相等，可以靠不同阶的误差互相抵消而蒙混过去；
    这一条逐阶看，抵消不掉。几何上它说的是无损球只能给散射波一个**相移**。
    """

    worst = 0.0
    for size, index in _lossless_configurations():
        coefficients = mie_coefficients(size_parameter=size, refractive_index=index)
        worst = max(worst, mie_unitarity_residual(coefficients))
    assert worst <= UNITARITY_TOLERANCE, f"逐阶幺正性残差{worst!r}"


def test_must_be_red_an_absorbing_sphere_breaks_the_unitarity_circle():
    """必须红：有吸收时``|a_n|^2 < Re(a_n)``，那个差**正是被吸收掉的部分**。

    所以`mie_unitarity_residual`对复折射率**拒答**而不是返回一个数——
    "这个构型下判据不成立"与"判据成立且等于某个数"是两件事。
    本条同时断两件：拒答真的发生了，且那个差确实是``O(0.1)``而不是噪声。
    """

    coefficients = mie_coefficients(size_parameter=5.0, refractive_index=ABSORBING_INDEX)
    with pytest.raises(OpticsError, match="幺正性判据只对无吸收介质成立"):
        mie_unitarity_residual(coefficients)
    gap = max(
        abs(value) ** 2 - value.real for value in coefficients.a + coefficients.b
    )
    assert min(
        abs(value) ** 2 - value.real for value in coefficients.a + coefficients.b
    ) < -1.0e-3, "有吸收球的系数居然还在幺正圆上，那这条判据分不开吸收与无吸收"
    assert gap <= UNITARITY_TOLERANCE, "居然有一阶跑到幺正圆外——那是增益不是吸收"


def test_the_absorbing_sphere_splits_the_extinction_into_two_positive_parts():
    """有吸收时``Q_ext``、``Q_sca``、``Q_abs``三者都为正且``Q_abs = Q_ext - Q_sca``。

    **这条不是能量守恒判据**（``Q_abs``按定义就是那个差），
    它守的是**符号**：时间约定翻了的话``Q_abs``会变成负的，
    而`mie.py`对那种构型失败关闭。本条正面确认没有翻。
    """

    result = mie_efficiencies(size_parameter=10.0, refractive_index=ABSORBING_INDEX)
    assert result.extinction > 0.0
    assert result.scattering > 0.0
    assert result.absorption > 0.0
    assert result.absorption == result.extinction - result.scattering


def test_a_negative_imaginary_index_fails_closed_and_says_which_time_convention():
    with pytest.raises(OpticsError, match="exp\\(-i\\*omega\\*t\\)"):
        mie_efficiencies(size_parameter=1.0, refractive_index=complex(1.5, -0.1))


# --- 判据二｜小球极限退化到瑞利 -------------------------------------------


def _invariant_spread(radius_m, wavelengths, index=complex(1.5, 0.0)):
    """``C_sca lambda^4``在若干波长上的相对散布。**只用严格级数**。"""

    values = []
    for wavelength in wavelengths:
        size = size_parameter_from_radius(radius_m=radius_m, wavelength_m=wavelength)
        result = mie_efficiencies(size_parameter=size, refractive_index=index)
        values.append(result.scattering * math.pi * radius_m**2 * wavelength**4)
    return (max(values) - min(values)) / min(values)


def test_the_small_sphere_scatters_as_one_over_the_fourth_power_of_the_wavelength():
    """``C_sca ∝ 1/lambda^4``——**这条门一个瑞利闭式都不碰**。

    固定半径与折射率，五个波长（400—1000纳米）上量``C_sca lambda^4``：
    它必须是常数。**残余散布不是误差而是严格级数比瑞利多出的``O(x^2)``修正**，
    所以本条还断"半径减半散布至少降到0.35倍"——即那个残余真的随``x^2``走，
    而不是一条恰好被容差盖住的常数偏差。
    """

    wavelengths = (400.0e-9, 500.0e-9, 632.8e-9, 800.0e-9, 1000.0e-9)
    spread_ten = _invariant_spread(10.0e-9, wavelengths)
    assert spread_ten <= INVERSE_FOURTH_POWER_SPREAD_AT_TEN_NANOMETRES, (
        f"C_sca·lambda^4在五个波长上的散布{spread_ten!r}"
    )
    previous = _invariant_spread(20.0e-9, wavelengths)
    for radius in (10.0e-9, 5.0e-9, 2.5e-9):
        current = _invariant_spread(radius, wavelengths)
        assert current <= INVERSE_FOURTH_POWER_MIN_DECAY * previous, (
            f"半径减到{radius!r}时散布只从{previous!r}降到{current!r}——"
            "那说明残余不是``O(x^2)``的修正，而是一条常数偏差"
        )
        previous = current


def test_the_exact_series_converges_to_the_rayleigh_closed_form_at_second_order():
    """严格级数 对 瑞利闭式，**并给收敛阶**。

    瑞利闭式来自一条完全不同的推导（球当成静电偶极子，
    极化率``4 pi a^3 (m^2-1)/(m^2+2)``），里面既没有球贝塞尔也没有递推。
    实测相对偏差8.4179e-3 → 2.6541e-3 → 6.9546e-4 → 1.7582e-4 → 4.4077e-5，
    阶**1.665 → 1.932 → 1.984 → 1.996**。
    """

    index = complex(1.5, 0.0)
    deviations = []
    for size in (0.4, 0.2, 0.1, 0.05, 0.025):
        exact = mie_efficiencies(size_parameter=size, refractive_index=index).scattering
        closed = rayleigh_scattering_efficiency(size_parameter=size, refractive_index=index)
        deviations.append(abs(exact / closed - 1.0))
    orders = [
        math.log(coarse / fine, 2.0)
        for coarse, fine in zip(deviations[1:-1], deviations[2:], strict=True)
    ]
    assert all(order >= RAYLEIGH_MIN_ORDER for order in orders), (
        f"退化到瑞利的收敛阶{orders!r}，实测偏差{deviations!r}"
    )
    assert deviations[-1] <= 1.0e-4


def test_the_absorption_also_converges_to_its_rayleigh_closed_form():
    """吸收的小球极限``Q_abs = 4 x Im[(m^2-1)/(m^2+2)]``——**一次方不是四次方**。

    散射按``x^4``、吸收按``x``，于是小球在吸收介质里吸收远大于散射。
    实测相对偏差8.3683e-2 → 2.1888e-2 → 5.5611e-3 → 1.3999e-3 → 3.5109e-4，
    阶1.935 → 1.977 → 1.990 → 1.995。
    """

    deviations = []
    for size in (0.4, 0.2, 0.1, 0.05, 0.025):
        exact = mie_efficiencies(
            size_parameter=size, refractive_index=ABSORBING_INDEX
        ).absorption
        closed = rayleigh_absorption_efficiency(
            size_parameter=size, refractive_index=ABSORBING_INDEX
        )
        deviations.append(abs(exact / closed - 1.0))
        #: 一次方对四次方：同一个`x`上吸收必须远大于散射。
        scattering = rayleigh_scattering_efficiency(
            size_parameter=size, refractive_index=ABSORBING_INDEX
        )
        assert closed > 10.0 * scattering
    orders = [
        math.log(coarse / fine, 2.0)
        for coarse, fine in zip(deviations[:-1], deviations[1:], strict=True)
    ]
    assert all(order >= RAYLEIGH_MIN_ORDER for order in orders), f"阶{orders!r}"


def test_a_real_refractive_index_absorbs_exactly_nothing_in_the_rayleigh_limit():
    """无吸收 ⟹ 瑞利吸收闭式**恰为0**（`(m^2-1)/(m^2+2)`是实数）。"""

    assert clausius_mossotti_factor(complex(1.5, 0.0)).imag == 0.0
    assert rayleigh_absorption_efficiency(
        size_parameter=0.1, refractive_index=complex(1.5, 0.0)
    ) == 0.0


# --- 判据三｜大球极限（Extinction Paradox） -------------------------------


def test_the_extinction_efficiency_approaches_two_as_the_sphere_grows():
    """``Q_ext -> 2``，而且**是按``x^(-2/3)``趋近的**。

    "趋近2"本身是一条软判据（任何缓慢下降的曲线都满足）。
    本条断的是**指数**：``|Q_ext - 2|``相邻档（`x`翻倍）的比值必须落在
    ``2^(-2/3) = 0.62996``附近的窗口里。实测0.63443／0.63434／0.63353／0.63266，
    **从上方单调趋近理论值**——那是边缘衍射修正的标度律。

    梯子用**吸收**球（``m = 1.5 + 0.1i``）：吸收把共振ripple压掉，
    无吸收球的``Q_ext``在大`x`上振荡得很厉害，**那条曲线不适合验"趋近"**。
    如实记在这里，不假装无吸收也单调。
    """

    gaps = []
    for size in EXTINCTION_LADDER:
        result = mie_efficiencies(size_parameter=size, refractive_index=ABSORBING_INDEX)
        gaps.append(abs(result.extinction - MIE_GEOMETRIC_OPTICS_EXTINCTION))
    assert all(later < earlier for earlier, later in zip(gaps[:-1], gaps[1:], strict=True))
    ratios = [later / earlier for earlier, later in zip(gaps[:-1], gaps[1:], strict=True)]
    for ratio in ratios:
        assert EXTINCTION_PARADOX_RATIO_LOW <= ratio <= EXTINCTION_PARADOX_RATIO_HIGH, (
            f"|Qext-2|的相邻比值{ratios!r}没有落在x^(-2/3)标度律"
            f"（{EXTINCTION_PARADOX_THEORY_RATIO!r}）的窗口里"
        )
    assert gaps[-1] <= 0.03


def test_must_be_red_a_geometric_cross_section_would_give_one_not_two():
    """必须红：把消光效率当成"挡住的几何截面"会给**1**不是2。

    这就是Extinction Paradox本身：球挡住的几何截面之外，
    还有等量的能量被衍射到前向小角内。本条断实测值离1远、离2近——
    一个把衍射那一半漏掉的实现正是给1。
    """

    result = mie_efficiencies(size_parameter=800.0, refractive_index=ABSORBING_INDEX)
    assert abs(result.extinction - 1.0) > 0.9
    assert abs(result.extinction - MIE_GEOMETRIC_OPTICS_EXTINCTION) < 0.05


# --- 判据四｜截断显式声明，不收敛失败关闭 ---------------------------------


def test_the_declared_truncation_order_is_the_wiscombe_criterion():
    for size in (1.0e-3, 0.1, 1.0, 10.0, 100.0, 1000.0):
        assert mie_max_order(size) == int(math.ceil(size + 4.0 * size ** (1 / 3) + 2.0))


def test_a_series_truncated_too_early_fails_closed():
    """截到第2阶的``x = 20``：末项还占大头 ⟹ **当场炸**。

    "这个构型我算不收敛"与"我给你一个截短了的答案"是两件事——
    与`propagation.py`拒绝在混叠区给数同一条纪律。
    """

    with pytest.raises(OpticsError, match="不收敛就失败关闭"):
        mie_efficiencies(size_parameter=20.0, refractive_index=complex(1.5, 0.0), order_count=2)


def test_the_tail_ratio_really_bounds_the_truncation_error():
    """**判据本身也要被验**：末项占比要真的盖住``n_max``对``n_max+10``的差。

    这条不是可有可无的：本轨的第一版判据只看末项，
    而末项**可以恰好落在一个近零点上**——那时它是0而后面的项不是。
    本条把两者一起量出来对比。实测128个构型上末项占比最坏1.6963e-10、
    截断误差最坏1.2287e-10。
    """

    for size, index in (
        (0.1, complex(1.5, 0.1)),
        (1.0, complex(1.5, 1.0)),
        (10.0, complex(3.0, 0.5)),
        (50.0, complex(1.5, 0.0)),
        (100.0, ABSORBING_INDEX),
    ):
        base = mie_efficiencies(size_parameter=size, refractive_index=index)
        extended = mie_efficiencies(
            size_parameter=size,
            refractive_index=index,
            order_count=base.order_count + TRUNCATION_PROBE_EXTRA_ORDERS,
        )
        truncation = max(
            abs(base.extinction - extended.extinction) / abs(extended.extinction),
            abs(base.scattering - extended.scattering) / abs(extended.scattering),
        )
        assert base.series_tail_ratio <= MIE_SERIES_TAIL_TOLERANCE
        assert truncation <= MIE_SERIES_TAIL_TOLERANCE, (
            f"x={size!r} m={index!r}：真正的截断误差{truncation!r}"
            f"超过了申报的收敛判据{MIE_SERIES_TAIL_TOLERANCE!r}"
        )


def test_the_logarithmic_derivative_start_is_high_enough():
    """把对数导数向下递推的起点抬高再算一遍，结果必须**逐位不变**。

    实测四个构型（含``x = 500``、``m = 1.33``）上抬高20/50/100/200/400阶
    **一位都不变**。
    """

    for size, index in (
        (5.213, complex(1.55, 0.0)),
        (50.0, complex(1.5, 0.0)),
        (100.0, ABSORBING_INDEX),
        (500.0, complex(1.33, 0.0)),
    ):
        base = mie_efficiencies(size_parameter=size, refractive_index=index)
        lifted = mie_efficiencies(
            size_parameter=size, refractive_index=index, logarithmic_derivative_lift=50
        )
        assert base.extinction.hex() == lifted.extinction.hex()
        assert base.scattering.hex() == lifted.scattering.hex()


def test_must_be_red_the_textbook_plus_fifteen_start_rule_is_not_enough():
    """必须红：**Bohren & Huffman的`BHMIE`那条起点规则**
    ``n_start = max(n_max, ceil|mx|) + 15``。

    这是本轨抓到的**真缺陷，而且它在教科书算法里**：
    ``x = 500``、``m = 1.33``（无吸收）上它给``Q_ext = 2.031189119014``，
    而收敛值是``2.030373894631``——**相对差4.0e-4**，
    对一个自称"严格级数解"的实现来说这不是舍入而是错。
    ``x = 50``、``m = 1.5``上同样的形态是6.7e-8。**一个字都不报。**

    本模块因此把对数导数的起点换成与`spherical_bessel_j_array`同一条规则
    （``max(阶, |z|) + max(25, sqrt(40 * 那个数))``）。
    """

    for size, index, minimum_damage in (
        (500.0, complex(1.33, 0.0), 1.0e-5),
        (50.0, complex(1.5, 0.0), 1.0e-9),
    ):
        top = mie_max_order(size)
        argument = index * size
        textbook_start = max(top, int(math.ceil(abs(argument)))) + 15
        derivative = [complex(0.0, 0.0)] * (textbook_start + 1)
        for order in range(textbook_start, 0, -1):
            ratio = order / argument
            derivative[order - 1] = ratio - 1.0 / (derivative[order] + ratio)
        extinction = _extinction_from_derivative(size, index, top, derivative)
        converged = mie_efficiencies(size_parameter=size, refractive_index=index).extinction
        damage = abs(extinction / converged - 1.0)
        assert damage > minimum_damage, (
            f"x={size!r} m={index!r}：教科书那条+15起点的相对误差只有{damage!r}——"
            "本条必红据以成立的那个事实没有出现，判据要重查"
        )


def _extinction_from_derivative(size, index, top, derivative):
    """用给定的对数导数数组算一遍``Q_ext``——只给上面那条必红用。

    **它复述的是被验实现的系数式子**，这是必红的代价：
    要证明"起点规则错了会怎样"，就得把除了起点之外的一切都保持一致。
    """

    from physics_engine.optics.spherical_bessel import riccati_bessel_xi_array

    xi = riccati_bessel_xi_array(top, size)
    psi = tuple(value.real for value in xi)
    total = 0.0
    for order in range(1, top + 1):
        ratio = order / size
        common_a = derivative[order] / index + ratio
        common_b = index * derivative[order] + ratio
        a_value = (common_a * psi[order] - psi[order - 1]) / (
            common_a * xi[order] - xi[order - 1]
        )
        b_value = (common_b * psi[order] - psi[order - 1]) / (
            common_b * xi[order] - xi[order - 1]
        )
        total += (2 * order + 1) * (a_value.real + b_value.real)
    return 2.0 / (size * size) * total


# --- 域外失败关闭与单位边界 -----------------------------------------------


@pytest.mark.parametrize(
    "bad", (0.0, -1.0, float("nan"), MIE_TESTED_SIZE_PARAMETER_MIN / 2.0,
            MIE_TESTED_SIZE_PARAMETER_MAX * 2.0)
)
def test_a_size_parameter_outside_the_tested_range_fails_closed(bad):
    with pytest.raises(OpticsError):
        mie_efficiencies(size_parameter=bad, refractive_index=complex(1.5, 0.0))


def test_a_refractive_index_outside_the_tested_range_fails_closed():
    with pytest.raises(OpticsError, match="域外是「未测」不是「成立」"):
        mie_efficiencies(
            size_parameter=1.0,
            refractive_index=complex(MIE_TESTED_REFRACTIVE_INDEX_MODULUS_MAX * 2.0, 0.0),
        )


def test_the_size_parameter_takes_a_radius_not_a_diameter():
    """``x = 2 pi a / lambda``吃**半径**。传直径进来所有效率都会错。

    与`diffraction.airy_argument`吃半径同一条实践：**签名替调用方记住**。
    """

    radius = 100.0e-9
    wavelength = 628.3185307179587e-9
    assert abs(size_parameter_from_radius(radius_m=radius, wavelength_m=wavelength) - 1.0) < 1e-9
    doubled = size_parameter_from_radius(radius_m=2.0 * radius, wavelength_m=wavelength)
    assert abs(doubled - 2.0) < 1e-9
    #: 半径与直径搞反 ⟹ 尺度参数差2倍 ⟹ 小球散射效率差``2^4 = 16``倍。
    index = complex(1.5, 0.0)
    small = rayleigh_scattering_efficiency(size_parameter=0.05, refractive_index=index)
    large = rayleigh_scattering_efficiency(size_parameter=0.1, refractive_index=index)
    assert abs(large / small - 16.0) < 1.0e-12


def test_the_new_names_stay_out_of_the_package_facade():
    """本块只从子模块直接import；`physics_engine.__all__`不该多出名字。

    与`optics/field.py`、`optics/propagation.py`同一条（0086第七节第2条）：
    `optics/__init__.py`是共享文件，本轨不碰，re-export由主代理裁。
    """

    import physics_engine

    for name in ("mie", "mie_efficiencies", "spherical_bessel", "MieEfficiencies"):
        assert name not in physics_engine.__all__
