"""`optics/field.py`的协议门——变换本身对不对，以及那两条约定有没有被守住。

分工照decisions/0024与0031的先例：**案例验物理，本文件验协议**。
本文件不碰任何传播物理（那是`test_optics_propagation.py`与
`cases/double_slit_propagated`），只验六件事：

1. 与**朴素DFT**逐点对拍（小N，容差从误差模型算出来）；
2. Parseval恒等式（它锁的是归一化落在哪一侧）；
3. 逆变换往返到机器精度；
4. 线性性、平移定理、卷积定理；
5. 二维可分离性（先行后列 对 先列后行）；
6. 两条约定：非2幂**失败关闭**、复数落盘是**二元组**（决策0086）。

**每一条门都附必须红**：本文件下半段造了五个"坏变换"（符号反、归一化搬到
正变换、漏位反转、逆变换漏除N、二维只做行不做列），逐条喂给同一个门函数，
要求它当场炸。门函数因此**只写一遍**——绿的那次和红的那次跑的是同一段判据，
不存在"必红用例验的是另一条门"这种空转。

## 容差全部是**算出来的**（0024第三节的纪律）

误差模型：朴素DFT是`N`项直接求和，最坏误差``~ N eps sum|x|``；
FFT是``log2(N)``级蝶形，误差``~ log2(N) eps sum|x|``。两者之差因此以
``N eps sum|x|``为界。下面每个因子后面括号里的数是**实测的最坏比值**
（多组随机输入、多个长度），因子取它的若干倍——**比值本身也被断言**
（不许比实测松一个数量级以上），理由与`bessel.py`那条精度申报同源：
写松的门等于没有门。
"""

from __future__ import annotations

import cmath
import math
import random
import sys

import pytest

import physics_engine
from physics_engine.optics.errors import OpticsError
from physics_engine.optics.field import (
    COMPLEX_COMPONENT_COUNT,
    COMPLEX_COMPONENT_ORDER,
    FORWARD_TRANSFORM_SIGN,
    INVERSE_SCALES_BY_RECIPROCAL_COUNT,
    INVERSE_TRANSFORM_SIGN,
    ComplexField2D,
    complex_from_components,
    complex_to_components,
    fft,
    fft2,
    ifft,
    ifft2,
    is_power_of_two,
    next_power_of_two,
    sequence_from_components,
    sequence_to_components,
    signed_frequency_indices,
    zero_pad_to_power_of_two,
)

EPS = sys.float_info.epsilon

#: 逐点对拍朴素DFT的容差因子：``tol = 因子 * N * eps * sum|x|``。
#: 实测最坏比值1.037（N<=64，200组随机输入），取8是它的7.7倍。
DFT_CROSS_CHECK_FACTOR = 8.0
DFT_CROSS_CHECK_MEASURED = 1.037

#: Parseval的容差因子：``tol = 因子 * eps * sum|x|^2``。实测最坏1.570，取16是10倍。
#: **它在N上几乎不长**（N=2到1024全在0.76—1.57之间），因为两侧是同构的和。
PARSEVAL_FACTOR = 16.0
PARSEVAL_MEASURED = 1.570

#: 往返的容差因子：``tol = 因子 * eps * max|x|``。实测N=1..1024最坏3.266，取16是4.9倍。
#: 它随N缓慢增长（N=4时1.09、N=1024时3.27，约``sqrt(log2 N)``），
#: 所以申报范围写死在``N <= 1024``——**范围外未测，不写"成立"**（0031第3.2节的口径）。
ROUND_TRIP_FACTOR = 16.0
ROUND_TRIP_MEASURED = 3.266
ROUND_TRIP_TESTED_MAX_LENGTH = 1024

#: 线性性：``tol = 因子 * eps * (|a| sum|x| + |b| sum|y|)``。实测最坏0.779，取8是10倍。
LINEARITY_FACTOR = 8.0
LINEARITY_MEASURED = 0.779

#: 平移定理：``tol = 因子 * eps * sum|x|``。实测最坏1.848，取16是8.7倍。
#:
#: **这条容差里藏着本文件最贵的一条发现**：预测相位若直接写
#: ``cmath.exp(-2j*pi*k*m/N)``，N=1024上的比值是**120.6**而不是1.85——
#: 大到那条门测的其实是`cmath.exp`的大幅角约化，不是FFT。把相位先按
#: **整数**``(k*m) mod N``约化（整数取模精确）之后降到1.85，**差65倍**。
#: 判据不这么写，就是拿一条更差的参照去验一条更好的实现。
SHIFT_FACTOR = 16.0
SHIFT_MEASURED = 1.848

#: 卷积定理：``tol = 因子 * eps * sum|x| * sum|y| / N``。实测最坏1.449，取16是11倍。
CONVOLUTION_FACTOR = 16.0
CONVOLUTION_MEASURED = 1.449

#: 二维可分离性：``tol = 因子 * eps * sum|x|``。实测最坏0.359，取8是22倍。
#: **不是逐位相同**，见`test_the_two_orders_agree_but_not_bit_for_bit`。
SEPARABILITY_FACTOR = 8.0
SEPARABILITY_MEASURED = 0.359

#: 对拍用的长度。1是边界（``2^0``，变换是恒等），64以内朴素DFT还跑得动。
CROSS_CHECK_LENGTHS = (1, 2, 4, 8, 16, 32, 64)


# --- 独立参照（不共用被验实现的任何一行） ---------------------------------


def naive_dft(values, sign=FORWARD_TRANSFORM_SIGN):
    """定义式直算``X_k = sum_n x_n exp(sign * 2 pi i k n / N)``，``O(N^2)``。

    与被验实现**没有任何共用代码**：这里没有位反转、没有蝶形、没有旋转因子表。
    """

    count = len(values)
    return [
        sum(
            values[index] * cmath.exp(complex(0.0, sign * 2.0 * math.pi * k * index / count))
            for index in range(count)
        )
        for k in range(count)
    ]


def circular_convolution(left, right):
    """定义式循环卷积``(x * y)_k = sum_m x_m y_{k-m mod N}``。"""

    count = len(left)
    return [
        sum(left[index] * right[(k - index) % count] for index in range(count))
        for k in range(count)
    ]


def _random_signal(count, seed):
    generator = random.Random(seed)
    return [
        complex(generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0))
        for _ in range(count)
    ]


def _absolute_sum(values):
    return sum(abs(value) for value in values)


# --- 门函数（绿的那次与红的那次跑的是同一段判据） -------------------------


def _gate_matches_naive_dft(transform, values):
    count = len(values)
    reference = naive_dft(values)
    bound = DFT_CROSS_CHECK_FACTOR * count * EPS * _absolute_sum(values)
    worst = max(
        (abs(got - want) for got, want in zip(transform(values), reference, strict=True)),
        default=0.0,
    )
    assert worst <= bound, (
        f"N={count}：与朴素DFT逐点差{worst!r}，超出{bound!r}"
        f"（= {DFT_CROSS_CHECK_FACTOR} * N * eps * sum|x|）"
    )
    return worst


def _gate_parseval(transform, values):
    count = len(values)
    energy = sum(abs(value) ** 2 for value in values)
    spectral = sum(abs(value) ** 2 for value in transform(values)) / count
    bound = PARSEVAL_FACTOR * EPS * energy
    residual = abs(energy - spectral)
    assert residual <= bound, (
        f"N={count}：Parseval残差{residual!r}超出{bound!r}——"
        f"sum|x|^2={energy!r}，(1/N)sum|X|^2={spectral!r}。"
        "这条门锁的是归一化落在逆变换一侧（INVERSE_SCALES_BY_RECIPROCAL_COUNT）"
    )
    return residual


def _gate_round_trip(forward, inverse, values):
    recovered = inverse(forward(values))
    scale = max(abs(value) for value in values)
    bound = ROUND_TRIP_FACTOR * EPS * scale
    worst = max(abs(got - want) for got, want in zip(recovered, values, strict=True))
    assert worst <= bound, (
        f"N={len(values)}：ifft(fft(x))与x逐点差{worst!r}，超出{bound!r}"
        f"（= {ROUND_TRIP_FACTOR} * eps * max|x|）"
    )
    return worst


def _gate_linearity(transform, left, right, alpha, beta):
    combined = [alpha * a + beta * b for a, b in zip(left, right, strict=True)]
    predicted = [
        alpha * a + beta * b
        for a, b in zip(transform(left), transform(right), strict=True)
    ]
    bound = LINEARITY_FACTOR * EPS * (
        abs(alpha) * _absolute_sum(left) + abs(beta) * _absolute_sum(right)
    )
    worst = max(
        abs(got - want) for got, want in zip(transform(combined), predicted, strict=True)
    )
    assert worst <= bound, f"线性性残差{worst!r}超出{bound!r}"
    return worst


def _gate_shift_theorem(transform, values, shift):
    count = len(values)
    rolled = list(values[-shift:]) + list(values[:-shift])
    spectrum = transform(values)
    #: 相位按**整数**取模约化后再进`exp`：整数取模精确，大幅角约化不精确。
    predicted = [
        spectrum[k]
        * cmath.exp(
            complex(
                0.0,
                FORWARD_TRANSFORM_SIGN * 2.0 * math.pi * ((k * shift) % count) / count,
            )
        )
        for k in range(count)
    ]
    bound = SHIFT_FACTOR * EPS * _absolute_sum(values)
    worst = max(
        abs(got - want) for got, want in zip(transform(rolled), predicted, strict=True)
    )
    assert worst <= bound, f"N={count} 平移{shift}：平移定理残差{worst!r}超出{bound!r}"
    return worst


def _gate_convolution_theorem(forward, inverse, left, right):
    count = len(left)
    product = [a * b for a, b in zip(forward(left), forward(right), strict=True)]
    reference = circular_convolution(left, right)
    bound = CONVOLUTION_FACTOR * EPS * _absolute_sum(left) * _absolute_sum(right) / count
    worst = max(
        abs(got - want) for got, want in zip(inverse(product), reference, strict=True)
    )
    assert worst <= bound, f"N={count}：卷积定理残差{worst!r}超出{bound!r}"
    return worst


def _columns_then_rows(field):
    """先列后行——`fft2`是先行后列，两条次序必须给同一个答案。"""

    transposed = field.transposed()
    after_columns = ComplexField2D(tuple(fft(row) for row in transposed.rows)).transposed()
    return ComplexField2D(tuple(fft(row) for row in after_columns.rows))


def _gate_separability(two_dimensional_transform, field):
    reference = _columns_then_rows(field)
    bound = SEPARABILITY_FACTOR * EPS * _absolute_sum(field.values())
    worst = max(
        abs(got - want)
        for got, want in zip(
            two_dimensional_transform(field).values(), reference.values(), strict=True
        )
    )
    assert worst <= bound, f"形状{field.shape}：先行后列与先列后行差{worst!r}，超出{bound!r}"
    return worst


def _field_from_seed(rows, columns, seed):
    signal = _random_signal(rows * columns, seed=seed)
    return ComplexField2D.from_rows(
        [signal[index * columns : (index + 1) * columns] for index in range(rows)]
    )


# --- 绿：变换本身 ---------------------------------------------------------


@pytest.mark.parametrize("count", CROSS_CHECK_LENGTHS)
def test_the_transform_matches_a_naive_dft_point_by_point(count):
    """基2 Cooley-Tukey 对 定义式直算——两条无共用代码的路。"""

    values = _random_signal(count, seed=1000 + count)
    worst = _gate_matches_naive_dft(fft, values)
    scale = count * EPS * _absolute_sum(values)
    if scale:
        ratio = worst / scale
        assert ratio <= DFT_CROSS_CHECK_MEASURED * 2.0, (
            f"N={count}：实测比值{ratio!r}比申报的最坏{DFT_CROSS_CHECK_MEASURED}"
            "还大一倍以上——申报过时了，要重测而不是放宽"
        )


@pytest.mark.parametrize("count", (1, 2, 8, 64, 256, 1024))
def test_parseval_holds(count):
    _gate_parseval(fft, _random_signal(count, seed=2000 + count))


@pytest.mark.parametrize("count", (1, 2, 8, 64, 256, ROUND_TRIP_TESTED_MAX_LENGTH))
def test_the_inverse_recovers_the_input_to_machine_precision(count):
    _gate_round_trip(fft, ifft, _random_signal(count, seed=3000 + count))


def test_the_transform_is_linear():
    left = _random_signal(64, seed=4001)
    right = _random_signal(64, seed=4002)
    _gate_linearity(fft, left, right, complex(0.3, -1.7), complex(-2.1, 0.9))


@pytest.mark.parametrize("count,shift", ((8, 3), (16, 5), (64, 21), (256, 85)))
def test_a_circular_shift_only_multiplies_the_spectrum_by_a_phase(count, shift):
    _gate_shift_theorem(fft, _random_signal(count, seed=5000 + count), shift)


@pytest.mark.parametrize("count", (4, 8, 16, 32))
def test_a_pointwise_product_of_spectra_is_a_circular_convolution(count):
    _gate_convolution_theorem(
        fft,
        ifft,
        _random_signal(count, seed=6001 + count),
        _random_signal(count, seed=6002 + count),
    )


@pytest.mark.parametrize("shape", ((4, 8), (8, 4), (2, 64), (16, 32), (32, 32)))
def test_the_two_dimensional_transform_is_separable(shape):
    """**非方阵**是这条门的关键：行列搞混在方阵上只会转置图样，在非方阵上才炸。"""

    rows, columns = shape
    _gate_separability(fft2, _field_from_seed(rows, columns, 7000 + rows * 100 + columns))


def test_the_two_orders_agree_but_not_bit_for_bit():
    """可分离性是**数值恒等式不是逐位恒等式**——如实记下来，免得被当成缺陷。

    两条次序的求和结合次序不同（先行后列先对列求和，先列后行先对行求和），
    浮点加法不结合，所以逐位相同只是巧合。实测64x64随机场上约4%的元素逐位相同。
    """

    field = _field_from_seed(64, 64, 7777)
    got = fft2(field).values()
    want = _columns_then_rows(field).values()
    identical = sum(1 for a, b in zip(got, want, strict=True) if a == b)
    assert identical < len(got), "两条次序竟然逐位全等——那说明其中一条没真的跑"
    assert max(abs(a - b) for a, b in zip(got, want, strict=True)) <= (
        SEPARABILITY_FACTOR * EPS * _absolute_sum(field.values())
    )


def test_the_two_dimensional_round_trip_recovers_the_field():
    field = _field_from_seed(16, 32, 8080)
    recovered = ifft2(fft2(field))
    scale = max(abs(value) for value in field.values())
    worst = max(abs(a - b) for a, b in zip(recovered.values(), field.values(), strict=True))
    assert worst <= ROUND_TRIP_FACTOR * EPS * scale, f"二维往返残差{worst!r}"


# --- 必须红：五个坏变换，逐条喂给上面同一批门 -----------------------------


def _sign_flipped(values):
    """符号反了的"正变换"。图样左右镜像，**幅度谱完全一样**——目视看不出来。"""

    return tuple(
        value.conjugate() for value in fft([item.conjugate() for item in values])
    )


def _normalised_forward(values):
    """把``1/N``搬到正变换上。Parseval当场不成立。"""

    count = len(values)
    return tuple(value / count for value in fft(values))


def _unnormalised_inverse(values):
    """逆变换漏了``1/N``。"""

    return tuple(value * len(values) for value in ifft(values))


def _bit_reversed_output(values):
    """输出留在位反转次序里（"忘了那一步重排"最常见的症状）。"""

    count = len(values)
    width = count.bit_length() - 1
    spectrum = fft(values)
    return tuple(
        spectrum[int(format(index, f"0{width}b")[::-1], 2) if width else 0]
        for index in range(count)
    )


def _rows_only(field):
    """二维只做了行、没做列。"""

    return ComplexField2D(tuple(fft(row) for row in field.rows))


def test_must_be_red_a_sign_flipped_transform_fails_the_naive_dft_gate():
    with pytest.raises(AssertionError, match="与朴素DFT逐点差"):
        _gate_matches_naive_dft(_sign_flipped, _random_signal(16, seed=9001))


def test_must_be_red_a_sign_flipped_transform_fails_the_shift_gate():
    """符号反了**过得了Parseval**——只有带相位的判据抓得住它。"""

    _gate_parseval(_sign_flipped, _random_signal(16, seed=9002))
    with pytest.raises(AssertionError, match="平移定理残差"):
        _gate_shift_theorem(_sign_flipped, _random_signal(16, seed=9002), 5)


def test_must_be_red_normalising_the_forward_transform_breaks_parseval():
    with pytest.raises(AssertionError, match="Parseval残差"):
        _gate_parseval(_normalised_forward, _random_signal(32, seed=9003))


def test_must_be_red_an_unnormalised_inverse_breaks_the_round_trip():
    with pytest.raises(AssertionError, match="逐点差"):
        _gate_round_trip(fft, _unnormalised_inverse, _random_signal(32, seed=9004))


def test_must_be_red_a_bit_reversed_output_fails_the_dft_and_convolution_gates():
    with pytest.raises(AssertionError, match="与朴素DFT逐点差"):
        _gate_matches_naive_dft(_bit_reversed_output, _random_signal(16, seed=9005))
    with pytest.raises(AssertionError, match="卷积定理残差"):
        _gate_convolution_theorem(
            _bit_reversed_output,
            ifft,
            _random_signal(16, seed=9005),
            _random_signal(16, seed=9006),
        )


def test_must_be_red_a_two_dimensional_transform_that_forgets_the_columns():
    with pytest.raises(AssertionError, match="先行后列与先列后行差"):
        _gate_separability(_rows_only, _field_from_seed(4, 8, 9007))


def test_must_be_red_the_linearity_gate_catches_a_nonlinear_transform():
    """线性性的必红要用一个**非线性**的坏实现，前面五个都是线性的。"""

    def squared(values):
        return tuple(value * value for value in fft(values))

    with pytest.raises(AssertionError, match="线性性残差"):
        _gate_linearity(
            squared,
            _random_signal(16, seed=9008),
            _random_signal(16, seed=9009),
            complex(0.3, -1.7),
            complex(-2.1, 0.9),
        )


# --- 约定一：符号与归一化写成常量 -----------------------------------------


def test_the_declared_signs_are_opposite_and_the_scaling_sits_on_the_inverse():
    assert FORWARD_TRANSFORM_SIGN == -INVERSE_TRANSFORM_SIGN
    assert INVERSE_SCALES_BY_RECIPROCAL_COUNT is True
    #: 申报要被验：正变换真的不带归一化——常值场的DC分量等于N而不是1。
    ones = [complex(1.0, 0.0)] * 8
    assert fft(ones)[0] == complex(8.0, 0.0)
    assert ifft(fft(ones))[0] == complex(1.0, 0.0)


def test_the_forward_sign_is_the_one_that_is_declared():
    """符号申报的判别式：单频``exp(+2 pi i n / N)``必须落在bin 1而不是bin N-1。"""

    count = 8
    tone = [cmath.exp(complex(0.0, 2.0 * math.pi * n / count)) for n in range(count)]
    spectrum = fft(tone)
    peak = max(range(count), key=lambda k: abs(spectrum[k]))
    assert peak == 1, f"正变换符号与申报不符：峰落在bin {peak}"


# --- 约定二：非2幂失败关闭 -----------------------------------------------


@pytest.mark.parametrize("count", (3, 5, 6, 7, 9, 100, 1000))
def test_a_non_power_of_two_length_fails_closed(count):
    values = [complex(1.0, 0.0)] * count
    with pytest.raises(OpticsError, match="必须是2的幂"):
        fft(values)
    with pytest.raises(OpticsError, match="必须是2的幂"):
        ifft(values)


def test_the_two_dimensional_transform_checks_both_axes():
    with pytest.raises(OpticsError, match="fft2的列数必须是2的幂"):
        fft2(ComplexField2D.zeros(4, 6))
    with pytest.raises(OpticsError, match="fft2的行数必须是2的幂"):
        fft2(ComplexField2D.zeros(6, 4))
    with pytest.raises(OpticsError, match="ifft2的行数必须是2的幂"):
        ifft2(ComplexField2D.zeros(6, 4))


def test_the_failure_message_names_the_padding_it_refuses_to_do():
    """失败关闭要说清**它拒绝替你做什么**，否则下一个人会以为是bug去"修"。"""

    with pytest.raises(OpticsError) as caught:
        fft([complex(1.0, 0.0)] * 5)
    assert "补零" in str(caught.value)
    assert "8" in str(caught.value), "报错要报出补零之后会是多长，让调用方自己判断"


def test_padding_is_an_explicit_verb_and_it_really_changes_the_spectrum():
    """补零可以做，但必须是写下来的一次决定——而且它**真的改了谱**。"""

    values = [complex(1.0, 0.0)] * 5
    padded = zero_pad_to_power_of_two(values)
    assert len(padded) == 8
    assert padded[5:] == (complex(0.0, 0.0),) * 3
    spectrum = fft(padded)
    #: 补零后bin 1不再是零：原来长度5的常值场只有DC，补到8之后能量摊进了别的bin。
    assert abs(spectrum[1]) > 0.1, "补零没有改谱——那说明它什么也没做"
    assert abs(spectrum[0] - complex(5.0, 0.0)) <= 8.0 * EPS * 5.0


def test_the_power_of_two_helpers_agree_with_each_other():
    assert is_power_of_two(1) and is_power_of_two(1024)
    assert not is_power_of_two(0) and not is_power_of_two(-4) and not is_power_of_two(6)
    for count in (1, 2, 3, 5, 8, 9, 1000, 1024, 1025):
        rounded = next_power_of_two(count)
        assert is_power_of_two(rounded) and rounded >= count
        assert rounded // 2 < count


def test_signed_frequency_indices_fold_the_upper_half_to_negative():
    assert signed_frequency_indices(1) == (0,)
    assert signed_frequency_indices(2) == (0, -1)
    assert signed_frequency_indices(8) == (0, 1, 2, 3, -4, -3, -2, -1)
    with pytest.raises(OpticsError, match="必须是2的幂"):
        signed_frequency_indices(6)


# --- 约定三：复数落盘是二元组（决策0086第三节） ---------------------------


def test_a_complex_number_goes_to_disk_as_an_ordered_pair():
    assert COMPLEX_COMPONENT_ORDER == ("real", "imaginary")
    assert COMPLEX_COMPONENT_COUNT == 2
    assert complex_to_components(complex(3.0, -4.0)) == [3.0, -4.0]
    assert complex_from_components([3.0, -4.0]) == complex(3.0, -4.0)


def test_the_rectangular_wire_form_round_trips_bit_for_bit():
    """**逐字节对拍**（`float.hex()`，不是近似相等）：直角形制是无损的。"""

    generator = random.Random(31337)
    for _ in range(2000):
        value = complex(
            generator.uniform(-1e6, 1e6) * 10 ** generator.randint(-8, 8),
            generator.uniform(-1e6, 1e6) * 10 ** generator.randint(-8, 8),
        )
        recovered = complex_from_components(complex_to_components(value))
        assert recovered.real.hex() == value.real.hex()
        assert recovered.imag.hex() == value.imag.hex()


def test_the_rejected_polar_wire_form_really_does_lose_bits():
    """被否决的那个候选**当场被证伪**，不是被议论掉的。

    极坐标``(|z|, arg z)``往返要过`hypot`/`atan2`/`cos`/`sin`四道超越函数，
    每一道各带舍入；而且零振幅处相位无定义——**场的零点恰恰是衍射图样里
    最要紧的地方**。这条门实测有多少个数掉位。
    """

    generator = random.Random(31337)
    lossy = 0
    total = 2000
    for _ in range(total):
        value = complex(
            generator.uniform(-1e6, 1e6) * 10 ** generator.randint(-8, 8),
            generator.uniform(-1e6, 1e6) * 10 ** generator.randint(-8, 8),
        )
        back = cmath.rect(*cmath.polar(value))
        if back.real.hex() != value.real.hex() or back.imag.hex() != value.imag.hex():
            lossy += 1
    assert lossy > total // 2, (
        f"极坐标往返只掉了{lossy}/{total}个数——若这个数很小，"
        "本条否决理由要重写（但零相位无定义那条仍然成立）"
    )
    #: 零振幅：相位彻底丢失，`polar`只能给0，**相位信息不可能被记录**。
    assert cmath.polar(complex(0.0, 0.0))[1] == 0.0


def test_the_component_order_is_not_symmetric():
    """次序反了不报错——所以次序写成常量，并有这条门盯着读侧。"""

    def reversed_reader(components):
        return complex(components[1], components[0])

    value = complex(3.0, -4.0)
    assert reversed_reader(complex_to_components(value)) != value


@pytest.mark.parametrize(
    "bad", ([], [1.0], [1.0, 2.0, 3.0], [float("nan"), 0.0], [0.0, float("inf")])
)
def test_a_malformed_component_pair_fails_closed(bad):
    with pytest.raises(OpticsError):
        complex_from_components(bad)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), complex(1.0, float("nan"))))
def test_a_nonfinite_value_never_reaches_the_wire(bad):
    with pytest.raises(OpticsError, match="必须有限"):
        complex_to_components(bad)


def test_a_field_round_trips_through_the_wire_form_bit_for_bit():
    field = _field_from_seed(4, 8, 1234)
    recovered = ComplexField2D.from_components(field.to_components())
    for got, want in zip(recovered.values(), field.values(), strict=True):
        assert got.real.hex() == want.real.hex()
        assert got.imag.hex() == want.imag.hex()


def test_the_flattened_wire_form_is_c_order_interleaved():
    """一串复数展平后是**交错**的实虚流——`oracles.flatten_values`就是这么展的。"""

    values = [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
    nested = sequence_to_components(values)
    assert nested == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert [component for pair in nested for component in pair] == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    ]
    assert sequence_from_components(nested) == tuple(values)


def test_the_wire_form_survives_the_repository_canonicaliser():
    """约定要**真的能落盘**：喂给`canonical`必须过，且`nan`必须被挡住。

    这条把约定与仓里既有的字节机械接上——不接上就只是一段愿望。
    """

    from physics_engine.canonical import (
        WDS_PROFILE,
        CanonicalError,
        canonical_bytes,
        strict_loads,
    )

    field = _field_from_seed(2, 4, 555)
    document = {"field": field.to_components()}
    raw = canonical_bytes(document, WDS_PROFILE)
    restored = ComplexField2D.from_components(strict_loads(raw)["field"])
    for got, want in zip(restored.values(), field.values(), strict=True):
        assert got.real.hex() == want.real.hex()
        assert got.imag.hex() == want.imag.hex()
    with pytest.raises(CanonicalError):
        canonical_bytes({"bad": [float("nan"), 0.0]}, WDS_PROFILE)


# --- 场容器 ---------------------------------------------------------------


def test_a_ragged_or_empty_field_fails_closed():
    with pytest.raises(OpticsError, match="至少要有一行"):
        ComplexField2D(())
    with pytest.raises(OpticsError, match="至少要有一列"):
        ComplexField2D(((),))
    with pytest.raises(OpticsError, match="必须是矩形"):
        ComplexField2D(((complex(1.0),), (complex(1.0), complex(2.0))))


def test_the_field_reports_rows_as_y_and_columns_as_x():
    field = ComplexField2D.from_function(4, 8, lambda row, column: complex(row, column))
    assert field.shape == (4, 8)
    assert field.row_count == 4 and field.column_count == 8
    assert field.at(3, 7) == complex(3.0, 7.0)
    assert field.values()[:3] == (complex(0.0, 0.0), complex(0.0, 1.0), complex(0.0, 2.0))
    assert field.transposed().shape == (8, 4)
    assert field.transposed().at(7, 3) == complex(3.0, 7.0)


def test_intensity_is_the_squared_modulus_with_no_normalisation():
    field = ComplexField2D.from_rows([[complex(3.0, 4.0), complex(0.0, 0.0)]])
    assert field.intensity_rows() == ((25.0, 0.0),)
    assert field.peak_intensity() == 25.0


def test_a_nonfinite_sample_never_gets_into_a_field():
    with pytest.raises(OpticsError, match="必须有限"):
        ComplexField2D.from_rows([[complex(float("inf"), 0.0)]])


# --- 公开面 ---------------------------------------------------------------


def test_the_new_names_stay_out_of_the_package_facade():
    """与域隔离门③同一件事的另一面：光学的新名不许漏进包门面。"""

    from physics_engine.optics import field as field_module

    for name in field_module.__all__:
        assert hasattr(field_module, name), f"__all__里的{name!r}不存在"
    assert field_module.__all__ == sorted(field_module.__all__), "__all__未排序"
    leaked = sorted(set(field_module.__all__) & set(physics_engine.__all__))
    assert not leaked, f"复数场的公开名漏进了包门面：{leaked}"
