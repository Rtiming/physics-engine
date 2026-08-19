#!/usr/bin/env python3
"""角谱传播的金标生成器——**独立算法，不调被验内核**。

被验内核（`optics/propagation.propagate_angular_spectrum`）走的是
基2 Cooley-Tukey（`optics/field.fft2`：位反转重排、``log2(N)``级蝶形、
缓存的旋转因子表）正变换 → 乘传递函数 → 同一条路逆变换。

本生成器走**二维直接求和**：

    S(v,u) = sum_{(r,c) in 孔内} exp(-2 pi i (r v + c u)/N)
    U(r,c) = (1/N^2) sum_v sum_u S(v,u) H(u,v) exp(+2 pi i (r v + c u)/N)

**没有行列分解、没有位反转、没有旋转因子表**——两条路唯一共用的是
"二维DFT是什么"这条定义本身。相位先按**整数**``(r v + c u) mod N``约化再进`exp`：
整数取模精确，而`exp`的大幅角约化不精确（`tests/test_optics_field.py`实测过，
不这么写参照自己的误差会盖过被验对象，差65倍）。

生成器**不import `physics_engine.optics`**：孔径的采样位置、传递函数、
频率序号全部自己按几何与定义算。

## 四条oracle，以及为什么其中两条**傍轴给不出**

`cases/double_slit_propagated`冻的是**夫琅禾费**那一支（单次FFT、傍轴）。
角谱那一侧到今天为止**没有任何案例冻结金标**（0086第八节自己登记的GAP，
capability_ledger的S4.5因此是`partial`而不是`done`：
本仓的分档是``tests/``守实现、**案例冻判据**）。本页把它冻下来。

1. `plane_wave_eigenvalues`：平面波是角谱传播的**本征函数**，本征值是
   ``exp(i k z sqrt(1 - (lambda f)^2))``。**根号是精确的不是傍轴的**——
   傍轴把它展成``1 - (lambda f)^2 / 2``，两者在``lambda f = 0.9888``处
   实测差**7.10e-1**（模长都是1，差的全在相位里）。
   半群、能量、``z->0``退化三条门对精确与傍轴**一视同仁**，只有这一条分得开；
2. `evanescent_power_loss`：**这一条是角谱独有里最硬的一条，而且它只是一个数**。
   ``(lambda f)^2 > 1``的分量是倏逝波，精确传递函数给``exp(-k z sqrt(...)) < 1``，
   于是一个含倏逝内容的场传过去**会掉能量**。而傍轴传递函数的模**恒为1**
   （指数是纯虚的），它把倏逝波当成传播波，能量比**恰好是1.0**。
   亚波长网格（间距100纳米、波长632.8纳米，32格里只有11格是传播的）上
   4采样宽的缝实测：精确**0.895526867531937**、傍轴**恰好1.0**。
   验这个数不需要复述任何公式，也没有任何傍轴形制能给出它；
3. `rectangular_hole_field`：**两个方向都有界的矩孔**（不是仓里那个
   ``half_width_y_m = 1e9 * pitch``的"缝"）近场复振幅，二维直接求和 对 蝶形。
   顺带把矩孔的三个整数（亮采样32、亮行4、亮列8）零容差冻下来——
   能力位S4.6的label里"矩孔"那一格此前只有构件没有判据；
4. `self_consistency`：``z -> 0``退化（与``ifft2(fft2(U0))``**逐位相同**）、
   半群（传两次``z/2``等于一次``z``）、能量守恒（传播分量全在光锥内时``|H| = 1``）。

参考解出处：Goodman《Introduction to Fourier Optics》第3.10节
（角谱与传递函数``H = exp(i k z sqrt(1 - (lambda fx)^2 - (lambda fy)^2))``、
倏逝分量与它的指数衰减）；第4.2节（菲涅耳傍轴近似正是把该根号展到二阶，
即本页第1、2条oracle分开的那两件事）。
"""

from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/angular_spectrum_propagation"
ALGORITHM_VERSION = "1.0.0"

#: HeNe，与`scalar_diffraction_airy`、`double_slit_propagated`同一个波长。
WAVELENGTH_M = 632.8e-9

# --- 网格甲：宏观网格，传播分量全在光锥内（间距 >> 波长） -----------------

#: 方阵。角谱的输出网格与入射面**同间距**，所以两个方向必须都真的有频率内容；
#: 取方阵还让两个轴的``z_c``相等，适用域上界没有歧义。
GRID_A_COUNT = 32
GRID_A_PITCH_M = 5.0e-6

#: 矩孔：x方向8个采样、y方向4个采样。**两个方向取不同的数**——
#: 取成一样的话，行列搞混的实现在方形孔上完全看不出来。
APERTURE_SAMPLES_X = 8
APERTURE_SAMPLES_Y = 4

#: 半宽偏四分之一格：边界不许落在采样点上（0086第5.1节抓到的真缺陷，
#: 引擎侧对那种规格**失败关闭**）。中心偏半格：以采样点为中心的对称孔
#: 覆盖的采样数必然是**奇数**，而我们要2的幂。
EDGE_OFFSET_IN_PITCHES = 0.25

#: 冻结复振幅的采样点``(行, 列)``。**故意跨四个象限**：
#: 角谱的坐标是FFT次序（``0, d, ..., -(N/2)d, ..., -d``），
#: 只冻左上角那一块的话，把负频率折回正侧的实现照样全绿。
FIELD_SAMPLES: tuple[tuple[int, int], ...] = (
    (0, 0), (0, 1), (0, 4), (0, 8),
    (1, 0), (2, 3), (4, 4), (8, 8),
    (16, 16), (31, 31), (30, 5), (5, 30),
)

# --- 网格乙：亚波长网格，倏逝波真的出现 -----------------------------------

#: 间距100纳米、波长632.8纳米：``lambda / (2 dx) = 3.164``，
#: 32格里只有**11格**的``|lambda f| <= 1``，其余21格是倏逝的。
GRID_B_COUNT = 32
GRID_B_PITCH_M = 1.0e-7

#: 传播分量（``|lambda f| <= 1``）与倏逝分量各取四个bin。
PROPAGATING_BINS: tuple[int, ...] = (0, 1, 3, 5)
EVANESCENT_BINS: tuple[int, ...] = (6, 8, 12, 16)

#: 倏逝功率损失那条用的缝宽（采样数）。取4：窄到谱铺满整个带宽
#: （于是倏逝分量真的被激发），又宽到主瓣不止一个采样。
EVANESCENT_SLIT_SAMPLES = 4

#: 传播距离取``z_c``的多少倍。角谱的适用域是``0 <= z <= z_c``；
#: 取0.5与0.4是为了离两端都远，**不是**踩在边界上——边界那一格
#: 由`tests/test_optics_propagation.py`的适用域门守着，不是本案例的题。
GRID_A_DISTANCE_FRACTION = 0.5
GRID_B_DISTANCE_FRACTION = 0.4

#: 被验实现与本生成器之间的**实测**最坏偏差（本机，见案例页第三节）。
#: 容差由它们乘余量得到——**容差是算出来的不是拍的**（0024第三节）。
MEASURED_EIGENVALUE_DEVIATION = 3.5253e-15
MEASURED_FIELD_DEVIATION = 9.9920e-16
MEASURED_EVANESCENT_POWER_DEVIATION = 2.2204e-16
MEASURED_ZERO_DISTANCE_DEVIATION = 2.0024e-16
MEASURED_SEMIGROUP_DEVIATION = 3.1402e-16
MEASURED_POWER_RATIO_DEVIATION = 1.1102e-16


def signed_index(index: int, count: int) -> int:
    """FFT次序的第`index`格对应的有符号序号。生成器自己算，不调引擎的helper。"""

    half = count // 2
    return index if index < half else index - count


def transfer_function_max_distance_m(count: int, pitch_m: float) -> float:
    """``z_c = N dx^2 / lambda``——角谱的适用域上界。生成器自己算。"""

    return count * pitch_m * pitch_m / WAVELENGTH_M


def angular_wavenumber() -> float:
    """``k = 2 pi / lambda``（弧度/米）。**不是**谱学波数``1/lambda``。"""

    return 2.0 * math.pi / WAVELENGTH_M


def exact_transfer(reduced_square: float, distance_m: float) -> complex:
    """精确（标量）传递函数``exp(i k z sqrt(1 - q))``；``q > 1``时是倏逝衰减。"""

    wavenumber = angular_wavenumber()
    if reduced_square <= 1.0:
        return cmath.exp(
            complex(0.0, wavenumber * distance_m * math.sqrt(1.0 - reduced_square))
        )
    return complex(math.exp(-wavenumber * distance_m * math.sqrt(reduced_square - 1.0)), 0.0)


def paraxial_transfer(reduced_square: float, distance_m: float) -> complex:
    """傍轴传递函数``exp(i k z (1 - q/2))``——把根号展到二阶。

    **它的模恒为1**（指数是纯虚的），对`q > 1`也一样。
    第2条oracle分开的正是这件事：它把倏逝波当成传播波。
    """

    wavenumber = angular_wavenumber()
    return cmath.exp(complex(0.0, wavenumber * distance_m * (1.0 - 0.5 * reduced_square)))


def reduced_square_1d(bin_index: int, count: int, pitch_m: float) -> float:
    """``(lambda f)^2``，`f`是该bin的空间频率``k /(N dx)``（有符号）。"""

    frequency = signed_index(bin_index, count) / (count * pitch_m)
    return (WAVELENGTH_M * frequency) ** 2


def rectangular_hole_positions() -> tuple[tuple[int, int], ...]:
    """矩孔覆盖的采样点``(行序号, 列序号)``，**按几何算出来**。

    两个方向都有界——这正是它与仓里那个``half_width_y_m = 1e9 * pitch``
    的写法的分别：那一个在y方向占满窗口，**是缝不是孔**。
    """

    half_x = (APERTURE_SAMPLES_X - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES
    half_y = (APERTURE_SAMPLES_Y - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES
    selected: list[tuple[int, int]] = []
    for row in range(GRID_A_COUNT):
        position_y = signed_index(row, GRID_A_COUNT)
        if abs(position_y - 0.5) > half_y:
            continue
        for column in range(GRID_A_COUNT):
            position_x = signed_index(column, GRID_A_COUNT)
            if abs(position_x - 0.5) <= half_x:
                selected.append((row, column))
    return tuple(selected)


def slit_positions() -> tuple[int, ...]:
    """网格乙上那条缝覆盖的列序号（单行网格，y方向只有一个采样）。"""

    half = (EVANESCENT_SLIT_SAMPLES - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES
    return tuple(
        column
        for column in range(GRID_B_COUNT)
        if abs(signed_index(column, GRID_B_COUNT) - 0.5) <= half
    )


def direct_spectrum_2d(
    positions: tuple[tuple[int, int], ...], count: int
) -> dict[tuple[int, int], complex]:
    """``S(v,u) = sum exp(-2 pi i (r v + c u)/N)``，**二维直接求和**。

    没有行列分解、没有位反转、没有旋转因子表。相位按整数取模约化。
    """

    spectrum: dict[tuple[int, int], complex] = {}
    for v in range(count):
        for u in range(count):
            total = complex(0.0, 0.0)
            for row, column in positions:
                phase = -2.0 * math.pi * (((row * v + column * u) % count) / count)
                total += cmath.exp(complex(0.0, phase))
            spectrum[(v, u)] = total
    return spectrum


def direct_inverse_2d(
    spectrum: dict[tuple[int, int], complex], sample: tuple[int, int], count: int
) -> complex:
    """``U(r,c) = (1/N^2) sum_v sum_u S(v,u) exp(+2 pi i (r v + c u)/N)``。"""

    row, column = sample
    total = complex(0.0, 0.0)
    for v in range(count):
        for u in range(count):
            phase = 2.0 * math.pi * (((row * v + column * u) % count) / count)
            total += spectrum[(v, u)] * cmath.exp(complex(0.0, phase))
    return total / (count * count)


def main() -> int:
    distance_a = GRID_A_DISTANCE_FRACTION * transfer_function_max_distance_m(
        GRID_A_COUNT, GRID_A_PITCH_M
    )
    distance_b = GRID_B_DISTANCE_FRACTION * transfer_function_max_distance_m(
        GRID_B_COUNT, GRID_B_PITCH_M
    )

    # --- oracle一：平面波本征值 ------------------------------------------
    def eigenvalue(bin_index: int) -> complex:
        return exact_transfer(
            reduced_square_1d(bin_index, GRID_B_COUNT, GRID_B_PITCH_M), distance_b
        )

    propagating = [eigenvalue(index) for index in PROPAGATING_BINS]
    evanescent = [eigenvalue(index) for index in EVANESCENT_BINS]
    paraxial_gaps = [
        abs(
            eigenvalue(index)
            - paraxial_transfer(
                reduced_square_1d(index, GRID_B_COUNT, GRID_B_PITCH_M), distance_b
            )
        )
        for index in PROPAGATING_BINS + EVANESCENT_BINS
    ]

    # --- oracle二：倏逝功率损失 ------------------------------------------
    #: 用Parseval在**频域**算：``ratio = sum |H|^2 |S|^2 / sum |S|^2``。
    #: 被验实现是在**实空间**逐点求``|U|^2``再乘``dx dy``——两条路不同。
    columns = slit_positions()
    numerator = 0.0
    denominator = 0.0
    propagating_bin_count = 0
    for u in range(GRID_B_COUNT):
        amplitude = complex(0.0, 0.0)
        for column in columns:
            phase = -2.0 * math.pi * (((column * u) % GRID_B_COUNT) / GRID_B_COUNT)
            amplitude += cmath.exp(complex(0.0, phase))
        reduced = reduced_square_1d(u, GRID_B_COUNT, GRID_B_PITCH_M)
        if reduced <= 1.0:
            propagating_bin_count += 1
        gain = abs(exact_transfer(reduced, distance_b)) ** 2
        weight = abs(amplitude) ** 2
        numerator += gain * weight
        denominator += weight
    evanescent_power_ratio = numerator / denominator

    #: 傍轴那一侧：``|H| = 1``逐bin，于是比值**恒等于1**。
    #: 它不是"约等于1"——是同一条求和的分子分母逐项相同。
    paraxial_power_ratio = 1.0

    # --- oracle三：矩孔近场复振幅 ----------------------------------------
    positions = rectangular_hole_positions()
    if len(positions) != APERTURE_SAMPLES_X * APERTURE_SAMPLES_Y:
        raise SystemExit(
            f"矩孔的采样数{len(positions)}不是期望的"
            f"{APERTURE_SAMPLES_X * APERTURE_SAMPLES_Y}，不落盘"
        )
    spectrum = direct_spectrum_2d(positions, GRID_A_COUNT)
    filtered = {}
    for (v, u), value in spectrum.items():
        reduced = reduced_square_1d(u, GRID_A_COUNT, GRID_A_PITCH_M) + reduced_square_1d(
            v, GRID_A_COUNT, GRID_A_PITCH_M
        )
        filtered[(v, u)] = value * exact_transfer(reduced, distance_a)
    field = [direct_inverse_2d(filtered, sample, GRID_A_COUNT) for sample in FIELD_SAMPLES]

    oracles = [
        {
            "id": "oracle:angular_spectrum/plane_wave_eigenvalues",
            "inputs": {
                "kind": "closed_form_exact_transfer_function",
                "count": GRID_B_COUNT,
                "pitch_m": GRID_B_PITCH_M,
                "wavelength_m": WAVELENGTH_M,
                "distance_m": distance_b,
                "propagating_bins": list(PROPAGATING_BINS),
                "evanescent_bins": list(EVANESCENT_BINS),
                "complex_component_order": ["real", "imaginary"],
            },
            "expected": {
                "propagating_eigenvalue_components": [
                    [value.real, value.imag] for value in propagating
                ],
                "evanescent_eigenvalue_components": [
                    [value.real, value.imag] for value in evanescent
                ],
                "paraxial_eigenvalue_gap": paraxial_gaps,
            },
            "tolerances": {
                "propagating_eigenvalue_components": {
                    "abs": 1.6e-14,
                    "rel": 0.0,
                    "reason": "平面波是角谱传播的**本征函数**，本征值是闭式"
                              "``exp(i k z sqrt(1 - (lambda f)^2))``。被验路径是"
                              "长度32的正逆变换各一次（各约``log2(32)=5``级蝶形）"
                              "再逐点相除，误差模型``2 * 16 * 32 * eps = 2.3e-13``；"
                              f"实测最坏{MEASURED_EIGENVALUE_DEVIATION:.4e}"
                              f"（余量{1.6e-14 / MEASURED_EIGENVALUE_DEVIATION:.1f}倍）。"
                              "**判绝对不判相对**：本征值的实部虚部各自会过零，"
                              "过零点上相对误差没有意义",
                },
                "evanescent_eigenvalue_components": {
                    "abs": 1.6e-14,
                    "rel": 0.0,
                    "reason": "倏逝分量按``exp(-k z sqrt(q-1))``衰减——**不丢也不涨**。"
                              "虚部恒为0（衰减因子是实的），实部从0.2773降到0.0024。"
                              "容差与上一行同源同值：走的是同一条变换路径。"
                              "**傍轴传递函数在这四个bin上的模恒为1**，"
                              "即它给不出这一行的任何一个数",
                },
                "paraxial_eigenvalue_gap": {
                    "abs": 1.0e-14,
                    "rel": 0.0,
                    "reason": "精确本征值与**傍轴**本征值之差的模，逐bin。"
                              "它冻的是「这条判据分得开精确与傍轴」这件事本身："
                              "``lambda f = 0.1978``处差3.9161e-4、``0.9888``处差7.1041e-1、"
                              "倏逝的四个bin上差7.8580e-1到1.0116。"
                              "**半群、能量守恒、``z->0``退化三条门对精确与傍轴一视同仁**，"
                              "只有本条oracle所依赖的那条判据分得开。"
                              "容差按两个模长相减的浮点误差取（各约几个eps）",
                },
            },
        },
        {
            "id": "oracle:angular_spectrum/evanescent_power_loss",
            "inputs": {
                "kind": "parseval_in_the_frequency_domain",
                "count": GRID_B_COUNT,
                "rows": 1,
                "pitch_m": GRID_B_PITCH_M,
                "wavelength_m": WAVELENGTH_M,
                "distance_m": distance_b,
                "slit_samples": EVANESCENT_SLIT_SAMPLES,
                "slit_columns": list(columns),
                "edge_offset_in_pitches": EDGE_OFFSET_IN_PITCHES,
                "propagating_bin_count": propagating_bin_count,
                "note": "间距100纳米、波长632.8纳米，32格里只有"
                        f"{propagating_bin_count}格是传播的，其余是倏逝的",
            },
            "expected": {
                "power_ratio_after_propagation": evanescent_power_ratio,
                "paraxial_power_ratio_would_be": paraxial_power_ratio,
            },
            "tolerances": {
                "power_ratio_after_propagation": {
                    "abs": 1.0e-14,
                    "rel": 0.0,
                    "reason": "**本案例最硬的一条角谱独有判据，而且它只是一个数。**"
                              "含倏逝内容的场传过去会掉能量：精确传递函数在倏逝bin上"
                              "``|H| = exp(-k z sqrt(q-1)) < 1``。金标由**频域Parseval**"
                              "算（``sum |H|^2 |S|^2 / sum |S|^2``，`S`是定义式直接求和），"
                              "被验实现是在**实空间**逐点求``|U|^2``再乘``dx dy``——"
                              "两条路只共用Parseval恒等式本身。"
                              f"实测偏差{MEASURED_EVANESCENT_POWER_DEVIATION:.4e}"
                              f"（余量{1.0e-14 / MEASURED_EVANESCENT_POWER_DEVIATION:.0f}倍）",
                },
                "paraxial_power_ratio_would_be": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差，而且是结构性的**：傍轴传递函数的指数是纯虚的，"
                              "``|H| = 1``逐bin成立，于是同一条Parseval求和的分子分母"
                              "**逐项相同**，比值恒等于1.0。它与上一行的差"
                              "（1.0 - 0.8955 = 0.1045）就是「傍轴给不出角谱」这句话的量。"
                              "本行**不是**被验实现的输出，是被验实现必须与之不同的那个数——"
                              "conformance那边用它构造必红",
                },
            },
        },
        {
            "id": "oracle:angular_spectrum/rectangular_hole_field",
            "inputs": {
                "kind": "direct_two_dimensional_summation",
                "count": GRID_A_COUNT,
                "pitch_m": GRID_A_PITCH_M,
                "wavelength_m": WAVELENGTH_M,
                "distance_m": distance_a,
                "aperture_samples_x": APERTURE_SAMPLES_X,
                "aperture_samples_y": APERTURE_SAMPLES_Y,
                "edge_offset_in_pitches": EDGE_OFFSET_IN_PITCHES,
                "samples": [list(sample) for sample in FIELD_SAMPLES],
                "complex_component_order": ["real", "imaginary"],
            },
            "expected": {
                "field_components": [[value.real, value.imag] for value in field],
                "aperture_sample_count": float(len(positions)),
                "aperture_lit_row_count": float(len({row for row, _ in positions})),
                "aperture_lit_column_count": float(len({column for _, column in positions})),
            },
            "tolerances": {
                "field_components": {
                    "abs": 8.0e-15,
                    "rel": 0.0,
                    "reason": "二维直接求和 对 基2蝶形（**MFT-FFT等价**在角谱那一支上的"
                              "兑现；`double_slit_propagated`冻的是夫琅禾费那一支）。"
                              "两侧的浮点误差：正变换32项累加、乘一次传递函数、"
                              "逆变换1024项累加再除``N^2``，合起来约``30 eps = 6.7e-15``；"
                              f"实测最坏{MEASURED_FIELD_DEVIATION:.4e}"
                              f"（余量{8.0e-15 / MEASURED_FIELD_DEVIATION:.1f}倍）。"
                              "冻的采样点**跨四个象限**——角谱的坐标是FFT次序，"
                              "只冻左上角那一块的话把负频率折回正侧的实现照样全绿",
                },
                "aperture_sample_count": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "孔内采样数是整数，零容差。它锁的是**采样孔宽**——"
                              "边界落在采样点上时这个数会整差一格（0086第5.1节实测踩过："
                              "期望32个采样、实得31个），而图样照样漂亮",
                },
                "aperture_lit_row_count": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**这一行是「孔」与「缝」的分别**：亮行恰好4（不是全部32）。"
                              "仓里此前每一处`rectangular_aperture`调用的"
                              "``half_width_y_m``都取``1e9 * pitch``，亮满所有行——"
                              "那是缝，能力位S4.6的label里「矩孔」那一格因此长期是空的",
                },
                "aperture_lit_column_count": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "亮列恰好8。与上一行成对：两个数**不相等**，"
                              "于是行列搞混的实现在这里当场露（方形孔看不出来）",
                },
            },
        },
        {
            "id": "oracle:angular_spectrum/self_consistency",
            "inputs": {
                "kind": "transfer_function_self_consistency",
                "count": GRID_A_COUNT,
                "pitch_m": GRID_A_PITCH_M,
                "wavelength_m": WAVELENGTH_M,
                "distance_m": distance_a,
                "transfer_function_max_distance_m": transfer_function_max_distance_m(
                    GRID_A_COUNT, GRID_A_PITCH_M
                ),
                "note": "三条恒等式：z->0退化、半群、能量守恒。"
                        "它们对精确角谱与傍轴形制**一视同仁**——分不开两者，"
                        "所以本案例另有前两条oracle",
            },
            "expected": {
                "zero_distance_deviation_from_the_pure_round_trip": 0.0,
                "zero_distance_deviation_from_the_incident_field": 0.0,
                "semigroup_max_deviation": 0.0,
                "power_ratio_after_propagation": 1.0,
            },
            "tolerances": {
                "zero_distance_deviation_from_the_pure_round_trip": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "``z = 0``时传递函数**逐位等于1**，于是结果与"
                              "``ifft2(fft2(U0))``是同一串浮点数。**零容差不是乐观**："
                              "传递函数里多算一步（例如把``exp(0)``写成别的什么）当场露。"
                              "实测最大偏差返回`0.0`",
                },
                "zero_distance_deviation_from_the_incident_field": {
                    "abs": 1.0e-15,
                    "rel": 0.0,
                    "reason": "``ifft2(fft2(U0))``对`U0`的最大偏差。期望是精确的0"
                              "（往返是恒等），容差按``16 eps max|U0| = 3.6e-15``的"
                              f"误差模型取1e-15（掩模``max|U0| = 1``，实测"
                              f"{MEASURED_ZERO_DISTANCE_DEVIATION:.4e}，余量"
                              f"{1.0e-15 / MEASURED_ZERO_DISTANCE_DEVIATION:.1f}倍）",
                },
                "semigroup_max_deviation": {
                    "abs": 4.0e-15,
                    "rel": 0.0,
                    "reason": "传两次``z/2``与传一次``z``是同一个结果——**角谱有半群性质，"
                              "而单次FFT的菲涅耳形制没有**（两段的输出网格不同）。"
                              "误差模型``64 eps * 峰值振幅 = 64 * eps * 1.148 = 1.6e-14``；"
                              f"实测{MEASURED_SEMIGROUP_DEVIATION:.4e}"
                              f"（余量{4.0e-15 / MEASURED_SEMIGROUP_DEVIATION:.0f}倍）。"
                              "**这条依赖平台的三角函数幅角约化**：本例的``k z = 6.28e3``弧度，"
                              "比0086那条门的``1e5``小一个多数量级，所以本案例对平台的"
                              "敏感度低于`tests/`里那一条；它红了仍然先查平台不要先放宽",
                },
                "power_ratio_after_propagation": {
                    "abs": 1.0e-14,
                    "rel": 0.0,
                    "reason": "网格甲的间距是波长的7.9倍，**所有bin都在光锥内**"
                              "（``max |lambda f| = lambda/(2 dx) = 0.063``），"
                              "于是``|H| = 1``逐bin成立、能量精确守恒。"
                              f"实测偏差{MEASURED_POWER_RATIO_DEVIATION:.4e}"
                              f"（余量{1.0e-14 / MEASURED_POWER_RATIO_DEVIATION:.0f}倍）。"
                              "**与第二条oracle成对**：那里21/32的bin是倏逝的、比值0.8955；"
                              "这里0/32是倏逝的、比值1。同一个传播器、两个物理构型",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/angular_spectrum_propagation",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/angular_spectrum_propagation/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"grid A z={distance_a!r} m, grid B z={distance_b!r} m; "
        f"aperture {len(positions)} samples "
        f"({len({r for r, _ in positions})} rows x {len({c for _, c in positions})} columns); "
        f"evanescent power ratio {evanescent_power_ratio!r} "
        f"(paraxial would give {paraxial_power_ratio!r}); "
        f"{propagating_bin_count}/{GRID_B_COUNT} bins propagating on grid B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
