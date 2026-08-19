#!/usr/bin/env python3
"""双缝夫琅禾费图样的金标生成器——**独立算法，不调被验内核**。

被验内核走的是基2 Cooley-Tukey（`optics/field.py`的`fft2`）：位反转重排、
``log2(N)``级蝶形、缓存的旋转因子表。本生成器走**定义式直接求和**

    U(k) = sum_{n in 孔内} exp(-2 pi i n k / N)

孔内只有8个采样，所以这条路便宜到可以逐bin算；它与蝶形**没有任何共用代码**，
也不共用旋转因子表。这正是research/05第2.3节光学族第13条要的
"**MFT-FFT等价**"——矩阵式（定义式）变换与快速变换必须给同一个答案。

生成器**不import `physics_engine.optics`**，孔径的采样位置由它自己按几何算，
不调`rectangular_aperture`。

金标里的五样：

1. `relative_amplitude_components`：若干bin上的**复振幅比**``U(k)/U(0)``，
   按决策0086的复数字节形制落成``[实部, 虚部]``二元组。
   **比值而不是绝对振幅**：绝对值里含一个``exp(i k z)``，
   ``k z ~ 1.99e7``弧度，两侧都只是在调`cmath.exp`同一个大幅角——
   冻结它测的是libm不是引擎。比值把它精确消掉，留下的是变换、观察面
   二次相位与坐标映射，那三样才是被验的东西；
2. `normalised_intensity`：条纹各级的归一化强度；
3. `fringe_sines`：条纹极大的``sin(theta) = m lambda / d``（连续闭式，
   `d`是中心距）；
4. `missing_order`：``d / w = 8``时第8级被单缝包络的第一个零点压掉，
   归一化强度**恰为0**；
5. `flux_and_reversibility`：通量守恒与往返可逆（同为第13条三件套的另两件）。

参考解出处：Born & Wolf《Principles of Optics》第8.5.1节（矩孔与多缝的
夫琅禾费衍射）；离散孔径的变换是Dirichlet核
``sum_{n=0}^{M-1} exp(-2 pi i n k / N) = exp(...) sin(pi M k/N)/sin(pi k/N)``，
其零点在``k = m N / M``——本案例的缝宽与中心距都取2的幂，
于是零点与条纹极大**都落在整数bin上**，可以零容差地判。
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

ALGORITHM_ID = "algorithm:oracle/double_slit_propagated"
ALGORITHM_VERSION = "1.0.0"

#: 网格与光路。行数取8让二维路径真的被走到；缝在y方向占满窗口。
COLUMNS = 256
ROWS = 8
PITCH_M = 10.0e-6
WAVELENGTH_M = 632.8e-9
SCREEN_DISTANCE_M = 2.0

#: 每缝4个采样、中心距32个采样 → ``d / w = 8``，第8级缺级。
SLIT_SAMPLES = 4
SEPARATION_SAMPLES = 32

#: 半宽偏四分之一格，让"边界上那个采样点归谁"没有歧义
#: （引擎侧对落在采样点上的边界**失败关闭**，见`propagation.py`）。
EDGE_OFFSET_IN_PITCHES = 0.25

#: 冻结复振幅比的bin：条纹极大（8的倍数）、条纹极小（4、12、20）、缺级（64）。
AMPLITUDE_BINS: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)

#: 被验实现与本生成器之间的**实测**最坏偏差（本机，见案例页第三节）。
#: 容差由它们乘余量得到——**容差是算出来的不是拍的**（0024第三节）。
MEASURED_AMPLITUDE_DEVIATION = 2.7756e-16
MEASURED_INTENSITY_DEVIATION = 2.2204e-16
MEASURED_FRINGE_SINE_DEVIATION = 1.7347e-18
MEASURED_POWER_RATIO_DEVIATION = 6.0e-16
MEASURED_ROUND_TRIP_DEVIATION = 1.4016e-16

#: 冻结归一化强度的条纹级次。第7级之后包络已经压到很低，第8级是缺级。
FRINGE_ORDERS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)


def signed_index(index: int, count: int) -> int:
    """FFT次序的第`index`格对应的有符号序号。生成器自己算，不调引擎的helper。"""

    half = count // 2
    return index if index < half else index - count


def aperture_indices() -> tuple[int, ...]:
    """两条缝覆盖的采样点（有符号序号），**按几何算出来**。

    每缝中心在``(0.5 -+ P/2) * dx``、半宽``((M-1)/2 + 0.25) * dx``，
    于是各覆盖恰好`M`个采样。中心偏半格是必须的：以采样点为中心的对称孔
    覆盖的采样数必然是奇数，而我们要2的幂。
    """

    half_width = (SLIT_SAMPLES - 1) / 2.0 + EDGE_OFFSET_IN_PITCHES
    selected: list[int] = []
    for sign in (-1.0, 1.0):
        centre = 0.5 + sign * SEPARATION_SAMPLES / 2.0
        for index in range(COLUMNS):
            position = signed_index(index, COLUMNS)
            if abs(position - centre) <= half_width:
                selected.append(position)
    return tuple(sorted(selected))


def direct_transform(positions: tuple[int, ...], bin_index: int) -> complex:
    """定义式直接求和``sum_n exp(-2 pi i n k / N)``。

    相位先按**整数**``(n k) mod N``约化再进`exp`：整数取模精确，
    而``exp``的大幅角约化不精确。不这么写，这条参照自己的误差会盖过被验对象
    （同样的坑在`tests/test_optics_field.py`的平移定理上实测过，差65倍）。
    """

    total = complex(0.0, 0.0)
    for position in positions:
        phase = -2.0 * math.pi * ((position * bin_index) % COLUMNS) / COLUMNS
        total += cmath.exp(complex(0.0, phase))
    return total


def main() -> int:
    positions = aperture_indices()
    if len(positions) != 2 * SLIT_SAMPLES:
        raise SystemExit(f"孔内采样数{len(positions)}不是期望的{2 * SLIT_SAMPLES}，不落盘")

    observation_pitch_m = WAVELENGTH_M * SCREEN_DISTANCE_M / (COLUMNS * PITCH_M)
    reduced = WAVELENGTH_M * SCREEN_DISTANCE_M

    def relative_amplitude(bin_index: int) -> complex:
        """``U(k)/U(0)``：全局前因子与``exp(ikz)``精确消掉，观察面二次相位留下。"""

        coordinate = signed_index(bin_index, COLUMNS) * observation_pitch_m
        chirp = cmath.exp(complex(0.0, math.pi * coordinate * coordinate / reduced))
        return direct_transform(positions, bin_index) / direct_transform(positions, 0) * chirp

    amplitudes = [relative_amplitude(index) for index in AMPLITUDE_BINS]

    #: 直接求和在缺级那一格的残值——它是"恰为0"这条判据的容差来源。
    missing_index = COLUMNS // SLIT_SAMPLES
    missing_residual = abs(direct_transform(positions, missing_index)) / len(positions)

    width_m = SLIT_SAMPLES * PITCH_M
    separation_m = SEPARATION_SAMPLES * PITCH_M
    fringe_step = COLUMNS // SEPARATION_SAMPLES

    intensities = []
    sines = []
    for order in FRINGE_ORDERS:
        value = relative_amplitude(order * fringe_step)
        intensities.append(value.real * value.real + value.imag * value.imag)
        sines.append(order * WAVELENGTH_M / separation_m)

    oracles = [
        {
            "id": "oracle:double_slit/relative_amplitude_table",
            "inputs": {
                "kind": "direct_summation_over_aperture_samples",
                "columns": COLUMNS,
                "rows": ROWS,
                "pitch_m": PITCH_M,
                "wavelength_m": WAVELENGTH_M,
                "screen_distance_m": SCREEN_DISTANCE_M,
                "slit_samples": SLIT_SAMPLES,
                "separation_samples": SEPARATION_SAMPLES,
                "edge_offset_in_pitches": EDGE_OFFSET_IN_PITCHES,
                "bins": list(AMPLITUDE_BINS),
                "aperture_positions": list(positions),
                "complex_component_order": ["real", "imaginary"],
            },
            "expected": {
                "relative_amplitude_components": [
                    [value.real, value.imag] for value in amplitudes
                ],
                "aperture_sample_count": float(len(positions)),
            },
            "tolerances": {
                "relative_amplitude_components": {
                    "abs": 4.0e-15,
                    "rel": 0.0,
                    "reason": "定义式直接求和 对 基2蝶形。两侧各自的浮点误差："
                              "直接求和是8项累加（~8 eps），蝶形是log2(256)=8级"
                              "（~8 eps），比值再除一次；观察面二次相位两侧"
                              "同一条闭式但幅角不同。合起来~30 eps=6.7e-15的量级，"
                              f"实测最坏{MEASURED_AMPLITUDE_DEVIATION:.4e}"
                              f"（余量{4.0e-15 / MEASURED_AMPLITUDE_DEVIATION:.1f}倍）。"
                              "**判绝对不判相对**：条纹极小处比值近零，相对误差没有意义",
                },
                "aperture_sample_count": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "孔内采样数是整数，零容差。它锁的是**采样缝宽**——"
                              "边界落在采样点上时这个数会整差一格（实测踩过："
                              "期望32个采样、实得31个），而图样照样是漂亮的sinc平方",
                },
            },
        },
        {
            "id": "oracle:double_slit/fringes_and_missing_order",
            "inputs": {
                "kind": "closed_form_fringe_positions",
                "orders": list(FRINGE_ORDERS),
                "fringe_bin_step": fringe_step,
                "slit_width_m": width_m,
                "slit_separation_m": separation_m,
                "missing_order": SEPARATION_SAMPLES // SLIT_SAMPLES,
                "missing_order_bin": missing_index,
                "direct_summation_residual_at_missing_order": missing_residual,
            },
            "expected": {
                "fringe_sines": sines,
                "normalised_intensity": intensities,
                "missing_order_intensity": 0.0,
                "envelope_first_zero_sine": WAVELENGTH_M / width_m,
            },
            "tolerances": {
                "fringe_sines": {
                    "abs": 0.0,
                    "rel": 4.0e-16,
                    "reason": "条纹极大在``sin(theta) = m lambda / d``。观察面坐标是"
                              "``k lambda z /(N dx)``，在``k = m N / P``处约掉z与N，"
                              "**恒等地**落回``m lambda /(P dx)``——这是一条"
                              "单位边界的往返，理论偏差为0，容差只留2 eps给"
                              f"两侧不同的乘除次序。实测最坏{MEASURED_FRINGE_SINE_DEVIATION:.4e}"
                              "，落在第6级上（余量2.7倍）",
                },
                "normalised_intensity": {
                    "abs": 4.0e-15,
                    "rel": 0.0,
                    "reason": "强度是复振幅比的模平方，误差是振幅误差的2倍"
                              "（|z|^2的相对误差=2倍|z|的），而振幅比的量级<=1，"
                              f"所以与上一条同量级。实测最坏{MEASURED_INTENSITY_DEVIATION:.4e}"
                              f"（余量{4.0e-15 / MEASURED_INTENSITY_DEVIATION:.1f}倍）",
                },
                "missing_order_intensity": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "``d/w = 8``时第8级条纹极大恰好压在单缝包络的第一个"
                              "零点上。**零容差不是乐观是结构性的**：缝宽4是2的幂、"
                              "该bin是``N/4``，蝶形最后一级配对的两项相位差恰为pi"
                              "（精确取负），IEEE下逐位抵消，实测返回`0.0`。"
                              f"直接求和那一侧的残值是{missing_residual:.3e}，"
                              "两条路都到地板。这条一旦不为0，说明变换的结构变了",
                },
                "envelope_first_zero_sine": {
                    "abs": 0.0,
                    "rel": 4.0e-16,
                    "reason": "包络首零在``sin(theta) = lambda / w``，`w`是**缝宽**"
                              "不是中心距。把两者搞反的实现在前6级条纹上全绿，"
                              "只有缺级与这一条会红",
                },
            },
        },
        {
            "id": "oracle:double_slit/flux_and_reversibility",
            "inputs": {
                "kind": "transform_layer_self_consistency",
                "note": "research/05第2.3节光学族第13条的三件套里的另两件："
                        "通量守恒与往返可逆（MFT-FFT等价是上面第一条oracle）",
            },
            "expected": {
                "power_ratio_after_propagation": 1.0,
                "round_trip_max_deviation": 0.0,
            },
            "tolerances": {
                "power_ratio_after_propagation": {
                    "abs": 1.0e-14,
                    "rel": 0.0,
                    "reason": "菲涅耳单次FFT的前因子``dx dy/(i lambda z)``与观察面"
                              "间距``lambda z/(N dx)``在总能量上恰好互相抵消，"
                              "所以比值的期望是精确的1。误差只来自Parseval的浮点"
                              f"累积（实测{MEASURED_POWER_RATIO_DEVIATION:.1e}，"
                              f"余量{1.0e-14 / MEASURED_POWER_RATIO_DEVIATION:.0f}倍）。"
                              "**这条是前因子唯一的捕手**：前因子写错时归一化图样一个字都不变",
                },
                "round_trip_max_deviation": {
                    "abs": 1.0e-15,
                    "rel": 0.0,
                    "reason": "``ifft2(fft2(U0))``对`U0`的最大偏差。期望是精确的0"
                              "（往返是恒等），容差按``16 eps max|U0|=3.6e-15``的"
                              f"误差模型取1e-15（掩模的max|U0|=1，实测"
                              f"{MEASURED_ROUND_TRIP_DEVIATION:.4e}，余量"
                              f"{1.0e-15 / MEASURED_ROUND_TRIP_DEVIATION:.1f}倍）",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/double_slit_propagated",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/double_slit_propagated/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"aperture samples {positions}, "
        f"missing order {SEPARATION_SAMPLES // SLIT_SAMPLES} at bin {missing_index}, "
        f"direct-sum residual there {missing_residual:.4e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
