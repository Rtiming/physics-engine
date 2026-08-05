#!/usr/bin/env python3
"""FTS仪器线型与切趾的金标生成器——**独立算法，不调被验内核**。

被验内核走两条闭式：ILS的半高全宽是一个**写死的常量**
`UNAPODISED_FWHM_IN_SINC_UNITS`，切趾的通量代价是一个**求和闭式**
``sum_i C_i 2^(2i) (i!)^2 / (2i+1)!``。本生成器两条都不走：

* 半高全宽**直接对``sin(pi z) / (pi z) = 1/2``二分求根**——
  于是"那个常量是不是1.2067091288"变成一条真判据，而不是抄写核对；
* 通量代价**用复合Simpson数值积分**切趾窗——于是那个求和闭式的推导
  （每阶积分比1、2/3、8/15、16/35）被独立验一遍。Simpson对三次以下精确，
  窗是x的六次多项式，因此截断误差按h^4走：两档节点自校验，不收敛即拒绝落盘。

Norton-Beer系数在本生成器里**另抄一份**（来源见下），与内核那一份互为对拍。
抄错一位，`coefficient_sums`或`window_at_scan_end`当场红。

参考解出处：
* 无切趾ILS ``sinc(2 L dsigma)``、首零`1/(2L)`、FWHM`1.20671/(2L)`——
  FTS教科书通用结论，与消费方fts-digital-twin
  `assessment/instrument_line_shape.py`的约定一致；
* Norton-Beer三组系数——Norton & Beer 1976（JOSA 66, 259）及1977勘误
  （JOSA 67, 419）；同值见Naylor & Tahic 2007（JOSA A 24, 3644）表1。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/fts_instrument_line_shape"
ALGORITHM_VERSION = "1.0.0"

#: 单边最大光程差（米）。5 cm对应分辨率1/(2L)=10周/米=0.1 cm^-1。
MAX_OPD_M = 0.05

#: 教科书四舍五入到第五位小数的半高全宽因子。判"实现与教科书一致"到这一位为止。
TEXTBOOK_FWHM = 1.20671

#: 教科书值的四舍五入半宽：末位是1e-5，半宽5e-6。
TEXTBOOK_HALF_STEP = 5.0e-6

#: Norton-Beer系数**另抄一份**（见模块docstring的出处），与内核那份互为对拍。
COEFFICIENTS: dict[str, tuple[float, float, float, float]] = {
    "weak": (0.384093, -0.087577, 0.703484, 0.0),
    "medium": (0.152442, -0.136176, 0.983734, 0.0),
    "strong": (0.045335, 0.0, 0.554883, 0.399782),
}
STRENGTHS: tuple[str, ...] = ("weak", "medium", "strong")

#: Simpson的两档节点（都取偶数）与收敛地板。
SIMPSON_NODES = 20000
SIMPSON_FLOOR = 1.0e-15

#: 求窗最小值时读侧扫的等距点数（含两端）。判据容差由它的间距算出来。
MINIMUM_SCAN_POINTS = 4096


def window(reduced_opd: float, coefficients: tuple[float, ...]) -> float:
    """``A(x) = sum_i C_i (1 - x^2)^i``，`x = Delta / L`；扫描区间外为0。"""

    if abs(reduced_opd) > 1.0:
        return 0.0
    base = 1.0 - reduced_opd * reduced_opd
    return math.fsum(
        coefficient * base**index for index, coefficient in enumerate(coefficients)
    )


def simpson(coefficients: tuple[float, ...], nodes: int) -> float:
    """``int_0^1 A(x) dx``（偶函数，半区间即全部）。`fsum`把累加舍入压到一次。"""

    step = 1.0 / nodes
    terms = [window(0.0, coefficients), window(1.0, coefficients)]
    terms.extend(
        (4.0 if index % 2 else 2.0) * window(index * step, coefficients)
        for index in range(1, nodes)
    )
    return math.fsum(terms) * step / 3.0


def converged_throughput(coefficients: tuple[float, ...]) -> float:
    """两档Simpson互相印证；不收敛即拒绝落盘。"""

    coarse = simpson(coefficients, SIMPSON_NODES // 2)
    fine = simpson(coefficients, SIMPSON_NODES)
    if abs(coarse - fine) > SIMPSON_FLOOR:
        raise SystemExit(f"Simpson未收敛：两档差{abs(coarse - fine)!r} > {SIMPSON_FLOOR!r}")
    return fine


def window_minimum(coefficients: tuple[float, ...]) -> tuple[float, float, float]:
    """窗在扫描区间上的**闭式**最小值：`(x*, A(x*), |d2A/dx2|(x*))`。

    以``t = 1 - x^2``为自变量，``A(t) = C0 + C1 t + C2 t^2 + C3 t^3``单调地
    随t增大而增大**只有在C1 >= 0时才成立**。已发表的weak与medium两组
    ``C1 < 0``，所以它们在扫描端点附近有一个**真实的内部极小**，
    过了它窗会回升到`A(L) = C0`。这不是bug是这两组系数的性质
    （Norton-Beer从未要求切趾窗单调），本判据把它钉成一条可断言的数。

    驻点解``C1 + 2 C2 t + 3 C3 t^2 = 0``；取落在(0,1)内的根与两端点里最小的那个。
    """

    c0, c1, c2, c3 = coefficients
    candidates = [0.0, 1.0]
    if c3 == 0.0:
        if c2 != 0.0:
            candidates.append(-c1 / (2.0 * c2))
    else:
        discriminant = 4.0 * c2 * c2 - 12.0 * c3 * c1
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            candidates.extend(
                (-2.0 * c2 + sign * root) / (6.0 * c3) for sign in (1.0, -1.0)
            )
    best_t = min(
        (t for t in candidates if 0.0 <= t <= 1.0),
        key=lambda t: c0 + c1 * t + c2 * t * t + c3 * t**3,
    )
    best_x = math.sqrt(max(0.0, 1.0 - best_t))
    # d2A/dx2 = -2 (C1 + 2 C2 t + 3 C3 t^2) + 4 x^2 (2 C2 + 6 C3 t)；
    # 驻点上第一项为零（内部极小）或x=1（端点）。
    slope_in_t = c1 + 2.0 * c2 * best_t + 3.0 * c3 * best_t * best_t
    curvature = abs(
        -2.0 * slope_in_t + 4.0 * best_x * best_x * (2.0 * c2 + 6.0 * c3 * best_t)
    )
    return best_x, c0 + c1 * best_t + c2 * best_t**2 + c3 * best_t**3, curvature


def normalised_sinc(z: float) -> float:
    """``sin(pi z) / (pi z)``，`sinc(0) = 1`。生成器自己的一份，不import内核。"""

    if z == 0.0:
        return 1.0
    argument = math.pi * z
    return math.sin(argument) / argument


def bisect_half_maximum() -> float:
    """``sinc(z) = 1/2``的正根。半高全宽是它的两倍（sinc是偶函数）。"""

    low, high = 0.5, 0.7
    for _ in range(100):
        middle = 0.5 * (low + high)
        if (normalised_sinc(low) - 0.5) * (normalised_sinc(middle) - 0.5) <= 0.0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def main() -> int:
    half_maximum_z = bisect_half_maximum()
    fwhm_in_sinc_units = 2.0 * half_maximum_z
    textbook_gap = abs(fwhm_in_sinc_units - TEXTBOOK_FWHM)
    if textbook_gap > TEXTBOOK_HALF_STEP:
        raise SystemExit(f"求根值{fwhm_in_sinc_units!r}与教科书{TEXTBOOK_FWHM}差{textbook_gap!r}")

    double_sided = 2.0 * MAX_OPD_M
    first_zero_per_m = 1.0 / double_sided
    fwhm_per_m = fwhm_in_sinc_units / double_sided

    sums = [math.fsum(COEFFICIENTS[name]) for name in STRENGTHS]
    at_zero = [window(0.0, COEFFICIENTS[name]) for name in STRENGTHS]
    at_end = [window(1.0, COEFFICIENTS[name]) for name in STRENGTHS]
    beyond = [window(1.5, COEFFICIENTS[name]) for name in STRENGTHS]
    throughput = [converged_throughput(COEFFICIENTS[name]) for name in STRENGTHS]

    minima = [window_minimum(COEFFICIENTS[name]) for name in STRENGTHS]
    minimum_positions = [position for position, _, _ in minima]
    minimum_values = [value for _, value, _ in minima]
    # 读侧只能在等距网格上取最小；网格半间距是位置误差上界，
    # 值误差是 (1/2)|A''| delta^2。两条容差都由它们算出来。
    half_spacing = 0.5 / (MINIMUM_SCAN_POINTS - 1)
    value_bound = max(
        0.5 * curvature * half_spacing * half_spacing for _, _, curvature in minima
    )
    if value_bound > 1.0e-7 or half_spacing > 3.0e-4:
        raise SystemExit(f"网格误差界{value_bound!r}/{half_spacing!r}超出写死的容差")

    oracles = [
        {
            "id": "oracle:fts/unapodised_line_shape",
            "inputs": {
                "kind": "boxcar_instrument_line_shape",
                "max_opd_m": MAX_OPD_M,
                "textbook_fwhm_in_sinc_units": TEXTBOOK_FWHM,
                "textbook_half_step": TEXTBOOK_HALF_STEP,
            },
            "expected": {
                "fwhm_in_sinc_units": fwhm_in_sinc_units,
                "fwhm_matches_textbook_quote": True,
                "first_zero_per_m": first_zero_per_m,
                "fwhm_per_m": fwhm_per_m,
                "line_shape_at_peak": 1.0,
                "line_shape_at_first_zero": 0.0,
                "line_shape_at_plus_half_fwhm": 0.5,
                "line_shape_at_minus_half_fwhm": 0.5,
            },
            "tolerances": {
                "fwhm_in_sinc_units": {
                    "abs": 1.0e-15, "rel": 0.0,
                    "reason": "内核里是写死的常量，本金标是**独立二分求根**的结果——"
                              "所以这条真的在验那个常量。二分100次把区间压到浮点地板"
                              "（0.2/2^100远小于eps），残差只剩sinc求值的1eps量级；"
                              "1e-15约4.5eps",
                },
                "fwhm_matches_textbook_quote": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：|因子 - 1.20671| <= 5e-6（教科书末位1e-5的四舍五入半宽）。"
                              "实测8.712e-7。这条锁的是research/05第2.3节写下的那个判据数，"
                              "与上一条分工：上一条验精度，这条验没写错数",
                },
                "first_zero_per_m": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "1/(2L)，一次乘一次除；1e-15约4.5eps。"
                              "**这条同时是L与2L的捕手**：把单边最大光程差当成全程，"
                              "首零差2倍且看起来完全合理",
                },
                "fwhm_per_m": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "因子/(2L)，两次乘除；同上",
                },
                "line_shape_at_peak": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "sinc(0)=1是极限的精确值，实现直接返回1.0不走除法——零容差",
                },
                "line_shape_at_first_zero": {
                    "abs": 1.0e-16, "rel": 0.0,
                    "reason": "sinc(1)在float64上不是0而是sin(pi_float)/pi_float："
                              "sin(pi的浮点表示)=1.2246e-16，除以pi得3.898e-17。"
                              "**容差是算出来的**：1e-16是这个必然残值的2.6倍，"
                              "不是'放宽到能过'",
                },
                "line_shape_at_plus_half_fwhm": {
                    "abs": 1.0e-15, "rel": 0.0,
                    "reason": "半高点上ILS必须恰为0.5——**这条才是半高全宽那个常量的物理门**"
                              "（常量错了这里就不是0.5）。实测0.4999999999999999，"
                              "偏差1.11e-16即0.5eps；1e-15约4.5eps",
                },
                "line_shape_at_minus_half_fwhm": {
                    "abs": 1.0e-15, "rel": 0.0,
                    "reason": "负侧同值——ILS是偶函数。两侧都验是因为只验一侧的话，"
                              "把偏移取了绝对值的实现与真正对称的实现分不开",
                },
            },
        },
        {
            "id": "oracle:fts/norton_beer_apodisation",
            "inputs": {
                "kind": "norton_beer_windows",
                "strengths": list(STRENGTHS),
                "max_opd_m": MAX_OPD_M,
                "simpson_nodes": SIMPSON_NODES,
                "simpson_floor": SIMPSON_FLOOR,
                "beyond_scan_factor": 1.5,
                "minimum_scan_points": MINIMUM_SCAN_POINTS,
            },
            "expected": {
                "coefficient_sums": sums,
                "window_at_zero_opd": at_zero,
                "window_at_scan_end": at_end,
                "window_beyond_scan": beyond,
                "throughput": throughput,
                "throughput_strictly_decreasing": True,
                "window_minimum_reduced_opd": minimum_positions,
                "window_minimum_value": minimum_values,
            },
            "tolerances": {
                "coefficient_sums": {
                    "abs": 1.0e-6, "rel": 0.0,
                    "reason": "**每组sum(Ci)=1是一条门**（research/05第2.3节）。"
                              "它是物理必要条件：A(0)=sum(Ci)，零光程差点承载全部通量，"
                              "sum!=1会把整条谱线整体缩放。1e-6是文献系数的申报位数"
                              "（六位小数），也就是'这组数确实求和为一'能被断言的分辨率。"
                              "本仓三组实测残差**恰为0**（十进制系数在二进制下相加正好落回1.0）",
                },
                "window_at_zero_opd": {
                    "abs": 1.0e-6, "rel": 0.0,
                    "reason": "A(0)=sum(Ci)的另一面：上一条验系数表，这一条验窗函数"
                              "真的在Delta=0处取到那个和（幂次写错、区间判断写反都会在这里露出）。"
                              "容差同源",
                },
                "window_at_scan_end": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "A(L)=C0**逐位相等**：x=1时(1-x^2)=0，高阶项全灭，"
                              "只剩C0，没有任何算术发生。零容差。"
                              "这个残值越小切趾越强：弱0.384 > 中0.152 > 强0.045",
                },
                "window_beyond_scan": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "扫描区间外没有数据，窗**是0不是延拓**。零容差。"
                              "不写这条的话，一个把(1-x^2)^i在|x|>1上继续算的实现"
                              "会给出正负乱跳的'窗'，且在别的判据上全绿",
                },
                "throughput": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "内核走求和闭式sum_i C_i 2^(2i)(i!)^2/(2i+1)!，"
                              "金标走复合Simpson数值积分——**两条不同的路**，"
                              "所以这条验的是那个闭式推导本身。Simpson对三次以下精确、"
                              "窗是六次多项式，h=5e-5时截断约3e-18，远低于浮点地板；"
                              "两条路的实测差<=1.11e-16（<=0.5eps）。1e-15约4.5eps",
                },
                "throughput_strictly_decreasing": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔排序判据（spec/12第6.2节写法2）：弱>中>强，"
                              "**越强的切趾扔掉越多信号**。排序不受实现常数影响。"
                              "假通过口子是三者全零时的0>0>0——所以测试里先各自断言"
                              "落在(0,1)开区间，再断严格不等号",
                },
                "window_minimum_reduced_opd": {
                    "abs": 3.0e-4, "rel": 0.0,
                    "reason": f"读侧在{MINIMUM_SCAN_POINTS}个等距点上取最小，"
                              f"位置误差上界是网格半间距{half_spacing:.3e}；"
                              "3e-4是它的2.5倍。**weak与medium的极小落在扫描端点内侧**"
                              "（x*≈0.968与0.965），strong的落在端点上（x*=1）——"
                              "这个差别只由C1的符号决定，是C1抄错的直接捕手",
                },
                "window_minimum_value": {
                    "abs": 1.0e-7, "rel": 0.0,
                    "reason": "网格取最小的值误差是(1/2)|A''|delta^2，"
                              f"三组里最大的界是{value_bound:.3e}；1e-7约它的1.8倍。"
                              "**这条判的是一件反直觉的真事**：weak与medium的切趾窗"
                              "在扫描端点附近**不单调**——C1为负，窗先跌到极小"
                              "（比A(L)低2.7e-3与4.7e-3）再回升到C0。"
                              "Norton-Beer从未要求窗单调；本仓第一版把"
                              "'窗单调不增'写成判据，当场红，红得对",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/fts_instrument_line_shape",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/fts_instrument_line_shape/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"fwhm factor {fwhm_in_sinc_units!r} (textbook gap {textbook_gap:.4e}), "
        f"throughput {throughput!r}, "
        f"window minima {list(zip(minimum_positions, minimum_values, strict=True))!r}, "
        f"grid value bound {value_bound:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
