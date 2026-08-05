#!/usr/bin/env python3
"""艾里斑的金标生成器——**独立算法，不调被验内核**。

被验内核用"上升级数 + Hankel渐近展开"求`J1`。本生成器用**完全不同的一条路**：
贝塞尔积分的周期梯形求值

    J_n(x) = (1/pi) * int_0^pi cos(n*t - x*sin(t)) dt

被积函数在整个复平面解析且以2pi为周期，周期解析函数上的梯形法**几何收敛**——
所以这条路在float64上直接到浮点地板，且与级数/渐近展开没有任何共用代码。
生成器自己验这一点：同一个点算N与2N两档，互差超过`REFERENCE_FLOOR`即拒绝落盘。

金标里的三样：

1. `J1`与``E(x) = 2 J1(x)/x``在16个自变量上的参考值（覆盖级数段、
   分段点两侧、渐近段，含实测最坏点x=12.028）；
2. 首零：生成器**二分自己的参考`J1`**求出来，与文献引用值3.8317059702对照；
3. 单位边界的往返：``theta_1 = arcsin(1.2197 lambda / D)``算出的角度再喂回
   ``x = 2 pi a sin(theta) / lambda``必须落回首零——半径/直径、
   角度/空间频率这两道换算错了，这条往返立刻不闭合。

参考解出处：Born & Wolf《Principles of Optics》第8.5.2节；
首零文献值取Abramowitz & Stegun表9.5的`j_{1,1}`。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/scalar_diffraction_airy"
ALGORITHM_VERSION = "1.0.0"

#: 周期梯形的节点数与自校验档。两档互差超过地板即拒绝落盘。
REFERENCE_NODES = 4096
REFERENCE_FLOOR = 1.0e-15

#: 采样点：级数段（<=12）、分段点两侧、渐近段，含实测最坏点12.028。
SAMPLE_ARGUMENTS: tuple[float, ...] = (
    0.0, 0.5, 1.0, 2.0, 3.8317059702, 5.0, 8.0, 11.9,
    12.0, 12.028, 12.1, 14.0, 20.0, 30.0, 50.0, 60.0,
)

#: 文献引用的首零（A&S表9.5，截到1e-10）。research/05第2.3节列的正是这个数。
LITERATURE_FIRST_ZERO = 3.8317059702

#: 引用值末位的量级：判"实现的常量与文献引用一致"到这一位为止。
LITERATURE_QUOTE_STEP = 1.0e-10

#: 单位边界往返用的构型：HeNe 632.8 nm 打在 1 mm 圆孔上。
WAVELENGTH_M = 632.8e-9
APERTURE_DIAMETER_M = 1.0e-3


def bessel_reference(order: int, x: float, nodes: int) -> float:
    """``J_order(x)``的周期梯形求值（中点形式，周期函数上二者等价）。"""

    if x == 0.0:
        return 1.0 if order == 0 else 0.0
    total = math.fsum(
        math.cos(order * t - x * math.sin(t))
        for t in (math.pi * (index + 0.5) / nodes for index in range(nodes))
    )
    return total / nodes


def converged_reference(order: int, x: float) -> float:
    """两档求值互相印证；不收敛即拒绝落盘（金标不许带未申报的不确定度）。"""

    coarse = bessel_reference(order, x, REFERENCE_NODES // 4)
    fine = bessel_reference(order, x, REFERENCE_NODES)
    if abs(coarse - fine) > REFERENCE_FLOOR:
        raise SystemExit(
            f"参考求值未收敛：J{order}({x}) 两档差{abs(coarse - fine)!r}"
            f" > {REFERENCE_FLOOR!r}，不落盘"
        )
    return fine


def amplitude_reference(x: float) -> float:
    """``E(x) = 2 J1(x) / x``，`E(0) = 1`（可去奇点，极限精确为1）。"""

    return 1.0 if x == 0.0 else 2.0 * converged_reference(1, x) / x


def bisect_first_zero(low: float, high: float) -> float:
    """在参考`J1`上二分首零。100次足以把区间压到浮点地板。"""

    f_low = converged_reference(1, low)
    for _ in range(100):
        middle = 0.5 * (low + high)
        if f_low * converged_reference(1, middle) <= 0.0:
            high = middle
        else:
            low, f_low = middle, converged_reference(1, middle)
    return 0.5 * (low + high)


def main() -> int:
    j1_values = [converged_reference(1, x) for x in SAMPLE_ARGUMENTS]
    amplitudes = [amplitude_reference(x) for x in SAMPLE_ARGUMENTS]

    first_zero = bisect_first_zero(3.0, 4.5)
    truncation = abs(first_zero - LITERATURE_FIRST_ZERO)
    if truncation > LITERATURE_QUOTE_STEP:
        raise SystemExit(f"参考首零{first_zero!r}与文献引用差{truncation!r}，超出引用位数")

    # E在首零处的斜率：|dE/dx| = |2 J0(x1) / x1|（因为J1(x1)=0）。
    # 文献引用值被截断了truncation，所以E(引用值)必然偏离零约 斜率*截断量。
    slope = abs(2.0 * converged_reference(0, first_zero) / first_zero)
    predicted_residual = slope * truncation

    # 单位边界往返：角度→空间频率→艾里自变量，必须落回首零。
    # **因子用的是文献引用值不是上面二分出的真值**：引擎的1.22因子按定义就是
    # 引用值除以pi，金标若换成真值，这条判据测的就变成"那个引用值被截了7.5e-12"
    # 而不是单位换算——第一版正是这么写的，当场红。独立性要求的是**算法**不同，
    # 不是把物理常数换掉。
    factor = LITERATURE_FIRST_ZERO / math.pi  # sin(theta_1) = factor * lambda / D
    sine = factor * WAVELENGTH_M / APERTURE_DIAMETER_M
    half_angle_rad = math.asin(sine)

    oracles = [
        {
            "id": "oracle:airy/bessel_j1_reference_table",
            "inputs": {
                "kind": "bessel_integral_periodic_trapezoid",
                "arguments": list(SAMPLE_ARGUMENTS),
                "reference_nodes": REFERENCE_NODES,
                "reference_floor": REFERENCE_FLOOR,
            },
            "expected": {"j1_values": j1_values, "airy_amplitude_values": amplitudes},
            "tolerances": {
                "j1_values": {
                    "abs": 1.0e-12, "rel": 0.0,
                    "reason": "被验实现申报的绝对精度（bessel.py的J1_ABSOLUTE_ACCURACY）。"
                              "**判绝对不判相对**：J1有零点，x=3.83处值3e-12而绝对误差9e-17，"
                              "相对误差3e-5毫无意义。实测最坏8.27e-13落在x=12.028（分段点右侧），"
                              "余量1.2倍——收到5e-13会红，红的是渐近展开在分段点附近的固有误差",
                },
                "airy_amplitude_values": {
                    "abs": 5.0e-13, "rel": 0.0,
                    "reason": "E = 2 J1/x，除以x把J1的绝对误差按1/x缩小；"
                              "最坏点仍是x=12.028，实测1.38e-13。取5e-13是实测的3.6倍",
                },
            },
        },
        {
            "id": "oracle:airy/first_zero_and_units",
            "inputs": {
                "kind": "airy_first_zero_and_unit_round_trip",
                "literature_first_zero": LITERATURE_FIRST_ZERO,
                "literature_quote_step": LITERATURE_QUOTE_STEP,
                "wavelength_m": WAVELENGTH_M,
                "aperture_diameter_m": APERTURE_DIAMETER_M,
                "bisection_bracket": [3.0, 4.5],
            },
            "expected": {
                "first_zero_x": first_zero,
                "constant_matches_literature_quote": True,
                "amplitude_at_axis": 1.0,
                "amplitude_at_literature_zero": 0.0,
                "intensity_at_literature_zero": 0.0,
                "first_minimum_half_angle_rad": half_angle_rad,
                "argument_at_first_minimum": LITERATURE_FIRST_ZERO,
            },
            "tolerances": {
                "first_zero_x": {
                    "abs": 1.0e-14, "rel": 0.0,
                    "reason": "两侧都二分到浮点地板，差异只来自J1求值本身："
                              "首零邻域`J1`走级数段无相消，绝对误差约1e-16，"
                              f"经|J1'(x1)|={slope * first_zero / 2.0:.5f}折算成"
                              "自变量偏差约2.5e-16。实测4.4e-16，取1e-14是它的23倍",
                },
                "constant_matches_literature_quote": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：模块常量AIRY_FIRST_ZERO_X与A&S表9.5引用值"
                              "逐位相同（引用值截到1e-10）。这条锁的是research/05第2.3节"
                              "写下的那个判据数没有在实现里被改掉",
                },
                "amplitude_at_axis": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "E(0)=1是极限的精确值，实现里直接返回1.0不走除法——"
                              "零容差；写成除法会给出一个带舍入的1",
                },
                "amplitude_at_literature_zero": {
                    "abs": 3.0e-12, "rel": 0.0,
                    "reason": f"物理上E在首零处**恰为0**。偏差全部来自文献引用值被截断了"
                              f"{truncation:.3e}：|dE/dx|={slope:.5f}，"
                              f"预测残值{predicted_residual:.4e}，实测1.5793e-12与之吻合。"
                              "取3e-12≈预测值的1.9倍。**容差是算出来的**："
                              "换一个位数更多的引用值它会更小，而不是反过来调这个数",
                },
                "intensity_at_literature_zero": {
                    "abs": 1.0e-23, "rel": 0.0,
                    "reason": "I = E^2，容差是振幅容差的平方(3e-12)^2=9e-24。"
                              "实测2.49e-24。这条与上一条不是重复：它验的是"
                              "intensity真的是amplitude的平方而不是另写了一遍",
                },
                "first_minimum_half_angle_rad": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "arcsin(x1/pi * lambda/D)，三次乘除一次arcsin，"
                              "各1eps量级；1e-15约4.5eps",
                },
                "argument_at_first_minimum": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "**单位边界的往返**：角度→sin/lambda（空间频率，周/米）→"
                              "乘2pi*a（角波数×半径）必须落回首零。把半径写成直径差2倍、"
                              "把谱学波数当角波数差2pi倍——两者都不会报错，只有这条往返能抓。"
                              "**期望值是文献引用值不是真值**：内核的1.22因子由被截过的引用值"
                              "除以pi得到，所以往返闭合在引用值上，"
                              f"与真零点差{truncation:.3e}（第一次写成真值时这条当场红，"
                              "红得对——它测出了那个截断）。实测往返偏差恰为0",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/scalar_diffraction_airy",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/scalar_diffraction_airy/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"first zero {first_zero!r}, quote truncation {truncation:.4e}, "
        f"|dE/dx|={slope:.6f}, predicted residual {predicted_residual:.4e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
