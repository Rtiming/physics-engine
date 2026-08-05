#!/usr/bin/env python3
"""同轴圆环互感的金标生成器——**独立算法，不调被验内核**。

被验内核（`physics_engine.electromagnetics`）走的是**Maxwell闭式 + AGM椭圆积分**。
本生成器走两条**完全不同**的路，一条都不与它共用代码：

1. **互感值**：Neumann双回路线积分

       M = (mu0/4pi) * closed_int closed_int (dl1 . dl2) / |r1 - r2|

   对共轴圆环按角差 phi = phi2 - phi1 化成单重积分

       M = (mu0 r1 r2 / 2) * int_0^{2pi} cos(phi) d(phi)
           / sqrt(r1^2 + r2^2 + d^2 - 2 r1 r2 cos(phi))

   被积函数解析且2pi周期，**周期解析函数上梯形法几何收敛**
   （`cases/scalar_diffraction_airy`用的是同一条形制）。
   生成器自己两档互证，互差超过`REFERENCE_FLOOR`即拒绝落盘。

   **但这个写法在远场会被相消吃掉**：d >> r 时 `int cos(phi) ~ 0`，
   被积函数量级1而积分量级1e-6，直接算会丢十几位有效数字（实测d=1m处
   相对误差1.1e-8，比要测的量还大）。所以改写成**全正被积函数**：
   令 c = r1^2+r2^2+d^2、q = 2 r1 r2 / c、s = sqrt(1 - q cos(phi))，由
   `1/s - 1 = q cos(phi) / (s (1+s))` 且 `int cos(phi) d(phi) = 0` 得

       M = (mu0 r1 r2 / 2) * (q / sqrt(c)) * int_0^{2pi} cos^2(phi) / (s (1+s)) d(phi)

   整条被积函数恒正，**一次减法都没有**。实测这一步把远场的相对误差
   从1.1e-8压到浮点地板。

2. **椭圆积分**：Carlson对称形式`R_F`/`R_D`的重复化算法，
   **原样复用`cases/large_deflection_cantilever/generate_oracle.py`那一份**
   （生成器之间复用是正当的——两侧生成器与被验内核仍然无共用代码）。
   注意那一份收的是**参数m = k^2**，而Maxwell闭式按传统写`K(k)`收**模k**。
   本文件在`_complete_k_of_parameter`处显式做这次换算，
   案例页第二节把这条约定写死。

参考解出处：Maxwell《A Treatise on Electricity and Magnetism》vol.2 sect.701；
Grover《Inductance Calculations: Working Formulas and Tables》ch.13。
Carlson对称形式：Carlson 1979（`R_F`/`R_D`重复化）。
mu0取CODATA 2022推荐值——**2019 SI重定义之后它不再是按定义精确的4pi*1e-7**。

本生成器不import `physics_engine.electromagnetics`；只用`oracles`写清单。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/mutual_inductance_coaxial"
ALGORITHM_VERSION = "1.0.0"

#: 周期梯形的节点数与自校验档。两档互差超过地板即拒绝落盘。
#: 8192对最坏构型（d=1mm、r=30mm，q=0.99944）实测两档互差2.2e-15；
#: 其余构型两档**逐位相同**。
REFERENCE_NODES = 8192
REFERENCE_SELF_CHECK_NODES = 2048
REFERENCE_FLOOR = 1.0e-14

#: 真空磁导率（CODATA 2022，H/m）。**不是按定义精确**——见模块docstring。
#: 这个字面量与`units.VACUUM_PERMEABILITY_H_PER_M`是**两份独立的抄写**，
#: 抄错一处判据当场红（这正是它不import内核常量的原因）。
MU0_H_PER_M = 1.25663706127e-6

#: 2019年之前按定义精确的旧值。用来把"新旧差多少"钉成一个数。
MU0_LEGACY_EXACT_H_PER_M = 4.0e-7 * math.pi

#: 闭式判据的构型：`(r1_m, r2_m, d_m)`。选点有理由，逐条见案例页第一节。
CONFIGURATIONS: tuple[tuple[float, float, float], ...] = (
    (0.050, 0.050, 0.020),   # 等径、近距，k=0.9806
    (0.100, 0.020, 0.050),   # 不等径5:1，k=0.6880
    (0.030, 0.030, 0.001),   # 近重合，k=0.99986——本组条件数最差
    (0.020, 0.040, 0.300),   # 中远场，k=0.1849
    (0.100, 0.010, 0.000),   # **共面同心，d恰为0**——合法构型不是退化
    (0.010, 0.010, 1.000),   # 远场，k=0.0200
    (0.005, 0.005, 5.000),   # 极远场，k=0.0020——教科书分组在此处已崩
)

#: 远场退化判据的构型：半径固定，d按倍频程排。
FAR_FIELD_RADII = (0.010, 0.020)
FAR_FIELD_FIRST_SEPARATION_M = 0.2
FAR_FIELD_OCTAVES = 6

#: 匝数判据的构型与匝数对。
TURNS_CONFIGURATION = (0.050, 0.050, 0.020)
TURNS_PAIRS: tuple[tuple[int, int], ...] = ((1, 1), (3, 5), (12, 40), (200, 7))

#: 单位边界判据：mm制声明的几何，与它逐位对应的米制声明。
UNIT_RADIUS_MM = 50.0
UNIT_SEPARATION_MM = 20.0
ROUND_TRIP_MILLIMETRES: tuple[float, ...] = (50.0, 20.0, 0.1, 1234.5678)

#: 磁链判据的电流。
FLUX_LINKAGE_CURRENTS_A: tuple[float, ...] = (0.0, 1.0, -3.5, 1.0e6)
FLUX_LINKAGE_TURNS = (2, 3)


# ---------------------------------------------------------------------------
# 路径一：Neumann积分的周期梯形求值（全正被积函数）
# ---------------------------------------------------------------------------


def neumann_mutual_inductance_h(r1: float, r2: float, d: float, nodes: int) -> float:
    """单匝共轴圆环互感的Neumann积分求值（H）。被积函数恒正，无相消。"""

    c = r1 * r1 + r2 * r2 + d * d
    q = 2.0 * r1 * r2 / c
    terms = []
    for index in range(nodes):
        angle = 2.0 * math.pi * (index + 0.5) / nodes
        cosine = math.cos(angle)
        root = math.sqrt(1.0 - q * cosine)
        terms.append(cosine * cosine / (root * (1.0 + root)))
    integral = math.fsum(terms) * (2.0 * math.pi / nodes) * q / math.sqrt(c)
    return MU0_H_PER_M * r1 * r2 / 2.0 * integral


def converged_mutual_inductance_h(r1: float, r2: float, d: float) -> float:
    """两档求值互相印证；不收敛即拒绝落盘（金标不许带未申报的不确定度）。"""

    coarse = neumann_mutual_inductance_h(r1, r2, d, REFERENCE_SELF_CHECK_NODES)
    fine = neumann_mutual_inductance_h(r1, r2, d, REFERENCE_NODES)
    if abs(coarse - fine) > REFERENCE_FLOOR * abs(fine):
        raise SystemExit(
            f"参考求值未收敛：M({r1}, {r2}, {d}) 两档相对差"
            f"{abs(coarse - fine) / abs(fine)!r} > {REFERENCE_FLOOR!r}，不落盘"
        )
    return fine


def dipole_mutual_inductance_h(r1: float, r2: float, d: float) -> float:
    """远场偶极子近似`M ~ mu0 pi r1^2 r2^2 / (2 d^3)`。

    独立推导：回路1在轴上距离d处的磁场`B = mu0 r1^2 I / (2 (r1^2+d^2)^{3/2})`，
    d >> r1时取`mu0 r1^2 I / (2 d^3)`，乘回路2的面积`pi r2^2`得磁通，除以I得M。
    """

    return MU0_H_PER_M * math.pi * r1 * r1 * r2 * r2 / (2.0 * d**3)


# ---------------------------------------------------------------------------
# 路径二：Carlson对称形式（复用大挠度悬臂生成器那一份，逐字未改）
# ---------------------------------------------------------------------------


def carlson_rf(x: float, y: float, z: float) -> float:
    """Carlson对称形式``R_F(x,y,z)``，重复化算法（Carlson 1979）。"""

    for _ in range(200):
        root_x, root_y, root_z = math.sqrt(x), math.sqrt(y), math.sqrt(z)
        lam = root_x * (root_y + root_z) + root_y * root_z
        x, y, z = 0.25 * (x + lam), 0.25 * (y + lam), 0.25 * (z + lam)
        average = (x + y + z) / 3.0
        dx, dy, dz = (average - x) / average, (average - y) / average, (average - z) / average
        if max(abs(dx), abs(dy), abs(dz)) < 0.0008:
            break
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    return (1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0) / math.sqrt(average)


def carlson_rd(x: float, y: float, z: float) -> float:
    """Carlson对称形式``R_D(x,y,z)``，重复化算法。"""

    total = 0.0
    factor = 1.0
    for _ in range(200):
        root_x, root_y, root_z = math.sqrt(x), math.sqrt(y), math.sqrt(z)
        lam = root_x * (root_y + root_z) + root_y * root_z
        total += factor / (root_z * (z + lam))
        factor *= 0.25
        x, y, z = 0.25 * (x + lam), 0.25 * (y + lam), 0.25 * (z + lam)
        average = 0.2 * (x + y + 3.0 * z)
        dx, dy, dz = (average - x) / average, (average - y) / average, (average - z) / average
        if max(abs(dx), abs(dy), abs(dz)) < 0.0004:
            break
    ea, eb = dx * dy, dz * dz
    ec, ed = ea - eb, ea - 6.0 * eb
    ee = ed + ec + ec
    c1, c2, c3, c4 = 3.0 / 14.0, 1.0 / 6.0, 9.0 / 22.0, 3.0 / 26.0
    tail = (1.0 + ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * dz * ee)
            + dz * (c2 * ee + dz * (-c3 * ec + dz * c4 * ea)))
    return 3.0 * total + factor * tail / (average * math.sqrt(average))


def complete_k_of_parameter(parameter: float) -> float:
    """``K``按**参数m = k^2**取值（悬臂生成器的约定）。"""

    return carlson_rf(0.0, 1.0 - parameter, 1.0)


def complete_e_of_parameter(parameter: float) -> float:
    """``E``按**参数m = k^2**取值。"""

    return complete_k_of_parameter(parameter) - (parameter / 3.0) * carlson_rd(
        0.0, 1.0 - parameter, 1.0
    )


def modulus_of(r1: float, r2: float, d: float) -> float:
    """`k = sqrt(4 r1 r2 / ((r1+r2)^2 + d^2))`——Maxwell闭式唯一的形状参数。

    **`+ d^2`是这条式子最容易漏的一项**：漏了它k恒等于`2 sqrt(r1 r2)/(r1+r2)`，
    与间距无关；等径时更是恒等于1，K发散。案例里有一条判据专钉它。
    """

    return math.sqrt(4.0 * r1 * r2 / ((r1 + r2) ** 2 + d * d))


def main() -> int:
    # --- 第一层：闭式互感值 + 模 ---
    moduli = [modulus_of(*config) for config in CONFIGURATIONS]
    inductances = [converged_mutual_inductance_h(*config) for config in CONFIGURATIONS]

    # --- 第二层（独立于闭式）：椭圆积分本身 ---
    elliptic_k = [complete_k_of_parameter(k * k) for k in moduli]
    elliptic_e = [complete_e_of_parameter(k * k) for k in moduli]

    # --- 第三层：远场退化到偶极子 ---
    separations = [
        FAR_FIELD_FIRST_SEPARATION_M * (2.0**octave) for octave in range(FAR_FIELD_OCTAVES)
    ]
    far_r1, far_r2 = FAR_FIELD_RADII
    deviations = [
        converged_mutual_inductance_h(far_r1, far_r2, d)
        / dipole_mutual_inductance_h(far_r1, far_r2, d)
        - 1.0
        for d in separations
    ]
    orders = [
        math.log2(deviations[index] / deviations[index + 1])
        for index in range(len(deviations) - 1)
    ]
    # 偏差的一阶预测：M/M_dipole - 1 ~ -3(r1^2+r2^2)/(2 d^2)，来自
    # (1 + (r1^2+r2^2)/d^2)^{-3/2}的展开。它给出**收敛阶恰为2**，
    # 也是下面容差理由里那个"预测值"。
    predicted = [
        -1.5 * (far_r1 * far_r1 + far_r2 * far_r2) / (d * d) for d in separations
    ]

    # --- 第四层：匝数 ---
    single_turn = converged_mutual_inductance_h(*TURNS_CONFIGURATION)
    turns_inductances = [n1 * n2 * single_turn for n1, n2 in TURNS_PAIRS]

    # --- 第五层：单位边界 ---
    unit_radius_m = UNIT_RADIUS_MM / 1.0e3
    unit_separation_m = UNIT_SEPARATION_MM / 1.0e3
    unit_inductance = converged_mutual_inductance_h(
        unit_radius_m, unit_radius_m, unit_separation_m
    )
    mu0_deviation = abs(MU0_H_PER_M - MU0_LEGACY_EXACT_H_PER_M) / MU0_LEGACY_EXACT_H_PER_M

    # --- 第六层：磁链 ---
    flux_turns_1, flux_turns_2 = FLUX_LINKAGE_TURNS
    flux_base = flux_turns_1 * flux_turns_2 * converged_mutual_inductance_h(
        0.05, 0.05, 0.02
    )
    flux_linkages = [flux_base * current for current in FLUX_LINKAGE_CURRENTS_A]

    oracles = [
        {
            "id": "oracle:mutual_inductance_coaxial/maxwell_closed_form",
            "inputs": {
                "kind": "neumann_double_loop_integral_periodic_trapezoid",
                "configurations_r1_r2_d_m": [list(config) for config in CONFIGURATIONS],
                "reference_nodes": REFERENCE_NODES,
                "reference_self_check_nodes": REFERENCE_SELF_CHECK_NODES,
                "reference_floor": REFERENCE_FLOOR,
                "vacuum_permeability_h_per_m": MU0_H_PER_M,
            },
            "expected": {"moduli": moduli, "mutual_inductances_h": inductances},
            "tolerances": {
                "moduli": {
                    "abs": 0.0, "rel": 4.0e-16,
                    "reason": "k = sqrt(4 r1 r2 / ((r1+r2)^2 + d^2))，两侧同一表达式，"
                              "差异只有一次sqrt的舍入（<=1eps=2.2e-16）；4e-16约2eps。"
                              "**这条钉的是k^2的表达式本身**，尤其是最容易漏的`+ d^2`："
                              "漏了它k与间距无关、等径时恒为1，本条当场红且报的就是k",
                },
                "mutual_inductances_h": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "闭式（AGM）对Neumann周期梯形积分，两条无共用代码的路。"
                              "实测最坏4.86e-14落在k=0.99986那一组（近重合）——那里`1-k^2`"
                              "相消到只剩2.8e-4，K对k'的对数依赖把k'的4e-13相对误差"
                              "折算成K的4.5e-14，与实测吻合。其余六组<=1.4e-15。"
                              "取1e-12是实测最坏的20.6倍。**它同时是教科书分组的捕手**："
                              "把方括号按(2/k-k)K-(2/k)E直接算，k=0.002那一组的相对误差"
                              "是1.1e-4（相消放大约1/k^4），红过头1e8倍",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_coaxial/elliptic_integrals",
            "inputs": {
                "kind": "carlson_symmetric_forms_rf_rd",
                "moduli": moduli,
                "convention": "本清单的moduli是**模k**；Carlson那一份收参数m = k^2，"
                              "生成器在complete_k_of_parameter处显式换算",
                "convention_probe_modulus": 0.5,
            },
            "expected": {
                "complete_elliptic_k": elliptic_k,
                "complete_elliptic_e": elliptic_e,
                "parameter_convention_agrees": True,
                "parameter_convention_is_not_the_same_as_modulus": True,
            },
            "tolerances": {
                "complete_elliptic_k": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": "AGM 对 Carlson重复化，两条不同算法。实测最坏3.24e-15"
                              "落在k=0.99986——差异主要来自Carlson那一侧："
                              "它收参数m并算`1-m`，而AGM算`(1-k)(1+k)`，后者在k->1处"
                              "少一次相消。其余六组<=4.4e-16（1-2eps）。取1e-14是3.1倍。"
                              "**这一条是K/E互换的唯一捕手**：本仓的M不经过K与E的差"
                              "（见elliptic.py第三节的无相消改写），所以把两个公开函数"
                              "对调，上一条oracle一个字都不会红",
                },
                "complete_elliptic_e": {
                    "abs": 0.0, "rel": 5.0e-15,
                    "reason": "同上。E没有对数奇性，k->1处条件数远好于K，"
                              "实测最坏6.66e-16（3eps）。取5e-15是7.5倍",
                },
                "parameter_convention_agrees": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：`K_of_parameter(0.25)`与`K(0.5)`**逐位相同**。"
                              "零容差因为前者的实现就是`K(sqrt(m))`，而sqrt(0.25)=0.5精确",
                },
                "parameter_convention_is_not_the_same_as_modulus": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：`K_of_parameter(0.5) != K(0.5)`。"
                              "**一条判据要守两件事**：换算对、以及两种约定确实不同。"
                              "只断前一条的话，一个把两个入口写成同一个函数的实现照样绿——"
                              "而那正是模块docstring警告的那个错（差10%，不会离谱到一眼看出）",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_coaxial/reciprocity",
            "inputs": {
                "kind": "self_consistency_independent_of_the_closed_form",
                "configurations_r1_r2_d_m": [list(config) for config in CONFIGURATIONS],
                "loop_turns": [3, 7],
            },
            "expected": {
                "scalar_reciprocity_max_abs_difference_h": 0.0,
                "loop_reciprocity_max_abs_difference_h": 0.0,
            },
            "tolerances": {
                "scalar_reciprocity_max_abs_difference_h": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "M(1->2)与M(2->1)必须**逐位相同**，不是"
                              "近似相同。理由是表达式在IEEE754下真的对称："
                              "`r1*r2 == r2*r1`、`r1+r2 == r2+r1`、`sqrt(r1*r2)`对称、"
                              "`|z2-z1| == |z1-z2|`。**因此零容差是算出来的不是许愿**。"
                              "它抓的是闭式抓不到的一类错：把`sqrt(r1*r2)`写成`r1`"
                              "（量纲、量级、远场退化阶全部照旧，只有对称性没了）",
                },
                "loop_reciprocity_max_abs_difference_h": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "回路级同上，且**匝数不同**（3与7）——`N1*N2`是整数乘法，"
                              "可交换且精确，所以零容差在带匝数时照样成立。"
                              "它另外抓一类错：匝数因子若写成`N1*single`再乘`N2`与"
                              "写成`N2*single`再乘`N1`，浮点下末位可能不同，本条会红",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_coaxial/far_field_dipole",
            "inputs": {
                "kind": "degeneration_to_the_magnetic_dipole_limit",
                "radii_m": list(FAR_FIELD_RADII),
                "separations_m": separations,
                "first_order_prediction": predicted,
            },
            "expected": {
                "dipole_ratio_deviations": deviations,
                "dipole_convergence_orders": orders,
            },
            "tolerances": {
                "dipole_ratio_deviations": {
                    "abs": 5.0e-14, "rel": 0.0,
                    "reason": "偏差`M/M_dipole - 1`。两侧的M_dipole是同一表达式同一mu0，"
                              "所以偏差之差只等于M的相对误差（因为M/M_dipole ~ 1），"
                              "实测<=8.2e-16。**判绝对不判相对**：偏差本身从1.8e-2"
                              "扫到1.8e-5跨三个数量级，相对容差会在小端把噪声放大成判据。"
                              "取5e-14是实测的61倍。**这一条只有无相消求值做得到**："
                              "教科书分组在d=6.4m处的相对误差就已经淹没掉1.8e-5的待测偏差",
                },
                "dipole_convergence_orders": {
                    "abs": 1.0e-9, "rel": 0.0,
                    "reason": "相邻两档偏差之比取log2。理论值恰为2："
                              "`M/M_dipole - 1 ~ -3(r1^2+r2^2)/(2 d^2)`，d翻倍偏差降4倍。"
                              "实测1.9806 -> 1.9999单调逼近2，**不写死为2**"
                              "（harmonic_oscillator那条纪律：收敛阶是测出来的不是断言的），"
                              "而是把这五个数逐个钉住。误差来源是M的1e-15折算到"
                              "最小偏差1.8e-5上再除ln2，约8e-11；取1e-9是12倍",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_coaxial/turns",
            "inputs": {
                "kind": "lumped_turns_are_a_linear_factor",
                "configuration_r1_r2_d_m": list(TURNS_CONFIGURATION),
                "turns_pairs": [list(pair) for pair in TURNS_PAIRS],
            },
            "expected": {
                "mutual_inductances_h": turns_inductances,
                "turns_factor_is_bit_exact": True,
            },
            "tolerances": {
                "mutual_inductances_h": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "`M_N = N1 N2 M_1`，金标是`N1 N2 x Neumann积分值`。"
                              "容差与第一条oracle同源（同一构型k=0.9806，实测1.37e-15），"
                              "外加一次整数乘法的舍入（<=1eps）。"
                              "它抓的是匝数写漏、写成`N1+N2`、只乘一侧、或按`(N1 N2)^2`——"
                              "四种错在N=(200,7)那一组分别差1400倍、6.8倍、7倍、1400倍",
                },
                "turns_factor_is_bit_exact": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：`M(N1,N2)`与`(N1*N2)*M(1,1)`**逐位相同**。"
                              "这条与上一条不重复：上一条验的是数对不对，"
                              "本条验的是**因子的结合次序**——`N1*(N2*M)`与`(N1*N2)*M`"
                              "在浮点下可以差末位，实现若改了次序本条会红而上一条不会。"
                              "0030记过同型的一次末位漂移（`k*(d_a.d_b)`写成`(k*d_a).d_b`）",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_coaxial/unit_boundary",
            "inputs": {
                "kind": "millimetre_to_metre_boundary_and_mu0_grade",
                "radius_mm": UNIT_RADIUS_MM,
                "separation_mm": UNIT_SEPARATION_MM,
                "round_trip_millimetres": list(ROUND_TRIP_MILLIMETRES),
                "flux_linkage_currents_a": list(FLUX_LINKAGE_CURRENTS_A),
                "flux_linkage_turns": list(FLUX_LINKAGE_TURNS),
                "legacy_exact_vacuum_permeability_h_per_m": MU0_LEGACY_EXACT_H_PER_M,
            },
            "expected": {
                "millimetre_round_trip": list(ROUND_TRIP_MILLIMETRES),
                "mutual_inductance_from_millimetres_h": unit_inductance,
                "millimetre_and_metre_declarations_agree": True,
                "mu0_relative_deviation_from_legacy": mu0_deviation,
                "flux_linkages_wb": flux_linkages,
            },
            "tolerances": {
                "millimetre_round_trip": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "mm -> m -> mm必须**逐位复原**。零容差是算出来的："
                              "两个方向共用同一个因子1000，一除一乘；1000是2的幂乘5^3，"
                              "对这四个测试值除1000再乘1000在double下精确可逆（实测四个全等）。"
                              "**方向写反不会报任何错**，只会把互感放大1e6倍——"
                              "本仓有过三次单位事故（1000倍、被抵消、静默1e12方向反），"
                              "往返判据是唯一抓得住方向的门",
                },
                "mutual_inductance_from_millimetres_h": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "从mm制声明的几何算出的互感，金标由米制的Neumann积分给出。"
                              "与上一条不重复：往返判据只验换算可逆，**验不出方向**"
                              "（乘1000再除1000同样可逆）。本条把换算的**方向**钉死在"
                              "一个物理值上，方向反了差1e6倍。容差与第一条oracle同源",
                },
                "millimetre_and_metre_declarations_agree": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：同一几何用mm声明与用m声明给出**逐位相同**的M。"
                              "零容差成立是因为50.0/1000恰好是0.05的double表示（实测相等）。"
                              "**若哪天不成立，那说明换算路径上多了一步运算，本条该红**",
                },
                "mu0_relative_deviation_from_legacy": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "本仓采信的mu0（CODATA 2022）与2019年前按定义精确的"
                              "4pi*1e-7的相对偏差，实测1.3203e-10——**落在CODATA的相对"
                              "标准不确定度1.6e-10之内**，即两个值物理上不冲突，"
                              "冲突的只是`benchmark_constant`那条'按定义精确'的声明"
                              "（research/08第4.4节）。两侧是同一对字面量的两次独立抄写，"
                              "差异只有一次减法与一次除法的舍入(<=2eps)；1e-12约4500eps。"
                              "**这条是mu0量级错的捕手**：写成1.2566e-5，本量从1.32e-10"
                              "跳到9.0，任何有理由的容差都拦得住",
                },
                "flux_linkages_wb": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "磁链`lambda = M x I_source`，M已含两侧匝数。"
                              "它抓的是**再乘一次匝数**这个最常见的错（本组差6倍），"
                              "以及对电流的线性（I=0给出恰为0、I=1e6不溢出）。"
                              "容差与第一条oracle同源加一次乘法舍入",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/mutual_inductance_coaxial",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/mutual_inductance_coaxial/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"moduli {min(moduli):.6f}..{max(moduli):.6f}; "
        f"M {min(inductances):.4e}..{max(inductances):.4e}; "
        f"far-field orders {[round(order, 6) for order in orders]}; "
        f"mu0 legacy deviation {mu0_deviation:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
