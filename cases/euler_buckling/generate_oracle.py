#!/usr/bin/env python3
"""Euler屈曲的金标生成器——**临界载荷、有效长度因子、离散链的精确特征值**。

轴向受压细长杆的失稳。控制方程``EI·y'''' + F·y'' = 0``，通解
``y = A + B·s + C·sin(k·s) + D·cos(k·s)``（``k² = F/EI``）。四个边界条件给出一个
2×2或4×4齐次系统，其行列式为零即**特征方程**；最小正根``u = k·L``给出临界载荷

    Fc = (u/L)²·EI = π²·EI/(b·L)²   其中 **b = π/u**

``b``就是有效长度因子（等效铰支柱长度``b·L``，即两个拐点之间的距离）。

**本生成器不抄b的表，它把b算出来。** 三种边界条件的特征方程各自求根：

| 边界条件 | 特征方程 | 最小正根``u`` | ``b = π/u`` |
|---|---|---|---|
| 两端铰支 | ``sin u = 0`` | ``π`` | 1 |
| 一端固支一端自由 | ``cos u = 0`` | ``π/2`` | 2 |
| 两端固支 | ``(1−cos u)·… → sin(u/2)·[(u/2)cos(u/2) − sin(u/2)] = 0`` | ``2π`` | 1/2 |

两端固支那一条有**两支**根：``sin(u/2) = 0``给``u = 2π``、``tan(v) = v``（``v = u/2``）
给``u = 8.9868…``。**取小的那个**，所以是``2π``——自校验把这条比较写进去了，
因为"b用错"最常见的形态就是拿了另一支根。文献值（Timoshenko & Gere,
*Theory of Elastic Stability*, 2nd ed., 1961, §2.1—2.4；AISC的理论K值表同）
是1.0/2.0/0.5，自校验拿它和求根结果对，**但写进金标的是求根结果**——
表里的数只当交叉验证，不当来源。

### 离散链的**精确**特征值（第二条参考路线）

引擎把杆离散成``n``段、在``n−1``个内部顶点上放弯曲能``(EI/(2h))·(Δφ)²``。
线性化后这正是铰接刚性链模型，其刚度矩阵是三对角Toeplitz阵``tridiag(−1, 2, −1)``，
特征值有闭式``2(1 − cos(kπ/n))``，于是（两端铰支）

    Fc_n = (4·EI/h²)·sin²(k·π/(2n))   k = 1, 2, …

**这条路线把离散误差与实现误差分开**：引擎的实测值应当命中``Fc_n``（而不是
命中连续解），二者之差是纯粹的离散误差``−π²/(12n²)``。它同时给出二阶模态比
``sin²(2π/(2n)) / sin²(π/(2n))``——连续极限下是4，有限``n``下略小。

**本生成器只算这两条闭式：不import physics_engine的任何力学模块**
（除写盘工具`oracles`外不碰引擎）。

四条自校验在生成时跑，任何一条不过就**不出金标**：

1. 求根得到的``b``与文献表逐项相对偏差 < 1e-12，且两端固支的另一支根确实更大；
2. **离散特征对是恒等式**：把``v_j = sin(jkπ/n)``代回差分方程，
   残差/量级 < 1e-14（这验的是闭式本身，不是它的某一个值）；
3. **离散→连续的第一修正恰为``−π²/(12n²)``**：断言"n翻倍则偏差降4倍"，
   比只断言"趋近π²EI/L²"强得多，且校验里不需要写下任何系数；
4. 单调性：``Fc_n``随n单调增且恒小于连续值；二阶模态比随n单调增且恒小于4。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/euler_buckling"
ALGORITHM_VERSION = "1.0.0"

LENGTH_MM = 100.0
BENDING_STIFFNESS_NMM2 = 100.0
#: 轴向刚度。Euler理论假定杆**不可伸长**；引擎侧只有弹簧，所以细长比要选在
#: 伸长效应可忽略之处。预压缩使每段缩短``F/EA``，于是离散临界载荷被抬高
#: **恰好``+Fc/EA``**（实测见案例页第三节）。取1e5时该项是9.87e-7，
#: 比最细档的离散误差1.28e-4小两个数量级。
#: **不能一味加大**：轴向残差的可达地板随EA成正比上升（EA=1e6时地板约1.4e-9 N，
#: 已经顶到本案例的残差容差），那是``l − l0``的相消——应变只有1e-6量级。
AXIAL_STIFFNESS_N = 1.0e5
REFINEMENTS = (10, 20, 40, 80)
#: 牛顿绝对残差容差（spec/12第4.3节要求说清绝对还是相对）。
#: **预屈曲解是线性问题**（直链上弯曲对轴向的Hessian恰为零），牛顿一步收敛。
#: 实测残差地板：EA=1e4时1.4e-11 N、EA=1e5时约1.4e-10 N——取1e-8留70倍余量。
#: 它对结果的影响可算：1e-8 N对应位移1e-8/(EA/h) = 5e-13 mm，相对5e-15。
RESIDUAL_TOL_N = 1.0e-8
#: 载荷二分：在``[low·Fc, high·Fc]``上对"切线刚度是否正定"二分。
#: 40次把区间压到``5.9·Fc/2⁴⁰ = 5.4e-12·Fc``，远低于Cholesky判正定的数值地板
#: （EA=1e5时实测约2e-10相对）。
BISECTION_ITERATIONS = 40
BRACKET_LOW_FACTOR = 0.1
BRACKET_HIGH_FACTOR = 6.0
#: 不可伸长扰动的链节转角幅值（rad）。定性门与模态门都用它。
#: 上界：Rayleigh商的四阶修正是``O(A²)``，实测A=0.01给6e-6、A=0.003给6e-7相对。
#: 下界：能量差``ΔU``在A=0.003时约6.6e-6 N·mm，而总能量约9.9 N·mm、
#: 双精度分辨率约1e-15 N·mm——余量九个数量级。
PERTURBATION_AMPLITUDE_RAD = 0.003
SUBCRITICAL_LOAD_FACTOR = 0.7
SUPERCRITICAL_LOAD_FACTOR = 1.3
#: 固支顶点的Voronoi长度系数（decisions/0027、0029推出的``3h/2``）。
#: **本案例是它的第三个独立实例，而且是在一个完全不同的物理问题上**：
#: 前两次验的是静挠度，这次验的是特征值。实测扫描见案例页第三节。
CLAMP_VORONOI_FACTOR = 1.5
#: 模态层与定性层用的加密档（不必最细——它们验的是形状与符号，不是收敛阶）。
MODE_REFINEMENT = 20
#: 预屈曲牛顿迭代上界。线性问题，实测恒为1。
NEWTON_ITERATIONS_BOUND = 5

#: 文献值，**只作交叉验证不作来源**：Timoshenko & Gere, *Theory of Elastic
#: Stability*, 2nd ed., 1961, §2.1（两端铰支）、§2.2（一端固支一端自由）、
#: §2.4（两端固支）；AISC钢结构规范附录的"理论K值"同为1.0/2.0/0.5。
LITERATURE_EFFECTIVE_LENGTH_FACTORS = {
    "pinned_pinned": 1.0,
    "fixed_free": 2.0,
    "fixed_fixed": 0.5,
}


def bisect_root(function, low: float, high: float, iterations: int = 200) -> float:
    """区间内的变号根，纯二分（无导数、无外推——这里只要稳）。"""

    f_low = function(low)
    if f_low == 0.0:
        return low
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if (function(middle) > 0.0) == (f_low > 0.0):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def characteristic_roots() -> dict[str, float]:
    """三种边界条件的最小正根``u = k·L``，各自由特征方程求出。"""

    pinned = bisect_root(math.sin, 0.5 * math.pi, 1.5 * math.pi)
    free = bisect_root(math.cos, 0.1, math.pi)
    # 两端固支：``u·sin u = 2(1 − cos u)``，因式分解成
    # ``4·sin(u/2)·[(u/2)·cos(u/2) − sin(u/2)] = 0``。第一支给u=2π。
    clamped = 2.0 * bisect_root(math.sin, 0.5 * math.pi, 1.5 * math.pi)
    return {"pinned_pinned": pinned, "fixed_free": free, "fixed_fixed": clamped}


def clamped_second_branch() -> float:
    """两端固支特征方程的**另一支**：``tan v = v``，``u = 2v``。

    它存在的意义是证明``2π``真的是最小的那个——"b用错"最常见的形态
    正是拿了另一支根。
    """

    root = bisect_root(lambda v: math.tan(v) - v, math.pi, 1.5 * math.pi - 1.0e-12)
    return 2.0 * root


def critical_load(factor_b: float) -> float:
    """``Fc = π²·EI/(b·L)²``。"""

    return math.pi**2 * BENDING_STIFFNESS_NMM2 / (factor_b * LENGTH_MM) ** 2


def discrete_critical_load(segments: int, mode: int = 1) -> float:
    """铰接刚性链（两端铰支）的**精确**第``mode``阶临界载荷。

    ``Fc = (4·EI/h²)·sin²(mode·π/(2n))``，来自``tridiag(−1, 2, −1)``的闭式特征值。
    """

    step = LENGTH_MM / segments
    return (
        4.0 * BENDING_STIFFNESS_NMM2 / step**2
        * math.sin(mode * math.pi / (2.0 * segments)) ** 2
    )


def self_check() -> None:
    """四条自校验；任何一条不过就不出金标。"""

    roots = characteristic_roots()
    for name, root in roots.items():
        derived = math.pi / root
        known = LITERATURE_EFFECTIVE_LENGTH_FACTORS[name]
        if abs(derived - known) / known > 1.0e-12:
            raise SystemExit(
                f"{name}: 特征方程求根给b={derived!r}，文献表说{known!r}——"
                "两者不符时**不许改表迁就求根，也不许改求根迁就表**，先查特征方程"
            )
    other = clamped_second_branch()
    if not roots["fixed_fixed"] < other:
        raise SystemExit(
            f"两端固支取到了大的那一支根：{roots['fixed_fixed']!r} 对 {other!r}"
        )
    if not (8.9 < other < 9.1):
        raise SystemExit(f"tan v = v 的第一根不对：u={other!r}（应约8.9868）")

    # 2. 离散特征对是恒等式：v_j = sin(jkπ/n) 满足 −v_{j−1} + 2v_j − v_{j+1} = λ_k·v_j
    for segments in (10, 17):
        for mode in range(1, segments):
            eigenvalue = 2.0 * (1.0 - math.cos(mode * math.pi / segments))
            def component(index: int, mode: int = mode, segments: int = segments) -> float:
                # **先把整数相位折回一个周期再乘π**：``index·mode``可以到255，
                # 直接``sin(255π/17)``会把π的舍入放大255倍（实测残差1.27e-14，
                # 恰是那个放大而不是恒等式的误差）。``fmod``对整数是精确的。
                return math.sin(math.fmod(index * mode, 2 * segments) * math.pi / segments)
            worst = 0.0
            for index in range(1, segments):
                residual = (
                    -component(index - 1) + 2.0 * component(index) - component(index + 1)
                    - eigenvalue * component(index)
                )
                worst = max(worst, abs(residual))
            if worst > 1.0e-14:
                raise SystemExit(
                    f"离散特征对不是恒等式：n={segments} k={mode} 残差{worst!r}"
                )

    # 3. 离散→连续的第一修正恰为 −π²/(12n²)：n翻倍则偏差降4倍。
    continuum = critical_load(LITERATURE_EFFECTIVE_LENGTH_FACTORS["pinned_pinned"])
    deviations = []
    for segments in (20, 40, 80, 160):
        relative = discrete_critical_load(segments) / continuum - 1.0
        leading = -math.pi**2 / (12.0 * segments**2)
        deviations.append(relative / leading - 1.0)
    # 余项符号必须一致（同一个渐近展开的同一项），且非零（全零说明算错了首项）。
    if not (all(deviation < 0.0 for deviation in deviations)
            or all(deviation > 0.0 for deviation in deviations)):
        raise SystemExit(f"离散偏差的余项变号了，不是一个干净的展开：{deviations!r}")
    if any(deviation == 0.0 for deviation in deviations):
        raise SystemExit(f"离散偏差的余项恰为零——首项被算成了全部：{deviations!r}")
    for index in range(len(deviations) - 1):
        ratio = deviations[index] / deviations[index + 1]
        if not (3.8 <= ratio <= 4.2):
            raise SystemExit(
                f"离散偏差的**次级**修正不是二阶：n翻倍而余项只降{ratio!r}倍（应约4）"
            )

    # 4. 单调性。
    previous = discrete_critical_load(4)
    for segments in (8, 16, 32, 64, 128):
        current = discrete_critical_load(segments)
        if not (previous < current < continuum):
            raise SystemExit(f"离散临界载荷在n={segments}处不单调或越过连续值")
        previous = current
    previous_ratio = 0.0
    for segments in (4, 8, 16, 32, 64):
        ratio = discrete_critical_load(segments, 2) / discrete_critical_load(segments, 1)
        if not (previous_ratio < ratio < 4.0):
            raise SystemExit(f"二阶模态比在n={segments}处不单调或越过4：{ratio!r}")
        previous_ratio = ratio


def main() -> int:
    self_check()
    roots = characteristic_roots()
    factors = {name: math.pi / root for name, root in roots.items()}
    loads = {name: critical_load(factor) for name, factor in factors.items()}
    finest = REFINEMENTS[-1]
    oracles = [{
        "id": "oracle:euler_buckling/critical_load_and_mode",
        "inputs": {
            "kind": "axially_loaded_column_stability",
            "length_mm": LENGTH_MM,
            "bending_stiffness_nmm2": BENDING_STIFFNESS_NMM2,
            "axial_stiffness_n": AXIAL_STIFFNESS_N,
            "refinements": list(REFINEMENTS),
            "value_refinement": finest,
            "mode_refinement": MODE_REFINEMENT,
            "residual_tol_n": RESIDUAL_TOL_N,
            "bisection_iterations": BISECTION_ITERATIONS,
            "bracket_low_factor": BRACKET_LOW_FACTOR,
            "bracket_high_factor": BRACKET_HIGH_FACTOR,
            "perturbation_amplitude_rad": PERTURBATION_AMPLITUDE_RAD,
            "subcritical_load_factor": SUBCRITICAL_LOAD_FACTOR,
            "supercritical_load_factor": SUPERCRITICAL_LOAD_FACTOR,
            "clamp_voronoi_factor": CLAMP_VORONOI_FACTOR,
            "newton_iterations_bound": NEWTON_ITERATIONS_BOUND,
        },
        "expected": {
            "critical_load_pinned_pinned_n": loads["pinned_pinned"],
            "critical_load_fixed_free_n": loads["fixed_free"],
            "critical_load_fixed_fixed_n": loads["fixed_fixed"],
            "effective_length_factor_pinned_pinned": factors["pinned_pinned"],
            "effective_length_factor_fixed_free": factors["fixed_free"],
            "effective_length_factor_fixed_fixed": factors["fixed_fixed"],
            "discrete_critical_loads_pinned_pinned": [
                discrete_critical_load(segments) for segments in REFINEMENTS
            ],
            "prebuckling_relative_shortening": loads["pinned_pinned"] / AXIAL_STIFFNESS_N,
            "convergence_ratio_low": 3.9,
            "convergence_ratio_high": 4.1,
            "clamp_correction_error_advantage_min": 50.0,
            "clamp_uncorrected_ratio_low": 1.9,
            "clamp_uncorrected_ratio_high": 2.1,
            "first_mode_rayleigh_gap_max": 5.0e-6,
            "second_mode_ratio": (
                discrete_critical_load(MODE_REFINEMENT, 2)
                / discrete_critical_load(MODE_REFINEMENT, 1)
            ),
            "trial_shape_excess_min": 0.15,
            "subcritical_energy_rises": True,
            "supercritical_energy_falls": True,
            "converged_and_stable_below_critical": True,
            "converged_but_unstable_above_critical": True,
        },
        "tolerances": {
            "critical_load_pinned_pinned_n": {
                "abs": 0.0, "rel": 2.0e-4,
                "reason": "容差就是**离散误差**：n=80实测−1.275174e-04（比n=40的"
                          "−5.129502e-04小4.02倍，干净二阶）。取2e-4留1.57倍余量。"
                          "**不是数值精度**——二分把区间压到5.4e-12·Fc，Cholesky判"
                          "正定的数值地板约2e-10相对，两者都比离散误差小六个数量级",
            },
            "critical_load_fixed_free_n": {
                "abs": 0.0, "rel": 6.0e-5,
                "reason": "同上。n=80实测−2.947022e-05。这一档的离散误差比两端铰支"
                          "小4.3倍，**因为固支顶点的3h/2修正把首阶项消掉了**"
                          "（不修正时是+1.2588e-02，大427倍）。取6e-5留2.04倍余量",
            },
            "critical_load_fixed_fixed_n": {
                "abs": 0.0, "rel": 8.0e-4,
                "reason": "同上。n=80实测−4.331911e-04。三条里最大，因为两端各有一个"
                          "固支顶点、两处修正的余项叠加，且Fc本身是两端铰支的4倍"
                          "（弯曲更剧烈、离散更吃力）。取8e-4留1.85倍余量",
            },
            "effective_length_factor_pinned_pinned": {
                "abs": 0.0, "rel": 1.0e-4,
                "reason": "b由实测Fc反解：``b = π·sqrt(EI/Fc)/L``，故``b``的相对误差"
                          "**恰为Fc的一半**（−1/2次幂）。1.275e-4的一半是6.4e-5，"
                          "取1e-4留1.57倍余量——与Fc那条同源、同余量，不是另立一个数",
            },
            "effective_length_factor_fixed_free": {
                "abs": 0.0, "rel": 3.0e-5,
                "reason": "同上：6e-5的一半",
            },
            "effective_length_factor_fixed_fixed": {
                "abs": 0.0, "rel": 4.0e-4,
                "reason": "同上：8e-4的一半",
            },
            "discrete_critical_loads_pinned_pinned": {
                "abs": 0.0, "rel": 2.0e-6,
                "reason": "**这条是紧的那一条**：金标是离散链的精确特征值，所以离散误差"
                          "在这里不出现。剩下的只有轴向可伸长性——预压缩使每段缩短"
                          "``Fc/EA``，把离散临界载荷抬高恰好``+Fc/EA = 9.87e-7``。"
                          "实测四档的相对偏差9.7885e-07/9.8487e-07/9.8654e-07/"
                          "9.8665e-07，**扣掉``Fc/EA``后残差 ≤ 1.9e-10**。"
                          "取2e-6是容纳那个物理偏置并留2倍余量；"
                          "**它不是放宽，它是一条比连续解紧100倍的门**",
            },
            "prebuckling_relative_shortening": {
                "abs": 0.0, "rel": 1.0e-9,
                "reason": "``(EA/h)(l−h) + F = 0 → l = h(1 − F/EA)``，逐段精确，"
                          "所以总缩短率恰为``F/EA``。误差只来自牛顿停在残差1e-8 N处："
                          "对应位移1e-8/(EA/h)=5e-13 mm，相对5e-15。取1e-9留六个数量级。"
                          "**这条门守的是`PointLoad`梯度的方向与大小**——符号写反"
                          "会变成伸长，量级写错会差在第一位",
            },
            "convergence_ratio_low": {"abs": 0.0, "rel": 0.0,
                                      "reason": "区间下界，零容差比较（非数值判据）"},
            "convergence_ratio_high": {
                "abs": 0.0, "rel": 0.0,
                "reason": "区间上界。**不写死为4**（spec/12第4.3节：比阶不比单点）。"
                          "两端铰支实测三档3.9916/4.0033/4.0226。上界4.1而不是更松，"
                          "是因为轴向可伸长性给一个**不随n下降**的常数偏置"
                          "(+9.87e-7)，它在最细档占离散误差的0.8%，会把比值往上推——"
                          "EA再大一个数量级就会顶穿这条上界（那时轴向残差地板也顶穿"
                          "残差容差）。这两件事是同一个权衡的两端",
            },
            "clamp_correction_error_advantage_min": {
                "abs": 0.0, "rel": 0.0,
                "reason": "固支顶点Voronoi取3h/2对取h的误差优势倍数下界。"
                          "实测一端固支一端自由三档：126×/143×/232×。取50是留2.5倍"
                          "余量的**下界**，不是实测值——写实测值等于让门迁就当前实现",
            },
            "clamp_uncorrected_ratio_low": {"abs": 0.0, "rel": 0.0,
                                            "reason": "退回h时的收敛比区间下界，零容差"},
            "clamp_uncorrected_ratio_high": {
                "abs": 0.0, "rel": 0.0,
                "reason": "退回h时**必须是干净一阶**：实测2.0539/2.0271/2.0140。"
                          "[1.9, 2.1]把它与3h/2那一档（2.33/3.30/3.70，正在趋近4）"
                          "分开。这是正向的必须红——后人改那个系数时看得见为什么",
            },
            "first_mode_rayleigh_gap_max": {
                "abs": 0.0, "rel": 0.0,
                "reason": "正弦半波试验形的Rayleigh临界载荷对实测离散Fc的相对超出量"
                          "上界。**变分原理保证它非负**（任何试验形都给上界），"
                          "而它有多小就是「一阶模态是不是正弦半波」的度量。"
                          "实测A=0.003时5.9e-7（A=0.01时7.0e-6，差约11倍=幅值平方比，"
                          "证实余项是O(A²)）。取5e-6留8.5倍余量",
            },
            "second_mode_ratio": {
                "abs": 0.0, "rel": 5.0e-5,
                "reason": "全波试验形给出的临界载荷与一阶之比。连续极限是4=2²；"
                          "**金标取的是离散链的精确比**"
                          "``sin²(2π/2n)/sin²(π/2n)``（n=20时3.9753766812），"
                          "因为被测的是离散模型不是连续模型。实测3.9753910，"
                          "偏差3.6e-6，其中O(A²)的Rayleigh余项占主要部分。"
                          "取5e-5留14倍余量",
            },
            "trial_shape_excess_min": {
                "abs": 0.0, "rel": 0.0,
                "reason": "**非正弦半波**的试验形必须给出至少高15%的临界载荷。"
                          "实测抛物线+16.03%、尖顶折线+712%、全波+298%。"
                          "取0.15是抛物线那一档留0.7%余量的下界——它是最接近的那个"
                          "竞争形状，门的分辨力由它决定，所以不能取更大",
            },
            "subcritical_energy_rises": {
                "abs": 0.0, "rel": 0.0,
                "reason": "布尔：0.7Fc处**不可伸长**横向扰动的能量差必须为正。"
                          "零容差因为它是符号判据。实测+6.616380e-06 N·mm"
                          "（两端铰支）与+1.671423e-06（一端固支一端自由）",
            },
            "supercritical_energy_falls": {
                "abs": 0.0, "rel": 0.0,
                "reason": "布尔：1.3Fc处同一扰动的能量差必须为负。"
                          "实测−6.707552e-06与−1.493016e-06。**这一对是本案例唯一"
                          "验``energy()``表达式的门**：扰动保长，轴向能量不变，"
                          "失稳全部来自`PointLoad`做的功——``E = −F·x``的符号写反，"
                          "这一条立刻红，而临界载荷那几条照绿",
            },
            "converged_and_stable_below_critical": {
                "abs": 0.0, "rel": 0.0,
                "reason": "布尔：0.7Fc处直链既是收敛解、切线刚度又正定",
            },
            "converged_but_unstable_above_critical": {
                "abs": 0.0, "rel": 0.0,
                "reason": "布尔：1.3Fc处直链**仍然是收敛解**（牛顿一步、残差达标、"
                          "解逐位仍是直的），而切线刚度**不正定**。"
                          "``converged=True``不是稳定性——这条门把那句话钉死",
            },
        },
    }]
    document = {
        "facet": "engine_oracle_manifest", "facet_version": "0.1",
        "case_id": "case/euler_buckling", "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/euler_buckling/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles, "arrays": {}, "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
