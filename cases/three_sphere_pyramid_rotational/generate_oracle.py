#!/usr/bin/env python3
"""三球金字塔**含力矩平衡**那一支的金标——**`Q(√3)`上的精确有理算术**。

本脚本**不是**`cases/three_sphere_pyramid/generate_oracle.py`的改写：那一条写的是
球-球无摩擦、且**不写**力矩平衡的6方程6未知变体，闭式``μc = 1/(3√3)``。
本条写的是research/15第三节独立复推的**全粘着支**：三个刚体各`ΣFx`、`ΣFz`、
**`ΣM`**共9条方程、8个未知量，精确秩8，``μc = 2 − √3``。

**两支差1.392305倍**（0063第二节），而那正是"转动自由度缺席时接触问题的答案
系统性地偏约四成"的第一条证据。

## 为什么这里重做一遍精确算术，而不是把research/15的四个数抄下来

抄下来的数是**别人的结论**，不是金标。research/15第三节自己就写着方法纪律：
"别人的数不是真值，是另一个证人"。本脚本在`Q(√3)`（形如``a + b√3``，
``a``、``b``是`Fraction`）上**全程无浮点**地装配并消元，最后才把结果化成double。
于是金标是**算出来的**，而"与research/15逐位相同"变成一条可判的门，
不是一条约定。

## 未知量与方程（research/15第3.1节）

等径``r``、等质量（自重``W``）三球，平面``x``-``z``问题：

    顶球  T  = (0, r(1+√3))
    底球  BR = (r, r)          BL = (−r, r)
    单位法向（底球指向顶球）  n_R = (−1/2, √3/2)   n_L = (1/2, √3/2)

未知8个（**不预设对称**）：两个球-球接触各``(F_n, f_t)``、两个球-地接触各``(N, T)``。
方程9条：三个刚体各``ΣFx = 0``、``ΣFz = 0``、``ΣM = 0``。

**三个前提**（research/15第3.3节，缺一条闭式就不成立）：
点接触不传力偶；平面问题；**两底球之间那个真实存在的接触不传力**
（顶球把它们往外推，单边接触分离——这一条是算出来的，不是略去的）。

零运行时依赖，纯标准库。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/three_sphere_pyramid_rotational"
#: 1.1.0：加"两侧"那一组（决策0080）。**旧的五组一个数都没动**——
#: 新增oracle不改既有oracle，那是三前提第三条在金标上的形式。
ALGORITHM_VERSION = "1.1.0"

RADIUS_MM = 10.0
MASS_KG = 1.5
GRAVITY_MM_S2 = 9810.0
WEIGHT_N = MASS_KG * GRAVITY_MM_S2 / 1000.0


@dataclass(frozen=True)
class Q3:
    """``a + b√3``，``a``/``b``为`Fraction`。**全程没有浮点。**

    这个域对加减乘封闭，且``(a+b√3)(a−b√3) = a² − 3b²``是有理数，故也对除法封闭
    （只要不除以零）。高斯消元要的就是这四条。
    """

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)

    def __add__(self, other: Q3) -> Q3:
        return Q3(self.a + other.a, self.b + other.b)

    def __sub__(self, other: Q3) -> Q3:
        return Q3(self.a - other.a, self.b - other.b)

    def __mul__(self, other: Q3) -> Q3:
        return Q3(self.a * other.a + 3 * self.b * other.b, self.a * other.b + self.b * other.a)

    def __truediv__(self, other: Q3) -> Q3:
        denominator = other.a * other.a - 3 * other.b * other.b
        if denominator == 0:
            raise ZeroDivisionError("division by zero in Q(sqrt 3)")
        conjugate = Q3(other.a, -other.b)
        product = self * conjugate
        return Q3(product.a / denominator, product.b / denominator)

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    def to_float(self) -> float:
        return float(self.a) + float(self.b) * math.sqrt(3.0)

    def text(self) -> str:
        if self.b == 0:
            return str(self.a)
        #: **不用`f"{self.b:+}"`**：`Fraction.__format__`要到**Python 3.13**才接受
        #: "只有符号、没有呈现类型"这种格式串，而`pyproject.toml`声明的地板是`>=3.11`。
        #: 2026-08-18实测：本机3.13跑得动，master的3.12.2当场`ValueError`——
        #: 于是**这个生成器在本仓自己声明的地板上跑不动，而在此之前没有任何东西会红**
        #: （案例判据走的是内核那条路、不走生成器那条路）。
        #: 手写符号在3.11—3.13上给出同一串字符。
        sign = "+" if self.b >= 0 else "-"
        return f"({self.a}{sign}{abs(self.b)}√3)"


def rational(value) -> Q3:
    return Q3(Fraction(value), Fraction(0))


def _rate_key(mu: float) -> str:
    """速率律那一组的量名：``mu``进名字，因为**每一档各有各的容差**。

    小数点写成``p``——量名要能当标识符读，而`oracle.json`是给人看的。
    """

    return f"slip_per_contact_mm_at_mu_{mu:.2f}".replace(".", "p")


ZERO = rational(0)
ONE = rational(1)
SQRT3 = Q3(Fraction(0), Fraction(1))
HALF = Q3(Fraction(1, 2), Fraction(0))


def exact_solution() -> dict[str, Q3]:
    """装配9×8并做精确高斯消元。返回8个未知量（以``W``为单位、``r = 1``）。

    未知量次序（**次序是形制**，改它就改了解向量的读法）：

        0: F1  球-球法向（顶球-右底球）      1: f1  球-球切向
        2: F2  球-球法向（顶球-左底球）      3: f2  球-切向
        4: N1  右底球-地法向                 5: T1  右底球-地切向
        6: N2  左底球-地法向                 7: T2  左底球-地切向
    """

    #: 单位法向（底球指向顶球）与切向。**切向按镜像取**：``t_L``是``t_R``关于
    #: ``x = 0``的镜像，地面切向同理（右球取``+x̂``、左球取``−x̂``）。
    #: 这不是口味：对称的构型配对称的正方向，"对称性是解出来的"才是一句可判的话——
    #: 配反了``f1 = −f2``照样是正确解，但那四条差为零的断言就验不到任何东西。
    n_r = (ZERO - HALF, SQRT3 * HALF)
    t_r = (SQRT3 * HALF, HALF)
    n_l = (HALF, SQRT3 * HALF)
    t_l = (ZERO - SQRT3 * HALF, HALF)
    ground_sign_r = ONE
    ground_sign_l = ZERO - ONE

    #: 力矩系数``c = n_x·t_z − n_z·t_x``（臂长``r = 1``）。
    #: **对顶球与对底球是同一个数**：两边的``ℓ``与力同时反号，叉积不变。
    #: 精确算术下``c_R = −1``、``c_L = +1``——两个整数，一眼可查。
    c_r = n_r[0] * t_r[1] - n_r[1] * t_r[0]
    c_l = n_l[0] * t_l[1] - n_l[1] * t_l[0]
    if not (c_r + ONE).is_zero() or not (c_l - ONE).is_zero():
        raise AssertionError(f"力矩系数不是∓1：c_R={c_r.text()} c_L={c_l.text()}")

    rows: list[list[Q3]] = []
    rhs: list[Q3] = []

    def add_row(coefficients: dict[int, Q3], right: Q3) -> None:
        row = [ZERO] * 8
        for index, value in coefficients.items():
            row[index] = row[index] + value
        rows.append(row)
        rhs.append(right)

    #: —— 顶球：底球给它``+F·n + f·t``（法向指向顶球），自重``−W ẑ`` ——
    add_row({0: n_r[0], 1: t_r[0], 2: n_l[0], 3: t_l[0]}, ZERO)
    add_row({0: n_r[1], 1: t_r[1], 2: n_l[1], 3: t_l[1]}, ONE)
    #: ``ΣM``（对顶球心）：法向过球心不产生力矩，只剩切向。
    add_row({1: c_r, 3: c_l}, ZERO)

    #: —— 右底球：顶球给它``−(F·n + f·t)``；地面给``N·ẑ + T·(+x̂)``；自重 ——
    add_row({0: ZERO - n_r[0], 1: ZERO - t_r[0], 5: ground_sign_r}, ZERO)
    add_row({0: ZERO - n_r[1], 1: ZERO - t_r[1], 4: ONE}, ONE)
    #: ``ΣM``（对右底球心）：球-球接触给``f·c_R``，地面接触点在``−ẑ``给``−T·sign``。
    add_row({1: c_r, 5: ZERO - ground_sign_r}, ZERO)

    #: —— 左底球：地面切向取``−x̂`` ——
    add_row({2: ZERO - n_l[0], 3: ZERO - t_l[0], 7: ground_sign_l}, ZERO)
    add_row({2: ZERO - n_l[1], 3: ZERO - t_l[1], 6: ONE}, ONE)
    add_row({3: c_l, 7: ZERO - ground_sign_l}, ZERO)

    #: —— 精确高斯消元（部分主元按"非零"选，不按大小：精确算术里没有病态） ——
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(rows)]
    pivot_row = 0
    pivots: list[int] = []
    for column in range(8):
        target = None
        for candidate in range(pivot_row, len(augmented)):
            if not augmented[candidate][column].is_zero():
                target = candidate
                break
        if target is None:
            continue
        augmented[pivot_row], augmented[target] = augmented[target], augmented[pivot_row]
        pivot = augmented[pivot_row][column]
        augmented[pivot_row] = [value / pivot for value in augmented[pivot_row]]
        for other in range(len(augmented)):
            if other == pivot_row:
                continue
            factor = augmented[other][column]
            if factor.is_zero():
                continue
            augmented[other] = [
                augmented[other][index] - factor * augmented[pivot_row][index]
                for index in range(9)
            ]
        pivots.append(column)
        pivot_row += 1

    rank = len(pivots)
    if rank != 8:
        raise AssertionError(f"精确秩不是8而是{rank}——问题不再静定，闭式μc就没了")
    #: 剩下的行必须整行为零（含右端），否则9条方程不相容。
    for row in augmented[rank:]:
        if not all(value.is_zero() for value in row):
            raise AssertionError("9条方程不相容——第9条不是冗余的")

    names = ("F1", "f1", "F2", "f2", "N1", "T1", "N2", "T2")
    return {names[column]: augmented[index][8] for index, column in enumerate(pivots)}


def main() -> int:
    solution = exact_solution()

    #: —— 精确断言（**它们在生成期就跑**，抄错一个符号这里当场炸） ——
    for left, right in (("F1", "F2"), ("f1", "f2"), ("N1", "N2"), ("T1", "T2")):
        if not (solution[left] - solution[right]).is_zero():
            raise AssertionError(f"对称性不是解出来的：{left} != {right}")
    if not (solution["F1"] - HALF).is_zero():
        raise AssertionError("球-球法向不是W/2")
    if not (solution["N1"] - Q3(Fraction(3, 2), Fraction(0))).is_zero():
        raise AssertionError("球-地法向不是3W/2")

    sphere_ratio = solution["f1"] / solution["F1"]
    ground_ratio = (ZERO - solution["T1"]) / solution["N1"]
    exact_mu = rational(2) - SQRT3
    if not (sphere_ratio - exact_mu).is_zero():
        raise AssertionError(f"球-球摩擦需求不是2−√3而是{sphere_ratio.text()}")
    if not (ground_ratio - exact_mu / rational(3)).is_zero():
        raise AssertionError("球-地摩擦需求不是(2−√3)/3")

    #: **顶球把两个底球往外推**（research/15第3.3节前提三）：这个数必须为正，
    #: 否则两底球之间那个真实存在的接触会传力、问题变超静定、闭式作废。
    outward = (ZERO - solution["F1"]) * n_r_x() + (ZERO - solution["f1"]) * t_r_x()
    if not (outward.a > 0 or (outward.a == 0 and outward.b > 0)):
        raise AssertionError("顶球没有把底球往外推——两底球接触会传力，闭式作废")

    critical_friction = exact_mu.to_float()
    sphere_contact_force_n = solution["F1"].to_float() * WEIGHT_N
    sphere_tangential_n = solution["f1"].to_float() * WEIGHT_N
    ground_normal_n = solution["N1"].to_float() * WEIGHT_N
    ground_tangential_n = abs(solution["T1"].to_float()) * WEIGHT_N

    #: 罚刚度扫描。**只取一阶区间的三档**：``k = 2e7``起实测偏差已经进浮点地板
    #: （力的绝对噪声约``7e-15·k``，而力本身是``O(W)``），拿它算阶就是在算噪声。
    #: 这与既有`three_sphere_pyramid`那条同源，只是本案例的地板高一个数量级——
    #: 转动块让同一个力经过更长的装配链，舍入积得更多。
    convergence_stiffnesses = (2.0e4, 2.0e5, 2.0e6)
    force_stiffness = 2.0e6
    #: 2e6档实测``μ``偏差1.766e-07，取5e-7约2.8倍余量。
    force_relative_tolerance = 5.0e-7

    #: —— 两侧那一组（决策0080）：撑住是不动点，塌是恒速率的不可逆滑移 ——
    #:
    #: **塌那一边的速率有闭式**，所以它不是一句定性话：
    #: 全粘着解在球-球接触上要求``|f| = μc·F``，而摩擦锥只给得起``μ·F``，
    #: 差出来的那一截除以切向刚度就是这一步滑掉的距离
    #:
    #:     滑距/步 = (μc − μ)·F/k_t，   F = W/2
    #:
    #: ``μ → μc⁻``时它线性趋于零，``μ ≥ μc``时**逐位为零**——
    #: 于是"撑住/塌"这条定性判据有了一个连续的、可对拍的量。
    hold_friction = 0.30
    collapse_friction = 0.20
    rate_law_frictions = (0.26, 0.20, 0.10)
    sphere_normal_force_n = solution["F1"].to_float() * WEIGHT_N

    def slip_per_step_mm(mu: float) -> float:
        """一步一个球-球接触滑掉多少。``exact_mu``是精确的，只有``μ``与``k``是输入。"""

        return (critical_friction - mu) * sphere_normal_force_n / force_stiffness

    def rate_tolerance(mu: float) -> float:
        """速率律的容差是**推出来的，不是试出来的**。

        实测``|f|/F``比精确``μc``大``δ = 1.766e-07``（相对），
        而滑距正比于``μc − μ``，所以同一个``δ``在滑距上被放大``μc/(μc − μ)``倍。
        取与``μc``那条同一个声明容差``5e-7``再乘这个放大因子，
        余量因此**逐档保持2.6—2.8倍**（实测5.98e-6/7.24e-7/3.09e-7）。

        **不写成一个统一的魔数**：统一魔数要么在``μ``靠近``μc``时过紧、
        要么在远处松到抓不住错——而这条判据的全部意义就在``μ → μc``那一端。
        """

        return force_relative_tolerance * critical_friction / (critical_friction - mu)

    oracles = [
        {
            "id": "oracle:pyramid_rot/critical_friction",
            "inputs": {
                "kind": "three_sphere_pyramid_rotational_critical_friction",
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
                "stiffness_n_per_mm": force_stiffness,
            },
            "expected": {
                "critical_friction": critical_friction,
                "sphere_sphere_is_the_binding_contact": True,
            },
            "tolerances": {
                "critical_friction": {
                    "abs": 0.0,
                    "rel": force_relative_tolerance,
                    "reason": "``μc = 2−√3``由`Q(√3)`精确算术解出（本脚本`exact_solution`，"
                              "9方程/8未知、精确秩8）。罚柔度让实测``|f|/F``带``O(1/k)``"
                              "偏差——穿透改变接触几何本身。``k = 2e6``档实测1.766e-07，"
                              "取5e-7约2.8倍余量。**这个容差是模型的柔度，不是实现的余量**，"
                              "收敛阶那条门专门验这一点",
                },
                "sphere_sphere_is_the_binding_contact": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差**：卡住的是球-球接触（需求``2−√3``）而不是地面"
                              "（需求``(2−√3)/3``，宽松3倍）。research/15第3.2节末点明："
                              "将来若给两处配不同的``μ``，``μc``这一个数就不够用了——"
                              "所以哪一处卡住必须是被断言的，不是被默认的",
                },
            },
        },
        {
            "id": "oracle:pyramid_rot/force_decomposition",
            "inputs": {
                "kind": "pyramid_rotational_static_decomposition",
                "stiffness_n_per_mm": force_stiffness,
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
            },
            "expected": {
                "sphere_contact_force_n": sphere_contact_force_n,
                "sphere_tangential_n": sphere_tangential_n,
                "ground_normal_n": ground_normal_n,
                "ground_tangential_n": ground_tangential_n,
            },
            "tolerances": {
                "sphere_contact_force_n": {
                    "abs": 0.0, "rel": force_relative_tolerance,
                    "reason": "``F = W/2``（精确有理数）。**它与无摩擦变体的``W/√3``不同**"
                              "——加力矩平衡改的是力怎么分。``O(1/k)``柔度偏差同上",
                },
                "sphere_tangential_n": {
                    "abs": 0.0, "rel": force_relative_tolerance,
                    "reason": "``f = (1 − √3/2)·W``。**这一项在无摩擦变体里恒为零**，"
                              "它就是转动自由度带进来的那个量",
                },
                "ground_normal_n": {
                    "abs": 0.0, "rel": force_relative_tolerance,
                    "reason": "``N = 3W/2``。**两个模型完全相同**（竖直平衡不经过接触角），"
                              "所以它是一条**回归判据**而不是新能力：转动进来不许改总承载。"
                              "实测偏差7e-11量级，远严于本容差；容差仍取统一值，"
                              "理由是判据表的可读性优先于逐条压紧",
                },
                "ground_tangential_n": {
                    "abs": 0.0, "rel": force_relative_tolerance,
                    "reason": "``|T| = (1 − √3/2)·W``。无摩擦变体给``W/(2√3)``，"
                              "两者比``1.392305``——**0063第二节那个1.4倍就落在这一格上**",
                },
            },
        },
        {
            "id": "oracle:pyramid_rot/top_sphere_moment_is_redundant",
            "inputs": {
                "kind": "pyramid_rotational_redundant_equation",
                "stiffness_n_per_mm": force_stiffness,
            },
            "expected": {"top_sphere_moment_residual_n_mm": 0.0},
            "tolerances": {
                "top_sphere_moment_residual_n_mm": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**逐位零容差。** research/15第3.2节写着第9条方程冗余："
                              "构型镜像对称、顶球在对称面上，故它的``ΣM``自动满足。"
                              "本案例把顶球自旋钉住（否则'整体滚动'是Hessian的精确零模、"
                              "牛顿走不动），于是那条方程变成一个**约束反力矩**；"
                              "它必须**恰为0.0**，否则钉住这一步就改了问题。"
                              "**这是把research/15的一句论断变成可判的门**",
                },
            },
        },
        {
            "id": "oracle:pyramid_rot/compliance_is_first_order",
            "inputs": {
                "kind": "penalty_compliance_convergence_order",
                "stiffnesses_n_per_mm": list(convergence_stiffnesses),
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
            },
            "expected": {
                "deviation_ratio_low": 8.0,
                "deviation_ratio_high": 12.0,
                "deviations_shrink": True,
            },
            "tolerances": {
                "deviation_ratio_low": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "区间端点是**声明的判据**不是测出来的数：刚度每涨10倍偏差应降约10倍。"
                              "实测9.998与10.017，区间取[8,12]。**不写死为10**——"
                              "一阶是渐近性质，写死会让正确实现在别的构型上红",
                },
                "deviation_ratio_high": {
                    "abs": 0.0, "rel": 0.0, "reason": "同上，区间上端",
                },
                "deviations_shrink": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**前置断言**：偏差必须逐档单调减小。"
                              "不减小时比值区间那条门是在拿噪声算阶",
                },
            },
        },
        {
            "id": "oracle:pyramid_rot/two_sided_hold_and_collapse",
            "inputs": {
                "kind": "three_sphere_pyramid_rotational_two_sided",
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
                "stiffness_n_per_mm": force_stiffness,
                "hold_friction": hold_friction,
                "collapse_friction": collapse_friction,
                "hold_steps": 25,
                "collapse_steps": 60,
            },
            "expected": {
                "hold_slip_increment_mm": 0.0,
                "hold_is_a_fixed_point": True,
                "hold_regime": 1.0,
                "hold_demand_ratio": critical_friction,
                "collapse_sphere_regime": 2.0,
                "collapse_ground_regime": 1.0,
                "collapse_ground_slip_increment_mm": 0.0,
                "collapse_base_gap_grows_strictly": True,
                "collapse_top_descends_strictly": True,
                "collapse_slip_does_not_decay": True,
                "collapse_every_step_converges": True,
                "collapse_slip_per_contact_mm": slip_per_step_mm(collapse_friction),
            },
            "tolerances": {
                "hold_slip_increment_mm": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**逐位零**（判`float.hex()`必须是``0x0.0p+0``）。"
                              "``μ > μc``时试探力落在锥内，return-map返回的锚点修正是"
                              "**构造性的零矢量**，不是一个很小的数——"
                              "给它任何非零容差都等于允许一次不该发生的耗散",
                },
                "hold_is_a_fixed_point": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "整条状态向量**逐位不变**（25步全判`float.hex()`）。"
                              "只判滑距为零不够：滑距为零而位形在漂，说明有别的东西"
                              "在写状态——**"
                              "撑住的含义是这一步什么都没发生**，那是可以逐位判的",
                },
                "hold_regime": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "四个槽全是`REGIME_STICK`。**它与滑距为零不是同一条**："
                              "一个把regime写死成STICK的实现照样滑，"
                              "一个忘了写regime的实现照样不滑",
                },
                "hold_demand_ratio": {
                    "abs": 0.0, "rel": force_relative_tolerance,
                    "reason": "撑住那一边等价于``max(|f|/N) ≤ μ``，而这个上确界就是``μc``。"
                              "**判它而不只判“没滑”**：没滑可能是因为力算错了，"
                              "而需求比对上``2−√3``说明力是对的、锥也是对的",
                },
                "collapse_sphere_regime": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "球-球两槽全程`REGIME_SLIP`——**卡住的是球-球**，"
                              "与`sphere_sphere_is_the_binding_contact`同一条物理，"
                              "但这一次是从**滑移**那一侧独立判出来的",
                },
                "collapse_ground_regime": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "地面两槽全程`REGIME_STICK`：地面的需求是``(2−√3)/3``，"
                              "宽松3倍，``μ = 0.20``按不动它。"
                              "**“塌”不是“到处都在滑”**，分不清这两件事的实现要红",
                },
                "collapse_ground_slip_increment_mm": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "逐位零（`float.hex()`）。与上一条配对：判别与滑距**都**要对，"
                              "因为它们由两段不同的代码写出来",
                },
                "collapse_base_gap_grows_strictly": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**两底球逐步严格分开**（不是“平均趋势”，是逐步比较）。"
                              "这是“塌”的方向判据：滑移必须朝金字塔散架那一侧走，"
                              "**朝反方向单调也满足“锚点动了”**——那正是把符号取反的实现"
                              "会给出的样子（见`_assemble_stick`那段符号注释）",
                },
                "collapse_top_descends_strictly": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "顶球逐步严格下降。与上一条一起把“塌”钉成两个独立方向上的"
                              "单调，而不是一个可以靠噪声凑出来的标量",
                },
                "collapse_slip_does_not_decay": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**这一条才是“塌”与“松一下就稳住”的分界**：末步滑距不小于首步。"
                              "``μ < μc``时静解**不存在**，所以每一步滑掉同样多、"
                              "累计滑移线性发散；一个“滑一下就收敛”的实现在这里必红。"
                              "实测60步内比值1.0000028（增，不减）",
                },
                "collapse_every_step_converges": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**必红专防：不许拿“求解器不收敛”冒充“塌了”。**"
                              "两者是两回事——塌是“静解不存在”这条物理，"
                              "不收敛是数值。本判据要求塌那一边的**每一步**都干净收敛，"
                              "于是“塌”这个结论只能来自滑移与位形，不能来自异常。"
                              "近阈值确有一条不收敛带（见case.md第四节第7条），"
                              "本组的``μ``刻意远离它",
                },
                "collapse_slip_per_contact_mm": {
                    "abs": 0.0, "rel": rate_tolerance(collapse_friction),
                    "reason": "闭式``(μc − μ)·F/k_t``，``F = W/2``。"
                              "**把定性的“塌”变成一个有闭式的速率**——"
                              "它在``μ → μc⁻``线性趋零，于是两侧判据不是一刀切的开关，"
                              "而是一条连续的、可对拍的曲线。容差由`rate_tolerance`推出",
                },
            },
        },
        {
            "id": "oracle:pyramid_rot/slip_onset_threshold",
            "inputs": {
                "kind": "three_sphere_pyramid_rotational_slip_onset",
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
                "stiffness_n_per_mm": force_stiffness,
                "bracket_low": 0.20,
                "bracket_high": 0.35,
                "bracket_width": 1.0e-12,
            },
            "expected": {
                "critical_friction": critical_friction,
                "onset_matches_stick_demand": True,
            },
            "tolerances": {
                "critical_friction": {
                    "abs": 0.0, "rel": force_relative_tolerance,
                    "reason": "**这一条是从另一条路走到同一个数**：不解全粘着解、"
                              "不读任何力，只对``μ``二分“走一步之后锚点动没动”。"
                              "锚点是历史，动没动是一个布尔量，"
                              "所以这条判据不含任何容差——容差只在最后与``2−√3``对拍时出现。"
                              "实测阈值0.26794923975367058，相对偏差1.766e-07，"
                              "与全粘着解那条**同一个数**（0079第5.1节）",
                },
                "onset_matches_stick_demand": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "二分出来的阈值必须与全粘着解的需求比``max(|f|/N)``"
                              "落在同一个夹取区间内（实测差1.33e-13 ≤ 半宽5e-13）。"
                              "**没有这一条，两条路各自对上``2−√3``也可能是巧合**——"
                              "5e-7的容差留得下一个1e-7量级的共同偏差",
                },
            },
        },
        {
            "id": "oracle:pyramid_rot/collapse_rate_law",
            "inputs": {
                "kind": "three_sphere_pyramid_rotational_collapse_rate",
                "radius_mm": RADIUS_MM,
                "mass_kg": MASS_KG,
                "gravity_mm_s2": GRAVITY_MM_S2,
                "stiffness_n_per_mm": force_stiffness,
                "frictions": list(rate_law_frictions),
            },
            "expected": {
                _rate_key(mu): slip_per_step_mm(mu) for mu in rate_law_frictions
            },
            "tolerances": {
                _rate_key(mu): {
                    "abs": 0.0,
                    "rel": rate_tolerance(mu),
                    #: **逐档一个容差，不是三档共用一个**。`_parse_tolerance`只收标量，
                    #: 而这恰好逼出了对的形制：``μ = 0.26``要1.7e-5、``μ = 0.10``只要8.0e-7，
                    #: 共用一个数就等于让远端那两档白判。
                    "reason": f"``μ = {mu}``：闭式``(μc − μ)·F/k_t``，容差由"
                              "`rate_tolerance`推出（``5e-7·μc/(μc−μ)``）。"
                              "实测偏差5.98e-6/7.24e-7/3.09e-7（三档），余量2.8/2.7/2.6倍。"
                              "**三档一起才有内容**：单档对上只说明那一个数凑巧，"
                              "三档沿``μc − μ``线性对上说明速率律本身是对的",
                }
                for mu in rate_law_frictions
            },
        },
        {
            "id": "oracle:pyramid_rot/rotation_free_variant_differs",
            "inputs": {"kind": "pyramid_rotational_versus_frictionless_variant"},
            "expected": {
                "critical_friction_ratio": critical_friction / (1.0 / (3.0 * math.sqrt(3.0))),
            },
            "tolerances": {
                "critical_friction_ratio": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "``(2−√3)/(1/(3√3)) = 1.392305``。**两个数都是闭式**，"
                              "所以这条门判的是两支闭式之比，不含任何实测——"
                              "它是0063第二节'转动缺席时系统性偏约四成'那句话的算术形式。"
                              "容差1e-12只留给double的表示误差",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/three_sphere_pyramid_rotational",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/three_sphere_pyramid_rotational/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print("精确解（以W为单位）：")
    for name in ("F1", "f1", "N1", "T1"):
        print(f"   {name} = {solution[name].text():>16} = {solution[name].to_float():+.15f}·W")
    print(f"球-球 |f|/F = {sphere_ratio.text()}  精确等于 2−√3 ? True")
    print(f"mu_c = {critical_friction:.17f}")
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


def n_r_x() -> Q3:
    """``n_R``的``x``分量。单独成函数只为让上面那条"往外推"的断言读起来是物理。"""

    return ZERO - HALF


def t_r_x() -> Q3:
    """``t_R``的``x``分量。"""

    return SQRT3 * HALF


if __name__ == "__main__":
    raise SystemExit(main())
