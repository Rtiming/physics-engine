#!/usr/bin/env python3
"""`AxialStretch`梯度与Hessian的独立解析金标——**精确有理算术，不调被验内核**。

本案例闭合的是决策0024第六节登记的那处缺口：拉伸项的Hessian此前**只有有限差分
背书**。按spec/12第6.1节的口径，有限差分门只验"雅可比是不是我写的那个能量的导数"，
不验"那个能量对不对"——能量本身写错时FD照样全绿。这份金标是那一栏缺的第二道门。

## 独立性（轴7规则4的力学版，spec/12第3.2节）

本脚本**不import`physics_engine.energies`**，一个函数也不复用。它只从
`physics_engine.oracles`取清单落盘的机械件（那是形制层，不是被验物理）。

## 两条各自独立的精确路径，互为对拍源

* **路径A（机械求导）**：二阶前向`Jet`（值+一阶+二阶），只实现`+`/`−`/`×`/`√`
  四条链式法则，然后把单条边的能量
  ``U = ½·(EA/l0)·(|x_j − x_i| − l0)²``
  照定义写一遍让它自己求二阶导。**它一个字也没提`d⊗d`、没提横向项**——
  它不知道答案长什么样，这正是它作为独立路径的价值。
* **路径B（手推闭式）**：手工对同一能量求二阶导，得
  ``∂²U/∂Δ_a∂Δ_b = k·[ δ_ab·(1 − l0/L) + l0·Δ_aΔ_b/L³ ]``。
  这个形式与被验内核里那个``k·d⊗d + (k·ε/L)·(I − d⊗d)``代数上等价但**写法不同**，
  所以它不是"在测试里复述oracle公式"，它是第三种写法。

两条路径全程用``fractions.Fraction``，**逐位相等**是脚本的自检（`assert`），
不是容差比对。冻结进清单的是路径A的值。

## 精确性的边界（如实登记）

``√``要精确就必须让被开方数是完全平方。六个构型里有五个的边矢量是勾股型
（如(3,4,0)→5、(12,15,16)→25、(2,3,6)→7），这五个的金标是**数学上精确的**；
第六个`irrational_length`故意取非完全平方，此时``rational_sqrt``退化成
60位十进制的有理截断——截断相对误差 < 1e-60，比float64的机器epsilon
（2.2e-16）低44个数量级，对判据没有任何影响。这一条写进清单理由栏。
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/axial_stretch_hessian"
ALGORITHM_VERSION = "1.0.0"

#: 非完全平方时``√``的有理截断位数。60位十进制 → 截断相对误差 < 1e-60，
#: 比float64机器epsilon低44个数量级；金标的自身误差因此不进判据。
SQRT_DIGITS = 60

#: IEEE-754双精度的单位舍入 u = 2⁻⁵³。容差全部以它为单位表达——
#: **判据的量纲是"多少个ulp"，不是一个从天上掉下来的小数**。
UNIT_ROUNDOFF = 2.0**-53


def rational_sqrt(value: Fraction) -> tuple[Fraction, bool]:
    """``√value``的有理表示。完全平方时精确；否则截断到``SQRT_DIGITS``位十进制。

    ``√(p/q) = √(p·q)/q``——先化成整数开方，用``math.isqrt``（精确整数运算，
    不经过float）判定并求根。返回``(值, 是否精确)``。
    """

    if value < 0:
        raise ValueError("sqrt of a negative rational")
    numerator, denominator = value.numerator, value.denominator
    product = numerator * denominator
    root = math.isqrt(product)
    if root * root == product:
        return Fraction(root, denominator), True
    scale = 10**SQRT_DIGITS
    root = math.isqrt(product * scale * scale)
    return Fraction(root, denominator * scale), False


class Jet:
    """``n``元二阶前向jet：值、一阶导、二阶导，全程``Fraction``。

    **这一层不知道拉伸能长什么样**。它只知道四条链式法则；把能量按定义写一遍，
    二阶导就自己出来了。这就是"机械求导"作为独立路径的含义——
    它不可能"抄"被验实现的形式，因为它根本没有形式可抄。
    """

    __slots__ = ("value", "grad", "hess")

    def __init__(
        self, value: Fraction, grad: list[Fraction], hess: list[list[Fraction]]
    ) -> None:
        self.value = value
        self.grad = grad
        self.hess = hess

    @staticmethod
    def variable(value: float, index: int, size: int) -> Jet:
        grad = [Fraction(0)] * size
        grad[index] = Fraction(1)
        return Jet(Fraction(value), grad, [[Fraction(0)] * size for _ in range(size)])

    @staticmethod
    def constant(value: Fraction | float | int, size: int) -> Jet:
        return Jet(
            Fraction(value),
            [Fraction(0)] * size,
            [[Fraction(0)] * size for _ in range(size)],
        )

    def __add__(self, other: Jet) -> Jet:
        size = len(self.grad)
        return Jet(
            self.value + other.value,
            [self.grad[a] + other.grad[a] for a in range(size)],
            [
                [self.hess[a][b] + other.hess[a][b] for b in range(size)]
                for a in range(size)
            ],
        )

    def __sub__(self, other: Jet) -> Jet:
        size = len(self.grad)
        return Jet(
            self.value - other.value,
            [self.grad[a] - other.grad[a] for a in range(size)],
            [
                [self.hess[a][b] - other.hess[a][b] for b in range(size)]
                for a in range(size)
            ],
        )

    def __mul__(self, other: Jet) -> Jet:
        """``(fg)'' = f·g'' + g·f'' + f'⊗g' + g'⊗f'``。"""

        size = len(self.grad)
        return Jet(
            self.value * other.value,
            [
                self.value * other.grad[a] + other.value * self.grad[a]
                for a in range(size)
            ],
            [
                [
                    self.value * other.hess[a][b]
                    + other.value * self.hess[a][b]
                    + self.grad[a] * other.grad[b]
                    + self.grad[b] * other.grad[a]
                    for b in range(size)
                ]
                for a in range(size)
            ],
        )

    def sqrt(self) -> tuple[Jet, bool]:
        """``f = √v``：``f' = 1/(2r)``、``f'' = −1/(4r³)``，其中``r = √v``。

        二阶导写成``−1/(4r³)``而不是``−1/(4v·r)``是有意的：截断开方时
        ``r² ≠ v``，用``r``贯穿才能让本路径与手推闭式路径**逐位相等**。
        """

        size = len(self.grad)
        root, exact = rational_sqrt(self.value)
        first = Fraction(1, 2) / root
        second = Fraction(-1, 4) / (root * root * root)
        return (
            Jet(
                root,
                [first * self.grad[a] for a in range(size)],
                [
                    [
                        first * self.hess[a][b] + second * self.grad[a] * self.grad[b]
                        for b in range(size)
                    ]
                    for a in range(size)
                ],
            ),
            exact,
        )


def path_a_mechanical(
    coordinates: tuple[float, ...], edges: tuple[tuple[int, int, float, float], ...]
) -> tuple[list[Fraction], list[list[Fraction]], bool]:
    """路径A：把能量按定义写一遍，让jet自己求一阶与二阶导。"""

    size = len(coordinates)
    variables = [Jet.variable(value, index, size) for index, value in enumerate(coordinates)]
    total = Jet.constant(0, size)
    all_exact = True
    for node_i, node_j, rest_mm, stiffness_n in edges:
        delta = [variables[3 * node_j + a] - variables[3 * node_i + a] for a in range(3)]
        squared = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
        length, exact = squared.sqrt()
        all_exact = all_exact and exact
        elongation = length - Jet.constant(rest_mm, size)
        half_k = Jet.constant(Fraction(stiffness_n) / Fraction(rest_mm) / 2, size)
        total = total + half_k * elongation * elongation
    return total.grad, total.hess, all_exact


def path_b_closed_form(
    coordinates: tuple[float, ...], edges: tuple[tuple[int, int, float, float], ...]
) -> tuple[list[Fraction], list[list[Fraction]]]:
    """路径B：手推闭式。

    对单条边，令``Δ = x_j − x_i``、``L = |Δ|``、``k = EA/l0``：

        ∂U/∂Δ_a       = k·(L − l0)·Δ_a/L
        ∂²U/∂Δ_a∂Δ_b  = k·[ δ_ab·(1 − l0/L) + l0·Δ_a·Δ_b/L³ ]

    第二式的推法（一行，可独立核对）：把``U``写成
    ``½k·s − k·l0·√s + ½k·l0²``（``s = Δ·Δ``），只有``√s``不是多项式，
    对它求两次导即得。**它与被验内核那个``k·d⊗d + (k·ε/L)(I − d⊗d)``
    代数等价但形式不同**——两个形式的等价性是本脚本的自检内容之一。

    链到节点：``∂Δ/∂x_j = +I``、``∂Δ/∂x_i = −I``，于是``(i,i)``与``(j,j)``块取``+``、
    ``(i,j)``与``(j,i)``块取``−``。
    """

    size = len(coordinates)
    gradient = [Fraction(0)] * size
    hessian = [[Fraction(0)] * size for _ in range(size)]
    for node_i, node_j, rest_mm, stiffness_n in edges:
        delta = [
            Fraction(coordinates[3 * node_j + a]) - Fraction(coordinates[3 * node_i + a])
            for a in range(3)
        ]
        squared = sum((component * component for component in delta), Fraction(0))
        length, _ = rational_sqrt(squared)
        k = Fraction(stiffness_n) / Fraction(rest_mm)
        rest = Fraction(rest_mm)
        for a in range(3):
            force_a = k * (length - rest) * delta[a] / length
            gradient[3 * node_j + a] += force_a
            gradient[3 * node_i + a] -= force_a
            for b in range(3):
                identity = Fraction(1) if a == b else Fraction(0)
                block = k * (
                    identity * (Fraction(1) - rest / length)
                    + rest * delta[a] * delta[b] / (length * length * length)
                )
                hessian[3 * node_i + a][3 * node_i + b] += block
                hessian[3 * node_j + a][3 * node_j + b] += block
                hessian[3 * node_i + a][3 * node_j + b] -= block
                hessian[3 * node_j + a][3 * node_i + b] -= block
    return gradient, hessian


#: 六个固定构型。坐标全部取**二进制可精确表示**的数，于是``x_j − x_i``逐位精确，
#: 被验内核的舍入只可能出现在开方及其下游——误差预算因此可以逐步算出来。
#: 五个的边长是勾股型（完全平方），第六个故意不是。
CONFIGURATIONS: tuple[dict, ...] = (
    {
        "name": "at_rest",
        "nodes": 2,
        "coordinates": (0.0, 0.0, 0.0, 3.0, 4.0, 0.0),
        "edges": ((0, 1, 5.0, 2000.0),),
        "note": "伸长量恰为0：横向项整块消失，Hessian退化成秩1的k·d⊗d，力恰为0。",
    },
    {
        "name": "large_strain",
        "nodes": 2,
        "coordinates": (0.0, 0.0, 0.0, 3.0, 4.0, 0.0),
        "edges": ((0, 1, 4.0, 2000.0),),
        "note": "25%应变：横向项是轴向项的1/5，量级最大，漏掉它最容易被这一条抓住。",
    },
    {
        "name": "near_rest_ratio_200",
        "nodes": 2,
        "coordinates": (1.0, -2.0, 3.0, 13.0, 13.0, 19.0),
        "edges": ((0, 1, 24.875, 45.0),),
        "note": "L/δ=200——与决策0024第三节那个相消放大因子同一个数。",
    },
    {
        "name": "near_rest_ratio_25600",
        "nodes": 2,
        "coordinates": (1.0, -2.0, 3.0, 13.0, 13.0, 19.0),
        "edges": ((0, 1, 24.9990234375, 45.0),),
        "note": "L/δ=25600：把相消推到极端，看它到底进不进Hessian的判据。",
    },
    {
        "name": "irrational_length",
        "nodes": 2,
        "coordinates": (0.0, 0.0, 0.0, 1.5, 2.25, 0.75),
        "edges": ((0, 1, 2.8, 1200.0),),
        "note": "边长不是完全平方——**唯一一个内核的fl(L)带舍入误差的构型**。",
    },
    {
        "name": "chain_stretch_and_compression",
        "nodes": 3,
        "coordinates": (0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 5.0, 7.0, 6.0),
        "edges": ((0, 1, 4.9375, 800.0), (1, 2, 7.25, 1500.0)),
        "note": "两条边共用节点1，一拉一压：这一条同时验装配（叠加与符号）。",
    },
)


def _float_length(coordinates, edge) -> float:
    """内核会算出的那个``fl(L)``——**只用于推导容差，不参与金标**。"""

    node_i, node_j, _, _ = edge
    squared = 0.0
    for a in range(3):
        component = coordinates[3 * node_j + a] - coordinates[3 * node_i + a]
        squared += component * component
    return math.sqrt(squared)


def _amplification(coordinates, edges) -> float:
    """相消放大因子``L/|δ|``的最大值；**仅在``fl(L)``不精确时才计入容差**。

    ``fl(L)``精确时（边长是完全平方），``δ = fl(L) − l0``按Sterbenz引理也精确，
    相消根本没有发生——那时把``L/δ``写进容差是凭空放宽。
    """

    worst = 0.0
    for edge in edges:
        exact_length, exact = rational_sqrt(
            sum(
                (
                    Fraction(coordinates[3 * edge[1] + a]) - Fraction(coordinates[3 * edge[0] + a])
                ) ** 2
                for a in range(3)
            )
        )
        if exact and Fraction(_float_length(coordinates, edge)) == exact_length:
            continue
        elongation = exact_length - Fraction(edge[2])
        if elongation == 0:
            continue
        worst = max(worst, float(abs(exact_length / elongation)))
    return worst


def main() -> int:
    oracles = []
    for config in CONFIGURATIONS:
        coordinates = config["coordinates"]
        edges = config["edges"]
        grad_a, hess_a, sqrt_exact = path_a_mechanical(coordinates, edges)
        grad_b, hess_b = path_b_closed_form(coordinates, edges)
        # **自检：两条独立路径逐位相等**。这不是容差比对，是精确有理数相等。
        assert grad_a == grad_b, f"{config['name']}: 两条独立路径的梯度不一致"
        assert hess_a == hess_b, f"{config['name']}: 两条独立路径的Hessian不一致"

        stiffness_scale = max(stiffness / rest for _, _, rest, stiffness in edges)
        at_rest = all(value == 0 for value in grad_a)
        amplification = _amplification(coordinates, edges)

        # 梯度容差：力 = k·δ，**δ的相对误差直接进结果**，所以相消放大因子
        # L/δ在这里是真的。16u是与δ无关的那部分（k、方向余弦、两次乘法）。
        gradient_rel = 0.0 if at_rest else (16.0 + 4.0 * amplification) * UNIT_ROUNDOFF
        # Hessian容差：**按刚度尺度的绝对判据**，不是逐项相对。
        hessian_abs = 16.0 * UNIT_ROUNDOFF * stiffness_scale

        exact_note = (
            "边长是完全平方，本条金标在数学上**精确**（无截断）。"
            if sqrt_exact
            else f"边长不是完全平方，开方截断到{SQRT_DIGITS}位十进制，"
            "金标自身相对误差<1e-60，比float64机器epsilon低44个数量级。"
        )
        gradient_reason = (
            "**零容差**：静止构型上``l0``恰等于``fl(L)``（边长是完全平方，"
            "两者都是精确的float），伸长量恰为0.0，力恰为0.0，"
            "内核与金标必须逐位相等。零容差同样要交代理由——这就是它。" + exact_note
            if at_rest
            else (
                "梯度是逐项**相对**判据，因为力=k·δ**正比于伸长量**，"
                f"δ的相对误差原封不动进结果。本构型的相消放大因子L/|δ|={amplification:.1f}"
                "（0表示fl(L)精确，于是δ=fl(L)−l0按Sterbenz引理也精确，"
                "相消根本没有发生，那时把L/δ写进容差是凭空放宽）。"
                "预算=(16 + 4·放大因子)·u，u=2⁻⁵³：16u是与δ无关的那部分"
                "（k一次除法、方向余弦两次除法、两次乘法各≤u，取宽），"
                "4u·(L/δ)是fl(L)那一个ulp经相消放大后的部分。"
                "**δ取得更小这一项会更大**——与决策0024第三节能量判据里那个200倍"
                "是同一个机制。" + exact_note
            )
        )
        oracles.append(
            {
                "id": f"oracle:axial_stretch_hessian/{config['name']}",
                "inputs": {
                    "kind": "analytic_derivatives",
                    "term": "axial_stretch",
                    "coordinates_mm": list(coordinates),
                    "edges": [list(edge) for edge in edges],
                    "note": config["note"],
                },
                "expected": {
                    "gradient_n": [float(value) for value in grad_a],
                    "hessian_n_mm": [float(value) for row in hess_a for value in row],
                },
                "tolerances": {
                    "gradient_n": {
                        "abs": 0.0,
                        "rel": gradient_rel,
                        "reason": gradient_reason,
                    },
                    "hessian_n_mm": {
                        "abs": hessian_abs,
                        "rel": 0.0,
                        "reason": (
                            "Hessian用**按刚度尺度的绝对**判据："
                            f"abs = 16·u·k_max = 16·2⁻⁵³·{stiffness_scale:.6g} = {hessian_abs:.3e} N/mm，"
                            "u=2⁻⁵³。逐项相对判据在这里是错的选择——横向对角项量级是k·δ/L，"
                            "δ取小它趋于零，相对判据会在一个物理上无关紧要的小量上炸。"
                            "**相消放大因子L/δ不进这条判据**，而且这是算出来的不是猜的："
                            "横向系数t=k·δ/L的绝对误差 = (k/L)·|Δδ| = (k/L)·(c·u·L) = c·u·k，"
                            "δ在里面约掉了；而块的尺度就是k，所以相对精度回到O(u)。"
                            "16u的来处：方向余弦外积≤7u、乘k再累加≤2u、横向项≤5u、"
                            "装配最多两条边叠加，取宽记16u。"
                            "**这条容差可以取紧**，因为内核只用IEEE-754基本运算与正确舍入的"
                            "sqrt且次序固定，输出跨平台逐位相同，不需要留机器余量。" + exact_note
                        ),
                    },
                },
            }
        )

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/axial_stretch_hessian",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/axial_stretch_hessian/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
