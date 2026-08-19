"""距离场进接触——决策0074第二节的落地（甲2、0074执行路线阶段四的内核那一半）。

**形制一个字不改。** 0074第二节把话说死了：`contact/penalty.py`里每一个接触项
本来就是一个解析距离场，接触层要的输入只有三样——``g``、``∇g``、``∇²g``：

    U = ½·k·g²（仅g<0）,  ∇U = k·g·∇g,  ∇²U = k·(∇g⊗∇g + g·∇²g)

本模块做的只有一件事：**把这三样从解析式换成可查询的场**。
罚势的写法、活动条件、量纲、`normal_force_n`是唯一精确输出这些性质，
与`PenaltyNormalContact`逐条相同——**不同的只有``g``从哪来**。

## 两块东西，各自解一道题

1. `SignedDistanceField`——**窄带块稀疏存储**（0074第5.2节）。
   照OpenVDB公开描述的结构做简化版：定长块（缺省8³＝512个节点，与它的叶节点同尺寸）
   ＋按块坐标哈希。**纯数字、零依赖**，不引入OpenVDB（无macOS arm64 wheel）；
2. **三次B样条插值**——`solve.py`第29行自己申报的适用域是"``U``二次连续可微"。
   三线性只有C⁰、梯度跨胞界跳变，牛顿会在胞界上抖。
   **这不是精度偏好，是适用域的硬要求**（0074第二节第4条）。

## 精度：为什么是**二阶**，以及那个二阶是从哪来的

本模块用**采样值直接当B样条系数**（不做预滤波）。一维上这是标准的拟插值：

    S(x) = Σ_i f(x_i)·B((x − x_i)/h) = f(x) + (h²/6)·f''(x) + O(h⁴)

那个``1/6``是三次B样条的二阶矩：``Σ_k (k − t)²·B(k − t) = 1/3``**与``t``无关**
（本模块在``t = 0``与``t = 0.5``两处各验过一次，都是1/3）。逐条后果：

* **值、梯度、Hessian都是二阶**（把上式逐次求导，系数函数光滑）；
* **仿射函数被精确重构**——``f'' ≡ 0``。于是半空间那条解析SDF
  在**任何**分辨率下都逐位精确，误差里连一个截断项都没有。
  这条是可测的，也是本模块与`PenaltyNormalContact`并排那道门判的东西；
* **预滤波能到四阶但本模块不做**：那要在整条带上解一个三对角系统，
  而窄带块稀疏存储**没有全局的"整条带"可解**（块是按需存在的，
  边界条件无处安放）。这条登记成GAP，触发条件写在决策0085里。

## 窄带外：**失败关闭**（0074没裁，本模块裁）

查询点的4×4×4支撑只要缺一个节点，`SignedDistanceFieldError`当场抛。
**不外推**，理由是一条实测得出的不对称：

> 窄带外有两种点——**远在体外**（``g ≫ 0``，不接触）与**深在体内**
> （``g ≪ 0``，接触且力很大）。稀疏存储里这两种点长得**一模一样**：
> 都是"这个块不存在"。要区分它们必须另有一个内外判据
> （OpenVDB是靠上层瓦片的背景符号），而本模块的块表**不带那个东西**。

于是外推等于在"不接触"和"接触力很大"之间猜一个，而猜错的那一半是静默的：
模型会安静地穿过工件。**AGENTS.md诚实可信度条款：不知道就说不知道。**
调用方要的是`contains_stencil`——先问再算，或者把带烘宽一点。

**GAP与触发条件在决策0085第五节**：带上背景符号瓦片之后，
窄带外才可以变成"按符号外推到±∞"，而那要等第一个真实件烘进来（甲3）。

## 单位

长度mm、``g``mm、``k``N/mm、``U``N·mm——与`PenaltyNormalContact`逐字相同，
理由也相同（``½kg²``是``N/mm · mm² = N·mm``，直接就是本仓的能量单位）。
``∇g``无量纲（距离对距离求导），``∇²g``是1/mm。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.shapes import Vector3
from physics_engine.state import State

#: 块边长的2的对数。3 → 8³ = 512个节点/块，与OpenVDB叶节点同尺寸（0074第5.2节）。
#: **它是形制常量不是调参旋钮**：块尺寸进`SignedDistanceField`的字段，
#: 换了尺寸的场与旧场不是同一个字节形制。
DEFAULT_BLOCK_LOG2 = 3

#: 三次B样条的支撑宽度（每轴4个节点）。查询点两侧各要2个节点——
#: **窄带必须比这个宽**，否则带内的点也查不了。
STENCIL_WIDTH = 4


class SignedDistanceFieldError(ContactError):
    """距离场的失败关闭。

    继承`ContactError`是有意的：调用方按"接触算不出来"这一类接住即可。
    窄带外的查询走这一条——见模块docstring"窄带外"那一节。
    """


# --------------------------------------------------------------------------
# 三条解析SDF：0074第二节那张表的前三行，也是本模块全部逼近误差判据的金标
# --------------------------------------------------------------------------


def half_space_distance_mm(
    point_mm: Vector3, plane_point_mm: Vector3, unit_normal: Vector3
) -> float:
    """``(x − p)·n``——`PenaltyNormalContact`的``g``去掉节点半径那一项。

    **仿射**，所以三次B样条对它逐位精确（模块docstring第二条）。
    """

    return sum(
        (point_mm[axis] - plane_point_mm[axis]) * unit_normal[axis] for axis in range(3)
    )


def sphere_distance_mm(point_mm: Vector3, centre_mm: Vector3, radius_mm: float) -> float:
    """``|x − c| − R``——`PenaltySphereContact`的``g``去掉节点半径那一项。"""

    return (
        math.sqrt(sum((point_mm[axis] - centre_mm[axis]) ** 2 for axis in range(3)))
        - radius_mm
    )


def cylinder_distance_mm(
    point_mm: Vector3, axis_point_mm: Vector3, unit_axis: Vector3, radius_mm: float
) -> float:
    """``ρ − R``——`PenaltyCylinderContact`的``g``去掉节点半径那一项。

    **无限长**：`PenaltyCylinderContact`那条"``|s| ≤ half_width``"的轴向硬切
    在这里不出现。理由是本模块判的是**插值对光滑场的逼近阶**，
    而硬切处场根本不光滑——把不光滑的东西塞进阶的判据里，量出来的不是插值的阶。
    轴向有限长那一档要进场，走的是把端面也烘进去（那时场在棱上只有C⁰），
    **不是把这一条改成有限长**。
    """

    delta = tuple(point_mm[axis] - axis_point_mm[axis] for axis in range(3))
    along = sum(delta[axis] * unit_axis[axis] for axis in range(3))
    radial = tuple(delta[axis] - along * unit_axis[axis] for axis in range(3))
    return math.sqrt(sum(component * component for component in radial)) - radius_mm


# --------------------------------------------------------------------------
# 三次B样条的权重：值、一阶、二阶
# --------------------------------------------------------------------------


def _weights(t: float) -> tuple[float, float, float, float]:
    """四个基函数在局部参数``t ∈ [0, 1)``处的值。和恒为1（单位分解）。"""

    s = 1.0 - t
    return (
        s * s * s / 6.0,
        (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0,
        (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
        t * t * t / 6.0,
    )


def _weights_d1(t: float) -> tuple[float, float, float, float]:
    """对``t``的一阶导。和恒为0——常函数的梯度是零，这条是可断言的。"""

    s = 1.0 - t
    return (
        -0.5 * s * s,
        (3.0 * t * t - 4.0 * t) / 2.0,
        (-3.0 * t * t + 2.0 * t + 1.0) / 2.0,
        0.5 * t * t,
    )


def _weights_d2(t: float) -> tuple[float, float, float, float]:
    """对``t``的二阶导。和恒为0，且**处处连续**——C²就是这里来的。"""

    return (1.0 - t, 3.0 * t - 2.0, -3.0 * t + 1.0, t)


_WEIGHT_TABLE = (_weights, _weights_d1, _weights_d2)


# --------------------------------------------------------------------------
# 窄带块稀疏存储
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedDistanceField:
    """规则栅格上的有符号距离场，**只存窄带**，按定长块哈希。

    栅格节点``(i, j, k)``的世界坐标是``origin_mm + (i, j, k)·spacing_mm``。
    块坐标是``(i >> block_log2, ...)``；块内偏移按``(di, dj, dk)``行主序。
    **块不存在 = 那一带没烘**，不是"距离为零"也不是"在体外"——
    见模块docstring"窄带外"那一节。

    ``blocks``是可变的``dict``（`frozen=True`只挡重新绑定字段）。
    **约定：构造之后不再改它**——`sample_narrow_band`是唯一的正常来路，
    它建完就不再持有引用。这条约定没有门守着，写在这里是因为它是承重的。
    """

    origin_mm: Vector3
    spacing_mm: float
    node_counts: tuple[int, int, int]
    band_mm: float
    blocks: dict[tuple[int, int, int], tuple[float, ...]] = dataclass_field(
        default_factory=dict
    )
    block_log2: int = DEFAULT_BLOCK_LOG2

    def __post_init__(self) -> None:
        if len(self.origin_mm) != 3 or not all(math.isfinite(v) for v in self.origin_mm):
            raise SignedDistanceFieldError(
                f"origin_mm must be a finite 3-vector: {self.origin_mm!r}"
            )
        if not (self.spacing_mm > 0.0 and math.isfinite(self.spacing_mm)):
            raise SignedDistanceFieldError(
                f"spacing_mm must be positive and finite: {self.spacing_mm!r}"
            )
        if len(self.node_counts) != 3 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < STENCIL_WIDTH
            for n in self.node_counts
        ):
            raise SignedDistanceFieldError(
                f"node_counts must be three ints >= {STENCIL_WIDTH} "
                f"(三次B样条每轴要4个节点): {self.node_counts!r}"
            )
        if not (self.band_mm > 0.0 and math.isfinite(self.band_mm)):
            raise SignedDistanceFieldError(
                f"band_mm must be positive and finite: {self.band_mm!r}"
            )
        if isinstance(self.block_log2, bool) or not isinstance(self.block_log2, int):
            raise SignedDistanceFieldError(f"block_log2 must be an int: {self.block_log2!r}")
        if not 1 <= self.block_log2 <= 6:
            raise SignedDistanceFieldError(
                f"block_log2 must be in [1, 6]: {self.block_log2!r}"
            )
        width = 1 << self.block_log2
        expected = width * width * width
        for coordinate, values in self.blocks.items():
            if len(values) != expected:
                raise SignedDistanceFieldError(
                    f"block {coordinate} holds {len(values)} samples, expected {expected}"
                )

    @property
    def block_width(self) -> int:
        return 1 << self.block_log2

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def stored_node_count(self) -> int:
        """实际存下来的节点数。**0074第5.1节那张MB表量的就是它。**"""

        width = self.block_width
        return len(self.blocks) * width * width * width

    @property
    def dense_node_count(self) -> int:
        """整个包围盒稠密存要多少节点——窄带省了多少，比这两个数即得。"""

        return self.node_counts[0] * self.node_counts[1] * self.node_counts[2]

    def node_position_mm(self, i: int, j: int, k: int) -> Vector3:
        h = self.spacing_mm
        return (
            self.origin_mm[0] + i * h,
            self.origin_mm[1] + j * h,
            self.origin_mm[2] + k * h,
        )

    def sample_at(self, i: int, j: int, k: int) -> float | None:
        """节点采样值；**块没烘时返回``None``**，不返回0。"""

        shift = self.block_log2
        block = self.blocks.get((i >> shift, j >> shift, k >> shift))
        if block is None:
            return None
        mask = self.block_width - 1
        width = self.block_width
        return block[((i & mask) * width + (j & mask)) * width + (k & mask)]

    def _corner(self, point_mm: Vector3) -> tuple[int, int, int, float, float, float]:
        """支撑左下角的节点下标与三个局部参数。**不校验是否在带内。**"""

        h = self.spacing_mm
        u = (point_mm[0] - self.origin_mm[0]) / h
        v = (point_mm[1] - self.origin_mm[1]) / h
        w = (point_mm[2] - self.origin_mm[2]) / h
        i = math.floor(u)
        j = math.floor(v)
        k = math.floor(w)
        return (i - 1, j - 1, k - 1, u - i, v - j, w - k)

    def contains_stencil(self, point_mm: Vector3) -> bool:
        """4×4×4支撑是否**整个**在已烘的块里。查询前先问它，就不会吃到异常。"""

        if not all(math.isfinite(component) for component in point_mm):
            return False
        i0, j0, k0, _, _, _ = self._corner(point_mm)
        for a in range(STENCIL_WIDTH):
            for b in range(STENCIL_WIDTH):
                for c in range(STENCIL_WIDTH):
                    if self.sample_at(i0 + a, j0 + b, k0 + c) is None:
                        return False
        return True

    def _gather(self, point_mm: Vector3) -> tuple[list[float], float, float, float]:
        if not all(math.isfinite(component) for component in point_mm):
            raise SignedDistanceFieldError(f"query point must be finite: {point_mm!r}")
        i0, j0, k0, tx, ty, tz = self._corner(point_mm)
        values: list[float] = []
        append = values.append
        for a in range(STENCIL_WIDTH):
            for b in range(STENCIL_WIDTH):
                for c in range(STENCIL_WIDTH):
                    sample = self.sample_at(i0 + a, j0 + b, k0 + c)
                    if sample is None:
                        raise SignedDistanceFieldError(
                            f"point {point_mm!r} falls outside the narrow band "
                            f"(node ({i0 + a}, {j0 + b}, {k0 + c}) was never baked). "
                            "**失败关闭而不是外推**：窄带外的'远在体外'与'深在体内'"
                            "在稀疏块表里长得一模一样，外推等于在'不接触'与"
                            "'接触力很大'之间猜一个，而猜错的那一半是静默的。"
                            "先问contains_stencil，或者把band_mm烘宽一点。"
                        )
                    append(sample)
        return values, tx, ty, tz

    def evaluate(
        self,
        point_mm: Vector3,
        *,
        need_gradient: bool = False,
        need_hessian: bool = False,
    ) -> tuple[float, Vector3 | None, tuple[Vector3, Vector3, Vector3] | None]:
        """一次取``g``、``∇g``、``∇²g``。**融合路径与单独取值逐字节相同**。

        做到逐字节的方式是"同一串运算同一个次序"——值那一支无论要不要梯度
        都走同一段代码，不是算完再比（`PenaltyNormalContact.quantities`同一条纪律）。

        ``∇g``无量纲、``∇²g``单位1/mm。Hessian**对称**（混合偏导只算一次填两处），
        这不是省时间：算两次再让它们相差一个ulp，会让Newton的对称性假设静默失效。
        """

        values, tx, ty, tz = self._gather(point_mm)
        top = 2 if need_hessian else (1 if need_gradient else 0)

        wx = [_WEIGHT_TABLE[order](tx) for order in range(top + 1)]
        wy = [_WEIGHT_TABLE[order](ty) for order in range(top + 1)]
        wz = [_WEIGHT_TABLE[order](tz) for order in range(top + 1)]

        #: 可分离约化：先塌缩z，再塌缩y，最后塌缩x。
        #: 直接三重循环是64×10次乘加；分离之后值那一支只要84次。
        stage_a: list[list[list[float]]] = []
        for order in range(top + 1):
            weight = wz[order]
            plane = [
                [
                    (
                        values[(a * STENCIL_WIDTH + b) * STENCIL_WIDTH] * weight[0]
                        + values[(a * STENCIL_WIDTH + b) * STENCIL_WIDTH + 1] * weight[1]
                        + values[(a * STENCIL_WIDTH + b) * STENCIL_WIDTH + 2] * weight[2]
                        + values[(a * STENCIL_WIDTH + b) * STENCIL_WIDTH + 3] * weight[3]
                    )
                    for b in range(STENCIL_WIDTH)
                ]
                for a in range(STENCIL_WIDTH)
            ]
            stage_a.append(plane)

        def _collapse(order_y: int, order_z: int) -> list[float]:
            weight = wy[order_y]
            plane = stage_a[order_z]
            return [
                plane[a][0] * weight[0]
                + plane[a][1] * weight[1]
                + plane[a][2] * weight[2]
                + plane[a][3] * weight[3]
                for a in range(STENCIL_WIDTH)
            ]

        def _finish(order_x: int, line: list[float]) -> float:
            weight = wx[order_x]
            return (
                line[0] * weight[0]
                + line[1] * weight[1]
                + line[2] * weight[2]
                + line[3] * weight[3]
            )

        line_00 = _collapse(0, 0)
        value = _finish(0, line_00)
        if top == 0:
            return (value, None, None)

        h = self.spacing_mm
        line_10 = _collapse(1, 0)
        line_01 = _collapse(0, 1)
        gradient = (
            _finish(1, line_00) / h,
            _finish(0, line_10) / h,
            _finish(0, line_01) / h,
        )
        if top == 1:
            return (value, gradient, None)

        h2 = h * h
        line_20 = _collapse(2, 0)
        line_02 = _collapse(0, 2)
        line_11 = _collapse(1, 1)
        xx = _finish(2, line_00) / h2
        yy = _finish(0, line_20) / h2
        zz = _finish(0, line_02) / h2
        xy = _finish(1, line_10) / h2
        xz = _finish(1, line_01) / h2
        yz = _finish(0, line_11) / h2
        hessian = ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))
        return (value, gradient, hessian)

    def value_mm(self, point_mm: Vector3) -> float:
        return self.evaluate(point_mm)[0]

    def gradient(self, point_mm: Vector3) -> Vector3:
        return self.evaluate(point_mm, need_gradient=True)[1]  # type: ignore[return-value]

    def hessian_per_mm(self, point_mm: Vector3) -> tuple[Vector3, Vector3, Vector3]:
        return self.evaluate(point_mm, need_gradient=True, need_hessian=True)[2]  # type: ignore[return-value]


def sample_narrow_band(
    distance_mm: Callable[[Vector3], float],
    *,
    origin_mm: Vector3,
    spacing_mm: float,
    node_counts: tuple[int, int, int],
    band_mm: float,
    block_log2: int = DEFAULT_BLOCK_LOG2,
) -> SignedDistanceField:
    """把一个可调用的距离函数烘成窄带块稀疏场。

    **``distance_mm``必须是真的有符号距离**，即1-Lipschitz
    （``|φ(x) − φ(y)| ≤ |x − y|``）。本函数用这条性质剔块：
    块中心的``|φ|``大于``band_mm + 半对角线``时，块内不可能有节点落进带里。
    传一个非距离的水平集函数进来，剔块会**静默剔错**——
    这条前提写在这里，没有门守得住它（要守就得逐节点采样，那正是剔块要省掉的）。

    带宽的下限：``band_mm``要**至少**覆盖``2·spacing_mm·√3``，
    否则带内某些点的4×4×4支撑仍然缺节点。本函数按这条校验，不合格即拒——
    烘出一个"带里也查不了"的场是纯粹的浪费。
    """

    field = SignedDistanceField(
        origin_mm=origin_mm,
        spacing_mm=spacing_mm,
        node_counts=node_counts,
        band_mm=band_mm,
        blocks={},
        block_log2=block_log2,
    )
    minimum_band = 2.0 * spacing_mm * math.sqrt(3.0)
    if band_mm < minimum_band:
        raise SignedDistanceFieldError(
            f"band_mm = {band_mm!r} is thinner than {minimum_band!r} = "
            "2·spacing·√3，带里的点也会缺支撑节点——烘一个查不了的场是纯粹的浪费"
        )

    width = 1 << block_log2
    blocks: dict[tuple[int, int, int], tuple[float, ...]] = {}
    half_diagonal = 0.5 * math.sqrt(3.0) * (width - 1) * spacing_mm
    counts = tuple((count + width - 1) // width for count in node_counts)

    for bi in range(counts[0]):
        for bj in range(counts[1]):
            for bk in range(counts[2]):
                base = (bi * width, bj * width, bk * width)
                offset = (width - 1) / 2.0
                centre = (
                    origin_mm[0] + (base[0] + offset) * spacing_mm,
                    origin_mm[1] + (base[1] + offset) * spacing_mm,
                    origin_mm[2] + (base[2] + offset) * spacing_mm,
                )
                if abs(distance_mm(centre)) > band_mm + half_diagonal:
                    continue
                values: list[float] = []
                keep = False
                for di in range(width):
                    for dj in range(width):
                        for dk in range(width):
                            sample = distance_mm(
                                field.node_position_mm(
                                    base[0] + di, base[1] + dj, base[2] + dk
                                )
                            )
                            values.append(sample)
                            if abs(sample) <= band_mm:
                                keep = True
                if keep:
                    blocks[(bi, bj, bk)] = tuple(values)

    return SignedDistanceField(
        origin_mm=field.origin_mm,
        spacing_mm=field.spacing_mm,
        node_counts=field.node_counts,
        band_mm=field.band_mm,
        blocks=blocks,
        block_log2=block_log2,
    )


# --------------------------------------------------------------------------
# 接触项：协议照`PenaltyNormalContact`一字不差
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PenaltySignedDistanceField:
    """罚函数式法向接触，间隙由**距离场**给：``U = Σ ½·k·g²``（仅``g < 0``），N·mm。

    ``g = φ(x) − r``，``φ``是场、``r``是节点代表的球半径。
    与`PenaltyNormalContact`的差别**只有``φ``怎么来**：
    那里是``(x − p)·n``一条闭式，这里是一次B样条查询。

    ## 半径仍然是**显式参数**

    照抄`PenaltyNormalContact`那条：数学上把场整体偏移``r``是等价的，
    但那条路在三球金字塔上绊过一次（底球被判成悬空、切线刚度出现零模态）。
    **让半径出现在调用点，那个误会就没有发生的余地。** 质点写``0.0``。

    ## Hessian有两块，第二块只有场能给

        H = k·(∇g ⊗ ∇g) + k·g·∇²g

    半空间那一族第二块恒为零（``∇g``是常量），所以它的Hessian只有一块。
    **一般的场``∇²g ≠ 0``**——曲率进来了，而那正是"真实件"与"一个平面"的差别。
    ``g < 0``时``k·g·∇²g``是负的（凸面上），与`PenaltySphereContact`那块
    横向softening同源：**漏掉它梯度照样对、平衡点照样对，
    只有收敛速度与稳定性判据会变**，所以它必须被有限差分单独验一次。

    ## 光滑性：场是C²，**但U仍然只是C¹**

    B样条把``g``做到了C²（这是`solve.py`适用域要的那一档），
    但``U``在``g = 0``处二阶导仍然从``k``跳到``0``——那是**罚势自己的**性质，
    与插值阶次无关。0050第四节登记的那条脆点原样成立，一条也没多、一条也没少。
    **不要把"场是C²"读成"U是C²"。**
    """

    #: 场。**一项配一个场**——多个障碍物就是多个项，不在项内做场的并集
    #: （并集的``min``不可微，那正好会破掉上面刚立的C²）。
    field: SignedDistanceField
    #: (节点索引, 罚刚度N/mm, 球半径mm)。**半径必填**，质点写``0.0``。
    contacts: tuple[tuple[int, float, float], ...]
    name: str = "sdf_contact"
    kind: ClassVar[Literal["potential"]] = POTENTIAL

    def __post_init__(self) -> None:
        if not self.contacts:
            raise SignedDistanceFieldError("sdf_contact needs at least one node")
        for node, stiffness, radius in self.contacts:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise SignedDistanceFieldError(
                    f"contact node index must be a nonnegative int: {node!r}"
                )
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise SignedDistanceFieldError(
                    f"penalty stiffness must be positive: {stiffness!r}"
                )
            if radius < 0.0 or not math.isfinite(radius):
                raise SignedDistanceFieldError(
                    f"contact radius must be finite and nonnegative: {radius!r}"
                )

    def node_index_bound(self) -> int:
        return max(node for node, _, _ in self.contacts) + 1

    def _point(self, vector: tuple[float, ...], node: int) -> Vector3:
        base = 3 * node
        return (vector[base], vector[base + 1], vector[base + 2])

    def gap_mm(self, state: State) -> tuple[float, ...]:
        """每个接触点的间隙``g``。``> 0``分离、``< 0``穿透。"""

        return tuple(
            self.field.value_mm(self._point(state.vector, node)) - radius
            for node, _stiffness, radius in self.contacts
        )

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, stiffness, radius in self.contacts:
            gap = self.field.value_mm(self._point(state.vector, node)) - radius
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, stiffness, radius in self.contacts:
            value, slope, _ = self.field.evaluate(
                self._point(state.vector, node), need_gradient=True
            )
            gap = value - radius
            if gap < 0.0:
                force = stiffness * gap
                base = 3 * node
                for axis in range(3):
                    result[base + axis] += force * slope[axis]  # type: ignore[index]
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        size = len(state.vector)
        result = [[0.0] * size for _ in range(size)]
        for row, column, value in self.hessian_entries(state, context):
            result[row][column] += value
        return tuple(tuple(row) for row in result)

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        """``k·(∇g⊗∇g) + k·g·∇²g``，仅活动接触。**分离的接触一个非零项都不出。**"""

        entries: list[tuple[int, int, float]] = []
        for node, stiffness, radius in self.contacts:
            value, slope, curvature = self.field.evaluate(
                self._point(state.vector, node), need_gradient=True, need_hessian=True
            )
            gap = value - radius
            if gap >= 0.0:
                continue
            base = 3 * node
            bend = stiffness * gap
            for a in range(3):
                for b in range(3):
                    entries.append(
                        (
                            base + a,
                            base + b,
                            stiffness * slope[a] * slope[b]  # type: ignore[index]
                            + bend * curvature[a][b],  # type: ignore[index]
                        )
                    )
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。

        与`PenaltyNormalContact`同一条做法：**同一串运算同一个次序**，
        不是算完再比。这里额外多守一条——``field.evaluate``的值那一支
        无论要不要梯度都走同一段代码，所以``value``跨三条路逐位相同。
        """

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, stiffness, radius in self.contacts:
            value, slope, _ = self.field.evaluate(
                self._point(vector, node), need_gradient=need_gradient
            )
            gap = value - radius
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    force = stiffness * gap
                    base = 3 * node
                    for axis in range(3):
                        gradient[base + axis] += force * slope[axis]  # type: ignore[index]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def normal_force_n(self, state: State) -> tuple[float, ...]:
        """每个接触点上的法向力大小``N = k·|g|``（分离时为0）。

        **这是本项唯一精确的输出**——照`PenaltyNormalContact`那条：
        平衡时它等于理论法向力，与罚刚度无关。摩擦锥要用的正是它。

        **但这里多一层误差**：``g``本身带插值误差``O(h²)``，
        所以"与k无关"这句话在场这一档说的是"与k无关、与h二阶相关"。
        这条差别不能省，它正是0074第二节第4条末段那句
        "误差不是随机的、在高曲率处系统性偏保守或偏松"的量化出口。
        """

        return tuple(
            stiffness * -gap
            if (gap := self.field.value_mm(self._point(state.vector, node)) - radius) < 0.0
            else 0.0
            for node, stiffness, radius in self.contacts
        )


__all__ = [
    "DEFAULT_BLOCK_LOG2",
    "STENCIL_WIDTH",
    "PenaltySignedDistanceField",
    "SignedDistanceField",
    "SignedDistanceFieldError",
    "cylinder_distance_mm",
    "half_space_distance_mm",
    "sample_narrow_band",
    "sphere_distance_mm",
]
