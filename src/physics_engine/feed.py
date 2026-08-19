"""喂料前沿——带材一段一段喂进来，而状态布局**定长**（力学域，决策0062丙1）。

守能力位S5.4（喂料与移动前沿）。

## 它撞上的那条承重不变量

S5.4的`missing`原文：**"十几匝不是一次装配出来的，是一匝一匝喂出来的——
自由度数在过程中变"**，并点名"它与0033/0050那条'状态布局固定而数量随构型变'
是同一类问题"。

那条不变量是0050的承重条款：**布局定长 ⟹ 指纹跨步不变 ⟹ `fingerprint()`
的语义一个字不用改**。喂料如果真的往状态里加自由度，这条当场破。

## 解法直接复用0050已经裁过的形制

0050当年对锚点的裁法是：**三条候选路的分母都取错了**——它们假定锚点数由
**活动**接触集决定，而接触对是**声明**出来的。按声明的对分槽，布局就是定长的，
**活动与否变成向量里的一个值，不是一次布局变更**。

喂料是同一道题：节点数不由"已经喂进来多少"决定，由**这一卷总共要喂多少**决定。
于是：

* **节点预算在声明期定死**（``node_budget``），布局定长；
* ``fed_count``是**当前已喂进来的节点数**，它是一个值不是一次布局变更；
* 未喂的节点被**钉住**（进``fixed_indices``），一个自由度都不参与求解。

代价与0050那条逐字相同：**自由度按声明算而不是按活动算**。
它可接受的理由也相同——**总匝数是使用者控制的量**，
本仓从不做"带材可以无限长"这种声明。

## 未喂的节点停在哪：**停在喂料口，不是停在原点**

停在原点会让`AxialStretch`的第一条未喂边长度是``|x_喂料口 − 0|``——
一个巨大的、假的伸长量。停在喂料口则未喂边长度恒为零，
而**零长度边在`AxialStretch`里是失败关闭的**（方向未定义）。

所以未喂的节点停在**喂料口沿进给方向的静止长度间隔上**：
未喂边的长度恰是静止长度、伸长量恰为零、能量恰为零。
**"贡献恰为零"必须是构造出来的，不是靠一个if跳过的**——
跳过会让能量项的求和次序随``fed_count``变，而求和次序是形制（spec/12第3.3节）。

## 本模块不做什么

* **不做匝数与半径生长**：那是`drives.SpoolTension`的``turns``。
  **2026-08-18（决策0093）补上了那根接线**：`winding.WindingFront`吃一个
  `FeedFront`，把"喂进来了几段"换成"盘上几匝、半径多大"，
  而`drives`要的正是那个匝数。**本模块一个字节没动**——
  接线住在`winding`那一侧，因为它要的是卷绕的堆积律，不是喂料的布局；
* **不做自接触**：新喂的一匝压在已绕的匝上是S5.2，要网格/连续体窄相；
* **不做材料注入**（WDS `research/05`第三节称"当前最大的单项缺口"的那一条）：
  那是**带材从轮面上流过**、边界随材料移动，与本模块"往前接长度"不是一回事。
  本模块是**拉格朗日**的：每个节点始终是同一块材料。
  **0093复核过这一条并且没有改它**：欧拉式材料注入是一次**新裁决**
  （要裁"节点是不是物质点"，而0050／0062两条承重条款都建立在"是"上面），
  不是一次接线。裁不动就留空——登记成GAP，触发条件在0093第五节。

## 材料守恒记在**整数段**上（决策0093）

``fed_material_length_mm``给的是浮点长度，而``(k−1)·h``与``(k−2)·h + h``
在浮点上**不是同一个数**（``h = 0.1``、``k = 4``时差一个ulp）。
于是"送进去多少＝盘上多了多少"这条守恒**不写在长度上**，
写在``fed_segment_count``给的整数段数上——`winding.WindingFront`那一侧
判的是``喂进来的段 == 跨距里的段 + 盘上的段``，整数、零容差、
与浮点求和次序无关。实测：``h = 0.1``时1392个喂料档里**373档**
浮点长度逐位不守恒，而整数恒等式**处处**成立。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.state import StateLayout


class FeedError(ValueError):
    """喂料前沿的一切失败关闭。"""


@dataclass(frozen=True)
class FeedFront:
    """喂料前沿的声明：节点预算、静止段长、喂料口位置与进给方向。

    ``node_budget``是**这一卷总共要喂多少个节点**，声明期定死，布局据它定长。
    ``fed_count``是当前已喂进来的节点数，随过程变——**它是一个值，不是一次布局变更**。
    """

    node_budget: int
    rest_length_mm: float
    #: 喂料口：未喂的节点停在这里沿``direction``排开。
    inlet_mm: tuple[float, float, float]
    #: 进给方向（单位矢量）：未喂节点沿它以``rest_length_mm``为间隔停放。
    direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.node_budget, bool) or not isinstance(self.node_budget, int):
            raise FeedError(f"node_budget must be an int: {self.node_budget!r}")
        if self.node_budget < 2:
            raise FeedError(
                f"node_budget must be at least 2: {self.node_budget!r} —— "
                "一个节点连一条边都构不成，那不是一段带材"
            )
        if not (self.rest_length_mm > 0.0 and math.isfinite(self.rest_length_mm)):
            raise FeedError(f"rest_length_mm must be positive: {self.rest_length_mm!r}")
        for name in ("inlet_mm", "direction"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(item) for item in value):
                raise FeedError(f"{name} must be a finite 3-vector: {value!r}")
        norm = math.sqrt(sum(item * item for item in self.direction))
        if abs(norm - 1.0) > 1.0e-12:
            raise FeedError(
                f"direction must be a unit vector (|d| = {norm!r}) —— "
                "不归一化会把静止段长悄悄乘上|d|，而调用方以为自己给的是长度"
            )

    def assert_fed_count(self, fed_count: int) -> None:
        """``fed_count``必须在``[2, node_budget]``内。

        **下界是2不是0**：喂料是一个已经在走的过程，起手至少有一条边。
        上界是预算——超过它就是布局不够，而那必须在声明期发现，不是在第900匝发现。
        """

        if isinstance(fed_count, bool) or not isinstance(fed_count, int):
            raise FeedError(f"fed_count must be an int: {fed_count!r}")
        if not (2 <= fed_count <= self.node_budget):
            raise FeedError(
                f"fed_count must be in [2, {self.node_budget}]: {fed_count!r} —— "
                "超过预算说明这一卷的节点预算在声明期就估少了"
            )

    def parked_position_mm(self, node: int) -> tuple[float, float, float]:
        """未喂节点的停放位置：喂料口沿进给方向退``k``个静止段长。

        **退不是进**：未喂的材料还在喂料口**后面**。停在前面会让它与已喂的
        那一段重叠，而重叠的节点在接触检测里是一对零距离的候选。
        """

        offset = -self.rest_length_mm * node
        return tuple(
            self.inlet_mm[axis] + offset * self.direction[axis] for axis in range(3)
        )  # type: ignore[return-value]

    def initial_positions_mm(self, fed_positions_mm: tuple[float, ...]) -> tuple[float, ...]:
        """把已喂节点的位置与未喂节点的停放位置拼成完整的节点块。

        ``fed_positions_mm``是已喂那一段的展平坐标，长度必须是``3·fed_count``。
        """

        if len(fed_positions_mm) % 3 != 0:
            raise FeedError(
                f"fed positions must be a flat xyz triple list: {len(fed_positions_mm)}"
            )
        fed_count = len(fed_positions_mm) // 3
        self.assert_fed_count(fed_count)
        parked: list[float] = []
        for index in range(fed_count, self.node_budget):
            parked.extend(self.parked_position_mm(index - fed_count + 1))
        return (*fed_positions_mm, *parked)

    def edges(self, axial_stiffness_n: float) -> tuple[tuple[int, int, float, float], ...]:
        """**全预算**的拉伸边——已喂与未喂一视同仁。

        未喂边的两端停在相距恰好``rest_length_mm``的位置上，伸长量恰为零、
        能量恰为零、梯度恰为零。**"贡献恰为零"是构造出来的，不是跳过来的**：
        跳过会让求和次序随``fed_count``变，而求和次序是形制（spec/12第3.3节）。
        """

        if not (axial_stiffness_n > 0.0 and math.isfinite(axial_stiffness_n)):
            raise FeedError(f"axial stiffness must be positive: {axial_stiffness_n!r}")
        return tuple(
            (index, index + 1, self.rest_length_mm, axial_stiffness_n)
            for index in range(self.node_budget - 1)
        )

    def parked_fixed_indices(self, fed_count: int) -> frozenset[int]:
        """未喂节点的全部标量自由度——**它们必须被钉住**。

        不钉的话未喂那一段是一条不受任何约束的自由链，
        `solve_equilibrium`当场报"欠约束即失败关闭"。
        """

        self.assert_fed_count(fed_count)
        return frozenset(
            3 * node + axis
            for node in range(fed_count, self.node_budget)
            for axis in range(3)
        )

    def fed_segment_count(self, fed_count: int) -> int:
        """已经离开料卷的**段数**``fed_count − 1``——**整数**。

        它与`fed_material_length_mm`是同一件事的两种记法，
        而**守恒记在这一种上**（见模块文档末节）：段数是整数，
        长度是段数乘静止段长，于是守恒不依赖``rest_length_mm``的二进制形状。
        """

        self.assert_fed_count(fed_count)
        return fed_count - 1

    def fed_material_length_mm(self, fed_count: int) -> float:
        """已喂进来的材料长度``(fed_count − 1)·rest_length``。

        它随喂料**线性**增长——这条是本模块唯一的"物理量"，
        也是判据里拿来与卷径生长对账的那个量。
        """

        self.assert_fed_count(fed_count)
        return (fed_count - 1) * self.rest_length_mm


def assert_layout_matches_budget(layout: StateLayout, front: FeedFront) -> None:
    """布局的节点块必须恰好装得下预算——**多一个少一个都失败关闭**。

    这条是本模块与0050的`ContactLayout.assert_matches_context`同一条纪律：
    **两个地方各说各的节点数，是本仓已经吃过三次亏的那种洞**（plans/09教训一）。
    """

    expected = 3 * front.node_budget
    #: ``node_dof_count is None``的语义是"整份布局都是节点块"（`state.py`那条字段的
    #: 默认值），**不是"没声明所以不校验"**。把``None``当不匹配会让所有不带锚点槽的
    #: 布局都过不去；当"跳过校验"则又回到"两处各说各的节点数"。取``dof_count``兜底。
    actual = layout.dof_count if layout.node_dof_count is None else layout.node_dof_count
    if actual != expected:
        raise FeedError(
            f"布局的节点块有{actual}个自由度，"
            f"而喂料预算要{expected}个（{front.node_budget}个节点）——"
            "**两处各说各的节点数**是plans/09教训一记的那种洞"
        )


__all__ = [
    "FeedError",
    "FeedFront",
    "assert_layout_matches_budget",
]
