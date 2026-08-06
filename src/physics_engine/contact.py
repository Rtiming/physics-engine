"""接触层——决策0050的形制落地（第一片：**锚点布局**，力学域）。

本片**不算任何物理**。它只回答一个问题：粘着锚点放在状态的什么位置。
理由见0050第一节——接触的形制一旦定错后面全要返工，
而锚点的位置是形制里最先被别的东西压上的那一块。

## 0033那道题，以及为什么这里不是三条路里的任何一条

0033裁过：粘着锚点**是真历史**（"这一点现在粘着还是滑动"不能从当前位形算出来，
只能从历史知道），必须进状态并随状态被复现。
0043随后实测确认`StateLayout`装不下它，并列了三条路，**代价都很硬**：
每种活动集一份布局（跨步守恒断言全废）、可变长段（`fingerprint()`语义要重定义，
而那是0019的承重条款）、锚点留状态外（0033已否）。

**三条路的分母都取错了**：它们假定锚点数由**活动**接触集决定。
而接触对是**声明**出来的——`scene.ContactPair`是冻结、有序的，声明期就定死。

**按声明的对分槽，布局就是定长的**：指纹跨步不变、`fingerprint()`语义一个字不用改、
跨步守恒量断言继续成立。**活动与否变成向量里的一个值，不是一次布局变更。**

代价如实写在0050第一节：自由度按**声明**的对数算而不是按活动的对数算。
它是可接受的，因为`declare_contact_between`本来就是显式声明——
**声明的对数是使用者控制的量**，本仓从不做"任意两个体都可能碰"的全局接触。

## 声明从哪来：今天由调用方给，**不由本模块去问`Scene`**

`FinalizedScene.contact_pairs`是**将来**的来源，但本片不去接它。
理由是三前提第二条：把场景里的"体"映射到力学状态里的"节点"是另一个未解的问题
（`AssembledBody`是位姿+几何，不是自由度），**现在接等于替一个还不存在的
消费方预支一套映射**。

**触发条件**：第一个既走`Scene`装配又走能量求解的案例出现时，那时映射的形状才清楚。
本模块只要求调用方给出**已冻结、有序**的声明——那是0050真正依赖的性质，
而不是"声明必须来自`Scene`"。

## 本片顺带关掉的一个洞

`energies.resolve_node_count`（本轮上一片）只看得见``len(vector)``，
**看不见"布局里哪一段是节点块"**——于是一份声明了3个质量而布局只有2个节点的
上下文，重力仍会落到锚点上。那条洞当时如实登记了，触发条件写的正是
"**锚点布局构造器进仓时把边界带上来**"。

`ContactLayout`知道边界（它就是造边界的那个东西），
故`assert_matches_context`在这里把第二次比对补上。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.energies import EnergyContext, Matrix, Vector
from physics_engine.state import State, StateField, StateLayout

#: 每对声明能分到几个锚点槽，按**形状族**定（0050第一节）。
#: 今天窄相只有球/胶囊族（`collision.py`第一片明写圆柱/盒/网格"等下一片，不冒充"），
#: 该族的最近点是唯一的，故为1。
#:
#: **盒/网格族进来时它会>1，那时必须在声明期按形状族定，不许由求解器临时决定**——
#: 求解器临时决定就等于布局随构型变，那正是本模块要避免的。
MAX_POINTS_PER_PAIR_SPHERE_CAPSULE = 1

#: 一个锚点槽占几个标量：活动标志1 + 锚点3 + 粘/滑判别1。
SLOT_WIDTH = 5

#: 法向必须是单位矢量的容差。取1e-12：归一化误差进的是刚度，
#: 而刚度是使用者声明的量——悄悄改它比拒收更坏。
NORMAL_UNIT_TOLERANCE = 1e-12

#: `regime`字段的取值。**用数而不用字符串**，因为状态是一维浮点向量
#: （spec/12第2.2节：显式数组不是对象图）。
REGIME_SEPARATED = 0.0
REGIME_STICK = 1.0
REGIME_SLIP = 2.0


class ContactError(ValueError):
    """接触层的一切失败关闭。"""


@dataclass(frozen=True)
class ContactDeclaration:
    """一对**被声明要算接触**的体，以及它分到几个锚点槽。

    ``pair_id``只用于报错与可读性，**不进布局指纹**——指纹描述的是次序与宽度
    （见`StateLayout.to_document`），改名字不该让既有产物的指纹全变一遍
    （0001三前提第三条）。
    """

    pair_id: str
    max_points: int = MAX_POINTS_PER_PAIR_SPHERE_CAPSULE

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise ContactError("contact declaration needs a non-empty pair_id")
        if isinstance(self.max_points, bool) or not isinstance(self.max_points, int):
            raise ContactError(f"max_points must be an int: {self.max_points!r}")
        if self.max_points < 1:
            raise ContactError(
                f"{self.pair_id}: max_points must be at least 1 — "
                "声明了一对却不给槽位，等于声明没发生"
            )


@dataclass(frozen=True)
class ContactSlot:
    """一个锚点槽在布局里的位置。``base``是它第一个标量的下标。"""

    pair_id: str
    point_index: int
    base: int

    @property
    def active_index(self) -> int:
        return self.base

    @property
    def anchor_base(self) -> int:
        return self.base + 1

    @property
    def regime_index(self) -> int:
        return self.base + 4


@dataclass(frozen=True)
class ContactLayout:
    """节点块 + 锚点块，**外加节点块的边界**。

    边界（``node_count``）是本类存在的主要理由之一：没有它，
    "这条向量的前多少个数是节点位置"就只能靠约定，
    而`energies.resolve_node_count`那条洞正是这么来的。
    """

    layout: StateLayout
    node_count: int
    slots: tuple[ContactSlot, ...]

    @property
    def node_dof_count(self) -> int:
        return 3 * self.node_count

    def slot_of(self, pair_id: str, point_index: int = 0) -> ContactSlot:
        for slot in self.slots:
            if slot.pair_id == pair_id and slot.point_index == point_index:
                return slot
        raise ContactError(f"no contact slot for {pair_id!r} point {point_index}")

    def assert_matches_context(self, context: EnergyContext) -> None:
        """上下文的质量表必须与**本布局的节点块**逐个对上。

        这是`energies.resolve_node_count`登记的那条洞的第二次比对。
        那里只看得见向量长度，所以一份"声明3个质量而布局只有2个节点"的上下文
        在那里判不出来——**而重力会因此落到锚点上**。

        这里判得出来，因为本类**就是造边界的那个东西**。
        """

        declared = len(context.node_masses_kg)
        if declared != self.node_count:
            raise ContactError(
                f"context declares {declared} node masses but the layout's node "
                f"block holds {self.node_count} nodes — "
                "对不上时重力会落到锚点自由度上（决策0049第八节那条洞）"
            )

    def initial_vector(self, node_positions_mm: tuple[float, ...]) -> tuple[float, ...]:
        """节点位置 + **全部槽位置零**的初始向量。

        槽位出生即``REGIME_SEPARATED``、锚点为原点、活动标志为0：
        **一个还没碰过的接触没有历史**，而`REGIME_SEPARATED`恰好是0.0，
        所以"全零"就是正确的出生态，不需要额外的初始化步骤。
        """

        if len(node_positions_mm) != self.node_dof_count:
            raise ContactError(
                f"expected {self.node_dof_count} node scalars, "
                f"got {len(node_positions_mm)}"
            )
        return tuple(node_positions_mm) + (0.0,) * (len(self.slots) * SLOT_WIDTH)


def build_contact_layout(
    *,
    layout_id: str,
    node_count: int,
    declarations: tuple[ContactDeclaration, ...],
) -> ContactLayout:
    """按**声明**的接触对造定长布局。次序即声明次序。

    次序留住的理由与`scene.ContactPair`那条逐字相同：接触对清单要交给求解器
    按序处理，而集合的遍历次序随``PYTHONHASHSEED``变——
    **同一份声明两次跑出不同次序，接触结果就不可复现**（轴3规则5）。
    这里更硬一层：次序还决定了每个槽落在向量的哪一格，
    **次序变了梯度与Hessian的索引全错，而多数测试不会发现**（spec/12第2.2节）。
    """

    if node_count < 1:
        raise ContactError("a contact layout needs at least one node")
    identifiers = [declaration.pair_id for declaration in declarations]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates:
        raise ContactError(
            f"duplicate contact pair_id: {duplicates} — "
            "重名意味着槽位指向谁说不清，而槽位是历史的住处"
        )

    fields: list[StateField] = [
        StateField(f"node{index}_{axis}_mm", 1)
        for index in range(node_count)
        for axis in ("x", "y", "z")
    ]
    slots: list[ContactSlot] = []
    base = 3 * node_count
    for declaration in declarations:
        for point in range(declaration.max_points):
            tag = f"contact_{declaration.pair_id}_{point}"
            fields.extend(
                (
                    StateField(f"{tag}_active", 1, is_history=True, is_dimensionless=True),
                    StateField(f"{tag}_anchor_mm", 3, is_history=True),
                    StateField(f"{tag}_regime", 1, is_history=True, is_dimensionless=True),
                )
            )
            slots.append(
                ContactSlot(pair_id=declaration.pair_id, point_index=point, base=base)
            )
            base += SLOT_WIDTH

    return ContactLayout(
        layout=StateLayout(layout_id=layout_id, fields=tuple(fields)),
        node_count=node_count,
        slots=tuple(slots),
    )




@dataclass(frozen=True)
class PenaltyNormalContact:
    """罚函数式法向接触（**半空间**）：``U = Σ ½·k·g²``（仅``g < 0``），单位N·mm。

    ``g = (x − p)·n``是**间隙**：``n``是半空间的外法向（指向节点该待的那一侧），
    ``p``是面上一点。``g > 0``分离、``g < 0``穿透。

    ## 量纲（本仓已因单位吃过两次亏，所以这里逐个写出来）

    ``k``是N/mm、``g``是mm，故``½kg²``是``N/mm · mm² = N·mm``——
    **直接就是本仓的能量单位，不需要`MM_PER_M`**。
    与`PointLoad`同理、与`UniformGravity`相反（后者拿的是kg与mm/s²）。
    **量纲是算出来的，不是照抄相邻代码抄出来的。**

    ## 这个模型给对了什么、给错了什么（0050第二节的代价，写在实现里）

    平衡时``k·δ = N_理论``，于是：

    * **法向力是精确的**，与``k``无关——``δ = N/k``，``N = k·δ = N``恒成立；
    * **穿透不为零**，``δ = N/k``是``O(1/k)``。**这是模型不是缺陷**，
      但它必须被声明：刚度是**输入**，不是代码里的魔数。

    换句话说：**位置有``O(1/k)``的误差，力没有误差。**
    这条性质决定了判据该判什么——`cases/`里的门判力与阈值，不判位置。

    ## 光滑性：``C¹``而**不是**``C²``

    ``U``在``g = 0``处值与一阶导都连续（``U = ½kg²``、``U' = kg``，两者在
    ``g → 0⁻``都趋于0），**但二阶导从``k``跳到``0``**。

    后果写明：**牛顿法的残差连续、切线刚度不连续**。活动集在迭代中翻转时，
    线搜索可能在那一步失效——0050第四节登记的正是这条，
    与0029第八节那条强非线性全局化的脆点同源。
    """

    name: str = "normal_contact"
    #: 半空间：(节点索引, 面上一点mm, 外法向单位矢量, 罚刚度N/mm)
    planes: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float], float], ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.planes:
            raise ContactError("normal_contact needs at least one half-space")
        for node, point, normal, stiffness in self.planes:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"contact node index must be a nonnegative int: {node!r}")
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"plane point must be a finite 3-vector: {point!r}")
            if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
                raise ContactError(f"plane normal must be a finite 3-vector: {normal!r}")
            norm = math.sqrt(sum(component * component for component in normal))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(
                    f"plane normal must be a unit vector (|n| = {norm!r}) — "
                    "不归一化就等于把刚度悄悄乘上|n|²，而调用方以为自己给的是k"
                )
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _ in self.planes) + 1

    @staticmethod
    def _gap_mm(
        vector: tuple[float, ...],
        node: int,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
    ) -> float:
        base = 3 * node
        return sum(
            (vector[base + axis] - point[axis]) * normal[axis] for axis in range(3)
        )

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, point, normal, stiffness in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal)
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, point, normal, stiffness in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal)
            if gap < 0.0:
                force = stiffness * gap
                base = 3 * node
                for axis in range(3):
                    result[base + axis] += force * normal[axis]
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
        """``k·(n ⊗ n)``，仅活动接触。**分离的接触一个非零项都不出**。"""

        entries: list[tuple[int, int, float]] = []
        for node, point, normal, stiffness in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal)
            if gap < 0.0:
                base = 3 * node
                for a in range(3):
                    for b in range(3):
                        entries.append(
                            (base + a, base + b, stiffness * normal[a] * normal[b])
                        )
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。

        这里做到逐字节的方式是"同一串运算同一个次序"，不是"算完再比"——
        两条路各写一遍求和次序，迟早会在某个构型上差一个ulp。
        """

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, point, normal, stiffness in self.planes:
            gap = self._gap_mm(vector, node, point, normal)
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    force = stiffness * gap
                    base = 3 * node
                    for axis in range(3):
                        gradient[base + axis] += force * normal[axis]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def normal_force_n(self, state: State) -> tuple[float, ...]:
        """每个半空间上的法向力大小``N = k·|g|``（分离时为0）。

        **这是本项唯一精确的输出**（见类docstring）：平衡时它等于理论法向力，
        与罚刚度无关。摩擦锥要用的正是它——所以它是公开面而不是内部量。
        """

        return tuple(
            stiffness * -gap if (gap := self._gap_mm(state.vector, node, point, normal)) < 0.0
            else 0.0
            for node, point, normal, stiffness in self.planes
        )

__all__ = [
    "MAX_POINTS_PER_PAIR_SPHERE_CAPSULE",
    "REGIME_SEPARATED",
    "REGIME_SLIP",
    "REGIME_STICK",
    "SLOT_WIDTH",
    "ContactDeclaration",
    "ContactError",
    "ContactLayout",
    "ContactSlot",
    "ContactStep",
    "FrictionOutcome",
    "NORMAL_UNIT_TOLERANCE",
    "PenaltyNormalContact",
    "TangentialStickSpring",
    "advance_contact_quasistatic",
    "build_contact_layout",
    "coulomb_return_map",
]


@dataclass(frozen=True)
class FrictionOutcome:
    """一次return-map的结果：实际切向力、粘/滑判别、以及**锚点该挪到哪**。

    ``anchor_correction_mm``是滑移那一步锚点要平移的量（粘住时为零矢量）。
    **它是这一步产生的不可逆位移**——把它写回状态，历史就被记住了；
    不写回，下一步会以为自己还粘在原处，于是摩擦力凭空多出一截。
    """

    tangential_force_n: tuple[float, float, float]
    regime: float
    anchor_correction_mm: tuple[float, float, float]

    @property
    def is_stick(self) -> bool:
        return self.regime == REGIME_STICK


def coulomb_return_map(
    *,
    trial_force_n: tuple[float, float, float],
    normal_force_n: float,
    friction_coefficient: float,
    tangential_stiffness_n_per_mm: float,
) -> FrictionOutcome:
    """库仑摩擦的return-map：试探力落在摩擦锥内就粘，超出就投影回锥面并挪锚点。

    ## 这一段为什么不是能量项（0050第二节）

    库仑摩擦**耗散且非associative**——它做的功依赖路径，写不成任何位置的势函数。
    所以接触在本仓是"**半个能量项**"：法向在`PenaltyNormalContact`里，
    切向在这里，而这里**不满足**`EnergyTerm`四方法协议，也不该假装满足。

    ## 判据

    ``|T_trial| ≤ μN`` → **粘**：实际力就是试探力，锚点不动；
    否则 → **滑**：``T = μN · T_trial/|T_trial|``，并把锚点沿滑移方向挪
    ``(|T_trial| − μN)/k_t``——挪完之后，用新锚点重算的试探力恰好落在锥面上。
    **这一条是return-map的定义，也是它可被验证的地方**（见`tests/`那条自洽门）。

    ## 边界

    ``N = 0``（分离）时摩擦锥退化成一个点：**没有法向力就没有摩擦**，
    一切试探力都被投影成零，判别是`REGIME_SEPARATED`而不是`REGIME_SLIP`——
    "分离"与"在滑"是两件事，混起来会让案例分不清"飞出去了"和"在蹭着走"。
    """

    if normal_force_n < 0.0 or not math.isfinite(normal_force_n):
        raise ContactError(f"normal force must be finite and nonnegative: {normal_force_n!r}")
    if friction_coefficient < 0.0 or not math.isfinite(friction_coefficient):
        raise ContactError(
            f"friction coefficient must be finite and nonnegative: {friction_coefficient!r}"
        )
    if not (tangential_stiffness_n_per_mm > 0.0 and math.isfinite(tangential_stiffness_n_per_mm)):
        raise ContactError(
            f"tangential stiffness must be positive: {tangential_stiffness_n_per_mm!r}"
        )

    zero = (0.0, 0.0, 0.0)
    if normal_force_n == 0.0:
        return FrictionOutcome(zero, REGIME_SEPARATED, zero)

    limit = friction_coefficient * normal_force_n
    magnitude = math.sqrt(sum(component * component for component in trial_force_n))
    if magnitude <= limit:
        return FrictionOutcome(trial_force_n, REGIME_STICK, zero)

    scale = limit / magnitude
    force = tuple(component * scale for component in trial_force_n)
    #: 超出锥面的那一截除以切向刚度，就是这一步滑掉的距离。
    slip_mm = (magnitude - limit) / tangential_stiffness_n_per_mm
    correction = tuple(component / magnitude * slip_mm for component in trial_force_n)
    return FrictionOutcome(force, REGIME_SLIP, correction)


@dataclass(frozen=True)
class TangentialStickSpring:
    """粘着弹簧：``U = Σ ½·k_t·|P(x − a)|²``，``P = I − n⊗n``是切平面投影。单位N·mm。

    ## 它为什么**是**能量项，而滑移不是

    0033调研的结论：带粘着的库仑摩擦把切向相对位移**分解成可逆的"粘"分量与
    不可逆的"滑"分量**（与塑性力学的形制是同一个）。

    **可逆的那一半是弹性的，因此有势函数**——就是这个类。
    不可逆的那一半（滑移）耗散、非associative，**写不成任何位置的势**——
    那是`coulomb_return_map`。

    这就是0050第二节"接触是**半个能量项**"那句话的具体形状：
    法向 + 粘着在能量里（进`EnergyRegistry`、进牛顿的切线刚度），
    滑移在return-map里（改锚点，即改状态里的历史）。
    **把这条分界写在这里，是因为下一个人最可能犯的错是想把整个摩擦写成势能。**

    ## 它给对了什么

    与法向项同构：平衡时``k_t·|Δ| = T_理论``，于是**切向力精确、切向位移是``O(1/k_t)``**。
    斜面上实测：``T = W·sinθ``与``k_t``无关。

    ## 锚点是**输入**，不是这里算出来的

    本项读锚点，不写锚点。写锚点的是return-map——那一步才是历史发生的地方。
    锚点住在状态向量的槽位里（`ContactLayout`），本项从调用方拿到它的值。
    """

    name: str = "tangential_stick"
    #: (节点索引, 锚点mm, 面法向单位矢量, 切向刚度N/mm)
    springs: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float], float], ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.springs:
            raise ContactError("tangential_stick needs at least one spring")
        for node, anchor, normal, stiffness in self.springs:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"stick node index must be a nonnegative int: {node!r}")
            if len(anchor) != 3 or not all(math.isfinite(value) for value in anchor):
                raise ContactError(f"anchor must be a finite 3-vector: {anchor!r}")
            norm = math.sqrt(sum(component * component for component in normal))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(f"stick normal must be a unit vector (|n| = {norm!r})")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"tangential stiffness must be positive: {stiffness!r}")

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _ in self.springs) + 1

    @staticmethod
    def _tangential_offset_mm(
        vector: tuple[float, ...],
        node: int,
        anchor: tuple[float, float, float],
        normal: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """``P(x − a)``：位移里**扣掉法向分量**的那一部分。

        扣掉法向是这个项的全部要害：不扣，粘着弹簧会连法向也一起拉，
        于是它与法向罚函数**重复计入法向刚度**，而两者的刚度通常差好几个数量级——
        结果是法向力悄悄变成``(k_n + k_t)·δ``，**而`normal_force_n`报的仍是``k_n·δ``**。
        """

        base = 3 * node
        delta = tuple(vector[base + axis] - anchor[axis] for axis in range(3))
        along_normal = sum(delta[axis] * normal[axis] for axis in range(3))
        return tuple(delta[axis] - along_normal * normal[axis] for axis in range(3))

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(state.vector, node, anchor, normal)
            total += 0.5 * stiffness * sum(value * value for value in offset)
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(state.vector, node, anchor, normal)
            base = 3 * node
            for axis in range(3):
                result[base + axis] += stiffness * offset[axis]
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
        """``k_t·(I − n⊗n)``。**常量**——粘着是线性的，这是它比滑移好对付的原因。"""

        entries: list[tuple[int, int, float]] = []
        for node, _, normal, stiffness in self.springs:
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    value = stiffness * ((1.0 if a == b else 0.0) - normal[a] * normal[b])
                    entries.append((base + a, base + b, value))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(vector, node, anchor, normal)
            total += 0.5 * stiffness * sum(value * value for value in offset)
            if gradient is not None:
                base = 3 * node
                for axis in range(3):
                    gradient[base + axis] += stiffness * offset[axis]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def tangential_force_n(self, state: State) -> tuple[tuple[float, float, float], ...]:
        """每根弹簧的切向力矢量``k_t·P(x − a)``——**摩擦锥要判的就是它的模**。"""

        return tuple(
            tuple(
                stiffness * value
                for value in self._tangential_offset_mm(state.vector, node, anchor, normal)
            )
            for node, anchor, normal, stiffness in self.springs
        )


@dataclass(frozen=True)
class ContactStep:
    """一步准静态接触求解的结果。``state``里的锚点槽**已经被更新**。

    ``slip_increment_mm``是这一步滑掉的距离——**不可逆的那一部分**。
    它是本仓第一个真正被写回状态的历史量（0033裁"锚点是真历史"之后的兑现）。
    """

    state: State
    normal_force_n: float
    tangential_force_n: tuple[float, float, float]
    regime: float
    slip_increment_mm: float

    @property
    def is_stick(self) -> bool:
        return self.regime == REGIME_STICK


def advance_contact_quasistatic(
    *,
    registry_without_stick,
    context: EnergyContext,
    contact_layout: ContactLayout,
    slot: ContactSlot,
    vector: tuple[float, ...],
    node: int,
    normal: tuple[float, float, float],
    normal_stiffness_n_per_mm: float,
    tangential_stiffness_n_per_mm: float,
    friction_coefficient: float,
    fixed_indices: frozenset[int],
    residual_tol_n: float = 1.0e-12,
    max_iterations: int = 60,
) -> ContactStep:
    """走一步准静态接触：**弹性预测 → 求解 → return-map修正 → 锚点写回状态**。

    ## 这一步与前三片的差别

    前三片里锚点是**输入**：调用方给一个值，它整个过程不动。
    本函数是第一个**改写**它的东西——而改写它就是"历史发生了"。

    ## 为什么一趟预测-修正就够（不需要在牛顿里反复迭代）

    理想塑性（无硬化）的return-map有一条性质：**修正之后屈服条件恰好成立**。
    锚点被挪到``k_t·|x − a_new| = μN``，所以再解一次得到的还是同一个点。

    **这条性质对本片成立，是因为切向是位移控制的**（``x``被钉住）。
    载荷控制下``x``会随锚点变、锚点又随``x``变，那才需要真的迭代。
    **本函数不做那件事，也不假装做了**——触发条件登记在plans/07：
    第一个载荷控制的接触问题出现时。

    ## 法向与切向在本片是解耦的

    粘着弹簧扣掉了法向分量（``P = I − n⊗n``），法向罚只作用在法向，
    所以两者的Hessian块正交、``N``与``T``互不影响。
    **斜面与拖拽都落在这个前提内；一般曲面接触不落在**（法向随位置转），
    同样登记不假装。

    ``registry_without_stick``是**不含粘着项**的注册表——本函数按当前锚点
    自己造粘着项并接上去。这么设计是因为锚点每步都变，
    而`EnergyRegistry`是冻结的：**让调用方每步重建注册表，等于让它每步重写
    求和次序**，而求和次序是形制（spec/12第3.3节）。
    """

    from physics_engine.energies import EnergyRegistry
    from physics_engine.solve import solve_equilibrium

    anchor = tuple(vector[slot.anchor_base : slot.anchor_base + 3])
    normal_term = PenaltyNormalContact(
        planes=((node, (0.0, 0.0, 0.0), normal, normal_stiffness_n_per_mm),)
    )
    stick_term = TangentialStickSpring(
        springs=((node, anchor, normal, tangential_stiffness_n_per_mm),)
    )
    registry = EnergyRegistry(
        terms=(*registry_without_stick.terms, normal_term, stick_term)
    )
    solved = solve_equilibrium(
        registry,
        context,
        contact_layout.layout,
        vector,
        fixed_indices=fixed_indices,
        residual_tol_n=residual_tol_n,
        max_iterations=max_iterations,
    )
    if not solved.converged:
        raise ContactError(f"contact step did not converge: {solved.reason}")

    normal_force = normal_term.normal_force_n(solved.state)[0]
    trial = stick_term.tangential_force_n(solved.state)[0]
    outcome = coulomb_return_map(
        trial_force_n=trial,
        normal_force_n=normal_force,
        friction_coefficient=friction_coefficient,
        tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
    )

    updated = list(solved.state.vector)
    for axis in range(3):
        updated[slot.anchor_base + axis] = anchor[axis] + outcome.anchor_correction_mm[axis]
    updated[slot.active_index] = 0.0 if normal_force == 0.0 else 1.0
    updated[slot.regime_index] = outcome.regime
    slip = math.sqrt(
        sum(value * value for value in outcome.anchor_correction_mm)
    )
    return ContactStep(
        state=State(layout=contact_layout.layout, vector=tuple(updated)),
        normal_force_n=normal_force,
        tangential_force_n=outcome.tangential_force_n,
        regime=outcome.regime,
        slip_increment_mm=slip,
    )
