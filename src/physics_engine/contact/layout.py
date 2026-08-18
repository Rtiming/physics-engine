"""锚点布局与接触声明的**词汇**（决策0050第一节）。

本模块不算任何物理，只回答"粘着锚点放在状态的什么位置"。
它同时是本子包的**词汇底座**：槽宽、每对槽数、regime取值、法向来源类型
都在这里定义，`penalty`/`friction`/`damping`/`stepper`单向依赖它。

拆分自原`contact.py`（2026-08-17，plans/14第三节的并行波准备）——
**函数体逐字节未动**，只是搬了文件。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from physics_engine.contact.errors import ContactError
from physics_engine.energies import EnergyContext
from physics_engine.state import StateField, StateLayout

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

#: 法向的来源：固定元组，或「给定当前状态向量返回当前法向」的可调用对象。
#: 后者是曲面接触需要的——法向随位形转时一趟预测-修正不够
#: （见`advance_contact_quasistatic`的实测数据）。
NormalSource = (
    tuple[float, float, float]
    | Callable[[tuple[float, ...]], tuple[float, float, float]]
)

#: `regime`字段的取值。**用数而不用字符串**，因为状态是一维浮点向量
#: （spec/12第2.2节：显式数组不是对象图）。
REGIME_SEPARATED = 0.0
REGIME_STICK = 1.0
REGIME_SLIP = 2.0

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
    #: 带转动块的节点号，按声明次序（决策0080）。**空元组时本类与2026-08-18之前
    #: 逐字节相同**——字段清单、向量长度、指纹全都不变，那是三前提第三条的执行面。
    rotating_bodies: tuple[int, ...] = ()

    @property
    def node_dof_count(self) -> int:
        return 3 * self.node_count

    @property
    def rotation_dof_count(self) -> int:
        return 3 * len(self.rotating_bodies)

    def rotation_base(self, body: int) -> int:
        """该体转动块第一个标量的**绝对下标**。调用方永不手写偏移量。

        ## 转动块排在锚点槽**之后**，而不是像`rotation.RigidBodyLayout`那样紧跟节点块

        两处的次序不同**不是不一致，是各自的承重约束不同**：

        * `RigidBodyLayout`没有锚点槽，转动块紧跟节点块是唯一的选择；
        * 这里有锚点槽，而槽的下标是**历史的住处**。把转动块插在节点块与槽之间
          会把每一个既有槽的``base``整体后移——**既有产物的锚点会读到别人的历史**，
          而那是0001三前提第三条明令禁止的那一类改动。

        转动块能排在最后，是因为`MaterialPoint.rotation_base`吃的是**绝对下标**，
        它对"转动块在哪"没有任何假设（决策0080第二节）。
        """

        try:
            order = self.rotating_bodies.index(body)
        except ValueError:
            raise ContactError(
                f"node {body} carries no rotation block — "
                f"带转动块的是{list(self.rotating_bodies)}"
            ) from None
        return self.node_dof_count + len(self.slots) * SLOT_WIDTH + 3 * order

    def rotation_indices(self) -> frozenset[int]:
        """全部转动标量下标——"把转动自由度全部钉住"就是把它交给`fixed_indices`。"""

        return frozenset(
            self.rotation_base(body) + axis
            for body in self.rotating_bodies
            for axis in range(3)
        )

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

        转动块同样全零——**局部图的原点就是``θ = 0``**，不需要额外初始化
        （与`rotation.RigidBodyLayout.initial_vector`逐字同源）。
        """

        if len(node_positions_mm) != self.node_dof_count:
            raise ContactError(
                f"expected {self.node_dof_count} node scalars, "
                f"got {len(node_positions_mm)}"
            )
        return tuple(node_positions_mm) + (0.0,) * (
            len(self.slots) * SLOT_WIDTH + self.rotation_dof_count
        )


def build_contact_layout(
    *,
    layout_id: str,
    node_count: int,
    declarations: tuple[ContactDeclaration, ...],
    rotating_bodies: tuple[int, ...] = (),
) -> ContactLayout:
    """按**声明**的接触对造定长布局。次序即声明次序。

    次序留住的理由与`scene.ContactPair`那条逐字相同：接触对清单要交给求解器
    按序处理，而集合的遍历次序随``PYTHONHASHSEED``变——
    **同一份声明两次跑出不同次序，接触结果就不可复现**（轴3规则5）。
    这里更硬一层：次序还决定了每个槽落在向量的哪一格，
    **次序变了梯度与Hessian的索引全错，而多数测试不会发现**（spec/12第2.2节）。

    ## ``rotating_bodies``：转动块（决策0080）

    默认空元组，此时产出**与2026-08-18之前逐字节相同**——字段清单、向量长度、
    `StateLayout`指纹全都不变。这不是"算出来恰好相等"，是那几行根本不执行。

    非空时在锚点槽**之后**追加``body{b}_theta_{x,y,z}_rad``。
    排在最后的理由见`ContactLayout.rotation_base`：插在中间会把每个既有槽的
    ``base``后移，而槽是历史的住处。
    """

    if node_count < 1:
        raise ContactError("a contact layout needs at least one node")
    seen_bodies: list[int] = []
    for body in rotating_bodies:
        if isinstance(body, bool) or not isinstance(body, int):
            raise ContactError(f"rotating body must be an int node index: {body!r}")
        if not (0 <= body < node_count):
            raise ContactError(
                f"rotating body {body} is outside the node block [0, {node_count})"
            )
        if body in seen_bodies:
            raise ContactError(
                f"node {body} declared twice as a rotating body — "
                "重复声明意味着同一个体有两个转动块，而哪一个说了算只能靠读实现"
            )
        seen_bodies.append(body)
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
    #: 转动块**不是历史**（决策0079第二节裁的那条放宽），所以这几个字段
    #: 不带``is_history``——它们是真自由度，牛顿会解它们。
    fields.extend(
        StateField(f"body{body}_theta_{axis}_rad", 1)
        for body in rotating_bodies
        for axis in ("x", "y", "z")
    )

    return ContactLayout(
        layout=StateLayout(
            layout_id=layout_id,
            fields=tuple(fields),
            #: **边界带上来**：没有它，能量层与积分桥都只能信上下文的质量表，
            #: 而那两处各自都判不出"上下文与布局不符"（实测重力落到锚点上）。
            node_dof_count=3 * node_count,
        ),
        node_count=node_count,
        slots=tuple(slots),
        rotating_bodies=tuple(rotating_bodies),
    )


__all__ = [
    "MAX_POINTS_PER_PAIR_SPHERE_CAPSULE",
    "NORMAL_UNIT_TOLERANCE",
    "REGIME_SEPARATED",
    "REGIME_SLIP",
    "REGIME_STICK",
    "SLOT_WIDTH",
    "ContactDeclaration",
    "ContactLayout",
    "ContactSlot",
    "NormalSource",
    "build_contact_layout",
]
