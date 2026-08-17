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
        layout=StateLayout(
            layout_id=layout_id,
            fields=tuple(fields),
            #: **边界带上来**：没有它，能量层与积分桥都只能信上下文的质量表，
            #: 而那两处各自都判不出"上下文与布局不符"（实测重力落到锚点上）。
            node_dof_count=3 * node_count,
        ),
        node_count=node_count,
        slots=tuple(slots),
    )




