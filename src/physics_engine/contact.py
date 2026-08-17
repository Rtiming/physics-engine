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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.energies import DISSIPATION, POTENTIAL, EnergyContext, Matrix, Vector
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


class ContactError(ValueError):
    """接触层的一切失败关闭。"""


def _force_zero_contact_time_factor(damping_ratio: float) -> float:
    """返回``ω0·t_c``；定义域覆盖欠阻尼、临界与过阻尼。"""

    if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
        raise ContactError(
            f"damping ratio must be finite and nonnegative: {damping_ratio!r}"
        )
    if damping_ratio == 1.0:
        return 2.0
    if damping_ratio < 1.0:
        root = math.sqrt((1.0 - damping_ratio) * (1.0 + damping_ratio))
        return 2.0 * math.acos(damping_ratio) / root
    root = math.sqrt(damping_ratio - 1.0) * math.sqrt(damping_ratio + 1.0)
    return 2.0 * math.acosh(damping_ratio) / root


def restitution_from_damping_ratio(damping_ratio: float) -> float:
    """合力归零分离约定下的线性弹簧-dashpot恢复系数。"""

    factor = _force_zero_contact_time_factor(damping_ratio)
    return math.exp(-damping_ratio * factor)


def damping_ratio_from_restitution(restitution: float) -> float:
    """反解``0 < e ≤ 1``对应的唯一有限阻尼比；过阻尼同样允许。"""

    if not math.isfinite(restitution) or not (0.0 < restitution <= 1.0):
        raise ContactError(
            f"restitution must be finite and in (0, 1], got {restitution!r}; "
            "e=0 only occurs in the infinite-damping limit"
        )
    if restitution == 1.0:
        return 0.0
    lower, upper = 0.0, 1.0
    while restitution_from_damping_ratio(upper) > restitution:
        upper *= 2.0
        if not math.isfinite(upper):
            raise ContactError(
                f"restitution {restitution!r} requires a damping ratio beyond float range"
            )
    #: e(ζ)在[0,∞)严格单调；固定120次让结果跨平台确定，不用容差提前退出。
    for _ in range(120):
        middle = 0.5 * (lower + upper)
        if restitution_from_damping_ratio(middle) > restitution:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


@dataclass(frozen=True)
class LinearDashpotParameters:
    """目标恢复系数派生出的接触动力学参数（mm-N-kg-s单位制）。"""

    restitution: float
    damping_ratio: float
    stiffness_n_per_mm: float
    effective_mass_kg: float
    damping_n_s_per_mm: float
    omega0_rad_per_s: float
    contact_duration_s: float
    stability_rate_per_s: float


def linear_dashpot_parameters(
    *, stiffness_n_per_mm: float, effective_mass_kg: float, restitution: float
) -> LinearDashpotParameters:
    """从``(k, m_eff, e)``派生dashpot系数、接触时长与显式稳定率。"""

    if not math.isfinite(stiffness_n_per_mm) or stiffness_n_per_mm <= 0.0:
        raise ContactError(
            f"penalty stiffness must be positive and finite: {stiffness_n_per_mm!r}"
        )
    if not math.isfinite(effective_mass_kg) or effective_mass_kg <= 0.0:
        raise ContactError(
            f"effective mass must be positive and finite: {effective_mass_kg!r}"
        )
    damping_ratio = damping_ratio_from_restitution(restitution)
    omega0 = math.sqrt(1000.0 * stiffness_n_per_mm / effective_mass_kg)
    damping = 2.0 * damping_ratio * effective_mass_kg * omega0 / 1000.0
    duration = _force_zero_contact_time_factor(damping_ratio) / omega0
    if damping_ratio <= 1.0:
        stability_rate = omega0
    else:
        root = math.sqrt(damping_ratio - 1.0) * math.sqrt(damping_ratio + 1.0)
        stability_rate = (damping_ratio + root) * omega0
    return LinearDashpotParameters(
        restitution=restitution,
        damping_ratio=damping_ratio,
        stiffness_n_per_mm=stiffness_n_per_mm,
        effective_mass_kg=effective_mass_kg,
        damping_n_s_per_mm=damping,
        omega0_rad_per_s=omega0,
        contact_duration_s=duration,
        stability_rate_per_s=stability_rate,
    )


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




@dataclass(frozen=True)
class PenaltyNormalContact:
    """罚函数式法向接触（**半空间**）：``U = Σ ½·k·g²``（仅``g < 0``），单位N·mm。

    ``g = (x − p)·n − r``是**间隙**：``n``是半空间的外法向（指向节点该待的那一侧），
    ``p``是面上一点，``r``是节点代表的球半径。``g > 0``分离、``g < 0``穿透。

    ## 半径为什么是**显式参数**而不是让调用方自己偏移平面

    数学上把面沿法向抬高``r``是等价的，但那条路**刚在三球金字塔上绊过一次**：
    球心在高度``R``、面在``z = 0``，间隙算出来是``+R``——**底球被判成悬空**，
    于是切线刚度里出现一个零模态、牛顿当场走不动。
    症状是"求解器坏了"，病根是"调用方以为传的是球、而这里当它是质点"。

    **让半径出现在调用点，那个误会就没有发生的余地。** 质点写``0.0``。

    ## 量纲（本仓已因单位吃过两次亏，所以这里逐个写出来）

    ``k``是N/mm、``g``是mm，故``½kg²``是``N/mm · mm² = N·mm``——
    **直接就是本仓的能量单位，不需要`MM_PER_M`**。
    与`PointLoad`同理、与`UniformGravity`相反（后者拿的是kg与mm/s²）。
    **量纲是算出来的，不是照抄相邻代码抄出来的。**

    ## 这个模型给对了什么、给错了什么（0050第二节的代价，写在实现里）

    平衡时``k·δ = N_理论``，于是：

    * **法向力是精确的**，与``k``无关——``δ = N/k``，``N = k·δ = N``恒成立；
    * **穿透不为零**。**准静态**下``δ = N/k``是``O(1/k)``；
      **瞬态冲击下不是这条**——那时``δ_max = v_in·sqrt(m/k)``即``O(k^(−1/2))``
      （实测``k = 1e5``时准静态式差**1010倍**，见research/13第五节）。

      **上面那条瞬态式设重力在接触段不作用。** 含重力时是

          δ_max = δ_eq + sqrt(δ_eq² + (v_in/ω)²)，  δ_eq = mg/k

      重力修正对"两律之比"是**加性的约+1**，所以它**只在低刚度档要命**——
      而那正是research/13第五节原表用来展示"比值≈1"的那一行，
      **该行真值是表中值的2.40倍**（2026-08-12轨道D三条腿复核：闭式、
      独立RK4、引擎实测，见research/15）。
      `cases/bouncing_ball_restitution`把重力**刻意设为零**并写进了参数表，
      因此不受这条限定影响。
      **两条律各管各的域，瞬态案例的判据不许照抄准静态那条**，
      否则刚度提100倍时判据会松100倍而不是10000倍。
      穿透本身**是模型不是缺陷**，但刚度必须是**输入**不是代码里的魔数。

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
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: 半空间：(节点索引, 面上一点mm, 外法向单位矢量, 罚刚度N/mm, 球半径mm)
    #: **半径是必填的**，质点写``0.0``——理由见类docstring第二节。
    planes: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float], float, float],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.planes:
            raise ContactError("normal_contact needs at least one half-space")
        for node, point, normal, stiffness, radius in self.planes:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"contact node index must be a nonnegative int: {node!r}")
            if radius < 0.0 or not math.isfinite(radius):
                raise ContactError(f"contact radius must be finite and nonnegative: {radius!r}")
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
        return max(node for node, _, _, _, _ in self.planes) + 1

    @staticmethod
    def _gap_mm(
        vector: tuple[float, ...],
        node: int,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
        radius: float,
    ) -> float:
        base = 3 * node
        return sum(
            (vector[base + axis] - point[axis]) * normal[axis] for axis in range(3)
        ) - radius

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal, radius)
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal, radius)
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
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal, radius)
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
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(vector, node, point, normal, radius)
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
            stiffness * -gap
            if (gap := self._gap_mm(state.vector, node, point, normal, radius)) < 0.0
            else 0.0
            for node, point, normal, stiffness, radius in self.planes
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
    "ContactPoint",
    "ContactSlot",
    "ContactStep",
    "FrictionOutcome",
    "LinearDashpotParameters",
    "LinearNormalDashpot",
    "MultiContactStep",
    "NORMAL_UNIT_TOLERANCE",
    "PenaltyCylinderContact",
    "PenaltyNormalContact",
    "PenaltySphereContact",
    "TangentialStickSpring",
    "advance_contact_quasistatic",
    "advance_contacts_quasistatic",
    "build_contact_layout",
    "coulomb_return_map",
    "damping_ratio_from_restitution",
    "linear_dashpot_parameters",
    "restitution_from_damping_ratio",
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

    #: **试探力此前完全不校验**，而同一函数对``N``/``μ``/``k_t``都很严
    #: （2026-08-06对抗审核）：nan进来会原样变成锚点修正**写进状态向量**，
    #: 而状态是复现契约；长度2或4的元组会被原样返回。
    if len(trial_force_n) != 3 or not all(math.isfinite(v) for v in trial_force_n):
        raise ContactError(f"trial force must be a finite 3-vector: {trial_force_n!r}")
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
    kind: ClassVar[Literal["potential"]] = POTENTIAL
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
            #: **单位矢量那道门挡不住nan**：``abs(nan − 1.0) > tol``是``False``，
            #: 于是nan法向一路通过、能量与梯度全变nan（2026-08-06对抗审核实测）。
            #: 同门的`PenaltyNormalContact`有这两条检查，这里此前没有。
            if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
                raise ContactError(f"stick normal must be a finite 3-vector: {normal!r}")
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
    #: 实际走了几趟预测-修正。固定法向下恒为1。
    passes: int = 1
    #: **最后一趟的屈服超出量**``|T_trial| − μN``：预测-修正是否收敛的唯一观测量。
    #:
    #: `tangential_force_n`是**投影后**的力，滑移时它按构造恒等于``μN``——
    #: 拿它去判收敛永远得到0。起草时正是这样量了个寂寞。
    #: **一个观测不到的收敛判据不是判据**，所以它必须是公开字段。
    #: 粘着时它≤0。
    yield_excess_n: float = 0.0

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
    normal: NormalSource,
    normal_force_of: Callable[[State], float],
    tangential_stiffness_n_per_mm: float,
    friction_coefficient: float,
    fixed_indices: frozenset[int],
    residual_tol_n: float = 1.0e-12,
    max_iterations: int = 60,
    max_passes: int = 1,
    yield_tol_n: float = 0.0,
    require_pass_convergence: bool = False,
) -> ContactStep:
    """走一步准静态接触：**弹性预测 → 求解 → return-map修正 → 锚点写回状态**。

    ## 这一步与前三片的差别

    前三片里锚点是**输入**：调用方给一个值，它整个过程不动。
    本函数是第一个**改写**它的东西——而改写它就是"历史发生了"。

    ## 一趟够不够，取决于**法向转不转**——这是实测出来的，不是推的

    **法向固定时一趟就够**：理想塑性的return-map修正后屈服条件恰好成立
    （锚点被挪到``k_t·|x − a_new| = μN``），法向不动意味着再解一次还是同一个点。
    斜面与位移控制的拖拽都落在这个前提内。

    **法向随位形转时一趟远远不够。** 实测（两个固定球夹一个受横载的球，
    ``μ = 0.35``、``k = 1e5``）：一趟之后屈服残差``|T| − μN``是**2.32 N**，
    此后**每多一趟恰好减半**——2.319、1.160、0.580、0.290、0.145……
    **线性收敛、压缩因子1/2。**

    因此``normal``可以是**可调用对象**：给定当前向量返回当前法向；
    固定法向传元组即可（等价于常值可调用）。
    ``max_passes``是预测-修正的趟数，``yield_tol_n``是屈服残差的收敛判据。

    **两者的默认值刻意保守（1趟、0容差）**：老调用点全是固定法向、一趟本来就精确，
    **默认值不该替既有调用方改变行为**——那是把一次能力扩展偷偷变成一次行为变更。

    ``registry_without_stick``是**不含粘着项**的注册表——本函数按当前锚点
    自己造粘着项并接上去。这么设计是因为锚点每步都变，
    而`EnergyRegistry`是冻结的：**让调用方每步重建注册表，等于让它每步重写
    求和次序**，而求和次序是形制（spec/12第3.3节）。

    ## 法向力由调用方给，**本函数不拥有法向接触**

    第一版自己造了一个"过原点的半空间"法向项。**那个设计在球-球接触上当场失效**：
    球心离原点27 mm，那个半空间把它判成分离，于是法向力为0、
    return-map走"分离"分支、锚点一动不动——**而屈服超出量明明是2.9 N**。

    症状是"迭代不收敛"，病根是**步进器越权拥有了一个它不该拥有的东西**。
    法向接触可能是半空间、可能是球-球、将来可能是网格，
    **它属于调用方的注册表，而这里只需要一个数：当前法向力**。
    """

    from physics_engine.energies import EnergyRegistry
    from physics_engine.solve import solve_equilibrium

    if max_passes < 1:
        raise ContactError(f"max_passes must be at least 1: {max_passes!r}")

    #: **`slot`与`node`的落位校验**（plans/09第七节第7条，2026-08-12补）。
    #:
    #: 这两个参数各说各的：`node`指节点块里的第几个节点，`slot`指锚点块里的一个槽，
    #: 而**`ContactSlot`不带node、`ContactDeclaration`也不带**——
    #: 两者之间今天没有任何东西把它们绑在一起。
    #:
    #: 完整的对应校验要改0050的布局承重设计（让声明带上节点），**本轮不做**。
    #: 能证明的这一半先做掉——它挡的是两种会静默写坏状态的混淆：
    #:
    #: * 把槽下标当节点号传 → 锚点写进节点块，**位形被悄悄改掉**；
    #: * 节点号越过节点块 → 写进锚点块，**改的是别人的历史**。
    #:
    #: 这两种都不会抛`IndexError`——负下标与越界写在元组切片上是静默的，
    #: 而那正是决策0050落地时`PenaltySphereContact`吃过的亏
    #: （``node = -1``被接受、``vector[-3:]``读的正是锚点槽）。
    node_dof_count = contact_layout.layout.node_dof_count
    if not isinstance(node, int) or isinstance(node, bool) or node < 0:
        raise ContactError(f"contact node index must be a nonnegative int: {node!r}")
    if 3 * node + 3 > node_dof_count:
        raise ContactError(
            f"node {node} 落在节点块之外（节点块只有{node_dof_count}个自由度）"
            "——**再往后是锚点槽，写进去就是改别人的历史**"
        )
    if slot.base < node_dof_count:
        raise ContactError(
            f"slot.base={slot.base} 落在节点块之内（节点块{node_dof_count}个自由度）"
            "——**锚点槽必须在节点块之后，写进节点块就是悄悄改位形**"
        )
    if slot.regime_index >= len(vector):
        raise ContactError(
            f"slot {slot.pair_id!r} 点{slot.point_index}的槽越过了状态向量末尾"
            f"（需要下标{slot.regime_index}，向量只有{len(vector)}个）"
        )

    normal_of = normal if callable(normal) else (lambda _v, fixed=normal: fixed)

    anchor = tuple(vector[slot.anchor_base : slot.anchor_base + 3])
    current = vector
    stick_term = solved = outcome = None
    normal_force = 0.0
    passes = 0
    #: **分离时不许接粘着项**（2026-08-06对抗审核抓到的静默错值）。
    #:
    #: `TangentialStickSpring`**没有间隙判据**——它只有锚点、法向、刚度，
    #: 看不见接触还在不在。此前本函数**无条件**把它接进注册表，后果实测：
    #: 节点抬到面上方500 mm，法向力正确归零、regime正确报SEPARATED、
    #: 报告的切向力也是0，**而平衡位置仍被一根不存在的摩擦弹簧顶在``T/k_t``上**。
    #:
    #: 更难看的是**正确的活动集会失败关闭**：把粘着项拿掉，
    #: 切向没有任何刚度，`solve_equilibrium`当场报奇异——
    #: **也就是说正确的做法炸，而错误的做法静默给答案**，
    #: 且切线刚度照样正定，求解器那张"欠约束即失败关闭"的网罩不到切向。
    #:
    #: 判据用**上一趟的法向力**（第一趟用入参状态算），
    #: 因为活动集必须在装配之前定——这正是罚接触"活动集变化处不可微"的那条边界。
    engaged = normal_force_of(State(layout=contact_layout.layout, vector=current)) > 0.0
    for _ in range(max_passes):
        passes += 1
        direction = normal_of(current)
        stick_term = TangentialStickSpring(
            springs=((node, anchor, direction, tangential_stiffness_n_per_mm),)
        )
        terms = (
            (*registry_without_stick.terms, stick_term)
            if engaged
            else registry_without_stick.terms
        )
        registry = EnergyRegistry(terms=terms)
        solved = solve_equilibrium(
            registry,
            context,
            contact_layout.layout,
            current,
            fixed_indices=fixed_indices,
            residual_tol_n=residual_tol_n,
            max_iterations=max_iterations,
        )
        if not solved.converged:
            raise ContactError(f"contact step did not converge: {solved.reason}")
        current = solved.state.vector

        normal_force = normal_force_of(solved.state)
        engaged = normal_force > 0.0
        trial = stick_term.tangential_force_n(solved.state)[0] if engaged else (0.0,) * 3
        outcome = coulomb_return_map(
            trial_force_n=trial,
            normal_force_n=normal_force,
            friction_coefficient=friction_coefficient,
            tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
        )
        anchor = tuple(
            anchor[axis] + outcome.anchor_correction_mm[axis] for axis in range(3)
        )
        #: 屈服残差：试探力超出锥面多少。粘着时它按定义≤0，
        #: 故本判据只在滑移分支上有内容——粘着一趟就退出，与旧行为一致。
        excess = (
            math.sqrt(sum(value * value for value in trial))
            - friction_coefficient * normal_force
        )
        if excess <= yield_tol_n:
            break
    else:
        #: **趟数用尽而屈服残差仍超标**（plans/09第七节第6条，2026-08-12补）。
        #:
        #: 同一个函数里，**内层牛顿不收敛是`raise`**，而外层趟数用尽此前
        #: 什么都不做——两种不收敛，两种待遇。
        #:
        #: 但直接改成无条件抛**会打断所有既有调用方**：`excess`量的是
        #: **修正前**的试探力，而`max_passes=1`（默认）加滑移时
        #: "用尽且excess>0"正是**正常且正确**的情形——
        #: 理想塑性的修正让屈服条件在**修正后**成立，本判据看不到那一步。
        #:
        #: 所以做成**选择进入**：默认`False`保持既有行为逐字节不变。
        #: 这条依据是本函数docstring自己写过的原则——
        #: **默认值不该替既有调用方改变行为，那是把一次能力扩展偷偷变成一次行为变更**。
        #:
        #: 观测量一直都在（`ContactStep.yield_excess_n`），缺的只是"要不要当错"。
        if require_pass_convergence:
            raise ContactError(
                f"预测-修正走满{max_passes}趟仍未收敛："
                f"屈服残差{excess:.6g} N > 容差{yield_tol_n:.6g} N。"
                "**加大max_passes，或按yield_excess_n自己判**——"
                "实测法向随位形转时每趟压缩因子约1/2（0050第五节）"
            )

    updated = list(solved.state.vector)
    #: **分离时清切向历史**（同一次对抗审核的第二条静默错值）。
    #:
    #: ``N = 0``时return-map返回零修正、锚点纹丝不动，而此前**没有任何地方
    #: 在再接触时重置它**。实测的后果：贴地拖到2 mm、抬到空中横移到50 mm、
    #: 再放回地面——那一步报出``slip_increment_mm = 48``，
    #: **凭空记了282.5 N·mm的摩擦功，而那48 mm是在空中走的**。
    #: 同一个洞的另一面：空中把切向放开，幽灵弹簧在50 mm处出力1.5e6 N。
    #:
    #: 同行的罚摩擦实现（Chrono DEM、LAMMPS granular）在失去接触时一律清切向历史。
    if not engaged:
        anchor = (0.0, 0.0, 0.0)
    for axis in range(3):
        updated[slot.anchor_base + axis] = anchor[axis]
    updated[slot.active_index] = 0.0 if normal_force == 0.0 else 1.0
    updated[slot.regime_index] = outcome.regime
    #: **整步的总滑移，不是最后一趟的修正量。** 多趟时锚点是逐趟累加的，
    #: 取最后一趟等于把前面几趟滑掉的距离丢掉——而那正是这一步的不可逆位移。
    origin = vector[slot.anchor_base : slot.anchor_base + 3]
    #: 分离那一步的滑移是**零**，不是"锚点从旧值归零"的那段距离——
    #: 后者会把清历史这个动作本身记成一次滑移。
    slip = (
        0.0
        if not engaged
        else math.sqrt(sum((anchor[axis] - origin[axis]) ** 2 for axis in range(3)))
    )
    return ContactStep(
        state=State(layout=contact_layout.layout, vector=tuple(updated)),
        normal_force_n=normal_force,
        tangential_force_n=outcome.tangential_force_n,
        regime=outcome.regime,
        slip_increment_mm=slip,
        passes=passes,
        yield_excess_n=excess,
    )


@dataclass(frozen=True)
class ContactPoint:
    """多槽位步进器的一个接触点：槽、节点、法向来源、法向力读数与两个材料常数。

    把它做成一个记录而不是六条平行元组，是因为**它们必须同进同出**——
    0050落地时`PenaltySphereContact`吃过的亏正是"节点号与槽位各说各的"，
    而那两个参数当时就是分开传的。
    """

    slot: ContactSlot
    node: int
    normal: NormalSource
    normal_force_of: Callable[[State], float]
    tangential_stiffness_n_per_mm: float
    friction_coefficient: float


@dataclass(frozen=True)
class MultiContactStep:
    """多槽位一步的结果：新状态 + **逐点**的力、判别、滑移。

    ``passes``与``max_yield_excess_n``是整步的，其余按``contacts``的声明次序逐点给出。
    """

    state: State
    normal_force_n: tuple[float, ...]
    tangential_force_n: tuple[tuple[float, float, float], ...]
    regime: tuple[float, ...]
    slip_increment_mm: tuple[float, ...]
    passes: int
    max_yield_excess_n: float


def advance_contacts_quasistatic(
    *,
    registry_without_stick,
    context: EnergyContext,
    contact_layout: ContactLayout,
    contacts: Sequence[ContactPoint],
    vector: tuple[float, ...],
    fixed_indices: frozenset[int],
    residual_tol_n: float = 1.0e-12,
    max_iterations: int = 60,
    max_passes: int = 1,
    yield_tol_n: float = 0.0,
    require_pass_convergence: bool = False,
) -> MultiContactStep:
    """走一步准静态接触，**多个槽位同时迭代**——0050第一节登记的那笔欠账。

    ## 为什么单槽位版本不够

    `advance_contact_quasistatic`一次只处理一个槽。那对斜面、对拖拽都够，
    对**带材贴在导轮上**不够：一条带材同时压在几十个节点上，
    每个节点各有自己的锚点与粘/滑判别，**而它们通过带材的轴向刚度互相牵制**——
    这正是绞盘公式里张力沿包角累积的机制。逐个槽轮流走会把这个耦合拆散，
    算出来的张力比不再是``e^{μθ}``。

    S3.6"多点同时接触"的`missing`里写的"步进器一次只处理一个槽位"，指的就是这里。

    ## 与单槽位版本的关系：**新函数，老函数一个字不动**

    三前提第三条要求既有产物逐字节不变，而合并两条路径必然要重排求和次序
    （spec/12第3.3节：求和次序是形制）。所以这里是并列的第二条路径，
    并由`test_contact_multi_stepper.py`守着"单个接触时两者逐字节相同"——
    **那条门是这次泛化忠实与否的证据，不是装饰**。

    ## 装配次序：所有粘着弹簧进**同一个**`TangentialStickSpring`

    不是每个接触一个项。理由同上——项数随活动集变，
    则`EnergyRegistry`的求和次序随活动集变，而活动集每趟都可能翻转。
    一个项、按``contacts``的声明次序排列，**次序就只由声明决定**。

    ## 收敛判据是**全体**，不是任一

    只要还有一个接触的屈服残差超标就继续下一趟。实测的压缩因子见单槽位版本
    的docstring（法向随位形转时约1/2）；多点耦合下不保证同样的因子，
    所以``require_pass_convergence``在这里更该开——但默认仍是``False``，
    与单槽位版本同口径。
    """

    from physics_engine.energies import EnergyRegistry
    from physics_engine.solve import solve_equilibrium

    if max_passes < 1:
        raise ContactError(f"max_passes must be at least 1: {max_passes!r}")
    if not contacts:
        raise ContactError("advance_contacts_quasistatic needs at least one contact point")

    node_dof_count = contact_layout.layout.node_dof_count
    seen_slots: set[int] = set()
    for index, point in enumerate(contacts):
        #: 逐点复用单槽位版本那四条落位校验——它们挡的是**静默写坏状态**，
        #: 而多槽位下写坏的概率恰好乘以点数。
        if not isinstance(point.node, int) or isinstance(point.node, bool) or point.node < 0:
            raise ContactError(f"contact node index must be a nonnegative int: {point.node!r}")
        if 3 * point.node + 3 > node_dof_count:
            raise ContactError(
                f"contacts[{index}]的节点{point.node}落在节点块之外"
                f"（节点块只有{node_dof_count}个自由度）——再往后是锚点槽"
            )
        if point.slot.base < node_dof_count:
            raise ContactError(
                f"contacts[{index}]的slot.base={point.slot.base}落在节点块之内"
                f"（节点块{node_dof_count}个自由度）——写进节点块就是悄悄改位形"
            )
        if point.slot.regime_index >= len(vector):
            raise ContactError(
                f"contacts[{index}]的槽越过了状态向量末尾"
                f"（需要下标{point.slot.regime_index}，向量只有{len(vector)}个）"
            )
        #: **两个接触点共用一个槽是静默灾难**：后写的覆盖先写的，
        #: 于是一段历史凭空消失而两点都报"我记住了"。单槽位版本没有这条，
        #: 因为它一次只有一个槽——**多槽位是这条检查第一次有意义的地方**。
        if point.slot.anchor_base in seen_slots:
            raise ContactError(
                f"contacts[{index}]与前面某点共用锚点槽{point.slot.anchor_base}"
                "——后写的会覆盖先写的，那段历史会凭空消失"
            )
        seen_slots.add(point.slot.anchor_base)
        if not (
            point.tangential_stiffness_n_per_mm > 0.0
            and math.isfinite(point.tangential_stiffness_n_per_mm)
        ):
            raise ContactError(
                f"contacts[{index}]的切向刚度必须为正："
                f"{point.tangential_stiffness_n_per_mm!r}"
            )

    normal_of = tuple(
        point.normal
        if callable(point.normal)
        else (lambda _v, fixed=point.normal: fixed)  # noqa: B008
        for point in contacts
    )
    origins = tuple(
        tuple(vector[point.slot.anchor_base : point.slot.anchor_base + 3])
        for point in contacts
    )
    anchors = list(origins)
    current = vector
    count = len(contacts)

    start_state = State(layout=contact_layout.layout, vector=current)
    engaged = [point.normal_force_of(start_state) > 0.0 for point in contacts]
    normal_forces = [0.0] * count
    outcomes: list[FrictionOutcome | None] = [None] * count
    excesses = [0.0] * count
    passes = 0
    solved = None

    for _ in range(max_passes):
        passes += 1
        springs = tuple(
            (
                contacts[index].node,
                anchors[index],
                normal_of[index](current),
                contacts[index].tangential_stiffness_n_per_mm,
            )
            for index in range(count)
            if engaged[index]
        )
        terms = (
            (*registry_without_stick.terms, TangentialStickSpring(springs=springs))
            if springs
            else registry_without_stick.terms
        )
        solved = solve_equilibrium(
            EnergyRegistry(terms=terms),
            context,
            contact_layout.layout,
            current,
            fixed_indices=fixed_indices,
            residual_tol_n=residual_tol_n,
            max_iterations=max_iterations,
        )
        if not solved.converged:
            raise ContactError(f"contact step did not converge: {solved.reason}")
        current = solved.state.vector

        #: 试探力要按**装配时**的活动集取——`tangential_force_n`按springs的次序返回，
        #: 而springs只含当时engaged的那些。用求解后的新活动集去索引会错位。
        trials: list[tuple[float, float, float]] = []
        spring_cursor = 0
        stick_forces = (
            TangentialStickSpring(springs=springs).tangential_force_n(solved.state)
            if springs
            else ()
        )
        for index in range(count):
            if engaged[index]:
                trials.append(stick_forces[spring_cursor])
                spring_cursor += 1
            else:
                trials.append((0.0, 0.0, 0.0))

        for index, point in enumerate(contacts):
            normal_force = point.normal_force_of(solved.state)
            normal_forces[index] = normal_force
            engaged[index] = normal_force > 0.0
            trial = trials[index] if engaged[index] else (0.0, 0.0, 0.0)
            outcome = coulomb_return_map(
                trial_force_n=trial,
                normal_force_n=normal_force,
                friction_coefficient=point.friction_coefficient,
                tangential_stiffness_n_per_mm=point.tangential_stiffness_n_per_mm,
            )
            outcomes[index] = outcome
            anchors[index] = tuple(
                anchors[index][axis] + outcome.anchor_correction_mm[axis] for axis in range(3)
            )
            excesses[index] = (
                math.sqrt(sum(value * value for value in trial))
                - point.friction_coefficient * normal_force
            )
        if max(excesses) <= yield_tol_n:
            break
    else:
        if require_pass_convergence:
            worst = max(range(count), key=lambda index: excesses[index])
            raise ContactError(
                f"预测-修正走满{max_passes}趟仍未收敛：最大屈服残差"
                f"{excesses[worst]:.6g} N出现在contacts[{worst}]，容差{yield_tol_n:.6g} N"
            )

    assert solved is not None
    updated = list(solved.state.vector)
    slips: list[float] = []
    for index, point in enumerate(contacts):
        anchor = anchors[index]
        #: 分离时清切向历史——与单槽位版本同一条理由（幽灵弹簧与凭空的摩擦功）。
        if not engaged[index]:
            anchor = (0.0, 0.0, 0.0)
        for axis in range(3):
            updated[point.slot.anchor_base + axis] = anchor[axis]
        updated[point.slot.active_index] = 0.0 if normal_forces[index] == 0.0 else 1.0
        outcome = outcomes[index]
        assert outcome is not None
        updated[point.slot.regime_index] = outcome.regime
        slips.append(
            0.0
            if not engaged[index]
            else math.sqrt(
                sum((anchor[axis] - origins[index][axis]) ** 2 for axis in range(3))
            )
        )

    return MultiContactStep(
        state=State(layout=contact_layout.layout, vector=tuple(updated)),
        normal_force_n=tuple(normal_forces),
        tangential_force_n=tuple(
            outcome.tangential_force_n for outcome in outcomes if outcome is not None
        ),
        regime=tuple(outcome.regime for outcome in outcomes if outcome is not None),
        slip_increment_mm=tuple(slips),
        passes=passes,
        max_yield_excess_n=max(excesses),
    )


@dataclass(frozen=True)
class PenaltySphereContact:
    """两球之间的罚函数法向接触：``U = Σ ½·k·g²``（仅``g < 0``），单位N·mm。

    ``g = |x_j − x_i| − (r_i + r_j)``是间隙：两球心距减去半径之和。

    ## 与`PenaltyNormalContact`（半空间）的差别：**法向随位形转**

    半空间的法向是常量，所以那个项的Hessian只有``k·(n⊗n)``一块。
    这里法向是``d = (x_j − x_i)/|x_j − x_i|``，**它随位置变**，
    于是Hessian多出一块**几何刚度**：

        H = k·(d⊗d) + (k·g/L)·(I − d⊗d)

    第二块在接触时``g < 0``故**是负的**——横向softening。
    这与`AxialStretch`压缩时的几何刚度同源（0046屈曲案例里的临界载荷正是它给的），
    **漏掉它梯度照样对、平衡点照样对，只有收敛速度与稳定性判据会变**。
    因此本项的Hessian必须被有限差分单独验一次。

    ## 这一项让"多体接触"第一次成立

    此前接触只发生在**节点与固定半空间**之间——固定面不参与自由度。
    本项两端都是自由度，于是Hessian有**跨节点的耦合块**，
    而那是多体接触与单体接触的真正分界。
    """

    name: str = "sphere_contact"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点i, 节点j, 半径之和mm, 罚刚度N/mm)
    pairs: tuple[tuple[int, int, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ContactError("sphere_contact needs at least one pair")
        for i, j, radii_sum, stiffness in self.pairs:
            for index in (i, j):
                #: **两个同门（半空间、粘着）都校验了，只有这里漏了。**
                #: 2026-08-06对抗审核实测：``node = -1``被接受、``node_index_bound()``
                #: 返回1所以装配门放行，而``vector[-3:]``读的正是**接触锚点槽**——
                #: 算出316681 N·mm的能量，全部由历史值来。
                #: **这逐字就是`EnergyRegistry.assert_within_nodes`docstring描述的
                #: 那个失败模式，而那道门只挡上界。**
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ContactError(
                        f"sphere contact node index must be a nonnegative int: {index!r}"
                    )
            if i == j:
                raise ContactError(f"a sphere cannot contact itself: node {i}")
            if not (radii_sum > 0.0 and math.isfinite(radii_sum)):
                raise ContactError(f"radii sum must be positive: {radii_sum!r}")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")

    @classmethod
    def _from_validated_pairs(
        cls, pairs: tuple[tuple[int, int, float, float], ...]
    ) -> PenaltySphereContact:
        """由同包内已验证装配层构造；调用方承担全部``__post_init__``不变量。"""

        term = object.__new__(cls)
        object.__setattr__(term, "name", "sphere_contact")
        object.__setattr__(term, "pairs", pairs)
        return term

    def node_index_bound(self) -> int:
        return max(max(i, j) for i, j, _, _ in self.pairs) + 1

    @staticmethod
    def _pair_state(vector: tuple[float, ...], pair) -> tuple[float, float, tuple[float, ...]]:
        """返回``(间隙g, 心距L, 单位方向d)``。``d``由i指向j。"""

        i, j, radii_sum, _ = pair
        delta = tuple(vector[3 * j + axis] - vector[3 * i + axis] for axis in range(3))
        length = math.sqrt(sum(component * component for component in delta))
        if length == 0.0:
            raise ContactError(
                f"spheres {i} and {j} share a centre — 方向未定义，能量在此不可微"
            )
        return length - radii_sum, length, tuple(c / length for c in delta)

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for pair in self.pairs:
            gap, _, _ = self._pair_state(state.vector, pair)
            if gap < 0.0:
                total += 0.5 * pair[3] * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for pair in self.pairs:
            i, j, _, stiffness = pair
            gap, _, direction = self._pair_state(state.vector, pair)
            if gap < 0.0:
                force = stiffness * gap
                for axis in range(3):
                    result[3 * i + axis] -= force * direction[axis]
                    result[3 * j + axis] += force * direction[axis]
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
        entries: list[tuple[int, int, float]] = []
        for pair in self.pairs:
            i, j, _, stiffness = pair
            gap, length, direction = self._pair_state(state.vector, pair)
            if gap >= 0.0:
                continue
            transverse = stiffness * gap / length
            for a in range(3):
                for b in range(3):
                    outer = direction[a] * direction[b]
                    identity = 1.0 if a == b else 0.0
                    block = stiffness * outer + transverse * (identity - outer)
                    entries.append((3 * i + a, 3 * i + b, block))
                    entries.append((3 * j + a, 3 * j + b, block))
                    entries.append((3 * i + a, 3 * j + b, -block))
                    entries.append((3 * j + a, 3 * i + b, -block))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for pair in self.pairs:
            i, j, _, stiffness = pair
            gap, _, direction = self._pair_state(vector, pair)
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    force = stiffness * gap
                    for axis in range(3):
                        gradient[3 * i + axis] -= force * direction[axis]
                        gradient[3 * j + axis] += force * direction[axis]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def contact_force_n(self, state: State) -> tuple[float, ...]:
        """每对的法向接触力``|k·g|``（分离时为0）。与半空间项同理：**平衡时精确**。"""

        forces = []
        for pair in self.pairs:
            gap, _, _ = self._pair_state(state.vector, pair)
            forces.append(pair[3] * -gap if gap < 0.0 else 0.0)
        return tuple(forces)


@dataclass(frozen=True)
class PenaltyCylinderContact:
    """罚函数式法向接触（**有限长圆柱的侧面**）：``U = Σ ½·k·g²``（仅活动），单位N·mm。

    决策0062轨道甲第一片，守能力位S6.5（带材过导向轮的接触）。

    记``d = x − p``（``p``是轴上一点）、``s = d·a``（轴向坐标，``a``是轴单位方向）、
    ``w = d − s·a``（径向矢量）、``ρ = |w|``。则

        g = ρ − (R + r_节点)，   n = w/ρ

    ``g > 0``分离、``g < 0``穿透；``n``是**由轴指向外**的径向单位矢量，恒有``n·a = 0``。

    ## 活动条件是**两条**，不是一条

    ``g < 0``**且**``|s| ≤ half_width``。第二条不是可选的修饰：
    本项只表达**侧面**，节点轴向越出筒宽之后侧面在那里根本不存在，
    继续按``g``出力等于凭空造一个无限长圆柱。

    **这条边界是硬切，力在``|s| = half_width``处从``k|g|``跳到``0``。**
    不平滑不是疏忽——把它做成圆角要引入端面圆环与环面两套几何，
    而那是0062第七节明确不做的。代价如实登记：**节点贴着端沿时牛顿会抖**，
    所以本项给出``axial_clearance_mm``，案例必须为它设门（`cases/`里的门判它，
    不判位置）。**"力会跳"如果没有门看着，就等于一条没有门的分支**（plans/09教训三）。

    ## Hessian：几何刚度**只出现在周向**

    ``∂g/∂x = n``、``∂n/∂x = (P − n⊗n)/ρ``（``P = I − a⊗a``），而
    ``P − n⊗n = t⊗t``，``t = a × n``是周向单位矢量。于是

        H = k·(n⊗n) + (k·g/ρ)·(t⊗t)

    第二块在接触时``g < 0``故**是负的**——**周向**softening，与`PenaltySphereContact`
    的横向softening同源（那里是``(kg/L)(I − d⊗d)``，各向同性的两个横向）。

    **这里只有一个横向而不是两个**，因为沿轴移动不改变``ρ``：
    ``H``在``a``方向上恒为零。这不是近似，是圆柱的几何——
    **它同时是一条可测判据**（沿轴的方向导数必须恰为0），本模块的门判它。

    ## 精度的地板：``k·ulp(R)``——**半空间那条"力精确"在这里不成立**

    `PenaltyNormalContact`记着"跨六个数量级刚度，法向力一个ulp都不动"。
    **那条不能搬过来。** 半空间的间隙是``z − 0``，圆柱的间隙是``ρ − R``——
    后者是两个``O(R)``量相减，**灾难性相消**。于是：

    * 可达残差的地板是``0.5·k·ulp(R)``（2026-08-17实测两半径×五档刚度共10组，
      比值全在``[0.15, 0.48]``，无一超过0.5）；
    * 法向力的可达精度因此是``k·ulp(R)``的**绝对**量，不是相对量。

    绕线机导轮``R = 50 mm``、``k = 1e4 N/mm``时地板是``3.6e-11 N``，
    而链路上要分辨的张力是10—30 N——**够用，但求解器容差必须按它定**。

    **它同时是一条设计约束**：想把残差压到``1e-13``就得把``k``压到``30 N/mm``以下，
    那时穿透``N/k``约0.65 mm。**精度与穿透是同一个旋钮的两头**，
    这一点在半空间上看不出来，因为那里没有``R``。

    ## 轴上奇点：失败关闭

    ``ρ = 0``（节点正好在轴上）时法向**没有定义**，本项当场抛。
    它不是数值噪声：罚接触的穿透量级是``O(N/k)``，穿到轴上意味着模型早已离开
    定义域，此时静默取一个方向比抛更坏——那个方向会被牛顿当成真的。

    近轴处法向的相对精度约``eps·|d|/ρ``，这条写在这里而不是靠使用者猜。
    """

    name: str = "cylinder_contact"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点索引, 轴上一点mm, 轴单位方向, 圆柱半径mm, 轴向半宽mm, 罚刚度N/mm, 节点半径mm)
    #: **节点半径必填**，质点写``0.0``——理由同`PenaltyNormalContact`第二节。
    cylinders: tuple[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            float,
            float,
        ],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.cylinders:
            raise ContactError("cylinder_contact needs at least one cylinder")
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(
                    f"cylinder contact node index must be a nonnegative int: {node!r}"
                )
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"cylinder axis point must be a finite 3-vector: {point!r}")
            if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
                raise ContactError(f"cylinder axis must be a finite 3-vector: {axis!r}")
            norm = math.sqrt(sum(component * component for component in axis))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(
                    f"cylinder axis must be a unit vector (|a| = {norm!r}) — "
                    "不归一化会同时改掉轴向投影与径向距离，而调用方以为自己给的是几何"
                )
            if not (radius > 0.0 and math.isfinite(radius)):
                raise ContactError(f"cylinder radius must be positive: {radius!r}")
            if not (half_width > 0.0 and math.isfinite(half_width)):
                raise ContactError(f"cylinder half width must be positive: {half_width!r}")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")
            if node_radius < 0.0 or not math.isfinite(node_radius):
                raise ContactError(
                    f"contact radius must be finite and nonnegative: {node_radius!r}"
                )

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _, _, _, _ in self.cylinders) + 1

    @staticmethod
    def _frame(
        vector: tuple[float, ...],
        node: int,
        point: tuple[float, float, float],
        axis: tuple[float, float, float],
        radius: float,
        node_radius: float,
    ) -> tuple[float, float, float, tuple[float, float, float]]:
        """返回``(间隙g, 轴向坐标s, 径向距离ρ, 径向单位法向n)``。

        ``ρ = 0``时抛——理由见类docstring末节。
        """

        base = 3 * node
        delta = tuple(vector[base + component] - point[component] for component in range(3))
        axial = sum(delta[component] * axis[component] for component in range(3))
        radial = tuple(delta[component] - axial * axis[component] for component in range(3))
        distance = math.sqrt(sum(component * component for component in radial))
        if distance == 0.0:
            raise ContactError(
                f"node {node} sits on the cylinder axis (rho = 0) — 法向没有定义。"
                "罚接触的穿透量级是O(N/k)，穿到轴上说明模型已离开定义域；"
                "此处静默取一个方向会被牛顿当成真的"
            )
        normal = tuple(component / distance for component in radial)
        return distance - (radius + node_radius), axial, distance, normal  # type: ignore[return-value]

    @classmethod
    def _is_active(cls, gap: float, axial: float, half_width: float) -> bool:
        """活动条件是**两条**：穿透且轴向仍在筒宽内。"""

        return gap < 0.0 and abs(axial) <= half_width

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            if self._is_active(gap, axial, half_width):
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, normal = self._frame(
                state.vector, node, point, axis, radius, node_radius
            )
            if self._is_active(gap, axial, half_width):
                force = stiffness * gap
                base = 3 * node
                for component in range(3):
                    result[base + component] += force * normal[component]
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
        """``k·(n⊗n) + (k·g/ρ)·(t⊗t)``，``t = a × n``。**分离的接触一个非零项都不出**。"""

        entries: list[tuple[int, int, float]] = []
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, distance, normal = self._frame(
                state.vector, node, point, axis, radius, node_radius
            )
            if not self._is_active(gap, axial, half_width):
                continue
            circumferential = (
                axis[1] * normal[2] - axis[2] * normal[1],
                axis[2] * normal[0] - axis[0] * normal[2],
                axis[0] * normal[1] - axis[1] * normal[0],
            )
            geometric = stiffness * gap / distance
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    value = stiffness * normal[a] * normal[b] + geometric * (
                        circumferential[a] * circumferential[b]
                    )
                    entries.append((base + a, base + b, value))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。"""

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, normal = self._frame(vector, node, point, axis, radius, node_radius)
            if self._is_active(gap, axial, half_width):
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    force = stiffness * gap
                    base = 3 * node
                    for component in range(3):
                        gradient[base + component] += force * normal[component]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def normal_force_n(self, state: State) -> tuple[float, ...]:
        """每个圆柱上的法向力大小``N = k·|g|``（不活动时为0）。

        与半空间项同理：**平衡时它精确等于理论法向力，与罚刚度无关**。
        绞盘判据要用的正是它，所以它是公开面。
        """

        forces = []
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            forces.append(
                stiffness * -gap if self._is_active(gap, axial, half_width) else 0.0
            )
        return tuple(forces)

    def axial_clearance_mm(self, state: State) -> tuple[float, ...]:
        """每个圆柱上``half_width − |s|``：**离端沿还有多远**，可正可负。

        本项在``|s| > half_width``处力硬跳到零（类docstring第二节），
        所以案例必须为这个量设门。**给出它，是为了让那条不连续有门看着而不是靠记性。**
        """

        clearances = []
        for node, point, axis, radius, half_width, _, node_radius in self.cylinders:
            _, axial, _, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            clearances.append(half_width - abs(axial))
        return tuple(clearances)

    def radial_distance_mm(self, state: State) -> tuple[float, ...]:
        """每个圆柱上的``ρ``。近轴处法向精度约``eps·|d|/ρ``，门判它才能发现失精。"""

        distances = []
        for node, point, axis, radius, _, _, node_radius in self.cylinders:
            _, _, distance, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            distances.append(distance)
        return tuple(distances)

    def outward_normal(self, state: State) -> tuple[tuple[float, float, float], ...]:
        """每个圆柱上的径向单位外法向``n``。

        摩擦项要的正是它：`TangentialStickSpring`吃`NormalSource`，
        曲面接触必须给**随位形转**的那一种（0050第四节实测：法向不随位形转时
        一趟预测-修正就够，转时不够）。本方法是绞盘接线的接口。
        """

        normals = []
        for node, point, axis, radius, _, _, node_radius in self.cylinders:
            _, _, _, normal = self._frame(state.vector, node, point, axis, radius, node_radius)
            normals.append(normal)
        return tuple(normals)


@dataclass(frozen=True)
class LinearNormalDashpot:
    """半空间与球-球的线性法向dashpot，按**合力归零**截断。

    本项只给耗散力；弹簧势能仍由``PenaltyNormalContact``或
    ``PenaltySphereContact``给。先算弹簧压缩力``N_s``，再令
    ``N = max(0, N_s − c·g_dot)``，本项贡献``N−N_s``。因此出射阶段允许
    dashpot抵消弹簧，但总接触力永不成为拉力。
    """

    name: str = "normal_dashpot"
    kind: ClassVar[Literal["dissipation"]] = DISSIPATION
    #: (节点, 面上一点, 外法向, 弹簧刚度N/mm, 阻尼N·s/mm, 半径mm)
    planes: tuple[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            float,
        ],
        ...,
    ] = ()
    #: (节点i, 节点j, 半径和mm, 弹簧刚度N/mm, 阻尼N·s/mm)
    sphere_pairs: tuple[tuple[int, int, float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.planes and not self.sphere_pairs:
            raise ContactError("normal_dashpot needs at least one plane or sphere pair")
        for node, point, normal, stiffness, damping, radius in self.planes:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"dashpot node index must be a nonnegative int: {node!r}")
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"dashpot plane point must be a finite 3-vector: {point!r}")
            if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
                raise ContactError(f"dashpot normal must be a finite 3-vector: {normal!r}")
            norm = math.sqrt(sum(value * value for value in normal))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(f"dashpot normal must be a unit vector (|n| = {norm!r})")
            self._validate_coefficients(stiffness, damping, radius)
        for i, j, radii_sum, stiffness, damping in self.sphere_pairs:
            for node in (i, j):
                if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                    raise ContactError(
                        f"dashpot sphere node index must be a nonnegative int: {node!r}"
                    )
            if i == j:
                raise ContactError(f"a sphere cannot damp contact with itself: node {i}")
            self._validate_coefficients(stiffness, damping, radii_sum)

    @classmethod
    def _from_validated_parts(
        cls,
        *,
        planes: tuple[
            tuple[
                int,
                tuple[float, float, float],
                tuple[float, float, float],
                float,
                float,
                float,
            ],
            ...,
        ],
        sphere_pairs: tuple[tuple[int, int, float, float, float], ...],
    ) -> LinearNormalDashpot:
        """由同包内已验证装配层构造；调用方承担全部``__post_init__``不变量。"""

        term = object.__new__(cls)
        object.__setattr__(term, "name", "normal_dashpot")
        object.__setattr__(term, "planes", planes)
        object.__setattr__(term, "sphere_pairs", sphere_pairs)
        return term

    @staticmethod
    def _validate_coefficients(stiffness: float, damping: float, radius: float) -> None:
        if not math.isfinite(stiffness) or stiffness <= 0.0:
            raise ContactError(f"dashpot stiffness must be positive: {stiffness!r}")
        if not math.isfinite(damping) or damping <= 0.0:
            raise ContactError(f"dashpot damping must be positive: {damping!r}")
        if not math.isfinite(radius) or radius < 0.0:
            raise ContactError(f"dashpot radius must be finite and nonnegative: {radius!r}")

    def node_index_bound(self) -> int:
        indices = [plane[0] for plane in self.planes]
        for i, j, _, _, _ in self.sphere_pairs:
            indices.extend((i, j))
        return max(indices) + 1

    @staticmethod
    def _damping_magnitude(
        *, gap_mm: float, gap_rate_mm_per_s: float, stiffness: float, damping: float
    ) -> float:
        if gap_mm >= 0.0:
            return 0.0
        spring = -stiffness * gap_mm
        total = max(0.0, spring - damping * gap_rate_mm_per_s)
        return total - spring

    def force_and_power(
        self, state: State, velocity: Sequence[float], context: EnergyContext
    ) -> tuple[Vector, float]:
        if len(velocity) != len(state.vector):
            raise ContactError("dashpot velocity and state vector must have the same length")
        force = [0.0] * len(state.vector)
        power = 0.0
        for node, point, normal, stiffness, damping, radius in self.planes:
            gap = PenaltyNormalContact._gap_mm(state.vector, node, point, normal, radius)
            base = 3 * node
            gap_rate = sum(velocity[base + axis] * normal[axis] for axis in range(3))
            magnitude = self._damping_magnitude(
                gap_mm=gap,
                gap_rate_mm_per_s=gap_rate,
                stiffness=stiffness,
                damping=damping,
            )
            for axis in range(3):
                force[base + axis] += magnitude * normal[axis]
            power += max(0.0, -magnitude * gap_rate)

        for i, j, radii_sum, stiffness, damping in self.sphere_pairs:
            delta = tuple(
                state.vector[3 * j + axis] - state.vector[3 * i + axis]
                for axis in range(3)
            )
            length = math.sqrt(sum(value * value for value in delta))
            if length == 0.0:
                raise ContactError(
                    f"spheres {i} and {j} share a centre — dashpot direction is undefined"
                )
            direction = tuple(value / length for value in delta)
            gap = length - radii_sum
            gap_rate = sum(
                (velocity[3 * j + axis] - velocity[3 * i + axis]) * direction[axis]
                for axis in range(3)
            )
            magnitude = self._damping_magnitude(
                gap_mm=gap,
                gap_rate_mm_per_s=gap_rate,
                stiffness=stiffness,
                damping=damping,
            )
            for axis in range(3):
                component = magnitude * direction[axis]
                force[3 * i + axis] -= component
                force[3 * j + axis] += component
            power += max(0.0, -magnitude * gap_rate)
        return tuple(force), power
