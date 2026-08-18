"""准静态接触步进器：单槽位与多槽位。

多槽位那条兑现的是0050第一节挂了11天的欠账。两条纪律：
全部粘着弹簧进**同一个**`TangentialStickSpring`并按声明顺序排列（求和次序是形制）；
收敛判据是``max(excesses) <= yield_tol_n``（**全部**，不是任一）。
单槽位那条**一个字没动**，忠实性由12组合的逐位对拍门守着。

拆分自原`contact.py`（2026-08-17）——**函数体逐字节未动**。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from physics_engine.contact.errors import ContactError
from physics_engine.contact.friction import (
    FrictionEllipse,
    FrictionEllipseSpec,
    FrictionOutcome,
    TangentialStickSpring,
    anisotropic_return_map,
    coulomb_return_map,
    yield_excess_n,
)
from physics_engine.contact.layout import (
    REGIME_SEPARATED,
    REGIME_STICK,
    SLOT_WIDTH,
    ContactLayout,
    ContactSlot,
    NormalSource,
)
from physics_engine.energies import EnergyContext
from physics_engine.rotation import (
    MaterialPoint,
    MaterialPointStickSpring,
    RotationStickCoupling,
    StickSpring,
)
from physics_engine.state import State


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


def _yield_surface(
    *,
    friction_coefficient: float | None,
    friction_ellipse: FrictionEllipseSpec | None,
    where: str,
) -> None:
    """``friction_coefficient``与``friction_ellipse``**必须恰好给一个**。

    两个都给会出现两份μ，而"哪一份说了算"只能靠读实现——
    本仓在`PenaltyAnnulusLimit`上吃过同源的亏（朝向被编码进另一个量，
    单元门永远抓不到）。两个都不给则连屈服面都没有，那不是默认，是漏参。

    **不给`friction_coefficient`一个默认标量值**：默认值一旦有内容，
    "忘了传"就变成"用了一个没人声明过的摩擦系数"，而它会一路静默算到底。
    """

    if (friction_coefficient is None) == (friction_ellipse is None):
        raise ContactError(
            f"{where}: friction_coefficient与friction_ellipse必须**恰好给一个**"
            f"（现在给的是 coefficient={friction_coefficient!r}、"
            f"ellipse={friction_ellipse!r}）——两个都给等于有两份μ而哪份说了算只能靠读实现；"
            "两个都不给等于没有屈服面"
        )


def _return_map(
    *,
    trial: tuple[float, float, float],
    normal_force_n: float,
    friction_coefficient: float | None,
    ellipse: FrictionEllipse | None,
    tangential_stiffness_n_per_mm: float,
) -> tuple[FrictionOutcome, float]:
    """一次return-map + 它的屈服残差。**两条屈服面在这里合流，且只在这里。**

    ``ellipse is None``那一支是**旧路径逐字未动**——包括``excess``那串运算的
    求和次序（spec/12第3.3节：次序是形制）。这是"不给椭圆时逐字节不变"的构造性保证：
    不是"算出来恰好相等"，是**同一串代码**。

    椭圆那一支在``μ_∥ == μ_⊥``时由`anisotropic_return_map`与`yield_excess_n`
    各自转交回圆的写法，故它也是逐位退化的（决策0068第4.1节同一条纪律）。
    """

    if ellipse is None and friction_coefficient is None:
        #: 声明了椭圆、但这一趟没有屈服面对象。多槽位版本只给**装配时活动**的点
        #: 造椭圆（理由见那里：给分离的点求法向等于凭空加一条报错路径），
        #: 于是"装配时就分离、求解后仍分离"的点会走到这里。
        #: ``N = 0``上两条屈服面给的是**同一个**答案——分离、零力、零修正，
        #: 屈服残差等于试探力的模而那也是零——**所以这里不需要知道是哪一条**。
        assert normal_force_n == 0.0, "没有屈服面却拿到了非零法向力"
        zero = (0.0, 0.0, 0.0)
        return FrictionOutcome(zero, REGIME_SEPARATED, zero), 0.0
    if ellipse is None:
        assert friction_coefficient is not None
        outcome = coulomb_return_map(
            trial_force_n=trial,
            normal_force_n=normal_force_n,
            friction_coefficient=friction_coefficient,
            tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
        )
        excess = (
            math.sqrt(sum(value * value for value in trial))
            - friction_coefficient * normal_force_n
        )
        return outcome, excess
    outcome = anisotropic_return_map(
        trial_force_n=trial,
        normal_force_n=normal_force_n,
        ellipse=ellipse,
        tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
    )
    excess = yield_excess_n(
        trial_force_n=trial, normal_force_n=normal_force_n, ellipse=ellipse
    )
    return outcome, excess


#: 接触点在力学状态里的那一端：**节点号**（质点），或**物质点**（刚体上带杠杆臂的点）。
#:
#: 决策0080第一节裁的接口形态。`MaterialPoint`本身就是`int`的推广——
#: ``MaterialPoint(n, (0,0,0), None)``与``n``描述的是同一个东西——
#: 所以它进的是**同一个参数**，而不是另开一个并列参数让两者"哪个说了算"靠读实现。
#: 那正是`_yield_surface`挡的同一类错（`PenaltyAnnulusLimit`的前科）。
ContactEnd = int | MaterialPoint


def _check_material_point(
    point: MaterialPoint,
    *,
    contact_layout: ContactLayout,
    vector: tuple[float, ...],
    where: str,
) -> None:
    """物质点的落位校验——**把上面那四条推广到"转动块也不许写进别人的地盘"**。

    节点那一半与`int`口径逐字相同。新出来的是**转动块**那一半，而它挡的错更狠：

    * 转动块落进**节点块** → 牛顿把某个节点坐标当转角解，位形被静默改掉；
    * 转动块与**锚点槽**重叠 → 牛顿把锚点当自由度解。
      锚点是历史（0033），**历史被当成未知数去解就不再是历史**——
      而这一条不会抛任何异常，它只会安静地给出一个"收敛了"的错答案。

    第二条是`int`口径下不存在的新危险：节点块与锚点槽之间隔着一条
    "槽必须在节点块之后"的检查就够了，而转动块**两边都要挡**。
    """

    node_dof_count = contact_layout.layout.node_dof_count
    if 3 * point.node + 3 > node_dof_count:
        raise ContactError(
            f"{where}: 物质点的节点{point.node}落在节点块之外"
            f"（节点块只有{node_dof_count}个自由度）——再往后是锚点槽"
        )
    base = point.rotation_base
    if base is None:
        return
    if base < node_dof_count:
        raise ContactError(
            f"{where}: rotation_base={base}落在节点块之内"
            f"（节点块{node_dof_count}个自由度）——牛顿会把某个节点坐标当转角解"
        )
    if base + 3 > len(vector):
        raise ContactError(
            f"{where}: rotation_base={base}的转动块越过了状态向量末尾"
            f"（需要下标{base + 2}，向量只有{len(vector)}个）"
        )
    for slot in contact_layout.slots:
        if base < slot.base + SLOT_WIDTH and slot.base < base + 3:
            raise ContactError(
                f"{where}: rotation_base={base}的转动块与锚点槽"
                f"{slot.pair_id!r}点{slot.point_index}（[{slot.base}, "
                f"{slot.base + SLOT_WIDTH})）重叠"
                "——**牛顿会把锚点当自由度解，而锚点是历史**（决策0033）"
            )


def _check_end(
    end: ContactEnd,
    counterpart: MaterialPoint | None,
    *,
    contact_layout: ContactLayout,
    vector: tuple[float, ...],
    where: str,
) -> None:
    """一个接触端的落位校验。``int``那一支**逐字保留2026-08-12补的四条**。"""

    if isinstance(end, MaterialPoint):
        _check_material_point(
            end, contact_layout=contact_layout, vector=vector, where=f"{where} first"
        )
        if counterpart is not None:
            _check_material_point(
                counterpart,
                contact_layout=contact_layout,
                vector=vector,
                where=f"{where} second",
            )
        return
    if counterpart is not None:
        raise ContactError(
            f"{where}: 给了counterpart却把这一端写成节点号{end!r}——"
            "对边只能是`MaterialPoint`，而`int`那一支按构造是"
            "**节点对世界锚点**（`TangentialStickSpring`只做这一种）"
        )
    node_dof_count = contact_layout.layout.node_dof_count
    if not isinstance(end, int) or isinstance(end, bool) or end < 0:
        raise ContactError(f"contact node index must be a nonnegative int: {end!r}")
    if 3 * end + 3 > node_dof_count:
        raise ContactError(
            f"{where}: node {end} 落在节点块之外"
            f"（节点块只有{node_dof_count}个自由度）"
            "——**再往后是锚点槽，写进去就是改别人的历史**"
        )


def _rotates(end: ContactEnd, counterpart: MaterialPoint | None) -> bool:
    if not isinstance(end, MaterialPoint):
        return False
    if end.rotation_base is not None:
        return True
    return counterpart is not None and counterpart.rotation_base is not None


def _assemble_stick(entries):
    """把这一趟的粘着弹簧装成能量项，并给出"每个接触点的试探力"。

    ``entries``是``(end, counterpart, anchor, normal, stiffness)``的序列，
    **按声明次序**——次序即形制（spec/12第3.3节），所以它一路传到项里面。

    ## 三种项，装配次序是**声明的**：legacy → 物质点 → 转动增量

    | 端的形态 | 项 |
    |---|---|
    | ``int`` | `friction.TangentialStickSpring`（**一个**，含全部legacy弹簧） |
    | `MaterialPoint`，两端都不转 | `rotation.MaterialPointStickSpring` |
    | `MaterialPoint`，有一端转 | 上一行 **＋** `rotation.RotationStickCoupling` |

    全是``int``时返回的元组是``(TangentialStickSpring(...),)``——
    与2026-08-17那版**同一个对象、同一个求和次序**，
    所以既有调用方的产物逐位不变不是"算出来相等"，是**同一串代码**。

    ## 试探力的符号：**两个类差一个负号，而这里必须统一到``+k·P(d)``**

    `TangentialStickSpring.tangential_force_n`给``+k·P(x − a)``；
    `MaterialPointStickSpring`/`RotationStickCoupling`给的是**作用在first端的力**
    ``−k·P(d)``。return-map拿试探力做两件事：判模长（符号无关）、
    **定锚点修正的方向**（符号要命）。

    锚点修正的语义是"**锚点追着物质点走**"——滑移把零应力位形挪到当前位形那一侧，
    所以修正必须沿``+P(d)``。取成``−P(d)``会让锚点往反方向跑，
    于是下一步的弹性伸长**变大**而不是被削平，屈服残差发散而不是收敛，
    **而这一切都不会报错**：它只会给出一个越滑越紧的摩擦。
    因此这里对物质点那两支取负号，把两条路径统一到同一个符号约定上。
    """

    legacy_positions: list[int] = []
    legacy_springs: list[tuple[int, tuple[float, float, float], tuple[float, float, float], float]] = []
    material_positions: list[int] = []
    material_springs: list[StickSpring] = []
    coupling_positions: list[int] = []
    coupling_springs: list[StickSpring] = []

    for position, (end, counterpart, anchor, normal, stiffness) in enumerate(entries):
        if not isinstance(end, MaterialPoint):
            legacy_positions.append(position)
            legacy_springs.append((end, anchor, normal, stiffness))
            continue
        spring = StickSpring(
            first=end,
            normal=normal,
            stiffness_n_per_mm=stiffness,
            anchor_mm=anchor,
            second=counterpart,
        )
        material_positions.append(position)
        material_springs.append(spring)
        #: **只把有转动端的弹簧交给耦合项**：`RotationStickCoupling`对
        #: "一端都不转"的弹簧是失败关闭的（那是个恒零项，声明它只会让读者
        #: 以为转动接上了）。混着声明因此是允许的，而不是被这条检查挡住。
        if _rotates(end, counterpart):
            coupling_positions.append(position)
            coupling_springs.append(spring)

    terms: list[object] = []
    legacy_term = (
        TangentialStickSpring(springs=tuple(legacy_springs)) if legacy_springs else None
    )
    if legacy_term is not None:
        terms.append(legacy_term)
    material_term = (
        MaterialPointStickSpring(springs=tuple(material_springs))
        if material_springs
        else None
    )
    if material_term is not None:
        terms.append(material_term)
    coupling_term = (
        RotationStickCoupling(springs=tuple(coupling_springs)) if coupling_springs else None
    )
    if coupling_term is not None:
        terms.append(coupling_term)

    #: 转动端的弹簧在``coupling_term``里排第几个。**每个位置只被写一次**——
    #: 起草时写的是"先按不含转动的项写一遍、再让转动项覆盖"，
    #: 那一版的取舍分支**观测不到**：把它拆掉，覆盖那一步照样给出同一个答案。
    #: 注错验证第一轮就是这么抓出来的（0080第七节空门一）。
    #: **一个观测不到的分支不是判据，是障眼法。**
    coupling_order = {
        position: order for order, position in enumerate(coupling_positions)
    }

    def trial_of(state: State) -> tuple[tuple[float, float, float], ...]:
        result: list[tuple[float, float, float] | None] = [None] * len(entries)
        if legacy_term is not None:
            for position, force in zip(
                legacy_positions, legacy_term.tangential_force_n(state), strict=True
            ):
                result[position] = force
        if material_term is not None:
            plain = material_term.tangential_force_n(state)
            rotated = (
                coupling_term.tangential_force_n(state)
                if coupling_term is not None
                else ()
            )
            for offset, position in enumerate(material_positions):
                order = coupling_order.get(position)
                force = plain[offset] if order is None else rotated[order]
                result[position] = (-force[0], -force[1], -force[2])
        assert all(value is not None for value in result)
        return tuple(result)  # type: ignore[arg-type]

    return tuple(terms), trial_of


def advance_contact_quasistatic(
    *,
    registry_without_stick,
    context: EnergyContext,
    contact_layout: ContactLayout,
    slot: ContactSlot,
    vector: tuple[float, ...],
    node: ContactEnd,
    counterpart: MaterialPoint | None = None,
    normal: NormalSource,
    normal_force_of: Callable[[State], float],
    tangential_stiffness_n_per_mm: float,
    friction_coefficient: float | None = None,
    friction_ellipse: FrictionEllipseSpec | None = None,
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

    ## 屈服面是**圆还是椭圆，由调用方声明**（plans/15第1.6条，2026-08-17补）

    ``friction_coefficient``（圆）与``friction_ellipse``（椭圆，决策0068）
    **恰好给一个**。给椭圆时本函数每趟用**这一趟的法向**造一个`FrictionEllipse`——
    与粘着弹簧用的是同一个法向值，那是试探力落在椭圆所在切平面里的前提。

    **默认没有变**：不给`friction_ellipse`时走的是与2026-08-17之前
    逐字相同的那串代码（见`_return_map`）。0068第五节第2条裁的
    "不切默认"在这里继续成立——**能走这条路**与**默认走这条路**是两件事，
    而今天仍然没有消费方声明过两个系数。

    ## ``node``可以是**物质点**（决策0080，本函数第一次与转动块联动）

    刚体上的接触点不是质心，是"质心 + 参考构型下的杠杆臂 + 转动基址"，
    即`rotation.MaterialPoint`。``node``因此吃``int | MaterialPoint``，
    对边由``counterpart``给（``None``时另一端是世界锚点）。

    **``int``那一支一个字节都没变**：装配的项、求和次序、试探力的符号
    全部是同一串代码（见`_assemble_stick`），所以既有调用方的产物逐位不变
    是构造性的，不是对拍出来的。

    ### 锚点记的是什么，滑移按什么算

    这是本次扩展**唯一**改了语义的地方，而它只对物质点那一支成立：

    > 锚点``a``记的是**两个物质点世界位置之差**在上一次粘住时的值，
    > 即``a = (x₁ + R₁ℓ₁) − (x₂ + R₂ℓ₂)``；
    > 滑移量是``|Δa|``，方向沿``P(d₀+u)``。

    ``R(θ)ℓ``在里面，所以**转动直接进滑移**：同一个物质点转过去之后世界位置变了，
    那一段位移与质心平移一样是真实的相对滑动，一样要耗散。
    按节点位置算等于把杠杆臂上的那一段抹掉——**摩擦耗散会系统性偏小，
    而所有法向量、所有力都还是对的**，那是最难看的一类错。
    """

    from physics_engine.energies import EnergyRegistry
    from physics_engine.solve import solve_equilibrium

    if max_passes < 1:
        raise ContactError(f"max_passes must be at least 1: {max_passes!r}")
    _yield_surface(
        friction_coefficient=friction_coefficient,
        friction_ellipse=friction_ellipse,
        where="advance_contact_quasistatic",
    )

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
    #: 物质点进来之后这四条推广成两条（`_check_end`/`_check_material_point`），
    #: 而`int`那一支**逐字保留**——推广不许顺手改既有口径。
    node_dof_count = contact_layout.layout.node_dof_count
    _check_end(
        node,
        counterpart,
        contact_layout=contact_layout,
        vector=vector,
        where="advance_contact_quasistatic",
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
    solved = outcome = None
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
        #: 椭圆**每趟现造**，法向取``direction``——与下一行粘着弹簧用的是同一个值。
        #: 取求解**之后**的法向会让试探力落在另一个切平面里，
        #: `anisotropic_return_map`当场以"试探力带法向分量"报错（那条报错是对的）。
        ellipse = (
            None
            if friction_ellipse is None
            else friction_ellipse.at(normal=direction, vector=current)
        )
        stick_terms, trial_of = _assemble_stick(
            (
                (
                    node,
                    counterpart,
                    anchor,
                    direction,
                    tangential_stiffness_n_per_mm,
                ),
            )
        )
        terms = (
            (*registry_without_stick.terms, *stick_terms)
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
        trial = trial_of(solved.state)[0] if engaged else (0.0,) * 3
        #: 屈服残差：试探力超出屈服面多少。粘着时它按定义≤0，
        #: 故本判据只在滑移分支上有内容——粘着一趟就退出，与旧行为一致。
        outcome, excess = _return_map(
            trial=trial,
            normal_force_n=normal_force,
            friction_coefficient=friction_coefficient,
            ellipse=ellipse,
            tangential_stiffness_n_per_mm=tangential_stiffness_n_per_mm,
        )
        anchor = tuple(
            anchor[axis] + outcome.anchor_correction_mm[axis] for axis in range(3)
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

    ## 屈服面**逐点**声明（plans/15第1.6条）

    ``friction_coefficient``与``friction_ellipse``恰好给一个，**每个点各判各的**：
    带材压在导轮上时几十个接触点共用同一套材料常数，但把它做成整步一个参数
    就等于宣布"一步之内不许有两种屈服面"，而那句话本仓没有理由说。

    ## ``node``可以是物质点，``counterpart``给对边（决策0080）

    与单槽位版本同一条口径。**混着声明是允许的**：同一步里一部分点是质点
    （``int``）、另一部分是刚体上的物质点，因为"这一端是不是刚体"是**体**的属性，
    不是这一步的属性——与屈服面逐点声明那条逐字同源。
    """

    slot: ContactSlot
    node: ContactEnd
    normal: NormalSource
    normal_force_of: Callable[[State], float]
    tangential_stiffness_n_per_mm: float
    friction_coefficient: float | None = None
    friction_ellipse: FrictionEllipseSpec | None = None
    #: 对边的物质点。``None``时另一端是世界锚点（地面那一类）。
    #: **只能与`MaterialPoint`一起给**，理由见`_check_end`。
    counterpart: MaterialPoint | None = None

    def __post_init__(self) -> None:
        _yield_surface(
            friction_coefficient=self.friction_coefficient,
            friction_ellipse=self.friction_ellipse,
            where=f"ContactPoint(slot={self.slot.pair_id!r}, node={self.node!r})",
        )


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

    ## 屈服面逐点声明（plans/15第1.6条，2026-08-17补）

    每个`ContactPoint`各带``friction_coefficient``（圆）或``friction_ellipse``
    （椭圆，决策0068），恰好一个。**混着声明是允许的**：同一步里一部分点走圆、
    另一部分走椭圆，因为屈服面是**材料对**的属性而不是这一步的属性。
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
        #: 逐点复用单槽位版本那几条落位校验——它们挡的是**静默写坏状态**，
        #: 而多槽位下写坏的概率恰好乘以点数。
        _check_end(
            point.node,
            point.counterpart,
            contact_layout=contact_layout,
            vector=vector,
            where=f"contacts[{index}]",
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
        #: 法向**只对活动点求值**——这一条不是省算力，是保行为：
        #: `PenaltyCylinderContact`的法向在轴上没有定义并**失败关闭**，
        #: 而一个分离的点完全可能落在那里。对全体求值等于给旧调用方
        #: 凭空加一条新的报错路径。
        directions: list[tuple[float, float, float] | None] = [None] * count
        #: 椭圆与粘着弹簧共用**同一个**``directions[index]``——
        #: 两处各算一次法向会在曲面上给出两个差着舍入的切平面，
        #: 而试探力必须精确落在椭圆那一个里面。
        ellipses: list[FrictionEllipse | None] = [None] * count
        for index in range(count):
            if not engaged[index]:
                continue
            direction = normal_of[index](current)
            directions[index] = direction
            spec = contacts[index].friction_ellipse
            if spec is not None:
                ellipses[index] = spec.at(normal=direction, vector=current)
        active = tuple(index for index in range(count) if engaged[index])
        entries = tuple(
            (
                contacts[index].node,
                contacts[index].counterpart,
                anchors[index],
                directions[index],
                contacts[index].tangential_stiffness_n_per_mm,
            )
            for index in active
        )
        stick_terms, trial_of = _assemble_stick(entries)
        terms = (
            (*registry_without_stick.terms, *stick_terms)
            if entries
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

        #: 试探力要按**装配时**的活动集取——`trial_of`按``entries``的次序返回，
        #: 而``entries``只含当时engaged的那些。用求解后的新活动集去索引会错位。
        trials: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * count
        if entries:
            for index, force in zip(active, trial_of(solved.state), strict=True):
                trials[index] = force

        for index, point in enumerate(contacts):
            normal_force = point.normal_force_of(solved.state)
            normal_forces[index] = normal_force
            engaged[index] = normal_force > 0.0
            trial = trials[index] if engaged[index] else (0.0, 0.0, 0.0)
            ellipse = ellipses[index]
            if ellipse is None and point.friction_ellipse is not None and engaged[index]:
                #: 装配时它还是分离的（故上面没给它造椭圆），求解后又贴上了。
                #: 此刻``trials[index]``按构造恒为零矢量，屈服面只用来报一个margin，
                #: 所以拿求解后的法向现造一个就够。
                ellipse = point.friction_ellipse.at(
                    normal=normal_of[index](current), vector=current
                )
            outcome, excess = _return_map(
                trial=trial,
                normal_force_n=normal_force,
                friction_coefficient=point.friction_coefficient,
                ellipse=ellipse,
                tangential_stiffness_n_per_mm=point.tangential_stiffness_n_per_mm,
            )
            outcomes[index] = outcome
            anchors[index] = tuple(
                anchors[index][axis] + outcome.anchor_correction_mm[axis] for axis in range(3)
            )
            excesses[index] = excess
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


__all__ = [
    "ContactEnd",
    "ContactPoint",
    "ContactStep",
    "MultiContactStep",
    "advance_contact_quasistatic",
    "advance_contacts_quasistatic",
]
