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
    FrictionOutcome,
    TangentialStickSpring,
    coulomb_return_map,
)
from physics_engine.contact.layout import (
    REGIME_STICK,
    ContactLayout,
    ContactSlot,
    NormalSource,
)
from physics_engine.energies import EnergyContext
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


__all__ = [
    "ContactPoint",
    "ContactStep",
    "MultiContactStep",
    "advance_contact_quasistatic",
    "advance_contacts_quasistatic",
]
