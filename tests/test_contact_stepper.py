"""准静态接触步进器的门：**一趟够不够，取决于法向转不转**。

这是**求解器的性质**，不是物理基准，所以它在`tests/`而不在`cases/`——
`cases/`那边判的是"引擎算得对不对物理"，这里判的是"预测-修正收敛不收敛"。

## 实测结论（三条，都被下面的门钉着）

1. **法向固定 → 一趟精确。** 理想塑性的return-map修正后屈服条件恰好成立，
   而法向不动意味着再解一次还是同一个点。`friction_hysteresis_loop`整个案例
   就跑在这个前提上。
2. **法向随位形转 → 线性收敛，每趟压缩因子约1.43。** 实测跨四次倍增
   （1→2→4→8→16趟）比值分别是1.427、2.036、4.146、17.193，
   而``1.427² = 2.036``、``1.427⁴ = 4.146``、``1.427⁸ = 17.19``——
   **逐格吻合，即每趟的因子是同一个数**。
3. **粘着时一趟就停。** 屈服超出量为负（在锥内），无论``max_passes``给多大
   都只走一趟——**迭代不该为一个已经满足的条件多花一次求解**。

## 屈服超出量为什么必须是公开字段

`ContactStep.tangential_force_n`是**投影后**的力，滑移时它按构造恒等于``μN``。
拿它去判收敛**永远得到0**——起草时正是这样量了个寂寞，
以为迭代"一趟就收敛了"，实际上是量错了对象。

**一个观测不到的收敛判据不是判据。**
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    ContactError,
    PenaltyNormalContact,
    PenaltySphereContact,
    advance_contact_quasistatic,
    build_contact_layout,
)
from physics_engine.energies import (
    EnergyContext,
    EnergyRegistry,
    PointLoad,
    UniformGravity,
)

RADIUS_MM = 10.0
MASS_KG = 1.0
GRAVITY_MM_S2 = 9810.0
STIFFNESS_N_PER_MM = 1.0e5
FRICTION = 0.35
UP = (0.0, 0.0, 1.0)

#: 内层牛顿的残差容差按相消地板给（``k·2r·ε``）——理由见
#: `tests/cases/test_three_sphere_pyramid.py`的同名常量。
RESIDUAL_TOL_N = 4.0 * STIFFNESS_N_PER_MM * 2.0 * RADIUS_MM * 2.220446049250313e-16


def _groove(lateral_load_n: float):
    """两个**固定**球夹住一个受横载的球——法向随位形转的最小装置。

    上球被推向一侧时会沿两个接触面爬升，两条连心线跟着转，
    于是"当前法向"每趟都不一样。这正是一趟不够的构型。
    """

    contact_layout = build_contact_layout(
        layout_id="layout/contact-stepper",
        node_count=3,
        declarations=(ContactDeclaration("left"),),
    )
    context = EnergyContext(
        context_id="context/contact-stepper",
        node_masses_kg=(MASS_KG,) * 3,
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    spheres = PenaltySphereContact(
        pairs=(
            (0, 2, 2.0 * RADIUS_MM, STIFFNESS_N_PER_MM),
            (1, 2, 2.0 * RADIUS_MM, STIFFNESS_N_PER_MM),
        )
    )
    registry = EnergyRegistry(
        terms=(
            UniformGravity(),
            spheres,
            PointLoad(loads=((2, (lateral_load_n, 0.0, 0.0)),)),
        )
    )
    height = RADIUS_MM * math.sqrt(3.0)
    vector = list(
        contact_layout.initial_vector(
            (
                -RADIUS_MM, 0.0, RADIUS_MM,
                RADIUS_MM, 0.0, RADIUS_MM,
                0.0, 0.0, RADIUS_MM + height - 2.0e-4,
            )
        )
    )
    slot = contact_layout.slot_of("left")
    vector[slot.anchor_base : slot.anchor_base + 3] = [0.0, 0.0, RADIUS_MM + height]
    return contact_layout, context, spheres, registry, slot, tuple(vector)


def _step(lateral_load_n: float, max_passes: int):
    contact_layout, context, spheres, registry, slot, vector = _groove(lateral_load_n)
    return advance_contact_quasistatic(
        registry_without_stick=registry,
        context=context,
        contact_layout=contact_layout,
        slot=slot,
        vector=vector,
        node=2,
        normal=lambda current: spheres._pair_state(current, spheres.pairs[0])[2],
        normal_force_of=lambda state: spheres.contact_force_n(state)[0],
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION,
        fixed_indices=frozenset(
            set(range(0, 6)) | {7} | set(range(9, contact_layout.layout.dof_count))
        ),
        residual_tol_n=RESIDUAL_TOL_N,
        max_iterations=300,
        max_passes=max_passes,
        yield_tol_n=1.0e-9,
    )


def test_a_turning_normal_needs_more_than_one_pass():
    """**加迭代的依据**：一趟之后屈服条件还差得远。

    这条门存在的理由是"先量再建"——没有它，多趟机械就只是一个看起来更严谨的选择。
    """

    step = _step(1.5, max_passes=1)
    assert not step.is_stick
    assert step.passes == 1
    assert step.yield_excess_n > 0.1, (
        f"一趟之后屈服超出只有{step.yield_excess_n}——若它已经足够小，"
        "多趟机械就没有存在理由，这条门该删"
    )


def test_the_passes_contract_the_yield_excess_by_a_constant_factor():
    """**线性收敛，每趟同一个压缩因子。**

    比较的是倍增的趟数，所以比值应当是逐趟因子的幂：``f``、``f²``、``f⁴``、``f⁸``。
    实测1.427、2.036、4.146、17.193，而``1.427⁸ = 17.19``——**逐格吻合**。

    门判的是"每趟因子在各档之间一致"，**不写死1.427**：
    那个数是这个构型的，换构型会变；**一致才是收敛的证据，具体值不是**
    （与`harmonic_oscillator`那条"不写死为4"同源）。
    """

    counts = (1, 2, 4, 8, 16)
    excesses = [_step(1.5, max_passes=count).yield_excess_n for count in counts]
    assert all(value > 0.0 for value in excesses), f"没在滑移分支上：{excesses}"
    assert all(
        excesses[index] < excesses[index - 1] for index in range(1, len(excesses))
    ), f"屈服超出没有逐档减小：{excesses}"

    per_pass = [
        (excesses[index - 1] / excesses[index]) ** (1.0 / counts[index - 1])
        for index in range(1, len(excesses))
    ]
    spread = max(per_pass) / min(per_pass)
    assert spread < 1.02, (
        f"每趟压缩因子在各档之间不一致（{per_pass}，散布{spread:.4f}）——"
        "那说明它不是线性收敛，比值区间那条门是在拿噪声算阶"
    )
    assert all(factor > 1.05 for factor in per_pass), (
        f"压缩因子太接近1，迭代几乎没在推进：{per_pass}"
    )


def test_sticking_stops_after_one_pass_no_matter_the_budget():
    """粘着时屈服超出为负（在锥内），**不该为一个已经满足的条件多花一次求解**。"""

    for budget in (1, 8, 64):
        step = _step(1.0, max_passes=budget)
        assert step.is_stick
        assert step.passes == 1, f"粘着却走了{step.passes}趟"
        assert step.yield_excess_n < 0.0


def test_a_fixed_normal_is_exact_in_one_pass():
    """**法向固定时多趟不改变任何东西**——这条守的是"默认值没有偷偷改行为"。

    `friction_hysteresis_loop`整个案例跑在一趟上。若多趟在固定法向下也会动结果，
    那说明一趟本来就不精确，而那个案例的判据全部要重估。
    """

    contact_layout = build_contact_layout(
        layout_id="layout/fixed-normal",
        node_count=1,
        declarations=(ContactDeclaration("ground"),),
    )
    context = EnergyContext(
        context_id="context/fixed-normal",
        node_masses_kg=(MASS_KG,),
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    weight = MASS_KG * GRAVITY_MM_S2 / 1000.0
    ground = PenaltyNormalContact(
        planes=((0, (0.0, 0.0, 0.0), UP, STIFFNESS_N_PER_MM, 0.0),)
    )
    registry = EnergyRegistry(terms=(UniformGravity(), ground))
    slot = contact_layout.slot_of("ground")
    vector = list(
        contact_layout.initial_vector((0.01, 0.0, -weight / STIFFNESS_N_PER_MM))
    )

    results = []
    for budget in (1, 5):
        step = advance_contact_quasistatic(
            registry_without_stick=registry,
            context=context,
            contact_layout=contact_layout,
            slot=slot,
            vector=tuple(vector),
            node=0,
            normal=UP,
            normal_force_of=lambda state: ground.normal_force_n(state)[0],
            tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
            friction_coefficient=FRICTION,
            fixed_indices=frozenset(
                {0, 1} | set(range(3, contact_layout.layout.dof_count))
            ),
            residual_tol_n=RESIDUAL_TOL_N,
            max_iterations=100,
            max_passes=budget,
            yield_tol_n=1.0e-9,
        )
        results.append(step)
    anchors = [
        step.state.vector[slot.anchor_base : slot.anchor_base + 3] for step in results
    ]
    assert anchors[0] == anchors[1], (
        f"固定法向下多趟改变了锚点：{anchors}——**历史必须一趟就精确**，"
        "否则`friction_hysteresis_loop`的全部判据都要重估"
    )
    assert results[0].slip_increment_mm == results[1].slip_increment_mm

    #: **但力与`regime`不是逐位相同的，理由要写清楚**：
    #: 第一趟报的是**投影后**的力（构造上恰为``μN``），第二趟是拿修正后的锚点
    #: **重算**出来的，两者差约8个ulp。而修正后的状态**恰好落在锥面上**，
    #: 于是``|T| ≤ μN``把它判成**粘**——`regime`因此从"滑"翻成"粘"。
    #:
    #: **这是边界约定不是缺陷**，但它意味着：`regime`这个标签在滑移步之后
    #: **依赖趟数预算**。默认1趟保住了既有案例的语义，
    #: 而这条注释保证下一个改默认值的人先看见它。
    forces = [
        math.sqrt(sum(value * value for value in step.tangential_force_n))
        for step in results
    ]
    assert forces[0] == pytest.approx(forces[1], rel=1.0e-14)
    assert not results[0].is_stick, "第一趟应当报滑（试探力超出锥面）"
    assert results[1].is_stick, "第二趟应当报粘（修正后恰在锥面上）"
    assert results[1].passes == 2, "固定法向下第二趟就是不动点"


def test_a_zero_pass_budget_fails_closed():
    with pytest.raises(ContactError, match="at least 1"):
        _step(1.5, max_passes=0)


# ---------------------------------------------------------------------------
# 分离：活动集与历史（2026-08-06对抗审核抓到的两条静默错值）
# ---------------------------------------------------------------------------


def _ground_drag(lateral_load_n: float, height_mm: float, anchor_x_mm: float = 0.0):
    """块在水平地面上，`height_mm`控制它贴地还是悬在空中。"""

    contact_layout = build_contact_layout(
        layout_id="layout/separation",
        node_count=1,
        declarations=(ContactDeclaration("ground"),),
    )
    context = EnergyContext(
        context_id="context/separation",
        node_masses_kg=(MASS_KG,),
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    ground = PenaltyNormalContact(
        planes=((0, (0.0, 0.0, 0.0), UP, STIFFNESS_N_PER_MM, 0.0),)
    )
    registry = EnergyRegistry(
        terms=(UniformGravity(), ground, PointLoad(loads=((0, (lateral_load_n, 0.0, 0.0)),)))
    )
    slot = contact_layout.slot_of("ground")
    vector = list(contact_layout.initial_vector((0.0, 0.0, height_mm)))
    vector[slot.anchor_base] = anchor_x_mm
    return contact_layout, context, ground, registry, slot, tuple(vector)


def _advance(contact_layout, context, ground, registry, slot, vector, *, free_x: bool):
    fixed = set(range(3, contact_layout.layout.dof_count)) | {1}
    if not free_x:
        fixed.add(0)
    return advance_contact_quasistatic(
        registry_without_stick=registry,
        context=context,
        contact_layout=contact_layout,
        slot=slot,
        vector=vector,
        node=0,
        normal=UP,
        normal_force_of=lambda state: ground.normal_force_n(state)[0],
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION,
        fixed_indices=frozenset(fixed),
        residual_tol_n=RESIDUAL_TOL_N,
        max_iterations=200,
    )


def test_a_separated_contact_exerts_no_tangential_force():
    """**分离时不许接粘着项。**

    对抗审核实测：节点抬到面上方500mm，法向力正确归零、regime正确报SEPARATED、
    报告的切向力也是0，**而平衡位置仍被一根不存在的摩擦弹簧顶在`T/k_t`上**。

    更难看的是**正确的活动集会失败关闭**——把粘着项拿掉，切向没有任何刚度，
    `solve_equilibrium`当场报奇异。**正确的做法炸，错误的做法静默给答案。**
    """

    setup = _ground_drag(2.0, height_mm=-2.0e-4, anchor_x_mm=0.0)
    engaged = _advance(*setup, free_x=True)
    assert engaged.normal_force_n > 0.0
    held_x = engaged.state.vector[0]
    assert held_x != 0.0, "贴地时应当被摩擦弹簧顶住一个位移，否则本门在验空气"

    # 抬到空中：没有法向力就没有摩擦，切向**不该**再有任何刚度
    airborne = _ground_drag(2.0, height_mm=5.0, anchor_x_mm=0.0)
    with pytest.raises((ContactError, Exception)) as excinfo:
        _advance(*airborne, free_x=True)
    assert "singular" in str(excinfo.value) or "converge" in str(excinfo.value), (
        f"悬空且切向自由时本该欠约束失败关闭，实际：{excinfo.value}"
    )


def test_lifting_and_moving_in_the_air_invents_no_slip():
    """**分离要清切向历史。**

    对抗审核实测的原始形态：贴地拖到2mm（锚点跟着到1.9998）、抬到空中横移到50mm、
    再放回地面——那一步报出`slip_increment_mm = 48`，
    **凭空记了282.5 N·mm的摩擦功，而那48mm是在空中走的**。

    同行的罚摩擦实现（Chrono DEM、LAMMPS granular）在失去接触时一律清切向历史。

    **装置上的一条限制先说清**：悬空的节点在**准静态下根本没有平衡**——
    法向分离后z方向也没有任何刚度（重力的Hessian恒为零），
    所以"抬起—空中横移—放下"整条路本质是**瞬态**问题。
    因此这里把节点钉在空中的一个位置上，只验**历史记账**那一半：
    分离那一步不许报滑移、不许留旧锚点。

    这条限制本身值得记：**分离的准静态解不存在，是模型的性质不是实现的缺陷**。
    """

    # 空中的一步：锚点里带着旧历史（贴地时拖出来的），位置已横移到50mm。
    # z留一个自由度给求解器（否则`solve_equilibrium`报"每个自由度都被钉住"），
    # 但节点被一个把它按在空中的外载托住——本门只关心历史记账。
    airborne = _ground_drag(0.0, height_mm=5.0, anchor_x_mm=1.9998)
    contact_layout, context, ground, _registry, slot, vector = airborne
    weight = MASS_KG * GRAVITY_MM_S2 / 1000.0
    # 恰好托住自重的外载：空中因此有一个平凡的平衡
    airborne_registry = EnergyRegistry(
        terms=(UniformGravity(), ground, PointLoad(loads=((0, (0.0, 0.0, weight)),)))
    )
    vector = list(vector)
    vector[0] = 50.0
    fixed = frozenset({0, 1} | set(range(3, contact_layout.layout.dof_count)))
    step = advance_contact_quasistatic(
        registry_without_stick=airborne_registry,
        context=context,
        contact_layout=contact_layout,
        slot=slot,
        vector=tuple(vector),
        node=0,
        normal=UP,
        normal_force_of=lambda state: ground.normal_force_n(state)[0],
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION,
        fixed_indices=fixed,
        residual_tol_n=RESIDUAL_TOL_N,
        max_iterations=200,
    )

    assert step.normal_force_n == 0.0
    assert step.slip_increment_mm == 0.0, (
        f"空中那一步报了{step.slip_increment_mm}mm滑移——那段路是在空中走的"
    )
    anchor = step.state.vector[slot.anchor_base : slot.anchor_base + 3]
    assert anchor == (0.0, 0.0, 0.0), f"分离后旧锚点还留着：{anchor}——再接触时会凭空滑一次"
    assert step.state.vector[slot.active_index] == 0.0


# ---------------------------------------------------------------------------
# 趟数用尽的失败关闭（plans/09第七节第6条，2026-08-12补）
#
# 同一个函数里**内层牛顿不收敛是`raise`**，而外层趟数用尽此前什么都不做——
# 两种不收敛，两种待遇。
# ---------------------------------------------------------------------------


def _step_requiring_convergence(
    lateral_load_n: float, max_passes: int, yield_tol_n: float = 1.0e-9
):
    contact_layout, context, spheres, registry, slot, vector = _groove(lateral_load_n)
    return advance_contact_quasistatic(
        registry_without_stick=registry,
        context=context,
        contact_layout=contact_layout,
        slot=slot,
        vector=vector,
        node=2,
        normal=lambda current: spheres._pair_state(current, spheres.pairs[0])[2],
        normal_force_of=lambda state: spheres.contact_force_n(state)[0],
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION,
        fixed_indices=frozenset(
            set(range(0, 6)) | {7} | set(range(9, contact_layout.layout.dof_count))
        ),
        residual_tol_n=RESIDUAL_TOL_N,
        max_iterations=300,
        max_passes=max_passes,
        yield_tol_n=yield_tol_n,
        require_pass_convergence=True,
    )


def test_exhausting_the_pass_budget_can_now_fail_closed():
    """**必红**：要求收敛时，趟数不够必须抛，不许静默返回一个没收敛的结果。

    **注错方式**：法向随位形转（实测每趟压缩因子约1.43），只给1趟。
    """

    with pytest.raises(ContactError, match="仍未收敛"):
        _step_requiring_convergence(2.0, max_passes=1)


def test_the_default_still_returns_quietly_because_changing_it_would_break_callers():
    """**反向必红**：默认值下**行为一个字节不变**。

    这条守的是本模块docstring自己写过的原则——
    **默认值不该替既有调用方改变行为，那是把一次能力扩展偷偷变成一次行为变更**。

    而且这里的"没收敛"多半是**正常且正确**的：`yield_excess_n`量的是
    **修正前**的试探力，理想塑性的修正让屈服条件在**修正后**成立，
    本判据看不到那一步。
    """

    step = _step(2.0, max_passes=1)
    assert step.passes == 1
    assert step.yield_excess_n > 0.0, "这个构型本该是滑移且残差为正，否则本组用例失效"


def test_enough_passes_converge_and_do_not_raise():
    """**绿分支**：趟数够、容差够，就不该抛。

    **构型是量出来的，不是猜的**（2026-08-12）：横载1.5 N时
    8趟残差2.478e-02、16趟1.441e-03——压缩因子与本文件第2条实测的1.43一致。
    """

    step = _step_requiring_convergence(1.5, max_passes=16, yield_tol_n=1.0e-2)
    assert step.yield_excess_n <= 1.0e-2
    assert step.passes > 1, "绿分支要走多趟，否则它验的是粘着的一趟即停"


def test_there_is_no_pass_budget_that_reaches_a_tight_tolerance_here():
    """**实测记录**：这个构型下**不存在**能到1e-9的趟数预算。

    外层每趟压缩约1.43（16趟到1.441e-03），**而内层牛顿在32趟左右先死**
    （线搜索40次回溯后仍无法降低能量）。要到1e-9还需约48趟。

    这正是plans/07第六节登记的"**迭代发散时的兜底**"那笔账——
    今天两种失败**都会抛**，所以没有静默错值；但**没有载荷步回退**，
    使用者除了调小载荷或放松容差之外无路可走。

    本用例把这条钉住：**它红了说明求解器变强了或变弱了，两种都该有人来看。**
    """

    with pytest.raises(ContactError):
        _step_requiring_convergence(1.5, max_passes=64, yield_tol_n=1.0e-9)


# ---------------------------------------------------------------------------
# slot与node的落位校验（plans/09第七节第7条，2026-08-12补）
#
# `ContactSlot`不带node、`ContactDeclaration`也不带——两者之间今天没有任何
# 东西把它们绑在一起。完整对应要改0050的布局承重设计，本轮只做能证明的一半。
# ---------------------------------------------------------------------------


def _stepper_kwargs_with(**overrides):
    contact_layout, context, spheres, registry, slot, vector = _groove(1.0)
    call = {
        "registry_without_stick": registry,
        "context": context,
        "contact_layout": contact_layout,
        "slot": slot,
        "vector": vector,
        "node": 2,
        "normal": lambda current: spheres._pair_state(current, spheres.pairs[0])[2],
        "normal_force_of": lambda state: spheres.contact_force_n(state)[0],
        "tangential_stiffness_n_per_mm": STIFFNESS_N_PER_MM,
        "friction_coefficient": FRICTION,
        "fixed_indices": frozenset(
            set(range(0, 6)) | {7} | set(range(9, contact_layout.layout.dof_count))
        ),
        "residual_tol_n": RESIDUAL_TOL_N,
        "max_iterations": 300,
    }
    call.update(overrides)
    return call, contact_layout


def test_a_node_past_the_node_block_fails_closed():
    """**必红**：节点号越过节点块 → 再往后是锚点槽，**写进去就是改别人的历史**。

    **注错方式**：布局有3个节点（9个自由度），传`node=3`。
    此前它不会抛——元组切片越界是静默的。
    """

    call, _ = _stepper_kwargs_with(node=3)
    with pytest.raises(ContactError, match="落在节点块之外"):
        advance_contact_quasistatic(**call)


def test_a_negative_node_fails_closed():
    """**必红**（同判据第二个分支）：``node = -1``。

    这不是假想——决策0050落地时`PenaltySphereContact`吃过这个亏：
    ``node = -1``被接受、``vector[-3:]``读的**正是锚点槽**，
    于是算出316681 N·mm能量，全部由历史值来。
    """

    call, _ = _stepper_kwargs_with(node=-1)
    with pytest.raises(ContactError, match="nonnegative"):
        advance_contact_quasistatic(**call)


def test_a_slot_pointing_into_the_node_block_fails_closed():
    """**必红**（第三个分支）：把槽下标指进节点块 → **锚点写进节点块就是悄悄改位形**。

    **注错方式**：造一个`base=0`的槽（节点块的第一个自由度）。
    """

    from physics_engine.contact import ContactSlot

    call, _ = _stepper_kwargs_with()
    call["slot"] = ContactSlot(pair_id="fake", point_index=0, base=0)
    with pytest.raises(ContactError, match="落在节点块之内"):
        advance_contact_quasistatic(**call)


def test_a_slot_past_the_vector_end_fails_closed():
    """**必红**（第四个分支）：槽越过状态向量末尾。"""

    from physics_engine.contact import ContactSlot

    call, layout = _stepper_kwargs_with()
    call["slot"] = ContactSlot(
        pair_id="fake", point_index=0, base=len(call["vector"])
    )
    with pytest.raises(ContactError, match="越过了状态向量末尾"):
        advance_contact_quasistatic(**call)


# ------------------------------------------------- 报错文本逐字（不是子串）
#: **2026-08-18补的门,理由是它挡的那件事已经发生过一次。**
#: 对抗审核实测:轨S重构`_check_end`时给`int`那一支的报错加了`{where}: `前缀、
#: 多点那一支还把"的节点"改成"node"并加了粗——**而本文件与
#: `test_contact_multi_stepper.py`里所有相关用例都是
#: `pytest.raises(ContactError, match="落在节点块之外")`,子串匹配对措辞漂移完全不可见**,
#: 于是那次改动一路绿到收口,还被写进0080的"24条确定性报错文本diff为空"里。
#:
#: 下面两条常量**逐字取自基线树`d082b65`**(用`git archive`还原后实跑取回),
#: 判的是**相等**不是`in`。
LEGACY_SINGLE_OUT_OF_BLOCK = (
    "node 7 落在节点块之外（节点块只有3个自由度）"
    "——**再往后是锚点槽，写进去就是改别人的历史**"
)
LEGACY_MULTI_OUT_OF_BLOCK = (
    "contacts[0]的节点99落在节点块之外（节点块只有3个自由度）——再往后是锚点槽"
)


def _minimal_stepper_pieces():
    from physics_engine.contact import build_contact_layout
    from physics_engine.contact.layout import ContactDeclaration
    from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity

    layout = build_contact_layout(
        layout_id="layout/text",
        node_count=1,
        declarations=(ContactDeclaration(pair_id="contact/text", max_points=1),),
    )
    registry = EnergyRegistry(terms=(UniformGravity(),))
    context = EnergyContext(
        context_id="context/text", node_masses_kg=(1.0,), gravity_mm_s2=(0.0, 0.0, 0.0)
    )
    vector = tuple(0.0 for _ in range(layout.layout.dof_count))
    return layout, registry, context, vector


def test_the_single_point_out_of_block_text_is_byte_for_byte_the_legacy_one():
    """单槽位那条报错文本必须与基线**逐字相等**。"""

    layout, registry, context, vector = _minimal_stepper_pieces()
    with pytest.raises(ContactError) as caught:
        advance_contact_quasistatic(
            registry_without_stick=registry,
            context=context,
            contact_layout=layout,
            slot=layout.slots[0],
            vector=vector,
            node=7,
            normal=(0.0, 0.0, 1.0),
            normal_force_of=lambda state: 1.0,
            tangential_stiffness_n_per_mm=1.0,
            friction_coefficient=0.3,
            fixed_indices=frozenset(),
        )
    assert str(caught.value) == LEGACY_SINGLE_OUT_OF_BLOCK, (
        "报错文本漂了 —— 既有调用方能观测到的字节变了。"
        "子串匹配看不见这件事，所以这条门判的是相等"
    )


def test_the_multi_point_out_of_block_text_is_byte_for_byte_the_legacy_one():
    """多槽位那条报错文本必须与基线**逐字相等**——它与单槽位那条**措辞本来就不同**。

    两条不同这件事本身要被判:重构把它们统一成一种,而"统一"改掉了既有字节。
    """

    from physics_engine.contact.stepper import ContactPoint, advance_contacts_quasistatic

    layout, registry, context, vector = _minimal_stepper_pieces()
    with pytest.raises(ContactError) as caught:
        advance_contacts_quasistatic(
            registry_without_stick=registry,
            context=context,
            contact_layout=layout,
            contacts=(
                ContactPoint(
                    slot=layout.slots[0],
                    node=99,
                    normal=(0.0, 0.0, 1.0),
                    normal_force_of=lambda state: 1.0,
                    tangential_stiffness_n_per_mm=1.0,
                    friction_coefficient=0.3,
                ),
            ),
            vector=vector,
            fixed_indices=frozenset(),
        )
    assert str(caught.value) == LEGACY_MULTI_OUT_OF_BLOCK
    assert LEGACY_MULTI_OUT_OF_BLOCK != LEGACY_SINGLE_OUT_OF_BLOCK, (
        "两条本来措辞不同 —— 若它们相等，上面两条门就退化成同一条"
    )
