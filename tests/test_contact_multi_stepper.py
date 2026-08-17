"""多槽位准静态步进器的门（决策0062轨道甲，兑现0050第一节登记的欠账）。

`advance_contact_quasistatic`一次只处理一个槽位——那是S3.6"多点同时接触"的
`missing`里写着的那一条。带材贴在导轮上时几十个节点同时接触、
**且通过带材的轴向刚度互相牵制**，逐个槽轮流走会把这个耦合拆散。

## 本文件的承重门只有一条

`test_a_single_contact_matches_the_old_stepper_bit_for_bit`：
**同一个构型下，多槽位版本对单个接触必须与老函数逐字节相同。**

它是这次泛化忠实与否的**证据**。没有它，"新写了一个更通用的"只是一句自称——
而本仓已经吃过一次同形态的亏（0050落地时两条路径各写一遍求和次序）。

## 其余的门守什么

多槽位**新增**的失败模式，逐条各一道：两点共用一个槽（后写覆盖先写，
一段历史凭空消失）、活动集在趟间翻转时试探力错位（`tangential_force_n`
按springs次序返回，而springs只含当时engaged的那些）、
收敛判据取全体而不是任一。

## 必红矩阵（2026-08-17逐条注错**实测**）

| 注错 | 红掉 |
|---|---|
| 每个接触各自一个粘着项（求和次序随活动集变） | 5 |
| 滑移从当前锚点起算而不是整步起点 | 4 |
| 锚点更新用整步起点而不是逐趟累加 | 4 |
| 收敛判据取任一而不是全体（``max``→``min``） | 2 |
| 试探力不按装配时的活动集索引（错位） | 1 |
| 分离时不清切向历史 | 1 |
| 共用槽的检查去掉 | 1 |

**七条全被抓到，但第五条是补出来的**：第一轮实测它**红0条、活了下来**。
病根是本文件当时所有构型里未接触的点都排在**最后**，下标恰好对得上。
补的门是`test_a_separated_point_declared_first_does_not_steal_the_engaged_point_trial_force`，
而**它的第一版判据（判趟数）照样抓不住**——错位真正偷走的不是空中那点的东西，
是**接地那点自己的切向力**。判对量才抓得住，这是plans/09教训三的又一个实例。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    ContactError,
    ContactPoint,
    PenaltyNormalContact,
    PenaltySphereContact,
    advance_contact_quasistatic,
    advance_contacts_quasistatic,
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

#: 内层牛顿的残差容差按相消地板给（``k·2r·ε``）——与`test_contact_stepper.py`同源。
RESIDUAL_TOL_N = 4.0 * STIFFNESS_N_PER_MM * 2.0 * RADIUS_MM * 2.220446049250313e-16

#: **双槽位的容差不能照抄单槽位的。** 两根粘着弹簧各贡献一份``k·ulp(L)``的相消地板
#: （``L``是节点坐标量级，这个构型约``2r√3 ≈ 34.6 mm``），而**判别翻转的那一趟**
#: 实测再高约4.5倍：2026-08-17实测load=1.5的第2趟走了**9311次回溯**、
#: 残差停在``3.19e-9``，恰好卡在单槽位容差``1.78e-9``与地板``7.1e-10``之间。
#:
#: 这不是"调松容差"，是**这个构型在判别翻转处能达到的精度就是这么多**——
#: 病根是0050第四节登记的``C¹``而非``C²``：活动集在迭代中翻转时线搜索失效。
#: 绕线机上判别会随带材移动不停翻转，所以这条界是要带进主线的，不是这里的权宜。
TWO_SLOT_RESIDUAL_TOL_N = (
    8.0 * 2.0 * STIFFNESS_N_PER_MM * math.ulp(2.0 * RADIUS_MM * math.sqrt(3.0))
)


def _groove(lateral_load_n: float, declarations: tuple[ContactDeclaration, ...]):
    """两个**固定**球夹住一个受横载的球——与老门逐字相同的构型。

    只有`declarations`不同：老门声明一个槽，这里按需要声明一个或两个。
    **构型必须逐字相同**，否则"逐字节相同"那条门比的是两个不同的问题。
    """

    contact_layout = build_contact_layout(
        layout_id="layout/contact-stepper",
        node_count=3,
        declarations=declarations,
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
    for declaration in declarations:
        slot = contact_layout.slot_of(declaration.pair_id)
        vector[slot.anchor_base : slot.anchor_base + 3] = [0.0, 0.0, RADIUS_MM + height]
    return contact_layout, context, spheres, registry, tuple(vector)


def _fixed(contact_layout) -> frozenset[int]:
    return frozenset(set(range(0, 6)) | {7} | set(range(9, contact_layout.layout.dof_count)))


# ---------------------------------------------------------------------------
# 承重门：单个接触必须与老函数逐字节相同
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_passes", [1, 2, 4, 8])
@pytest.mark.parametrize("load", [0.2, 1.0, 1.5])
def test_a_single_contact_matches_the_old_stepper_bit_for_bit(load: float, max_passes: int):
    """**这次泛化的全部证据。**

    横载三档跨粘着/滑移两个分支、趟数四档跨"一趟就退出"与"走满"两条路径，
    十二组全部要求**逐字节**相同——不是`approx`。

    多槽位版本把粘着弹簧装进**同一个**`TangentialStickSpring`并按声明次序排列，
    单点时那个次序退化成老函数的唯一那一个，所以求和次序不变、
    浮点结果才可能逐字节相同。**如果哪天它只差一个ulp，那说明装配次序变了，
    而按spec/12第3.3节那是形制变更，要走决策记录而不是调容差。**
    """

    declarations = (ContactDeclaration("left"),)
    contact_layout, context, spheres, registry, vector = _groove(load, declarations)
    slot = contact_layout.slot_of("left")
    common = dict(
        registry_without_stick=registry,
        context=context,
        contact_layout=contact_layout,
        vector=vector,
        fixed_indices=_fixed(contact_layout),
        residual_tol_n=RESIDUAL_TOL_N,
        max_iterations=300,
        max_passes=max_passes,
        yield_tol_n=1.0e-9,
    )
    normal = lambda current: spheres._pair_state(current, spheres.pairs[0])[2]  # noqa: E731
    force_of = lambda state: spheres.contact_force_n(state)[0]  # noqa: E731

    old = advance_contact_quasistatic(
        slot=slot, node=2, normal=normal, normal_force_of=force_of,
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION, **common,
    )
    new = advance_contacts_quasistatic(
        contacts=(
            ContactPoint(
                slot=slot, node=2, normal=normal, normal_force_of=force_of,
                tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
                friction_coefficient=FRICTION,
            ),
        ),
        **common,
    )

    assert new.state.vector == old.state.vector
    assert new.normal_force_n == (old.normal_force_n,)
    assert new.tangential_force_n == (old.tangential_force_n,)
    assert new.regime == (old.regime,)
    assert new.slip_increment_mm == (old.slip_increment_mm,)
    assert new.passes == old.passes
    assert new.max_yield_excess_n == old.yield_excess_n


def test_the_two_steppers_fail_identically_where_the_configuration_diverges():
    """**失败一致比成功一致更强。**

    横载3.0 N落在S3.6登记的发散区外（原文：实测横载2.0 N时内层牛顿第8趟失败）。
    在那里老新两条路径必须**同样地失败**——若新的"更稳健"，那说明它悄悄改了
    求解路径，而那就不再是同一个问题的两种写法。
    """

    declarations = (ContactDeclaration("left"),)
    contact_layout, context, spheres, registry, vector = _groove(3.0, declarations)
    slot = contact_layout.slot_of("left")
    common = dict(
        registry_without_stick=registry, context=context, contact_layout=contact_layout,
        vector=vector, fixed_indices=_fixed(contact_layout),
        residual_tol_n=RESIDUAL_TOL_N, max_iterations=300, max_passes=8, yield_tol_n=1.0e-9,
    )
    normal = lambda current: spheres._pair_state(current, spheres.pairs[0])[2]  # noqa: E731
    force_of = lambda state: spheres.contact_force_n(state)[0]  # noqa: E731

    with pytest.raises(ContactError, match="did not converge"):
        advance_contact_quasistatic(
            slot=slot, node=2, normal=normal, normal_force_of=force_of,
            tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
            friction_coefficient=FRICTION, **common,
        )
    with pytest.raises(ContactError, match="did not converge"):
        advance_contacts_quasistatic(
            contacts=(
                ContactPoint(
                    slot=slot, node=2, normal=normal, normal_force_of=force_of,
                    tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
                    friction_coefficient=FRICTION,
                ),
            ),
            **common,
        )


# ---------------------------------------------------------------------------
# 多槽位真的在同时走
# ---------------------------------------------------------------------------


def _two_slot_step(load: float, max_passes: int, **overrides):
    declarations = (ContactDeclaration("left"), ContactDeclaration("right"))
    contact_layout, context, spheres, registry, vector = _groove(load, declarations)
    call = dict(
        registry_without_stick=registry,
        context=context,
        contact_layout=contact_layout,
        contacts=tuple(
            ContactPoint(
                slot=contact_layout.slot_of(name),
                node=2,
                normal=(
                    lambda current, pair=spheres.pairs[index]: spheres._pair_state(
                        current, pair
                    )[2]
                ),
                normal_force_of=(
                    lambda state, index=index: spheres.contact_force_n(state)[index]
                ),
                tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
                friction_coefficient=FRICTION,
            )
            for index, name in enumerate(("left", "right"))
        ),
        vector=vector,
        fixed_indices=_fixed(contact_layout),
        residual_tol_n=TWO_SLOT_RESIDUAL_TOL_N,
        max_iterations=300,
        max_passes=max_passes,
        yield_tol_n=1.0e-9,
    )
    call.update(overrides)
    return advance_contacts_quasistatic(**call)


def test_a_regime_flip_costs_thousands_of_backtracks():
    """**判别翻转那一趟的代价是可量的，写下来免得下一个人以为是随机的。**

    2026-08-17实测（load=1.5、两点、``k = k_t = 1e5``）：第1趟两点都滑移、
    0回溯、残差``3.6e-11``；**第2趟右点从滑移翻成粘着，回溯9311次、
    残差涨到``3.19e-9``**；第3、4趟回落到0回溯、``7.9e-11``/``2.9e-10``。

    量级差了近**两个数量级**，而且只在翻转那一趟。病根是0050第四节的``C¹``非``C²``。
    本门判的是"用单槽位容差会失败、用按翻转标定的容差会成功"——
    **两边都判，只判一边证明不了容差是必要的而不是随手放松的**。
    """

    with pytest.raises(ContactError, match="did not converge"):
        _two_slot_step(1.5, max_passes=4, residual_tol_n=RESIDUAL_TOL_N)

    step = _two_slot_step(1.5, max_passes=4)
    assert step.passes == 4
    assert step.regime[0] == 2.0
    assert step.regime[1] == 1.0, "右点该在第2趟翻成粘着——翻转是这条门的前提"


def test_both_slots_carry_their_own_anchor_and_regime():
    """两个接触各写各的槽，**互不覆盖**。

    这是多槽位最基本的正确性：左右两个接触面的法向不同（一个朝右上、一个朝左上），
    于是锚点修正的方向也不同。若两点共用一段历史，两个锚点会相同——
    本门判它们**不相同**。
    """

    step = _two_slot_step(1.5, max_passes=8)
    assert len(step.normal_force_n) == 2
    assert len(step.regime) == 2
    assert all(force > 0.0 for force in step.normal_force_n)

    vector = step.state.vector
    contact_layout, *_ = _groove(1.5, (ContactDeclaration("left"), ContactDeclaration("right")))
    left = contact_layout.slot_of("left")
    right = contact_layout.slot_of("right")
    left_anchor = vector[left.anchor_base : left.anchor_base + 3]
    right_anchor = vector[right.anchor_base : right.anchor_base + 3]
    assert left_anchor != right_anchor, (
        "两个接触面的法向不同，锚点修正方向就该不同；相同说明两点共用了一段历史"
    )


def test_the_yield_excess_contracts_across_passes_for_both_slots():
    """**趟数增加，全体的最大屈服残差单调下降。**

    单槽位版本实测压缩因子约1/2；多点耦合不保证同一个因子，
    所以这条门判**单调**而不写死因子——与`test_contact_stepper.py`
    那条"判一致不判具体值"同源，但更弱一档，理由如实写在这里。
    """

    excesses = [_two_slot_step(1.5, max_passes=n).max_yield_excess_n for n in (1, 2, 4, 8)]
    for earlier, later in zip(excesses, excesses[1:], strict=False):
        assert later < earlier, f"趟数增加而残差没降：{excesses}"


def test_the_convergence_criterion_is_all_not_any():
    """收敛要**全体**达标才停。

    构型：一个接触远在屈服面内（粘着、残差为负），另一个在滑移。
    若判据写成"任一达标就停"，粘着那个会让整步在第一趟就退出，
    而滑移那个的残差还很大。**本门判`passes > 1`**。
    """

    step = _two_slot_step(1.5, max_passes=8)
    assert step.passes > 1, (
        f"只走了{step.passes}趟——判据可能写成了'任一达标'，"
        "那样粘着的那个点会替滑移的那个点提前宣布收敛"
    )


# ---------------------------------------------------------------------------
# 多槽位新增的失败关闭
# ---------------------------------------------------------------------------


def test_two_contacts_sharing_one_slot_fails_closed():
    """**后写覆盖先写，一段历史凭空消失**——单槽位版本没有这条检查，
    因为它一次只有一个槽；多槽位是它第一次有意义的地方。"""

    declarations = (ContactDeclaration("left"),)
    contact_layout, context, spheres, registry, vector = _groove(1.5, declarations)
    slot = contact_layout.slot_of("left")
    point = ContactPoint(
        slot=slot, node=2,
        normal=lambda current: spheres._pair_state(current, spheres.pairs[0])[2],
        normal_force_of=lambda state: spheres.contact_force_n(state)[0],
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION,
    )
    with pytest.raises(ContactError, match="共用锚点槽"):
        advance_contacts_quasistatic(
            registry_without_stick=registry, context=context,
            contact_layout=contact_layout, contacts=(point, point), vector=vector,
            fixed_indices=_fixed(contact_layout), residual_tol_n=RESIDUAL_TOL_N,
        )


def test_no_contacts_fails_closed():
    declarations = (ContactDeclaration("left"),)
    contact_layout, context, _, registry, vector = _groove(1.5, declarations)
    with pytest.raises(ContactError, match="at least one contact point"):
        advance_contacts_quasistatic(
            registry_without_stick=registry, context=context,
            contact_layout=contact_layout, contacts=(), vector=vector,
            fixed_indices=_fixed(contact_layout), residual_tol_n=RESIDUAL_TOL_N,
        )


@pytest.mark.parametrize(
    ("node", "message"),
    [(-1, "nonnegative int"), (True, "nonnegative int"), (99, "落在节点块之外")],
)
def test_a_bad_node_index_fails_closed(node, message):
    """逐点复用单槽位那四条落位校验——多槽位下写坏状态的概率乘以点数。"""

    declarations = (ContactDeclaration("left"),)
    contact_layout, context, spheres, registry, vector = _groove(1.5, declarations)
    point = ContactPoint(
        slot=contact_layout.slot_of("left"), node=node,
        normal=(0.0, 0.0, 1.0),
        normal_force_of=lambda state: spheres.contact_force_n(state)[0],
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION,
    )
    with pytest.raises(ContactError, match=message):
        advance_contacts_quasistatic(
            registry_without_stick=registry, context=context,
            contact_layout=contact_layout, contacts=(point,), vector=vector,
            fixed_indices=_fixed(contact_layout), residual_tol_n=RESIDUAL_TOL_N,
        )


def test_a_nonpositive_tangential_stiffness_fails_closed():
    declarations = (ContactDeclaration("left"),)
    contact_layout, context, spheres, registry, vector = _groove(1.5, declarations)
    point = ContactPoint(
        slot=contact_layout.slot_of("left"), node=2,
        normal=(0.0, 0.0, 1.0),
        normal_force_of=lambda state: spheres.contact_force_n(state)[0],
        tangential_stiffness_n_per_mm=0.0,
        friction_coefficient=FRICTION,
    )
    with pytest.raises(ContactError, match="切向刚度必须为正"):
        advance_contacts_quasistatic(
            registry_without_stick=registry, context=context,
            contact_layout=contact_layout, contacts=(point,), vector=vector,
            fixed_indices=_fixed(contact_layout), residual_tol_n=RESIDUAL_TOL_N,
        )


def test_max_passes_below_one_fails_closed():
    declarations = (ContactDeclaration("left"),)
    contact_layout, context, spheres, registry, vector = _groove(1.5, declarations)
    point = ContactPoint(
        slot=contact_layout.slot_of("left"), node=2,
        normal=(0.0, 0.0, 1.0),
        normal_force_of=lambda state: spheres.contact_force_n(state)[0],
        tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
        friction_coefficient=FRICTION,
    )
    with pytest.raises(ContactError, match="max_passes"):
        advance_contacts_quasistatic(
            registry_without_stick=registry, context=context,
            contact_layout=contact_layout, contacts=(point,), vector=vector,
            fixed_indices=_fixed(contact_layout), residual_tol_n=RESIDUAL_TOL_N,
            max_passes=0,
        )


def test_running_out_of_passes_can_be_made_to_fail_closed():
    """``require_pass_convergence``开着时走满趟数必须抛，且消息点名**哪个**点最差。

    默认仍是`False`——与单槽位版本同口径，理由同样是"默认值不该替既有调用方
    改变行为"。**多槽位没有既有调用方，但两条路径的默认不一致本身会变成一个坑。**
    """

    with pytest.raises(ContactError, match=r"contacts\[\d+\]"):
        _two_slot_step_strict(1.5)


def _two_slot_step_strict(load: float):
    declarations = (ContactDeclaration("left"), ContactDeclaration("right"))
    contact_layout, context, spheres, registry, vector = _groove(load, declarations)
    return advance_contacts_quasistatic(
        registry_without_stick=registry, context=context, contact_layout=contact_layout,
        contacts=tuple(
            ContactPoint(
                slot=contact_layout.slot_of(name), node=2,
                normal=(
                    lambda current, pair=spheres.pairs[index]: spheres._pair_state(
                        current, pair
                    )[2]
                ),
                normal_force_of=(
                    lambda state, index=index: spheres.contact_force_n(state)[index]
                ),
                tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
                friction_coefficient=FRICTION,
            )
            for index, name in enumerate(("left", "right"))
        ),
        vector=vector, fixed_indices=_fixed(contact_layout),
        residual_tol_n=RESIDUAL_TOL_N, max_iterations=300,
        max_passes=1, yield_tol_n=1.0e-9, require_pass_convergence=True,
    )


# ---------------------------------------------------------------------------
# 分离时清历史（与单槽位同一条理由，但要验它对每个点各自成立）
# ---------------------------------------------------------------------------


def test_a_separated_point_clears_its_own_history_without_touching_the_other():
    """一个点分离、另一个仍接触：**分离的清零，接触的原样保留**。

    单槽位版本的那条门只能验"清零"，验不了"不碰别人"——
    而"不碰别人"正是多槽位新增的失败面。
    """

    contact_layout = build_contact_layout(
        layout_id="layout/multi-clear",
        node_count=2,
        declarations=(ContactDeclaration("ground0"), ContactDeclaration("ground1")),
    )
    context = EnergyContext(
        context_id="context/multi-clear",
        node_masses_kg=(MASS_KG, MASS_KG),
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    #: 节点0压在地面上，节点1**悬在500 mm高空**——后者的接触必然分离。
    planes = PenaltyNormalContact(
        planes=(
            (0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), STIFFNESS_N_PER_MM, 0.0),
            (1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), STIFFNESS_N_PER_MM, 0.0),
        )
    )
    registry = EnergyRegistry(terms=(planes, PointLoad(loads=((0, (1.0, 0.0, 0.0)),))))
    vector = list(contact_layout.initial_vector((0.0, 0.0, -1.0e-4, 0.0, 0.0, 500.0)))
    airborne = contact_layout.slot_of("ground1")
    #: 给空中那个点塞一段**假历史**——不清就会被下一步当真。
    vector[airborne.anchor_base : airborne.anchor_base + 3] = [50.0, 0.0, 0.0]

    step = advance_contacts_quasistatic(
        registry_without_stick=registry, context=context, contact_layout=contact_layout,
        contacts=tuple(
            ContactPoint(
                slot=contact_layout.slot_of(name), node=index,
                normal=(0.0, 0.0, 1.0),
                normal_force_of=(
                    lambda state, index=index: planes.normal_force_n(state)[index]
                ),
                tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
                friction_coefficient=FRICTION,
            )
            for index, name in enumerate(("ground0", "ground1"))
        ),
        vector=tuple(vector),
        fixed_indices=frozenset({1, 2, 3, 4, 5} | set(range(6, contact_layout.layout.dof_count))),
        residual_tol_n=1.0e-9, max_iterations=200,
    )

    result = step.state.vector
    assert result[airborne.anchor_base : airborne.anchor_base + 3] == (0.0, 0.0, 0.0)
    assert step.slip_increment_mm[1] == 0.0, "空中那段50 mm不许被记成滑移"
    assert step.normal_force_n[0] > 0.0
    assert step.normal_force_n[1] == 0.0


def test_a_separated_point_declared_first_does_not_steal_the_engaged_point_trial_force():
    """**未接触的点排在前面时，试探力必须不错位。**

    `TangentialStickSpring.tangential_force_n`按``springs``的次序返回，
    而``springs``**只含装配时engaged的那些**。用接触点的下标去索引它，
    在"未接触的排在后面"时恰好对得上——本文件此前所有构型都是那样，
    于是2026-08-17的必红实测里这条注错**红0条，活了下来**。

    抓住它要把未接触的排在**前面**。但**光看空中那点看不出来**：
    循环里``trial``还会按该点自己的engaged再gate一次，所以错位的值到不了它身上。
    真正被偷走的是**接地那点自己的切向力**——错位实现给它``stick_forces[1]``，
    而``stick_forces``只有一个元素，于是它拿到零，报出的切向力从0.5 N变成0。

    **所以本门判的是接地点的切向力大小**，不是趟数：趟数在两种实现下都是1
    （2026-08-17实测，第一版判据写成`passes == 1`时这条注错照样活着）。
    """

    contact_layout = build_contact_layout(
        layout_id="layout/multi-order",
        node_count=2,
        declarations=(ContactDeclaration("air"), ContactDeclaration("ground")),
    )
    context = EnergyContext(
        context_id="context/multi-order",
        node_masses_kg=(MASS_KG, MASS_KG),
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    #: 节点0**悬在500 mm高空**（声明在前），节点1压在地面上（声明在后）。
    planes = PenaltyNormalContact(
        planes=(
            (0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), STIFFNESS_N_PER_MM, 0.0),
            (1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), STIFFNESS_N_PER_MM, 0.0),
        )
    )
    registry = EnergyRegistry(
        terms=(UniformGravity(), planes, PointLoad(loads=((1, (0.5, 0.0, 0.0)),)))
    )
    vector = contact_layout.initial_vector(
        (0.0, 0.0, 500.0, 0.0, 0.0, -MASS_KG * GRAVITY_MM_S2 / 1000.0 / STIFFNESS_N_PER_MM)
    )

    step = advance_contacts_quasistatic(
        registry_without_stick=registry, context=context, contact_layout=contact_layout,
        contacts=tuple(
            ContactPoint(
                slot=contact_layout.slot_of(name), node=index,
                normal=(0.0, 0.0, 1.0),
                normal_force_of=(
                    lambda state, index=index: planes.normal_force_n(state)[index]
                ),
                tangential_stiffness_n_per_mm=STIFFNESS_N_PER_MM,
                friction_coefficient=FRICTION,
            )
            for index, name in enumerate(("air", "ground"))
        ),
        vector=vector,
        fixed_indices=frozenset(
            {0, 1, 2, 4} | set(range(6, contact_layout.layout.dof_count))
        ),
        residual_tol_n=1.0e-9, max_iterations=200, max_passes=4, yield_tol_n=0.0,
    )

    assert step.normal_force_n[0] == 0.0, "空中那点不该有法向力——构型前提"
    assert step.normal_force_n[1] > 0.0, "接地那点该有法向力——构型前提"
    assert step.regime[1] == 1.0, "μN≈3.4 N远大于横载0.5 N，接地那点该粘着——构型前提"

    tangential = math.sqrt(sum(value * value for value in step.tangential_force_n[1]))
    assert tangential == pytest.approx(0.5, rel=1.0e-6), (
        f"接地那点报出的切向力是{tangential:.6g} N而不是横载的0.5 N——"
        "试探力按接触点下标而不是按装配次序索引了，"
        "而springs里只有它一个，于是它自己的力被读成了越界的零"
    )
