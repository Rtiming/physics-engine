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
