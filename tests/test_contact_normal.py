"""罚函数法向接触的门（决策0050第二节）。

本文件守两类事：**协议契约**（四方法齐备、融合路径逐字节、Hessian对得上梯度）
与**模型的两条已知性质**——

1. **法向力精确、穿透是``O(1/k)``**。这条不是"误差小"，是"误差在哪"：
   平衡时``k·δ = N``恒成立，所以``N``与罚刚度**无关**，而位置差``N/k``。
   实测跨``k``六个数量级，``N``的相对偏差**恒为0.00e+00**。
   **判据据此只判力与阈值，不判位置。**
2. **分离态下切线刚度奇异**。法向项在``g > 0``时对Hessian零贡献，
   于是"悬在半空的自由度"没有任何刚度——**牛顿法在那里迈不出第一步**。
   这是0050第四节登记的非光滑性的最直接后果，本文件把它钉成一条正向的门，
   免得将来有人以为那是个bug然后"顺手"加个正则化把它盖掉。

## 必红矩阵（逐条注错实测）

| 注错 | 本文件红掉几条 |
|---|---|
| 去掉``g < 0``判据（分离也施力＝隔空吸引） | 2 |
| 法向力符号反（接触把节点吸进去） | 8 |
| 能量漏掉``½`` | 8 |
| **Hessian当成轴对齐（丢``n ⊗ n``）** | **1** |

**最后一行是这张表里唯一要紧的一行。** 丢掉外积只红一条，
因为**除了斜法向那条门，本文件所有构型的法向都是``+z``**——
轴对齐时``n ⊗ n``恰好退化成单位阵，两种写法给出同一个数。

换句话说：**一个把法向硬编码成轴对齐的实现，能从这里几乎全部门底下走过去。**
抓住它的是`test_hessian_is_the_outer_product_of_the_normal`那一条，
而斜面案例（``θ``不为零，法向天然是斜的）是它在案例层的第二道防线。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    ContactError,
    PenaltyNormalContact,
    build_contact_layout,
)
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
from physics_engine.solve import SolveError, solve_equilibrium
from physics_engine.state import State

MASS_KG = 2.0
GRAVITY_MM_S2 = 9810.0
#: ``N = m·g``。``g``是mm/s²而``N``要的是m/s²，故除以1000（与`UniformGravity`同源）。
THEORY_NORMAL_N = MASS_KG * GRAVITY_MM_S2 / 1000.0

GROUND = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def _setup(stiffness_n_per_mm: float):
    contact_layout = build_contact_layout(
        layout_id="layout/normal-contact",
        node_count=1,
        declarations=(ContactDeclaration("node_ground"),),
    )
    context = EnergyContext(
        context_id="context/normal-contact",
        node_masses_kg=(MASS_KG,),
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    contact_layout.assert_matches_context(context)
    term = PenaltyNormalContact(planes=((0, GROUND[0], GROUND[1], stiffness_n_per_mm, 0.0),))
    return contact_layout, context, term


def _state(contact_layout, z_mm: float) -> State:
    return State(
        layout=contact_layout.layout, vector=contact_layout.initial_vector((0.0, 0.0, z_mm))
    )


# ---------------------------------------------------------------------------
# 模型性质一：力精确，位置是O(1/k)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stiffness", [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e8])
def test_normal_force_is_exact_and_penetration_is_one_over_k(stiffness: float):
    """**跨六个数量级的刚度，法向力一个ulp都不动。**

    这是罚函数模型唯一精确的输出：平衡条件``k·δ = m·g``直接给出
    ``N = k·δ = m·g``，``k``约掉了。而位置误差``δ = N/k``是模型自带的穿透。

    实测（写下来免得将来有人以为"差不多"）：六档刚度的``N``相对偏差**全是0.0**，
    ``δ``与``N/k``逐位相同。
    """

    contact_layout, context, term = _setup(stiffness)
    registry = EnergyRegistry(terms=(UniformGravity(), term))
    fixed = frozenset({0, 1} | set(range(3, contact_layout.layout.dof_count)))
    # 从"已接触但不在解上"起步——分离态起步会奇异，见本文件最后一节
    start = contact_layout.initial_vector((0.0, 0.0, -0.5 * THEORY_NORMAL_N / stiffness))

    result = solve_equilibrium(
        registry, context, contact_layout.layout, start,
        fixed_indices=fixed, residual_tol_n=1.0e-13, max_iterations=50,
    )
    assert result.converged, result.reason
    assert result.iterations == 1, (
        f"活动接触下能量是二次的，牛顿该一步到位，实测{result.iterations}步"
    )

    penetration = -result.state.vector[2]
    assert penetration == pytest.approx(THEORY_NORMAL_N / stiffness, rel=1e-12)

    normal = term.normal_force_n(result.state)[0]
    assert normal == pytest.approx(THEORY_NORMAL_N, rel=1e-14), (
        f"法向力随刚度漂了：k={stiffness:g}给出{normal}，理论{THEORY_NORMAL_N}"
    )


def test_penetration_shrinks_exactly_as_one_over_k():
    """刚度乘10，穿透除以10——**比值必须是10，不是"大致减小"**。"""

    ratios = []
    for stiffness in (1.0e3, 1.0e4, 1.0e5):
        contact_layout, context, term = _setup(stiffness)
        registry = EnergyRegistry(terms=(UniformGravity(), term))
        fixed = frozenset({0, 1} | set(range(3, contact_layout.layout.dof_count)))
        result = solve_equilibrium(
            registry, context, contact_layout.layout,
            contact_layout.initial_vector((0.0, 0.0, -0.5 * THEORY_NORMAL_N / stiffness)),
            fixed_indices=fixed, residual_tol_n=1.0e-13, max_iterations=50,
        )
        ratios.append(-result.state.vector[2])
    assert ratios[0] / ratios[1] == pytest.approx(10.0, rel=1e-12)
    assert ratios[1] / ratios[2] == pytest.approx(10.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 模型性质二：分离态切线刚度奇异（**这是模型的性质，不是缺陷**）
# ---------------------------------------------------------------------------


def test_newton_cannot_find_contact_from_a_separated_start():
    """悬在半空的自由度**没有任何刚度**，牛顿迈不出第一步。

    0050第四节登记的非光滑性在这里第一次有了正向的门：
    ``U = ½kg²·[g<0]``的二阶导在``g = 0``处从``k``跳到``0``，
    分离侧那一半是**零**，于是切线刚度奇异、`_solve_dense`失败关闭。

    **这条要被钉住，否则将来有人会把它当bug"顺手"正则化掉**——
    而那样做等于给分离的接触也加上刚度，即一个"隔空吸引"的假物理。
    真正的解法是载荷步或活动集策略，不是把奇异抹平。
    """

    contact_layout, context, term = _setup(1.0e4)
    registry = EnergyRegistry(terms=(UniformGravity(), term))
    fixed = frozenset({0, 1} | set(range(3, contact_layout.layout.dof_count)))
    with pytest.raises(SolveError, match="singular"):
        solve_equilibrium(
            registry, context, contact_layout.layout,
            contact_layout.initial_vector((0.0, 0.0, 1.0)),  # 悬在面上方1mm
            fixed_indices=fixed, residual_tol_n=1.0e-12, max_iterations=50,
        )


def test_a_separated_contact_contributes_nothing_at_all():
    """分离时能量、梯度、Hessian非零项**全为零**——不是"很小"，是零。"""

    contact_layout, context, term = _setup(1.0e4)
    state = _state(contact_layout, 5.0)
    assert term.energy(state, context) == 0.0
    assert set(term.gradient(state, context)) == {0.0}
    assert term.hessian_entries(state, context) == ()


# ---------------------------------------------------------------------------
# 协议契约
# ---------------------------------------------------------------------------


def test_energy_is_the_hand_computed_half_k_g_squared():
    """手算数：``g = −0.3``、``k = 1000`` → ``U = ½·1000·0.09 = 45`` N·mm。"""

    contact_layout, context, term = _setup(1000.0)
    state = _state(contact_layout, -0.3)
    assert term.energy(state, context) == pytest.approx(45.0, rel=1e-15)


def test_fused_path_matches_the_separate_calls_bit_for_bit():
    """spec/12第3.1节的承重条款。"""

    contact_layout, context, term = _setup(1234.5)
    state = _state(contact_layout, -0.271828)
    fused_energy, fused_gradient, fused_hessian = term.quantities(
        state, context, need_gradient=True, need_hessian=True
    )
    assert fused_energy == term.energy(state, context)
    assert fused_gradient == term.gradient(state, context)
    assert fused_hessian == term.hessian(state, context)


def test_hessian_matches_a_finite_difference_of_the_gradient():
    """有限差分验的是"雅可比是不是我写的那个能量的导数"，**不验能量对不对**。

    能量对不对由上面那条手算门与`normal_force_n`的精确性验
    （spec/12第6.1节：有限差分验不了物理）。
    """

    contact_layout, context, term = _setup(2000.0)
    z = -0.4
    step = 1.0e-6
    plus = term.gradient(_state(contact_layout, z + step), context)
    minus = term.gradient(_state(contact_layout, z - step), context)
    numerical = (plus[2] - minus[2]) / (2.0 * step)
    analytic = term.hessian(_state(contact_layout, z), context)[2][2]
    assert analytic == pytest.approx(2000.0, rel=1e-15)
    assert numerical == pytest.approx(analytic, rel=1e-7)


def test_hessian_is_the_outer_product_of_the_normal():
    """``k·(n ⊗ n)``——斜法向下必须出现耦合项，否则法向被当成了轴对齐的。"""

    normal = (0.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    contact_layout, context, _ = _setup(1.0)
    term = PenaltyNormalContact(planes=((0, (0.0, 0.0, 0.0), normal, 100.0, 0.0),))
    state = _state(contact_layout, -1.0)
    hessian = term.hessian(state, context)
    assert hessian[1][1] == pytest.approx(50.0, rel=1e-15)
    assert hessian[2][2] == pytest.approx(50.0, rel=1e-15)
    assert hessian[1][2] == pytest.approx(50.0, rel=1e-15), "斜法向的耦合项丢了"
    assert hessian[0][0] == 0.0


def test_the_gradient_pushes_the_node_out_not_in():
    """**符号门**：穿透时的力必须把节点推**出去**。

    符号写反了求解器照样收敛——收敛到一个"接触把东西吸进去"的世界。
    与`PointLoad`那条符号门同源。
    """

    contact_layout, context, term = _setup(1000.0)
    state = _state(contact_layout, -0.2)
    gradient = term.gradient(state, context)
    # 力 = −梯度；法向是+z，节点在下方，力必须朝+z
    assert -gradient[2] > 0.0, f"接触把节点往里吸：force_z={-gradient[2]}"
    assert -gradient[2] == pytest.approx(200.0, rel=1e-15)


def test_node_index_bound_covers_every_declared_plane():
    contact_layout, context, _ = _setup(1.0)
    term = PenaltyNormalContact(
        planes=(
            (0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 10.0, 0.0),
            (3, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 10.0, 0.0),
        )
    )
    assert term.node_index_bound() == 4


# ---------------------------------------------------------------------------
# 失败关闭
# ---------------------------------------------------------------------------


def test_a_non_unit_normal_fails_closed():
    """不归一化等于把刚度悄悄乘上``|n|²``，而调用方以为自己给的是``k``。"""

    with pytest.raises(ContactError, match="unit vector"):
        PenaltyNormalContact(planes=((0, (0.0, 0.0, 0.0), (0.0, 0.0, 2.0), 10.0, 0.0),))


def test_a_nonpositive_stiffness_fails_closed():
    with pytest.raises(ContactError, match="stiffness must be positive"):
        PenaltyNormalContact(planes=((0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, 0.0),))


def test_no_planes_fails_closed():
    with pytest.raises(ContactError, match="at least one half-space"):
        PenaltyNormalContact(planes=())


# ---------------------------------------------------------------------------
# 分离态的**报数**（2026-08-06对抗审核补：这四条此前一条门都没有）
# ---------------------------------------------------------------------------


def test_a_separated_contact_reports_zero_force_not_just_zero_energy():
    """**`test_a_separated_contact_contributes_nothing_at_all`只查了能量/梯度/Hessian。**

    对抗审核实测：把`normal_force_n`改成分离时也报``k·|g|``，
    **全部977条测试一条都不红**。而`normal_force_n`正是喂给摩擦锥的那个数
    （`advance_contact_quasistatic`），也是活动标志
    ``active = 0.0 if normal_force == 0.0 else 1.0``的**唯一依据**——
    一个飞在空中的接触会带着幽灵摩擦力，且永远被标成active。

    **能量为零不蕴含报数为零**：前者走`g < 0`分支，后者是另一段代码。
    """

    contact_layout, context, term = _setup(1.0e4)
    for height in (0.5, 5.0, 500.0):
        state = _state(contact_layout, height)
        assert term.normal_force_n(state) == (0.0,), (
            f"节点在面上方{height}mm，法向力却不是零——摩擦锥会拿到一个幽灵法向力"
        )
    # 边界：恰好接触（g == 0）也报零，与Hessian的活动集判据(`g < 0`)一致
    assert term.normal_force_n(_state(contact_layout, 0.0)) == (0.0,)
    assert term.hessian_entries(_state(contact_layout, 0.0), context) == ()


def test_a_separated_sphere_pair_reports_zero_force():
    """球-球那一侧同源，同样此前无门。"""

    from physics_engine.contact import PenaltySphereContact

    contact_layout = build_contact_layout(
        layout_id="layout/sphere-separated",
        node_count=2,
        declarations=(ContactDeclaration("pair"),),
    )
    context = EnergyContext(
        context_id="context/sphere-separated",
        node_masses_kg=(1.0, 1.0),
        gravity_mm_s2=(0.0, 0.0, 0.0),
    )
    term = PenaltySphereContact(pairs=((0, 1, 20.0, 1.0e4),))
    for separation in (20.5, 100.0):
        state = State(
            layout=contact_layout.layout,
            vector=contact_layout.initial_vector((0.0, 0.0, 0.0, separation, 0.0, 0.0)),
        )
        assert term.contact_force_n(state) == (0.0,), f"心距{separation}mm却报了接触力"
        assert term.energy(state, context) == 0.0
        assert term.hessian_entries(state, context) == ()


def test_a_zero_normal_force_gives_separated_not_stick():
    """**没有法向力就没有摩擦，而且"分离"与"在滑"是两件事。**

    `coulomb_return_map`的docstring花了一整节讲这条分界
    （"混起来会让案例分不清'飞出去了'和'在蹭着走'"），
    但对抗审核实测：把`N == 0`分支改成返回`REGIME_STICK`，**977条测试一条不红**。
    `REGIME_SEPARATED`此前在全仓测试里**只出现在出生态那条常量断言里**，
    从来没有一条测试断言过return-map会返回它。
    """

    from physics_engine.contact import (
        REGIME_SEPARATED,
        REGIME_SLIP,
        REGIME_STICK,
        coulomb_return_map,
    )

    outcome = coulomb_return_map(
        trial_force_n=(5.0, 0.0, 0.0),
        normal_force_n=0.0,
        friction_coefficient=0.3,
        tangential_stiffness_n_per_mm=1.0e4,
    )
    assert outcome.regime == REGIME_SEPARATED, (
        f"法向力为零却报了{outcome.regime}——分离被当成了粘着或滑动"
    )
    assert not outcome.is_stick
    assert outcome.tangential_force_n == (0.0, 0.0, 0.0), "没有法向力却有摩擦力"
    assert outcome.anchor_correction_mm == (0.0, 0.0, 0.0), "分离却挪了锚点（伪历史）"

    # 反向：有法向力时不许再报分离
    engaged = coulomb_return_map(
        trial_force_n=(5.0, 0.0, 0.0),
        normal_force_n=10.0,
        friction_coefficient=0.3,
        tangential_stiffness_n_per_mm=1.0e4,
    )
    assert engaged.regime in (REGIME_STICK, REGIME_SLIP)
    assert engaged.regime != REGIME_SEPARATED


def test_the_cone_boundary_is_inclusive():
    """**锥面上恰好那一点判成粘，这条边界此前从没被真的取到过。**

    清册把它登记成一条载荷条款（"`|T| ≤ μN`把边界判成粘，故1趟报滑、2趟报粘，
    改默认趟数会改语义"），而`test_contact_stepper`里那条看似在钉它的断言，
    实测那里的量比``μN``小约8个ulp——**`<=`与`<`在那里给同一个答案**。

    这里**构造精确落在锥面上的试探力**，把边界真的取到。
    """

    from physics_engine.contact import REGIME_STICK, coulomb_return_map

    normal_force = 8.0
    friction = 0.25
    limit = friction * normal_force  # 恰为2.0，二进制可精确表示
    outcome = coulomb_return_map(
        trial_force_n=(limit, 0.0, 0.0),
        normal_force_n=normal_force,
        friction_coefficient=friction,
        tangential_stiffness_n_per_mm=1.0e4,
    )
    assert outcome.regime == REGIME_STICK, "恰在锥面上被判成滑——边界约定变了"
    assert outcome.anchor_correction_mm == (0.0, 0.0, 0.0), "边界上不许挪锚点"
    assert outcome.tangential_force_n == (limit, 0.0, 0.0), "边界上力被投影动了"
