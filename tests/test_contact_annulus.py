"""法兰内环面对带材边缘的单边接触——蹭边（决策0062轨道甲第二片，能力位S6.6）。

## 这一项与前两项的分界

| 项 | 间隙 | Hessian |
|---|---|---|
| `PenaltyNormalContact`（半空间） | `(x−p)·n − r`，**线性** | `k·(n⊗n)` |
| `PenaltyCylinderContact`（圆柱侧面） | `ρ − R`，**非线性** | `k·(n⊗n) + (kg/ρ)(t⊗t)` |
| `PenaltyAnnulusLimit`（本项） | `inward·(L − s_e)`，**线性** | `k·(a⊗a)`，**没有几何刚度** |

**别照抄圆柱那一项的Hessian**——本项的间隙是位置的线性函数，二阶导恰为零。
有一条门判它。

## 单边是这一项的要害

两片法兰各是一个独立的限位面。**一片被顶住时另一片必须一个牛顿都不出**。
互补条件``g > 0 ⟹ f ≡ 0``在这里是**零容差**判据。

真机数字：导轮有效宽度17 mm、带宽4 mm ⟹ **半间隙恰好6.5 mm**。
带材横向跑偏超过6.5 mm才蹭得上，不到就一点力都没有。

## 必红矩阵（2026-08-17逐条注错**实测**）

见文件末尾`test_the_mutation_matrix_is_measured`的docstring。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import ContactError, PenaltyAnnulusLimit
from physics_engine.energies import EnergyContext, EnergyRegistry, PointLoad
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State, StateField, StateLayout

#: 筒半径与法兰外径：`modelgen.generate_spool`产出的`barrel` / `flange_*`那两个数。
BARREL_RADIUS_MM = 60.0
FLANGE_OUTER_MM = 75.0
#: 槽半宽：真机导轮有效宽度17 mm ⟹ 8.5 mm（`HARDWARE_TOPOLOGY.md`，2026-07-07现场确认）。
CHANNEL_HALF_WIDTH_MM = 8.5
#: 带半宽：4 mm宽REBCO带材 ⟹ 2.0 mm。
TAPE_HALF_WIDTH_MM = 2.0
#: **半间隙**：跑偏超过它才蹭得上。``8.5 − 2.0 = 6.5``。
HALF_CLEARANCE_MM = CHANNEL_HALF_WIDTH_MM - TAPE_HALF_WIDTH_MM
STIFFNESS_N_PER_MM = 1.0e4
AXIS_Z = (0.0, 0.0, 1.0)
ORIGIN = (0.0, 0.0, 0.0)


def _layout() -> StateLayout:
    return StateLayout(
        layout_id="layout/annulus",
        fields=(
            StateField("node0_x_mm", 1),
            StateField("node0_y_mm", 1),
            StateField("node0_z_mm", 1),
        ),
    )


CONTEXT = EnergyContext(
    context_id="context/annulus",
    node_masses_kg=(1.0e-9,),
    gravity_mm_s2=(0.0, 0.0, 0.0),
)


def _flanges(
    *,
    axis: tuple[float, float, float] = AXIS_Z,
    stiffness: float = STIFFNESS_N_PER_MM,
    inner: float = BARREL_RADIUS_MM,
    outer: float = FLANGE_OUTER_MM,
) -> PenaltyAnnulusLimit:
    """两片法兰：``+``号那片限住``+a``方向的边缘，``−``号那片限住另一侧。"""

    return PenaltyAnnulusLimit(
        faces=(
            (0, ORIGIN, axis, inner, outer, CHANNEL_HALF_WIDTH_MM, 1.0,
             TAPE_HALF_WIDTH_MM, stiffness),
            (0, ORIGIN, axis, inner, outer, -CHANNEL_HALF_WIDTH_MM, -1.0,
             -TAPE_HALF_WIDTH_MM, stiffness),
        )
    )


def _state(position: tuple[float, float, float]) -> State:
    return State(layout=_layout(), vector=position)


# ---------------------------------------------------------------------------
# 半间隙：蹭不蹭得上就看这一个数
# ---------------------------------------------------------------------------


def test_the_half_clearance_is_the_channel_minus_the_tape():
    """带材居中时，两侧间隙都恰是``(W − w)/2 = 6.5 mm``。

    这是蹭边判据的全部：**跑偏超过6.5 mm才蹭得上**。
    真机数字——导轮有效宽度17 mm、带宽4 mm。
    """

    term = _flanges()
    high, low = term.edge_clearance_mm(_state((62.0, 0.0, 0.0)))
    assert high == pytest.approx(HALF_CLEARANCE_MM, rel=1e-15)
    assert low == pytest.approx(HALF_CLEARANCE_MM, rel=1e-15)


@pytest.mark.parametrize(
    ("offset", "rubs"),
    [(0.0, False), (6.0, False), (6.5, False), (6.5 + 1e-12, True), (7.0, True)],
)
def test_the_rub_starts_exactly_at_the_half_clearance(offset: float, rubs: bool):
    """阈值两侧行为定性相反，**零容差**。

    ``g = 0``恰好落在活动集外（判据是``g < 0``），与本仓其余罚接触同口径：
    **边界闭开是一条声明**。
    """

    term = _flanges()
    force = term.rub_force_n(_state((62.0, 0.0, offset)))[0]
    assert (force > 0.0) is rubs, f"偏移{offset}处蹭边判定与预期相反：力={force!r}"


def test_only_one_flange_ever_carries_load():
    """**单边**：一片被顶住时另一片必须一个牛顿都不出。

    带材不可能同时贴住两侧还各受一个法向力——除非槽宽比带宽还窄，
    而那是几何声明本身就错了。这条互补是零容差判据。
    """

    term = _flanges()
    for offset in (7.0, 8.0, 20.0, -7.0, -8.0, -20.0):
        forces = term.rub_force_n(_state((62.0, 0.0, offset)))
        active = [index for index, value in enumerate(forces) if value != 0.0]
        assert len(active) == 1, f"偏移{offset}处有{len(active)}片法兰在出力：{forces}"


def test_the_two_flanges_are_mirror_antisymmetric():
    """``±offset``给出**逐位相同**的蹭边力，只是换了一片法兰。

    这条是J3（镜像反对称）在蹭边上的落点：几何是对称的，
    任何把某一侧的符号写错的实现都会在这里破缺。
    """

    term = _flanges()
    for offset in (7.0, 9.5, 30.0):
        high = term.rub_force_n(_state((62.0, 0.0, offset)))
        low = term.rub_force_n(_state((62.0, 0.0, -offset)))
        assert high[0] == low[1], f"偏移±{offset}的蹭边力不对称：{high} vs {low}"
        assert high[1] == low[0] == 0.0


# ---------------------------------------------------------------------------
# 边缘偏移：拿中心线判蹭边会差半个带宽
# ---------------------------------------------------------------------------


def test_ignoring_the_edge_offset_moves_the_threshold_by_half_the_tape_width():
    """**中心线不是边缘。**

    忽略``e``等于把阈值从``6.5``挪到``8.5``——**差整整2.0 mm，即半个带宽**。
    在那个区间里带材已经蹭上而模型说没有。本门量的就是那个区间。
    """

    term = _flanges()
    centreline = PenaltyAnnulusLimit(
        faces=tuple(
            (node, point, axis, inner, outer, limit, inward, 0.0, stiffness)
            for node, point, axis, inner, outer, limit, inward, _, stiffness in term.faces
        )
    )
    for offset in (6.6, 7.5, 8.4):
        assert term.rub_force_n(_state((62.0, 0.0, offset)))[0] > 0.0
        assert centreline.rub_force_n(_state((62.0, 0.0, offset)))[0] == 0.0
    gap = CHANNEL_HALF_WIDTH_MM - HALF_CLEARANCE_MM
    assert gap == pytest.approx(TAPE_HALF_WIDTH_MM, rel=1e-15)


def test_the_radial_distance_is_unchanged_by_the_edge_offset():
    """边缘点与中心线的``ρ``**相同**——沿轴平移不改变到轴的距离。

    这不是近似，是``|d − (d·a)a|``对``d → d + e·a``不变。
    写成门是因为下一个人可能以为要重算一次，然后算出一个不同的数。
    """

    term = _flanges()
    for position in ((62.0, 0.0, 0.0), (40.0, 45.0, 7.3), (-61.0, 3.0, -9.0)):
        expected = math.hypot(position[0], position[1])
        for distance in term.radial_distance_mm(_state(position)):
            assert distance == pytest.approx(expected, rel=1e-15)


# ---------------------------------------------------------------------------
# 环带：法兰只在它的径向范围里存在
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [BARREL_RADIUS_MM - 1.0, FLANGE_OUTER_MM + 1.0])
def test_outside_the_annulus_the_flange_is_not_there(radius: float):
    """径向越出``[R_筒, R_法兰]``时四样输出全零。

    内侧越出＝带材还没绕到法兰的径向范围；外侧越出＝已经绕过法兰外径。
    两端各判一次——**一端的绿证明不了另一端**。
    """

    term = _flanges()
    state = _state((radius, 0.0, 20.0))
    assert term.energy(state, CONTEXT) == 0.0
    assert term.gradient(state, CONTEXT) == (0.0, 0.0, 0.0)
    assert term.hessian_entries(state, CONTEXT) == ()
    assert term.rub_force_n(state) == (0.0, 0.0)


@pytest.mark.parametrize("radius", [BARREL_RADIUS_MM, FLANGE_OUTER_MM])
def test_the_annulus_boundaries_are_inclusive(radius: float):
    """环带两端**算在里面**，与摩擦锥、轴向端沿同一口径。"""

    term = _flanges()
    assert term.rub_force_n(_state((radius, 0.0, 20.0)))[0] > 0.0


# ---------------------------------------------------------------------------
# Hessian：没有几何刚度
# ---------------------------------------------------------------------------


def test_the_hessian_is_exactly_the_axis_outer_product():
    """``H = k·(a⊗a)``，**一块，没有几何刚度**。

    **这条门守的是没有人照抄圆柱那一项。** 圆柱的间隙``ρ − R``是非线性的，
    于是多一块周向softening；本项的间隙是位置的线性函数，二阶导恰为零。
    抄过来会在垂直于轴的方向上多出``kg/ρ``——梯度照样对、平衡点照样对。
    """

    term = _flanges()
    state = _state((62.0, 0.0, 7.0))
    hessian = term.hessian(state, CONTEXT)
    for row in range(3):
        for column in range(3):
            expected = STIFFNESS_N_PER_MM * AXIS_Z[row] * AXIS_Z[column]
            assert hessian[row][column] == expected, (
                f"H[{row}][{column}]={hessian[row][column]!r}，期望{expected!r}——"
                "多出来的项多半是从圆柱那一项抄来的几何刚度"
            )


def test_hessian_matches_a_finite_difference_of_the_gradient():
    term = _flanges()
    position = (62.0, 0.0, 7.0)
    analytic = term.hessian(_state(position), CONTEXT)
    step = 1.0e-6
    for column in range(3):
        forward, backward = list(position), list(position)
        forward[column] += step
        backward[column] -= step
        high = term.gradient(_state(tuple(forward)), CONTEXT)
        low = term.gradient(_state(tuple(backward)), CONTEXT)
        for row in range(3):
            numeric = (high[row] - low[row]) / (2.0 * step)
            assert numeric == pytest.approx(analytic[row][column], rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# 力精确、位置是O(1/k)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stiffness", [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
def test_the_rub_force_equals_the_lateral_load_exactly(stiffness: float):
    """横向载荷把带材推到法兰上，平衡时蹭边力**恰等于横载**，与``k``无关。

    与半空间同构：**位置有``O(1/k)``的穿透误差，力没有**。

    **但那条`k·ulp(坐标)`的相消地板仍在，只是坐标换了。**
    写这条门时我判定"``s``是``O(10)``而不是``O(R)``，所以没有圆柱那条地板"——
    **实测否掉**：地板是``k·ulp(8.5)``，坐标是**槽半宽**不是筒半径，
    于是比圆柱那条（``k·ulp(50)``）小约6倍，**但不是零**。
    2026-08-17五档刚度实测比值``0.640 / 0.064 / 0.394 / 0.639 / 0.736``，全部小于1。

    通则：**罚接触的可达精度是``k`` 乘以"间隙表达式里被相减的那个量"的ulp**。
    半空间那里那个量是0（面过原点），所以只有那里能声称"一个ulp都不动"。
    """

    lateral_n = 3.0
    term = _flanges(stiffness=stiffness)
    registry = EnergyRegistry(terms=(term, PointLoad(loads=((0, (0.0, 0.0, lateral_n)),))))
    floor = stiffness * math.ulp(CHANNEL_HALF_WIDTH_MM)
    result = solve_equilibrium(
        registry,
        CONTEXT,
        _layout(),
        (62.0, 0.0, HALF_CLEARANCE_MM + 0.5 * lateral_n / stiffness),
        fixed_indices=frozenset({0, 1}),
        residual_tol_n=2.0 * floor,
    )
    assert result.converged, result.reason
    assert result.iterations == 1, (
        f"能量在活动集内严格二次，牛顿该一步到位，实测{result.iterations}步"
    )
    assert term.rub_force_n(result.state)[0] == pytest.approx(lateral_n, abs=2.0 * floor)
    penetration = -term.edge_clearance_mm(result.state)[0]
    assert penetration == pytest.approx(
        lateral_n / stiffness, abs=2.0 * math.ulp(CHANNEL_HALF_WIDTH_MM)
    )


@pytest.mark.parametrize("stiffness", [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
def test_the_residual_floor_is_the_stiffness_times_one_ulp_of_the_channel(
    stiffness: float,
):
    """**同一条地板律，第三次出现，坐标第三次不同。**

    `PenaltyCylinderContact`那里是``k·ulp(R)``（``R = 50 mm``筒半径），
    这里是``k·ulp(W/2)``（``8.5 mm``槽半宽），`PenaltyNormalContact`那里
    面过原点、被相减的量是0，所以只有那里能声称"跨六个数量级一个ulp都不动"。

    **判据落在"地板存在且不超过``k·ulp``"上**，不写死具体值——
    具体值随构型变，律不变。
    """

    lateral_n = 3.0
    term = _flanges(stiffness=stiffness)
    registry = EnergyRegistry(terms=(term, PointLoad(loads=((0, (0.0, 0.0, lateral_n)),))))
    result = solve_equilibrium(
        registry,
        CONTEXT,
        _layout(),
        (62.0, 0.0, HALF_CLEARANCE_MM + 0.5 * lateral_n / stiffness),
        fixed_indices=frozenset({0, 1}),
        residual_tol_n=1.0e-18,
        max_iterations=60,
    )
    assert not result.converged, "容差取1e-18还能收敛，说明这条地板不存在——判据要重写"
    floor = stiffness * math.ulp(CHANNEL_HALF_WIDTH_MM)
    assert result.residual_n <= floor, (
        f"k={stiffness:g}：残差{result.residual_n:.4e}超过k·ulp(W/2)={floor:.4e}，"
        "与2026-08-17的五组实测不符"
    )


# ---------------------------------------------------------------------------
# 倾斜轴
# ---------------------------------------------------------------------------


def test_a_tilted_axis_keeps_every_invariant():
    """轴取``(1,1,1)/√3``：一个把轴硬编码成``z``的实现要在这里死。"""

    axis = tuple(1.0 / math.sqrt(3.0) for _ in range(3))
    term = _flanges(axis=axis, inner=0.0, outer=1.0e6)
    #: 沿轴走到``s = 7.0``：位置就是``7·a``，径向距离为零，故环带内半径取0。
    position = tuple(7.0 * component for component in axis)
    state = _state(position)
    high, low = term.edge_clearance_mm(state)
    assert high == pytest.approx(CHANNEL_HALF_WIDTH_MM - (7.0 + TAPE_HALF_WIDTH_MM))
    assert low == pytest.approx((7.0 - TAPE_HALF_WIDTH_MM) + CHANNEL_HALF_WIDTH_MM)

    hessian = term.hessian(state, CONTEXT)
    for row in range(3):
        for column in range(3):
            assert hessian[row][column] == pytest.approx(
                STIFFNESS_N_PER_MM * axis[row] * axis[column], rel=1e-12
            )


# ---------------------------------------------------------------------------
# 协议契约与失败关闭
# ---------------------------------------------------------------------------


def test_the_fused_path_matches_the_separate_calls_bit_for_bit():
    term = _flanges()
    state = _state((62.0, 0.0, 7.3))
    energy, gradient, hessian = term.quantities(
        state, CONTEXT, need_gradient=True, need_hessian=True
    )
    assert energy == term.energy(state, CONTEXT)
    assert gradient == term.gradient(state, CONTEXT)
    assert hessian == term.hessian(state, CONTEXT)


def test_the_gradient_pushes_the_edge_back_into_the_channel():
    """势能的负梯度是力：蹭上时力必须把带材推**回槽里**。"""

    term = _flanges()
    high_force = tuple(-value for value in term.gradient(_state((62.0, 0.0, 7.0)), CONTEXT))
    assert high_force[2] < 0.0, "顶住上侧法兰时力该指向−z"
    low_force = tuple(-value for value in term.gradient(_state((62.0, 0.0, -7.0)), CONTEXT))
    assert low_force[2] > 0.0, "顶住下侧法兰时力该指向+z"


@pytest.mark.parametrize(
    ("face", "message"),
    [
        ((0, ORIGIN, AXIS_Z, 60.0, 75.0, 8.5, 0.0, 2.0, 1.0), "inward must be exactly"),
        ((0, ORIGIN, AXIS_Z, 60.0, 75.0, 8.5, 2.0, 2.0, 1.0), "inward must be exactly"),
        ((0, ORIGIN, AXIS_Z, 60.0, 75.0, float("nan"), 1.0, 2.0, 1.0), "limit must be finite"),
        ((0, ORIGIN, AXIS_Z, 75.0, 60.0, 8.5, 1.0, 2.0, 1.0), "outer radius must exceed"),
        ((0, ORIGIN, AXIS_Z, 60.0, 60.0, 8.5, 1.0, 2.0, 1.0), "outer radius must exceed"),
        ((0, ORIGIN, (0.0, 0.0, 2.0), 60.0, 75.0, 8.5, 1.0, 2.0, 1.0), "unit vector"),
        ((0, ORIGIN, AXIS_Z, -1.0, 75.0, 8.5, 1.0, 2.0, 1.0), "inner radius"),
        ((0, ORIGIN, AXIS_Z, 60.0, 75.0, 8.5, 1.0, 2.0, 0.0), "stiffness must be"),
        ((-1, ORIGIN, AXIS_Z, 60.0, 75.0, 8.5, 1.0, 2.0, 1.0), "nonnegative int"),
        ((0, ORIGIN, AXIS_Z, 60.0, 75.0, 8.5, 1.0, float("nan"), 1.0), "edge offset"),
    ],
)
def test_malformed_faces_fail_closed(face, message):
    with pytest.raises(ContactError, match=message):
        PenaltyAnnulusLimit(faces=(face,))


def test_no_faces_fails_closed():
    with pytest.raises(ContactError, match="at least one face"):
        PenaltyAnnulusLimit()


def test_the_mutation_matrix_is_measured():
    """必红矩阵（2026-08-17逐条注错**实测**，数字是本文件红掉的条数）。

    | 注错 | 红掉 |
    |---|---|
    | 单边写成双边（``abs(g)``） | 21 |
    | 边缘偏移被忽略（拿中心线判蹭边） | 19 |
    | Hessian照抄圆柱的几何刚度 | 3 |
    | 环带判据丢掉（法兰变成无限大圆盘） | 2 |
    | 环带边界写成开区间 | 2 |
    | **朝向从``limit``的符号推出来**（横动过原点即失效） | **1** |
    | 能量漏掉``½`` | 1 |

    **七条全被抓到，最低一条。** 前两行红得多，是因为它们改掉了**间隙本身**，
    而本文件几乎每条门都读间隙。要紧的是红得少的那四行：

    * **朝向从``limit``符号推出来只红1条**，而那一条
      （`test_a_traversed_channel_keeps_both_faces_pointing_the_right_way`）
      **是端到端装配跑出来之后才补的**。在它之前这条注错**红0条**——
      本文件所有构型的槽心都在原点，那里位置的符号与朝向恒等。
      **只有端到端才发现得了它**；
    * **Hessian照抄圆柱的几何刚度**只红3条。梯度照样对、平衡点照样对，
      与`test_contact_cylinder.py`里"照抄球-球的几何刚度"是同一形态的反向；
    * **能量漏掉``½``只红1条**——力与阈值判据都只看梯度，
      而``½``已被求导吃掉，唯一看得见它的是判能量值的那条门。

    本条只承载这张表，不做断言——数字的证据是实测脚本，
    与`test_contact_cylinder.py`同一形制。
    """


def test_a_traversed_channel_keeps_both_faces_pointing_the_right_way():
    """**这一条就是2026-08-17端到端跑出来的那个bug。**

    第一版把朝向编码在``limit``的**符号**里。收线盘排线横动到9 mm时，
    下侧法兰的位置变成``9 − 8.5 = +0.5``——**符号翻了**，那一片被当成上侧法兰，
    于是判据方向反了、**蹭边力凭空归零**：横动7 mm与8 mm都算得出2.46 N与7.44 N，
    唯独9 mm给0。

    病根是**位置的符号与朝向是两件事**，只在槽心恰好在原点时才碰巧一致。
    单元门里的构型永远是槽心在原点的，**所以只有端到端装配发现得了它**。

    本门把槽心横动到9 mm（整条槽``[0.5, 17.5]``都在正半轴），
    逐点判两片法兰各自朝向正确。
    """

    traverse = 9.0
    term = PenaltyAnnulusLimit(
        faces=(
            (0, ORIGIN, AXIS_Z, 0.0, FLANGE_OUTER_MM,
             traverse + CHANNEL_HALF_WIDTH_MM, 1.0, TAPE_HALF_WIDTH_MM,
             STIFFNESS_N_PER_MM),
            (0, ORIGIN, AXIS_Z, 0.0, FLANGE_OUTER_MM,
             traverse - CHANNEL_HALF_WIDTH_MM, -1.0, -TAPE_HALF_WIDTH_MM,
             STIFFNESS_N_PER_MM),
        )
    )
    #: 槽中心在9.0，带材可待的区间是``[9−6.5, 9+6.5] = [2.5, 15.5]``。
    low_bound = traverse - HALF_CLEARANCE_MM
    high_bound = traverse + HALF_CLEARANCE_MM
    assert low_bound == 2.5 and high_bound == 15.5

    for inside in (2.5, 9.0, 15.5):
        assert term.rub_force_n(_state((60.0, 0.0, inside))) == (0.0, 0.0), (
            f"z={inside}在槽内却报了蹭边力"
        )
    #: 低于下界 → **下侧**那片出力；高于上界 → **上侧**那片。
    low = term.rub_force_n(_state((60.0, 0.0, low_bound - 0.5)))
    assert low[0] == 0.0 and low[1] == pytest.approx(0.5 * STIFFNESS_N_PER_MM)
    high = term.rub_force_n(_state((60.0, 0.0, high_bound + 0.5)))
    assert high[0] == pytest.approx(0.5 * STIFFNESS_N_PER_MM) and high[1] == 0.0

    #: **旧写法在这里给零**——把朝向从limit符号推出来，两片都变成"朝下"。
    inferred = PenaltyAnnulusLimit(
        faces=tuple(
            (node, point, axis, inner, outer, limit,
             1.0 if limit > 0.0 else -1.0, offset, stiffness)
            for node, point, axis, inner, outer, limit, _, offset, stiffness in term.faces
        )
    )
    assert inferred.rub_force_n(_state((60.0, 0.0, low_bound - 0.5))) == (0.0, 0.0), (
        "旧写法在横动过原点的槽上不再给零——那本门记的病根就要重写"
    )
