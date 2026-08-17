"""有限长圆柱侧面罚接触的门（决策0062轨道甲第一片，能力位S6.5）。

本文件守三类事：**协议契约**（四方法齐备、融合路径逐字节、Hessian对得上梯度）、
**这一项独有的两条几何性质**、以及**活动集的两条判据各自的必红**。

## 独有的两条几何性质——它们是本项与半空间/球-球的分界

1. **法向恒垂直于轴**（``n·a = 0``）。半空间的法向是常量、球-球的法向是心连线，
   只有圆柱的法向被轴投影掉一个方向。
2. **Hessian在轴方向上恒为零**。沿轴移动不改变到轴的距离，所以
   ``H·a = 0``是精确的，不是"很小"。这条同时是几何刚度只出现在**周向**的直接后果：

       H = k·(n⊗n) + (k·g/ρ)·(t⊗t)，  t = a × n

   球-球那项的几何刚度是``(kg/L)(I − d⊗d)``——**两个**横向；这里只有一个。

## 必红矩阵（2026-08-17逐条注错**实测**，数字是本文件红掉的条数）

| 注错 | 红掉 |
|---|---|
| 法向不做轴向投影（当成球心距） | 8 |
| 两条活动判据写成``or``而不是``and`` | 6 |
| 活动集丢掉``|s| ≤ half_width``（变成无限长圆柱） | 4 |
| 轴向判据漏掉绝对值（``s ≤ half_width``） | 2 |
| 几何刚度整块丢掉（只留``k·n⊗n``） | 2 |
| 几何刚度照抄球-球的``(kg/ρ)(I − n⊗n)``（多一个轴向） | 2 |
| 能量漏掉``½`` | 2 |
| 轴上奇点不抛、静默取``+x``方向 | 1 |
| 节点半径被忽略（接触面不外移） | 1 |

**九条注错全部被抓到，最低一条。**

**第六行是这张表里最要紧的一行**：把球-球的几何刚度照抄过来，梯度照样对、
平衡点照样对，只有轴方向上多出一块本不该有的刚度。它是"抄相邻代码"的
典型失败模式，抓住它的是`test_the_hessian_is_exactly_zero_along_the_axis`。

**第三、四行分家不是凑数**：轴向判据整条丢掉红4条，只漏掉绝对值红2条——
后者在``+z``端行为完全正确，只在``−z``端出错。**一端的绿证明不了另一端**，
所以那条门按两端参数化（plans/09教训三：必红要覆盖判据的每个分支，不是每个规则）。

**倾斜轴的门单列**（`test_a_tilted_axis_keeps_every_invariant`）：
本文件多数构型的轴是``+z``或``+x``，都是坐标轴，而**坐标轴上"投影掉轴分量"
与"把某个分量置零"是同一件事**，一个把轴硬编码的实现能从别的门底下走过去——
这与`test_contact_normal.py`开头记的那条教训是同一个形态。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import ContactError, PenaltyCylinderContact
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State, StateField, StateLayout

RADIUS_MM = 50.0
HALF_WIDTH_MM = 8.5
STIFFNESS_N_PER_MM = 1.0e4
AXIS_POINT = (0.0, 0.0, 0.0)
AXIS_Z = (0.0, 0.0, 1.0)

MASS_KG = 2.0
GRAVITY_MM_S2 = 9810.0
#: ``N = m·g``；``g``是mm/s²而牛顿要的是m/s²，故除以1000（与`UniformGravity`同源）。
THEORY_NORMAL_N = MASS_KG * GRAVITY_MM_S2 / 1000.0


def _layout(nodes: int = 1) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/cylinder-n{nodes}",
        fields=tuple(
            field
            for index in range(nodes)
            for field in (
                StateField(f"node{index}_x_mm", 1),
                StateField(f"node{index}_y_mm", 1),
                StateField(f"node{index}_z_mm", 1),
            )
        ),
    )


CONTEXT = EnergyContext(
    context_id="context/cylinder",
    node_masses_kg=(MASS_KG,),
    gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
)


def _term(
    *,
    stiffness: float = STIFFNESS_N_PER_MM,
    half_width: float = HALF_WIDTH_MM,
    axis: tuple[float, float, float] = AXIS_Z,
    point: tuple[float, float, float] = AXIS_POINT,
    node_radius: float = 0.0,
) -> PenaltyCylinderContact:
    return PenaltyCylinderContact(
        cylinders=((0, point, axis, RADIUS_MM, half_width, stiffness, node_radius),)
    )


def _state(position: tuple[float, float, float], nodes: int = 1) -> State:
    vector = list(position) + [0.0] * (3 * (nodes - 1))
    return State(layout=_layout(nodes), vector=tuple(vector))


# ---------------------------------------------------------------------------
# 独有几何性质一：法向恒垂直于轴
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position",
    [
        (49.0, 0.0, 0.0),
        (0.0, 48.5, 3.0),
        (30.0, 30.0, -2.0),
        (-20.0, 40.0, 8.0),
    ],
)
def test_the_outward_normal_is_perpendicular_to_the_axis(position):
    """``n·a``必须**恰为0**，不是"很小"。

    径向矢量是``d − (d·a)a``，与``a``的内积在精确算术下恒为零；
    浮点下它是两个乘积相减，量级``eps·|d|``。判据取``4·eps·|d|``——
    **判绝对不判相对**，因为这个量的真值就是0（与艾里斑判J1零点同理）。
    """

    term = _term()
    state = _state(position)
    (normal,) = term.outward_normal(state)
    dot = sum(normal[axis] * AXIS_Z[axis] for axis in range(3))
    magnitude = math.sqrt(sum(component * component for component in position))
    assert abs(dot) <= 4.0 * math.ulp(1.0) * magnitude

    unit = math.sqrt(sum(component * component for component in normal))
    assert abs(unit - 1.0) <= 8.0 * math.ulp(1.0)


# ---------------------------------------------------------------------------
# 独有几何性质二：Hessian在轴方向上恒为零
# ---------------------------------------------------------------------------


def test_the_hessian_is_exactly_zero_along_the_axis():
    """``H·a = 0``是精确的。

    **这一条是本文件的承重门**：把球-球的几何刚度``(kg/ρ)(I − n⊗n)``照抄过来，
    梯度照样对、平衡点照样对，唯独轴方向上会多出``kg/ρ``一块。
    圆柱的几何决定了沿轴移动不改变``ρ``，所以那一块不存在。
    """

    term = _term()
    state = _state((30.0, 30.0, 1.0))
    hessian = term.hessian(state, CONTEXT)

    for row in range(3):
        along_axis = sum(hessian[row][column] * AXIS_Z[column] for column in range(3))
        assert along_axis == 0.0, f"row {row} has stiffness along the axis: {along_axis!r}"


def test_the_geometric_stiffness_is_circumferential_softening():
    """周向的二次型必须是**负的**，且恰等于``k·g/ρ``。

    ``g < 0``（活动时必然），故这一块是softening——与`PenaltySphereContact`
    压缩时的横向softening同源。**它的大小也被判**，不只判符号：
    只判符号的话把系数写成``k·g/(2ρ)``照样绿。
    """

    term = _term()
    position = (30.0, 30.0, 1.0)
    state = _state(position)
    hessian = term.hessian(state, CONTEXT)
    (normal,) = term.outward_normal(state)
    (distance,) = term.radial_distance_mm(state)
    gap = distance - RADIUS_MM
    assert gap < 0.0

    circumferential = (
        AXIS_Z[1] * normal[2] - AXIS_Z[2] * normal[1],
        AXIS_Z[2] * normal[0] - AXIS_Z[0] * normal[2],
        AXIS_Z[0] * normal[1] - AXIS_Z[1] * normal[0],
    )
    quadratic = sum(
        circumferential[row] * hessian[row][column] * circumferential[column]
        for row in range(3)
        for column in range(3)
    )
    expected = STIFFNESS_N_PER_MM * gap / distance
    assert quadratic < 0.0
    assert quadratic == pytest.approx(expected, rel=1e-12)


def test_hessian_matches_a_finite_difference_of_the_gradient():
    """一致切线：整块Hessian对梯度的中心差分。

    **这条同时验几何刚度**——丢掉它中心差分立刻对不上，
    因为那一块在``ρ = 50mm``、``g``约``−1mm``时是``k/50``量级，不是舍入噪声。
    """

    term = _term()
    position = (33.0, 27.0, 2.0)
    state = _state(position)
    analytic = term.hessian(state, CONTEXT)
    step = 1.0e-6

    for column in range(3):
        forward = list(position)
        backward = list(position)
        forward[column] += step
        backward[column] -= step
        high = term.gradient(_state(tuple(forward)), CONTEXT)
        low = term.gradient(_state(tuple(backward)), CONTEXT)
        for row in range(3):
            numeric = (high[row] - low[row]) / (2.0 * step)
            assert numeric == pytest.approx(analytic[row][column], rel=2e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# 活动集的两条判据
# ---------------------------------------------------------------------------


def test_a_separated_node_contributes_nothing_at_all():
    """``g > 0``：能量、梯度、Hessian**三样全零**，不是"很小"。"""

    term = _term()
    state = _state((60.0, 0.0, 0.0))
    assert term.energy(state, CONTEXT) == 0.0
    assert term.gradient(state, CONTEXT) == (0.0, 0.0, 0.0)
    assert term.hessian_entries(state, CONTEXT) == ()
    assert term.normal_force_n(state) == (0.0,)


@pytest.mark.parametrize("side", [1.0, -1.0], ids=["high_end", "low_end"])
def test_a_node_past_the_axial_edge_gets_no_force_even_while_penetrating(side: float):
    """**活动集的第二条判据**：轴向越出筒宽后，即便径向仍在穿透也不出力。

    这是"侧面"这个词的兑现——节点轴向越过端沿之后，侧面在那里不存在。
    把这一条丢掉就等于声明了一个**无限长**圆柱，而那不是任何真实导轮。

    **两端各判一次不是凑数**：判据是``|s| ≤ half_width``，
    漏掉绝对值的写法（``s ≤ half_width``）在``+z``端表现完全正确，
    只在``−z``端把整条筒外的半空间judged成活动。**一端的绿证明不了另一端。**
    """

    term = _term()
    inside = _state((49.0, 0.0, side * (HALF_WIDTH_MM - 0.1)))
    outside = _state((49.0, 0.0, side * (HALF_WIDTH_MM + 0.1)))

    assert term.normal_force_n(inside)[0] > 0.0
    assert term.energy(outside, CONTEXT) == 0.0
    assert term.gradient(outside, CONTEXT) == (0.0, 0.0, 0.0)
    assert term.hessian_entries(outside, CONTEXT) == ()
    assert term.normal_force_n(outside) == (0.0,)


@pytest.mark.parametrize("side", [1.0, -1.0], ids=["high_end", "low_end"])
def test_deep_radial_penetration_past_the_edge_still_gives_nothing(side: float):
    """穿透深到``ρ``只剩半径的一半，只要轴向越出，四样输出仍全零。

    分出这一条，是因为上一条的穿透只有1mm——一个"轴向判据只在浅穿透时生效"
    的实现（比如把两条判据写成``or``而不是``and``再靠浅穿透侥幸）能从那里走过去。
    **深穿透是这条判据的另一个分支，不是同一条的加强版。**
    """

    term = _term()
    outside = _state((0.5 * RADIUS_MM, 0.0, side * (HALF_WIDTH_MM + 5.0)))
    assert term.energy(outside, CONTEXT) == 0.0
    assert term.gradient(outside, CONTEXT) == (0.0, 0.0, 0.0)
    assert term.hessian_entries(outside, CONTEXT) == ()
    assert term.normal_force_n(outside) == (0.0,)


def test_the_axial_edge_is_inclusive():
    """``|s| = half_width``**算在里面**。

    边界闭开是一条声明不是一个实现细节：闭区间意味着端沿上的节点仍受力，
    这与`coulomb_return_map`的摩擦锥取闭边界同一口径。
    """

    term = _term()
    state = _state((49.0, 0.0, HALF_WIDTH_MM))
    assert term.normal_force_n(state)[0] == pytest.approx(
        STIFFNESS_N_PER_MM * (RADIUS_MM - 49.0)
    )


def test_axial_clearance_reports_the_distance_to_the_edge_with_sign():
    """越出的一侧必须是**负**的——门要看得见它已经越过去了。"""

    term = _term()
    assert term.axial_clearance_mm(_state((49.0, 0.0, 0.0)))[0] == pytest.approx(HALF_WIDTH_MM)
    assert term.axial_clearance_mm(_state((49.0, 0.0, 6.5)))[0] == pytest.approx(2.0)
    assert term.axial_clearance_mm(_state((49.0, 0.0, -6.5)))[0] == pytest.approx(2.0)
    assert term.axial_clearance_mm(_state((49.0, 0.0, 10.5)))[0] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# 模型性质：力精确、穿透是O(1/k)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stiffness", [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
def test_the_normal_force_is_exact_and_the_penetration_is_one_over_k(stiffness: float):
    """节点被重力压在圆柱**顶上**，平衡时``N = m·g``与``k``无关。

    轴沿``x``、节点停在``(0, 0, +R)``：那里外法向是``+z``、与重力共线，
    于是法向平衡直接给出``N = m·g``，**判据里没有别的分量掺进来**。
    这一条要验的就是"罚接触的力精确、位置差``N/k``"这条性质本身，
    所以刻意选最干净的构型；法向不与重力共线的情形由倾斜轴那条门管。

    **注意起点必须在圆柱外侧一侧**：``ρ``是到轴的距离，节点从``R``往下沉
    ``ρ``才变小、``g``才变负。起点写到``−R``去是把"外侧"认成了"下方"，
    那时节点越掉越远、间隙恒为正，接触一次都不活动。

    起点取**半穿透**而不是``z = R``：``g = 0``恰好落在活动集边界外
    （判据是``g < 0``），那里切线刚度为零、牛顿迈不出第一步——
    这与`test_contact_normal.py`记的"分离态切线刚度奇异"是同一条模型性质。
    """

    term = PenaltyCylinderContact(
        cylinders=(
            (0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), RADIUS_MM, HALF_WIDTH_MM, stiffness, 0.0),
        )
    )
    registry = EnergyRegistry(terms=(UniformGravity(), term))
    result = solve_equilibrium(
        registry,
        CONTEXT,
        _layout(),
        (0.0, 0.0, RADIUS_MM - 0.5 * THEORY_NORMAL_N / stiffness),
        fixed_indices=frozenset({0, 1}),
        #: 容差按残差地板定，不写死1e-13——理由与实测见
        #: `test_the_residual_floor_is_the_stiffness_times_one_ulp_of_the_radius`。
        residual_tol_n=stiffness * math.ulp(RADIUS_MM),
    )
    assert result.converged, result.reason
    assert result.iterations == 1, (
        "只放开z、轴沿x时``g = z − R``是线性的，能量严格二次，牛顿该一步到位；"
        f"实测{result.iterations}步说明间隙不是按到轴的距离算的"
    )

    #: **判绝对不判相对，而且绝对界就是残差地板。**
    #: 半空间那道门写着"跨六个数量级，法向力一个ulp都不动"——**那条在圆柱上不成立**，
    #: 因为间隙``g = ρ − R``是两个``O(R)``量相减（灾难性相消），
    #: 而半空间的``g = z − 0``没有这一步。力的可达精度因此是``k·ulp(R)``的绝对量，
    #: 不是相对量。抄那条相对判据过来，k=1e5起就会红——**实测正是从1e5开始红的**。
    (force,) = term.normal_force_n(result.state)
    assert force == pytest.approx(THEORY_NORMAL_N, abs=stiffness * math.ulp(RADIUS_MM))

    (distance,) = term.radial_distance_mm(result.state)
    penetration = RADIUS_MM - distance
    assert penetration == pytest.approx(
        THEORY_NORMAL_N / stiffness, abs=math.ulp(RADIUS_MM)
    )


@pytest.mark.parametrize("radius", [50.0, 6.25])
@pytest.mark.parametrize("stiffness", [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
def test_the_residual_floor_is_the_stiffness_times_one_ulp_of_the_radius(
    radius: float, stiffness: float
):
    """**罚接触在半径``R``的圆柱上，残差降不到``0.5·k·ulp(R)``以下。**

    机理：解在``ρ ≈ R``处，而``z``的表示误差最多半个ulp，残差是``k``乘这个误差。
    半空间那道门察觉不到这条——它的解在``z ≈ 0``附近，``ulp(0)``小到无关紧要。
    **圆柱把坐标原点推到了半径上，于是同一个刚度的可达残差差了十几个数量级。**

    2026-08-17实测两个半径×五档刚度共10组，比值全在``[0.15, 0.48]``，
    **无一超过0.5**。这条不是"精度不够"的托词，是一条可预测的界：
    绕线机导轮``R = 50mm``、罚刚度``1e4``时地板是``3.6e-11 N``，
    而链路上要分辨的张力是10—30N——**够用，但求解器容差必须按它定，
    不能照抄别处的绝对数**。

    **它也是一条设计约束**：想把残差压到``1e-13``就必须把``k``压到``30 N/mm``
    以下，而那时穿透是``N/k``约0.65mm——**精度与穿透在这里是同一个旋钮的两头**。
    """

    term = PenaltyCylinderContact(
        cylinders=((0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), radius, HALF_WIDTH_MM, stiffness, 0.0),)
    )
    registry = EnergyRegistry(terms=(UniformGravity(), term))
    result = solve_equilibrium(
        registry,
        CONTEXT,
        _layout(),
        (0.0, 0.0, radius - 0.5 * THEORY_NORMAL_N / stiffness),
        fixed_indices=frozenset({0, 1}),
        #: 要一个够不到的容差才量得出地板——收敛了就停在容差上而不是地板上。
        residual_tol_n=1.0e-18,
        max_iterations=60,
    )
    assert not result.converged, "容差取1e-18还能收敛，说明这条地板不存在——判据要重写"
    floor = stiffness * math.ulp(radius)
    assert result.residual_n <= 0.5 * floor, (
        f"R={radius} k={stiffness:g}：残差{result.residual_n:.4e}超过半个"
        f"k·ulp(R)={0.5 * floor:.4e}，与2026-08-17的10组实测不符"
    )


# ---------------------------------------------------------------------------
# 倾斜轴：一个把轴硬编码成z的实现要在这里死
# ---------------------------------------------------------------------------


def test_a_tilted_axis_keeps_every_invariant():
    """轴取``(1,1,1)/√3``，四条不变量逐条复验。

    本文件其余构型的轴是``+z``或``+x``，都是坐标轴。**坐标轴上，
    "投影掉轴分量"与"把某一个分量置零"是同一件事**，于是一个偷懒的实现
    分不出来。这条门就是那个分辨器。
    """

    axis = tuple(1.0 / math.sqrt(3.0) for _ in range(3))
    term = PenaltyCylinderContact(
        cylinders=(
            (0, (0.0, 0.0, 0.0), axis, RADIUS_MM, HALF_WIDTH_MM, STIFFNESS_N_PER_MM, 0.0),
        )
    )
    position = (40.0, -12.0, 5.0)
    state = _state(position)

    (normal,) = term.outward_normal(state)
    dot = sum(normal[index] * axis[index] for index in range(3))
    magnitude = math.sqrt(sum(component * component for component in position))
    assert abs(dot) <= 8.0 * math.ulp(1.0) * magnitude

    hessian = term.hessian(state, CONTEXT)
    for row in range(3):
        along = sum(hessian[row][column] * axis[column] for column in range(3))
        assert abs(along) <= 1.0e-9 * STIFFNESS_N_PER_MM

    step = 1.0e-6
    for column in range(3):
        forward = list(position)
        backward = list(position)
        forward[column] += step
        backward[column] -= step
        high = term.gradient(_state(tuple(forward)), CONTEXT)
        low = term.gradient(_state(tuple(backward)), CONTEXT)
        for row in range(3):
            numeric = (high[row] - low[row]) / (2.0 * step)
            assert numeric == pytest.approx(hessian[row][column], rel=2e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# 协议契约
# ---------------------------------------------------------------------------


def test_the_fused_path_matches_the_separate_calls_bit_for_bit():
    """spec/12第3.1节：融合路径与单独调**逐字节**相同，不是"约等于"。"""

    term = _term()
    state = _state((31.0, 29.0, 1.5))
    energy, gradient, hessian = term.quantities(
        state, CONTEXT, need_gradient=True, need_hessian=True
    )
    assert energy == term.energy(state, CONTEXT)
    assert gradient == term.gradient(state, CONTEXT)
    assert hessian == term.hessian(state, CONTEXT)


def test_the_energy_is_the_hand_computed_half_k_g_squared():
    term = _term()
    state = _state((49.0, 0.0, 0.0))
    gap = 49.0 - RADIUS_MM
    assert term.energy(state, CONTEXT) == pytest.approx(
        0.5 * STIFFNESS_N_PER_MM * gap * gap, rel=1e-15
    )


def test_the_gradient_pushes_the_node_out_not_in():
    """势能的负梯度是力：穿透时力必须**向外**（沿``+n``）。"""

    term = _term()
    state = _state((49.0, 0.0, 0.0))
    gradient = term.gradient(state, CONTEXT)
    force = tuple(-component for component in gradient)
    assert force[0] > 0.0
    assert force[1] == 0.0
    assert force[2] == 0.0


def test_the_node_radius_shifts_the_contact_surface_outward():
    """节点半径是显式参数：给``r``等于把接触面抬到``R + r``。"""

    plain = _term(node_radius=0.0)
    fat = _term(node_radius=2.0)
    state = _state((51.0, 0.0, 0.0))
    assert plain.normal_force_n(state) == (0.0,)
    assert fat.normal_force_n(state)[0] == pytest.approx(STIFFNESS_N_PER_MM * 1.0)


def test_node_index_bound_covers_every_declared_cylinder():
    term = PenaltyCylinderContact(
        cylinders=(
            (0, AXIS_POINT, AXIS_Z, RADIUS_MM, HALF_WIDTH_MM, STIFFNESS_N_PER_MM, 0.0),
            (4, AXIS_POINT, AXIS_Z, RADIUS_MM, HALF_WIDTH_MM, STIFFNESS_N_PER_MM, 0.0),
        )
    )
    assert term.node_index_bound() == 5


# ---------------------------------------------------------------------------
# 失败关闭
# ---------------------------------------------------------------------------


def test_a_node_on_the_axis_fails_closed():
    """``ρ = 0``法向没有定义——**抛，不猜方向**。"""

    term = _term()
    state = _state((0.0, 0.0, 0.0))
    with pytest.raises(ContactError, match="cylinder axis"):
        term.energy(state, CONTEXT)


def test_a_non_unit_axis_fails_closed():
    with pytest.raises(ContactError, match="unit vector"):
        PenaltyCylinderContact(
            cylinders=(
                (0, AXIS_POINT, (0.0, 0.0, 2.0), RADIUS_MM, HALF_WIDTH_MM, 1.0, 0.0),
            )
        )


@pytest.mark.parametrize(
    ("cylinder", "message"),
    [
        ((0, AXIS_POINT, AXIS_Z, 0.0, HALF_WIDTH_MM, 1.0, 0.0), "radius must be positive"),
        ((0, AXIS_POINT, AXIS_Z, RADIUS_MM, 0.0, 1.0, 0.0), "half width must be positive"),
        ((0, AXIS_POINT, AXIS_Z, RADIUS_MM, HALF_WIDTH_MM, 0.0, 0.0), "stiffness must be"),
        ((0, AXIS_POINT, AXIS_Z, RADIUS_MM, HALF_WIDTH_MM, 1.0, -1.0), "nonnegative"),
        ((-1, AXIS_POINT, AXIS_Z, RADIUS_MM, HALF_WIDTH_MM, 1.0, 0.0), "nonnegative int"),
        ((True, AXIS_POINT, AXIS_Z, RADIUS_MM, HALF_WIDTH_MM, 1.0, 0.0), "nonnegative int"),
    ],
)
def test_malformed_declarations_fail_closed(cylinder, message):
    with pytest.raises(ContactError, match=message):
        PenaltyCylinderContact(cylinders=(cylinder,))


def test_no_cylinders_fails_closed():
    with pytest.raises(ContactError, match="at least one cylinder"):
        PenaltyCylinderContact()
