"""`PenaltyAnnulusLimit`的**扭转接线**（决策0088丁2，能力位S6.6）。

## 它兑现的是台账上的哪一句

S6.6的``missing``原话：「**边缘点仍假定材料标架不绕切线转**——扭转自由度本身
已由`rod`落地（0065），槽壁挡扭转也已接上（0072），但**这条接线没有走到
边缘点的位置生成上**，有扭转时边缘位置仍错``(w/2)·sin(扭角)``」。

## 那句``(w/2)·sin(扭角)``落在哪一个坐标上——本文件把它算清楚了

轴向``a``、切向``t``、无扭时带宽方向``d2 = a``。绕切线转``γ``之后
``m̂2 = −sin γ·d1 + cos γ·d2``，于是边缘点``q = x + e·m̂2``相对
"不转"那个假设的误差有三个可读的分量：

| 分量 | 闭式 | γ=0.7、e=2 mm时 |
|---|---|---|
| **轴向**（进限位判据的那一个） | ``e·(1 − cos γ)`` | 0.470316 mm |
| **径向**（进环带活动条件的那一个） | ``e·sin γ`` | **1.288435 mm ← 就是``(w/2)·sin γ``** |
| 总位移 | ``2e·|sin(γ/2)|`` | 1.371591 mm |

**台账那句话说的是径向那一个。** 它不进间隙、进的是活动条件
``ρ_e ∈ [inner, outer]``——也就是说扭转在这里**不是把力算错一点，
而是可能把一片法兰整个开掉或关上**。本文件有一条门专门判这件事。

## 分辨力：``γ ≡ 0``且``d2 = a``时**逐位相同**

判的是`float.hex()`，不是`pytest.approx`（0068吃过``-0.0 == 0.0``为真
而`canonical_bytes`不同那一课）。逐位成立的算术理由写在
`test_a_zero_twist_declaration_is_bit_for_bit_the_untwisted_path`里。

**另有一档更彻底的退化不在本文件**：不声明``edge_twists``时走的是
原来那串代码一个字节没动，400组随机构型 × 七个面的`float.hex()`对拍
记在决策0088第三节。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import PenaltyAnnulusLimit
from physics_engine.contact.errors import ContactError
from physics_engine.energies import EnergyContext
from physics_engine.rod import PenaltyGrooveWall, RodMaterialFrame, build_rod_layout
from physics_engine.state import State

#: 三节点杆：2条边、1个内顶点。位置块9个 ＋ 边扭角2个 ＝ 11个自由度。
NODE_COUNT = 3
ROD = build_rod_layout(layout_id="layout/annulus-twist", node_count=NODE_COUNT)
LAYOUT = ROD.layout
GAMMA_LEFT = ROD.twist_index(0)
GAMMA_RIGHT = ROD.twist_index(1)

CONTEXT = EnergyContext(
    context_id="context/annulus-twist",
    node_masses_kg=(1.0,) * NODE_COUNT,
    gravity_mm_s2=(0.0, 0.0, 0.0),
)

#: 法兰：轴沿``z``、心在原点、环带``ρ ∈ [0, 200]``。
AXIS = (0.0, 0.0, 1.0)
AXIS_POINT = (0.0, 0.0, 0.0)
INNER_MM = 0.0
OUTER_MM = 200.0
STIFFNESS = 1.0e4
#: 4 mm带材的上边缘。
HALF_WIDTH_MM = 2.0
#: 带材在半径50处、切向沿``y``。材料帧取``d2 = a``（无扭时带宽平行于轴），
#: ``d1``由`RodMaterialFrame`那条``d2 = t × d1``反解出``(−1, 0, 0)``。
TANGENT = (0.0, 1.0, 0.0)
D1 = (-1.0, 0.0, 0.0)
D2 = (0.0, 0.0, 1.0)
RADIUS_MM = 50.0

FRAME = RodMaterialFrame(
    tangents=(TANGENT, TANGENT),
    d1=(D1, D1),
    d2=(D2, D2),
    reference_twist=(0.0,),
)
TWIST = (GAMMA_LEFT, GAMMA_RIGHT, D1, D2, D1, D2)


def _state(gamma_left: float, gamma_right: float, node_z: float = 0.0) -> State:
    """三个节点排在``y``上，中间那个（内顶点）在``(50, 0, node_z)``。"""

    vector = (
        RADIUS_MM, -10.0, node_z,
        RADIUS_MM, 0.0, node_z,
        RADIUS_MM, 10.0, node_z,
        gamma_left, gamma_right,
    )
    return State(layout=LAYOUT, vector=vector)


def _face(limit_mm: float, inward: float, offset_mm: float = HALF_WIDTH_MM):
    return (1, AXIS_POINT, AXIS, INNER_MM, OUTER_MM, limit_mm, inward,
            offset_mm, STIFFNESS)


def _twisted(limit_mm: float, inward: float, offset_mm: float = HALF_WIDTH_MM):
    return PenaltyAnnulusLimit(
        faces=(_face(limit_mm, inward, offset_mm),), edge_twists=(TWIST,)
    )


def _plain(limit_mm: float, inward: float, offset_mm: float = HALF_WIDTH_MM):
    return PenaltyAnnulusLimit(faces=(_face(limit_mm, inward, offset_mm),))


# ------------------------------------------------------- 一、逐位退化 -------


def test_a_zero_twist_declaration_is_bit_for_bit_the_untwisted_path() -> None:
    """**本片的分辨力**：``γ ≡ 0``且``d2 = a``时，两条路给出逐位相同的一切。

    ## 为什么它在IEEE-754下是精确的，不是"碰巧很接近"

    ``sin 0 = 0.0``、``cos 0 = 1.0``**都是精确的**，于是平分线的分子

        −0.0·d1 + 1.0·d2 − 0.0·d1 + 1.0·d2

    逐分量恰是``2·d2``（``±0.0``加到一个有限数上不改变它），长度恰是``2.0``，
    相除恰是``d2 = a``。再往下``q = x + e·a``、``s_e = (q − p)·a``——
    轴对齐时那个点积的三项里两项恰是``0.0``，与旧路径``(x − p)·a + e``
    走的是同一串加法。

    **这条只在轴对齐的构型上是逐位的。** 一般轴向下两条路的求和次序不同
    （旧路径用`sum()`＝补偿求和，新路径用`ad_dot`＝顺序累加），差在1 ulp量级。
    本仓已经因为这两种求和不是同一个算法红过一次门（`autodiff.ad_dot`那条
    跨机实测），**所以这句限定必须写出来，不能让人以为哪里都逐位**。
    """

    state = _state(0.0, 0.0)
    for limit, inward in ((2.5, -1.0), (1.5, 1.0), (-3.0, -1.0)):
        plain, twisted = _plain(limit, inward), _twisted(limit, inward)
        assert plain.energy(state, CONTEXT).hex() == twisted.energy(
            state, CONTEXT
        ).hex(), f"limit={limit} inward={inward}：能量不逐位相同"
        assert [v.hex() for v in plain.gradient(state, CONTEXT)] == [
            v.hex() for v in twisted.gradient(state, CONTEXT)
        ], f"limit={limit}：梯度不逐位相同"
        for name in ("rub_force_n", "edge_clearance_mm", "radial_distance_mm"):
            assert [v.hex() for v in getattr(plain, name)(state)] == [
                v.hex() for v in getattr(twisted, name)(state)
            ], f"limit={limit}：{name}不逐位相同"
        #: 带宽方向本身也退回轴——**它是上面这些成立的病根所在**。
        assert [v.hex() for v in twisted.edge_width_direction(state)[0]] == [
            v.hex() for v in AXIS
        ]


def test_the_zero_twist_hessian_keeps_the_position_block_and_not_the_gamma_block() -> None:
    """``γ ≡ 0``时Hessian的位置块仍是``k·(a ⊗ a)``，**而γ那两行两列不是零**。

    这一条是上一条的补面，也是一条**反证**：如果扭转路径的Hessian在γ=0处
    整块塌成零，那说明γ的二阶项被漏掉了——``∂g/∂γ``在γ=0处为零，
    但``∂²g/∂γ²``不为零，而``½kg²``的二阶导里有``k·g·∂²g/∂γ²``这一项。
    **"退化"退的是位置块，不是整块。**
    """

    state = _state(0.0, 0.0)
    plain, twisted = _plain(2.5, -1.0), _twisted(2.5, -1.0)
    assert max(plain.rub_force_n(state)) > 0.0, "构型前提：这一片必须是活动的"
    dense_plain = {(r, c): v for r, c, v in plain.hessian_entries(state, CONTEXT)}
    dense_twist: dict[tuple[int, int], float] = {}
    for row, column, value in twisted.hessian_entries(state, CONTEXT):
        dense_twist[(row, column)] = dense_twist.get((row, column), 0.0) + value
    for axis_a in range(3):
        for axis_b in range(3):
            key = (3 + axis_a, 3 + axis_b)
            assert dense_twist[key].hex() == dense_plain[key].hex(), (
                f"位置块{key}在γ=0处不逐位退化"
            )
    #: γ那一块非零，且两条边**对称**（同一条平分线各承一半）。
    gamma_block = dense_twist[(GAMMA_LEFT, GAMMA_LEFT)]
    assert gamma_block != 0.0, "γ的二阶项整块为零——那说明扭转根本没进Hessian"
    assert dense_twist[(GAMMA_LEFT, GAMMA_RIGHT)] == pytest.approx(
        gamma_block, rel=1e-14
    )
    assert dense_twist[(GAMMA_RIGHT, GAMMA_RIGHT)] == pytest.approx(
        gamma_block, rel=1e-14
    )


# --------------------------------------------- 二、解析量：错的正是那一句 ---


@pytest.mark.parametrize("gamma", (0.1, 0.35, 0.7, 1.0))
def test_the_edge_position_error_is_the_ledger_quantity(gamma: float) -> None:
    """台账那句``(w/2)·sin(扭角)``**落在径向**，轴向那一份是``(w/2)·(1 − cos γ)``。

    三个分量各自与闭式对拍。**它们不是同一个数**——只报"误差是``(w/2)·sin γ``"
    而把轴向那一份也当成它，会让人以为间隙差了1.288 mm（γ=0.7时），
    实际间隙只差0.470 mm。
    """

    state = _state(gamma, gamma)
    twisted = _twisted(2.5, -1.0)
    width = twisted.edge_width_direction(state)[0]
    #: ``m̂2 = −sin γ·(−1,0,0) + cos γ·(0,0,1) = (sin γ, 0, cos γ)``。
    assert width[0] == pytest.approx(math.sin(gamma), abs=1e-15)
    assert width[1] == pytest.approx(0.0, abs=1e-15)
    assert width[2] == pytest.approx(math.cos(gamma), abs=1e-15)

    axial_error = HALF_WIDTH_MM * (1.0 - math.cos(gamma))
    radial_error = HALF_WIDTH_MM * math.sin(gamma)
    total_error = 2.0 * HALF_WIDTH_MM * abs(math.sin(0.5 * gamma))

    plain = _plain(2.5, -1.0)
    #: 轴向：间隙差。``inward = −1`` ⟹ ``g = s_e − limit``，而``s_e = e·cos γ``。
    gap_twist = twisted.edge_clearance_mm(state)[0]
    gap_plain = plain.edge_clearance_mm(state)[0]
    assert gap_plain - gap_twist == pytest.approx(axial_error, rel=1e-13)
    #: 径向：**台账那一句**。
    rho_twist = twisted.radial_distance_mm(state)[0]
    rho_plain = plain.radial_distance_mm(state)[0]
    assert rho_twist - rho_plain == pytest.approx(radial_error, rel=1e-13)
    #: 总位移：``|e|·|m̂2 − a| = 2e·|sin(γ/2)|``。
    measured_total = HALF_WIDTH_MM * math.sqrt(
        sum((width[a] - AXIS[a]) ** 2 for a in range(3))
    )
    assert measured_total == pytest.approx(total_error, rel=1e-13)


def test_the_bisector_angle_is_exactly_the_average_of_the_two_edge_twists() -> None:
    """两条边的帧相同（直杆）时，平分线的倾角**精确**等于``(γ_l + γ_r)/2``。

    0072第3.3节对`PenaltyGrooveWall`记的正是这一条（"不是近似"）。
    本项的``m̂2``与那里是同一个表达式，所以这条闭式在这里必须同样成立——
    **它是"归一化不可省"的可观测形式**：不归一化时倾角仍对，但长度会掉到
    ``cos(Δγ/2)``，于是半宽悄悄缩水。
    """

    for left, right in ((0.2, 0.8), (-0.3, 0.9), (1.1, 1.1)):
        state = _state(left, right)
        width = _twisted(2.5, -1.0).edge_width_direction(state)[0]
        assert math.atan2(width[0], width[2]) == pytest.approx(
            0.5 * (left + right), abs=1e-15
        )
        assert math.sqrt(sum(v * v for v in width)) == pytest.approx(1.0, abs=1e-15), (
            "平分线没有归一化——半宽会随扭角悄悄缩水"
        )


def test_twist_can_switch_a_flange_off_through_the_annulus_band() -> None:
    """**扭转不只是把力算错一点，它能把一片法兰整个关掉。**

    径向那一份误差``e·sin γ``进的是活动条件``ρ_e ∈ [inner, outer]``。
    把环带外径收到边缘点扭出去就够不着的位置：无扭时``ρ_e = 50.0``在带内、
    有扭时``ρ_e = 50 + 2·sin γ``跑到带外，**蹭边力从有到无**。

    这条是本片"接线接对了没有"最不像数值的一个判据：它判的是**分支**。
    """

    outer = RADIUS_MM + HALF_WIDTH_MM * math.sin(0.5)
    narrow = PenaltyAnnulusLimit(
        faces=((1, AXIS_POINT, AXIS, INNER_MM, outer, 2.5, -1.0,
                HALF_WIDTH_MM, STIFFNESS),),
        edge_twists=(TWIST,),
    )
    assert narrow.rub_force_n(_state(0.0, 0.0))[0] > 0.0, "无扭时该顶上"
    assert narrow.rub_force_n(_state(0.4, 0.4))[0] > 0.0, "扭0.4 rad仍在环带内"
    assert narrow.rub_force_n(_state(0.6, 0.6))[0] == 0.0, (
        "扭0.6 rad时边缘点已经跑出环带外径，这一片法兰在那里不存在"
    )
    #: 反证：同一个扭角、不收外径时它**仍然**顶着——上一行判的是环带不是扭角。
    assert _twisted(2.5, -1.0).rub_force_n(_state(0.6, 0.6))[0] > 0.0


def test_the_complementarity_is_zero_tolerance_with_twist_too() -> None:
    """``g > 0 ⟹ f ≡ 0``在扭转路径上仍是**零容差**，梯度也整条为零。"""

    twisted = _twisted(-3.0, -1.0)
    state = _state(0.6, 0.4)
    assert twisted.edge_clearance_mm(state)[0] > 0.0
    assert twisted.rub_force_n(state) == (0.0,)
    assert twisted.energy(state, CONTEXT) == 0.0
    assert all(value == 0.0 for value in twisted.gradient(state, CONTEXT))
    assert twisted.hessian_entries(state, CONTEXT) == ()


# ------------------------------------------- 三、梯度与Hessian是导数 --------


def _perturbed(state: State, index: int, delta: float) -> State:
    vector = list(state.vector)
    vector[index] += delta
    return State(layout=state.layout, vector=tuple(vector))


def test_the_gradient_along_gamma_is_second_order_convergent() -> None:
    """**γ方向的中心差分二阶收敛，实测比落在4.0上**——本片的技术前提。

    ``g``经``m̂2(γ)``里的两次三角函数与一次归一化非线性地依赖γ，
    于是截断项是真的。**位置方向不同**：``g``对``x``是线性的、``½kg²``是二次的，
    中心差分**恒精确**，那一档由下一条门判（形态与
    `test_contact_groove_sweep.py`记的冻结帧那一族相同）。
    """

    twisted = _twisted(2.5, -1.0)
    state = _state(0.35, 0.20)
    assert max(twisted.rub_force_n(state)) > 0.0, "构型前提：必须是活动的"
    analytic = twisted.gradient(state, CONTEXT)
    errors = []
    for step in (0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625):
        worst = 0.0
        for index in (GAMMA_LEFT, GAMMA_RIGHT):
            ahead = twisted.energy(_perturbed(state, index, step), CONTEXT)
            behind = twisted.energy(_perturbed(state, index, -step), CONTEXT)
            worst = max(worst, abs((ahead - behind) / (2.0 * step) - analytic[index]))
        errors.append(worst)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    #: 2026-08-18实测比3.9999／4.0000 × 4，误差2.353e-1 → 2.298e-4。
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    #: 先断"误差确实降到一个小数"，否则上面那条比值可以由几个大数凑出来。
    assert errors[-1] < 5.0e-4, errors


def test_the_gradient_along_position_is_exact_because_the_energy_is_quadratic() -> None:
    """位置方向的中心差分**没有截断误差**——``½kg²``是``x``的二次多项式。

    误差因此不随``h``下降。**这条形态本身是一条分辨力**：一个把``m̂2``
    错写成依赖``x``的实现在这里会突然长出一条4.0的收敛比。
    """

    twisted = _twisted(2.5, -1.0)
    state = _state(0.35, 0.20)
    analytic = twisted.gradient(state, CONTEXT)
    for step in (0.1, 0.01, 0.001):
        for index in (3, 4, 5):
            ahead = twisted.energy(_perturbed(state, index, step), CONTEXT)
            behind = twisted.energy(_perturbed(state, index, -step), CONTEXT)
            measured = (ahead - behind) / (2.0 * step)
            assert measured == pytest.approx(analytic[index], abs=1.0e-8), (
                f"位置{index}、步长{step}：二次型上的中心差分应当没有截断误差"
            )


def test_the_hessian_is_the_derivative_of_the_gradient() -> None:
    """Hessian经中心差分独立验过，**位置与γ都被扰动**，二阶收敛比落在4.0上。"""

    twisted = _twisted(2.5, -1.0)
    state = _state(0.35, 0.20)
    size = len(state.vector)
    dense = [[0.0] * size for _ in range(size)]
    for row, column, value in twisted.hessian_entries(state, CONTEXT):
        dense[row][column] += value
    probes = (3, 4, 5, GAMMA_LEFT, GAMMA_RIGHT)
    errors = []
    for step in (0.02, 0.01, 0.005, 0.0025):
        worst = 0.0
        for index in probes:
            ahead = twisted.gradient(_perturbed(state, index, step), CONTEXT)
            behind = twisted.gradient(_perturbed(state, index, -step), CONTEXT)
            for probe in probes:
                measured = (ahead[probe] - behind[probe]) / (2.0 * step)
                worst = max(worst, abs(measured - dense[probe][index]))
        errors.append(worst)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    #: 2026-08-18实测比3.9999／4.0000／4.0000，误差3.678e-1 → 5.748e-3。
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    assert errors[-1] < 1.0e-2, errors


def test_the_fused_path_and_the_plain_energy_agree_bit_for_bit() -> None:
    """spec/12第3.1节：融合路径的能量值与单独调`energy`**逐字节相同**。

    扭转路径上它是**按构造**成立的而不是量出来的：三个导数阶走的是同一串
    `ad_*`运算（顺序累加，不是`sum()`——见`autodiff.ad_dot`那条跨机实测）。
    """

    twisted = _twisted(2.5, -1.0)
    for left, right in ((0.0, 0.0), (0.35, 0.20), (-0.5, 0.9)):
        state = _state(left, right)
        plain = twisted.energy(state, CONTEXT)
        for need_gradient in (False, True):
            for need_hessian in (False, True):
                fused = twisted.quantities(
                    state, CONTEXT,
                    need_gradient=need_gradient, need_hessian=need_hessian,
                )[0]
                assert fused.hex() == plain.hex(), (left, right)


def test_the_width_direction_matches_the_rod_groove_wall_bit_for_bit() -> None:
    """**与`rod.PenaltyGrooveWall`的``m̂2``逐位对拍**——重复实现的漂移由门守。

    0072第3.2节裁定不开`contact → rod`这条import边，于是本项的平分线是
    **第二份实现**。两份实现之间不靠约定、靠这一条：同一个状态上
    `float.hex()`必须相同。它红了说明两份实现已经分家。
    """

    wall = PenaltyGrooveWall(
        layout=ROD,
        frame=FRAME,
        faces=((0, HALF_WIDTH_MM, (0.0, 0.0, 9.0), (0.0, 0.0, -1.0), STIFFNESS),),
    )
    twisted = _twisted(2.5, -1.0)
    for left, right in ((0.0, 0.0), (0.35, 0.20), (-0.5, 0.9), (1.3, -1.3)):
        state = _state(left, right)
        assert [v.hex() for v in wall.width_direction(state)[0]] == [
            v.hex() for v in twisted.edge_width_direction(state)[0]
        ], f"γ=({left}, {right})：两份平分线实现已经分家"


# --------------------------------------------------------- 四、必须红 -------


def test_a_short_edge_twists_tuple_fails_closed() -> None:
    """短一项等于让某一片法兰**静默**退回无扭转。"""

    with pytest.raises(ContactError, match="as long as faces"):
        PenaltyAnnulusLimit(
            faces=(_face(2.5, -1.0), _face(-2.5, 1.0)), edge_twists=(TWIST,)
        )


def test_a_gamma_index_colliding_with_the_position_block_fails_closed() -> None:
    """γ下标撞上节点自己的位置块 ⟹ 同一个自由度在5×5模板里出现两次。"""

    with pytest.raises(ContactError, match="撞上了节点"):
        PenaltyAnnulusLimit(
            faces=(_face(2.5, -1.0),),
            edge_twists=((4, GAMMA_RIGHT, D1, D2, D1, D2),),
        )


def test_two_identical_gamma_indices_fail_closed() -> None:
    """一个内顶点夹在**两条**边之间；两条边共用一个扭角不是杆的自由度划分。"""

    with pytest.raises(ContactError, match="两个γ下标相同"):
        PenaltyAnnulusLimit(
            faces=(_face(2.5, -1.0),),
            edge_twists=((GAMMA_LEFT, GAMMA_LEFT, D1, D2, D1, D2),),
        )


def test_a_non_unit_material_axis_fails_closed() -> None:
    """不归一化等于让半宽悄悄乘上``|d|``，而调用方以为自己给的是``offset``。"""

    with pytest.raises(ContactError, match="不是单位矢量"):
        PenaltyAnnulusLimit(
            faces=(_face(2.5, -1.0),),
            edge_twists=((GAMMA_LEFT, GAMMA_RIGHT, D1, (0.0, 0.0, 1.5), D1, D2),),
        )


def test_a_non_orthogonal_material_frame_fails_closed() -> None:
    """``m2 = −sin γ·d1 + cos γ·d2``只在``d1 ⊥ d2``时才是单位矢量。"""

    skew = (-math.sqrt(0.5), 0.0, math.sqrt(0.5))
    with pytest.raises(ContactError, match="不正交"):
        PenaltyAnnulusLimit(
            faces=(_face(2.5, -1.0),),
            edge_twists=((GAMMA_LEFT, GAMMA_RIGHT, skew, D2, D1, D2),),
        )


def test_a_zero_offset_with_a_twist_declaration_fails_closed() -> None:
    """偏移为零 ⟹ 边缘点退回中心线，**对γ的依赖当场消失**而力还在。

    与`rod.PenaltyGrooveWall`的``offset = 0``拒收同一条理由。
    **无扭转那一路仍然允许``offset = 0``**（那里它就是中心线接触，没有静默），
    下面第二行判的正是这个不对称是有意的。
    """

    with pytest.raises(ContactError, match="边缘偏移是零"):
        PenaltyAnnulusLimit(faces=(_face(2.5, -1.0, 0.0),), edge_twists=(TWIST,))
    assert PenaltyAnnulusLimit(faces=(_face(2.5, -1.0, 0.0),)).faces[0][7] == 0.0


def test_a_wrong_sized_edge_twist_entry_fails_closed() -> None:
    with pytest.raises(ContactError, match="must be"):
        PenaltyAnnulusLimit(
            faces=(_face(2.5, -1.0),),
            edge_twists=((GAMMA_LEFT, GAMMA_RIGHT, D1, D2, D1),),
        )


def test_antiparallel_material_frames_fail_closed() -> None:
    """两条边的帧几乎反向时平分线由舍入决定——**那不是方向是噪声**。"""

    twisted = PenaltyAnnulusLimit(
        faces=(_face(2.5, -1.0),),
        edge_twists=((GAMMA_LEFT, GAMMA_RIGHT, D1, D2, D1, D2),),
    )
    with pytest.raises(ContactError, match="几乎反向"):
        twisted.edge_clearance_mm(_state(0.0, math.pi))
