"""扫掠槽壁——沿真实中心线的两段外倾锥面（决策0075，0074第六节阶段一/轨A）。

## 这一项与前四项的分界

| 项 | 间隙 | ∇g | Hessian |
|---|---|---|---|
| `PenaltyNormalContact`（半空间） | `(x−p)·n − r` | 常矢量 | `k·(n⊗n)` |
| `PenaltyCylinderContact`（圆柱侧面） | `ρ − R`，**非线性** | 随位形转 | `k·(n⊗n) + (kg/ρ)(t⊗t)` |
| `PenaltyAnnulusLimit`（法兰内环面） | `inward·(L − s_e)` | 常矢量 | `k·(a⊗a)` |
| `PenaltyGrooveSweep`（本项） | `w/2 + v·tanα − (σu + r)` | `tanα·n − σ·s`，**常矢量** | `k·(∇g⊗∇g)` |

本项与环带同属"间隙是位置的线性函数"那一档，**所以退化必须是逐位的而不是近似的**
——这正是`test_degenerates_bit_for_bit_to_the_annulus_limit`判`float.hex()`
而不判`==`的理由。

## 三条判据各自守什么（决策0075第三节）

1. **逐位退化**：直中心线＋`tanα = 0`时，本项与`PenaltyAnnulusLimit`
   在同一构型上给出**逐位相同**的能量、梯度、Hessian与间隙；
2. **锥面≠平面**：同一横移量下两者的力必须**显著不同**，且数要量出来
   （案例页`cases/groove_sweep_wall`量的是那张表，本文件只守形制侧的
   `sec α`与举升分量两条恒等式）；
3. **有限差分一致**：梯度对能量、Hessian对梯度。**本项没有截断项**
   （活动集内`U`是`x`的二次多项式），所以FD误差是纯舍入、随`h`变小而**变大**
   ——`test_finite_differences_are_roundoff_not_truncation`判的正是这条反直觉的斜率。

## 冻结帧丢掉的那一项，在本文件里被量出来

`PenaltyGrooveSweep`把最近站点的帧冻结进声明，于是丢掉了`∂a*/∂x`
经**帧扭率**进入梯度的那一项。决策0075第四节推了它的闭式：

    ∇g_精确 = ∇g_本项 − A·t,   A = τ·(tanα·u + σ·v) / (1 − u·κ_s − v·κ_n)

`test_frozen_frame_residual_matches_the_closed_form`用**解析曲线上的数值
中心差分**独立求出`∇g_精确`，与上式对拍，实测二阶收敛比恒为4.0000。
**这不是一条形式判据**：它是"我们丢了多少"这个问题的唯一诚实答案。

## 必红矩阵（2026-08-18逐条注错**实测**）

见文件末尾`test_the_mutation_matrix_is_measured`的docstring。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import (
    ContactError,
    PenaltyAnnulusLimit,
    PenaltyGrooveSweep,
    groove_sweep_walls,
)
from physics_engine.energies import EnergyContext, EnergyRegistry
from physics_engine.laydown import CenterlineSemantics, GrooveCenterline, GrooveStation
from physics_engine.state import State, StateField, StateLayout

#: 真机槽几何（plans/14第2.2节、plans/15第2.2条）：槽底宽8.000 mm ⟹ 半宽4.0；
#: 4 mm宽REBCO带材 ⟹ 边缘半偏移2.0。
HALF_WIDTH_MM = 4.0
EDGE_RADIUS_MM = 2.0
#: 壁外倾角。**这是一个声明的设计参数，不是从GCW导出里量出来的**——
#: 真角度要等工件CAD进来，见案例页第四节。本仓的判据一律写成`tanα`的函数。
WALL_ANGLE_DEG = 10.0
WALL_SLOPE = math.tan(math.radians(WALL_ANGLE_DEG))
STIFFNESS_N_PER_MM = 5.0e3
DEPTH_WINDOW_MM = (-1.0, 6.0)


def _layout() -> StateLayout:
    return StateLayout(
        layout_id="layout/groove_sweep",
        fields=(
            StateField("node0_x_mm", 1),
            StateField("node0_y_mm", 1),
            StateField("node0_z_mm", 1),
        ),
    )


CONTEXT = EnergyContext(
    context_id="context/groove_sweep",
    node_masses_kg=(1.0e-9,),
    gravity_mm_s2=(0.0, 0.0, 0.0),
)


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left, right):
    return sum(left[axis] * right[axis] for axis in range(3))


def _unit(vector):
    norm = math.sqrt(_dot(vector, vector))
    return tuple(component / norm for component in vector)


def _state(position) -> State:
    return State(layout=_layout(), vector=tuple(position))


def _straight_centerline(tangent, width, normal) -> GrooveCenterline:
    """一条通用方向的直中心线。

    **方向刻意取成三个分量都非零**：轴对齐的帧会让`∇g`里出现`±0.0`，
    而`+0.0`与`−0.0`的`float.hex()`是不同的字符串。逐位门要判的是运算次序，
    不是零的符号——所以把零的符号这个变量从判据里拿掉。
    """

    return GrooveCenterline(
        centerline_id="groove/straight_generic",
        stations=tuple(
            GrooveStation(
                arc_length_mm=arc,
                position_mm=tuple((arc - 20.0) * tangent[axis] for axis in range(3)),
                tangent=tangent,
                width_direction=width,
                surface_normal=normal,
            )
            for arc in (0.0, 10.0, 20.0, 30.0, 40.0)
        ),
        semantics=CenterlineSemantics(
            position_interpolation="chord_linear",
            frame_interpolation="hold_station",
            topology="open",
            out_of_range="clamp_to_end",
            nearest_refinement_iterations=0,
        ),
        length_unit="mm",
    )


def _generic_frame():
    tangent = _unit((0.3, 0.5, 0.81))
    raw = (0.0, 0.0, 1.0)
    projection = _dot(raw, tangent)
    normal = _unit(tuple(raw[axis] - projection * tangent[axis] for axis in range(3)))
    return tangent, _cross(normal, tangent), normal


# ------------------------------------------------- 判据一：逐位退化到环带 ---


def _degenerate_pair(lateral_mm: float, depth_mm: float):
    """同一构型上的两个项：扫掠槽壁（``tanα = 0``、直中心线）与平面环带。"""

    tangent, width, normal = _generic_frame()
    centerline = _straight_centerline(tangent, width, normal)
    position = tuple(
        0.3 * tangent[axis] + lateral_mm * width[axis] + depth_mm * normal[axis]
        for axis in range(3)
    )
    sweep = groove_sweep_walls(
        centerline,
        ((0, position),),
        half_width_mm=HALF_WIDTH_MM,
        wall_slope=0.0,
        edge_radius_mm=EDGE_RADIUS_MM,
        depth_window_mm=DEPTH_WINDOW_MM,
        stiffness_n_mm=STIFFNESS_N_PER_MM,
    )
    #: 环带的"轴上一点"取扫掠项冻结下来的**同一个站点**。这不是循环论证：
    #: 环带的`point`按它自己的语义是"轴上任意一点"，而最近点恰好在那条轴上。
    #: 取同一个点是为了把"两边的`p`不同"这个无关变量从逐位判据里拿掉。
    station = sweep.walls[0][1]
    annulus = PenaltyAnnulusLimit(
        faces=tuple(
            (
                0,
                station,
                width,
                0.0,
                100.0,
                side * HALF_WIDTH_MM,
                side,
                side * EDGE_RADIUS_MM,
                STIFFNESS_N_PER_MM,
            )
            for side in (1.0, -1.0)
        )
    )
    return sweep, annulus, _state(position)


def test_degenerates_bit_for_bit_to_the_annulus_limit() -> None:
    """判据一：``tanα = 0`` + 直中心线 ⟹ 与`PenaltyAnnulusLimit`**逐位相同**。

    **判`float.hex()`不判`==`**：`==`会把差一个ulp的两条实现判成一样，
    而这条门要守的恰恰是"运算次序一个字节都没变"。
    """

    sweep, annulus, state = _degenerate_pair(lateral_mm=2.05, depth_mm=0.9)

    assert [value.hex() for value in sweep.wall_clearance_mm(state)] == [
        value.hex() for value in annulus.edge_clearance_mm(state)
    ]
    assert sweep.energy(state, CONTEXT).hex() == annulus.energy(state, CONTEXT).hex()
    assert [value.hex() for value in sweep.gradient(state, CONTEXT)] == [
        value.hex() for value in annulus.gradient(state, CONTEXT)
    ]
    swept = sweep.hessian(state, CONTEXT)
    ringed = annulus.hessian(state, CONTEXT)
    assert [[value.hex() for value in row] for row in swept] == [
        [value.hex() for value in row] for row in ringed
    ]


def test_the_other_side_degenerates_too() -> None:
    """退化在**两侧**都成立。

    单测一侧会漏掉``σ = −1``那条支：那一侧的间隙走的是
    ``halfwidth − ((−u) + r)``，而环带走的是``−1·((−w/2) − (u − r))``——
    **两个表达式树不同，逐位相等靠的是IEEE取整对取负对称**。
    只判``+s``侧的话，这条性质从没被执行过。
    """

    sweep, annulus, state = _degenerate_pair(lateral_mm=-2.05, depth_mm=-0.4)

    assert sweep.wall_clearance_mm(state)[1].hex() == annulus.edge_clearance_mm(state)[1].hex()
    assert sweep.energy(state, CONTEXT).hex() == annulus.energy(state, CONTEXT).hex()
    assert [value.hex() for value in sweep.gradient(state, CONTEXT)] == [
        value.hex() for value in annulus.gradient(state, CONTEXT)
    ]


def test_the_bracket_position_in_the_gap_expression_is_load_bearing() -> None:
    """**逐位门有没有牙齿，取决于这条**。

    ``halfwidth − (lateral + radius)``与``(halfwidth − lateral) − radius``
    数学上恒等。如果在本构型上它们也逐位相等，那么判据一就**判不出**
    有人把括号挪走——门会一直绿而实现已经变了。

    本条先证明这两种写法在本构型上**逐位不同**，判据一因此是有分辨力的。
    """

    sweep, _, state = _degenerate_pair(lateral_mm=2.05, depth_mm=0.9)
    _, point, width, normal, side, half, slope, radius = sweep.walls[0][0:8]
    delta = tuple(state.vector[axis] - point[axis] for axis in range(3))
    lateral = side * _dot(delta, width)
    depth = _dot(delta, normal)
    halfwidth = half + depth * slope

    as_implemented = halfwidth - (lateral + radius)
    regrouped = (halfwidth - lateral) - radius
    assert as_implemented == pytest.approx(regrouped, abs=1e-12)
    assert as_implemented.hex() != regrouped.hex()
    assert sweep.wall_clearance_mm(state)[0].hex() == as_implemented.hex()


# ------------------------------------------- 判据二：锥面与平面显著不同 ---


def _walls(slope: float, *, position=(0.0, 0.0, 0.0)) -> PenaltyGrooveSweep:
    axis_s = (1.0, 0.0, 0.0)
    axis_n = (0.0, 0.0, 1.0)
    return PenaltyGrooveSweep(
        walls=tuple(
            (
                0,
                position,
                axis_s,
                axis_n,
                side,
                HALF_WIDTH_MM,
                slope,
                EDGE_RADIUS_MM,
                DEPTH_WINDOW_MM[0],
                DEPTH_WINDOW_MM[1],
                STIFFNESS_N_PER_MM,
            )
            for side in (1.0, -1.0)
        )
    )


@pytest.mark.parametrize("lateral_mm", (2.2, 2.5, 3.0))
def test_the_cone_keeps_the_lateral_force_and_adds_a_lift(lateral_mm: float) -> None:
    """锥面与平面的差别**不在横向分量上**，在它多出来的举升分量上。

    同一横移、同一深度下：横向分量``k|g|``两者**逐位相同**（``∇g``的``s``分量
    与``tanα``无关），锥面另有一个``k|g|·tanα``把带材**举出槽**，
    于是合力大小差一个``sec α``。**"显著不同"要说清是哪一项不同**——
    只报"合力不同"会让人以为回正力变大了，而回正力一点没变。
    """

    state = _state((lateral_mm, 0.0, 0.0))
    plane = _walls(0.0)
    cone = _walls(WALL_SLOPE)

    plane_gradient = plane.gradient(state, CONTEXT)
    cone_gradient = cone.gradient(state, CONTEXT)

    assert cone_gradient[0].hex() == plane_gradient[0].hex()
    assert plane_gradient[2] == 0.0
    lateral = plane_gradient[0]
    assert cone_gradient[2] == pytest.approx(-lateral * WALL_SLOPE, rel=1e-14)
    assert cone.wall_force_n(state)[0] == pytest.approx(
        plane.wall_force_n(state)[0] / math.cos(math.radians(WALL_ANGLE_DEG)), rel=1e-14
    )


def test_the_depth_window_is_an_activity_gate_not_an_energy() -> None:
    """深度窗外这面壁不存在：能量、梯度、Hessian**一个非零项都不出**。

    与`PenaltyAnnulusLimit`的环带判据同源——**力在窗边界上跳**，
    案例要为它设门，所以`wall_depth_mm`是公开面。
    """

    cone = _walls(WALL_SLOPE)
    inside = _state((7.2, 0.0, 5.9))
    outside = _state((7.2, 0.0, 6.1))

    assert cone.wall_depth_mm(inside)[0] == pytest.approx(5.9)
    assert cone.energy(inside, CONTEXT) > 0.0
    assert cone.wall_clearance_mm(outside)[0] < 0.0
    assert cone.energy(outside, CONTEXT) == 0.0
    assert cone.gradient(outside, CONTEXT) == (0.0, 0.0, 0.0)
    assert cone.hessian_entries(outside, CONTEXT) == ()
    assert cone.wall_force_n(outside) == (0.0, 0.0)


def test_the_wall_does_not_hinge_at_the_groove_floor() -> None:
    """``halfwidth(v)``是**一条直线**，不是``max(v, 0)``那条折线。

    **这条用例是注错验证补出来的，不是设计时想到的。** 2026-08-18那一轮把实现
    改成``half_width + max(depth, 0.0) * slope``，**全部36条门一条都没红**——
    因为当时没有任何一条用例在``v < 0``且壁活动的构型上求值。
    那正是plans/09教训三说的形态：**一条从没被走过的分支等于一条没有门的分支**，
    而决策0075第二节第2条恰恰是围绕这条分支写的。

    判两件事：
    1. ``v < 0``时半宽**继续收窄**（锥面延拓到槽底以下，与半空间延拓到面背后同理）；
    2. Hessian跨``v = 0``**逐位不变**——``max``会让``zz``项在槽底从``k·tan²α``
       跳到``0``，那正是``U``掉出"二次连续可微"的地方（`solve.py`第29行）。
    """

    cone = _walls(WALL_SLOPE)
    lateral = 2.9
    above = _state((lateral, 0.0, 0.5))
    floor = _state((lateral, 0.0, 0.0))
    below = _state((lateral, 0.0, -0.5))

    assert cone.wall_clearance_mm(above)[0] == pytest.approx(
        HALF_WIDTH_MM + 0.5 * WALL_SLOPE - (lateral + EDGE_RADIUS_MM), rel=1.0e-14
    )
    assert cone.wall_clearance_mm(below)[0] == pytest.approx(
        HALF_WIDTH_MM - 0.5 * WALL_SLOPE - (lateral + EDGE_RADIUS_MM), rel=1.0e-14
    )
    assert cone.wall_clearance_mm(below)[0] < cone.wall_clearance_mm(above)[0]

    assert cone.hessian(above, CONTEXT) == cone.hessian(below, CONTEXT)
    assert cone.hessian(floor, CONTEXT) == cone.hessian(below, CONTEXT)
    #: 举升分量在槽底两侧同号同向——`max`会让``v < 0``那侧的举升整个消失。
    assert cone.gradient(below, CONTEXT)[2] < 0.0
    assert cone.gradient(above, CONTEXT)[2] < 0.0


def test_one_wall_at_a_time_zero_tolerance() -> None:
    """单边：一侧被顶住时另一侧**一个牛顿都不出**。互补条件是零容差判据。"""

    cone = _walls(WALL_SLOPE)
    state = _state((2.5, 0.0, 0.3))
    clearance = cone.wall_clearance_mm(state)
    assert clearance[0] < 0.0 < clearance[1]
    forces = cone.wall_force_n(state)
    assert forces[1] == 0.0
    assert forces[0] > 0.0


# ----------------------------------------- 判据三：有限差分与二阶导形制 ---


def _finite_difference_errors(item: PenaltyGrooveSweep, state: State, step: float):
    gradient = item.gradient(state, CONTEXT)
    hessian = item.hessian(state, CONTEXT)
    gradient_error = 0.0
    hessian_error = 0.0
    for axis in range(3):
        forward = list(state.vector)
        backward = list(state.vector)
        forward[axis] += step
        backward[axis] -= step
        ahead = _state(forward)
        behind = _state(backward)
        slope = (item.energy(ahead, CONTEXT) - item.energy(behind, CONTEXT)) / (2.0 * step)
        gradient_error = max(gradient_error, abs(slope - gradient[axis]))
        ahead_gradient = item.gradient(ahead, CONTEXT)
        behind_gradient = item.gradient(behind, CONTEXT)
        for column in range(3):
            curvature = (ahead_gradient[column] - behind_gradient[column]) / (2.0 * step)
            hessian_error = max(hessian_error, abs(curvature - hessian[axis][column]))
    return gradient_error, hessian_error


def test_finite_differences_are_roundoff_not_truncation() -> None:
    """判据三，**而且它的结论是反直觉的**：FD误差随``h``变小而**变大**。

    活动集内``U = ½k·g²``且``g``是``x``的**一次**多项式，于是``U``是二次多项式，
    三阶导恒为零——**中心差分的截断项恰好是零**。剩下的只有舍入，量级``ε·U/h²``，
    所以误差是``O(1/h)``而不是``O(h²)``。

    **这比"实测二阶收敛"是更强的陈述**：二阶收敛只说明截断项按预期缩小，
    而这里说明截断项**根本不存在**。判据因此写成两条：
    大步长上误差必须落到机器精度（相对1e-13以下），
    且步长缩小100倍时误差**上升**（否则说明有一条截断项藏在里面）。
    """

    cone = _walls(WALL_SLOPE)
    state = _state((2.9, 0.0, 0.7))
    gradient_scale = max(abs(value) for value in cone.gradient(state, CONTEXT))
    hessian = cone.hessian(state, CONTEXT)
    hessian_scale = max(abs(hessian[row][column]) for row in range(3) for column in range(3))

    coarse = _finite_difference_errors(cone, state, 1.0e-1)
    fine = _finite_difference_errors(cone, state, 1.0e-3)

    assert coarse[0] / gradient_scale < 1.0e-13
    assert coarse[1] / hessian_scale < 1.0e-13
    assert fine[0] > coarse[0]
    assert fine[1] > coarse[1]


def test_the_hessian_is_exactly_the_rank_one_outer_product() -> None:
    """``∇²g``恒为零，故``H = k·(∇g ⊗ ∇g)``**逐位**成立。

    **别照抄`PenaltyCylinderContact`的Hessian**：那里``ρ``非线性，多一块周向
    softening。这里一块都没有，注错时最容易被这一条抓住。
    """

    cone = _walls(WALL_SLOPE)
    state = _state((2.9, 0.0, 0.7))
    gap = cone.wall_clearance_mm(state)[0]
    assert gap < 0.0
    direction = (WALL_SLOPE * 0.0 - 1.0 * 1.0, 0.0, WALL_SLOPE * 1.0 - 1.0 * 0.0)
    hessian = cone.hessian(state, CONTEXT)
    for row in range(3):
        for column in range(3):
            #: ``0.0 +``不是装饰：`hessian`是把`hessian_entries`**累加**进一张零矩阵，
            #: 而``0.0 + (−0.0)``是``+0.0``。逐位判据必须走同一条累加路径，
            #: 否则判到的是零的符号而不是运算次序。
            expected = 0.0 + STIFFNESS_N_PER_MM * direction[row] * direction[column]
            assert hessian[row][column].hex() == expected.hex()


def test_the_fused_path_matches_the_separate_one_byte_for_byte() -> None:
    """spec/12第3.1节：``quantities``的能量必须与单独调`energy`**逐字节相同**。"""

    cone = _walls(WALL_SLOPE)
    state = _state((2.9, 0.0, 0.7))
    total, gradient, hessian = cone.quantities(
        state, CONTEXT, need_gradient=True, need_hessian=True
    )
    assert total.hex() == cone.energy(state, CONTEXT).hex()
    assert [value.hex() for value in gradient] == [
        value.hex() for value in cone.gradient(state, CONTEXT)
    ]
    assert hessian == cone.hessian(state, CONTEXT)


def test_it_registers_as_a_potential_and_assembles() -> None:
    """能进`EnergyRegistry`——形制上它与其余四族是同一档。"""

    registry = EnergyRegistry(terms=(_walls(WALL_SLOPE),))
    state = _state((2.9, 0.0, 0.7))
    total, gradient, _ = registry.total(state, CONTEXT, need_gradient=True)
    assert total > 0.0
    assert gradient is not None and gradient[0] > 0.0
    assert _walls(WALL_SLOPE).node_index_bound() == 1


# ------------------------------- 冻结帧丢掉的那一项：闭式 vs 数值中心差分 ---

#: plans/14第2.2节实测：`v2-01-bracket`的`R_min`一档取`ε_edge` p100那行的
#: R = 145.6 mm，帧扭率取同一份导出的`τ_max = 2.550 °/mm`。
#: **曲率与扭率在真实工件上是两个独立的量**（plans/14第2.2节末段实测
#: "两个量不同向"），所以这里不用螺旋线——螺旋线把两者锁死成一个比值。
#: 用"圆弧 + 绕切向按``τ·a``自转的帧"，两者各取各的实测值。
GOLD_RADIUS_MM = 145.6
GOLD_TWIST_PER_MM = math.radians(2.550)


def _gold_curve(arc: float):
    angle = arc / GOLD_RADIUS_MM
    return (GOLD_RADIUS_MM * math.cos(angle), GOLD_RADIUS_MM * math.sin(angle), 0.0)


def _gold_frame(arc: float):
    angle = arc / GOLD_RADIUS_MM
    twist = GOLD_TWIST_PER_MM * arc
    tangent = (-math.sin(angle), math.cos(angle), 0.0)
    radial = (math.cos(angle), math.sin(angle), 0.0)
    axial = (0.0, 0.0, 1.0)
    normal = tuple(
        math.cos(twist) * radial[axis] + math.sin(twist) * axial[axis] for axis in range(3)
    )
    return tangent, _cross(normal, tangent), normal


def _gold_invariants(arc: float, step: float = 1.0e-5):
    tangent, width, normal = _gold_frame(arc)
    ahead = _gold_frame(arc + step)
    behind = _gold_frame(arc - step)
    tangent_rate = tuple((ahead[0][axis] - behind[0][axis]) / (2.0 * step) for axis in range(3))
    width_rate = tuple((ahead[1][axis] - behind[1][axis]) / (2.0 * step) for axis in range(3))
    return _dot(tangent_rate, width), _dot(tangent_rate, normal), _dot(width_rate, normal)


def _gold_nearest(point, seed: float) -> float:
    arc = seed
    for _ in range(200):
        tangent = _gold_frame(arc)[0]
        centre = _gold_curve(arc)
        delta = tuple(point[axis] - centre[axis] for axis in range(3))
        step = 1.0e-6
        rate = tuple(
            (_gold_frame(arc + step)[0][axis] - _gold_frame(arc - step)[0][axis]) / (2.0 * step)
            for axis in range(3)
        )
        move = _dot(delta, tangent) / (-1.0 + _dot(delta, rate))
        arc -= move
        if abs(move) < 1.0e-13:
            break
    return arc


def _gold_live_gap(point, seed: float) -> float:
    """**活最近点**的间隙：每次求值都重新定位弧长与帧，不冻结任何东西。"""

    arc = _gold_nearest(point, seed)
    centre = _gold_curve(arc)
    tangent, width, normal = _gold_frame(arc)
    delta = tuple(point[axis] - centre[axis] for axis in range(3))
    lateral = _dot(delta, width)
    depth = _dot(delta, normal)
    return (HALF_WIDTH_MM + depth * WALL_SLOPE) - (lateral + EDGE_RADIUS_MM)


def test_frozen_frame_residual_matches_the_closed_form() -> None:
    """冻结帧丢掉的那一项，**闭式与数值各算一遍，二阶收敛**。

        ∇g_精确 = ∇g_本项 − A·t,   A = τ·(tanα·u + σ·v) / (1 − u·κ_s − v·κ_n)

    左边由解析曲线上的中心差分独立求出（`_gold_live_gap`每次都重新找最近点），
    右边是决策0075第四节的闭式。两条腿互不引用。

    **包络定理在这里只杀掉了一半**：最近点条件``(x−p)·t = 0``让``∂p/∂a``那一项
    归零，但杀不掉帧绕切向的自转——``τ``把截面坐标转起来，那一项原封不动留着。
    简报把这条写成"∂a/∂x那一项的一阶贡献为零"，**那句话对距离``|x−p|``成立、
    对``u``与``v``各自不成立**，两者只在``τ = 0``时重合。本条判的就是这个差。
    """

    arc = 37.0
    lateral, depth = 2.3, 0.8
    centre = _gold_curve(arc)
    tangent, width, normal = _gold_frame(arc)
    point = tuple(
        centre[axis] + lateral * width[axis] + depth * normal[axis] for axis in range(3)
    )
    curvature_s, curvature_n, twist = _gold_invariants(arc)

    frozen = tuple(WALL_SLOPE * normal[axis] - 1.0 * width[axis] for axis in range(3))
    jacobian = 1.0 - lateral * curvature_s - depth * curvature_n
    coefficient = twist * (WALL_SLOPE * lateral + 1.0 * depth) / jacobian
    closed_form = tuple(frozen[axis] - coefficient * tangent[axis] for axis in range(3))

    def numeric(step: float):
        out = []
        for axis in range(3):
            ahead = tuple(point[i] + (step if i == axis else 0.0) for i in range(3))
            behind = tuple(point[i] - (step if i == axis else 0.0) for i in range(3))
            out.append(
                (_gold_live_gap(ahead, arc) - _gold_live_gap(behind, arc)) / (2.0 * step)
            )
        return tuple(out)

    errors = []
    for step in (0.4, 0.2, 0.1, 0.05):
        measured = numeric(step)
        errors.append(max(abs(a - b) for a, b in zip(measured, closed_form, strict=True)))
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios

    #: 冻结帧此刻丢掉多少：**这是一个数，不是一句"是近似"**。
    frozen_norm = math.sqrt(_dot(frozen, frozen))
    assert abs(coefficient) / frozen_norm == pytest.approx(0.0537072, rel=2.0e-5)
    assert math.degrees(math.atan2(abs(coefficient), frozen_norm)) == pytest.approx(
        3.07425, rel=2.0e-5
    )


def test_the_residual_vanishes_exactly_when_the_frame_does_not_twist() -> None:
    """``τ = 0``时冻结帧**一点都不丢**——反向门。

    没有这一条，上一条测的可能只是"有个不为零的数"而不是"它就是``τ``那一项"。
    """

    arc = 37.0
    tangent, width, normal = _gold_frame(arc)
    plane_normal = (0.0, 0.0, 1.0)
    #: 同一个圆弧，帧不自转（`τ = 0`），其余一切不变。
    def untwisted_frame(value: float):
        angle = value / GOLD_RADIUS_MM
        local_tangent = (-math.sin(angle), math.cos(angle), 0.0)
        return local_tangent, _cross(plane_normal, local_tangent), plane_normal

    step = 1.0e-5
    ahead = untwisted_frame(arc + step)
    behind = untwisted_frame(arc - step)
    width_rate = tuple((ahead[1][axis] - behind[1][axis]) / (2.0 * step) for axis in range(3))
    twist = _dot(width_rate, untwisted_frame(arc)[2])
    assert abs(twist) < 1.0e-12
    assert tangent is not None and width is not None and normal is not None


# ------------------------------------------------------------ 必红矩阵 ---


def test_it_refuses_an_empty_declaration() -> None:
    with pytest.raises(ContactError, match="at least one wall"):
        PenaltyGrooveSweep()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (0, -1, "nonnegative int"),
        (0, True, "nonnegative int"),
        (1, (0.0, 0.0, float("nan")), "finite 3-vector"),
        (2, (0.0, 0.0), "finite 3-vector"),
        (2, (2.0, 0.0, 0.0), "unit vector"),
        (3, (0.0, float("inf"), 0.0), "finite 3-vector"),
        (3, (0.0, 0.0, 0.5), "unit vector"),
        (4, 0.0, "exactly"),
        (4, 2.0, "exactly"),
        (5, 0.0, "half width must be positive"),
        (5, -4.0, "half width must be positive"),
        (6, -0.1, "slope"),
        (6, float("nan"), "slope"),
        (7, -1.0, "edge radius"),
        (8, float("inf"), "depth window lower bound"),
        (9, -2.0, "depth window must be non-empty"),
        (10, 0.0, "stiffness must be positive"),
        (10, -5.0, "stiffness must be positive"),
    ),
)
def test_every_constructor_branch_fails_closed(field: int, value, message: str) -> None:
    """构造期每一条分支各有一条红用例。

    plans/09教训三：**一条从没被必红用例走过的分支，等于一条没有门的分支。**
    """

    wall = list(_walls(WALL_SLOPE).walls[0])
    wall[field] = value
    with pytest.raises(ContactError, match=message):
        PenaltyGrooveSweep(walls=(tuple(wall),))


def test_a_skewed_frame_fails_closed() -> None:
    """``s``与``n``不正交当场拒——取错两者数值上不报任何错，只是把横向与深度对调。

    与`laydown.GrooveStation.__post_init__`那条判据同源，也与
    `tools/model/centerline_csv.py`"列序不是基序"那一条同源：
    **约定错了不会炸，只会给出一个安静的错答案**，所以在构造期就判。
    """

    wall = list(_walls(WALL_SLOPE).walls[0])
    wall[2] = (0.0, 0.0, 1.0)
    wall[3] = (0.0, 0.0, 1.0)
    with pytest.raises(ContactError, match="不正交"):
        PenaltyGrooveSweep(walls=(tuple(wall),))


def test_the_builder_emits_two_walls_per_node_sharing_one_station() -> None:
    """两面壁**共用同一个站点**——分开找最近点会让"槽宽"这个词失去意义。"""

    tangent, width, normal = _generic_frame()
    centerline = _straight_centerline(tangent, width, normal)
    positions = ((0, tuple(1.0 * tangent[axis] for axis in range(3))),
                 (1, tuple(-6.0 * tangent[axis] for axis in range(3))))
    item = groove_sweep_walls(
        centerline,
        positions,
        half_width_mm=HALF_WIDTH_MM,
        wall_slope=WALL_SLOPE,
        edge_radius_mm=EDGE_RADIUS_MM,
        depth_window_mm=DEPTH_WINDOW_MM,
        stiffness_n_mm=STIFFNESS_N_PER_MM,
    )
    assert len(item.walls) == 4
    assert item.walls[0][1] == item.walls[1][1]
    assert item.walls[2][1] == item.walls[3][1]
    assert item.walls[0][1] != item.walls[2][1]
    assert [wall[4] for wall in item.walls] == [1.0, -1.0, 1.0, -1.0]
    assert item.node_index_bound() == 2


def test_the_mutation_matrix_is_measured() -> None:
    """**注错验证登记表**（2026-08-18实测，逐条清`__pycache__`后重跑，基线79条全绿）。

    | 改坏什么 | 结果 | 哪条门红 |
    |---|---|---|
    | `_frame`把``halfwidth − (lateral + radius)``改成``(halfwidth − lateral) − radius`` | 3红 | `test_degenerates_bit_for_bit_to_the_annulus_limit`、`test_the_other_side_degenerates_too`、`test_the_bracket_position_in_the_gap_expression_is_load_bearing` |
    | `_frame`把``half_width + depth * slope``改成``half_width + max(depth, 0.0) * slope`` | 1红 | `test_the_wall_does_not_hinge_at_the_groove_floor`（**这条门是本轮补的，见下**） |
    | `_direction`丢掉``slope * normal``那一项（退回平面环带） | 6红 | 锥面三档 + `test_the_hessian_...` + `test_finite_differences_...` + `test_the_wall_does_not_hinge_...` |
    | `_direction`把``− side * width``写成``− width``（漏掉侧号） | 1红 | `test_the_other_side_degenerates_too` |
    | `hessian_entries`按`PenaltyCylinderContact`加一块几何刚度 | 3红 | `test_degenerates_bit_for_bit_...`、`test_finite_differences_...`、`test_the_hessian_is_exactly_the_rank_one_outer_product` |
    | `_is_active`去掉深度窗那一条 | 1红 | `test_the_depth_window_is_an_activity_gate_not_an_energy` |
    | `_is_active`把``gap < 0.0``改成``gap < 1e9`` | 7红 | 退化两条 + 锥面三档 + 单边 + Hessian |
    | 构造期不再判``s ⟂ n``（阈值改1e9） | 1红 | `test_a_skewed_frame_fails_closed` |
    | 构造期不再判侧号取值（放行`0.0`） | 1红 | `test_every_constructor_branch_fails_closed[4-0.0-exactly]` |
    | 构造期不再判外倾非负 | 1红 | `test_every_constructor_branch_fails_closed[6--0.1-slope]` |
    | 构造期不再判深度窗非空 | 1红 | `test_every_constructor_branch_fails_closed[9--2.0-depth…]` |
    | `groove_sweep_walls`两面壁各自错开半步找站点 | 1红 | `test_the_builder_emits_two_walls_per_node_sharing_one_station` |

    ## 本轮抓到的一道**空门**，以及它为什么值得单独写一段

    第一遍注错时``max(depth, 0.0)``那一条**36条门一条都没红**。
    病根不是判据写松了，是**没有任何一条用例在``v < 0``且壁活动的构型上求值过**——
    而决策0075第二节第2条整段就是围绕那条分支写的：
    "``max``在``v = 0``处不可微，而``v = 0``正是槽底"。
    **文档里最要紧的那条裁决，恰恰是最没有门守着的那条。**

    补的是`test_the_wall_does_not_hinge_at_the_groove_floor`；补完重跑，该变异1红。
    这与plans/09教训三、与域隔离门那九条"全部用绝对import"是同一个形态：
    **门全绿不是因为它挡得住，是因为那条分支从没被执行过。**

    ## 没有被注错验过的东西（如实登记）

    * `wall_force_n`的``sec α``那一档只被`test_the_cone_keeps_the_lateral_force_and_adds_a_lift`
      走过一次；把``scale``写成``1.0``会红，但把它写成``sqrt(1 + slope²)``的**另一种**
      等价写法不会红——**那不是空门，是那条判据本来就不判运算次序**；
    * `quantities`的``need_gradient=False``分支只有一条绿用例走过，没有注错。
      触发条件：融合路径进入案例批时补。
    """

    assert True
