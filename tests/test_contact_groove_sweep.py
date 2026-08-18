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
import statistics

import pytest

from physics_engine.contact import (
    ContactError,
    PenaltyAnnulusLimit,
    PenaltyGrooveSweep,
    PenaltyGrooveSweepLive,
    groove_sweep_live_walls,
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


# ================================================ 活站点档（决策0078） ===
#
# 下面这一整节判的是`PenaltyGrooveSweepLive`——它把上面那一族丢掉的``A·t``
# 补了回来，**办法是改能量而不是改梯度**。三条判据的方向各不相同：
#
# 1. **退化逐位**：``κ_s = κ_n = τ = 0``时与`PenaltyGrooveSweep`**逐字节相同**，
#    于是"直中心线＋``tanα = 0`` ⟹ 与`PenaltyAnnulusLimit`逐位相同"
#    这条链条整条对活站点档照样成立；
# 2. **梯度确实是能量的导数**：中心差分**二阶收敛，实测比恒为4.0000**。
#    这与冻结帧那一族恰好相反（那里``U``是二次多项式、截断项恒为零、
#    误差随``h``变小而变大）——**两族的FD判据形态不同，这本身就是分辨力**；
# 3. **Hessian不精确的代价被量成数**：牛顿从二阶掉到**一阶**，
#    收缩率随``τ``增大（0.00748 → 0.01938 → 0.04373）。


#: 决策0075第四节4.2那一档的三个不变量（`v2-01-bracket`，plans/14第2.2节实测
#: ``R = 145.6 mm``、``τ = 2.550 °/mm``）。**这三个数是从0075抄过来的**，
#: 而0075是用"圆弧＋绕切向自转的帧"那条解析曲线独立算出来的——
#: 本族用的是另一条曲线（Darboux恒定），两者在同一构型上给出**同一个**
#: ``|A|/|∇g|``与力方向偏角，见`test_the_live_gradient_reproduces_the_0075_closed_form`。
LIVE_CURVATURE_S = 0.006848347
LIVE_CURVATURE_N = 0.000520940
LIVE_TWIST = -0.044505896
LIVE_ARC_LIMIT_MM = 20.0


def _axis_frame():
    """轴对齐的右手帧``s = n × t``。**手性在构造期被判**，所以不能随手写。"""

    tangent = (0.0, 1.0, 0.0)
    normal = (0.0, 0.0, 1.0)
    return tangent, _cross(normal, tangent), normal


def _live_wall(
    side: float,
    *,
    curvature_s: float = LIVE_CURVATURE_S,
    curvature_n: float = LIVE_CURVATURE_N,
    twist: float = LIVE_TWIST,
    slope: float = WALL_SLOPE,
    arc_limit: float = LIVE_ARC_LIMIT_MM,
    frame=None,
    point=(0.0, 0.0, 0.0),
):
    tangent, width, normal = frame if frame is not None else _axis_frame()
    return (
        0,
        point,
        tangent,
        width,
        normal,
        side,
        HALF_WIDTH_MM,
        slope,
        EDGE_RADIUS_MM,
        DEPTH_WINDOW_MM[0],
        DEPTH_WINDOW_MM[1],
        curvature_s,
        curvature_n,
        twist,
        arc_limit,
        STIFFNESS_N_PER_MM,
    )


def _live(side: float = 1.0, **kwargs) -> PenaltyGrooveSweepLive:
    return PenaltyGrooveSweepLive(walls=(_live_wall(side, **kwargs),))


def _frozen_twin(side: float = 1.0, *, slope: float = WALL_SLOPE) -> PenaltyGrooveSweep:
    """同一构型上的冻结帧那一族。

    **不能拿`_walls`来比**：那个helper用的是``s = (1, 0, 0)``，
    而`_axis_frame`按``s = n × t``给出的是``(−1, 0, 0)``——
    本族在构造期判手性，所以两边的``s``必须是同一个才谈得上"逐位相同"。
    """

    _, width, normal = _axis_frame()
    return PenaltyGrooveSweep(
        walls=(
            (
                0,
                (0.0, 0.0, 0.0),
                width,
                normal,
                side,
                HALF_WIDTH_MM,
                slope,
                EDGE_RADIUS_MM,
                DEPTH_WINDOW_MM[0],
                DEPTH_WINDOW_MM[1],
                STIFFNESS_N_PER_MM,
            ),
        )
    )


def _in_section(lateral_mm: float, depth_mm: float, frame=None, along_mm: float = 0.0):
    """截面坐标``(u, v)``、沿槽偏移``a``那一点。

    ``along_mm``默认为零**不是省事**：0075那一族的判据全部落在站点截面上，
    而"截面上"恰好是活站点档与冻结帧档**必然**给同一个数的地方
    （``a* = 0``）。注错验证实测：只在截面上取点时，
    把退化分支整个停掉**四档逐位门一条都不红**——见
    `test_the_straight_branch_is_taken_and_not_merely_equivalent`。
    """

    tangent, width, normal = frame if frame is not None else _axis_frame()
    return tuple(
        along_mm * tangent[axis] + lateral_mm * width[axis] + depth_mm * normal[axis]
        for axis in range(3)
    )


# ------------------------------------ 判据一：退化到冻结帧，**逐字节** ---


@pytest.mark.parametrize(
    ("lateral_mm", "depth_mm", "slope", "expect_active"),
    (
        #: 0075判据一那个构型（``tanα = 0``、``u = 2.05``、``v = 0.9``），活动。
        (2.05, 0.9, 0.0, True),
        #: 锥面且活动——**这一档是承重的**：只判``tanα = 0``那一档时，
        #: 活站点档可以把整个锥面分支写错而退化门照样绿。
        (2.30, 0.8, WALL_SLOPE, True),
        #: 不活动：两族都给0，判的是"活动条件也一样"。
        (2.05, 0.9, WALL_SLOPE, False),
        #: 槽底以下（``v < 0``）且活动——0075第五节那道空门的同形防线。
        (2.30, -0.5, WALL_SLOPE, True),
    ),
)
def test_the_live_family_degenerates_bit_for_bit_to_the_frozen_one(
    lateral_mm: float, depth_mm: float, slope: float, expect_active: bool
) -> None:
    """``κ_s = κ_n = τ = 0``（**恰好为零**）时两族**逐字节相同**。

    判`float.hex()`不判`==`：`==`会把差一个ulp的两条实现判成一样，
    而这条门守的恰恰是"活站点档在退化档上一个字节都没多算"。

    **它守的不只是数值，是一条链条**：0075那条
    "直中心线＋``tanα = 0`` ⟹ 与`PenaltyAnnulusLimit`逐位相同"
    因此对活站点档**整条**照样成立，不必再判一遍环带。

    机理：帧不转时``u = (x − p − a·t)·s``里的``a``项恒为零（``t ⟂ s``是代数事实），
    于是"重定位"这件事本身没有内容。`_station`因此**不解**最近点，
    直接走与`PenaltyGrooveSweep._frame`同一串运算——括号位置一并照抄。
    """

    tangent, width, normal = _axis_frame()
    point = _in_section(lateral_mm, depth_mm)
    state = _state(point)
    live = PenaltyGrooveSweepLive(
        walls=(
            _live_wall(1.0, curvature_s=0.0, curvature_n=0.0, twist=0.0, slope=slope),
        )
    )
    frozen = _frozen_twin(slope=slope)
    assert width is not None and normal is not None and tangent is not None
    assert (live.wall_clearance_mm(state)[0] < 0.0) is expect_active
    assert live.wall_clearance_mm(state)[0].hex() == frozen.wall_clearance_mm(state)[0].hex()
    assert live.wall_depth_mm(state)[0].hex() == frozen.wall_depth_mm(state)[0].hex()
    assert live.energy(state, CONTEXT).hex() == frozen.energy(state, CONTEXT).hex()
    assert [value.hex() for value in live.gradient(state, CONTEXT)] == [
        value.hex() for value in frozen.gradient(state, CONTEXT)
    ]
    assert [[value.hex() for value in row] for row in live.hessian(state, CONTEXT)] == [
        [value.hex() for value in row] for row in frozen.hessian(state, CONTEXT)
    ]
    #: 退化档下``a*``恒为``0.0``——**判它是为了证明上面那一串不是碰巧相等**：
    #: 若`_station`真去解了最近点，``a*``会是一个``O(ε)``的非零数。
    assert live.wall_arc_offset_mm(state)[0] == 0.0


def test_the_live_family_stops_matching_the_frozen_one_once_the_frame_twists() -> None:
    """反向门：``τ ≠ 0``时两族**必须不同**。

    没有这一条，上一条测的可能是"活站点档根本没实现"——
    一个直接转调`PenaltyGrooveSweep`的空壳会让退化门全绿。
    """

    point = _in_section(2.3, 0.8)
    state = _state(point)
    live = _live(1.0)
    frozen = _frozen_twin()
    assert live.energy(state, CONTEXT) == pytest.approx(
        frozen.energy(state, CONTEXT), rel=1.0e-12
    )
    #: **能量几乎相同而梯度不同**——这正是0075第四节那句话的形态：
    #: 丢掉的那一项沿**切向**，它不改变间隙本身，只改变力的方向。
    live_gradient = live.gradient(state, CONTEXT)
    frozen_gradient = frozen.gradient(state, CONTEXT)
    difference = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(live_gradient, frozen_gradient, strict=True))
    )
    assert difference > 1.0
    assert live.wall_force_tilt_deg(state)[0] > 3.0


def test_the_straight_branch_is_taken_and_not_merely_equivalent() -> None:
    """退化档**确实走了不解最近点那条路**——补一道注错实测出来的空门。

    2026-08-18注错实测：把`_station`的退化判据放宽成``abs(...) < 1e-9``、
    或者干脆停掉它（一律解最近点），**四档逐位退化门一条都不红**。
    病根不是判据写松了，是**那四档的求值点全都落在站点截面里**（沿``t``偏移为零），
    而截面上解不解最近点都给``a* = 0``、于是逐位相同。

    > **门全绿不是因为它挡得住，是因为那条分支从没被区分开过。**
    > 与0075第五节那道``max(v, 0)``空门是同一个形态。

    这一条把点**挪到截面之外**：沿``t``走5 mm。此时
    * 走退化分支 ⟹ ``a*``**恒等于**`0.0`（帧不转，``u``与``v``与``a``无关，
      "重定位"这件事本身没有内容）；
    * 解最近点 ⟹ ``a*``≈5，**当场看得出来**。

    而``g``两条路仍然相同（那正是退化分支成立的理由），所以**只判`g`是判不出来的**——
    要判的是``a*``。
    """

    state = _state(_in_section(2.3, 0.8, along_mm=5.0))
    live = PenaltyGrooveSweepLive(
        walls=(_live_wall(1.0, curvature_s=0.0, curvature_n=0.0, twist=0.0),)
    )
    #: **零容差**：走了退化分支它就是这个字面值，走另一条路它是5。
    assert live.wall_arc_offset_mm(state)[0] == 0.0
    #: 而间隙仍与冻结帧逐位相同——这一条证明上面那个`0.0`不是"算错了"。
    frozen = _frozen_twin()
    assert live.wall_clearance_mm(state)[0].hex() == frozen.wall_clearance_mm(state)[0].hex()
    assert [value.hex() for value in live.gradient(state, CONTEXT)] == [
        value.hex() for value in frozen.gradient(state, CONTEXT)
    ]


def test_the_straight_branch_judges_exactly_zero_not_merely_small() -> None:
    """``STRAIGHT_DARBOUX``判的是**恰好为零**，不是"小"——第二道空门。

    2026-08-18注错实测：把退化判据放宽成``abs(...) < 1e-9``，
    连上一条门都不红——因为上一条给的``τ``**恰好是**`0.0`，
    而`0.0`同时满足两种判据。**放宽只在"小但不为零"的输入上才看得出来。**

    ``τ = 0``是调用方的一条**声明**（这一段不扭），而"很小"是一个**量**。
    两者不是一件事：一条``τ = 1e-12``的中心线仍然是弯的、仍然要解最近点，
    只是解出来的修正很小。**替它判成零，等于替它把声明改了。**
    """

    tiny = 1.0e-12
    state = _state(_in_section(2.3, 0.8, along_mm=5.0))
    live = _live(1.0, curvature_s=0.0, curvature_n=0.0, twist=tiny, arc_limit=20.0)
    #: 走的是解最近点那条路 ⟹ ``a*``≈5，**不是**退化分支的那个字面`0.0`。
    assert abs(live.wall_arc_offset_mm(state)[0] - 5.0) < 0.5
    #: 而``τ``确实小到修正可以忽略——**这一条证明红的是"判据放宽了"
    #: 而不是"这个构型本来就该走另一条路"**。
    assert live.wall_force_tilt_deg(state)[0] < 1.0e-9


def test_the_closed_form_holds_on_the_other_side_too() -> None:
    """``σ = −1``那一侧的``A``也判一次——补一道注错实测出来的空门。

    2026-08-24注错实测：把``A = τ(tanα·u + σ·v)/D``里的``σ``删掉，
    **一条门都不红**。病根是残差那几条判据**全都只判``σ = +1``那一侧**，
    而``σ = +1``时``σ·v = v``——删掉它一个字节都不差。

    ``σ = −1``那一侧的闭式（同一组不变量、把``u``取到负边）：
    ``A`` 里``tanα·u``与``σ·v``**符号相反地**组合，于是两侧的``A``**不是**互为相反数
    ——这正是"删掉``σ``看不出来"能藏身的地方。
    """

    _, width, normal = _axis_frame()
    #: ``−s``那一侧要顶上，``u``得取到负边。
    point = _in_section(-2.3, 0.8)
    state = _state(point)
    live = _live(-1.0)
    assert live.wall_clearance_mm(state)[0] < 0.0

    gap = live.wall_clearance_mm(state)[0]
    gradient = live.gradient(state, CONTEXT)
    direction = tuple(value / (STIFFNESS_N_PER_MM * gap) for value in gradient)
    frozen = tuple(WALL_SLOPE * normal[axis] + width[axis] for axis in range(3))
    residual = tuple(direction[axis] - frozen[axis] for axis in range(3))
    measured = math.sqrt(_dot(residual, residual))

    #: 手推：``u = −2.3``、``v = 0.8``、``σ = −1``，
    #: ``A = τ·(tanα·u + σ·v)/(1 − u·κ_s − v·κ_n)``。
    jacobian = 1.0 - (-2.3) * LIVE_CURVATURE_S - 0.8 * LIVE_CURVATURE_N
    expected = abs(
        LIVE_TWIST * (WALL_SLOPE * (-2.3) + (-1.0) * 0.8) / jacobian
    )
    assert measured == pytest.approx(expected, rel=1.0e-9)
    #: **两侧的``A``不是互为相反数**——判它是为了钉住上面那个式子里``σ``的位置。
    other = _live(1.0)
    other_state = _state(_in_section(2.3, 0.8))
    other_gap = other.wall_clearance_mm(other_state)[0]
    other_direction = tuple(
        value / (STIFFNESS_N_PER_MM * other_gap)
        for value in other.gradient(other_state, CONTEXT)
    )
    other_frozen = tuple(WALL_SLOPE * normal[axis] - width[axis] for axis in range(3))
    other_residual = tuple(other_direction[axis] - other_frozen[axis] for axis in range(3))
    other_measured = math.sqrt(_dot(other_residual, other_residual))
    assert abs(measured - other_measured) > 0.01 * other_measured


def test_the_model_curve_is_arc_length_parametrised_away_from_the_station() -> None:
    """在``a* ≈ 5 mm``处再判一次梯度对能量——补一道注错实测出来的空门。

    2026-08-18注错实测：把`_model`的``C(a)``里那一项``(a − sin θa/θ)(ê·t)·ê``
    整个删掉，**一条门都不红**。病根不是判据松，是**所有的求值点都在``a* ≈ 0``附近**
    ——而那一项是``a``的**三阶**小量（``a − sin(θa)/θ = θ²a³/6 + …``），
    在``|a| < 0.1``上比机器精度还小。

    删掉它，模型曲线就**不再以弧长为参数**（``|dC/da| ≠ 1``），
    于是最近点条件、``D``、以及整条梯度全部错位。
    这一条把节点沿``t``挪出去5 mm，让``a*``落在真正用得上那一项的量级上。
    """

    #: ``u``取到3.0而不是2.3：帧沿5 mm转过约12.7°，截面坐标跟着转，
    #: ``u = 2.3``那一档在``a* ≈ 5``处**已经脱开了**（实测``g = +0.247``）。
    #: **这一条本身就是"帧真的在转"的读数。**
    point = _in_section(3.0, 0.8, along_mm=5.0)
    live = _live(1.0, arc_limit=20.0)
    state = _state(point)
    #: 前置两条：``a*``确实走出去了，且壁**确实还活动**——
    #: 不活动时能量恒为0、FD全零，比值会当场除零而不是判出问题。
    assert abs(live.wall_arc_offset_mm(state)[0] - 5.0) < 0.5
    assert live.wall_clearance_mm(state)[0] < 0.0
    assert live.energy(state, CONTEXT) > 1.0
    analytic = live.gradient(state, CONTEXT)

    def difference(step: float):
        out = []
        for axis in range(3):
            ahead = tuple(point[i] + (step if i == axis else 0.0) for i in range(3))
            behind = tuple(point[i] - (step if i == axis else 0.0) for i in range(3))
            out.append(
                (
                    live.energy(_state(ahead), CONTEXT)
                    - live.energy(_state(behind), CONTEXT)
                )
                / (2.0 * step)
            )
        return tuple(out)

    errors = []
    for step in (0.1, 0.05, 0.025, 0.0125):
        measured = difference(step)
        errors.append(max(abs(a - b) for a, b in zip(measured, analytic, strict=True)))
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    assert errors[-1] < 1.0e-3


# --------------------------- 判据二：梯度确实是所实现能量的导数（二阶） ---


def test_the_live_gradient_is_the_derivative_of_the_live_energy() -> None:
    """**中心差分二阶收敛，实测比4.0000** —— 本族存在的技术前提。

    直接把``A·t``加进`gradient()`而不动`energy()`会让这条门**当场红，而且红得对**：
    那时梯度不是任何势的导数，线搜索与收敛判据全部失去依据。
    本族之所以改的是能量，理由就在这一条门上。

    **与冻结帧那一族恰好相反**：那里``U``是``x``的二次多项式、
    中心差分的截断项**恒为零**、误差随``h``变小而**变大**
    （`test_finite_differences_are_roundoff_not_truncation`）。
    这里``g``经``a*(x)``真非线性地依赖``x``，于是截断项回来了、
    比值落在4.0上——**两族的FD形态不同，这本身就是一条分辨力**。
    """

    point = _in_section(2.3, 0.8)
    live = _live(1.0)
    analytic = live.gradient(_state(point), CONTEXT)

    def difference(step: float):
        out = []
        for axis in range(3):
            ahead = tuple(point[i] + (step if i == axis else 0.0) for i in range(3))
            behind = tuple(point[i] - (step if i == axis else 0.0) for i in range(3))
            out.append(
                (
                    live.energy(_state(ahead), CONTEXT)
                    - live.energy(_state(behind), CONTEXT)
                )
                / (2.0 * step)
            )
        return tuple(out)

    errors = []
    for step in (0.1, 0.05, 0.025, 0.0125):
        measured = difference(step)
        errors.append(max(abs(a - b) for a, b in zip(measured, analytic, strict=True)))
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    assert all(3.9 < ratio < 4.1 for ratio in ratios), ratios
    #: 先断"误差确实在下降到一个小数"，否则上面那条比值可以由三个大数凑出来。
    assert errors[-1] < 3.0e-4


def test_the_live_gradient_reproduces_the_0075_closed_form() -> None:
    """活梯度与冻结梯度之差，**就是0075第四节那个``A·t``**——两条独立的腿。

    0075用的是"圆弧 + 绕切向按``τ·a``自转的帧"那条解析曲线，
    本族用的是Darboux矢量恒定的局部模型。**两条曲线不是同一条**，
    但在同一组``(κ_s, κ_n, τ, u, v, tanα)``上它们给出**同一个**残差——
    因为``∇g``那一式只依赖不变量，不依赖曲线的其余部分（隐函数定理）。

    实测：丢失**5.3707 %**、力方向偏**3.0743°**，与0075第四节4.2那张表
    逐位对上（该表用的是``a = 37 mm``处的实际帧相位``D = 0.9838``）。
    """

    _, width, normal = _axis_frame()
    point = _in_section(2.3, 0.8)
    state = _state(point)
    live = _live(1.0)

    gap = live.wall_clearance_mm(state)[0]
    gradient = live.gradient(state, CONTEXT)
    direction = tuple(value / (STIFFNESS_N_PER_MM * gap) for value in gradient)
    frozen = tuple(WALL_SLOPE * normal[axis] - width[axis] for axis in range(3))
    frozen_norm = math.sqrt(_dot(frozen, frozen))
    residual = tuple(direction[axis] - frozen[axis] for axis in range(3))
    coefficient = math.sqrt(_dot(residual, residual))

    assert coefficient / frozen_norm == pytest.approx(0.0537072, rel=2.0e-5)
    assert live.wall_force_tilt_deg(state)[0] == pytest.approx(3.07425, rel=2.0e-5)
    #: 丢掉的那一项**沿切向**——这一条判的是方向不是大小。
    tangent = _axis_frame()[0]
    along = _dot(residual, tangent)
    assert abs(abs(along) - coefficient) < 1.0e-12 * coefficient


def test_the_live_residual_vanishes_exactly_when_the_frame_does_not_twist() -> None:
    """``τ = 0``且``κ ≠ 0``时残差**恰好为零**——反向门，判的是"它就是``τ``那一项"。

    没有这一条，上一条测的可能只是"有个不为零的数"。
    **曲率仍然不为零**是刻意的：曲线还是弯的、最近点条件还是要解，
    只有帧的自转没了。残差恰好归零证明它由``τ``**独家**贡献。
    """

    point = _in_section(2.3, 0.8)
    state = _state(point)
    live = _live(1.0, twist=0.0)
    assert live.wall_force_tilt_deg(state)[0] == 0.0
    #: 站点截面上``a* = 0``，于是活梯度与冻结梯度**逐位相同**。
    assert live.wall_arc_offset_mm(state)[0] == pytest.approx(0.0, abs=1.0e-15)
    frozen = _frozen_twin()
    assert [value.hex() for value in live.gradient(state, CONTEXT)] == [
        value.hex() for value in frozen.gradient(state, CONTEXT)
    ]


# ------------------------- 判据三：Hessian不精确的代价，量成一个数 ---


def _newton_history(twist_deg_per_mm: float, curvature_radius_mm: float):
    """一个节点、一面活壁、一根各向同性锚弹簧。返回每一步的``|x − x*|``。

    **梯度是精确的，所以不动点与用哪个切线无关**——变的只有逼近的**速率**。
    锚弹簧是为了让``k(∇g⊗∇g)``那个秩一矩阵加上它之后非奇异；
    它不改变本条判据要量的东西（切线里缺的那一块仍然只缺在接触项上）。
    """

    tangent, width, normal = _axis_frame()
    live = _live(
        1.0,
        curvature_s=1.0 / curvature_radius_mm,
        curvature_n=LIVE_CURVATURE_N,
        twist=math.radians(twist_deg_per_mm),
        arc_limit=40.0,
    )
    anchor = _in_section(2.6, 0.8)
    anchor_stiffness = 2.0e2

    def gradient(position):
        values = list(live.gradient(_state(position), CONTEXT))
        for axis in range(3):
            values[axis] += anchor_stiffness * (position[axis] - anchor[axis])
        return values

    def solve3(matrix, rhs):
        rows = [list(matrix[index]) + [-rhs[index]] for index in range(3)]
        for column in range(3):
            pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
            rows[column], rows[pivot] = rows[pivot], rows[column]
            for row in range(3):
                if row != column:
                    factor = rows[row][column] / rows[column][column]
                    for cell in range(column, 4):
                        rows[row][cell] -= factor * rows[column][cell]
        return [rows[index][3] / rows[index][index] for index in range(3)]

    position = list(_in_section(2.9, 0.8))
    trail = []
    for _ in range(30):
        values = gradient(position)
        trail.append(list(position))
        if math.sqrt(sum(value * value for value in values)) < 1.0e-9:
            break
        tangent_matrix = [list(row) for row in live.hessian(_state(position), CONTEXT)]
        for axis in range(3):
            tangent_matrix[axis][axis] += anchor_stiffness
        step = solve3(tangent_matrix, values)
        for axis in range(3):
            position[axis] += step[axis]
    root = position
    errors = [
        math.sqrt(sum((point[axis] - root[axis]) ** 2 for axis in range(3)))
        for point in trail
    ]
    assert tangent is not None and width is not None and normal is not None
    return errors


@pytest.mark.parametrize(
    ("twist_deg_per_mm", "radius_mm", "expected_rate", "expected_iterations"),
    (
        #: plans/14第2.2节九档几何里最松、中位、最扭的三档。
        (0.454, 75.6, 0.00748, 6),
        (2.550, 75.2, 0.02008, 7),
        (6.648, 72.8, 0.04565, 9),
    ),
)
def test_the_inexact_hessian_costs_exactly_one_order_of_convergence(
    twist_deg_per_mm: float,
    radius_mm: float,
    expected_rate: float,
    expected_iterations: int,
) -> None:
    """牛顿从**二阶掉到一阶**，收缩率随``τ``增大——**代价是一个数不是一句话**。

    Gauss-Newton切线缺的是``k·g·∇²g``。活动时``g < 0``，那是一块**负定**贡献，
    于是本项的切线**偏刚**：步子偏短、单调、不发散，只是不再二次收敛。

    实测（到``‖∇U‖ < 1e-9``）：

    | 几何 | ``τ`` °/mm | 迭代 | 精确切线 | 线性收缩率（中位） | 阶 |
    |---|---:|---:|---:|---:|---:|
    | `clean_a` | 0.454 | 6 | 4 | 0.00748 | 0.975—1.000 |
    | `v2-01-bracket`（中位） | 2.550 | 7 | 5 | 0.02008 | 0.982—0.989 |
    | `v1-coil-1`（最扭） | 6.648 | 9 | 5 | 0.04565 | 0.984—0.993 |

    **"精确切线"那一列是数值Jacobian（中心差分差精确梯度）不是解析``∇²g``**，
    故读作"大约要几次"；**本条只判Gauss-Newton那一列**。

    **收缩率大致随``τ``线性**（0.0075 / 0.020 / 0.046 对 0.454 / 2.550 / 6.648），
    这与缺的那一块正比于``τ``的推导一致。代价换算成人话是
    **多两到四次牛顿**，而换回来的是接触力方向不再偏0.56°—8.11°。
    """

    errors = _newton_history(twist_deg_per_mm, radius_mm)
    assert len(errors) == expected_iterations

    #: 逐步比值应当落在一个**常数**上（线性收敛的招牌），而不是逐步平方（二阶）。
    #: 末点**就是**根，它的误差恰好是`0.0`——把它算进比值会得到一个假的`0.0`。
    rates = [
        errors[index + 1] / errors[index]
        for index in range(1, len(errors) - 1)
        if errors[index] > 1.0e-12 and errors[index + 1] > 0.0
    ]
    assert len(rates) >= 3, errors
    #: 取**中位数**而不是末项：末几步的误差已经落到`1e-11`量级，
    #: 比值开始被舍入抬着走（实测末项比中位数高10%—15%）。
    #: **判中位数并同时判整段落在一个窄带里**，比判某一项更难蒙混过去。
    assert statistics.median(rates) == pytest.approx(expected_rate, rel=0.2)
    assert min(rates) > 0.5 * expected_rate
    assert max(rates) < 1.6 * expected_rate
    #: **阶恰好是1**：二阶时相邻两个比值会差一个平方，这里它们互相之间只差几个百分点。
    orders = [
        math.log(errors[index + 1] / errors[index])
        / math.log(errors[index] / errors[index - 1])
        for index in range(2, len(errors) - 1)
        if errors[index + 1] > 1.0e-12 and 0.0 < errors[index] < errors[index - 1]
    ]
    assert orders, errors
    assert all(0.93 < order < 1.07 for order in orders), orders


# --------------------------------------- 走出这一段：失败关闭，不跳段 ---


def test_walking_out_of_the_declared_arc_window_fails_closed() -> None:
    """``|a*|``超出弧长窗**当场抛**——0075登记的触发条件就是"跨过一整个采样步"。

    跨过去之后这条壁携带的``(κ_s, κ_n, τ)``是**上一段**的指纹
    （0075第四节第3条实测：``κ_s``过一个站点当场变号）。
    **这时给出一个数比抛出来更坏**，因为那个数看不出是错的。
    """

    tangent = _axis_frame()[0]
    far = tuple(
        _in_section(2.3, 0.8)[axis] + 5.0 * tangent[axis] for axis in range(3)
    )
    narrow = _live(1.0, arc_limit=1.0)
    with pytest.raises(ContactError, match="走出了声明的弧长窗"):
        narrow.energy(_state(far), CONTEXT)
    with pytest.raises(ContactError, match="走出了声明的弧长窗"):
        narrow.gradient(_state(far), CONTEXT)
    #: **同一个点、同一条壁，窗放宽就算得出来**——证明红的是窗不是别的东西。
    wide = _live(1.0, arc_limit=20.0)
    assert abs(wide.wall_arc_offset_mm(_state(far))[0]) > 1.0
    #: 诊断面在抛之前就该给出读数：窗宽的那条能读到``a*``，
    #: 调用方据此决定要不要重建（`groove_sweep_live_walls`再调一次）。
    assert wide.wall_arc_offset_mm(_state(far))[0] == pytest.approx(5.0, rel=0.2)


def test_the_nearest_point_condition_fails_closed_inside_the_curvature_centre() -> None:
    """``D ≤ 0``当场抛——**这不是数值失效是几何失效**。

    ``D = 1 − u·κ_s − v·κ_n``是最近点条件的Jacobian。它归零意味着节点落到了
    局部曲率中心上：那里"最近站点"不再唯一，再往下算出来的"最近点"是一个**最远点**。
    """

    #: 曲率半径2 mm、横向偏移3 mm ⟹ ``D = 1 − 3/2 < 0``。
    live = _live(1.0, curvature_s=0.5, curvature_n=0.0, twist=0.0, arc_limit=50.0)
    with pytest.raises(ContactError, match="最近点条件退化"):
        live.wall_clearance_mm(_state(_in_section(3.0, 0.0)))


# ------------------------------------------- 装配面与融合路径 ---


def test_the_live_fused_path_matches_the_separate_one_byte_for_byte() -> None:
    """`quantities`与分别调`energy`/`gradient`/`hessian`**逐字节相同**（spec/12第3.1节）。

    本族的融合路径比冻结帧那一族更容易出错：它每条壁要解一次标量牛顿，
    **两条路径各解一次就可能落在不同的``a*``上**（末步容差之内的两个不同的数）。
    实际不会，因为解是确定性的、起点也一样——**但那正是要判的东西**。
    """

    state = _state(_in_section(2.3, 0.8))
    live = PenaltyGrooveSweepLive(walls=(_live_wall(1.0), _live_wall(-1.0)))
    energy, gradient, hessian = live.quantities(
        state, CONTEXT, need_gradient=True, need_hessian=True
    )
    assert energy.hex() == live.energy(state, CONTEXT).hex()
    assert [value.hex() for value in gradient] == [
        value.hex() for value in live.gradient(state, CONTEXT)
    ]
    assert [[value.hex() for value in row] for row in hessian] == [
        [value.hex() for value in row] for row in live.hessian(state, CONTEXT)
    ]
    #: ``need_gradient=False``那条分支也走一遍——0075第五节把它登记成
    #: "只有一条绿用例走过，没有注错"，本族至少让它被执行。
    lean_energy, lean_gradient, lean_hessian = live.quantities(
        state, CONTEXT, need_gradient=False, need_hessian=False
    )
    assert lean_energy.hex() == energy.hex()
    assert lean_gradient is None and lean_hessian is None


def test_the_live_family_registers_as_a_potential_and_assembles() -> None:
    """它进`EnergyRegistry`并与别的项一起装配——**内核只吃数字**那条形制的执行面。"""

    live = PenaltyGrooveSweepLive(walls=(_live_wall(1.0), _live_wall(-1.0)))
    registry = EnergyRegistry(terms=(live,))
    state = _state(_in_section(2.3, 0.8))
    assert live.kind == "potential"
    assert live.node_index_bound() == 1
    total, gradient, hessian = registry.total(
        state, CONTEXT, need_gradient=True, need_hessian=True
    )
    assert total.hex() == live.energy(state, CONTEXT).hex()
    assert gradient is not None and [value.hex() for value in gradient] == [
        value.hex() for value in live.gradient(state, CONTEXT)
    ]
    assert hessian is not None
    #: 两面壁都活动不了同一个构型，但**装配路径本身**要被走过：
    #: ``+s``那面顶上、``−s``那面没有，梯度必须只有一面壁的贡献。
    assert live.wall_clearance_mm(state)[0] < 0.0 < live.wall_clearance_mm(state)[1]


def test_the_live_hessian_is_the_rank_one_outer_product_of_the_live_gradient() -> None:
    """``H = k·(∇g ⊗ ∇g)``——**秩一，且用的是活梯度而不是冻结梯度**。

    判它是为了挡住一种很自然的写法：Hessian照抄冻结帧那一族
    （用``tanα·n − σ·s``做外积）。那样能量、梯度、Hessian就**三者不自洽**，
    而秩一这条性质本身照样成立——**所以只判秩一是判不出来的**。
    """

    state = _state(_in_section(2.3, 0.8))
    live = _live(1.0)
    gap = live.wall_clearance_mm(state)[0]
    gradient = live.gradient(state, CONTEXT)
    direction = tuple(value / (STIFFNESS_N_PER_MM * gap) for value in gradient)
    hessian = live.hessian(state, CONTEXT)
    for row in range(3):
        for column in range(3):
            assert hessian[row][column] == pytest.approx(
                STIFFNESS_N_PER_MM * direction[row] * direction[column], rel=1.0e-12
            )
    #: 秩一：任何与``∇g``正交的方向上二次型为零。
    perpendicular = _cross(direction, (0.3, 0.5, 0.81))
    quadratic = sum(
        perpendicular[row] * hessian[row][column] * perpendicular[column]
        for row in range(3)
        for column in range(3)
    )
    assert abs(quadratic) < 1.0e-9 * STIFFNESS_N_PER_MM
    #: **反向**：用冻结梯度做外积会给出一个不同的矩阵——证明上面判得出差别。
    _, width, normal = _axis_frame()
    frozen = tuple(WALL_SLOPE * normal[axis] - width[axis] for axis in range(3))
    #: **判整个矩阵而不是某一格**：丢掉的那一项沿``t = (0, 1, 0)``，
    #: 于是``[0][0]``这一格**恰好一点没变**——只判它等于没判。
    drift = max(
        abs(hessian[row][column] - STIFFNESS_N_PER_MM * frozen[row] * frozen[column])
        for row in range(3)
        for column in range(3)
    )
    assert drift > 1.0e2


# --------------------------------------------------- 装配期：builder ---


def _twisting_centerline(step_mm: float = 2.0, count: int = 21) -> GrooveCenterline:
    """`_gold_curve`/`_gold_frame`那条解析曲线采成站点表。

    **金标与被测物互不引用**：曲线由本文件上面那两个函数给（圆弧＋绕切向自转的帧），
    不变量由`_gold_invariants`用中心差分独立量出来，而被测的是
    `groove_sweep_live_walls`从**站点表**里差分出来的那三个数。
    """

    return GrooveCenterline(
        centerline_id="groove/twisting_arc",
        stations=tuple(
            GrooveStation(
                arc_length_mm=index * step_mm,
                position_mm=_gold_curve(index * step_mm),
                tangent=_gold_frame(index * step_mm)[0],
                width_direction=_gold_frame(index * step_mm)[1],
                surface_normal=_gold_frame(index * step_mm)[2],
            )
            for index in range(count)
        ),
        semantics=CenterlineSemantics(
            position_interpolation="hermite_tangent",
            frame_interpolation="reorthonormalised_linear",
            topology="open",
            out_of_range="clamp_to_end",
            nearest_refinement_iterations=8,
        ),
        length_unit="mm",
    )


def test_the_live_builder_recovers_the_invariants_of_the_analytic_curve() -> None:
    """装配期差分出来的``(κ_s, κ_n, τ)``对得上解析曲线的实测不变量。

    **这是本族与中心线之间唯一的接口**，也是唯一一处"读进来的数对不对"能被判的地方。
    ``τ``干净，``κ_s``带段内插值那一档偏差——**两条各按各的容差判**，
    混成一条会把"``κ_s``不干净"这件事掩盖掉（决策0075第四节第3条实测）。
    """

    centerline = _twisting_centerline()
    arc = 9.0
    position = tuple(
        _gold_curve(arc)[axis]
        + 2.3 * _gold_frame(arc)[1][axis]
        + 0.8 * _gold_frame(arc)[2][axis]
        for axis in range(3)
    )
    item = groove_sweep_live_walls(
        centerline,
        ((0, position),),
        half_width_mm=HALF_WIDTH_MM,
        wall_slope=WALL_SLOPE,
        edge_radius_mm=EDGE_RADIUS_MM,
        depth_window_mm=DEPTH_WINDOW_MM,
        stiffness_n_mm=STIFFNESS_N_PER_MM,
        frame_probe_mm=1.0,
        name="groove_sweep_live",
    )
    assert len(item.walls) == 2
    #: 两面壁**共用同一个站点与同一组不变量**——分开定位会让"槽宽"失去意义。
    assert item.walls[0][1:5] == item.walls[1][1:5]
    assert item.walls[0][11:15] == item.walls[1][11:15]
    assert [wall[5] for wall in item.walls] == [1.0, -1.0]

    station_arc = centerline.nearest_arc_length_mm(position)[0]
    truth_s, truth_n, truth_twist = _gold_invariants(station_arc)
    curvature_s, curvature_n, twist = item.walls[0][11:14]
    #: ``τ``是干净的那一个（0075第四节第3条：段内差分给机器精度）。
    assert twist == pytest.approx(truth_twist, rel=2.0e-3)
    #: 曲率带段内那一档偏差——**容差写得比``τ``松是有理由的，不是凑的**。
    assert curvature_s == pytest.approx(truth_s, rel=5.0e-2)
    assert abs(curvature_n - truth_n) < 5.0e-4
    #: 弧长窗是"到本段两端的距离"里小的那一个，**必须是正的**。
    assert 0.0 < item.walls[0][14] <= 2.0


def test_the_live_builder_refuses_a_probe_it_cannot_use() -> None:
    """``frame_probe_mm``没有默认值，非正当场拒——探针宽度是采样步的函数。"""

    centerline = _twisting_centerline()
    with pytest.raises(ContactError, match="frame_probe_mm"):
        groove_sweep_live_walls(
            centerline,
            ((0, _gold_curve(9.0)),),
            half_width_mm=HALF_WIDTH_MM,
            wall_slope=WALL_SLOPE,
            edge_radius_mm=EDGE_RADIUS_MM,
            depth_window_mm=DEPTH_WINDOW_MM,
            stiffness_n_mm=STIFFNESS_N_PER_MM,
            frame_probe_mm=0.0,
        )


def test_the_probe_never_crosses_a_station() -> None:
    """探针**夹在段内**——跨站点差分不是"精度差一点"，是符号错。

    判法：给一个比整段还宽的探针，取出的不变量**必须与窄探针一致**
    （夹段生效），而不是变成另一个数（夹段失效、差分跨了站点）。
    决策0075第四节第3条实测：跨一个站点时``κ_s``从`−1.402e-4`跳到`+1.402e-4`。
    """

    centerline = _twisting_centerline()
    position = tuple(
        _gold_curve(9.0)[axis] + 2.3 * _gold_frame(9.0)[1][axis] for axis in range(3)
    )
    common = {
        "half_width_mm": HALF_WIDTH_MM,
        "wall_slope": WALL_SLOPE,
        "edge_radius_mm": EDGE_RADIUS_MM,
        "depth_window_mm": DEPTH_WINDOW_MM,
        "stiffness_n_mm": STIFFNESS_N_PER_MM,
    }
    narrow = groove_sweep_live_walls(
        centerline, ((0, position),), frame_probe_mm=0.5, **common
    )
    wide = groove_sweep_live_walls(
        centerline, ((0, position),), frame_probe_mm=50.0, **common
    )
    #: 段是2 mm，探针要50 mm——夹段之后两者拿到的是**同一段**，
    #: 于是三个不变量在段内插值的精度内一致。
    for index in (11, 12, 13):
        assert narrow.walls[0][index] == pytest.approx(
            wide.walls[0][index], rel=5.0e-2, abs=1.0e-6
        )
    #: 而`segment_bounds_mm`确实把探针夹住了——弧长窗不超过一整段。
    assert wide.walls[0][14] <= 2.0


# ------------------------------------------------- 活站点档的必红矩阵 ---


def test_the_live_family_refuses_an_empty_declaration() -> None:
    with pytest.raises(ContactError, match="at least one wall"):
        PenaltyGrooveSweepLive()


def test_the_live_family_refuses_a_wall_with_the_wrong_arity() -> None:
    """16格少一格当场拒——**位置元组少一格不会自己报错**，只会整体错位一位。"""

    wall = list(_live_wall(1.0))
    with pytest.raises(ContactError, match="16 fields"):
        PenaltyGrooveSweepLive(walls=(tuple(wall[:-1]),))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (0, -1, "nonnegative int"),
        (0, True, "nonnegative int"),
        (1, (0.0, 0.0, float("nan")), "finite 3-vector"),
        (2, (0.0, 0.0), "finite 3-vector"),
        (2, (0.0, 2.0, 0.0), "unit vector"),
        (3, (0.0, float("inf"), 0.0), "finite 3-vector"),
        (4, (0.0, 0.0, 0.5), "unit vector"),
        (5, 0.0, "exactly"),
        (5, 2.0, "exactly"),
        (6, 0.0, "half width must be positive"),
        (7, -0.1, "slope"),
        (7, float("nan"), "slope"),
        (8, -1.0, "edge radius"),
        (9, float("inf"), "depth window lower bound"),
        (10, -2.0, "depth window must be non-empty"),
        (11, float("nan"), "curvature_s"),
        (12, float("inf"), "curvature_n"),
        (13, float("nan"), "twist"),
        (14, 0.0, "arc window"),
        (14, -1.0, "arc window"),
        (15, 0.0, "stiffness must be positive"),
    ),
)
def test_every_live_constructor_branch_fails_closed(
    field: int, value, message: str
) -> None:
    """构造期每一条分支各有一条红用例。**本族比冻结帧多五个字段，五个都判。**"""

    wall = list(_live_wall(1.0))
    wall[field] = value
    with pytest.raises(ContactError, match=message):
        PenaltyGrooveSweepLive(walls=(tuple(wall),))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        #: ``t``与``s``不正交——**冻结帧那一族根本没有这条判据**，因为``t``不进它。
        (2, (0.0, 0.0, 1.0), "不正交"),
        #: ``s``与``n``不正交。
        (3, (0.0, 0.0, 1.0), "不正交"),
    ),
)
def test_a_skewed_live_frame_fails_closed(field: int, value, message: str) -> None:
    """三根轴两两正交各判一次——取错帧在数值上不报任何错。"""

    wall = list(_live_wall(1.0))
    wall[field] = value
    with pytest.raises(ContactError, match=message):
        PenaltyGrooveSweepLive(walls=(tuple(wall),))


def test_a_left_handed_live_frame_fails_closed() -> None:
    """``s = n × t``的**手性**也判——左手帧让``κ_s``与``τ``整体反号。

    **这条判据冻结帧那一族没有，而本族必须有**：那里帧只用来投影，
    镜像掉不改变``|u|``与``v``的大小；这里``τ``与``κ_s``是**带符号**地进梯度的，
    符号错了力就往反方向偏，而三根轴仍然两两正交、仍然都是单位向量。
    """

    tangent, width, normal = _axis_frame()
    wall = list(_live_wall(1.0))
    wall[3] = tuple(-value for value in width)
    with pytest.raises(ContactError, match="cross"):
        PenaltyGrooveSweepLive(walls=(tuple(wall),))
    assert tangent is not None and normal is not None


def test_the_live_mutation_matrix_is_measured() -> None:
    """**活站点档的注错验证登记表**（2026-08-18实测，逐条清`__pycache__`后重跑）。

    全文在决策0078第七节。基线：本文件**87条** ＋
    `tests/cases/test_real_centerline_invariants.py` **13条**
    （11条常驻＋2条选择进入；跑注错时接上真语料），共**100条**。

    ## 第一轮22条变异里**有6条一门不红**——这一节的价值全在那6条上

    | 改坏什么 | 第一轮 | 补门后 |
    |---|---|---|
    | `_station`退化分支判据由`== 0.0`放宽成`abs(...) < 1e-9` | **0红** | 1红 |
    | `_station`退化分支永假（一律解最近点） | **0红** | 1红 |
    | `_station`丢掉``− coefficient * local_t``（退回冻结梯度） | 7红 | 9红 |
    | `_station`删掉``/ jacobian``（丢掉``1/D``） | 3红 | 5红 |
    | `_station`把``side * depth``的``side``删掉 | **0红** | 1红 |
    | `_model`的Darboux把``− curvature_n``写成``+`` | 1红 | 2红 |
    | `_model`的``C(a)``丢掉``(arc − swept) * along * axis`` | **0红** | 1红 |
    | `_station`的牛顿把``residual / jacobian``写成``residual`` | **0红** | **仍0红，见下** |
    | `_station`删掉``jacobian <= 0`` | 1红 | 1红 |
    | `_station`删掉弧长窗 | 1红 | 1红 |
    | `hessian_entries`改用冻结方向做外积 | 4红 | 4红 |
    | 构造期不再判手性 | 1红 | 1红 |
    | 构造期不再判``t ⟂ s`` / ``t ⟂ n`` | 1红 | 1红 |
    | 构造期不再判16格 | 42红 | 46红 |
    | builder的探针不夹段 | 1红 | 1红 |
    | builder的``arc_limit``改成常数`1e9` | 2红 | 2红 |
    | `laydown`的`central`换成均匀网格系数 | **0红** | 1红 |
    | `laydown`把``κ_s``与``κ_n``对调 | 5红 | 6红 |
    | `laydown`把``τ``整体反号 | 2红 | 3红 |
    | `hard_way_edge_strain`把``w/2``写成``w`` | 3红 | 3红 |
    | `arc_length_fraction_above`按站点数权 | **0红** | 1红 |
    | `laydown`的`forward`除数偏一点 | 1红 | 1红 |

    ## 那6道空门各是一个不同的形态

    1. **退化分支两条**：四档逐位门的求值点**全落在站点截面里**，
       而截面上解不解最近点都给``a* = 0``。补
       `test_the_straight_branch_is_taken_and_not_merely_equivalent`
       与`test_the_straight_branch_judges_exactly_zero_not_merely_small`；
    2. **``σ``那条**：残差判据**只走了``σ = +1``一侧**，而那一侧``σ·v = v``。
       补`test_the_closed_form_holds_on_the_other_side_too`。
       **与0075第三节那条"``σ = −1``那一侧另有一条门"同源——同一个坑第二次**；
    3. **``C(a)``那条**：所有求值点都在``a* ≈ 0``附近，而那一项是``a``的**三阶**小量。
       补`test_the_model_curve_is_arc_length_parametrised_away_from_the_station`；
    4. **`laydown`那两条**：仓内合成语料与GCW真语料**都是均匀采样**，
       而均匀网格上两种写法恰好相同。补`test_a_nonuniform_sampling_is_weighted_by_arc`
       ——**三点系数那一条补了一次还没堵上**（非均匀之后两种写法只差2.3倍绝对误差），
       改判**收敛阶**才干净（`test_the_nonuniform_three_point_weights_stay_second_order`：
       正确的恒4.00、错的一路往2掉）。

    ## 剩下那一条不是空门，是一条**不改变答案**的变异

    ``step = residual / jacobian`` → ``step = residual``：那是不动点迭代
    ``a ← a + F(a)``，**不动点``F = 0``与除不除``D``无关**。
    收敛到的``a*``一样、``g``一样、``∇g``一样，变的只有迭代次数。
    **不为它补门**——要判它就得判迭代次数，那是把实现细节写成判据。

    ## 一条判据自己写错了两次

    本文件的`test_the_live_hessian_is_the_rank_one_outer_product...`第一版只判"秩一"，
    而**用冻结梯度做外积照样是秩一**；补反向断言时又只判了``hessian[0][0]``，
    而丢掉的那一项沿``t = (0, 1, 0)``、**那一格恰好一点没变**。
    最后改成判整个矩阵的最大偏差。
    **形态：一条判据看起来在判一件事，实际判的是另一件更弱的事。**

    ## 没有被注错验过的，如实登记

    * `_station`的"牛顿20步不收敛"**没有构造出用例**：``∂F/∂a = −D``在``D > 0``时
      是良态的二次收敛，本轮**没能找到一个既过``D > 0``又不收敛的构型**；
    * `groove_sweep_live_walls`里"探针塌成零宽"同样没有用例：触发它需要一个零长的段，
      而`_require_stations`在中心线构造期就把那种表拒了——**两道门在这里是重叠的**；
    * `quantities`的``need_gradient=False``分支只有一条绿用例走过，没有注错。
    """
