"""槽壁挡住扭转的门（plans/15第1.7条，决策0072）。

关的是0065第八节"**没有槽壁接触**"与第九节第4条"没有端到端案例证明扭多少"
那两条欠账的**单元级**那一半。

## 这一片凭什么算"`rod`接上了接触"

`PenaltyGrooveWall`是本仓第一个**同时索引位置块与γ块**的接触项。
在它之前接触把自由度硬分成两类：罚法向四族只按``3·node + axis``索引位置，
准静态步进器的落位校验明写"节点块之外是锚点槽"。γ两边都不属于，
于是`PenaltyAnnulusLimit`只能声明「无扭转假设」，并写着
「一旦有扭转，边缘点位置就错了``(w/2)·sin(扭角)``」——**那句话现在有实现了**。

## 四道判据

| # | 门 | 判什么 |
|---|---|---|
| 1 | **闭式挡多少** | 一条钳位的离散扭簧链有分段线性闭式，engine与它对拍 |
| 2 | **1/k标度** | 与闭式的差**是罚穿透**：``k``升十倍，差降十倍（实测四档整整齐齐） |
| 3 | **无壁退回`θ = TL/GJ`** | 槽张得很开时与"根本没有这一项"**逐位相同**，且θ＝0065那个金标 |
| 4 | 局部模板真的看见了γ | 梯度在位置块与γ块**同时**非零 |

## 判据一的闭式是怎么来的（推导写在这里，因为它是本文件的分辨力所在）

直杆上弯曲项恒零（0065门三的配套断言），扭转是一条**均匀的**离散扭簧链：
``τ_i = (GJ/l̄)(γ_{i+1} − γ_i)``。规定两端扭角、槽只覆盖中间一段时，
γ在无壁下严格线性（实测扭率极差``1.6e-17``）。

加上槽壁之后，**只有一个约束是活动的**：γ单调上升，而槽是对称的，
所以只有槽内**最扭的那个顶点**顶上壁。它把该顶点的带宽方向倾角钳在

    θ_c = arcsin(half_gap / half_width)

（直杆的Bishop帧逐边相同 ⟹ 平分线的倾角**精确**等于``(γ_v + γ_{v+1})/2``，
不是近似）。约束通过平分线**对称地**作用在``γ_v``与``γ_{v+1}``上，
于是扭矩平衡给出``s_M − s_L = s_R − s_M``（``s``是三段各自的斜率）。
连同两端条件解出

    γ_v = (4θ_c + (2θ_c − θ_end)/R) / (4 + 1/L + 1/R)

``L``是槽左侧的区间数、``R``是槽右侧的。**这条闭式不import被验内核的任何东西。**
"""

from __future__ import annotations

import math

import pytest

from physics_engine.energies import EnergyContext, EnergyRegistry
from physics_engine.rod import (
    AnisotropicRodBending,
    PenaltyGrooveWall,
    RodEndMoment,
    RodError,
    RodReference,
    RodTwist,
    build_bishop_frame,
    build_rod_layout,
)
from physics_engine.solve import solve_equilibrium

pytestmark = pytest.mark.batch

NODE_COUNT = 21
EDGE_COUNT = NODE_COUNT - 1
LENGTH_MM = 200.0
EI_EASY_NMM2 = 8.0e4
EI_HARD_NMM2 = 8.0e7
GJ_NMM2 = 77.0
#: 真实REBCO带材：4 mm宽 ⟹ 半宽2 mm（plans/14第2.3节）。
HALF_WIDTH_MM = 2.0
HALF_GAP_MM = 0.5
WALL_STIFFNESS_N_PER_MM = 1.0e4
END_TWIST_RAD = 0.9
#: 槽只覆盖中间七个内顶点，两端各留自由跨段——**被挡住的那部分扭转要有地方去**。
FIRST_GROOVED, LAST_GROOVED = 6, 12
GROOVED = tuple(range(FIRST_GROOVED, LAST_GROOVED + 1))
LOAD_STEPS = 20

#: 求解器容差**按γ块梯度的量级给**，不按位置块给。
#: 实测γ块梯度到``3e4 N·mm/rad``（罚刚度1e4×半宽2mm×…），
#: ``1e-6``对它是``3e-11``相对，已经贴着装配的舍入地板：
#: 同一批算例取``1e-8``时十档里两档在活动集翻转处走不动、取``1e-10``时八档走不动，
#: **而答案本身不随容差变**（三档的"挡住百分比"逐位一致到小数点后四位）。
#: **不收敛是停机判据太紧，不是模型的问题**——这条写出来，是因为
#: 反过来归因会去改罚刚度或载荷步，而那两个都改不动它。
WALLED_RESIDUAL_TOL_N = 1.0e-6


def _straight():
    """直杆＋Bishop帧。``d1 = ẑ``（穿厚）⟹ ``d2 = t × d1 = x̂ × ẑ = −ŷ``。

    于是``m2(γ) = (0, −cosγ, −sinγ)``：**γ一转，带材的边缘点就往``z``上走**，
    而槽壁正好立在``z = ±half_gap``上。这个摆法让"扭转被槽壁挡住"是一条
    可以手算的几何，而不是要靠画图才看得出的构型。
    """

    step = LENGTH_MM / (NODE_COUNT - 1)
    nodes = tuple((step * index, 0.0, 0.0) for index in range(NODE_COUNT))
    layout = build_rod_layout(layout_id="layout/rod-groove", node_count=NODE_COUNT)
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=(0.0, 0.0, 1.0))
    reference = RodReference(rest_lengths_mm=(step,) * (NODE_COUNT - 1), frame=frame)
    return layout, reference, nodes


def _faces(*, vertices, half_gap, stiffness=WALL_STIFFNESS_N_PER_MM, half_width=HALF_WIDTH_MM):
    """一条槽＝两片壁；带材两条边各声明一次 ⟹ 每个顶点**四条**face。

    逐条声明而不是塞一个``±``，理由与`PenaltyAnnulusLimit`的``inward``同源：
    **哪条边压哪片壁必须是读得出来的**。
    """

    entries = []
    for vertex in vertices:
        for offset in (+half_width, -half_width):
            entries.append((vertex, offset, (0.0, 0.0, +half_gap), (0.0, 0.0, -1.0), stiffness))
            entries.append((vertex, offset, (0.0, 0.0, -half_gap), (0.0, 0.0, +1.0), stiffness))
    return tuple(entries)


def _terms(layout, reference):
    return [
        AnisotropicRodBending(
            layout=layout, reference=reference,
            ei_easy_nmm2=(EI_EASY_NMM2,) * (NODE_COUNT - 2),
            ei_hard_nmm2=(EI_HARD_NMM2,) * (NODE_COUNT - 2),
        ),
        RodTwist(layout=layout, reference=reference, gj_nmm2=(GJ_NMM2,) * (NODE_COUNT - 2)),
    ]


def _context():
    return EnergyContext(
        context_id="context/rod-groove", node_masses_kg=(1.0,) * NODE_COUNT
    )


def _prescribed_twist(*, half_gap, stiffness=WALL_STIFFNESS_N_PER_MM, vertices=GROOVED):
    """两端规定扭角，端扭角按载荷步爬到`END_TWIST_RAD`。返回收敛解与槽壁项。

    载荷步不是装饰：罚势是``C¹``而不是``C²``（`PenaltyNormalContact`第四节），
    活动集在迭代中翻转时线搜索会失效。一步到位实测在多数gap上走不动。
    """

    layout, reference, nodes = _straight()
    terms = _terms(layout, reference)
    wall = None
    if half_gap is not None:
        wall = PenaltyGrooveWall(
            layout=layout, frame=reference.frame,
            faces=_faces(vertices=vertices, half_gap=half_gap, stiffness=stiffness),
        )
        terms.append(wall)
    registry = EnergyRegistry(terms=tuple(terms))
    last = layout.edge_count - 1
    fixed = layout.position_indices() | {layout.twist_index(0), layout.twist_index(last)}
    vector = layout.initial_state(
        positions_mm=nodes, edge_twist_angles=(0.0,) * layout.edge_count
    ).vector
    result = None
    for index in range(1, LOAD_STEPS + 1):
        current = list(vector)
        current[layout.twist_index(last)] = END_TWIST_RAD * index / LOAD_STEPS
        result = solve_equilibrium(
            registry, _context(), layout.layout, tuple(current),
            fixed_indices=fixed,
            residual_tol_n=1.0e-12 if half_gap is None else WALLED_RESIDUAL_TOL_N,
            max_iterations=120,
        )
        assert result.converged, f"载荷步{index}不收敛：{result.reason}"
        vector = result.state.vector
    assert result is not None
    return result, layout, wall


def _closed_form(half_gap: float) -> tuple[float, float, float]:
    """``(θ_c, γ_v, 槽段实际扭角)``——**不import被验内核的任何东西**，见模块docstring。"""

    theta_c = math.asin(half_gap / HALF_WIDTH_MM)
    left = LAST_GROOVED
    right = EDGE_COUNT - 2 - LAST_GROOVED
    gamma_v = (4.0 * theta_c + (2.0 * theta_c - END_TWIST_RAD) / right) / (
        4.0 + 1.0 / left + 1.0 / right
    )
    grooved = (2.0 * theta_c - gamma_v) - FIRST_GROOVED * (gamma_v / left)
    return theta_c, gamma_v, grooved


#: 无壁时槽段承担的扭角：γ严格线性，槽跨7个区间、全杆19个区间。
FREE_GROOVED_TWIST_RAD = END_TWIST_RAD * (LAST_GROOVED + 1 - FIRST_GROOVED) / (EDGE_COUNT - 1)


@pytest.fixture(scope="module")
def prescribed():
    """四档罚刚度＋五档槽宽各跑一次，六道门共用。"""

    stiffness_runs = {
        stiffness: _prescribed_twist(half_gap=HALF_GAP_MM, stiffness=stiffness)
        for stiffness in (1.0e3, 1.0e4, 1.0e5, 1.0e6)
    }
    gap_runs = {
        gap: (
            stiffness_runs[WALL_STIFFNESS_N_PER_MM]
            if gap == HALF_GAP_MM
            else _prescribed_twist(half_gap=gap)
        )
        for gap in (0.9, 0.7, 0.5, 0.3, 0.1)
    }
    return {
        "free": _prescribed_twist(half_gap=None),
        "stiffness": stiffness_runs,
        "gaps": gap_runs,
    }


def test_the_same_gap_and_stiffness_fixture_is_computed_once(prescribed):
    """0.5mm与1e4N/mm是两条扫描的交点，不许重跑同一个确定性求解。"""

    assert prescribed["gaps"][HALF_GAP_MM] is prescribed["stiffness"][WALL_STIFFNESS_N_PER_MM]


def _grooved_twist(result, layout) -> float:
    gammas = layout.twist_angles(result.state)
    return gammas[LAST_GROOVED + 1] - gammas[FIRST_GROOVED]


# ------------------------------------------------------- 门四：局部模板看见了γ


def test_the_local_template_sees_positions_and_edge_twist_angles_at_once() -> None:
    """**第三类自由度开了**：同一个接触项在位置块与γ块上都索引得到。

    对照两条既有事实：
    `RodTwist`在位置块上的梯度恒为零（0065第五节那条断言）、
    罚法向四族在γ块上一个下标都不索引（它们只按``3·node + axis``走）。
    **本项是第一个两边都够得着的**，而那正是"槽壁挡得住扭转"的前提——
    梯度在γ上为零的项，无论罚多硬都挡不住一个扭转自由度。

    ## 对称槽上位置梯度**恰好为零**，这是物理不是缺陷

    这道门起草时断言的是"位置块也非零"，**当场红了**，而红得对：
    对称的一条槽里，带材两条边一条压上壁、另一条压下壁，两个法向力
    大小相等方向相反——**净力恰好为零，槽壁给的是一个纯力偶**。
    所以这里判两条：对称槽上位置梯度**逐位为零**（零容差，那是力偶的定义），
    只声明一片壁时位置梯度**非零**（那才证明模板真的索引得到位置块）。
    """

    layout, reference, nodes = _straight()
    state = layout.initial_state(
        positions_mm=nodes,
        edge_twist_angles=tuple(0.05 * edge for edge in range(layout.edge_count)),
    )
    symmetric = PenaltyGrooveWall(
        layout=layout, frame=reference.frame,
        faces=_faces(vertices=(4, 5, 6), half_gap=0.2),
    )
    gradient = symmetric.gradient(state, _context())
    assert symmetric.energy(state, _context()) > 0.0, "没顶上壁，这道门什么都没验"
    assert all(
        gradient[index] == 0.0 for index in range(layout.twist_offset)
    ), "对称槽的净力必须**恰好**为零——它给的是一个纯力偶"
    twists = [gradient[index] for index in range(layout.twist_offset, len(gradient))]
    assert any(value != 0.0 for value in twists), (
        "**γ块全零**——那就是`PenaltyAnnulusLimit`的『无扭转假设』，本项白写了"
    )
    #: γ的耦合只出现在被声明的那三个顶点的**四条边**上（4,5 / 5,6 / 6,7）。
    live = {index for index, value in enumerate(twists) if value != 0.0}
    assert live == {4, 5, 6, 7}, live

    #: 只留下侧那一片壁：力偶不再平衡，位置块当场有内容。
    single = PenaltyGrooveWall(
        layout=layout, frame=reference.frame,
        faces=tuple(
            entry
            for entry in _faces(vertices=(4, 5, 6), half_gap=0.2)
            if entry[3] == (0.0, 0.0, 1.0)
        ),
    )
    lopsided = single.gradient(state, _context())
    assert any(
        lopsided[index] != 0.0 for index in range(layout.twist_offset)
    ), "**位置块全零**——局部模板根本没索引到位置那一类"
    assert {index for index in range(layout.twist_offset) if lopsided[index] != 0.0} == {
        3 * (vertex + 1) + 2 for vertex in (4, 5, 6)
    }, "力只该出现在被声明顶点的z分量上——壁的法向就是ẑ"


def test_the_gradient_and_hessian_match_central_differences() -> None:
    """归一化＋两次三角函数的二阶链式法则由`autodiff`的`Jet2`给，**要被独立验一次**。

    位置与γ都被扰动过（只验γ那一半会漏掉平分线对位置的那条耦合是常量这件事）。
    实测中心差分最大绝对差：梯度``9.0e-6``、Hessian``3.6e-6``，
    量级都是``h²·|U'''|``（``h = 1e-6``、能量本身``6e4``量级）。
    """

    node_count = 7
    step = 60.0 / (node_count - 1)
    nodes = tuple((step * index, 0.0, 0.0) for index in range(node_count))
    layout = build_rod_layout(layout_id="layout/rod-groove-fd", node_count=node_count)
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=(0.0, 0.0, 1.0))
    wall = PenaltyGrooveWall(
        layout=layout, frame=frame,
        faces=tuple(
            (vertex, offset, (0.0, 0.0, sign * 0.35), (0.0, 0.0, -sign), 1.0e4)
            for vertex in range(node_count - 2)
            for offset in (+HALF_WIDTH_MM, -HALF_WIDTH_MM)
            for sign in (1.0, -1.0)
        ),
    )
    context = EnergyContext(
        context_id="context/rod-groove-fd", node_masses_kg=(1.0,) * node_count
    )
    positions = tuple(
        (step * index, 0.013 * index, 0.021 * index) for index in range(node_count)
    )
    state = layout.initial_state(
        positions_mm=positions,
        edge_twist_angles=tuple(0.30 * edge for edge in range(layout.edge_count)),
    )
    energy, gradient, hessian = wall.quantities(
        state, context, need_gradient=True, need_hessian=True
    )
    assert energy > 0.0, "构型没顶上壁，这道门什么都没验"
    delta = 1.0e-6
    for index in range(len(state.vector)):
        plus, minus = list(state.vector), list(state.vector)
        plus[index] += delta
        minus[index] -= delta
        up = state.with_vector(tuple(plus))
        down = state.with_vector(tuple(minus))
        numeric = (wall.energy(up, context) - wall.energy(down, context)) / (2.0 * delta)
        assert abs(numeric - gradient[index]) <= 1.0e-4, index
        rows_up = wall.gradient(up, context)
        rows_down = wall.gradient(down, context)
        for column in range(len(state.vector)):
            second = (rows_up[column] - rows_down[column]) / (2.0 * delta)
            assert abs(second - hessian[index][column]) <= 1.0e-4, (index, column)


def test_the_fused_path_gives_the_same_energy_byte_for_byte() -> None:
    """spec/12第3.1节：融合路径的能量必须与单独调`energy`**逐字节**相同。

    本项靠共用`_RodEnergyTerm`那一个装配循环做到——**不是算完再比**。
    """

    layout, reference, nodes = _straight()
    wall = PenaltyGrooveWall(
        layout=layout, frame=reference.frame,
        faces=_faces(vertices=GROOVED, half_gap=0.2),
    )
    state = layout.initial_state(
        positions_mm=nodes,
        edge_twist_angles=tuple(0.07 * edge for edge in range(layout.edge_count)),
    )
    for need_gradient, need_hessian in ((False, False), (True, False), (True, True)):
        fused = wall.quantities(
            state, _context(), need_gradient=need_gradient, need_hessian=need_hessian
        )[0]
        assert fused.hex() == wall.energy(state, _context()).hex()


# ---------------------------------------------- 门三：无槽壁退回 θ = M·L_eff/GJ


def _end_moment(*, half_gap, moment_n_mm=0.5):
    layout, reference, nodes = _straight()
    last = layout.edge_count - 1
    terms = _terms(layout, reference)
    terms.append(RodEndMoment(layout=layout, edge=last, moment_n_mm=moment_n_mm))
    wall = None
    if half_gap is not None:
        wall = PenaltyGrooveWall(
            layout=layout, frame=reference.frame,
            faces=_faces(
                vertices=tuple(range(layout.interior_vertex_count)), half_gap=half_gap
            ),
        )
        terms.append(wall)
    state = layout.initial_state(
        positions_mm=nodes, edge_twist_angles=(0.0,) * layout.edge_count
    )
    result = solve_equilibrium(
        EnergyRegistry(terms=tuple(terms)), _context(), layout.layout, state.vector,
        fixed_indices=layout.position_indices() | {layout.twist_index(0)},
        residual_tol_n=1.0e-12, max_iterations=50,
    )
    assert result.converged, result.reason
    return result, layout, reference, wall, last


@pytest.mark.parametrize("half_gap", [1.99, 1.9])
def test_a_wide_open_groove_falls_back_to_the_free_torsion_closed_form(half_gap) -> None:
    """**门三**：槽张得开时``θ = M·L_eff/GJ``，且与"根本没有这一项"**逐位相同**。

    这里判两件事，第二件比第一件强：

    1. ``θ = 1.2337662337662278``、闭式``1.2337662337662338``、相对偏差**4.885e-15**——
       **与0065第二节门三报的是同一串数字**，一位不差。接线没有动既有物理；
    2. 整条状态向量与不含本项的那一次求解**逐位相同**。
       "很接近"会放过一个漏了互补条件（``g > 0 ⟹ f ≡ 0``）的实现，
       而那种实现会给出一个小而系统的偏置。**互补在这里是零容差判据。**

    ``half_gap = 1.9``时最小间隙只有``0.0350 mm``——**它贴着壁而没碰上**，
    所以这不是"离得远所以碰不到"的平凡通过。
    """

    bare, layout_bare, reference, _, last = _end_moment(half_gap=None)
    walled, layout, _, wall, _ = _end_moment(half_gap=half_gap)
    theta = walled.state.vector[layout.twist_index(last)]
    closed = 0.5 * sum(reference.dual_lengths_mm) / GJ_NMM2
    assert theta == 1.2337662337662278
    assert abs(theta / closed - 1.0) <= 1.0e-14
    assert [value.hex() for value in walled.state.vector] == [
        value.hex() for value in bare.state.vector
    ]
    assert max(wall.wall_force_n(walled.state)) == 0.0
    assert min(wall.gaps_mm(walled.state)) > 0.0


def test_a_groove_that_just_touches_is_not_inert() -> None:
    """**判据本身被验**：上一条的"逐位相同"不是因为这一项做不动任何事。

    把半间隙收到``1.5 mm``（最小间隙``−6.0e-6 mm``，**刚碰上**），
    ``θ``从1.2337掉到0.8757——差**29.02%**。
    活动集的边界因此是锐的，而不是"反正都碰不到"。
    """

    walled, layout, reference, wall, last = _end_moment(half_gap=1.5)
    theta = walled.state.vector[layout.twist_index(last)]
    closed = 0.5 * sum(reference.dual_lengths_mm) / GJ_NMM2
    assert theta == pytest.approx(0.8756950701146152, rel=1.0e-9)
    assert abs(theta / closed - 1.0) == pytest.approx(0.29023, rel=1.0e-3)
    assert min(wall.gaps_mm(walled.state)) < 0.0
    assert max(wall.wall_force_n(walled.state)) > 0.0


# --------------------------------------------------- 门一与门二：挡住多少、扭多少


@pytest.mark.batch
def test_a_prescribed_frame_twist_is_blocked_by_the_closed_form_amount(prescribed) -> None:
    """**门一**：给一段带材一个强制的帧扭转，**槽壁挡住多少、实际扭多少**。

    2026-08-17实测（21节点、带宽4 mm、半间隙0.5 mm、``k = 1e4 N/mm``、
    两端规定扭角0.9 rad、槽覆盖内顶点6—12）：

    | 量 | 值 |
    |---|---|
    | 无槽壁时槽段承担 | 0.331578947368421 rad |
    | **有槽壁时实际扭** | **0.171853604667 rad** |
    | **槽壁挡住** | **0.159725342701 rad（48.171%）** |
    | 闭式给的实际扭 | 0.171849531832 rad（差4.073e-06，见门二） |
    | 壁力 | 0.167599 N（**两条face活动**，都在顶点12上） |
    | 钳住的带宽方向倾角 | 0.252680 rad ＝ ``arcsin(0.5/2.0)`` |

    **只有一条约束活动**是这个构型的性质不是巧合：γ单调上升、槽左右对称，
    于是只有槽内最扭的那个顶点顶上壁。判它是为了让闭式的适用条件
    **在门里被检查**，而不是写在注释里。

    **必红方式**：把`PenaltyGrooveWall._width_direction`里的``m̂2``换成常量
    ``frame.d2[vertex]``（＝把γ从采样点里拿掉，回到`PenaltyAnnulusLimit`的
    "无扭转假设"）——槽壁再也挡不住任何东西，槽段扭角回到0.3316，本用例当场红。
    """

    free_result, free_layout, _ = prescribed["free"]
    assert _grooved_twist(free_result, free_layout) == pytest.approx(
        FREE_GROOVED_TWIST_RAD, rel=1.0e-14
    )

    result, layout, wall = prescribed["stiffness"][1.0e4]
    actual = _grooved_twist(result, layout)
    assert actual == pytest.approx(0.17185360466736044, rel=1.0e-9)
    blocked = FREE_GROOVED_TWIST_RAD - actual
    assert blocked == pytest.approx(0.15972534270106059, rel=1.0e-8)
    assert blocked / FREE_GROOVED_TWIST_RAD == pytest.approx(0.481711, rel=1.0e-5)

    gaps = wall.gaps_mm(result.state)
    active = [index for index, value in enumerate(gaps) if value < 0.0]
    assert len(active) == 2, (
        f"活动face{len(active)}条——闭式只对**一条约束活动**成立，"
        "多于一条时门一那串数字不再是这个构型的答案"
    )
    forces = wall.wall_force_n(result.state)
    assert max(forces) == pytest.approx(0.167599, rel=1.0e-4)

    theta_c, _, _ = _closed_form(HALF_GAP_MM)
    assert theta_c == pytest.approx(0.25268025514207865, rel=1.0e-15)
    #: 直杆Bishop帧逐边相同 ⟹ 平分线倾角**精确**等于``(γ_v + γ_{v+1})/2``。
    gammas = layout.twist_angles(result.state)
    clamped = 0.5 * (gammas[LAST_GROOVED] + gammas[LAST_GROOVED + 1])
    assert clamped - theta_c == pytest.approx(8.654776063377057e-06, rel=1.0e-6)
    tilt = [
        math.asin(max(-1.0, min(1.0, -direction[2])))
        for direction in wall.width_direction(result.state)
    ]
    assert max(tilt) == pytest.approx(clamped, abs=1.0e-15)


@pytest.mark.batch
def test_the_gap_to_the_closed_form_is_the_penalty_penetration(prescribed) -> None:
    """**门二**：与闭式的差**是罚穿透**——``k``升十倍，差降十倍。

    实测四档，``half_gap = 0.5``：

    | ``k`` N/mm | 槽段实际扭 rad | 与闭式的差 |
    |---:|---|---:|
    | 1e3 | 0.171890252064 | 4.0720e-05 |
    | 1e4 | 0.171853604667 | 4.0728e-06 |
    | 1e5 | 0.171849939123 | 4.0729e-07 |
    | 1e6 | 0.171849572561 | 4.0729e-08 |

    **每档整整齐齐差十倍**，且四档的壁力收敛到同一个数（0.167564 → 0.167603 N）。
    这一条把"engine与闭式差4e-6"从"精度不明的接近"变成"``δ = N/k``这条已知的模型代价"——
    与`PenaltyNormalContact`那句"位置有``O(1/k)``的误差，力没有误差"是同一件事。

    **必红方式**：把间隙里的``offset·m̂2``写成``offset·m̂2/2``（半宽记错一半），
    四档的差全部跑掉，而且**不再是1/k标度**——这道门比门一更早抓到量纲类的错。
    """

    _, _, closed = _closed_form(HALF_GAP_MM)
    assert closed == pytest.approx(0.17184953183156637, rel=1.0e-15)
    previous = None
    for stiffness in (1.0e3, 1.0e4, 1.0e5, 1.0e6):
        result, layout, _ = prescribed["stiffness"][stiffness]
        difference = _grooved_twist(result, layout) - closed
        assert difference > 0.0, "穿透只会让带材比闭式**多**扭一点"
        assert difference == pytest.approx(4.0729e-2 / stiffness, rel=5.0e-4)
        if previous is not None:
            assert previous / difference == pytest.approx(10.0, rel=1.0e-3)
        previous = difference


@pytest.mark.batch
def test_a_narrower_groove_blocks_more_and_tracks_the_closed_form(prescribed) -> None:
    """五档槽宽逐档对闭式，**并且挡住的比例单调**。

    实测（``k = 1e4``）：0.9→17.788%、0.7→33.285%、0.5→48.171%、
    0.3→62.663%、0.1→76.933%，与闭式的差全部在``1.8e-6``—``6.2e-6``之间
    （＝各自的``N/k``穿透）。

    **一档对上闭式可能是凑的，五档跨一个数量级的槽宽同时对上不是。**
    """

    previous_fraction = 0.0
    for gap in (0.9, 0.7, 0.5, 0.3, 0.1):
        result, layout, _ = prescribed["gaps"][gap]
        actual = _grooved_twist(result, layout)
        _, _, closed = _closed_form(gap)
        assert actual - closed == pytest.approx(0.0, abs=7.0e-6)
        assert actual > closed
        fraction = 1.0 - actual / FREE_GROOVED_TWIST_RAD
        assert fraction > previous_fraction
        previous_fraction = fraction
    assert previous_fraction == pytest.approx(0.76933, rel=1.0e-4)


# ------------------------------------------------------------------ 失败关闭


def test_a_zero_edge_offset_fails_closed() -> None:
    """偏移为零＝采样点退回中心线＝**本项对γ的依赖当场消失**，而它仍会安静算出法向接触。"""

    layout, reference, _ = _straight()
    with pytest.raises(RodError, match="边缘偏移是零"):
        PenaltyGrooveWall(
            layout=layout, frame=reference.frame,
            faces=((3, 0.0, (0.0, 0.0, 0.5), (0.0, 0.0, -1.0), 1.0e4),),
        )


def test_a_non_unit_wall_normal_fails_closed() -> None:
    """不归一化等于把刚度悄悄乘上``|n|²``，而调用方以为自己给的是``k``。"""

    layout, reference, _ = _straight()
    with pytest.raises(RodError, match="不是单位矢量"):
        PenaltyGrooveWall(
            layout=layout, frame=reference.frame,
            faces=((3, 2.0, (0.0, 0.0, 0.5), (0.0, 0.0, -1.5), 1.0e4),),
        )


@pytest.mark.parametrize("vertex", [-1, 19, 100])
def test_a_vertex_outside_the_interior_range_fails_closed(vertex) -> None:
    """端顶点只有一条边，带宽方向的平分线在那里没有定义。"""

    layout, reference, _ = _straight()
    with pytest.raises(RodError, match="落在内顶点之外"):
        PenaltyGrooveWall(
            layout=layout, frame=reference.frame,
            faces=((vertex, 2.0, (0.0, 0.0, 0.5), (0.0, 0.0, -1.0), 1.0e4),),
        )


def test_an_empty_face_list_fails_closed() -> None:
    """声明了一条槽却不给任何一片壁，等于声明没发生。"""

    layout, reference, _ = _straight()
    with pytest.raises(RodError, match="at least one face"):
        PenaltyGrooveWall(layout=layout, frame=reference.frame, faces=())


def test_a_frame_that_disagrees_with_the_layout_fails_closed() -> None:
    """帧与布局的边数对不上时，``frame.d1[vertex + 1]``会静默读到别人的帧。"""

    layout, reference, _ = _straight()
    shorter = build_bishop_frame(
        positions_mm=tuple((10.0 * index, 0.0, 0.0) for index in range(5)),
        seed_d1=(0.0, 0.0, 1.0),
    )
    with pytest.raises(RodError, match="edge count"):
        PenaltyGrooveWall(
            layout=layout, frame=shorter,
            faces=((3, 2.0, (0.0, 0.0, 0.5), (0.0, 0.0, -1.0), 1.0e4),),
        )


def test_two_edges_whose_material_frames_are_antiparallel_fail_closed() -> None:
    """相邻边扭角差接近π时平分线由舍入决定——**拿噪声当带宽方向用比报错坏得多**。

    与`contact.friction.IN_PLANE_DIRECTION_MIN_SINE`同一条纪律。
    """

    layout, reference, nodes = _straight()
    wall = PenaltyGrooveWall(
        layout=layout, frame=reference.frame,
        faces=((3, 2.0, (0.0, 0.0, 0.5), (0.0, 0.0, -1.0), 1.0e4),),
    )
    gammas = [0.0] * layout.edge_count
    gammas[4] = math.pi
    state = layout.initial_state(positions_mm=nodes, edge_twist_angles=tuple(gammas))
    with pytest.raises(RodError, match="几乎反向"):
        wall.energy(state, _context())
