"""整杆各向异性弯曲与扭转的门（决策0065）。

本文件先于实现落地并实跑为红：首次收集因
``ModuleNotFoundError: physics_engine.rod``失败。

它守的东西按重要性排：

1. **三道物理门**（决策0064第4.4节点名，后两道同行WDS没有）——
   螺旋线运动学闭式（且``κ2``在螺旋线上必须是机器零）、
   易/难轴互换（参考``d1``转90°，挠度比**必须等于**``EI_hard/EI_easy``）、
   闭式扭转``θ = M·L/GJ``（**全仓此前没有任何扭转金标**）；
2. **"抄了公式不抄外循环就红"那道门**（0064第4.3节第1条点名，
   "**这是本轮最容易被漏掉的一条**"）——球面三角holonomy算例，
   不重输运时扭转能量**恒等于零**；
3. **``EI_easy``配``κ1``的运行时门**——同行那边接反只会给出一个1600倍偏小的
   挠度而不报错；
4. **``natural_kappa2``是活通道**——同行有这个槽但生产里恒为零，它从没跑过。

**测试债不继承**：WDS没有任何测试验证各向异性弯曲的行为，其验证协议因为没有
oracle而明确拒绝建各向异性对比算例。本文件的三道门全部自带独立闭式。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.energies import (
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    PointLoad,
)
from physics_engine.rod import (
    AnisotropicRodBending,
    RodEndMoment,
    RodError,
    RodModel,
    RodReference,
    RodSolveStage,
    RodTwist,
    build_bishop_frame,
    build_material_frame,
    build_rod_layout,
    edge_tangents,
    gammas_from_material_directors,
    parallel_transport,
    signed_angle,
    solve_rod_with_retransport,
    unwrap_phases,
)
from physics_engine.solve import solve_equilibrium

#: 带材的两个真实刚度量级（plans/14第2.3节：4mm宽×0.1mm厚、E=100 GPa）。
#: ``EI_hard = E·t·w³/12 = 8.0e4 N·mm²``；这里的``EI_easy``取1/1000，
#: 是为了让易/难轴门的比值一眼可读，不是那条带材的真实比（真实是1/1600）。
EI_EASY_NMM2 = 8.0e4
EI_HARD_NMM2 = 8.0e7
GJ_NMM2 = 77.0


def _context(node_count: int) -> EnergyContext:
    return EnergyContext(
        context_id="context/rod-test", node_masses_kg=(1.0,) * node_count
    )


def _straight(node_count: int, *, length: float, seed_d1, clamped: bool = False):
    """直杆＋Bishop帧。``clamped``给固支顶点还回半格柔度（见`RodReference`那条注释）。"""

    step = length / (node_count - 1)
    nodes = tuple((step * index, 0.0, 0.0) for index in range(node_count))
    layout = build_rod_layout(layout_id="layout/rod-test", node_count=node_count)
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=seed_d1)
    dual = (
        tuple((1.5 if vertex == 0 else 1.0) * step for vertex in range(node_count - 2))
        if clamped
        else None
    )
    reference = RodReference(
        rest_lengths_mm=(step,) * (node_count - 1), frame=frame, dual_lengths_mm=dual
    )
    return layout, reference, nodes, step


def _helix(node_count: int, *, radius: float, pitch: float, sweep: float):
    """离散螺旋线，外加**逐边中点**的解析主法线——``m1``就取它。

    取中点而不是端点不是随手：螺旋线的两条相邻弦的``κb``与顶点处的解析副法线
    严格共线，而两个中点法线之和严格平行于顶点法线，于是
    ``κ2 = −0.5(m1_l + m1_r)·κb``**恰好为零**（不是趋于零）。
    """

    step = sweep / (node_count - 1)
    nodes = tuple(
        (
            radius * math.cos(index * step),
            radius * math.sin(index * step),
            pitch * index * step,
        )
        for index in range(node_count)
    )
    normals = tuple(
        (-math.cos((index + 0.5) * step), -math.sin((index + 0.5) * step), 0.0)
        for index in range(node_count - 1)
    )
    return nodes, normals


def _helix_model(node_count: int, *, radius: float, pitch: float, sweep: float):
    nodes, normals = _helix(node_count, radius=radius, pitch=pitch, sweep=sweep)
    layout = build_rod_layout(layout_id="layout/rod-helix", node_count=node_count)
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=normals[0])
    rest = tuple(math.dist(nodes[i], nodes[i + 1]) for i in range(node_count - 1))
    reference = RodReference(rest_lengths_mm=rest, frame=frame)
    gammas = gammas_from_material_directors(frame=frame, m1=normals)
    model = RodModel(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
        gj_nmm2=(GJ_NMM2,) * (node_count - 2),
    )
    state = layout.initial_state(positions_mm=nodes, edge_twist_angles=gammas)
    return model, state


# --------------------------------------------------------------- 平行输运与解缠


def test_parallel_transport_preserves_length_and_the_angle_to_the_tangent() -> None:
    """输运是转动：长度不变、与切向的夹角不变。**这是整条链的地基。**"""

    source = (1.0, 0.0, 0.0)
    target = (0.3, 0.8, -0.5)
    length = math.sqrt(sum(value * value for value in target))
    target = tuple(value / length for value in target)
    for vector in ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.2, -0.7, 0.4)):
        moved = parallel_transport(vector, source, target)
        before = math.sqrt(sum(value * value for value in vector))
        after = math.sqrt(sum(value * value for value in moved))
        assert abs(after - before) <= 1.0e-15 * before
        dot_before = sum(a * b for a, b in zip(vector, source, strict=True))
        dot_after = sum(a * b for a, b in zip(moved, target, strict=True))
        assert abs(dot_after - dot_before) <= 1.0e-15 * before


def test_parallel_transport_between_antiparallel_tangents_fails_closed() -> None:
    """转轴不唯一时**失败关闭**，不返回一个"差不多"的向量。"""

    with pytest.raises(RodError, match="antiparallel"):
        parallel_transport((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))


def test_unwrap_removes_the_two_pi_jumps_it_is_there_for() -> None:
    """解缠把相邻差压回一个周期内。

    **必红方式**：把`rod.unwrap_phases`改成``return tuple(raw)``（直通），
    本用例第二条断言当场红——那正是"整匝657°的扭转在γ序列里跳一圈"的形态。
    """

    #: 一条单调增到超过一圈的相位，被`atan2`折回``(−π, π]``之后的样子。
    truth = tuple(0.35 * index for index in range(24))
    wrapped = tuple(
        math.atan2(math.sin(value), math.cos(value)) for value in truth
    )
    assert max(abs(a - b) for a, b in zip(wrapped, truth, strict=True)) > 3.0
    recovered = unwrap_phases(wrapped)
    assert max(abs(a - b) for a, b in zip(recovered, truth, strict=True)) <= 1.0e-12
    for index in range(1, len(recovered)):
        assert abs(recovered[index] - recovered[index - 1]) <= math.pi


def test_unwrap_leaves_a_sequence_that_never_jumps_untouched() -> None:
    """**判据本身被验**：没有跳变时解缠不许动数——否则它会掩盖真实的大扭转。"""

    values = (0.1, 0.2, 0.15, -0.3, 0.0)
    assert unwrap_phases(values) == values


# ----------------------------------------------------------------------- 帧


def test_the_bishop_frame_reference_twist_is_measured_and_comes_out_machine_zero() -> None:
    """沿链平行输运出的帧，``m_ref``必须是机器零——**而它是量出来的，不是假定的**。

    这条把"``m_ref``是不是一个真通道"和"Bishop帧对不对"分开：下一条门给一个
    **非**Bishop帧，同一个字段必须给出非零值。
    """

    _, reference, _, _ = _straight(9, length=90.0, seed_d1=(0.0, 1.0, 0.0))
    assert max(abs(value) for value in reference.frame.reference_twist) <= 1.0e-15
    nodes, normals = _helix(41, radius=40.0, pitch=12.0, sweep=2.0)
    curved = build_bishop_frame(positions_mm=nodes, seed_d1=normals[0])
    assert max(abs(value) for value in curved.reference_twist) <= 1.0e-15


def test_a_non_bishop_reference_frame_carries_a_real_m_ref() -> None:
    """``m_ref``不是恒零的装饰位：给一个逐边多转``δ``的帧，它就必须报出``δ``。"""

    nodes = tuple((10.0 * index, 0.0, 0.0) for index in range(6))
    tangents = edge_tangents(nodes)
    delta = 0.21
    directors = tuple(
        (0.0, math.cos(delta * index), math.sin(delta * index)) for index in range(5)
    )
    frame = build_material_frame(tangents=tangents, edge_d1=directors)
    for value in frame.reference_twist:
        assert abs(value - delta) <= 1.0e-14


def test_a_frame_that_is_not_orthonormal_right_handed_fails_closed() -> None:
    """``d2 = t × d1``是形制不是建议：左手系、非单位、不正交，三种都拒收。"""

    from physics_engine.rod import RodMaterialFrame

    tangent = (1.0, 0.0, 0.0)
    with pytest.raises(RodError, match="orthonormal right-handed"):
        RodMaterialFrame((tangent,), ((0.0, 1.0, 0.0),), ((0.0, 0.0, -1.0),), ())
    with pytest.raises(RodError, match="orthonormal right-handed"):
        RodMaterialFrame((tangent,), ((0.0, 2.0, 0.0),), ((0.0, 0.0, 2.0),), ())
    with pytest.raises(RodError, match="orthonormal right-handed"):
        RodMaterialFrame((tangent,), ((1.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),), ())


# ------------------------------------------------------------ 门一：螺旋线运动学


def test_helix_kinematics_matches_the_closed_form_and_kappa2_is_machine_zero() -> None:
    """**物理门一**：``κ = R/(R²+p²)``、``τ = p/(R²+p²)``，且``κ2``必须是机器零。

    闭式与被验内核独立。``κ2``那一条是本门的锋刃：材料帧的``m1``取解析主法线时，
    螺旋线上**没有任何hard-way弯曲**——不是"很小"，是恰好为零。
    实测``|κ2|``在N=21/41/81/161上分别是1.07e-15/1.92e-15/5.20e-15/8.83e-15，
    **不随h下降**，正说明它是机器零而不是一个收敛量。

    **必红方式**：把`rod.AnisotropicRodBending._curvatures`里``m1_sum``的
    ``cos_left``换成``sin_left``（等价于把``d1``取错），``κ2``当场从1e-15
    跳到1e-1量级。
    """

    radius, pitch = 40.0, 12.0
    kappa = radius / (radius**2 + pitch**2)
    tau = pitch / (radius**2 + pitch**2)
    model, state = _helix_model(41, radius=radius, pitch=pitch, sweep=2.0)
    dual = model.reference.dual_lengths_mm
    curvatures = model.bending().curvatures(state)
    assert max(abs(pair[1]) for pair in curvatures) <= 1.0e-13
    physical = [pair[0] / dual[index] for index, pair in enumerate(curvatures)]
    assert max(abs(value / kappa - 1.0) for value in physical) <= 3.0e-4
    rates = model.twist().twist_rates_per_mm(state)
    assert max(abs(value / tau - 1.0) for value in rates) <= 5.0e-4


def test_the_helix_curvature_and_torsion_converge_second_order() -> None:
    """离散化误差是二阶的——**收敛阶是判据，单点偏差不是**。

    实测（N=21/41/81/161）：``κ``偏差9.753e-4/2.437e-4/6.093e-5/1.523e-5，
    ``τ``偏差1.531e-3/3.824e-4/9.557e-5/2.389e-5，两列比值都是4.00。
    """

    radius, pitch = 40.0, 12.0
    kappa = radius / (radius**2 + pitch**2)
    tau = pitch / (radius**2 + pitch**2)
    curvature_errors = []
    torsion_errors = []
    for node_count in (21, 41, 81, 161):
        model, state = _helix_model(node_count, radius=radius, pitch=pitch, sweep=2.0)
        dual = model.reference.dual_lengths_mm
        curvatures = model.bending().curvatures(state)
        curvature_errors.append(
            max(
                abs(pair[0] / dual[index] / kappa - 1.0)
                for index, pair in enumerate(curvatures)
            )
        )
        torsion_errors.append(
            max(abs(value / tau - 1.0) for value in model.twist().twist_rates_per_mm(state))
        )
    for errors in (curvature_errors, torsion_errors):
        ratios = [
            errors[index - 1] / errors[index] for index in range(1, len(errors))
        ]
        assert min(ratios) > 3.8, f"收敛比掉到{ratios}——不再是二阶"
        assert max(ratios) < 4.2, f"收敛比{ratios}——比二阶还快，多半是判据写错了"


# ------------------------------------------------------- 门二：易/难轴互换


def _cantilever_tip_deflection(*, node_count: int, seed_d1, force_n: float, length: float):
    layout, reference, nodes, step = _straight(
        node_count, length=length, seed_d1=seed_d1, clamped=True
    )
    bending = AnisotropicRodBending(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
    )
    stretch = AxialStretch(
        edges=tuple((index, index + 1, step, 4.0e4) for index in range(node_count - 1))
    )
    load = PointLoad(loads=((node_count - 1, (0.0, force_n, 0.0)),))
    registry = EnergyRegistry(terms=(bending, stretch, load))
    state = layout.initial_state(
        positions_mm=nodes, edge_twist_angles=(0.0,) * (node_count - 1)
    )
    #: 前两个节点钉死＝固支（第一条边被钉住），γ全钉住＝把扭转通道关掉，
    #: 这样量到的就是纯弯曲的各向异性，不掺"杆自己转过去用软轴"那件事。
    fixed = frozenset(range(6)) | layout.twist_indices()
    #: 容差取1e-9 N而不是`solve.py`推荐的"总载荷的1e-9到1e-10"（=5e-11）：
    #: **实测本问题的绝对残差地板是2.5e-10 N**，比那条参考线还高，
    #: 取5e-11时60次迭代、399次回溯仍不收敛。这正是`solve.py`docstring里
    #: "绝对残差的可达地板随问题规模上升"那一条，不是本模块的毛病。
    result = solve_equilibrium(
        registry,
        _context(node_count),
        layout.layout,
        state.vector,
        fixed_indices=fixed,
        residual_tol_n=1.0e-9,
        max_iterations=40,
    )
    assert result.converged, result.reason
    return result.state.vector[3 * (node_count - 1) + 1]


def test_rotating_the_reference_d1_by_ninety_degrees_swaps_easy_and_hard() -> None:
    """**物理门二**：参考``d1``转90°，挠度比**必须等于**``EI_hard/EI_easy``。

    同行自己在`test_gravity_cantilever.py`的docstring里点名了这个失效模式
    （参考``d1``取错会让``EI_hard``接管、挠度差1600倍、**不报任何错**），
    **而那个失效模式在他们那边没有门守着**。这就是那道门。

    ``d1 = ŷ``＝穿厚方向沿y，载荷沿y ⟹ 朝``m1``弯 ⟹ ``κ1`` ⟹ ``EI_easy``；
    ``d1 = ẑ``＝穿厚方向沿z ⟹ 同一个载荷变成朝``m2``弯 ⟹ ``κ2`` ⟹ ``EI_hard``。

    实测比值999.99972，相对偏差**2.79e-7**，且它随载荷平方下降
    （F=0.05/0.005/0.0005 N时是2.79e-7/2.79e-9/2.65e-11）——
    **那是几何非线性，不是这道门的噪声**。判据取1e-6。

    **必红方式**：把`rod.AnisotropicRodBending._vertex_energy`里的
    ``ei_easy_nmm2``与``ei_hard_nmm2``对调，比值当场变成1/1000。
    """

    easy = _cantilever_tip_deflection(
        node_count=21, seed_d1=(0.0, 1.0, 0.0), force_n=-0.05, length=50.0
    )
    hard = _cantilever_tip_deflection(
        node_count=21, seed_d1=(0.0, 0.0, 1.0), force_n=-0.05, length=50.0
    )
    ratio = easy / hard
    assert abs(ratio / (EI_HARD_NMM2 / EI_EASY_NMM2) - 1.0) <= 1.0e-6


def test_the_easy_axis_cantilever_converges_to_the_closed_form_deflection() -> None:
    """挠度本身也要对：``δ = F·L³/(3·EI_easy)``，二阶收敛。

    实测（N=11/21/41/81，含固支顶点``3h/2``订正）：相对偏差
    2.350e-2/6.063e-3/1.539e-3/3.880e-4，比值3.876/3.939/3.968。
    **不加那条订正就只有一阶**（0.145/0.0738/0.0372/0.0187，比值≈2）——
    这与`energies.clamped_chain_bending_vertices`推的是同一件事，
    见`RodReference.dual_lengths_mm`那条注释。
    """

    length, force = 50.0, -0.05
    closed = force * length**3 / (3.0 * EI_EASY_NMM2)
    errors = []
    for node_count in (11, 21, 41, 81):
        tip = _cantilever_tip_deflection(
            node_count=node_count, seed_d1=(0.0, 1.0, 0.0), force_n=force, length=length
        )
        errors.append(abs(tip / closed - 1.0))
    ratios = [errors[index - 1] / errors[index] for index in range(1, len(errors))]
    assert min(ratios) > 3.7, f"收敛比{ratios}——固支半格订正没起作用？"
    assert errors[-1] <= 5.0e-4


# ------------------------------------------------------------ 门三：闭式扭转


def test_an_end_torque_reproduces_theta_equals_moment_times_length_over_gj() -> None:
    """**物理门三**：``θ = M·L/GJ``。**全仓此前没有任何扭转金标。**

    ``L``取"第一条边中点到最后一条边中点"的距离``Σ l̄_i``——离散模型里扭转弹簧
    住在内顶点上，端边之间正好跨这么多。等分时它是``(边数−1)·h``，
    实测N=21、h=10时``L_eff = 190``而杆长``L = 200``：**那10mm的差是形制不是误差**，
    写出来是因为拿``L = 200``去对会得到5%的"偏差"然后被人当成精度问题去调。

    这个方程在离散模型里是**线性**的，所以偏差应当在机器精度：
    实测``θ = 1.2337662337662278``、闭式``1.2337662337662338``，
    相对偏差**4.885e-15**，牛顿1步收敛，逐顶点扭率的极差6.85e-17（严格均匀）。

    **必红方式**：把`rod.RodTwist._vertex_energy`里的``/ dual_lengths_mm[vertex]``
    删掉，``θ``当场差``l̄``倍（本例10倍）。
    """

    node_count, length, moment = 21, 200.0, 0.5
    layout, reference, nodes, _ = _straight(
        node_count, length=length, seed_d1=(0.0, 1.0, 0.0)
    )
    bending = AnisotropicRodBending(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
    )
    twist = RodTwist(
        layout=layout, reference=reference, gj_nmm2=(GJ_NMM2,) * (node_count - 2)
    )
    last_edge = layout.edge_count - 1
    load = RodEndMoment(layout=layout, edge=last_edge, moment_n_mm=moment)
    registry = EnergyRegistry(terms=(bending, twist, load))
    state = layout.initial_state(
        positions_mm=nodes, edge_twist_angles=(0.0,) * layout.edge_count
    )
    #: 悬臂扭转BC＝钉住``3N+0``（第一条边的γ）。位置全钉住，只留扭转通道。
    fixed = layout.position_indices() | {layout.twist_index(0)}
    result = solve_equilibrium(
        registry,
        _context(node_count),
        layout.layout,
        state.vector,
        fixed_indices=fixed,
        residual_tol_n=1.0e-12,
        max_iterations=20,
    )
    assert result.converged, result.reason
    effective_length = sum(reference.dual_lengths_mm)
    assert effective_length == pytest.approx(190.0, abs=1.0e-12)
    theta = result.state.vector[layout.twist_index(last_edge)]
    assert abs(theta / (moment * effective_length / GJ_NMM2) - 1.0) <= 1.0e-13
    rates = twist.twist_rates_per_mm(result.state)
    assert max(rates) - min(rates) <= 1.0e-15


def test_a_straight_rod_carries_no_bending_energy_so_the_torsion_gate_is_clean() -> None:
    """**判据本身被验**：门三里那个弯曲项在直杆上必须是恒零，否则它在污染``θ``。"""

    node_count = 21
    layout, reference, nodes, _ = _straight(
        node_count, length=200.0, seed_d1=(0.0, 1.0, 0.0)
    )
    bending = AnisotropicRodBending(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
    )
    state = layout.initial_state(
        positions_mm=nodes,
        edge_twist_angles=tuple(0.1 * index for index in range(layout.edge_count)),
    )
    context = _context(node_count)
    assert bending.energy(state, context) == 0.0
    assert max(abs(value) for value in bending.gradient(state, context)) == 0.0


# ------------------------------- 门四：不抄retransport外层循环就红（球面三角holonomy）


def _spherical_triangle(step: float):
    """切向序列``x̂ → ŷ → ẑ → x̂``：球面上三个直角的测地三角形，面积``4π/8 = π/2``。

    由Gauss-Bonnet，沿这条闭合切向路径平行输运一周的holonomy**恰是π/2**——
    这是一条与被验内核完全独立的闭式，也是本门的oracle。
    """

    return (
        (0.0, 0.0, 0.0),
        (step, 0.0, 0.0),
        (step, step, 0.0),
        (step, step, step),
        (2.0 * step, step, step),
    )


def _holonomy_model(step: float, *, ei: float):
    node_count = 5
    straight = tuple((step * index, 0.0, 0.0) for index in range(node_count))
    layout = build_rod_layout(layout_id="layout/rod-holonomy", node_count=node_count)
    frame = build_bishop_frame(positions_mm=straight, seed_d1=(0.0, 1.0, 0.0))
    reference = RodReference(rest_lengths_mm=(step,) * 4, frame=frame)
    model = RodModel(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(ei,) * 3,
        ei_hard_nmm2=(ei,) * 3,
        gj_nmm2=(GJ_NMM2,) * 3,
    )
    return model, straight


def test_the_spherical_triangle_holonomy_appears_only_after_retransport() -> None:
    """**外层循环门（运动学半）**：holonomy必须恰好是``π/2``，不重输运时是**零**。

    杆从直（切向全是``x̂``）走到球面三角构型（切向``x̂ → ŷ → ẑ → x̂``）。
    材料帧被保住不变，而**新构型的Bishop帧带着π/2的holonomy**，
    于是``γ``必须把它吃下去——这就是"弯曲转成扭转"。

    实测：重输运后``γ = (0, 0, −π/2, −π/2)``，总扭转``γ₃ − γ₀``与``−π/2``
    **逐位相同**（差0.0），``m_ref``严格为0.0，扭转能量从**0.0**变成9.4995 N·mm。

    **必红方式**：不调用`RodModel.retransport`（即"抄了公式不抄外循环"），
    ``γ``停在全零，扭转能量恒为``0.0``——偏差100%，而且**不报任何错**。
    """

    step = 10.0
    model, _ = _holonomy_model(step, ei=1.0e3)
    context = _context(5)
    moved = model.layout.initial_state(
        positions_mm=_spherical_triangle(step), edge_twist_angles=(0.0,) * 4
    )
    #: 不重输运时的对照：这就是"抄了公式不抄外循环"拿到的答案。
    assert model.twist().energy(moved, context) == 0.0
    retransported = model.retransport(moved)
    gammas = retransported.model.layout.twist_angles(retransported.state)
    assert gammas[3] - gammas[0] == -0.5 * math.pi
    assert retransported.max_reference_twist == 0.0
    assert retransported.model.twist().energy(retransported.state, context) > 1.0


def test_retransport_preserves_the_material_directors_it_is_supposed_to_preserve() -> None:
    """重输运换的是**坐标**不是**物理**：``m1``在输运前后必须是同一根（模去切向转动）。"""

    step = 10.0
    model, _ = _holonomy_model(step, ei=1.0e3)
    moved = model.layout.initial_state(
        positions_mm=_spherical_triangle(step), edge_twist_angles=(0.0,) * 4
    )
    before = model.material_directors(moved)
    retransported = model.retransport(moved)
    after = retransported.model.material_directors(retransported.state)
    old_tangents = model.reference.frame.tangents
    new_tangents = retransported.model.reference.frame.tangents
    for edge in range(4):
        expected = parallel_transport(before[edge], old_tangents[edge], new_tangents[edge])
        assert max(abs(a - b) for a, b in zip(after[edge], expected, strict=True)) <= 1.0e-15


def test_without_the_outer_loop_the_rod_never_twists() -> None:
    """**外层循环门（求解半）**：同一个算例，走外层循环 vs 只求解一次。

    钉住全部位置与两端边的γ（＝两端材料帧夹持），内部两个γ自由。

    实测：外层循环第0轮扭转能量``0.0``（此时参考帧还是直杆的），
    第1轮起``3.166498078682835``，与串联扭簧闭式
    ``0.5·GJ·(π/2)²/(3h) = 3.1664980786828356``相对偏差**1.4e-16**；
    ``γ = (0, −π/6, −π/3, −π/2)``严格均匀。
    **只求解一次拿到的扭转能量是``0.0``——偏差100%。**

    弯曲刚度在本门里刻意压到``1e-3 N·mm²``（GJ的1e-5量级）：
    球面三角的90°折角会给出很大的离散曲率，弯曲项对γ的耦合会把扭转分布拉歪，
    而**本门要单独验的是扭转通道**。弯曲通道由门一与门二各自验过。
    """

    step, ei = 10.0, 1.0e-3
    model, straight = _holonomy_model(step, ei=ei)
    layout = model.layout
    context = _context(5)
    target = _spherical_triangle(step)
    prescribed = tuple(
        (3 * node + axis, target[node][axis]) for node in range(5) for axis in range(3)
    )
    fixed = layout.position_indices() | {layout.twist_index(0), layout.twist_index(3)}
    initial = layout.initial_state(positions_mm=straight, edge_twist_angles=(0.0,) * 4)
    equilibrium = solve_rod_with_retransport(
        model=model,
        context=context,
        initial=initial,
        stages=(
            RodSolveStage(
                fixed_indices=fixed, prescribed=prescribed, retransport_rounds=4
            ),
        ),
        residual_tol_n=1.0e-12,
    )
    assert equilibrium.converged
    assert equilibrium.rounds[0].twist_energy_n_mm == 0.0
    closed = 0.5 * GJ_NMM2 * (0.5 * math.pi) ** 2 / (3.0 * step)
    assert abs(equilibrium.rounds[-1].twist_energy_n_mm / closed - 1.0) <= 1.0e-14
    assert equilibrium.rounds[-1].max_gamma_change <= 1.0e-15
    gammas = layout.twist_angles(equilibrium.state)
    for index in range(1, 4):
        assert abs((gammas[index] - gammas[index - 1]) + math.pi / 6.0) <= 1.0e-15

    #: 对照：只求解一次（不重输运），同样的钉法、同样的初值。
    once = solve_equilibrium(
        EnergyRegistry(terms=model.terms()),
        context,
        layout.layout,
        layout.initial_state(positions_mm=target, edge_twist_angles=(0.0,) * 4).vector,
        fixed_indices=fixed,
        residual_tol_n=1.0e-12,
        max_iterations=40,
    )
    assert model.twist().energy(once.state, context) == 0.0


def test_the_twist_term_has_no_positional_dependence_which_is_why_the_loop_exists() -> None:
    """把"为什么必须有外层循环"写成一条断言：扭转项在位置块上的梯度**恒为零**。

    这不是缺陷登记，是形制本身（``m_ref``冻结）。它意味着**单次
    `solve_equilibrium`里弯曲与扭转的交换根本不存在**——0064第4.3节第1条那句话
    在这里变成可执行的。
    """

    step = 10.0
    model, _ = _holonomy_model(step, ei=1.0e3)
    state = model.layout.initial_state(
        positions_mm=_spherical_triangle(step), edge_twist_angles=(0.1, -0.2, 0.3, 0.05)
    )
    gradient = model.twist().gradient(state, _context(5))
    assert all(value == 0.0 for value in gradient[: model.layout.twist_offset])
    assert any(value != 0.0 for value in gradient[model.layout.twist_offset :])


# ------------------------------------------------------- 配对门与自然曲率通道


def test_declaring_ei_easy_above_ei_hard_fails_closed() -> None:
    """**配对门**：``EI_easy > EI_hard``即失败关闭。

    同行那边这个配对没有任何运行时校验，接反了只会给出一个1600倍偏小的挠度
    而不报错。easy按定义就是软的那一轴——反过来就说明``κ1``/``κ2``与两个刚度的
    配对写反了。等号放行（各向同性是合法的退化）。

    **必红方式**：把`rod.AnisotropicRodBending.__post_init__`里的``soft > stiff``
    改成``False``，本用例当场绿不了。
    """

    node_count = 5
    layout, reference, _, _ = _straight(node_count, length=40.0, seed_d1=(0.0, 1.0, 0.0))
    with pytest.raises(RodError, match="easy按定义就是软的那一轴"):
        AnisotropicRodBending(
            layout=layout,
            reference=reference,
            ei_easy_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
            ei_hard_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        )
    #: 各向同性必须放行——否则这道门会挡住一个合法用法。
    AnisotropicRodBending(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        ei_hard_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
    )


def test_natural_kappa2_is_a_live_channel_not_a_dead_slot() -> None:
    """**``natural_kappa2``显式开通道**：同行有这个槽但生产里恒为零，它从没跑过。

    判据是硬的：给一个非零``κ̄2``，直杆（``κ2 = 0``）上的能量必须恰好等于
    ``0.5·EI_hard·κ̄2²/l̄``，而``κ̄1``同值时给出``0.5·EI_easy·κ̄1²/l̄``——
    **两个槽走的是两个刚度**，接错立刻是1000倍的差。
    """

    node_count = 5
    layout, base, nodes, step = _straight(
        node_count, length=40.0, seed_d1=(0.0, 1.0, 0.0)
    )
    vertices = node_count - 2
    context = _context(node_count)
    state = layout.initial_state(
        positions_mm=nodes, edge_twist_angles=(0.0,) * layout.edge_count
    )
    natural = 0.03
    for slot, stiffness in (("natural_kappa2", EI_HARD_NMM2), ("natural_kappa1", EI_EASY_NMM2)):
        reference = RodReference(
            rest_lengths_mm=base.rest_lengths_mm,
            frame=base.frame,
            **{slot: (natural,) * vertices},
        )
        bending = AnisotropicRodBending(
            layout=layout,
            reference=reference,
            ei_easy_nmm2=(EI_EASY_NMM2,) * vertices,
            ei_hard_nmm2=(EI_HARD_NMM2,) * vertices,
        )
        expected = vertices * 0.5 * stiffness * natural**2 / step
        assert bending.energy(state, context) == pytest.approx(expected, rel=1.0e-14)


def test_the_discrete_curvature_convention_is_not_one_over_mm() -> None:
    """离散曲率与物理曲率差``l̄``倍——把它写成断言，因为混用是**静默**的。"""

    radius, pitch = 40.0, 12.0
    model, state = _helix_model(41, radius=radius, pitch=pitch, sweep=2.0)
    dual = model.reference.dual_lengths_mm
    discrete = model.bending().curvatures(state)[0][0]
    physical = radius / (radius**2 + pitch**2)
    assert abs(discrete / dual[0] / physical - 1.0) <= 3.0e-4
    #: 直接拿离散量当1/mm用会差``l̄``倍——本例``l̄ ≈ 2.1 mm``，那是两个数量级。
    assert abs(discrete / physical - 1.0) > 1.0


# --------------------------------------------------------------- 协议与装配


def test_the_fused_path_reproduces_the_separate_calls_bytewise() -> None:
    """spec/12第3.1节的承重条款：融合路径的能量值必须与单独调`energy`**逐字节**相同。"""

    model, state = _helix_model(15, radius=40.0, pitch=12.0, sweep=1.0)
    context = _context(15)
    for term in model.terms():
        fused, gradient, hessian = term.quantities(
            state, context, need_gradient=True, need_hessian=True
        )
        assert fused == term.energy(state, context)
        assert gradient == term.gradient(state, context)
        assert hessian == term.hessian(state, context)


def test_the_sparse_hessian_entries_reproduce_the_dense_matrix() -> None:
    """稀疏读法与稠密面必须是同一份数学（`solve_equilibrium`只走稀疏那条）。"""

    model, state = _helix_model(9, radius=40.0, pitch=12.0, sweep=0.8)
    context = _context(9)
    size = len(state.vector)
    for term in model.terms():
        dense = term.hessian(state, context)
        accumulated = [[0.0] * size for _ in range(size)]
        for row, column, value in term.hessian_entries(state, context):
            accumulated[row][column] += value
        for row in range(size):
            for column in range(size):
                assert accumulated[row][column] == pytest.approx(
                    dense[row][column], rel=1.0e-12, abs=1.0e-12
                )


def test_the_jet_gradient_and_hessian_match_central_differences() -> None:
    """有限差分门：验"雅可比是不是我写的那个能量的导数"。

    **它验不了物理**（spec/12第6.1节）——验物理的是上面那三道闭式门。
    """

    model, state = _helix_model(7, radius=40.0, pitch=12.0, sweep=0.7)
    context = _context(7)
    registry = EnergyRegistry(terms=model.terms())
    _, gradient, hessian = registry.total(
        state, context, need_gradient=True, need_hessian=True
    )
    step = 1.0e-6
    scale = max(abs(value) for value in gradient)
    for index in range(len(state.vector)):
        forward = list(state.vector)
        backward = list(state.vector)
        forward[index] += step
        backward[index] -= step
        numeric = (
            registry.total(state.with_vector(tuple(forward)), context)[0]
            - registry.total(state.with_vector(tuple(backward)), context)[0]
        ) / (2.0 * step)
        assert abs(numeric - gradient[index]) <= 1.0e-5 * scale
    curvature_scale = max(abs(value) for row in hessian for value in row)
    for index in range(len(state.vector)):
        forward = list(state.vector)
        backward = list(state.vector)
        forward[index] += step
        backward[index] -= step
        plus = registry.total(
            state.with_vector(tuple(forward)), context, need_gradient=True
        )[1]
        minus = registry.total(
            state.with_vector(tuple(backward)), context, need_gradient=True
        )[1]
        for column in range(len(state.vector)):
            numeric = (plus[column] - minus[column]) / (2.0 * step)
            assert abs(numeric - hessian[index][column]) <= 1.0e-5 * curvature_scale


def test_the_rod_energy_is_invariant_under_a_large_rigid_rotation() -> None:
    """刚体转动不变性——**几何精确的定义**，也是"帧跟着转"的直接证据。

    把杆与参考帧一起转90°，能量必须逐位不变；**只转杆不转帧**（即忘了重输运）
    则同一个构型的能量会按``EI_hard/EI_easy``跳一个量级。
    """

    node_count = 9
    layout, reference, nodes, step = _straight(
        node_count, length=80.0, seed_d1=(0.0, 1.0, 0.0)
    )
    context = _context(node_count)
    bent = tuple(
        (node[0], node[1] + 0.004 * node[0] ** 2, node[2]) for node in nodes
    )

    def energy(positions, seed):
        frame = build_bishop_frame(positions_mm=positions, seed_d1=seed)
        moved_reference = RodReference(
            rest_lengths_mm=reference.rest_lengths_mm, frame=frame
        )
        term = AnisotropicRodBending(
            layout=layout,
            reference=moved_reference,
            ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
            ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
        )
        state = layout.initial_state(
            positions_mm=positions, edge_twist_angles=(0.0,) * layout.edge_count
        )
        return term.energy(state, context)

    #: 绕x轴转90°：y → z、z → −y。帧一起转（种子从ŷ变成ẑ）。
    rotated = tuple((node[0], -node[2], node[1]) for node in bent)
    straight_energy = energy(bent, (0.0, 1.0, 0.0))
    rotated_energy = energy(rotated, (0.0, 0.0, 1.0))
    assert abs(rotated_energy / straight_energy - 1.0) <= 1.0e-14
    #: 忘了把帧转过去：同一个构型，能量按``EI_hard/EI_easy``跳。
    stale_energy = energy(rotated, (0.0, 1.0, 0.0))
    assert abs(stale_energy / straight_energy - EI_HARD_NMM2 / EI_EASY_NMM2) <= 1.0e-6


# --------------------------------------------------------------- 形制的失败关闭


def test_the_layout_puts_positions_first_and_gammas_at_the_end() -> None:
    """自由度次序是**裁决**（0064第4.2节），所以它被断言而不是被假定。"""

    layout = build_rod_layout(layout_id="layout/rod-order", node_count=6)
    assert [field.name for field in layout.layout.fields] == [
        "node_positions_mm",
        "edge_twist_angles",
    ]
    assert layout.layout.node_dof_count == 18
    assert layout.twist_offset == 18
    assert layout.twist_index(0) == 18
    assert layout.twist_index(4) == 22
    assert layout.layout.dof_count == 23


def test_shape_mismatches_fail_closed() -> None:
    """静长、刚度、自然曲率的长度对不上时一律拒收——**长度对得上但物理是错的**最坏。"""

    node_count = 6
    layout, reference, _, _ = _straight(node_count, length=50.0, seed_d1=(0.0, 1.0, 0.0))
    with pytest.raises(RodError, match="one easy and one hard value per vertex"):
        AnisotropicRodBending(
            layout=layout,
            reference=reference,
            ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 1),
            ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 1),
        )
    with pytest.raises(RodError, match="one value per interior vertex"):
        RodTwist(layout=layout, reference=reference, gj_nmm2=(GJ_NMM2,) * 2)
    with pytest.raises(RodError, match="one entry per interior vertex"):
        RodReference(
            rest_lengths_mm=reference.rest_lengths_mm,
            frame=reference.frame,
            natural_kappa1=(0.0,) * 2,
        )
    with pytest.raises(RodError, match="at least three nodes"):
        build_rod_layout(layout_id="layout/rod-too-short", node_count=2)


def test_a_folded_back_vertex_fails_closed_rather_than_returning_a_big_number() -> None:
    """``θ → π``是``κb = 2·tan(θ/2)``自身的奇点，不是可以返回大数的地方。"""

    layout = build_rod_layout(layout_id="layout/rod-fold", node_count=3)
    nodes = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0))
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=(0.0, 1.0, 0.0))
    reference = RodReference(rest_lengths_mm=(10.0, 10.0), frame=frame)
    bending = AnisotropicRodBending(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,),
        ei_hard_nmm2=(EI_HARD_NMM2,),
    )
    folded = layout.initial_state(
        positions_mm=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        edge_twist_angles=(0.0, 0.0),
    )
    with pytest.raises(RodError, match="folded back"):
        bending.energy(folded, _context(3))


def test_the_outer_loop_needs_at_least_one_stage_and_one_round() -> None:
    step = 10.0
    model, straight = _holonomy_model(step, ei=1.0e3)
    initial = model.layout.initial_state(
        positions_mm=straight, edge_twist_angles=(0.0,) * 4
    )
    with pytest.raises(RodError, match="at least one stage"):
        solve_rod_with_retransport(
            model=model, context=_context(5), initial=initial,
            stages=(), residual_tol_n=1.0e-12,
        )
    with pytest.raises(RodError, match="at least one round"):
        solve_rod_with_retransport(
            model=model, context=_context(5), initial=initial,
            stages=(RodSolveStage(retransport_rounds=0),), residual_tol_n=1.0e-12,
        )


def test_signed_angle_and_the_frame_agree_on_the_sign_convention() -> None:
    """符号约定被写成断言：绕``t``从``d1``转到``d2``是``+π/2``。"""

    tangent = (1.0, 0.0, 0.0)
    assert signed_angle((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), tangent) == pytest.approx(
        0.5 * math.pi, abs=1.0e-15
    )
    frame = build_bishop_frame(
        positions_mm=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        seed_d1=(0.0, 1.0, 0.0),
    )
    assert frame.d2[0] == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-15)


def test_the_fused_path_matches_per_vertex_not_only_in_the_sum() -> None:
    """**逐顶点**相同，不只是求和后相同——2026-08-18跨机实测逼出来的一条门。

    既有的`test_the_fused_path_reproduces_the_separate_calls_bytewise`只比**总能量**。
    实测发现两条路**在每个顶点上本来就不同**，而那道门在本机一直绿，
    **靠的是求和时误差恰好抵消**：

    | 平台 | 逐顶点不同的顶点数 | 总能量 |
    |---|---|---|
    | macOS arm64 / CPython 3.13 | 1个 | 恰好抵消，门绿 |
    | Linux x86-64 / CPython 3.12 | 4个 | 差1 ULP，门红 |

    根因在`autodiff.ad_dot`：它原来用`sum()`，而**CPython对纯`float`的`sum()`
    走补偿求和、对`Jet`只能走泛型`__add__`**——同一个函数在两种输入上用了两套加法。

    **这条门判的位置比那条更靠里**：抵消发生在求和这一步，
    所以判总和的门永远看不见它。本仓plans/09教训二的通则
    （"判据要落在结构位置上"）在这里的形态就是**判到还没被求和的那一层**。
    """

    model, state = _helix_model(15, radius=40.0, pitch=12.0, sweep=1.0)
    for term in model.terms():
        for vertex in range(term._vertex_count()):
            plain = term._vertex_energy(state, vertex, order=0)
            jet = term._vertex_energy(state, vertex, order=2)
            jet_value = jet.value if hasattr(jet, "value") else jet
            assert plain.hex() == float(jet_value).hex(), (
                f"{term.name} 顶点{vertex}：float路 {plain.hex()} 与 jet路 "
                f"{float(jet_value).hex()} 不同 —— 两条路在被求和之前就已经分叉，"
                "而判总和的那道门会因为误差抵消而看不见"
            )


def test_ad_dot_does_not_use_compensated_summation() -> None:
    """必红：`ad_dot`必须与**顺序累加**逐位相同，不许退回`sum()`。

    构造一组会让两种算法分道扬镳的输入（大数相消）：
    `sum()`的补偿项会救回那个1.0，顺序累加不会。**判的是`ad_dot`站在后一边。**
    """

    from physics_engine.autodiff import ad_dot

    left = (1.0e16, 1.0, -1.0e16)
    right = (1.0, 1.0, 1.0)
    sequential = left[0] * right[0]
    for index in range(1, 3):
        sequential = sequential + left[index] * right[index]

    assert sum(a * b for a, b in zip(left, right, strict=True)) != sequential, (
        "这组输入没能把两种求和算法分开 —— 那本条用例就没有分辨力了"
    )
    assert ad_dot(left, right).hex() == sequential.hex(), (
        "ad_dot 走回了补偿求和 —— 那会让它在float与Jet两种输入上用两套加法"
    )
