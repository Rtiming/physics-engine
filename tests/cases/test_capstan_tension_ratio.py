"""绞盘张力比的conformance门（案例`cases/capstan_tension_ratio`，能力位S6.5）。

**一条装置穿四层**：`PenaltyCylinderContact`（法向）、`TangentialStickSpring`
与`coulomb_return_map`（摩擦）、`advance_contacts_quasistatic`（多槽位同时迭代）、
`AxialStretch`（张力沿带材传递）。这是决策0062第五节判据表里的J1，
**本轮判据强度最高的一条**。

## 四条判据各自守什么、各自的精度从哪来

| 判据 | 实测精度 | 精度由什么定 |
|---|---|---|
| 每个接触精确落在摩擦锥上 | **2.22e-16** | return-map理想塑性修正的**定义性质**，与载荷步数无关 |
| 法向力＝两侧张力的法向合量 | **2.44e-07** | 求解器残差容差`1e-6 N`除以法向力`≈4 N` |
| 逐节点张力比对精确离散式 | 4.26e-03 | **载荷步一阶误差**，步数翻倍减半 |
| 收敛阶 | 比2.10/2.06 | 同上 |

**前两条是恒等式，后两条是收敛结果**。分开写是因为它们红了说明的事完全不同：
第一条红说明return-map错了，第二条红说明力平衡装配错了，
后两条红说明载荷步不够或摩擦系数用错了。

## 为什么不判端到端的``T_last/T_first``

因为两端是**半节点**：尖端加载、根部固支，各自少一侧张力。
端到端比里混着这两个边界效应，2026-08-17实测它随载荷步数**非单调**
（8段：120步给1.6030、240步给1.5562、480步给1.5341），
而逐节点比在同一批数据上是干净的一阶收敛。
**端到端那个数看起来更像教科书，但它判不动东西。**
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    ContactPoint,
    PenaltyCylinderContact,
    advance_contacts_quasistatic,
    build_contact_layout,
)
from physics_engine.energies import (
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    PointLoad,
)
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State

CASE = Path(__file__).resolve().parents[2] / "cases" / "capstan_tension_ratio"
MANIFEST = load_manifest(CASE / "oracle.json")

#: 真机导轮（`winding-machine/HARDWARE_TOPOLOGY.md` 2026-07-07现场确认）：
#: 外径100 mm、有效宽度17 mm。
RADIUS_MM = 50.0
HALF_WIDTH_MM = 8.5
WRAP_RAD = math.pi / 2.0
FRICTION = 0.30
SEGMENTS = 8

#: 带材：4 mm宽×0.1 mm厚、``E ≈ 150 GPa`` ⟹ ``EA = 60 kN``。**假设输入**
#: （0062第二节裁决2的五项数据缺口之一），故本案例永久`hypothesis_only`。
EA_N = 60000.0
PRETENSION_N = 10.0
PULL_N = 30.0
#: 罚刚度：法向与切向同取``1e4 N/mm``。穿透``N/k``约``4e-4 mm``，
#: 是带厚的0.4%，在这个案例的判据（力与比值）上不出现。
PENALTY_N_PER_MM = 1.0e4

#: **求解器容差按能量分辨率地板给，不按刚度地板给。**
#: 2026-08-17实测：`PointLoad`让总能量到``−550 N·mm``量级，
#: 而接触与拉伸的能量变化是``1e-4``量级——线搜索在残差``4e-8``处
#: 再也分辨不出"能量真的下降了"。取``1e-6``留25倍余量。
#: 残差``1e-6 N``对张力``10—30 N``是``1e-7``相对，判据判的量都在这之上。
RESIDUAL_TOL_N = 1.0e-6


def _oracle(oracle_id: str):
    for entry in MANIFEST.oracles:
        if entry.id == oracle_id:
            return entry
    raise AssertionError(f"清单里没有{oracle_id}")


def _build(segments: int):
    nodes = segments + 1
    dphi = WRAP_RAD / segments
    chord = 2.0 * RADIUS_MM * math.sin(dphi / 2.0)
    #: 静止长度取成初始几何下张力恰为预张——**起点就在平衡附近**。
    rest = chord / (1.0 + PRETENSION_N / EA_N)
    #: 起点半径下沉``N/k``，让接触**一开始就活动**：``g = 0``落在活动集边界外
    #: （判据是``g < 0``），那里径向没有刚度、牛顿迈不出第一步。
    sink = PRETENSION_N * dphi / PENALTY_N_PER_MM
    angles = [index * dphi for index in range(nodes)]
    positions = [
        ((RADIUS_MM - sink) * math.cos(a), (RADIUS_MM - sink) * math.sin(a), 0.0)
        for a in angles
    ]

    contact_nodes = list(range(1, nodes))
    layout = build_contact_layout(
        layout_id="layout/capstan",
        node_count=nodes,
        declarations=tuple(ContactDeclaration(f"wrap{i}") for i in contact_nodes),
    )
    context = EnergyContext(
        context_id="context/capstan",
        #: 质量只为满足`EnergyContext`的正数要求；重力设为零——
        #: 整匝带材自重``3.5e-2 N``比张力小三个量级（WDS design/14同一裁决）。
        node_masses_kg=(1.0e-9,) * nodes,
        gravity_mm_s2=(0.0, 0.0, 0.0),
    )
    stretch = AxialStretch(
        edges=tuple((i, i + 1, rest, EA_N) for i in range(segments))
    )
    cylinder = PenaltyCylinderContact(
        cylinders=tuple(
            (i, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), RADIUS_MM, HALF_WIDTH_MM,
             PENALTY_N_PER_MM, 0.0)
            for i in contact_nodes
        )
    )
    tangent = (-math.sin(angles[-1]), math.cos(angles[-1]), 0.0)

    def registry(load_n: float) -> EnergyRegistry:
        return EnergyRegistry(
            terms=(
                stretch,
                cylinder,
                PointLoad(loads=((nodes - 1, tuple(c * load_n for c in tangent)),)),
            )
        )

    initial = layout.initial_vector(tuple(c for p in positions for c in p))
    #: 全部``z``钉住：圆柱接触在**轴向零刚度**（`PenaltyCylinderContact`的
    #: ``H·a = 0``），不钉就是欠约束、`solve_equilibrium`当场报奇异。
    #: **这条是那个几何性质在案例层的直接后果，不是随手加的边界条件。**
    fixed = frozenset(
        {0, 1, 2}
        | {3 * i + 2 for i in range(nodes)}
        | set(range(layout.layout.node_dof_count, layout.layout.dof_count))
    )
    return layout, context, registry, stretch, cylinder, contact_nodes, initial, fixed


def _run(segments: int, load_steps: int):
    """预张→贴合→载荷步连续化到全滑移。返回最终一步与两个观测项。"""

    layout, context, registry, stretch, cylinder, contact_nodes, initial, fixed = _build(
        segments
    )
    settled = solve_equilibrium(
        registry(PRETENSION_N), context, layout.layout, initial,
        fixed_indices=fixed, residual_tol_n=1.0e-7, max_iterations=200,
    )
    assert settled.converged, settled.reason

    current = list(settled.state.vector)
    #: 锚点 = 当前位置：此刻带材还没有滑过，粘着弹簧的自然长度为零。
    for node in contact_nodes:
        slot = layout.slot_of(f"wrap{node}")
        current[slot.anchor_base : slot.anchor_base + 3] = current[
            3 * node : 3 * node + 3
        ]

    def normal_of(order: int):
        def call(vector: tuple[float, ...]):
            return cylinder.outward_normal(
                State(layout=layout.layout, vector=vector)
            )[order]

        return call

    contacts = tuple(
        ContactPoint(
            slot=layout.slot_of(f"wrap{node}"), node=node,
            normal=normal_of(order),
            normal_force_of=(lambda state, o=order: cylinder.normal_force_n(state)[o]),
            tangential_stiffness_n_per_mm=PENALTY_N_PER_MM,
            friction_coefficient=FRICTION,
        )
        for order, node in enumerate(contact_nodes)
    )

    vector = tuple(current)
    step = None
    for index in range(1, load_steps + 1):
        load = PRETENSION_N + (PULL_N - PRETENSION_N) * index / load_steps
        step = advance_contacts_quasistatic(
            registry_without_stick=registry(load), context=context,
            contact_layout=layout, contacts=contacts, vector=vector,
            fixed_indices=fixed, residual_tol_n=RESIDUAL_TOL_N,
            max_iterations=200, max_passes=4, yield_tol_n=1.0e-7,
        )
        vector = step.state.vector
    assert step is not None
    return step, stretch, cylinder


@pytest.fixture(scope="module")
def sweep():
    """三档载荷步跑一次，四条门共用——每档一次求解，不重复跑。"""

    return {steps: _run(SEGMENTS, steps) for steps in (60, 120, 240)}


def _tension_ratios(step, stretch) -> list[float]:
    forces = stretch.axial_force_n(step.state)
    return [forces[i + 1] / forces[i] for i in range(len(forces) - 1)]


# ---------------------------------------------------------------------------
# 恒等式两条（与载荷步数无关）
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_every_contact_sits_exactly_on_the_friction_cone(sweep):
    """**全滑移下每个接触精确落在摩擦锥面上**：``|T_i| / (μ·N_i) == 1``。

    这是return-map理想塑性修正的**定义性质**，不是收敛结果——
    所以它与载荷步数无关，三档全判。2026-08-17实测最大偏差**2.22e-16**，
    即一个ulp。

    **它是绞盘公式成立的前提**：教科书那条式子的推导第一步就是"摩擦取满"。
    这一条红了，后面两条判什么都没有意义。
    """

    entry = _oracle("oracle:capstan/cone_saturation")
    limit = entry.tolerances["tangential_over_mu_normal"].abs_tol
    for steps, (step, _, cylinder) in sweep.items():
        normals = cylinder.normal_force_n(step.state)
        for index, force in enumerate(step.tangential_force_n):
            magnitude = math.sqrt(sum(value * value for value in force))
            assert normals[index] > 0.0, f"{steps}步：接触{index}没有法向力"
            ratio = magnitude / (FRICTION * normals[index])
            assert ratio == pytest.approx(1.0, abs=limit), (
                f"{steps}步：接触{index}的|T|/(μN) = {ratio!r}，不在锥面上"
            )


@pytest.mark.batch
def test_the_normal_force_is_the_transverse_component_of_the_tension_kink(sweep):
    """``N_i = −(T⁺·d⁺ − T⁻·d⁻)·n_i``：法向力就是两侧张力的法向合量。

    **用实际段方向``d±``，不用名义的``sin(Δφ/2)``。** 带材滑过之后节点的角间距
    不再是均匀的``Δφ``，照名义角算会有``2.8e-4``的系统偏差
    （2026-08-17实测）——那个偏差是几何的，不是物理的，
    **拿它当判据会把一个正确的实现判红**。

    用实际方向时实测最大相对偏差``2.44e-07``，恰等于求解器残差容差``1e-6 N``
    除以法向力``≈4 N``。**判据的精度由求解容差定，这条关系本身是精确的。**
    """

    for steps, (step, stretch, cylinder) in sweep.items():
        vector = step.state.vector
        tensions = stretch.axial_force_n(step.state)
        normals = cylinder.outward_normal(step.state)
        forces = cylinder.normal_force_n(step.state)

        def direction(a: int, b: int, state=vector) -> tuple[float, ...]:
            delta = [state[3 * b + k] - state[3 * a + k] for k in range(3)]
            length = math.sqrt(sum(c * c for c in delta))
            return tuple(c / length for c in delta)

        for order in range(len(forces) - 1):
            node = order + 1
            incoming = direction(node - 1, node)
            outgoing = direction(node, node + 1)
            normal = normals[order]
            predicted = -sum(
                (tensions[node] * outgoing[k] - tensions[node - 1] * incoming[k])
                * normal[k]
                for k in range(3)
            )
            tolerance = 4.0 * RESIDUAL_TOL_N
            assert predicted == pytest.approx(forces[order], abs=tolerance), (
                f"{steps}步：节点{node}的法向力{forces[order]!r}与张力折角"
                f"的法向合量{predicted!r}对不上"
            )


# ---------------------------------------------------------------------------
# 收敛结果两条
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_node_tension_ratio_matches_the_exact_discrete_capstan_value(sweep):
    """包角中点的逐节点张力比对精确离散式
    ``(1 + μ·tan(Δφ/2)) / (1 − μ·tan(Δφ/2))``。

    **判的是离散式不是``e^{μΔφ}``。** 两者差``O(Δφ²)``，在``Δφ = π/8``时
    是``1.6e-3``相对——**比本门的容差还大**，照连续式判会把正确实现判红。
    这条陷阱写在金标生成器的docstring第二节里。
    """

    entry = _oracle("oracle:capstan/node_tension_ratio")
    expected = entry.expected["node_ratio"]
    limit = entry.tolerances["node_ratio"].rel_tol
    step, stretch, _ = sweep[240]
    ratios = _tension_ratios(step, stretch)
    middle = ratios[len(ratios) // 2]
    assert middle == pytest.approx(expected, rel=limit), (
        f"包角中点的张力比{middle!r}对不上精确离散值{expected!r}"
    )


@pytest.mark.batch
def test_the_load_step_error_is_first_order(sweep):
    """**载荷步翻倍，误差减半。**

    2026-08-17实测（8段、包角中点）：60步``1.833e-2``、120步``8.749e-3``、
    240步``4.256e-3``，两个比值**2.10与2.06**。

    门判比值落在``[1.7, 2.4]``而**不写死为2**——与`harmonic_oscillator`
    那条"收敛比不写死为4"同源：**一致才是收敛的证据，具体值不是**。

    这一条与上一条必须并存：**只判落点会被一个碰巧的常数骗过，
    只判阶会被一个系统偏移骗过**（偏移不改变阶）。
    """

    entry = _oracle("oracle:capstan/node_tension_ratio")
    expected = entry.expected["node_ratio"]
    errors = []
    for steps in (60, 120, 240):
        step, stretch, _ = sweep[steps]
        ratios = _tension_ratios(step, stretch)
        middle = ratios[len(ratios) // 2]
        errors.append(abs(middle - expected) / expected)

    assert errors[0] > errors[1] > errors[2], f"误差没有单调下降：{errors}"
    for earlier, later in zip(errors, errors[1:], strict=False):
        assert 1.7 <= earlier / later <= 2.4, (
            f"载荷步误差不是一阶：比值{earlier / later!r}，实测序列{errors}"
        )


# ---------------------------------------------------------------------------
# 金标自洽（不碰引擎，秒级）
# ---------------------------------------------------------------------------


def test_the_discrete_ratio_converges_to_euler_eytelwein_at_second_order():
    """离散连乘随段数**二阶**收敛到``exp(μθ)``。

    这一条不判引擎，判的是**案例自己的两条闭式自洽**：
    离散式的极限必须真的是Euler-Eytelwein，否则上一条容差在比什么就说不清。

    实测16/32/64段对连续式的相对偏差``4.13e-4 / 1.03e-4 / 2.58e-5``，
    比值**4.00 / 4.00**。
    """

    entry = _oracle("oracle:capstan/continuum_limit")
    continuum = entry.expected["continuum_ratio"]
    assert continuum == pytest.approx(math.exp(FRICTION * WRAP_RAD), rel=1e-15)

    errors = [
        abs(entry.expected[f"discrete_{n}"] - continuum) / continuum
        for n in (16, 32, 64)
    ]
    assert errors[0] > errors[1] > errors[2]
    for earlier, later in zip(errors, errors[1:], strict=False):
        assert earlier / later == pytest.approx(4.0, rel=0.05), (
            f"离散→连续不是二阶：比值{earlier / later!r}，序列{errors}"
        )


def test_the_continuum_formula_would_put_a_floor_under_an_error_that_should_vanish():
    """**判据选错会怎样**：拿``e^{μΔφ}``当逐节点判据，等于给误差垫一个floor。

    这条门不注错任何代码，它量的是**两条闭式之间的系统偏差**
    ``exact − continuum ≈ μ·Δφ³/12``（2026-08-17实测三档：
    ``Δφ = π/16``给``2.07e-4``、``π/8``给``1.68e-3``、``π/4``给``1.41e-2``，
    与``μΔφ³/12``的``1.89e-4``/``1.51e-3``/``1.21e-2``逐档同量级）。

    **本案例今天用它不会红**：``2.07e-4``只占``6e-3``容差的3.5%。
    要紧的不是会不会红，是**它不随载荷步数变小**——
    载荷步误差是一阶趋零的，而这个偏差是几何的、恒定的。
    于是`test_the_load_step_error_is_first_order`那条门在载荷步足够多时
    **会先撞上这个floor然后停住**，而症状看起来像"收敛阶坏了"。

    **另一个方向也要说清**：偏差按``Δφ³``走，所以**加密网格是往好里走**，
    真正会咬人的是往粗里走——2段（``Δφ = π/4``）时``1.41e-2``直接越界。
    """

    entry = _oracle("oracle:capstan/node_tension_ratio")
    exact = entry.expected["node_ratio"]
    limit = entry.tolerances["node_ratio"].rel_tol

    delta_phi = WRAP_RAD / SEGMENTS
    gap = abs(math.exp(FRICTION * delta_phi) - exact) / exact
    assert gap == pytest.approx(2.07e-4, rel=0.05)
    assert gap == pytest.approx(FRICTION * delta_phi**3 / 12.0, rel=0.15), (
        "系统偏差与``μΔφ³/12``对不上——那这条门给出的机理解释是错的"
    )
    assert gap < limit, "本案例这一档不该因此红——若红了说明容差表要重算"

    #: 往粗里走两档，偏差必须真的越界；否则"选错判据会咬人"只是一句话。
    for segments, expected_gap in ((4, 1.68e-3), (2, 1.41e-2)):
        coarse = WRAP_RAD / segments
        tangent = math.tan(coarse / 2.0)
        coarse_exact = (1.0 + FRICTION * tangent) / (1.0 - FRICTION * tangent)
        coarse_gap = abs(math.exp(FRICTION * coarse) - coarse_exact) / coarse_exact
        assert coarse_gap == pytest.approx(expected_gap, rel=0.05)
    assert coarse_gap > limit, (
        f"最粗一档的偏差{coarse_gap!r}仍没越界——那本门的结论要重写"
    )
