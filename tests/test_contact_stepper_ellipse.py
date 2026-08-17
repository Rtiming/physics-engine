"""步进器接摩擦椭圆的门（plans/15第1.6条，决策0072）。

0068把椭圆落进了`anisotropic_return_map`，但**两个步进器一个字没动**：
它们把``friction_coefficient``当标量写死，于是各向异性案例走不到`solve_equilibrium`。
本文件守的是那条接线，四件事：

| # | 门 | 判什么 |
|---|---|---|
| 1 | **逐位退化** | ``μ_∥ = μ_⊥``时与旧标量路径**逐位**相同（判的是`float.hex()`） |
| 2 | **通用支也退化** | 把转交掐掉（μ差``2⁻⁴⁰``相对），偏差**由扰动带出来**而不是由路径带出来 |
| 3 | **走整条路** | 椭圆经过`solve_equilibrium`跑出一个真的平衡解，**并且改了它** |
| 4 | 失败关闭 | 两个屈服面给两个或给零个，当场拒收 |

## 第3条本来差点是一道空门——**平面绞盘分辨不出两条屈服面**

派活书说"同一个绞盘算例，各向同性与椭圆两条都跑通"。**跑通了，
而且给出的是逐位相同的答案**（`test_a_planar_capstan_cannot_tell_the_two_apart`
把这一条钉成断言）。原因是几何的：平面绞盘上所有``z``都被钉住，
接触法向是径向、试探力恒为周向，而周向恰好**就是椭圆的主轴**——
0068第二节写着"椭圆的外法向不平行于``f``，**除非``a = b``或``f``落在主轴上**"，
这个算例落的正是后一个例外。

所以本文件的椭圆算例把纵向轴绕接触法向转45°（＝**轮轴偏斜**，
plans/15第4.1节第3项那个待实测量），滑移这才落到混合角上。
**没有这一步，第3条门会全绿而什么都没验。**
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    ContactError,
    ContactPoint,
    FrictionEllipse,
    FrictionEllipseSpec,
    PenaltyCylinderContact,
    PenaltyNormalContact,
    advance_contact_quasistatic,
    advance_contacts_quasistatic,
    build_contact_layout,
    yield_excess_n,
)
from physics_engine.energies import (
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    PointLoad,
    UniformGravity,
)
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State

# --------------------------------------------------------------- 装置一：固定法向

MASS_KG = 1.0
GRAVITY_MM_S2 = 9810.0
FLAT_STIFFNESS_N_PER_MM = 1.0e5
FLAT_MU = 0.35
MU_ALONG = 0.5
MU_ACROSS = 0.1
FLAT_RESIDUAL_TOL_N = 1.0e-9


def _flat(*, load_n: float, angle_rad: float):
    """一个质点压在半空间上、被一个面内力斜拉——**法向恒为``ẑ``，一趟即精确**。

    法向不转是本装置的全部理由：椭圆是拿**装配时**的法向造的，
    法向一转，事后拿求解后的法向复原出来的椭圆就不是同一个，
    ``Φ = 1``那条判据于是量到的是"法向转了多少"而不是"映射准不准"
    （绞盘那边实测正是如此，见第3条门的注释）。
    """

    contact_layout = build_contact_layout(
        layout_id="layout/ellipse-flat",
        node_count=1,
        declarations=(ContactDeclaration("floor"),),
    )
    context = EnergyContext(
        context_id="context/ellipse-flat",
        node_masses_kg=(MASS_KG,),
        gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
    )
    plane = PenaltyNormalContact(
        planes=((0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), FLAT_STIFFNESS_N_PER_MM, 0.0),)
    )
    pull = (load_n * math.cos(angle_rad), load_n * math.sin(angle_rad), 0.0)
    registry = EnergyRegistry(
        terms=(UniformGravity(), plane, PointLoad(loads=((0, pull),)))
    )
    #: 起点已在法向平衡上（``δ = mg/k``），锚点就在起点：粘着弹簧自然长度为零。
    sink = MASS_KG * GRAVITY_MM_S2 * 1.0e-3 / FLAT_STIFFNESS_N_PER_MM
    vector = list(contact_layout.initial_vector((0.0, 0.0, -sink)))
    slot = contact_layout.slot_of("floor")
    vector[slot.anchor_base : slot.anchor_base + 3] = [0.0, 0.0, -sink]
    return contact_layout, context, plane, registry, slot, tuple(vector)


def _flat_step(*, load_n: float, angle_rad: float, surface: dict, max_passes: int = 1):
    contact_layout, context, plane, registry, slot, vector = _flat(
        load_n=load_n, angle_rad=angle_rad
    )
    return advance_contact_quasistatic(
        registry_without_stick=registry,
        context=context,
        contact_layout=contact_layout,
        slot=slot,
        vector=vector,
        node=0,
        normal=(0.0, 0.0, 1.0),
        normal_force_of=lambda state: plane.normal_force_n(state)[0],
        tangential_stiffness_n_per_mm=FLAT_STIFFNESS_N_PER_MM,
        fixed_indices=frozenset(
            range(contact_layout.layout.node_dof_count, contact_layout.layout.dof_count)
        ),
        residual_tol_n=FLAT_RESIDUAL_TOL_N,
        max_iterations=200,
        max_passes=max_passes,
        yield_tol_n=1.0e-9,
        **surface,
    )


# --------------------------------------------------------------- 装置二：绞盘（法向转）

RADIUS_MM = 50.0
HALF_WIDTH_MM = 8.5
WRAP_RAD = math.pi / 2.0
SEGMENTS = 8
EA_N = 60000.0
PRETENSION_N = 10.0
PULL_N = 30.0
PENALTY_N_PER_MM = 1.0e4
RESIDUAL_TOL_N = 1.0e-6
#: 载荷步数**不是随手取的**：实测12/20/30步时内层牛顿在第一步就走不动
#: （200趟不收敛），60步才活。这条与`cases/capstan_tension_ratio`同源——
#: 那里三档取60/120/240，60是它的下界。
LOAD_STEPS = 60
#: 纵向轴绕接触法向转的角度＝轮轴偏斜。取45°是因为0068量出**混合角耗散短缺
#: 在45°上取最大**（61.538%，闭式``(a−b)²/(a²+b²) = 8/13``）。
SKEW_RAD = math.pi / 4.0


def _capstan():
    nodes = SEGMENTS + 1
    dphi = WRAP_RAD / SEGMENTS
    chord = 2.0 * RADIUS_MM * math.sin(dphi / 2.0)
    rest = chord / (1.0 + PRETENSION_N / EA_N)
    sink = PRETENSION_N * dphi / PENALTY_N_PER_MM
    angles = [index * dphi for index in range(nodes)]
    positions = [
        ((RADIUS_MM - sink) * math.cos(a), (RADIUS_MM - sink) * math.sin(a), 0.0)
        for a in angles
    ]
    contact_nodes = list(range(1, nodes))
    layout = build_contact_layout(
        layout_id="layout/ellipse-capstan",
        node_count=nodes,
        declarations=tuple(ContactDeclaration(f"wrap{i}") for i in contact_nodes),
    )
    context = EnergyContext(
        context_id="context/ellipse-capstan",
        node_masses_kg=(1.0e-9,) * nodes,
        gravity_mm_s2=(0.0, 0.0, 0.0),
    )
    stretch = AxialStretch(edges=tuple((i, i + 1, rest, EA_N) for i in range(SEGMENTS)))
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
    fixed = frozenset(
        {0, 1, 2}
        | {3 * i + 2 for i in range(nodes)}
        | set(range(layout.layout.node_dof_count, layout.layout.dof_count))
    )
    return layout, context, registry, stretch, cylinder, contact_nodes, initial, fixed


def _tape_axis(node: int, skew_rad: float):
    """带材纵轴：周向绕**接触法向**转``skew_rad``。``skew_rad = 0``即纯周向。

    它是可调用对象而不是常量元组，因为带材绕着轮走时纵轴跟着位形转——
    `FrictionEllipseSpec`吃`NormalSource`就是为这一条。
    """

    def call(vector: tuple[float, ...]) -> tuple[float, float, float]:
        x, y = vector[3 * node], vector[3 * node + 1]
        length = math.hypot(x, y)
        circumferential = (-y / length, x / length, 0.0)
        cosine, sine = math.cos(skew_rad), math.sin(skew_rad)
        return tuple(
            cosine * circumferential[axis] + sine * (0.0, 0.0, 1.0)[axis]
            for axis in range(3)
        )

    return call


def _run_capstan(surface):
    """预张→贴合→60个载荷步推到全滑移。``surface(node)``给出这个点的屈服面参数。"""

    layout, context, registry, stretch, cylinder, contact_nodes, initial, fixed = _capstan()
    settled = solve_equilibrium(
        registry(PRETENSION_N), context, layout.layout, initial,
        fixed_indices=fixed, residual_tol_n=1.0e-7, max_iterations=200,
    )
    assert settled.converged, settled.reason
    current = list(settled.state.vector)
    for node in contact_nodes:
        slot = layout.slot_of(f"wrap{node}")
        current[slot.anchor_base : slot.anchor_base + 3] = current[3 * node : 3 * node + 3]

    def normal_of(order: int):
        def call(vector: tuple[float, ...]):
            return cylinder.outward_normal(State(layout=layout.layout, vector=vector))[order]

        return call

    contacts = tuple(
        ContactPoint(
            slot=layout.slot_of(f"wrap{node}"), node=node,
            normal=normal_of(order),
            normal_force_of=(lambda state, o=order: cylinder.normal_force_n(state)[o]),
            tangential_stiffness_n_per_mm=PENALTY_N_PER_MM,
            **surface(node),
        )
        for order, node in enumerate(contact_nodes)
    )
    vector = tuple(current)
    step = None
    for index in range(1, LOAD_STEPS + 1):
        load = PRETENSION_N + (PULL_N - PRETENSION_N) * index / LOAD_STEPS
        step = advance_contacts_quasistatic(
            registry_without_stick=registry(load), context=context,
            contact_layout=layout, contacts=contacts, vector=vector,
            fixed_indices=fixed, residual_tol_n=RESIDUAL_TOL_N,
            max_iterations=200, max_passes=4, yield_tol_n=1.0e-7,
        )
        vector = step.state.vector
    assert step is not None
    return step, layout, stretch, cylinder, contact_nodes


def _scalar(_node: int) -> dict:
    return {"friction_coefficient": MU_ALONG}


def _ellipse(mu_along: float, mu_across: float, skew_rad: float):
    def surface(node: int) -> dict:
        return {
            "friction_ellipse": FrictionEllipseSpec(
                mu_along, mu_across, _tape_axis(node, skew_rad)
            )
        }

    return surface


@pytest.fixture(scope="module")
def capstan_runs():
    """五条绞盘跑一次，六道门共用——每条约0.2秒，重复跑没有意义。"""

    #: ``2⁻⁴⁰``相对＝9.09e-13：小到椭圆在物理上就是圆，大到`is_isotropic`为假，
    #: 于是**转交被掐掉、通用支被强制走一遍**。这是0068第4.1节
    #: "关掉转交、强走通用椭圆路径"那条门在步进器层的对应物。
    nudge = MU_ALONG * 2.0**-40
    return {
        "scalar": _run_capstan(_scalar),
        "isotropic_ellipse": _run_capstan(_ellipse(MU_ALONG, MU_ALONG, SKEW_RAD)),
        "nudged_ellipse": _run_capstan(_ellipse(MU_ALONG, MU_ALONG + nudge, SKEW_RAD)),
        "skewed_ellipse": _run_capstan(_ellipse(MU_ALONG, MU_ACROSS, SKEW_RAD)),
        "aligned_ellipse": _run_capstan(_ellipse(MU_ALONG, MU_ACROSS, 0.0)),
        "nudge": nudge,
    }


def _hex(values) -> list[str]:
    return [float(value).hex() for value in values]


# ------------------------------------------------------------------ 门一：逐位退化


def test_an_isotropic_ellipse_degenerates_bit_for_bit_in_the_single_slot_stepper() -> None:
    """``μ_∥ = μ_⊥``时单槽位步进器与旧标量路径**逐位**相同。

    判的是`float.hex()`**不是**``==``：0068第4.1节吃过那一课——
    ``-0.0 == 0.0``为真而两者的`canonical_bytes`不同，于是
    "``==``全过"可以掩盖一次真实的指纹变更。这里判的是IEEE-754位型。

    **必红方式**：把`friction.yield_excess_n`里那条``if ellipse.is_isotropic``
    短路删掉（强走通用支）——通用支多两次点积，第三位有效数字之后就不一样了。
    """

    common = dict(load_n=6.0, angle_rad=math.pi / 4.0, max_passes=1)
    scalar = _flat_step(surface={"friction_coefficient": FLAT_MU}, **common)
    ellipse = _flat_step(
        surface={"friction_ellipse": FrictionEllipseSpec(FLAT_MU, FLAT_MU, (1.0, 0.0, 0.0))},
        **common,
    )
    assert scalar.regime == 2.0, "这道门要的是**滑移**分支；粘着分支上两条路都没做事"
    assert _hex(ellipse.state.vector) == _hex(scalar.state.vector)
    assert _hex(ellipse.tangential_force_n) == _hex(scalar.tangential_force_n)
    assert ellipse.regime == scalar.regime
    assert ellipse.slip_increment_mm.hex() == scalar.slip_increment_mm.hex()
    assert ellipse.yield_excess_n.hex() == scalar.yield_excess_n.hex()
    assert ellipse.passes == scalar.passes


def test_an_isotropic_ellipse_degenerates_bit_for_bit_in_the_multi_slot_stepper(
    capstan_runs,
) -> None:
    """多槽位同一条，跑的是**八个接触＋六十个载荷步＋每步四趟**的绞盘。

    单槽位那条只走一趟一个点；这条把"逐位"拉到一条真实链路上——
    退化若只在第一趟成立，六十步之后一定看得见。
    """

    scalar = capstan_runs["scalar"][0]
    ellipse = capstan_runs["isotropic_ellipse"][0]
    assert set(ellipse.regime) == {2.0}, "八个点都要在滑移分支上，否则这道门没碰到映射"
    assert _hex(ellipse.state.vector) == _hex(scalar.state.vector)
    assert [_hex(f) for f in ellipse.tangential_force_n] == [
        _hex(f) for f in scalar.tangential_force_n
    ]
    assert _hex(ellipse.normal_force_n) == _hex(scalar.normal_force_n)
    assert _hex(ellipse.slip_increment_mm) == _hex(scalar.slip_increment_mm)
    assert ellipse.regime == scalar.regime
    assert ellipse.max_yield_excess_n.hex() == scalar.max_yield_excess_n.hex()
    assert ellipse.passes == scalar.passes


def test_not_passing_an_ellipse_leaves_the_old_path_byte_for_byte(capstan_runs) -> None:
    """**判据本身被验**：上一条判的是"转交路径与旧路径同"，
    这一条判"不给椭圆时走的确实还是旧那串代码"。

    做法是把同一个绞盘用旧签名（只给``friction_coefficient``）跑一遍，
    与`cases/capstan_tension_ratio`同参数下的逐节点张力比对齐——
    那条比值是既有产物的指纹面。
    """

    step, _, stretch, _, _ = capstan_runs["scalar"]
    forces = stretch.axial_force_n(step.state)
    ratios = [forces[i + 1] / forces[i] for i in range(len(forces) - 1)]
    #: 2026-08-17实测（8段、60步）：1.1085677512897048 … 1.1287788684931237。
    #: 它是`cases/capstan_tension_ratio`那条判据的同一个量，**接线不许动它**。
    assert ratios[0] == pytest.approx(1.1085677512897048, rel=1.0e-15)
    assert ratios[-1] == pytest.approx(1.1287788684931237, rel=1.0e-15)


# -------------------------------------------------------------- 门二：通用支也退化


def test_the_general_ellipse_path_also_degenerates_when_the_handoff_is_switched_off(
    capstan_runs,
) -> None:
    """把转交掐掉（``μ_⊥ = μ_∥·(1 + 2⁻⁴⁰)``）——**偏差是被扰动带出来的**。

    只判"通用支给的数很接近"是不够的：一个写错的通用支也可能凑巧很接近。
    所以判的是**响应比**：状态的相对偏差与μ扰动的相对幅度同量级。
    2026-08-17实测扫四档（``2⁻³⁶/2⁻³⁸/2⁻⁴⁰/2⁻⁴²``）：

    | 扰动相对 | 状态相对偏差 | 比 |
    |---|---|---|
    | 1.4552e-11 | 4.6198e-12 | 0.318 |
    | 3.6380e-12 | 8.2647e-13 | 0.227 |
    | 9.0949e-13 | **5.8561e-13** | 0.644 |
    | 2.2737e-13 | 3.1641e-13 | 1.392 |

    偏差随扰动一起下降到``3e-13``附近就不再降——**那是求解器容差的地板
    （``residual_tol_n = 1e-6 N``对20 N量级的力）不是通用支的误差**，
    最后一档的比值大于1正是这条地板露出来的样子。

    **必红方式**：把`anisotropic_return_map`第三节的最近点投影换成径向缩
    （0068第七节第一条变异），本用例的偏差从1e-13跳到1e-2量级。
    """

    scalar = capstan_runs["scalar"][0]
    nudged = capstan_runs["nudged_ellipse"][0]
    relative = capstan_runs["nudge"] / MU_ALONG
    worst = max(
        abs(a - b) / abs(b)
        for a, b in zip(nudged.state.vector, scalar.state.vector, strict=True)
        if b != 0.0
    )
    assert worst <= 1.0e-11, worst
    assert worst >= 0.1 * relative, (
        f"偏差{worst:.3e}比扰动{relative:.3e}小一个数量级以上——"
        "**通用支多半根本没被走到**（`is_isotropic`把它转交掉了？）"
    )


# ---------------------------------------------------------------- 门三：走整条路


def test_a_planar_capstan_cannot_tell_the_two_yield_surfaces_apart(capstan_runs) -> None:
    """**判据本身被验，而且它救了门三**。

    平面绞盘上``z``全钉住 ⟹ 接触法向恒在``xy``面内、面外轴恒为``ẑ``、
    试探力的``z``分量**恒等于零**（`TangentialStickSpring`扣掉法向之后
    剩下的就是周向那一格）。于是试探力永远落在椭圆的主轴上，而
    0068第二节写着主轴是最近点投影与径向缩重合的那个例外。

    实测：``μ_∥:μ_⊥ = 5:1``的椭圆与``μ = μ_∥``的圆，
    整条状态向量**逐位相同**。

    **所以"绞盘上椭圆也跑通了"这句话本身不构成任何证据。**
    门三必须把纵向轴转开（`SKEW_RAD`），这一条就是那个必要性的证明。
    """

    scalar = capstan_runs["scalar"][0]
    aligned = capstan_runs["aligned_ellipse"][0]
    assert _hex(aligned.state.vector) == _hex(scalar.state.vector)


def test_the_ellipse_really_runs_through_solve_equilibrium_and_changes_it(
    capstan_runs,
) -> None:
    """**门三**：椭圆经过`solve_equilibrium`跑出一个平衡解，**并且改了它**。

    椭圆只经由**锚点**影响求解——粘着弹簧的刚度是各向同性的``k_t·I``，
    映射改的是锚点，而锚点是下一趟装配粘着弹簧的输入。所以
    "椭圆进了求解"这句话的可观测形式就是：**位形与张力真的变了**。

    2026-08-17实测（8段、60步、偏斜45°、``μ_∥:μ_⊥ = 5:1``）：

    | 量 | 圆（``μ = μ_∥``） | 椭圆 |
    |---|---|---|
    | 逐节点张力比（首/末） | 1.108568 / 1.128779 | **1.078824 / 1.101176** |
    | 首节点张力 | 12.690751 N | **15.377579 N** |
    | 位置最大差 | —— | **2.3261e-3 mm** |

    张力比降下来是**对的方向**：偏斜45°时沿滑移方向的有效摩擦是
    支撑函数``h = N·√(μ_∥²cos²β + μ_⊥²sin²β) = N·√0.13 = 0.36056·N``，
    比``μ_∥ = 0.5``小，于是绞盘攒不起那么多张力。

    **必红方式**：`_return_map`的椭圆支改回`coulomb_return_map`
    （拿``ellipse.mu_along``当标量）——张力比回到圆那一列，本用例当场红。
    """

    scalar, _, scalar_stretch, _, _ = capstan_runs["scalar"]
    skewed, layout, stretch, _, _ = capstan_runs["skewed_ellipse"]

    assert set(skewed.regime) == {2.0}
    assert skewed.max_yield_excess_n <= 1.0e-7 * 20.0 or skewed.passes == 4

    circle = scalar_stretch.axial_force_n(scalar.state)
    ellipse = stretch.axial_force_n(skewed.state)
    circle_ratios = [circle[i + 1] / circle[i] for i in range(len(circle) - 1)]
    ellipse_ratios = [ellipse[i + 1] / ellipse[i] for i in range(len(ellipse) - 1)]
    assert circle_ratios[0] == pytest.approx(1.1085677512897048, rel=1.0e-12)
    assert ellipse_ratios[0] == pytest.approx(1.0788240201895256, rel=1.0e-12)
    assert ellipse_ratios[-1] == pytest.approx(1.101176490311631, rel=1.0e-12)
    #: 每一格都必须变小——**一格变小可能是噪声，八格同向不是**。
    for anisotropic, isotropic in zip(ellipse_ratios, circle_ratios, strict=True):
        assert anisotropic < isotropic

    count = layout.layout.node_dof_count
    shift = max(
        abs(a - b)
        for a, b in zip(skewed.state.vector[:count], scalar.state.vector[:count], strict=True)
    )
    assert shift == pytest.approx(2.3260943331969604e-3, rel=1.0e-6)


def test_the_slipping_force_satisfies_the_support_function_after_the_whole_stepper(
    capstan_runs,
) -> None:
    """0068在**映射**上验过的支撑函数恒等式，这里在**整条步进器之后**再验一次。

    ``f·m̂ = h(m̂) = √((a·m̂_∥)² + (b·m̂_⊥)²)``，``m̂``取椭圆在``f``处的外法向
    （关联流动方向）。它对方向误差是二阶不敏感的（0068第3.2节），
    所以它守的是"力的大小落在屈服面上"而不是"方向对不对"。

    实测最大相对差**1.1549e-12**。它比0068在映射层量到的1.9e-15大三个数量级，
    **而那个差额有名字**：这里的椭圆是拿求解**之后**的法向与纵轴复原出来的，
    而步进器用的是**装配时**的那一个；两者之间法向转了一点。
    逐节点看得很清楚：从入口的1.3e-14单调涨到出口的1.2e-12，
    **单调性本身就说明它是"转了多少"而不是"算错了多少"**。
    """

    step, layout, _, cylinder, contact_nodes = capstan_runs["skewed_ellipse"]
    state = step.state
    normals = cylinder.outward_normal(state)
    normal_forces = cylinder.normal_force_n(state)
    worst = 0.0
    for order, node in enumerate(contact_nodes):
        surface = FrictionEllipse(
            MU_ALONG, MU_ACROSS, _tape_axis(node, SKEW_RAD)(state.vector), normals[order]
        )
        axis_along, axis_across = surface.in_plane_axes()
        force = step.tangential_force_n[order]
        along = sum(force[a] * axis_along[a] for a in range(3))
        across = sum(force[a] * axis_across[a] for a in range(3))
        semi_along = MU_ALONG * normal_forces[order]
        semi_across = MU_ACROSS * normal_forces[order]
        outward = (along / semi_along**2, across / semi_across**2)
        length = math.hypot(*outward)
        unit = (outward[0] / length, outward[1] / length)
        support = math.hypot(semi_along * unit[0], semi_across * unit[1])
        worst = max(worst, abs((along * unit[0] + across * unit[1]) / support - 1.0))
    assert worst <= 5.0e-12, worst


def test_a_fixed_normal_puts_the_force_exactly_on_the_ellipse() -> None:
    """法向不转时``Φ = 1``落到机器精度——**上一条那个1e-12确实是法向转出来的**。

    半空间上法向恒为``ẑ``，于是事后复原的椭圆与步进器用的**是同一个**。
    实测``Φ − 1 = 2.2204e-16``（恰好一个ulp；0068给的η迭代残差地板是``8·eps``量级）。
    """

    load, angle = 6.0, math.pi / 4.0
    step = _flat_step(
        load_n=load,
        angle_rad=angle,
        surface={"friction_ellipse": FrictionEllipseSpec(MU_ALONG, MU_ACROSS, (1.0, 0.0, 0.0))},
    )
    assert step.regime == 2.0
    surface = FrictionEllipse(MU_ALONG, MU_ACROSS, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    axis_along, axis_across = surface.in_plane_axes()
    force = step.tangential_force_n
    along = sum(force[a] * axis_along[a] for a in range(3))
    across = sum(force[a] * axis_across[a] for a in range(3))
    semi_along = MU_ALONG * step.normal_force_n
    semi_across = MU_ACROSS * step.normal_force_n
    quadratic = (along / semi_along) ** 2 + (across / semi_across) ** 2
    assert abs(quadratic - 1.0) <= 1.0e-15, quadratic


def test_swapping_the_two_coefficients_gives_a_different_answer() -> None:
    """**反向门**（0068第4.2节第3条的步进器版）：交换``μ_∥``与``μ_⊥``必须换答案。

    前面那些门会被一个"把``along_direction``整个忽略掉"的实现全部通过——
    只有这一条能验出纵向轴真的被用上了。
    """

    common = dict(load_n=6.0, angle_rad=math.pi / 3.0)
    one = _flat_step(
        surface={"friction_ellipse": FrictionEllipseSpec(MU_ALONG, MU_ACROSS, (1.0, 0.0, 0.0))},
        **common,
    )
    other = _flat_step(
        surface={"friction_ellipse": FrictionEllipseSpec(MU_ACROSS, MU_ALONG, (1.0, 0.0, 0.0))},
        **common,
    )
    assert one.regime == other.regime == 2.0
    difference = max(
        abs(a - b) for a, b in zip(one.tangential_force_n, other.tangential_force_n, strict=True)
    )
    assert difference > 1.0, difference


# ------------------------------------------------------------------ 门四：失败关闭


@pytest.mark.parametrize(
    "surface",
    [
        {},
        {
            "friction_coefficient": FLAT_MU,
            "friction_ellipse": FrictionEllipseSpec(FLAT_MU, FLAT_MU, (1.0, 0.0, 0.0)),
        },
    ],
    ids=["neither", "both"],
)
def test_the_single_slot_stepper_refuses_zero_or_two_yield_surfaces(surface) -> None:
    """两个都不给＝没有屈服面；两个都给＝两份μ而哪份说了算只能靠读实现。

    **两个都给那一支比不给更该拒**：它会静默地按某一个跑，
    而调用方以为按另一个跑——`PenaltyAnnulusLimit`那次端到端事故同一形态。
    """

    with pytest.raises(ContactError, match="恰好给一个"):
        _flat_step(load_n=6.0, angle_rad=0.0, surface=surface)


@pytest.mark.parametrize(
    "surface",
    [
        {},
        {
            "friction_coefficient": FLAT_MU,
            "friction_ellipse": FrictionEllipseSpec(FLAT_MU, FLAT_MU, (1.0, 0.0, 0.0)),
        },
    ],
    ids=["neither", "both"],
)
def test_a_contact_point_refuses_zero_or_two_yield_surfaces(surface) -> None:
    """多槽位那条在**构造期**就拒——一个点的屈服面说不清，不必等到跑起来。"""

    layout = build_contact_layout(
        layout_id="layout/ellipse-point",
        node_count=1,
        declarations=(ContactDeclaration("only"),),
    )
    with pytest.raises(ContactError, match="恰好给一个"):
        ContactPoint(
            slot=layout.slot_of("only"),
            node=0,
            normal=(0.0, 0.0, 1.0),
            normal_force_of=lambda state: 1.0,
            tangential_stiffness_n_per_mm=1.0,
            **surface,
        )


# --------------------------------------------------------- 屈服残差那个量本身


def test_the_yield_excess_reduces_to_the_cone_formula_bit_for_bit() -> None:
    """``μ_∥ = μ_⊥``时``yield_excess_n``**逐位**等于``|f| − μN``。

    这一条单独立，是因为它是门一逐位成立的**唯一**非平凡依赖：
    `anisotropic_return_map`的转交是0068已经守住的，残差这一半是新加的。
    """

    surface = FrictionEllipse(FLAT_MU, FLAT_MU, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    for trial in ((3.0, 4.0, 0.0), (0.7, -0.2, 0.0), (0.0, 0.0, 0.0)):
        for normal_force in (0.0, 1.0, 9.81):
            measured = yield_excess_n(
                trial_force_n=trial, normal_force_n=normal_force, ellipse=surface
            )
            expected = (
                math.sqrt(sum(value * value for value in trial)) - FLAT_MU * normal_force
            )
            assert measured.hex() == expected.hex()


def test_the_yield_excess_is_negative_inside_the_ellipse_and_zero_on_it() -> None:
    """粘着时≤0、恰好落在屈服面上时＝0——这是收敛判据能用的前提。"""

    surface = FrictionEllipse(MU_ALONG, MU_ACROSS, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    normal_force = 4.0
    semi_along, semi_across = MU_ALONG * normal_force, MU_ACROSS * normal_force
    inside = yield_excess_n(
        trial_force_n=(0.5 * semi_along, 0.0, 0.0),
        normal_force_n=normal_force,
        ellipse=surface,
    )
    assert inside == pytest.approx(-0.5 * semi_along, rel=1.0e-15)
    for direction in ((1.0, 0.0), (0.0, 1.0), (math.sqrt(0.5), math.sqrt(0.5))):
        on = yield_excess_n(
            trial_force_n=(semi_along * direction[0], semi_across * direction[1], 0.0),
            normal_force_n=normal_force,
            ellipse=surface,
        )
        assert abs(on) <= 1.0e-15, direction
    outside = yield_excess_n(
        trial_force_n=(3.0 * semi_along, 0.0, 0.0),
        normal_force_n=normal_force,
        ellipse=surface,
    )
    assert outside == pytest.approx(2.0 * semi_along, rel=1.0e-15)


def test_a_zero_trial_force_reports_the_shortest_half_axis_as_the_margin() -> None:
    """零试探力时"沿试探力方向"没有内容，取**最短**半轴。

    取最长会报出一个比真实余量大的数，而这个量是收敛判据——
    **判据宁可保守不可乐观**。
    """

    surface = FrictionEllipse(MU_ALONG, MU_ACROSS, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    margin = yield_excess_n(
        trial_force_n=(0.0, 0.0, 0.0), normal_force_n=4.0, ellipse=surface
    )
    assert margin == -MU_ACROSS * 4.0
