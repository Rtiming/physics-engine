"""conformance：放线—导向—张力—收线端到端（`cases/winding_line_endtoend`，能力位S6.7）。

**这是本仓第一次把四件事装在一条链路上跑**：放线端的张力（力边界条件）、
导向轮上的罚接触与库仑摩擦（多槽位同时滑移）、收线端的位移控制（卷走带材）、
排线横动把带材边缘顶上法兰内环面。

## 混合控制，照真机来

真机上力在**放线端**（磁粉离合器给制动张力），位移在**收线端**（伺服卷走带材）。
第一版反着装——在放线端加力、把落位点钉死——结果张力**沿走向递减**，
比值0.5587，与绞盘公式反向。**方向不是判据能挑出来的，是构型定的。**

## 它跑出来一个真bug

`PenaltyAnnulusLimit`第一版把法兰朝向编码在限位坐标的**符号**里。
收线盘排线横动到9 mm时，下侧法兰的位置变成``+0.5``——符号翻了、方向反了、
**蹭边力凭空归零**（7 mm与8 mm都算得出2.46 N与7.44 N，唯独9 mm给0）。

单元门里的构型永远是槽心在原点的，那里位置符号与朝向恒等，
**所以只有端到端发现得了它**。已改成显式的``inward``字段并补了必红。

## 主分母没有因此变动

场景⑥七位里S6.3/S6.4/S6.5/S6.6都是partial，全done才算端到端完成。
**本案例让S6.7从todo进partial，主分母仍是0/6**——这不是谦虚，是清单的算法。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    ContactPoint,
    PenaltyAnnulusLimit,
    PenaltyCylinderContact,
    advance_contacts_quasistatic,
    build_contact_layout,
)
from physics_engine.drives import capstan_transfer_ratio
from physics_engine.energies import (
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    PointLoad,
)
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State

CASE = Path(__file__).resolve().parents[2] / "cases" / "winding_line_endtoend"
MANIFEST = load_manifest(CASE / "oracle.json")

ROLLER_RADIUS_MM = 50.0
ROLLER_HALF_WIDTH_MM = 8.5
WRAP_RAD = math.pi / 2.0
WRAP_SEGMENTS = 8
FRICTION = 0.30
EA_N = 60000.0
PENALTY_N_PER_MM = 1.0e4

SPOOL_RADIUS_MM = 60.0
CHANNEL_HALF_WIDTH_MM = 8.5
FLANGE_OUTER_MM = 75.0
TAPE_HALF_WIDTH_MM = 2.0
HALF_CLEARANCE_MM = CHANNEL_HALF_WIDTH_MM - TAPE_HALF_WIDTH_MM

FREE_SEGMENTS = 2
PAYOUT_TENSION_N = 30.0
#: 收线距离与步数：**0.02 mm / 480步**是实测选出来的。0.05 mm起内层牛顿
#: 在同一构型里走不完（判别翻转叠加大位移），0.02 mm在480步时比值1.604693、
#: 相对绞盘闭式1.695e-3。
WIND_MM = 0.02
WIND_STEPS = 480
RESIDUAL_TOL_N = 1.0e-6


def _oracle(oracle_id: str):
    for entry in MANIFEST.oracles:
        if entry.id == oracle_id:
            return entry
    raise AssertionError(f"清单里没有{oracle_id}")


def _lay_start_z(traverse_mm: float) -> float:
    """落位点``z``的起点：带材自然位置（0）投影进槽内。

    **起点必须已经接触但穿透很浅。** 从``z = 0``起步时法兰一上来就穿透
    ``|横动| − 半间隙``，罚力``k·δ``把节点甩飞、牛顿走不动。
    这与`test_contact_normal.py`记的"分离态切线刚度奇异"是同一族的另一面：
    **那边是刚度为零迈不出第一步，这边是刚度过大第一步就飞了。**
    """

    return max(
        traverse_mm - HALF_CLEARANCE_MM, min(traverse_mm + HALF_CLEARANCE_MM, 0.0)
    )


def _geometry(traverse_mm: float):
    """入口自由段 → 绕导向轮 → 出口自由段 → 落位点。

    导向轮轴沿``z``、心在原点；带材在``xy``平面内从``φ=0``绕到``φ=WRAP``。
    **带材的横向位置不随排线横动变**——横动的是收线盘（连带它的两片法兰）。
    """

    dphi = WRAP_RAD / WRAP_SEGMENTS
    chord = 2.0 * ROLLER_RADIUS_MM * math.sin(dphi / 2.0)
    points: list[tuple[float, float, float]] = []
    for step in range(FREE_SEGMENTS, 0, -1):
        points.append((ROLLER_RADIUS_MM, -step * chord, 0.0))
    wrap_start = len(points)
    for index in range(WRAP_SEGMENTS + 1):
        angle = index * dphi
        points.append(
            (ROLLER_RADIUS_MM * math.cos(angle), ROLLER_RADIUS_MM * math.sin(angle), 0.0)
        )
    wrap_end = len(points) - 1
    tip = points[wrap_end]
    tangent = (-math.sin(WRAP_RAD), math.cos(WRAP_RAD), 0.0)
    for step in range(1, FREE_SEGMENTS + 1):
        points.append(
            (
                tip[0] + tangent[0] * step * chord,
                tip[1] + tangent[1] * step * chord,
                0.0 if step < FREE_SEGMENTS else _lay_start_z(traverse_mm),
            )
        )
    return points, wrap_start, wrap_end, len(points) - 1, chord


def _assemble(traverse_mm: float):
    points, wrap_start, wrap_end, lay, chord = _geometry(traverse_mm)
    nodes = len(points)
    contact_nodes = list(range(wrap_start + 1, wrap_end + 1))
    layout = build_contact_layout(
        layout_id="layout/winding-line",
        node_count=nodes,
        declarations=tuple(ContactDeclaration(f"roll{i}") for i in contact_nodes),
    )
    context = EnergyContext(
        context_id="context/winding-line",
        node_masses_kg=(1.0e-9,) * nodes,
        gravity_mm_s2=(0.0, 0.0, 0.0),
    )
    rest = chord / (1.0 + PAYOUT_TENSION_N / EA_N)
    stretch = AxialStretch(
        edges=tuple((i, i + 1, rest, EA_N) for i in range(nodes - 1))
    )
    roller = PenaltyCylinderContact(
        cylinders=tuple(
            (i, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), ROLLER_RADIUS_MM,
             ROLLER_HALF_WIDTH_MM, PENALTY_N_PER_MM, 0.0)
            for i in contact_nodes
        )
    )
    #: **法兰随排线横动整体平移**，朝向用显式的``inward``——
    #: 把它从限位坐标的符号推出来，横动过原点时会翻（本文件开头那个bug）。
    spool_centre = (points[lay][0], points[lay][1] - SPOOL_RADIUS_MM, 0.0)
    flanges = PenaltyAnnulusLimit(
        faces=(
            (lay, spool_centre, (0.0, 0.0, 1.0), 0.0, FLANGE_OUTER_MM,
             traverse_mm + CHANNEL_HALF_WIDTH_MM, 1.0, TAPE_HALF_WIDTH_MM,
             PENALTY_N_PER_MM),
            (lay, spool_centre, (0.0, 0.0, 1.0), 0.0, FLANGE_OUTER_MM,
             traverse_mm - CHANNEL_HALF_WIDTH_MM, -1.0, -TAPE_HALF_WIDTH_MM,
             PENALTY_N_PER_MM),
        )
    )
    #: 放线端的张力：沿入口切线向外拉。**这是离合器给的力边界条件。**
    pull = (0.0, -PAYOUT_TENSION_N, 0.0)
    registry = EnergyRegistry(
        terms=(stretch, roller, flanges, PointLoad(loads=((0, pull),)))
    )
    vector = list(layout.initial_vector(tuple(c for p in points for c in p)))
    for node in contact_nodes:
        slot = layout.slot_of(f"roll{node}")
        vector[slot.anchor_base : slot.anchor_base + 3] = list(points[node])
    #: **混合控制**：放线端受力、自由；落位点``x``/``y``位移控制（收线盘卷走），
    #: **``z``自由**——它由张力的横向分量与法兰限位平衡决定，那才是物理接触力。
    #: 钉住``z``算出来的是"规定穿透量的反力"，第一版就是那样算出5000 N的。
    fixed = {3 * lay, 3 * lay + 1}
    fixed |= {3 * i + 2 for i in range(nodes) if i != lay}
    fixed |= set(range(layout.layout.node_dof_count, layout.layout.dof_count))
    return (layout, context, registry, stretch, roller, flanges,
            contact_nodes, tuple(vector), frozenset(fixed), lay, chord)


def _run(traverse_mm: float = 0.0, steps: int = WIND_STEPS, wind_mm: float = WIND_MM):
    (layout, context, registry, stretch, roller, flanges,
     contact_nodes, vector, fixed, lay, chord) = _assemble(traverse_mm)
    settled = solve_equilibrium(
        registry, context, layout.layout, vector,
        fixed_indices=fixed, residual_tol_n=1.0e-7, max_iterations=300,
    )
    assert settled.converged, settled.reason
    current = list(settled.state.vector)
    for node in contact_nodes:
        slot = layout.slot_of(f"roll{node}")
        current[slot.anchor_base : slot.anchor_base + 3] = current[
            3 * node : 3 * node + 3
        ]
    current = tuple(current)

    def normal_of(order: int):
        return lambda vec: roller.outward_normal(
            State(layout=layout.layout, vector=vec)
        )[order]

    contacts = tuple(
        ContactPoint(
            slot=layout.slot_of(f"roll{node}"), node=node, normal=normal_of(order),
            normal_force_of=(lambda st, o=order: roller.normal_force_n(st)[o]),
            tangential_stiffness_n_per_mm=PENALTY_N_PER_MM,
            friction_coefficient=FRICTION,
        )
        for order, node in enumerate(contact_nodes)
    )
    tangent = (-math.sin(WRAP_RAD), math.cos(WRAP_RAD), 0.0)
    step = None
    for _ in range(steps):
        moved = list(current)
        for axis in range(3):
            moved[3 * lay + axis] = current[3 * lay + axis] + tangent[axis] * wind_mm / steps
        step = advance_contacts_quasistatic(
            registry_without_stick=registry, context=context, contact_layout=layout,
            contacts=contacts, vector=tuple(moved), fixed_indices=fixed,
            residual_tol_n=RESIDUAL_TOL_N, max_iterations=200,
            max_passes=4, yield_tol_n=1.0e-7,
        )
        current = step.state.vector
    assert step is not None
    return step, stretch, roller, flanges, lay, chord


@pytest.fixture(scope="module")
def centred():
    """排线横动为零的基准运行，三条门共用。"""

    return _run(0.0)


@pytest.mark.batch
def test_the_payout_tension_is_exactly_the_applied_load(centred):
    """放线端的轴力恰是离合器给的张力——**它是边界条件不是解出来的量**。"""

    step, stretch, *_ = centred
    entry = _oracle("oracle:line/payout_tension_is_the_boundary_condition")
    tensions = stretch.axial_force_n(step.state)
    assert tensions[0] == pytest.approx(
        entry.expected["payout_tension_n"],
        rel=entry.tolerances["payout_tension_n"].rel_tol,
    )


@pytest.mark.batch
def test_the_lay_point_tension_is_the_payout_times_the_capstan_ratio(centred):
    """**审计缺漏A在真装配上的样子。**

    带材朝收线端滑，摩擦朝放线端，于是张力沿走向**递增**：
    ``T_落位 = T_放线 · exp(μθ)``。传感器装在放线端读30 N，
    而落位点上是**48 N**——闭环调的是它测到的量。

    **方向不是判据挑出来的，是构型定的**：第一版在放线端加力、把落位点钉死，
    张力反而沿走向递减（比值0.5587）。混合控制必须照真机来。
    """

    step, stretch, *_ = centred
    entry = _oracle("oracle:line/lay_point_tension_over_payout")
    tensions = stretch.axial_force_n(step.state)
    ratio = tensions[-1] / tensions[0]
    assert ratio == pytest.approx(
        entry.expected["capstan_ratio"],
        rel=entry.tolerances["lay_point_tension_n"].rel_tol,
    )
    assert tensions[-1] == pytest.approx(
        entry.expected["lay_point_tension_n"],
        rel=entry.tolerances["lay_point_tension_n"].rel_tol,
    )
    #: 与`drives.capstan_transfer_ratio`互钉——两处不共享实现。
    assert entry.expected["capstan_ratio"] == pytest.approx(
        capstan_transfer_ratio(friction_coefficient=FRICTION, wrap_angle_rad=WRAP_RAD),
        rel=1e-15,
    )


@pytest.mark.batch
def test_every_roller_contact_is_on_the_friction_cone(centred):
    """全滑移：每个导轮接触精确落在摩擦锥上。它是上一条成立的前提。"""

    step, _, roller, *_ = centred
    normals = roller.normal_force_n(step.state)
    for index, force in enumerate(step.tangential_force_n):
        magnitude = math.sqrt(sum(value * value for value in force))
        assert normals[index] > 0.0, f"接触{index}没有法向力"
        assert magnitude / (FRICTION * normals[index]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.batch
def test_the_tape_stays_clear_until_the_traverse_exceeds_the_half_clearance():
    """排线横动不到半间隙就**一点蹭边力都没有**，超过就有——零容差。"""

    entry = _oracle("oracle:line/half_clearance")
    assert HALF_CLEARANCE_MM == entry.expected["half_clearance_mm"]
    for traverse, rubs in ((0.0, False), (6.0, False), (6.5, False), (7.0, True)):
        step, _, _, flanges, lay, _ = _run(traverse)
        forces = flanges.rub_force_n(step.state)
        assert (max(forces) > 0.0) is rubs, (
            f"横动{traverse} mm处蹭边判定与预期相反：{forces}"
        )
        if not rubs:
            assert step.state.vector[3 * lay + 2] == pytest.approx(0.0, abs=1e-9), (
                "没蹭上时落位点该停在带材的自然横向位置"
            )


@pytest.mark.batch
def test_the_rub_force_is_the_lateral_component_of_the_tension():
    """**蹭边力正比于张力**：``F = T_落位 · (|横动| − 半间隙) / L``。

    2026-08-17实测：横动7 mm给2.4560 N（闭式2.4588）、8 mm给7.4422 N（闭式7.4507），
    两档都差**0.11%**——那是``sin θ ≈ θ``的``O((δ/L)²)``。

    这条的物理含义要紧：**张力调高，蹭边就更狠**。真机上"把张力开大一点"
    是操作员最容易做的调整，而它同时把蹭边力按比例放大。
    """

    entry = _oracle("oracle:line/rub_force_scales_with_tension")
    span = entry.expected["free_span_mm"]
    for traverse, key in ((7.0, "overshoot_at_seven_mm"), (8.0, "overshoot_at_eight_mm")):
        step, stretch, _, flanges, _, chord = _run(traverse)
        assert chord == pytest.approx(span, rel=1e-15)
        tensions = stretch.axial_force_n(step.state)
        closed_form = tensions[-1] * entry.expected[key] / span
        measured = max(flanges.rub_force_n(step.state))
        assert measured == pytest.approx(closed_form, rel=2.0e-3), (
            f"横动{traverse} mm：蹭边力{measured!r}与横向分量闭式{closed_form!r}对不上"
        )


@pytest.mark.batch
def test_the_rub_is_mirror_antisymmetric_across_the_traverse():
    """``±横动``给出**逐位相同**的蹭边力，只是换了一片法兰。

    **哪一片被顶住与单元门里相反**：那里动的是带材、这里动的是**槽**。
    槽往``+z``横动，等于带材相对往``−z``跑——顶住的是**下侧**那片
    （``faces[1]``）。第一版这里写反了，实测当场打红。
    **"镜像"这个词不指定是哪一片，构型才指定。**
    """

    plus = _run(7.0)
    minus = _run(-7.0)
    plus_forces = plus[3].rub_force_n(plus[0].state)
    minus_forces = minus[3].rub_force_n(minus[0].state)
    #: 槽向``+z``横动 → 下侧那片出力；向``−z`` → 上侧那片。
    assert plus_forces[1] == minus_forces[0] > 0.0
    assert plus_forces[0] == minus_forces[1] == 0.0


def test_the_traversed_channel_is_why_the_flange_direction_became_explicit():
    """**本案例跑出来的那个bug，在这里留一条不碰求解器的快门。**

    收线盘横动到9 mm时整条槽是``[0.5, 17.5]``，**两个限位坐标都是正的**。
    把朝向从符号推出来，两片法兰都变成"朝下"，下侧那片再也拦不住带材。

    这条门只判几何，秒级；完整的必红在`tests/test_contact_annulus.py`。
    """

    traverse = 9.0
    assert traverse - CHANNEL_HALF_WIDTH_MM > 0.0, "构型前提：整条槽在正半轴"
    assert traverse + CHANNEL_HALF_WIDTH_MM > 0.0
    explicit = PenaltyAnnulusLimit(
        faces=(
            (0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, FLANGE_OUTER_MM,
             traverse + CHANNEL_HALF_WIDTH_MM, 1.0, TAPE_HALF_WIDTH_MM, PENALTY_N_PER_MM),
            (0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, FLANGE_OUTER_MM,
             traverse - CHANNEL_HALF_WIDTH_MM, -1.0, -TAPE_HALF_WIDTH_MM, PENALTY_N_PER_MM),
        )
    )
    inferred = PenaltyAnnulusLimit(
        faces=tuple(
            (node, point, axis, inner, outer, limit,
             1.0 if limit > 0.0 else -1.0, offset, stiffness)
            for node, point, axis, inner, outer, limit, _, offset, stiffness
            in explicit.faces
        )
    )
    from physics_engine.state import StateField, StateLayout

    layout = StateLayout(
        layout_id="layout/traverse-probe",
        fields=(
            StateField("node0_x_mm", 1),
            StateField("node0_y_mm", 1),
            StateField("node0_z_mm", 1),
        ),
    )
    #: 带材停在自然位置``z = 0``，而槽的下界是``2.5``——该被下侧法兰顶住。
    state = State(layout=layout, vector=(60.0, 0.0, 0.0))
    assert max(explicit.rub_force_n(state)) > 0.0
    assert explicit.rub_force_n(state)[1] == pytest.approx(2.5 * PENALTY_N_PER_MM)
    assert inferred.rub_force_n(state) == (0.0, 0.0), (
        "旧写法在横动过原点的槽上不再给零——那本门记的病根就要重写"
    )
