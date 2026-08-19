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

## 2026-08-18（决策0088轨丁）：放线端那个30 N不再是写死的

丁1把S6.3的欠账第一句接上了：``drives``的张力回路跑到稳态，
它的输出经`PointLoad`真的加载在带材上，与导向轮的罚接触一起进
`solve_equilibrium`。判的是**两条独立的腿在落位点合拢**——
驱动链那条只有一个换算比、求解器那条只有接触与摩擦，两条不共享任何一行代码。

**原有的七条门一个字未动**，新加的四条（三道正门＋一道必须红）在文件末尾。
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
from physics_engine.drives import (
    MagneticParticleClutch,
    PidController,
    SpoolTension,
    TensionLoop,
    capstan_transfer_ratio,
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

#: 丁1：驱动链的三件（决策0088）。参数与`tests/test_drives.py`逐字相同——
#: POC-050基型外推的23256 N·mm/A与50000 N·mm、假设的0.05 s磁滞
#: （0062第二节裁决2那张"只有现场实测能补"的清单）。
CLUTCH = MagneticParticleClutch(
    torque_per_ampere_nmm=23256.0, rated_torque_nmm=50000.0, lag_s=0.05
)
SPOOL = SpoolTension(barrel_radius_mm=60.0, tape_thickness_mm=0.1)
#: 闭环增益``K = k_M/R``（N/A）与临界阻尼下的``Ki = 1/(4ζ²τK)``。
LOOP_GAIN_N_PER_A = 23256.0 / 60.0
LOOP_INTEGRAL_GAIN = 1.0 / (4.0 * 1.0 * 1.0 * 0.05 * LOOP_GAIN_N_PER_A)
LOOP_DT_S = 1.0e-3
LOOP_STEPS = 3000
#: **传感器在放线端、被控点在落位点**，中间隔着这一只轮的包角。
#: ``measurement_transfer = T_传感器/T_被控点 = exp(−μθ)``。
#: 没有默认值是`TensionLoop`有意为之的：写``1.0``等于声明"中间一个包角都没有"。
MEASUREMENT_TRANSFER = 1.0 / capstan_transfer_ratio(
    friction_coefficient=FRICTION, wrap_angle_rad=WRAP_RAD
)
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


def _assemble(
    traverse_mm: float,
    payout_tension_n: float = PAYOUT_TENSION_N,
    extra_terms: tuple = (),
):
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
    rest = chord / (1.0 + payout_tension_n / EA_N)
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
    #: 丁1之后它可以由`drives.TensionLoop`的稳态输出给（见本文件末两条门），
    #: 而本函数**看不出区别**——它拿到的就只是一个数。那正是分辨力那条门要判的。
    pull = (0.0, -payout_tension_n, 0.0)
    #: ``extra_terms``只为**必须红**那一条存在（见本文件末）：它让"接线时顺手
    #: 多加了一项"这件事可以被真的做出来一次，于是逐位门证明得了自己会红。
    #: 默认``()``时元组拼接是恒等式，**默认路径逐位不变**。
    registry = EnergyRegistry(
        terms=(stretch, roller, flanges, PointLoad(loads=((0, pull),))) + extra_terms
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


def _run(
    traverse_mm: float = 0.0,
    steps: int = WIND_STEPS,
    wind_mm: float = WIND_MM,
    payout_tension_n: float = PAYOUT_TENSION_N,
    extra_terms: tuple = (),
):
    (layout, context, registry, stretch, roller, flanges,
     contact_nodes, vector, fixed, lay, chord) = _assemble(
        traverse_mm, payout_tension_n, extra_terms
    )
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


# ---------------------------------------------------------------------------
# 丁1：张力真的进求解器（决策0088，兑现S6.3的``missing``第一句）
# ---------------------------------------------------------------------------


def _tension_loop(measurement_transfer: float = MEASUREMENT_TRANSFER) -> TensionLoop:
    """真机形状的张力回路：磁粉离合器 ＋ 卷径换算 ＋ 纯积分PID。

    **纯积分**（``Kp = Kd = 0``）是有意的：它精确是一个二阶系统，
    稳态无静差可以论证而不是观测（`tests/test_drives.py`那一族门的口径）。
    ``sensor=None``也是有意的——量化台阶会让稳态停在台阶上，
    那条误差本身有它自己的门（S6.4），**在这里它只会污染分辨力那一条**。
    """

    return TensionLoop(
        clutch=CLUTCH,
        spool=SPOOL,
        controller=PidController(
            proportional=0.0,
            integral_gain=LOOP_INTEGRAL_GAIN,
            derivative=0.0,
            integral_limit=1.0e6,
        ),
        setpoint_n=PAYOUT_TENSION_N,
        dt_s=LOOP_DT_S,
        delay_line=None,
        sensor=None,
        measurement_transfer=measurement_transfer,
    )


def _settled_loop(measurement_transfer: float = MEASUREMENT_TRANSFER) -> TensionLoop:
    """跑到稳态的回路。**返回的是回路本身**，落位点张力是它的`tension_n`。"""

    settled, _ = _tension_loop(measurement_transfer).run(LOOP_STEPS)
    return settled


def _payout_load_from_drives(loop: TensionLoop) -> float:
    """回路 → 放线端的力边界条件。

    回路的`tension_n`是**被控点**（落位点）的张力；传感器装在放线端，
    于是放线端的张力是``T_被控点 · measurement_transfer``。
    **这一步是本片全部的"接线"**：从这里往后，求解器拿到的只是一个数。
    """

    return loop.tension_n * loop.measurement_transfer


@pytest.fixture(scope="module")
def drives_driven():
    """放线端外载**由张力回路给**的那一次运行。"""

    loop = _settled_loop()
    payout = _payout_load_from_drives(loop)
    return loop, payout, _run(0.0, payout_tension_n=payout)


@pytest.mark.batch
def test_the_payout_load_comes_from_the_drives_loop_and_the_two_legs_agree(drives_driven):
    """**S6.3的``missing``第一句在这里被兑现**：`drives`产出的力经`PointLoad`
    真的加载在带材上，并与导向轮的罚接触和库仑摩擦一起进`solve_equilibrium`。

    ## 判的是两条独立的腿对不对得上，不是"跑通了"

    * **驱动链那条腿**：电流→扭矩（线性段＋饱和＋零阶保持磁滞）→``T = M/R(n)``，
      再经``measurement_transfer``把被控点张力换算到传感器位置。
      它全程**没有一个几何量**，只有一个换算比；
    * **求解器那条腿**：把放线端那个力加上去，让8个接触在收线的连续化里
      同时滑移到摩擦锥上，落位点的轴力是解出来的。

    两条给出同一个落位点张力，而它们**不共享任何一行代码**。
    一条腿把``exp``写成``exp(−·)``、或者把``T = M/R``写成``T = M·R``，
    这条门当场红。

    ## 实测（2026-08-18）

    | 量 | 值 |
    |---|---|
    | 回路稳态读数（传感器＝放线端） | 29.999641126347463 N |
    | 回路自己算的被控点张力 | 48.05875685149555 N |
    | 求解器解出的落位点轴力 | 见断言，相对闭式1.7e-3 |
    """

    loop, payout, (step, stretch, *_) = drives_driven
    entry = _oracle("oracle:line/the_drives_loop_sets_the_payout_load")
    #: 一、回路真的收敛到设定值——**它红了说明回路没收敛，不是求解器不对**。
    assert payout == pytest.approx(
        entry.expected["sensor_tension_n"],
        rel=entry.tolerances["sensor_tension_n"].rel_tol,
    )
    #: 二、回路自己算的被控点张力＝设定值·exp(μθ)。**纯驱动链，无几何。**
    assert loop.tension_n == pytest.approx(
        entry.expected["controlled_point_tension_n"],
        rel=entry.tolerances["sensor_tension_n"].rel_tol,
    )
    #: 三、求解器解出来的落位点轴力与上一条对上。**两条腿在这里合拢。**
    tensions = stretch.axial_force_n(step.state)
    assert tensions[0] == pytest.approx(payout, rel=1.0e-9), (
        "第一段轴力必须恰是外载——它是边界条件不是解出来的量"
    )
    assert tensions[-1] == pytest.approx(
        loop.tension_n,
        rel=entry.tolerances["controlled_point_tension_n"].rel_tol,
    )
    assert tensions[-1] == pytest.approx(
        entry.expected["controlled_point_tension_n"],
        rel=entry.tolerances["controlled_point_tension_n"].rel_tol,
    )


@pytest.mark.batch
def test_dropping_the_drives_path_and_handing_over_the_same_number_is_bit_for_bit(
    drives_driven,
):
    """**分辨力**：把`drives`那一路整个拿掉、直接把同一个浮点数当力边界条件，
    整条状态向量**逐位相同**（`float.hex()`，不是`pytest.approx`）。

    ## 这条门证明什么、不证明什么

    **证明**：`drives`接进来的**只有那一个数**。它没有顺手往registry里塞第二个
    能量项、没有改动任何一个节点的自由度划分、没有改任何一段的自然长度以外的东西。
    一个"顺手把离合器扭矩也当成一个力矩加上去"的实现在这里当场红。

    **不证明**：那个数本身对不对。那是上一条门的活。
    **两条门必须都在**——只有这一条等于验了个恒等式，只有上一条则放得过
    "在装配里偷偷再加一项"这类错误。

    ## 为什么判`float.hex()`

    0068吃过``-0.0 == 0.0``为真而`canonical_bytes`不同那一课（0072第1.2节引）。
    "逐位"这个词在本仓只有一个意思，就是IEEE-754位型相同。
    """

    _, payout, (step, *_) = drives_driven
    plain = _run(0.0, payout_tension_n=payout)[0]
    assert [value.hex() for value in plain.state.vector] == [
        value.hex() for value in step.state.vector
    ], "同一个数、两条路，状态向量却不逐位相同——`drives`那一路带进来的不只是那个力"


@pytest.mark.batch
def test_flipping_the_measurement_transfer_squares_the_error():
    """**必须红的那一半**：传递比方向搞反时误差是**平方**——``exp(2μθ) = 2.566``。

    ``measurement_transfer = T_传感器/T_被控点``。传感器在放线端、
    被控点在落位点时它是``exp(−μθ)``；写成``exp(+μθ)``就等于声明
    "传感器在落位点、被控点在放线端"，而回路会照着这条错误声明去调。

    这条不用跑求解器（回路的稳态是闭式），**所以它是秒级的**。
    它守的是S6.4能力位那句"方向搞反误差是平方（r² = 2.566）"——
    那句话在0062里只是一段说明，从这里起它是一道门。
    """

    entry = _oracle("oracle:line/the_drives_loop_sets_the_payout_load")
    right = _settled_loop().tension_n
    flipped = _settled_loop(1.0 / MEASUREMENT_TRANSFER).tension_n
    assert right / flipped == pytest.approx(
        entry.expected["direction_flip_ratio"], rel=1.0e-4
    ), "方向搞反的比值不是exp(2μθ)——传递比不再是T_传感器/T_被控点了"
    #: 顺带钉住"这个数就是2.566"，免得闭式与实现一起漂。
    assert entry.expected["direction_flip_ratio"] == pytest.approx(2.5663, rel=1.0e-4)


@pytest.mark.batch
def test_one_extra_load_on_the_payout_node_makes_the_bit_for_bit_gate_red(drives_driven):
    """**上一条逐位门的"必须红"**：接线时顺手多加一项，它就当场红。

    多加的那一项**只有放线端外载的万分之一**（3.0 mN）。取这么小是有意的：
    逐位门要证明的是它连一份小到看不出来的多余载荷都拦得住，而不是只拦得住灾难。

    ## 起草这条门时撞到的两件事，两件都留下

    **一、`EnergyRegistry`本来就不收同名项。** 第一版的额外项用了`PointLoad`
    的默认名，registry当场拒：``duplicate energy term names: ['point_load']``。
    于是"顺手再加一个`PointLoad`"这个最直白的错法根本走不到求解器——
    要犯这个错，得先绕过一道已经存在的门。**那条护栏记在这里。**

    **二、这条链路的准静态连续化对收线步数不是单调的。** 起草时为了省时间
    把步数压到4，结果：干净的4步**过**、加了扰动的4步**不收敛**；
    再试干净的20步**也不收敛**、60步又过。**4过、20红、60过、480过**——
    非单调。所以本门老老实实用与被验运行相同的480步。
    这条已作为GAP登记在决策0088第五节。
    """

    _, payout, (clean, *_) = drives_driven
    surplus = payout * 1.0e-4
    dirty, dirty_stretch, *_ = _run(
        0.0, payout_tension_n=payout,
        extra_terms=(
            PointLoad(
                name="point_load/one_ten_thousandth_too_much",
                loads=((0, (0.0, -surplus, 0.0)),),
            ),
        ),
    )
    assert [v.hex() for v in clean.state.vector] != [
        v.hex() for v in dirty.state.vector
    ], "多加了一项而逐位门没红——那道门是空的"
    #: 顺带钉住"多出来的正是那一份"：第一段轴力从``payout``变成``payout + surplus``。
    assert dirty_stretch.axial_force_n(dirty.state)[0] == pytest.approx(
        payout + surplus, rel=1.0e-9
    )
