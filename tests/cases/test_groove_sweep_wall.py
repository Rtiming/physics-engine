"""conformance：外倾锥面槽壁 vs 平面环带（`cases/groove_sweep_wall`）。

判据正本是`cases/groove_sweep_wall/oracle.json`，闭式出处在
`cases/groove_sweep_wall/generate_oracle.py`的模块docstring。
本文件只做三件事：把内核摆到清单说的构型上、取数、按清单的容差比。

**内核的数拿来撞闭式，不是反过来**（spec/08规则1）。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import (
    PenaltyGrooveSweep,
    PenaltyNormalContact,
)
from physics_engine.energies import EnergyContext, EnergyRegistry, PointLoad
from physics_engine.oracles import load_manifest
from physics_engine.solve import SolveError, solve_equilibrium
from physics_engine.state import State, StateField, StateLayout

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/groove_sweep_wall/oracle.json", root=ROOT)
ORACLES = {oracle.id.rsplit("/", 1)[1]: oracle for oracle in MANIFEST.oracles}

WIDTH_AXIS = (1.0, 0.0, 0.0)
NORMAL_AXIS = (0.0, 0.0, 1.0)
ORIGIN = (0.0, 0.0, 0.0)

LAYOUT = StateLayout(
    layout_id="layout/groove_sweep_wall",
    fields=(
        StateField("node0_x_mm", 1),
        StateField("node0_y_mm", 1),
        StateField("node0_z_mm", 1),
    ),
)
CONTEXT = EnergyContext(
    context_id="context/groove_sweep_wall",
    node_masses_kg=(1.0e-9,),
    gravity_mm_s2=(0.0, 0.0, 0.0),
)


def _constants():
    inputs = ORACLES["frozen_depth_2p2"].inputs
    residual = ORACLES["frozen_frame_residual"].inputs
    escape = ORACLES["escape_threshold"].inputs
    free = ORACLES["free_depth_2p2"].inputs
    return {
        "slope": math.tan(math.radians(inputs["wall_angle_deg"])),
        "half_width": 4.0,
        "radius": 2.0,
        "stiffness": 5.0e3,
        "depth_max": escape["depth_max_mm"],
        "hold": free["hold_down_n"],
        "residual": residual,
    }


CONST = _constants()


def _walls(slope: float) -> PenaltyGrooveSweep:
    return PenaltyGrooveSweep(
        walls=tuple(
            (
                0,
                ORIGIN,
                WIDTH_AXIS,
                NORMAL_AXIS,
                side,
                CONST["half_width"],
                slope,
                CONST["radius"],
                -1.0,
                CONST["depth_max"],
                CONST["stiffness"],
            )
            for side in (1.0, -1.0)
        )
    )


def _check(oracle, key: str, measured: float) -> None:
    """按清单自己的``Tolerance.holds``判——**判据正本在清单里，不在这里重写一遍**。"""

    expected = oracle.expected[key]
    tolerance = oracle.tolerances[key]
    assert tolerance.holds(measured, expected), (
        f"{oracle.id}/{key}: 实测{measured!r} 对 金标{expected!r}，"
        f"超出{tolerance.exceeded_by(measured, expected)!r}（{tolerance.reason}）"
    )


# ------------------------------------------- 一、深度冻结：差别全在举升上 ---


@pytest.mark.parametrize("tag", ("frozen_depth_2p2", "frozen_depth_2p5", "frozen_depth_3p0"))
def test_frozen_depth_lateral_force_is_identical_and_the_lift_is_not(tag: str) -> None:
    """深度钉住时锥面与平面的**横向力恒等**，锥面另有一个把带材举出槽的分量。

    这条判据存在的理由是它挡住一句说反了的话：
    "锥面的回正力更软"在深度被钉住时**是错的**。
    """

    oracle = ORACLES[tag]
    lateral = oracle.inputs["lateral_mm"]
    state = State(layout=LAYOUT, vector=(lateral, 0.0, oracle.inputs["depth_mm"]))
    plane = _walls(0.0)
    cone = _walls(CONST["slope"])

    #: 符号约定：力是``−∇U``。``+s``侧壁上``g < 0``，故``∇U``沿``+s``、
    #: 力沿``−s``（回正）。清单记的是**大小**，所以横向取``+gradient[0]``、
    #: 举升取``−gradient[2]``（``∇U``的``n``分量为负，力朝``+n``即出槽方向）。
    #: **两个分量各取各的符号，不是笔误**——写反了会把"举出槽"读成"压进槽"。
    _check(oracle, "gap_mm", plane.wall_clearance_mm(state)[0])
    _check(oracle, "plane_lateral_force_n", plane.gradient(state, CONTEXT)[0])
    _check(oracle, "cone_lateral_force_n", cone.gradient(state, CONTEXT)[0])
    _check(oracle, "cone_lift_force_n", -cone.gradient(state, CONTEXT)[2])
    _check(
        oracle,
        "cone_over_plane_magnitude",
        cone.wall_force_n(state)[0] / plane.wall_force_n(state)[0],
    )
    #: 横向分量**逐位**相同——这条不进清单容差，它是恒等式不是数值结果。
    assert cone.gradient(state, CONTEXT)[0].hex() == plane.gradient(state, CONTEXT)[0].hex()


# ---------------------------------- 二、深度自由：平面线性上升、锥面饱和 ---


def _solve_free_depth(slope: float, lateral: float):
    """带材被压紧力按在槽底、横向被钉在``lateral``，深度自由求平衡。"""

    registry = EnergyRegistry(
        terms=(
            _walls(slope),
            PenaltyNormalContact(planes=((0, ORIGIN, NORMAL_AXIS, CONST["stiffness"], 0.0),)),
            PointLoad(loads=((0, (0.0, 0.0, -CONST["hold"])),)),
        )
    )
    result = solve_equilibrium(
        registry,
        CONTEXT,
        LAYOUT,
        (lateral, 0.0, -CONST["hold"] / CONST["stiffness"]),
        fixed_indices=frozenset({0, 1}),
        residual_tol_n=1.0e-11,
        max_iterations=300,
    )
    assert result.converged, result.reason
    state = State(layout=LAYOUT, vector=result.state.vector)
    return state, _walls(slope).gradient(state, CONTEXT)[0], result.state.vector[2]


@pytest.mark.parametrize(
    "tag",
    ("free_depth_2p05", "free_depth_2p2", "free_depth_2p5", "free_depth_2p8", "free_depth_3p0"),
)
def test_free_depth_cone_saturates_while_the_plane_climbs_linearly(tag: str) -> None:
    """plans/15第2.2条那句判据的定量形式。

    锥面的横向力**与横移量无关**（饱和在``F_hold/tanα``），
    平面的按``k``线性上升；两者之比随横移线性发散。
    """

    oracle = ORACLES[tag]
    lateral = oracle.inputs["lateral_mm"]
    _, plane_force, plane_depth = _solve_free_depth(0.0, lateral)
    _, cone_force, cone_depth = _solve_free_depth(CONST["slope"], lateral)

    _check(oracle, "plane_lateral_force_n", plane_force)
    _check(oracle, "cone_lateral_force_n", cone_force)
    _check(oracle, "cone_depth_mm", cone_depth)
    _check(oracle, "plane_over_cone", plane_force / cone_force)
    #: 平面那一侧带材**没离开槽底**——罚柔度下沉``F_hold/k``而已。
    assert plane_depth == pytest.approx(-CONST["hold"] / CONST["stiffness"], rel=1.0e-9)


def test_the_cone_force_is_the_same_number_at_every_lateral_offset() -> None:
    """饱和是一句**关于整条曲线**的话，逐点判会漏掉"它是不是常数"。"""

    forces = [
        _solve_free_depth(CONST["slope"], lateral)[1] for lateral in (2.05, 2.2, 2.5, 2.8, 3.0)
    ]
    assert max(forces) - min(forces) == pytest.approx(0.0, abs=1.0e-9)


# ------------------------------------------------ 三、爬出槽口的横移阈值 ---


def test_the_escape_threshold_matches_the_closed_form() -> None:
    """超过阈值带材爬出槽口，**平衡问题本身失去良态**（求解器报奇异）。

    **这不是数值故障，是"带材跳出槽"这件事的物理**——壁不在了，
    深度自由度不再受任何能量项约束。案例页第四节把这一条写在已知失效清单里
    而不是藏起来。
    """

    oracle = ORACLES["escape_threshold"]
    low, high = 2.5, 4.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        try:
            _solve_free_depth(CONST["slope"], middle)
        except SolveError:
            high = middle
        else:
            low = middle
    assert high - low < 1.0e-6
    _check(oracle, "escape_lateral_mm", 0.5 * (low + high))


# ------------------------------------ 四、冻结帧丢掉的那一项，量出来是多少 ---


def _gold_frame(arc: float, radius: float, twist: float):
    angle = arc / radius
    phase = twist * arc
    tangent = (-math.sin(angle), math.cos(angle), 0.0)
    radial = (math.cos(angle), math.sin(angle), 0.0)
    axial = (0.0, 0.0, 1.0)
    normal = tuple(
        math.cos(phase) * radial[axis] + math.sin(phase) * axial[axis] for axis in range(3)
    )
    width = (
        normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2],
        normal[0] * tangent[1] - normal[1] * tangent[0],
    )
    return tangent, width, normal


def test_the_frozen_frame_residual_is_a_number_not_an_adjective() -> None:
    """`PenaltyGrooveSweep`冻结帧丢掉多少：**闭式与数值中心差分各算一遍**。

    数值那条腿在这里独立重算一次帧不变量（对解析帧做中心差分），
    与清单里那条由``κ⃗``投影推出来的闭式互不引用。
    收敛阶那一条在`tests/test_contact_groove_sweep.py`判，**两件事不混**。
    """

    oracle = ORACLES["frozen_frame_residual"]
    radius = oracle.inputs["radius_mm"]
    twist = math.radians(oracle.inputs["twist_deg_per_mm"])
    arc = oracle.inputs["arc_mm"]
    lateral = oracle.inputs["lateral_mm"]
    depth = oracle.inputs["depth_mm"]
    slope = CONST["slope"]

    step = 1.0e-6
    tangent, width, normal = _gold_frame(arc, radius, twist)
    ahead = _gold_frame(arc + step, radius, twist)
    behind = _gold_frame(arc - step, radius, twist)
    tangent_rate = tuple((ahead[0][a] - behind[0][a]) / (2.0 * step) for a in range(3))
    width_rate = tuple((ahead[1][a] - behind[1][a]) / (2.0 * step) for a in range(3))
    curvature_s = sum(tangent_rate[a] * width[a] for a in range(3))
    curvature_n = sum(tangent_rate[a] * normal[a] for a in range(3))
    measured_twist = sum(width_rate[a] * normal[a] for a in range(3))

    jacobian = 1.0 - lateral * curvature_s - depth * curvature_n
    coefficient = measured_twist * (slope * lateral + 1.0 * depth) / jacobian
    frozen_norm = math.sqrt(1.0 + slope * slope)

    #: 数值帧微分只有``O(h²)``＋舍入，放宽到1e-6相对；清单容差是给闭式自身的。
    assert curvature_s == pytest.approx(oracle.expected["curvature_s_per_mm"], rel=1.0e-6)
    assert curvature_n == pytest.approx(oracle.expected["curvature_n_per_mm"], rel=1.0e-5)
    assert measured_twist == pytest.approx(oracle.expected["twist_rad_per_mm"], rel=1.0e-6)
    assert jacobian == pytest.approx(oracle.expected["jacobian"], rel=1.0e-9)
    assert coefficient == pytest.approx(oracle.expected["coefficient_a"], rel=1.0e-6)

    _check(oracle, "frozen_gradient_norm", frozen_norm)
    assert abs(coefficient) / frozen_norm == pytest.approx(
        oracle.expected["relative_loss"], rel=1.0e-6
    )
    assert math.degrees(math.atan2(abs(coefficient), frozen_norm)) == pytest.approx(
        oracle.expected["force_tilt_deg"], rel=1.0e-6
    )
