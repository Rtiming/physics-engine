"""conformance：摩擦迟滞回线（`cases/friction_hysteresis_loop`）。

判据值一律从清单读，**不在本文件复述闭式**（轴7规则3）。

本案例是第一个**改写锚点**的——前一个接触案例是静置问题，碰不到这件事。
最要紧的一条是路径相关：**同一位置、两条路径、力必须不同**。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.contact import (
    ContactDeclaration,
    advance_contact_quasistatic,
    build_contact_layout,
)
from physics_engine.energies import EnergyContext, EnergyRegistry, UniformGravity
from physics_engine.oracles import load_manifest

CASE = Path(__file__).resolve().parents[2] / "cases" / "friction_hysteresis_loop"
MANIFEST = load_manifest(CASE / "oracle.json")

MASS_KG = 2.0
GRAVITY_MM_S2 = 9810.0
WEIGHT_N = MASS_KG * GRAVITY_MM_S2 / 1000.0
NORMAL = (0.0, 0.0, 1.0)

pytestmark = pytest.mark.batch


def _oracle(identifier: str):
    for oracle in MANIFEST.oracles:
        if oracle.id == identifier:
            return oracle
    raise AssertionError(f"清单里没有{identifier}")


class _Drag:
    """位移控制的拖拽装置：钉住``x``、``y``，放开``z``，逐步走一条位移路径。"""

    def __init__(self, *, friction_coefficient: float, normal_stiffness: float,
                 tangential_stiffness: float) -> None:
        self.contact_layout = build_contact_layout(
            layout_id="layout/friction_hysteresis_loop",
            node_count=1,
            declarations=(ContactDeclaration("block_ground"),),
        )
        self.context = EnergyContext(
            context_id="context/friction_hysteresis_loop",
            node_masses_kg=(MASS_KG,),
            gravity_mm_s2=(0.0, 0.0, -GRAVITY_MM_S2),
        )
        self.contact_layout.assert_matches_context(self.context)
        self.base = EnergyRegistry(terms=(UniformGravity(),))
        self.slot = self.contact_layout.slot_of("block_ground")
        self.friction_coefficient = friction_coefficient
        self.normal_stiffness = normal_stiffness
        self.tangential_stiffness = tangential_stiffness
        self.fixed = frozenset(
            {0, 1} | set(range(3, self.contact_layout.layout.dof_count))
        )
        self.vector = list(
            self.contact_layout.initial_vector((0.0, 0.0, -WEIGHT_N / normal_stiffness))
        )
        self.step = None

    def drive(self, path: list[float]) -> list[tuple[float, float]]:
        trace: list[tuple[float, float]] = []
        for displacement in path:
            self.vector[0] = displacement
            self.step = advance_contact_quasistatic(
                registry_without_stick=self.base,
                context=self.context,
                contact_layout=self.contact_layout,
                slot=self.slot,
                vector=tuple(self.vector),
                node=0,
                normal=NORMAL,
                normal_stiffness_n_per_mm=self.normal_stiffness,
                tangential_stiffness_n_per_mm=self.tangential_stiffness,
                friction_coefficient=self.friction_coefficient,
                fixed_indices=self.fixed,
            )
            self.vector = list(self.step.state.vector)
            trace.append((displacement, self.step.tangential_force_n[0]))
        return trace

    @property
    def anchor_x_mm(self) -> float:
        return self.vector[self.slot.anchor_base]


def _leg(start: float, end: float, steps: int) -> list[float]:
    return [start + (end - start) * index / steps for index in range(1, steps + 1)]


def _loop_area(trace: list[tuple[float, float]]) -> float:
    return sum(
        0.5 * (trace[index][1] + trace[index - 1][1])
        * (trace[index][0] - trace[index - 1][0])
        for index in range(1, len(trace))
    )


def test_the_force_saturates_exactly_at_the_cone_limit():
    """``|T| = μN``是**构造保证的等式**，故零容差。法向力全程不动。"""

    oracle = _oracle("oracle:friction/cone_saturation")
    limit = oracle.expected["cone_limit_n"]
    yield_mm = oracle.expected["yield_displacement_mm"]
    tangential_stiffness = oracle.inputs["tangential_stiffness_n_per_mm"]

    assert limit / tangential_stiffness == pytest.approx(
        yield_mm,
        rel=oracle.tolerances["yield_displacement_mm"].rel_tol,
        abs=oracle.tolerances["yield_displacement_mm"].abs_tol,
    )

    drag = _Drag(
        friction_coefficient=oracle.inputs["friction_coefficient"],
        normal_stiffness=oracle.inputs["normal_stiffness_n_per_mm"],
        tangential_stiffness=tangential_stiffness,
    )
    peak = oracle.inputs["amplitude_in_yield"] * yield_mm
    steps = oracle.inputs["steps_per_leg"]
    normal_forces = []
    trace = []
    for path in (_leg(0.0, peak, steps), _leg(peak, -peak, steps)):
        trace.extend(drag.drive(path))
        normal_forces.append(drag.step.normal_force_n)

    forces = [force for _, force in trace]
    cone_tolerance = oracle.tolerances["cone_limit_n"]
    assert max(forces) == pytest.approx(
        limit, rel=cone_tolerance.rel_tol, abs=cone_tolerance.abs_tol
    )
    assert min(forces) == pytest.approx(
        -limit, rel=cone_tolerance.rel_tol, abs=cone_tolerance.abs_tol
    )

    normal_tolerance = oracle.tolerances["normal_force_n"]
    for normal_force in normal_forces:
        assert normal_force == pytest.approx(
            oracle.expected["normal_force_n"],
            rel=normal_tolerance.rel_tol,
            abs=normal_tolerance.abs_tol,
        ), "法向力随切向位移漂了——粘着弹簧多半漏了法向投影"


def test_the_loop_dissipation_matches_the_closed_form():
    """整循环耗散``4·T_max·(u_max − u_y)``，三个幅值。

    **处女加载段不计入**：闭式算的是稳态回线，而``0 → u_max``那一段不属于它。
    起草时把它一起积进去，偏差是28%——不是实现错，是积错了区间。
    """

    oracle = _oracle("oracle:friction/loop_dissipation")
    saturation = _oracle("oracle:friction/cone_saturation")
    yield_mm = saturation.expected["yield_displacement_mm"]
    steps = oracle.inputs["steps_per_leg"]
    tolerance = oracle.tolerances["dissipation_n_mm"]

    for index, ratio in enumerate(oracle.inputs["amplitudes_in_yield"]):
        peak = ratio * yield_mm
        drag = _Drag(
            friction_coefficient=oracle.inputs["friction_coefficient"],
            normal_stiffness=saturation.inputs["normal_stiffness_n_per_mm"],
            tangential_stiffness=oracle.inputs["tangential_stiffness_n_per_mm"],
        )
        virgin = drag.drive(_leg(0.0, peak, steps))
        loop = [virgin[-1]]
        loop.extend(drag.drive(_leg(peak, -peak, steps) + _leg(-peak, peak, steps)))
        assert _loop_area(loop) == pytest.approx(
            oracle.expected["dissipation_n_mm"][index],
            rel=tolerance.rel_tol,
            abs=tolerance.abs_tol,
        ), f"幅值{ratio}·u_y的回线面积"


def test_two_paths_to_the_same_position_disagree():
    """**本案例最要紧的一条：验形制不验公式。**

    锚点若能从当前位形算出来，两条路径会给出同一个答案，
    而`incline_slide_threshold`的判据在那种实现下**照样全绿**——它们只看当前位形。
    """

    oracle = _oracle("oracle:friction/path_dependence")
    saturation = _oracle("oracle:friction/cone_saturation")
    yield_mm = saturation.expected["yield_displacement_mm"]
    steps = oracle.inputs["steps_per_leg"]
    final = oracle.inputs["final_in_yield"] * yield_mm
    peak = oracle.inputs["path_a_peak_in_yield"] * yield_mm

    def _fresh() -> _Drag:
        return _Drag(
            friction_coefficient=oracle.inputs["friction_coefficient"],
            normal_stiffness=saturation.inputs["normal_stiffness_n_per_mm"],
            tangential_stiffness=oracle.inputs["tangential_stiffness_n_per_mm"],
        )

    path_a = _fresh()
    path_a.drive(_leg(0.0, peak, steps) + _leg(peak, final, steps))
    path_b = _fresh()
    path_b.drive(_leg(0.0, final, steps))

    assert path_a.vector[0] == pytest.approx(path_b.vector[0], rel=1e-15), (
        "两条路径的终点位置本该相同——不同就不叫路径相关了"
    )

    ratio_tolerance = oracle.tolerances["force_ratio_a_over_b"]
    ratio = path_a.step.tangential_force_n[0] / path_b.step.tangential_force_n[0]
    assert ratio == pytest.approx(
        oracle.expected["force_ratio_a_over_b"],
        rel=ratio_tolerance.rel_tol,
        abs=ratio_tolerance.abs_tol,
    ), f"同一位置两条路径给出同一个力（比值{ratio}）——历史没被记住"

    for drag, key in ((path_a, "anchor_a_mm"), (path_b, "anchor_b_mm")):
        tolerance = oracle.tolerances[key]
        assert drag.anchor_x_mm == pytest.approx(
            oracle.expected[key], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        ), f"{key}对不上——历史没有被正确写回状态"

    assert path_a.step.is_stick is oracle.expected["path_a_sticks"]
    assert (not path_b.step.is_stick) is oracle.expected["path_b_slips"]


def test_the_anchor_actually_lives_in_the_state_vector():
    """历史必须在**状态向量里**，不在某个求解器实例里（0033的承重条款）。

    锚点若藏在求解器里，那个运行就没法被逐字节复现——
    连run package都进不去（轴3规则5）。
    """

    oracle = _oracle("oracle:friction/path_dependence")
    saturation = _oracle("oracle:friction/cone_saturation")
    yield_mm = saturation.expected["yield_displacement_mm"]
    drag = _Drag(
        friction_coefficient=oracle.inputs["friction_coefficient"],
        normal_stiffness=saturation.inputs["normal_stiffness_n_per_mm"],
        tangential_stiffness=oracle.inputs["tangential_stiffness_n_per_mm"],
    )
    assert drag.anchor_x_mm == 0.0, "出生态的锚点不是零"
    drag.drive(_leg(0.0, 3.0 * yield_mm, 400))
    assert drag.anchor_x_mm != 0.0, "滑移之后锚点没有被写回状态"

    slot = drag.slot
    assert drag.step.state.vector[slot.anchor_base] == drag.anchor_x_mm
    history = drag.contact_layout.layout.history_fields()
    assert any(name.endswith("_anchor_mm") for name in history), (
        "锚点字段没有被声明成历史——复现时它不会被当作必须还原的量"
    )


def test_a_pure_stick_excursion_never_moves_the_anchor():
    """反向门：**没越过屈服面就不许留下任何不可逆位移**。

    只验"滑了会挪"是不够的——一个每步都挪锚点的实现同样能过那条，
    而它会让纯弹性的往返也产生虚假耗散。
    """

    oracle = _oracle("oracle:friction/path_dependence")
    saturation = _oracle("oracle:friction/cone_saturation")
    yield_mm = saturation.expected["yield_displacement_mm"]
    drag = _Drag(
        friction_coefficient=oracle.inputs["friction_coefficient"],
        normal_stiffness=saturation.inputs["normal_stiffness_n_per_mm"],
        tangential_stiffness=oracle.inputs["tangential_stiffness_n_per_mm"],
    )
    # **起点必须显式补上**：`_leg`不含起点，少了它回路就不闭合，
    # 而不闭合的"面积"里会混进一片`½·k_t·u₀²`的三角形。
    # 起草时正是这样多出了1.17e-08——与该三角形逐位吻合。
    # 这与耗散那条门"处女加载段不能积进去"是同一类错：**积错了区间，不是算错了物理**。
    trace = [(0.0, 0.0)]
    trace.extend(
        drag.drive(_leg(0.0, 0.9 * yield_mm, 200) + _leg(0.9 * yield_mm, 0.0, 200))
    )
    assert drag.anchor_x_mm == 0.0, f"纯粘着的往返挪了锚点：{drag.anchor_x_mm}"
    assert drag.step.is_stick
    assert abs(_loop_area(trace)) < 1e-18, (
        f"纯弹性往返产生了耗散：{_loop_area(trace)}"
    )
    assert math.isclose(trace[-1][1], 0.0, abs_tol=1e-12)
