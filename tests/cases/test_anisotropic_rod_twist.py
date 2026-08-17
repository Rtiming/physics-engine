"""conformance：整杆各向异性弯曲与扭转（`cases/anisotropic_rod_twist`）。

**这一条回答的是"带材在非平面槽里到底往哪个轴弯、扭多少"。**

plans/14量出来：这批工件**没有一个是平面件**（离面距离18.4—116.9 mm），
整匝帧扭转`∫|τ|ds`跨236°—657°，而带材`EI_hard/GJ ≈ 1040`。
**带材一定优先扭而不是硬弯**——而各向同性`EI`连"往哪个轴弯"都表达不了。

## 四条判据各自独立于被验内核

* 螺旋线的`κ`/`τ`是教科书闭式；**`κ2 = 0`是结构零**（推导见生成器第一节）；
* 挠度比是两个二阶矩之比，离散误差在两个构型上完全相同、相除即消；
* `θ = M·L/GJ`在离散模型里是线性方程，牛顿一步收敛；
* holonomy由**Gauss-Bonnet**给出：三个直角的球面测地三角形，面积`4π/8 = π/2`。

## 最后一条判据期望的是一个**错误答案**，这是故意的

`oracle:rod/spherical_triangle_holonomy`里有一项
`twist_energy_without_retransport_n_mm = 0.0`，**零容差**。
它记的是"抄了公式不抄retransport外层循环"会拿到什么：
`m_ref`冻结时扭转项对位置没有依赖，单次`solve_equilibrium`里
**弯曲与扭转的交换根本不存在**，扭转能量恰是浮点零。

把那个错误答案写进清单，是因为**它是这道门唯一的分辨力**：
如果哪天有人改了形制使得单次求解也能出扭转，这一条会红，
而那时该红的正是"这道门还在不在区分两条路"这个问题本身。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.energies import (
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    PointLoad,
)
from physics_engine.oracles import load_manifest
from physics_engine.rod import (
    AnisotropicRodBending,
    RodEndMoment,
    RodModel,
    RodReference,
    RodSolveStage,
    RodTwist,
    build_bishop_frame,
    build_rod_layout,
    gammas_from_material_directors,
    solve_rod_with_retransport,
)
from physics_engine.solve import solve_equilibrium

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "anisotropic_rod_twist"
MANIFEST = load_manifest(CASE / "oracle.json")

HELIX_RADIUS_MM = 40.0
HELIX_PITCH_MM = 12.0
HELIX_SWEEP_RAD = 2.0
EI_EASY_NMM2 = 8.0e4
EI_HARD_NMM2 = 8.0e7
GJ_NMM2 = 77.0
CANTILEVER_LENGTH_MM = 50.0
CANTILEVER_LOAD_N = -0.05
AXIAL_STIFFNESS_N = 4.0e4
#: 绝对残差容差取1e-9 N。`solve.py`的参考取法是"总载荷的1e-9到1e-10"（=5e-11），
#: **但本问题的可达残差地板实测是2.5e-10 N**，取5e-11时60次迭代、399次回溯仍不收敛。
#: 这正是`solve.py`docstring里"绝对残差的可达地板随问题规模上升"那一条。
RESIDUAL_TOL_N = 1.0e-9
HOLONOMY_SEGMENT_MM = 10.0


def _context(node_count: int) -> EnergyContext:
    return EnergyContext(
        context_id="context/anisotropic-rod-twist", node_masses_kg=(1.0,) * node_count
    )


# ------------------------------------------------------------------ 螺旋线


def _helix(node_count: int):
    step = HELIX_SWEEP_RAD / (node_count - 1)
    nodes = tuple(
        (
            HELIX_RADIUS_MM * math.cos(index * step),
            HELIX_RADIUS_MM * math.sin(index * step),
            HELIX_PITCH_MM * index * step,
        )
        for index in range(node_count)
    )
    #: `m1`取**边中点**的解析主法线——`κ2`的结构零建立在这个取法上（生成器第一节）。
    normals = tuple(
        (-math.cos((index + 0.5) * step), -math.sin((index + 0.5) * step), 0.0)
        for index in range(node_count - 1)
    )
    return nodes, normals


def _helix_model(node_count: int):
    nodes, normals = _helix(node_count)
    layout = build_rod_layout(layout_id="layout/case-helix", node_count=node_count)
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=normals[0])
    rest = tuple(math.dist(nodes[i], nodes[i + 1]) for i in range(node_count - 1))
    reference = RodReference(rest_lengths_mm=rest, frame=frame)
    model = RodModel(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
        gj_nmm2=(GJ_NMM2,) * (node_count - 2),
    )
    state = layout.initial_state(
        positions_mm=nodes,
        edge_twist_angles=gammas_from_material_directors(frame=frame, m1=normals),
    )
    return model, state


def _helix_measurements(node_count: int) -> dict[str, float]:
    model, state = _helix_model(node_count)
    dual = model.reference.dual_lengths_mm
    curvatures = model.bending().curvatures(state)
    return {
        #: **离散曲率除``l̄``才是1/mm**——混用是静默的``l̄``倍错误。
        "curvature_per_mm": max(
            (pair[0] / dual[index] for index, pair in enumerate(curvatures)),
            key=lambda value: abs(value - HELIX_RADIUS_MM
                                  / (HELIX_RADIUS_MM**2 + HELIX_PITCH_MM**2)),
        ),
        "hard_way_curvature": max(
            (pair[1] for pair in curvatures), key=abs
        ),
        "torsion_per_mm": max(
            model.twist().twist_rates_per_mm(state),
            key=lambda value: abs(value - HELIX_PITCH_MM
                                  / (HELIX_RADIUS_MM**2 + HELIX_PITCH_MM**2)),
        ),
    }


def test_helix_kinematics_and_the_hard_way_curvature_that_must_be_zero() -> None:
    """`κ = R/(R²+p²)`、`τ = p/(R²+p²)`，且`κ2`在螺旋线上必须是机器零。

    第三条是本案例最锋利的一条：它不是"很小"，是**结构零**——
    `m1`取解析主法线时螺旋线上没有任何hard-way弯曲，与离散步长无关。
    """

    MANIFEST.oracle("oracle:rod/helix_kinematics").check_all(_helix_measurements(41))


def test_the_helix_kinematics_converge_second_order() -> None:
    """收敛阶是判据，单点落点不是。**同一条链上`κ2`不参与收敛——它本来就是零。**"""

    kappa = HELIX_RADIUS_MM / (HELIX_RADIUS_MM**2 + HELIX_PITCH_MM**2)
    tau = HELIX_PITCH_MM / (HELIX_RADIUS_MM**2 + HELIX_PITCH_MM**2)
    curvature_errors = []
    torsion_errors = []
    hard_way = []
    for node_count in (21, 41, 81, 161):
        measured = _helix_measurements(node_count)
        curvature_errors.append(abs(measured["curvature_per_mm"] / kappa - 1.0))
        torsion_errors.append(abs(measured["torsion_per_mm"] / tau - 1.0))
        hard_way.append(abs(measured["hard_way_curvature"]))
    for errors in (curvature_errors, torsion_errors):
        ratios = [errors[i - 1] / errors[i] for i in range(1, len(errors))]
        assert min(ratios) > 3.8 and max(ratios) < 4.2, f"收敛比{ratios}不是二阶"
    #: `κ2`**不下降**正是它是结构零的证据：它若是收敛量，细化会让它掉。
    assert max(hard_way) <= 1.0e-13
    assert hard_way[-1] > hard_way[0] * 0.5, (
        f"|κ2|随h细化而显著下降（{hard_way}）——那说明它是一个收敛量而不是结构零，"
        "本案例关于`κ2`的整条论证要重写"
    )


# --------------------------------------------------------------- 易/难轴互换


def _cantilever_tip_mm(*, node_count: int, seed_d1) -> float:
    step = CANTILEVER_LENGTH_MM / (node_count - 1)
    nodes = tuple((step * index, 0.0, 0.0) for index in range(node_count))
    layout = build_rod_layout(layout_id="layout/case-cantilever", node_count=node_count)
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=seed_d1)
    #: 固支顶点的等效对偶长度是`3h/2`不是`h`——被钉死的第一条边吞掉了半格柔度。
    #: 推导见`energies.clamped_chain_bending_vertices`；不加它只有一阶收敛。
    reference = RodReference(
        rest_lengths_mm=(step,) * (node_count - 1),
        frame=frame,
        dual_lengths_mm=tuple(
            (1.5 if vertex == 0 else 1.0) * step for vertex in range(node_count - 2)
        ),
    )
    bending = AnisotropicRodBending(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(EI_EASY_NMM2,) * (node_count - 2),
        ei_hard_nmm2=(EI_HARD_NMM2,) * (node_count - 2),
    )
    stretch = AxialStretch(
        edges=tuple(
            (index, index + 1, step, AXIAL_STIFFNESS_N)
            for index in range(node_count - 1)
        )
    )
    load = PointLoad(loads=((node_count - 1, (0.0, CANTILEVER_LOAD_N, 0.0)),))
    state = layout.initial_state(
        positions_mm=nodes, edge_twist_angles=(0.0,) * layout.edge_count
    )
    #: γ全钉住＝把扭转通道关掉，量到的才是**纯弯曲**的各向异性。
    #: 不钉，杆会自己转过去用软轴——那是另一件事，由holonomy那两条门验。
    result = solve_equilibrium(
        EnergyRegistry(terms=(bending, stretch, load)),
        _context(node_count),
        layout.layout,
        state.vector,
        fixed_indices=frozenset(range(6)) | layout.twist_indices(),
        residual_tol_n=RESIDUAL_TOL_N,
        max_iterations=40,
    )
    assert result.converged, result.reason
    return result.state.vector[3 * (node_count - 1) + 1]


@pytest.mark.batch
def test_rotating_the_reference_frame_by_ninety_degrees_swaps_easy_and_hard() -> None:
    """参考`d1`转90°，挠度比必须**等于**`EI_hard/EI_easy`。

    同行自己点名了这个失效模式（取错`d1`让`EI_hard`接管、挠度差1600倍、
    **不报任何错**）却没有门守着。这就是那道门。
    """

    easy = _cantilever_tip_mm(node_count=81, seed_d1=(0.0, 1.0, 0.0))
    hard = _cantilever_tip_mm(node_count=81, seed_d1=(0.0, 0.0, 1.0))
    MANIFEST.oracle("oracle:rod/easy_hard_axis_swap").check_all(
        {
            "easy_tip_mm": easy,
            "hard_tip_mm": hard,
            "deflection_ratio": easy / hard,
        }
    )


@pytest.mark.batch
def test_the_cantilever_deflection_converges_second_order() -> None:
    """带固支半格订正时是干净的二阶。**不带就只有一阶**——那是决策0027的第三例。"""

    closed = CANTILEVER_LOAD_N * CANTILEVER_LENGTH_MM**3 / (3.0 * EI_EASY_NMM2)
    errors = [
        abs(_cantilever_tip_mm(node_count=count, seed_d1=(0.0, 1.0, 0.0)) / closed - 1.0)
        for count in (11, 21, 41, 81)
    ]
    ratios = [errors[i - 1] / errors[i] for i in range(1, len(errors))]
    assert min(ratios) > 3.7, f"收敛比{ratios}——固支半格订正没起作用？"


# ------------------------------------------------------------------ 闭式扭转


def test_the_end_torque_reproduces_the_closed_form_twist() -> None:
    """`θ = M·L/GJ`，`L`是端边中点之间的距离（190 mm，**不是杆长200 mm**）。

    全仓在本案例之前**没有任何扭转金标**。
    """

    node_count = 21
    length = 200.0
    moment = 0.5
    step = length / (node_count - 1)
    nodes = tuple((step * index, 0.0, 0.0) for index in range(node_count))
    layout = build_rod_layout(layout_id="layout/case-torsion", node_count=node_count)
    frame = build_bishop_frame(positions_mm=nodes, seed_d1=(0.0, 1.0, 0.0))
    reference = RodReference(rest_lengths_mm=(step,) * (node_count - 1), frame=frame)
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
    registry = EnergyRegistry(
        terms=(
            bending,
            twist,
            RodEndMoment(layout=layout, edge=last_edge, moment_n_mm=moment),
        )
    )
    state = layout.initial_state(
        positions_mm=nodes, edge_twist_angles=(0.0,) * layout.edge_count
    )
    #: 悬臂扭转BC＝钉住`3N+0`（第一条边的γ）。**不需要任何新的约束机制。**
    result = solve_equilibrium(
        registry,
        _context(node_count),
        layout.layout,
        state.vector,
        fixed_indices=layout.position_indices() | {layout.twist_index(0)},
        residual_tol_n=1.0e-12,
        max_iterations=20,
    )
    assert result.converged, result.reason
    assert result.iterations == 1, "线性方程应当一步收敛；多步说明有非线性混进来了"
    rates = twist.twist_rates_per_mm(result.state)
    MANIFEST.oracle("oracle:rod/end_torque_twist").check_all(
        {
            "effective_length_mm": sum(reference.dual_lengths_mm),
            "tip_twist_rad": result.state.vector[layout.twist_index(last_edge)],
            "uniform_twist_rate_per_mm": max(rates, key=abs),
        }
    )
    assert max(rates) - min(rates) <= 1.0e-15, "扭率必须逐顶点相同"


# ------------------------------------------------- retransport外层循环与holonomy


def _spherical_triangle():
    """切向序列`x̂ → ŷ → ẑ → x̂`——三个直角的球面测地三角形，面积`4π/8 = π/2`。"""

    step = HOLONOMY_SEGMENT_MM
    return (
        (0.0, 0.0, 0.0),
        (step, 0.0, 0.0),
        (step, step, 0.0),
        (step, step, step),
        (2.0 * step, step, step),
    )


def _holonomy_model(*, ei_nmm2: float):
    step = HOLONOMY_SEGMENT_MM
    straight = tuple((step * index, 0.0, 0.0) for index in range(5))
    layout = build_rod_layout(layout_id="layout/case-holonomy", node_count=5)
    frame = build_bishop_frame(positions_mm=straight, seed_d1=(0.0, 1.0, 0.0))
    reference = RodReference(rest_lengths_mm=(step,) * 4, frame=frame)
    model = RodModel(
        layout=layout,
        reference=reference,
        ei_easy_nmm2=(ei_nmm2,) * 3,
        ei_hard_nmm2=(ei_nmm2,) * 3,
        gj_nmm2=(GJ_NMM2,) * 3,
    )
    return model, straight


def test_the_holonomy_and_the_twist_it_produces_and_the_answer_without_it() -> None:
    """三个数一起判：holonomy、串联扭簧能量、**以及不重输运时那个错误答案**。

    弯曲刚度在本门里刻意压到`1e-3 N·mm²`（GJ的1e-5量级）：球面三角的90°折角
    给出很大的离散曲率，弯曲项对γ的耦合会把扭转分布拉歪，
    而**本门单独验的是扭转通道**。弯曲通道由上面两组门各自验过。
    """

    model, straight = _holonomy_model(ei_nmm2=1.0e-3)
    layout = model.layout
    context = _context(5)
    target = _spherical_triangle()
    prescribed = tuple(
        (3 * node + axis, target[node][axis]) for node in range(5) for axis in range(3)
    )
    fixed = layout.position_indices() | {layout.twist_index(0), layout.twist_index(3)}
    equilibrium = solve_rod_with_retransport(
        model=model,
        context=context,
        initial=layout.initial_state(
            positions_mm=straight, edge_twist_angles=(0.0,) * 4
        ),
        stages=(
            RodSolveStage(
                fixed_indices=fixed, prescribed=prescribed, retransport_rounds=4
            ),
        ),
        residual_tol_n=1.0e-12,
    )
    assert equilibrium.converged
    gammas = layout.twist_angles(equilibrium.state)

    #: 对照路径：只求解一次，不重输运——即"抄了公式不抄外循环"。
    single = solve_equilibrium(
        EnergyRegistry(terms=model.terms()),
        context,
        layout.layout,
        layout.initial_state(positions_mm=target, edge_twist_angles=(0.0,) * 4).vector,
        fixed_indices=fixed,
        residual_tol_n=1.0e-12,
        max_iterations=40,
    )
    MANIFEST.oracle("oracle:rod/spherical_triangle_holonomy").check_all(
        {
            "holonomy_rad": abs(gammas[3] - gammas[0]),
            "series_spring_twist_energy_n_mm": equilibrium.rounds[-1].twist_energy_n_mm,
            "twist_energy_without_retransport_n_mm": model.twist().energy(
                single.state, context
            ),
        }
    )
    #: 第0轮（参考帧还是直杆的）扭转能量同样恰为零——**外层循环的第一轮就是对照组**。
    assert equilibrium.rounds[0].twist_energy_n_mm == 0.0
    assert equilibrium.rounds[-1].max_gamma_change <= 1.0e-15, "外层循环没有收敛"


def test_the_outer_loop_converges_and_the_twist_is_uniform() -> None:
    """收敛后γ沿链均匀分布——两端夹持、中间自由的串联扭簧本来就该这样。"""

    model, straight = _holonomy_model(ei_nmm2=1.0e-3)
    layout = model.layout
    target = _spherical_triangle()
    equilibrium = solve_rod_with_retransport(
        model=model,
        context=_context(5),
        initial=layout.initial_state(
            positions_mm=straight, edge_twist_angles=(0.0,) * 4
        ),
        stages=(
            RodSolveStage(
                fixed_indices=(
                    layout.position_indices()
                    | {layout.twist_index(0), layout.twist_index(3)}
                ),
                prescribed=tuple(
                    (3 * node + axis, target[node][axis])
                    for node in range(5)
                    for axis in range(3)
                ),
                retransport_rounds=4,
            ),
        ),
        residual_tol_n=1.0e-12,
    )
    gammas = layout.twist_angles(equilibrium.state)
    for index in range(1, 4):
        assert abs((gammas[index] - gammas[index - 1]) + math.pi / 6.0) <= 1.0e-15


# --------------------------------------------------------------------- 清单自身


def test_the_manifest_binds_the_generator_it_was_produced_by() -> None:
    """生成器改了而清单没重生成即红——轴7规则2的执行面。"""

    MANIFEST.verify_generator(ROOT)
    assert MANIFEST.case_id == "case/anisotropic_rod_twist"
    assert MANIFEST.load_tier == "local_batch"
    assert len(MANIFEST.oracles) == 4


def test_the_generator_does_not_import_the_mechanics_it_is_supposed_to_check() -> None:
    """**金标必须独立于被验内核**：生成器不许碰`rod`/`solve`/`energies`。

    判据本身被验：`physics_engine.oracles`（写清单用）是允许的，
    所以这条门不能简单地判"出现过physics_engine"。
    """

    source = (CASE / "generate_oracle.py").read_text(encoding="utf-8")
    for forbidden in (
        "physics_engine.rod",
        "physics_engine.solve",
        "physics_engine.energies",
        "physics_engine.state",
    ):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source, (
            f"生成器import了{forbidden}——金标不再独立于被验内核"
        )
    assert "from physics_engine.oracles import" in source
