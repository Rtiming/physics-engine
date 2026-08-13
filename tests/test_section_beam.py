"""截面纤维本构接入WDS式三节点Kirchhoff顶点的红门（决策0060）。

本文件先于实现落地并实跑为红：首次收集因
``ModuleNotFoundError: physics_engine.section_beam``失败。它守四件不同的事：

* 运动学逐数对住WDS当前物理源，而不是拿小转角梁公式替代；
* 截面弯矩与一致切线经链式法则进入全局梯度/Hessian；
* 纤维历史在Newton trial期间固定，只有收敛后才提交；
* 第一片只有easy-axis弯曲，不能静默宣称已有轴向、hard-axis或扭转截面装配。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.energies import EnergyContext, EnergyError, EnergyRegistry
from physics_engine.section_beam import (
    KirchhoffFiberSectionBending,
    KirchhoffSectionError,
    KirchhoffSectionReference,
    build_kirchhoff_section_vertex_layout,
    solve_kirchhoff_section_equilibrium,
)
from physics_engine.sections import ElasticPerfectlyPlastic1D, RectangularFiberSection

WDS_HEAD = "c1b8fe6"
WDS_STATE_SHA256 = "ea61bf2611ce30fb91248f9092d5cdf2eff82a0688926253ac9e929b30577c27"
WDS_ENERGIES_SHA256 = "2d3e4d1784c94898dd2efb185091e29c2041fe4513e569e0fb0e5b99c1ed7d77"
WDS_POSITIONS_MM = (
    (0.0, 0.0, 0.0),
    (80.0, 1.0, 2.0),
    (155.0, 12.0, 7.0),
)
WDS_GAMMA = (0.23, -0.17)
WDS_KAPPA1_PER_MM = 0.001697465689615586
WDS_ELASTIC_ENERGY_N_MM = 0.11165385348760701
WDS_ELASTIC_GRADIENT = (
    -0.0002790653820939029,
    0.020833473490058232,
    0.0007458785387269992,
    0.0035048233815929072,
    -0.0425883243807178,
    -0.0012715765717610213,
    -0.003225757999499006,
    0.021754850890659577,
    0.0005256980330340221,
    0.008401897219114144,
    0.05366861042942003,
)


def _reference(*, length_mm: float = 100.0) -> KirchhoffSectionReference:
    return KirchhoffSectionReference(
        rest_lengths_mm=(length_mm, length_mm),
        reference_d1=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        reference_d2=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        natural_kappa1=0.0,
    )


def _wds_reference() -> KirchhoffSectionReference:
    return KirchhoffSectionReference(
        rest_lengths_mm=(80.0, 75.0),
        reference_d1=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        reference_d2=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        natural_kappa1=0.0,
    )


def _layout(*, point_count: int = 16, reference=None):
    section = RectangularFiberSection(
        section_id="section/kirchhoff-vertex-test",
        width_mm=12.0,
        thickness_mm=4.0,
        point_count=point_count,
    )
    return build_kirchhoff_section_vertex_layout(
        layout_id=f"layout/kirchhoff-vertex-n{point_count}",
        section=section,
        reference=reference or _reference(),
    )


def test_layout_separates_eleven_wds_kinematic_dofs_from_point_history():
    layout = _layout(point_count=8)
    state = layout.initial_state(
        positions_mm=((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)),
        edge_twist_angles=(0.0, 0.0),
    )

    assert layout.layout.node_dof_count == 9
    assert layout.kinematic_dof_count == 11
    assert layout.history_scalar_count == 16
    assert layout.layout.dof_count == 27
    assert layout.history_indices == frozenset(range(11, 27))
    assert state.vector[:11] == (
        0.0, 0.0, 0.0,
        100.0, 0.0, 0.0,
        200.0, 0.0, 0.0,
        0.0, 0.0,
    )
    assert state.vector[11:] == (0.0,) * 16


def test_easy_axis_kinematics_and_elastic_derivatives_match_wds_fixture():
    """夹具由WDS ``c1b8fe6``的两份干净源文件只读生成，源SHA钉在本文件顶部。"""

    # b=1,h=4,n=2的中点纤维二次矩恰为4 mm^4；E=250使离散EI恰为1000 Nmm²，
    # 因而可逐量对拍WDS以EI_easy=1000、EI_hard=0生成的独立结果。
    section = RectangularFiberSection(
        section_id="section/wds-parity",
        width_mm=1.0,
        thickness_mm=4.0,
        point_count=2,
    )
    layout = build_kirchhoff_section_vertex_layout(
        layout_id="layout/wds-parity",
        section=section,
        reference=_wds_reference(),
    )
    committed = layout.initial_state(
        positions_mm=WDS_POSITIONS_MM,
        edge_twist_angles=WDS_GAMMA,
    )
    term = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=ElasticPerfectlyPlastic1D(
            young_modulus_n_mm2=250.0,
            yield_stress_n_mm2=1.0e9,
        ),
        committed_state=committed,
    )
    context = EnergyContext(
        context_id="context/wds-parity",
        node_masses_kg=(1.0, 1.0, 1.0),
    )

    trial = term.trial_response(committed)
    energy, gradient, hessian = term.quantities(
        committed, context, need_gradient=True, need_hessian=True
    )
    assert WDS_HEAD and WDS_STATE_SHA256 and WDS_ENERGIES_SHA256
    # profile后的0/1/2阶按需路径必须逐位对住二阶参考通道；优化不能改物理字节。
    assert trial.curvature_per_mm == term.kinematics(committed).curvature_per_mm
    assert energy == term.energy(committed, context)
    assert gradient == term.gradient(committed, context)
    assert trial.curvature_per_mm == pytest.approx(WDS_KAPPA1_PER_MM, abs=2.0e-18)
    assert energy == pytest.approx(WDS_ELASTIC_ENERGY_N_MM, rel=3.0e-15, abs=1.0e-15)
    assert gradient[:11] == pytest.approx(WDS_ELASTIC_GRADIENT, rel=3.0e-13, abs=3.0e-14)
    assert gradient[11:] == (0.0,) * layout.history_scalar_count
    assert hessian is not None
    assert all(
        hessian[row][column] == pytest.approx(hessian[column][row], abs=2.0e-14)
        for row in range(11)
        for column in range(11)
    )

    # Hessian不拿同一套二阶AD自证：对生产梯度做中心差分。
    for column in range(11):
        step = 1.0e-6 * max(1.0, abs(committed.vector[column]))
        plus = list(committed.vector)
        minus = list(committed.vector)
        plus[column] += step
        minus[column] -= step
        g_plus = term.gradient(committed.with_vector(tuple(plus)), context)
        g_minus = term.gradient(committed.with_vector(tuple(minus)), context)
        for row in range(11):
            measured = (g_plus[row] - g_minus[row]) / (2.0 * step)
            assert measured == pytest.approx(hessian[row][column], rel=2.0e-6, abs=3.0e-9)


def test_global_solver_commits_history_only_after_convergence():
    layout = _layout(point_count=32)
    virgin = layout.initial_state(
        positions_mm=((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)),
        edge_twist_angles=(0.0, 0.0),
    )
    loading_term = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=ElasticPerfectlyPlastic1D(
            young_modulus_n_mm2=200_000.0,
            yield_stress_n_mm2=250.0,
        ),
        committed_state=virgin,
    )
    context = EnergyContext(
        context_id="context/global-section-load",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    fixed = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 10})
    prescribed = list(virgin.vector)
    prescribed[7] = 12.55
    loaded_trial = loading_term.trial_response(virgin.with_vector(tuple(prescribed)))
    assert any(point.yielded for point in loaded_trial.section_response.points)
    loaded_state = loaded_trial.next_state
    unloading_term = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=loading_term.material,
        committed_state=loaded_state,
    )

    stopped = solve_kirchhoff_section_equilibrium(
        section_term=unloading_term,
        context=context,
        fixed_indices=fixed,
        residual_tol_n=1.0e-9,
        max_iterations=1,
    )
    assert not stopped.equilibrium.converged
    assert not stopped.history_committed
    assert stopped.committed_state.pack() == loaded_state.pack()

    unloaded = solve_kirchhoff_section_equilibrium(
        section_term=unloading_term,
        context=context,
        fixed_indices=fixed,
        residual_tol_n=1.0e-9,
        max_iterations=50,
    )
    assert unloaded.equilibrium.converged, unloaded.equilibrium.reason
    assert unloaded.history_committed
    assert 0.0 < unloaded.committed_state.vector[7] < loaded_state.vector[7]
    assert abs(unloaded.trial.section_response.bending_moment_n_mm) < 1.0e-7
    assert unloaded.committed_state.block("section_point_plastic_strain") == (
        loaded_state.block("section_point_plastic_strain")
    )


def test_reference_singularity_and_unimplemented_axes_fail_closed():
    with pytest.raises(KirchhoffSectionError, match="orthonormal"):
        KirchhoffSectionReference(
            rest_lengths_mm=(1.0, 1.0),
            reference_d1=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            reference_d2=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            natural_kappa1=0.0,
        )

    layout = _layout(point_count=2)
    folded = layout.initial_state(
        positions_mm=((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        edge_twist_angles=(0.0, 0.0),
    )
    term = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=ElasticPerfectlyPlastic1D(200_000.0, 250.0),
        committed_state=folded,
    )
    with pytest.raises(KirchhoffSectionError, match="antiparallel"):
        term.trial_response(folded)

    straight = layout.initial_state(
        positions_mm=((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)),
        edge_twist_angles=(0.0, 0.0),
    )
    guarded = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=term.material,
        committed_state=straight,
    )
    changed_history = list(straight.vector)
    changed_history[min(layout.history_indices)] = 1.0e-4
    with pytest.raises(KirchhoffSectionError, match="committed history"):
        guarded.trial_response(straight.with_vector(tuple(changed_history)))

    with pytest.raises(EnergyError, match="quasistatic-only"):
        EnergyRegistry((guarded,)).acceleration(
            EnergyContext(
                context_id="context/no-dynamic-plastic-section",
                node_masses_kg=(1.0, 1.0, 1.0),
            ),
            layout.layout,
        )

    assert term.supported_generalized_strains == ("easy_axis_curvature",)
    assert "axial" in term.unsupported_axes
    assert "hard_axis" in term.unsupported_axes
    assert "twist" in term.unsupported_axes
    assert math.isfinite(layout.reference.dual_length_mm)
