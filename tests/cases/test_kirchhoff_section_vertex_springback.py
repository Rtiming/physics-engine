"""conformance：WDS运动学三节点顶点的弹塑性截面回弹（决策0060）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.energies import EnergyContext
from physics_engine.oracles import OracleError, load_manifest
from physics_engine.section_beam import (
    KirchhoffFiberSectionBending,
    KirchhoffSectionReference,
    build_kirchhoff_section_vertex_layout,
    solve_kirchhoff_section_equilibrium,
)
from physics_engine.sections import ElasticPerfectlyPlastic1D, RectangularFiberSection

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "kirchhoff_section_vertex_springback"
MANIFEST = load_manifest(CASE / "oracle.json", root=ROOT)
GLOBAL = MANIFEST.oracle("oracle:section/global_vertex_springback")
WDS = MANIFEST.oracle("oracle:section/wds_easy_axis_fixture")


def _global_model():
    inputs = GLOBAL.inputs
    length = inputs["length_mm"]
    section = RectangularFiberSection(
        section_id="section/kirchhoff-global-oracle",
        width_mm=inputs["width_mm"],
        thickness_mm=inputs["thickness_mm"],
        point_count=inputs["point_count"],
    )
    reference = KirchhoffSectionReference(
        rest_lengths_mm=(length, length),
        reference_d1=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        reference_d2=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        natural_kappa1=0.0,
    )
    layout = build_kirchhoff_section_vertex_layout(
        layout_id="layout/kirchhoff-global-oracle",
        section=section,
        reference=reference,
    )
    virgin = layout.initial_state(
        positions_mm=((0.0, 0.0, 0.0), (length, 0.0, 0.0), (2.0 * length, 0.0, 0.0)),
        edge_twist_angles=(0.0, 0.0),
    )
    material = ElasticPerfectlyPlastic1D(
        young_modulus_n_mm2=inputs["young_modulus_n_mm2"],
        yield_stress_n_mm2=inputs["yield_stress_n_mm2"],
    )
    context = EnergyContext(
        context_id="context/kirchhoff-global-oracle",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    return layout, virgin, material, context


def _load_and_springback():
    layout, virgin, material, context = _global_model()
    prescribed = list(virgin.vector)
    prescribed[GLOBAL.inputs["free_global_index"]] = GLOBAL.expected[
        "loaded_vertical_displacement_mm"
    ]
    loaded_geometry = virgin.with_vector(tuple(prescribed))
    loading_term = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=material,
        committed_state=virgin,
    )
    loaded_trial = loading_term.trial_response(loaded_geometry)
    loaded_state = loaded_trial.next_state
    unloading_term = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=material,
        committed_state=loaded_state,
    )
    fixed = frozenset(index for index in range(11) if index != GLOBAL.inputs["free_global_index"])
    failed = solve_kirchhoff_section_equilibrium(
        section_term=unloading_term,
        context=context,
        fixed_indices=fixed,
        residual_tol_n=1.0e-9,
        max_iterations=1,
    )
    solved = solve_kirchhoff_section_equilibrium(
        section_term=unloading_term,
        context=context,
        fixed_indices=fixed,
        residual_tol_n=1.0e-9,
        max_iterations=50,
    )
    return loading_term, loaded_geometry, loaded_trial, loaded_state, failed, solved, context


def test_prescribed_loading_reaches_the_fraction_global_force_and_tangent_oracle():
    """一次量齐加载、回弹与commit边界；遗漏任何清单量也失败。"""

    term, state, trial, loaded, failed, solved, context = _load_and_springback()
    _, _, _, loaded_b, _, solved_b, _ = _load_and_springback()
    _, gradient, hessian = term.quantities(
        state,
        context,
        need_gradient=True,
        need_hessian=True,
    )
    assert gradient is not None and hessian is not None
    free = GLOBAL.inputs["free_global_index"]
    GLOBAL.check_all({
        "loaded_vertical_displacement_mm": state.vector[free],
        "loaded_curvature_per_mm": trial.curvature_per_mm,
        "loaded_moment_n_mm": trial.section_response.bending_moment_n_mm,
        "loaded_generalized_force_n": gradient[free],
        "loaded_tangent_n_per_mm": hessian[free][free],
        "loaded_yielded_point_count": sum(
            point.yielded for point in trial.section_response.points
        ),
        "solver_converged": solved.equilibrium.converged,
        "springback_vertical_displacement_mm": solved.committed_state.vector[free],
        "springback_curvature_per_mm": solved.trial.curvature_per_mm,
        "springback_moment_n_mm": solved.trial.section_response.bending_moment_n_mm,
        "history_unchanged_on_elastic_unload": all(
            loaded.block(name) == solved.committed_state.block(name)
            for name in loaded.layout.history_fields()
        ),
        "replay_bytes_equal": (
            loaded.pack() == loaded_b.pack()
            and solved.committed_state.pack() == solved_b.committed_state.pack()
        ),
        "failed_trial_does_not_commit": (
            not failed.equilibrium.converged
            and not failed.history_committed
            and failed.committed_state.pack() == loaded.pack()
        ),
    })


def test_global_newton_springback_commits_only_the_converged_history():
    """回弹位置由全局节点平衡求出；失败trial与成功卸载都守commit边界。"""

    _, _, _, loaded, failed_a, solved_a, _ = _load_and_springback()
    _, _, _, loaded_b, failed_b, solved_b, _ = _load_and_springback()
    history = loaded.layout.history_fields()
    assert solved_a.equilibrium.converged and solved_b.equilibrium.converged
    assert all(
        loaded.block(name) == solved_a.committed_state.block(name) for name in history
    )
    assert not failed_a.history_committed and not failed_b.history_committed
    assert failed_a.committed_state.pack() == loaded.pack()
    assert failed_b.committed_state.pack() == loaded_b.pack()
    assert solved_a.committed_state.pack() == solved_b.committed_state.pack()
    assert solved_a.history_committed
    assert solved_a.equilibrium.iterations <= 8


def test_wds_state_order_kinematics_energy_and_gradient_match_the_captured_fixture():
    """只读兼容证据；不import兄弟仓，也不据此宣称WDS已采用本模块。"""

    inputs = WDS.inputs
    section = RectangularFiberSection(
        section_id="section/wds-easy-axis-fixture",
        width_mm=1.0,
        thickness_mm=4.0,
        point_count=2,
    )
    layout = build_kirchhoff_section_vertex_layout(
        layout_id="layout/wds-easy-axis-fixture",
        section=section,
        reference=KirchhoffSectionReference(
            rest_lengths_mm=tuple(inputs["rest_lengths_mm"]),
            reference_d1=tuple(tuple(row) for row in inputs["reference_d1"]),
            reference_d2=tuple(tuple(row) for row in inputs["reference_d2"]),
            natural_kappa1=inputs["natural_kappa1"],
        ),
    )
    state = layout.initial_state(
        positions_mm=tuple(tuple(row) for row in inputs["positions_mm"]),
        edge_twist_angles=tuple(inputs["edge_twist_angles"]),
    )
    term = KirchhoffFiberSectionBending(
        vertex_layout=layout,
        material=ElasticPerfectlyPlastic1D(
            young_modulus_n_mm2=250.0,
            yield_stress_n_mm2=1.0e9,
        ),
        committed_state=state,
    )
    context = EnergyContext(
        context_id="context/wds-easy-axis-fixture",
        node_masses_kg=(1.0, 1.0, 1.0),
    )
    energy, gradient, _ = term.quantities(
        state, context, need_gradient=True, need_hessian=False
    )
    assert gradient is not None
    WDS.check_all(
        {
            "curvature_per_mm": term.trial_response(state).curvature_per_mm,
            "elastic_energy_n_mm": energy,
            "elastic_gradient": gradient[:11],
        }
    )
    assert inputs["source_commit"] == "c1b8fe6"
    assert len(inputs["state_source_sha256"]) == 64
    assert len(inputs["energies_source_sha256"]) == 64


RED_MATRIX = (
    (
        "loaded_curvature_per_mm",
        GLOBAL.expected["loaded_vertical_displacement_mm"] / GLOBAL.inputs["length_mm"] ** 2,
        "用小转角kappa=v/L²替代WDS的几何精确曲率",
    ),
    (
        "loaded_moment_n_mm",
        16_000.0,
        "回到线弹性EI，丢掉32个已屈服纤维",
    ),
    (
        "loaded_generalized_force_n",
        GLOBAL.expected["loaded_moment_n_mm"] / GLOBAL.inputs["length_mm"],
        "把dκ/dv误写成小转角常数1/L²",
    ),
    (
        "loaded_tangent_n_per_mm",
        0.0,
        "漏装截面一致切线与M乘曲率二阶导两项",
    ),
    (
        "springback_vertical_displacement_mm",
        0.0,
        "卸载时丢掉纤维塑性历史，伪造回到直杆",
    ),
    (
        "failed_trial_does_not_commit",
        False,
        "失败Newton把trial塑性状态偷写成committed",
    ),
)


@pytest.mark.parametrize(("quantity", "wrong", "reason"), RED_MATRIX)
def test_each_global_section_gate_has_a_known_wrong_implementation_that_must_red(
    quantity, wrong, reason
):
    try:
        GLOBAL.check(quantity, wrong)
    except OracleError:
        return
    pytest.fail(f"这条错实现没有被判红：{reason}")
