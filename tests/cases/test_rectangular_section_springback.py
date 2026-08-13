"""conformance：矩形弹塑性纤维截面的单调弯曲与自由回弹（决策0059）。

金标生成器只用``Fraction``分段闭式，不import被验的``physics_engine.sections``。
本文件只负责走生产路径、读取清单并测量；不在测试里另抄一套截面闭式。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.oracles import OracleError, load_manifest
from physics_engine.sections import (
    ElasticPerfectlyPlastic1D,
    RectangularFiberSection,
    build_rectangular_section_layout,
    evaluate_section_response,
    solve_section_curvature,
)

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases" / "rectangular_section_springback"
MANIFEST = load_manifest(CASE / "oracle.json", root=ROOT)
MONOTONIC = MANIFEST.oracle("oracle:section/monotonic_bending")
SPRINGBACK = MANIFEST.oracle("oracle:section/free_springback")


def _model(point_count: int):
    inputs = MONOTONIC.inputs
    section = RectangularFiberSection(
        section_id="section/rectangular-springback",
        width_mm=inputs["width_mm"],
        thickness_mm=inputs["thickness_mm"],
        point_count=point_count,
    )
    layout = build_rectangular_section_layout(
        layout_id=f"layout/rectangular-springback-n{point_count}",
        section=section,
    )
    material = ElasticPerfectlyPlastic1D(
        young_modulus_n_mm2=inputs["young_modulus_n_mm2"],
        yield_stress_n_mm2=inputs["yield_stress_n_mm2"],
    )
    return section, layout, material


def _loaded_response(point_count: int, curvature_per_mm: float):
    _, layout, material = _model(point_count)
    response = evaluate_section_response(
        section_layout=layout,
        material=material,
        previous_state=layout.initial_state(),
        axial_strain=0.0,
        curvature_per_mm=curvature_per_mm,
    )
    return layout, material, response


def test_nonlinear_point_distribution_and_moment_converge_to_the_continuum_oracle():
    """同时验点级分布、非线性汇总量与加密收敛；只验任一层都不够。"""

    curvature = MONOTONIC.inputs["loaded_curvature_per_mm"]
    counts = MONOTONIC.inputs["point_counts"]
    responses = [_loaded_response(count, curvature)[2] for count in counts]
    moments = [response.bending_moment_n_mm for response in responses]
    continuum = MONOTONIC.expected["continuum_loaded_moment_n_mm"]
    errors = [abs(continuum - moment) for moment in moments]
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    finest = responses[-1]

    _, _, at_first_yield = _loaded_response(counts[-1], MONOTONIC.inputs["yield_curvature_per_mm"])
    # 足够大的曲率让最靠近中性的两个纤维也屈服；这条验完全塑性平台，
    # 不把某个Gauss/中点位置冒充真实表面应力。
    _, _, fully_plastic = _loaded_response(counts[-1], 0.1)

    MONOTONIC.check_all(
        {
            "axial_force_n": finest.axial_force_n,
            "continuum_loaded_moment_n_mm": finest.bending_moment_n_mm,
            "elastic_yield_moment_n_mm": at_first_yield.bending_moment_n_mm,
            "plastic_moment_n_mm": fully_plastic.bending_moment_n_mm,
            "fiber_loaded_moments_n_mm": moments,
            "moment_error_ratios": ratios,
            "loaded_point_stresses_n_mm2": [point.stress_n_mm2 for point in finest.points],
            "loaded_point_plastic_strains": [point.plastic_strain for point in finest.points],
            "loaded_point_yielded": [point.yielded for point in finest.points],
        }
    )
    assert any(point.yielded for point in finest.points)
    assert any(not point.yielded for point in finest.points)
    assert max(abs(point.y_mm) for point in finest.points) < (
        MONOTONIC.inputs["thickness_mm"] / 2.0
    ), "中点纤维不在表面；不得把最外点结果冒充真实表面应力"


def _solve_springback():
    count = SPRINGBACK.inputs["point_count"]
    layout, material, loaded = _loaded_response(count, SPRINGBACK.inputs["loaded_curvature_per_mm"])
    solved = solve_section_curvature(
        section_layout=layout,
        material=material,
        previous_state=loaded.next_state,
        axial_strain=0.0,
        target_moment_n_mm=SPRINGBACK.inputs["target_moment_n_mm"],
        curvature_bracket_per_mm=tuple(SPRINGBACK.inputs["curvature_bracket_per_mm"]),
        residual_tol_n_mm=SPRINGBACK.inputs["residual_tol_n_mm"],
    )
    return layout, material, loaded, solved


def test_free_springback_solves_equilibrium_without_hiding_or_polluting_history():
    """回弹曲率由``M=0``求出；trial不污染历史，同一路径逐字节可复现。"""

    layout_a, material_a, loaded_a, solved_a = _solve_springback()
    _, _, loaded_b, solved_b = _solve_springback()
    virgin_at_same_curvature = evaluate_section_response(
        section_layout=layout_a,
        material=material_a,
        previous_state=layout_a.initial_state(),
        axial_strain=0.0,
        curvature_per_mm=solved_a.curvature_per_mm,
    )
    history_fields = layout_a.layout.history_fields()
    history_unchanged = all(
        loaded_a.next_state.block(name) == solved_a.response.next_state.block(name)
        for name in history_fields
    )
    same_curvature_different_history = tuple(
        point.stress_n_mm2 for point in solved_a.response.points
    ) != tuple(point.stress_n_mm2 for point in virgin_at_same_curvature.points)

    SPRINGBACK.check_all(
        {
            "solver_converged": solved_a.converged,
            "equilibrium_moment_n_mm": solved_a.response.bending_moment_n_mm,
            "fiber_springback_curvature_per_mm": solved_a.curvature_per_mm,
            "continuum_springback_curvature_per_mm": solved_a.curvature_per_mm,
            "springback_point_stresses_n_mm2": [
                point.stress_n_mm2 for point in solved_a.response.points
            ],
            "point_history_unchanged_on_elastic_unload": history_unchanged,
            "replay_bytes_equal": (
                loaded_a.next_state.pack() == loaded_b.next_state.pack()
                and solved_a.response.next_state.pack() == solved_b.response.next_state.pack()
            ),
            "same_curvature_different_history_differs": same_curvature_different_history,
        }
    )


RED_MATRIX = (
    (
        MONOTONIC,
        "continuum_loaded_moment_n_mm",
        16_000.0,
        "把截面永远当线弹性EI，完全看不见外层屈服",
    ),
    (
        MONOTONIC,
        "fiber_loaded_moments_n_mm",
        [0.0, 0.0, 0.0, 0.0],
        "把全部截面塌成质心一个点，弯矩静默变零",
    ),
    (
        MONOTONIC,
        "loaded_point_stresses_n_mm2",
        list(reversed(MONOTONIC.expected["loaded_point_stresses_n_mm2"])),
        "点序反了却沿用旧历史槽",
    ),
    (
        MONOTONIC,
        "loaded_point_plastic_strains",
        [MONOTONIC.expected["loaded_point_plastic_strains"][0]]
        * len(MONOTONIC.expected["loaded_point_plastic_strains"]),
        "所有积分点错误共享一份可变材料历史",
    ),
    (
        SPRINGBACK,
        "fiber_springback_curvature_per_mm",
        0.0,
        "卸载时丢掉塑性历史，于是伪造零回弹曲率",
    ),
    (
        SPRINGBACK,
        "same_curvature_different_history_differs",
        False,
        "把点应力当当前曲率的无历史函数",
    ),
)


@pytest.mark.parametrize(("oracle", "quantity", "wrong", "reason"), RED_MATRIX)
def test_each_stage_four_gate_has_a_known_wrong_implementation_that_must_red(
    oracle, quantity, wrong, reason
):
    """判据本身被验：六种典型错法逐格必须红。"""

    try:
        oracle.check(quantity, wrong)
    except OracleError:
        return
    pytest.fail(f"这条错实现没有被判红：{reason}")
