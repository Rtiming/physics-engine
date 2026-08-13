"""截面积分点与一维弹塑性本构的单元门（决策0059）。

本文件先于实现落地并实跑为红：最初收集阶段因
``ModuleNotFoundError: physics_engine.sections``失败。它守的不是“多写一个积分循环”，
而是阶段4的四条承重边界：积分点不冒充全局自由度、材料历史显式进``State``、
截面分布真的非线性、广义曲率能由截面平衡解出来。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.sections import (
    ElasticPerfectlyPlastic1D,
    LinearElastic1D,
    RectangularFiberSection,
    SectionError,
    build_rectangular_section_layout,
    evaluate_section_response,
    solve_section_curvature,
)
from physics_engine.state import State, StateField, StateLayout

WIDTH_MM = 12.0
THICKNESS_MM = 4.0
YOUNG_N_MM2 = 200_000.0
YIELD_N_MM2 = 250.0
POINTS = 64
YIELD_CURVATURE_PER_MM = YIELD_N_MM2 / (YOUNG_N_MM2 * THICKNESS_MM / 2.0)
LOADED_CURVATURE_PER_MM = 2.0 * YIELD_CURVATURE_PER_MM


def _model(point_count: int = POINTS):
    section = RectangularFiberSection(
        section_id="section/rectangular-test",
        width_mm=WIDTH_MM,
        thickness_mm=THICKNESS_MM,
        point_count=point_count,
    )
    layout = build_rectangular_section_layout(
        layout_id=f"layout/rectangular-section-n{point_count}",
        section=section,
    )
    material = ElasticPerfectlyPlastic1D(
        young_modulus_n_mm2=YOUNG_N_MM2,
        yield_stress_n_mm2=YIELD_N_MM2,
    )
    return section, layout, material


def test_integration_points_add_history_slots_but_not_global_unknowns():
    """两个广义变形是运动学坐标；每点两项材料历史不增加全局运动自由度。"""

    section, section_layout, _ = _model()
    points = section.integration_points

    assert len(points) == POINTS
    assert [point.index for point in points] == list(range(POINTS))
    assert sum(point.area_mm2 for point in points) == WIDTH_MM * THICKNESS_MM
    for left, right in zip(points, reversed(points), strict=True):
        assert left.y_mm == -right.y_mm
        assert left.area_mm2 == right.area_mm2

    assert section_layout.generalized_dof_count == 2
    assert section_layout.history_scalar_count == 2 * POINTS
    assert section_layout.layout.dof_count == 2 + 2 * POINTS
    assert section_layout.layout.history_fields() == (
        "section_point_plastic_strain",
        "section_point_accumulated_plastic_strain",
    )
    assert section_layout.plastic_strain_index(0) == 2
    assert section_layout.plastic_strain_index(POINTS - 1) == 1 + POINTS
    assert section_layout.accumulated_plastic_strain_index(0) == 2 + POINTS
    assert section_layout.accumulated_plastic_strain_index(POINTS - 1) == 1 + 2 * POINTS

    state = section_layout.initial_state()
    assert state.block("section_axial_strain") == (0.0,)
    assert state.block("section_curvature_per_mm") == (0.0,)
    assert state.block("section_point_plastic_strain") == (0.0,) * POINTS
    assert state.layout.fingerprint() == section_layout.layout.fingerprint()


def test_section_reference_and_quadrature_identity_are_bound_into_the_layout():
    """同点数不代表同截面；厚度或规则变了，旧点历史绝不能按旧索引套过去。"""

    _, first, material = _model()
    changed_section = RectangularFiberSection(
        section_id="section/rectangular-test",
        width_mm=WIDTH_MM,
        thickness_mm=THICKNESS_MM * 2.0,
        point_count=POINTS,
    )
    changed = build_rectangular_section_layout(
        layout_id=f"layout/rectangular-section-n{POINTS}",
        section=changed_section,
    )

    assert first.layout.fingerprint() != changed.layout.fingerprint()
    with pytest.raises(SectionError, match="layout"):
        evaluate_section_response(
            section_layout=first,
            material=material,
            previous_state=changed.initial_state(),
            axial_strain=0.0,
            curvature_per_mm=0.0,
        )


def test_mixed_elastic_plastic_distribution_and_tangent_are_real():
    """外层屈服、内层弹性；切线须对住独立有限差分，不能退回一个等效EI常数。"""

    _, section_layout, material = _model()
    virgin = section_layout.initial_state()
    curvature = 1.6 * YIELD_CURVATURE_PER_MM
    response = evaluate_section_response(
        section_layout=section_layout,
        material=material,
        previous_state=virgin,
        axial_strain=0.0,
        curvature_per_mm=curvature,
    )

    assert abs(response.axial_force_n) < 1.0e-12
    assert any(point.yielded for point in response.points)
    assert any(not point.yielded for point in response.points)
    assert max(abs(point.stress_n_mm2) for point in response.points) == YIELD_N_MM2
    assert 0.0 < response.bending_tangent_n_mm2 < (YOUNG_N_MM2 * WIDTH_MM * THICKNESS_MM**3 / 12.0)

    step = 1.0e-9
    low = evaluate_section_response(
        section_layout=section_layout,
        material=material,
        previous_state=virgin,
        axial_strain=0.0,
        curvature_per_mm=curvature - step,
    )
    high = evaluate_section_response(
        section_layout=section_layout,
        material=material,
        previous_state=virgin,
        axial_strain=0.0,
        curvature_per_mm=curvature + step,
    )
    finite_difference = (high.bending_moment_n_mm - low.bending_moment_n_mm) / (2.0 * step)
    assert finite_difference == pytest.approx(
        response.bending_tangent_n_mm2, rel=2.0e-9, abs=1.0e-5
    )


def test_explicit_linear_elastic_material_has_no_yield_sentinel_or_history():
    section, section_layout, _ = _model(point_count=8)
    material = LinearElastic1D(young_modulus_n_mm2=YOUNG_N_MM2)
    virgin = section_layout.initial_state()
    curvature = 100.0 * YIELD_CURVATURE_PER_MM

    response = evaluate_section_response(
        section_layout=section_layout,
        material=material,
        previous_state=virgin,
        axial_strain=0.0,
        curvature_per_mm=curvature,
    )
    discrete_second_moment = sum(
        point.area_mm2 * point.y_mm**2 for point in section.integration_points
    )

    assert response.bending_moment_n_mm == pytest.approx(
        YOUNG_N_MM2 * discrete_second_moment * curvature,
        rel=2.0e-15,
    )
    assert response.bending_tangent_n_mm2 == pytest.approx(
        YOUNG_N_MM2 * discrete_second_moment,
        rel=2.0e-15,
    )
    assert not any(point.yielded for point in response.points)
    assert response.next_state.block("section_point_plastic_strain") == (0.0,) * 8
    assert response.next_state.block("section_point_accumulated_plastic_strain") == (0.0,) * 8

    contaminated = list(virgin.vector)
    contaminated[section_layout.plastic_strain_index(0)] = 1.0e-6
    contaminated[section_layout.accumulated_plastic_strain_index(0)] = 1.0e-6
    with pytest.raises(SectionError, match="linear elastic material"):
        evaluate_section_response(
            section_layout=section_layout,
            material=material,
            previous_state=State(section_layout.layout, tuple(contaminated)),
            axial_strain=0.0,
            curvature_per_mm=curvature,
        )


def test_plastic_history_is_explicit_replayable_and_path_dependent():
    """同一零曲率，处女态与“加载后卸回”必须给出不同应力；同一路径逐字节复现。"""

    _, section_layout, material = _model()
    virgin = section_layout.initial_state()

    def load_and_unload():
        loaded = evaluate_section_response(
            section_layout=section_layout,
            material=material,
            previous_state=virgin,
            axial_strain=0.0,
            curvature_per_mm=LOADED_CURVATURE_PER_MM,
        )
        unloaded = evaluate_section_response(
            section_layout=section_layout,
            material=material,
            previous_state=loaded.next_state,
            axial_strain=0.0,
            curvature_per_mm=0.0,
        )
        return loaded, unloaded

    loaded_a, unloaded_a = load_and_unload()
    loaded_b, unloaded_b = load_and_unload()
    virgin_zero = evaluate_section_response(
        section_layout=section_layout,
        material=material,
        previous_state=virgin,
        axial_strain=0.0,
        curvature_per_mm=0.0,
    )

    assert loaded_a.next_state.pack() == loaded_b.next_state.pack()
    assert unloaded_a.next_state.pack() == unloaded_b.next_state.pack()
    assert unloaded_a.next_state.pack() != virgin_zero.next_state.pack()
    assert unloaded_a.bending_moment_n_mm != 0.0
    assert virgin_zero.bending_moment_n_mm == 0.0
    assert any(value != 0.0 for value in loaded_a.next_state.block("section_point_plastic_strain"))
    assert all(
        value >= 0.0
        for value in loaded_a.next_state.block("section_point_accumulated_plastic_strain")
    )


def test_springback_curvature_is_solved_from_section_equilibrium():
    """加载后的自由回弹解``M=0``；曲率是求出来的，不是从解析式抄进生产代码。"""

    _, section_layout, material = _model()
    loaded = evaluate_section_response(
        section_layout=section_layout,
        material=material,
        previous_state=section_layout.initial_state(),
        axial_strain=0.0,
        curvature_per_mm=LOADED_CURVATURE_PER_MM,
    )
    solved = solve_section_curvature(
        section_layout=section_layout,
        material=material,
        previous_state=loaded.next_state,
        axial_strain=0.0,
        target_moment_n_mm=0.0,
        curvature_bracket_per_mm=(0.0, LOADED_CURVATURE_PER_MM),
        residual_tol_n_mm=1.0e-9,
        max_iterations=80,
    )

    assert solved.converged, solved.reason
    assert solved.iterations <= 8, "64点回弹有可用一致切线；若仍耗40次纯二分，逐点装配会被无谓重复"
    assert 0.0 < solved.curvature_per_mm < LOADED_CURVATURE_PER_MM
    assert abs(solved.residual_n_mm) <= 1.0e-9
    assert solved.response.next_state.block("section_curvature_per_mm") == (
        solved.curvature_per_mm,
    )
    assert solved.response.next_state.layout.fingerprint() == loaded.next_state.layout.fingerprint()


def test_invalid_geometry_material_state_and_solver_bracket_fail_closed():
    with pytest.raises(SectionError, match="point_count"):
        RectangularFiberSection(
            section_id="section/bad",
            width_mm=1.0,
            thickness_mm=1.0,
            point_count=1,
        )
    with pytest.raises(SectionError, match="young_modulus"):
        ElasticPerfectlyPlastic1D(
            young_modulus_n_mm2=math.inf,
            yield_stress_n_mm2=1.0,
        )

    _, section_layout, material = _model(point_count=8)
    wrong_layout = StateLayout(
        layout_id="layout/wrong-section-state",
        fields=(StateField("curvature_per_mm", 1),),
    )
    with pytest.raises(SectionError, match="layout"):
        evaluate_section_response(
            section_layout=section_layout,
            material=material,
            previous_state=State(wrong_layout, (0.0,)),
            axial_strain=0.0,
            curvature_per_mm=0.0,
        )

    bad_history = list(section_layout.initial_state().vector)
    bad_history[section_layout.accumulated_plastic_strain_index(0)] = -1.0
    with pytest.raises(SectionError, match="accumulated plastic strain"):
        evaluate_section_response(
            section_layout=section_layout,
            material=material,
            previous_state=State(section_layout.layout, tuple(bad_history)),
            axial_strain=0.0,
            curvature_per_mm=0.0,
        )

    with pytest.raises(SectionError, match="does not bracket"):
        solve_section_curvature(
            section_layout=section_layout,
            material=material,
            previous_state=section_layout.initial_state(),
            axial_strain=0.0,
            target_moment_n_mm=1.0e9,
            curvature_bracket_per_mm=(-1.0e-3, 1.0e-3),
            residual_tol_n_mm=1.0e-9,
        )
