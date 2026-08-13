#!/usr/bin/env python3
"""矩形理想弹塑性纤维截面的独立解析金标（决策0059）。

本脚本不import``physics_engine.sections``。全部计算用``fractions.Fraction``：

* 连续矩形截面在``kappa = 2*kappa_y``时，弹性核半高恰为``c/2``，
  闭式弯矩``M = 11*M_p/12``；
* 8/16/32/64个等面积中点纤维的屈服边界都恰落在单元边界，故每档纤维应力、
  塑性应变、弯矩与离散回弹曲率都是精确有理数；
* 卸载不发生反向屈服时，``M_unload = M_loaded + E*I_f*(kappa-kappa_loaded)``，
  令它为零即可得离散截面的自由回弹曲率。

生产代码走逐点return-map与区间保护Newton平衡；金标走分段闭式与精确求和，
两边不共享数值实现。
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/rectangular_section_springback"
ALGORITHM_VERSION = "1.0.0"

WIDTH_MM = Fraction(12)
THICKNESS_MM = Fraction(4)
YOUNG_N_MM2 = Fraction(200_000)
YIELD_N_MM2 = Fraction(250)
POINT_COUNTS = (8, 16, 32, 64)
HALF_HEIGHT_MM = THICKNESS_MM / 2
YIELD_CURVATURE_PER_MM = YIELD_N_MM2 / (YOUNG_N_MM2 * HALF_HEIGHT_MM)
LOADED_CURVATURE_PER_MM = 2 * YIELD_CURVATURE_PER_MM


def _float(value: Fraction) -> float:
    return float(value)


def _clip(value: Fraction, low: Fraction, high: Fraction) -> Fraction:
    return max(low, min(high, value))


def fibre_loading(point_count: int) -> dict[str, object]:
    """处女态单调加载到``2*kappa_y``，全程精确有理算术。"""

    depth = THICKNESS_MM / point_count
    area = WIDTH_MM * depth
    coordinates = [
        -HALF_HEIGHT_MM + (Fraction(index) + Fraction(1, 2)) * depth for index in range(point_count)
    ]
    stresses: list[Fraction] = []
    plastic_strains: list[Fraction] = []
    yielded: list[bool] = []
    for coordinate in coordinates:
        strain = LOADED_CURVATURE_PER_MM * coordinate
        trial = YOUNG_N_MM2 * strain
        stress = _clip(trial, -YIELD_N_MM2, YIELD_N_MM2)
        stresses.append(stress)
        plastic_strains.append(strain - stress / YOUNG_N_MM2)
        yielded.append(abs(trial) > YIELD_N_MM2)

    moment = sum(
        stress * area * coordinate for stress, coordinate in zip(stresses, coordinates, strict=True)
    )
    second_moment = sum(area * coordinate * coordinate for coordinate in coordinates)
    springback_curvature = LOADED_CURVATURE_PER_MM - moment / (YOUNG_N_MM2 * second_moment)
    springback_stresses = [
        YOUNG_N_MM2 * (springback_curvature * coordinate - plastic_strain)
        for coordinate, plastic_strain in zip(coordinates, plastic_strains, strict=True)
    ]
    springback_moment = sum(
        stress * area * coordinate
        for stress, coordinate in zip(springback_stresses, coordinates, strict=True)
    )
    assert springback_moment == 0
    assert all(abs(stress) <= YIELD_N_MM2 for stress in springback_stresses)
    return {
        "coordinates": coordinates,
        "stresses": stresses,
        "plastic_strains": plastic_strains,
        "yielded": yielded,
        "moment": moment,
        "second_moment": second_moment,
        "springback_curvature": springback_curvature,
        "springback_stresses": springback_stresses,
    }


def main() -> int:
    fibres = {count: fibre_loading(count) for count in POINT_COUNTS}
    continuum_second_moment = WIDTH_MM * THICKNESS_MM**3 / 12
    plastic_moment = WIDTH_MM * YIELD_N_MM2 * HALF_HEIGHT_MM**2
    continuum_loaded_moment = Fraction(11, 12) * plastic_moment
    elastic_yield_moment = Fraction(2, 3) * plastic_moment
    continuum_springback = LOADED_CURVATURE_PER_MM - continuum_loaded_moment / (
        YOUNG_N_MM2 * continuum_second_moment
    )
    moment_errors = [continuum_loaded_moment - fibres[count]["moment"] for count in POINT_COUNTS]
    moment_error_ratios = [
        moment_errors[index] / moment_errors[index + 1] for index in range(len(moment_errors) - 1)
    ]
    assert moment_error_ratios == [Fraction(4), Fraction(4), Fraction(4)]

    finest = fibres[POINT_COUNTS[-1]]
    shared_inputs = {
        "width_mm": _float(WIDTH_MM),
        "thickness_mm": _float(THICKNESS_MM),
        "young_modulus_n_mm2": _float(YOUNG_N_MM2),
        "yield_stress_n_mm2": _float(YIELD_N_MM2),
        "point_counts": list(POINT_COUNTS),
        "loaded_curvature_per_mm": _float(LOADED_CURVATURE_PER_MM),
        "yield_curvature_per_mm": _float(YIELD_CURVATURE_PER_MM),
        "integration_rule_id": "section_rule/midpoint_equal_area/1",
    }
    monotonic = {
        "id": "oracle:section/monotonic_bending",
        "inputs": shared_inputs,
        "expected": {
            "axial_force_n": 0.0,
            "continuum_loaded_moment_n_mm": _float(continuum_loaded_moment),
            "elastic_yield_moment_n_mm": _float(elastic_yield_moment),
            "plastic_moment_n_mm": _float(plastic_moment),
            "fiber_loaded_moments_n_mm": [
                _float(fibres[count]["moment"]) for count in POINT_COUNTS
            ],
            "moment_error_ratios": [_float(value) for value in moment_error_ratios],
            "loaded_point_stresses_n_mm2": [_float(value) for value in finest["stresses"]],
            "loaded_point_plastic_strains": [_float(value) for value in finest["plastic_strains"]],
            "loaded_point_yielded": list(finest["yielded"]),
        },
        "tolerances": {
            "axial_force_n": {
                "abs": 1.0e-12,
                "rel": 0.0,
                "reason": "对称截面纯弯的轴力解析为零；绝对容差约束浮点求和残差。",
            },
            "continuum_loaded_moment_n_mm": {
                "abs": 2.0,
                "rel": 0.0,
                "reason": (
                    "连续矩形闭式在kappa=2*kappa_y时为11*M_p/12=11000 N·mm；"
                    "64点中点纤维精确误差1.953125 N·mm，abs=2给2.4%离散余量。"
                    "线弹性EI*kappa会给16000，离边界2500倍以上。"
                ),
            },
            "elastic_yield_moment_n_mm": {
                "abs": 2.0,
                "rel": 0.0,
                "reason": (
                    "连续截面初屈服弯矩2*M_p/3=8000 N·mm；64点中点规则在纯弹性y²积分上"
                    "误差1.953125 N·mm，abs=2与加载点取同一离散余量。"
                ),
            },
            "plastic_moment_n_mm": {
                "abs": 1.0e-10,
                "rel": 0.0,
                "reason": (
                    "完全塑性时应力为正负常数，中点规则在上下半截各自精确积分线性杠杆；"
                    "1e-10只留浮点求和余量。"
                ),
            },
            "fiber_loaded_moments_n_mm": {
                "abs": 1.0e-10,
                "rel": 2.0e-15,
                "reason": (
                    "独立Fraction逐纤维精确和；生产端只含基本浮点运算。"
                    "绝对项照顾接近零的扩展构型，相对项约18个单位舍入。"
                ),
            },
            "moment_error_ratios": {
                "abs": 1.0e-10,
                "rel": 0.0,
                "reason": (
                    "屈服边界对8/16/32/64点都落在纤维边界，中点规则误差精确按h^2缩放，"
                    "故加密一倍误差比为4；1e-10只留浮点相减与相除余量。"
                ),
            },
            "loaded_point_stresses_n_mm2": {
                "abs": 1.0e-11,
                "rel": 2.0e-15,
                "reason": ("64点全分布对独立Fraction clip；必须验点级分布，不能只验汇总弯矩。"),
            },
            "loaded_point_plastic_strains": {
                "abs": 1.0e-17,
                "rel": 3.0e-15,
                "reason": "每点return-map历史对精确有理值；绝对项覆盖应为零的内核点。",
            },
            "loaded_point_yielded": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "64个布尔屈服标志逐位比较，内32点弹性、外32点塑性。",
            },
        },
    }
    springback = {
        "id": "oracle:section/free_springback",
        "inputs": {
            **shared_inputs,
            "point_count": POINT_COUNTS[-1],
            "target_moment_n_mm": 0.0,
            "curvature_bracket_per_mm": [0.0, _float(LOADED_CURVATURE_PER_MM)],
            "residual_tol_n_mm": 1.0e-9,
        },
        "expected": {
            "solver_converged": True,
            "equilibrium_moment_n_mm": 0.0,
            "fiber_springback_curvature_per_mm": _float(finest["springback_curvature"]),
            "continuum_springback_curvature_per_mm": _float(continuum_springback),
            "springback_point_stresses_n_mm2": [
                _float(value) for value in finest["springback_stresses"]
            ],
            "point_history_unchanged_on_elastic_unload": True,
            "replay_bytes_equal": True,
            "same_curvature_different_history_differs": True,
        },
        "tolerances": {
            "solver_converged": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "定性判据：明确区间夹根时局部平衡必须收敛。",
            },
            "equilibrium_moment_n_mm": {
                "abs": 1.0e-9,
                "rel": 0.0,
                "reason": "与调用方声明的局部平衡残差容差逐字相同，不拿曲率误差代替平衡。",
            },
            "fiber_springback_curvature_per_mm": {
                "abs": 1.0e-15,
                "rel": 0.0,
                "reason": (
                    "64点离散截面的精确有理回弹曲率；生产端二分到1e-9 N·mm，"
                    "对应曲率误差远低于1e-15/mm。"
                ),
            },
            "continuum_springback_curvature_per_mm": {
                "abs": 6.0e-8,
                "rel": 0.0,
                "reason": (
                    "连续闭式为1/2560；64点离散误差5.723443e-8/mm，"
                    "abs=6e-8给4.8%离散余量，丢历史会给0并失败。"
                ),
            },
            "springback_point_stresses_n_mm2": {
                "abs": 1.0e-9,
                "rel": 2.0e-15,
                "reason": (
                    "独立Fraction卸载分布；绝对项由1e-9 N·mm平衡容差经最大纤维杠杆反推后宽取。"
                ),
            },
            "point_history_unchanged_on_elastic_unload": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "本卸载段无反向屈服，trial/commit不能凭空改塑性历史。",
            },
            "replay_bytes_equal": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "同一路径的State规范字节必须逐字节复现。",
            },
            "same_curvature_different_history_differs": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "同一曲率、处女态与塑性加载史必须给出不同点应力和状态。",
            },
        },
    }
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/rectangular_section_springback",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/rectangular_section_springback/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [monotonic, springback],
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote 2 oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
