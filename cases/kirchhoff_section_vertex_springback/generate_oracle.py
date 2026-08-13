#!/usr/bin/env python3
"""三节点Kirchhoff纤维截面回弹的独立有理数金标（决策0060）。

本脚本不import``physics_engine.section_beam``或``physics_engine.sections``。
平面三节点几何、矩形中点纤维、理想弹塑性首次加载和弹性卸载全部用
``fractions.Fraction``独立计算；生产端走二阶jet、return-map与全局Newton。

第二条oracle是2026-08-13从WDS提交``c1b8fe6``两份干净物理源只读生成的兼容夹具。
它不是“WDS已迁移”的证据，只钉住第一片复用的状态次序、easy-axis曲率与弹性导数。
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/kirchhoff_section_vertex_springback"
ALGORITHM_VERSION = "1.0.0"

LENGTH_MM = Fraction(100)
WIDTH_MM = Fraction(12)
THICKNESS_MM = Fraction(4)
POINT_COUNT = 64
YOUNG_N_MM2 = Fraction(200_000)
YIELD_N_MM2 = Fraction(250)
LOADED_CURVATURE_PER_MM = Fraction(1, 800)


def _float(value: Fraction) -> float:
    return float(value)


def _clip(value: Fraction, low: Fraction, high: Fraction) -> Fraction:
    return max(low, min(high, value))


def vertical_displacement(curvature_per_mm: Fraction) -> Fraction:
    """精确反解``kappa = 2v/[L(sqrt(L²+v²)+L)]``。

    令``q=kappa*L=2*tan(theta/2)``，则``v/L=tan(theta)=q/(1-q²/4)``；
    因此输入曲率为有理数时位移仍是有理数，不需要用同一个浮点sqrt自证。
    """

    q = curvature_per_mm * LENGTH_MM
    return LENGTH_MM * q / (1 - q * q / 4)


def section_loading() -> dict[str, object]:
    depth = THICKNESS_MM / POINT_COUNT
    area = WIDTH_MM * depth
    half = THICKNESS_MM / 2
    coordinates = [
        -half + (Fraction(index) + Fraction(1, 2)) * depth
        for index in range(POINT_COUNT)
    ]
    stresses: list[Fraction] = []
    plastic: list[Fraction] = []
    yielded: list[bool] = []
    tangent_terms: list[Fraction] = []
    second_moment = sum(area * y * y for y in coordinates)
    for y in coordinates:
        strain = LOADED_CURVATURE_PER_MM * y
        trial = YOUNG_N_MM2 * strain
        stress = _clip(trial, -YIELD_N_MM2, YIELD_N_MM2)
        is_yielded = abs(trial) > YIELD_N_MM2
        stresses.append(stress)
        plastic.append(strain - stress / YOUNG_N_MM2)
        yielded.append(is_yielded)
        tangent_terms.append((0 if is_yielded else YOUNG_N_MM2) * area * y * y)
    moment = sum(
        stress * area * y for stress, y in zip(stresses, coordinates, strict=True)
    )
    section_tangent = sum(tangent_terms)
    elastic_tangent = YOUNG_N_MM2 * second_moment
    springback_curvature = LOADED_CURVATURE_PER_MM - moment / elastic_tangent
    springback_stresses = [
        YOUNG_N_MM2 * (springback_curvature * y - ep)
        for y, ep in zip(coordinates, plastic, strict=True)
    ]
    assert sum(
        stress * area * y
        for stress, y in zip(springback_stresses, coordinates, strict=True)
    ) == 0
    assert all(abs(stress) <= YIELD_N_MM2 for stress in springback_stresses)
    return {
        "coordinates": coordinates,
        "plastic": plastic,
        "yielded": yielded,
        "moment": moment,
        "section_tangent": section_tangent,
        "springback_curvature": springback_curvature,
    }


def global_derivatives(moment: Fraction, section_tangent: Fraction) -> tuple[Fraction, Fraction]:
    """加载构型下``dU/dv``与``d²U/dv²``，含几何刚度项。"""

    kappa = LOADED_CURVATURE_PER_MM
    q = kappa * LENGTH_MM
    a_kappa_sq = q * q / 4
    first = (1 - a_kappa_sq) ** 2 / (LENGTH_MM**2 * (1 + a_kappa_sq))
    second = -(
        kappa
        * (3 + a_kappa_sq)
        * (1 - a_kappa_sq) ** 3
        / (2 * LENGTH_MM**2 * (1 + a_kappa_sq) ** 3)
    )
    force = LENGTH_MM * moment * first
    tangent = LENGTH_MM * (section_tangent * first * first + moment * second)
    return force, tangent


def main() -> int:
    section = section_loading()
    force, tangent = global_derivatives(section["moment"], section["section_tangent"])
    loaded_v = vertical_displacement(LOADED_CURVATURE_PER_MM)
    springback_kappa = section["springback_curvature"]
    springback_v = vertical_displacement(springback_kappa)
    global_oracle = {
        "id": "oracle:section/global_vertex_springback",
        "inputs": {
            "length_mm": _float(LENGTH_MM),
            "width_mm": _float(WIDTH_MM),
            "thickness_mm": _float(THICKNESS_MM),
            "point_count": POINT_COUNT,
            "young_modulus_n_mm2": _float(YOUNG_N_MM2),
            "yield_stress_n_mm2": _float(YIELD_N_MM2),
            "loaded_curvature_per_mm": _float(LOADED_CURVATURE_PER_MM),
            "fixed_axial_strain": 0.0,
            "free_global_index": 7,
            "kinematics_id": "wds/bergou-kappa1/1",
        },
        "expected": {
            "loaded_vertical_displacement_mm": _float(loaded_v),
            "loaded_curvature_per_mm": _float(LOADED_CURVATURE_PER_MM),
            "loaded_moment_n_mm": _float(section["moment"]),
            "loaded_generalized_force_n": _float(force),
            "loaded_tangent_n_per_mm": _float(tangent),
            "loaded_yielded_point_count": sum(section["yielded"]),
            "solver_converged": True,
            "springback_vertical_displacement_mm": _float(springback_v),
            "springback_curvature_per_mm": _float(springback_kappa),
            "springback_moment_n_mm": 0.0,
            "history_unchanged_on_elastic_unload": True,
            "replay_bytes_equal": True,
            "failed_trial_does_not_commit": True,
        },
        "tolerances": {
            "loaded_vertical_displacement_mm": {
                "abs": 2.0e-14,
                "rel": 0.0,
                "reason": "由Bergou曲率的有理反函数精确生成；容差约2 ulp。",
            },
            "loaded_curvature_per_mm": {
                "abs": 2.0e-18,
                "rel": 0.0,
                "reason": "有理反函数后的生产sqrt路径；容差约9 ulp，足以判红小转角替代式。",
            },
            "loaded_moment_n_mm": {
                "abs": 1.0e-10,
                "rel": 2.0e-15,
                "reason": "64点Fraction纤维逐项精确和；线弹性EI会偏约5002 Nmm。",
            },
            "loaded_generalized_force_n": {
                "abs": 2.0e-11,
                "rel": 3.0e-14,
                "reason": "独立有理链式法则dU/dv；约百牛量级保留数十ulp。",
            },
            "loaded_tangent_n_per_mm": {
                "abs": 2.0e-9,
                "rel": 3.0e-12,
                "reason": "独立有理二阶链式法则，且包含M乘曲率二阶导的几何刚度项。",
            },
            "loaded_yielded_point_count": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "屈服前沿由精确有理比较决定，整数逐位一致。",
            },
            "solver_converged": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "释放一个全局节点坐标后必须由全局平衡求解器收敛。",
            },
            "springback_vertical_displacement_mm": {
                "abs": 2.0e-10,
                "rel": 2.0e-12,
                "reason": "Fraction截面回弹曲率再经Bergou有理反函数；容差含1e-9 N求解残差。",
            },
            "springback_curvature_per_mm": {
                "abs": 2.0e-14,
                "rel": 0.0,
                "reason": "全局Newton结果对独立Fraction回弹曲率；容差由声明力残差保守反推。",
            },
            "springback_moment_n_mm": {
                "abs": 2.0e-7,
                "rel": 0.0,
                "reason": "全局残差是M乘非零几何Jacobian；该弯矩界由1e-9 N残差反推并宽取。",
            },
            "history_unchanged_on_elastic_unload": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "该卸载段没有反向屈服，逐点塑性历史必须逐位不变。",
            },
            "replay_bytes_equal": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "相同加载与全局回弹路径的State规范字节必须逐字节复现。",
            },
            "failed_trial_does_not_commit": {
                "abs": 0.0,
                "rel": 0.0,
                "reason": "限制为一次Newton而失败时，提交态必须逐字节等于输入态。",
            },
        },
    }

    wds_fixture = {
        "id": "oracle:section/wds_easy_axis_fixture",
        "inputs": {
            "source_commit": "c1b8fe6",
            "state_source_sha256": "ea61bf2611ce30fb91248f9092d5cdf2eff82a0688926253ac9e929b30577c27",
            "energies_source_sha256": "2d3e4d1784c94898dd2efb185091e29c2041fe4513e569e0fb0e5b99c1ed7d77",
            "captured_at": "2026-08-13",
            "positions_mm": [[0.0, 0.0, 0.0], [80.0, 1.0, 2.0], [155.0, 12.0, 7.0]],
            "edge_twist_angles": [0.23, -0.17],
            "rest_lengths_mm": [80.0, 75.0],
            "reference_d1": [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            "reference_d2": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            "natural_kappa1": 0.0,
            "elastic_ei_easy_n_mm2": 1000.0,
            "state_pack_order": "x.ravel_then_gamma",
        },
        "expected": {
            "curvature_per_mm": 0.001697465689615586,
            "elastic_energy_n_mm": 0.11165385348760701,
            "elastic_gradient": [
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
            ],
        },
        "tolerances": {
            "curvature_per_mm": {
                "abs": 2.0e-18,
                "rel": 0.0,
                "reason": "WDS material_strains只读实测值；约5 ulp并能抓符号、frame或dual-length错误。",
            },
            "elastic_energy_n_mm": {
                "abs": 1.0e-15,
                "rel": 3.0e-15,
                "reason": "WDS BendingEnergy以EI_easy=1000、EI_hard=0只读实测值。",
            },
            "elastic_gradient": {
                "abs": 3.0e-14,
                "rel": 3.0e-13,
                "reason": "WDS二阶AD能量项的一阶通道只读实测；覆盖9位置加2扭角的打包次序。",
            },
        },
    }

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/kirchhoff_section_vertex_springback",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/kirchhoff_section_vertex_springback/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [global_oracle, wds_fixture],
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote 2 oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
