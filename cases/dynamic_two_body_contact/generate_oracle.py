#!/usr/bin/env python3
"""P3-M3 oracle: two equal masses release one millimetre of normal overlap."""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402


def main() -> int:
    stiffness = 100.0
    effective_mass = 0.5
    omega = math.sqrt(1000.0 * stiffness / effective_mass)
    quarter_period = math.pi / (2.0 * omega)
    steps = 400
    expected_speed = 0.5 * omega
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/dynamic_two_body_contact",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": "algorithm:oracle/dynamic_two_body_contact",
            "algorithm_version": "1.0.0",
            "path_relative": "cases/dynamic_two_body_contact/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": [
            {
                "id": "oracle:dynamic-contact/eccentric-wrench",
                "inputs": {
                    "radius_mm": 10.0,
                    "centre_a_x_mm": -9.5,
                    "centre_b_x_mm": 9.5,
                    "centroid_y_mm": 2.0,
                    "normal_stiffness_n_per_mm": stiffness,
                    "normal_damping_n_s_per_mm": 0.0,
                },
                "expected": {
                    "penetration_mm": 1.0,
                    "normal_ab": [-1.0, 0.0, 0.0],
                    "witness_a_mm": [0.5, 0.0, 0.0],
                    "witness_b_mm": [-0.5, 0.0, 0.0],
                    "force_a_world_n": [-100.0, 0.0, 0.0],
                    "force_b_world_n": [100.0, 0.0, 0.0],
                    "lever_a_world_mm": [10.0, -2.0, 0.0],
                    "lever_b_world_mm": [-10.0, -2.0, 0.0],
                    "torque_a_body_nmm": [0.0, 0.0, -200.0],
                    "torque_b_body_nmm": [0.0, 0.0, 200.0],
                    "total_force_world_n": [0.0, 0.0, 0.0],
                    "total_torque_about_origin_world_nmm": [0.0, 0.0, 0.0],
                },
                "tolerances": {
                    name: {
                        "abs": 1.0e-13 if "total_torque" in name else 1.0e-14,
                        "rel": 1.0e-14,
                        "reason": "球心、见证点、作用反作用与r×F的独立手算。",
                    }
                    for name in (
                        "penetration_mm",
                        "normal_ab",
                        "witness_a_mm",
                        "witness_b_mm",
                        "force_a_world_n",
                        "force_b_world_n",
                        "lever_a_world_mm",
                        "lever_b_world_mm",
                        "torque_a_body_nmm",
                        "torque_b_body_nmm",
                        "total_force_world_n",
                        "total_torque_about_origin_world_nmm",
                    )
                },
            },
            {
                "id": "oracle:dynamic-contact/aligned-quarter-period",
                "inputs": {
                    "mass_a_kg": 1.0,
                    "mass_b_kg": 1.0,
                    "effective_mass_kg": effective_mass,
                    "initial_penetration_mm": 1.0,
                    "normal_stiffness_n_per_mm": stiffness,
                    "omega_rad_per_s": omega,
                    "quarter_period_s": quarter_period,
                    "dt_s": quarter_period / steps,
                    "steps": steps,
                },
                "expected": {
                    "final_centre_a_x_mm": -10.0,
                    "final_centre_b_x_mm": 10.0,
                    "final_velocity_a_x_mm_per_s": -expected_speed,
                    "final_velocity_b_x_mm_per_s": expected_speed,
                    "total_linear_momentum_kg_mm_per_s": [0.0, 0.0, 0.0],
                    "final_kinetic_energy_nmm": 50.0,
                    "derivative_evaluations": 4 * steps,
                    "renormalisations_per_body": steps,
                },
                "tolerances": {
                    "final_centre_a_x_mm": {
                        "abs": 2.0e-8,
                        "rel": 2.0e-9,
                        "reason": "δ(t)=δ0 cos(ωt)在四分之一周期归零。",
                    },
                    "final_centre_b_x_mm": {
                        "abs": 2.0e-8,
                        "rel": 2.0e-9,
                        "reason": "等质量对称性。",
                    },
                    "final_velocity_a_x_mm_per_s": {
                        "abs": 5.0e-6,
                        "rel": 2.0e-8,
                        "reason": "相对速度δ0ω由两等质量各分一半。",
                    },
                    "final_velocity_b_x_mm_per_s": {
                        "abs": 5.0e-6,
                        "rel": 2.0e-8,
                        "reason": "作用反作用与对称性。",
                    },
                    "total_linear_momentum_kg_mm_per_s": {
                        "abs": 1.0e-12,
                        "rel": 1.0e-12,
                        "reason": "只有内力，初始总动量为零。",
                    },
                    "final_kinetic_energy_nmm": {
                        "abs": 5.0e-6,
                        "rel": 1.0e-7,
                        "reason": "初始罚簧能0.5kδ²=50N·mm。",
                    },
                    "derivative_evaluations": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "经典RK4每步四次耦合导数求值。",
                    },
                    "renormalisations_per_body": {
                        "abs": 0.0,
                        "rel": 0.0,
                        "reason": "每个完整步分别归一化两个四元数。",
                    },
                },
            },
        ],
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote 2 oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
