#!/usr/bin/env python3
"""绞盘张力比的金标生成器——**闭式解，独立于被验内核**。

绕在固定圆柱上的柔性带材，全段滑移时张紧端与松弛端的张力比是
Euler-Eytelwein（绞盘）公式：

    T2 / T1 = exp(mu * theta)

它是连续极限。本仓的杆是**离散**的，所以真正该判的是离散关系，
而连续式只作为离散式的极限：

## 一、单节点的精确关系（本案例判据强度最高的两条）

一个接触节点两侧各有一段带材，张力分别是``T⁻``、``T⁺``，
两段各偏离节点处切线``Δφ/2``。对该节点做力平衡：

    法向    N = (T⁻ + T⁺) · sin(Δφ/2)
    切向    (T⁺ − T⁻) · cos(Δφ/2) = μ·N          （全滑移，摩擦力取满）

两式联立消去``N``：

    T⁺ / T⁻ = (1 + μ·tan(Δφ/2)) / (1 − μ·tan(Δφ/2))

**这两条都是精确的代数恒等式，不是近似。** 法向那条只用到几何，
切向那条只额外用到"全滑移⟹摩擦取满"。

## 二、连续极限

``tan(Δφ/2) → Δφ/2``时逐节点比``→ 1 + μΔφ``，``N``个节点连乘``→ exp(μθ)``。
所以连续式是离散式的极限，**不是另一个独立判据**——
把连续式直接当逐节点判据用会引入``O(Δφ²)``的系统偏差，
而那个偏差在``Δφ = π/32``时是``2.5e-4``相对，**大于本案例的收敛判据要分辨的量**。

## 三、生成器不做什么

不调`solve_equilibrium`、不调任何接触项、不引任何引擎的力学模块。
它只写上面三条代数式。**判据的独立性来自这一条**（轴7规则3）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/capstan_tension_ratio"
ALGORITHM_VERSION = "1.0.0"

#: 几何取真机导轮（`winding-machine/HARDWARE_TOPOLOGY.md` 2026-07-07现场确认）：
#: 外径100 mm ⟹ 半径50 mm；有效宽度17 mm ⟹ 轴向半宽8.5 mm。
RADIUS_MM = 50.0
HALF_WIDTH_MM = 8.5
#: 包角90°：导轮转向的常见值。**它是设定不是实测**——真机穿带顺序R4→R1的
#: 各轮包角没有实测记录（0062第二节裁决2的五项数据缺口之一）。
WRAP_RAD = math.pi / 2.0
#: 带-轮摩擦系数：**假设输入**。WDS `research/05`第三节把``μ_∥``/``μ_⊥``
#: 列为"只有现场实测能补"的五项之一，所以本案例永久是`hypothesis_only`。
FRICTION = 0.30


def node_ratio(segments: int) -> float:
    """逐节点张力比``(1 + μ·tan(Δφ/2)) / (1 − μ·tan(Δφ/2))``。"""

    half = math.tan(WRAP_RAD / segments / 2.0)
    return (1.0 + FRICTION * half) / (1.0 - FRICTION * half)


def main() -> int:
    continuum = math.exp(FRICTION * WRAP_RAD)
    oracles = [
        {
            "id": "oracle:capstan/node_tension_ratio",
            "inputs": {
                "kind": "discrete_capstan_node_ratio",
                "friction_coefficient": FRICTION,
                "wrap_angle_rad": WRAP_RAD,
                "segments": 8,
            },
            "expected": {
                "node_ratio": node_ratio(8),
                "half_angle_tangent": math.tan(WRAP_RAD / 8 / 2.0),
            },
            "tolerances": {
                "node_ratio": {
                    "abs": 0.0,
                    "rel": 6.0e-3,
                    "reason": (
                        "引擎侧是准静态载荷步连续化到全滑移，**载荷步误差是一阶**"
                        "（步数翻倍误差减半，2026-08-17实测比2.10/2.06）。"
                        "8段240载荷步在包角中点实测4.256e-3，取1.4倍余量。"
                        "**收敛阶本身另有一条门判**，这一条只判落点——"
                        "只判落点会被一个碰巧的常数骗过，只判阶会被系统偏移骗过"
                    ),
                },
                "half_angle_tangent": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "纯几何，双精度往返即可",
                },
            },
        },
        {
            "id": "oracle:capstan/normal_force_kink",
            "inputs": {
                "kind": "discrete_capstan_normal_force",
                "wrap_angle_rad": WRAP_RAD,
                "segments": 8,
            },
            "expected": {"half_angle_sine": math.sin(WRAP_RAD / 8 / 2.0)},
            "tolerances": {
                "half_angle_sine": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "纯几何常数",
                },
            },
        },
        {
            "id": "oracle:capstan/cone_saturation",
            "inputs": {
                "kind": "full_slip_cone_saturation",
                "friction_coefficient": FRICTION,
            },
            "expected": {"tangential_over_mu_normal": 1.0},
            "tolerances": {
                "tangential_over_mu_normal": {
                    "abs": 1.0e-12, "rel": 0.0,
                    "reason": (
                        "全滑移下每个接触**精确落在摩擦锥面上**，"
                        "这是return-map理想塑性修正的定义性质而不是收敛结果。"
                        "判绝对不判相对：真值恰为1，且它是本案例唯一"
                        "**不随载荷步数变化**的判据"
                    ),
                },
            },
        },
        {
            "id": "oracle:capstan/continuum_limit",
            "inputs": {
                "kind": "euler_eytelwein",
                "friction_coefficient": FRICTION,
                "wrap_angle_rad": WRAP_RAD,
            },
            "expected": {
                "continuum_ratio": continuum,
                "discrete_16": node_ratio(16) ** 16,
                "discrete_32": node_ratio(32) ** 32,
                "discrete_64": node_ratio(64) ** 64,
            },
            "tolerances": {
                "continuum_ratio": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``exp``的双精度求值",
                },
                "discrete_16": {"abs": 0.0, "rel": 1.0e-15, "reason": "同上"},
                "discrete_32": {"abs": 0.0, "rel": 1.0e-15, "reason": "同上"},
                "discrete_64": {"abs": 0.0, "rel": 1.0e-15, "reason": "同上"},
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/capstan_tension_ratio",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/capstan_tension_ratio/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
