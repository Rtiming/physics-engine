#!/usr/bin/env python3
"""导轮轴偏斜引起的带材稳态横漂——**闭式，独立于被验内核**。

## 一、它是一个梁-弦边值问题，不是"需要材料输运"的问题

WDS `research/05`第三节把"材料注入"列为横漂定量化的最大缺口，理由是
"持续横走稳态需要带材真的流过轮面"。**那条对瞬态与持续横走成立，对稳态不成立。**

Shelton正规入轮定律说：**稳态当且仅当带材垂直于（倾斜后的）下游轮轴入轮**，
即``y'(L) = θ_r``。**那个条件本身就是输运的结果**——把它当成边界条件写下来，
稳态解就是一个静力边值问题，不需要把材料搬过去。

自由跨段满足梁-弦方程（`research/04`第2节）：

    EI₁·y'''' − T·y'' = 0,    K = sqrt(T/EI₁),    u = K·L

四个边界条件：

| 条件 | 出处 |
|---|---|
| ``y(0) = 0`` | 上游包覆的stick弧把位置钉住 |
| ``y'(0) = 0`` | 同上，出射方向也被冻结 |
| ``y''(L) = 0`` | 下游入轮线经短包覆微滑，**不能传递弯矩** |
| ``y'(L) = θ_r`` | **正规入轮**：稳态时带材垂直于倾斜后的下游轮轴入轮 |

## 二、解出来的闭式

通解``y = C₁ + C₂x + C₃cosh(Kx) + C₄sinh(Kx)``代入四条边界条件，得

    y_ss = θ_r · L · f(u),    f(u) = (sinh u − u·cosh u) / (u·(1 − cosh u))

**两个极限各有物理意义**：

* ``f(0) = 2/3``——纯梁极限（弯曲主导，回正最强）；
* ``f(∞) = 1``——纯弦极限（``y_ss = θ_r·L``，带材径直沿倾斜方向走）。

``f``在两者之间**单调递增**。于是一条反直觉但要紧的推论：

> **提高张力不会减小稳态横漂，反而略微增大它。**

张力越大``u``越大、越接近纯弦，弯曲的回正作用越小。
这条对张力算法直接相关：**横漂不能靠加张力压下去**。

`research/04`第2节明写本工位"处在梁弦之间的中间区，
**纯弦或纯梁公式均不适用，必须带f(KL)**"。

## 三、与research/04的五个独立数字对拍

本页的闭式与WDS `research/04`第2节引的数逐个核过：

| 量 | research/04 | 本页 |
|---|---|---|
| 硬轴``EI₁ = E·t·w³/12`` | ≈8.0×10⁴ N·mm² | 8.000×10⁴ |
| ``1/K``（T=20N） | ≈63 mm | 63.25 |
| ``u``区间（L=100—500mm） | ≈1.6—8 | 1.58—7.91 |
| ``f``区间 | ≈0.68—0.88 | 0.692—0.874 |
| ``y_ss``（L=200mm、θ_r=1°） | ≈2.6 mm | 2.6114 |

## 四、生成器不做什么

不调`solve_equilibrium`、不调任何能量项、不引任何引擎的力学模块。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/roller_skew_lateral_drift"
ALGORITHM_VERSION = "1.0.0"

#: REBCO带材：4 mm宽×0.1 mm厚，``E ≈ 150 GPa``。**假设输入**——
#: `research/05`第三节把带材EA/EI/GJ列为"只有现场实测能补"的五项之一。
YOUNGS_MODULUS_N_MM2 = 150.0e3
THICKNESS_MM = 0.1
WIDTH_MM = 4.0
#: 硬轴（面内横向弯曲）：`I₁ = t·w³/12`。它控制横漂与蹭边的跨段刚度。
BENDING_STIFFNESS_NMM2 = YOUNGS_MODULUS_N_MM2 * THICKNESS_MM * WIDTH_MM**3 / 12.0
AXIAL_STIFFNESS_N = YOUNGS_MODULUS_N_MM2 * THICKNESS_MM * WIDTH_MM

FREE_SPAN_MM = 200.0
TENSION_N = 20.0
SKEW_DEG = 1.0
#: **半间隙有两个，而带材两个都要过。2026-08-17订正：先撞的不是原来写的那一个。**
#:
#: 本案例最初只写了6.5 mm，出处记的是"真机导轮有效宽度17 mm、带宽4 mm"。
#: 那个数属实，但它是**现场两只可调平轮**（Ø100×17、平面无凸缘）的半宽——
#: 带材**先**过的是张力机上的R4→R1，而它们是**带凸缘的槽轮**。
#:
#: R1槽型在消费方WII自己展示的那份`static_009_new1.stl`上被复量过
#: （`docs/records/r1-contact-radius-cross-check-20260726.md`第104—111行），
#: 并与WDS从另一份网格提取的`config/roller_flange_profile.v1.json`互证：
#: 槽底半径45.224 mm、**槽底宽度8.000 mm**、凸缘半径60.398 mm、外径120.796 mm。
#:
#: 于是4 mm带材在R1里的半间隙是`(8.000 − 4.0)/2 = 2.0 mm`，
#: **只有平轮那条的0.3077倍**。原案例把更松的那一条当成了约束。
R1_GROOVE_HALF_CLEARANCE_MM = (8.0 - WIDTH_MM) / 2.0
FLAT_GUIDE_HALF_CLEARANCE_MM = (17.0 - WIDTH_MM) / 2.0
#: 起约束作用的是更小的那个。**这一行是本案例的判据入口，不许改成另一个。**
HALF_CLEARANCE_MM = min(R1_GROOVE_HALF_CLEARANCE_MM, FLAT_GUIDE_HALF_CLEARANCE_MM)


def beam_string_factor(u: float) -> float:
    """``f(u) = (sinh u − u·cosh u) / (u·(1 − cosh u))``，``u → 0``时取``2/3``。"""

    if u < 1.0e-6:
        return 2.0 / 3.0 + u * u / 45.0
    return (math.sinh(u) - u * math.cosh(u)) / (u * (1.0 - math.cosh(u)))


def steady_drift_mm(*, skew_rad: float, span_mm: float, tension_n: float) -> float:
    u = math.sqrt(tension_n / BENDING_STIFFNESS_NMM2) * span_mm
    return skew_rad * span_mm * beam_string_factor(u)


def critical_skew_deg(*, half_clearance_mm: float, span_mm: float) -> float:
    """反解``y_ss = 半间隙``那个偏斜角。**半间隙是显式入参，不许取模块常量。**

    做成入参的理由就是本案例被订正的那件事：原来它取一个写死的6.5，
    而那个6.5是**另一只轮**的半宽。半间隙一旦是入参，
    "这条判据说的是哪只轮"就必须由调用方每次写出来。
    """

    u = math.sqrt(TENSION_N / BENDING_STIFFNESS_NMM2) * span_mm
    return math.degrees(half_clearance_mm / (span_mm * beam_string_factor(u)))


def main() -> int:
    skew = math.radians(SKEW_DEG)
    u = math.sqrt(TENSION_N / BENDING_STIFFNESS_NMM2) * FREE_SPAN_MM
    oracles = [
        {
            "id": "oracle:drift/beam_string_factor_limits",
            "inputs": {"kind": "f_of_u_limits"},
            "expected": {
                "pure_beam_limit": 2.0 / 3.0,
                "at_1p6": beam_string_factor(1.6),
                "at_8": beam_string_factor(8.0),
                "at_100": beam_string_factor(100.0),
            },
            "tolerances": {
                "pure_beam_limit": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``u → 0``的解析极限，纯梁；判据是闭式自洽不是数值过程",
                },
                "at_1p6": {"abs": 0.0, "rel": 1.0e-15, "reason": "闭式求值"},
                "at_8": {"abs": 0.0, "rel": 1.0e-15, "reason": "同上"},
                "at_100": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``u → ∞``趋于1（纯弦）；100处已到0.99",
                },
            },
        },
        {
            "id": "oracle:drift/steady_state_at_one_degree",
            "inputs": {
                "kind": "shelton_normal_entry",
                "skew_deg": SKEW_DEG,
                "free_span_mm": FREE_SPAN_MM,
                "tension_n": TENSION_N,
                "bending_stiffness_nmm2": BENDING_STIFFNESS_NMM2,
            },
            "expected": {
                "u": u,
                "factor": beam_string_factor(u),
                "drift_mm": steady_drift_mm(
                    skew_rad=skew, span_mm=FREE_SPAN_MM, tension_n=TENSION_N
                ),
            },
            "tolerances": {
                "u": {"abs": 0.0, "rel": 1.0e-15, "reason": "纯几何与材料常数"},
                "factor": {"abs": 0.0, "rel": 1.0e-15, "reason": "闭式求值"},
                "drift_mm": {
                    "abs": 0.0, "rel": 3.0e-4,
                    "reason": (
                        "引擎侧是DER弯曲＋轴向张力的准静态解，**二阶收敛**"
                        "（N=20/40/80/160/320实测比值3.739/3.861/3.928/3.964）。"
                        "3e-4是N=160实测1.319e-4的约2.3倍余量；"
                        "**收敛阶另有一条门判**，这一条只判落点"
                    ),
                },
            },
        },
        {
            "id": "oracle:drift/tension_makes_it_worse_not_better",
            "inputs": {
                "kind": "monotonicity_of_f",
                "tensions_n": [10.0, 20.0, 30.0, 40.0],
                "free_span_mm": FREE_SPAN_MM,
                "skew_deg": SKEW_DEG,
            },
            "expected": {
                "drift_at_10n": steady_drift_mm(skew_rad=skew, span_mm=FREE_SPAN_MM, tension_n=10.0),
                "drift_at_20n": steady_drift_mm(skew_rad=skew, span_mm=FREE_SPAN_MM, tension_n=20.0),
                "drift_at_30n": steady_drift_mm(skew_rad=skew, span_mm=FREE_SPAN_MM, tension_n=30.0),
                "drift_at_40n": steady_drift_mm(skew_rad=skew, span_mm=FREE_SPAN_MM, tension_n=40.0),
            },
            "tolerances": {
                f"drift_at_{n}n": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "闭式求值；本条判的是**单调性方向**，值本身只是它的载体",
                }
                for n in (10, 20, 30, 40)
            },
        },
        {
            "id": "oracle:drift/skew_that_reaches_the_flange",
            "inputs": {
                "kind": "critical_skew_for_rub",
                "half_clearance_mm": HALF_CLEARANCE_MM,
                "r1_groove_half_clearance_mm": R1_GROOVE_HALF_CLEARANCE_MM,
                "flat_guide_half_clearance_mm": FLAT_GUIDE_HALF_CLEARANCE_MM,
                "tension_n": TENSION_N,
                "spans_mm": [100.0, 200.0, 300.0, 500.0],
            },
            "expected": {
                **{
                    f"critical_skew_deg_at_{int(span)}mm": critical_skew_deg(
                        half_clearance_mm=HALF_CLEARANCE_MM, span_mm=span
                    )
                    for span in (100.0, 200.0, 300.0, 500.0)
                },
                #: 更松的那一条也留着，**不是为了完整，是为了让订正可判**：
                #: 两组的比值必须恒为两个半间隙之比（`f(u)`与半间隙无关），
                #: 于是"谁先撞"这件事有一条能红的判据，而不只是一句话。
                **{
                    f"flat_guide_critical_skew_deg_at_{int(span)}mm": critical_skew_deg(
                        half_clearance_mm=FLAT_GUIDE_HALF_CLEARANCE_MM, span_mm=span
                    )
                    for span in (100.0, 200.0, 300.0, 500.0)
                },
            },
            "tolerances": {
                **{
                    f"critical_skew_deg_at_{int(span)}mm": {
                        "abs": 0.0, "rel": 1.0e-15,
                        "reason": (
                            "闭式反解；**它是装配公差的直接输入**，所以单列一条。"
                            "半间隙取R1槽底的2.0 mm——**2026-08-17订正**，"
                            "此前取的6.5 mm是现场平轮的半宽，那是更松的一条"
                        ),
                    }
                    for span in (100.0, 200.0, 300.0, 500.0)
                },
                **{
                    f"flat_guide_critical_skew_deg_at_{int(span)}mm": {
                        "abs": 0.0, "rel": 1.0e-15,
                        "reason": "更松的那一条，留着让「谁先撞」有一条能红的判据",
                    }
                    for span in (100.0, 200.0, 300.0, 500.0)
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/roller_skew_lateral_drift",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/roller_skew_lateral_drift/generate_oracle.py",
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
