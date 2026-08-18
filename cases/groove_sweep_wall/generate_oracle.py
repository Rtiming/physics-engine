#!/usr/bin/env python3
"""外倾锥面槽壁 vs 平面环带——**闭式静力学，独立于被验内核**。

本脚本一行都不import`physics_engine.contact`。它解的是同一道题的手推答案，
conformance测试再拿内核去撞它。

## 一、几何与三条闭式

截面里记``u``（横向）、``v``（深度）。第σ侧壁的间隙

    g_σ(u, v) = w/2 + v·tanα − (σ·u + r)

罚势``U = ½k g²``（``g < 0``），故接触力``F = −k·g·∇g``、``∇g = tanα·n − σ·s``。
拆到两个轴上：

    F_横向 = k·|g|            （**与tanα无关**）
    F_举升 = k·|g|·tanα       （平面环带恒为0）
    |F|    = k·|g|·sec α

### 闭式一：深度冻结时，锥面与平面的横向力**恒等**，差别全在举升上

这条是本案例最容易被说反的一条。"锥面给的回正力更软"这句话
**在深度被钉住时是错的**——那时两者的横向力逐位相同。
锥面真正做的事是**把一部分接触力转成把带材举出槽的分量**，
而"回正力变软"是那个举升分量在深度自由时反过来改变了平衡位置的结果，
不是间隙公式本身给的。**两句话差一个"深度自由"的前提，判据必须把它写出来。**

### 闭式二：深度自由 + 压紧力 ``F_hold`` ⟹ 锥面的横向力**饱和**

带材被``F_hold``压在槽底，横向被推。深度方向的平衡：

    k·|g|·tanα = F_hold          （带材离开槽底之后）
 ⟹ F_横向 = k·|g| = F_hold / tanα      **与横移量无关**

平面环带（``tanα = 0``）没有举升分量，带材永远压在槽底，于是

    F_横向 = k·(u + r − w/2)      **随横移线性无界上升**

**这就是plans/15第2.2条要的那句"锥面给的是逐渐爬升的回正力、平面给的是硬碰"
的定量形式**：一个是常数、一个是斜率``k``的直线，两者之比随横移线性发散。

### 闭式三：爬出槽口的横移阈值

壁只在``v ≤ depth_max``那一段存在。把闭式二的``|g| = F_hold/(k·tanα)``
代回间隙式并令``v = depth_max``：

    u_逃逸 = w/2 + depth_max·tanα − r + F_hold/(k·tanα)

超过它，带材爬出槽口、这面壁不再存在，**平衡问题本身失去良态**
（求解器当场报奇异）。**这不是数值故障，是"带材跳出槽"这件事的物理**。

## 二、冻结帧丢掉的那一项（决策0075第四节）

    ∇g_精确 = ∇g_冻结 − A·t,   A = τ·(tanα·u + σ·v) / (1 − u·κ_s − v·κ_n)

代入plans/14第2.2节的实测不变量（`v2-01-bracket`：``R = 145.6 mm``那一档的
``ε_edge`` p100行、``τ_max = 2.550 °/mm``），本清单冻结的是
``|A| / |∇g_冻结|``与由它得出的**力方向偏角**。

## 三、金标为什么是闭式而不是实测

上面三条全是初等代数，没有一处求积、没有一处迭代。
**实测数不作金标**（spec/08规则1）——内核的数拿来撞它，不是反过来。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/groove_sweep_wall"
ALGORITHM_VERSION = "1.0.0"

#: 槽底宽8.000 mm（plans/15第2.2条实测） ⟹ 半宽4.0；4 mm带材 ⟹ 边缘半偏移2.0。
HALF_WIDTH_MM = 4.0
EDGE_RADIUS_MM = 2.0
#: **声明的设计参数，不是量出来的**——真角度要等工件CAD进来，见案例页第四节。
WALL_ANGLE_DEG = 10.0
SLOPE = math.tan(math.radians(WALL_ANGLE_DEG))
STIFFNESS_N_PER_MM = 5.0e3
DEPTH_MAX_MM = 6.0
DEPTH_MIN_MM = -1.0
HOLD_DOWN_N = 3.0

#: plans/14第2.2节：`v2-01-bracket`的``ε_edge`` p100行R=145.6 mm、``τ_max``=2.550 °/mm。
#: **曲率与扭率是两个独立的量**（同节末段"两个量不同向"），所以金标用
#: "圆弧 + 绕切向按``τ·a``自转的帧"，两者各取各的实测值，而不用螺旋线
#: （螺旋线把``κ``与``τ``锁死成一个比值）。
GOLD_RADIUS_MM = 145.6
GOLD_TWIST_DEG_PER_MM = 2.550
GOLD_ARC_MM = 37.0
GOLD_LATERAL_MM = 2.3
GOLD_DEPTH_MM = 0.8

FROZEN_LATERALS = (2.2, 2.5, 3.0)
FREE_LATERALS = (2.05, 2.2, 2.5, 2.8, 3.0)


def frozen_depth_row(lateral: float) -> dict[str, float]:
    """深度钉在``v = 0``：横向力两者恒等，锥面多一个举升分量。"""

    gap = HALF_WIDTH_MM - (lateral + EDGE_RADIUS_MM)
    lateral_force = STIFFNESS_N_PER_MM * -gap
    return {
        "gap_mm": gap,
        "plane_lateral_force_n": lateral_force,
        "cone_lateral_force_n": lateral_force,
        "cone_lift_force_n": lateral_force * SLOPE,
        "cone_over_plane_magnitude": 1.0 / math.cos(math.radians(WALL_ANGLE_DEG)),
    }


def free_depth_row(lateral: float) -> dict[str, float]:
    """深度自由、压紧``F_hold``：平面线性上升，锥面饱和在``F_hold/tanα``。"""

    plane = STIFFNESS_N_PER_MM * (lateral + EDGE_RADIUS_MM - HALF_WIDTH_MM)
    cone = HOLD_DOWN_N / SLOPE
    #: 带材爬到哪个深度：把``|g| = F_hold/(k tanα)``代回间隙式解``v``。
    depth = (lateral + EDGE_RADIUS_MM - HALF_WIDTH_MM - HOLD_DOWN_N / (STIFFNESS_N_PER_MM * SLOPE)) / SLOPE
    return {
        "plane_lateral_force_n": plane,
        "cone_lateral_force_n": cone,
        "cone_depth_mm": depth,
        "plane_over_cone": plane / cone,
    }


def escape_lateral_mm() -> float:
    return (
        HALF_WIDTH_MM
        + DEPTH_MAX_MM * SLOPE
        - EDGE_RADIUS_MM
        + HOLD_DOWN_N / (STIFFNESS_N_PER_MM * SLOPE)
    )


def frame_residual() -> dict[str, float]:
    """冻结帧丢掉的那一项，代入plans/14实测不变量。"""

    twist = math.radians(GOLD_TWIST_DEG_PER_MM)
    angle = GOLD_ARC_MM / GOLD_RADIUS_MM
    phase = twist * GOLD_ARC_MM
    #: ``κ⃗ = t' = −(1/R)·(外向径向)``；帧绕切向自转``ψ = τ·a``后投到``(s, n)``：
    #: ``κ_n = −(1/R)·cos ψ``、``κ_s = +(1/R)·sin ψ``（右手系``s = n × t``）。
    curvature = 1.0 / GOLD_RADIUS_MM
    curvature_s = curvature * math.sin(phase)
    curvature_n = -curvature * math.cos(phase)
    jacobian = 1.0 - GOLD_LATERAL_MM * curvature_s - GOLD_DEPTH_MM * curvature_n
    coefficient = -twist * (SLOPE * GOLD_LATERAL_MM + 1.0 * GOLD_DEPTH_MM) / jacobian
    frozen_norm = math.sqrt(1.0 + SLOPE * SLOPE)
    del angle
    return {
        "curvature_s_per_mm": curvature_s,
        "curvature_n_per_mm": curvature_n,
        "twist_rad_per_mm": -twist,
        "jacobian": jacobian,
        "coefficient_a": coefficient,
        "frozen_gradient_norm": frozen_norm,
        "relative_loss": abs(coefficient) / frozen_norm,
        "force_tilt_deg": math.degrees(math.atan2(abs(coefficient), frozen_norm)),
    }


def main() -> int:
    oracles: list[dict] = []

    frozen_reason = (
        "**零容差**：全部是初等代数（一次减法、一次乘法），金标与内核走同一串"
        "IEEE-754基本运算，没有求积没有迭代没有开方。这条判据的价值不在数值精度，"
        "在于它钉住了一句最容易被说反的话——**深度钉住时锥面与平面的横向力恒等**。"
        "把它写成非零容差等于承认自己不知道这条是不是恒等式。"
    )
    for lateral in FROZEN_LATERALS:
        row = frozen_depth_row(lateral)
        oracles.append(
            {
                "id": f"oracle:groove_sweep_wall/frozen_depth_{str(lateral).replace('.', 'p')}",
                "inputs": {
                    "kind": "frozen_depth_statics",
                    "lateral_mm": lateral,
                    "depth_mm": 0.0,
                    "wall_angle_deg": WALL_ANGLE_DEG,
                    "note": "深度钉在槽底：差别全在举升分量上，横向分量逐位相同。",
                },
                "expected": row,
                "tolerances": {
                    key: {"rel": 0.0, "abs": 0.0, "reason": frozen_reason}
                    for key in ("gap_mm", "plane_lateral_force_n", "cone_lateral_force_n")
                }
                | {
                    key: {
                        "rel": 1.0e-14,
                        "abs": 0.0,
                        "reason": (
                            "带一次``tan``与一次``cos``的超越函数求值，两侧各自调"
                            "``math``一次，误差≤1 ulp；留两个数量级余量到1e-14。"
                            "**不写零容差**：超越函数不在IEEE-754的正确舍入承诺里。"
                        ),
                    }
                    for key in ("cone_lift_force_n", "cone_over_plane_magnitude")
                },
            }
        )

    free_reason = (
        "深度自由那一档是**牛顿解出来的平衡**，位置带``O(1/k)``的罚柔度误差，"
        "力不带（0050第二节：平衡时``k·δ``精确等于理论法向力）。故力判1e-9相对、"
        "深度判1e-6相对——**两者不是同一个精度档，写在一起才不会被人抄错**。"
        "1e-9是牛顿残差判据``1e-11 N``除以本构型的力尺度(约17 N)再留两个量级；"
        "1e-6是深度尺度(约6 mm)上同一残差经``1/(k·tanα)``放大后的量级。"
    )
    for lateral in FREE_LATERALS:
        row = free_depth_row(lateral)
        oracles.append(
            {
                "id": f"oracle:groove_sweep_wall/free_depth_{str(lateral).replace('.', 'p')}",
                "inputs": {
                    "kind": "free_depth_statics",
                    "lateral_mm": lateral,
                    "hold_down_n": HOLD_DOWN_N,
                    "wall_angle_deg": WALL_ANGLE_DEG,
                    "note": "深度自由：平面线性上升、锥面饱和在 F_hold/tanα。",
                },
                "expected": row,
                "tolerances": {
                    "plane_lateral_force_n": {"rel": 1.0e-9, "abs": 0.0, "reason": free_reason},
                    "cone_lateral_force_n": {"rel": 1.0e-9, "abs": 0.0, "reason": free_reason},
                    "cone_depth_mm": {"rel": 1.0e-6, "abs": 0.0, "reason": free_reason},
                    "plane_over_cone": {"rel": 1.0e-9, "abs": 0.0, "reason": free_reason},
                },
            }
        )

    oracles.append(
        {
            "id": "oracle:groove_sweep_wall/escape_threshold",
            "inputs": {
                "kind": "escape_threshold",
                "hold_down_n": HOLD_DOWN_N,
                "depth_max_mm": DEPTH_MAX_MM,
                "wall_angle_deg": WALL_ANGLE_DEG,
                "note": "超过它带材爬出槽口，壁不再存在，平衡问题失去良态。",
            },
            "expected": {"escape_lateral_mm": escape_lateral_mm()},
            "tolerances": {
                "escape_lateral_mm": {
                    "rel": 1.0e-9,
                    "abs": 0.0,
                    "reason": (
                        "阈值本身是闭式；内核侧用二分把"
                        "**收敛/求解器报奇异**这条定性分界夹到1e-6 mm，"
                        "再与闭式比。1e-9是给闭式求值本身的容差（一次tan），"
                        "夹逼宽度另有一条独立判据，不混在这一条里。"
                    ),
                }
            },
        }
    )

    residual = frame_residual()
    oracles.append(
        {
            "id": "oracle:groove_sweep_wall/frozen_frame_residual",
            "inputs": {
                "kind": "frozen_frame_residual",
                "radius_mm": GOLD_RADIUS_MM,
                "twist_deg_per_mm": GOLD_TWIST_DEG_PER_MM,
                "arc_mm": GOLD_ARC_MM,
                "lateral_mm": GOLD_LATERAL_MM,
                "depth_mm": GOLD_DEPTH_MM,
                "wall_angle_deg": WALL_ANGLE_DEG,
                "note": "决策0075第四节的闭式；不变量取plans/14第2.2节实测值。",
            },
            "expected": residual,
            "tolerances": {
                key: {
                    "rel": 1.0e-9,
                    "abs": 0.0,
                    "reason": (
                        "闭式一侧只有三角函数与四则运算；内核一侧是解析曲线上的"
                        "**数值中心差分**（`tests/test_contact_groove_sweep.py`实测"
                        "二阶收敛比4.0000），故本条判的是闭式自身的可复现性，"
                        "不是差分的收敛——那一条在单元门里判，**两件事不混**。"
                    ),
                }
                for key in residual
            },
        }
    )

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/groove_sweep_wall",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/groove_sweep_wall/generate_oracle.py",
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
