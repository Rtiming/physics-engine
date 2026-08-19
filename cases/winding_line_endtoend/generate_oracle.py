#!/usr/bin/env python3
"""放线—导向—张力—收线端到端的金标生成器——**闭式，独立于被验内核**。

本案例把四件事装到一条链路上跑一次：放线端的张力（**由张力回路算出来的**力边界条件）、
导向轮上的罚接触与库仑摩擦、收线端的位移控制（卷走带材）、
以及排线横动把带材边缘顶上法兰内环面。

## 六条闭式（前四条是2026-08-17第一版，后两条是丁1加的）

### 一、放线端张力恰为设定值

它是**力边界条件**，不是解出来的量。写成判据是为了挡住一类实现错误：
把外载在装配里乘错单位或漏掉某一段。

### 二、落位点张力 = 放线端 × exp(μθ)

绞盘（Euler-Eytelwein）。**方向由带材的走向定**：带材朝收线端滑，
摩擦朝放线端，于是张力**沿走向递增**。

    T_落位 = T_放线 · exp(μ·θ)

这条同时是`cases/capstan_tension_ratio`那条式子在整条链路上的样子，
也是本轮审计里"传感器位置≠被控点"那条缺漏的落点：
传感器装在放线端读到30 N，而落位点上是48 N。

### 三、半间隙 = (槽宽 − 带宽)/2

排线横动超过它，带材边缘才顶上法兰。真机：槽宽17 mm、带宽4 mm ⟹ **6.5 mm**。

### 四、蹭边力 = 张力 × 越界量 / 自由段长度

带材从最后一个接触点到落位点是一段受张力的直段，长``L``。
落位点被法兰顶住偏离直线``δ``时，张力的横向分量是``T·δ/L``（小角）。
平衡时它等于蹭边力：

    F_蹭 = T_落位 · (|横动| − 半间隙) / L

**这条说的是"蹭边力正比于张力"**——张力调高，蹭边就更狠。
它是小角近似，`sin θ ≈ θ`的误差是``O((δ/L)²)``；δ=1.5 mm、L=9.8 mm时约1.2%。

### 五、放线端的张力**由驱动链算出来**（2026-08-18，决策0088丁1）

本案例第一版把放线端张力写成一个字面量``30.0``。S6.3的欠账原话是
"`drives`产出的是一个力的数值，**尚无案例让它经`PointLoad`真的加载在带材上
并与接触联立**"。丁1接的就是这条线：张力回路跑到稳态，它的输出经`PointLoad`
进`solve_equilibrium`，与导向轮的罚接触和库仑摩擦一起解。

闭式是回路自己的稳态，**与求解器无关**：积分作用下稳态读数等于设定值，于是

    T_传感器 = 设定值
    T_被控点 = 设定值 / measurement_transfer = 设定值 · exp(μθ)

``measurement_transfer = exp(−μθ)``是"传感器在放线端、被控点在落位点"这个
构型的声明。**这条闭式与第二条是两条独立的腿**：一条是驱动链自己的记账
（一个换算比），一条是求解器解出来的接触与摩擦。两条必须给出同一个落位点张力，
而它们之间不共享任何一行代码。

### 六、传递比方向搞反时误差是**平方**

把``measurement_transfer``写成``exp(+μθ)``（"传感器在落位点、被控点在放线端"）
时，被控点张力算出来是``设定值·exp(−μθ)``，与正确值差``exp(2μθ)``倍。
μ=0.3、θ=90°时是**2.566**。这个数与S6.4能力位记的``r² = 2.566``是同一个。

## 生成器不做什么

不调`solve_equilibrium`、不调任何接触项、不引任何引擎的力学模块。
**`drives`那一路也不引**：第五条闭式是手写的``设定值·exp(μθ)``，
金标里不出现`TensionLoop`。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/winding_line_endtoend"
ALGORITHM_VERSION = "1.1.0"

#: 真机导轮（`winding-machine/HARDWARE_TOPOLOGY.md`，2026-07-07现场确认）：
#: 外径100 mm ⟹ 半径50 mm；有效宽度17 mm ⟹ 槽半宽8.5 mm。
ROLLER_RADIUS_MM = 50.0
CHANNEL_HALF_WIDTH_MM = 8.5
#: 4 mm宽REBCO带材 ⟹ 半宽2.0 mm。
TAPE_HALF_WIDTH_MM = 2.0
#: 包角90°与摩擦0.30都是**设定/假设**，不是实测（0062第二节裁决2）。
WRAP_RAD = math.pi / 2.0
FRICTION = 0.30
WRAP_SEGMENTS = 8
#: 放线端张力：真机张力区间10—30 N的上端。**它同时是张力回路的设定值**——
#: 丁1之后这个数不再直接当力边界条件用，而是回路的``setpoint_n``。
PAYOUT_TENSION_N = 30.0


def main() -> int:
    chord = 2.0 * ROLLER_RADIUS_MM * math.sin(WRAP_RAD / WRAP_SEGMENTS / 2.0)
    half_clearance = CHANNEL_HALF_WIDTH_MM - TAPE_HALF_WIDTH_MM
    capstan = math.exp(FRICTION * WRAP_RAD)
    oracles = [
        {
            "id": "oracle:line/payout_tension_is_the_boundary_condition",
            "inputs": {"kind": "applied_load", "tension_n": PAYOUT_TENSION_N},
            "expected": {"payout_tension_n": PAYOUT_TENSION_N},
            "tolerances": {
                "payout_tension_n": {
                    "abs": 0.0, "rel": 1.0e-9,
                    "reason": (
                        "它是**力边界条件**不是解出来的量：外载多大，"
                        "第一段的轴力就多大。相对1e-9只留给准静态残差，"
                        "**不留给建模误差**——这一条红了说明装配里有一段力丢了或乘错了"
                    ),
                },
            },
        },
        {
            "id": "oracle:line/lay_point_tension_over_payout",
            "inputs": {
                "kind": "euler_eytelwein_along_the_line",
                "friction_coefficient": FRICTION,
                "wrap_angle_rad": WRAP_RAD,
            },
            "expected": {
                "capstan_ratio": capstan,
                "lay_point_tension_n": PAYOUT_TENSION_N * capstan,
            },
            "tolerances": {
                "capstan_ratio": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``exp``的双精度求值",
                },
                "lay_point_tension_n": {
                    "abs": 0.0, "rel": 6.0e-3,
                    "reason": (
                        "引擎侧是位移控制的收线连续化到全滑移，**误差随收线步数下降**。"
                        "480步实测比值1.604693、相对偏差1.695e-3，取约3.5倍余量。"
                        "**这一条同时是本轮审计缺漏A的落点**：传感器在放线端读30 N，"
                        "而落位点上是48 N——闭环调的是它测到的量"
                    ),
                },
            },
        },
        {
            "id": "oracle:line/half_clearance",
            "inputs": {
                "kind": "channel_minus_tape",
                "channel_half_width_mm": CHANNEL_HALF_WIDTH_MM,
                "tape_half_width_mm": TAPE_HALF_WIDTH_MM,
            },
            "expected": {"half_clearance_mm": half_clearance},
            "tolerances": {
                "half_clearance_mm": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**零容差**：它是两个声明尺寸的差，没有任何数值过程参与。"
                        "排线横动不到6.5 mm就一点蹭边力都没有，超过就有——"
                        "阈值两侧定性相反，与`incline_slide_threshold`同口径"
                    ),
                },
            },
        },
        {
            "id": "oracle:line/rub_force_scales_with_tension",
            "inputs": {
                "kind": "lateral_component_of_tension",
                "free_span_mm": chord,
                "traverse_mm": [7.0, 8.0],
            },
            "expected": {
                "free_span_mm": chord,
                "overshoot_at_seven_mm": 7.0 - half_clearance,
                "overshoot_at_eight_mm": 8.0 - half_clearance,
            },
            "tolerances": {
                "free_span_mm": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "弦长闭式，纯几何",
                },
                "overshoot_at_seven_mm": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "两个声明尺寸的差",
                },
                "overshoot_at_eight_mm": {
                    "abs": 0.0, "rel": 1.0e-15, "reason": "同上",
                },
            },
        },
        {
            "id": "oracle:line/the_drives_loop_sets_the_payout_load",
            "inputs": {
                "kind": "closed_loop_steady_state",
                "setpoint_n": PAYOUT_TENSION_N,
                "measurement_transfer": 1.0 / capstan,
                "note": (
                    "传感器在放线端、被控点在落位点；积分作用下稳态读数等于设定值"
                ),
            },
            "expected": {
                "sensor_tension_n": PAYOUT_TENSION_N,
                "controlled_point_tension_n": PAYOUT_TENSION_N * capstan,
                "direction_flip_ratio": capstan * capstan,
            },
            "tolerances": {
                "sensor_tension_n": {
                    "abs": 0.0, "rel": 1.0e-4,
                    "reason": (
                        "**回路自己的稳态残差**，不是建模误差。纯积分环走3000步"
                        "（3 s、dt=1 ms、ζ=1）实测读数29.999641126347463，"
                        "相对1.2e-5；取约8倍余量。**这一条红了说明回路没收敛**，"
                        "而不是求解器不对"
                    ),
                },
                "controlled_point_tension_n": {
                    "abs": 0.0, "rel": 6.0e-3,
                    "reason": (
                        "**两条独立的腿要在这里对上**：驱动链侧是"
                        "``设定值/measurement_transfer``（一个换算比），"
                        "求解器侧是位移控制连续化到全滑移解出来的接触与摩擦。"
                        "容差取与`lay_point_tension_n`同一档——瓶颈是同一个"
                        "（收线步数），不是驱动链"
                    ),
                },
                "direction_flip_ratio": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``exp(μθ)``的平方，纯闭式。**方向搞反的误差是平方**——"
                        "μ=0.3、90°时2.566，与能力位S6.4记的``r² = 2.566``同一个数"
                    ),
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/winding_line_endtoend",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/winding_line_endtoend/generate_oracle.py",
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
