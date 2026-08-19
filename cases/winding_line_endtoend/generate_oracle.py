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

### 七、多轮路由：**逐只轮的比连乘等于总比**（2026-08-18，决策0088丁3）

S6.5的``missing``原话："多轮路由（R4→R1）与活动小导轮未接"。第一版只有**一只**轮。

一条带材连续过``K``只轮，第``k``只有自己的包角``θ_k``与摩擦``μ_k``。
两只轮之间是一段**张力恒定**的直段（没有摩擦源），于是

    T_出 / T_入 = ∏_k (T_出k / T_入k) = ∏_k exp(μ_k·θ_k) = exp(Σ_k μ_k·θ_k)

**右边那个等号是指数函数的恒等式，左边那个是"自由段张力恒定"这件事。**
判据要判的是**左边**：逐只轮各自的比先各自对上自己的闭式，
再连乘等于整条链的总比。连乘那一步在引擎侧是**望远镜式**的
（第``k``只轮的出口张力就是第``k+1``只轮的入口张力，读的是同一条边旁边那条边），
所以它判的其实是"**没有哪只轮的接触力漏进了自由段**"。

### 八、逐只轮判的是**离散**绞盘式，不是``exp(μθ)``

与`cases/capstan_tension_ratio`同一条纪律：本仓的杆是离散的，
逐节点的精确关系是

    T⁺/T⁻ = (1 + μ·tan(Δφ/2)) / (1 − μ·tan(Δφ/2))

``n``段连乘就是这只轮的闭式。连续式``exp(μθ)``是它的极限，
在``Δφ = π/16``、``μ = 0.15``时两者差**2.9e-4**相对——
**大于本案例要分辨的量**，所以逐只轮那条门判离散式。

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
ALGORITHM_VERSION = "1.2.0"

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
#: 多轮路由（丁3）：``((段数, 摩擦), ...)``，一项一只轮，次序即R4→R3→R2→R1。
#: 段角恒为``WRAP_RAD/WRAP_SEGMENTS = 11.25°``。
#: **四个摩擦系数都比0.30小**，理由不是物理而是数值：整条链的总张力比越大，
#: 把最远那只轮也拖进全滑移所需的收线位移就越长，而位移一长几何就漂了
#: （见决策0088第四节那张扫描表）。μ本来就是假设输入（0062第二节裁决2）。
ROUTE_FOUR = ((3, 0.12), (4, 0.10), (3, 0.15), (6, 0.08))
#: 中间一只**零摩擦**的三轮路由：它是"逐只轮各用自己的μ"那条门的必须红。
ROUTE_WITH_A_FREE_ROLLER = ((3, 0.15), (3, 0.0), (3, 0.15))

#: 放线端张力：真机张力区间10—30 N的上端。**它同时是张力回路的设定值**——
#: 丁1之后这个数不再直接当力边界条件用，而是回路的``setpoint_n``。
PAYOUT_TENSION_N = 30.0


def node_ratio(friction: float) -> float:
    """逐节点的**精确离散**绞盘比``(1 + μ·tan(Δφ/2)) / (1 − μ·tan(Δφ/2))``。

    与`cases/capstan_tension_ratio`那一份**同式不同源**：两份生成器各写一遍，
    互不import。连续式``exp(μθ)``是它的极限，不是另一个独立判据。
    """

    half = math.tan(WRAP_RAD / WRAP_SEGMENTS / 2.0)
    return (1.0 + friction * half) / (1.0 - friction * half)


def route_ratios(route):
    """逐只轮的离散闭式比。"""

    return tuple(node_ratio(friction) ** segments for segments, friction in route)


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
        {
            "id": "oracle:line/multi_roller_route",
            "inputs": {
                "kind": "discrete_capstan_per_roller",
                "route": [list(entry) for entry in ROUTE_FOUR],
                "segment_angle_rad": WRAP_RAD / WRAP_SEGMENTS,
            },
            "expected": {
                "per_roller_ratio": list(route_ratios(ROUTE_FOUR)),
                "total_ratio": math.prod(route_ratios(ROUTE_FOUR)),
                "continuum_total": math.exp(
                    sum(
                        friction * segments * WRAP_RAD / WRAP_SEGMENTS
                        for segments, friction in ROUTE_FOUR
                    )
                ),
                "free_roller_ratio": 1.0,
            },
            "tolerances": {
                "per_roller_ratio": {
                    "abs": 0.0, "rel": 1.0e-2,
                    "reason": (
                        "**这一档松，而松的理由不是「差不多就行」**：同一条装配上"
                        "单只轮的偏差随段数单调下降（1段1.84e-2、2段1.53e-2、"
                        "3段1.23e-2、4段9.50e-3、6段4.34e-3、8段2.54e-4，"
                        "μ=0.30、收线0.03 mm实测），**那是端效应**——"
                        "弧的两端各有一个非接触的``ψ=0``节点，转角在那里没有摩擦承接。"
                        "本路由每只轮只有3—6段，四只实测6.59e-3／2.45e-3／"
                        "5.43e-3／2.05e-3，取1e-2。"
                        "**今天单只轮那条2.5e-4不是模型精度，是8段恰好落在交叉点上**"
                    ),
                },
                "total_ratio": {
                    "abs": 0.0, "rel": 2.5e-2,
                    "reason": "四只轮的端效应同号累加，实测1.64e-2",
                },
                "continuum_total": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``∏exp(μᵢθᵢ) = exp(Σμᵢθᵢ)``是指数函数的恒等式，双精度求值。"
                        "**它与上面那条离散总比不是同一个数**——两者差``O(Δφ²)``，"
                        "本路由上是2.4e-3相对。写两条正是为了不让人把它们混成一条"
                    ),
                },
                "free_roller_ratio": {
                    "abs": 4.0e-3, "rel": 0.0,
                    "reason": (
                        "μ=0的那只轮：**切向力逐个恰为0.0（零容差，另判）**，"
                        "而张力比只是落回1附近——实测0.996633，差3.37e-3，"
                        "与上面那条端效应同量级同来源。**绝对容差**，"
                        "因为这里要判的是「落回1」而不是「精确等于1」"
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
