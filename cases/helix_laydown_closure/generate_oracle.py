#!/usr/bin/env python3
"""落位点几何与闭合残差——**闭式，独立于被验内核**。

## 一、为什么金标可以全闭式

真实工件的槽中心线在GCW那边（plans/14第二节），本仓没有。所以本案例
**自造金标**：拿一条解析曲线当中心线，拿一条解析位姿当机器人。
两者一配，落位点的每一个量都有闭式。

### 曲线：圆柱螺旋线

    C(θ) = (R cos θ, R sin θ, p θ),   s = a θ,   a = sqrt(R² + p²)

    κ = R/a²（常数）、τ = p/a²（常数）

取``R = 60 mm``、``p = 25 mm/rad``，于是``a = 65 mm``**恰好是整数**
（5倍的勾股数(12, 5, 13)），闭式因此干净。

帧按GCW约定（``s = n × t``、``(t, s, n)``右手系，plans/14第二节）：

    t(θ) = (1/a)(−R sin θ,  R cos θ,  p)        切向
    n(θ) = (cos θ,  sin θ,  0)                  槽面外法线（径向朝外）
    s(θ) = n × t = (1/a)(p sin θ, −p cos θ, R)  带宽方向（＝Frenet副法线）

### 位姿：姿态随便选，**平移反解**

这是本页最关键的一步。闭合条件``pose(t) · C(σ(t)) = 入带点``是三个标量约束
配一个未知量``σ``，**一般无解**。但机器人有六个自由度：姿态三个可以随便选，
平移三个正好把落位点钉在固定入带点上。于是

    Rot(t) = Rot_z(−ω t)                    （姿态：绕世界z轴匀速转）
    σ(t)   = σ₀ + u t                        （送带账：匀速）
    p(t)   = 入带点 − Rot(t) · C(σ(t))       （平移：**反解**出来）

闭合因此是**构造出来的**而不是碰巧成立的，残差的解析值是**恒等于零**。
把送带账或入带点故意挪开之后，残差同样有闭式（第三、四节）。

### 落位点处的世界系三标架

    triad_world(t) = Rot_z(−ω t) · triad(θ(t)),   θ(t) = σ(t)/a

而``Rot_z(−ωt)``作用在``t(θ)``、``n(θ)``、``s(θ)``上恰好等于把参数换成
``χ(t) = θ(t) − ω t``。所以**世界系三标架只随``χ``转**，转速

    dχ/dt = u/a − ω

取``u = 2aω``让它等于``ω``——**这一步是有意的**：若取``u = aω``（材料前进恰好
抵消转动）三标架会**恒定**，那是螺旋线自身的螺旋对称性，
入射角变成常数，**这一档就退化成测不出东西的算例**。

## 二、入射角的闭式

自由段方向取``d = t_world(0)``（即``t = 0``处无折角落位）。记``φ = (dχ/dt)·t``：

    沿槽    d · t_world = (R² cos φ + p²)/a²
    带宽向  d · s_world = (R p/a²)(1 − cos φ)
    法向    d · n_world = (R/a) sin φ

    入射角      θ_inc = atan2(hypot(带宽向, 法向), 沿槽)
    槽面内方位  α     = atan2(带宽向, 沿槽)
    离面仰角    β     = asin(法向)

``φ = 0``给``θ_inc = 0``——**理想落位处入射角为零**，这是符号约定的锚点
（`laydown.py`模块文档"入射角的符号约定"一段）。

## 三、闭合残差之一：送带账偏了``Δ``

把送带账原点挪``Δ``，落位点变成``C(θ + Δ/a)``而入带点仍在``C(θ)``的像上。
记``δ = Δ/a``，三个分量**都与``θ`` 无关**（螺旋线的齐次性），逐条推出来：

    |残差|   = sqrt(4R² sin²(δ/2) + p² δ²)              （弦长）
    沿槽     = (R² sin δ + p² δ)/a
    法向     = R(1 − cos δ)
    带宽向   = (R p/a)(δ − sin δ)
    弧长坐标差 = Δ                                       （**恰好**）

**这一档的意义**：残差几乎全部落在沿槽分量上，横向只有``R(1−cos δ)``那一点点
曲率效应。**多放``Δ``毫米带材就对上了。**

## 四、闭合残差之二：位姿把线圈举偏了``δ_n``

把入带点沿``t = 0``处的世界系槽面法向挪``δ_n``。此时

    弧长坐标差 = 0        （**送带账一点问题都没有**）
    沿槽       = 0
    法向       = −δ_n
    横向       = δ_n
    位姿不可约偏距 = δ_n  （入带点到曲线本身的距离）

前提是``δ_n``远小于曲率半径``a²/R = 70.4 mm``，否则最近点会跳到别的圈上。
本案例取``δ_n = 1.5 mm``，余量47倍。

**第三节与第四节的残差模长可以完全一样，而病因与处置完全相反。**
一个只报"残差2.3 mm"的实现分不清它们——**那正是本案例要判的事**。

## 五、平面圆退化档（``p = 0``）

同一套构造，取``p = 0``、``u = Rω``（材料前进恰好抵消转动）：

    τ = 0、世界系三标架恒定、入射角恒为0、所需送带率恒为 Rω

**这一档任何实现都该算对，算不对说明连基本盘都没接上。** 它同时是闭合拓扑
（``topology='closed'``、``out_of_range='wrap'``）唯一的金标——
真实工件的槽是闭合曲线，而闭合曲线的弧长坐标要模总长回绕。

## 六、生成器不做什么

不import任何力学模块，不调`laydown`、不调`solve_equilibrium`、不调任何能量项。
本文件只用``math``与`physics_engine.oracles`（验收基座）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/helix_laydown_closure"
ALGORITHM_VERSION = "1.0.0"

#: 螺旋线：``a = sqrt(60² + 25²) = 65``恰好是整数。**假设输入**——
#: 真实中心线在GCW那边，本仓没有（plans/14第四节"没验证哪一条是现场在用的那条"）。
RADIUS_MM = 60.0
PITCH_MM_PER_RAD = 25.0
HELIX_SCALE_MM = 65.0

#: 位姿与送带账。**全部是设定，不是实测。**
SPIN_RAD_S = 0.4
FEED_RATE_MM_S = 2.0 * HELIX_SCALE_MM * SPIN_RAD_S  # 52.0
START_THETA_RAD = 0.6
HORIZON_S = 4.0

#: 世界系三标架的转速``u/a − ω``。取``u = 2aω``让它等于``ω``——见模块文档第一节。
TRIAD_SPIN_RAD_S = FEED_RATE_MM_S / HELIX_SCALE_MM - SPIN_RAD_S

#: 自由跨段：入带点抄WII那个"Frame-A原点``x``恒为1126.0"的数（plans/14第3.3节），
#: 跨长200 mm与`cases/roller_skew_lateral_drift`同源。**两者都是设定。**
ENTRY_POINT_MM = (1126.0, 0.0, 300.0)
SPAN_LENGTH_MM = 200.0

#: 两种病因各一个数。
FEED_DRIFT_MM = 0.7
POSE_LIFT_MM = 1.5

#: 平面圆退化档。
CIRCLE_RADIUS_MM = 60.0
CIRCLE_FEED_RATE_MM_S = CIRCLE_RADIUS_MM * SPIN_RAD_S  # 24.0

#: 采样时刻。避开``t = 0``与``t = horizon``，中心差分在端点上取不到样点。
SAMPLE_TIMES_S = (0.5, 1.5, 2.5, 3.5)


def incidence_components(phi: float) -> tuple[float, float, float]:
    """``(沿槽, 带宽向, 法向)``——自由段方向在落位点三标架上的三个投影。"""

    scale_squared = HELIX_SCALE_MM * HELIX_SCALE_MM
    along = (RADIUS_MM * RADIUS_MM * math.cos(phi) + PITCH_MM_PER_RAD**2) / scale_squared
    across_width = (
        RADIUS_MM * PITCH_MM_PER_RAD * (1.0 - math.cos(phi)) / scale_squared
    )
    across_normal = RADIUS_MM * math.sin(phi) / HELIX_SCALE_MM
    return (along, across_width, across_normal)


def incidence_angles(phi: float) -> tuple[float, float, float]:
    """``(入射角, 槽面内方位角, 离面仰角)``，弧度。"""

    along, across_width, across_normal = incidence_components(phi)
    return (
        math.atan2(math.hypot(across_width, across_normal), along),
        math.atan2(across_width, along),
        math.asin(max(-1.0, min(1.0, across_normal))),
    )


def drifted_closure(drift_mm: float) -> dict[str, float]:
    """送带账偏``drift_mm``之后的闭合残差。**三个分量都与``θ``无关。**"""

    delta = drift_mm / HELIX_SCALE_MM
    return {
        "magnitude_mm": math.sqrt(
            4.0 * RADIUS_MM**2 * math.sin(0.5 * delta) ** 2
            + PITCH_MM_PER_RAD**2 * delta * delta
        ),
        "along_tangent_mm": (
            RADIUS_MM**2 * math.sin(delta) + PITCH_MM_PER_RAD**2 * delta
        ) / HELIX_SCALE_MM,
        "across_normal_mm": RADIUS_MM * (1.0 - math.cos(delta)),
        "across_width_mm": RADIUS_MM * PITCH_MM_PER_RAD * (delta - math.sin(delta))
        / HELIX_SCALE_MM,
        "arc_gap_mm": drift_mm,
    }


def main() -> int:
    drifted = drifted_closure(FEED_DRIFT_MM)
    drifted["transverse_mm"] = math.hypot(
        drifted["across_width_mm"], drifted["across_normal_mm"]
    )
    incidence: dict[str, float] = {}
    for time_s in SAMPLE_TIMES_S:
        phi = TRIAD_SPIN_RAD_S * time_s
        angle, in_plane, out_of_plane = incidence_angles(phi)
        key = f"{time_s:g}".replace(".", "p")
        incidence[f"incidence_rad_at_t{key}"] = angle
        incidence[f"in_plane_rad_at_t{key}"] = in_plane
        incidence[f"out_of_plane_rad_at_t{key}"] = out_of_plane

    oracles = [
        {
            "id": "oracle:laydown/helix_kinematics",
            "inputs": {
                "kind": "circular_helix",
                "radius_mm": RADIUS_MM,
                "pitch_mm_per_rad": PITCH_MM_PER_RAD,
                "spin_rad_s": SPIN_RAD_S,
                "feed_rate_mm_s": FEED_RATE_MM_S,
            },
            "expected": {
                "helix_scale_mm": HELIX_SCALE_MM,
                "curvature_per_mm": RADIUS_MM / (HELIX_SCALE_MM**2),
                "torsion_per_mm": PITCH_MM_PER_RAD / (HELIX_SCALE_MM**2),
                "arc_per_turn_mm": 2.0 * math.pi * HELIX_SCALE_MM,
                "required_feed_rate_mm_s": FEED_RATE_MM_S,
                "triad_spin_rad_s": TRIAD_SPIN_RAD_S,
            },
            "tolerances": {
                "helix_scale_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``sqrt(60² + 25²) = 65``是精确整数，闭式自洽",
                },
                "curvature_per_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``κ = R/a²``，闭式；螺旋线上κ与τ都是常数，与弧长无关",
                },
                "torsion_per_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``τ = p/a²``，闭式。折成0.339 °/mm，落在真实工件"
                        "0.454—6.648 °/mm（plans/14第2.2节）的下沿——**量级对得上，"
                        "但这是设定不是实测**"
                    ),
                },
                "arc_per_turn_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``2πa = 408.4 mm``。真实工件每匝850—1966 mm，本档偏短，"
                        "**因为它验的是几何层不是工件本身**"
                    ),
                },
                "required_feed_rate_mm_s": {
                    "abs": 1.0e-6, "rel": 0.0,
                    "reason": (
                        "``dσ_pose/dt``由**位姿**独立算出，闭式是``u = 2aω``。"
                        "引擎侧走离散中心线上的最近点搜索再差分，"
                        "实测（Hermite＋2步细化、512站点/匝、中心差分h=1e-3 s）"
                        "最大偏差**2.03e-09 mm/s**；1e-6是其五百倍余量。"
                        "**收敛阶另有一条门判**"
                    ),
                },
                "triad_spin_rad_s": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "世界系三标架的转速``u/a − ω``。**这个数不为零是刻意的**："
                        "取``u = aω``会让三标架恒定（螺旋对称），入射角退化成常数"
                    ),
                },
            },
        },
        {
            "id": "oracle:laydown/incidence_over_time",
            "inputs": {
                "kind": "incidence_against_a_fixed_free_span",
                "free_span_direction": "落位点在t=0处的世界系切向（无折角）",
                "free_span_length_mm": SPAN_LENGTH_MM,
                "entry_point_mm": list(ENTRY_POINT_MM),
                "sample_times_s": list(SAMPLE_TIMES_S),
            },
            "expected": incidence,
            "tolerances": {
                key: {
                    "abs": 1.0e-5, "rel": 3.0e-4,
                    "reason": (
                        "闭式``cos θ_inc = (R² cos φ + p²)/a²``（φ = (u/a − ω)t）。"
                        "引擎侧的偏差**全部来自中心线帧的插值**，位置那一路早就到"
                        "1e-9量级了。512站点/匝实测最大绝对偏差逐量不同："
                        "总角6.7e-07、离面2.1e-08、**槽面内方位6.7e-06** rad。"
                        "方位角那一路最差，因为它是两个大量之差"
                        "（``(Rp/a²)(1 − cos φ)``在小φ处只有7e-03 rad），"
                        "**相对判据在它身上没有意义**——所以abs给到1e-5、"
                        "rel 3e-4只对大角那几档起作用。**先量后写。**"
                    ),
                }
                for key in incidence
            },
        },
        {
            "id": "oracle:laydown/closure_from_a_drifted_feed_account",
            "inputs": {
                "kind": "feed_account_origin_shifted",
                "drift_mm": FEED_DRIFT_MM,
                "note": "位姿没动，只把送带账的原点挪了；病因是账不是臂",
            },
            "expected": drifted,
            "tolerances": {
                "magnitude_mm": {
                    "abs": 0.0, "rel": 1.0e-6,
                    "reason": (
                        "闭式弦长``sqrt(4R² sin²(δ/2) + p²δ²)``；"
                        "512站点/匝实测rel **2.7e-11**（位置那一路是四阶）"
                    ),
                },
                "along_tangent_mm": {
                    "abs": 0.0, "rel": 1.0e-6,
                    "reason": (
                        "``(R² sin δ + p² δ)/a``，**与θ无关**（螺旋线齐次）。"
                        "实测rel 1.1e-10：投影方向几乎与残差平行，帧的角误差只以"
                        "二阶进来。**这一份是送带账修得掉的**"
                    ),
                },
                "across_normal_mm": {
                    "abs": 0.0, "rel": 1.0e-4,
                    "reason": (
                        "``R(1 − cos δ)``，纯曲率效应，量级只有沿槽那份的1/200。"
                        "实测rel 4.7e-06——这一路吃的是帧的角误差"
                    ),
                },
                "across_width_mm": {
                    "abs": 0.0, "rel": 1.0e-1,
                    "reason": (
                        "``(Rp/a)(δ − sin δ)``，**三阶小量**，本档只有4.80e-06 mm。"
                        "**512站点/匝根本判不了它**——实测相对偏差96%，"
                        "因为残差(0.7 mm)乘上帧的角误差(约2e-06 rad)就已经和它同量级。"
                        "所以本条**只在2048站点/匝那一档判**，实测相对偏差6.0e-02，"
                        "rel 1e-1是其1.7倍余量。收敛实测：256/512/1024/2048/4096档"
                        "相对偏差3.64/0.96/0.24/0.060/0.015——**二阶**。"
                        "已知失效清单第2条登记了这一条"
                    ),
                },
                "transverse_mm": {
                    "abs": 0.0, "rel": 1.0e-4,
                    "reason": (
                        "横向合成，由``across_normal``主导。**这一份送带账修不掉**，"
                        "但它只有0.0035 mm。实测rel 5.4e-06"
                    ),
                },
                "arc_gap_mm": {
                    "abs": 1.0e-8, "rel": 0.0,
                    "reason": (
                        "``σ_feed − σ_pose``**恰等于**挪动量``Δ``——"
                        "512站点/匝实测4.4e-12，abs 1e-8是其两千倍余量"
                    ),
                },
            },
        },
        {
            "id": "oracle:laydown/closure_from_a_lifted_pose",
            "inputs": {
                "kind": "entry_point_offset_along_the_groove_surface_normal",
                "lift_mm": POSE_LIFT_MM,
                "curvature_radius_mm": HELIX_SCALE_MM**2 / RADIUS_MM,
                "note": "送带账没动，只把入带点沿槽面法向挪了；病因是臂不是账",
            },
            "expected": {
                "magnitude_mm": POSE_LIFT_MM,
                "transverse_mm": POSE_LIFT_MM,
                "across_normal_mm": -POSE_LIFT_MM,
                "pose_only_offset_mm": POSE_LIFT_MM,
                "along_tangent_mm": 0.0,
                "arc_gap_mm": 0.0,
                "incidence_rad": 0.0,
            },
            "tolerances": {
                "magnitude_mm": {
                    "abs": 1.0e-7, "rel": 0.0,
                    "reason": (
                        "沿法向挪``δ_n``，残差模长恰为``δ_n``；"
                        "512站点/匝实测5.2e-10，abs 1e-7是其两百倍余量"
                    ),
                },
                "transverse_mm": {
                    "abs": 1.0e-7, "rel": 0.0,
                    "reason": "**全部落在横向**——与上一条oracle形成对照的那一半。实测5.2e-10",
                },
                "across_normal_mm": {
                    "abs": 1.0e-7, "rel": 0.0,
                    "reason": (
                        "带符号：落位点在入带点的**里侧**（残差指向曲线）。"
                        "符号丢了就分不清'举高了'与'压低了'"
                    ),
                },
                "pose_only_offset_mm": {
                    "abs": 1.0e-7, "rel": 0.0,
                    "reason": (
                        "入带点到曲线本身的距离。``δ_n = 1.5``远小于曲率半径70.4 mm"
                        "（余量47倍），所以最近点仍是原来那一点，**不会跳到别的圈上**"
                    ),
                },
                "along_tangent_mm": {
                    "abs": 1.0e-6, "rel": 0.0,
                    "reason": "**恰为零**：法向偏移在切向上没有分量。实测3.2e-08",
                },
                "arc_gap_mm": {
                    "abs": 1.0e-6, "rel": 0.0,
                    "reason": (
                        "**恰为零，这是本案例最要紧的一个数**："
                        "位姿把线圈举偏1.5 mm，而送带账一点问题都没有。"
                        "报一个非零的弧长差就是把账算到了无辜的一方。"
                        "实测1.6e-08，abs 1e-6是其六十倍余量"
                    ),
                },
                "incidence_rad": {
                    "abs": 1.0e-5, "rel": 0.0,
                    "reason": (
                        "自由段仍沿落位点切向，举偏不改变方向，故入射角仍为零。"
                        "实测2.6e-06——它吃的是帧插值那一路，不是举偏本身"
                    ),
                },
            },
        },
        {
            "id": "oracle:laydown/plane_circle_degenerate",
            "inputs": {
                "kind": "plane_circle_closed_loop",
                "radius_mm": CIRCLE_RADIUS_MM,
                "pitch_mm_per_rad": 0.0,
                "spin_rad_s": SPIN_RAD_S,
                "feed_rate_mm_s": CIRCLE_FEED_RATE_MM_S,
                "topology": "closed",
            },
            "expected": {
                "torsion_per_mm": 0.0,
                "curvature_per_mm": 1.0 / CIRCLE_RADIUS_MM,
                "arc_per_turn_mm": 2.0 * math.pi * CIRCLE_RADIUS_MM,
                "required_feed_rate_mm_s": CIRCLE_FEED_RATE_MM_S,
                "incidence_rad": 0.0,
                "triad_spin_rad_s": 0.0,
            },
            "tolerances": {
                "torsion_per_mm": {
                    "abs": 1.0e-15, "rel": 0.0,
                    "reason": (
                        "平面曲线挠率**恒为零**。引擎侧的可判形式是"
                        "带宽方向恒为``(0, 0, 1)``——它不绕切向转"
                    ),
                },
                "curvature_per_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``κ = 1/R``，闭式",
                },
                "arc_per_turn_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``2πR``。它同时是闭合拓扑下弧长坐标回绕的模",
                },
                "required_feed_rate_mm_s": {
                    "abs": 1.0e-5, "rel": 0.0,
                    "reason": (
                        "``Rω``。**这一档任何实现都该算对**；实测偏差1.8e-08 mm/s，"
                        "abs 1e-5留三个数量级余量"
                    ),
                },
                "incidence_rad": {
                    "abs": 1.0e-5, "rel": 0.0,
                    "reason": (
                        "``u = Rω``让材料前进恰好抵消转动 ⟹ 世界系三标架恒定 ⟹ "
                        "入射角**恒为零**。实测最大2.4e-07 rad"
                    ),
                },
                "triad_spin_rad_s": {
                    "abs": 1.0e-15, "rel": 0.0,
                    "reason": (
                        "``u/R − ω = 0``。**这一档与螺旋线档的分水岭**："
                        "那边这个数是0.4，入射角随之变；这边是0，入射角不变"
                    ),
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/helix_laydown_closure",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/helix_laydown_closure/generate_oracle.py",
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
