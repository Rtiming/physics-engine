#!/usr/bin/env python3
"""真实中心线的几何不变量——**闭式，独立于被验内核**。

## 零、这一页存在的理由

`cases/groove_sweep_wall`第四节第2条写着"本页没有读过一行真实`centerline.csv`"。
本仓**36个案例没有一个读过真实工件**——全是解析构造的螺旋线、平面圆与直链。
本案例是第一条，兑现[plans/15](../../docs/plans/15_从这里到真机介入的分阶段计划_20260817.md)
阶段二第2.1条与第2.3条。

**所以"读进来的数对不对"比"力算得多准"更要紧**：本页一个力都不算，
只回答"这条工件曲线弯了多少、扭了多少，以及那个数依赖谁"。

## 一、常驻判据用**合成**中心线，真实语料**选择进入**

决策0073裁决真实资产永不进仓。于是本页分两层：

* **仓内常驻**：两条合成站点表（下面第二、第三节），闭式全在本文件里；
* **选择进入**：设`PE_REAL_CENTERLINE_CSV`指向一份GCW的22列导出才跑真实那一档，
  不指则明示skip（形制抄`tests/test_provenance.py`的`PE_REPLAY_CASE_RUNS`）。

## 二、金标一：圆弧＋绕切向自转的帧——**不变量被逐位取回**

    C(a) = (R cos(a/R), R sin(a/R), 0)
    t(a) = (−sin(a/R), cos(a/R), 0)
    n(a) = cos(τ·a)·径向 + sin(τ·a)·ẑ
    s(a) = n × t

取``R = 145.6 mm``、``τ = 2.550 °/mm``（plans/14第2.2节`v2-01-bracket`实测的
两个不变量）。**两者是不同向的独立量**（同节末段），所以不用螺旋线——
螺旋线会把``κ``与``τ``锁死成一个比值。

这条曲线上的闭式（推导在第五节）：

    κ_total = 1/R                       （平面圆，恒定）
    κ_s     = (1/R)·(−cos(τ·a))         ← 随``a``转，因为帧在转
    κ_n     = (1/R)·(−sin(τ·a))·0 …     见第五节，实测项
    τ_frame = −τ                        （符号：见第五节第3条，**这一条栽过**）

**站点表上的差分把这三个数取回来时，误差落在1e-7以下且随``h``二阶下降**
——这一档判的是"提取器本身是对的"。

## 三、金标二：**一段之内的扭转尖峰**——两种差分取法差**恰好2倍**

这一档是本页最要紧的一条，它解释了第四节那个真实语料上的分歧。

直链站点表，帧只在**一个采样步**里绕切向转过``Δ``、其余步一动不动：

    forward 在尖峰站点上给 sin(Δ)/h            ← 一站吃满
    central 在尖峰站点上给 sin(Δ)/(2h)         ← **恰好一半**
    central 在下一个站点上也给 sin(Δ)/(2h)     ← 另一半摊过去了

于是

    τ_max(central) / τ_max(forward) = 1/2      **恰好，零容差**
    ∫|τ|ds 两种取法**相同**（sin Δ）           **恰好，零容差**

**这不是数值巧合，是两个差分模板的代数事实**（推导见第五节第4条）。
它说明：**峰值统计依赖差分取法，积分统计不依赖**。

## 四、真实语料上的实测（选择进入那一档的期望值）

本文件把plans/14第2.2节那张表**逐档抄进`inputs`**，由conformance在真CSV上重算。
2026-08-18实测（`forward`档，`w = 4.0 mm`，开曲线口径）：

| 几何 | 弧长mm | R_min | τ_max °/mm | ε p100 | >0.6% | >0.4% |
|---|---:|---:|---:|---:|---:|---:|
| `v2-coil-01` | 1212.9834 | 75.239 | 2.5496 | 1.3736% | 5.941% | 12.541% |
| `v2-03-provisional` | 850.6617 | 89.519 | 2.1837 | 0.8070% | 2.118% | 4.471% |
| `v2-04-provisional` | 887.7549 | 92.744 | 3.1458 | 0.8150% | 1.351% | 6.982% |
| `v1-coil-1` | 1966.3250 | 72.816 | **6.5686** | 0.5314% | 0.000% | 3.764% |
| `v2-coil-02` | 851.9355 | 97.859 | **6.6467** | 0.5923% | 0.000% | 2.817% |

**两条与plans/14对不上的，如实登记在案例页第四节**：
一条是``τ_max``那两格疑似**在原表里对调了**，一条是那批导出**只有5份不同**
而plans/14报的是9个几何（另外4个不在GCW的`handoff_runs`里）。

## 五、闭式推导

### 1. 圆弧的``dT/ds``

``t(a) = (−sin θ, cos θ, 0)``、``θ = a/R``，故``dt/da = (−cos θ, −sin θ, 0)/R
= −径向/R``，模长``1/R``。**这就是``κ_total``**。

### 2. 投到带宽方向

``n = cos(τa)·径向 + sin(τa)·ẑ``、``s = n × t``。径向``⊥ t``、``ẑ ⊥ t``，
且``径向 × t = ẑ``、``ẑ × t = −径向``，故

    s = cos(τa)·(径向 × t) + sin(τa)·(ẑ × t) = cos(τa)·ẑ − sin(τa)·径向

于是

    κ_s = (dt/da)·s = (−径向/R)·(cos(τa)·ẑ − sin(τa)·径向) = sin(τa)/R
    κ_n = (dt/da)·n = (−径向/R)·(cos(τa)·径向 + sin(τa)·ẑ) = −cos(τa)/R

**两者的平方和恒为``1/R²``**——这条恒等式本身是一道门（第六节）。

### 3. 帧扭率的符号——**这一条栽过，所以单独写一段**

    ds/da = −τ·sin(τa)·ẑ − τ·cos(τa)·径向 + （径向与ẑ随θ转出来的项）
    τ_frame = (ds/da)·n

直接代入得``τ_frame = −τ``。**决策0075第四节那张表里``τ = −0.044505896``
就是这个负号**，而plans/14报的``τ_max = 2.550 °/mm``是**绝对值**。
两者不是矛盾：一个带符号一个不带。**本页判绝对值，并单独判一次符号**，
因为符号错了`contact.PenaltyGrooveSweepLive`的力会往反方向偏（决策0078）。

### 4. 一段尖峰上两个模板的代数

帧绕``t``转``Δ``时``s → cos Δ·s + sin Δ·n``（因为``t × s = n``）。均匀步长``h``：

    forward:  (s[k+1] − s[k])·n[k] / h = (cos Δ·s + sin Δ·n − s)·n / h = sin Δ / h
    central:  (s[k+1] − s[k−1])·n[k] / (2h) = sin Δ / (2h)

下一个站点上``n[k+1] = cos Δ·n[k] − sin Δ·s[k]``（``t × n = −s``），代入同样得
``sin Δ/(2h)``。**两站各拿一半，和不变**——这就是"积分统计不依赖取法"的由来。

## 六、本文件不做的

* **不算力**。本页一个接触项都不装，`contact`一个字节都不碰；
* **不算"扭转吃掉了多少硬弯"**。plans/14第2.3节：那需要"双轴弯曲＋扭转＋
  槽壁接触"三件一起算，本仓这三件今天凑不齐。本页给的是
  ``ε_edge = (w/2)|κ_s|``这条**上界**，以及帧自己已经扭了多少——
  **两个数并排放着，差额是谁吃掉的本页不裁**；
* **不裁哪种差分取法是对的**。本页判的是"两者差多少、以及为什么"。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/real_centerline_invariants"
ALGORITHM_VERSION = "1.0.0"

#: plans/14第2.2节`v2-01-bracket`实测的两个不变量。**不同向的独立量**。
ARC_RADIUS_MM = 145.6
FRAME_TWIST_DEG_PER_MM = 2.550
#: plans/14第2.1节那张表用的带宽。**它不是那批工件自己的槽宽**
#: （实测8 mm级与10 mm级两档），换成各自的槽宽后超标占比全部变成13.4%—23.3%。
#: 本页沿用4.0是为了与plans/14逐档对得上，**并且把这件事写在案例页第四节**。
STRIP_WIDTH_MM = 4.0

#: 尖峰那一档：步长与一段之内转过的角。
SPIKE_STEP_MM = 2.0
SPIKE_ANGLE_RAD = 0.05
SPIKE_STATIONS = 21
SPIKE_INDEX = 10

#: 收敛阶那一档扫的四个步长。**求值点必须在四个步长上都是站点**——
#: 否则细化时"探针落在哪一站"会跟着变，量到的就不是截断误差而是选站误差
#: （实测取``a = 37``时最粗那一档的比值是300而不是4）。
#: ``36 = 2·18 = 1·36 = 0.5·72 = 0.25·144``，四档全中。
REFINEMENT_STEPS_MM = (2.0, 1.0, 0.5, 0.25)
PROBE_ARC_MM = 36.0


def analytic_invariants(arc_mm: float) -> dict[str, float]:
    """第五节第2、3条的闭式。**手推，不调`laydown`。**"""

    twist = math.radians(FRAME_TWIST_DEG_PER_MM)
    return {
        "curvature_total_per_mm": 1.0 / ARC_RADIUS_MM,
        "curvature_s_per_mm": math.sin(twist * arc_mm) / ARC_RADIUS_MM,
        "curvature_n_per_mm": -math.cos(twist * arc_mm) / ARC_RADIUS_MM,
        "twist_per_mm": -twist,
    }


def main() -> int:
    truth = analytic_invariants(PROBE_ARC_MM)
    oracles = [
        {
            "id": "oracle:centerline/analytic_arc_invariants",
            "inputs": {
                "kind": "circular_arc_with_twisting_frame",
                "radius_mm": ARC_RADIUS_MM,
                "frame_twist_deg_per_mm": FRAME_TWIST_DEG_PER_MM,
                "probe_arc_mm": PROBE_ARC_MM,
                "refinement_steps_mm": list(REFINEMENT_STEPS_MM),
                "strip_width_mm": STRIP_WIDTH_MM,
            },
            "expected": {
                **truth,
                #: ``κ_s² + κ_n² = 1/R²``——**一条恒等式，与上面三个数不独立**。
                #: 判它是因为它对"帧的两根轴被对调"这个错**有分辨力**而单点没有：
                #: 对调之后两个数互换，平方和一点不变、而每一个都错了。
                "curvature_quadrature_per_mm": 1.0 / ARC_RADIUS_MM,
                "twist_is_negative": True,
                "edge_strain_at_probe": 0.5 * STRIP_WIDTH_MM
                * abs(truth["curvature_s_per_mm"]),
                "both_schemes_agree": True,
                #: 两种取法在这条**平滑**曲线上的相对差。实测``τ``差4.16e-6、
                #: ``κ_s``差1.25e-14。判1e-4是**留了两个量级**，
                #: 而真语料上两者差**2e-1**——判据的分辨力有三个量级的余地。
                "scheme_agreement_relative_bound": 1.0e-4,
            },
            "tolerances": {
                "curvature_total_per_mm": {
                    "abs": 0.0, "rel": 1.0e-6,
                    "reason": "站点表上的差分，截断项``O(h²)``。``h = 0.25 mm``、"
                              "``R = 145.6 mm``下``(h/R)² ≈ 3e-6``，留一个量级",
                },
                "curvature_s_per_mm": {
                    "abs": 1.0e-9, "rel": 5.0e-5,
                    "reason": "同上，另加一条**绝对**地板：``κ_s = sin(τa)/R``在"
                              "``τa = kπ``附近过零，过零点上相对容差没有意义"
                              "（`scalar_diffraction_airy`那条'判绝对不判相对：J1有零点'同源）",
                },
                "curvature_n_per_mm": {
                    "abs": 1.0e-9, "rel": 5.0e-5,
                    "reason": "同上，同一条零点理由（``κ_n = −cos(τa)/R``也过零）",
                },
                "twist_per_mm": {
                    "abs": 0.0, "rel": 5.0e-5,
                    "reason": "**初稿把这一条写成1e-6并声称'恰好没有截断项'，那是错的。**"
                              "帧不是绕``t``刚性自转——``n``里的'径向'那一半自己随``θ``转，"
                              "于是``ds/da``带一个``O(h²·τ²)``的截断项。"
                              "实测``h = 0.25``时相对2.06e-5（`central`档），"
                              "比值恒为4.000（二阶）。5e-5是那个数留一档余量",
                },
                "curvature_quadrature_per_mm": {
                    "abs": 0.0, "rel": 1.0e-6,
                    "reason": "恒等式``κ_s² + κ_n² = κ_total²``，与`curvature_total`同档",
                },
                "twist_is_negative": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**布尔，零容差**。符号错了`PenaltyGrooveSweepLive`的"
                              "接触力往反方向偏，而三根轴仍然正交、仍然都是单位向量"
                              "——那是一个**安静的**错（决策0078第五节第3条）",
                },
                "edge_strain_at_probe": {
                    "abs": 1.0e-9, "rel": 5.0e-5,
                    "reason": "``(w/2)|κ_s|``继承``κ_s``的档；一次乘法不放大",
                },
                "scheme_agreement_relative_bound": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**它是判据的参数不是被判的量**，零容差逐字比。"
                              "实测``τ``差4.16e-6、``κ_s``差1.25e-14，判1e-4留两个量级"
                              "——而真语料上两者差2e-1，分辨力还有三个量级的余地",
                },
                "both_schemes_agree": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**布尔，零容差**：这条曲线的``κ``与``τ``沿弧长是"
                              "**平滑**的，两种差分取法必须落在同一个数上。"
                              "**这一条是下一个oracle的对照组**——那里两者差恰好2倍",
                },
            },
        },
        {
            "id": "oracle:centerline/one_segment_twist_spike",
            "inputs": {
                "kind": "straight_chain_with_one_segment_frame_spike",
                "step_mm": SPIKE_STEP_MM,
                "spike_angle_rad": SPIKE_ANGLE_RAD,
                "station_count": SPIKE_STATIONS,
                "spike_index": SPIKE_INDEX,
            },
            "expected": {
                "forward_twist_peak_per_mm": math.sin(SPIKE_ANGLE_RAD) / SPIKE_STEP_MM,
                "central_twist_peak_per_mm": math.sin(SPIKE_ANGLE_RAD)
                / (2.0 * SPIKE_STEP_MM),
                "peak_ratio_central_over_forward": 0.5,
                "forward_twist_integral_rad": math.sin(SPIKE_ANGLE_RAD),
                "central_twist_integral_rad": math.sin(SPIKE_ANGLE_RAD),
                "central_spreads_over_two_stations": True,
            },
            "tolerances": {
                "forward_twist_peak_per_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``sin Δ / h``，一次`sin`一次除法。超越函数不在"
                              "IEEE-754的正确舍入承诺里，故不写零容差；1 ulp≈2.2e-16，留一档",
                },
                "central_twist_peak_per_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "同上，多一次除以2（2的幂，**精确**）",
                },
                "peak_ratio_central_over_forward": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**本页最要紧的一条零容差。** 它不是数值巧合而是两个"
                              "差分模板的代数事实（第五节第4条），所以`sin Δ`在比值里"
                              "**整个约掉**——写非零容差等于承认不知道这是不是恒等式",
                },
                "forward_twist_integral_rad": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``(sin Δ/h)·h``；与峰值同档",
                },
                "central_twist_integral_rad": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "``2·(sin Δ/2h)·h``。**与forward逐位相同这件事本身是判据**"
                              "——峰值差2倍而积分不差，正是真实语料上实测到的形态",
                },
                "central_spreads_over_two_stations": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**布尔，零容差**：只判峰值比是判不出'摊到哪里去了'的。"
                              "没有这一条，一个把central整体乘0.5的实现照样全绿",
                },
            },
        },
        {
            "id": "oracle:centerline/plans14_table_row",
            "inputs": {
                "kind": "published_table_rows",
                "source": "docs/plans/14_现场场景勘察_真实工件是非平面槽_20260817.md 第2.2节",
                "scheme": "forward",
                "strip_width_mm": STRIP_WIDTH_MM,
                "opt_in_env": "PE_REAL_CENTERLINE_CSV",
                #: **按弧长认几何**——它是那张表里最有分辨力的一格。
                #: 每一行两个``τ_max``：``published``是plans/14第2.2节印的那个数，
                #: ``twist_max_deg_per_mm``是本页实测的。**两者只在最后两行不同，
                #: 而那两行恰好互为对方**——见第七节。
                "rows_by_arc_length_mm": {
                    "1212.98": {"name": "v2-coil-01", "r_min_mm": 75.2,
                                "twist_max_deg_per_mm": 2.550,
                                "published_twist_max_deg_per_mm": 2.550,
                                "edge_strain_p100": 0.01374,
                                "above_006": 0.059, "above_004": 0.125},
                    "850.66": {"name": "v2-03-provisional", "r_min_mm": 89.5,
                               "twist_max_deg_per_mm": 2.184,
                               "published_twist_max_deg_per_mm": 2.184,
                               "edge_strain_p100": 0.00808,
                               "above_006": 0.021, "above_004": 0.045},
                    "887.75": {"name": "v2-04-provisional", "r_min_mm": 92.7,
                               "twist_max_deg_per_mm": 3.146,
                               "published_twist_max_deg_per_mm": 3.146,
                               "edge_strain_p100": 0.00816,
                               "above_006": 0.014, "above_004": 0.070},
                    "1966.33": {"name": "v1-coil-1", "r_min_mm": 72.8,
                                "twist_max_deg_per_mm": 6.5686,
                                "published_twist_max_deg_per_mm": 6.648,
                                "edge_strain_p100": 0.00532,
                                "above_006": 0.000, "above_004": 0.038},
                    "851.94": {"name": "v2-coil-02", "r_min_mm": 97.9,
                               "twist_max_deg_per_mm": 6.6467,
                               "published_twist_max_deg_per_mm": 6.568,
                               "edge_strain_p100": 0.00592,
                               "above_006": 0.000, "above_004": 0.028},
                },
            },
            "expected": {
                #: 逐档相对偏差的上界。**``τ_max``那一格单独放宽到0.02**，
                #: 理由见案例页第四节第2条：`v1-coil-1`与`v2-coil-02`两行
                #: 疑似在原表里**对调**了，本页按对调后判并把这件事登记出来。
                "r_min_relative_tolerance": 0.005,
                "twist_max_relative_tolerance": 0.005,
                "edge_strain_p100_relative_tolerance": 0.005,
                "arc_fraction_absolute_tolerance": 0.001,
                "geometries_matched": 5,
                "published_rows_without_a_file": 4,
                #: **本页最要紧的一个整数。** 有几行的实测``τ_max``对不上原表。
                "rows_disagreeing_with_the_published_twist": 2,
                #: 而那两行**互换之后就对上了**——这是"疑似对调"这句话的
                #: 全部证据，也是它与"我们算错了"的分界。
                "the_two_disagreeing_rows_swap_into_each_other": True,
            },
            "tolerances": {
                "r_min_relative_tolerance": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**它是判据的参数不是被判的量**，零容差逐字比。"
                              "0.5%是原表的印刷位数（75.2、89.5……三位有效数字）",
                },
                "twist_max_relative_tolerance": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "同上。原表印四位有效数字（2.550、6.648），0.5%宽出两档",
                },
                "edge_strain_p100_relative_tolerance": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "同上。原表印三位小数的百分数（1.374%）",
                },
                "arc_fraction_absolute_tolerance": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "占比判**绝对**不判相对：`>0.6%`那几格原表就是`0.0%`，"
                              "相对容差在零上没有意义",
                },
                "geometries_matched": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**整数，零容差。** GCW的`handoff_runs`里按内在几何归并"
                              "只有5份不同（三个run名各有两个时间戳、逐字节相同）。"
                              "**判这个数是为了让'语料变了'当场红**",
                },
                "rows_disagreeing_with_the_published_twist": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**整数，零容差。** 写成判据而不是散文：'大体对上'不是"
                              "一个可以被后来的人验的说法，'恰好两行对不上'是",
                },
                "the_two_disagreeing_rows_swap_into_each_other": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**布尔，零容差。** 这一条把'原表对调了'与'我们算错了'"
                              "分开：算错不会让两个错数**恰好互为对方**。"
                              "它红了要么是语料变了、要么是原表被订正了，两者都该有人来看",
                },
                "published_rows_without_a_file": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "**整数，零容差。** plans/14报9个几何，GCW侧只到得了5个"
                              "——`test`、`clean_a`、`v1-coil-3`、`v1-coil-2`四行"
                              "**本页够不到**。把够不到的条数写成判据，"
                              "比在散文里说一句'部分覆盖'诚实",
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/real_centerline_invariants",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/real_centerline_invariants/generate_oracle.py",
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
