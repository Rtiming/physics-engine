#!/usr/bin/env python3
"""翻倒阈值的金标——**几何阈值`tanθ ≤ w/h`自己推一遍，不抄**。

plans/16的M4。0081第四节写着"没有裁翻倒那一支的判据……等实现到那一步再裁"，
本脚本与`case.md`第二节就是那一次裁。

## 一、几何阈值：两条独立的推法，同一个数

**推法甲（力矩平衡）**：斜面倾角`θ`、底面半宽`w`（沿坡向）、重心高`h`（垂直底面）。
取下坡侧底棱为支点，重心相对支点的位置在斜面标架里是`(−w, h)`，
重力是`m·g·(sinθ, −cosθ)`。绕支点的力矩（取`t̂ × n̂`方向为正）

    τ = (−w)·(−mg cosθ) − h·(mg sinθ) = mg·(w cosθ − h sinθ)

`τ > 0`是复位、`τ < 0`是倾覆，于是**分界是`w cosθ = h sinθ`，即`tanθ = w/h`**。

**推法乙（重心投影）**：从重心沿重力方向`(sinθ, −cosθ)`走到底面`n = 0`，
走过的参数是`h/cosθ`，落点的坡向坐标是`h·tanθ`。
**它落在底面内的条件是`h·tanθ ≤ w`，同一个不等式。**

`Fraction`精确算`w/h = 1/2`——`w = 5`、`h = 10`。**这个数是有理数，不是测量值。**

## 二、把倾角推广到"已经倾了`φ`"（本案例判"真的翻过去"要用）

把上面的力矩式子里的`θ`换成`θ + φ`（体绕支点转了`φ`就等于坡陡了`φ`）：

    τ(φ) = mg·[w cos(θ+φ) − h sin(θ+φ)] = mg·√(w²+h²)·sin(ψc − θ − φ)

其中`ψc = arctan(w/h)`。于是**平衡点在`φ = ψc − θ`**：`θ > ψc`时它是负的，
意思是"从`φ = 0`就已经过了平衡点"，倾覆力矩单调把`φ`推大——**这才是"翻倒"**。

## 三、罚接触把阈值挪了多少：一条可算的修正

刚体阈值假定底面贴死在平面上。罚接触不是：下坡侧压进去`δ_dn = F_dn/k`、
上坡侧`δ_up = F_up/k`，于是**体自己相对斜面又倾了**

    sin β = (δ_dn − δ_up) / (2w)

而"体相对重力的倾角"才是判据里那个角，于是**临界角变成`θc = ψc − β`**。
在阈值上`F_up = 0`、下坡侧承全部法向载荷`W cosθ`，若下坡侧有`n_d`个支承点：

    β_c = W·cos(ψc) / (2·n_d·w·k)

**这一条不是拟合出来的**：它只用了"罚力等于刚度乘穿透"与几何，
而它给出的预言是**偏差随`1/k`趋零**——本案例在两个刚度上各二分一次去验它。

## 四、静态承载的闭式（稳定侧那一组判的）

`n_b`个底面支承点（沿坡向对半分，各`n_b/2`个），静力矩平衡给

    F_dn = W cosθ/n_b + W·h·sinθ/(n_b·w) ·(1/1)   ← 见下（按半宽`w`取矩）
    F_up = W cosθ/n_b − W·h·sinθ/(n_b·w)

而**压心位置是一条恒等式**：绕重心取矩，法向力给`Σ x_i F_i`、
摩擦力全在底面高度`h`上给`h·W sinθ`，两者相消 ⟹

    x_N ≡ Σ x_i F_i / Σ F_i = h·tanθ

**`x_N = w`就是翻倒**——与第一节推法乙同一句话，只是这一次它是**测出来的**。

## 五、速度型摩擦的稳态蠕滑，以及**它会饱和**

速度型库仑（`contact_dynamics`那一档）在稳态给`Σ min(k_t·v, μF_i) = W sinθ`。
上坡侧承载小，**先饱和**：本案例的参数下`μF_up = 0.03762 N`
小于每点需求`W sinθ/4 = 0.04194 N`，于是

    v = (W sinθ − 2μF_up) / (2 k_t)      ← 两点饱和、两点线性

**朴素的`W sinθ/(4k_t)`会差10.5%**，而这条差异正是"哪几个点饱和"被判到的地方。

零运行时依赖，纯标准库。
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/box_tipping_threshold"
ALGORITHM_VERSION = "1.0.0"

#: 几何与材料。半宽`w`沿坡向、`d`横坡向、`h`是重心到底面的高（长方体的半高）。
HALF_W_MM, HALF_D_MM, HALF_H_MM = 5.0, 5.0, 10.0
MASS_KG = 0.05
GRAVITY_MM_PER_S2 = 9810.0
WEIGHT_N = MASS_KG * GRAVITY_MM_PER_S2 / 1000.0

#: 接触参数。`k_t`取小的理由与`rolling_ball_incline`第四节第1条同源但方向相反：
#: 那里`k_t`大会进颤振区，这里`k_t`大会让**切向阻尼模态的显式稳定上限**
#: 压到步长以下（实测见`case.md`第四节第2条）。
NORMAL_STIFFNESS_N_PER_MM = 50.0
TANGENTIAL_STIFFNESS = 0.03
NORMAL_DAMPING = 0.01
FRICTION = 1.2

STABLE_INCLINE_DEG = 20.0
TOPPLE_INCLINE_DEG = 32.0

#: 二分那一组的两个刚度。取4倍而不是5倍，是为了让『偏差之比 == 刚度之比』
#: 这条判据的两端各自都被夹得够紧（见`case.md`第三节该行的理由）。
BISECTION_STIFFNESSES = (50.0, 200.0)


def exact_tangent_threshold() -> Fraction:
    """`w/h`——**有理数，`Fraction`算，不走浮点**。"""

    return Fraction(int(HALF_W_MM * 2), int(HALF_H_MM * 2))


def static_normal_forces(theta: float, base_points: int) -> tuple[float, float]:
    """底面`base_points`个支承点时上坡侧/下坡侧的单点法向力（第四节）。"""

    mean = WEIGHT_N * math.cos(theta) / base_points
    half = WEIGHT_N * HALF_H_MM * math.sin(theta) / (base_points * HALF_W_MM)
    return mean - half, mean + half


def compliance_tilt(theta: float, stiffness: float, base_points: int) -> float:
    """两侧穿透不等造成的体倾角`β`（第三节）。"""

    up, down = static_normal_forces(theta, base_points)
    return math.asin((down - up) / stiffness / (2.0 * HALF_W_MM))


def main() -> int:
    tangent = exact_tangent_threshold()
    psi_c = math.atan2(HALF_W_MM, HALF_H_MM)

    theta_s = math.radians(STABLE_INCLINE_DEG)
    f_up, f_dn = static_normal_forces(theta_s, 4)
    demand_per_point = WEIGHT_N * math.sin(theta_s) / 4.0
    saturates = FRICTION * f_up < demand_per_point
    creep = (WEIGHT_N * math.sin(theta_s) - 2.0 * FRICTION * f_up) / (
        2.0 * TANGENTIAL_STIFFNESS
    )
    pressure_centre = HALF_H_MM * math.tan(theta_s)

    #: 二分那两个刚度各自的柔度修正与预言。下坡侧只有**一个**支承点
    #: （二分那一组是平面两点），所以`n_d = 1`。
    soft, stiff = BISECTION_STIFFNESSES
    beta_soft = WEIGHT_N * math.cos(psi_c) / (2.0 * 1 * HALF_W_MM * soft)
    beta_stiff = WEIGHT_N * math.cos(psi_c) / (2.0 * 1 * HALF_W_MM * stiff)

    common = {
        "half_extents_mm": [HALF_W_MM, HALF_D_MM, HALF_H_MM],
        "mass_kg": MASS_KG,
        "gravity_mm_per_s2": GRAVITY_MM_PER_S2,
        "friction_coefficient": FRICTION,
        "tangential_stiffness": TANGENTIAL_STIFFNESS,
        "normal_damping": NORMAL_DAMPING,
    }

    oracles = [
        {
            "id": "oracle:box_tipping/stable_side",
            "inputs": {
                **common,
                "kind": "box_on_incline_below_threshold",
                "incline_deg": STABLE_INCLINE_DEG,
                "normal_stiffness_n_per_mm": NORMAL_STIFFNESS_N_PER_MM,
                "dt_s": 2.0e-4,
                "steps": 3000,
                #: 位姿漂移的**上界**（不是等式，所以住在inputs里）。
                "tilt_swing_bound_rad": 1.0e-6,
                "out_of_plane_bound_rad_per_s": 1.0e-10,
                "penetration_bound_mm": 0.05,
                #: 横坡两侧单点法向力的相对差上界。**逐位在这里是假的**：
                #: 四个点按声明次序累加，`±y`两项不精确相消，实测4e-14。
                "cross_slope_asymmetry_bound": 1.0e-11,
                #: 平衡态合力矩的上界（判的是`≈0`，所以是bound不是expected）。
                #: 实测1.02e-12 N·mm，取1e-8约一万倍余量——**它挡的是一项真的
                #: 净力矩，不是舍入**，所以不许按"反正很小"随手放宽。
                "residual_torque_bound_nmm": 1.0e-8,
            },
            "expected": {
                "loaded_support_count": 4,
                "normal_force_uphill_n": f_up,
                "normal_force_downhill_n": f_dn,
                "pressure_centre_offset_mm": pressure_centre,
                "pressure_centre_over_halfwidth": pressure_centre / HALF_W_MM,
                "settled_tilt_rad": compliance_tilt(
                    theta_s, NORMAL_STIFFNESS_N_PER_MM, 4
                ),
                "creep_speed_mm_per_s": creep,
                "uphill_support_saturates": saturates,
                #: 法向力对质心的力矩：`τ_n = −(Σ x_i F_i)·ŷ = −h·W·sinθ·ŷ`。
                #: **球那一档它是结构性的零，这一档它是倾覆力矩本身**——
                #: 同一个字段两档语义不同，所以两档各配一条判据。
                "normal_torque_cross_slope_nmm": -HALF_H_MM * WEIGHT_N * math.sin(theta_s),
            },
            "tolerances": {
                "loaded_support_count": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差**：撑住的定义就是四个支承点全在承载。少一个就是已经在抬边了，而那是另一件事——把它归进容差等于把翻倒的前兆当噪声",
                },
                "normal_force_uphill_n": {
                    "abs": 0.0,
                    "rel": 4.0e-3,
                    "reason": "闭式是**刚体**的载荷分配，而罚接触的平衡带`O(1/k)`柔度修正，且上坡侧摩擦已饱和（见清单同条`uphill_support_saturates`），切向力分配因此不是闭式假设的均分。实测−2.52e-3，取4e-3约1.6倍余量。**这一条比压心那一条松一个数量级，正是本案例要展示的事**：单点力带模型自带偏差，而它们的一阶矩不带",
                },
                "normal_force_downhill_n": {
                    "abs": 0.0,
                    "rel": 4.0e-3,
                    "reason": "同上，与上坡侧共用一条容差以免两条各自被调松。实测+3.96e-4，余量10倍",
                },
                "pressure_centre_offset_mm": {
                    "abs": 0.0,
                    "rel": 1.0e-4,
                    "reason": "`x_N = h·tanθ`是**力矩平衡的恒等式**不是载荷分配的近似——法向力的一阶矩恒等于摩擦力的力偶，柔度在两边同阶相消。实测+1.76e-5，取1e-4约5.7倍余量。**它比单点力紧40倍，这个差本身是判据**：一个把载荷分配算错但总力矩仍对的实现会在上面两条红、这一条绿",
                },
                "pressure_centre_over_halfwidth": {
                    "abs": 0.0,
                    "rel": 1.0e-4,
                    "reason": "同上，除以`w`之后就是『重心投影离底面边缘还有多远』这个无量纲量（<1 撑住、>1 翻倒）。它与上一条同源，写成两条是因为**判据要读的是这个比值**而不是那个长度",
                },
                "settled_tilt_rad": {
                    "abs": 0.0,
                    "rel": 3.0e-3,
                    "reason": "`β`由两侧穿透差算，因此继承法向力那一条的柔度偏差。实测+9.40e-4，取3e-3约3.2倍余量",
                },
                "creep_speed_mm_per_s": {
                    "abs": 0.0,
                    "rel": 5.0e-3,
                    "reason": "速度型摩擦的稳态蠕滑闭式**含饱和分支**。实测+2.05e-3；**朴素闭式`W sinθ/(4k_t)`会差10.5%，是本容差的21倍**——所以这条容差同时在判『哪几个点饱和』，不是在判一个速度",
                },
                "normal_torque_cross_slope_nmm": {
                    "abs": 0.0,
                    "rel": 1.0e-4,
                    "reason": "它与压心那一条是同一条恒等式的两种写法（`τ_n = −x_N·N`），所以共用同一档紧容差1e-4。**分开写两条的理由是它们坏起来不一样**：压心那条从逐点力现场加权，这一条读的是`SupportSetResponse`里**装配好的**那个力矩——一个只把总力矩算错（比如切向力也混进了法向那一半）的实现在压心上绿、在这一条上红",
                },
                "uphill_support_saturates": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**布尔量零容差**。它不是被测的物理量而是判据本身：蠕滑那条闭式取哪一支由它决定，而`sliding`标志是引擎自己报的——两者对不上就是实现与模型脱节",
                },
            },
        },
        {
            "id": "oracle:box_tipping/topple",
            "inputs": {
                **common,
                "kind": "box_on_incline_above_threshold",
                "incline_deg": TOPPLE_INCLINE_DEG,
                "normal_stiffness_n_per_mm": NORMAL_STIFFNESS_N_PER_MM,
                "dt_s": 2.0e-4,
                "steps": 1500,
                "lift_off_step": 250,
                #: 『真的翻过去』的门槛。取1.5 rad（85.9°）——它远在平衡角
                #: `ψc − θ = −0.0948 rad`之外，**不可能是一次晃动**。
                "tilt_gate_rad": 1.5,
                "penetration_bound_mm": 0.05,
            },
            "expected": {
                #: `box_corner_points_mm`的次序声明：0—3是底面。
                "initial_loaded_indices": [0, 1, 2, 3],
                #: 1/3/5/7 是`x_body = +w`那**一整个面**——下坡侧面。
                "final_loaded_indices": [1, 3, 5, 7],
                "lift_off_loaded_indices": [1, 3],
                #: 落到侧面上时底面法向恰好转到坡向：`φ = π/2`。
                "final_tilt_rad": 0.5 * math.pi,
            },
            "tolerances": {
                "initial_loaded_indices": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差**：这是一组下标不是一个测量。起手必须是整个底面在承载，否则『接触点从一侧转到另一侧』这句话就没有起点",
                },
                "final_loaded_indices": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差，而且这是本案例最要紧的一条**：末态承载的是`x_body = +w`那个面的四个角，**与起手那四个角一个都不重合**。姿态角可以靠一次数值发散做出来，承载点从一个面整体换到另一个面不能",
                },
                "lift_off_loaded_indices": {
                    "abs": 0.0,
                    "rel": 0.0,
                    "reason": "**零容差**：中途必须先抬起上坡侧那两个角。没有这一步就不是绕下坡棱翻，而是整体弹起来——两者末态可能相同，过程不同",
                },
                "final_tilt_rad": {
                    "abs": 2.0e-3,
                    "rel": 0.0,
                    "reason": "落到侧面后底面法向与坡面法向差`π/2`，偏差只由该姿态下的罚穿透造成（两对角穿透不等，量级`δ/(2h)`）。实测+6.50e-5 rad，取2e-3约31倍余量。**用abs不用rel**：判的是与直角的差，不是`π/2`这个数本身的相对精度",
                },
            },
        },
        {
            "id": "oracle:box_tipping/critical_angle_bracket",
            "inputs": {
                **common,
                "kind": "box_tipping_critical_angle_bisection",
                "support_points_body_mm": [
                    [-HALF_W_MM, 0.0, -HALF_H_MM],
                    [HALF_W_MM, 0.0, -HALF_H_MM],
                ],
                "soft_stiffness_n_per_mm": soft,
                "stiff_stiffness_n_per_mm": stiff,
                "soft_iterations": 7,
                "stiff_iterations": 9,
                "search_halfwidth_rad": 0.01,
                "dt_s": 4.0e-4,
                "soft_steps": 1000,
                "stiff_steps": 1100,
                "tip_gate_rad": 0.05,
                #: 刚体闭式与两条柔度修正——它们是**判不等式**用的，
                #: 不是被测量，所以住在inputs里（与`expected`的分工见案例页第三节）。
                "rigid_critical_angle_rad": psi_c,
                "compliance_tilt_soft_rad": beta_soft,
                "compliance_tilt_stiff_rad": beta_stiff,
                "exact_tangent_threshold_numerator": tangent.numerator,
                "exact_tangent_threshold_denominator": tangent.denominator,
            },
            "expected": {
                "critical_angle_soft_rad": psi_c - beta_soft,
                "critical_angle_stiff_rad": psi_c - beta_stiff,
                "deviation_ratio_soft_over_stiff": stiff / soft,
            },
            "tolerances": {
                "critical_angle_soft_rad": {
                    "abs": 7.8125e-5,
                    "rel": 0.0,
                    "reason": "**容差就是二分的半夹取宽度**（`0.01·2/2⁷/2`）——比这更紧的声称二分给不出来，比这更松就白夹了。实测偏差+1.81e-05，落在半宽之内。**用abs不用rel**：夹取宽度是绝对量，与角度大小无关",
                },
                "critical_angle_stiff_rad": {
                    "abs": 1.953125e-5,
                    "rel": 0.0,
                    "reason": "同上，`0.01·2/2⁹/2`。实测偏差+4.51e-06。刚度4倍时夹得紧4倍，**是为了让偏差之比那一条两端的分辨力同阶**",
                },
                "deviation_ratio_soft_over_stiff": {
                    "abs": 0.0,
                    "rel": 0.2,
                    "reason": "偏差随`1/k`趋零的直接判据：软/硬两档偏离刚体闭式的量之比应等于刚度之比4。**容差由夹取宽度决定而不是由物理**——两端各带约9.1%的夹取不确定度，合计约18%，取0.2。实测比值恰为4.0000，那是二分落在同一个格点上的巧合，**不许被读成额外精度**",
                },
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/box_tipping_threshold",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/box_tipping_threshold/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)

    print(f"tan 阈值（精确）  = {tangent} = {float(tangent):.17f}")
    print(f"ψc = arctan(w/h) = {psi_c:.17f} rad = {math.degrees(psi_c):.9f}°")
    print(f"稳定侧 θ={STABLE_INCLINE_DEG}°: F_up={f_up:.9f} F_dn={f_dn:.9f}")
    print(f"   每点需求={demand_per_point:.9f}  μF_up={FRICTION * f_up:.9f}  饱和={saturates}")
    print(f"   x_N={pressure_centre:.9f} mm （w={HALF_W_MM}，比值{pressure_centre / HALF_W_MM:.9f}）")
    print(f"   蠕滑（含饱和）={creep:.9f} mm/s   朴素式={WEIGHT_N * math.sin(theta_s) / (4 * TANGENTIAL_STIFFNESS):.9f}")
    print(f"柔度修正 β(k={soft})={beta_soft:.9e}  β(k={stiff})={beta_stiff:.9e}")
    print(f"预言临界角 软={psi_c - beta_soft:.12f}  硬={psi_c - beta_stiff:.12f}")
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
