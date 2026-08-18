#!/usr/bin/env python3
"""无滑滚球与滑动滚球的金标——**四条闭式，用`Fraction`精确算，不import被验的任何模块**。

    .venv/bin/python cases/rolling_ball_incline/generate_oracle.py

执行计划见`docs/plans/16_自主长跑_路径B与性能模块化_20260818.md`的M2、M3，
开工前置见`docs/decisions/0081`第三节（那四条闭式在那里推的）。

## 为什么这里再推一遍，而不是把0081的数抄下来

与`three_sphere_pyramid_rotational`同一条纪律：**别人的数不是真值，是另一个证人**。
0081第三节是我用`Fraction`推的，本脚本**再独立推一遍**，两处对不上就说明有一处错了。

**而且这条纪律刚刚被验证过一次**：0081第二节记着派活方（我）一度把无滑上限写成`3μ`，
那是**实心圆盘**的数——实心球是`(7/2)μ`。**一个凭印象写下的数活到了任务书里**，
只因为写代码前用精确算术核了一遍才没进实现。

## 四条闭式（`I = k·mR²`，实心球`k = 2/5`）

    无滑滚下坡的质心加速度     a = g·sinθ / (1 + k)              = (5/7)·g·sinθ
    维持无滑所需的摩擦         f = k/(1+k) · m·g·sinθ            = (2/7)·m·g·sinθ
    无滑与滑的分界             tanθ ≤ (1 + k)/k · μ              = (7/2)·μ
    滑着滚那一支               a = g(sinθ − μ·cosθ)
                               α = (μ·g·cosθ)/(k·R)             = 5μ·g·cosθ/(2R)

**第四条那两个量不再由`a = αR`联系——那正是"滑了"的可观测定义**（0081第三节第4条）。
本案例把它做成一个**比值判据**：滚动时`a/(αR) = 1`，滑动时它是一个**可算的、不等于1的数**：

    a/(αR) = (sinθ − μcosθ) / (μcosθ/k) = k·(tanθ/μ − 1)

## 一次真正独立的交叉验证

research/17第四节实测：**同行三家（Chrono／Bullet／MuJoCo）没有一家为这条闭式立过数值门**
——Chrono那两条是终态阈值、Bullet那条压根没有断言、MuJoCo那条是隐式积分的能量上界。
唯一给出闭式推导的是Fang & Negrut 2021附录D，它推出

    α ≤ tan⁻¹(3.5·μs − 5·ηr·μk)

**令滚阻项`ηr = 0`恰是上面第三条**。两条独立路径（我们的`Fraction`、他们的解析推导）同一个数。
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

ALGORITHM_ID = "algorithm:oracle/rolling_ball_incline"
ALGORITHM_VERSION = "1.0.0"

#: 实心球的惯量系数`I = k·mR²`。**用`Fraction`不用0.4**——本脚本的全部代数在有理数上做。
SOLID_SPHERE_K = Fraction(2, 5)

#: 案例参数。`g`是mm/s²；力用N，故重量是`m·g/1000`（`1 N = 1000 kg·mm/s²`）。
GRAVITY_MM_PER_S2 = 9810.0
RADIUS_MM = 10.0
MASS_KG = 1.0
NORMAL_STIFFNESS_N_PER_MM = 5.0e5
#: **切向刚度取50 N·s/mm是被实测逼出来的，不是随手取的**——见案例页第四节第1条：
#: `k_t`太大时模型进入颤振区（平均对、瞬时被钉在摩擦锥上、`sliding`恒为真），
#: 那会让"无滑"这条判据失去分辨力。
TANGENTIAL_STIFFNESS_N_S_PER_MM = 50.0
INCLINE_DEG = 20.0
ROLLING_FRICTION = 0.30
SLIDING_FRICTION = 0.05
STEPS = 20000
DT_S = 1.0e-6


def exact_no_slip_limit(k: Fraction) -> Fraction:
    """无滑上限的系数：`tanθ ≤ ((1+k)/k)·μ`。实心球给`7/2`。"""

    return (1 + k) / k


def exact_rolling_coefficient(k: Fraction) -> Fraction:
    """`a = coefficient · g·sinθ`。实心球给`5/7`。"""

    return 1 / (1 + k)


def exact_required_friction_coefficient(k: Fraction) -> Fraction:
    """维持无滑所需的摩擦：`f = coefficient · m·g·sinθ`。实心球给`2/7`。"""

    return k / (1 + k)


def main() -> int:
    k = SOLID_SPHERE_K
    rolling_coefficient = exact_rolling_coefficient(k)
    friction_coefficient = exact_required_friction_coefficient(k)
    no_slip_limit = exact_no_slip_limit(k)

    #: 三条精确有理数，落进清单前才化成double。**顺手判一遍它们是不是那三个熟数**——
    #: 若哪天有人把`SOLID_SPHERE_K`改成球壳的`2/3`，这三条断言会当场把它拦下。
    assert rolling_coefficient == Fraction(5, 7), rolling_coefficient
    assert friction_coefficient == Fraction(2, 7), friction_coefficient
    assert no_slip_limit == Fraction(7, 2), no_slip_limit

    theta = math.radians(INCLINE_DEG)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    tan_theta = math.tan(theta)
    weight_n = MASS_KG * GRAVITY_MM_PER_S2 / 1000.0

    critical_friction = tan_theta / float(no_slip_limit)
    rolling_acceleration = float(rolling_coefficient) * GRAVITY_MM_PER_S2 * sin_theta
    required_friction_n = float(friction_coefficient) * weight_n * sin_theta
    sliding_acceleration = GRAVITY_MM_PER_S2 * (
        sin_theta - SLIDING_FRICTION * cos_theta
    )
    sliding_angular_acceleration = (
        SLIDING_FRICTION * GRAVITY_MM_PER_S2 * cos_theta / (float(k) * RADIUS_MM)
    )
    #: `a/(αR)`：滚动时恰为1，滑动时是`k·(tanθ/μ − 1)`。**这个比值就是"滑了"的定义**。
    sliding_ratio = float(k) * (tan_theta / SLIDING_FRICTION - 1.0)

    shared = {
        "gravity_mm_per_s2": GRAVITY_MM_PER_S2,
        "radius_mm": RADIUS_MM,
        "mass_kg": MASS_KG,
        "incline_deg": INCLINE_DEG,
        "normal_stiffness_n_per_mm": NORMAL_STIFFNESS_N_PER_MM,
        "tangential_stiffness_n_s_per_mm": TANGENTIAL_STIFFNESS_N_S_PER_MM,
        "steps": STEPS,
        "dt_s": DT_S,
    }

    oracles = [
        {
            "id": "oracle:rolling_ball/exact_coefficients",
            "inputs": {"inertia_coefficient_num": 2, "inertia_coefficient_den": 5},
            "expected": {
                "rolling_coefficient": float(rolling_coefficient),
                "required_friction_coefficient": float(friction_coefficient),
                "no_slip_limit_coefficient": float(no_slip_limit),
                "critical_friction_at_incline": critical_friction,
            },
            "tolerances": {
                name: {
                    "rel": 0.0,
                    "abs": 0.0,
                    "reason": (
                        "Q上的精确有理数（5/7、2/7、7/2）化成double —— "
                        "它不是测量值是代数恒等式，容差留一点就等于允许它变成别的数"
                    ),
                }
                for name in (
                    "rolling_coefficient",
                    "required_friction_coefficient",
                    "no_slip_limit_coefficient",
                    "critical_friction_at_incline",
                )
            },
        },
        {
            "id": "oracle:rolling_ball/rolling_branch",
            "inputs": {**shared, "friction_coefficient": ROLLING_FRICTION},
            "expected": {
                "acceleration_mm_per_s2": rolling_acceleration,
                "required_friction_n": required_friction_n,
                "acceleration_over_alpha_radius": 1.0,
                #: 布尔量编码成1/0并用零容差判——清单的形制只收数，
                #: 而"滑没滑"必须进判据（0081第三节第4条）。
                "sliding_flag": 0.0,
            },
            "tolerances": {
                "acceleration_mm_per_s2": {"rel": 5.0e-3, "abs": 0.0, "reason": '罚接触的穿透是O(1/k_n)的模型自带量，速度型摩擦另有O(μN/k_t)的残余滑移——两者都进这条判据。实测偏差1.1e-04，余量约45倍。零容差在这里是错的：它会把模型自带量当成实现缺陷'},
                "required_friction_n": {"rel": 5.0e-3, "abs": 0.0, "reason": '罚接触的穿透是O(1/k_n)的模型自带量，速度型摩擦另有O(μN/k_t)的残余滑移——两者都进这条判据。实测偏差1.1e-04，余量约45倍。零容差在这里是错的：它会把模型自带量当成实现缺陷'},
                "acceleration_over_alpha_radius": {"rel": 5.0e-3, "abs": 0.0, "reason": '罚接触的穿透是O(1/k_n)的模型自带量，速度型摩擦另有O(μN/k_t)的残余滑移——两者都进这条判据。实测偏差1.1e-04，余量约45倍。零容差在这里是错的：它会把模型自带量当成实现缺陷'},
                "sliding_flag": {"rel": 0.0, "abs": 0.0, "reason": '布尔量编码成1/0，零容差——「滑没滑」是判据本身不是被测量（0081第三节第4条）'},
            },
        },
        {
            "id": "oracle:rolling_ball/sliding_branch",
            "inputs": {**shared, "friction_coefficient": SLIDING_FRICTION},
            "expected": {
                "acceleration_mm_per_s2": sliding_acceleration,
                "angular_acceleration_rad_per_s2": sliding_angular_acceleration,
                "acceleration_over_alpha_radius": sliding_ratio,
                "sliding_flag": 1.0,
            },
            "tolerances": {
                "acceleration_mm_per_s2": {"rel": 5.0e-3, "abs": 0.0, "reason": '同上两项模型自带量；滑动支实测偏差1.3e-05（a）与8.4e-05（α），余量两个数量级以上'},
                "angular_acceleration_rad_per_s2": {"rel": 5.0e-3, "abs": 0.0, "reason": '同上两项模型自带量；滑动支实测偏差1.3e-05（a）与8.4e-05（α），余量两个数量级以上'},
                "acceleration_over_alpha_radius": {"rel": 5.0e-3, "abs": 0.0, "reason": '同上两项模型自带量；滑动支实测偏差1.3e-05（a）与8.4e-05（α），余量两个数量级以上'},
                "sliding_flag": {"rel": 0.0, "abs": 0.0, "reason": '布尔量编码成1/0，零容差——「滑没滑」是判据本身不是被测量（0081第三节第4条）'},
            },
        },
    ]

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/rolling_ball_incline",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/rolling_ball_incline/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print("精确系数（Fraction，全程无浮点）：")
    print(f"  a/(g sinθ)      = {rolling_coefficient}")
    print(f"  f/(m g sinθ)    = {friction_coefficient}")
    print(f"  无滑上限 tanθ/μ = {no_slip_limit}")
    print(f"  本案例 μc       = {critical_friction!r}")
    print(f"  滑动支 a/(αR)   = {sliding_ratio!r}")
    print(f"oracle.json {len(written)} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
