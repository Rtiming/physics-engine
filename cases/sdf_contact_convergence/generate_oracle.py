#!/usr/bin/env python3
"""距离场接触的收敛判据——**闭式，独立于被验内核**。

本脚本一行都不import`physics_engine.contact`。它解的是同一道题的手推答案，
conformance测试再拿内核去撞它。

## 一、构型

一个节点（代表半径``r``的球）被一个恒定的下压力``F``按在障碍上，
横向两个自由度钉住、只留``z``。障碍有两档：

* **半空间**：``φ(x) = z``（过原点、外法向``+z``）；
* **球**：``φ(x) = |x| − R``，球心在原点。

罚接触``U = ½k·g²``（``g = φ − r < 0``时活动），平衡由``k·|g| = F``给出。

## 二、三条闭式

### 闭式一：平衡位置

    半空间：z* = r − F/k
    球　　：z* = R + r − F/k

初等代数，**没有一处求积、没有一处迭代**。

### 闭式二：三次B样条场把``z*``推到哪里去（**本案例的主角**）

采样值直接当B样条系数的拟插值，其展开是（`contact/field.py`模块docstring）

    S(x) = φ(x) + (h²/6)·∇²φ(x) + O(h⁴)

那个``1/6``是三次B样条二阶矩``Σ_k (k − t)²·B(k − t) = 1/3``的一半，
**与胞内相位``t``无关**（`tests/test_contact_field.py`对``t = 0/0.25/0.5/0.75``各验一次）。
逐档代进去：

* **半空间**：``φ``是**仿射**的，``∇²φ ≡ 0`` ⟹ **误差项恒为零**。
  于是场与解析接触项给出**同一个平衡位置**，**与``h``无关**；
* **球**：``∇²(|x| − R) = 2/|x|`` ⟹ 在``z*``附近

      Δz = z_h − z* = −h²/(3·z*)

  **符号是负的**——场报出来的距离比真值**大**，于是物体**沉得更深**。
  这就是0074第二节第4条那句"误差不是随机的、在高曲率处系统性偏保守或偏松"的
  定量形式：**对凸障碍它是偏松，不是偏保守。**

### 闭式三：这个偏差与罚穿透的比

罚穿透是``F/k``（0050第二节），场引入的偏差是``h²/(3z*)``，于是

    偏差 / 穿透 = k·h² / (3·z*·F)

本构型（``k = 5000 N/mm``、``F = 25 N``、``z* = 10.495 mm``、``h = 1 mm``）
给**6.3746**——**场的位置误差是罚穿透的六倍多**。
这条比值是本案例最要紧的一个数：它说明在这个刚度档上，
**决定位置精度的是分辨率而不是罚刚度**，把``k``提十倍一点用都没有。

## 三、金标为什么是这个脚本而不是文献

按案例页第二节的三选一，本案例走的是**②无闭式解则生成脚本入库**——
但要说清楚：三条闭式全是初等代数与一次泰勒展开，本可以直接写在页上；
入库脚本是为了让那些数**有一个被SHA钉住的出处**，
而不是因为它们算不出来。**实测数不作金标**（spec/08规则1）：
内核的数拿来撞这里的数，不是反过来。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/sdf_contact_convergence"
ALGORITHM_VERSION = "1.0.0"

#: 障碍与节点。球半径10 mm是0074第5.1节那张表里``e1_carrier``包围盒的量级，
#: 节点半径0.5 mm对应带材边缘的圆角档（0075那一族用的是2.0 mm）。
SPHERE_RADIUS_MM = 10.0
NODE_RADIUS_MM = 0.5
STIFFNESS_N_PER_MM = 5.0e3
HOLD_DOWN_N = 25.0
#: 三档分辨率，**每档减半**——收敛比落在4上时那个4是货真价实的二阶。
SPACINGS_MM = (1.0, 0.5, 0.25)
#: 半空间那一档的分辨率跨度更大（2.0—0.5）：那里的判据是"与``h``无关"，
#: 跨度越大这句话越有分量。
PLANE_SPACINGS_MM = (2.0, 1.0, 0.5)


def sphere_equilibrium_z_mm() -> float:
    return SPHERE_RADIUS_MM + NODE_RADIUS_MM - HOLD_DOWN_N / STIFFNESS_N_PER_MM


def plane_equilibrium_z_mm() -> float:
    return NODE_RADIUS_MM - HOLD_DOWN_N / STIFFNESS_N_PER_MM


def sphere_offset_mm(spacing_mm: float) -> float:
    """``Δz = −h²/(3 z*)``——闭式二的球那一档。"""

    return -spacing_mm * spacing_mm / (3.0 * sphere_equilibrium_z_mm())


def main() -> int:
    oracles: list[dict] = []
    star = sphere_equilibrium_z_mm()

    algebra_reason = (
        "**零容差**，理由是一条代数事实而不是精度自信：活动接触段的能量"
        "``U = ½k(z − c)² − F·z``对``z``**是严格二次的**（球那一档在轴上``|x| = z``，"
        "所以也是），于是牛顿**一步走到精确解**，那一步就是一次除法``F/k``。"
        "实测两档都是``diff = 0.0``（逐位）、迭代次数1。"
        "**这是本案例唯一两条零容差**；它们一旦在别的平台上红，"
        "红的是'牛顿那一步在那台机器上不是精确的'这件事——**而那本身值得知道**。"
    )
    oracles.append(
        {
            "id": "oracle:sdf_contact_convergence/analytic_equilibrium",
            "inputs": {
                "kind": "analytic_equilibrium",
                "sphere_radius_mm": SPHERE_RADIUS_MM,
                "node_radius_mm": NODE_RADIUS_MM,
                "stiffness_n_per_mm": STIFFNESS_N_PER_MM,
                "hold_down_n": HOLD_DOWN_N,
                "note": (
                    "解析接触项的平衡，闭式一。半空间用`PenaltyNormalContact`，"
                    "球用`PenaltySphereContact`把第二个节点钉在球心——"
                    "**仓里本来就有一个解析球障碍**，不需要为本案例新造一个。"
                ),
            },
            "expected": {
                "plane_equilibrium_z_mm": plane_equilibrium_z_mm(),
                "sphere_equilibrium_z_mm": star,
                "contact_gap_mm": -HOLD_DOWN_N / STIFFNESS_N_PER_MM,
                "normal_force_n": HOLD_DOWN_N,
            },
            "tolerances": {
                key: {"rel": 0.0, "abs": 0.0, "reason": algebra_reason}
                for key in ("plane_equilibrium_z_mm", "sphere_equilibrium_z_mm")
            }
            | {
                "contact_gap_mm": {
                    "rel": 0.0,
                    "abs": 1.0e-16,
                    "reason": (
                        "**这一条不是零容差**，与上面两条差在表达式上：内核算的是"
                        "``(x − p)·n − r``（一条减法链），闭式算的是``F/k``（一次除法），"
                        "**两串运算不同**，逐位相等没有理由成立。"
                        "实测差``4.337e-18 mm``（5 ulp），1e-16留两个量级。"
                    ),
                },
                "normal_force_n": {
                    "rel": 1.0e-12,
                    "abs": 0.0,
                    "reason": (
                        "力在平衡处**精确**等于外载（0050第二节：``k·δ = N``恒成立，"
                        "与``k``无关），但它是牛顿迭代出来的，带残差容差``1e-9 N``；"
                        "1e-12相对 = 2.5e-11 N，比残差容差还紧两个量级——"
                        "**这条能过说明收敛得比判据要求的更深**，不是说判据松。"
                    ),
                }
            },
        }
    )

    plane_reason = (
        "**仿射函数被三次B样条精确重构**（一阶矩为零），于是场与解析项解的是"
        "同一个方程，差别只有求和次序带来的舍入。实测三档``h``全部差**1 ulp**"
        "（2.220e-16 mm）。判据取``abs = 1e-12 mm``——留四个量级余量，"
        "**但不写零**：两条路的求和次序不同，逐位相等是没验过的、也不该被承诺的性质。"
    )
    for spacing in PLANE_SPACINGS_MM:
        oracles.append(
            {
                "id": (
                    "oracle:sdf_contact_convergence/plane_field_h_"
                    + str(spacing).replace(".", "p")
                ),
                "inputs": {
                    "kind": "plane_field_equilibrium",
                    "spacing_mm": spacing,
                    "note": "半空间是仿射的：场给的平衡位置与h无关，也与解析项一致。",
                },
                "expected": {
                    "equilibrium_z_mm": plane_equilibrium_z_mm(),
                    "normal_force_n": HOLD_DOWN_N,
                },
                "tolerances": {
                    "equilibrium_z_mm": {
                        "rel": 0.0,
                        "abs": 1.0e-12,
                        "reason": plane_reason,
                    },
                    "normal_force_n": {
                        "rel": 1.0e-9,
                        "abs": 0.0,
                        "reason": plane_reason,
                    },
                },
            }
        )

    sphere_reason = (
        "闭式二给的是**主项**``−h²/(3z*)``，实测值还带一个``O(h⁴)``的余项。"
        "实测（h = 1 / 0.5 / 0.25）主项与实测之差是"
        "``−1.1164e-04 / −9.8074e-06 / −1.3220e-06``mm，"
        "而主项本身是``−3.176e-02 / −7.940e-03 / −1.985e-03``mm。"
        "于是本条按**相对主项**判：容差取``rel = 5e-3``，"
        "覆盖``h = 1``那一档最差的0.35%余项，并留一个量级。"
        "**不按绝对判**：绝对容差会让粗档松、细档紧，量不出'主项抓对了没有'。"
    )
    for spacing in SPACINGS_MM:
        oracles.append(
            {
                "id": (
                    "oracle:sdf_contact_convergence/sphere_field_h_"
                    + str(spacing).replace(".", "p")
                ),
                "inputs": {
                    "kind": "sphere_field_equilibrium",
                    "spacing_mm": spacing,
                    "note": "球是凸的：场系统性偏松，物体沉得更深，主项 −h²/(3z*)。",
                },
                "expected": {
                    "equilibrium_offset_mm": sphere_offset_mm(spacing),
                    "equilibrium_z_mm": star + sphere_offset_mm(spacing),
                },
                "tolerances": {
                    "equilibrium_offset_mm": {
                        "rel": 5.0e-3,
                        "abs": 0.0,
                        "reason": sphere_reason,
                    },
                    "equilibrium_z_mm": {
                        "rel": 0.0,
                        "abs": 5.0e-3 * abs(sphere_offset_mm(SPACINGS_MM[0])),
                        "reason": (
                            "同一条判据的另一种写法（绝对位置），容差按最粗那一档的"
                            "主项×5e-3折算成绝对量。**两种写法都留着**是有意的："
                            "偏差那一列判'主项抓对了没有'，位置那一列判"
                            "'算出来的还是不是同一个平衡'——后者能挡住"
                            "'偏差对了但基准位置整体挪了'这一类错。"
                        ),
                    },
                },
            }
        )

    oracles.append(
        {
            "id": "oracle:sdf_contact_convergence/order_and_penalty_ratio",
            "inputs": {
                "kind": "order_and_penalty_ratio",
                "spacings_mm": list(SPACINGS_MM),
                "note": "阶＋闭式三：场的位置误差与罚穿透之比。",
            },
            "expected": {
                "position_error_order_ratio": 4.0,
                "gradient_finite_difference_ratio": 4.0,
                "field_error_over_penalty_penetration_at_coarsest": (
                    abs(sphere_offset_mm(SPACINGS_MM[0]))
                    / (HOLD_DOWN_N / STIFFNESS_N_PER_MM)
                ),
            },
            "tolerances": {
                "position_error_order_ratio": {
                    "rel": 2.5e-2,
                    "abs": 0.0,
                    "reason": (
                        "**收敛比是区间不是恒等于4**（`cases/harmonic_oscillator`那条"
                        "同源：4是渐近值，粗档还没完全进渐近区）。2.5e-2相对 = "
                        "``[3.9, 4.1]``，与本仓其余二阶门同一个窗口，"
                        "实测**4.0091 / 4.0023**。"
                    ),
                },
                "gradient_finite_difference_ratio": {
                    "rel": 2.5e-2,
                    "abs": 0.0,
                    "reason": (
                        "中心差分对能量的二阶收敛，实测比**恒4.0000**。"
                        "窗口与上一条同为``[3.9, 4.1]``——"
                        "**两条判的不是同一件事**：上一条判场的分辨率误差随h的阶，"
                        "这一条判梯度是不是所实现能量的导数。"
                    ),
                },
                "field_error_over_penalty_penetration_at_coarsest": {
                    "rel": 1.0e-2,
                    "abs": 0.0,
                    "reason": (
                        "闭式三是初等代数，但内核那一侧的分子是**解出来的位置误差**，"
                        "带那条``O(h⁴)``余项（h=1时占主项的0.35%）。"
                        "1e-2相对覆盖它并留一档。**这条数是本案例最要紧的一个**："
                        "6.37倍说明这个刚度档上决定位置精度的是分辨率不是罚刚度。"
                    ),
                },
            },
        }
    )

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/sdf_contact_convergence",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/sdf_contact_convergence/generate_oracle.py",
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
