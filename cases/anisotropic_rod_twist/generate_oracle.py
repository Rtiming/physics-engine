#!/usr/bin/env python3
"""整杆各向异性弯曲与扭转的金标——**四条闭式，全部独立于被验内核**。

## 一、螺旋线运动学（含一条恰好为零的判据）

参数化``x(θ) = (R cos θ, R sin θ, p θ)``的曲率与挠率是教科书闭式：

    κ = R/(R² + p²),    τ = p/(R² + p²)

**第三个数是零，而且它是结构零不是收敛零。** 把材料帧的``m1``取成解析主法线
``n(θ) = −(cos θ, sin θ, 0)``后，螺旋线上**没有任何hard-way弯曲**：

* 顶点``θ_v``两侧的两条弦，其叉积（离散曲率二法矢``κb``）与解析副法线共线；
* 两条边**中点**的法线之和``n(θ_v−Δ/2) + n(θ_v+Δ/2) = 2cos(Δ/2)·n(θ_v)``，
  **严格平行于**顶点法线；
* 而``b(θ_v)·n(θ_v) = 0``。

于是``κ2 = −0.5·(m1_l + m1_r)·κb``**恰好为零**，与离散步长``Δ``无关。
这一条比前两条锋利：前两条是二阶收敛量，这一条是**机器零**。

## 二、各向异性悬臂的挠度比

矩形截面绕两个主轴的二阶矩差``(w/h)²``倍。带材（宽``w``、厚``h``）：

    EI_easy = E·w·h³/12   （朝厚度方向弯，中性轴沿带宽）
    EI_hard = E·h·w³/12   （在带宽面内弯，edgewise）

小挠度端载悬臂``δ = F·L³/(3·EI)``，于是**同一根杆、同一个载荷、
只把参考``d1``绕轴转90°**，挠度比必须恰是``EI_hard/EI_easy``。

这条是本页最便宜也最要紧的一条：同行（WDS）自己在
`test_gravity_cantilever.py`第40—46行的docstring里点名了这个失效模式
——参考``d1``取错会让``EI_hard``接管、挠度差1600倍、**不报任何错**
——**而那个失效模式在他们那边没有门守着**。

## 三、端扭矩杆

``θ = M·L/(GJ)``。**``L``是端边中点之间的距离**，离散模型里等于``Σ l̄_i``：
扭转弹簧住在内顶点上，``n_edges``条边之间有``n_edges − 1``段。
等分时它是``(n_edges − 1)·h``，**比杆长少一个``h``**。
这条形制差别写进案例页的失效清单，因为拿杆长去对会得到一个假的百分数偏差。

## 四、球面三角的平行输运holonomy

切向序列``x̂ → ŷ → ẑ → x̂``在单位球面上围出一个**三个角都是直角**的测地三角形。
球面三角形的面积（＝立体角）是``(A+B+C−π)·R² = (3·π/2 − π) = π/2``，
也等于八分之一球面``4π/8``。**由Gauss-Bonnet，沿这条闭合路径平行输运一周的
holonomy恰是这个立体角**，即``π/2``。

这条是retransport外层循环唯一的独立oracle：杆从直走到这个构型时，
材料帧保住不变而新构型的Bishop帧带着``π/2``的holonomy，边扭角必须把它吃下去。
**不重输运则那个π/2永远不出现，扭转能量恒等于零。**

两端材料帧夹持、中间自由时，串联扭簧的能量是``0.5·GJ·Θ²/Σl̄``。

## 五、生成器不做什么

不调`solve_equilibrium`、不调`physics_engine.rod`、不调任何能量项、
不引任何引擎的力学模块。只有`math`与清单写入。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/anisotropic_rod_twist"
ALGORITHM_VERSION = "1.0.0"

#: 螺旋线：半径与螺距参数（mm）。取``p/R = 0.3``是为了让挠率与曲率同量级——
#: ``p``太小时``τ``会被离散误差淹没，那样"验挠率"就名不副实。
HELIX_RADIUS_MM = 40.0
HELIX_PITCH_MM = 12.0

#: 悬臂：两个主轴刚度取1:1000。**不是那条带材的真实比**（4mm×0.1mm是1600），
#: 取整千是为了让比值判据一眼可读；真实带材的两个数在plans/14第2.3节。
EI_EASY_NMM2 = 8.0e4
EI_HARD_NMM2 = 8.0e7
CANTILEVER_LENGTH_MM = 50.0
CANTILEVER_LOAD_N = -0.05

#: 扭转：``GJ ≈ G·(1/3)·w·t³ = 77 N·mm²``（plans/14第2.3节，4mm×0.1mm带材、ν=0.3）。
GJ_NMM2 = 77.0
TORSION_NODE_COUNT = 21
TORSION_LENGTH_MM = 200.0
END_MOMENT_N_MM = 0.5

#: holonomy算例：四条等长边，切向``x̂ → ŷ → ẑ → x̂``。
HOLONOMY_SEGMENT_MM = 10.0


def helix_curvature_per_mm(radius_mm: float, pitch_mm: float) -> float:
    return radius_mm / (radius_mm**2 + pitch_mm**2)


def helix_torsion_per_mm(radius_mm: float, pitch_mm: float) -> float:
    return pitch_mm / (radius_mm**2 + pitch_mm**2)


def tip_deflection_mm(force_n: float, length_mm: float, bending_nmm2: float) -> float:
    return force_n * length_mm**3 / (3.0 * bending_nmm2)


def torsion_effective_length_mm(node_count: int, length_mm: float) -> float:
    """端边中点之间的距离``Σ l̄_i = (边数 − 1)·h``。**它比杆长少一个``h``。**"""

    step = length_mm / (node_count - 1)
    return (node_count - 2) * step


def main() -> int:
    kappa = helix_curvature_per_mm(HELIX_RADIUS_MM, HELIX_PITCH_MM)
    tau = helix_torsion_per_mm(HELIX_RADIUS_MM, HELIX_PITCH_MM)
    easy = tip_deflection_mm(CANTILEVER_LOAD_N, CANTILEVER_LENGTH_MM, EI_EASY_NMM2)
    hard = tip_deflection_mm(CANTILEVER_LOAD_N, CANTILEVER_LENGTH_MM, EI_HARD_NMM2)
    effective = torsion_effective_length_mm(TORSION_NODE_COUNT, TORSION_LENGTH_MM)
    holonomy = 0.5 * math.pi
    twist_energy = (
        0.5 * GJ_NMM2 * holonomy**2 / (3.0 * HOLONOMY_SEGMENT_MM)
    )
    oracles = [
        {
            "id": "oracle:rod/helix_kinematics",
            "inputs": {
                "kind": "helix_curvature_torsion",
                "radius_mm": HELIX_RADIUS_MM,
                "pitch_mm": HELIX_PITCH_MM,
                "material_m1": "analytic principal normal at the edge midpoint",
            },
            "expected": {
                "curvature_per_mm": kappa,
                "torsion_per_mm": tau,
                "hard_way_curvature": 0.0,
            },
            "tolerances": {
                "curvature_per_mm": {
                    "abs": 0.0, "rel": 3.0e-4,
                    "reason": (
                        "离散弦近似是**二阶**：N=21/41/81/161实测相对偏差"
                        "9.753e-4/2.437e-4/6.093e-5/1.523e-5，比值恒为4.00。"
                        "本条判N=41那一档的落点，3e-4是实测2.437e-4的约1.2倍余量；"
                        "**收敛阶另有一条门判**"
                    ),
                },
                "torsion_per_mm": {
                    "abs": 0.0, "rel": 5.0e-4,
                    "reason": (
                        "挠率由材料帧相对Bishop帧的扭率给出，同样二阶："
                        "1.531e-3/3.824e-4/9.557e-5/2.389e-5，比值4.00。"
                        "5e-4是N=41实测3.824e-4的约1.3倍余量"
                    ),
                },
                "hard_way_curvature": {
                    "abs": 1.0e-13, "rel": 0.0,
                    "reason": (
                        "**判绝对不判相对**：期望值恰为零，相对容差没有意义。"
                        "这一条是**结构零**不是收敛零——实测N=21/41/81/161上"
                        "|κ2|=1.07/1.92/5.20/8.83e-15，**不随h下降**。"
                        "1e-13留的是链长增长带来的舍入累积余量"
                    ),
                },
            },
        },
        {
            "id": "oracle:rod/easy_hard_axis_swap",
            "inputs": {
                "kind": "tip_loaded_cantilever_with_frame_rotated_ninety_degrees",
                "length_mm": CANTILEVER_LENGTH_MM,
                "force_n": CANTILEVER_LOAD_N,
                "ei_easy_nmm2": EI_EASY_NMM2,
                "ei_hard_nmm2": EI_HARD_NMM2,
            },
            "expected": {
                "easy_tip_mm": easy,
                "hard_tip_mm": hard,
                "deflection_ratio": EI_HARD_NMM2 / EI_EASY_NMM2,
            },
            "tolerances": {
                "easy_tip_mm": {
                    "abs": 0.0, "rel": 1.0e-3,
                    "reason": (
                        "端载悬臂闭式；离散侧含**固支半格柔度**订正后二阶收敛"
                        "（N=11/21/41/81实测2.350e-2/6.063e-3/1.539e-3/3.880e-4，"
                        "比值3.876/3.939/3.968）。**本条判N=81那一档**，"
                        "1e-3是实测3.880e-4的约2.6倍余量；"
                        "N=41及更粗的档由收敛阶那条门判，不由本条判"
                    ),
                },
                "hard_tip_mm": {
                    "abs": 0.0, "rel": 1.0e-3,
                    "reason": "同上；hard轴挠度是easy的1/1000，判据同形",
                },
                "deflection_ratio": {
                    "abs": 0.0, "rel": 1.0e-6,
                    "reason": (
                        "**比值比落点紧三个数量级**，因为离散误差在两个构型上完全相同、"
                        "相除即消。实测999.99972、偏差2.79e-7，且它随载荷**平方**下降"
                        "（F=0.05/0.005/0.0005 N → 2.79e-7/2.79e-9/2.65e-11）"
                        "——那是几何非线性，不是这条门的噪声。1e-6是3.6倍余量"
                    ),
                },
            },
        },
        {
            "id": "oracle:rod/end_torque_twist",
            "inputs": {
                "kind": "cantilever_end_torque",
                "moment_n_mm": END_MOMENT_N_MM,
                "gj_nmm2": GJ_NMM2,
                "node_count": TORSION_NODE_COUNT,
                "rod_length_mm": TORSION_LENGTH_MM,
            },
            "expected": {
                "effective_length_mm": effective,
                "tip_twist_rad": END_MOMENT_N_MM * effective / GJ_NMM2,
                "uniform_twist_rate_per_mm": END_MOMENT_N_MM / GJ_NMM2,
            },
            "tolerances": {
                "effective_length_mm": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "``Σ l̄_i = (边数−1)·h = 190 mm``，**杆长是200 mm**。"
                        "那10 mm是形制不是误差，见案例页失效清单第2条"
                    ),
                },
                "tip_twist_rad": {
                    "abs": 0.0, "rel": 1.0e-13,
                    "reason": (
                        "离散模型里这条方程是**线性**的，牛顿一步收敛，"
                        "所以偏差应当在机器精度：实测4.885e-15。"
                        "1e-13是约20倍余量，留给链长变化"
                    ),
                },
                "uniform_twist_rate_per_mm": {
                    "abs": 0.0, "rel": 1.0e-13,
                    "reason": (
                        "等截面等分杆上扭率必须逐顶点相同（实测极差6.85e-17）；"
                        "**这一条与上一条不是同一件事**——上一条判总量，本条判分布"
                    ),
                },
            },
        },
        {
            "id": "oracle:rod/spherical_triangle_holonomy",
            "inputs": {
                "kind": "gauss_bonnet_geodesic_triangle",
                "tangent_path": ["+x", "+y", "+z", "+x"],
                "segment_mm": HOLONOMY_SEGMENT_MM,
                "gj_nmm2": GJ_NMM2,
            },
            "expected": {
                "holonomy_rad": holonomy,
                "series_spring_twist_energy_n_mm": twist_energy,
                "twist_energy_without_retransport_n_mm": 0.0,
            },
            "tolerances": {
                "holonomy_rad": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": (
                        "球面三角形三个角都是直角，面积``4π/8 = π/2``；"
                        "离散侧的平行输运是**精确转动的复合**，没有离散化误差——"
                        "实测重输运后``γ₃−γ₀``与``−π/2``**逐位相同**"
                    ),
                },
                "series_spring_twist_energy_n_mm": {
                    "abs": 0.0, "rel": 1.0e-14,
                    "reason": (
                        "两端夹持、中间自由的串联扭簧闭式``0.5·GJ·Θ²/Σl̄``；"
                        "实测3.166498078682835 vs 闭式…356，相对偏差1.4e-16"
                    ),
                },
                "twist_energy_without_retransport_n_mm": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": (
                        "**零容差、判绝对相等**：``m_ref``冻结时扭转项对位置没有依赖，"
                        "单次求解拿到的扭转能量**恰是浮点零**。"
                        "这一条是「抄了公式不抄外循环」的判据本身——"
                        "它期望的就是那个错误答案，用来证明这道门真的在区分两条路"
                    ),
                },
            },
        },
    ]
    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/anisotropic_rod_twist",
        "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/anisotropic_rod_twist/generate_oracle.py",
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
