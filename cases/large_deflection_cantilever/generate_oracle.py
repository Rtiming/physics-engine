#!/usr/bin/env python3
"""大挠度悬臂梁的金标生成器——**Bisshopp-Drucker 1945的椭圆积分闭式**。

平面细长杆（elastica），固支端在``s=0``、自由端在``s=L``，端部受方向固定的
横向集中载荷``P``（dead load）。弯矩``M(s) = P·(x_L − x(s))``，
``EI·θ'' + P·cosθ = 0``、``θ(0)=0``、``θ'(L)=0``。首次积分给出

    θ'(s) = sqrt(2P/EI)·sqrt(sinα − sinθ)，α = θ(L)

代换``1 + sinθ = 2k²·sin²φ``（``2k² = 1 + sinα``）把三个积分化成椭圆积分：

    sqrt(β) = K(k) − F(φ₁, k)          β = P·L²/EI，sin²φ₁ = 1/(1 + sinα)
    x_L / L = sqrt(2·sinα / β)         （**这一支是初等的**：换元u=sinθ后积分显式）
    y_L / L = 1 − (2/sqrt(β))·[E(k) − E(φ₁, k)]

第一式对α单调（α从0到π/2时K增、F(φ₁)减），故用二分求α，再代入后两式。
椭圆积分用**Carlson对称形式**``R_F``/``R_D``的重复化算法求值（纯标准库、
无运行时依赖）：``F(φ,k)=sinφ·R_F``、``E(φ,k)=sinφ·R_F − (k²/3)sin³φ·R_D``。

**本生成器只算椭圆积分：不import physics_engine.energies、不调求解器。**
（除`oracles`的清单写盘工具外不碰引擎——那不是被验对象。）

三条自校验在生成时跑，任何一条不过就**不出金标**：

1. ``K``/``E``对已知值（m=0.25与0.5）逐项相对偏差 < 1e-15；
2. **小载荷极限的第一修正恰为β的二次**：``y_L/L = (β/3)·(1 − 0.1142·β + …)``、
   ``α = (β/2)·(1 − 0.0917·β + …)``。断言"β十倍则偏差百倍"（实测100.004与99.79），
   比只断言"趋近``δ=PL³/(3EI)``"强得多，且**不需要在校验里写下任何系数**；
3. 单调性：``β``增大时``y_L/L``增、``x_L/L``减。

同行先例（research/05第2.3节的C2）：WDS `validation/cantilever.py`用
scipy的QUADPACK对**同一首次积分**做端点奇点代换后的正交求积，是完全不同的
数值路线。本闭式与它在β=0.2/0.8/1.5/3.0/5.0五档上逐点相对偏差 ≤ 1.3e-14
（对拍记录见decisions/0029），两条独立路线互证。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/large_deflection_cantilever"
ALGORITHM_VERSION = "1.0.0"

LENGTH_MM = 100.0
BENDING_STIFFNESS_NMM2 = 100.0
#: 轴向刚度。elastica是**不可伸长**的；引擎侧用一根很硬的轴向弹簧逼近它。
#: EA·L²/EI = 1e6，端部轴向应变约P/EA = 3e-6，对端点位置的影响约3e-6相对——
#: 比最细档的离散误差3.2e-4低两个数量级。实测把EA换成3e4，n=80的误差从
#: 3.234393e-04变到3.244194e-04（0.3%的变化），确认它不是误差的来源。
AXIAL_STIFFNESS_N = 1.0e4
#: 无量纲载荷 β = P·L²/EI。取3.0是**故意远离小挠度**：
#: 线性理论给y/L = β/3 = 1.0，闭式给0.6033，差49.1%（见下面的判据）。
LOAD_PARAMETER = 3.0
GRAVITY_MM_S2 = 9806.65
#: 寄生质量。引擎今天只有`UniformGravity`一个外载项，端部集中力只能用
#: 集中质量表示，而`EnergyContext`要求所有节点质量为正——非端点节点取1e-15 kg。
#: 其合力 80×1e-15×9806.65/1000 = 7.8e-13 N，占端载荷0.03 N的2.6e-11。
#: 实测把它在1e-12/1e-15/1e-18之间换，n=80的误差只变到第5位有效数字。
#: **这是一处声明并有门守着的近似，不是隐藏的近似**（缺口登记见案例页第四节）。
PARASITIC_MASS_KG = 1.0e-15
REFINEMENTS = (10, 20, 40, 80)
#: 载荷步数。求解器**自己没有载荷步生长**（decisions/0027第四节），
#: 分步是**案例显式做的**：从直链一步加到β=3牛顿不收敛（线搜索在几何非线性上
#: 反复回溯）。同行先例：WDS的B1也是6步喂载。
LOAD_STEPS = 4
#: 绝对残差容差（spec/12第4.3节要求说清绝对还是相对）。**它有下界**：
#: 能量量级约0.7 N·mm，机器eps让能量在残差约1e-8 N时不再可分辨，
#: 线搜索于是无法再降能量、牛顿停滞。实测tol=1e-8时ramp=8的第2载荷步
#: 停在残差1.35e-8**物理上已经收敛却判不收敛**。取1e-7留7倍余量。
#: 上界侧：1e-7对应端点位移误差约1e-7/(3EI/L³) = 3.3e-4 mm = 3.3e-6相对，
#: 仍比最细档的离散误差3.2e-4低两个数量级。
RESIDUAL_TOL_N = 1.0e-7
NEWTON_ITERATIONS_BOUND = 60


def carlson_rf(x: float, y: float, z: float) -> float:
    """Carlson对称形式``R_F(x,y,z)``，重复化算法（Carlson 1979）。"""

    for _ in range(200):
        root_x, root_y, root_z = math.sqrt(x), math.sqrt(y), math.sqrt(z)
        lam = root_x * (root_y + root_z) + root_y * root_z
        x, y, z = 0.25 * (x + lam), 0.25 * (y + lam), 0.25 * (z + lam)
        average = (x + y + z) / 3.0
        dx, dy, dz = (average - x) / average, (average - y) / average, (average - z) / average
        if max(abs(dx), abs(dy), abs(dz)) < 0.0008:
            break
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    return (1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0) / math.sqrt(average)


def carlson_rd(x: float, y: float, z: float) -> float:
    """Carlson对称形式``R_D(x,y,z)``，重复化算法。"""

    total = 0.0
    factor = 1.0
    for _ in range(200):
        root_x, root_y, root_z = math.sqrt(x), math.sqrt(y), math.sqrt(z)
        lam = root_x * (root_y + root_z) + root_y * root_z
        total += factor / (root_z * (z + lam))
        factor *= 0.25
        x, y, z = 0.25 * (x + lam), 0.25 * (y + lam), 0.25 * (z + lam)
        average = 0.2 * (x + y + 3.0 * z)
        dx, dy, dz = (average - x) / average, (average - y) / average, (average - z) / average
        if max(abs(dx), abs(dy), abs(dz)) < 0.0004:
            break
    ea, eb = dx * dy, dz * dz
    ec, ed = ea - eb, ea - 6.0 * eb
    ee = ed + ec + ec
    c1, c2, c3, c4 = 3.0 / 14.0, 1.0 / 6.0, 9.0 / 22.0, 3.0 / 26.0
    tail = (1.0 + ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * dz * ee)
            + dz * (c2 * ee + dz * (-c3 * ec + dz * c4 * ea)))
    return 3.0 * total + factor * tail / (average * math.sqrt(average))


def incomplete_f(phi: float, m: float) -> float:
    sine = math.sin(phi)
    return sine * carlson_rf(math.cos(phi) ** 2, 1.0 - m * sine * sine, 1.0)


def incomplete_e(phi: float, m: float) -> float:
    sine = math.sin(phi)
    cosine_sq = math.cos(phi) ** 2
    delta = 1.0 - m * sine * sine
    return (sine * carlson_rf(cosine_sq, delta, 1.0)
            - (m / 3.0) * sine ** 3 * carlson_rd(cosine_sq, delta, 1.0))


def complete_k(m: float) -> float:
    return carlson_rf(0.0, 1.0 - m, 1.0)


def complete_e(m: float) -> float:
    return carlson_rf(0.0, 1.0 - m, 1.0) - (m / 3.0) * carlson_rd(0.0, 1.0 - m, 1.0)


def elastica_tip(beta: float) -> tuple[float, float, float]:
    """返回``(x_L/L, y_L/L, α)``。``beta = P·L²/EI``。"""

    def arclength(alpha: float) -> float:
        m = 0.5 * (1.0 + math.sin(alpha))
        phi1 = math.asin(math.sqrt(1.0 / (1.0 + math.sin(alpha))))
        return complete_k(m) - incomplete_f(phi1, m)

    target = math.sqrt(beta)
    low, high = 1.0e-13, 0.5 * math.pi - 1.0e-13
    for _ in range(200):  # 二分：arclength对α严格单调增
        middle = 0.5 * (low + high)
        if arclength(middle) < target:
            low = middle
        else:
            high = middle
    alpha = 0.5 * (low + high)
    m = 0.5 * (1.0 + math.sin(alpha))
    phi1 = math.asin(math.sqrt(1.0 / (1.0 + math.sin(alpha))))
    return (
        math.sqrt(2.0 * math.sin(alpha) / beta),
        1.0 - (2.0 / math.sqrt(beta)) * (complete_e(m) - incomplete_e(phi1, m)),
        alpha,
    )


def self_check() -> None:
    """三条自校验；任何一条不过就不出金标。"""

    for m, known_k, known_e in (
        (0.25, 1.6857503548125960, 1.4674622093394272),
        (0.50, 1.8540746773013719, 1.3506438810476755),
    ):
        for label, value, known in (("K", complete_k(m), known_k), ("E", complete_e(m), known_e)):
            if abs(value - known) / known > 1.0e-15:
                raise SystemExit(f"{label}(m={m})={value!r} 偏离已知值{known!r}")

    # 小载荷极限：不只断言"趋近线性悬臂"，断言**第一修正恰为β的二次**。
    # 后者强得多——它同时验了闭式的形状而不只是它的一个端点值，
    # 且不需要在测试里写下任何系数（β十倍则偏差百倍，这条与实现无关）。
    deviations = []
    for beta in (1.0e-3, 1.0e-2, 1.0e-1):
        _, y, alpha = elastica_tip(beta)
        deviations.append((y / (beta / 3.0) - 1.0, alpha / (beta / 2.0) - 1.0))
    if not all(deviation < 0.0 for pair in deviations for deviation in pair):
        raise SystemExit(f"小载荷修正的符号不对（应当是软化）：{deviations!r}")
    for index in range(2):  # 0=挠度、1=端部转角
        for step in range(2):
            ratio = deviations[step + 1][index] / deviations[step][index]
            if not (99.0 <= ratio <= 101.0):
                raise SystemExit(
                    f"小载荷修正不是二阶：β十倍而偏差{ratio!r}倍（应当约100）"
                )

    previous = elastica_tip(0.1)
    for beta in (0.5, 1.0, 2.0, 4.0, 8.0):
        current = elastica_tip(beta)
        if not (current[1] > previous[1] and current[0] < previous[0]):
            raise SystemExit(f"β={beta}处单调性破了：{previous} → {current}")
        previous = current


def main() -> int:
    self_check()
    tip_x, tip_y, alpha = elastica_tip(LOAD_PARAMETER)
    # 小挠度理论在同一载荷下的预言：δ = PL³/(3EI) 即 y/L = β/3，且不缩短（x/L = 1）。
    linear_error = math.hypot(1.0 - tip_x, 1.0 - tip_y) / math.hypot(tip_x, tip_y)
    oracles = [{
        "id": "oracle:large_deflection_cantilever/elliptic_integral_tip",
        "inputs": {
            "kind": "transverse_tip_load_elastica",
            "length_mm": LENGTH_MM,
            "bending_stiffness_nmm2": BENDING_STIFFNESS_NMM2,
            "axial_stiffness_n": AXIAL_STIFFNESS_N,
            "load_parameter": LOAD_PARAMETER,
            "tip_force_n": LOAD_PARAMETER * BENDING_STIFFNESS_NMM2 / LENGTH_MM**2,
            "gravity_mm_s2": GRAVITY_MM_S2,
            "parasitic_mass_kg": PARASITIC_MASS_KG,
            "refinements": list(REFINEMENTS),
            "load_steps": LOAD_STEPS,
            "residual_tol_n": RESIDUAL_TOL_N,
            "newton_iterations_bound": NEWTON_ITERATIONS_BOUND,
        },
        "expected": {
            "tip_x_over_length": tip_x,
            "tip_y_over_length": tip_y,
            "tip_angle_rad": alpha,
            "tip_relative_error_max": 1.0e-3,
            "error_ratio_low": 3.9,
            "error_ratio_high": 4.1,
            "linear_theory_relative_error": linear_error,
            "geometric_nonlinearity_margin": 100.0,
            "all_refinements_converged": True,
            "newton_iterations_within_bound": True,
        },
        "tolerances": {
            "tip_x_over_length": {"abs": 0.0, "rel": 1.0e-14,
                                  "reason": "x_L/L = sqrt(2·sinα/β)是**初等闭式**，"
                                            "唯一的数值步骤是求α的二分（200次，已到机器精度）。"
                                            "1e-14是与WDS独立正交求积对拍的实测偏差量级"},
            "tip_y_over_length": {"abs": 0.0, "rel": 1.0e-14,
                                  "reason": "y_L/L要两个椭圆积分之差；Carlson重复化在此处的"
                                            "截断由ERRTOL=8e-4定，实测对已知K/E值偏差<1e-15，"
                                            "对WDS的QUADPACK路线偏差1.1e-15"},
            "tip_angle_rad": {"abs": 0.0, "rel": 1.0e-14,
                              "reason": "二分200次的α；与WDS独立路线偏差6.9e-16"},
            "tip_relative_error_max": {"abs": 0.0, "rel": 0.0,
                                       "reason": "**上界，不是等式**。最细档(n=80)实测3.234393e-04；"
                                                 "取1e-3留3.1倍余量，同时比固支权取h时的一阶值"
                                                 "8.804656e-03低8.8倍——门仍能分开二阶与一阶"},
            "error_ratio_low": {"abs": 0.0, "rel": 0.0, "reason": "区间下界，零容差比较（非数值判据）"},
            "error_ratio_high": {"abs": 0.0, "rel": 0.0,
                                 "reason": "区间上界。**不写死为4**——4是渐近值（spec/12第4.3节：比阶不比单点）。"
                                           "实测三档3.9612/4.0512/4.0600；n=5那一档比值只有3.6422，"
                                           "即h=20mm还没进渐近区，所以加密阶梯从n=10起"},
            "linear_theory_relative_error": {"abs": 0.0, "rel": 1.0e-14,
                                             "reason": "由闭式直接算：小挠度理论在β=3给(x/L,y/L)=(1, 1)，"
                                                       "对闭式(0.7456, 0.6033)偏差49.1%。"
                                                       "**这个数存在的意义是证明本案例真的在考几何非线性**"},
            "geometric_nonlinearity_margin": {"abs": 0.0, "rel": 0.0,
                                              "reason": "几何精确项的误差必须比小挠度理论小至少100倍。"
                                                        "实测倍数1519×（4.914e-01 / 3.234e-04）。"
                                                        "取100倍是留余量的下界，不是实测值——"
                                                        "写实测值等于让门迁就当前实现"},
            "all_refinements_converged": {"abs": 0.0, "rel": 0.0,
                                          "reason": "布尔前置断言：**不收敛的解不许参与收敛阶比较**——"
                                                    "那会拿一个没解出来的数去算阶"},
            "newton_iterations_within_bound": {"abs": 0.0, "rel": 0.0,
                                               "reason": "单个载荷步的牛顿迭代数上界。实测最大48步。"
                                                         "**能量不再是二次型，所以这里没有「一步收敛」**"
                                                         "（对比cases/cantilever_self_weight）；"
                                                         "60是留25%余量的上界，迭代数突增是"
                                                         "线搜索在几何非线性上失步的信号"},
        },
    }]
    document = {
        "facet": "engine_oracle_manifest", "facet_version": "0.1",
        "case_id": "case/large_deflection_cantilever", "load_tier": "local_batch",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/large_deflection_cantilever/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles, "arrays": {}, "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
