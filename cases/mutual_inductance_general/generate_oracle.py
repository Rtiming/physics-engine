#!/usr/bin/env python3
"""一般位形互感的金标生成器——**三条独立路径，都不调被验内核**。

被验内核（`physics_engine.electromagnetics.neumann`）走的是
**Neumann双回路线积分的二维中点切元求积**。本生成器走三条不同的路：

1. **同轴构型的金标**：**约化Neumann单重积分**（全正被积函数）。
   共轴对称性把双重线积分解析地约化成单重，再改写成一条恒正的被积函数
   （细节在`reduced_neumann_coaxial_h`）。**这条约化用掉的正是被验内核
   一点都用不上的那条对称性**，所以两侧除了"都叫Neumann"之外没有共同之处。
   原样复用`cases/mutual_inductance_coaxial/generate_oracle.py`那一份。

   **另有一条Maxwell闭式（Carlson对称形式）只当交叉验证不当金标**，
   理由是一条实测：本生成器按教科书分组算方括号`(2/k-k)K-(2/k)E`，
   它在小k（远场）下相消放大约`1/k^4`，(0.01,0.01,0.2)那一组实测差6.6e-12，
   做不了1e-13的金标。**这条相消正是`electromagnetics/elliptic.py`第三节
   整节在处理的东西**——本生成器把它又撞了一次，记在这里。

2. **一般位形的金标**：**Biot-Savart场 + 圆盘面磁通**

       M = Phi_2 / I_1 = (1/I_1) int_{S_2} B_1 . n_2 dA

   `B_1`由Biot-Savart线积分给出，`S_2`取回路2张成的**平面圆盘**。
   这不是"同一个积分换一种求积"——**它是另一个物理表述**：
   Neumann公式积的是两条线，本路积的是一条线加一张面，
   中间要过一次旋度（Biot-Savart由A的旋度得来）。两者相等本身就是一条定理。

   求积：源线积分用**周期中点法**（被积函数解析且2pi周期，几何收敛）；
   圆盘用**极坐标**，角向周期中点法、径向**Gauss-Legendre**
   （径向不是周期方向，GL对解析被积函数同样几何收敛）。
   Legendre节点用Newton迭代现算，零依赖。

   **这条路的适用边界**：圆盘不许被源回路穿过（否则那不是一张合法的
   张成曲面，磁通与链数不再相等），且源线不许贴着圆盘（近奇异）。
   本生成器选的构型全部远离这两条，并且**用第1条路在同轴构型上交叉验证**
   ——两条路对不上就拒绝落盘。

3. **远场参照**：一般位形的偶极近似

       M ~ (mu0 A1 A2 / (4 pi d^3)) [3 (n1.dhat)(n2.dhat) - n1.n2]

   共轴时方括号=2，退化成`cases/mutual_inductance_coaxial`里那条。
   本生成器独立写一遍（它是判据里的参照，不在主路上——
   一条只出现在判据里的公式同样会写错）。

mu0取CODATA 2022推荐值（**2019 SI重定义之后不再是按定义精确的4pi*1e-7**）；
这个字面量与`units.VACUUM_PERMEABILITY_H_PER_M`是两份独立抄写。

参考解出处：Neumann 1845（互感的双回路线积分）；
Maxwell《A Treatise on Electricity and Magnetism》vol.2 sect.701；
Grover《Inductance Calculations》ch.13；Carlson 1979（`R_F`/`R_D`重复化）；
偶极-偶极互感式见Jackson《Classical Electrodynamics》3rd ed. sect.5.6
（两个磁偶极子的相互作用能，除以I1 I2即互感）。

本生成器不import `physics_engine.electromagnetics`；只用`oracles`写清单。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/mutual_inductance_general"
ALGORITHM_VERSION = "1.0.0"

#: 真空磁导率（CODATA 2022，H/m）。**不是按定义精确**。
MU0_H_PER_M = 1.25663706127e-6

#: Biot-Savart面积分的两档网格`(源线节点, 径向GL节点, 角向节点)`。
#: 两档互差超过地板即拒绝落盘——金标不许带未申报的不确定度。
FLUX_GRID = (192, 32, 64)
FLUX_SELF_CHECK_GRID = (128, 24, 48)
#: **两档都是标定过的，不是随手写的**：(96,16,32)那一档实测两档相对差2.97e-12，
#: 过不了下面这条地板——生成器当场拒绝落盘。粗档因此不能再粗。
FLUX_FLOOR = 1.0e-13

#: 同轴构型上两条数值路径（约化Neumann单重积分 / Biot-Savart面积分）的交叉验证地板。
CROSS_CHECK_FLOOR = 1.0e-13

#: **Maxwell闭式那一路只当交叉验证，不当金标**，地板因此松得多。理由是实测：
#: 本生成器按教科书分组算方括号`(2/k-k)K-(2/k)E`，它在小k下相消放大约`1/k^4`
#: （`electromagnetics/elliptic.py`第三节有全表）。(0.01, 0.01, 0.2)那一组k=0.0995，
#: 实测与另外两条路差**6.6e-12**——与该表在k=0.1处的8.9e-12同量级。
#: **这不是本案例的缺陷而是它的一条实测**：教科书分组在远场做不了1e-13的金标，
#: 所以金标那一侧走的是全正被积函数的约化Neumann积分。
CLOSED_FORM_CROSS_CHECK_FLOOR = 1.0e-10

#: 约化Neumann单重积分的节点数与自校验档（形制与`cases/mutual_inductance_coaxial`同）。
REDUCED_NODES = 8192
REDUCED_SELF_CHECK_NODES = 2048
REDUCED_FLOOR = 1.0e-14

#: 判据统一采用的分段数。选96的理由：本案例全部构型在此处的分辨比δ/h ≥ 5，
#: 而标定表（模块常量`RESOLUTION_RATIO_CALIBRATION`）在δ/h ≥ 4已到1e-11以下。
PRODUCTION_SEGMENTS = 96

# ---------------------------------------------------------------------------
# 构型
# ---------------------------------------------------------------------------

#: 第一层：同轴退化。`(r1_m, r2_m, d_m)`——每一组都能用Maxwell闭式独立求出。
COAXIAL_CONFIGURATIONS: tuple[tuple[float, float, float], ...] = (
    (0.050, 0.050, 0.030),   # 等径近距，k=0.9578
    (0.100, 0.020, 0.050),   # 半径比5:1
    (0.010, 0.010, 0.200),   # 远场，条件数kappa约5.1e2
    (0.030, 0.030, 0.010),   # 最近的一组：N=96时delta/h只有5.09
)

#: 第二层：几何收敛。取最近的那一组（收敛最慢、可用档位最多）。
CONVERGENCE_CONFIGURATION = (0.030, 0.030, 0.010)
CONVERGENCE_SEGMENTS: tuple[int, ...] = (40, 48, 56, 64, 80)
#: 误差包络：各档实测值的**2倍**（见案例页第三节的实测表）。
CONVERGENCE_ENVELOPE: tuple[float, ...] = (
    2.0e-6, 1.3e-7, 8.4e-9, 5.5e-10, 2.5e-12,
)
#: 每加密8段的log2误差比下限（跨16段的档位归一到8段）。
#: 实测3.9599 / 3.9399 / 3.9253 / 3.9093，即每加8段误差降15.4倍。
#: **同一构型同一批档位上，折线弦（多边形）离散实测只有
#: 0.5253 / 0.4449 / 0.3854 / 0.3220**——与代数二阶的理论预测
#: 0.5261 / 0.4448 / 0.3853 / 0.3219逐位吻合。取3.5作下限把两者分得干干净净。
CONVERGENCE_LOG2_RATIO_FLOOR = 3.5
#: 同一批比值的最大/最小之比。几何收敛下它接近1（实测**1.0129**）；
#: 代数阶方法的比值一路衰减，折线弦那一版实测**1.6315**。
CONVERGENCE_BAND_MAX = 1.10
#: 为什么最细一档停在80而不是96：N=96的相对误差5.68e-15只有浮点地板的十几倍，
#: 换一个libm就可能翻倍，而它一翻倍最后那个log2比就掉到3.4——**门会变得看运气**。
#: 收敛判据的档位必须全部落在**误差远高于地板**的区间里，这条是实测撞出来的。

#: 第三层：一般位形。`(名字, r_a, centre_a, normal_a, r_b, centre_b, normal_b)`。
_TILT30 = (0.0, math.sin(math.radians(30.0)), math.cos(math.radians(30.0)))
_TILT60 = (math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0)))
GENERAL_CONFIGURATIONS: tuple[tuple[str, float, tuple[float, float, float], tuple[float, float, float], float, tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("tilt_30_on_axis", 0.050, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
     0.030, (0.0, 0.0, 0.080), _TILT30),
    ("offset_parallel", 0.050, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
     0.040, (0.030, 0.020, 0.060), (0.0, 0.0, 1.0)),
    ("tilt_60_offset", 0.050, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
     0.020, (0.025, -0.015, 0.045), _TILT60),
    ("coplanar_side_by_side", 0.010, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
     0.020, (0.200, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("skew_normal_close", 0.050, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
     0.021, (0.013, -0.007, 0.044), (0.3, -0.5, 0.81)),
)

#: 第三层附带的**零磁通**构型：回路b的法向垂直于z轴、圆心落在z轴上。
#: 源场对z轴轴对称，圆盘落在x=0平面上而B_x在该平面上按镜像对称恒为零——
#: **M按对称性精确为零**，不是"很小"。
PERPENDICULAR_CONFIGURATION = (
    0.050, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    0.030, (0.0, 0.0, 0.070), (1.0, 0.0, 0.0),
)
PERPENDICULAR_SEGMENT_PAIRS: tuple[tuple[int, int], ...] = (
    (64, 64), (96, 96), (128, 96), (37, 53),
)

#: 第四层：互易。两组，一组带不同匝数，**两条回路的分段数故意不同**——
#: 分段数写反是"角色写反"里最难看出来的一种。
RECIPROCITY_CONFIGURATIONS: tuple[tuple[float, tuple[float, float, float], tuple[float, float, float], float, tuple[float, float, float], tuple[float, float, float], int, int, int, int], ...] = (
    (0.050, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
     0.021, (0.013, -0.007, 0.044), (0.3, -0.5, 0.81), 1, 1, 96, 61),
    (0.050, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
     0.030, (0.0, 0.0, 0.080), _TILT30, 3, 7, 71, 53),
)

#: 第五层：远场退化。三族取向 × 五档距离（倍频程）。
FAR_FIELD_RADII = (0.010, 0.020)
FAR_FIELD_SEPARATIONS_M: tuple[float, ...] = (0.2, 0.4, 0.8, 1.6, 3.2)
FAR_FIELD_SEGMENTS = 96
FAR_FIELD_FAMILIES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("coaxial", (0.0, 0.0, 1.0)),
    ("coplanar", (0.0, 0.0, 1.0)),
    ("tilt_60", _TILT60),
)

#: 第六层：奇异性与分辨。等径同轴、**同一个分段数**时样本点按角对齐，
#: 于是最小样本距离恰为d，分辨比有闭式`d N / (2 pi r)`——
#: 这一格因此能被独立算出来，不必复述实现的采样。
RESOLUTION_PROBE = (0.050, 0.050, 0.020)
RESOLUTION_PROBE_SEGMENTS = 64
#: 拒跑档：同一构型分段数取24，分辨比落到1.528（2以下）。
#: **不取32**——32那一档的分辨比是2.037，恰好在门的合法一侧，
#: 选它做拒跑档等于没验到门。这条差别是写判据时实测撞出来的。
RESOLUTION_REFUSAL_SEGMENTS = 24

#: 第八层：有截面的线圈（S2.4）。两条同轴矩形截面线圈，`(半径, 轴向位置, 径向全宽, 轴向全高)`。
BUNDLE_COIL_A = (0.050, 0.000, 0.010, 0.008)
BUNDLE_COIL_B = (0.030, 0.040, 0.006, 0.005)
BUNDLE_SEGMENTS = 24
#: 截面细分档位。**不取8**：grid=8是4096对细丝、实测一次1.1秒，
#: 而它在收敛表上只是第四个点——interactive档买不起（案例页第五节记了那个数）。
BUNDLE_GRIDS: tuple[int, ...] = (1, 2, 4)
#: 各档相对误差的包络，取实测的2倍：1.64463e-3 / 4.13424e-4 / 1.03504e-4。
BUNDLE_ENVELOPE: tuple[float, ...] = (3.3e-3, 8.3e-4, 2.1e-4)
#: 收敛阶的区间。二维中点求积是**代数二阶**，实测1.99207 / 1.99794。
BUNDLE_ORDER_BRACKET = (1.9, 2.1)
#: 细丝极限：截面按比例缩小，偏差按s^2降。实测2.00075 / 2.00022 / 2.00006。
BUNDLE_LIMIT_GRID = 2
BUNDLE_LIMIT_SCALES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
#: 四维Gauss-Legendre参考的节点数与自校验档（每维）。
BUNDLE_REFERENCE_NODES = 12
BUNDLE_REFERENCE_SELF_CHECK_NODES = 8
BUNDLE_REFERENCE_FLOOR = 1.0e-13
#: 互易与匝数档：两条线圈的截面细分与分段数**都不同**。
BUNDLE_RECIPROCITY_GRIDS = (4, 3)
BUNDLE_RECIPROCITY_SEGMENTS = (48, 31)
BUNDLE_TURNS = (3, 7)

#: 第七层：单位边界。mm制声明与米制声明必须给出逐位相同的M。
UNIT_RADIUS_A_MM = 50.0
UNIT_RADIUS_B_MM = 30.0
UNIT_CENTRE_B_MM = (10.0, -20.0, 80.0)
UNIT_NORMAL_B = _TILT30
UNIT_SEGMENTS = 96


# ---------------------------------------------------------------------------
# 路径一：Maxwell闭式（Carlson对称形式）
# ---------------------------------------------------------------------------


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


def reduced_neumann_coaxial_h(r1: float, r2: float, d: float, nodes: int) -> float:
    """同轴圆环互感的**约化Neumann单重积分**（H），全正被积函数。

    共轴时按角差`phi`把双重线积分化成单重：

        M = (mu0 r1 r2 / 2) int_0^{2pi} cos(phi) d(phi)
            / sqrt(r1^2 + r2^2 + d^2 - 2 r1 r2 cos(phi))

    再令`c = r1^2+r2^2+d^2`、`q = 2 r1 r2/c`、`s = sqrt(1 - q cos phi)`，
    由`1/s - 1 = q cos(phi)/(s(1+s))`且`int cos(phi) d(phi) = 0`得

        M = (mu0 r1 r2/2)(q/sqrt(c)) int_0^{2pi} cos^2(phi)/(s(1+s)) d(phi)

    **整条被积函数恒正、一次减法都没有**——远场不会被相消吃掉
    （直接写第一式在d=1m处实测已有1.1e-8的相对误差）。

    **原样复用`cases/mutual_inductance_coaxial/generate_oracle.py`那一份**：
    生成器之间复用正当，两侧与被验内核（一般位形的二维双重求积）仍然无共用代码，
    而且**这里的约化用掉了共轴对称性，被验内核一点都用不上它**。
    """

    squared_sum = r1 * r1 + r2 * r2 + d * d
    ratio = 2.0 * r1 * r2 / squared_sum
    terms = []
    for index in range(nodes):
        angle = 2.0 * math.pi * (index + 0.5) / nodes
        cosine = math.cos(angle)
        root = math.sqrt(1.0 - ratio * cosine)
        terms.append(cosine * cosine / (root * (1.0 + root)))
    integral = math.fsum(terms) * (2.0 * math.pi / nodes) * ratio / math.sqrt(squared_sum)
    return MU0_H_PER_M * r1 * r2 / 2.0 * integral


def converged_reduced_neumann_h(r1: float, r2: float, d: float) -> float:
    """两档求值互相印证；不收敛即拒绝落盘。"""

    coarse = reduced_neumann_coaxial_h(r1, r2, d, REDUCED_SELF_CHECK_NODES)
    fine = reduced_neumann_coaxial_h(r1, r2, d, REDUCED_NODES)
    if abs(coarse - fine) > REDUCED_FLOOR * abs(fine):
        raise SystemExit(
            f"约化Neumann积分未收敛：M({r1}, {r2}, {d})两档相对差"
            f"{abs(coarse - fine) / abs(fine)!r} > {REDUCED_FLOOR!r}，不落盘"
        )
    return fine


def maxwell_coaxial_mutual_h(r1: float, r2: float, d: float) -> float:
    """同轴圆环互感的Maxwell闭式（H）。**收模k，内部换算成参数m = k^2**。"""

    modulus_squared = 4.0 * r1 * r2 / ((r1 + r2) ** 2 + d * d)
    modulus = math.sqrt(modulus_squared)
    complete_k = carlson_rf(0.0, 1.0 - modulus_squared, 1.0)
    complete_e = complete_k - (modulus_squared / 3.0) * carlson_rd(
        0.0, 1.0 - modulus_squared, 1.0
    )
    bracket = (2.0 / modulus - modulus) * complete_k - (2.0 / modulus) * complete_e
    return MU0_H_PER_M * math.sqrt(r1 * r2) * bracket


# ---------------------------------------------------------------------------
# 路径二：Biot-Savart场 + 圆盘面磁通
# ---------------------------------------------------------------------------


def legendre_nodes(count: int) -> tuple[list[float], list[float]]:
    """Gauss-Legendre节点与权重（区间[-1,1]），Newton迭代，零依赖。

    初值取`cos(pi (i+0.75)/(n+0.5))`（Press等的经典近似），
    递推算`P_n`与`P_n'`；实测n<=64时迭代4次内收敛到1e-16。
    """

    nodes: list[float] = []
    weights: list[float] = []
    for index in range(count):
        x = math.cos(math.pi * (index + 0.75) / (count + 0.5))
        derivative = 0.0
        for _ in range(100):
            value, previous = 1.0, 0.0
            for degree in range(count):
                value, previous = (
                    ((2.0 * degree + 1.0) * x * value - degree * previous) / (degree + 1.0),
                    value,
                )
            derivative = count * (x * value - previous) / (x * x - 1.0)
            step = -value / derivative
            x += step
            if abs(step) < 1.0e-16:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * derivative * derivative))
    return nodes, weights


def plane_frame(normal: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """单位法向与平面内一组正交基。**本生成器自己写一遍**，不看内核怎么写的。"""

    norm = math.sqrt(sum(component * component for component in normal))
    unit = tuple(component / norm for component in normal)
    seed = (1.0, 0.0, 0.0)
    if abs(unit[0]) > abs(unit[1]):
        seed = (0.0, 1.0, 0.0)
    first = (
        seed[1] * unit[2] - seed[2] * unit[1],
        seed[2] * unit[0] - seed[0] * unit[2],
        seed[0] * unit[1] - seed[1] * unit[0],
    )
    length = math.sqrt(sum(component * component for component in first))
    first = tuple(component / length for component in first)
    second = (
        unit[1] * first[2] - unit[2] * first[1],
        unit[2] * first[0] - unit[0] * first[2],
        unit[0] * first[1] - unit[1] * first[0],
    )
    return unit, first, second  # type: ignore[return-value]


def flux_mutual_h(
    radius_a: float,
    centre_a: tuple[float, float, float],
    normal_a: tuple[float, float, float],
    radius_b: float,
    centre_b: tuple[float, float, float],
    normal_b: tuple[float, float, float],
    grid: tuple[int, int, int],
) -> float:
    """回路a载单位电流时穿过回路b圆盘的磁通（=互感，H）。"""

    line_count, radial_count, angular_count = grid
    _, source_u, source_v = plane_frame(normal_a)
    target_n, target_u, target_v = plane_frame(normal_b)
    arc = 2.0 * math.pi * radius_a / line_count
    source: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for index in range(line_count):
        angle = 2.0 * math.pi * (index + 0.5) / line_count
        cosine, sine = math.cos(angle), math.sin(angle)
        position = (
            centre_a[0] + radius_a * (source_u[0] * cosine + source_v[0] * sine),
            centre_a[1] + radius_a * (source_u[1] * cosine + source_v[1] * sine),
            centre_a[2] + radius_a * (source_u[2] * cosine + source_v[2] * sine),
        )
        element = (
            arc * (-source_u[0] * sine + source_v[0] * cosine),
            arc * (-source_u[1] * sine + source_v[1] * cosine),
            arc * (-source_u[2] * sine + source_v[2] * cosine),
        )
        source.append((position, element))
    abscissas, gauss_weights = legendre_nodes(radial_count)
    angular_weight = 2.0 * math.pi / angular_count
    terms: list[float] = []
    for abscissa, gauss_weight in zip(abscissas, gauss_weights, strict=True):
        radius = 0.5 * radius_b * (abscissa + 1.0)
        radial_weight = 0.5 * radius_b * gauss_weight * radius
        for angular_index in range(angular_count):
            angle = 2.0 * math.pi * (angular_index + 0.5) / angular_count
            cosine, sine = math.cos(angle), math.sin(angle)
            field_x = centre_b[0] + radius * (target_u[0] * cosine + target_v[0] * sine)
            field_y = centre_b[1] + radius * (target_u[1] * cosine + target_v[1] * sine)
            field_z = centre_b[2] + radius * (target_u[2] * cosine + target_v[2] * sine)
            b_x = b_y = b_z = 0.0
            for position, element in source:
                offset_x = field_x - position[0]
                offset_y = field_y - position[1]
                offset_z = field_z - position[2]
                squared = offset_x * offset_x + offset_y * offset_y + offset_z * offset_z
                inverse = 1.0 / (squared * math.sqrt(squared))
                b_x += (element[1] * offset_z - element[2] * offset_y) * inverse
                b_y += (element[2] * offset_x - element[0] * offset_z) * inverse
                b_z += (element[0] * offset_y - element[1] * offset_x) * inverse
            terms.append(
                radial_weight
                * angular_weight
                * (b_x * target_n[0] + b_y * target_n[1] + b_z * target_n[2])
            )
    return MU0_H_PER_M / (4.0 * math.pi) * math.fsum(terms)


def converged_flux_mutual_h(
    radius_a: float,
    centre_a: tuple[float, float, float],
    normal_a: tuple[float, float, float],
    radius_b: float,
    centre_b: tuple[float, float, float],
    normal_b: tuple[float, float, float],
) -> float:
    """两档面积分互相印证；不收敛即拒绝落盘。"""

    coarse = flux_mutual_h(
        radius_a, centre_a, normal_a, radius_b, centre_b, normal_b, FLUX_SELF_CHECK_GRID
    )
    fine = flux_mutual_h(
        radius_a, centre_a, normal_a, radius_b, centre_b, normal_b, FLUX_GRID
    )
    if abs(coarse - fine) > FLUX_FLOOR * abs(fine):
        raise SystemExit(
            f"Biot-Savart面积分未收敛：两档相对差{abs(coarse - fine) / abs(fine)!r}"
            f" > {FLUX_FLOOR!r}，不落盘"
        )
    return fine


# ---------------------------------------------------------------------------
# 路径三：一般位形的偶极近似（判据参照，不在主路上）
# ---------------------------------------------------------------------------


def dipole_general_h(
    radius_a: float,
    centre_a: tuple[float, float, float],
    normal_a: tuple[float, float, float],
    radius_b: float,
    centre_b: tuple[float, float, float],
    normal_b: tuple[float, float, float],
    turns_a: int = 1,
    turns_b: int = 1,
) -> float:
    """``M ~ (mu0 A1 A2 / (4 pi d^3)) [3 (n1.dhat)(n2.dhat) - n1.n2]``。"""

    unit_a = plane_frame(normal_a)[0]
    unit_b = plane_frame(normal_b)[0]
    offset = tuple(centre_b[axis] - centre_a[axis] for axis in range(3))
    distance = math.sqrt(sum(component * component for component in offset))
    direction = tuple(component / distance for component in offset)
    area_a = turns_a * math.pi * radius_a * radius_a
    area_b = turns_b * math.pi * radius_b * radius_b
    projection_a = sum(unit_a[axis] * direction[axis] for axis in range(3))
    projection_b = sum(unit_b[axis] * direction[axis] for axis in range(3))
    alignment = sum(unit_a[axis] * unit_b[axis] for axis in range(3))
    angular = 3.0 * projection_a * projection_b - alignment
    return MU0_H_PER_M * area_a * area_b * angular / (4.0 * math.pi * distance**3)


def section_averaged_maxwell_h(
    radius_a: float,
    axial_a: float,
    radial_extent_a: float,
    axial_extent_a: float,
    radius_b: float,
    axial_b: float,
    radial_extent_b: float,
    axial_extent_b: float,
    nodes: int,
) -> float:
    """两条同轴矩形截面线圈的互感：**对两个截面各做二维Gauss-Legendre，核用Maxwell闭式**。

    这是被验内核那一侧（截面二维中点求积 + Neumann双重求积）的**双重独立**：
    求积法不同（GL对中点）、核不同（椭圆积分闭式对二维线积分求和）。

    四维张量积`nodes^4`次核求值。这里的k落在0.8—0.9，
    离小k那条相消区很远，所以闭式在**这一格**是可以当金标的
    （与第一层不同——那里有一组k=0.0995的远场构型）。
    """

    abscissas, weights = legendre_nodes(nodes)
    terms = []
    for index_a, abscissa_a in enumerate(abscissas):
        source_radius = radius_a + 0.5 * radial_extent_a * abscissa_a
        for index_za, abscissa_za in enumerate(abscissas):
            source_axial = axial_a + 0.5 * axial_extent_a * abscissa_za
            weight_a = weights[index_a] * weights[index_za]
            for index_b, abscissa_b in enumerate(abscissas):
                target_radius = radius_b + 0.5 * radial_extent_b * abscissa_b
                for index_zb, abscissa_zb in enumerate(abscissas):
                    target_axial = axial_b + 0.5 * axial_extent_b * abscissa_zb
                    weight = weight_a * weights[index_b] * weights[index_zb]
                    terms.append(
                        weight
                        * maxwell_coaxial_mutual_h(
                            source_radius,
                            target_radius,
                            abs(target_axial - source_axial),
                        )
                    )
    # 四个方向各除以2（GL权重之和是2），即对两个截面取**平均**而不是积分。
    return math.fsum(terms) / 16.0


def far_field_placement(
    family: str, normal: tuple[float, float, float], separation: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """三族取向各自的第二条回路位置与法向。"""

    if family == "coaxial":
        return (0.0, 0.0, separation), normal
    if family == "coplanar":
        return (separation, 0.0, 0.0), normal
    if family == "tilt_60":
        return (0.3 * separation, 0.0, separation), normal
    raise SystemExit(f"未知的远场族：{family!r}")


# ---------------------------------------------------------------------------


def main() -> int:
    # --- 生成器自校验：三条独立路径在同轴构型上必须对上 ---
    #     金标那一路是约化Neumann单重积分（全正被积函数）；
    #     Biot-Savart面积分与Maxwell闭式各当一次交叉验证。
    flux_cross_check = []
    closed_form_cross_check = []
    coaxial_values = []
    for radius_a, radius_b, separation in COAXIAL_CONFIGURATIONS:
        reference = converged_reduced_neumann_h(radius_a, radius_b, separation)
        coaxial_values.append(reference)
        by_flux = converged_flux_mutual_h(
            radius_a, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            radius_b, (0.0, 0.0, separation), (0.0, 0.0, 1.0),
        )
        deviation = abs(reference - by_flux) / abs(reference)
        if deviation > CROSS_CHECK_FLOOR:
            raise SystemExit(
                f"约化Neumann与Biot-Savart在({radius_a}, {radius_b}, {separation})上"
                f"相对差{deviation!r} > {CROSS_CHECK_FLOOR!r}——生成器自己就不自洽，不落盘"
            )
        flux_cross_check.append(deviation)
        closed_form = maxwell_coaxial_mutual_h(radius_a, radius_b, separation)
        closed_deviation = abs(reference - closed_form) / abs(reference)
        if closed_deviation > CLOSED_FORM_CROSS_CHECK_FLOOR:
            raise SystemExit(
                f"Maxwell闭式在({radius_a}, {radius_b}, {separation})上相对差"
                f"{closed_deviation!r} > {CLOSED_FORM_CROSS_CHECK_FLOOR!r}，不落盘"
            )
        closed_form_cross_check.append(closed_deviation)

    # --- 第二层：几何收敛 ---
    convergence_exact = converged_reduced_neumann_h(*CONVERGENCE_CONFIGURATION)

    # --- 第三层：一般位形 ---
    general_values = [
        converged_flux_mutual_h(*config[1:]) for config in GENERAL_CONFIGURATIONS
    ]
    perpendicular_by_flux = flux_mutual_h(*PERPENDICULAR_CONFIGURATION, FLUX_GRID)

    # --- 第五层：远场退化 ---
    far_field_deviations: list[float] = []
    far_field_orders: list[float] = []
    far_field_dipoles: list[float] = []
    radius_a, radius_b = FAR_FIELD_RADII
    for family, normal in FAR_FIELD_FAMILIES:
        family_deviations = []
        for separation in FAR_FIELD_SEPARATIONS_M:
            centre_b, normal_b = far_field_placement(family, normal, separation)
            if family == "coaxial":
                exact = converged_reduced_neumann_h(radius_a, radius_b, separation)
            else:
                exact = converged_flux_mutual_h(
                    radius_a, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                    radius_b, centre_b, normal_b,
                )
            dipole = dipole_general_h(
                radius_a, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                radius_b, centre_b, normal_b,
            )
            far_field_dipoles.append(dipole)
            family_deviations.append(exact / dipole - 1.0)
        far_field_deviations.extend(family_deviations)
        far_field_orders.extend(
            math.log2(family_deviations[index] / family_deviations[index + 1])
            for index in range(len(family_deviations) - 1)
        )

    # --- 第六层：分辨比的闭式 ---
    probe_radius, _, probe_separation = RESOLUTION_PROBE
    resolution_ratio = probe_separation * RESOLUTION_PROBE_SEGMENTS / (
        2.0 * math.pi * probe_radius
    )
    refusal_ratio = probe_separation * RESOLUTION_REFUSAL_SEGMENTS / (
        2.0 * math.pi * probe_radius
    )

    # --- 第八层：有截面的线圈 ---
    bundle_reference = section_averaged_maxwell_h(
        *BUNDLE_COIL_A, *BUNDLE_COIL_B, BUNDLE_REFERENCE_NODES
    )
    bundle_reference_coarse = section_averaged_maxwell_h(
        *BUNDLE_COIL_A, *BUNDLE_COIL_B, BUNDLE_REFERENCE_SELF_CHECK_NODES
    )
    bundle_self_check = abs(bundle_reference - bundle_reference_coarse) / abs(bundle_reference)
    if bundle_self_check > BUNDLE_REFERENCE_FLOOR:
        raise SystemExit(
            f"四维GL参考未收敛：两档相对差{bundle_self_check!r} > {BUNDLE_REFERENCE_FLOOR!r}，不落盘"
        )

    # --- 第七层：单位边界 ---
    unit_value = converged_flux_mutual_h(
        UNIT_RADIUS_A_MM / 1.0e3, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        UNIT_RADIUS_B_MM / 1.0e3,
        tuple(value / 1.0e3 for value in UNIT_CENTRE_B_MM),  # type: ignore[arg-type]
        UNIT_NORMAL_B,
    )

    oracles = [
        {
            "id": "oracle:mutual_inductance_general/coaxial_degeneration",
            "inputs": {
                "kind": "reduced_neumann_single_integral_cross_checked_three_ways",
                "configurations_r1_r2_d_m": [list(config) for config in COAXIAL_CONFIGURATIONS],
                "segments": PRODUCTION_SEGMENTS,
                "vacuum_permeability_h_per_m": MU0_H_PER_M,
                "cross_check_against_flux_integral": flux_cross_check,
                "cross_check_against_maxwell_closed_form": closed_form_cross_check,
                "why_the_closed_form_is_only_a_cross_check":
                    "本生成器按教科书分组算方括号(2/k-k)K-(2/k)E，它在小k（=远场）下"
                    "相消放大约1/k^4。(0.01,0.01,0.2)那一组k=0.0995，实测与另两条路差6.6e-12，"
                    "与elliptic.py第三节那张表在k=0.1处的8.9e-12同量级。"
                    "**所以金标走的是全正被积函数的约化Neumann单重积分**，闭式只当交叉验证",
            },
            "expected": {"mutual_inductances_h": coaxial_values},
            "tolerances": {
                "mutual_inductances_h": {
                    "abs": 0.0, "rel": 1.0e-13,
                    "reason": "一般位形的双重求积（N=96）对**共轴那一支的约化Neumann单重积分**"
                              "（另经Maxwell闭式与Biot-Savart面积分两次交叉验证）——"
                              "本案例最强的一条金标：共轴是一般位形的特例，"
                              "而共轴那一支有解析约化。四组实测2.85e-16 / 0（逐位相同）/ "
                              "1.65e-15 / **5.68e-15**。最坏那一组是(0.03,0.03,0.01)，"
                              "它的delta/h只有5.09，误差是**求积截断**；"
                              "而(0.01,0.01,0.2)那组的kappa=5.1e2、误差1.65e-15，是**相消**——"
                              "两种误差来源在同一张表上各占一组，这是选点时故意安排的。"
                              "取1e-13是实测最坏的17.6倍，另留出kappa*eps=1.1e-13的相消上界。"
                              "**这一条同时钉住`from_coaxial`那条接口**："
                              "共轴回路搬进一般位形若把轴向位置放错分量，本条当场红",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_general/geometric_convergence",
            "inputs": {
                "kind": "midpoint_tangent_quadrature_converges_geometrically",
                "configuration_r1_r2_d_m": list(CONVERGENCE_CONFIGURATION),
                "segment_counts": list(CONVERGENCE_SEGMENTS),
                "exact_mutual_inductance_h": convergence_exact,
                "relative_error_envelope": list(CONVERGENCE_ENVELOPE),
                "log2_ratio_floor_per_eight_segments": CONVERGENCE_LOG2_RATIO_FLOOR,
                "log2_ratio_band_max": CONVERGENCE_BAND_MAX,
            },
            "expected": {
                "relative_errors_fall_under_the_envelope": True,
                "log2_ratios_exceed_the_floor": True,
                "log2_ratios_stay_in_a_band": True,
            },
            "tolerances": {
                "relative_errors_fall_under_the_envelope": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：五档分段数（40/48/56/64/80）的相对误差逐档落在"
                              "声明的包络内。包络取实测值的2倍："
                              "9.9624e-7 / 6.4019e-8 / 4.1713e-9 / 2.7457e-10 / 1.2163e-12。"
                              "**为什么判包络而不判数本身**：这些误差是被验实现的属性，"
                              "生成器要独立算出它们就得把同一套求积再写一遍——"
                              "那是'抄两遍抄得一样'（轴7规则4明禁）。包络两侧都有意义："
                              "上界抓精度退化，而下一条的比值抓收敛形态",
                },
                "log2_ratios_exceed_the_floor": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：每加密8段的`log2(误差比)`都>=3.5（跨16段的档位归一到8段）。"
                              "实测3.9599/3.9399/3.9253/3.9093，即每加8段误差降15.4倍。"
                              "**这一条是'折线弦离散'的捕手**：把中点切元换成多边形弦，"
                              "误差就不在求积上而在几何上（多边形的周长与面积本身差O(1/N^2)），"
                              "收敛退化成代数二阶。同一构型同一批档位实测"
                              "0.5253/0.4449/0.3854/0.3220，与二阶理论值"
                              "0.5261/0.4448/0.3853/0.3219逐位吻合，低于下限6.7倍；"
                              "它在N=64处的相对误差是4.79e-4，而本实现是2.75e-10",
                },
                "log2_ratios_stay_in_a_band": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：同一批log2比的max/min <= 1.10（实测**1.0129**）。"
                              "**几何收敛的指纹是比值不随加密而衰减**——"
                              "`err ~ C exp(-c N)`时每加固定段数降固定倍数；"
                              "代数阶p的方法这个比是`p log2((N+8)/N)`，一路衰减，"
                              "折线弦那一版实测max/min=**1.6315**，过不了1.10。"
                              "**上一条只判下限，本条判形态**：一个碰巧很准但仍是代数阶的"
                              "实现能过上一条，过不了本条",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_general/general_position",
            "inputs": {
                "kind": "biot_savart_field_flux_through_the_spanning_disc",
                "configurations": [
                    {
                        "name": name,
                        "radius_a_m": radius_a_value,
                        "centre_a_m": list(centre_a),
                        "normal_a": list(normal_a),
                        "radius_b_m": radius_b_value,
                        "centre_b_m": list(centre_b),
                        "normal_b": list(normal_b),
                    }
                    for (name, radius_a_value, centre_a, normal_a, radius_b_value, centre_b, normal_b)
                    in GENERAL_CONFIGURATIONS
                ],
                "segments": PRODUCTION_SEGMENTS,
                "flux_grid": list(FLUX_GRID),
                "flux_self_check_grid": list(FLUX_SELF_CHECK_GRID),
                "perpendicular_configuration": {
                    "radius_a_m": PERPENDICULAR_CONFIGURATION[0],
                    "centre_a_m": list(PERPENDICULAR_CONFIGURATION[1]),
                    "normal_a": list(PERPENDICULAR_CONFIGURATION[2]),
                    "radius_b_m": PERPENDICULAR_CONFIGURATION[3],
                    "centre_b_m": list(PERPENDICULAR_CONFIGURATION[4]),
                    "normal_b": list(PERPENDICULAR_CONFIGURATION[5]),
                },
                "perpendicular_segment_pairs": [list(pair) for pair in PERPENDICULAR_SEGMENT_PAIRS],
                "perpendicular_by_flux_integral_h": perpendicular_by_flux,
            },
            "expected": {
                "mutual_inductances_h": general_values,
                "perpendicular_mutual_inductances_h": [0.0] * len(PERPENDICULAR_SEGMENT_PAIRS),
            },
            "tolerances": {
                "mutual_inductances_h": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "Neumann双重求积（N=96）对**Biot-Savart场的圆盘面磁通**——"
                              "两个不同的物理表述，不是同一积分的两种求积。"
                              "实测最坏1.94e-15（共面并排那一组，kappa=5.0e2）、"
                              "其余四组<=6.8e-16；参考侧自身两档自校验<=1.0e-15。"
                              "取1e-12是实测最坏的515倍——**留这么宽是因为参考侧的"
                              "GL节点由Newton迭代现算**，不同libm下末位会动，"
                              "而这一层要判的是'一般位形算得对不对'，不是末位复现",
                },
                "perpendicular_mutual_inductances_h": {
                    "abs": 1.0e-24, "rel": 0.0,
                    "reason": "**正交零磁通构型：M按对称性精确为零**（源场对z轴轴对称，"
                              "而圆盘落在x=0平面上、B_x在该平面按镜像对称恒为零）。"
                              "四组分段数实测|M| <= 7.3e-26，参考侧独立给出-4.9e-26。"
                              "**判绝对不判相对**（零值没有相对尺度）。"
                              "1e-24这个数是算出来的：该构型的sum|项|折算成M的量级是4.5e-9，"
                              "乘eps得1.0e-24——即浮点能给出的最小非零残差。"
                              "**它抓的是把M取了绝对值、或把方括号符号写反的实现**："
                              "那类实现在这里给出1e-11量级的数，红过头1e13倍",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_general/reciprocity",
            "inputs": {
                "kind": "self_consistency_independent_of_any_reference_path",
                "configurations": [
                    {
                        "radius_a_m": entry[0],
                        "centre_a_m": list(entry[1]),
                        "normal_a": list(entry[2]),
                        "radius_b_m": entry[3],
                        "centre_b_m": list(entry[4]),
                        "normal_b": list(entry[5]),
                        "turns_a": entry[6],
                        "turns_b": entry[7],
                        "segments_a": entry[8],
                        "segments_b": entry[9],
                    }
                    for entry in RECIPROCITY_CONFIGURATIONS
                ],
            },
            "expected": {"reciprocity_max_abs_difference_h": 0.0},
            "tolerances": {
                "reciprocity_max_abs_difference_h": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "`M(a,b)`与`M(b,a)`必须**逐位相同**，且两条回路的**分段数"
                              "跟着交换**（96/61与71/53）。零容差是算出来的三步："
                              "① `t_i.t_j`与`t_j.t_i`是同一串乘法同一个加序；"
                              "② `|p_i-p_j|`与`|p_j-p_i|`只差一次精确取反；"
                              "③ **`math.fsum`精确求和后只舍一次，因此对项的置换不变**"
                              "——两个方向的项是同一个多重集。前因子`arc_a*arc_b`与"
                              "匝数`N_a*N_b`都可交换。**它抓的是角色写反那一类错**："
                              "把segments_a用在两条回路上、把centre_a当成两条的圆心、"
                              "把arc_a当成两条的弧长——这些错保持量纲、量级与远场退化阶，"
                              "只有互易性没了。若哪天改成`sum()`求和，本条也会红"
                              "（置换不变性随之消失）",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_general/far_field_dipole",
            "inputs": {
                "kind": "degeneration_to_the_dipole_dipole_limit_in_three_orientations",
                "radii_m": list(FAR_FIELD_RADII),
                "separations_m": list(FAR_FIELD_SEPARATIONS_M),
                "families": [
                    {"name": family, "normal_b": list(normal)}
                    for family, normal in FAR_FIELD_FAMILIES
                ],
                "segments": FAR_FIELD_SEGMENTS,
            },
            "expected": {
                "dipole_mutual_inductances_h": far_field_dipoles,
                "dipole_ratio_deviations": far_field_deviations,
                "dipole_convergence_orders": far_field_orders,
            },
            "tolerances": {
                "dipole_mutual_inductances_h": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "一般位形偶极式`(mu0 A1 A2/(4 pi d^3))[3(n1.dh)(n2.dh)-n1.n2]`"
                              "的两份独立抄写（内核一份、生成器一份）。分组不同，"
                              "实测最坏差1 ulp（1.6e-16）。取1e-15是4.5倍。"
                              "**共面并排那五个数是负的**——本条同时钉住符号，"
                              "而方括号在那一族恰好是-1",
                },
                "dipole_ratio_deviations": {
                    "abs": 5.0e-12, "rel": 0.0,
                    "reason": "偏差`M/M_dipole - 1`，三族取向各五档。"
                              "两侧的M_dipole是同一式的两份抄写（上一条已钉到1 ulp），"
                              "所以偏差之差约等于M本身的相对误差。"
                              "**判绝对不判相对**：偏差从1.8e-2扫到1.4e-5跨三个数量级，"
                              "相对容差会在小端把噪声当判据。"
                              "1e-12这个尺度来自条件数：d=3.2m那档kappa=1.3e5，"
                              "kappa*eps=2.9e-11——**这是本条最有信息量的一句**："
                              "远场判据的精度由相消而不是由求积决定，"
                              "再加密分段数一点用都没有。取5e-12覆盖它",
                },
                "dipole_convergence_orders": {
                    "abs": 1.0e-5, "rel": 0.0,
                    "reason": "相邻两档偏差之比取log2。理论值恰为2"
                              "（偏差的首阶是`O((R/d)^2)`，d翻倍降4倍）。"
                              "三族实测：共轴1.98062->1.99969、共面2.01652->2.00026、"
                              "倾斜60度2.01039->2.00017，**都单调逼近2但都不等于2**"
                              "（`harmonic_oscillator`那条纪律：收敛阶是测出来的不是断言的），"
                              "所以这十二个数是逐个钉住的、不是断言成2。"
                              "容差按最坏一档算：最小偏差1.376768e-5上叠加"
                              "kappa*eps=2.9e-11的绝对扰动，log2偏移"
                              "2.9e-11/1.38e-5/ln2=3.0e-6。取1e-5是3.3倍",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_general/singularity_and_resolution",
            "inputs": {
                "kind": "the_self_inductance_singularity_fails_closed",
                "probe_r1_r2_d_m": list(RESOLUTION_PROBE),
                "probe_segments": RESOLUTION_PROBE_SEGMENTS,
                "refusal_segments": RESOLUTION_REFUSAL_SEGMENTS,
                "closed_form_ratio": "delta/h = d N / (2 pi r)，等径同轴且两条回路同分段数时"
                                     "样本点按角对齐，最小样本距离恰为d",
            },
            "expected": {
                "resolution_ratio": resolution_ratio,
                "refusal_resolution_ratio": refusal_ratio,
                "self_pairing_refuses": True,
                "unresolved_pairing_refuses": True,
                "resolved_pairing_is_accepted": True,
            },
            "tolerances": {
                "resolution_ratio": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "`delta/h = d N/(2 pi r)`。等径同轴、两条回路同分段数时"
                              "样本点按角一一对齐，最小样本距离**恰为**d，"
                              "所以这一格能被独立算出来而不必复述实现的采样。"
                              "两侧都是几次乘除，差<=2eps；取1e-15是2.3倍。"
                              "**它钉的是分辨比的定义**：分母若写成较细那条的弧长"
                              "（而不是较粗那条），本条在两条回路分段数不同时会红",
                },
                "refusal_resolution_ratio": {
                    "abs": 0.0, "rel": 1.0e-15,
                    "reason": "同上，分段数减半那一档：分辨比落到2以下，属拒跑区。"
                              "**这个数必须被算出来而不是随口说'太近了'**——"
                              "拒跑的边界是一个可复算的量，不是一句判断",
                },
                "self_pairing_refuses": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：把同一条回路传两次必须**失败关闭**。"
                              "丝状回路的自感对数发散；离散之后它**不报错、"
                              "只给一个随分段数变化的有限数**——那正是必须拒跑的理由。"
                              "本仓不做GMD一类的正则化（要导线截面半径，那是S2.4那一格），"
                              "GAP与触发条件写在决策0092",
                },
                "unresolved_pairing_refuses": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：分辨比<2的构型必须拒跑。该档（N=24、delta/h=1.528）"
                              "把门拆掉后实测相对误差**5.78e-5**，而返回值有限、量级正确、"
                              "符号正确——**没有拒跑这道门，没有任何东西看得出它错**。"
                              "同一构型的N=16那一档是1.70e-3",
                },
                "resolved_pairing_is_accepted": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：分辨比>=2的同一构型必须**接受**。"
                              "**门要有两侧**：只验拒跑的话，一个把什么都拒掉的实现全绿",
                },
            },
        },
        {
            "id": "oracle:mutual_inductance_general/unit_boundary",
            "inputs": {
                "kind": "millimetre_declaration_must_agree_with_the_metre_one",
                "radius_a_mm": UNIT_RADIUS_A_MM,
                "radius_b_mm": UNIT_RADIUS_B_MM,
                "centre_b_mm": list(UNIT_CENTRE_B_MM),
                "normal_b": list(UNIT_NORMAL_B),
                "segments": UNIT_SEGMENTS,
            },
            "expected": {
                "mutual_inductance_from_millimetres_h": unit_value,
                "millimetre_and_metre_declarations_agree": True,
            },
            "tolerances": {
                "mutual_inductance_from_millimetres_h": {
                    "abs": 0.0, "rel": 1.0e-12,
                    "reason": "mm制声明的几何算出的互感，金标由米制的Biot-Savart面积分给出。"
                              "**它钉的是换算的方向**：往返判据（mm->m->mm）只验可逆，"
                              "验不出方向——乘1000再除1000同样可逆。方向反了差1e6倍。"
                              "容差与一般位形那一层同源",
                },
                "millimetre_and_metre_declarations_agree": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：同一几何用mm声明与用m声明给出**逐位相同**的M。"
                              "零容差成立是因为50.0/1000恰是0.05的double表示。"
                              "**法向不换算**（它是方向没有长度单位）——"
                              "这一条同时钉住那件事：若实现把法向也除以1000，"
                              "归一化之后结果碰巧照样对，于是'哪些量该换算'变成靠运气；"
                              "本条用一个**分量量级差很大**的法向"
                              "(0, 0.5, 0.866)让那种实现在别处露馅",
                },
            },
        },
    ]

    oracles.append(
        {
            "id": "oracle:mutual_inductance_general/section_bundle",
            "inputs": {
                "kind": "four_dimensional_gauss_legendre_over_both_sections",
                "coil_a_radius_axial_radial_extent_axial_extent_m": list(BUNDLE_COIL_A),
                "coil_b_radius_axial_radial_extent_axial_extent_m": list(BUNDLE_COIL_B),
                "segments": BUNDLE_SEGMENTS,
                "section_grids": list(BUNDLE_GRIDS),
                "relative_error_envelope": list(BUNDLE_ENVELOPE),
                "order_bracket": list(BUNDLE_ORDER_BRACKET),
                "filament_limit_grid": BUNDLE_LIMIT_GRID,
                "filament_limit_scales": list(BUNDLE_LIMIT_SCALES),
                "reference_nodes": BUNDLE_REFERENCE_NODES,
                "reference_self_check_nodes": BUNDLE_REFERENCE_SELF_CHECK_NODES,
                "reference_self_check": bundle_self_check,
                "reciprocity_grids": list(BUNDLE_RECIPROCITY_GRIDS),
                "reciprocity_segments": list(BUNDLE_RECIPROCITY_SEGMENTS),
                "turns": list(BUNDLE_TURNS),
                "current_density_assumption":
                    "截面上电流密度**均匀**（每根丝等份电流）。这是一个建模选择不是数值参数："
                    "直流成立、低频近似成立、高频（趋肤与邻近效应）不成立。S2.4只解开一半",
            },
            "expected": {
                "mutual_inductance_at_the_finest_grid_h": bundle_reference,
                "section_errors_fall_under_the_envelope": True,
                "section_refinement_orders_bracket_two": True,
                "filament_limit_orders_bracket_two": True,
                "zero_section_equals_the_centre_filament": True,
                "bundle_reciprocity_max_abs_difference_h": 0.0,
                "turns_factor_is_bit_exact": True,
            },
            "tolerances": {
                "mutual_inductance_at_the_finest_grid_h": {
                    "abs": 0.0, "rel": 2.1e-4,
                    "reason": "grid=4的截面平均对**四维Gauss-Legendre参考**"
                              "（每维12节点、核用Maxwell闭式，与被验侧求积法与核都不同；"
                              "两档8/12节点自校验相对差<=1e-15）。"
                              "**这条容差不是舍入而是声明的截断**：二维中点求积在grid=4上的"
                              "相对误差实测1.03504e-4，容差取它的2倍。"
                              "一条容差比舍入大九个数量级的判据要说清它在判什么——"
                              "它判的是**装配**（丝对平均、权重、匝数因子、截面网格的位置），"
                              "不是数值精度；数值精度由下面两条收敛判据判",
                },
                "section_errors_fall_under_the_envelope": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：grid=1/2/4三档对四维GL参考的相对误差落在包络内。"
                              "包络取实测的2倍：1.64463e-3 / 4.13424e-4 / 1.03504e-4。"
                              "**grid=1那一档就是中心细丝**，所以这条同时说明了"
                              "'把有截面的线圈当细丝算'要付多少代价：本构型1.6e-3",
                },
                "section_refinement_orders_bracket_two": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：相邻两档的log2误差比落在[1.9, 2.1]。实测1.99207 / 1.99794。"
                              "**这一条与`geometric_convergence`那一层方向相反、必须分开读**："
                              "那里加密的是回路角向（解析周期被积函数，几何收敛、没有固定阶），"
                              "这里加密的是截面（区间不周期，二维中点法就是代数二阶）。"
                              "**同一块代码里两条收敛阶不同不是矛盾，是两件事**。"
                              "实测N=24与N=32给出**同样的三个阶**，说明细丝求积没有污染这一层",
                },
                "filament_limit_orders_bracket_two": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：截面按1/2/4/8缩小时，与中心细丝的偏差比取log2落在[1.9, 2.1]。"
                              "实测2.00075 / 2.00022 / 2.00006。**首阶为零是因为截面对中心对称**"
                              "——一阶项相消，这正是'取中心细丝当零阶近似'能有二阶精度的原因。"
                              "**它抓的是截面网格摆歪**：若中点公式写成`i/n`而不是`(i+0.5)/n`，"
                              "截面整体偏半格，一阶项不再相消，阶掉到1",
                },
                "zero_section_equals_the_centre_filament": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：截面尺寸缩到0时，一束丝与中心那一根丝**逐位相同**。"
                              "零容差是算出来的：n^2根丝的位置与半径全部相同，"
                              "于是丝对的值也全部相同，`fsum(n^2个相同的值)/n^2`"
                              "在double下精确复原那个值（n^2是2的幂时更是显然，"
                              "本档n=4、n^2=16）。**它钉住`scaled_section(0)`这条退化路**",
                },
                "bundle_reciprocity_max_abs_difference_h": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "`M(A,B)`与`M(B,A)`**逐位相同**，且两条线圈的**截面细分(4与3)与"
                              "分段数(48与31)都不同**。零容差是算出来的：丝对的值集合在交换时"
                              "是同一个多重集（每一对本身逐位对称），`math.fsum`对置换不变，"
                              "而`n_a*n_b`与`N_a*N_b`都是可交换的乘法。"
                              "**截面细分不同这一点是故意的**：它是'角色写反'在本层的新形态",
                },
                "turns_factor_is_bit_exact": {
                    "abs": 0.0, "rel": 0.0,
                    "reason": "布尔：`M(N_a,N_b)`与`(N_a*N_b)*M(1,1)`**逐位相同**。"
                              "与`mutual_inductance_coaxial`那条同源，"
                              "但在本层多守一件事：**匝数与截面细分是两个完全无关的数**，"
                              "而把匝数当细分数（或反过来）是这块最容易犯的错——"
                              "本条与上一条（细分4与3、匝数3与7）合起来把两者钉死",
                },
            },
        }
    )

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/mutual_inductance_general",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/mutual_inductance_general/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"wrote {len(oracles)} oracles, {len(written)} bytes; "
        f"cross-check flux {max(flux_cross_check):.3e} / "
        f"closed form {max(closed_form_cross_check):.3e}; "
        f"coaxial M {min(coaxial_values):.4e}..{max(coaxial_values):.4e}; "
        f"general M {min(general_values):.4e}..{max(general_values):.4e}; "
        f"perpendicular by flux {perpendicular_by_flux:.3e}; "
        f"far-field orders {[round(order, 5) for order in far_field_orders]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
