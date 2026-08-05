"""同轴圆环互感的Maxwell闭式解——**引擎第三个物理域的第一个数**。

## 一、采信的闭式与它的出处

    k² = 4·r₁·r₂ / ((r₁+r₂)² + d²)
    M  = μ₀·√(r₁·r₂)·[ (2/k − k)·K(k) − (2/k)·E(k) ]

其中``K``、``E``是第一、二类完全椭圆积分，**按模k取值**（不是参数m = k²，
见``elliptic.py``第二节那条约定陷阱）；``d``是两回路平面的轴向间距。

出处：Maxwell《A Treatise on Electricity and Magnetism》第二卷第701节；
Grover《Inductance Calculations: Working Formulas and Tables》第十三章
（同轴共面圆环的互感）。两处都是同一式的不同排版。

**本仓不靠引文相信它**——`cases/mutual_inductance_coaxial`的金标由
**Neumann双回路线积分**独立求出（见第三节），闭式是被验的一方。

## 二、几何约定（每一个都被判据钉住）

* 两条回路**共轴**、平面垂直于公共轴；
* ``d = |z₂ − z₁| ≥ 0``；``d = 0``（共面同心）是**合法**构型，不是退化；
* ``M``与电流无关——互感是纯几何量（乘上μ₀）；
* ``N₁``匝与``N₂``匝：``M_N = N₁·N₂·M₁``（集中匝理想化，见第五节负空间）。

## 三、金标那一侧走的是Neumann积分，不是另一个椭圆积分实现

    M = (μ₀/4π)·∮∮ (dl₁·dl₂)/|r₁ − r₂|

对共轴圆环，按角差φ = φ₂ − φ₁化成单重积分：

    M = (μ₀·r₁·r₂/2)·∫₀^{2π} cos φ dφ / √(r₁² + r₂² + d² − 2·r₁·r₂·cos φ)

被积函数在整个复平面解析且2π周期，**周期解析函数上梯形法几何收敛**——
与`cases/scalar_diffraction_airy`用的是同一条形制。生成器还把这个积分改写成
**全正被积函数**（否则远场下`∫cos φ ≈ 0`的相消会吃掉十几位），细节在生成器里。

这条路与本模块**没有任何共用代码**：它不算椭圆积分、不用AGM、不import本包。
它同时把plans/04第五节第3条（一般位形互感的Neumann数值积分）的地基打了。

## 四、数值上要小心的那一处，以及一条背景订正

方括号在小k（=远场）下相消放大约``1/k⁴``。本模块**不按教科书分组求值**，
改用AGM的中间量算同一个量，全程无减法——推导、实测对比表与两个AGM自身的坑
都写在``elliptic.py``的docstring里。这不是优化，是**能不能做远场判据**的前提：
照教科书分组写，d增大两三个倍频程后误差就淹没掉待测的偏差本身。

**背景订正（2026-08-05，与research/12同批）**：research/08默认"电磁 = FEM棱边元"，
因此把"体网格与关联表"列为门槛2。**对互感/电容这一族那个默认是错的**——
工业主路是FastHenry/FasterCap一类的**积分方程**方法，只离散导体、不离散空气。
本块走的正是这条路的解析端，**不为"将来要接网格"预留任何东西**。

## 五、明确不做的（负空间声明，Drake形制）

* **不做自感**。丝状回路自感对数发散，要导线截面半径才有限；
* **不做一般位形**（倾斜、偏心、非共轴）。要Neumann双重线积分与细丝离散；
* **不做电容**。静电场求解要网格，除同轴/平板/球形几个闭式构型外无解析解
  （plans/04第二节明写"互感先做，电容等网格"）；
* **不做磁介质**。``M``里的μ₀是真空值；把它换成μ只在**无限大均匀介质**里成立，
  真实磁芯要解场，不是乘一个数；
* **不做匝间几何**。``turns``是集中匝：N匝全部叠在同一条几何回路上。
  真实多层线圈的匝有轴向与径向间距，``N₁·N₂·M₁``只是它的零阶近似——
  本模块**只声称集中匝那个理想化是精确的**，不声称它近似真实线圈；
* **不做力与力矩**。两条载流回路之间有轴向力``F = I₁I₂·dM/dd``，
  那要``M``对``d``的导数，是下一块；
* **不做时变**。互感是静磁量；感应电动势要时间导数，本块连时间都没有。
"""

from __future__ import annotations

import math

from physics_engine.electromagnetics.elliptic import MODULUS_MAX, maxwell_mutual_bracket
from physics_engine.electromagnetics.errors import ElectromagneticsError
from physics_engine.electromagnetics.loops import CircularLoop
from physics_engine.electromagnetics.units import VACUUM_PERMEABILITY_H_PER_M


def coaxial_modulus(
    *, radius_a_m: float, radius_b_m: float, axial_separation_m: float
) -> float:
    """Maxwell闭式里的模``k = √(4·r₁·r₂ / ((r₁+r₂)² + d²))``。

    **公开它是有理由的**：``k``是这条闭式唯一的形状参数，
    案例的判据表按``k``分档写理由，而"判据表里的数从哪来"不该只有注释交代。

    表达式对两个半径**逐位对称**（``r₁*r₂``与``(r₁+r₂)``在IEEE754下都可交换），
    这正是互易判据能声明零容差的原因。
    """

    radius_a = _require_positive(radius_a_m, "radius_a_m")
    radius_b = _require_positive(radius_b_m, "radius_b_m")
    separation = _require_nonnegative(axial_separation_m, "axial_separation_m")
    denominator = (radius_a + radius_b) ** 2 + separation * separation
    squared = 4.0 * radius_a * radius_b / denominator
    # AM-GM保证squared ≤ 1，等号仅当r₁=r₂且d=0（两条回路重合）。
    # 浮点下等号附近可能给出略大于1的值，所以下面这条判定按MODULUS_MAX做，
    # 而不是先开方再判——开方一个>1的数直接就是ValueError，信息还更差。
    if squared > MODULUS_MAX * MODULUS_MAX:
        raise ElectromagneticsError(
            f"两条回路过于接近重合（k²={squared!r}）：丝状回路重合时互感对数发散，"
            "有限值要引入导线截面半径——那是另一条闭式，本模块不做。"
            "拒跑，不夹边界也不返回inf"
        )
    return math.sqrt(squared)


def coaxial_mutual_inductance_h(
    *, radius_a_m: float, radius_b_m: float, axial_separation_m: float
) -> float:
    """两条**单匝**共轴圆环的互感（亨利）。

    输入按米（``units.EM_LENGTH_UNIT``）。mm几何请先走
    ``CircularLoop.from_millimetres``或``units.metres_from_millimetres``——
    本函数**不猜**单位，传mm进来会给出一个大一千倍的、看起来完全正常的数。
    """

    modulus = coaxial_modulus(
        radius_a_m=radius_a_m,
        radius_b_m=radius_b_m,
        axial_separation_m=axial_separation_m,
    )
    geometric_mean_radius_m = math.sqrt(float(radius_a_m) * float(radius_b_m))
    return (
        VACUUM_PERMEABILITY_H_PER_M
        * geometric_mean_radius_m
        * maxwell_mutual_bracket(modulus)
    )


def mutual_inductance_h(loop_a: CircularLoop, loop_b: CircularLoop) -> float:
    """两条共轴圆形回路的互感（亨利），**含匝数**。

    ``M = N_a·N_b·M₁``，``M₁``是单匝值。匝数因子先乘在一起再乘单匝值，
    所以``M``对两条回路的交换是**逐位对称**的（整数乘法可交换且精确）。
    """

    for name, loop in (("loop_a", loop_a), ("loop_b", loop_b)):
        if not isinstance(loop, CircularLoop):
            raise ElectromagneticsError(f"{name}必须是CircularLoop：{loop!r}")
    single_turn = coaxial_mutual_inductance_h(
        radius_a_m=loop_a.radius_m,
        radius_b_m=loop_b.radius_m,
        axial_separation_m=loop_a.axial_separation_m(loop_b),
    )
    return (loop_a.turns * loop_b.turns) * single_turn


def dipole_mutual_inductance_h(
    *, radius_a_m: float, radius_b_m: float, axial_separation_m: float
) -> float:
    """远场偶极子近似``M ≈ μ₀·π·r₁²·r₂²/(2·d³)``（单匝）。

    由回路轴上磁场``B = μ₀·r₁²·I/(2·(r₁²+d²)^{3/2})``在``d ≫ r₁``下取
    ``μ₀·r₁²·I/(2d³)``、再乘``π·r₂²``得到。

    **这是一条近似不是闭式**，公开它是为了让"闭式在远场退化到它"成为一条可算的判据，
    不是为了让人拿它算数——真要算就用``coaxial_mutual_inductance_h``，
    在d/r = 5处这条近似已经差1.8%。
    """

    radius_a = _require_positive(radius_a_m, "radius_a_m")
    radius_b = _require_positive(radius_b_m, "radius_b_m")
    separation = _require_nonnegative(axial_separation_m, "axial_separation_m")
    if separation == 0.0:
        raise ElectromagneticsError(
            "偶极子近似在d=0处发散——它按定义只在d ≫ r时有意义。拒跑"
        )
    return (
        VACUUM_PERMEABILITY_H_PER_M
        * math.pi
        * radius_a
        * radius_a
        * radius_b
        * radius_b
        / (2.0 * separation**3)
    )


def flux_linkage_wb(*, source: CircularLoop, target: CircularLoop) -> float:
    """``source``的电流在``target``上产生的磁链``λ = M·I_source``（韦伯）。

    互感已含两侧匝数，所以这里不再乘任何匝数——**多乘一次是这条式子最容易犯的错**，
    案例里有一条判据按``λ``对``I``严格线性来守它。
    """

    return mutual_inductance_h(source, target) * source.current_a


def _require_positive(value: object, name: str) -> float:
    number = _require_finite(value, name)
    if number <= 0.0:
        raise ElectromagneticsError(f"{name}必须为正：{value!r}")
    return number


def _require_nonnegative(value: object, name: str) -> float:
    number = _require_finite(value, name)
    if number < 0.0:
        raise ElectromagneticsError(
            f"{name}必须≥0：{value!r}——轴向间距按定义取绝对值，负值说明调用点算错了方向"
        )
    return number


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ElectromagneticsError(f"{name}必须是实数：{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ElectromagneticsError(f"{name}必须是有限值：{value!r}")
    return number


__all__ = [
    "coaxial_modulus",
    "coaxial_mutual_inductance_h",
    "dipole_mutual_inductance_h",
    "flux_linkage_wb",
    "mutual_inductance_h",
]
