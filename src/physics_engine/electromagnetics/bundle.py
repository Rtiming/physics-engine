"""有截面的线圈：把导体截面离散成一束细丝——S2.4那一格的**数值那一半**。

`neumann.py`算的是两条**无粗细的丝**。真实线圈有截面
（`docs/capability_ledger.json`的S2.4原文："`CircularLoop`是半径+匝数的细丝；
导体截面上的电流分布无处安放"）。本模块把截面按矩形网格离散成一束同轴细丝，
每根丝分到**等份**电流，互感取全部丝对的平均：

    M = (N_a·N_b / (n_a·n_b))·Σ_{i,j} M₁(f_i, f_j)

## 一、这一格解开了哪一半，另一半为什么没解开

**解开的是"截面几何"**：导体不再是一条线，它有径向厚度与轴向高度，
截面上不同位置的丝有不同的半径与轴向位置，互感是它们的平均——
这正是Grover一族"矩形截面线圈互感"公式在做的事，只是本模块用数值不用级数。

**没解开的是"电流分布"**，而且是**明写不做**：本模块假设截面上**电流密度均匀**。
真实导体在交流下有趋肤与邻近效应，电流会挤到截面一侧，
那时"每根丝等份电流"就是错的。**均匀电流密度是一个建模选择，不是一个数值参数**：
它对直流成立、对低频近似成立，对高频不成立，而本模块**不判断调用方在哪一档**。
S2.4那一格因此只解开一半——`docs/decisions/0092`第六节把另一半登记成GAP。

## 二、两条收敛，方向不同，必须分开验

1. **截面细分收敛**（固定截面，加密n_r×n_z）：二维中点求积，**代数二阶**。
   实测（两条同轴矩形截面线圈）：grid 2→4→8→16的相对误差
   3.5e-4 / 8.8e-5 / 2.2e-5 / 5.5e-6，相邻比log2 = 2.00/2.00/2.00。
   **注意它与`neumann.py`那一条的方向相反**：那里加密的是**回路的角向**，
   被积函数解析且周期，几何收敛；这里加密的是**截面**，区间不周期，
   中点法就是二阶。**同一块代码里两条收敛阶不同，这不是矛盾是两件事**；
2. **细丝极限**（截面按比例缩小）：截面缩小s倍时与细丝值的偏差按**s²**降。
   实测s = 1/1、1/2、1/4、1/8：偏差2.0e-3 / 5.0e-4 / 1.2e-4 / 3.1e-5，
   相邻比log2 = 2.00/2.00/2.00。首阶为零是因为**截面对中心对称**——
   一阶项在对称截面上相消，这也是"取中心细丝当零阶近似"能有二阶精度的原因。

## 三、代价：丝对数是乘出来的

一次求值的代价是``n_a·n_b``对细丝、每对``O(N_a·N_b)``。
16×16截面对16×16截面是**65536对**，每对N=32时又是1024项——
实测本机Mac上grid=8（64×64=4096对、N=32）一次约**1.1 s**。
**这是本仓第一块"参数一调就慢两个数量级"的电磁代码**，
所以`bundle_mutual_inductance_h`的分段数与截面细分**都是必填**，没有默认值。

加速档的正当位置在这里比在`neumann.py`还明显（丝对之间完全独立），
但0014的零设施承诺要求纯Python档先存在且被判据钉住。GAP登记在决策0092。

## 四、自感仍然拒跑

把同一束丝传两次，丝对里有``f_i``与自己配对——`neumann.py`那道重合门当场拒。
**这不是本模块的局限，是它必须有的行为**：有截面之后自感**确实有限**
（这正是GMD一类方法存在的理由），但要算它必须回答"同一根丝与自己"那一项，
而那一项要的是**导线内部的自感**（含内部磁能，与电流分布直接相关）——
回到第一节那个没解开的一半。**给一个静默的数比拒跑坏得多。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.electromagnetics.errors import ElectromagneticsError
from physics_engine.electromagnetics.neumann import (
    PlacedCircularLoop,
    neumann_mutual_inductance_h,
)
from physics_engine.electromagnetics.units import metres_from_millimetres

#: 截面每个方向最少几份。1份=退化成中心细丝，**是合法的**：
#: 它正是"细丝极限"那条判据的一端，也是调用方检查自己有没有被截面影响的最快办法。
SECTION_DIVISIONS_MIN: int = 1


@dataclass(frozen=True)
class RectangularSectionCoil:
    """矩形截面的圆形线圈：平均半径 + 圆心 + 法向 + 截面尺寸 + 截面细分。

    截面是**径向×轴向**的矩形，中心落在``(mean_radius_m, centre_m)``上：
    径向半宽``radial_extent_m/2``、轴向半高``axial_extent_m/2``。

    ``turns``是这一束的**总匝数**，按``N/(n_r·n_z)``均分到每根丝——
    这就是第一节那条"电流密度均匀"的假设在代码里的样子。
    **它是一个建模选择，本类型不判断调用方在哪一档**。

    ``radial_filaments``/``axial_filaments``是**求积细分**不是物理：
    同一根导体分成4×4还是8×8，物理不变、数值精度变。
    两者与``turns``**完全无关**——把匝数当细分数写是这块最容易犯的错，
    案例里有一条判据钉住"细分加倍而M不变（在收敛容差内）"。
    """

    mean_radius_m: float
    centre_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    radial_extent_m: float
    axial_extent_m: float
    radial_filaments: int
    axial_filaments: int
    turns: int = 1

    def __post_init__(self) -> None:
        # 位形与匝数的合法性交给`PlacedCircularLoop`——它是同一套规矩的正本，
        # 在这里再写一遍就会有两份会各自漂移的校验。
        probe = PlacedCircularLoop(
            radius_m=self.mean_radius_m,
            centre_m=self.centre_m,
            normal=self.normal,
            turns=self.turns,
        )
        radial = _require_nonnegative(self.radial_extent_m, "radial_extent_m")
        axial = _require_nonnegative(self.axial_extent_m, "axial_extent_m")
        if radial >= 2.0 * probe.radius_m:
            raise ElectromagneticsError(
                f"radial_extent_m={self.radial_extent_m!r}不小于平均直径"
                f"{2.0 * probe.radius_m!r}：截面会跨过轴心，最内侧那圈丝的半径≤0。"
                "拒跑——那不是一个线圈截面"
            )
        _require_divisions(self.radial_filaments, "radial_filaments")
        _require_divisions(self.axial_filaments, "axial_filaments")
        object.__setattr__(self, "centre_m", probe.centre_m)
        object.__setattr__(self, "normal", probe.normal)
        object.__setattr__(self, "radial_extent_m", radial)
        object.__setattr__(self, "axial_extent_m", axial)

    @classmethod
    def from_millimetres(
        cls,
        *,
        mean_radius_mm: float,
        centre_mm: tuple[float, float, float],
        normal: tuple[float, float, float],
        radial_extent_mm: float,
        axial_extent_mm: float,
        radial_filaments: int,
        axial_filaments: int,
        turns: int = 1,
    ) -> RectangularSectionCoil:
        """从mm制几何构造。``normal``**不换算**（方向没有长度单位）。"""

        centre = tuple(metres_from_millimetres(value) for value in centre_mm)
        return cls(
            mean_radius_m=metres_from_millimetres(mean_radius_mm),
            centre_m=(centre[0], centre[1], centre[2]),
            normal=normal,
            radial_extent_m=metres_from_millimetres(radial_extent_mm),
            axial_extent_m=metres_from_millimetres(axial_extent_mm),
            radial_filaments=radial_filaments,
            axial_filaments=axial_filaments,
            turns=turns,
        )

    def scaled_section(self, factor: float) -> RectangularSectionCoil:
        """同一条线圈、截面按``factor``缩放——**细丝极限那条判据的整条路**。

        ``factor = 0``给出退化的零截面（等价于中心细丝）。
        """

        scale = _require_nonnegative(factor, "factor")
        return RectangularSectionCoil(
            mean_radius_m=self.mean_radius_m,
            centre_m=self.centre_m,
            normal=self.normal,
            radial_extent_m=self.radial_extent_m * scale,
            axial_extent_m=self.axial_extent_m * scale,
            radial_filaments=self.radial_filaments,
            axial_filaments=self.axial_filaments,
            turns=self.turns,
        )

    def centre_filament(self) -> PlacedCircularLoop:
        """截面中心那一根丝，带**整条线圈的匝数**——细丝极限的比较对象。"""

        return PlacedCircularLoop(
            radius_m=self.mean_radius_m,
            centre_m=self.centre_m,
            normal=self.normal,
            turns=self.turns,
        )

    def filament_count(self) -> int:
        return self.radial_filaments * self.axial_filaments


def section_filaments(coil: RectangularSectionCoil) -> tuple[PlacedCircularLoop, ...]:
    """截面上的``n_r × n_z``根细丝，**每根`turns=1`**。

    取每个小格的**中点**（与`neumann.py`的角向采样同一条约定）：

        r_i = R + Δr·((i+½)/n_r − ½)，  z_j = Δz·((j+½)/n_z − ½)

    匝数**不在这里**乘：它是整束的属性，在
    `bundle_mutual_inductance_h`里与丝对平均一起乘，
    这样"平均"与"匝数"两件事在代码里各出现一次。
    """

    if not isinstance(coil, RectangularSectionCoil):
        raise ElectromagneticsError(f"section_filaments只接受RectangularSectionCoil：{coil!r}")
    normal = coil.normal
    centre = coil.centre_m
    filaments = []
    for radial_index in range(coil.radial_filaments):
        radius = coil.mean_radius_m + coil.radial_extent_m * (
            (radial_index + 0.5) / coil.radial_filaments - 0.5
        )
        for axial_index in range(coil.axial_filaments):
            offset = coil.axial_extent_m * (
                (axial_index + 0.5) / coil.axial_filaments - 0.5
            )
            filaments.append(
                PlacedCircularLoop(
                    radius_m=radius,
                    centre_m=(
                        centre[0] + offset * normal[0],
                        centre[1] + offset * normal[1],
                        centre[2] + offset * normal[2],
                    ),
                    normal=normal,
                )
            )
    return tuple(filaments)


def bundle_mutual_inductance_h(
    coil_a: RectangularSectionCoil,
    coil_b: RectangularSectionCoil,
    *,
    segments_a: int,
    segments_b: int,
) -> float:
    """两条有截面线圈的互感（亨利），**含匝数**。

    ``segments_*``是每根丝的角向分段数，截面细分在线圈自己身上——
    **两者都必填**：这块的代价是``n_a·n_b·N_a·N_b``，
    给任何一个默认值都等于替调用方做一个它看不见的时间与精度决定。

    **互易逐位成立**：丝对的值集合在交换两个参数时是同一个多重集
    （每一对本身逐位对称，见`neumann.py`第五节），`math.fsum`对置换不变，
    而``n_a·n_b``与``N_a·N_b``都是可交换的乘法。
    """

    filaments_a = section_filaments(coil_a)
    filaments_b = section_filaments(coil_b)
    values = [
        neumann_mutual_inductance_h(
            filament_a, filament_b, segments_a=segments_a, segments_b=segments_b
        )
        for filament_a in filaments_a
        for filament_b in filaments_b
    ]
    average = math.fsum(values) / (len(filaments_a) * len(filaments_b))
    return (coil_a.turns * coil_b.turns) * average


def _require_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ElectromagneticsError(f"{name}必须是实数：{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ElectromagneticsError(f"{name}必须是有限值：{value!r}")
    if number < 0.0:
        raise ElectromagneticsError(
            f"{name}必须≥0：{value!r}——零是合法的（退化成细丝），负的不是"
        )
    return number


def _require_divisions(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ElectromagneticsError(f"{name}必须是整数：{value!r}")
    if value < SECTION_DIVISIONS_MIN:
        raise ElectromagneticsError(
            f"{name}必须≥{SECTION_DIVISIONS_MIN}：{value!r}"
        )
    return value


__all__ = [
    "RectangularSectionCoil",
    "SECTION_DIVISIONS_MIN",
    "bundle_mutual_inductance_h",
    "section_filaments",
]
