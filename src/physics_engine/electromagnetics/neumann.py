"""一般位形（倾斜／偏心／非共轴）两条圆形丝状回路的互感——Neumann双回路线积分。

`inductance.py`那条Maxwell闭式**只对共轴成立**，而真实线圈组不共轴
（plans/04第二节：场景②从"一半"走到"整条"的那一步）。本模块把
plans/04第二节那句"Neumann公式可以用细丝离散算，不需要体网格"落成代码：

    M = (μ₀/4π)·∮∮ (dl₁·dl₂)/|r₁ − r₂|

**没有网格、没有自由度、没有求解**——与`inductance.py`一样绕开research/08那六道门槛。

## 一、离散约定：中点切元，**不是折线弦**。这条差别决定收敛阶

回路按角度均匀取``segments``个**中点**样本，每个样本带**该点的精确切向**
与**弧长权重**``2πR/N``：

    θ_i = 2π(i + ½)/N，  p_i = c + R(u·cos θ_i + v·sin θ_i)
    t_i = −u·sin θ_i + v·cos θ_i，  |dl_i| = 2πR/N

于是本模块算的是**精确Neumann积分的中点求积**，不是"用多边形代替圆"。
被积函数在两条回路不相交时解析且对两个角都2π周期，
**周期解析函数上中点／梯形法几何收敛**（与`cases/scalar_diffraction_airy`、
`cases/mutual_inductance_coaxial`的生成器同一条形制）。

**两种约定的实测差别（同轴r₁=r₂=50mm、d=20mm，对Maxwell闭式）**：

| N | 本模块（中点切元） | 折线弦（多边形细丝） |
|---|---|---|
| 8 | 5.9e-2 | 8.0e-3 |
| 16 | 1.7e-3 | 7.8e-3 |
| 32 | 2.1e-6 | 2.3e-3 |
| 64 | **4.4e-12** | 5.8e-4 |
| 128 | 2.0e-16（浮点地板） | 1.4e-4 |
| 256 | 2.0e-16 | 3.6e-5 |

（这张表用的是**不带分辨门**的对照实现，只为把两种约定的收敛形态摆在一起：
该构型在N=8与N=16处的δ/h只有0.51与1.02，公开入口会拒跑，见第三节。）

折线弦那一列的相邻比**恰好是2.000**——它的误差不在求积上，在**几何**上：
多边形的周长与所围面积本身就差O(1/N²)，再怎么加密求积也修不掉。
本模块的那一列没有固定阶，**相邻两档的表观阶随加密上升**
（2.98 → 5.12 → 9.67 → 18.84），那正是几何收敛的指纹。

**所以"收敛阶是多少"这个问题在本模块没有一个数**：它不是二阶、不是四阶，
是几何收敛，直到撞上浮点地板。`tests/test_neumann_inductance.py`里有一条
**必须红**的门把折线弦那一列的二阶行为钉住——否则上面这张表只是一句话。

## 二、相消：条件数是``Σ|项| / |Σ项|``，求和用``math.fsum``

双重和的项有正有负，而和远小于项的量级。定义

    κ = Σ|dl_i·dl_j / r_ij| / |Σ dl_i·dl_j / r_ij|

它就是这条求和的条件数，**实测随距离平方增长**（同轴r=10mm）：

| d/r | κ | 由κ推出的相对误差地板 |
|---|---|---|
| 2 | 8.0 | 1.8e-15 |
| 20 | 5.1e2 | 1.1e-13 |
| 160（共面并排d=3.2m） | 1.3e5 | 2.9e-11 |

处理是`math.fsum`：它把项的多重集**精确求和后只舍入一次**，
于是"求和"这一步不再贡献任何误差，剩下的只有每一项自身的舍入乘以κ。
朴素累加（``total += 项``）实测把远场共面并排那一组再劣化2.5e-12，
**并且当场破坏逐位互易**（同一构型两个方向差2.0e-12）——
`tests/test_neumann_inductance.py`有一条**必须红**的门钉住它。

**这里有一条要写清楚的实测，否则下一个人会"顺手优化"掉`fsum`**：
CPython 3.12起内置``sum()``对浮点改用了Neumaier补偿求和，
本模块实测的五组构型里``sum(项表)``与``math.fsum(项表)``**逐位相同**
（含κ=1.3e5那一组）。所以"朴素求和很糟"这条老论证**对今天的``sum()``不成立**。
仍然取`fsum`的理由只有一条、而且是硬的：**`fsum`按定义精确舍入，
因此"对项的置换不变"是有保证的；``sum()``的补偿求和没有这个保证，只是碰巧对上**。
零容差的互易判据靠的正是那条保证。代价实测：9216项时`fsum` 123μs、``sum`` 22μs，
而一次N=96的求值本身约2.5ms——**代价是4%，买的是一条能写成零容差的判据**。

**κ是公开量**（`neumann_condition_number`）：远场判据要靠它说明"能算到几位"，
而不是靠"看起来对"。

## 三、近距：决定精度的那个数是``δ/h``，不是N

``δ``=两条回路样本点间的最小距离，``h``=较粗那条的弧长步长。
**同一个δ/h下误差几乎与N无关**（实测，同轴r=50mm）：

| δ/h | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---|---|---|---|---|
| 0.25 | 1.5e-1 | 1.2e-1 | 1.0e-1 | 8.9e-2 | 7.9e-2 |
| 0.5 | 2.5e-2 | 2.0e-2 | 1.6e-2 | 1.4e-2 | 1.2e-2 |
| 1 | 1.1e-3 | 7.7e-4 | 5.9e-4 | 4.8e-4 | 4.1e-4 |
| 2 | 2.6e-6 | 1.5e-6 | 1.0e-6 | 7.9e-7 | 6.5e-7 |
| 4 | 2.6e-11 | 7.0e-12 | 3.7e-12 | 2.5e-12 | 2.0e-12 |
| 8 | 0 | 3.9e-16 | 0 | 4.9e-16 | 2.4e-15 |

**这张表是本模块最贵的一条实测**：它说明"多加点就更准"是错的——
把N翻倍而两条回路仍靠得一样近（δ不变），δ/h跟着翻倍才是变准的原因；
若两条回路的距离随N一起缩（例如自感），加密**一点用都没有**。

门开在``RESOLUTION_RATIO_MIN = 2``：**低于它拒跑**，
因为那以下的误差已经到1e-3量级而返回值看起来完全正常（有限、量级对、符号对）。
δ/h ≥ 2只保证约1e-6，**要工程精度请自己把δ/h推到4以上**——
上表是公开常量`RESOLUTION_RATIO_CALIBRATION`，不是注释。

## 四、自感：**拒跑**，本模块不做任何正则化

同一条回路上的Neumann积分对数发散（``r_ij → 0``）。离散之后它**不会报错**，
只会给出一个有限的、随N变化的数——**那正是本模块必须拒跑的理由**：
一个静默的自感值比一个异常危险得多。

两道门：① 两条回路完全重合（半径、圆心、法向平行）当场拒，报的是物理理由；
② 一般地``δ/h < 2``拒——**自感是它的极限情形**（δ=0，无论N多大都过不去）。

**本模块不做GMD（几何平均距离）一类的正则化**。理由不是它难：
GMD要一个**导线截面半径**，而"导体有截面"是S2.4那一格
（`docs/capability_ledger.json`），它同时改变电流分布的模型假设。
装一个"取一个小半径当正则化参数"的旋钮，等于把一个建模选择伪装成数值参数。
触发条件写在决策0092的GAP表里。

## 五、互易``M₁₂ = M₂₁``**逐位**成立，而且是算出来的不是许愿的

三步都逐位可证：① ``dl_i·dl_j``与``dl_j·dl_i``是同一串乘法同一个加序；
② ``|p_i − p_j|``与``|p_j − p_i|``只差一次精确取反，平方后逐位相同；
③ **`math.fsum`精确求和后只舍一次，因此对项的置换不变**——
两个方向的项集合是同一个多重集，只是枚举次序反了。
前因子``arc_a·arc_b``与匝数``N_a·N_b``都是可交换的乘法。

**它抓的是"两条回路的角色被写反"这一类错**：把``segments_a``用在两条回路上、
把``centre_a``当成两条回路的圆心、把``arc_a``当成两条的弧长——
这些错都保持量纲、量级与远场退化阶，**只有互易性没了**。

## 六、明确不做的（负空间声明）

* **不做自感**（上面第四节）；
* **不做导线截面**。`PlacedCircularLoop`是无粗细的丝，``turns``仍是集中匝
  （N匝叠在同一条几何回路上）——与`loops.CircularLoop`同一个理想化；
* **不做非圆回路**。任意曲线的Neumann积分只要换一个采样器，
  但"引擎里的曲线是什么"今天没有裁决（`shapes.py`没有回路词汇，
  决策0042待裁2登记着触发条件）。本模块**不为它预留入口**（0001第二前提）；
* **不做力与力矩**（要M对位形的导数）、**不做时变**、**不做磁介质**、
  **不做电容**——与`inductance.py`第五节逐条相同；
* **不做加速档**。双重和是O(N_a·N_b)标量运算，NumPy档的正当位置正是这里
  （`electromagnetics/__init__.py`那句"加速档的正当位置是细丝离散的一般位形互感"
  说的就是本模块），但0014的零设施承诺要求纯Python档先存在且被判据钉住。
  实测N=128×128一次求值约4ms、192×192约11ms（本机Mac，墙钟不作数），
  今天没有任何消费方被这个数挡住。**加速档留GAP，触发条件在决策0092**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.electromagnetics.errors import ElectromagneticsError
from physics_engine.electromagnetics.loops import CircularLoop
from physics_engine.electromagnetics.units import (
    VACUUM_PERMEABILITY_H_PER_M,
    metres_from_millimetres,
)

#: 一条回路最少几段。三段是"闭合回路"这个词在离散上的下限（三角形）；
#: **它不是精度下限**——N=4在近距构型上实测差47%。精度看的是δ/h那张表。
SEGMENTS_MIN: int = 3

#: 法向向量的模下限。**不是为了防除零**：模在这个量级的"方向"通常来自
#: 两个近平行向量的叉积，归一化会把噪声放大成一个看起来正常的方向。
#: 拒跑，不替调用方猜。
NORMAL_NORM_MIN: float = 1.0e-12

#: 分辨比``δ/h``的下限，低于它拒跑。取2的理由是下面那张实测标定表：
#: δ/h=2对应约1e-6的相对误差，而δ/h=1已经到1e-3——
#: **两者都返回有限的、量级正确的数**，不拒跑就没有任何东西能发现它。
RESOLUTION_RATIO_MIN: float = 2.0

#: 分辨比标定表``(δ/h, 相对误差)``：同轴r₁=r₂=50mm、N从32扫到512、
#: 对Maxwell闭式实测的**最坏**值（模块docstring第三节有全表）。
#: 公开它是因为`RESOLUTION_RATIO_MIN`只是**门**不是**精度承诺**：
#: 要多准，调用方得自己按这张表把δ/h推上去。
RESOLUTION_RATIO_CALIBRATION: tuple[tuple[float, float], ...] = (
    (0.25, 1.5e-1),
    (0.5, 2.5e-2),
    (1.0, 1.1e-3),
    (2.0, 2.6e-6),
    (4.0, 2.6e-11),
    (8.0, 2.5e-15),
)

#: ``μ₀/4π``。**只算一次**：两个方向的互易判据要求前因子逐位相同，
#: 而每次现算``VACUUM_PERMEABILITY_H_PER_M/(4.0*math.pi)``虽然也逐位相同，
#: 却让"为什么它相同"多一条要论证的路。
NEUMANN_PREFACTOR_H_PER_M: float = VACUUM_PERMEABILITY_H_PER_M / (4.0 * math.pi)


@dataclass(frozen=True)
class PlacedCircularLoop:
    """一般位形的圆形丝状回路：半径 + 圆心 + 单位法向（+ 匝数、载流）。

    与`loops.CircularLoop`的分工**不是继承而是并列**：那一个按定义共轴
    （只有``axial_position_m``一个位形自由度，因此Maxwell闭式对它成立）；
    本类型有完整的位形，因此**只能走数值积分**。两者用
    ``from_coaxial``单向相通——共轴是一般位形的特例，反过来不是。

    ``normal``在构造时**归一化并存回**。方向向量的模没有物理含义，
    所以这不是"偷偷改了输入"那一类事；模小于``NORMAL_NORM_MIN``则拒跑
    （见该常量的理由）。

    ``turns``与`CircularLoop`同为**集中匝**理想化：N匝叠在同一条几何回路上。
    真实多层线圈的匝有轴向与径向间距，那是S2.3那一格，本类型不声称覆盖它。

    ``current_a``今天**没有任何本模块的函数读它**——互感是纯几何量。
    它在这里是为了让本类型能与`CircularLoop`对等地描述"一条回路"，
    并且让将来的磁链有一个可算的定义；**若这条理由半年后仍然只是这句话，
    该删的是这个字段**。
    """

    radius_m: float
    centre_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    turns: int = 1
    current_a: float = 0.0

    def __post_init__(self) -> None:
        radius = _require_finite(self.radius_m, "radius_m")
        if radius <= 0.0:
            raise ElectromagneticsError(
                f"radius_m必须为正：{self.radius_m!r}——半径为零的回路互感恒为零，那不是一条回路"
            )
        centre = _require_vector(self.centre_m, "centre_m")
        normal = _require_vector(self.normal, "normal")
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        if norm < NORMAL_NORM_MIN:
            raise ElectromagneticsError(
                f"normal的模{norm!r}小于{NORMAL_NORM_MIN!r}：这样的'方向'通常是"
                "两个近平行向量叉积出来的噪声，归一化会把噪声放大成一个看起来正常的方向。"
                "拒跑，不替调用方猜"
            )
        turns = self.turns
        if isinstance(turns, bool) or not isinstance(turns, int):
            raise ElectromagneticsError(f"turns必须是整数：{self.turns!r}")
        if turns < 1:
            raise ElectromagneticsError(f"turns必须≥1：{self.turns!r}")
        _require_finite(self.current_a, "current_a")
        object.__setattr__(self, "centre_m", centre)
        object.__setattr__(
            self, "normal", (normal[0] / norm, normal[1] / norm, normal[2] / norm)
        )

    @classmethod
    def from_millimetres(
        cls,
        *,
        radius_mm: float,
        centre_mm: tuple[float, float, float],
        normal: tuple[float, float, float],
        turns: int = 1,
        current_a: float = 0.0,
    ) -> PlacedCircularLoop:
        """从mm制几何构造——**mm进入本模块的唯一入口**（与`CircularLoop`同规矩）。

        ``normal``**不换算**：它是方向，没有长度单位。
        这条不写清楚就会有人把法向也除以1000（结果碰巧照样对，因为要归一化），
        于是"哪些量该换算"这件事变成靠运气。
        """

        centre = _require_vector(centre_mm, "centre_mm")
        return cls(
            radius_m=metres_from_millimetres(radius_mm),
            centre_m=(
                metres_from_millimetres(centre[0]),
                metres_from_millimetres(centre[1]),
                metres_from_millimetres(centre[2]),
            ),
            normal=normal,
            turns=turns,
            current_a=current_a,
        )

    @classmethod
    def from_coaxial(cls, loop: CircularLoop) -> PlacedCircularLoop:
        """把一条共轴回路搬到一般位形里：圆心``(0,0,z)``、法向``+z``。

        **这是"一般位形退化到同轴"那条金标判据的接口**：同一条回路两种表示，
        一条走Maxwell闭式、一条走本模块的双重求积，两个数必须对上。
        单向的——一般位形回不去`CircularLoop`（它没有位置和倾角可放）。
        """

        if not isinstance(loop, CircularLoop):
            raise ElectromagneticsError(f"from_coaxial只接受CircularLoop：{loop!r}")
        return cls(
            radius_m=loop.radius_m,
            centre_m=(0.0, 0.0, loop.axial_position_m),
            normal=(0.0, 0.0, 1.0),
            turns=loop.turns,
            current_a=loop.current_a,
        )

    def plane_frame(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """回路平面内的一组正交单位基``(u, v)``，满足``u × v = normal``。

        种子轴取``normal``**分量绝对值最小**的那根坐标轴：这样叉积的两个向量
        夹角最远离0，归一化前的模最大（≥ 1/√3），**因此这一步不会放大误差**。
        取"第一个非零分量"那种写法在``normal ≈ (1, 1e-9, 0)``上就会退化。

        标架的**相位**是规范自由度：换一组``(u, v)``只是把两条回路各自的采样
        转一个角，求积值只在收敛量级上变。协议门里有一条整体刚体旋转不变性
        判据钉住这一点（旋转会同时换掉两条回路的标架）。
        """

        normal_x, normal_y, normal_z = self.normal
        magnitudes = (abs(normal_x), abs(normal_y), abs(normal_z))
        smallest = min(magnitudes)
        if magnitudes[0] == smallest:
            seed = (1.0, 0.0, 0.0)
        elif magnitudes[1] == smallest:
            seed = (0.0, 1.0, 0.0)
        else:
            seed = (0.0, 0.0, 1.0)
        u_x = seed[1] * normal_z - seed[2] * normal_y
        u_y = seed[2] * normal_x - seed[0] * normal_z
        u_z = seed[0] * normal_y - seed[1] * normal_x
        u_norm = math.sqrt(u_x * u_x + u_y * u_y + u_z * u_z)
        u = (u_x / u_norm, u_y / u_norm, u_z / u_norm)
        v = (
            normal_y * u[2] - normal_z * u[1],
            normal_z * u[0] - normal_x * u[2],
            normal_x * u[1] - normal_y * u[0],
        )
        return u, v

    def segment_arc_length_m(self, segments: int) -> float:
        """离散步长``h = 2πR/N``——分辨比那张表的分母。"""

        return 2.0 * math.pi * self.radius_m / _require_segments(segments, "segments")

    def magnetic_area_m2(self) -> float:
        """回路的有向面积大小``πR²``（乘``turns``），偶极近似要用。"""

        return self.turns * math.pi * self.radius_m * self.radius_m


def filament_samples(
    loop: PlacedCircularLoop, segments: int
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """回路的``segments``个中点样本：``(位置, 单位切向)``。

    **只依赖这一条回路**——这正是互易性能逐位成立的前提之一：
    两个方向拿到的样本必须是同一串字节，而不是"另一条回路参与算出来的"。
    """

    count = _require_segments(segments, "segments")
    u, v = loop.plane_frame()
    radius = loop.radius_m
    centre_x, centre_y, centre_z = loop.centre_m
    samples = []
    for index in range(count):
        angle = 2.0 * math.pi * (index + 0.5) / count
        cosine = math.cos(angle)
        sine = math.sin(angle)
        position = (
            centre_x + radius * (u[0] * cosine + v[0] * sine),
            centre_y + radius * (u[1] * cosine + v[1] * sine),
            centre_z + radius * (u[2] * cosine + v[2] * sine),
        )
        tangent = (
            -u[0] * sine + v[0] * cosine,
            -u[1] * sine + v[1] * cosine,
            -u[2] * sine + v[2] * cosine,
        )
        samples.append((position, tangent))
    return tuple(samples)


def filament_resolution_ratio(
    loop_a: PlacedCircularLoop,
    loop_b: PlacedCircularLoop,
    *,
    segments_a: int,
    segments_b: int,
) -> float:
    """分辨比``δ/h``：样本点间最小距离 除以 **较粗**那条的弧长步长。

    这是本模块唯一决定近距精度的数（模块docstring第三节的实测表），
    公开它是为了让"这次求值能算到几位"可被调用方自己判断，
    而不是只能相信一句"应该够准了"。
    """

    samples_a = filament_samples(loop_a, segments_a)
    samples_b = filament_samples(loop_b, segments_b)
    closest = float("inf")
    for position_a, _ in samples_a:
        a_x, a_y, a_z = position_a
        for position_b, _ in samples_b:
            delta_x = a_x - position_b[0]
            delta_y = a_y - position_b[1]
            delta_z = a_z - position_b[2]
            squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
            if squared < closest:
                closest = squared
    step = max(
        loop_a.segment_arc_length_m(segments_a), loop_b.segment_arc_length_m(segments_b)
    )
    return math.sqrt(closest) / step


def neumann_mutual_inductance_h(
    loop_a: PlacedCircularLoop,
    loop_b: PlacedCircularLoop,
    *,
    segments_a: int,
    segments_b: int,
) -> float:
    """一般位形两条圆形回路的互感（亨利），**含匝数**。

    ``segments_a``/``segments_b``是**必填**的：离散粗细决定精度，
    而"默认多少段"这个选择只有调用方知道自己要几位——
    给一个默认值等于替调用方做了一个它看不见的精度决定
    （模块docstring第一、三节两张实测表就是选它的依据）。

    **符号是有意义的**：``M < 0``表示两条回路的正方向互相反向链磁通
    （例如共面并排的两条同向回路），不是错误。
    """

    samples_a = filament_samples(loop_a, segments_a)
    samples_b = filament_samples(loop_b, segments_b)
    arc_a = loop_a.segment_arc_length_m(segments_a)
    arc_b = loop_b.segment_arc_length_m(segments_b)
    # 项与最小距离**一趟算完**：分辨比是O(N_a·N_b)的量，
    # 单独调一次`filament_resolution_ratio`会让主路的代价翻倍。
    terms, closest_m = _pair_terms_and_closest(samples_a, samples_b)
    ratio = closest_m / max(arc_a, arc_b)
    if ratio < RESOLUTION_RATIO_MIN:
        raise ElectromagneticsError(
            f"分辨比δ/h={ratio!r}低于{RESOLUTION_RATIO_MIN!r}：此处求积的相对误差"
            f"已在1e-3量级，而返回值有限、量级正确、符号正确，**没有任何东西看得出它错**。"
            "拒跑。要么把segments加大（h变小、δ/h变大），要么把两条回路分开；"
            "标定表见RESOLUTION_RATIO_CALIBRATION"
        )
    single_turn = NEUMANN_PREFACTOR_H_PER_M * (arc_a * arc_b) * math.fsum(terms)
    return (loop_a.turns * loop_b.turns) * single_turn


def neumann_condition_number(
    loop_a: PlacedCircularLoop,
    loop_b: PlacedCircularLoop,
    *,
    segments_a: int,
    segments_b: int,
) -> float:
    """双重和的条件数``κ = Σ|项| / |Σ项|``——"这次求值能算到几位"的那个数。

    相对误差地板约``κ·ε``（``ε = 2.2e-16``）。远场判据必须报它：
    ``M ∝ 1/d³``的偏差本身随d衰减，而κ随d²增长，
    **两条曲线迟早相交，那一点就是这条路的远场边界**。

    与`neumann_mutual_inductance_h`分开是有代价的（样本与项算两遍），
    换来的是主路上不带任何只为诊断存在的返回值。诊断量按需算，
    这与`collision`的`confidence`是同一条纪律的两种形态。
    """

    ratio = filament_resolution_ratio(
        loop_a, loop_b, segments_a=segments_a, segments_b=segments_b
    )
    if ratio < RESOLUTION_RATIO_MIN:
        raise ElectromagneticsError(
            f"分辨比δ/h={ratio!r}低于{RESOLUTION_RATIO_MIN!r}：条件数在未分辨的构型上"
            "没有意义（它诊断的是相消，不是离散误差）。拒跑"
        )
    terms, _ = _pair_terms_and_closest(
        filament_samples(loop_a, segments_a), filament_samples(loop_b, segments_b)
    )
    total = math.fsum(terms)
    if total == 0.0:
        raise ElectromagneticsError(
            "双重和恰为0（例如两条回路正交的零磁通构型）：条件数按定义要除以它。"
            "拒跑——这类构型的误差要按**绝对**尺度判，不是相对尺度"
        )
    return math.fsum([abs(term) for term in terms]) / abs(total)


def dipole_mutual_inductance_general_h(
    loop_a: PlacedCircularLoop, loop_b: PlacedCircularLoop
) -> float:
    """一般位形的远场偶极近似

        ``M ≈ (μ₀·A₁·A₂/(4π·d³))·[3·(n̂₁·d̂)(n̂₂·d̂) − n̂₁·n̂₂]``

    ``A = N·πR²``是含匝数的有向面积，``d̂``是两圆心连线方向。
    共轴时方括号为``3−1 = 2``，退化成`inductance.dipole_mutual_inductance_h`
    （两者是同一式的两种写法，案例里有一条判据钉住它们逐点一致）。

    **这是一条近似不是闭式**，公开它与`inductance.dipole_mutual_inductance_h`
    同理：让"数值积分在远场退化到它"成为可算的判据。它同时是本模块**符号**的
    独立见证——方括号在共面并排时是``−1``，即``M < 0``，
    而一个偷偷取了绝对值的实现在别的判据上全绿。
    """

    centre_a = loop_a.centre_m
    centre_b = loop_b.centre_m
    offset = (
        centre_b[0] - centre_a[0],
        centre_b[1] - centre_a[1],
        centre_b[2] - centre_a[2],
    )
    distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2)
    if distance == 0.0:
        raise ElectromagneticsError(
            "两条回路同心（圆心重合）：偶极近似按定义只在d ≫ R时有意义，d=0处发散。拒跑"
        )
    direction = (offset[0] / distance, offset[1] / distance, offset[2] / distance)
    normal_a = loop_a.normal
    normal_b = loop_b.normal
    projection_a = _dot(normal_a, direction)
    projection_b = _dot(normal_b, direction)
    angular = 3.0 * projection_a * projection_b - _dot(normal_a, normal_b)
    return (
        NEUMANN_PREFACTOR_H_PER_M
        * loop_a.magnetic_area_m2()
        * loop_b.magnetic_area_m2()
        * angular
        / (distance * distance * distance)
    )


def _pair_terms_and_closest(
    samples_a: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
    samples_b: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> tuple[list[float], float]:
    """全部``(t_i·t_j)/|p_i − p_j|``项，外加样本点间的**最小距离**。

    **逐位对称**：交换两个参数得到的是同一个多重集（每一项的两个因子
    在IEEE754下可交换，距离只差一次精确取反），只是枚举次序反了。
    `math.fsum`对置换不变，互易的零容差就落在这两句上。
    最小距离也对称（``min``与枚举次序无关），所以两个方向连拒跑与否都一致。

    样本点**重合**（距离恰为0）在这里失败关闭，而不是让除法抛
    `ZeroDivisionError`：同一条丝状回路上的Neumann积分对数发散是**物理事实**，
    报错文本要说的是这件事，不是"除以零"。
    """

    terms: list[float] = []
    closest_squared = math.inf
    for position_a, tangent_a in samples_a:
        a_x, a_y, a_z = position_a
        tangent_a_x, tangent_a_y, tangent_a_z = tangent_a
        for position_b, tangent_b in samples_b:
            delta_x = a_x - position_b[0]
            delta_y = a_y - position_b[1]
            delta_z = a_z - position_b[2]
            squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
            if squared < closest_squared:
                closest_squared = squared
            if squared == 0.0:
                raise ElectromagneticsError(
                    "两条回路有重合的样本点：同一条丝状回路上的Neumann积分对数发散，"
                    "离散之后**不会报错、只会给一个随分段数变化的有限数**。"
                    "自感要导线截面半径（GMD一类的正则化要它），本模块不做"
                    "（决策0092第四节的GAP）。拒跑"
                )
            dot = (
                tangent_a_x * tangent_b[0]
                + tangent_a_y * tangent_b[1]
                + tangent_a_z * tangent_b[2]
            )
            terms.append(dot / math.sqrt(squared))
    return terms, math.sqrt(closest_squared)


def _dot(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _require_segments(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ElectromagneticsError(f"{name}必须是整数：{value!r}")
    if value < SEGMENTS_MIN:
        raise ElectromagneticsError(
            f"{name}必须≥{SEGMENTS_MIN}：{value!r}——少于三段的"
            "'闭合回路'在几何上不存在。**注意这不是精度下限**，精度看分辨比"
        )
    return value


def _require_vector(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ElectromagneticsError(f"{name}必须是三个实数的元组：{value!r}")
    if len(value) != 3:
        raise ElectromagneticsError(
            f"{name}必须恰有三个分量，收到{len(value)}个：{value!r}"
        )
    return (
        _require_finite(value[0], f"{name}[0]"),
        _require_finite(value[1], f"{name}[1]"),
        _require_finite(value[2], f"{name}[2]"),
    )


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ElectromagneticsError(f"{name}必须是实数：{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ElectromagneticsError(f"{name}必须是有限值：{value!r}")
    return number


__all__ = [
    "NEUMANN_PREFACTOR_H_PER_M",
    "NORMAL_NORM_MIN",
    "PlacedCircularLoop",
    "RESOLUTION_RATIO_CALIBRATION",
    "RESOLUTION_RATIO_MIN",
    "SEGMENTS_MIN",
    "dipole_mutual_inductance_general_h",
    "filament_resolution_ratio",
    "filament_samples",
    "neumann_condition_number",
    "neumann_mutual_inductance_h",
]
