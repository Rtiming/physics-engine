"""卷绕堆积律——**匝 → 层 → 有效半径 → 已绕长度**（力学域，决策0093）。

守能力位S5.3（线圈盘几何与半径随匝数生长）与S5.4（喂料与移动前沿）里
"匝数与半径生长的接线"那一条。单位：长度一律**mm**（本仓口径）。

## 开工第一件事：本仓已经有两份半径式子了

不是零，是两份，而且**它们的自变量不是同一个**：

| 在哪 | 式子 | 自变量 | 它回答什么 |
|---|---|---|---|
| `drives.SpoolTension.radius_mm` | ``R = R₀ + n·t`` | **匝**``n`` | 这卷现在拉得多紧（``T = M/R``） |
| `modelgen.generate_spool` | ``R_eff = (r₀ + m·τ)·L`` | **层**``m`` | 这卷长什么样（产出`FiniteCylinder`） |

**本模块不是第三份，它是那两份中间缺的那一段**：``匝 → 层``的换算
（``turns_per_layer``）与**堆积因子**，外加``长度 ↔ 匝``。
判据把这句话钉死：

* ``turns_per_layer = 1``、``packing_factor = 1``、``layer_advance = "continuous"``
  时，`WindingPack.radius_mm`与`drives.SpoolTension.radius_mm`**逐位相同**
  （同一个浮点数，不是"在容差内"）——`tests/test_winding.py`有一道门判它；
* `modelgen.generate_spool`那一份走的是**无量纲比值**、且它的声明指纹是冻结的
  金标（`cases/generator_determinism`），**本模块一个字节都不动它**；
  两者的对应关系是``m = layers_at(n)``、``R_eff = radius_mm(n)/L``，
  同样有一道门判（取``layer_advance = "stepped"``，因为几何层的``wound_layers``
  是**整数层**）。

**为什么不把`drives`那一份提出来改成调用本模块**：本批的分槽纪律里
`drives.py`是只读的（0089第三节：每条轨只碰自己的新文件）。
"一条律住一个地方"这件事因此**没有做完**——登记成GAP，触发条件写在本页末尾
与决策0093第五节。逐位相同那道门是这段时间里的替代品：
**它保证两份式子今天说的是同一句话，但它保证不了明天有人只改其中一份。**

## 半径长在层上，不长在匝上

一匝不一定升一层。带材在盘上排开``turns_per_layer``匝才铺满一层宽度，
下一匝才叠到上一层的背上。于是：

    有效层厚   t_eff = t / packing_factor        （堆积因子：材料占层厚的比例）
    层数       m(n)  = n / turns_per_layer       （连续式）
              m(n)  = ⌊n / turns_per_layer⌋      （台阶式）
    半径       R(n)  = R₀ + t_eff · m(n)

``packing_factor``取``(0, 1]``：``1``是理想密排（``t_eff = t``，且**除以1.0在浮点上
无舍入**，这是上面那条"逐位相同"能成立的原因之一）；小于1表示层里有绝缘、
空气或排布不齐，同样厚度的材料占掉更厚的一层。

**两种``layer_advance``不是口味，是两种真机**：

* ``"continuous"``——**盘香式**（pancake / edge-wound）：每一匝就是新的一层，
  半径随匝**连续**长。HTS带材绕单饼线圈是这一类，也是场景⑤的默认；
* ``"stepped"``——**排绕式**（traverse-wound）：一层里排``turns_per_layer``匝，
  层内半径**不动**，跨层才跳一格。`modelgen.generate_spool`的``wound_layers``
  说的是这一类。

## 长度是半径的积分，不是半径乘匝数

    L(n) = ∫₀ⁿ 2π·R(s) ds

**不是``2π·R(n)·n``，也不是``2π·R₀·n``**。两个都错，而且错得像对的：
前者把整卷都当最外一匝算，后者把整卷都当最里一匝算。

对每一匝取中点半径``R(k+½)``，则

    L(n) = 2π · Σ_{k<n} R(k+½)          （逐匝周长之和）

与上面那条积分**恒等**——``R(s)``在一匝之内要么线性（连续式，中点法则对线性
被积函数精确）、要么常数（台阶式），两种情形中点法则都不是近似。
`tests/test_winding.py`把这条做成**零容差恒等式**：取二进制精确的参数，
`sum(turn_mean_radius_mm(k))`与`radius_integral_mm(n)`判``==``。

**这条恒等式的价值在于它对"半径更新早了一匝还是晚了一匝"有分辨力**：
把``R(k+½)``写成``R(k)``或``R(k+1)``，单点检查照过（差一匝的半径差只有一个``t``，
在任何一条"rel 1e-6"的判据下都是绿的），而这条求和当场差``n·t/2·2π``。
必红矩阵里那两条就是它抓的。

**2π是两边的公因子**：把它乘进去之后剩下的差是``2π``这个浮点常数的舍入，
**那不是守恒的问题**。所以零容差那道门判的是**提出2π之后的半径积分**，
带``2π``的长度那一版给实测相对偏差（约1e-16）。**这句话必须说清楚，
否则"零容差"就是一句吹的。**

## 本模块不做什么

* **不做欧拉式材料注入**。`feed.FeedFront`是**拉格朗日**的——每个节点始终是
  同一块材料，喂料只是往前接长度；本模块顺着它，把"喂进来的长度"换成
  "盘上的匝数与半径"。**带材从轮面上流过、边界随材料移动**（WDS `research/05`
  第三节那条"当前最大的单项缺口"）**本模块给不了**，横漂稳态与绞盘全滑
  同样给不了。这是一次新裁决不是接线，登记成GAP（决策0093第五节）；
* **不做自接触**（S5.2）：新绕的一匝压在已绕的匝上要网格/连续体窄相；
* **不做力**：``T = M/R``在`drives`，本模块只给``R``；
* **不做形状**：产出`FiniteCylinder`是`modelgen.generate_spool`的事；
* **不做真实阿基米德螺线的弧长**。本模块的``L(n)``是**同心圆理想化**
  （每匝是一个闭合圆）。真螺线的弧长闭式是`cases/spool_winding_growth`的
  **独立金标**，两者的相对偏差在``t/R₀ → 0``时以**二阶**收敛——那条案例量的
  就是这个阶。**把螺线弧长搬进本模块会让金标与被验量出自同一支笔**，
  所以它留在案例那边。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.feed import FeedFront

#: ``2π``。整圈周长``2πR``用它，**只算一次**——两处各写一次``2*math.pi``
#: 在浮点上是同一个数，但在阅读上是两处各自的定义。
TAU = 2.0 * math.pi

#: ``layer_advance``的两个合法值。**没有默认的"随便"档**：
#: 盘香式与排绕式在同一组参数下给出不同的半径，猜错一个方向就是一整层的误差。
LAYER_ADVANCE: frozenset[str] = frozenset({"continuous", "stepped"})


class WindingError(ValueError):
    """卷绕堆积律的一切失败关闭。"""


@dataclass(frozen=True)
class WindingPack:
    """一卷带材的堆积声明：筒半径、带厚、每层匝数、堆积因子与层推进形制。

    全部量为**声明期常量**——匝数``n``是入参不是字段，
    因为"这卷现在绕到第几匝"是过程量，塞进声明会让同一卷在不同时刻是不同的对象
    （0050对锚点裁过同一件事：**活动与否是向量里的一个值，不是一次布局变更**）。
    """

    barrel_radius_mm: float
    tape_thickness_mm: float
    turns_per_layer: int = 1
    packing_factor: float = 1.0
    layer_advance: str = "continuous"

    def __post_init__(self) -> None:
        for name in ("barrel_radius_mm", "tape_thickness_mm"):
            value = getattr(self, name)
            if not (value > 0.0 and math.isfinite(value)):
                raise WindingError(f"{name} must be positive and finite: {value!r}")
        if isinstance(self.turns_per_layer, bool) or not isinstance(self.turns_per_layer, int):
            raise WindingError(
                f"turns_per_layer must be an int: {self.turns_per_layer!r} —— "
                "半整数匝铺满一层不是一种绕法，是一个没定住的声明"
            )
        if self.turns_per_layer < 1:
            raise WindingError(
                f"turns_per_layer must be at least 1: {self.turns_per_layer!r}"
            )
        if not (
            math.isfinite(self.packing_factor) and 0.0 < self.packing_factor <= 1.0
        ):
            raise WindingError(
                f"packing_factor must be in (0, 1]: {self.packing_factor!r} —— "
                "大于1表示材料占的层厚比它自己还薄，那是把带材压进了自己里面"
            )
        if self.layer_advance not in LAYER_ADVANCE:
            raise WindingError(
                f"layer_advance must be one of {sorted(LAYER_ADVANCE)}: "
                f"{self.layer_advance!r}"
            )

    # ------------------------------------------------------------------ 半径

    @property
    def effective_layer_thickness_mm(self) -> float:
        """``t_eff = t / packing_factor``——**一层占掉的径向厚度**。

        ``packing_factor = 1.0``时**除法无舍入**，``t_eff``与``tape_thickness_mm``
        逐位相同。"与`drives`逐位相同"那道门靠的就是这一条。
        """

        return self.tape_thickness_mm / self.packing_factor

    def _assert_turns(self, turns: float) -> None:
        if isinstance(turns, bool) or not isinstance(turns, (int, float)):
            raise WindingError(f"turns must be a real number: {turns!r}")
        if not (turns >= 0.0 and math.isfinite(turns)):
            raise WindingError(
                f"turns must be finite and nonnegative: {turns!r} —— "
                "负匝数是把带材从盘上倒着抽走，那是另一个工况（本模块不假装能算它）"
            )

    def layers_at(self, turns: float) -> float:
        """已铺满的层数``m(n)``。连续式给分数层，台阶式给整数层。"""

        self._assert_turns(turns)
        quotient = turns / self.turns_per_layer
        if self.layer_advance == "stepped":
            return float(math.floor(quotient))
        return quotient

    def radius_mm(self, turns: float) -> float:
        """有效半径``R(n) = R₀ + t_eff·m(n)``。

        ``turns_per_layer = 1``、``packing_factor = 1.0``、``layer_advance
        = "continuous"``三条同时成立时，本式与`drives.SpoolTension.radius_mm`
        **给出同一个浮点数**（``R₀ + t·n``与``R₀ + n·t``，浮点乘法可交换）。
        """

        return self.barrel_radius_mm + self.effective_layer_thickness_mm * self.layers_at(
            turns
        )

    def turn_mean_radius_mm(self, index: int) -> float:
        """第``index``匝（**0起**）的平均半径``R(index + ½)``。

        **不是``R(index)``也不是``R(index+1)``**：那两个各自把这一匝当成它开始时
        或结束时的半径，逐匝求和之后与闭式差``n·t/2``——这正是模块文档里
        "早了一匝还是晚了一匝"那条恒等式抓的东西。

        台阶式下``⌊(index+½)/tpl⌋ = ⌊index/tpl⌋``（``index``与``tpl``都是整数），
        于是中点与层内任何一点同半径，两种形制共用这一个式子。
        """

        if isinstance(index, bool) or not isinstance(index, int):
            raise WindingError(f"turn index must be an int: {index!r}")
        if index < 0:
            raise WindingError(f"turn index must be nonnegative: {index!r}")
        return self.radius_mm(index + 0.5)

    def turn_length_mm(self, index: int) -> float:
        """第``index``匝的材料长度``2π·R(index+½)``。"""

        return TAU * self.turn_mean_radius_mm(index)

    # ------------------------------------------------------------------ 长度

    def _layer_radius_integral(self, layers: int) -> float:
        """绕满``layers``个**整层**之后的半径积分``∫R ds``（台阶式专用）。

        ``tpl·(m·R₀ + t_eff·m(m−1)/2)``——层内半径不动，所以整层就是
        "每层匝数 × 该层半径"再对层求和，等差数列求和给出闭式。
        """

        thickness = self.effective_layer_thickness_mm
        return self.turns_per_layer * (
            layers * self.barrel_radius_mm
            + thickness * layers * (layers - 1) * 0.5
        )

    def radius_integral_mm(self, turns: float) -> float:
        """``∫₀ⁿ R(s) ds``——**已绕长度除以2π**。

        单独暴露它而不是只给`wound_length_mm`，是因为``2π``是恒等式两边的
        公因子：**零容差那道门判的是本函数**，带``2π``的那一版判实测相对偏差。
        """

        self._assert_turns(turns)
        if self.layer_advance == "continuous":
            slope = self.effective_layer_thickness_mm / self.turns_per_layer
            return self.barrel_radius_mm * turns + slope * turns * turns * 0.5
        layers = math.floor(turns / self.turns_per_layer)
        remainder = turns - layers * self.turns_per_layer
        return self._layer_radius_integral(layers) + remainder * (
            self.barrel_radius_mm + self.effective_layer_thickness_mm * layers
        )

    def wound_length_mm(self, turns: float) -> float:
        """绕上去的材料长度``L(n) = 2π·∫₀ⁿ R(s) ds``。"""

        return TAU * self.radius_integral_mm(turns)

    def turns_at_length_mm(self, length_mm: float) -> float:
        """``L(n) = ℓ``的解——**喂进来这么长的料，盘上是几匝**。

        连续式是一个二次方程：``(c/2)n² + R₀n = S``（``S = ℓ/2π``、``c = t_eff/tpl``），
        取**有理化后的那一支**``n = 2S / (R₀ + √(R₀² + 2cS))``而不是
        ``(−R₀ + √(R₀²+2cS))/c``：后者在``c → 0``（薄带）时是两个几乎相等的数相减，
        **相对误差随``R₀/(cS)``发散**，而这正是场景⑤的常用区（十几匝、带厚
        比筒径小三个数量级）。**这条写法今天没有被判据验到位**——
        必红实测里换成坏的那一支只红2处，两处参数都还没进到严重相消区；
        登记成GAP（决策0093第五节），不假装它验过了。

        台阶式没有全局二次式（半径是分段常数），走"先定整层再定层内余量"：
        由**连续式的解**起手猜整层数，再用两条单调修正把它推到真值——
        ``_layer_radius_integral``对层数单调增，所以修正必终止。
        """

        if isinstance(length_mm, bool) or not isinstance(length_mm, (int, float)):
            raise WindingError(f"length must be a real number: {length_mm!r}")
        if not (length_mm >= 0.0 and math.isfinite(length_mm)):
            raise WindingError(
                f"length must be finite and nonnegative: {length_mm!r} —— "
                "负的已绕长度不是一个工况，是一次记账错误"
            )
        target = length_mm / TAU
        radius = self.barrel_radius_mm
        if self.layer_advance == "continuous":
            slope = self.effective_layer_thickness_mm / self.turns_per_layer
            return 2.0 * target / (
                radius + math.sqrt(radius * radius + 2.0 * slope * target)
            )
        thickness = self.effective_layer_thickness_mm
        slope = thickness / self.turns_per_layer
        guess = 2.0 * target / (radius + math.sqrt(radius * radius + 2.0 * slope * target))
        layers = max(0, math.floor(guess / self.turns_per_layer))
        while layers > 0 and self._layer_radius_integral(layers) > target:
            layers -= 1
        while self._layer_radius_integral(layers + 1) <= target:
            layers += 1
        remainder = (target - self._layer_radius_integral(layers)) / (
            radius + thickness * layers
        )
        if remainder < 0.0:
            remainder = 0.0
        return layers * self.turns_per_layer + remainder

    def front_angular_rate_rad_s(self, turns: float, line_speed_mm_s: float) -> float:
        """落位点的角速度``ω = v / R(n)``——**前沿推进的运动学闭式**。

        由``dL/dt = v``与``L(n) = ∫2πR``直接得：``v = 2πR·dn/dt``，
        而``ω = 2π·dn/dt``，两边的``2π``抵掉，剩``ω = v/R``。

        它与`turns_at_length_mm`是同一条律的两种写法（一条积分、一条微分），
        案例里用**中心差分**把后者数值求导去对前者，实测二阶收敛——
        这道门抓的是"半径在积分里长、在角速度里没长"这种半边接线。
        """

        if not math.isfinite(line_speed_mm_s) or line_speed_mm_s < 0.0:
            raise WindingError(
                f"line speed must be finite and nonnegative: {line_speed_mm_s!r} —— "
                "负线速度是在退绕，本模块的匝数只增不减"
            )
        return line_speed_mm_s / self.radius_mm(turns)


@dataclass(frozen=True)
class WindingFront:
    """把**拉格朗日**喂料前沿接到卷绕堆积律上——S5.4那条"尚未接在一起"。

    S5.4的`missing`原文点名："**另无匝数与半径生长的接线**
    （那是`drives.SpoolTension`的``turns``，两者尚未接在一起）"。本类是那根线。

    ## 材料守恒记在**整数段**上，不记在浮点长度上

    `feed.FeedFront`把带材切成``rest_length_mm``的等长段，
    喂``fed_count``个节点就是喂了``fed_count − 1``段——**一个整数**。
    这些段分两处：还在喂料口与落位点之间的**自由跨距**里，或者已经绕上盘。于是

        segments_fed = segments_in_free_span + segments_on_spool

    是一条**整数恒等式**，判``==``、零容差、与浮点求和次序无关。

    **为什么不把守恒写在长度上**：``k·h``与``(k−1)·h + h``在浮点上不是同一个数
    （``h = 0.1``、``k = 3``时差一个ulp）。把守恒写在浮点长度上，
    "零容差"就只能靠``h``恰好是二进制精确来兜——**那是运气不是形制**。
    段数是整数，长度是段数乘静止段长，守恒因此不依赖``h``的二进制形状。
    `tests/test_winding.py`两组参数各判一次：``h = 0.25``（二进制精确）下
    连浮点长度都逐位守恒，``h = 0.1``下浮点长度差1e-16而整数恒等式**照样``==``**。

    ## ``free_span_segments``为什么是整数段而不是一个长度

    因为它要与段数直接相减。给一个长度就得先除以``rest_length_mm``再取整，
    而那一步的取整方向（早半段还是晚半段）会变成一个没人写下来的约定——
    plans/09教训一记的"两处各说各的"就是这么长出来的。

    落位前（前沿还没走到盘上）``segments_on_spool``恰为``0``而不是负数：
    ``segments_in_free_span = min(fed − 1, free_span_segments)``。
    **这不是钳位，是记账**——那些料真的还在跨距里，恒等式两边都认。
    """

    pack: WindingPack
    front: FeedFront
    free_span_segments: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.pack, WindingPack):
            raise WindingError(f"pack must be a WindingPack: {self.pack!r}")
        if not isinstance(self.front, FeedFront):
            raise WindingError(f"front must be a FeedFront: {self.front!r}")
        if isinstance(self.free_span_segments, bool) or not isinstance(
            self.free_span_segments, int
        ):
            raise WindingError(
                f"free_span_segments must be an int: {self.free_span_segments!r} —— "
                "半段材料停在跨距里说明段长与跨长各说各的（见类文档）"
            )
        if self.free_span_segments < 0:
            raise WindingError(
                f"free_span_segments must be nonnegative: {self.free_span_segments!r}"
            )
        if self.free_span_segments > self.front.node_budget - 1:
            raise WindingError(
                f"free_span_segments {self.free_span_segments!r} 超过整卷的段预算 "
                f"{self.front.node_budget - 1} —— 这一卷全喂完也够不到盘，"
                "而那必须在声明期发现"
            )

    def segments_fed(self, fed_count: int) -> int:
        """已经离开料卷的段数（整数）——**转发给`feed.FeedFront`，不自己算**。

        自己写一遍``fed_count − 1``就是plans/09教训一那种"两处各说各的"：
        两边今天相同，明天有人只改一边。
        """

        return self.front.fed_segment_count(fed_count)

    def segments_in_free_span(self, fed_count: int) -> int:
        """还悬在喂料口与落位点之间的段数（整数）。"""

        return min(self.segments_fed(fed_count), self.free_span_segments)

    def segments_on_spool(self, fed_count: int) -> int:
        """已经绕上盘的段数（整数）——**守恒记在这个量上**。"""

        return self.segments_fed(fed_count) - self.segments_in_free_span(fed_count)

    def length_on_spool_mm(self, fed_count: int) -> float:
        """盘上的材料长度``segments_on_spool · rest_length``。"""

        return self.segments_on_spool(fed_count) * self.front.rest_length_mm

    def turns_on_spool(self, fed_count: int) -> float:
        """盘上的匝数——把盘上的长度喂给`WindingPack.turns_at_length_mm`。"""

        return self.pack.turns_at_length_mm(self.length_on_spool_mm(fed_count))

    def radius_on_spool_mm(self, fed_count: int) -> float:
        """当前有效半径——**这就是`drives.SpoolTension`要的那个``turns``的出处**。

        接法是``drives.SpoolTension(...).tension_n(torque, turns=front.turns_on_spool(k))``：
        本模块给匝数，`drives`给力。**两处的``R₀``与``t``必须是同一对数**，
        而今天没有任何一道门守着这一点（两个类各自持有自己的一份）——
        登记成GAP，触发条件见决策0093第五节。
        """

        return self.pack.radius_mm(self.turns_on_spool(fed_count))


__all__ = [
    "LAYER_ADVANCE",
    "TAU",
    "WindingError",
    "WindingFront",
    "WindingPack",
]
