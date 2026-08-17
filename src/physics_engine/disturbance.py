"""扰动通道：臂动与人手触碰怎么进跨段张力（力学域，决策0071）。

守plans/15第三节阶段一的1.3与1.4。本模块**一条新物理都不加**，
它做的是把`laydown`（落位点几何）与`transport`（跨段输运）之间那根线接上，
并给"有人突然碰了一下带材"一个**显式的**外扰接口。

## 一、先把一件事判掉：``L_geo``到底变不变

`transport`第0066轮把跨段几何长度取成常数，并在第九、十节把这件事登记成
"欠账"，措辞是"**真机上跨长逐样点变**"、"轨道C给出``L_geo(t)``那天改成入参"。

**那句措辞是错的，本模块不照它做。**

plans/14第3.3节记的用户2026-08-14口述订正了场景：张力机不动、入带点是
**世界系里固定的一个点**、在途自由段的两个端点（最后那只导向轮的出带点、入带点）
**都是世界系固定的**。两个固定点之间的直线段长度**是一个常数**——
这不是简化，是场景本身给的。轨道C（0067）已经把它冻在类型上：
`laydown.FreeSpanGeometry.length_mm()`连``t``都收不到。

于是：

* **臂动不改``L_geo``。** 臂动改的是**落位点的几何**——落位点在槽上前进的速率
  （所需送带率）随槽的曲率与臂的转速变。那条通道走`FreeSpan`的**收线端速度**，
  不走几何长度。这就是1.3；
* **``L_geo``只有一条真的会变的通道：有东西横向顶在跨段上。** 两端仍固定，
  但带材走的不再是直线而是折线，**折线比直线长**。这就是1.4，
  而那一份增量在`transport.FreeSpan.path_length_mm`里叫``path_excess_mm``。

**两条通道不许互相折算。** 路径增量进的是应变的分子，收线端速度进的是长度账的
导数；把前者折算成"等效收线速度"要乘``L_mat/L_path``，而丢掉那个``O(ε)``因子
正是本仓最怕的那种静默错（0066第五节那条"应变的分母是``L_mat``不是``L_geo``"同源）。

## 二、1.3的机制：``v_required = Ω_tangent / κ``

落位点必须待在那个固定入带点上，所以**这一瞬要放出多少带材**＝
线圈上"此刻正经过入带点的那个材料点"的速率。把它写成落位点在槽上的弧长速率
``σ'(t)``，再用Frenet关系换一次：落位点处槽切向的转速是``Ω_tan = κ(σ)·σ'``，于是

    σ'(t) = Ω_tan(t) / κ(σ(t))

**这一条式子就是1.3的全部内容**，它同时解释了那个必须有的退化档：

* **平面圆**（κ恒定）＋臂把切向匀速转 ⟹ ``σ'``**恒定** ⟹ 扰动**恒为零**；
* 槽的曲率沿路径变 ⟹ 同样一条匀速转切向的臂运动，``σ'``就在变 ⟹ 张力被扰动。

**扰动来自"曲率在变"，不是来自"切向在转"本身。** 切向转得再快，只要κ是常数，
送带率就是常数，张力一动不动。这一条与直觉相反，值得单独写一行。

## 三、1.4的形制：方向、作用点、起止时刻**逐条显式**

本仓吃过一次亏：`PenaltyAnnulusLimit`把法兰朝向**编码在坐标符号里**，
于是单元门永远抓不到，端到端跑一次才炸（`winding_line_endtoend`案例页记着）。
所以`TransverseTouch`的每一条都是**独立字段**：

* ``push_direction``是一个显式单位向量，**必须与跨段正交**，构造期当场判；
* ``offset_mm``**必须为正**——方向由``push_direction``携带，
  **不许用它的符号兼职表示方向**。这一条就是那次教训的直接产物；
* ``station_from_guide_mm``写明**从哪一端量**（导轮端），不靠约定记忆；
* ``start_time_s``与``end_time_s``两端都显式，区间取**左闭右开**并写在文档里。

## 四、账：横向冲量与材料长度账之间有一条**恒等式**

不是"两边都记一下看着差不多"，是一条能逐位对的式子。设触碰窗口内
``g``（横向力的几何因子）恒定、制动力矩恒定，对半隐式Euler的两条更新求和：

    dt·R·ΣT_n = (J/1000 − c·dt)·Δω + M·N·dt + (c/R)·(ΔL_mat + dt·Σv_收线)

于是横向冲量

    ∫F dt = g·dt·ΣT_n = (g/R)·[(J/1000 − c·dt)·Δω + M·N·dt + (c/R)·(ΔL_mat + dt·Σv)]

**那个``−c·dt``是半隐式的指纹**：长度账用的是**步末**转速而力矩账用的是**步首**，
两者差一个``dt·Δω``。写成``J/1000``（即忽略它）在``dt → 0``时看不出来，
而它恰恰是"把半隐式误当显式"的捕手。

## 五、本模块明确不做的

* **不建闭环**。控制器可插拔与真闭环是轨道E（决策0070）的活，`drives.py`归它独占。
  本模块建的是**扰动通道本身**，判据全部走`transport`已有的**开环**链路。
  于是1.4那条"触碰→尖峰→回落"只做到**前两段**：开环下尖峰**不回落**，
  它按``exp(−ζω_n t)``慢慢振荡衰减（真实量级轴承给``ζ = 0.0132``），
  而"有控制器时回落"那一半**没做**，如实登记在决策0071第七节；
* **不做力控触碰**。本模块的原语是**位移控制**（把带材横向按开``δ``），
  横向力是**输出**而不是输入。力控要在每一步解一个关于``δ``的非线性方程
  （``T``自己也随``δ``变），而位移控给出的是精确几何；
* **不做触碰点的接触力学**。手指与带材之间没有摩擦、没有接触斑、没有局部弯曲——
  这里的带材仍是0066那根一维弹簧，只有``EA``；
* **不改跨段的两个端点**。它们是世界系常量，见第一节。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from physics_engine.laydown import FreeSpanGeometry, LaydownModel
from physics_engine.transport import FreeSpan, SpanTransportLoop, SpanTransportSample

Vector3 = tuple[float, float, float]

#: 单位向量与正交性的绝对容差。**与`laydown.FRAME_ORTHONORMAL_ABS_TOL`同量级**——
#: 一个方向在这里合法、装进落位点几何却被判非正交（或反过来）是最难查的那种不一致。
DIRECTION_ABS_TOL = 1.0e-9

#: 两个模块对**同一条线段**的长度必须是同一个数。1e-9 mm是纳米级，
#: 比任何真实几何不确定度小六个数量级——**这不是精度门，是接错线的门**：
#: `transport.FreeSpan`拿到的跨长与`laydown.FreeSpanGeometry`两个端点之间的距离
#: 对不上，说明装配层把两条不同的跨段接在了一起，而那件事今天没有任何东西看得见。
SPAN_LENGTH_MATCH_ABS_TOL_MM = 1.0e-9

#: 触碰的时间剖面白名单，**没有默认值**。今天只有一条被声明：
#: 起止时刻之间横向位移**恒定**，两端各是一次阶跃。
#: 不给"斜坡"或"半正弦"，理由与`transport.RATE_SEMANTICS`同源：
#: 一条真实的人手触碰是什么时间形状，**本仓一次都没有量过**，
#: 而给一个平滑剖面会让人以为那个形状是有出处的。矩形是最诚实的理想化：
#: 它自认是理想化，且两端的阶跃给出**精确**的闭式（见案例页）。
TOUCH_PROFILES: frozenset[str] = frozenset({"rectangular_hold"})


class DisturbanceError(ValueError):
    """扰动通道的一切失败关闭。"""


def _require_finite(value: object, what: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DisturbanceError(f"{what} must be a real number: {value!r}")
    if not math.isfinite(float(value)):
        raise DisturbanceError(f"{what} must be finite: {value!r}")
    return float(value)


def _require_namespace(value: object, prefix: str, what: str) -> str:
    if not isinstance(value, str) or not value.startswith(f"{prefix}/") or value == f"{prefix}/":
        raise DisturbanceError(
            f"{what} must look like {prefix}/<name>: {value!r} —— "
            "命名空间前缀是轴2规则1，不是装饰"
        )
    return value


def _require_vector(value: object, what: str) -> Vector3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise DisturbanceError(f"{what} must be a 3-tuple: {value!r}")
    return (
        _require_finite(value[0], f"{what}[0]"),
        _require_finite(value[1], f"{what}[1]"),
        _require_finite(value[2], f"{what}[2]"),
    )


def _sub(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


# ------------------------------------------------------------- 收线端速度源 ---


@runtime_checkable
class TakeupSource(Protocol):
    """收线端速度的来源。**它就是1.3那条扰动通道的类型**。

    实现者今天有三个：`ConstantTakeup`（不受扰的基准）、
    `AnalyticTakeup`（金标那一侧的解析速率）、
    `ArmLaydownTakeup`（真的从`laydown`的落位点几何取所需送带率）。
    """

    def takeup_speed_mm_s(self, t_s: float) -> float: ...


@dataclass(frozen=True)
class ConstantTakeup:
    """恒定收线端速度——**基准档**，用来判"没有扰动时张力一动不动"。"""

    channel_id: str
    speed_mm_s: float

    def __post_init__(self) -> None:
        _require_namespace(self.channel_id, "takeup", "channel_id")
        _require_finite(self.speed_mm_s, f"{self.channel_id}: speed_mm_s")

    def takeup_speed_mm_s(self, t_s: float) -> float:
        _require_finite(t_s, f"{self.channel_id}: t_s")
        return self.speed_mm_s


@dataclass(frozen=True)
class AnalyticTakeup:
    """任意可调用的收线端速率，**构造期做一次证伪**（纯不纯）。

    与`laydown.FeedAccount`同一条纪律：库证明不了一个函数是纯的，
    所以这里做的是**证伪**——在``probe_times_s``上各求值两次并比较。
    抓得住"读时钟""读随机数"，**抓不住**"只在探针之外某段变卦"，如实登记。
    """

    channel_id: str
    speed_fn: Callable[[float], float]
    probe_times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.channel_id, "takeup", "channel_id")
        if not callable(self.speed_fn):
            raise DisturbanceError(f"{self.channel_id}: speed_fn must be callable")
        if not isinstance(self.probe_times_s, tuple) or not self.probe_times_s:
            raise DisturbanceError(
                f"{self.channel_id}: 至少要给一个探针时刻——"
                "一条没有配证伪尝试的纯函数声明就是冒充（AGENTS.md诚实可信度）"
            )
        for probe in self.probe_times_s:
            time_s = _require_finite(probe, f"{self.channel_id}: probe time")
            first = self.takeup_speed_mm_s(time_s)
            second = self.takeup_speed_mm_s(time_s)
            if first != second:
                raise DisturbanceError(
                    f"{self.channel_id}: speed_fn在t_s={time_s!r}给了两个不同的值"
                    f"（{first!r}然后{second!r}）——它不是纯函数，这条扰动不可重放"
                )

    def takeup_speed_mm_s(self, t_s: float) -> float:
        value = self.speed_fn(_require_finite(t_s, f"{self.channel_id}: t_s"))
        return _require_finite(value, f"{self.channel_id}: speed at t_s={t_s!r}")


@dataclass(frozen=True)
class ArmLaydownTakeup:
    """**1.3的接线本体**：`laydown`的所需送带率 ⟹ `transport`的收线端速度。

    落位点必须待在那个固定的入带点上，所以"这一瞬要放出多少带材"
    就是落位点在槽上前进的速率``dσ_pose/dt``——`laydown`已经算得出它
    （`LaydownModel.required_feed_rate_mm_s`），本类做的只是把它接到收线端。

    ## 为什么接的是``required``（位姿定的）而不是``accounted``（送带账定的）

    0067裁过：闭合的两条来源一般不自洽，**本仓两条各算一次、把差额按方向拆开报，
    不挑哪一条当成对的**。而这里要的是"跨段里的材料被消耗得多快"这件**物理**，
    它由**位姿**决定：线圈转过去了，材料就被带走了，送带账同不同意都一样。
    送带账那一条是**上游的声明**，两者之差是闭合残差在速率上的那一面
    （`LaydownPoint.feed_rate_gap_mm_s`），**它是要报的量不是要用的量**。

    ## 构造期那道跨模块的长度门

    `transport.FreeSpan.geometric_length_mm`与`laydown.FreeSpanGeometry`
    两个端点之间的距离必须是同一个数（`SPAN_LENGTH_MATCH_ABS_TOL_MM`）。
    **这不是精度门是接错线的门**：两个模块各拿着一条不同的跨段照样都能跑完，
    而算出来的张力没有任何东西说得清是哪一条跨段的。
    """

    channel_id: str
    laydown: LaydownModel
    span: FreeSpan

    def __post_init__(self) -> None:
        _require_namespace(self.channel_id, "takeup", "channel_id")
        if not isinstance(self.laydown, LaydownModel):
            raise DisturbanceError(
                f"{self.channel_id}: laydown必须是LaydownModel，得到{self.laydown!r}"
            )
        if not isinstance(self.span, FreeSpan):
            raise DisturbanceError(
                f"{self.channel_id}: span必须是transport.FreeSpan，得到{self.span!r}"
            )
        geometric = self.laydown.span.length_mm()
        gap = abs(geometric - self.span.geometric_length_mm)
        if gap > SPAN_LENGTH_MATCH_ABS_TOL_MM:
            raise DisturbanceError(
                f"{self.channel_id}: `transport`那一侧的跨长"
                f"{self.span.geometric_length_mm!r} mm与`laydown`两个端点之间的距离"
                f"{geometric!r} mm差{gap!r} mm —— 两个模块拿着的不是同一条自由跨段。"
                "这道门不判精度判接线：接错了两边照样跑得完，"
                "而算出来的张力没有任何东西说得清是哪一条跨段的"
            )

    def takeup_speed_mm_s(self, t_s: float) -> float:
        """所需送带率``dσ_pose/dt``——**位姿要的**那一条。"""

        return self.laydown.required_feed_rate_mm_s(t_s)

    def feed_rate_gap_mm_s(self, t_s: float) -> float:
        """位姿要的速率 减 送带账给的速率。**报出来，不拿它去修任何东西。**"""

        return self.laydown.required_feed_rate_mm_s(t_s) - self.laydown.accounted_feed_rate_mm_s(
            t_s
        )

    def groove_tangent_turn_rate_rad_s(self, t_s: float) -> float:
        """**落位点处槽切向的转速**``|dt̂/dt| = κ(σ)·σ'``（工件系，材料意义上的）。

        这是1.3那条判据里的"槽切向转多快"。取工件系而不是世界系，
        理由是：世界系的切向在**理想绕线**下恒定（带材无折角地续上槽，
        正是`laydown`把入射角的理想值定在0的那件事），
        于是世界系转速在理想档上恒为零、当不了旋钮；
        而工件系的这一条是曲线自己的性质乘以送带率，**它才是σ'的那个共轭量**。

        用`LaydownModel`自己声明的那个速率探针差分，所以它的截断阶
        与``required_feed_rate_mm_s``**是同一档**——两个量的误差因此可比。
        """

        left_time, right_time, denominator = self.laydown.rate_probe.stencil(
            _require_finite(t_s, f"{self.channel_id}: t_s")
        )
        left = self.laydown.centerline.sample_at(
            self.laydown.pose_arc_length_mm(left_time)[0]
        ).tangent
        right = self.laydown.centerline.sample_at(
            self.laydown.pose_arc_length_mm(right_time)[0]
        ).tangent
        return _norm(_sub(right, left)) / denominator

    def world_tangent_turn_rate_rad_s(self, t_s: float) -> float:
        """世界系里落位点处槽切向的转速。**理想绕线下它恒为零**，见上一条。"""

        left_time, right_time, denominator = self.laydown.rate_probe.stencil(
            _require_finite(t_s, f"{self.channel_id}: t_s")
        )
        left = self.laydown.world_sample(
            left_time, self.laydown.pose_arc_length_mm(left_time)[0]
        ).tangent
        right = self.laydown.world_sample(
            right_time, self.laydown.pose_arc_length_mm(right_time)[0]
        ).tangent
        return _norm(_sub(right, left)) / denominator


# --------------------------------------------------------------- 人手触碰 ---


@dataclass(frozen=True)
class TransverseTouch:
    """**1.4的接口**：在跨段上某一点、某时刻起、持续一段，把带材横向按开。

    ## 每一条都是独立字段，一条都不从别的量的符号里推

    本仓吃过一次亏：`PenaltyAnnulusLimit`把法兰朝向编码在限位坐标的符号里，
    单元门永远抓不到，`winding_line_endtoend`端到端跑一次才炸。所以这里：

    | 字段 | 它单独承担什么 |
    |---|---|
    | ``push_direction`` | **方向**，显式单位向量，且必须与跨段正交 |
    | ``offset_mm`` | **量值**，必须为正——它不兼职表示方向 |
    | ``station_from_guide_mm`` | **作用点**，写明从导轮端量 |
    | ``start_time_s`` / ``end_time_s`` | **起止**，两端都显式，左闭右开 |

    ## 它怎么变成张力：折线比直线长

    两个端点仍然固定，被按开的是中间。跨段从一条长``L``的直线段变成两段折线：

        L_path(δ) = sqrt(a² + δ²) + sqrt(b² + δ²)        a + b = L
        path_excess = L_path(δ) − L

    这是**几何恒等式**，不是一个模型。小角度下``path_excess ≈ δ²·L/(2ab)``——
    **二次**，所以按开1 mm与按开2 mm差的是四倍不是两倍。

    横向力是**输出**：带材两侧张力的横向分量必须与手指的力平衡，

        F = T·(δ/sqrt(a²+δ²) + δ/sqrt(b²+δ²)) ≡ T·g(δ)

    ``g``只由几何定，所以触碰窗口内它是常数——第四节那条冲量恒等式就建在这上面。

    ## 位移控制不是力控制，这一条是裁出来的

    真的手指给的是介于两者之间的东西。取位移控制的理由：
    **它给出精确几何**（``δ``是输入，``path_excess``是恒等式），
    而力控要在每一步解一个关于``δ``的非线性方程（``T``自己随``δ``变），
    于是判据里会混进一个求解器容差，而那正好盖住这道门本该分辨的东西。
    力控档**没做**，登记在模块文档第五节。
    """

    touch_id: str
    #: 自由跨段的两个端点。**世界系常量**，见模块文档第一节。
    geometry: FreeSpanGeometry
    #: 时间剖面，取自`TOUCH_PROFILES`。**没有默认值。**
    profile: str
    #: 作用点到**导轮端出带点**的距离。写明从哪一端量，不靠约定记忆。
    station_from_guide_mm: float
    #: 手指推的方向。**显式单位向量，必须与跨段正交。**
    push_direction: Vector3
    start_time_s: float
    end_time_s: float
    #: 横向位移量值。**必须为正**——方向由``push_direction``携带。
    offset_mm: float

    def __post_init__(self) -> None:
        _require_namespace(self.touch_id, "touch", "touch_id")
        if not isinstance(self.geometry, FreeSpanGeometry):
            raise DisturbanceError(
                f"{self.touch_id}: geometry必须是laydown.FreeSpanGeometry，"
                f"得到{self.geometry!r}"
            )
        if self.profile not in TOUCH_PROFILES:
            raise DisturbanceError(
                f"{self.touch_id}: profile must be one of {sorted(TOUCH_PROFILES)}: "
                f"{self.profile!r} —— 时间剖面没有默认值，"
                "一条真实的人手触碰是什么形状本仓一次都没有量过"
            )
        length = self.geometry.length_mm()
        station = _require_finite(
            self.station_from_guide_mm, f"{self.touch_id}: station_from_guide_mm"
        )
        if not (0.0 < station < length):
            raise DisturbanceError(
                f"{self.touch_id}: 作用点{station!r} mm不在跨段(0, {length!r})之内——"
                "端点上的横向位移不改路径长度（两段之一退化成零长），"
                "那不是一次触碰，是把端点搬了"
            )
        direction = _require_vector(self.push_direction, f"{self.touch_id}: push_direction")
        norm = _norm(direction)
        if abs(norm - 1.0) > DIRECTION_ABS_TOL:
            raise DisturbanceError(
                f"{self.touch_id}: push_direction不是单位向量（模长{norm!r}）——"
                "本层不替调用方归一化：一个模长1.4的'方向'会让位移量值静默变成1.4倍"
            )
        along = self.geometry.material_increasing_direction()
        projection = _dot(direction, along)
        if abs(projection) > DIRECTION_ABS_TOL:
            raise DisturbanceError(
                f"{self.touch_id}: push_direction与跨段方向的内积是{projection!r}，"
                "不正交——沿跨段方向的那一份不是'把带材按开'，"
                "它等于把作用点沿着带材挪一挪，路径长度一分钱都不变"
            )
        start = _require_finite(self.start_time_s, f"{self.touch_id}: start_time_s")
        end = _require_finite(self.end_time_s, f"{self.touch_id}: end_time_s")
        if start < 0.0:
            raise DisturbanceError(
                f"{self.touch_id}: start_time_s = {start!r} < 0 —— "
                "时间轴与上游的``time_origin = run_start``同一个约定"
            )
        if not end > start:
            raise DisturbanceError(
                f"{self.touch_id}: 触碰区间[{start!r}, {end!r})不是正长度——"
                "**起止两端都是显式字段**，零长度触碰只会让下游以为它生效过"
            )
        offset = _require_finite(self.offset_mm, f"{self.touch_id}: offset_mm")
        if not offset > 0.0:
            raise DisturbanceError(
                f"{self.touch_id}: offset_mm = {offset!r} 必须为正 —— "
                "方向由push_direction携带，**量值不兼职表示方向**。"
                "这一条是`PenaltyAnnulusLimit`把法兰朝向编码在坐标符号里那次的直接产物："
                "符号兼职的字段，单元门永远抓不到"
            )

    # -- 几何 --------------------------------------------------------------

    def span_length_mm(self) -> float:
        return self.geometry.length_mm()

    def leg_lengths_mm(self) -> tuple[float, float]:
        """``(a, b)``——作用点到导轮端、到入带端的距离，``a + b = L``。"""

        length = self.geometry.length_mm()
        return (self.station_from_guide_mm, length - self.station_from_guide_mm)

    def is_active(self, t_s: float) -> bool:
        """区间取**左闭右开**``[start, end)``。

        显式写下来是因为它决定了触碰窗口里有几步：闭右端会让最后一步
        既在窗口内又在窗口外，而第四节那条冲量恒等式是按窗口逐步求和的。
        """

        time_s = _require_finite(t_s, f"{self.touch_id}: t_s")
        return self.start_time_s <= time_s < self.end_time_s

    def offset_at_mm(self, t_s: float) -> float:
        """``rectangular_hold``：窗口内恒为``offset_mm``，窗口外恰为0。"""

        return self.offset_mm if self.is_active(t_s) else 0.0

    def path_excess_for_offset_mm(self, offset_mm: float) -> float:
        """``sqrt(a²+δ²) + sqrt(b²+δ²) − L``。**几何恒等式，不是模型。**"""

        offset = _require_finite(offset_mm, f"{self.touch_id}: offset_mm")
        first, second = self.leg_lengths_mm()
        return (
            math.hypot(first, offset)
            + math.hypot(second, offset)
            - self.geometry.length_mm()
        )

    def path_excess_mm(self, t_s: float) -> float:
        """本时刻的路径增量。窗口外恰为0.0（``hypot(a,0) = a``逐位成立）。"""

        return self.path_excess_for_offset_mm(self.offset_at_mm(t_s))

    def force_geometry_factor(self, t_s: float) -> float:
        """``g(δ) = δ/sqrt(a²+δ²) + δ/sqrt(b²+δ²)``——两侧张力横向分量的方向余弦之和。

        窗口内是**常数**（``δ``恒定），第四节那条冲量恒等式就建在这一点上。
        """

        offset = self.offset_at_mm(t_s)
        if offset == 0.0:
            return 0.0
        first, second = self.leg_lengths_mm()
        return offset / math.hypot(first, offset) + offset / math.hypot(second, offset)

    def transverse_force_n(self, t_s: float, *, tension_n: float) -> float:
        """``F = T·g(δ)``——手指要顶住的横向力。**它是输出不是输入。**"""

        tension = _require_finite(tension_n, f"{self.touch_id}: tension_n")
        return tension * self.force_geometry_factor(t_s)

    def contact_point_mm(self, t_s: float) -> Vector3:
        """被按开之后那个接触点的世界系位置。判据用它看"方向真的是显式的"。"""

        guide = self.geometry.guide_exit_mm
        #: ``material_increasing_direction``是从入带点指向导轮的，
        #: 所以从导轮端往入带点走要取它的反向——**这一步不许靠符号猜**。
        along = self.geometry.material_increasing_direction()
        offset = self.offset_at_mm(t_s)
        station = self.station_from_guide_mm
        return (
            guide[0] - along[0] * station + self.push_direction[0] * offset,
            guide[1] - along[1] * station + self.push_direction[1] * offset,
            guide[2] - along[2] * station + self.push_direction[2] * offset,
        )


# ------------------------------------------------------------------- 账 ---


@dataclass(frozen=True)
class DisturbanceLedger:
    """一次受扰运行的账。**每一条都有一个独立算出来的对手，不是自报自销。**"""

    steps: int
    duration_s: float
    #: 材料长度账：状态之差 对 逐步流量之和。
    material_length_change_mm: float
    material_length_flow_mm: float
    material_length_residual_mm: float
    #: 角冲量账：``(J/1000)·Δω`` 对 ``Σdt·(T·R − M − c·ω)``。
    angular_momentum_change_nmm_s: float
    angular_impulse_nmm_s: float
    angular_impulse_residual_nmm_s: float
    #: 横向冲量：逐步求和 对 第四节那条恒等式重构出来的值。
    transverse_impulse_n_s: float
    transverse_impulse_from_ledger_n_s: float
    transverse_impulse_residual_n_s: float
    touched_steps: int
    #: 张力的极值与起点，判"尖峰"用。
    initial_tension_n: float
    peak_tension_n: float
    trough_tension_n: float

    def peak_excursion_n(self) -> float:
        """尖峰相对起点的高度。"""

        return self.peak_tension_n - self.initial_tension_n


def _touch_window_indices(
    samples: Sequence[SpanTransportSample], touch: TransverseTouch
) -> tuple[int, int]:
    """触碰窗口在样点序列里的``[首, 末+1)``。窗口必须是**连续**的一段。"""

    active = [index for index, sample in enumerate(samples) if touch.is_active(sample.time_s)]
    if not active:
        return (0, 0)
    first, last = active[0], active[-1]
    if last - first + 1 != len(active):
        raise DisturbanceError(
            f"{touch.touch_id}: 触碰窗口在样点上不连续（{len(active)}个活动步"
            f"却横跨{last - first + 1}步）——冲量恒等式是按窗口逐步求和的，"
            "不连续的窗口对不上账"
        )
    return (first, last + 1)


def run_disturbed_span(
    loop: SpanTransportLoop,
    *,
    steps: int,
    brake_torque_nmm: float,
    takeup: TakeupSource,
    touch: TransverseTouch | None = None,
) -> tuple[SpanTransportLoop, tuple[SpanTransportSample, ...], DisturbanceLedger]:
    """两条扰动通道一起推进，并把账算出来。

    ``takeup``给1.3那条（臂动经落位点几何变成收线端速度），
    ``touch``给1.4那条（横向侵入让路径变长）。**两条互不折算**，
    也都可以单独用：``touch=None``就是纯臂动扰动，
    ``takeup``取`ConstantTakeup`就是纯触碰扰动。

    制动力矩恒定是**冲量恒等式的前提**，所以它是一个标量而不是一条时间线；
    要让制动力矩随时间变，那是控制器的活，归轨道E。
    """

    if not isinstance(loop, SpanTransportLoop):
        raise DisturbanceError(f"loop必须是transport.SpanTransportLoop：{loop!r}")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise DisturbanceError(f"steps must be a positive int: {steps!r}")
    if not isinstance(takeup, TakeupSource):
        raise DisturbanceError(
            f"takeup必须实现takeup_speed_mm_s(t_s)：{takeup!r} —— "
            "收线端速度是1.3那条扰动通道的类型，不许随手传一个裸函数进来"
        )
    if touch is not None and not isinstance(touch, TransverseTouch):
        raise DisturbanceError(f"touch必须是TransverseTouch或None：{touch!r}")
    brake = _require_finite(brake_torque_nmm, "brake_torque_nmm")

    start_length = loop.material_length_mm
    start_omega = loop.angular_velocity_rad_s
    dt_s = loop.dt_s
    samples: list[SpanTransportSample] = []
    flow_mm = 0.0
    impulse_nmm_s = 0.0
    current = loop
    for _ in range(steps):
        time_s = current.step_index * dt_s
        excess = touch.path_excess_mm(time_s) if touch is not None else 0.0
        speed = takeup.takeup_speed_mm_s(time_s)
        nxt, sample = current.step(
            brake_torque_nmm=brake,
            takeup_speed_mm_s=speed,
            path_excess_mm=excess,
        )
        #: 长度账用**步末**转速（半隐式），力矩账用**步首**转速——
        #: 两条各自照着推进器里那一条写，**不许在这里"统一"成一条**。
        flow_mm += dt_s * (nxt.angular_velocity_rad_s * current.reel.radius_mm - speed)
        impulse_nmm_s += dt_s * (
            sample.tension_n * current.reel.radius_mm
            - brake
            - current.reel.bearing_damping_nmm_s * sample.angular_velocity_rad_s
        )
        samples.append(sample)
        current = nxt

    tensions = [sample.tension_n for sample in samples]
    momentum_change = (
        current.reel.inertia_kg_mm2
        / 1000.0
        * (current.angular_velocity_rad_s - start_omega)
    )
    length_change = current.material_length_mm - start_length

    transverse_impulse = 0.0
    reconstructed = 0.0
    touched = 0
    if touch is not None:
        first, stop = _touch_window_indices(samples, touch)
        touched = stop - first
        if touched:
            factor = touch.force_geometry_factor(samples[first].time_s)
            window = samples[first:stop]
            transverse_impulse = factor * dt_s * sum(sample.tension_n for sample in window)
            radius = current.reel.radius_mm
            damping = current.reel.bearing_damping_nmm_s
            inertia = current.reel.inertia_kg_mm2
            #: 窗口两端的状态：``ω``与``L_mat``都是**步首**量，
            #: 所以窗口末端要取"窗口最后一步之后"的状态，即``stop``号样点的步首值；
            #: ``stop == len(samples)``时那就是推进结束后的当前状态。
            if stop < len(samples):
                omega_end = samples[stop].angular_velocity_rad_s
                length_end = samples[stop].material_length_mm
            else:
                omega_end = current.angular_velocity_rad_s
                length_end = current.material_length_mm
            delta_omega = omega_end - samples[first].angular_velocity_rad_s
            delta_length = length_end - samples[first].material_length_mm
            takeup_sum = dt_s * sum(sample.takeup_speed_mm_s for sample in window)
            reconstructed = (factor / radius) * (
                (inertia / 1000.0 - damping * dt_s) * delta_omega
                + brake * touched * dt_s
                + (damping / radius) * (delta_length + takeup_sum)
            )

    ledger = DisturbanceLedger(
        steps=steps,
        duration_s=steps * dt_s,
        material_length_change_mm=length_change,
        material_length_flow_mm=flow_mm,
        material_length_residual_mm=length_change - flow_mm,
        angular_momentum_change_nmm_s=momentum_change,
        angular_impulse_nmm_s=impulse_nmm_s,
        angular_impulse_residual_nmm_s=momentum_change - impulse_nmm_s,
        transverse_impulse_n_s=transverse_impulse,
        transverse_impulse_from_ledger_n_s=reconstructed,
        transverse_impulse_residual_n_s=transverse_impulse - reconstructed,
        touched_steps=touched,
        initial_tension_n=tensions[0],
        peak_tension_n=max(tensions),
        trough_tension_n=min(tensions),
    )
    return current, tuple(samples), ledger


# --------------------------------------------------------- 闭式（判据用）---


def tangent_turn_feed_rate_mm_s(
    *, tangent_turn_rate_rad_s: float, curvature_per_mm: float
) -> float:
    """``σ' = Ω_tan / κ``——**1.3那条机制的全部内容**。

    它同时给出那个必须有的退化档：``κ``是常数时``σ'``跟着``Ω_tan``走，
    ``Ω_tan``匀速则``σ'``**恒定**、扰动**恒为零**。
    **扰动来自曲率在变，不是来自切向在转。**
    """

    rate = _require_finite(tangent_turn_rate_rad_s, "tangent_turn_rate_rad_s")
    curvature = _require_finite(curvature_per_mm, "curvature_per_mm")
    if not curvature > 0.0:
        raise DisturbanceError(
            f"curvature_per_mm = {curvature!r} 必须为正 —— "
            "曲率为零的一段是直线，直线上'切向转多快'定义不出送带率"
        )
    return rate / curvature


def harmonic_tension_amplitude_n(
    *,
    span_stiffness_n_per_mm: float,
    takeup_amplitude_mm_s: float,
    forcing_rad_s: float,
    natural_frequency_rad_s: float,
    damping_ratio: float,
) -> float:
    """收线端速度**正弦**扰动下张力的稳态幅值（线性化闭式）。

        δT'' + 2ζω_n·δT' + ω_n²·δT = K·a·[2ζω_n·cos ωt − ω·sin ωt]

        |δT| = K·a·sqrt(4ζ²ω_n² + ω²) / sqrt((ω_n² − ω²)² + 4ζ²ω_n²ω²)

    **右端那两项不是一项**：速度扰动同时经"稳态张力随线速度走"（``2ζω_n``那一项，
    即``c·v/R²``）与"长度账的导数"（``ω``那一项）进来。只留前者会在
    ``ω → ω_n``附近整个错掉，只留后者会在``ω → 0``给出零——
    而``ω → 0``的正确极限恰是稳态关系``dT/dv = c/R²``，
    本式在那里给``2ζK/ω_n``，两者**逐位相等**（有一条门判它）。
    """

    stiffness = _require_finite(span_stiffness_n_per_mm, "span_stiffness_n_per_mm")
    amplitude = _require_finite(takeup_amplitude_mm_s, "takeup_amplitude_mm_s")
    forcing = _require_finite(forcing_rad_s, "forcing_rad_s")
    natural = _require_finite(natural_frequency_rad_s, "natural_frequency_rad_s")
    ratio = _require_finite(damping_ratio, "damping_ratio")
    if not natural > 0.0:
        raise DisturbanceError(f"natural_frequency_rad_s必须为正：{natural!r}")
    if not 0.0 < ratio < 1.0:
        raise DisturbanceError(
            f"damping ratio must be in (0, 1): {ratio!r} —— "
            "ζ = 0时``ω = ω_n``处幅值无界，ζ ≥ 1时这条式子仍成立但本仓没有验过那一档"
        )
    numerator = math.sqrt(
        4.0 * ratio * ratio * natural * natural + forcing * forcing
    )
    gap = natural * natural - forcing * forcing
    denominator = math.sqrt(
        gap * gap + 4.0 * ratio * ratio * natural * natural * forcing * forcing
    )
    return stiffness * abs(amplitude) * numerator / denominator


def path_step_tension_ring_n(
    *,
    step_tension_n: float,
    natural_frequency_rad_s: float,
    damping_ratio: float,
    time_s: float,
) -> float:
    """路径长度**阶跃**之后张力的振铃（线性化闭式，相对稳态）。

        δT(t) = ΔT₀·e^{−ζω_n t}·[cos ω_d t + (ζ/sqrt(1−ζ²))·sin ω_d t]

    初值是``δT(0) = ΔT₀ = EA·path_excess/L_mat``（材料长度是状态、不会瞬变，
    所以路径一跳张力**当场跳**同样多）、``δT'(0) = 0``（``δT' = −K·δv_放线``
    而放线速度也是状态、同样不会瞬变）。

    **这两条初值与`transport.velocity_step_overshoot`那一条恰好相反**：
    那里是张力连续而速度差当场跳（于是初值带一个斜率冲击、传递函数多一个零点），
    这里是速度连续而张力当场跳。**同一个二阶系统，两条完全不同的响应**——
    把其中一条套到另一条上，峰值时刻会差``acos ζ``、超调会差一个``1/(2ζ)``。
    """

    step = _require_finite(step_tension_n, "step_tension_n")
    natural = _require_finite(natural_frequency_rad_s, "natural_frequency_rad_s")
    ratio = _require_finite(damping_ratio, "damping_ratio")
    time_s = _require_finite(time_s, "time_s")
    if not natural > 0.0:
        raise DisturbanceError(f"natural_frequency_rad_s必须为正：{natural!r}")
    if not 0.0 <= ratio < 1.0:
        raise DisturbanceError(f"这条振铃闭式要欠阻尼：{ratio!r}")
    root = math.sqrt(1.0 - ratio * ratio)
    damped = natural * root
    return step * math.exp(-ratio * natural * time_s) * (
        math.cos(damped * time_s) + (ratio / root) * math.sin(damped * time_s)
    )


def ring_envelope_ratio(
    *, natural_frequency_rad_s: float, damping_ratio: float, elapsed_s: float
) -> float:
    """振铃包络在``elapsed_s``之后剩下多少：``exp(−ζω_n·t)``。

    **这就是"撤掉控制器时不回落"那条判据的闭式对手**。
    开环下唯一的阻尼通道是轴承（`PayoutReel.bearing_damping_nmm_s`），
    真实量级给``ζ = 0.0132``、时间常数``1/(ζω_n) ≈ 0.2 s``——
    **比一次落位动作还长**，所以尖峰在观测窗口里根本不回落，
    只是慢慢地振。有控制器时它应该在一个周期之内被压掉，**那一半归轨道E**。
    """

    natural = _require_finite(natural_frequency_rad_s, "natural_frequency_rad_s")
    ratio = _require_finite(damping_ratio, "damping_ratio")
    elapsed = _require_finite(elapsed_s, "elapsed_s")
    if not natural > 0.0:
        raise DisturbanceError(f"natural_frequency_rad_s必须为正：{natural!r}")
    if ratio < 0.0:
        raise DisturbanceError(f"damping_ratio不能为负：{ratio!r}")
    if elapsed < 0.0:
        raise DisturbanceError(f"elapsed_s不能为负：{elapsed!r}")
    return math.exp(-ratio * natural * elapsed)


__all__ = [
    "DIRECTION_ABS_TOL",
    "SPAN_LENGTH_MATCH_ABS_TOL_MM",
    "TOUCH_PROFILES",
    "AnalyticTakeup",
    "ArmLaydownTakeup",
    "ConstantTakeup",
    "DisturbanceError",
    "DisturbanceLedger",
    "TakeupSource",
    "TransverseTouch",
    "harmonic_tension_amplitude_n",
    "path_step_tension_ring_n",
    "ring_envelope_ratio",
    "run_disturbed_span",
    "tangent_turn_feed_rate_mm_s",
]
