"""线速度与输运——张力由速度差与带材弹性生成（力学域，决策0066）。

守plans/14第3.2节登记的二号缺口（原甲1）。

## 它替掉的那句话：``T = M/R``**不是一条定律，是一个稳态**

`drives.SpoolTension.tension_n`把张力写成"扭矩除以卷径"。那条式子里
**没有时间、没有速度、没有带材**——于是**控制器没有任何真东西可控**：
一个必须抑制速度扰动的控制律，在那个模型里连扰动都不存在。

本模块把它拆成三件真的东西：

1. **输运账**：自由跨段里的材料长度是一个状态，
   ``dL_mat/dt = v_放线 − v_收线``；
2. **带材弹性**：几何长度与材料长度之差生成应变，``T = EA·ε``；
3. **放线端的力矩平衡**：磁粉离合器是**制动**不是驱动，
   放线盘的转速由带材拉着走，``J·dω/dt = T·R − M − c·ω``。

**稳态时第3条给出``T = M/R + c·v/R²``**——``c = 0``那一项没了就正好是``T = M/R``。
所以旧模型不是错的，它是**零轴承阻力矩下的零线速度稳态**。
这句话在`drives.SpoolTension`的docstring里同批写了一遍。

## 数据源在上游已经有了，缺的一直是下游

消费方WII发布的``wii_motion_timeline.v2``逐样点带``material_feed_length_mm``
（单调非减）、``time_s``（严格递增、``time_origin = "run_start"``）与
``pause_intervals``。`MaterialFeedTimeline`就是把那一列的**时间导数**变成线速度
的那一层；`motion.SampledPoseTimeline`此前吃的是位姿，**没有任何东西吃长度**。

## 本模块的四条边界（不写在这里就算没写）

1. **几何长度取常数**。真机上自由跨的两端一端固定在R1、一端随机械臂走
   （plans/14第一节），跨长每个样点都在变。那是plans/14第3.3节登记的三号缺口，
   由另一条轨道的落位点几何层供给；本模块把``geometric_length_mm``收成一个字段，
   **将来它变成时间函数时改的是这里的一个入参，不是这套物理**；
2. **输运账全部记在未拉伸（材料）长度上**。``ω·R``被当作放线端的**材料**吞吐率，
   即盘上的带材按未拉伸算。这是一条声明不是一个实现细节：真机上盘上的带材
   也带着张力，误差是``O(ε)``，本模块工况下``ε ≈ 3.3e-4``；
3. **只建"已经在放线"的工况**（``ω > 0``）。转速降到零及以下时磁粉制动器进入
   静摩擦保持段（``|T·R| ≤ M``时盘不转），那是一条**互补条件**、要一套return-map，
   本模块**不假装能算它**——``ω ≤ 0``当场失败关闭；
4. **不做材料注入**。`feed.py`的docstring自陈那是"带材从轮面流过、边界随材料移动"，
   与本模块的长度账不是一回事。本模块是**欧拉**的长度账（跨段是一个控制体），
   `feed.py`是**拉格朗日**的节点账，两者互不覆盖也互不冲突。

## 判据：速度阶跃下张力**精确**是一个带零点的二阶系统

在稳态附近线性化（``K ≡ −dT/dL_mat = EA·L_geo/L_mat²``）：

    dδT/dt = −K·δv_放线
    dδv_放线/dt = (1000·R²/J)·δT − (1000·c/J)·δv_放线

消去``δv``：

    δT'' + (1000c/J)·δT' + (1000·R²·K/J)·δT = 0

于是

    ω_n = sqrt(1000·R²·K/J)
    ζ   = (1000·c/J) / (2·ω_n)

**那个``1000``是``N = kg·m/s²``与本仓mm制之间的换算**，
掉了它``ω_n``会差``sqrt(1000) = 31.6``倍。有一条门专判它
（案例`free_span_tension_step`，与`two_body_spring`那条"1000倍单位bug的捕手"同源）。

收线端速度阶跃``Δv``时的响应**不是**教科书那条二阶阶跃：初值条件带一个斜率冲击
（张力连续、而速度差当场跳``−Δv``），等价于传递函数多一个零点。闭式仍是精确的：

    y(t) = 1 − e^{−ζω_n t}·[cos(ω_d t) − ((1−2ζ²)/(2ζ√(1−ζ²)))·sin(ω_d t)]
    峰值时刻   t_p = (π − acos ζ) / ω_d
    相对超调   exp(−ζ(π − acos ζ)/√(1−ζ²)) / (2ζ)

## ``exp(−ζΦ/√(1−ζ²))``这个形状在本仓出现了**第三次**，而``Φ``第三次不同

| 出处 | ``Φ`` | 还有没有前因子 |
|---|---|---|
| `contact.restitution_from_damping_ratio` | ``2·acos ζ`` | 无（0052第一节的截断约定） |
| `drives.step_response_overshoot` | ``π`` | 无（半个阻尼周期） |
| 本模块`velocity_step_overshoot` | ``π − acos ζ`` | **有**，``1/(2ζ)`` |

`drives.py`已经记着前两条"只在``ζ = 0``处相等"。**本模块这一条更难看**：
``π − acos ζ = 2·acos ζ`` ⟺ ``acos ζ = π/3`` ⟺ **``ζ = 0.5``**，
而``1/(2ζ)``在``ζ = 0.5``处**恰好是1**。于是本式与恢复系数式在``ζ = 0.5``处
**逐位相同**（2026-08-17实测两边都是``0.29841584422129015``）。

``ζ = 0.5``是写测试时最顺手的那个值。**只在``ζ = 0.5``上判，判不出这两件事是两件事**——
有一条门专记这个盲区，并强制在第二个``ζ``上再判一次。

## 离散化：半隐式Euler，实测**一阶**，且离散不动点与连续不动点**恰好相同**

    ω_{n+1} = ω_n + dt·α(T_n, ω_n)
    L_{n+1} = L_n + dt·(ω_{n+1}·R − v_收线)      ← 用**新**转速

不动点要求``α = 0``且``ω·R = v_收线``，与连续方程的稳态**逐字相同**——
**离散化没有给稳态引入偏差**，所以稳态那条判据判的是模型不是步长。

固定时刻上的误差实测一阶（2026-08-17，``dt``逐次减半：
``3.605e-5 / 1.804e-5 / 9.021e-6 / 4.513e-6``，比值``1.9991 / 1.9991 / 1.9987``）。
**阶是量出来的**：半隐式Euler在`integrate.py`里是一阶（`ballistic_free_flight`
量到误差恰为``+a·T·h/2``），这里量出来也是一阶，但那不是从那里推出来的。

## 全部输入是假设，产物永久``hypothesis_only``

带材``EA``、跨段长度、放线盘转动惯量、轴承阻力矩、真实线速度——
**一条实测都没有**，出处逐条见0062第二节裁决2那张"只有现场实测能补"的清单。
本模块证明的是**机制**：速度差怎么变成张力、张力怎么反过来改速度。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, replace

from physics_engine.motion import PauseInterval

#: ``N = kg·m/s²``与本仓mm制之间的换算。``M[N·mm]``作用在``J[kg·mm²]``上时
#: ``α[rad/s²] = 1000·M/J``——**掉了这个1000，``ω_n``差31.6倍**。
#: 与`energies.MM_PER_M`同源同值，**没有import它**：那是能量项的单位常数，
#: 这里是力矩-惯量的单位常数，两处各自要能被独立读懂。
MM_PER_M = 1000.0

#: 松弛判据。``ε ≤ 0``时带材不受压，张力取零而不是负值——理由见`FreeSpan`。
MIN_STRAIN = 0.0


class TransportError(ValueError):
    """线速度与输运的一切失败关闭。"""


def _require_positive(value: float, name: str) -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise TransportError(f"{name} must be a real number: {value!r}")
    if not (value > 0.0 and math.isfinite(value)):
        raise TransportError(f"{name} must be positive and finite: {value!r}")
    return float(value)


def _require_finite(value: float, name: str) -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise TransportError(f"{name} must be a real number: {value!r}")
    if not math.isfinite(value):
        raise TransportError(f"{name} must be finite: {value!r}")
    return float(value)


def _require_namespace(value: object, prefix: str, name: str) -> None:
    if not isinstance(value, str) or not value.startswith(f"{prefix}/") or value == f"{prefix}/":
        raise TransportError(
            f"{name} must look like {prefix}/<name>: {value!r} —— "
            "命名空间前缀是轴2规则1，不是装饰"
        )


# --------------------------------------------------------------- 上游时间线 ---

#: 段内速度语义。今天**只有一条**被声明：段内恒速、样点处右连续。
#:
#: 不给"线性插值速度"或"样条"，理由与`motion.InterpolationSemantics`同源：
#: 上游给的是**累计长度**，累计长度的线性插值就等于段内恒速。
#: 要更高阶就要声明"累计长度是几阶的"，而WII没有声明过那件事——
#: **替上游声明它，等于替上游拍板**。
RATE_SEMANTICS: frozenset[str] = frozenset({"piecewise_constant"})

#: 样点之外怎么办。与`motion.EXTRAPOLATIONS`同名同义，**故意不import**：
#: 那是位姿时间线的选择集，这是长度时间线的选择集，
#: 两者今天恰好一样不代表将来一样（长度可以有"按最后速率外推"，位姿没有）。
EXTRAPOLATIONS: frozenset[str] = frozenset({"reject", "clamp"})


@dataclass(frozen=True)
class MaterialFeedTimeline:
    """``material_feed_length_mm``那一列，加**显式声明的**速率语义与暂停区间。

    形制取自WII的``wii_motion_timeline.v2``（`src/timeline/wii_motion_timeline.py`
    第1236—1265行）：``time_s``严格递增且``time_origin = "run_start"``、
    ``material_feed_length_mm``单调非减、``pause_intervals``显式给出。

    **没有一个字段带默认值**，理由与`motion.SampledPoseTimeline`逐字相同：
    本层每一条都是声明者要拿主意的东西，而默认值正是替他拿了主意的那种写法。

    ## 三条门，各自挡的东西不一样

    1. **单调非减**：喂料长度回退意味着带材被吸回去，那不是这台机器能做的事。
       上游已经保证它；本仓**照样自己判**，因为跨仓契约里"上游保证过"
       与"本仓验过"不是同一件事；
    2. **零增量段必须被声明的暂停覆盖**。plans/14与WII都点过这一条：
       零运动段**没有可推断的时长**。一段长度不变的采样，
       与一段丢失的数据**在字节上没有区别**——要区分只能靠显式时间戳。
       所以本仓的判法是反过来的：**你要么声明它是暂停，要么它就是丢数据**；
    3. **暂停区间的两端必须落在样点时刻上**。段内恒速语义下，
       一个从段中间开始的暂停会让这一段的速率无从定义
       （"这一段平均20 mm/s，其中后半段是停的"——那前半段是多少？没人知道）。
    """

    source_id: str
    times_s: tuple[float, ...]
    feed_length_mm: tuple[float, ...]
    #: 段内速率语义，取自`RATE_SEMANTICS`。
    rate_semantics: str
    #: 样点之外怎么办，取自`EXTRAPOLATIONS`。
    extrapolation: str
    pauses: tuple[PauseInterval, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.source_id, "transport", "source_id")
        if self.rate_semantics not in RATE_SEMANTICS:
            raise TransportError(
                f"{self.source_id}: rate_semantics must be one of "
                f"{sorted(RATE_SEMANTICS)}: {self.rate_semantics!r}"
            )
        if self.extrapolation not in EXTRAPOLATIONS:
            raise TransportError(
                f"{self.source_id}: extrapolation must be one of "
                f"{sorted(EXTRAPOLATIONS)}: {self.extrapolation!r}"
            )
        if len(self.times_s) != len(self.feed_length_mm):
            raise TransportError(
                f"{self.source_id}: times_s有{len(self.times_s)}个而"
                f"feed_length_mm有{len(self.feed_length_mm)}个——"
                "两列各说各的样点数，是plans/09教训一记的那种洞"
            )
        if len(self.times_s) < 2:
            raise TransportError(
                f"{self.source_id}: 一条时间线至少要两个样点，"
                f"只有{len(self.times_s)}个时连一段速率都定义不出来"
            )
        for index, value in enumerate(self.times_s):
            _require_finite(value, f"{self.source_id}: times_s[{index}]")
        for index, value in enumerate(self.feed_length_mm):
            _require_finite(value, f"{self.source_id}: feed_length_mm[{index}]")
        if self.times_s[0] != 0.0:
            raise TransportError(
                f"{self.source_id}: times_s must start at run_start=0, "
                f"got {self.times_s[0]!r} —— 与WII的``time_origin``同一个约定，"
                "两仓对同一条时间线的0起算不许是口头协议"
            )
        for left, right in zip(self.times_s, self.times_s[1:], strict=False):
            if right <= left:
                raise TransportError(
                    f"{self.source_id}: times_s must be strictly increasing, "
                    f"got {left!r} then {right!r}"
                )
        for left, right in zip(
            self.feed_length_mm, self.feed_length_mm[1:], strict=False
        ):
            if right < left:
                raise TransportError(
                    f"{self.source_id}: feed_length_mm must be monotone "
                    f"non-decreasing, got {left!r} then {right!r} —— "
                    "喂料长度回退意味着带材被吸回去，那不是这台机器能做的事"
                )
        horizon = self.times_s[-1]
        sample_times = set(self.times_s)
        for pause in self.pauses:
            if not isinstance(pause, PauseInterval):
                raise TransportError(f"{self.source_id}: not a PauseInterval: {pause!r}")
            if pause.end_time_s > horizon:
                raise TransportError(
                    f"{self.source_id}: {pause.pause_id}结束于{pause.end_time_s!r}，"
                    f"超出时间线终点{horizon!r}"
                )
            if pause.start_time_s not in sample_times or pause.end_time_s not in sample_times:
                raise TransportError(
                    f"{self.source_id}: {pause.pause_id}的端点"
                    f"({pause.start_time_s!r}, {pause.end_time_s!r})没有落在样点时刻上——"
                    "段内恒速语义下，从段中间开始的暂停会让这一段的速率无从定义"
                )
        self._assert_zero_segments_are_declared_pauses()

    def _assert_zero_segments_are_declared_pauses(self) -> None:
        """零增量段必须被声明的暂停覆盖，**双向都判**。

        正向：一段长度不变的采样与一段丢失的数据在字节上没有区别，
        要区分只能靠显式时间戳。反向：一段被声明为暂停、期间材料却还在走，
        是声明与数据自相矛盾——**留着它比没有声明更坏**，
        因为下游会照着声明去跳过那一段。
        """

        for index in range(len(self.times_s) - 1):
            start = self.times_s[index]
            end = self.times_s[index + 1]
            moved = self.feed_length_mm[index + 1] != self.feed_length_mm[index]
            paused = any(
                pause.start_time_s <= start and end <= pause.end_time_s
                for pause in self.pauses
            )
            if not moved and not paused:
                raise TransportError(
                    f"{self.source_id}: 段[{start!r}, {end!r}]喂料长度零增量，"
                    "却没有任何声明的暂停覆盖它——**零运动段没有可推断的时长**，"
                    "它与一段丢失的数据在字节上没有区别，要么声明成暂停，"
                    "要么它就是丢数据"
                )
            if moved and paused:
                raise TransportError(
                    f"{self.source_id}: 段[{start!r}, {end!r}]被声明为暂停，"
                    f"期间喂料长度却从{self.feed_length_mm[index]!r}走到"
                    f"{self.feed_length_mm[index + 1]!r}——声明与数据自相矛盾"
                )

    def horizon_s(self) -> float:
        """最后一个样点的时刻。时间轴从``run_start = 0``起，它同时是时长。"""

        return self.times_s[-1]

    def _clamp(self, t_s: float) -> float:
        t_s = _require_finite(t_s, f"{self.source_id}: t_s")
        horizon = self.horizon_s()
        if t_s < 0.0 or t_s > horizon:
            if self.extrapolation == "reject":
                raise TransportError(
                    f"{self.source_id}: t_s={t_s!r} 在[0, {horizon!r}]之外而"
                    "extrapolation='reject' —— 样点之外没有任何样点支持的喂料，"
                    "要外推就去多给样点"
                )
            t_s = 0.0 if t_s < 0.0 else horizon
        return t_s

    def _segment(self, t_s: float) -> int:
        """含``t_s``的段号。终点归最后一段（**右端点左连续**）。"""

        if t_s >= self.horizon_s():
            return len(self.times_s) - 2
        return max(0, bisect_right(self.times_s, t_s) - 1)

    def length_mm(self, t_s: float) -> float:
        """累计喂料长度，段内线性（＝段内恒速的积分）。"""

        t_s = self._clamp(t_s)
        index = self._segment(t_s)
        left, right = self.times_s[index], self.times_s[index + 1]
        if t_s == left:
            return self.feed_length_mm[index]
        if t_s == right:
            return self.feed_length_mm[index + 1]
        u = (t_s - left) / (right - left)
        low, high = self.feed_length_mm[index], self.feed_length_mm[index + 1]
        return low + (high - low) * u

    def speed_mm_s(self, t_s: float) -> float:
        """线速度＝累计长度的时间导数。段内恒定，样点处**右连续**。

        这就是plans/14第3.2节说的"把``material_feed_length_mm``的时间导数变成
        放线速度"那一步。**上游一直有，下游一直没接**——本方法是那条接缝。
        """

        t_s = self._clamp(t_s)
        index = self._segment(t_s)
        span = self.times_s[index + 1] - self.times_s[index]
        return (self.feed_length_mm[index + 1] - self.feed_length_mm[index]) / span

    def segment_speeds_mm_s(self) -> tuple[float, ...]:
        """逐段速率。判据要对的就是这一串，不必逐点采样。"""

        return tuple(
            (self.feed_length_mm[index + 1] - self.feed_length_mm[index])
            / (self.times_s[index + 1] - self.times_s[index])
            for index in range(len(self.times_s) - 1)
        )


# ------------------------------------------------------------------- 自由跨 ---


@dataclass(frozen=True)
class FreeSpan:
    """自由跨段：几何长度固定，**材料长度是状态**，张力由二者之差与带材弹性生成。

        ε = (L_geo − L_mat) / L_mat
        T = EA·ε            （ε ≤ 0 时 T = 0）

    ## 应变的分母是``L_mat``不是``L_geo``

    工程应变的定义就是"伸长量除以**未拉伸**长度"。写成除以``L_geo``在
    ``ε ≈ 3e-4``这一档只差``3e-4``相对，但它会让`material_length_for_tension_mm`
    的反解不再是同一条式子的逆——**一个模型里两条互逆的式子不互逆，
    是本仓已经吃过亏的那种洞**。

    ## ``ε ≤ 0``取零而不是失败关闭——这一条是裁出来的，理由写在这里

    带材不受压，所以``T``不能为负。剩下的两条路：

    * **失败关闭**：松弛在这个模型里是**可达状态**（收线端慢于放线端时必然到达），
      对一个可达状态失败关闭，等于让模型在它本该描述的那一刻炸掉；
    * **静默取零**：松弛时``L_geo``**不再是跨段的长度**——带材垂下来了，
      跨段是一条悬链线而不是直线。取零之后这个模型给出的一切都不再有意义，
      而调用方看不见这件事。

    **裁决：原语取零并同时给出``is_slack``**，把"张力是零"与"模型已经出界"
    分成两个可读的事实；**是否容忍出界由`SpanTransportLoop.forbid_slack`显式声明**，
    没有默认值。这条与`TensionLoop.measurement_transfer`"没有默认值是有意的"同源：
    **把它做成默认值，等于让这条边界默默消失。**
    """

    span_id: str
    #: 跨段**不受扰时**的几何长度。**它是常数，而且这一条是场景给的不是省事**：
    #: plans/14第3.3节订正后的场景里，自由跨的两个端点——最后那只导向轮的出带点、
    #: 入带点——**都是世界系固定的**，所以在途自由段是空间里一条**不动的线段**
    #: （`laydown.FreeSpanGeometry`把这一条冻在了类型上：``length_mm()``连``t``都收不到）。
    #: **臂动不改这个数**；臂动改的是落位点的几何，那条通道走`takeup_speed_mm_s`。
    #: 决策0071第二节把0066第九、十节"真机上跨长逐样点变"那句措辞判为**错的**。
    geometric_length_mm: float
    #: ``EA``，单位N（与`energies.AxialStretch`的``axial_stiffness_n``同一个量）。
    axial_stiffness_n: float

    def __post_init__(self) -> None:
        _require_namespace(self.span_id, "span", "span_id")
        _require_positive(self.geometric_length_mm, f"{self.span_id}: geometric_length_mm")
        _require_positive(self.axial_stiffness_n, f"{self.span_id}: axial_stiffness_n")

    def path_length_mm(self, path_excess_mm: float = 0.0) -> float:
        """带材**实际走过的**路径长度＝不受扰跨长 ＋ 显式的路径增量。

        ``path_excess_mm``是**唯一**能让路径长度不等于`geometric_length_mm`的入口，
        而且它只有一个来源：**有东西横向顶在跨段上**（`disturbance.TransverseTouch`）。
        两个端点固定 ⟹ 直线段长度固定；顶一下之后带材走的是**折线**，
        折线比直线长——这条增量是**几何恒等式**不是模型参数。

        **默认值0.0在这里是一个物理事实不是一个替调用方拿的主意**：
        没有东西碰它时增量恰为零。这与`SpanTransportLoop.forbid_slack`
        "没有默认值是有意的"不矛盾——那一条是**声明**（容忍不容忍出界），
        这一条是**状态**（这一瞬有没有人碰）。
        """

        return self.geometric_length_mm + self.require_path_excess_mm(path_excess_mm)

    def require_path_excess_mm(self, path_excess_mm: float) -> float:
        """把路径增量验掉并原样返回。**非负是一条几何事实，不是一个约定。**"""

        excess = _require_finite(path_excess_mm, f"{self.span_id}: path_excess_mm")
        if excess < 0.0:
            raise TransportError(
                f"{self.span_id}: path_excess_mm = {excess!r} < 0 —— "
                "两个端点之间的最短路径就是直线段，任何横向侵入只会让路径**变长**。"
                "负增量意味着有人拿它当'跨长可以缩'的旋钮，"
                "而那正是plans/14第3.3节订正掉的那条错误因果"
            )
        return excess

    def strain(self, material_length_mm: float, *, path_excess_mm: float = 0.0) -> float:
        """``ε = (L_path − L_mat)/L_mat``。可以为负——负的意思是**松弛**。"""

        length = _require_positive(material_length_mm, f"{self.span_id}: material_length_mm")
        return (self.path_length_mm(path_excess_mm) - length) / length

    def is_slack(self, material_length_mm: float, *, path_excess_mm: float = 0.0) -> bool:
        """跨段里的材料比路径长度还长⟹带材垂下来了。"""

        return self.strain(material_length_mm, path_excess_mm=path_excess_mm) <= MIN_STRAIN

    def tension_n(self, material_length_mm: float, *, path_excess_mm: float = 0.0) -> float:
        """``T = EA·ε``，松弛时恰为0。"""

        strain = self.strain(material_length_mm, path_excess_mm=path_excess_mm)
        return self.axial_stiffness_n * strain if strain > MIN_STRAIN else 0.0

    def material_length_for_tension_mm(self, tension_n: float) -> float:
        """反解：``L_mat = EA·L_geo/(EA + T)``。标定与稳态起点要用。

        它与`tension_n`是**同一条式子的逆**，两边可以逐位对拍——
        本模块有一条门就是这么判的。
        """

        tension = _require_finite(tension_n, f"{self.span_id}: tension_n")
        if tension < 0.0:
            raise TransportError(
                f"{self.span_id}: 反解不接受负张力{tension!r} —— 带材不受压"
            )
        return self.axial_stiffness_n * self.geometric_length_mm / (
            self.axial_stiffness_n + tension
        )

    def stiffness_n_per_mm(
        self, material_length_mm: float, *, path_excess_mm: float = 0.0
    ) -> float:
        """``K = −dT/dL_mat = EA·L_path/L_mat²``——**跨段作为一根弹簧的刚度**。

        它不是``EA/L``：那是把应变的分母写成``L_geo``时的近似值，
        两者差``(1 + ε)²``。这一档``ε ≈ 3.3e-4``，差``6.7e-4``相对——
        **比本案例阶跃判据要分辨的量大**，所以判据必须用这一条而不是``EA/L``。
        """

        length = _require_positive(material_length_mm, f"{self.span_id}: material_length_mm")
        return self.axial_stiffness_n * self.path_length_mm(path_excess_mm) / (length * length)

    def tension_rate_n_per_s(
        self, *, material_length_mm: float, payout_speed_mm_s: float, takeup_speed_mm_s: float
    ) -> float:
        """``dT/dt = −K·(v_放线 − v_收线)``——**方向门判的就是这条的两个符号**。

        放线端多送 ⟹ 跨段里材料变多 ⟹ 应变变小 ⟹ **张力下降**；
        收线端多收 ⟹ 材料被抽走 ⟹ **张力上升**。
        **这两个符号搞反，闭环立刻变成正反馈**，而症状是指数发散而不是一个错的数——
        本仓在`winding_line_endtoend`上已经把一个方向搞反过一次（比值0.5587）。
        """

        payout = _require_finite(payout_speed_mm_s, f"{self.span_id}: payout_speed_mm_s")
        takeup = _require_finite(takeup_speed_mm_s, f"{self.span_id}: takeup_speed_mm_s")
        return -self.stiffness_n_per_mm(material_length_mm) * (payout - takeup)


# ------------------------------------------------------------------- 放线端 ---


@dataclass(frozen=True)
class PayoutReel:
    """放线盘：磁粉离合器**制动**，转速由带材拉着走。

        J·dω/dt = T·R − M_制动 − c·ω          （ω > 0）

    ## 制动不是驱动——这条决定了整个符号约定

    磁粉离合器给的是**滑差扭矩**，它反抗相对转动。放线盘上没有电机，
    带材是唯一的驱动源：**张力升高 ⟹ 放线盘加速**（不是减速）。
    加速之后跨段里材料变多、应变变小、张力回落——负反馈就是这么闭合的。

    "张力升高让哪一端慢下来"这个问法在本模型里**没有答案**：
    收线端是伺服给定的（`ED3L-08AEA`在CSP位置模式），张力改不动它；
    放线端**加速**而不是减速。真正慢下来的是**跨段材料的净流失速率**，
    而那不是任何一端的速度。有一条门专判这三个符号。

    ## ``ω ≤ 0``失败关闭

    转速降到零时磁粉制动器进入静摩擦保持段（``|T·R| ≤ M``时盘不转），
    那是一条互补条件，要一套return-map。本模块**不假装能算它**。

    ## 单位

    ``inertia_kg_mm2``是``kg·mm²``、``bearing_damping_nmm_s``是``N·mm``每``rad/s``。
    ``α = 1000·(T·R − M − c·ω)/J``里的1000是`MM_PER_M`，**不是手滑**。
    """

    reel_id: str
    radius_mm: float
    inertia_kg_mm2: float
    #: 轴承粘性阻力矩系数。**假设输入**，一条实测都没有。
    #: 它是本模型里**唯一的阻尼通道**：置零则跨段是一个无阻尼振子，
    #: 速度扰动引起的张力振荡**永远不衰减**。那不是模型的毛病，
    #: 是这条链路开环的真实性质——也正是控制器存在的理由。
    bearing_damping_nmm_s: float

    def __post_init__(self) -> None:
        _require_namespace(self.reel_id, "reel", "reel_id")
        _require_positive(self.radius_mm, f"{self.reel_id}: radius_mm")
        _require_positive(self.inertia_kg_mm2, f"{self.reel_id}: inertia_kg_mm2")
        damping = _require_finite(
            self.bearing_damping_nmm_s, f"{self.reel_id}: bearing_damping_nmm_s"
        )
        if damping < 0.0:
            raise TransportError(
                f"{self.reel_id}: bearing_damping_nmm_s不能为负{damping!r} —— "
                "负的粘性阻力矩是一个往系统里灌能量的轴承"
            )

    def angular_acceleration_rad_s2(
        self, *, tension_n: float, brake_torque_nmm: float, angular_velocity_rad_s: float
    ) -> float:
        """``α = 1000·(T·R − M − c·ω)/J``。``ω ≤ 0``失败关闭。"""

        tension = _require_finite(tension_n, f"{self.reel_id}: tension_n")
        brake = _require_finite(brake_torque_nmm, f"{self.reel_id}: brake_torque_nmm")
        omega = _require_finite(
            angular_velocity_rad_s, f"{self.reel_id}: angular_velocity_rad_s"
        )
        if brake < 0.0:
            raise TransportError(
                f"{self.reel_id}: brake_torque_nmm是**制动**力矩的量值{brake!r}，"
                "不能为负——负的制动力矩是一台电机，那不是磁粉离合器"
            )
        if omega <= 0.0:
            raise TransportError(
                f"{self.reel_id}: angular_velocity_rad_s = {omega!r} —— "
                "放线盘停转或反转时磁粉制动器进入静摩擦保持段（|T·R| ≤ M 时盘不转），"
                "那是一条互补条件、要一套return-map，本模块不假装能算它"
            )
        return MM_PER_M * (tension * self.radius_mm - brake - self.bearing_damping_nmm_s * omega) / self.inertia_kg_mm2

    def payout_speed_mm_s(self, angular_velocity_rad_s: float) -> float:
        """``v = ω·R``。按第二条边界，它被当作**材料**（未拉伸）吞吐率。"""

        omega = _require_finite(
            angular_velocity_rad_s, f"{self.reel_id}: angular_velocity_rad_s"
        )
        return omega * self.radius_mm

    def payout_acceleration_per_tension_mm_s2_per_n(self) -> float:
        """``∂(dv_放线/dt)/∂T = 1000·R²/J``——**恒正**，因为离合器是制动不是驱动。

        方向门用的两个偏导之一。
        """

        return MM_PER_M * self.radius_mm * self.radius_mm / self.inertia_kg_mm2


# ------------------------------------------------------------------- 方向门 ---


def tension_feedback_gain_per_s2(
    *,
    tension_rate_per_payout_speed_n_per_mm: float,
    payout_acceleration_per_tension_mm_s2_per_n: float,
) -> float:
    """两个偏导的乘积——**回路增益**，负反馈时为负。

    判据本体抽成**纯函数**（0064第八节那条形制），理由是它必须能被注错验红：
    两个偏导都是结构性正/负的，在真实对象上永远造不出一次红。
    **一条从没被注错验过的门，你不知道它在验什么。**
    """

    first = _require_finite(
        tension_rate_per_payout_speed_n_per_mm, "tension_rate_per_payout_speed_n_per_mm"
    )
    second = _require_finite(
        payout_acceleration_per_tension_mm_s2_per_n,
        "payout_acceleration_per_tension_mm_s2_per_n",
    )
    return first * second


def assert_tension_feedback_is_negative(gain_per_s2: float) -> None:
    """回路增益必须为负。为正＝正反馈，闭环指数发散而不是给一个错的数。"""

    gain = _require_finite(gain_per_s2, "gain_per_s2")
    if gain >= 0.0:
        raise TransportError(
            f"回路增益{gain!r} ≥ 0 —— 张力→放线速度→张力这条环是**正反馈**。"
            "两个偏导里必有一个符号反了：离合器被写成了驱动，"
            "或者放线端与收线端在长度账里被对调了"
        )


def assert_span_transport_directions(
    tension_rate: Callable[..., float],
    *,
    material_length_mm: float,
    line_speed_mm_s: float,
    probe_mm_s: float,
) -> None:
    """三个符号，逐条判——**这是本轨道那道方向门的本体**。

    1. 两端等速 ⟹ ``dT/dt`` 恰为 **0**（零容差：跨段里的材料一进一出正好抵消，
       这是一条恒等式不是一个收敛结果）；
    2. 放线端多送 ⟹ ``dT/dt < 0``（张力**下降**）；
    3. 收线端多收 ⟹ ``dT/dt > 0``（张力**上升**）。

    ``tension_rate``是一个可调用对象而不是一个`FreeSpan`，正是为了让注错用例
    能把放线/收线对调着传进来——**本仓在`winding_line_endtoend`上把一个方向
    搞反过一次（比值0.5587）**，这道门守的就是那一类。
    """

    probe = _require_positive(probe_mm_s, "probe_mm_s")
    balanced = tension_rate(
        material_length_mm=material_length_mm,
        payout_speed_mm_s=line_speed_mm_s,
        takeup_speed_mm_s=line_speed_mm_s,
    )
    if balanced != 0.0:
        raise TransportError(
            f"两端等速时dT/dt = {balanced!r}，不是恒等的0 —— "
            "一进一出没有正好抵消，长度账里有一项算重了或算漏了"
        )
    faster_payout = tension_rate(
        material_length_mm=material_length_mm,
        payout_speed_mm_s=line_speed_mm_s + probe,
        takeup_speed_mm_s=line_speed_mm_s,
    )
    if not faster_payout < 0.0:
        raise TransportError(
            f"放线端多送时dT/dt = {faster_payout!r}，没有下降 —— "
            "多送的材料应该让跨段松下来，符号反了"
        )
    faster_takeup = tension_rate(
        material_length_mm=material_length_mm,
        payout_speed_mm_s=line_speed_mm_s,
        takeup_speed_mm_s=line_speed_mm_s + probe,
    )
    if not faster_takeup > 0.0:
        raise TransportError(
            f"收线端多收时dT/dt = {faster_takeup!r}，没有上升 —— "
            "被抽走的材料应该让跨段绷紧，符号反了"
        )


# --------------------------------------------------------------- 闭式（判据）---


def steady_state_tension_n(
    *,
    brake_torque_nmm: float,
    radius_mm: float,
    bearing_damping_nmm_s: float,
    line_speed_mm_s: float,
) -> float:
    """稳态张力``T = M/R + c·v/R²``。

    ``c = 0``时它退化成`drives.SpoolTension.tension_n`那条``T = M/R``——
    **那条换算不是一条独立的定律，是本式在零轴承阻力矩下的特例**，
    而且它同时是"零线速度"下的值。有一条门判这个退化逐位成立。
    """

    radius = _require_positive(radius_mm, "radius_mm")
    brake = _require_finite(brake_torque_nmm, "brake_torque_nmm")
    damping = _require_finite(bearing_damping_nmm_s, "bearing_damping_nmm_s")
    speed = _require_finite(line_speed_mm_s, "line_speed_mm_s")
    return brake / radius + damping * speed / (radius * radius)


def span_natural_frequency_rad_s(
    *, span_stiffness_n_per_mm: float, radius_mm: float, inertia_kg_mm2: float
) -> float:
    """``ω_n = sqrt(1000·R²·K/J)``。那个1000是`MM_PER_M`，掉了差31.6倍。"""

    stiffness = _require_positive(span_stiffness_n_per_mm, "span_stiffness_n_per_mm")
    radius = _require_positive(radius_mm, "radius_mm")
    inertia = _require_positive(inertia_kg_mm2, "inertia_kg_mm2")
    return math.sqrt(MM_PER_M * radius * radius * stiffness / inertia)


def span_damping_ratio(
    *,
    span_stiffness_n_per_mm: float,
    radius_mm: float,
    inertia_kg_mm2: float,
    bearing_damping_nmm_s: float,
) -> float:
    """``ζ = (1000·c/J)/(2·ω_n)``——**唯一的阻尼来自轴承**。

    ``c = 0``给``ζ = 0``：跨段是一个**无阻尼**振子，
    速度扰动引起的张力振荡永远不衰减。
    """

    damping = _require_finite(bearing_damping_nmm_s, "bearing_damping_nmm_s")
    if damping < 0.0:
        raise TransportError(f"bearing_damping_nmm_s不能为负: {damping!r}")
    inertia = _require_positive(inertia_kg_mm2, "inertia_kg_mm2")
    natural = span_natural_frequency_rad_s(
        span_stiffness_n_per_mm=span_stiffness_n_per_mm,
        radius_mm=radius_mm,
        inertia_kg_mm2=inertia_kg_mm2,
    )
    return (MM_PER_M * damping / inertia) / (2.0 * natural)


def velocity_step_overshoot(damping_ratio: float) -> float:
    """收线端速度阶跃下张力的**相对超调**
    ``exp(−ζ(π − acos ζ)/√(1−ζ²)) / (2ζ)``。

    **它不是`drives.step_response_overshoot`**：那一条是没有零点的标准二阶阶跃
    （``Φ = π``、无前因子），这一条的初值带一个斜率冲击（张力连续而速度差当场跳），
    等价于传递函数多一个零点。``ζ = 0.5``时前者0.16303、本式0.29842。

    **也不是`contact.restitution_from_damping_ratio`**，尽管
    ``ζ = 0.5``处两者**逐位相同**——那是``π − acos ζ = 2·acos ζ``与
    ``1/(2ζ) = 1``在同一个点上同时发生的巧合。模块docstring记着这个盲区。

    ``ζ → 0``时它发散：无阻尼时稳态变化为零而振荡幅值不为零，
    相对超调因此无界。**那是物理不是数值**——要看绝对幅值就去看
    ``ΔT_ss·(1 + 超调)``。
    """

    ratio = _require_finite(damping_ratio, "damping_ratio")
    if not (0.0 < ratio < 1.0):
        raise TransportError(
            f"damping ratio must be in (0, 1): {ratio!r} —— "
            "ζ = 0时相对超调无界（稳态变化为零），ζ ≥ 1时没有峰值，"
            "两端都不是这条闭式的定义域"
        )
    phi = math.pi - math.acos(ratio)
    return math.exp(-ratio * phi / math.sqrt(1.0 - ratio * ratio)) / (2.0 * ratio)


def velocity_step_peak_time_s(
    *, natural_frequency_rad_s: float, damping_ratio: float
) -> float:
    """峰值时刻``t_p = (π − acos ζ)/(ω_n·√(1−ζ²))``。

    标准二阶阶跃是``π/ω_d``；**这里的斜率冲击把峰值往前挪了``acos ζ``**。
    ``ζ → 0``时它趋于``(π/2)/ω_n``——**四分之一周期**，正是无阻尼时
    ``sin(ω_n t)``的第一个峰。
    """

    natural = _require_positive(natural_frequency_rad_s, "natural_frequency_rad_s")
    ratio = _require_finite(damping_ratio, "damping_ratio")
    if not (0.0 <= ratio < 1.0):
        raise TransportError(f"peak time needs an underdamped ratio: {ratio!r}")
    return (math.pi - math.acos(ratio)) / (natural * math.sqrt(1.0 - ratio * ratio))


# --------------------------------------------------------------------- 推进 ---


@dataclass(frozen=True)
class SpanTransportSample:
    """一步的观测，全部取**步首**状态。**产物是这个，不是内部状态。**"""

    time_s: float
    material_length_mm: float
    tension_n: float
    strain: float
    angular_velocity_rad_s: float
    #: 步首的``ω·R``。**长度账里用的是步末的``ω``**（半隐式），
    #: 两者差一个``dt·α·R``——记步首是因为样点是"``time_s``时刻的状态快照"。
    payout_speed_mm_s: float
    takeup_speed_mm_s: float
    brake_torque_nmm: float
    #: 本步生效的路径增量（横向侵入让直线段变成折线的那一份）。
    #: **默认0.0＝没有东西碰它**，见`FreeSpan.path_length_mm`。
    path_excess_mm: float = 0.0


@dataclass(frozen=True)
class SpanTransportLoop:
    """跨段输运的推进器：状态是``(材料长度, 放线盘转速)``，两个输入是
    ``(制动力矩, 收线端速度)``。

    ## 两个输入正是真机的两个执行器

    ``brake_torque_nmm``经`drives.MagneticParticleClutch`由线圈电流生成，
    ``takeup_speed_mm_s``是`ED3L-08AEA`伺服在CSP位置模式下的给定。
    **本类不替它们做闭环**——控制器接在哪一路上是使用者的事，
    本类只负责把两路输入变成张力。

    ## 半隐式Euler，且离散不动点＝连续不动点

        ω_{n+1} = ω_n + dt·α(T_n, ω_n)
        L_{n+1} = L_n + dt·(ω_{n+1}·R − v_收线)

    不动点要求``α = 0``且``ω·R = v_收线``，与连续稳态**逐字相同**。
    **离散化没有给稳态引入偏差**——所以稳态判据判的是模型，不是步长。

    ``forbid_slack``**没有默认值**：松弛（``ε ≤ 0``）时``L_geo``不再是跨段的长度，
    带材垂成一条悬链线，此后这个模型给出的一切都不再有意义。
    容忍还是当场炸，**是声明者要拿的主意**。
    """

    span: FreeSpan
    reel: PayoutReel
    dt_s: float
    material_length_mm: float
    angular_velocity_rad_s: float
    #: ``True`` ⟹ 一旦进入松弛当场失败关闭。见类docstring。
    forbid_slack: bool
    step_index: int = 0

    def __post_init__(self) -> None:
        _require_positive(self.dt_s, "dt_s")
        _require_positive(self.material_length_mm, "material_length_mm")
        _require_finite(self.angular_velocity_rad_s, "angular_velocity_rad_s")
        if not isinstance(self.forbid_slack, bool):
            raise TransportError(
                f"forbid_slack must be an explicit bool: {self.forbid_slack!r}"
            )
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TransportError(f"step_index must be an int: {self.step_index!r}")
        #: 构造期就把方向门上膛：两个偏导的乘积必须为负。
        #: 放在这里而不是每步判，是因为它是**结构性质**——每步判只是把同一个
        #: 结论算几万遍。
        assert_tension_feedback_is_negative(
            tension_feedback_gain_per_s2(
                tension_rate_per_payout_speed_n_per_mm=(
                    -self.span.stiffness_n_per_mm(self.material_length_mm)
                ),
                payout_acceleration_per_tension_mm_s2_per_n=(
                    self.reel.payout_acceleration_per_tension_mm_s2_per_n()
                ),
            )
        )

    @classmethod
    def at_steady_state(
        cls,
        *,
        span: FreeSpan,
        reel: PayoutReel,
        dt_s: float,
        brake_torque_nmm: float,
        line_speed_mm_s: float,
        forbid_slack: bool,
    ) -> SpanTransportLoop:
        """从**闭式稳态**起手：``T* = M/R + c·v/R²``、``L* = EA·L_geo/(EA+T*)``、
        ``ω* = v/R``。

        起点取闭式而不是"跑一段等它稳下来"，理由与`capstan_tension_ratio`
        那条"起点必须已经穿透"同源：**判据要判的东西不该被起点的瞬态污染**。
        """

        speed = _require_finite(line_speed_mm_s, "line_speed_mm_s")
        if speed <= 0.0:
            raise TransportError(
                f"line_speed_mm_s = {speed!r} —— 稳态起点要求放线盘在转（ω > 0）"
            )
        tension = steady_state_tension_n(
            brake_torque_nmm=brake_torque_nmm,
            radius_mm=reel.radius_mm,
            bearing_damping_nmm_s=reel.bearing_damping_nmm_s,
            line_speed_mm_s=speed,
        )
        return cls(
            span=span,
            reel=reel,
            dt_s=dt_s,
            material_length_mm=span.material_length_for_tension_mm(tension),
            angular_velocity_rad_s=speed / reel.radius_mm,
            forbid_slack=forbid_slack,
        )

    @property
    def tension_n(self) -> float:
        """当前张力——**由材料长度算出，不是一个独立的状态**。"""

        return self.span.tension_n(self.material_length_mm)

    def step(
        self,
        *,
        brake_torque_nmm: float,
        takeup_speed_mm_s: float,
        path_excess_mm: float = 0.0,
    ) -> tuple[SpanTransportLoop, SpanTransportSample]:
        """走一步，返回``(新回路, 本步观测)``。观测是**步首**快照。

        ``path_excess_mm``是**外扰通道之二**：横向侵入让带材走折线，路径变长。
        它与``takeup_speed_mm_s``（外扰通道之一，臂动经落位点几何进来）
        **不是同一件事，也不许互相折算**：路径增量进的是应变的分子，
        收线端速度进的是长度账的导数。两者只在``L_mat/L_path``这个``O(ε)``因子上
        可以互相换算，而把那个因子丢掉正是本仓最怕的那种静默错。
        """

        takeup = _require_finite(takeup_speed_mm_s, "takeup_speed_mm_s")
        #: 先把增量验掉（非负、有限），**原样记进样点**——
        #: 用``path_length_mm``减回来会掉几个ulp，而样点是要被逐位比对的产物。
        excess = self.span.require_path_excess_mm(path_excess_mm)
        strain = self.span.strain(self.material_length_mm, path_excess_mm=path_excess_mm)
        if self.forbid_slack and strain <= MIN_STRAIN:
            raise TransportError(
                f"第{self.step_index}步：应变{strain!r} ≤ 0，跨段已经松了。"
                "松弛时几何长度不再是跨段的长度（带材垂成悬链线），"
                "此后这个模型给出的一切都不再有意义——forbid_slack=True故当场关闭"
            )
        tension = self.span.axial_stiffness_n * strain if strain > MIN_STRAIN else 0.0
        acceleration = self.reel.angular_acceleration_rad_s2(
            tension_n=tension,
            brake_torque_nmm=brake_torque_nmm,
            angular_velocity_rad_s=self.angular_velocity_rad_s,
        )
        sample = SpanTransportSample(
            time_s=self.step_index * self.dt_s,
            material_length_mm=self.material_length_mm,
            tension_n=tension,
            strain=strain,
            angular_velocity_rad_s=self.angular_velocity_rad_s,
            payout_speed_mm_s=self.reel.payout_speed_mm_s(self.angular_velocity_rad_s),
            takeup_speed_mm_s=takeup,
            brake_torque_nmm=brake_torque_nmm,
            path_excess_mm=excess,
        )
        omega = self.angular_velocity_rad_s + self.dt_s * acceleration
        length = self.material_length_mm + self.dt_s * (omega * self.reel.radius_mm - takeup)
        if not (length > 0.0 and math.isfinite(length)):
            raise TransportError(
                f"第{self.step_index}步：材料长度推到{length!r} —— "
                "跨段里的材料被抽空了，步长太大或收线端速度不是这条链路能跟上的"
            )
        return (
            replace(
                self,
                material_length_mm=length,
                angular_velocity_rad_s=omega,
                step_index=self.step_index + 1,
            ),
            sample,
        )

    def run(
        self, steps: int, *, brake_torque_nmm: float, takeup_speed_mm_s: float
    ) -> tuple[SpanTransportLoop, tuple[SpanTransportSample, ...]]:
        """两个输入都恒定地连走``steps``步。"""

        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise TransportError(f"steps must be a positive int: {steps!r}")
        loop = self
        samples: list[SpanTransportSample] = []
        for _ in range(steps):
            loop, sample = loop.step(
                brake_torque_nmm=brake_torque_nmm, takeup_speed_mm_s=takeup_speed_mm_s
            )
            samples.append(sample)
        return loop, tuple(samples)

    def run_timeline(
        self, timeline: MaterialFeedTimeline, *, brake_torque_nmm: float, steps: int
    ) -> tuple[SpanTransportLoop, tuple[SpanTransportSample, ...]]:
        """收线端速度由上游的喂料长度时间线给——**这就是那条接缝**。

        每步取``timeline.speed_mm_s(t)``，``t``是**步首**时刻
        （段内恒速语义下速率在样点处右连续，所以步首取值就是这一步的速率）。
        """

        if not isinstance(timeline, MaterialFeedTimeline):
            raise TransportError(f"timeline must be a MaterialFeedTimeline: {timeline!r}")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise TransportError(f"steps must be a positive int: {steps!r}")
        loop = self
        samples: list[SpanTransportSample] = []
        for _ in range(steps):
            loop, sample = loop.step(
                brake_torque_nmm=brake_torque_nmm,
                takeup_speed_mm_s=timeline.speed_mm_s(loop.step_index * loop.dt_s),
            )
            samples.append(sample)
        return loop, tuple(samples)


__all__ = [
    "EXTRAPOLATIONS",
    "MIN_STRAIN",
    "MM_PER_M",
    "RATE_SEMANTICS",
    "FreeSpan",
    "MaterialFeedTimeline",
    "PayoutReel",
    "SpanTransportLoop",
    "SpanTransportSample",
    "TransportError",
    "assert_span_transport_directions",
    "assert_tension_feedback_is_negative",
    "span_damping_ratio",
    "span_natural_frequency_rad_s",
    "steady_state_tension_n",
    "tension_feedback_gain_per_s2",
    "velocity_step_overshoot",
    "velocity_step_peak_time_s",
]
