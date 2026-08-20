"""运动来源——spec/10第二节的执行体。

规范原话：

    **"路径从哪来"与"物理怎么算"解耦。** 外部只写位姿、不写力（MuJoCo mocap形制）：

        pose_at(t_s) -> Pose
        horizon_s() -> float
        is_replayable() -> bool   # 不可重放的来源禁止进入需要确定性的运行

    WII位姿时间线是第一个实现候选；解析轨迹是第二个；将来的控制器是第三个。
    ``is_replayable``与轴3规则5（复现指纹）联动：不可重放来源的运行不得声称指纹。

## 本模块的范围：**运动学，不是力学**

位姿来源**不是状态**（spec/12第2.3节原话："``MotionSource``喂进来的位姿时间线是
外部驱动，不是被求解的自由度"）。所以本模块不import任何力学模块，也不会去碰
能量项、求解器或状态数组——它回答的只是"``t``时刻那个东西在哪"。

与``sensors.py``把``read``留在门外**不同**，``pose_at``在这里是实现的。理由不是
"这块比较简单"，而是二者的性质不同：``read``要跨过"物理上真测得到"这条判据去读
状态，那是接口冻结前不该替它拍板的事；而``pose_at``只是把**已声明的**离散样点按
**已声明的**语义取值，它不产生任何新物理。本模块与``sensors.py``同纪律的地方在
另一处：**语义不许由库来猜**（见下）。

## 插值语义为什么必须由声明者逐条给出

消费方WII发布的``wii_motion_timeline``只给**离散样点**：``sample_times_s``、
``translation_parent_from_child_mm``、``rotation_parent_from_child_xyzw``、
外加显式的``pause_intervals``。**样点之间发生了什么，artifact里没有这个信息**——
它在声明者的脑子里。库替他挑一个"合理默认"的后果是确定的：两个调用方拿同一份
样点算出不同的位姿、不同的接触时刻、不同的物理，而**两边都以为自己是对的**。
这正是本仓最怕的那种"跑得通但全错"。

所以``InterpolationSemantics``的五条**逐条必须显式给出，缺一即拒**，
且每条都是失败关闭的白名单（加一个取值要改本文件并补测试，不许在调用方就地放行）：

1. ``translation_interpolation``——位置在样点之间是线性走还是零阶保持。
   这两条给出的速度完全不同（线性给分段常速度，保持给冲击），而WII自己算twist时
   用的是分段线性；一个回放控制器设定点的消费方要的却是保持。
2. ``rotation_interpolation``——``slerp``（等角速度）、``nlerp``（便宜但角速度不均匀）
   还是零阶保持。
3. ``rotation_arc``——**这一条最容易被漏掉**：``q``与``−q``是同一个旋转，
   但在它们之间插值走的是**相反的两条弧**。WII在发布时按规范符号把四元数
   归一化（其``_canonical_quaternion_xyzw``取``(w,x,y,z)``首个非零分量为正），
   这个归一化**不是**"取短弧"——它会把一段本该走长弧的运动翻成短弧，反之亦然。
   走短弧还是照声明的符号走，只有声明者知道。
4. ``pause_hold``——暂停区间内取哪个值。WII的时序模型明写零运动段"没有可推断的
   时长，必须用显式时间戳加``pause_intervals``编码"，于是暂停区间是一等公民；
   区间内是保持区间起点的位姿，还是照常插值穿过去，是两种不同的物理。
5. ``extrapolation``——样点之外怎么办。只有"拒绝"与"夹到端点"两个取值；
   **不提供线性外推**，因为外推是在没有任何样点支持的地方发明运动，
   要外推就去多给样点。

第4条即使这份时间线今天一个暂停区间都没有，也必须声明。理由：有没有暂停区间是
**数据**的事，暂停区间怎么取值是**声明者**的事；今天没有不等于明天喂进来的那条
时间线也没有，而那时语义不能因此退回"由库来猜"。

## 面（轴1规则1）

本模块**不落盘、不跨边界**，因此**不需要新的面**。位姿时间线哪天要写进场景文件或
run package，那时才需要一个``physics_motion_timeline``面，且要先去
``engine_facets.py``登记再落盘——那个文件是闸门。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from physics_engine.identity import parse_namespace_id


class MotionError(ValueError):
    """运动来源层的一切失败关闭。"""


#: 单位四元数的绝对容差。**与``shapes.PosedBody``逐位相同**（其``abs_tol=1.0e-9``）——
#: 两处判"这是不是一个单位四元数"必须给同一个答案，否则一个位姿在``motion``里合法、
#: 装进``PosedBody``却被拒（或反过来），而那种不一致最难查。
QUATERNION_NORM_ABS_TOL = 1.0e-9

#: 位姿平移的单位。**本仓已经栽过两次1000倍单位bug**，所以这一条写成有名字的常量
#: 并由白名单守着：以米声明的时间线会被当场拒，而不是被静默读成千分之一。
POSE_TRANSLATION_UNIT = "mm"
ACCEPTED_TRANSLATION_UNITS: frozenset[str] = frozenset({POSE_TRANSLATION_UNIT})
MILLIMETRES_PER_METRE = 1000.0

#: 小角退化阈值，判据是``1 − |cos θ|``。**这个数是算出来的**：
#: θ=1e-3 rad处``1 − cos θ = 5.0e-7``，实测slerp与nlerp的最大分量分歧为1.95e-12,
#: 比``QUATERNION_NORM_ABS_TOL``小约500倍——在这以下两条路给出的四元数在本仓的
#: 单位容差下**不可分辨**，因此走便宜的那条（也顺带避开``sin θ → 0``的除法）。
#: 对照：θ=1e-2 rad处分歧已达1.95e-9，**越过**容差，所以阈值不能再放宽。
#: 推导脚本与实测表见决策0038第四节。
SMALL_ANGLE_ONE_MINUS_COS = 5.0e-7

#: 五条插值语义各自的白名单（失败关闭；轴2"只增不改"同样适用）。
TRANSLATION_INTERPOLATIONS: frozenset[str] = frozenset({"linear", "hold_previous"})
ROTATION_INTERPOLATIONS: frozenset[str] = frozenset({"slerp", "nlerp", "hold_previous"})
ROTATION_ARCS: frozenset[str] = frozenset({"shortest", "as_declared", "not_applicable"})
PAUSE_HOLDS: frozenset[str] = frozenset({"hold_interval_start", "interpolate_through"})
EXTRAPOLATIONS: frozenset[str] = frozenset({"reject", "clamp_to_endpoint"})


# ---------------------------------------------------------------- 校验原语 ---
# 这些是**有名字的模块级函数**而不是内联代码，为的是测试能把某一条换成空操作
# （等价于把它写成``if False``），从而证明红是**那一条**红的，不是别的规则顺手拦下的。


def _require_namespace(value: object, prefix: str, what: str) -> str:
    if not isinstance(value, str):
        raise MotionError(f"{what} must be a string: {value!r}")
    if not value.startswith(f"{prefix}/"):
        raise MotionError(f"{what} must be namespaced like {prefix!r}/…: {value!r}")
    try:
        parse_namespace_id(value)
    except ValueError as error:  # IdentityError继承自ValueError
        raise MotionError(f"{what} is not a valid namespace id: {error}") from error
    return value


def _require_finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionError(f"{what} must be a real number: {value!r}")
    if not math.isfinite(value):
        raise MotionError(f"{what} must be finite: {value!r}")
    return float(value)


def _require_declared_choice(value: object, allowed: frozenset[str], what: str) -> str:
    """一条插值语义。**缺省不受理**——没有默认值可用，白名单外一律拒。"""

    if value is None:
        raise MotionError(
            f"{what} must be declared explicitly — 插值语义是声明者的事，不是库替他猜的。"
            f"可选：{sorted(allowed)}"
        )
    if not isinstance(value, str) or value not in allowed:
        raise MotionError(
            f"{what} must be one of {sorted(allowed)}, got {value!r} — "
            "白名单失败关闭；加一个取值要改motion.py并补一条测试"
        )
    return value


def _require_arc_matches_rotation(rotation: str, arc: str) -> None:
    """弧向与旋转插值的相容性：零阶保持没有弧可走，slerp/nlerp必须有。"""

    holds = rotation == "hold_previous"
    if holds and arc != "not_applicable":
        raise MotionError(
            f"rotation_interpolation={rotation!r}下不存在插值弧，"
            f"rotation_arc必须显式写'not_applicable'，得到{arc!r}"
        )
    if not holds and arc == "not_applicable":
        raise MotionError(
            f"rotation_interpolation={rotation!r}要在两个四元数之间走一条弧，"
            "rotation_arc不能是'not_applicable'——"
            "q与−q是同一个旋转但插值走相反的弧，走哪条只有声明者知道"
        )


def _require_unit_quaternion(rotation_xyzw: object, what: str) -> tuple[float, ...]:
    if not isinstance(rotation_xyzw, tuple) or len(rotation_xyzw) != 4:
        raise MotionError(f"{what} must be a 4-tuple (x, y, z, w): {rotation_xyzw!r}")
    components = tuple(
        _require_finite(value, f"{what}[{index}]")
        for index, value in enumerate(rotation_xyzw)
    )
    norm = math.sqrt(sum(value * value for value in components))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=QUATERNION_NORM_ABS_TOL):
        raise MotionError(f"{what} must be a unit quaternion, norm is {norm!r}")
    return components


def _require_translation_unit(unit: object, source_id: str) -> str:
    if unit not in ACCEPTED_TRANSLATION_UNITS:
        raise MotionError(
            f"{source_id}: translation_unit must be one of "
            f"{sorted(ACCEPTED_TRANSLATION_UNITS)}, got {unit!r} — "
            f"以米声明的时间线会整整差{MILLIMETRES_PER_METRE:.0f}倍，"
            "本仓已经栽过两次这个bug，所以这里失败关闭而不是替你换算"
        )
    return str(unit)


def _require_sample_times(samples: tuple[PoseSample, ...], source_id: str) -> None:
    """样点时间：至少两个、从run_start=0起、严格递增。

    "从0起"与"严格递增"两条直接采WII的``_strict_times``（其原话
    ``time_s must start at run_start=0`` / ``time_s must be strictly increasing``）——
    两仓对同一份时间线的时间轴必须是同一个约定，否则``horizon_s()``是谁的0起算
    就成了口头协议。
    """

    if len(samples) < 2:
        raise MotionError(f"{source_id}: a sampled timeline needs at least two samples")
    if samples[0].time_s != 0.0:
        raise MotionError(
            f"{source_id}: sample times must start at run_start=0, "
            f"got {samples[0].time_s!r}"
        )
    for left, right in zip(samples, samples[1:], strict=False):
        if right.time_s <= left.time_s:
            raise MotionError(
                f"{source_id}: sample times must be strictly increasing, "
                f"got {left.time_s!r} then {right.time_s!r}"
            )


def _require_pauses_in_range(
    pauses: tuple[PauseInterval, ...], horizon_s: float, source_id: str
) -> None:
    """暂停区间：有序、互不重叠、落在``[0, horizon]``内（WII的``_pause_intervals``同款）。"""

    previous_end = -1.0
    seen: set[str] = set()
    for pause in pauses:
        if pause.pause_id in seen:
            raise MotionError(f"{source_id}: duplicate pause {pause.pause_id!r}")
        seen.add(pause.pause_id)
        if pause.start_time_s <= previous_end:
            raise MotionError(
                f"{source_id}: pause intervals must be ordered and disjoint, "
                f"{pause.pause_id!r} starts at {pause.start_time_s!r} but the previous "
                f"one ends at {previous_end!r}"
            )
        if pause.end_time_s > horizon_s:
            raise MotionError(
                f"{source_id}: pause {pause.pause_id!r} ends at {pause.end_time_s!r}, "
                f"past the timeline horizon {horizon_s!r}"
            )
        previous_end = pause.end_time_s


# -------------------------------------------------------------------- 位姿 ---


@dataclass(frozen=True)
class Pose:
    """一个位姿。表示法与``shapes.PosedBody``**逐字段相同**：毫米平移 + xyzw单位四元数。

    没有复用``PosedBody``是因为那个类型要求一个``SimBody``——位姿来源不知道也
    不需要知道被驱动的是哪个体（spec/11规则3"运动体与形状解耦"）。
    """

    translation_mm: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.translation_mm, tuple) or len(self.translation_mm) != 3:
            raise MotionError(f"translation_mm must be a 3-tuple: {self.translation_mm!r}")
        for index, value in enumerate(self.translation_mm):
            _require_finite(value, f"translation_mm[{index}]")
        _require_unit_quaternion(self.rotation_xyzw, "rotation_xyzw")


@dataclass(frozen=True)
class PoseSample:
    """时间线上的一个样点。"""

    time_s: float
    pose: Pose

    def __post_init__(self) -> None:
        _require_finite(self.time_s, "time_s")
        if not isinstance(self.pose, Pose):
            raise MotionError(f"sample at {self.time_s!r} is not a Pose: {self.pose!r}")


@dataclass(frozen=True)
class PauseInterval:
    """一段显式的暂停。形制取自WII的``pause_intervals``（去掉其纯回放字段）。"""

    pause_id: str
    start_time_s: float
    end_time_s: float
    reason: str

    def __post_init__(self) -> None:
        _require_namespace(self.pause_id, "pause", "pause_id")
        _require_finite(self.start_time_s, f"{self.pause_id}: start_time_s")
        _require_finite(self.end_time_s, f"{self.pause_id}: end_time_s")
        if self.start_time_s < 0.0 or self.end_time_s <= self.start_time_s:
            raise MotionError(
                f"{self.pause_id}: a pause needs 0 <= start < end, got "
                f"{self.start_time_s!r}..{self.end_time_s!r}"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise MotionError(
                f"{self.pause_id}: a pause must say why it is there — "
                "一段没有理由的暂停与一段丢失的数据在字节上没有区别"
            )


@dataclass(frozen=True)
class InterpolationSemantics:
    """五条插值语义。**逐条显式，缺一即拒**——理由见模块文档。"""

    translation_interpolation: str
    rotation_interpolation: str
    rotation_arc: str
    pause_hold: str
    extrapolation: str

    def __post_init__(self) -> None:
        _require_declared_choice(
            self.translation_interpolation,
            TRANSLATION_INTERPOLATIONS,
            "translation_interpolation",
        )
        _require_declared_choice(
            self.rotation_interpolation, ROTATION_INTERPOLATIONS, "rotation_interpolation"
        )
        _require_declared_choice(self.rotation_arc, ROTATION_ARCS, "rotation_arc")
        _require_declared_choice(self.pause_hold, PAUSE_HOLDS, "pause_hold")
        _require_declared_choice(self.extrapolation, EXTRAPOLATIONS, "extrapolation")
        _require_arc_matches_rotation(self.rotation_interpolation, self.rotation_arc)


# ------------------------------------------------------------ 接口与两实现 ---


@runtime_checkable
class MotionSource(Protocol):
    """spec/10第二节的三个方法。**只有这三个**——本Protocol不替规范加字段。

    识别用的``source_id``**故意不进这个Protocol**：spec/10没写它，而本页尚未冻结,
    往未冻结的接口上加字段等于替它拍板。门函数用``getattr``取它，取不到就用``repr``。
    """

    def pose_at(self, t_s: float) -> Pose: ...

    def horizon_s(self) -> float: ...

    def is_replayable(self) -> bool: ...


def _normalise_quaternion(components: tuple[float, ...], what: str) -> tuple[float, ...]:
    """插值结果投回单位球。这不是语义变换，是把一次浮点舍入的偏离收回来。"""

    norm = math.sqrt(sum(value * value for value in components))
    if norm == 0.0:
        raise MotionError(f"{what}: interpolated quaternion collapsed to zero")
    return tuple(value / norm for value in components)


def _interpolate_rotation(
    left: tuple[float, ...],
    right: tuple[float, ...],
    u: float,
    semantics: InterpolationSemantics,
) -> tuple[float, ...]:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    if semantics.rotation_arc == "shortest" and dot < 0.0:
        right = tuple(-value for value in right)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot <= -(1.0 - SMALL_ANGLE_ONE_MINUS_COS):
        # 只有``as_declared``到得了这里（``shortest``翻符号后``dot >= 0``）。
        # ``dot ≈ −1``是两个四元数**反号**，不是"旋转差180°"（那对应``dot ≈ 0``）：
        # 反号的一对描述的是同一个姿态，中间那条弧是绕**某个说不出来的轴**转一整圈。
        raise MotionError(
            "两个相邻样点的四元数接近反号（q1 ≈ −q0），rotation_arc='as_declared'下"
            f"这是一条绕任意轴转360°的路径，转轴不唯一 (dot={dot!r})——"
            "这不是数值问题，是声明问题：加一个中间样点把那一圈写明白，"
            "或改用rotation_arc='shortest'（它把这一对当成同一个姿态）"
        )
    if (
        semantics.rotation_interpolation == "nlerp"
        or dot >= 1.0 - SMALL_ANGLE_ONE_MINUS_COS
    ):
        return _normalise_quaternion(
            tuple(a + (b - a) * u for a, b in zip(left, right, strict=True)), "nlerp"
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    left_weight = math.sin((1.0 - u) * theta) / sin_theta
    right_weight = math.sin(u * theta) / sin_theta
    return _normalise_quaternion(
        tuple(
            left_weight * a + right_weight * b
            for a, b in zip(left, right, strict=True)
        ),
        "slerp",
    )


def _interpolate(
    left: Pose, right: Pose, u: float, semantics: InterpolationSemantics
) -> Pose:
    """``u ∈ (0, 1)``。端点由调用方短路——见``SampledPoseTimeline.pose_at``。"""

    if semantics.translation_interpolation == "linear":
        translation = tuple(
            a + (b - a) * u
            for a, b in zip(left.translation_mm, right.translation_mm, strict=True)
        )
    else:  # hold_previous
        translation = left.translation_mm
    if semantics.rotation_interpolation == "hold_previous":
        rotation = left.rotation_xyzw
    else:
        rotation = _interpolate_rotation(
            left.rotation_xyzw, right.rotation_xyzw, u, semantics
        )
    return Pose(
        translation_mm=translation,  # type: ignore[arg-type]
        rotation_xyzw=rotation,  # type: ignore[arg-type]
    )


def interpolate_pose_fraction(
    left: Pose,
    right: Pose,
    fraction: float,
    semantics: InterpolationSemantics,
) -> Pose:
    """按声明语义在两个位姿之间取无量纲分数，不赋予它时间含义。

    ``SampledPoseTimeline``的旧热路径继续直接走私有``_interpolate``，避免为了
    新的planning-scale调用面改变既有数值路径。本函数给无时间计划复用同一套
    线性/SLERP语义；端点原对象直接返回，保持指纹稳定。
    """

    if not isinstance(left, Pose) or not isinstance(right, Pose):
        raise MotionError("interpolate_pose_fraction expects two Pose values")
    if not isinstance(semantics, InterpolationSemantics):
        raise MotionError("semantics must be InterpolationSemantics")
    value = _require_finite(fraction, "fraction")
    if not 0.0 <= value <= 1.0:
        raise MotionError(f"fraction must live in [0, 1], got {value!r}")
    if value == 0.0:
        return left
    if value == 1.0:
        return right
    return _interpolate(left, right, value, semantics)


@dataclass(frozen=True)
class SampledPoseTimeline:
    """离散样点 + **显式声明的**插值语义。WII位姿时间线的形制。

    没有一个字段带默认值：本层的每一条都是声明者要拿主意的东西，
    而"默认值"正是把主意悄悄替他拿了的那种写法。
    """

    source_id: str
    samples: tuple[PoseSample, ...]
    semantics: InterpolationSemantics
    translation_unit: str
    pauses: tuple[PauseInterval, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.source_id, "motion", "source_id")
        if not isinstance(self.semantics, InterpolationSemantics):
            raise MotionError(
                f"{self.source_id}: semantics must be an InterpolationSemantics, "
                f"got {self.semantics!r}"
            )
        _require_translation_unit(self.translation_unit, self.source_id)
        for sample in self.samples:
            if not isinstance(sample, PoseSample):
                raise MotionError(f"{self.source_id}: not a PoseSample: {sample!r}")
        _require_sample_times(self.samples, self.source_id)
        for pause in self.pauses:
            if not isinstance(pause, PauseInterval):
                raise MotionError(f"{self.source_id}: not a PauseInterval: {pause!r}")
        _require_pauses_in_range(self.pauses, self.samples[-1].time_s, self.source_id)

    # -- spec/10第二节的三个方法 ------------------------------------------

    def pose_at(self, t_s: float) -> Pose:
        t_s = _require_finite(t_s, f"{self.source_id}: t_s")
        horizon = self.horizon_s()
        if t_s < 0.0 or t_s > horizon:
            if self.semantics.extrapolation == "reject":
                raise MotionError(
                    f"{self.source_id}: t_s={t_s!r} is outside [0, {horizon!r}] and "
                    "extrapolation='reject' — 样点之外没有任何样点支持的运动，"
                    "要外推就去多给样点"
                )
            t_s = 0.0 if t_s < 0.0 else horizon
        if self.semantics.pause_hold == "hold_interval_start":
            for pause in self.pauses:
                if pause.start_time_s <= t_s <= pause.end_time_s:
                    t_s = pause.start_time_s
                    break
        if t_s >= horizon:
            return self.samples[-1].pose
        index = bisect_right(self.samples, t_s, key=lambda sample: sample.time_s) - 1
        left = self.samples[index]
        right = self.samples[index + 1]
        u = (t_s - left.time_s) / (right.time_s - left.time_s)
        # 端点逐字节返回样点本身。``a + (b − a)·1.0``不保证等于``b``,
        # 所以两个端点都不许走插值——否则``pose_at(样点时刻)``会与样点差几个ULP,
        # 而那种差在指纹上是可见的。
        if u == 0.0:
            return left.pose
        return _interpolate(left.pose, right.pose, u, self.semantics)

    def horizon_s(self) -> float:
        """最后一个样点的时刻。因为时间轴从0起，它同时就是时间线的时长。"""

        return self.samples[-1].time_s

    def is_replayable(self) -> bool:
        """恒真，理由要说清楚。

        可重放性说的是**这个来源对象**能不能重放，不是那些数字当初怎么来的：
        样点一旦在手，它就是一张有限的表，配上已声明的确定性语义，
        同一个``t_s``永远给同一个位姿。哪怕那些数字来自一次一次性的实机采集,
        重放这张表仍然逐字节可复现。
        """

        return True


@dataclass(frozen=True)
class AnalyticPose:
    """解析轨迹：一个纯函数``t → 位姿``。

    **五条插值语义在这里只需要一条**（``extrapolation``），因为另外四条问的都是
    "样点之间怎么办"，而解析轨迹没有样点。定义域之外仍然要声明——``horizon_s()``
    是一条被声明的边界，越过它发生什么同样不许由库来猜。

    ``replayable``必须由声明者显式给出：本模块拿到的是一个任意可调用对象,
    它可以去读时钟、读随机数、读文件。**库证明不了一个函数是纯的**，
    所以这里做的是**证伪**而不是证明：声明为可重放时，构造期在
    ``replay_probe_times_s``上各求值两次，逐字节不同即当场拒。
    这条能抓住读时钟/读随机数这类最常见的写法，**抓不住**"每逢质数秒返回别的值"
    那种——如实登记在此，不许被"过了校验"盖过。
    """

    source_id: str
    pose_fn: Callable[[float], Pose]
    declared_horizon_s: float
    extrapolation: str
    replayable: bool
    replay_probe_times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.source_id, "motion", "source_id")
        if not callable(self.pose_fn):
            raise MotionError(f"{self.source_id}: pose_fn must be callable")
        horizon = _require_finite(self.declared_horizon_s, f"{self.source_id}: horizon")
        if horizon <= 0.0:
            raise MotionError(
                f"{self.source_id}: declared_horizon_s must be positive, got {horizon!r}"
            )
        _require_declared_choice(
            self.extrapolation, EXTRAPOLATIONS, f"{self.source_id}: extrapolation"
        )
        if type(self.replayable) is not bool:
            raise MotionError(
                f"{self.source_id}: replayable must be a real bool, "
                f"got {self.replayable!r} — 可重放性是一条声明，不是一个真值性"
            )
        if self.replayable:
            self._probe_determinism()

    def _probe_determinism(self) -> None:
        if not self.replay_probe_times_s:
            raise MotionError(
                f"{self.source_id}: replayable=True needs at least one probe time — "
                "一条没有配证伪尝试的可重放声明就是冒充（AGENTS.md诚实可信度）"
            )
        for t_s in self.replay_probe_times_s:
            probe = _require_finite(t_s, f"{self.source_id}: probe time")
            if probe < 0.0 or probe > self.declared_horizon_s:
                raise MotionError(
                    f"{self.source_id}: probe time {probe!r} is outside "
                    f"[0, {self.declared_horizon_s!r}]"
                )
            first = self._evaluate(probe)
            second = self._evaluate(probe)
            if first != second:
                raise MotionError(
                    f"{self.source_id}: pose_fn gave two different poses at t_s={probe!r} "
                    f"({first} then {second}) — 它不是纯函数，replayable=True不成立。"
                    "轴3规则5：不可重放来源的运行不得声称复现指纹"
                )

    def _evaluate(self, t_s: float) -> Pose:
        pose = self.pose_fn(t_s)
        if not isinstance(pose, Pose):
            raise MotionError(
                f"{self.source_id}: pose_fn returned {pose!r}, not a Pose"
            )
        return pose

    # -- spec/10第二节的三个方法 ------------------------------------------

    def pose_at(self, t_s: float) -> Pose:
        t_s = _require_finite(t_s, f"{self.source_id}: t_s")
        if t_s < 0.0 or t_s > self.declared_horizon_s:
            if self.extrapolation == "reject":
                raise MotionError(
                    f"{self.source_id}: t_s={t_s!r} is outside "
                    f"[0, {self.declared_horizon_s!r}] and extrapolation='reject'"
                )
            t_s = 0.0 if t_s < 0.0 else self.declared_horizon_s
        return self._evaluate(t_s)

    def horizon_s(self) -> float:
        return self.declared_horizon_s

    def is_replayable(self) -> bool:
        return self.replayable


# ------------------------------------------------ 轴3规则5的联动门 ---------


def assert_replayable_for_fingerprint(
    sources: Iterable[MotionSource], *, run_label: str
) -> None:
    """**不可重放的来源，其运行不得声称复现指纹**（spec/10第二节 × 轴3规则5）。

    要声称指纹的运行在算指纹之前调它，一票否决。三种拒法：

    1. 传进来的东西根本不是``MotionSource``——它连"我可不可重放"都答不上；
    2. ``is_replayable()``返回的不是真正的``bool``——返回``1``或``"yes"``是在
       **回避**这个问题，按失败关闭处理（与轴2规则5"留空装有"同一种病）；
    3. ``is_replayable()``返回``False``——那就是本条要挡的那一类。

    **这道门只会拒，不会证。** 它挡不住一个谎称``True``的来源；今天唯一的证伪手段是
    ``AnalyticPose``构造期那次双求值，而它也只是必要条件。写在这里，
    比让读者以为过了门就等于指纹可信诚实。
    """

    if not isinstance(run_label, str) or not run_label.strip():
        raise MotionError("run_label must be a nonempty string")
    for index, source in enumerate(sources):
        name = getattr(source, "source_id", None) or f"sources[{index}]"
        if not isinstance(source, MotionSource):
            raise MotionError(
                f"{run_label}: {name} is not a MotionSource "
                "(pose_at/horizon_s/is_replayable are all required)"
            )
        verdict = source.is_replayable()
        if type(verdict) is not bool:
            raise MotionError(
                f"{run_label}: {name}.is_replayable() returned {verdict!r}, not a bool — "
                "回避这个问题按不可重放处理"
            )
        if not verdict:
            raise MotionError(
                f"{run_label}: motion source {name} is not replayable, so this run "
                "must not claim a reproduction fingerprint "
                "(spec/10第二节 × 轴3规则5：不可重放来源的运行不得声称指纹)"
            )


__all__ = [
    "ACCEPTED_TRANSLATION_UNITS",
    "EXTRAPOLATIONS",
    "MILLIMETRES_PER_METRE",
    "PAUSE_HOLDS",
    "POSE_TRANSLATION_UNIT",
    "QUATERNION_NORM_ABS_TOL",
    "ROTATION_ARCS",
    "ROTATION_INTERPOLATIONS",
    "SMALL_ANGLE_ONE_MINUS_COS",
    "TRANSLATION_INTERPOLATIONS",
    "AnalyticPose",
    "InterpolationSemantics",
    "MotionError",
    "MotionSource",
    "PauseInterval",
    "Pose",
    "PoseSample",
    "SampledPoseTimeline",
    "assert_replayable_for_fingerprint",
    "interpolate_pose_fraction",
]
