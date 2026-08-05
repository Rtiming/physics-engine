"""驱动器声明层与时延机械——spec/10第三节的执行体。

规范原话：

    **把控制量变成物理作用。** 张力机放线端是第一个实现候选（WDS用途4）：

        command_space() -> Space
        apply(command, state, dt_s) -> ActuationResult
        delay_s() -> float

    时延不是可选项——真实控制回路有延迟，没有时延训练出的策略上真机会翻车
    （MuJoCo环形缓冲形制）。

## 本模块只做声明、校验与时延机械，**不做``apply``的物理**

``apply``要把一条命令变成力/力矩，那要接能量项与求解器，是另一块。所以本模块
与``sensors.py``同纪律：**接口未冻结时不替它拍板**。有一条门断言
``apply``与``command_space``在本模块的任何公开类上都不存在。

``command_space``缺席的理由与``sensors.py``缺``observation_space``是同一条：
返回值类型``Space``在spec/10里只有名字没有定义，在这里发明一个就等于替冻结拍板。
本层给的是``CommandChannel``与声明期常数``command_dimension``——
真机上执行器有几路命令、每路几个标量、各自的行程/力矩上限是多少，都是硬件定死的。

``ActuationResult``**只带"这条命令被延迟成什么、什么时候生效"，不带力**。

## 时延为什么是三条硬规则而不是一个参数

1. **``delay_s``必须显式给出，零也必须是显式的零并附理由。** 缺省一个时延等于
   假装真实回路没有延迟，而那正是规范点名的翻车路径。所以``delay_s``没有默认值,
   且``delay_s == 0``时必须同时给``zero_delay_rationale``（非零时给理由反而被拒——
   那个字段只在声明为零时有意义，留着会变成一个没人读的死字段）。
2. **时延用环形缓冲实现，而缓冲只表达得了``dt_s``的整数倍。** 于是
   ``delay_s``是**声明**，``realized_delay_s = steps × dt_s``是**实际**,
   两者之差``quantization_residual_s``必须被算出来并如实带在对象上。
3. **``delay_s``不是``dt_s``整数倍时，默认失败关闭。** 见下节。

## 非整数倍的语义：默认拒，唯一的放行方式是显式声明取整方向

四舍五入是被明令禁止的（它把时延悄悄改掉，且改的方向不确定）。本模块给两个取值：

* ``quantization="exact"``（默认口径）：要求``delay_s``是``dt_s``的整数倍，
  容差``DELAY_STEP_MATCH_ULPS``个ULP——只吃浮点表示误差，不吃"差不多"。
  不满足就当场炸，并在消息里把残差和两条出路写出来。
* ``quantization="ceil_to_step"``：向**上**取整到下一个整步，并把
  ``quantization_residual_s > 0``带出来。

**为什么只有向上、没有向下也没有就近**：欠时延是规范点名的危险方向
（"没有时延训练出的策略上真机会翻车"），``floor``与``round``都可能欠时延,
而``ceil``最多多延不到一步。多延是保守方向，少延是事故方向,
两者不对称，所以白名单里只有保守的那一个。

**为什么不干脆默认``ceil``**：因为那样时延还是被悄悄改了，只是改的方向好看一点。
默认拒逼声明者做一次选择：要么把``delay_s``或``dt_s``挑成能整除的，
要么明写"我接受向上取整"。**取整方向是一条声明，不是一个实现细节。**

## 在途命令是真历史，所以时延线是值不是可变对象

spec/12规则1："状态是显式数组，内核是无状态纯函数……一切随时间变的量进显式状态"；
第2.2节又把"真历史"与"求解器便利"分开，前者必须进状态并随状态被复现。
**缓冲里那几条在途命令是真历史**：丢了它们，同一个初始状态重跑会得到不同的轨迹。
所以``ActuationDelayLine``是**冻结的值**，``advance()``返回``(新的线, 结果)``而不是
就地改自己。一个藏在对象里的可变环形缓冲会让"这次运行依据哪些在途命令"
变成谁也说不清的事。

## 面（轴1规则1）

本模块**不落盘、不跨边界**，因此**不需要新的面**。在途命令哪天要随状态进
run package，那时才需要一个``physics_actuation_delay_line``面，
且要先去``engine_facets.py``登记再落盘——那个文件是闸门。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from physics_engine.identity import has_unit_suffix, parse_namespace_id


class ActuatorError(ValueError):
    """驱动器声明层的一切失败关闭。"""


#: 真机上装得出来的驱动器种类（白名单，**失败关闭**；与``sensors.py``的仪器表同纪律）。
#: 它逼声明者说出真机上那台东西叫什么——一个说不出种类的"驱动器"多半是仿真里
#: 直接往方程右端加的一项，而那正是spec/10第三节说的"把答案先塞进去"。
#: 前两项就是WDS张力机放线端的真实构成（其``end_driver.py``：
#: "带材的张力最终来自放线盘上的伺服电机，经磁粉离合器传到卷筒"）。
REALIZABLE_ACTUATOR_KINDS: frozenset[str] = frozenset(
    {
        "servo_motor",  # 伺服电机：转速/力矩/位置
        "magnetic_particle_clutch",  # 磁粉离合器：可控滑差力矩
        "stepper_motor",  # 步进电机
        "tension_brake",  # 张力制动器
        "linear_stage",  # 直线运动平台
        "pneumatic_cylinder",  # 气缸
        "hydraulic_cylinder",  # 液压缸
        "piezo_stack",  # 压电叠堆（微位移）
        "solenoid",  # 电磁铁（离散通断）
        "heater",  # 加热器（热域的第一个候选）
    }
)

#: 时延量化方式。**没有``round``也没有``floor``**——理由见模块文档。
DELAY_QUANTIZATIONS: frozenset[str] = frozenset({"exact", "ceil_to_step"})

#: "``delay_s``是``dt_s``的整数倍"这一判据的容差，单位是ULP。**这个数是算出来的**：
#: 对1e-5s到0.1s的常用步长与1..10000步的组合逐一实测（决策0038第四节），
#: **真整数倍对的最大ULP距离是1.000**；而把``delay_s``相对偏移1e-13（远小于任何
#: 物理意义）之后，**最小ULP距离已是450**。取4给真整数倍留4倍余量,
#: 同时比最近的假整数倍小两个数量级——两侧都不是紧的。
DELAY_STEP_MATCH_ULPS = 4

#: 环形缓冲深度上限。它同时是一道**单位门**：一个把毫秒当秒写的``delay_s``
#: （比如4而不是0.004）在这里被抓住。定量：``dt_s=1e-4``（10kHz）时10万步等于10秒时延,
#: 而没有哪个真实控制回路有10秒的驱动时延；要突破它，先解释那是什么回路。
MAX_DELAY_STEPS = 100_000


# ---------------------------------------------------------------- 校验原语 ---
# 与``motion.py``同理由：**有名字的模块级函数**，测试可以把某一条换成空操作
#（等价于把它写成``if False``），从而证明红是那一条红的。


def _require_namespace(value: object, prefix: str, what: str) -> str:
    if not isinstance(value, str):
        raise ActuatorError(f"{what} must be a string: {value!r}")
    if not value.startswith(f"{prefix}/"):
        raise ActuatorError(f"{what} must be namespaced like {prefix!r}/…: {value!r}")
    try:
        parse_namespace_id(value)
    except ValueError as error:  # IdentityError继承自ValueError
        raise ActuatorError(f"{what} is not a valid namespace id: {error}") from error
    return value


def _require_finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActuatorError(f"{what} must be a real number: {value!r}")
    if not math.isfinite(value):
        raise ActuatorError(f"{what} must be finite: {value!r}")
    return float(value)


def _require_quantity(value: object, what: str) -> str:
    """命令量必须带单位后缀（轴2规则3）。**裸名字不受理**。"""

    if not isinstance(value, str) or not value:
        raise ActuatorError(f"{what} must be a nonempty string: {value!r}")
    if not has_unit_suffix(value):
        raise ActuatorError(
            f"{what} must carry a unit suffix (axis 2 rule 3): {value!r} — "
            "命令量没有单位就没法与真机的控制器对上"
        )
    return value


def _require_explicit_delay(delay_s: object, rationale: object, actuator_id: str) -> float:
    """时延不是可选项：零必须是**显式的**零并附理由。"""

    value = _require_finite(delay_s, f"{actuator_id}: delay_s")
    if value < 0.0:
        raise ActuatorError(
            f"{actuator_id}: delay_s must be >= 0, got {value!r} — 命令不会在提交之前生效"
        )
    if value == 0.0:
        if not isinstance(rationale, str) or not rationale.strip():
            raise ActuatorError(
                f"{actuator_id}: delay_s=0 needs an explicit zero_delay_rationale — "
                "零时延是一条关于真机的强声明（spec/10第三节：没有时延训练出的策略"
                "上真机会翻车），它必须是被写下来的选择，不是一个缺省值"
            )
    elif rationale is not None:
        raise ActuatorError(
            f"{actuator_id}: zero_delay_rationale is only meaningful when delay_s=0, "
            f"but delay_s={value!r} — 留一个没人读的字段与留空装有是同一种病"
        )
    return value


def _steps_match_delay(delay_s: float, dt_s: float, steps: int) -> bool:
    """``steps × dt_s``是否就是``delay_s``（只容许浮点表示误差）。"""

    realized = steps * dt_s
    scale = max(abs(delay_s), abs(realized))
    if scale == 0.0:
        return delay_s == 0.0
    return abs(delay_s - realized) <= DELAY_STEP_MATCH_ULPS * math.ulp(scale)


def _require_integer_step_delay(
    delay_s: float, dt_s: float, steps: int, actuator_id: str
) -> None:
    if not _steps_match_delay(delay_s, dt_s, steps):
        raise ActuatorError(
            f"{actuator_id}: delay_s={delay_s!r} is not an integer multiple of "
            f"dt_s={dt_s!r} (closest is {steps} steps = {steps * dt_s!r}, residual "
            f"{delay_s - steps * dt_s!r}s) — 环形缓冲只表达得了整步时延，"
            "而四舍五入会把时延悄悄改掉。两条出路：把delay_s或dt_s挑成能整除的，"
            "或显式声明quantization='ceil_to_step'接受向上取整"
        )


def _delay_steps(
    delay_s: float, dt_s: float, quantization: str, actuator_id: str
) -> tuple[int, float]:
    """``(缓冲深度, 量化残差)``。残差``= steps × dt_s − delay_s``，恒为非负（保守方向）。"""

    ratio = delay_s / dt_s
    if ratio > MAX_DELAY_STEPS:
        raise ActuatorError(
            f"{actuator_id}: delay_s={delay_s!r} at dt_s={dt_s!r} needs {ratio:.3g} "
            f"buffer steps, past MAX_DELAY_STEPS={MAX_DELAY_STEPS} — "
            "先检查单位：把毫秒当秒写会正好落在这里"
        )
    if quantization == "exact":
        steps = round(ratio)
        _require_integer_step_delay(delay_s, dt_s, steps, actuator_id)
    else:  # ceil_to_step
        steps = math.ceil(ratio)
        # ``ceil``在"其实已经是整数倍、只差几个ULP"时会多给一步，收回来。
        if steps > 0 and _steps_match_delay(delay_s, dt_s, steps - 1):
            steps -= 1
    return steps, steps * dt_s - delay_s


# ---------------------------------------------------------------- 命令空间 ---


@dataclass(frozen=True)
class CommandChannel:
    """命令空间的一路。维数与上下界都是**声明期常数**，由硬件定死。

    界限不是装饰：一个能下达无穷大力矩的驱动器，与一个能读全场状态的传感器
    （spec/10第四节）是同一种谎——真机上那台东西有行程、有力矩上限、有电流上限。
    """

    channel_id: str
    quantity_id: str
    dimension: int
    #: 每个标量的下界与上界（长度必须等于``dimension``）。
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.channel_id, "command", "channel_id")
        _require_quantity(self.quantity_id, f"{self.channel_id}: quantity_id")
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 1
        ):
            raise ActuatorError(
                f"{self.channel_id}: dimension must be a positive integer, "
                f"got {self.dimension!r}"
            )
        for name, bounds in (("lower", self.lower), ("upper", self.upper)):
            if not isinstance(bounds, tuple) or len(bounds) != self.dimension:
                raise ActuatorError(
                    f"{self.channel_id}: {name} must be a {self.dimension}-tuple, "
                    f"got {bounds!r}"
                )
            for index, value in enumerate(bounds):
                _require_finite(value, f"{self.channel_id}: {name}[{index}]")
        for index, (low, high) in enumerate(zip(self.lower, self.upper, strict=True)):
            if not low < high:
                raise ActuatorError(
                    f"{self.channel_id}: bound {index} needs lower < upper, "
                    f"got {low!r} >= {high!r}"
                )


@dataclass(frozen=True)
class ActuatorDeclaration:
    """一个驱动器的声明。**装得住才叫驱动器**——校验不过就不存在这个驱动器。

    ``delay_s``在spec/10里写成方法，本层给的是数据字段（与``sensors.py``把
    ``delay_s``给成属性同一口径）：方法形态等接口冻结时随实现一起定。
    """

    actuator_id: str
    kind: str
    channels: tuple[CommandChannel, ...]
    #: 命令时延（秒）。**没有默认值**；为0时必须同时给``zero_delay_rationale``。
    delay_s: float
    #: 仅在``delay_s == 0``时可给且必须给；非零时给它会被拒。
    zero_delay_rationale: str | None

    def __post_init__(self) -> None:
        _require_namespace(self.actuator_id, "actuator", "actuator_id")
        if self.kind not in REALIZABLE_ACTUATOR_KINDS:
            raise ActuatorError(
                f"unknown actuator kind {self.kind!r} — "
                f"可实现驱动器登记表里没有它。已登记：{sorted(REALIZABLE_ACTUATOR_KINDS)}。"
                "加一种要改actuators.py并补一条测试，不许在调用方就地放行"
            )
        if not self.channels:
            raise ActuatorError(
                f"{self.actuator_id}: an actuator needs at least one command channel"
            )
        seen: set[str] = set()
        for channel in self.channels:
            if not isinstance(channel, CommandChannel):
                raise ActuatorError(
                    f"{self.actuator_id}: not a CommandChannel: {channel!r}"
                )
            if channel.channel_id in seen:
                raise ActuatorError(
                    f"{self.actuator_id}: duplicate channel {channel.channel_id!r}"
                )
            seen.add(channel.channel_id)
        _require_explicit_delay(self.delay_s, self.zero_delay_rationale, self.actuator_id)

    @property
    def command_dimension(self) -> int:
        """命令空间的标量总数。**声明期常数**，不随场景规模变化。"""

        return sum(channel.dimension for channel in self.channels)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """按通道次序摊平的``(下界, 上界)``。次序即命令向量的次序。"""

        return tuple(
            (low, high)
            for channel in self.channels
            for low, high in zip(channel.lower, channel.upper, strict=True)
        )

    def assert_command_admissible(self, command: ActuationCommand) -> None:
        """一条命令能不能下达给这个驱动器。越界当场炸，不夹取。

        **不夹取**是有意的：悄悄把越界命令夹到边界，会让一个饱和的控制器
        看起来一直在正常工作，而真机上那是执行器打满——两回事。
        """

        if not isinstance(command, ActuationCommand):
            raise ActuatorError(
                f"{self.actuator_id}: expected an ActuationCommand, got {command!r}"
            )
        if command.actuator_id != self.actuator_id:
            raise ActuatorError(
                f"{self.actuator_id}: command is addressed to "
                f"{command.actuator_id!r}"
            )
        if len(command.values) != self.command_dimension:
            raise ActuatorError(
                f"{self.actuator_id}: command has {len(command.values)} scalars but the "
                f"command space declares {self.command_dimension}"
            )
        for index, (value, (low, high)) in enumerate(
            zip(command.values, self.bounds, strict=True)
        ):
            if value < low or value > high:
                raise ActuatorError(
                    f"{self.actuator_id}: command scalar {index} is {value!r}, "
                    f"outside the declared range [{low!r}, {high!r}] — "
                    "真机上那台东西到不了这个值；本层不替你夹取"
                )


@dataclass(frozen=True)
class ActuationCommand:
    """一条命令：按声明的通道次序摊平的标量。**不含时间**——时间由时延线记账。"""

    actuator_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.actuator_id, "actuator", "actuator_id")
        if not isinstance(self.values, tuple) or not self.values:
            raise ActuatorError(
                f"{self.actuator_id}: values must be a nonempty tuple, got {self.values!r}"
            )
        for index, value in enumerate(self.values):
            _require_finite(value, f"{self.actuator_id}: values[{index}]")


@dataclass(frozen=True)
class ActuationResult:
    """一步时延推进的结果。**只说这条命令被延迟成什么、什么时候生效，不带力。**

    ``apply``那半（把命令变成力/力矩）要接能量项与求解器，是另一块；
    在spec/10冻结之前，这里不替它拍板。
    """

    actuator_id: str
    #: 本步**生效**的那条命令（多半是若干步之前提交的那条）。
    command: ActuationCommand
    #: 它是什么时候被提交的。填充期为负——那些命令确实早于本次运行。
    issued_at_s: float
    #: 它什么时候生效（就是本步）。
    effective_at_s: float
    #: 实际时延``= steps × dt_s``。**不是**``effective_at_s − issued_at_s``:
    #: 那个减法在浮点下不恒等于前者，且**随步号变**（实测：``dt_s=0.1``、``steps=7``,
    #: 第2步的减法给0.7，而``7 × 0.1``给0.7000000000000001）。
    #: 契约取与步号无关的那一个——一个随步号抖动的"实际时延"不是时延。
    realized_delay_s: float
    #: 这条命令是不是缓冲的初始填充（即：不是本次运行提交的）。
    from_initial_fill: bool


@dataclass(frozen=True)
class ActuationDelayLine:
    """MuJoCo形制的环形缓冲，写成**冻结的值**。

    直接构造要自己把导出量算对（``__post_init__``会逐条复核）；正常入口是
    ``declare()``。推进用``advance()``，它返回``(新的线, 结果)``——
    在途命令是真历史，不许藏成可变状态（见模块文档）。
    """

    declaration: ActuatorDeclaration
    dt_s: float
    quantization: str
    steps: int
    realized_delay_s: float
    quantization_residual_s: float
    #: 在途命令，长度恒等于``steps``。**这是真历史**（spec/12第2.2节）。
    pending: tuple[ActuationCommand, ...]
    step_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, ActuatorDeclaration):
            raise ActuatorError("declaration must be an ActuatorDeclaration")
        dt_s = _require_finite(self.dt_s, f"{self.actuator_id}: dt_s")
        if dt_s <= 0.0:
            raise ActuatorError(f"{self.actuator_id}: dt_s must be positive, got {dt_s!r}")
        if self.quantization not in DELAY_QUANTIZATIONS:
            raise ActuatorError(
                f"{self.actuator_id}: quantization must be one of "
                f"{sorted(DELAY_QUANTIZATIONS)}, got {self.quantization!r} — "
                "没有'round'也没有'floor'：欠时延是规范点名的危险方向"
            )
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ActuatorError(f"{self.actuator_id}: steps must be >= 0, got {self.steps!r}")
        if self.steps > MAX_DELAY_STEPS:
            raise ActuatorError(
                f"{self.actuator_id}: steps={self.steps} past MAX_DELAY_STEPS="
                f"{MAX_DELAY_STEPS}"
            )
        if self.realized_delay_s != self.steps * dt_s:
            raise ActuatorError(
                f"{self.actuator_id}: realized_delay_s must be exactly steps × dt_s "
                f"({self.steps} × {dt_s!r} = {self.steps * dt_s!r}), "
                f"got {self.realized_delay_s!r}"
            )
        if self.realized_delay_s < self.declaration.delay_s and not _steps_match_delay(
            self.declaration.delay_s, dt_s, self.steps
        ):
            raise ActuatorError(
                f"{self.actuator_id}: realized_delay_s={self.realized_delay_s!r} is "
                f"shorter than the declared delay_s={self.declaration.delay_s!r} — "
                "欠时延是spec/10第三节点名的危险方向，本层只许多延不许少延"
            )
        if len(self.pending) != self.steps:
            raise ActuatorError(
                f"{self.actuator_id}: pending holds {len(self.pending)} commands but the "
                f"buffer is {self.steps} steps deep"
            )
        for command in self.pending:
            self.declaration.assert_command_admissible(command)
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ActuatorError(
                f"{self.actuator_id}: step_index must be >= 0, got {self.step_index!r}"
            )

    @property
    def actuator_id(self) -> str:
        return self.declaration.actuator_id

    @classmethod
    def declare(
        cls,
        declaration: ActuatorDeclaration,
        *,
        dt_s: float,
        quantization: str,
        initial_command: ActuationCommand,
    ) -> ActuationDelayLine:
        """按``dt_s``把声明的时延量化成缓冲深度。

        ``initial_command``**必须显式给出**：缓冲在第0步之前就已经装满了东西,
        那是"本次运行开始之前这台执行器在做什么"。默认填零等于替声明者说
        "它当时闲着"，而那是一条关于真机的断言。
        """

        if not isinstance(declaration, ActuatorDeclaration):
            raise ActuatorError("declaration must be an ActuatorDeclaration")
        actuator_id = declaration.actuator_id
        dt_value = _require_finite(dt_s, f"{actuator_id}: dt_s")
        if dt_value <= 0.0:
            raise ActuatorError(f"{actuator_id}: dt_s must be positive, got {dt_value!r}")
        if quantization not in DELAY_QUANTIZATIONS:
            raise ActuatorError(
                f"{actuator_id}: quantization must be one of "
                f"{sorted(DELAY_QUANTIZATIONS)}, got {quantization!r} — "
                "没有'round'也没有'floor'：欠时延是规范点名的危险方向"
            )
        declaration.assert_command_admissible(initial_command)
        steps, residual = _delay_steps(
            declaration.delay_s, dt_value, quantization, actuator_id
        )
        return cls(
            declaration=declaration,
            dt_s=dt_value,
            quantization=quantization,
            steps=steps,
            realized_delay_s=steps * dt_value,
            quantization_residual_s=residual,
            pending=(initial_command,) * steps,
            step_index=0,
        )

    def advance(
        self, command: ActuationCommand
    ) -> tuple[ActuationDelayLine, ActuationResult]:
        """提交一条命令，推进一步，返回``(新的线, 本步生效的结果)``。

        时刻由步号乘``dt_s``算，不是逐步累加——累加会漂，乘法不会,
        且同一步号在任何一次重跑里给同一个浮点数。
        """

        self.declaration.assert_command_admissible(command)
        if self.steps == 0:
            emitted = command
            pending: tuple[ActuationCommand, ...] = ()
            from_initial_fill = False
        else:
            emitted = self.pending[0]
            pending = (*self.pending[1:], command)
            from_initial_fill = self.step_index < self.steps
        result = ActuationResult(
            actuator_id=self.actuator_id,
            command=emitted,
            issued_at_s=(self.step_index - self.steps) * self.dt_s,
            effective_at_s=self.step_index * self.dt_s,
            realized_delay_s=self.realized_delay_s,
            from_initial_fill=from_initial_fill,
        )
        return replace(self, pending=pending, step_index=self.step_index + 1), result


__all__ = [
    "DELAY_QUANTIZATIONS",
    "DELAY_STEP_MATCH_ULPS",
    "MAX_DELAY_STEPS",
    "REALIZABLE_ACTUATOR_KINDS",
    "ActuationCommand",
    "ActuationDelayLine",
    "ActuationResult",
    "ActuatorDeclaration",
    "ActuatorError",
    "CommandChannel",
]
