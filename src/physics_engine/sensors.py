"""传感器声明层——spec/10第四节那条**本仓少数原创法条**的执行体。

规范原话：

    硬约束：传感器只能读物理上真测得到的量。带材每个节点的精确位置不是传感器——
    真机上没有这个量；把它塞进观测空间训练出的策略无法迁移。
    这条要写成门（首个"必须红"用例：声明一个读全场状态的传感器，校验必须拒）。

那句话写下来之后一直**连红例都没有**（plans/02第一节点名）。本模块补上。

## 本模块只做声明与校验，**不做任何读取**

spec/10的``Sensor``接口（``observation_space``/``read``/``delay_s``）**尚未冻结**——
按0003的纪律，没有实现的设计只能0.x起步，冻结条件是"至少一个消费方有通过其全部
门禁的实现"。所以本模块**故意停在声明层**：它回答"这个传感器声明得住吗"，
不回答"它读出来是多少"。``read()``不在这里，也不该在这里。

## 校验为什么能拒掉"读全场状态"

一个"读全场状态"的传感器只有三种写法，本层各堵一条：

1. **把状态布局的字段名当观测量**——带材每个节点的精确位置在布局里就叫
   ``node17_x_mm``。规则R8：通道的量名不得是所校验布局里的字段名。
2. **声明一个维数等于自由度数的大通道**——规则R9：单个传感器的总观测维数必须
   **严格小于**布局的``dof_count``。真机上没有哪台仪器一次给出全部自由度。
3. **编一台"能测全场"的仪器**——规则R5要求通道的量必须被它绑定的仪器的
   ``measures``覆盖，规则R6要求仪器种类在``REALIZABLE_INSTRUMENT_KINDS``内。
   登记表是白名单、失败关闭，加一种仪器要改代码、要过评审。

**如实登记这一层的边界**：R9是**必要条件不是充分条件**——声明维数
``dof_count − 1``的传感器照样过得去。本层能拒掉的是**结构上不可能**的声明，
它证明不了一个声明真机可实现。那件事要等真机标定数据，不是校验器能给的。
把这句话写在这里，比让读者以为过了校验就等于能上真机诚实。

## 面（轴1规则1）

本模块**不落盘、不跨边界**，因此**不需要新的面**。传感器声明哪天要写进场景文件
或run package的manifest，那时才需要一个``physics_sensor_declaration``面，
且要先去``engine_facets.py``登记再落盘——那个文件是闸门，加面走一次闸门提交。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.identity import has_unit_suffix, parse_namespace_id
from physics_engine.state import StateLayout


class SensorError(ValueError):
    """传感器声明层的一切失败关闭。"""


#: 真机上装得出来的仪器种类（白名单，**失败关闭**；轴2的"只增不改"同样适用）。
#: 这份表的作用不是穷举世上所有仪器，是**逼声明者说出真机上那台东西叫什么**——
#: 一个说不出仪器种类的"传感器"，多半是仿真里的一个读数而不是真机上的一个器件。
#: 加一种：改这里 + 补一条测试；不许在调用方就地放行。
REALIZABLE_INSTRUMENT_KINDS: frozenset[str] = frozenset(
    {
        "encoder",  # 增量/绝对编码器：角位移、线位移
        "load_cell",  # 力/张力传感器
        "torque_sensor",  # 扭矩传感器
        "laser_displacement",  # 激光位移/测距
        "linear_potentiometer",  # 拉线位移计
        "imu",  # 惯性测量单元：加速度与角速度
        "camera",  # 视觉：像素或经标定的特征位置
        "thermocouple",  # 温度
        "limit_switch",  # 限位开关（离散）
        "pressure_sensor",  # 压力
        "photodiode",  # 光强（光学域的第一个候选）
    }
)


def _require_namespace(value: object, prefix: str, what: str) -> str:
    if not isinstance(value, str):
        raise SensorError(f"{what} must be a string: {value!r}")
    if not value.startswith(f"{prefix}/"):
        raise SensorError(f"{what} must be namespaced like {prefix!r}/…: {value!r}")
    try:
        parse_namespace_id(value)
    except ValueError as error:  # IdentityError继承自ValueError
        raise SensorError(f"{what} is not a valid namespace id: {error}") from error
    return value


def _require_quantity(value: object, what: str) -> str:
    """物理量ID必须带单位后缀（轴2规则3）。**裸名字不受理**。"""

    if not isinstance(value, str) or not value:
        raise SensorError(f"{what} must be a nonempty string: {value!r}")
    if not has_unit_suffix(value):
        raise SensorError(
            f"{what} must carry a unit suffix (axis 2 rule 3): {value!r} — "
            "观测量没有单位就没法与真机标定对上，也没法跨域复用"
        )
    return value


@dataclass(frozen=True)
class Instrument:
    """真机上的一台仪器。**只登记它的身份与能测什么**，不含任何读取实现。"""

    instrument_id: str
    #: 必须在``REALIZABLE_INSTRUMENT_KINDS``内——失败关闭。
    kind: str
    #: 这台仪器**真测得到**的量（带单位后缀）。它是通道量的白名单。
    measures: tuple[str, ...]
    #: 采样周期（秒），必须为正。
    sample_period_s: float
    #: 观测时延（秒）。**必须显式给出**，零时延也要写``0.0``——
    #: 真实回路有延迟，没有时延训练出的策略上真机会翻车（spec/10第三节对
    #: ``Actuator``写的那条，对``Sensor``同样成立）。
    delay_s: float

    def __post_init__(self) -> None:
        _require_namespace(self.instrument_id, "instrument", "instrument_id")
        if self.kind not in REALIZABLE_INSTRUMENT_KINDS:
            raise SensorError(
                f"unknown instrument kind {self.kind!r} — "
                f"可实现仪器登记表里没有它。已登记：{sorted(REALIZABLE_INSTRUMENT_KINDS)}。"
                "加一种要改sensors.py并补一条测试，不许在调用方就地放行"
            )
        if not self.measures:
            raise SensorError(
                f"{self.instrument_id}: an instrument must declare what it measures"
            )
        for quantity in self.measures:
            _require_quantity(quantity, f"{self.instrument_id}: measured quantity")
        if len(set(self.measures)) != len(self.measures):
            raise SensorError(f"{self.instrument_id}: duplicate entry in measures")
        if not (
            isinstance(self.sample_period_s, (int, float))
            and not isinstance(self.sample_period_s, bool)
            and self.sample_period_s > 0.0
            and math.isfinite(self.sample_period_s)
        ):
            raise SensorError(
                f"{self.instrument_id}: sample_period_s must be positive and finite, "
                f"got {self.sample_period_s!r}"
            )
        if not (
            isinstance(self.delay_s, (int, float))
            and not isinstance(self.delay_s, bool)
            and self.delay_s >= 0.0
            and math.isfinite(self.delay_s)
        ):
            raise SensorError(
                f"{self.instrument_id}: delay_s must be a finite value >= 0 — "
                "零时延要显式写0.0；缺省一个时延等于假装真机没有延迟"
            )


@dataclass(frozen=True)
class ObservationChannel:
    """观测空间的一路。维数是**声明期常数**，不随场景规模变化。"""

    channel_id: str
    quantity_id: str
    #: 这一路给出几个标量。真机上它由硬件定死，**不是网格分辨率的函数**。
    dimension: int
    #: 由哪台已声明的仪器给出。
    instrument_id: str

    def __post_init__(self) -> None:
        _require_namespace(self.channel_id, "channel", "channel_id")
        _require_namespace(self.instrument_id, "instrument", "instrument_id")
        _require_quantity(self.quantity_id, f"{self.channel_id}: quantity_id")
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 1
        ):
            raise SensorError(
                f"{self.channel_id}: dimension must be a positive integer, "
                f"got {self.dimension!r}"
            )


@dataclass(frozen=True)
class SensorDeclaration:
    """一个传感器的声明。**装得住才叫传感器**——校验不过就不存在这个传感器。"""

    sensor_id: str
    instruments: tuple[Instrument, ...]
    channels: tuple[ObservationChannel, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.sensor_id, "sensor", "sensor_id")
        if not self.instruments:
            raise SensorError(f"{self.sensor_id}: a sensor needs at least one instrument")
        if not self.channels:
            raise SensorError(f"{self.sensor_id}: a sensor needs at least one channel")
        by_id: dict[str, Instrument] = {}
        for instrument in self.instruments:
            if instrument.instrument_id in by_id:
                raise SensorError(
                    f"{self.sensor_id}: duplicate instrument {instrument.instrument_id!r}"
                )
            by_id[instrument.instrument_id] = instrument
        seen: set[str] = set()
        for channel in self.channels:
            if channel.channel_id in seen:
                raise SensorError(
                    f"{self.sensor_id}: duplicate channel {channel.channel_id!r}"
                )
            seen.add(channel.channel_id)
            instrument = by_id.get(channel.instrument_id)
            if instrument is None:
                raise SensorError(
                    f"{self.sensor_id}: channel {channel.channel_id!r} is bound to "
                    f"undeclared instrument {channel.instrument_id!r} — "
                    "真机上没有那台东西，这一路观测就不存在"
                )
            if channel.quantity_id not in instrument.measures:
                raise SensorError(
                    f"{self.sensor_id}: channel {channel.channel_id!r} wants "
                    f"{channel.quantity_id!r} but instrument "
                    f"{instrument.instrument_id!r} ({instrument.kind}) only measures "
                    f"{sorted(instrument.measures)} — 仪器测不到的量不是观测量"
                )

    @property
    def observation_dimension(self) -> int:
        """观测空间的标量总数。**声明期常数**，不随状态布局变化。"""

        return sum(channel.dimension for channel in self.channels)

    @property
    def delay_s(self) -> float:
        """整个传感器的观测时延=各通道所用仪器时延的最大值。

        取最大不是取平均：一次完整观测要等最慢的那一路到齐。
        """

        by_id = {item.instrument_id: item for item in self.instruments}
        return max(by_id[channel.instrument_id].delay_s for channel in self.channels)

    def assert_realizable_on(self, layout: StateLayout) -> None:
        """把这个传感器放到一个具体状态布局上校验——**读全场状态的当场拒**。

        与``__post_init__``里那些规则的分工：那些只看声明自己（仪器在不在、
        量测不测得到、单位有没有），本方法看的是**声明与被观测系统的关系**，
        因此必须拿着布局才判得了。
        """

        if not isinstance(layout, StateLayout):
            raise SensorError(f"{self.sensor_id}: expected a StateLayout, got {layout!r}")
        field_names = {field.name for field in layout.fields}

        # R8：状态布局的字段名不是观测量。
        # **这就是spec/10那句"带材每个节点的精确位置不是传感器"的执行体**——
        # 那个量在布局里就叫`node17_x_mm`，真机上没有任何仪器给得出它。
        for channel in self.channels:
            if channel.quantity_id in field_names:
                raise SensorError(
                    f"{self.sensor_id}: channel {channel.channel_id!r} observes "
                    f"{channel.quantity_id!r}, which is a state degree of freedom of "
                    f"layout {layout.layout_id!r} — 传感器只能读物理上真测得到的量"
                    "（spec/10第四节）。真机上没有这个量；把它塞进观测空间训练出的"
                    "策略无法迁移"
                )

        # R9：总观测维数必须严格小于自由度数。
        # 维数达到自由度数就是全场读取——真机上没有哪台仪器一次给出全部自由度。
        # **这是必要条件不是充分条件**，边界写在模块文档里。
        dimension = self.observation_dimension
        if dimension >= layout.dof_count:
            raise SensorError(
                f"{self.sensor_id}: observation dimension {dimension} reaches the "
                f"{layout.dof_count} degrees of freedom of layout {layout.layout_id!r} — "
                "这是在读全场状态。真机上没有这样的仪器（spec/10第四节）；"
                "观测维数由硬件定死，不是网格分辨率的函数"
            )


__all__ = [
    "REALIZABLE_INSTRUMENT_KINDS",
    "Instrument",
    "ObservationChannel",
    "SensorDeclaration",
    "SensorError",
]
