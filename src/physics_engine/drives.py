"""张力驱动链——磁粉离合器、卷径换算与理想PID（力学域，决策0062轨道乙）。

守能力位S6.3（执行器的物理：张力轮的控制量变成力）。

## 为什么不长在``actuators.py``里

0038登记的欠账原文是"``actuators``的``apply``物理——落地那天它会import ``state``，
域登记要同批改"。**本模块把那笔账兑现了，但没有按那个形状兑现。**

理由是spec/15的环序：``actuators``登记在**基座**的``scene``圈，而基座
**不依赖任何物理域**（spec/01"内核不依赖任何上层"，由域隔离门守着）。
把``apply``的物理塞进``actuators``，等于让基座反向依赖力学——
那正是0035那次抓到的"方向反了"的错，而当时的判词是
**"我最初把它登记成基座接口，是把愿望当成了事实"**。

所以：``actuators``继续只做声明与时延机械（它那套门一条没动、逐字节不变），
物理长在本模块，登记在**力学域**。本模块单向import``actuators``（基座→力学是向下的）。

**代价如实写**：spec/10的``Actuator.apply``这个方法名今天仍然不存在。
如果将来某个消费方按接口名调用它，要么在本模块外套一层适配，
要么重新裁``actuators``的环归属——**那是一次决策，不是一次重构**。

## 真机链路（`winding-machine/HARDWARE_TOPOLOGY.md`，2026-06-08现场确认）

    LTS1-5悬臂张力传感器(5kg, 0—20mV)
      → ATC600张力控制器（**内部PID闭环**）
      → BOSENSE TBA420功放（0—10V入 / 0—24V·4A出）
      → POC-050L对轴型磁粉离合器
      → 卷材张力对象

本模块建的是**从电流到张力**那一段的物理，加一个**理想**PID。
**不复现ATC600**：官方没有公开完整的通信/参数/寄存器手册
（`ED3L_张力控制资料包`2026-06-03已核），复现一个拿不到参数的黑箱回路，
写出来的PID系数是编的。0062第二节裁决3。

## 判据：I控制下闭环**精确**是二阶系统

设离合器一阶时滞``τ``、电流—扭矩增益``k_M``、卷径``R``，记``K = k_M/R``：

    离合器   dM/dt = (k_M·I − M) / τ
    换算     T = M / R
    纯积分   dI/dt = Ki·(T_set − T)

消去``I``与``M``：

    τ·T'' + T' + K·Ki·T = K·Ki·T_set

于是``ω_n = sqrt(K·Ki/τ)``、``ζ = 1/(2·sqrt(τ·K·Ki))``，超调

    超调量 = exp(−ζπ / sqrt(1 − ζ²))

## 它**不是**`contact.restitution_from_damping_ratio`——这一条是实测纠正的

写这一段时我判定两者"逐位相同"，理由是形状一样。**实测当场否掉**：
``ζ = 0.5``时超调式给``0.16303``、恢复系数给``0.29844``，差了1.8倍。

病根不在浮点，在本仓自己的一次裁决。两条式子确实共享同一个形状

    exp(−ζ · Φ(ζ) / sqrt(1 − ζ²))

**只差``Φ``**：阶跃响应的峰值出现在半个阻尼周期处，``Φ = π``；
而0052第一节裁定接触在**分离瞬间截断拉力**（判据＝合力归零），
那条约定下的接触时长给出``Φ = 2·acos(ζ)``。

两者**只在``ζ = 0``处重合**（``2·acos(0) = π``），此后越离越远。
若当初裁的是"不截断"，两条式子就真的相同了——
**"看起来该相同"在这里恰好被本仓的一条裁决打断，而那条裁决是有意的。**

本模块因此不import它，并有一条门判两者**不同且只在``ζ = 0``处相等**——
那条门红了说明有人把两件事合并了，或者0052那条约定被悄悄改了。

**为什么判据取纯积分而不取PI**：PI闭环的传递函数带一个零点``−Ki/Kp``，
标准二阶超调式对它不成立。纯积分没有零点，闭式是精确的。
``Kp``照样实现、照样能用，只是**判据挑一个闭式说得清的工况**。

## 离散化：对象精确，整环实测**二阶**

离合器用**零阶保持的精确离散**——命令在一步内恒定时

    M_{n+1} = M_cmd + (M_n − M_cmd)·exp(−dt/τ)

**这一步没有离散误差**。误差全部来自积分器的前向Euler。

**但整环的超调误差实测是二阶不是一阶**（2026-08-17，``ζ = 0.5``、
``dt``逐次减半：``1.350e-3 / 3.353e-4 / 8.719e-5 / 1.926e-5``，
比值``4.025 / 3.845 / 4.527``）。写下来是因为"前向Euler是一阶"这句话
会让人以为整环也是一阶，**而实测不是**——精确ZOH的对象与Euler的控制器
组合起来给了更高的阶。**阶是量出来的，不是从某一个部件的阶推出来的。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from physics_engine.actuators import ActuatorError

#: 张力必须为正才谈得上"卷材被拉住"。零张力时带材松了，那是另一个工况
#: （松卷），本模块**不假装**能算它——失败关闭。
MIN_TENSION_N = 0.0


class DriveError(ValueError):
    """张力驱动链的一切失败关闭。"""


@dataclass(frozen=True)
class MagneticParticleClutch:
    """磁粉离合器：电流→可控滑差扭矩，线性段＋饱和＋一阶磁滞时滞。

    ## 三个参数各自的出处与不确定度

    * ``torque_per_ampere_nmm``：POC-050基型样本给50 N·m / 2.15 A
      ⟹ ``23256 N·mm/A``。**L后缀的专用线圈参数厂家资料没有**
      （`ED3L_张力控制资料包`2026-06-08已核），所以这个数是**基型外推**；
    * ``rated_torque_nmm``：同上，50 N·m ⟹ ``50000 N·mm``；
    * ``lag_s``：磁粉离合器的磁滞响应时间，**假设输入**，没有实测。

    三条都进0062第二节裁决2那张"只有现场实测能补"的清单。

    ## 饱和是**截断不是压缩**

    ``|I| > rated/k`` 时扭矩停在额定值，而不是按比例缩到额定值。
    截断意味着**超过饱和点后增大电流不再增大扭矩**，那是磁粉离合器的真实行为；
    压缩会让整条曲线都变形，且在小电流处也不对。

    ## 一阶时滞用**零阶保持的精确离散**

    ``M_{n+1} = M_cmd + (M_n − M_cmd)·exp(−dt/τ)``。命令在一步内恒定时
    这一步**没有离散误差**——不是"足够小的dt下近似成立"，是恒等式。
    用前向Euler（``M + dt/τ·(M_cmd − M)``）会在``dt > 2τ``时发散，
    而那个发散是纯数值的、没有物理对应。
    """

    torque_per_ampere_nmm: float
    rated_torque_nmm: float
    lag_s: float

    def __post_init__(self) -> None:
        for name in ("torque_per_ampere_nmm", "rated_torque_nmm", "lag_s"):
            value = getattr(self, name)
            if not (value > 0.0 and math.isfinite(value)):
                raise DriveError(f"{name} must be positive and finite: {value!r}")

    @property
    def saturation_current_a(self) -> float:
        """扭矩达到额定值所需的电流。超过它再加电流不再增大扭矩。"""

        return self.rated_torque_nmm / self.torque_per_ampere_nmm

    def commanded_torque_nmm(self, current_a: float) -> float:
        """稳态扭矩（**时滞之外的那一半**）：线性段``k·I``，饱和后停在额定值。

        负电流给负扭矩并同样饱和——磁粉离合器的扭矩方向由滑差方向定，
        这里的符号约定是"命令的符号即扭矩的符号"，由调用方负责与滑差一致。
        """

        if not math.isfinite(current_a):
            raise DriveError(f"current must be finite: {current_a!r}")
        raw = self.torque_per_ampere_nmm * current_a
        return math.copysign(min(abs(raw), self.rated_torque_nmm), raw)

    def advance_torque_nmm(self, torque_nmm: float, current_a: float, dt_s: float) -> float:
        """推进一步的磁滞：零阶保持下的**精确**一阶响应。"""

        if not (dt_s > 0.0 and math.isfinite(dt_s)):
            raise DriveError(f"dt_s must be positive and finite: {dt_s!r}")
        target = self.commanded_torque_nmm(current_a)
        return target + (torque_nmm - target) * math.exp(-dt_s / self.lag_s)


@dataclass(frozen=True)
class SpoolTension:
    """扭矩→张力：``T = M / R(n)``，卷径随匝数生长。

    ``R(n) = barrel_radius_mm + turns·tape_thickness_mm``。

    ## 半径生长为什么在这里而不在几何层

    因为它改变的是**力**：同一个扭矩在卷满时给出的张力比空卷时小
    ``R0/(R0+n·t)``倍。这是张力控制在真机上最基本的一条非线性——
    ATC600的锥度张力功能就是为它存在的。放在几何层会让"半径变了"
    与"力变了"两件事分开发生，而它们是同一件事。

    ``modelgen.generate_spool``的层数生长是**几何**的（产出形状），
    与本类不重复：那里回答"这卷长什么样"，这里回答"这卷现在拉得多紧"。
    """

    barrel_radius_mm: float
    tape_thickness_mm: float

    def __post_init__(self) -> None:
        for name in ("barrel_radius_mm", "tape_thickness_mm"):
            value = getattr(self, name)
            if not (value > 0.0 and math.isfinite(value)):
                raise DriveError(f"{name} must be positive and finite: {value!r}")

    def radius_mm(self, turns: float) -> float:
        if not (turns >= 0.0 and math.isfinite(turns)):
            raise DriveError(f"turns must be finite and nonnegative: {turns!r}")
        return self.barrel_radius_mm + turns * self.tape_thickness_mm

    def tension_n(self, torque_nmm: float, turns: float = 0.0) -> float:
        """``T = M / R(n)``。单位：``N·mm / mm = N``。"""

        if not math.isfinite(torque_nmm):
            raise DriveError(f"torque must be finite: {torque_nmm!r}")
        return torque_nmm / self.radius_mm(turns)

    def torque_nmm(self, tension_n: float, turns: float = 0.0) -> float:
        """反向换算：要这个张力需要多大扭矩。标定与前馈要用。"""

        if not math.isfinite(tension_n):
            raise DriveError(f"tension must be finite: {tension_n!r}")
        return tension_n * self.radius_mm(turns)


@dataclass(frozen=True)
class PidController:
    """理想PID：``u = Kp·e + Ki·∫e + Kd·de/dt``，积分带限幅。

    **理想**是一条声明不是一句谦辞：没有滤波、没有抗饱和的条件积分、
    没有采样保持的相位补偿、没有增益调度。真机ATC600里有什么本模块不知道
    （0062第二节裁决3），所以这里给的是控制教科书上那个能被闭式验证的东西。

    ## 状态是不可变的，一步一个新对象

    ``step``返回``(新控制器, 输出)``而不是就地改。理由与`ContactStep`同源：
    **控制器的积分项是历史**，而历史在本仓一律显式传递、不藏在对象里被偷偷改。

    ## 积分限幅是**必须显式给的**

    没有限幅的积分器在执行器饱和时会一直积（windup），
    解除饱和后要花很久才吐出来——那是真机上最常见的一类控制事故。
    ``integral_limit``没有默认值：**不写就得想一次**。
    """

    proportional: float
    integral_gain: float
    derivative: float
    integral_limit: float
    integral: float = 0.0
    previous_error: float | None = None

    def __post_init__(self) -> None:
        for name in ("proportional", "integral_gain", "derivative"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise DriveError(f"{name} must be finite: {value!r}")
        if not (self.integral_limit > 0.0 and math.isfinite(self.integral_limit)):
            raise DriveError(f"integral_limit must be positive: {self.integral_limit!r}")

    def step(self, error: float, dt_s: float) -> tuple[PidController, float]:
        """走一步：**前向Euler积分、后向差分微分**。

        微分项第一步为零（没有``previous_error``就没有差分），
        **不拿零当上一次误差**——那会在第一步造出一个与阶跃幅值成正比的冲击，
        而真机上那个冲击不存在。
        """

        if not math.isfinite(error):
            raise DriveError(f"error must be finite: {error!r}")
        if not (dt_s > 0.0 and math.isfinite(dt_s)):
            raise DriveError(f"dt_s must be positive and finite: {dt_s!r}")

        integral = self.integral + error * dt_s
        integral = max(-self.integral_limit, min(self.integral_limit, integral))
        rate = 0.0 if self.previous_error is None else (error - self.previous_error) / dt_s
        output = (
            self.proportional * error
            + self.integral_gain * integral
            + self.derivative * rate
        )
        return replace(self, integral=integral, previous_error=error), output


def second_order_natural_frequency_rad_s(
    *, gain_n_per_ampere: float, integral_gain: float, lag_s: float
) -> float:
    """``ω_n = sqrt(K·Ki/τ)``——纯积分闭环的无阻尼自然频率。"""

    if lag_s <= 0.0:
        raise DriveError(f"lag_s must be positive: {lag_s!r}")
    product = gain_n_per_ampere * integral_gain
    if not (product > 0.0 and math.isfinite(product)):
        raise DriveError(f"K·Ki must be positive and finite: {product!r}")
    return math.sqrt(product / lag_s)


def second_order_damping_ratio(
    *, gain_n_per_ampere: float, integral_gain: float, lag_s: float
) -> float:
    """``ζ = 1/(2·sqrt(τ·K·Ki))``——纯积分闭环的阻尼比。

    **它随``Ki``增大而减小**：积分调得越猛越震荡。这条单调性是本闭式最容易
    被判据抓住的性质，比超调值本身更难被一个错误实现凑对。
    """

    if lag_s <= 0.0:
        raise DriveError(f"lag_s must be positive: {lag_s!r}")
    product = lag_s * gain_n_per_ampere * integral_gain
    if not (product > 0.0 and math.isfinite(product)):
        raise DriveError(f"τ·K·Ki must be positive and finite: {product!r}")
    return 0.5 / math.sqrt(product)


def step_response_overshoot(damping_ratio: float) -> float:
    """欠阻尼二阶系统单位阶跃的**相对超调量**``exp(−ζπ/√(1−ζ²))``。

    ``ζ ≥ 1``时没有超调，返回0.0——**过阻尼分支是零不是这条式子的解析延拓**
    （那条延拓在实数域上根本不存在，与research/15对恢复系数得到的结论同源）。

    **它与`contact.restitution_from_damping_ratio`不是同一个函数**，
    尽管两者共享形状``exp(−ζΦ/√(1−ζ²))``：这里``Φ = π``（半个阻尼周期），
    那里``Φ = 2·acos(ζ)``（0052第一节裁定的截断约定下的接触时长）。
    **只在``ζ = 0``处相等**。模块docstring记着这条是怎么被实测纠正的。
    """

    if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
        raise DriveError(f"damping ratio must be finite and nonnegative: {damping_ratio!r}")
    if damping_ratio >= 1.0:
        return 0.0
    return math.exp(-damping_ratio * math.pi / math.sqrt(1.0 - damping_ratio * damping_ratio))


def step_response_peak_time_s(*, natural_frequency_rad_s: float, damping_ratio: float) -> float:
    """峰值时刻``t_p = π/(ω_n·√(1−ζ²))``。

    它与超调量是**独立的两条**：超调只看``ζ``，峰值时刻还看``ω_n``。
    只判超调的话，把``τ``与``Ki``同时缩放同一个倍数不会被发现
    （``ζ``不变而``ω_n``变）——本函数就是那条盲区的封堵。
    """

    if not (natural_frequency_rad_s > 0.0 and math.isfinite(natural_frequency_rad_s)):
        raise DriveError(f"natural frequency must be positive: {natural_frequency_rad_s!r}")
    if not (0.0 <= damping_ratio < 1.0):
        raise DriveError(f"peak time needs an underdamped ratio: {damping_ratio!r}")
    return math.pi / (natural_frequency_rad_s * math.sqrt(1.0 - damping_ratio * damping_ratio))


@dataclass(frozen=True)
class TensionSensor:
    """悬臂式张力传感器（BOSENSE LTS族）：满量程截断＋mV输出＋ADC量化。

    真机（`ED3L_张力控制资料包`2026-06-08已核）：LTS1-5，**5 kg满量程、
    20 mV max输出**，直接接入ATC600的``0—20mV``传感器输入。

    ## 它给对了什么、给错了什么

    * **量化是真误差**：ADC把连续的mV切成台阶，于是闭环再准也停不到台阶之间。
      `resolution_n`是那个台阶折算到牛顿的大小——**它是这条回路精度的地板**；
    * **满量程是截断不是压缩**：超过量程读数停在满量程，与磁粉离合器的饱和同理；
    * **没有的**：零漂、温漂、非线性、蠕变、安装角误差、动态响应。
      真机的LTS装在一只悬臂上，读的是**合力在悬臂方向的分量**，
      而那个方向由两侧包角决定——本类**不建那一层**，
      它把"传感器读到多少张力"当成一个已经标定好的标量。
      真机标定比例`VR451`的换算至今"仍需现场标定确认"（`HARDWARE_TOPOLOGY.md`）。
    """

    full_scale_n: float
    output_at_full_scale_mv: float
    adc_bits: int

    def __post_init__(self) -> None:
        for name in ("full_scale_n", "output_at_full_scale_mv"):
            value = getattr(self, name)
            if not (value > 0.0 and math.isfinite(value)):
                raise DriveError(f"{name} must be positive and finite: {value!r}")
        if isinstance(self.adc_bits, bool) or not isinstance(self.adc_bits, int):
            raise DriveError(f"adc_bits must be an int: {self.adc_bits!r}")
        if not (1 <= self.adc_bits <= 32):
            raise DriveError(
                f"adc_bits must be in [1, 32]: {self.adc_bits!r} —— "
                "位数是硬件定死的；32位以上的张力ADC不存在"
            )

    @property
    def counts(self) -> int:
        """量化级数``2^bits − 1``。"""

        return (1 << self.adc_bits) - 1

    @property
    def resolution_n(self) -> float:
        """一个ADC台阶折算到牛顿——**这条回路精度的地板**。"""

        return self.full_scale_n / self.counts

    def millivolts(self, tension_n: float) -> float:
        """线性输出，超过满量程**截断**。"""

        if not math.isfinite(tension_n):
            raise DriveError(f"tension must be finite: {tension_n!r}")
        clamped = max(0.0, min(self.full_scale_n, tension_n))
        return clamped / self.full_scale_n * self.output_at_full_scale_mv

    def read_n(self, tension_n: float) -> float:
        """读数：截断→量化→折回牛顿。

        量化取**就近**（`round`）而不是截尾：真机ADC是就近舍入的，
        截尾会给出一个系统性偏低的读数，而那会被闭环当成真实的张力不足去补。
        """

        clamped = max(0.0, min(self.full_scale_n, tension_n))
        step = self.resolution_n
        return round(clamped / step) * step


def capstan_transfer_ratio(*, friction_coefficient: float, wrap_angle_rad: float) -> float:
    """一段包角上的张力比``exp(μθ)``（Euler-Eytelwein，全滑移）。

    **这是`cases/capstan_tension_ratio`验的那条式子的连续极限**，
    在这里被用作"传感器测到的张力"与"落位点上的张力"之间的换算。

    符号约定：返回的是**张紧端比松弛端**，恒``≥ 1``。
    调用方按带材走向决定乘还是除——**那个方向搞反，误差是平方**。
    """

    if not math.isfinite(friction_coefficient) or friction_coefficient < 0.0:
        raise DriveError(
            f"friction coefficient must be finite and nonnegative: {friction_coefficient!r}"
        )
    if not math.isfinite(wrap_angle_rad) or wrap_angle_rad < 0.0:
        raise DriveError(f"wrap angle must be finite and nonnegative: {wrap_angle_rad!r}")
    return math.exp(friction_coefficient * wrap_angle_rad)


@dataclass(frozen=True)
class TensionSample:
    """一步的观测：时刻、张力、扭矩、电流命令、误差。**产物是这个，不是内部状态。**"""

    time_s: float
    #: **被控点上的真实张力**（离合器扭矩除以当前卷径）。
    tension_n: float
    #: **传感器读到的张力**：真实张力经包角换算到传感器位置，再过量程截断与量化。
    #: 它与``tension_n``不同不是噪声——见`TensionLoop`第二节。
    measured_n: float
    torque_nmm: float
    current_a: float
    error_n: float


@dataclass(frozen=True)
class TensionLoop:
    """把离合器、卷径换算、PID与时延线串成一个闭环，一步一步推进。

    ## 一步里发生的事，按真机的因果次序

    1. 由**当前**扭矩与卷径算出张力（传感器读到的就是它）；
    2. 误差``e = T_set − T``进PID，出一个电流命令；
    3. 命令进时延线——**这一步是`actuators.ActuationDelayLine`做的**，
       本模块不自己实现时延（0038那套时延机械已经在那里，重写一遍
       等于开第二条真相源）；
    4. 时延线吐出**当步生效**的那条命令（可能是几步之前下的）；
    5. 离合器按那条命令走一步磁滞，得到新扭矩。

    **次序不是随手排的**：把第1步放到第5步之后，等于让传感器读到
    这一步刚刚施加的控制的结果——那是一个物理上不存在的零延迟测量，
    而它会让闭环看起来比真机稳得多。

    ``delay_line``为``None``表示**没有下发时延**。它不是默认值：
    构造时必须显式传``None``，理由与`ActuatorDeclaration`要求
    ``zero_delay_rationale``同源——**零时延是一条声明，不是一次省略**。

    ## 二、闭环调的是它**测到**的量，不是要紧的那个量

    真机的张力传感器（LTS1-5）装在链路中的某一只轮上，而要紧的张力在**落位点**。
    两者之间隔着若干包角，**每个包角把张力乘一个``exp(μθ)``**——
    这正是`cases/capstan_tension_ratio`验的那条式子。

    于是稳态时``T_传感器 = 设定值``，而

        T_落位点 = 设定值 / measurement_transfer

    ``measurement_transfer``就是``T_传感器 / T_被控点``。取``μ = 0.3``、
    总包角90°时它是``1/1.602 = 0.624``——**设定30 N，被控点上是48.1 N，超60%**。

    **这个误差不是控制器不好，是它看不见。** 再好的PID也只能把它测到的量调准。

    ``measurement_transfer``**必须显式给**，没有默认值。给``1.0``是一条声明——
    "传感器就在被控点上、中间一个包角都没有"，而那在真机上通常不成立。
    **把它做成默认1.0，等于让这条误差默默消失。**
    """

    clutch: MagneticParticleClutch
    spool: SpoolTension
    controller: PidController
    setpoint_n: float
    dt_s: float
    delay_line: object | None
    #: ``None``表示**理想传感器**（无量程、无量化）。同样必须显式给。
    sensor: TensionSensor | None
    #: ``T_传感器 / T_被控点``。见类docstring第二节——**没有默认值是有意的**。
    measurement_transfer: float
    torque_nmm: float = 0.0
    turns: float = 0.0
    step_index: int = 0

    def __post_init__(self) -> None:
        if not (self.dt_s > 0.0 and math.isfinite(self.dt_s)):
            raise DriveError(f"dt_s must be positive and finite: {self.dt_s!r}")
        if not (self.setpoint_n > MIN_TENSION_N and math.isfinite(self.setpoint_n)):
            raise DriveError(
                f"setpoint must be a positive tension: {self.setpoint_n!r} —— "
                "零或负的张力设定是松卷工况，本模块不假装能算它"
            )
        if not math.isfinite(self.torque_nmm):
            raise DriveError(f"torque must be finite: {self.torque_nmm!r}")
        if not (self.measurement_transfer > 0.0 and math.isfinite(self.measurement_transfer)):
            raise DriveError(
                f"measurement_transfer must be positive and finite: "
                f"{self.measurement_transfer!r} —— 它是T_传感器/T_被控点，"
                "零或负的传递比意味着传感器读到的不是张力"
            )

    @property
    def tension_n(self) -> float:
        """当前张力——**由当前扭矩与卷径算出，不是一个独立的状态**。"""

        return self.spool.tension_n(self.torque_nmm, self.turns)

    def step(self, *, turns_increment: float = 0.0) -> tuple[TensionLoop, TensionSample]:
        """走一步，返回``(新回路, 本步观测)``。"""

        #: **被控点上的真实张力**，与传感器读到的不是同一个数。
        true_tension = self.tension_n
        at_sensor = true_tension * self.measurement_transfer
        measured = at_sensor if self.sensor is None else self.sensor.read_n(at_sensor)
        error = self.setpoint_n - measured
        controller, current = self.controller.step(error, self.dt_s)

        delay_line = self.delay_line
        if delay_line is None:
            effective_current = current
        else:
            from physics_engine.actuators import ActuationCommand

            channel = delay_line.declaration.channels[0]
            #: ``lower``/``upper``是**逐标量的元组**（一路通道可以有多维），
            #: 不是两个标量。当成标量写会静默拿元组去比大小——
            #: Python会拒，但拒的位置离病根很远。
            clamped = max(channel.lower[0], min(channel.upper[0], current))
            try:
                delay_line, result = delay_line.advance(
                    ActuationCommand(
                        actuator_id=delay_line.actuator_id,
                        values=(clamped,),
                    )
                )
            except ActuatorError as error_from_delay:  # pragma: no cover - 形制变更时才会走到
                raise DriveError(
                    f"时延线拒收命令：{error_from_delay}"
                ) from error_from_delay
            effective_current = result.command.values[0]

        torque = self.clutch.advance_torque_nmm(self.torque_nmm, effective_current, self.dt_s)
        sample = TensionSample(
            time_s=self.step_index * self.dt_s,
            tension_n=true_tension,
            measured_n=measured,
            torque_nmm=self.torque_nmm,
            current_a=effective_current,
            error_n=error,
        )
        return (
            replace(
                self,
                controller=controller,
                delay_line=delay_line,
                torque_nmm=torque,
                turns=self.turns + turns_increment,
                step_index=self.step_index + 1,
            ),
            sample,
        )

    def run(self, steps: int, *, turns_increment: float = 0.0) -> tuple[TensionLoop, tuple[TensionSample, ...]]:
        """连走``steps``步，返回``(末态, 逐步观测)``。"""

        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise DriveError(f"steps must be a positive int: {steps!r}")
        loop = self
        samples: list[TensionSample] = []
        for _ in range(steps):
            loop, sample = loop.step(turns_increment=turns_increment)
            samples.append(sample)
        return loop, tuple(samples)


__all__ = [
    "MIN_TENSION_N",
    "DriveError",
    "MagneticParticleClutch",
    "PidController",
    "SpoolTension",
    "TensionLoop",
    "TensionSample",
    "TensionSensor",
    "capstan_transfer_ratio",
    "second_order_damping_ratio",
    "second_order_natural_frequency_rad_s",
    "step_response_overshoot",
    "step_response_peak_time_s",
]
