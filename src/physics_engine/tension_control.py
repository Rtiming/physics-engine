"""张力闭环装配层——把`drives`的执行链接到`transport`的对象上（力学域，决策0070）。

守plans/15第三节阶段一的1.2与1.5。

## 在这之前，两条链各跑各的

    drives      电流 → 扭矩 → 时延 → PID          （**对象里没有扰动**）
    transport   扭矩 → 转速 → 材料长度 → 张力      （**没有控制器**）

0066第九节把"接起来"记成一笔明账，原文是：
"接起来要一层把``brake_torque_nmm``交给`MagneticParticleClutch`的装配层，
且要重裁'控制器接在哪一路'。**那是一次装配决策，不是一次import**。"

本模块就是那一层。**它不新增任何物理**：跨段弹性、力矩平衡、磁滞时滞、
量化、时延一条都不是这里写的，本模块只决定**谁在什么时刻把哪个数交给谁**。

## 一、裁决A：控制器接**制动力矩**，不接收线速度

真机是双执行器（`HARDWARE_TOPOLOGY.md`）：`ED3L-08AEA`伺服在CSP位置模式下
定收线速度，POC-050L磁粉离合器定制动力矩。`transport.SpanTransportLoop`
恰好也是这两个输入。裁"接哪一路"要看三件事：

### 1. 稳态权限：一个通道的增益与最不确定的那个参数无关，另一个正比于它

稳态是``T = M/R + c·v/R²``（0066第三节），于是

    ∂T/∂M = 1/R                 ← **与``c``无关**，纯几何
    ∂T/∂v = c/R²                ← **正比于``c``**，而``c``是本模型里
                                   唯一的假设阻尼通道，一条实测都没有

``c = 0``（理想轴承）时收线速度对张力的稳态增益**恰为零**。
**把整条回路的权限建在唯一一个我们没有实测的参数上，等于什么都没建。**

本工况（``R = 60 mm``、``c = 50``、``M₀ = 1200 N·mm``、``v₀ = 20 mm/s``）
按各自额定量的1%算：制动力矩给``0.2 N``、收线速度给``0.0027778 N``——
**72倍**。这个数在案例里有一条门判。

### 2. 收线速度是**被绕线程序占用的过程量**

伺服在CSP位置模式下跟的是绕线程序给的位置-时间表。拿它去抑制张力扰动，
等于让张力环去改**带材落在哪里**——那是落位的事（`laydown`），不是张力环的事。
0066第5.1节已经写过同一句话的另一半："收线端是伺服给定的，张力改不动它"。

### 3. 真机就是这么接的

    LTS1-5 → ATC600 →（手动模式，只当功率级）→ TBA420 → POC-050L
    用户自己的环：读`VR451`（张力）→ 写`VR481`（手动量）

**读的是张力，写的是离合器。收线端不在这条环里。**

### 裁决的代价，用数说：这条通道**够不着**跨段谐振

代价必须同批写下来，否则这条裁决就成了一句只报好处的话。
磁粉离合器的磁滞时滞是**假设输入**``τ = 0.05 s``（0062第二节裁决2），
它给出的执行器极点是``1/τ = 20 rad/s``；而跨段-放线盘谐振是
``ω_n = 379.6 rad/s``（60.42 Hz）——**执行器比对象慢19倍**。

后果实测在案例`closed_loop_tension_step`里：额定离合器下闭环把峰值从
``1.060453 N``压到``1.057105 N``（**0.32%**），基本等于没压。
**这不是控制器不好，是这条通道的带宽不在那里。**

真机上这条谐振是靠**摆杆/储带机构**（dancer）压下去的，而本仓没有摆杆模型——
这条已登记进`docs/plans/07`第六节。

## 二、裁决B：两条链路的时钟——**整数倍抽取＋零阶保持，比值不许不是整数**

`transport`是半隐式Euler，实测一阶（0066第6.3节），步长要能分辨
``ω_n = 379.6 rad/s``；控制器是采样系统，真机上跑在毫秒量级。
**两者不是同一个时钟**，而"让库去猜"正是本仓反复吃亏的那一类。

裁决三条，逐条都是失败关闭：

1. **控制周期 ＝ ``control_decimation × plant_dt_s``**，
   ``control_decimation``是正整数且**没有默认值**。
   为什么必须是整数：非整数比值会让某个控制拍落在推进步的中间，
   而那一步的零阶保持边界**无从定义**——"这一步前半段用旧命令、
   后半段用新命令"要么改推进格式、要么就是一次静默的近似。
   **本层不做静默近似。**
2. **命令在两个控制拍之间零阶保持**。保持的是**电流**不是扭矩：
   电流是执行器的输入，扭矩是它的状态。保持扭矩等于把磁滞旁路掉。
3. **时延线的``dt_s``必须逐位等于控制周期**。`ActuationDelayLine`的时延是
   按拍数记的（0038），拿推进步长去填它会让声明的``delay_s``
   变成另一个时间——**两条时钟对不齐的最常见形态就是这一个**。

### 一步里发生的事，按真机的因果次序

    1. 取**步首**的跨段张力（这是被控点上的真值）
    2. 若本步是控制拍：过传感器 → 控制器 → 前馈相加 → 时延线 → 更新保持电流
    3. 离合器按保持电流走**一个推进步**的磁滞
    4. 跨段按**步首**的制动力矩走一个推进步
    5. 观测取步首快照

第1步在第2步之前、第4步用**步首**扭矩，两条都不是随手排的：
把测量放到执行之后等于一个物理上不存在的零延迟测量，
而它会让闭环看起来比真机稳得多（`drives.TensionLoop`那条同源）。

## 三、传感器位置≠被控点：绞盘比进环（plans/15第1.5条）

`drives.capstan_transfer_ratio`早就有了，缺的是把它接进闭环。
本模块的口径与`drives.TensionLoop.measurement_transfer`**不同，而且更清楚**：

* 传感器装在**跨段上**，它读到的就是`transport`算出来的那个张力
  （量程截断与ADC量化之外没有别的变换）；
* **落位点在下游隔着一段包角**，张力沿走向被绞盘放大``exp(μθ)``。

于是闭环稳态时``T_传感器 = 设定值``，而``T_落位点 = 设定值·exp(μθ)``。
``μ = 0.3``、90°包角给``exp(0.4712) = 1.601937``——**落位点比设定高60.19%**。

**这个误差不是控制器不好，是它看不见。** 再好的环也只能把它测到的量调准。

### 绞盘不给跨段加载，这一条是物理不是省事

导轮把两侧解耦：跨段（放线盘↔导轮）里的张力是``T``，导轮下游是``T·exp(μθ)``，
放线盘看到的是``T``。**这正是绞盘存在的理由**，所以`laydown_tension_n`
是一层**观测**而不是一个动力学环节。代价如实写：本模块因此**算不了**
"落位点张力反过来改变跨段张力"，而真机上导轮的滑移状态一变那件事就会发生。

### ``sensor_on_tight_side``没有默认值——方向搞反误差是**平方**

`capstan_transfer_ratio`返回的恒是**张紧端比松弛端**（``≥ 1``），
乘还是除由调用方按走向定。搞反一次，落位点张力的估计差
``exp(μθ)²``而不是``exp(μθ)``：本工况**2.566203**对1.601937，
即157%的误差对60%。有一条门判这个平方关系。

## 四、闭环的闭式：**一个四阶系统，而它的稳定界是解出来的**

在稳态附近线性化（``K ≡ −dT/dL_mat``、``a ≡ 1000R²/J``、``b ≡ 1000R/J``、
``d ≡ 1000c/J``、``G ≡ K·b·k_M``），把离合器一阶时滞与理想PID串进0066那两条：

    dδT/dt      = −K·δv_放线 + K·δv_收线
    dδv_放线/dt = a·δT − b·δM − d·δv_放线
    δM          = k_M·δI/(1 + τs)
    δI          = −(Kp + Ki/s + Kd·s)·δT          ← 设定值不动时``e = −δT``

消元得**闭环特征多项式**

    D(s) = τ·s⁴ + (1 + dτ)·s³ + (d + Ka·τ + G·Kd)·s²
           + (Ka + G·Kp)·s + G·Ki

以及收线端速度阶跃``Δv``下张力的**精确**像函数

    δT(s) = Δv·K·(τ·s² + (1 + dτ)·s + d) / D(s)

三条读得出来的结论，逐条都有门：

1. **积分项吃掉阻尼**。``τ → 0``时``D``退化成三次，Routh条件给
   ``Ki < (d + G·Kd)(Ka + G·Kp)/G``——**纯积分（``Kp = Kd = 0``）时它是
   ``d·Ka/G``**，而``d = 2ζω_n``。积分调得越猛，振荡衰减得越慢，
   过了这条界闭环发散。`integral_gain_stability_limit`是这条界的通用形（含``τ``）；
2. **微分项加阻尼**：``Kd``只出现在``s²``系数上，而那正是``2ζ_闭环·ω_闭环``那一项。
   **能压住振荡的是``Kd``，不是``Ki``**；
3. **积分项把稳态误差杀成恒零**：``δT(s)``的分子在``s = 0``处是``K·d``、
   分母是``G·Ki``，终值定理给``lim s·δT(s) = 0``。开环那一侧是
   ``Δv·c/R²``（本工况``0.027778 N``）。

**闭式在这里是可对拍的，不是装饰**：`cases/closed_loop_tension_step`
的金标生成器独立重写了上面每一条，用Durand-Kerner求根＋留数展开给出
逐时刻的``δT(t)``，与本模块的离散推进逐点对拍。

## 五、全部输入是假设，产物永久``hypothesis_only``

带材``EA``、跨段长度、放线盘转动惯量、轴承阻力矩``c``、离合器时滞``τ``与
电流-扭矩增益、绞盘``μ``与包角——**一条实测都没有**，
逐条出处见0062第二节裁决2那张清单。本模块证明的是**机制**：
两条链怎么闭合、时钟怎么对、以及绞盘比为什么闭环看不见。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from physics_engine.actuators import ActuatorError
from physics_engine.drives import (
    MagneticParticleClutch,
    TensionController,
    TensionSensor,
    capstan_transfer_ratio,
)
from physics_engine.transport import (
    MM_PER_M,
    FreeSpan,
    PayoutReel,
    SpanTransportLoop,
    steady_state_tension_n,
)


class TensionControlError(ValueError):
    """张力闭环装配的一切失败关闭。"""


def _require_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TensionControlError(f"{name} must be a real number: {value!r}")
    if not math.isfinite(value):
        raise TensionControlError(f"{name} must be finite: {value!r}")
    return float(value)


def _require_positive(value: float, name: str) -> float:
    value = _require_finite(value, name)
    if value <= 0.0:
        raise TensionControlError(f"{name} must be positive: {value!r}")
    return value


# --------------------------------------------------------------- 绞盘观测层 ---


@dataclass(frozen=True)
class CapstanSpan:
    """传感器与落位点之间的那段包角——**一层观测，不是一个动力学环节**。

        T_张紧端 = T_松弛端 · exp(μ·θ)

    这是`cases/capstan_tension_ratio`验的那条式子的连续极限，
    在这里回答的是"我测到的这个数，和真正要紧的那个数差多少"。

    ## ``sensor_on_tight_side``**没有默认值**

    ``True``  ⟹ 传感器在张紧端，落位点在松弛端 ⟹ ``T_落位 = 读数/exp(μθ)``；
    ``False`` ⟹ 传感器在松弛端，落位点在张紧端 ⟹ ``T_落位 = 读数·exp(μθ)``。

    真机构型是后者：LTS1-5在跨段上、落位点在导轮下游，
    带材沿走向被绞盘**放大**。

    **写成默认值等于替声明者拿主意，而搞反的代价是平方**
    （``exp(μθ)²``而不是``exp(μθ)``）——与`TensionLoop.measurement_transfer`
    "没有默认值是有意的"同源。
    """

    friction_coefficient: float
    wrap_angle_rad: float
    #: 传感器在张紧端还是松弛端。**没有默认值**，理由见类docstring。
    sensor_on_tight_side: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_on_tight_side, bool):
            raise TensionControlError(
                f"sensor_on_tight_side must be an explicit bool: "
                f"{self.sensor_on_tight_side!r} —— 它是走向的声明，"
                "搞反的代价是平方"
            )
        #: 两个参数的合法性交给`drives.capstan_transfer_ratio`判——
        #: **不重写一遍**，那会开第二条真相源。
        capstan_transfer_ratio(
            friction_coefficient=self.friction_coefficient,
            wrap_angle_rad=self.wrap_angle_rad,
        )

    @property
    def ratio(self) -> float:
        """``exp(μθ)``，恒``≥ 1``。"""

        return capstan_transfer_ratio(
            friction_coefficient=self.friction_coefficient,
            wrap_angle_rad=self.wrap_angle_rad,
        )

    def laydown_tension_n(self, sensor_tension_n: float) -> float:
        """由传感器读数给出**落位点**的张力。方向由``sensor_on_tight_side``定。"""

        reading = _require_finite(sensor_tension_n, "sensor_tension_n")
        return reading / self.ratio if self.sensor_on_tight_side else reading * self.ratio


# --------------------------------------------------------------- 闭环的闭式 ---


def closed_loop_characteristic_polynomial(
    *,
    span_stiffness_n_per_mm: float,
    radius_mm: float,
    inertia_kg_mm2: float,
    bearing_damping_nmm_s: float,
    torque_per_ampere_nmm: float,
    clutch_lag_s: float,
    proportional: float,
    integral_gain: float,
    derivative: float,
) -> tuple[float, float, float, float, float]:
    """闭环特征多项式的系数，从``s⁴``到``s⁰``（模块docstring第四节的推导）。

        τ·s⁴ + (1+dτ)·s³ + (d + Ka·τ + G·Kd)·s² + (Ka + G·Kp)·s + G·Ki

    其中``a = 1000R²/J``、``d = 1000c/J``、``Ka = K·a``、``G = K·(1000R/J)·k_M``。
    那三个``1000``都是`transport.MM_PER_M`——**掉一个，``ω_n``差31.6倍**。

    ``clutch_lag_s = 0``是允许的：它是"执行器无限快"的极限，
    此时最高次系数为零、多项式降为三次。**那不是一台离合器，
    是把执行器带宽从判据里摘出去的口径**，判据要分辨"环写错了"
    与"执行器太慢了"这两件事时用它。
    """

    stiffness = _require_positive(span_stiffness_n_per_mm, "span_stiffness_n_per_mm")
    radius = _require_positive(radius_mm, "radius_mm")
    inertia = _require_positive(inertia_kg_mm2, "inertia_kg_mm2")
    damping = _require_finite(bearing_damping_nmm_s, "bearing_damping_nmm_s")
    if damping < 0.0:
        raise TensionControlError(
            f"bearing_damping_nmm_s不能为负: {damping!r} —— "
            "负的粘性阻力矩是一台往系统里灌能量的轴承"
        )
    gain = _require_positive(torque_per_ampere_nmm, "torque_per_ampere_nmm")
    lag = _require_finite(clutch_lag_s, "clutch_lag_s")
    if lag < 0.0:
        raise TensionControlError(
            f"clutch_lag_s不能为负: {lag!r} —— 负的一阶时滞是一个反因果的执行器"
        )
    for name, value in (
        ("proportional", proportional),
        ("integral_gain", integral_gain),
        ("derivative", derivative),
    ):
        _require_finite(value, name)

    acceleration_per_tension = MM_PER_M * radius * radius / inertia
    acceleration_per_torque = MM_PER_M * radius / inertia
    velocity_damping = MM_PER_M * damping / inertia
    stiffness_times_a = stiffness * acceleration_per_tension
    loop_gain = stiffness * acceleration_per_torque * gain
    return (
        lag,
        1.0 + velocity_damping * lag,
        velocity_damping + stiffness_times_a * lag + loop_gain * derivative,
        stiffness_times_a + loop_gain * proportional,
        loop_gain * integral_gain,
    )


def integral_gain_stability_limit(
    *,
    span_stiffness_n_per_mm: float,
    radius_mm: float,
    inertia_kg_mm2: float,
    bearing_damping_nmm_s: float,
    torque_per_ampere_nmm: float,
    clutch_lag_s: float,
    proportional: float,
    derivative: float,
) -> float:
    """积分增益的Routh稳定界——**超过它闭环发散**。

    四阶Routh（``a₄s⁴+a₃s³+a₂s²+a₁s+a₀``全系数为正时）要求

        a₃a₂ > a₄a₁        且        (a₃a₂ − a₄a₁)·a₁ > a₃²·a₀

    而``a₀ = G·Ki``，于是

        Ki_界 = (a₃a₂ − a₄a₁)·a₁ / (a₃²·G)

    ``τ = 0``时``a₄ = 0``，上式退化成三次的Routh``a₂a₁ > a₃a₀``即
    ``Ki < (d + G·Kd)(Ka + G·Kp)/G``——**两条是同一条**，有一条门判这个退化。

    ## 它说的话比一个数更要紧

    纯积分（``Kp = Kd = 0``、``τ = 0``）时界是``d·Ka/G``，而``d = 2ζω_n``：
    **界正比于开环阻尼**。开环阻尼是``ζ = 0.0132``那一档（0066第6.2节），
    于是这条界很低——**积分项在这条链路上不是用来压振荡的，
    它压振荡的能力是负的**（``Ki``增大 ⟹ 振荡模态的阻尼减小）。
    压振荡的是``Kd``。

    第一象限之外（某个系数为负）本函数**失败关闭**而不是返回一个负数：
    那时"积分增益的界"这个说法本身就不成立，闭环已经因为别的原因不稳。
    """

    coefficients = closed_loop_characteristic_polynomial(
        span_stiffness_n_per_mm=span_stiffness_n_per_mm,
        radius_mm=radius_mm,
        inertia_kg_mm2=inertia_kg_mm2,
        bearing_damping_nmm_s=bearing_damping_nmm_s,
        torque_per_ampere_nmm=torque_per_ampere_nmm,
        clutch_lag_s=clutch_lag_s,
        proportional=proportional,
        derivative=derivative,
        integral_gain=0.0,
    )
    fourth, third, second, first, _ = coefficients
    if not (third > 0.0 and second > 0.0 and first > 0.0):
        raise TensionControlError(
            f"特征多项式的低阶系数不全为正{coefficients!r} —— "
            "此时'积分增益的界'这个说法不成立：闭环已经因为别的原因不稳，"
            "本函数不假装能给一个界"
        )
    routh = third * second - fourth * first
    if routh <= 0.0:
        raise TensionControlError(
            f"Routh第一列已经非正（a₃a₂ − a₄a₁ = {routh!r}）—— "
            "执行器时滞相对对象太大，任何正的积分增益都不稳"
        )
    loop_gain = closed_loop_characteristic_polynomial(
        span_stiffness_n_per_mm=span_stiffness_n_per_mm,
        radius_mm=radius_mm,
        inertia_kg_mm2=inertia_kg_mm2,
        bearing_damping_nmm_s=bearing_damping_nmm_s,
        torque_per_ampere_nmm=torque_per_ampere_nmm,
        clutch_lag_s=clutch_lag_s,
        proportional=proportional,
        derivative=derivative,
        integral_gain=1.0,
    )[4]
    return routh * first / (third * third * loop_gain)


# ------------------------------------------------------------------- 推进 ---


@dataclass(frozen=True)
class TensionControlSample:
    """一步的观测，全部取**步首**状态。**产物是这个，不是内部状态。**"""

    time_s: float
    #: 跨段（＝被控点，＝传感器所在处）的**真实**张力。
    tension_n: float
    #: 传感器读到的张力：真值过量程截断与ADC量化。``sensor``为``None``时等于真值。
    measured_n: float
    #: **落位点**的张力：读数经绞盘比换算。它与``measured_n``差``exp(μθ)``，
    #: 而**闭环看不见这个差**——见模块docstring第三节。
    laydown_tension_n: float
    #: ``设定值 − 读数``。**回路自己的记账**，不是控制器看到的量（决策0070）。
    error_n: float
    #: 本步生效的**保持电流**（含前馈）。控制拍之间它不变。
    current_a: float
    brake_torque_nmm: float
    angular_velocity_rad_s: float
    payout_speed_mm_s: float
    takeup_speed_mm_s: float
    #: 本步是不是控制拍。**判多速率对没对上就靠它**。
    control_tick: bool


@dataclass(frozen=True)
class ClosedTensionLoop:
    """`transport`的对象 ＋ `drives`的执行链 ＋ 一个协议控制器，多速率零阶保持。

    两条时钟的关系见模块docstring第二节：
    ``控制周期 = control_decimation × plant.dt_s``，比值必须是正整数。

    ## 状态里为什么有一个``held_current_a``

    它是**零阶保持的那个值**——控制拍之间执行器输入不变。
    把它藏成"每步都重算一遍"会让抽取比失去意义（那等于控制器一直在跑），
    而把它记成扭矩会把磁滞旁路掉。**保持的是执行器的输入，不是它的状态。**

    ## ``feedforward_current_a``：稳态起点的定义，不是一个默认值

    误差型控制器在稳态处输出零，于是**没有前馈就没有稳态**——
    回路会从"制动力矩为零"起手，那不是本入口说的那个起点。
    `at_steady_state`按``M₀/k_M``反解它，**这正是0066那条
    `SpoolTension.brake_torque_for_tension_nmm`前馈的用处第一次被用上**。
    """

    #: 对象。**本模块不重写它的推进**（那会开第二条真相源）。
    plant: SpanTransportLoop
    clutch: MagneticParticleClutch
    #: 协议不是具体类（决策0070）。
    controller: TensionController
    #: ``None``表示**传感器就在被控点上、中间一个包角都没有**。显式声明。
    capstan: CapstanSpan | None
    #: ``None``表示**理想传感器**（无量程、无量化）。显式声明。
    sensor: TensionSensor | None
    setpoint_n: float
    #: 控制周期 ÷ 推进步长。正整数，**没有默认值**。
    control_decimation: int
    feedforward_current_a: float
    #: ``None``表示**没有下发时延**。它不是默认值——零时延是一条声明。
    delay_line: object | None
    brake_torque_nmm: float
    held_current_a: float
    step_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.plant, SpanTransportLoop):
            raise TensionControlError(f"plant must be a SpanTransportLoop: {self.plant!r}")
        if not isinstance(self.clutch, MagneticParticleClutch):
            raise TensionControlError(
                f"clutch must be a MagneticParticleClutch: {self.clutch!r}"
            )
        if not isinstance(self.controller, TensionController):
            raise TensionControlError(
                f"controller must satisfy drives.TensionController: {self.controller!r}"
            )
        if self.capstan is not None and not isinstance(self.capstan, CapstanSpan):
            raise TensionControlError(f"capstan must be a CapstanSpan or None: {self.capstan!r}")
        if self.sensor is not None and not isinstance(self.sensor, TensionSensor):
            raise TensionControlError(f"sensor must be a TensionSensor or None: {self.sensor!r}")
        if isinstance(self.control_decimation, bool) or not isinstance(
            self.control_decimation, int
        ):
            raise TensionControlError(
                f"control_decimation must be an int: {self.control_decimation!r} —— "
                "非整数比值会让某个控制拍落在推进步中间，那一步的零阶保持边界无从定义"
            )
        if self.control_decimation < 1:
            raise TensionControlError(
                f"control_decimation must be >= 1: {self.control_decimation!r}"
            )
        _require_positive(self.setpoint_n, "setpoint_n")
        _require_finite(self.feedforward_current_a, "feedforward_current_a")
        _require_finite(self.brake_torque_nmm, "brake_torque_nmm")
        _require_finite(self.held_current_a, "held_current_a")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TensionControlError(f"step_index must be an int: {self.step_index!r}")
        if self.delay_line is not None:
            delay_dt = getattr(self.delay_line, "dt_s", None)
            if delay_dt != self.control_period_s:
                raise TensionControlError(
                    f"时延线的dt_s是{delay_dt!r}而控制周期是{self.control_period_s!r} —— "
                    "`ActuationDelayLine`按**拍数**记时延，两条时钟对不齐时"
                    "声明的delay_s会变成另一个时间。这是裁决B的第3条"
                )

    @property
    def control_period_s(self) -> float:
        """``control_decimation × plant.dt_s``。**两条时钟的关系就是这一行。**"""

        return self.control_decimation * self.plant.dt_s

    @property
    def tension_n(self) -> float:
        """跨段张力——由对象的材料长度算出，不是一个独立的状态。"""

        return self.plant.tension_n

    @classmethod
    def at_steady_state(
        cls,
        *,
        span: FreeSpan,
        reel: PayoutReel,
        clutch: MagneticParticleClutch,
        controller: TensionController,
        capstan: CapstanSpan | None,
        sensor: TensionSensor | None,
        plant_dt_s: float,
        control_decimation: int,
        brake_torque_nmm: float,
        line_speed_mm_s: float,
        delay_line: object | None,
        forbid_slack: bool,
    ) -> ClosedTensionLoop:
        """从**闭式稳态**起手：对象、执行器与控制器三者同时在不动点上。

        对象那一半交给`transport.SpanTransportLoop.at_steady_state`
        （``T* = M/R + c·v/R²``、``L* = EA·L_geo/(EA+T*)``、``ω* = v/R``）；
        本入口再补两件：

        * **离合器起点扭矩取``M₀``**——不是零。从零起手要先走几个``τ``把扭矩
          建起来，那段瞬态与判据要判的阶跃响应叠在一起，
          与`capstan_tension_ratio`那条"起点必须已经穿透"同源；
        * **前馈电流取``M₀/k_M``**，于是误差型控制器在起点输出零而总电流恰好是
          ``I₀``。**设定值由稳态定义**（``T*``），给一个别的设定值等于让回路
          从非零误差起手——那是设定值阶跃工况，不是本入口。

        起点电流必须落在离合器的线性段（``|I₀| ≤ 饱和电流``），否则
        "从稳态起手"这句话不成立，当场失败关闭。
        """

        plant = SpanTransportLoop.at_steady_state(
            span=span,
            reel=reel,
            dt_s=plant_dt_s,
            brake_torque_nmm=brake_torque_nmm,
            line_speed_mm_s=line_speed_mm_s,
            forbid_slack=forbid_slack,
        )
        torque = _require_finite(brake_torque_nmm, "brake_torque_nmm")
        feedforward = torque / clutch.torque_per_ampere_nmm
        if abs(feedforward) > clutch.saturation_current_a:
            raise TensionControlError(
                f"稳态前馈电流{feedforward!r} A超出离合器饱和电流"
                f"{clutch.saturation_current_a!r} A —— 起点在饱和段上，"
                "'从稳态起手'这句话不成立"
            )
        setpoint = steady_state_tension_n(
            brake_torque_nmm=torque,
            radius_mm=reel.radius_mm,
            bearing_damping_nmm_s=reel.bearing_damping_nmm_s,
            line_speed_mm_s=line_speed_mm_s,
        )
        return cls(
            plant=plant,
            clutch=clutch,
            controller=controller,
            capstan=capstan,
            sensor=sensor,
            setpoint_n=setpoint,
            control_decimation=control_decimation,
            feedforward_current_a=feedforward,
            delay_line=delay_line,
            brake_torque_nmm=torque,
            held_current_a=feedforward,
        )

    def step(self, *, takeup_speed_mm_s: float) -> tuple[ClosedTensionLoop, TensionControlSample]:
        """走一个**推进步**，返回``(新回路, 本步观测)``。次序见模块docstring第二节。"""

        true_tension = self.plant.tension_n
        measured = (
            true_tension if self.sensor is None else self.sensor.read_n(true_tension)
        )
        laydown = (
            measured if self.capstan is None else self.capstan.laydown_tension_n(measured)
        )
        error = self.setpoint_n - measured

        is_tick = self.step_index % self.control_decimation == 0
        controller = self.controller
        delay_line = self.delay_line
        current = self.held_current_a
        if is_tick:
            controller, command = self.controller.step(
                measurement_n=measured,
                setpoint_n=self.setpoint_n,
                dt_s=self.control_period_s,
            )
            current = self.feedforward_current_a + command
            if delay_line is not None:
                from physics_engine.actuators import ActuationCommand

                channel = delay_line.declaration.channels[0]
                #: ``lower``/``upper``是**逐标量的元组**（`TensionLoop`那条同源）。
                clamped = max(channel.lower[0], min(channel.upper[0], current))
                try:
                    delay_line, result = delay_line.advance(
                        ActuationCommand(
                            actuator_id=delay_line.actuator_id, values=(clamped,)
                        )
                    )
                except ActuatorError as error_from_delay:  # pragma: no cover - 形制变更时才走到
                    raise TensionControlError(
                        f"时延线拒收命令：{error_from_delay}"
                    ) from error_from_delay
                current = result.command.values[0]

        sample = TensionControlSample(
            time_s=self.step_index * self.plant.dt_s,
            tension_n=true_tension,
            measured_n=measured,
            laydown_tension_n=laydown,
            error_n=error,
            current_a=current,
            brake_torque_nmm=self.brake_torque_nmm,
            angular_velocity_rad_s=self.plant.angular_velocity_rad_s,
            payout_speed_mm_s=self.plant.reel.payout_speed_mm_s(
                self.plant.angular_velocity_rad_s
            ),
            takeup_speed_mm_s=_require_finite(takeup_speed_mm_s, "takeup_speed_mm_s"),
            control_tick=is_tick,
        )

        #: 对象用**步首**扭矩推进；离合器同步走一步磁滞。两者各自用对方的步首值，
        #: 组合的阶与半隐式Euler同级（一阶）——**阶是量出来的**，案例有一条门。
        plant, _ = self.plant.step(
            brake_torque_nmm=self.brake_torque_nmm, takeup_speed_mm_s=takeup_speed_mm_s
        )
        torque = self.clutch.advance_torque_nmm(
            self.brake_torque_nmm, current, self.plant.dt_s
        )
        return (
            replace(
                self,
                plant=plant,
                controller=controller,
                delay_line=delay_line,
                brake_torque_nmm=torque,
                held_current_a=current,
                step_index=self.step_index + 1,
            ),
            sample,
        )

    def run(
        self, steps: int, *, takeup_speed_mm_s: float
    ) -> tuple[ClosedTensionLoop, tuple[TensionControlSample, ...]]:
        """收线端速度恒定地连走``steps``个推进步。"""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise TensionControlError(f"steps must be a positive int: {steps!r}")
        loop = self
        samples: list[TensionControlSample] = []
        for _ in range(steps):
            loop, sample = loop.step(takeup_speed_mm_s=takeup_speed_mm_s)
            samples.append(sample)
        return loop, tuple(samples)


__all__ = [
    "CapstanSpan",
    "ClosedTensionLoop",
    "TensionControlError",
    "TensionControlSample",
    "closed_loop_characteristic_polynomial",
    "integral_gain_stability_limit",
]
