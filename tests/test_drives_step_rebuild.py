"""`drives`那两处**每步重建**的门——决策0087丙3的注错轮补上的四条。

## 这个文件为什么存在

0083第四节把`transport`/`tension_control`那两处`dataclasses.replace`换成直接构造，
本轮把`drives`那1,780,613次也换掉。改法逐字相同、语义逐字相同、
`float.hex()`逐字节对拍（184080行、SHA相同）。

**但注错轮抓到既有门里的三个洞**：`replace`把"没被覆盖的字段原样带过去"这件事
做得太安静，于是**从来没有一条门在判那些字段真的被带过去了**。
换成直接构造之后它们变成了十一个手写的赋值，**每一个都可以写错而没人知道**：

| 注错 | 首轮 |
|---|---|
| D7 `turns=self.turns`（卷径永不生长） | **全绿** |
| D12 `sensor=None`（量化悄悄消失） | **全绿** |
| D13 `setpoint_n=30.0`（写死成测试自己用的那个数） | **全绿** |

三条都是"**改了以后行为不同、而所有门都看不见**"。本文件把它们补上。
D13尤其值得记：既有门全部用`SETPOINT_N = 30.0`，
于是**把设定值写死成30.0在它们下面是完全等价的**——
一条判据只要它的输入在全仓只出现过一个取值，它就判不出"这个值被写死了"。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.actuators import (
    ActuationCommand,
    ActuationDelayLine,
    ActuatorDeclaration,
    CommandChannel,
)
from physics_engine.drives import (
    MagneticParticleClutch,
    PidController,
    SpoolTension,
    TensionLoop,
    TensionSensor,
)

CLUTCH = MagneticParticleClutch(
    torque_per_ampere_nmm=23256.0, rated_torque_nmm=50000.0, lag_s=0.05
)
SPOOL = SpoolTension(barrel_radius_mm=60.0, tape_thickness_mm=0.1)


def _loop(**overrides) -> TensionLoop:
    base = {
        "clutch": CLUTCH,
        "spool": SPOOL,
        "controller": PidController(
            proportional=0.0,
            integral_gain=0.05,
            derivative=0.0,
            integral_limit=1.0e6,
        ),
        "setpoint_n": 30.0,
        "dt_s": 1.0e-3,
        "delay_line": None,
        "sensor": None,
        "measurement_transfer": 1.0,
    }
    base.update(overrides)
    return TensionLoop(**base)  # type: ignore[arg-type]


def test_the_spool_really_winds_and_the_tension_falls_with_the_growing_radius() -> None:
    """**必红**（注错D7）：`turns`不增长时本门红。

    卷径生长是本仓在真机上最基本的一条非线性（`SpoolTension`的docstring：
    "同一个扭矩在卷满时给出的张力比空卷时小`R0/(R0+n·t)`倍"），
    而**既有门里没有一条判它跨步累积**。

    判两件事，两件都要：`turns`逐步累加（数值）、
    **同一个扭矩下的张力真的按那个比例掉下去**（物理后果）。
    只判前者的门挡不住"`turns`加对了但没被用上"。
    """

    increment = 0.25
    loop = _loop()
    loop, samples = loop.run(40, turns_increment=increment)
    assert loop.turns == pytest.approx(40 * increment, rel=0, abs=1e-12)
    assert loop.step_index == 40

    #: 物理后果：同一个扭矩、不同匝数，张力比等于半径反比。
    torque = 3000.0
    ratio = SPOOL.tension_n(torque, loop.turns) / SPOOL.tension_n(torque, 0.0)
    expected = SPOOL.radius_mm(0.0) / SPOOL.radius_mm(loop.turns)
    assert ratio == pytest.approx(expected, rel=1e-15)
    assert ratio < 0.99, "卷径一步都没长——`turns`没有被带进重建"

    #: 与不卷的那次并排：同一条回路、同一个控制律，末态张力必须不同。
    still = _loop()
    still, still_samples = still.run(40, turns_increment=0.0)
    assert still.turns == 0.0
    assert samples[-1].tension_n != still_samples[-1].tension_n


def test_the_sensor_survives_every_step_not_just_the_first() -> None:
    """**必红**（注错D12）：重建时把`sensor`丢掉，本门红。

    丢掉传感器的后果是**量化悄悄消失**——回路突然读到一个真机上不存在的
    无限精度张力，而它看起来只是"控得更准了一点"。

    判据是**逐步**判：每一步的`measured_n`都必须落在ADC台阶的整数倍上。
    只判第一步的门抓不住"第0步用了传感器、第1步之后没有"。
    """

    sensor = TensionSensor(
        full_scale_n=60.0, output_at_full_scale_mv=10.0, adc_bits=8
    )
    resolution = sensor.resolution_n
    loop = _loop(sensor=sensor)
    loop, samples = loop.run(200)
    assert loop.sensor is sensor
    for index, sample in enumerate(samples):
        quotient = sample.measured_n / resolution
        assert quotient == pytest.approx(round(quotient), rel=0, abs=1e-9), (
            f"第{index}步的读数{sample.measured_n!r}不是量化台阶的整数倍——"
            "传感器在这一步之前就掉了"
        )
    #: 前置断言：量化真的在动（读数不是恒为零，也不是恒等于真实张力）。
    assert any(
        sample.measured_n != sample.tension_n for sample in samples
    ), "量化一次都没生效，这条门在空判"


def test_the_setpoint_survives_the_step_and_is_not_a_hard_coded_thirty() -> None:
    """**必红**（注错D13）：把`setpoint_n`写死成30.0，本门红。

    既有门全部用`SETPOINT_N = 30.0`，于是写死成30.0在它们下面**完全等价**。
    一条判据只要它的输入在全仓只出现过一个取值，
    **它就判不出"这个值被写死了"**——这一条与0049第六节那句
    "门认得『被提到』、认不得『被登记』"同族。

    本门因此刻意取一个**别处没有出现过的**设定值。
    """

    setpoint = 17.5
    loop = _loop(setpoint_n=setpoint)
    loop, samples = loop.run(1500)
    assert loop.setpoint_n == setpoint
    for index, sample in enumerate(samples):
        assert sample.error_n == pytest.approx(
            setpoint - sample.measured_n, rel=0, abs=0
        ), f"第{index}步的误差不是按声明的设定值算的"
    #: 纯积分环在这个设定值上必须收敛到它，而不是收敛到30。
    assert samples[-1].tension_n == pytest.approx(setpoint, rel=2e-3)
    assert abs(samples[-1].tension_n - 30.0) > 10.0


def test_every_field_the_step_does_not_advance_comes_through_untouched() -> None:
    """重建后**不该变的字段一个都没变**——直接构造把十一个赋值摊在明处，
    每一个都可以写错，这条门一次性钉住那一整类。

    `dt_s`、`measurement_transfer`、`clutch`、`spool`按**对象同一性**判
    （`is`，不是`==`）：一个"重建时顺手复制了一份等值离合器"的实现在`==`下绿，
    而它会让调用方手里的引用与回路里的那个悄悄分家。
    """

    delay_line = ActuationDelayLine.declare(
        declaration=ActuatorDeclaration(
            actuator_id="actuator/tension_clutch",
            kind="magnetic_particle_clutch",
            channels=(
                CommandChannel(
                    channel_id="command/coil",
                    quantity_id="coil_current_amp",
                    dimension=1,
                    lower=(-3.0,),
                    upper=(3.0,),
                ),
            ),
            delay_s=5.0e-3,
            zero_delay_rationale=None,
        ),
        dt_s=1.0e-3,
        quantization="exact",
        initial_command=ActuationCommand(
            actuator_id="actuator/tension_clutch", values=(0.0,)
        ),
    )
    sensor = TensionSensor(
        full_scale_n=60.0, output_at_full_scale_mv=10.0, adc_bits=12
    )
    loop = _loop(
        setpoint_n=17.5,
        dt_s=1.0e-3,
        delay_line=delay_line,
        sensor=sensor,
        measurement_transfer=0.624,
    )
    stepped, _sample = loop.step(turns_increment=0.05)
    assert stepped.clutch is loop.clutch
    assert stepped.spool is loop.spool
    assert stepped.sensor is loop.sensor
    assert stepped.dt_s == loop.dt_s
    assert stepped.setpoint_n == loop.setpoint_n
    assert stepped.measurement_transfer == loop.measurement_transfer
    #: 该变的那四个也各判一次，否则本门会把"一步什么都没做"判成通过。
    assert stepped.step_index == loop.step_index + 1
    assert stepped.turns == pytest.approx(loop.turns + 0.05, rel=0, abs=0)
    assert stepped.controller is not loop.controller
    assert stepped.delay_line is not loop.delay_line


def test_the_controller_rebuild_keeps_the_gains_and_advances_only_the_state() -> None:
    """`PidController.step_on_error`那一处的同一条：三个增益与限幅原样，
    两个状态量前进。

    限幅那一支单独判：`replace`换直接构造时最容易漏的正是
    "**限幅之后的那个值才是新状态**"。
    """

    controller = PidController(
        proportional=2.0, integral_gain=3.0, derivative=5.0, integral_limit=7.0
    )
    stepped, output = controller.step_on_error(1.5, 0.25)
    assert stepped.proportional == 2.0
    assert stepped.integral_gain == 3.0
    assert stepped.derivative == 5.0
    assert stepped.integral_limit == 7.0
    assert stepped.integral == pytest.approx(1.5 * 0.25, rel=0, abs=0)
    assert stepped.previous_error == 1.5
    #: 第一步没有`previous_error`，微分项必须是零而不是拿零当上一次误差。
    assert output == pytest.approx(2.0 * 1.5 + 3.0 * 0.375, rel=0, abs=0)

    #: 限幅：把积分推爆，新状态必须是**限幅之后**那个数。
    windup = controller
    for _ in range(200):
        windup, _ = windup.step_on_error(1.0e3, 1.0)
    assert windup.integral == 7.0
    negative = windup
    for _ in range(400):
        negative, _ = negative.step_on_error(-1.0e3, 1.0)
    assert negative.integral == -7.0
    assert math.isfinite(negative.previous_error or math.nan)
