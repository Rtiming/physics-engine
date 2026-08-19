#!/usr/bin/env python3
"""逐字节对拍的证据生产器——`drives`的`PidController`与`TensionLoop`每步的**每一个浮点数**。

形制照`tools/parity_transport_samples.py`（决策0083第四节那一份），
用途是同一条：**声称"结果不变"必须附逐字节对拍，不是近似相等**。

## 为什么不比"最终产物"

0083当天有一个活生生的反例：`autodiff.ad_dot`的融合路径与float路径**在每个顶点上
本来就不同**，而判总和的那道门一直绿——**靠的是求和时误差抵消，换台机器就红**。
所以本脚本判的是**还没被求和的那一层**：`TensionSample`的每个字段、每一步，
外加控制器自己的两个状态量（`integral`与`previous_error`），全部`float.hex()`。

**控制器状态必须单独打出来**：`TensionSample`里没有它们，而
`step_on_error`那1,780,613次`replace`改的正是它们。只打样点的对拍
**盖不住这次改动的落点**。

## 用法

    .venv/bin/python tools/parity_drives_samples.py > before.txt
    （改代码）
    .venv/bin/python tools/parity_drives_samples.py > after.txt
    diff before.txt after.txt      # 必须一个字节都不差

工况取`tests/test_drives.py`的**声明输入**（POC-050基型的离合器参数、
60 mm筒径、30 N设定）与`cases/closed_loop_tension_step`的那一组，
不是随手编的数。
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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

#: 与`tests/test_drives.py`同一组声明输入（POC-050基型）。
TORQUE_PER_AMPERE_NMM = 23256.0
RATED_TORQUE_NMM = 50000.0
LAG_S = 0.05
BARREL_RADIUS_MM = 60.0
TAPE_THICKNESS_MM = 0.1
SETPOINT_N = 30.0
GAIN_N_PER_AMPERE = TORQUE_PER_AMPERE_NMM / BARREL_RADIUS_MM

CLUTCH = MagneticParticleClutch(
    torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
    rated_torque_nmm=RATED_TORQUE_NMM,
    lag_s=LAG_S,
)
SPOOL = SpoolTension(
    barrel_radius_mm=BARREL_RADIUS_MM, tape_thickness_mm=TAPE_THICKNESS_MM
)
CHANNEL = CommandChannel(
    channel_id="command/coil",
    quantity_id="coil_current_amp",
    dimension=1,
    lower=(-3.0,),
    upper=(3.0,),
)


def _integral_gain_for(damping_ratio: float) -> float:
    return 1.0 / (4.0 * damping_ratio * damping_ratio * LAG_S * GAIN_N_PER_AMPERE)


def _delay_line(delay_s: float, dt_s: float) -> ActuationDelayLine:
    return ActuationDelayLine.declare(
        declaration=ActuatorDeclaration(
            actuator_id="actuator/tension_clutch",
            kind="magnetic_particle_clutch",
            channels=(CHANNEL,),
            delay_s=delay_s,
            zero_delay_rationale=None,
        ),
        dt_s=dt_s,
        quantization="exact",
        initial_command=ActuationCommand(
            actuator_id="actuator/tension_clutch", values=(0.0,)
        ),
    )


def _show(value: object) -> str:
    return value.hex() if isinstance(value, float) else repr(value)


def _emit_samples(tag: str, samples) -> None:
    names = [f.name for f in fields(samples[0])]
    print(f"# {tag}: {len(samples)} 步 × {len(names)} 字段")
    for index, sample in enumerate(samples):
        for name in names:
            print(f"{tag}\t{index}\t{name}\t{_show(getattr(sample, name))}")


def controller_only() -> None:
    """裸的`PidController.step_on_error`：**改动落点就在这里**，一步一行。

    误差序列取三种形状——阶跃、正弦、把积分推到限幅——
    **限幅那一支必须在**：`replace`换成直接构造时最容易漏的正是"限幅之后的那个值
    才是新状态"，而不限幅的工况看不见这件事。
    """

    import math

    shapes = (
        ("step", lambda k: 5.0),
        ("sine", lambda k: 5.0 * math.sin(0.01 * k)),
        ("windup", lambda k: 1.0e4),
        ("sign_flip", lambda k: 5.0 if (k // 37) % 2 == 0 else -5.0),
    )
    for name, shape in shapes:
        for derivative in (0.0, 1.0e-8):
            controller = PidController(
                proportional=3.0e-6,
                integral_gain=6.0e-5,
                derivative=derivative,
                integral_limit=1.0e6,
            )
            tag = f"pid_{name}_d{derivative:g}"
            print(f"# {tag}: 2000 步 × 3 量")
            for k in range(2000):
                controller, output = controller.step_on_error(shape(k), 1.0e-4)
                print(f"{tag}\t{k}\toutput\t{_show(output)}")
                print(f"{tag}\t{k}\tintegral\t{_show(controller.integral)}")
                print(f"{tag}\t{k}\tprevious_error\t{_show(controller.previous_error)}")


def tension_loop() -> None:
    """`TensionLoop.step`：样点逐字段，外加**控制器状态与回路自身的状态**。

    四个工况覆盖四条分支：理想传感器／带量化传感器、无时延／有时延、
    `turns_increment`为零／非零（卷径在长）。**每一条都改到`step`末尾那次重建**。
    """

    cases = (
        ("ideal", 0.7, 1.0e-3, None, None, 1.0, 0.0),
        ("delayed", 0.7, 1.0e-3, 5.0e-3, None, 1.0, 0.0),
        ("sensed", 0.5, 1.0e-3, None, (60.0, 10.0, 12), 1.0, 0.0),
        ("winding", 0.7, 1.0e-3, 5.0e-3, None, 0.624, 0.05),
    )
    for tag, zeta, dt_s, delay_s, sensor_spec, transfer, turns_increment in cases:
        sensor = None
        if sensor_spec is not None:
            full_scale, millivolts, bits = sensor_spec
            sensor = TensionSensor(
                full_scale_n=full_scale,
                output_at_full_scale_mv=millivolts,
                adc_bits=bits,
            )
        loop = TensionLoop(
            clutch=CLUTCH,
            spool=SPOOL,
            controller=PidController(
                proportional=0.0,
                integral_gain=_integral_gain_for(zeta),
                derivative=0.0,
                integral_limit=1.0e6,
            ),
            setpoint_n=SETPOINT_N,
            dt_s=dt_s,
            delay_line=None if delay_s is None else _delay_line(delay_s, dt_s),
            sensor=sensor,
            measurement_transfer=transfer,
        )
        loop, samples = loop.run(2000, turns_increment=turns_increment)
        _emit_samples(f"loop_{tag}", samples)
        #: 末态也进对拍——样点是步首快照，最后一步之后的状态没有样点盖住它。
        print(f"loop_{tag}\tfinal\ttorque_nmm\t{_show(loop.torque_nmm)}")
        print(f"loop_{tag}\tfinal\tturns\t{_show(loop.turns)}")
        print(f"loop_{tag}\tfinal\tstep_index\t{_show(loop.step_index)}")
        print(f"loop_{tag}\tfinal\tsetpoint_n\t{_show(loop.setpoint_n)}")
        print(f"loop_{tag}\tfinal\tdt_s\t{_show(loop.dt_s)}")
        print(
            f"loop_{tag}\tfinal\tmeasurement_transfer\t{_show(loop.measurement_transfer)}"
        )
        print(f"loop_{tag}\tfinal\ttension_n\t{_show(loop.tension_n)}")
        print(f"loop_{tag}\tfinal\tsensor_is_none\t{loop.sensor is None!r}")
        print(f"loop_{tag}\tfinal\tdelay_line_is_none\t{loop.delay_line is None!r}")
        controller = loop.controller
        print(f"loop_{tag}\tfinal\tctl_integral\t{_show(controller.integral)}")
        print(
            f"loop_{tag}\tfinal\tctl_previous_error\t{_show(controller.previous_error)}"
        )
        print(f"loop_{tag}\tfinal\tctl_proportional\t{_show(controller.proportional)}")
        print(f"loop_{tag}\tfinal\tctl_integral_gain\t{_show(controller.integral_gain)}")
        print(f"loop_{tag}\tfinal\tctl_derivative\t{_show(controller.derivative)}")
        print(
            f"loop_{tag}\tfinal\tctl_integral_limit\t{_show(controller.integral_limit)}"
        )


def closed_loop_case() -> None:
    """`cases/closed_loop_tension_step`那条路：`PidController`在闭环里被调的那1,780,613次。

    这一支与`tools/parity_transport_samples.py`有重叠，**重叠是有意的**：
    那一份是`transport`/`tension_control`那次改动的证据，这一份要证明的是
    **`drives`这次改动没有把那条路上的任何一个数动一位**。
    """

    from physics_engine.tension_control import ClosedTensionLoop
    from physics_engine.transport import FreeSpan, PayoutReel

    span = FreeSpan(
        span_id="span/free", geometric_length_mm=300.0, axial_stiffness_n=2.0e5
    )
    reel = PayoutReel(
        reel_id="reel/payout",
        radius_mm=60.0,
        inertia_kg_mm2=5000.0,
        bearing_damping_nmm_s=50.0,
    )
    for tag, lag_s, proportional, integral_gain, derivative, decimation, steps in (
        ("case_lag1us_dec1", 1.0e-6, 3.0e-6, 6.0e-5, 0.0, 1, 4000),
        ("case_lag5ms_dec4", 5.0e-3, 3.0e-6, 6.0e-5, 1.0e-8, 4, 4000),
    ):
        loop = ClosedTensionLoop.at_steady_state(
            span=span,
            reel=reel,
            clutch=MagneticParticleClutch(
                torque_per_ampere_nmm=23256.0, rated_torque_nmm=50000.0, lag_s=lag_s
            ),
            controller=PidController(
                proportional=proportional,
                integral_gain=integral_gain,
                derivative=derivative,
                integral_limit=1.0e6,
            ),
            capstan=None,
            sensor=None,
            plant_dt_s=1.0e-6,
            control_decimation=decimation,
            brake_torque_nmm=1200.0,
            line_speed_mm_s=20.0,
            delay_line=None,
            forbid_slack=True,
        )
        loop, samples = loop.run(steps, takeup_speed_mm_s=22.0)
        _emit_samples(tag, samples)
        print(f"{tag}\tfinal\theld_current_a\t{_show(loop.held_current_a)}")
        print(f"{tag}\tfinal\tctl_integral\t{_show(loop.controller.integral)}")
        print(f"{tag}\tfinal\tctl_previous_error\t{_show(loop.controller.previous_error)}")


if __name__ == "__main__":
    controller_only()
    tension_loop()
    closed_loop_case()
