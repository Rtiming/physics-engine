#!/usr/bin/env python3
"""逐字节对拍的证据生产器——`transport`/`tension_control`每步样点的**每一个浮点数**。

## 为什么不比"最终产物"

本仓当天刚有一个活生生的反例：`autodiff.ad_dot`的融合路径与float路径**在每个
顶点上本来就不同**，而判总和的那道门一直绿——**靠的是求和时误差抵消，换台机器就红**。
所以本脚本判的是**还没被求和的那一层**：`SpanTransportSample`与
`TensionControlSample`的每个字段、每一步、`float.hex()`。

## 用法

    .venv/bin/python tools/parity_transport_samples.py > before.txt
    （改代码）
    .venv/bin/python tools/parity_transport_samples.py > after.txt
    diff before.txt after.txt      # 必须一个字节都不差

工况取`cases/closed_loop_tension_step`与`cases/free_span_tension_step`的**声明输入**
（不是随手编的数），这样对拍覆盖的正是`accept full`里最重的那条路。
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.drives import MagneticParticleClutch, PidController
from physics_engine.tension_control import ClosedTensionLoop
from physics_engine.transport import FreeSpan, PayoutReel, SpanTransportLoop

#: 与`tests/cases/test_closed_loop_tension_step.py`同一组声明输入。
AXIAL_STIFFNESS_N = 2.0e5
SPAN = FreeSpan(
    span_id="span/free", geometric_length_mm=300.0, axial_stiffness_n=AXIAL_STIFFNESS_N
)
REEL = PayoutReel(
    reel_id="reel/payout",
    radius_mm=60.0,
    inertia_kg_mm2=5000.0,
    bearing_damping_nmm_s=50.0,
)
BRAKE_TORQUE_NMM = 1200.0
LINE_SPEED_MM_S = 20.0
STEP_MM_S = 2.0


def _emit(tag: str, samples) -> None:
    """把每一步每一个字段都打出来。浮点走`float.hex()`，其余走`repr`。"""

    names = [f.name for f in fields(samples[0])]
    print(f"# {tag}: {len(samples)} 步 × {len(names)} 字段")
    for index, sample in enumerate(samples):
        for name in names:
            value = getattr(sample, name)
            shown = value.hex() if isinstance(value, float) else repr(value)
            print(f"{tag}\t{index}\t{name}\t{shown}")


def plant_only() -> None:
    """裸的`SpanTransportLoop`：`replace`那条路每步都走一次。"""

    #: 第二档`dt=1e-5`只走200步：**再多就真的松了**（实测第242步应变转负，
    #: `forbid_slack=True`当场关闭）。对拍要的是同一条路被走到，不是把它走炸。
    for dt_s, steps, excess in ((1.0e-6, 4000, 0.0), (1.0e-5, 200, 0.35)):
        loop = SpanTransportLoop.at_steady_state(
            span=SPAN,
            reel=REEL,
            dt_s=dt_s,
            brake_torque_nmm=BRAKE_TORQUE_NMM,
            line_speed_mm_s=LINE_SPEED_MM_S,
            forbid_slack=True,
        )
        samples = []
        for _ in range(steps):
            loop, sample = loop.step(
                brake_torque_nmm=BRAKE_TORQUE_NMM,
                takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S,
                path_excess_mm=excess,
            )
            samples.append(sample)
        _emit(f"plant_dt{dt_s:g}_excess{excess:g}", samples)
        #: 末态也进对拍——样点是步首快照，最后一步之后的状态没有样点盖住它。
        print(f"plant_dt{dt_s:g}_excess{excess:g}\tfinal\tmaterial_length_mm\t"
              f"{loop.material_length_mm.hex()}")
        print(f"plant_dt{dt_s:g}_excess{excess:g}\tfinal\tangular_velocity_rad_s\t"
              f"{loop.angular_velocity_rad_s.hex()}")


def closed_loop() -> None:
    """闭环：`ClosedTensionLoop.step`每步再套一层`replace`。"""

    cases = (
        #: `lag_s`必须为正（`drives`那边的失败关闭），所以最快那档取1e-6而不是0。
        ("pid_lag1us_dec1", 1.0e-6, 3.0e-6, 6.0e-5, 0.0, 1, 4000),
        ("pid_lag5ms_dec4", 5.0e-3, 3.0e-6, 6.0e-5, 1.0e-8, 4, 4000),
        ("p_only", 1.0e-6, 8.0e-6, 0.0, 0.0, 1, 3000),
    )
    for tag, lag_s, proportional, integral_gain, derivative, decimation, steps in cases:
        loop = ClosedTensionLoop.at_steady_state(
            span=SPAN,
            reel=REEL,
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
            brake_torque_nmm=BRAKE_TORQUE_NMM,
            line_speed_mm_s=LINE_SPEED_MM_S,
            delay_line=None,
            forbid_slack=True,
        )
        loop, samples = loop.run(steps, takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S)
        _emit(tag, samples)
        print(f"{tag}\tfinal\tbrake_torque_nmm\t{loop.brake_torque_nmm.hex()}")
        print(f"{tag}\tfinal\theld_current_a\t{loop.held_current_a.hex()}")
        print(f"{tag}\tfinal\tmaterial_length_mm\t{loop.plant.material_length_mm.hex()}")


if __name__ == "__main__":
    plant_only()
    closed_loop()
