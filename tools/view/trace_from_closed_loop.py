#!/usr/bin/env python3
"""`cases/closed_loop_tension_step`跑一次 → `engine_run_trace`字节。**这一侧不认识rerun。**

## 为什么演示用的是这条案例

0074第5.3节选rerun的第二条理由是**标量时间序列曲线**，原文：
"**没有波形就开发不了张力算法**，那是2026-08-17晚定下的目标甲"。
本案例正是那件事的靶子——张力闭环对收线速度阶跃的响应。

本仓36个案例的`oracle.json`里`arrays`**全部为空**（实测），
所以今天没有任何一条时间序列落过盘。**本脚本是第一条。**

## 这一侧的依赖：`physics_engine` ＋ 标准库，**没有rerun**

这不是省事，是边界的形状。产轨迹的一侧只认识引擎，画轨迹的一侧
（`replay.py`）只认识rerun，中间只有一份JSON。于是：

* 装不上rerun的机器**照样能产轨迹**；
* 没有引擎的机器（比如给别人看的那一台）**照样能画**；
* 而`src/`两边都不认识——0074那三条硬边界在目录里就是这个形状。

## 三维几何是**合成**的，而且轨迹里写着它是合成的

盘、跨段、绞盘在本仓里**没有网格资产**——它们是`transport.PayoutReel`的
半径与惯量，不是三角形。本脚本按声明的半径生成一个圆柱网格，
并在轨迹里把每一块几何标成`"synthetic": true`。

**位姿不是合成的**：盘的转角是对求解器实际吐出的``angular_velocity_rad_s``
逐步积分来的，与画面上转多快是同一个数。
**曲线也不是合成的**：每一条都是``TensionControlSample``的字段原样。

## 抽样与它漏掉的东西

0.2秒 ÷ ``dt = 1e-6`` = 20万步。全量进rerun没意义，所以抽样。
**而抽样会漏掉峰值**，那正是本案例三条判据的第一条。
所以本脚本把未抽样的极值单独算一遍写进``sampling.undecimated_extrema``，
由`replay.py`贴成静态说明——**曲线上看不到的那个峰，文字里写着**。

## 参数逐条与案例同源

`AXIAL_STIFFNESS_N`等九个常数与`cases/closed_loop_tension_step/generate_oracle.py`
及`tests/cases/test_closed_loop_tension_step.py`取同一组数。
**增益从`oracle.json`读，不在这里重算**——重算就等于在查看器里又写了一遍设计，
而那是清单的事。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.drives import MagneticParticleClutch, PidController  # noqa: E402
from physics_engine.tension_control import ClosedTensionLoop  # noqa: E402
from physics_engine.transport import FreeSpan, PayoutReel  # noqa: E402

FACET = "engine_run_trace"
FACET_VERSION = "0.1"

CASE_ID = "case/closed_loop_tension_step"
ORACLE_PATH = ROOT / "cases/closed_loop_tension_step/oracle.json"

#: 逐条与案例同源（见模块docstring末节）。
AXIAL_STIFFNESS_N = 60000.0
GEOMETRIC_LENGTH_MM = 300.0
REEL_RADIUS_MM = 60.0
REEL_INERTIA_KG_MM2 = 5000.0
BEARING_DAMPING_NMM_S = 50.0
BRAKE_TORQUE_NMM = 1200.0
LINE_SPEED_MM_S = 20.0
STEP_MM_S = 2.0
TORQUE_PER_AMPERE_NMM = 23256.0
RATED_TORQUE_NMM = 50000.0

PLANT_DT_S = 1.0e-6
HORIZON_S = 0.2

#: 圆柱网格的周向分段。**这个数只影响画面，不影响任何一个物理量**——
#: 它在轨迹里被标成合成几何，正是为了让这句话可查。
CYLINDER_SEGMENTS = 48
#: 盘宽（画面用）。同样是合成的：`PayoutReel`没有宽度这个字段。
CYLINDER_HALF_WIDTH_MM = 12.0


def cylinder_mesh(
    radius_mm: float, half_width_mm: float, segments: int
) -> tuple[list[list[float]], list[list[int]]]:
    """绕**y轴**的圆柱侧面。轴向的选择与下面的四元数是同一个约定，改一个必须改另一个。

    只出侧面，不封端盖：端盖对"盘转了多少"这件事一个像素的信息都不加，
    而少两圈三角形让顶点数从``4S+2``降到``2S``。
    """

    vertices: list[list[float]] = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        x = radius_mm * math.cos(angle)
        z = radius_mm * math.sin(angle)
        vertices.append([x, -half_width_mm, z])
        vertices.append([x, +half_width_mm, z])
    triangles: list[list[int]] = []
    for index in range(segments):
        a = 2 * index
        b = 2 * index + 1
        c = 2 * ((index + 1) % segments)
        d = c + 1
        triangles.append([a, b, c])
        triangles.append([b, d, c])
    return vertices, triangles


def quaternion_about_y(angle_rad: float) -> list[float]:
    """绕y轴转``angle``的四元数，**xyzw序**（rerun的`quaternion=`吃的就是这个序）。

    序写错不会报错，画面会转成另一个样子——所以名字里带`xyzw`，
    与`tools/model/centerline_csv.py`把基序写进函数名同一条纪律。
    """

    half = 0.5 * angle_rad
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def load_band_gains(band: str) -> dict[str, float]:
    """从案例的`oracle.json`读增益。**不在本文件重算。**"""

    manifest = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    wanted = f"oracle:closed_loop/{band}_step"
    for entry in manifest["oracles"]:
        if entry["id"] == wanted:
            return dict(entry["inputs"])
    available = sorted(
        e["id"].split("/", 1)[1].removesuffix("_step")
        for e in manifest["oracles"]
        if e["id"].startswith("oracle:closed_loop/") and e["id"].endswith("_step")
    )
    raise SystemExit(f"`oracle.json`里没有`{wanted}`；有的是：{available}")


def simulate(band: str) -> tuple[tuple[Any, ...], float]:
    """跑一次，返回全量样本与设定值。**推进全部交给内核，本文件不算物理。**"""

    gains = load_band_gains(band)
    loop = ClosedTensionLoop.at_steady_state(
        span=FreeSpan(
            span_id="span/free",
            geometric_length_mm=GEOMETRIC_LENGTH_MM,
            axial_stiffness_n=AXIAL_STIFFNESS_N,
        ),
        reel=PayoutReel(
            reel_id="reel/payout",
            radius_mm=REEL_RADIUS_MM,
            inertia_kg_mm2=REEL_INERTIA_KG_MM2,
            bearing_damping_nmm_s=BEARING_DAMPING_NMM_S,
        ),
        clutch=MagneticParticleClutch(
            torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
            rated_torque_nmm=RATED_TORQUE_NMM,
            lag_s=gains["clutch_lag_s"],
        ),
        controller=PidController(
            proportional=gains["proportional_a_per_n"],
            integral_gain=gains["integral_gain_a_per_n_s"],
            derivative=gains["derivative_a_s_per_n"],
            integral_limit=1.0e6,
        ),
        capstan=None,
        sensor=None,
        plant_dt_s=PLANT_DT_S,
        control_decimation=1,
        brake_torque_nmm=BRAKE_TORQUE_NMM,
        line_speed_mm_s=LINE_SPEED_MM_S,
        delay_line=None,
        forbid_slack=True,
    )
    steps = int(round(HORIZON_S / PLANT_DT_S))
    _, samples = loop.run(steps, takeup_speed_mm_s=LINE_SPEED_MM_S + STEP_MM_S)
    return samples, loop.setpoint_n


def build_trace(band: str, stride: int) -> dict[str, Any]:
    samples, setpoint_n = simulate(band)

    #: **未抽样**的极值先算——抽样之后就算不出来了，而那正是这个字段存在的理由。
    deviations = [s.tension_n - setpoint_n for s in samples]
    peak_index = max(range(len(deviations)), key=deviations.__getitem__)
    trough_index = min(range(len(deviations)), key=deviations.__getitem__)
    undecimated = {
        "tension_deviation_n": {
            "peak": deviations[peak_index],
            "peak_time_s": samples[peak_index].time_s,
        },
        "tension_deviation_n_min": {
            "peak": deviations[trough_index],
            "peak_time_s": samples[trough_index].time_s,
        },
    }

    kept = samples[::stride]

    #: 转角对**求解器吐出的**角速度逐步积分（梯形），不是用``v/R``反推——
    #: 反推等于把画面接到设定值上，那样画面永远"对"，而它正是我们要看的东西。
    angle_payout = 0.0
    angles: list[float] = []
    previous = samples[0].angular_velocity_rad_s
    for index, sample in enumerate(samples):
        if index:
            angle_payout += 0.5 * (previous + sample.angular_velocity_rad_s) * PLANT_DT_S
            previous = sample.angular_velocity_rad_s
        if index % stride == 0:
            angles.append(angle_payout)

    times = [s.time_s for s in kept]
    takeup_omega = (LINE_SPEED_MM_S + STEP_MM_S) / REEL_RADIUS_MM
    vertices, triangles = cylinder_mesh(REEL_RADIUS_MM, CYLINDER_HALF_WIDTH_MM,
                                        CYLINDER_SEGMENTS)
    span_x = GEOMETRIC_LENGTH_MM
    tangent = [0.0, 0.0, REEL_RADIUS_MM]
    laydown = [span_x, 0.0, REEL_RADIUS_MM]

    scalar_series = {
        "tension_n": [s.tension_n for s in kept],
        "tension_deviation_n": [s.tension_n - setpoint_n for s in kept],
        "current_a": [s.current_a for s in kept],
        "angular_velocity_rad_s": [s.angular_velocity_rad_s for s in kept],
        "payout_speed_mm_s": [s.payout_speed_mm_s for s in kept],
        "takeup_speed_mm_s": [s.takeup_speed_mm_s for s in kept],
    }
    scalar_units = {
        "tension_n": "N",
        "tension_deviation_n": "N",
        "current_a": "A",
        "angular_velocity_rad_s": "rad/s",
        "payout_speed_mm_s": "mm/s",
        "takeup_speed_mm_s": "mm/s",
    }

    return {
        "facet": FACET,
        "facet_version": FACET_VERSION,
        "run_id": f"engine_run_trace/closed_loop_tension_step/{band}",
        "producer": {
            "case_id": CASE_ID,
            "path_relative": "tools/view/trace_from_closed_loop.py",
            "band": band,
        },
        #: **单位是声明，不是约定**——`replay.py`缺了它当场抛。
        "units": {"length": "mm", "time": "s", "force": "N", "angle": "rad"},
        "timeline": {"name": "sim_time", "unit": "s", "times": times},
        "sampling": {
            "source_step_count": len(samples),
            "source_dt_s": PLANT_DT_S,
            "stride": stride,
            "kept": len(kept),
            "undecimated_extrema": undecimated,
        },
        "notes": [
            "## 这条轨迹里什么是真的",
            "- **曲线**：`TensionControlSample`的字段原样，没有插值也没有平滑；",
            "- **盘的转角**：对求解器吐出的``angular_velocity_rad_s``梯形积分；",
            "- **圆柱网格与盘宽**：**合成的**。`PayoutReel`只有半径与惯量，没有三角形；",
            f"- **设定值**：``T* = {setpoint_n!r}`` N，闭式稳态（``M/R + c·v/R²``）。",
        ],
        "geometry": [
            {
                "entity_path": "/rig/payout/synthetic_reel",
                "kind": "mesh",
                #: 必填、无默认。理由见`replay.py`第一节。
                "synthetic": True,
                "vertex_positions": vertices,
                "triangle_indices": triangles,
            },
            {
                "entity_path": "/rig/takeup/synthetic_reel",
                "kind": "mesh",
                "synthetic": True,
                "vertex_positions": vertices,
                "triangle_indices": triangles,
            },
        ],
        "poses": [
            {
                "entity_path": "/rig/payout",
                "translations": [[0.0, 0.0, 0.0]] * len(kept),
                "quaternions_xyzw": [quaternion_about_y(-a) for a in angles],
            },
            {
                "entity_path": "/rig/takeup",
                "translations": [[span_x, 0.0, 0.0]] * len(kept),
                "quaternions_xyzw": [quaternion_about_y(-takeup_omega * t) for t in times],
            },
        ],
        "points": [
            {
                "entity_path": "/rig/contacts",
                "frames": [[tangent, laydown] for _ in kept],
                "radii": [3.0, 3.0],
                "colors": [[255, 96, 0], [0, 160, 255]],
            }
        ],
        "scalars": [
            {"entity_path": f"/signals/{name}", "unit": scalar_units[name], "values": values}
            for name, values in sorted(scalar_series.items())
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="closed_loop_tension_step跑一次 → engine_run_trace（JSON）。不认识rerun。"
    )
    parser.add_argument("--band", default="nominal_band",
                        help="oracle.json里的档位，如open_loop/fast_band/nominal_band")
    parser.add_argument("--stride", type=int, default=200,
                        help="抽样步长。**它会漏峰**，未抽样极值另记（见模块docstring）")
    parser.add_argument("--out", type=Path, required=True, help="产出的轨迹JSON")
    arguments = parser.parse_args(argv)
    if arguments.stride < 1:
        print("stride必须是正整数", file=sys.stderr)
        return 1
    trace = build_trace(arguments.band, arguments.stride)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(trace, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    sampling = trace["sampling"]
    print(json.dumps({
        "out": str(arguments.out),
        "bytes": arguments.out.stat().st_size,
        "source_step_count": sampling["source_step_count"],
        "kept": sampling["kept"],
        "undecimated_extrema": sampling["undecimated_extrema"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
