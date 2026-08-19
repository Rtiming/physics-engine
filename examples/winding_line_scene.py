#!/usr/bin/env python3
"""场景⑥：放线盘→两个导向轮→张力轮→收线盘的**装配**（plans/04第九节）。

    .venv/bin/python examples/winding_line_scene.py

这个脚本证明的是一件事，也只有一件事：**那条链路今天装得出来**。
plans/04第九节把它的六个要素逐个对到引擎里，其中一格写着

    各轮上的偏移/未对中 | 多体场景装配 | ❌ `Scene`装配容器没有

本脚本是那一格的对账：13个体、10对显式声明的接触、4对允许重叠不报的对、
两个运动源（放线盘转动 / 收线盘转动+横动）、一个驱动器（张力轮），
`finalize()`一次性校验通过。

## 装出来了什么，没装出来什么（**先说没装出来的**）

* **接触一个都没算。** `declare_contact_between`记的是"这一对的接触要算"，
  算不算得出来是接触内核的事，而**本脚本不产任何接触事件，也不调碰撞查询**。

  > **2026-08-18订正一句已经不成立的话。** 本段原文写的是"引擎今天没有接触内核
  > （plans/04第九节的第2条墙，卡在决策0033的状态布局）"——**那句话从0050起就不成立了**。
  > 今天`contact/`是一个六文件的子包（罚法向六族、库仑return-map含摩擦椭圆、
  > 阻尼、准静态步进器、锚点布局），另有`contact_pipeline`（动态检测响应）与
  > `contact_dynamics`（力矩装配），0085还把有符号距离场接了进来。
  > 0033那条"状态布局装不下锚点"的墙**也早已由0050拆掉**（按声明的对分槽，布局是定长的）。
  >
  > **本脚本不算接触仍然是对的，但理由变了**：不是引擎不能算，是**装配层只声明拓扑**——
  > 这正是下一句在说的事。**一句正确的结论配一个过期的理由，会把下一个人送上一条不存在的路**
  > （与plans/07里0071那条订正同形）；
* **带材不动。** 四段带材是静态体：带材的运动是被求解出来的，不是被声明的。
  盘在转、盘在横动，带材停着——这不是bug，这是"装配层只声明拓扑"的样子；
* **导向轮与张力轮不转。** 它们是从动轮，转速由带材摩擦决定，同样是物理不是声明。
  张力轮身上只有一份`ActuatorDeclaration`（命令空间+时延），**没有`apply`**
  ——决策0038的边界原样保持；
* **法兰的轴向尺寸**在`shapes.FiniteCylinder`里仍然缺（spec/11第二之二节的已知缺口，
  决策0034维持失败关闭）。本脚本按`modelgen`的既有纪律绕开它：一个带法兰的盘由
  **筒一件+法兰盘两件**三个既有原语表达，词汇一个字没动。
  蹭边真要算那天，0034第四节自动重开。

## 几何与位置怎么来的

站位沿世界x轴排开，五个回转体的轴**一律平行于世界z**；带材沿x走，
宽度方向沿z（与回转轴平行），厚度方向沿y。这样自转是绕z、横动是沿z，
两样都不需要复合位姿——**这是本例站位挑出来的便利，不是一条通则**：
装配层没有运动学树，一个盘的筒与两片法兰是三个独立的体、各带各的运动源
（见`SceneAssembly`类文档"它不做什么"）。

四段带材落在相邻两个回转体的**外公切线**上（闭式：切线长`√(d²−Δr²)`，
倾角`arcsin(Δr/d)`）。这只是一个**静态初始摆放**，不是解出来的带材路径。

## 蹭边为什么会发生（这不是意外，是场景要的）

收线盘法兰内侧到中面10mm、带材半宽6mm，自由行程只有±4mm；
而横动幅度声明成±8mm。8+6=14>10——**带材必然压到法兰**。
用户原话是"给这个盘加偏移，模拟实际蹭边的情况"，所以
`带材↔收线盘法兰`那两对接触是被**故意**声明出来的。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physics_engine.actuators import ActuatorDeclaration, CommandChannel  # noqa: E402
from physics_engine.modelgen import generate_roller, generate_spool  # noqa: E402
from physics_engine.motion import AnalyticPose, Pose  # noqa: E402
from physics_engine.scene import FinalizedScene, SceneAssembly  # noqa: E402
from physics_engine.shapes import (  # noqa: E402
    CollisionShape,
    PosedBody,
    RoundedBox,
    SimBody,
)

# --------------------------------------------------------------------------
# 声明（全部集中在这里，脚本正文里不出现第二个字面量）
# --------------------------------------------------------------------------

SCENE_ID = "scene/winding_line"

#: 带盘：特征长度100mm → 筒半径50、筒全宽20、法兰外径75、单片法兰厚6。
SPOOL_LENGTH_MM = 100.0
SPOOL_BARREL_RADIUS_RATIO = 0.5
SPOOL_BARREL_WIDTH_RATIO = 0.2
SPOOL_FLANGE_OUTER_RADIUS_RATIO = 0.75
SPOOL_FLANGE_WIDTH_RATIO = 0.06
#: 放线盘出场时是满的：12层，每层0.5mm → 有效筒径56mm（`WindingSurface`形制）。
PAYOFF_WOUND_LAYERS = 12
PAYOFF_LAYER_THICKNESS_RATIO = 0.005

#: 导轮：特征长度90mm → 半径45、面宽18（examples/collision_preview_cell里的R45同款）。
ROLLER_LENGTH_MM = 90.0
ROLLER_RADIUS_RATIO = 0.5
ROLLER_FACE_WIDTH_RATIO = 0.2

#: 带材：12mm宽、0.15mm厚。截面是矩形，所以用`RoundedBox`（圆角为零）而不是胶囊——
#: 胶囊是圆截面，表达不了"带"（plans/04零之一：中心线杆表示不了带材的宽与厚）。
STRIP_WIDTH_MM = 12.0
STRIP_THICKNESS_MM = 0.15

#: 站位：x坐标 + 带材实际骑上去的那个半径。
PAYOFF_X_MM = 0.0
GUIDE_1_X_MM = 400.0
GUIDE_2_X_MM = 800.0
TENSION_X_MM = 1200.0
TAKEUP_X_MM = 1600.0

#: 导向轮2的轴向未对中（静态位姿，无运动源）——"各轮上的偏移"那一格。
GUIDE_2_AXIAL_MISALIGNMENT_MM = 0.8

#: 运动学声明。转速取负号表示放线方向与收线相反。
HORIZON_S = 20.0
PAYOFF_SPIN_RAD_PER_S = -1.2
TAKEUP_SPIN_RAD_PER_S = 1.5
TAKEUP_TRAVERSE_AMPLITUDE_MM = 8.0
TAKEUP_TRAVERSE_PERIOD_S = 8.0
REPLAY_PROBE_TIMES_S = (0.0, 5.0, HORIZON_S)

#: 展示位姿的时刻。取5秒：横动正弦在此既不在零点也不在峰值（偏移−5.657mm），
#: 转角也不在整周附近——一眼就看得出两个盘确实在动。
REPORT_TIME_S = 5.0

#: 张力轮：磁粉离合器，20ms命令时延（`REALIZABLE_ACTUATOR_KINDS`里的真机种类）。
CLUTCH_DELAY_S = 0.02
CLUTCH_TORQUE_LIMIT_NMM = 4000.0


# --------------------------------------------------------------------------
# 装配辅助（都是本例自己的事，不进库——装配层不替调用方摆几何）
# --------------------------------------------------------------------------


def _zero(value: float) -> float:
    """把`-0.0`归一成`0.0`。理由与`modelgen._clean`逐字相同：两者`==`相等，
    格式化出来却是两串不同的字符，而本例的输出要逐字节可比。"""

    return 0.0 if value == 0.0 else value


def _spin_about_z(angle_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * angle_rad
    return (0.0, 0.0, _zero(math.sin(half)), _zero(math.cos(half)))


def _turntable(
    source_id: str, x_mm: float, z_mm: float, spin_rad_per_s: float
) -> AnalyticPose:
    """原地自转的回转体。挂载位置写进运动源本身——位姿只能有一个来源。"""

    def pose_fn(t_s: float) -> Pose:
        return Pose(
            translation_mm=(x_mm, 0.0, z_mm),
            rotation_xyzw=_spin_about_z(spin_rad_per_s * t_s),
        )

    return AnalyticPose(
        source_id=source_id,
        pose_fn=pose_fn,
        declared_horizon_s=HORIZON_S,
        extrapolation="reject",
        replayable=True,
        replay_probe_times_s=REPLAY_PROBE_TIMES_S,
    )


def _traversing_turntable(
    source_id: str, x_mm: float, z_mm: float, spin_rad_per_s: float
) -> AnalyticPose:
    """自转 + 沿自身轴（世界z）横动的回转体——收线盘的排线运动。

    横动取正弦。真实排线是等速三角波，正弦只是一条**声明的**光滑近似:
    它是纯函数、可重放，够本例用；哪天要真排线，换一条时间线即可，
    装配层一个字不用改。
    """

    omega = 2.0 * math.pi / TAKEUP_TRAVERSE_PERIOD_S

    def pose_fn(t_s: float) -> Pose:
        offset = TAKEUP_TRAVERSE_AMPLITUDE_MM * math.sin(omega * t_s)
        return Pose(
            translation_mm=(x_mm, 0.0, _zero(z_mm + offset)),
            rotation_xyzw=_spin_about_z(spin_rad_per_s * t_s),
        )

    return AnalyticPose(
        source_id=source_id,
        pose_fn=pose_fn,
        declared_horizon_s=HORIZON_S,
        extrapolation="reject",
        replayable=True,
        replay_probe_times_s=REPLAY_PROBE_TIMES_S,
    )


def _declare_generated_parts(
    assembly: SceneAssembly,
    prefix: str,
    parts,
    *,
    x_mm: float,
    spin_rad_per_s: float | None,
    traversing: bool = False,
    static_z_mm: float = 0.0,
) -> tuple[str, ...]:
    """把一次`modelgen`调用产出的若干件登记成若干个体。

    件的局部偏移沿z、自转绕z，所以偏移**不受自转影响**，可以直接加进各件的
    挂载位置。这是本例站位挑出来的便利：装配层没有运动学树，换一个不绕自身
    偏移轴转的装配，这里就得自己算复合位姿了。
    """

    body_ids = []
    for part in parts:
        body_id = f"body/{prefix}_{part.part_id}"
        z_mm = static_z_mm + part.offset_mm[2]
        moving = spin_rad_per_s is not None
        assembly.declare_body(
            PosedBody(
                body=SimBody(
                    body_id=body_id,
                    collision=CollisionShape(shape=part.shape, direction="fitted"),
                ),
                translation_mm=(0.0, 0.0, 0.0) if moving else (x_mm, 0.0, z_mm),
            )
        )
        if moving:
            source_id = f"motion/{prefix}_{part.part_id}"
            factory = _traversing_turntable if traversing else _turntable
            assembly.declare_motion_source(
                body_id, factory(source_id, x_mm, z_mm, spin_rad_per_s)
            )
        body_ids.append(body_id)
    return tuple(body_ids)


def _declare_strip_span(
    assembly: SceneAssembly,
    index: int,
    upstream: tuple[float, float],
    downstream: tuple[float, float],
) -> str:
    """相邻两个回转体之间的一段带材，摆在两圆的**外公切线**上。

    `upstream`/`downstream`各是`(中心x, 带材骑上去的半径)`。闭式：
    切线长`√(d² − Δr²)`、倾角`arcsin(Δr/d)`、切点在中心加半径乘法线方向处。
    """

    (x_up, r_up), (x_down, r_down) = upstream, downstream
    distance = x_down - x_up
    delta_r = r_down - r_up
    tilt_rad = math.asin(delta_r / distance)
    span_mm = math.sqrt(distance * distance - delta_r * delta_r)
    normal = (-math.sin(tilt_rad), math.cos(tilt_rad))
    touch_up = (x_up + r_up * normal[0], r_up * normal[1])
    touch_down = (x_down + r_down * normal[0], r_down * normal[1])

    body_id = f"body/strip_span_{index}"
    assembly.declare_body(
        PosedBody(
            body=SimBody(
                body_id=body_id,
                collision=CollisionShape(
                    shape=RoundedBox(
                        half_extents_mm=(
                            0.5 * span_mm,
                            0.5 * STRIP_THICKNESS_MM,
                            0.5 * STRIP_WIDTH_MM,
                        ),
                        fillet_radius_mm=0.0,
                    ),
                    direction="fitted",
                ),
            ),
            translation_mm=(
                _zero(0.5 * (touch_up[0] + touch_down[0])),
                _zero(0.5 * (touch_up[1] + touch_down[1])),
                0.0,
            ),
            rotation_xyzw=_spin_about_z(tilt_rad),
        )
    )
    return body_id


# --------------------------------------------------------------------------
# 装配本体
# --------------------------------------------------------------------------


def build_winding_line() -> FinalizedScene:
    """把plans/04第九节那条链路装出来并`finalize()`。"""

    assembly = SceneAssembly(SCENE_ID)

    payoff_parts = generate_spool(
        characteristic_length_mm=SPOOL_LENGTH_MM,
        barrel_radius_ratio=SPOOL_BARREL_RADIUS_RATIO,
        barrel_width_ratio=SPOOL_BARREL_WIDTH_RATIO,
        flange_outer_radius_ratio=SPOOL_FLANGE_OUTER_RADIUS_RATIO,
        flange_width_ratio=SPOOL_FLANGE_WIDTH_RATIO,
        wound_layers=PAYOFF_WOUND_LAYERS,
        layer_thickness_ratio=PAYOFF_LAYER_THICKNESS_RATIO,
    )
    takeup_parts = generate_spool(
        characteristic_length_mm=SPOOL_LENGTH_MM,
        barrel_radius_ratio=SPOOL_BARREL_RADIUS_RATIO,
        barrel_width_ratio=SPOOL_BARREL_WIDTH_RATIO,
        flange_outer_radius_ratio=SPOOL_FLANGE_OUTER_RADIUS_RATIO,
        flange_width_ratio=SPOOL_FLANGE_WIDTH_RATIO,
    )
    roller_parts = generate_roller(
        characteristic_length_mm=ROLLER_LENGTH_MM,
        radius_ratio=ROLLER_RADIUS_RATIO,
        face_width_ratio=ROLLER_FACE_WIDTH_RATIO,
    )
    payoff_radius_mm = payoff_parts[0].shape.shape.radius_mm
    takeup_radius_mm = takeup_parts[0].shape.shape.radius_mm
    roller_radius_mm = roller_parts[0].shape.shape.radius_mm

    payoff = _declare_generated_parts(
        assembly,
        "payoff_spool",
        payoff_parts,
        x_mm=PAYOFF_X_MM,
        spin_rad_per_s=PAYOFF_SPIN_RAD_PER_S,
    )
    guide_1 = _declare_generated_parts(
        assembly, "guide_roller_1", roller_parts, x_mm=GUIDE_1_X_MM, spin_rad_per_s=None
    )
    guide_2 = _declare_generated_parts(
        assembly,
        "guide_roller_2",
        roller_parts,
        x_mm=GUIDE_2_X_MM,
        spin_rad_per_s=None,
        static_z_mm=GUIDE_2_AXIAL_MISALIGNMENT_MM,
    )
    tension = _declare_generated_parts(
        assembly, "tension_roller", roller_parts, x_mm=TENSION_X_MM, spin_rad_per_s=None
    )
    takeup = _declare_generated_parts(
        assembly,
        "takeup_spool",
        takeup_parts,
        x_mm=TAKEUP_X_MM,
        spin_rad_per_s=TAKEUP_SPIN_RAD_PER_S,
        traversing=True,
    )

    assembly.declare_actuator(
        tension[0],
        ActuatorDeclaration(
            actuator_id="actuator/tension_roller",
            kind="magnetic_particle_clutch",
            channels=(
                CommandChannel(
                    channel_id="command/clutch_torque",
                    quantity_id="clutch_torque_nmm",
                    dimension=1,
                    lower=(0.0,),
                    upper=(CLUTCH_TORQUE_LIMIT_NMM,),
                ),
            ),
            delay_s=CLUTCH_DELAY_S,
            zero_delay_rationale=None,
        ),
    )

    # 带材：四段，每段落在相邻两个回转体的外公切线上。
    stations = (
        (PAYOFF_X_MM, payoff_radius_mm),
        (GUIDE_1_X_MM, roller_radius_mm),
        (GUIDE_2_X_MM, roller_radius_mm),
        (TENSION_X_MM, roller_radius_mm),
        (TAKEUP_X_MM, takeup_radius_mm),
    )
    contact_ends = (payoff[0], guide_1[0], guide_2[0], tension[0], takeup[0])
    spans = tuple(
        _declare_strip_span(assembly, index, stations[index], stations[index + 1])
        for index in range(len(stations) - 1)
    )

    # 接触按对声明：每一段带材与它上下游那两个体。
    for index, span in enumerate(spans):
        assembly.declare_contact_between(span, contact_ends[index])
        assembly.declare_contact_between(span, contact_ends[index + 1])
    # 蹭边：最后一段带材与收线盘的两片法兰（横动幅度超出自由行程，见模块文档）。
    for flange in takeup[1:]:
        assembly.declare_contact_between(spans[-1], flange)

    # 允许重叠不报：盘的筒与自己的两片法兰按构造就贴在一起。
    for group in (payoff, takeup):
        for flange in group[1:]:
            assembly.declare_allowed_pair(group[0], flange)

    return assembly.finalize()


# --------------------------------------------------------------------------
# 打印
# --------------------------------------------------------------------------


def _format_vector(values) -> str:
    return "(" + ", ".join(f"{_zero(value):+.3f}" for value in values) + ")"


def render(scene: FinalizedScene) -> str:
    lines = [
        "场景⑥ 放线—导向—张力—收线链路的装配（plans/04第九节）",
        f"scene_id: {scene.scene_id}",
        "",
        f"体 {len(scene.bodies)}个（声明次序）：",
    ]
    for index, body in enumerate(scene.bodies, start=1):
        source = body.motion_source
        source_id = getattr(source, "source_id", "—") if source is not None else "—"
        actuator = body.actuator.actuator_id if body.actuator is not None else "—"
        lines.append(
            f"  {index:2d} {body.body_id:<34} 运动源 {source_id:<32} 驱动器 {actuator}"
        )

    lines += ["", f"接触对 {len(scene.contact_pairs)}对（次序即声明次序）："]
    for index, pair in enumerate(scene.contact_pairs, start=1):
        lines.append(f"  {index:2d} {pair.body_a:<24} ↔ {pair.body_b}")

    allowed = sorted(sorted(pair) for pair in scene.allowed_pairs)
    lines += ["", f"允许重叠不报的对 {len(allowed)}对（allowed_pairs，排序输出）："]
    for index, (body_a, body_b) in enumerate(allowed, start=1):
        lines.append(f"  {index:2d} {body_a:<24} ↔ {body_b}")

    lines += ["", f"t = {REPORT_TIME_S}s 各体位姿（平移mm，旋转xyzw）："]
    for posed in scene.posed_bodies_at(REPORT_TIME_S):
        lines.append(
            f"  {posed.body.body_id:<34} "
            f"{_format_vector(posed.translation_mm)}  "
            f"{_format_vector(posed.rotation_xyzw)}"
        )

    lines += [
        "",
        "结论：finalize()通过，装配成立。",
        "**本脚本不算任何接触**——接触对是被声明的意图，不是被算出的事件。",
        "**理由是装配层只声明拓扑，不是引擎不能算**：contact/子包、contact_pipeline、",
        "contact_dynamics与0085的距离场今天都在（2026-08-18订正，原文写的",
        "「引擎今天没有接触内核」从决策0050起就不成立了）。",
    ]
    return "\n".join(lines)


def main() -> int:
    print(render(build_winding_line()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
