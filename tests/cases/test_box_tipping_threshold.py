"""`cases/box_tipping_threshold`的门——plans/16的M4：**翻倒**。

本文件判四件事，一件都不能少：

1. **两侧**：`tanθ < w/h`时撑住（位姿漂移给上界）、`tanθ > w/h`时**真的翻过去**
   （承载点从底面整体换到下坡侧面）；
2. **阈值与闭式的偏差**：二分夹住临界角，给夹取宽度、给对刚体闭式的偏差、
   并证明那个偏差是**罚柔度**而不是实现缺陷（它随`1/k`趋零）；
3. **不许拿"积分器炸了"冒充翻倒**——形制抄
   `test_three_sphere_pyramid_rotational::test_the_collapse_never_leans_on_non_convergence`；
4. **退化逐位**：不接触时`integrate_free_flight`产物`float.hex()`逐位不变。

**为什么第3条在这里格外要紧**：翻倒的可观测量是"姿态角变得很大"，
而一个发散的显式积分器**也会**让姿态角变得很大。两者必须被分开判，
否则本文件最响亮的那条判据可以被一次数值发散冒充。
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import pytest

from physics_engine.contact_dynamics import (
    box_corner_points_mm,
    support_points_plane_callbacks,
    support_points_plane_contact,
)
from physics_engine.oracles import load_manifest
from physics_engine.rigidbody import (
    QUATERNION_NORM_STEP_ABS_TOL,
    RK4_BODY,
    RigidBodyError,
    RigidBodyInertia,
    attitude_matrix,
    integrate_free_flight,
    make_state,
)
from physics_engine.shapes import RoundedBox

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "cases/box_tipping_threshold/oracle.json"

#: 本案例整体是本机批级：三组积分合计数千步，实测约5秒。
pytestmark = pytest.mark.batch


@pytest.fixture(scope="module")
def oracles():
    manifest = load_manifest(MANIFEST)
    return {entry.id: entry for entry in manifest.oracles}


# ---------------------------------------------------------------------------
# 装置：斜面标架、初始位形、行进
# ---------------------------------------------------------------------------


def _frame(theta: float):
    """斜面的两个方向。**法向与`rolling_ball_incline`同一约定**——
    `n̂ = (sinθ, 0, cosθ)`、下坡`t̂ = (cosθ, 0, −sinθ)`，于是横坡向恰是`ŷ`。

    两条案例共用一个约定不是巧合：约定不同的话"下坡"这个词在两页里指不同方向，
    而没有任何门看得见（spec/12第2.2节那条"次序是形制的一部分"的同族）。
    """

    return (math.sin(theta), 0.0, math.cos(theta)), (math.cos(theta), 0.0, -math.sin(theta))


def _inertia(inputs) -> RigidBodyInertia:
    """惯量走`geometry.mass_properties`，**本文件一行都不重推**。"""

    half = tuple(float(value) for value in inputs["half_extents_mm"])
    return RigidBodyInertia.from_shape(
        RoundedBox(half_extents_mm=half, fillet_radius_mm=0.0),
        mass_kg=float(inputs["mass_kg"]),
    )


def _weight_n(inputs) -> float:
    return float(inputs["mass_kg"]) * float(inputs["gravity_mm_per_s2"]) / 1000.0


def _tilt(vector, theta: float) -> float:
    """底面法向相对坡面法向的**有符号**倾角，正=向下坡倒。

    读的是`R(q)`的第三列（体系`ẑ`到世界系），不是四元数分量——
    分量与"倒向哪边"之间隔着一层约定，而判据不该跨那一层。
    """

    normal, downhill = _frame(theta)
    rows = attitude_matrix(vector[9:13])
    base_normal = (rows[0][2], rows[1][2], rows[2][2])
    return math.atan2(
        sum(base_normal[i] * downhill[i] for i in range(3)),
        sum(base_normal[i] * normal[i] for i in range(3)),
    )


def _settled_start(theta, inputs, *, stiffness, base_points, creep):
    """**闭式静态位形**：逐角穿透`δ_i = F_i/k`、体倾`sin β = (δ_dn − δ_up)/(2w)`、
    质心高`C = h·cos β − (δ_up + δ_dn)/2`。

    起手就放在平衡位形上，是为了让"撑住"这条判据判的是**平衡的稳定性**
    而不是一次瞬态有没有被阻尼掉。`creep`是稳态蠕滑速度（速度型摩擦不是零）。

    **这三条几何式子是解出来的不是凑的**：两个角的间隙方程
    `C ± w sin β − h cos β = −δ`相减得`β`、相加得`C`。
    """

    half_w, _, half_h = (float(value) for value in inputs["half_extents_mm"])
    weight = _weight_n(inputs)
    mean = weight * math.cos(theta) / base_points
    half = weight * half_h * math.sin(theta) / (base_points * half_w)
    delta_up, delta_down = (mean - half) / stiffness, (mean + half) / stiffness
    beta = math.asin((delta_down - delta_up) / (2.0 * half_w))
    height = half_h * math.cos(beta) - 0.5 * (delta_up + delta_down)
    normal, downhill = _frame(theta)
    alpha = theta + beta
    return make_state(
        position_mm=tuple(height * c for c in normal),
        velocity_mm_per_s=tuple(creep * c for c in downhill),
        attitude_xyzw=(0.0, math.sin(0.5 * alpha), 0.0, math.cos(0.5 * alpha)),
    )


def _callbacks(theta, inputs, *, stiffness, points):
    normal, _ = _frame(theta)
    return support_points_plane_callbacks(
        support_points_body_mm=points,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=normal,
        normal_stiffness_n_per_mm=stiffness,
        tangential_stiffness_n_per_mm=float(inputs["tangential_stiffness"]),
        friction_coefficient=float(inputs["friction_coefficient"]),
        gravity_world_n=(0.0, 0.0, -_weight_n(inputs)),
        normal_damping_n_s_per_mm=float(inputs["normal_damping"]),
    )


def _response(vector, theta, inputs, *, stiffness, points):
    normal, _ = _frame(theta)
    return support_points_plane_contact(
        vector,
        support_points_body_mm=points,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=normal,
        normal_stiffness_n_per_mm=stiffness,
        tangential_stiffness_n_per_mm=float(inputs["tangential_stiffness"]),
        friction_coefficient=float(inputs["friction_coefficient"]),
        normal_damping_n_s_per_mm=float(inputs["normal_damping"]),
    )


def _loaded_indices(response) -> list[int]:
    return [index for index, point in enumerate(response.points) if point.in_contact]


def _pressure_centre_mm(response, theta: float) -> float:
    """压心沿坡向相对质心投影的位置`x_N = Σ x_i F_i / Σ F_i`。

    `x_i`取**世界系杆臂在下坡方向上的投影**、`F_i`取法向力的大小——
    两者都从`SupportSetResponse`逐点读出来，**没有一个是声明的**。
    """

    _, downhill = _frame(theta)
    total = 0.0
    moment = 0.0
    for point in response.points:
        if not point.in_contact:
            continue
        magnitude = math.sqrt(sum(value * value for value in point.force_world_n))
        normal_magnitude = abs(
            sum(point.force_world_n[i] * _frame(theta)[0][i] for i in range(3))
        )
        assert magnitude >= normal_magnitude  # 切向分量非负，纯粹是自洽检查
        arm = sum(point.lever_world_mm[i] * downhill[i] for i in range(3))
        total += normal_magnitude
        moment += arm * normal_magnitude
    return moment / total


# ---------------------------------------------------------------------------
# 一、稳定侧：`tanθ < w/h`
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stable_run(oracles):
    entry = oracles["oracle:box_tipping/stable_side"]
    inputs = entry.inputs
    theta = math.radians(float(inputs["incline_deg"]))
    stiffness = float(inputs["normal_stiffness_n_per_mm"])
    points = box_corner_points_mm(tuple(inputs["half_extents_mm"]))[:4]
    weight = _weight_n(inputs)
    half_w = float(inputs["half_extents_mm"][0])
    mean = weight * math.cos(theta) / 4.0
    half = weight * float(inputs["half_extents_mm"][2]) * math.sin(theta) / (4.0 * half_w)
    creep = (
        weight * math.sin(theta) - 2.0 * float(inputs["friction_coefficient"]) * (mean - half)
    ) / (2.0 * float(inputs["tangential_stiffness"]))
    state = _settled_start(
        theta, inputs, stiffness=stiffness, base_points=4, creep=creep
    )
    force, torque = _callbacks(theta, inputs, stiffness=stiffness, points=points)
    tilts: list[float] = []
    final, diagnostics = integrate_free_flight(
        RK4_BODY,
        state=state,
        inertia=_inertia(inputs),
        dt_s=float(inputs["dt_s"]),
        steps=int(inputs["steps"]),
        force_world_n=force,
        torque_body_nmm=torque,
        observer=lambda index, t, current: tilts.append(_tilt(current.vector, theta)),
    )
    response = _response(
        final.vector, theta, inputs, stiffness=stiffness, points=points
    )
    return {
        "entry": entry, "inputs": inputs, "theta": theta, "points": points,
        "final": final, "diagnostics": diagnostics, "response": response,
        "tilts": tilts,
    }


def test_the_box_below_the_threshold_keeps_every_support_loaded(stable_run) -> None:
    """撑住的定义是**四个角一个都没抬起来**，而不是"看着没倒"。

    先判这一条，因为下面所有静态闭式都以"四点全承载"为前提——
    前提没被判就用它，是spec/12第6.2节点名的那类假通过。
    """

    response = stable_run["response"]
    assert response.contact_count == 4
    assert _loaded_indices(response) == [0, 1, 2, 3]
    stable_run["entry"].check("loaded_support_count", response.contact_count)


def test_the_static_load_split_matches_the_closed_form(stable_run) -> None:
    """上坡/下坡两侧的单点法向力对闭式，**并且横坡对称被单独判一条**。

    横坡方向没有任何物理激励，所以`y = ±d`的两个角本该给出同一个数。
    **而它不是逐位相同的**——实测相对差4e-14。

    病根是求和次序：四个点的贡献按声明次序累加，`+y`与`−y`那两项
    进入`total_torque_world`的时机不同，于是它们在浮点上不精确相消。
    **写成一条带上界的判据而不是逐位判据，是因为逐位在这里是假的**
    （两点平面组那一档才真是逐位的零，见
    `test_the_planar_setup_keeps_the_out_of_plane_state_bitwise_zero`）。
    上界`cross_slope_asymmetry_bound`留约250倍余量：它挡的是**真的横向力**，
    不是舍入。
    """

    entry, theta = stable_run["entry"], stable_run["theta"]
    normal, _ = _frame(theta)
    forces = [
        abs(sum(point.force_world_n[i] * normal[i] for i in range(3)))
        for point in stable_run["response"].points
    ]
    bound = float(stable_run["inputs"]["cross_slope_asymmetry_bound"])
    #: `box_corner_points_mm`的次序：0/2是上坡（`x = −w`）、1/3是下坡。
    for left, right in ((0, 2), (1, 3)):
        gap = abs(forces[left] - forces[right]) / forces[left]
        assert gap <= bound, f"横坡两侧相对差{gap!r}超出上界——有东西在横向推它"
    entry.check("normal_force_uphill_n", forces[0])
    entry.check("normal_force_downhill_n", forces[1])


def test_the_pressure_centre_is_where_the_closed_form_puts_it(stable_run) -> None:
    """**"重心投影"这件事被测出来了**：压心`x_N = h·tanθ`，且`x_N/w < 1`。

    这一条是本案例的物理核心。翻倒的几何阈值就是`x_N = w`，
    而`x_N`不是一个几何声明——它是**法向力的一阶矩除以合力**，
    由引擎逐点算出来的力现场加权得到。

    它比单点力那两条**紧40倍**（1e-4 对 4e-3），因为它是力矩平衡的恒等式
    而不是载荷分配的近似：柔度在分子分母同阶相消。
    """

    entry, theta = stable_run["entry"], stable_run["theta"]
    half_w = float(stable_run["inputs"]["half_extents_mm"][0])
    centre = _pressure_centre_mm(stable_run["response"], theta)
    entry.check("pressure_centre_offset_mm", centre)
    entry.check("pressure_centre_over_halfwidth", centre / half_w)
    assert centre < half_w, "压心已经出了底面，那就不该还撑着"


def test_the_pose_drift_stays_inside_its_declared_bound(stable_run) -> None:
    """**位姿漂移的上界**——本条回答"长时间保持静止"到底静到什么程度。

    三个量各给一条，因为它们坏起来的样子不同：

    * **倾角摆幅**有上界（`tilt_swing_bound_rad`）——它是翻倒判据读的那个角；
    * **横坡角速度**有上界——速度型摩擦在饱和支上对横向扰动没有回复力，
      本案例的对称性把它压在舍入量级，**而那必须是被判的**；
    * **沿坡蠕滑**不设上界而是**对闭式**：速度型摩擦本来就不静止
      （`contact_dynamics`那条"残余滑移不是零"），把它判成零是错的。
    """

    entry, inputs = stable_run["entry"], stable_run["inputs"]
    tilts = stable_run["tilts"]
    swing = max(tilts) - min(tilts)
    assert swing <= float(inputs["tilt_swing_bound_rad"]), (
        f"倾角摆幅{swing!r}超出声明上界——它没在原地待着"
    )
    entry.check("settled_tilt_rad", tilts[-1])

    vector = stable_run["final"].vector
    for axis in (0, 2):
        assert abs(vector[6 + axis]) <= float(inputs["out_of_plane_bound_rad_per_s"])

    _, downhill = _frame(stable_run["theta"])
    speed = sum(vector[3 + i] * downhill[i] for i in range(3))
    entry.check("creep_speed_mm_per_s", speed)


def test_the_assembled_normal_torque_is_the_overturning_couple(stable_run) -> None:
    """**装配好的那个法向力矩本身被判一条**，不只是逐点力。

    球那一档`normal_torque_body_nmm`是**结构性的零**（杆臂沿`n̂`、法向力也沿`n̂`）；
    支承点这一档它是**倾覆力矩本身**`τ_n = −(Σ x_i F_i)·ŷ = −h·W·sinθ·ŷ`。
    同一个字段两档语义不同，所以两档各配一条判据——这是本仓
    "一个结构性的零如果不被判，实现里多出一项来也没人知道"的对偶：
    **一个非零如果不被判，实现里少一项来也没人知道。**

    另判合力矩在平衡态`≈ 0`：那是"它确实处在平衡上"这句话的定量形式，
    没有它，上面那个力矩可以对而体正在被一个没人看的净力矩推着。
    """

    entry, inputs = stable_run["entry"], stable_run["inputs"]
    response = stable_run["response"]
    entry.check("normal_torque_cross_slope_nmm", response.normal_torque_body_nmm[1])
    bound = float(inputs["residual_torque_bound_nmm"])
    for axis, value in enumerate(response.torque_body_nmm):
        assert abs(value) <= bound, (
            f"平衡态合力矩第{axis}个分量是{value!r}——有净力矩在推它"
        )


def test_which_supports_saturate_is_judged_not_assumed(stable_run) -> None:
    """**哪几个点饱和是判据本身**，不是背景设定。

    蠕滑速度那条闭式取哪一支由它决定：均分支给1.3980 mm/s、饱和支给1.5420，
    **差10.5%，是那条容差的21倍**。所以若引擎报的`sliding`与闭式取的支不一致，
    上面那条蠕滑判据会红——本条把这层依赖**写成断言**，
    免得它在重构里悄悄变成一句旁白（形制同金字塔那条"塌了不许靠不收敛"）。
    """

    entry = stable_run["entry"]
    flags = [point.sliding for point in stable_run["response"].points]
    entry.check("uphill_support_saturates", flags[0])
    assert flags[0] is flags[2], "两个上坡角的饱和状态必须相同"
    assert flags[1] is False and flags[3] is False, "下坡角不该饱和"


# ---------------------------------------------------------------------------
# 二、失稳侧：`tanθ > w/h`，**真的翻过去**
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def topple_run(oracles):
    entry = oracles["oracle:box_tipping/topple"]
    inputs = entry.inputs
    theta = math.radians(float(inputs["incline_deg"]))
    stiffness = float(inputs["normal_stiffness_n_per_mm"])
    #: **八个角**，不是四个：翻过去之后贴地的是另一个面，
    #: 只声明底面四个角的话那个面根本不存在，体会直接穿过去。
    points = box_corner_points_mm(tuple(inputs["half_extents_mm"]))
    half_h = float(inputs["half_extents_mm"][2])
    #: 起手是**平放**（均匀穿透、零倾角、零速度），于是"承载点从底面
    #: 整体换到侧面"这句话有一个干净的起点。
    penetration = _weight_n(inputs) * math.cos(theta) / (4.0 * stiffness)
    normal, _ = _frame(theta)
    state = make_state(
        position_mm=tuple((half_h - penetration) * c for c in normal),
        attitude_xyzw=(0.0, math.sin(0.5 * theta), 0.0, math.cos(0.5 * theta)),
    )
    force, torque = _callbacks(theta, inputs, stiffness=stiffness, points=points)
    initial = _response(state.vector, theta, inputs, stiffness=stiffness, points=points)

    tilts: list[float] = []
    deepest = [0.0]

    def observe(index, t, current):
        tilts.append(_tilt(current.vector, theta))

    lift_state, lift_diag = integrate_free_flight(
        RK4_BODY, state=state, inertia=_inertia(inputs), dt_s=float(inputs["dt_s"]),
        steps=int(inputs["lift_off_step"]), force_world_n=force,
        torque_body_nmm=torque, observer=observe,
    )
    lift = _response(
        lift_state.vector, theta, inputs, stiffness=stiffness, points=points
    )
    final, diagnostics = integrate_free_flight(
        RK4_BODY, state=lift_state, inertia=_inertia(inputs),
        dt_s=float(inputs["dt_s"]),
        steps=int(inputs["steps"]) - int(inputs["lift_off_step"]),
        force_world_n=force, torque_body_nmm=torque, observer=observe,
    )
    response = _response(
        final.vector, theta, inputs, stiffness=stiffness, points=points
    )
    deepest[0] = min(point.gap_mm for point in response.points)
    return {
        "entry": entry, "inputs": inputs, "theta": theta,
        "initial": initial, "lift": lift, "response": response,
        "tilts": tilts, "final": final,
        "diagnostics": (lift_diag, diagnostics), "deepest": deepest[0],
    }


def test_the_contact_set_moves_from_the_base_to_the_downhill_side_face(topple_run) -> None:
    """**本文件最要紧的一条**：承载点从底面整体换到了下坡侧面。

    三个时刻各判一次下标集合，零容差：

    * 起手 `[0, 1, 2, 3]` —— 整个底面；
    * 抬边 `[1, 3]` —— 上坡侧那两个角离地了（**绕下坡棱翻，不是整体弹起**）；
    * 末态 `[1, 3, 5, 7]` —— `x_body = +w`那一整个面，**与起手一个都不重合**。

    姿态角变大可以由一次数值发散做出来；**承载点从一个面整体换到另一个面不能**。
    这就是本条与下面那条姿态角判据分工的理由。
    """

    entry = topple_run["entry"]
    entry.check("initial_loaded_indices", _loaded_indices(topple_run["initial"]))
    entry.check("lift_off_loaded_indices", _loaded_indices(topple_run["lift"]))
    entry.check("final_loaded_indices", _loaded_indices(topple_run["response"]))


def test_the_attitude_really_passes_the_geometric_balance_angle(topple_run) -> None:
    """姿态角越过闭式给的平衡角，而且**末态恰好是直角**。

    平衡角是`φ_eq = ψc − θ`（生成器第二节）。本组`θ = 32° > ψc`，
    于是`φ_eq < 0`——**从起手就已经过了平衡点**，倾覆力矩单调把`φ`推大。
    判据取`tilt_gate_rad = 1.5`（85.9°），远在任何晃动之外。

    末态`φ = π/2`是几何预言：落到侧面上时底面法向恰好转到坡向。
    """

    entry, inputs = topple_run["entry"], topple_run["inputs"]
    half_w, _, half_h = (float(v) for v in inputs["half_extents_mm"])
    psi_c = math.atan2(half_w, half_h)
    balance = psi_c - topple_run["theta"]
    assert balance < 0.0, "本组的倾角必须已经越过阈值，否则判的是另一件事"

    tilts = topple_run["tilts"]
    assert max(tilts) >= float(inputs["tilt_gate_rad"])
    #: 单调那一段：从起手到第一次越过1.0 rad，倾角不许回头。
    #: 它把"翻过去"与"晃了一下又回来"分开——后者在这一段必然回头。
    climbing = []
    for value in tilts:
        climbing.append(value)
        if value >= 1.0:
            break
    assert len(climbing) > 1, "第一步就越过1 rad——那不是翻倒，是起手位形就错了"
    for index in range(1, len(climbing)):
        assert climbing[index] >= climbing[index - 1], (
            "倾角在越过1 rad之前回过头——那不是翻倒是晃动"
        )
    entry.check("final_tilt_rad", tilts[-1])


def test_the_topple_never_leans_on_the_integrator_blowing_up(topple_run) -> None:
    """**必红专防：不许拿"积分器炸了"冒充翻倒。**

    形制抄`test_three_sphere_pyramid_rotational::
    test_the_collapse_never_leans_on_non_convergence`——那里挡的是"不收敛"，
    这里挡的是"发散"，同一族。

    三条正面：`integrate_free_flight`没抛（跑完了本身就是证据，但"它没抛"
    太容易在重构里悄悄消失，所以写成断言）；四元数范数偏离**远低于**护栏；
    最深穿透没有跨过盒子的量级（穿透失控是发散的另一副面孔）。

    **必红那一半在最后**：把步长放到显式稳定上限之外，本装置**必须抛**。
    没有它，"没发散"在一个永远不会抛的实现上也是绿的。
    """

    inputs = topple_run["inputs"]
    for diagnostics in topple_run["diagnostics"]:
        assert diagnostics.renormalisations == diagnostics.steps
        assert diagnostics.max_norm_deviation < 0.1 * QUATERNION_NORM_STEP_ABS_TOL, (
            "四元数范数偏离逼近护栏——这一趟的姿态角不能算数"
        )
    assert topple_run["deepest"] >= -float(inputs["penetration_bound_mm"]), (
        "末态穿透超出声明上界——体在往平面里陷，那不是落稳"
    )

    #: 必红：稳定上限是`h < 2.785/ω`，`ω = √(k·1000/m) = 1000 rad/s`，
    #: 于是`h > 2.785e-3`必炸。取5e-3，**远离边界**免得它变成一条运气门。
    theta = topple_run["theta"]
    points = box_corner_points_mm(tuple(inputs["half_extents_mm"]))
    stiffness = float(inputs["normal_stiffness_n_per_mm"])
    normal, _ = _frame(theta)
    half_h = float(inputs["half_extents_mm"][2])
    penetration = _weight_n(inputs) * math.cos(theta) / (4.0 * stiffness)
    state = make_state(
        position_mm=tuple((half_h - penetration) * c for c in normal),
        attitude_xyzw=(0.0, math.sin(0.5 * theta), 0.0, math.cos(0.5 * theta)),
    )
    force, torque = _callbacks(theta, inputs, stiffness=stiffness, points=points)
    with pytest.raises(RigidBodyError, match="drifted"):
        integrate_free_flight(
            RK4_BODY, state=state, inertia=_inertia(inputs), dt_s=5.0e-3, steps=400,
            force_world_n=force, torque_body_nmm=torque,
        )


# ---------------------------------------------------------------------------
# 三、二分夹住临界角
# ---------------------------------------------------------------------------


def _tips(theta, inputs, *, stiffness, steps) -> bool:
    """跑一趟，回答一个布尔量：**它翻了没有**。

    支承点只有两个（`y = 0`的一条底棱），因为翻倒这件事是**平面问题**：
    所有力与杆臂都在`x`-`z`面内，力矩因此只有`y`分量，横坡自由度
    一步都不被激励。省掉的那一半点数直接变成二分的迭代次数。
    """

    points = tuple(tuple(float(v) for v in point) for point in inputs["support_points_body_mm"])
    state = _settled_start(
        theta, inputs, stiffness=stiffness, base_points=2,
        creep=_weight_n(inputs) * math.sin(theta) / (2.0 * float(inputs["tangential_stiffness"])),
    )
    force, torque = _callbacks(theta, inputs, stiffness=stiffness, points=points)
    final, _ = integrate_free_flight(
        RK4_BODY, state=state, inertia=_inertia(inputs), dt_s=float(inputs["dt_s"]),
        steps=steps, force_world_n=force, torque_body_nmm=torque,
    )
    return abs(_tilt(final.vector, theta)) > float(inputs["tip_gate_rad"])


def _bisect(inputs, *, stiffness, steps, iterations) -> tuple[float, float]:
    centre = float(inputs["rigid_critical_angle_rad"])
    half = float(inputs["search_halfwidth_rad"])
    low, high = centre - half, centre + half
    assert not _tips(low, inputs, stiffness=stiffness, steps=steps), "下端应该撑住"
    assert _tips(high, inputs, stiffness=stiffness, steps=steps), "上端应该翻"
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if _tips(middle, inputs, stiffness=stiffness, steps=steps):
            high = middle
        else:
            low = middle
    return 0.5 * (low + high), high - low


@pytest.fixture(scope="module")
def bracket(oracles):
    entry = oracles["oracle:box_tipping/critical_angle_bracket"]
    inputs = entry.inputs
    soft, soft_width = _bisect(
        inputs, stiffness=float(inputs["soft_stiffness_n_per_mm"]),
        steps=int(inputs["soft_steps"]), iterations=int(inputs["soft_iterations"]),
    )
    stiff, stiff_width = _bisect(
        inputs, stiffness=float(inputs["stiff_stiffness_n_per_mm"]),
        steps=int(inputs["stiff_steps"]), iterations=int(inputs["stiff_iterations"]),
    )
    return {
        "entry": entry, "inputs": inputs,
        "soft": soft, "soft_width": soft_width,
        "stiff": stiff, "stiff_width": stiff_width,
    }


def test_the_bisection_brackets_the_critical_angle(bracket) -> None:
    """二分夹住临界角，**夹取宽度就是容差**。

    两个刚度各夹一次：软档7次迭代（宽度1.5625e-4 rad）、硬档9次（3.906e-5 rad）。
    金标是**柔度修正后的**闭式`ψc − β_c`，而不是刚体闭式——理由在下一条。
    """

    entry, inputs = bracket["entry"], bracket["inputs"]
    assert bracket["soft_width"] == pytest.approx(
        2.0 * float(inputs["search_halfwidth_rad"]) / 2 ** int(inputs["soft_iterations"]),
        rel=1e-12,
    )
    entry.check("critical_angle_soft_rad", bracket["soft"])
    entry.check("critical_angle_stiff_rad", bracket["stiff"])


def test_the_rigid_closed_form_is_outside_the_bracket_and_the_gap_is_compliance(
    bracket,
) -> None:
    """**偏差是罚柔度，不是实现缺陷——而这句话被判成了两条**。

    第一条：刚体闭式`ψc = arctan(w/h)`**落在夹取区间之外**。
    若它落在里面，"柔度修正"就没有可观测后果，那条修正也就不值得写。

    第二条（决定性的那条）：软/硬两档偏离刚体闭式的量之比**等于刚度之比**。
    实现缺陷不会随罚刚度按`1/k`缩小，**柔度会**。
    """

    entry, inputs = bracket["entry"], bracket["inputs"]
    rigid = float(inputs["rigid_critical_angle_rad"])
    soft_dev = rigid - bracket["soft"]
    stiff_dev = rigid - bracket["stiff"]
    assert soft_dev > 0.5 * bracket["soft_width"], (
        f"刚体闭式落在软档夹取区间内（偏差{soft_dev!r}）——那这一页就没有东西可说"
    )
    assert stiff_dev > 0.5 * bracket["stiff_width"]
    entry.check("deviation_ratio_soft_over_stiff", soft_dev / stiff_dev)


def test_the_exact_tangent_threshold_is_a_rational_number(bracket) -> None:
    """`tanθ_c = w/h`是**有理数**：`5/10 = 1/2`，`Fraction`算，零容差。

    写成一条门是因为它是本页唯一一个"不该有任何容差"的量——
    几何阈值不是测量值。清单里存的是分子与分母两个整数，
    **不是一个浮点数**，于是"它是不是有理数"这件事本身也被形制守住了。
    """

    inputs = bracket["inputs"]
    numerator = int(inputs["exact_tangent_threshold_numerator"])
    denominator = int(inputs["exact_tangent_threshold_denominator"])
    half_w, _, half_h = (float(v) for v in inputs["half_extents_mm"])
    assert Fraction(numerator, denominator) == Fraction(int(half_w * 2), int(half_h * 2))
    assert math.tan(float(inputs["rigid_critical_angle_rad"])) == pytest.approx(
        numerator / denominator, rel=1e-15, abs=0.0
    )


# ---------------------------------------------------------------------------
# 四、退化与形制：逐位判据
# ---------------------------------------------------------------------------


def test_no_contact_reproduces_free_flight_bit_for_bit(oracles) -> None:
    """**退化逐位**：不接触时，接过接触回调的积分与纯自由飞行`float.hex()`逐位相同。

    "不接触时不加力"听起来自明，但罚接触的实现里最容易多出来的正是
    一项"很小的"东西（间隙判据写成`<= 0`、阻尼在分离时没关、
    切向力在零滑移时除了个零再乘回来）。**逐位是唯一能挡住"很小"的容差。**
    """

    entry = oracles["oracle:box_tipping/stable_side"]
    inputs = entry.inputs
    theta = math.radians(float(inputs["incline_deg"]))
    points = box_corner_points_mm(tuple(inputs["half_extents_mm"]))
    weight = _weight_n(inputs)
    normal, _ = _frame(theta)
    #: 抬到平面上方一整个盒高之外，任何一个角都够不着。
    high = 4.0 * float(inputs["half_extents_mm"][2])
    state = make_state(
        position_mm=tuple(high * c for c in normal),
        velocity_mm_per_s=(3.0, -2.0, 1.0),
        angular_velocity_rad_per_s=(0.7, -0.3, 0.5),
        attitude_xyzw=(0.0, math.sin(0.5 * theta), 0.0, math.cos(0.5 * theta)),
    )
    with_contact, _ = integrate_free_flight(
        RK4_BODY, state=state, inertia=_inertia(inputs), dt_s=1.0e-4, steps=200,
        force_world_n=_callbacks(
            theta, inputs, stiffness=float(inputs["normal_stiffness_n_per_mm"]),
            points=points,
        )[0],
        torque_body_nmm=_callbacks(
            theta, inputs, stiffness=float(inputs["normal_stiffness_n_per_mm"]),
            points=points,
        )[1],
    )
    plain, _ = integrate_free_flight(
        RK4_BODY, state=state, inertia=_inertia(inputs), dt_s=1.0e-4, steps=200,
        force_world_n=lambda vector, t: (0.0, 0.0, -weight),
        torque_body_nmm=None,
    )
    assert [value.hex() for value in with_contact.vector] == [
        value.hex() for value in plain.vector
    ], "不接触那一趟与纯自由飞行逐位不同——有东西在零间隙之外还在加力"


def test_the_one_slot_memo_changes_nothing(oracles) -> None:
    """回调里那一格记忆**不许改一个字节**（本仓性能条款第二句）。

    回调走记忆、直接调用不走，两条路的力与力矩必须`float.hex()`逐位相同。
    记忆的键是**对象同一性**而不是数值相等，所以这条不是"应该相同"是"必然相同"——
    而"必然"这个词正是要被判一次的那种词。
    """

    entry = oracles["oracle:box_tipping/stable_side"]
    inputs = entry.inputs
    theta = math.radians(float(inputs["incline_deg"]))
    stiffness = float(inputs["normal_stiffness_n_per_mm"])
    points = box_corner_points_mm(tuple(inputs["half_extents_mm"]))[:4]
    state = _settled_start(theta, inputs, stiffness=stiffness, base_points=4, creep=1.5)
    force, torque = _callbacks(theta, inputs, stiffness=stiffness, points=points)
    direct = _response(state.vector, theta, inputs, stiffness=stiffness, points=points)
    weight = _weight_n(inputs)
    expected_force = tuple(
        value + component
        for value, component in zip(
            direct.force_world_n, (0.0, 0.0, -weight), strict=True
        )
    )
    assert [v.hex() for v in force(state.vector, 0.0)] == [
        v.hex() for v in expected_force
    ]
    assert [v.hex() for v in torque(state.vector, 0.0)] == [
        v.hex() for v in direct.torque_body_nmm
    ]
    #: 再问一次同一个向量——命中记忆的那一路也必须逐位相同。
    assert [v.hex() for v in torque(state.vector, 0.0)] == [
        v.hex() for v in direct.torque_body_nmm
    ]


def _flat_probe(
    oracles, *, penetration_mm, normal_velocity_mm_per_s, damping, sideways=0.0
):
    """水平面上的单次求值——**倾角取零、姿态取单位四元数**，于是杆臂逐位等于
    体系点、间隙逐位等于`centre_z + p_z`。两条边界判据要的正是这种可控到位的算例。"""

    inputs = oracles["oracle:box_tipping/stable_side"].inputs
    half_h = float(inputs["half_extents_mm"][2])
    points = box_corner_points_mm(tuple(inputs["half_extents_mm"]))[:4]
    state = make_state(
        position_mm=(0.0, 0.0, half_h - penetration_mm),
        velocity_mm_per_s=(sideways, 0.0, normal_velocity_mm_per_s),
    )
    return points, support_points_plane_contact(
        state.vector,
        support_points_body_mm=points,
        plane_point_mm=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        normal_stiffness_n_per_mm=float(inputs["normal_stiffness_n_per_mm"]),
        tangential_stiffness_n_per_mm=float(inputs["tangential_stiffness"]),
        friction_coefficient=float(inputs["friction_coefficient"]),
        normal_damping_n_s_per_mm=damping,
    )


def test_a_support_point_exactly_on_the_plane_is_not_loaded(oracles) -> None:
    """**间隙恰为零的那个点不承载**——`gap < 0`，不是`<= 0`。

    零间隙的接触力本来就是零（`−k·0`），所以"算不算承载"影响不到任何一个力；
    影响的是`in_contact`与`contact_count`，**而本页最响亮的那条判据读的正是它们**
    （承载点下标集合）。把边界判成承载，翻倒过程里"上坡边什么时候抬起来"
    就会整整差一帧，而力那一侧一个字节都不变——**没有这条门就没人会发现。**

    口径与`contact/penalty.py`同源：本仓只许有一个"接触与否"的定义。
    """

    points, response = _flat_probe(
        oracles, penetration_mm=0.0, normal_velocity_mm_per_s=0.0, damping=0.0
    )
    assert len(response.points) == len(points)
    for index, point in enumerate(response.points):
        assert point.gap_mm.hex() == 0.0.hex(), f"第{index}点的间隙不是逐位的零"
        assert point.in_contact is False, "间隙为零被判成了承载"
        assert [v.hex() for v in point.force_world_n] == [0.0.hex()] * 3
    assert response.contact_count == 0
    assert [v.hex() for v in response.torque_body_nmm] == [0.0.hex()] * 3

    #: 同一个边界再问一次，这次带切向速度：**不承载的点报的滑移速率必须是零**。
    #: 力那一侧对`gap >= 0`与`gap > 0`两种写法**恰好逐位相同**
    #: （`−k·0 = −0.0`，而`−0.0 + 0.0 = +0.0`），所以力挡不住这个改法——
    #: 挡得住的只有这个诊断量。注错表M3那一行记的就是这件事。
    _, moving = _flat_probe(
        oracles, penetration_mm=0.0, normal_velocity_mm_per_s=0.0,
        damping=0.0, sideways=3.0,
    )
    for index, point in enumerate(moving.points):
        assert point.slip_speed_mm_per_s.hex() == 0.0.hex(), (
            f"第{index}点没接触却报了滑移速率{point.slip_speed_mm_per_s!r}"
        )
        assert point.sliding is False


def test_the_damper_pushes_on_compression_and_never_pulls_on_separation(oracles) -> None:
    """**法向阻尼是单向的**：压缩时加力、分离时一个字节都不加。

    两侧各判一次，因为只判一侧的话两种错都溜得过去：

    * 只判"分离时不加力"——一个把阻尼整个删掉的实现照样绿；
    * 只判"压缩时加力"——一个在分离时把体**吸住**的实现照样绿，
      而那正是罚接触最经典的那个错（弹簧-阻尼器在分离段变成拉力）。

    分离那一侧判的是**逐位等于纯弹簧力**，不是"差不多"。
    """

    inputs = oracles["oracle:box_tipping/stable_side"].inputs
    damping = float(inputs["normal_damping"])
    stiffness = float(inputs["normal_stiffness_n_per_mm"])
    speed = 7.0

    _, spring_only = _flat_probe(
        oracles, penetration_mm=0.01, normal_velocity_mm_per_s=speed, damping=0.0
    )
    _, separating = _flat_probe(
        oracles, penetration_mm=0.01, normal_velocity_mm_per_s=speed, damping=damping
    )
    assert [v.hex() for v in separating.force_world_n] == [
        v.hex() for v in spring_only.force_world_n
    ], "分离时阻尼还在加力——体被吸住了"

    _, compressing = _flat_probe(
        oracles, penetration_mm=0.01, normal_velocity_mm_per_s=-speed, damping=damping
    )
    _, compressing_spring = _flat_probe(
        oracles, penetration_mm=0.01, normal_velocity_mm_per_s=-speed, damping=0.0
    )
    added = compressing.force_world_n[2] - compressing_spring.force_world_n[2]
    assert added == pytest.approx(4.0 * damping * speed, rel=1e-12), (
        "压缩时阻尼没有按`c·|v_n|`逐点加上去"
    )
    assert compressing_spring.force_world_n[2] == pytest.approx(
        4.0 * stiffness * 0.01, rel=1e-12
    )


def test_the_planar_setup_keeps_the_out_of_plane_state_bitwise_zero(oracles) -> None:
    """二分那一组是**严格平面**的：横坡向的状态分量逐位是零。

    支承点在`y = 0`、力与杆臂都在`x`-`z`面内，于是力矩只有`y`分量、
    姿态四元数只在`(y, w)`子空间里动。**这不是"很小"，是零**——
    与球那一档"法向力矩恒为零"同一族的结构性事实。

    判它的理由也同一族：一个多出一项来的实现，在别的判据上完全看不出来。
    """

    entry = oracles["oracle:box_tipping/critical_angle_bracket"]
    inputs = entry.inputs
    theta = float(inputs["rigid_critical_angle_rad"]) - 0.02
    stiffness = float(inputs["soft_stiffness_n_per_mm"])
    points = tuple(
        tuple(float(v) for v in point) for point in inputs["support_points_body_mm"]
    )
    state = _settled_start(
        theta, inputs, stiffness=stiffness, base_points=2,
        creep=_weight_n(inputs) * math.sin(theta)
        / (2.0 * float(inputs["tangential_stiffness"])),
    )
    force, torque = _callbacks(theta, inputs, stiffness=stiffness, points=points)
    final, _ = integrate_free_flight(
        RK4_BODY, state=state, inertia=_inertia(inputs), dt_s=float(inputs["dt_s"]),
        steps=300, force_world_n=force, torque_body_nmm=torque,
    )
    zero = 0.0.hex()
    #: y位置、y速度、体系角速度的x与z、姿态四元数的x与z。
    for index in (1, 4, 6, 8, 9, 11):
        assert final.vector[index].hex() in (zero, (-0.0).hex()), (
            f"分量{index}不是逐位的零：{final.vector[index]!r}"
        )
