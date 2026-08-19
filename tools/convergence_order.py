#!/usr/bin/env python3
"""步长收敛阶的扫描器——plans/16第四节GAP第1条（决策0087丙4）。

`rolling_ball_incline`与`box_tipping_threshold`两条案例**只做过不变性，没做过收敛阶**。
案例页自己写着理由："不变性证明不了误差随h怎么走"。本脚本补的就是那件事。

## 判据本身要先被验（本仓`tests/governance/`的样板纪律）

`verify`子命令把同一套阶估计器套到一个**已知阶的解析问题**上
（`cases/harmonic_oscillator`那条：解析解``x(t) = cos(ωt)``、
显式Euler形式一阶、velocity Verlet形式二阶）。**估计器先在那上面对，
才谈得上拿它去量别的东西。**

## 两条估计路径，都要给，因为它们坏起来不一样

* **对解析真值**：``e_i = |u(h_i) − u*|``，阶``p = log₂(e_i/e_{i+1})``（相邻档减半）；
* **Richardson**（没有真值时唯一能用的）：``p = log₂(|u_i − u_{i+1}| / |u_{i+1} − u_{i+2}|)``。

`verify`同时跑两条并要求它们给同一个阶——**一个把Richardson写反的实现
在"对真值"那条上仍然全对**，反之亦然。

## 步长阶梯不是随手取的：它要**先过`contact_dynamics`那两条上限**

决策0087丙1／丙2给了`h`与`k_t`的显式稳定上限。收敛阶扫描最容易犯的错是
把最粗那一档取到上限之外——那一档不是"误差大一点"，是**发散**，
于是整条阶梯拟合出来的"阶"是一个假数。本脚本因此在跑之前**先问上限**，
越界当场失败关闭并报出是哪一条越界。

## 用法

    .venv/bin/python tools/convergence_order.py verify
    .venv/bin/python tools/convergence_order.py run --problem all
    .venv/bin/python tools/convergence_order.py run --problem rolling_ball_roll --levels 5

**本机只跑得动小规模**（`--scale local`，默认）：`--scale full`那一档
最细一档要跑十万量级的接触步，要上master——本机Mac负载常年5—20，
在它上面量墙钟没有意义（plans/07已登记），而**收敛阶本身不是墙钟量、
它跨机器稳定**，上master只是为了跑得完。

    bash tools/master/run_on_master.sh 'python tools/convergence_order.py run --problem all --scale full'
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.contact_dynamics import (  # noqa: E402
    box_corner_points_mm,
    contact_dynamics_step_bound,
    sphere_plane_callbacks,
    support_points_plane_callbacks,
)
from physics_engine.integrate import (  # noqa: E402
    EXPLICIT_EULER,
    SYMPLECTIC_EULER,
    VELOCITY_VERLET,
    integrate,
)
from physics_engine.rigidbody import (  # noqa: E402
    RK4_BODY,
    RigidBodyInertia,
    attitude_matrix,
    integrate_free_flight,
    make_state,
)
from physics_engine.shapes import RoundedBox  # noqa: E402


class ConvergenceOrderError(ValueError):
    """扫描器的失败关闭。**不返回一个"大概是这么多阶"的数。**"""


# ---------------------------------------------------------------------------
# 估计器本体
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderEstimate:
    """一次阶估计的全部中间量。

    中间量全部进结果，理由与`SolveResult.backtracks`同源：
    **一个只给"阶 = 3.97"的结果，读的人无法判断它是不是算错了。**
    """

    name: str
    method: str
    dt_ladder: tuple[float, ...]
    values: tuple[float, ...]
    #: 对真值那条是``|u − u*|``；Richardson那条是相邻两档之差``|u_i − u_{i+1}|``。
    differences: tuple[float, ...]
    #: 相邻两个差的比值。理想是``2^p``。
    ratios: tuple[float, ...]
    #: 每个比值给出的阶。
    orders: tuple[float, ...]
    #: 最细两档给出的那个阶——**渐近区在细端，不在粗端**。测不出来时是``nan``。
    asymptotic_order: float
    #: 一句话结论。**"测不出来"是一条结论，不是一次失败**——
    #: 把它写成一个数（哪怕是`nan`）而不写清楚，读的人会以为那个数有意义。
    verdict: str = ""

    def render(self) -> str:
        lines = [f"=== {self.name}（{self.method}）==="]
        lines.append("  h            值")
        for dt_s, value in zip(self.dt_ladder, self.values, strict=True):
            lines.append(f"  {dt_s:<12.6g} {value!r}")
        lines.append("  差            比值      阶")
        for index, difference in enumerate(self.differences):
            if index < len(self.ratios):
                lines.append(
                    f"  {difference:<13.6e} {self.ratios[index]:<9.4f} "
                    f"{self.orders[index]:.4f}"
                )
            else:
                #: 最后一档没有下一档可比——**留白，不填一个`nan`**：
                #: 一个`nan`在读的人眼里像是"这里算坏了"。
                lines.append(f"  {difference:<13.6e} —         —")
        if math.isnan(self.asymptotic_order):
            lines.append(f"  渐近阶：测不出来 —— {self.verdict}")
        else:
            lines.append(f"  渐近阶（最细两档）：{self.asymptotic_order:.4f}")
            if self.verdict:
                lines.append(f"  {self.verdict}")
        return "\n".join(lines)


def _orders_from(differences: Sequence[float], ladder: Sequence[float]) -> tuple[
    tuple[float, ...], tuple[float, ...]
]:
    """由逐档的差算比值与阶。**比值的底取相邻两档步长之比**，不写死2。

    写死2的实现在等比减半的阶梯上一样对，在别的阶梯上悄悄给错数——
    而"悄悄给错数"正是本仓反复记的那一类。
    """

    ratios: list[float] = []
    orders: list[float] = []
    for index in range(len(differences) - 1):
        coarse, fine = differences[index], differences[index + 1]
        if not (coarse > 0.0 and fine > 0.0):
            raise ConvergenceOrderError(
                f"第{index}档的差不是正数（{coarse!r} / {fine!r}）——"
                "误差被别的东西吞了（或者两档给出了逐位相同的结果），阶无从谈起"
            )
        refinement = ladder[index] / ladder[index + 1]
        if refinement <= 1.0:
            raise ConvergenceOrderError(
                f"步长阶梯没有变细：{ladder[index]!r} → {ladder[index + 1]!r}"
            )
        ratio = coarse / fine
        ratios.append(ratio)
        orders.append(math.log(ratio) / math.log(refinement))
    if not orders:
        raise ConvergenceOrderError("阶梯太短——至少要能给出一个比值")
    return tuple(ratios), tuple(orders)


def order_against_truth(
    name: str,
    runner: Callable[[float], float],
    ladder: Sequence[float],
    truth: float,
) -> OrderEstimate:
    """有解析真值时的阶：``e_i = |u(h_i) − u*|``。"""

    values = tuple(runner(dt_s) for dt_s in ladder)
    differences = tuple(abs(value - truth) for value in values)
    ratios, orders = _orders_from(differences, ladder)
    return OrderEstimate(
        name=name,
        method="对解析真值",
        dt_ladder=tuple(ladder),
        values=values,
        differences=differences,
        ratios=ratios,
        orders=orders,
        asymptotic_order=orders[-1],
    )


#: 舍入地板：逐档差小到这个相对量以下时，**它量的是舍入不是截断**。
#: 取``2^-40 ≈ 9.1e-13``而不是`eps`本身：一条走上万步的RK4轨迹积累的舍入
#: 本来就在`√N·eps`量级（N = 1e4时约2e-14），再留一个量级余量。
#: **这个数是声明，不是调出来的**；地板以下本脚本不报阶。
RICHARDSON_NOISE_FLOOR_REL = 2.0**-40


def order_by_richardson(
    name: str,
    runner: Callable[[float], float],
    ladder: Sequence[float],
    *,
    noise_floor_rel: float = RICHARDSON_NOISE_FLOOR_REL,
) -> OrderEstimate:
    """没有真值时的阶：相邻两档之差当误差的代理。

    **这条路要求阶梯至少三档**（两个差才给得出一个比值），
    而"对真值"那条只要两档。差别写在这里是因为它决定了`--levels`的下限。

    ## 差落到舍入地板以下时**不报阶**

    `cases/harmonic_oscillator`那道门第一行就是``assert all(error > 0)``，
    理由写在案例页上："误差全为零——离散误差被别的东西吞了，收敛阶无从谈起"。
    本函数把那条纪律做成连续的：差**接近**舍入量级时报的阶同样是噪声，
    而它长得跟一个真的阶一模一样。**报一个假的阶比不报更坏。**
    """

    if len(ladder) < 3:
        raise ConvergenceOrderError(
            f"Richardson至少要三档步长，收到{len(ladder)}档——"
            "两档只给得出一个差，给不出比值"
        )
    values = tuple(runner(dt_s) for dt_s in ladder)
    differences = tuple(
        abs(values[index] - values[index + 1]) for index in range(len(values) - 1)
    )
    scale = max(abs(value) for value in values)
    floor = noise_floor_rel * scale
    if min(differences) <= floor:
        return OrderEstimate(
            name=name,
            method="Richardson（无真值）",
            dt_ladder=tuple(ladder),
            values=values,
            differences=differences,
            ratios=(),
            orders=(),
            asymptotic_order=float("nan"),
            verdict=(
                f"逐档差最小{min(differences):.3e}已落到舍入地板"
                f"{floor:.3e}（相对{noise_floor_rel:.2e}×{scale:.4g}）以下——"
                "**这一档的截断误差比双精度还小，阶无从测**。"
                "这本身是一条结论：在这个步长上该观测量已经收敛到机器精度"
            ),
        )
    ratios, orders = _orders_from(differences, ladder[:-1])
    monotone = all(
        differences[index] > differences[index + 1]
        for index in range(len(differences) - 1)
    )
    # **带符号的差也必须同号**（2026-08-18补，master首次全规模扫描当场抓到）。
    #
    # 上面那条只看`abs`。`box_rocking`那一档的绝对差是**规规矩矩递减的**
    # （1.645e-8 → 1.467e-8 → 3.654e-9 → 2.224e-10 → 1.543e-10），
    # 于是它**一句警告都没有地报出了"渐近阶0.5267"**——而它的带符号差是
    # `−, −, +, −, −`：**变了两次号**。
    #
    # 一个还在渐近区里的Richardson序列，逐档差是同号的（值从一侧单调逼近极限）。
    # 变号意味着某个别的东西（这里多半是舍入地板与截断误差换了主导）已经接管，
    # 那时相邻两差之比是噪声之比，**而它长得跟一个真的阶一模一样**——
    # 与本函数上面那段"报一个假的阶比不报更坏"是同一条纪律的第二种形态。
    signed = tuple(
        values[index] - values[index + 1] for index in range(len(values) - 1)
    )
    same_sign = all(
        signed[index] * signed[index + 1] > 0.0 for index in range(len(signed) - 1)
    )
    return OrderEstimate(
        name=name,
        method="Richardson（无真值）",
        dt_ladder=tuple(ladder),
        values=values,
        differences=differences,
        ratios=ratios,
        orders=orders,
        asymptotic_order=orders[-1],
        verdict=_richardson_verdict(monotone=monotone, same_sign=same_sign),
    )


def _richardson_verdict(*, monotone: bool, same_sign: bool) -> str:
    """两条"没进渐近区"的判据各自报各自的话。

    **分开报不是排版讲究**：绝对值不递减与带符号差变号是两种不同的病，
    而第二种正是2026-08-18 master首扫时**绕过第一条**的那一种。
    合成一句话会让下一个人以为只有一条判据。
    """

    if not monotone and not same_sign:
        return (
            "**逐档差既不单调、带符号差还变了号——没有进渐近区**，"
            "这个阶不许当成收敛阶引用"
        )
    if not monotone:
        return "**逐档差不单调——没有进渐近区**，这个阶不许当成收敛阶引用"
    if not same_sign:
        return (
            "**逐档差的绝对值在递减，但带符号差变了号——没有进渐近区**。"
            "还在渐近区里的Richardson序列是从一侧单调逼近极限的；变号说明"
            "舍入地板与截断误差换了主导，此时相邻两差之比是噪声之比，"
            "**而它长得跟一个真的阶一模一样**。这个阶不许当成收敛阶引用"
        )
    return ""


def _ladder(coarsest: float, levels: int, refinement: float = 2.0) -> tuple[float, ...]:
    return tuple(coarsest / refinement**index for index in range(levels))


def assert_horizon_is_an_integer_number_of_steps(
    name: str, ladder: Sequence[float], horizon_s: float
) -> None:
    """每一档都必须**恰好**走到同一个末时刻。

    ## 这一条是实测逼出来的，不是防御性编程

    第一版的滚球阶梯取`h₀ = 1.2e-5`、`T = 0.02`，于是`T/h`是1666.67、3333.33……
    `round`之后各档的真实末时刻差到`±4e-6 s`。球那时正以`2348 mm/s²`加速，
    于是**末时刻差`4e-6 s`就等于速度差`0.0094 mm/s`**——
    而逐档差实测正是`0.0144 / 0.0072 / 0.0036 / 0.0018`。

    **量到的"一阶收敛"整条是末时刻偏移，一点截断误差都没有。**
    它甚至给出了教科书般漂亮的比值（恒为2.0000、四档全同），
    **一个漂亮的数字在这里恰恰是错的那个**。

    换成`h₀ = 1.25e-5`（`T/h = 1600`，逐次减半仍是整数）之后阶就完全变了。
    """

    for dt_s in ladder:
        quotient = horizon_s / dt_s
        if abs(quotient - round(quotient)) > 1.0e-9 * max(1.0, abs(quotient)):
            raise ConvergenceOrderError(
                f"{name}：步长{dt_s!r}走不到整数步（T/h = {quotient!r}）——"
                "各档末时刻不同，量到的是末时刻偏移而不是截断误差"
            )


# ---------------------------------------------------------------------------
# 判据本身要被验的那个解析问题：`cases/harmonic_oscillator`
# ---------------------------------------------------------------------------

HARMONIC_OMEGA = 2.0
HARMONIC_HORIZON = 3.0


def _harmonic_runner(integrator):
    def run(dt_s: float) -> float:
        steps = int(round(HARMONIC_HORIZON / dt_s))

        def acceleration(x, v, t):
            return (-HARMONIC_OMEGA * HARMONIC_OMEGA * x[0],)

        position, _velocity, _t = integrate(
            integrator, x0=(1.0,), v0=(0.0,), dt_s=dt_s, steps=steps,
            acceleration=acceleration,
        )
        return position[0]

    return run


#: `(名字, 积分器, 形式阶)`。三条形式阶各不相同——**一个"永远返回形式阶"的
#: 估计器在单条问题上分辨不出来，三条一起就分辨得出来**。
HARMONIC_PROBLEMS = (
    ("harmonic/explicit_euler", EXPLICIT_EULER, 1),
    ("harmonic/symplectic_euler", SYMPLECTIC_EULER, 1),
    ("harmonic/velocity_verlet", VELOCITY_VERLET, 2),
)


def verify_estimator(levels: int = 4) -> list[tuple[str, int, OrderEstimate, OrderEstimate]]:
    """把估计器套在已知阶的解析问题上，两条路径各跑一遍。

    返回``(名字, 形式阶, 对真值那条, Richardson那条)``。
    """

    truth = math.cos(HARMONIC_OMEGA * HARMONIC_HORIZON)
    ladder = _ladder(0.02, levels)
    out = []
    for name, integrator, formal in HARMONIC_PROBLEMS:
        runner = _harmonic_runner(integrator)
        out.append(
            (
                name,
                formal,
                order_against_truth(name, runner, ladder, truth),
                order_by_richardson(name, runner, ladder),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 两条接触案例
# ---------------------------------------------------------------------------

GRAVITY = 9810.0

BALL = {
    "radius_mm": 10.0,
    "mass_kg": 1.0,
    "incline_deg": 20.0,
    "normal_stiffness_n_per_mm": 5.0e5,
    "tangential_stiffness_n_per_mm": 50.0,
}

BOX = {
    "half_extents_mm": (5.0, 5.0, 10.0),
    "mass_kg": 0.05,
    "normal_stiffness_n_per_mm": 50.0,
    "normal_damping_n_s_per_mm": 0.01,
    "tangential_stiffness_n_per_mm": 0.03,
    "friction_coefficient": 1.2,
}


def _incline_frame(theta: float):
    return (
        (math.sin(theta), 0.0, math.cos(theta)),
        (math.cos(theta), 0.0, -math.sin(theta)),
    )


def _ball_runner(friction: float, horizon_s: float):
    theta = math.radians(BALL["incline_deg"])
    normal, downhill = _incline_frame(theta)
    weight = BALL["mass_kg"] * GRAVITY / 1000.0
    scalar = 0.4 * BALL["mass_kg"] * BALL["radius_mm"] ** 2
    inertia = RigidBodyInertia(
        mass_kg=BALL["mass_kg"],
        inertia_body_kg_mm2=(
            (scalar, 0.0, 0.0), (0.0, scalar, 0.0), (0.0, 0.0, scalar)
        ),
    )
    squash = weight * math.cos(theta) / BALL["normal_stiffness_n_per_mm"]

    def run(dt_s: float) -> float:
        force, torque = sphere_plane_callbacks(
            radius_mm=BALL["radius_mm"],
            plane_point_mm=(0.0, 0.0, 0.0),
            plane_normal=normal,
            normal_stiffness_n_per_mm=BALL["normal_stiffness_n_per_mm"],
            tangential_stiffness_n_per_mm=BALL["tangential_stiffness_n_per_mm"],
            friction_coefficient=friction,
            gravity_world_n=(0.0, 0.0, -weight),
        )
        final, _ = integrate_free_flight(
            RK4_BODY,
            state=make_state(
                position_mm=tuple((BALL["radius_mm"] - squash) * c for c in normal)
            ),
            inertia=inertia,
            dt_s=dt_s,
            steps=int(round(horizon_s / dt_s)),
            force_world_n=force,
            torque_body_nmm=torque,
        )
        #: 观测量取**沿坡向的质心速度**：它是案例判据读的那个量的直接来源
        #: （案例判的是`v/T`当平均加速度），而速度比位置对步长更敏感。
        velocity = final.block("centre_of_mass_velocity_mm_per_s")
        return sum(velocity[i] * downhill[i] for i in range(3))

    return run


def _box_inertia() -> RigidBodyInertia:
    return RigidBodyInertia.from_shape(
        RoundedBox(half_extents_mm=BOX["half_extents_mm"], fillet_radius_mm=0.0),
        mass_kg=BOX["mass_kg"],
    )


def _box_settled_state(theta: float, tilt_perturbation_rad: float = 0.0):
    """闭式静态位形，与`cases/box_tipping_threshold`同一组式子。

    ``tilt_perturbation_rad``给一个**小到不会让任何一角离地**的初始倾角偏置，
    于是体绕着平衡位形做一次干净的阻尼摇摆：四个角全程承载、
    右端项全程光滑、**没有接触集切换**。那正是"力矩装配这条路本身几阶"
    的干净靶子；带切换的那一档另有一条（`box_tipping_topple`）。

    多大算"小"是可以算的：平衡穿透量``δ = W·cosθ/(4·k_n) ≈ 2.3e-3 mm``，
    半宽5 mm，于是倾角偏置要远小于``δ/w ≈ 4.6e-4 rad``。本模块取``1e-4``，
    留4.6倍余量，并由`box_rocking`那一档的"承载点恒为四个"实测守着。
    """

    weight = BOX["mass_kg"] * GRAVITY / 1000.0
    half_w, _, half_h = BOX["half_extents_mm"]
    mean = weight * math.cos(theta) / 4.0
    half = weight * half_h * math.sin(theta) / (4.0 * half_w)
    creep = (
        weight * math.sin(theta)
        - 2.0 * BOX["friction_coefficient"] * (mean - half)
    ) / (2.0 * BOX["tangential_stiffness_n_per_mm"])
    stiffness = BOX["normal_stiffness_n_per_mm"]
    delta_up, delta_down = (mean - half) / stiffness, (mean + half) / stiffness
    beta = math.asin((delta_down - delta_up) / (2.0 * half_w))
    height = half_h * math.cos(beta) - 0.5 * (delta_up + delta_down)
    normal, downhill = _incline_frame(theta)
    alpha = theta + beta + tilt_perturbation_rad
    return make_state(
        position_mm=tuple(height * axis for axis in normal),
        velocity_mm_per_s=tuple(creep * axis for axis in downhill),
        attitude_xyzw=(0.0, math.sin(0.5 * alpha), 0.0, math.cos(0.5 * alpha)),
    )


def _box_flat_state(theta: float):
    """平放起手（失稳侧那一组）：底面贴着坡面、零速度。"""

    _, half_h = BOX["half_extents_mm"][0], BOX["half_extents_mm"][2]
    normal, _ = _incline_frame(theta)
    return make_state(
        position_mm=tuple(half_h * axis for axis in normal),
        attitude_xyzw=(0.0, math.sin(0.5 * theta), 0.0, math.cos(0.5 * theta)),
    )


def _box_tilt(vector, theta: float) -> float:
    """底面法向相对坡面法向的有符号倾角，与案例同一个读法。"""

    normal, downhill = _incline_frame(theta)
    rows = attitude_matrix(vector[9:13])
    base_normal = (rows[0][2], rows[1][2], rows[2][2])
    return math.atan2(
        sum(base_normal[i] * downhill[i] for i in range(3)),
        sum(base_normal[i] * normal[i] for i in range(3)),
    )


def _box_runner(
    incline_deg: float,
    horizon_s: float,
    *,
    flat_start: bool,
    corners: int,
    tilt_perturbation_rad: float = 0.0,
    observable: str = "tilt",
):
    """``observable``取``tilt``（末态倾角）或``pitch_rate``（体系``ω_y``）。

    **摇摆那一档必须读``pitch_rate``**：末态倾角里**平衡位形那一份是常数**
    （实测3.3675e-4 rad），而要测的瞬态在它上面只有1e-5量级，
    于是相对精度被那个常数吃掉三个数量级——实测逐档差直接掉到1e-12、
    落在舍入地板上，报出来的"阶"全是噪声。
    ``ω_y``在平衡位形上**恒为零**，于是它读到的全是瞬态本身。
    **观测量选得不对，扫多少档都白扫。**
    """
    theta = math.radians(incline_deg)
    normal, _ = _incline_frame(theta)
    points = box_corner_points_mm(BOX["half_extents_mm"])[:corners]
    inertia = _box_inertia()
    weight = BOX["mass_kg"] * GRAVITY / 1000.0

    def run(dt_s: float) -> float:
        force, torque = support_points_plane_callbacks(
            support_points_body_mm=points,
            plane_point_mm=(0.0, 0.0, 0.0),
            plane_normal=normal,
            normal_stiffness_n_per_mm=BOX["normal_stiffness_n_per_mm"],
            tangential_stiffness_n_per_mm=BOX["tangential_stiffness_n_per_mm"],
            friction_coefficient=BOX["friction_coefficient"],
            gravity_world_n=(0.0, 0.0, -weight),
            normal_damping_n_s_per_mm=BOX["normal_damping_n_s_per_mm"],
        )
        if flat_start:
            state = _box_flat_state(theta)
        else:
            state = _box_settled_state(theta, tilt_perturbation_rad)
        final, _ = integrate_free_flight(
            RK4_BODY,
            state=state,
            inertia=inertia,
            dt_s=dt_s,
            steps=int(round(horizon_s / dt_s)),
            force_world_n=force,
            torque_body_nmm=torque,
        )
        if observable == "pitch_rate":
            return final.block("angular_velocity_body_rad_per_s")[1]
        return _box_tilt(final.vector, theta)

    return run


def _assert_ladder_is_inside_the_declared_bound(name: str, ladder: Sequence[float]) -> None:
    """最粗那一档必须在丙1那条步长上限之内。**越界那一档不是"误差大"，是发散。**"""

    if name.startswith("rolling_ball"):
        scalar = 0.4 * BALL["mass_kg"] * BALL["radius_mm"] ** 2
        bound = contact_dynamics_step_bound(
            support_points_body_mm=((0.0, 0.0, -BALL["radius_mm"]),),
            mass_kg=BALL["mass_kg"],
            inertia_body_kg_mm2=(
                (scalar, 0.0, 0.0), (0.0, scalar, 0.0), (0.0, 0.0, scalar)
            ),
            normal_stiffness_n_per_mm=BALL["normal_stiffness_n_per_mm"],
            tangential_stiffness_n_per_mm=BALL["tangential_stiffness_n_per_mm"],
            plane_normal_body=(0.0, 0.0, 1.0),
        )
    else:
        inertia = _box_inertia()
        bound = contact_dynamics_step_bound(
            support_points_body_mm=box_corner_points_mm(BOX["half_extents_mm"])[:4],
            mass_kg=BOX["mass_kg"],
            inertia_body_kg_mm2=inertia.inertia_body_kg_mm2,
            normal_stiffness_n_per_mm=BOX["normal_stiffness_n_per_mm"],
            tangential_stiffness_n_per_mm=BOX["tangential_stiffness_n_per_mm"],
            normal_damping_n_s_per_mm=BOX["normal_damping_n_s_per_mm"],
            plane_normal_body=(0.0, 0.0, 1.0),
        )
    coarsest = max(ladder)
    if coarsest >= bound.step_bound_s:
        raise ConvergenceOrderError(
            f"{name}最粗那档{coarsest!r}越过了声明的步长上限{bound.step_bound_s!r}"
            f"（当家的是{bound.governed_by}）——越界那一档给的不是"
            "『误差大一点』而是发散，整条阶梯拟合出来的阶会是假数"
        )


#: `(名字, 构造runner, 本机小规模的最粗步长/时长, full档的最粗步长/时长)`。
#: **本机那一档是"跑得通"，不是"最终结果"**——细端不够细时渐近区没进去。
PROBLEMS: dict[str, dict] = {
    "rolling_ball_roll": {
        "make": lambda horizon: _ball_runner(0.3, horizon),
        "local": (1.25e-5, 4.0e-3),
        "full": (1.25e-5, 2.0e-2),
        "note": "无滑支（μ = 0.3 > μc = 0.104）",
    },
    "rolling_ball_slide": {
        "make": lambda horizon: _ball_runner(0.05, horizon),
        "local": (1.25e-5, 4.0e-3),
        "full": (1.25e-5, 2.0e-2),
        "note": "滑动支（μ = 0.05 < μc），切向力全程饱和在摩擦锥上",
    },
    "box_rocking": {
        "make": lambda horizon: _box_runner(
            20.0,
            horizon,
            flat_start=False,
            corners=4,
            tilt_perturbation_rad=1.0e-4,
            observable="pitch_rate",
        ),
        "local": (5.0e-5, 4.0e-3),
        "full": (5.0e-5, 1.6e-2),
        "note": "稳定侧绕平衡位形的一次阻尼摇摆，观测量取体系`ω_y`，"
        "**四个角全程承载、没有接触集切换**——这一档量的是力矩装配这条路本身的阶",
    },
    "box_tipping_topple": {
        "make": lambda horizon: _box_runner(
            32.0, horizon, flat_start=True, corners=8
        ),
        "local": (2.0e-4, 6.0e-2),
        "full": (2.0e-4, 1.5e-1),
        "note": "失稳侧（tanθ > w/h），**承载点在扫描过程中会换**——"
        "接触集切换让右端项只有C0，阶因此不该按四阶读",
    },
}


def run_problem(name: str, *, levels: int, scale: str) -> OrderEstimate:
    spec = PROBLEMS[name]
    coarsest, horizon = spec[scale]
    ladder = _ladder(coarsest, levels)
    _assert_ladder_is_inside_the_declared_bound(name, ladder)
    assert_horizon_is_an_integer_number_of_steps(name, ladder, horizon)
    runner = spec["make"](horizon)
    estimate = order_by_richardson(f"{name}（{spec['note']}，T = {horizon:g} s）", runner, ladder)
    return estimate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="步长收敛阶扫描器（决策0087丙4）")
    sub = parser.add_subparsers(dest="mode", required=True)
    verify = sub.add_parser("verify", help="判据本身对已知阶的解析问题验一次")
    verify.add_argument("--levels", type=int, default=4)
    run = sub.add_parser("run", help="扫一条案例的收敛阶")
    run.add_argument("--problem", default="all", choices=("all", *PROBLEMS))
    run.add_argument("--levels", type=int, default=4)
    run.add_argument("--scale", default="local", choices=("local", "full"))
    args = parser.parse_args(argv)

    if args.mode == "verify":
        failures = 0
        for _name, formal, truth_side, richardson in verify_estimator(args.levels):
            print(truth_side.render())
            print(richardson.render())
            for estimate in (truth_side, richardson):
                deviation = abs(estimate.asymptotic_order - formal)
                verdict = "OK" if deviation <= 0.15 else "**偏离形式阶**"
                print(
                    f"  形式阶 {formal}，{estimate.method}给 "
                    f"{estimate.asymptotic_order:.4f}（偏差{deviation:.4f}）{verdict}"
                )
                failures += deviation > 0.15
            print()
        if failures:
            print(f"判据自验失败{failures}处——**在拿它去量别的东西之前先修它**")
            return 1
        print("判据自验通过：三个形式阶（1／1／2）两条路径都对上了")
        return 0

    names = tuple(PROBLEMS) if args.problem == "all" else (args.problem,)
    print(f"# 规模档 {args.scale}、{args.levels}档步长")
    if args.scale == "local":
        print("# **本机小规模、非最终**：细端不够细时渐近区没进去，真值要上master跑")
    for name in names:
        print(run_problem(name, levels=args.levels, scale=args.scale).render())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
