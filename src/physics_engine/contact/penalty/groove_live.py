"""罚接触族：`groove_live`。2026-08-19从`penalty.py`拆出（见`__init__.py`）。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.contact.layout import NORMAL_UNIT_TOLERANCE
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.laydown import GrooveCenterline
from physics_engine.state import State

#: 局部模型退化成"直线＋不转的帧"的判据。**判恰好为零，不判"小"**：
#: ``κ_s = κ_n = τ = 0``是调用方的一条**声明**（这一段是直的、不扭的），
#: 而"很小"是一个量，两者不是一件事。恰好为零时本族与`PenaltyGrooveSweep`
#: 走**逐字节相同**的那一串运算（见`PenaltyGrooveSweepLive._station`）。
STRAIGHT_DARBOUX = 0.0

#: 站点重定位的牛顿上限与收敛判据。``F(a) = (x − C(a))·t(a)``的导数恰是``−D``，
#: 所以这是一条**二次收敛**的标量牛顿；20步是给病态构型的余量，正常构型3—5步。
#: 判据写成"步长落到`ARC_SOLVE_TOL_MM`以下"而不是"残差小"——残差的量纲是mm²，
#: 步长的量纲是mm，而本族要的精度是弧长的精度。
ARC_SOLVE_ITERATIONS = 20
ARC_SOLVE_TOL_MM = 1.0e-14


@dataclass(frozen=True)
class PenaltyGrooveSweepLive:
    """扫掠槽壁的**活站点**档——把`PenaltyGrooveSweep`丢掉的``A·t``补回去。

    决策0078，还[0075](../../../docs/decisions/0075_扫掠槽壁接触_两段外倾锥面与冻结帧的代价_20260818.md)
    第四节量出来的那笔账：冻结帧在plans/14实测的九档几何上
    **丢失0.9721%—14.2522%、接触力方向偏0.5569°—8.1113°**，
    而摩擦锥是按力的**方向**算的。

    ## 补的办法不是"往梯度里加一项"——那样会当场把FD门判红，而且判得对

    直接把``A·t``加进`gradient()`而不动`energy()`，梯度就**不再是所实现能量的
    导数**。本仓的FD门（"梯度对能量"）会红，**它应该红**：那时求解器沿着一个
    不是任何势的下降方向走，线搜索与收敛判据全部失去依据。

    所以本族改的是**能量本身**：``g(x)``里的``(p, s, n)``由``x``定，
    每次求值都重新定位最近站点。于是``A·t``**自己就在梯度里**，
    三样之间的关系重新说得清。

    ## 内核仍然只吃数字（0074第二节第1条）——多吃的是**帧的变化率**

    本族**不持有`GrooveCenterline`**，与`PenaltyGrooveSweep`同源。
    它每条壁多拿三个数：站点处的``κ_s``、``κ_n``、``τ``
    （帧沿弧长的变化率，plans/14管``τ``叫"帧扭率"）。由这三个数与站点的三标架，
    局部曲线被**唯一确定**为一条**Darboux矢量恒定**的曲线：

        ω = τ·t − κ_n·s + κ_s·n        （动帧分量恒定 ⟹ ``dω/da = ω × ω = 0``
                                          ⟹ 它在**空间里**也是常矢量）
        t' = ω × t,   s' = ω × s,   n' = ω × n

    即帧绕空间里一根**固定轴**``ω̂``以恒定角速率``|ω|``刚性转动，
    而``C(a) = p + ∫₀^a t(ξ)dξ``有闭式（Rodrigues的积分）。**三件事同时成立**：

    1. 帧在任何``a``上都**严格正交归一**（是一个真转动，不是线性外推后再正交化）；
    2. 曲线在``a = 0``处的``κ_s``、``κ_n``、``τ``**恰好**是给进来的那三个数；
    3. 整条模型曲线上``dκ_s/da = dκ_n/da = dτ/da ≡ 0``——**这一条是刻意的**，
       见下面"Hessian为什么仍然不精确"。

    **这不是把中心线当成螺旋线。** 0075第四节末段警告过"螺旋线把``κ``与``τ``
    锁死成一个比值"，那条警告针对的是拿一条螺旋线当**整条**中心线的金标。
    这里是**局部**模型，三个不变量各自独立取值（``ω``有三个自由分量），
    没有任何比值被锁死。

    ## 三样

    最近点条件``F(x, a) = (x − C(a))·t(a) = 0``定出``a*(x)``（标量牛顿，
    ``∂F/∂a = −D``）。记``u = (x−C)·s(a*)``、``v = (x−C)·n(a*)``：

        g   = w/2 + v·tanα − (σ·u + r)
        D   = 1 − u·κ_s − v·κ_n
        A   = τ·(tanα·u + σ·v) / D
        ∇g  = tanα·n(a*) − σ·s(a*) − A·t(a*)
        H   ≈ k·(∇g ⊗ ∇g)          ← **约等号是本族唯一的近似，见下节**

    ``∇g``那一式**对任何**具备上述不变量的局部曲线都成立（隐函数定理：
    ``∂F/∂x = t``、``∂F/∂a = −D`` ⟹ ``∇a = t/D``；
    ``∂u/∂a = −t·s + v·τ = v·τ``——**这里就是包络定理，它只杀掉``−t·s``那一半**）。
    本族之所以还要把模型钉成"Darboux恒定"，是为了让三个不变量在整条模型曲线上
    是同一组数，于是这一式在``a* ≠ 0``处照样成立。

    ## Hessian为什么仍然不精确——**这是一个裁决，不是一个遗漏**

    本族的模型里``dκ/da = dτ/da ≡ 0``，所以**模型自己的**``∇²g``推得完。
    没有推，理由只有一条：**真实中心线上那两个量不为零，而且今天取不出来。**
    0075第四节第3条实测：`hermite_tangent`＋`reorthonormalised_linear`下
    ``dτ/da``干净（−2.5e-14），而``dκ_s/da``**过一个站点当场变号**
    （±1.402e-4，真值0）——病根是位置插值跨站点只有``C¹``。

    推一个"对模型精确、对真曲线不精确"的Hessian，等于把一条近似**伪装成**精确：
    FD门会全绿，而绿的是模型不是工件。**本仓的纪律是"不知道就说不知道"**，
    所以本族交的是**精确能量＋精确梯度＋Gauss-Newton切线**——
    一对自洽的势与力，加一个明说不精确的切线（inexact Newton）。
    代价是牛顿从二阶掉下来，**那个代价被量成了数**（决策0078第三节）。

    缺的那一块是``k·g·∇²g``：``g``活动时``g < 0``，所以它是一块**负定**贡献，
    Gauss-Newton丢掉它等于用一个**偏刚**的切线——步子偏短，单调而不发散。
    这也是选Gauss-Newton而不是别的近似的理由。

    ## 走出这一段就**失败关闭**，不静默跳段

    每条壁另带一个``arc_limit_mm``：``|a*|``超过它就抛`ContactError`。
    0075登记的触发条件正是"一次牛顿里跨过一整个采样步"——跨过去之后，
    这条壁携带的``(κ_s, κ_n, τ)``是**上一段**的指纹而不是曲线在那里的性质
    （同一节实测：``κ_s``过站点变号）。**这时给出一个数比抛出来更坏**，
    因为那个数看不出是错的。诊断面是`wall_arc_offset_mm`：
    调用方可以在求解之间读它，判断该不该重建。

    ## 与`PenaltyGrooveSweep`的关系：**并存，冻结帧仍是默认**

    `PenaltyGrooveSweep`一个字节没动，0075那10条案例判据与逐位退化门原样绿。
    默认走哪条**由案例声明**，本族不替它裁（理由见决策0078第五节：
    本族每次求值多一条标量牛顿，而绝大多数构型上冻结帧够用——
    **需要接触力方向的案例才付这个代价**）。

    ``κ_s = κ_n = τ = 0``（恰好为零）时本族**不解**最近点，直接走
    与`PenaltyGrooveSweep`**逐字节相同**的那一串运算：帧不转时``u``、``v``
    与``a``无关，那是代数事实不是近似。于是"直中心线＋``tanα = 0``
    ⟹ 与`PenaltyAnnulusLimit`逐位相同"这条退化链条**整条**对本族照样成立。
    """

    name: str = "groove_sweep_live"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点索引, 站点位置mm, 切向t, 带宽方向s, 槽面外法向n, 侧σ=±1,
    #:  槽底半宽mm, 壁外倾tanα, 带材边缘半径mm, 槽深窗下界mm, 槽深窗上界mm,
    #:  κ_s /mm, κ_n /mm, τ /mm, 弧长窗半宽mm, 罚刚度N/mm)
    #:
    #: **没有一个字段有默认值**，与`PenaltyGrooveSweep`同源。多出来的五个
    #: （``t``、``κ_s``、``κ_n``、``τ``、``arc_limit``）尤其不许有默认值：
    #: ``τ = 0``是"这一段不扭"这条**声明**，而不是"没填"。
    walls: tuple[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.walls:
            raise ContactError("groove_sweep_live needs at least one wall")
        for wall in self.walls:
            if len(wall) != 16:
                raise ContactError(
                    f"groove live wall must declare 16 fields, got {len(wall)}: {wall!r} — "
                    "**位置元组少一格不会自己报错**，它只是把后面每一格都读错一位"
                )
            (
                node,
                point,
                tangent,
                width,
                normal,
                side,
                half_width,
                slope,
                radius,
                depth_min,
                depth_max,
                curvature_s,
                curvature_n,
                twist,
                arc_limit,
                stiffness,
            ) = wall
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(
                    f"groove wall node index must be a nonnegative int: {node!r}"
                )
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(
                    f"groove station point must be a finite 3-vector: {point!r}"
                )
            for label, vector in (
                ("tangent", tangent),
                ("width_direction", width),
                ("surface_normal", normal),
            ):
                if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
                    raise ContactError(f"groove {label} must be a finite 3-vector: {vector!r}")
                norm = math.sqrt(sum(component * component for component in vector))
                if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                    raise ContactError(
                        f"groove {label} must be a unit vector (|v| = {norm!r}) — "
                        "不归一化就等于把刚度悄悄乘上|v|²，而调用方以为自己给的是k"
                    )
            #: 三条正交性各判一次。**本族比`PenaltyGrooveSweep`多判两条**：
            #: 那里``t``根本没进来，这里``t``是模型曲线的起始切向，
            #: 它与``s``、``n``不正交时Darboux那一式给的就不是一个转动。
            for left_label, left, right_label, right in (
                ("width_direction", width, "surface_normal", normal),
                ("tangent", tangent, "width_direction", width),
                ("tangent", tangent, "surface_normal", normal),
            ):
                skew = sum(left[axis] * right[axis] for axis in range(3))
                if abs(skew) > NORMAL_UNIT_TOLERANCE:
                    raise ContactError(
                        f"groove {left_label}与{right_label}不正交，内积{skew!r} — "
                        "取错帧在数值上不报任何错，只是把三根轴对调，"
                        "所以这一条在构造期就判（与`laydown.GrooveStation`同源）"
                    )
            #: 手性也判。``s = n × t``是本仓上下一致的约定（`laydown.GrooveStation`
            #: 构造期判的就是它）；左手帧让``τ``与``κ_s``**整体反号**，
            #: 而那两个数正是本族比冻结帧多做的全部事情。
            handed = (
                normal[1] * tangent[2] - normal[2] * tangent[1],
                normal[2] * tangent[0] - normal[0] * tangent[2],
                normal[0] * tangent[1] - normal[1] * tangent[0],
            )
            chirality = math.sqrt(
                sum((handed[axis] - width[axis]) ** 2 for axis in range(3))
            )
            if chirality > NORMAL_UNIT_TOLERANCE:
                raise ContactError(
                    "groove帧不满足width_direction = cross(surface_normal, tangent)，"
                    f"差{chirality!r} — 左手帧让κ_s与τ整体反号，"
                    "而那两个数正是本族比冻结帧多做的全部事情"
                )
            if side not in (1.0, -1.0):
                raise ContactError(
                    f"groove wall side must be exactly +1.0 or -1.0: {side!r} — "
                    "**哪一侧是一条声明，不是从半宽的符号推出来的**"
                )
            if not (half_width > 0.0 and math.isfinite(half_width)):
                raise ContactError(f"groove half width must be positive: {half_width!r}")
            if not (slope >= 0.0 and math.isfinite(slope)):
                raise ContactError(
                    f"groove wall slope tanα must be finite and >= 0: {slope!r} — "
                    "负外倾是倒扣的槽（越往上越窄），本仓没有任何案例验过那个regime"
                )
            if radius < 0.0 or not math.isfinite(radius):
                raise ContactError(f"groove edge radius must be finite and >= 0: {radius!r}")
            if not math.isfinite(depth_min):
                raise ContactError(
                    f"groove depth window lower bound must be finite: {depth_min!r}"
                )
            if not (depth_max > depth_min and math.isfinite(depth_max)):
                raise ContactError(
                    f"groove depth window must be non-empty: {depth_max!r} <= {depth_min!r}"
                )
            for label, rate in (
                ("curvature_s", curvature_s),
                ("curvature_n", curvature_n),
                ("twist", twist),
            ):
                if not math.isfinite(rate):
                    raise ContactError(f"groove {label} must be finite: {rate!r}")
            if not (arc_limit > 0.0 and math.isfinite(arc_limit)):
                raise ContactError(
                    f"groove arc window half width must be positive: {arc_limit!r} — "
                    "它是'这条壁的局部模型在多长一段上还算数'，"
                    "**零意味着一站都不许动**，那不是一条可用的声明"
                )
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")

    def node_index_bound(self) -> int:
        return max(wall[0] for wall in self.walls) + 1

    # -- 局部模型 ----------------------------------------------------------

    @staticmethod
    def _rotate(
        vector: tuple[float, float, float],
        axis: tuple[float, float, float],
        cosine: float,
        sine: float,
    ) -> tuple[float, float, float]:
        """Rodrigues：把``vector``绕单位轴``axis``转过``(cosine, sine)``所定的角。"""

        cross = (
            axis[1] * vector[2] - axis[2] * vector[1],
            axis[2] * vector[0] - axis[0] * vector[2],
            axis[0] * vector[1] - axis[1] * vector[0],
        )
        projection = sum(axis[component] * vector[component] for component in range(3))
        return tuple(
            cosine * vector[component]
            + sine * cross[component]
            + (1.0 - cosine) * projection * axis[component]
            for component in range(3)
        )

    @classmethod
    def _model(
        cls,
        point: tuple[float, float, float],
        tangent: tuple[float, float, float],
        width: tuple[float, float, float],
        normal: tuple[float, float, float],
        curvature_s: float,
        curvature_n: float,
        twist: float,
        arc: float,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """模型曲线在弧长偏移``arc``处的``(C, t, s, n)``。见类docstring第三节。

            C(a) = p + (sinθa/θ)·t + ((1−cosθa)/θ)·(ê×t) + (a − sinθa/θ)(ê·t)·ê
        """

        darboux = (
            twist * tangent[0] - curvature_n * width[0] + curvature_s * normal[0],
            twist * tangent[1] - curvature_n * width[1] + curvature_s * normal[1],
            twist * tangent[2] - curvature_n * width[2] + curvature_s * normal[2],
        )
        rate = math.sqrt(sum(component * component for component in darboux))
        if rate == 0.0:
            return (
                tuple(point[component] + arc * tangent[component] for component in range(3)),
                tangent,
                width,
                normal,
            )
        axis = tuple(component / rate for component in darboux)
        angle = rate * arc
        cosine, sine = math.cos(angle), math.sin(angle)
        cross = (
            axis[1] * tangent[2] - axis[2] * tangent[1],
            axis[2] * tangent[0] - axis[0] * tangent[2],
            axis[0] * tangent[1] - axis[1] * tangent[0],
        )
        along = sum(axis[component] * tangent[component] for component in range(3))
        swept = sine / rate
        position = tuple(
            point[component]
            + swept * tangent[component]
            + ((1.0 - cosine) / rate) * cross[component]
            + (arc - swept) * along * axis[component]
            for component in range(3)
        )
        return (
            position,
            cls._rotate(tangent, axis, cosine, sine),
            cls._rotate(width, axis, cosine, sine),
            cls._rotate(normal, axis, cosine, sine),
        )

    @classmethod
    def _station(cls, vector: tuple[float, ...], wall: tuple) -> tuple:
        """重定位最近站点。返回``(g, v, σu, a*, ∇g, node, v_min, v_max, k, A)``。

        **``κ_s = κ_n = τ``恰好为零那一档不解最近点**：帧不转时``u``与``v``
        与``a``无关（``t ⟂ s``、``t ⟂ n``是代数事实），于是直接走
        `PenaltyGrooveSweep._frame`**逐字节相同**的那一串运算——
        括号位置``halfwidth − (lateral + radius)``照抄，那是退化逐位门的承重墙。
        """

        (
            node,
            point,
            tangent,
            width,
            normal,
            side,
            half_width,
            slope,
            radius,
            depth_min,
            depth_max,
            curvature_s,
            curvature_n,
            twist,
            arc_limit,
            stiffness,
        ) = wall
        base = 3 * node
        if (
            curvature_s == STRAIGHT_DARBOUX
            and curvature_n == STRAIGHT_DARBOUX
            and twist == STRAIGHT_DARBOUX
        ):
            delta = tuple(vector[base + component] - point[component] for component in range(3))
            lateral = side * sum(delta[component] * width[component] for component in range(3))
            depth = sum(delta[component] * normal[component] for component in range(3))
            halfwidth = half_width + depth * slope
            direction = tuple(
                slope * normal[axis] - side * width[axis] for axis in range(3)
            )
            return (
                halfwidth - (lateral + radius),
                depth,
                lateral,
                0.0,
                direction,
                node,
                depth_min,
                depth_max,
                stiffness,
                0.0,
            )

        target = (vector[base], vector[base + 1], vector[base + 2])
        arc, step = 0.0, math.inf
        for _ in range(ARC_SOLVE_ITERATIONS):
            centre, local_t, local_s, local_n = cls._model(
                point, tangent, width, normal, curvature_s, curvature_n, twist, arc
            )
            delta = tuple(target[component] - centre[component] for component in range(3))
            residual = sum(delta[component] * local_t[component] for component in range(3))
            across = sum(delta[component] * local_s[component] for component in range(3))
            depth = sum(delta[component] * local_n[component] for component in range(3))
            jacobian = 1.0 - across * curvature_s - depth * curvature_n
            if jacobian <= 0.0:
                raise ContactError(
                    f"groove_sweep_live: 最近点条件退化，D = {jacobian!r} <= 0 — "
                    "节点落到局部曲率中心上或它之外，那里'最近站点'不再唯一。"
                    "**这不是数值失效是几何失效**：槽的局部曲率半径已经小于"
                    "节点的截面偏移，再往下算出来的'最近点'是一个最远点"
                )
            step = residual / jacobian
            arc += step
            if abs(step) < ARC_SOLVE_TOL_MM:
                break
        else:
            raise ContactError(
                f"groove_sweep_live: 站点重定位在{ARC_SOLVE_ITERATIONS}步内没有收敛"
                f"（末步{step!r} mm）— **不返回未收敛的站点**，"
                "那会让能量与梯度各自对着不同的a*，于是FD门红得莫名其妙"
            )
        if abs(arc) > arc_limit:
            raise ContactError(
                "groove_sweep_live: 站点重定位走出了声明的弧长窗，"
                f"|a*| = {abs(arc)!r} > {arc_limit!r} mm — "
                "这条壁携带的(κ_s, κ_n, τ)是**上一段**的指纹而不是曲线在那里的性质"
                "（决策0075第四节第3条实测：κ_s过一个站点当场变号）。"
                "**不许静默跳段**：调用方应当重调`groove_sweep_live_walls`"
                "重新线性化，诊断面是`wall_arc_offset_mm`"
            )
        centre, local_t, local_s, local_n = cls._model(
            point, tangent, width, normal, curvature_s, curvature_n, twist, arc
        )
        delta = tuple(target[component] - centre[component] for component in range(3))
        across = sum(delta[component] * local_s[component] for component in range(3))
        lateral = side * across
        depth = sum(delta[component] * local_n[component] for component in range(3))
        halfwidth = half_width + depth * slope
        jacobian = 1.0 - across * curvature_s - depth * curvature_n
        coefficient = twist * (slope * across + side * depth) / jacobian
        direction = tuple(
            slope * local_n[axis] - side * local_s[axis] - coefficient * local_t[axis]
            for axis in range(3)
        )
        return (
            halfwidth - (lateral + radius),
            depth,
            lateral,
            arc,
            direction,
            node,
            depth_min,
            depth_max,
            stiffness,
            coefficient,
        )

    @staticmethod
    def _is_active(gap: float, depth: float, depth_min: float, depth_max: float) -> bool:
        """活动条件仍是**两条**，与`PenaltyGrooveSweep`逐字相同。"""

        return gap < 0.0 and depth_min <= depth <= depth_max

    # -- 三样 --------------------------------------------------------------

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for wall in self.walls:
            gap, depth, _, _, _, _, low, high, stiffness, _ = self._station(state.vector, wall)
            if self._is_active(gap, depth, low, high):
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for wall in self.walls:
            gap, depth, _, _, direction, node, low, high, stiffness, _ = self._station(
                state.vector, wall
            )
            if not self._is_active(gap, depth, low, high):
                continue
            force = stiffness * gap
            base = 3 * node
            for component in range(3):
                result[base + component] += force * direction[component]
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        size = len(state.vector)
        result = [[0.0] * size for _ in range(size)]
        for row, column, value in self.hessian_entries(state, context):
            result[row][column] += value
        return tuple(tuple(row) for row in result)

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        """``k·(∇g ⊗ ∇g)``**用活梯度**——Gauss-Newton，**明说不精确**。

        缺的是``k·g·∇²g``那一块。``g``活动时``g < 0``，所以缺的是一块**负定**贡献，
        本项因此是一个**偏刚**的切线：步子偏短、单调、不发散，只是不再二次收敛。
        掉到几阶是量出来的，见决策0078第三节那张表。

        **为什么不推完**：模型自己的``∇²g``推得完（模型里``dκ/da ≡ 0``），
        但真中心线上``dκ_s/da``今天取不出来（0075第四节第3条：过站点变号）。
        推一个"对模型精确、对工件不精确"的Hessian会让FD门全绿而绿的是模型。
        """

        entries: list[tuple[int, int, float]] = []
        for wall in self.walls:
            gap, depth, _, _, direction, node, low, high, stiffness, _ = self._station(
                state.vector, wall
            )
            if not self._is_active(gap, depth, low, high):
                continue
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    entries.append(
                        (base + a, base + b, stiffness * direction[a] * direction[b])
                    )
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。"""

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for wall in self.walls:
            gap, depth, _, _, direction, node, low, high, stiffness, _ = self._station(
                vector, wall
            )
            if not self._is_active(gap, depth, low, high):
                continue
            total += 0.5 * stiffness * gap * gap
            if gradient is not None:
                force = stiffness * gap
                base = 3 * node
                for component in range(3):
                    gradient[base + component] += force * direction[component]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    # -- 诊断面 ------------------------------------------------------------

    def wall_clearance_mm(self, state: State) -> tuple[float, ...]:
        """每面壁上的``g``：离壁还有多少横移余量。**判它而不是判位置。**"""

        return tuple(self._station(state.vector, wall)[0] for wall in self.walls)

    def wall_depth_mm(self, state: State) -> tuple[float, ...]:
        """每面壁上节点的深度坐标``v``。**槽深窗边界上力会跳，门要看得见它在哪。**"""

        return tuple(self._station(state.vector, wall)[1] for wall in self.walls)

    def wall_arc_offset_mm(self, state: State) -> tuple[float, ...]:
        """每面壁重定位到的``a*``——**"这个站点还算不算数"的诊断面**。

        接近``arc_limit_mm``就该重调`groove_sweep_live_walls`重新线性化。
        **本方法不替调用方裁**：它只把数报出来，与`wall_depth_mm`同源。
        """

        return tuple(self._station(state.vector, wall)[3] for wall in self.walls)

    def wall_force_n(self, state: State) -> tuple[float, ...]:
        """每面壁上的接触力大小``k·|g|·|∇g|``（没顶上时为0）。

        **与`PenaltyGrooveSweep.wall_force_n`差的不止一个``secα``**：那里``|∇g|``
        恒为``secα``，这里``∇g``还带着``−A·t``那一项，于是模长是
        ``sqrt(sec²α + A²)``——**这个差正是0075量出来的那0.97%—14.25%**。
        """

        forces = []
        for wall in self.walls:
            gap, depth, _, _, direction, _, low, high, stiffness, _ = self._station(
                state.vector, wall
            )
            if not self._is_active(gap, depth, low, high):
                forces.append(0.0)
                continue
            scale = math.sqrt(sum(component * component for component in direction))
            forces.append(stiffness * -gap * scale)
        return tuple(forces)

    def wall_force_tilt_deg(self, state: State) -> tuple[float, ...]:
        """接触力方向**相对冻结帧那一档**偏了多少度：``atan2(|A|, secα)``。

        **这是本族存在的全部理由的读数**：0075量出九档几何偏0.5569°—8.1113°，
        而摩擦锥是按力的方向算的。没顶上时为0。
        """

        tilts = []
        for wall in self.walls:
            gap, depth, _, _, _, _, low, high, _, coefficient = self._station(
                state.vector, wall
            )
            if not self._is_active(gap, depth, low, high):
                tilts.append(0.0)
                continue
            slope = wall[7]
            tilts.append(
                math.degrees(math.atan2(abs(coefficient), math.sqrt(1.0 + slope * slope)))
            )
        return tuple(tilts)


def groove_sweep_live_walls(
    centerline: GrooveCenterline,
    nodes: Sequence[tuple[int, Sequence[float]]],
    *,
    half_width_mm: float,
    wall_slope: float,
    edge_radius_mm: float,
    depth_window_mm: tuple[float, float],
    stiffness_n_mm: float,
    frame_probe_mm: float,
    name: str = "groove_sweep_live",
) -> PenaltyGrooveSweepLive:
    """沿中心线扫出**活站点**的两面壁：站点 + 站点处的``(κ_s, κ_n, τ)``。

    与`groove_sweep_walls`同形，多做一件事：在站点处**沿弧长中心差分**
    `GrooveCenterline.sample_at`，取出三个帧变化率。

    ## 探针为什么钉在段内——以及``κ_s``为什么仍然不干净

    ``frame_probe_mm``是中心差分的**全宽**，探针两端被
    `GrooveCenterline.segment_bounds_mm`**夹进最近点所在的那一段**。
    夹段不是精细化，是判据：0075第四节第3条实测，跨一个站点差分时
    ``κ_s``**当场变号**（±1.402e-4，真值0）——那是位置插值跨站点只有``C¹``
    留下的指纹，不是曲线的性质。**跨站点的差分不是"精度差一点"，是符号错。**

    夹进段内之后``τ``干净（同节实测−2.5e-14对真值0），
    而``κ_s``仍带**段内**那一档偏差。**本函数不掩盖这一条**：
    ``κ_s``只经``D = 1 − u·κ_s − v·κ_n``进入结果，而``D``是
    "百分之几量级的修正"上的一个百分之一量级的修正，
    灵敏度被量在决策0078第四节那张表里。

    ``frame_probe_mm``**没有默认值**：探针取多长是采样步的函数，
    而采样步是中心线的性质、不是本函数能猜的（与`CenterlineSemantics`同一条纪律）。
    """

    if not (frame_probe_mm > 0.0 and math.isfinite(frame_probe_mm)):
        raise ContactError(
            f"frame_probe_mm must be positive and finite: {frame_probe_mm!r} — "
            "它是中心差分的全宽，零宽差分不是'不做差分'而是0/0"
        )
    low, high = depth_window_mm
    walls: list[tuple] = []
    for node, position in nodes:
        arc, _ = centerline.nearest_arc_length_mm(position)
        sample = centerline.sample_at(arc)
        segment_low, segment_high = centerline.segment_bounds_mm(arc)
        half = 0.5 * frame_probe_mm
        behind_arc = max(segment_low, sample.arc_length_mm - half)
        ahead_arc = min(segment_high, sample.arc_length_mm + half)
        span = ahead_arc - behind_arc
        if span <= 0.0:
            raise ContactError(
                f"groove_sweep_live_walls: 弧长{arc!r}处夹进段内的探针塌成零宽"
                f"（段[{segment_low!r}, {segment_high!r}]）— "
                "**不跨段差分**是本函数的判据，见docstring"
            )
        behind = centerline.sample_at(behind_arc)
        ahead = centerline.sample_at(ahead_arc)
        tangent_rate = tuple(
            (ahead.tangent[axis] - behind.tangent[axis]) / span for axis in range(3)
        )
        width_rate = tuple(
            (ahead.width_direction[axis] - behind.width_direction[axis]) / span
            for axis in range(3)
        )
        curvature_s = sum(
            tangent_rate[axis] * sample.width_direction[axis] for axis in range(3)
        )
        curvature_n = sum(
            tangent_rate[axis] * sample.surface_normal[axis] for axis in range(3)
        )
        twist = sum(width_rate[axis] * sample.surface_normal[axis] for axis in range(3))
        #: 弧长窗取"到本段两端的距离"里小的那一个。**走出这一段这条壁就不算数了**，
        #: 因为它带的三个不变量是本段的指纹（见类docstring末节）。
        #: 站点恰好落在段端时那个距离是0，此时退回整段长度——
        #: **零窗会让每一次求值都当场抛**，而落在段端是常态不是异常。
        arc_limit = min(
            sample.arc_length_mm - segment_low, segment_high - sample.arc_length_mm
        )
        if arc_limit <= 0.0:
            arc_limit = segment_high - segment_low
        for side in (1.0, -1.0):
            walls.append(
                (
                    node,
                    sample.position_mm,
                    sample.tangent,
                    sample.width_direction,
                    sample.surface_normal,
                    side,
                    half_width_mm,
                    wall_slope,
                    edge_radius_mm,
                    low,
                    high,
                    curvature_s,
                    curvature_n,
                    twist,
                    arc_limit,
                    stiffness_n_mm,
                )
            )
    return PenaltyGrooveSweepLive(name=name, walls=tuple(walls))
