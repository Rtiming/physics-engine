"""罚接触族：`groove`。2026-08-19从`penalty.py`拆出（见`__init__.py`）。"""

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


@dataclass(frozen=True)
class PenaltyGrooveSweep:
    """扫掠槽壁——**沿真实中心线的两段外倾锥面**，每侧壁各是一项单边约束。

    决策0075，兑现0074第六节阶段一（轨A），守plans/15第2.2条那条判据：
    "真实槽几何：槽底宽8.000 mm、**两段外倾锥面**槽壁（现有`PenaltyAnnulusLimit`
    是平面环带，形制对不上）"。

    ## 几何：截面里的两条直线，不是一条带绝对值的曲线

    在最近站点的截面里记``d = x − p``、``u = d·s``（横向，带宽方向）、
    ``v = d·n``（深度，槽面外法向）。半宽随深度线性外张：

        halfwidth(v) = w/2 + v·tanα

    第σ侧（``σ = +1``是``+s``那侧壁、``σ = −1``是``−s``那侧）的间隙是

        g_σ = halfwidth(v) − (σ·u + r)

    ``g < 0``即顶上了，``U = ½k·g²``。**每侧壁是一项，两项各自单边。**

    ## 三处几何形制上的改动，逐条写明为什么

    ### 1. `|u|`不写绝对值，写成左右两个单边壁

    简报给的形式是``g = halfwidth(v) − |u| − r``。``|u|``在``u = 0``处不可微，
    于是``U``只有``C¹``——而`solve.py`第29行申报的适用域原文是
    "``U``二次连续可微、Hessian在解附近正定"。**在槽心那条线上，
    每一个带材节点都恰好落在那个不可微点上**，那不是罕见构型，那是标称构型。

    拆成两个单边壁之后，每一项在自己的活动域里是**多项式**（见下一节），
    ``C^∞``。这与`PenaltyAnnulusLimit`把两片法兰写成两个独立限位面
    是同一条形制，也继承它那条互补纪律：**一侧被顶住时另一侧一个牛顿都不出**。

    ### 2. `max(v, 0)`换成`v`——同一条理由的第二次

    简报给的是``halfwidth(v) = w/2 + max(v, 0)·tanα``。``max(v, 0)``在
    ``v = 0``处不可微，而``v = 0``正是槽底——**带材平躺在槽底时就在那个点上**。
    它会让``halfwidth``只有``C⁰``、``g``只有``C⁰``、``U``只有``C¹``，
    与第1条犯的是同一个错，只是换了一根轴。

    换成线性的``w/2 + v·tanα``等于**把锥面当成一整张平面延拓到槽底以下**，
    与`PenaltyNormalContact`把半空间延拓到面背后是同一件事：
    ``v < 0``那一侧本来就被槽底接触挡着到不了，**延拓的那一段没有物理，
    但它换来的光滑性是求解器明文要的**。

    ### 3. 槽深窗是**活动条件**，不是能量的一部分

    真槽有深度：``v``超出``[depth_min, depth_max]``时这面壁在那儿根本不存在
    （带材已经爬出槽口，或还没进到壁的起始高度）。与`PenaltyAnnulusLimit`的
    环带判据、`PenaltyCylinderContact`的轴向硬切同源——**力在窗边界上跳**，
    所以本项给出`wall_depth_mm`让案例为它设门。

    ## 帧是**输入的数**，不是本项去查的东西——以及这样做丢了什么

    本项的每一条壁都拿到一个**冻结的**站点``(p, s, n)``。沿中心线的扫掠由
    `groove_sweep_walls`在装配期完成：它对每个节点调
    `GrooveCenterline.nearest_arc_length_mm`找最近弧长、`sample_at`取局部帧，
    然后交出一串数。**内核只吃数字**（0074第二节第1条），
    与`PenaltyAnnulusLimit`拿``(轴上一点, 轴方向)``而不是拿一个圆柱对象同构。

    于是``u``与``v``都是``x``的**线性**函数，``g``也是，因此

        ∇g = tanα·n − σ·s        （常矢量）
        ∇²g = 0                  （**恒为零，不是近似为零**）
        H  = k·(∇g ⊗ ∇g)         （仅活动）

    三样输入齐了，且**能量、梯度、Hessian三者互为对方的精确导数**——
    有限差分对拍在机器精度上收敛，没有一条被截断的项藏在里面。

    **代价必须写在明处，这是本类最要紧的一段。** 真实节点动起来时最近弧长
    ``a*(x)``会跟着动，而冻结帧把``∂a*/∂x``那一项丢掉了。它丢了多少是可算的：

        u_a = (x − p)·s' = v·τ,      v_a = (x − p)·n' = −u·τ
        ∇a  = t / D,                 D = 1 − u·κ_s − v·κ_n
        ∇g_精确 = tanα·n − σ·s − (τ/D)·(tanα·u + σ·v)·t
                = ∇g_本项 − A·t

    **包络定理在这里只杀掉了一半**：最近点条件``(x − p)·t = 0``确实让
    ``∂p/∂a``那一项的一阶贡献归零（这就是``u_a``里没有``−t·s``的原因），
    **但它杀不掉帧本身绕切向的转动**——``τ``（帧扭率）把截面坐标
    ``(u, v)``沿弧长转起来，那一项原封不动留着。简报把这条写成
    "∂a/∂x那一项对g的一阶贡献为零"，**实测与推导都表明那句话对距离``|x−p|``
    成立、对``u``与``v``各自不成立**，两者只在``τ = 0``时重合。

    丢掉的那一项**沿切向**，大小是``|A| = |τ|·|tanα·u + σ·v| / D``。
    plans/14实测的真实工件τ最大2.550—6.648 °/mm，代进去是梯度的百分之几量级
    ——**不是可以忽略不计的数，是可以量出来并写在案例页上的数**。
    `cases/groove_sweep_wall`第三条判据量的正是它，量法是数值有限差分
    "活最近点"的间隙、再与本项的常梯度比，**与上面这条闭式互为独立证人**。

    **触发条件**（不许含糊）：当案例需要在一次牛顿里跨过一整个采样步、
    或者需要接触力的切向分量本身进判据时，本项的冻结帧不够用，
    那时按上式把``A·t``补进`gradient`，并把``∇²g``一并推完
    （它会牵进``dκ/da``与``dτ/da``，而那两个量在今天的
    `CenterlineSemantics`下**不可表示**——`reorthonormalised_linear`的帧在站点上只有``C⁰``，
    二阶帧导数是站点处的一个δ。所以补它之前要先动中心线的插值语义）。

    ## 退化：``tanα = 0`` + 直中心线 ⟹ 与`PenaltyAnnulusLimit`**逐位相同**

    此时``∇g = −σ·s``、``g = w/2 − (σ·u + r)``，把``s``当环带的轴、
    ``w/2``当``limit``、``σ``当``inward``、``σ·r``当边缘偏移，
    两项的能量/梯度/Hessian**逐位相同**（判`float.hex()`不判`==`）。
    这不是巧合而是设计：运算的**次序**是照`PenaltyAnnulusLimit`抄的，
    `_frame`里那个``halfwidth − (lateral + radius)``的括号位置不许动——
    改成``(halfwidth − lateral) − radius``数学上一样、**逐位不一样**，
    退化门会当场红。
    """

    name: str = "groove_sweep"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点索引, 站点位置mm, 带宽方向s, 槽面外法向n, 侧σ=±1, 槽底半宽mm,
    #:  壁外倾tanα, 带材边缘半径mm, 槽深窗下界mm, 槽深窗上界mm, 罚刚度N/mm)
    #:
    #: **没有一个字段有默认值**，与`laydown.GrooveCenterline`同源：
    #: 这里每一条都是声明者要拿主意的东西，默认值等于替他拿了主意。
    walls: tuple[
        tuple[
            int,
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
        ],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.walls:
            raise ContactError("groove_sweep needs at least one wall")
        for (
            node,
            point,
            width,
            normal,
            side,
            half_width,
            slope,
            radius,
            depth_min,
            depth_max,
            stiffness,
        ) in self.walls:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(
                    f"groove wall node index must be a nonnegative int: {node!r}"
                )
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"groove station point must be a finite 3-vector: {point!r}")
            for label, vector in (("width_direction", width), ("surface_normal", normal)):
                if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
                    raise ContactError(f"groove {label} must be a finite 3-vector: {vector!r}")
                norm = math.sqrt(sum(component * component for component in vector))
                if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                    raise ContactError(
                        f"groove {label} must be a unit vector (|v| = {norm!r}) — "
                        "不归一化就等于把刚度悄悄乘上|v|²，而调用方以为自己给的是k"
                    )
            skew = sum(width[axis] * normal[axis] for axis in range(3))
            if abs(skew) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(
                    f"groove width_direction与surface_normal不正交，内积{skew!r} — "
                    "取错n与s在数值上不报任何错，只是把横向与深度对调，"
                    "所以这一条在构造期就判（与`laydown.GrooveStation`同源）"
                )
            if side not in (1.0, -1.0):
                raise ContactError(
                    f"groove wall side must be exactly +1.0 or -1.0: {side!r} — "
                    "**哪一侧是一条声明，不是从半宽的符号推出来的**"
                    "（`PenaltyAnnulusLimit`那条端到端bug就是这么来的）"
                )
            if not (half_width > 0.0 and math.isfinite(half_width)):
                raise ContactError(f"groove half width must be positive: {half_width!r}")
            if not (slope >= 0.0 and math.isfinite(slope)):
                raise ContactError(
                    f"groove wall slope tanα must be finite and >= 0: {slope!r} — "
                    "负外倾是倒扣的槽（越往上越窄），本仓没有任何案例验过那个regime，"
                    "**失败关闭而不是替它猜**"
                )
            if radius < 0.0 or not math.isfinite(radius):
                raise ContactError(f"groove edge radius must be finite and >= 0: {radius!r}")
            if not math.isfinite(depth_min):
                raise ContactError(f"groove depth window lower bound must be finite: {depth_min!r}")
            if not (depth_max > depth_min and math.isfinite(depth_max)):
                raise ContactError(
                    f"groove depth window must be non-empty: {depth_max!r} <= {depth_min!r}"
                    " —— 窗塌成一条线时这面壁在任何深度上都不存在，那不是一面壁"
                )
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")

    def node_index_bound(self) -> int:
        return max(wall[0] for wall in self.walls) + 1

    @staticmethod
    def _frame(
        vector: tuple[float, ...],
        node: int,
        point: tuple[float, float, float],
        width: tuple[float, float, float],
        normal: tuple[float, float, float],
        side: float,
        half_width: float,
        slope: float,
        radius: float,
    ) -> tuple[float, float, float]:
        """返回``(间隙g, 深度v, 朝壁的横向坐标σ·u)``。

        **运算次序照`PenaltyAnnulusLimit._frame`抄**：``halfwidth − (lateral + radius)``
        这个括号位置是退化逐位门的承重墙，见类docstring末节。
        """

        base = 3 * node
        delta = tuple(vector[base + component] - point[component] for component in range(3))
        lateral = side * sum(delta[component] * width[component] for component in range(3))
        depth = sum(delta[component] * normal[component] for component in range(3))
        halfwidth = half_width + depth * slope
        return halfwidth - (lateral + radius), depth, lateral

    @staticmethod
    def _direction(
        width: tuple[float, float, float],
        normal: tuple[float, float, float],
        side: float,
        slope: float,
    ) -> tuple[float, float, float]:
        """``∇g = tanα·n − σ·s``。**常矢量**——``g``是位置的线性函数。

        它不是单位矢量：模长``sqrt(1 + tan²α) = 1/cosα``。**不归一化是有意的**，
        因为``g``按定义是**截面里量的横向余量**而不是到壁面的欧氏距离，
        归一化会把``g``的语义从"还差多少横移"改成"还差多远"，
        而判蹭边的是前者。两者差一个``cosα``，案例页第六节写明了这一条。
        """

        return tuple(slope * normal[axis] - side * width[axis] for axis in range(3))

    @staticmethod
    def _is_active(gap: float, depth: float, depth_min: float, depth_max: float) -> bool:
        """活动条件是**两条**：顶上了，且深度确实落在这面壁存在的窗里。"""

        return gap < 0.0 and depth_min <= depth <= depth_max

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, point, width, normal, side, half, slope, radius, low, high, k in self.walls:
            gap, depth, _ = self._frame(
                state.vector, node, point, width, normal, side, half, slope, radius
            )
            if self._is_active(gap, depth, low, high):
                total += 0.5 * k * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, point, width, normal, side, half, slope, radius, low, high, k in self.walls:
            gap, depth, _ = self._frame(
                state.vector, node, point, width, normal, side, half, slope, radius
            )
            if self._is_active(gap, depth, low, high):
                force = k * gap
                direction = self._direction(width, normal, side, slope)
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
        """``k·(∇g ⊗ ∇g)``，仅活动。**没有几何刚度**——``∇²g``恒为零。

        与`PenaltyAnnulusLimit`同构而**与`PenaltyCylinderContact`不同**：
        那里``ρ``是非线性的，于是多出一块周向softening。**别照抄圆柱那一项。**
        """

        entries: list[tuple[int, int, float]] = []
        for node, point, width, normal, side, half, slope, radius, low, high, k in self.walls:
            gap, depth, _ = self._frame(
                state.vector, node, point, width, normal, side, half, slope, radius
            )
            if not self._is_active(gap, depth, low, high):
                continue
            direction = self._direction(width, normal, side, slope)
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    entries.append((base + a, base + b, k * direction[a] * direction[b]))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。"""

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, point, width, normal, side, half, slope, radius, low, high, k in self.walls:
            gap, depth, _ = self._frame(
                vector, node, point, width, normal, side, half, slope, radius
            )
            if self._is_active(gap, depth, low, high):
                total += 0.5 * k * gap * gap
                if gradient is not None:
                    force = k * gap
                    direction = self._direction(width, normal, side, slope)
                    base = 3 * node
                    for component in range(3):
                        gradient[base + component] += force * direction[component]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def wall_force_n(self, state: State) -> tuple[float, ...]:
        """每面壁上的**接触力大小**``k·|g|·|∇g| = k·|g|/cosα``（没顶上时为0）。

        **它与`PenaltyAnnulusLimit.rub_force_n`差一个``1/cosα``，那不是笔误。**
        平面环带的力全在横向，锥面的力是斜的：横向分量仍是``k·|g|``，
        另有一个``k·|g|·tanα``的分量把带材**举出槽**。摩擦锥要的是合力大小，
        所以这里给的是合力；只要横向那一档的案例自己乘回``cosα``。
        """

        forces = []
        for node, point, width, normal, side, half, slope, radius, low, high, k in self.walls:
            gap, depth, _ = self._frame(
                state.vector, node, point, width, normal, side, half, slope, radius
            )
            if not self._is_active(gap, depth, low, high):
                forces.append(0.0)
                continue
            direction = self._direction(width, normal, side, slope)
            scale = math.sqrt(sum(component * component for component in direction))
            forces.append(k * -gap * scale)
        return tuple(forces)

    def wall_clearance_mm(self, state: State) -> tuple[float, ...]:
        """每面壁上的``g``：**离壁还有多少横移余量**，正为未顶、负为已顶。

        **判它而不是判位置**——位置有``O(1/k)``的穿透误差，力与阈值没有。
        """

        return tuple(
            self._frame(state.vector, *wall[0:4], *wall[4:8])[0] for wall in self.walls
        )

    def wall_depth_mm(self, state: State) -> tuple[float, ...]:
        """每面壁上节点的深度坐标``v``。**槽深窗边界上力会跳，门要看得见它在哪。**"""

        return tuple(
            self._frame(state.vector, *wall[0:4], *wall[4:8])[1] for wall in self.walls
        )


def groove_sweep_walls(
    centerline: GrooveCenterline,
    nodes: Sequence[tuple[int, Sequence[float]]],
    *,
    half_width_mm: float,
    wall_slope: float,
    edge_radius_mm: float,
    depth_window_mm: tuple[float, float],
    stiffness_n_mm: float,
    name: str = "groove_sweep",
) -> PenaltyGrooveSweep:
    """沿中心线把每个节点扫成**两面壁**（``+s``侧与``−s``侧各一面）。

    这里就是"扫掠"发生的地方：逐节点调
    `GrooveCenterline.nearest_arc_length_mm`定位、`sample_at`取局部帧，
    把``(p, s, n)``冻结进`PenaltyGrooveSweep.walls`。**节点位置是工件系的**，
    与中心线同系——中心线到世界系的那一层由`laydown.LaydownModel`的位姿负责，
    本函数不替它做，因为"这条槽此刻在哪"是时间线的事而不是接触的事。

    **要重新线性化就再调一次。** 帧在一次求解内是冻结的（理由与代价见
    `PenaltyGrooveSweep`类docstring），节点走过一个采样步以后调用方应当
    重建一遍——与活动集更新同一档节奏，不是本函数替它决定的事。

    两面壁**共用同一个站点**：它们是同一个截面上的两条直线，
    分开找最近点会让左右壁落在不同的弧长上，那时"槽宽"这个词就没有意义了。
    """

    low, high = depth_window_mm
    walls: list[tuple] = []
    for node, position in nodes:
        arc, _ = centerline.nearest_arc_length_mm(position)
        sample = centerline.sample_at(arc)
        for side in (1.0, -1.0):
            walls.append(
                (
                    node,
                    sample.position_mm,
                    sample.width_direction,
                    sample.surface_normal,
                    side,
                    half_width_mm,
                    wall_slope,
                    edge_radius_mm,
                    low,
                    high,
                    stiffness_n_mm,
                )
            )
    return PenaltyGrooveSweep(name=name, walls=tuple(walls))


# ------------------------------------------------ 第六族：活站点的扫掠槽壁 ---
