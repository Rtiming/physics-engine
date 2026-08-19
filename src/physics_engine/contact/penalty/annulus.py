"""罚接触族：`annulus`。2026-08-19从`penalty.py`拆出（见`__init__.py`）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.autodiff import Jet1, Jet2, ad_cos, ad_dot, ad_norm, ad_sin
from physics_engine.contact.errors import ContactError
from physics_engine.contact.layout import NORMAL_UNIT_TOLERANCE
from physics_engine.contact.penalty._edges import (
    EDGE_FRAME_TOLERANCE,
    EDGE_WIDTH_MIN_LENGTH,
    _edge_jets,
)
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.state import State


@dataclass(frozen=True)
class PenaltyAnnulusLimit:
    """环带限位面——**法兰内环面对带材边缘的单边接触**（蹭边）。

    决策0062轨道甲第二片，守能力位S6.6。

    记``d = x − p``、``s = d·a``（轴向坐标）、``ρ = |d − s·a|``（径向距离）。
    带材边缘点在``x + e·a``处（``e``是**带符号**的半宽偏移），其轴向坐标是
    ``s_e = s + e``，**而径向距离与中心线的完全相同**（沿轴平移不改变到轴的距离）。

    限位面用``limit``（面的轴向位置）加``inward``（**朝里是哪一边**，取``±1``）声明：

        inward = +1：边缘必须满足 s_e ≤ limit
        inward = −1：边缘必须满足 s_e ≥ limit

    合起来``g = inward·(limit − s_e)``。``g < 0``即**蹭上了**。

    ## ``inward``为什么必须是独立字段——这是端到端跑出来的一个真bug

    第一版把方向编码在``limit``的**符号**里（"正号那片管上侧、负号那片管下侧"），
    看起来省一个字段。**2026-08-17的端到端装配当场打红**：收线盘排线横动到9 mm时，
    下侧法兰的位置变成``9 − 8.5 = +0.5``——**符号翻了**，那一片被当成上侧法兰，
    于是判据方向反了、**蹭边力凭空归零**（横动7 mm与8 mm都算得出2.46 N与7.44 N，
    唯独9 mm给0）。

    病根：**位置的符号与朝向是两件事**，只在几何恰好跨过原点两侧时才碰巧一致。
    任何一次平移都会拆散它们，而收线盘横动就是一次平移。

    这条只有端到端装配才发现得了——单元门里的构型永远是槽心在原点的，
    那里两者恒等。**它是"端到端跑一次"这件事本身的价值证明。**

    ## 为什么不改spec/11的形状词汇（0034第四节重开后的裁决）

    0034把"法兰轴向尺寸"登记成spec/11缺口并维持失败关闭，触发条件写的是
    "WDS碰撞预演批次或case2给出带法兰导轮的书面需求"。**那个条件2026-08-17到达**
    （用户点名"带材蹭边"），0062第三节重开并重新裁决：**仍不改词汇**。

    理由是牵引来了、而它要的东西不在词汇层：蹭边要的是法兰的**内环面**，
    而`modelgen.generate_spool`按0032已经把带法兰带盘**精确分解**为
    `barrel`＋`flange_low`＋`flange_high`三件独立`FiniteCylinder`。
    内环面就是``s = ±W/2``这张平面**限制在环带``ρ ∈ [R_筒, R_法兰]``上**——
    既有原语已经表达得了。新增的词汇在**接触侧**，就是本类。

    ``geometry.mass_properties``对带`flange_outer_radius_mm`的圆柱**维持失败关闭**：
    那条缺口问的是复合体的质量分布，而接触求解不需要它，**它的触发条件没有到来**。

    ## 单边：这是本项与半空间**唯一但要命**的差别

    两片法兰各是一个独立的限位面。**一片被顶住时另一片必须一个牛顿都不出**——
    带材不可能同时贴住两侧还各受一个法向力，除非槽宽比带宽还窄，
    而那是几何声明本身就错了（构造时失败关闭）。

    互补条件``g > 0 ⟹ f ≡ 0``在这里是**零容差**判据，不是"很小"。

    ## Hessian没有几何刚度：``H = k·(a ⊗ a)``，就这一块

    ``s_e``是位置的**线性**函数，所以``g``也是，``∂²g/∂x² = 0``。
    与`PenaltyNormalContact`（常法向半空间）同构，
    **与`PenaltyCylinderContact`不同**——那里``g = ρ − R``里的``ρ``是非线性的，
    于是多出一块周向softening。**别照抄圆柱那一项的Hessian**。

    ## 环带判据是活动条件，不是能量的一部分

    ``ρ ∉ [inner, outer]``时法兰在那里不存在（带材还没绕到法兰的径向范围里，
    或者已经绕过了法兰外径）。与`PenaltyCylinderContact`的轴向硬切同源：
    力在环带边界上跳，**所以本项给出``radial_distance_mm``让案例为它设门**。

    ## 扭转：从"一条声明"变成一个可选的第二条路（2026-08-18，决策0088丁2）

    **2026-08-17到2026-08-18之间，本类这一段写的是一条声明**："边缘点由``x + e·a``
    生成，即假定带材的宽度方向平行于轴、材料标架不绕切线转……一旦有扭转，
    边缘点位置就错了``(w/2)·sin(扭角)``"。扭转自由度本身早已由`rod`落地（0065）、
    槽壁挡扭转也已接上（0072），**只差这条接线**——能力位S6.6的``missing``原话是
    "这条接线没有走到边缘点的位置生成上"。丁2走的就是这条。

    ### 形制：``edge_twists``是与``faces``等长的一列，逐face可为``None``

    一条不为``None``的项是6元组

        (γ_左的全局下标, γ_右的全局下标, d1_左, d2_左, d1_右, d2_右)

    带宽方向取**相邻两条边的材料帧合成再归一**（平分线），与`rod.PenaltyGrooveWall`
    第3.1节同式：

        m2(e) = −sin(γ_e)·d1[e] + cos(γ_e)·d2[e]
        m̂2    = (m2(左) + m2(右)) / |m2(左) + m2(右)|
        q     = x + offset·m̂2                        ← 边缘点

    **归一化不可省**：两条单位矢量的和长度是``cos(Δγ/2)``，不归一等于让半宽
    随扭角悄悄缩水。

    ### 一条**顺带被修掉**的东西：径向距离原来也不对

    无扭转那一路的``ρ``用**中心线**算，理由写在`_frame`里："沿轴平移不改变
    到轴的距离"。**扭转把这条理由拆了**——偏移方向一旦离开轴，边缘点的``ρ``
    就与中心线的不同。有扭转那一路因此按**边缘点自己**算``ρ``。
    这不是丁2要做的事，是做的过程中发现原来那条注释的适用条件被扭转破坏了。

    ### 逐位退化是本片的分辨力

    ``edge_twists``为空（默认）时走的是**原来那串代码，一个字节没动**。
    另外，声明了扭转但``γ ≡ 0``且两条边的``d2``都取``a``时，
    新路径与旧路径给出**逐位相同**（`float.hex()`）的间隙、力与状态——
    因为``m̂2 = (0·d1 + 1·a + 0·d1 + 1·a)/2 = a``在IEEE-754下是精确的。
    **这条只在轴对齐的构型上是逐位的**：一般轴向下两条路的求和次序不同，
    差一个ulp量级。门判的是前者，`tests/test_contact_annulus_twist.py`写明了这一点。

    ### 边界：它仍然没有边缘摩擦，也仍然看不见板屈曲

    本片只把**位置**接上了扭转。壁面/法兰面上的切向本构没有（与
    `PenaltyGrooveWall`同一条边界），杆模型原理上看不见边缘载荷下的板屈曲
    （WDS research/04第5.3节）也没有变。
    """

    name: str = "annulus_limit"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点索引, 轴上一点mm, 轴单位方向, 环带内半径mm, 环带外半径mm,
    #:  限位面的轴向位置mm, 朝里方向±1, 带符号的边缘偏移mm, 罚刚度N/mm)
    faces: tuple[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            float,
            float,
            float,
            float,
        ],
        ...,
    ] = ()
    #: 与``faces``等长（或为空＝全部无扭转）。每项是``None``或
    #: ``(γ_左下标, γ_右下标, d1_左, d2_左, d1_右, d2_右)``。见类docstring。
    edge_twists: tuple[
        tuple[
            int,
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.faces:
            raise ContactError("annulus_limit needs at least one face")
        for node, point, axis, inner, outer, limit, inward, offset, stiffness in self.faces:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(
                    f"annulus limit node index must be a nonnegative int: {node!r}"
                )
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"annulus axis point must be a finite 3-vector: {point!r}")
            if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
                raise ContactError(f"annulus axis must be a finite 3-vector: {axis!r}")
            norm = math.sqrt(sum(component * component for component in axis))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(
                    f"annulus axis must be a unit vector (|a| = {norm!r})"
                )
            if not (inner >= 0.0 and math.isfinite(inner)):
                raise ContactError(f"annulus inner radius must be finite and >= 0: {inner!r}")
            if not (outer > inner and math.isfinite(outer)):
                raise ContactError(
                    f"annulus outer radius must exceed the inner one: {outer!r} <= {inner!r}"
                    " —— 环带塌成一条线时法兰的径向范围为零，那不是一片法兰"
                )
            if not math.isfinite(limit):
                raise ContactError(f"annulus limit must be finite: {limit!r}")
            if inward not in (1.0, -1.0):
                raise ContactError(
                    f"annulus inward must be exactly +1.0 or -1.0: {inward!r} —— "
                    "**朝向是一条声明，不是从limit的符号推出来的**"
                    "（那条推断在几何平移过原点时失效，见类docstring）"
                )
            if not math.isfinite(offset):
                raise ContactError(f"edge offset must be finite: {offset!r}")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")
        self._check_edge_twists()

    def _check_edge_twists(self) -> None:
        """``edge_twists``的失败关闭。空表示全部无扭转，**那是默认也是旧行为**。"""

        if not self.edge_twists:
            return
        if len(self.edge_twists) != len(self.faces):
            raise ContactError(
                f"edge_twists must be empty or as long as faces: "
                f"{len(self.edge_twists)} vs {len(self.faces)} —— "
                "**逐face一项**，短一项等于让某一片法兰静默退回无扭转"
            )
        for order, entry in enumerate(self.edge_twists):
            if entry is None:
                continue
            if len(entry) != 6:
                raise ContactError(
                    f"edge_twists[{order}] must be (γ_left, γ_right, d1_l, d2_l, d1_r, d2_r)"
                )
            left, right, d1_left, d2_left, d1_right, d2_right = entry
            for label, index in (("left", left), ("right", right)):
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ContactError(
                        f"edge_twists[{order}] {label} twist index must be a "
                        f"nonnegative int: {index!r}"
                    )
            node = self.faces[order][0]
            if left in (3 * node, 3 * node + 1, 3 * node + 2) or right in (
                3 * node, 3 * node + 1, 3 * node + 2
            ):
                raise ContactError(
                    f"edge_twists[{order}]的γ下标撞上了节点{node}自己的位置块"
                    f"（{3 * node}—{3 * node + 2}）—— "
                    "**那会让同一个自由度在局部模板里出现两次**，"
                    "5×5的jet随即把它的二阶项加两遍"
                )
            if left == right:
                raise ContactError(
                    f"edge_twists[{order}]的两个γ下标相同（{left}）—— "
                    "一个内顶点夹在**两条**边之间，两条边共用一个扭角"
                    "等于声明这两条边永远一起转，那不是杆的自由度划分"
                )
            for label, frame in (
                ("d1_left", d1_left), ("d2_left", d2_left),
                ("d1_right", d1_right), ("d2_right", d2_right),
            ):
                if len(frame) != 3 or not all(math.isfinite(v) for v in frame):
                    raise ContactError(
                        f"edge_twists[{order}] {label} must be a finite 3-vector: {frame!r}"
                    )
                norm = math.sqrt(sum(v * v for v in frame))
                if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                    raise ContactError(
                        f"edge_twists[{order}] {label}不是单位矢量（|d| = {norm!r}）—— "
                        "不归一化等于让半宽悄悄乘上|d|，而调用方以为自己给的是offset"
                    )
            for label, (first, second) in (
                ("left", (d1_left, d2_left)), ("right", (d1_right, d2_right)),
            ):
                dot = sum(a * b for a, b in zip(first, second, strict=True))
                if abs(dot) > EDGE_FRAME_TOLERANCE:
                    raise ContactError(
                        f"edge_twists[{order}] {label}的材料帧不正交"
                        f"（d1·d2 = {dot!r}）—— **它不是一个标架**，"
                        "而``m2 = −sin γ·d1 + cos γ·d2``只在正交时才是单位矢量"
                    )
            if self.faces[order][7] == 0.0:
                raise ContactError(
                    f"faces[{order}]的边缘偏移是零而edge_twists[{order}]声明了扭转—— "
                    "偏移为零时边缘点退回中心线、**本项对γ的依赖当场消失**，"
                    "而它仍会安静地算出一个法向接触。"
                    "与`rod.PenaltyGrooveWall`的``offset = 0``拒收同一条理由"
                )

    def _twist_of(self, order: int):
        """第``order``片法兰的扭转声明，没有就是``None``。"""

        return self.edge_twists[order] if self.edge_twists else None

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _, _, _, _, _, _ in self.faces) + 1

    @staticmethod
    def _frame(
        vector: tuple[float, ...],
        node: int,
        point: tuple[float, float, float],
        axis: tuple[float, float, float],
        limit: float,
        inward: float,
        offset: float,
    ) -> tuple[float, float, float]:
        """返回``(间隙g, 边缘轴向坐标s_e, 径向距离ρ)``。

        ``ρ``用**中心线**算：沿轴平移不改变到轴的距离，边缘点与中心线的``ρ``相同。
        这不是近似，是``|d − (d·a)a|``对``d → d + e·a``不变。
        """

        base = 3 * node
        delta = tuple(vector[base + component] - point[component] for component in range(3))
        axial = sum(delta[component] * axis[component] for component in range(3))
        radial = tuple(delta[component] - axial * axis[component] for component in range(3))
        distance = math.sqrt(sum(component * component for component in radial))
        edge_axial = axial + offset
        return inward * (limit - edge_axial), edge_axial, distance

    @staticmethod
    def _is_active(gap: float, distance: float, inner: float, outer: float) -> bool:
        """活动条件是**两条**：顶上了，且边缘确实在法兰的径向范围内。"""

        return gap < 0.0 and inner <= distance <= outer

    # ---------------------------------------------------------------- 扭转 --
    # 下面五个方法是决策0088丁2新增的**第二条路**。无扭转那一路上面一个字未动。

    def _twist_indices(self, order: int) -> tuple[int, ...]:
        """局部模板的5个全局下标：``[x_i(3), γ_左, γ_右]``。"""

        node = self.faces[order][0]
        twist = self.edge_twists[order]
        assert twist is not None
        return (3 * node, 3 * node + 1, 3 * node + 2, twist[0], twist[1])

    @staticmethod
    def _bisector(local, twist):
        """带宽方向``m̂2``：**归一化的平分线**。

        表达式与`rod.PenaltyGrooveWall._width_direction`**逐字相同**——
        `tests/test_contact_annulus_twist.py`有一条门拿两边的``m̂2``逐位对拍
        （`float.hex()`），重复实现的漂移由那条门守，不由约定守。

        它与`AnisotropicRodBending._curvatures`里的``0.5·(m2_l + m2_r)``**不同**：
        那里要的是对偶元平均（已经积分过的离散量），这里要的是一个**方向**。
        两处都不许照抄对方（0072第3.1节原文）。
        """

        _, _, d1_left, d2_left, d1_right, d2_right = twist
        cos_left, sin_left = ad_cos(local[3]), ad_sin(local[3])
        cos_right, sin_right = ad_cos(local[4]), ad_sin(local[4])
        total = tuple(
            -sin_left * d1_left[a] + cos_left * d2_left[a]
            - sin_right * d1_right[a] + cos_right * d2_right[a]
            for a in range(3)
        )
        length = ad_norm(total)
        raw = length.value if isinstance(length, (Jet1, Jet2)) else length
        if raw < EDGE_WIDTH_MIN_LENGTH:
            raise ContactError(
                f"两条边的材料帧几乎反向（|m2_l + m2_r| = {raw!r}）—— "
                "带宽方向的平分线在这里由舍入决定，那不是方向是噪声"
            )
        return tuple(component / length for component in total)

    def _twisted_gap(self, vector: tuple[float, ...], order: int, derivative: int):
        """有扭转那一路的``g``，按``derivative``给出float／Jet1／Jet2。

        ``q = x + offset·m̂2``、``s_e = (q − p)·a``、``g = inward·(limit − s_e)``。
        **求和一律走`ad_dot`（顺序累加）**，于是三个阶给出的``g.value``
        逐位相同——spec/12第3.1节那条"融合路径与单独调`energy`逐字节相同"
        在这里是按构造成立的，不是量出来的。
        """

        node, point, axis, _, _, limit, inward, offset, _ = self.faces[order]
        twist = self.edge_twists[order]
        local = _edge_jets(vector, self._twist_indices(order), derivative)
        width = self._bisector(local, twist)
        delta = tuple(
            local[a] + offset * width[a] - point[a] for a in range(3)
        )
        edge_axial = ad_dot(delta, axis)
        return inward * (limit - edge_axial), edge_axial, delta, width

    def _twisted_frame(self, vector: tuple[float, ...], order: int):
        """有扭转那一路的``(g, s_e, ρ_e, m̂2)``，**全部是裸float**。

        ``ρ_e``按**边缘点自己**算，不按中心线——见类docstring那一段：
        无扭转那一路"沿轴平移不改变到轴的距离"这条理由被扭转拆了。
        它只进活动判定，不进能量，所以这里不需要它的导数（也就没有
        ``ρ = 0``处`ad_sqrt`失败关闭那个新分支）。
        """

        axis = self.faces[order][2]
        gap, edge_axial, delta, width = self._twisted_gap(vector, order, 0)
        radial = tuple(delta[a] - edge_axial * axis[a] for a in range(3))
        distance = math.sqrt(sum(component * component for component in radial))
        return gap, edge_axial, distance, width

    def _face_frame(self, vector: tuple[float, ...], order: int):
        """统一入口：无扭转走原来那串代码，有扭转走上面那条。"""

        node, point, axis, _, _, limit, inward, offset, _ = self.faces[order]
        if self._twist_of(order) is None:
            return self._frame(vector, node, point, axis, limit, inward, offset)
        gap, edge_axial, distance, _ = self._twisted_frame(vector, order)
        return gap, edge_axial, distance

    def edge_width_direction(
        self, state: State
    ) -> tuple[tuple[float, float, float], ...]:
        """诊断面：逐face的**单位带宽方向**（无扭转那一路就是轴``a``）。

        "带材扭到哪去了"最直接的观测量。判方向没有``O(1/k)``的穿透误差，
        判位置有——与`rod.PenaltyGrooveWall.width_direction`同一条理由。
        """

        result = []
        for order, entry in enumerate(self.faces):
            if self._twist_of(order) is None:
                result.append(entry[2])
            else:
                result.append(self._twisted_frame(state.vector, order)[3])
        return tuple(result)

    # ------------------------------------------------------------ 能量四件 --

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for order, entry in enumerate(self.faces):
            _, _, _, inner, outer, _, _, _, stiffness = entry
            gap, _, distance = self._face_frame(state.vector, order)
            if self._is_active(gap, distance, inner, outer):
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for order, entry in enumerate(self.faces):
            node, _, axis, inner, outer, _, inward, _, stiffness = entry
            gap, _, distance = self._face_frame(state.vector, order)
            if not self._is_active(gap, distance, inner, outer):
                continue
            if self._twist_of(order) is None:
                #: ``∂g/∂x = −sign(limit)·a``，故``∇U = k·g·(−sign·a)``。
                scale = -stiffness * gap * inward
                base = 3 * node
                for component in range(3):
                    result[base + component] += scale * axis[component]
                continue
            #: 有扭转：``g``经``m̂2``非线性地依赖γ，走一阶jet。
            jet = self._twisted_gap(state.vector, order, 1)[0]
            energy = 0.5 * stiffness * jet * jet
            for slot, index in enumerate(self._twist_indices(order)):
                result[index] += energy.gradient[slot]
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
        """无扭转：``k·(a ⊗ a)``，**没有几何刚度**——``g``是位置的线性函数。

        **有扭转就不是了。** ``g``经``m̂2(γ)``非线性地依赖γ，于是
        ``∂²(½kg²)/∂q²``除了``k·∇g⊗∇g``还多出``k·g·∂²g/∂q²``那一块。
        这一路走`autodiff`的二阶jet，与`rod.PenaltyGrooveWall`同一条裁决
        （0064第4.1节）：归一化＋两次三角函数的二阶链式法则手写一遍，
        是在零实测下押注常数因子。5变量的``Jet2``每次乘除建25项。
        """

        entries: list[tuple[int, int, float]] = []
        for order, entry in enumerate(self.faces):
            node, _, axis, inner, outer, _, _, _, stiffness = entry
            gap, _, distance = self._face_frame(state.vector, order)
            if not self._is_active(gap, distance, inner, outer):
                continue
            if self._twist_of(order) is None:
                base = 3 * node
                for a in range(3):
                    for b in range(3):
                        entries.append((base + a, base + b, stiffness * axis[a] * axis[b]))
                continue
            jet = self._twisted_gap(state.vector, order, 2)[0]
            energy = 0.5 * stiffness * jet * jet
            indices = self._twist_indices(order)
            for a, row in enumerate(indices):
                for b, column in enumerate(indices):
                    entries.append((row, column, energy.hessian[a][b]))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。"""

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for order, entry in enumerate(self.faces):
            node, _, axis, inner, outer, _, inward, _, stiffness = entry
            gap, _, distance = self._face_frame(vector, order)
            if self._is_active(gap, distance, inner, outer):
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    if self._twist_of(order) is None:
                        scale = -stiffness * gap * inward
                        base = 3 * node
                        for component in range(3):
                            gradient[base + component] += scale * axis[component]
                    else:
                        jet = self._twisted_gap(vector, order, 1)[0]
                        energy = 0.5 * stiffness * jet * jet
                        for slot, index in enumerate(self._twist_indices(order)):
                            gradient[index] += energy.gradient[slot]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def rub_force_n(self, state: State) -> tuple[float, ...]:
        """每个限位面上的**蹭边力**``|k·g|``（没蹭上时为0）。

        与半空间项同理：**平衡时它精确等于理论法向力，与罚刚度无关**。
        蹭边事件表判的正是它，所以它是公开面。
        """

        forces = []
        for order, entry in enumerate(self.faces):
            _, _, _, inner, outer, _, _, _, stiffness = entry
            gap, _, distance = self._face_frame(state.vector, order)
            forces.append(
                stiffness * -gap if self._is_active(gap, distance, inner, outer) else 0.0
            )
        return tuple(forces)

    def edge_clearance_mm(self, state: State) -> tuple[float, ...]:
        """每个限位面上的``g``：**离法兰还有多远**，正为未蹭、负为已蹭。

        它是蹭边事件的判据量：一段连续的``g < 0``就是一次蹭边事件。
        **判它而不是判位置**——位置有``O(1/k)``的穿透误差，力与阈值没有。
        """

        return tuple(
            self._face_frame(state.vector, order)[0] for order in range(len(self.faces))
        )

    def radial_distance_mm(self, state: State) -> tuple[float, ...]:
        """每个限位面上边缘点的``ρ``。环带边界上力会跳，门要看得见它在哪。

        **无扭转那一路给的是中心线的``ρ``**（沿轴平移不改变到轴的距离），
        有扭转那一路给的是**边缘点自己的**——扭转把前者那条理由拆了。
        """

        return tuple(
            self._face_frame(state.vector, order)[2] for order in range(len(self.faces))
        )
