"""罚法向接触的五个几何族：半空间、球-球、有限长圆柱侧面、法兰内环面、扫掠槽壁。

五个都只算**法向**，切向在`friction`。五个共享同一条纪律：
罚势的精度地板是``k·ulp(间隙表达式里被相减的那个量)``——半空间那条
"跨六个数量级一个ulp都不动"之所以成立，是因为它那个量恰好是0；
圆柱是``k·ulp(R)``、环带是``k·ulp(W/2)``、扫掠槽壁是``k·ulp(w/2)``。见plans/13。

拆分自原`contact.py`（2026-08-17）——**函数体逐字节未动**。
第五族`PenaltyGrooveSweep`是2026-08-18新增（决策0075），
**前四族一个字节未动**：退化逐位门判的正是它与`PenaltyAnnulusLimit`的逐位相等。
"""

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
class PenaltyNormalContact:
    """罚函数式法向接触（**半空间**）：``U = Σ ½·k·g²``（仅``g < 0``），单位N·mm。

    ``g = (x − p)·n − r``是**间隙**：``n``是半空间的外法向（指向节点该待的那一侧），
    ``p``是面上一点，``r``是节点代表的球半径。``g > 0``分离、``g < 0``穿透。

    ## 半径为什么是**显式参数**而不是让调用方自己偏移平面

    数学上把面沿法向抬高``r``是等价的，但那条路**刚在三球金字塔上绊过一次**：
    球心在高度``R``、面在``z = 0``，间隙算出来是``+R``——**底球被判成悬空**，
    于是切线刚度里出现一个零模态、牛顿当场走不动。
    症状是"求解器坏了"，病根是"调用方以为传的是球、而这里当它是质点"。

    **让半径出现在调用点，那个误会就没有发生的余地。** 质点写``0.0``。

    ## 量纲（本仓已因单位吃过两次亏，所以这里逐个写出来）

    ``k``是N/mm、``g``是mm，故``½kg²``是``N/mm · mm² = N·mm``——
    **直接就是本仓的能量单位，不需要`MM_PER_M`**。
    与`PointLoad`同理、与`UniformGravity`相反（后者拿的是kg与mm/s²）。
    **量纲是算出来的，不是照抄相邻代码抄出来的。**

    ## 这个模型给对了什么、给错了什么（0050第二节的代价，写在实现里）

    平衡时``k·δ = N_理论``，于是：

    * **法向力是精确的**，与``k``无关——``δ = N/k``，``N = k·δ = N``恒成立；
    * **穿透不为零**。**准静态**下``δ = N/k``是``O(1/k)``；
      **瞬态冲击下不是这条**——那时``δ_max = v_in·sqrt(m/k)``即``O(k^(−1/2))``
      （实测``k = 1e5``时准静态式差**1010倍**，见research/13第五节）。

      **上面那条瞬态式设重力在接触段不作用。** 含重力时是

          δ_max = δ_eq + sqrt(δ_eq² + (v_in/ω)²)，  δ_eq = mg/k

      重力修正对"两律之比"是**加性的约+1**，所以它**只在低刚度档要命**——
      而那正是research/13第五节原表用来展示"比值≈1"的那一行，
      **该行真值是表中值的2.40倍**（2026-08-12轨道D三条腿复核：闭式、
      独立RK4、引擎实测，见research/15）。
      `cases/bouncing_ball_restitution`把重力**刻意设为零**并写进了参数表，
      因此不受这条限定影响。
      **两条律各管各的域，瞬态案例的判据不许照抄准静态那条**，
      否则刚度提100倍时判据会松100倍而不是10000倍。
      穿透本身**是模型不是缺陷**，但刚度必须是**输入**不是代码里的魔数。

    换句话说：**位置有``O(1/k)``的误差，力没有误差。**
    这条性质决定了判据该判什么——`cases/`里的门判力与阈值，不判位置。

    ## 光滑性：``C¹``而**不是**``C²``

    ``U``在``g = 0``处值与一阶导都连续（``U = ½kg²``、``U' = kg``，两者在
    ``g → 0⁻``都趋于0），**但二阶导从``k``跳到``0``**。

    后果写明：**牛顿法的残差连续、切线刚度不连续**。活动集在迭代中翻转时，
    线搜索可能在那一步失效——0050第四节登记的正是这条，
    与0029第八节那条强非线性全局化的脆点同源。
    """

    name: str = "normal_contact"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: 半空间：(节点索引, 面上一点mm, 外法向单位矢量, 罚刚度N/mm, 球半径mm)
    #: **半径是必填的**，质点写``0.0``——理由见类docstring第二节。
    planes: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float], float, float],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.planes:
            raise ContactError("normal_contact needs at least one half-space")
        for node, point, normal, stiffness, radius in self.planes:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"contact node index must be a nonnegative int: {node!r}")
            if radius < 0.0 or not math.isfinite(radius):
                raise ContactError(f"contact radius must be finite and nonnegative: {radius!r}")
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"plane point must be a finite 3-vector: {point!r}")
            if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
                raise ContactError(f"plane normal must be a finite 3-vector: {normal!r}")
            norm = math.sqrt(sum(component * component for component in normal))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(
                    f"plane normal must be a unit vector (|n| = {norm!r}) — "
                    "不归一化就等于把刚度悄悄乘上|n|²，而调用方以为自己给的是k"
                )
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _, _ in self.planes) + 1

    @staticmethod
    def _gap_mm(
        vector: tuple[float, ...],
        node: int,
        point: tuple[float, float, float],
        normal: tuple[float, float, float],
        radius: float,
    ) -> float:
        base = 3 * node
        return sum(
            (vector[base + axis] - point[axis]) * normal[axis] for axis in range(3)
        ) - radius

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal, radius)
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal, radius)
            if gap < 0.0:
                force = stiffness * gap
                base = 3 * node
                for axis in range(3):
                    result[base + axis] += force * normal[axis]
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
        """``k·(n ⊗ n)``，仅活动接触。**分离的接触一个非零项都不出**。"""

        entries: list[tuple[int, int, float]] = []
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(state.vector, node, point, normal, radius)
            if gap < 0.0:
                base = 3 * node
                for a in range(3):
                    for b in range(3):
                        entries.append(
                            (base + a, base + b, stiffness * normal[a] * normal[b])
                        )
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。

        这里做到逐字节的方式是"同一串运算同一个次序"，不是"算完再比"——
        两条路各写一遍求和次序，迟早会在某个构型上差一个ulp。
        """

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, point, normal, stiffness, radius in self.planes:
            gap = self._gap_mm(vector, node, point, normal, radius)
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    force = stiffness * gap
                    base = 3 * node
                    for axis in range(3):
                        gradient[base + axis] += force * normal[axis]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def normal_force_n(self, state: State) -> tuple[float, ...]:
        """每个半空间上的法向力大小``N = k·|g|``（分离时为0）。

        **这是本项唯一精确的输出**（见类docstring）：平衡时它等于理论法向力，
        与罚刚度无关。摩擦锥要用的正是它——所以它是公开面而不是内部量。
        """

        return tuple(
            stiffness * -gap
            if (gap := self._gap_mm(state.vector, node, point, normal, radius)) < 0.0
            else 0.0
            for node, point, normal, stiffness, radius in self.planes
        )

__all__ = [
    "PenaltyAnnulusLimit",
    "PenaltyCylinderContact",
    "PenaltyGrooveSweep",
    "PenaltyNormalContact",
    "PenaltySphereContact",
    "groove_sweep_walls",
]


@dataclass(frozen=True)
class PenaltySphereContact:
    """两球之间的罚函数法向接触：``U = Σ ½·k·g²``（仅``g < 0``），单位N·mm。

    ``g = |x_j − x_i| − (r_i + r_j)``是间隙：两球心距减去半径之和。

    ## 与`PenaltyNormalContact`（半空间）的差别：**法向随位形转**

    半空间的法向是常量，所以那个项的Hessian只有``k·(n⊗n)``一块。
    这里法向是``d = (x_j − x_i)/|x_j − x_i|``，**它随位置变**，
    于是Hessian多出一块**几何刚度**：

        H = k·(d⊗d) + (k·g/L)·(I − d⊗d)

    第二块在接触时``g < 0``故**是负的**——横向softening。
    这与`AxialStretch`压缩时的几何刚度同源（0046屈曲案例里的临界载荷正是它给的），
    **漏掉它梯度照样对、平衡点照样对，只有收敛速度与稳定性判据会变**。
    因此本项的Hessian必须被有限差分单独验一次。

    ## 这一项让"多体接触"第一次成立

    此前接触只发生在**节点与固定半空间**之间——固定面不参与自由度。
    本项两端都是自由度，于是Hessian有**跨节点的耦合块**，
    而那是多体接触与单体接触的真正分界。
    """

    name: str = "sphere_contact"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点i, 节点j, 半径之和mm, 罚刚度N/mm)
    pairs: tuple[tuple[int, int, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ContactError("sphere_contact needs at least one pair")
        for i, j, radii_sum, stiffness in self.pairs:
            for index in (i, j):
                #: **两个同门（半空间、粘着）都校验了，只有这里漏了。**
                #: 2026-08-06对抗审核实测：``node = -1``被接受、``node_index_bound()``
                #: 返回1所以装配门放行，而``vector[-3:]``读的正是**接触锚点槽**——
                #: 算出316681 N·mm的能量，全部由历史值来。
                #: **这逐字就是`EnergyRegistry.assert_within_nodes`docstring描述的
                #: 那个失败模式，而那道门只挡上界。**
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ContactError(
                        f"sphere contact node index must be a nonnegative int: {index!r}"
                    )
            if i == j:
                raise ContactError(f"a sphere cannot contact itself: node {i}")
            if not (radii_sum > 0.0 and math.isfinite(radii_sum)):
                raise ContactError(f"radii sum must be positive: {radii_sum!r}")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")

    @classmethod
    def _from_validated_pairs(
        cls, pairs: tuple[tuple[int, int, float, float], ...]
    ) -> PenaltySphereContact:
        """由同包内已验证装配层构造；调用方承担全部``__post_init__``不变量。"""

        term = object.__new__(cls)
        object.__setattr__(term, "name", "sphere_contact")
        object.__setattr__(term, "pairs", pairs)
        return term

    def node_index_bound(self) -> int:
        return max(max(i, j) for i, j, _, _ in self.pairs) + 1

    @staticmethod
    def _pair_state(vector: tuple[float, ...], pair) -> tuple[float, float, tuple[float, ...]]:
        """返回``(间隙g, 心距L, 单位方向d)``。``d``由i指向j。"""

        i, j, radii_sum, _ = pair
        delta = tuple(vector[3 * j + axis] - vector[3 * i + axis] for axis in range(3))
        length = math.sqrt(sum(component * component for component in delta))
        if length == 0.0:
            raise ContactError(
                f"spheres {i} and {j} share a centre — 方向未定义，能量在此不可微"
            )
        return length - radii_sum, length, tuple(c / length for c in delta)

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for pair in self.pairs:
            gap, _, _ = self._pair_state(state.vector, pair)
            if gap < 0.0:
                total += 0.5 * pair[3] * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for pair in self.pairs:
            i, j, _, stiffness = pair
            gap, _, direction = self._pair_state(state.vector, pair)
            if gap < 0.0:
                force = stiffness * gap
                for axis in range(3):
                    result[3 * i + axis] -= force * direction[axis]
                    result[3 * j + axis] += force * direction[axis]
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
        entries: list[tuple[int, int, float]] = []
        for pair in self.pairs:
            i, j, _, stiffness = pair
            gap, length, direction = self._pair_state(state.vector, pair)
            if gap >= 0.0:
                continue
            transverse = stiffness * gap / length
            for a in range(3):
                for b in range(3):
                    outer = direction[a] * direction[b]
                    identity = 1.0 if a == b else 0.0
                    block = stiffness * outer + transverse * (identity - outer)
                    entries.append((3 * i + a, 3 * i + b, block))
                    entries.append((3 * j + a, 3 * j + b, block))
                    entries.append((3 * i + a, 3 * j + b, -block))
                    entries.append((3 * j + a, 3 * i + b, -block))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for pair in self.pairs:
            i, j, _, stiffness = pair
            gap, _, direction = self._pair_state(vector, pair)
            if gap < 0.0:
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    force = stiffness * gap
                    for axis in range(3):
                        gradient[3 * i + axis] -= force * direction[axis]
                        gradient[3 * j + axis] += force * direction[axis]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def contact_force_n(self, state: State) -> tuple[float, ...]:
        """每对的法向接触力``|k·g|``（分离时为0）。与半空间项同理：**平衡时精确**。"""

        forces = []
        for pair in self.pairs:
            gap, _, _ = self._pair_state(state.vector, pair)
            forces.append(pair[3] * -gap if gap < 0.0 else 0.0)
        return tuple(forces)


@dataclass(frozen=True)
class PenaltyCylinderContact:
    """罚函数式法向接触（**有限长圆柱的侧面**）：``U = Σ ½·k·g²``（仅活动），单位N·mm。

    决策0062轨道甲第一片，守能力位S6.5（带材过导向轮的接触）。

    记``d = x − p``（``p``是轴上一点）、``s = d·a``（轴向坐标，``a``是轴单位方向）、
    ``w = d − s·a``（径向矢量）、``ρ = |w|``。则

        g = ρ − (R + r_节点)，   n = w/ρ

    ``g > 0``分离、``g < 0``穿透；``n``是**由轴指向外**的径向单位矢量，恒有``n·a = 0``。

    ## 活动条件是**两条**，不是一条

    ``g < 0``**且**``|s| ≤ half_width``。第二条不是可选的修饰：
    本项只表达**侧面**，节点轴向越出筒宽之后侧面在那里根本不存在，
    继续按``g``出力等于凭空造一个无限长圆柱。

    **这条边界是硬切，力在``|s| = half_width``处从``k|g|``跳到``0``。**
    不平滑不是疏忽——把它做成圆角要引入端面圆环与环面两套几何，
    而那是0062第七节明确不做的。代价如实登记：**节点贴着端沿时牛顿会抖**，
    所以本项给出``axial_clearance_mm``，案例必须为它设门（`cases/`里的门判它，
    不判位置）。**"力会跳"如果没有门看着，就等于一条没有门的分支**（plans/09教训三）。

    ## Hessian：几何刚度**只出现在周向**

    ``∂g/∂x = n``、``∂n/∂x = (P − n⊗n)/ρ``（``P = I − a⊗a``），而
    ``P − n⊗n = t⊗t``，``t = a × n``是周向单位矢量。于是

        H = k·(n⊗n) + (k·g/ρ)·(t⊗t)

    第二块在接触时``g < 0``故**是负的**——**周向**softening，与`PenaltySphereContact`
    的横向softening同源（那里是``(kg/L)(I − d⊗d)``，各向同性的两个横向）。

    **这里只有一个横向而不是两个**，因为沿轴移动不改变``ρ``：
    ``H``在``a``方向上恒为零。这不是近似，是圆柱的几何——
    **它同时是一条可测判据**（沿轴的方向导数必须恰为0），本模块的门判它。

    ## 精度的地板：``k·ulp(R)``——**半空间那条"力精确"在这里不成立**

    `PenaltyNormalContact`记着"跨六个数量级刚度，法向力一个ulp都不动"。
    **那条不能搬过来。** 半空间的间隙是``z − 0``，圆柱的间隙是``ρ − R``——
    后者是两个``O(R)``量相减，**灾难性相消**。于是：

    * 可达残差的地板是``0.5·k·ulp(R)``（2026-08-17实测两半径×五档刚度共10组，
      比值全在``[0.15, 0.48]``，无一超过0.5）；
    * 法向力的可达精度因此是``k·ulp(R)``的**绝对**量，不是相对量。

    绕线机导轮``R = 50 mm``、``k = 1e4 N/mm``时地板是``3.6e-11 N``，
    而链路上要分辨的张力是10—30 N——**够用，但求解器容差必须按它定**。

    **它同时是一条设计约束**：想把残差压到``1e-13``就得把``k``压到``30 N/mm``以下，
    那时穿透``N/k``约0.65 mm。**精度与穿透是同一个旋钮的两头**，
    这一点在半空间上看不出来，因为那里没有``R``。

    ## 轴上奇点：失败关闭

    ``ρ = 0``（节点正好在轴上）时法向**没有定义**，本项当场抛。
    它不是数值噪声：罚接触的穿透量级是``O(N/k)``，穿到轴上意味着模型早已离开
    定义域，此时静默取一个方向比抛更坏——那个方向会被牛顿当成真的。

    近轴处法向的相对精度约``eps·|d|/ρ``，这条写在这里而不是靠使用者猜。
    """

    name: str = "cylinder_contact"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点索引, 轴上一点mm, 轴单位方向, 圆柱半径mm, 轴向半宽mm, 罚刚度N/mm, 节点半径mm)
    #: **节点半径必填**，质点写``0.0``——理由同`PenaltyNormalContact`第二节。
    cylinders: tuple[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            float,
            float,
            float,
            float,
        ],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.cylinders:
            raise ContactError("cylinder_contact needs at least one cylinder")
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(
                    f"cylinder contact node index must be a nonnegative int: {node!r}"
                )
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ContactError(f"cylinder axis point must be a finite 3-vector: {point!r}")
            if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
                raise ContactError(f"cylinder axis must be a finite 3-vector: {axis!r}")
            norm = math.sqrt(sum(component * component for component in axis))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(
                    f"cylinder axis must be a unit vector (|a| = {norm!r}) — "
                    "不归一化会同时改掉轴向投影与径向距离，而调用方以为自己给的是几何"
                )
            if not (radius > 0.0 and math.isfinite(radius)):
                raise ContactError(f"cylinder radius must be positive: {radius!r}")
            if not (half_width > 0.0 and math.isfinite(half_width)):
                raise ContactError(f"cylinder half width must be positive: {half_width!r}")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"penalty stiffness must be positive: {stiffness!r}")
            if node_radius < 0.0 or not math.isfinite(node_radius):
                raise ContactError(
                    f"contact radius must be finite and nonnegative: {node_radius!r}"
                )

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _, _, _, _ in self.cylinders) + 1

    @staticmethod
    def _frame(
        vector: tuple[float, ...],
        node: int,
        point: tuple[float, float, float],
        axis: tuple[float, float, float],
        radius: float,
        node_radius: float,
    ) -> tuple[float, float, float, tuple[float, float, float]]:
        """返回``(间隙g, 轴向坐标s, 径向距离ρ, 径向单位法向n)``。

        ``ρ = 0``时抛——理由见类docstring末节。
        """

        base = 3 * node
        delta = tuple(vector[base + component] - point[component] for component in range(3))
        axial = sum(delta[component] * axis[component] for component in range(3))
        radial = tuple(delta[component] - axial * axis[component] for component in range(3))
        distance = math.sqrt(sum(component * component for component in radial))
        if distance == 0.0:
            raise ContactError(
                f"node {node} sits on the cylinder axis (rho = 0) — 法向没有定义。"
                "罚接触的穿透量级是O(N/k)，穿到轴上说明模型已离开定义域；"
                "此处静默取一个方向会被牛顿当成真的"
            )
        normal = tuple(component / distance for component in radial)
        return distance - (radius + node_radius), axial, distance, normal  # type: ignore[return-value]

    @classmethod
    def _is_active(cls, gap: float, axial: float, half_width: float) -> bool:
        """活动条件是**两条**：穿透且轴向仍在筒宽内。"""

        return gap < 0.0 and abs(axial) <= half_width

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            if self._is_active(gap, axial, half_width):
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, normal = self._frame(
                state.vector, node, point, axis, radius, node_radius
            )
            if self._is_active(gap, axial, half_width):
                force = stiffness * gap
                base = 3 * node
                for component in range(3):
                    result[base + component] += force * normal[component]
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
        """``k·(n⊗n) + (k·g/ρ)·(t⊗t)``，``t = a × n``。**分离的接触一个非零项都不出**。"""

        entries: list[tuple[int, int, float]] = []
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, distance, normal = self._frame(
                state.vector, node, point, axis, radius, node_radius
            )
            if not self._is_active(gap, axial, half_width):
                continue
            circumferential = (
                axis[1] * normal[2] - axis[2] * normal[1],
                axis[2] * normal[0] - axis[0] * normal[2],
                axis[0] * normal[1] - axis[1] * normal[0],
            )
            geometric = stiffness * gap / distance
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    value = stiffness * normal[a] * normal[b] + geometric * (
                        circumferential[a] * circumferential[b]
                    )
                    entries.append((base + a, base + b, value))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。"""

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, normal = self._frame(vector, node, point, axis, radius, node_radius)
            if self._is_active(gap, axial, half_width):
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    force = stiffness * gap
                    base = 3 * node
                    for component in range(3):
                        gradient[base + component] += force * normal[component]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def normal_force_n(self, state: State) -> tuple[float, ...]:
        """每个圆柱上的法向力大小``N = k·|g|``（不活动时为0）。

        与半空间项同理：**平衡时它精确等于理论法向力，与罚刚度无关**。
        绞盘判据要用的正是它，所以它是公开面。
        """

        forces = []
        for node, point, axis, radius, half_width, stiffness, node_radius in self.cylinders:
            gap, axial, _, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            forces.append(
                stiffness * -gap if self._is_active(gap, axial, half_width) else 0.0
            )
        return tuple(forces)

    def axial_clearance_mm(self, state: State) -> tuple[float, ...]:
        """每个圆柱上``half_width − |s|``：**离端沿还有多远**，可正可负。

        本项在``|s| > half_width``处力硬跳到零（类docstring第二节），
        所以案例必须为这个量设门。**给出它，是为了让那条不连续有门看着而不是靠记性。**
        """

        clearances = []
        for node, point, axis, radius, half_width, _, node_radius in self.cylinders:
            _, axial, _, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            clearances.append(half_width - abs(axial))
        return tuple(clearances)

    def radial_distance_mm(self, state: State) -> tuple[float, ...]:
        """每个圆柱上的``ρ``。近轴处法向精度约``eps·|d|/ρ``，门判它才能发现失精。"""

        distances = []
        for node, point, axis, radius, _, _, node_radius in self.cylinders:
            _, _, distance, _ = self._frame(state.vector, node, point, axis, radius, node_radius)
            distances.append(distance)
        return tuple(distances)

    def outward_normal(self, state: State) -> tuple[tuple[float, float, float], ...]:
        """每个圆柱上的径向单位外法向``n``。

        摩擦项要的正是它：`TangentialStickSpring`吃`NormalSource`，
        曲面接触必须给**随位形转**的那一种（0050第四节实测：法向不随位形转时
        一趟预测-修正就够，转时不够）。本方法是绞盘接线的接口。
        """

        normals = []
        for node, point, axis, radius, _, _, node_radius in self.cylinders:
            _, _, _, normal = self._frame(state.vector, node, point, axis, radius, node_radius)
            normals.append(normal)
        return tuple(normals)


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

    ## 无扭转假设——一条声明，不是一个实现细节

    边缘点由``x + e·a``生成，即**假定带材的宽度方向平行于轴**、材料标架不绕切线转。
    带材平贴在筒上时这是精确的；一旦有扭转，边缘点位置就错了``(w/2)·sin(扭角)``。
    扭转是0029登记的欠账，0062第六节第3条与第八节写明触发条件。
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

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, point, axis, inner, outer, limit, inward, offset, stiffness in self.faces:
            gap, _, distance = self._frame(state.vector, node, point, axis, limit, inward, offset)
            if self._is_active(gap, distance, inner, outer):
                total += 0.5 * stiffness * gap * gap
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, point, axis, inner, outer, limit, inward, offset, stiffness in self.faces:
            gap, _, distance = self._frame(state.vector, node, point, axis, limit, inward, offset)
            if self._is_active(gap, distance, inner, outer):
                #: ``∂g/∂x = −sign(limit)·a``，故``∇U = k·g·(−sign·a)``。
                scale = -stiffness * gap * inward
                base = 3 * node
                for component in range(3):
                    result[base + component] += scale * axis[component]
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
        """``k·(a ⊗ a)``，仅活动。**没有几何刚度**——``g``是位置的线性函数。"""

        entries: list[tuple[int, int, float]] = []
        for node, point, axis, inner, outer, limit, inward, offset, stiffness in self.faces:
            gap, _, distance = self._frame(state.vector, node, point, axis, limit, inward, offset)
            if not self._is_active(gap, distance, inner, outer):
                continue
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    entries.append((base + a, base + b, stiffness * axis[a] * axis[b]))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合路径。**能量值必须与单独调`energy`逐字节相同**（spec/12第3.1节）。"""

        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, point, axis, inner, outer, limit, inward, offset, stiffness in self.faces:
            gap, _, distance = self._frame(vector, node, point, axis, limit, inward, offset)
            if self._is_active(gap, distance, inner, outer):
                total += 0.5 * stiffness * gap * gap
                if gradient is not None:
                    scale = -stiffness * gap * inward
                    base = 3 * node
                    for component in range(3):
                        gradient[base + component] += scale * axis[component]
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
        for node, point, axis, inner, outer, limit, inward, offset, stiffness in self.faces:
            gap, _, distance = self._frame(state.vector, node, point, axis, limit, inward, offset)
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
            self._frame(state.vector, node, point, axis, limit, inward, offset)[0]
            for node, point, axis, _, _, limit, inward, offset, _ in self.faces
        )

    def radial_distance_mm(self, state: State) -> tuple[float, ...]:
        """每个限位面上边缘点的``ρ``。环带边界上力会跳，门要看得见它在哪。"""

        return tuple(
            self._frame(state.vector, node, point, axis, limit, inward, offset)[2]
            for node, point, axis, _, _, limit, inward, offset, _ in self.faces
        )




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
