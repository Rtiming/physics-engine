"""罚接触族：`cylinder`。2026-08-19从`penalty.py`拆出（见`__init__.py`）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.contact.layout import NORMAL_UNIT_TOLERANCE
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.state import State


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
