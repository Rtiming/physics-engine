"""罚接触族：`halfspace`。2026-08-19从`penalty.py`拆出（见`__init__.py`）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.contact.layout import NORMAL_UNIT_TOLERANCE
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
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
