"""罚接触族：`sphere`。2026-08-19从`penalty.py`拆出（见`__init__.py`）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.state import State


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
