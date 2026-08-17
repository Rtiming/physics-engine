"""切向：库仑return-map与粘着弹簧。

**已知的形制边界**：`coulomb_return_map`是**各向同性径向返回**。
WDS已证伪它在``μ_∥:μ_⊥ = 5:1``时混合角耗散短缺最高60%，且**会系统性偏置横向落位**；
真做各向异性要补摩擦椭圆（η-return），那是另立的一件事，本模块今天不冒充。

拆分自原`contact.py`（2026-08-17）——**函数体逐字节未动**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.contact.errors import ContactError
from physics_engine.contact.layout import (
    NORMAL_UNIT_TOLERANCE,
    REGIME_SEPARATED,
    REGIME_SLIP,
    REGIME_STICK,
)
from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.state import State


@dataclass(frozen=True)
class FrictionOutcome:
    """一次return-map的结果：实际切向力、粘/滑判别、以及**锚点该挪到哪**。

    ``anchor_correction_mm``是滑移那一步锚点要平移的量（粘住时为零矢量）。
    **它是这一步产生的不可逆位移**——把它写回状态，历史就被记住了；
    不写回，下一步会以为自己还粘在原处，于是摩擦力凭空多出一截。
    """

    tangential_force_n: tuple[float, float, float]
    regime: float
    anchor_correction_mm: tuple[float, float, float]

    @property
    def is_stick(self) -> bool:
        return self.regime == REGIME_STICK


def coulomb_return_map(
    *,
    trial_force_n: tuple[float, float, float],
    normal_force_n: float,
    friction_coefficient: float,
    tangential_stiffness_n_per_mm: float,
) -> FrictionOutcome:
    """库仑摩擦的return-map：试探力落在摩擦锥内就粘，超出就投影回锥面并挪锚点。

    ## 这一段为什么不是能量项（0050第二节）

    库仑摩擦**耗散且非associative**——它做的功依赖路径，写不成任何位置的势函数。
    所以接触在本仓是"**半个能量项**"：法向在`PenaltyNormalContact`里，
    切向在这里，而这里**不满足**`EnergyTerm`四方法协议，也不该假装满足。

    ## 判据

    ``|T_trial| ≤ μN`` → **粘**：实际力就是试探力，锚点不动；
    否则 → **滑**：``T = μN · T_trial/|T_trial|``，并把锚点沿滑移方向挪
    ``(|T_trial| − μN)/k_t``——挪完之后，用新锚点重算的试探力恰好落在锥面上。
    **这一条是return-map的定义，也是它可被验证的地方**（见`tests/`那条自洽门）。

    ## 边界

    ``N = 0``（分离）时摩擦锥退化成一个点：**没有法向力就没有摩擦**，
    一切试探力都被投影成零，判别是`REGIME_SEPARATED`而不是`REGIME_SLIP`——
    "分离"与"在滑"是两件事，混起来会让案例分不清"飞出去了"和"在蹭着走"。
    """

    #: **试探力此前完全不校验**，而同一函数对``N``/``μ``/``k_t``都很严
    #: （2026-08-06对抗审核）：nan进来会原样变成锚点修正**写进状态向量**，
    #: 而状态是复现契约；长度2或4的元组会被原样返回。
    if len(trial_force_n) != 3 or not all(math.isfinite(v) for v in trial_force_n):
        raise ContactError(f"trial force must be a finite 3-vector: {trial_force_n!r}")
    if normal_force_n < 0.0 or not math.isfinite(normal_force_n):
        raise ContactError(f"normal force must be finite and nonnegative: {normal_force_n!r}")
    if friction_coefficient < 0.0 or not math.isfinite(friction_coefficient):
        raise ContactError(
            f"friction coefficient must be finite and nonnegative: {friction_coefficient!r}"
        )
    if not (tangential_stiffness_n_per_mm > 0.0 and math.isfinite(tangential_stiffness_n_per_mm)):
        raise ContactError(
            f"tangential stiffness must be positive: {tangential_stiffness_n_per_mm!r}"
        )

    zero = (0.0, 0.0, 0.0)
    if normal_force_n == 0.0:
        return FrictionOutcome(zero, REGIME_SEPARATED, zero)

    limit = friction_coefficient * normal_force_n
    magnitude = math.sqrt(sum(component * component for component in trial_force_n))
    if magnitude <= limit:
        return FrictionOutcome(trial_force_n, REGIME_STICK, zero)

    scale = limit / magnitude
    force = tuple(component * scale for component in trial_force_n)
    #: 超出锥面的那一截除以切向刚度，就是这一步滑掉的距离。
    slip_mm = (magnitude - limit) / tangential_stiffness_n_per_mm
    correction = tuple(component / magnitude * slip_mm for component in trial_force_n)
    return FrictionOutcome(force, REGIME_SLIP, correction)


@dataclass(frozen=True)
class TangentialStickSpring:
    """粘着弹簧：``U = Σ ½·k_t·|P(x − a)|²``，``P = I − n⊗n``是切平面投影。单位N·mm。

    ## 它为什么**是**能量项，而滑移不是

    0033调研的结论：带粘着的库仑摩擦把切向相对位移**分解成可逆的"粘"分量与
    不可逆的"滑"分量**（与塑性力学的形制是同一个）。

    **可逆的那一半是弹性的，因此有势函数**——就是这个类。
    不可逆的那一半（滑移）耗散、非associative，**写不成任何位置的势**——
    那是`coulomb_return_map`。

    这就是0050第二节"接触是**半个能量项**"那句话的具体形状：
    法向 + 粘着在能量里（进`EnergyRegistry`、进牛顿的切线刚度），
    滑移在return-map里（改锚点，即改状态里的历史）。
    **把这条分界写在这里，是因为下一个人最可能犯的错是想把整个摩擦写成势能。**

    ## 它给对了什么

    与法向项同构：平衡时``k_t·|Δ| = T_理论``，于是**切向力精确、切向位移是``O(1/k_t)``**。
    斜面上实测：``T = W·sinθ``与``k_t``无关。

    ## 锚点是**输入**，不是这里算出来的

    本项读锚点，不写锚点。写锚点的是return-map——那一步才是历史发生的地方。
    锚点住在状态向量的槽位里（`ContactLayout`），本项从调用方拿到它的值。
    """

    name: str = "tangential_stick"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (节点索引, 锚点mm, 面法向单位矢量, 切向刚度N/mm)
    springs: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float], float], ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.springs:
            raise ContactError("tangential_stick needs at least one spring")
        for node, anchor, normal, stiffness in self.springs:
            if isinstance(node, bool) or not isinstance(node, int) or node < 0:
                raise ContactError(f"stick node index must be a nonnegative int: {node!r}")
            if len(anchor) != 3 or not all(math.isfinite(value) for value in anchor):
                raise ContactError(f"anchor must be a finite 3-vector: {anchor!r}")
            #: **单位矢量那道门挡不住nan**：``abs(nan − 1.0) > tol``是``False``，
            #: 于是nan法向一路通过、能量与梯度全变nan（2026-08-06对抗审核实测）。
            #: 同门的`PenaltyNormalContact`有这两条检查，这里此前没有。
            if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
                raise ContactError(f"stick normal must be a finite 3-vector: {normal!r}")
            norm = math.sqrt(sum(component * component for component in normal))
            if abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ContactError(f"stick normal must be a unit vector (|n| = {norm!r})")
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise ContactError(f"tangential stiffness must be positive: {stiffness!r}")

    def node_index_bound(self) -> int:
        return max(node for node, _, _, _ in self.springs) + 1

    @staticmethod
    def _tangential_offset_mm(
        vector: tuple[float, ...],
        node: int,
        anchor: tuple[float, float, float],
        normal: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """``P(x − a)``：位移里**扣掉法向分量**的那一部分。

        扣掉法向是这个项的全部要害：不扣，粘着弹簧会连法向也一起拉，
        于是它与法向罚函数**重复计入法向刚度**，而两者的刚度通常差好几个数量级——
        结果是法向力悄悄变成``(k_n + k_t)·δ``，**而`normal_force_n`报的仍是``k_n·δ``**。
        """

        base = 3 * node
        delta = tuple(vector[base + axis] - anchor[axis] for axis in range(3))
        along_normal = sum(delta[axis] * normal[axis] for axis in range(3))
        return tuple(delta[axis] - along_normal * normal[axis] for axis in range(3))

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(state.vector, node, anchor, normal)
            total += 0.5 * stiffness * sum(value * value for value in offset)
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(state.vector, node, anchor, normal)
            base = 3 * node
            for axis in range(3):
                result[base + axis] += stiffness * offset[axis]
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
        """``k_t·(I − n⊗n)``。**常量**——粘着是线性的，这是它比滑移好对付的原因。"""

        entries: list[tuple[int, int, float]] = []
        for node, _, normal, stiffness in self.springs:
            base = 3 * node
            for a in range(3):
                for b in range(3):
                    value = stiffness * ((1.0 if a == b else 0.0) - normal[a] * normal[b])
                    entries.append((base + a, base + b, value))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        vector = state.vector
        total = 0.0
        gradient = [0.0] * len(vector) if need_gradient else None
        for node, anchor, normal, stiffness in self.springs:
            offset = self._tangential_offset_mm(vector, node, anchor, normal)
            total += 0.5 * stiffness * sum(value * value for value in offset)
            if gradient is not None:
                base = 3 * node
                for axis in range(3):
                    gradient[base + axis] += stiffness * offset[axis]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def tangential_force_n(self, state: State) -> tuple[tuple[float, float, float], ...]:
        """每根弹簧的切向力矢量``k_t·P(x − a)``——**摩擦锥要判的就是它的模**。"""

        return tuple(
            tuple(
                stiffness * value
                for value in self._tangential_offset_mm(state.vector, node, anchor, normal)
            )
            for node, anchor, normal, stiffness in self.springs
        )


