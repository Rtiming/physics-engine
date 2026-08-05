"""能量项协议与第一批能量项——spec/12第三节（T5搬迁轨第一块）。

**这是搬迁的接缝。** 形制直接采WDS `model/energies.py`的`EnergyTerm`协议：
一个能量项要满足四方法（``energy``/``gradient``/``hessian``/``quantities``）
才能进引擎，求解器只认协议、不认具体项。

三条承重条款：

1. **``quantities``不是性能糖，是承重条款**。已测事实：单解75%时间在能量装配
   （spec/12第8.1节）。融合路径存在的理由是"算一次拿三样"，而不是省几个函数调用。
2. **融合路径的能量值必须与单独调``energy``逐字节相同**（spec/12第3.1节）。
   WDS为守这条专门保留了零阶读值通道——不是顺手声明一个容差。本模块有门守着。
3. **注册表``enabled``显式、求和次序固定**（spec/12第3.3节）。浮点加法不结合，
   次序变了总能量的末位就变；次序是形制不是实现细节。

**本块的数值形态**（0016与spec/12第五节）：纯Python实现先行。
批量加速档随真正的DER内核搬迁进来，**那时对拍义务才附着**——
今天没有第二个实现，就没有可对拍的对象，声明这一点比假装有门更诚实。
融合路径与单独调之间的逐字节门今天就在，它守的是另一件事。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from physics_engine.state import State, StateLayout

Vector = tuple[float, ...]
#: 稠密对称矩阵，行优先。本块的规模是"几个节点"，稠密够用且可逐字节对拍；
#: 稀疏表示随真正的杆内核进来（那时才有几百到几千自由度）。
Matrix = tuple[tuple[float, ...], ...]


#: 单位边界的显式常量。梯度是N、质量是kg，而 N/kg = m/s²（**米制**）；
#: 状态是mm制，所以从力算加速度时必须乘它。写成有名字的常量而不是字面量1000，
#: 是因为一个裸的1000在半年后没人认得出它是单位换算还是某个物理系数。
MM_PER_M = 1000.0


class EnergyError(ValueError):
    """能量层的一切失败关闭。"""


@dataclass(frozen=True)
class EnergyContext:
    """案例内冻结的参数与外载声明（spec/12第2.1节的上下文层）。"""

    context_id: str
    #: 每个节点的质量。节点数由布局定，这里只存值。
    node_masses_kg: tuple[float, ...]
    #: 重力加速度矢量（mm/s²）。零矢量表示不加重力。
    gravity_mm_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.context_id.startswith("context/"):
            raise EnergyError("context_id must be namespaced like 'context/...'")
        if not self.node_masses_kg:
            raise EnergyError("a context needs at least one node mass")
        if any(mass <= 0.0 or not math.isfinite(mass) for mass in self.node_masses_kg):
            raise EnergyError("node masses must be positive and finite")
        if len(self.gravity_mm_s2) != 3:
            raise EnergyError("gravity_mm_s2 must be a 3-vector")


class EnergyTerm(Protocol):
    """四方法协议（形制采WDS `model/energies.py:43`的`EnergyTerm`）。"""

    name: str

    def energy(self, state: State, context: EnergyContext) -> float: ...
    def gradient(self, state: State, context: EnergyContext) -> Vector: ...
    def hessian(self, state: State, context: EnergyContext) -> Matrix: ...
    def quantities(
        self, state: State, context: EnergyContext, *,
        need_gradient: bool, need_hessian: bool,
    ) -> tuple[float, Vector | None, Matrix | None]: ...

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        """结构非零项``(行, 列, 值)``。

        稠密``hessian()``是协议面、保持不变；本方法是**同一份数学的稀疏读法**。
        实测（decisions/0026）：结构非零只占稠密的0.5852%，稠密累加做了170.9倍
        多余的工作，且让装配的标度指数从约1升到2.02——那是画像里唯一的复杂度问题。
        """
        ...


def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _zero_matrix(n: int) -> list[list[float]]:
    return [[0.0] * n for _ in range(n)]


class UniformGravity:
    """均匀重力势能：``U = −Σ m_i · g · x_i / MM_PER_M``（单位N·mm）。

    **那个除以1000不是可选的**：``m``是kg、``g``是mm/s²、``x``是mm，
    三者相乘得kg·mm²/s²，而``1 N·mm = 1000 kg·mm²/s²``。不除，重力能就与
    拉伸能不同量纲，两者相加是没有意义的数；且经`acceleration()`那一步
    （除以质量再乘MM_PER_M）会得到1000·g的自由落体加速度。

    这是本块抓到的**第二个**单位bug，而且比第一个更隐蔽：`two_body_spring`的
    重力能判据一度"通过"，是因为测试把g除以1000传进来，**两个错误互相抵消**。
    抵消掉的错误比暴露的错误危险得多——它会一直静默到某个不再抵消的调用点。
    补上的门是自由落体：经`acceleration()`桥出来的加速度必须**恰好等于g**。

    梯度是常量、Hessian恒为零——**正因为它平凡，它是"能量本身写错"的第一道门**。
    WDS把自重项的独立解析基准列为最高优先级，理由写在其源码顶部：
    有限差分门只验"雅可比是不是我写的那个能量的导数"，不验"那个能量对不对"。
    """

    name = "uniform_gravity"

    def _nodes(self, state: State) -> int:
        return len(state.vector) // 3

    def energy(self, state: State, context: EnergyContext) -> float:
        gx, gy, gz = context.gravity_mm_s2
        total = 0.0
        for index in range(self._nodes(state)):
            mass = context.node_masses_kg[index]
            x, y, z = state.vector[3 * index : 3 * index + 3]
            total -= mass * (gx * x + gy * y + gz * z) / MM_PER_M
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        gx, gy, gz = context.gravity_mm_s2
        result = _zeros(len(state.vector))
        for index in range(self._nodes(state)):
            mass = context.node_masses_kg[index]
            result[3 * index] = -mass * gx / MM_PER_M
            result[3 * index + 1] = -mass * gy / MM_PER_M
            result[3 * index + 2] = -mass * gz / MM_PER_M
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        n = len(state.vector)
        return tuple(tuple(row) for row in _zero_matrix(n))

    def hessian_entries(self, state, context):
        """恒为零——**没有一个非零项**。稠密路径为它构造并累加了整个n²零矩阵。"""

        return ()

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合：一次遍历同时出能量与梯度（spec/12第3.1节的承重条款）。"""

        gx, gy, gz = context.gravity_mm_s2
        vector = state.vector
        total = 0.0
        gradient = _zeros(len(vector)) if need_gradient else None
        for index in range(len(vector) // 3):
            mass = context.node_masses_kg[index]
            base = 3 * index
            total -= mass * (
                gx * vector[base] + gy * vector[base + 1] + gz * vector[base + 2]
            ) / MM_PER_M
            if gradient is not None:
                gradient[base] = -mass * gx / MM_PER_M
                gradient[base + 1] = -mass * gy / MM_PER_M
                gradient[base + 2] = -mass * gz / MM_PER_M
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            self.hessian(state, context) if need_hessian else None,
        )


@dataclass(frozen=True)
class AxialStretch:
    """逐单元轴向拉伸能：``U = Σ ½·(EA/l0_i)·(|x_j − x_i| − l0_i)²``。

    这是DER杆五个能量项里最刚的一个（轴向刚度比弯曲刚度高约七个数量级，
    spec/12第8.1节），也是搬迁顺序里第一个真正非平凡的项——
    梯度与Hessian都不是常量，能被有限差分门真正考验。

    **逐单元标量闭包就是独立oracle路径**（spec/12第3.2节，形制采WDS
    `_LocalEnergyBase.local_terms`的"per-element scalar closures"注释）：
    每条边的能量单独可算，装配只是求和，因此"装配对不对"与"单元公式对不对"
    可以分开验。
    """

    name: str = "axial_stretch"
    #: 边：(节点i, 节点j, 静止长度mm, 轴向刚度EA_n)
    edges: tuple[tuple[int, int, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.edges:
            raise EnergyError("axial_stretch needs at least one edge")
        for i, j, rest_mm, stiffness_n in self.edges:
            if i == j:
                raise EnergyError(f"edge connects a node to itself: {i}")
            if not (rest_mm > 0.0 and math.isfinite(rest_mm)):
                raise EnergyError(f"rest length must be positive and finite: {rest_mm!r}")
            if not (stiffness_n > 0.0 and math.isfinite(stiffness_n)):
                raise EnergyError(f"axial stiffness must be positive: {stiffness_n!r}")

    def _edge_energy(self, state: State, edge) -> tuple[float, Vector, float, float]:
        """单条边的标量闭包：返回(能量, 单位方向, 伸长量, 刚度/l0)。"""

        i, j, rest_mm, stiffness_n = edge
        xi = state.vector[3 * i : 3 * i + 3]
        xj = state.vector[3 * j : 3 * j + 3]
        delta = tuple(b - a for a, b in zip(xi, xj, strict=True))
        length = math.sqrt(sum(component * component for component in delta))
        if length == 0.0:
            raise EnergyError(
                f"edge ({i},{j}) has zero length — 方向未定义，能量在此不可微"
            )
        direction = tuple(component / length for component in delta)
        elongation = length - rest_mm
        k = stiffness_n / rest_mm
        return 0.5 * k * elongation * elongation, direction, elongation, k

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for edge in self.edges:
            total += self._edge_energy(state, edge)[0]
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = _zeros(len(state.vector))
        for edge in self.edges:
            i, j = edge[0], edge[1]
            _, direction, elongation, k = self._edge_energy(state, edge)
            force = k * elongation
            for axis in range(3):
                result[3 * i + axis] -= force * direction[axis]
                result[3 * j + axis] += force * direction[axis]
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        n = len(state.vector)
        result = _zero_matrix(n)
        for edge in self.edges:
            i, j, rest_mm, _ = edge
            _, direction, elongation, k = self._edge_energy(state, edge)
            length = rest_mm + elongation
            for a in range(3):
                for b in range(3):
                    # d²U/dx² = k·(d⊗d) + (k·ε/L)·(I − d⊗d)
                    outer = direction[a] * direction[b]
                    identity = 1.0 if a == b else 0.0
                    block = k * outer + (k * elongation / length) * (identity - outer)
                    for si, sj, sign in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
                        result[3 * si + a][3 * sj + b] += sign * block
        return tuple(tuple(row) for row in result)

    def _edge_blocks(self, state: State, edge):
        """一条边的完整贡献：(能量, 力矢, 3×3块)。**边核只求值一次**。"""

        i, j, rest_mm, _ = edge
        energy, direction, elongation, k = self._edge_energy(state, edge)
        length = rest_mm + elongation
        force = k * elongation
        # **求值结构必须与`hessian()`逐字一致**：`k*(d_a·d_b)`不是`(k*d_a)*d_b`，
        # 浮点乘法不结合。第一版融合就是在这里丢了末位，被spec/12第3.1节的
        # 逐字节门当场抓住——那条门不是形式主义。
        transverse = k * elongation / length
        block = []
        for a in range(3):
            row = []
            for b in range(3):
                outer = direction[a] * direction[b]
                identity = 1.0 if a == b else 0.0
                row.append(k * outer + transverse * (identity - outer))
            block.append(tuple(row))
        block = tuple(block)
        return i, j, energy, direction, force, block

    def hessian_entries(self, state, context):
        entries = []
        for edge in self.edges:
            i, j, _, _, _, block = self._edge_blocks(state, edge)
            for a in range(3):
                for b in range(3):
                    value = block[a][b]
                    for si, sj, sign in ((i, i, 1.0), (j, j, 1.0),
                                         (i, j, -1.0), (j, i, -1.0)):
                        entries.append((3 * si + a, 3 * sj + b, sign * value))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合：**边核只求值一次**，能量/梯度/Hessian一次遍历同时出。

        分开调会把边核跑三遍——实测边核求值恰为边数的1.0×/2.0×/3.0×
        （decisions/0026）。spec/12第3.1节把融合写成承重条款，理由正是这个。
        """

        n = len(state.vector)
        total = 0.0
        gradient = _zeros(n) if need_gradient else None
        hessian = _zero_matrix(n) if need_hessian else None
        for edge in self.edges:
            i, j, energy, direction, force, block = self._edge_blocks(state, edge)
            total += energy
            if gradient is not None:
                for axis in range(3):
                    gradient[3 * i + axis] -= force * direction[axis]
                    gradient[3 * j + axis] += force * direction[axis]
            if hessian is not None:
                for a in range(3):
                    for b in range(3):
                        value = block[a][b]
                        hessian[3 * i + a][3 * i + b] += value
                        hessian[3 * j + a][3 * j + b] += value
                        hessian[3 * i + a][3 * j + b] -= value
                        hessian[3 * j + a][3 * i + b] -= value
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            tuple(tuple(row) for row in hessian) if hessian is not None else None,
        )


@dataclass(frozen=True)
class LinearBending:
    """小挠度Euler-Bernoulli弯曲能：``U = Σ_s (scale_s/2)·|Σ_k c_k·x_k|²``。

    每个"模板"是一组(节点, 系数)与一个标度。三点模板`(1,−2,1)`除以`ℓ²`就是
    离散二阶导，于是``|Δ²x|²·EI/ℓ³``是``∫EI·(x'')²``的求和。

    **能量是位置的二次型**——梯度线性、Hessian**常量**。这既让解析式可写，
    也让牛顿法一步收敛（可验证的性质，案例里有门）。

    **它不是DER的几何精确弯曲**（`κ = 2·tan(θ/2)`）。适用域：小挠度、
    无几何非线性。真正的DER弯曲随杆内核搬迁进来。这条分界写进`scope_excludes`
    式的文档而不是留给读者猜——spec/12第4.2节对积分器的要求，对能量项同样成立。

    **模板带仿射偏置**：``U = Σ_s (scale_s/2)·|Σ_k c_k·x_k − offset_s|²``。
    内部模板的偏置是零；**固支端不是**——固支是"切向等于给定方向"这个条件，
    写成纯Laplacian模板会把**轴向**也罚进去（直链上`2(x₁−x₀)`不为零）。
    偏置取``2·h·t̂``后，直链的固支端能量恰为零，而横向分量退化成
    一维原型里的``2·y₁``。偏置只平移梯度，不改变Hessian——能量仍是二次型。

    **端点求积权是承重的**：`∫(x'')²`按梯形规则求和时，固支端那个模板取半权。
    漏掉它会让整条收敛曲线从二阶掉到一阶——本仓实测：全权时误差比1.895/1.949/
    1.975（一阶），半权时**恰好4.000/4.000/4.000**（二阶）。
    症状与WDS登记的"固支只有一阶"一模一样，**但病根不同**：那不是边界条件的阶次，
    是求积权重。归因错一次就会去改边界条件，而那改不动它。
    """

    name: str = "linear_bending"
    #: 模板：((节点索引, 系数), ...)、标度（含EI、ℓ³与求积权）、仿射偏置（3矢量）
    stencils: tuple[
        tuple[tuple[tuple[int, float], ...], float, tuple[float, float, float]], ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.stencils:
            raise EnergyError("linear_bending needs at least one stencil")
        for coefficients, scale, offset in self.stencils:
            if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
                raise EnergyError("bending stencil offset must be a finite 3-vector")
            if len(coefficients) < 2:
                raise EnergyError("a bending stencil needs at least two nodes")
            if not (scale > 0.0 and math.isfinite(scale)):
                raise EnergyError(f"bending stencil scale must be positive: {scale!r}")
            total = sum(coefficient for _, coefficient in coefficients)
            if abs(total) > 1.0e-12:
                raise EnergyError(
                    f"bending stencil coefficients must sum to zero (got {total!r}) — "
                    "否则刚体平移会产生弯曲能，那不是弯曲"
                )

    def _stencil_vector(self, state: State, coefficients, offset) -> Vector:
        return tuple(
            sum(
                coefficient * state.vector[3 * node + axis]
                for node, coefficient in coefficients
            )
            - offset[axis]
            for axis in range(3)
        )

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for coefficients, scale, offset in self.stencils:
            d = self._stencil_vector(state, coefficients, offset)
            total += 0.5 * scale * sum(component * component for component in d)
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = _zeros(len(state.vector))
        for coefficients, scale, offset in self.stencils:
            d = self._stencil_vector(state, coefficients, offset)
            for node, coefficient in coefficients:
                for axis in range(3):
                    result[3 * node + axis] += scale * coefficient * d[axis]
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        n = len(state.vector)
        result = _zero_matrix(n)
        for coefficients, scale, _offset in self.stencils:
            for node_a, coefficient_a in coefficients:
                for node_b, coefficient_b in coefficients:
                    block = scale * coefficient_a * coefficient_b
                    for axis in range(3):
                        result[3 * node_a + axis][3 * node_b + axis] += block
        return tuple(tuple(row) for row in result)

    def hessian_entries(self, state, context):
        entries = []
        for coefficients, scale, _offset in self.stencils:
            for node_a, coefficient_a in coefficients:
                for node_b, coefficient_b in coefficients:
                    value = scale * coefficient_a * coefficient_b
                    for axis in range(3):
                        entries.append((3 * node_a + axis, 3 * node_b + axis, value))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合：模板矢量只求值一次。"""

        n = len(state.vector)
        total = 0.0
        gradient = _zeros(n) if need_gradient else None
        hessian = _zero_matrix(n) if need_hessian else None
        for coefficients, scale, offset in self.stencils:
            d = self._stencil_vector(state, coefficients, offset)
            total += 0.5 * scale * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            if gradient is not None:
                for node, coefficient in coefficients:
                    factor = scale * coefficient
                    for axis in range(3):
                        gradient[3 * node + axis] += factor * d[axis]
            if hessian is not None:
                for node_a, coefficient_a in coefficients:
                    for node_b, coefficient_b in coefficients:
                        value = scale * coefficient_a * coefficient_b
                        for axis in range(3):
                            hessian[3 * node_a + axis][3 * node_b + axis] += value
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            tuple(tuple(row) for row in hessian) if hessian is not None else None,
        )


def clamped_chain_bending_stencils(
    node_count: int,
    segment_length_mm: float,
    bending_stiffness_nmm2: float,
    *,
    tangent: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[tuple[tuple[tuple[int, float], ...], float], ...]:
    """等分链的弯曲模板，节点0固支（位置与斜率）、末端自由。

    * 内部模板 ``i=1..n−1``：``(1, −2, 1)``，权1；
    * **固支端模板**：``(−2, 2)``配偏置``2·h·tangent``，**权1/2**
      （梯形求积的端点权）。偏置让直链的固支端能量恰为零；
      半权是二阶收敛的必要条件之一。两者都见`LinearBending`的docstring。
      ``tangent``是固支处的**单位切向**，缺省沿+x。
    * 自由端不加模板——自然边界条件由能量极小自动给出。
    """

    if node_count < 3:
        raise EnergyError("a bending chain needs at least three nodes")
    if not (segment_length_mm > 0.0 and math.isfinite(segment_length_mm)):
        raise EnergyError("segment length must be positive and finite")
    if not (bending_stiffness_nmm2 > 0.0 and math.isfinite(bending_stiffness_nmm2)):
        raise EnergyError("bending stiffness must be positive and finite")
    scale = bending_stiffness_nmm2 / segment_length_mm**3
    zero = (0.0, 0.0, 0.0)
    stencils = [
        (((index - 1, 1.0), (index, -2.0), (index + 1, 1.0)), scale, zero)
        for index in range(1, node_count - 1)
    ]
    clamped_offset = tuple(2.0 * segment_length_mm * component for component in tangent)
    stencils.append((((0, -2.0), (1, 2.0)), 0.5 * scale, clamped_offset))
    return tuple(stencils)


@dataclass(frozen=True)
class EnergyRegistry:
    """注册表：``enabled``显式、**求和次序固定**（spec/12第3.3节）。

    浮点加法不结合——次序变了总能量的末位就变。所以次序是声明的一部分，
    不是"字典恰好这么排"。
    """

    terms: tuple[EnergyTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise EnergyError("an energy registry needs at least one enabled term")
        names = [term.name for term in self.terms]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise EnergyError(f"duplicate energy term names: {duplicates}")

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.terms)

    def total(
        self, state: State, context: EnergyContext, *,
        need_gradient: bool = False, need_hessian: bool = False,
    ) -> tuple[float, Vector | None, Matrix | None]:
        n = len(state.vector)
        energy = 0.0
        gradient = _zeros(n) if need_gradient else None
        hessian = _zero_matrix(n) if need_hessian else None
        for term in self.terms:  # 次序即声明次序，不排序、不并行
            term_energy, term_gradient, term_hessian = term.quantities(
                state, context, need_gradient=need_gradient, need_hessian=need_hessian
            )
            energy += term_energy
            if gradient is not None and term_gradient is not None:
                for index in range(n):
                    gradient[index] += term_gradient[index]
            if hessian is not None and term_hessian is not None:
                for row in range(n):
                    for column in range(n):
                        hessian[row][column] += term_hessian[row][column]
        return (
            energy,
            tuple(gradient) if gradient is not None else None,
            tuple(tuple(row) for row in hessian) if hessian is not None else None,
        )

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> dict[tuple[int, int], float]:
        """稀疏Hessian：``{(行, 列): 值}``，只含结构非零。

        **累加次序逐字复刻稠密路径**：先把每个能量项自己的项累成该项的稠密值
        （`(a1+a2)`），再按声明次序把各项相加（`(a1+a2)+(b1+b2)`）。
        直接把所有项流式相加会得到`((a1+a2)+b1)+b2`——浮点加法不结合，
        那是另一个数。这一条有门守着（`test_sparse_hessian_matches_the_dense_one`）。

        实测（decisions/0026）：结构非零占稠密的0.5852%，稠密累加多做170.9倍工作，
        并把装配的标度指数从约1推到2.02。
        """

        accumulated: dict[tuple[int, int], float] = {}
        for term in self.terms:
            per_term: dict[tuple[int, int], float] = {}
            for row, column, value in term.hessian_entries(state, context):
                key = (row, column)
                per_term[key] = per_term.get(key, 0.0) + value
            for key, value in per_term.items():
                accumulated[key] = accumulated.get(key, 0.0) + value
        return accumulated

    def acceleration(self, context: EnergyContext, layout: StateLayout):
        """把能量梯度变成加速度回调，接`integrate`——**这是内核与积分器的接缝**。

        ``a_i = −∇U_i / m_i × MM_PER_M``。

        **那个换算因子不是凑出来的，它是本仓单位制的必然结果**：能量是N·mm、
        位置是mm，所以梯度的单位是N；质量是kg；而``N/kg = m/s²``——**米制**。
        状态是mm制，于是必须乘1000。

        这正是spec/14第五节盯着的那类静默1000倍。它在本块的开发中真的发生过一次：
        有限差分门（梯度对能量、Hessian对梯度）**全绿**，因为那道门只验
        "雅可比是不是我写的那个能量的导数"——换算因子不在能量里，FD看不见它。
        抓住它的是`cases/two_body_spring`的解析频率门。这就是spec/12第6.1节
        "有限差分验不了物理"最锋利的那句话的一个活标本。
        """

        def acceleration_of(x: Sequence[float], v: Sequence[float], t: float):
            state = State(layout=layout, vector=tuple(x))
            _, gradient, _ = self.total(state, context, need_gradient=True)
            assert gradient is not None
            return tuple(
                -gradient[index] / context.node_masses_kg[index // 3] * MM_PER_M
                for index in range(len(gradient))
            )

        return acceleration_of


__all__ = [
    "AxialStretch",
    "LinearBending",
    "EnergyContext",
    "EnergyError",
    "EnergyRegistry",
    "EnergyTerm",
    "Matrix",
    "UniformGravity",
    "Vector",
    "clamped_chain_bending_stencils",
]
