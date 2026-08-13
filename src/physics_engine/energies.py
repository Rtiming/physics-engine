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
from typing import ClassVar, Literal, Protocol

from physics_engine.state import State, StateLayout

Vector = tuple[float, ...]
#: 稠密对称矩阵，行优先。本块的规模是"几个节点"，稠密够用且可逐字节对拍；
#: 稀疏表示随真正的杆内核进来（那时才有几百到几千自由度）。
Matrix = tuple[tuple[float, ...], ...]


#: 单位边界的显式常量。梯度是N、质量是kg，而 N/kg = m/s²（**米制**）；
#: 状态是mm制，所以从力算加速度时必须乘它。写成有名字的常量而不是字面量1000，
#: 是因为一个裸的1000在半年后没人认得出它是单位换算还是某个物理系数。
MM_PER_M = 1000.0

POTENTIAL = "potential"
DISSIPATION = "dissipation"
TermKind = Literal["potential", "dissipation"]


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


def resolve_node_count(state: State, context: EnergyContext) -> int:
    """节点数的**唯一口径**：来自上下文的质量表，并与状态向量交叉校验。

    ### 为什么不是``len(state.vector) // 3``

    那个写法假定**整条状态向量都是节点坐标**。它在T4—T5一直成立，
    因为那时状态里除了节点位置什么都没有。**决策0050让它不再成立**：
    接触锚点按声明的接触对分槽，挂在节点块之后。

    `EnergyContext`第52行的注释从一开始就写着"**节点数由布局定，这里只存值**"——
    **契约写对了，实现从来没照做**。这不是新需求，是把既有契约兑现。

    ### 实测的后果（写下来，因为它是静默的）

    在两节点布局后面挂一个锚点槽、而质量表恰好有三项时（上下文按体数或按声明
    填质量表，这很容易发生）：重力能从29.43变成98.1（**3.3倍**），
    **梯度长度仍然等于布局长度所以求解器察觉不到**，
    而且**重力出现在锚点自由度上**——求解器会真的把摩擦锚点往下拉。
    质量表不够长时报的是裸``IndexError``，也不是能读的错。

    ### 口径

    节点块是状态向量的**前缀**：``[0, 3n)``是节点位置，其后是辅助自由度
    （锚点等），**能量项一律不许碰后面那段**。

    ### 布局是权威，上下文只提供值（2026-08-06补齐）

    此前这里只按"上下文的质量表说了算"，于是**一份声明3个质量而布局只有2个节点的
    上下文判不出来**——实测重力会落到接触锚点槽上。那条洞当时如实登记了，
    触发条件写的是"锚点布局构造器进仓时把边界带上来"。

    **构造器进仓后第一次修只修了一半**：能量项这一侧改了，
    而`EnergyRegistry.acceleration`那条桥是另一处实现，它照样信上下文。
    实测两处都要修，而**真正的病根是节点数有两个来源而没有一个是权威的**。

    现在`StateLayout.node_dof_count`让布局成为权威：声明了就必须与质量表一致，
    不一致即失败关闭。**老形制（不声明）保持"上下文说了算"**——
    它对纯节点布局永远正确，而带辅助段的布局一律由`build_contact_layout`产出、
    一律声明边界。
    """

    count = len(context.node_masses_kg)
    declared = state.layout.node_dof_count
    if declared is not None and declared != 3 * count:
        raise EnergyError(
            f"layout declares a node block of {declared} scalars "
            f"({declared // 3} nodes) but the context carries {count} node masses — "
            "**布局是权威**：对不上时重力会落到节点块之后的辅助自由度（接触锚点）上"
        )
    if 3 * count > len(state.vector):
        raise EnergyError(
            f"context declares {count} node masses ({3 * count} position dof) "
            f"but the state vector is only {len(state.vector)} long — "
            "节点块必须是状态向量的前缀"
        )
    return count


class EnergyTerm(Protocol):
    """四方法协议（形制采WDS `model/energies.py:43`的`EnergyTerm`）。"""

    name: str
    kind: Literal["potential"]

    def node_index_bound(self) -> int:
        """本项索引到的**最大节点号+1**；不按索引取节点的项返回0。

        由`EnergyRegistry`在装配时与`resolve_node_count`比对**一次**，
        越界即失败关闭。**不做逐次索引的边界检查**：0026的画像里装配是唯一的
        复杂度问题，往每个``3*i``上加一次比较是在热点上收税；
        而索引清单在项构造时就冻结了，**构造期算一次、装配期比一次**足够。

        这条门挡的是：一条边/一个载荷/一个模板点名了**超出节点数**的索引，
        于是它读写的是锚点槽——**长度对得上、求解器不报错、结果是错的**。
        """
        ...

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


class DissipationTerm(Protocol):
    """速度相关、没有势函数的项；力与耗散率必须由同一次求值给出。"""

    name: str
    kind: Literal["dissipation"]

    def node_index_bound(self) -> int: ...

    def force_and_power(
        self, state: State, velocity: Sequence[float], context: EnergyContext
    ) -> tuple[Vector, float]:
        """返回``(广义力N, 非负耗散率N·mm/s)``。"""
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
    kind: ClassVar[Literal["potential"]] = POTENTIAL

    def node_index_bound(self) -> int:
        """0——本项不按索引点名节点，它作用在**上下文说有多少就是多少**个节点上。"""

        return 0

    def energy(self, state: State, context: EnergyContext) -> float:
        gx, gy, gz = context.gravity_mm_s2
        total = 0.0
        for index in range(resolve_node_count(state, context)):
            mass = context.node_masses_kg[index]
            x, y, z = state.vector[3 * index : 3 * index + 3]
            total -= mass * (gx * x + gy * y + gz * z) / MM_PER_M
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        gx, gy, gz = context.gravity_mm_s2
        result = _zeros(len(state.vector))
        for index in range(resolve_node_count(state, context)):
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
        for index in range(resolve_node_count(state, context)):
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
class PointLoad:
    """指定节点上的固定外力势能：``U = −Σ F_i · x_i``（单位N·mm）。

    **符号是这个项的全部内容**：外力做的功是**负**势能。载荷把节点往``+F``方向
    推，节点顺着走时系统势能下降——写成``+F·x``就得到一个把节点往力的反方向推的
    "外力"，而且**求解器照样会收敛**，只是收敛到一个物理上相反的解。

    ``gradient = −F``（常量）、``hessian`` **恒为零**（能量对位置线性）。

    **零Hessian的陷阱**：本项对切线刚度没有任何贡献。一个只含``PointLoad``的
    注册表给出全零Hessian，`solve_equilibrium`在那里**必须失败关闭**而不是返回
    垃圾解——`_solve_dense`/`_solve_banded`的奇异判据守着这一条，
    `tests/test_energies.py`有正向断言。这不是理论顾虑：屈曲案例里恰恰要
    ``PointLoad``与几何刚度配合，而几何刚度全部来自`AxialStretch`那个
    ``(k·ε/L)·(I − d⊗d)``项——**压缩时它是负的**，那才是屈曲的来源。

    ### 单位：这里**没有**``MM_PER_M``，而`UniformGravity`那里有

    力是N、位置是mm，``N × mm = N·mm``**直接就是本仓的能量单位**，不需要换算。
    重力项要除以1000是因为它拿的是kg与mm/s²：``kg·mm²/s²``不是N·mm。
    两个项一个除一个不除，看起来不对称，实际上是同一条口径的两种情形——
    **量纲是算出来的，不是照抄相邻代码抄出来的**。本仓已经因单位吃过两次亏
    （0024的静默1000倍、`two_body_spring`那次两个错误互相抵消），
    所以这里配了两道门：量纲门（1 N力移动1 mm恰好给1 N·mm）与
    自由体门（单质点在``F``下的加速度恰为``F/m``，且``a·m``与质量无关）。
    """

    name: str = "point_load"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: 载荷：(节点索引, (Fx, Fy, Fz) 单位N)。同一节点**不许出现两次**——
    #: 两条同节点载荷该由调用方自己合并，库不替它猜求和次序（浮点加法不结合）。
    loads: tuple[tuple[int, tuple[float, float, float]], ...] = ()

    def __post_init__(self) -> None:
        if not self.loads:
            raise EnergyError("point_load needs at least one load")
        seen: set[int] = set()
        for node, force in self.loads:
            if not isinstance(node, int) or node < 0:
                raise EnergyError(f"point load node index must be a nonnegative int: {node!r}")
            if node in seen:
                raise EnergyError(
                    f"node {node} carries two point loads — 请调用方自己合并，"
                    "库不替它猜求和次序"
                )
            seen.add(node)
            if len(force) != 3 or not all(math.isfinite(value) for value in force):
                raise EnergyError(f"point load force must be a finite 3-vector: {force!r}")

    def node_index_bound(self) -> int:
        return max(node for node, _ in self.loads) + 1

    def _base(self, state: State, context: EnergyContext, node: int) -> int:
        """节点号→向量偏移。

        **界限取节点块而不是向量长度**（决策0050）：向量末尾挂着接触锚点槽时，
        按向量长度判界会把越界的节点号放进锚点里——**长度检查照样通过，
        力加到了锚点上**。所以这里问的是`resolve_node_count`，不是``len(vector)//3``。

        `EnergyRegistry`在装配时已按`node_index_bound`比过一次；
        这里兜住的是**直接调用本项**的路径（案例与测试都这么调）。
        """

        if node >= resolve_node_count(state, context):
            raise EnergyError(
                f"point load names node {node} but the context only has "
                f"{resolve_node_count(state, context)} nodes"
            )
        return 3 * node

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for node, force in self.loads:
            base = self._base(state, context, node)
            total -= (
                force[0] * state.vector[base]
                + force[1] * state.vector[base + 1]
                + force[2] * state.vector[base + 2]
            )
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = _zeros(len(state.vector))
        for node, force in self.loads:
            base = self._base(state, context, node)
            for axis in range(3):
                result[base + axis] -= force[axis]
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        n = len(state.vector)
        return tuple(tuple(row) for row in _zero_matrix(n))

    def hessian_entries(self, state, context):
        """恒为零——**一个非零项都没有**，与`UniformGravity`同理。"""

        return ()

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合：一次遍历同时出能量与梯度。

        **能量的求值表达式与`energy()`逐字一致**（同样的括号、同样的次序），
        所以逐字节门不靠巧合——spec/12第3.1节的承重条款。
        """

        vector = state.vector
        total = 0.0
        gradient = _zeros(len(vector)) if need_gradient else None
        for node, force in self.loads:
            base = self._base(state, context, node)
            total -= (
                force[0] * vector[base]
                + force[1] * vector[base + 1]
                + force[2] * vector[base + 2]
            )
            if gradient is not None:
                for axis in range(3):
                    gradient[base + axis] -= force[axis]
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
    kind: ClassVar[Literal["potential"]] = POTENTIAL
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

    def node_index_bound(self) -> int:
        return max(max(i, j) for i, j, _, _ in self.edges) + 1

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
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: 模板：((节点索引, 系数), ...)、标度（含EI、ℓ³与求积权）、仿射偏置（3矢量）
    stencils: tuple[
        tuple[tuple[tuple[int, float], ...], float, tuple[float, float, float]], ...
    ] = ()

    def node_index_bound(self) -> int:
        return max(node for coefficients, _, _ in self.stencils for node, _ in coefficients) + 1

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
class DiscreteElasticBending:
    """几何精确弯曲（DER形制）：``U = Σ_i (EI/(2·ℓ_i))·κ_i²``，``κ_i = 2·tan(θ_i/2)``。

    ``θ_i``是顶点``i``处两条相邻边的转角，``ℓ_i``是该顶点的**参考构型**Voronoi长度
    （形制采Bergou等2008《Discrete Elastic Rods》的曲率二法矢弯曲能；
    WDS `model/energies.py`的`BendingEnergy`是它的各向异性版本，用同一个``κb``）。

    **与`LinearBending`的分界**：那一个是位置的二次型，只在小挠度成立；
    本项对**刚体转动严格不变**（实测偏差9.0e-16），大挠度才有意义。
    两者都在仓里、都不改对方——`LinearBending`不是它的退化版本，是另一套适用域。

    ### 求值形制：走DER的分母，不走``(1−c)/(1+c)``

    ``κ² = 4·|e_a × e_b|² / (|e_a||e_b| + e_a·e_b)²``。这与``4(1−cosθ)/(1+cosθ)``
    数学相等，但**近直链时前者无相消**：``1 − cosθ``在``θ→0``时是两个接近1的数相减，
    相对误差被放大``2/θ²``倍；叉积平方没有这个问题。分母``D = |e_a||e_b| + e_a·e_b``
    在``θ→π``（对折）时趋零——那是``2·tan(θ/2)``自身的奇点，**失败关闭**，不返回大数。

    ### 梯度与Hessian是解析的（不是有限差分）

    ``ℓ_i``取参考构型，所以``U``只经``c = cosθ``依赖位置：``U = (EI/(2ℓ))·g(c)``、
    ``g(c) = 4(1−c)/(1+c)``、``g'(c) = −8/(1+c)²``、``g''(c) = 16/(1+c)³``
    （代码里用``1+c = D/(|e_a||e_b|)``改写，同样避开相消）。于是

        ∇U = (EI/(2ℓ))·g'(c)·∇c
        ∇²U = (EI/(2ℓ))·[g''(c)·∇c⊗∇c + g'(c)·∇²c]

    以边矢量``u = x_i − x_{i−1}``、``v = x_{i+1} − x_i``为中间变量（``â``表单位化）：

        ∂c/∂u = (v̂ − c·û)/|u|,   ∂c/∂v = (û − c·v̂)/|v|
        ∂²c/∂u∂u = [3c·û⊗û − û⊗v̂ − v̂⊗û − c·I] / |u|²
        ∂²c/∂v∂v = [3c·v̂⊗v̂ − û⊗v̂ − v̂⊗û − c·I] / |v|²
        ∂²c/∂u∂v = [I − û⊗û − v̂⊗v̂ + c·û⊗v̂] / (|u||v|)

    再经常量线性映射``(u,v) ← (x_{i−1}, x_i, x_{i+1})``（系数``u:(−1,1,0)``、
    ``v:(0,−1,1)``）落到九个自由度上——链式法则一次，无近似。
    有限差分门实测：梯度相对偏差6.8e-9、Hessian偏差/量级3.3e-10。
    **但那道门验不了物理**（spec/12第6.1节）；验物理的是
    `cases/large_deflection_cantilever`的椭圆积分闭式。
    """

    name: str = "discrete_elastic_bending"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: 顶点：(左节点, 中节点, 右节点, 弯曲刚度EI_nmm2, 参考Voronoi长度ℓ_mm)
    vertices: tuple[tuple[int, int, int, float, float], ...] = ()

    def node_index_bound(self) -> int:
        return max(max(left, middle, right) for left, middle, right, _, _ in self.vertices) + 1

    def __post_init__(self) -> None:
        if not self.vertices:
            raise EnergyError("discrete_elastic_bending needs at least one vertex")
        for left, middle, right, stiffness, voronoi in self.vertices:
            if len({left, middle, right}) != 3:
                raise EnergyError(
                    f"a bending vertex needs three distinct nodes: {(left, middle, right)}"
                )
            if not (stiffness > 0.0 and math.isfinite(stiffness)):
                raise EnergyError(f"bending stiffness must be positive: {stiffness!r}")
            if not (voronoi > 0.0 and math.isfinite(voronoi)):
                raise EnergyError(f"voronoi length must be positive: {voronoi!r}")

    def _vertex_terms(self, state: State, vertex):
        """一个顶点的标量闭包：曲率核**只在这里求值**。

        返回``(能量, û, v̂, |u|, |v|, c, EI/(2ℓ)·g', EI/(2ℓ)·g'')``。
        """

        left, middle, right, stiffness, voronoi = vertex
        x = state.vector
        u = tuple(x[3 * middle + axis] - x[3 * left + axis] for axis in range(3))
        v = tuple(x[3 * right + axis] - x[3 * middle + axis] for axis in range(3))
        length_a = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        length_b = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if length_a == 0.0 or length_b == 0.0:
            raise EnergyError(
                f"bending vertex {middle} has a zero-length edge — 转角未定义，能量在此不可微"
            )
        product = length_a * length_b
        dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        denominator = product + dot
        if denominator <= 0.0:
            raise EnergyError(
                f"bending vertex {middle} is folded back (θ→π) — "
                "κ = 2·tan(θ/2)在此发散，这是模型自身的奇点，不是可以返回大数的地方"
            )
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        curvature_sq = 4.0 * (cross[0] * cross[0] + cross[1] * cross[1]
                              + cross[2] * cross[2]) / (denominator * denominator)
        scale = 0.5 * stiffness / voronoi
        energy = scale * curvature_sq
        cosine = dot / product
        unit_a = tuple(component / length_a for component in u)
        unit_b = tuple(component / length_b for component in v)
        # g'(c) = −8/(1+c)² 与 g''(c) = 16/(1+c)³，用 1+c = D/(|u||v|) 改写。
        ratio = product / denominator
        first = scale * (-8.0) * ratio * ratio
        second = scale * 16.0 * ratio * ratio * ratio
        return energy, unit_a, unit_b, length_a, length_b, cosine, first, second

    def _vertex_blocks(self, state: State, vertex):
        """一个顶点的完整贡献：(三个节点, 能量, 三个梯度块, 3×3个Hessian块)。

        **曲率核只求值一次**——`quantities`存在的理由就是这个（spec/12第3.1节）。
        能量值由`_vertex_terms`原样带出，**没有第二个求值表达式**，
        所以融合路径与单独调`energy`逐字节相同不靠巧合。
        """

        left, middle, right, _, _ = vertex
        energy, unit_a, unit_b, length_a, length_b, cosine, first, second = (
            self._vertex_terms(state, vertex)
        )
        grad_u = tuple(
            (unit_b[axis] - cosine * unit_a[axis]) / length_a for axis in range(3)
        )
        grad_v = tuple(
            (unit_a[axis] - cosine * unit_b[axis]) / length_b for axis in range(3)
        )
        square_a = length_a * length_a
        square_b = length_b * length_b
        product = length_a * length_b
        block_uu = []
        block_vv = []
        block_uv = []
        for a in range(3):
            row_uu = []
            row_vv = []
            row_uv = []
            for b in range(3):
                identity = 1.0 if a == b else 0.0
                mixed = unit_a[a] * unit_b[b] + unit_b[a] * unit_a[b]
                curvature_uu = (
                    3.0 * cosine * unit_a[a] * unit_a[b] - mixed - cosine * identity
                ) / square_a
                curvature_vv = (
                    3.0 * cosine * unit_b[a] * unit_b[b] - mixed - cosine * identity
                ) / square_b
                curvature_uv = (
                    identity - unit_a[a] * unit_a[b] - unit_b[a] * unit_b[b]
                    + cosine * unit_a[a] * unit_b[b]
                ) / product
                row_uu.append(second * grad_u[a] * grad_u[b] + first * curvature_uu)
                row_vv.append(second * grad_v[a] * grad_v[b] + first * curvature_vv)
                row_uv.append(second * grad_u[a] * grad_v[b] + first * curvature_uv)
            block_uu.append(row_uu)
            block_vv.append(row_vv)
            block_uv.append(row_uv)

        #: 常量线性映射的系数：u = x_i − x_{i−1}、v = x_{i+1} − x_i。
        weights_u = (-1.0, 1.0, 0.0)
        weights_v = (0.0, -1.0, 1.0)
        nodes = (left, middle, right)
        gradients = tuple(
            tuple(
                first * (weights_u[m] * grad_u[axis] + weights_v[m] * grad_v[axis])
                for axis in range(3)
            )
            for m in range(3)
        )
        blocks = tuple(
            tuple(
                tuple(
                    tuple(
                        weights_u[m] * weights_u[n] * block_uu[a][b]
                        + weights_u[m] * weights_v[n] * block_uv[a][b]
                        + weights_v[m] * weights_u[n] * block_uv[b][a]
                        + weights_v[m] * weights_v[n] * block_vv[a][b]
                        for b in range(3)
                    )
                    for a in range(3)
                )
                for n in range(3)
            )
            for m in range(3)
        )
        return nodes, energy, gradients, blocks

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for vertex in self.vertices:
            total += self._vertex_terms(state, vertex)[0]
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = _zeros(len(state.vector))
        for vertex in self.vertices:
            nodes, _, gradients, _ = self._vertex_blocks(state, vertex)
            for m in range(3):
                for axis in range(3):
                    result[3 * nodes[m] + axis] += gradients[m][axis]
        return tuple(result)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        n = len(state.vector)
        result = _zero_matrix(n)
        for vertex in self.vertices:
            nodes, _, _, blocks = self._vertex_blocks(state, vertex)
            for m in range(3):
                for n_index in range(3):
                    for a in range(3):
                        for b in range(3):
                            result[3 * nodes[m] + a][3 * nodes[n_index] + b] += (
                                blocks[m][n_index][a][b]
                            )
        return tuple(tuple(row) for row in result)

    def hessian_entries(self, state, context):
        entries = []
        for vertex in self.vertices:
            nodes, _, _, blocks = self._vertex_blocks(state, vertex)
            for m in range(3):
                for n_index in range(3):
                    for a in range(3):
                        for b in range(3):
                            entries.append((
                                3 * nodes[m] + a, 3 * nodes[n_index] + b,
                                blocks[m][n_index][a][b],
                            ))
        return tuple(entries)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        """融合：**曲率核只求值一次**，能量/梯度/Hessian一次遍历同时出。"""

        n = len(state.vector)
        total = 0.0
        gradient = _zeros(n) if need_gradient else None
        hessian = _zero_matrix(n) if need_hessian else None
        for vertex in self.vertices:
            nodes, energy, gradients, blocks = self._vertex_blocks(state, vertex)
            total += energy
            if gradient is not None:
                for m in range(3):
                    for axis in range(3):
                        gradient[3 * nodes[m] + axis] += gradients[m][axis]
            if hessian is not None:
                for m in range(3):
                    for n_index in range(3):
                        for a in range(3):
                            for b in range(3):
                                hessian[3 * nodes[m] + a][3 * nodes[n_index] + b] += (
                                    blocks[m][n_index][a][b]
                                )
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            tuple(tuple(row) for row in hessian) if hessian is not None else None,
        )


def clamped_chain_bending_vertices(
    node_count: int,
    segment_length_mm: float,
    bending_stiffness_nmm2: float,
) -> tuple[tuple[int, int, int, float, float], ...]:
    """等分链的几何精确弯曲顶点；节点0与节点1被钉住即为固支。

    **固支顶点的Voronoi长度是``3h/2``，不是``h``——这不是配出来的数，是推出来的。**

    以边角``φ_j``（第``j``条边与固支切向的夹角）写离散弯曲能：正确的固支处理是
    在钳口另加一个"半格顶点"``(EI/(2·(h/2)))·φ_0²``，因为夹持点到第一条边中点
    只有``h/2``的弧长。把第一条边整根钉死（本案例的做法，也是DER的常规做法）
    等于强行令``φ_0 = 0``，**把那半格的柔度整个删掉**，离散梁在钳口处偏刚。
    把``φ_0``做静态凝聚消掉即可看出该还回多少：

        min_{φ_0} (EI/h)·φ_0² + (EI/(2h))·(φ_1 − φ_0)²  →  φ_0 = φ_1/3
        代回得 (EI/(3h))·φ_1² = (EI/(2·(3h/2)))·φ_1²

    即固支顶点的等效Voronoi长度**恰为``3h/2``**。实测：取``h``时收敛比
    2.058/2.032/2.017（一阶）、取``3h/2``时3.961/4.051/4.060（二阶），
    且该系数与载荷无关（β=0.8/3.0/5.0三档都成立），邻近取值1.45与1.55都退化。

    这是决策0027那条教训的第二例，**症状相同、病根同类而不同处**：
    那次是梯形求积的端点半权，这次是被钉死的第一条边吞掉的半格柔度。
    两次都不是"边界条件的阶次"——0027警告过，归因错就会去改边界条件而那改不动它。
    自由端不需要对应修正（那里``κ(L)=0``，漏掉的半格是O(h³)）：
    实测把自由端权乘0.5或1.5，收敛比4.063/4.066与3.935/4.040均不变。
    """

    if node_count < 3:
        raise EnergyError("a bending chain needs at least three nodes")
    if not (segment_length_mm > 0.0 and math.isfinite(segment_length_mm)):
        raise EnergyError("segment length must be positive and finite")
    if not (bending_stiffness_nmm2 > 0.0 and math.isfinite(bending_stiffness_nmm2)):
        raise EnergyError("bending stiffness must be positive and finite")
    return tuple(
        (
            index - 1, index, index + 1, bending_stiffness_nmm2,
            segment_length_mm * (1.5 if index == 1 else 1.0),
        )
        for index in range(1, node_count - 1)
    )


@dataclass(frozen=True)
class DissipationQuantities:
    """同一注册表一次耗散求值的力与账，次序与注册次序一致。"""

    force_n: Vector
    rate_nmm_per_s: float
    by_term: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class EnergyRegistry:
    """注册表：``enabled``显式、**求和次序固定**（spec/12第3.3节）。

    浮点加法不结合——次序变了总能量的末位就变。所以次序是声明的一部分，
    不是"字典恰好这么排"。
    """

    terms: tuple[EnergyTerm | DissipationTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise EnergyError("an energy registry needs at least one enabled term")
        for term in self.terms:
            if getattr(term, "kind", None) not in (POTENTIAL, DISSIPATION):
                raise EnergyError(
                    f"term {getattr(term, 'name', '<unnamed>')!r} must declare kind as "
                    f"{POTENTIAL!r} or {DISSIPATION!r}"
                )
        names = [term.name for term in self.terms]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise EnergyError(f"duplicate energy term names: {duplicates}")

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.terms)

    def assert_within_nodes(self, state: State, context: EnergyContext) -> int:
        """装配前比一次：**没有一个项索引到节点块之外**。返回节点数。

        决策0050让状态向量不再"整条都是节点"——接触锚点按声明的接触对分槽挂在
        节点块之后。于是一个越界的节点索引不再撞上``IndexError``，
        **它会静默地读写锚点槽**：长度对得上、求解器不报错、结果是错的。

        比较放在这里而不是每次索引处，理由见`EnergyTerm.node_index_bound`——
        0026的画像里装配是唯一的复杂度问题，**热点上不加逐元素比较**。
        这里是O(项数)，与一次装配的O(自由度²)相比可忽略。
        """

        count = resolve_node_count(state, context)
        for term in self.terms:
            bound = term.node_index_bound()
            if bound > count:
                raise EnergyError(
                    f"energy term {term.name!r} indexes node {bound - 1} but the "
                    f"context declares only {count} nodes — "
                    "越界的节点号会落进节点块之后的辅助自由度（接触锚点）里，"
                    "那里长度对得上但物理是错的"
                )
        return count

    def total(
        self, state: State, context: EnergyContext, *,
        need_gradient: bool = False, need_hessian: bool = False,
    ) -> tuple[float, Vector | None, Matrix | None]:
        self.assert_within_nodes(state, context)
        n = len(state.vector)
        energy = 0.0
        gradient = _zeros(n) if need_gradient else None
        hessian = _zero_matrix(n) if need_hessian else None
        for term in self.terms:  # 次序即声明次序，不排序、不并行
            if term.kind == DISSIPATION:
                continue
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

        self.assert_within_nodes(state, context)
        accumulated: dict[tuple[int, int], float] = {}
        for term in self.terms:
            if term.kind == DISSIPATION:
                continue
            per_term: dict[tuple[int, int], float] = {}
            for row, column, value in term.hessian_entries(state, context):
                key = (row, column)
                per_term[key] = per_term.get(key, 0.0) + value
            for key, value in per_term.items():
                accumulated[key] = accumulated.get(key, 0.0) + value
        return accumulated

    def dissipation_quantities(
        self,
        state: State,
        velocity: Sequence[float],
        context: EnergyContext,
    ) -> DissipationQuantities:
        """从**同一个注册表**按声明次序装配耗散力与非负功率账。"""

        self.assert_within_nodes(state, context)
        if len(velocity) != len(state.vector):
            raise EnergyError("velocity and state vector must have the same length")
        force = _zeros(len(state.vector))
        rate = 0.0
        by_term: list[tuple[str, float]] = []
        for term in self.terms:
            if term.kind == POTENTIAL:
                continue
            term_force, term_rate = term.force_and_power(state, velocity, context)
            if len(term_force) != len(state.vector):
                raise EnergyError(
                    f"dissipation term {term.name!r} returned {len(term_force)} forces "
                    f"for a {len(state.vector)}-dof state"
                )
            if not math.isfinite(term_rate) or term_rate < 0.0:
                raise EnergyError(
                    f"dissipation term {term.name!r} returned a nonphysical rate "
                    f"{term_rate!r} N·mm/s"
                )
            if not all(math.isfinite(value) for value in term_force):
                raise EnergyError(
                    f"dissipation term {term.name!r} returned a non-finite force"
                )
            for index, value in enumerate(term_force):
                force[index] += value
            rate += term_rate
            by_term.append((term.name, term_rate))
        return DissipationQuantities(tuple(force), rate, tuple(by_term))

    def dissipation_rate(self, context: EnergyContext, layout: StateLayout):
        """接``integrate_with_dissipation``的耗散率回调。"""

        def rate_of(x: Sequence[float], v: Sequence[float], t: float) -> float:
            state = State(layout=layout, vector=tuple(x))
            return self.dissipation_quantities(state, v, context).rate_nmm_per_s

        return rate_of

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

        ## 辅助自由度**不是动力学自由度**，它们的加速度恒为零

        决策0050让状态向量在节点块之后挂上接触锚点槽。那些槽位是**历史**，
        由return-map推进（`contact.advance_contact_quasistatic`），
        **不由积分器推进**——把它们当成有质量的点去积分，等于给摩擦锚点编造一个惯性。

        本方法此前按``node_masses_kg[index // 3]``取质量，**那是"整条向量都是节点"
        的假定**，遇到含锚点槽的布局直接抛裸``IndexError``——实测确认过。
        修法不是"给锚点也编个质量"，是**在节点块之外一律返回0.0**并在这里写明理由。

        零加速度加上零初速度意味着积分器不会动它们，但**别把正确性寄托在
        "梯度恰好是零"上**：能量项不许碰节点块之外（`EnergyTerm.node_index_bound`
        那道门守着），而这里是同一条纪律在积分侧的另一半。
        """

        quasistatic_only = tuple(
            term.name for term in self.terms if getattr(term, "supports_dynamics", True) is False
        )
        if quasistatic_only:
            raise EnergyError(
                f"energy terms {quasistatic_only!r} are quasistatic-only — "
                "共享项的动力学质量映射、时间积分与回归尚未通过验收；"
                "若项的energy_interpretation为incremental_potential，"
                "还必须先分离可恢复势能与材料耗散"
            )
        has_dissipation = any(term.kind == DISSIPATION for term in self.terms)

        def acceleration_of(x: Sequence[float], v: Sequence[float], t: float):
            state = State(layout=layout, vector=tuple(x))
            node_count = resolve_node_count(state, context)
            _, gradient, _ = self.total(state, context, need_gradient=True)
            assert gradient is not None
            node_dof = 3 * node_count
            if not has_dissipation:
                #: 原保守路径保持原表达式与求值次序，既有轨迹逐字节不变。
                return tuple(
                    -gradient[index] / context.node_masses_kg[index // 3] * MM_PER_M
                    if index < node_dof
                    else 0.0
                    for index in range(len(gradient))
                )
            damping = self.dissipation_quantities(state, v, context).force_n
            return tuple(
                (-gradient[index] + damping[index])
                / context.node_masses_kg[index // 3]
                * MM_PER_M
                if index < node_dof
                else 0.0
                for index in range(len(gradient))
            )

        return acceleration_of


__all__ = [
    "AxialStretch",
    "DISSIPATION",
    "DiscreteElasticBending",
    "DissipationQuantities",
    "DissipationTerm",
    "LinearBending",
    "EnergyContext",
    "EnergyError",
    "EnergyRegistry",
    "EnergyTerm",
    "Matrix",
    "PointLoad",
    "POTENTIAL",
    "TermKind",
    "UniformGravity",
    "Vector",
    "clamped_chain_bending_stencils",
    "clamped_chain_bending_vertices",
]
