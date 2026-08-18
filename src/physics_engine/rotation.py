"""准静态路径上的转动自由度：SO(3)指数映射的局部图（决策0079）。

本模块兑现[0063](../../docs/decisions/0063_转动自由度进接触的两条路与它们各自解的题_20260817.md)
第五节第1步，并按[0074](../../docs/decisions/0074_真实网格进接触_20260818.md)第四节
收紧的那一句落地：

> 转动自由度的权威表示是SO(3)；**准静态用指数映射的局部图、动态用四元数**，
> 两者互为同一状态的两种坐标。

## 为什么不是四元数（0063第三节，本模块的存在理由）

`solve_equilibrium`的`fixed_indices`是**标量下标**，它没有任何"带一个约束的四个数"
的位置。四元数进不来不是求解器的缺陷——**约束该由参数化消掉，而不是由求解器背着**。
`rigidbody`那边每步归一化对时间积分成立（把状态拉回流形），
**对能量极小化不成立**：归一化不是能量的一部分，牛顿步会径直穿出单位球。

指数映射给的正是**三个无约束参数**``θ ∈ R³``，姿态``R = exp([θ]×)·R_ref``。
仓内先例是`section_beam`的**边扭角**——准静态路径上一个无约束的转动自由度，
本模块是它的三维推广。

## 局部图不是全局参数化：`retransport`

``θ``只在``|θ| < π``上是双射，而且离``π``越近雅可比越病态。因此本模块的口径是：
**``θ``是相对`R_ref`的增量**，大转角时把``θ``折进`R_ref`并把``θ``归零
（`retransport_levers`）。仓内先例是`cases/anisotropic_rod_twist`的外层重输运循环。
本模块**不替调用方决定什么时候重输运**——那是案例的载荷步策略，不是库的默认。

## 布局：转动块挂在节点块之后

`StateLayout.node_dof_count`强制``% 3 == 0``且节点块必须是向量前缀
（`energies.resolve_node_count`）。转动块**每体三个标量**，挂在节点块**之后**，
所以``node_dof_count``仍是``3·节点数``，重力、罚接触这些只认前缀的项一个字都不用改。

`resolve_node_count`的docstring写着"能量项一律不许碰后面那段"——那句话写于
节点块之后只有**接触锚点（真历史）**的时候。转动块是**真自由度**不是历史，
本模块的两个能量项索引它是显式的、由布局给出的，不是靠约定算偏移量。
这条放宽写在决策0079第三节，**不许由别的模块外推**：
锚点槽仍然一个字都不许碰。

## 接触力对质心取矩：两个项，一个是另一个的增量

本模块把带转动的粘着弹簧**拆成两项**，这不是实现口味，是一条硬性质：

* `MaterialPointStickSpring`——**不含转动**的物质点粘着弹簧
  （两端都可以是自由体，这是它与`contact.friction.TangentialStickSpring`的差别，
  后者只做"节点对固定锚点"）；
* `RotationStickCoupling`——**增量项**
  ``ΔU = ½k(|P(d₀+u)|² − |P(d₀)|²)``，其中``u = Σ ±(R(θ)ℓ − ℓ)``。

拆开的理由是**退化必须逐位**（0001三前提第三条）：``θ = 0``时``u``恒等于零矢量，
于是增量项的能量、节点块梯度、节点块Hessian**结构性地恰为0.0**——
不是"很接近0"，是那几个数根本不进求和。把转动自由度全部钉住时，
牛顿在自由自由度上解的那个约化系统与不带转动块的系统**逐位相同**。

而``∂ΔU/∂θ``恰好是``−M``（接触力对质心的力矩），``∂²ΔU/∂θ²``里带
``offset·P(∂²(Rℓ)/∂θ²)``那一块——**几何刚度**，即力矩随姿态的变化。
两者都必须被有限差分单独验一次（`tests/test_rotation.py`）。

零运行时依赖，纯标准库。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.energies import POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.state import State, StateField, StateLayout

Vec3 = tuple[float, float, float]


class RotationError(ValueError):
    """转动层的一切失败关闭。"""


#: 级数与闭式的切换阈值。**两侧都会坏，所以阈值不是随手取的**：
#:
#: * 闭式在小``φ``上是重相消。最凶的是`_versine_curvature`
#:   ``B2 = (φ²cosφ − 5φsinφ + 8 − 8cosφ)/φ⁶``：分子约``φ⁶/90``而各项约8，
#:   ``φ = 0.1``时抵消掉约9位有效数字；
#: * 级数在大``φ``上截断误差起来。
#:
#: **取0.2是实测出来的交叉点**，不是猜的。两条分支在同一个``φ``上互比（相对偏差，
#: 六个系数取最坏的那个`B2`）：
#:
#: | ``φ`` | 0.05 | 0.1 | **0.2** | 0.3 | 0.5 | 0.7 |
#: |---|---|---|---|---|---|---|
#: | 最坏相对偏差 | 2.9e-06 | 5.4e-08 | **1.5e-11** | 1.3e-11 | 4.9e-10 | 7.3e-09 |
#:
#: 左半边是闭式在烂（相消），右半边是级数在烂（截断），谷底在0.2附近。
#: **起草时取的是0.1，注错验证补的那条跨阈值门把它抓了出来**：
#: 0.1处最坏偏差5.4e-08，比0.2处差3500倍，而改这个数只要一行。
#: 门在`tests/test_rotation.py::test_the_series_and_the_closed_form_agree_across_the_threshold`。
_SERIES_THRESHOLD = 0.2


def _sinc(phi: float) -> float:
    """``sin φ / φ``。小角走级数（无相消，但保持与其余四个同一个分支口径）。"""

    if phi < _SERIES_THRESHOLD:
        s = phi * phi
        return 1.0 - s / 6.0 + s * s / 120.0 - s * s * s / 5040.0 + s * s * s * s / 362880.0
    return math.sin(phi) / phi


def _versine(phi: float) -> float:
    """``(1 − cos φ) / φ²``。"""

    if phi < _SERIES_THRESHOLD:
        s = phi * phi
        return (
            0.5 - s / 24.0 + s * s / 720.0 - s * s * s / 40320.0 + s * s * s * s / 3628800.0
        )
    return (1.0 - math.cos(phi)) / (phi * phi)


def _sinc_slope(phi: float) -> float:
    """``A = (1/φ)·d(sinc)/dφ = (φcosφ − sinφ)/φ³``。``∂a/∂θ_k = A·θ_k``。"""

    if phi < _SERIES_THRESHOLD:
        s = phi * phi
        return (
            -1.0 / 3.0
            + s / 30.0
            - s * s / 840.0
            + s * s * s / 45360.0
            - s * s * s * s / 3991680.0
        )
    return (phi * math.cos(phi) - math.sin(phi)) / (phi * phi * phi)


def _sinc_curvature(phi: float) -> float:
    """``A2 = (1/φ)·dA/dφ = (3sinφ − 3φcosφ − φ²sinφ)/φ⁵``。

    ``∂²a/∂θ_j∂θ_k = A·δ_jk + A2·θ_j·θ_k``。
    """

    if phi < _SERIES_THRESHOLD:
        s = phi * phi
        return (
            1.0 / 15.0
            - s / 210.0
            + s * s / 7560.0
            - s * s * s / 498960.0
        )
    sin_phi = math.sin(phi)
    numerator = 3.0 * sin_phi - 3.0 * phi * math.cos(phi) - phi * phi * sin_phi
    return numerator / phi**5


def _versine_slope(phi: float) -> float:
    """``B = (1/φ)·d(versine)/dφ = (φsinφ + 2cosφ − 2)/φ⁴``。``∂b/∂θ_k = B·θ_k``。"""

    if phi < _SERIES_THRESHOLD:
        s = phi * phi
        return (
            -1.0 / 12.0
            + s / 180.0
            - s * s / 6720.0
            + s * s * s / 453600.0
        )
    return (phi * math.sin(phi) + 2.0 * math.cos(phi) - 2.0) / phi**4


def _versine_curvature(phi: float) -> float:
    """``B2 = (1/φ)·dB/dφ = (φ²cosφ − 5φsinφ + 8 − 8cosφ)/φ⁶``。

    ``∂²b/∂θ_j∂θ_k = B·δ_jk + B2·θ_j·θ_k``。
    """

    if phi < _SERIES_THRESHOLD:
        s = phi * phi
        return 1.0 / 90.0 - s / 1680.0 + s * s / 75600.0 - s * s * s / 5987520.0
    cos_phi = math.cos(phi)
    numerator = phi * phi * cos_phi - 5.0 * phi * math.sin(phi) + 8.0 - 8.0 * cos_phi
    return numerator / phi**6


def _cross(first: Vec3, second: Vec3) -> Vec3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _assert_theta(theta: tuple[float, ...]) -> Vec3:
    if len(theta) != 3 or not all(math.isfinite(value) for value in theta):
        raise RotationError(f"rotation vector must be a finite 3-vector: {theta!r}")
    return (float(theta[0]), float(theta[1]), float(theta[2]))


def _assert_vec3(vector: tuple[float, ...], what: str) -> Vec3:
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise RotationError(f"{what} must be a finite 3-vector: {vector!r}")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def rotate(theta: tuple[float, ...], vector: tuple[float, ...]) -> Vec3:
    """``exp([θ]×)·v``（Rodrigues）。

    ``θ = 0``时返回**逐位等于**``v``的元组——增量项的结构性零就建在这一条上。
    """

    t = _assert_theta(theta)
    v = _assert_vec3(vector, "vector")
    if t == (0.0, 0.0, 0.0):
        return v
    phi = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
    a = _sinc(phi)
    b = _versine(phi)
    u = _cross(t, v)
    w = _cross(t, u)
    return tuple(v[i] + a * u[i] + b * w[i] for i in range(3))  # type: ignore[return-value]


def rotation_matrix(theta: tuple[float, ...]) -> tuple[Vec3, Vec3, Vec3]:
    """``exp([θ]×)``按行给出。只在需要显式姿态（重输运、与`rigidbody`换坐标）时用。"""

    columns = [rotate(theta, basis) for basis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    return tuple(  # type: ignore[return-value]
        tuple(columns[column][row] for column in range(3)) for row in range(3)
    )


def rotate_jacobian(
    theta: tuple[float, ...], vector: tuple[float, ...]
) -> tuple[Vec3, Vec3, Vec3]:
    """``∂(R(θ)v)/∂θ_k``，按``k``给出三个3矢量。

    ``θ = 0``时退化为``e_k × v``——即经典的"角速度叉乘半径"，
    这条是本函数最容易人工查的一档。
    """

    t = _assert_theta(theta)
    v = _assert_vec3(vector, "vector")
    phi = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
    a = _sinc(phi)
    b = _versine(phi)
    slope_a = _sinc_slope(phi)
    slope_b = _versine_slope(phi)
    u = _cross(t, v)
    w = _cross(t, u)
    result: list[Vec3] = []
    for k in range(3):
        basis = (1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0)
        basis_cross_v = _cross(basis, v)
        term = _cross(basis, u)
        term2 = _cross(t, basis_cross_v)
        result.append(
            tuple(  # type: ignore[arg-type]
                slope_a * t[k] * u[i]
                + a * basis_cross_v[i]
                + slope_b * t[k] * w[i]
                + b * (term[i] + term2[i])
                for i in range(3)
            )
        )
    return tuple(result)  # type: ignore[return-value]


def rotate_hessian(
    theta: tuple[float, ...], vector: tuple[float, ...]
) -> tuple[tuple[Vec3, ...], ...]:
    """``∂²(R(θ)v)/∂θ_j∂θ_k``，按``[j][k]``给出3矢量。**几何刚度的来源。**"""

    t = _assert_theta(theta)
    v = _assert_vec3(vector, "vector")
    phi = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
    b = _versine(phi)
    slope_a = _sinc_slope(phi)
    curve_a = _sinc_curvature(phi)
    slope_b = _versine_slope(phi)
    curve_b = _versine_curvature(phi)
    u = _cross(t, v)
    w = _cross(t, u)
    bases = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    basis_cross_v = [_cross(basis, v) for basis in bases]
    #: ``∂w/∂θ_k = e_k×u + θ×(e_k×v)``
    w_slope = [
        tuple(
            _cross(bases[k], u)[i] + _cross(t, basis_cross_v[k])[i] for i in range(3)
        )
        for k in range(3)
    ]
    rows: list[tuple[Vec3, ...]] = []
    for j in range(3):
        row: list[Vec3] = []
        for k in range(3):
            delta = 1.0 if j == k else 0.0
            second_a = slope_a * delta + curve_a * t[j] * t[k]
            second_b = slope_b * delta + curve_b * t[j] * t[k]
            mixed = _cross(bases[k], basis_cross_v[j])
            mixed2 = _cross(bases[j], basis_cross_v[k])
            row.append(
                tuple(  # type: ignore[arg-type]
                    second_a * u[i]
                    + slope_a * t[k] * basis_cross_v[j][i]
                    + slope_a * t[j] * basis_cross_v[k][i]
                    + second_b * w[i]
                    + slope_b * t[k] * w_slope[j][i]
                    + slope_b * t[j] * w_slope[k][i]
                    + b * (mixed[i] + mixed2[i])
                    for i in range(3)
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def compose(outer: tuple[float, ...], inner: tuple[float, ...]) -> Vec3:
    """``log(exp([outer]×)·exp([inner]×))``——局部图的坐标复合。

    重输运用它把已经转出去的``θ``折进参考姿态。**返回的``|θ|``取主值**``[0, π]``。
    """

    first = rotation_matrix(outer)
    second = rotation_matrix(inner)
    product = tuple(
        tuple(sum(first[r][m] * second[m][c] for m in range(3)) for c in range(3))
        for r in range(3)
    )
    return log_rotation(product)


def log_rotation(matrix: tuple[tuple[float, ...], ...]) -> Vec3:
    """``log``：旋转矩阵回到``θ``。主值``|θ| ∈ [0, π]``。

    ``φ → π``附近用对称部分取轴（反对称部分在那里退化），
    这是本函数唯一一处非平凡分支，**它被`test_rotation.py`的往返门覆盖**。
    """

    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    cos_phi = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    phi = math.acos(cos_phi)
    if phi < 1.0e-8:
        return (
            0.5 * (matrix[2][1] - matrix[1][2]),
            0.5 * (matrix[0][2] - matrix[2][0]),
            0.5 * (matrix[1][0] - matrix[0][1]),
        )
    if math.pi - phi < 1.0e-6:
        #: ``R + I = 2·n⊗n``（当``φ = π``）——取模最大的一列做轴，避免除以近零。
        columns = [
            tuple(matrix[r][c] + (1.0 if r == c else 0.0) for r in range(3))
            for c in range(3)
        ]
        best = max(columns, key=lambda column: sum(v * v for v in column))
        norm = math.sqrt(sum(v * v for v in best))
        axis = tuple(v / norm for v in best)
        #: 符号由反对称部分定；它在``φ = π``处恰为零，此时两个符号等价（``±π·n``同一个旋转）。
        skew = (
            matrix[2][1] - matrix[1][2],
            matrix[0][2] - matrix[2][0],
            matrix[1][0] - matrix[0][1],
        )
        sign = 1.0 if sum(axis[i] * skew[i] for i in range(3)) >= 0.0 else -1.0
        return tuple(sign * phi * value for value in axis)  # type: ignore[return-value]
    scale = phi / (2.0 * math.sin(phi))
    return (
        scale * (matrix[2][1] - matrix[1][2]),
        scale * (matrix[0][2] - matrix[2][0]),
        scale * (matrix[1][0] - matrix[0][1]),
    )


def retransport_levers(
    theta: tuple[float, ...], levers_mm: tuple[Vec3, ...]
) -> tuple[Vec3, ...]:
    """重输运：把当前``θ``折进杠杆臂，调用方随后把``θ``归零。

    **本函数不改状态**——它只回答"新的局部图原点上，这些杠杆臂是什么"。
    归零那一步由调用方在自己的状态向量上做，因为只有它知道哪些下标是转动块。
    """

    return tuple(rotate(theta, lever) for lever in levers_mm)


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidBodyLayout:
    """节点块（质心）+ 转动块。**转动块在节点块之后**，故``node_dof_count``不变。"""

    layout: StateLayout
    node_count: int
    #: 带转动块的节点号，按声明次序。次序即形制（spec/12第2.2节）。
    rotating_bodies: tuple[int, ...]

    def rotation_base(self, body: int) -> int:
        """该体转动块第一个标量的**绝对下标**。调用方永不手写偏移量。"""

        try:
            order = self.rotating_bodies.index(body)
        except ValueError:
            raise RotationError(
                f"node {body} carries no rotation block — "
                f"带转动块的是{list(self.rotating_bodies)}"
            ) from None
        return 3 * self.node_count + 3 * order

    def rotation_indices(self) -> frozenset[int]:
        """全部转动标量下标——"把转动自由度全部钉住"就是把它交给`fixed_indices`。"""

        return frozenset(
            self.rotation_base(body) + axis
            for body in self.rotating_bodies
            for axis in range(3)
        )

    def initial_vector(self, node_positions_mm: tuple[float, ...]) -> tuple[float, ...]:
        """质心位置 + **转动块全零**。局部图的原点就是``θ = 0``，不需要额外初始化。"""

        if len(node_positions_mm) != 3 * self.node_count:
            raise RotationError(
                f"expected {3 * self.node_count} node scalars, got {len(node_positions_mm)}"
            )
        return tuple(node_positions_mm) + (0.0,) * (3 * len(self.rotating_bodies))


def build_rigid_body_layout(
    *, layout_id: str, node_count: int, rotating_bodies: tuple[int, ...]
) -> RigidBodyLayout:
    """按声明造"质心 + 增量转动"的定长布局。次序即声明次序。

    `rotating_bodies`是**节点号**清单：不是每个节点都必须有转动块
    （质点与刚体可以混在同一条向量里，这正是`incline`那类案例要的）。
    """

    if node_count < 1:
        raise RotationError("a rigid-body layout needs at least one node")
    seen: list[int] = []
    for body in rotating_bodies:
        if isinstance(body, bool) or not isinstance(body, int):
            raise RotationError(f"rotating body must be an int node index: {body!r}")
        if not (0 <= body < node_count):
            raise RotationError(
                f"rotating body {body} is outside the node block [0, {node_count})"
            )
        if body in seen:
            raise RotationError(
                f"node {body} declared twice as a rotating body — "
                "重复声明意味着同一个体有两个转动块，而哪一个说了算只能靠读实现"
            )
        seen.append(body)

    fields: list[StateField] = [
        StateField(f"node{index}_{axis}_mm", 1)
        for index in range(node_count)
        for axis in ("x", "y", "z")
    ]
    fields.extend(
        StateField(f"body{body}_theta_{axis}_rad", 1)
        for body in rotating_bodies
        for axis in ("x", "y", "z")
    )
    return RigidBodyLayout(
        layout=StateLayout(
            layout_id=layout_id, fields=tuple(fields), node_dof_count=3 * node_count
        ),
        node_count=node_count,
        rotating_bodies=tuple(rotating_bodies),
    )


# ---------------------------------------------------------------------------
# 物质点粘着弹簧与它的转动增量
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialPoint:
    """刚体上的一个**物质点**：质心节点 + 参考构型下的杠杆臂。

    ``rotation_base``为``None``表示这一端不转（质点，或被钉死的体）——
    那时它退化成一个跟着质心平移的点。
    """

    node: int
    lever_mm: Vec3
    rotation_base: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.node, bool) or not isinstance(self.node, int) or self.node < 0:
            raise RotationError(f"material point node must be a nonnegative int: {self.node!r}")
        _assert_vec3(self.lever_mm, "lever_mm")
        if self.rotation_base is not None:
            if (
                isinstance(self.rotation_base, bool)
                or not isinstance(self.rotation_base, int)
                or self.rotation_base < 0
            ):
                raise RotationError(
                    f"rotation_base must be a nonnegative int or None: {self.rotation_base!r}"
                )


@dataclass(frozen=True)
class StickSpring:
    """两个物质点之间的切向粘着弹簧。``second``为``None``时另一端是世界锚点。

    ``anchor_mm``是**零应力时的相对位移**``p₁ − p₂``（对地时就是世界锚点坐标）。
    """

    first: MaterialPoint
    normal: Vec3
    stiffness_n_per_mm: float
    anchor_mm: Vec3 = (0.0, 0.0, 0.0)
    second: MaterialPoint | None = None

    def __post_init__(self) -> None:
        normal = _assert_vec3(self.normal, "spring normal")
        norm = math.sqrt(sum(value * value for value in normal))
        if abs(norm - 1.0) > 1.0e-12:
            raise RotationError(
                f"spring normal must be a unit vector (|n| = {norm!r}) — "
                "不归一化等于把刚度悄悄乘上|n|²，而调用方以为自己给的是k"
            )
        _assert_vec3(self.anchor_mm, "anchor_mm")
        if not (self.stiffness_n_per_mm > 0.0 and math.isfinite(self.stiffness_n_per_mm)):
            raise RotationError(
                f"tangential stiffness must be positive: {self.stiffness_n_per_mm!r}"
            )
        if self.second is not None and self.second.node == self.first.node:
            raise RotationError(
                "a stick spring between two material points of the same node is a no-op "
                "on the translation block — 声明它多半是把杠杆臂写错了"
            )


def _project(value: Vec3, normal: Vec3) -> Vec3:
    """``P·v = v − (v·n)n``。扣掉法向是这类项的要害（见`TangentialStickSpring`）。"""

    along = value[0] * normal[0] + value[1] * normal[1] + value[2] * normal[2]
    return (
        value[0] - along * normal[0],
        value[1] - along * normal[1],
        value[2] - along * normal[2],
    )


def _reference_offset(vector: tuple[float, ...], spring: StickSpring) -> Vec3:
    """``d₀ = (x₁ + ℓ₁) − (x₂ + ℓ₂) − a``：**不含转动**的相对位移。"""

    first = spring.first
    base = 3 * first.node
    delta = [
        vector[base + axis] + first.lever_mm[axis] - spring.anchor_mm[axis]
        for axis in range(3)
    ]
    if spring.second is not None:
        other = 3 * spring.second.node
        for axis in range(3):
            delta[axis] -= vector[other + axis] + spring.second.lever_mm[axis]
    return (delta[0], delta[1], delta[2])


def _rotation_displacement(vector: tuple[float, ...], spring: StickSpring) -> Vec3:
    """``u = Σ ±(R(θ)ℓ − ℓ)``。**全部``θ``为零时逐位是``(0.0, 0.0, 0.0)``。**"""

    total = [0.0, 0.0, 0.0]
    for sign, point in ((1.0, spring.first), (-1.0, spring.second)):
        if point is None or point.rotation_base is None:
            continue
        base = point.rotation_base
        theta = (vector[base], vector[base + 1], vector[base + 2])
        if theta == (0.0, 0.0, 0.0):
            continue
        rotated = rotate(theta, point.lever_mm)
        for axis in range(3):
            total[axis] += sign * (rotated[axis] - point.lever_mm[axis])
    return (total[0], total[1], total[2])


def _assert_springs(springs: tuple[StickSpring, ...], what: str) -> None:
    if not springs:
        raise RotationError(f"{what} needs at least one spring")


@dataclass(frozen=True)
class MaterialPointStickSpring:
    """``U = Σ ½·k·|P((x₁+ℓ₁) − (x₂+ℓ₂) − a)|²``——**不含转动**的物质点粘着弹簧。

    与`contact.friction.TangentialStickSpring`的差别只有一条但它是承重的：
    **两端都可以是自由体**。金字塔的球-球切向接触两端都在动，
    那个项做不了（它只接"节点对固定锚点"）。

    杠杆臂``ℓ``在本项里是**常量**，所以它对能量的贡献与把锚点平移``ℓ``完全等价——
    这正是"今天的模型"：接触力挂在质心上，力矩无处可去。
    转动进来那一刻它才不再等价，而那是`RotationStickCoupling`的事。
    """

    name: str = "material_stick"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    springs: tuple[StickSpring, ...] = ()

    def __post_init__(self) -> None:
        _assert_springs(self.springs, "material_stick")

    def node_index_bound(self) -> int:
        bound = 0
        for spring in self.springs:
            bound = max(bound, spring.first.node + 1)
            if spring.second is not None:
                bound = max(bound, spring.second.node + 1)
        return bound

    def _offset(self, vector: tuple[float, ...], spring: StickSpring) -> Vec3:
        return _project(_reference_offset(vector, spring), spring.normal)

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for spring in self.springs:
            offset = self._offset(state.vector, spring)
            total += 0.5 * spring.stiffness_n_per_mm * sum(v * v for v in offset)
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for spring in self.springs:
            offset = self._offset(state.vector, spring)
            stiffness = spring.stiffness_n_per_mm
            base = 3 * spring.first.node
            for axis in range(3):
                result[base + axis] += stiffness * offset[axis]
            if spring.second is not None:
                other = 3 * spring.second.node
                for axis in range(3):
                    result[other + axis] -= stiffness * offset[axis]
        return tuple(result)

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        """``k·(I − n⊗n)``的四个符号块。**常量**——粘着是线性的。"""

        entries: list[tuple[int, int, float]] = []
        for spring in self.springs:
            normal = spring.normal
            stiffness = spring.stiffness_n_per_mm
            blocks = [(3 * spring.first.node, 1.0)]
            if spring.second is not None:
                blocks.append((3 * spring.second.node, -1.0))
            for row_base, row_sign in blocks:
                for column_base, column_sign in blocks:
                    scale = stiffness * row_sign * column_sign
                    for a in range(3):
                        for b in range(3):
                            value = scale * (
                                (1.0 if a == b else 0.0) - normal[a] * normal[b]
                            )
                            entries.append((row_base + a, column_base + b, value))
        return tuple(entries)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        size = len(state.vector)
        result = [[0.0] * size for _ in range(size)]
        for row, column, value in self.hessian_entries(state, context):
            result[row][column] += value
        return tuple(tuple(row) for row in result)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        return (
            self.energy(state, context),
            self.gradient(state, context) if need_gradient else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def tangential_force_n(self, state: State) -> tuple[Vec3, ...]:
        """每根弹簧作用在``first``端的切向力``−k·P(...)``。**摩擦锥判的就是它的模。**

        **不含转动增量**——带转动时要用`RotationStickCoupling.tangential_force_n`，
        两者在``θ = 0``上逐位相同。
        """

        return tuple(
            tuple(  # type: ignore[misc]
                -spring.stiffness_n_per_mm * value
                for value in self._offset(state.vector, spring)
            )
            for spring in self.springs
        )


@dataclass(frozen=True)
class RotationStickCoupling:
    """粘着弹簧的**转动增量**：``ΔU = Σ ½k(|P(d₀+u)|² − |P(d₀)|²)``。

    ## 它给出的正是"接触力对质心取矩"

    ``∂ΔU/∂θ_k = k·P(d₀+u)·P(∂(Rℓ)/∂θ_k)``，而``θ = 0``时
    ``∂(Rℓ)/∂θ_k = e_k × ℓ``，于是该式恰为``−(ℓ × f)_k = −M_k``。
    **令它为零就是``ΣM = 0``**——三球金字塔精确解里那三条被丢掉的方程。

    ## 退化是结构性的，不是数值上的巧合

    ``u``在全部``θ = 0``时逐位是``(0.0, 0.0, 0.0)``（`rotate`在``θ = 0``上返回输入本身），
    于是：能量增量为``0.5·k·(s − s)``即**恰为0.0**；节点块梯度为``k·(o − o)``即**恰为0.0**；
    节点块Hessian**结构上恒为零**（``∂²ΔU/∂x∂x = kP − kP``，任何``θ``下都是零，
    因此本项一个节点-节点条目都不出）。

    把转动块交给`fixed_indices`时，约化系统与不带转动块的系统**逐位相同**。

    ## Hessian的三块

    * ``∂²/∂x∂θ = ±k·P(∂(Rℓ)/∂θ_k)``——平动与转动的耦合；
    * ``∂²/∂θ∂θ``第一块``k·P(∂(Rℓ)/∂θ_j)·P(∂(Rℓ)/∂θ_k)``——弹簧本身的刚度；
    * ``∂²/∂θ∂θ`` 第二块``k·P(d₀+u)·P(∂²(Rℓ)/∂θ_j∂θ_k)``——**几何刚度**，
      即力矩随姿态的变化。漏掉它梯度照样对、平衡点照样对，
      **只有收敛速度与切线刚度的正定性判据会变**（与`PenaltySphereContact`同源）。
    """

    name: str = "rotation_coupling"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    springs: tuple[StickSpring, ...] = ()

    def __post_init__(self) -> None:
        _assert_springs(self.springs, "rotation_coupling")
        for spring in self.springs:
            rotating = [
                point
                for point in (spring.first, spring.second)
                if point is not None and point.rotation_base is not None
            ]
            if not rotating:
                raise RotationError(
                    "rotation_coupling was handed a spring with no rotating end — "
                    "那是一个恒为零的项，声明它只会让读者以为转动接上了"
                )

    def node_index_bound(self) -> int:
        bound = 0
        for spring in self.springs:
            bound = max(bound, spring.first.node + 1)
            if spring.second is not None:
                bound = max(bound, spring.second.node + 1)
        return bound

    def _ends(self, spring: StickSpring):
        for sign, point in ((1.0, spring.first), (-1.0, spring.second)):
            if point is not None and point.rotation_base is not None:
                yield sign, point

    def _offsets(self, vector: tuple[float, ...], spring: StickSpring) -> tuple[Vec3, Vec3]:
        """``(P(d₀), P(d₀+u))``。``u``为零时**返回同一个对象**，增量因此逐位为零。"""

        reference = _reference_offset(vector, spring)
        base_offset = _project(reference, spring.normal)
        shift = _rotation_displacement(vector, spring)
        if shift == (0.0, 0.0, 0.0):
            return base_offset, base_offset
        moved = (
            reference[0] + shift[0],
            reference[1] + shift[1],
            reference[2] + shift[2],
        )
        return base_offset, _project(moved, spring.normal)

    def energy(self, state: State, context: EnergyContext) -> float:
        total = 0.0
        for spring in self.springs:
            base_offset, offset = self._offsets(state.vector, spring)
            half_k = 0.5 * spring.stiffness_n_per_mm
            total += half_k * sum(v * v for v in offset) - half_k * sum(
                v * v for v in base_offset
            )
        return total

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for spring in self.springs:
            base_offset, offset = self._offsets(state.vector, spring)
            stiffness = spring.stiffness_n_per_mm
            node_blocks = [(3 * spring.first.node, 1.0)]
            if spring.second is not None:
                node_blocks.append((3 * spring.second.node, -1.0))
            for node_base, node_sign in node_blocks:
                for axis in range(3):
                    result[node_base + axis] += node_sign * (
                        stiffness * offset[axis] - stiffness * base_offset[axis]
                    )
            for sign, point in self._ends(spring):
                theta = (
                    state.vector[point.rotation_base],
                    state.vector[point.rotation_base + 1],
                    state.vector[point.rotation_base + 2],
                )
                jacobian = rotate_jacobian(theta, point.lever_mm)
                for k in range(3):
                    direction = _project(jacobian[k], spring.normal)
                    result[point.rotation_base + k] += (
                        sign
                        * stiffness
                        * sum(offset[i] * direction[i] for i in range(3))
                    )
        return tuple(result)

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        entries: list[tuple[int, int, float]] = []
        for spring in self.springs:
            _, offset = self._offsets(state.vector, spring)
            stiffness = spring.stiffness_n_per_mm
            node_blocks = [(3 * spring.first.node, 1.0)]
            if spring.second is not None:
                node_blocks.append((3 * spring.second.node, -1.0))
            ends = list(self._ends(spring))
            directions: dict[int, list[Vec3]] = {}
            for sign, point in ends:
                base = point.rotation_base
                assert base is not None
                theta = (
                    state.vector[base],
                    state.vector[base + 1],
                    state.vector[base + 2],
                )
                jacobian = rotate_jacobian(theta, point.lever_mm)
                directions[base] = [_project(jacobian[k], spring.normal) for k in range(3)]
                #: 几何刚度：``offset·P(∂²(Rℓ)/∂θ_j∂θ_k)``。
                curvature = rotate_hessian(theta, point.lever_mm)
                for j in range(3):
                    for k in range(3):
                        second = _project(curvature[j][k], spring.normal)
                        entries.append(
                            (
                                base + j,
                                base + k,
                                sign
                                * stiffness
                                * sum(offset[i] * second[i] for i in range(3)),
                            )
                        )
                #: 平动-转动耦合。**节点-节点块一个条目都不出**——它结构上恒为零。
                for node_base, node_sign in node_blocks:
                    for m in range(3):
                        for k in range(3):
                            value = (
                                node_sign * sign * stiffness * directions[base][k][m]
                            )
                            entries.append((node_base + m, base + k, value))
                            entries.append((base + k, node_base + m, value))
            #: 两端都转时的转动-转动交叉块。
            for row_sign, row_point in ends:
                for column_sign, column_point in ends:
                    if row_point.rotation_base == column_point.rotation_base:
                        continue
                    row_base = row_point.rotation_base
                    column_base = column_point.rotation_base
                    assert row_base is not None and column_base is not None
                    for j in range(3):
                        for k in range(3):
                            entries.append(
                                (
                                    row_base + j,
                                    column_base + k,
                                    row_sign
                                    * column_sign
                                    * stiffness
                                    * sum(
                                        directions[row_base][j][i]
                                        * directions[column_base][k][i]
                                        for i in range(3)
                                    ),
                                )
                            )
            #: 同一端的弹簧刚度块（与几何刚度相加）。
            for sign, point in ends:
                base = point.rotation_base
                assert base is not None
                for j in range(3):
                    for k in range(3):
                        entries.append(
                            (
                                base + j,
                                base + k,
                                stiffness
                                * sum(
                                    directions[base][j][i] * directions[base][k][i]
                                    for i in range(3)
                                ),
                            )
                        )
        return tuple(entries)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        size = len(state.vector)
        result = [[0.0] * size for _ in range(size)]
        for row, column, value in self.hessian_entries(state, context):
            result[row][column] += value
        return tuple(tuple(row) for row in result)

    def quantities(self, state, context, *, need_gradient, need_hessian):
        return (
            self.energy(state, context),
            self.gradient(state, context) if need_gradient else None,
            self.hessian(state, context) if need_hessian else None,
        )

    def tangential_force_n(self, state: State) -> tuple[Vec3, ...]:
        """**带转动**的切向力``−k·P(d₀+u)``，作用在``first``端。"""

        return tuple(
            tuple(  # type: ignore[misc]
                -spring.stiffness_n_per_mm * value
                for value in self._offsets(state.vector, spring)[1]
            )
            for spring in self.springs
        )

    def contact_moment_n_mm(self, state: State) -> tuple[tuple[int, Vec3], ...]:
        """每根弹簧在每个转动端上的力矩``ℓ' × f``（``ℓ' = R(θ)ℓ``）。

        它与``−∂ΔU/∂θ``在``θ = 0``上逐位相同；``θ ≠ 0``时两者差一个投影
        （广义力不是笛卡尔力矩），**这条差别写在这里是因为下一个人一定会拿它们对拍**。
        """

        result: list[tuple[int, Vec3]] = []
        for index, spring in enumerate(self.springs):
            _, offset = self._offsets(state.vector, spring)
            force = tuple(-spring.stiffness_n_per_mm * value for value in offset)
            for sign, point in self._ends(spring):
                base = point.rotation_base
                assert base is not None
                theta = (state.vector[base], state.vector[base + 1], state.vector[base + 2])
                lever = rotate(theta, point.lever_mm)
                moment = _cross(lever, tuple(sign * v for v in force))  # type: ignore[arg-type]
                result.append((index, moment))
        return tuple(result)


@dataclass(frozen=True)
class AppliedMoment:
    """转动块上的固定广义力矩：``U = −Σ M·θ``（单位N·mm）。`PointLoad`的转动对应物。

    **符号与`PointLoad`逐字同源**：外力矩做的功是**负**势能。写成``+M·θ``
    求解器照样收敛，只是收敛到一个物理上相反的解。

    ## 它是**当前局部图里的广义力矩**，不是体固连扭矩

    ``U``对``θ``线性，故``gradient = −M``、``hessian``**恒为零**。
    这在``θ = 0``（局部图原点）上就是笛卡尔力矩；``θ ≠ 0``时它仍是
    "与``θ_k``共轭的广义力"，**而那与"绕体固连轴的扭矩"不是同一个量**。
    重输运把``θ``归零，因此按0079的口径用它时两者一致。
    **这条限定写在这里，是因为下一个人最可能犯的错是把它当成随体转的扭矩。**

    零Hessian的陷阱与`PointLoad`逐字相同：只含本项的注册表给出全零Hessian，
    `solve_equilibrium`在那里必须失败关闭而不是返回垃圾解。
    """

    name: str = "applied_moment"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: (转动块首标量的绝对下标, (Mx, My, Mz) 单位N·mm)。同一个块不许出现两次——
    #: 两条同块力矩该由调用方自己合并，库不替它猜求和次序（浮点加法不结合）。
    moments: tuple[tuple[int, Vec3], ...] = ()

    def __post_init__(self) -> None:
        if not self.moments:
            raise RotationError("applied_moment needs at least one moment")
        seen: set[int] = set()
        for base, moment in self.moments:
            if isinstance(base, bool) or not isinstance(base, int) or base < 0:
                raise RotationError(f"moment rotation base must be a nonnegative int: {base!r}")
            if base in seen:
                raise RotationError(
                    f"rotation block {base} carries two applied moments — "
                    "同块两条该由调用方自己合并，库不替它猜求和次序"
                )
            seen.add(base)
            _assert_vec3(moment, "applied moment")

    def node_index_bound(self) -> int:
        """本项**不按索引取节点**——它只写转动块，故返回0（协议规定的取值）。"""

        return 0

    def energy(self, state: State, context: EnergyContext) -> float:
        return -sum(
            sum(moment[axis] * state.vector[base + axis] for axis in range(3))
            for base, moment in self.moments
        )

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        result = [0.0] * len(state.vector)
        for base, moment in self.moments:
            for axis in range(3):
                result[base + axis] -= moment[axis]
        return tuple(result)

    def hessian_entries(self, state, context) -> tuple[tuple[int, int, float], ...]:
        return ()

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        size = len(state.vector)
        return tuple(tuple(0.0 for _ in range(size)) for _ in range(size))

    def quantities(self, state, context, *, need_gradient, need_hessian):
        return (
            self.energy(state, context),
            self.gradient(state, context) if need_gradient else None,
            self.hessian(state, context) if need_hessian else None,
        )


__all__ = [
    "AppliedMoment",
    "MaterialPoint",
    "MaterialPointStickSpring",
    "RigidBodyLayout",
    "RotationError",
    "RotationStickCoupling",
    "StickSpring",
    "build_rigid_body_layout",
    "compose",
    "log_rotation",
    "retransport_levers",
    "rotate",
    "rotate_hessian",
    "rotate_jacobian",
    "rotation_matrix",
]
