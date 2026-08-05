"""刚体转动与自由飞行——spec/12第四节在三维姿态上的落地（决策0043）。

**在它之前引擎只有质点。** `integrate.py`（决策0019）推进的是位置与速度，
三个自由度、没有姿态；`geometry.py`（决策0022）早就把四种解析原语的惯量张量
算好了，却**没有任何东西去用它**。本模块把那两半接上：体坐标系的Euler方程

    I·ω̇ + ω × (I·ω) = τ

加一条单位四元数的姿态运动学，合成六自由度的自由飞行。惯量张量**直接取
`geometry.mass_properties`**，本模块一行都不重推——重推一遍就等于给同一个
物理量造了第二个真值来源。

## 一、姿态为什么是单位四元数，不是欧拉角

1. **万向锁是定义上的洞，不是数值上的难**。任何三参数姿态表示都在SO(3)上有
   坐标奇点（拓扑事实：SO(3)盖不住一张三维图卡）。取ZYX欧拉角时运动学映射
   `ω → (φ̇, θ̇, ψ̇)`里带`1/cos θ`，`θ = ±90°`处右端项**根本不存在**。
   积分器在那里不是"精度掉了"，是被喂了一个没有定义的导数。
   自由飞行的姿态是全程乱转的，它**必然**穿过那个角。
2. **欧拉角的约定有24种**（12种轴序×内旋/外旋），换一种就是换一个物理，
   而两仓之间只能靠口头约定对齐。本仓已经有一套四元数约定在用
   （`shapes.PosedBody.rotation_xyzw`、`motion`的位姿时间线，都是**xyzw**次序、
   都表示**体→世界**），再引入第二套姿态词汇是自找的。
3. 四元数唯一的缺陷是**双重覆盖**（`q`与`−q`是同一个旋转）。这条本仓已经登记过：
   `motion`的`rotation_arc`语义就是为它立的（"q与−q是同一个旋转但插值走相反的弧"）。
   它影响的是**插值走哪条弧**，不影响运动学微分方程——`q̇ = ½·q ⊗ (ω, 0)`
   对`q`与`−q`给出同一条轨迹。所以本模块不需要为它做任何事，只需要不假装它不存在。

**不选旋转矩阵**的理由同样具体：9个数带6个约束，正交性漂移要靠Gram-Schmidt
拉回，而那是一次**六维**的投影；四元数是4个数带1个约束，拉回就是一次除法。

## 二、归一化漂移：怎么处理，门守在哪一侧

`|q| = 1`不是任何显式积分器的不变量——RK4推进的是`q̇`，它把`q`推离单位球。
本模块的处置是**每步归一化一次**（与`motion._normalise_quaternion`同一句话：
"这不是语义变换，是把一次浮点舍入的偏离收回来"）。

**门必须守在归一化之前那一侧。** 归一化之后`|q|`恒等于1（到舍入为止），
所以"断言`||q| − 1| < ε`"是一条**永远通过**的断言——spec/12第6.2节点名的那类
假通过，只是换了个物理外衣。因此：

* 诊断报出的是**归一化前**的单步最大偏离`max ||q| − 1|`；
* 超过`QUATERNION_NORM_STEP_ABS_TOL`即**失败关闭**，不是二分重试、不是静默拉回——
  单步就把四元数推离单位球那么远，说明步长对这个转速已经没有意义了；
* 归一化次数是个**确定性整数**，一并报出（决策0018：进门的是确定性量）。

## 三、单位边界（本仓已经栽过的那一类）

状态是mm制：惯量`kg·mm²`（`geometry`的`inertia_about_centroid_kg_mm2`）、
角速度`rad/s`、力矩`N·mm`、转动动能`N·mm`。而`1 N·mm = 1000 kg·mm²/s²`，
所以Euler方程里力矩那一项必须乘`MM_PER_M`。这个常量**从`energies`导入而不是
在这里再写一个1000**——同一个换算有两份字面量，迟早只改一份
（spec/14第五节盯的正是这类静默1000倍；`energies`的重力项已经栽过一次）。

**本模块不管什么**：无接触、无约束、无关节与多体、无隐式族、无自适应步长；
不做柔性体；质量属性一律走`geometry`（带法兰的圆柱与网格资产在那里就失败关闭，
本模块不给它们开后门）。姿态与质心平动在**自由**刚体上严格解耦，本模块把两者
放在同一个状态向量里一起推，是为了让"自由飞行"是一个状态而不是两次调用。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from physics_engine.energies import MM_PER_M
from physics_engine.geometry import Matrix3, mass_properties
from physics_engine.integrate import IntegratorDeclaration, VectorOps, default_ops
from physics_engine.shapes import Shape
from physics_engine.state import State, StateField, StateLayout

Vector3 = tuple[float, float, float]
#: 单位四元数，**xyzw**次序、表示**体→世界**——与`shapes.PosedBody.rotation_xyzw`
#: 和`motion.Pose.rotation_xyzw`是同一个约定。本仓只有这一套。
Quaternion = tuple[float, float, float, float]
Vector = tuple[float, ...]

#: 世界系外力回调：`f(state_vector, t) -> (fx, fy, fz)`，单位N。
ForceCallback = Callable[[Vector, float], Sequence[float]]
#: **体坐标系**外力矩回调：`τ(state_vector, t) -> (τx, τy, τz)`，单位N·mm。
#: 体系而不是世界系，是因为Euler方程本身写在体系里——在这里换系等于把一次
#: 坐标变换藏进回调的约定里，而那正是"体系/惯性系混淆"的温床。
TorqueCallback = Callable[[Vector, float], Sequence[float]]

#: 单步归一化**之前**允许的`||q| − 1|`。比`motion.QUATERNION_NORM_ABS_TOL`
#: （1e-9，那是**声明**一个四元数时的入口容差）松三个量级，因为这里量的是
#: 一步积分自己造出来的偏离：RK4在`q̇ = ½q⊗ω`上的单步截断量级是`(ωh)⁵`，
#: 而入口容差量的是"调用方给的这个四元数配不配叫单位四元数"，两件事不同源。
#: 取1e-6的判据：`|q|`偏离1e-6时旋转矩阵`R = R(q)/|q|²`的元素偏离约2e-6mm/mm,
#: 在mm制下对一个百毫米级的体是亚微米——**已经超出本模块任何判据的容差**,
#: 所以到这里就该停，而不是继续算下去再让判据去发现。
QUATERNION_NORM_STEP_ABS_TOL = 1.0e-6

#: 惯量张量的对称性与三角不等式检查的相对松量。纯几何量由`geometry`闭式给出，
#: 舍入在ulp量级，1e-9留六个数量级余量；它挡的是**手写错的张量**，不是舍入。
INERTIA_REL_TOL = 1.0e-9

#: 自由飞行的打包次序（spec/12第2.2节："打包次序是形制的一部分"）。
#: 四块：质心位置、质心速度、体系角速度、体→世界姿态四元数，共13个自由度。
#:
#: **次序不是随手排的**：前六个与`integrate.py`推的质点态同序（位置在前速度在后），
#: 于是自由刚体的平动那一半可以与`cases/ballistic_free_flight`逐条对照；
#: 转动那两块排在后面，角速度在姿态之前——因为Euler方程算`ω̇`不需要`q`,
#: 而姿态运动学算`q̇`需要`ω`，按依赖方向排，读代码的人不必回头找。
#:
#: 四元数分量是**真无量纲**，因此显式声明`is_dimensionless`
#: （轴2规则5：无量纲必须列出来，不是"没写单位就算没有"）。
RIGID_BODY_LAYOUT = StateLayout(
    layout_id="layout/rigid_body_free_flight/v1",
    fields=(
        StateField(name="centre_of_mass_position_mm", width=3),
        StateField(name="centre_of_mass_velocity_mm_per_s", width=3),
        StateField(name="angular_velocity_body_rad_per_s", width=3),
        StateField(
            name="attitude_body_to_world_xyzw", width=4, is_dimensionless=True
        ),
    ),
)

_POSITION = "centre_of_mass_position_mm"
_VELOCITY = "centre_of_mass_velocity_mm_per_s"
_OMEGA = "angular_velocity_body_rad_per_s"
_ATTITUDE = "attitude_body_to_world_xyzw"


def _slice_of(name: str) -> slice:
    """块在一维向量里的区间——**由布局算，不写死**。

    积分器内部曾经写过`y[9:13]`这种字面量。那是本页最容易犯的错：布局改了
    它不跟着改，而`fingerprint()`会照常变、门会照常绿（因为门比的是布局，
    不是积分器里那个数）。所以偏移量在这里一次性从布局导出，
    与`State.block`"调用方永不手写偏移量"是同一条纪律。
    """

    offset = RIGID_BODY_LAYOUT.offset_of(name)
    width = next(
        field.width for field in RIGID_BODY_LAYOUT.fields if field.name == name
    )
    return slice(offset, offset + width)


_POSITION_SLICE = _slice_of(_POSITION)
_VELOCITY_SLICE = _slice_of(_VELOCITY)
_OMEGA_SLICE = _slice_of(_OMEGA)
_ATTITUDE_SLICE = _slice_of(_ATTITUDE)

#: 导数内部把四块算成四个元组，次序是这个；下面的置换把它们按**布局**次序拼回去。
_BLOCK_ORDER = (_POSITION, _VELOCITY, _OMEGA, _ATTITUDE)
_ASSEMBLY = tuple(
    _BLOCK_ORDER.index(field.name) for field in RIGID_BODY_LAYOUT.fields
)


class RigidBodyError(ValueError):
    """刚体层的一切失败关闭。"""


# ---------------------------------------------------------------------------
# 四元数与向量的小工具（xyzw，体→世界）
# ---------------------------------------------------------------------------


def cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    """叉乘。**次序即物理**：`ω × (I·ω)`与`(I·ω) × ω`差一个负号，
    而两者都能量守恒——只有带符号的判据分得开（见案例第三节）。"""

    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> Quaternion:
    """Hamilton积，xyzw次序。**不可交换**——`q ⊗ p ≠ p ⊗ q`。

    姿态运动学取`q̇ = ½·q ⊗ (ω_body, 0)`：ω在**体**系时四元数左乘、
    ω在**世界**系时右乘（`½·(ω_world, 0) ⊗ q`）。写反了姿态照样在转、
    角速度照样守恒，只有**惯性系**里的量能指出来（案例第六节的必红3）。
    """

    x1, y1, z1, w1 = left[0], left[1], left[2], left[3]
    x2, y2, z2, w2 = right[0], right[1], right[2], right[3]
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def attitude_matrix(quaternion: Sequence[float]) -> Matrix3:
    """体→世界的旋转矩阵`R(q)`。

    与`shapes.PosedBody.rotate_local_mm`是**同一个**式子，测试逐位对拍两者——
    引擎里只许有一套四元数约定，第二套一出现就是静默的坐标系错。
    """

    x, y, z, w = (
        quaternion[0], quaternion[1], quaternion[2], quaternion[3],
    )
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def rotate_body_to_world(quaternion: Sequence[float], vector: Sequence[float]) -> Vector3:
    """体系向量 → 世界系向量。反向（世界→体）用转置，**没有隐式路径**。"""

    rows = attitude_matrix(quaternion)
    return tuple(  # type: ignore[return-value]
        sum(row[index] * vector[index] for index in range(3)) for row in rows
    )


def rotate_world_to_body(quaternion: Sequence[float], vector: Sequence[float]) -> Vector3:
    """世界系向量 → 体系向量（`Rᵀ`）。与上一个函数是彼此的逆，测试往返一次。"""

    rows = attitude_matrix(quaternion)
    return tuple(  # type: ignore[return-value]
        sum(rows[index][axis] * vector[index] for index in range(3)) for axis in range(3)
    )


def normalise_quaternion(quaternion: Sequence[float]) -> Quaternion:
    """投回单位球。零四元数失败关闭——它不表示任何旋转。"""

    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        raise RigidBodyError("quaternion collapsed to zero — it is not a rotation")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 惯量：直接用geometry算好的那个，本模块不重推
# ---------------------------------------------------------------------------


def _symmetric_eigenvalues(matrix: Matrix3) -> Vector3:
    """对称3×3的三个特征值（闭式，降序）。用于主惯量矩与三角不等式检查。

    闭式而不是迭代：3×3对称阵的特征多项式是三次的，有实根的三角解，
    因此不需要Jacobi迭代，也就不需要"迭代到什么时候停"这个额外声明。
    """

    a00, a01, a02 = matrix[0]
    _, a11, a12 = matrix[1]
    _, _, a22 = matrix[2]
    off = a01 * a01 + a02 * a02 + a12 * a12
    if off == 0.0:
        return tuple(sorted((a00, a11, a22), reverse=True))  # type: ignore[return-value]
    trace_third = (a00 + a11 + a22) / 3.0
    deviation = (
        (a00 - trace_third) ** 2
        + (a11 - trace_third) ** 2
        + (a22 - trace_third) ** 2
        + 2.0 * off
    )
    radius = math.sqrt(deviation / 6.0)
    b = tuple(
        tuple(
            (matrix[i][j] - (trace_third if i == j else 0.0)) / radius for j in range(3)
        )
        for i in range(3)
    )
    determinant = (
        b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])
    )
    angle = math.acos(min(1.0, max(-1.0, determinant / 2.0))) / 3.0
    high = trace_third + 2.0 * radius * math.cos(angle)
    low = trace_third + 2.0 * radius * math.cos(angle + 2.0 * math.pi / 3.0)
    return (high, 3.0 * trace_third - high - low, low)


@dataclass(frozen=True)
class RigidBodyInertia:
    """绕**质心**、在**体**坐标系下表达的惯量张量（kg·mm²）加质量（kg）。

    参考点与坐标系都写进了字段名与本文档——`geometry`那一页已经把
    "换参考点必须显式调平行轴定理"立成条款，本层原样继承，不提供隐式路径。

    三条入口校验，都是"手写错的张量"才会破而舍入不会破的：对称、主惯量为正、
    **三角不等式**`I₁ + I₂ ≥ I₃`。第三条最有用：它是任何真实质量分布都满足的
    充要形状约束，一个凭空捏的对角阵（比如把三个数随手写成1、1、10）会当场被挡下。
    """

    mass_kg: float
    inertia_body_kg_mm2: Matrix3

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise RigidBodyError(f"mass_kg must be positive and finite: {self.mass_kg!r}")
        matrix = self.inertia_body_kg_mm2
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise RigidBodyError("inertia_body_kg_mm2 must be a 3x3 matrix")
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise RigidBodyError("inertia_body_kg_mm2 must be finite")
        scale = max(abs(value) for row in matrix for value in row)
        if scale <= 0.0:
            raise RigidBodyError("inertia_body_kg_mm2 is all zeros")
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(matrix[i][j] - matrix[j][i]) > INERTIA_REL_TOL * scale:
                    raise RigidBodyError(
                        f"inertia tensor is not symmetric at ({i},{j}): "
                        f"{matrix[i][j]!r} vs {matrix[j][i]!r}"
                    )
        moments = _symmetric_eigenvalues(matrix)
        if moments[2] <= 0.0:
            raise RigidBodyError(
                f"principal moments must be positive, got {moments!r} — "
                "一个非正的主惯量不对应任何质量分布"
            )
        biggest = moments[0]
        if moments[1] + moments[2] < biggest * (1.0 - INERTIA_REL_TOL):
            raise RigidBodyError(
                f"principal moments {moments!r} break the triangle inequality "
                "I1 + I2 >= I3 — 没有任何质量分布给得出这样一个张量"
            )

    @classmethod
    def from_shape(
        cls,
        shape: Shape,
        *,
        density_kg_m3: float | None = None,
        mass_kg: float | None = None,
    ) -> RigidBodyInertia:
        """从形状取质量属性——**走`geometry.mass_properties`，不重推**。

        `geometry`那一层的失败关闭（带法兰的圆柱、网格资产）因此原样继承：
        本层不知道的几何，本层也不猜。
        """

        properties = mass_properties(
            shape, density_kg_m3=density_kg_m3, mass_kg=mass_kg
        )
        return cls(
            mass_kg=properties.mass_kg,
            inertia_body_kg_mm2=properties.inertia_about_centroid_kg_mm2,
        )

    def principal_moments_kg_mm2(self) -> Vector3:
        """三个主惯量矩，降序。"""

        return _symmetric_eigenvalues(self.inertia_body_kg_mm2)

    def apply(self, omega: Sequence[float]) -> Vector3:
        """`I·ω`——体系角动量，单位kg·mm²/s。"""

        matrix = self.inertia_body_kg_mm2
        return tuple(  # type: ignore[return-value]
            sum(matrix[i][j] * omega[j] for j in range(3)) for i in range(3)
        )

    def solve(self, vector: Sequence[float]) -> Vector3:
        """`I⁻¹·v`。奇异即失败关闭（对角占优的惯量张量不该奇异，奇异说明它是错的）。"""

        m = self.inertia_body_kg_mm2
        cofactor = (
            (
                m[1][1] * m[2][2] - m[1][2] * m[2][1],
                m[0][2] * m[2][1] - m[0][1] * m[2][2],
                m[0][1] * m[1][2] - m[0][2] * m[1][1],
            ),
            (
                m[1][2] * m[2][0] - m[1][0] * m[2][2],
                m[0][0] * m[2][2] - m[0][2] * m[2][0],
                m[0][2] * m[1][0] - m[0][0] * m[1][2],
            ),
            (
                m[1][0] * m[2][1] - m[1][1] * m[2][0],
                m[0][1] * m[2][0] - m[0][0] * m[2][1],
                m[0][0] * m[1][1] - m[0][1] * m[1][0],
            ),
        )
        determinant = (
            m[0][0] * cofactor[0][0] + m[0][1] * cofactor[1][0] + m[0][2] * cofactor[2][0]
        )
        scale = max(abs(value) for row in m for value in row)
        if abs(determinant) <= INERTIA_REL_TOL * scale**3:
            raise RigidBodyError("inertia tensor is singular — I⁻¹ does not exist")
        return tuple(  # type: ignore[return-value]
            sum(cofactor[i][j] * vector[j] for j in range(3)) / determinant
            for i in range(3)
        )


# ---------------------------------------------------------------------------
# 状态的读写：偏移量由布局算，调用方永不手写
# ---------------------------------------------------------------------------


def make_state(
    *,
    position_mm: Sequence[float] = (0.0, 0.0, 0.0),
    velocity_mm_per_s: Sequence[float] = (0.0, 0.0, 0.0),
    angular_velocity_rad_per_s: Sequence[float] = (0.0, 0.0, 0.0),
    attitude_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
) -> State:
    """按`RIGID_BODY_LAYOUT`装一个状态。四元数在入口就要求是单位的。"""

    for name, block, width in (
        ("position_mm", position_mm, 3),
        ("velocity_mm_per_s", velocity_mm_per_s, 3),
        ("angular_velocity_rad_per_s", angular_velocity_rad_per_s, 3),
        ("attitude_xyzw", attitude_xyzw, 4),
    ):
        if len(block) != width:
            raise RigidBodyError(f"{name} must have {width} components: {block!r}")
    norm = math.sqrt(sum(value * value for value in attitude_xyzw))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=QUATERNION_NORM_STEP_ABS_TOL):
        raise RigidBodyError(
            f"attitude_xyzw must be a unit quaternion, norm is {norm!r}"
        )
    blocks = {
        _POSITION: position_mm,
        _VELOCITY: velocity_mm_per_s,
        _OMEGA: angular_velocity_rad_per_s,
        _ATTITUDE: attitude_xyzw,
    }
    return State(
        layout=RIGID_BODY_LAYOUT,
        vector=tuple(
            float(value)
            for field in RIGID_BODY_LAYOUT.fields  # **按布局次序装**，不按参数次序
            for value in blocks[field.name]
        ),
    )


def _require_layout(state: State) -> None:
    """布局指纹是**进函数的门**。次序换了指纹就变，本函数当场拒——
    spec/12第2.2节说次序换了"多数测试不会发现"，这里让它发现。"""

    if state.layout.fingerprint() != RIGID_BODY_LAYOUT.fingerprint():
        raise RigidBodyError(
            f"state layout {state.layout.layout_id!r} does not match "
            f"{RIGID_BODY_LAYOUT.layout_id!r} — 打包次序是形制的一部分，"
            "指纹不同就是另一个契约"
        )


def angular_velocity_body_rad_per_s(state: State) -> Vector3:
    _require_layout(state)
    return state.block(_OMEGA)  # type: ignore[return-value]


def attitude_xyzw(state: State) -> Quaternion:
    _require_layout(state)
    return state.block(_ATTITUDE)  # type: ignore[return-value]


def angular_momentum_world_kg_mm2_per_s(
    inertia: RigidBodyInertia, state: State
) -> Vector3:
    """**惯性系**角动量`L = R(q)·(I·ω)`，单位kg·mm²/s。

    无力矩时守恒的是**这个**，不是体系里的`I·ω`——体系里那个对一般刚体
    根本不守恒（它就是Euler方程在转的那个量）。本仓的判据必须写在惯性系里,
    案例第六节的三条必红全靠它抓。
    """

    _require_layout(state)
    return rotate_body_to_world(state.block(_ATTITUDE), inertia.apply(state.block(_OMEGA)))


def angular_momentum_body_kg_mm2_per_s(
    inertia: RigidBodyInertia, state: State
) -> Vector3:
    """体系角动量`I·ω`。**它不守恒**——提供它是为了让案例能断言"它确实在变",
    从而证明惯性系那条判据不是恒等式（spec/12第6.2节的堵法：先断参照量非零）。"""

    _require_layout(state)
    return inertia.apply(state.block(_OMEGA))


def rotational_kinetic_energy_nmm(inertia: RigidBodyInertia, state: State) -> float:
    """转动动能`T = ½·ωᵀ·I·ω`，单位N·mm（`kg·mm²/s²`除以`MM_PER_M`）。"""

    _require_layout(state)
    omega = state.block(_OMEGA)
    moment = inertia.apply(omega)
    return 0.5 * sum(a * b for a, b in zip(omega, moment, strict=True)) / MM_PER_M


# ---------------------------------------------------------------------------
# 时间推进：五项出生声明 + 一份公式源两个求值后端
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidBodyIntegrator:
    """转动积分器。声明用`integrate.IntegratorDeclaration`——**同一份五项契约**，
    不给转动另开一套（另开一套就意味着有一天两套会长得不一样）。"""

    declaration: IntegratorDeclaration
    step: Callable[
        [Vector, float, float, Callable[[Vector, float], Vector], VectorOps], Vector
    ]


def _rk4_step(y, t, h, derivative, ops):
    """经典RK4。`ops`只承担线性组合——导数是"调用方的代码"，两个后端都用元组调它,
    与`integrate.py`的加速档边界画在同一处（决策0019第2.2节的诚实边界）。"""

    y0 = ops.load(y)
    k1 = ops.load(derivative(y, t))
    k2 = ops.load(
        derivative(ops.dump(ops.add(y0, ops.scale(0.5 * h, k1))), t + 0.5 * h)
    )
    k3 = ops.load(
        derivative(ops.dump(ops.add(y0, ops.scale(0.5 * h, k2))), t + 0.5 * h)
    )
    k4 = ops.load(derivative(ops.dump(ops.add(y0, ops.scale(h, k3))), t + h))
    weighted = ops.add(
        ops.add(k1, ops.scale(2.0, k2)), ops.add(ops.scale(2.0, k3), k4)
    )
    return ops.dump(ops.add(y0, ops.scale(h / 6.0, weighted)))


def _explicit_euler_step(y, t, h, derivative, ops):
    """一阶显式Euler。存在的理由只有一个：给漂移**排序**判据当对照组
    （spec/12第6.2节写法2——排序断言不受实现常数影响，是弱数值环境下最稳的判据）。"""

    return ops.dump(ops.add(ops.load(y), ops.scale(h, ops.load(derivative(y, t)))))


RK4_BODY = RigidBodyIntegrator(
    declaration=IntegratorDeclaration(
        name="rk4_rigid_body",
        scope_excludes=(
            "不管接触、不管约束与关节、不管刚性问题；**不是辛的**，"
            "所以长时间（远超本仓案例的2秒量级）能量与角动量会单调漂移而不是有界振荡；"
            "姿态靠每步归一化拉回单位球，不走李群指数映射"
        ),
        formal_order=4,
        measured_order=(
            "4（本仓B档实测：无力矩非对称刚体2秒，角动量漂移随步长减半降"
            "15.99/16.00倍、动能漂移降15.99/16.00倍，隐含阶4.00/4.00）"
        ),
        stability="explicit_conditional",
        step_bound=(
            "h < 2.785/|ω|_max（RK4实轴稳定区半径2.785除以最快转动模态的角频率）；"
            "自由刚体的最快模态是`|ω|`本身与进动率`|ω3|·(I3−I1)/I1`中的大者"
        ),
        dissipation_accounting=(
            "无算法耗散项，也**不宣称守恒**：RK4非辛，能量与角动量的漂移是"
            "O(h⁴)的截断误差而不是被建模的耗散，案例把它当**收敛量**测阶，不当零"
        ),
        failure_ladder=(
            "无步长拒绝/二分（失败关闭v1）。只有一条单向护栏："
            "归一化前`||q| − 1| > QUATERNION_NORM_STEP_ABS_TOL`时抛"
            "`RigidBodyError`并报出是第几步——不重试、不缩步长"
        ),
        production_ready=False,
    ),
    step=_rk4_step,
)

EXPLICIT_EULER_BODY = RigidBodyIntegrator(
    declaration=IntegratorDeclaration(
        name="explicit_euler_rigid_body",
        scope_excludes=(
            "不管任何生产用途；它只是漂移排序判据的对照组。"
            "一阶、反耗散，转动能量与角动量都单调增长"
        ),
        formal_order=1,
        measured_order="1（本仓B档实测：漂移随步长减半降约2倍）",
        stability="explicit_conditional",
        step_bound="h < 2/|ω|_max（显式Euler的实轴稳定区退化，实际长时间必发散）",
        dissipation_accounting=(
            "无算法耗散；能量**单调增长**（反耗散），这正是它被排在排序判据最差一档的原因"
        ),
        failure_ladder="无步长拒绝阶梯；与RK4共用同一条四元数范数护栏",
        production_ready=False,
    ),
    step=_explicit_euler_step,
)

RIGID_BODY_INTEGRATORS: dict[str, RigidBodyIntegrator] = {
    integrator.declaration.name: integrator
    for integrator in (RK4_BODY, EXPLICIT_EULER_BODY)
}


@dataclass(frozen=True)
class RotationDiagnostics:
    """一次推进的诊断。三个量全是**确定性**的（决策0018：进门的是确定性量）。"""

    steps: int
    #: 归一化**之前**的单步最大`||q| − 1|`。门守在这一侧，见模块文档第二节。
    max_norm_deviation: float
    #: 归一化次数——等于步数，写出来是为了让"某一步跳过了归一化"这件事可被断言。
    renormalisations: int


def _derivative_factory(
    inertia: RigidBodyInertia,
    force_world_n: ForceCallback | None,
    torque_body_nmm: TorqueCallback | None,
) -> Callable[[Vector, float], Vector]:
    """把Euler方程与姿态运动学写成一份公式源。**只写这一遍。**"""

    inverse_mass = MM_PER_M / inertia.mass_kg

    def derivative(y: Vector, t: float) -> Vector:
        velocity = y[_VELOCITY_SLICE]
        omega = y[_OMEGA_SLICE]
        quaternion = y[_ATTITUDE_SLICE]
        force = force_world_n(y, t) if force_world_n is not None else (0.0, 0.0, 0.0)
        torque = torque_body_nmm(y, t) if torque_body_nmm is not None else (0.0, 0.0, 0.0)
        # 平动：`a = f/m`，N/kg = m/s²是**米制**，到mm制乘MM_PER_M（与energies同一个常量）。
        acceleration = tuple(component * inverse_mass for component in force)
        # 转动：`I·ω̇ = τ·MM_PER_M − ω × (I·ω)`。陀螺项少一个负号或者叉乘反一次,
        # 能量照样守恒——只有带符号的进动率与惯性系角动量能指出来。
        gyroscopic = cross(omega, inertia.apply(omega))
        angular_acceleration = inertia.solve(
            tuple(
                torque[axis] * MM_PER_M - gyroscopic[axis] for axis in range(3)
            )
        )
        # 姿态：`q̇ = ½·q ⊗ (ω_body, 0)`。ω在体系所以四元数在**左**。
        rate = quaternion_multiply(
            quaternion, (omega[0], omega[1], omega[2], 0.0)
        )
        computed = (
            velocity,
            acceleration,
            angular_acceleration,
            (0.5 * rate[0], 0.5 * rate[1], 0.5 * rate[2], 0.5 * rate[3]),
        )
        # **按布局次序拼**：`_ASSEMBLY`是从`RIGID_BODY_LAYOUT`导出的置换。
        return (
            computed[_ASSEMBLY[0]]
            + computed[_ASSEMBLY[1]]
            + computed[_ASSEMBLY[2]]
            + computed[_ASSEMBLY[3]]
        )

    return derivative


def integrate_free_flight(
    integrator: RigidBodyIntegrator,
    *,
    state: State,
    inertia: RigidBodyInertia,
    dt_s: float,
    steps: int,
    force_world_n: ForceCallback | None = None,
    torque_body_nmm: TorqueCallback | None = None,
    t0_s: float = 0.0,
    ops: VectorOps | None = None,
    observer: Callable[[int, float, State], None] | None = None,
) -> tuple[State, RotationDiagnostics]:
    """推进`steps`步，返回`(末态, 诊断)`。

    步长与步数是**显式的**，不做自适应——步长二分是抢救不是收敛证据
    （spec/12第4.3节）。`observer`在每步之后被调用一次
    （`observer(step_index, t, state)`），让案例能量全程守恒量而不必把轨迹全存下来；
    它**不影响**积分，去掉它逐位结果不变（测试守着这一条）。
    """

    _require_layout(state)
    if steps < 0:
        raise RigidBodyError("steps must be nonnegative")
    if not (dt_s > 0.0):
        raise RigidBodyError("dt_s must be positive — 零步长不是不动，是没有定义")
    backend = ops or default_ops()
    derivative = _derivative_factory(inertia, force_world_n, torque_body_nmm)
    y = state.vector
    t = t0_s
    max_deviation = 0.0
    renormalisations = 0
    for index in range(steps):
        y = integrator.step(y, t, dt_s, derivative, backend)
        quaternion = y[_ATTITUDE_SLICE]
        norm = math.sqrt(sum(value * value for value in quaternion))
        deviation = abs(norm - 1.0)
        if deviation > max_deviation:
            max_deviation = deviation
        if deviation > QUATERNION_NORM_STEP_ABS_TOL:
            raise RigidBodyError(
                f"step {index}: |q| drifted to {norm!r} before renormalisation "
                f"(deviation {deviation!r} > {QUATERNION_NORM_STEP_ABS_TOL!r}) — "
                "步长对这个转速已经没有意义了；本积分器没有二分阶梯，这里失败关闭"
            )
        y = (
            y[: _ATTITUDE_SLICE.start]
            + normalise_quaternion(quaternion)
            + y[_ATTITUDE_SLICE.stop :]
        )
        renormalisations += 1
        t += dt_s
        if observer is not None:
            observer(index, t, State(layout=RIGID_BODY_LAYOUT, vector=y))
    return (
        State(layout=RIGID_BODY_LAYOUT, vector=y),
        RotationDiagnostics(
            steps=steps,
            max_norm_deviation=max_deviation,
            renormalisations=renormalisations,
        ),
    )


__all__ = [
    "EXPLICIT_EULER_BODY",
    "INERTIA_REL_TOL",
    "QUATERNION_NORM_STEP_ABS_TOL",
    "RIGID_BODY_INTEGRATORS",
    "RIGID_BODY_LAYOUT",
    "RK4_BODY",
    "ForceCallback",
    "Quaternion",
    "RigidBodyError",
    "RigidBodyIntegrator",
    "RigidBodyInertia",
    "RotationDiagnostics",
    "TorqueCallback",
    "Vector3",
    "angular_momentum_body_kg_mm2_per_s",
    "angular_momentum_world_kg_mm2_per_s",
    "angular_velocity_body_rad_per_s",
    "attitude_matrix",
    "attitude_xyzw",
    "cross",
    "integrate_free_flight",
    "make_state",
    "normalise_quaternion",
    "quaternion_multiply",
    "rotate_body_to_world",
    "rotate_world_to_body",
    "rotational_kinetic_energy_nmm",
]
