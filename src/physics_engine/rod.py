"""整杆各向异性弯曲与扭转（决策0065）——第一条有材料帧的全局杆。

本模块补的是plans/14第3.1节量出来的一号缺口。现场那条槽是**非平面3D曲线**：
整匝帧扭转``∫|τ|ds``跨236°—657°，而带材的``EI_hard/GJ ≈ 1040``——
**带材一定优先扭而不是硬弯，扭多少是要算的题**。本仓此前
`DiscreteElasticBending`是各向同性（每顶点一个标量EI），
`section_beam.py`只有一个三节点站点且只做easy-axis，**全仓一条平行输运都没有**。

装进来的四件：

1. **平行输运与材料帧**——切向→沿链Rodrigues输运``d1``→重新正交归一→
   ``d2 = t × d1``→用带符号角初始化边扭角``γ``；
2. **各向异性弯曲**——逐顶点两个曲率分量、逐顶点两个刚度；
3. **扭转**——``U = Σ 0.5·GJ·(γ_i − γ_{i−1} + m_ref_i)²/l̄_i``；
4. **retransport外层循环**——理由见第五节，**这一条最容易被漏掉**。

## 一、坐标约定：``d1``是穿厚方向，``d2``是带宽方向

这一段是整个模块最容易接反的地方，所以写在最前面，并且**有一道运行时门**。

    m1 = cosγ·d1 + sinγ·d2          m2 = −sinγ·d1 + cosγ·d2
    κb = 2·(e0 × e1)/(|e0||e1| + e0·e1)
    κ1 = 0.5·(m2_left + m2_right)·κb        ← 配 EI_easy
    κ2 = −0.5·(m1_left + m1_right)·κb       ← 配 EI_hard

推一遍就知道为什么是这个配对：``κb``沿副法线，``κ1 = κb·m2``在杆**朝``m1``弯**
时才非零（右手系``t × m1 = m2``下，朝``m1``弯给``κb ∥ m2``）。所以
``m1``是哪个方向，``κ1``就是往那个方向弯的曲率。取``m1``＝**穿过带厚**的方向，
则``κ1``是easy-way弯曲，中性轴沿带宽，``I = w·h³/12``——这就是``EI_easy``。
``κ2``对应朝``m2``（带宽方向）弯，``I = h·w³/12``——``EI_hard``。
与plans/14第2.3节那两个数一致：``EI₁(hard) = E·t·w³/12``。

**同行那套（WDS）这个配对没有任何运行时校验，接反了只会给出一个1600倍偏小的
挠度而不报错**（4mm宽/0.1mm厚的带材，``(w/h)² = 1600``）。本模块的门是
`AnisotropicRodBending.__post_init__`里那句``ei_easy > ei_hard``失败关闭：
easy按定义就是软的那一轴，**反过来即是把配对写反了**。
另一半由测试守：`tests/test_rod.py`的易/难轴互换门把参考``d1``转90°，
断言挠度比**等于**``EI_hard/EI_easy``。

## 二、离散曲率不是1/mm——混用是静默的``l̄``倍错误

``κ1``/``κ2``与WDS的``natural_kappa1``同为**离散曲率**：无量纲、
**已经在对偶元上积分过**。物理曲率是``κ_discrete / l̄``，单位1/mm。
自然曲率``natural_kappa1``/``natural_kappa2``也是离散量，同一口径。
把物理曲率直接填进自然曲率槽，得到的是一个静默的``l̄``倍错误——
不报错、量纲上也看不出来。这条约定与`section_beam.KirchhoffSectionReference`
逐字相同，那里也是这么写的。

``l̄_i = 0.5·(rest[i−1] + rest[i])``取**参考构型**的边长，**不随位形变**。

## 三、逐顶点刚度，不退回全局标量

本仓`DiscreteElasticBending`是**逐顶点**EI（``vertices``元组里每项自带刚度），
比WDS的全局标量更一般。本模块保持逐顶点：``ei_easy_nmm2``与``ei_hard_nmm2``
都是长度等于内顶点数的元组。变截面、局部退化、分段材料都不需要再改形制。

## 四、梯度/Hessian走jet，自由度次序位置在前γ在末尾——这两条是裁决不是选择

决策0064第4.1、4.2节把它们先裁掉了，理由都是"量之前不许改形制"：

* **4.1**：一上来写十一变量两曲率分量的二阶解析链式法则等于在零实测下押注
  常数因子。先用`physics_engine.autodiff`的`Jet2`拿到正确答案与一条真实墙钟。
  **那两个墙钟数在决策0065第三节，是量出来的。**
* **4.2**：γ块整个放向量末尾（WDS的形制），于是一个内顶点同时耦合
  ``x[i−1..i+1]``与``γ[i−1..i]``，半带宽``O(3N)``；本仓`solve.py`的带状判据是
  结构性的（``2b+1 < m/3``才走带状）。**先按这个布局写，同批实测带状还触不触发**——
  交错布局是一次不可逆的形制选择，而"带状会不会失效"可以先量。
  **实测的半带宽、是否触发、两条路的墙钟都在决策0065第四节。**

## 五、retransport外层循环：不抄它，杆弯不出扭

``m_ref``被冻结成常数，于是`RodTwist`**对位置没有任何依赖**——
`RodTwist.gradient`在位置块上恒为零。这意味着**单次`solve_equilibrium`调用里
弯曲与扭转的交换根本不存在**，只能靠载荷步之间重新输运参考帧找回来。

`RodModel.retransport`做的四步：

1. 用**冻结帧**与当前``γ``算出材料director ``m1``（这是杆真正的物理朝向）；
2. 把``m1``沿各自边的切向变化**时间输运**到新切向上（材料朝向不变）；
3. 在**新构型**上重建Bishop帧（沿链空间平行输运），种子取时间输运后的``d1[0]``；
4. 用带符号角把``γ``从保住的``m1``里反算回来，并沿链解缠。

第3步是关键：新Bishop帧带着**新构型的holonomy**，而``γ``必须把它吃下去——
弯曲于是转成了扭转。`solve_rod_with_retransport`把"求解→重输运→重建context→
再求解"写在`solve_equilibrium`**之上**，不需要第二个求解入口。

守它的门是`tests/test_rod.py`那个球面三角算例：切向序列``x̂ → ŷ → ẑ → x̂``
围出的球面三角形三个角都是直角，面积``4π/8 = π/2``，
**由Gauss-Bonnet，平行输运绕它一周的holonomy恰是π/2**。这是一条与被验内核
完全独立的闭式。不重输运时该算例的扭转能量**恒等于零**。

## 六、边界条件不需要任何新机制

`solve_equilibrium(..., fixed_indices=...)`就够：悬臂扭转BC＝钉住``3N+0``
（第一条边的γ）；两端夹持再钉一个；"规定端扭角"＝钉住那个γ并在初值里给数。
WDS也没有约束机制，它就是这么做的。**注意**：retransport会重算``γ``，
所以被钉住的是**材料帧**而不是那个坐标值——这正是想要的语义。

## 七、布局id不绑参考帧（与`section_beam`的形制差别，写清楚）

`section_beam`把截面与顶点参考的指纹绑进``layout_id``。本模块**不绑**：
retransport每一轮都换一个参考帧，绑进去就意味着每轮产出一个不兼容的`State`，
外层循环第一步就会撞`StateError`。布局只是打包契约（位置块＋γ块），
参考帧是另一个有自己`fingerprint()`的冻结对象。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.autodiff import Jet1, Jet2, ad_cos, ad_cross, ad_dot, ad_norm, ad_sin
from physics_engine.canonical import WDS_PROFILE, canonical_sha256
from physics_engine.energies import (
    POTENTIAL,
    EnergyContext,
    EnergyRegistry,
    EnergyTerm,
    Matrix,
    Vector,
)
from physics_engine.solve import SolveResult, solve_equilibrium
from physics_engine.state import State, StateField, StateLayout

Vec3 = tuple[float, float, float]

TAU = 2.0 * math.pi

#: 参考帧正交归一的绝对容差。与`section_beam.KirchhoffSectionReference`同值，
#: 理由也相同：这是"调用方给的帧到底是不是一个帧"的判据，不是数值路径的容差。
FRAME_TOLERANCE = 1.0e-10


class RodError(ValueError):
    """整杆层的一切失败关闭。"""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RodError(f"{name} must be finite: {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise RodError(f"{name} must be finite: {value!r}")
    return converted


def _positive_finite(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise RodError(f"{name} must be positive: {value!r}")
    return converted


def _vector3(name: str, value) -> Vec3:
    try:
        raw = tuple(value)
    except TypeError as error:
        raise RodError(f"{name} must be a finite 3-vector") from error
    if len(raw) != 3:
        raise RodError(f"{name} must be a finite 3-vector")
    return tuple(_finite(f"{name}[{index}]", component) for index, component in enumerate(raw))


def _dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Vec3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalized(name: str, vector: Vec3) -> Vec3:
    length = _norm(vector)
    if length <= 0.0:
        raise RodError(f"{name} has zero length — 方向未定义")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _orthonormalized(name: str, vector: Vec3, axis: Vec3) -> Vec3:
    """把``vector``投到``axis``的正交补上再归一——输运后重新正交归一那一步。

    平行输运在精确算术下保正交，浮点下会漂；不收这一步，漂移会沿链累积，
    而``κ1``/``κ2``的分解正建立在``(t, d1, d2)``是正交归一基上。
    """

    projection = _dot(vector, axis)
    residual = (
        vector[0] - projection * axis[0],
        vector[1] - projection * axis[1],
        vector[2] - projection * axis[2],
    )
    if _norm(residual) <= FRAME_TOLERANCE:
        raise RodError(f"{name} is parallel to the tangent — 无法从它定出材料帧")
    return _normalized(name, residual)


def parallel_transport(vector: Vec3, source_tangent: Vec3, target_tangent: Vec3) -> Vec3:
    """把``vector``沿最短测地线从``source_tangent``输运到``target_tangent``。

    走的是Bergou等2008《Discrete Elastic Rods》那条无三角函数的形式：
    转轴``b = t₀ × t₁``归一后，``(t₀, t₀×b, b)``与``(t₁, t₁×b, b)``是同一次转动
    下的两组基，逐分量换基即得。比先算角度再Rodrigues少两次超越函数，
    且在``t₀ ≈ t₁``时无相消。

    两个退化都**失败关闭**、不返回近似：切向反平行（转动轴不唯一），
    以及任一切向长度为零。
    """

    axis = _cross(source_tangent, target_tangent)
    axis_norm = _norm(axis)
    aligned = _dot(source_tangent, target_tangent)
    if axis_norm <= FRAME_TOLERANCE:
        if aligned < 0.0:
            raise RodError(
                "parallel transport between antiparallel tangents is undefined — "
                "转轴不唯一，这是模型自身的奇点，不是可以取一个近似的地方"
            )
        return vector
    unit_axis = (axis[0] / axis_norm, axis[1] / axis_norm, axis[2] / axis_norm)
    source_normal = _cross(source_tangent, unit_axis)
    target_normal = _cross(target_tangent, unit_axis)
    along = _dot(vector, source_tangent)
    across = _dot(vector, source_normal)
    out_of_plane = _dot(vector, unit_axis)
    return (
        along * target_tangent[0] + across * target_normal[0] + out_of_plane * unit_axis[0],
        along * target_tangent[1] + across * target_normal[1] + out_of_plane * unit_axis[1],
        along * target_tangent[2] + across * target_normal[2] + out_of_plane * unit_axis[2],
    )


def signed_angle(source: Vec3, target: Vec3, axis: Vec3) -> float:
    """绕``axis``从``source``转到``target``的带符号角，取值在``(−π, π]``。

    ``axis``要是单位向量且与两者都近似正交；本函数**不替调用方检查这一条**，
    因为它在热路径上，而所有生产调用点的正交性都由帧构造器保证过。
    """

    return math.atan2(_dot(axis, _cross(source, target)), _dot(source, target))


def unwrap_phases(values, *, period: float = TAU) -> tuple[float, ...]:
    """相位解缠的零依赖标量版——本仓没有``np.unwrap``的替代，所以自己写一个。

    逐项加上``period``的整数倍，使**相邻差落进``[−period/2, period/2)``**。
    第一项原样保留（解缠只定到一个整体常数，锚点由调用方定）。

    ### 为什么它是必须的，而不是"顺手加的稳健性"

    带符号角只能给出``(−π, π]``里的一个代表元。整杆扭转跨越半圈时，
    ``γ``序列会在某处跳``2π``——而扭转能量算的是``(γ_i − γ_{i−1})²``，
    一个假的``2π``跳变会凭空造出``0.5·GJ·(2π)²/l̄``的能量。
    plans/14量出真实工件整匝``∫|τ|ds``跨236°—**657°**，**657°接近两圈**，
    所以这不是理论上的边角，是这批工件上一定会发生的事。

    门：`tests/test_rod.py::test_unwrap_removes_the_two_pi_jumps_it_is_there_for`
    与它配套的必红用例。
    """

    raw = tuple(_finite(f"unwrap_phases[{index}]", value) for index, value in enumerate(values))
    if not raw:
        raise RodError("unwrap_phases needs at least one value")
    half = 0.5 * _positive_finite("period", period)
    result = [raw[0]]
    offset = 0.0
    for index in range(1, len(raw)):
        delta = raw[index] + offset - result[index - 1]
        offset -= period * math.floor((delta + half) / period)
        result.append(raw[index] + offset)
    return tuple(result)


@dataclass(frozen=True)
class RodMaterialFrame:
    """整杆的冻结参考帧：逐边``(t, d1, d2)``，逐内顶点一个参考扭转``m_ref``。

    ``reference_twist[i]``是"把``d1[i−1]``平行输运到边``i``的切向后，
    它与``d1[i]``之间绕``t[i]``的带符号角"。**Bishop帧（沿链平行输运出来的）
    这一项恒为零**，但本类**不假定**它为零——它是量出来的，
    于是调用方可以给一个非Bishop的帧（例如从槽的装配帧直接读的帧），
    而扭转能量里那个``m_ref``就是真的有内容的通道。
    """

    tangents: tuple[Vec3, ...]
    d1: tuple[Vec3, ...]
    d2: tuple[Vec3, ...]
    reference_twist: tuple[float, ...]

    def __post_init__(self) -> None:
        tangents = tuple(
            _vector3(f"tangents[{index}]", value) for index, value in enumerate(self.tangents)
        )
        d1 = tuple(_vector3(f"d1[{index}]", value) for index, value in enumerate(self.d1))
        d2 = tuple(_vector3(f"d2[{index}]", value) for index, value in enumerate(self.d2))
        if not tangents or len(d1) != len(tangents) or len(d2) != len(tangents):
            raise RodError("a material frame needs one tangent, d1 and d2 per edge")
        twist = tuple(
            _finite(f"reference_twist[{index}]", value)
            for index, value in enumerate(self.reference_twist)
        )
        if len(twist) != max(len(tangents) - 1, 0):
            raise RodError("reference_twist needs one entry per interior vertex")
        for edge in range(len(tangents)):
            expected_d2 = _cross(tangents[edge], d1[edge])
            deviation = max(
                abs(_norm(tangents[edge]) - 1.0),
                abs(_norm(d1[edge]) - 1.0),
                abs(_dot(tangents[edge], d1[edge])),
                max(abs(a - b) for a, b in zip(d2[edge], expected_d2, strict=True)),
            )
            if deviation > FRAME_TOLERANCE:
                raise RodError(
                    f"edge {edge} directors are not an orthonormal right-handed frame "
                    f"(|t|=1, |d1|=1, d1⊥t, d2 = t × d1); worst deviation {deviation:.3e}"
                )
        object.__setattr__(self, "tangents", tangents)
        object.__setattr__(self, "d1", d1)
        object.__setattr__(self, "d2", d2)
        object.__setattr__(self, "reference_twist", twist)

    @property
    def edge_count(self) -> int:
        return len(self.tangents)

    def material_directors(self, edge_twist_angles) -> tuple[tuple[Vec3, Vec3], ...]:
        """逐边的``(m1, m2)``——把``γ``作用到冻结帧上得到的**材料**朝向。"""

        gammas = tuple(
            _finite(f"edge_twist_angles[{index}]", value)
            for index, value in enumerate(edge_twist_angles)
        )
        if len(gammas) != self.edge_count:
            raise RodError("edge_twist_angles needs one angle per edge")
        directors = []
        for edge, gamma in enumerate(gammas):
            cosine, sine = math.cos(gamma), math.sin(gamma)
            first = self.d1[edge]
            second = self.d2[edge]
            m1 = tuple(cosine * first[a] + sine * second[a] for a in range(3))
            m2 = tuple(-sine * first[a] + cosine * second[a] for a in range(3))
            directors.append((m1, m2))
        return tuple(directors)

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "tangents": [list(vector) for vector in self.tangents],
                "d1": [list(vector) for vector in self.d1],
                "d2": [list(vector) for vector in self.d2],
                "reference_twist": list(self.reference_twist),
                "kinematics_id": "engine/rod-anisotropic-bending-twist/1",
            },
            WDS_PROFILE,
        )


def edge_tangents(positions_mm) -> tuple[Vec3, ...]:
    """节点位置 → 逐边单位切向。零长边**失败关闭**。"""

    nodes = tuple(
        _vector3(f"positions_mm[{index}]", value) for index, value in enumerate(positions_mm)
    )
    if len(nodes) < 2:
        raise RodError("a rod needs at least two nodes")
    return tuple(
        _normalized(
            f"edge {index}",
            (
                nodes[index + 1][0] - nodes[index][0],
                nodes[index + 1][1] - nodes[index][1],
                nodes[index + 1][2] - nodes[index][2],
            ),
        )
        for index in range(len(nodes) - 1)
    )


def build_material_frame(*, tangents, edge_d1) -> RodMaterialFrame:
    """从逐边切向与逐边``d1``装出一个帧，并**量出**``m_ref``。

    ``d1``先对切向正交归一（调用方给的方向只需要不与切向平行），
    ``d2 = t × d1``。``m_ref``按定义算，不假定为零——这就是非Bishop参考帧
    进得来的那条通道。
    """

    tangent_values = tuple(
        _vector3(f"tangents[{index}]", value) for index, value in enumerate(tangents)
    )
    raw_d1 = tuple(_vector3(f"edge_d1[{index}]", value) for index, value in enumerate(edge_d1))
    if len(raw_d1) != len(tangent_values):
        raise RodError("build_material_frame needs one d1 per tangent")
    normalized_d1 = tuple(
        _orthonormalized(f"edge_d1[{index}]", raw_d1[index], tangent_values[index])
        for index in range(len(tangent_values))
    )
    d2 = tuple(
        _cross(tangent_values[index], normalized_d1[index]) for index in range(len(tangent_values))
    )
    reference_twist = tuple(
        signed_angle(
            parallel_transport(
                normalized_d1[index - 1], tangent_values[index - 1], tangent_values[index]
            ),
            normalized_d1[index],
            tangent_values[index],
        )
        for index in range(1, len(tangent_values))
    )
    return RodMaterialFrame(tangent_values, normalized_d1, d2, reference_twist)


def build_bishop_frame(*, positions_mm, seed_d1) -> RodMaterialFrame:
    """沿链平行输运出一个Bishop帧。``m_ref``由构造保证为零，但仍是量出来的。

    这是本模块唯一的帧生成器（`build_material_frame`是它的一般化入口）。
    整个retransport外层循环的第3步就是在**新构型**上重跑它一次。
    """

    tangents = edge_tangents(positions_mm)
    directors = [_orthonormalized("seed_d1", _vector3("seed_d1", seed_d1), tangents[0])]
    for edge in range(1, len(tangents)):
        transported = parallel_transport(directors[-1], tangents[edge - 1], tangents[edge])
        directors.append(_orthonormalized(f"transported d1[{edge}]", transported, tangents[edge]))
    return build_material_frame(tangents=tangents, edge_d1=tuple(directors))


def gammas_from_material_directors(*, frame: RodMaterialFrame, m1) -> tuple[float, ...]:
    """给定想要的材料director ``m1``，反算边扭角``γ``并沿链解缠。

    这是"用带符号角初始化边扭角"那一步的公开面：调用方拿着从槽帧/Frenet帧
    读出来的材料朝向进来，拿走一串``γ``。
    """

    wanted = tuple(_vector3(f"m1[{index}]", value) for index, value in enumerate(m1))
    if len(wanted) != frame.edge_count:
        raise RodError("gammas_from_material_directors needs one m1 per edge")
    raw = [
        signed_angle(
            frame.d1[edge],
            _orthonormalized(f"m1[{edge}]", wanted[edge], frame.tangents[edge]),
            frame.tangents[edge],
        )
        for edge in range(frame.edge_count)
    ]
    return unwrap_phases(raw)


@dataclass(frozen=True)
class RodReference:
    """整杆的冻结参考量：静长、材料帧、逐内顶点的两个自然曲率。

    ``natural_kappa1``与``natural_kappa2``都是**离散曲率**（无量纲，
    已在对偶元上积分过）——见模块docstring第二节。
    **``natural_kappa2``是显式开着的通道**：WDS有这个槽但生产里恒为零，
    也就是说**它从没跑过**；本模块有一条门专门跑它
    （`test_natural_kappa2_is_a_live_channel_not_a_dead_slot`）。
    """

    rest_lengths_mm: tuple[float, ...]
    frame: RodMaterialFrame
    natural_kappa1: tuple[float, ...] = ()
    natural_kappa2: tuple[float, ...] = ()
    #: 对偶长度的**显式覆盖**。``None``时按``0.5·(rest[i−1]+rest[i])``推。
    #:
    #: 留这个口子只有一个理由，而那个理由本仓已经推导并实测过一次：
    #: **被钉死的第一条边会吞掉半格柔度**，正确的固支等效对偶长度是``3h/2``而不是
    #: ``h``（`energies.clamped_chain_bending_vertices`的docstring里有静态凝聚推导
    #: 与实测收敛比）。不给这个口子，整杆的固支悬臂就只能一阶收敛——
    #: 实测本模块不加订正时挠度偏差按``h``走（0.145/0.0738/0.0372/0.0187，比值≈2），
    #: 加订正后是干净的二阶。**这不是边界条件的阶次问题**，0027那条教训警告过归因错
    #: 会去改边界条件而那改不动它。
    dual_lengths_mm: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        rest = tuple(
            _positive_finite(f"rest_lengths_mm[{index}]", value)
            for index, value in enumerate(self.rest_lengths_mm)
        )
        if len(rest) != self.frame.edge_count:
            raise RodError("rest_lengths_mm needs one entry per edge")
        vertex_count = max(len(rest) - 1, 0)
        kappa1 = tuple(self.natural_kappa1) or (0.0,) * vertex_count
        kappa2 = tuple(self.natural_kappa2) or (0.0,) * vertex_count
        if len(kappa1) != vertex_count or len(kappa2) != vertex_count:
            raise RodError("natural curvatures need one entry per interior vertex")
        object.__setattr__(self, "rest_lengths_mm", rest)
        object.__setattr__(
            self,
            "natural_kappa1",
            tuple(_finite(f"natural_kappa1[{i}]", v) for i, v in enumerate(kappa1)),
        )
        object.__setattr__(
            self,
            "natural_kappa2",
            tuple(_finite(f"natural_kappa2[{i}]", v) for i, v in enumerate(kappa2)),
        )
        if self.dual_lengths_mm is None:
            dual = tuple(
                0.5 * (rest[index - 1] + rest[index]) for index in range(1, len(rest))
            )
        else:
            dual = tuple(
                _positive_finite(f"dual_lengths_mm[{i}]", v)
                for i, v in enumerate(self.dual_lengths_mm)
            )
            if len(dual) != vertex_count:
                raise RodError("dual_lengths_mm needs one entry per interior vertex")
        object.__setattr__(self, "dual_lengths_mm", dual)

    @property
    def interior_vertex_count(self) -> int:
        return max(len(self.rest_lengths_mm) - 1, 0)

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "rest_lengths_mm": list(self.rest_lengths_mm),
                "frame": self.frame.fingerprint(),
                "natural_kappa1": list(self.natural_kappa1),
                "natural_kappa2": list(self.natural_kappa2),
                "dual_lengths_mm": list(self.dual_lengths_mm),
                "curvature_units": "discrete/dimensionless",
            },
            WDS_PROFILE,
        )


_POSITIONS_FIELD = "node_positions_mm"
_TWIST_FIELD = "edge_twist_angles"


@dataclass(frozen=True)
class RodLayout:
    """打包契约：``3N``个位置在前，``N−1``个边扭角在末尾（决策0064第4.2节）。

    次序是**裁决**不是实现细节：它决定半带宽，而半带宽决定`solve.py`走不走
    带状快路。0064裁定先按这个布局写并同批实测，**不预先改成交错**。
    """

    node_count: int
    layout: StateLayout

    def __post_init__(self) -> None:
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int):
            raise RodError(f"node_count must be an int: {self.node_count!r}")
        if self.node_count < 3:
            raise RodError("a rod needs at least three nodes (one interior vertex)")
        expected = (
            (_POSITIONS_FIELD, 3 * self.node_count),
            (_TWIST_FIELD, self.node_count - 1),
        )
        measured = tuple((field.name, field.width) for field in self.layout.fields)
        if measured != expected or self.layout.node_dof_count != 3 * self.node_count:
            raise RodError("layout does not match the rod packing contract")

    @property
    def edge_count(self) -> int:
        return self.node_count - 1

    @property
    def interior_vertex_count(self) -> int:
        return self.node_count - 2

    @property
    def twist_offset(self) -> int:
        return 3 * self.node_count

    def position_index(self, node: int, axis: int) -> int:
        if not 0 <= node < self.node_count or not 0 <= axis < 3:
            raise RodError(f"position index out of range: node {node}, axis {axis}")
        return 3 * node + axis

    def twist_index(self, edge: int) -> int:
        if not 0 <= edge < self.edge_count:
            raise RodError(f"twist index out of range: edge {edge}")
        return self.twist_offset + edge

    def assert_state(self, state: State) -> None:
        if state.layout != self.layout:
            raise RodError(
                f"state layout {state.layout.layout_id!r} does not match "
                f"rod layout {self.layout.layout_id!r}"
            )

    def initial_state(self, *, positions_mm, edge_twist_angles) -> State:
        nodes = tuple(
            _vector3(f"positions_mm[{index}]", value)
            for index, value in enumerate(positions_mm)
        )
        gammas = tuple(
            _finite(f"edge_twist_angles[{index}]", value)
            for index, value in enumerate(edge_twist_angles)
        )
        if len(nodes) != self.node_count or len(gammas) != self.edge_count:
            raise RodError(
                f"initial state needs {self.node_count} positions and "
                f"{self.edge_count} edge twist angles"
            )
        flat = tuple(component for node in nodes for component in node)
        return State(layout=self.layout, vector=flat + gammas)

    def positions(self, state: State) -> tuple[Vec3, ...]:
        self.assert_state(state)
        x = state.vector
        return tuple(
            (x[3 * node], x[3 * node + 1], x[3 * node + 2]) for node in range(self.node_count)
        )

    def twist_angles(self, state: State) -> tuple[float, ...]:
        self.assert_state(state)
        start = self.twist_offset
        return tuple(state.vector[start : start + self.edge_count])

    def twist_indices(self) -> frozenset[int]:
        return frozenset(range(self.twist_offset, self.twist_offset + self.edge_count))

    def position_indices(self) -> frozenset[int]:
        return frozenset(range(self.twist_offset))


def build_rod_layout(*, layout_id: str, node_count: int) -> RodLayout:
    """建立整杆布局：位置块在前、γ块在末尾。**不把参考帧绑进id**——理由见第七节。"""

    if not isinstance(layout_id, str) or not layout_id.startswith("layout/"):
        raise RodError("layout_id must be namespaced like 'layout/...'")
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 3:
        raise RodError("a rod needs at least three nodes (one interior vertex)")
    layout = StateLayout(
        layout_id=f"{layout_id}/rod-{node_count}",
        fields=(
            StateField(_POSITIONS_FIELD, 3 * node_count),
            StateField(_TWIST_FIELD, node_count - 1, is_dimensionless=True),
        ),
        node_dof_count=3 * node_count,
    )
    return RodLayout(node_count, layout)


def _local_jets(state: State, indices: tuple[int, ...], order: int):
    if order == 0:
        return tuple(state.vector[index] for index in indices)
    if order == 1:
        return tuple(
            Jet1.variable(state.vector[index], slot, len(indices))
            for slot, index in enumerate(indices)
        )
    if order == 2:
        return tuple(
            Jet2.variable(state.vector[index], slot, len(indices))
            for slot, index in enumerate(indices)
        )
    raise RodError(f"derivative order must be 0, 1 or 2: {order!r}")


@dataclass(frozen=True)
class _RodEnergyTerm:
    """两个杆能量项共用的装配面：逐顶点局部jet → 全局能量/梯度/Hessian。

    子类只提供三件：顶点数、局部→全局的下标映射、以及**一个顶点的能量表达式**。
    抽出来的理由是本仓自己的条款——`EnergyTerm`协议那四个方法在两个项里逐字相同，
    而spec/12第3.1节要求**融合路径与单独调`energy`逐字节相同**：
    两份复制粘贴的装配循环是那条承诺最容易破掉的地方。
    """

    def _vertex_count(self) -> int:
        raise NotImplementedError

    def _local_indices(self, vertex: int) -> tuple[int, ...]:
        raise NotImplementedError

    def _vertex_energy(self, state: State, vertex: int, *, order: int):
        raise NotImplementedError

    def energy(self, state: State, context: EnergyContext) -> float:
        self.layout.assert_state(state)
        total = 0.0
        for vertex in range(self._vertex_count()):
            total += self._vertex_energy(state, vertex, order=0)
        return total

    def quantities(
        self, state: State, context: EnergyContext, *,
        need_gradient: bool, need_hessian: bool,
    ) -> tuple[float, Vector | None, Matrix | None]:
        self.layout.assert_state(state)
        order = 2 if need_hessian else (1 if need_gradient else 0)
        size = len(state.vector)
        total = 0.0
        gradient = [0.0] * size if need_gradient else None
        hessian = [[0.0] * size for _ in range(size)] if need_hessian else None
        for vertex in range(self._vertex_count()):
            local = self._vertex_energy(state, vertex, order=order)
            indices = self._local_indices(vertex)
            total += local.value if isinstance(local, (Jet1, Jet2)) else local
            if gradient is not None:
                for slot, index in enumerate(indices):
                    gradient[index] += local.gradient[slot]
            if hessian is not None:
                for row, row_index in enumerate(indices):
                    for column, column_index in enumerate(indices):
                        hessian[row_index][column_index] += local.hessian[row][column]
        return (
            total,
            tuple(gradient) if gradient is not None else None,
            tuple(tuple(row) for row in hessian) if hessian is not None else None,
        )

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        gradient = self.quantities(state, context, need_gradient=True, need_hessian=False)[1]
        assert gradient is not None
        return gradient

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        hessian = self.quantities(state, context, need_gradient=False, need_hessian=True)[2]
        assert hessian is not None
        return hessian

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        self.layout.assert_state(state)
        entries: list[tuple[int, int, float]] = []
        for vertex in range(self._vertex_count()):
            local = self._vertex_energy(state, vertex, order=2)
            indices = self._local_indices(vertex)
            for row, row_index in enumerate(indices):
                for column, column_index in enumerate(indices):
                    entries.append((row_index, column_index, local.hessian[row][column]))
        return tuple(entries)


@dataclass(frozen=True)
class AnisotropicRodBending(_RodEnergyTerm):
    """整杆各向异性弯曲：``U = Σ_i (1/(2·l̄_i))·[EI_easy·Δκ1² + EI_hard·Δκ2²]``。

    ``EI_easy``配``κ1``、``EI_hard``配``κ2``——**配对的理由与那道门在模块
    docstring第一节**。``κ1``/``κ2``是离散曲率（无量纲），物理曲率要除``l̄``。

    梯度与Hessian由`physics_engine.autodiff`的二阶jet作解析链式法则得到
    （决策0064第4.1节裁定，不写解析闭式）。局部变量次序是
    ``[x_{i−1}(3), x_i(3), x_{i+1}(3), γ_{i−1}, γ_i]``，与`section_beam`一致。
    """

    layout: RodLayout
    reference: RodReference
    #: 逐内顶点的easy-axis弯曲刚度（N·mm²）。**逐顶点不退回全局标量**——
    #: 本仓`DiscreteElasticBending`就是逐顶点的，比WDS的全局标量更一般。
    ei_easy_nmm2: tuple[float, ...] = ()
    ei_hard_nmm2: tuple[float, ...] = ()
    name: str = "anisotropic_rod_bending"
    kind: ClassVar[Literal["potential"]] = POTENTIAL

    def __post_init__(self) -> None:
        vertices = self.layout.interior_vertex_count
        if self.reference.interior_vertex_count != vertices:
            raise RodError("reference and layout disagree on the interior vertex count")
        if self.reference.frame.edge_count != self.layout.edge_count:
            raise RodError("reference frame and layout disagree on the edge count")
        easy = tuple(
            _positive_finite(f"ei_easy_nmm2[{i}]", v) for i, v in enumerate(self.ei_easy_nmm2)
        )
        hard = tuple(
            _positive_finite(f"ei_hard_nmm2[{i}]", v) for i, v in enumerate(self.ei_hard_nmm2)
        )
        if len(easy) != vertices or len(hard) != vertices:
            raise RodError("bending stiffness needs one easy and one hard value per vertex")
        for index, (soft, stiff) in enumerate(zip(easy, hard, strict=True)):
            if soft > stiff:
                raise RodError(
                    f"vertex {index}: ei_easy_nmm2={soft!r} exceeds ei_hard_nmm2={stiff!r} — "
                    "**easy按定义就是软的那一轴**，反过来说明κ1/κ2与两个刚度的配对写反了。"
                    "本仓的约定是 EI_easy 配 κ1（朝d1弯，穿过带厚）、"
                    "EI_hard 配 κ2（朝d2弯，在带宽面内）；接反在同行那套里"
                    "只会给出一个1600倍偏小的挠度而不报任何错"
                )
        object.__setattr__(self, "ei_easy_nmm2", easy)
        object.__setattr__(self, "ei_hard_nmm2", hard)

    def node_index_bound(self) -> int:
        return self.layout.node_count

    def _vertex_count(self) -> int:
        return self.layout.interior_vertex_count

    def _local_indices(self, vertex: int) -> tuple[int, ...]:
        left, middle, right = vertex, vertex + 1, vertex + 2
        return (
            3 * left, 3 * left + 1, 3 * left + 2,
            3 * middle, 3 * middle + 1, 3 * middle + 2,
            3 * right, 3 * right + 1, 3 * right + 2,
            self.layout.twist_index(vertex),
            self.layout.twist_index(vertex + 1),
        )

    def _curvatures(self, state: State, vertex: int, *, order: int):
        """一个内顶点的``(κ1, κ2)``离散曲率。**曲率核只在这里求值。**"""

        local = _local_jets(state, self._local_indices(vertex), order)
        p0, p1, p2 = local[0:3], local[3:6], local[6:9]
        edge0 = tuple(p1[axis] - p0[axis] for axis in range(3))
        edge1 = tuple(p2[axis] - p1[axis] for axis in range(3))
        denominator = ad_norm(edge0) * ad_norm(edge1) + ad_dot(edge0, edge1)
        raw_denominator = (
            denominator.value if isinstance(denominator, (Jet1, Jet2)) else denominator
        )
        if abs(raw_denominator) < 1.0e-12:
            raise RodError(
                f"rod vertex {vertex} is folded back (θ→π) — "
                "κb = 2·tan(θ/2)在此发散，这是模型自身的奇点，不是可以返回大数的地方"
            )
        binormal = tuple(
            2.0 * component / denominator for component in ad_cross(edge0, edge1)
        )
        frame = self.reference.frame
        cos_left, sin_left = ad_cos(local[9]), ad_sin(local[9])
        cos_right, sin_right = ad_cos(local[10]), ad_sin(local[10])
        d1_left, d2_left = frame.d1[vertex], frame.d2[vertex]
        d1_right, d2_right = frame.d1[vertex + 1], frame.d2[vertex + 1]
        m1_sum = tuple(
            cos_left * d1_left[a] + sin_left * d2_left[a]
            + cos_right * d1_right[a] + sin_right * d2_right[a]
            for a in range(3)
        )
        m2_sum = tuple(
            -sin_left * d1_left[a] + cos_left * d2_left[a]
            - sin_right * d1_right[a] + cos_right * d2_right[a]
            for a in range(3)
        )
        return 0.5 * ad_dot(m2_sum, binormal), -0.5 * ad_dot(m1_sum, binormal)

    def _vertex_energy(self, state: State, vertex: int, *, order: int):
        kappa1, kappa2 = self._curvatures(state, vertex, order=order)
        delta1 = kappa1 - self.reference.natural_kappa1[vertex]
        delta2 = kappa2 - self.reference.natural_kappa2[vertex]
        scale = 0.5 / self.reference.dual_lengths_mm[vertex]
        return scale * (
            self.ei_easy_nmm2[vertex] * delta1 * delta1
            + self.ei_hard_nmm2[vertex] * delta2 * delta2
        )

    def curvatures(self, state: State) -> tuple[tuple[float, float], ...]:
        """诊断面：逐内顶点的``(κ1, κ2)``**离散**曲率。除``l̄``才是1/mm。"""

        self.layout.assert_state(state)
        return tuple(
            self._curvatures(state, vertex, order=0)
            for vertex in range(self.layout.interior_vertex_count)
        )

@dataclass(frozen=True)
class RodTwist(_RodEnergyTerm):
    """整杆扭转：``U = Σ_i 0.5·GJ_i·(γ_i − γ_{i−1} + m_ref_i)²/l̄_i``。

    **它对位置没有任何依赖**——``m_ref``是冻结的常数，``γ``是自己的自由度。
    这不是简化，是DER形制本身：位置的影响经`RodModel.retransport`
    重算``m_ref``与``γ``进来。**不抄外层循环，这一项就永远与弯曲无关。**
    模块docstring第五节记着为什么，以及守它的那道门。
    """

    layout: RodLayout
    reference: RodReference
    #: 逐内顶点的扭转刚度``GJ``（N·mm²）。
    gj_nmm2: tuple[float, ...] = ()
    name: str = "rod_twist"
    kind: ClassVar[Literal["potential"]] = POTENTIAL

    def __post_init__(self) -> None:
        vertices = self.layout.interior_vertex_count
        if self.reference.interior_vertex_count != vertices:
            raise RodError("reference and layout disagree on the interior vertex count")
        values = tuple(
            _positive_finite(f"gj_nmm2[{i}]", v) for i, v in enumerate(self.gj_nmm2)
        )
        if len(values) != vertices:
            raise RodError("torsional stiffness needs one value per interior vertex")
        object.__setattr__(self, "gj_nmm2", values)

    def node_index_bound(self) -> int:
        return 0

    def _vertex_count(self) -> int:
        return self.layout.interior_vertex_count

    def _local_indices(self, vertex: int) -> tuple[int, ...]:
        return (self.layout.twist_index(vertex), self.layout.twist_index(vertex + 1))

    def _vertex_energy(self, state: State, vertex: int, *, order: int):
        local = _local_jets(state, self._local_indices(vertex), order)
        twist = local[1] - local[0] + self.reference.frame.reference_twist[vertex]
        scale = 0.5 * self.gj_nmm2[vertex] / self.reference.dual_lengths_mm[vertex]
        return scale * twist * twist

    def twist_rates_per_mm(self, state: State) -> tuple[float, ...]:
        """诊断面：逐内顶点的物理扭率``(γ_i − γ_{i−1} + m_ref_i)/l̄_i``（1/mm）。"""

        self.layout.assert_state(state)
        gammas = self.layout.twist_angles(state)
        dual = self.reference.dual_lengths_mm
        return tuple(
            (gammas[vertex + 1] - gammas[vertex] + self.reference.frame.reference_twist[vertex])
            / dual[vertex]
            for vertex in range(self.layout.interior_vertex_count)
        )

@dataclass(frozen=True)
class RodEndMoment:
    """端扭矩载荷：``U = −M·γ_e``。恒定外力矩，梯度是常量、Hessian恒为零。

    有了它，闭式扭转门``θ = M·L/GJ``里的``θ``才是**答案**而不是输入——
    钉住一个端扭角再去核对扭矩，验的是同一个方程的另一半，说服力弱一档。
    """

    layout: RodLayout
    edge: int
    moment_n_mm: float
    name: str = "rod_end_moment"
    kind: ClassVar[Literal["potential"]] = POTENTIAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "moment_n_mm", _finite("moment_n_mm", self.moment_n_mm))
        self.layout.twist_index(self.edge)

    def node_index_bound(self) -> int:
        return 0

    def energy(self, state: State, context: EnergyContext) -> float:
        self.layout.assert_state(state)
        return -self.moment_n_mm * state.vector[self.layout.twist_index(self.edge)]

    def quantities(
        self, state: State, context: EnergyContext, *,
        need_gradient: bool, need_hessian: bool,
    ) -> tuple[float, Vector | None, Matrix | None]:
        size = len(state.vector)
        value = self.energy(state, context)
        gradient = None
        hessian = None
        if need_gradient:
            raw = [0.0] * size
            raw[self.layout.twist_index(self.edge)] = -self.moment_n_mm
            gradient = tuple(raw)
        if need_hessian:
            hessian = tuple(tuple([0.0] * size) for _ in range(size))
        return value, gradient, hessian

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        gradient = self.quantities(state, context, need_gradient=True, need_hessian=False)[1]
        assert gradient is not None
        return gradient

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        hessian = self.quantities(state, context, need_gradient=False, need_hessian=True)[2]
        assert hessian is not None
        return hessian

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        index = self.layout.twist_index(self.edge)
        return ((index, index, 0.0),)


@dataclass(frozen=True)
class RodRetransport:
    """一次重输运的产物：新模型、新状态，以及两个可读的诊断量。"""

    model: RodModel
    state: State
    #: 重输运前后``γ``的最大变化（rad）。外层循环收敛就是看它趋零。
    max_gamma_change: float
    #: 新帧里``m_ref``的最大绝对值。Bishop帧应当是机器零——**量出来的，不是假定的**。
    max_reference_twist: float


@dataclass(frozen=True)
class RodModel:
    """布局＋参考帧＋三组刚度。**它是retransport外层循环里被换掉的那个对象。**"""

    layout: RodLayout
    reference: RodReference
    ei_easy_nmm2: tuple[float, ...]
    ei_hard_nmm2: tuple[float, ...]
    gj_nmm2: tuple[float, ...]

    def bending(self) -> AnisotropicRodBending:
        return AnisotropicRodBending(
            layout=self.layout,
            reference=self.reference,
            ei_easy_nmm2=self.ei_easy_nmm2,
            ei_hard_nmm2=self.ei_hard_nmm2,
        )

    def twist(self) -> RodTwist:
        return RodTwist(layout=self.layout, reference=self.reference, gj_nmm2=self.gj_nmm2)

    def terms(self) -> tuple[EnergyTerm, ...]:
        """**求和次序是形制**（spec/12第3.3节）：先弯曲后扭转，固定不变。"""

        return (self.bending(), self.twist())

    def material_directors(self, state: State) -> tuple[Vec3, ...]:
        """当前``γ``下逐边的``m1``——杆真正的物理朝向，retransport要保住的量。"""

        gammas = self.layout.twist_angles(state)
        return tuple(m1 for m1, _ in self.reference.frame.material_directors(gammas))

    def retransport(self, state: State) -> RodRetransport:
        """求解之后重输运参考帧：保住``m1``，重建Bishop帧，反算``γ``。

        四步见模块docstring第五节。**静长与自然曲率不变**——它们是材料属性，
        不随位形走；变的只有帧与``γ``。
        """

        self.layout.assert_state(state)
        old_frame = self.reference.frame
        old_gammas = self.layout.twist_angles(state)
        new_tangents = edge_tangents(self.layout.positions(state))
        old_m1 = self.material_directors(state)
        new_m1 = tuple(
            _orthonormalized(
                f"retransported m1[{edge}]",
                parallel_transport(old_m1[edge], old_frame.tangents[edge], new_tangents[edge]),
                new_tangents[edge],
            )
            for edge in range(self.layout.edge_count)
        )
        seed = parallel_transport(old_frame.d1[0], old_frame.tangents[0], new_tangents[0])
        frame = build_bishop_frame(
            positions_mm=self.layout.positions(state), seed_d1=seed
        )
        gammas = list(gammas_from_material_directors(frame=frame, m1=new_m1))
        #: 时间方向的解缠：整体平移``2π``的整数倍，让``γ[0]``贴住上一轮的``γ[0]``。
        #: 空间解缠只把相邻差压进一个周期，**整条序列的分支还得有个锚**，
        #: 否则连续几轮之间会出现整条跳一圈的假历史。
        shift = TAU * round((old_gammas[0] - gammas[0]) / TAU)
        gammas = [value + shift for value in gammas]
        reference = RodReference(
            rest_lengths_mm=self.reference.rest_lengths_mm,
            frame=frame,
            natural_kappa1=self.reference.natural_kappa1,
            natural_kappa2=self.reference.natural_kappa2,
            dual_lengths_mm=self.reference.dual_lengths_mm,
        )
        model = RodModel(
            layout=self.layout,
            reference=reference,
            ei_easy_nmm2=self.ei_easy_nmm2,
            ei_hard_nmm2=self.ei_hard_nmm2,
            gj_nmm2=self.gj_nmm2,
        )
        new_state = state.with_vector(state.vector[: self.layout.twist_offset] + tuple(gammas))
        return RodRetransport(
            model,
            new_state,
            max((abs(a - b) for a, b in zip(gammas, old_gammas, strict=True)), default=0.0),
            max((abs(value) for value in frame.reference_twist), default=0.0),
        )


@dataclass(frozen=True)
class RodSolveStage:
    """外层循环的一个载荷步：钉哪些自由度、加哪些项、先把哪些值写进去。

    ``prescribed``在**进入本阶段时**写一次，之后各轮不再写——因为retransport
    会重算``γ``，而被钉住的语义是**材料帧**不是那个坐标值（模块docstring第六节）。
    """

    fixed_indices: frozenset[int] = frozenset()
    additional_terms: tuple[EnergyTerm, ...] = ()
    prescribed: tuple[tuple[int, float], ...] = ()
    #: 本阶段跑几轮"求解→重输运"。**1轮意味着最后一次重输运之后没有再求解**，
    #: 交换只走了一半；真要收敛取≥2并看`RodSolveRound.max_gamma_change`趋零。
    retransport_rounds: int = 2


@dataclass(frozen=True)
class RodSolveRound:
    """一轮的回执。能量是**重输运之前**那个收敛构型上的值，即真正解出来的那个。"""

    stage_index: int
    round_index: int
    result: SolveResult
    bending_energy_n_mm: float
    twist_energy_n_mm: float
    max_gamma_change: float
    max_reference_twist: float


@dataclass(frozen=True)
class RodEquilibrium:
    """外层循环的产物。``model``与``state``是**重输运之后**配套的一对。"""

    model: RodModel
    state: State
    rounds: tuple[RodSolveRound, ...]

    @property
    def converged(self) -> bool:
        return bool(self.rounds) and all(round.result.converged for round in self.rounds)


def solve_rod_with_retransport(
    *,
    model: RodModel,
    context: EnergyContext,
    initial: State,
    stages: tuple[RodSolveStage, ...],
    residual_tol_n: float,
    max_iterations: int = 50,
    max_backtracks: int = 40,
) -> RodEquilibrium:
    """求解 → 重输运参考帧 → 重建context → 再求解。**写在`solve_equilibrium`之上。**

    本函数是决策0064第4.3节第1条要求的那个外层循环，也是模块docstring第五节
    说的"不抄它，杆弯不出扭"的那一件。它不引入第二个求解入口：每一轮就是一次
    普通的`solve_equilibrium`，轮与轮之间换掉参考帧与``γ``。

    ``residual_tol_n``是**绝对**残差，与`solve_equilibrium`同义、同样没有默认值。
    """

    if not stages:
        raise RodError("solve_rod_with_retransport needs at least one stage")
    model.layout.assert_state(initial)
    current_model = model
    state = initial
    rounds: list[RodSolveRound] = []
    for stage_index, stage in enumerate(stages):
        if stage.retransport_rounds < 1:
            raise RodError("a stage needs at least one round")
        vector = list(state.vector)
        for index, value in stage.prescribed:
            if not 0 <= index < len(vector):
                raise RodError(f"prescribed index out of range: {index}")
            vector[index] = _finite(f"prescribed[{index}]", value)
        state = state.with_vector(tuple(vector))
        for round_index in range(stage.retransport_rounds):
            terms = (*current_model.terms(), *stage.additional_terms)
            registry = EnergyRegistry(terms=terms)
            result = solve_equilibrium(
                registry,
                context,
                current_model.layout.layout,
                state.vector,
                fixed_indices=frozenset(stage.fixed_indices),
                residual_tol_n=residual_tol_n,
                max_iterations=max_iterations,
                max_backtracks=max_backtracks,
            )
            bending = current_model.bending().energy(result.state, context)
            twist = current_model.twist().energy(result.state, context)
            retransported = current_model.retransport(result.state)
            rounds.append(
                RodSolveRound(
                    stage_index,
                    round_index,
                    result,
                    bending,
                    twist,
                    retransported.max_gamma_change,
                    retransported.max_reference_twist,
                )
            )
            current_model = retransported.model
            state = retransported.state
    return RodEquilibrium(current_model, state, tuple(rounds))


__all__ = [
    "AnisotropicRodBending",
    "FRAME_TOLERANCE",
    "RodEndMoment",
    "RodEquilibrium",
    "RodError",
    "RodLayout",
    "RodMaterialFrame",
    "RodModel",
    "RodReference",
    "RodRetransport",
    "RodSolveRound",
    "RodSolveStage",
    "RodTwist",
    "build_bishop_frame",
    "build_material_frame",
    "build_rod_layout",
    "edge_tangents",
    "gammas_from_material_directors",
    "parallel_transport",
    "signed_angle",
    "solve_rod_with_retransport",
    "unwrap_phases",
]
