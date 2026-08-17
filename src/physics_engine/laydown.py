"""落位点几何层——位姿时间线与送带账之间缺的那一环（决策0067）。

## 场景（**先把它搞对，plans/14第3.3节第一版写错过**）

机器人搬着线圈骨架走，**张力机不动**。带材从固定的导轮组出来，入带点是
**世界系里固定的一个点**。不是带材去找线圈，是**机器人把线圈上该绕的那一段
送到这个固定点**。在途自由段的两个端点——最后那只导向轮的出带点、入带点——
**都是世界系固定的**，所以在途自由段是空间里**一条不动的线段**。

于是臂动扰动张力的通道**不是跨段长度**（它是常数），**是落位点的几何**：
线圈从那条不动的线段下方转过，落位点处**槽的切向在转、槽面法线在转**，
于是入射角与"这一瞬要放出多少带材"都在变。

`FreeSpanGeometry`因此是**两个纯常量端点**的冻结记录：跨段长度由类型保证
与``t``无关，**不给任何"跨段随臂变长"的写法留位置**。

## 本模块算什么

    位姿时间线(t) ＋ 槽中心线 ＋ 累计送带长度(t)
      →  落位点处的：弧长坐标、世界系槽三标架、入射角、所需送带率、**闭合残差**

它是`drives`（张力）与接触层之间缺的那一环：`motion`只回答"``t``时刻那个东西
在哪"，`drives`只回答"给定放线速度张力是多少"，**中间"这一瞬落位的是槽上哪一点、
它的帧朝哪、要放多少带材"没有任何东西回答**。

## 闭合条件：本模块的核心判据，也是它唯一不肯替调用方拿主意的地方

落位点必须**同时**满足两件事：

1. **它在槽上**——弧长坐标由送带账定：``σ_feed(t) = arc_origin + feed(t)``；
2. **它在那个固定入带点上**——由位姿定：``pose(t) · C(σ) = entry_point``。

**这两条一般不自洽。** 三个自由度的位置约束配一个未知量``σ``，超定二维。
本模块**不挑其中一条当成对的**，而是两条各算一次并把差额报出来：

* ``σ_feed``——送带账定的弧长坐标；
* ``σ_pose``——位姿定的：把固定入带点变回工件系，取中心线上离它最近的那一点；
* ``ClosureResidual``——两者的差额，**并且按方向拆开**：

  | 分量 | 含义 | 谁能修掉它 |
  |---|---|---|
  | ``along_tangent_mm`` | 沿槽切向 | **送带账**（多放或少放一点带材就能对上） |
  | ``transverse_mm`` | 垂直于槽切向（带宽向＋法向） | **送带账修不掉**——位姿根本没把槽送到入带点上 |
  | ``pose_only_offset_mm`` | 入带点到中心线本身的距离 | **谁都修不掉**（它只由位姿与几何定） |

**这个拆分是本模块存在的理由。** 一个只报"残差2.3 mm"的实现分不清
"送带账少算了2.3 mm"与"机器人把线圈举偏了2.3 mm"，而这两件事的处置完全不同。

## 语义为什么必须由声明者逐条给出（与`motion`同纪律）

中心线是一串**离散站点**。站点之间发生了什么，数据里没有这个信息——
它在声明者的脑子里。库替他挑一个"合理默认"的后果是确定的：两个调用方拿同一份
站点算出不同的落位点、不同的入射角、不同的物理，而**两边都以为自己是对的**。

所以``CenterlineSemantics``的五条**逐条必须显式给出，缺一即拒**，
且每条都是失败关闭的白名单：

1. ``position_interpolation``——``chord_linear``（弦线性）还是``hermite_tangent``
   （拿站点切向做三次Hermite）。两条的收敛阶差两阶，**不是口味问题**；
2. ``frame_interpolation``——``hold_station``（零阶保持）还是
   ``reorthonormalised_linear``（三个向量各自线性混合后重正交）。
   非平面槽的整匝帧扭转跨236°—657°（plans/14第2.2节），零阶保持在粗采样下
   直接把这段扭转丢掉；
3. ``topology``——``open``还是``closed``。真实槽是**闭合空间曲线**，
   而闭合曲线的弧长坐标要模``L``回绕，开曲线不许回绕；
4. ``out_of_range``——弧长坐标跑出站点范围怎么办。``wrap``**当且仅当**闭合
   （与`motion`的``rotation_arc='not_applicable'``同一条相容性纪律）；
5. ``nearest_refinement_iterations``——最近点搜索在粗扫之后再走几步牛顿。
   ``0``＝纯分段投影（对``chord_linear``**就是精确解**）；``hermite_tangent``
   下不加细化就等于用弦去逼近三次曲线，**把语义选择又还回去了**。

## 帧约定（抄GCW的`centerline.meta.json`，不自己发明）

``ordered_basis = [tangent, width_direction, surface_normal]``、
``width_direction = cross(surface_normal, tangent)``，即
``s = n × t``、``(t, s, n)``右手系。plans/14第二节把它钉死过一次，本模块照抄
并**当场校验**：站点帧不满足这条就拒。取错``n``与``s``在数值上不报任何错，
只是把带宽方向与法向对调——WDS `test_gravity_cantilever.py`记过同族失效
（参考``d1``取错让挠度差1600倍而**不报任何错**）。

## 入射角的符号约定（**读之前先看这一段，否则会差一个π**）

自由段上的带材，**材料坐标是往上游增大的**：入带点处正在落位的是弧长``σ(t)``，
而自由段上更靠近导轮的那一段要**更晚**才落位，对应更大的``σ``。所以自由段
"材料坐标增大"的方向是**从入带点指向导轮**，即``normalize(guide − entry)``。

于是本模块把入射角定义为**那个方向与落位点槽切向的夹角**：

    incidence_angle_rad = angle(normalize(guide − entry), tangent_world)

**``0``表示带材无折角地续上槽**。这条不是"哪个更好看"，是唯一让"理想落位＝0"
成立的取法；取反向的话理想值是``π``，而一个理想值在``π``的量，
它的小量展开、它的容差、它在张力式子里的位置全部要跟着变号。

同时给出两个分量（在落位点的``(t, s, n)``系里）：

* ``incidence_in_plane_rad``——槽面内的方位角``atan2(d·s, d·t)``，
  它是横向导入/蹭边那一路（对照`cases/roller_skew_lateral_drift`的``θ_r``）；
* ``incidence_out_of_plane_rad``——离面仰角``asin(d·n)``，它是压紧/起翘那一路。

## 面（轴1规则1）

本模块**不落盘、不跨边界**，因此**不需要新的面**。落位点时间线哪天要写进
run package，那时才需要一个``physics_laydown_track``面，且要先去
``engine_facets.py``登记再落盘。

## 本模块**不**算什么

* **不算张力**。张力是`drives`的事；本层只交出它要的那几个几何量；
* **不算接触**。槽壁是两段外倾锥面（plans/14第3.4节），本层连槽宽都不知道；
* **不算带材自己的形状**。带材可以绕自身切向扭转把硬弯换成扭转
  （plans/14第2.3节），那是轨道A的杆；本层给的是**槽的**帧，不是带材的帧；
* **不解闭合**。它只报残差，**不去调送带账也不去调位姿**——两条都不是本层的权力。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from physics_engine.identity import parse_namespace_id
from physics_engine.motion import MotionSource, Pose

Vector3 = tuple[float, float, float]


class LaydownError(ValueError):
    """落位点几何层的一切失败关闭。"""


#: 站点帧的正交/单位容差。**与`motion.QUATERNION_NORM_ABS_TOL`同量级**（1e-9）——
#: 一个帧在这里合法、装进位姿链却被判非正交（或反过来）是最难查的那种不一致。
FRAME_ORTHONORMAL_ABS_TOL = 1.0e-9

#: 弧长/弦长的上限。**这条门判的是"采样解析得了这条曲线吗"，不是精度。**
#: 圆弧上``弧/弦 = θ/(2 sin(θ/2))``：θ=60°给1.0472、θ=62°给1.0503，
#: 所以1.05这条线≈"一段站点间隔不许转过62°"。真实工件的采样步是2 mm、
#: 最小曲率半径27.6 mm（plans/14第2.2节），每段转4.2°、比值1.0002——
#: **这条门离真实数据有三个数量级余量，它抓的是"把序号当弧长传进来"那类错**。
ARC_OVER_CHORD_CEILING = 1.05

#: 长度单位。**本仓已经栽过两次1000倍单位bug**（`motion.POSE_TRANSLATION_UNIT`
#: 那条常量就是为此立的），所以这里同样白名单失败关闭而不是替调用方换算。
CENTERLINE_LENGTH_UNIT = "mm"
ACCEPTED_LENGTH_UNITS: frozenset[str] = frozenset({CENTERLINE_LENGTH_UNIT})

#: 五条中心线语义各自的白名单（失败关闭；轴2"只增不改"同样适用）。
POSITION_INTERPOLATIONS: frozenset[str] = frozenset({"chord_linear", "hermite_tangent"})
FRAME_INTERPOLATIONS: frozenset[str] = frozenset(
    {"hold_station", "reorthonormalised_linear"}
)
CENTERLINE_TOPOLOGIES: frozenset[str] = frozenset({"open", "closed"})
ARC_OUT_OF_RANGE: frozenset[str] = frozenset({"reject", "clamp_to_end", "wrap"})

#: 速率探针的差分格式白名单。**没有默认值**：中心差分与单边差分的截断阶不同
#: （h²对h），而在时间线端点上中心差分**根本取不到样点**——取哪一条只有声明者知道。
RATE_SCHEMES: frozenset[str] = frozenset({"central", "forward", "backward"})


# ---------------------------------------------------------------- 向量原语 ---
# 零运行时依赖（AGENTS.md本仓纪律）：不许numpy。这些是**有名字的模块级函数**，
# 为的是测试能单独钉住它们，而不是把三行内联代码复制十遍。


def _sub(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _add(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale(vector: Sequence[float], factor: float) -> Vector3:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalise(vector: Sequence[float], what: str) -> Vector3:
    length = _norm(vector)
    if length == 0.0:
        raise LaydownError(f"{what}: 零向量没有方向")
    return _scale(vector, 1.0 / length)


def _attitude_rows(rotation_xyzw: Sequence[float]) -> tuple[Vector3, Vector3, Vector3]:
    """体→世界的旋转矩阵``R(q)``，**xyzw**次序。

    式子与`rigidbody.attitude_matrix`、`shapes.PosedBody.rotate_local_mm`
    **逐字符相同**（连括号次序都没动），所以三者在同一份输入上给出的是
    **逐位相同**的浮点数，而不是"数学上等价"。

    没有直接import `rigidbody`是**冷启动预算**的考虑而不是洁癖：`rigidbody`
    会连带拉进`state`/`integrate`/`energies`/`geometry`/`shapes`五个模块
    （`__init__.py`那段"顶层eager import的模块数是冷启动延迟的结构代理"），
    而本模块只要这九个表达式。`tests/test_laydown.py`有一条门**逐位对拍**两者
    ——抄一份公式而不对拍它，就是本仓最怕的"第二套四元数约定"。
    """

    x, y, z, w = (
        rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2], rotation_xyzw[3],
    )
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def rotate_by_quaternion(rotation_xyzw: Sequence[float], vector: Sequence[float]) -> Vector3:
    """体系向量 → 世界系向量。与`rigidbody.rotate_body_to_world`逐位相同。"""

    rows = _attitude_rows(rotation_xyzw)
    return tuple(  # type: ignore[return-value]
        sum(row[index] * vector[index] for index in range(3)) for row in rows
    )


def _inverse_rotate(rotation_xyzw: Sequence[float], vector: Sequence[float]) -> Vector3:
    """世界系向量 → 体系向量（``Rᵀ``）。与`rigidbody.rotate_world_to_body`逐位相同。"""

    rows = _attitude_rows(rotation_xyzw)
    return tuple(  # type: ignore[return-value]
        sum(rows[index][axis] * vector[index] for index in range(3)) for axis in range(3)
    )


# ---------------------------------------------------------------- 校验原语 ---


def _require_namespace(value: object, prefix: str, what: str) -> str:
    if not isinstance(value, str):
        raise LaydownError(f"{what} must be a string: {value!r}")
    if not value.startswith(f"{prefix}/"):
        raise LaydownError(f"{what} must be namespaced like {prefix!r}/…: {value!r}")
    try:
        parse_namespace_id(value)
    except ValueError as error:  # IdentityError继承自ValueError
        raise LaydownError(f"{what} is not a valid namespace id: {error}") from error
    return value


def _require_finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LaydownError(f"{what} must be a real number: {value!r}")
    if not math.isfinite(value):
        raise LaydownError(f"{what} must be finite: {value!r}")
    return float(value)


def _require_vector(value: object, what: str) -> Vector3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise LaydownError(f"{what} must be a 3-tuple: {value!r}")
    return (
        _require_finite(value[0], f"{what}[0]"),
        _require_finite(value[1], f"{what}[1]"),
        _require_finite(value[2], f"{what}[2]"),
    )


def _require_unit_vector(value: object, what: str) -> Vector3:
    vector = _require_vector(value, what)
    norm = _norm(vector)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=FRAME_ORTHONORMAL_ABS_TOL):
        raise LaydownError(f"{what} must be a unit vector, norm is {norm!r}")
    return vector


def _require_declared_choice(value: object, allowed: frozenset[str], what: str) -> str:
    """一条中心线语义。**缺省不受理**——没有默认值可用，白名单外一律拒。"""

    if value is None:
        raise LaydownError(
            f"{what} must be declared explicitly — 中心线语义是声明者的事，"
            f"不是库替他猜的。可选：{sorted(allowed)}"
        )
    if not isinstance(value, str) or value not in allowed:
        raise LaydownError(
            f"{what} must be one of {sorted(allowed)}, got {value!r} — "
            "白名单失败关闭；加一个取值要改laydown.py并补一条测试"
        )
    return value


def _require_wrap_matches_topology(topology: str, out_of_range: str) -> None:
    """回绕与拓扑的相容性——与`motion._require_arc_matches_rotation`同一条纪律。

    开曲线没有"绕回来"这回事：``σ = L + 1``在开曲线上是**出界**，
    在闭曲线上是**站点1 mm之后**，两者差一整圈的材料。
    """

    if topology == "closed" and out_of_range != "wrap":
        raise LaydownError(
            f"topology='closed'的中心线上弧长坐标必须模总长回绕，"
            f"out_of_range必须是'wrap'，得到{out_of_range!r}"
        )
    if topology == "open" and out_of_range == "wrap":
        raise LaydownError(
            "topology='open'的中心线没有'绕回来'这回事——"
            "out_of_range='wrap'会把出界静默读成'又一圈'，差的是一整匝材料"
        )


# ------------------------------------------------------------------ 站点 ---


@dataclass(frozen=True)
class GrooveStation:
    """槽中心线上的一个站点：弧长 + 位置 + 三标架。**工件系**。

    帧约定抄GCW的``centerline.meta.json``（plans/14第二节）：
    ``ordered_basis = [tangent, width_direction, surface_normal]``、
    ``width_direction = cross(surface_normal, tangent)``。
    即``s = n × t``、``(t, s, n)``右手系。**构造期当场校验**，不留"以后再说"。
    """

    arc_length_mm: float
    position_mm: Vector3
    tangent: Vector3
    width_direction: Vector3
    surface_normal: Vector3

    def __post_init__(self) -> None:
        _require_finite(self.arc_length_mm, "arc_length_mm")
        _require_vector(self.position_mm, "position_mm")
        tangent = _require_unit_vector(self.tangent, "tangent")
        width = _require_unit_vector(self.width_direction, "width_direction")
        normal = _require_unit_vector(self.surface_normal, "surface_normal")
        expected = _cross(normal, tangent)
        gap = _norm(_sub(width, expected))
        if gap > FRAME_ORTHONORMAL_ABS_TOL:
            raise LaydownError(
                f"arc {self.arc_length_mm!r}: 帧不满足width_direction = cross("
                f"surface_normal, tangent)，差{gap!r} > {FRAME_ORTHONORMAL_ABS_TOL!r}"
                "——取错n与s在数值上不报任何错，只是把带宽方向与法向对调，"
                "所以这一条在构造期就判"
            )
        if abs(_dot(tangent, normal)) > FRAME_ORTHONORMAL_ABS_TOL:
            raise LaydownError(
                f"arc {self.arc_length_mm!r}: tangent与surface_normal不正交，"
                f"内积{_dot(tangent, normal)!r}"
            )

    def triad(self) -> tuple[Vector3, Vector3, Vector3]:
        return (self.tangent, self.width_direction, self.surface_normal)


@dataclass(frozen=True)
class GrooveSample:
    """中心线上任一弧长坐标处的位置与三标架。与`GrooveStation`同系（工件系）。"""

    arc_length_mm: float
    position_mm: Vector3
    tangent: Vector3
    width_direction: Vector3
    surface_normal: Vector3


@dataclass(frozen=True)
class CenterlineSemantics:
    """五条中心线语义。**逐条显式，缺一即拒**——理由见模块文档。"""

    position_interpolation: str
    frame_interpolation: str
    topology: str
    out_of_range: str
    nearest_refinement_iterations: int

    def __post_init__(self) -> None:
        _require_declared_choice(
            self.position_interpolation, POSITION_INTERPOLATIONS, "position_interpolation"
        )
        _require_declared_choice(
            self.frame_interpolation, FRAME_INTERPOLATIONS, "frame_interpolation"
        )
        _require_declared_choice(self.topology, CENTERLINE_TOPOLOGIES, "topology")
        _require_declared_choice(self.out_of_range, ARC_OUT_OF_RANGE, "out_of_range")
        _require_wrap_matches_topology(self.topology, self.out_of_range)
        if type(self.nearest_refinement_iterations) is not int:
            raise LaydownError(
                "nearest_refinement_iterations must be a real int, got "
                f"{self.nearest_refinement_iterations!r}"
            )
        if self.nearest_refinement_iterations < 0:
            raise LaydownError(
                "nearest_refinement_iterations不能为负："
                f"{self.nearest_refinement_iterations!r}"
            )


# -------------------------------------------------------------- 中心线 ---


def _require_stations(
    stations: tuple[GrooveStation, ...], semantics: CenterlineSemantics, centerline_id: str
) -> None:
    """站点表的四条结构校验。每一条各挡一类**不会自己报错**的输入。"""

    if len(stations) < 3:
        raise LaydownError(
            f"{centerline_id}: 一条中心线至少要三个站点，得到{len(stations)}个"
            "——两个站点连'曲线'都谈不上，最近点搜索会退化成一条线段"
        )
    if stations[0].arc_length_mm != 0.0:
        raise LaydownError(
            f"{centerline_id}: 弧长坐标必须从0起，得到{stations[0].arc_length_mm!r}"
        )
    for left, right in zip(stations, stations[1:], strict=False):
        if right.arc_length_mm <= left.arc_length_mm:
            raise LaydownError(
                f"{centerline_id}: 弧长坐标必须严格递增，"
                f"得到{left.arc_length_mm!r}之后是{right.arc_length_mm!r}"
            )
        chord = _norm(_sub(right.position_mm, left.position_mm))
        span = right.arc_length_mm - left.arc_length_mm
        if chord == 0.0:
            raise LaydownError(
                f"{centerline_id}: arc {left.arc_length_mm!r}与{right.arc_length_mm!r}"
                "两个站点位置重合，弧长却在前进"
            )
        if span < chord * (1.0 - FRAME_ORTHONORMAL_ABS_TOL):
            raise LaydownError(
                f"{centerline_id}: arc {left.arc_length_mm!r}→{right.arc_length_mm!r}的"
                f"弧长增量{span!r}小于弦长{chord!r}——**弧长永远不小于弦长**，"
                "这一条不是精度问题，是弧长列根本不是弧长"
            )
        if span > chord * ARC_OVER_CHORD_CEILING:
            raise LaydownError(
                f"{centerline_id}: arc {left.arc_length_mm!r}→{right.arc_length_mm!r}的"
                f"弧长/弦长 = {span / chord!r} 超过上限{ARC_OVER_CHORD_CEILING!r}"
                "——采样解析不了这一段（≈一段转过62°以上），或者弧长列传错了"
            )
        if _dot(left.tangent, _sub(right.position_mm, left.position_mm)) <= 0.0:
            raise LaydownError(
                f"{centerline_id}: arc {left.arc_length_mm!r}处的切向与前进方向反了"
                "——切向必须指向弧长增大的一侧。整表反号在数值上不报任何错，"
                "只是把s = n × t整个镜像掉"
            )
    if semantics.topology == "closed":
        first, last = stations[0], stations[-1]
        seam = _norm(_sub(last.position_mm, first.position_mm))
        if seam > FRAME_ORTHONORMAL_ABS_TOL:
            raise LaydownError(
                f"{centerline_id}: topology='closed'要求**末站点逐位重复首站点**"
                f"（位置差{seam!r}）。这不是形式：闭合曲线的总长``L``就是末站点的"
                "弧长坐标，而'首末差一个采样步'那种表（真实CSV就是这样）没有任何东西"
                "能告诉库那一步有多长——补上闭合站点是声明者的事"
            )
        for name, left_vec, right_vec in (
            ("tangent", first.tangent, last.tangent),
            ("width_direction", first.width_direction, last.width_direction),
            ("surface_normal", first.surface_normal, last.surface_normal),
        ):
            if _norm(_sub(left_vec, right_vec)) > FRAME_ORTHONORMAL_ABS_TOL:
                raise LaydownError(
                    f"{centerline_id}: topology='closed'的首末站点{name}不一致——"
                    "帧在接缝处跳变，绕过接缝的入射角会跳一个台阶"
                )


@dataclass(frozen=True)
class GrooveCenterline:
    """槽中心线：一串站点 + **显式声明的**五条语义。**工件系**，由位姿变到世界系。

    没有一个字段带默认值：本层的每一条都是声明者要拿主意的东西，
    而"默认值"正是把主意悄悄替他拿了的那种写法（与`motion.SampledPoseTimeline`同源）。
    """

    centerline_id: str
    stations: tuple[GrooveStation, ...]
    semantics: CenterlineSemantics
    length_unit: str

    def __post_init__(self) -> None:
        _require_namespace(self.centerline_id, "groove", "centerline_id")
        if not isinstance(self.semantics, CenterlineSemantics):
            raise LaydownError(
                f"{self.centerline_id}: semantics must be a CenterlineSemantics, "
                f"got {self.semantics!r}"
            )
        if self.length_unit not in ACCEPTED_LENGTH_UNITS:
            raise LaydownError(
                f"{self.centerline_id}: length_unit must be one of "
                f"{sorted(ACCEPTED_LENGTH_UNITS)}, got {self.length_unit!r} — "
                "以米声明的中心线会整整差1000倍，本仓已经栽过两次这个bug，"
                "所以这里失败关闭而不是替你换算"
            )
        if not isinstance(self.stations, tuple):
            raise LaydownError(f"{self.centerline_id}: stations must be a tuple")
        for station in self.stations:
            if not isinstance(station, GrooveStation):
                raise LaydownError(
                    f"{self.centerline_id}: not a GrooveStation: {station!r}"
                )
        _require_stations(self.stations, self.semantics, self.centerline_id)

    # -- 基本量 ------------------------------------------------------------

    def total_arc_length_mm(self) -> float:
        """末站点的弧长坐标。闭合时它就是一整匝的长度（末站点重复首站点）。"""

        return self.stations[-1].arc_length_mm

    def segment_count(self) -> int:
        return len(self.stations) - 1

    # -- 弧长坐标的定义域策略 ----------------------------------------------

    def resolve_arc_length_mm(self, arc_mm: float) -> float:
        """把任意弧长坐标按声明的``out_of_range``policy落进``[0, L]``。"""

        arc_mm = _require_finite(arc_mm, f"{self.centerline_id}: arc_length_mm")
        total = self.total_arc_length_mm()
        if self.semantics.out_of_range == "wrap":
            return arc_mm % total
        if 0.0 <= arc_mm <= total:
            return arc_mm
        if self.semantics.out_of_range == "reject":
            raise LaydownError(
                f"{self.centerline_id}: 弧长坐标{arc_mm!r}落在[0, {total!r}]之外，"
                "out_of_range='reject'——开曲线之外没有任何站点支持的槽，"
                "要接着走就去把中心线给全"
            )
        return 0.0 if arc_mm < 0.0 else total

    def arc_difference_mm(self, later_mm: float, earlier_mm: float) -> float:
        """两个弧长坐标之差。**闭合曲线在接缝处要解卷**，否则差出一整匝。

        取法：把差额折进``(−L/2, L/2]``。这条对"一步走过半匝"的时间线不成立——
        那种时间线本来就采样不足，**本模块不替它猜是绕了几圈**。
        """

        difference = later_mm - earlier_mm
        if self.semantics.topology != "closed":
            return difference
        total = self.total_arc_length_mm()
        folded = (difference + 0.5 * total) % total - 0.5 * total
        return folded

    # -- 采样 --------------------------------------------------------------

    def _segment_index(self, arc_mm: float) -> int:
        index = bisect_right(
            self.stations, arc_mm, key=lambda station: station.arc_length_mm
        ) - 1
        return min(max(index, 0), self.segment_count() - 1)

    def _position_on_segment(self, index: int, arc_mm: float) -> tuple[Vector3, Vector3, Vector3]:
        """段内位置与一阶/二阶导（对弧长坐标求）。牛顿细化要用后两个。"""

        left = self.stations[index]
        right = self.stations[index + 1]
        span = right.arc_length_mm - left.arc_length_mm
        u = (arc_mm - left.arc_length_mm) / span
        if self.semantics.position_interpolation == "chord_linear":
            chord = _sub(right.position_mm, left.position_mm)
            return (
                _add(left.position_mm, _scale(chord, u)),
                _scale(chord, 1.0 / span),
                (0.0, 0.0, 0.0),
            )
        # hermite_tangent：以**弧长**为参数、以单位切向为导数的三次Hermite。
        # 弧长参数化下``dC/ds``恰是单位切向，所以这不是拟合而是精确的一阶匹配，
        # 截断阶因此比弦线性高两阶。
        uu = u * u
        uuu = uu * u
        h00, h10 = 2.0 * uuu - 3.0 * uu + 1.0, uuu - 2.0 * uu + u
        h01, h11 = -2.0 * uuu + 3.0 * uu, uuu - uu
        d00, d10 = 6.0 * uu - 6.0 * u, 3.0 * uu - 4.0 * u + 1.0
        d01, d11 = -6.0 * uu + 6.0 * u, 3.0 * uu - 2.0 * u
        e00, e10 = 12.0 * u - 6.0, 6.0 * u - 4.0
        e01, e11 = -12.0 * u + 6.0, 6.0 * u - 2.0
        position = _add(
            _add(_scale(left.position_mm, h00), _scale(left.tangent, h10 * span)),
            _add(_scale(right.position_mm, h01), _scale(right.tangent, h11 * span)),
        )
        first = _scale(
            _add(
                _add(_scale(left.position_mm, d00), _scale(left.tangent, d10 * span)),
                _add(_scale(right.position_mm, d01), _scale(right.tangent, d11 * span)),
            ),
            1.0 / span,
        )
        second = _scale(
            _add(
                _add(_scale(left.position_mm, e00), _scale(left.tangent, e10 * span)),
                _add(_scale(right.position_mm, e01), _scale(right.tangent, e11 * span)),
            ),
            1.0 / (span * span),
        )
        return (position, first, second)

    def _frame_on_segment(self, index: int, arc_mm: float) -> tuple[Vector3, Vector3, Vector3]:
        left = self.stations[index]
        if self.semantics.frame_interpolation == "hold_station":
            return left.triad()
        right = self.stations[index + 1]
        span = right.arc_length_mm - left.arc_length_mm
        u = (arc_mm - left.arc_length_mm) / span
        tangent = _normalise(
            _add(_scale(left.tangent, 1.0 - u), _scale(right.tangent, u)),
            f"{self.centerline_id}: 混合切向",
        )
        blended_normal = _add(
            _scale(left.surface_normal, 1.0 - u), _scale(right.surface_normal, u)
        )
        normal = _normalise(
            _sub(blended_normal, _scale(tangent, _dot(blended_normal, tangent))),
            f"{self.centerline_id}: 混合法向",
        )
        # 带宽方向**不混合而是重算**：``s = n × t``是约定不是数据，
        # 混出来再正交化等于给这条约定留一个数值误差的口子。
        return (tangent, _cross(normal, tangent), normal)

    def sample_at(self, arc_mm: float) -> GrooveSample:
        """弧长坐标 → 工件系的位置与三标架。"""

        resolved = self.resolve_arc_length_mm(arc_mm)
        index = self._segment_index(resolved)
        position, _, _ = self._position_on_segment(index, resolved)
        tangent, width, normal = self._frame_on_segment(index, resolved)
        return GrooveSample(
            arc_length_mm=resolved,
            position_mm=position,
            tangent=tangent,
            width_direction=width,
            surface_normal=normal,
        )

    # -- 最近点 ------------------------------------------------------------

    def nearest_arc_length_mm(self, point_mm: Sequence[float]) -> tuple[float, float]:
        """中心线上离``point_mm``（**工件系**）最近的点：``(弧长坐标, 距离)``。

        两步：**分段弦投影粗扫**（对``chord_linear``就是精确解），
        再在最优段与它的两个邻段上各走``nearest_refinement_iterations``步
        安全牛顿（对``hermite_tangent``必要）。

        **已知的失效形态**：粗扫用弦、细化用声明的插值式，两者在
        ``hermite_tangent``＋极粗采样下可能选到相邻段；取三段候选里最好的一个
        正是为此。真正病态（曲率半径与采样步同量级）时本方法**不保证全局最近点**，
        `case.md`第四节把这一条列在已知失效清单里而不是藏起来。
        """

        target = _require_vector(tuple(point_mm), f"{self.centerline_id}: point_mm")
        best_index, best_arc, best_distance = 0, 0.0, math.inf
        for index in range(self.segment_count()):
            left = self.stations[index]
            right = self.stations[index + 1]
            chord = _sub(right.position_mm, left.position_mm)
            span = right.arc_length_mm - left.arc_length_mm
            u = _dot(_sub(target, left.position_mm), chord) / _dot(chord, chord)
            u = min(1.0, max(0.0, u))
            candidate = _add(left.position_mm, _scale(chord, u))
            distance = _norm(_sub(target, candidate))
            if distance < best_distance:
                best_index, best_arc = index, left.arc_length_mm + u * span
                best_distance = distance
        if self.semantics.nearest_refinement_iterations == 0:
            return (best_arc, best_distance)
        segments = self.segment_count()
        neighbours = [best_index]
        for offset in (-1, 1):
            index = best_index + offset
            if self.semantics.topology == "closed":
                index %= segments
            if 0 <= index < segments and index not in neighbours:
                neighbours.append(index)
        for index in neighbours:
            arc, distance = self._refine_on_segment(index, target, best_arc)
            if distance < best_distance:
                best_arc, best_distance = arc, distance
        return (best_arc, best_distance)

    def _refine_on_segment(
        self, index: int, target: Vector3, seed_arc_mm: float
    ) -> tuple[float, float]:
        """段内安全牛顿求``g(σ) = (C(σ) − p)·C'(σ) = 0``。步出段就夹回段内。"""

        low = self.stations[index].arc_length_mm
        high = self.stations[index + 1].arc_length_mm
        arc = min(high, max(low, seed_arc_mm))
        for _ in range(self.semantics.nearest_refinement_iterations):
            position, first, second = self._position_on_segment(index, arc)
            delta = _sub(position, target)
            gradient = _dot(delta, first)
            curvature = _dot(first, first) + _dot(delta, second)
            if curvature <= 0.0:
                break
            step = gradient / curvature
            updated = min(high, max(low, arc - step))
            if updated == arc:
                break
            arc = updated
        position, _, _ = self._position_on_segment(index, arc)
        return (arc, _norm(_sub(position, target)))


# ------------------------------------------------------------ 自由跨段 ---


@dataclass(frozen=True)
class FreeSpanGeometry:
    """在途自由段：世界系里**一条不动的线段**。

    两个端点都是世界系常量——最后那只导向轮的出带点、入带点。
    plans/14第3.3节记的用户2026-08-14口述订正了更早的说法：
    "在途自由段的两个端点……**都是世界系固定的**"。

    **跨段长度因此与``t``无关，由类型保证。** 这不是省事：本页第一版写成
    "一端固定一端随臂运动"，于是"臂动⟹跨段变长⟹张力变"这条错误的因果
    看起来完全自洽。把两个端点冻成常量之后，那条路**在类型上就走不通了**。
    """

    span_id: str
    guide_exit_mm: Vector3
    entry_point_mm: Vector3

    def __post_init__(self) -> None:
        _require_namespace(self.span_id, "span", "span_id")
        guide = _require_vector(self.guide_exit_mm, f"{self.span_id}: guide_exit_mm")
        entry = _require_vector(self.entry_point_mm, f"{self.span_id}: entry_point_mm")
        if _norm(_sub(entry, guide)) == 0.0:
            raise LaydownError(
                f"{self.span_id}: 出带点与入带点重合，自由段没有方向——"
                "而入射角整个建立在这个方向上"
            )

    def length_mm(self) -> float:
        """跨段长度。**常数**：它不接收``t``，也没有任何路径能让它接收``t``。"""

        return _norm(_sub(self.entry_point_mm, self.guide_exit_mm))

    def material_increasing_direction(self) -> Vector3:
        """自由段上**材料坐标增大**的方向：从入带点指向导轮。

        符号约定见模块文档"入射角的符号约定"一段。取反的话理想落位角是``π``。
        """

        return _normalise(
            _sub(self.guide_exit_mm, self.entry_point_mm),
            f"{self.span_id}: material_increasing_direction",
        )


# -------------------------------------------------------------- 送带账 ---


@dataclass(frozen=True)
class FeedAccount:
    """累计送带长度``t → mm``，单调非减。

    与`motion.AnalyticPose`同一条纪律：本层拿到的是一个**任意可调用对象**，
    它可以去读时钟、读随机数。**库证明不了一个函数单调、更证明不了它是纯的**，
    所以这里做的是**证伪**而不是证明——构造期在``probe_times_s``上
    各求值两次（判纯）并逐点比较（判单调）。

    这条能抓住"把速度当成累计长度传进来"（非单调）与"读时钟"（不纯）
    这两类最常见的写法，**抓不住**"只在探针之外的某段回退"那种。
    如实登记在此，不许被"过了校验"盖过。
    """

    account_id: str
    length_fn: Callable[[float], float]
    probe_times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.account_id, "feed", "account_id")
        if not callable(self.length_fn):
            raise LaydownError(f"{self.account_id}: length_fn must be callable")
        if not isinstance(self.probe_times_s, tuple) or not self.probe_times_s:
            raise LaydownError(
                f"{self.account_id}: 至少要给一个探针时刻——"
                "一条没有配证伪尝试的单调声明就是冒充（AGENTS.md诚实可信度）"
            )
        previous_time = -math.inf
        previous_length = -math.inf
        for probe in self.probe_times_s:
            time_s = _require_finite(probe, f"{self.account_id}: probe time")
            if time_s <= previous_time:
                raise LaydownError(
                    f"{self.account_id}: probe_times_s必须严格递增，"
                    f"{previous_time!r}之后是{time_s!r}——乱序的探针判不了单调"
                )
            first = self.length_at(time_s)
            second = self.length_at(time_s)
            if first != second:
                raise LaydownError(
                    f"{self.account_id}: length_fn在t_s={time_s!r}给了两个不同的值"
                    f"（{first!r}然后{second!r}）——它不是纯函数，送带账不可重放"
                )
            if first < previous_length:
                raise LaydownError(
                    f"{self.account_id}: 累计送带长度必须单调非减，"
                    f"{previous_length!r}之后是{first!r}——"
                    "**带材不会被吸回张力机**；把放线速度当累计长度传进来是这条门的常客"
                )
            previous_time, previous_length = time_s, first

    def length_at(self, t_s: float) -> float:
        value = self.length_fn(_require_finite(t_s, f"{self.account_id}: t_s"))
        return _require_finite(value, f"{self.account_id}: length at t_s={t_s!r}")


@dataclass(frozen=True)
class ArcRateProbe:
    """速率探针：差分格式 + 步长。**两条都必须显式给出。**

    为什么不给默认值：中心差分是``h²``、单边是``h``，而在时间线**端点**上
    中心差分根本取不到样点（`motion`那边``extrapolation='reject'``会当场拒）。
    一个默认值会让端点上的速率静默变成"夹到端点之后的差商"，
    那不是截断误差，是**另一个量**。
    """

    scheme: str
    step_s: float

    def __post_init__(self) -> None:
        _require_declared_choice(self.scheme, RATE_SCHEMES, "scheme")
        step = _require_finite(self.step_s, "step_s")
        if step <= 0.0:
            raise LaydownError(f"step_s必须为正，得到{step!r}")

    def stencil(self, t_s: float) -> tuple[float, float, float]:
        """``(左时刻, 右时刻, 分母)``。"""

        if self.scheme == "central":
            return (t_s - self.step_s, t_s + self.step_s, 2.0 * self.step_s)
        if self.scheme == "forward":
            return (t_s, t_s + self.step_s, self.step_s)
        return (t_s - self.step_s, t_s, self.step_s)


# ------------------------------------------------------------ 闭合残差 ---


@dataclass(frozen=True)
class ClosureResidual:
    """闭合残差：送带账定的落位点与位姿定的入带点之间对不上的那一部分。

    **按方向拆开**，因为"沿槽差2 mm"与"离槽2 mm"处置完全不同：

    * ``along_tangent_mm``——沿槽切向。**送带账多放/少放一点就能对上**；
    * ``across_width_mm`` / ``across_normal_mm``——槽面内横向、离面。
      ``transverse_mm``是这两者的模，**送带账修不掉它**；
    * ``arc_gap_mm``——``σ_feed − σ_pose``，沿槽那一份换算成弧长坐标之差
      （闭合曲线上已解卷）；
    * ``pose_only_offset_mm``——入带点到中心线本身的最近距离。
      **它只由位姿与几何定，改送带账一分钱都动不了它。**

    ``transverse_mm``与``pose_only_offset_mm``的区别值得写清楚：前者是
    **落位点处**的横向差，后者是**整条曲线**离入带点最近能到多近。
    ``σ_feed = σ_pose``时两者相等；``σ_feed``偏了之后前者会被曲线自身的
    曲率放大，而后者纹丝不动。
    """

    position_mm: Vector3
    magnitude_mm: float
    along_tangent_mm: float
    across_width_mm: float
    across_normal_mm: float
    transverse_mm: float
    arc_gap_mm: float
    pose_only_offset_mm: float


@dataclass(frozen=True)
class ClosureTolerance:
    """闭合残差的验收线。**理由必须写**（GROMACS式成对容差的本仓纪律）。

    没有默认值：多大的闭合残差算"这条时间线与这本送带账是一致的"，
    是**声明者**要拿主意的事。库给一个默认值等于替他把整条轨道的判据定了。
    """

    position_abs_mm: float
    arc_abs_mm: float
    reason: str

    def __post_init__(self) -> None:
        position = _require_finite(self.position_abs_mm, "position_abs_mm")
        arc = _require_finite(self.arc_abs_mm, "arc_abs_mm")
        if position < 0.0 or arc < 0.0:
            raise LaydownError(
                f"闭合容差不能为负：position_abs_mm={position!r}、arc_abs_mm={arc!r}"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise LaydownError(
                "一条闭合容差必须说明它凭什么是这个数——"
                "没有理由的容差与'调到能过为止'在字节上没有区别"
            )


# ------------------------------------------------------------ 落位点 ---


@dataclass(frozen=True)
class LaydownPoint:
    """一个时刻的落位点几何。**两条弧长坐标都给出，本层不挑哪一条是对的。**"""

    time_s: float
    feed_arc_length_mm: float
    pose_arc_length_mm: float
    world_position_mm: Vector3
    tangent: Vector3
    width_direction: Vector3
    surface_normal: Vector3
    incidence_angle_rad: float
    incidence_in_plane_rad: float
    incidence_out_of_plane_rad: float
    required_feed_rate_mm_s: float
    accounted_feed_rate_mm_s: float
    closure: ClosureResidual

    def feed_rate_gap_mm_s(self) -> float:
        """位姿要的速率 减 送带账给的速率。闭合残差在**速率**上的那一面。"""

        return self.required_feed_rate_mm_s - self.accounted_feed_rate_mm_s


@dataclass(frozen=True)
class LaydownModel:
    """把四件输入装在一起：位姿时间线、槽中心线、自由跨段、送带账。

    ``arc_origin_mm``是**显式**的：送带账给的是"从开机到现在放了多少米"，
    而"开机那一刻落位的是槽上哪一点"是另一件事，两者之差就是这个原点。
    留默认值等于替调用方把线圈的起绕点定在弧长0处。
    """

    model_id: str
    motion: MotionSource
    centerline: GrooveCenterline
    span: FreeSpanGeometry
    feed: FeedAccount
    arc_origin_mm: float
    rate_probe: ArcRateProbe

    def __post_init__(self) -> None:
        _require_namespace(self.model_id, "laydown", "model_id")
        if not isinstance(self.motion, MotionSource):
            raise LaydownError(
                f"{self.model_id}: motion不是MotionSource"
                "（pose_at/horizon_s/is_replayable三个都要有）"
            )
        for field_name, expected in (
            ("centerline", GrooveCenterline),
            ("span", FreeSpanGeometry),
            ("feed", FeedAccount),
            ("rate_probe", ArcRateProbe),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, expected):
                raise LaydownError(
                    f"{self.model_id}: {field_name}必须是{expected.__name__}，"
                    f"得到{value!r}"
                )
        _require_finite(self.arc_origin_mm, f"{self.model_id}: arc_origin_mm")

    # -- 两条各自独立的弧长坐标 -------------------------------------------

    def feed_arc_length_mm(self, t_s: float) -> float:
        """**送带账定的**弧长坐标：``arc_origin + 累计送带长度(t)``。"""

        return self.centerline.resolve_arc_length_mm(
            self.arc_origin_mm + self.feed.length_at(t_s)
        )

    def pose_arc_length_mm(self, t_s: float) -> tuple[float, float]:
        """**位姿定的**弧长坐标与它的不可约偏距：``(σ_pose, 入带点到中心线的距离)``。

        做法是把**世界系固定的入带点变回工件系**再在中心线上找最近点——
        而不是把整条中心线变到世界系。两者数学等价，但前者只转一个点。
        """

        pose = self._pose_at(t_s)
        local = _inverse_rotate(
            pose.rotation_xyzw, _sub(self.span.entry_point_mm, pose.translation_mm)
        )
        return self.centerline.nearest_arc_length_mm(local)

    def _pose_at(self, t_s: float) -> Pose:
        pose = self.motion.pose_at(_require_finite(t_s, f"{self.model_id}: t_s"))
        if not isinstance(pose, Pose):
            raise LaydownError(f"{self.model_id}: pose_at返回了{pose!r}，不是Pose")
        return pose

    def world_sample(self, t_s: float, arc_mm: float) -> GrooveSample:
        """槽上某个弧长坐标处的**世界系**位置与三标架。"""

        pose = self._pose_at(t_s)
        local = self.centerline.sample_at(arc_mm)
        rotation = pose.rotation_xyzw
        return GrooveSample(
            arc_length_mm=local.arc_length_mm,
            position_mm=_add(
                rotate_by_quaternion(rotation, local.position_mm), pose.translation_mm
            ),
            tangent=rotate_by_quaternion(rotation, local.tangent),
            width_direction=rotate_by_quaternion(rotation, local.width_direction),
            surface_normal=rotate_by_quaternion(rotation, local.surface_normal),
        )

    # -- 速率 --------------------------------------------------------------

    def _rate(self, t_s: float, value_at: Callable[[float], float], wrap: bool) -> float:
        left_time, right_time, denominator = self.rate_probe.stencil(t_s)
        horizon = self.motion.horizon_s()
        for probe in (left_time, right_time):
            if probe < 0.0 or probe > horizon:
                raise LaydownError(
                    f"{self.model_id}: 速率探针要取t_s={probe!r}，"
                    f"落在时间线[0, {horizon!r}]之外——"
                    f"端点上要么换单边格式（scheme='forward'/'backward'），"
                    f"要么把步长{self.rate_probe.step_s!r}调小，"
                    "**本层不夹到端点**：夹过之后的差商不是那个导数"
                )
        left = value_at(left_time)
        right = value_at(right_time)
        difference = (
            self.centerline.arc_difference_mm(right, left) if wrap else right - left
        )
        return difference / denominator

    def required_feed_rate_mm_s(self, t_s: float) -> float:
        """**位姿要的**送带率：``dσ_pose/dt``。闭合曲线上已解卷。"""

        return self._rate(t_s, lambda probe: self.pose_arc_length_mm(probe)[0], True)

    def accounted_feed_rate_mm_s(self, t_s: float) -> float:
        """**送带账给的**送带率：累计送带长度的时间导数。累计量不回绕，故不解卷。"""

        return self._rate(t_s, self.feed.length_at, False)

    # -- 一个时刻的全部 ----------------------------------------------------

    def at(self, t_s: float) -> LaydownPoint:
        t_s = _require_finite(t_s, f"{self.model_id}: t_s")
        feed_arc = self.feed_arc_length_mm(t_s)
        pose_arc, pose_offset = self.pose_arc_length_mm(t_s)
        sample = self.world_sample(t_s, feed_arc)
        tangent, width, normal = (
            sample.tangent,
            sample.width_direction,
            sample.surface_normal,
        )

        direction = self.span.material_increasing_direction()
        along = _dot(direction, tangent)
        across_s = _dot(direction, width)
        across_n = _dot(direction, normal)
        # 总角取``atan2(|横向|, 沿向)``而**不是**``acos(沿向)``。
        # 理由是条件数：理想落位处入射角趋于0，那正是``acos``最病态的地方
        # （``cos θ = 1 − θ²/2``，θ=1e-6时余弦离1只差5e-13，已在双精度分辨率的
        # 五倍以内）。实测``acos``写法在θ≈1.1e-6处与两分量的合成差**4.8e-06相对**，
        # 而``atan2``写法差1e-12量级。**一个理想值在0的量不许用acos去算。**
        transverse_direction = math.hypot(across_s, across_n)
        incidence = math.atan2(transverse_direction, along)
        in_plane = math.atan2(across_s, along)
        out_of_plane = math.asin(min(1.0, max(-1.0, across_n)))

        offset = _sub(sample.position_mm, self.span.entry_point_mm)
        residual_t = _dot(offset, tangent)
        residual_s = _dot(offset, width)
        residual_n = _dot(offset, normal)
        closure = ClosureResidual(
            position_mm=offset,
            magnitude_mm=_norm(offset),
            along_tangent_mm=residual_t,
            across_width_mm=residual_s,
            across_normal_mm=residual_n,
            transverse_mm=math.hypot(residual_s, residual_n),
            arc_gap_mm=self.centerline.arc_difference_mm(feed_arc, pose_arc),
            pose_only_offset_mm=pose_offset,
        )
        return LaydownPoint(
            time_s=t_s,
            feed_arc_length_mm=feed_arc,
            pose_arc_length_mm=pose_arc,
            world_position_mm=sample.position_mm,
            tangent=tangent,
            width_direction=width,
            surface_normal=normal,
            incidence_angle_rad=incidence,
            incidence_in_plane_rad=in_plane,
            incidence_out_of_plane_rad=out_of_plane,
            required_feed_rate_mm_s=self.required_feed_rate_mm_s(t_s),
            accounted_feed_rate_mm_s=self.accounted_feed_rate_mm_s(t_s),
            closure=closure,
        )

    def track(self, times_s: Iterable[float]) -> tuple[LaydownPoint, ...]:
        return tuple(self.at(t_s) for t_s in times_s)

    def is_replayable(self) -> bool:
        """位姿来源可不可重放。**送带账那一半只被证伪过，不在这条里冒充。**

        `FeedAccount`构造期那次双求值只是必要条件（`motion.AnalyticPose`同款），
        所以本方法只转发位姿来源的声明，不把"探针过了"包装成"送带账可重放"。
        """

        return self.motion.is_replayable()


def assert_closure(
    points: Iterable[LaydownPoint], tolerance: ClosureTolerance, *, run_label: str
) -> None:
    """闭合门：残差超过**已声明的**容差就一票否决。

    **这道门只会拒，不会证。** 它挡不住"两条都错但错得一样"的时间线；
    它挡得住的是"送带账与位姿各说各话"这件今天没有任何东西看得见的事。
    写在这里，比让读者以为过了门就等于两条输入都对诚实。
    """

    if not isinstance(tolerance, ClosureTolerance):
        raise LaydownError(f"{run_label}: tolerance必须是ClosureTolerance")
    if not isinstance(run_label, str) or not run_label.strip():
        raise LaydownError("run_label must be a nonempty string")
    seen = 0
    for point in points:
        seen += 1
        residual = point.closure
        if residual.magnitude_mm > tolerance.position_abs_mm:
            raise LaydownError(
                f"{run_label}: t_s={point.time_s!r}处闭合残差"
                f"{residual.magnitude_mm!r} mm超过容差{tolerance.position_abs_mm!r} mm"
                f"（沿槽{residual.along_tangent_mm!r}、横向{residual.transverse_mm!r}）"
                f"——容差理由：{tolerance.reason}"
            )
        if abs(residual.arc_gap_mm) > tolerance.arc_abs_mm:
            raise LaydownError(
                f"{run_label}: t_s={point.time_s!r}处弧长坐标差"
                f"{residual.arc_gap_mm!r} mm超过容差{tolerance.arc_abs_mm!r} mm"
                f"——容差理由：{tolerance.reason}"
            )
    if seen == 0:
        raise LaydownError(
            f"{run_label}: 一个落位点都没有——**零执行绝不pass**"
            "（`tools/accept.py`那条『零执行绝不pass』纪律在这里同样适用）"
        )


__all__ = [
    "ACCEPTED_LENGTH_UNITS",
    "ARC_OUT_OF_RANGE",
    "ARC_OVER_CHORD_CEILING",
    "CENTERLINE_LENGTH_UNIT",
    "CENTERLINE_TOPOLOGIES",
    "FRAME_INTERPOLATIONS",
    "FRAME_ORTHONORMAL_ABS_TOL",
    "POSITION_INTERPOLATIONS",
    "RATE_SCHEMES",
    "ArcRateProbe",
    "CenterlineSemantics",
    "ClosureResidual",
    "ClosureTolerance",
    "FeedAccount",
    "FreeSpanGeometry",
    "GrooveCenterline",
    "GrooveSample",
    "GrooveStation",
    "LaydownError",
    "LaydownModel",
    "LaydownPoint",
    "assert_closure",
    "rotate_by_quaternion",
]
