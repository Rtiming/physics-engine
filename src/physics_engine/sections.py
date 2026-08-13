"""截面积分点与一维弹塑性纤维截面——plans/08阶段4的最小诚实切片。

这一层故意把三个常被混写成“截面自由度”的东西分开：

1. **广义截面变形**：轴向应变``epsilon_0``与曲率``kappa``，共两个运动学坐标；
2. **积分点响应**：``epsilon_i = epsilon_0 + kappa * y_i``，由广义变形导出，
   积分点本身不是全局运动自由度；
3. **材料历史**：每点的塑性应变与累积塑性应变，是真历史，显式放进``State``，
   但不作为截面平衡的独立未知量。

这正是决策0059对plans/06与plans/08文字冲突的裁决：成熟有限元里的截面点是
本构/求积点，不因“有编号、有状态”就变成全局未知量。本模块因此能算矩形截面上
非线性的应力分布、内力与回弹，却**不能**表示截面翘曲、压扁、局部屈曲或电磁场
自由度；它不关闭S1.2，也不把“体积与厚度”整条报成完成。

## 数值与状态边界

* 核心零依赖：矩形截面沿厚度用等面积中点纤维；点号从负``y``到正``y``稳定编号；
* 材料是小应变一维理想弹塑性，径向回归在一维退化为闭式return-map；
* ``evaluate_section_response``是**trial/commit分离**的：它不改传入状态，返回
  ``next_state``。非线性求解可从同一个已提交状态试任意多个候选，不会把失败迭代
  偷写进历史；
* ``solve_section_curvature``用**区间保护的Newton**求``M(kappa)=M_target``：一致切线
  给出的Newton点留在夹根区间内才采用，否则退回二分。理想塑性会出现零切线平台，
  裸Newton在那里没有可靠方向；保护区间让它保持失败形态清楚。区间不夹根即失败关闭，
  不偷偷扩区间。

单位固定为本仓力学侧的mm制：应力与杨氏模量N/mm²，面积mm²，曲率1/mm，
轴力N，弯矩N·mm，弯曲切线``dM/dkappa``为N·mm²。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from physics_engine.canonical import WDS_PROFILE, canonical_sha256
from physics_engine.state import State, StateField, StateLayout


class SectionError(ValueError):
    """截面层的一切失败关闭。"""


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SectionError(f"{name} must be a positive finite number: {value!r}")
    if not math.isfinite(float(value)) or value <= 0.0:
        raise SectionError(f"{name} must be a positive finite number: {value!r}")


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SectionError(f"{name} must be finite: {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise SectionError(f"{name} must be finite: {value!r}")
    return converted


@dataclass(frozen=True)
class LinearElastic1D:
    """小应变一维线弹性材料；没有屈服哨兵，点历史槽保持为零。

    WDS的第一片真实采纳只替换一个既有线弹性easy-axis站点。用一个任意大的
    ``yield_stress``模拟线弹性会把“本工况没有屈服”误写成“材料具有某个未经声明的
    屈服强度”，所以线弹性必须是显式材料语义，而不是弹塑性参数的特殊数值。为与
    截面通用布局兼容，当前状态仍携带每点两个恒零历史槽，但这些槽永不演化。
    """

    young_modulus_n_mm2: float

    def __post_init__(self) -> None:
        _positive_finite("young_modulus_n_mm2", self.young_modulus_n_mm2)


@dataclass(frozen=True)
class ElasticPerfectlyPlastic1D:
    """小应变一维理想弹塑性材料（关联流动、零硬化）。

    这不是``materials.MaterialRecord``的替代品。后者是带证据的多域属性记录；
    本类是力学域的本构上下文，正是spec/14第七节明确留在力学域的那一半。
    """

    young_modulus_n_mm2: float
    yield_stress_n_mm2: float

    def __post_init__(self) -> None:
        _positive_finite("young_modulus_n_mm2", self.young_modulus_n_mm2)
        _positive_finite("yield_stress_n_mm2", self.yield_stress_n_mm2)


@dataclass(frozen=True)
class SectionIntegrationPoint:
    """一个稳定编号的截面求积点；权重已吸收为纤维面积。"""

    index: int
    y_mm: float
    area_mm2: float


@dataclass(frozen=True)
class RectangularFiberSection:
    """沿厚度均匀离散的矩形纤维截面。

    第一片只离散厚度方向；宽度被吸收到每点面积里。这足以让纯弯下的应力分布与
    塑性前沿出现，但表示不了宽度方向翘曲——负空间是API的一部分，不作隐式推广。
    """

    section_id: str
    width_mm: float
    thickness_mm: float
    point_count: int
    integration_rule_id: ClassVar[str] = "section_rule/midpoint_equal_area/1"
    _integration_points: tuple[SectionIntegrationPoint, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.section_id, str) or not self.section_id.startswith("section/"):
            raise SectionError("section_id must be namespaced like 'section/...'")
        _positive_finite("width_mm", self.width_mm)
        _positive_finite("thickness_mm", self.thickness_mm)
        if (
            isinstance(self.point_count, bool)
            or not isinstance(self.point_count, int)
            or self.point_count < 2
        ):
            raise SectionError("point_count must be an integer >= 2")

        depth = float(self.thickness_mm) / self.point_count
        area = float(self.width_mm) * depth
        lower = -0.5 * float(self.thickness_mm)
        points = tuple(
            SectionIntegrationPoint(
                index=index,
                y_mm=lower + (index + 0.5) * depth,
                area_mm2=area,
            )
            for index in range(self.point_count)
        )
        object.__setattr__(self, "_integration_points", points)

    @property
    def integration_points(self) -> tuple[SectionIntegrationPoint, ...]:
        """负``y``到正``y``的稳定点序；这个次序也是历史槽次序。"""

        return self._integration_points

    def reference_fingerprint(self) -> str:
        """几何+求积规则的内容地址；旧历史只能装回同一份参考。"""

        return canonical_sha256(
            {
                "section_id": self.section_id,
                "width_mm": self.width_mm,
                "thickness_mm": self.thickness_mm,
                "point_count": self.point_count,
                "integration_rule_id": self.integration_rule_id,
            },
            WDS_PROFILE,
        )


_AXIAL_FIELD = "section_axial_strain"
_CURVATURE_FIELD = "section_curvature_per_mm"
_PLASTIC_FIELD = "section_point_plastic_strain"
_ACCUMULATED_FIELD = "section_point_accumulated_plastic_strain"


@dataclass(frozen=True)
class RectangularSectionLayout:
    """截面几何与显式状态布局的绑定。

    ``StateLayout.dof_count``历史上把整条状态向量的标量数都叫dof；这里另给
    ``generalized_dof_count``与``history_scalar_count``，防止把“向量有130格”误读成
    “Newton要解130个全局未知量”。64点案例里真实关系是``2 + 2*64``；当前回弹
    入口固定轴向应变，只把曲率作为标量平衡未知量。
    """

    section: RectangularFiberSection
    layout: StateLayout

    def __post_init__(self) -> None:
        expected = (
            (_AXIAL_FIELD, 1, False),
            (_CURVATURE_FIELD, 1, False),
            (_PLASTIC_FIELD, self.section.point_count, True),
            (_ACCUMULATED_FIELD, self.section.point_count, True),
        )
        measured = tuple(
            (field.name, field.width, field.is_history) for field in self.layout.fields
        )
        if measured != expected:
            raise SectionError(
                "section layout fields do not match the rectangular-section state contract"
            )
        reference_suffix = f"/section-{self.section.reference_fingerprint()}"
        if not self.layout.layout_id.endswith(reference_suffix):
            raise SectionError(
                "section layout id does not bind the section geometry and integration rule"
            )

    @property
    def generalized_dof_count(self) -> int:
        return 2

    @property
    def history_scalar_count(self) -> int:
        return 2 * self.section.point_count

    def _point_index(self, point_index: int) -> int:
        if (
            isinstance(point_index, bool)
            or not isinstance(point_index, int)
            or not 0 <= point_index < self.section.point_count
        ):
            raise SectionError(
                f"point index {point_index!r} is outside [0, {self.section.point_count})"
            )
        return point_index

    def plastic_strain_index(self, point_index: int) -> int:
        return self.layout.offset_of(_PLASTIC_FIELD) + self._point_index(point_index)

    def accumulated_plastic_strain_index(self, point_index: int) -> int:
        return self.layout.offset_of(_ACCUMULATED_FIELD) + self._point_index(point_index)

    def initial_state(self, *, axial_strain: float = 0.0, curvature_per_mm: float = 0.0) -> State:
        axial = _finite("axial_strain", axial_strain)
        curvature = _finite("curvature_per_mm", curvature_per_mm)
        return State(
            layout=self.layout,
            vector=(axial, curvature) + (0.0,) * self.history_scalar_count,
        )

    def assert_state(self, state: State) -> None:
        # 不能只比fingerprint：``is_dimensionless``为保持旧指纹不变而不进内容地址。
        # 这里需要的是完整语义相等，宽松到只比指纹会让错误量纲的状态混进来。
        if state.layout != self.layout:
            raise SectionError(
                f"state layout {state.layout.layout_id!r} does not match "
                f"section layout {self.layout.layout_id!r}"
            )


def build_rectangular_section_layout(
    *, layout_id: str, section: RectangularFiberSection
) -> RectangularSectionLayout:
    """给固定积分点清单建立定长布局；点数一变就是另一份布局。"""

    if not isinstance(layout_id, str) or not layout_id.startswith("layout/"):
        raise SectionError("layout_id must be namespaced like 'layout/...'")
    effective_layout_id = f"{layout_id}/section-{section.reference_fingerprint()}"
    layout = StateLayout(
        layout_id=effective_layout_id,
        fields=(
            StateField(_AXIAL_FIELD, 1, is_dimensionless=True),
            StateField(_CURVATURE_FIELD, 1),
            StateField(
                _PLASTIC_FIELD,
                section.point_count,
                is_history=True,
                is_dimensionless=True,
            ),
            StateField(
                _ACCUMULATED_FIELD,
                section.point_count,
                is_history=True,
                is_dimensionless=True,
            ),
        ),
    )
    return RectangularSectionLayout(section=section, layout=layout)


@dataclass(frozen=True)
class SectionPointResponse:
    """一个积分点在本次trial后的可审计响应。"""

    point_index: int
    y_mm: float
    area_mm2: float
    strain: float
    stress_n_mm2: float
    plastic_strain: float
    accumulated_plastic_strain: float
    tangent_n_mm2: float
    yielded: bool


@dataclass(frozen=True)
class SectionResponse:
    """截面增量势、内力、算法一致切线、点分布与可提交的新状态。

    ``incremental_potential_n``是**每单位梁长**的增量势：应力对应的一维材料势
    （N/mm²）乘纤维面积（mm²）后求和，单位为N。外层梁站点再乘对偶长度mm，
    得到能进``EnergyRegistry``的N·mm。它的曲率一阶导恰是
    ``bending_moment_n_mm``，二阶导恰是``bending_tangent_n_mm2``。
    """

    incremental_potential_n: float
    axial_force_n: float
    bending_moment_n_mm: float
    axial_tangent_n: float
    axial_curvature_coupling_n_mm: float
    bending_tangent_n_mm2: float
    points: tuple[SectionPointResponse, ...]
    next_state: State


def _material_trial(
    material: LinearElastic1D | ElasticPerfectlyPlastic1D,
    *,
    strain: float,
    previous_plastic_strain: float,
    previous_accumulated_plastic_strain: float,
) -> tuple[float, float, float, float, bool]:
    """一维闭式return-map → ``stress, tangent, ep, accumulated, yielded``。"""

    if previous_accumulated_plastic_strain < 0.0:
        raise SectionError("accumulated plastic strain must be nonnegative")
    if previous_accumulated_plastic_strain + 1.0e-15 < abs(previous_plastic_strain):
        raise SectionError("accumulated plastic strain cannot be smaller than |plastic strain|")

    young = float(material.young_modulus_n_mm2)
    if isinstance(material, LinearElastic1D):
        if previous_plastic_strain != 0.0 or previous_accumulated_plastic_strain != 0.0:
            raise SectionError("linear elastic material cannot consume nonzero plastic history")
        stress = young * strain
        if not math.isfinite(stress):
            raise SectionError("material trial stress overflowed to NaN/Inf")
        return stress, young, 0.0, 0.0, False

    yield_stress = float(material.yield_stress_n_mm2)
    trial = young * (strain - previous_plastic_strain)
    if not math.isfinite(trial):
        raise SectionError("material trial stress overflowed to NaN/Inf")
    excess = abs(trial) - yield_stress
    if excess <= 0.0:
        return (
            trial,
            young,
            previous_plastic_strain,
            previous_accumulated_plastic_strain,
            False,
        )

    direction = math.copysign(1.0, trial)
    increment = excess / young
    return (
        direction * yield_stress,
        0.0,
        previous_plastic_strain + direction * increment,
        previous_accumulated_plastic_strain + increment,
        True,
    )


def _material_incremental_potential(
    material: LinearElastic1D | ElasticPerfectlyPlastic1D,
    *,
    strain: float,
    previous_plastic_strain: float,
) -> float:
    """理想弹塑性的闭式增量势；对应``_material_trial``的同一条应力曲线。

    这是
    ``min_ep 0.5*E*(epsilon-ep)^2 + sigma_y*|ep-ep_old|``
    消去``ep``后的标量函数。它在屈服点连续且一阶连续，导数是clip后的应力；
    塑性耗散已经包含在增量项里，不能只拿弹性储能给Newton线搜索。
    """

    young = float(material.young_modulus_n_mm2)
    if isinstance(material, LinearElastic1D):
        if previous_plastic_strain != 0.0:
            raise SectionError("linear elastic material cannot consume nonzero plastic history")
        return 0.5 * young * strain * strain

    yield_stress = float(material.yield_stress_n_mm2)
    elastic_trial_strain = strain - previous_plastic_strain
    yield_strain = yield_stress / young
    magnitude = abs(elastic_trial_strain)
    if magnitude <= yield_strain:
        return 0.5 * young * elastic_trial_strain * elastic_trial_strain
    return yield_stress * magnitude - 0.5 * yield_stress * yield_strain


def evaluate_section_response(
    *,
    section_layout: RectangularSectionLayout,
    material: LinearElastic1D | ElasticPerfectlyPlastic1D,
    previous_state: State,
    axial_strain: float,
    curvature_per_mm: float,
) -> SectionResponse:
    """从一个已提交状态求候选广义变形下的截面trial响应。

    调用方只在外层平衡迭代收敛后提交``next_state``。本函数从不修改
    ``previous_state``；因此二分/Newton里的失败候选不会污染塑性历史。
    """

    section_layout.assert_state(previous_state)
    axial = _finite("axial_strain", axial_strain)
    curvature = _finite("curvature_per_mm", curvature_per_mm)
    previous_plastic = previous_state.block(_PLASTIC_FIELD)
    previous_accumulated = previous_state.block(_ACCUMULATED_FIELD)

    points: list[SectionPointResponse] = []
    plastic: list[float] = []
    accumulated: list[float] = []
    axial_terms: list[float] = []
    moment_terms: list[float] = []
    axial_tangent_terms: list[float] = []
    coupling_terms: list[float] = []
    bending_tangent_terms: list[float] = []
    potential_terms: list[float] = []

    for point in section_layout.section.integration_points:
        strain = axial + curvature * point.y_mm
        stress, tangent, next_plastic, next_accumulated, yielded = _material_trial(
            material,
            strain=strain,
            previous_plastic_strain=previous_plastic[point.index],
            previous_accumulated_plastic_strain=previous_accumulated[point.index],
        )
        points.append(
            SectionPointResponse(
                point_index=point.index,
                y_mm=point.y_mm,
                area_mm2=point.area_mm2,
                strain=strain,
                stress_n_mm2=stress,
                plastic_strain=next_plastic,
                accumulated_plastic_strain=next_accumulated,
                tangent_n_mm2=tangent,
                yielded=yielded,
            )
        )
        plastic.append(next_plastic)
        accumulated.append(next_accumulated)
        axial_terms.append(stress * point.area_mm2)
        moment_terms.append(stress * point.area_mm2 * point.y_mm)
        axial_tangent_terms.append(tangent * point.area_mm2)
        coupling_terms.append(tangent * point.area_mm2 * point.y_mm)
        bending_tangent_terms.append(tangent * point.area_mm2 * point.y_mm**2)
        potential_terms.append(
            _material_incremental_potential(
                material,
                strain=strain,
                previous_plastic_strain=previous_plastic[point.index],
            )
            * point.area_mm2
        )

    next_state = State(
        layout=section_layout.layout,
        vector=(axial, curvature, *plastic, *accumulated),
    )
    return SectionResponse(
        incremental_potential_n=math.fsum(potential_terms),
        axial_force_n=math.fsum(axial_terms),
        bending_moment_n_mm=math.fsum(moment_terms),
        axial_tangent_n=math.fsum(axial_tangent_terms),
        axial_curvature_coupling_n_mm=math.fsum(coupling_terms),
        bending_tangent_n_mm2=math.fsum(bending_tangent_terms),
        points=tuple(points),
        next_state=next_state,
    )


@dataclass(frozen=True)
class SectionSolveResult:
    """局部截面平衡结果；不收敛是可读结果，不伪装成异常通过。"""

    response: SectionResponse
    curvature_per_mm: float
    converged: bool
    iterations: int
    residual_n_mm: float
    reason: str = ""


def solve_section_curvature(
    *,
    section_layout: RectangularSectionLayout,
    material: LinearElastic1D | ElasticPerfectlyPlastic1D,
    previous_state: State,
    axial_strain: float,
    target_moment_n_mm: float,
    curvature_bracket_per_mm: tuple[float, float],
    residual_tol_n_mm: float,
    max_iterations: int = 80,
) -> SectionSolveResult:
    """在显式区间内用受保护Newton求``M(kappa) - M_target = 0``。

    本函数只解纯弯方向的一维局部平衡；轴向应变由调用方给定。完整的``N/M``二维
    截面平衡、与全局梁节点装配、载荷步自适应都不在第一片里。
    """

    section_layout.assert_state(previous_state)
    axial = _finite("axial_strain", axial_strain)
    target = _finite("target_moment_n_mm", target_moment_n_mm)
    _positive_finite("residual_tol_n_mm", residual_tol_n_mm)
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise SectionError("max_iterations must be a positive integer")
    try:
        raw_low, raw_high = curvature_bracket_per_mm
    except (TypeError, ValueError) as error:
        raise SectionError("curvature_bracket_per_mm must contain exactly two values") from error
    low = _finite("curvature_bracket_per_mm[0]", raw_low)
    high = _finite("curvature_bracket_per_mm[1]", raw_high)
    if not low < high:
        raise SectionError("curvature bracket must be strictly increasing")

    def at(curvature: float) -> tuple[float, SectionResponse]:
        response = evaluate_section_response(
            section_layout=section_layout,
            material=material,
            previous_state=previous_state,
            axial_strain=axial,
            curvature_per_mm=curvature,
        )
        return response.bending_moment_n_mm - target, response

    residual_low, response_low = at(low)
    if abs(residual_low) <= residual_tol_n_mm:
        return SectionSolveResult(response_low, low, True, 0, residual_low)
    residual_high, response_high = at(high)
    if abs(residual_high) <= residual_tol_n_mm:
        return SectionSolveResult(response_high, high, True, 0, residual_high)
    if (residual_low < 0.0) == (residual_high < 0.0):
        raise SectionError(
            "curvature bracket does not bracket the target moment: "
            f"residuals are {residual_low!r} and {residual_high!r} N·mm"
        )

    candidate = low + 0.5 * (high - low)
    evaluated_curvature = candidate
    residual_candidate = residual_low
    response_candidate = response_low
    for iteration in range(1, max_iterations + 1):
        if candidate == low or candidate == high:
            return SectionSolveResult(
                response_candidate,
                candidate,
                False,
                iteration - 1,
                residual_candidate,
                "曲率区间已缩到float64无法再二分，但弯矩残差仍高于声明容差",
            )
        evaluated_curvature = candidate
        residual_candidate, response_candidate = at(candidate)
        if abs(residual_candidate) <= residual_tol_n_mm:
            return SectionSolveResult(
                response_candidate,
                candidate,
                True,
                iteration,
                residual_candidate,
            )
        if (residual_low < 0.0) == (residual_candidate < 0.0):
            low = candidate
            residual_low = residual_candidate
        else:
            high = candidate
            residual_high = residual_candidate

        # 一致切线给出候选；只要它离开夹根区间、为零或非有限，就退回二分。
        # 这是“Newton负责快、区间负责不撒谎”，不是两个独立求解器。
        tangent = response_candidate.bending_tangent_n_mm2
        newton = (
            candidate - residual_candidate / tangent
            if tangent > 0.0 and math.isfinite(tangent)
            else math.nan
        )
        candidate = newton if low < newton < high else low + 0.5 * (high - low)

    return SectionSolveResult(
        response_candidate,
        evaluated_curvature,
        False,
        max_iterations,
        residual_candidate,
        f"达到最大迭代次数{max_iterations}仍未把弯矩残差压到{residual_tol_n_mm!r} N·mm",
    )


__all__ = [
    "ElasticPerfectlyPlastic1D",
    "LinearElastic1D",
    "RectangularFiberSection",
    "RectangularSectionLayout",
    "SectionError",
    "SectionIntegrationPoint",
    "SectionPointResponse",
    "SectionResponse",
    "SectionSolveResult",
    "build_rectangular_section_layout",
    "evaluate_section_response",
    "solve_section_curvature",
]
