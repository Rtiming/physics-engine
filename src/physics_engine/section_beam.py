"""WDS运动学一致的三节点Kirchhoff截面弯曲装配——阶段4的全局接缝。

本模块把 :mod:`physics_engine.sections` 的局部纤维本构接进第一条真实全局路径：

``3个节点位置 + 2条边的材料扭角``
    → WDS/Bergou曲率副法线
    → easy-axis物理曲率
    → 逐纤维trial本构
    → 截面弯矩/一致切线
    → 全局势能、残差与Hessian。

边界刻意很窄：只有一个三节点顶点、固定轴向应变、easy-axis弯曲。没有轴向``N``
装配、hard-axis弯曲、扭转截面本构、多站历史投影或WDS消费仓迁移。积分点历史放在
全局``State``尾部以便复现，但受保护求解入口会自动固定这些槽；Newton只解11个
运动学自由度中的调用方未固定部分，历史在收敛后才提交。

曲率的一阶与二阶导数由零依赖二阶jet作解析链式法则，不在生产路径用有限差分。
这与WDS当前二阶AD走的是同一条数学，但实现独立；WDS只读夹具和中心差分门分别验证
运动学兼容与二阶导正确性。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from physics_engine.canonical import WDS_PROFILE, canonical_sha256
from physics_engine.energies import (
    POTENTIAL,
    EnergyContext,
    EnergyRegistry,
    EnergyTerm,
    Matrix,
    Vector,
)
from physics_engine.sections import (
    ElasticPerfectlyPlastic1D,
    RectangularFiberSection,
    RectangularSectionLayout,
    SectionResponse,
    build_rectangular_section_layout,
    evaluate_section_response,
)
from physics_engine.solve import SolveResult, solve_equilibrium
from physics_engine.state import State, StateField, StateLayout


class KirchhoffSectionError(ValueError):
    """Kirchhoff截面接缝的一切失败关闭。"""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KirchhoffSectionError(f"{name} must be finite: {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise KirchhoffSectionError(f"{name} must be finite: {value!r}")
    return converted


def _positive_finite(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise KirchhoffSectionError(f"{name} must be positive: {value!r}")
    return converted


def _vector3(name: str, value) -> tuple[float, float, float]:
    try:
        raw = tuple(value)
    except TypeError as error:
        raise KirchhoffSectionError(f"{name} must be a finite 3-vector") from error
    if len(raw) != 3:
        raise KirchhoffSectionError(f"{name} must be a finite 3-vector")
    return tuple(_finite(f"{name}[{index}]", component) for index, component in enumerate(raw))


@dataclass(frozen=True)
class KirchhoffSectionReference:
    """一个WDS式内顶点的冻结参考量。

    ``natural_kappa1``与WDS的``RodReference.natural_kappa1``同为**离散曲率**，
    是无量纲量；物理自然曲率要除以``dual_length_mm``。
    """

    rest_lengths_mm: tuple[float, float]
    reference_d1: tuple[tuple[float, float, float], tuple[float, float, float]]
    reference_d2: tuple[tuple[float, float, float], tuple[float, float, float]]
    natural_kappa1: float = 0.0

    def __post_init__(self) -> None:
        try:
            rest = tuple(self.rest_lengths_mm)
            d1 = tuple(self.reference_d1)
            d2 = tuple(self.reference_d2)
        except TypeError as error:
            raise KirchhoffSectionError("reference needs exactly two edges") from error
        if len(rest) != 2 or len(d1) != 2 or len(d2) != 2:
            raise KirchhoffSectionError("reference needs exactly two edges")
        rest_values = tuple(
            _positive_finite(f"rest_lengths_mm[{index}]", value)
            for index, value in enumerate(rest)
        )
        d1_values = tuple(_vector3(f"reference_d1[{index}]", value) for index, value in enumerate(d1))
        d2_values = tuple(_vector3(f"reference_d2[{index}]", value) for index, value in enumerate(d2))
        for edge, (first, second) in enumerate(zip(d1_values, d2_values, strict=True)):
            first_norm = math.sqrt(sum(value * value for value in first))
            second_norm = math.sqrt(sum(value * value for value in second))
            dot = sum(a * b for a, b in zip(first, second, strict=True))
            if not (
                math.isclose(first_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10)
                and math.isclose(second_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10)
                and abs(dot) <= 1.0e-10
            ):
                raise KirchhoffSectionError(
                    f"reference directors for edge {edge} must be orthonormal"
                )
        object.__setattr__(self, "rest_lengths_mm", rest_values)
        object.__setattr__(self, "reference_d1", d1_values)
        object.__setattr__(self, "reference_d2", d2_values)
        object.__setattr__(self, "natural_kappa1", _finite("natural_kappa1", self.natural_kappa1))

    @property
    def dual_length_mm(self) -> float:
        return 0.5 * (self.rest_lengths_mm[0] + self.rest_lengths_mm[1])

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "rest_lengths_mm": list(self.rest_lengths_mm),
                "reference_d1": [list(vector) for vector in self.reference_d1],
                "reference_d2": [list(vector) for vector in self.reference_d2],
                "natural_kappa1": self.natural_kappa1,
                "kinematics_id": "wds/bergou-kappa1/1",
            },
            WDS_PROFILE,
        )


_POSITIONS_FIELD = "node_positions_mm"
_TWIST_FIELD = "edge_twist_angles"
_PLASTIC_FIELD = "section_point_plastic_strain"
_ACCUMULATED_FIELD = "section_point_accumulated_plastic_strain"


@dataclass(frozen=True)
class KirchhoffSectionVertexLayout:
    """三节点顶点、两条边扭角与一个截面站点历史的显式打包契约。"""

    section: RectangularFiberSection
    reference: KirchhoffSectionReference
    layout: StateLayout
    section_layout: RectangularSectionLayout

    def __post_init__(self) -> None:
        expected = (
            (_POSITIONS_FIELD, 9, False),
            (_TWIST_FIELD, 2, False),
            (_PLASTIC_FIELD, self.section.point_count, True),
            (_ACCUMULATED_FIELD, self.section.point_count, True),
        )
        measured = tuple(
            (field.name, field.width, field.is_history) for field in self.layout.fields
        )
        if measured != expected or self.layout.node_dof_count != 9:
            raise KirchhoffSectionError(
                "layout does not match the three-node Kirchhoff section contract"
            )
        suffix = (
            f"/section-{self.section.reference_fingerprint()}"
            f"/reference-{self.reference.fingerprint()}"
        )
        if not self.layout.layout_id.endswith(suffix):
            raise KirchhoffSectionError("layout id does not bind section and vertex reference")
        if self.section_layout.section != self.section:
            raise KirchhoffSectionError("local section layout does not bind the same section")

    @property
    def kinematic_dof_count(self) -> int:
        return 11

    @property
    def history_scalar_count(self) -> int:
        return 2 * self.section.point_count

    @property
    def history_indices(self) -> frozenset[int]:
        return frozenset(range(self.kinematic_dof_count, self.layout.dof_count))

    def assert_state(self, state: State) -> None:
        if state.layout != self.layout:
            raise KirchhoffSectionError(
                f"state layout {state.layout.layout_id!r} does not match "
                f"Kirchhoff layout {self.layout.layout_id!r}"
            )

    def initial_state(
        self,
        *,
        positions_mm,
        edge_twist_angles,
    ) -> State:
        try:
            raw_positions = tuple(positions_mm)
            raw_twist = tuple(edge_twist_angles)
        except TypeError as error:
            raise KirchhoffSectionError("initial state needs three positions and two twists") from error
        if len(raw_positions) != 3 or len(raw_twist) != 2:
            raise KirchhoffSectionError("initial state needs three positions and two twists")
        positions = tuple(
            component
            for node, raw in enumerate(raw_positions)
            for component in _vector3(f"positions_mm[{node}]", raw)
        )
        twists = tuple(
            _finite(f"edge_twist_angles[{index}]", value)
            for index, value in enumerate(raw_twist)
        )
        return State(
            layout=self.layout,
            vector=positions + twists + (0.0,) * self.history_scalar_count,
        )

    def local_committed_section_state(self, state: State) -> State:
        """把全局尾部真历史投影到既有单站截面布局；运动学量只作占位。"""

        self.assert_state(state)
        return State(
            layout=self.section_layout.layout,
            vector=(
                0.0,
                0.0,
                *state.block(_PLASTIC_FIELD),
                *state.block(_ACCUMULATED_FIELD),
            ),
        )

    def with_section_history(self, state: State, section_state: State) -> State:
        self.assert_state(state)
        self.section_layout.assert_state(section_state)
        vector = list(state.vector)
        plastic_offset = self.layout.offset_of(_PLASTIC_FIELD)
        accumulated_offset = self.layout.offset_of(_ACCUMULATED_FIELD)
        plastic = section_state.block(_PLASTIC_FIELD)
        accumulated = section_state.block(_ACCUMULATED_FIELD)
        vector[plastic_offset : plastic_offset + len(plastic)] = plastic
        vector[accumulated_offset : accumulated_offset + len(accumulated)] = accumulated
        return state.with_vector(tuple(vector))


def build_kirchhoff_section_vertex_layout(
    *,
    layout_id: str,
    section: RectangularFiberSection,
    reference: KirchhoffSectionReference,
) -> KirchhoffSectionVertexLayout:
    """建立第一片固定形制：3节点、2边扭角、1个截面站点。"""

    if not isinstance(layout_id, str) or not layout_id.startswith("layout/"):
        raise KirchhoffSectionError("layout_id must be namespaced like 'layout/...'")
    effective_id = (
        f"{layout_id}/section-{section.reference_fingerprint()}"
        f"/reference-{reference.fingerprint()}"
    )
    layout = StateLayout(
        layout_id=effective_id,
        fields=(
            StateField(_POSITIONS_FIELD, 9),
            StateField(_TWIST_FIELD, 2, is_dimensionless=True),
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
        node_dof_count=9,
    )
    local = build_rectangular_section_layout(
        layout_id=f"{layout_id}/local-section",
        section=section,
    )
    return KirchhoffSectionVertexLayout(section, reference, layout, local)


@dataclass(frozen=True)
class _Jet1:
    """只携一阶导的轻量jet；残差装配不为未请求的121个二阶量付费。"""

    value: float
    gradient: tuple[float, ...]

    @property
    def size(self) -> int:
        return len(self.gradient)

    @classmethod
    def constant(cls, value: float, size: int) -> _Jet1:
        return cls(float(value), (0.0,) * size)

    @classmethod
    def variable(cls, value: float, index: int, size: int) -> _Jet1:
        gradient = [0.0] * size
        gradient[index] = 1.0
        return cls(float(value), tuple(gradient))

    def _coerce(self, other) -> _Jet1:
        if isinstance(other, _Jet1):
            if other.size != self.size:
                raise KirchhoffSectionError("first-order jets have inconsistent widths")
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return _Jet1.constant(float(other), self.size)
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return _Jet1(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.gradient, other.gradient, strict=True)),
        )

    __radd__ = __add__

    def __neg__(self):
        return _Jet1(-self.value, tuple(-value for value in self.gradient))

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return _Jet1(
            self.value * other.value,
            tuple(
                self.gradient[index] * other.value
                + self.value * other.gradient[index]
                for index in range(self.size)
            ),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> _Jet1:
        if self.value == 0.0:
            raise KirchhoffSectionError("division by zero in Kirchhoff kinematics")
        inverse = 1.0 / self.value
        return _Jet1(
            inverse,
            tuple(-inverse * inverse * component for component in self.gradient),
        )

    def __truediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        inverse = other.reciprocal()
        # 导数用乘倒数展开，但值通道保留Python直接除法的舍入；这样0/1/2阶
        # 入口在屈服分支上逐位选择同一个标量曲率。
        return _Jet1(
            self.value / other.value,
            tuple(
                self.gradient[index] * inverse.value
                + self.value * inverse.gradient[index]
                for index in range(self.size)
            ),
        )

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other / self


@dataclass(frozen=True)
class _Jet2:
    value: float
    gradient: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]

    @property
    def size(self) -> int:
        return len(self.gradient)

    @classmethod
    def constant(cls, value: float, size: int) -> _Jet2:
        return cls(float(value), (0.0,) * size, tuple((0.0,) * size for _ in range(size)))

    @classmethod
    def variable(cls, value: float, index: int, size: int) -> _Jet2:
        gradient = [0.0] * size
        gradient[index] = 1.0
        return cls(float(value), tuple(gradient), tuple((0.0,) * size for _ in range(size)))

    def _coerce(self, other) -> _Jet2:
        if isinstance(other, _Jet2):
            if other.size != self.size:
                raise KirchhoffSectionError("second-order jets have inconsistent widths")
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return _Jet2.constant(float(other), self.size)
        return NotImplemented

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return _Jet2(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.gradient, other.gradient, strict=True)),
            tuple(
                tuple(a + b for a, b in zip(left, right, strict=True))
                for left, right in zip(self.hessian, other.hessian, strict=True)
            ),
        )

    __radd__ = __add__

    def __neg__(self):
        return _Jet2(
            -self.value,
            tuple(-value for value in self.gradient),
            tuple(tuple(-value for value in row) for row in self.hessian),
        )

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        gradient = tuple(
            self.gradient[index] * other.value + self.value * other.gradient[index]
            for index in range(self.size)
        )
        hessian = tuple(
            tuple(
                self.hessian[row][column] * other.value
                + self.gradient[row] * other.gradient[column]
                + other.gradient[row] * self.gradient[column]
                + self.value * other.hessian[row][column]
                for column in range(self.size)
            )
            for row in range(self.size)
        )
        return _Jet2(self.value * other.value, gradient, hessian)

    __rmul__ = __mul__

    def reciprocal(self) -> _Jet2:
        if self.value == 0.0:
            raise KirchhoffSectionError("division by zero in Kirchhoff kinematics")
        inverse = 1.0 / self.value
        first = -inverse * inverse
        second = 2.0 * inverse * inverse * inverse
        return _jet_unary(self, inverse, first, second)

    def __truediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        inverse = other.reciprocal()
        gradient = tuple(
            self.gradient[index] * inverse.value
            + self.value * inverse.gradient[index]
            for index in range(self.size)
        )
        hessian = tuple(
            tuple(
                self.hessian[row][column] * inverse.value
                + self.gradient[row] * inverse.gradient[column]
                + inverse.gradient[row] * self.gradient[column]
                + self.value * inverse.hessian[row][column]
                for column in range(self.size)
            )
            for row in range(self.size)
        )
        return _Jet2(self.value / other.value, gradient, hessian)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other / self


def _jet_unary(value: _Jet2, result: float, first: float, second: float) -> _Jet2:
    return _Jet2(
        result,
        tuple(first * component for component in value.gradient),
        tuple(
            tuple(
                second * value.gradient[row] * value.gradient[column]
                + first * value.hessian[row][column]
                for column in range(value.size)
            )
            for row in range(value.size)
        ),
    )


def _ad_sqrt(value):
    raw = value.value if isinstance(value, (_Jet1, _Jet2)) else value
    if raw <= 0.0:
        raise KirchhoffSectionError("Kirchhoff edge has zero length")
    result = math.sqrt(raw)
    if isinstance(value, _Jet2):
        return _jet_unary(value, result, 0.5 / result, -0.25 / (raw * result))
    if isinstance(value, _Jet1):
        return _Jet1(
            result,
            tuple(0.5 / result * component for component in value.gradient),
        )
    return result


def _ad_sin(value):
    raw = value.value if isinstance(value, (_Jet1, _Jet2)) else value
    result = math.sin(raw)
    if isinstance(value, _Jet2):
        return _jet_unary(value, result, math.cos(raw), -result)
    if isinstance(value, _Jet1):
        return _Jet1(
            result,
            tuple(math.cos(raw) * component for component in value.gradient),
        )
    return result


def _ad_cos(value):
    raw = value.value if isinstance(value, (_Jet1, _Jet2)) else value
    result = math.cos(raw)
    if isinstance(value, _Jet2):
        return _jet_unary(value, result, -math.sin(raw), -result)
    if isinstance(value, _Jet1):
        return _Jet1(
            result,
            tuple(-math.sin(raw) * component for component in value.gradient),
        )
    return result


def _ad_dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def _ad_cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _ad_norm(vector):
    return _ad_sqrt(_ad_dot(vector, vector))


@dataclass(frozen=True)
class KirchhoffVertexKinematics:
    curvature_per_mm: float
    gradient: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class KirchhoffSectionTrial:
    curvature_per_mm: float
    section_response: SectionResponse
    next_state: State


@dataclass(frozen=True)
class KirchhoffFiberSectionBending:
    """一个easy-axis纤维截面站点的全局势能项。"""

    vertex_layout: KirchhoffSectionVertexLayout
    material: ElasticPerfectlyPlastic1D
    committed_state: State
    fixed_axial_strain: float = 0.0
    name: str = "kirchhoff_fiber_section_bending"
    kind: ClassVar[Literal["potential"]] = POTENTIAL
    #: 增量势包含本步塑性耗散，只供准静态平衡；不是瞬态可恢复能量账。
    supports_dynamics: ClassVar[bool] = False
    energy_interpretation: ClassVar[str] = "incremental_potential"

    def __post_init__(self) -> None:
        self.vertex_layout.assert_state(self.committed_state)
        _finite("fixed_axial_strain", self.fixed_axial_strain)
        if not isinstance(self.name, str) or not self.name:
            raise KirchhoffSectionError("section energy term name must be nonempty")

    @property
    def supported_generalized_strains(self) -> tuple[str, ...]:
        return ("easy_axis_curvature",)

    @property
    def unsupported_axes(self) -> tuple[str, ...]:
        return ("axial", "hard_axis", "twist")

    def node_index_bound(self) -> int:
        return 3

    def _curvature(self, state: State, *, order: int):
        self.vertex_layout.assert_state(state)
        for field in self.vertex_layout.layout.history_fields():
            if state.block(field) != self.committed_state.block(field):
                raise KirchhoffSectionError(
                    f"trial state field {field!r} differs from the committed history — "
                    "材料历史只可在全局平衡收敛后提交；不能把候选历史静默忽略"
                )
        if order == 0:
            local = state.vector[:11]
        elif order == 1:
            local = tuple(
                _Jet1.variable(state.vector[index], index, 11) for index in range(11)
            )
        elif order == 2:
            local = tuple(
                _Jet2.variable(state.vector[index], index, 11) for index in range(11)
            )
        else:
            raise KirchhoffSectionError(f"derivative order must be 0, 1 or 2: {order!r}")
        p0, p1, p2 = local[0:3], local[3:6], local[6:9]
        edge0 = tuple(p1[axis] - p0[axis] for axis in range(3))
        edge1 = tuple(p2[axis] - p1[axis] for axis in range(3))
        length0 = _ad_norm(edge0)
        length1 = _ad_norm(edge1)
        denominator = length0 * length1 + _ad_dot(edge0, edge1)
        denominator_value = (
            denominator.value if isinstance(denominator, (_Jet1, _Jet2)) else denominator
        )
        if abs(denominator_value) < 1.0e-12:
            raise KirchhoffSectionError(
                "adjacent Kirchhoff edges are near antiparallel; curvature is singular"
            )
        binormal = tuple(
            2.0 * component / denominator for component in _ad_cross(edge0, edge1)
        )
        gamma_left, gamma_right = local[9], local[10]
        cos_left, sin_left = _ad_cos(gamma_left), _ad_sin(gamma_left)
        cos_right, sin_right = _ad_cos(gamma_right), _ad_sin(gamma_right)
        reference = self.vertex_layout.reference
        m2_left = tuple(
            -sin_left * reference.reference_d1[0][axis]
            + cos_left * reference.reference_d2[0][axis]
            for axis in range(3)
        )
        m2_right = tuple(
            -sin_right * reference.reference_d1[1][axis]
            + cos_right * reference.reference_d2[1][axis]
            for axis in range(3)
        )
        kappa1_discrete = 0.5 * _ad_dot(
            tuple(m2_left[axis] + m2_right[axis] for axis in range(3)),
            binormal,
        )
        return (
            kappa1_discrete - reference.natural_kappa1
        ) / reference.dual_length_mm

    def kinematics(self, state: State) -> KirchhoffVertexKinematics:
        curvature = self._curvature(state, order=2)
        assert isinstance(curvature, _Jet2)
        return KirchhoffVertexKinematics(
            curvature.value,
            curvature.gradient,
            curvature.hessian,
        )

    def _section_response(self, curvature_per_mm: float) -> SectionResponse:
        previous = self.vertex_layout.local_committed_section_state(self.committed_state)
        return evaluate_section_response(
            section_layout=self.vertex_layout.section_layout,
            material=self.material,
            previous_state=previous,
            axial_strain=float(self.fixed_axial_strain),
            curvature_per_mm=curvature_per_mm,
        )

    def trial_response(self, state: State) -> KirchhoffSectionTrial:
        curvature = self._curvature(state, order=0)
        assert isinstance(curvature, float)
        section_response = self._section_response(curvature)
        next_state = self.vertex_layout.with_section_history(
            state,
            section_response.next_state,
        )
        return KirchhoffSectionTrial(curvature, section_response, next_state)

    def _quantities(self, state: State, *, need_gradient: bool, need_hessian: bool):
        order = 2 if need_hessian else (1 if need_gradient else 0)
        curvature = self._curvature(state, order=order)
        curvature_value = (
            curvature.value if isinstance(curvature, (_Jet1, _Jet2)) else curvature
        )
        assert isinstance(curvature_value, float)
        response = self._section_response(curvature_value)
        length = self.vertex_layout.reference.dual_length_mm
        energy = length * response.incremental_potential_n
        gradient = None
        hessian = None
        if need_gradient:
            assert isinstance(curvature, (_Jet1, _Jet2))
            result = [0.0] * len(state.vector)
            moment = response.bending_moment_n_mm
            for index in range(11):
                result[index] = length * moment * curvature.gradient[index]
            gradient = tuple(result)
        if need_hessian:
            assert isinstance(curvature, _Jet2)
            result = [[0.0] * len(state.vector) for _ in state.vector]
            moment = response.bending_moment_n_mm
            tangent = response.bending_tangent_n_mm2
            for row in range(11):
                for column in range(11):
                    result[row][column] = length * (
                        tangent * curvature.gradient[row] * curvature.gradient[column]
                        + moment * curvature.hessian[row][column]
                    )
            hessian = tuple(tuple(row) for row in result)
        return energy, gradient, hessian

    def energy(self, state: State, context: EnergyContext) -> float:
        return self._quantities(state, need_gradient=False, need_hessian=False)[0]

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        gradient = self._quantities(state, need_gradient=True, need_hessian=False)[1]
        assert gradient is not None
        return gradient

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        hessian = self._quantities(state, need_gradient=False, need_hessian=True)[2]
        assert hessian is not None
        return hessian

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        curvature = self._curvature(state, order=2)
        assert isinstance(curvature, _Jet2)
        response = self._section_response(curvature.value)
        length = self.vertex_layout.reference.dual_length_mm
        moment = response.bending_moment_n_mm
        tangent = response.bending_tangent_n_mm2
        return tuple(
            (
                row,
                column,
                length
                * (
                    tangent * curvature.gradient[row] * curvature.gradient[column]
                    + moment * curvature.hessian[row][column]
                ),
            )
            for row in range(11)
            for column in range(11)
        )

    def quantities(
        self,
        state: State,
        context: EnergyContext,
        *,
        need_gradient: bool,
        need_hessian: bool,
    ) -> tuple[float, Vector | None, Matrix | None]:
        return self._quantities(
            state,
            need_gradient=need_gradient,
            need_hessian=need_hessian,
        )


@dataclass(frozen=True)
class KirchhoffSectionEquilibriumResult:
    equilibrium: SolveResult
    trial: KirchhoffSectionTrial
    committed_state: State
    history_committed: bool


def solve_kirchhoff_section_equilibrium(
    *,
    section_term: KirchhoffFiberSectionBending,
    context: EnergyContext,
    additional_terms: tuple[EnergyTerm, ...] = (),
    fixed_indices: frozenset[int] = frozenset(),
    residual_tol_n: float,
    max_iterations: int = 50,
    max_backtracks: int = 40,
) -> KirchhoffSectionEquilibriumResult:
    """受保护的全局平衡入口：历史槽自动固定，且只在收敛后提交。

    不收敛时``equilibrium.state``保留最后trial构型供诊断，而``committed_state``逐字节
    返回输入已提交态；调用方不能误把失败Newton产生的塑性点状态当成已提交结果。
    """

    layout = section_term.vertex_layout
    protected = frozenset(fixed_indices) | layout.history_indices
    registry = EnergyRegistry(terms=(section_term, *additional_terms))
    equilibrium = solve_equilibrium(
        registry,
        context,
        layout.layout,
        section_term.committed_state.vector,
        fixed_indices=protected,
        residual_tol_n=residual_tol_n,
        max_iterations=max_iterations,
        max_backtracks=max_backtracks,
    )
    trial = section_term.trial_response(equilibrium.state)
    if equilibrium.converged:
        return KirchhoffSectionEquilibriumResult(
            equilibrium,
            trial,
            trial.next_state,
            True,
        )
    return KirchhoffSectionEquilibriumResult(
        equilibrium,
        trial,
        section_term.committed_state,
        False,
    )


__all__ = [
    "KirchhoffFiberSectionBending",
    "KirchhoffSectionEquilibriumResult",
    "KirchhoffSectionError",
    "KirchhoffSectionReference",
    "KirchhoffSectionTrial",
    "KirchhoffSectionVertexLayout",
    "KirchhoffVertexKinematics",
    "build_kirchhoff_section_vertex_layout",
    "solve_kirchhoff_section_equilibrium",
]
