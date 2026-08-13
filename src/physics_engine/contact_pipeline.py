"""动态接触整合层：声明候选→碰撞窄相→法向响应。

这层存在是为了接上此前断开的两件真东西：

* ``FinalizedScene.contact_pairs``给固定、有序的候选身份；
* ``collision.BroadPhaseCollisionQuery``每步判断哪些候选此刻真的活动。

候选池不随时间改变，因此历史槽位身份稳定；活动集是每步的值，不是布局变更。
本模块的动态检测只覆盖**平动球体的球-球法向罚接触+线性dashpot**。解析平面可经
``fixed_planes``直通同一耗散项以保留迁移前加法树，但不经过动态检测；摩擦、转动、
网格窄相和活动历史写回不在这里冒充完成。

全部对象只在进程内流动，不落盘、不进run package，因此本批没有新增字节facet。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import ClassVar, Literal

from physics_engine.collision import BroadPhaseCollisionQuery, CollisionQueryResult
from physics_engine.contact import (
    ContactDeclaration,
    LinearNormalDashpot,
    PenaltySphereContact,
)
from physics_engine.energies import DISSIPATION, POTENTIAL, EnergyContext, Matrix, Vector
from physics_engine.scene import FinalizedScene
from physics_engine.shapes import GeneratedShape, PosedBody, Sphere, Vector3
from physics_engine.state import State


class ContactPipelineError(ValueError):
    """检测—响应整合层的失败关闭。"""


@dataclass(frozen=True)
class SphereNodeBinding:
    """一个场景球体到状态向量平动节点的稳定绑定。"""

    body_id: str
    node_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.body_id, str) or not self.body_id:
            raise ContactPipelineError("sphere binding needs a nonempty body_id")
        if (
            isinstance(self.node_index, bool)
            or not isinstance(self.node_index, int)
            or self.node_index < 0
        ):
            raise ContactPipelineError(
                f"sphere binding node index must be a nonnegative int: {self.node_index!r}"
            )


@dataclass(frozen=True)
class ActiveSphereContact:
    """一个由窄相确认、可以进入响应的球-球接触。"""

    pair_id: str
    body_a: str
    body_b: str
    node_a: int
    node_b: int
    radii_sum_mm: float
    stiffness_n_per_mm: float
    damping_n_s_per_mm: float
    penetration_mm: float

    @property
    def spring_pair(self) -> tuple[int, int, float, float]:
        return (self.node_a, self.node_b, self.radii_sum_mm, self.stiffness_n_per_mm)

    @property
    def dashpot_pair(self) -> tuple[int, int, float, float, float]:
        return (
            self.node_a,
            self.node_b,
            self.radii_sum_mm,
            self.stiffness_n_per_mm,
            self.damping_n_s_per_mm,
        )

    @property
    def normal_spring_force_n(self) -> float:
        return self.stiffness_n_per_mm * self.penetration_mm


@dataclass(frozen=True)
class SphereContactEvaluation:
    """一帧的活动集与检测工作量回执。"""

    active_contacts: tuple[ActiveSphereContact, ...]
    query: CollisionQueryResult


@dataclass
class _EvaluationCache:
    """同一位置帧的单项缓存；锁只保护这一份pipeline实例。"""

    lock: Lock = field(default_factory=Lock)
    vector: Vector | None = None
    evaluation: SphereContactEvaluation | None = None


def _pair_tag(index: int, body_a: str, body_b: str) -> str:
    def clean(value: str) -> str:
        return "".join(character if character.isalnum() else "_" for character in value)

    return f"candidate_{index:04d}_{clean(body_a)}__{clean(body_b)}"


@dataclass(frozen=True)
class SphereContactPipeline:
    """把冻结场景的球体候选池接到每步窄相与法向响应。

    第一片刻意使用**全候选统一**的刚度与阻尼：十球漏斗是当前唯一消费方，
    还没有逐材料对参数的真实需求。需要逐对本构时再扩，不预支第三套配置表。

    位姿只有一个来源：本层以``State``节点位置为权威，因此被绑定体不得再带
    ``MotionSource``或非单位静态位姿。否则同一球在一帧里会有两份位置。
    """

    scene: FinalizedScene
    bindings: tuple[SphereNodeBinding, ...]
    stiffness_n_per_mm: float
    damping_n_s_per_mm: float
    _candidate_pairs: tuple[tuple[str, str], ...] = field(init=False, repr=False)
    _candidate_ids: tuple[str, ...] = field(init=False, repr=False)
    _radius_by_body: dict[str, float] = field(init=False, repr=False, compare=False)
    _node_by_body: dict[str, int] = field(init=False, repr=False, compare=False)
    _posed_body_by_id: dict[str, PosedBody] = field(init=False, repr=False, compare=False)
    _evaluation_cache: _EvaluationCache = field(
        default_factory=_EvaluationCache,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.scene, FinalizedScene):
            raise ContactPipelineError("scene must be a finalized SceneAssembly")
        if not self.scene.contact_pairs:
            raise ContactPipelineError("dynamic contact pipeline needs declared contact pairs")
        if not math.isfinite(self.stiffness_n_per_mm) or self.stiffness_n_per_mm <= 0.0:
            raise ContactPipelineError("stiffness_n_per_mm must be positive and finite")
        if not math.isfinite(self.damping_n_s_per_mm) or self.damping_n_s_per_mm <= 0.0:
            raise ContactPipelineError("damping_n_s_per_mm must be positive and finite")
        if not self.bindings:
            raise ContactPipelineError("dynamic contact pipeline needs sphere bindings")

        body_ids = [binding.body_id for binding in self.bindings]
        duplicate_bodies = sorted({name for name in body_ids if body_ids.count(name) > 1})
        if duplicate_bodies:
            raise ContactPipelineError(f"duplicate sphere body bindings: {duplicate_bodies}")
        node_indices = [binding.node_index for binding in self.bindings]
        duplicate_nodes = sorted({node for node in node_indices if node_indices.count(node) > 1})
        if duplicate_nodes:
            raise ContactPipelineError(
                f"multiple sphere bodies are bound to the same node: {duplicate_nodes}"
            )

        assembled = {body.body_id: body for body in self.scene.bodies}
        unknown_bindings = sorted(set(body_ids) - set(assembled))
        if unknown_bindings:
            raise ContactPipelineError(
                f"sphere bindings reference bodies outside the scene: {unknown_bindings}"
            )
        required = {
            member
            for pair in self.scene.contact_pairs
            for member in (pair.body_a, pair.body_b)
        }
        missing = sorted(required - set(body_ids))
        if missing:
            raise ContactPipelineError(
                f"declared contact bodies have no state-node binding: {missing}"
            )

        radii: dict[str, float] = {}
        posed: dict[str, PosedBody] = {}
        for binding in self.bindings:
            body = assembled[binding.body_id]
            if body.motion_source is not None:
                raise ContactPipelineError(
                    f"{binding.body_id}: state binding and MotionSource are two pose sources"
                )
            declared_pose = body.posed
            if declared_pose.translation_mm != (0.0, 0.0, 0.0) or (
                declared_pose.rotation_xyzw != (0.0, 0.0, 0.0, 1.0)
            ):
                raise ContactPipelineError(
                    f"{binding.body_id}: state-bound sphere must use an identity static pose"
                )
            shape = declared_pose.body.collision.shape
            if isinstance(shape, GeneratedShape):
                shape = shape.shape
            if not isinstance(shape, Sphere):
                raise ContactPipelineError(
                    f"{binding.body_id}: dynamic pipeline first slice only supports Sphere, "
                    f"got {type(shape).__name__}"
                )
            radii[binding.body_id] = shape.radius_mm
            posed[binding.body_id] = declared_pose

        pairs = tuple((pair.body_a, pair.body_b) for pair in self.scene.contact_pairs)
        pair_ids = tuple(
            _pair_tag(index, body_a, body_b)
            for index, (body_a, body_b) in enumerate(pairs)
        )
        object.__setattr__(self, "_candidate_pairs", pairs)
        object.__setattr__(self, "_candidate_ids", pair_ids)
        object.__setattr__(self, "_radius_by_body", radii)
        object.__setattr__(
            self,
            "_node_by_body",
            {binding.body_id: binding.node_index for binding in self.bindings},
        )
        object.__setattr__(self, "_posed_body_by_id", posed)

    @property
    def candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        return self._candidate_pairs

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self._candidate_ids

    @property
    def scene_id(self) -> str:
        return self.scene.scene_id

    def contact_declarations(self) -> tuple[ContactDeclaration, ...]:
        """给0050定长历史布局的候选清单；活动集不改变它。"""

        return tuple(ContactDeclaration(pair_id) for pair_id in self._candidate_ids)

    def node_index_bound(self) -> int:
        return max(self._node_by_body.values()) + 1

    def _assert_state_covers_bindings(self, state: State) -> None:
        node_dof = state.layout.node_dof_count
        if node_dof is None:
            node_dof = len(state.vector)
        required = 3 * self.node_index_bound()
        if required > node_dof:
            raise ContactPipelineError(
                f"pipeline indexes {required // 3} nodes but layout "
                f"{state.layout.layout_id!r} exposes only {node_dof // 3}"
            )

    def evaluate(self, state: State) -> SphereContactEvaluation:
        """重建本帧球位姿、跑声明候选池、只接收窄相确认事件。

        检测只依赖位置。能量梯度、耗散力与耗散功率在同一积分帧会依次问同一份
        活动集，因此缓存最近一个逐位相同的状态向量。缓存命中不改变事件或响应字节；
        每个pipeline实例自带锁，跨线程共享时不会把两帧回执串在一起。
        """

        self._assert_state_covers_bindings(state)
        cache = self._evaluation_cache
        with cache.lock:
            if cache.vector == state.vector and cache.evaluation is not None:
                return cache.evaluation
            evaluation = self._evaluate_uncached(state)
            cache.vector = state.vector
            cache.evaluation = evaluation
            return evaluation

    def _evaluate_uncached(self, state: State) -> SphereContactEvaluation:
        """执行一次真实检测；调用方必须已持有本实例缓存锁。"""

        posed_bodies = tuple(
            PosedBody(
                body=self._posed_body_by_id[binding.body_id].body,
                translation_mm=tuple(
                    state.vector[3 * binding.node_index + axis] for axis in range(3)
                ),
            )
            for binding in self.bindings
        )
        query = BroadPhaseCollisionQuery(
            posed_bodies,
            allowed_pairs=self.scene.allowed_pairs,
            candidate_pairs=self._candidate_pairs,
        ).check_state_with_stats()
        if query.candidate_pair_count != len(self._candidate_pairs):
            raise ContactPipelineError(
                "collision query changed the declared candidate count — "
                "candidate identity is a frozen scene property"
            )

        index_of = {pair: index for index, pair in enumerate(self._candidate_pairs)}
        active: list[ActiveSphereContact] = []
        seen: set[tuple[str, str]] = set()
        for event in query.events:
            pair = (event.body_a, event.body_b)
            if pair not in index_of:
                raise ContactPipelineError(
                    f"collision query returned undeclared pair {pair!r}"
                )
            if pair in seen:
                raise ContactPipelineError(f"collision query returned pair {pair!r} twice")
            seen.add(pair)
            if event.confidence != "narrow_phase" or event.penetration_mm is None:
                raise ContactPipelineError(
                    f"pair {pair!r} has confidence {event.confidence!r}; "
                    "only narrow_phase events with penetration may enter response"
                )
            if not math.isfinite(event.penetration_mm) or event.penetration_mm <= 0.0:
                raise ContactPipelineError(
                    f"pair {pair!r} has invalid penetration {event.penetration_mm!r}"
                )
            body_a, body_b = pair
            node_a, node_b = self._node_by_body[body_a], self._node_by_body[body_b]
            radii_sum = self._radius_by_body[body_a] + self._radius_by_body[body_b]
            response_pair = (node_a, node_b, radii_sum, self.stiffness_n_per_mm)
            gap, _, _ = PenaltySphereContact._pair_state(state.vector, response_pair)
            expected_penetration = -gap
            if not math.isclose(
                event.penetration_mm,
                expected_penetration,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ContactPipelineError(
                    f"pair {pair!r} collision penetration {event.penetration_mm!r} "
                    f"does not match response geometry {expected_penetration!r}"
                )
            index = index_of[pair]
            active.append(
                ActiveSphereContact(
                    pair_id=self._candidate_ids[index],
                    body_a=body_a,
                    body_b=body_b,
                    node_a=node_a,
                    node_b=node_b,
                    radii_sum_mm=radii_sum,
                    stiffness_n_per_mm=self.stiffness_n_per_mm,
                    damping_n_s_per_mm=self.damping_n_s_per_mm,
                    penetration_mm=event.penetration_mm,
                )
            )
        return SphereContactEvaluation(tuple(active), query)


@dataclass(frozen=True)
class DetectedSphereContactPotential:
    """只把``SphereContactPipeline``当前确认的活动对送进罚势。"""

    pipeline: SphereContactPipeline
    name: str = "detected_sphere_contact"
    kind: ClassVar[Literal["potential"]] = POTENTIAL

    def node_index_bound(self) -> int:
        return self.pipeline.node_index_bound()

    def _term(self, state: State) -> PenaltySphereContact | None:
        pairs = tuple(contact.spring_pair for contact in self.pipeline.evaluate(state).active_contacts)
        return PenaltySphereContact._from_validated_pairs(pairs) if pairs else None

    def energy(self, state: State, context: EnergyContext) -> float:
        term = self._term(state)
        return 0.0 if term is None else term.energy(state, context)

    def gradient(self, state: State, context: EnergyContext) -> Vector:
        term = self._term(state)
        return (0.0,) * len(state.vector) if term is None else term.gradient(state, context)

    def hessian(self, state: State, context: EnergyContext) -> Matrix:
        term = self._term(state)
        if term is None:
            return tuple((0.0,) * len(state.vector) for _ in state.vector)
        return term.hessian(state, context)

    def hessian_entries(
        self, state: State, context: EnergyContext
    ) -> tuple[tuple[int, int, float], ...]:
        term = self._term(state)
        return () if term is None else term.hessian_entries(state, context)

    def quantities(
        self,
        state: State,
        context: EnergyContext,
        *,
        need_gradient: bool,
        need_hessian: bool,
    ) -> tuple[float, Vector | None, Matrix | None]:
        term = self._term(state)
        if term is not None:
            return term.quantities(
                state,
                context,
                need_gradient=need_gradient,
                need_hessian=need_hessian,
            )
        return (
            0.0,
            (0.0,) * len(state.vector) if need_gradient else None,
            tuple((0.0,) * len(state.vector) for _ in state.vector)
            if need_hessian
            else None,
        )


@dataclass(frozen=True)
class DetectedSphereContactDissipation:
    """只把动态窄相确认的活动对送进线性法向dashpot。

    ``fixed_planes``是迁移兼容口：解析平面不经过球体检测，但与活动球对保持在
    **同一个**``LinearNormalDashpot``里，从而保留既有力与功率的浮点加法树。
    """

    pipeline: SphereContactPipeline
    fixed_planes: tuple[
        tuple[int, Vector3, Vector3, float, float, float], ...
    ] = ()
    name: str = "detected_sphere_dashpot"
    kind: ClassVar[Literal["dissipation"]] = DISSIPATION

    def __post_init__(self) -> None:
        if self.fixed_planes:
            LinearNormalDashpot(planes=self.fixed_planes)

    def node_index_bound(self) -> int:
        plane_bound = max((plane[0] + 1 for plane in self.fixed_planes), default=0)
        return max(self.pipeline.node_index_bound(), plane_bound)

    def force_and_power(
        self,
        state: State,
        velocity: Sequence[float],
        context: EnergyContext,
    ) -> tuple[Vector, float]:
        contacts = self.pipeline.evaluate(state).active_contacts
        if not contacts and not self.fixed_planes:
            return (0.0,) * len(state.vector), 0.0
        return LinearNormalDashpot._from_validated_parts(
            planes=self.fixed_planes,
            sphere_pairs=tuple(contact.dashpot_pair for contact in contacts)
        ).force_and_power(state, velocity, context)


__all__ = [
    "ActiveSphereContact",
    "ContactPipelineError",
    "DetectedSphereContactDissipation",
    "DetectedSphereContactPotential",
    "SphereContactEvaluation",
    "SphereContactPipeline",
    "SphereNodeBinding",
]
