"""P3.1模型输入到现有Scene/MotionSource/CollisionQuery的模块化装配。

本模块只做编排：位姿组合在``pose_math``，资产字节与形状记录在
``scene_resources``，既有场景冻结仍由``scene.SceneAssembly``负责。它不读取WII/GCW，
不解析网格，也不自动生成全体两两接触。

首片支持static、kinematic与虚拟frame。dynamic合同已经存在，但它还缺“状态frame相对
质心”的冻结语义；本模块遇到dynamic明确拒绝，绝不把它冻结成static冒充已装配。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

from physics_engine.collision import BroadPhaseCollisionQuery
from physics_engine.model_physics import (
    BodyBehavior,
    GeometrySource,
    PhysicsBodyBinding,
    PhysicsModelMotionInput,
    VirtualFrameBinding,
)
from physics_engine.model_snapshot import ModelComponent, ModelSnapshot
from physics_engine.motion import (
    MotionError,
    MotionSource,
    Pose,
    interpolate_pose_fraction,
)
from physics_engine.planned_motion import MotionParameterization, PlannedMotion
from physics_engine.pose_math import IDENTITY_POSE, compose_pose
from physics_engine.scene import FinalizedScene, SceneAssembly, SceneError
from physics_engine.scene_resources import SceneResourceCatalog, SceneResourceError
from physics_engine.shapes import PosedBody, ShapeError, SimBody


class ModelSceneError(ValueError):
    """输入包、资源、运动或场景装配没有形成闭包。"""


def _pair(value: object, name: str) -> tuple[str, str]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ModelSceneError(f"{name} must be a two-item tuple of body IDs")
    return (value[0], value[1])


@dataclass(frozen=True)
class SceneInteractionPlan:
    """装配调用方显式声明的接触候选与允许重叠对。"""

    scene_id: str
    contact_pairs: tuple[tuple[str, str], ...]
    allowed_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.startswith("scene/"):
            raise ModelSceneError("scene_id must be namespaced like 'scene/...'")
        for index, pair in enumerate(self.contact_pairs):
            _pair(pair, f"contact_pairs[{index}]")
        for index, pair in enumerate(self.allowed_pairs):
            _pair(pair, f"allowed_pairs[{index}]")


@dataclass(frozen=True)
class _OffsetMotionSource:
    """把``root_from_component``后乘固定``component_from_geometry``。"""

    source: MotionSource
    component_from_geometry: Pose

    def __post_init__(self) -> None:
        if not isinstance(self.source, MotionSource):
            raise ModelSceneError("source must implement MotionSource")
        if not isinstance(self.component_from_geometry, Pose):
            raise ModelSceneError("component_from_geometry must be a Pose")

    def pose_at(self, t_s: float) -> Pose:
        return compose_pose(self.source.pose_at(t_s), self.component_from_geometry)

    def horizon_s(self) -> float:
        return self.source.horizon_s()

    def is_replayable(self) -> bool:
        return self.source.is_replayable()


@dataclass(frozen=True)
class _PlanningScalePoseSource:
    motion: PlannedMotion
    track_id: str
    suffix: Pose = IDENTITY_POSE

    def __post_init__(self) -> None:
        if not isinstance(self.motion, PlannedMotion):
            raise ModelSceneError("motion must be PlannedMotion")
        if self.motion.parameterization is not MotionParameterization.PLANNING_SCALE:
            raise ModelSceneError("planning-scale source requires planning_scale motion")
        self.motion.track(self.track_id)
        if not isinstance(self.suffix, Pose):
            raise ModelSceneError("planning source suffix must be a Pose")

    def pose_at(self, scale: float) -> Pose:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ModelSceneError(f"planning scale must be numeric: {scale!r}")
        value = float(scale)
        if not math.isfinite(value):
            raise ModelSceneError(f"planning scale must be finite: {scale!r}")
        track = self.motion.track(self.track_id)
        horizon = self.motion.samples[-1].coordinate
        if value < 0.0 or value > horizon:
            if track.interpolation.extrapolation == "reject":
                raise ModelSceneError(
                    f"planning scale {value!r} is outside [0, {horizon!r}]"
                )
            value = 0.0 if value < 0.0 else horizon
        if value >= horizon:
            return compose_pose(self.motion.samples[-1].pose(self.track_id), self.suffix)
        index = (
            bisect_right(
                self.motion.samples,
                value,
                key=lambda sample: sample.coordinate,
            )
            - 1
        )
        left = self.motion.samples[index]
        right = self.motion.samples[index + 1]
        fraction = (value - left.coordinate) / (right.coordinate - left.coordinate)
        pose = interpolate_pose_fraction(
            left.pose(self.track_id),
            right.pose(self.track_id),
            fraction,
            track.interpolation,
        )
        return compose_pose(pose, self.suffix)


@dataclass(frozen=True)
class PhysicsBodyRuntime:
    binding: PhysicsBodyBinding
    component_from_geometry: Pose
    resource_id: str
    planning_source: _PlanningScalePoseSource | None

    @property
    def body_id(self) -> str:
        return self.binding.body_id

    @property
    def behavior(self) -> BodyBehavior:
        return self.binding.behavior


@dataclass(frozen=True)
class VirtualFrameRuntime:
    binding: VirtualFrameBinding
    parameterization: MotionParameterization
    time_source: MotionSource | None
    planning_source: _PlanningScalePoseSource | None

    @property
    def virtual_frame_id(self) -> str:
        return self.binding.virtual_frame_id

    def pose_at_time(self, t_s: float) -> Pose:
        if self.parameterization is not MotionParameterization.TIME_S:
            raise ModelSceneError(
                f"{self.virtual_frame_id} is planning_scale, not physical time"
            )
        assert self.time_source is not None
        return self.time_source.pose_at(t_s)

    def pose_at_planning_scale(self, scale: float) -> Pose:
        if self.parameterization is not MotionParameterization.PLANNING_SCALE:
            raise ModelSceneError(
                f"{self.virtual_frame_id} is time_s, not planning_scale"
            )
        assert self.planning_source is not None
        return self.planning_source.pose_at(scale)


@dataclass(frozen=True)
class PreparedModelScene:
    """已冻结Scene，加上物理行为与虚拟frame运行时。"""

    source_input_sha256: str
    parameterization: MotionParameterization
    scene: FinalizedScene
    physics_bodies: tuple[PhysicsBodyRuntime, ...]
    virtual_frames: tuple[VirtualFrameRuntime, ...]
    excluded_component_ids: tuple[str, ...]
    excluded_motion_track_ids: tuple[str, ...]

    def body_runtime(self, body_id: str) -> PhysicsBodyRuntime:
        for runtime in self.physics_bodies:
            if runtime.body_id == body_id:
                return runtime
        raise ModelSceneError(f"prepared scene has no physics body {body_id!r}")

    def _virtual_frame(self, frame_id: str) -> VirtualFrameRuntime:
        for runtime in self.virtual_frames:
            if runtime.virtual_frame_id == frame_id:
                return runtime
        raise ModelSceneError(f"prepared scene has no virtual frame {frame_id!r}")

    def posed_bodies_at_time(self, t_s: float) -> tuple[PosedBody, ...]:
        if self.parameterization is not MotionParameterization.TIME_S:
            raise ModelSceneError("prepared scene is planning_scale, not physical time")
        return self.scene.posed_bodies_at(t_s)

    def posed_bodies_at_planning_scale(self, scale: float) -> tuple[PosedBody, ...]:
        if self.parameterization is not MotionParameterization.PLANNING_SCALE:
            raise ModelSceneError("prepared scene is time_s, not planning_scale")
        posed = []
        for assembled, runtime in zip(
            self.scene.bodies, self.physics_bodies, strict=True
        ):
            if runtime.behavior is BodyBehavior.KINEMATIC:
                assert runtime.planning_source is not None
                pose = runtime.planning_source.pose_at(scale)
                posed.append(
                    PosedBody(
                        body=assembled.posed.body,
                        translation_mm=pose.translation_mm,
                        rotation_xyzw=pose.rotation_xyzw,
                    )
                )
            else:
                posed.append(assembled.posed)
        return tuple(posed)

    def virtual_frame_pose_at_time(self, frame_id: str, t_s: float) -> Pose:
        return self._virtual_frame(frame_id).pose_at_time(t_s)

    def virtual_frame_pose_at_planning_scale(self, frame_id: str, scale: float) -> Pose:
        return self._virtual_frame(frame_id).pose_at_planning_scale(scale)

    def _query(self, bodies: tuple[PosedBody, ...]) -> BroadPhaseCollisionQuery:
        candidates = tuple(
            (pair.body_a, pair.body_b) for pair in self.scene.contact_pairs
        )
        return BroadPhaseCollisionQuery(
            bodies,
            allowed_pairs=self.scene.allowed_pairs,
            candidate_pairs=candidates,
        )

    def collision_query_at_time(self, t_s: float) -> BroadPhaseCollisionQuery:
        return self._query(self.posed_bodies_at_time(t_s))

    def collision_query_at_planning_scale(
        self, scale: float
    ) -> BroadPhaseCollisionQuery:
        return self._query(self.posed_bodies_at_planning_scale(scale))


def reference_component_poses(model: ModelSnapshot) -> tuple[tuple[str, Pose], ...]:
    """按模型声明次序解析每个组件的``root_from_component``参考位姿。"""

    if not isinstance(model, ModelSnapshot):
        raise ModelSceneError("model must be ModelSnapshot")
    components = {component.component_id: component for component in model.components}
    resolved: dict[str, Pose] = {}

    def resolve(component: ModelComponent) -> Pose:
        cached = resolved.get(component.component_id)
        if cached is not None:
            return cached
        if component.parent_component_id is None:
            pose = component.parent_from_component
        else:
            pose = compose_pose(
                resolve(components[component.parent_component_id]),
                component.parent_from_component,
            )
        resolved[component.component_id] = pose
        return pose

    return tuple((component.component_id, resolve(component)) for component in model.components)


def _geometry(
    binding: PhysicsBodyBinding,
    component: ModelComponent,
    resources: SceneResourceCatalog,
):
    if binding.geometry_source is GeometrySource.COLLISION_ASSET:
        assert component.collision_asset is not None
        try:
            loaded = resources.collision_asset(component.collision_asset.asset_id)
        except SceneResourceError as error:
            raise ModelSceneError(f"body {binding.body_id}: {error}") from error
        if loaded.asset != component.collision_asset:
            raise ModelSceneError(
                f"body {binding.body_id}: loaded collision asset identity differs from model"
            )
        return (
            loaded.collision,
            component.collision_asset.component_from_asset,
            component.collision_asset.asset_id,
        )
    assert binding.analytic_shape_id is not None
    try:
        analytic = resources.analytic_shape(binding.analytic_shape_id)
    except SceneResourceError as error:
        raise ModelSceneError(f"body {binding.body_id}: {error}") from error
    return analytic.collision, analytic.component_from_shape, analytic.shape_id


def assemble_model_physics_scene(
    package: PhysicsModelMotionInput,
    resources: SceneResourceCatalog,
    interactions: SceneInteractionPlan,
) -> PreparedModelScene:
    """把已验证输入包装成现有Scene及显式时间/规划尺度运行面。"""

    if not isinstance(package, PhysicsModelMotionInput) or package.content_sha256 is None:
        raise ModelSceneError("package must be a sealed PhysicsModelMotionInput")
    if not isinstance(resources, SceneResourceCatalog):
        raise ModelSceneError("resources must be SceneResourceCatalog")
    if not isinstance(interactions, SceneInteractionPlan):
        raise ModelSceneError("interactions must be SceneInteractionPlan")
    dynamic = [
        binding.body_id
        for binding in package.relation.body_bindings
        if binding.behavior is BodyBehavior.DYNAMIC
    ]
    if dynamic:
        raise ModelSceneError(
            f"dynamic bodies {dynamic} cannot enter Scene yet: dynamic state frame relative "
            "to the mass centroid is not frozen; freezing them as static would create two "
            "different physics meanings"
        )

    reference = dict(reference_component_poses(package.model))
    assembly = SceneAssembly(interactions.scene_id)
    runtimes: list[PhysicsBodyRuntime] = []
    try:
        for binding in package.relation.body_bindings:
            component = package.model.component(binding.component_id)
            collision, component_from_geometry, resource_id = _geometry(
                binding, component, resources
            )
            planning_source = None
            motion_source = None
            if binding.behavior is BodyBehavior.KINEMATIC:
                assert binding.motion_track_id is not None
                if package.motion.parameterization is MotionParameterization.TIME_S:
                    motion_source = _OffsetMotionSource(
                        package.motion.as_time_source(binding.motion_track_id),
                        component_from_geometry,
                    )
                    initial = IDENTITY_POSE
                else:
                    planning_source = _PlanningScalePoseSource(
                        package.motion,
                        binding.motion_track_id,
                        component_from_geometry,
                    )
                    initial = planning_source.pose_at(0.0)
            else:
                initial = compose_pose(
                    reference[component.component_id], component_from_geometry
                )
            posed = PosedBody(
                body=SimBody(body_id=binding.body_id, collision=collision),
                translation_mm=initial.translation_mm,
                rotation_xyzw=initial.rotation_xyzw,
            )
            assembly.declare_body(posed)
            if motion_source is not None:
                assembly.declare_motion_source(binding.body_id, motion_source)
            runtimes.append(
                PhysicsBodyRuntime(
                    binding=binding,
                    component_from_geometry=component_from_geometry,
                    resource_id=resource_id,
                    planning_source=planning_source,
                )
            )
        for body_a, body_b in interactions.contact_pairs:
            assembly.declare_contact_between(body_a, body_b)
        for body_a, body_b in interactions.allowed_pairs:
            assembly.declare_allowed_pair(body_a, body_b)
        scene = assembly.finalize()

        virtual_frames = []
        for binding in package.relation.virtual_frame_bindings:
            if package.motion.parameterization is MotionParameterization.TIME_S:
                time_source = package.motion.as_time_source(binding.motion_track_id)
                planning_source = None
            else:
                time_source = None
                planning_source = _PlanningScalePoseSource(
                    package.motion, binding.motion_track_id
                )
            virtual_frames.append(
                VirtualFrameRuntime(
                    binding=binding,
                    parameterization=package.motion.parameterization,
                    time_source=time_source,
                    planning_source=planning_source,
                )
            )
    except (MotionError, SceneError, SceneResourceError, ShapeError) as error:
        raise ModelSceneError(f"model scene assembly failed: {error}") from error

    return PreparedModelScene(
        source_input_sha256=package.content_sha256,
        parameterization=package.motion.parameterization,
        scene=scene,
        physics_bodies=tuple(runtimes),
        virtual_frames=tuple(virtual_frames),
        excluded_component_ids=package.relation.excluded_component_ids,
        excluded_motion_track_ids=package.relation.excluded_motion_track_ids,
    )


__all__ = [
    "ModelSceneError",
    "PhysicsBodyRuntime",
    "PreparedModelScene",
    "SceneInteractionPlan",
    "VirtualFrameRuntime",
    "assemble_model_physics_scene",
    "reference_component_poses",
]
