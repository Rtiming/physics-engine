"""上游无关的规划运动合同——时间与无时间规划尺度绝不混写。

WII、轨迹规划器或人工工艺表可以适配成``PlannedMotion``。时间参数化计划可转成
``motion.SampledPoseTimeline``进入动态边界；只有几何规划尺度的计划保留状态与路径进度，
但调用``as_time_source``会失败关闭，防止把ξ之类无量纲参数冒充秒。
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace
from typing import Any

from physics_engine.canonical import WDS_PROFILE, canonical_sha256
from physics_engine.identity import IdentityError, parse_namespace_id
from physics_engine.model_snapshot import pose_from_document, pose_to_document
from physics_engine.motion import (
    InterpolationSemantics,
    MotionError,
    PauseInterval,
    Pose,
    PoseSample,
    SampledPoseTimeline,
)

PLANNED_MOTION_CANONICAL_PROFILE = WDS_PROFILE


class PlannedMotionError(ValueError):
    """规划坐标、状态、track或字节闭包错误。"""


class MotionParameterization(enum.StrEnum):
    TIME_S = "time_s"
    PLANNING_SCALE = "planning_scale"


def _require_namespace(value: object, namespace: str, name: str) -> str:
    if not isinstance(value, str):
        raise PlannedMotionError(f"{name} must be a string: {value!r}")
    try:
        parsed, _ = parse_namespace_id(value)
    except IdentityError as error:
        raise PlannedMotionError(f"{name} is not a valid namespace id: {error}") from error
    if parsed != namespace:
        raise PlannedMotionError(f"{name} must live in {namespace!r}: {value!r}")
    return value


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlannedMotionError(f"{name} must be a nonempty string")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
        raise PlannedMotionError(f"{name} must use lowercase identifier characters: {value!r}")
    return value


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannedMotionError(f"{name} must be numeric, not {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise PlannedMotionError(f"{name} must be finite: {value!r}")
    return result


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PlannedMotionError(f"{name} must be 64 lowercase hex characters")
    return value


def _semantics_to_document(value: InterpolationSemantics) -> dict[str, str]:
    if not isinstance(value, InterpolationSemantics):
        raise PlannedMotionError("interpolation must be InterpolationSemantics")
    return {
        "translation_interpolation": value.translation_interpolation,
        "rotation_interpolation": value.rotation_interpolation,
        "rotation_arc": value.rotation_arc,
        "pause_hold": value.pause_hold,
        "extrapolation": value.extrapolation,
    }


def _semantics_from_document(value: object, name: str) -> InterpolationSemantics:
    expected = {
        "translation_interpolation",
        "rotation_interpolation",
        "rotation_arc",
        "pause_hold",
        "extrapolation",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise PlannedMotionError(f"{name} fields differ from interpolation contract")
    try:
        return InterpolationSemantics(**value)
    except (TypeError, ValueError) as error:
        raise PlannedMotionError(f"invalid {name}: {error}") from error


@dataclass(frozen=True)
class MotionStateCoordinate:
    coordinate_id: str
    unit: str

    def __post_init__(self) -> None:
        _require_namespace(self.coordinate_id, "state-coordinate", "coordinate_id")
        if self.unit not in {"deg", "rad", "mm", "1"}:
            raise PlannedMotionError(f"unsupported source-state unit: {self.unit!r}")

    def to_document(self) -> dict[str, str]:
        return {"coordinate_id": self.coordinate_id, "unit": self.unit}


@dataclass(frozen=True)
class MotionSourceArtifact:
    role: str
    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.role, "source artifact role")
        _require_namespace(self.artifact_id, "artifact", "artifact_id")
        _require_sha256(self.sha256, "source artifact sha256")

    def to_document(self) -> dict[str, str]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MotionTrack:
    track_id: str
    component_id: str | None
    frame_id: str
    interpolation: InterpolationSemantics

    def __post_init__(self) -> None:
        _require_namespace(self.track_id, "motion-track", "track_id")
        if self.component_id is not None:
            _require_namespace(self.component_id, "model-component", "component_id")
        _require_namespace(self.frame_id, "frame", "frame_id")
        if not isinstance(self.interpolation, InterpolationSemantics):
            raise PlannedMotionError("interpolation must be InterpolationSemantics")

    def to_document(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "component_id": self.component_id,
            "frame_id": self.frame_id,
            "interpolation": _semantics_to_document(self.interpolation),
        }


@dataclass(frozen=True)
class TrackPose:
    track_id: str
    parent_from_track: Pose

    def __post_init__(self) -> None:
        _require_namespace(self.track_id, "motion-track", "track_id")
        if not isinstance(self.parent_from_track, Pose):
            raise PlannedMotionError("parent_from_track must be a Pose")

    def to_document(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "parent_from_track": pose_to_document(self.parent_from_track),
        }


@dataclass(frozen=True)
class PlannedMotionSample:
    sample_id: str
    coordinate: float
    path_progress: float
    track_poses: tuple[TrackPose, ...]
    source_state_values: tuple[float, ...]
    material_feed_length_mm: float
    stage_id: str

    def __post_init__(self) -> None:
        _require_namespace(self.sample_id, "motion-sample", "sample_id")
        coordinate = _require_finite(self.coordinate, "coordinate")
        if coordinate < 0.0:
            raise PlannedMotionError("sample coordinate must be nonnegative")
        progress = _require_finite(self.path_progress, "path_progress")
        if not 0.0 <= progress <= 1.0:
            raise PlannedMotionError("path_progress must live in [0, 1]")
        if not self.track_poses or not all(
            isinstance(item, TrackPose) for item in self.track_poses
        ):
            raise PlannedMotionError("track_poses must contain TrackPose values")
        if len({item.track_id for item in self.track_poses}) != len(self.track_poses):
            raise PlannedMotionError("a sample may name each motion track only once")
        for index, value in enumerate(self.source_state_values):
            _require_finite(value, f"source_state_values[{index}]")
        feed = _require_finite(self.material_feed_length_mm, "material_feed_length_mm")
        if feed < 0.0:
            raise PlannedMotionError("material_feed_length_mm must be nonnegative")
        _require_identifier(self.stage_id, "stage_id")

    def pose(self, track_id: str) -> Pose:
        for item in self.track_poses:
            if item.track_id == track_id:
                return item.parent_from_track
        raise PlannedMotionError(f"sample {self.sample_id} has no track {track_id}")

    def to_document(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "coordinate": self.coordinate,
            "path_progress": self.path_progress,
            "track_poses": [item.to_document() for item in self.track_poses],
            "source_state_values": list(self.source_state_values),
            "material_feed_length_mm": self.material_feed_length_mm,
            "stage_id": self.stage_id,
        }


@dataclass(frozen=True)
class PlannedMotion:
    motion_id: str
    producer_id: str
    root_frame_id: str
    parameterization: MotionParameterization
    coordinate_unit: str
    source_artifacts: tuple[MotionSourceArtifact, ...]
    tracks: tuple[MotionTrack, ...]
    state_coordinates: tuple[MotionStateCoordinate, ...]
    samples: tuple[PlannedMotionSample, ...]
    pauses: tuple[PauseInterval, ...]
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_namespace(self.motion_id, "motion", "motion_id")
        _require_namespace(self.producer_id, "producer", "producer_id")
        _require_namespace(self.root_frame_id, "frame", "root_frame_id")
        if not isinstance(self.parameterization, MotionParameterization):
            raise PlannedMotionError("parameterization must be MotionParameterization")
        expected_unit = "s" if self.parameterization is MotionParameterization.TIME_S else "1"
        if self.coordinate_unit != expected_unit:
            raise PlannedMotionError(
                f"{self.parameterization.value} requires coordinate_unit={expected_unit!r}"
            )
        if not self.source_artifacts or not all(
            isinstance(item, MotionSourceArtifact) for item in self.source_artifacts
        ):
            raise PlannedMotionError(
                "source_artifacts must contain at least one MotionSourceArtifact"
            )
        roles = tuple(item.role for item in self.source_artifacts)
        artifact_ids = tuple(item.artifact_id for item in self.source_artifacts)
        if len(set(roles)) != len(roles) or len(set(artifact_ids)) != len(artifact_ids):
            raise PlannedMotionError("source artifact roles and IDs must be unique")
        if not self.tracks or not all(isinstance(item, MotionTrack) for item in self.tracks):
            raise PlannedMotionError("tracks must contain MotionTrack values")
        track_ids = tuple(track.track_id for track in self.tracks)
        if len(set(track_ids)) != len(track_ids):
            raise PlannedMotionError("motion track IDs must be unique")
        coordinate_ids = tuple(item.coordinate_id for item in self.state_coordinates)
        if len(set(coordinate_ids)) != len(coordinate_ids):
            raise PlannedMotionError("state coordinate IDs must be unique")
        if len(self.samples) < 2 or not all(
            isinstance(sample, PlannedMotionSample) for sample in self.samples
        ):
            raise PlannedMotionError("planned motion needs at least two samples")
        if self.samples[0].coordinate != 0.0:
            raise PlannedMotionError("planned coordinate must start at zero")
        if any(
            right.coordinate <= left.coordinate
            for left, right in zip(self.samples, self.samples[1:], strict=False)
        ):
            raise PlannedMotionError("planned coordinates must be strictly increasing")
        if self.parameterization is MotionParameterization.PLANNING_SCALE:
            if self.samples[-1].coordinate != 1.0:
                raise PlannedMotionError("planning_scale must end at one")
        if self.samples[0].path_progress != 0.0 or self.samples[-1].path_progress != 1.0:
            raise PlannedMotionError("path_progress must start at zero and end at one")
        if any(
            right.path_progress < left.path_progress
            for left, right in zip(self.samples, self.samples[1:], strict=False)
        ):
            raise PlannedMotionError("path_progress must be monotone nondecreasing")
        expected_tracks = set(track_ids)
        expected_state_count = len(self.state_coordinates)
        for sample in self.samples:
            actual_tracks = {item.track_id for item in sample.track_poses}
            if actual_tracks != expected_tracks:
                raise PlannedMotionError(
                    f"sample {sample.sample_id} track set differs from the plan"
                )
            if len(sample.source_state_values) != expected_state_count:
                raise PlannedMotionError(
                    f"sample {sample.sample_id} source-state width differs from the plan"
                )
        feeds = [sample.material_feed_length_mm for sample in self.samples]
        if feeds[0] != 0.0 or any(
            right < left for left, right in zip(feeds, feeds[1:], strict=False)
        ):
            raise PlannedMotionError("material feed must start at zero and not decrease")
        if self.parameterization is not MotionParameterization.TIME_S and self.pauses:
            raise PlannedMotionError("pause intervals require physical time parameterization")
        if not all(isinstance(pause, PauseInterval) for pause in self.pauses):
            raise PlannedMotionError("pauses must contain PauseInterval values")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
            if self.content_sha256 != self.content_address():
                raise PlannedMotionError("planned motion content_sha256 does not match")

    @classmethod
    def create(
        cls,
        *,
        motion_id: str,
        producer_id: str,
        root_frame_id: str,
        parameterization: MotionParameterization,
        coordinate_unit: str,
        source_artifacts: tuple[MotionSourceArtifact, ...],
        tracks: tuple[MotionTrack, ...],
        state_coordinates: tuple[MotionStateCoordinate, ...],
        samples: tuple[PlannedMotionSample, ...],
        pauses: tuple[PauseInterval, ...] = (),
    ) -> PlannedMotion:
        return cls(
            motion_id=motion_id,
            producer_id=producer_id,
            root_frame_id=root_frame_id,
            parameterization=parameterization,
            coordinate_unit=coordinate_unit,
            source_artifacts=source_artifacts,
            tracks=tracks,
            state_coordinates=state_coordinates,
            samples=samples,
            pauses=pauses,
        ).sealed()

    def track(self, track_id: str) -> MotionTrack:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        raise PlannedMotionError(f"unknown motion track: {track_id}")

    def as_time_source(self, track_id: str) -> SampledPoseTimeline:
        if self.parameterization is not MotionParameterization.TIME_S:
            raise MotionError(
                f"{self.motion_id} has no physical time; planning_scale cannot masquerade as seconds"
            )
        track = self.track(track_id)
        motion_name = self.motion_id.split("/", 1)[1]
        track_name = track.track_id.split("/", 1)[1]
        return SampledPoseTimeline(
            source_id=f"motion/{motion_name}--{track_name}",
            samples=tuple(
                PoseSample(time_s=sample.coordinate, pose=sample.pose(track_id))
                for sample in self.samples
            ),
            semantics=track.interpolation,
            translation_unit="mm",
            pauses=self.pauses,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "motion_id": self.motion_id,
            "producer_id": self.producer_id,
            "root_frame_id": self.root_frame_id,
            "parameterization": self.parameterization.value,
            "coordinate_unit": self.coordinate_unit,
            "source_artifacts": [item.to_document() for item in self.source_artifacts],
            "tracks": [track.to_document() for track in self.tracks],
            "state_coordinates": [item.to_document() for item in self.state_coordinates],
            "samples": [sample.to_document() for sample in self.samples],
            "pauses": [
                {
                    "pause_id": pause.pause_id,
                    "start_time_s": pause.start_time_s,
                    "end_time_s": pause.end_time_s,
                    "reason": pause.reason,
                }
                for pause in self.pauses
            ],
            "content_sha256": self.content_sha256,
        }

    def content_address(self) -> str:
        document = self.to_document()
        document.pop("content_sha256")
        return canonical_sha256(document, PLANNED_MOTION_CANONICAL_PROFILE)

    def sealed(self) -> PlannedMotion:
        return replace(self, content_sha256=self.content_address())


def planned_motion_from_document(value: object) -> PlannedMotion:
    expected = {
        "motion_id",
        "producer_id",
        "root_frame_id",
        "parameterization",
        "coordinate_unit",
        "source_artifacts",
        "tracks",
        "state_coordinates",
        "samples",
        "pauses",
        "content_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise PlannedMotionError("planned motion fields differ from the contract")
    try:
        parameterization = MotionParameterization(value["parameterization"])
    except (TypeError, ValueError) as error:
        raise PlannedMotionError("planned motion parameterization is invalid") from error
    raw_sources = value["source_artifacts"]
    if not isinstance(raw_sources, list):
        raise PlannedMotionError("source_artifacts must be an array")
    sources = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict) or set(raw) != {"role", "artifact_id", "sha256"}:
            raise PlannedMotionError(f"source_artifact[{index}] is invalid")
        sources.append(
            MotionSourceArtifact(raw["role"], raw["artifact_id"], raw["sha256"])
        )
    raw_tracks = value["tracks"]
    if not isinstance(raw_tracks, list):
        raise PlannedMotionError("tracks must be an array")
    track_keys = {"track_id", "component_id", "frame_id", "interpolation"}
    tracks = []
    for index, raw in enumerate(raw_tracks):
        if not isinstance(raw, dict) or set(raw) != track_keys:
            raise PlannedMotionError(f"track[{index}] fields differ from the contract")
        tracks.append(
            MotionTrack(
                track_id=raw["track_id"],
                component_id=raw["component_id"],
                frame_id=raw["frame_id"],
                interpolation=_semantics_from_document(
                    raw["interpolation"], f"track[{index}].interpolation"
                ),
            )
        )
    raw_coordinates = value["state_coordinates"]
    if not isinstance(raw_coordinates, list):
        raise PlannedMotionError("state_coordinates must be an array")
    state_coordinates = []
    for index, raw in enumerate(raw_coordinates):
        if not isinstance(raw, dict) or set(raw) != {"coordinate_id", "unit"}:
            raise PlannedMotionError(f"state_coordinate[{index}] is invalid")
        state_coordinates.append(MotionStateCoordinate(raw["coordinate_id"], raw["unit"]))
    raw_samples = value["samples"]
    if not isinstance(raw_samples, list):
        raise PlannedMotionError("samples must be an array")
    sample_keys = {
        "sample_id",
        "coordinate",
        "path_progress",
        "track_poses",
        "source_state_values",
        "material_feed_length_mm",
        "stage_id",
    }
    samples = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict) or set(raw) != sample_keys:
            raise PlannedMotionError(f"sample[{index}] fields differ from the contract")
        raw_poses = raw["track_poses"]
        if not isinstance(raw_poses, list):
            raise PlannedMotionError(f"sample[{index}].track_poses must be an array")
        track_poses = []
        for pose_index, pose_raw in enumerate(raw_poses):
            if not isinstance(pose_raw, dict) or set(pose_raw) != {
                "track_id",
                "parent_from_track",
            }:
                raise PlannedMotionError(
                    f"sample[{index}].track_pose[{pose_index}] is invalid"
                )
            track_poses.append(
                TrackPose(
                    track_id=pose_raw["track_id"],
                    parent_from_track=pose_from_document(
                        pose_raw["parent_from_track"],
                        f"sample[{index}].track_pose[{pose_index}]",
                    ),
                )
            )
        if not isinstance(raw["source_state_values"], list):
            raise PlannedMotionError(f"sample[{index}].source_state_values must be an array")
        samples.append(
            PlannedMotionSample(
                sample_id=raw["sample_id"],
                coordinate=raw["coordinate"],
                path_progress=raw["path_progress"],
                track_poses=tuple(track_poses),
                source_state_values=tuple(raw["source_state_values"]),
                material_feed_length_mm=raw["material_feed_length_mm"],
                stage_id=raw["stage_id"],
            )
        )
    raw_pauses = value["pauses"]
    if not isinstance(raw_pauses, list):
        raise PlannedMotionError("pauses must be an array")
    pauses = []
    for index, raw in enumerate(raw_pauses):
        if not isinstance(raw, dict) or set(raw) != {
            "pause_id",
            "start_time_s",
            "end_time_s",
            "reason",
        }:
            raise PlannedMotionError(f"pause[{index}] fields differ from the contract")
        pauses.append(
            PauseInterval(
                pause_id=raw["pause_id"],
                start_time_s=raw["start_time_s"],
                end_time_s=raw["end_time_s"],
                reason=raw["reason"],
            )
        )
    motion = PlannedMotion(
        motion_id=value["motion_id"],
        producer_id=value["producer_id"],
        root_frame_id=value["root_frame_id"],
        parameterization=parameterization,
        coordinate_unit=value["coordinate_unit"],
        source_artifacts=tuple(sources),
        tracks=tuple(tracks),
        state_coordinates=tuple(state_coordinates),
        samples=tuple(samples),
        pauses=tuple(pauses),
        content_sha256=value["content_sha256"],
    )
    if motion.to_document() != value:
        raise PlannedMotionError("planned motion is not canonical")
    return motion


__all__ = [
    "MotionParameterization",
    "MotionStateCoordinate",
    "MotionSourceArtifact",
    "MotionTrack",
    "PLANNED_MOTION_CANONICAL_PROFILE",
    "PlannedMotion",
    "PlannedMotionError",
    "PlannedMotionSample",
    "TrackPose",
    "planned_motion_from_document",
]
