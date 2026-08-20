"""T-M2张力电气读出——把“物理上受了多少力”变成“控制器看到多少张力”。

本模块接在``tension_measurement``之后，补四个此前不能含糊处理的层：

1. 原始gross桥路与tare桥路；
2. tare在ADC之前还是之后扣除；
3. 五个载荷等级、正反程共同拟合的线性标定，holdout永不进拟合；
4. 显式采样周期、整数样点时延和零阶保持。

``drives.TensionSensor``保持不变。它仍是理想单极性换能器；本模块通过组合明确
什么时候给它gross、tare或net力。gross过载始终单独记录，tare不能把物理过载“清零”。

本模块仍不复现ATC600内部电路。``TareMode``是两个可判候选，不是对现场结构的猜测；
现场结构未确认时，读出资格保持``hypothesis_only``。
"""

from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any

from physics_engine.canonical import WDS_PROFILE, canonical_sha256, strict_loads
from physics_engine.drives import DriveError, TensionSensor
from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    TENSION_READOUT_SAMPLE_FACET,
    TENSION_READOUT_SAMPLE_VERSION,
)
from physics_engine.identity import IdentityError, parse_namespace_id
from physics_engine.materials import EvidenceRef
from physics_engine.tension_measurement import (
    MeasuringRoll,
    TensionMeasurementSample,
    Vector3,
)

TENSION_READOUT_CANONICAL_PROFILE = WDS_PROFILE
FORCE_ABS_TOL_N = 1.0e-12
_QUALIFYING_GRADES = frozenset({"calibrated", "measured"})

_SAMPLE_KEYS = frozenset(
    {
        "facet",
        "facet_version",
        "qualification",
        "readout",
        "gross_axis_force_n",
        "tare_axis_force_n",
        "net_axis_force_n",
        "raw_bridge_output_mv",
        "tare_bridge_output_mv",
        "zeroed_bridge_output_mv",
        "digitized_gross_axis_force_n",
        "digitized_tare_axis_force_n",
        "digitized_net_axis_force_n",
        "displayed_span_tension_n",
        "is_gross_saturated",
        "is_zeroed_path_saturated",
        "content_sha256",
    }
)
_READOUT_KEYS = frozenset(
    {"readout_id", "tare_mode", "transducer", "calibration", "evidence"}
)
_TRANSDUCER_KEYS = frozenset(
    {"full_scale_force_n", "output_at_full_scale_mv", "adc_bits"}
)
_CALIBRATION_KEYS = frozenset(
    {
        "calibration_id",
        "points",
        "evidence",
        "uncertainty_n",
        "slope_span_per_sensor",
        "intercept_n",
        "fit_rms_error_n",
        "fit_max_abs_error_n",
        "hysteresis_max_n",
        "fit_point_ids",
        "reference_levels_n",
        "qualification",
    }
)
_POINT_KEYS = frozenset(
    {
        "point_id",
        "sensor_force_n",
        "reference_span_tension_n",
        "direction",
        "purpose",
    }
)
_EVIDENCE_KEYS = frozenset({"grade", "evidence_id", "method", "source_sha256"})


class ReadoutError(ValueError):
    """标定、电气读出、采样时钟与字节形制的一切失败关闭。"""


class CalibrationDirection(enum.StrEnum):
    INCREASING = "increasing"
    DECREASING = "decreasing"


class CalibrationPurpose(enum.StrEnum):
    FIT = "fit"
    HOLDOUT = "holdout"


class TareMode(enum.StrEnum):
    """tare在哪一层扣除；没有默认值。"""

    ANALOG_PRE_ADC = "analog_pre_adc"
    DIGITAL_POST_ADC = "digital_post_adc"


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReadoutError(f"{name} must be numeric, not {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ReadoutError(f"{name} must be finite: {value!r}")
    return result


def _require_nonnegative(value: object, name: str) -> float:
    result = _require_finite(value, name)
    if result < 0.0:
        raise ReadoutError(f"{name} must be nonnegative: {value!r}")
    return result


def _require_positive(value: object, name: str) -> float:
    result = _require_finite(value, name)
    if result <= 0.0:
        raise ReadoutError(f"{name} must be positive: {value!r}")
    return result


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReadoutError(f"{name} must be an int >= {minimum}: {value!r}")
    return value


def _require_namespace(value: object, namespace: str, name: str) -> str:
    if not isinstance(value, str):
        raise ReadoutError(f"{name} must be a string: {value!r}")
    try:
        parsed, _ = parse_namespace_id(value)
    except IdentityError as error:
        raise ReadoutError(f"{name} is not a valid namespace id: {error}") from error
    if parsed != namespace:
        raise ReadoutError(f"{name} must live in {namespace!r}: {value!r}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReadoutError(f"{name} must be 64 lowercase hex characters")
    return value


def _exact_keys(document: dict[str, Any], expected: frozenset[str], where: str) -> None:
    unknown = sorted(set(document) - expected)
    missing = sorted(expected - set(document))
    if unknown:
        raise ReadoutError(f"unknown keys in {where}: {unknown}")
    if missing:
        raise ReadoutError(f"missing keys in {where}: {missing}")


def _parse_enum(enum_type, value: object, name: str):
    if not isinstance(value, str):
        raise ReadoutError(f"{name} must be a string: {value!r}")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = [member.value for member in enum_type]
        raise ReadoutError(f"{name} must be one of {allowed}: {value!r}") from error


def _parse_evidence(document: object) -> EvidenceRef:
    if not isinstance(document, dict):
        raise ReadoutError("evidence must be an object")
    _exact_keys(document, _EVIDENCE_KEYS, "evidence")
    try:
        return EvidenceRef(
            grade=document["grade"],
            evidence_id=document["evidence_id"],
            method=document["method"],
            source_sha256=document["source_sha256"],
        )
    except ValueError as error:
        raise ReadoutError(f"invalid readout evidence: {error}") from error


@dataclass(frozen=True)
class TensionCalibrationPoint:
    point_id: str
    sensor_force_n: float
    reference_span_tension_n: float
    direction: CalibrationDirection
    purpose: CalibrationPurpose

    def __post_init__(self) -> None:
        _require_namespace(self.point_id, "calibration-point", "point_id")
        _require_nonnegative(self.sensor_force_n, "sensor_force_n")
        _require_nonnegative(self.reference_span_tension_n, "reference_span_tension_n")
        if not isinstance(self.direction, CalibrationDirection):
            raise ReadoutError(f"direction must be a CalibrationDirection: {self.direction!r}")
        if not isinstance(self.purpose, CalibrationPurpose):
            raise ReadoutError(f"purpose must be a CalibrationPurpose: {self.purpose!r}")

    def to_document(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "sensor_force_n": self.sensor_force_n,
            "reference_span_tension_n": self.reference_span_tension_n,
            "direction": self.direction.value,
            "purpose": self.purpose.value,
        }


@dataclass(frozen=True)
class TensionCalibrationCheck:
    point_id: str
    predicted_span_tension_n: float
    reference_span_tension_n: float
    error_n: float


@dataclass(frozen=True)
class LinearSpanCalibration:
    """传感器轴向力到跨段张力的线性标定；拟合点身份与残差一起保存。"""

    calibration_id: str
    points: tuple[TensionCalibrationPoint, ...]
    evidence: EvidenceRef
    uncertainty_n: float | None
    slope_span_per_sensor: float
    intercept_n: float
    fit_rms_error_n: float
    fit_max_abs_error_n: float
    hysteresis_max_n: float
    fit_point_ids: tuple[str, ...]
    reference_levels_n: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_namespace(self.calibration_id, "calibration", "calibration_id")
        if not self.points or not all(
            isinstance(point, TensionCalibrationPoint) for point in self.points
        ):
            raise ReadoutError("points must contain TensionCalibrationPoint values")
        if not isinstance(self.evidence, EvidenceRef):
            raise ReadoutError(f"evidence must be an EvidenceRef: {self.evidence!r}")
        _require_positive(self.slope_span_per_sensor, "slope_span_per_sensor")
        _require_finite(self.intercept_n, "intercept_n")
        _require_nonnegative(self.fit_rms_error_n, "fit_rms_error_n")
        _require_nonnegative(self.fit_max_abs_error_n, "fit_max_abs_error_n")
        _require_nonnegative(self.hysteresis_max_n, "hysteresis_max_n")
        if self.evidence.grade in _QUALIFYING_GRADES:
            if self.uncertainty_n is None:
                raise ReadoutError("calibrated/measured fit needs uncertainty_n")
            _require_nonnegative(self.uncertainty_n, "uncertainty_n")
        elif self.uncertainty_n is not None:
            raise ReadoutError("uncalibrated fit must not carry uncertainty_n")

    @property
    def qualification(self) -> str:
        return (
            "calibrated_model"
            if self.evidence.grade in _QUALIFYING_GRADES
            else "hypothesis_only"
        )

    @classmethod
    def fit(
        cls,
        *,
        calibration_id: str,
        points: tuple[TensionCalibrationPoint, ...],
        evidence: EvidenceRef,
        uncertainty_n: float | None,
    ) -> LinearSpanCalibration:
        _require_namespace(calibration_id, "calibration", "calibration_id")
        if not points:
            raise ReadoutError("calibration points must not be empty")
        if len({point.point_id for point in points}) != len(points):
            raise ReadoutError("calibration point IDs must be unique")
        holdouts = [point.point_id for point in points if point.purpose is not CalibrationPurpose.FIT]
        if holdouts:
            raise ReadoutError(
                f"holdout points must not contaminate the fit: {sorted(holdouts)}"
            )
        levels = tuple(sorted({float(point.reference_span_tension_n) for point in points}))
        if len(levels) < 5:
            raise ReadoutError(
                f"five distinct reference levels are required, got {len(levels)}"
            )
        both = {CalibrationDirection.INCREASING, CalibrationDirection.DECREASING}
        for level in levels:
            directions = {
                point.direction
                for point in points
                if float(point.reference_span_tension_n) == level
            }
            if directions != both:
                raise ReadoutError(
                    f"reference level {level!r} needs increasing and decreasing points"
                )

        xs = tuple(_require_nonnegative(point.sensor_force_n, "sensor_force_n") for point in points)
        ys = tuple(
            _require_nonnegative(point.reference_span_tension_n, "reference_span_tension_n")
            for point in points
        )
        mean_x = math.fsum(xs) / len(xs)
        mean_y = math.fsum(ys) / len(ys)
        variance = math.fsum((value - mean_x) ** 2 for value in xs)
        if variance <= 0.0:
            raise ReadoutError("calibration sensor forces need nonzero spread")
        covariance = math.fsum(
            (sensor - mean_x) * (reference - mean_y)
            for sensor, reference in zip(xs, ys, strict=True)
        )
        slope = covariance / variance
        if slope <= 0.0:
            raise ReadoutError(f"calibration slope must be positive: {slope!r}")
        intercept = mean_y - slope * mean_x
        residuals = tuple(
            slope * sensor + intercept - reference
            for sensor, reference in zip(xs, ys, strict=True)
        )
        rms = math.sqrt(math.fsum(value * value for value in residuals) / len(residuals))
        maximum = max(abs(value) for value in residuals)

        hysteresis = []
        for level in levels:
            increasing = tuple(
                point.sensor_force_n
                for point in points
                if point.reference_span_tension_n == level
                and point.direction is CalibrationDirection.INCREASING
            )
            decreasing = tuple(
                point.sensor_force_n
                for point in points
                if point.reference_span_tension_n == level
                and point.direction is CalibrationDirection.DECREASING
            )
            mean_increasing = math.fsum(increasing) / len(increasing)
            mean_decreasing = math.fsum(decreasing) / len(decreasing)
            hysteresis.append(abs(mean_increasing - mean_decreasing) * slope)

        return cls(
            calibration_id=calibration_id,
            points=tuple(points),
            evidence=evidence,
            uncertainty_n=uncertainty_n,
            slope_span_per_sensor=slope,
            intercept_n=intercept,
            fit_rms_error_n=rms,
            fit_max_abs_error_n=maximum,
            hysteresis_max_n=max(hysteresis),
            fit_point_ids=tuple(point.point_id for point in points),
            reference_levels_n=levels,
        )

    def predict_span_tension_n(self, sensor_force_n: float) -> float:
        force = _require_nonnegative(sensor_force_n, "sensor_force_n")
        return self.slope_span_per_sensor * force + self.intercept_n

    def evaluate_holdout(self, point: TensionCalibrationPoint) -> TensionCalibrationCheck:
        if point.purpose is not CalibrationPurpose.HOLDOUT:
            raise ReadoutError("evaluate_holdout requires a holdout point")
        predicted = self.predict_span_tension_n(point.sensor_force_n)
        return TensionCalibrationCheck(
            point_id=point.point_id,
            predicted_span_tension_n=predicted,
            reference_span_tension_n=point.reference_span_tension_n,
            error_n=predicted - point.reference_span_tension_n,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "points": [point.to_document() for point in self.points],
            "evidence": self.evidence.to_document(),
            "uncertainty_n": self.uncertainty_n,
            "slope_span_per_sensor": self.slope_span_per_sensor,
            "intercept_n": self.intercept_n,
            "fit_rms_error_n": self.fit_rms_error_n,
            "fit_max_abs_error_n": self.fit_max_abs_error_n,
            "hysteresis_max_n": self.hysteresis_max_n,
            "fit_point_ids": list(self.fit_point_ids),
            "reference_levels_n": list(self.reference_levels_n),
            "qualification": self.qualification,
        }


@dataclass(frozen=True)
class TensionReadout:
    """固定的电气/标定配置；采样与时延状态在``TensionReadoutChannel``。"""

    readout_id: str
    transducer: TensionSensor
    calibration: LinearSpanCalibration
    tare_mode: TareMode
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        _require_namespace(self.readout_id, "readout", "readout_id")
        if not isinstance(self.transducer, TensionSensor):
            raise ReadoutError(f"transducer must be a TensionSensor: {self.transducer!r}")
        if not isinstance(self.calibration, LinearSpanCalibration):
            raise ReadoutError(
                f"calibration must be a LinearSpanCalibration: {self.calibration!r}"
            )
        if not isinstance(self.tare_mode, TareMode):
            raise ReadoutError(f"tare_mode must be an explicit TareMode: {self.tare_mode!r}")
        if not isinstance(self.evidence, EvidenceRef):
            raise ReadoutError(f"evidence must be an EvidenceRef: {self.evidence!r}")

    @property
    def qualification(self) -> str:
        return (
            "calibrated_model"
            if self.calibration.qualification == "calibrated_model"
            and self.evidence.grade in _QUALIFYING_GRADES
            else "hypothesis_only"
        )

    def measure(
        self, *, gross_axis_force_n: float, tare_axis_force_n: float
    ) -> TensionReadoutSample:
        gross = _require_nonnegative(gross_axis_force_n, "gross_axis_force_n")
        tare = _require_nonnegative(tare_axis_force_n, "tare_axis_force_n")
        net = gross - tare
        if net < -FORCE_ABS_TOL_N:
            raise ReadoutError(
                f"gross_axis_force_n {gross!r} is below tare_axis_force_n {tare!r}"
            )
        net = max(0.0, net)
        raw_mv = self.transducer.millivolts(gross)
        tare_mv = self.transducer.millivolts(tare)
        digitized_gross: float | None = None
        digitized_tare: float | None = None
        if self.tare_mode is TareMode.ANALOG_PRE_ADC:
            zeroed_mv = self.transducer.millivolts(net)
            digitized_net = self.transducer.read_n(net)
            zeroed_saturated = net > self.transducer.full_scale_n
        else:
            zeroed_mv = raw_mv - tare_mv
            digitized_gross = self.transducer.read_n(gross)
            digitized_tare = self.transducer.read_n(tare)
            digitized_net = max(0.0, digitized_gross - digitized_tare)
            zeroed_saturated = (
                gross > self.transducer.full_scale_n
                or tare > self.transducer.full_scale_n
            )
        return TensionReadoutSample(
            readout=self,
            qualification=self.qualification,
            gross_axis_force_n=gross,
            tare_axis_force_n=tare,
            net_axis_force_n=net,
            raw_bridge_output_mv=raw_mv,
            tare_bridge_output_mv=tare_mv,
            zeroed_bridge_output_mv=zeroed_mv,
            digitized_gross_axis_force_n=digitized_gross,
            digitized_tare_axis_force_n=digitized_tare,
            digitized_net_axis_force_n=digitized_net,
            displayed_span_tension_n=self.calibration.predict_span_tension_n(
                digitized_net
            ),
            is_gross_saturated=gross > self.transducer.full_scale_n,
            is_zeroed_path_saturated=zeroed_saturated,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "readout_id": self.readout_id,
            "tare_mode": self.tare_mode.value,
            "transducer": {
                "full_scale_force_n": self.transducer.full_scale_n,
                "output_at_full_scale_mv": self.transducer.output_at_full_scale_mv,
                "adc_bits": self.transducer.adc_bits,
            },
            "calibration": self.calibration.to_document(),
            "evidence": self.evidence.to_document(),
        }


@dataclass(frozen=True)
class TensionReadoutSample:
    readout: TensionReadout
    qualification: str
    gross_axis_force_n: float
    tare_axis_force_n: float
    net_axis_force_n: float
    raw_bridge_output_mv: float
    tare_bridge_output_mv: float
    zeroed_bridge_output_mv: float
    digitized_gross_axis_force_n: float | None
    digitized_tare_axis_force_n: float | None
    digitized_net_axis_force_n: float
    displayed_span_tension_n: float
    is_gross_saturated: bool
    is_zeroed_path_saturated: bool
    facet_version: str = TENSION_READOUT_SAMPLE_VERSION
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        ENGINE_REGISTRY.assert_reader_compatible(TENSION_READOUT_SAMPLE_FACET, self.facet_version)
        if not isinstance(self.readout, TensionReadout):
            raise ReadoutError(f"readout must be a TensionReadout: {self.readout!r}")
        if self.qualification not in {"hypothesis_only", "calibrated_model"}:
            raise ReadoutError(f"unknown qualification: {self.qualification!r}")
        if self.qualification != self.readout.qualification:
            raise ReadoutError(
                f"qualification {self.qualification!r} does not match readout "
                f"qualification {self.readout.qualification!r}"
            )
        for name in ("is_gross_saturated", "is_zeroed_path_saturated"):
            if not isinstance(getattr(self, name), bool):
                raise ReadoutError(f"{name} must be a bool")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
            if self.content_sha256 != self.content_address():
                raise ReadoutError("content_sha256 does not match the readout document")

    def to_document(self) -> dict[str, Any]:
        return {
            "facet": TENSION_READOUT_SAMPLE_FACET,
            "facet_version": self.facet_version,
            "qualification": self.qualification,
            "readout": self.readout.to_document(),
            "gross_axis_force_n": self.gross_axis_force_n,
            "tare_axis_force_n": self.tare_axis_force_n,
            "net_axis_force_n": self.net_axis_force_n,
            "raw_bridge_output_mv": self.raw_bridge_output_mv,
            "tare_bridge_output_mv": self.tare_bridge_output_mv,
            "zeroed_bridge_output_mv": self.zeroed_bridge_output_mv,
            "digitized_gross_axis_force_n": self.digitized_gross_axis_force_n,
            "digitized_tare_axis_force_n": self.digitized_tare_axis_force_n,
            "digitized_net_axis_force_n": self.digitized_net_axis_force_n,
            "displayed_span_tension_n": self.displayed_span_tension_n,
            "is_gross_saturated": self.is_gross_saturated,
            "is_zeroed_path_saturated": self.is_zeroed_path_saturated,
            "content_sha256": self.content_sha256,
        }

    def content_address(self) -> str:
        document = self.to_document()
        document.pop("content_sha256")
        return canonical_sha256(document, TENSION_READOUT_CANONICAL_PROFILE)

    def sealed(self) -> TensionReadoutSample:
        return replace(self, content_sha256=self.content_address())


def _parse_point(document: object) -> TensionCalibrationPoint:
    if not isinstance(document, dict):
        raise ReadoutError("calibration point must be an object")
    _exact_keys(document, _POINT_KEYS, "calibration point")
    return TensionCalibrationPoint(
        point_id=document["point_id"],
        sensor_force_n=document["sensor_force_n"],
        reference_span_tension_n=document["reference_span_tension_n"],
        direction=_parse_enum(
            CalibrationDirection, document["direction"], "calibration direction"
        ),
        purpose=_parse_enum(
            CalibrationPurpose, document["purpose"], "calibration purpose"
        ),
    )


def _parse_calibration(document: object) -> LinearSpanCalibration:
    if not isinstance(document, dict):
        raise ReadoutError("calibration must be an object")
    _exact_keys(document, _CALIBRATION_KEYS, "calibration")
    points_document = document["points"]
    if not isinstance(points_document, list):
        raise ReadoutError("calibration points must be an array")
    rebuilt = LinearSpanCalibration.fit(
        calibration_id=document["calibration_id"],
        points=tuple(_parse_point(point) for point in points_document),
        evidence=_parse_evidence(document["evidence"]),
        uncertainty_n=document["uncertainty_n"],
    )
    if rebuilt.to_document() != document:
        raise ReadoutError("derived calibration fields do not match the fit points")
    return rebuilt


def _parse_readout(document: object) -> TensionReadout:
    if not isinstance(document, dict):
        raise ReadoutError("readout must be an object")
    _exact_keys(document, _READOUT_KEYS, "readout")
    transducer_document = document["transducer"]
    if not isinstance(transducer_document, dict):
        raise ReadoutError("transducer must be an object")
    _exact_keys(transducer_document, _TRANSDUCER_KEYS, "transducer")
    try:
        transducer = TensionSensor(
            full_scale_n=_require_positive(
                transducer_document["full_scale_force_n"], "full_scale_force_n"
            ),
            output_at_full_scale_mv=_require_positive(
                transducer_document["output_at_full_scale_mv"],
                "output_at_full_scale_mv",
            ),
            adc_bits=transducer_document["adc_bits"],
        )
    except DriveError as error:
        raise ReadoutError(f"invalid transducer: {error}") from error
    return TensionReadout(
        readout_id=document["readout_id"],
        transducer=transducer,
        calibration=_parse_calibration(document["calibration"]),
        tare_mode=_parse_enum(TareMode, document["tare_mode"], "tare_mode"),
        evidence=_parse_evidence(document["evidence"]),
    )


def load_tension_readout_sample(
    payload: bytes, *, expected_file_sha256: str | None = None
) -> TensionReadoutSample:
    """严格读取、自指验哈希，并从力与配置重算全部派生字段。"""

    if expected_file_sha256 is not None:
        _require_sha256(expected_file_sha256, "expected_file_sha256")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_file_sha256:
            raise ReadoutError(
                f"locked readout bytes changed: expected {expected_file_sha256}, got {actual}"
            )
    document = strict_loads(payload)
    if not isinstance(document, dict):
        raise ReadoutError("readout payload must contain one object")
    _exact_keys(document, _SAMPLE_KEYS, "tension readout sample")
    if document["facet"] != TENSION_READOUT_SAMPLE_FACET:
        raise ReadoutError(f"facet must be {TENSION_READOUT_SAMPLE_FACET!r}")
    if not isinstance(document["facet_version"], str):
        raise ReadoutError("facet_version must be a string")
    version = document["facet_version"]
    ENGINE_REGISTRY.assert_reader_compatible(TENSION_READOUT_SAMPLE_FACET, version)
    content_sha256 = _require_sha256(document["content_sha256"], "content_sha256")
    address_input = dict(document)
    address_input.pop("content_sha256")
    if canonical_sha256(address_input, TENSION_READOUT_CANONICAL_PROFILE) != content_sha256:
        raise ReadoutError("content_sha256 does not match the readout document")
    readout = _parse_readout(document["readout"])
    recomputed = readout.measure(
        gross_axis_force_n=document["gross_axis_force_n"],
        tare_axis_force_n=document["tare_axis_force_n"],
    ).sealed()
    if recomputed.to_document() != document:
        raise ReadoutError(
            "derived readout fields do not match recomputation from force and configuration"
        )
    return recomputed


@dataclass(frozen=True)
class TimedTensionReadoutSample:
    sampled_step_index: int
    mechanical: TensionMeasurementSample
    electrical: TensionReadoutSample

    def __post_init__(self) -> None:
        _require_int(self.sampled_step_index, "sampled_step_index", minimum=0)
        if not isinstance(self.mechanical, TensionMeasurementSample):
            raise ReadoutError("mechanical must be a TensionMeasurementSample")
        if not isinstance(self.electrical, TensionReadoutSample):
            raise ReadoutError("electrical must be a TensionReadoutSample")


@dataclass(frozen=True)
class TensionChannelSample:
    time_s: float
    sample_tick: bool
    measured_at_time_s: float
    mechanical: TensionMeasurementSample
    electrical: TensionReadoutSample

    @property
    def displayed_span_tension_n(self) -> float:
        return self.electrical.displayed_span_tension_n


@dataclass(frozen=True)
class TensionReadoutChannel:
    """plant时钟上的采样、整数样点时延与零阶保持。"""

    channel_id: str
    roll: MeasuringRoll
    incoming_tangent_xyz: Vector3
    outgoing_tangent_xyz: Vector3
    readout: TensionReadout
    plant_dt_s: float
    sample_decimation: int
    delay_samples: int
    pending: tuple[TimedTensionReadoutSample, ...]
    held: TimedTensionReadoutSample
    step_index: int = 0

    def __post_init__(self) -> None:
        _require_namespace(self.channel_id, "measurement-channel", "channel_id")
        if not isinstance(self.roll, MeasuringRoll):
            raise ReadoutError(f"roll must be a MeasuringRoll: {self.roll!r}")
        if not isinstance(self.readout, TensionReadout):
            raise ReadoutError(f"readout must be a TensionReadout: {self.readout!r}")
        _require_positive(self.plant_dt_s, "plant_dt_s")
        _require_int(self.sample_decimation, "sample_decimation", minimum=1)
        _require_int(self.delay_samples, "delay_samples", minimum=0)
        _require_int(self.step_index, "step_index", minimum=0)
        if not isinstance(self.held, TimedTensionReadoutSample):
            raise ReadoutError("held must be a TimedTensionReadoutSample")
        if not all(isinstance(sample, TimedTensionReadoutSample) for sample in self.pending):
            raise ReadoutError("pending must contain TimedTensionReadoutSample values")
        if len(self.pending) != self.delay_samples:
            raise ReadoutError(
                f"pending length {len(self.pending)} must equal delay_samples "
                f"{self.delay_samples}"
            )

    @property
    def sample_period_s(self) -> float:
        return self.sample_decimation * self.plant_dt_s

    @property
    def delay_s(self) -> float:
        return self.delay_samples * self.sample_period_s

    @staticmethod
    def _instantaneous_for(
        *,
        roll: MeasuringRoll,
        incoming_tangent_xyz: Vector3,
        outgoing_tangent_xyz: Vector3,
        readout: TensionReadout,
        span_tension_n: float,
        step_index: int,
    ) -> TimedTensionReadoutSample:
        mechanical = roll.measure(
            incoming_tension_n=span_tension_n,
            outgoing_tension_n=span_tension_n,
            incoming_tangent_xyz=incoming_tangent_xyz,
            outgoing_tangent_xyz=outgoing_tangent_xyz,
        )
        electrical = readout.measure(
            gross_axis_force_n=mechanical.gross_axis_force_n,
            tare_axis_force_n=mechanical.tare_axis_force_n,
        )
        return TimedTensionReadoutSample(
            sampled_step_index=step_index,
            mechanical=mechanical,
            electrical=electrical,
        )

    def _instantaneous(
        self, span_tension_n: float, step_index: int
    ) -> TimedTensionReadoutSample:
        return self._instantaneous_for(
            roll=self.roll,
            incoming_tangent_xyz=self.incoming_tangent_xyz,
            outgoing_tangent_xyz=self.outgoing_tangent_xyz,
            readout=self.readout,
            span_tension_n=span_tension_n,
            step_index=step_index,
        )

    @classmethod
    def at_steady_state(
        cls,
        *,
        channel_id: str,
        roll: MeasuringRoll,
        incoming_tangent_xyz: Vector3,
        outgoing_tangent_xyz: Vector3,
        readout: TensionReadout,
        plant_dt_s: float,
        sample_decimation: int,
        delay_samples: int,
        span_tension_n: float,
    ) -> TensionReadoutChannel:
        _require_positive(plant_dt_s, "plant_dt_s")
        _require_int(sample_decimation, "sample_decimation", minimum=1)
        _require_int(delay_samples, "delay_samples", minimum=0)
        initial = cls._instantaneous_for(
            roll=roll,
            incoming_tangent_xyz=incoming_tangent_xyz,
            outgoing_tangent_xyz=outgoing_tangent_xyz,
            readout=readout,
            span_tension_n=_require_nonnegative(span_tension_n, "span_tension_n"),
            step_index=0,
        )
        return cls(
            channel_id=channel_id,
            roll=roll,
            incoming_tangent_xyz=incoming_tangent_xyz,
            outgoing_tangent_xyz=outgoing_tangent_xyz,
            readout=readout,
            plant_dt_s=plant_dt_s,
            sample_decimation=sample_decimation,
            delay_samples=delay_samples,
            pending=tuple(initial for _ in range(delay_samples)),
            held=initial,
        )

    def advance(
        self, *, span_tension_n: float
    ) -> tuple[TensionReadoutChannel, TensionChannelSample]:
        tension = _require_nonnegative(span_tension_n, "span_tension_n")
        tick = self.step_index % self.sample_decimation == 0
        pending = self.pending
        held = self.held
        if tick:
            instantaneous = self._instantaneous(tension, self.step_index)
            if self.delay_samples == 0:
                held = instantaneous
            else:
                pending = (*pending[1:], instantaneous)
                held = self.pending[0]
        output = TensionChannelSample(
            time_s=self.step_index * self.plant_dt_s,
            sample_tick=tick,
            measured_at_time_s=held.sampled_step_index * self.plant_dt_s,
            mechanical=held.mechanical,
            electrical=held.electrical,
        )
        return (
            TensionReadoutChannel(
                channel_id=self.channel_id,
                roll=self.roll,
                incoming_tangent_xyz=self.incoming_tangent_xyz,
                outgoing_tangent_xyz=self.outgoing_tangent_xyz,
                readout=self.readout,
                plant_dt_s=self.plant_dt_s,
                sample_decimation=self.sample_decimation,
                delay_samples=self.delay_samples,
                pending=pending,
                held=held,
                step_index=self.step_index + 1,
            ),
            output,
        )


__all__ = [
    "CalibrationDirection",
    "CalibrationPurpose",
    "FORCE_ABS_TOL_N",
    "LinearSpanCalibration",
    "ReadoutError",
    "TENSION_READOUT_CANONICAL_PROFILE",
    "TareMode",
    "TensionCalibrationCheck",
    "TensionCalibrationPoint",
    "TensionChannelSample",
    "TensionReadout",
    "TensionReadoutChannel",
    "TensionReadoutSample",
    "TimedTensionReadoutSample",
    "load_tension_readout_sample",
]
