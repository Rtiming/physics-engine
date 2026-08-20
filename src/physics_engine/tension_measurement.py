"""测力轮张力测量物理——两侧张力不是传感器读数。

本模块补[决策0096]与[plans/19 P1]点名的断层：

``span tension``
→ 两侧带材对测力轮的矢量合力
→ 导轮自重/tare与支承分配
→ LTS敏感轴分量
→ tare清零后既有``drives.TensionSensor``的mV/ADC
→ 标定后的显示张力。

**旧``TensionSensor``一个字不改。** 它继续回答“一个已经映射到传感器输入的正标量
怎样变成mV与量化值”；本模块负责此前缺的力学半边。二者组合而不互相扩义，旧产物
因此可以逐位保持不变。

## 方向约定

``incoming_tangent_xyz``与``outgoing_tangent_xyz``都取**材料行进方向**。
上游带段从测力轮向来料侧延伸，所以它对轮的拉力是``-T_in*t_in``；下游带段
从轮向出料侧延伸，所以拉力是``+T_out*t_out``：

    F_web = -T_in*t_in + T_out*t_out

两侧等张力、行进方向转角为``beta``时，合力模长退化为
``2*T*sin(beta/2)``。Maxcess/ABB/FMS的测力轮选型与标定图使用同一关系，完整出处见
``docs/research/19``第六节。

## 证据边界

本模块不假装知道现场LTS是单支承还是双支承、敏感轴朝向、导轮/轴承重量或VR451比例。
这些量由``MeasuringRoll``显式输入；没有标定时样点资格恒为``hypothesis_only``。
即使几何与标定来自实测，本模块产出的仍是``calibrated_model``，不是直接硬件测量。
当前电气组合明确从``net_axis_force_n``起步，所以``zeroed_bridge_output_mv``是tare已经
清零后的理想信号；原始gross桥路、tare扣除位置和gross载荷下的物理过载属于T-M2，
本片不编造现场电气拓扑。

## 面

``TensionMeasurementSample``落盘时使用``tension_measurement_sample/0.1``draft面；
文档包含全部输入和派生量，读取端会重算并逐字段比对。手改一个结果，即使重新计算
自指内容哈希，也会因派生量不闭合被拒。

[决策0096]: ../../docs/decisions/0096_绕制偏差物理基础设施归本仓与WDS单向消费边界_20260820.md
[plans/19 P1]: ../../docs/plans/19_绕制偏差物理基础设施与张力测量小场景计划_20260820.md
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any

from physics_engine.canonical import WDS_PROFILE, canonical_sha256, strict_loads
from physics_engine.drives import DriveError, TensionSensor
from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    TENSION_MEASUREMENT_SAMPLE_FACET,
    TENSION_MEASUREMENT_SAMPLE_VERSION,
)
from physics_engine.identity import IdentityError, parse_namespace_id
from physics_engine.materials import EvidenceRef

Vector3 = tuple[float, float, float]

TENSION_MEASUREMENT_CANONICAL_PROFILE = WDS_PROFILE
VECTOR_ABS_TOL = 1.0e-9
AXIS_FORCE_ABS_TOL_N = 1.0e-12
MEASUREMENT_EVIDENCE_GRADES = frozenset(
    {"measured", "calibrated", "derived", "manufacturer", "estimated"}
)

_TOP_KEYS = frozenset(
    {
        "facet",
        "facet_version",
        "measurement_id",
        "qualification",
        "incoming_tension_n",
        "outgoing_tension_n",
        "incoming_tangent_xyz",
        "outgoing_tangent_xyz",
        "web_force_n_xyz",
        "tare_force_n_xyz",
        "gross_force_n_xyz",
        "sensor_axis_xyz",
        "gross_axis_force_n",
        "tare_axis_force_n",
        "net_axis_force_n",
        "support_shares",
        "support_gross_forces_n",
        "support_tare_forces_n",
        "support_net_forces_n",
        "evidence",
        "calibration_id",
        "uncertainty_n",
        "transducer",
        "zeroed_bridge_output_mv",
        "digitized_net_axis_force_n",
        "displayed_span_tension_n",
        "is_zeroed_model_saturated",
        "content_sha256",
    }
)
_EVIDENCE_KEYS = frozenset({"grade", "evidence_id", "method", "source_sha256"})
_TRANSDUCER_KEYS = frozenset(
    {
        "full_scale_force_n",
        "output_at_full_scale_mv",
        "adc_bits",
        "sensor_force_per_span_tension_gain",
    }
)


class MeasurementError(ValueError):
    """张力测量物理与字节形制的一切失败关闭。"""


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{name} must be numeric, not {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise MeasurementError(f"{name} must be finite: {value!r}")
    return result


def _require_nonnegative(value: object, name: str) -> float:
    result = _require_finite(value, name)
    if result < 0.0:
        raise MeasurementError(f"{name} must be nonnegative: {value!r}")
    return result


def _require_positive(value: object, name: str) -> float:
    result = _require_finite(value, name)
    if result <= 0.0:
        raise MeasurementError(f"{name} must be positive: {value!r}")
    return result


def _require_vector(value: object, name: str) -> Vector3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise MeasurementError(f"{name} must contain exactly three components: {value!r}")
    return tuple(_require_finite(component, f"{name}[{axis}]") for axis, component in enumerate(value))  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _require_unit_vector(value: object, name: str) -> Vector3:
    vector = _require_vector(value, name)
    deviation = abs(_norm(vector) - 1.0)
    if deviation > VECTOR_ABS_TOL:
        raise MeasurementError(
            f"{name} must be a unit vector; |norm-1|={deviation!r} exceeds "
            f"{VECTOR_ABS_TOL!r}"
        )
    return vector


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _scale(scale: float, vector: Vector3) -> Vector3:
    return tuple(scale * component for component in vector)  # type: ignore[return-value]


def _require_namespace(value: object, namespace: str, name: str) -> str:
    if not isinstance(value, str):
        raise MeasurementError(f"{name} must be a string: {value!r}")
    try:
        parsed, _ = parse_namespace_id(value)
    except IdentityError as error:
        raise MeasurementError(f"{name} is not a valid namespace id: {error}") from error
    if parsed != namespace:
        raise MeasurementError(f"{name} must live in {namespace!r}: {value!r}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MeasurementError(f"{name} must be 64 lowercase hex characters")
    return value


def _exact_keys(document: dict[str, Any], expected: frozenset[str], where: str) -> None:
    unknown = sorted(set(document) - expected)
    missing = sorted(expected - set(document))
    if unknown:
        raise MeasurementError(f"unknown keys in {where}: {unknown}")
    if missing:
        raise MeasurementError(f"missing keys in {where}: {missing}")


def web_force_on_roll_n(
    *,
    incoming_tension_n: float,
    outgoing_tension_n: float,
    incoming_tangent_xyz: Vector3,
    outgoing_tangent_xyz: Vector3,
) -> Vector3:
    """两侧带材对测力轮的矢量合力，方向约定见模块文档。"""

    incoming_tension = _require_nonnegative(incoming_tension_n, "incoming_tension_n")
    outgoing_tension = _require_nonnegative(outgoing_tension_n, "outgoing_tension_n")
    incoming = _require_unit_vector(incoming_tangent_xyz, "incoming_tangent_xyz")
    outgoing = _require_unit_vector(outgoing_tangent_xyz, "outgoing_tangent_xyz")
    return _add(_scale(-incoming_tension, incoming), _scale(outgoing_tension, outgoing))


def equal_tension_resultant_force_n(*, tension_n: float, wrap_angle_rad: float) -> float:
    """等张力测力轮闭式``2*T*sin(beta/2)``，``beta``限0—π。"""

    tension = _require_nonnegative(tension_n, "tension_n")
    angle = _require_finite(wrap_angle_rad, "wrap_angle_rad")
    if not 0.0 <= angle <= math.pi:
        raise MeasurementError(f"wrap_angle_rad must be in [0, pi]: {wrap_angle_rad!r}")
    return 2.0 * tension * math.sin(0.5 * angle)


@dataclass(frozen=True)
class MeasuringRoll:
    """测力轮几何、tare、支承和证据；每个未知量都必须显式。"""

    measurement_id: str
    sensor_axis_xyz: Vector3
    tare_force_n_xyz: Vector3
    support_shares: tuple[float, ...]
    evidence: EvidenceRef
    calibration_id: str | None = None
    uncertainty_n: float | None = None

    def __post_init__(self) -> None:
        _require_namespace(self.measurement_id, "measurement", "measurement_id")
        _require_unit_vector(self.sensor_axis_xyz, "sensor_axis_xyz")
        _require_vector(self.tare_force_n_xyz, "tare_force_n_xyz")
        if not self.support_shares:
            raise MeasurementError("support_shares must name at least one support")
        shares = tuple(
            _require_nonnegative(value, f"support_shares[{index}]")
            for index, value in enumerate(self.support_shares)
        )
        if any(value > 1.0 for value in shares) or not math.isclose(
            sum(shares), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise MeasurementError(
                f"support_shares must be nonnegative, <=1, and sum to 1: {shares!r}"
            )
        if self.evidence.grade not in MEASUREMENT_EVIDENCE_GRADES:
            raise MeasurementError(
                f"measurement evidence grade {self.evidence.grade!r} is not supported"
            )
        calibrated = self.evidence.grade in {"calibrated", "measured"}
        if calibrated:
            if self.calibration_id is None or self.uncertainty_n is None:
                raise MeasurementError(
                    "calibrated/measured geometry needs calibration_id and uncertainty_n"
                )
            _require_namespace(self.calibration_id, "calibration", "calibration_id")
            _require_nonnegative(self.uncertainty_n, "uncertainty_n")
        elif self.calibration_id is not None or self.uncertainty_n is not None:
            raise MeasurementError(
                "uncalibrated geometry must not carry calibration_id or uncertainty_n"
            )

    @property
    def qualification(self) -> str:
        return (
            "calibrated_model"
            if self.evidence.grade in {"calibrated", "measured"}
            else "hypothesis_only"
        )

    def measure(
        self,
        *,
        incoming_tension_n: float,
        outgoing_tension_n: float,
        incoming_tangent_xyz: Vector3,
        outgoing_tangent_xyz: Vector3,
        transducer: TensionSensor | None = None,
        sensor_force_per_span_tension_gain: float | None = None,
        facet_version: str = TENSION_MEASUREMENT_SAMPLE_VERSION,
    ) -> TensionMeasurementSample:
        """从物理张力生成一份分层测量样点；不把中间层压成一个`tension`。"""

        ENGINE_REGISTRY.assert_reader_compatible(
            TENSION_MEASUREMENT_SAMPLE_FACET, facet_version
        )
        incoming = _require_unit_vector(incoming_tangent_xyz, "incoming_tangent_xyz")
        outgoing = _require_unit_vector(outgoing_tangent_xyz, "outgoing_tangent_xyz")
        incoming_tension = _require_nonnegative(incoming_tension_n, "incoming_tension_n")
        outgoing_tension = _require_nonnegative(outgoing_tension_n, "outgoing_tension_n")
        axis = _require_unit_vector(self.sensor_axis_xyz, "sensor_axis_xyz")
        tare = _require_vector(self.tare_force_n_xyz, "tare_force_n_xyz")
        web = web_force_on_roll_n(
            incoming_tension_n=incoming_tension,
            outgoing_tension_n=outgoing_tension,
            incoming_tangent_xyz=incoming,
            outgoing_tangent_xyz=outgoing,
        )
        gross = _add(web, tare)
        gross_axis = _dot(gross, axis)
        tare_axis = _dot(tare, axis)
        net_axis = gross_axis - tare_axis

        bridge_output: float | None = None
        digitized_force: float | None = None
        displayed_tension: float | None = None
        saturated: bool | None = None
        gain: float | None = None
        if transducer is None:
            if sensor_force_per_span_tension_gain is not None:
                raise MeasurementError(
                    "sensor_force_per_span_tension_gain needs a declared transducer"
                )
        else:
            if sensor_force_per_span_tension_gain is None:
                raise MeasurementError(
                    "a transducer needs sensor_force_per_span_tension_gain; "
                    "sensor force is not automatically web tension"
                )
            gain = _require_positive(
                sensor_force_per_span_tension_gain,
                "sensor_force_per_span_tension_gain",
            )
            if net_axis < -AXIS_FORCE_ABS_TOL_N:
                raise MeasurementError(
                    f"net force {net_axis!r} N opposes the declared positive sensor axis"
                )
            sensor_force = max(0.0, net_axis)
            bridge_output = transducer.millivolts(sensor_force)
            digitized_force = transducer.read_n(sensor_force)
            displayed_tension = digitized_force / gain
            saturated = sensor_force > transducer.full_scale_n

        shares = tuple(float(value) for value in self.support_shares)
        return TensionMeasurementSample(
            facet_version=facet_version,
            measurement_id=self.measurement_id,
            qualification=self.qualification,
            incoming_tension_n=incoming_tension,
            outgoing_tension_n=outgoing_tension,
            incoming_tangent_xyz=incoming,
            outgoing_tangent_xyz=outgoing,
            web_force_n_xyz=web,
            tare_force_n_xyz=tare,
            gross_force_n_xyz=gross,
            sensor_axis_xyz=axis,
            gross_axis_force_n=gross_axis,
            tare_axis_force_n=tare_axis,
            net_axis_force_n=net_axis,
            support_shares=shares,
            support_gross_forces_n=tuple(gross_axis * share for share in shares),
            support_tare_forces_n=tuple(tare_axis * share for share in shares),
            support_net_forces_n=tuple(net_axis * share for share in shares),
            evidence=self.evidence,
            calibration_id=self.calibration_id,
            uncertainty_n=self.uncertainty_n,
            transducer=transducer,
            sensor_force_per_span_tension_gain=gain,
            zeroed_bridge_output_mv=bridge_output,
            digitized_net_axis_force_n=digitized_force,
            displayed_span_tension_n=displayed_tension,
            is_zeroed_model_saturated=saturated,
        )


@dataclass(frozen=True)
class TensionMeasurementSample:
    """完整分层样点；跨边界读取必须走``load_tension_measurement_sample``。"""

    facet_version: str
    measurement_id: str
    qualification: str
    incoming_tension_n: float
    outgoing_tension_n: float
    incoming_tangent_xyz: Vector3
    outgoing_tangent_xyz: Vector3
    web_force_n_xyz: Vector3
    tare_force_n_xyz: Vector3
    gross_force_n_xyz: Vector3
    sensor_axis_xyz: Vector3
    gross_axis_force_n: float
    tare_axis_force_n: float
    net_axis_force_n: float
    support_shares: tuple[float, ...]
    support_gross_forces_n: tuple[float, ...]
    support_tare_forces_n: tuple[float, ...]
    support_net_forces_n: tuple[float, ...]
    evidence: EvidenceRef
    calibration_id: str | None
    uncertainty_n: float | None
    transducer: TensionSensor | None
    sensor_force_per_span_tension_gain: float | None
    zeroed_bridge_output_mv: float | None
    digitized_net_axis_force_n: float | None
    displayed_span_tension_n: float | None
    is_zeroed_model_saturated: bool | None
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        ENGINE_REGISTRY.assert_reader_compatible(
            TENSION_MEASUREMENT_SAMPLE_FACET, self.facet_version
        )
        _require_namespace(self.measurement_id, "measurement", "measurement_id")
        if self.qualification not in {"hypothesis_only", "calibrated_model"}:
            raise MeasurementError(f"unknown qualification: {self.qualification!r}")
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
            if self.content_sha256 != self.content_address():
                raise MeasurementError("content_sha256 does not match the measurement document")

    def to_document(self) -> dict[str, Any]:
        transducer = None
        if self.transducer is not None:
            transducer = {
                "full_scale_force_n": self.transducer.full_scale_n,
                "output_at_full_scale_mv": self.transducer.output_at_full_scale_mv,
                "adc_bits": self.transducer.adc_bits,
                "sensor_force_per_span_tension_gain": self.sensor_force_per_span_tension_gain,
            }
        return {
            "facet": TENSION_MEASUREMENT_SAMPLE_FACET,
            "facet_version": self.facet_version,
            "measurement_id": self.measurement_id,
            "qualification": self.qualification,
            "incoming_tension_n": self.incoming_tension_n,
            "outgoing_tension_n": self.outgoing_tension_n,
            "incoming_tangent_xyz": list(self.incoming_tangent_xyz),
            "outgoing_tangent_xyz": list(self.outgoing_tangent_xyz),
            "web_force_n_xyz": list(self.web_force_n_xyz),
            "tare_force_n_xyz": list(self.tare_force_n_xyz),
            "gross_force_n_xyz": list(self.gross_force_n_xyz),
            "sensor_axis_xyz": list(self.sensor_axis_xyz),
            "gross_axis_force_n": self.gross_axis_force_n,
            "tare_axis_force_n": self.tare_axis_force_n,
            "net_axis_force_n": self.net_axis_force_n,
            "support_shares": list(self.support_shares),
            "support_gross_forces_n": list(self.support_gross_forces_n),
            "support_tare_forces_n": list(self.support_tare_forces_n),
            "support_net_forces_n": list(self.support_net_forces_n),
            "evidence": self.evidence.to_document(),
            "calibration_id": self.calibration_id,
            "uncertainty_n": self.uncertainty_n,
            "transducer": transducer,
            "zeroed_bridge_output_mv": self.zeroed_bridge_output_mv,
            "digitized_net_axis_force_n": self.digitized_net_axis_force_n,
            "displayed_span_tension_n": self.displayed_span_tension_n,
            "is_zeroed_model_saturated": self.is_zeroed_model_saturated,
            "content_sha256": self.content_sha256,
        }

    def content_address(self) -> str:
        document = self.to_document()
        document.pop("content_sha256")
        return canonical_sha256(document, TENSION_MEASUREMENT_CANONICAL_PROFILE)

    def sealed(self) -> TensionMeasurementSample:
        return replace(self, content_sha256=self.content_address())


def _parse_evidence(document: object) -> EvidenceRef:
    if not isinstance(document, dict):
        raise MeasurementError("evidence must be an object")
    _exact_keys(document, _EVIDENCE_KEYS, "evidence")
    try:
        return EvidenceRef(
            grade=document["grade"],
            evidence_id=document["evidence_id"],
            method=document["method"],
            source_sha256=document["source_sha256"],
        )
    except ValueError as error:
        raise MeasurementError(f"invalid measurement evidence: {error}") from error


def load_tension_measurement_sample(
    payload: bytes, *, expected_file_sha256: str | None = None
) -> TensionMeasurementSample:
    """严格读取、验面/自指哈希并重算全部派生量。"""

    if expected_file_sha256 is not None:
        _require_sha256(expected_file_sha256, "expected_file_sha256")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_file_sha256:
            raise MeasurementError(
                f"locked measurement bytes changed: expected {expected_file_sha256}, got {actual}"
            )
    document = strict_loads(payload)
    if not isinstance(document, dict):
        raise MeasurementError("measurement payload must contain one object")
    _exact_keys(document, _TOP_KEYS, "tension measurement sample")
    if document["facet"] != TENSION_MEASUREMENT_SAMPLE_FACET:
        raise MeasurementError(
            f"facet must be {TENSION_MEASUREMENT_SAMPLE_FACET!r}: {document['facet']!r}"
        )
    if not isinstance(document["facet_version"], str):
        raise MeasurementError("facet_version must be a string")
    version = document["facet_version"]
    ENGINE_REGISTRY.assert_reader_compatible(TENSION_MEASUREMENT_SAMPLE_FACET, version)
    content_sha256 = _require_sha256(document["content_sha256"], "content_sha256")
    address_input = dict(document)
    address_input.pop("content_sha256")
    if canonical_sha256(address_input, TENSION_MEASUREMENT_CANONICAL_PROFILE) != content_sha256:
        raise MeasurementError("content_sha256 does not match the measurement document")

    transducer_document = document["transducer"]
    transducer: TensionSensor | None = None
    gain: float | None = None
    if transducer_document is not None:
        if not isinstance(transducer_document, dict):
            raise MeasurementError("transducer must be an object or null")
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
            raise MeasurementError(f"invalid measurement transducer: {error}") from error
        gain = _require_positive(
            transducer_document["sensor_force_per_span_tension_gain"],
            "sensor_force_per_span_tension_gain",
        )

    if not isinstance(document["support_shares"], list):
        raise MeasurementError("support_shares must be an array")
    roll = MeasuringRoll(
        measurement_id=document["measurement_id"],
        sensor_axis_xyz=_require_vector(document["sensor_axis_xyz"], "sensor_axis_xyz"),
        tare_force_n_xyz=_require_vector(
            document["tare_force_n_xyz"], "tare_force_n_xyz"
        ),
        support_shares=tuple(document["support_shares"]),
        evidence=_parse_evidence(document["evidence"]),
        calibration_id=document["calibration_id"],
        uncertainty_n=document["uncertainty_n"],
    )
    recomputed = roll.measure(
        incoming_tension_n=document["incoming_tension_n"],
        outgoing_tension_n=document["outgoing_tension_n"],
        incoming_tangent_xyz=_require_vector(
            document["incoming_tangent_xyz"], "incoming_tangent_xyz"
        ),
        outgoing_tangent_xyz=_require_vector(
            document["outgoing_tangent_xyz"], "outgoing_tangent_xyz"
        ),
        transducer=transducer,
        sensor_force_per_span_tension_gain=gain,
        facet_version=version,
    ).sealed()
    if recomputed.to_document() != document:
        raise MeasurementError(
            "derived measurement fields do not match recomputation from the declared inputs"
        )
    return recomputed


__all__ = [
    "AXIS_FORCE_ABS_TOL_N",
    "MEASUREMENT_EVIDENCE_GRADES",
    "TENSION_MEASUREMENT_CANONICAL_PROFILE",
    "MeasurementError",
    "MeasuringRoll",
    "TensionMeasurementSample",
    "equal_tension_resultant_force_n",
    "load_tension_measurement_sample",
    "web_force_on_roll_n",
]
