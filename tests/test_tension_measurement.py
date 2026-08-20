"""测力轮张力测量物理：矢量合力、敏感轴、tare、支承与电气链。"""

from __future__ import annotations

import json
import math

import pytest

from physics_engine.canonical import canonical_sha256
from physics_engine.drives import TensionSensor
from physics_engine.facets import FacetError
from physics_engine.materials import EvidenceRef
from physics_engine.tension_measurement import (
    TENSION_MEASUREMENT_CANONICAL_PROFILE,
    MeasurementError,
    MeasuringRoll,
    equal_tension_resultant_force_n,
    load_tension_measurement_sample,
    web_force_on_roll_n,
)

_SOURCE_SHA = "c" * 64


def _estimated() -> EvidenceRef:
    return EvidenceRef(
        grade="estimated",
        evidence_id="evidence/tension-roll-synthetic",
        method="Synthetic geometry for the closed-form measurement case.",
    )


def _calibrated() -> EvidenceRef:
    return EvidenceRef(
        grade="calibrated",
        evidence_id="evidence/tension-roll-five-point-calibration",
        method="Five-point forward/reverse calibration with an independent force reference.",
        source_sha256=_SOURCE_SHA,
    )


def _axis_for_equal_tension(beta_rad: float) -> tuple[float, float, float]:
    force = web_force_on_roll_n(
        incoming_tension_n=1.0,
        outgoing_tension_n=1.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(math.cos(beta_rad), math.sin(beta_rad), 0.0),
    )
    magnitude = math.sqrt(sum(value * value for value in force))
    return tuple(value / magnitude for value in force)  # type: ignore[return-value]


def _roll(
    *,
    beta_rad: float = math.pi / 2.0,
    tare_force_n_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    support_shares: tuple[float, ...] = (1.0,),
    evidence: EvidenceRef | None = None,
    calibration_id: str | None = None,
    uncertainty_n: float | None = None,
) -> MeasuringRoll:
    return MeasuringRoll(
        measurement_id="measurement/tension-roll-static",
        sensor_axis_xyz=_axis_for_equal_tension(beta_rad),
        tare_force_n_xyz=tare_force_n_xyz,
        support_shares=support_shares,
        evidence=evidence or _estimated(),
        calibration_id=calibration_id,
        uncertainty_n=uncertainty_n,
    )


@pytest.mark.parametrize(
    ("degrees", "factor"),
    ((0.0, 0.0), (30.0, 2.0 * math.sin(math.pi / 12.0)),
     (60.0, 1.0), (90.0, math.sqrt(2.0)), (180.0, 2.0)),
)
def test_equal_tension_resultant_matches_the_wrap_angle_closed_form(degrees, factor):
    tension_n = 17.0
    assert equal_tension_resultant_force_n(
        tension_n=tension_n, wrap_angle_rad=math.radians(degrees)
    ) == pytest.approx(tension_n * factor, rel=4.0e-16, abs=0.0)


def test_the_vector_sum_handles_unequal_tensions_without_inventing_one_tension():
    assert web_force_on_roll_n(
        incoming_tension_n=10.0,
        outgoing_tension_n=20.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
    ) == (-10.0, 20.0, 0.0)


def test_tare_and_two_symmetric_supports_are_kept_as_separate_layers():
    axis = _axis_for_equal_tension(math.pi / 2.0)
    tare = tuple(5.0 * value for value in axis)
    sample = _roll(tare_force_n_xyz=tare, support_shares=(0.5, 0.5)).measure(
        incoming_tension_n=10.0,
        outgoing_tension_n=10.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
    )
    expected_net = 10.0 * math.sqrt(2.0)
    assert sample.net_axis_force_n == pytest.approx(expected_net, rel=2.0e-16)
    assert sample.tare_axis_force_n == pytest.approx(5.0, rel=2.0e-16)
    assert sample.gross_axis_force_n == pytest.approx(expected_net + 5.0, rel=2.0e-16)
    assert sample.support_net_forces_n == pytest.approx((expected_net / 2.0,) * 2)
    assert sample.support_tare_forces_n == pytest.approx((2.5, 2.5))
    assert sample.qualification == "hypothesis_only"


def test_the_electrical_chain_consumes_sensor_axis_force_not_web_tension():
    transducer = TensionSensor(
        full_scale_n=49.033,
        output_at_full_scale_mv=20.0,
        adc_bits=12,
    )
    gain = math.sqrt(2.0)
    sample = _roll(
        evidence=_calibrated(),
        calibration_id="calibration/lts-five-point-20260820",
        uncertainty_n=0.05,
    ).measure(
        incoming_tension_n=10.0,
        outgoing_tension_n=10.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
        transducer=transducer,
        sensor_force_per_span_tension_gain=gain,
    )
    expected_force = 10.0 * gain
    assert sample.zeroed_bridge_output_mv == pytest.approx(
        expected_force / transducer.full_scale_n * 20.0, rel=2.0e-16
    )
    assert sample.digitized_net_axis_force_n == transducer.read_n(expected_force)
    assert sample.displayed_span_tension_n == pytest.approx(
        transducer.read_n(expected_force) / gain, rel=2.0e-16
    )
    assert sample.is_zeroed_model_saturated is False
    assert sample.qualification == "calibrated_model"


def test_sealed_sample_round_trips_and_recomputes_every_derived_quantity():
    sample = _roll(
        support_shares=(0.5, 0.5),
        evidence=_calibrated(),
        calibration_id="calibration/lts-five-point-20260820",
        uncertainty_n=0.05,
    ).measure(
        incoming_tension_n=10.0,
        outgoing_tension_n=10.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
        transducer=TensionSensor(49.033, 20.0, 12),
        sensor_force_per_span_tension_gain=math.sqrt(2.0),
    ).sealed()
    payload = json.dumps(sample.to_document(), ensure_ascii=False).encode()
    assert load_tension_measurement_sample(payload) == sample

    tampered = sample.to_document()
    tampered["net_axis_force_n"] += 0.25
    without_address = dict(tampered)
    without_address.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(
        without_address, TENSION_MEASUREMENT_CANONICAL_PROFILE
    )
    with pytest.raises(MeasurementError, match="derived measurement fields"):
        load_tension_measurement_sample(json.dumps(tampered).encode())


# --- 必须红：输入/证据/字节边界 -----------------------------------------


def test_red_non_unit_tangent_is_rejected():
    with pytest.raises(MeasurementError, match="incoming_tangent_xyz must be a unit"):
        web_force_on_roll_n(
            incoming_tension_n=10.0,
            outgoing_tension_n=10.0,
            incoming_tangent_xyz=(2.0, 0.0, 0.0),
            outgoing_tangent_xyz=(0.0, 1.0, 0.0),
        )


def test_red_negative_tension_is_rejected():
    with pytest.raises(MeasurementError, match="incoming_tension_n must be nonnegative"):
        web_force_on_roll_n(
            incoming_tension_n=-1.0,
            outgoing_tension_n=10.0,
            incoming_tangent_xyz=(1.0, 0.0, 0.0),
            outgoing_tangent_xyz=(0.0, 1.0, 0.0),
        )


def test_red_non_unit_sensor_axis_is_rejected():
    with pytest.raises(MeasurementError, match="sensor_axis_xyz must be a unit"):
        MeasuringRoll(
            measurement_id="measurement/bad-axis",
            sensor_axis_xyz=(2.0, 0.0, 0.0),
            tare_force_n_xyz=(0.0, 0.0, 0.0),
            support_shares=(1.0,),
            evidence=_estimated(),
        )


@pytest.mark.parametrize("shares", ((), (0.4, 0.4), (1.1, -0.1)))
def test_red_support_shares_must_be_explicit_nonnegative_and_sum_to_one(shares):
    with pytest.raises(MeasurementError, match="support_shares"):
        _roll(support_shares=shares)


def test_red_calibrated_geometry_needs_a_calibration_id_and_uncertainty():
    with pytest.raises(MeasurementError, match="calibrated/measured geometry"):
        _roll(evidence=_calibrated())


def test_red_unidirectional_transducer_rejects_force_against_its_positive_axis():
    roll = MeasuringRoll(
        measurement_id="measurement/reversed-axis",
        sensor_axis_xyz=tuple(-value for value in _axis_for_equal_tension(math.pi / 2.0)),
        tare_force_n_xyz=(0.0, 0.0, 0.0),
        support_shares=(1.0,),
        evidence=_estimated(),
    )
    with pytest.raises(MeasurementError, match="opposes the declared positive sensor axis"):
        roll.measure(
            incoming_tension_n=10.0,
            outgoing_tension_n=10.0,
            incoming_tangent_xyz=(1.0, 0.0, 0.0),
            outgoing_tangent_xyz=(0.0, 1.0, 0.0),
            transducer=TensionSensor(49.033, 20.0, 12),
            sensor_force_per_span_tension_gain=math.sqrt(2.0),
        )


def test_red_unknown_document_key_is_rejected():
    document = _roll().measure(
        incoming_tension_n=10.0,
        outgoing_tension_n=10.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
    ).sealed().to_document()
    document["surprise"] = 1
    with pytest.raises(MeasurementError, match="unknown keys"):
        load_tension_measurement_sample(json.dumps(document).encode())


def test_red_json_types_are_not_silently_stringified_or_iterated():
    document = _roll().measure(
        incoming_tension_n=10.0,
        outgoing_tension_n=10.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
    ).sealed().to_document()
    document["evidence"]["method"] = None
    without_address = dict(document)
    without_address.pop("content_sha256")
    document["content_sha256"] = canonical_sha256(
        without_address, TENSION_MEASUREMENT_CANONICAL_PROFILE
    )
    with pytest.raises(MeasurementError, match="invalid measurement evidence"):
        load_tension_measurement_sample(json.dumps(document).encode())


def test_red_facet_version_must_be_a_json_string():
    document = _roll().measure(
        incoming_tension_n=10.0,
        outgoing_tension_n=10.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
    ).sealed().to_document()
    document["facet_version"] = 0.1
    without_address = dict(document)
    without_address.pop("content_sha256")
    document["content_sha256"] = canonical_sha256(
        without_address, TENSION_MEASUREMENT_CANONICAL_PROFILE
    )
    with pytest.raises(MeasurementError, match="facet_version must be a string"):
        load_tension_measurement_sample(json.dumps(document).encode())


def test_red_untested_facet_minor_is_rejected():
    document = _roll().measure(
        incoming_tension_n=10.0,
        outgoing_tension_n=10.0,
        incoming_tangent_xyz=(1.0, 0.0, 0.0),
        outgoing_tangent_xyz=(0.0, 1.0, 0.0),
    ).sealed().to_document()
    document["facet_version"] = "0.9"
    without_address = dict(document)
    without_address.pop("content_sha256")
    document["content_sha256"] = canonical_sha256(
        without_address, TENSION_MEASUREMENT_CANONICAL_PROFILE
    )
    with pytest.raises(FacetError, match="tension_measurement_sample"):
        load_tension_measurement_sample(json.dumps(document).encode())
