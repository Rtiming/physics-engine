"""材料记录的门——spec/14第六节六条"必须红"全在此，附真实参数记录。

绿例的参数取自WDS `config/materials/cu_c110_p9318__...__v002.material.json`
（Surepure 9318铜带，6.35mm×0.127mm，20°C），外加一条光学域字段，
用来验"一份记录聚合多域字段，各域各取所需"这句话是真的。
"""

from __future__ import annotations

import hashlib
import json

import pytest

from physics_engine.canonical import canonical_file_bytes
from physics_engine.facets import FacetError
from physics_engine.materials import (
    MATERIAL_CANONICAL_PROFILE,
    AppearanceRef,
    EvidenceRef,
    MaterialError,
    MaterialProperty,
    MaterialRecord,
    load_material_record,
    unit_suffix_of,
)

_SOURCE_REGISTER_SHA = "759ba78d99cf6f1bcd5acc0760cbb947a896a2b6808eee09684fbc4928a79caf"
_APPEARANCE_SHA = "50556eb4e7df6dc27fb2361710fa3419a938ccd8ef6f84ea822d937ab2eb973d"
_MATERIAL_ID = "material/cu_c110_p9318__w6p35_t0p127__20c__vendor_derived__v002"


def _evidence(
    grade: str = "manufacturer",
    name: str = "width",
    source: str | None = _SOURCE_REGISTER_SHA,
) -> EvidenceRef:
    return EvidenceRef(
        grade=grade,
        evidence_id=f"evidence/mat-src-005-p9318-{name}",
        method=(
            "Surepure product 9318 nominal geometry converted exactly from inches; "
            "no lot certificate or finished-strip test is claimed."
        ),
        source_sha256=source,
    )


def _properties() -> tuple[MaterialProperty, ...]:
    return (
        # 几何：力学要它算截面，光学要它算程差——同一个字段两个域。
        MaterialProperty(
            name="width_mm", value=6.35, domains=("mechanics",), evidence=_evidence()
        ),
        MaterialProperty(
            name="thickness_mm",
            value=0.127,
            domains=("mechanics", "optics"),
            evidence=_evidence(name="thickness"),
        ),
        MaterialProperty(
            name="density_kg_m3",
            value=8910.0,
            domains=("mechanics",),
            evidence=_evidence(name="density"),
        ),
        MaterialProperty(
            name="EI_easy_N_mm2",
            value=127.04908639095021,
            domains=("mechanics",),
            evidence=_evidence(grade="derived", name="ei-easy"),
        ),
        # 光学侧：反射率无量纲，必须显式登记（轴2规则5禁止留空装有）。
        MaterialProperty(
            name="specular_reflectance",
            value=0.94,
            domains=("optics",),
            evidence=_evidence(grade="estimated", name="reflectance", source=None),
        ),
        # 未测量的量：FTS的`unset`语义——不给占位数，给None。
        MaterialProperty(
            name="axial_damping_n_s_mm",
            value=None,
            domains=("mechanics",),
            evidence=EvidenceRef(
                grade="unset",
                evidence_id="evidence/cu-c110-p9318-damping-unmeasured",
                method=(
                    "Neither the product page nor a decay test supplies damping; "
                    "recorded as unset rather than as a zero that reads like a claim."
                ),
            ),
        ),
    )


def _record(**overrides) -> MaterialRecord:
    fields: dict = {
        "material_id": _MATERIAL_ID,
        "applicable_domains": ("mechanics", "optics", "appearance"),
        "properties": _properties(),
        "length_unit": "mm",
        "appearance": AppearanceRef(
            asset_id="appearance/cu-c110-brushed",
            path_relative="appearance/cu_c110_brushed.material.json",
            sha256=_APPEARANCE_SHA,
        ),
        "dimensionless": frozenset({"specular_reflectance"}),
    }
    fields.update(overrides)
    return MaterialRecord(**fields)


# --- spec/14 规则7的六条"必须红" ------------------------------------------


def test_red_1_bare_quantity_without_a_unit_suffix_is_rejected():
    """轴2规则3：裸量不得跨边界；无量纲必须显式登记，不许留空装有。"""

    naked = MaterialProperty(
        name="yield_stress", value=68.9476, domains=("mechanics",), evidence=_evidence()
    )
    with pytest.raises(MaterialError, match="without a unit suffix"):
        _record(properties=(*_properties(), naked))


def test_red_2_property_escaping_the_declared_domains_is_rejected():
    """本仓增量：属性声明的域必须在记录的``applicable_domains``之内。"""

    stray = MaterialProperty(
        name="thermal_conductivity_w",
        value=391.0,
        domains=("thermal",),
        evidence=_evidence(name="conductivity"),
    )
    with pytest.raises(MaterialError, match="does not declare in applicable_domains"):
        _record(properties=(*_properties(), stray))


def test_red_3_mixing_length_systems_in_one_record_is_rejected():
    """静默1000倍的入口：mm制记录里冒出米制字段。"""

    metric = MaterialProperty(
        name="coating_thickness_m",
        value=2.0e-7,
        domains=("optics",),
        evidence=_evidence(grade="estimated", name="coating", source=None),
    )
    with pytest.raises(MaterialError, match="silent factor-1000"):
        _record(properties=(*_properties(), metric))


def test_red_4_tampered_self_referential_address_is_rejected():
    """轴3规则4：自指哈希必须自校验。"""

    assert _record().sealed().content_sha256 is not None
    with pytest.raises(MaterialError, match="self-referential address failed"):
        _record(content_sha256="0" * 64)


def test_red_5_locked_bytes_changed_by_one_byte_are_rejected():
    """轴3规则1：被锁定的资产改一个字节→加载必须拒收，且拒在解析之前。"""

    payload = canonical_file_bytes(_record().sealed().to_document(), MATERIAL_CANONICAL_PROFILE)
    lock = hashlib.sha256(payload).hexdigest()
    mutated = payload.replace(b"6.35", b"6.36", 1)
    assert mutated != payload
    with pytest.raises(MaterialError, match="locked material bytes do not match"):
        load_material_record(mutated, expected_sha256=lock)


def test_red_6_a_value_carried_under_the_unset_grade_is_rejected():
    """FTS ``E_UNSET_HAS_VALUE``：未测量就不许有数，占位零是伪装的声明。"""

    with pytest.raises(MaterialError, match="unset"):
        MaterialProperty(
            name="axial_damping_n_s_mm",
            value=0.0,
            domains=("mechanics",),
            evidence=EvidenceRef(
                grade="unset",
                evidence_id="evidence/cu-c110-p9318-damping-unmeasured",
                method="Zero placeholder standing in for an unmeasured damping.",
            ),
        )


# --- 其余失败关闭 --------------------------------------------------------


def test_evidence_without_a_method_is_rejected():
    """轴2规则5：'无出处可追'也必须显式写出。"""

    with pytest.raises(MaterialError, match="nonempty method"):
        EvidenceRef(grade="estimated", evidence_id="evidence/x", method="   ")


def test_external_grade_without_a_source_hash_is_rejected():
    with pytest.raises(MaterialError, match="source_sha256 is mandatory"):
        _evidence(grade="measured", source=None)


def test_unknown_evidence_grade_is_rejected():
    with pytest.raises(MaterialError, match="evidence grade must be one of"):
        _evidence(grade="illustrative")


def test_valued_property_without_a_grade_that_allows_it_is_rejected():
    with pytest.raises(MaterialError, match="only 'unset' may be valueless"):
        MaterialProperty(
            name="width_mm", value=None, domains=("mechanics",), evidence=_evidence()
        )


def test_material_id_outside_the_material_namespace_is_rejected():
    with pytest.raises(MaterialError, match="material.*namespace"):
        _record(material_id="scenario/spool-wrap")


def test_unregistered_facet_version_is_rejected():
    with pytest.raises(FacetError, match="engine_material_record"):
        _record(facet_version="9.9")


def test_unknown_top_level_key_is_rejected():
    document = _record().sealed().to_document()
    document["surprise"] = 1
    with pytest.raises(MaterialError, match="unknown keys in material record"):
        load_material_record(json.dumps(document).encode("utf-8"))


def test_dimensionless_declaring_a_field_the_record_lacks_is_rejected():
    with pytest.raises(MaterialError, match="dimensionless declares fields"):
        _record(dimensionless=frozenset({"specular_reflectance", "ghost"}))


def test_reading_a_domain_the_record_does_not_serve_is_rejected():
    with pytest.raises(MaterialError, match="does not declare domain"):
        _record().properties_for_domain("thermal")


def test_cross_system_read_without_explicit_conversion_is_rejected():
    """安全带：声明了期望长度制就必须对得上，绝不悄悄给差1000倍的数。"""

    with pytest.raises(MaterialError, match="call converted_to"):
        _record().properties_for_domain("optics", expect_length_unit="m")


def test_composite_unit_without_a_registered_conversion_is_rejected():
    """`N/mm²`换到米制要先决定是不是MPa——那是消费方采纳声明的事，这里不猜。"""

    with pytest.raises(MaterialError, match="no registered mm->m conversion"):
        _record().converted_to("m")


# --- 绿例：聚合、访问、字节、锁 ------------------------------------------


def test_one_record_serves_two_domains_with_different_field_sets():
    record = _record()
    mechanics = record.properties_for_domain("mechanics")
    optics = record.properties_for_domain("optics")
    assert set(mechanics) == {
        "width_mm",
        "thickness_mm",
        "density_kg_m3",
        "EI_easy_N_mm2",
        "axial_damping_n_s_mm",
    }
    assert set(optics) == {"thickness_mm", "specular_reflectance"}
    # 共享字段是同一个值，不是两份拷贝——这正是聚合记录的意义。
    assert mechanics["thickness_mm"] == optics["thickness_mm"] == 0.127


def test_weakest_grade_is_reported_per_domain():
    record = _record()
    assert record.weakest_grade("mechanics") == "unset"
    assert record.weakest_grade("optics") == "estimated"
    assert record.weakest_grade() == "unset"


def test_density_in_si_survives_inside_an_mm_record():
    """WDS实测形制：mm制记录带``density_kg_m3``是正常的，写法本身钉死了量纲。"""

    assert unit_suffix_of("density_kg_m3") == "kg_m3"
    assert _record().properties_for_domain("mechanics")["density_kg_m3"] == 8910.0


def test_sealed_record_round_trips_through_canonical_bytes_and_its_lock():
    sealed = _record().sealed()
    payload = canonical_file_bytes(sealed.to_document(), MATERIAL_CANONICAL_PROFILE)
    lock = hashlib.sha256(payload).hexdigest()
    reloaded = load_material_record(payload, expected_sha256=lock)
    assert reloaded == sealed
    assert reloaded.content_sha256 == reloaded.content_address()


def test_explicit_conversion_scales_pure_length_quantities():
    optics_only = _record(
        applicable_domains=("optics",),
        properties=(
            MaterialProperty(
                name="thickness_mm",
                value=0.127,
                domains=("optics",),
                evidence=_evidence(name="thickness"),
            ),
            MaterialProperty(
                name="specular_reflectance",
                value=0.94,
                domains=("optics",),
                evidence=_evidence(grade="estimated", name="reflectance", source=None),
            ),
        ),
        appearance=None,
    )
    metric = optics_only.converted_to("m")
    assert metric.length_unit == "m"
    assert metric.properties_for_domain("optics", expect_length_unit="m") == {
        "thickness_m": 0.000127,
        "specular_reflectance": 0.94,
    }
    # 换制换了字节，因此换了身份——旧的自指地址不许跟着走。
    assert metric.content_sha256 is None
    assert metric.content_address() != optics_only.sealed().content_address()
    assert metric.converted_to("mm").properties_for_domain("optics")["thickness_mm"] == 0.127


def test_evidence_is_reachable_per_field():
    assert _record().evidence_for("EI_easy_N_mm2").grade == "derived"
    with pytest.raises(MaterialError, match="carries no property"):
        _record().evidence_for("nope_mm")
