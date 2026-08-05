"""轴2参考实现的门：四段身份、命名空间ID、单位后缀。"""

import pytest

from physics_engine.identity import (
    IdentityError,
    assert_quantity_fields_have_units,
    has_unit_suffix,
    parse_case_identity,
    parse_namespace_id,
)


def test_four_segment_identity_roundtrips():
    parsed = parse_case_identity("spool_wrap__single_turn__cu_c110_p9318__r250__v001")
    assert parsed.family == "spool_wrap"
    assert parsed.version == 1
    assert parsed.canonical() == "spool_wrap__single_turn__cu_c110_p9318__r250__v001"


@pytest.mark.parametrize(
    "bad",
    [
        "a__b__c__v001",                    # 只有三段+版本
        "a__b__c__d__e__v001",              # 六段
        "a__b__c__d__001",                  # 版本缺v
        "a__b__c__d__v0",                   # 版本不足三位
        "a__b__c__d__v000",                 # 版本为零
        "A__b__c__d__v001",                 # 大写
        "__b__c__d__v001",                  # 空段
    ],
)
def test_malformed_identities_are_rejected(bad):
    with pytest.raises(IdentityError):
        parse_case_identity(bad)


def test_namespace_id_parses_and_rejects():
    assert parse_namespace_id("material/cu_c110_p9318")[0] == "material"
    with pytest.raises(IdentityError):
        parse_namespace_id("material")
    with pytest.raises(IdentityError):
        parse_namespace_id("a/b/c")


def test_unit_suffixes_from_both_repos_are_recognised():
    for name in ("tape_width_mm", "tension_N", "EA_N", "dt_s", "density_kg_m3",
                 "stiffness_N_mm2", "amplitude_w", "angle_rad"):
        assert has_unit_suffix(name), name
    assert not has_unit_suffix("tape_width")


def test_naked_quantity_field_fails_closed():
    with pytest.raises(IdentityError, match="tape_width"):
        assert_quantity_fields_have_units(("tape_width", "tension_N"))


def test_dimensionless_must_be_declared_not_assumed():
    assert_quantity_fields_have_units(
        ("mu", "tension_N"), dimensionless=frozenset({"mu"})
    )
    with pytest.raises(IdentityError, match="mu"):
        assert_quantity_fields_have_units(("mu", "tension_N"))


def test_extra_units_extend_but_do_not_replace_the_base_set():
    assert has_unit_suffix("flux_wb", extra_units=frozenset({"wb"}))
    assert has_unit_suffix("tension_N", extra_units=frozenset({"wb"}))
