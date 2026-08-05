"""轴1参考实现的门，含全部"必须红"用例。"""

import pytest

from physics_engine.facets import (
    Facet,
    FacetError,
    FacetRegistry,
    FacetStatus,
    parse_version,
)


def _registry() -> FacetRegistry:
    return FacetRegistry(
        Facet(name="case_input", major=1, max_tested_minor=2, status=FacetStatus.FROZEN),
        Facet(name="scratch", major=0, max_tested_minor=0, status=FacetStatus.DRAFT),
    )


def test_known_facet_within_tested_range_passes():
    _registry().assert_reader_compatible("case_input", "1.2.0")
    _registry().assert_reader_compatible("case_input", "1.0")


def test_unknown_facet_is_rejected():
    with pytest.raises(FacetError, match="unknown facet"):
        _registry().assert_reader_compatible("nope", "1.0")


def test_wrong_major_is_rejected():
    with pytest.raises(FacetError, match="unsupported facet major"):
        _registry().assert_reader_compatible("case_input", "2.0")


def test_untested_future_minor_is_rejected():
    with pytest.raises(FacetError, match="untested facet minor"):
        _registry().assert_reader_compatible("case_input", "1.3")


def test_duplicate_facet_name_fails_at_assembly():
    with pytest.raises(FacetError, match="duplicate facet name"):
        FacetRegistry(
            Facet(name="a", major=1, max_tested_minor=0, status=FacetStatus.INTERNAL),
            Facet(name="a", major=2, max_tested_minor=0, status=FacetStatus.INTERNAL),
        )


def test_registry_registers_itself_as_a_facet():
    facet = _registry().get("facet_registry")
    assert facet.status is FacetStatus.INTERNAL
    with pytest.raises(FacetError, match="duplicate facet name"):
        FacetRegistry(
            Facet(name="facet_registry", major=9, max_tested_minor=0, status=FacetStatus.INTERNAL)
        )


def test_draft_facet_cannot_be_consumed_externally():
    registry = _registry()
    registry.assert_externally_consumable("case_input")
    with pytest.raises(FacetError, match="draft facet"):
        registry.assert_externally_consumable("scratch")


@pytest.mark.parametrize("bad", ["1", "1.2.3.4", "1.-2", "a.b", "1.2a", "", "1..2", "+1.2"])
def test_malformed_versions_are_rejected(bad):
    with pytest.raises(FacetError):
        parse_version(bad)


def test_version_patch_segment_is_parsed_but_not_compared():
    assert parse_version("1.2.9") == (1, 2)


@pytest.mark.parametrize("major,minor", [(-1, 0), (0, -1), (True, 0)])
def test_facet_field_validation_fails_closed(major, minor):
    with pytest.raises(FacetError):
        Facet(name="x", major=major, max_tested_minor=minor, status=FacetStatus.INTERNAL)
