"""T-M2读出样点面必须在跨边界字节出生前登记并保持draft。"""

from __future__ import annotations

import pytest

from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    TENSION_READOUT_SAMPLE_FACET,
    TENSION_READOUT_SAMPLE_VERSION,
)
from physics_engine.facets import FacetError, FacetStatus


def test_tension_readout_sample_is_registered_as_draft():
    facet = ENGINE_REGISTRY.get(TENSION_READOUT_SAMPLE_FACET)
    assert TENSION_READOUT_SAMPLE_VERSION == "0.1"
    assert facet.status is FacetStatus.DRAFT
    assert facet.major == 0
    assert facet.max_tested_minor == 1


@pytest.mark.parametrize("version", ("0.2", "0.9", "1.0"))
def test_red_untested_or_wrong_major_readout_versions_are_rejected(version):
    with pytest.raises(FacetError):
        ENGINE_REGISTRY.assert_reader_compatible(TENSION_READOUT_SAMPLE_FACET, version)


def test_readout_facet_constants_are_exported():
    from physics_engine import engine_facets

    assert "TENSION_READOUT_SAMPLE_FACET" in engine_facets.__all__
    assert "TENSION_READOUT_SAMPLE_VERSION" in engine_facets.__all__
