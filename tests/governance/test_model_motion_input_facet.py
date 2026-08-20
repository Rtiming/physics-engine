"""模型—运动—虚拟物理输入面必须先登记且保持draft失败关闭。"""

from __future__ import annotations

import pytest

from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    PHYSICS_MODEL_MOTION_INPUT_FACET,
    PHYSICS_MODEL_MOTION_INPUT_VERSION,
)
from physics_engine.facets import FacetError, FacetStatus


def test_model_motion_input_is_registered_as_draft():
    facet = ENGINE_REGISTRY.get(PHYSICS_MODEL_MOTION_INPUT_FACET)
    assert PHYSICS_MODEL_MOTION_INPUT_VERSION == "0.1"
    assert facet.status is FacetStatus.DRAFT
    assert facet.major == 0
    assert facet.max_tested_minor == 1


@pytest.mark.parametrize("version", ("0.2", "0.9", "1.0"))
def test_red_untested_or_wrong_major_versions_are_rejected(version):
    with pytest.raises(FacetError):
        ENGINE_REGISTRY.assert_reader_compatible(
            PHYSICS_MODEL_MOTION_INPUT_FACET, version
        )


def test_model_motion_facet_constants_are_exported():
    from physics_engine import engine_facets

    assert "PHYSICS_MODEL_MOTION_INPUT_FACET" in engine_facets.__all__
    assert "PHYSICS_MODEL_MOTION_INPUT_VERSION" in engine_facets.__all__
