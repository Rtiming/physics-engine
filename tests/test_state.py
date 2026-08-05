"""状态层的门（spec/12第二节）。每条校验附"必须红"。"""

from __future__ import annotations

import pytest

from physics_engine.state import State, StateError, StateField, StateLayout

LAYOUT = StateLayout(
    layout_id="layout/point_mass_1d",
    fields=(
        StateField(name="position_mm", width=1),
        StateField(name="velocity_mm_s", width=1),
    ),
)


def test_field_without_a_unit_suffix_is_rejected():
    """轴2：状态字段是跨边界的量，裸名字进不来。"""

    with pytest.raises(StateError, match="unit suffix"):
        StateField(name="position", width=1)


@pytest.mark.parametrize("width", [0, -1, True])
def test_field_width_must_be_a_positive_integer(width):
    with pytest.raises(StateError, match="width"):
        StateField(name="position_mm", width=width)


def test_layout_id_must_be_namespaced():
    with pytest.raises(StateError, match="layout_id"):
        StateLayout(layout_id="point_mass", fields=(StateField("position_mm", 1),))


def test_duplicate_field_names_are_rejected():
    with pytest.raises(StateError, match="duplicate"):
        StateLayout(
            layout_id="layout/x",
            fields=(StateField("position_mm", 1), StateField("position_mm", 1)),
        )


def test_vector_length_must_match_the_declared_layout():
    with pytest.raises(StateError, match="declares 2"):
        State(layout=LAYOUT, vector=(1.0,))


def test_non_finite_values_never_enter_state():
    with pytest.raises(StateError, match="finite"):
        State(layout=LAYOUT, vector=(1.0, float("nan")))


def test_blocks_are_addressed_by_name_not_by_hand_written_offsets():
    state = State(layout=LAYOUT, vector=(3.0, 4.0))
    assert state.block("position_mm") == (3.0,)
    assert state.block("velocity_mm_s") == (4.0,)
    with pytest.raises(StateError, match="unknown state field"):
        state.block("torsion_rad")


def test_packing_order_is_part_of_the_form_and_has_its_own_fingerprint():
    """次序换了、指纹就变——这正是"跑得通但全错"要被挡住的地方。

    两个布局字段完全相同、只是次序相反，`dof_count`一样、向量长度一样，
    任何只看长度的检查都发现不了；指纹发现得了。
    """

    reversed_layout = StateLayout(
        layout_id="layout/point_mass_1d",
        fields=(
            StateField(name="velocity_mm_s", width=1),
            StateField(name="position_mm", width=1),
        ),
    )
    assert reversed_layout.dof_count == LAYOUT.dof_count
    assert reversed_layout.fingerprint() != LAYOUT.fingerprint()
    assert reversed_layout.offset_of("position_mm") != LAYOUT.offset_of("position_mm")


def test_history_fields_are_declared_not_inferred():
    """真历史（塑性set、粘着锚点）必须显式声明；分不清就按保守方向声明为真历史。"""

    layout = StateLayout(
        layout_id="layout/with_history",
        fields=(
            StateField("position_mm", 1),
            StateField("plastic_curvature_per_mm", 1, is_history=True),
        ),
    )
    assert layout.history_fields() == ("plastic_curvature_per_mm",)
    assert LAYOUT.history_fields() == ()


def test_pack_is_canonical_bytes_so_two_implementations_can_be_compared():
    state = State(layout=LAYOUT, vector=(1.5, -2.25))
    assert state.pack() == State(layout=LAYOUT, vector=(1.5, -2.25)).pack()
    assert state.pack() != state.with_vector((1.5, -2.26)).pack()
