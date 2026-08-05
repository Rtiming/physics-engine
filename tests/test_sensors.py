"""传感器声明层的门——spec/10第四节那条原创法条的红例。

**首个必须红用例就是规范点名的那一条**：声明一个读全场状态的传感器，校验必须拒。
它有两种写法（把布局字段名当观测量、声明一个维数等于自由度数的大通道），
两条都在下面，两条都必须红。
"""

from __future__ import annotations

import pytest

from physics_engine.sensors import (
    Instrument,
    ObservationChannel,
    SensorDeclaration,
    SensorError,
)
from physics_engine.state import StateField, StateLayout


def _layout(nodes: int = 8) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/strip_n{nodes}",
        fields=tuple(
            field
            for index in range(nodes)
            for field in (
                StateField(f"node{index}_x_mm", 1),
                StateField(f"node{index}_y_mm", 1),
                StateField(f"node{index}_z_mm", 1),
            )
        ),
    )


LOAD_CELL = Instrument(
    instrument_id="instrument/payout_load_cell",
    kind="load_cell",
    measures=("payout_tension_n",),
    sample_period_s=0.001,
    delay_s=0.004,
)
ENCODER = Instrument(
    instrument_id="instrument/spool_encoder",
    kind="encoder",
    measures=("spool_angle_rad",),
    sample_period_s=0.0005,
    delay_s=0.001,
)


def _realistic() -> SensorDeclaration:
    """张力机放线端：一个力传感器 + 一个编码器。真机上这两样都装得出来。"""

    return SensorDeclaration(
        sensor_id="sensor/payout_station",
        instruments=(LOAD_CELL, ENCODER),
        channels=(
            ObservationChannel(
                channel_id="channel/tension",
                quantity_id="payout_tension_n",
                dimension=1,
                instrument_id="instrument/payout_load_cell",
            ),
            ObservationChannel(
                channel_id="channel/spool_angle",
                quantity_id="spool_angle_rad",
                dimension=1,
                instrument_id="instrument/spool_encoder",
            ),
        ),
    )


def test_a_realistic_sensor_declaration_is_accepted():
    sensor = _realistic()
    sensor.assert_realizable_on(_layout())
    assert sensor.observation_dimension == 2
    # 时延取最大：一次完整观测要等最慢的那一路到齐。
    assert sensor.delay_s == 0.004


def test_the_observation_dimension_does_not_depend_on_the_scene_size():
    """真机上的通道数由硬件定死。同一个声明放到更细的网格上，维数一个也不多。"""

    sensor = _realistic()
    for nodes in (8, 64, 512):
        sensor.assert_realizable_on(_layout(nodes))
    assert sensor.observation_dimension == 2


# ------------------------------------------------------------ 首个必须红 -----
# spec/10第四节点名的那一条："声明一个读全场状态的传感器，校验必须拒"。


def test_a_sensor_that_names_state_degrees_of_freedom_must_be_rejected():
    """写法一：把状态布局的字段名当观测量。

    带材每个节点的精确位置在布局里就叫``node3_y_mm``——真机上没有这个量。
    """

    layout = _layout()
    omniscient = Instrument(
        instrument_id="instrument/magic_tracker",
        kind="camera",
        measures=("node3_y_mm",),
        sample_period_s=0.01,
        delay_s=0.0,
    )
    sensor = SensorDeclaration(
        sensor_id="sensor/full_field_by_name",
        instruments=(omniscient,),
        channels=(
            ObservationChannel(
                channel_id="channel/node3_y",
                quantity_id="node3_y_mm",
                dimension=1,
                instrument_id="instrument/magic_tracker",
            ),
        ),
    )
    with pytest.raises(SensorError, match="state degree of freedom"):
        sensor.assert_realizable_on(layout)


def test_a_sensor_whose_dimension_reaches_the_dof_count_must_be_rejected():
    """写法二：一个维数等于自由度数的大通道——就是"把整个状态向量读出来"。"""

    layout = _layout()
    omniscient = Instrument(
        instrument_id="instrument/magic_tracker",
        kind="camera",
        measures=("strip_node_positions_mm",),
        sample_period_s=0.01,
        delay_s=0.0,
    )
    sensor = SensorDeclaration(
        sensor_id="sensor/full_field_by_dimension",
        instruments=(omniscient,),
        channels=(
            ObservationChannel(
                channel_id="channel/all_nodes",
                quantity_id="strip_node_positions_mm",
                dimension=layout.dof_count,
                instrument_id="instrument/magic_tracker",
            ),
        ),
    )
    with pytest.raises(SensorError, match="reaches the"):
        sensor.assert_realizable_on(layout)


def test_reading_the_full_field_stays_rejected_at_every_resolution():
    """全场读取在任何网格分辨率上都必须红——它不是"太大了"，是**种类不对**。"""

    for nodes in (2, 8, 512):
        layout = _layout(nodes)
        instrument = Instrument(
            instrument_id="instrument/magic_tracker",
            kind="camera",
            measures=("strip_node_positions_mm",),
            sample_period_s=0.01,
            delay_s=0.0,
        )
        sensor = SensorDeclaration(
            sensor_id="sensor/full_field",
            instruments=(instrument,),
            channels=(
                ObservationChannel(
                    channel_id="channel/all_nodes",
                    quantity_id="strip_node_positions_mm",
                    dimension=layout.dof_count,
                    instrument_id="instrument/magic_tracker",
                ),
            ),
        )
        with pytest.raises(SensorError):
            sensor.assert_realizable_on(layout)


# ------------------------------------------------------- 其余必须红用例 -----


def test_an_unknown_instrument_kind_must_be_rejected():
    with pytest.raises(SensorError, match="unknown instrument kind"):
        Instrument(
            instrument_id="instrument/oracle_probe",
            kind="omniscient_probe",
            measures=("tension_n",),
            sample_period_s=0.01,
            delay_s=0.0,
        )


def test_an_instrument_must_say_what_it_measures():
    with pytest.raises(SensorError, match="must declare what it measures"):
        Instrument(
            instrument_id="instrument/empty",
            kind="load_cell",
            measures=(),
            sample_period_s=0.01,
            delay_s=0.0,
        )


@pytest.mark.parametrize("delay", [None, -0.001, float("nan"), float("inf")])
def test_a_missing_or_absurd_delay_must_be_rejected(delay):
    """时延不是可选项——零时延要显式写0.0，缺省一个等于假装真机没有延迟。"""

    with pytest.raises(SensorError, match="delay_s"):
        Instrument(
            instrument_id="instrument/probe",
            kind="load_cell",
            measures=("tension_n",),
            sample_period_s=0.01,
            delay_s=delay,
        )


@pytest.mark.parametrize("period", [0.0, -1.0, float("inf")])
def test_a_nonpositive_sample_period_must_be_rejected(period):
    with pytest.raises(SensorError, match="sample_period_s"):
        Instrument(
            instrument_id="instrument/probe",
            kind="load_cell",
            measures=("tension_n",),
            sample_period_s=period,
            delay_s=0.0,
        )


def test_a_quantity_without_a_unit_suffix_must_be_rejected():
    """轴2规则3：观测量没有单位就没法与真机标定对上。"""

    with pytest.raises(SensorError, match="unit suffix"):
        Instrument(
            instrument_id="instrument/probe",
            kind="load_cell",
            measures=("tension",),
            sample_period_s=0.01,
            delay_s=0.0,
        )


@pytest.mark.parametrize(
    "identifier", ["payout_load_cell", "sensor/payout", "instrument/a/b"]
)
def test_an_instrument_id_must_be_namespaced(identifier):
    with pytest.raises(SensorError, match="instrument_id"):
        Instrument(
            instrument_id=identifier,
            kind="load_cell",
            measures=("tension_n",),
            sample_period_s=0.01,
            delay_s=0.0,
        )


def test_a_channel_bound_to_an_undeclared_instrument_must_be_rejected():
    with pytest.raises(SensorError, match="undeclared instrument"):
        SensorDeclaration(
            sensor_id="sensor/dangling",
            instruments=(LOAD_CELL,),
            channels=(
                ObservationChannel(
                    channel_id="channel/angle",
                    quantity_id="spool_angle_rad",
                    dimension=1,
                    instrument_id="instrument/spool_encoder",
                ),
            ),
        )


def test_a_channel_asking_for_something_its_instrument_cannot_measure_must_be_rejected():
    """仪器测不到的量不是观测量——这一条堵的是"编一台能测全场的仪器"那条路。"""

    with pytest.raises(SensorError, match="only measures"):
        SensorDeclaration(
            sensor_id="sensor/wishful",
            instruments=(LOAD_CELL,),
            channels=(
                ObservationChannel(
                    channel_id="channel/angle",
                    quantity_id="spool_angle_rad",
                    dimension=1,
                    instrument_id="instrument/payout_load_cell",
                ),
            ),
        )


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True])
def test_a_channel_dimension_must_be_a_positive_integer(dimension):
    with pytest.raises(SensorError, match="dimension"):
        ObservationChannel(
            channel_id="channel/tension",
            quantity_id="payout_tension_n",
            dimension=dimension,
            instrument_id="instrument/payout_load_cell",
        )


def test_duplicate_channels_and_instruments_must_be_rejected():
    channel = ObservationChannel(
        channel_id="channel/tension",
        quantity_id="payout_tension_n",
        dimension=1,
        instrument_id="instrument/payout_load_cell",
    )
    with pytest.raises(SensorError, match="duplicate channel"):
        SensorDeclaration(
            sensor_id="sensor/dup", instruments=(LOAD_CELL,), channels=(channel, channel)
        )
    with pytest.raises(SensorError, match="duplicate instrument"):
        SensorDeclaration(
            sensor_id="sensor/dup",
            instruments=(LOAD_CELL, LOAD_CELL),
            channels=(channel,),
        )


def test_an_empty_sensor_must_be_rejected():
    with pytest.raises(SensorError, match="at least one instrument"):
        SensorDeclaration(sensor_id="sensor/empty", instruments=(), channels=())
    with pytest.raises(SensorError, match="at least one channel"):
        SensorDeclaration(
            sensor_id="sensor/empty", instruments=(LOAD_CELL,), channels=()
        )


def test_the_layout_argument_is_type_checked():
    """传个字符串进来不该被当成布局默默放行。"""

    with pytest.raises(SensorError, match="expected a StateLayout"):
        _realistic().assert_realizable_on("layout/strip_n8")


def test_the_module_declares_no_read_implementation():
    """spec/10的``Sensor``接口尚未冻结——本模块**故意**停在声明层。

    这条门守的是范围：``read``一旦出现在这里，就等于在接口冻结前替它拍了板。
    """

    from physics_engine import sensors

    assert not hasattr(sensors.SensorDeclaration, "read")
    assert sensors.__all__ == [
        "REALIZABLE_INSTRUMENT_KINDS",
        "Instrument",
        "ObservationChannel",
        "SensorDeclaration",
        "SensorError",
    ]


def test_sensors_do_not_enter_the_top_level_re_export():
    """预算纪律：实验档模块不进顶层``__init__``（那是eager import成本）。"""

    import physics_engine

    assert not hasattr(physics_engine, "SensorDeclaration")
    assert "SensorDeclaration" not in physics_engine.__all__
