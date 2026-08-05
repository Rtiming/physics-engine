"""驱动器声明层与时延机械的门——spec/10第三节。

**首个必须红用例就是规范点名的那一条**：时延不是可选项。它有三种躲法
（不写、写0不说理由、写一个不是``dt_s``整数倍的值然后指望库四舍五入），
三条都在下面，三条都必须红。

形制照``tests/test_sensors.py``，**外加反证**——把那条校验换成空操作
（等价于写成``if False``），同一个声明当场变绿，以此排除"红是别的规则顺手红的"。
"""

from __future__ import annotations

import math

import pytest

from physics_engine import actuators
from physics_engine.actuators import (
    DELAY_QUANTIZATIONS,
    DELAY_STEP_MATCH_ULPS,
    MAX_DELAY_STEPS,
    REALIZABLE_ACTUATOR_KINDS,
    ActuationCommand,
    ActuationDelayLine,
    ActuationResult,
    ActuatorDeclaration,
    ActuatorError,
    CommandChannel,
)

PAYOUT = "actuator/payout_station"

#: WDS张力机放线端（其用途4）：伺服电机给力矩、磁粉离合器给滑差力矩。
#: 单位按本仓的毫米制词汇表——力矩是``_nmm``（N·mm）不是``_nm``。
TORQUE = CommandChannel(
    channel_id="command/servo_torque",
    quantity_id="payout_servo_torque_nmm",
    dimension=1,
    lower=(-8.0e3,),
    upper=(8.0e3,),
)
CLUTCH = CommandChannel(
    channel_id="command/clutch_torque",
    quantity_id="clutch_slip_torque_nmm",
    dimension=1,
    lower=(0.0,),
    upper=(4.0e3,),
)


def _declaration(**overrides) -> ActuatorDeclaration:
    fields = {
        "actuator_id": PAYOUT,
        "kind": "servo_motor",
        "channels": (TORQUE, CLUTCH),
        "delay_s": 0.004,
        "zero_delay_rationale": None,
    }
    fields.update(overrides)
    return ActuatorDeclaration(**fields)


def _command(*values: float, actuator_id: str = PAYOUT) -> ActuationCommand:
    return ActuationCommand(actuator_id=actuator_id, values=values)


def _line(**overrides) -> ActuationDelayLine:
    fields = {
        "declaration": _declaration(),
        "dt_s": 0.001,
        "quantization": "exact",
        "initial_command": _command(0.0, 0.0),
    }
    fields.update(overrides)
    declaration = fields.pop("declaration")
    return ActuationDelayLine.declare(declaration, **fields)


# --------------------------------------------------------------- 正常路径 ---


def test_the_wds_payout_station_declares_cleanly():
    declaration = _declaration()
    assert declaration.command_dimension == 2
    assert declaration.bounds == ((-8.0e3, 8.0e3), (0.0, 4.0e3))
    assert declaration.delay_s == 0.004


def test_a_four_step_delay_line_emits_the_initial_fill_first():
    """MuJoCo形制：缓冲一出生就是满的，前``steps``步生效的是**运行之前**的那条命令。"""

    line = _line()
    assert (line.steps, line.realized_delay_s, line.quantization_residual_s) == (
        4,
        0.004,
        0.0,
    )
    emitted = []
    current = line
    for index in range(6):
        current, result = current.advance(_command(float(index + 1), 0.0))
        emitted.append((result.command.values[0], result.from_initial_fill))
    assert emitted == [
        (0.0, True), (0.0, True), (0.0, True), (0.0, True), (1.0, False), (2.0, False)
    ]


def test_the_timestamps_say_when_a_command_was_issued_and_when_it_lands():
    line = _line()
    current, first = line.advance(_command(1.0, 0.0))
    # 填充期那条命令确实早于本次运行——``issued_at_s``为负是实话，不是bug。
    assert first.issued_at_s == -0.004
    assert first.effective_at_s == 0.0
    assert first.from_initial_fill is True
    for _ in range(3):
        current, _result = current.advance(_command(1.0, 0.0))
    _current, fifth = current.advance(_command(9.0, 0.0))
    assert fifth.command.values == (1.0, 0.0)
    assert fifth.issued_at_s == 0.0
    assert fifth.effective_at_s == 0.004


def test_realized_delay_is_steps_times_dt_and_not_the_subtraction():
    """``effective_at_s − issued_at_s``**随步号抖**，所以契约取与步号无关的那个。

    实测反例：``dt_s=0.1``、``steps=7``，第2步的减法给0.7，而``7 × 0.1``给
    0.7000000000000001。一个随步号抖动的"实际时延"不是时延。
    """

    line = _line(dt_s=0.1, declaration=_declaration(delay_s=0.7))
    assert line.steps == 7
    current = line
    saw_a_mismatch = False
    for _ in range(6):
        current, result = current.advance(_command(0.0, 0.0))
        assert result.realized_delay_s == 7 * 0.1
        if result.effective_at_s - result.issued_at_s != result.realized_delay_s:
            saw_a_mismatch = True
    assert saw_a_mismatch, "找不到减法与乘法不等的步号，这条测试就没在验它想验的东西"


def test_an_explicit_zero_delay_line_passes_the_command_through_the_same_step():
    declaration = _declaration(
        actuator_id="actuator/ideal_valve",
        kind="solenoid",
        channels=(CommandChannel("command/open", "valve_open_pct", 1, (0.0,), (100.0,)),),
        delay_s=0.0,
        zero_delay_rationale="本案例只验几何干涉，回路时延不进物理；真机回路时延见WDS标定单",
    )
    line = ActuationDelayLine.declare(
        declaration,
        dt_s=0.01,
        quantization="exact",
        initial_command=_command(0.0, actuator_id="actuator/ideal_valve"),
    )
    assert line.steps == 0 and line.pending == ()
    _next_line, result = line.advance(_command(50.0, actuator_id="actuator/ideal_valve"))
    assert result.command.values == (50.0,)
    assert result.issued_at_s == result.effective_at_s == 0.0
    assert result.from_initial_fill is False


def test_the_delay_line_is_a_value_so_a_run_can_be_forked_and_replayed():
    """在途命令是真历史（spec/12第2.2节）。时延线是值，所以同一个线可以分叉两次跑。"""

    line = _line()
    branch_a, _ = line.advance(_command(1.0, 0.0))
    branch_b, _ = line.advance(_command(-1.0, 0.0))
    assert line.step_index == 0 and line.pending == (_command(0.0, 0.0),) * 4
    assert branch_a.pending[-1].values == (1.0, 0.0)
    assert branch_b.pending[-1].values == (-1.0, 0.0)
    assert branch_a != branch_b


# ------------------------------------------- 必须红：时延不是可选项（三躲）


def test_a_missing_delay_must_be_rejected():
    """躲法一：不写。``delay_s``没有默认值，构造期直接缺参数。"""

    with pytest.raises(TypeError, match="delay_s"):
        ActuatorDeclaration(  # type: ignore[call-arg]
            actuator_id=PAYOUT, kind="servo_motor", channels=(TORQUE,)
        )


def test_a_zero_delay_without_a_rationale_must_be_rejected():
    """躲法二：写0但不说为什么。零时延是一条关于真机的强声明。"""

    with pytest.raises(ActuatorError, match="needs an explicit zero_delay_rationale"):
        _declaration(delay_s=0.0, zero_delay_rationale=None)
    with pytest.raises(ActuatorError, match="needs an explicit zero_delay_rationale"):
        _declaration(delay_s=0.0, zero_delay_rationale="   ")


def test_a_rationale_on_a_nonzero_delay_must_be_rejected():
    """理由字段只在声明为0时有意义；留一个没人读的字段与留空装有是同一种病。"""

    with pytest.raises(ActuatorError, match="only meaningful when delay_s=0"):
        _declaration(delay_s=0.004, zero_delay_rationale="随手写的")


@pytest.mark.parametrize("delay", [-0.001, float("nan"), float("inf"), None, "0.004", True])
def test_an_absurd_delay_must_be_rejected(delay):
    with pytest.raises(ActuatorError, match="delay_s"):
        _declaration(delay_s=delay)


def test_counterproof_the_explicit_delay_rule_is_load_bearing(monkeypatch):
    """反证：换掉时延校验，一个不说理由的零时延声明与一个负时延声明双双变绿。"""

    monkeypatch.setattr(actuators, "_require_explicit_delay", lambda *args: 0.0)
    assert _declaration(delay_s=0.0, zero_delay_rationale=None).delay_s == 0.0
    assert _declaration(delay_s=-1.0).delay_s == -1.0


# ------------------------- 必须红：躲法三——不是整数倍时指望库四舍五入 -----


def test_a_delay_that_is_not_an_integer_multiple_of_dt_must_be_rejected():
    """默认口径失败关闭，并把残差与两条出路写在消息里。"""

    with pytest.raises(ActuatorError, match="not an integer multiple of"):
        _line(dt_s=0.0003)


def test_the_rejection_message_offers_both_ways_out():
    with pytest.raises(ActuatorError) as caught:
        _line(dt_s=0.0003)
    message = str(caught.value)
    assert "ceil_to_step" in message
    assert "residual" in message


def test_ceil_to_step_rounds_up_and_reports_the_residual():
    """向上取整是**保守方向**：仿真里的时延只会比声明的长，不会短。"""

    line = _line(dt_s=0.0003, quantization="ceil_to_step")
    assert line.steps == 14  # ceil(0.004 / 0.0003) = ceil(13.33) = 14
    assert line.realized_delay_s == pytest.approx(0.0042)
    assert line.quantization_residual_s > 0.0
    assert line.realized_delay_s > line.declaration.delay_s


def test_ceil_to_step_does_not_add_a_spurious_step_on_a_true_multiple():
    """真整数倍上``ceil``会多给一步，必须收回来。

    实测样例：``delay_s = 13 × 0.0001``，商是13.000000000000002，
    裸``ceil``给14——那是凭空多出来的0.1ms时延，而它本该是精确的13步。
    """

    delay = 13 * 0.0001
    assert math.ceil(delay / 0.0001) == 14
    line = _line(dt_s=0.0001, quantization="ceil_to_step", declaration=_declaration(delay_s=delay))
    assert line.steps == 13
    assert line.quantization_residual_s == 0.0


@pytest.mark.parametrize("mode", ["round", "floor", "nearest", "", None, 7])
def test_rounding_down_or_to_nearest_is_not_on_the_menu(mode):
    """**没有``round``也没有``floor``**：欠时延是spec/10第三节点名的危险方向。"""

    with pytest.raises(ActuatorError, match="quantization must be one of"):
        _line(quantization=mode)
    assert mode not in DELAY_QUANTIZATIONS


def test_counterproof_the_integer_multiple_rule_has_an_independent_second_gate(monkeypatch):
    """反证第一步，**它翻出了一件本来不知道的事**：只关掉整数倍校验并不会变绿。

    ``round(0.004 / 0.0003) = 13``给出0.0039，比声明的0.004**短**，于是
    ``ActuationDelayLine.__post_init__``里那条"只许多延不许少延"的门接住了它。
    两条门互相独立，这一步把它证出来了——原本这个反证是按"关一条就绿"写的。
    """

    monkeypatch.setattr(actuators, "_require_integer_step_delay", lambda *args: None)
    with pytest.raises(ActuatorError, match="shorter than the declared delay_s"):
        _line(dt_s=0.0003)


def test_counterproof_both_delay_gates_are_load_bearing(monkeypatch):
    """反证第二步：两条门一起关掉，同一个声明当场变绿——**而且时延悄悄短了**。

    这正是四舍五入被禁的样子：实际时延变成0.0039、残差为负,
    仿真里的回路比真机快了0.1ms，而没有任何东西会说一声。
    """

    monkeypatch.setattr(actuators, "_require_integer_step_delay", lambda *args: None)
    monkeypatch.setattr(actuators, "_steps_match_delay", lambda *args: True)
    green = _line(dt_s=0.0003)
    assert green.steps == 13
    assert green.realized_delay_s < green.declaration.delay_s
    assert green.quantization_residual_s < 0.0


def test_a_delay_line_constructed_directly_cannot_under_delay():
    """绕开``declare()``直接构造也拦得住——欠时延这条只许多延不许少延。"""

    with pytest.raises(ActuatorError, match="shorter than the declared delay_s"):
        ActuationDelayLine(
            declaration=_declaration(),
            dt_s=0.001,
            quantization="exact",
            steps=3,
            realized_delay_s=0.003,
            quantization_residual_s=-0.001,
            pending=(_command(0.0, 0.0),) * 3,
            step_index=0,
        )


def test_a_delay_line_must_carry_a_consistent_realized_delay():
    with pytest.raises(ActuatorError, match="exactly steps × dt_s"):
        ActuationDelayLine(
            declaration=_declaration(),
            dt_s=0.001,
            quantization="exact",
            steps=4,
            realized_delay_s=0.5,
            quantization_residual_s=0.0,
            pending=(_command(0.0, 0.0),) * 4,
            step_index=0,
        )


def test_the_pending_buffer_length_must_equal_the_step_count():
    with pytest.raises(ActuatorError, match="steps deep"):
        ActuationDelayLine(
            declaration=_declaration(),
            dt_s=0.001,
            quantization="exact",
            steps=4,
            realized_delay_s=0.004,
            quantization_residual_s=0.0,
            pending=(_command(0.0, 0.0),),
            step_index=0,
        )


# --------------------------------------------- 必须红：单位门与缓冲深度 -----


def test_writing_the_delay_in_milliseconds_is_caught_by_the_buffer_ceiling():
    """把4ms写成``delay_s=4``：10kHz步长下要4万步……而4000ms=4s的回路不存在。"""

    with pytest.raises(ActuatorError, match="MAX_DELAY_STEPS"):
        _line(dt_s=1.0e-5, declaration=_declaration(delay_s=4.0))


def test_the_buffer_ceiling_is_stated_in_seconds_of_delay():
    """上限本身要被验：它必须真的挡得住"没有哪个回路有10秒时延"那一档。"""

    assert MAX_DELAY_STEPS * 1.0e-4 == pytest.approx(10.0)


@pytest.mark.parametrize("dt", [0.0, -0.001, float("nan"), float("inf"), None])
def test_a_nonpositive_step_must_be_rejected(dt):
    with pytest.raises(ActuatorError, match="dt_s"):
        _line(dt_s=dt)


# ------------------------------------------------- 必须红：命令空间与界限 ---


def test_an_unknown_actuator_kind_must_be_rejected():
    with pytest.raises(ActuatorError, match="unknown actuator kind"):
        _declaration(kind="force_field_generator")


def test_a_command_quantity_without_a_unit_suffix_must_be_rejected():
    """轴2规则3：命令量没有单位就没法与真机的控制器对上。"""

    with pytest.raises(ActuatorError, match="unit suffix"):
        CommandChannel("command/torque", "torque", 1, (-1.0,), (1.0,))


@pytest.mark.parametrize(
    ("lower", "upper"), [((1.0,), (1.0,)), ((2.0,), (1.0,)), ((0.0,), (-1.0,))]
)
def test_a_channel_needs_lower_strictly_below_upper(lower, upper):
    """一个能下达无穷大力矩的驱动器与一个能读全场状态的传感器是同一种谎。"""

    with pytest.raises(ActuatorError, match="lower < upper"):
        CommandChannel("command/torque", "servo_torque_nmm", 1, lower, upper)


@pytest.mark.parametrize("bounds", [(-1.0,), (-1.0, -2.0, -3.0), [-1.0, -2.0], "xx"])
def test_the_bounds_length_must_match_the_dimension(bounds):
    with pytest.raises(ActuatorError, match="2-tuple"):
        CommandChannel("command/pair", "servo_torque_nmm", 2, bounds, (9.0, 9.0))


def test_a_command_outside_the_declared_range_must_be_rejected():
    """**不夹取**：悄悄夹到边界会让一个打满的执行器看起来一直在正常工作。"""

    line = _line()
    with pytest.raises(ActuatorError, match="outside the declared range"):
        line.advance(_command(9.0e3, 0.0))
    with pytest.raises(ActuatorError, match="outside the declared range"):
        line.advance(_command(0.0, -1.0))


def test_counterproof_the_bounds_check_is_load_bearing(monkeypatch):
    """反证：换掉可下达性校验，一条9000 N·mm的命令在8000上限的驱动器上当场变绿。"""

    monkeypatch.setattr(
        ActuatorDeclaration, "assert_command_admissible", lambda self, command: None
    )
    line = _line()
    _next_line, result = line.advance(_command(9.0e3, 0.0))
    assert result.command.values == (0.0, 0.0)  # 生效的是填充命令
    assert _next_line.pending[-1].values == (9.0e3, 0.0)  # 越界命令进了缓冲


def test_a_command_with_the_wrong_dimension_must_be_rejected():
    with pytest.raises(ActuatorError, match="command space declares"):
        _line().advance(_command(1.0))


def test_a_command_addressed_to_another_actuator_must_be_rejected():
    with pytest.raises(ActuatorError, match="addressed to"):
        _line().advance(_command(1.0, 0.0, actuator_id="actuator/somebody_else"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "1.0"])
def test_a_command_with_a_nonsense_scalar_must_be_rejected(value):
    with pytest.raises(ActuatorError, match="values"):
        _command(value, 0.0)


def test_an_actuator_needs_at_least_one_command_channel():
    with pytest.raises(ActuatorError, match="at least one command channel"):
        _declaration(channels=())


def test_duplicate_channels_must_be_rejected():
    with pytest.raises(ActuatorError, match="duplicate channel"):
        _declaration(channels=(TORQUE, TORQUE))


@pytest.mark.parametrize("identifier", ["payout", "sensor/payout", "actuator/a/b", 7, None])
def test_an_actuator_id_must_be_namespaced(identifier):
    with pytest.raises(ActuatorError, match="actuator_id"):
        _declaration(actuator_id=identifier)


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True])
def test_a_channel_dimension_must_be_a_positive_integer(dimension):
    with pytest.raises(ActuatorError, match="dimension"):
        CommandChannel("command/torque", "servo_torque_nmm", dimension, (-1.0,), (1.0,))


def test_the_initial_command_is_required_and_validated():
    """缓冲在第0步之前就是满的；默认填零等于替声明者断言"运行之前它闲着"。"""

    with pytest.raises(TypeError, match="initial_command"):
        ActuationDelayLine.declare(  # type: ignore[call-arg]
            _declaration(), dt_s=0.001, quantization="exact"
        )
    with pytest.raises(ActuatorError, match="outside the declared range"):
        _line(initial_command=_command(9.0e3, 0.0))


# ------------------------------------------------- 容差本身要被验（轴6规则6）


def test_the_step_match_tolerance_sits_between_representation_error_and_any_real_mismatch():
    """**容差是算出来的**：真整数倍对的最大ULP距离必须落在阈值内，
    而一个相对偏移1e-13（远小于任何物理意义）的假整数倍必须还差两个数量级。"""

    grids = [1 / 1000, 1 / 500, 1 / 240, 1 / 200, 1 / 120, 1 / 100, 1 / 60, 1 / 20, 1 / 10]
    worst_true = 0.0
    for dt in grids:
        for steps in (1, 2, 3, 7, 13, 60, 137, 1000, 4096):
            delay = steps * dt
            residual = abs(delay - steps * dt)
            scale = max(abs(delay), abs(steps * dt))
            worst_true = max(worst_true, residual / math.ulp(scale))
    assert worst_true <= DELAY_STEP_MATCH_ULPS

    smallest_false = None
    for dt in grids:
        for steps in (1, 2, 7, 40, 300):
            delay = (steps * dt) * (1.0 + 1.0e-13)
            closest = round(delay / dt)
            residual = abs(delay - closest * dt)
            scale = max(abs(delay), abs(closest * dt))
            if residual == 0.0:
                continue
            ulps = residual / math.ulp(scale)
            smallest_false = ulps if smallest_false is None else min(smallest_false, ulps)
    assert smallest_false is not None
    assert smallest_false > 100 * DELAY_STEP_MATCH_ULPS


def test_a_true_integer_multiple_written_as_a_decimal_literal_is_accepted():
    """``0.004 / 0.001 = 3.9999999999999996``——判据必须吃得下这一点表示误差。"""

    for delay, dt, steps in ((0.004, 0.001, 4), (0.05, 0.01, 5), (0.7, 0.1, 7)):
        line = _line(dt_s=dt, declaration=_declaration(delay_s=delay))
        assert line.steps == steps
        assert line.quantization_residual_s == pytest.approx(0.0, abs=1e-15)


# ------------------------------------------------------------- 范围与预算 ---


def test_the_module_stops_at_the_declaration_and_delay_layer():
    """``apply``的物理不在这里。这条门守的是范围——它一旦出现，就等于替冻结拍了板。"""

    for holder in (ActuatorDeclaration, ActuationDelayLine, ActuationResult):
        assert not hasattr(holder, "apply")
        assert not hasattr(holder, "command_space")
    for name in ("force_n", "torque_nmm", "energy_j", "gradient", "state"):
        assert not hasattr(ActuationResult, name)
    assert set(ActuationResult.__dataclass_fields__) == {
        "actuator_id",
        "command",
        "issued_at_s",
        "effective_at_s",
        "realized_delay_s",
        "from_initial_fill",
    }


def test_the_public_surface_is_exactly_what_all_says():
    assert actuators.__all__ == [
        "DELAY_QUANTIZATIONS",
        "DELAY_STEP_MATCH_ULPS",
        "MAX_DELAY_STEPS",
        "REALIZABLE_ACTUATOR_KINDS",
        "ActuationCommand",
        "ActuationDelayLine",
        "ActuationResult",
        "ActuatorDeclaration",
        "ActuatorError",
        "CommandChannel",
    ]
    assert "servo_motor" in REALIZABLE_ACTUATOR_KINDS
    assert "magnetic_particle_clutch" in REALIZABLE_ACTUATOR_KINDS


def test_actuators_do_not_enter_the_top_level_re_export():
    import physics_engine

    assert not hasattr(physics_engine, "ActuatorDeclaration")
    assert "ActuatorDeclaration" not in physics_engine.__all__


def test_actuators_do_not_import_any_physics_domain():
    """``apply``的物理没落地，所以本模块够不着``state``——环归属就是这么定的。"""

    source = actuators.__file__ or ""
    assert source.endswith("actuators.py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in (
        "physics_engine.state",
        "physics_engine.energies",
        "physics_engine.solve",
    ):
        assert forbidden not in text
