"""张力闭环装配层的门（决策0070，plans/15第三节1.2与1.5）。

案例级的判据（三档阶跃、压制比、可插拔、稳定界、绞盘）在
`tests/cases/test_closed_loop_tension_step.py`。**本文件判的是装配本身**：
两条时钟对没对上、零阶保持是不是真的、失败关闭挡不挡得住。

## 本文件最要紧的一条：时延线的``dt_s``必须等于控制周期

`ActuationDelayLine`按**拍数**记时延（0038）。拿推进步长去填它，
声明的``delay_s``就会变成另一个时间——**而它不会报错，只会给出一个
看起来很合理的错数**。裁决B第3条把它做成失败关闭，本文件判那条关闭。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.actuators import (
    ActuationCommand,
    ActuationDelayLine,
    ActuatorDeclaration,
    CommandChannel,
)
from physics_engine.drives import (
    DriveError,
    MagneticParticleClutch,
    PidController,
    TensionSensor,
)
from physics_engine.tension_control import (
    CapstanSpan,
    ClosedTensionLoop,
    TensionControlError,
    closed_loop_characteristic_polynomial,
    integral_gain_stability_limit,
)
from physics_engine.transport import FreeSpan, PayoutReel

#: 与`cases/closed_loop_tension_step`同一组**假设输入**。
SPAN = FreeSpan(span_id="span/free", geometric_length_mm=300.0, axial_stiffness_n=60000.0)
REEL = PayoutReel(
    reel_id="reel/payout", radius_mm=60.0, inertia_kg_mm2=5000.0,
    bearing_damping_nmm_s=50.0,
)
CLUTCH = MagneticParticleClutch(
    torque_per_ampere_nmm=23256.0, rated_torque_nmm=50000.0, lag_s=0.0005
)
BRAKE_TORQUE_NMM = 1200.0
LINE_SPEED_MM_S = 20.0
PLANT_DT_S = 1.0e-5


def _controller(**overrides) -> PidController:
    base = {
        "proportional": 0.0077399,
        "integral_gain": 2.3909,
        "derivative": 1.7561e-5,
        "integral_limit": 1.0e6,
    }
    return PidController(**{**base, **overrides})


def _loop(**overrides) -> ClosedTensionLoop:
    base = {
        "span": SPAN,
        "reel": REEL,
        "clutch": CLUTCH,
        "controller": _controller(),
        "capstan": None,
        "sensor": None,
        "plant_dt_s": PLANT_DT_S,
        "control_decimation": 1,
        "brake_torque_nmm": BRAKE_TORQUE_NMM,
        "line_speed_mm_s": LINE_SPEED_MM_S,
        "delay_line": None,
        "forbid_slack": True,
    }
    return ClosedTensionLoop.at_steady_state(**{**base, **overrides})


def _delay_line(delay_s: float, dt_s: float) -> ActuationDelayLine:
    declaration = ActuatorDeclaration(
        actuator_id="actuator/tension_clutch",
        kind="magnetic_particle_clutch",
        channels=(
            CommandChannel(
                channel_id="command/coil",
                quantity_id="coil_current_amp",
                dimension=1,
                lower=(-3.0,),
                upper=(3.0,),
            ),
        ),
        delay_s=delay_s,
        zero_delay_rationale=None if delay_s > 0.0 else "判据构型：本档专量别的东西",
    )
    return ActuationDelayLine.declare(
        declaration=declaration,
        dt_s=dt_s,
        quantization="exact",
        initial_command=ActuationCommand(
            actuator_id="actuator/tension_clutch",
            values=(BRAKE_TORQUE_NMM / CLUTCH.torque_per_ampere_nmm,),
        ),
    )


# ---------------------------------------------------------------------------
# 稳态起点：对象、执行器、控制器三者同时在不动点上
# ---------------------------------------------------------------------------


def test_the_steady_state_entry_puts_all_three_on_the_fixed_point():
    """从闭式稳态起手，两端等速时**一步都不动**。

    `SpanTransportLoop`那一半已经证过"离散不动点＝连续不动点"（0066）。
    本条要证的是装配之后**仍然**如此：前馈电流恰好维持``M₀``、
    误差型控制器输出零、离合器的ZOH更新是恒等。

    **红了说明前馈没接对**：起点若前馈不住，"从稳态起手"这句话不成立，
    而此后每一条阶跃判据都会被起点的瞬态污染。
    """

    loop = _loop()
    assert loop.feedforward_current_a == pytest.approx(
        BRAKE_TORQUE_NMM / CLUTCH.torque_per_ampere_nmm, rel=1e-15
    )
    assert loop.held_current_a == loop.feedforward_current_a
    assert loop.brake_torque_nmm == BRAKE_TORQUE_NMM

    base = loop.tension_n
    final, samples = loop.run(2000, takeup_speed_mm_s=LINE_SPEED_MM_S)
    drift = max(abs(sample.tension_n - base) for sample in samples)
    assert drift < 1.0e-9, f"稳态起点自己漂了{drift!r} N —— 三者没有同时在不动点上"
    assert final.brake_torque_nmm == pytest.approx(BRAKE_TORQUE_NMM, rel=1e-12)
    assert all(sample.error_n == pytest.approx(0.0, abs=1e-9) for sample in samples)


def test_the_plant_advances_on_the_step_start_torque_bit_for_bit():
    """**因果次序**：对象用**步首**扭矩推进，不用离合器刚刚走完那一步的新扭矩。

    ## 这条门是注错验证补出来的

    2026-08-17第一轮注错：把两句对调（先推进离合器、再拿**步末**扭矩喂对象），
    **九条注错里唯一一条没红的**。病根是那个差是``O(dt)``的，
    而``dt = 1e-6``下它落在每一条容差之下——**一道分辨不出因果次序的门，
    等于没有判过因果次序**。

    ## 补法：一步、逐位、而且先证明这条门不是空的

    从稳态起手但让控制器吐一个**恒定的偏置电流**，于是离合器扭矩在这一步里
    真的在动（``dt = 1e-4``、``τ = 5e-4`` ⟹ 一步走完18%的路）。
    然后拿**步首**扭矩独立算一遍角加速度，与实现给的新转速**逐位**对拍。

    第二条断言判的是**这条门有没有内容**：步首扭矩与步末扭矩必须真的不同。
    它们相等时上面那条逐位断言照样过，而那时它什么也没证明——
    **本仓已经因为"一道从没被注错验过的门"吃过亏**（0066第5.2节）。
    """

    class ConstantCommand:
        """一个既不是`PidController`、也不看测量的控制器。**协议只要求``step``。**"""

        def step(self, *, measurement_n, setpoint_n, dt_s):
            return self, 0.02

    coarse_dt = 1.0e-4
    loop = _loop(controller=ConstantCommand(), plant_dt_s=coarse_dt)
    final, samples = loop.run(1, takeup_speed_mm_s=LINE_SPEED_MM_S + 2.0)
    first = samples[0]

    #: **先证明这条门不是空的**：步首扭矩与步末扭矩必须真的不同。
    assert first.brake_torque_nmm != final.brake_torque_nmm, (
        "这一步里离合器扭矩没动 —— 那么'用步首还是步末'这道门什么也没证明，"
        "偏置电流或步长要重挑"
    )

    #: 用**步首**扭矩独立算一遍——这是实现该走的那条路。
    right = REEL.angular_acceleration_rad_s2(
        tension_n=first.tension_n,
        brake_torque_nmm=first.brake_torque_nmm,
        angular_velocity_rad_s=first.angular_velocity_rad_s,
    )
    assert final.plant.angular_velocity_rad_s == (
        first.angular_velocity_rad_s + coarse_dt * right
    ), "对象没有用步首扭矩推进 —— 测量与执行的因果次序反了"

    #: 用**步末**扭矩会给出另一个数——两条必须分得开，否则上面那条判了个寂寞。
    wrong = REEL.angular_acceleration_rad_s2(
        tension_n=first.tension_n,
        brake_torque_nmm=final.brake_torque_nmm,
        angular_velocity_rad_s=first.angular_velocity_rad_s,
    )
    assert final.plant.angular_velocity_rad_s != (
        first.angular_velocity_rad_s + coarse_dt * wrong
    )


def test_a_saturating_feedforward_fails_closed():
    """前馈电流落在饱和段上时，"从稳态起手"这句话不成立——当场关闭。"""

    weak = MagneticParticleClutch(
        torque_per_ampere_nmm=23256.0, rated_torque_nmm=100.0, lag_s=0.0005
    )
    with pytest.raises(TensionControlError, match="饱和"):
        _loop(clutch=weak)


# ---------------------------------------------------------------------------
# 裁决B：两条时钟
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decimation", [0, -1, 1.5, True, "10"])
def test_a_malformed_decimation_fails_closed(decimation):
    """**必须红**：抽取比不是正整数时当场关闭。

    非整数比值会让某个控制拍落在推进步中间，而那一步的零阶保持边界
    **无从定义**——"前半段用旧命令、后半段用新命令"要么改推进格式，
    要么就是一次静默的近似。**本层不做静默近似。**
    """

    with pytest.raises(TensionControlError, match="control_decimation"):
        _loop(control_decimation=decimation)


def test_the_control_period_is_the_decimation_times_the_plant_step():
    """两条时钟的关系就是这一行，而且它必须是**逐位**的乘积。"""

    for decimation in (1, 3, 100):
        loop = _loop(control_decimation=decimation)
        assert loop.control_period_s == decimation * PLANT_DT_S


def test_the_command_is_held_between_control_ticks():
    """**零阶保持是真的**：两个控制拍之间生效电流一个字不变。

    保持的是**电流**不是扭矩——电流是执行器的输入，扭矩是它的状态。
    保持扭矩等于把磁滞旁路掉，而那会让离合器的一阶时滞在多速率下消失。
    **本条判的正是"保持的是哪一个"**：扭矩必须每步都在动，电流必须只在拍上动。
    """

    decimation = 10
    loop = _loop(control_decimation=decimation)
    _, samples = loop.run(50, takeup_speed_mm_s=LINE_SPEED_MM_S + 2.0)

    ticks = [index for index, s in enumerate(samples) if s.control_tick]
    assert ticks == list(range(0, 50, decimation))
    for index, sample in enumerate(samples):
        if index % decimation != 0:
            assert sample.current_a == samples[index - index % decimation].current_a, (
                f"第{index}步不是控制拍，电流却变了 —— 零阶保持没生效"
            )
    #: 而扭矩必须**每步都在动**（磁滞在走），否则保持的是错的那个量。
    torques = {sample.brake_torque_nmm for sample in samples}
    assert len(torques) == len(samples), "扭矩在非控制拍上不动 —— 磁滞被旁路了"


def test_a_delay_line_clocked_on_the_plant_step_fails_closed():
    """**必须红，而且是本文件最要紧的一条**：时延线的``dt_s``必须等于控制周期。

    `ActuationDelayLine`按**拍数**记时延。用推进步长去填它时，
    声明的``delay_s``会变成另一个时间——**而它不会报错**，
    只会给出一个看起来很合理的错数。这正是"两条时钟对不齐"最常见的形态。

    抽取比为1时两者相等，那一档必须**过**；抽取比不为1时必须**红**。
    """

    #: 抽取比1：时延线按推进步长走，正好就是控制周期 ⟹ 过。
    _loop(control_decimation=1, delay_line=_delay_line(0.0, PLANT_DT_S))

    #: 抽取比10：时延线仍按推进步长走 ⟹ 声明的时延会被记成十分之一 ⟹ 红。
    with pytest.raises(TensionControlError, match="控制周期"):
        _loop(control_decimation=10, delay_line=_delay_line(0.0, PLANT_DT_S))

    #: 对齐之后同一条时延线过得去。
    loop = _loop(control_decimation=10, delay_line=_delay_line(0.0, 10 * PLANT_DT_S))
    assert loop.control_period_s == 10 * PLANT_DT_S


def test_a_declared_delay_shifts_the_command_by_whole_control_ticks_and_costs_margin():
    """时延以**控制拍**计，而且它**要付代价**。

    三条一起判，因为分开判每一条都能被糊弄过去：

    1. ``delay_s = 3 × 控制周期`` ⟹ ``steps == 3``（**整拍**，不是推进步）；
    2. 前3个控制拍生效的是初值，**逐位相同**——环形缓冲还没吐出新命令；
    3. **同一套增益下，带时延那一趟的峰值更大**。

    第3条是这条门的物理内容。spec/10第三节的原话是"没有时延训练出的策略
    上真机会翻车"，`tests/test_drives.py`已经把它在`TensionLoop`上量成过
    一张表（20 ms把超调翻了一倍还多）。**闭环这一侧也必须付同一笔账**——
    只判1与2的话，一个把时延线接上却不让它influence任何东西的实现照样全绿。
    """

    decimation = 5
    control_period = decimation * PLANT_DT_S
    line = _delay_line(3 * control_period, control_period)
    assert line.steps == 3, "声明的时延没有按控制拍取整"

    delayed = _loop(control_decimation=decimation, delay_line=line)
    _, delayed_samples = delayed.run(4000, takeup_speed_mm_s=LINE_SPEED_MM_S + 2.0)

    ticks = [s for s in delayed_samples if s.control_tick]
    initial = BRAKE_TORQUE_NMM / CLUTCH.torque_per_ampere_nmm
    for index, sample in enumerate(ticks[:3]):
        assert sample.current_a == initial, (
            f"第{index}个控制拍就吐出了新命令（{sample.current_a!r}）—— "
            "环形缓冲深度3意味着前三拍必须是初值，逐位相同"
        )

    #: 同一套增益、同一个抽取比，**只加时延**。
    undelayed = _loop(
        control_decimation=decimation, delay_line=_delay_line(0.0, control_period)
    )
    _, undelayed_samples = undelayed.run(4000, takeup_speed_mm_s=LINE_SPEED_MM_S + 2.0)

    def peak(samples, setpoint):
        return max(sample.tension_n - setpoint for sample in samples)

    with_delay = peak(delayed_samples, delayed.setpoint_n)
    without_delay = peak(undelayed_samples, undelayed.setpoint_n)
    assert with_delay > without_delay, (
        f"加了{line.steps}拍下发时延而峰值没有变大（{without_delay!r} → "
        f"{with_delay!r}）—— 时延线接上了却没有influence任何东西"
    )


# ---------------------------------------------------------------------------
# 绞盘观测层
# ---------------------------------------------------------------------------


def test_the_capstan_side_has_no_default():
    """**没有默认值是有意的**：方向是一条声明，而搞反的代价是平方。"""

    with pytest.raises(TypeError):
        CapstanSpan(friction_coefficient=0.3, wrap_angle_rad=math.pi / 2)  # type: ignore[call-arg]


@pytest.mark.parametrize("side", [0, 1, "tight", None])
def test_a_non_boolean_capstan_side_fails_closed(side):
    with pytest.raises(TensionControlError, match="sensor_on_tight_side"):
        CapstanSpan(
            friction_coefficient=0.3, wrap_angle_rad=math.pi / 2, sensor_on_tight_side=side
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"friction_coefficient": -0.1}, "friction"),
        ({"wrap_angle_rad": -1.0}, "wrap angle"),
        ({"friction_coefficient": float("nan")}, "friction"),
    ],
)
def test_a_malformed_capstan_defers_to_the_drives_validator(kwargs, message):
    """参数合法性**交给`drives.capstan_transfer_ratio`判，不重写一遍**。

    重写等于开第二条真相源，而两条真相源迟早会分叉。
    本条同时钉住"错误从哪个模块出来"——它出来的是`DriveError`不是本模块的错。
    """

    base = {
        "friction_coefficient": 0.3,
        "wrap_angle_rad": math.pi / 2,
        "sensor_on_tight_side": False,
    }
    with pytest.raises(DriveError, match=message):
        CapstanSpan(**{**base, **kwargs})


def test_the_capstan_ratio_is_one_when_there_is_no_wrap():
    """零包角 ⟹ 比值恰为1 ⟹ 两个方向给出**同一个数**。

    这是"传感器就在被控点上"那条声明的退化形。零容差：``exp(0) = 1``。
    """

    for side in (True, False):
        capstan = CapstanSpan(
            friction_coefficient=0.3, wrap_angle_rad=0.0, sensor_on_tight_side=side
        )
        assert capstan.ratio == 1.0
        assert capstan.laydown_tension_n(17.5) == 17.5


# ---------------------------------------------------------------------------
# 形制失败关闭
# ---------------------------------------------------------------------------


def test_a_controller_that_is_not_a_controller_fails_closed():
    class NotAController:
        pass

    with pytest.raises(TensionControlError, match="TensionController"):
        _loop(controller=NotAController())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"span": "not a span"}, "SpanTransportLoop|geometric"),
        ({"clutch": "not a clutch"}, "torque_per_ampere_nmm|MagneticParticleClutch"),
    ],
)
def test_a_malformed_assembly_fails_closed(kwargs, message):
    with pytest.raises((TensionControlError, AttributeError, TypeError, DriveError)):
        _loop(**kwargs)


def test_a_wrong_typed_capstan_or_sensor_fails_closed():
    with pytest.raises(TensionControlError, match="CapstanSpan"):
        _loop(capstan=1.6)
    with pytest.raises(TensionControlError, match="TensionSensor"):
        _loop(sensor=12)


def test_the_sensor_quantisation_reaches_the_controller():
    """传感器在环里是**真的**：ADC台阶决定了读数的粒度。

    LTS1-5是5 kg满量程（≈49 N）、12位。台阶是``49/4095 = 0.011966 N``——
    **它是这条回路精度的地板**，闭环再准也停不到台阶之间。
    """

    sensor = TensionSensor(full_scale_n=49.0, output_at_full_scale_mv=20.0, adc_bits=12)
    loop = _loop(sensor=sensor)
    _, samples = loop.run(200, takeup_speed_mm_s=LINE_SPEED_MM_S + 2.0)
    step = sensor.resolution_n
    for sample in samples:
        remainder = sample.measured_n / step
        assert remainder == pytest.approx(round(remainder), abs=1e-9), (
            f"读数{sample.measured_n!r}没有落在ADC台阶上 —— 量化没进环"
        )
    #: 而真值**不**落在台阶上——否则量化根本没起作用。
    assert any(
        abs(sample.tension_n / step - round(sample.tension_n / step)) > 1e-6
        for sample in samples
    )


# ---------------------------------------------------------------------------
# 闭式辅助函数的失败关闭
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"span_stiffness_n_per_mm": 0.0}, "span_stiffness"),
        ({"radius_mm": -1.0}, "radius"),
        ({"inertia_kg_mm2": 0.0}, "inertia"),
        ({"bearing_damping_nmm_s": -1.0}, "阻力矩"),
        ({"torque_per_ampere_nmm": 0.0}, "torque_per_ampere"),
        ({"clutch_lag_s": -1.0}, "反因果"),
        ({"proportional": float("nan")}, "proportional"),
    ],
)
def test_a_malformed_characteristic_polynomial_fails_closed(kwargs, message):
    base = {
        "span_stiffness_n_per_mm": 200.0,
        "radius_mm": 60.0,
        "inertia_kg_mm2": 5000.0,
        "bearing_damping_nmm_s": 50.0,
        "torque_per_ampere_nmm": 23256.0,
        "clutch_lag_s": 0.05,
        "proportional": 0.0,
        "integral_gain": 1.0,
        "derivative": 0.0,
    }
    with pytest.raises(TensionControlError, match=message):
        closed_loop_characteristic_polynomial(**{**base, **kwargs})


def test_the_integral_limit_scales_with_the_open_loop_damping():
    """``Ki_界 ∝ 开环阻尼``——**这条单调性比界的数值更难被凑对**。

    ``τ = 0``时界是``d·Ka/G``而``d = 2ζω_n``，于是轴承阻力矩翻倍界就翻倍。
    实测（``c = 25/50/100``）比值恰为``0.5/1/2``。

    **它说的话**：开环阻尼是``ζ = 0.0132``那一档，所以这条界很低——
    积分项在这条链路上压振荡的能力是**负的**。
    """

    def limit(damping: float) -> float:
        return integral_gain_stability_limit(
            span_stiffness_n_per_mm=200.13520802897804,
            radius_mm=60.0,
            inertia_kg_mm2=5000.0,
            bearing_damping_nmm_s=damping,
            torque_per_ampere_nmm=23256.0,
            clutch_lag_s=0.0,
            proportional=0.0,
            derivative=0.0,
        )

    base = limit(50.0)
    assert limit(25.0) == pytest.approx(0.5 * base, rel=1e-14)
    assert limit(100.0) == pytest.approx(2.0 * base, rel=1e-14)

    #: 而``Kd``只把界抬高、不改这条正比关系——**加阻尼的是它**。
    with_derivative = integral_gain_stability_limit(
        span_stiffness_n_per_mm=200.13520802897804, radius_mm=60.0,
        inertia_kg_mm2=5000.0, bearing_damping_nmm_s=50.0,
        torque_per_ampere_nmm=23256.0, clutch_lag_s=0.0,
        proportional=0.0, derivative=1.0e-5,
    )
    assert with_derivative > base
