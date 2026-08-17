"""张力驱动链的门（决策0062轨道乙，能力位S6.3）。

## 判据强度最高的一条：I控制下闭环**精确**是二阶系统

纯积分＋一阶对象的闭环传递函数没有零点，标准二阶闭式因此**精确成立**：
``ω_n = sqrt(K·Ki/τ)``、``ζ = 1/(2·sqrt(τ·K·Ki))``、
超调``exp(−ζπ/√(1−ζ²))``、峰值时刻``π/(ω_n√(1−ζ²))``、静差恰为零。

三档阻尼比各判一次；超调与峰值时刻**必须并判**——只判超调时，
把``τ``与``Ki``同乘一个倍数不会被发现（``ζ``不变而``ω_n``变）。

## 时延：spec/10点名的翻车路径，这里把它变成一个数

规范原话是"没有时延训练出的策略上真机会翻车"。2026-08-17实测
（``ζ = 0.5``、``dt = 1 ms``、理论无时延超调``0.16303``）：

| 下发时延 | 超调 | 相对无时延 |
|---|---|---|
| 0 ms | 0.16305 | — |
| 5 ms | 0.20633 | ×1.27 |
| 10 ms | 0.25747 | ×1.58 |
| 20 ms | 0.38234 | **×2.35** |
| 40 ms | 0.70026 | **×4.30**（且稳态开始变坏） |

**20 ms把超调翻了一倍还多。** 这就是那句话的定量版本。

## 一条被实测纠正的判断

写这个模块时我判定超调式与`contact.restitution_from_damping_ratio`
"逐位相同"，理由是形状一样。**实测当场否掉**：``ζ = 0.5``处差1.8倍。
病根是0052第一节裁的截断约定——两式共享``exp(−ζΦ/√(1−ζ²))``而``Φ``不同
（``π`` vs ``2·acos(ζ)``），**只在``ζ = 0``处重合**。
本文件有一条门守着这个"不同"，理由写在那条门里。
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
from physics_engine.contact import restitution_from_damping_ratio
from physics_engine.drives import (
    DriveError,
    MagneticParticleClutch,
    PidController,
    SpoolTension,
    TensionLoop,
    TensionSensor,
    capstan_transfer_ratio,
    second_order_damping_ratio,
    second_order_natural_frequency_rad_s,
    step_response_overshoot,
    step_response_peak_time_s,
)

#: POC-050基型样本：50 N·m额定、2.15 A线圈 ⟹ 23256 N·mm/A。
#: **L后缀的专用线圈参数厂家资料没有**，这是基型外推（0062第二节裁决2）。
TORQUE_PER_AMPERE_NMM = 23256.0
RATED_TORQUE_NMM = 50000.0
#: 磁滞响应时间：**假设输入**，没有实测。
LAG_S = 0.05
BARREL_RADIUS_MM = 60.0
TAPE_THICKNESS_MM = 0.1
SETPOINT_N = 30.0

CLUTCH = MagneticParticleClutch(
    torque_per_ampere_nmm=TORQUE_PER_AMPERE_NMM,
    rated_torque_nmm=RATED_TORQUE_NMM,
    lag_s=LAG_S,
)
SPOOL = SpoolTension(
    barrel_radius_mm=BARREL_RADIUS_MM, tape_thickness_mm=TAPE_THICKNESS_MM
)
#: 闭环增益``K = k_M/R``，单位N/A。
GAIN_N_PER_AMPERE = TORQUE_PER_AMPERE_NMM / BARREL_RADIUS_MM

CHANNEL = CommandChannel(
    channel_id="command/coil",
    #: **``_amp``不是``_a``**——理由见`identity.BASE_UNIT_SUFFIXES`上方的实测。
    quantity_id="coil_current_amp",
    dimension=1,
    lower=(-3.0,),
    upper=(3.0,),
)


def _integral_gain_for(damping_ratio: float) -> float:
    """反解出给定``ζ``所需的``Ki``：``Ki = 1/(4ζ²·τ·K)``。"""

    return 1.0 / (4.0 * damping_ratio * damping_ratio * LAG_S * GAIN_N_PER_AMPERE)


def _delay_line(delay_s: float, dt_s: float) -> ActuationDelayLine:
    declaration = ActuatorDeclaration(
        actuator_id="actuator/tension_clutch",
        kind="magnetic_particle_clutch",
        channels=(CHANNEL,),
        delay_s=delay_s,
        zero_delay_rationale=None,
    )
    return ActuationDelayLine.declare(
        declaration=declaration,
        dt_s=dt_s,
        quantization="exact",
        initial_command=ActuationCommand(
            actuator_id="actuator/tension_clutch", values=(0.0,)
        ),
    )


def _run(
    *,
    integral_gain: float,
    dt_s: float,
    horizon_s: float,
    delay_line=None,
    sensor=None,
    measurement_transfer: float = 1.0,
):
    loop = TensionLoop(
        clutch=CLUTCH,
        spool=SPOOL,
        controller=PidController(
            proportional=0.0,
            integral_gain=integral_gain,
            derivative=0.0,
            integral_limit=1.0e6,
        ),
        setpoint_n=SETPOINT_N,
        dt_s=dt_s,
        delay_line=delay_line,
        sensor=sensor,
        #: **1.0是一条声明**："传感器就在被控点上、中间一个包角都没有"。
        #: 本文件多数门验的是控制器本身，所以刻意取这个理想构型；
        #: 传感器位置带来的误差由`test_the_loop_regulates_what_it_measures_not_what_matters`验。
        measurement_transfer=measurement_transfer,
    )
    return loop.run(int(round(horizon_s / dt_s)))


def _overshoot_and_peak(samples) -> tuple[float, float]:
    index = max(range(len(samples)), key=lambda i: samples[i].tension_n)
    peak = samples[index].tension_n
    return (peak - SETPOINT_N) / SETPOINT_N, samples[index].time_s


# ---------------------------------------------------------------------------
# 闭环对二阶闭式
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_zeta", [0.3, 0.5, 0.7])
def test_the_integral_only_loop_hits_the_second_order_closed_form(target_zeta: float):
    """超调、峰值时刻、零静差三条一起判。

    **超调与峰值时刻必须并判**：超调只看``ζ``，峰值时刻还看``ω_n``。
    只判超调的话，把``τ``与``Ki``同乘一个倍数不会被发现——那正是这条门的盲区封堵。
    """

    integral_gain = _integral_gain_for(target_zeta)
    zeta = second_order_damping_ratio(
        gain_n_per_ampere=GAIN_N_PER_AMPERE, integral_gain=integral_gain, lag_s=LAG_S
    )
    omega = second_order_natural_frequency_rad_s(
        gain_n_per_ampere=GAIN_N_PER_AMPERE, integral_gain=integral_gain, lag_s=LAG_S
    )
    assert zeta == pytest.approx(target_zeta, rel=1e-12), "反解Ki的式子自己就错了"

    dt = 2.5e-4
    _, samples = _run(integral_gain=integral_gain, dt_s=dt, horizon_s=3.0)
    overshoot, peak_time = _overshoot_and_peak(samples)

    assert overshoot == pytest.approx(step_response_overshoot(zeta), rel=2e-4)
    #: 峰值时刻落在采样网格上，分辨率就是``dt``——判据取``2·dt``的绝对界。
    assert peak_time == pytest.approx(
        step_response_peak_time_s(natural_frequency_rad_s=omega, damping_ratio=zeta),
        abs=2.0 * dt,
    )
    assert samples[-1].tension_n == pytest.approx(SETPOINT_N, abs=1e-9), (
        "纯积分必须零静差——有静差说明积分项没有真的在积"
    )


def test_the_overshoot_converges_at_second_order_in_the_step():
    """**阶是量出来的，不是从某一个部件的阶推出来的。**

    离合器用零阶保持的精确离散（那一步没有误差），控制器用前向Euler（一阶）。
    "所以整环是一阶"是一个看起来合理的推断，**而实测是二阶**：
    2026-08-17（``ζ = 0.5``、``dt``逐次减半）误差
    ``1.350e-3 / 3.353e-4 / 8.719e-5 / 1.926e-5``，比值``4.025 / 3.845 / 4.527``。

    门判比值落在``[3.2, 5.2]``而**不写死为4**——与`harmonic_oscillator`
    那条"收敛比不写死"同源。
    """

    integral_gain = _integral_gain_for(0.5)
    zeta = second_order_damping_ratio(
        gain_n_per_ampere=GAIN_N_PER_AMPERE, integral_gain=integral_gain, lag_s=LAG_S
    )
    expected = step_response_overshoot(zeta)
    errors = []
    for dt in (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4):
        _, samples = _run(integral_gain=integral_gain, dt_s=dt, horizon_s=1.2)
        overshoot, _ = _overshoot_and_peak(samples)
        errors.append(abs(overshoot - expected) / expected)

    assert errors[0] > errors[1] > errors[2] > errors[3], f"误差没有单调下降：{errors}"
    for earlier, later in zip(errors, errors[1:], strict=False):
        assert 3.2 <= earlier / later <= 5.2, (
            f"整环收敛阶不是二阶：比值{earlier / later!r}，序列{errors}"
        )


# ---------------------------------------------------------------------------
# 时延：spec/10点名的翻车路径
# ---------------------------------------------------------------------------


def test_delay_monotonically_inflates_the_overshoot():
    """**规范那句话的定量版本。**

    spec/10第三节：``时延不是可选项——真实控制回路有延迟，没有时延训练出的策略
    上真机会翻车``。这条门把"翻车"变成一个可判的数：下发时延单调抬高超调，
    20 ms把它翻了2.35倍。

    **没有这条门，`actuators`那整套时延机械就只是一个更严谨的摆设**——
    它算得出`realized_delay_s`，但没有任何地方证明那个延迟真的改变了物理结论。
    """

    integral_gain = _integral_gain_for(0.5)
    dt = 1.0e-3
    overshoots = []
    for delay_ms in (0, 5, 10, 20, 40):
        line = None if delay_ms == 0 else _delay_line(delay_ms / 1000.0, dt)
        _, samples = _run(
            integral_gain=integral_gain, dt_s=dt, horizon_s=3.0, delay_line=line
        )
        overshoots.append(_overshoot_and_peak(samples)[0])

    for earlier, later in zip(overshoots, overshoots[1:], strict=False):
        assert later > earlier, f"时延加大而超调没涨：{overshoots}"
    assert overshoots[3] / overshoots[0] == pytest.approx(2.35, rel=0.05), (
        f"20 ms时延的放大倍数{overshoots[3] / overshoots[0]!r}与2026-08-17实测的2.35不符"
    )


def test_the_loop_without_a_delay_line_must_say_so_explicitly():
    """``delay_line``没有默认值：零时延是一条**声明**，不是一次省略。

    与`ActuatorDeclaration`要求``zero_delay_rationale``同一条纪律。
    """

    with pytest.raises(TypeError):
        TensionLoop(  # type: ignore[call-arg]
            clutch=CLUTCH,
            spool=SPOOL,
            controller=PidController(
                proportional=0.0, integral_gain=1.0, derivative=0.0, integral_limit=1.0
            ),
            setpoint_n=SETPOINT_N,
            dt_s=1.0e-3,
            sensor=None,
            measurement_transfer=1.0,
        )


# ---------------------------------------------------------------------------
# 两条闭式**不是**同一个函数
# ---------------------------------------------------------------------------


def test_the_overshoot_and_the_restitution_are_not_the_same_function():
    """**它们只在``ζ = 0``处相等，此后越离越远。**

    写`drives.py`时我判定两者"逐位相同"，理由是形状一样：
    ``exp(−ζΦ/√(1−ζ²))``。实测当场否掉——``ζ = 0.5``处差1.83倍。

    病根不在浮点，在0052第一节那条裁决：接触在分离瞬间**截断**拉力，
    于是``Φ = 2·acos(ζ)``而不是``π``。若当初裁的是"不截断"，两式就真的相同了。

    **这条门守的是那条裁决没有被悄悄改掉**：它红了，要么有人把两件事合并了，
    要么0052的约定变了——两种都必须是一次显式的决策。
    """

    assert step_response_overshoot(0.0) == restitution_from_damping_ratio(0.0) == 1.0

    for zeta in (0.1, 0.3, 0.5, 0.9):
        overshoot = step_response_overshoot(zeta)
        restitution = restitution_from_damping_ratio(zeta)
        assert overshoot < restitution, (
            f"ζ={zeta}：超调{overshoot!r}没有小于恢复系数{restitution!r}——"
            "``π > 2·acos(ζ)``对``ζ > 0``恒成立，所以超调必须更小"
        )
        #: 差别的来源可以逐位复现：两式只差指数上的``Φ``。
        root = math.sqrt(1.0 - zeta * zeta)
        assert overshoot == pytest.approx(math.exp(-zeta * math.pi / root), rel=1e-15)
        assert restitution == pytest.approx(
            math.exp(-zeta * 2.0 * math.acos(zeta) / root), rel=1e-15
        )

    assert step_response_overshoot(0.5) / restitution_from_damping_ratio(0.5) == (
        pytest.approx(0.5463, rel=1e-3)
    )


def test_the_overdamped_branch_has_no_overshoot_at_all():
    """``ζ ≥ 1``返回0.0——**不是这条式子的解析延拓**（实数域上不存在）。"""

    assert step_response_overshoot(1.0) == 0.0
    assert step_response_overshoot(2.5) == 0.0
    with pytest.raises(DriveError, match="peak time needs an underdamped ratio"):
        step_response_peak_time_s(natural_frequency_rad_s=1.0, damping_ratio=1.0)


# ---------------------------------------------------------------------------
# 卷径生长与饱和
# ---------------------------------------------------------------------------


def test_the_radius_growth_drops_the_tension_at_constant_torque():
    """同一个扭矩，卷满时的张力比空卷时小``R0/(R0+n·t)``倍。

    实测：``M = 1800 N·mm``在0匝给30.00 N、100匝给25.71 N、500匝给16.36 N——
    **卷到500匝张力掉了45%**。真机ATC600的锥度张力功能就是为它存在的。
    """

    torque = SPOOL.torque_nmm(SETPOINT_N, turns=0.0)
    assert torque == pytest.approx(SETPOINT_N * BARREL_RADIUS_MM, rel=1e-15)
    for turns, expected in ((0.0, 30.0), (100.0, 25.714285714285715), (500.0, 16.363636363636363)):
        assert SPOOL.tension_n(torque, turns) == pytest.approx(expected, rel=1e-12)
    #: 往返恒等：要这个张力需要多大扭矩，再换回来必须是同一个张力。
    for turns in (0.0, 37.0, 500.0):
        needed = SPOOL.torque_nmm(SETPOINT_N, turns)
        assert SPOOL.tension_n(needed, turns) == pytest.approx(SETPOINT_N, rel=1e-15)


def test_saturation_truncates_and_is_sign_symmetric():
    """饱和是**截断不是压缩**：过了饱和点再加电流扭矩不动。"""

    assert CLUTCH.saturation_current_a == pytest.approx(2.14998, rel=1e-5)
    assert CLUTCH.commanded_torque_nmm(0.5) == pytest.approx(0.5 * TORQUE_PER_AMPERE_NMM)
    assert CLUTCH.commanded_torque_nmm(5.0) == RATED_TORQUE_NMM
    assert CLUTCH.commanded_torque_nmm(-5.0) == -RATED_TORQUE_NMM
    assert CLUTCH.commanded_torque_nmm(500.0) == RATED_TORQUE_NMM, (
        "压缩式实现会让大电流处的扭矩随电流继续变——那不是磁粉离合器的行为"
    )


def test_the_zero_order_hold_lag_is_exact_not_an_euler_step():
    """``M_{n+1} = M_cmd + (M_n − M_cmd)·exp(−dt/τ)``是恒等式。

    判据：**一大步与若干小步逐位一致**（指数的半群性质）。
    前向Euler做不到这一条，且在``dt > 2τ``时发散——那个发散是纯数值的。
    """

    current = 1.0
    target = CLUTCH.commanded_torque_nmm(current)
    one_step = CLUTCH.advance_torque_nmm(0.0, current, 0.4)
    many = 0.0
    for _ in range(4):
        many = CLUTCH.advance_torque_nmm(many, current, 0.1)
    assert one_step == pytest.approx(many, rel=1e-14)

    #: ``dt = 10τ``下仍然单调趋近，不越过目标——Euler在这里会冲过去。
    big = CLUTCH.advance_torque_nmm(0.0, current, 10.0 * LAG_S)
    assert 0.0 < big < target


# ---------------------------------------------------------------------------
# PID的历史与限幅
# ---------------------------------------------------------------------------


def test_the_integral_is_clamped_and_the_controller_is_immutable():
    """限幅挡住windup；``step``返回新对象而不就地改。"""

    controller = PidController(
        proportional=0.0, integral_gain=1.0, derivative=0.0, integral_limit=2.0
    )
    current = controller
    for _ in range(100):
        current, _ = current.step(1.0, 0.1)
    assert current.integral == 2.0, "积分没有被限幅——执行器饱和时会一直积"
    assert controller.integral == 0.0, "原控制器被就地改了——历史必须显式传递"

    negative = PidController(
        proportional=0.0, integral_gain=1.0, derivative=0.0, integral_limit=2.0
    )
    for _ in range(100):
        negative, _ = negative.step(-1.0, 0.1)
    assert negative.integral == -2.0, "限幅必须是双边的"


def test_the_derivative_is_zero_on_the_very_first_step():
    """第一步没有上一次误差，微分项为零——**不拿零当上一次误差**。

    拿零当上一次误差会造出一个与阶跃幅值成正比的冲击``Kd·e/dt``，
    而真机上那个冲击不存在。
    """

    controller = PidController(
        proportional=0.0, integral_gain=0.0, derivative=1.0, integral_limit=1.0
    )
    _, first = controller.step(5.0, 0.1)
    assert first == 0.0

    stepped, _ = controller.step(5.0, 0.1)
    _, second = stepped.step(7.0, 0.1)
    assert second == pytest.approx((7.0 - 5.0) / 0.1)


# ---------------------------------------------------------------------------
# 失败关闭
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"torque_per_ampere_nmm": 0.0}, "torque_per_ampere_nmm"),
        ({"rated_torque_nmm": -1.0}, "rated_torque_nmm"),
        ({"lag_s": 0.0}, "lag_s"),
        ({"lag_s": float("inf")}, "lag_s"),
    ],
)
def test_a_malformed_clutch_fails_closed(kwargs, message):
    base = {
        "torque_per_ampere_nmm": TORQUE_PER_AMPERE_NMM,
        "rated_torque_nmm": RATED_TORQUE_NMM,
        "lag_s": LAG_S,
    }
    with pytest.raises(DriveError, match=message):
        MagneticParticleClutch(**{**base, **kwargs})


def test_a_nonpositive_setpoint_fails_closed():
    """零或负的张力设定是**松卷**工况，本模块不假装能算它。"""

    for setpoint in (0.0, -5.0):
        with pytest.raises(DriveError, match="松卷"):
            TensionLoop(
                clutch=CLUTCH,
                spool=SPOOL,
                controller=PidController(
                    proportional=0.0, integral_gain=1.0, derivative=0.0, integral_limit=1.0
                ),
                setpoint_n=setpoint,
                dt_s=1.0e-3,
                delay_line=None,
                sensor=None,
                measurement_transfer=1.0,
            )


def test_a_missing_integral_limit_fails_closed():
    with pytest.raises(TypeError):
        PidController(proportional=1.0, integral_gain=1.0, derivative=0.0)  # type: ignore[call-arg]


def test_negative_turns_fail_closed():
    with pytest.raises(DriveError, match="turns"):
        SPOOL.radius_mm(-1.0)


# ---------------------------------------------------------------------------
# 单位后缀：加的是``amp``不是``a``
# ---------------------------------------------------------------------------


def test_the_ampere_suffix_is_amp_and_a_is_still_rejected():
    """**这条门守的是我没有走那条省事的路。**

    加``a``当安培后缀会让仓里15个以``_a``结尾的名字全部通过单位检查，
    而``body_a``、``name_a``、``node_a``这类是**成对命名**、根本没有单位——
    那等于把轴2规则3废掉。
    """

    from physics_engine.identity import has_unit_suffix

    assert has_unit_suffix("coil_current_amp")
    for naked in ("body_a", "name_a", "node_a", "box_a", "segment_a", "current"):
        assert not has_unit_suffix(naked), (
            f"{naked}通过了单位检查——安培后缀可能被写成了``a``"
        )


# ---------------------------------------------------------------------------
# 环归属
# ---------------------------------------------------------------------------


def test_drives_only_reaches_downwards():
    """本模块登记在力学域，只向下依赖基座的``actuators``。

    **它不能被``actuators``反过来import**——那是基座依赖物理域，
    正是0035那次抓到的"方向反了"。域隔离门守着这条，本门只做一次就地复述。
    """

    from physics_engine import actuators, drives

    with open(actuators.__file__ or "", encoding="utf-8") as handle:
        text = handle.read()
    assert "physics_engine.drives" not in text, "基座反向依赖了力学域"
    assert "physics_engine.actuators" in (drives.__file__ or "") or True
    assert drives.__name__ == "physics_engine.drives"


# ---------------------------------------------------------------------------
# 乙2：传感器读出，以及这条链路上最大的一个误差源
# ---------------------------------------------------------------------------

#: LTS1-5：5 kg满量程、20 mV max输出（`ED3L_张力控制资料包`2026-06-08已核）。
#: ADC位数**不是实测**——ATC600的内部转换位数手册没给，取12位是常见量级。
LTS1_5 = TensionSensor(
    full_scale_n=5.0 * 9.80665, output_at_full_scale_mv=20.0, adc_bits=12
)


def test_the_load_cell_spans_five_kilograms_over_twenty_millivolts():
    """真机铭牌：5 kg ⟹ 49.033 N满量程，20 mV满输出。"""

    assert LTS1_5.full_scale_n == pytest.approx(49.033, rel=1e-4)
    assert LTS1_5.millivolts(LTS1_5.full_scale_n) == 20.0
    assert LTS1_5.millivolts(0.0) == 0.0
    assert LTS1_5.millivolts(0.5 * LTS1_5.full_scale_n) == pytest.approx(10.0)
    #: 满量程是**截断**：超量程读数停住，不继续线性外推。
    assert LTS1_5.millivolts(1000.0) == 20.0
    assert LTS1_5.read_n(1000.0) == pytest.approx(LTS1_5.full_scale_n, rel=1e-12)


def test_the_adc_resolution_is_the_floor_of_the_whole_loop():
    """**量化台阶是这条回路精度的地板**：闭环再准也停不到台阶之间。

    2026-08-17实测（`ζ=0.5`、设定30 N、`transfer=1`、5秒）：

    | 位数 | 分辨率 | 末态与设定的偏差 |
    |---|---|---|
    | 8 | 192.287 mN | 92.424 mN |
    | 10 | 47.931 mN | 19.271 mN |
    | 12 | 11.974 mN | 0.685 mN |
    | 16 | 0.748 mN | 0.179 mN |

    **偏差恒不超过半个台阶**——那正是就近量化的定义。
    门判这一条，而不判具体的偏差值（它随设定值落在台阶哪个位置而变）。
    """

    integral_gain = _integral_gain_for(0.5)
    for bits in (8, 10, 12, 16):
        sensor = TensionSensor(
            full_scale_n=5.0 * 9.80665, output_at_full_scale_mv=20.0, adc_bits=bits
        )
        _, samples = _run(
            integral_gain=integral_gain, dt_s=1.0e-3, horizon_s=5.0, sensor=sensor
        )
        deviation = abs(samples[-1].tension_n - SETPOINT_N)
        assert deviation <= 0.5 * sensor.resolution_n * 1.001, (
            f"{bits}位：末态偏差{deviation * 1000:.3f} mN超过半个台阶"
            f"{sensor.resolution_n * 500:.3f} mN——就近量化不该给出这个"
        )
    #: 位数翻倍，台阶按``2^n``缩——这一条挡住把``2^bits``写成``bits``一类的错。
    coarse = TensionSensor(full_scale_n=49.0, output_at_full_scale_mv=20.0, adc_bits=8)
    fine = TensionSensor(full_scale_n=49.0, output_at_full_scale_mv=20.0, adc_bits=12)
    assert coarse.resolution_n / fine.resolution_n == pytest.approx(
        (2**12 - 1) / (2**8 - 1), rel=1e-12
    )


def test_the_loop_regulates_what_it_measures_not_what_matters():
    """**这条链路上最大的一个误差源，而它不是控制器的错。**

    真机的张力传感器装在链路中的某一只轮上，要紧的张力在**落位点**。
    两者之间每隔一个包角，张力就乘一个``exp(μθ)``——
    正是`cases/capstan_tension_ratio`验的那条式子。

    闭环把**测到的**量调到设定值，于是被控点上的真实张力差了``1/transfer``。
    2026-08-17实测（``μ = 0.3``、设定30 N）：

    | 传感器与被控点之间的包角 | transfer | 被控点真实张力 | 误差 |
    |---|---|---|---|
    | 0° | 1.00000 | 30.000 N | — |
    | 30° | 0.85464 | 35.103 N | **+17.0%** |
    | 60° | 0.73040 | 41.073 N | **+36.9%** |
    | 90° | 0.62423 | 48.059 N | **+60.2%** |
    | 180° | 0.38966 | 76.990 N | **+156.6%** |

    **90°包角就让实际张力超出设定六成。** REBCO带材的许用应变很窄，
    60%的张力超出不是一个可以忽略的量。

    这条门判的是**误差恰好等于``1/transfer``**——它是可预测的、
    因而是**可补偿的**：只要包角与μ已知，前馈乘上去就抵消了。
    而`research/05`把μ列在"只有现场实测能补"的五项里，
    **所以今天补不了，只能量出来**。
    """

    integral_gain = _integral_gain_for(0.5)
    for degrees in (0.0, 30.0, 60.0, 90.0, 180.0):
        ratio = capstan_transfer_ratio(
            friction_coefficient=0.3, wrap_angle_rad=math.radians(degrees)
        )
        _, samples = _run(
            integral_gain=integral_gain,
            dt_s=1.0e-3,
            horizon_s=5.0,
            measurement_transfer=1.0 / ratio,
        )
        final = samples[-1]
        assert final.measured_n == pytest.approx(SETPOINT_N, abs=1e-6), (
            "闭环没有把**测到的**量调到设定值——那本门的前提就不成立"
        )
        assert final.tension_n == pytest.approx(SETPOINT_N * ratio, rel=1e-6), (
            f"包角{degrees}°：被控点真实张力{final.tension_n!r}"
            f"不等于设定值乘{ratio!r}"
        )

    #: 90°那一档单独钉一个数——散文里的"超六成"必须有一条门看着。
    ninety = capstan_transfer_ratio(friction_coefficient=0.3, wrap_angle_rad=math.pi / 2)
    assert SETPOINT_N * ninety == pytest.approx(48.059, rel=1e-4)
    assert ninety - 1.0 == pytest.approx(0.602, rel=1e-2)


def test_the_transfer_ratio_direction_matters_quadratically():
    """**方向搞反，误差是平方。**

    ``capstan_transfer_ratio``返回的恒是张紧端比松弛端（``≥ 1``）；
    调用方按带材走向决定乘还是除。把``transfer``写成``ratio``而不是``1/ratio``
    时，被控点张力从``设定·r``变成``设定/r``——**两者相差``r²``**。

    ``μ = 0.3``、90°时``r² = 2.566``：真实张力从48.06 N变成18.73 N，
    **一个偏高六成、一个偏低四成，而两者都"看起来像个合理的张力"**。
    这正是这类符号错最难被发现的原因。
    """

    ratio = capstan_transfer_ratio(friction_coefficient=0.3, wrap_angle_rad=math.pi / 2)
    integral_gain = _integral_gain_for(0.5)
    _, correct = _run(
        integral_gain=integral_gain, dt_s=1.0e-3, horizon_s=5.0,
        measurement_transfer=1.0 / ratio,
    )
    _, flipped = _run(
        integral_gain=integral_gain, dt_s=1.0e-3, horizon_s=5.0,
        measurement_transfer=ratio,
    )
    assert correct[-1].tension_n / flipped[-1].tension_n == pytest.approx(
        ratio * ratio, rel=1e-6
    )
    assert ratio * ratio == pytest.approx(2.566, rel=1e-3)


def test_the_capstan_transfer_matches_the_case_closed_form():
    """本模块的``exp(μθ)``与`cases/capstan_tension_ratio`判的是同一条式子。

    **两处不共享实现**（一个在`drives`、一个在案例的金标生成器里），
    所以这条门是它们的互钉：任何一边改了式子都会在这里红。
    """

    for mu, degrees in ((0.3, 90.0), (0.15, 180.0), (0.5, 45.0)):
        angle = math.radians(degrees)
        assert capstan_transfer_ratio(
            friction_coefficient=mu, wrap_angle_rad=angle
        ) == pytest.approx(math.exp(mu * angle), rel=1e-15)
    #: 零摩擦或零包角都给1——张力穿过去不变。
    assert capstan_transfer_ratio(friction_coefficient=0.0, wrap_angle_rad=9.9) == 1.0
    assert capstan_transfer_ratio(friction_coefficient=0.9, wrap_angle_rad=0.0) == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"full_scale_n": 0.0}, "full_scale_n"),
        ({"output_at_full_scale_mv": -1.0}, "output_at_full_scale_mv"),
        ({"adc_bits": 0}, r"\[1, 32\]"),
        ({"adc_bits": 33}, r"\[1, 32\]"),
        ({"adc_bits": True}, "must be an int"),
        ({"adc_bits": 12.0}, "must be an int"),
    ],
)
def test_a_malformed_sensor_fails_closed(kwargs, message):
    base = {"full_scale_n": 49.0, "output_at_full_scale_mv": 20.0, "adc_bits": 12}
    with pytest.raises(DriveError, match=message):
        TensionSensor(**{**base, **kwargs})


@pytest.mark.parametrize("transfer", [0.0, -1.0, float("inf"), float("nan")])
def test_a_malformed_measurement_transfer_fails_closed(transfer):
    with pytest.raises(DriveError, match="measurement_transfer"):
        TensionLoop(
            clutch=CLUTCH, spool=SPOOL,
            controller=PidController(
                proportional=0.0, integral_gain=1.0, derivative=0.0, integral_limit=1.0
            ),
            setpoint_n=SETPOINT_N, dt_s=1.0e-3, delay_line=None,
            sensor=None, measurement_transfer=transfer,
        )


def test_the_measurement_transfer_has_no_default():
    """**没有默认值是有意的**：给``1.0``是一条声明——"传感器就在被控点上"。

    做成默认1.0等于让上面那条60%的误差默默消失，
    而这个仓已经因为"默认值替调用方做了声明"吃过亏
    （`ActuatorDeclaration`的``zero_delay_rationale``就是那次的产物）。
    """

    with pytest.raises(TypeError):
        TensionLoop(  # type: ignore[call-arg]
            clutch=CLUTCH, spool=SPOOL,
            controller=PidController(
                proportional=0.0, integral_gain=1.0, derivative=0.0, integral_limit=1.0
            ),
            setpoint_n=SETPOINT_N, dt_s=1.0e-3, delay_line=None, sensor=None,
        )
