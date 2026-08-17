"""各向异性摩擦椭圆（η-return）的单元门——决策0068。

案例`anisotropic_friction_ellipse`判的是**物理量**（耗散、回线、外法向）；
本文件判的是**形制**：转交是否逐位、朝向是不是显式的、误用会不会静默。

每条门都附了它防的那个具体错法。**"这条门要是删了什么都不会红"是本文件的反面**。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.contact import (
    IN_PLANE_DIRECTION_MIN_SINE,
    REGIME_SEPARATED,
    REGIME_SLIP,
    REGIME_STICK,
    ContactError,
    FrictionEllipse,
    FrictionOutcome,
    anisotropic_return_map,
    coulomb_return_map,
)
from physics_engine.contact import friction as friction_module

NORMAL = (0.0, 0.0, 1.0)
ALONG = (1.0, 0.0, 0.0)
NORMAL_FORCE_N = 4.0
STIFFNESS = 3.0e4


def _ellipse(mu_along: float, mu_across: float, **overrides) -> FrictionEllipse:
    return FrictionEllipse(
        mu_along=mu_along,
        mu_across=mu_across,
        along_direction=overrides.get("along_direction", ALONG),
        normal=overrides.get("normal", NORMAL),
    )


def _map(trial, ellipse, normal_force=NORMAL_FORCE_N) -> FrictionOutcome:
    return anisotropic_return_map(
        trial_force_n=trial,
        normal_force_n=normal_force,
        ellipse=ellipse,
        tangential_stiffness_n_per_mm=STIFFNESS,
    )


def _bits(outcome: FrictionOutcome) -> tuple[str, ...]:
    return (
        *(value.hex() for value in outcome.tangential_force_n),
        outcome.regime.hex(),
        *(value.hex() for value in outcome.anchor_correction_mm),
    )


def test_the_degenerate_path_hands_the_error_behaviour_over_too():
    """``μ_∥ == μ_⊥``时**连报错都必须是**`coulomb_return_map`**的**。

    防的是这个错法：先自己校验一遍再转交。那样"逐位相同"就只覆盖了正常路径，
    而异常路径（nan试探力、负法向力）悄悄换了报错文本甚至换了异常类型——
    **调用方按报错文本分支的代码会静默走错**。
    """

    ellipse = _ellipse(0.3, 0.3)
    bad_inputs = (
        ((float("nan"), 0.0, 0.0), NORMAL_FORCE_N),
        ((1.0, 0.0), NORMAL_FORCE_N),
        ((1.0, 0.0, 0.0), -1.0),
        ((1.0, 0.0, 0.0), float("inf")),
    )
    for trial, normal_force in bad_inputs:
        with pytest.raises(ContactError) as elliptic:
            _map(trial, ellipse, normal_force)
        with pytest.raises(ContactError) as circular:
            coulomb_return_map(
                trial_force_n=trial,
                normal_force_n=normal_force,
                friction_coefficient=0.3,
                tangential_stiffness_n_per_mm=STIFFNESS,
            )
        assert str(elliptic.value) == str(circular.value), (
            f"退化路径的报错与各向同性不一致：{str(elliptic.value)!r} "
            f"vs {str(circular.value)!r}"
        )


def test_the_ellipse_does_not_care_about_the_sign_of_along_direction():
    """``d_∥ → −d_∥``必须**逐位**不变——椭圆是中心对称的。

    这条门防的是一整类静默朝向错：若哪天有人在面内轴上引入了符号相关的分支
    （比如"横向落位偏差取正负"），它会当场红。
    **`PenaltyAnnulusLimit`把朝向编码在坐标符号里那次，缺的就是这条门。**
    """

    forward = _ellipse(0.5, 0.1, along_direction=(0.6, 0.8, 0.0))
    backward = _ellipse(0.5, 0.1, along_direction=(-0.6, -0.8, 0.0))
    for index in range(72):
        angle = 2.0 * math.pi * index / 72.0
        trial = (7.0 * math.cos(angle), 7.0 * math.sin(angle), 0.0)
        assert _bits(_map(trial, forward)) == _bits(_map(trial, backward)), (
            f"翻转纵向轴改变了结果（角{index * 5}°）——朝向被当成了有符号的量"
        )


def test_rotating_the_whole_problem_rotates_the_answer():
    """整体转一个刚体转动，答案必须跟着转。

    防的是"朝向其实是从坐标轴推出来的"这一类：若实现里偷偷假设了
    ``e_∥ = x轴``，本门在任何非平凡转动下当场红。
    """

    def rotate(vector, angle, axis_angle):
        """先绕z转``angle``，再绕x转``axis_angle``。"""

        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x = vector[0] * cos_a - vector[1] * sin_a
        y = vector[0] * sin_a + vector[1] * cos_a
        z = vector[2]
        cos_b, sin_b = math.cos(axis_angle), math.sin(axis_angle)
        return (x, y * cos_b - z * sin_b, y * sin_b + z * cos_b)

    base = _ellipse(0.5, 0.1)
    angle, tilt = 0.7, 0.4
    turned = FrictionEllipse(
        mu_along=0.5,
        mu_across=0.1,
        along_direction=rotate(ALONG, angle, tilt),
        normal=rotate(NORMAL, angle, tilt),
    )
    worst = 0.0
    for index in range(36):
        theta = 2.0 * math.pi * index / 36.0
        trial = (5.0 * math.cos(theta), 5.0 * math.sin(theta), 0.0)
        plain = _map(trial, base)
        spun = _map(rotate(trial, angle, tilt), turned)
        expected = rotate(plain.tangential_force_n, angle, tilt)
        worst = max(
            worst,
            max(abs(spun.tangential_force_n[axis] - expected[axis]) for axis in range(3)),
        )
        assert spun.regime == plain.regime
    assert worst < 1.0e-14, f"转动不变性最大偏差{worst}——朝向被坐标轴污染了"


def test_swapping_the_two_coefficients_changes_the_answer():
    """交换``μ_∥``与``μ_⊥``必须给出**不同**的答案。

    **这是一条反向门**：上面三条都在验"某个变换不改变结果"，
    而一个把``along_direction``整个忽略掉的实现会把它们全部通过。
    只有这一条能把"朝向真的被用上了"验出来。
    """

    trial = (3.0, 3.0, 0.0)
    straight = _map(trial, _ellipse(0.5, 0.1))
    swapped = _map(trial, _ellipse(0.1, 0.5))
    assert straight.tangential_force_n != swapped.tangential_force_n
    #: 而且交换系数等于把问题镜像到另一条主轴上：分量应当互换。
    assert straight.tangential_force_n[0] == pytest.approx(
        swapped.tangential_force_n[1], rel=1e-15
    )
    assert straight.tangential_force_n[1] == pytest.approx(
        swapped.tangential_force_n[0], rel=1e-15
    )


def test_the_maximum_dissipation_inequality_holds_against_a_sampled_yield_surface():
    """Hill不等式的**原样**：``f·Δu ≥ f'·Δu``对屈服面上任意``f'``成立。

    上一层案例判的是"等于支撑函数"，那用了凸分析的闭式。
    这一条**不用任何闭式**，直接在椭圆上采1024个点比大小——
    两条路径若同时被同一个错误骗过，那个错误得同时骗过闭式与采样。
    """

    mu_along, mu_across = 0.5, 0.1
    ellipse = _ellipse(mu_along, mu_across)
    semi_along = mu_along * NORMAL_FORCE_N
    semi_across = mu_across * NORMAL_FORCE_N
    samples = tuple(
        (
            semi_along * math.cos(2.0 * math.pi * k / 1024.0),
            semi_across * math.sin(2.0 * math.pi * k / 1024.0),
        )
        for k in range(1024)
    )
    checked = 0
    for index in range(1, 180):
        theta = math.radians(index * 2.0)
        trial = (
            2.5 * semi_along * math.cos(theta),
            2.5 * semi_across * math.sin(theta),
            0.0,
        )
        outcome = _map(trial, ellipse)
        assert outcome.regime == REGIME_SLIP
        slip = outcome.anchor_correction_mm
        force = outcome.tangential_force_n
        actual = force[0] * slip[0] + force[1] * slip[1]
        assert actual > 0.0, "耗散必须为正——滑移在给摩擦力做负功"
        best = max(point[0] * slip[0] + point[1] * slip[1] for point in samples)
        assert actual >= best - 1.0e-15 * abs(actual), (
            f"屈服面上存在耗散更大的力（差{best - actual}）——最大耗散原理不成立"
        )
        checked += 1
    assert checked == 179, f"只判了{checked}个方向"


def test_the_returned_point_satisfies_the_return_map_definition():
    """挪完锚点之后**重算的试探力恰好等于返回的力**——return-map的定义性质。

    与`coulomb_return_map`的自洽门同一条口径。它防的是
    "锚点修正与力的投影各算各的"：那时力在椭圆上而锚点挪多了或挪少了，
    **下一步会凭空多出或少掉一截摩擦力**，而单看某一步的力完全正常。
    """

    ellipse = _ellipse(0.5, 0.1)
    semi_along, semi_across = 0.5 * NORMAL_FORCE_N, 0.1 * NORMAL_FORCE_N
    worst = 0.0
    for index in range(120):
        theta = 2.0 * math.pi * index / 120.0
        for factor in (1.01, 1.5, 40.0):
            trial = (
                factor * semi_along * math.cos(theta),
                factor * semi_across * math.sin(theta),
                0.0,
            )
            outcome = _map(trial, ellipse)
            recomputed = tuple(
                trial[axis] - STIFFNESS * outcome.anchor_correction_mm[axis]
                for axis in range(3)
            )
            worst = max(
                worst,
                max(
                    abs(recomputed[axis] - outcome.tangential_force_n[axis])
                    for axis in range(3)
                ),
            )
    assert worst < 1.0e-14, f"重算的试探力与返回的力差{worst} N——两者不是同一次投影"


def test_the_yield_surface_boundary_counts_as_stick():
    """恰落在椭圆上判**粘**，与``|T| ≤ μN``同口径。边界约定变了这里当场红。"""

    ellipse = _ellipse(0.5, 0.25)
    #: ``μ_⊥N = 1.0``，二进制可精确表示；构造精确落在椭圆上的点。
    on_surface = (0.0, 0.25 * NORMAL_FORCE_N, 0.0)
    outcome = _map(on_surface, ellipse)
    assert outcome.regime == REGIME_STICK, "恰在椭圆上被判成滑——边界约定变了"
    assert outcome.anchor_correction_mm == (0.0, 0.0, 0.0), "边界上不许挪锚点"
    assert outcome.tangential_force_n == pytest.approx(on_surface, abs=0.0, rel=1e-16)


def test_zero_normal_force_gives_separated_not_stick():
    """没有法向力就没有摩擦，而且"分离"与"在滑"是两件事——与各向同性同口径。"""

    outcome = _map((5.0, 3.0, 0.0), _ellipse(0.5, 0.1), normal_force=0.0)
    assert outcome.regime == REGIME_SEPARATED
    assert outcome.tangential_force_n == (0.0, 0.0, 0.0)
    assert outcome.anchor_correction_mm == (0.0, 0.0, 0.0)


def test_a_trial_force_with_a_normal_component_is_refused():
    """带法向分量的"切向试探力"要报错，不许静默扔掉。

    静默扔掉的后果是**法向力被切向映射悄悄改小**而没有任何地方报告它——
    与`TangentialStickSpring`漏掉法向投影时"``normal_force_n``报的仍是``k_n·δ``"
    是同一种病。
    """

    ellipse = _ellipse(0.5, 0.1)
    #: 舍入量级的法向分量**必须放过**（`TangentialStickSpring`的投影残余）。
    tolerated = _map((1.0, 0.0, 1.0e-12), ellipse)
    assert tolerated.regime in (REGIME_STICK, REGIME_SLIP)
    with pytest.raises(ContactError, match="法向分量"):
        _map((1.0, 0.0, 0.5), ellipse)


def test_the_ellipse_refuses_the_inputs_that_would_go_silently_wrong():
    """构造期六条校验，每条各一个用例。**它们全都是静默错值的入口。**"""

    with pytest.raises(ContactError, match="mu_along"):
        _ellipse(0.0, 0.1)
    with pytest.raises(ContactError, match="mu_across"):
        _ellipse(0.5, -0.1)
    with pytest.raises(ContactError, match="mu_across"):
        _ellipse(0.5, float("nan"))
    with pytest.raises(ContactError, match="along_direction"):
        _ellipse(0.5, 0.1, along_direction=(2.0, 0.0, 0.0))
    with pytest.raises(ContactError, match="normal"):
        _ellipse(0.5, 0.1, normal=(float("nan"), 0.0, 0.0))
    with pytest.raises(ContactError, match="夹角太小"):
        _ellipse(0.5, 0.1, along_direction=NORMAL)


def test_a_barely_off_normal_along_direction_is_refused_but_a_tilted_one_is_projected():
    """近乎平行于法向的纵向轴要拒收，**倾斜但可辨的要投影进切平面**。

    这一条把那个门槛真的取到两侧：``sin``略低于门槛拒收、略高于门槛接受。
    没有它，`IN_PLANE_DIRECTION_MIN_SINE`就只是一个没人验过的常数。
    """

    def tilted(sine):
        return (sine, 0.0, math.sqrt(1.0 - sine * sine))

    with pytest.raises(ContactError, match="夹角太小"):
        _ellipse(0.5, 0.1, along_direction=tilted(0.5 * IN_PLANE_DIRECTION_MIN_SINE))
    accepted = _ellipse(0.5, 0.1, along_direction=tilted(2.0 * IN_PLANE_DIRECTION_MIN_SINE))
    axis_along, axis_across = accepted.in_plane_axes()
    assert axis_along == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-9)
    assert axis_across == pytest.approx((0.0, 1.0, 0.0), abs=1.0e-9)

    #: 一般的倾斜：45°倾角的纵向轴投影后仍是``x``。
    slanted = _ellipse(
        0.5, 0.1, along_direction=(math.sqrt(0.5), 0.0, math.sqrt(0.5))
    )
    assert slanted.in_plane_axes()[0] == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-15)


def test_the_in_plane_axes_are_orthonormal_and_right_handed():
    """``(e_∥, e_⊥, n)``必须是右手正交标架。符号约定被写下来了就要被判。"""

    ellipse = _ellipse(0.5, 0.1, along_direction=(0.6, 0.8, 0.0))
    axis_along, axis_across = ellipse.in_plane_axes()
    normal = ellipse.normal
    for vector in (axis_along, axis_across):
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-15)
        assert sum(vector[i] * normal[i] for i in range(3)) == pytest.approx(0.0, abs=1e-15)
    assert sum(axis_along[i] * axis_across[i] for i in range(3)) == pytest.approx(
        0.0, abs=1e-15
    )
    cross = (
        axis_along[1] * axis_across[2] - axis_along[2] * axis_across[1],
        axis_along[2] * axis_across[0] - axis_along[0] * axis_across[2],
        axis_along[0] * axis_across[1] - axis_along[1] * axis_across[0],
    )
    assert cross == pytest.approx(normal, abs=1e-15), "e_∥ × e_⊥ 不是 n——不是右手系"


def test_the_plastic_multiplier_trip_wire_actually_raises(monkeypatch):
    """趟数上限是**绊线不是预算**：走满必须抛，不许返回一个没收敛的η。

    把上限压到1趟制造一次"走满"。**没有这条门，那个`raise`分支
    在全仓测试里一次也不会被执行**——而一个从没跑过的错误路径等于没有。
    """

    monkeypatch.setattr(friction_module, "_PLASTIC_MAX_ITERATIONS", 1)
    with pytest.raises(ContactError, match="塑性乘子迭代走满"):
        _map((9.0, 0.9, 0.0), _ellipse(0.5, 0.1))


def test_the_returned_force_lands_on_the_ellipse():
    """返回点必须落在屈服面上（``Φ = 1``），残差量级要被断言而不是被相信。"""

    mu_along, mu_across = 0.5, 0.1
    ellipse = _ellipse(mu_along, mu_across)
    semi_along = mu_along * NORMAL_FORCE_N
    semi_across = mu_across * NORMAL_FORCE_N
    worst = 0.0
    for index in range(180):
        theta = 2.0 * math.pi * index / 180.0
        for factor in (1.001, 1.2, 8.0, 5.0e5):
            trial = (
                factor * semi_along * math.cos(theta),
                factor * semi_across * math.sin(theta),
                0.0,
            )
            force = _map(trial, ellipse).tangential_force_n
            quadratic = (force[0] / semi_along) ** 2 + (force[1] / semi_across) ** 2
            worst = max(worst, abs(quadratic - 1.0))
    assert worst < 1.0e-14, f"返回点离屈服面{worst}——标量方程没解到位"
