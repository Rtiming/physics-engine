"""`case/norris_thin_strip`的conformance门（轴7规则3）。

**场景①（超导带电流分布）第一个可跑通的案例**，判据是第1档解析闭式：
Norris 1970薄带临界态的片电流分布与两条交流损耗式
（research/12第3.1节的A2，判据来源逐条核实见案例页第二节）。

四层判据各自独立，**任何一层单独绿都不足以说明公式对**：

1. 片电流分布对50位十进制参考（另一套算术，另一种代数写法）；
2. 电流守恒`∫K dx = I`——**唯一抓得住"b公式写错"的那一层**；
3. 磁通前沿的两个零容差极限（`I=0 ⟹ b=w/2`、`I=Ic ⟹ b=0`）；
4. 结构判据：偶对称、`b`随`I`单调减、`K`随`|x|`单调增、`0 < K ≤ Kc`。

判据数全部来自清单；本文件不复述任何公式（轴7规则4）。求积是**测量手段**
不是oracle：它调生产内核算出一个数，与清单里冻结的`I`比。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.electromagnetics.superconductor import (
    SuperconductorError,
    flux_free_half_width_m,
    norris_ellipse_loss_j_per_m_per_cycle,
    norris_ellipse_normalised_loss,
    norris_strip_loss_j_per_m_per_cycle,
    norris_strip_normalised_loss,
    sheet_current_density_a_per_m,
)
from physics_engine.oracles import OracleError, load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/norris_thin_strip/oracle.json", root=ROOT)
PROFILE = MANIFEST.oracle("oracle:norris/sheet_current_profile")
CONSERVATION = MANIFEST.oracle("oracle:norris/current_conservation")
FRONT = MANIFEST.oracle("oracle:norris/flux_front")
LOSS = MANIFEST.oracle("oracle:norris/ac_loss")


# ---------------------------------------------------------------------------
# 测量手段：两条求积。都只调生产内核，不复述闭式。
# ---------------------------------------------------------------------------
def _simpson(function, low: float, high: float, intervals: int) -> float:
    """复合Simpson。`intervals`必须是偶数。"""

    step = (high - low) / intervals
    total = function(low) + function(high)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * function(low + index * step)
    return total * step / 3.0


def integrate_by_arcsine_substitution(
    *, width_m: float, sheet_a_per_m: float, current_a: float, critical_a: float,
    intervals: int, sheet_function=sheet_current_density_a_per_m,
    front_function=flux_free_half_width_m,
) -> float:
    """``∫K dx``，未穿透区做``x = b·sin t``代换。

    代换把`|x| → b`处的竖直斜率吃掉：被积函数变成`K(b·sin t)·b·cos t`，
    在`t ∈ [0, π/2]`上解析，Simpson恢复四阶收敛。**代换只换积分变量，
    不碰被积函数本身**——`K`仍旧是生产内核算的。
    """

    half_width = 0.5 * width_m
    front = front_function(
        width_m=width_m, transport_current_a=current_a, critical_current_a=critical_a
    )
    saturated = 2.0 * sheet_a_per_m * (half_width - front)
    if front <= 0.0:
        return saturated

    def integrand(angle: float) -> float:
        position = front * math.sin(angle)
        density = sheet_function(
            position_m=position,
            width_m=width_m,
            transport_current_a=current_a,
            critical_sheet_current_a_per_m=sheet_a_per_m,
        )
        return density * front * math.cos(angle)

    return saturated + 2.0 * _simpson(integrand, 0.0, 0.5 * math.pi, intervals)


def integrate_directly(
    *, width_m: float, sheet_a_per_m: float, current_a: float, intervals: int
) -> float:
    """``∫K dx``，直接在`x`上做复合Simpson——**故意不做代换的那一条**。"""

    half_width = 0.5 * width_m

    def integrand(position: float) -> float:
        return sheet_current_density_a_per_m(
            position_m=position,
            width_m=width_m,
            transport_current_a=current_a,
            critical_sheet_current_a_per_m=sheet_a_per_m,
        )

    return _simpson(integrand, -half_width, half_width, intervals)


# ---------------------------------------------------------------------------
# 第一层：片电流分布
# ---------------------------------------------------------------------------
def test_the_sheet_current_profile_matches_a_fifty_digit_reference():
    """float64闭式 对 50位十进制参考——两套算术，且代数写法也不同。

    生产内核把arctan自变量写成`(a·i)/sqrt((b−|x|)(b+|x|))`（无相消），
    参考按原式`sqrt((a²−b²)/(b²−x²))`在50位下算。两边连形式都不一样。
    """

    positions = PROFILE.inputs["positions_m"]
    ratios = PROFILE.inputs["current_ratios"]
    fractions = PROFILE.inputs["position_fractions"]
    width = PROFILE.inputs["width_m"]
    sheet = PROFILE.inputs["critical_sheet_current_a_per_m"]
    critical = PROFILE.inputs["critical_current_a"]

    currents = [ratio * critical for ratio in ratios for _ in fractions]
    measured = [
        sheet_current_density_a_per_m(
            position_m=position,
            width_m=width,
            transport_current_a=current,
            critical_sheet_current_a_per_m=sheet,
        )
        for position, current in zip(positions, currents, strict=True)
    ]
    saturated_currents = [ratio * critical for ratio in ratios for _ in range(2)]
    saturated = [
        sheet_current_density_a_per_m(
            position_m=position,
            width_m=width,
            transport_current_a=current,
            critical_sheet_current_a_per_m=sheet,
        )
        for position, current in zip(
            PROFILE.inputs["saturated_positions_m"], saturated_currents, strict=True
        )
    ]
    PROFILE.check_all({
        "sheet_current_values_a_per_m": measured,
        "saturated_branch_values_a_per_m": saturated,
    })


# ---------------------------------------------------------------------------
# 第二层：电流守恒
# ---------------------------------------------------------------------------
def test_the_current_integral_returns_the_transport_current():
    """`∫K dx = I`。两条求积同时跑：代换式到1e-9，直接式只到1e-6。

    两条并存不是冗余：它们冻的是**同一个恒等式在两种求积下的不同病态**，
    差三个数量级这件事本身就是判据表里那两条理由的证据。
    """

    width = CONSERVATION.inputs["width_m"]
    sheet = CONSERVATION.inputs["critical_sheet_current_a_per_m"]
    critical = CONSERVATION.inputs["critical_current_a"]

    arcsine = [
        integrate_by_arcsine_substitution(
            width_m=width, sheet_a_per_m=sheet, current_a=ratio * critical,
            critical_a=critical, intervals=CONSERVATION.inputs["arcsine_simpson_nodes"],
        )
        for ratio in CONSERVATION.inputs["current_ratios"]
    ]
    direct = [
        integrate_directly(
            width_m=width, sheet_a_per_m=sheet, current_a=ratio * critical,
            intervals=CONSERVATION.inputs["direct_simpson_nodes"],
        )
        for ratio in CONSERVATION.inputs["direct_current_ratios"]
    ]
    CONSERVATION.check_all({
        "arcsine_quadrature_currents_a": arcsine,
        "direct_quadrature_currents_a": direct,
    })


# ---------------------------------------------------------------------------
# 第三层：磁通前沿与它的两个零容差极限
# ---------------------------------------------------------------------------
def test_the_flux_front_and_its_two_zero_tolerance_limits():
    """`b`的取值表，以及`I=0 ⟹ b=w/2`、`I=Ic ⟹ b=0`两个零容差极限。"""

    width = FRONT.inputs["width_m"]
    critical = FRONT.inputs["critical_current_a"]
    FRONT.check_all({
        "flux_front_positions_m": [
            flux_free_half_width_m(
                width_m=width, transport_current_a=ratio * critical,
                critical_current_a=critical,
            )
            for ratio in FRONT.inputs["current_ratios"]
        ],
        "front_at_zero_current_m": flux_free_half_width_m(
            width_m=width, transport_current_a=0.0, critical_current_a=critical
        ),
        "front_at_critical_current_m": flux_free_half_width_m(
            width_m=width, transport_current_a=critical, critical_current_a=critical
        ),
    })


# ---------------------------------------------------------------------------
# 第二层之二：两条交流损耗式与它们的幂次
# ---------------------------------------------------------------------------
def test_the_two_ac_loss_laws_and_their_power_law_asymptotics():
    """薄带`→ i⁴/3`、椭圆`→ i³/3`——**三次方与四次方的差别是这两条式的招牌**。"""

    ratios = LOSS.inputs["current_ratios"]
    asymptotic = LOSS.inputs["asymptotic_ratio"]
    low, high = LOSS.inputs["decade_ratios"]
    critical = LOSS.inputs["critical_current_a"]
    amplitude = LOSS.inputs["loss_amplitude_a"]

    def slope(function) -> float:
        return math.log(function(high) / function(low)) / math.log(high / low)

    LOSS.check_all({
        "strip_normalised_losses": [norris_strip_normalised_loss(r) for r in ratios],
        "ellipse_normalised_losses": [norris_ellipse_normalised_loss(r) for r in ratios],
        "strip_power_law_ratio":
            norris_strip_normalised_loss(asymptotic) / (asymptotic**4 / 3.0),
        "ellipse_power_law_ratio":
            norris_ellipse_normalised_loss(asymptotic) / (asymptotic**3 / 3.0),
        "strip_decade_slope": slope(norris_strip_normalised_loss),
        "ellipse_decade_slope": slope(norris_ellipse_normalised_loss),
        "strip_loss_j_per_m_per_cycle": norris_strip_loss_j_per_m_per_cycle(
            critical_current_a=critical, current_amplitude_a=amplitude
        ),
        "ellipse_loss_j_per_m_per_cycle": norris_ellipse_loss_j_per_m_per_cycle(
            critical_current_a=critical, current_amplitude_a=amplitude
        ),
    })


# ---------------------------------------------------------------------------
# 第四层：结构判据（不靠清单）
# ---------------------------------------------------------------------------
WIDTH_M = 4.0e-3
SHEET_A_PER_M = 3.0e4
CRITICAL_A = WIDTH_M * SHEET_A_PER_M


def structural_violations(sheet_function, front_function) -> list[str]:
    """结构判据的纯函数体：偶对称、`b`单调减、`K`单调增、`0 < K ≤ Kc`。

    抽成函数是为了让"必须红"的错法实现走**同一段判据**——
    判据本身也要被验（轴6规则6）。
    """

    problems: list[str] = []
    half_width = 0.5 * WIDTH_M

    def density(position: float, current: float) -> float:
        return sheet_function(
            position_m=position, width_m=WIDTH_M, transport_current_a=current,
            critical_sheet_current_a_per_m=SHEET_A_PER_M,
        )

    previous_front = None
    for index in range(0, 101):
        current = CRITICAL_A * index / 100.0
        front = front_function(
            width_m=WIDTH_M, transport_current_a=current, critical_current_a=CRITICAL_A
        )
        if previous_front is not None and not front < previous_front:
            problems.append(f"b在I={current!r} A处不再严格随I下降：{previous_front!r}→{front!r}")
        previous_front = front

    for ratio in (0.05, 0.2, 0.5, 0.8, 0.99):
        current = CRITICAL_A * ratio
        front = front_function(
            width_m=WIDTH_M, transport_current_a=current, critical_current_a=CRITICAL_A
        )
        previous = None
        for step in range(0, 65):
            position = half_width * step / 64.0
            value = density(position, current)
            if density(-position, current) != value:
                problems.append(f"K在x={position!r} m处不是偶函数")
            if not 0.0 < value <= SHEET_A_PER_M:
                problems.append(f"K在x={position!r} m处落在(0, Kc]之外：{value!r}")
            if previous is not None and value < previous:
                problems.append(f"K在x={position!r} m处随|x|下降：{previous!r}→{value!r}")
            previous = value
        # 磁通前沿处闭式连续：内支的极限恰为Kc。
        inner = density(front * (1.0 - 1.0e-12), current)
        if not 0.0 < SHEET_A_PER_M - inner < 1.0e-4 * SHEET_A_PER_M:
            problems.append(f"K在i={ratio}的磁通前沿内侧不连续：Kc−K={SHEET_A_PER_M - inner!r}")
    return problems


def test_the_profile_is_even_and_monotone_and_the_front_recedes():
    """四条结构判据，一条不靠清单——它们抓的是清单抓不到的那一类错。

    对照airy案例那条注释：一个把`J1`写成`J0`的实现在首零判据上会红，
    但一个把自变量取绝对值**之后又平方**的实现照样绿。这里同理：
    把`2/π`写成`π/2`的实现在`0 < K ≤ Kc`这条上红，而它并不需要任何金标。
    """

    assert not structural_violations(
        sheet_current_density_a_per_m, flux_free_half_width_m
    )


def test_the_strip_loses_less_than_the_ellipse_at_every_amplitude():
    """薄带`~i⁴`比椭圆`~i³`低一阶，所以全区间上薄带损耗更小且比值随`i`增长。

    这一条也不靠清单：它验的是**两条式的相对关系**，
    而两个各自算错但错法相同的实现在逐点金标上会一起红、在这里未必。
    """

    previous = None
    for index in range(1, 101):
        ratio = index / 100.0
        strip = norris_strip_normalised_loss(ratio)
        ellipse = norris_ellipse_normalised_loss(ratio)
        assert 0.0 < strip < ellipse, f"i={ratio}处薄带损耗没有低于椭圆截面"
        quotient = strip / ellipse
        if previous is not None:
            assert quotient > previous, f"i={ratio}处薄带/椭圆之比没有随i增长"
        previous = quotient


def test_the_closed_form_fails_closed_outside_its_domain():
    """域外失败关闭：超临界电流、带外横坐标、负电流、非正宽度。"""

    with pytest.raises(SuperconductorError):
        flux_free_half_width_m(
            width_m=WIDTH_M, transport_current_a=CRITICAL_A * 1.000001,
            critical_current_a=CRITICAL_A,
        )
    with pytest.raises(SuperconductorError):
        sheet_current_density_a_per_m(
            position_m=0.5 * WIDTH_M * 1.000001, width_m=WIDTH_M,
            transport_current_a=0.5 * CRITICAL_A,
            critical_sheet_current_a_per_m=SHEET_A_PER_M,
        )
    with pytest.raises(SuperconductorError):
        flux_free_half_width_m(
            width_m=WIDTH_M, transport_current_a=-1.0, critical_current_a=CRITICAL_A
        )
    with pytest.raises(SuperconductorError):
        flux_free_half_width_m(
            width_m=0.0, transport_current_a=0.0, critical_current_a=CRITICAL_A
        )
    with pytest.raises(SuperconductorError):
        norris_strip_normalised_loss(1.0000001)
    with pytest.raises(SuperconductorError):
        norris_ellipse_normalised_loss(-1.0e-15)


# ---------------------------------------------------------------------------
# 必须红：五种典型错法 × 四条门的矩阵
# ---------------------------------------------------------------------------
#: 五种错法。前四种是任务书点名的典型错，第五种（`b`里漏掉平方）是与第一种
#: 同族的另一个典型笔误——留着它是因为它复现了同一条发现：
#: **两个零容差极限对这一族错法一条都抓不住**。
WRONG_KINDS: tuple[str, ...] = (
    "front_without_sqrt",     # b = (w/2)(1 − i²)，漏掉sqrt
    "ratio_inverted",         # arctan里分子分母颠倒
    "two_over_pi_flipped",    # 2/π 写成 π/2
    "branches_swapped",       # |x|<b 与 |x|≥b 两支条件写反
    "front_without_square",   # b = (w/2)sqrt(1 − i)，漏掉平方
)


def wrong_front(kind: str, *, width_m, transport_current_a, critical_current_a) -> float:
    ratio = transport_current_a / critical_current_a
    half_width = 0.5 * width_m
    if kind == "front_without_sqrt":
        return half_width * (1.0 - ratio * ratio)
    if kind == "front_without_square":
        return half_width * math.sqrt(1.0 - ratio)
    return half_width * math.sqrt((1.0 - ratio) * (1.0 + ratio))


def wrong_sheet(
    kind: str, *, position_m, width_m, transport_current_a, critical_sheet_current_a_per_m
) -> float:
    half_width = 0.5 * width_m
    critical = width_m * critical_sheet_current_a_per_m
    front = wrong_front(
        kind, width_m=width_m, transport_current_a=transport_current_a,
        critical_current_a=critical,
    )
    distance = abs(position_m)
    inside = distance < front
    if kind == "branches_swapped":
        inside = not inside
    if not inside:
        return critical_sheet_current_a_per_m
    numerator = (half_width - front) * (half_width + front)
    denominator = (front - distance) * (front + distance)
    if kind == "ratio_inverted":
        numerator, denominator = denominator, numerator
    factor = (
        math.pi * critical_sheet_current_a_per_m / 2.0
        if kind == "two_over_pi_flipped"
        else 2.0 * critical_sheet_current_a_per_m / math.pi
    )
    return factor * math.atan(math.sqrt(numerator / denominator))


def _is_red(action) -> bool:
    """门红了没有。清单不过抛`OracleError`；错法把闭式算炸抛`ValueError`等。"""

    try:
        action()
    except (OracleError, ValueError, ZeroDivisionError, OverflowError):
        return True
    return False


def _profile_gate(kind: str) -> None:
    ratios = PROFILE.inputs["current_ratios"]
    fractions = PROFILE.inputs["position_fractions"]
    critical = PROFILE.inputs["critical_current_a"]
    currents = [ratio * critical for ratio in ratios for _ in fractions]
    PROFILE.check(
        "sheet_current_values_a_per_m",
        [
            wrong_sheet(
                kind, position_m=position, width_m=PROFILE.inputs["width_m"],
                transport_current_a=current,
                critical_sheet_current_a_per_m=(
                    PROFILE.inputs["critical_sheet_current_a_per_m"]
                ),
            )
            for position, current in zip(
                PROFILE.inputs["positions_m"], currents, strict=True
            )
        ],
    )


def _conservation_gate(kind: str) -> None:
    critical = CONSERVATION.inputs["critical_current_a"]
    CONSERVATION.check(
        "arcsine_quadrature_currents_a",
        [
            integrate_by_arcsine_substitution(
                width_m=CONSERVATION.inputs["width_m"],
                sheet_a_per_m=CONSERVATION.inputs["critical_sheet_current_a_per_m"],
                current_a=ratio * critical, critical_a=critical,
                intervals=CONSERVATION.inputs["arcsine_simpson_nodes"],
                sheet_function=lambda **kwargs: wrong_sheet(kind, **kwargs),
                front_function=lambda **kwargs: wrong_front(kind, **kwargs),
            )
            for ratio in CONSERVATION.inputs["current_ratios"]
        ],
    )


def _front_table_gate(kind: str) -> None:
    critical = FRONT.inputs["critical_current_a"]
    FRONT.check(
        "flux_front_positions_m",
        [
            wrong_front(
                kind, width_m=FRONT.inputs["width_m"],
                transport_current_a=ratio * critical, critical_current_a=critical,
            )
            for ratio in FRONT.inputs["current_ratios"]
        ],
    )


def _front_limits_gate(kind: str) -> None:
    critical = FRONT.inputs["critical_current_a"]
    for quantity, current in (
        ("front_at_zero_current_m", 0.0),
        ("front_at_critical_current_m", critical),
    ):
        FRONT.check(
            quantity,
            wrong_front(
                kind, width_m=FRONT.inputs["width_m"], transport_current_a=current,
                critical_current_a=critical,
            ),
        )


def _structural_gate(kind: str) -> None:
    problems = structural_violations(
        lambda **kwargs: wrong_sheet(kind, **kwargs),
        lambda **kwargs: wrong_front(kind, **kwargs),
    )
    if problems:
        raise ValueError("；".join(problems[:3]))


#: 实测出来的矩阵（决策0047第三节逐条记账）。`True`=该门抓住了这条错法。
#: **`front_without_sqrt`与`front_without_square`那两行的`limits=False`是本轮
#: 最贵的一条发现**：两个零容差极限对这一族错法一条都抓不住——
#: 它们在`i=0`与`i=1`上恰好给出与正确公式相同的值。
RED_MATRIX: dict[str, dict[str, bool]] = {
    "front_without_sqrt":   {"profile": True, "conservation": True,
                             "front_table": True, "front_limits": False, "structural": False},
    "ratio_inverted":       {"profile": True, "conservation": True,
                             "front_table": False, "front_limits": False, "structural": True},
    "two_over_pi_flipped":  {"profile": True, "conservation": True,
                             "front_table": False, "front_limits": False, "structural": True},
    "branches_swapped":     {"profile": True, "conservation": True,
                             "front_table": False, "front_limits": False, "structural": True},
    "front_without_square": {"profile": True, "conservation": True,
                             "front_table": True, "front_limits": False, "structural": False},
}

GATES = {
    "profile": _profile_gate,
    "conservation": _conservation_gate,
    "front_table": _front_table_gate,
    "front_limits": _front_limits_gate,
    "structural": _structural_gate,
}


@pytest.mark.parametrize("kind", WRONG_KINDS)
def test_must_be_red_every_wrong_form_is_caught_by_the_declared_gates(kind: str):
    """五种典型错法 × 五条门的矩阵**逐格实测**。

    这条测试不只断言"至少有一条门红"——那太弱，它会让"哪条门抓住了"
    这件事无从审计。它断言的是**整张矩阵逐格与`RED_MATRIX`一致**：
    某条门变得更强或更弱都会让这里红，逼人回来更新决策0047第三节的记账。
    """

    expected = RED_MATRIX[kind]
    measured = {name: _is_red(lambda gate=gate: gate(kind)) for name, gate in GATES.items()}
    assert measured == expected, (
        f"错法{kind!r}的必红矩阵与记账不符：实测{measured}，记账{expected}"
    )
    assert any(measured.values()), f"错法{kind!r}没有被任何一条门抓住"


def test_the_red_matrix_is_not_vacuous():
    """判据本身要被验（轴6规则6）：正确实现在**每一条门上都必须绿**。

    没有这一条，上面那张矩阵可以靠"所有门永远红"来假通过。
    """

    for name, gate in GATES.items():
        assert not _is_red(lambda gate=gate: gate("correct")), (
            f"门{name!r}对正确实现也红——它测的不是错法，是它自己坏了"
        )
