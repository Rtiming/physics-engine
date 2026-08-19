"""`winding`的门——**匝→层→半径→长度**这条律的四条判据（决策0093）。

本文件守的是**模块自洽与两处旧式子的关系**；对**外部金标**（真阿基米德螺线的
弧长闭式）的对拍与收敛阶在`cases/spool_winding_growth`，
因为金标与被验量不该出自同一支笔。

四条判据：

1. **与`drives.SpoolTension.radius_mm`逐位相同**（退化档：每层1匝、堆积因子1、
   连续式）。判``==``不是判容差——"旧式子是新式子的特例"这句话
   **要么逐位成立，要么就不该说**（`cases/free_span_tension_step`对
   ``T = M/R``用过同一条口径）；
2. **与`modelgen.generate_spool`的层生长对上**（台阶式）。那一份走无量纲比值，
   取`cases/generator_determinism`那组**二进制精确**入参时两边同样判``==``；
3. **材料守恒的零容差恒等式**：``Σ_{k<n} R(k+½) == ∫₀ⁿ R(s) ds``。
   取二进制精确参数时这是一次真的浮点``==``，**它对"半径更新早了一匝还是
   晚了一匝"有分辨力**，而任何单点半径检查都没有；
4. **前沿推进的运动学**：把积分形式（`turns_at_length_mm`）中心差分去对
   微分形式（``ω = v/R``），实测**二阶**。

## 零容差在本文件里是什么意思

**是浮点``==``，不是"1e-15"。** 三处判``==``的地方各自说明了它凭什么能成立：

* 第1条：``t/1.0``与``turns/1``在IEEE754下**无舍入**，剩下的是``R₀ + t·n``
  与``R₀ + n·t``——浮点乘法可交换，同一个数；
* 第2、3条：入参取2的幂之和（``64.0``、``0.5``、``0.001953125 = 1/512``），
  乘加全程无舍入；
* 第4条（整数守恒）：段数是**整数**，与浮点无关。

**带``2π``的长度那一版判的是实测相对偏差**（2026-08-18实测最坏
``2.26e-16``，取``1e-15``留约4.4倍余量）：``2π``是恒等式两边的公因子，
乘进去之后剩的是这个浮点常数的舍入，**那不是守恒的问题**。
把这两件事混在一起说"长度守恒是零容差"是吹的。

## 必红矩阵（2026-08-18逐条注错**实测**）

| 注错 | 红掉 |
|---|---|
| `turn_mean_radius_mm`取``R(index)``（半径更新晚一匝） | 6 |
| `turn_mean_radius_mm`取``R(index+1)``（半径更新早一匝） | 6 |
| `radius_integral_mm`连续式写成``R₀·n + slope·n²``（漏掉½） | 5 |
| `radius_integral_mm`写成``R(n)·n``（整卷按最外匝算） | 5 |
| `effective_layer_thickness_mm`写成``t·packing``（堆积因子反了） | 4 |
| `layers_at`台阶式改用`ceil`（层数早跳一格） | 4 |
| `radius_mm`漏掉``t_eff``（半径不长） | 8 |
| `segments_in_free_span`漏掉``min``（跨距吃掉全部料） | 3 |
| `segments_on_spool`写成``fed_count − free_span``（少减1） | 2 |
| `turns_at_length_mm`连续式取``(−R₀+√…)/c``那一支 | 1 |

**十条全被抓到，最低一条。** 最后一条只红一处是**有信息的**：
那一支在本文件的参数下还没进到相消区（``R₀/(cS)``不够大），
真正把它打红的是`cases/spool_winding_growth`薄带那几档——
**这说明"数值稳定的写法"这件事只有薄带极限才验得动**，
判据表里那条容差因此写了它的适用区。

注错测法自己的坑（`tests/test_feed.py`第30行已记）：**同字节数的变异会留下
一份被当成新鲜的`.pyc`**。本轮每次变异前清`__pycache__`。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.drives import SpoolTension
from physics_engine.feed import FeedFront
from physics_engine.modelgen import generate_spool
from physics_engine.shapes import FiniteCylinder
from physics_engine.winding import TAU, WindingError, WindingFront, WindingPack

#: 二进制精确的一组卷绕参数：``64 = 2⁶``、``0.5 = 2⁻¹``。
#: 乘加全程无舍入，于是第3条判据可以判真的``==``。
EXACT_BARREL_MM = 64.0
EXACT_THICKNESS_MM = 0.5

#: 一组**不**二进制精确的参数（真机量级：φ127筒、0.213mm带）。
#: 同一条恒等式在这组上给实测相对偏差，两组一起才说得清"零容差"是什么意思。
REAL_BARREL_MM = 63.7
REAL_THICKNESS_MM = 0.213

#: `cases/generator_determinism`冻结的那组带盘入参（该案例的金标依赖它们，
#: 本文件只读不改）：``R_eff = (0.25 + 8×0.001953125)×256 = 68.0``，**精确**。
SPOOL_LENGTH_MM = 256.0
SPOOL_BARREL_RATIO = 0.25
SPOOL_LAYER_RATIO = 0.001953125
SPOOL_LAYERS = 8


def _exact_pack(**overrides: object) -> WindingPack:
    kwargs: dict = {
        "barrel_radius_mm": EXACT_BARREL_MM,
        "tape_thickness_mm": EXACT_THICKNESS_MM,
    }
    kwargs.update(overrides)
    return WindingPack(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------- 判据1：与drives


def test_the_degenerate_pack_and_the_drive_side_radius_are_the_same_float() -> None:
    """退化档下本模块与`drives.SpoolTension`给**同一个浮点数**。

    这条判的不是"两个模型都对"，而是**"旧式子是新式子的特例"这句话成不成立**。
    它一旦只在容差内成立，那句话就该改写成"两者近似一致"，
    而那是完全不同的一句声明。
    """

    pack = WindingPack(
        barrel_radius_mm=REAL_BARREL_MM, tape_thickness_mm=REAL_THICKNESS_MM
    )
    drive = SpoolTension(
        barrel_radius_mm=REAL_BARREL_MM, tape_thickness_mm=REAL_THICKNESS_MM
    )
    for turns in (0.0, 1.0, 2.5, 7.0, 13.0, 100.0, 997.3, 12345.0):
        assert pack.radius_mm(turns) == drive.radius_mm(turns)


def test_the_effective_thickness_is_bit_identical_when_the_packing_is_ideal() -> None:
    """``packing_factor = 1.0``时``t/1.0``无舍入——上一条判据的地基。"""

    pack = WindingPack(
        barrel_radius_mm=REAL_BARREL_MM,
        tape_thickness_mm=REAL_THICKNESS_MM,
        packing_factor=1.0,
    )
    assert pack.effective_layer_thickness_mm == REAL_THICKNESS_MM


def test_a_loose_packing_makes_the_layer_thicker_not_thinner() -> None:
    """堆积因子小于1 ⟹ 一层占掉的厚度**变大**。反了的话半径长得太慢。"""

    dense = _exact_pack(packing_factor=1.0)
    loose = _exact_pack(packing_factor=0.5)
    assert loose.effective_layer_thickness_mm > dense.effective_layer_thickness_mm
    assert loose.effective_layer_thickness_mm == 2.0 * EXACT_THICKNESS_MM
    assert loose.radius_mm(4.0) > dense.radius_mm(4.0)


def test_the_radius_matches_the_hand_written_closed_form() -> None:
    """``R(n) = R₀ + (t/packing)·m(n)``逐档对**手算**闭式。

    手算值写死在表里（不复述实现的式子），二进制精确参数下判``==``。
    这一档同时把**堆积因子**与**每层匝数**都放进了自变量——
    上面那条求和恒等式对堆积因子是**盲的**（它是两边的公因子），
    只有这张表按得住它。
    """

    #: ``t_eff = 0.5 / 0.5 = 1.0``（精确）。列：(每层匝数, 形制, 匝, 手算半径)
    table = [
        (3, "stepped", 0.0, 64.0),
        (3, "stepped", 2.0, 64.0),
        (3, "stepped", 2.999, 64.0),
        (3, "stepped", 3.0, 65.0),
        (3, "stepped", 8.0, 66.0),
        (3, "stepped", 9.0, 67.0),
        (3, "continuous", 1.5, 64.5),
        (3, "continuous", 3.0, 65.0),
        (3, "continuous", 9.0, 67.0),
        (1, "continuous", 13.0, 77.0),
        (1, "stepped", 13.0, 77.0),
    ]
    for turns_per_layer, advance, turns, expected_mm in table:
        pack = _exact_pack(
            turns_per_layer=turns_per_layer, packing_factor=0.5, layer_advance=advance
        )
        assert pack.radius_mm(turns) == expected_mm, (turns_per_layer, advance, turns)

    #: 理想密排那一列（``t_eff = t``）：``R(13) = 64 + 0.5×13 = 70.5``。
    assert _exact_pack().radius_mm(13.0) == 70.5


# ------------------------------------------------------------ 判据2：与modelgen


def test_the_stepped_pack_reproduces_the_generator_layer_growth_bit_for_bit() -> None:
    """台阶式的``R(m)``与`modelgen.generate_spool`的``R_eff``逐位相同。

    生成器走**无量纲比值再乘特征长度**，本模块走**mm**。取
    `cases/generator_determinism`那组二进制精确入参时两条路都无舍入，
    于是可以判``==``——**换一组参数它就只是"在1e-16内"**，
    所以这条判据把入参写死在文件顶上，而不是随手挑两个数。
    """

    parts = generate_spool(
        characteristic_length_mm=SPOOL_LENGTH_MM,
        barrel_radius_ratio=SPOOL_BARREL_RATIO,
        barrel_width_ratio=0.375,
        wound_layers=SPOOL_LAYERS,
        layer_thickness_ratio=SPOOL_LAYER_RATIO,
    )
    barrel = parts[0].shape.shape
    assert isinstance(barrel, FiniteCylinder)

    pack = WindingPack(
        barrel_radius_mm=SPOOL_BARREL_RATIO * SPOOL_LENGTH_MM,
        tape_thickness_mm=SPOOL_LAYER_RATIO * SPOOL_LENGTH_MM,
        turns_per_layer=1,
        layer_advance="stepped",
    )
    assert pack.radius_mm(float(SPOOL_LAYERS)) == barrel.radius_mm


def test_the_stepped_radius_holds_still_inside_a_layer_and_jumps_between_them() -> None:
    """排绕：层内半径不动，跨层才跳一格``t_eff``。

    连续式在同一组参数下**每一匝都在长**——两种形制不是口味，
    在同一组参数下它们给出不同的半径。
    """

    stepped = _exact_pack(turns_per_layer=4, layer_advance="stepped")
    inside = [stepped.radius_mm(float(turn)) for turn in range(4)]
    assert inside == [EXACT_BARREL_MM] * 4
    assert stepped.radius_mm(4.0) == EXACT_BARREL_MM + EXACT_THICKNESS_MM

    continuous = _exact_pack(turns_per_layer=4, layer_advance="continuous")
    assert continuous.radius_mm(2.0) > continuous.radius_mm(1.0)
    assert continuous.radius_mm(2.0) != stepped.radius_mm(2.0)


# ------------------------------------------------- 判据3：材料守恒的零容差恒等式


@pytest.mark.parametrize("turns_per_layer,advance", [
    (1, "continuous"), (4, "continuous"), (1, "stepped"), (4, "stepped"), (7, "stepped"),
])
def test_the_sum_of_the_turn_radii_equals_the_radius_integral_exactly(
    turns_per_layer: int, advance: str
) -> None:
    """``Σ_{k<n} R(k+½) == ∫₀ⁿ R(s) ds``——**判浮点``==``**。

    二进制精确的参数下这一条是真的相等，不是"在容差内"。
    它对**半径更新早一匝还是晚一匝**有分辨力：把中点半径写成``R(k)``或
    ``R(k+1)``，任何单点检查照过，这一条当场差``n·t/2``。
    """

    pack = _exact_pack(turns_per_layer=turns_per_layer, layer_advance=advance)
    for count in range(1, 33):
        summed = 0.0
        for index in range(count):
            summed += pack.turn_mean_radius_mm(index)
        assert summed == pack.radius_integral_mm(float(count))


def test_the_wound_length_matches_the_sum_of_the_turn_circumferences() -> None:
    """带``2π``的那一版：**实测相对偏差**，不是零容差。

    ``2π``是恒等式两边的公因子，乘进去之后剩下的是这个浮点常数的舍入。
    2026-08-18实测最坏``2.2573e-16``（两组参数、五种形制、n最大32；
    最坏那组是``R₀=63.7``、``t=0.213``、每层1匝、台阶式、n=5），
    判据取``1e-15``留约4.4倍余量。**把它写成"零容差"是吹的**。
    """

    worst = 0.0
    for barrel, thickness in (
        (EXACT_BARREL_MM, EXACT_THICKNESS_MM), (REAL_BARREL_MM, REAL_THICKNESS_MM)
    ):
        for turns_per_layer, advance in (
            (1, "continuous"), (4, "continuous"), (1, "stepped"),
            (4, "stepped"), (7, "stepped"),
        ):
            pack = WindingPack(
                barrel_radius_mm=barrel, tape_thickness_mm=thickness,
                turns_per_layer=turns_per_layer, layer_advance=advance,
            )
            for count in (1, 2, 5, 13, 32):
                summed = 0.0
                for index in range(count):
                    summed += pack.turn_length_mm(index)
                closed = pack.wound_length_mm(float(count))
                worst = max(worst, abs(summed - closed) / closed)
    assert worst <= 1.0e-15, worst


def test_the_length_is_the_integral_of_the_radius_not_the_radius_times_the_turns() -> None:
    """整卷长度**夹在**"全按最里匝算"与"全按最外匝算"之间。

    这条挡的是两个错得像对的写法：``2π·R₀·n``（低估）与``2π·R(n)·n``（高估）。
    它比恒等式弱，但它在**任何**参数下都成立，不依赖二进制精确。
    """

    pack = WindingPack(
        barrel_radius_mm=REAL_BARREL_MM, tape_thickness_mm=REAL_THICKNESS_MM
    )
    for turns in (1.0, 5.0, 13.0, 200.0):
        inner = TAU * pack.barrel_radius_mm * turns
        outer = TAU * pack.radius_mm(turns) * turns
        assert inner < pack.wound_length_mm(turns) < outer


@pytest.mark.parametrize("turns_per_layer,advance", [
    (1, "continuous"), (5, "continuous"), (1, "stepped"), (5, "stepped"),
])
def test_the_length_to_turns_inverse_round_trips(
    turns_per_layer: int, advance: str
) -> None:
    """``turns_at_length_mm(wound_length_mm(n)) ≈ n``。

    2026-08-18实测最坏相对偏差``2.2204e-16``（四种形制、八个匝数，即1 ulp），
    判据取``1e-14``——反解走一次开方，比正向多一步舍入。
    """

    pack = WindingPack(
        barrel_radius_mm=REAL_BARREL_MM, tape_thickness_mm=REAL_THICKNESS_MM,
        turns_per_layer=turns_per_layer, layer_advance=advance,
    )
    assert pack.turns_at_length_mm(0.0) == 0.0
    worst = 0.0
    for turns in (0.5, 1.0, 3.7, 12.0, 13.5, 64.0, 200.0, 1000.25):
        back = pack.turns_at_length_mm(pack.wound_length_mm(turns))
        worst = max(worst, abs(back - turns) / turns)
    assert worst <= 1.0e-14, worst


def test_the_radius_and_the_length_never_go_backwards() -> None:
    """半径单调不减、长度严格单调增——喂进去的料不会从盘上消失。"""

    for advance in ("continuous", "stepped"):
        pack = _exact_pack(turns_per_layer=3, layer_advance=advance)
        radii = [pack.radius_mm(turn * 0.25) for turn in range(0, 80)]
        lengths = [pack.wound_length_mm(turn * 0.25) for turn in range(0, 80)]
        assert all(b >= a for a, b in zip(radii, radii[1:]))
        assert all(b > a for a, b in zip(lengths[1:], lengths[2:]))


# ------------------------------------------------------- 判据4：喂料前沿与运动学


def _front(rest_length_mm: float, node_budget: int = 1400) -> FeedFront:
    return FeedFront(
        node_budget=node_budget, rest_length_mm=rest_length_mm,
        inlet_mm=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
    )


@pytest.mark.parametrize("rest_length_mm", [0.25, 0.1])
def test_the_material_balance_is_an_exact_integer_identity(rest_length_mm: float) -> None:
    """``喂进来的段 == 跨距里的段 + 盘上的段``——**整数，判``==``**。

    两组``rest_length``一起跑是有意的：``0.25``二进制精确、``0.1``不精确。
    **整数恒等式对两组都成立**，因为守恒记在段数上不记在浮点长度上。
    """

    front = _front(rest_length_mm)
    winding = WindingFront(
        pack=_exact_pack(), front=front, free_span_segments=7
    )
    for fed_count in range(2, front.node_budget + 1):
        assert (
            winding.segments_in_free_span(fed_count)
            + winding.segments_on_spool(fed_count)
            == winding.segments_fed(fed_count)
        )
    assert winding.segments_on_spool(8) == 0
    assert winding.segments_on_spool(9) == 1
    assert winding.segments_on_spool(front.node_budget) == front.node_budget - 8


def test_the_spool_never_holds_a_negative_amount_of_material() -> None:
    """落位前盘上恰为**零**段，任何时候都**不为负**。

    ## 这条判据是必红矩阵逼出来的（2026-08-18）

    把`segments_in_free_span`的``min``删掉——让跨距无条件吃掉
    ``free_span_segments``段——**整数守恒那道门一条都不红**：
    ``7 + (k−1−7) == k−1``在``k < 8``时照样成立，只不过盘上是**负**的。

    **守恒恒等式对"两边同时错了一个常数"是盲的。** 这正是本仓
    plans/09教训一那种洞的数值版：两处各自自洽，合起来无意义。
    补这一条之后那个变异体红3处。
    """

    front = _front(0.25, node_budget=64)
    winding = WindingFront(pack=_exact_pack(), front=front, free_span_segments=7)
    for fed_count in range(2, front.node_budget + 1):
        assert winding.segments_on_spool(fed_count) >= 0, fed_count
        assert winding.segments_in_free_span(fed_count) >= 0, fed_count
        assert winding.length_on_spool_mm(fed_count) >= 0.0, fed_count
    assert [winding.segments_on_spool(fed) for fed in range(2, 11)] == [
        0, 0, 0, 0, 0, 0, 0, 1, 2
    ]

    #: 零跨距是合法声明：喂料口就在落位点上，第一段料立刻上盘。
    touching = WindingFront(pack=_exact_pack(), front=front, free_span_segments=0)
    assert touching.segments_on_spool(2) == 1


def test_the_float_length_balance_is_exact_only_when_the_segment_is_binary_exact() -> None:
    """浮点长度的守恒**要看``rest_length``的二进制形状**，整数守恒不看。

    这条是上一条的负空间：``0.25``下浮点长度也逐位守恒，``0.1``下差一个ulp。
    2026-08-18实测``0.1``那组1392个喂料档里**373档**逐位不守恒，
    最坏相对偏差``2.2170e-16``。
    **把守恒写在浮点长度上，"零容差"就只能靠运气兜**——这条判据把那份运气量出来。
    """

    pack = _exact_pack()
    exact = WindingFront(pack=pack, front=_front(0.25), free_span_segments=7)
    for fed_count in (9, 50, 400, 1400):
        span = exact.segments_in_free_span(fed_count) * 0.25
        assert exact.length_on_spool_mm(fed_count) + span == (fed_count - 1) * 0.25

    inexact = WindingFront(pack=pack, front=_front(0.1), free_span_segments=7)
    worst = 0.0
    mismatched = 0
    for fed_count in range(9, 1401):
        span = inexact.segments_in_free_span(fed_count) * 0.1
        total = (fed_count - 1) * 0.1
        drift = inexact.length_on_spool_mm(fed_count) + span - total
        if drift != 0.0:
            mismatched += 1
        worst = max(worst, abs(drift) / total)
    assert mismatched > 0, "0.1的浮点长度居然处处逐位守恒——那说明这条判据没在验它想验的"
    assert worst <= 1.0e-15, worst


def test_the_front_angular_rate_is_the_derivative_of_the_turn_count() -> None:
    """``ω = v/R``与``n(ℓ)``的中心差分对上，实测**二阶**。

    积分形式（`turns_at_length_mm`）与微分形式（`front_angular_rate_rad_s`）
    是同一条律的两种写法。这道门抓的是**半边接线**——半径在积分里长了、
    在角速度里没长，单点值照样对得上，收敛阶当场掉到零阶。

    2026-08-18实测五档阶：2.0000007 / 2.0000002 / 1.9999984 / 1.9999960 / 2.0001425，
    判据取``≥ 1.9``。
    """

    pack = _exact_pack(tape_thickness_mm=0.25)
    line_speed_mm_s = 120.0
    at_s = 3.0
    errors: list[float] = []
    for level in range(6):
        step_s = 0.5 / (2 ** level)
        ahead = pack.turns_at_length_mm(line_speed_mm_s * (at_s + step_s))
        behind = pack.turns_at_length_mm(line_speed_mm_s * (at_s - step_s))
        measured = TAU * (ahead - behind) / (2.0 * step_s)
        here = pack.turns_at_length_mm(line_speed_mm_s * at_s)
        closed = pack.front_angular_rate_rad_s(here, line_speed_mm_s)
        errors.append(abs(measured - closed) / closed)
    orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert all(order >= 1.9 for order in orders), orders


def test_the_spool_radius_grows_as_the_front_feeds() -> None:
    """喂料—半径的接线：多喂料 ⟹ 匝数增 ⟹ 半径增。

    这是S5.4的`missing`点名的"**匝数与半径生长的接线**"那一条，
    也是`drives.SpoolTension`那个``turns``入参今天的出处。
    """

    winding = WindingFront(
        pack=_exact_pack(barrel_radius_mm=32.0), front=_front(4.0, node_budget=1400),
        free_span_segments=3,
    )
    samples = [200, 500, 900, 1400]
    turns = [winding.turns_on_spool(fed) for fed in samples]
    radii = [winding.radius_on_spool_mm(fed) for fed in samples]
    assert all(b > a for a, b in zip(turns, turns[1:]))
    assert all(b > a for a, b in zip(radii, radii[1:]))
    assert turns[-1] > 13.0, turns[-1]


# --------------------------------------------------------------------- 失败关闭


@pytest.mark.parametrize("kwargs", [
    {"barrel_radius_mm": 0.0},
    {"barrel_radius_mm": -1.0},
    {"barrel_radius_mm": math.inf},
    {"tape_thickness_mm": 0.0},
    {"tape_thickness_mm": math.nan},
    {"turns_per_layer": 0},
    {"turns_per_layer": -3},
    {"turns_per_layer": 2.5},
    {"turns_per_layer": True},
    {"packing_factor": 0.0},
    {"packing_factor": 1.5},
    {"packing_factor": -0.5},
    {"layer_advance": "spiral"},
    {"layer_advance": ""},
])
def test_an_impossible_pack_fails_closed(kwargs: dict) -> None:
    """声明期的每一条都失败关闭——**没有一条走"默认成一个合理值"**。"""

    with pytest.raises(WindingError):
        _exact_pack(**kwargs)


@pytest.mark.parametrize("turns", [-1.0, -1e-30, math.nan, math.inf, True, "3"])
def test_a_bad_turn_count_fails_closed(turns: object) -> None:
    with pytest.raises(WindingError):
        _exact_pack().radius_mm(turns)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [-1, 2.0, True, "0"])
def test_a_bad_turn_index_fails_closed(index: object) -> None:
    with pytest.raises(WindingError):
        _exact_pack().turn_mean_radius_mm(index)  # type: ignore[arg-type]


@pytest.mark.parametrize("length", [-1.0, math.nan, math.inf, True, None])
def test_a_bad_wound_length_fails_closed(length: object) -> None:
    with pytest.raises(WindingError):
        _exact_pack().turns_at_length_mm(length)  # type: ignore[arg-type]


def test_a_negative_line_speed_fails_closed() -> None:
    """负线速度是在退绕——本模块的匝数只增不减，不假装能算它。"""

    with pytest.raises(WindingError):
        _exact_pack().front_angular_rate_rad_s(3.0, -1.0)


@pytest.mark.parametrize("free_span_segments", [-1, 1400, 5000, 2.0, True])
def test_a_bad_free_span_fails_closed(free_span_segments: object) -> None:
    """跨距比整卷还长 ⟹ 全喂完也够不到盘，**那必须在声明期发现**。"""

    with pytest.raises(WindingError):
        WindingFront(
            pack=_exact_pack(), front=_front(0.25, node_budget=1400),
            free_span_segments=free_span_segments,  # type: ignore[arg-type]
        )


def test_the_front_and_the_pack_must_be_the_real_types() -> None:
    """鸭子类型在这里是失败关闭的：两处各说各的段长是plans/09教训一那种洞。"""

    with pytest.raises(WindingError):
        WindingFront(pack=object(), front=_front(0.25))  # type: ignore[arg-type]
    with pytest.raises(WindingError):
        WindingFront(pack=_exact_pack(), front=object())  # type: ignore[arg-type]
