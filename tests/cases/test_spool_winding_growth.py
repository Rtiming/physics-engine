"""`case/spool_winding_growth`的conformance门（轴7规则3）。

四条oracle各验一件不同的事：

1. **半径逐档对手算闭式**——``R(n) = R₀ + (t/packing)·m(n)``，零容差；
2. **材料守恒**——总长度＝每匝周长之和，且``长度 → 匝数``反解能还原；
3. **同心圆理想化对真阿基米德螺线的建模偏差以二阶收敛**——
   本案例最硬的一条：金标是螺线弧长的**独立闭式**，不是本仓任何一段代码；
4. **喂料段账**——拉格朗日喂料前沿送进来的整数段，逐段落到盘上。

判据数全部来自清单，测试只读不算（spec/08规则3）。
第3条的**阶**在测试里由清单冻结的逐档偏差重算——那不是"复述oracle公式"，
是把清单里的一串数按0087那条纪律**判单调**：
"逐档差不单调就是没进渐近区，那个阶不许当收敛阶引用"。

## 必红矩阵（2026-08-18逐条注错**实测**，与`tests/test_winding.py`同一批变异体）

| 注错 | 本文件红掉 |
|---|---|
| `turn_mean_radius_mm`取``R(index)``（半径更新晚一匝） | 2 |
| `turn_mean_radius_mm`取``R(index+1)``（半径更新早一匝） | 2 |
| `radius_integral_mm`连续式漏掉½ | 3 |
| `radius_integral_mm`写成``R(n)·n`` | 3 |
| `effective_layer_thickness_mm`写成``t·packing`` | 3 |
| `layers_at`台阶式改用`ceil` | 1 |
| `radius_mm`漏掉``t_eff`` | 4 |
| `segments_in_free_span`漏掉``min`` | 1 |
| `segments_on_spool`写成``fed_count − free_span`` | 1 |
| `turns_at_length_mm`连续式取``(−R₀+√…)/c``那一支 | 1 |

**十条全被抓到。** 收敛阶那一条判据（第3条）单独看抓了其中4条——
它是本文件里分辨力最强的一格，因为**建模偏差是1e-7量级，
任何一处半径的错都比它大四个数量级以上，当场把阶打到零**。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.feed import FeedFront
from physics_engine.oracles import load_manifest
from physics_engine.winding import WindingFront, WindingPack

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/spool_winding_growth/oracle.json", root=ROOT)


def _case(oracle_id: str):
    return MANIFEST.oracle(oracle_id)


def test_the_radius_ladder_matches_the_hand_written_closed_form() -> None:
    """半径逐档对手算闭式，**零容差**。

    ``layers``与``radius_mm``两格都判：半径对了不代表层数对了——
    ``t_eff``与``m``可以一个偏大一个偏小而乘积不变。
    """

    case = _case("oracle:winding/radius_ladder")
    radii: list[float] = []
    layers: list[float] = []
    thickness: float | None = None
    for turns_per_layer, advance, turns in case.inputs["ladder"]:
        pack = WindingPack(
            barrel_radius_mm=case.inputs["barrel_radius_mm"],
            tape_thickness_mm=case.inputs["tape_thickness_mm"],
            turns_per_layer=int(turns_per_layer),
            packing_factor=case.inputs["packing_factor"],
            layer_advance=advance,
        )
        thickness = pack.effective_layer_thickness_mm
        radii.append(pack.radius_mm(turns))
        layers.append(pack.layers_at(turns))
    case.check_all({
        "effective_layer_thickness_mm": thickness,
        "radius_mm": radii,
        "layers": layers,
    })


def test_the_wound_material_is_conserved_turn_by_turn() -> None:
    """材料守恒：总长度＝每匝周长之和，且反解还原匝数。

    **这一格对"半径更新早了一匝还是晚了一匝"有分辨力**——
    差一匝时两边差``2π·n·t_eff/2``，13匝那一档约``408mm``，
    不是几个ulp，任何一条rel判据都挡不住它蒙混。
    """

    case = _case("oracle:winding/concentric_material_length")
    pack = WindingPack(
        barrel_radius_mm=case.inputs["barrel_radius_mm"],
        tape_thickness_mm=case.inputs["tape_thickness_mm"],
        packing_factor=case.inputs["packing_factor"],
    )
    lengths = [pack.wound_length_mm(turns) for turns in case.inputs["turns"]]
    sums: list[float] = []
    for turns in case.inputs["integer_turns"]:
        total = 0.0
        for index in range(int(turns)):
            total += pack.turn_length_mm(index)
        sums.append(total)
    recovered = [
        pack.turns_at_length_mm(pack.wound_length_mm(turns))
        for turns in case.inputs["turns"]
    ]
    case.check_all({
        "wound_length_mm": lengths,
        "per_turn_circumference_sum_mm": sums,
        "turns_recovered": recovered,
    })


def test_the_concentric_model_converges_to_the_archimedean_spiral_at_second_order() -> None:
    """同心圆理想化对**真螺线弧长**的偏差以二阶收敛。

    金标（螺线弧长闭式）由生成器算出并冻在``inputs``里，与被验路径
    **不出自同一支笔**——这是本案例区别于`tests/test_winding.py`那些
    自洽判据的地方。它住``inputs``而不是``expected``是有理由的：
    ``expected``的语义是"内核应当产出什么"，而引擎**有意不算**螺线弧长；
    放进``expected``会让测试只能拿冻结值对它自己（第一版就是那样，
    那一格当时永远绿）。

    除了逐档数值，另判四条**关系型**判据——它们才是这条oracle的主判据：

    1. 逐档偏差**单调下降**（0087：不单调就是没进渐近区，那个阶不许引用）；
    2. 逐档阶``≥ 1.9``；
    3. 阶本身**单调递增**；
    4. 阶**从下方**逼近2（展开的次项符号决定了这一点）。
    """

    case = _case("oracle:winding/concentric_versus_archimedean_order")
    spiral = case.inputs["spiral_arc_length_mm"]
    lengths: list[float] = []
    for thickness in case.inputs["tape_thickness_mm"]:
        pack = WindingPack(
            barrel_radius_mm=case.inputs["barrel_radius_mm"],
            tape_thickness_mm=thickness,
        )
        lengths.append(pack.wound_length_mm(case.inputs["turns"]))
    deviations = [
        abs(model - exact) / exact for model, exact in zip(lengths, spiral, strict=True)
    ]
    orders = [
        math.log2(before / after) for before, after in zip(deviations, deviations[1:])
    ]
    case.check_all({
        "concentric_length_mm": lengths,
        "relative_deviation": deviations,
        "observed_order": orders,
    })

    claim = case.inputs["asymptotic_order_claim"]
    assert all(after < before for before, after in zip(deviations, deviations[1:])), (
        "逐档偏差不单调下降——没有进渐近区，下面那个阶不许当收敛阶引用", deviations
    )
    assert all(order >= 1.9 for order in orders), orders
    assert all(after > before for before, after in zip(orders, orders[1:])), (
        "阶本身不单调趋近——首项以外还有同量级的东西在，二阶这个说法要重审", orders
    )
    assert orders[-1] < claim, (
        "阶从**上方**逼近2了——展开的次项符号与推导相反，推导要重看", orders[-1]
    )
    assert claim - orders[-1] < 0.01, (
        "最细一档离2还差得远——七档扫下来没进渐近区，或者阶根本不是2", orders[-1]
    )


def test_the_feed_front_hands_every_segment_to_the_spool() -> None:
    """喂料段账：``喂进来 = 跨距里 + 盘上``，整数、零容差。

    这条是S5.4的`missing`点名的"**匝数与半径生长的接线**"的兑现：
    喂料给长度、长度给匝数、匝数给半径，而半径正是`drives.SpoolTension`
    那个``turns``入参要的东西。
    """

    case = _case("oracle:winding/feed_front_material_balance")
    front = FeedFront(
        node_budget=int(case.inputs["node_budget"]),
        rest_length_mm=case.inputs["rest_length_mm"],
        inlet_mm=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
    )
    winding = WindingFront(
        pack=WindingPack(
            barrel_radius_mm=case.inputs["barrel_radius_mm"],
            tape_thickness_mm=case.inputs["tape_thickness_mm"],
        ),
        front=front,
        free_span_segments=int(case.inputs["free_span_segments"]),
    )
    samples = [int(count) for count in case.inputs["fed_counts"]]
    case.check_all({
        "segments_fed": [winding.segments_fed(count) for count in samples],
        "segments_in_free_span": [
            winding.segments_in_free_span(count) for count in samples
        ],
        "segments_on_spool": [winding.segments_on_spool(count) for count in samples],
        "length_on_spool_mm": [winding.length_on_spool_mm(count) for count in samples],
        "turns_on_spool": [winding.turns_on_spool(count) for count in samples],
        "radius_on_spool_mm": [
            winding.radius_on_spool_mm(count) for count in samples
        ],
    })

    #: 关系型判据（清单里没有、也不该有：它不是一个冻结的数）：
    #: 盘上的段数**任何时候都不为负**。守恒等式对"两边同时错了一个常数"是盲的。
    for count in range(2, front.node_budget + 1):
        assert winding.segments_on_spool(count) >= 0, count


def test_the_angular_rate_is_the_line_speed_over_the_current_radius() -> None:
    """``ω = v/R``——前沿推进的运动学，用清单里冻结的半径直接对。

    落位点的角速度随卷径增大而**下降**：同一条线速度下，
    盘越满转得越慢。这是绕线机上最直观的一条现象，也是"半径真的在长"
    最便宜的一次独立确认。
    """

    case = _case("oracle:winding/feed_front_material_balance")
    pack = WindingPack(
        barrel_radius_mm=case.inputs["barrel_radius_mm"],
        tape_thickness_mm=case.inputs["tape_thickness_mm"],
    )
    line_speed_mm_s = 250.0
    radii = case.expected["radius_on_spool_mm"]
    rates = [
        pack.front_angular_rate_rad_s(
            pack.turns_at_length_mm(length), line_speed_mm_s
        )
        for length in case.expected["length_on_spool_mm"]
    ]
    for radius, rate in zip(radii, rates, strict=True):
        assert rate == pytest.approx(line_speed_mm_s / radius, rel=1e-14)
    assert rates[-1] < rates[0], (rates[0], rates[-1])
    #: 一圈的时间``2π/ω``——空盘与满盘之比恰是半径之比。
    assert (rates[0] / rates[-1]) == pytest.approx(radii[-1] / radii[0], rel=1e-14)


def test_every_declared_oracle_is_actually_exercised() -> None:
    """清单里的四条oracle**全部**被上面的测试用过——不许有挂着没人验的金标。

    `check_case_pages.py`判的是页与清单的存在性，判不了"清单里的某一条
    其实没有任何测试读它"。那种条目会随内核演进悄悄失效而没人知道。
    """

    exercised = {
        "oracle:winding/radius_ladder",
        "oracle:winding/concentric_material_length",
        "oracle:winding/concentric_versus_archimedean_order",
        "oracle:winding/feed_front_material_balance",
    }
    declared = {case.id for case in MANIFEST.oracles}
    assert declared == exercised, declared ^ exercised
