#!/usr/bin/env python3
"""`case/spool_winding_growth`的金标——**闭式解与手推常数，独立于被验内核**。

本脚本**不import`physics_engine.winding`**，一行都不。它写四样东西：

1. **半径闭式逐档**：``R(n) = R₀ + (t/packing)·m(n)``。入参取二进制精确值
   （``64 = 2⁶``、``0.5 = 2⁻¹``、``packing = 0.5``⟹``t_eff = 1.0``），
   于是这一档的期望值是**手算整数**，写在下面的注释里逐条推过；

2. **同心圆理想化的材料长度**：``L(n) = 2π·(R₀n + c·n²/2)``。
   这是把每一匝当成一个闭合圆、半径取该匝中点值的结果；

3. **真阿基米德螺线的弧长（本案例最硬的一条金标）**。螺线``r(θ) = R₀ + aθ``、
   ``a = t/2π``，弧长

       L_spiral = ∫₀^Θ √(r² + a²) dθ = (1/a)∫_{R₀}^{R₁} √(u² + a²) du
                = [ u√(u²+a²) + a²·asinh(u/a) ] / (2a)  from R₀ to R₁

   （标准结果，见 Gray, *Modern Differential Geometry of Curves and Surfaces*,
   2nd ed., §1.5"Arc Length"，把``ds² = dr² + r²dθ²``代入``r = R₀ + aθ``即得；
   本式在任何一本微积分教材的"极坐标弧长"一节也能查到。）

   **第2条是第3条在``t/R₀ → 0``时的极限**，两者相对偏差应当以**二阶**收敛：
   把被积函数展开``√(u²+a²) = u(1 + a²/2u² + …)``，一阶项积出来恰是第2条，
   剩下的首项是``(a/2)·ln(R₁/R₀)``，相对于``L ≈ 2πnR₀``是``O((t/R₀)²)``。
   本案例量的就是这个阶——**它同时是"同心圆理想化"这个建模选择的误差预算**；

4. **喂料段账**：整数段数、盘上长度与由长度反解的匝数。

## 为什么阶要量，而不是只判一个偏差

只判一个偏差说明不了"偏差来自建模还是来自bug"。**阶说得了**：
同心圆理想化的误差必须是二阶的；掉到一阶说明半径在某处少长了半匝，
掉到零阶说明两条路根本不是同一个模型。0087那条收敛阶纪律
（"逐档差不单调就是没进渐近区，那个阶不许当收敛阶引用"）在这里同样适用，
所以本脚本把**逐档偏差**也写进金标，让测试自己判单调。

## 阶为什么不是恰好2.000

有限档的``log₂``估计带着次阶项。实测七档给出 1.9354 / 1.9670 / 1.9833 /
1.9916 / 1.9958 / 1.9986——**单调趋近2且从下方逼近**，符合"首项``a²lnR``、
次项``a⁴``"的展开。判据因此取"每一档 ≥ 1.9 且逐档递增"，
而不是"最后一档 ≈ 2 ± 0.01"——后者对"两条路都错了同一个因子"是盲的。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/spool_winding_growth"
ALGORITHM_VERSION = "1.0.0"

TAU = 2.0 * math.pi

#: 二进制精确的一卷：φ128筒、0.5mm带、堆积因子0.5（⟹有效层厚恰1.0mm）。
BARREL_MM = 64.0
THICKNESS_MM = 0.5
PACKING = 0.5
EFFECTIVE_THICKNESS_MM = THICKNESS_MM / PACKING  # = 1.0，精确

#: 半径逐档的入参表：(每层匝数, 形制, 匝数)。期望值全部手算，见下。
RADIUS_LADDER = [
    (1, "continuous", 0.0), (1, "continuous", 1.0), (1, "continuous", 6.5),
    (1, "continuous", 13.0), (1, "continuous", 100.0),
    (4, "continuous", 2.0), (4, "continuous", 4.0), (4, "continuous", 13.0),
    (4, "stepped", 0.0), (4, "stepped", 3.0), (4, "stepped", 4.0),
    (4, "stepped", 7.0), (4, "stepped", 8.0), (4, "stepped", 13.0),
]

#: 手算半径（mm）。``t_eff = 1.0``，所以``R = 64 + m``，``m``是层数：
#: 连续式``m = n/tpl``、台阶式``m = ⌊n/tpl⌋``。
#:   tpl=1 连续：n=0→64；1→65；6.5→70.5；13→77；100→164
#:   tpl=4 连续：n=2→64.5；4→65；13→67.25
#:   tpl=4 台阶：n=0→64；3→64；4→65；7→65；8→66；13→67
RADIUS_EXPECTED_MM = [
    64.0, 65.0, 70.5, 77.0, 164.0,
    64.5, 65.0, 67.25,
    64.0, 64.0, 65.0, 65.0, 66.0, 67.0,
]

#: 材料长度逐档的匝数（连续式、每层1匝）。
LENGTH_LADDER = [1.0, 2.0, 5.0, 13.0, 13.5, 64.0]

#: 收敛阶阶梯：带厚逐档减半，筒径与匝数不动 ⟹ ``ε = t/R₀``逐档减半。
ORDER_BARREL_MM = 64.0
ORDER_TURNS = 12.0
ORDER_THICKNESSES_MM = [0.5 / (2 ** level) for level in range(7)]

#: 喂料段账：整卷1400个节点、静止段长4mm（二进制精确）、自由跨距3段。
FEED_NODE_BUDGET = 1400
FEED_REST_LENGTH_MM = 4.0
FEED_FREE_SPAN_SEGMENTS = 3
FEED_SAMPLES = [2, 4, 5, 200, 500, 900, 1400]
FEED_BARREL_MM = 32.0
FEED_THICKNESS_MM = 0.5


def _layers(turns: float, turns_per_layer: int, advance: str) -> float:
    quotient = turns / turns_per_layer
    return float(math.floor(quotient)) if advance == "stepped" else quotient


def _concentric_length_mm(barrel_mm: float, thickness_mm: float, turns: float) -> float:
    """同心圆理想化，写成**平均直径式**``L = π·n·(D₀ + Dₙ)/2 = π·n·(2R₀ + t·n)``。

    ## 这个写法是有意选的（2026-08-18，一次被抓到的**假独立**）

    第一版写的是``2π·(R₀n + t·n²/2)``——与`winding.radius_integral_mm`
    **逐字符同构**，于是"金标 vs 被验"实测相对偏差恰为``0.0``。
    那个零不是精度好，是**两边根本是同一支笔**：这条oracle当时验不了任何东西。

    改成平均直径式之后两边代数恒等而**结合次序不同**，那个零就变成了真的舍入。
    这条经历写在这里是因为它是本仓"金标必须独立"那条纪律的一次实例：
    **相对偏差恰好为零，往往是金标出了问题而不是实现特别好。**

    机械手册里卷料估长用的正是这一式（``L = πn(D_out + D_in)/2``）。
    """

    return math.pi * turns * (2.0 * barrel_mm + thickness_mm * turns)


def _spiral_arc_length_mm(barrel_mm: float, thickness_mm: float, turns: float) -> float:
    """真阿基米德螺线的弧长闭式（见模块文档第3条）。"""

    growth = thickness_mm / TAU
    outer = barrel_mm + thickness_mm * turns

    def primitive(radius: float) -> float:
        return 0.5 * (
            radius * math.hypot(radius, growth)
            + growth * growth * math.asinh(radius / growth)
        )

    return (primitive(outer) - primitive(barrel_mm)) / growth


def radius_oracle() -> dict:
    return {
        "id": "oracle:winding/radius_ladder",
        "inputs": {
            "kind": "radius_versus_turns",
            "barrel_radius_mm": BARREL_MM,
            "tape_thickness_mm": THICKNESS_MM,
            "packing_factor": PACKING,
            "ladder": [list(entry) for entry in RADIUS_LADDER],
        },
        "expected": {
            "effective_layer_thickness_mm": EFFECTIVE_THICKNESS_MM,
            "radius_mm": list(RADIUS_EXPECTED_MM),
            "layers": [
                _layers(turns, per_layer, advance)
                for per_layer, advance, turns in RADIUS_LADDER
            ],
        },
        "tolerances": {
            "effective_layer_thickness_mm": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**零容差，判逐位相等**：``0.5/0.5``在IEEE754下无舍入，"
                          "期望恰是``1.0``。它是本条其余期望值全为整数或半整数的前提——"
                          "这一格松了，下面那14个手算值就不再是手算得出来的",
            },
            "radius_mm": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**零容差，判逐位相等**：入参全为2的幂之和，``R₀ + t_eff·m``"
                          "这一步乘加无舍入，期望值是手算整数/半整数（见生成器注释里"
                          "逐条的推导）。**这一格不该有容差**——有容差就意味着有一步"
                          "我们没算清楚是哪来的",
            },
            "layers": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**零容差**：层数在台阶式下是整数、在连续式下是``n/tpl``"
                          "且两者都二进制精确。把它单列出来是因为**半径对了不代表层数对了**"
                          "（``t_eff``与``m``可以一个偏大一个偏小而乘积不变）",
            },
        },
    }


def length_oracle() -> dict:
    lengths = [
        _concentric_length_mm(BARREL_MM, EFFECTIVE_THICKNESS_MM, turns)
        for turns in LENGTH_LADDER
    ]
    #: 逐匝周长之和（k从0到n−1，取中点半径）——**只对整数匝算得动**。
    #: 半匝那一档不进这一格（"已知失效清单"里写明了为什么）。
    integer_turns = [turns for turns in LENGTH_LADDER if turns == math.floor(turns)]
    per_turn_sums = []
    for turns in integer_turns:
        total = 0.0
        for index in range(int(turns)):
            #: 第``index``匝的周长``π·D``，直径取该匝中点值——同样与被验路径
            #: （``2π·R``）代数恒等而结合次序不同。
            diameter = 2.0 * BARREL_MM + 2.0 * EFFECTIVE_THICKNESS_MM * (index + 0.5)
            total += math.pi * diameter
        per_turn_sums.append(total)
    return {
        "id": "oracle:winding/concentric_material_length",
        "inputs": {
            "kind": "wound_length_versus_turns",
            "barrel_radius_mm": BARREL_MM,
            "tape_thickness_mm": THICKNESS_MM,
            "packing_factor": PACKING,
            "turns": list(LENGTH_LADDER),
            "integer_turns": integer_turns,
        },
        "expected": {
            "wound_length_mm": lengths,
            "per_turn_circumference_sum_mm": per_turn_sums,
            "turns_recovered": list(LENGTH_LADDER),
        },
        "tolerances": {
            "wound_length_mm": {
                "rel": 1e-15, "abs": 0.0,
                "reason": "本脚本走**平均直径式**``πn(2R₀+tn)``、内核走**半径积分式**"
                          "``2π(R₀n+tn²/2)``，代数恒等而结合次序不同，差几个ulp。"
                          "2026-08-18实测最坏相对偏差``1.5794e-16``（六档），"
                          "取``1e-15``留约6.3倍余量。**零容差的那一版是提出2π之后的"
                          "半径积分**，它在`tests/test_winding.py`里判``==``",
            },
            "per_turn_circumference_sum_mm": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**零容差，判逐位相等**，而且这个零是**结构性的不是运气**："
                          "本脚本每匝写``π·D``、内核写``2π·R``，而``D = 2R``且"
                          "**乘2在二进制浮点上精确**，两式因此是同一个实数的同一次舍入。"
                          "2026-08-18实测五档全部``0.0``。**代价要写清楚**：这一格因此"
                          "验不了『结合次序』这件事，它换来的是另一样东西——"
                          "求和次序在本仓是形制（spec/12第3.3节），零容差把它钉住。"
                          "而它真正的分辨力在于对『半径更新早了一匝还是晚了一匝』敏感："
                          "差一匝时两边差``2π·n·t_eff/2``，13匝那一档``≈ 408mm``",
            },
            "turns_recovered": {
                "rel": 1e-14, "abs": 0.0,
                "reason": "``长度 → 匝数``的反解走一次开方，比正向多一步舍入。"
                          "期望值就是入参那一列匝数本身（**独立by construction**："
                          "本脚本一个字都没算，它冻的是『反解应当还原原值』）。"
                          "2026-08-18实测最坏相对偏差``1.3158e-16``（六档），"
                          "取``1e-14``留约76倍余量——余量留这么大是因为"
                          "**这一档的入参恰好二进制精确，真机参数下相消会更重**，"
                          "不能拿这六档当反解精度的一般结论",
            },
        },
    }


def order_oracle() -> dict:
    """同心圆理想化 vs 真螺线：**建模偏差的收敛阶**。

    ## 螺线弧长住在``inputs``而不是``expected``（2026-08-18，一次被抓到的同义反复）

    第一版把``spiral_arc_length_mm``放进``expected``，而测试拿不到第二条算它的路
    （引擎有意不算螺线弧长），于是测试只能把``case.expected[...]``原样喂回
    ``check_all``——**拿冻结值对它自己**。那一格当时永远绿，验的是JSON读得对不对。

    ``expected``的语义是"**被验内核应当产出什么**"。螺线弧长不是内核产出的，
    它是**参照**，所以它住``inputs``：像`free_span_tension_step`把``reel_radius_mm``
    放进``inputs``一样。它的可审性由生成器的SHA与本函数的推导注释保证。

    留在``expected``里的三格全部由内核产出或由内核产出推得：
    同心圆长度、相对偏差、逐档阶。
    """

    spiral = [
        _spiral_arc_length_mm(ORDER_BARREL_MM, thickness, ORDER_TURNS)
        for thickness in ORDER_THICKNESSES_MM
    ]
    concentric = [
        _concentric_length_mm(ORDER_BARREL_MM, thickness, ORDER_TURNS)
        for thickness in ORDER_THICKNESSES_MM
    ]
    deviations = [
        abs(model - exact) / exact for model, exact in zip(concentric, spiral)
    ]
    orders = [
        math.log2(before / after) for before, after in zip(deviations, deviations[1:])
    ]
    return {
        "id": "oracle:winding/concentric_versus_archimedean_order",
        "inputs": {
            "kind": "modelling_error_convergence",
            "barrel_radius_mm": ORDER_BARREL_MM,
            "turns": ORDER_TURNS,
            "tape_thickness_mm": list(ORDER_THICKNESSES_MM),
            "spiral_arc_length_mm": spiral,
            "asymptotic_order_claim": 2.0,
        },
        "expected": {
            "concentric_length_mm": concentric,
            "relative_deviation": deviations,
            "observed_order": orders,
        },
        "tolerances": {
            "concentric_length_mm": {
                "rel": 1e-15, "abs": 0.0,
                "reason": "本脚本走**平均直径式**``πn(2R₀+tn)``、内核走**半径积分式**"
                          "``2π(R₀n+tn²/2)``，代数恒等而结合次序不同。"
                          "2026-08-18实测最坏相对偏差``1.58e-16``（`concentric_material_length`"
                          "那条同式子六档），取``1e-15``留约6倍余量",
            },
            "relative_deviation": {
                "rel": 1e-5, "abs": 0.0,
                "reason": "**这一格的容差是由条件数定的，不是由实测定的**。"
                          "偏差是两个约``4830mm``的数相减，最细一档差值只有``9.1e-7 mm``——"
                          "而``4830``的一个ulp是``9.1e-13 mm``，于是**任何一次重结合都会把这个"
                          "差搅动约1e-6的相对量**。2026-08-18实测恰为``0.0``"
                          "（内核与本脚本这两式在二进制精确入参下逐位相同，"
                          "因为乘2在浮点上精确），但**拿那个0当容差是自欺**。取``1e-5``。"
                          "真正扛判据的是conformance里那三条关系型判据（逐档下降、"
                          "阶≥1.9、阶单调），它们对这一格的最后几位不敏感",
            },
            "observed_order": {
                "rel": 0.0, "abs": 1e-4,
                "reason": "阶由上一格取``log₂``得出，实测``0.0``（同上，两式逐位相同）。"
                          "容差由上一格推来而不是另立：``log₂``把相对扰动压成绝对扰动"
                          "``Δd/ln2``，两档各1e-5的搅动在阶上最多约``2.9e-5``，取``1e-4``"
                          "留约3.4倍。**两格的容差必须自洽**——上一格1e-5而这一格1e-6"
                          "是一对互相矛盾的声明（第一版就是那样写的）。"
                          "这一格留着是为了让阶的具体数进清单可审，不当主判据用",
            },
        },
    }


def feed_oracle() -> dict:
    segments_fed = [count - 1 for count in FEED_SAMPLES]
    in_span = [min(fed, FEED_FREE_SPAN_SEGMENTS) for fed in segments_fed]
    on_spool = [fed - span for fed, span in zip(segments_fed, in_span)]
    lengths = [count * FEED_REST_LENGTH_MM for count in on_spool]
    #: 由盘上长度反解匝数：``2π(R₀n + t n²/2) = ℓ`` ⟹
    #: ``n = 2S/(R₀ + √(R₀² + 2tS))``，``S = ℓ/2π``。
    turns = []
    radii = []
    for length in lengths:
        #: 由平均直径式反解：``q = ℓ/π = n(2R₀ + tn)`` ⟹
        #: ``n = q / (R₀ + √(R₀² + t·q))``（有理化后的那一支，避免大数相减）。
        #: 与被验路径同为有理化式子但**结合次序不同**（``ℓ/π``对``2·(ℓ/2π)``）。
        quotient = length / math.pi
        root = math.sqrt(
            FEED_BARREL_MM * FEED_BARREL_MM + FEED_THICKNESS_MM * quotient
        )
        count = quotient / (FEED_BARREL_MM + root)
        turns.append(count)
        radii.append(FEED_BARREL_MM + FEED_THICKNESS_MM * count)
    return {
        "id": "oracle:winding/feed_front_material_balance",
        "inputs": {
            "kind": "lagrangian_feed_to_spool",
            "node_budget": FEED_NODE_BUDGET,
            "rest_length_mm": FEED_REST_LENGTH_MM,
            "free_span_segments": FEED_FREE_SPAN_SEGMENTS,
            "barrel_radius_mm": FEED_BARREL_MM,
            "tape_thickness_mm": FEED_THICKNESS_MM,
            "fed_counts": list(FEED_SAMPLES),
        },
        "expected": {
            "segments_fed": segments_fed,
            "segments_in_free_span": in_span,
            "segments_on_spool": on_spool,
            "length_on_spool_mm": lengths,
            "turns_on_spool": turns,
            "radius_on_spool_mm": radii,
        },
        "tolerances": {
            "segments_fed": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**整数，零容差**：``fed_count − 1``。守恒记在段数上而不是"
                          "浮点长度上，正是为了让这三格能判``==``",
            },
            "segments_in_free_span": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**整数，零容差**：``min(已喂段, 跨距段)``。那个``min``是"
                          "落位前的记账（料真的还在跨距里），不是钳位——"
                          "去掉它盘上会出现**负**段而守恒等式照样成立，"
                          "所以conformance另有一条判非负",
            },
            "segments_on_spool": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**整数，零容差**：三格相加是本案例的材料守恒恒等式",
            },
            "length_on_spool_mm": {
                "rel": 0.0, "abs": 0.0,
                "reason": "**零容差**：``段数 × 4.0``，静止段长二进制精确，一次乘法无舍入。"
                          "**换成``0.1``这一格就不再是零容差**——那正是"
                          "『守恒为什么记在整数上』的理由，`tests/test_winding.py`量过："
                          "1392个喂料档里373档浮点长度逐位不守恒",
            },
            "turns_on_spool": {
                "rel": 1e-14, "abs": 0.0,
                "reason": "长度反解匝数：一次除法、一次开方、一次除法。本脚本走"
                          "``ℓ/π``那一路、内核走``2·(ℓ/2π)``那一路，同为有理化式子"
                          "而结合次序不同。2026-08-18实测最坏相对偏差``0.0``"
                          "（七档；除以π与除以2π再乘2在这组入参下给出同一个数）。"
                          "取``1e-14``是**给将来换写法与非精确入参留的**，"
                          "不是今天量出来的余量——**这句话必须写，否则半年后"
                          "有人会以为这里量出过1e-14**",
            },
            "radius_on_spool_mm": {
                "rel": 1e-14, "abs": 0.0,
                "reason": "``R₀ + t·n``，误差随``n``进来。这一格是**S5.4那条"
                          "『匝数与半径生长尚未接在一起』的兑现**："
                          "喂料给长度、长度给匝数、匝数给半径、半径给`drives`要的那个``turns``",
            },
        },
    }


def main() -> int:
    oracles = [radius_oracle(), length_oracle(), order_oracle(), feed_oracle()]
    document = {
        "facet": "engine_oracle_manifest", "facet_version": "0.1",
        "case_id": "case/spool_winding_growth", "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/spool_winding_growth/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles, "arrays": {}, "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
