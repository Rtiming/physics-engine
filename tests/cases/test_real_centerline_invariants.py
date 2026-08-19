"""conformance：真实中心线的几何不变量（`cases/real_centerline_invariants`）。

判据正本是`cases/real_centerline_invariants/oracle.json`，闭式出处在
`cases/real_centerline_invariants/generate_oracle.py`的模块docstring。

**本仓36个案例没有一个读过真实工件，这是第一条。** 于是本文件的重心与别处不同：
**"读进来的数对不对"比"力算得多准"更要紧**——本文件一个力都不算。

三档判据，方向各不相同：

1. **合成圆弧**：闭式不变量被站点差分逐条取回（两种差分取法**必须一致**）；
2. **合成尖峰**：一段之内的扭转，两种取法差**恰好2倍**而积分**恰好相同**
   ——这一档解释了第3档在真语料上的分歧；
3. **真语料（选择进入）**：`PE_REAL_CENTERLINE_CSV`指了才跑，不指明示skip
   （决策0073裁决真实资产永不进仓）。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

from physics_engine.laydown import (
    CenterlineSemantics,
    GrooveCenterline,
    GrooveStation,
    arc_length_fraction_above,
    centerline_invariants,
    hard_way_edge_strain,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/real_centerline_invariants/oracle.json", root=ROOT)
ORACLES = {oracle.id.rsplit("/", 1)[1]: oracle for oracle in MANIFEST.oracles}

#: 真实语料的入口。**决策0073：真实资产永不进仓**，所以这一档走选择进入，
#: 形制抄`tests/test_provenance.py`的`PE_REPLAY_CASE_RUNS`。
REAL_CENTERLINE_ROOT = os.environ.get("PE_REAL_CENTERLINE_CSV")

#: 中心线语义五条。**真实导出是开曲线**：GCW把闭合曲线导成开放折线，
#: 首末之间差恰好一个采样步，而`_require_stations`对`topology="closed"`要求
#: **末站点逐位重复首站点**——真CSV不满足，所以这里一律声明`open`
#: （`tools/model/centerline_csv.py`模块docstring末节写的就是这件事）。
SEMANTICS = CenterlineSemantics(
    position_interpolation="hermite_tangent",
    frame_interpolation="reorthonormalised_linear",
    topology="open",
    out_of_range="clamp_to_end",
    nearest_refinement_iterations=4,
)


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


# ------------------------------------------- 金标一：圆弧＋自转的帧 ---


def _arc_station(arc_mm: float, radius_mm: float, twist_per_mm: float) -> GrooveStation:
    """`generate_oracle.py`第二节那条解析曲线上的一个站点。"""

    angle = arc_mm / radius_mm
    twist = twist_per_mm * arc_mm
    radial = (math.cos(angle), math.sin(angle), 0.0)
    tangent = (-math.sin(angle), math.cos(angle), 0.0)
    normal = tuple(
        math.cos(twist) * radial[axis] + math.sin(twist) * (0.0, 0.0, 1.0)[axis]
        for axis in range(3)
    )
    return GrooveStation(
        arc_length_mm=arc_mm,
        position_mm=(radius_mm * math.cos(angle), radius_mm * math.sin(angle), 0.0),
        tangent=tangent,
        width_direction=_cross(normal, tangent),
        surface_normal=normal,
    )


def _arc_centerline(step_mm: float, radius_mm: float, twist_per_mm: float, span_mm: float):
    count = int(round(span_mm / step_mm)) + 1
    return GrooveCenterline(
        centerline_id="groove/analytic_arc",
        stations=tuple(
            _arc_station(index * step_mm, radius_mm, twist_per_mm) for index in range(count)
        ),
        semantics=SEMANTICS,
        length_unit="mm",
    )


def _probe(centerline, scheme: str, arc_mm: float):
    invariants = centerline_invariants(centerline, scheme=scheme)
    return min(invariants, key=lambda item: abs(item.arc_length_mm - arc_mm))


@pytest.mark.parametrize("scheme", ("forward", "central"))
def test_the_analytic_arc_invariants_come_back(scheme: str) -> None:
    """判据一：闭式的``κ_s``、``κ_n``、``κ_total``、``τ``被站点差分逐条取回。

    **这一档判的是提取器本身**：曲线是手推的，四个不变量各有闭式
    （`generate_oracle.py`第五节第2、3条），差分只是把它们再取一遍。

    **两种差分取法在这条曲线上必须给同一个数**——它的``κ``与``τ``沿弧长是
    平滑的。下一条判据里两者差恰好2倍，**两条一起才有分辨力**：
    只有这一条，一个把`central`写成`forward`的实现照样全绿。
    """

    oracle = ORACLES["analytic_arc_invariants"]
    radius = oracle.inputs["radius_mm"]
    twist = math.radians(oracle.inputs["frame_twist_deg_per_mm"])
    #: 最细的那一档——收敛阶另有一条判据，这里判的是值。
    step = min(oracle.inputs["refinement_steps_mm"])
    centerline = _arc_centerline(step, radius, twist, 80.0)
    measured = _probe(centerline, scheme, oracle.inputs["probe_arc_mm"])

    for field, attribute in (
        ("curvature_total_per_mm", "curvature_total_per_mm"),
        ("curvature_s_per_mm", "curvature_s_per_mm"),
        ("curvature_n_per_mm", "curvature_n_per_mm"),
        ("twist_per_mm", "twist_per_mm"),
    ):
        tolerance = oracle.tolerances[field]
        assert getattr(measured, attribute) == pytest.approx(
            oracle.expected[field], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        ), field

    #: ``κ_s² + κ_n² = κ_total²``——**对"两根轴被对调"有分辨力而单点没有**。
    quadrature = math.hypot(measured.curvature_s_per_mm, measured.curvature_n_per_mm)
    tolerance = oracle.tolerances["curvature_quadrature_per_mm"]
    assert quadrature == pytest.approx(
        oracle.expected["curvature_quadrature_per_mm"],
        rel=tolerance.rel_tol,
        abs=tolerance.abs_tol,
    )

    #: **符号单独判一次**：plans/14报的``τ_max``是绝对值，而带符号的``τ``
    #: 才是`contact.PenaltyGrooveSweepLive`要用的东西（决策0078）。
    assert (measured.twist_per_mm < 0.0) is oracle.expected["twist_is_negative"]

    strain = hard_way_edge_strain((measured,), strip_width_mm=oracle.inputs["strip_width_mm"])
    tolerance = oracle.tolerances["edge_strain_at_probe"]
    assert strain[0] == pytest.approx(
        oracle.expected["edge_strain_at_probe"], rel=tolerance.rel_tol, abs=tolerance.abs_tol
    )


def test_the_two_difference_schemes_agree_on_a_smooth_curve() -> None:
    """**对照组**：曲线平滑时两种取法落在同一个数上。

    没有这一条，下一条判据（两者差恰好2倍）说明不了任何事——
    那个2倍可能只是"其中一种一直是另一种的一半"。
    """

    oracle = ORACLES["analytic_arc_invariants"]
    centerline = _arc_centerline(
        min(oracle.inputs["refinement_steps_mm"]),
        oracle.inputs["radius_mm"],
        math.radians(oracle.inputs["frame_twist_deg_per_mm"]),
        80.0,
    )
    arc = oracle.inputs["probe_arc_mm"]
    forward = _probe(centerline, "forward", arc)
    central = _probe(centerline, "central", arc)
    assert forward.arc_length_mm == central.arc_length_mm
    bound = oracle.expected["scheme_agreement_relative_bound"]
    agree = (
        abs(forward.twist_per_mm - central.twist_per_mm)
        < bound * abs(forward.twist_per_mm)
        and abs(forward.curvature_s_per_mm - central.curvature_s_per_mm)
        < bound * abs(forward.curvature_s_per_mm)
    )
    assert agree is oracle.expected["both_schemes_agree"]
    #: **同一个判据在真语料上必须判False**——那一档差2e-1，
    #: 比这里的1e-4大三个量级。没有这条注脚，`both_schemes_agree`
    #: 看起来像"两种取法总是一样"，而本页整节都在说它们不一样。


@pytest.mark.parametrize("scheme", ("forward", "central"))
def test_the_arc_invariants_converge_second_order(scheme: str) -> None:
    """细化站点表时误差**二阶下降**——判的是"它在收敛"，不是"它碰巧对"。

    `κ_s`那一档的截断项是``O(h²)``（平面圆的``dT/ds``沿径向，
    而三点模板的一阶误差项沿切向、投到``s``上归零——**所以这条曲线上
    单边差分也是二阶**）。**这一条恰恰说明合成曲线区分不了两种取法**，
    真区分在下一条判据上。
    """

    oracle = ORACLES["analytic_arc_invariants"]
    radius = oracle.inputs["radius_mm"]
    twist = math.radians(oracle.inputs["frame_twist_deg_per_mm"])
    arc = oracle.inputs["probe_arc_mm"]
    truth = oracle.expected["curvature_s_per_mm"]

    errors = []
    for step in oracle.inputs["refinement_steps_mm"]:
        measured = _probe(_arc_centerline(step, radius, twist, 80.0), scheme, arc)
        errors.append(abs(measured.curvature_s_per_mm - truth))
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    assert all(3.8 < ratio < 4.2 for ratio in ratios), (scheme, ratios, errors)
    assert errors[-1] < 1.0e-8, errors


def _nonuniform_arc_centerline(
    radius_mm: float, twist_per_mm: float, scale: float = 1.0, span_mm: float = 40.0
):
    """同一条解析圆弧，**采样步在1.0与3.0之间交替**。

    补的是一道注错实测出来的空门（见`test_a_nonuniform_sampling_is_weighted_by_arc`）：
    GCW那批导出采样是**均匀**的2 mm，本页此前的合成语料也是均匀的，
    于是`centerline_invariants`的非均匀三点系数与
    `arc_length_fraction_above`的按弧长权**两条都没有任何门看着**。
    **"采样均匀"是那一批语料的性质，不是这两个函数的前提。**
    """

    arcs = [0.0]
    while arcs[-1] < span_mm:
        arcs.append(arcs[-1] + (1.0 if len(arcs) % 2 == 1 else 3.0) * scale)
    return GrooveCenterline(
        centerline_id="groove/nonuniform_arc",
        stations=tuple(_arc_station(arc, radius_mm, twist_per_mm) for arc in arcs),
        semantics=SEMANTICS,
        length_unit="mm",
    )


def test_a_nonuniform_sampling_is_weighted_by_arc() -> None:
    """采样非均匀时**两件事**都要对：中心差分的系数，与占比的权。

    2026-08-18注错实测，**这两条各是一道空门**：

    | 改坏什么 | 当时的结果 |
    |---|---|
    | `centerline_invariants`的`central`换成均匀网格系数`(f[i+1]−f[i−1])/(2h)` | **0红** |
    | `arc_length_fraction_above`改成按站点数权 | **0红** |

    两条空门是**同一个病根**：仓内合成语料与GCW真语料**都是均匀采样**，
    而均匀网格上两种写法恰好相同。**门全绿不是因为它挡得住，
    是因为那条分支从没被执行过**——0075第五节那道空门的同一个形态。

    本条用1.0/3.0交替的采样步把两条同时逼出来。
    """

    oracle = ORACLES["analytic_arc_invariants"]
    radius = oracle.inputs["radius_mm"]
    twist = math.radians(oracle.inputs["frame_twist_deg_per_mm"])
    centerline = _nonuniform_arc_centerline(radius, twist)

    #: 第一件：非均匀三点系数。曲率仍然是闭式的``sin(τa)/R``，与采样无关。
    invariants = centerline_invariants(centerline, scheme="central")
    worst = 0.0
    for item in invariants:
        expected = math.sin(twist * item.arc_length_mm) / radius
        worst = max(worst, abs(item.curvature_s_per_mm - expected))
    #: ``h = 3 mm``那一半的截断项是``O(h²)``：``(3/145.6)² ≈ 4.3e-4``乘``1/R``。
    #: **换成均匀网格系数会差一个``O(h)``的量**，比这个界大两个数量级。
    assert worst < 3.0e-6, worst

    #: 第二件：占比按**弧长**权。构造一个"只有短步那一半超阈值"的取值序列——
    #: 按站点数权会给≈1/2，按弧长权给≈1/4（短步1.0、长步3.0）。
    #: **奇数下标**：`central`档从站点1起算，而站点1之后是长步（3.0）。
    #: 取奇数下标才落在短步（1.0）那一半上，于是按弧长权给1/4、按站点数权给1/2。
    values = [
        1.0 if index % 2 == 1 else 0.0 for index in range(len(invariants))
    ]
    fraction = arc_length_fraction_above(
        centerline, invariants, values, threshold=0.5
    )
    by_station = sum(1 for value in values if value > 0.5) / len(values)
    assert by_station == pytest.approx(0.5, abs=0.03)
    #: 名义值是``1/(1+3) = 0.25``；容差留到0.04是因为**两端各有一个半段**
    #: （`central`档从站点1起算、末站点没有后邻），实测0.225。
    #: **承重的不是这个数而是下面那条"两者差得开"**。
    assert fraction == pytest.approx(0.25, abs=0.04), fraction
    #: **两个数必须差得开**，否则这条门在这份语料上仍然是空的。
    assert abs(fraction - by_station) > 0.2


def test_the_nonuniform_three_point_weights_stay_second_order() -> None:
    """非均匀网格上细化，``τ``的误差**比值恒为4.00**——第三道空门，最难堵的一道。

    2026-08-18注错实测：把`centerline_invariants`的非均匀三点系数换成
    均匀网格那一组``(f[i+1] − f[i−1]) / (back + fore)``，
    **上一条门也没红**。病根不是采样均匀（上一条已经把它改成非均匀了），
    是**那个错写法在这一档语料上只差2.3倍绝对误差**——
    一条按绝对值判的门要么松得放它过去、要么紧得把正确写法也判红。

    **改判收敛阶就干净了**：正确的三点系数在非均匀网格上仍是**二阶**，
    而均匀系数掉成**一阶**（误差项正比于``(fore − back)/2``，细化时只线性缩小）。
    实测：

    | 细化 | 正确 | 均匀系数（错） |
    |---|---:|---:|
    | 1.0 → 0.5 | **4.00** | 3.38 |
    | 0.5 → 0.25 | **4.00** | 3.06 |
    | 0.25 → 0.125 | **4.00** | 2.72 |

    错写法的比值**一路往2掉**——那是一阶的招牌。
    **判``τ``而不判``κ_s``**：`κ_s`那一档两种写法都给4.00
    （误差的一阶项沿切向、投到``s``上归零，与`test_the_arc_invariants_converge_second_order`
    那条注脚同源）——**只判`κ_s`这道门仍然是空的**。
    """

    oracle = ORACLES["analytic_arc_invariants"]
    radius = oracle.inputs["radius_mm"]
    twist = math.radians(oracle.inputs["frame_twist_deg_per_mm"])

    errors = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        invariants = centerline_invariants(
            _nonuniform_arc_centerline(radius, twist, scale=scale), scheme="central"
        )
        errors.append(max(abs(item.twist_per_mm + twist) for item in invariants))
    ratios = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
    assert all(3.9 < ratio < 4.1 for ratio in ratios), (ratios, errors)
    assert errors[-1] < 1.0e-6, errors


# --------------------------------------------- 金标二：一段之内的尖峰 ---


def _spike_centerline(step_mm: float, angle_rad: float, count: int, spike: int):
    """直链，帧只在第``spike``段里绕切向转过``angle_rad``，其余步一动不动。

    **站点表就是数据**——与真CSV同一形制。本函数不构造任何"底下的连续曲线"，
    因为提取器判的正是站点，不是我们替它编的那条曲线。
    """

    tangent = (0.0, 1.0, 0.0)
    stations = []
    for index in range(count):
        turned = angle_rad if index > spike else 0.0
        normal = (
            math.sin(turned) * -1.0,
            0.0,
            math.cos(turned),
        )
        stations.append(
            GrooveStation(
                arc_length_mm=index * step_mm,
                position_mm=(0.0, index * step_mm, 0.0),
                tangent=tangent,
                width_direction=_cross(normal, tangent),
                surface_normal=normal,
            )
        )
    return GrooveCenterline(
        centerline_id="groove/spike_chain",
        stations=tuple(stations),
        semantics=SEMANTICS,
        length_unit="mm",
    )


def test_a_one_segment_twist_spike_splits_exactly_in_half() -> None:
    """判据二：**峰值差恰好2倍、积分恰好相同**——本页最要紧的一条。

    它是`generate_oracle.py`第五节第4条那个代数事实的执行面，
    也是第三档判据在真语料上那个19%—25%分歧的**解释**：
    那些峰只占一个采样步，`central`把它摊到两站上，各拿一半。

    **峰值统计依赖差分取法，积分统计不依赖。** 于是plans/14那张表里的
    ``τ_max``是"这个几何在2 mm采样下的峰"，而``∫|τ|ds``是曲线的性质。
    """

    oracle = ORACLES["one_segment_twist_spike"]
    step = oracle.inputs["step_mm"]
    centerline = _spike_centerline(
        step,
        oracle.inputs["spike_angle_rad"],
        oracle.inputs["station_count"],
        oracle.inputs["spike_index"],
    )
    forward = centerline_invariants(centerline, scheme="forward")
    central = centerline_invariants(centerline, scheme="central")

    forward_peak = max(abs(item.twist_per_mm) for item in forward)
    central_peak = max(abs(item.twist_per_mm) for item in central)
    for name, value in (
        ("forward_twist_peak_per_mm", forward_peak),
        ("central_twist_peak_per_mm", central_peak),
    ):
        tolerance = oracle.tolerances[name]
        assert value == pytest.approx(
            oracle.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        ), name

    #: **零容差**：``sin Δ``在比值里整个约掉，这不是数值巧合而是代数事实。
    assert central_peak / forward_peak == oracle.expected["peak_ratio_central_over_forward"]

    forward_integral = sum(abs(item.twist_per_mm) for item in forward) * step
    central_integral = sum(abs(item.twist_per_mm) for item in central) * step
    for name, value in (
        ("forward_twist_integral_rad", forward_integral),
        ("central_twist_integral_rad", central_integral),
    ):
        tolerance = oracle.tolerances[name]
        assert value == pytest.approx(
            oracle.expected[name], rel=tolerance.rel_tol, abs=tolerance.abs_tol
        ), name

    #: **摊到了哪里**——只判峰值比是判不出这件事的：一个把`central`整体
    #: 乘0.5的实现峰值比照样是0.5，而它的非零站点仍然只有一个。
    threshold = 0.5 * central_peak
    spread = [item for item in central if abs(item.twist_per_mm) > threshold]
    assert (len(spread) == 2) is oracle.expected["central_spreads_over_two_stations"]
    assert len([item for item in forward if abs(item.twist_per_mm) > threshold]) == 1


# ------------------------------------ 金标三：真语料（**选择进入**） ---


def _real_centerlines():
    """`PE_REAL_CENTERLINE_CSV`指向的那份（或那个目录下的全部）导出。

    **按内在几何归并**：`handoff_runs`里三个run名各有两个时间戳而CSV逐字节相同，
    按"站点数＋总弧长"去重之后只剩5份——这个数本身是一条判据。
    """

    sys.path.insert(0, str(ROOT / "tools"))
    from model.centerline_csv import read_stations  # noqa: PLC0415

    root = Path(REAL_CENTERLINE_ROOT)
    #: 目录与单文件两种形态都吃。**2026-08-18与`tests/test_model_tools.py`那一侧对齐**：
    #: 此前两边对同一个变量各有一套约定，于是**不存在一个值能让选择进入档全部跑过**——
    #: 指目录那一侧硬错、指单文件这一侧硬错，两边都不是skip而是红，
    #: 所以这条通道从来没有被整体跑过一次。
    paths = (
        sorted(root.rglob("centerline.csv")) if root.is_dir() else [root]
    )
    seen: dict[tuple[int, float], GrooveCenterline] = {}
    for path in paths:
        rows = read_stations(path)
        key = (len(rows), round(rows[-1][0] - rows[0][0], 4))
        if key in seen:
            continue
        seen[key] = GrooveCenterline(
            centerline_id="groove/real_export",
            stations=tuple(GrooveStation(*row) for row in rows),
            semantics=SEMANTICS,
            length_unit="mm",
        )
    return seen


@pytest.mark.skipif(
    REAL_CENTERLINE_ROOT is None,
    reason="set PE_REAL_CENTERLINE_CSV to a GCW centerline.csv (or a directory of them) — "
    "真实资产永不进仓（决策0073），所以这一档走选择进入而不是silently pass",
)
def test_the_real_exports_reproduce_the_plans14_table() -> None:
    """判据三：真中心线读进来的曲率/扭率与plans/14第2.2节那张表**逐档对上**。

    plans/15第2.1条原文的判据。**两条对不上的如实登记在案例页第四节**：

    1. ``τ_max``那两格疑似在原表里**对调**了（`v1-coil-1`实测6.5686而表里6.648，
       `v2-coil-02`实测6.6467而表里6.568——**互换之后两行都对到四位有效数字**）；
    2. GCW的`handoff_runs`里按内在几何只有**5份**不同，而plans/14报9个几何。
       另外四个（`test`、`clean_a`、`v1-coil-3`、`v1-coil-2`）**本页够不到**。
    """

    oracle = ORACLES["plans14_table_row"]
    rows = oracle.inputs["rows_by_arc_length_mm"]
    scheme = oracle.inputs["scheme"]
    width = oracle.inputs["strip_width_mm"]
    r_tol = oracle.expected["r_min_relative_tolerance"]
    t_tol = oracle.expected["twist_max_relative_tolerance"]
    e_tol = oracle.expected["edge_strain_p100_relative_tolerance"]
    f_tol = oracle.expected["arc_fraction_absolute_tolerance"]

    centerlines = _real_centerlines()
    #: **这一条判的是整批语料，不是某一份导出**：按内在几何归并后恰好5份，
    #: 而那个数本身就是判据之一。给它一份单独的CSV它判不了——
    #: **明示skip，不硬错**（2026-08-18修）。
    #:
    #: 此前它是硬错，于是`PE_REAL_CENTERLINE_CSV`这个变量出现了两套互不兼容的约定：
    #: `tests/test_model_tools.py`那一侧要单文件、本条要整批，
    #: **不存在一个值能让选择进入档全部跑过**，所以这条通道从来没被整体跑过一次。
    if len(centerlines) < oracle.expected["geometries_matched"]:
        pytest.skip(
            f"这一条要整批语料：按内在几何归并后要{oracle.expected['geometries_matched']}份，"
            f"当前`PE_REAL_CENTERLINE_CSV`只解析出{len(centerlines)}份。"
            "把它指向GCW的`handoff_runs`那一级目录（而不是其中一份CSV）本条才会执行。"
        )
    assert len(centerlines) == oracle.expected["geometries_matched"], sorted(centerlines)

    matched = set()
    disagreeing: dict[str, float] = {}
    for centerline in centerlines.values():
        total = centerline.total_arc_length_mm()
        key = min(rows, key=lambda text: abs(float(text) - total))
        assert abs(float(key) - total) < 0.01, (key, total)
        matched.add(key)
        published = rows[key]

        invariants = centerline_invariants(centerline, scheme=scheme)
        strain = hard_way_edge_strain(invariants, strip_width_mm=width)

        r_min = 1.0 / max(item.curvature_total_per_mm for item in invariants)
        twist_max = math.degrees(max(abs(item.twist_per_mm) for item in invariants))
        assert r_min == pytest.approx(published["r_min_mm"], rel=r_tol), (key, r_min)
        assert twist_max == pytest.approx(
            published["twist_max_deg_per_mm"], rel=t_tol
        ), (key, twist_max)
        #: 与**原表印的那个数**再比一次。对不上的登记下来，不当场红——
        #: 红在下面那两条整数/布尔判据上，**它们说得比一条AssertionError清楚**。
        if twist_max != pytest.approx(
            published["published_twist_max_deg_per_mm"], rel=t_tol
        ):
            disagreeing[key] = twist_max
        assert max(strain) == pytest.approx(
            published["edge_strain_p100"], rel=e_tol
        ), (key, max(strain))
        for threshold, field in ((0.006, "above_006"), (0.004, "above_004")):
            fraction = arc_length_fraction_above(
                centerline, invariants, strain, threshold=threshold
            )
            assert fraction == pytest.approx(published[field], abs=f_tol), (key, field, fraction)

    assert matched == set(rows), sorted(set(rows) - matched)
    assert 9 - len(rows) == oracle.expected["published_rows_without_a_file"]

    #: **恰好两行对不上原表**——写成整数而不是散文。
    assert len(disagreeing) == oracle.expected[
        "rows_disagreeing_with_the_published_twist"
    ], {key: (value, rows[key]["published_twist_max_deg_per_mm"])
        for key, value in disagreeing.items()}

    #: 而那两行**互换之后就对上了**。这一条把"原表对调了"与"我们算错了"分开：
    #: **算错不会让两个错数恰好互为对方。**
    left, right = sorted(disagreeing)
    swapped = (
        disagreeing[left]
        == pytest.approx(rows[right]["published_twist_max_deg_per_mm"], rel=t_tol)
        and disagreeing[right]
        == pytest.approx(rows[left]["published_twist_max_deg_per_mm"], rel=t_tol)
    )
    assert swapped is oracle.expected["the_two_disagreeing_rows_swap_into_each_other"], (
        {key: disagreeing[key] for key in (left, right)},
        {key: rows[key]["published_twist_max_deg_per_mm"] for key in (left, right)},
    )


@pytest.mark.skipif(
    REAL_CENTERLINE_ROOT is None,
    reason="set PE_REAL_CENTERLINE_CSV — 见上一条",
)
def test_the_two_schemes_disagree_on_the_real_exports() -> None:
    """真语料上两种取法**必须差得开**——金标二那个2倍在真数据上的回声。

    实测：``τ_max``差19%—25%，而``∫|τ|ds``差 < 1%。
    **这不是哪一种错了，是那些峰在2 mm采样下没有收敛**——
    于是plans/14那张表里的``τ_max``是"这个几何在这个采样密度下的峰"，
    换一次重采样它就变。**本条判的是这件事本身，不裁哪一种对。**
    """

    for centerline in _real_centerlines().values():
        forward = centerline_invariants(centerline, scheme="forward")
        central = centerline_invariants(centerline, scheme="central")
        forward_peak = max(abs(item.twist_per_mm) for item in forward)
        central_peak = max(abs(item.twist_per_mm) for item in central)
        assert 0.70 < central_peak / forward_peak < 0.85, (
            centerline.total_arc_length_mm(),
            central_peak / forward_peak,
        )

        def integral(items):
            arcs = [item.arc_length_mm for item in items]
            spans = [arcs[index + 1] - arcs[index] for index in range(len(arcs) - 1)]
            spans.append(spans[-1])
            return sum(
                abs(item.twist_per_mm) * span
                for item, span in zip(items, spans, strict=True)
            )

        assert integral(central) == pytest.approx(integral(forward), rel=0.02)


# --------------------------------------------------------- 必红矩阵 ---


def test_the_extractor_refuses_an_undeclared_scheme() -> None:
    """差分取法**没有默认值**，没声明或声明了别的当场拒。

    理由与`CenterlineSemantics`那五条同源，而且更硬：本页判据二实测
    **取哪一种改变答案（峰值差2倍）**。默认值等于替声明者拿了主意。
    """

    from physics_engine.laydown import LaydownError  # noqa: PLC0415

    centerline = _arc_centerline(1.0, 145.6, 0.0445, 40.0)
    for bad in ("backward", "", "Central", "central_difference"):
        with pytest.raises(LaydownError):
            centerline_invariants(centerline, scheme=bad)


def test_the_strain_helper_refuses_a_width_it_was_not_given() -> None:
    """``strip_width_mm``没有默认值——plans/14那张表用4.0，而工件自己的槽宽是8/10 mm。

    **换一个宽度换一个结论**（同节第3条实测：超标占比从0%—5.9%变成13.4%—23.3%），
    所以它是一条声明。
    """

    from physics_engine.laydown import LaydownError  # noqa: PLC0415

    centerline = _arc_centerline(1.0, 145.6, 0.0445, 40.0)
    invariants = centerline_invariants(centerline, scheme="forward")
    for bad in (0.0, -4.0, float("nan")):
        with pytest.raises(LaydownError, match="strip_width_mm"):
            hard_way_edge_strain(invariants, strip_width_mm=bad)


def test_the_arc_fraction_refuses_mismatched_lengths() -> None:
    """站点数与取值数对不上当场拒——错位一格会安静地把占比算在别的段上。"""

    from physics_engine.laydown import LaydownError  # noqa: PLC0415

    centerline = _arc_centerline(1.0, 145.6, 0.0445, 40.0)
    invariants = centerline_invariants(centerline, scheme="forward")
    with pytest.raises(LaydownError, match="长度不一致"):
        arc_length_fraction_above(
            centerline, invariants, [0.0] * (len(invariants) - 1), threshold=0.0
        )
