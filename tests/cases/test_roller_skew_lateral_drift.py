"""conformance：导轮轴偏斜的稳态横漂（`cases/roller_skew_lateral_drift`）。

**这一条回答的是"几个轮子不在一个平面时带材偏多少"，而且它不需要材料输运。**

WDS `research/05`第三节把材料注入列为横漂定量化的最大缺口。那条对**瞬态**
（τ≈3 s）与**持续横走**成立，对**稳态**不成立——Shelton正规入轮定律
``y'(L) = θ_r``**本身就是输运的结果**，把它当边界条件写下来，
稳态就是一个静力边值问题。

## 引擎侧怎么解那个"斜率给定、位置待求"的边界条件

`solve_equilibrium`只有`fixed_indices`一种边界条件，钉不了"斜率"。
但小挠度下问题是**线性**的，于是：**钉尖端位置``A``、解、读出尖端斜率``s(A)``**，
再按线性缩放``y_ss = θ_r·A/s(A)``。一次求解，无需求根。

**线性区不是假定，是判出来的**：本文件有一条门扫振幅——
``A ≤ 0.01 mm``的偏差在求解器残差量级，``A ≥ 0.1 mm``起是**干净的二次律**
（系数约``9.9e-3 /mm²``），``A = 1.0 mm``时偏**+0.9%**。
那0.9%是**横漂本身把张力抬起来**的二阶效应（额外弧长``∫y'²/2``）——
独立验算：`ΔT ≈ +6.7%` ⟹ `u` 涨3.35% ⟹ `y_ss` 涨0.94%，**与实测对上**。
**闭式看不见这一项而引擎算得出。**

## 一条反直觉但要紧的结论

``f(u)``随``u = L·sqrt(T/EI₁)``**单调递增**（2/3 → 1），所以

> **提高张力不会减小稳态横漂，反而略微增大它。**

实测T从10 N到40 N，``y_ss``从2.491到2.774 mm（**+11.3%**）。
**横漂不能靠加张力压下去**——这条对张力算法直接相关。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.energies import (
    AxialStretch,
    DiscreteElasticBending,
    EnergyContext,
    EnergyRegistry,
    clamped_chain_bending_vertices,
)
from physics_engine.oracles import load_manifest
from physics_engine.solve import solve_equilibrium
from physics_engine.state import StateField, StateLayout

CASE = Path(__file__).resolve().parents[2] / "cases" / "roller_skew_lateral_drift"
MANIFEST = load_manifest(CASE / "oracle.json")

YOUNGS_MODULUS_N_MM2 = 150.0e3
THICKNESS_MM = 0.1
WIDTH_MM = 4.0
BENDING_STIFFNESS_NMM2 = YOUNGS_MODULUS_N_MM2 * THICKNESS_MM * WIDTH_MM**3 / 12.0
AXIAL_STIFFNESS_N = YOUNGS_MODULUS_N_MM2 * THICKNESS_MM * WIDTH_MM

FREE_SPAN_MM = 200.0
TENSION_N = 20.0
SKEW_RAD = math.radians(1.0)
#: 线性区的振幅：由`test_the_linear_regime_has_a_measured_amplitude_boundary`判出来的。
LINEAR_TIP_MM = 1.0e-3


def _oracle(oracle_id: str):
    for entry in MANIFEST.oracles:
        if entry.id == oracle_id:
            return entry
    raise AssertionError(f"清单里没有{oracle_id}")


def _layout(nodes: int) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/drift{nodes}",
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


def _slope_per_unit_tip(segments: int, tip_mm: float, tension_n: float = TENSION_N) -> float:
    """钉住尖端横向位置``tip_mm``求解，返回**单位尖端的**尖端斜率``s/A``。

    自由度安排：``x``与``y``全钉，只留``z``当横向——那是梁-弦的线性化口径。
    上游固支由钉住节点0与节点1的``z``实现（``y(0)=0``与``y'(0)=0``两条）。
    尖端不设弯曲模板，故``y''(L)=0``是自然边界条件，**不需要额外施加**。
    """

    nodes = segments + 1
    pitch = FREE_SPAN_MM / segments
    #: 静止段长按预张反解：``(EA/l0)(h − l0) = T``。于是初始直构型上张力恰为``T``。
    rest = pitch / (1.0 + tension_n / AXIAL_STIFFNESS_N)
    positions: list[float] = []
    for index in range(nodes):
        guess = tip_mm * (index / segments) ** 2 if index >= 2 else 0.0
        positions += [index * pitch, 0.0, guess]
    positions[3 * segments + 2] = tip_mm

    registry = EnergyRegistry(
        terms=(
            AxialStretch(
                edges=tuple((i, i + 1, rest, AXIAL_STIFFNESS_N) for i in range(segments))
            ),
            DiscreteElasticBending(
                vertices=clamped_chain_bending_vertices(
                    nodes, pitch, BENDING_STIFFNESS_NMM2
                )
            ),
        )
    )
    fixed = {index for node in range(nodes) for index in (3 * node, 3 * node + 1)}
    fixed |= {2, 5, 3 * segments + 2}
    result = solve_equilibrium(
        registry,
        EnergyContext(
            context_id="context/drift",
            node_masses_kg=(1.0e-9,) * nodes,
            gravity_mm_s2=(0.0, 0.0, 0.0),
        ),
        _layout(nodes),
        tuple(positions),
        fixed_indices=frozenset(fixed),
        residual_tol_n=1.0e-9,
        max_iterations=300,
    )
    assert result.converged, result.reason
    tip = result.state.vector[3 * segments + 2]
    previous = result.state.vector[3 * (segments - 1) + 2]
    return ((tip - previous) / pitch) / tip_mm


def _engine_drift_mm(segments: int, tension_n: float = TENSION_N) -> float:
    return SKEW_RAD / _slope_per_unit_tip(segments, LINEAR_TIP_MM, tension_n)


# ---------------------------------------------------------------------------
# 闭式自洽（不碰引擎，秒级）
# ---------------------------------------------------------------------------


def test_the_beam_string_factor_runs_from_two_thirds_to_one():
    """``f(0) = 2/3``（纯梁）、``f(∞) → 1``（纯弦），中间**单调递增**。

    `research/04`第2节明写本工位"处在梁弦之间的中间区，
    **纯弦或纯梁公式均不适用，必须带f(KL)**"——本门是那句话的可判形式。
    """

    entry = _oracle("oracle:drift/beam_string_factor_limits")

    def factor(u: float) -> float:
        if u < 1.0e-6:
            return 2.0 / 3.0 + u * u / 45.0
        return (math.sinh(u) - u * math.cosh(u)) / (u * (1.0 - math.cosh(u)))

    assert factor(0.0) == pytest.approx(entry.expected["pure_beam_limit"], rel=1e-15)
    assert factor(1.6) == pytest.approx(entry.expected["at_1p6"], rel=1e-15)
    assert factor(8.0) == pytest.approx(entry.expected["at_8"], rel=1e-15)
    assert factor(100.0) == pytest.approx(entry.expected["at_100"], rel=1e-15)
    #: 单调：这是"张力越大横漂越大"那条推论的来源。
    samples = [factor(u) for u in (0.0, 0.5, 1.0, 1.6, 3.0, 8.0, 20.0, 100.0)]
    for earlier, later in zip(samples, samples[1:], strict=False):
        assert later > earlier, f"f(u)不单调：{samples}"
    assert samples[-1] < 1.0, "f(u)必须从下方趋近1（纯弦是上界）"


def test_the_closed_form_matches_the_five_numbers_wds_quoted():
    """与WDS `research/04`第2节引的**五个独立数字**对拍。

    它们不是本仓算的，是另一个仓的推导页上写的：
    ``EI₁≈8.0e4``、``1/K≈63mm``、``u≈1.6—8``、``f≈0.68—0.88``、``y_ss≈2.6mm``。
    **五个全对上，才说明本页的``f(u)``推对了。**
    """

    assert BENDING_STIFFNESS_NMM2 == pytest.approx(8.0e4, rel=1e-12)
    stiffness_length = math.sqrt(BENDING_STIFFNESS_NMM2 / TENSION_N)
    assert stiffness_length == pytest.approx(63.0, abs=0.5)

    entry = _oracle("oracle:drift/steady_state_at_one_degree")
    assert entry.expected["u"] == pytest.approx(3.162, rel=1e-3)
    assert entry.expected["factor"] == pytest.approx(0.748, rel=1e-2)
    assert entry.expected["drift_mm"] == pytest.approx(2.6, abs=0.05)

    limits = _oracle("oracle:drift/beam_string_factor_limits")
    for span, expected_u in ((100.0, 1.58), (500.0, 7.91)):
        u = math.sqrt(TENSION_N / BENDING_STIFFNESS_NMM2) * span
        assert u == pytest.approx(expected_u, rel=1e-2)
    assert limits.expected["at_1p6"] == pytest.approx(0.68, abs=0.02)
    assert limits.expected["at_8"] == pytest.approx(0.88, abs=0.02)


def test_raising_the_tension_makes_the_drift_worse_not_better():
    """**反直觉但要紧**：张力越大稳态横漂越大。

    ``f(u)``随``u = L·sqrt(T/EI₁)``单调增——张力越大越接近纯弦，
    弯曲的回正作用越小。实测10 N→40 N，``y_ss``从2.491到2.774 mm（**+11.3%**）。

    **这条对张力算法直接相关：横漂不能靠加张力压下去。**
    如果哪天这条门红了，要么`f(u)`写错了，要么有人把"张力抑制横漂"这个
    直觉写进了模型。
    """

    entry = _oracle("oracle:drift/tension_makes_it_worse_not_better")
    drifts = [entry.expected[f"drift_at_{n}n"] for n in (10, 20, 30, 40)]
    for earlier, later in zip(drifts, drifts[1:], strict=False):
        assert later > earlier, f"张力升高而横漂没有变大：{drifts}"
    assert drifts[-1] / drifts[0] == pytest.approx(1.113, rel=1e-2)


def test_the_critical_skew_that_reaches_the_flange():
    """**装配公差的直接输入**：多少度轮轴偏斜会让带材蹭上法兰。

    半间隙6.5 mm（真机导轮有效宽度17 mm、带宽4 mm，与`winding_line_endtoend`同源）。
    实测：跨长100/200/300/500 mm对应**5.381°/2.489°/1.546°/0.852°**。

    **500 mm跨长时不到1°就蹭上。** 跨长越大越敏感——因为``y_ss ∝ L·f(KL)``
    而``f``还随``L``增（``u = KL``），**是超线性的**。
    """

    entry = _oracle("oracle:drift/skew_that_reaches_the_flange")
    values = [
        entry.expected[f"critical_skew_deg_at_{span}mm"] for span in (100, 200, 300, 500)
    ]
    assert values == pytest.approx([5.381, 2.489, 1.546, 0.852], abs=1e-3)
    for earlier, later in zip(values, values[1:], strict=False):
        assert later < earlier, f"跨长增大而临界偏斜没有变小：{values}"
    #: 超线性：跨长×5而临界角小了**6.3倍**，不是5倍。
    assert values[0] / values[3] == pytest.approx(6.315, rel=1e-2)
    assert values[0] / values[3] > 5.0, "只按`y_ss ∝ L`会给恰好5倍——那说明f(KL)被漏掉了"


# ---------------------------------------------------------------------------
# 引擎对闭式
# ---------------------------------------------------------------------------


@pytest.mark.batch
def test_the_engine_reproduces_the_shelton_drift():
    """引擎的DER弯曲＋轴向张力准静态解对Shelton闭式。"""

    entry = _oracle("oracle:drift/steady_state_at_one_degree")
    measured = _engine_drift_mm(160)
    assert measured == pytest.approx(
        entry.expected["drift_mm"], rel=entry.tolerances["drift_mm"].rel_tol
    )


@pytest.mark.batch
def test_the_engine_converges_at_second_order():
    """**二阶收敛**：N=20/40/80/160实测误差比 3.739/3.861/3.928。

    门判比值落在``[3.2, 4.4]``而**不写死为4**——与`harmonic_oscillator`
    那条"收敛比不写死"同源。
    """

    expected = _oracle("oracle:drift/steady_state_at_one_degree").expected["drift_mm"]
    errors = []
    for segments in (20, 40, 80, 160):
        drift = _engine_drift_mm(segments)
        errors.append(abs(drift - expected) / expected)
    assert errors[0] > errors[1] > errors[2] > errors[3], f"误差没有单调下降：{errors}"
    for earlier, later in zip(errors, errors[1:], strict=False):
        assert 3.2 <= earlier / later <= 4.4, (
            f"收敛阶不是二阶：比值{earlier / later!r}，序列{errors}"
        )


@pytest.mark.batch
def test_the_nonlinearity_is_quadratic_in_amplitude_and_the_mechanism_checks_out():
    """**线性区不是假定，是判出来的；而越出线性区那一项也是物理不是误差。**

    按线性缩放``y_ss = θ_r·A/s(A)``的前提是``s/A``与``A``无关。
    2026-08-17实测（N=80，以``A = 1e-5``为参考）：

    | ``A`` (mm) | 相对偏差 | 偏差/``A²`` |
    |---|---|---|
    | 1e-4 | 2.4e-11 | — |
    | 1e-3 | 1.9e-09 | — |
    | 1e-2 | 1.9e-07 | — |
    | 0.1 | 9.97e-05 | **9.97e-03** |
    | 0.3 | 8.94e-04 | **9.93e-03** |
    | 1.0 | 9.47e-03 | 9.47e-03 |

    ``A ≤ 1e-2``那三档的偏差在求解器残差量级（``1e-9 N``），**不是物理**；
    ``A ≥ 0.1``起是**干净的二次律**，系数约``9.9e-3 /mm²``。

    ## 那一项的机理，独立验算对得上

    横漂让带材多走弧长``∫y'²/2 dx``，张力被抬高。``A = 1 mm``时估得
    ``ΔT ≈ 1.34 N``（20 N的**+6.7%**）；``u ∝ sqrt(T)``故``u``涨**+3.35%**；
    ``u = 3.16``附近``dlnf/dlnu ≈ 0.28``，于是``y_ss``涨约**+0.94%**——
    **与实测的+0.9%对上**。

    **闭式是线性化的，看不见这一项；引擎算得出。**
    方向也对：非线性让横漂**变大**（张力自升⟹更像纯弦⟹``f``更大）。
    """

    reference = _slope_per_unit_tip(80, 1.0e-5)
    #: 线性区：三档振幅的偏差都在求解器噪声量级。
    for tip, bound in ((1.0e-4, 1.0e-9), (1.0e-3, 1.0e-8), (1.0e-2, 1.0e-6)):
        deviation = abs(_slope_per_unit_tip(80, tip) / reference - 1.0)
        assert deviation < bound, f"A={tip}的偏差{deviation:.3e}超出线性区判据{bound:.0e}"

    #: 二次律：两个振幅的``偏差/A²``必须一致（这才是"二次"的证据，单点不是）。
    coefficients = []
    for tip in (0.1, 0.3):
        deviation = abs(_slope_per_unit_tip(80, tip) / reference - 1.0)
        coefficients.append(deviation / (tip * tip))
    assert coefficients[0] == pytest.approx(coefficients[1], rel=0.02), (
        f"偏差不按``A²``走：{coefficients}"
    )
    assert coefficients[0] == pytest.approx(9.9e-3, rel=0.05)

    #: 1 mm处的0.9%，以及它的方向。
    nonlinear = _slope_per_unit_tip(80, 1.0)
    excess = reference / nonlinear - 1.0
    assert excess == pytest.approx(9.5e-3, rel=0.05)
    assert excess > 0.0, "非线性应当让横漂**变大**（张力自升⟹更像纯弦）"


@pytest.mark.batch
def test_the_engine_also_says_more_tension_means_more_drift():
    """闭式那条反直觉结论，**引擎独立复现一次**。

    两条路径互钉：闭式走``f(u)``的单调性，引擎走DER弯曲＋轴向张力的准静态解。
    **一条红另一条不红，说明其中一条错了，而不是说明该调容差。**
    """

    drifts = [_engine_drift_mm(80, tension) for tension in (10.0, 20.0, 30.0, 40.0)]
    for earlier, later in zip(drifts, drifts[1:], strict=False):
        assert later > earlier, f"引擎侧张力升高而横漂没有变大：{drifts}"
    entry = _oracle("oracle:drift/tension_makes_it_worse_not_better")
    closed = [entry.expected[f"drift_at_{n}n"] for n in (10, 20, 30, 40)]
    for measured, reference in zip(drifts, closed, strict=True):
        assert measured == pytest.approx(reference, rel=1.0e-3)
