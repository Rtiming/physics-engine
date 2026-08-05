"""`case/axial_stretch_hessian`的conformance门（轴7规则3）。

**这一条闭合决策0024第六节登记的缺口**：拉伸项的Hessian此前只有有限差分背书。
按spec/12第6.1节，那是一道验不了物理的门——它只验"雅可比是不是我写的那个能量的
导数"，能量本身写错时它照样全绿。本文件里的最后一条测试把这句话做成了实验：
同一个被改错的静止长度，**有限差分门绿、解析金标红**。

判据数与容差全部来自清单（`cases/axial_stretch_hessian/oracle.json`），
本文件一个公式也不复述。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physics_engine.energies import AxialStretch, EnergyContext
from physics_engine.oracles import OracleError, load_manifest
from physics_engine.state import State, StateField, StateLayout

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/axial_stretch_hessian/oracle.json", root=ROOT)


def _layout(nodes: int) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/axial_stretch_hessian_n{nodes}",
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


def _setup(entry, *, edges=None, coordinates=None):
    """按一条oracle的输入装出(能量项, 状态, 上下文)。"""

    coordinates = tuple(entry.inputs["coordinates_mm"] if coordinates is None else coordinates)
    raw_edges = entry.inputs["edges"] if edges is None else edges
    term = AxialStretch(
        edges=tuple((int(i), int(j), float(rest), float(stiffness)) for i, j, rest, stiffness in raw_edges)
    )
    nodes = len(coordinates) // 3
    state = State(layout=_layout(nodes), vector=coordinates)
    context = EnergyContext(
        context_id="context/axial_stretch_hessian",
        node_masses_kg=tuple(1.0 for _ in range(nodes)),
    )
    return term, state, context


def _measured(term, state, context) -> dict:
    return {
        "gradient_n": list(term.gradient(state, context)),
        "hessian_n_mm": [value for row in term.hessian(state, context) for value in row],
    }


@pytest.mark.parametrize("entry", MANIFEST.oracles, ids=lambda entry: entry.id)
def test_derivatives_match_the_independent_exact_oracle(entry):
    """生产内核 对 精确有理算术金标。**金标不知道内核的公式长什么样。**"""

    term, state, context = _setup(entry)
    entry.check_all(_measured(term, state, context))


@pytest.mark.parametrize("entry", MANIFEST.oracles, ids=lambda entry: entry.id)
def test_the_fused_path_reproduces_the_same_derivatives(entry):
    """融合路径也要过同一条金标——spec/12第3.1节的承重条款不许两条路各说各话。"""

    term, state, context = _setup(entry)
    _, gradient, hessian = term.quantities(
        state, context, need_gradient=True, need_hessian=True
    )
    assert gradient is not None and hessian is not None
    entry.check_all({
        "gradient_n": list(gradient),
        "hessian_n_mm": [value for row in hessian for value in row],
    })


# ---------------------------------------------------------------- 必须红 -----
# 轴7规则6：每道物理门要有"它必须红"的输入。下面四条各调错一样东西，
# 门必须红；最后一条同时证明有限差分门在同一个错误上**照样绿**。


def _perturbed_edges(entry, *, rest_factor=1.0, stiffness_factor=1.0):
    return [
        [i, j, rest * rest_factor, stiffness * stiffness_factor]
        for i, j, rest, stiffness in entry.inputs["edges"]
    ]


@pytest.mark.parametrize("entry", MANIFEST.oracles, ids=lambda entry: entry.id)
def test_a_wrong_rest_length_must_redden_the_gate(entry):
    """静止长度错十亿分之一——判据必须红。"""

    term, state, context = _setup(entry, edges=_perturbed_edges(entry, rest_factor=1.0 + 1.0e-9))
    with pytest.raises(OracleError):
        entry.check_all(_measured(term, state, context))


@pytest.mark.parametrize("entry", MANIFEST.oracles, ids=lambda entry: entry.id)
def test_a_wrong_axial_stiffness_must_redden_the_gate(entry):
    """轴向刚度错十亿分之一——判据必须红。"""

    term, state, context = _setup(
        entry, edges=_perturbed_edges(entry, stiffness_factor=1.0 + 1.0e-9)
    )
    with pytest.raises(OracleError):
        entry.check_all(_measured(term, state, context))


@pytest.mark.parametrize("entry", MANIFEST.oracles, ids=lambda entry: entry.id)
def test_a_perturbed_coordinate_must_redden_the_gate(entry):
    """把一个节点挪十亿分之一毫米——判据必须红。"""

    coordinates = list(entry.inputs["coordinates_mm"])
    coordinates[1] += 1.0e-9
    term, state, context = _setup(entry, coordinates=coordinates)
    with pytest.raises(OracleError):
        entry.check_all(_measured(term, state, context))


def _dropped_transverse_edges(entry):
    """构造"漏掉横向项"那个经典错误——**不复述公式，让内核自己算出错版本**。

    把静止长度改成当前长度、刚度按同比例放大，于是``k = EA/l0``不变而伸长量恰为0，
    内核算出来的正好是``k·d⊗d``——也就是把``(k·ε/L)·(I − d⊗d)``整块丢掉的那个
    错误实现。这个错误是"轴向弹簧"的直觉写法，也是这条判据存在的首要理由。
    """

    coordinates = entry.inputs["coordinates_mm"]
    edges = []
    for i, j, rest, stiffness in entry.inputs["edges"]:
        i, j = int(i), int(j)
        length = math.sqrt(
            sum((coordinates[3 * j + a] - coordinates[3 * i + a]) ** 2 for a in range(3))
        )
        edges.append([i, j, length, stiffness * length / rest])
    return edges


@pytest.mark.parametrize(
    "entry",
    [entry for entry in MANIFEST.oracles if not entry.id.endswith("/at_rest")],
    ids=lambda entry: entry.id,
)
def test_dropping_the_transverse_term_must_redden_the_gate(entry):
    """漏掉``(k·ε/L)·(I − d⊗d)``——**Hessian那条判据**必须红。

    **只比Hessian，不比梯度**：这个错版本的伸长量恰为0，梯度也跟着变成零矢量，
    ``check_all``会先在梯度上红——那证明不了Hessian这条判据有没有牙。
    单比`hessian_n_mm`才是这条必须红要证的事。

    ``at_rest``构型除外：那里伸长量本就是0、横向项本就不存在，
    "漏掉它"与正确实现是同一个矩阵，**它红不了也不该红**。
    如实排除比让它假装通过诚实（这条写进案例页第四节）。
    """

    term, state, context = _setup(entry, edges=_dropped_transverse_edges(entry))
    with pytest.raises(OracleError):
        entry.check("hessian_n_mm", _measured(term, state, context)["hessian_n_mm"])


def test_a_wrong_rest_length_is_invisible_to_finite_difference_but_red_here():
    """**spec/12第6.1节那句话的实验版**：同一个错误，FD绿、解析金标红。

    有限差分验的是"这个雅可比是不是这个能量的导数"。把静止长度改错之后，
    能量、梯度、Hessian**仍然互相自洽**——它们只是同一个错能量的导数，
    所以FD一点异样也看不见。看得见的是知道正确答案的那条独立路径。
    """

    entry = MANIFEST.oracle("oracle:axial_stretch_hessian/near_rest_ratio_200")
    term, state, context = _setup(entry, edges=_perturbed_edges(entry, rest_factor=1.0 + 1.0e-9))

    # 1. 解析金标：红。
    with pytest.raises(OracleError):
        entry.check_all(_measured(term, state, context))

    # 2. 有限差分：绿（口径与`tests/cases/test_two_body_spring.py`那条同款）。
    vector = state.vector
    gradient = term.gradient(state, context)
    hessian = term.hessian(state, context)
    step = 1.0e-6
    for index in range(len(vector)):
        plus, minus = list(vector), list(vector)
        plus[index] += step
        minus[index] -= step
        state_plus = State(layout=state.layout, vector=tuple(plus))
        state_minus = State(layout=state.layout, vector=tuple(minus))
        numerical = (
            term.energy(state_plus, context) - term.energy(state_minus, context)
        ) / (2 * step)
        assert abs(numerical - gradient[index]) <= 1e-6 * max(abs(gradient[index]), 1.0)
        gradient_plus = term.gradient(state_plus, context)
        gradient_minus = term.gradient(state_minus, context)
        for column in range(len(vector)):
            numerical_second = (gradient_plus[column] - gradient_minus[column]) / (2 * step)
            assert abs(numerical_second - hessian[index][column]) <= 1e-5 * max(
                abs(hessian[index][column]), 1.0
            )
