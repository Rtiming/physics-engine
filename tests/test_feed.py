"""喂料前沿的门（决策0062丙1，能力位S5.4）。

## 承重门只有一条

`test_the_layout_never_changes_while_the_tape_feeds`：**喂料全程布局定长、
指纹长度不变**。它是这一片存在的全部理由——S5.4的`missing`原文就是
"自由度数在过程中变"，而0050的承重条款要求它**不变**。

## 第二条门守的是"零贡献是构造出来的"

未喂的边不是被``if``跳过的，是两端恰好相距静止段长、伸长量恰为零。
跳过会让`EnergyRegistry`的求和次序随``fed_count``变，
而**求和次序是形制**（spec/12第3.3节）。判据：
**同一构型下把未喂段数从1改到10，已喂那一段的能量与梯度逐字节不变**。

## 必红矩阵（2026-08-17逐条注错**实测**）

| 注错 | 红掉 |
|---|---|
| 未喂节点停在原点（而不是喂料口后方） | 5 |
| 已喂材料长度写成`fed_count·rest`（少减1） | 4 |
| `parked_fixed_indices`漏掉一个分量 | 4 |
| 停放间隔不等于静止段长（0.5倍） | 2 |
| 未喂边被跳过而不是构造零贡献 | 2 |
| 未喂节点停在喂料口**前**方（与已喂段重叠） | 1 |
| 布局与预算的一致性校验去掉 | 1 |

**七条全被抓到，最低一条。**

## 注错测法自己的一条坑（2026-08-17实测）

`range(3)` → `range(2)` 是**同字节数**的改动。CPython的`.pyc`失效判据是
mtime＋size，**两者都没变**时缓存被当成新鲜的——于是还原源文件之后测试
**照样红**，看起来像"还原失败"。

**同长度的变异会留下一份被当成新鲜的旧缓存。** 注错脚本必须每轮清
`__pycache__`，否则红的条数可能是上一个变异体的。本文件这七条在清缓存
前后数字一致（其余变异体都改了字节数），但那是运气。
"""

from __future__ import annotations

import math

import pytest

from physics_engine.energies import (
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    PointLoad,
    UniformGravity,
)
from physics_engine.feed import FeedError, FeedFront, assert_layout_matches_budget
from physics_engine.solve import solve_equilibrium
from physics_engine.state import State, StateField, StateLayout

NODE_BUDGET = 12
REST_MM = 10.0
EA_N = 60000.0
INLET = (0.0, 0.0, 0.0)
DIRECTION = (1.0, 0.0, 0.0)


def _front(**overrides) -> FeedFront:
    base = {
        "node_budget": NODE_BUDGET,
        "rest_length_mm": REST_MM,
        "inlet_mm": INLET,
        "direction": DIRECTION,
    }
    return FeedFront(**{**base, **overrides})


def _layout(nodes: int = NODE_BUDGET) -> StateLayout:
    return StateLayout(
        layout_id=f"layout/feed{nodes}",
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


CONTEXT = EnergyContext(
    context_id="context/feed",
    node_masses_kg=(1.0e-9,) * NODE_BUDGET,
    gravity_mm_s2=(0.0, 0.0, 0.0),
)


def _fed_positions(fed_count: int, stretch_mm: float = 0.0) -> tuple[float, ...]:
    """已喂那一段：**节点0是引出端（下游），节点``fed_count−1``在喂料口**。

    次序要紧：拉格朗日模型里每个节点始终是同一块材料，**先喂进来的走得更远**。
    把它排反（节点0在喂料口、往下游长）会让接缝那条边横跨整段——
    2026-08-17第一版就是那样，能量从7.5爆到303015。
    """

    pitch = REST_MM + stretch_mm
    return tuple(
        component
        for index in range(fed_count)
        for component in (
            INLET[0] + DIRECTION[0] * pitch * (fed_count - 1 - index),
            INLET[1] + DIRECTION[1] * pitch * (fed_count - 1 - index),
            INLET[2] + DIRECTION[2] * pitch * (fed_count - 1 - index),
        )
    )


# ---------------------------------------------------------------------------
# 承重门：布局定长
# ---------------------------------------------------------------------------


def test_the_layout_never_changes_while_the_tape_feeds():
    """**S5.4的`missing`说"自由度数在过程中变"，而这里它不变。**

    喂料全程布局的``dof_count``、字段数、以及状态向量的长度都恒定。
    这是0050那条承重条款（指纹跨步不变）在喂料上的兑现：
    **节点数不由"已经喂进来多少"决定，由"这一卷总共要喂多少"决定**，
    活动与否变成向量里的一个值。
    """

    front = _front()
    layout = _layout()
    assert_layout_matches_budget(layout, front)
    lengths = set()
    for fed_count in range(2, NODE_BUDGET + 1):
        positions = front.initial_positions_mm(_fed_positions(fed_count))
        assert len(positions) == 3 * NODE_BUDGET
        state = State(layout=layout, vector=positions)
        lengths.add(len(state.vector))
        assert layout.dof_count == 3 * NODE_BUDGET
    assert lengths == {3 * NODE_BUDGET}, f"向量长度在喂料过程中变过：{lengths}"


def test_the_unfed_segments_contribute_exactly_zero_by_construction():
    """未喂边的贡献**逐字节为零**——不是"很小"，也不是被``if``跳过的。

    判据：**全预算的`AxialStretch`与只含已喂边的那个，能量与梯度逐字节相同**。
    多出来的那些边各加一个恰好的``0.0``，而浮点加零是精确的。
    跳过会让求和次序随``fed_count``变，而**求和次序是形制**（spec/12第3.3节）。

    **不判"不同`fed_count`之间逐字节相同"**：喂料口在空间里固定、引出端随喂料
    前移，所以同一条材料边在不同`fed_count`下落在**不同的绝对坐标**上，
    差一个1e-13量级的舍入。那是几何的、正确的，判它会把一个正确实现判红——
    2026-08-17第一版就那么写过。
    """

    stretch_mm = 0.05
    for fed_count in range(2, NODE_BUDGET + 1):
        front = _front()
        full = AxialStretch(edges=front.edges(EA_N))
        fed_only = AxialStretch(edges=front.edges(EA_N)[: fed_count - 1])
        state = State(
            layout=_layout(),
            vector=front.initial_positions_mm(_fed_positions(fed_count, stretch_mm)),
        )
        assert full.energy(state, CONTEXT) == fed_only.energy(state, CONTEXT), (
            f"喂到{fed_count}个节点时未喂边贡献了能量"
        )
        assert full.gradient(state, CONTEXT) == fed_only.gradient(state, CONTEXT)
        #: 已喂那一段的能量对手算：``(fed_count − 1)``条边各拉长``stretch_mm``。
        expected = (fed_count - 1) * 0.5 * (EA_N / REST_MM) * stretch_mm * stretch_mm
        assert full.energy(state, CONTEXT) == pytest.approx(expected, rel=1e-12)


def test_the_parked_nodes_sit_behind_the_inlet_one_rest_length_apart():
    """未喂节点停在喂料口**后方**，间隔恰是静止段长。

    停在**前**方会与已喂那一段重叠——重叠的节点在接触检测里是一对零距离的候选。
    间隔不等于静止段长则未喂边有伸长量，那一段会凭空产生能量。
    """

    front = _front()
    positions = front.initial_positions_mm(_fed_positions(4))
    #: 已喂的四个节点在``x = 30, 20, 10, 0``（节点0是引出端）；未喂的从``x = −10``往后排。
    for index in range(4):
        assert positions[3 * index] == pytest.approx(REST_MM * (3 - index))
    for offset, node in enumerate(range(4, NODE_BUDGET), start=1):
        assert positions[3 * node] == pytest.approx(-REST_MM * offset), (
            f"节点{node}没有停在喂料口后方第{offset}个静止段长上"
        )
        assert positions[3 * node + 1] == 0.0
        assert positions[3 * node + 2] == 0.0
    #: 未喂边的长度恰是静止段长 ⟹ 伸长量恰为零。
    for node in range(4, NODE_BUDGET - 1):
        gap = abs(positions[3 * (node + 1)] - positions[3 * node])
        assert gap == pytest.approx(REST_MM, rel=1e-15)


# ---------------------------------------------------------------------------
# 已喂那一段的物理就是一根短杆
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fed_count", [3, 6, 12])
def test_the_fed_portion_behaves_like_a_rod_of_that_length(fed_count: int):
    """在喂料口固支、末端拉一个力：伸长量对``F·L/EA``。

    ``L``是**已喂**的材料长度``(fed_count − 1)·rest``——
    未喂那一段一个牛顿都不承。这条把"布局定长"与"物理是短杆"两件事钉在一起。
    """

    front = _front()
    load_n = 30.0
    term = AxialStretch(edges=front.edges(EA_N))
    registry = EnergyRegistry(
        terms=(term, PointLoad(loads=((0, (load_n, 0.0, 0.0)),)))
    )
    #: 固支**喂料口那一端**（节点``fed_count−1``）＋钉住全部未喂节点＋
    #: 把已喂节点的``y``/``z``钉住（一维拉伸）。拉的是引出端（节点0）。
    inlet_node = fed_count - 1
    fixed = {3 * inlet_node, 3 * inlet_node + 1, 3 * inlet_node + 2}
    fixed |= set(front.parked_fixed_indices(fed_count))
    fixed |= {3 * node + axis for node in range(fed_count) for axis in (1, 2)}
    result = solve_equilibrium(
        registry, CONTEXT, _layout(),
        front.initial_positions_mm(_fed_positions(fed_count)),
        fixed_indices=frozenset(fixed), residual_tol_n=1.0e-10, max_iterations=50,
    )
    assert result.converged, result.reason

    length = front.fed_material_length_mm(fed_count)
    assert length == pytest.approx((fed_count - 1) * REST_MM, rel=1e-15)
    tip = result.state.vector[0]
    assert tip - length == pytest.approx(load_n * length / EA_N, rel=1e-9)


def test_the_fed_material_length_grows_linearly():
    """已喂材料长度``(fed_count − 1)·rest``——**少减那个1，起手就多一段**。"""

    front = _front()
    for fed_count in range(2, NODE_BUDGET + 1):
        assert front.fed_material_length_mm(fed_count) == pytest.approx(
            (fed_count - 1) * REST_MM, rel=1e-15
        )
    increments = [
        front.fed_material_length_mm(n + 1) - front.fed_material_length_mm(n)
        for n in range(2, NODE_BUDGET)
    ]
    assert all(value == pytest.approx(REST_MM, rel=1e-15) for value in increments)


def test_gravity_would_drag_the_unfed_nodes_if_they_were_not_pinned():
    """**不钉未喂节点的真危害不是奇异，是那段材料会被拉动。**

    写这条门时我判定"不钉就欠约束、`solve_equilibrium`当场报奇异"。
    **实测否掉**：未喂段通过接缝那条边与已喂段相连，链本身有刚度，一点都不奇异。

    真危害是**还没喂进来的材料被外力拉动**——它在真机上还在放线盘里。
    本门判的是那个外力真的存在：重力在每个未喂节点上的残差**非零**，
    而`parked_fixed_indices`恰好盖住它们。

    判残差而不判解，是因为带重力的悬链从近直线起步收敛很脆
    （横向刚度全部来自张力，而起点几乎无张力）——**判据要挑它判得准的事情判**。
    """

    front = _front()
    fed_count = 4
    heavy = EnergyContext(
        context_id="context/feed-gravity",
        node_masses_kg=(1.0e-3,) * NODE_BUDGET,
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )
    registry = EnergyRegistry(
        terms=(AxialStretch(edges=front.edges(EA_N)), UniformGravity())
    )
    state = State(
        layout=_layout(), vector=front.initial_positions_mm(_fed_positions(fed_count))
    )
    _, gradient, _ = registry.total(state, heavy, need_gradient=True)
    assert gradient is not None

    parked = front.parked_fixed_indices(fed_count)
    #: 每个未喂节点的``z``分量上都有一份重力残差——它们若自由就会被拉下去。
    weight_n = 1.0e-3 * 9810.0 / 1000.0
    for node in range(fed_count, NODE_BUDGET):
        assert gradient[3 * node + 2] == pytest.approx(weight_n, rel=1e-12), (
            f"未喂节点{node}的``z``上没有重力残差——那本门的前提不成立"
        )
        assert 3 * node + 2 in parked, f"未喂节点{node}的``z``没有被钉住"
    #: 已喂节点一个都不在钉集里——钉多了等于把要解的东西也钉住。
    for node in range(fed_count):
        for axis in range(3):
            assert 3 * node + axis not in parked


def test_the_parked_indices_cover_exactly_the_unfed_nodes():
    front = _front()
    for fed_count in range(2, NODE_BUDGET + 1):
        parked = front.parked_fixed_indices(fed_count)
        assert len(parked) == 3 * (NODE_BUDGET - fed_count)
        assert all(index >= 3 * fed_count for index in parked)
        assert max(parked, default=0) < 3 * NODE_BUDGET


# ---------------------------------------------------------------------------
# 失败关闭
# ---------------------------------------------------------------------------


def test_a_layout_that_does_not_match_the_budget_fails_closed():
    """**两处各说各的节点数**是plans/09教训一记的那种洞。"""

    front = _front()
    with pytest.raises(FeedError, match="各说各的节点数"):
        assert_layout_matches_budget(_layout(NODE_BUDGET - 1), front)
    with pytest.raises(FeedError, match="各说各的节点数"):
        assert_layout_matches_budget(_layout(NODE_BUDGET + 1), front)
    assert_layout_matches_budget(_layout(NODE_BUDGET), front)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"node_budget": 1}, "at least 2"),
        ({"node_budget": True}, "must be an int"),
        ({"node_budget": 3.0}, "must be an int"),
        ({"rest_length_mm": 0.0}, "rest_length_mm"),
        ({"direction": (0.0, 0.0, 2.0)}, "unit vector"),
        ({"inlet_mm": (0.0, float("nan"), 0.0)}, "inlet_mm"),
    ],
)
def test_a_malformed_front_fails_closed(overrides, message):
    with pytest.raises(FeedError, match=message):
        _front(**overrides)


@pytest.mark.parametrize("fed_count", [0, 1, NODE_BUDGET + 1, True, 3.0])
def test_a_bad_fed_count_fails_closed(fed_count):
    front = _front()
    with pytest.raises(FeedError):
        front.assert_fed_count(fed_count)


def test_feed_only_reaches_downwards():
    """本模块登记在力学域，只import同域的`state`——没有一条边越到别的域。"""

    from physics_engine import feed

    with open(feed.__file__ or "", encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in ("physics_engine.optics", "physics_engine.electromagnetics"):
        assert forbidden not in text
    assert "physics_engine.state" in text
    assert math.isfinite(REST_MM)
