"""接触锚点布局的门（决策0050第一节）。

物理判据不在这里——本片**不算任何物理**。
本文件守的是形制：**布局定长、指纹跨步不变、次序即形制、边界能被比对**。

其中第一条是0050整条裁决的承重点：0043列的三条路都会让布局随构型变，
而**布局一变指纹就变，跨步守恒量断言当场全废**（spec/12第2.1节）。
按声明的对分槽之所以成立，全部理由就是"**声明期定死**"这一件事——
所以它必须有一条正向的门钉着，而不是只写在决策记录里。
"""

from __future__ import annotations

import pytest

from physics_engine.contact import (
    REGIME_SEPARATED,
    REGIME_SLIP,
    REGIME_STICK,
    SLOT_WIDTH,
    ContactDeclaration,
    ContactError,
    ContactLayout,
    build_contact_layout,
)
from physics_engine.energies import (
    EnergyContext,
    EnergyError,
    UniformGravity,
    resolve_node_count,
)
from physics_engine.state import State


def _layout(node_count: int = 2, pairs: tuple[str, ...] = ("block_ground",)) -> ContactLayout:
    return build_contact_layout(
        layout_id="layout/contact-test",
        node_count=node_count,
        declarations=tuple(ContactDeclaration(pair_id) for pair_id in pairs),
    )


# ---------------------------------------------------------------------------
# 承重条款：布局定长，活动集变化不改指纹
# ---------------------------------------------------------------------------


def test_fingerprint_does_not_move_when_the_active_set_changes():
    """**0050整条裁决压在这一条上。**

    0043列的三条路里，"每种活动集一份布局"会让指纹随构型变，
    于是跨步守恒量断言在活动集变化的那一步全部失去意义。
    按声明的对分槽之后，活动与否是**向量里的一个值**——
    这里把"值怎么变指纹都不动"钉死。
    """

    contact = _layout(pairs=("a", "b"))
    before = contact.layout.fingerprint()

    separated = list(contact.initial_vector((0.0,) * 6))
    engaged = list(separated)
    slot = contact.slot_of("a")
    engaged[slot.active_index] = 1.0
    engaged[slot.regime_index] = REGIME_STICK
    engaged[slot.anchor_base : slot.anchor_base + 3] = [1.0, 2.0, 3.0]

    first = State(layout=contact.layout, vector=tuple(separated))
    second = State(layout=contact.layout, vector=tuple(engaged))

    assert first.layout.fingerprint() == before
    assert second.layout.fingerprint() == before, (
        "活动集变化改了布局指纹——那正是0043三条路的代价，本裁决要避免的就是它"
    )
    assert len(first.vector) == len(second.vector)


def test_the_node_block_is_a_prefix_and_slots_come_after():
    """节点块在前、槽在后。这是`energies.resolve_node_count`那条口径的另一半。"""

    contact = _layout(node_count=3, pairs=("a", "b"))
    assert contact.node_dof_count == 9
    assert contact.layout.dof_count == 9 + 2 * SLOT_WIDTH
    assert [slot.base for slot in contact.slots] == [9, 9 + SLOT_WIDTH]
    names = [field.name for field in contact.layout.fields]
    assert names[:9] == [
        f"node{index}_{axis}_mm" for index in range(3) for axis in ("x", "y", "z")
    ]
    assert all(name.startswith("contact_") for name in names[9:])


def test_every_slot_field_is_declared_history():
    """0033裁的是"锚点是**真历史**"。分不清就都当真历史（保守方向）。"""

    contact = _layout(pairs=("a",))
    history = set(contact.layout.history_fields())
    slot_fields = {field.name for field in contact.layout.fields if field.name.startswith("contact_")}
    assert slot_fields <= history, f"槽位字段没被声明成历史：{slot_fields - history}"
    assert not (history & {field.name for field in contact.layout.fields[:6]}), (
        "节点位置被误声明成历史了"
    )


def test_declaration_order_is_the_layout(
):
    """次序即形制：换个声明次序就是另一个布局，**指纹必须不同**。

    理由比"可复现"更硬一层：次序决定每个槽落在向量的哪一格，
    次序变了梯度与Hessian的索引全错，**而多数测试不会发现**（spec/12第2.2节）。
    """

    first = build_contact_layout(
        layout_id="layout/order",
        node_count=1,
        declarations=(ContactDeclaration("a"), ContactDeclaration("b")),
    )
    second = build_contact_layout(
        layout_id="layout/order",
        node_count=1,
        declarations=(ContactDeclaration("b"), ContactDeclaration("a")),
    )
    assert first.layout.dof_count == second.layout.dof_count
    assert first.layout.fingerprint() != second.layout.fingerprint(), (
        "两个次序不同的布局指纹相同——指纹就不再是打包次序的内容地址了"
    )
    assert first.slot_of("a").base != second.slot_of("a").base


# ---------------------------------------------------------------------------
# 必红：这一片顺带关掉的那条洞
# ---------------------------------------------------------------------------


def test_context_that_disagrees_with_the_node_block_now_fails_closed():
    """**那条洞的两半，现在都关上了。**

    历史：`resolve_node_count`起初只看得见``len(vector)``，看不见"哪一段是节点块"，
    于是一份"声明3个质量而布局只有2个节点"的上下文判不出来——**重力会落到锚点上**。
    第一次修只加了`ContactLayout.assert_matches_context`这条**要调用方主动去调**的路，
    而`EnergyRegistry.acceleration`那条桥是另一处实现，它照样信上下文。

    **本条的前半段曾经断言"那里确实判不出来"，并写明"洞不在了就该删"。
    2026-08-06`StateLayout.node_dof_count`让布局成为权威之后，它红了，于是删了。**
    这就是那句话该兑现的样子——**门不许验一件已经不成立的事**。

    今天两条路各守一段：布局声明了边界，`resolve_node_count`见不一致即失败关闭；
    `assert_matches_context`仍在，给不走能量层的调用方用。
    """

    contact = _layout(node_count=2, pairs=("a",))
    bad = EnergyContext(
        context_id="context/mismatch",
        node_masses_kg=(1.0, 1.0, 1.0),
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )
    state = State(layout=contact.layout, vector=contact.initial_vector((0.0,) * 6))

    # 布局是权威：不一致即失败关闭，不再返回一个错的节点数
    with pytest.raises(EnergyError, match="node block"):
        resolve_node_count(state, bad)
    with pytest.raises(EnergyError, match="node block"):
        UniformGravity().gradient(state, bad)

    with pytest.raises(ContactError, match="node masses"):
        contact.assert_matches_context(bad)

    good = EnergyContext(
        context_id="context/match",
        node_masses_kg=(1.0, 1.0),
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )
    contact.assert_matches_context(good)
    assert resolve_node_count(state, good) == 2


def test_the_acceleration_bridge_is_covered_by_the_same_authority():
    """**桥是那条洞的另一半，它必须走同一个权威。**

    第一次修桥只让它"不崩"（节点块之外返回0.0），没让它拒绝一个与布局不符的上下文。
    实测那时桥仍会把重力放到锚点槽的前三格上。
    """

    from physics_engine.energies import EnergyRegistry

    contact = _layout(node_count=2, pairs=("a",))
    bad = EnergyContext(
        context_id="context/bridge-mismatch",
        node_masses_kg=(1.0, 1.0, 1.0),
        gravity_mm_s2=(0.0, 0.0, -9810.0),
    )
    acceleration = EnergyRegistry(terms=(UniformGravity(),)).acceleration(
        bad, contact.layout
    )
    vector = contact.initial_vector((0.0,) * 6)
    with pytest.raises(EnergyError, match="node block"):
        acceleration(vector, (0.0,) * len(vector), 0.0)


# ---------------------------------------------------------------------------
# 失败关闭
# ---------------------------------------------------------------------------


def test_duplicate_pair_ids_fail_closed():
    with pytest.raises(ContactError, match="duplicate contact pair_id"):
        _layout(pairs=("a", "a"))


def test_a_declaration_without_slots_fails_closed():
    with pytest.raises(ContactError, match="at least 1"):
        ContactDeclaration("a", max_points=0)


def test_a_layout_without_nodes_fails_closed():
    with pytest.raises(ContactError, match="at least one node"):
        build_contact_layout(
            layout_id="layout/empty", node_count=0, declarations=(ContactDeclaration("a"),)
        )


def test_initial_vector_checks_the_node_block_length():
    contact = _layout(node_count=2, pairs=("a",))
    with pytest.raises(ContactError, match="expected 6 node scalars"):
        contact.initial_vector((0.0, 0.0, 0.0))


def test_slot_lookup_fails_closed_on_an_unknown_pair():
    with pytest.raises(ContactError, match="no contact slot"):
        _layout(pairs=("a",)).slot_of("nope")


def test_a_fresh_slot_carries_no_history():
    """出生态就是"没碰过"：全零，且`REGIME_SEPARATED`恰好是0.0。

    这条钉住的是"不需要额外初始化步骤"——一个需要初始化的历史字段，
    迟早会有人忘了初始化它。
    """

    contact = _layout(pairs=("a",))
    vector = contact.initial_vector((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    slot = contact.slot_of("a")
    assert vector[slot.active_index] == 0.0
    assert vector[slot.anchor_base : slot.anchor_base + 3] == (0.0, 0.0, 0.0)
    assert vector[slot.regime_index] == REGIME_SEPARATED
    assert REGIME_SEPARATED == 0.0
    assert {REGIME_SEPARATED, REGIME_STICK, REGIME_SLIP} == {0.0, 1.0, 2.0}


def test_multi_point_pairs_get_contiguous_slots():
    """盒/网格族进来时每对会有多个点。槽位次序在那时就必须是确定的。"""

    contact = build_contact_layout(
        layout_id="layout/multi",
        node_count=1,
        declarations=(ContactDeclaration("face", max_points=4),),
    )
    bases = [slot.base for slot in contact.slots]
    assert bases == [3, 3 + SLOT_WIDTH, 3 + 2 * SLOT_WIDTH, 3 + 3 * SLOT_WIDTH]
    assert [slot.point_index for slot in contact.slots] == [0, 1, 2, 3]
