#!/usr/bin/env python3
"""能力位清单的计数门——**分子从清单里长出来，不从散文里来**（决策0052第二节）。

    python tools/check_capability_ledger.py

## 为什么需要它

[plans/07](../docs/plans/07_缺口清册_20260805.md)第四节第一行登记的欠账原文是
"**没有门看着分子**"：6/13这个数从散文里来，实测被多算过一条、四手复制无人回表，
订正花了一份决策记录（0049第十节）。**同一种病在本仓出现过三次**。

0052第二节把主分母也改成了能力位（"六个场景各拆成若干可核对的能力位，
做到哪位报哪位"），理由是plans/08阶段3做的是"平面围成的漏斗+一个球"，
而场景③原文是"**不规则体**落漏斗"——**把前者报成"场景③做完了"是虚报**。

于是这道门守两件事：**清单里的每一位都能被核对**，以及**两个分子都是算出来的**。

## 这道门判什么

数据在`docs/capability_ledger.json`。**它里面没有任何总数字段**——
加载器的键集是严格的（见``_strict_keys``），一个手写的总数**存不进去**。
所以本门打印的"主分母X/Y位、C档A/B条"只可能是数出来的。

十九条判据，每条一个稳定的错误码（``LedgerError.code``），
必红矩阵按**错误码**组织而不是按规则组织，元测试守"每个码至少被一条红用例走过"。
理由见plans/09教训三：域隔离门有九条必红**全部用绝对import**，
而它对相对import完全失明——**门全绿不是因为它挡得住，是那条分支从没被执行过**。

## 判据落在结构位置上，不落在"出现过"上

这是plans/09教训二的通则，本仓已经栽过四次
（`peer_fcl_distance`挂在"在建"那句话里就算登记、`check_gap_register`第一版
认子串于是删掉整行照样绿、缺口清册一页之内三处互斥而门全绿）。

本门的三处落点：

1. **证据里的测试认全限定名**：``test:<路径>::<函数>``要求那个函数
   **在文件里以``def <函数>(``开头的形式被定义**。写在docstring里、
   注释里、字符串里的同名文字**不算**——那正是"被提到"与"被登记"的分界；
2. **证据里的案例认目录**：``case:<id>``要求``cases/<id>/case.md``是一个文件，
   不是"这个名字在某处出现过"；
3. **位号认结构位置**：``S3.4``必须住在``S3``底下并且是它的第4位，
   同行C档的``index``必须恰好是1—13各一次。**分母因此不能被悄悄改大改小**。

## 明确挡不住的

- **拆位拆得对不对**。门能判"两位的证据集完全相同"（那是把一件事拆成两位
  来抬分子），**判不了两位在语义上是不是一件事**。语义不重叠靠人读，
  靠的是每一位的``label``与``why``写得能被反驳。
  **一道试图判断"这两位是不是同一件事"的门会立刻变成自然语言处理，然后被关掉。**
- **状态报得诚不诚实**。门能判"done的位必须能解析到真实存在的案例或测试"，
  **判不了那个案例是不是真的验到了这一位**。这一条只能靠案例页的判据表与
  `check_case_pages`那一侧，以及人读。
## 散文里的数：2026-08-18补上（决策0084第八节）

上一版这里写着"散文里的数与本清单一致与否"挡不住，并登记了触发条件
（0056那条的措辞是"**下一次任一分子变化前**，先让门从清单计算值核对这些固定结构位置"）。
**基础设施批次四条轨都会动分子，触发条件到了，故补上。**

补法是**按固定结构位置逐个锚定**，不是全文扫分数——全文扫会当场误红：
README那一行里`6/13`后面紧跟着"2026-08-18曾报7/13，当天被否掉并退回"，
而**那个7/13是一段历史，不是一个待更新的读数**。
一道会把历史记录判成陈旧的门会被关掉，然后这一条就白立了。

所以`PROSE_ANCHORS`逐条写死"哪份文件、哪一行（按行内锚串认）、哪个分母、取第几个"，
取的是**该行第一个**该分母的分数——历史值一律写在真值之后（本仓三处现状如此）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "capability_ledger.json"

SCHEMA = "capability_ledger/1"

#: 六个场景的正本编号（plans/04与plans/05的①—⑥）。**分母的骨架，改它要走决策记录。**
SCENARIO_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

#: 0057经第二次人工语义复核后冻结的主分母骨架。冻结**每场景位数**而不只冻总数，
#: 否则从S1删一位、给S3添一位仍可保持42，却已经换了一套口径。
#: 新能力可以改变状态，不可静默拆位/并位；确需改口径先写后续决策记录再改这里。
FROZEN_SCENARIO_BIT_COUNTS: dict[str, int] = {
    "S1": 7,
    "S2": 5,
    "S3": 10,
    "S4": 7,
    "S5": 6,
    "S6": 7,
}

#: 同行C档的条数。外部给定（research/05第2.3节，0040第二节点名的13条），
#: **不是我们自己划的及格线**——所以它是常数，不是清单里数出来的。
PEER_TIER_C_COUNT = 13

STATUSES: frozenset[str] = frozenset({"done", "partial", "todo"})

#: plans/06立的四条结构性缺口。位可以声明自己阻塞在哪一条上，但**只能选这四个**——
#: 自由文本会让"有多少位卡在同一堵墙上"这个数立刻失去意义。
STRUCTURAL_GAPS: frozenset[str] = frozenset({"体积与厚度", "接触", "耦合", "历史"})

_EVIDENCE_CASE = re.compile(r"^case:([A-Za-z0-9_]+)$")
_EVIDENCE_TEST = re.compile(r"^test:([A-Za-z0-9_./-]+\.py)::([A-Za-z_][A-Za-z0-9_]*)$")


class LedgerError(Exception):
    """清单的一切失败关闭。``code``是判据分支的稳定身份，必红矩阵按它组织。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class LedgerEmpty(LedgerError):
    """空跑：文件不在、读不动、或者一位都没解析出来。**空跑不是通过。**"""


@dataclass(frozen=True)
class Bit:
    """一个能力位。``owner``是它所属的场景或"C档"，用于报错时定位。"""

    id: str
    label: str
    status: str
    exercises_physics: bool
    evidence: tuple[str, ...]
    why: str
    missing: str | None
    shared_gap: str | None
    owner: str
    #: 同行C档专有：外部给定的条号（1—13）。场景位为None。
    index: int | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    ordinal: str
    label: str
    dominant_domain: str
    bits: tuple[Bit, ...]


@dataclass(frozen=True)
class Ledger:
    scenarios: tuple[Scenario, ...]
    peer_tier_c: tuple[Bit, ...]
    path: Path

    def all_bits(self) -> tuple[Bit, ...]:
        bits: list[Bit] = []
        for scenario in self.scenarios:
            bits.extend(scenario.bits)
        bits.extend(self.peer_tier_c)
        return tuple(bits)


@dataclass(frozen=True)
class ScenarioCount:
    id: str
    ordinal: str
    label: str
    done: int
    partial: int
    todo: int

    @property
    def total(self) -> int:
        return self.done + self.partial + self.todo


@dataclass(frozen=True)
class Counts:
    """**全部字段都是算出来的**——清单里没有一个总数字段可以抄。"""

    scenarios: tuple[ScenarioCount, ...] = ()
    main_done: int = 0
    main_total: int = 0
    main_partial: int = 0
    main_done_physics: int = 0
    peer_done: int = 0
    peer_partial: int = 0
    peer_total: int = 0
    end_to_end_done: int = 0
    gap_load: tuple[tuple[str, int], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 加载：严格键集。**手写的总数在这里就进不来。**
# ---------------------------------------------------------------------------

_TOP_REQUIRED = frozenset({"schema", "scenarios", "peer_tier_c"})
_TOP_OPTIONAL = frozenset({"why_this_file_is_data", "sources", "structural_gaps"})
_SCENARIO_REQUIRED = frozenset({"id", "ordinal", "label", "dominant_domain", "bits"})
_BIT_REQUIRED = frozenset({"id", "label", "status", "exercises_physics", "evidence", "why"})
_BIT_OPTIONAL = frozenset({"missing", "shared_gap"})
_PEER_REQUIRED = _BIT_REQUIRED | frozenset({"index", "family", "criterion"})


def _strict_keys(
    node: object, required: frozenset[str], optional: frozenset[str], where: str
) -> dict:
    """严格键集：少一个红，多一个也红。

    **多一个也红**是这道门"分子必须算出来"的执行体：清单里加不进
    ``"done_count": 11``这样的字段——它会被当作未知键当场拒收。
    """

    if not isinstance(node, dict):
        raise LedgerError("NODE_NOT_OBJECT", f"{where}：应当是一个对象，拿到{type(node).__name__}")
    keys = set(node)
    unknown = sorted(keys - required - optional)
    if unknown:
        raise LedgerError(
            "UNKNOWN_KEY",
            f"{where}：出现未登记的键{unknown}。"
            "**清单里不许有总数字段**——分子由本门算，不由清单声明",
        )
    absent = sorted(required - keys)
    if absent:
        raise LedgerError("MISSING_KEY", f"{where}：缺必填键{absent}")
    return node


def _text(node: dict, key: str, where: str) -> str:
    value = node.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LedgerError("TEXT_EMPTY", f"{where}：`{key}`必须是非空字符串，拿到{value!r}")
    return value


def _parse_bit(node: object, *, owner: str, peer: bool) -> Bit:
    required = _PEER_REQUIRED if peer else _BIT_REQUIRED
    where = f"{owner}的某一位"
    entry = _strict_keys(node, required, _BIT_OPTIONAL, where)
    bit_id = entry.get("id")
    if not isinstance(bit_id, str) or not bit_id:
        raise LedgerError("BIT_ID_MISSING", f"{where}：`id`必须是非空字符串")
    where = f"{owner} {bit_id}"

    status = entry.get("status")
    if status not in STATUSES:
        raise LedgerError(
            "STATUS_UNKNOWN",
            f"{where}：status必须是{sorted(STATUSES)}之一，拿到{status!r}",
        )

    exercises = entry.get("exercises_physics")
    if not isinstance(exercises, bool):
        raise LedgerError(
            "PHYSICS_FLAG_SHAPE",
            f"{where}：`exercises_physics`必须是布尔量（穿不穿物理机械是二选一，"
            f"不许留白），拿到{exercises!r}",
        )

    evidence = entry.get("evidence")
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        raise LedgerError("EVIDENCE_SHAPE", f"{where}：`evidence`必须是字符串数组")

    missing = entry.get("missing")
    if status == "done":
        if missing is not None:
            raise LedgerError(
                "MISSING_ON_DONE",
                f"{where}：done的位不许带`missing`——**还缺东西就不是done**",
            )
    else:
        if not isinstance(missing, str) or not missing.strip():
            raise LedgerError(
                "MISSING_ABSENT",
                f"{where}：{status}的位必须写`missing`说明**缺在哪**"
                "（0052第二节：partial必须写明partial在哪）",
            )

    gap = entry.get("shared_gap")
    if gap is not None and gap not in STRUCTURAL_GAPS:
        raise LedgerError(
            "SHARED_GAP_UNKNOWN",
            f"{where}：`shared_gap`只能取plans/06那四条{sorted(STRUCTURAL_GAPS)}，拿到{gap!r}",
        )

    index = entry.get("index") if peer else None
    if peer and not isinstance(index, int):
        raise LedgerError("PEER_INDEX_MISSING", f"{where}：C档每条必须带整数`index`，拿到{index!r}")

    return Bit(
        id=bit_id,
        label=_text(entry, "label", where),
        status=status,
        exercises_physics=exercises,
        evidence=tuple(evidence),
        why=_text(entry, "why", where),
        missing=missing,
        shared_gap=gap,
        owner=owner,
        index=index,
    )


def load_ledger(path: Path) -> Ledger:
    """严格加载。文件不在／读不动／一位都没有 → ``LedgerEmpty``（空跑不是通过）。"""

    if not path.is_file():
        raise LedgerEmpty("LEDGER_MISSING", f"能力位清单不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LedgerEmpty("LEDGER_NOT_JSON", f"能力位清单读不动：{path}——{error}") from error

    top = _strict_keys(raw, _TOP_REQUIRED, _TOP_OPTIONAL, "清单顶层")
    if top.get("schema") != SCHEMA:
        raise LedgerError(
            "SCHEMA_UNKNOWN",
            f"清单顶层：schema必须是{SCHEMA!r}，拿到{top.get('schema')!r}。"
            "**版本不认识就不许判**——判不了不等于判过了",
        )

    scenarios_raw = top.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise LedgerEmpty("EMPTY_SCENARIOS", "清单里一个场景都没有——这不是通过，是空跑")
    scenarios: list[Scenario] = []
    for node in scenarios_raw:
        entry = _strict_keys(node, _SCENARIO_REQUIRED, frozenset(), "某个场景")
        scenario_id = entry.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise LedgerError("SCENARIO_ID_MISSING", "某个场景：`id`必须是非空字符串")
        bits_raw = entry.get("bits")
        if not isinstance(bits_raw, list) or not bits_raw:
            raise LedgerError(
                "SCENARIO_EMPTY",
                f"场景{scenario_id}一位都没有——**一个拆不出能力位的场景没法报"
                "「第几位／共几位」**",
            )
        scenarios.append(
            Scenario(
                id=scenario_id,
                ordinal=_text(entry, "ordinal", f"场景{scenario_id}"),
                label=_text(entry, "label", f"场景{scenario_id}"),
                dominant_domain=_text(entry, "dominant_domain", f"场景{scenario_id}"),
                bits=tuple(
                    _parse_bit(bit, owner=scenario_id, peer=False) for bit in bits_raw
                ),
            )
        )

    peer_raw = top.get("peer_tier_c")
    if not isinstance(peer_raw, list) or not peer_raw:
        raise LedgerEmpty("EMPTY_PEER_TIER", "清单里C档一条都没有——这不是通过，是空跑")
    peer = tuple(_parse_bit(node, owner="C档", peer=True) for node in peer_raw)
    return Ledger(scenarios=tuple(scenarios), peer_tier_c=peer, path=path)


# ---------------------------------------------------------------------------
# 结构位置：位号、场景集合、C档条号
# ---------------------------------------------------------------------------


def assert_ids_are_structural(ledger: Ledger) -> int:
    """位号必须**住在它声明的位置上**，不是随便一个不重复的字符串。

    ``S3.4``要求：属于场景``S3``、且是该场景第4位（从1连号）。
    连号是分母的防篡改：跳号或重号会让"共几位"这个数变得可以商量。
    """

    if tuple(scenario.id for scenario in ledger.scenarios) != SCENARIO_IDS:
        raise LedgerError(
            "SCENARIO_SET",
            f"场景集合必须恰好是{list(SCENARIO_IDS)}且按序，拿到"
            f"{[scenario.id for scenario in ledger.scenarios]}",
        )
    measured_counts = {scenario.id: len(scenario.bits) for scenario in ledger.scenarios}
    if measured_counts != FROZEN_SCENARIO_BIT_COUNTS:
        raise LedgerError(
            "FROZEN_MAIN_DENOMINATOR",
            f"主分母骨架已由0057冻结为{FROZEN_SCENARIO_BIT_COUNTS}（合计42位），"
            f"拿到{measured_counts}——新增能力改状态，拆位/并位须先走后续决策记录",
        )
    seen = 0
    for scenario in ledger.scenarios:
        for position, bit in enumerate(scenario.bits, start=1):
            expected = f"{scenario.id}.{position}"
            if bit.id != expected:
                raise LedgerError(
                    "BIT_ID_SHAPE",
                    f"场景{scenario.id}第{position}位的id应当是{expected}，拿到{bit.id}——"
                    "**位号是结构位置，不是标签**",
                )
            seen += 1
    return seen


def assert_peer_tier_c_is_the_declared_thirteen(ledger: Ledger) -> int:
    """C档的分母是**外部给定的13**（research/05第2.3节），不是清单里数出来的。

    条号必须是1—13各恰好一次，``id``必须是``C<条号>``。
    多一条少一条都红——**分母被悄悄改大，分数就会好看**。
    """

    indices = [bit.index for bit in ledger.peer_tier_c]
    if sorted(index for index in indices if index is not None) != list(
        range(1, PEER_TIER_C_COUNT + 1)
    ):
        raise LedgerError(
            "PEER_INDEX_SET",
            f"C档条号必须是1—{PEER_TIER_C_COUNT}各一次，拿到{indices}",
        )
    for bit in ledger.peer_tier_c:
        if bit.id != f"C{bit.index}":
            raise LedgerError(
                "PEER_ID_SHAPE", f"C档第{bit.index}条的id应当是C{bit.index}，拿到{bit.id}"
            )
    return len(ledger.peer_tier_c)


def assert_labels_do_not_repeat(ledger: Ledger) -> int:
    """同一个所有者底下不许有两位叫同一个名字。

    这是"不重叠"能自动判的那一半的第一层。语义上的不重叠判不了（见模块docstring）。
    """

    checked = 0
    groups: dict[str, list[Bit]] = {}
    for bit in ledger.all_bits():
        groups.setdefault(bit.owner, []).append(bit)
    for owner, bits in groups.items():
        labels = [bit.label for bit in bits]
        for label in labels:
            if labels.count(label) > 1:
                raise LedgerError(
                    "DUPLICATE_LABEL", f"{owner}底下有两位叫『{label}』——那是一位不是两位"
                )
        checked += len(bits)
    return checked


def assert_evidence_matches_status(ledger: Ledger) -> int:
    """三种状态各自的证据义务。

    * ``done``/``partial``：**至少一条证据**——做到了就必须指得出来；
    * ``todo``：**证据必须为空**——"未做"是一个明确的声明，
      带着证据的todo要么是状态写错，要么是证据摆好看的。
    """

    checked = 0
    for bit in ledger.all_bits():
        if bit.status in ("done", "partial") and not bit.evidence:
            raise LedgerError(
                "EVIDENCE_ABSENT",
                f"{bit.owner} {bit.id}报{bit.status}却一条证据都没有——"
                "**可核对是这份清单的入场条件**",
            )
        if bit.status == "todo" and bit.evidence:
            raise LedgerError(
                "EVIDENCE_ON_TODO",
                f"{bit.owner} {bit.id}报todo却挂着证据{list(bit.evidence)}——"
                "未做就是未做，别挂装饰",
            )
        checked += 1
    return checked


def assert_evidence_resolves(ledger: Ledger, root: Path) -> int:
    """每一条证据必须**解析到真实存在的东西**。

    ``case:<id>`` → ``cases/<id>/case.md``是文件；
    ``test:<路径>::<函数>`` → 文件在，**且那个函数在文件里被定义**。

    第二条是这道门的要害：只要求"函数名出现在文件里"，
    写在docstring或注释里的同名文字就能骗过它——那正是plans/09教训二
    （门认得"被提到"，认不得"被登记"）在本门上的形态。
    """

    resolved = 0
    for bit in ledger.all_bits():
        for item in bit.evidence:
            case = _EVIDENCE_CASE.match(item)
            test = _EVIDENCE_TEST.match(item)
            if case is None and test is None:
                raise LedgerError(
                    "EVIDENCE_UNKNOWN_KIND",
                    f"{bit.owner} {bit.id}的证据『{item}』形制不认识——"
                    "只收`case:<案例目录名>`与`test:<路径>::<测试函数名>`",
                )
            if case is not None:
                page = root / "cases" / case.group(1) / "case.md"
                if not page.is_file():
                    raise LedgerError(
                        "CASE_MISSING",
                        f"{bit.owner} {bit.id}指向的案例不存在：{page}",
                    )
            else:
                assert test is not None
                relative, function = test.group(1), test.group(2)
                path = root / relative
                if not path.is_file():
                    raise LedgerError(
                        "TEST_FILE_MISSING",
                        f"{bit.owner} {bit.id}指向的测试文件不存在：{path}",
                    )
                source = path.read_text(encoding="utf-8")
                if not re.search(rf"^(?:async )?def {re.escape(function)}\(", source, re.M):
                    raise LedgerError(
                        "TEST_FUNCTION_MISSING",
                        f"{bit.owner} {bit.id}指向的`{function}`在{relative}里"
                        "**没有被定义**（名字出现在正文里不算——认结构位置，不认被提到）",
                    )
            resolved += 1
    return resolved


def assert_no_two_bits_make_the_same_claim(ledger: Ledger) -> int:
    """**同一个分母里**，两位不许拿完全相同的一组证据。

    这是"不重叠"能自动判的另一半，而且正是计数门最该防的那一手：
    把一件已经做成的事拆成两位来抬分子。证据集相同 = 同一个主张换了两个名字。
    ``todo``的位证据为空，天然相同，故不参与本条（它们的不重叠靠人读）。

    **为什么按分母分组而不是全清单一把抓**：本仓的成功标准是**两条并列的分母**
    （0048第二节、cases/README"计入成功标准的是哪些"）——主分母验"算不算得了
    我们要算的"，C档验"算得对不对"。同一个案例**同时服务两条分母是设计如此**，
    不是重复计数：`fts_instrument_line_shape`既是场景④的一位，也是C档第11条。
    第一版没分组，实测当场把这条正当的复用打红了——**规则本身错了，不是数据错了**。
    """

    checked = 0
    for group, bits in (
        ("主分母", [bit for scenario in ledger.scenarios for bit in scenario.bits]),
        ("C档", list(ledger.peer_tier_c)),
    ):
        seen: dict[frozenset[str], Bit] = {}
        for bit in bits:
            if not bit.evidence:
                continue
            key = frozenset(bit.evidence)
            if key in seen:
                other = seen[key]
                raise LedgerError(
                    "DUPLICATE_CLAIM",
                    f"{group}里{bit.owner} {bit.id}与{other.owner} {other.id}"
                    f"的证据集完全相同（{sorted(key)}）——**同一件事被数了两次**",
                )
            seen[key] = bit
            checked += 1
    return checked


# ---------------------------------------------------------------------------
# 计数：**唯一的分子来源**
# ---------------------------------------------------------------------------


def count(ledger: Ledger) -> Counts:
    """从清单数出两个分子。清单里没有任何总数字段，所以这里是它们唯一的出处。"""

    scenario_counts: list[ScenarioCount] = []
    gap_load: dict[str, int] = {}
    for scenario in ledger.scenarios:
        tally = {status: 0 for status in STATUSES}
        for bit in scenario.bits:
            tally[bit.status] += 1
            if bit.shared_gap is not None:
                gap_load[bit.shared_gap] = gap_load.get(bit.shared_gap, 0) + 1
        scenario_counts.append(
            ScenarioCount(
                id=scenario.id,
                ordinal=scenario.ordinal,
                label=scenario.label,
                done=tally["done"],
                partial=tally["partial"],
                todo=tally["todo"],
            )
        )
    main_bits = [bit for scenario in ledger.scenarios for bit in scenario.bits]
    return Counts(
        scenarios=tuple(scenario_counts),
        main_done=sum(entry.done for entry in scenario_counts),
        main_total=sum(entry.total for entry in scenario_counts),
        main_partial=sum(entry.partial for entry in scenario_counts),
        main_done_physics=sum(
            1 for bit in main_bits if bit.status == "done" and bit.exercises_physics
        ),
        peer_done=sum(1 for bit in ledger.peer_tier_c if bit.status == "done"),
        peer_partial=sum(1 for bit in ledger.peer_tier_c if bit.status == "partial"),
        peer_total=len(ledger.peer_tier_c),
        end_to_end_done=sum(
            1 for entry in scenario_counts if entry.total and entry.done == entry.total
        ),
        gap_load=tuple(sorted(gap_load.items(), key=lambda item: (-item[1], item[0]))),
    )


def format_report(counts: Counts) -> str:
    """报告体。第一行就是0052要的那两个数，随后逐场景报"第几位／共几位"。"""

    lines = [
        f"capability ledger: 主分母 {counts.main_done}/{counts.main_total}位、"
        f"C档 {counts.peer_done}/{counts.peer_total}条",
        f"  其中穿过物理机械的done位：{counts.main_done_physics}/{counts.main_done}"
        "（其余是闭式计算器与装配基座——plans/05第二节的通则）",
        f"  另有partial：主分母{counts.main_partial}位、C档{counts.peer_partial}条"
        "（**做了一半不进位**）",
    ]
    for entry in counts.scenarios:
        lines.append(
            f"  {entry.id} {entry.ordinal} {entry.label}："
            f"第{entry.done}位／共{entry.total}位"
            f"（partial {entry.partial}、todo {entry.todo}）"
        )
    lines.append(
        f"  端到端做完的场景：{counts.end_to_end_done}/{len(counts.scenarios)}"
    )
    if counts.gap_load:
        load = "、".join(f"{name} {number}位" for name, number in counts.gap_load)
        lines.append(f"  卡在同一条结构性缺口上的位（plans/06四条）：{load}")
    return "\n".join(lines)


def check(path: Path, root: Path) -> Counts:
    """跑完全部判据并返回计数。任何一条不过都抛``LedgerError``。"""

    ledger = load_ledger(path)
    assert_ids_are_structural(ledger)
    assert_peer_tier_c_is_the_declared_thirteen(ledger)
    assert_labels_do_not_repeat(ledger)
    assert_evidence_matches_status(ledger)
    assert_evidence_resolves(ledger, root)
    assert_no_two_bits_make_the_same_claim(ledger)
    return count(ledger)


# ---------------------------------------------------------------------------
# 散文对账：固定结构位置（决策0084第八节，兑现0056那条"下一次任一分子变化前"）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProseAnchor:
    """一处写着分数的散文位置。

    `line_contains`认的是**行内锚串**而不是行号——行号会被上下文的增删推着走，
    而一道因为别人加了一段就红的门会被当成噪声关掉。
    """

    #: 仓库根相对路径。
    path: str
    #: 行内锚串，唯一确定那一行。不唯一即红（见`assert_prose_matches_counts`）。
    line_contains: str
    #: 分母。
    denominator: int
    #: 这个分数应当等于`Counts`的哪个字段。
    counts_field: str


#: **三份散文、六处分数。** 每一处都是人手写的，而人手写的数会静默陈旧——
#: 0056登记这条时的原话是"没有门看着分子"，本表是那句话的最后一段。
PROSE_ANCHORS: tuple[ProseAnchor, ...] = (
    ProseAnchor("README.md", "主｜用户六场景端到端", 6, "end_to_end_done"),
    ProseAnchor("README.md", "从｜同行C档13条标准案例", 13, "peer_done"),
    ProseAnchor("README.md", "能力位清单的当前机械计数是", 42, "main_done"),
    ProseAnchor("cases/README.md", "主｜用户六场景端到端", 6, "end_to_end_done"),
    ProseAnchor("cases/README.md", "从｜同行C档13条标准案例", 13, "peer_done"),
    ProseAnchor("cases/README.md", "主分母的逐位机械计数当前为", 42, "main_done"),
)


def first_fraction_with(line: str, denominator: int) -> int | None:
    r"""一行里**第一个**分母为`denominator`的分数的分子。

    `(?<![\d/])`与`(?![\d/])`两个界防的是`7/5/10/7/6/7`那种逐场景位数串——
    里面的`7/6`不是"六个场景做完了七个"，而`cases/README.md`第166行就有这么一串。
    **不设这两个界，本门第一次跑就是误红。**
    """

    match = re.search(rf"(?<![\d/])(\d+)/{denominator}(?![\d/])", line)
    return int(match.group(1)) if match else None


def assert_prose_matches_counts(counts: Counts, root: Path) -> None:
    """散文里的固定结构位置必须与算出来的数相等。**算出来的是唯一真值。**"""

    for anchor in PROSE_ANCHORS:
        path = root / anchor.path
        if not path.is_file():
            raise LedgerError(
                "PROSE_FILE_MISSING",
                f"散文对账：{anchor.path}不存在——本表指着一份不存在的文件",
            )
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if anchor.line_contains in line
        ]
        if not lines:
            raise LedgerError(
                "PROSE_ANCHOR_LOST",
                f"散文对账：{anchor.path}里找不到锚串{anchor.line_contains!r}——"
                "**锚串失效等于这一处不再被看着**，比数字错更坏。改散文时把锚串一起改。"
            )
        if len(lines) > 1:
            raise LedgerError(
                "PROSE_ANCHOR_AMBIGUOUS",
                f"散文对账：{anchor.path}里锚串{anchor.line_contains!r}命中{len(lines)}行，"
                "唯一性没了就说不清在核对哪一处。"
            )
        expected = getattr(counts, anchor.counts_field)
        found = first_fraction_with(lines[0], anchor.denominator)
        if found is None:
            raise LedgerError(
                "PROSE_FRACTION_ABSENT",
                f"散文对账：{anchor.path}那一行没有写分母为{anchor.denominator}的分数——"
                f"锚串{anchor.line_contains!r}"
            )
        if found != expected:
            raise LedgerError(
                "PROSE_STALE",
                f"散文对账：{anchor.path}写着{found}/{anchor.denominator}，"
                f"从清单算出来是{expected}/{anchor.denominator}——"
                "**清单是唯一真值**，散文要跟着改（决策0052第二节）。"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="能力位清单的计数门")
    parser.add_argument("--root", default=str(ROOT), help="仓库根（证据按它解析）")
    parser.add_argument("--ledger", default=None, help="清单路径，默认docs/capability_ledger.json")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    path = Path(args.ledger).resolve() if args.ledger else root / "docs" / "capability_ledger.json"
    try:
        counts = check(path, root)
        # **散文对账不在`check()`里，这是一条有意的分工。**
        # `check(path, root)`回答的是"这份清单本身立不立得住"，它的调用方包括
        # 一批喂**人造清单**的必红用例；把仓库散文的对账塞进去，那些用例就会
        # 因为"人造清单的分子与真README不符"而红——**红得毫无意义**。
        # 散文对账问的是另一件事："**这个仓**里手写的那几个分数跟不跟得上"，
        # 所以它住在跑一次真仓的这一层。
        assert_prose_matches_counts(counts, root)
    except LedgerEmpty as error:
        print(f"{error}\n—— 空跑不是通过。", file=sys.stderr)
        return 2
    except LedgerError as error:
        print(str(error), file=sys.stderr)
        print(
            "—— 清单是分子的唯一出处（决策0052第二节）；改状态先改"
            "docs/capability_ledger.json，再让本门重新数一次。",
            file=sys.stderr,
        )
        return 1
    print(format_report(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
