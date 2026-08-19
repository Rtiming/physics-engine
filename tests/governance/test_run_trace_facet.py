"""`engine_run_trace`的面兼容门（决策0084第四节，承0076第五节那条待裁项）。

## 这道门补的洞

0076把轨迹形制写成了`tools/view/README.md`第五节的**草案**，理由是那一轨
承诺`src/`零字节改动、而面清册在`src/`里。于是形制**跨了边界却没进清册**——
`trace_from_closed_loop.py`写、`replay.py`读，两个工具住在互不认识对方依赖的
两个环境里，中间只有那份JSON。0076自己如实登记了后果：
**上游改一个字段，没有任何地方会红。**

0084按所有者裁决把源码上限抬到4 MiB，那条轨道范围内的约束不复存在，故裁升。

## 为什么这道门要静态扫而不是import

读的那一侧`replay.py`**不能**import`physics_engine`——它住在rerun那个venv里，
让它认识内核就破了0076那条"两个环境互不认识对方的依赖、中间只有一份JSON"，
而那条承诺正是`test_view_tools_stay_out_of_the_kernel.py`第四条判据在守的。

**所以读端的常量只能是抄的。** 抄本身不是问题，抄了之后没人核对才是问题——
本门就是那个核对：静态解析`replay.py`的模块级常量，与清册逐项比。
**门在这里正是因为不能靠import。**

## 三条判据

1. 清册里有这个面，且状态是draft（它今天只有仓内两个工具在用）；
2. **读端抄的名字与大版本必须与清册一致**；
3. **产端不许自己抄**——它认识`physics_engine`，没有理由抄；抄了就会出现
   "清册改了而落盘的字节没改"，那正是面清册要防的那件事。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    ENGINE_RUN_TRACE_FACET,
    ENGINE_RUN_TRACE_VERSION,
)
from physics_engine.facets import FacetError, FacetStatus

ROOT = Path(__file__).resolve().parents[2]
READER = ROOT / "tools/view/replay.py"
PRODUCER = ROOT / "tools/view/trace_from_closed_loop.py"


def module_level_constants(source: str) -> dict[str, object]:
    """模块级的`NAME = <字面量>`赋值。静态扫，**不执行**——读端import不了。

    只收字面量：读端如果把常量写成表达式（例如从别处算出来），
    本函数收不到它，第2条判据就会因为"缺这个名字"而红。**那是对的**——
    一个算出来的读端常量与清册之间同样没有核对，红了才有人回来看。
    """

    out: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = value
    return out


def imported_names(source: str) -> set[str]:
    """一段源码里所有`from ... import NAME`的名字。"""

    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.add(alias.asname or alias.name)
    return out


# ---------------------------------------------------------------------------
# 判据一：清册里有它，且是draft
# ---------------------------------------------------------------------------


def test_the_run_trace_facet_is_registered_as_a_draft():
    facet = ENGINE_REGISTRY.get(ENGINE_RUN_TRACE_FACET)
    assert facet.status is FacetStatus.DRAFT, (
        "轨迹面今天只有仓内两个工具在用，不作对外兼容承诺——"
        "升档要走决策记录（轴1规则5）。"
    )
    ENGINE_REGISTRY.assert_reader_compatible(
        ENGINE_RUN_TRACE_FACET, ENGINE_RUN_TRACE_VERSION
    )


def test_the_registry_still_refuses_an_incompatible_trace_version():
    """**判据本身也要被验**（tests/governance是样板）。

    清册若对任何版本都点头，上面那条`assert_reader_compatible`就是空的。
    """

    with pytest.raises(FacetError):
        ENGINE_REGISTRY.assert_reader_compatible(ENGINE_RUN_TRACE_FACET, "1.0")
    with pytest.raises(FacetError):
        ENGINE_REGISTRY.assert_reader_compatible(ENGINE_RUN_TRACE_FACET, "0.9")


# ---------------------------------------------------------------------------
# 判据二：读端抄的常量与清册一致
# ---------------------------------------------------------------------------


def test_the_reader_constants_match_the_registry():
    constants = module_level_constants(READER.read_text(encoding="utf-8"))
    registry_major, _ = _split(ENGINE_RUN_TRACE_VERSION)

    assert constants.get("FACET") == ENGINE_RUN_TRACE_FACET, (
        f"`replay.py`认的面名是{constants.get('FACET')!r}，清册是"
        f"{ENGINE_RUN_TRACE_FACET!r}——两边不一致时读端会静默拒收一份合法轨迹，"
        "而拒收的理由看起来像'文件坏了'。"
    )
    assert constants.get("SUPPORTED_MAJOR") == registry_major, (
        f"`replay.py`认的大版本是{constants.get('SUPPORTED_MAJOR')!r}，"
        f"清册是{registry_major}——大版本是本形制唯一的兼容闸门（读端明写不做'尽量读'）。"
    )


@pytest.mark.parametrize(
    "planted",
    [
        'FACET = "engine_run_trace_v2"\nSUPPORTED_MAJOR = 0\n',
        'FACET = "engine_run_trace"\nSUPPORTED_MAJOR = 1\n',
        'SUPPORTED_MAJOR = 0\n',  # 名字整个不见了
        'FACET = "engine_run_trace"\n',  # 大版本整个不见了
    ],
)
def test_the_reader_check_is_not_an_empty_gate(planted: str):
    """**必须红**：四种形态的读端漂移各喂一次，判据一条都不许放过。

    第三、四条喂的是"常量不见了"——上面那条判据用`.get()`取值，
    取不到时拿到的是`None`，与清册不相等，故仍然红。**这一条是在验那个`.get()`。**
    """

    constants = module_level_constants(planted)
    registry_major, _ = _split(ENGINE_RUN_TRACE_VERSION)
    drifted = (
        constants.get("FACET") != ENGINE_RUN_TRACE_FACET
        or constants.get("SUPPORTED_MAJOR") != registry_major
    )
    assert drifted, f"这份植入的读端应当被判漂移，但没有：{planted!r}"


# ---------------------------------------------------------------------------
# 判据三：产端不许自己抄
# ---------------------------------------------------------------------------


def test_the_producer_takes_the_facet_name_from_the_registry():
    source = PRODUCER.read_text(encoding="utf-8")
    names = imported_names(source)
    assert {"ENGINE_RUN_TRACE_FACET", "ENGINE_RUN_TRACE_VERSION"} <= names, (
        "产端认识`physics_engine`，没有理由抄一份常量——抄一份就意味着"
        "清册改了而落盘的字节没改，**而那正是面清册要防的那件事**（决策0017的教训）。"
    )
    constants = module_level_constants(source)
    assert "FACET" not in constants, (
        "产端的`FACET`被写成了字面量——它应当是清册那个名字的别名。"
        "字面量在这里是一次静默的分叉。"
    )


def test_the_producer_check_is_not_an_empty_gate():
    """**必须红**：一份自己抄常量的产端必须被判出来。"""

    planted = 'FACET = "engine_run_trace"\nFACET_VERSION = "0.1"\n'
    constants = module_level_constants(planted)
    assert "FACET" in constants, "植入的字面量产端应当被判出来，但没有"
    assert not ({"ENGINE_RUN_TRACE_FACET"} <= imported_names(planted))


def _split(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)
