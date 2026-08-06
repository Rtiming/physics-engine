"""公开面清册门：`src/`下每个模块都必须出现在`__init__.py`的API清单里。

## 为什么有这道门

**本仓已经两次让一个模块悄悄成为公开面**：

1. `cli.py`（历史前科，0017登记过同类）；
2. **`contact.py`**（2026-08-06对抗审核抓到）——42KB、18个公开名、
   三个案例在用，而`__init__.py`那份"公开API分两档"的清单里**一次都没出现**。

第二次尤其难看：`README.md`与`__init__.py`的能力边界段**同时**还写着
"没有接触与摩擦"，而接触是当天最大的一块代码。README自己第80行就写着
"**两段必须同批改**"——**规矩写了，没有任何东西执行它**。

## 这道门判什么

**只判"有没有被列出来"，不判描述对不对。**
描述的正确性靠人读，而人只有在东西被列出来时才读得到它。

`test_domain_isolation.py`的完备性门（第④条）守的是**域环登记**——
那是给隔离用的，与"使用者读不读得到"是两件事。**`contact`当时已经在域环里了，
所以那道门全绿**，而公开面清单里没有它。**两个清册，两件事，各要一道门。**
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "physics_engine"
INIT = PACKAGE / "__init__.py"

#: 不需要出现在公开面清单里的模块，**逐个带理由**。
#: 加进这张表要在这里写清楚为什么——**豁免必须有名字**。
EXEMPT: dict[str, str] = {
    "__init__": "它就是清单本身",
    "engine_facets": "面清册的自登记，不是给使用者调的API；由`facets`那一档间接暴露",
}


def _documented_modules(text: str) -> set[str]:
    """从`__init__.py`的docstring里取出被点名的模块名。

    只认``physics_engine.<名字>``这种带包前缀的写法——散文里提一句"接触"不算。
    **与案例页校验器那条教训同源**：`peer_fcl_distance`曾长期挂在"在建"那句话里，
    门认得"被提到"、认不得"被登记"。这里要求的是**全限定名**。
    """

    documented: set[str] = set()
    marker = "physics_engine."
    start = 0
    while (index := text.find(marker, start)) != -1:
        tail = text[index + len(marker) :]
        name = ""
        for character in tail:
            if character.isalnum() or character == "_":
                name += character
            else:
                break
        if name:
            documented.add(name)
        start = index + len(marker)
    return documented


def _package_modules() -> set[str]:
    names = {path.stem for path in PACKAGE.glob("*.py")}
    names |= {path.name for path in PACKAGE.iterdir() if (path / "__init__.py").is_file()}
    return names


def test_every_module_appears_in_the_public_surface_listing() -> None:
    """新模块不进公开面清单即红。"""

    documented = _documented_modules(INIT.read_text(encoding="utf-8"))
    missing = sorted(_package_modules() - documented - set(EXEMPT))
    assert not missing, (
        f"这些模块不在`__init__.py`的公开面清单里：{missing}——"
        "**使用者从API面上读不出它们存在**。要么补进清单，要么进EXEMPT并写明理由"
    )


def test_the_exemptions_are_real_modules() -> None:
    """豁免表不许留幽灵条目——模块删了豁免还在，下一个同名模块会白白继承豁免。"""

    ghosts = sorted(set(EXEMPT) - _package_modules())
    assert not ghosts, f"豁免表里的模块不存在：{ghosts}"


def test_the_scan_is_not_empty() -> None:
    """零执行绝不pass。"""

    modules = _package_modules()
    assert len(modules) > 15, f"只扫到{len(modules)}个模块——包结构变了还是扫错目录了"
    assert "contact" in modules, "contact模块不见了？本门就是为它建的"


def test_a_prose_mention_does_not_count_as_registration() -> None:
    """**判据本身被验**：散文里提一句不算登记，必须是全限定名。

    这条挡的是本仓已经吃过一次的亏——案例页校验器只要求案例名"出现在页上"，
    于是一个已落地的案例长期挂在"在建"那句话里。
    """

    assert "contact" not in _documented_modules("这里提到了 contact 但没有全限定名")
    assert "contact" in _documented_modules("见`physics_engine.contact`")
