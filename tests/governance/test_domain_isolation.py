"""域隔离门——spec/15的执行体，也是spec/01第一节两句边界第一次有牙齿。

spec/01从2026-08-04起就写着"**物理域之间互不依赖**（力学不import光学）"与
"跨域耦合以显式`couplings/`模块出现，**不许在域内偷连**"。在本文件之前，
这两句只是文字：没有任何东西阻止有人在`energies.py`里写一行
`from physics_engine.optics import ...`，也没有任何东西会在半年后发现它。

本门用**AST静态扫import**（不是运行时探测），理由有三条且都是硬的：

1. 运行时探测只能看见"这次真的执行到的"import，函数体里的延迟import看不见——
   而"偷连"最常见的形态恰恰是延迟import（写在函数里，说是"避免循环依赖"）；
2. 静态扫不需要导入被扫的模块，因此**加速档缺席、可选依赖缺席时门照样成立**
   （0014零设施承诺：没有NumPy的机器上一切公开操作必须可用，门也一样）；
3. 静态扫能覆盖到**今天没有任何测试导入过**的模块——覆盖率不是它的前提。

静态扫唯一扫不到的是字符串动态import（`importlib.import_module("...")`）。
那个口子由第五条门堵：物理域模块与耦合模块**一概不许出现动态import机械**。

四条登记规则（表在下面`ENGINE_RINGS`）：

* 物理域模块 → 只许到**同域**与**基座**；不许到另一个域，不许到`couplings`；
* `couplings`（模块位已留，今天不存在）→ 许到任意域与基座。**它是包内唯一
  允许同时依赖多于一个物理域的地方**；
* 基座模块（契约/溯源/验收/场景/编排/模型生成/材料）→ 只许到基座。
  基座不依赖任何上层，这是spec/01第一节"依赖只许向下"的执行面；
* 包内**没有任何模块**允许import `couplings`——组合发生在库之上
  （案例、消费方、examples），不发生在库之内。见spec/15第四节。

完备性（第四条门）是本文件最省心的一条：`src/physics_engine/`下每个模块都
必须在登记表里出现，**新模块不登记即红**。没有这一条，前三条门会随着新模块
悄悄进仓而慢慢空掉——门会一直绿，只是不再管任何事。
"""

from __future__ import annotations

import ast
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "physics_engine"
PACKAGE = "physics_engine"

#: 耦合模块位（spec/15第四节）。**今天不存在**——留位不等于预支实现。
COUPLINGS_RING = "couplings"

#: 物理域登记：域名 → 该域拥有的模块（包名相对`physics_engine`，
#: 子包名即覆盖其一切子模块）。
#:
#: `state`/`integrate`按**spec/12**归力学：这两块是spec/12第二节与第四节的
#: 参考实现，其字段语义（`node*_x_mm`、位置速度二阶系统）是力学的。
#: 有人会想把`state`当成域间通用容器放进基座——**那要走决策记录**，
#: 不许由某个光学模块直接import它来事实上完成升格。宽松方向的改动必须显式。
#:
#: **2026-08-05复核（决策0035）**：确实考虑过把`state`升进基座，**否决**。
#: 理由是本仓的三前提第二条——今天没有任何光学代码需要它，
#: 现在升格就是"为想象中的第三个消费方预支通用性"。
#: 而本门有一个好性质：光学模块一旦import `state`，第①③条门当场红——
#: **升格因此无法悄悄发生**，只能显式走决策。触发条件写明：
#: 某个非力学域真的需要状态容器时重开。
PHYSICS_DOMAINS: dict[str, tuple[str, ...]] = {
    #: `sensors`归力学而不是基座（2026-08-05，决策0035）：它import `state`来校验
    #: "这个通道是不是在读一个自由度"，而`state`是力学的。门当场抓到"基座依赖物理域"
    #: 这个方向反了的错——**我最初把它登记成基座接口，是把愿望当成了事实**。
    #: 今天的`sensors`就是力学传感器，因为它唯一能校验的状态布局是力学的。
    #: 触发条件：第二个域长出自己的状态容器时，`sensors`要么升成域中立
    #: （连带`state`升基座，走决策），要么按域分裂。
    #: `rigidbody`归力学（2026-08-05，决策0041第三节预登记、决策0043落实）：
    #: 它import `state`（力学）、`integrate`（力学）、`energies`（力学，取`MM_PER_M`）
    #: 与`geometry`/`shapes`（基座modelgen）——**没有一条边越到别的域**。
    #: 判据仍是0035那条：import决定环，不是愿望决定环。
    "mechanics": ("energies", "integrate", "rigidbody", "sensors", "solve", "state"),
    "optics": ("optics",),
}

#: 基座登记：spec/01四圈里除物理域圈之外的部分，按其在spec/01第一节的格命名。
#: 基座内部不再分层——本门不裁决"契约能不能import模型生成"，
#: 那是另一条边界，本门只管物理域这一条（不发明未被裁决的规则）。
SUBSTRATE_RINGS: dict[str, tuple[str, ...]] = {
    "contracts": ("__init__", "canonical", "engine_facets", "facets", "identity"),
    "provenance": ("provenance", "run_package"),
    "verification": ("oracles",),
    #: `scene`圈=spec/10那一页的内核接口面。`motion`与`actuators`归这里而不是力学
    #: （2026-08-05，决策0038），判据仍是0035那条：**import决定环，不是愿望决定环**。
    #: 两者的包内import只有`identity`（基座contracts）——位姿来源按spec/12第2.3节
    #: 明确"不是状态"，驱动器的`apply`物理未实现所以够不着`state`。
    #: 对照`sensors`：它import `state`（力学）来判"这一路是不是在读一个自由度"，
    #: 于是只能归力学。同一页的三个接口分在两个圈，正是这条判据在起作用。
    #: 触发条件：`actuators`的`apply`物理落地那天它会import `state`，届时门当场红,
    #: 要么整体改归力学、要么按"声明层留在scene / 物理半边进力学"分裂——两条都走决策。
    "scene": ("actuators", "motion", "scene"),
    "orchestration": ("cli",),
    "modelgen": ("collision", "geometry", "modelgen", "shapes"),
    "materials": ("materials",),
}

#: 动态import机械：静态扫唯一的盲区，因此在物理域与耦合模块里一概禁用。
DYNAMIC_IMPORT_NAMES = frozenset({"import_module", "__import__"})


@dataclass(frozen=True)
class Registry:
    """一份环登记。测试对真实src跑它，"必须红"用例对临时树跑同一份。"""

    package: str
    domains: dict[str, tuple[str, ...]]
    substrate: dict[str, tuple[str, ...]]
    couplings: str = COUPLINGS_RING

    def ring_of(self, module: str) -> tuple[str, str] | None:
        """模块 → `(种类, 名字)`；`种类`取`domain`/`substrate`/`couplings`。

        按**最长前缀**匹配，所以子包名一条登记覆盖其全部子模块。
        没登记返回`None`——由完备性门去炸，不在这里默默放行。
        """

        if not module.startswith(self.package):
            return None
        tail = module[len(self.package) :].lstrip(".")
        head = tail.split(".", 1)[0] if tail else "__init__"
        if head == self.couplings:
            return ("couplings", self.couplings)
        for domain, members in self.domains.items():
            if head in members:
                return ("domain", domain)
        for ring, members in self.substrate.items():
            if head in members:
                return ("substrate", ring)
        return None


ENGINE_RINGS = Registry(package=PACKAGE, domains=PHYSICS_DOMAINS, substrate=SUBSTRATE_RINGS)


@dataclass(frozen=True)
class InternalImport:
    """一条包内import边：谁、在第几行、import了谁。"""

    source: str
    target: str
    lineno: int

    def __str__(self) -> str:
        return f"{self.source}:{self.lineno} → {self.target}"


def discover_modules(package_root: Path, package: str) -> dict[str, Path]:
    """包内全部模块名 → 文件路径。`__init__.py`记在包名下。"""

    modules: dict[str, Path] = {}
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(package_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join([package, *parts])] = path
    return modules


def _package_of(module: str, path: Path) -> str:
    """相对import的锚点：包模块锚在自己身上，普通模块锚在父包。"""

    return module if path.name == "__init__.py" else module.rsplit(".", 1)[0]


def _resolve_relative(anchor: str, level: int) -> str | None:
    parts = anchor.split(".")
    if level - 1 > len(parts):
        return None
    return ".".join(parts[: len(parts) - (level - 1)]) if level > 1 else anchor


def internal_imports(package_root: Path, package: str) -> tuple[InternalImport, ...]:
    """AST静态扫出包内的全部import边（含函数体里的延迟import）。

    `from X import a`里的`a`如果本身是模块就指向`X.a`，否则指向`X`——
    因为`from physics_engine.optics import fts`与
    `from physics_engine.optics.fts import sinc`是同一条依赖边。
    """

    modules = discover_modules(package_root, package)
    known = set(modules)
    edges: list[InternalImport] = []
    for module, path in modules.items():
        anchor = _package_of(module, path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == package or alias.name.startswith(package + "."):
                        edges.append(InternalImport(module, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(anchor, node.level)
                    if node.level
                    else (node.module or "")
                )
                if base is None or not (base == package or base.startswith(package + ".")):
                    continue
                for alias in node.names:
                    candidate = f"{base}.{alias.name}"
                    edges.append(
                        InternalImport(
                            module, candidate if candidate in known else base, node.lineno
                        )
                    )
    return tuple(edges)


def dynamic_import_sites(package_root: Path, package: str) -> tuple[str, ...]:
    """动态import机械的出现位置（`importlib.import_module`/`__import__`）。"""

    found: list[str] = []
    for module, path in discover_modules(package_root, package).items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name in DYNAMIC_IMPORT_NAMES:
                found.append(f"{module}:{node.lineno} {name}(...)")
    return tuple(found)


def cross_domain_edges(
    edges: tuple[InternalImport, ...], registry: Registry
) -> tuple[str, ...]:
    """门①与门②：任一物理域模块import另一物理域模块。"""

    offences: list[str] = []
    for edge in edges:
        source = registry.ring_of(edge.source)
        target = registry.ring_of(edge.target)
        if source is None or target is None:
            continue
        if source[0] == "domain" and target[0] == "domain" and source[1] != target[1]:
            offences.append(
                f"{edge}：物理域{source[1]!r}直接依赖物理域{target[1]!r}"
                "（spec/01第一节：物理域之间互不依赖）"
            )
    return tuple(offences)


def channel_violations(
    edges: tuple[InternalImport, ...], registry: Registry
) -> tuple[str, ...]:
    """门③：跨域往来只许经基座或显式`couplings/`。

    三条一起判：域模块只许到同域+基座；基座不许到域也不许到couplings；
    包内没有任何模块可以import couplings（组合发生在库之上）。
    """

    offences: list[str] = []
    for edge in edges:
        source = registry.ring_of(edge.source)
        target = registry.ring_of(edge.target)
        if source is None or target is None:
            continue
        if target[0] == "couplings" and source[0] != "couplings":
            offences.append(
                f"{edge}：{source[0]}/{source[1]}不得import耦合模块——"
                "耦合是编排层的组合面，被域import就等于把跨域依赖藏进域内"
                "（spec/15第四节）"
            )
            continue
        if source[0] == "domain" and target[0] not in {"domain", "substrate"}:
            offences.append(f"{edge}：物理域只许到同域或基座，落到了{target[0]!r}")
        if source[0] == "substrate" and target[0] == "domain":
            offences.append(
                f"{edge}：基座{source[1]!r}依赖物理域{target[1]!r}——"
                "spec/01第一节'内核不依赖任何上层'反了。"
                "（`physics_engine/__init__.py`落在这条门下：域的公开名不进包门面）"
            )
    return tuple(offences)


def multi_domain_importers(
    edges: tuple[InternalImport, ...], registry: Registry
) -> tuple[str, ...]:
    """门③后半：只有`couplings`可以同时依赖多于一个物理域。"""

    reach: dict[str, set[str]] = {}
    for edge in edges:
        target = registry.ring_of(edge.target)
        if target is not None and target[0] == "domain":
            reach.setdefault(edge.source, set()).add(target[1])
    offences = []
    for module, domains in sorted(reach.items()):
        ring = registry.ring_of(module)
        if len(domains) > 1 and (ring is None or ring[0] != "couplings"):
            offences.append(
                f"{module}同时依赖物理域{sorted(domains)}——"
                f"包内只有{registry.couplings!r}允许这样做"
            )
    return tuple(offences)


def unregistered_modules(package_root: Path, registry: Registry) -> tuple[str, ...]:
    """门④：没登记的模块。登记表不完备，前三条门就会慢慢空掉。"""

    return tuple(
        module
        for module in discover_modules(package_root, registry.package)
        if registry.ring_of(module) is None
    )


# ---------------------------------------------------------------------------
# 真实src上的四条门
# ---------------------------------------------------------------------------


def test_registry_is_not_vacuous():
    """判据本身要被验（轴6规则6）：登记表空了，下面三条门全部假通过。"""

    assert len(ENGINE_RINGS.domains) >= 2, "少于两个物理域时'域间互不依赖'无从谈起"
    assert ENGINE_RINGS.ring_of("physics_engine.energies") == ("domain", "mechanics")
    assert ENGINE_RINGS.ring_of("physics_engine.optics.fts") == ("domain", "optics")
    assert ENGINE_RINGS.ring_of("physics_engine.canonical") == ("substrate", "contracts")
    assert ENGINE_RINGS.ring_of("physics_engine.couplings.thermo_mech") == (
        "couplings",
        "couplings",
    )
    assert ENGINE_RINGS.ring_of("physics_engine.not_registered_yet") is None
    # 登记表里的模块必须真的存在——写幽灵模块进登记表与漏登记是同一种病。
    present = set(discover_modules(PACKAGE_ROOT, PACKAGE))
    for domain, members in ENGINE_RINGS.domains.items():
        for member in members:
            assert f"{PACKAGE}.{member}" in present, f"域{domain}登记了不存在的模块{member!r}"
    for ring, members in ENGINE_RINGS.substrate.items():
        for member in members:
            name = PACKAGE if member == "__init__" else f"{PACKAGE}.{member}"
            assert name in present, f"基座{ring}登记了不存在的模块{member!r}"


def test_gate_one_mechanics_and_optics_do_not_import_each_other():
    """门①：力学（energies/solve/integrate/state）与光学互不import，两个方向都验。"""

    edges = internal_imports(PACKAGE_ROOT, PACKAGE)
    assert edges, "一条包内import都没扫到——扫描器坏了，下面的绿全是假的"
    pairs = [
        (edge, ENGINE_RINGS.ring_of(edge.source), ENGINE_RINGS.ring_of(edge.target))
        for edge in edges
    ]
    forbidden = [
        str(edge)
        for edge, source, target in pairs
        if source is not None
        and target is not None
        and {source[1], target[1]} == {"mechanics", "optics"}
        and source[0] == target[0] == "domain"
    ]
    assert not forbidden, "力学与光学之间出现了直接依赖：\n" + "\n".join(forbidden)


def test_gate_two_no_physics_domain_imports_another():
    """门②：门①的一般式——任意两个物理域之间都不许有直接依赖。

    写成一般式而不是把力学-光学再抄一遍，是为了热域/电磁域进来时
    **不需要改这条门**：登记表加一行，门自动覆盖。
    """

    offences = cross_domain_edges(internal_imports(PACKAGE_ROOT, PACKAGE), ENGINE_RINGS)
    assert not offences, "物理域之间出现直接依赖：\n" + "\n".join(offences)


def test_gate_three_cross_domain_traffic_only_through_substrate_or_couplings():
    """门③：域间只许经基座或显式`couplings/`；基座不依赖域；只有耦合可跨多域。"""

    edges = internal_imports(PACKAGE_ROOT, PACKAGE)
    offences = channel_violations(edges, ENGINE_RINGS) + multi_domain_importers(
        edges, ENGINE_RINGS
    )
    assert not offences, "跨域通道违规：\n" + "\n".join(offences)


def test_gate_four_every_module_is_registered():
    """门④完备性：新模块不登记即红——否则前三条门会随新模块进仓慢慢空掉。"""

    missing = unregistered_modules(PACKAGE_ROOT, ENGINE_RINGS)
    assert not missing, (
        "以下模块没有在ENGINE_RINGS登记，无法判定它属于哪个圈：\n"
        + "\n".join(missing)
        + "\n登记它（并在spec/15第三节的表里同步一行），不要把登记表当可选项。"
    )


def test_gate_five_domains_do_not_use_dynamic_import():
    """门⑤：动态import是静态扫唯一的盲区，物理域与耦合模块里一概禁用。"""

    sites = [
        site
        for site in dynamic_import_sites(PACKAGE_ROOT, PACKAGE)
        for ring in [ENGINE_RINGS.ring_of(site.split(":", 1)[0])]
        if ring is not None and ring[0] in {"domain", "couplings"}
    ]
    assert not sites, (
        "物理域/耦合模块里出现了动态import——它绕过本文件的静态扫：\n" + "\n".join(sites)
    )


def test_scanner_sees_deferred_imports_inside_functions():
    """扫描器自身的门：函数体里的延迟import必须被看见（偷连最常见的形态）。"""

    edges = internal_imports(PACKAGE_ROOT, PACKAGE)
    known = {(edge.source, edge.target) for edge in edges}
    assert ("physics_engine.solve", "physics_engine.energies") in known, (
        "模块顶层的import都没扫到，扫描器坏了"
    )
    # 延迟import的覆盖由下面的临时树用例实测（真实src里今天没有延迟import）。


# ---------------------------------------------------------------------------
# 必须红：每条门各配一个真实违规的临时树
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_copy(tmp_path: Path) -> Path:
    """真实包的一份临时拷贝——"必须红"用例在它上面注入违规，不动真实src。"""

    target = tmp_path / PACKAGE
    shutil.copytree(
        PACKAGE_ROOT, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    return target


def _inject(path: Path, line: str) -> None:
    path.write_text(line + "\n" + path.read_text(encoding="utf-8"), encoding="utf-8")


def test_must_be_red_gate_one_mechanics_imports_optics(engine_copy: Path):
    """力学import光学 → 门①与门②都必须红。"""

    _inject(engine_copy / "energies.py", "from physics_engine.optics import airy_amplitude")
    offences = cross_domain_edges(internal_imports(engine_copy, PACKAGE), ENGINE_RINGS)
    assert offences, "力学直接import光学，门却是绿的"
    assert "mechanics" in offences[0] and "optics" in offences[0]


def test_must_be_red_gate_one_optics_imports_mechanics(engine_copy: Path):
    """反方向同样必须红——只验一个方向的门只挡住一半。"""

    _inject(engine_copy / "optics" / "fts.py", "from physics_engine.state import State")
    offences = cross_domain_edges(internal_imports(engine_copy, PACKAGE), ENGINE_RINGS)
    assert offences, "光学直接import力学，门却是绿的"
    assert "optics" in offences[0] and "mechanics" in offences[0]


def test_must_be_red_gate_two_covers_a_third_domain(engine_copy: Path):
    """门②的一般式：新登记一个域，跨域边照样被抓——不必为它改门。"""

    (engine_copy / "thermal.py").write_text(
        "from physics_engine.optics import airy_amplitude\n", encoding="utf-8"
    )
    registry = Registry(
        package=PACKAGE,
        domains={**PHYSICS_DOMAINS, "thermal": ("thermal",)},
        substrate=SUBSTRATE_RINGS,
    )
    offences = cross_domain_edges(internal_imports(engine_copy, PACKAGE), registry)
    assert offences and "thermal" in offences[0]


def test_must_be_red_deferred_import_inside_a_function(engine_copy: Path):
    """延迟import（写在函数体里，常被说成"避免循环依赖"）必须照样被抓。"""

    path = engine_copy / "optics" / "fts.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\ndef _sneak():\n    from physics_engine.solve import solve_equilibrium\n"
        "    return solve_equilibrium\n",
        encoding="utf-8",
    )
    offences = cross_domain_edges(internal_imports(engine_copy, PACKAGE), ENGINE_RINGS)
    assert offences, "函数体里的延迟import没被抓到——运行时探测的盲区正是这个"


def test_must_be_red_gate_three_substrate_imports_a_domain(engine_copy: Path):
    """基座反向依赖物理域必须红——包门面`__init__.py`导出域公开名就落在这条上。"""

    _inject(engine_copy / "__init__.py", "from physics_engine.optics import airy_amplitude")
    offences = channel_violations(internal_imports(engine_copy, PACKAGE), ENGINE_RINGS)
    assert offences, "包门面导出了光学公开名，门却是绿的"
    assert "__init__" in offences[0] or "physics_engine:" in offences[0]


def test_must_be_red_gate_three_domain_imports_couplings(engine_copy: Path):
    """域import耦合模块必须红——那是把跨域依赖藏进域内的标准手法。"""

    couplings = engine_copy / COUPLINGS_RING
    couplings.mkdir()
    (couplings / "__init__.py").write_text(
        "from physics_engine.optics import airy_amplitude\n"
        "from physics_engine.state import State\n",
        encoding="utf-8",
    )
    _inject(engine_copy / "optics" / "fts.py", "from physics_engine.couplings import x")
    offences = channel_violations(internal_imports(engine_copy, PACKAGE), ENGINE_RINGS)
    assert any("耦合模块" in offence for offence in offences), (
        f"域import了couplings，门却没红：{offences}"
    )


def test_couplings_is_the_only_module_allowed_to_span_domains(engine_copy: Path):
    """门③后半的两面：耦合模块跨多域是**合法的**，别的模块跨多域必须红。"""

    couplings = engine_copy / COUPLINGS_RING
    couplings.mkdir()
    (couplings / "__init__.py").write_text(
        "from physics_engine.optics import airy_amplitude\n"
        "from physics_engine.state import State\n",
        encoding="utf-8",
    )
    edges = internal_imports(engine_copy, PACKAGE)
    assert not multi_domain_importers(edges, ENGINE_RINGS), (
        "耦合模块同时依赖两个域是它的**职责**，不该被判违规"
    )
    assert not cross_domain_edges(edges, ENGINE_RINGS), "耦合模块自身不是物理域，不该触发门②"

    _inject(engine_copy / "scene.py", "from physics_engine.optics import airy_amplitude")
    _inject(engine_copy / "scene.py", "from physics_engine.state import State")
    offences = multi_domain_importers(internal_imports(engine_copy, PACKAGE), ENGINE_RINGS)
    assert offences and "scene" in offences[0]


def test_must_be_red_gate_four_new_module_is_not_registered(engine_copy: Path):
    """新模块不登记必须红——这条门是前三条门不会慢慢空掉的唯一保证。"""

    (engine_copy / "acoustics.py").write_text("VALUE = 1\n", encoding="utf-8")
    missing = unregistered_modules(engine_copy, ENGINE_RINGS)
    assert missing == ("physics_engine.acoustics",)


def test_must_be_red_gate_five_dynamic_import_in_a_domain(engine_copy: Path):
    """动态import必须红——它是静态扫的盲区，所以在域内直接禁用整个机械。"""

    path = engine_copy / "optics" / "fts.py"
    path.write_text(
        "import importlib\n\n\ndef _sneak():\n"
        '    return importlib.import_module("physics_engine.solve")\n\n\n'
        + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    sites = [
        site
        for site in dynamic_import_sites(engine_copy, PACKAGE)
        for ring in [ENGINE_RINGS.ring_of(site.split(":", 1)[0])]
        if ring is not None and ring[0] in {"domain", "couplings"}
    ]
    assert sites and "import_module" in sites[0]
