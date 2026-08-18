"""接触分层的方向门——research/17第六节那条结论第一次有牙齿（plans/16的M7）。

## 这道门判什么，以及它为什么不能由域隔离门代劳

`test_domain_isolation.py`判的是**跨物理域**的边（力学不许import光学）。
`rigidbody`、`rotation`、`contact`、`contact_dynamics`、`contact_pipeline`
**全都在力学域**，于是它们之间怎么连、连成什么方向，那道门一个字都不管。

而research/17第六节实测的同行分层是有方向的：

    broad（AABB重叠）→ narrow（法向/穿透/杆臂）→ 力矩装配 → 积分器回调

其中最要紧的一句是：**积分器不该知道接触**。
Bullet的`btRigidBody::setAngularFactor`把"要不要转动参与"这个开关挂在
**积分响应层**，不挂在窄相层；`btSphereSphereCollisionAlgorithm.cpp`全文件
没有一次`getRotation`调用。**本仓今天符合这条**——本文件把"今天符合"
变成"以后也必须符合"。

没有这道门，一行`from physics_engine.contact import ...`写进`rigidbody.py`
不会让任何东西变红：域隔离门绿（同域）、公开面门绿（模块都登记着）、
全套物理判据绿（那一行只是让积分器多认识了一个模块）。
**而它一旦写下去，"积分器不知道接触"这句话就永久地不再成立，且无人知道。**

## 它**不**判什么

不判"接触产出的几何量够不够"。`contact_pipeline`今天不产法向也不产杆臂，
于是"接触→力矩装配"那条边是**断的**——那是decisions/0083第六节的裁决
（今天不接，理由与触发条件在那里），**不是这道门的事**。
断链不是错误方向，本门只管方向。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "physics_engine"

#: 积分器一侧：这些模块**不许**认识接触。
#: `rigidbody`是刚体积分器本体；`integrate`是通用积分器；`rotation`是准静态转动
#: 参数化——三者都在接触的**上游或旁边**，不在它的下游。
INTEGRATOR_SIDE: tuple[str, ...] = ("rigidbody", "integrate", "rotation")

#: 接触一侧的模块名前缀。`contact`写成前缀，覆盖`contact/`整个子包。
CONTACT_SIDE: tuple[str, ...] = ("contact", "contact_dynamics", "contact_pipeline")


def _module_files(name: str) -> list[Path]:
    """一个登记名对应的全部源文件（子包名覆盖其一切子模块）。"""

    single = PACKAGE_ROOT / f"{name}.py"
    if single.exists():
        return [single]
    package = PACKAGE_ROOT / name
    if package.is_dir():
        return sorted(package.rglob("*.py"))
    raise AssertionError(f"登记名{name!r}在src下既不是模块也不是子包——登记表过期了")


def _internal_imports(path: Path) -> list[tuple[int, str]]:
    """``(行号, 被import的包内模块全名)``。

    静态扫，与域隔离门同源同理由：**函数体里的延迟import也要看得见**
    （"偷连"最常见的形态恰恰是写在函数里、说是"避免循环依赖"）。
    """

    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                #: 相对import：本仓禁用（域隔离门第⑤条已判），这里保守地记成
                #: "指向本包某处"，让下面的判定去看它落在谁身上。
                found.append((node.lineno, f"physics_engine.{node.module or ''}"))
            elif node.module and node.module.startswith("physics_engine"):
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("physics_engine"):
                    found.append((node.lineno, alias.name))
    return found


def _head(module: str) -> str:
    tail = module[len("physics_engine") :].lstrip(".")
    return tail.split(".", 1)[0] if tail else "__init__"


@pytest.mark.parametrize("name", INTEGRATOR_SIDE)
def test_the_integrator_side_never_imports_contact(name):
    """**积分器不该知道接触**（research/17第六节）。

    这一条是方向门的正面。反面（"接触可以认识积分器"）由下一条守着——
    两面都要有，只判一面的门会把"方向反了"和"根本没有这条边"混为一谈。
    """

    offenders = [
        (path.relative_to(ROOT), lineno, module)
        for path in _module_files(name)
        for lineno, module in _internal_imports(path)
        if _head(module) in CONTACT_SIDE
    ]
    assert not offenders, (
        f"{name}认识了接触：{offenders} —— research/17第六节实测的同行分层是"
        "「接触产出几何量 → 力矩装配 → 积分器回调」，**积分器不该知道接触**。"
        "要开这条边先走决策记录（decisions/0083第六节写了触发条件）"
    )


def test_the_contact_side_is_allowed_to_know_the_integrator():
    """反面：方向是**接触→积分器**，不是没有边。

    `contact_dynamics`（力矩装配层）就是靠import`rigidbody`拿到布局、`cross`
    与两个方向的姿态换算的。**这条边必须在**——它要是没了，说明装配层
    要么被并进了积分器（那就等于积分器知道了接触），
    要么自己重新发明了一套姿态换算（那是第二条真相源）。
    """

    edges = {
        _head(module)
        for path in _module_files("contact_dynamics")
        for _, module in _internal_imports(path)
    }
    assert "rigidbody" in edges, (
        "`contact_dynamics`不再import`rigidbody`了——力矩装配层要么被并掉了、"
        "要么自己重新发明了一套姿态换算。两种都是决策记录级的改动"
    )


def test_must_be_red_a_contact_import_inside_the_integrator_is_caught(tmp_path):
    """必红：把一行接触import写进积分器一侧，本门必须抓到——**包括写在函数体里的**。

    这一条对一棵临时树跑同一份扫描逻辑。它守的是扫描器本身：
    一个只看文件头几行、或者只认模块级import的实现，在真实src上一样全绿。
    """

    fake = tmp_path / "physics_engine"
    fake.mkdir()
    (fake / "rigidbody.py").write_text(
        "from physics_engine.state import State\n"
        "\n"
        "\n"
        "def make():\n"
        "    from physics_engine.contact_pipeline import SphereContactPipeline\n"
        "    return SphereContactPipeline\n",
        encoding="utf-8",
    )
    hits = [
        (lineno, module)
        for lineno, module in _internal_imports(fake / "rigidbody.py")
        if _head(module) in CONTACT_SIDE
    ]
    assert hits == [(5, "physics_engine.contact_pipeline")], (
        f"扫描器没抓到函数体里那一行：{hits!r}"
    )


def test_the_registry_covers_every_module_it_names():
    """登记表不许指向不存在的模块——**过期的登记表会让门静默变空**。

    与域隔离门第④条同源：那道门守的是"每个模块都要被登记"，
    这一条守的反方向"每条登记都要指得到东西"。
    """

    for name in INTEGRATOR_SIDE + CONTACT_SIDE:
        assert _module_files(name), f"{name}在src下找不到"
