"""`tools/model/`的身份边界门（决策0073第四节）。

这道门与`validation/`那条自检**逐字同源**（决策0025第一节第2条：
`grep -rn "import fcl" src/`必须零命中）——不是新纪律，
是把一条已经过门的纪律复用到第二个目录上。

**为什么这道门要存在**：0073裁决"工具住仓内、依赖住仓外"的**全部安全性**
就压在"工具永不被内核import"这一条上。它一旦被破，
`tools/model/`将来加的任何重依赖（OCCT、gmsh）就成了内核的传递依赖，
而`runtime_dependencies`那道上限恒为0的门**看不见它**——
因为那道门测的是`pyproject.toml`的`dependencies`，不是import图。

**同行的反面证据**（0073第七节第2条）：Drake issue #21868记着
"只是构建`MultibodyPlant`就把vtk_internals、X11、OpenGL、GLX全拉进来了"——
那是viewer与网格解析住在求解器仓里、而边界没有门守着的账。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src/physics_engine"
TOOLS_MODEL = ROOT / "tools/model"

#: 内核里出现这些名字之一即红。`tools`整个前缀都挡——
#: 内核不该import`tools/`下的**任何**东西，不只是`tools/model/`。
FORBIDDEN_ROOTS = frozenset({"tools"})


def _internal_source_files() -> list[Path]:
    return [
        path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _imported_roots(path: Path) -> set[str]:
    """一个源文件里所有import的**顶层**名字。静态扫，不执行。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            #: 相对import（`level > 0`）的`module`不是顶层名字，跳过。
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def test_the_kernel_never_imports_the_model_tools():
    """**这道门是0073那条裁决的全部安全性所在。**"""

    offences = [
        f"{path.relative_to(ROOT)} → {sorted(FORBIDDEN_ROOTS & _imported_roots(path))}"
        for path in _internal_source_files()
        if FORBIDDEN_ROOTS & _imported_roots(path)
    ]
    assert not offences, (
        "内核import了`tools/`下的东西：\n" + "\n".join(offences) + "\n"
        "——0073裁决'工具住仓内、依赖住仓外'的安全性全压在这一条上。"
        "破了它，`tools/model/`将来的任何重依赖都会成为内核的传递依赖，"
        "而`runtime_dependencies`那道门看不见（它测的是pyproject的dependencies，不是import图）。"
    )


def test_the_gate_would_go_red_on_a_planted_import():
    """必须红：这道门自己得能红。

    判据本身也要被验（`tests/governance/`是样板）——一个永远绿的门
    与没有门是同一件事，**而且更坏，因为它让人以为有人在看**。
    """

    planted = ast.parse("from tools.model import mesh_aabb\nimport tools\n")
    roots: set[str] = set()
    for node in ast.walk(planted):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert FORBIDDEN_ROOTS & roots, "植入的import没有被这条判据看见"


def test_the_model_tools_are_pure_standard_library_today():
    """今天`tools/model/`的两个工具是**纯标准库**的（0073第四节理由2）。

    **这一条不是禁令，是登记**：将来的`step_extract/`按裁决可以有重依赖，
    但它必须住在自己的子目录里、装进`tools/model/.venv`。
    所以本条只判**顶层的`.py`**——子目录不在判据内，
    而那正是"重依赖档独立venv"这条边界在测试里的形状。

    真正的禁令是上一条（内核不许import它们）。
    """

    third_party: list[str] = []
    standard = {
        "argparse", "hashlib", "json", "struct", "sys", "pathlib",
        "__future__", "collections", "dataclasses", "math", "csv", "typing",
        "itertools", "functools", "re", "os",
    }
    for path in sorted(TOOLS_MODEL.glob("*.py")):
        for root in _imported_roots(path):
            if root not in standard:
                third_party.append(f"{path.relative_to(ROOT)} → {root}")
    assert not third_party, (
        "`tools/model/`顶层出现了非标准库import：\n" + "\n".join(third_party) + "\n"
        "——重依赖要住进自己的子目录并装进`tools/model/.venv`（0073第四节边界3）。"
    )
