"""`tools/view/`的身份边界门（决策0076，承0074第5.3／5.4节）。

与`test_model_tools_stay_out_of_the_kernel.py`**同源不同料**：那道门守的是
"内核不import`tools/`下的任何东西"（顶层名字`tools`），本门守的是
**具体那个重依赖的名字**——`rerun`。

## 为什么这两道门必须都在

它们防的不是同一个失手：

* 那道门看的是**import路径**。它挡住"内核`from tools.view import replay`"；
* 本门看的是**依赖名字**。它挡住"有人把`rerun`直接写进`src/`"——
  那种写法根本不经过`tools.view`这个名字，**上一道门一个字都看不见**。

`validation/`那条自检（决策0025第一节第2条：`grep -rn "import fcl" src/`必须零命中）
是本门的原型。rerun比FCL更需要它：rerun的wheel实测**133.6 MB**，
而`pyproject.toml`的`dependencies = []`是0014立的承诺。

## 第四条判据在守一件别的事

`trace_from_closed_loop.py`（产轨迹的一侧）**不许import rerun**。
这不是洁癖，是0076那条"两个环境互不认识对方的依赖、中间只有一份JSON"的**可执行形式**：
它一旦破了，"装不上rerun的机器照样能产轨迹"这句话就是假的，
而**没有任何别的地方会红**——那条承诺只活在README里。

## 同行的反面证据

Drake issue #21868：只是构建`MultibodyPlant`就把vtk_internals、X11、OpenGL、GLX
全拉进来了（0073第七节第2条）。那正是"viewer与求解器同住一仓而边界没有门守着"的账。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src/physics_engine"
TOOLS_VIEW = ROOT / "tools/view"
PYPROJECT = ROOT / "pyproject.toml"

#: 查看器档的重依赖顶层名字。`rerun_sdk`是发行名，`rerun`是import名，
#: `rerun_bindings`是它的原生扩展——**三个都挡**，只挡一个等于留了两扇门。
VIEW_DEPENDENCY_ROOTS = frozenset({"rerun", "rerun_sdk", "rerun_bindings"})

#: 产轨迹那一侧。它只许认识`physics_engine`与标准库。
PRODUCER_SIDE = TOOLS_VIEW / "trace_from_closed_loop.py"


def imported_roots(source: str) -> set[str]:
    """一段源码里所有import的**顶层**名字。静态扫，不执行。

    与`test_model_tools_stay_out_of_the_kernel.py`的同名逻辑形制一致；
    这里收字符串而不是路径，**正是为了让必红用例能喂进植入的源码**。
    """

    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            #: 相对import（`level > 0`）的`module`不是顶层名字，跳过。
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def _python_files(root: Path) -> list[Path]:
    """**只扫git跟踪的`.py`**，不扫整棵目录树。

    第一版用`rglob`扫全仓，**在主仓当场就是红的**（2026-08-18对抗审核实测）：
    本仓的多代理机制把worktree副本建在`.claude/worktrees/`下，
    副本里那份`tools/view/replay.py`当然import rerun，于是门把**自己的产物**判成了违规。

    这不是加个豁免就完的事——`rglob`扫到的东西里有多少不属于"本仓的源码"
    （`.git`的钩子样例、别人的worktree、临时解包目录）本来就说不清。
    **改成问git"哪些文件是这个仓的"**，判据于是落在一个结构位置上而不是一条路径黑名单上，
    这正是plans/09教训二的通则。

    `git ls-files`拿不到时（不是git仓、或git不在）**失败关闭**：
    一道扫不到文件的门与没有门等价，而"0个文件"在本仓已经冒充过一次通过
    （`rtime-project-check`那条，plans/09第六节第4条）。
    """

    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "*.py"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "`git ls-files`拿不到跟踪文件清单，这道门无法确定它该扫什么 —— "
            "失败关闭，不退回rglob（退回去就是把已经红过的那条路又走一遍）"
        )
    names = [name for name in completed.stdout.decode().split("\0") if name]
    if not names:
        raise AssertionError("git跟踪的`.py`文件为0个 —— 这不是通过，是空跑")
    return [root / name for name in names]


def offending_files(roots: frozenset[str], paths: list[Path]) -> list[str]:
    """判据本体：这些文件里哪些import了禁名。**必红用例喂它假路径。**"""

    return [
        f"{path.relative_to(ROOT)} → {sorted(roots & imported_roots(path.read_text(encoding='utf-8')))}"
        for path in paths
        if roots & imported_roots(path.read_text(encoding="utf-8"))
    ]


def declared_dependency_names(pyproject_text: str) -> set[str]:
    """`dependencies`与**每一个**`optional-dependencies`分组里声明的发行名。

    两处一起收是要害：0074第5.4节写的是"不进`dependencies`"，
    而0025第一节第1条把它说全了——"**连`[optional-dependencies]`都不进**"。
    只查前者会漏掉`[project.optional-dependencies] view = ["rerun-sdk"]`这一种写法，
    而那种写法一样会让`pip install physics-engine[view]`拖进133.6 MB。
    """

    data = tomllib.loads(pyproject_text)
    project = data.get("project", {})
    specs = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    names = set()
    for spec in specs:
        #: 取到第一个版本/标记符号为止，再把`-`归一成`_`（发行名与import名的差）。
        name = spec.split(";")[0].strip()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", " "):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("-", "_"))
    return names


# ------------------------------------------------------------------ 判据 ---


def test_the_kernel_never_imports_the_viewer_dependency():
    """**边界2**：`src/physics_engine`永不import rerun。一行都不许。"""

    offences = offending_files(VIEW_DEPENDENCY_ROOTS, _python_files(PACKAGE_ROOT))
    assert not offences, (
        "内核import了查看器档的重依赖：\n" + "\n".join(offences) + "\n"
        "——rerun的wheel实测133.6 MB，而`dependencies = []`是0014立的承诺。"
        "这条一破，零依赖承诺就换掉了（0074第5.4节／0025第一节第2条）。"
    )


def test_the_viewer_dependency_never_enters_pyproject():
    """**边界1**：不进`dependencies`，**连`[optional-dependencies]`都不进**。"""

    declared = declared_dependency_names(PYPROJECT.read_text(encoding="utf-8"))
    offences = sorted(declared & VIEW_DEPENDENCY_ROOTS)
    assert not offences, (
        f"`pyproject.toml`里声明了查看器档的依赖：{offences}\n"
        "——0025第一节第1条的原文是'连`[optional-dependencies]`都不进'。"
        "加速档那条口子（0016给NumPy开的）是为**本仓自己的数值内核**开的，不是给查看器的。"
    )


def test_the_viewer_dependency_lives_only_under_tools_view():
    """**边界3**：全仓只有`tools/view/`认识rerun。

    这一条比前两条宽——它连`tests/`、`cases/`、`benchmarks/`、`validation/`一起扫。
    理由：**重依赖会顺着任何一条路爬进CI**。一个在`tests/`里`import rerun`的用例，
    会让"跑本仓的门需要装什么"这件事悄悄变掉，
    而`dependencies`那道上限恒为0的门**看不见它**（它测的是pyproject，不是import图）。
    """

    scanned = [
        path
        for path in _python_files(ROOT)
        if TOOLS_VIEW not in path.parents and path.parent != TOOLS_VIEW
    ]
    #: 本文件自己会提到这些名字，但只在**字符串**里，不在import里——
    #: 判据扫的是AST的import节点，所以本文件不会自己撞上自己。
    offences = offending_files(VIEW_DEPENDENCY_ROOTS, scanned)
    assert not offences, (
        "`tools/view/`之外出现了rerun的import：\n" + "\n".join(offences) + "\n"
        "——重依赖会顺着任何一条路爬进CI。"
        "要看轨迹就用`tools/view/.venv`跑`replay.py`，不要把rerun请进本仓的门。"
    )


def test_the_trace_producer_never_imports_the_viewer_dependency():
    """**第四条**：产轨迹的一侧不认识rerun（0076第三节）。

    "装不上rerun的机器照样能产轨迹"——这句话的可执行形式就是这一条。
    """

    assert PRODUCER_SIDE.exists(), f"产轨迹那一侧不见了：{PRODUCER_SIDE}"
    roots = imported_roots(PRODUCER_SIDE.read_text(encoding="utf-8"))
    offences = sorted(VIEW_DEPENDENCY_ROOTS & roots)
    assert not offences, (
        f"`{PRODUCER_SIDE.name}`import了{offences} ——\n"
        "它是**产**轨迹的一侧，只许认识`physics_engine`与标准库。"
        "这一条一破，'两个环境互不认识对方的依赖、中间只有一份JSON'就只是README里的一句话了。"
    )


# -------------------------------------------------------------- 必红用例 ---
#
# 每条判据配一条。**一个永远绿的门与没有门是同一件事，而且更坏，
# 因为它让人以为有人在看**（`tools/model/`那道门的原话）。


def test_the_import_criterion_would_go_red_on_a_planted_import(tmp_path):
    """必红①③④：植入的import必须被`offending_files`看见。

    三种写法各来一遍——**只挡`import rerun`会漏掉另外两种**，
    而`from rerun.blueprint import ...`恰恰是查看器代码最常见的那一种。
    """

    for index, planted in enumerate((
        "import rerun as rr\n",
        "from rerun import Scalars\n",
        "from rerun.blueprint import Blueprint\n",
        "import rerun_bindings\n",
    )):
        victim = tmp_path / f"planted_{index}.py"
        victim.write_text(planted, encoding="utf-8")
        assert VIEW_DEPENDENCY_ROOTS & imported_roots(planted), (
            f"植入的import没有被判据看见：{planted!r}"
        )


def test_the_import_criterion_does_not_fire_on_mere_mentions():
    """必红①的**反面**：判据不许把字符串里的"rerun"当成import。

    一条会误报的判据与一条不报的判据一样坏——它会被人加豁免，
    而豁免一旦开口子就再也关不上。本文件的docstring里"rerun"出现十几次，
    **判据必须一次都不响**。
    """

    innocent = '"""这段docstring里写满了rerun。"""\nrerun = "rerun"\nimport json\n'
    assert not (VIEW_DEPENDENCY_ROOTS & imported_roots(innocent)), (
        "判据把字符串/变量名里的rerun当成了import —— 它扫的应该是AST的import节点"
    )


def test_the_pyproject_criterion_would_go_red_on_a_planted_optional_dependency():
    """必红②：**关键是`[optional-dependencies]`那一半**。

    只查`dependencies`的实现会在这条上绿——而那正是0025第一节第1条
    专门点名的那个口子。
    """

    planted_runtime = '[project]\nname = "x"\ndependencies = ["rerun-sdk==0.34.1"]\n'
    assert VIEW_DEPENDENCY_ROOTS & declared_dependency_names(planted_runtime), (
        "植进`dependencies`的rerun-sdk没有被判据看见"
    )

    planted_optional = (
        '[project]\nname = "x"\ndependencies = []\n'
        '[project.optional-dependencies]\nview = ["rerun-sdk>=0.34"]\n'
    )
    assert VIEW_DEPENDENCY_ROOTS & declared_dependency_names(planted_optional), (
        "植进`[optional-dependencies]`的rerun-sdk没有被判据看见 —— "
        "**这正是0025点名的那个口子**：`pip install physics-engine[view]`一样拖进133.6 MB"
    )

    #: 发行名`rerun-sdk`与import名`rerun`不同，归一化少一步就漏。
    assert "rerun_sdk" in declared_dependency_names(planted_runtime), (
        "`-`没有被归一成`_` —— 发行名与import名的差是这条判据最容易漏的地方"
    )


def test_the_pyproject_criterion_stays_quiet_on_todays_pyproject():
    """必红②的**反面**：今天的`pyproject.toml`必须是干净的，而且判据得真的读到了东西。

    **这一条防的是空跑**：一个把TOML读成空字典的实现会让上面那条判据永远绿。
    所以这里不判"没有rerun"（那是判据本身的事），判的是
    **判据确实解析出了本仓已知的那几个dev依赖**。
    """

    declared = declared_dependency_names(PYPROJECT.read_text(encoding="utf-8"))
    assert {"pytest", "ruff", "numpy"} <= declared, (
        f"判据没解析出本仓已知的dev依赖，实际拿到：{sorted(declared)} —— "
        "**读到空集的判据永远绿**，那比没有判据更坏"
    )


# ---------------------------------------------------------------- 动态import
#: **静态判据有一个说得清的盲区**：它扫的是AST的`Import`/`ImportFrom`节点，
#: 而`importlib.import_module("re" + "run")`、`__import__("rerun")`、
#: `exec("import rerun")`在AST里都是`Call`，一个都不命中。
#: 2026-08-18对抗审核实测四种写法全部能让上面四条判据保持绿。
#:
#: 补法不是把静态判据写得更花（那是追着绕过手法跑，永远慢一步），
#: 而是**换一把量的是结果的尺子**：真的import一次内核，看`sys.modules`里有没有它。
#: 静态那条管"源码长什么样"，这条管"最终发生了什么"——**后者绕不过去**。
def modules_after_importing(package: str, path_entry: Path | None = None) -> frozenset[str]:
    """在**干净子进程**里import`package`，返回事后`sys.modules`的顶层名字集合。

    必须是子进程：本进程早就import过`physics_engine`，在这里查`sys.modules`
    量到的是**测试自己的**依赖闭包，那正好是一条量错了对象的判据。
    """

    code = (
        "import sys, json\n"
        f"__import__({package!r})\n"
        "print(json.dumps(sorted({name.split('.', 1)[0] for name in sys.modules})))"
    )
    environment = dict(os.environ)
    if path_entry is not None:
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{path_entry}{os.pathsep}{existing}" if existing else str(path_entry)
        )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        raise AssertionError(f"子进程import`{package}`失败：{completed.stderr[-2000:]}")
    return frozenset(json.loads(completed.stdout))


def test_importing_the_kernel_never_pulls_the_viewer_dependency_at_runtime():
    """**结果尺**：import完`physics_engine`，`sys.modules`里不许出现禁名。

    这条挡的是静态判据挡不住的那一族（动态import），
    而且它**顺带守住了将来任何形式的"绕过"**——不管源码写成什么样，
    只要跑完内核之后rerun在`sys.modules`里，它就红。
    """

    loaded = modules_after_importing("physics_engine")
    assert not (VIEW_DEPENDENCY_ROOTS & loaded), (
        f"import physics_engine之后`sys.modules`里出现了{sorted(VIEW_DEPENDENCY_ROOTS & loaded)}"
        " —— 内核在运行时拉进了查看器依赖"
    )


def test_the_runtime_criterion_catches_a_dynamic_import_the_ast_one_misses(tmp_path):
    """必红：一个**只用动态import**的包，静态判据看不见、运行时判据必须看见。

    用`xml.dom`当替身禁名（标准库、几乎没人会顺带import进来），
    这样这条用例不依赖rerun装没装。
    """

    stub = tmp_path / "dynamic_stub"
    stub.mkdir()
    (stub / "__init__.py").write_text(
        'import importlib\n_pulled = importlib.import_module("xml" + ".dom")\n',
        encoding="utf-8",
    )
    source = (stub / "__init__.py").read_text(encoding="utf-8")

    #: 静态判据对它**完全失明**——这一行就是那个盲区的证据，不是推测。
    assert "xml" not in imported_roots(source)
    #: 运行时判据抓得住。
    assert "xml" in modules_after_importing("dynamic_stub", path_entry=tmp_path)
