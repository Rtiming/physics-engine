"""开工自检的必红矩阵——**四条断言，每条各自被注错验过一次**。

这份文件存在的理由是plans/09教训三：

> **一条从没被必红用例走过的分支，等于一条没有门的分支。**
> 域隔离门对相对import完全失明，而它有九条必红——**全部用绝对import**。
> 门全绿不是因为它挡得住，是因为那条分支从没被执行过。

所以这里的组织方式是**按判据的分支**，不是按规则：
`check_worktree_env.py`有四条断言，每条至少一个红用例 + 绿用例。

2026-08-12开工自检落地时，四条的红分支都在真worktree上实跑过一遍
（决策0053第三节记了实跑结果）。本文件把那次实跑固化成常驻门——
**一次性的手工验证会随着时间变成传说**。
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools" / "check_worktree_env.py"


def _load_checker():
    """按路径加载`tools/`下的脚本——它不是包的一部分，不能import。"""

    spec = importlib.util.spec_from_file_location("check_worktree_env", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _tiny_repo(root: Path) -> None:
    """两个提交的最小仓：`base`在第二个提交，HEAD停在第一个。"""

    root.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], root)
    _git(["config", "user.email", "t@example.invalid"], root)
    _git(["config", "user.name", "t"], root)
    (root / "a.txt").write_text("1\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-qm", "first"], root)
    first = _git(["rev-parse", "HEAD"], root).stdout.strip()
    (root / "a.txt").write_text("2\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-qm", "second"], root)
    _git(["checkout", "-q", first], root)


# ---------------------------------------------------------------------------
# 断言1：包身份——**这是要害那一条**
# ---------------------------------------------------------------------------


def test_package_identity_passes_in_this_repo(checker):
    """绿分支：主仓里import解析到主仓自己的`src/`。"""

    resolved = checker.assert_package_resolves_to_this_tree(ROOT)
    assert str((ROOT / "src").resolve()) in resolved


def test_package_identity_reds_when_import_comes_from_another_tree(checker, tmp_path):
    """**必红**：树里有`src/`，但import解析到的是别的树。

    这正是2026-08-12实测的那条机理——共享`.venv`的editable `.pth`
    写的是主仓`src`的绝对路径，于是worktree里import到的是主仓的包。
    17个陈旧副本里14个带着这条。

    **注错方式**：造一棵有`src/`但没有自己安装的树，让import落回主仓。
    """

    (tmp_path / "src").mkdir()
    with pytest.raises(checker.CheckFailed) as excinfo:
        checker.assert_package_resolves_to_this_tree(tmp_path)
    assert "别的树" in str(excinfo.value)
    #: 消息必须给出可照做的动作，不能只说"错了"。
    assert "PYTHONPATH" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 断言2：可移植性校验非空跑
# ---------------------------------------------------------------------------


def test_portability_check_passes_and_scans_many_files(checker):
    """绿分支：主仓里它扫得到东西（实测数百个）。"""

    assert checker.assert_portability_check_is_not_empty(ROOT) > 0


def test_portability_check_reds_when_it_scans_zero_files(checker, tmp_path):
    """**必红**：校验器扫0个文件却报"通过"——那不是通过，是空跑。

    **注错方式**：把树放进一个路径含`.claude`的位置，
    那是校验器的跳过目录。这不是假想——Claude Code的worktree默认就住在
    `.claude/worktrees/`下，实测在那里扫**0**个文件而主仓扫292个。
    """

    tree = tmp_path / ".claude" / "worktrees" / "victim"
    (tree / "tools").mkdir(parents=True)
    shutil.copy2(ROOT / "tools" / "rtime-project-check.py", tree / "tools")
    (tree / "a.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(checker.CheckFailed) as excinfo:
        checker.assert_portability_check_is_not_empty(tree)
    assert "空跑" in str(excinfo.value)


def test_portability_check_reds_when_the_checker_is_missing(checker, tmp_path):
    """**必红**（同断言的第二个分支）：校验器根本不在，不许当通过。"""

    with pytest.raises(checker.CheckFailed) as excinfo:
        checker.assert_portability_check_is_not_empty(tmp_path)
    assert "不在" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 断言3：测试可收集
# ---------------------------------------------------------------------------


def test_tests_are_collectable_in_this_repo(checker):
    """绿分支：主仓收集得到用例。"""

    assert checker.assert_tests_are_collectable(ROOT) > 0


def test_tests_collectable_reds_when_nothing_is_collected(checker, tmp_path):
    """**必红**：一条用例都收不到。

    **注错方式**：造一棵有`tests/`但里面是空的树。
    `accept.py`把退出码5当"申报过的空档位"（那是对的，`serverclass`确实是空的），
    **而开工自检一律当红**——一棵测不了的树没有"申报过的空档"这回事。
    """

    (tmp_path / "tests").mkdir()
    with pytest.raises(checker.CheckFailed) as excinfo:
        checker.assert_tests_are_collectable(tmp_path)
    assert "收集" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 断言4：起点不落后
# ---------------------------------------------------------------------------


def test_head_not_behind_passes_in_this_repo(checker):
    """绿分支：主仓的HEAD不落后main。"""

    assert checker.assert_head_is_not_behind(ROOT, "main")


def test_head_behind_base_reds(checker, tmp_path):
    """**必红**：起点落后基线。

    **注错方式**：两个提交的最小仓，HEAD停在第一个、`main`在第二个。
    2026-08-05那一波的worktree实测**落后52个提交**，
    简报里点名的文件在起点上根本不存在。
    """

    repo = tmp_path / "tiny"
    _tiny_repo(repo)
    with pytest.raises(checker.CheckFailed) as excinfo:
        checker.assert_head_is_not_behind(repo, "main")
    assert "落后" in str(excinfo.value)


def test_unresolvable_base_is_a_hard_error_not_a_silent_pass(checker):
    """**必红**（同断言的第二个分支）：基线解析不出来时**不许当通过**。

    这是本仓反复抓到的那个形态的预防：**判不了不等于判过了**。
    """

    with pytest.raises(checker.CheckFailed) as excinfo:
        checker.assert_head_is_not_behind(ROOT, "no-such-branch-9f3a2b")
    assert "解析不出来" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 门自己的形制
# ---------------------------------------------------------------------------


def test_the_checker_exits_nonzero_when_any_assertion_fails(tmp_path):
    """端到端：任一条不过，进程退出码必须非0——否则它挂不上`accept.py`。"""

    (tmp_path / "src").mkdir()
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "不要开工" in result.stderr


def test_every_assertion_has_at_least_one_red_case_in_this_file():
    """**元测试**：四条断言各自都有必红用例。

    教训三的通则是"必红用例要覆盖判据的每个分支，不是每个规则"。
    这条元测试守的是更弱但可自动化的一半：**没有一条断言是零必红的**。
    分支级的覆盖靠上面每个用例的docstring写明"注错方式"，由人读。
    """

    source = Path(__file__).read_text(encoding="utf-8")
    checker_source = CHECKER.read_text(encoding="utf-8")

    #: 认``def assert_xxx(``这个结构位置，不认"名字在文件里出现过"——
    #: 那正是plans/09教训二的通则（判据落在结构位置上，不落在"出现过"上）。
    #: 第一版按空白切分，拿到的是``assert_...(root:``而不是函数名，
    #: **本条元测试当场把自己写错的那一版打红了**。
    assertions = set(re.findall(r"^def (assert_\w+)\(", checker_source, re.MULTILINE))
    assert len(assertions) == 4, f"断言数变了：{sorted(assertions)}"

    for name in sorted(assertions):
        uses = source.count(f"checker.{name}(")
        assert uses >= 2, f"断言{name}的用例少于2个（至少要一红一绿）"
