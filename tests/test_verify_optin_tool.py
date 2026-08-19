"""选择进入档统一入口的门（`tools/verify_optin.py`，2026-08-18立）。

## 它守什么

本仓有三条"指了才跑、不指明示skip"的通道（决策0073：真实资产永不进仓），
而在这个入口之前**没有任何一条命令能把它们一起跑起来**——
于是这条通道的整体状态从来没有被看过一眼，**而skip在报告里看起来和pass一样绿**。

同一天实测到的两个直接后果，都是"没人整体跑过"才活得下来的：

1. `PE_REAL_CENTERLINE_CSV`在两侧有**互不兼容的两套约定**，
   指目录一侧硬错、指单文件另一侧硬错——**不存在一个值能让这条通道全部跑过**；
2. 发现器只按名字匹配时，`**/output`在这台机器上按字母序第一个命中的是
   **另一个项目的操作台输出目录**，测试红在"no run directories under …"上，
   **报错说的是语料不对，而真因是发现器挑错了树**。

## 三条判据

1. **发现之后要验**：只匹配名字不算数，候选底下得真有那种东西；
2. **空跑不是通过**：一条都没解析到时返回码非零；
3. **排序取第一个**：同一棵树上两次跑必须选中同一个，否则这条通道自己就不确定。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.verify_optin import CHANNELS, discover, main, resolve  # noqa: E402


def _tree(root: Path, relative: str, leaf: str = "") -> Path:
    made = root / relative
    made.mkdir(parents=True, exist_ok=True)
    if leaf:
        target = made / leaf
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    return made


def test_the_channel_table_is_not_empty():
    """空表会让下面每一条参数化门**一条都不跑**而全绿。"""

    assert len(CHANNELS) >= 3, f"通道表只有{len(CHANNELS)}条，本门在空跑"
    for channel in CHANNELS:
        assert channel.env.startswith("PE_"), channel
        assert channel.tests and channel.validate_glob and channel.why, channel


def test_discovery_skips_a_candidate_that_only_matches_by_name(tmp_path: Path):
    """**必须红的那一半**：名字对、内容不对的候选不许被选中。

    这就是2026-08-18实测撞上的形态：另一个项目的`output`目录名字完全匹配，
    里面一个`run_manifest.json`都没有。
    """

    channel = next(c for c in CHANNELS if c.env == "PE_REPLAY_OUTPUT_TREE")
    _tree(tmp_path, "aaa-imposter/output")            # 名字对、空的
    real = _tree(tmp_path, "zzz-real/output", "runs/x/run_manifest.json")

    found = discover(channel, tmp_path)
    assert found == real, (
        f"发现器选了{found}——它应当跳过只匹配名字的那个。"
        "**一个只匹配名字的发现器会把'没找到'变成'找错了'，而后者更难查**"
    )


def test_discovery_returns_none_when_nothing_validates(tmp_path: Path):
    channel = next(c for c in CHANNELS if c.env == "PE_REPLAY_OUTPUT_TREE")
    _tree(tmp_path, "only-a-name/output")
    assert discover(channel, tmp_path) is None


def test_discovery_is_reproducible_across_two_runs(tmp_path: Path):
    """排序取第一个——同一棵树两次跑必须选中同一个。"""

    channel = next(c for c in CHANNELS if c.env == "PE_REPLAY_OUTPUT_TREE")
    _tree(tmp_path, "b-tree/output", "runs/x/run_manifest.json")
    _tree(tmp_path, "a-tree/output", "runs/x/run_manifest.json")
    assert discover(channel, tmp_path) == discover(channel, tmp_path)
    assert discover(channel, tmp_path).parent.name == "a-tree"


def test_an_existing_environment_variable_wins_over_discovery(monkeypatch, tmp_path: Path):
    """人已经指了就用人指的——发现器是兜底，不是覆盖。"""

    monkeypatch.setenv("PE_REPLAY_CASE_RUNS", "/somewhere/given/by/hand")
    resolved = resolve(tmp_path)
    assert resolved["PE_REPLAY_CASE_RUNS"] == ("/somewhere/given/by/hand", "环境变量")


def test_zero_resolved_channels_is_not_a_pass(monkeypatch, tmp_path: Path, capsys):
    """**空跑不是通过**：一条都没解析到时返回码非零。

    这是本仓反复记的那条纪律（`accept.py`的"零执行命令→BLOCKED"、
    `check_capability_ledger.py`的`LedgerEmpty`、可移植性校验的"扫了0个文件"）。
    """

    for channel in CHANNELS:
        monkeypatch.delenv(channel.env, raising=False)
    code = main(["--search-root", str(tmp_path)])
    assert code == 2, "一条通道都没解析到却没有判红——那正是空跑冒充通过"
    assert "空跑不是通过" in capsys.readouterr().err


def test_require_all_is_stricter_than_the_default(monkeypatch, tmp_path: Path):
    """`--require-all`：解析到一部分时默认放行、加了这个开关就红。"""

    for channel in CHANNELS:
        monkeypatch.delenv(channel.env, raising=False)
    monkeypatch.setenv("PE_REPLAY_CASE_RUNS", str(tmp_path))
    assert main(["--search-root", str(tmp_path), "--require-all"]) == 1


def test_a_search_root_that_is_not_a_directory_fails_closed(tmp_path: Path):
    assert main(["--search-root", str(tmp_path / "nope")]) == 2


@pytest.mark.parametrize("channel", CHANNELS, ids=lambda c: c.env)
def test_every_channel_names_tests_that_exist(channel):
    """通道表里指的测试文件必须真的存在——**表指着一份不存在的文件比没表更坏**。"""

    for relative in channel.tests:
        assert (ROOT / relative).is_file(), f"{channel.env}指着不存在的{relative}"
