"""发布脚本的main与双远程失败关闭门。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("physics_engine_release", ROOT / "tools/release.py")
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_release_refuses_a_topic_branch_even_when_head_matches(monkeypatch):
    responses = {
        ("git", "branch", "--show-current"): "codex/candidate\n",
        ("git", "rev-parse", "HEAD"): "abc\n",
        ("git", "rev-parse", "main"): "abc\n",
        ("git", "remote"): "origin\ngithub\n",
    }
    monkeypatch.setattr(release, "run", lambda argv: responses[tuple(argv)])

    with pytest.raises(SystemExit, match="只允许从main"):
        release._assert_main_release_head()


def test_release_refuses_missing_second_remote(monkeypatch):
    responses = {
        ("git", "branch", "--show-current"): "main\n",
        ("git", "rev-parse", "HEAD"): "abc\n",
        ("git", "rev-parse", "main"): "abc\n",
        ("git", "remote"): "origin\n",
    }
    monkeypatch.setattr(release, "run", lambda argv: responses[tuple(argv)])

    with pytest.raises(SystemExit, match="github"):
        release._assert_main_release_head()


def test_release_pushes_main_and_the_same_tag_to_both_remotes():
    assert release._release_push_commands("0.6.0") == (
        ("git", "push", "origin", "main", "v0.6.0"),
        ("git", "push", "github", "main", "v0.6.0"),
    )


def test_release_refuses_candidate_changelog_heading():
    with pytest.raises(SystemExit, match="候选/未发布"):
        release._assert_release_changelog(
            "0.6.0", "## 0.6.0 — 候选（2026-08-13，未发布）\n"
        )


def test_release_accepts_a_final_changelog_heading():
    release._assert_release_changelog("0.6.0", "## 0.6.0 — 2026-08-14\n")
