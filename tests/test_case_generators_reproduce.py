"""每个案例的生成器都必须**能重跑，且重跑出来逐字节相同**（2026-08-18立）。

## 它补的洞：SHA钉的是"哪一个脚本"，不是"这个脚本还产得出这份字节"

`oracle.json`的`generator.sha256`钉住的是生成器**文件本身**，
`manifest_self_sha256`钉住的是清单**内容本身**。两者都对，但**它们都不重跑那个脚本**。

于是有两类事今天没有任何东西看得见：

1. **生成器烂掉了但案例测试没发现**——案例测试比的是"现算的值 vs oracle里的值"，
   它走的是内核那条路，**不走生成器那条路**。生成器里用到的某个API改了名，
   案例测试照样全绿，而那个脚本已经跑不动了；
2. **生成器还跑得动，但产出漂了**——漂在案例测试**没有逐条判的那些字段上**
   （`inputs`里的元数据、注释性字段、数组的未判部分）。

本仓的纪律是"生成脚本入库是为了让那些数**有一个被SHA钉住的出处**"
（`cases/sdf_contact_convergence/case.md`第二节原文）。
**一个跑不动的出处不是出处。**

## 它怎么做到不动主仓

把`cases/`整棵树拷到临时目录再跑——生成器的`HERE`是
`Path(__file__).parent`，于是它写进拷贝里；`ROOT = HERE.parents[1]`是拷贝的根，
而`write_manifest(root=ROOT)`只拿它算**相对**路径，算出来的字符串两边一样。
`PYTHONPATH`仍指向真`src`，所以测的是**真内核**产出的字节。

**不用"跑完再还原"那种做法**：一次中断就会把主仓留在改过的状态，
而`accept.py`的仓库稳定轴会把那判成BLOCKED——**一道会把仓弄脏的门比没有门更坏**。

## 本门第一次上master就抓到一件真事：**9个oracle的数组哈希是跟机器走的**

本机（macOS arm64）40个全部逐字节复现，而同一棵树在master（Linux x86-64）上
**9个红**：`broadphase_superset`、`closed_loop_tension_step`、`double_slit_propagated`、
`free_span_tension_step`、`helix_laydown_closure`、`mutual_inductance_coaxial`、
`roller_skew_lateral_drift`、`scalar_diffraction_airy`、`three_sphere_pyramid_rotational`。

**差的只有三个字段**（实测逐字段比过）：`arrays.<名>.raw_sha256`、
`arrays.<名>.logical_sha256`、以及跟着变的`manifest_self_sha256`。
**其余字段一个字节不差。** 也就是说：**数组里的值本身在两个架构上不同**，
而那九条的数组全都过了libm的超越函数（椭圆积分、贝塞尔、`exp`、三角）——
**不同平台的libm最后一位不同**。这与plans/09记的`ad_dot`那次是同一族的另一种成因
（那次是求和次序，这次是libm）。

**它们的案例判据在两个平台都是绿的**——案例比的是"现算的值 vs oracle里的值"、带容差；
**逐字节那条claim只有产出它的那台机器上成立**，而`engine_oracle_manifest`这个面
**不记产出平台**，于是那些哈希看起来像与机器无关的。已登记（plans/07第六节）。

**所以本门判两件事，分开判**：

1. **跨平台那一半**：除三个哈希字段外**逐字节相同**——这一条在任何机器上都该成立；
2. **同平台那一半**：三个哈希也相同。**它不成立时不判红，判skip并说清楚**——
   因为那不是缺陷，是一条**本仓从来没有声明过的性质**。
   等`engine_oracle_manifest`记上产出平台那天，这一条才有资格变成红。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "cases"

#: 每个案例目录下的生成器。**排序是为了可复现**。
GENERATORS = sorted(CASES.glob("*/generate_oracle.py"))


def test_there_are_generators_to_check():
    """空表全绿是这一族门最常见的失效方式（本仓已撞过：可移植性校验扫0个文件）。"""

    assert len(GENERATORS) >= 30, f"只扫到{len(GENERATORS)}个生成器，本门在空跑"


@pytest.fixture(scope="module")
def mirrored(tmp_path_factory) -> Path:
    """`cases/`的一份拷贝。整棵树只拷一次——40个生成器共用。"""

    target = tmp_path_factory.mktemp("regen") / "cases"
    shutil.copytree(CASES, target)
    return target


@pytest.mark.batch
@pytest.mark.parametrize("generator", GENERATORS, ids=lambda p: p.parent.name)
def test_the_generator_reruns_and_reproduces_its_oracle(generator: Path, mirrored: Path):
    """**跑得动**，而且**跑出来逐字节相同**。

    两件事分开断言：跑不动报的是它自己的stderr（那是给人看的），
    跑得动但字节不同报的是哪一个案例漂了。
    """

    name = generator.parent.name
    mirror = mirrored / name / "generate_oracle.py"
    committed = generator.parent / "oracle.json"
    if not committed.is_file():
        pytest.skip(f"{name}没有oracle.json——它不是oracle型案例")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(mirror)],
        cwd=mirror.parents[1].parent, env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, (
        f"{name}的生成器跑不动——**一个跑不动的出处不是出处**。\n"
        f"stderr尾部：\n{completed.stderr[-1200:]}"
    )

    produced_bytes = (mirrored / name / "oracle.json").read_bytes()
    committed_bytes = committed.read_bytes()
    if produced_bytes == committed_bytes:
        return

    produced = json.loads(produced_bytes)
    expected = json.loads(committed_bytes)
    stripped_produced = _without_machine_dependent_hashes(produced)
    stripped_expected = _without_machine_dependent_hashes(expected)

    # 判据一（跨平台）：除三个哈希字段外必须逐字段相同。
    assert stripped_produced == stripped_expected, (
        f"{name}的生成器重跑出来与入库的`oracle.json`**在哈希之外也不同**——"
        "要么内核那条路的数变了（那该由案例判据先红），"
        "要么漂在案例判据没有逐条判的那些字段上。**后者今天只有本门看得见。**\n"
        f"差异字段：{sorted(_differing_paths(stripped_produced, stripped_expected))[:8]}"
    )

    # 判据二（同平台）：哈希也相同。不成立时**不判红**——见模块docstring。
    pytest.skip(
        f"{name}：除数组哈希外逐字节相同，但哈希不同——**这台机器不是产出它的那台**。"
        "九个案例的数组值过libm的超越函数，而不同平台libm最后一位不同"
        "（本机macOS arm64全过、master的Linux x86-64上这九个哈希不同）。"
        "**这不是缺陷，是本仓从来没有声明过的性质**：`engine_oracle_manifest`不记产出平台。"
        "已登记plans/07第六节；等那个面记上平台，这一条才有资格变成红。"
    )


#: 跟机器走的三个字段。**列出来而不是"凡是叫sha256的都跳过"**——
#: 后者会把`generator.sha256`（生成器文件本身的哈希，与机器无关）也放过，
#: 而那一条**必须**判：它变了意味着入库的脚本与产出这份oracle的脚本不是同一个。
MACHINE_DEPENDENT = ("raw_sha256", "logical_sha256")


def _without_machine_dependent_hashes(document: dict) -> dict:
    """去掉数组的两个哈希与跟着变的清单自哈希，其余原样。"""

    pruned = json.loads(json.dumps(document))
    pruned.pop("manifest_self_sha256", None)
    for entry in (pruned.get("arrays") or {}).values():
        if isinstance(entry, dict):
            for key in MACHINE_DEPENDENT:
                entry.pop(key, None)
    return pruned


def _differing_paths(left, right, path: str = "") -> list[str]:
    """两份文档里不同的字段路径。报路径而不是报整份——整份没人读得完。"""

    if isinstance(left, dict) and isinstance(right, dict):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            out += _differing_paths(left.get(key), right.get(key), f"{path}/{key}")
        return out
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out += _differing_paths(a, b, f"{path}[{index}]")
        return out
    return [] if left == right else [path or "/"]
