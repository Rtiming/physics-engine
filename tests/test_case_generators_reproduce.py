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

## 本门第一次上master就抓到一件真事：**oracle的金标值是跟机器走的**

本机（macOS arm64）40个全部逐字节复现，而同一棵树在master（Linux x86-64）上
**9个不同**：`broadphase_superset`、`closed_loop_tension_step`、`double_slit_propagated`、
`free_span_tension_step`、`helix_laydown_closure`、`mutual_inductance_coaxial`、
`roller_skew_lateral_drift`、`scalar_diffraction_airy`、`three_sphere_pyramid_rotational`。

**差的不只是哈希，`expected`里的金标值本身也在最后几位不同**——
例如`double_slit_propagated`的`relative_amplitude_components`八个分量、
`free_span_tension_step`的`restitution_2`与`span_2`。
（**这一段订正过一次**：初稿只逐字段比了`broadphase_superset`一个案例，
那一个恰好只差三个哈希字段，于是把结论外推成了"差的只有哈希"。**外推是错的。**）

成因是那九条的数都过了libm的超越函数（椭圆积分、贝塞尔、`exp`、三角），
而**不同平台的libm最后一位不同**。与plans/09记的`ad_dot`那次同族、成因不同
（那次是求和次序）。

**它们的案例判据在两个平台都是绿的**——案例比的是"现算的值 vs oracle里的值"、
**带这份oracle自己声明的容差**。所以：

* **物理没问题**；
* **"逐字节复现"这条claim只在产出它的那台机器上成立**，
  而`engine_oracle_manifest`这个面**不记产出平台**，
  于是那些金标看起来像与机器无关的。已登记（plans/07第六节）。

## 所以本门判三件事

1. **生成器跑得动**（任何平台）；
2. **重跑出来的值落在这份oracle自己声明的容差内**（任何平台）——
   **用本仓自己的容差机械判，不另发明一套**。这一条才是"生成器还产得出这份东西"的真判据；
3. **逐字节相同**（只有同平台才成立）——不成立时**判skip并说清楚**，不判红：
   那不是缺陷，是**一条本仓从来没有声明过的性质**。
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

    # 判据二（跨平台）：**用这份oracle自己声明的容差**判重跑出来的值。
    # 不另发明一套容差——本仓的规矩是容差要带理由，而理由就写在这份文件里。
    outside = _values_outside_declared_tolerance(produced, expected)
    assert not outside, (
        f"{name}的生成器重跑出来**超出了这份oracle自己声明的容差**——"
        "那不是平台差异，是内核那条路的数真的变了。\n"
        + "\n".join(outside[:6])
    )

    # 判据一之外的结构性字段（inputs、元数据）仍要逐字相同。
    structural = _differing_paths(
        _without_values_and_hashes(produced), _without_values_and_hashes(expected)
    )
    assert not structural, (
        f"{name}的生成器重跑出来在**非数值字段**上也不同：{sorted(structural)[:8]}"
    )

    # 判据三（同平台）：逐字节。不成立时不判红——见模块docstring。
    pytest.skip(
        f"{name}：值都落在这份oracle自己声明的容差内，但**不是逐字节相同**——"
        "这台机器不是产出它的那台。九个案例的数过libm的超越函数，"
        "而不同平台libm最后一位不同（本机macOS arm64全部逐字节复现，"
        "master的Linux x86-64上这九个不同）。**这不是缺陷，是本仓从来没有声明过的性质**："
        "`engine_oracle_manifest`不记产出平台。已登记plans/07第六节；"
        "等那个面记上平台，这一条才有资格变成红。"
    )


def _values_outside_declared_tolerance(produced: dict, expected: dict) -> list[str]:
    """逐条oracle、逐个字段，用**这份文件自己写的**容差判。

    容差形制是`{"abs": …, "rel": …, "reason": …}`（`oracles.py`的`Tolerance`）。
    **没有声明容差的字段按零容差判**——那是本仓的默认，不是我在这里定的。
    """

    problems: list[str] = []
    by_id = {entry["id"]: entry for entry in expected.get("oracles", ())}
    for entry in produced.get("oracles", ()):
        reference = by_id.get(entry["id"])
        if reference is None:
            problems.append(f"重跑多出一条oracle：{entry['id']}")
            continue
        tolerances = reference.get("tolerances") or {}
        for key, actual in (entry.get("expected") or {}).items():
            want = (reference.get("expected") or {}).get(key)
            tolerance = tolerances.get(key) or {"abs": 0.0, "rel": 0.0}
            problems += _compare(f"{entry['id']}/{key}", actual, want, tolerance)
    return problems


def _compare(where: str, actual, want, tolerance: dict) -> list[str]:
    """标量按容差比；嵌套序列逐元素递归；其余按相等比。"""

    if isinstance(actual, (list, tuple)) and isinstance(want, (list, tuple)):
        if len(actual) != len(want):
            return [f"{where}: 长度 {len(actual)} != {len(want)}"]
        out: list[str] = []
        for index, (a, b) in enumerate(zip(actual, want)):
            out += _compare(f"{where}[{index}]", a, b, tolerance)
        return out
    if isinstance(actual, bool) or isinstance(want, bool) or not isinstance(actual, (int, float)):
        return [] if actual == want else [f"{where}: {actual!r} != {want!r}"]
    if not isinstance(want, (int, float)):
        return [f"{where}: {actual!r} != {want!r}"]
    slack = float(tolerance.get("abs", 0.0)) + float(tolerance.get("rel", 0.0)) * abs(want)
    if abs(actual - want) <= slack:
        return []
    return [f"{where}: |{actual!r} − {want!r}| 超出 abs={tolerance.get('abs')} rel={tolerance.get('rel')}"]


def _without_values_and_hashes(document: dict) -> dict:
    """去掉逐条oracle的`expected`与所有跟机器走的哈希，其余原样。

    **`generator.sha256`不去掉**——它是生成器**文件本身**的哈希、与机器无关，
    而它变了意味着入库的脚本与产出这份oracle的脚本不是同一个。**那一条必须判。**
    """

    pruned = json.loads(json.dumps(document))
    pruned.pop("manifest_self_sha256", None)
    for entry in (pruned.get("arrays") or {}).values():
        if isinstance(entry, dict):
            for key in MACHINE_DEPENDENT:
                entry.pop(key, None)
    for entry in pruned.get("oracles", ()):
        entry.pop("expected", None)
    return pruned


#: 跟机器走的两个数组哈希。**列出来而不是"凡是叫sha256的都跳过"**——
#: 后者会把`generator.sha256`（生成器文件本身的哈希，与机器无关）也放过，
#: 而那一条**必须**判：它变了意味着入库的脚本与产出这份oracle的脚本不是同一个。
MACHINE_DEPENDENT = ("raw_sha256", "logical_sha256")


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
