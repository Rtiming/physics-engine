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
"""

from __future__ import annotations

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

    produced = (mirrored / name / "oracle.json").read_bytes()
    assert produced == committed.read_bytes(), (
        f"{name}的生成器重跑出来与入库的`oracle.json`**不是逐字节相同**——"
        "要么内核那条路的数变了（那该由案例判据先红），"
        "要么漂在案例判据没有逐条判的那些字段上。**后者今天只有本门看得见。**"
    )
