#!/usr/bin/env python3
"""案例页校验器——plans/02第四节"案例页六必填字段，缺一即红，进accept"。

案例页不是文档义务，是**判据的可审性**：判据表的第三列（理由）、
已知失效清单、"本案例不是什么"三样，决定了半年后有人改内核时能不能判断
"这条红了到底说明什么"。缺了它们，案例目录就退化成PyElastica那种
0字节README的样例目录（research/05第一节的反面教材）。

六必填字段（**按标题正文匹配，中文序号前缀与后缀都忽略**；相对先后仍守着）：

1. `物理/几何设定`——全部参数与单位；
2. `参考解出处`——闭式解引文献作者/年/式号；无闭式解则生成脚本入库并给SHA；
3. `判据表`——量→rel/abs→**理由**，必须是表格且至少一行数据；
4. `已知失效清单`——每条一行理由，禁止静默skip；
5. `档位与负载级`——A/B/C档 + 交互级/本机批级/服务器级，与清单`load_tier`一致；
6. `本案例不是什么`——Drake形制的负空间声明。

**为什么不锁编号**：案例可以有第七第八节。`peer_fcl_distance`插了一节
"语义差异清单"（同行库两侧的胶囊定义、四元数次序、三条不同算子的语义），
那是该轨道最贵的一条发现——逼它为了凑编号删掉，是把形式看得比证据重。

外加三条结构校验：案例目录必须被`cases/README.md`索引到（新案例不许悄悄进仓）；
有`oracle.json`就必须能过`physics_engine.oracles`的严格加载器；
清单的`case_id`与`load_tier`必须在案例页里出现（页与清单不许各说各话）。

退出码：0=全绿；2=有案例不合格（与`pe-scene`的非法输入口径一致）。
用法：`.venv/bin/python tools/check_case_pages.py [cases目录]`
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import OracleError, load_manifest

#: 六必填是**内容义务，不是编号义务**。按标题正文匹配、忽略中文序号前缀，
#: 因为案例可以有第七第八节——`peer_fcl_distance`的"语义差异清单"就是那条
#: 轨道最贵的一条发现，逼它为了凑编号删掉是把形式看得比证据重。
#: 相对先后仍然守着（可读性），只是不再要求它们恰好是第一到第六节。
REQUIRED_FIELDS: tuple[str, ...] = (
    "物理/几何设定",
    "参考解出处",
    "判据表",
    "已知失效清单",
    "档位与负载级",
    "本案例不是什么",
)

CRITERIA_FIELD = "判据表"
TIER_FIELD = "档位与负载级"

#: 中文序号前缀：`## 三、判据表` → `判据表`。
_ORDINAL = re.compile(r"^##\s*[一二三四五六七八九十]+、\s*")


def normalise_heading(heading: str) -> str:
    """剥掉`## `与中文序号前缀，留下标题正文。"""

    stripped = _ORDINAL.sub("", heading.rstrip())
    return stripped[3:].strip() if stripped.startswith("## ") else stripped.strip()


def field_of(heading: str) -> str | None:
    """这个标题承担哪个必填字段？允许带后缀（如"（Drake形制）"）。"""

    text = normalise_heading(heading)
    for field in REQUIRED_FIELDS:
        if text.startswith(field):
            return field
    return None

# 下划线开头的目录是模板与脚手架，不是案例（`_template`）。
def _case_directories(cases_root: Path) -> list[Path]:
    return sorted(
        path
        for path in cases_root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    )


def _sections(text: str) -> dict[str, list[str]]:
    """按二级标题切段，返回 标题→段内非空正文行。"""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line.rstrip()
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line.rstrip())
    return sections


def check_case(case_dir: Path, index_text: str) -> list[str]:
    """校验一个案例目录，返回问题列表（空=合格）。"""

    problems: list[str] = []
    page = case_dir / "case.md"
    if not page.is_file():
        return [f"{case_dir.name}: 缺 case.md——案例页是案例的一部分，不是附件"]

    text = page.read_text(encoding="utf-8")
    sections = _sections(text)
    by_field: dict[str, list[str]] = {}
    order: list[str] = []
    for heading, body in sections.items():
        field = field_of(heading)
        if field is not None and field not in by_field:
            by_field[field] = body
            order.append(field)
    for field in REQUIRED_FIELDS:
        if field not in by_field:
            problems.append(f"{page}: 缺必填字段『{field}』")
        elif not by_field[field]:
            problems.append(f"{page}: 必填字段『{field}』是空的")
    if order != [field for field in REQUIRED_FIELDS if field in order]:
        problems.append(f"{page}: 六必填字段的先后顺序与规定不符：{order}")

    rows = [line for line in by_field.get(CRITERIA_FIELD, []) if line.startswith("|")]
    if len(rows) < 3:
        problems.append(
            f"{page}: 判据表至少要有表头、分隔行与一行判据（量→rel/abs→理由），"
            f"现在只有{len(rows)}行"
        )
    elif "理由" not in rows[0]:
        problems.append(f"{page}: 判据表缺『理由』列——第三列是这张表存在的原因")

    if case_dir.name not in index_text:
        problems.append(
            f"{case_dir.name}: 没有被 cases/README.md 索引到——新案例不许悄悄进仓"
        )

    manifest_path = case_dir / "oracle.json"
    if manifest_path.is_file():
        try:
            manifest = load_manifest(manifest_path, root=ROOT)
        except (OracleError, ValueError) as error:
            problems.append(f"{manifest_path}: 清单加载失败——{error}")
        else:
            if manifest.case_id not in text:
                problems.append(f"{page}: 页里没有出现清单声明的 case_id {manifest.case_id}")
            tier_body = " ".join(by_field.get(TIER_FIELD, ()))
            if manifest.load_tier not in tier_body:
                problems.append(
                    f"{page}: 第五节没有声明清单里的负载级 {manifest.load_tier}——"
                    "页与清单各说各话，marker就会跟着错"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    cases_root = Path(arguments[0]) if arguments else ROOT / "cases"
    if not cases_root.is_dir():
        print(f"no case suite at {cases_root} — nothing to check")
        return 0
    index = cases_root / "README.md"
    if not index.is_file():
        print(f"FAIL {index}: 案例套件缺中央索引")
        return 2
    index_text = index.read_text(encoding="utf-8")

    directories = _case_directories(cases_root)
    if not directories:
        print(f"FAIL {cases_root}: 有索引却没有案例目录")
        return 2

    problems: list[str] = []
    for case_dir in directories:
        problems.extend(check_case(case_dir, index_text))
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"case pages: {len(directories)} checked, {len(problems)} problems")
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
