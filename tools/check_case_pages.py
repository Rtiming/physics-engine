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
判据清单必须在（`oracle.json`或案例自带的判据正本），且`oracle.json`必须能过
`physics_engine.oracles`的严格加载器；清单的`case_id`与`load_tier`必须在案例页里
出现（页与清单不许各说各话）。

## 那三条结构校验各自补过的洞（2026-08-12，决策0056第六节）

plans/09第七节记着"`check_case_pages`三条结构校验各有一个洞"，逐条兑现：

1. **判据表全空单元格照过**。原判据只数行数、只认表头有没有"理由"二字，
   于是`| | | |`是一行合法的判据。现在数据行的**第一格（量）与最后一格（理由）
   都必须非空**——一张判据表的价值全在第三列，空着等于没有；
2. **删掉`oracle.json`等于关掉两条校验**。原代码是"有`oracle.json`才校验
   `case_id`与`load_tier`"，于是**删掉清单反而变干净**。现在每个案例必须有
   判据正本（`oracle.json`，或`peer_fcl_distance`那样的`criteria.json`），
   一个都没有当场红；
3. **案例只在散文里被提到就算登记**。原判据是`case_dir.name not in index_text`
   ——名字出现在索引页任何一句话里都算。这正是决策0049第六节记的那个形态
   （`peer_fcl_distance`长期挂在"在建"那句话里）。现在**必须是索引表格某一行的
   第一个单元格**："被提到"进不了第一格，"被登记"才进得去；
   判法与`check_gap_register.py`同源。

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

#: 判据正本可以叫哪几个名字。``oracle.json``是常态；``criteria.json``是
#: `peer_fcl_distance`那种同行库对拍的正本（2700组，页上写不下）。
#: **一个都没有是红**——原代码"有清单才校验"让删清单成了最省事的过门方式。
MANIFEST_NAMES: tuple[str, ...] = ("oracle.json", "criteria.json")

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


def table_cells(row: str) -> list[str]:
    """一行Markdown表格 → 去掉首尾竖线后的单元格文本。"""

    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def indexed_case_names(index_text: str) -> set[str]:
    """索引页表格里**作为第一个单元格**登记的案例名。

    只认第一格是这条校验的全部要害：案例名写在正文的任何一句话里都不算登记。
    决策0049第六节记的形态——`peer_fcl_distance`落地后长期挂在"在建"那句话里，
    而门只要求名字"出现在页上"，于是**门认得"被提到"，认不得"被登记"**。
    第一格里通常是``[`名字`](名字/case.md)``，所以按非标识符字符切开再收词。
    """

    names: set[str] = set()
    for line in index_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = table_cells(stripped)
        if len(cells) < 2:
            continue
        names.update(re.findall(r"[A-Za-z0-9_]+", cells[0]))
    return names


def is_separator_row(row: str) -> bool:
    """``|---|---|``这样的分隔行。它是"下一张表从这里开始"的结构标记。"""

    cells = table_cells(row)
    return bool(cells) and all(cell and set(cell) <= set("-: ") for cell in cells)


def first_table(rows: list[str]) -> list[str]:
    """一节里的**第一张表**。

    判据表那一节允许有第二第三张表——实测`generator_determinism`在那里放了
    一张实测偏差表、`mutual_inductance_coaxial`放了一张十二行的必红矩阵，
    两张都不是判据表，**列数与语义都不同**。拿"本节所有以竖线开头的行"
    当判据行会把它们一起判，那是门在管不该它管的事。
    第二张表的起点认**分隔行**这个结构标记，不认空行（分节时空行已被丢掉）。
    """

    if len(rows) < 3:
        return rows
    for index in range(2, len(rows)):
        if is_separator_row(rows[index]):
            return rows[: index - 1]
    return rows


def criteria_table_problems(page: Path, section_rows: list[str]) -> list[str]:
    """判据表的结构校验：**表头、分隔行、以及每一行数据都要真的有内容**。

    原判据只数行数与表头，于是``| | | |``是一行合法的判据——
    **一张判据表的价值全在第三列（理由），空着等于没有**。
    """

    rows = first_table(section_rows)
    if len(rows) < 3:
        return [
            f"{page}: 判据表至少要有表头、分隔行与一行判据（量→rel/abs→理由），"
            f"现在只有{len(rows)}行"
        ]
    if "理由" not in rows[0]:
        return [f"{page}: 判据表缺『理由』列——第三列是这张表存在的原因"]
    problems: list[str] = []
    for number, row in enumerate(rows[2:], start=1):
        cells = table_cells(row)
        if len(cells) < 3:
            problems.append(f"{page}: 判据表第{number}行只有{len(cells)}格，判据表要三列")
            continue
        if not cells[0]:
            problems.append(f"{page}: 判据表第{number}行没写是哪个量——空的第一格不是判据")
        if not cells[-1]:
            problems.append(
                f"{page}: 判据表第{number}行的『理由』是空的——"
                "容差没有理由就是拍脑袋，那正是这一列存在的原因"
            )
    return problems


def check_case(case_dir: Path, indexed: set[str]) -> list[str]:
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
    problems.extend(criteria_table_problems(page, rows))

    if case_dir.name not in indexed:
        problems.append(
            f"{case_dir.name}: 没有被 cases/README.md 的索引**表格**登记到"
            "——在正文里被提一句不算登记（决策0049第六节的形态）"
        )

    if not any((case_dir / name).is_file() for name in MANIFEST_NAMES):
        problems.append(
            f"{case_dir.name}: 一份判据正本都没有（要{list(MANIFEST_NAMES)}之一）"
            "——**删掉清单不是过门的办法**，那会连带关掉case_id与负载级两条校验"
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


#: 分层表的表头。**认标题不认位置**——把它当成结构位置，与`indexed_case_names`
#: 只认第一格是同一条通则（plans/09教训二）。
LAYER_TABLE_HEADING = "## 一之二"


def layer_table_problems(index_text: str, directories: list[Path]) -> list[str]:
    """索引页第一节之二那张**分层表**必须与案例目录对得上，三条都判。

    ## 为什么补这道门

    本仓已经栽过两次同一个形态，**两次都是"数字自己过期"而没有任何门会响**：

    * 波次二实测分类计数31、目录32（0064第7.4节登记，两张表漏一张）；
    * 波次三收口时登记了一句"**计数由收口时一次点清**"，
      而它**没有被执行**——到2026-08-18波次四收口时，表头还写着33、
      表里覆盖35条、目录已经36条。**登记不等于会被做，除非有门看着。**

    今天既有的那条只校验"案例目录必须出现在本页"（`check_case`里那条），
    **它管不到分层表**：一个案例可以在上面的索引表里登记得好好的，
    而分层表既没有它、求和也不对——那正是刚刚发生过的事。

    ## 三条判据

    1. 表头自称的条数 == 案例目录数；
    2. 各行"条数"列相加 == 案例目录数；
    3. 每个案例目录都在表里被提到（这一条允许出现在任何单元格里——
       分层表的语义就是"这条案例落在哪一行"，名字必然写在描述格中）。
    """

    names = {directory.name for directory in directories}
    if LAYER_TABLE_HEADING not in index_text:
        return [f"cases/README.md: 找不到分层表小节`{LAYER_TABLE_HEADING}`"]
    section = index_text.split(LAYER_TABLE_HEADING, 1)[1]
    #: 表止于紧随其后的那段解读文字；取到下一个二级标题为止已足够宽。
    section = section.split("\n## ", 1)[0]

    problems: list[str] = []
    declared = re.search(r"\*\*(\d+)个案例\*\*|：(\d+)个案例", section)
    if declared is None:
        problems.append("cases/README.md分层表没有自称条数——那个数就是过期的入口")
    else:
        stated = int(declared.group(1) or declared.group(2))
        if stated != len(names):
            problems.append(
                f"cases/README.md分层表自称{stated}个案例，实际目录{len(names)}个"
            )

    total = 0
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or is_separator_row(stripped):
            continue
        cells = table_cells(stripped)
        if len(cells) < 2:
            continue
        match = re.fullmatch(r"\*{0,2}(\d+)\*{0,2}", cells[1])
        if match:
            total += int(match.group(1))
    if total != len(names):
        problems.append(
            f"cases/README.md分层表各行相加{total}，实际目录{len(names)}个"
            "——一行漏了或一行数错了，两者都会让这张表安静地失去意义"
        )

    mentioned = set(re.findall(r"[A-Za-z0-9_]+", section))
    missing = sorted(names - mentioned)
    if missing:
        problems.append(
            f"cases/README.md分层表没有给这些案例分层：{missing}"
            "——新案例进了索引表却没进分层表，是本仓已发生过两次的形态"
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
    indexed = indexed_case_names(index.read_text(encoding="utf-8"))

    directories = _case_directories(cases_root)
    if not directories:
        print(f"FAIL {cases_root}: 有索引却没有案例目录")
        return 2

    problems: list[str] = []
    problems.extend(layer_table_problems(index.read_text(encoding="utf-8"), directories))
    for case_dir in directories:
        problems.extend(check_case(case_dir, indexed))
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"case pages: {len(directories)} checked, {len(problems)} problems")
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
