#!/usr/bin/env python3
"""GCW的22列`centerline.csv` → `laydown.GrooveStation`能直接吃的元组。**纯标准库。**

    python tools/model/centerline_csv.py 某个/centerline.csv
    python tools/model/centerline_csv.py 某个/centerline.csv --stations   # 全部站点

兑现[decisions/0073](../../docs/decisions/0073_工具住哪与三维模型以什么形态进来_20260817.md)
第五节第2步，解锁[plans/15](../../docs/plans/15_从这里到真机介入的分阶段计划_20260817.md)阶段二2.1。
在它落地之前，**35个案例没有一个读过真实工件**——全是解析构造的螺旋线、平面圆与直链。

## 这个工具做什么，不做什么

**做**：把CSV的22列与它的`centerline.meta.json`伴生文件读进来，
**把每一条能在字节里判的约定当场判掉**，然后交出`(弧长, 位置, t, s, n)`五元组。

**不做**：不算曲率、不算扭率、不插值、不重采样、不平滑。
那些是物理与语义，属于调用方（`laydown`）。**工具只负责"CSV说了什么"，不负责"它意味着什么"。**

**尤其不做**：不替调用方决定`laydown.CenterlineSemantics`那五条
（位置插值、帧插值、拓扑、越界、最近点细化趟数）。0067已裁它们**不许有默认值**，
所以本工具**连一个字段都不为它们输出**——输出里出现一个默认值，
调用方就会以为"工具查过了"。`mesh_aabb.py`把`units`与`convexity`留`None`是同一条纪律。

## 为什么列序不能当基序用（**这是本文件最要紧的一条**）

CSV的列序是`t → n → s`（`tx,ty,tz`、`nx,ny,nz`、`sx,sy,sz`），
而帧的**基序**是`[tangent, width_direction, surface_normal]`，即`t → s → n`。
GCW自己的sidecar把这件事写成了一个显式字段：``csv_field_order_is_basis_order: false``。

**按列序读成基序，数值上不报任何错**——它只是把带宽方向与法向对调，
于是"带材宽度往哪边"整体错了90°，而每一步都还是单位向量、还是右手系。
`laydown.GrooveStation.__post_init__`会判`s = n × t`，所以这个错**最终会被抓到**；
但抓到的地方离犯错的地方隔着一个仓，诊断信息也不会说"你把列序当基序了"。
**所以本工具在读的时候就判，并且判据的名字里带着这句话。**

## 失败关闭的清单（缺一即拒，不许"先读进来再说"）

1. sidecar缺失、或`frame_convention.schema`不是`gcw.centerline_frame.v1`；
2. sidecar自称`convention_valid`为假、或`tangent_order_aligned`为假；
3. sidecar的`ordered_basis`不是`[tangent, width_direction, surface_normal]`
   ——**约定变了必须有人重读本文件**，不许静默按老约定解析；
4. 22列里缺任何一列（认列名，不认列号）；
5. 任一向量不是单位向量、或`s ≠ n × t`、或`t·n ≠ 0`（容差见下）；
6. `arc_length`不是严格递增。

**容差取`1e-9`**（无量纲，向量都是单位向量）。理由：真实语料实测最坏
`|s − n×t| = 5.8e-16`、`|‖v‖−1| = 4.4e-16`，离`1e-9`还有六个数量级——
**这个容差挡得住"约定错了"，挡不住浮点噪声，正是要的那一档**。
它比`laydown`自己的`FRAME_ORTHONORMAL_ABS_TOL`松还是紧不重要：
两边各判各的，本工具判的是"文件对不对"，`laydown`判的是"传进来的对不对"。

## 闭合缺口不判，只报

真实闭合曲线导出成开放折线，首末点之间会差**恰好一个采样步**。
实测`coil_v2_01_face_5`：总弧长1212.98 mm、首末间距2.0016 mm、采样步2.0 mm。
**这是拓扑信息不是错误**——闭不闭合由`CenterlineSemantics.topology`说了算，
而那一条不许有默认值。所以本工具**只把这个数报出来**，一个字都不裁。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

#: 22列的列名，**逐字**取自GCW导出的表头。认名字不认列号：
#: 上游加一列或调一次顺序，按列号读的解析器会静默读错，按名字读的会当场缺列。
POSITION_COLUMNS = ("x", "y", "z")
TANGENT_COLUMNS = ("tx", "ty", "tz")
NORMAL_COLUMNS = ("nx", "ny", "nz")
WIDTH_DIRECTION_COLUMNS = ("sx", "sy", "sz")
REQUIRED_COLUMNS = (
    "index",
    "arc_length",
    "parameter_s",
    *POSITION_COLUMNS,
    *TANGENT_COLUMNS,
    *NORMAL_COLUMNS,
    *WIDTH_DIRECTION_COLUMNS,
    "width",
    "left_x",
    "left_y",
    "left_z",
    "right_x",
    "right_y",
    "right_z",
)

#: sidecar里那份帧约定的身份。**换了schema就必须有人重读本文件**，不许静默兼容。
FRAME_SCHEMA = "gcw.centerline_frame.v1"
EXPECTED_ORDERED_BASIS = ("tangent", "width_direction", "surface_normal")

#: 见模块文档"失败关闭的清单"末段：挡约定错，不挡浮点噪声。
FRAME_ABS_TOL = 1e-9

Vector3 = tuple[float, float, float]


class CenterlineReadError(ValueError):
    """读不成就抛，**不返回"尽力而为"的部分结果**。

    部分结果的害处在本仓有先例：一份半读进来的声明会一路跑到物理里，
    然后在离犯错点很远的地方给出一个看起来正常的错数。
    """


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _distance(left: Vector3, right: Vector3) -> float:
    return _norm((left[0] - right[0], left[1] - right[1], left[2] - right[2]))


def _vector(row: dict[str, str], columns: tuple[str, str, str], where: str) -> Vector3:
    values = []
    for column in columns:
        text = row[column]
        try:
            value = float(text)
        except (TypeError, ValueError) as error:
            raise CenterlineReadError(f"{where}：列{column}的值{text!r}不是数") from error
        if not math.isfinite(value):
            raise CenterlineReadError(f"{where}：列{column}的值{value!r}不是有限数")
        values.append(value)
    return (values[0], values[1], values[2])


def read_sidecar(csv_path: Path) -> dict[str, object]:
    """读同目录的``centerline.meta.json``并把帧约定判掉。

    **sidecar不是可选的。** 没有它就不知道`n`是朝内还是朝外、`s`是不是`n × t`——
    而这两件事错了，数值上一个异常都不会有（见模块文档）。
    """

    sidecar_path = csv_path.with_name("centerline.meta.json")
    if not sidecar_path.is_file():
        raise CenterlineReadError(
            f"缺少伴生文件{sidecar_path.name}——帧约定只住在那里，"
            "没有它这份CSV的n与s是哪个方向无从判断，本工具拒绝猜"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CenterlineReadError(f"{sidecar_path.name}读不动：{error}") from error
    if not isinstance(sidecar, dict):
        raise CenterlineReadError(f"{sidecar_path.name}的顶层不是一个对象")

    convention = sidecar.get("frame_convention")
    if not isinstance(convention, dict):
        raise CenterlineReadError(f"{sidecar_path.name}里没有frame_convention对象")

    schema = convention.get("schema")
    if schema != FRAME_SCHEMA:
        raise CenterlineReadError(
            f"帧约定的schema是{schema!r}，本工具只认{FRAME_SCHEMA!r}——"
            "换了版本就必须有人重读tools/model/centerline_csv.py再放行，"
            "静默兼容一个没读过的约定正是本工具要挡的那件事"
        )

    basis = convention.get("ordered_basis")
    if tuple(basis or ()) != EXPECTED_ORDERED_BASIS:
        raise CenterlineReadError(
            f"ordered_basis是{basis!r}，本工具按{list(EXPECTED_ORDERED_BASIS)!r}解析"
        )

    for flag in ("convention_valid", "tangent_order_aligned"):
        if convention.get(flag) is not True:
            raise CenterlineReadError(
                f"sidecar自称{flag}={convention.get(flag)!r}——"
                "生产者说这份导出的帧约定不可信，消费方不该替它翻案"
            )
    return sidecar


def read_stations(csv_path: Path) -> list[tuple[float, Vector3, Vector3, Vector3, Vector3]]:
    """22列CSV → ``(arc_length_mm, position, tangent, width_direction, surface_normal)``。

    **返回的顺序就是`laydown.GrooveStation`的字段顺序**（基序，不是列序）。
    调用方一行`GrooveStation(*station)`即可，不需要在两个仓之间对着列名重排——
    那正是最容易把`n`与`s`对调的地方。
    """

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise CenterlineReadError(
                f"{csv_path.name}缺列{missing!r}（本工具认列名不认列号）"
            )
        rows = list(reader)

    if not rows:
        raise CenterlineReadError(f"{csv_path.name}一行数据都没有")

    stations: list[tuple[float, Vector3, Vector3, Vector3, Vector3]] = []
    previous_arc: float | None = None
    for number, row in enumerate(rows):
        where = f"{csv_path.name}第{number}行"
        try:
            arc_length = float(row["arc_length"])
        except (TypeError, ValueError) as error:
            raise CenterlineReadError(
                f"{where}：arc_length的值{row['arc_length']!r}不是数"
            ) from error
        if not math.isfinite(arc_length):
            raise CenterlineReadError(f"{where}：arc_length不是有限数")
        if previous_arc is not None and not arc_length > previous_arc:
            raise CenterlineReadError(
                f"{where}：arc_length {arc_length!r}没有严格大于上一行的{previous_arc!r}"
                "——弧长是本仓一切按弧长取值的坐标，它不单调则二分查找与插值全部失去意义"
            )
        previous_arc = arc_length

        position = _vector(row, POSITION_COLUMNS, where)
        tangent = _vector(row, TANGENT_COLUMNS, where)
        normal = _vector(row, NORMAL_COLUMNS, where)
        width_direction = _vector(row, WIDTH_DIRECTION_COLUMNS, where)

        for name, vector in (
            ("tangent", tangent),
            ("surface_normal", normal),
            ("width_direction", width_direction),
        ):
            deviation = abs(_norm(vector) - 1.0)
            if deviation > FRAME_ABS_TOL:
                raise CenterlineReadError(
                    f"{where}：{name}不是单位向量，|‖v‖−1| = {deviation!r}"
                )

        expected_width_direction = _cross(normal, tangent)
        gap = _distance(width_direction, expected_width_direction)
        if gap > FRAME_ABS_TOL:
            raise CenterlineReadError(
                f"{where}：width_direction ≠ cross(surface_normal, tangent)，差{gap!r}。"
                "最常见的原因是**把CSV的列序当成了基序**——列序是t→n→s，"
                "基序是t→s→n，sidecar自己用csv_field_order_is_basis_order=false说过这件事"
            )
        orthogonality = abs(_dot(tangent, normal))
        if orthogonality > FRAME_ABS_TOL:
            raise CenterlineReadError(
                f"{where}：tangent与surface_normal不正交，内积{orthogonality!r}"
            )

        stations.append((arc_length, position, tangent, width_direction, normal))
    return stations


def describe(csv_path: Path) -> dict[str, object]:
    """一份中心线 → 可粘进案例声明的字段 + 一份"这条线长什么样"的体检。

    **闭合缺口只报不裁**（见模块文档末节）：闭不闭合是
    `laydown.CenterlineSemantics.topology`的事，那一条不许有默认值。
    """

    sidecar = read_sidecar(csv_path)
    stations = read_stations(csv_path)
    payload = csv_path.read_bytes()
    arcs = [station[0] for station in stations]
    #: ``strict=False``是刻意的：两个序列长度**必然**差1（相邻差分），
    #: 写``strict=True``会在每一次正常调用上抛。
    steps = [b - a for a, b in zip(arcs, arcs[1:], strict=False)]
    closure_gap = _distance(stations[0][1], stations[-1][1])
    summary = sidecar.get("analysis_summary")
    summary = summary if isinstance(summary, dict) else {}
    return {
        "path_relative": csv_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "station_count": len(stations),
        "arc_length_total_mm": arcs[-1] - arcs[0],
        "arc_step_min_mm": min(steps) if steps else None,
        "arc_step_max_mm": max(steps) if steps else None,
        #: 首末点间距。**恰好一个采样步**通常意味着"闭合曲线导出成了开放折线"，
        #: 但"通常"不是判据——所以只给数，不给结论。
        "endpoint_gap_mm": closure_gap,
        "frame_schema": FRAME_SCHEMA,
        #: 生产者自报的弧长，与本工具从行里数出来的并排放着**是刻意的**：
        #: 两个数不一致时，使用者应当去问上游，而不是由本工具挑一个。
        "producer_reported_midline_length_mm": summary.get("midline_length"),
        "producer_reported_sample_count": summary.get("sample_count"),
        #: 与`mesh_aabb.py`同一条纪律：**语义留空比填默认值诚实**。
        #: 这五条是`laydown.CenterlineSemantics`的字段，0067裁决它们不许有默认值。
        "centerline_semantics": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GCW的22列centerline.csv → GrooveStation元组（纯标准库，不进内核）"
    )
    parser.add_argument("centerline", type=Path, nargs="+", help="centerline.csv")
    parser.add_argument(
        "--stations",
        action="store_true",
        help="连同全部站点一起输出（默认只输出体检摘要，真实语料有六百多行）",
    )
    arguments = parser.parse_args(argv)
    results = []
    for path in arguments.centerline:
        try:
            record = describe(path)
            if arguments.stations:
                record["stations"] = [
                    {
                        "arc_length_mm": arc,
                        "position_mm": list(position),
                        "tangent": list(tangent),
                        "width_direction": list(width_direction),
                        "surface_normal": list(normal),
                    }
                    for arc, position, tangent, width_direction, normal in read_stations(path)
                ]
            results.append(record)
        except (CenterlineReadError, OSError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            return 1
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
