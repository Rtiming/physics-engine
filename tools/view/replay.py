#!/usr/bin/env python3
"""`engine_run_trace`字节 → rerun记录（`.rrd`）。承接决策0074第六节阶段五。

## 这个工具补的洞

**本仓今天一条时间序列产物都没有落盘。** 实测：36个案例的`oracle.json`里
`arrays`字段**全部为空**（`{}`），每一条金标都是标量——峰值、峰值时刻、ISE、
稳态偏移。那是对的，判据本来就该是标量（可判、可给容差、可逐位对拍）。

**但标量判不了"它长什么样"。** 0074第5.3节把这件事写死了：
"**没有波形就开发不了张力算法**"。一条ISE = 3.79e-4 N²·s的门告诉你压制够不够，
不告诉你它是"峰高但衰减快"还是"峰低但一直不衰减"——而这两者的ISE可以相同。

本工具与`trace_from_closed_loop.py`合起来把这个洞补上：**产一条轨迹、看一条轨迹**。

## 它不做什么

* **不进内核。** `src/physics_engine`永不import本模块，rerun也永不被`src/`import
  （0025／0073的硬边界2，由`tests/governance/test_view_tools_stay_out_of_the_kernel.py`守着）。
* **不算物理。** 本模块一个物理量都不算。轨迹里是什么，画出来就是什么。
  没有插值、没有平滑、没有重采样——**画的和落盘的逐个数相同**。
* **不做单位换算。** 与`tools/model/mesh_aabb.py`同一条纪律：
  单位是轨迹声明的字段，一个查看器悄悄把mm当m画过去，出来的图**看着还很合理**。
  本模块只把声明的单位原样贴到实体上当标签。
* **不猜缺省。** `units`缺席即抛，`synthetic`缺席即抛（见下）。
* **不产判据。** `.rrd`不是证据。案例门是证据。见第三节。

## 一、`synthetic`为什么是必填而不是可选

0017抓到的真实缺陷：`examples/collision_preview_cell.scene.json`里声明的AABB是
**编的**，与真实包盒在z轴上完全不相交。那条缺陷之所以能活那么久，是因为
**看上去是有东西的**——字段填满了，没人知道它是编的。

查看器把这个风险放大一个量级：一个合成的圆柱画在屏幕上，
与真实资产的网格**长得一样可信**。所以本形制要求每一块几何**显式声明**
自己是不是合成的，缺席即失败关闭；`replay.py`把合成几何的实体路径统一挂在
`.../synthetic_*`下并写进实体的标签。**留空比填默认值诚实，而这一条比留空更强：
它不许留空。**

## 二、抽样（`stride`）为什么必须进字节而不是留在脚本里

演示那条轨迹的源是20万步（0.2秒 ÷ 1e-6）。全量进rerun没有意义，
所以要抽样。**而抽样会漏掉峰值**——那正是本案例三条判据里的第一条。

于是形制强制两件事：

1. `sampling.stride`是**必填**，画出来的曲线自己说得清它是几分之一；
2. `sampling.undecimated_extrema`记**未抽样**的极值与其时刻。
   `replay.py`把它作为静态`TextDocument`贴在记录根上，
   **于是这张图没法在峰值上骗人**——曲线上看不到的那个峰，文字里写着。

**这一条不是给"画得好看"用的，是给"画得不会骗人"用的。**

## 三、`.rrd`不是判据

本工具的产物**永不进任何门**。理由是本仓的一条既有纪律
（spec/08规则1：实测数不作金标）在查看器上的直接推论——
一个查看器同时当判据，等于让被验的东西自己出考卷。

`cases/closed_loop_tension_step`那三条门（峰值／ISE／稳态）判的是
`generate_oracle.py`那份**独立闭式解**，与本工具零交集。
本工具画的是同一次运行，**但它画错了不会有人红**——所以它不许被信。

## 四、用法

    # 产轨迹（主环境，纯标准库＋physics_engine，**不认识rerun**）
    PYTHONPATH="$PWD/src" .venv/bin/python tools/view/trace_from_closed_loop.py \\
        --out work/view/nominal.trace.json --band nominal

    # 画轨迹（view环境，rerun＋标准库，**不认识physics_engine**）
    tools/view/.venv/bin/python tools/view/replay.py \\
        work/view/nominal.trace.json --out work/view/nominal.rrd

    # 独立打开（另一台机器上同样这一条）
    tools/view/.venv/bin/rerun work/view/nominal.rrd

**两个环境互不认识对方的依赖，中间只有一份JSON**——
这不是巧合，是0074那三条硬边界在目录里的形状。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

FACET = "engine_run_trace"
#: 认到哪一版为止。**大版本不同即抛**——不做"尽量读"。
SUPPORTED_MAJOR = 0

#: 时间线在rerun里的名字由轨迹自己给（`timeline.name`），本模块不硬编码；
#: 但单位是`duration`（秒）这件事是硬的——rerun的时间线要么是序号要么是时长，
#: 一个声明单位是`ms`的轨迹如果被当成秒喂进去，**时间轴会差1000倍而图看着照样正常**。
TIME_UNIT_TO_SECONDS = {"s": 1.0}


class TraceError(ValueError):
    """一切失败关闭。**不猜、不容错、不画一半。**

    画一半是这里最坏的结果：使用者拿到一个能打开的`.rrd`，
    里面少了一条曲线而**没有任何地方说少了**。
    """


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    """必填字段。**缺席即抛，不给默认值。**"""

    if key not in mapping:
        raise TraceError(
            f"{where}缺必填字段`{key}` —— 本形制没有默认值："
            "一个默认值会被当成'轨迹里说过了'"
        )
    return mapping[key]


def _require_finite_vec3(value: Any, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TraceError(f"{where}不是三分量向量：{value!r}")
    out = []
    for component in value:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise TraceError(f"{where}含非数：{component!r}")
        if not math.isfinite(component):
            raise TraceError(f"{where}含非有限值：{component!r} —— NaN画出来是个洞，不是个警告")
        out.append(float(component))
    return out


def _require_finite_quaternion(value: Any, where: str) -> list[float]:
    """四元数按**xyzw序**收。序错不会抛，画面会转成另一个样子——所以名字里带序。"""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TraceError(f"{where}不是四分量（xyzw序）：{value!r}")
    out = []
    for component in value:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise TraceError(f"{where}含非数：{component!r}")
        if not math.isfinite(component):
            raise TraceError(f"{where}含非有限值：{component!r}")
        out.append(float(component))
    #: **不归一化，只判**。一个悄悄归一化的查看器会把"产轨迹那侧算错了"
    #: 这件事吃掉，而那正是要看见的东西。容差取1e-6：再紧会被float32的
    #: 往返误差撞上（rerun的Transform3D存的是float32）。
    norm = math.sqrt(sum(c * c for c in out))
    if abs(norm - 1.0) > 1.0e-6:
        raise TraceError(
            f"{where}的模是{norm!r}，不是单位四元数 —— "
            "本工具**只判不归一化**：悄悄归一化会把'产轨迹那侧算错了'这件事吃掉"
        )
    return out


def _require_same_length(count: int, series: Any, where: str) -> list[Any]:
    if not isinstance(series, list):
        raise TraceError(f"{where}不是列表：{type(series).__name__}")
    if len(series) != count:
        raise TraceError(
            f"{where}有{len(series)}帧而时间线有{count}帧 —— "
            "**逐帧序列与时间线必须等长**。少一帧就画一帧，"
            "出来的图会把整条曲线在时间上错位，而它看起来完全正常"
        )
    return series


def load_trace(path: Path) -> dict[str, Any]:
    """读轨迹并**在画任何东西之前**把形制验完。

    次序是有理由的：先验完再画，才不会出现"画了三条实体然后抛"这种
    半成品`.rrd`——那种文件能打开，所以它比抛异常更坏。
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TraceError(f"{path}不是合法JSON：{error}") from error
    if not isinstance(raw, dict):
        raise TraceError(f"{path}的顶层不是对象：{type(raw).__name__}")

    facet = _require(raw, "facet", "轨迹根")
    if facet != FACET:
        raise TraceError(f"面名不是`{FACET}`而是`{facet!r}` —— 本工具不读别的面")
    version = str(_require(raw, "facet_version", "轨迹根"))
    major = version.split(".", 1)[0]
    if not major.isdigit() or int(major) != SUPPORTED_MAJOR:
        raise TraceError(
            f"面版本`{version}`的大版本与本工具支持的`{SUPPORTED_MAJOR}`不同 —— "
            "大版本不同即破坏性变更，不做'尽量读'"
        )

    units = _require(raw, "units", "轨迹根")
    if not isinstance(units, dict) or "length" not in units:
        raise TraceError(
            "`units.length`必填 —— 本工具不做单位换算（与`tools/model/mesh_aabb.py`同源）。"
            "一个把mm当m画的查看器，出来的图看着还很合理"
        )

    timeline = _require(raw, "timeline", "轨迹根")
    time_unit = _require(timeline, "unit", "`timeline`")
    if time_unit not in TIME_UNIT_TO_SECONDS:
        raise TraceError(
            f"时间线单位`{time_unit}`不在支持表{sorted(TIME_UNIT_TO_SECONDS)}内 —— "
            "换算表里没有的单位不许猜"
        )
    times = _require(timeline, "times", "`timeline`")
    if not isinstance(times, list) or not times:
        raise TraceError("`timeline.times`必须是非空列表 —— 空时间线画不出时间轴")

    sampling = _require(raw, "sampling", "轨迹根")
    _require(sampling, "stride", "`sampling`")

    frames = len(times)
    for block in raw.get("geometry", []):
        _require(block, "entity_path", "`geometry`条目")
        #: **必填、无默认**。理由见模块docstring第一节。
        if "synthetic" not in block:
            raise TraceError(
                f"几何`{block.get('entity_path')}`没声明`synthetic` —— "
                "合成网格与真实资产在屏幕上一样可信，"
                "这正是0017那条'AABB是编的'缺陷在查看器上的形态"
            )
    for block in raw.get("poses", []):
        path_ = _require(block, "entity_path", "`poses`条目")
        translations = _require_same_length(
            frames, _require(block, "translations", f"位姿`{path_}`"),
            f"位姿`{path_}`的`translations`")
        quaternions = _require_same_length(
            frames, _require(block, "quaternions_xyzw", f"位姿`{path_}`"),
            f"位姿`{path_}`的`quaternions_xyzw`")
        #: **有限性在这里查完，不留到画的时候**——见本函数docstring那条次序。
        for index, translation in enumerate(translations):
            _require_finite_vec3(translation, f"位姿`{path_}`第{index}帧的平移")
        for index, quaternion in enumerate(quaternions):
            _require_finite_quaternion(quaternion, f"位姿`{path_}`第{index}帧的四元数")
    for block in raw.get("scalars", []):
        path_ = _require(block, "entity_path", "`scalars`条目")
        #: **单位必填**——一条没有单位的曲线，纵轴上那个数是什么没人知道。
        _require(block, "unit", f"标量`{path_}`")
        values = _require_same_length(
            frames, _require(block, "values", f"标量`{path_}`"),
            f"标量`{path_}`的`values`")
        for index, value in enumerate(values):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TraceError(f"标量`{path_}`第{index}帧不是数：{value!r}")
            if not math.isfinite(value):
                raise TraceError(
                    f"标量`{path_}`第{index}帧是{value!r} —— "
                    "NaN在曲线上是**一段断掉的线**，看起来像『求解器那会儿没输出』，"
                    "而真相是它输出了一个非数"
                )
    for block in raw.get("points", []):
        path_ = _require(block, "entity_path", "`points`条目")
        per_frame = _require_same_length(
            frames, _require(block, "frames", f"点集`{path_}`"),
            f"点集`{path_}`的`frames`")
        for index, positions in enumerate(per_frame):
            if not isinstance(positions, list):
                raise TraceError(f"点集`{path_}`第{index}帧不是列表：{positions!r}")
            for point in positions:
                _require_finite_vec3(point, f"点集`{path_}`第{index}帧的点")
    return raw


def _summary_document(trace: dict[str, Any]) -> str:
    """贴在记录根上的静态说明。**它的第一段是"这不是判据"。**"""

    sampling = trace["sampling"]
    lines = [
        "# " + str(trace.get("run_id", "(无run_id)")),
        "",
        "**这份记录不是判据。** 案例门判的是`generate_oracle.py`那份独立闭式解，",
        "与本记录零交集；本记录画错了不会有人红——所以不许拿它当证据。",
        "",
        "## 抽样",
        "",
        f"- 源步数：`{sampling.get('source_step_count')}`，步长`{sampling.get('source_dt_s')}` s",
        f"- 抽样步长（stride）：`{sampling.get('stride')}` ⟹ 本记录`{len(trace['timeline']['times'])}`帧",
    ]
    extrema = sampling.get("undecimated_extrema")
    if extrema:
        lines += ["", "## 未抽样极值（曲线上可能看不到）", ""]
        for name, entry in sorted(extrema.items()):
            lines.append(
                f"- `{name}`：峰值`{entry.get('peak')}`，出现在`{entry.get('peak_time_s')}` s"
            )
        lines += [
            "",
            "**抽样后的曲线会漏峰。** 上面这几个数是从未抽样的全量算的——",
            "曲线上看不到的那个峰，这里写着。",
        ]
    for note in trace.get("notes", []):
        lines += ["", str(note)]
    return "\n".join(lines)


def replay(trace: dict[str, Any], out: Path) -> dict[str, Any]:
    """把已验形制的轨迹写成`.rrd`，返回一份**实测**的记账。

    返回的记账不是从轨迹抄的，是从"实际log了几次"数的——
    **声称写了什么与实际写了什么必须是同一个数**。
    """

    import rerun as rr

    times = [float(t) * TIME_UNIT_TO_SECONDS[trace["timeline"]["unit"]]
             for t in trace["timeline"]["times"]]
    timeline = trace["timeline"].get("name", "sim_time")
    length_unit = trace["units"]["length"]

    rr.init(str(trace.get("run_id", "engine_run_trace")), spawn=False)
    logged: dict[str, int] = {}

    def _log(entity_path: str, payload: Any, *, static: bool = False) -> None:
        rr.log(entity_path, payload, static=static)
        logged[entity_path] = logged.get(entity_path, 0) + 1

    _log("/", rr.TextDocument(_summary_document(trace),
                              media_type=rr.MediaType.MARKDOWN), static=True)

    #: 几何是**静态**的：它不随时间变，变的是挂在它上面的位姿。
    #: 这条分工是rerun的实体层级本来的形状，也是本形制把二者拆成两个块的理由。
    for block in trace.get("geometry", []):
        path_ = block["entity_path"]
        label = "合成几何" if block["synthetic"] else "真实资产"
        vertices = [_require_finite_vec3(v, f"`{path_}`顶点")
                    for v in _require(block, "vertex_positions", f"几何`{path_}`")]
        triangles = block.get("triangle_indices")
        _log(path_, rr.Mesh3D(vertex_positions=vertices, triangle_indices=triangles),
             static=True)
        _log(path_, rr.AnyValues(declared_synthetic=block["synthetic"],
                                 declared_kind=label,
                                 declared_length_unit=length_unit), static=True)

    for block in trace.get("scalars", []):
        #: 单位贴成静态标注——**曲线的纵轴上那个数是什么，记录自己说得清**。
        _log(block["entity_path"], rr.AnyValues(declared_unit=block["unit"]), static=True)

    for index, time_s in enumerate(times):
        rr.set_time(timeline, duration=time_s)
        for block in trace.get("poses", []):
            _log(block["entity_path"], rr.Transform3D(
                translation=_require_finite_vec3(
                    block["translations"][index], f"`{block['entity_path']}`平移"),
                quaternion=block["quaternions_xyzw"][index],
            ))
        for block in trace.get("scalars", []):
            _log(block["entity_path"], rr.Scalars(float(block["values"][index])))
        for block in trace.get("points", []):
            positions = [_require_finite_vec3(p, f"`{block['entity_path']}`点")
                         for p in block["frames"][index]]
            _log(block["entity_path"], rr.Points3D(
                positions, radii=block.get("radii"), colors=block.get("colors")))

    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(out)
    #: **`rr.save()`返回不等于文件写完。** 实测：紧接着`stat()`拿到321475字节，
    #: 而进程退出后文件是343875字节——**差22400字节**，因为rerun的落盘是异步的
    #: （数据在记录被关掉时才冲干净）。
    #:
    #: 不冲就报数，等于**回执里那个字节数是假的**——而回执是本工具唯一
    #: 对外声称的东西。`rerun_shutdown()`把记录关掉，之后`stat()`才是最终值。
    #: 这条与`run_package.publish_package`那套"写完复验哈希"同源：
    #: **声称写了什么与实际写了什么必须是同一个数。**
    rr.rerun_shutdown()
    return {
        "path": str(out),
        "bytes": out.stat().st_size,
        "timeline": timeline,
        "frames": len(times),
        "time_range_s": [times[0], times[-1]],
        "entities": dict(sorted(logged.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="engine_run_trace（JSON） → rerun记录（.rrd）。不进内核、不算物理。"
    )
    parser.add_argument("trace", type=Path, help="engine_run_trace形制的JSON")
    parser.add_argument("--out", type=Path, required=True, help="产出的.rrd路径")
    arguments = parser.parse_args(argv)
    try:
        trace = load_trace(arguments.trace)
        receipt = replay(trace, arguments.out)
    except (TraceError, OSError) as error:
        print(f"{arguments.trace}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
