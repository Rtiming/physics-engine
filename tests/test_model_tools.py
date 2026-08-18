"""`tools/model/`自己的门（决策0073）。

本文件判两件事：工具**算得对**（对既有案例金标独立复现），
以及它**失败关闭**（认不出的格式、不自洽的字节、留白的判断一律不猜）。

工具的身份边界（不进内核、不进wheel）由`tests/governance/`那道门守。
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from tools.model import centerline_csv, mesh_aabb

ROOT = Path(__file__).resolve().parents[1]
TETRA = ROOT / "cases/mesh_asset_integrity/assets/tetra.stl"


def _binary_stl(triangles: list[tuple[tuple[float, float, float], ...]]) -> bytes:
    """按二进制STL的字节形制造一份最小载荷（测试自己的构造器，不用工具的解析器）。"""

    payload = bytearray(b"\0" * 80)
    payload += struct.pack("<I", len(triangles))
    for corners in triangles:
        payload += struct.pack("<3f", 0.0, 0.0, 1.0)  # 法向，工具会跳过
        for corner in corners:
            payload += struct.pack("<3f", *corner)
        payload += struct.pack("<H", 0)
    return bytes(payload)


def test_the_tool_reproduces_the_case_oracle_on_the_repo_asset():
    """**这一条是本文件的理由**：工具独立复现`mesh_asset_integrity`已有的金标。

    那条案例的`true_aabb_min_mm`/`true_aabb_max_mm`是它自己的生成器算的，
    而生成器与本工具**是两份独立实现**。两边给同一个数，才说明
    "把包盒从人工填写换成工具生成"这件事不引入新的错。

    金标出处：`cases/mesh_asset_integrity/oracle.json`。
    """

    described = mesh_aabb.describe(TETRA)
    assert described["aabb_min"] == [-3.5, 1.25, -2.0]
    assert described["aabb_max"] == [26.5, 25.25, 16.0]
    #: SHA同时对上场景声明里写的那一串——**两个独立产物指向同一份字节**。
    assert described["sha256"] == (
        "f0b168923850a72248c783a1a797f29c71c16d2964a4b7f5cdc26481a70b4b71"
    )
    assert described["vertex_count"] == 12


def test_the_bounds_are_the_exact_extremes_with_no_margin():
    """**工具不许替声明者加安全余量。**

    余量是`envelope`语义下**声明者**的决定。工具加一点，那道包含门就变成
    在验工具自己加的那个数——而那正是0017抓到的"包络是编的"的另一种形态。
    """

    payload = _binary_stl([((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-4.0, 0.5, 0.25))])
    vertices = mesh_aabb.parse_binary_stl(payload)
    lower, upper = mesh_aabb.axis_aligned_bounds(vertices)
    assert lower == (-4.0, 0.0, 0.0)
    assert upper == (1.0, 2.0, 3.0)


def test_units_and_convexity_are_left_blank_not_defaulted():
    """**留白比默认值诚实。** 两者都是声明者的判断，从字节里读不出来。

    填一个默认值的后果是确定的：下一个人会把它当成"工具查过了"。
    凸性尤其如此——spec/11规则2点名的"MuJoCo教训"就是自动取凸包**静默改变形状**，
    而本次同行核查确认MuJoCo连关掉它的开关都在3.2.5删了。
    """

    described = mesh_aabb.describe(TETRA)
    assert described["units"] is None
    assert described["convexity"] is None


def test_an_ascii_stl_is_refused_not_guessed():
    """必须红：ASCII STL当场拒，不转去解析。

    一个工具同时支持两条文本化路径而不声明用的是哪条，
    产出的AABB就带一个说不清的来源。
    """

    with pytest.raises(mesh_aabb.MeshReadError, match="ASCII STL"):
        mesh_aabb.parse_binary_stl(b"solid tetra\nfacet normal 0 0 1\n")


def test_a_truncated_binary_stl_fails_closed():
    """必须红：声明的三角形数与实际字节数不符即抛，**不按"能读多少读多少"**。

    读一半得到的包盒是**偏小**的——而偏小的包络正是broad phase会漏掉接触的那个方向。
    """

    payload = bytearray(_binary_stl([((0.0, 0.0, 0.0),) * 3]))
    payload = payload[:-10]
    with pytest.raises(mesh_aabb.MeshReadError, match="declares 1 triangles"):
        mesh_aabb.parse_binary_stl(bytes(payload))


def test_an_unknown_suffix_is_refused():
    """必须红：后缀不在白名单即抛，不按内容猜格式。"""

    with pytest.raises(mesh_aabb.MeshReadError, match="unsupported mesh suffix"):
        mesh_aabb.describe(Path("model.ply"))


def test_an_obj_vertex_with_a_fourth_component_is_refused():
    """必须红：`v x y z w`的第四个分量（有理B样条的权）当场拒。

    带`w`说明这不是一个普通多边形网格，而包盒的含义要另说——
    **默默把它当成三维点，得到的是一个语义不明的数**。
    """

    with pytest.raises(mesh_aabb.MeshReadError, match="exactly three coordinates"):
        mesh_aabb.parse_obj(b"v 1.0 2.0 3.0 0.5\n")


def test_an_obj_without_vertices_fails_closed():
    """必须红：没有`v`行的OBJ不返回空包盒。

    空包盒会让包含判据**恒真**——那是一道被静默拆掉的门。
    """

    with pytest.raises(mesh_aabb.MeshReadError, match="no 'v' vertex lines"):
        mesh_aabb.parse_obj(b"# only a comment\nvt 0.0 0.0\n")


def test_bounding_an_empty_vertex_set_fails_closed():
    """必须红：空顶点集不许返回一个包盒。"""

    with pytest.raises(mesh_aabb.MeshReadError, match="empty vertex set"):
        mesh_aabb.axis_aligned_bounds([])


def test_the_obj_reader_ignores_texture_and_normal_lines():
    """`vt`与`vn`不参与包盒——包盒由几何顶点定。

    这一条是正向的：上一条判"只有`vt`时要红"，这一条判"有`v`时`vt`不许干扰"。
    """

    vertices = mesh_aabb.parse_obj(
        b"v 0 0 0\nvt 9 9\nvn 0 0 1\nv 1 2 3\n"
    )
    assert vertices == [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]


# --------------------------------------------------------------- centerline_csv
#: 22列表头逐字取自GCW的真实导出。测试自己写一份，**不从工具里import列名**——
#: 从被测者那里拿判据，等于让它自证。
_CENTERLINE_HEADER = (
    "index,arc_length,parameter_s,x,y,z,tx,ty,tz,nx,ny,nz,sx,sy,sz,"
    "width,left_x,left_y,left_z,right_x,right_y,right_z"
)

_SIDECAR = {
    "frame_convention": {
        "schema": "gcw.centerline_frame.v1",
        "ordered_basis": ["tangent", "width_direction", "surface_normal"],
        "convention_valid": True,
        "tangent_order_aligned": True,
    },
    "analysis_summary": {"midline_length": 2.0, "sample_count": 3},
}


def _centerline_rows(
    frames: list[tuple[float, tuple[float, float, float], tuple[float, float, float]]],
) -> str:
    """按`(弧长, tangent, surface_normal)`造行；`width_direction`由测试**自己**算`n × t`。

    工具算一遍、测试算一遍，两边独立——这样"帧约定判据"才是被验的，不是被复述的。
    """

    lines = [_CENTERLINE_HEADER]
    for index, (arc, tangent, normal) in enumerate(frames):
        width_direction = (
            normal[1] * tangent[2] - normal[2] * tangent[1],
            normal[2] * tangent[0] - normal[0] * tangent[2],
            normal[0] * tangent[1] - normal[1] * tangent[0],
        )
        position = (arc, 0.0, 0.0)
        lines.append(
            ",".join(
                repr(value)
                for value in (
                    index,
                    arc,
                    arc,
                    *position,
                    *tangent,
                    *normal,
                    *width_direction,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )
        )
    return "\n".join(lines) + "\n"


_STRAIGHT = [
    (0.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (2.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
]


def _write_centerline(directory: Path, rows: str, sidecar: dict | None = None) -> Path:
    import json

    path = directory / "centerline.csv"
    path.write_text(rows, encoding="utf-8")
    (directory / "centerline.meta.json").write_text(
        json.dumps(_SIDECAR if sidecar is None else sidecar), encoding="utf-8"
    )
    return path


def test_the_stations_come_out_in_basis_order_not_column_order(tmp_path):
    """**本文件为这条而写。** 列序是`t → n → s`，基序是`t → s → n`。

    工具交出的第四个分量必须是`width_direction`、第五个是`surface_normal`。
    按列序交出来在数值上不报任何错——它只是把带宽方向与法向对调。
    """

    path = _write_centerline(tmp_path, _centerline_rows(_STRAIGHT))
    stations = centerline_csv.read_stations(path)

    assert len(stations) == 3
    arc, position, tangent, width_direction, normal = stations[0]
    assert arc == 0.0
    assert position == (0.0, 0.0, 0.0)
    assert tangent == (1.0, 0.0, 0.0)
    #: `n × t`＝`(0,0,1) × (1,0,0)`＝`(0,1,0)`。若按列序读，这里会拿到`(0,0,1)`。
    assert width_direction == (0.0, 1.0, 0.0)
    assert normal == (0.0, 0.0, 1.0)


def test_the_tuple_order_matches_the_groove_station_field_order():
    """工具的元组顺序与内核`GrooveStation`的字段顺序必须逐位对齐。

    **这条门守的是一次跨仓重排**：调用方写`GrooveStation(*station)`，
    两边字段顺序一旦漂开，构造出来的站点每个分量都是别人的。
    """

    from dataclasses import fields

    from physics_engine.laydown import GrooveStation

    assert tuple(field.name for field in fields(GrooveStation)) == (
        "arc_length_mm",
        "position_mm",
        "tangent",
        "width_direction",
        "surface_normal",
    )


def test_a_real_groove_station_can_be_built_from_the_tool_output(tmp_path):
    """端到端：工具的输出直接喂给内核的`GrooveStation`，它自己的帧判据必须过。"""

    from physics_engine.laydown import GrooveStation

    path = _write_centerline(tmp_path, _centerline_rows(_STRAIGHT))
    stations = [GrooveStation(*station) for station in centerline_csv.read_stations(path)]
    assert len(stations) == 3
    assert stations[-1].arc_length_mm == 2.0


def test_a_swapped_normal_and_width_direction_is_refused(tmp_path):
    """把`n`与`s`对调（即"按列序当基序"那个错）必须当场红。"""

    rows = _centerline_rows(_STRAIGHT).splitlines()
    header, first = rows[0], rows[1].split(",")
    #: 列9—11是`n`、列12—14是`s`，直接对调这两段。
    first[9:12], first[12:15] = first[12:15], first[9:12]
    broken = "\n".join([header, ",".join(first), *rows[2:]]) + "\n"
    path = _write_centerline(tmp_path, broken)

    with pytest.raises(centerline_csv.CenterlineReadError, match="列序"):
        centerline_csv.read_stations(path)


def test_a_non_unit_frame_vector_is_refused(tmp_path):
    """帧向量不是单位向量必须红——它会让后面每一次投影都带一个静默的比例。"""

    scaled = [(0.0, (2.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
    path = _write_centerline(tmp_path, _centerline_rows(scaled))

    with pytest.raises(centerline_csv.CenterlineReadError, match="单位向量"):
        centerline_csv.read_stations(path)


def test_a_non_monotonic_arc_length_is_refused(tmp_path):
    """弧长不严格递增必须红：它是本仓一切按弧长取值的坐标。"""

    backwards = [
        (0.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        (1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        (1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    path = _write_centerline(tmp_path, _centerline_rows(backwards))

    with pytest.raises(centerline_csv.CenterlineReadError, match="严格大于"):
        centerline_csv.read_stations(path)


def test_a_missing_column_is_named_not_guessed(tmp_path):
    """缺列必须点名缺哪一列——按列号读的解析器在这里会静默读错。"""

    rows = _centerline_rows(_STRAIGHT).replace("sx,sy,sz", "sx,sy,sZZ", 1)
    path = _write_centerline(tmp_path, rows)

    with pytest.raises(centerline_csv.CenterlineReadError, match="sz"):
        centerline_csv.read_stations(path)


def test_a_missing_sidecar_fails_closed(tmp_path):
    """没有sidecar就没有帧约定，工具**拒绝猜**。"""

    path = tmp_path / "centerline.csv"
    path.write_text(_centerline_rows(_STRAIGHT), encoding="utf-8")

    with pytest.raises(centerline_csv.CenterlineReadError, match="伴生文件"):
        centerline_csv.read_sidecar(path)


def test_an_unknown_frame_schema_is_refused(tmp_path):
    """schema换了版本必须红——静默兼容一个没人读过的约定正是要挡的那件事。"""

    import copy

    sidecar = copy.deepcopy(_SIDECAR)
    sidecar["frame_convention"]["schema"] = "gcw.centerline_frame.v2"
    path = _write_centerline(tmp_path, _centerline_rows(_STRAIGHT), sidecar)

    with pytest.raises(centerline_csv.CenterlineReadError, match="schema"):
        centerline_csv.read_sidecar(path)


def test_a_producer_declared_invalid_convention_is_refused(tmp_path):
    """生产者自己说这份导出的帧不可信，消费方不许替它翻案。"""

    import copy

    sidecar = copy.deepcopy(_SIDECAR)
    sidecar["frame_convention"]["tangent_order_aligned"] = False
    path = _write_centerline(tmp_path, _centerline_rows(_STRAIGHT), sidecar)

    with pytest.raises(centerline_csv.CenterlineReadError, match="tangent_order_aligned"):
        centerline_csv.read_sidecar(path)


def test_the_five_semantics_are_left_blank_not_defaulted(tmp_path):
    """0067裁决那五条语义不许有默认值——工具输出里它必须是`None`。

    与`mesh_aabb`把`units`/`convexity`留空是同一条纪律：
    **填一个默认值会被当成"工具查过了"。**
    """

    path = _write_centerline(tmp_path, _centerline_rows(_STRAIGHT))
    record = centerline_csv.describe(path)

    assert record["centerline_semantics"] is None


def test_the_endpoint_gap_is_reported_not_judged(tmp_path):
    """闭合缺口只报不裁：闭不闭合由`CenterlineSemantics.topology`说了算。"""

    path = _write_centerline(tmp_path, _centerline_rows(_STRAIGHT))
    record = centerline_csv.describe(path)

    assert record["endpoint_gap_mm"] == pytest.approx(2.0)
    assert "topology" not in record
    #: 生产者自报的与工具数出来的**并排放着**，工具不挑一个。
    assert record["producer_reported_sample_count"] == 3
    assert record["station_count"] == 3


def test_a_different_ordered_basis_is_refused(tmp_path):
    """sidecar换了基序必须红。

    **这条是注错验证补出来的**：第一版有这个判据、没有这条用例，
    把判据整条拿掉后18条门全绿——它是一道没有必红用例的门，
    也就是plans/09教训三说的"从没被执行过的分支"。
    """

    import copy

    sidecar = copy.deepcopy(_SIDECAR)
    sidecar["frame_convention"]["ordered_basis"] = [
        "tangent",
        "surface_normal",
        "width_direction",
    ]
    path = _write_centerline(tmp_path, _centerline_rows(_STRAIGHT), sidecar)

    with pytest.raises(centerline_csv.CenterlineReadError, match="ordered_basis"):
        centerline_csv.read_sidecar(path)


def test_a_slightly_non_orthogonal_frame_is_refused(tmp_path):
    """`t·n ≠ 0`要单独判，而它的可达窗口很窄——**窄不等于空**。

    若`t·n = c`，则`|n × t| = sqrt(1−c²)`，于是`c`大到一定程度会先被
    "`width_direction`不是单位向量"挡住。两条判据一起把可达窗口压到
    大约`1e-9 < |c| ≲ 4.5e-5`：本用例取`c = 1e-6`，它落在窗口里——
    单位判据与`s = n×t`判据都过得去，只有正交判据会红。

    **这条同样是注错验证补出来的**：拿掉正交判据后原有的门一条都不红。
    """

    import math

    c = 1e-6
    tangent = (1.0, 0.0, 0.0)
    normal = (c, 0.0, math.sqrt(1.0 - c * c))
    rows = _centerline_rows([(0.0, tangent, normal)]).splitlines()
    header, first = rows[0], rows[1].split(",")
    #: `n × t`的模是`sqrt(1−c²)`，差单位1个`c²/2`＝5e-13——比1e-9容差还小，
    #: 所以要手工把`s`归一化，否则先撞上单位判据、正交判据仍然不被执行。
    raw = [float(value) for value in first[12:15]]
    scale = 1.0 / math.sqrt(sum(value * value for value in raw))
    first[12:15] = [repr(value * scale) for value in raw]
    path = _write_centerline(tmp_path, "\n".join([header, ",".join(first)]) + "\n")

    with pytest.raises(centerline_csv.CenterlineReadError, match="不正交"):
        centerline_csv.read_stations(path)


#: **真实语料的门，选择进入。** 0073第四节裁决"本目录永远不放真实资产"
#: （Bullet仓297MB而物理库只占2.4%那条单向门是判例），所以真实中心线不进仓；
#: 但"工具在真实字节上跑得动"这件事不能只靠合成三行来声称。
#: 形制抄`tests/test_provenance.py`已有的`PE_REPLAY_CASE_RUNS`：**指了才跑，不指明示skip**。
REAL_CENTERLINE = os.environ.get("PE_REAL_CENTERLINE_CSV")


@pytest.mark.skipif(
    not REAL_CENTERLINE,
    reason="set PE_REAL_CENTERLINE_CSV to a GCW centerline.csv (with its centerline.meta.json)",
)
def test_a_real_gcw_centerline_reads_and_builds_kernel_stations():
    """真实语料端到端：GCW的导出 → 工具 → 内核`GrooveStation`，一条都不许掉。

    **合成用例证明不了的两件事**：真实导出有六百多行、
    帧正交性只到浮点精度（实测最坏`|s − n×t| = 5.8e-16`）；
    以及sidecar的真实字段集比测试里造的那份大得多，多出来的字段不许让读取器崩。
    """

    from physics_engine.laydown import GrooveStation

    path = Path(REAL_CENTERLINE)
    record = centerline_csv.describe(path)
    stations = [GrooveStation(*station) for station in centerline_csv.read_stations(path)]

    assert len(stations) == record["station_count"] > 100
    assert record["arc_length_total_mm"] > 0.0
    #: 采样步在真实导出里是近似等距的；**只判同量级，不判等**——
    #: 判等会把"上游换了采样策略"这件事误报成读取器坏了。
    assert record["arc_step_max_mm"] < 10.0 * record["arc_step_min_mm"]
    #: 生产者自报的行数与工具数出来的必须一致；不一致说明有行被静默丢了。
    assert record["producer_reported_sample_count"] == record["station_count"]
