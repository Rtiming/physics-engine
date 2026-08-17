"""`tools/model/`自己的门（决策0073）。

本文件判两件事：工具**算得对**（对既有案例金标独立复现），
以及它**失败关闭**（认不出的格式、不自洽的字节、留白的判断一律不猜）。

工具的身份边界（不进内核、不进wheel）由`tests/governance/`那道门守。
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tools.model import mesh_aabb

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
