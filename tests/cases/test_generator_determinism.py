"""`case/generator_determinism`的conformance门（轴7规则3）。

**这是引擎第一次有真实的"参数→形状"纯函数**（spec/11规则2第二类）。
三条判据各验一件不同的事：

1. **确定性**——同参数逐字节相同，连换`PYTHONHASHSEED`起子进程都相同；
2. **一位变化必反映**——22条一位扰动全部改变声明字节；
3. **产的形是真形**——喂给`geometry.mass_properties`能算，且与教科书闭式一致。

第三条是**独立**判据：它不问生成器"你自洽吗"，而是拿另一个模块的解析质量属性
去对一个教材闭式解。前两条自洽了但第三条红，说明生成器稳定地产着错的形。

判据数全部来自清单，测试只读不算（spec/08规则3）。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from physics_engine.canonical import canonical_bytes
from physics_engine.geometry import mass_properties, shift_inertia_kg_mm2
from physics_engine.modelgen import (
    MODELGEN_PROFILE,
    declaration_bytes,
    declaration_document,
    declaration_sha256,
    generate_former,
    generate_roller,
    generate_spool,
)
from physics_engine.oracles import load_manifest
from physics_engine.shapes import FiniteCylinder

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/generator_determinism/oracle.json", root=ROOT)

GENERATORS = {"spool": generate_spool, "roller": generate_roller, "former": generate_former}

#: 参与齐次性比较的长度键。比值不在其中（它们本来就不该随L变），
#: `parameters`整块被排除——那里面既有随L变的特征长度也有不变的比值。
LENGTH_KEYS = frozenset(
    {"offset_mm", "radius_mm", "half_width_mm", "point_a_mm", "point_b_mm",
     "fillet_radius_mm", "half_extents_mm"}
)

#: 子进程程序：从stdin读调用表，打印各自的声明SHA。**故意不读清单**——
#: 它只需要复现同一次调用，越少的共享状态越能证明确定性来自函数本身。
_CHILD_PROGRAM = """
import json, sys
from physics_engine import modelgen
out = []
for name, kwargs in json.load(sys.stdin):
    kwargs = dict(kwargs)
    if "skeleton_ratios" in kwargs:
        kwargs["skeleton_ratios"] = [tuple(point) for point in kwargs["skeleton_ratios"]]
    parts = getattr(modelgen, "generate_" + name)(**kwargs)
    out.append(modelgen.declaration_sha256(parts))
print(json.dumps(out))
"""


def _call(name: str, kwargs: dict) -> tuple:
    payload = dict(kwargs)
    if "skeleton_ratios" in payload:
        payload["skeleton_ratios"] = [tuple(point) for point in payload["skeleton_ratios"]]
    return GENERATORS[name](**payload)


def _calls(entry) -> list[tuple[str, dict]]:
    return [
        (name, dict(entry.inputs[f"{name}_call"]))
        for name in ("spool", "roller", "former")
    ]


def _geometry_fingerprint(parts) -> bytes:
    """去掉参数表之后的声明字节——"形几何本身"的指纹。"""

    document = declaration_document(parts)
    for part in document["parts"]:
        part.pop("parameters")
    return canonical_bytes(document, MODELGEN_PROFILE)


def _lengths(parts) -> list[float]:
    collected: list[float] = []
    for part in declaration_document(parts)["parts"]:
        collected.extend(part["offset_mm"])
        for key, value in sorted(part["shape"].items()):
            if key not in LENGTH_KEYS or value is None:
                continue
            collected.extend(value if isinstance(value, list) else [value])
    return collected


def _perturbations(kwargs: dict) -> list[tuple[str, dict]]:
    """每个入参各扰一位：浮点走`nextafter`，整数`+1`，骨架点逐分量。"""

    out: list[tuple[str, dict]] = []
    for key, value in kwargs.items():
        if key == "skeleton_ratios":
            for index in range(len(value)):
                for axis in range(3):
                    points = [list(entry) for entry in value]
                    points[index][axis] = math.nextafter(points[index][axis], math.inf)
                    out.append((f"{key}[{index}][{axis}]", {**kwargs, key: points}))
        elif isinstance(value, int) and not isinstance(value, bool):
            out.append((key, {**kwargs, key: value + 1}))
        else:
            out.append((key, {**kwargs, key: math.nextafter(value, math.inf)}))
    return out


def _child_shas(calls: list[tuple[str, dict]], seed: str) -> list[str]:
    environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT / "src"))
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        input=json.dumps(calls), env=environment,
        capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


def _diagonal(matrix) -> list[float]:
    return [matrix[index][index] for index in range(3)]


def _offdiag_max(matrix) -> float:
    return max(
        abs(matrix[row][column])
        for row in range(3)
        for column in range(3)
        if row != column
    )


def test_the_same_parameters_produce_the_same_declaration_bytes():
    """确定性：进程内两次 + 两个不同`PYTHONHASHSEED`的子进程，四份声明同一份字节。"""

    entry = MANIFEST.oracle("oracle:modelgen/determinism")
    calls = _calls(entry)

    repeats_identical = all(
        declaration_bytes(_call(name, kwargs)) == declaration_bytes(_call(name, kwargs))
        for name, kwargs in calls
    )
    here = [declaration_sha256(_call(name, kwargs)) for name, kwargs in calls]
    seeds = list(entry.inputs["hash_seeds"])
    cross_identical = all(_child_shas(calls, seed) == here for seed in seeds)

    # `-0.0`与`0.0`是同一个几何点，必须给同一份字节。
    former_kwargs = dict(entry.inputs["former_call"])
    signed = [list(point) for point in former_kwargs["skeleton_ratios"]]
    signed[0] = [-0.0, -0.0, -0.0]
    negative_zero_ok = declaration_bytes(
        _call("former", {**former_kwargs, "skeleton_ratios": signed})
    ) == declaration_bytes(_call("former", former_kwargs))

    entry.check_all({
        "repeat_calls_identical": repeats_identical,
        "cross_process_identical": cross_identical,
        "negative_zero_normalised": negative_zero_ok,
        "perturbation_count": entry.expected["perturbation_count"],
        "perturbations_changing_declaration": entry.expected[
            "perturbations_changing_declaration"
        ],
        "absorbed_perturbation_labels": entry.expected["absorbed_perturbation_labels"],
    })


def test_every_one_bit_parameter_change_changes_the_declaration():
    """一位变化必反映：22条扰动全部改变声明字节；被形几何吃掉的那条逐名对上。"""

    entry = MANIFEST.oracle("oracle:modelgen/determinism")
    total = 0
    changed = 0
    absorbed: list[str] = []
    for name, kwargs in _calls(entry):
        base_sha = declaration_sha256(_call(name, kwargs))
        base_geometry = _geometry_fingerprint(_call(name, kwargs))
        for label, perturbed in _perturbations(kwargs):
            total += 1
            parts = _call(name, perturbed)
            if declaration_sha256(parts) != base_sha:
                changed += 1
            if _geometry_fingerprint(parts) == base_geometry:
                absorbed.append(f"{name}.{label}")

    entry.check_all({
        "repeat_calls_identical": entry.expected["repeat_calls_identical"],
        "cross_process_identical": entry.expected["cross_process_identical"],
        "negative_zero_normalised": entry.expected["negative_zero_normalised"],
        "perturbation_count": total,
        "perturbations_changing_declaration": changed,
        "absorbed_perturbation_labels": sorted(absorbed),
    })


def test_scaling_the_characteristic_length_scales_every_length_bit_exactly():
    """齐次性：特征长度乘2，产出的每一个长度逐位等于原值的2倍。

    这是"系数化不写死毫米"的可执行形式（case2 `robot_links.py`§J1）——
    任何一个写死的毫米量都不会跟着L走，当场破门。
    """

    entry = MANIFEST.oracle("oracle:modelgen/scale_homogeneity")
    factor = entry.inputs["scale_factor"]
    compared = 0
    exact = 0
    for name, kwargs in _calls(entry):
        base = _lengths(_call(name, kwargs))
        scaled = _lengths(_call(name, {
            **kwargs,
            "characteristic_length_mm": factor * kwargs["characteristic_length_mm"],
        }))
        assert len(base) == len(scaled), (
            f"{name}：缩放前后长度个数不同（{len(base)} vs {len(scaled)}）——"
            "件数随尺度变了，那不是齐次性问题而是结构问题"
        )
        compared += len(base)
        exact += sum(1 for a, b in zip(base, scaled, strict=True) if factor * a == b)

    entry.check_all({
        "compared_length_count": compared,
        "lengths_scaling_exactly": exact,
    })


def test_the_spool_products_are_real_shapes_with_closed_form_mass_properties():
    """带盘：3件、无一件带法兰字段、逐件与整装的质量属性都对上教科书闭式。"""

    entry = MANIFEST.oracle("oracle:modelgen/spool_mass_properties")
    parts = _call("spool", dict(entry.inputs["call"]))
    density = entry.inputs["density_kg_m3"]

    flanged = any(
        isinstance(part.shape.shape, FiniteCylinder)
        and part.shape.shape.flange_outer_radius_mm is not None
        for part in parts
    )
    properties = [mass_properties(part.shape, density_kg_m3=density) for part in parts]
    barrel, flange_low, flange_high = properties

    assembly_mass = barrel.mass_kg + flange_low.mass_kg + flange_high.mass_kg
    offset = entry.inputs["flange_offset_mm"]
    weighted = [
        sum(
            props.mass_kg * (props.centroid_mm[axis] + part.offset_mm[axis])
            for props, part in zip(properties, parts, strict=True)
        ) / assembly_mass
        for axis in range(3)
    ]
    shifted_low = shift_inertia_kg_mm2(
        flange_low.inertia_about_centroid_kg_mm2, flange_low.mass_kg, (0.0, 0.0, -offset)
    )
    shifted_high = shift_inertia_kg_mm2(
        flange_high.inertia_about_centroid_kg_mm2, flange_high.mass_kg, (0.0, 0.0, offset)
    )
    assembly_diag = [
        barrel.inertia_about_centroid_kg_mm2[axis][axis]
        + shifted_low[axis][axis]
        + shifted_high[axis][axis]
        for axis in range(3)
    ]

    entry.check_all({
        "part_ids": [part.part_id for part in parts],
        "any_flanged_cylinder_produced": flanged,
        "barrel_volume_mm3": barrel.volume_mm3,
        "barrel_mass_kg": barrel.mass_kg,
        "barrel_inertia_diag_kg_mm2": _diagonal(barrel.inertia_about_centroid_kg_mm2),
        "flange_volume_mm3": flange_low.volume_mm3,
        "flange_mass_kg": flange_low.mass_kg,
        "flange_inertia_diag_kg_mm2": _diagonal(flange_low.inertia_about_centroid_kg_mm2),
        "assembly_volume_mm3": sum(props.volume_mm3 for props in properties),
        "assembly_mass_kg": assembly_mass,
        "assembly_centroid_mm": weighted,
        "assembly_inertia_diag_kg_mm2": assembly_diag,
        "inertia_offdiag_max_kg_mm2": max(
            _offdiag_max(props.inertia_about_centroid_kg_mm2) for props in properties
        ),
    })


def test_the_roller_product_is_a_real_finite_cylinder():
    """导轮：半径与半宽逐位、质量属性对闭式。半宽那条钉的是全宽/半宽口径。"""

    entry = MANIFEST.oracle("oracle:modelgen/roller_mass_properties")
    parts = _call("roller", dict(entry.inputs["call"]))
    shape = parts[0].shape.shape
    assert isinstance(shape, FiniteCylinder)
    properties = mass_properties(parts[0].shape, density_kg_m3=entry.inputs["density_kg_m3"])

    entry.check_all({
        "part_ids": [part.part_id for part in parts],
        "radius_mm": shape.radius_mm,
        "half_width_mm": shape.half_width_mm,
        "volume_mm3": properties.volume_mm3,
        "mass_kg": properties.mass_kg,
        "inertia_diag_kg_mm2": _diagonal(properties.inertia_about_centroid_kg_mm2),
        "inertia_offdiag_max_kg_mm2": _offdiag_max(
            properties.inertia_about_centroid_kg_mm2
        ),
    })


def test_the_former_capsules_obey_the_closed_form_and_the_universal_inertia_laws():
    """骨架：锥度半径与质心逐位、体积对闭式；惯量只验对称性与主矩三角不等式。

    后两条是**普适律**（任何真实质量分布都满足），与胶囊闭式无关——
    本案例不独立验证胶囊惯量的数值，见案例页第四节。
    """

    entry = MANIFEST.oracle("oracle:modelgen/former_mass_properties")
    parts = _call("former", dict(entry.inputs["call"]))
    density = entry.inputs["density_kg_m3"]
    properties = [mass_properties(part.shape, density_kg_m3=density) for part in parts]

    symmetric = True
    triangle = True
    for props in properties:
        inertia = props.inertia_about_centroid_kg_mm2
        for row in range(3):
            for column in range(3):
                symmetric = symmetric and inertia[row][column] == inertia[column][row]
        diagonal = _diagonal(inertia)
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            triangle = triangle and diagonal[i] + diagonal[j] >= diagonal[k]

    entry.check_all({
        "part_ids": [part.part_id for part in parts],
        "link_radii_mm": [part.shape.shape.radius_mm for part in parts],
        "link_volumes_mm3": [props.volume_mm3 for props in properties],
        "link_masses_kg": [props.mass_kg for props in properties],
        "link_centroids_mm": [list(props.centroid_mm) for props in properties],
        "inertia_symmetric": symmetric,
        "inertia_triangle_inequality_holds": triangle,
        "inertia_offdiag_max_kg_mm2": max(
            _offdiag_max(props.inertia_about_centroid_kg_mm2) for props in properties
        ),
    })
