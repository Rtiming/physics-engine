#!/usr/bin/env python3
"""生成`cases/mesh_asset_integrity/`的场景语料与清单——资产完整性与包盒保守性。

**这是今天就必红的一条**：引擎至今没有任何门能发现"声明的SHA对得上、
声明的包盒却与资产真值在某个轴上完全不相交"这类错误（decisions/0017第四条
记的正是这个真实缺陷）。本案例建的就是那道门。

两条判据，都零容差：

1. `sha256(资产字节) == 场景里声明的sha256`；
2. `direction="envelope"`时，逐轴`declared_min ≤ true_min` 且 `declared_max ≥ true_max`。

**`fitted`语义不同**：贴合形不承诺包住资产（它是"贴着物体的近似形"，
可以比资产小），第2条对它不成立、也不许套用——门遇到`fitted`拒绝下判决
而不是判它红。区分见`case.md`第六节。

金标来源（轴7规则1）：资产的顶点表是**本案例自己定义的**（见
`assets/generate_tetra.py`的文档），真实包盒由顶点表手推，值全部是二进制小数
因而精确。测试侧从STL字节解析出包盒，与这份手推值对拍——两条路径不共享代码。

用法：`PYTHONPATH=src .venv/bin/python cases/mesh_asset_integrity/generate_oracle.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.canonical import FTS_PROFILE, canonical_file_bytes  # noqa: E402
from physics_engine.oracles import (  # noqa: E402
    ORACLE_MANIFEST_FACET,
    ORACLE_MANIFEST_VERSION,
    file_sha256,
    write_manifest,
)

ALGORITHM_ID = "algorithm:oracle/mesh_asset_integrity"
ALGORITHM_VERSION = "1.0.0"

ASSET_RELATIVE = "cases/mesh_asset_integrity/assets/tetra.stl"

#: 由`assets/generate_tetra.py`的四个顶点手推：逐轴取min/max。
#: v0=(−3.5,1.25,−2)、v1=(26.5,1.25,−2)、v2=(1.5,25.25,−2)、v3=(4.5,7.25,16)。
TRUE_AABB_MIN = (-3.5, 1.25, -2.0)
TRUE_AABB_MAX = (26.5, 25.25, 16.0)
TRIANGLE_COUNT = 4

#: 合格包络：真值向外取整到0.5mm。`envelope`语义要求它把资产整个包住。
ENVELOPE_MIN = (-4.0, 1.0, -2.5)
ENVELOPE_MAX = (27.0, 25.5, 16.5)

#: 必红语料一：z轴与真值完全不相交（这正是`examples/`那条真实缺陷的形状——
#: 声明[−300,300]、真值[1133.9,1559.7]，z轴毫无交集），外加x轴下界被砍进资产里。
#: 两个轴一起破，是为了验门**逐轴**报违规而不是只报第一条。
WRONG_AABB_MIN = (0.0, 1.0, 100.0)
WRONG_AABB_MAX = (27.0, 25.5, 120.0)

EXACT_REASON_SHA = (
    "内容寻址是逐字节判据（轴3规则1），容差概念不适用——零容差不是『选了个很严的数』，"
    "是『这里没有连续量』。SHA对不上说明字节变了，没有『变得不多』这回事。"
)
EXACT_REASON_ENVELOPE = (
    "包络保守性是不等式判据（declared_min ≤ true_min 且 declared_max ≥ true_max），"
    "零容差=不给不等式任何松弛。给了松弛就等于声明『漏出去一点点没关系』，"
    "而漏出去的那一点点正是broad phase会漏掉的接触。"
)
EXACT_REASON_GEOMETRY = (
    "顶点坐标全是±0.25的整数倍，float32存储无舍入、解析回float64精确；"
    "abs=1e-12mm是对『这里没有误差源』的声明，不写==0是不把解析实现的求值次序冻进契约。"
)
EXACT_REASON_COUNT = "三角形数与文件字节数是确定性整数，零容差。"


def _scene(scene_id: str, description: str, direction: str, sha256: str, low, high) -> dict:
    return {
        "contract_type": "physics_scene",
        "contract_version": "1.0.0",
        "scene_id": scene_id,
        "description": description,
        "extensions": [],
        "bodies": [
            {
                "body_id": "body/tetra",
                "collision": {
                    "direction": direction,
                    "shape": {
                        "kind": "mesh",
                        "path_relative": ASSET_RELATIVE,
                        "sha256": sha256,
                        "units": "mm",
                        "usage": "collision",
                        "convexity": "exact_convex",
                        "aabb_min_mm": list(low),
                        "aabb_max_mm": list(high),
                    },
                },
            }
        ],
        "allowed_pairs": [],
    }


def _corrupt(sha256: str) -> str:
    """改一个十六进制字符——『声明的SHA与资产不符』的最小反例。"""

    tail = "0" if sha256[-1] != "0" else "1"
    return sha256[:-1] + tail


def main() -> int:
    asset_sha256 = file_sha256(ROOT / ASSET_RELATIVE)
    asset_bytes = (ROOT / ASSET_RELATIVE).stat().st_size

    scenes = {
        "tetra_envelope.scene.json": (
            _scene(
                "scene/tetra_envelope",
                "合格语料：SHA对得上，包络把资产整个包住。",
                "envelope",
                asset_sha256,
                ENVELOPE_MIN,
                ENVELOPE_MAX,
            ),
            {
                "sha256_matches": True,
                "envelope_encloses_asset": True,
                "violated_axes": [],
            },
        ),
        "red_wrong_aabb.scene.json": (
            _scene(
                "scene/red_wrong_aabb",
                "必红语料：SHA对得上，但声明包盒的z轴与资产真值完全不相交、x轴下界砍进资产。",
                "envelope",
                asset_sha256,
                WRONG_AABB_MIN,
                WRONG_AABB_MAX,
            ),
            {
                "sha256_matches": True,
                "envelope_encloses_asset": False,
                "violated_axes": ["min_x", "min_z"],
            },
        ),
        "red_wrong_sha.scene.json": (
            _scene(
                "scene/red_wrong_sha",
                "必红语料：包络合格，但声明的SHA与资产字节不符。",
                "envelope",
                _corrupt(asset_sha256),
                ENVELOPE_MIN,
                ENVELOPE_MAX,
            ),
            {
                "sha256_matches": False,
                "envelope_encloses_asset": True,
                "violated_axes": [],
            },
        ),
    }
    #: 只给"门遇到fitted必须拒判"那条测试用，不是oracle——它验的是门的行为，
    #: 不是某个物理量的值。
    fitted = _scene(
        "scene/tetra_fitted",
        "非oracle语料：direction=fitted，包盒比资产小是合法的，本判据不适用。",
        "fitted",
        asset_sha256,
        (-1.0, 2.0, -1.0),
        (20.0, 20.0, 10.0),
    )
    (HERE / "fitted_not_judged.scene.json").write_bytes(
        canonical_file_bytes(fitted, FTS_PROFILE)
    )

    oracles = []
    for filename, (document, verdict) in scenes.items():
        payload = canonical_file_bytes(document, FTS_PROFILE)
        (HERE / filename).write_bytes(payload)
        expected = dict(verdict)
        expected.update(
            {
                "true_aabb_min_mm": list(TRUE_AABB_MIN),
                "true_aabb_max_mm": list(TRUE_AABB_MAX),
                "triangle_count": TRIANGLE_COUNT,
                "asset_bytes": asset_bytes,
            }
        )
        oracles.append(
            {
                "id": f"oracle:mesh_asset_integrity/{filename.split('.')[0]}",
                "inputs": {
                    "scene_path_relative": f"cases/mesh_asset_integrity/{filename}",
                    "scene_sha256": file_sha256(HERE / filename),
                    "asset_path_relative": ASSET_RELATIVE,
                    "asset_generator_path_relative": (
                        "cases/mesh_asset_integrity/assets/generate_tetra.py"
                    ),
                    "asset_generator_sha256": file_sha256(HERE / "assets/generate_tetra.py"),
                    "direction": "envelope",
                },
                "expected": expected,
                "tolerances": {
                    "sha256_matches": {"abs": 0.0, "rel": 0.0, "reason": EXACT_REASON_SHA},
                    "envelope_encloses_asset": {
                        "abs": 0.0, "rel": 0.0, "reason": EXACT_REASON_ENVELOPE,
                    },
                    "violated_axes": {"abs": 0.0, "rel": 0.0, "reason": EXACT_REASON_ENVELOPE},
                    "true_aabb_min_mm": {
                        "abs": 1.0e-12, "rel": 0.0, "reason": EXACT_REASON_GEOMETRY,
                    },
                    "true_aabb_max_mm": {
                        "abs": 1.0e-12, "rel": 0.0, "reason": EXACT_REASON_GEOMETRY,
                    },
                    "triangle_count": {"abs": 0.0, "rel": 0.0, "reason": EXACT_REASON_COUNT},
                    "asset_bytes": {"abs": 0.0, "rel": 0.0, "reason": EXACT_REASON_COUNT},
                },
            }
        )

    document = {
        "facet": ORACLE_MANIFEST_FACET,
        "facet_version": ORACLE_MANIFEST_VERSION,
        "case_id": "case/mesh_asset_integrity",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/mesh_asset_integrity/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(
        f"asset sha256={asset_sha256} bytes={asset_bytes}; "
        f"wrote {len(scenes)}+1 scenes and a {len(written)}B manifest"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
