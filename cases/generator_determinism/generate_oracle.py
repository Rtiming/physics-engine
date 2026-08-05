#!/usr/bin/env python3
"""参数化生成器的金标生成器——**闭式解与手推常数，独立于被验内核**。

本脚本**不import`physics_engine.modelgen`，也不import`physics_engine.geometry`**。
它写三样东西：

1. **确定性/一位变化/齐次性的判据**，全是布尔与计数——它们是关系型判据
   （"两次调用的字节相等"、"22条扰动全部改变声明"），不冻结任何一份具体字节。
   这是有意的：声明指纹的形制**不是已登记的面**，把它的哈希冻进金标等于让
   一个未登记的字节形制长出依赖者。
2. **质量属性的教科书闭式解**：实心圆柱与胶囊的体积/质心/绕质心惯量。
   圆柱三式（`V = πR²W`、`I_zz = ½mR²`、`I_xx = I_yy = m(3R² + W²)/12`）
   是任何一本刚体动力学教材的标准结果（如Beer & Johnston,
   *Vector Mechanics for Engineers*, 附录B 均质体惯量表）。
3. **锥度插值的手推值**：不写插值公式，直接写手算结果（见下方推导注释）。

参数刻意取**二进制精确**的值（L与各比值都是2的幂之和，如0.375=3/8、
0.0625=1/16、0.001953125=1/512），于是`比值×L`这一步**无舍入**，
两条路径之间剩下的差就只是乘法结合次序——判据表里那条容差算的正是它。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/generator_determinism"
ALGORITHM_VERSION = "1.0.0"

#: 通用碳钢名义密度。它是**输入**不是被验量——只为把体积换成质量、
#: 让惯量有个真实量级，不是对任何一个真实零件的材料声明。
DENSITY_KG_M3 = 7850.0

#: 三次生成器调用的入参。全部为二进制精确值（见模块文档）。
SPOOL_CALL = {
    "characteristic_length_mm": 256.0,
    "barrel_radius_ratio": 0.25,
    "barrel_width_ratio": 0.375,
    "flange_outer_radius_ratio": 0.4375,
    "flange_width_ratio": 0.0625,
    "wound_layers": 8,
    "layer_thickness_ratio": 0.001953125,
}
ROLLER_CALL = {
    "characteristic_length_mm": 128.0,
    "radius_ratio": 0.375,
    "face_width_ratio": 0.125,
}
FORMER_CALL = {
    "characteristic_length_mm": 512.0,
    "skeleton_ratios": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.25, 0.0]],
    "root_radius_ratio": 0.0625,
    "tip_radius_ratio": 0.03125,
}

# --- 由入参手算出的毫米尺寸（每一步都无舍入，因为比值与L都是二进制精确） ---
# 带盘：R_eff = (0.25 + 8×0.001953125)×256 = 0.265625×256 = 68.0
#       W     = 0.375×256 = 96.0；R_f = 0.4375×256 = 112.0；w = 0.0625×256 = 16.0
#       法兰盘心的轴向偏移 d = W/2 + w/2 = 48 + 8 = 56.0
BARREL_RADIUS_MM, BARREL_WIDTH_MM = 68.0, 96.0
FLANGE_RADIUS_MM, FLANGE_WIDTH_MM = 112.0, 16.0
FLANGE_OFFSET_MM = 56.0
# 导轮：R = 0.375×128 = 48.0；面宽 = 0.125×128 = 16.0
ROLLER_RADIUS_MM, ROLLER_WIDTH_MM = 48.0, 16.0
# 骨架：点列×512 → (0,0,0)、(256,0,0)、(256,128,0)；两段长 256.0 与 128.0。
#       锥度按段中点插值，两段的t是0.25与0.75（**手推，不写公式**）：
#         r_0 = (0.0625 + (0.03125 − 0.0625)×0.25)×512 = 0.0546875×512 = 28.0
#         r_1 = (0.0625 + (0.03125 − 0.0625)×0.75)×512 = 0.0390625×512 = 20.0
LINK_RADII_MM = (28.0, 20.0)
LINK_HEIGHTS_MM = (256.0, 128.0)
LINK_CENTROIDS_MM = ([128.0, 0.0, 0.0], [256.0, 64.0, 0.0])

#: 一位扰动的条数：带盘6个浮点+1个整数、导轮3个浮点、骨架3个浮点+9个骨架分量。
PERTURBATION_COUNT = 6 + 1 + 3 + 3 + 9
#: 只改参数表、不改形几何的那些扰动。`0.25 + 8t`把`t`的末位吃掉了——
#: 8×ulp(2⁻⁹)=2⁻⁵⁸远小于½ulp(0.265625)=2⁻⁵⁵。**这条正是"参数必须记进产出"的理由**。
ABSORBED_LABELS = ["spool.layer_thickness_ratio"]
#: 齐次性比较的长度个数：带盘15（3件×3偏移+1×2+2×2）、导轮5、骨架20（2件×3偏移+2×7）。
COMPARED_LENGTH_COUNT = 15 + 5 + 20
#: 缩放因子取2：2的幂缩放在IEEE-754二进制浮点下**无舍入**（除非溢出/次正规），
#: 所以齐次性判据可以是**逐位相等**，而不是一条数值容差。
SCALE_FACTOR = 2.0

#: 质量属性的相对容差。两条路径只差乘法结合次序：`geometry.mass_properties`
#: 先算体积、再用`scale = m/V`把质量乘回形状矩，oracle直接把质量乘进闭式——
#: 多出一次除法与一次乘法，界约3×(eps/2)=3.3e-16。实测最大2.34e-16
#: （法兰I_zz），1e-15是实测值的4.3倍、解析界的3倍余量。
#: **这不是"放宽到能过"**：把它收到2e-16实测即红——带盘的
#: `assembly_inertia_diag_kg_mm2[2]`与导轮的`inertia_diag_kg_mm2[0]`同时破
#: （`check_all`按字母序比，带盘那边先报到装配那条）。
MASS_PROPERTY_REL = 1.0e-15


def cylinder_closed_form(radius_mm: float, width_mm: float) -> dict[str, object]:
    """实心圆柱（轴向z）的体积/质量/绕质心惯量对角元。教科书闭式。"""

    volume = math.pi * radius_mm * radius_mm * width_mm
    mass = DENSITY_KG_M3 * volume / 1.0e9
    axial = 0.5 * mass * radius_mm * radius_mm
    transverse = mass * (3.0 * radius_mm * radius_mm + width_mm * width_mm) / 12.0
    return {"volume": volume, "mass": mass, "diag": [transverse, transverse, axial]}


def capsule_volume_mm3(radius_mm: float, height_mm: float) -> float:
    """胶囊 = 圆柱段 + 两个半球帽：`V = πr²h + (4/3)πr³`。"""

    return math.pi * radius_mm * radius_mm * height_mm + 4.0 * math.pi * radius_mm**3 / 3.0


def _zero(reason: str) -> dict[str, object]:
    return {"abs": 0.0, "rel": 0.0, "reason": reason}


def _mass(reason: str) -> dict[str, object]:
    return {"abs": 0.0, "rel": MASS_PROPERTY_REL, "reason": reason}


def determinism_oracle() -> dict[str, object]:
    return {
        "id": "oracle:modelgen/determinism",
        "inputs": {
            "spool_call": SPOOL_CALL,
            "roller_call": ROLLER_CALL,
            "former_call": FORMER_CALL,
            "hash_seeds": ["0", "1"],
        },
        "expected": {
            "repeat_calls_identical": True,
            "cross_process_identical": True,
            "negative_zero_normalised": True,
            "perturbation_count": PERTURBATION_COUNT,
            "perturbations_changing_declaration": PERTURBATION_COUNT,
            "absorbed_perturbation_labels": ABSORBED_LABELS,
        },
        "tolerances": {
            "repeat_calls_identical": _zero(
                "同一进程内两次调用的`declaration_bytes`逐字节相等。纯函数的最低门。"
            ),
            "cross_process_identical": _zero(
                "**换`PYTHONHASHSEED`起两个子进程**，`declaration_sha256`与本进程相同。"
                "进程内比两次抓不到集合/字典哈希序这类隐患——同一进程里哈希种子是同一个，"
                "两次调用会一起错。跨种子子进程是唯一能抓到它的口径。"
            ),
            "negative_zero_normalised": _zero(
                "骨架坐标写`-0.0`与写`0.0`产出同一份字节。`-0.0 == 0.0`为真而"
                "`json.dumps`给出`\"-0.0\"`与`\"0.0\"`——同一个几何点两份声明，"
                "是逐字节判据的真实隐患，不是理论顾虑。"
            ),
            "perturbation_count": _zero(
                "扰动条数本身冻进金标：**防止判据变空**。少扰几条时"
                "「全部改变」会自动成立，这个数把它钉死（带盘6浮点+1整数、"
                "导轮3、骨架3+9=12，共22）。"
            ),
            "perturbations_changing_declaration": _zero(
                "22条一位扰动**全部**改变声明字节。与上一个数相等才算通过——"
                "两个数分开写是为了红的时候能一眼看出是「漏扰」还是「扰了没变」。"
            ),
            "absorbed_perturbation_labels": _zero(
                "只改参数表、不改形几何的扰动**名单**（不是个数）：`layer_thickness_ratio`"
                "的末位被`0.25 + 8t`的加法吃掉。名单必须逐字相等——名单变长意味着"
                "有新的参数不再影响几何，那是参数面该重新审的信号。"
            ),
        },
    }


def homogeneity_oracle() -> dict[str, object]:
    return {
        "id": "oracle:modelgen/scale_homogeneity",
        "inputs": {
            "spool_call": SPOOL_CALL,
            "roller_call": ROLLER_CALL,
            "former_call": FORMER_CALL,
            "scale_factor": SCALE_FACTOR,
        },
        "expected": {
            "compared_length_count": COMPARED_LENGTH_COUNT,
            "lengths_scaling_exactly": COMPARED_LENGTH_COUNT,
        },
        "tolerances": {
            "compared_length_count": _zero(
                "参与比较的长度个数（带盘15+导轮5+骨架20=40）。同样是**防止判据变空**："
                "漏采一半长度时「全部逐位相等」会自动成立。"
            ),
            "lengths_scaling_exactly": _zero(
                "特征长度乘2后，产出的每一个长度**逐位**等于原值的2倍。"
                "零容差是算出来的而不是乐观：2的幂缩放在IEEE-754下无舍入"
                "（尾数不变、指数加1，除非溢出或落进次正规），所以这是一条位判据。"
                "**任何写死的毫米量都会当场破它**——这就是"
                "「系数化不写死毫米」的可执行形式（case2 robot_links §J1）。"
            ),
        },
    }


def spool_oracle() -> dict[str, object]:
    barrel = cylinder_closed_form(BARREL_RADIUS_MM, BARREL_WIDTH_MM)
    flange = cylinder_closed_form(FLANGE_RADIUS_MM, FLANGE_WIDTH_MM)
    assembly_volume = barrel["volume"] + 2.0 * flange["volume"]  # type: ignore[operator]
    assembly_mass = barrel["mass"] + 2.0 * flange["mass"]  # type: ignore[operator]
    # 平行轴：两片法兰各自搬到装配质心（原点），横向多出m·d²，轴向不变。
    shift = flange["mass"] * FLANGE_OFFSET_MM * FLANGE_OFFSET_MM  # type: ignore[operator]
    assembly_diag = [
        barrel["diag"][0] + 2.0 * (flange["diag"][0] + shift),  # type: ignore[index]
        barrel["diag"][1] + 2.0 * (flange["diag"][1] + shift),  # type: ignore[index]
        barrel["diag"][2] + 2.0 * flange["diag"][2],  # type: ignore[index]
    ]
    return {
        "id": "oracle:modelgen/spool_mass_properties",
        "inputs": {"call": SPOOL_CALL, "density_kg_m3": DENSITY_KG_M3,
                   "flange_offset_mm": FLANGE_OFFSET_MM},
        "expected": {
            "part_ids": ["barrel", "flange_low", "flange_high"],
            "any_flanged_cylinder_produced": False,
            "barrel_volume_mm3": barrel["volume"],
            "barrel_mass_kg": barrel["mass"],
            "barrel_inertia_diag_kg_mm2": barrel["diag"],
            "flange_volume_mm3": flange["volume"],
            "flange_mass_kg": flange["mass"],
            "flange_inertia_diag_kg_mm2": flange["diag"],
            "assembly_volume_mm3": assembly_volume,
            "assembly_mass_kg": assembly_mass,
            "assembly_centroid_mm": [0.0, 0.0, 0.0],
            "assembly_inertia_diag_kg_mm2": assembly_diag,
            "inertia_offdiag_max_kg_mm2": 0.0,
        },
        "tolerances": {
            "part_ids": _zero(
                "件名与件序逐字相等。法兰走**独立第二个形**那条路（spec/11第二之二节"
                "两条候选修法的后者），产物是3件不是1件——件数变了就是那条路被改了。"
            ),
            "any_flanged_cylinder_produced": _zero(
                "产出的任何`FiniteCylinder`的`flange_outer_radius_mm`**恒为None**。"
                "这是上一条的另一半：一旦有人产了带法兰的圆柱，它的质量属性就永远"
                "算不出来（`geometry.mass_properties`对它失败关闭），本案例第三条判据"
                "会从「红」退化成「不可达」——那比红更坏。"
            ),
            "barrel_volume_mm3": _mass("`V = πR²W`，R=68.0、W=96.0均由比值×L精确得到。"),
            "barrel_mass_kg": _mass("`m = ρV/1e9`，mm³→m³是spec/14第五节登记的纯长度换算。"),
            "barrel_inertia_diag_kg_mm2": _mass(
                "`I_xx = I_yy = m(3R²+W²)/12`、`I_zz = ½mR²`（Beer & Johnston附录B）。"
                "实测偏差1.4e-16（I_zz），横向两个恰为0——横向那条两边的乘法次序碰巧一致。"
            ),
            "flange_volume_mm3": _mass("同筒，R=112.0、w=16.0。实测偏差恰为0。"),
            "flange_mass_kg": _mass("同筒。实测偏差恰为0。"),
            "flange_inertia_diag_kg_mm2": _mass(
                "同筒。实测偏差2.34e-16（I_zz）——本案例全部质量属性里的最大值，"
                "容差1e-15由它定。"
            ),
            "assembly_volume_mm3": _mass("三件互不重叠，体积可直接相加（分解是精确的，不是近似）。"),
            "assembly_mass_kg": _mass("同上。"),
            "assembly_centroid_mm": _zero(
                "轴对称+两片法兰关于中面镜像，质心**恰在原点**。零容差不是乐观："
                "`(−56m) + (+56m)`在IEEE-754下精确为0（同幅反号相加无舍入）。"
            ),
            "assembly_inertia_diag_kg_mm2": _mass(
                "平行轴定理合并：`I_xx = I_xx,筒 + 2(I_xx,兰 + m_兰·d²)`、"
                "`I_zz = I_zz,筒 + 2I_zz,兰`，d=56.0。实测偏差1.75e-16。"
                "**这一条同时验了`shift_inertia_kg_mm2`用在生成产物上是对的。**"
            ),
            "inertia_offdiag_max_kg_mm2": _zero(
                "圆柱的惯量张量在局部轴下是对角的，`geometry._diagonal`写的是字面0.0——"
                "所以是位判据不是数值判据。非零即说明轴向搞错了。"
            ),
        },
    }


def roller_oracle() -> dict[str, object]:
    roller = cylinder_closed_form(ROLLER_RADIUS_MM, ROLLER_WIDTH_MM)
    return {
        "id": "oracle:modelgen/roller_mass_properties",
        "inputs": {"call": ROLLER_CALL, "density_kg_m3": DENSITY_KG_M3},
        "expected": {
            "part_ids": ["face"],
            "radius_mm": ROLLER_RADIUS_MM,
            "half_width_mm": 0.5 * ROLLER_WIDTH_MM,
            "volume_mm3": roller["volume"],
            "mass_kg": roller["mass"],
            "inertia_diag_kg_mm2": roller["diag"],
            "inertia_offdiag_max_kg_mm2": 0.0,
        },
        "tolerances": {
            "part_ids": _zero("导轮是单件（WDS `CylinderSurface`就是一个形）。"),
            "radius_mm": _zero(
                "`0.375×128`在二进制下精确，逐位判据。它顺带钉住了"
                "「`face_width_ratio`是**全**宽不是半宽」这条易错的口径。"
            ),
            "half_width_mm": _zero(
                "`0.5×0.125×128 = 8.0`精确。全宽16.0与半宽8.0写在两处，"
                "把全/半宽这条最容易差2倍的口径钉死。"
            ),
            "volume_mm3": _mass("`V = πR²W`。实测偏差恰为0。"),
            "mass_kg": _mass("`m = ρV/1e9`。"),
            "inertia_diag_kg_mm2": _mass("圆柱三式同带盘。实测偏差2.09e-16（横向）。"),
            "inertia_offdiag_max_kg_mm2": _zero("同带盘：局部轴下对角，位判据。"),
        },
    }


def former_oracle() -> dict[str, object]:
    volumes = [
        capsule_volume_mm3(radius, height)
        for radius, height in zip(LINK_RADII_MM, LINK_HEIGHTS_MM, strict=True)
    ]
    masses = [DENSITY_KG_M3 * volume / 1.0e9 for volume in volumes]
    return {
        "id": "oracle:modelgen/former_mass_properties",
        "inputs": {"call": FORMER_CALL, "density_kg_m3": DENSITY_KG_M3},
        "expected": {
            "part_ids": ["link_0", "link_1"],
            "link_radii_mm": list(LINK_RADII_MM),
            "link_volumes_mm3": volumes,
            "link_masses_kg": masses,
            "link_centroids_mm": [list(point) for point in LINK_CENTROIDS_MM],
            "inertia_symmetric": True,
            "inertia_triangle_inequality_holds": True,
            "inertia_offdiag_max_kg_mm2": 0.0,
        },
        "tolerances": {
            "part_ids": _zero("n个骨架点产n−1件胶囊，件名带段序。"),
            "link_radii_mm": _zero(
                "锥度插值的**手推值**：段中点t=0.25与0.75，"
                "`(0.0625−0.03125t)×512`给出28.0与20.0，两步都无舍入，故逐位判据。"
                "这条是case2 `fo_h_el→fo_h_tip`锥度形制的落点。"
            ),
            "link_volumes_mm3": _mass("`V = πr²h + (4/3)πr³`。实测两段偏差均恰为0。"),
            "link_masses_kg": _mass("`m = ρV/1e9`。"),
            "link_centroids_mm": _zero(
                "胶囊质心在两端点中点。端点坐标由比值×L精确得到，中点再除2也精确，"
                "故逐位判据。它钉住的是「骨架点确实进了端点」——插值搞反时它先红。"
            ),
            "inertia_symmetric": _zero(
                "`I_ij = I_ji`。惯量张量必对称，与胶囊闭式无关的**普适**判据。"
            ),
            "inertia_triangle_inequality_holds": _zero(
                "主矩三角不等式`I_i + I_j ≥ I_k`。任何真实质量分布必满足，"
                "同样与胶囊闭式无关。**本案例不独立验证胶囊惯量张量的数值**"
                "（见案例页第四节），这两条普适律是那个空缺处唯一诚实的判据。"
            ),
            "inertia_offdiag_max_kg_mm2": _zero(
                "两段的轴向分别是`(1,0,0)`与`(0,1,0)`，`_axisymmetric`的"
                "`Δ·n_i·n_j`含因子0.0，精确为零。位判据。"
            ),
        },
    }


def main() -> int:
    oracles = [
        determinism_oracle(),
        homogeneity_oracle(),
        spool_oracle(),
        roller_oracle(),
        former_oracle(),
    ]
    document = {
        "facet": "engine_oracle_manifest", "facet_version": "0.1",
        "case_id": "case/generator_determinism", "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID, "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/generator_determinism/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles, "arrays": {}, "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
