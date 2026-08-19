#!/usr/bin/env python3
"""窄相的判据——**全部闭式，且一行都不import`physics_engine.collision`**。

本脚本解的是同一道题的手推答案，conformance测试再拿内核去撞它。
**实测数不作金标**（spec/08规则1）：内核的数拿来撞这里的数，不是反过来。

## 一、五族判据各自的闭式

### 1. 球-球

    d = |c₂ − c₁| − (r₁ + r₂)

初等。**分离支与穿透支是同一条式子**——这正是本仓既有窄相"一个数两用"的
形制，也是EPA那条路给不了的（它只答穿透那一半，见决策0090第2.1节）。

### 2. 球-半空间

    d = (c − p)·n − r

仿射，处处精确。落地那条路走的是**支撑函数**（``min_q q·d = −R``），
与本式是两串不同的运算——所以判据不写零容差（第三节）。

### 3. 盒-盒，轴对齐

逐轴间隙``g_i = |Δc_i| − h_i^A − h_i^B``：

    有一轴分开： d = sqrt(Σ max(g_i, 0)²) − f_A − f_B
    全轴重叠：   d = max_i g_i − f_A − f_B

第二支的``max_i g_i``就是**最小重叠的负值**，也就是轴对齐盒对的**最小平移
穿透深度**——EPA在这一对上给的是同一个数。**本案例因此顺带回答了
"不做EPA会不会少一个量"：在这一对上不会。**

圆角那两项在末尾各减一次：``K ⊕ B_f``对凸``K``的有符号距离恰好是``K``的减去``f``。

### 4. 球-圆柱（无法兰）／球-旋转盒

圆柱：径向``dr = ρ − R``、轴向``dz = |z| − w``，

    体外： sqrt(max(dr,0)² + max(dz,0)²)
    体内： max(dr, dz)

旋转盒：把球心转进盒的局部系再套盒的SDF。
**本案例特意放了一个"看着在外面其实在里面"的构型**（第三节`rotated_box_inside`）——
点``(14, 0, 0)``对一个转过45°的半宽10的盒，沿轴到顶点是14.142，
所以它在**体内**，最近面是侧面而不是顶点。这条是写这个案例时第一版真错过的地方。

### 5. 距离场的偏差主项

采样值直接当三次B样条系数的拟插值（`contact/field.py`模块docstring）：

    S(x) = φ(x) + (h²/6)·∇²φ(x) + O(h⁴)

于是**偏差主项是可计算的**，只要``∇²φ``有闭式：

* **半空间**``φ = z``：``∇²φ ≡ 0`` ⟹ 偏差**恒为零，与``h``无关**；
* **球（凸）**``φ = ρ − R``：``∇²φ = 2/ρ`` ⟹ 偏差``+h²/(3ρ)``，**正的＝偏松**；
* **圆柱孔（凹、非凸）**``φ = R − ρ``：``φ_ρ = −1``、``φ_ρρ = 0`` ⟹
  ``∇²φ = −1/ρ`` ⟹ 偏差``−h²/(6ρ)``，**负的＝偏紧**。

**第三条是本案例最值钱的一行**：0074第二节第4条只说了"系统性偏保守或偏松"，
0085第三节把凸那一侧钉到"偏松"，**这里把凹那一侧钉到"偏紧"**——
两侧各有一条闭式，谁也不用猜。

## 二、非凸那一条为什么用环面

环面``φ = sqrt((sqrt(x²+y²) − R)² + z²) − r``是**精确的有符号距离**且**非凸**。
它在这里的作用不是量偏差（那两条由球与孔各自的闭式管），
而是量**场值误差随分辨率的阶**：非凸不给SDF带来任何额外代价，
这句话（0074第二节第2条）在本案例有一个可测的出口——比值落在4上。

走凸体窄相的话这个形状要先凸分解，而那条路已被0073第七节第1条裁掉。

## 三、为什么几乎没有零容差

**只有一条判据是零容差**（半空间场的偏差估计恒为0，那是``∇²φ ≡ 0``的直接后果）。
其余全是``abs 1e-12 mm``或``rel 5e-3``，理由是两条**本案例自己量出来的**事实：

1. `segment_segment_distance_mm`开方用``x ** 0.5``而不是`math.sqrt`。
   ``**``走libm的``pow``，**IEEE不要求它正确舍入**——本机实测20万个正数里
   277个两者差1 ulp；
2. **CPython 3.12起`sum()`对float走Neumaier补偿求和**，与手写的``a + b + c``
   不逐位相同——本机实测20万组三维点积里44972组不同（22.5%）。

两条都**没有被"修"**：改前者会破掉"既有球/胶囊窄相逐位不变"那条判据。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.oracles import file_sha256, write_manifest  # noqa: E402

ALGORITHM_ID = "algorithm:oracle/narrow_phase_signed_distance"
ALGORITHM_VERSION = "1.0.0"

#: 解析构型。**每个数都出现在案例页第一节**，改这里就要改那里。
SPHERE_A_RADIUS_MM = 3.0
SPHERE_B_RADIUS_MM = 4.0
SPHERE_SEPARATED_GAP_MM = 12.0
SPHERE_PENETRATING_GAP_MM = 5.0

PLANE_HEIGHT_MM = 1.0
PLANE_PROBE_RADIUS_MM = 2.0
PLANE_SEPARATED_Z_MM = 4.0
PLANE_PENETRATING_Z_MM = 2.0

BOX_A_HALF_MM = (10.0, 20.0, 30.0)
BOX_B_HALF_MM = (5.0, 5.0, 5.0)
BOX_SEPARATED_X_MM = 20.0
BOX_PENETRATING_X_MM = 12.0

CYLINDER_RADIUS_MM = 9.0
CYLINDER_HALF_WIDTH_MM = 4.0
CYLINDER_PROBE_RADIUS_MM = 1.0

ROTATED_BOX_HALF_MM = 10.0
ROTATED_BOX_PROBE_RADIUS_MM = 2.0
ROTATED_BOX_OUTSIDE_X_MM = 18.0
ROTATED_BOX_INSIDE_X_MM = 14.0

#: 场那一族。三档`h`每档减半——收敛比落在4上时那个4是货真价实的二阶。
PLANE_FIELD_SPACINGS_MM = (2.0, 1.0, 0.5)
#: 场那一族的探针高度**与解析那一族分开**：窄带只烘``|φ| ≤ band``那一层，
#: 而查询点的4×4×4支撑还要再往外伸``2h``。取2.5（band 4.0）留得下那两圈。
PLANE_FIELD_PROBE_Z_MM = 2.5
SPHERE_FIELD_SPACINGS_MM = (1.0, 0.5, 0.25)
BORE_FIELD_SPACINGS_MM = (0.5, 0.25)
TORUS_FIELD_SPACINGS_MM = (0.8, 0.4, 0.2)

SPHERE_FIELD_RADIUS_MM = 10.0
SPHERE_FIELD_PROBE_RHO_MM = 11.5
BORE_RADIUS_MM = 12.0
BORE_PROBE_RHO_MM = 10.5
TORUS_MAJOR_MM = 12.0
TORUS_MINOR_MM = 3.5


def _key(spacing_mm: float) -> str:
    """``0.25`` → ``h_0p25``。清单的键不许带点号（它要能当标识符读）。"""

    return "h_" + f"{spacing_mm:g}".replace(".", "p")


def sphere_sphere_mm(gap_mm: float) -> float:
    return gap_mm - (SPHERE_A_RADIUS_MM + SPHERE_B_RADIUS_MM)


def sphere_half_space_mm(centre_z_mm: float) -> float:
    return centre_z_mm - PLANE_HEIGHT_MM - PLANE_PROBE_RADIUS_MM


def axis_aligned_boxes_mm(offset_x_mm: float) -> float:
    gaps = [
        (offset_x_mm if axis == 0 else 0.0) - BOX_A_HALF_MM[axis] - BOX_B_HALF_MM[axis]
        for axis in range(3)
    ]
    if max(gaps) > 0.0:
        positive = [gap for gap in gaps if gap > 0.0]
        return math.sqrt(sum(gap * gap for gap in positive))
    return max(gaps)


def sphere_cylinder_mm(rho_mm: float, z_mm: float) -> float:
    radial = rho_mm - CYLINDER_RADIUS_MM
    axial = abs(z_mm) - CYLINDER_HALF_WIDTH_MM
    if radial > 0.0 or axial > 0.0:
        distance = math.sqrt(max(radial, 0.0) ** 2 + max(axial, 0.0) ** 2)
    else:
        distance = max(radial, axial)
    return distance - CYLINDER_PROBE_RADIUS_MM


def sphere_rotated_box_mm(centre_x_mm: float) -> float:
    """盒绕z转45°：球心转进局部系后两个横向分量都是``x/√2``。"""

    local = centre_x_mm / math.sqrt(2.0)
    q = local - ROTATED_BOX_HALF_MM
    if q > 0.0:
        distance = math.sqrt(2.0 * q * q)  # 两个横轴同时在外 ⟹ 最近的是顶点
    else:
        distance = q  # 体内：最近面是侧面，不是顶点
    return distance - ROTATED_BOX_PROBE_RADIUS_MM


def convex_sphere_bias_mm(spacing_mm: float) -> float:
    """``+h²/(3ρ)``——凸障碍，**偏松**。"""

    return spacing_mm * spacing_mm / (3.0 * SPHERE_FIELD_PROBE_RHO_MM)


def concave_bore_bias_mm(spacing_mm: float) -> float:
    """``−h²/(6ρ)``——凹面，**偏紧**。"""

    return -spacing_mm * spacing_mm / (6.0 * BORE_PROBE_RHO_MM)


def main() -> int:
    rounding_reason = (
        "**不写零容差**，理由是本案例自己量出来的两条：① "
        "`segment_segment_distance_mm`开方用``x ** 0.5``而不是`math.sqrt`，"
        "而``**``走libm的``pow``、IEEE不要求它正确舍入（本机20万个正数里277个差1 ulp）；"
        "② CPython 3.12起`sum()`对float走Neumaier补偿求和，与手写的``a + b + c``"
        "不逐位相同（本机20万组三维点积里44972组不同）。"
        "坐标量级50 mm上两条合起来给到约1.4e-14 mm，1e-12留约两个量级。"
        "**两条都没有被'修'**：改前者会破掉'既有球/胶囊窄相逐位不变'那条判据。"
    )
    leading_term_reason = (
        "闭式给的是**主项**``(h²/6)·∇²φ``，实测值还带一个``O(h⁴)``的余项。"
        "实测相对偏差随``h``二阶缩小（球那一族1.889e-3 / 4.709e-4 / 1.180e-4），"
        "5e-3覆盖最粗那一档并留一档。**不按绝对判**：绝对容差会让粗档松、细档紧，"
        "量不出'主项抓对了没有'。"
    )

    oracles: list[dict] = []

    analytic_expected = {
        "sphere_sphere_separated_mm": sphere_sphere_mm(SPHERE_SEPARATED_GAP_MM),
        "sphere_sphere_penetrating_mm": sphere_sphere_mm(SPHERE_PENETRATING_GAP_MM),
        "sphere_half_space_separated_mm": sphere_half_space_mm(PLANE_SEPARATED_Z_MM),
        "sphere_half_space_penetrating_mm": sphere_half_space_mm(PLANE_PENETRATING_Z_MM),
        "box_box_separated_mm": axis_aligned_boxes_mm(BOX_SEPARATED_X_MM),
        "box_box_penetrating_mm": axis_aligned_boxes_mm(BOX_PENETRATING_X_MM),
        "sphere_cylinder_side_mm": sphere_cylinder_mm(12.0, 0.0),
        "sphere_cylinder_end_mm": sphere_cylinder_mm(0.0, 6.0),
        "sphere_cylinder_inside_mm": sphere_cylinder_mm(0.0, 0.0),
        "sphere_rotated_box_outside_mm": sphere_rotated_box_mm(ROTATED_BOX_OUTSIDE_X_MM),
        "sphere_rotated_box_inside_mm": sphere_rotated_box_mm(ROTATED_BOX_INSIDE_X_MM),
    }
    oracles.append(
        {
            "id": "oracle:narrow_phase_signed_distance/analytic_pairs",
            "inputs": {
                "kind": "analytic_pairs",
                "note": (
                    "五族解析构型，每族分离支与穿透支各一条。"
                    "**`sphere_rotated_box_inside_mm`是本案例最要紧的一格**："
                    "点(14,0,0)对转过45°的半宽10的盒在**体内**（沿轴到顶点是14.142），"
                    "最近面是侧面不是顶点——把'沿轴到顶点'当成'到表面'是旋转凸体上"
                    "最容易犯的一个错，写这个案例时第一版真错过。"
                ),
                "sphere_radii_mm": [SPHERE_A_RADIUS_MM, SPHERE_B_RADIUS_MM],
                "box_a_half_extents_mm": list(BOX_A_HALF_MM),
                "box_b_half_extents_mm": list(BOX_B_HALF_MM),
                "cylinder_radius_mm": CYLINDER_RADIUS_MM,
                "cylinder_half_width_mm": CYLINDER_HALF_WIDTH_MM,
                "rotated_box_half_extent_mm": ROTATED_BOX_HALF_MM,
            },
            "expected": analytic_expected,
            "tolerances": {
                name: {"rel": 0.0, "abs": 1.0e-12, "reason": rounding_reason}
                for name in analytic_expected
            },
        }
    )

    plane_expected: dict[str, float] = {}
    plane_tolerances: dict[str, dict] = {}
    for spacing in PLANE_FIELD_SPACINGS_MM:
        plane_expected[f"separation_{_key(spacing)}_mm"] = PLANE_FIELD_PROBE_Z_MM
        plane_expected[f"bias_estimate_{_key(spacing)}_mm"] = 0.0
        plane_tolerances[f"separation_{_key(spacing)}_mm"] = {
            "rel": 0.0,
            "abs": 1.0e-13,
            "reason": (
                "**仿射函数被三次B样条精确重构**（一阶矩为零），于是场与闭式解的是"
                "同一个方程，差别只有求和次序带来的舍入。实测三档``h``的最差偏差是"
                "4.441e-16 mm（2 ulp）。1e-13留约两个量级——**但不写零**："
                "两条路的求和次序不同，逐位相等是没验过的、也不该被承诺的性质。"
            ),
        }
        plane_tolerances[f"bias_estimate_{_key(spacing)}_mm"] = {
            "rel": 0.0,
            "abs": 1.0e-15,
            "reason": (
                "**本案例唯一一条接近零容差的判据**，理由是代数事实不是精度自信："
                "``φ``仿射 ⟹ ``∇²φ ≡ 0`` ⟹ ``(h²/6)·tr(∇²φ)``恒为零。"
                "实测三档分别是7.401e-17 / 1.850e-17 / **0.0**——"
                "不写死0是因为Hessian那三条对角线各自是64项加权和，"
                "它们抵消到的是舍入量级而不是恒等的0。"
            ),
        }
    oracles.append(
        {
            "id": "oracle:narrow_phase_signed_distance/plane_field",
            "inputs": {
                "kind": "plane_field",
                "note": "仿射场：与``h``无关的精确重构，偏差估计恒为0。",
                "spacings_mm": list(PLANE_FIELD_SPACINGS_MM),
                "probe_z_mm": PLANE_FIELD_PROBE_Z_MM,
            },
            "expected": plane_expected,
            "tolerances": plane_tolerances,
        }
    )

    convex_expected = {
        f"bias_estimate_{_key(spacing)}_mm": convex_sphere_bias_mm(spacing)
        for spacing in SPHERE_FIELD_SPACINGS_MM
    }
    oracles.append(
        {
            "id": "oracle:narrow_phase_signed_distance/convex_sphere_field_bias",
            "inputs": {
                "kind": "convex_sphere_field_bias",
                "note": (
                    "凸障碍：``∇²(ρ − R) = 2/ρ`` ⟹ 偏差主项``+h²/(3ρ)``，"
                    "**正的＝场报出来的距离比真值大＝偏松**，与0085第三节同向。"
                ),
                "obstacle_radius_mm": SPHERE_FIELD_RADIUS_MM,
                "probe_rho_mm": SPHERE_FIELD_PROBE_RHO_MM,
                "spacings_mm": list(SPHERE_FIELD_SPACINGS_MM),
            },
            "expected": convex_expected,
            "tolerances": {
                name: {"rel": 5.0e-3, "abs": 0.0, "reason": leading_term_reason}
                for name in convex_expected
            },
        }
    )

    concave_expected = {
        f"bias_estimate_{_key(spacing)}_mm": concave_bore_bias_mm(spacing)
        for spacing in BORE_FIELD_SPACINGS_MM
    }
    oracles.append(
        {
            "id": "oracle:narrow_phase_signed_distance/concave_bore_field_bias",
            "inputs": {
                "kind": "concave_bore_field_bias",
                "note": (
                    "**本案例最值钱的一格**：圆柱孔（实体是圆柱的补集，非凸）"
                    "``φ = R − ρ`` ⟹ ``∇²φ = −1/ρ`` ⟹ 偏差主项``−h²/(6ρ)``，"
                    "**负的＝场报出来的距离比真值小＝偏紧**。"
                    "0074第二节第4条只说了'系统性偏保守或偏松'、0085第三节把凸那一侧"
                    "钉到'偏松'，这一格把凹那一侧钉到'偏紧'。"
                ),
                "bore_radius_mm": BORE_RADIUS_MM,
                "probe_rho_mm": BORE_PROBE_RHO_MM,
                "spacings_mm": list(BORE_FIELD_SPACINGS_MM),
            },
            "expected": concave_expected,
            "tolerances": {
                name: {"rel": 5.0e-3, "abs": 0.0, "reason": leading_term_reason}
                for name in concave_expected
            },
        }
    )

    order_reason = (
        "**收敛比是区间不是恒等于4**（`cases/harmonic_oscillator`那条同源：4是渐近值）。"
        "1e-1相对 = ``[3.6, 4.4]``，比本仓其余二阶门的``[3.9, 4.1]``宽一档——"
        "**理由要说清楚**：这里量的是**四个探针上的max范数**，而max范数对"
        "'哪个点最差'敏感（0085第9.3节腿A记过同一形状：max比3.2411而mean比3.8299）。"
        "实测4.0459 / 4.0127，离窗口边界还有一档余量。"
    )
    oracles.append(
        {
            "id": "oracle:narrow_phase_signed_distance/nonconvex_torus_order",
            "inputs": {
                "kind": "nonconvex_torus_order",
                "note": (
                    "非凸环面：场值误差随``h``二阶。它量的不是偏差（那两格由球与孔管），"
                    "是'非凸不给SDF带来额外代价'（0074第二节第2条）的可测出口。"
                ),
                "major_radius_mm": TORUS_MAJOR_MM,
                "minor_radius_mm": TORUS_MINOR_MM,
                "spacings_mm": list(TORUS_FIELD_SPACINGS_MM),
            },
            "expected": {"order_ratio_coarse": 4.0, "order_ratio_fine": 4.0},
            "tolerances": {
                "order_ratio_coarse": {"rel": 1.0e-1, "abs": 0.0, "reason": order_reason},
                "order_ratio_fine": {"rel": 1.0e-1, "abs": 0.0, "reason": order_reason},
            },
        }
    )

    document = {
        "facet": "engine_oracle_manifest",
        "facet_version": "0.1",
        "case_id": "case/narrow_phase_signed_distance",
        "load_tier": "interactive",
        "generator": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "path_relative": "cases/narrow_phase_signed_distance/generate_oracle.py",
            "sha256": file_sha256(HERE / "generate_oracle.py"),
        },
        "oracles": oracles,
        "arrays": {},
        "regenerated_by": None,
    }
    written = write_manifest(HERE / "oracle.json", document, root=ROOT)
    print(f"wrote {len(oracles)} oracles, {len(written)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
