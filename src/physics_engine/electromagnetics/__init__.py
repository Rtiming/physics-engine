"""电磁域第一块——同轴圆环互感的Maxwell闭式解（spec/15登记的**第三个物理域**）。

决策0041第三节把`electromagnetics`预登记为第三个物理域（力学、光学之后），
决策0042是它的第一块代码。**这也是plans/04六个真实场景里第一个能落地的电磁能力**
（场景②"多线圈之间的干扰"的互感那一半）。

## 边界（spec/15，与`optics/`同形制）

* 电磁域**不import力学域也不import光学域**，反向同样。这条不是自觉，
  是`tests/governance/test_domain_isolation.py`用AST静态扫import守着的门；
* 本子包今天**一次都没有向基座伸手**——真空里两条回路的互感只要几何与μ₀，
  不读材料记录。这是如实状态不是设计目标：真需要材料时按`optics/parameters.py`
  的形制加一个读者模块，**不预支**（0001第二前提）；
* 本子包的公开名**只进本文件的`__all__`，不进`physics_engine/__init__.py`**。
  包门面是基座，基座不依赖物理域，同样由域隔离门守着。

## 为什么第一块是互感而不是场求解

research/08给电磁域列了六道门槛（自由度住在棱上、体网格、复数/奇异求解、
能量项协议之外的装配面、单位后缀、复数状态），**互感一道都不撞**：

| research/08的门槛 | 互感这条路 |
|---|---|
| 门槛1 自由度挂在哪个拓扑实体上 | **没有自由度**——闭式，不进状态 |
| 门槛2 体网格与关联表 | **不要网格**（且见下面那条订正） |
| 门槛3 复数/不定/奇异求解 | **不解方程** |
| 门槛4 能量项协议之外的装配面 | **不装配**——不是`EnergyTerm` |
| 门槛5 七个电磁单位后缀 | **不写材料记录**，故未触发（下面第二条如实说明） |
| 门槛6 复数进`State` | **不进`State`** |

**一条对research/08的背景订正**：它默认"电磁 = FEM棱边元"，因此把体网格列为门槛2。
对互感/电容这一族**那个默认是错的**——工业主路（FastHenry/FasterCap）是
**积分方程**方法，只离散导体不离散空气。走这条路，门槛2整块绕得开。

## 撞上的那条边界：mm几何 vs 米制电磁

plans/03第四节D登记着"电磁会是第一个跑不通spec/14规则1的域"。本块**撞上了一半**：
引擎几何是mm（spec/11），而μ₀按定义是H/**m**，两者必须显式换算。
处理与门都在`units.py`，撞的过程写在决策0042。
**没撞上的那一半如实说**：本块不读任何材料记录，所以
"同一块铜的杨氏模量与电导率放不进同一条记录"那条边界**仍然开着**，
不是被本块关掉的。

## 数值形态（0016甲案）

纯Python、零运行时依赖、**本块不做NumPy加速档**，理由与0031第3.1节同：
全部公开操作都是每次调用几十个浮点运算的标量闭式，没有数组可向量化。
加速档的正当位置是细丝离散的一般位形互感（N×N条丝对），不是这里。

## 明确不做的（负空间声明汇总，逐条理由在各模块）

自感、一般位形（倾斜/偏心）、电容、磁介质、匝间几何、回路间的力与力矩、
时变与感应电动势——一个都不做。见`inductance.py`第五节。

## 长度制（决策0047实测）

电磁量天然米制（`μ0`是H/m、片电流密度是A/m），与力学的mm制不混一条材料记录。
**安培不在轴2的单位后缀基础集里**——`BASE_UNIT_SUFFIXES`有开尔文有瓦特没有安培——
`Jc`/`Ic`因此今天**进不了任何材料记录**（除非撒谎声明成无量纲）。
实测与裁决请求见决策0047第四节。这是**临时避开不是长期方案**。

## 本子包今天有什么

* `inductance`/`loops`/`elliptic`/`units`：同轴圆环互感的Maxwell闭式（决策0042）；
* `superconductor`：Norris 1970薄带临界态的片电流分布与两条交流损耗闭式（决策0047）。
  **它验的是公式不是引擎**——今天`shapes`里没有"带材"、`state`里没有电流自由度。
"""

from __future__ import annotations

from physics_engine.electromagnetics.elliptic import (
    AGM_ITERATION_BOUND,
    COMPLETE_E_RELATIVE_ACCURACY,
    COMPLETE_K_RELATIVE_ACCURACY,
    MAXWELL_BRACKET_RELATIVE_ACCURACY,
    MODULUS_MAX,
    agm_kernel,
    complete_elliptic_e,
    complete_elliptic_e_of_parameter,
    complete_elliptic_k,
    complete_elliptic_k_of_parameter,
    maxwell_mutual_bracket,
)
from physics_engine.electromagnetics.errors import ElectromagneticsError
from physics_engine.electromagnetics.inductance import (
    coaxial_modulus,
    coaxial_mutual_inductance_h,
    dipole_mutual_inductance_h,
    flux_linkage_wb,
    mutual_inductance_h,
)
from physics_engine.electromagnetics.loops import CircularLoop
from physics_engine.electromagnetics.superconductor import (
    LOSS_SERIES_LIMIT,
    NORRIS_LOSS_RELATIVE_ACCURACY,
    SHEET_CURRENT_RELATIVE_ACCURACY,
    SUPERCONDUCTOR_DIMENSIONLESS_RESULTS,
    SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES,
    SUPERCONDUCTOR_LENGTH_UNIT,
    SuperconductorError,
    current_ratio,
    flux_free_half_width_m,
    norris_ellipse_loss_j_per_m_per_cycle,
    norris_ellipse_normalised_loss,
    norris_strip_loss_j_per_m_per_cycle,
    norris_strip_normalised_loss,
    sheet_critical_current_a_per_m,
    sheet_current_density_a_per_m,
    strip_critical_current_a,
)
from physics_engine.electromagnetics.units import (
    EM_LENGTH_UNIT,
    GEOMETRY_LENGTH_UNIT,
    LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M,
    MILLIMETRES_PER_METRE,
    VACUUM_PERMEABILITY_EVIDENCE_GRADE,
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMEABILITY_RELATIVE_UNCERTAINTY,
    metres_from_millimetres,
    millimetres_from_metres,
    require_em_length_unit,
    vacuum_permeability_relative_deviation_from_legacy,
)

__all__ = [
    "AGM_ITERATION_BOUND",
    "COMPLETE_E_RELATIVE_ACCURACY",
    "COMPLETE_K_RELATIVE_ACCURACY",
    "CircularLoop",
    "EM_LENGTH_UNIT",
    "ElectromagneticsError",
    "GEOMETRY_LENGTH_UNIT",
    "LEGACY_EXACT_VACUUM_PERMEABILITY_H_PER_M",
    "LOSS_SERIES_LIMIT",
    "MAXWELL_BRACKET_RELATIVE_ACCURACY",
    "MILLIMETRES_PER_METRE",
    "MODULUS_MAX",
    "NORRIS_LOSS_RELATIVE_ACCURACY",
    "SHEET_CURRENT_RELATIVE_ACCURACY",
    "SUPERCONDUCTOR_DIMENSIONLESS_RESULTS",
    "SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES",
    "SUPERCONDUCTOR_LENGTH_UNIT",
    "SuperconductorError",
    "VACUUM_PERMEABILITY_EVIDENCE_GRADE",
    "VACUUM_PERMEABILITY_H_PER_M",
    "VACUUM_PERMEABILITY_RELATIVE_UNCERTAINTY",
    "agm_kernel",
    "coaxial_modulus",
    "coaxial_mutual_inductance_h",
    "complete_elliptic_e",
    "complete_elliptic_e_of_parameter",
    "complete_elliptic_k",
    "complete_elliptic_k_of_parameter",
    "current_ratio",
    "dipole_mutual_inductance_h",
    "flux_free_half_width_m",
    "flux_linkage_wb",
    "maxwell_mutual_bracket",
    "metres_from_millimetres",
    "millimetres_from_metres",
    "mutual_inductance_h",
    "norris_ellipse_loss_j_per_m_per_cycle",
    "norris_ellipse_normalised_loss",
    "norris_strip_loss_j_per_m_per_cycle",
    "norris_strip_normalised_loss",
    "require_em_length_unit",
    "sheet_critical_current_a_per_m",
    "sheet_current_density_a_per_m",
    "strip_critical_current_a",
    "vacuum_permeability_relative_deviation_from_legacy",
]
