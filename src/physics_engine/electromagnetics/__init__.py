"""电磁域——第三个物理域（力学、光学之后），域环由决策0041预登记。

## 边界（同spec/15，与光学域逐条对齐）

* 本域**不import力学域、不import光学域**，反向亦然。这条由
  `tests/governance/test_domain_isolation.py`的AST静态扫守着，不靠自觉；
* 域间往来只经基座或将来显式的`couplings/`；
* 本子包的公开名**只进本文件的`__all__`，不进`physics_engine/__init__.py`**
  （基座不依赖物理域）。

## 长度制

电磁量天然米制（`μ0`是H/m、片电流密度是A/m），与力学的mm制不混一条材料记录。
**安培不在轴2的单位后缀基础集里**，`Jc`/`Ic`因此今天进不了材料记录——
实测与裁决请求见`docs/decisions/0047_Norris薄带解析基准_20260805.md`第四节。

## 本子包今天有什么

`superconductor`：Norris 1970薄带临界态的片电流分布与两条交流损耗闭式
（案例`cases/norris_thin_strip`）。**它验的是公式不是引擎**——
今天`shapes`里没有"带材"、`state`里没有电流自由度。
"""

from __future__ import annotations

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

__all__ = [
    "LOSS_SERIES_LIMIT",
    "NORRIS_LOSS_RELATIVE_ACCURACY",
    "SHEET_CURRENT_RELATIVE_ACCURACY",
    "SUPERCONDUCTOR_DIMENSIONLESS_RESULTS",
    "SUPERCONDUCTOR_EXTRA_UNIT_SUFFIXES",
    "SUPERCONDUCTOR_LENGTH_UNIT",
    "SuperconductorError",
    "current_ratio",
    "flux_free_half_width_m",
    "norris_ellipse_loss_j_per_m_per_cycle",
    "norris_ellipse_normalised_loss",
    "norris_strip_loss_j_per_m_per_cycle",
    "norris_strip_normalised_loss",
    "sheet_critical_current_a_per_m",
    "sheet_current_density_a_per_m",
    "strip_critical_current_a",
]
