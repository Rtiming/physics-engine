"""光学域第一块——标量衍射闭式解与FTS仪器线型（spec/15登记的第二个物理域）。

用户2026-08-05裁决光学域进引擎（决策0031）。本子包是它的第一块代码。

## 边界（spec/15，本块与它同批落地）

* 光学域**不import力学域**，力学域也不import光学域。这条不是自觉，
  是`tests/governance/test_domain_isolation.py`用AST静态扫import守着的门；
* 域间往来只经**基座**（`materials`/`canonical`/`identity`这些）或将来显式的
  `couplings/`。本子包今天只向基座伸手一次：`parameters.py`读材料记录；
* 本子包的公开名**只进本文件的`__all__`，不进`physics_engine/__init__.py`**。
  包门面是基座，基座不依赖物理域（spec/01"内核不依赖任何上层"）——
  这条同样由域隔离门守着，不靠记性。

## 数值形态（0016甲案）

**纯Python，零运行时依赖，本块不做NumPy加速档**，理由不是"来不及"：
本块全部公开操作都是**每次调用几十个浮点运算的标量闭式**，没有数组可向量化。
research/06已测出本仓的标度指数约1.05（线性、没有可摊薄的固定成本），
在这种形状上套加速档只会多出一层派发。加速档的正当位置是二维场传播
（下一块），那里才有真正的数组工作量。这是spec/13第一节义务1
"优化先profile"的一个前置形态：**没有工作量的地方不谈加速**。

## 明确不做的（负空间声明，Drake形制）

* **二维FFT场传播**（角谱法、菲涅耳传播、可变孔径掩模）——要数组与FFT，
  属可选加速档，是下一块；
* **带切趾的ILS闭式**——Norton-Beer窗的傅里叶变换要球贝塞尔`j_i(u)/u^i`
  与它的小自变量级数（直接用sin/cos写会在`u -> 0`处相消到失去全部有效位）。
  本块只给窗本身与它的通量代价；
* **自切趾**（有限接收立体角把条纹对比度吃掉的那一项）、**相位误差**、
  **采样与混叠**——FTS一侧有成熟实现，搬迁按spec/12第七节的切换门走，
  不在本块预支；
* **偏振、相干、色散、非线性**——一个都没有；
* **不做第二套FTS**。物理正本在fts-digital-twin，本块是引擎侧的规范面与
  可对拍的闭式参照。
"""

from __future__ import annotations

from physics_engine.optics.bessel import (
    J1_ABSOLUTE_ACCURACY,
    J1_TESTED_ARGUMENT_MAX,
    SERIES_LIMIT,
    bessel_j1,
)
from physics_engine.optics.diffraction import (
    AIRY_FIRST_MINIMUM_DIAMETER_FACTOR,
    AIRY_FIRST_ZERO_TRUNCATION,
    AIRY_FIRST_ZERO_X,
    RADIANS_PER_CYCLE,
    airy_amplitude,
    airy_argument,
    airy_first_minimum_half_angle_rad,
    airy_intensity,
    angular_wavenumber_rad_per_m,
    spatial_frequency_per_m,
    spectroscopic_wavenumber_per_m,
)
from physics_engine.optics.errors import OpticsError
from physics_engine.optics.fts import (
    DOUBLE_SIDED_OPD_FACTOR,
    NORTON_BEER_COEFFICIENTS,
    NORTON_BEER_STRENGTHS,
    NORTON_BEER_UNIT_SUM_TOLERANCE,
    UNAPODISED_FWHM_IN_SINC_UNITS,
    UNAPODISED_FWHM_TEXTBOOK,
    normalised_sinc,
    norton_beer_coefficients,
    norton_beer_throughput,
    norton_beer_window,
    unapodised_first_zero_per_m,
    unapodised_fwhm_per_m,
    unapodised_line_shape,
)
from physics_engine.optics.parameters import (
    OPTICS_DOMAIN,
    OPTICS_LENGTH_UNIT,
    optics_evidence_grade,
    optics_parameters,
    require_optics_parameter,
)

__all__ = [
    "AIRY_FIRST_MINIMUM_DIAMETER_FACTOR",
    "AIRY_FIRST_ZERO_TRUNCATION",
    "AIRY_FIRST_ZERO_X",
    "DOUBLE_SIDED_OPD_FACTOR",
    "J1_ABSOLUTE_ACCURACY",
    "J1_TESTED_ARGUMENT_MAX",
    "NORTON_BEER_COEFFICIENTS",
    "NORTON_BEER_STRENGTHS",
    "NORTON_BEER_UNIT_SUM_TOLERANCE",
    "OPTICS_DOMAIN",
    "OPTICS_LENGTH_UNIT",
    "OpticsError",
    "RADIANS_PER_CYCLE",
    "SERIES_LIMIT",
    "UNAPODISED_FWHM_IN_SINC_UNITS",
    "UNAPODISED_FWHM_TEXTBOOK",
    "airy_amplitude",
    "airy_argument",
    "airy_first_minimum_half_angle_rad",
    "airy_intensity",
    "angular_wavenumber_rad_per_m",
    "bessel_j1",
    "normalised_sinc",
    "norton_beer_coefficients",
    "norton_beer_throughput",
    "norton_beer_window",
    "optics_evidence_grade",
    "optics_parameters",
    "require_optics_parameter",
    "spatial_frequency_per_m",
    "spectroscopic_wavenumber_per_m",
    "unapodised_first_zero_per_m",
    "unapodised_fwhm_per_m",
    "unapodised_line_shape",
]
