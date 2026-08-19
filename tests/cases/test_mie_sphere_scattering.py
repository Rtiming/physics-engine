"""`case/mie_sphere_scattering`的conformance门（轴7规则3）。

**全仓第一条散射案例**——能力位S4.7的`why`原文写着plans/05对场景④的判词是
"散射零"。Mie解是Maxwell方程组对均匀球的**严格解**，不是任何一种近似；
数值上唯一的近似是级数截断，而那一条在`optics/mie.py`里显式声明并失败关闭。

四条oracle里**金标的来路各不相同**，这是本案例分辨力的来源：

* 前两条来自**50位十进制的独立实现**（上升级数 + 直接相除的对数导数 +
  自己算的``sin``/``cos``），与被验的`float`向下递推没有任何共用代码；
* 第三条一半来自50位参照、一半来自**静电偶极子**那条完全不同的推导；
* 第四条的金标是**解析的标度律**``2^(-2/3)``，不依赖任何实现。

判据数全部来自清单；本文件不复述任何公式（轴7规则4）。
"""

from __future__ import annotations

from pathlib import Path

from physics_engine.optics.field import complex_to_components
from physics_engine.optics.mie import (
    mie_coefficients,
    mie_efficiencies,
    mie_unitarity_residual,
    rayleigh_scattering_efficiency,
)
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/mie_sphere_scattering/oracle.json", root=ROOT)
SERIES = MANIFEST.oracle("oracle:mie/lossless_series_against_fifty_digits")
BALANCE = MANIFEST.oracle("oracle:mie/unitarity_and_energy_balance")
RAYLEIGH = MANIFEST.oracle("oracle:mie/rayleigh_small_sphere_limit")
PARADOX = MANIFEST.oracle("oracle:mie/extinction_paradox")


def _lossless_index(setup):
    return complex(setup["refractive_index_real"], setup.get("refractive_index_imaginary", 0.0))


def test_the_coefficients_and_efficiencies_match_a_fifty_digit_independent_series():
    """逐阶``a_n``、``b_n``与两条效率，对50位十进制的独立实现。

    两侧的算法互不相干：被验侧走Miller向下递推与连分式对数导数，
    金标侧走上升级数与两阶球贝塞尔直接相除，连``sin``/``cos``都不共用
    （金标那边是自己的泰勒级数，不调libm）。
    """

    setup = SERIES.inputs
    index = _lossless_index(setup)
    coefficients = mie_coefficients(
        size_parameter=setup["size_parameter"],
        refractive_index=index,
        order_count=setup["order_count"],
    )
    efficiencies = mie_efficiencies(
        size_parameter=setup["size_parameter"],
        refractive_index=index,
        order_count=setup["order_count"],
    )
    SERIES.check_all(
        {
            "coefficient_a_components": [
                complex_to_components(value) for value in coefficients.a
            ],
            "coefficient_b_components": [
                complex_to_components(value) for value in coefficients.b
            ],
            "extinction_efficiency": efficiencies.extinction,
            "scattering_efficiency": efficiencies.scattering,
        }
    )


def test_a_lossless_sphere_conserves_energy_and_every_order_sits_on_the_unitarity_circle():
    """``Q_ext - Q_sca = 0``与**逐阶**``| |a_n|^2 - Re(a_n) | = 0``。

    两条式子（取实部 对 取模平方）给同一个数，才有第一行；
    第二行逐阶看，抵消不掉——它比第一行更硬。
    """

    setup = BALANCE.inputs
    index = complex(setup["refractive_index_real"], 0.0)
    efficiencies = mie_efficiencies(
        size_parameter=setup["size_parameter"],
        refractive_index=index,
        order_count=setup["order_count"],
    )
    coefficients = mie_coefficients(
        size_parameter=setup["size_parameter"],
        refractive_index=index,
        order_count=setup["order_count"],
    )
    BALANCE.check_all(
        {
            "extinction_minus_scattering": efficiencies.extinction - efficiencies.scattering,
            "unitarity_max_residual": mie_unitarity_residual(coefficients),
        }
    )


def test_the_small_sphere_limit_lands_on_the_electrostatic_dipole_closed_form():
    """小球极限：严格级数、瑞利闭式、以及**两者之差**三样一起冻。

    第三行（相对偏差）才是这条门的分辨力所在：只冻前两行的话，
    "级数收敛到瑞利"与"级数恰好在这三个点上对得上"分不开。
    冻住偏差本身等于冻住**这条极限是怎么收敛的**——三个数逐档掉到四分之一，
    因为`x`减半而修正是``O(x^2)``。
    """

    setup = RAYLEIGH.inputs
    index = complex(setup["refractive_index_real"], 0.0)
    exact = [
        mie_efficiencies(size_parameter=size, refractive_index=index).scattering
        for size in setup["size_parameters"]
    ]
    closed = [
        rayleigh_scattering_efficiency(size_parameter=size, refractive_index=index)
        for size in setup["size_parameters"]
    ]
    RAYLEIGH.check_all(
        {
            "scattering_efficiency": exact,
            "rayleigh_closed_form_efficiency": closed,
            "relative_departure": [
                value / reference - 1.0 for value, reference in zip(exact, closed, strict=True)
            ],
        }
    )


def test_the_extinction_efficiency_approaches_two_along_the_analytic_scaling_law():
    """大球极限：``Q_ext -> 2``，**而且按``x^(-2/3)``趋近**。

    金标是解析的``2^(-2/3)``，不依赖任何实现。
    "接近2"是软判据（任何缓慢下降的曲线都满足）；本条判的是**指数**。
    """

    setup = PARADOX.inputs
    index = complex(setup["refractive_index_real"], setup["refractive_index_imaginary"])
    gaps = [
        abs(mie_efficiencies(size_parameter=size, refractive_index=index).extinction - 2.0)
        for size in setup["size_parameters"]
    ]
    PARADOX.check_all(
        {
            "extinction_gap_ratio": [
                later / earlier for earlier, later in zip(gaps[:-1], gaps[1:], strict=True)
            ],
            "extinction_at_largest_size": mie_efficiencies(
                size_parameter=setup["size_parameters"][-1], refractive_index=index
            ).extinction,
        }
    )


def test_the_gap_really_shrinks_monotonically_which_the_ratios_alone_do_not_say():
    """一条不靠清单的结构判据：``|Q_ext - 2|``必须**逐档单调变小**。

    清单那条判的是相邻比值落在``2^(-2/3)``附近——**一条整体上升
    但比值恰好也是0.63的曲线在那条判据上照样绿**（比值是正的，
    它不管方向）。本条正面断方向。
    """

    setup = PARADOX.inputs
    index = complex(setup["refractive_index_real"], setup["refractive_index_imaginary"])
    gaps = [
        abs(mie_efficiencies(size_parameter=size, refractive_index=index).extinction - 2.0)
        for size in setup["size_parameters"]
    ]
    for earlier, later in zip(gaps[:-1], gaps[1:], strict=True):
        assert later < earlier, f"|Qext-2|没有逐档变小：{gaps!r}"
