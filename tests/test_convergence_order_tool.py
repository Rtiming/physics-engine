"""`tools/convergence_order.py`的门——**判据本身要先被验**（决策0087丙4）。

plans/16第四节GAP第1条要的是"三条案例的步长收敛阶"。补那件事之前先补这件：
**一个阶估计器如果自己算错了，它给出的每一条"收敛阶"都是假的，而它们看起来
和真的一模一样。** 本仓`tests/governance/`的样板纪律就是这一条。

本文件判五类：

1. **对一个阶精确已知的合成序列**``u(h) = u* + C·h^p``，两条路径都必须还原``p``；
2. **对本仓已有的解析问题**（`cases/harmonic_oscillator`）：形式阶1／1／2三条，
   两条路径都要对上——**一个"永远返回形式阶"的实现在单条问题上分辨不出来**；
3. **细化比不是2时仍然对**——把``log₂``写死的实现在等比减半的阶梯上一样绿；
4. **差落到舍入地板以下时不报阶**——报一个假的阶比不报更坏；
5. 两条失败关闭：末时刻走不到整数步、最粗那档越过声明的步长上限。
"""

from __future__ import annotations

import math

import pytest

from tools.convergence_order import (
    ConvergenceOrderError,
    OrderEstimate,
    _ladder,
    assert_horizon_is_an_integer_number_of_steps,
    order_against_truth,
    order_by_richardson,
    run_problem,
    verify_estimator,
)


def _synthetic(truth: float, coefficient: float, order: float):
    """``u(h) = u* + C·h^p``——阶**精确**是`p`的一条序列。"""

    return lambda dt_s: truth + coefficient * dt_s**order


@pytest.mark.parametrize("order", [1.0, 2.0, 3.0, 4.0])
def test_both_paths_recover_an_exactly_known_order(order: float) -> None:
    """合成序列上两条路径都必须还原那个阶。

    **两条都判**：一个把Richardson的分子分母写反的实现在"对真值"那条上全对，
    反之亦然。
    """

    truth, coefficient = 7.25, 3.0
    #: 四档而不是六档：``C·h⁴``在第六档已经小到``truth + C·h⁴ − truth``
    #: 这个减法要吃掉五位有效数字——**合成序列自己就撞上舍入地板了**，
    #: 而那是被判的对象而不是这条门要判的东西。容差因此取1e-6不取1e-9。
    ladder = _ladder(0.1, 4)
    runner = _synthetic(truth, coefficient, order)
    against = order_against_truth("synthetic", runner, ladder, truth)
    richardson = order_by_richardson("synthetic", runner, ladder)
    assert against.asymptotic_order == pytest.approx(order, rel=0, abs=1e-6)
    #: Richardson在纯幂律上仍然精确：相邻差是``C·h^p·(2^p − 1)``，
    #: 那个因子逐档相同于是约掉。这条断言把"精确"这件事钉住。
    assert richardson.asymptotic_order == pytest.approx(order, rel=0, abs=1e-6)


def test_the_ratio_base_follows_the_ladder_not_a_hard_coded_two() -> None:
    """**必红**：把细化比写死成2，本门红。

    等比减半的阶梯上写死2与读阶梯完全等价，于是那个错**只在别的阶梯上才现形**。
    这里取细化比4。
    """

    truth, order = 1.5, 2.0
    ladder = _ladder(0.2, 4, refinement=4.0)
    runner = _synthetic(truth, 5.0, order)
    estimate = order_against_truth("synthetic_r4", runner, ladder, truth)
    assert estimate.asymptotic_order == pytest.approx(order, rel=0, abs=1e-9)
    #: 比值本身是``4^p = 16``——把它一起判，免得"阶对了但比值报错了"。
    assert estimate.ratios[-1] == pytest.approx(16.0, rel=1e-9)


def test_the_estimator_reproduces_the_formal_orders_of_three_real_integrators() -> None:
    """本仓已有的解析问题：显式Euler一阶、辛Euler一阶、velocity Verlet二阶。

    **三条一起判**才排除得掉"永远返回形式阶"这种实现——
    三条的形式阶不全相同，而估计器拿不到形式阶。
    """

    results = verify_estimator(levels=4)
    assert [name for name, *_ in results] == [
        "harmonic/explicit_euler",
        "harmonic/symplectic_euler",
        "harmonic/velocity_verlet",
    ]
    for name, formal, against, richardson in results:
        for estimate in (against, richardson):
            assert estimate.asymptotic_order == pytest.approx(formal, rel=0, abs=0.15), (
                f"{name}的{estimate.method}给{estimate.asymptotic_order!r}，"
                f"形式阶是{formal}"
            )
        #: 两条路径之间也要一致——它们坏起来不一样，同时坏成同一个数不容易。
        assert against.asymptotic_order == pytest.approx(
            richardson.asymptotic_order, rel=0, abs=0.05
        )


def test_the_reported_order_comes_from_the_fine_end_not_the_coarse_end() -> None:
    """**必红**（注错E7）：渐近阶取``orders[0]``而不是``orders[-1]``时本门红。

    纯幂律序列上逐档的阶**全部相同**，于是取哪一头完全等价——
    注错第E7轮因此在前面那些门下面全绿。这里造一条**带高阶修正**的序列
    ``u = u* + h² + 500·h⁴``：粗端被``h⁴``那一项主导（阶接近4），
    细端才是真的``h²``。**渐近区在细端，不在粗端**，这条门把那句话钉住。
    """

    truth = 4.0
    ladder = _ladder(0.1, 6)
    runner = lambda dt_s: truth + dt_s**2 + 500.0 * dt_s**4  # noqa: E731
    #: **两条路径各判一次**：`orders[-1]`在两个函数里各写了一份，
    #: 只判一条的门抓不住另一条写错（注错第E7轮实测）。
    for estimate in (
        order_against_truth("two_scale", runner, ladder, truth),
        order_by_richardson("two_scale", runner, ladder),
    ):
        #: 容差0.15：Richardson比"对真值"那条**慢一档**（它的最后一个差用不上
        #: 最细那一档），于是同一条阶梯上它离渐近值远一点（实测2.0996 vs 2.0018）。
        #: 这个差本身是形制的一部分，不是精度问题。
        assert estimate.asymptotic_order == pytest.approx(2.0, rel=0, abs=0.15), (
            f"{estimate.method}给{estimate.asymptotic_order!r}"
        )
        assert estimate.orders[0] > 3.0, (
            f"{estimate.method}的粗端本该被h⁴那一项主导，"
            f"实测{estimate.orders[0]!r}——这条门的前提变了"
        )


def test_an_exact_hit_on_the_truth_fails_closed_instead_of_dividing_by_zero() -> None:
    """**必红**（注错E9）：某一档误差恰为零时不失败关闭，本门红。

    "对真值"那条路没有舍入地板（它判的是绝对误差，地板由调用方的真值精度定），
    于是零误差在它这里是**唯一**的护栏。`cases/harmonic_oscillator`那道门
    第一行判的正是这件事。
    """

    truth = 2.5

    def runner(dt_s: float) -> float:
        return truth if dt_s < 0.03 else truth + dt_s**2

    with pytest.raises(ConvergenceOrderError, match="不是正数"):
        order_against_truth("exact_hit", runner, _ladder(0.1, 4), truth)


def test_verify_really_runs_the_declared_integrators_not_a_stand_in() -> None:
    """**必红**（注错E10）：把`verify`里的runner换成一条同阶的合成序列，本门红。

    只判"阶对上了形式阶"的门**抓不住"根本没在跑那个积分器"**——
    一条``u = 1 + h^p``的合成序列在那些门下面全绿。
    这条从`verify_estimator`交出的**值本身**判：末位置必须落在
    ``cos(ω·T)``附近，而显式Euler必须明显偏离它（它一阶且反耗散）。
    """

    truth = math.cos(2.0 * 3.0)
    by_name = {name: (formal, against) for name, formal, against, _ in verify_estimator(4)}
    verlet = by_name["harmonic/velocity_verlet"][1]
    euler = by_name["harmonic/explicit_euler"][1]
    assert verlet.values[-1] == pytest.approx(truth, rel=0, abs=1e-5)
    assert abs(euler.values[-1] - truth) > 1e-4, (
        "显式Euler在这条阶梯上本该明显偏离解析解——跑的不是它"
    )
    #: 两条积分器给的值必须不同，否则"跑了三个"这句话是假的。
    assert verlet.values[-1] != euler.values[-1]
    assert by_name["harmonic/symplectic_euler"][1].values[-1] != euler.values[-1]


def test_a_step_independent_observable_reports_no_order_instead_of_a_number() -> None:
    """**必红**：差落到舍入地板以下时报一个数，本门红。

    形制照`cases/harmonic_oscillator`那道门第一行的``assert all(error > 0)``——
    案例页原话："误差全为零——离散误差被别的东西吞了，收敛阶无从谈起"。
    本仓实测撞上过：`box_tipping`的稳定侧末态倾角对步长**逐位不变**，
    而一个不判地板的估计器会从那些1e-16量级的差里算出一个煞有介事的"阶"。
    """

    estimate = order_by_richardson(
        "frozen", lambda dt_s: 3.5 + 1.0e-16 * dt_s, _ladder(0.1, 4)
    )
    assert math.isnan(estimate.asymptotic_order)
    assert "舍入地板" in estimate.verdict
    assert estimate.ratios == ()
    #: 地板之上的那一条必须照常报数——否则本门可以靠"永远说测不出来"通过。
    alive = order_by_richardson("alive", _synthetic(3.5, 2.0, 2.0), _ladder(0.1, 4))
    assert alive.asymptotic_order == pytest.approx(2.0, rel=0, abs=1e-9)
    assert not math.isnan(alive.asymptotic_order)


def test_non_monotone_differences_are_called_out_not_silently_averaged() -> None:
    """逐档差不单调时必须**说出来**：那不是一个收敛阶，是没进渐近区。

    实测两条接触案例在本机小规模上都不单调（决策0087第五节），
    而一个只报"最细两档的阶"的实现会把那个数交出去，**读的人无从知道**。
    """

    values = {0.1: 10.0, 0.05: 10.9, 0.025: 10.2, 0.0125: 10.25, 0.00625: 10.2}
    estimate = order_by_richardson(
        "wobbly", lambda dt_s: values[dt_s], _ladder(0.1, 5)
    )
    assert "不单调" in estimate.verdict
    assert not math.isnan(estimate.asymptotic_order)


def test_richardson_needs_three_rungs() -> None:
    """两档只给得出一个差，给不出比值——失败关闭而不是返回一个数。"""

    with pytest.raises(ConvergenceOrderError, match="至少要三档"):
        order_by_richardson("short", _synthetic(1.0, 1.0, 2.0), _ladder(0.1, 2))


def test_a_ladder_that_misses_the_horizon_fails_closed() -> None:
    """**必红**：不判"末时刻走不到整数步"，本门红。

    这一条是实测逼出来的：第一版滚球阶梯取``h₀ = 1.2e-5``、``T = 0.02``，
    各档真实末时刻差到``±4e-6 s``，而球那时正以``2348 mm/s²``加速——
    **量到的"一阶收敛"整条是末时刻偏移**，且比值恒为2.0000、四档全同，
    漂亮得像一个真结果。
    """

    with pytest.raises(ConvergenceOrderError, match="整数步"):
        assert_horizon_is_an_integer_number_of_steps(
            "ball", _ladder(1.2e-5, 4), 0.02
        )
    #: 修好之后那条阶梯必须过——只判红不判绿的门会把整条功能锁死。
    assert_horizon_is_an_integer_number_of_steps("ball", _ladder(1.25e-5, 4), 0.02)


def test_a_ladder_outside_the_declared_step_bound_fails_closed(monkeypatch) -> None:
    """**必红**：最粗那档越过丙1那条步长上限时不报错，本门红。

    越界那一档给的不是"误差大一点"而是**发散**，
    而一条含发散点的阶梯照样拟合得出一个像模像样的阶。
    """

    import tools.convergence_order as module

    spec = dict(module.PROBLEMS["box_tipping_topple"])
    #: 把最粗一档推到上限之外（实测上限约3.4e-4）。
    spec["local"] = (1.0e-3, 6.0e-2)
    monkeypatch.setitem(module.PROBLEMS, "box_tipping_topple", spec)
    with pytest.raises(ConvergenceOrderError, match="越过了声明的步长上限"):
        run_problem("box_tipping_topple", levels=3, scale="local")


def test_the_estimate_carries_every_intermediate_quantity() -> None:
    """中间量全部进结果——**只给"阶 = 3.97"的结果，读的人无法判断它是不是算错了**。

    与`SolveResult.backtracks`、`ContactStiffnessStepBound`同一条纪律。
    """

    ladder = _ladder(0.1, 4)
    estimate = order_against_truth("synthetic", _synthetic(2.0, 1.0, 2.0), ladder, 2.0)
    assert isinstance(estimate, OrderEstimate)
    assert estimate.dt_ladder == ladder
    assert len(estimate.values) == len(ladder)
    assert len(estimate.differences) == len(ladder)
    assert len(estimate.ratios) == len(ladder) - 1
    assert len(estimate.orders) == len(ladder) - 1
    rendered = estimate.render()
    for dt_s in ladder:
        assert f"{dt_s:<12.6g}".strip() in rendered
