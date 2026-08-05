"""准静态求解——spec/12第4.1节的另一条路（T5搬迁轨第二块）。

引擎至今只有瞬态推进（`integrate`）。**准静态与瞬态是两条并列的路，不是主次**：
准静态解一串平衡态、没有时间，张力这类量是**输入**；瞬态逐步积分、张力是**输出**。
引擎不得把准静态实现成"阻尼很大的瞬态"——那会把一个代数问题伪装成一个更难的
微分问题（spec/12第4.1节）。

本模块求 ``∇U(x) = 0`` 在给定固定自由度下的根：牛顿法 + 回溯线搜索，
稠密LU（部分主元）。零运行时依赖。

**三条申报义务**（spec/12第4.3节）：

* **容差是绝对还是相对必须说出来**——本模块的``residual_tol_n``是**绝对**残差
  （单位N），因为力的量级由载荷定、案例之间可比；相对残差在近零载荷时无意义。
* **没有步长自适应也要说没有**：本模块**无载荷步生长、无同伦延拓**。
  它一次解一个载荷；不收敛就如实返回不收敛，不偷偷二分。
* **步长二分是抢救不是收敛证据**——因此线搜索的回溯次数进结果，
  回溯次数高本身就是"这个问题不好"的信号，不许被"最终收敛了"盖过去。

**适用域**：`U`二次连续可微、Hessian在解附近正定（能量极小而非鞍点）。
不管接触、不管摩擦、不管约束——那些随WDS内核搬迁进来，且要先解spec/12第11.2节
登记的那条（"无状态纯函数"在接触层不成立）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics_engine.energies import EnergyContext, EnergyRegistry
from physics_engine.state import State, StateLayout


class SolveError(ValueError):
    """求解层的一切失败关闭。"""


@dataclass(frozen=True)
class SolveResult:
    """求解结果。**不收敛也是一个结果**，不是异常——调用方要能读到为什么。"""

    state: State
    converged: bool
    iterations: int
    residual_n: float
    #: 线搜索总回溯次数。高值是"这个问题不好"的信号，不许被"最终收敛了"盖过。
    backtracks: int
    #: 不收敛时的原因；收敛时为空串。
    reason: str = ""


def _solve_dense(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """稠密LU（部分主元）。规模是本块的"几十到几百自由度"，够用。

    奇异矩阵**失败关闭**——返回一个垃圾解比报错糟糕得多。
    """

    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-300:
            raise SolveError(
                f"singular system at column {column} — "
                "Hessian在此不可逆（可能是欠约束：有自由度不受任何能量项约束）"
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column] / augmented[column][column]
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                augmented[row][col] -= factor * augmented[column][col]
    return [augmented[index][size] / augmented[index][index] for index in range(size)]


def _max_abs(values) -> float:
    return max((abs(value) for value in values), default=0.0)


def solve_equilibrium(
    registry: EnergyRegistry,
    context: EnergyContext,
    layout: StateLayout,
    initial: tuple[float, ...],
    *,
    fixed_indices: frozenset[int] = frozenset(),
    residual_tol_n: float = 1.0e-9,
    max_iterations: int = 50,
    max_backtracks: int = 40,
) -> SolveResult:
    """牛顿 + 回溯线搜索求``∇U = 0``。

    ``fixed_indices``是被钉住的**标量自由度下标**（不是节点下标）——
    钉一个节点要列出它的三个分量。这一层不猜：钉法必须显式，
    因为"钉住哪些自由度"就是边界条件本身，猜错等于换了个问题。
    """

    if not initial:
        raise SolveError("initial state must be nonempty")
    if any(index < 0 or index >= len(initial) for index in fixed_indices):
        raise SolveError("fixed_indices out of range")
    free = [index for index in range(len(initial)) if index not in fixed_indices]
    if not free:
        raise SolveError("every degree of freedom is fixed — 没有要解的东西")

    state = State(layout=layout, vector=tuple(float(value) for value in initial))
    backtracks = 0
    for iteration in range(1, max_iterations + 1):
        energy, gradient, hessian = registry.total(
            state, context, need_gradient=True, need_hessian=True
        )
        assert gradient is not None and hessian is not None
        residual = _max_abs(gradient[index] for index in free)
        if residual <= residual_tol_n:
            return SolveResult(state, True, iteration - 1, residual, backtracks)

        reduced = [[hessian[row][column] for column in free] for row in free]
        rhs = [-gradient[index] for index in free]
        step = _solve_dense(reduced, rhs)
        if not all(math.isfinite(value) for value in step):
            return SolveResult(
                state, False, iteration, residual, backtracks,
                "牛顿步不是有限值——Hessian病态或能量在此不可微",
            )

        # 回溯线搜索：能量必须真的下降。**下降失败不是"再试一次"，是问题的信号**。
        scale = 1.0
        accepted = False
        for _ in range(max_backtracks):
            trial_vector = list(state.vector)
            for offset, index in enumerate(free):
                trial_vector[index] = state.vector[index] + scale * step[offset]
            try:
                trial = State(layout=layout, vector=tuple(trial_vector))
                trial_energy, _, _ = registry.total(trial, context)
            except (ValueError, ZeroDivisionError):
                scale *= 0.5
                backtracks += 1
                continue
            if trial_energy <= energy:
                state = trial
                accepted = True
                break
            scale *= 0.5
            backtracks += 1
        if not accepted:
            return SolveResult(
                state, False, iteration, residual, backtracks,
                f"线搜索{max_backtracks}次回溯后仍无法降低能量——"
                "这不是步长问题，是牛顿方向不再是下降方向（可能已在鞍点或Hessian非正定）",
            )

    _, gradient, _ = registry.total(state, context, need_gradient=True)
    assert gradient is not None
    return SolveResult(
        state, False, max_iterations, _max_abs(gradient[index] for index in free),
        backtracks, f"达到最大迭代次数{max_iterations}仍未收敛",
    )


__all__ = ["SolveError", "SolveResult", "solve_equilibrium"]
