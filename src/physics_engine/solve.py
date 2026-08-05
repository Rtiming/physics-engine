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
  **它没有默认值，必须由调用方显式给出**（决策0030第十节第1条）：绝对残差的
  可达地板随问题规模上升——实测同一个悬臂，10段时残差地板2.4e-10 N、
  160段时5.6e-8 N，因为弯曲刚度标度``EI/h³``按``h⁻³``增长。
  一个"看起来能用、一放大就不收敛"的默认值比没有默认值糟糕得多：
  它把"这个容差对我的载荷尺度合不合适"这个**必须由调用方回答**的问题
  伪装成了库的实现细节。**参考取法**：总载荷的1e-9到1e-10。
* **没有步长自适应也要说没有**：本模块**无载荷步生长、无同伦延拓**。
  它一次解一个载荷；不收敛就如实返回不收敛，不偷偷二分。
* **步长二分是抢救不是收敛证据**——因此线搜索的回溯次数进结果，
  回溯次数高本身就是"这个问题不好"的信号，不许被"最终收敛了"盖过去。
* **求解那一步是稠密LU**：实测成本按自由自由度约``n^2.6``增长（决策0030第四节），
  交互级边界约200个自由自由度、本机批级边界约1536个。装配已在0028稀疏化（16.4×），
  **求解这一步没有**——它是全仓当前最陡的复杂度，也是下一个在册杠杆。

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


def bandwidth_of(entries: dict[tuple[int, int], float], position: dict[int, int]) -> int:
    """约化矩阵的半带宽：``max|i−j|``（只数结构非零）。

    链式问题的Hessian是**带状**的——实测自重悬臂在40/160/320段上半带宽恒为**2**，
    因为弯曲模板只耦合相邻三个节点，而本案例每节点只有一个自由自由度。
    """

    band = 0
    for (row, column), value in entries.items():
        if value == 0.0:
            continue
        i, j = position.get(row), position.get(column)
        if i is not None and j is not None:
            band = max(band, abs(i - j))
    return band


def _solve_banded(matrix: list[list[float]], rhs: list[float], band: int) -> list[float]:
    """带状LU（部分主元）+ 前代 + 回代。复杂度``O(m·b²)``而不是``O(m³)``。

    **这是换数值路径，不是跳过零——如实声明。** 第一版试图让带状版与
    `_solve_dense`逐字节相同，做法是给高斯-约当加带宽限制。**那是错的**：
    高斯-约当上下都消，会把带状结构破坏掉、填充扩散到整个矩阵，
    限制带宽就得到错答案（实测收敛比从4.0掉到0.01）。带状只对**LU+回代**成立。

    因此本函数与`_solve_dense`的关系是spec/13第一节义务2的**第二档**：
    数值路径不同 → **分量声明容差**对拍，不是逐字节。容差与理由见
    `test_banded_matches_dense_within_a_declared_tolerance`。

    部分主元把上带宽从``b``扩到``2b``（LAPACK的``kl+ku``形制），
    `reach`按此取；主元只在带内找，因为带外该列恒为零。
    """

    size = len(rhs)
    lower = min(band, size - 1)
    upper = min(2 * band, size - 1)
    augmented = [row[:] for row in matrix]
    right = list(rhs)
    for step_index in range(size - 1):
        pivot_high = min(step_index + lower, size - 1)
        pivot = max(
            range(step_index, pivot_high + 1),
            key=lambda row: abs(augmented[row][step_index]),
        )
        if abs(augmented[pivot][step_index]) < 1.0e-300:
            raise SolveError(
                f"singular system at column {step_index} — "
                "Hessian在此不可逆（可能是欠约束：有自由度不受任何能量项约束）"
            )
        if pivot != step_index:
            augmented[step_index], augmented[pivot] = augmented[pivot], augmented[step_index]
            right[step_index], right[pivot] = right[pivot], right[step_index]
        column_high = min(step_index + upper, size - 1)
        for row in range(step_index + 1, pivot_high + 1):
            factor = augmented[row][step_index] / augmented[step_index][step_index]
            if factor == 0.0:
                continue
            augmented[row][step_index] = 0.0
            for column in range(step_index + 1, column_high + 1):
                augmented[row][column] -= factor * augmented[step_index][column]
            right[row] -= factor * right[step_index]
    if abs(augmented[size - 1][size - 1]) < 1.0e-300:
        raise SolveError(
            f"singular system at column {size - 1} — "
            "Hessian在此不可逆（可能是欠约束：有自由度不受任何能量项约束）"
        )
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = right[row]
        for column in range(row + 1, min(row + upper, size - 1) + 1):
            total -= augmented[row][column] * solution[column]
        solution[row] = total / augmented[row][row]
    return solution


def _max_abs(values) -> float:
    return max((abs(value) for value in values), default=0.0)


def solve_equilibrium(
    registry: EnergyRegistry,
    context: EnergyContext,
    layout: StateLayout,
    initial: tuple[float, ...],
    *,
    fixed_indices: frozenset[int] = frozenset(),
    residual_tol_n: float,
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
        energy, gradient, _ = registry.total(state, context, need_gradient=True)
        assert gradient is not None
        residual = _max_abs(gradient[index] for index in free)
        if residual <= residual_tol_n:
            return SolveResult(state, True, iteration - 1, residual, backtracks)

        # 稀疏取Hessian再压成自由自由度上的稠密块。**不建全尺寸稠密矩阵**——
        # 实测结构非零只占0.5852%，建它是本装配唯一的复杂度问题（decisions/0026）。
        entries = registry.hessian_entries(state, context)
        position = {index: offset for offset, index in enumerate(free)}
        reduced = [[0.0] * len(free) for _ in free]
        for (row, column), value in entries.items():
            row_offset = position.get(row)
            column_offset = position.get(column)
            if row_offset is not None and column_offset is not None:
                reduced[row_offset][column_offset] = value
        rhs = [-gradient[index] for index in free]
        # 带状问题走带状消元——链式Hessian的半带宽实测为2，`O(m³)`降到`O(m·b²)`。
        # 判据是**结构性的**：带宽小于规模的三分之一才走，否则带状没有意义。
        band = bandwidth_of(entries, position)
        step = (
            _solve_banded(reduced, rhs, band)
            if 2 * band + 1 < len(free) // 3
            else _solve_dense(reduced, rhs)
        )
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


def _banded_cholesky_is_positive(matrix: list[list[float]], band: int) -> bool:
    """带状Cholesky；**全部主元为正 ⟺ 矩阵正定**（Sylvester惯性定律）。

    Cholesky**不需要主元交换**（正定矩阵天然对角占优够用），所以带宽严格保持，
    复杂度``O(m·b²)``而不是``O(m³)``——与`_solve_banded`同一条理由。
    非正定时在某个主元上出现``<= 0``而当场返回False，不继续算完。

    **这不是特征值分解**：它只回答"是不是正定"，不给最小特征值。
    要那个数就得上真正的特征求解器，本仓今天没有，**不假装有**。
    """

    size = len(matrix)
    upper = min(band, size - 1) if size else 0
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(max(0, row - upper), row + 1):
            total = matrix[row][column]
            for k in range(max(0, row - upper), column):
                total -= lower[row][k] * lower[column][k]
            if row == column:
                if total <= 0.0:
                    return False
                lower[row][row] = math.sqrt(total)
            else:
                lower[row][column] = total / lower[column][column]
    return True


def tangent_stiffness_is_positive_definite(
    registry: EnergyRegistry,
    context: EnergyContext,
    state: State,
    *,
    fixed_indices: frozenset[int] = frozenset(),
) -> bool:
    """给定构型下**约化切线刚度是否正定**——即这个平衡态是极小还是鞍点。

    本模块开头申报的适用域写着"Hessian在解附近正定（能量极小而非鞍点）"，
    但在本函数之前**没有任何办法查这一条**——申报了一个无法验证的前提。
    本函数把它变成可查的：`solve_equilibrium`返回``converged=True``只说明
    ``∇U = 0``，**不说明那是不是极小**；牛顿法对鞍点一样收敛。

    **失稳判据就建在这上面。** Euler屈曲的定义正是"切线刚度失去正定性的那一点"：
    压缩使`AxialStretch`的横向项``(k·ε/L)·(I − d⊗d)``变负（``ε < 0``），
    到某个载荷时它抵消掉弯曲刚度。对载荷二分本函数的真假翻转点，
    就得到离散临界载荷——**而这条路径完全不需要牛顿法穿过分岔点**，
    所以它不受本模块"无信赖域、Hessian非正定时无修正"那条缺陷的影响
    （`cases/euler_buckling`第四节量化登记了直接穿越会发生什么）。

    ``fixed_indices``与`solve_equilibrium`同义：被钉住的**标量自由度下标**。
    自由自由度为空即失败关闭——"空矩阵是正定的"是个陷阱式的真命题。
    """

    size = len(state.vector)
    if any(index < 0 or index >= size for index in fixed_indices):
        raise SolveError("fixed_indices out of range")
    free = [index for index in range(size) if index not in fixed_indices]
    if not free:
        raise SolveError(
            "every degree of freedom is fixed — 空矩阵在形式上正定，"
            "但那句真话回答不了任何问题，所以这里失败关闭"
        )
    entries = registry.hessian_entries(state, context)
    position = {index: offset for offset, index in enumerate(free)}
    reduced = [[0.0] * len(free) for _ in free]
    for (row, column), value in entries.items():
        row_offset = position.get(row)
        column_offset = position.get(column)
        if row_offset is not None and column_offset is not None:
            reduced[row_offset][column_offset] = value
    return _banded_cholesky_is_positive(reduced, bandwidth_of(entries, position))


__all__ = [
    "SolveError",
    "SolveResult",
    "bandwidth_of",
    "solve_equilibrium",
    "tangent_stiffness_is_positive_definite",
]
