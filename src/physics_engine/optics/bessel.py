"""一阶第一类贝塞尔函数`J1`——艾里斑要的那一个，标准库自己实现。

**为什么自己写**：0016甲案裁定核心零运行时依赖，`scipy.special.j1`不在选项里；
而`math`没有贝塞尔。所以`J1`是本块唯一一处"要自己造轮子"的地方，
它的精度因此必须**申报**而不是假定。

两段实现（形制取Abramowitz & Stegun第9章的经典分段）：

* `|x| <= SERIES_LIMIT`：上升级数
  ``J1(x) = sum_k (-1)^k (x/2)^(2k+1) / (k! (k+1)!)``。
  级数本身是精确的，限制它的是**相消**：最大项与结果之比随x增长，
  x=20时已达1.1e8，把机器eps放大到1e-8量级。
* `|x| > SERIES_LIMIT`：Hankel渐近展开（A&S 9.2.5—9.2.10，nu=1即mu=4）
  ``J1 ~ sqrt(2/(pi x)) [P cos(x - 3pi/4) - Q sin(x - 3pi/4)]``，
  **按最优截断**（项一旦不再变小就停）。渐近展开是发散的，
  继续加项会让结果变坏——不设最优截断而写死项数是这类实现的常见错误。

分段点`SERIES_LIMIT = 12.0`是**测出来的**，不是猜的：两段各自的绝对误差
随x一升一降，12附近交叉。实测（对照见下）：

| 分段点 | [0,60]上最大绝对误差 |
|---|---|
| 11.0 | 2.60e-12 |
| **12.0** | **8.27e-13** |
| 13.0 | 1.24e-12 |
| 14.0 | 4.97e-12 |

**申报的精度**：`|J1_算 - J1_真| <= 1e-12`，`x`取遍`[0, 60]`步长0.031的1936个点，
对照是贝塞尔积分``J1(x) = (1/pi) int_0^pi cos(t - x sin t) dt``的周期梯形求值
（周期解析被积函数上梯形法几何收敛，128/512/2048节点三档互差<1e-15，
即该对照自身已到浮点地板）。**申报的是绝对误差不是相对误差**，
理由是硬的：`J1`有零点，零点邻域的相对误差没有意义
（x=3.83处`J1≈3e-12`，相对误差3e-5而绝对误差9e-17）。

**这条申报是被验的不是被声称的**：`tests/test_optics.py`有一条`batch`档门
逐点跑上面那1936个点（0.23秒），且**两个方向都断**——申报值不成立要红，
申报值比实测大一个数量级以上也要红（写松的申报会让下游容差跟着虚胖）。

**域外**：`|x| > 60`未测。渐近段在大x上只会更好（误差随x单调下降，
x=50实测1.4e-17），但"只会更好"是推断不是实测，所以按spec/13的诚实条款
写成"未测"而不是"成立"。

零运行时依赖，无状态纯函数。
"""

from __future__ import annotations

import math

from physics_engine.optics.errors import OpticsError

#: 上升级数与渐近展开的分界。实测选出（见模块docstring的对照表），改它要重测。
SERIES_LIMIT: float = 12.0

#: 申报精度：`[0, 60]`上的最大绝对误差上界。判据引用它，不许在测试里私改。
J1_ABSOLUTE_ACCURACY: float = 1.0e-12

#: 申报精度成立的自变量范围上界。超出即"未测"，不是"成立"。
J1_TESTED_ARGUMENT_MAX: float = 60.0

#: 级数的收敛判据与项数上限（上限只防病态输入，正常路径远达不到）。
_SERIES_EPS: float = 1.0e-18
_SERIES_MAX_TERMS: int = 200

#: Hankel展开的``mu = 4 nu^2``，nu=1。写成常量是因为**它不是4这个数字本身**，
#: 而是"四倍阶数平方"——换成J0要改成0，裸的4会被当成某个物理系数。
_HANKEL_MU: float = 4.0

#: 渐近展开的项数上限。真正的停止条件是最优截断（项不再变小），
#: 这个上限只防x极大时白跑。
_HANKEL_MAX_TERMS: int = 64

#: `x - 3pi/4`里的相位。J1的Hankel相位是``x - nu*pi/2 - pi/4``，nu=1。
_HANKEL_PHASE: float = 0.75 * math.pi


def _j1_series(x: float) -> float:
    """上升级数。小x精确，大x被相消吃掉——所以只在`|x| <= SERIES_LIMIT`上用。"""

    half = 0.5 * x
    term = half
    total = term
    for k in range(1, _SERIES_MAX_TERMS + 1):
        term *= -(half * half) / (k * (k + 1))
        total += term
        if abs(term) <= _SERIES_EPS * abs(total):
            break
    return total


def _j1_hankel(x: float) -> float:
    """Hankel渐近展开，**最优截断**：项一旦不再变小就停。

    渐近级数发散，继续加项结果会变坏。写死项数的实现在中等x上会比停得早的差
    一到两个数量级——这正是分段点必须实测的原因。
    """

    p = 1.0
    q = 0.0
    term = 1.0
    smallest = 1.0
    for k in range(1, _HANKEL_MAX_TERMS + 1):
        term *= (_HANKEL_MU - (2 * k - 1) ** 2) / (k * 8.0 * x)
        magnitude = abs(term)
        if magnitude > smallest:
            break
        smallest = magnitude
        if k % 2 == 0:
            p += term if (k // 2) % 2 == 0 else -term
        else:
            q += term if ((k - 1) // 2) % 2 == 0 else -term
    phase = x - _HANKEL_PHASE
    return math.sqrt(2.0 / (math.pi * x)) * (p * math.cos(phase) - q * math.sin(phase))


def bessel_j1(x: float) -> float:
    """一阶第一类贝塞尔函数。精度申报见模块docstring。

    `J1`是奇函数（``J1(-x) = -J1(x)``），负半轴走对称而不是另写一段。
    """

    value = float(x)
    if not math.isfinite(value):
        raise OpticsError(f"bessel_j1需要有限自变量，收到{x!r}")
    if value < 0.0:
        return -bessel_j1(-value)
    if value <= SERIES_LIMIT:
        return _j1_series(value)
    return _j1_hankel(value)


__all__ = [
    "J1_ABSOLUTE_ACCURACY",
    "J1_TESTED_ARGUMENT_MAX",
    "SERIES_LIMIT",
    "bessel_j1",
]
