"""二维复数场与基2 FFT——0031声明的"光学下一块"的地基（决策0086）。

0031第五节把"二维FFT场传播"列进负空间时写着"**是下一块**"；`optics/__init__.py`
的负空间声明里同一句话也在。本模块把那一半兑现：**场容器与变换**。
物理（角谱、菲涅耳、夫琅禾费）在`propagation.py`。

零运行时依赖（AGENTS.md本仓纪律）：只用`math`/`cmath`与内建`complex`。
**不import numpy**——0031第3.1节说"加速档的正当位置是二维场传播"，
那句话说的是**将来可以有**，不是"这一块必须有"。本块先把零依赖的正本写对；
加速档要进来，进来的条件是spec/13第一节的"优化先profile"加两实现逐字节对拍，
不是"这里有数组所以该上numpy"。

## 三条约定（每一条都是一道会静默错的边界，所以都有名字）

### 一、变换的符号与归一化

正变换取``exp(-2 pi i k n / N)``（`FORWARD_TRANSFORM_SIGN = -1`），
逆变换取``exp(+2 pi i k n / N)``并**只在逆变换上除N**
（`INVERSE_SCALES_BY_RECIPROCAL_COUNT = True`）。这与NumPy/FFTW的默认同口径。

**为什么写成常量而不是"大家都知道"**：符号错了图样左右镜像、归一化错了强度差`N`倍，
两者都不报错。0031第3.3节那张单位边界表立的就是这条实践——
"能量→力→加速度的每一处单位边界都要有名字"，变换的符号与归一化是同一类边界。
Parseval因此写作``sum|x|^2 = (1/N) sum|X|^2``——**归一化不对称，恒等式就不对称**。

### 二、非2幂的长度：**失败关闭**（本模块的裁决之一）

`fft`只吃2的幂，别的长度当场炸。三个候选里选它的理由：

* **补零不是同一个问题的另一种算法，是另一个物理问题**。把长度`N`的场补到`2^m`，
  频率栅距从``1/(N dx)``变成``1/(2^m dx)``、窗口的物理宽度变了、周期卷积的
  折返边界也变了。这些改动全都"看起来更好"（谱更密、图样更光滑），
  **而且一个字的错误都不报**——正是本仓反复吃亏的那一类形态（0024的1000倍、
  0031第3.3节整张表）。所以补零可以做，但**必须是调用方写下来的一次决定**：
  `zero_pad_to_power_of_two`是个有名字的动词，`fft`自己不偷偷做；
* **Bluestein（chirp-z）能算任意长度，但它的精度要另行申报**。它把DFT化成一次
  长度``>= 2N-1``的循环卷积，中间要乘两次chirp``exp(i pi n^2 / N)``——
  `n^2`在大N上先溢出有效位再取模，相位精度是另一套误差模型。本仓对自造轮子的
  纪律是"精度必须申报而不是假定"（0031第3.2节`J1`那一段）。今天没有任何
  消费方要非2幂，**为一个想象中的消费方预支一套没测过的精度**正是代码三前提
  第二条禁的事；
* 失败关闭与本仓的诚实条款同向：**"这个长度我不会算"与"我给你一个别的长度的答案"
  是两件事**。

触发条件（GAP）：**真的出现非2幂长度的消费方时**重开这条，届时要裁的是
"补零还是Bluestein"，并且要连同精度申报一起做。

### 三、复数怎么落盘：**一个复数就是一个二元组`[实部, 虚部]`**（本模块的第二条裁决）

这是decisions/0052第八节明写的未裁项，**它定的是全仓所有复数量的约定，
不只是光学的**。裁决与理由：

**约定**：一个复数在字节形制上是**长度2的浮点数组，次序固定为
``(实部, 虚部)``**（`COMPLEX_COMPONENT_ORDER`）。一串复数是二元组的序列；
要走`oracles`的定长二进制通道时按**C序展平**成`float64`（于是逐元素交错），
因为`oracles.flatten_values`对嵌套列表本来就是C序展平——
**这两种形态是同一条规则的两次应用，不是两条规则**。

四条理由：

1. **它不引入任何新机械**。`oracles._pair_up`对列表逐分量配对，于是
   实部虚部各自吃同一条容差、报错时路径带`[0]`/`[1]`；`canonical`的
   `allow_nan=False`自动挡住`nan`/`inf`；`array_logical_sha256`的
   `float64`小端流原样可用。换成对象``{"real":…,"imag":…}``要在
   `_pair_up`里加一条Mapping分支，换成字符串``"1+2j"``要加一套解析器——
   **两者都是为了一个复数去改全仓的判据机械**；
2. **它与本仓既有的向量形制同形**。三维向量在本仓就是长度3的浮点数组，
   分量次序靠位置不靠键名。复数是二维实向量，没有理由自成一格；
3. **极坐标（模+相位）被否决，理由是它在浮点上不是双射**。零振幅处相位无定义，
   而场的零点恰恰是衍射图样里最要紧的地方；即使避开零点，
   ``(r, theta) -> z -> (r', theta')``的往返也会掉位。
   本模块有一条门实测这件事（`float.hex()`逐字节对拍：直角形制往返逐位相同，
   极坐标形制在同一个数上掉位）；
4. **实部虚部的次序写成常量**，因为反了不报错——它与波数那两套差2pi的约定同类。

**这条约定今天的执行体在光学域里，但它不是光学的**。按域隔离门，
`electromagnetics`不许import `optics`——所以第二个域出现复数量时，
不能靠import本模块来遵守它。**升基座（`canonical.py`）的触发条件因此写死**：
第二个域真的要落盘复数量的那一天。今天不升，判据是0035那条——
今天没有任何非光学代码需要它，现在升格就是为想象中的第三个消费方预支通用性；
而本仓有个好性质：真去import它的那一刻域隔离门①当场红，**升格无法悄悄发生**。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from physics_engine.optics.errors import OpticsError

#: 正变换的指数符号：``X_k = sum_n x_n exp(FORWARD_TRANSFORM_SIGN * 2 pi i k n / N)``。
#: 写成常量而不是把``-1``埋进公式：符号反了图样左右镜像且不报错。
FORWARD_TRANSFORM_SIGN: int = -1

#: 逆变换的指数符号。与正变换互为共轭。
INVERSE_TRANSFORM_SIGN: int = +1

#: 归一化落在**逆变换**一侧（除以N），正变换不除。取NumPy/FFTW的默认口径。
#: 它是个布尔常量而不是一句注释，因为Parseval恒等式的形状直接由它决定：
#: ``sum|x|^2 = (1/N) sum|X|^2``——归一化搬到正变换上，这条恒等式就要改写。
INVERSE_SCALES_BY_RECIPROCAL_COUNT: bool = True

#: 复数落盘的分量次序（决策0086）。**位置定义语义，反了不报错**。
COMPLEX_COMPONENT_ORDER: tuple[str, str] = ("real", "imaginary")

#: 一个复数落盘后占几个浮点数。写成常量是给读侧用的：一段长度`2n`的扁平数组
#: 到底是`n`个复数还是`2n`个实数，只能由声明决定，猜不出来。
COMPLEX_COMPONENT_COUNT: int = 2

_TAU: float = 2.0 * math.pi

#: 旋转因子缓存：``(半长, 符号) -> 因子表``。变换尺寸就那么几种，缓存不设上限；
#: 逐次重算``cos/sin``是本实现最贵的一段（实测见决策0086第五节）。
_TWIDDLE_CACHE: dict[tuple[int, int], tuple[complex, ...]] = {}


def is_power_of_two(count: int) -> bool:
    """`count`是不是2的正整数幂（1也是，`2^0`）。"""

    value = int(count)
    return value > 0 and value & (value - 1) == 0


def next_power_of_two(count: int) -> int:
    """不小于`count`的最小2的幂。`count <= 1`给1。"""

    value = int(count)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _require_power_of_two(count: int, what: str) -> int:
    if not is_power_of_two(count):
        raise OpticsError(
            f"{what}必须是2的幂，收到{count!r}——本模块只做基2 Cooley-Tukey，"
            f"非2幂**失败关闭**而不是替你补零到{next_power_of_two(count)}："
            "补零会改频率栅距与窗口宽度且一个字都不报错。"
            "真要补零请显式调`zero_pad_to_power_of_two`（决策0086第二节）"
        )
    return int(count)


def _as_complex(value: object, what: str) -> complex:
    """把输入收成有限复数；`nan`/`inf`当场炸（与`canonical`的`allow_nan=False`同向）。"""

    if isinstance(value, bool):
        raise OpticsError(f"{what}不接受布尔值：{value!r}")
    if isinstance(value, complex):
        number = value
    elif isinstance(value, (int, float)):
        number = complex(value, 0.0)
    else:
        raise OpticsError(f"{what}需要复数或实数，收到{value!r}")
    if not (math.isfinite(number.real) and math.isfinite(number.imag)):
        raise OpticsError(f"{what}必须有限，收到{value!r}")
    return number


def _twiddles(half_length: int, sign: int) -> tuple[complex, ...]:
    """一层蝶形用的旋转因子``exp(sign * 2 pi i k / (2*half_length))``。

    **逐个直接算`cos`/`sin`，不用递推**：递推（每次乘一个固定旋转子）在长变换上
    会把相位误差累到``O(N)``，而逐个算是``O(1) eps``。这是FFT实现最经典的一处
    精度陷阱，代价是多算几次三角函数——所以缓存。
    """

    key = (half_length, sign)
    cached = _TWIDDLE_CACHE.get(key)
    if cached is not None:
        return cached
    length = 2 * half_length
    table = tuple(
        complex(math.cos(sign * _TAU * k / length), math.sin(sign * _TAU * k / length))
        for k in range(half_length)
    )
    _TWIDDLE_CACHE[key] = table
    return table


def _bit_reversed(data: list[complex]) -> list[complex]:
    """原地位反转置换（迭代式Cooley-Tukey的前置重排）。"""

    count = len(data)
    target = 0
    for source in range(1, count):
        bit = count >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target |= bit
        if source < target:
            data[source], data[target] = data[target], data[source]
    return data


def _transform(values: Sequence[complex], sign: int) -> list[complex]:
    """迭代式基2 Cooley-Tukey，原地蝶形。长度必须已校验为2的幂。"""

    data = _bit_reversed([complex(value) for value in values])
    count = len(data)
    length = 2
    while length <= count:
        half = length >> 1
        table = _twiddles(half, sign)
        for start in range(0, count, length):
            for offset in range(half):
                low = start + offset
                high = low + half
                product = table[offset] * data[high]
                even = data[low]
                data[low] = even + product
                data[high] = even - product
        length <<= 1
    return data


def fft(values: Sequence[complex | float]) -> tuple[complex, ...]:
    """正变换``X_k = sum_n x_n exp(-2 pi i k n / N)``，**不归一化**。

    长度必须是2的幂，否则失败关闭（决策0086第二节）。
    """

    checked = [_as_complex(value, "fft的输入") for value in values]
    _require_power_of_two(len(checked), "fft的长度")
    return tuple(_transform(checked, FORWARD_TRANSFORM_SIGN))


def ifft(values: Sequence[complex | float]) -> tuple[complex, ...]:
    """逆变换``x_n = (1/N) sum_k X_k exp(+2 pi i k n / N)``，**在这里除N**。"""

    checked = [_as_complex(value, "ifft的输入") for value in values]
    count = _require_power_of_two(len(checked), "ifft的长度")
    scale = 1.0 / count if INVERSE_SCALES_BY_RECIPROCAL_COUNT else 1.0
    return tuple(item * scale for item in _transform(checked, INVERSE_TRANSFORM_SIGN))


def zero_pad_to_power_of_two(values: Sequence[complex | float]) -> tuple[complex, ...]:
    """尾部补零到2的幂，**由调用方显式调用**。

    这个动词有名字，是因为它**改的是物理设定不是数值细节**：补零之后
    频率栅距从``1/(N dx)``变成``1/(N_pad dx)``（谱看起来更密，其实只是
    对同一条连续谱的插值），窗口的物理宽度也变成``N_pad dx``。
    在场传播里这等于在场的两侧加了一段真空——常常正是想要的，
    但必须是**写下来的一次决定**，不是`fft`替你做的。
    """

    checked = [_as_complex(value, "zero_pad_to_power_of_two的输入") for value in values]
    if not checked:
        raise OpticsError("zero_pad_to_power_of_two不接受空序列")
    padded = next_power_of_two(len(checked))
    checked.extend(complex(0.0, 0.0) for _ in range(padded - len(checked)))
    return tuple(checked)


def signed_frequency_indices(count: int) -> tuple[int, ...]:
    """FFT输出各bin对应的**有符号**频率序号``0,1,...,N/2-1,-N/2,...,-1``。

    与NumPy `fftfreq`的次序同口径（奈奎斯特那一格归到负侧）。
    传播用它把bin映射到空间频率，案例用它把bin映射到衍射角——
    **两处都不许在调用点自己写这个循环**：把``k > N/2``那一半忘了折回负侧，
    图样会在半屏处对折且不报错。
    """

    total = _require_power_of_two(count, "signed_frequency_indices的长度")
    half = total // 2
    return tuple(k if k < half or total == 1 else k - total for k in range(total))


# --- 复数的字节形制（决策0086第三节） -------------------------------------


def complex_to_components(value: complex | float) -> list[float]:
    """一个复数 → ``[实部, 虚部]``（`COMPLEX_COMPONENT_ORDER`）。

    非有限值当场炸——`canonical`的``allow_nan=False``本来就会拒，
    在这里先炸是为了让报错指向那个复数而不是指向一整份清单。
    """

    number = _as_complex(value, "complex_to_components的输入")
    return [number.real, number.imag]


def complex_from_components(components: Sequence[float]) -> complex:
    """``[实部, 虚部]`` → 一个复数。长度不是2当场炸。"""

    items = list(components)
    if len(items) != COMPLEX_COMPONENT_COUNT:
        raise OpticsError(
            f"一个复数的字节形制是长度{COMPLEX_COMPONENT_COUNT}的数组"
            f"（次序{COMPLEX_COMPONENT_ORDER}），收到{items!r}"
        )
    real, imaginary = (float(item) for item in items)
    if not (math.isfinite(real) and math.isfinite(imaginary)):
        raise OpticsError(f"复数分量必须有限：{items!r}")
    return complex(real, imaginary)


def sequence_to_components(values: Sequence[complex | float]) -> list[list[float]]:
    """一串复数 → 二元组的序列。C序展平后即``float64``交错流。"""

    return [complex_to_components(value) for value in values]


def sequence_from_components(rows: Sequence[Sequence[float]]) -> tuple[complex, ...]:
    """二元组的序列 → 一串复数。"""

    return tuple(complex_from_components(row) for row in rows)


# --- 二维场 ---------------------------------------------------------------


@dataclass(frozen=True)
class ComplexField2D:
    """行主序的二维复数场：一个**没有单位**的采样阵列。

    **它不带采样间距**，这是有意的：间距是物理量（米），一旦挂在容器上，
    "这个场是在哪个平面、哪个网格上采的"就会随着容器被到处传，而每一次传递
    都是一次可能错的单位边界。本模块只管数；带单位的东西在`propagation.py`
    的`PropagatedField`上，那里每个字段名都带单位后缀（0031第3.3节的实践）。

    不可变（`frozen`+元组），因为变换全都返回新场——一个被就地改过的场
    与它的FFT之间没有任何东西能保证对应关系。
    """

    rows: tuple[tuple[complex, ...], ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise OpticsError("ComplexField2D至少要有一行")
        width = len(self.rows[0])
        if width == 0:
            raise OpticsError("ComplexField2D至少要有一列")
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise OpticsError(
                    f"ComplexField2D必须是矩形：第0行{width}列，第{index}行{len(row)}列"
                )

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.rows[0])

    @property
    def shape(self) -> tuple[int, int]:
        """``(行数, 列数)``。行是y方向，列是x方向——**本模块全程行=y、列=x**。"""

        return (self.row_count, self.column_count)

    def at(self, row: int, column: int) -> complex:
        return self.rows[row][column]

    def values(self) -> tuple[complex, ...]:
        """行主序（C序）展平。与`oracles.flatten_values`的展平次序同口径。"""

        return tuple(value for row in self.rows for value in row)

    def intensity_rows(self) -> tuple[tuple[float, ...], ...]:
        """逐点``|U|^2``。振幅场→强度场，**这一步不带任何归一化**。"""

        return tuple(
            tuple(value.real * value.real + value.imag * value.imag for value in row)
            for row in self.rows
        )

    def peak_intensity(self) -> float:
        return max(max(row) for row in self.intensity_rows())

    def scaled(self, factor: complex | float) -> ComplexField2D:
        multiplier = _as_complex(factor, "scaled的因子")
        return ComplexField2D(
            tuple(tuple(value * multiplier for value in row) for row in self.rows)
        )

    def transposed(self) -> ComplexField2D:
        return ComplexField2D(tuple(zip(*self.rows, strict=True)))

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[complex | float]]) -> ComplexField2D:
        """从任意可迭代的行构造，逐点做有限性检查。"""

        built = tuple(
            tuple(_as_complex(value, "ComplexField2D的元素") for value in row)
            for row in rows
        )
        return cls(built)

    @classmethod
    def zeros(cls, row_count: int, column_count: int) -> ComplexField2D:
        if row_count <= 0 or column_count <= 0:
            raise OpticsError(f"场的形状必须是正整数：{(row_count, column_count)!r}")
        zero = complex(0.0, 0.0)
        row = (zero,) * column_count
        return cls((row,) * row_count)

    @classmethod
    def from_function(
        cls, row_count: int, column_count: int, function: object
    ) -> ComplexField2D:
        """``function(row, column) -> 复数``逐点求值。孔径掩模都从这里来。"""

        if row_count <= 0 or column_count <= 0:
            raise OpticsError(f"场的形状必须是正整数：{(row_count, column_count)!r}")
        if not callable(function):
            raise OpticsError(f"from_function需要一个可调用对象：{function!r}")
        return cls.from_rows(
            (function(row, column) for column in range(column_count))
            for row in range(row_count)
        )

    def to_components(self) -> list[list[list[float]]]:
        """落盘形制：``行 -> 列 -> [实部, 虚部]``（决策0086第三节）。

        C序展平即``float64``交错流，可以直接喂`oracles.array_logical_sha256`。
        """

        return [sequence_to_components(row) for row in self.rows]

    @classmethod
    def from_components(cls, rows: Sequence[Sequence[Sequence[float]]]) -> ComplexField2D:
        return cls(tuple(sequence_from_components(row) for row in rows))


def fft2(field: ComplexField2D) -> ComplexField2D:
    """二维正变换：**先行后列**（可分离，先列后行给同一个结果）。

    可分离性不是实现细节而是判据：`tests/test_optics_field.py`有一条门
    并排跑两条次序并要求逐位相同——把行列搞混的实现会在非方阵上当场炸，
    在方阵上则给出转置了的图样（**而那个错误在圆孔上完全看不出来**，
    因为圆对称。所以那条门用的是非方阵）。
    """

    _require_power_of_two(field.row_count, "fft2的行数")
    _require_power_of_two(field.column_count, "fft2的列数")
    after_rows = ComplexField2D(tuple(fft(row) for row in field.rows))
    transposed = after_rows.transposed()
    after_columns = ComplexField2D(tuple(fft(row) for row in transposed.rows))
    return after_columns.transposed()


def ifft2(field: ComplexField2D) -> ComplexField2D:
    """二维逆变换。归一化是``1/(行数*列数)``——两次一维逆变换各除一次。"""

    _require_power_of_two(field.row_count, "ifft2的行数")
    _require_power_of_two(field.column_count, "ifft2的列数")
    after_rows = ComplexField2D(tuple(ifft(row) for row in field.rows))
    transposed = after_rows.transposed()
    after_columns = ComplexField2D(tuple(ifft(row) for row in transposed.rows))
    return after_columns.transposed()


__all__ = [
    "COMPLEX_COMPONENT_COUNT",
    "COMPLEX_COMPONENT_ORDER",
    "FORWARD_TRANSFORM_SIGN",
    "INVERSE_SCALES_BY_RECIPROCAL_COUNT",
    "INVERSE_TRANSFORM_SIGN",
    "ComplexField2D",
    "complex_from_components",
    "complex_to_components",
    "fft",
    "fft2",
    "ifft",
    "ifft2",
    "is_power_of_two",
    "next_power_of_two",
    "sequence_from_components",
    "sequence_to_components",
    "signed_frequency_indices",
    "zero_pad_to_power_of_two",
]
