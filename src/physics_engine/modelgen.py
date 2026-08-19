"""参数化模型生成器——spec/11规则2**第二类**的首个实现。

规则2把形状词汇分三类：解析原语（`shapes.py`已有四种）、**参数化生成器**、
网格/SDF资产。第二类至今只有一个引用壳：`shapes.GeneratedShape`有
`algorithm_id`+版本+参数元组，却没有任何一个真实的"参数→形状"纯函数。
本模块补的就是那个空：spec/11规则2点名的三个生成器——**带盘、导轮、骨架**。

四条纪律写在最前面，因为它们各自都有门守着（案例`case/generator_determinism`）：

1. **纯函数**。无状态、无I/O、无随机、无时钟。生成器吃参数吐声明，
   同一组参数每次产出**逐字节相同**的声明——`declaration_bytes`给出那份字节，
   案例连**换`PYTHONHASHSEED`起子进程**都要求字节相同（进程内比两次抓不到
   集合迭代序这类隐患，因为同一进程里的哈希种子是同一个）。
2. **系数化，不写死毫米**。形制直接抄case2的`robot_links.py`：
   "一切长度均来自spec常量，本模块只写无量纲系数，不出现任何连杆长度字面量"。
   于是每个生成器只有**一个**带单位的入参`characteristic_length_mm`，其余全是
   无量纲比值。这条有结构判据守着：把特征长度乘λ，产出的每一个长度**恰好**
   跟着乘λ（λ取2这类2的幂时是逐位相等，见案例第三节）。
3. **参数记录进产出**。每个部件的`GeneratedShape.parameters`带**整次调用**的
   完整参数表，产物因此可溯源、可复现。只盖与自己有关的那几个是不够的：
   法兰盘的半径与筒卷了几层无关，但"这片法兰是哪次调用产的"才是溯源要回答的
   问题——参数表是**调用**的指纹，不是部件的属性。
4. **不猜没声明的几何**。参数不自洽（层数为正而层厚为零、绕满超出法兰、
   骨架点列有重复点）一律失败关闭，不做"取个合理默认"这种静默取舍。

**法兰怎么处理（spec/11第二之二节的缺口，决策0034已裁）**：`FiniteCylinder`
声明了`flange_outer_radius_mm`却没有法兰的轴向尺寸，`geometry.mass_properties`
对它**失败关闭**。决策0034第四节裁决：**本轮不动形状词汇，维持失败关闭**
（两条候选修法都要改规则2的词汇，而今天没有任何消费方在用带法兰的导轮），
并点名"`modelgen`的带盘生成器按同一纪律处理"。

本模块按那条纪律办：**不改词汇，在生成器层用既有原语组合表达带法兰带盘**——
筒一件、两片法兰盘各一件，全是既有的`FiniteCylinder`，
`flange_outer_radius_mm`在本模块产出的任何圆柱上**恒为None**。
词汇没动一个字，`geometry`的失败关闭没动一个字。三条理由：

- **组合表达不需要新词汇**。加字段那条要改规则2，是0034明确压住的；
  而"一个形拆成三件既有原语"用的是行业惯例本身（primitives优先、
  组合表达复杂形），落在生成器里，不落在词汇里。
- 这个分解是**精确的，不是近似**：带法兰带盘的实体
  `{r≤R_b, |z|≤W/2} ∪ {R_b<r≤R_f, W/2<|z|≤W/2+w}`
  与本模块产的"筒(R_b,W) + 两片盘(R_f,w)贴在筒两端外侧"**是同一个点集**，
  且三件互不重叠——所以体积与惯量可以逐件算再按平行轴定理合并，不会重复计数。
- 产出的每一个形都能直接喂给`geometry.mass_properties`。若产带法兰的圆柱，
  产物将**永远算不出质量属性**，"生成器产的形是真形"这条独立判据就无从谈起。

0034的触发条件（WDS碰撞预演批次或case2给出带法兰导轮的书面需求）真的到来时，
本模块是"独立第二个形"那条候选修法的一份**可用实现与一条精确分解的证明**——
是那场裁决的材料，不是对它的抢先。

法兰的两个尺寸都不是本模块发明的：WDS的`CylinderSurface`给法兰
**两个**尺寸——`flange_channel_width_mm`（两壁内侧面的轴向间距）与
`flange_height_mm`（壁高出接触面的径向高度）。本模块的`barrel_width_ratio`
就是前者、`flange_outer_radius_ratio − barrel_radius_ratio`就是后者。
**"法兰没有轴向尺寸"是引擎侧的缺口，不是消费方侧的**——WDS一直是有的。

蒸馏来源（"蒸馏不发明"，标注到条）：

| 本模块的哪一条 | 来源 |
|---|---|
| 只写无量纲系数、一个特征长度入参 | case2 `src/robot_links.py`§J1 |
| 生成器带版本串 | case2 `SCHEMA_VERSION` |
| 沿链插值的锥度（根粗梢细） | case2 `K["fo_h_el"]→K["fo_h_tip"]` |
| 有限宽圆柱的参数面（半径+面宽） | WDS `dynamic/cylinder_contact.py::CylinderSurface` |
| 法兰两尺寸必须成对给出或成对缺省 | 同上（`flange_channel_width_mm`/`flange_height_mm`） |
| 带盘半径随已卷层数生长 | WDS `dynamic/winding_surface.py::WindingSurface.effective_radius_mm` |
| 骨架相邻点重合即拒 | WDS `dynamic/static_obstacle.py::CapsuleObstacle`（退化轴即拒） |
| 装配内部局部偏移与世界位姿分离 | spec/11规则3（形状不携带位姿） |

**增量**（内部外部都没有的，显式标注）：`GeneratedPart`把"一次生成产出多件"
这件事显式化，并给每件一个装配内部的局部偏移。WDS/case2都是"一个对象一个形"
或"整块实体"，没有这层；本模块需要它，是因为法兰走了独立形那条路。

零运行时依赖：只用标准库 + 本仓`shapes`/`canonical`。
公开名只进本模块自己的`__all__`，**不进`__init__.py`**（实验档模块按全路径import）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from physics_engine.canonical import CanonicalProfile, canonical_bytes, canonical_sha256
from physics_engine.shapes import (
    Capsule,
    FiniteCylinder,
    GeneratedShape,
    RoundedBox,
    Sphere,
    Vector3,
)


class ModelGenError(ValueError):
    """参数化生成器的一切失败关闭。

    与`ShapeError`分开是有意的：**形状声明合法不等于参数自洽**。
    "绕了12层的带盘"在`shapes.py`那一层无从判断，因为层数根本不是形状字段;
    它是生成器的入参，只有生成器知道它把筒径顶到了法兰之外。
    """


#: 生成器身份（spec/11规则7：无`algorithm:`前缀即拒，门在`GeneratedShape`里）。
SPOOL_ALGORITHM_ID = "algorithm:modelgen/spool"
ROLLER_ALGORITHM_ID = "algorithm:modelgen/roller"
FORMER_ALGORITHM_ID = "algorithm:modelgen/former"

#: 版本各自独立递增——三个生成器是三份算法，改一个不该让另两个的产物"看起来变了"。
SPOOL_ALGORITHM_VERSION = "1.0.0"
ROLLER_ALGORITHM_VERSION = "1.0.0"
FORMER_ALGORITHM_VERSION = "1.0.0"

#: 声明指纹的规范化参数。取`ensure_ascii=True`：指纹只含数字与ASCII键名，
#: 不带中文，逃逸与否没有可读性代价，而ASCII字节在任何终端/管道下都不会被改写。
#: `file_trailing_newline=False`因为这份字节**不落盘**——它是进程内的指纹口径。
#:
#: **这不是一个已登记的面**（`engine_facets.py`里没有它）：本形制不跨边界、
#: 不进场景文件、不进oracle清单，只作"同一组参数是否产出同一份声明"的比较基准。
#: 哪天它要落盘或要被消费方读，必须先进面清册——那是`AGENTS.md`的面清册义务。
MODELGEN_PROFILE = CanonicalProfile(ensure_ascii=True, file_trailing_newline=False)

#: 产出形的键名。`kind`是本地判别式，不是面字段（同上：不跨边界）。
_SHAPE_KINDS: dict[type, str] = {
    Sphere: "sphere",
    Capsule: "capsule",
    RoundedBox: "rounded_box",
    FiniteCylinder: "finite_cylinder",
}


def _clean(value: object, name: str) -> float:
    """入口处把浮点收敛成一个**规范**的值：非有限即拒，`-0.0`归一成`0.0`。

    `-0.0`那一条是逐字节判据的真实隐患：`-0.0 == 0.0`为真，而
    `json.dumps(-0.0)`是`"-0.0"`、`json.dumps(0.0)`是`"0.0"`——
    同一个几何点会给出两份不同的声明字节。归一在**入口**做，
    因为一旦符号进了算式，它会顺着乘法传播到产出的每一个坐标上。
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelGenError(f"{name} must be a real number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ModelGenError(f"{name} must be finite: {value!r}")
    return 0.0 if number == 0.0 else number


def _positive(value: object, name: str) -> float:
    number = _clean(value, name)
    if number <= 0.0:
        raise ModelGenError(f"{name} must be positive: {value!r}")
    return number


def _nonnegative(value: object, name: str) -> float:
    number = _clean(value, name)
    if number < 0.0:
        raise ModelGenError(f"{name} must be nonnegative: {value!r}")
    return number


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelGenError(f"{name} must be an integer: {value!r}")
    if value < 0:
        raise ModelGenError(f"{name} must be nonnegative: {value!r}")
    return value


def _point_ratio(value: object, name: str) -> Vector3:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelGenError(f"{name} must be a 3-sequence: {value!r}")
    if len(value) != 3:
        raise ModelGenError(f"{name} must have exactly 3 components: {value!r}")
    return (
        _clean(value[0], f"{name}[0]"),
        _clean(value[1], f"{name}[1]"),
        _clean(value[2], f"{name}[2]"),
    )


@dataclass(frozen=True)
class GeneratedPart:
    """一次生成产出的一件：装配内部的局部偏移 + 形。

    **`offset_mm`是装配内部的局部平移，不是世界位姿**（spec/11规则3：
    形状不携带位姿）。世界位姿仍然只经`PosedBody`进来；本字段回答的是
    另一个问题——"这一件在它所属的那个生成装配里，相对装配原点在哪"。
    两者叠加是调用方的事，本模块不做。

    胶囊自带两个端点坐标，所以骨架生成器的偏移恒为零向量；圆柱是绕自身
    局部z轴的轴对称原语，除了偏移没有别的办法把法兰放到筒的两端。
    """

    part_id: str
    offset_mm: Vector3
    shape: GeneratedShape

    def __post_init__(self) -> None:
        if not self.part_id or "/" in self.part_id:
            raise ModelGenError(
                f"part_id must be a nonempty local name without '/': {self.part_id!r} — "
                "装配内部的件名不带命名空间，命名空间由调用方拼进body_id"
            )
        if len(self.offset_mm) != 3:
            raise ModelGenError(f"offset_mm must be a 3-vector: {self.offset_mm!r}")


def _generated(
    algorithm_id: str,
    version: str,
    parameters: tuple[tuple[str, float], ...],
    shape: Sphere | Capsule | RoundedBox | FiniteCylinder,
) -> GeneratedShape:
    return GeneratedShape(
        algorithm_id=algorithm_id,
        algorithm_version=version,
        parameters=parameters,
        shape=shape,
    )


# --------------------------------------------------------------------------
# 带盘 spool
# --------------------------------------------------------------------------

def generate_spool(
    *,
    characteristic_length_mm: float,
    barrel_radius_ratio: float,
    barrel_width_ratio: float,
    flange_outer_radius_ratio: float | None = None,
    flange_width_ratio: float | None = None,
    wound_layers: int = 0,
    layer_thickness_ratio: float = 0.0,
) -> tuple[GeneratedPart, ...]:
    """带盘：有限宽圆筒 + **可选的两片独立法兰盘**。

    产出1件（无法兰）或3件（有法兰）：`barrel`、`flange_low`、`flange_high`。
    法兰盘贴在筒的两端**外侧**（`|z| ∈ [W/2, W/2+w]`），三件互不重叠——
    这个分解与真实带盘实体是同一个点集，见模块文档。

    筒的半径随已卷层数生长（WDS `WindingSurface.effective_radius_mm`）：

        R_eff = (barrel_radius_ratio + wound_layers · layer_thickness_ratio) · L

    ## 本式与`winding`／`drives`那两份半径式子的关系（决策0093）

    本仓有**三处**写着"半径随卷绕生长"，而它们的自变量不同，**不是三份重复**：

    * **本函数**：自变量是**层**``wound_layers``，走无量纲比值，产出的是**形状**；
    * `drives.SpoolTension.radius_mm`：自变量是**匝**，走mm，回答的是**力**
      （``T = M/R``，同扭矩下卷满比空卷张力小``R₀/R``倍）；
    * `winding.WindingPack`：**匝→层**的换算（``turns_per_layer``）与**堆积因子**，
      外加``长度 ↔ 匝``。它是上面两者中间缺的那一段，**不是第三份同样的东西**。

    对应关系由判据钉住而不是由注释钉住：
    `tests/test_winding.py::test_the_stepped_pack_reproduces_the_generator_layer_growth_bit_for_bit`
    取本函数在`cases/generator_determinism`里冻结的那组二进制精确入参，
    判``WindingPack(layer_advance="stepped").radius_mm(m)``与本函数产出的
    ``FiniteCylinder.radius_mm``**逐位相同**。

    **本函数的数值路径与声明指纹一个字节都没动**——`cases/generator_determinism`
    的金标依赖它，0093只加了这段交叉引用。

    参数（除`characteristic_length_mm`外全是无量纲比值）：

    * `characteristic_length_mm`——特征长度L，本生成器唯一带单位的入参；
    * `barrel_radius_ratio`——空盘时的筒半径 / L；
    * `barrel_width_ratio`——筒的**全**宽 / L（即两片法兰内侧面的轴向间距，
      对应WDS的`flange_channel_width_mm`）；
    * `flange_outer_radius_ratio`——法兰外半径 / L，与`flange_width_ratio`
      **成对给出或成对缺省**（WDS形制）；
    * `flange_width_ratio`——单片法兰的轴向厚度 / L；
    * `wound_layers` / `layer_thickness_ratio`——已卷层数与单层厚 / L。

    失败关闭：层数为正而层厚为零（层数是个谎）；绕满后筒径超出法兰外径
    （盘已溢出，几何不再成立）；两个法兰比值只给一个。
    """

    length = _positive(characteristic_length_mm, "characteristic_length_mm")
    barrel_radius = _positive(barrel_radius_ratio, "barrel_radius_ratio")
    barrel_width = _positive(barrel_width_ratio, "barrel_width_ratio")
    layers = _count(wound_layers, "wound_layers")
    layer_thickness = _nonnegative(layer_thickness_ratio, "layer_thickness_ratio")
    if layers > 0 and layer_thickness <= 0.0:
        raise ModelGenError(
            "wound_layers > 0 requires a positive layer_thickness_ratio — "
            "层数不为零而层厚为零，等于声明了一沓不占厚度的带材"
        )
    if (flange_outer_radius_ratio is None) != (flange_width_ratio is None):
        raise ModelGenError(
            "flange_outer_radius_ratio and flange_width_ratio must be set together "
            "or both be None — 法兰的径向与轴向两个尺寸缺一个，形就没被定住"
            "（WDS CylinderSurface同款成对约束）"
        )

    effective_radius_ratio = barrel_radius + layers * layer_thickness
    parameters: list[tuple[str, float]] = [
        ("characteristic_length_mm", length),
        ("barrel_radius_ratio", barrel_radius),
        ("barrel_width_ratio", barrel_width),
        ("wound_layers", float(layers)),
        ("layer_thickness_ratio", layer_thickness),
    ]
    flange_radius = flange_width = None
    if flange_outer_radius_ratio is not None and flange_width_ratio is not None:
        flange_radius = _positive(flange_outer_radius_ratio, "flange_outer_radius_ratio")
        flange_width = _positive(flange_width_ratio, "flange_width_ratio")
        if flange_radius < effective_radius_ratio:
            raise ModelGenError(
                f"the wound barrel has overflowed its flange: effective barrel ratio "
                f"{effective_radius_ratio!r} exceeds flange_outer_radius_ratio "
                f"{flange_radius!r} — 绕满后筒径已越过法兰外径，几何不再成立"
            )
        parameters.append(("flange_outer_radius_ratio", flange_radius))
        parameters.append(("flange_width_ratio", flange_width))
    stamp = tuple(parameters)

    half_barrel_mm = 0.5 * barrel_width * length
    parts = [
        GeneratedPart(
            part_id="barrel",
            offset_mm=(0.0, 0.0, 0.0),
            shape=_generated(
                SPOOL_ALGORITHM_ID,
                SPOOL_ALGORITHM_VERSION,
                stamp,
                FiniteCylinder(
                    radius_mm=effective_radius_ratio * length,
                    half_width_mm=half_barrel_mm,
                ),
            ),
        )
    ]
    if flange_radius is not None and flange_width is not None:
        half_flange_mm = 0.5 * flange_width * length
        centre_mm = half_barrel_mm + half_flange_mm
        for part_id, sign in (("flange_low", -1.0), ("flange_high", 1.0)):
            parts.append(
                GeneratedPart(
                    part_id=part_id,
                    offset_mm=(0.0, 0.0, _clean(sign * centre_mm, "flange offset")),
                    shape=_generated(
                        SPOOL_ALGORITHM_ID,
                        SPOOL_ALGORITHM_VERSION,
                        stamp,
                        FiniteCylinder(
                            radius_mm=flange_radius * length,
                            half_width_mm=half_flange_mm,
                        ),
                    ),
                )
            )
    return tuple(parts)


# --------------------------------------------------------------------------
# 导轮 roller
# --------------------------------------------------------------------------

def generate_roller(
    *,
    characteristic_length_mm: float,
    radius_ratio: float,
    face_width_ratio: float,
) -> tuple[GeneratedPart, ...]:
    """导轮：一个有限宽圆柱（WDS `CylinderSurface`的参数面）。

    产出1件：`face`。参数：特征长度L、半径比、**面宽**比（全宽/L）。

    **不产什么，写在明处**：WDS的`CylinderSurface`还有`edge_radius_mm`
    （轮缘圆角）与`concave_depth_mm`/`concave_half_width_mm`（浅凹套），
    `shapes.FiniteCylinder`**没有对应字段**，所以本生成器既不吃它们也不产它们。
    这不是"暂未支持"式的含糊：带这两样的导轮，本模块产不出忠实声明，
    调用方要么接受锐边直筒近似，要么等词汇长出对应原语。
    """

    length = _positive(characteristic_length_mm, "characteristic_length_mm")
    radius = _positive(radius_ratio, "radius_ratio")
    face_width = _positive(face_width_ratio, "face_width_ratio")
    parameters = (
        ("characteristic_length_mm", length),
        ("radius_ratio", radius),
        ("face_width_ratio", face_width),
    )
    return (
        GeneratedPart(
            part_id="face",
            offset_mm=(0.0, 0.0, 0.0),
            shape=_generated(
                ROLLER_ALGORITHM_ID,
                ROLLER_ALGORITHM_VERSION,
                parameters,
                FiniteCylinder(
                    radius_mm=radius * length,
                    half_width_mm=0.5 * face_width * length,
                ),
            ),
        ),
    )


# --------------------------------------------------------------------------
# 骨架 former
# --------------------------------------------------------------------------

def generate_former(
    *,
    characteristic_length_mm: float,
    skeleton_ratios: Sequence[Vector3],
    root_radius_ratio: float,
    tip_radius_ratio: float,
) -> tuple[GeneratedPart, ...]:
    """骨架：由骨架点列生成**胶囊链**——case2骨架驱动形制的最小版。

    n个骨架点产n−1件：`link_0` … `link_{n−2}`，每件一个胶囊，端点就是
    相邻两个骨架点。半径沿链**按段序插值**（case2前臂的`fo_h_el → fo_h_tip`
    锥度）：第k段取中点参数`t = (k + 0.5) / 段数`，

        r_k = (root_radius_ratio + (tip_radius_ratio − root_radius_ratio) · t) · L

    `t ∈ (0, 1)`且两端半径皆正，所以`r_k`是两个正数的凸组合，恒为正——
    这里不需要再补一道半径检查，正性是构造性的。

    **骨架点是无量纲比值**（`skeleton_ratios`），乘L才是毫米。这是case2那条
    "不出现任何连杆长度字面量"的直接后果：骨架的形状由比值定，尺度由L定，
    换一台同构不同尺寸的机器只改L。

    失败关闭：点数少于2（连一段都构不成）；相邻点重合（胶囊轴退化，
    方向未定义——WDS `CapsuleObstacle`同款拒收）。
    """

    length = _positive(characteristic_length_mm, "characteristic_length_mm")
    root_radius = _positive(root_radius_ratio, "root_radius_ratio")
    tip_radius = _positive(tip_radius_ratio, "tip_radius_ratio")
    if isinstance(skeleton_ratios, (str, bytes)) or not isinstance(skeleton_ratios, Sequence):
        raise ModelGenError(f"skeleton_ratios must be a sequence: {skeleton_ratios!r}")
    points = tuple(
        _point_ratio(entry, f"skeleton_ratios[{index}]")
        for index, entry in enumerate(skeleton_ratios)
    )
    if len(points) < 2:
        raise ModelGenError(
            f"a skeleton needs at least 2 points, got {len(points)} — "
            "少于两个点连一段胶囊都构不成"
        )
    for index in range(len(points) - 1):
        if points[index] == points[index + 1]:
            raise ModelGenError(
                f"skeleton points {index} and {index + 1} coincide: {points[index]!r} — "
                "胶囊轴退化，方向未定义（WDS CapsuleObstacle同款拒收）"
            )

    parameters: list[tuple[str, float]] = [
        ("characteristic_length_mm", length),
        ("root_radius_ratio", root_radius),
        ("tip_radius_ratio", tip_radius),
        ("skeleton_point_count", float(len(points))),
    ]
    for index, point in enumerate(points):
        for axis, component in zip("xyz", point, strict=True):
            parameters.append((f"skeleton_ratio_{index}_{axis}", component))
    stamp = tuple(parameters)

    segments = len(points) - 1
    parts = []
    for index in range(segments):
        fraction = (index + 0.5) / segments
        radius_ratio = root_radius + (tip_radius - root_radius) * fraction
        parts.append(
            GeneratedPart(
                part_id=f"link_{index}",
                offset_mm=(0.0, 0.0, 0.0),
                shape=_generated(
                    FORMER_ALGORITHM_ID,
                    FORMER_ALGORITHM_VERSION,
                    stamp,
                    Capsule(
                        point_a_mm=_scaled(points[index], length),
                        point_b_mm=_scaled(points[index + 1], length),
                        radius_mm=radius_ratio * length,
                    ),
                ),
            )
        )
    return tuple(parts)


def _scaled(point: Vector3, length: float) -> Vector3:
    return (
        _clean(point[0] * length, "scaled x"),
        _clean(point[1] * length, "scaled y"),
        _clean(point[2] * length, "scaled z"),
    )


# --------------------------------------------------------------------------
# 声明指纹：确定性判据的**逐字节对象**
# --------------------------------------------------------------------------

def _shape_document(shape: Sphere | Capsule | RoundedBox | FiniteCylinder) -> dict[str, Any]:
    kind = _SHAPE_KINDS.get(type(shape))
    if kind is None:
        raise ModelGenError(f"no declaration form for {type(shape).__name__}")
    if isinstance(shape, Sphere):
        return {"kind": kind, "radius_mm": shape.radius_mm}
    if isinstance(shape, Capsule):
        return {
            "kind": kind,
            "point_a_mm": list(shape.point_a_mm),
            "point_b_mm": list(shape.point_b_mm),
            "radius_mm": shape.radius_mm,
        }
    if isinstance(shape, RoundedBox):
        return {
            "kind": kind,
            "half_extents_mm": list(shape.half_extents_mm),
            "fillet_radius_mm": shape.fillet_radius_mm,
        }
    # 法兰字段显式写出（本模块产出恒为null）：万一将来有人产了带法兰的圆柱，
    # 指纹会当场变，而不是与不带法兰的那个撞成同一份字节。
    return {
        "kind": kind,
        "radius_mm": shape.radius_mm,
        "half_width_mm": shape.half_width_mm,
        "flange_outer_radius_mm": shape.flange_outer_radius_mm,
    }


def declaration_document(parts: Sequence[GeneratedPart]) -> dict[str, Any]:
    """把产出摊成纯JSON结构——**指纹的基准**，不是落盘格式（见`MODELGEN_PROFILE`）。

    件的次序即产出次序（列表，不排序）；键序由规范化统一（`sort_keys`）。
    """

    return {
        "parts": [
            {
                "part_id": part.part_id,
                "offset_mm": list(part.offset_mm),
                "algorithm_id": part.shape.algorithm_id,
                "algorithm_version": part.shape.algorithm_version,
                "parameters": [[name, value] for name, value in part.shape.parameters],
                "shape": _shape_document(part.shape.shape),
            }
            for part in parts
        ]
    }


def declaration_bytes(parts: Sequence[GeneratedPart]) -> bytes:
    """产出的规范字节。**"同参数逐字节相同"里的那个"字节"就是这里的返回值。**"""

    return canonical_bytes(declaration_document(parts), MODELGEN_PROFILE)


def declaration_sha256(parts: Sequence[GeneratedPart]) -> str:
    """`declaration_bytes`的SHA-256，小写十六进制。跨进程比较用它，省得搬字节。"""

    return canonical_sha256(declaration_document(parts), MODELGEN_PROFILE)


__all__ = [
    "FORMER_ALGORITHM_ID",
    "FORMER_ALGORITHM_VERSION",
    "MODELGEN_PROFILE",
    "ROLLER_ALGORITHM_ID",
    "ROLLER_ALGORITHM_VERSION",
    "SPOOL_ALGORITHM_ID",
    "SPOOL_ALGORITHM_VERSION",
    "GeneratedPart",
    "ModelGenError",
    "declaration_bytes",
    "declaration_document",
    "declaration_sha256",
    "generate_former",
    "generate_roller",
    "generate_spool",
]
