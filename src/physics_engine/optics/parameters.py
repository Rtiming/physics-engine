"""光学域怎么从材料记录取参数——**用现成的域间接口，不另造材料通道**。

spec/14第一节把材料记录定成"一份记录聚合多域字段，域是属性的标签不是记录的分块"，
并给了`properties_for_domain`这个访问器。它**就是**域间接口：光学域读到的
`thickness_mm`与力学域读到的是同一个值同一条证据，不是两份会各自漂移的拷贝。

所以本模块不到200行也不该到：它只做三件本域特有的事——

1. **钉住长度制**。本域按米（`OPTICS_LENGTH_UNIT`）。毫米制的记录要显式
   `converted_to("m")`，`properties_for_domain`的`expect_length_unit`安全带
   替我们把这件事变成失败关闭而不是差1000倍的静默错（spec/14第五节）；
2. **把"未测量"挡在使用点**。spec/14规则4：`unset`的属性其值是`None`不是0。
   `properties_for_domain`如实把`None`交出来——**在这里把它变成拒跑**，
   而不是让一个`None`流进公式变成`TypeError`或（更糟）被`or 0.0`吃掉；
3. **按域申报证据分级**。记录整体的可信度没有意义，只有光学域用到的那些字段的
   可信度有意义（spec/14第四节）。

本模块**不**做：材料记录的加载、校验、内容寻址、换制——那些全在`materials.py`，
光学域只是它的读者。
"""

from __future__ import annotations

from physics_engine.materials import MaterialRecord
from physics_engine.optics.errors import OpticsError

#: 光学域的长度制。FTS一侧全用米，本域随之。
#: **不是风格选择**：`materials.properties_for_domain`的`expect_length_unit`
#: 只有在调用方真的声明了自己要哪一制时才起作用，声明得含糊就等于没有安全带。
OPTICS_LENGTH_UNIT: str = "m"

#: 光学域的域名，`materials.MATERIAL_DOMAINS`里的那一个。
OPTICS_DOMAIN: str = "optics"


def optics_parameters(record: MaterialRecord) -> dict[str, float | None]:
    """材料记录里服务光学域的字段。未测量的字段照实给`None`。

    长度制对不上即失败关闭——调用方要么给米制记录，要么显式
    `record.converted_to("m")`。**不在这里替他换**：换制换出来的是另一条记录
    （字节变了、内容地址变了），那件事必须显式发生在调用方的代码里。
    """

    return record.properties_for_domain(
        OPTICS_DOMAIN, expect_length_unit=OPTICS_LENGTH_UNIT
    )


def optics_evidence_grade(record: MaterialRecord) -> str:
    """光学域用到的那批字段里最弱的证据分级。

    用它申报"这次光学计算的输入有多可信"。记录整体的最弱分级不作数——
    一条力学字段是`unset`不该拖累光学的结论（spec/14第四节）。
    """

    return record.weakest_grade(OPTICS_DOMAIN)


def require_optics_parameter(record: MaterialRecord, field_name: str) -> float:
    """取一个光学参数，`unset`即拒跑。

    这是spec/12第2.3节那条力学法条的光学版：分级是`unset`时求解器应当拒跑
    而不是用零占位。失败信息里带上证据分级与方法说明，
    因为"为什么没有这个数"比"没有这个数"更值钱。
    """

    values = optics_parameters(record)
    if field_name not in values:
        available = ", ".join(sorted(values)) or "（无）"
        raise OpticsError(
            f"{record.material_id!r}的光学域没有字段{field_name!r}；"
            f"该记录光学域可用字段：{available}"
        )
    value = values[field_name]
    if value is None:
        evidence = record.evidence_for(field_name)
        raise OpticsError(
            f"{record.material_id!r}的{field_name!r}分级为{evidence.grade!r}（未测量），"
            f"没有值可用——拒跑，不用零占位。证据说明：{evidence.method}"
        )
    return value


__all__ = [
    "OPTICS_DOMAIN",
    "OPTICS_LENGTH_UNIT",
    "optics_evidence_grade",
    "optics_parameters",
    "require_optics_parameter",
]
