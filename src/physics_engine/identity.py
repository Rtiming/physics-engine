"""稳定ID与单位量的校验工具——轴2（spec/03）的参考实现。

三件事：四段身份（WDS design/15形制）、命名空间ID（``material/...``形制）、
字段名单位后缀（后缀制是轴2规则3两种合法机制之一；另一种是Quantity类型，
那属于将来契约基座的事）。

单位表是**开放的基础集**：收录两个创始消费方实测在用的后缀。
调用方可传补充集，但不能删基础集——删除是版本变更不是参数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION_SUFFIX = re.compile(r"^v(\d{3,})$")


class IdentityError(ValueError):
    """一切身份/单位校验的失败关闭。"""


#: 两仓实测在用的单位后缀基础集（字段名以``_<unit>``结尾）。
#:
#: **``amp``是安培，不是``a``**（2026-08-17，决策0062第十节）。spec/14第一节早就
#: 登记过"基础集一个电学单位都没有——有开尔文``k``、有瓦特``w``，**没有安培**"，
#: 0048把它挂成欠账、触发条件写的是"第一个要问这卷带材的Ic是多少的消费方"。
#: 来的是另一扇门：磁粉离合器的线圈电流（`drives.MagneticParticleClutch`）。
#:
#: 用``amp``而不用SI符号``a``的理由是**实测**：仓里已有15个以``_a``结尾的名字，
#: 而其中``body_a``、``name_a``、``node_a``、``box_a``、``segment_a``、``other_a``
#: 一类是**成对命名**（``_a``/``_b``），根本没有单位。加``a``会让它们全部通过
#: 单位检查——**那等于把轴2规则3废掉**。本仓已有的单字母后缀（``n``/``m``/``s``/
#: ``w``/``k``/``c``）没有这个问题，因为没有"以它们结尾的成对命名"这个惯例。
#: ``_amp``在加入时实测**零个已有名字命中**，是一次纯扩张。
BASE_UNIT_SUFFIXES: frozenset[str] = frozenset(
    {
        "mm", "m", "mm2", "mm3",
        "s", "ms", "ns", "hz",
        "n", "nmm", "n_mm2", "nmm2", "n_s_mm",
        "kg_m3", "kg",
        "amp",
        "rad", "deg",
        "per_s", "per_mm",
        "w", "k", "c",
        "pct",
    }
)


@dataclass(frozen=True)
class CaseIdentity:
    """四段身份：``family__variant__subject__context__vNNN``。"""

    family: str
    variant: str
    subject: str
    context: str
    version: int

    def canonical(self) -> str:
        return (
            f"{self.family}__{self.variant}__{self.subject}__{self.context}"
            f"__v{self.version:03d}"
        )


def parse_case_identity(identity: str) -> CaseIdentity:
    """解析四段身份，失败关闭：段数、空段、字符集、版本后缀逐项校验。"""

    parts = identity.split("__")
    if len(parts) != 5:
        raise IdentityError(
            f"case identity must have exactly 5 double-underscore segments: {identity!r}"
        )
    *segments, version_part = parts
    for segment in segments:
        if not _SEGMENT.match(segment):
            raise IdentityError(f"invalid identity segment {segment!r} in {identity!r}")
    match = _VERSION_SUFFIX.match(version_part)
    if match is None:
        raise IdentityError(f"identity must end with vNNN (N>=3 digits): {identity!r}")
    version = int(match.group(1))
    if version < 1:
        raise IdentityError(f"identity version must be >= 1: {identity!r}")
    return CaseIdentity(
        family=segments[0],
        variant=segments[1],
        subject=segments[2],
        context=segments[3],
        version=version,
    )


def parse_namespace_id(value: str) -> tuple[str, str]:
    """解析``namespace/name``形ID（如``material/cu_...``、``scenario/spool-...``）。"""

    if value.count("/") != 1:
        raise IdentityError(f"namespace id must contain exactly one slash: {value!r}")
    namespace, name = value.split("/")
    if not _SEGMENT.match(namespace):
        raise IdentityError(f"invalid namespace {namespace!r} in {value!r}")
    if not name or not _SEGMENT.match(name.replace("__", "_")):
        raise IdentityError(f"invalid name {name!r} in {value!r}")
    return namespace, name


def has_unit_suffix(field_name: str, extra_units: frozenset[str] = frozenset()) -> bool:
    """字段名是否以已知单位后缀结尾（大小写不敏感，匹配最长后缀）。"""

    lowered = field_name.lower()
    units = BASE_UNIT_SUFFIXES | {unit.lower() for unit in extra_units}
    return any(lowered.endswith(f"_{unit}") for unit in units)


def assert_quantity_fields_have_units(
    field_names: tuple[str, ...],
    *,
    dimensionless: frozenset[str] = frozenset(),
    extra_units: frozenset[str] = frozenset(),
) -> None:
    """物理量字段必须带单位后缀（轴2规则3）。

    无量纲字段必须**显式**列入``dimensionless``——留空装有是轴2规则5
    禁止的形状。任何既不带单位又不在无量纲清单里的字段，失败关闭。
    """

    naked = [
        name
        for name in field_names
        if name not in dimensionless and not has_unit_suffix(name, extra_units)
    ]
    if naked:
        raise IdentityError(
            "quantity fields without a unit suffix and not declared dimensionless: "
            + ", ".join(sorted(naked))
        )


__all__ = [
    "BASE_UNIT_SUFFIXES",
    "CaseIdentity",
    "IdentityError",
    "assert_quantity_fields_have_units",
    "has_unit_suffix",
    "parse_case_identity",
    "parse_namespace_id",
]
