"""面清册与失败关闭的读取端——轴1（spec/02）的参考实现。

形制取自WDS `contracts/versions.py`的`assert_reader_compatible`
（未知面/未支持major/未测试minor三连拒收），加上轴1规则5要求的状态分级
（frozen/internal/draft）。规则1说"清册本身也是一个面"，所以每个清册
构造时自动登记`facet_registry`条目——它的版本就是本实现的清册形制版本。

版本字符串接受`major.minor`与`major.minor.patch`两种（WDS用三段、FTS用两段，
轴1不强制段数）；解析失败关闭：非数字、负数、空段一律拒。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FacetStatus(enum.Enum):
    """面的兼容承诺分级（轴1规则5）。"""

    #: 改动须决策记录+版本跳变+迁移说明。
    FROZEN = "frozen"
    #: 仓内消费，明示不作兼容承诺。
    INTERNAL = "internal"
    #: 设计中，不得被外部消费。
    DRAFT = "draft"


#: 清册形制自身的面条目版本（轴1规则1：清册本身也是一个面）。
_REGISTRY_FACET_NAME = "facet_registry"
_REGISTRY_FACET_MAJOR = 0
_REGISTRY_FACET_MAX_TESTED_MINOR = 1


class FacetError(ValueError):
    """一切面清册相关的失败关闭。"""


def parse_version(version: str) -> tuple[int, int]:
    """把版本字符串解析成``(major, minor)``，失败关闭。

    接受``"1.0"``与``"1.0.0"``；不接受空段、非数字、负号、多余段。
    patch段（若有）被解析校验但不参与兼容判定——轴1的兼容语义只到minor。
    """

    parts = version.split(".")
    if len(parts) not in (2, 3):
        raise FacetError(f"version must have 2 or 3 dot-separated parts: {version!r}")
    numbers = []
    for part in parts:
        if not part.isdigit():
            # isdigit拒绝空串、负号、加号、小数点与一切非数字字符。
            raise FacetError(f"version part is not a nonnegative integer: {version!r}")
        numbers.append(int(part))
    return numbers[0], numbers[1]


@dataclass(frozen=True)
class Facet:
    """清册里的一个序列化面。

    ``max_tested_minor``是读取端敢接的minor上限（WDS形制）：
    未测试过的未来minor不是"大概兼容"，是拒收。
    """

    name: str
    major: int
    max_tested_minor: int
    status: FacetStatus

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise FacetError("facet name must be a nonempty string")
        if isinstance(self.major, bool) or not isinstance(self.major, int) or self.major < 0:
            raise FacetError(f"facet major must be a nonnegative integer: {self.name}")
        if (
            isinstance(self.max_tested_minor, bool)
            or not isinstance(self.max_tested_minor, int)
            or self.max_tested_minor < 0
        ):
            raise FacetError(f"facet max_tested_minor must be a nonnegative integer: {self.name}")
        if not isinstance(self.status, FacetStatus):
            raise FacetError(f"facet status must be a FacetStatus: {self.name}")

    def accepts(self, version: str) -> bool:
        major, minor = parse_version(version)
        return major == self.major and minor <= self.max_tested_minor


class FacetRegistry:
    """一仓一份的面清册。

    构造即校验：重名拒收。清册自动含``facet_registry``自条目；
    调用方不得再登记同名面。
    """

    def __init__(self, *facets: Facet) -> None:
        entries = [
            Facet(
                name=_REGISTRY_FACET_NAME,
                major=_REGISTRY_FACET_MAJOR,
                max_tested_minor=_REGISTRY_FACET_MAX_TESTED_MINOR,
                status=FacetStatus.INTERNAL,
            ),
            *facets,
        ]
        seen: dict[str, Facet] = {}
        for entry in entries:
            if not isinstance(entry, Facet):
                raise FacetError(f"registry entries must be Facet instances: {entry!r}")
            if entry.name in seen:
                raise FacetError(f"duplicate facet name: {entry.name}")
            seen[entry.name] = entry
        self._facets = seen

    def __iter__(self):
        return iter(self._facets.values())

    def get(self, name: str) -> Facet:
        facet = self._facets.get(name)
        if facet is None:
            raise FacetError(f"unknown facet: {name}")
        return facet

    def assert_reader_compatible(self, name: str, version: str) -> None:
        """未知面、未支持major、未测试minor三连失败关闭（轴1规则3）。"""

        facet = self.get(name)
        major, _ = parse_version(version)
        if major != facet.major:
            raise FacetError(f"unsupported facet major version: {name} {version}")
        if not facet.accepts(version):
            raise FacetError(f"untested facet minor version: {name} {version}")

    def assert_externally_consumable(self, name: str) -> None:
        """draft面不得被外部消费（轴1规则5）。"""

        facet = self.get(name)
        if facet.status is FacetStatus.DRAFT:
            raise FacetError(f"draft facet must not be consumed externally: {name}")


__all__ = [
    "Facet",
    "FacetError",
    "FacetRegistry",
    "FacetStatus",
    "parse_version",
]
