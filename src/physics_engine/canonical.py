"""规范化JSON的读写对——轴3规则2（spec/04）的参考实现。

轴3不强制两仓同参（实测分歧：WDS ``ensure_ascii=False``、FTS ``True``），
强制的是**参数显式声明**与**读侧对称防御**。所以这里的声明就是一个不可变
的``CanonicalProfile``对象：面契约引用哪个profile，哪个profile就是它的
规范化声明，不存在"隐式默认"。

两条不可配置的硬规则（是规则不是参数）：

* ``allow_nan``永远False——NaN/Inf进不了规范字节；
* 键排序永远开——无序对象没有规范形。

尾换行是**文件层**约定（轴3实测：WDS文件体带``\\n``而指纹不带），
所以给了两个动词：``canonical_bytes``（指纹用，不带）与
``canonical_file_bytes``（落盘用，按profile决定带不带）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


class CanonicalError(ValueError):
    """规范化读写的一切失败关闭。"""


@dataclass(frozen=True)
class CanonicalProfile:
    """一份显式的规范化参数声明。

    ``separators``固定最紧凑形；可声明的只有ASCII策略与文件尾换行——
    这正是两个创始消费方实测分歧所在的两个自由度。
    """

    ensure_ascii: bool
    file_trailing_newline: bool = True

    @property
    def encoding(self) -> str:
        return "ascii" if self.ensure_ascii else "utf-8"


#: 两个创始消费方的实测参数，作为现成声明供引用。
WDS_PROFILE = CanonicalProfile(ensure_ascii=False, file_trailing_newline=True)
FTS_PROFILE = CanonicalProfile(ensure_ascii=True, file_trailing_newline=True)


def canonical_bytes(document: Any, profile: CanonicalProfile) -> bytes:
    """规范字节（指纹口径，不带尾换行）。"""

    try:
        text = json.dumps(
            document,
            ensure_ascii=profile.ensure_ascii,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except ValueError as error:
        raise CanonicalError(f"document is not canonicalizable: {error}") from error
    return text.encode(profile.encoding)


def canonical_file_bytes(document: Any, profile: CanonicalProfile) -> bytes:
    """落盘字节（按profile决定是否带单个尾换行）。"""

    body = canonical_bytes(document, profile)
    return body + b"\n" if profile.file_trailing_newline else body


def canonical_sha256(document: Any, profile: CanonicalProfile) -> str:
    """规范字节的SHA-256，小写十六进制，不带前缀。"""

    return hashlib.sha256(canonical_bytes(document, profile)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(constant: str) -> Any:
    raise CanonicalError(f"nonfinite JSON constant is forbidden: {constant}")


def strict_loads(data: bytes | str) -> Any:
    """严格解析：拒重复键、拒NaN/Infinity字面量（轴3读侧对称防御）。

    只防**字面量层**；数值语义层的有限性检查（如数组内容）属各面契约，
    不在本函数越权代做。
    """

    if isinstance(data, bytes):
        data = data.decode("utf-8")
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except CanonicalError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CanonicalError(f"not valid JSON: {error}") from error


__all__ = [
    "CanonicalError",
    "CanonicalProfile",
    "FTS_PROFILE",
    "WDS_PROFILE",
    "canonical_bytes",
    "canonical_file_bytes",
    "canonical_sha256",
    "strict_loads",
]
