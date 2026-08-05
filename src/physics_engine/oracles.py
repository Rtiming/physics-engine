"""oracle清单面——轴7规则2（spec/08）的参考实现。

规则2要求清单最低带四样东西：**生成器身份**（algorithm_id+路径）、
**逐条expected与逐条tolerances**、**数组双哈希**（语义级+raw级）、
**清单自指哈希**。形制抄FTS `oracles/m0-candidate/manifest.json`，
容差形制抄GROMACS（成对rel/abs**并写理由**），金标重生成留痕抄FEBio
`acceptChanges.py`（`regenerated_by`必须指向一份决策记录——规则5的执行体）。

本模块是**读侧**：严格加载、失败关闭、自指哈希校验、生成器SHA校验、容差比较。
写侧只给``write_manifest``一个口子，它落盘前先把文档喂回加载器——
生成器**不可能**产出一份加载器拒收的清单。

三条纪律写进类型里：

* **容差不得在测试里私改**（规则3）：``OracleCase.check``只从清单读容差，
  测试拿不到覆写口子；
* **每个量都要有理由**：``expected``的每一个键必须在``tolerances``里有一条
  带``reason``的声明，多一条少一条都拒——"判据表"不是可选文档，是加载条件；
* **不得复述oracle公式**（规则3）：expected由``generate_oracle.py``预生成进
  清单，测试只读不算。本模块不提供任何"顺手帮你算一下期望值"的函数。

零运行时依赖（AGENTS.md本仓纪律）：只用标准库+本仓``canonical``/``facets``。
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physics_engine.canonical import (
    CanonicalError,
    CanonicalProfile,
    canonical_sha256,
    strict_loads,
)
from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    ORACLE_MANIFEST_FACET,
    ORACLE_MANIFEST_VERSION,
)

#: 清单面的规范化声明。取``ensure_ascii=False``——容差**理由**是中文，
#: 逃逸成\\uXXXX后人读不了，而判据表要给人读才有价值（轴3不强制两仓同参，
#: 强制的是参数显式声明，这里就是那份声明）。
MANIFEST_PROFILE = CanonicalProfile(ensure_ascii=False, file_trailing_newline=True)

#: 自指哈希字段名。自指哈希的基准=**去掉本字段后**的规范字节。
#: 注意基准是**规范字节**不是文件字节——所以清单文件可以缩进排版给人读，
#: 而身份不受排版影响。这正是轴3规范化的用处：身份与形制解耦。落盘因此
#: 取``indent=2``（金标改一个数，``git diff``就只显示那一行——规则5要审的
#: 正是这种改动；压成一行则整份清单一起变，审无可审）。
SELF_HASH_KEY = "manifest_self_sha256"

#: 落盘排版参数。键序与规范字节同为``sort_keys``，``allow_nan``同为False——
#: 与规范化的两条硬规则一致，只有分隔符与缩进不同。
MANIFEST_INDENT = 2

#: 负载级（spec/13零之二）。案例出生时申报，与pytest marker一一对应：
#: interactive=无marker、local_batch=``batch``、serverclass=``serverclass``。
LOAD_TIERS: tuple[str, ...] = ("interactive", "local_batch", "serverclass")

#: 数组语义级哈希支持的dtype→``struct``格式（小端固定，跨平台逐位一致）。
ARRAY_DTYPES: dict[str, str] = {"float64": "<d", "int64": "<q"}

_TOP_KEYS = frozenset(
    {
        "facet", "facet_version", "case_id", "load_tier", "generator",
        "oracles", "arrays", "regenerated_by", SELF_HASH_KEY,
    }
)
_GENERATOR_KEYS = frozenset({"algorithm_id", "algorithm_version", "path_relative", "sha256"})
_ORACLE_KEYS = frozenset({"id", "inputs", "expected", "tolerances"})
_ARRAY_KEYS = frozenset({"path_relative", "dtype", "count", "logical_sha256", "raw_sha256"})
_TOLERANCE_KEYS = frozenset({"abs", "rel", "reason"})

#: 重生成金标必须指向决策记录（规则5）；只认仓内``docs/decisions/``下的文件。
DECISION_PREFIX = "docs/decisions/"


class OracleError(ValueError):
    """oracle清单的一切失败关闭。"""


def _require_hex_sha256(value: object, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise OracleError(f"{what} must be 64 lowercase hex characters: {value!r}")
    return value


def _require_relative(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value or ".." in value.split("/"):
        raise OracleError(f"{what} must be a repository-relative path: {value!r}")
    return value


def file_sha256(path: Path) -> str:
    """文件字节的SHA-256（生成器写清单时用，读侧校验时也用同一口径）。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_values(values: object) -> tuple[float, ...]:
    """嵌套列表→C序展平的浮点元组。非数值/布尔/非有限一律失败关闭。"""

    flat: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
            return
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            raise OracleError(f"array values must be finite numbers: {node!r}")
        number = float(node)
        if number != number or number in (float("inf"), float("-inf")):
            raise OracleError("array values must be finite numbers")
        flat.append(number)

    walk(values)
    return tuple(flat)


def array_logical_sha256(values: Iterable[float], *, dtype: str = "float64") -> str:
    """语义级哈希：C序、声明dtype、小端字节流的SHA-256。

    对FTS口径（``c_order_declared_dtype_bytes``）的一处收紧：先喂
    ``dtype\\0count\\0``再喂字节。纯字节流下``float64``与``int64``的同一段
    字节会撞哈希，加了帧头就不会——与``accept.py``执行树哈希加长度前缀同理。
    """

    code = ARRAY_DTYPES.get(dtype)
    if code is None:
        raise OracleError(f"unsupported array dtype: {dtype!r}")
    items = list(values)
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii") + b"\0" + str(len(items)).encode("ascii") + b"\0")
    for item in items:
        digest.update(struct.pack(code, int(item) if code == "<q" else float(item)))
    return digest.hexdigest()


def manifest_self_sha256(document: Mapping[str, Any]) -> str:
    """自指哈希：**去掉自指字段后**的规范字节的SHA-256。"""

    if not isinstance(document, Mapping):
        raise OracleError("a manifest must be a JSON object")
    basis = {key: value for key, value in document.items() if key != SELF_HASH_KEY}
    try:
        return canonical_sha256(basis, MANIFEST_PROFILE)
    except CanonicalError as error:
        raise OracleError(f"manifest is not canonicalizable: {error}") from error


@dataclass(frozen=True)
class Tolerance:
    """一条判据：成对rel/abs**并带理由**（GROMACS形制）。

    ``abs_tol=rel_tol=0``表示零容差——用于确定性整数与哈希这类逐位判据，
    理由字段照样必填（"为什么可以是零容差"和"为什么是1e-9"一样需要交代）。
    """

    abs_tol: float
    rel_tol: float
    reason: str

    def exceeded_by(self, actual: float, expected: float) -> float:
        """超出量：``|a−e| − (abs + rel·|e|)``，``<= 0``即通过。"""

        return abs(actual - expected) - (self.abs_tol + self.rel_tol * abs(expected))

    def holds(self, actual: float, expected: float) -> bool:
        return self.exceeded_by(actual, expected) <= 0.0


@dataclass(frozen=True)
class GeneratorIdentity:
    """生成器身份（规则2）：算法ID+版本+仓内路径+**脚本自身的SHA-256**。

    SHA钉死脚本字节：改了生成器却没重生成金标，读侧当场红——这正是
    "金标与生成它的代码必须同批变"的执行体。
    """

    algorithm_id: str
    algorithm_version: str
    path_relative: str
    sha256: str

    def verify(self, root: Path) -> None:
        path = root / self.path_relative
        if not path.is_file():
            raise OracleError(f"generator script is missing: {self.path_relative}")
        measured = file_sha256(path)
        if measured != self.sha256:
            raise OracleError(
                f"generator script changed without regenerating the oracle: "
                f"{self.path_relative} measured={measured} declared={self.sha256}"
            )


@dataclass(frozen=True)
class ArrayDigest:
    """数组双哈希（规则2）：语义级（值+dtype）与raw级（落盘字节）各一份。

    两者分工：raw级抓"文件被动过"，语义级抓"换了存储形式但值该不变"。
    只留raw级的话，换个缩进就红；只留语义级的话，字节被改成另一种编码
    却算出同样的值时不红。
    """

    name: str
    path_relative: str
    dtype: str
    count: int
    logical_sha256: str
    raw_sha256: str

    def verify_bytes(self, raw: bytes) -> None:
        measured = hashlib.sha256(raw).hexdigest()
        if measured != self.raw_sha256:
            raise OracleError(
                f"array {self.name!r} raw bytes changed: measured={measured} declared={self.raw_sha256}"
            )

    def verify_values(self, values: Sequence[float]) -> None:
        if len(values) != self.count:
            raise OracleError(
                f"array {self.name!r} has {len(values)} values, manifest declares {self.count}"
            )
        measured = array_logical_sha256(values, dtype=self.dtype)
        if measured != self.logical_sha256:
            raise OracleError(
                f"array {self.name!r} logical hash changed: "
                f"measured={measured} declared={self.logical_sha256}"
            )


@dataclass(frozen=True)
class OracleCase:
    """一条oracle：输入、冻结的expected、逐量容差。"""

    id: str
    inputs: Mapping[str, Any]
    expected: Mapping[str, Any]
    tolerances: Mapping[str, Tolerance]

    def tolerance(self, quantity: str) -> Tolerance:
        tolerance = self.tolerances.get(quantity)
        if tolerance is None:
            raise OracleError(f"{self.id}: no tolerance declared for {quantity!r}")
        return tolerance

    def check(self, quantity: str, actual: Any) -> None:
        """拿生产内核算出的``actual``与冻结expected按清单容差比对，不过即炸。

        标量按容差比；序列逐元素同容差比（长度不符即炸）；
        非数值（bool/str）要求逐位相等，且其容差必须声明为零。
        """

        if quantity not in self.expected:
            raise OracleError(f"{self.id}: {quantity!r} is not an expected quantity")
        expected = self.expected[quantity]
        tolerance = self.tolerance(quantity)
        for path, want, got in _pair_up(self.id, quantity, expected, actual):
            if isinstance(want, bool) or isinstance(want, str) or want is None:
                if want != got:
                    raise OracleError(f"{self.id}: {path} expected {want!r}, got {got!r}")
                continue
            if isinstance(got, bool) or not isinstance(got, (int, float)):
                raise OracleError(f"{self.id}: {path} expected a number, got {got!r}")
            if not tolerance.holds(float(got), float(want)):
                raise OracleError(
                    f"{self.id}: {path} expected {want!r}, got {got!r} — "
                    f"exceeds abs={tolerance.abs_tol!r} rel={tolerance.rel_tol!r} by "
                    f"{tolerance.exceeded_by(float(got), float(want))!r} ({tolerance.reason})"
                )

    def check_all(self, actuals: Mapping[str, Any]) -> None:
        """全部expected量一次比对；漏算一个量也是失败（不许挑着比）。"""

        missing = set(self.expected) - set(actuals)
        if missing:
            raise OracleError(f"{self.id}: no measurement supplied for {sorted(missing)}")
        for quantity in sorted(self.expected):
            self.check(quantity, actuals[quantity])


def _pair_up(
    oracle_id: str, quantity: str, expected: Any, actual: Any
) -> list[tuple[str, Any, Any]]:
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise OracleError(
                f"{oracle_id}: {quantity} expects {len(expected)} components, got {actual!r}"
            )
        pairs: list[tuple[str, Any, Any]] = []
        for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            pairs.extend(_pair_up(oracle_id, f"{quantity}[{index}]", want, got))
        return pairs
    return [(quantity, expected, actual)]


@dataclass(frozen=True)
class OracleManifest:
    """一份加载并校验过的oracle清单。"""

    case_id: str
    load_tier: str
    generator: GeneratorIdentity
    oracles: tuple[OracleCase, ...]
    arrays: Mapping[str, ArrayDigest]
    regenerated_by: str | None
    self_sha256: str
    document: Mapping[str, Any]

    def oracle(self, oracle_id: str) -> OracleCase:
        for case in self.oracles:
            if case.id == oracle_id:
                return case
        raise OracleError(f"{self.case_id}: unknown oracle id {oracle_id!r}")

    def array(self, name: str) -> ArrayDigest:
        digest = self.arrays.get(name)
        if digest is None:
            raise OracleError(f"{self.case_id}: unknown array {name!r}")
        return digest

    def verify_generator(self, root: Path) -> None:
        self.generator.verify(root)

    def verify_regeneration(self, root: Path) -> None:
        """规则5：重生成过的金标必须指向一份**存在的**决策记录。"""

        if self.regenerated_by is None:
            return
        if not self.regenerated_by.startswith(DECISION_PREFIX):
            raise OracleError(
                f"regenerated_by must point into {DECISION_PREFIX}: {self.regenerated_by!r}"
            )
        if not (root / self.regenerated_by).is_file():
            raise OracleError(f"regenerated_by decision record is missing: {self.regenerated_by}")

    def load_array(self, name: str, root: Path) -> Mapping[str, Any]:
        """按双哈希校验后返回数组文档（``{"dtype","values",…}``）。

        raw级先校验字节、再解析、再按声明dtype与count校验语义级哈希——
        任何一层不符都在拿到值之前失败关闭。
        """

        digest = self.array(name)
        path = root / digest.path_relative
        if not path.is_file():
            raise OracleError(f"array file is missing: {digest.path_relative}")
        raw = path.read_bytes()
        digest.verify_bytes(raw)
        try:
            document = strict_loads(raw)
        except CanonicalError as error:
            raise OracleError(f"array {name!r} is not strict JSON: {error}") from error
        if not isinstance(document, Mapping) or "values" not in document:
            raise OracleError(f"array {name!r} must be an object carrying 'values'")
        if document.get("dtype") != digest.dtype:
            raise OracleError(
                f"array {name!r} declares dtype {document.get('dtype')!r}, "
                f"manifest says {digest.dtype!r}"
            )
        digest.verify_values(flatten_values(document["values"]))
        return document


def _parse_tolerance(oracle_id: str, quantity: str, raw: Any) -> Tolerance:
    if not isinstance(raw, Mapping) or set(raw) != _TOLERANCE_KEYS:
        raise OracleError(
            f"{oracle_id}: tolerance for {quantity!r} must declare exactly {sorted(_TOLERANCE_KEYS)}"
        )
    values = []
    for key in ("abs", "rel"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
            raise OracleError(f"{oracle_id}: tolerance {key} for {quantity!r} must be >= 0")
        values.append(float(value))
    reason = raw["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise OracleError(
            f"{oracle_id}: tolerance for {quantity!r} needs a reason — "
            "判据表的第三列不是装饰（plans/02第四节案例页六必填字段之三）"
        )
    return Tolerance(abs_tol=values[0], rel_tol=values[1], reason=reason)


def _parse_oracle(raw: Any) -> OracleCase:
    if not isinstance(raw, Mapping) or set(raw) != _ORACLE_KEYS:
        raise OracleError(f"each oracle must declare exactly {sorted(_ORACLE_KEYS)}: {raw!r}")
    oracle_id = raw["id"]
    if not isinstance(oracle_id, str) or not oracle_id.startswith("oracle:"):
        raise OracleError(f"oracle id must carry the 'oracle:' prefix: {oracle_id!r}")
    inputs, expected, tolerances = raw["inputs"], raw["expected"], raw["tolerances"]
    for name, value in (("inputs", inputs), ("expected", expected), ("tolerances", tolerances)):
        if not isinstance(value, Mapping):
            raise OracleError(f"{oracle_id}: {name} must be an object")
    if not expected:
        raise OracleError(f"{oracle_id}: expected must not be empty")
    if set(tolerances) != set(expected):
        raise OracleError(
            f"{oracle_id}: every expected quantity needs exactly one tolerance — "
            f"expected={sorted(expected)} tolerances={sorted(tolerances)}"
        )
    parsed = {
        quantity: _parse_tolerance(oracle_id, quantity, tolerances[quantity])
        for quantity in tolerances
    }
    for quantity, value in expected.items():
        tolerance = parsed[quantity]
        if _is_exact_kind(value) and (tolerance.abs_tol or tolerance.rel_tol):
            raise OracleError(
                f"{oracle_id}: {quantity!r} is compared bit-for-bit; its tolerance must be zero"
            )
    return OracleCase(id=oracle_id, inputs=inputs, expected=expected, tolerances=parsed)


def _is_exact_kind(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_is_exact_kind(item) for item in value)
    return isinstance(value, bool) or isinstance(value, str) or value is None


def _parse_generator(raw: Any) -> GeneratorIdentity:
    if not isinstance(raw, Mapping) or set(raw) != _GENERATOR_KEYS:
        raise OracleError(f"generator must declare exactly {sorted(_GENERATOR_KEYS)}")
    algorithm_id = raw["algorithm_id"]
    if not isinstance(algorithm_id, str) or not algorithm_id.startswith("algorithm:"):
        raise OracleError(f"generator identity must carry the 'algorithm:' prefix: {algorithm_id!r}")
    version = raw["algorithm_version"]
    if not isinstance(version, str) or not version:
        raise OracleError("generator requires a version")
    return GeneratorIdentity(
        algorithm_id=algorithm_id,
        algorithm_version=version,
        path_relative=_require_relative(raw["path_relative"], "generator path_relative"),
        sha256=_require_hex_sha256(raw["sha256"], "generator sha256"),
    )


def _parse_array(name: str, raw: Any) -> ArrayDigest:
    if not isinstance(raw, Mapping) or set(raw) != _ARRAY_KEYS:
        raise OracleError(f"array {name!r} must declare exactly {sorted(_ARRAY_KEYS)}")
    dtype = raw["dtype"]
    if dtype not in ARRAY_DTYPES:
        raise OracleError(f"array {name!r} has unsupported dtype: {dtype!r}")
    count = raw["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise OracleError(f"array {name!r} count must be a nonnegative integer")
    return ArrayDigest(
        name=name,
        path_relative=_require_relative(raw["path_relative"], f"array {name!r} path_relative"),
        dtype=dtype,
        count=count,
        logical_sha256=_require_hex_sha256(raw["logical_sha256"], f"array {name!r} logical_sha256"),
        raw_sha256=_require_hex_sha256(raw["raw_sha256"], f"array {name!r} raw_sha256"),
    )


def parse_manifest(document: Any) -> OracleManifest:
    """把已解析的文档校验成``OracleManifest``。失败关闭，绝不"尽力而为"。"""

    if not isinstance(document, Mapping):
        raise OracleError("a manifest must be a JSON object")
    unknown = set(document) - _TOP_KEYS
    if unknown:
        raise OracleError(f"unknown manifest keys: {sorted(unknown)}")
    missing = _TOP_KEYS - set(document)
    if missing:
        raise OracleError(f"manifest is missing required keys: {sorted(missing)}")
    if document["facet"] != ORACLE_MANIFEST_FACET:
        raise OracleError(f"facet must be {ORACLE_MANIFEST_FACET!r}: {document['facet']!r}")
    ENGINE_REGISTRY.assert_reader_compatible(ORACLE_MANIFEST_FACET, str(document["facet_version"]))
    case_id = document["case_id"]
    if not isinstance(case_id, str) or not case_id.startswith("case/"):
        raise OracleError(f"case_id must be namespaced like 'case/...': {case_id!r}")
    load_tier = document["load_tier"]
    if load_tier not in LOAD_TIERS:
        raise OracleError(f"load_tier must be one of {list(LOAD_TIERS)}: {load_tier!r}")
    raw_oracles = document["oracles"]
    if not isinstance(raw_oracles, list) or not raw_oracles:
        raise OracleError("a manifest requires a nonempty 'oracles' list")
    oracles = tuple(_parse_oracle(entry) for entry in raw_oracles)
    identifiers = [case.id for case in oracles]
    if len(set(identifiers)) != len(identifiers):
        raise OracleError("duplicate oracle id in manifest")
    raw_arrays = document["arrays"]
    if not isinstance(raw_arrays, Mapping):
        raise OracleError("'arrays' must be an object (use {} when the case has no arrays)")
    arrays = {name: _parse_array(name, raw) for name, raw in raw_arrays.items()}
    regenerated_by = document["regenerated_by"]
    if regenerated_by is not None and not isinstance(regenerated_by, str):
        raise OracleError("regenerated_by must be null or a decision record path")
    declared = _require_hex_sha256(document[SELF_HASH_KEY], "manifest_self_sha256")
    measured = manifest_self_sha256(document)
    if measured != declared:
        raise OracleError(
            f"manifest self hash mismatch: measured={measured} declared={declared} — "
            "清单被改过而没有重新生成"
        )
    return OracleManifest(
        case_id=case_id,
        load_tier=load_tier,
        generator=_parse_generator(document["generator"]),
        oracles=oracles,
        arrays=arrays,
        regenerated_by=regenerated_by,
        self_sha256=declared,
        document=document,
    )


def load_manifest(path: Path, *, root: Path | None = None) -> OracleManifest:
    """从文件严格加载一份清单。

    给了``root``就连生成器SHA与``regenerated_by``一并校验——conformance测试
    该走这条路：清单、生成它的脚本、重生成留痕三样一次性全查。
    """

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise OracleError(f"cannot read oracle manifest {path}: {error}") from error
    try:
        document = strict_loads(payload)
    except CanonicalError as error:
        raise OracleError(f"oracle manifest {path} is not strict JSON: {error}") from error
    manifest = parse_manifest(document)
    if root is not None:
        manifest.verify_generator(root)
        manifest.verify_regeneration(root)
    return manifest


def write_manifest(path: Path, document: Mapping[str, Any], *, root: Path) -> bytes:
    """生成器落盘口：补自指哈希→**喂回加载器**→缩进落盘。

    先过一遍加载器再落盘，是为了让"生成器能产出加载器拒收的清单"这件事
    不可能发生。身份是规范字节的哈希，排版不参与——见``SELF_HASH_KEY``。
    返回写入的字节。
    """

    payload = dict(document)
    payload[SELF_HASH_KEY] = manifest_self_sha256(payload)
    parse_manifest(payload).verify_generator(root)
    try:
        text = json.dumps(
            payload,
            ensure_ascii=MANIFEST_PROFILE.ensure_ascii,
            allow_nan=False,
            sort_keys=True,
            indent=MANIFEST_INDENT,
        )
    except ValueError as error:
        raise OracleError(f"manifest is not serialisable: {error}") from error
    data = text.encode(MANIFEST_PROFILE.encoding) + b"\n"
    path.write_bytes(data)
    return data


__all__ = [
    "ARRAY_DTYPES",
    "DECISION_PREFIX",
    "LOAD_TIERS",
    "MANIFEST_INDENT",
    "MANIFEST_PROFILE",
    "ORACLE_MANIFEST_FACET",
    "ORACLE_MANIFEST_VERSION",
    "SELF_HASH_KEY",
    "ArrayDigest",
    "GeneratorIdentity",
    "OracleCase",
    "OracleError",
    "OracleManifest",
    "Tolerance",
    "array_logical_sha256",
    "file_sha256",
    "flatten_values",
    "load_manifest",
    "manifest_self_sha256",
    "parse_manifest",
    "write_manifest",
]
