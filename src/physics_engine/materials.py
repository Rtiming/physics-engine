"""材料记录——一份记录聚合多域字段（spec/14的首个实现，v0）。

蒸馏来源（完整对照见spec/14第三节与decisions/0023）：

* **WDS `config/materials/`**：SHA-256锁定+单位后缀+适用域声明三件套，
  参数级`EvidenceRecord`（分级+方法+源工件哈希），命名即身份、只增不改；
* **FTS `contracts/common.py`**：`Quantity`的两条硬语义——`unset`不得带值、
  非`unset`必须带值。跨边界形制取严的这一侧，不取WDS的占位零。

本仓增量只有两处，都是两个消费方合起来才暴露的问题：**域是属性的标签**
（`thickness_mm`同时进力学与光学，分块形制表达不了），以及**长度制边界**
（WDS mm、FTS m，混用即静默1000倍；跨制取值必须显式`converted_to`，
换算表只覆盖纯长度量纲，复合量纲没登记就拒——登记缺口，不猜因子）。

零运行时依赖：只用标准库。红例集见spec/14第六节。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any

from physics_engine.canonical import WDS_PROFILE, canonical_sha256, strict_loads
from physics_engine.engine_facets import (
    ENGINE_REGISTRY,
    MATERIAL_RECORD_FACET,
    MATERIAL_RECORD_VERSION,
)
from physics_engine.identity import (
    BASE_UNIT_SUFFIXES,
    IdentityError,
    assert_quantity_fields_have_units,
    parse_namespace_id,
)


class MaterialError(ValueError):
    """材料记录的一切失败关闭。"""


#: 本面的规范化声明（轴3规则2）：随WDS参数——记录带中文显示名与限制说明。
MATERIAL_CANONICAL_PROFILE = WDS_PROFILE

#: 统一证据分级：WDS六档与FTS五档的映射结果（spec/14第三节的表）。
#: FTS的`illustrative`未采纳——它与`benchmark_constant`可信度相反，合并会稀释两者。
EVIDENCE_GRADES: tuple[str, ...] = (
    "measured", "calibrated", "derived", "benchmark_constant",
    "manufacturer", "estimated", "unset",
)

#: 强弱序（越大越弱），形制取WDS `EvidenceMap.weakest_status`。
_GRADE_WEAKNESS: dict[str, int] = {
    "measured": 0, "calibrated": 0, "benchmark_constant": 1, "derived": 1,
    "manufacturer": 2, "estimated": 3, "unset": 4,
}

#: 可以没有源工件哈希的分级；其余分级声称有外部出处，就必须锁得住。
_GRADES_WITHOUT_SOURCE: frozenset[str] = frozenset({"estimated", "unset"})

#: 域名集合：spec/01模块地图的物理域圈 + appearance。
MATERIAL_DOMAINS: tuple[str, ...] = ("mechanics", "optics", "thermal", "em", "appearance")

#: 长度制：两个创始消费方各一个（WDS mm、FTS m）。
LENGTH_UNITS: tuple[str, ...] = ("mm", "m")

#: 米制要用、轴2基础集里没有的后缀（identity.py明写"调用方可传补充集"；基础集一个不删）。
METRE_SYSTEM_UNITS: frozenset[str] = frozenset({"m2", "m3", "per_m"})

_ALL_UNITS: frozenset[str] = BASE_UNIT_SUFFIXES | METRE_SYSTEM_UNITS

#: 各长度制**专属**的后缀。同一条记录里出现另一制的后缀即拒。
_SYSTEM_SUFFIXES: dict[str, frozenset[str]] = {
    "mm": frozenset({"mm", "mm2", "mm3", "per_mm", "nmm", "nmm2", "n_mm2", "n_s_mm"}),
    "m": frozenset({"m", "m2", "m3", "per_m"}),
}

#: 两制通用、写法本身已钉死量纲的SI复合后缀——WDS的mm制记录实测就带
#: `density_kg_m3`，它不是换算风险点，不进边界检查。
_CROSS_SYSTEM_SUFFIXES: frozenset[str] = frozenset({"kg_m3"})

#: 纯长度量纲的换算表：`(源后缀, 目标制) -> (目标后缀, 因子)`。
#: 复合量纲（`n_mm2`等）**故意留空**：换算它要先决定是不是MPa，那是消费方
#: 采纳声明里的事，本面碰上就报错指路，不猜。
_LENGTH_CONVERSIONS: dict[tuple[str, str], tuple[str, float]] = {
    ("mm", "m"): ("m", 1.0e-3), ("mm2", "m"): ("m2", 1.0e-6),
    ("mm3", "m"): ("m3", 1.0e-9), ("per_mm", "m"): ("per_m", 1.0e3),
    ("m", "mm"): ("mm", 1.0e3), ("m2", "mm"): ("mm2", 1.0e6),
    ("m3", "mm"): ("mm3", 1.0e9), ("per_m", "mm"): ("per_mm", 1.0e-3),
}

_TOP_KEYS = frozenset({
    "contract_type", "contract_version", "material_id", "length_unit",
    "applicable_domains", "dimensionless", "properties", "appearance", "content_sha256",
})
_PROPERTY_KEYS = frozenset({"name", "value", "domains", "evidence"})
_EVIDENCE_KEYS = frozenset({"grade", "evidence_id", "method", "source_sha256"})
_APPEARANCE_KEYS = frozenset({"asset_id", "path_relative", "sha256"})


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MaterialError(f"{name} must be 64 lowercase hex characters")
    return value


def _require_namespace(value: object, namespace: str, name: str) -> str:
    if not isinstance(value, str):
        raise MaterialError(f"{name} must be a string")
    try:
        parsed, _ = parse_namespace_id(value)
    except IdentityError as error:
        raise MaterialError(f"{name} is not a valid namespaced id: {error}") from error
    if parsed != namespace:
        raise MaterialError(f"{name} must live in the {namespace!r} namespace: {value!r}")
    return value


def unit_suffix_of(field_name: str) -> str | None:
    """字段名的单位后缀（取最长匹配）；没有已知后缀返回``None``。

    **分母陷阱（2026-08-05修，research/07审计发现）**：单纯的最长匹配会把
    ``current_density_a_per_m2``的后缀判成``m2``——那是个**面积**单位，
    可它在这个名字里站在**分母**上。于是换算走了面积的``×1e6``，
    而``A/m² → A/mm²``的正确因子是``×1e-6``：**方向反了，差1e12，且不报错**。

    修法不是去补一张更大的换算表（那要为每个复合量纲猜因子，正是spec/14
    第五节禁止的），而是**让最长匹配不许跨过``per_``**：名字里出现``_per_{u}``时，
    后缀就是``per_{u}``整体。``per_{u}``没登记，换算表查不到，
    于是`converted_to`里那条早就写好的"复合量纲无登记换算即拒"当场触发——
    **那条错误消息在此之前一次都没触发过**。
    """

    lowered = field_name.lower()
    best: str | None = None
    for unit in _ALL_UNITS:
        if lowered.endswith(f"_{unit}") and (best is None or len(unit) > len(best)):
            best = unit
    if best is not None and not best.startswith("per_"):
        # 匹配到的裸单位是不是站在`per_`后面？是就把整段当后缀，交给换算表拒收。
        if lowered.endswith(f"_per_{best}"):
            return f"per_{best}"
    return best


@dataclass(frozen=True)
class EvidenceRef:
    """一条参数级证据——不是展示用徽章（WDS `EvidenceRecord`的最小骨架）。

    ``method``必填非空：轴2规则5的"无出处可追也必须显式写出"落在这里。
    """

    grade: str
    evidence_id: str
    method: str
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.grade not in EVIDENCE_GRADES:
            raise MaterialError(
                f"evidence grade must be one of {list(EVIDENCE_GRADES)}: {self.grade!r}"
            )
        _require_namespace(self.evidence_id, "evidence", "evidence_id")
        if not isinstance(self.method, str) or not self.method.strip():
            raise MaterialError(
                f"evidence {self.evidence_id!r} needs a nonempty method — "
                "'no traceable source' must be written out"
            )
        if self.source_sha256 is None:
            if self.grade not in _GRADES_WITHOUT_SOURCE:
                raise MaterialError(
                    f"grade {self.grade!r} claims an external source, so "
                    "source_sha256 is mandatory (axis 3 rule 1)"
                )
        else:
            _require_sha256(self.source_sha256, "source_sha256")

    def weakness(self) -> int:
        return _GRADE_WEAKNESS[self.grade]

    def to_document(self) -> dict[str, Any]:
        return {
            "grade": self.grade, "evidence_id": self.evidence_id,
            "method": self.method, "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class MaterialProperty:
    """带单位、带域标签、带证据的物理量。

    ``domains``是**属性的标签**而不是记录的分块——``thickness_mm``同时进力学
    与光学是常态，分块形制下它得写两遍，两遍就会漂移。
    """

    name: str
    value: float | None
    domains: tuple[str, ...]
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise MaterialError("property name must be a nonempty string")
        if not self.domains:
            raise MaterialError(f"property {self.name!r} must declare at least one domain")
        unknown = sorted(set(self.domains) - set(MATERIAL_DOMAINS))
        if unknown:
            raise MaterialError(f"property {self.name!r} has unknown domains {unknown}")
        if len(set(self.domains)) != len(self.domains):
            raise MaterialError(f"property {self.name!r} lists a domain twice")
        if self.evidence.grade == "unset":
            if self.value is not None:
                raise MaterialError(
                    f"property {self.name!r} is graded 'unset' but carries a value — "
                    "an unset quantity is None, not a placeholder number"
                )
        elif self.value is None:
            raise MaterialError(
                f"property {self.name!r} has no value but is graded "
                f"{self.evidence.grade!r}; only 'unset' may be valueless"
            )
        if self.value is not None and not math.isfinite(self.value):
            raise MaterialError(f"property {self.name!r} must be finite")

    def to_document(self) -> dict[str, Any]:
        return {
            "name": self.name, "value": self.value,
            "domains": list(self.domains), "evidence": self.evidence.to_document(),
        }


@dataclass(frozen=True)
class AppearanceRef:
    """外观资产**引用**——渲染参数不进材料面（spec/01：外观是资产属性不是渲染实现）。"""

    asset_id: str
    path_relative: str
    sha256: str

    def __post_init__(self) -> None:
        _require_namespace(self.asset_id, "appearance", "asset_id")
        if (
            not isinstance(self.path_relative, str)
            or not self.path_relative
            or self.path_relative.startswith("/")
            or "\\" in self.path_relative
        ):
            raise MaterialError("path_relative must be a nonempty relative path")
        _require_sha256(self.sha256, "sha256")

    def to_document(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "path_relative": self.path_relative,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MaterialRecord:
    """一份聚合多域字段的材料记录（面``engine_material_record``）。"""

    material_id: str
    applicable_domains: tuple[str, ...]
    properties: tuple[MaterialProperty, ...]
    length_unit: str = "mm"
    facet_version: str = MATERIAL_RECORD_VERSION
    appearance: AppearanceRef | None = None
    dimensionless: frozenset[str] = frozenset()
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_namespace(self.material_id, "material", "material_id")
        ENGINE_REGISTRY.assert_reader_compatible(MATERIAL_RECORD_FACET, self.facet_version)
        if self.length_unit not in LENGTH_UNITS:
            raise MaterialError(
                f"length_unit must be one of {list(LENGTH_UNITS)}: {self.length_unit!r}"
            )
        self._check_domains()
        self._check_properties()
        self._check_length_system()
        if self.content_sha256 is not None:
            _require_sha256(self.content_sha256, "content_sha256")
            if self.content_sha256 != self.content_address():
                raise MaterialError(
                    "content_sha256 does not match the record's canonical bytes — "
                    "the self-referential address failed its own check (axis 3 rule 4)"
                )

    # --- 校验 -----------------------------------------------------------

    def _check_domains(self) -> None:
        if not self.applicable_domains:
            raise MaterialError(f"{self.material_id!r} must declare applicable_domains")
        unknown = sorted(set(self.applicable_domains) - set(MATERIAL_DOMAINS))
        if unknown:
            raise MaterialError(
                f"unknown applicable_domains {unknown}; known are {list(MATERIAL_DOMAINS)}"
            )
        if len(set(self.applicable_domains)) != len(self.applicable_domains):
            raise MaterialError("applicable_domains lists a domain twice")

    def _check_properties(self) -> None:
        if not self.properties:
            raise MaterialError(f"{self.material_id!r} carries no properties")
        names = [prop.name for prop in self.properties]
        if len(set(names)) != len(names):
            raise MaterialError("a property name appears twice in the record")
        declared = set(self.applicable_domains)
        for prop in self.properties:
            escaping = sorted(set(prop.domains) - declared)
            if escaping:
                raise MaterialError(
                    f"property {prop.name!r} claims domains {escaping} that the record "
                    f"does not declare in applicable_domains {list(self.applicable_domains)}"
                )
        try:
            assert_quantity_fields_have_units(
                tuple(names),
                dimensionless=frozenset(self.dimensionless),
                extra_units=METRE_SYSTEM_UNITS,
            )
        except IdentityError as error:
            raise MaterialError(f"{self.material_id!r}: {error}") from error
        stray = sorted(set(self.dimensionless) - set(names))
        if stray:
            raise MaterialError(f"dimensionless declares fields the record does not carry: {stray}")

    def _check_length_system(self) -> None:
        foreign = next(unit for unit in LENGTH_UNITS if unit != self.length_unit)
        rejected = _SYSTEM_SUFFIXES[foreign] - _CROSS_SYSTEM_SUFFIXES
        offenders = sorted(
            prop.name for prop in self.properties if unit_suffix_of(prop.name) in rejected
        )
        if offenders:
            raise MaterialError(
                f"record declares length_unit={self.length_unit!r} but carries "
                f"{foreign!r}-system fields {offenders} — mixing length systems in one record "
                "is the silent factor-1000 (use converted_to for a cross-system read)"
            )

    # --- 访问器 ---------------------------------------------------------

    def properties_for_domain(
        self, domain: str, *, expect_length_unit: str | None = None
    ) -> dict[str, float | None]:
        """按域取字段。

        ``expect_length_unit``是给跨制消费方的安全带：声明了就必须对得上，
        对不上失败关闭，而不是悄悄给出差1000倍的数。
        """

        if domain not in MATERIAL_DOMAINS:
            raise MaterialError(f"unknown domain {domain!r}")
        if domain not in self.applicable_domains:
            raise MaterialError(
                f"{self.material_id!r} does not declare domain {domain!r}; "
                f"it serves {list(self.applicable_domains)}"
            )
        if expect_length_unit is not None and expect_length_unit != self.length_unit:
            raise MaterialError(
                f"caller expects length_unit={expect_length_unit!r} but the record is "
                f"{self.length_unit!r} — call converted_to({expect_length_unit!r}) explicitly"
            )
        return {prop.name: prop.value for prop in self.properties if domain in prop.domains}

    def evidence_for(self, field_name: str) -> EvidenceRef:
        for prop in self.properties:
            if prop.name == field_name:
                return prop.evidence
        raise MaterialError(f"{self.material_id!r} carries no property {field_name!r}")

    def weakest_grade(self, domain: str | None = None) -> str:
        """最弱证据分级（WDS ``weakest_status``形制）——记录整体或单域。

        记录整体的可信度没有意义，只有"某个域用到的那些字段"的可信度有意义。
        """

        selected = [
            prop.evidence for prop in self.properties if domain is None or domain in prop.domains
        ]
        if not selected:
            raise MaterialError(f"no properties serve domain {domain!r}")
        return max(selected, key=lambda ref: ref.weakness()).grade

    # --- 字节与内容地址 --------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        """规范化文档形（自指字段``content_sha256``照带，算哈希时再剔）。"""

        return {
            "contract_type": MATERIAL_RECORD_FACET,
            "contract_version": self.facet_version,
            "material_id": self.material_id,
            "length_unit": self.length_unit,
            "applicable_domains": list(self.applicable_domains),
            "dimensionless": sorted(self.dimensionless),
            "properties": [prop.to_document() for prop in self.properties],
            "appearance": self.appearance.to_document() if self.appearance else None,
            "content_sha256": self.content_sha256,
        }

    def content_address(self) -> str:
        """内容地址：剔自指字段后的规范字节SHA-256（轴3规则4）。"""

        document = self.to_document()
        document.pop("content_sha256", None)
        return canonical_sha256(document, MATERIAL_CANONICAL_PROFILE)

    def sealed(self) -> MaterialRecord:
        """填上自指内容地址的同一条记录（落盘前的最后一步）。"""

        return replace(self, content_sha256=self.content_address())

    # --- 跨长度制 --------------------------------------------------------

    def converted_to(self, length_unit: str) -> MaterialRecord:
        """显式换制。没有登记换算的复合量纲失败关闭——登记缺口，不猜因子。

        换出来的是**另一条记录**：字节变了，内容地址随之变，自指地址不跟着走。
        """

        if length_unit not in LENGTH_UNITS:
            raise MaterialError(f"length_unit must be one of {list(LENGTH_UNITS)}")
        if length_unit == self.length_unit:
            return self
        source_suffixes = _SYSTEM_SUFFIXES[self.length_unit] - _CROSS_SYSTEM_SUFFIXES
        converted: list[MaterialProperty] = []
        renamed: dict[str, str] = {}
        for prop in self.properties:
            suffix = unit_suffix_of(prop.name)
            if suffix not in source_suffixes:
                converted.append(prop)
                continue
            entry = _LENGTH_CONVERSIONS.get((suffix, length_unit))
            if entry is None:
                raise MaterialError(
                    f"property {prop.name!r} carries composite unit {suffix!r}, which has no "
                    f"registered {self.length_unit}->{length_unit} conversion — resolve it in "
                    "the consumer's adoption declaration instead of guessing a factor"
                )
            target, factor = entry
            new_name = f"{prop.name[: -(len(suffix) + 1)]}_{target}"
            renamed[prop.name] = new_name
            value = None if prop.value is None else prop.value * factor
            converted.append(replace(prop, name=new_name, value=value))
        return replace(
            self,
            length_unit=length_unit,
            properties=tuple(converted),
            dimensionless=frozenset(renamed.get(name, name) for name in self.dimensionless),
            content_sha256=None,
        )


# --- 严格加载 -----------------------------------------------------------


def _require_mapping(value: object, name: str, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaterialError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MaterialError(f"unknown keys in {name}: {unknown}")
    return value


def _load_property(payload: object) -> MaterialProperty:
    entry = _require_mapping(payload, "property", _PROPERTY_KEYS)
    value = entry.get("value")
    if not (value is None or (isinstance(value, int | float) and not isinstance(value, bool))):
        raise MaterialError(f"property value must be a number or null: {entry.get('name')!r}")
    domains = entry.get("domains")
    if not isinstance(domains, list) or not all(isinstance(item, str) for item in domains):
        raise MaterialError(f"property domains must be a list of strings: {entry.get('name')!r}")
    evidence = _require_mapping(entry.get("evidence"), "evidence", _EVIDENCE_KEYS)
    return MaterialProperty(
        name=str(entry.get("name", "")),
        value=None if value is None else float(value),
        domains=tuple(domains),
        evidence=EvidenceRef(
            grade=str(evidence.get("grade", "")),
            evidence_id=evidence.get("evidence_id", ""),
            method=str(evidence.get("method", "")),
            source_sha256=evidence.get("source_sha256"),
        ),
    )


def load_material_record(
    payload: bytes | str, *, expected_sha256: str | None = None
) -> MaterialRecord:
    """严格加载：字节锁、未知键、面版本、逐条校验一律失败关闭。

    ``expected_sha256``锁的是**文件字节**（轴3规则1，WDS案例引用材料的形制）——
    改一个字节就拒，且拒在解析之前。记录内部的``content_sha256``是另一件事
    （规则4的自指自校验），由``MaterialRecord``构造时验。
    """

    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise MaterialError(
                f"locked material bytes do not match: expected {expected_sha256}, got {actual}"
            )
    document = strict_loads(raw)
    if not isinstance(document, dict):
        raise MaterialError("a material record must be a JSON object")
    _require_mapping(document, "material record", _TOP_KEYS)
    if document.get("contract_type") != MATERIAL_RECORD_FACET:
        raise MaterialError(f"contract_type must be {MATERIAL_RECORD_FACET!r}")
    properties = document.get("properties")
    if not isinstance(properties, list):
        raise MaterialError("a material record requires a 'properties' list")
    domains = document.get("applicable_domains")
    if not isinstance(domains, list) or not all(isinstance(item, str) for item in domains):
        raise MaterialError("applicable_domains must be a list of strings")
    appearance_field = document.get("appearance")
    appearance = None
    if appearance_field is not None:
        entry = _require_mapping(appearance_field, "appearance", _APPEARANCE_KEYS)
        appearance = AppearanceRef(
            asset_id=entry.get("asset_id", ""),
            path_relative=str(entry.get("path_relative", "")),
            sha256=str(entry.get("sha256", "")),
        )
    return MaterialRecord(
        material_id=document.get("material_id", ""),
        applicable_domains=tuple(domains),
        properties=tuple(_load_property(item) for item in properties),
        length_unit=str(document.get("length_unit", "mm")),
        facet_version=str(document.get("contract_version", "")),
        appearance=appearance,
        dimensionless=frozenset(document.get("dimensionless", ())),
        content_sha256=document.get("content_sha256"),
    )


__all__ = [
    "EVIDENCE_GRADES",
    "LENGTH_UNITS",
    "MATERIAL_CANONICAL_PROFILE",
    "MATERIAL_DOMAINS",
    "METRE_SYSTEM_UNITS",
    "AppearanceRef",
    "EvidenceRef",
    "MaterialError",
    "MaterialProperty",
    "MaterialRecord",
    "load_material_record",
    "unit_suffix_of",
]
