"""physics-engine：多域数字孪生的引擎内核（规范先行，参考实现随消费方需求生长）。

安装（舰队wheelhouse链路，decisions/0010）：

    git clone ts-orangepi:wheelhouse.git ~/wheelhouse   # 每台机器一次
    uv add "physics-engine==0.2.0" --find-links ~/wheelhouse
    # 或 pip install "physics-engine==0.2.0" --find-links ~/wheelhouse

**0.x语义**：仓库处于快速演进期（模块地图spec/01大半尚空），minor跳变可以
破坏兼容。消费方必须**钉精确版本**并经自己的门禁自觉升级；升级永远不是
被动漂移。每次发版有决策记录。

公开API分两档（档位即本包自己的面状态，轴1规则5）：

**稳定倾向**（对应已冻结轴的参考实现，破坏性改动须决策记录）：

- `physics_engine.facets` —— 面清册与失败关闭读取端（轴1）
- `physics_engine.canonical` —— 声明式规范化JSON读写对（轴3规则2）
- `physics_engine.identity` —— 稳定ID与单位后缀校验（轴2）
- `physics_engine.provenance` —— 溯源机械层：耐久写/no-replace/保护读（轴5）
- `physics_engine.run_package` —— 装配、语义复读、生命周期法则（轴4/5）

**实验档**（随spec/10、11草案演进，minor内可破坏）：

- `physics_engine.shapes` —— 模拟形状声明层
- `physics_engine.collision` —— broad-phase碰撞查询

顶层只re-export最常用的名字；完整面见各模块`__all__`。
"""

from physics_engine.canonical import (
    FTS_PROFILE,
    WDS_PROFILE,
    CanonicalError,
    CanonicalProfile,
    canonical_bytes,
    canonical_file_bytes,
    canonical_sha256,
    strict_loads,
)
from physics_engine.facets import (
    Facet,
    FacetError,
    FacetRegistry,
    FacetStatus,
    parse_version,
)
from physics_engine.identity import (
    BASE_UNIT_SUFFIXES,
    CaseIdentity,
    IdentityError,
    assert_quantity_fields_have_units,
    has_unit_suffix,
    parse_case_identity,
    parse_namespace_id,
)
from physics_engine.provenance import (
    ProvenanceError,
    read_protected_file,
    rename_directory_noreplace,
    verified_bytes_snapshot,
    write_durable_exclusive,
)
from physics_engine.run_package import (
    assert_lifecycle_fail_closed,
    publish_package,
    read_verified_package,
)

__version__ = "0.4.0"

__all__ = [
    "BASE_UNIT_SUFFIXES",
    "CanonicalError",
    "CanonicalProfile",
    "CaseIdentity",
    "FTS_PROFILE",
    "Facet",
    "FacetError",
    "FacetRegistry",
    "FacetStatus",
    "IdentityError",
    "ProvenanceError",
    "WDS_PROFILE",
    "__version__",
    "assert_lifecycle_fail_closed",
    "assert_quantity_fields_have_units",
    "canonical_bytes",
    "canonical_file_bytes",
    "canonical_sha256",
    "has_unit_suffix",
    "parse_case_identity",
    "parse_namespace_id",
    "parse_version",
    "publish_package",
    "read_protected_file",
    "read_verified_package",
    "rename_directory_noreplace",
    "strict_loads",
    "verified_bytes_snapshot",
    "write_durable_exclusive",
]
