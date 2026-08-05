"""physics-engine：多域数字孪生的引擎内核（规范先行，参考实现随消费方需求生长）。

安装（舰队wheelhouse链路，decisions/0010）：

    git clone ts-orangepi:wheelhouse.git ~/wheelhouse   # 每台机器一次
    uv add "physics-engine==0.4.0" --find-links ~/wheelhouse
    # 或 pip install "physics-engine==0.4.0" --find-links ~/wheelhouse

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

**实验档**（随spec/10、11、12、14草案演进，minor内可破坏）：

- `physics_engine.shapes` —— 模拟形状声明层（spec/11）
- `physics_engine.collision` —— broad phase + 球/胶囊族narrow phase（spec/10）
- `physics_engine.scene` —— 场景文件与数据层入口
- `physics_engine.geometry` —— 体积/质心/惯量张量与rounded-core SDF（spec/12）
- `physics_engine.materials` —— 材料记录：多域字段+证据分级+单位边界（spec/14）
- `physics_engine.oracles` —— oracle清单面：expected/tolerances/双哈希（轴7）
- `physics_engine.state` —— 状态形制与打包次序契约（spec/12）
- `physics_engine.integrate` —— 时间推进：三个积分器与五项出生声明（spec/12）
- `physics_engine.energies` —— 能量项四方法协议与三个能量项（spec/12第三节）
- `physics_engine.solve` —— 准静态平衡：牛顿+回溯线搜索（spec/12第4.1节）

**能力边界（诚实条款）**：`integrate`能推进无接触、无约束的二阶系统
（显式/半隐式/velocity Verlet，三者`production_ready`全为`False`）。
能量项有均匀重力、轴向拉伸、小挠度弯曲；准静态平衡可解。
**没有隐式时间积分族、没有几何精确（DER）弯曲、没有扭转、
没有接触与摩擦、没有约束**——
力学与光学按decisions/0015仍在搬迁中，路线见`docs/plans/02`。

**加速档**：`pip install 'physics-engine[accel]'`装NumPy后走加速实现；
核心永不要求它（0014零设施承诺），两实现逐字节对拍是进仓门（0016甲案）。

**实验档模块不进顶层re-export，按全路径import**（`from physics_engine.geometry
import mass_properties`）。这不是疏漏是预算纪律：顶层eager import的模块数是
冷启动延迟的结构代理，`tests/perf/test_budgets.py`守着它；把三个实验档模块拉进
顶层要多付5个模块的import成本，而它们的调用方都明确知道自己要什么。
稳定倾向档才享有顶层便利——档位在这里是有价格的。

完整面见各模块`__all__`。
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
    "FTS_PROFILE",
    "WDS_PROFILE",
    "CanonicalError",
    "CanonicalProfile",
    "CaseIdentity",
    "Facet",
    "FacetError",
    "FacetRegistry",
    "FacetStatus",
    "IdentityError",
    "ProvenanceError",
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
