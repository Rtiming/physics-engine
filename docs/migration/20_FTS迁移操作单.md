# FTS迁移操作单（给fts-digital-twin的会话，自足执行）

本单由physics-engine侧维护（2026-08-04立）。执行者是FTS自己的会话，
在其自己的治理（ADR、validation matrix）下进行；本单不修改FTS任何字节。
背景裁决：用户2026-08-04定两仓走共同物理引擎仓（本仓decisions/0001、0002）。

## 第0步（闸门）：新ADR取代其0001

FTS的`docs/decisions/0001-standalone-boundary.md`明文禁止兄弟项目依赖。
**在新ADR落地前，本单其余步骤一步都不许做。** 新ADR建议要点：

1. 引用用户2026-08-04裁决与本仓`docs/decisions/0001`；
2. **只**放开对`physics-engine`一个上游的依赖（spec采纳先行，代码依赖
   须逐步、附证据）；对其余兄弟项目的standalone纪律原样保留；
3. 其"6个外部稳定面"承诺不受影响——本仓是它的上游，不是它的消费方；
4. 按其validation matrix走文档变更的最低验收门。

## 第1步：轴1采纳映射（其versions.py ↔ spec/02）

- 其5个分组面（CONTRACT/MODEL/RUN_PACKAGE/RULE_REGISTRY/SCENARIO_CATALOG）
  逐个登记；Pydantic模型内部的细粒度版本决定是否升为独立面；
- 显式化"已测minor上限"概念（轴1规则3的WDS侧形制，其现0.x版本制下如何
  对应由其自定，偏差写明理由）；
- 采纳声明落其仓内（形式由其治理定），并在本仓`spec/90`加登记行。

## 第2步：轴2采纳映射（其scene.py ↔ spec/03）

- StableId/Quantity/AssertionRecord已达标，登记即可；
- "只增不改"目前由scenario catalog的lifecycle承担，对**非scene面**
  （校准剖面、效果覆盖）是否同样append-only，显式声明或列偏差。

## 第3步：轴3—7采纳映射（spec/04—08）

其四级SHA、verify-after-write、accept.py预算与governance元门禁、解析oracle
纪律与各轴的对应关系逐条登记；两仓机制不同处按各轴"要求冻结、机制自选"的
条款声明机制归属。

## 第4步（远期）：存量迁移候选

每块的证据义务：其`package_sha256`与physics_fingerprint**逐字节不变**、
four-gate发布器全绿、governance测试全绿。不过门即回退。

## 明确不在本单内的

- 光学域（optics/effects/assessment）进不进引擎——远期分岔，用户裁决；
- 其M4（实测拟合）里程碑——与本迁移无关，照其自身计划走。
