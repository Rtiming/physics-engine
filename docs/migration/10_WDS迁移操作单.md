# WDS迁移操作单（给winding-deviation-sim的会话，自足执行）

本单由physics-engine侧维护（2026-08-04立）。执行者是WDS自己的会话；
本单不替WDS动任何字节。做完一步就在WDS的design/26采纳台账与本仓
spec/90消费方登记各记一行。

## 前置阅读（顺序）

1. WDS `docs/design/25`（形态与融合正本）与`docs/design/26`（采纳台账）
2. 本仓`docs/spec/00`总纲与`02`—`08`七条轴（2026-08-04全部v1.0冻结），
   重点看各轴第三节"已知缺口"里点名WDS的行
3. WDS `docs/教训登记.md`（尤其5.6冻结diff→重签→跑门的顺序、6.4文档保质期）

## 批次A：~~四件~~三件（**2026-08-04晚已执行**，WDS决策记录design/27）

1. ✅ **面清册加状态字段**：`_FACETS`单表38行（32 frozen/4 internal/2 draft），
   36个存量行由活模块程序化转录零手抄，读取端行为逐字不变。
2. ✅ **决策记录联动门**：`tests/test_facet_freeze_linkage.py`冻结清单门，
   建立当日**验过红**（无记录改frame_packet minor→红→还原复绿）。
3. ✅ **映射设施新契约出生即合规**：`contracts/groove_fit_sweep.py`两面
   出生为draft；声明通道集（排除必带理由）、键碰撞与分母纪律
   （失败点占行带理由禁携数值、丢点即炸）随契约立法。
4. ↷ **runner基类改排批次B**（2026-08-04工程判断，理由：批次B切换import
   时本就重写三插件接线层，指纹绑定的胶水只碰一次；实读发现的真实重复面
   =canonical_payload+身份绑定verifier+指纹排除三件套，另有
   `SpoolWrapOutcome`缺互斥校验的真缺口，一并归批次B）。

流程硬性提醒（教训5.6原文纪律，批次B沿用）：**冻结diff→重签两份性能回执
（负载<6，签完核status与resource_qualification两字段）→verify_project
general/full/actual→更新design/26与本仓spec/90→提交**。
中途改一个字节回执作废重来。竞争进程拦V2时等机器空再补，别杀别人的活。

## 批次B：存量迁移（阶段3，等本仓参考实现随批次A后落地）

顺序按依赖，每块独立可回退：

1. 契约基座（版本清册助手、单位/ID校验）改为import `physics_engine`
2. `storage/`（原子发布+严格复读）对齐轴5参考实现
3. 验收元门禁对齐轴6

每块的证据义务（缺一不可）：

- 全部现有案例产物**指纹逐字节不变**（跑`scripts/validate_case_double_run.py`
  级别的对比，改造前后各一次）
- V1/V2/VA全绿+两份回执重签
- 不过门即回退，不许"先合了再修"

## 明确不在本单内的

- 力学域（model/solve/contact/dynamic）进不进引擎——远期分岔，用户裁决
- V3-R真实发布门——固定由用户手工
- FTS侧任何操作——见20_FTS迁移操作单
