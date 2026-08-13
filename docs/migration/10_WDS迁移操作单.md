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

批次B的import切换机制**已裁决**（本仓decisions/0010）：钉`physics-engine==<版本>`+`--find-links ~/wheelhouse`，uv.lock的wheel哈希让引擎字节自动进执行树回执绑定；master侧在workspace_sync.manifest加wheelhouse一行。切换门=两实现输出逐字节一致+全部既有产物指纹不变。

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

## 力学批次C：easy-axis正式单站接缝（2026-08-13，决策0061，已完成）

用户已经授权力学迁移。为避开WDS主工作树尚未收批的groove-fit改动，本批在
`codex/wds-section-beam-adoption`独立worktree执行，范围冻结为：

1. physics-engine升0.6.0并提供显式`LinearElastic1D`；不能用任意大屈服应力
   冒充线弹性，也不能覆盖已发布0.5.0；
2. WDS的`EnergyContext`增加默认`None`的单站配置；关闭时原`BendingEnergy`逐句原路，
   打开时只把一个内顶点的easy-axis项交给`KirchhoffFiberSectionBending`，hard-axis、
   其余站点、全杆Newton、接触与continuation仍归WDS；
3. 第一片只准静态线弹性。动态入口显式拒绝；逐纤维塑性历史在continuation接受点的
   commit/回传未接好前不得启用；
4. 中点纤维的算法模量按`EI_easy/sum(A_i*y_i**2)`校准，避免把离散二次矩误当解析
   `w*t^3/12`造成有限点数漂移；
5. 依赖以仓内相对0.6.0 wheel和`uv.lock`哈希绑定；master同步清单必须核对wheel
   SHA-256，并在强制重装后逐项核已安装版本与RECORD字节，不能只验import/版本。

本地候选成立的最低证据是：引擎调用探针、energy/gradient/Hessian对原站点、独立FD、
默认路径状态字节不变、动态与静态绑定漂移必红、代表性案例改造前后物理指纹相同、一个
独立plugin身份的1.1命名案例真实启用，以及WDS本机general/full/actual。正式接缝可用
还需：0.6.0不可变发布wheel与vendor哈希一致、既有14个默认案例逐案对拍、master默认V2
通过，最后才合入WDS干净main；要称“物理采纳”还必须指明命名case与适用域。

**完成记录**：0.6.0已以不可变wheel正式发布；WDS main `f88986f7...`统一绑定正式
wheel SHA `53114e43...`，14案双版本门、两份性能正本、隔离master V1/V2/VA和开发Mac
Node门均通过。14案口径为14/14解析字节一致、10个收敛结果物理指纹一致、3个既有受控
失败诊断一致、1个保持input-only。R250 v002是唯一显式启用该接缝的命名案例；批次C
到此完成，但下节列出的扩大迁移仍全部在外。

## 明确不在本单内的

- 力学域的**扩大迁移**——批次C只放开一个默认关闭的easy-axis线弹性站点；整杆多站、
  轴向N、hard-axis、扭转、塑性历史与动态仍需新的消费需求和独立决策，不能从单站候选
  外推成WDS力学域已经搬入引擎
- V3-R真实发布门——固定由用户手工
- FTS侧任何操作——见20_FTS迁移操作单
