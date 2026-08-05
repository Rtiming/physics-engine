# 变更日志

形制按Keep a Changelog；版本语义见decisions/0010与0012（0.x期：
minor可破坏兼容，patch只含兼容修复，**不回port**——修复只进新版本）。
发版脚本强制：版本必须先在本页有条目才准发。

## Unreleased

并行开发期的分槽（plans/02第三节规避规则3）：每条轨道只在自己的小节内append，
合并冲突面为零。发版时由一次收口提交合并成版本条目。

### T0 收口（缺陷清零+面清册+文档对账）

- 面清册一次登记齐四个新面（`physics_scene_collision_events`、
  `engine_perf_baseline`、`engine_oracle_manifest`、`engine_behavior_baseline`），
  其中`physics_scene_collision_events`补登记了自0.3.0起就在落盘却未登记的形制
  （本仓自己违反轴1规则1近两个版本）
- `pyproject.toml`登记与负载三级对应的pytest marker（`batch`/`serverclass`）
- **失败关闭补齐（破坏性：此前被放行的非法场景现在会被拒）**：
  `MeshAsset`的`units`/`usage`/`convexity`与`CollisionShape.direction`
  运行时校验取值；场景文件缺`direction`不再放行；重复`body_id`与
  未知`allowed_pairs`成员在**加载期**失败关闭（此前推迟到查询构造期才炸）
- `pe-scene`退出码修复：查询构造异常纳入捕获，非法场景两个子命令一律退出码2
  （此前`check-collisions`抛栈回溯并以1退出，而1的语义是"有候选"）
- `collision_events.json`补`facet_version`字段（**破坏字节**，实验档minor内声明）
- `examples/collision_preview_cell.scene.json`的网格AABB改为资产真值
  （此前声明的包络在z轴上与资产完全不相交，`direction: "envelope"`不成立）

### T1 性能门

- **spec/13的预算第一次成为会红的门**（决策0018）：`tests/perf/test_budgets.py`
  守三条**确定性量**——源码字节（wheel体积的代理）、eager import模块数
  （冷启动的结构代理）、运行时依赖数（永远为0）。**墙钟不进门**：走
  `tools/bench.py`产报告（照Drake形制明写"本报告不是回退测试"），
  理由是共享runner墙钟CV=2.66%、2%阈值假阳率45%
- `benchmarks/engine_budgets.baseline.json`（`engine_perf_baseline`面）：
  预算与实测口径入册；**wheel 30KB与源码字节上限都改成"抬可以但必须记账"的
  历史账**——现行上限必须等于历史末行且每行带决策记录，悄悄抬过不去
- **`tools/accept.py`四处修**：双档真分家（此前quick与full命令集逐字相同，
  30/120退化成一件事，现按spec/13负载三级用marker分）；轴6规则4资源资格
  （负载不合格→计时`NOT_EVALUATED`，功能结论一个字不改）；轴6规则5执行树哈希
  （金标件与被测输入都在树内）；**超时改杀整棵后代进程树**
  （`subprocess.run(timeout=)`只终止直接子进程，实测确认老写法下孙进程存活）
- **空档位语义**：申报过的档位选不中测试（pytest退出码5）视同通过并在回执与
  stdout如实记名；**交互级绝不许为空**——那说明marker笔误让整档静默不跑
- 未跟踪符号链接不再让验收器崩（指向目录时`IsADirectoryError`）：
  链接的内容按其**目标字符串**入指纹，不跟随读目标字节
- governance元测试 11条 → 21条

### T2 几何

- **`physics_engine.geometry`（实验档）**：四种解析原语的体积、质心、
  绕质心惯量张量（kg·mm²，参考点写进字段名），与spec/11规则2点名却至今
  零实现的rounded-core SDF `φ = φ_core − r_f`（首个实现）。惯量是力学域
  接进来第一个撞上的缺口
- `mass_properties`收密度或质量**恰一个**；带法兰的`FiniteCylinder`与
  `MeshAsset`**失败关闭**——声明里没有法兰轴向尺寸，AABB是包络不是质量分布，
  缺口登记进spec/11第二之二节
- 33条门：解析闭式解 + 圆角盒三条退化极限 + 以SDF为独立oracle的数值求积 +
  平行轴/对称正定/主不变量三条自洽门；**十二条破法逐条实测必须红**，
  其中"圆角盒面板平移项"三条解析退化极限一条都抓不住、只有数值求积抓住
  （决策0022第五节）

### T3 oracle与案例

### T4 力学地基

- **`docs/spec/12_力学内核与时间推进_v0.md`**：T4的规范面前置就位
  （plans/02第六节第2项结案）。能量项四方法协议与注册表分发、状态三层与
  边界（形状/材料/位姿都不是状态）、准静态与瞬态两路并列、积分器五项出生
  声明、0016的条款化、**有限差分门验不了物理**（附1600倍活标本）与解析
  oracle的两道门、守恒量三种写法及各自的假通过堵法、搬迁五道切换门、
  B档四条判据、已知不成立的五字段登记形制
- **登记三处与既有裁决不一致**（spec/12第11.2节）：①0016设想的"纯Python↔
  NumPy"两实现在WDS并不存在，纯Python那侧要新写——每块内核是"搬一条+
  写一条+建对拍"（0016与plans/02 T5已据此修订）；②"无状态纯函数"只对
  能量层成立，接触摩擦层有真状态，T5搬到接触前要再裁一次；③对拍档位不是
  二选一，能逐字节就必须逐字节（0016已确认此为原意）
- **B档判据两处修正**（实测复核）：显式Euler误差恰为`−a·T·h/2`，与半隐式
  大小相同符号相反——按`|误差|`写的判据分不开两个积分器；`cos(ωT)`收敛
  比值不得写死为4（渐近值，粗档实测3.9985，写死会让正确实现红）

### T6 资产（materials）

- **`physics_engine.materials`（实验档）**：材料记录形制的参考实现——一份记录
  聚合多域字段，`applicable_domains`+属性级`domains`标签（一个字段可同时服务
  力学与光学，共享值共享证据）；参数级证据分级（WDS六档与FTS五档的统一映射，
  两处不一致如实登记为待收敛）；单位后缀校验走`identity`；内容寻址自指自校验
  与文件字节锁（轴3规则1/4）；`engine_material_record` 0.1面
- **mm制与米制边界钉死**：记录声明`length_unit`，同一条记录混用两制失败关闭
  （静默1000倍的入口）；跨制取值必须显式`converted_to`，换算表只覆盖纯长度
  量纲，`N/mm²`一类复合量纲无登记换算即拒——不猜因子
- 规范页`docs/spec/14_材料记录与物理性质_v0.md`（v0草案，六条最低红例全验，
  冻结条件三条）；决策记录0023

## 0.4.0 — 2026-08-05

新增：

- **narrow phase第一片（球/胶囊族）**：线段-线段闭式最近距离；同族对给
  `confidence="narrow_phase"`与精确`penetration_mm`；**broad命中但narrow
  判分离的对不再报事件**（假阳性消除）。圆柱/盒/网格对诚实保留
  `broad_phase`（下一片再做，不冒充）
- `PosedBody.transform_point_mm`/`rotate_local_mm`位姿变换
- `segment_segment_distance_mm`进公开面

## 0.3.0 — 2026-08-04

新增：

- `physics_engine.scene`：场景文件格式（`physics_scene`面1.0.0，draft）——
  严格加载、未知键失败关闭、`extensions`声明式扩展加载（Gazebo形制）
- `pe-scene`命令行入口：`validate`、`check-collisions`（linter式退出码；
  `--out-dir`走run package原子发布+同边界复读）
- `SHAPE_KINDS`注册表与`register_shape_kind`（重复登记拒收）
- 示例场景`examples/collision_preview_cell.scene.json`（真实参数）

## 0.2.0 — 2026-08-04

首个发布版。新增：

- `facets`：面清册+失败关闭读取端（轴1参考实现）
- `canonical`：声明式规范化JSON读写对（轴3规则2）
- `identity`：四段身份、命名空间ID、单位后缀校验（轴2）
- `provenance`：耐久独占写、no-replace目录改名、四点签名保护读、
  目录快照与机械严格复读（轴5机械半边）
- `run_package`：装配（manifest最后写）、语义闭包复读、生命周期法则（轴4/5）
- `shapes`+`collision`（实验档）：形状声明层与broad-phase碰撞查询
- `tools/accept.py`自吃药验收器（30/120双档、三轴正交）与
  `engine_acceptance_receipt`面

## 0.1.0 — 2026-08-04

骨架占位（未发wheel）：包结构、章程、七条规范轴、操作单。
