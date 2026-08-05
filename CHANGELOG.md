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

- **`validation/`落地**：同行库对比实验的独立目录与独立venv，身份边界写死
  （**永不进`dependencies`、永不被`src/physics_engine` import**）；建环境脚本、
  钉死版本的依赖清单、逐库许可证结论（决策0025）。这是0015第二条
  "把别人的库下载下来进行对比实验"的兑现
- **案例`peer_fcl_distance`**：与python-fcl 0.7.0.11逐点对拍**2700组**固定种子
  输入（球/球-胶囊/胶囊-胶囊 × 分离/擦边/穿透），最大绝对偏差**2.27e-13mm**、
  接触判据分歧**0**。产物走run package，manifest带`peer_library`
  （名称/版本/安装方式/许可证/随轮二进制SHA-256）与`runtime_environment`
- **对拍抓到同行两处真实缺陷**，并用`fractions.Fraction`精确算术复核：
  FCL的`distance(enable_signed_distance=True)`在**分离**构型上可错到45.6%相对误差；
  `collide().penetration_depth`在胶囊-胶囊上与**FCL自己的`distance()`**矛盾。
  两处均记录进产物但**不作判据**（spec/08规则1：同行的数不是真值，是另一个证人）
- `tests/cases/test_peer_fcl_distance.py`：conformance门+三条注错必须红例
  （四元数次序/漏减半径/胶囊半长当全长），负载级`batch`；**同行库缺席时skip**，
  accept不因缺库而红。**`batch`档从此不再是空档位**（4条真测试）
- `.gitignore`补`.venv`（无斜杠）：并行开发的worktree里`.venv`是符号链接，
  带斜杠的规则只匹配目录，此前它一直显示为未跟踪
- **`physics_engine.oracles`（实验档）**：轴7规则2的清单面参考实现——生成器身份
  （含**生成器脚本自身的SHA-256**）、逐条expected与逐条tolerances、数组双哈希
  （raw级+语义级）、清单自指哈希；严格加载失败关闭。三条本仓加强：每个量的容差
  必须带非空`reason`（判据表第三列从文档义务变成**加载条件**）；非数值量的容差
  必须为零；`regenerated_by`必须指向**存在的**`docs/decisions/`文件——轴7规则5
  第一次有了执行体
- **首批四条A档案例**（`cases/`，决策0020）：`segment_distance`（球/胶囊解析距离
  与穿透，abs 1e-12mm，五条退化分支+一条一般路径各一条手算用例，**分支覆盖经
  opcode级trace实测**且未改动源码一字节）、`rotated_aabb`（八角点枚举对Arvo
  中心-半边长闭式解，abs 1e-9mm；四类典型错重算得31.0/18.5/40.0/65.3mm偏差）、
  `broadphase_superset`（`separation<0 ⟹ AABB相交`，120对固定种子语料，反例严格为0）、
  `mesh_asset_integrity`（资产SHA+包盒保守性双零容差，**两条必红语料常驻**）
- **`tests/test_narrow_phase.py`的判据搬进清单**（轴7规则3：不得在测试里复述
  oracle公式）；原文件保留可信度分级与诚实降级两条接口诚实性判据
- **仓内自带合成网格资产**（Chrono形制）：284字节二进制STL+生成脚本一并入库。
  0017那条真实缺陷（声明SHA逐字节吻合而声明包盒与真值在z轴完全不相交）此前
  **引擎没有任何门能发现**，现在有了
- **`tools/check_case_pages.py`**：案例页六必填字段缺一即红，`accept.py` full档的
  可选槽位就此上膛；外加案例目录必须被`cases/README.md`索引、`oracle.json`必须过
  严格加载器、清单`case_id`与`load_tier`必须在页里出现三条结构校验。
  六必填按**标题正文**匹配、忽略中文序号——案例可以有第七第八节，
  逼它为凑编号删掉最贵的发现是把形式看得比证据重
- `accept.py`的ruff扫描范围加上`cases/`

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
- **`physics_engine.state`与`physics_engine.integrate`（实验档，决策0019）**：
  **引擎第一次能推进物理状态**。状态层是显式数组不是对象图，
  `StateLayout.fingerprint()`把**打包次序做成内容地址**——次序反了而
  `dof_count`与向量长度都不变，只看长度的检查发现不了，指纹发现得了。
  三个积分器（显式Euler／半隐式辛Euler／velocity Verlet）各带spec/12第4.2节的
  **五项出生声明**（适用域／形式阶与实测阶分开／稳定性／耗散记账／失败阶梯），
  缺一构造即拒；三者`production_ready`全为`False`并有门守着
- **0016甲案第一次真正落地**：积分公式**只写一遍**、按`VectorOps`求值，
  `PurePythonOps`与`NumpyOps`执行同一串运算同一个次序——**逐位相同是构造保证的**。
  按spec/12第5.3节"能逐字节就必须逐字节"，对拍档位是**逐字节**而非容差。
  实测600次积分零差异；破法验红：加速档改`float32`→红，
  改共享公式源的求和分组→**不红**（两个后端一起变了，正是"一份公式源"的直接结果，
  边界已写进案例页）
- **NumPy进`[project.optional-dependencies]`的`accel`档**，`dependencies`仍为`[]`；
  eager import模块数**97不变**（NumPy只在构造加速档时import）
- **两条B档案例**：`ballistic_free_flight`（9条oracle）与`harmonic_oscillator`
  （3条oracle）。四条判据与spec/12写规范时预先记下的实测值**逐位吻合**；
  "同幅反号"另立一道独立的门
- **`source_bytes`上限147456→229376并记账**（决策0019）：T4入仓后占用99.2%、
  余量1224字节，门虽未红但余量已无意义。上限的定性再次确认——
  不是"引擎必须保持小"，是"增长必须是一次被记录的决定"

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
