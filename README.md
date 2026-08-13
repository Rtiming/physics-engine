# physics-engine

rtime的**物理引擎**：多域物理（力学、光学、电磁，将来热）+ 模型生成 + 物理性质
（材料）+ 外观（颜色）+ 溯源纪律的仿真引擎仓。2026-08-04用户裁决建立并定位
（原名twin-engine，同晚更名）：走共同仓库、深度融合、接口规范统一。

创始消费方：`winding-deviation-sim`（力学，绕制偏差）与`fts-digital-twin`
（光学，傅里叶光谱仪）；潜在第三个消费方`case2-digital-twin`（专利装置数字样机）。
**用户级定位（decisions/0014）**：目标机器状态永远未知——单机冷启动性能是第一指标
（import、场景校验、quick/full验收与wheel体积的预算/实测正本都在
`benchmarks/engine_budgets.baseline.json`，有门守着陈旧度，不在README复制易漂移墙钟）、
功能操作不因宿主负载拒绝或变形、零设施假设是承诺。
**wheel体积随版本记账不写死数**（正本`benchmarks/engine_budgets.baseline.json`）：0.5.0实测**193066字节**，是0.4.0的6.24倍——三个物理域进来了，而同期源码涨了7.88倍，**wheel涨得比源码慢**。此前写的"30KB离线装"在0.5.0已不成立，见spec/13第一条。

目标模块图与各能力现居地见[docs/spec/01_模块地图与域划分_v0.md](docs/spec/01_模块地图与域划分_v0.md)。
仓内依赖图与产品级工程设计/计算平台拓扑要同时看：入口见
[docs/architecture.md](docs/architecture.md)。
这段时间跨GCW、WII、WDS、FTS形成的上位目标已集中到
[docs/platform/README.md](docs/platform/README.md)：建设自有的工程模型、工作流/计算
运行时、视觉交互内核、产品设计系统和Engineering Studio；物理仿真只是平台能力之一，
Blender、OCCT、Three.js、VTK等只是可替换适配器。四项目的
多表示模型、固定场景、硬件适配与提速专项见
[research/14](docs/research/14_多表示模型资产与硬件适配架构_20260809.md)，候选实施路线见
[plans/10](docs/plans/10_模型资产与呈现架构演进计划_20260809.md)。这些页面当前都是提案，
不表示平台、Studio、模型资产系统、Blender桥接或新facet已经实现。

## 能力边界（诚实条款）

本节按**用户给的六个真实场景**（[plans/04](docs/plans/04_真实使用场景与能力差距_20260805.md)）
写，不按模块清单写。理由：模块清单只回答"有什么"，回答不了"能不能算我要算的"，
而后者是决策0048第二节定的**主分母**。

### 两条分母必须并排看

| 分母 | 验什么 | 今天 |
|---|---|---|
| **主｜用户六场景端到端** | 算不算得了**我们要算的** | **0/6** |
| 从｜同行C档13条标准案例 | 算得**对不对** | **6/13** |

能力位清单的当前机械计数是**12/42**（其中穿过物理机械的done位5/12）；
42的每场景位数已由0057冻结为`7/5/10/7/6/7`。这个数不是加权完成度，
正本与逐位证据见[`docs/capability_ledger.json`](docs/capability_ledger.json)。

**只报后者是恭维自己**：6/13是用别人的题打的分，而那些题是在还不知道引擎
要用来算什么的时候选的（0040的分母经0048第二节修订为两条并列）。

### 六个场景逐条（可证伪）

| 场景 | 今天能算什么 | 端到端 | 缺哪一块 |
|---|---|---|---|
| ①超导带绕一圈的电流分布 | Norris 1970**直薄带**自场闭式（`electromagnetics.superconductor`） | ❌ | 绕成一匝几何就变了；**无电流自由度**；无截面 |
| ②多线圈互感与电容 | **理想细丝圆环**互感的Maxwell闭式与Neumann积分（`electromagnetics.inductance`） | 🔶部分 | 真线圈有截面、有多匝；**电容为零**（要静电求解） |
| ③不规则体落漏斗 | 刚体自由飞行；**10球平面漏斗最小组合已跑通**（场景候选→球体broad/narrow→活动响应+重力/解析壁面+耗散账） | 🔶部分 | 仍是解析球和三个平面；45对候选全量预声明且无空间索引，平面未进动态流水线；无网格窄相、转动、摩擦与实验标定 |
| ④光学散射／干涉／衍射 | 双光束干涉、杨氏双缝、艾里斑、FTS仪器线型（`optics`） | 🔶部分 | **散射为零**（要Mie）；一般衍射为零（要二维复数场+FFT） |
| ⑤平面带材绕十几匝 | — | ❌ | 自接触、半径生长、喂料、带宽与厚度自由度，全为零 |
| ⑥完整放线—导向—张力—收线 | 场景装配与接触对声明（`scene.Scene`）、位姿时间线（`motion`）、接触力（`contact`） | ❌ | `Scene`候选只接通了**状态绑定球体**的法向响应，尚不能表示导轮/带材接触；`Actuator.apply`的物理未实现；法兰轴向尺寸仍是spec/11已知缺口 |

**六个场景里三个有理想化的部分答案，三个一点都算不了。端到端能跑通的仍是零。**

### 案例穿过引擎哪几层

24个案例不是一类东西。走完`state`→`energies`→`solve`整条路的**6条**
（两条悬臂+Euler屈曲+**三条接触**：静置阈值、历史迟滞、多体金字塔）；
3条走`state`→`energies`→`integrate`（两体弹簧、单次弹跳、十球漏斗）；
3条走时间推进不碰能量装配与求解器；
1条只验`energies`协议层；**5条是闭式计算器，不碰引擎的任何一层**（三光学+两电磁）；
其余6条验几何查询、资产治理与模型生成。**6+3+3+1+5+6 = 24。**
逐案例标注见[cases/README.md](cases/README.md)第一节之二。

**"验公式"与"验引擎"是两类，混在一个计数里计数就不再有意义**（0048第三节）。
本仓的验证纪律强度在同行里罕见（research/12实测：NVIDIA 237个样例里带定量容差的
只有1个，SOFA是474:1），**但验证纪律不是物理能力**——把前者当后者是这个仓
最可能犯的错，因为它是从治理层长出来的。

### 结构性缺的四条（不是"再加几个案例"能补的）

**体积与厚度**（一切都是中心线或闭式，截面上没有任何自由度可放）、
**接触**（🔶罚法向、摩擦历史、法向dashpot、球-球动态检测—响应与10球最小组合已落；
仍缺网格窄相、空间索引、解析平面统一接线、载荷步回退和转动自由度）、**耦合**
（`src/physics_engine/couplings/`目录**根本不存在**，spec/15只留了模块位）、
**历史**（✅**已解**——0050按声明的接触对定长分槽，锚点写回状态）。
四条的完整判据见[plans/06](docs/plans/06_四条结构性缺口_20260805.md)，
补充路线与同行调研见[plans/08](docs/plans/08_子系统补充计划_20260806.md)——
**它们才是"能不能变成真引擎"的判据，案例计数不是。**

### 逐模块的能与不能

**能做**：声明形状与材料（`shapes`/`materials`）；算解析原语的质量属性——体积、
质心、绕质心惯量张量（`geometry`）；查碰撞——broad phase全域 + 球/胶囊族narrow phase
（`collision`/`scene`，其余族诚实保留`broad_phase`可信度）；装配能量项——均匀重力、
轴向拉伸、小挠度弯曲、**几何精确（DER）弯曲**、**点载荷**（`energies`）；求准静态平衡——
牛顿+回溯线搜索+稠密LU／带状求解，并可判平衡是不是极小（`solve`）；推时间积分——
显式Euler／半隐式辛Euler／velocity Verlet／`velocity_verlet_damped`
（`integrate`）与刚体姿态积分（`rigidbody`）；
**算接触**——罚法向（半空间与球-球）、粘着弹簧、库仑return-map、准静态步进器，
线性法向dashpot与全阻尼恢复系数标定，**多体成立、锚点是真历史、瞬态阻尼碰撞也通**
（`contact`，决策0050、0055）；
光学闭式（`optics`）与电磁闭式（`electromagnetics`）。

**不能做**：**没有隐式时间积分族**；**没有扭转**；**没有约束**——边界条件只有
"显式钉住某些自由度"这一种；**没有跨域耦合**；**没有场求解**——光学与电磁至今全是闭式，
没有网格、没有复数场、没有FFT。
球-球动态检测到响应的接线已落；**接触仍缺**网格窄相（圆柱/盒/网格族）、
候选空间索引与动态平面统一接线、载荷控制失败后的真回退、以及**接触体的转动自由度**
——今天是质点+半径、**没有力矩**，所以多点接触在今天的自由度下不多算任何物理。
六个积分器（四个平动+两个刚体）的`production_ready`**全为`False`**且有门守着。

源码里**零个**`TODO`与`NotImplementedError`——未做的事全靠这一段与
`src/physics_engine/__init__.py`的同名段落记账，`import`进来的人从API面上读不出
"哪些能力不存在"。**两段必须同批改**。

## 本仓是什么

两个消费方在互不知情的情况下长出了同构的骨架纪律（契约版本冻结、输入内容寻址、
写后严格复读、验收预算30/120秒、"已知不成立"清单）。本仓把这套共同纪律先做成
**规范**，再逐步做成**共享代码**：

- `docs/spec/` — 统一接口规范：七条轴（02—08，v1.0已冻结）+ 模块地图（01）
  + 内核接口草案（10，场景装配与六接口，v0）+ 形状表示草案（11，模拟形状
  非制造实体，v0）+ 消费方登记（90）
- `docs/research/` — 调研正本（01：成熟引擎与五仓形状表示盘点）
- `docs/architecture.md` — 仓内依赖图与产品级工程设计/计算平台拓扑的总入口，不覆盖spec正本
- `docs/platform/` — RTime工程设计与计算平台愿景；工程模型、科学计算、工程工作流、
  视觉交互四个内核+Engineering Studio，设计系统、资产、插件、部署、仓库迁移与纵切路线；
  均为候选架构，未声明已实现
- `docs/plans/00_引擎开发路线_v0.md` — 继续开发从哪下手（M-E1契约基座→
  M-E2溯源→M-E3场景与接口）
- `docs/migration/` — 两个消费方的迁移操作单（各自会话自足执行）
- `docs/decisions/` — 本仓自己的决策记录（0001—0058，编号有缺口见0049第八节）
- `src/physics_engine/` — 共享引擎代码；进入前提见包文档字符串三条

**规范先行、代码后置**：任何一段代码进入本仓的前提，是对应规范已冻结、
且两个消费方的既有产物指纹在采纳后逐字节不变。

## 阶段阶梯

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 建仓、章程、统一接口规范总纲v0、远程与包骨架 | 2026-08-04完成 |
| 1 | 规范逐轴冻结——七轴全部v1.0：轴1版本冻结制、轴2稳定ID与单位量、轴3内容寻址、轴4 run package布局与生命周期、轴5原子发布与严格复读、轴6验收预算与元门禁、轴7 oracle纪律 | **2026-08-04完成** |
| 2 | 新代码先行：M-E1契约基座、M-E2溯源/run package、M-E3首轮形状/碰撞/场景/运动/执行器/传感接口均已落地；当前不再按旧M-E编号猜下一手，而以plans/09→plans/08阶段1的真实场景主线为准 | **进行中** |
| 3 | 存量迁移：消费方把既有引擎层改为依赖本仓，各自附逐字节不变证据；操作单在`docs/migration/` | **进行中**——`winding-deviation-sim`批次B已切换（2026-08-05凌晨，其design/29：钉`v0.4.0`、vendor相对路径+uv.lock哈希绑定、常驻切换门`test_support_engine_parity`；批次B余项逐块可回退）；`fts-digital-twin`闸门0已过（其ADR 0036）但引擎层未切换。逐行事实以[spec/90消费方登记](docs/spec/90_消费方登记.md)为正本 |

## 消费方采纳的治理前提

- `fts-digital-twin`：其ADR 0001（standalone边界）明文禁止兄弟项目依赖，
  采纳本仓前**必须先在其仓内以新ADR取代0001**，在FTS自己的会话与治理下完成。
- `winding-deviation-sim`：引擎层依赖本仓属动`src/`，须走其冻结diff→重签回执→
  全量门禁流程，并附现有案例产物指纹逐字节不变证据。

## 安装与调用（舰队wheelhouse链路，decisions/0010）

```bash
git clone ts-orangepi:wheelhouse.git ~/wheelhouse   # 每台机器一次，更新=git pull
uv add "physics-engine==0.5.0" --find-links ~/wheelhouse
```

调用示例：

```python
from physics_engine import FacetRegistry, canonical_sha256, verified_bytes_snapshot
from physics_engine.run_package import publish_package, read_verified_package
```

数据层入口（场景文件+一条命令，装完即有）：

```bash
pe-scene check-collisions 场景.scene.json --out-dir runs
```

API分两档（见包docstring）：稳定倾向=facets/canonical/identity/provenance/
run_package；实验档=shapes/collision/scene。**0.x语义：minor可破坏兼容，
patch只含兼容修复且不回port；消费方钉精确版本、经自己门禁自觉升级。**

**老版本共存（decisions/0012）**：wheelhouse已发版本永不覆盖、永不删除——
钉住旧版的程序永远装得上同一份字节；同机多版本靠各自venv隔离并行，
引擎不装共享Python。随手脚本用PEP 723内联声明+`uv run`（每台机器设一次
`UV_FIND_LINKS=~/wheelhouse`后全自动）：

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["physics-engine==0.5.0"]
# ///
```

稳定倾向档弃用给一版缓冲；变更必须写[CHANGELOG](CHANGELOG.md)（发版脚本
强制检查）。发版：`.venv/bin/python tools/release.py`（干净仓+accept全绿+
CHANGELOG条目才发，版本不可覆盖；有github远程时自动镜像推送）。
公开镜像：github.com/Rtiming/physics-engine（MIT）。

## 工程约定

跨设备范式按rtime-project：git同步（远程：orangepi裸仓库
`ts-orangepi:physics-engine.git（/home/orangepi/，与fts-digital-twin.git、case2-digital-twin.git同处）`，开源时另加GitHub）、
路径可移植（`tools/rtime-project-check.py --strict`必须全绿）、
行尾LF归一、提交署名`[ai:名@设备]`。文档中文。
