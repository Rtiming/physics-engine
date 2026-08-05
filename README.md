# physics-engine

rtime的**物理引擎**：多域物理（力学、光学，将来热/电磁）+ 模型生成 + 物理性质
（材料）+ 外观（颜色）+ 溯源纪律的仿真引擎仓。2026-08-04用户裁决建立并定位
（原名twin-engine，同晚更名）：走共同仓库、深度融合、接口规范统一。

创始消费方：`winding-deviation-sim`（力学，绕制偏差）与`fts-digital-twin`
（光学，傅里叶光谱仪）；潜在第三个消费方`case2-digital-twin`（专利装置数字样机）。
**用户级定位（decisions/0014）**：目标机器状态永远未知——单机冷启动性能是第一指标（import 110ms、场景校验0.22s、验收full 3.4s、wheel 30KB离线装）、功能操作不因宿主负载拒绝或变形、零设施假设是承诺。

目标模块图与各能力现居地见[docs/spec/01_模块地图与域划分_v0.md](docs/spec/01_模块地图与域划分_v0.md)。

## 能力边界（诚实条款）

**能做**：声明形状与材料（`shapes`/`materials`）；算解析原语的质量属性——体积、
质心、绕质心惯量张量（`geometry`）；查碰撞——broad phase全域 + 球/胶囊族narrow phase
（`collision`/`scene`，其余族诚实保留`broad_phase`可信度）；装配能量项——均匀重力、
轴向拉伸、小挠度弯曲（`energies`）；求准静态平衡——牛顿+回溯线搜索+稠密LU（`solve`）；
推时间积分——显式Euler／半隐式辛Euler／velocity Verlet（`integrate`）。

**不能做**：**没有隐式时间积分族**；**没有几何精确（DER）弯曲**——只有小挠度
Euler-Bernoulli；**没有扭转**；**没有接触与摩擦**；**没有约束**——边界条件只有
"显式钉住某些自由度"这一种；**没有光学域**——spec/01模块地图里它至今是空的。
三个积分器的`production_ready`**全为`False`**且有门守着。
力学与光学按decisions/0015仍在搬迁中，路线见
[docs/plans/02](docs/plans/02_真引擎路线图_轨道与案例_20260805.md)。

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
- `docs/plans/00_引擎开发路线_v0.md` — 继续开发从哪下手（M-E1契约基座→
  M-E2溯源→M-E3场景与接口）
- `docs/migration/` — 两个消费方的迁移操作单（各自会话自足执行）
- `docs/decisions/` — 本仓自己的决策记录（0001—0006）
- `src/physics_engine/` — 共享引擎代码；进入前提见包文档字符串三条

**规范先行、代码后置**：任何一段代码进入本仓的前提，是对应规范已冻结、
且两个消费方的既有产物指纹在采纳后逐字节不变。

## 阶段阶梯

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 建仓、章程、统一接口规范总纲v0、远程与包骨架 | 2026-08-04完成 |
| 1 | 规范逐轴冻结——七轴全部v1.0：轴1版本冻结制、轴2稳定ID与单位量、轴3内容寻址、轴4 run package布局与生命周期、轴5原子发布与严格复读、轴6验收预算与元门禁、轴7 oracle纪律 | **2026-08-04完成** |
| 2 | 新代码先行：M-E1契约基座已落地（facets/canonical/identity+自吃药验收器，52测试全绿）；M-E2溯源、M-E3接口待需求牵引 | **进行中** |
| 3 | 存量迁移：消费方把既有引擎层改为依赖本仓，各自附逐字节不变证据；操作单在`docs/migration/` | **进行中**——`winding-deviation-sim`批次B已切换（2026-08-05凌晨，其design/29：钉`v0.4.0`、vendor相对路径+uv.lock哈希绑定、常驻切换门`test_support_engine_parity`；批次B余项逐块可回退）；`fts-digital-twin`闸门0已过（其ADR 0036）但引擎层未切换。逐行事实以[spec/90消费方登记](docs/spec/90_消费方登记.md)为正本 |

## 消费方采纳的治理前提

- `fts-digital-twin`：其ADR 0001（standalone边界）明文禁止兄弟项目依赖，
  采纳本仓前**必须先在其仓内以新ADR取代0001**，在FTS自己的会话与治理下完成。
- `winding-deviation-sim`：引擎层依赖本仓属动`src/`，须走其冻结diff→重签回执→
  全量门禁流程，并附现有案例产物指纹逐字节不变证据。

## 安装与调用（舰队wheelhouse链路，decisions/0010）

```bash
git clone ts-orangepi:wheelhouse.git ~/wheelhouse   # 每台机器一次，更新=git pull
uv add "physics-engine==0.4.0" --find-links ~/wheelhouse
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
# dependencies = ["physics-engine==0.4.0"]
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
