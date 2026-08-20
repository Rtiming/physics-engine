# physics-engine — AI协作规则

> 本仓遵循rtime-project跨设备范式。任何AI助手开工前先读本文件，
> 再读`README.md`与`docs/plans/01`置顶的**当前首要目标**。
> **引擎的开发就直接在本文件夹进行**（2026-08-05用户确认的工作模式）。

## 项目定位

- 一句话：rtime的物理引擎——规范先行、代码后置的多域仿真内核，**按用户级软件要求建设：目标机器状态未知，单机性能是一等产品需求（0014）**；
  消费方winding-deviation-sim（力学）、fts-digital-twin（光学）。
- 开发设备：Windows+macOS平级，直接在本仓工作；**重计算上master，不占本机Mac**（内部开发约定，非产品假设——产品定位见0014）。
- git远程：`ts-orangepi:physics-engine.git`（舰队主远程，/home/orangepi/，与fts-digital-twin.git同处）+ `github.com/Rtiming/physics-engine`（公开镜像，MIT）。发版脚本自动双推。 <!-- rtime-project: allow-abs -->

## 开工必读（顺序）

0. **[`docs/plans/09_交接_20260806`](docs/plans/09_交接_20260806.md)——最新交接页，
   先读它**：今天在哪、下一步做什么、三条教训、已知的坑、没做完的逐条；
1. `docs/plans/01_总执行计划与审核清单`——置顶首要目标+五波状态；
2. `docs/plans/00_引擎开发路线`——M-E里程碑到哪了；
3. 动哪个模块读哪份spec（`docs/spec/`地图在README）。
4. 动跨项目平台、前端运行时、Studio、多表示模型、资产转换、Blender或硬件适配时，另读
   [`docs/platform/README`](docs/platform/README.md)与
   [`docs/architecture`](docs/architecture.md)，再按任务读platform/00—05、
   [`research/14`](docs/research/14_多表示模型资产与硬件适配架构_20260809.md)与
   [`plans/10`](docs/plans/10_模型资产与呈现架构演进计划_20260809.md)；platform是上位
   产品提案，research/14与plans/10是资产/端点子系统资料，均不自动改写plans/08—09的
   当前物理主线或冻结spec。
5. 动绕制偏差物理基础设施、GCW/WII模型输入、张力测量或WDS采用时，另读
   [`research/19`](docs/research/19_绕制偏差物理基础设施与上游模型接入调研_20260820.md)、
   [`decision/0096`](docs/decisions/0096_绕制偏差物理基础设施归本仓与WDS单向消费边界_20260820.md)与
   [`plans/19`](docs/plans/19_绕制偏差物理基础设施与张力测量小场景计划_20260820.md)。
   物理基础设施归本仓；WDS只读消费，不改写GCW/WII；上游模型只通过版本化产物进入。

## 开发循环（快慢分层，服务"开发提速"首要目标）

- **内循环（秒级）**：`.venv/bin/python -m pytest tests/test_你在动的.py -q`；
- **批末**：`.venv/bin/python tools/accept.py full`（30/180双档、功能/计时/仓库稳定三轴正交——超时/漂移/零执行绝不pass）；
- **发版**：`.venv/bin/python tools/release.py`——干净仓+accept绿+CHANGELOG条目才发，**版本不可覆盖**，wheel入`~/wheelhouse`并自动镜像GitHub；
- 环境重建：`python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`。

## 代码三前提（decisions/0001，进仓硬门）

1. 对应规范轴已冻结（spec/02—08）或所属草案面已登记；
2. 有真实消费方在用（不为想象中的第三个消费方预支通用性）；
3. 消费方采纳后既有产物指纹逐字节不变。

## 本仓纪律（引擎专属）

- **零运行时依赖是承诺**：`dependencies = []`；新增运行时依赖须决策记录，dev依赖随意。
- **面清册义务**：任何跨边界字节形制先进`engine_facets.py`登记（draft出生），governance测试守回执面兼容。
- **每个门要红过**：新校验必须附"必须红"用例；判据本身也要被验（tests/governance/是样板）。
- **诚实可信度**：collision的confidence、shapes的凸性声明、后端确定性分级（spec/13）——不知道就说不知道，禁止冒充。
- **API两档**：稳定倾向档（facets/canonical/identity/provenance/run_package）破坏性改动须决策记录+弃用一版缓冲；实验档（shapes/collision/scene）minor内可破坏但CHANGELOG必写。
- 决策记录编号连续（docs/decisions/00NN）；文档中文，中英文之间不加空格，图注"图X-Y 描述"式。

## 性能条款（spec/13摘要）

优化先profile；声称结果不变附逐字节对拍；改数值路径附容差对拍；
fp64→fp32=换物理不是优化；GPU三门槛不过不动手；本机Mac无GPU求解器路线
（Metal无fp64，已测）。

## 路径策略与同步（范式硬性）

- 禁止新增写死的用户主目录/盘符绝对路径（`C:\Users\...`、`/Users/...`、`/home/...`、`/mnt/...`）；路径从仓库根/当前文件计算；跨仓引用用相对说明。 <!-- rtime-project: allow-abs -->
- 改完自检：`python3 tools/rtime-project-check.py . --strict`必须0错误。
  **在`.claude/worktrees/`里的并行开发副本内跑它等于空跑**——路径含`.claude/`
  被当忽略目录，实测"查了0个文件"（主仓同一命令查78个）。worktree内的"0错误"
  是空的，真验收要在主仓合入后跑一遍。
- **worktree内跑`accept.py`必须导出`PYTHONPATH="$PWD/src"`**：共享`.venv`里的
  editable安装指向**主仓**的`src`，不导出时新模块会报`No module named ...`而判FAIL，
  或者更糟——测的是主仓的代码而你以为测的是自己的。与上一条同源：
  **worktree里的绿和红都要先确认它测的是谁**。
- 代码只走git（不靠文件夹同步搬`.git`）；`.venv`/`work/`/`dist/`不进版本控制不同步。
- 不经用户明确指示不`git push`（发版脚本内的推送视为发版授权的一部分）。

## 多AI协作

- AI缓存（`.codex`/`.claude`）在仓库内但git/sync均排除；子代理只作建议，主代理拍板。
- 提交署名`[ai:名@设备]`；结构性大改动在`docs/decisions/`留审计痕迹。


## 并行 worktree 的三条环境陷阱（2026-08-05 实测，四条轨道各自撞上）

派活给 worktree 里的执行者之前，**派活方**必须处理这三条。
它们都不是执行者的问题，是派活方的环境没搭对。

1. **worktree 起点可能远落后主干**。2026-08-05 那一波实测**落后 52 个提交**——
   简报里点名的文件在起点上根本不存在。四条轨道各自 `reset --hard main` 才开工。
   **派活前先确认 worktree 的 HEAD。**
2. **worktree 内可能没有 `.venv` 符号链接**。不建就会**测到主仓的代码而不是自己的**——
   门全绿而验的是别人。简报里必须写 `export PYTHONPATH="$PWD/src"`，
   **并且派活前确认符号链接在**。
3. **`tools/rtime-project-check.py --strict` 在 worktree 内扫 0 个文件**。
   worktree 路径含 `.claude`，而那是检查器的跳过目录。
   **"0 错误 / 0 文件"不是证据，是空跑。** 真验收要么在主仓合入后跑，
   要么把树拷到 `.claude` 之外再跑。

**另加一条冲突面**（决策0041 清了三个共享文件，漏了这一类）：
**同点插入**。多条轨道往同一张表、同一个锚点各插一行时，
**分槽规则挡得住互改文本，挡不住同点插入**——2026-08-05 那一波因此冲突五次，
形态全部相同（保留双方即可）。**并行前预先分配表内行位。**
