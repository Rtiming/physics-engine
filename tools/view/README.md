# tools/view/ 查看器的家

承接[decisions/0074](../../docs/decisions/0074_真实网格进接触_碰撞归本仓_距离场是根本形制.md)第5.3／5.4节
与第六节**阶段五**，落位裁决见[decisions/0076](../../docs/decisions/0076_查看器进仓_轨迹是形制_rrd不是判据.md)。

## 一、身份边界（这一节是本目录存在的全部理由）

查看器在本仓的身份**只有一个**：**把一次运行画出来给人看**。
它**不是内核的一部分**，产物是一张图不是一条判据。

三条硬边界，与`validation/`（决策0025）、`tools/model/`（决策0073）**逐字同源**——
不是新纪律，是复用一条已经过两道门的：

1. **永不进`dependencies`。** `pyproject.toml`的`dependencies = []`是0014立的承诺。
   本目录的工具连`[optional-dependencies]`都不进。
2. **永不被`src/physics_engine`import。** 一行都不许。
   自检：`grep -rn "import rerun" src/` 必须零命中，
   由`tests/governance/test_view_tools_stay_out_of_the_kernel.py`守着。
3. **重依赖的档独立venv。** rerun只装进`tools/view/.venv`，主环境`.venv`保持
   只有pytest与ruff那几个dev依赖。`tools/view/.venv`已被仓根`.gitignore`的
   `.venv/`一行覆盖（实测），本目录另有一份`.gitignore`再写一遍，
   **好让本目录被单独拷走时边界仍然成立**。

**这三条在这里比在别处更硬，因为数更大**：`tools/view/.venv`装完实测**521 MB**，
rerun的wheel单个**133.6 MB**。而`tools/model/`那两个工具是纯标准库、
`validation/`的python-fcl不到它的零头。

**wheel不含本目录**：`[tool.hatch.build.targets.wheel] packages = ["src/physics_engine"]`。
本目录在sdist内、wheel外——与`tools/`其余部分同样。

## 二、为什么是rerun（0074第5.3节那张表的结论）

**三者里只有它同时具备我们真正要的三样**：

| 能力 | rerun | viser | meshcat |
|---|---|---|---|
| 三角网格／位姿时间线／点云 | 有 | 有 | 有 |
| **标量时间序列曲线**（张力波形） | 有（`Scalars`） | 有（uPlot集成） | **没查到** |
| **时间轴拖动回放** | 有（内建Timeline） | 没查到 | 没查到 |
| **一次run存成文件离线分享** | 有（`rr.save()` → `.rrd`，CLI可merge） | **没查到等价机制** | **没查到** |

**理由要写在这里，不是只写"选了rerun"**，所以逐行说清楚：

* **第三行是决定性的。** 本仓的产物要**能被别人在别的机器上打开**，
  而viser与meshcat都是"实时服务器＋浏览器"的形制，
  没有查到可分享的录制格式。一个只能"我这台机器现在开着"才看得到的查看器，
  在一个跨Windows／macOS／服务器三处开发的仓里等于没有。
* **第二行同样要紧。** 0074的原话：**没有波形就开发不了张力算法**——
  那是2026-08-17晚定下的目标甲。而本仓36个案例的`oracle.json`里
  `arrays`字段**实测全部为空**（`{}`），每一条金标都是标量。
  那是对的（判据本来就该可判、可给容差、可逐位对拍），
  **但标量判不了"它长什么样"**：一条`ISE = 0.0669 N²·s`的门告诉你压制够不够，
  不告诉你它是"峰高但衰减快"还是"峰低但一直不衰减"，而两者的ISE可以相同。
* **第一行三家都有，所以它不参与决策**——列出来是为了说明
  "选rerun不是因为它画得好看"。
* **许可证**：MIT OR Apache-2.0（实测`importlib.metadata`的`License-Expression`），
  与本仓的MIT相容。相容不等于可复制——本仓对rerun**只调用、不移植**。
* **可得性**：0.34.1有macOS arm64 wheel（`cp310-abi3-macosx_11_0_arm64`），
  实测在本机Python 3.13装得上。这一条是0074把`openvdb`判出局的同一把尺子
  （"出局（可得性，不是能力）"）——**对我们自己也得用同一把**。

**一条实测到的、0074没写的事**：rerun的CLI**默认开启匿名使用统计**。
`setup_view_env.sh`因此显式跑一次`rerun analytics disable`。
这不改本仓任何字节，改的是使用者自己的rerun配置。

## 三、目录与两个环境的分工

```text
tools/view/
  README.md                  本文件——身份边界与选型理由
  requirements-view.txt      钉死版本的依赖清单
  setup_view_env.sh          建环境脚本（POSIX，抄validation那份的形制）
  .gitignore                 .venv/ 与 *.rrd 都不进版本控制
  trace_from_closed_loop.py  **产**轨迹：physics_engine ＋ 标准库，**不认识rerun**
  replay.py                  **画**轨迹：rerun ＋ 标准库，**不认识physics_engine**
```

**两侧互不认识对方的依赖，中间只有一份JSON。** 这不是省事，是边界的形状：

* 装不上rerun的机器（网络、平台、Python版本任一原因）**照样能产轨迹**；
* 没有引擎的机器（比如要给别人看的那一台）**照样能画**；
* 而`src/`两边都不认识。

这条分工有一道门守着（governance那份的第四条判据）——
它一旦破了，上面三句话就只是README里的三句话，**而没有任何别的地方会红**。

跑一次端到端：

```bash
bash tools/view/setup_view_env.sh

# 产（主环境）
PYTHONPATH="$PWD/src" .venv/bin/python tools/view/trace_from_closed_loop.py \
    --band nominal_band --out work/view/nominal_band.trace.json

# 画（view环境）
tools/view/.venv/bin/python tools/view/replay.py \
    work/view/nominal_band.trace.json --out work/view/nominal_band.rrd

# 开窗看
tools/view/.venv/bin/rerun work/view/nominal_band.rrd
# 不开窗地验它读得回来
tools/view/.venv/bin/rerun rrd verify work/view/nominal_band.rrd
```

## 四、`.rrd`不是判据

本目录的产物**永不进任何门**。理由是本仓一条既有纪律
（spec/08规则1：实测数不作金标）在查看器上的直接推论——
**一个查看器同时当判据，等于让被验的东西自己出考卷**。

`cases/closed_loop_tension_step`那三条门（峰值／ISE／稳态）判的是
`generate_oracle.py`那份**独立闭式解**，与本目录零交集。
本目录画的是同一次运行，**但它画错了不会有人红**——所以它不许被信。

一条实测过的旁证：演示轨迹跑出来的峰值是`1.1375139838369677` N，
案例清单里的闭式金标是`1.137505182988744` N，相对差`7.7e-6`——
那是内核一阶积分器对精确解的离散误差，正是案例门本来就在容忍的量。
**这个数说明轨迹画的确实是门在判的那条运行**，但它**不代替**那道门。

**还有一条更硬的理由，是本轮实测撞出来的**：同一份轨迹连画两次，
`.rrd`**不是逐字节一致的**（两次SHA-256实测不同）——记录里带`log_time`
与随机的`recording_id`。而本仓代码三前提第3条是"消费方采纳后既有产物指纹
**逐字节不变**"。**一个指纹不稳的产物在结构上就当不了判据**，
这不是纪律选择，是它自己的性质。

顺带一条落盘纪律：**`rr.save()`返回不等于文件写完**。
实测紧接着`stat()`拿到321475字节，而进程退出后是343875字节——差22400字节，
因为rerun的落盘是异步的。`replay.py`因此在`save()`之后显式
`rerun_shutdown()`再报数，**好让回执里那个字节数是真的**
（与`run_package.publish_package`那套"写完复验哈希"同源）。

## 五、`engine_run_trace`形制（**草案，未进面清册**）

`src/physics_engine/engine_facets.py`是本仓的面清册，
"任何跨边界字节形制先登记再落盘"是硬纪律（AGENTS.md／决策0017）。
**本形制今天没有登记**，理由与做法：

* 本轨的卖点之一是**它不吃源码预算**，`src/`零字节改动；
  而`engine_facets.py`在`src/`里。**偷偷加一个进去比不加更坏。**
* 所以形制写在这里作为**草案**，落盘的字节里`"facet": "engine_run_trace"`
  照写——**这样它一旦升进清册，已有的轨迹文件不用改一个字节**。
* **要不要升进`engine_facets`由后续批次裁**（0076第五节登记了这条待裁项）。
  升，就补一条`ENGINE_RUN_TRACE_FACET`出生draft；
  不升，就把它明确记成"工具间的私有形制，不作跨边界承诺"。
  **今天两者都没有裁，本节就是这个状态的如实记录。**

### 5.1 字段

```jsonc
{
  "facet": "engine_run_trace",          // 必填，不是这个名字即抛
  "facet_version": "0.1",               // 大版本不同即抛，不做"尽量读"
  "run_id": "…",                        // rerun的application_id
  "producer": {…},                      // 谁产的（自由字段，只作记录）
  "units":    {"length": "mm", "time": "s", …},   // **必填，无默认**
  "timeline": {"name": "sim_time", "unit": "s", "times": [...]},
  "sampling": {"source_step_count": 200000, "source_dt_s": 1e-6,
               "stride": 200, "kept": 1000,
               "undecimated_extrema": {…}},       // **stride必填**
  "notes":    ["…"],                    // 贴到记录根上的静态说明
  "geometry": [{"entity_path": "…", "kind": "mesh",
                "synthetic": true,      // **必填，无默认**
                "vertex_positions": [[x,y,z], …],
                "triangle_indices": [[i,j,k], …]}],
  "poses":    [{"entity_path": "…",
                "translations": [[x,y,z], …],          // 与times等长，否则抛
                "quaternions_xyzw": [[x,y,z,w], …]}],  // 与times等长，否则抛
  "scalars":  [{"entity_path": "…", "unit": "N",       // **unit必填**
                "values": [...]}],                     // 与times等长，否则抛
  "points":   [{"entity_path": "…",
                "frames": [[[x,y,z], …], …],           // 与times等长，否则抛
                "radii": [...], "colors": [[r,g,b], …]}]
}
```

### 5.2 三条"没有默认值"是从本仓踩过的坑里来的

1. **`units`缺席即抛。** 与`tools/model/mesh_aabb.py`同一条纪律
   （"本工具不做单位换算，只如实报字节里的数"）。
   一个把mm当m画的查看器，**出来的图看着还很合理**——
   那正是本仓最怕的那种"跑得通但全错"。
2. **`synthetic`缺席即抛。** 0017抓到的真实缺陷是
   `examples/collision_preview_cell.scene.json`里声明的AABB是**编的**，
   与真实包盒在z轴上完全不相交。那条缺陷活得久，是因为**看上去是有东西的**。
   查看器把这个风险放大一个量级：一个合成圆柱画在屏幕上，
   与真实资产的网格**长得一样可信**。
   所以这一条比"留空比填默认值诚实"更强——**它不许留空**。
3. **`stride`缺席即抛，且必须配`undecimated_extrema`。**
   演示轨迹的源是20万步，抽样是必须的，**而抽样会漏掉峰值**——
   那正是本案例三条判据里的第一条。所以未抽样的极值单独算一遍写进字节，
   由`replay.py`贴成静态`TextDocument`：
   **曲线上看不到的那个峰，文字里写着。**

### 5.3 逐帧序列必须与时间线等长

少一帧就画一帧，出来的图会把整条曲线在时间上**错位，而它看起来完全正常**。
所以`replay.py`**在画任何东西之前**把形制验完再动手——
半成品`.rrd`能打开，所以它比抛异常更坏。

## 六、这个工具补的是哪个洞

本仓今天**一条时间序列产物都没有落盘**：36个案例的`oracle.json`里`arrays`
字段实测全是`{}`。`tools/view/trace_from_closed_loop.py`产的是**第一条**。

**还没有的**：

* **接触点今天是几何构造的，不是接触求解器吐的。** 演示轨迹里那两个点是
  带材离开放线盘与落到收线盘的切点，按声明的半径算出来——
  轨A（0074阶段一）的槽壁接触、轨B（阶段四）的距离场接触项落地后，
  `points`块应当改喂**真实接触点**，那时`engine_run_trace`才算完整。
* **真实网格资产还没有一份进来。** `geometry`块今天两条都是`synthetic: true`。
  `tools/model/mesh_aabb.py`已经能把STL翻成声明，
  但把网格字节本身搬进轨迹是另一件事（体积——见0073第二节Bullet那297 MB的账）。
* **多次run的对比**（rerun CLI有`rrd merge`）没做。
