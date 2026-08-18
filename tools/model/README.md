# tools/model/ 模型资产工具的家

承接[decisions/0073](../../docs/decisions/0073_工具住哪与三维模型以什么形态进来_20260817.md)。
本目录是"三维模型怎么进来"那条裁决的落地处。

## 一、身份边界（这一节是本目录存在的全部理由）

这些工具在本仓的身份**只有一个**：**把外部模型资产翻译成本仓的声明**。
它们**不是内核的一部分**，产物是声明不是物理。

三条硬边界，与`validation/`（决策0025）**逐字同源**——不是新纪律，是复用一条已经过门的：

1. **永不进`dependencies`。** `pyproject.toml`的`dependencies = []`是0014立的承诺。
   本目录的工具连`[optional-dependencies]`都不进。
2. **永不被`src/physics_engine`import。** 一行都不许。
   自检：`grep -rn "tools\.model\|from tools" src/` 必须零命中，
   由`tests/governance/test_model_tools_stay_out_of_the_kernel.py`守着。
3. **重依赖的档独立venv。** `mesh_aabb.py`与`centerline_csv.py`是**纯标准库**、
   不需要venv；将来的`step_extract/`（OCCT或gmsh）只装进`tools/model/.venv`。

**wheel不含本目录**：`[tool.hatch.build.targets.wheel] packages = ["src/physics_engine"]`，
0.7.0实测48个条目全是`.py`、`.dist-info`之外零个非`.py`条目。
本目录在sdist内、wheel外——与`tools/`其余部分同样。

## 二、为什么工具在仓内而不是另建仓（0073第四节）

**同行证据支持"工具在仓内、依赖在外"，不支持另建仓**：

* SOFA把无头GUI管道留在仓内，**只把开窗GUI推到外部仓**；
* MuJoCo的`simulate/`在仓内，靠CMake选项挡住GLFW；
* **Drake拆出去的是资产不是工具**，而且是被PyPI 100MB硬限制逼的
  （PR #19160，`+714 / −1186238`行）——**本仓没有那个限制**。

**反面证据同样清楚**：Bullet issue #2918——仓297MB而物理库只占2.4%，
维护者的答复大意是"没法在不破坏所有已有clone和fork的前提下改写历史，只能忍着"。
**资产一旦进历史就是单向门。** Chrono仓内`data/`是1165MB。

**所以本目录永远不放真实资产。** 合成小资产（如`cases/mesh_asset_integrity/assets/tetra.stl`
那个284字节的四面体）是对的形态。

## 三、目录

```text
tools/model/
  README.md            本文件——身份边界
  mesh_aabb.py         STL/OBJ → 真实AABB + SHA-256（**纯标准库**）
  centerline_csv.py    22列CSV＋sidecar → GrooveStation元组（**纯标准库**）
```

`centerline_csv.py`（2026-08-18落地，0073第五节第2步）**守着一条数值上不报错的错**：
CSV的列序是`t → n → s`，而帧的基序是`t → s → n`——按列序读成基序，
每一步都还是单位向量、还是右手系，只是"带宽方向往哪边"整体错了90°。
GCW自己的sidecar用`csv_field_order_is_basis_order: false`说过这件事。
工具交出的元组**按基序排**，调用方一行`GrooveStation(*station)`即可，
不需要在两个仓之间对着列名重排——那正是最容易把`n`与`s`对调的地方。

**一条纪律**：`laydown`那五条语义**不许有默认值**（0067已裁），
工具把CSV变成元组，**语义仍由调用方显式给出**，工具不许替它拿主意；
闭合缺口同理**只报不裁**（真实语料首末差恰好一个采样步2.0016 mm，
是拓扑信息不是错误）。

**注错验证抓到自己两道空门**：`ordered_basis`判据与`t·n`正交判据各自有判据、
没有必红用例，整条拿掉后其余门全绿。两条用例已补
（正交那条的可达窗口只有`1e-9 < |t·n| ≲ 4.5e-5`，因为再大会先被单位向量判据挡住——
**窄不等于空**）。

**还没有的**（0073第五节，按依赖顺序）：

* `step_extract/`——从STEP取命名边并按弧长均布采样。
  **本仓不开工**：中心线的权威生产者是GCW，spec/11第三节明写不迁移其B-Rep管线。
  **但本仓要定它的输出契约**，即`centerline_csv.py`认什么。

## 四、这些工具补的是哪个洞

`cases/mesh_asset_integrity`那道"碰撞代理必须包住"的门，
**在已核查的同行里没有一家做**（SOFA的`SceneCheck*`零几何；
OpenUSD的物理validator四条全是结构性；MuJoCo、Chrono、Drake都没有）。
USD里`convexHull`加在凹网格上会**静默改变物体形状**而三个validator一个字都不说。

**但本仓那道门今天只覆盖一个语料**——仓内284字节的合成四面体。
`examples/`里的真实资产在消费方仓，案例页第四节明写"无字节可查、不判"。
**所以0017抓到的"包络是编的"那种缺陷，在真实资产上今天仍然没有门。**

`mesh_aabb.py`把"包盒是编的"从**人工填写**变成**工具生成＋案例门复验**。
