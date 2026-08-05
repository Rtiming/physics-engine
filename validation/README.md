# validation/ 同行库对比实验的家

承接decisions/0015张力B与[decisions/0025](../docs/decisions/0025_同行库对比实验的身份与合规_20260805.md)。
本目录是"把别人的库下载下来进行对比实验"这条裁决的落地处。

## 一、身份边界（这一节是本目录存在的全部理由）

同行库在本仓的身份**只有一个**：**验证期的oracle与对比证人**。

三条硬边界，破一条就等于把引擎的零依赖承诺换掉了：

1. **永不进`dependencies`。** `pyproject.toml`的`dependencies = []`是0014立的承诺
   （破坏即破坏性变更）。同行库连`[optional-dependencies]`都不进——加速档那条口子
   （0016给NumPy开的）是为**本仓自己的数值内核**开的，不是给同行库的。
2. **永不被`src/physics_engine`import。** 一行都不许。要验的东西自己算，
   同行库的数只在`validation/`里出现。
   自检：`grep -rn "import fcl\|import trimesh" src/` 必须零命中。
3. **独立目录、独立venv。** 同行库只装进`validation/.venv`，主环境`.venv`保持
   只有`pytest`与`ruff`两个dev依赖。`validation/.venv`已被仓根`.gitignore`的
   `.venv/`一行覆盖，不进版本控制。

再加一条来自轴7规则4的纪律：**不得抄同行的实现进内核**。抄了，对拍就不独立了。
本仓对FCL的使用**只有调用，没有阅读其源码后的移植**。

## 二、目录

```text
validation/
  README.md            本文件——身份边界与建环境
  LICENSES.md          逐库许可证结论（硬要求：本仓LICENSE是MIT）
  requirements-peer.txt 钉死版本的依赖清单
  setup_peer_env.sh    建环境脚本（POSIX；Windows见第三节）
  peer_fcl/
    harness.py         算子层：输入生成、两侧适配、逐点比对（语义映射表在文件头）
    run_comparison.py  跑一次对拍，产物走run package落盘
```

案例页与判据在仓内另一处：`cases/peer_fcl_distance/`（`case.md` + `criteria.json`）。
conformance门在`tests/cases/test_peer_fcl_distance.py`——**同行库缺席时skip并给理由**，
本仓的accept绝不因为一台机器没装同行库而红。

## 三、建环境

```bash
# POSIX（macOS/Linux）
bash validation/setup_peer_env.sh

# Windows PowerShell（脚本未提供，三条命令等价）
python -m venv validation\.venv
validation\.venv\Scripts\python -m pip install --upgrade pip
validation\.venv\Scripts\pip install -r validation\requirements-peer.txt
```

用与主环境同一个Python小版本（本机是3.13）——`run_comparison.py`要在同行环境里
import`physics_engine`（纯Python零依赖，走`sys.path`，**不安装**）。

跑一次对拍：

```bash
validation/.venv/bin/python validation/peer_fcl/run_comparison.py \
    --out-dir work/peer_fcl --name run-001
```

产物落`work/`（已被`.gitignore`排除）。**产物不入库**：它带机器指纹（平台、BLAS、
二进制哈希），入库等于把一台机器的状态当成仓库事实。要留证据就留manifest里的
那几行溯源字段，或按轴5另行归档。

## 四、换版本/换库时必须重做的事

同行库的版本在`cases/peer_fcl_distance/criteria.json`里钉死，conformance测试会核对。
版本一动，**下面四步一步都不能省**：

1. **重跑语义复核**：同行库的同名API在不同版本里可以是不同算子。本案例实测到FCL的
   三条距离路精度差六个量级（case.md第三节），这种事只能靠实测，不能靠文档；
2. **更新`expected_peer_operator`路由表**并说明为什么变；
3. **重新算容差**：容差的理由必须重写，不许沿用上一版的数字；
4. **更新`LICENSES.md`**：新版本可能换了随轮的第三方二进制（FCL的wheel里就捆了
   libccd与octomap两家）。

这四步不是流程洁癖：**同行漂了而我们自动继承结论，等于把别人的回归当成我们的证据**。

## 五、候选库与实际结论（2026-08-05实测）

| 候选 | 结论 |
|---|---|
| **python-fcl** | ✅ **已装并跑通**，0.7.0.11，PyPI有macOS arm64 / cp313预编译wheel，`pip install python-fcl`直接成功，无需本地编译FCL/libccd。它的距离查询与本仓的`segment_segment_distance_mm`直接对应，是首选 |
| trimesh（proximity） | 未装。它的proximity面向网格，与本仓今天的解析球/胶囊族不对应；且其精确查询链路要`rtree`/`embreex`等额外二进制。**首选已跑通，不为凑数再装一个** |
| libccd | 未单独装——**它已经在python-fcl的wheel里**（`libccd.2.0.dylib`），本案例正是通过FCL的libccd路径间接测到了它（并抓到了它的失效面，见case.md第五节） |
| scipy | 未装。退而求其次的几何对拍在本案例里已无必要 |

未装的三个都是**主动不装**，不是装不上：首选一次成功，再堆库只会增加环境维护面
而不增加证据强度。要扩到圆柱/盒/网格时（plans/02 T2），trimesh是第一顺位候选。
