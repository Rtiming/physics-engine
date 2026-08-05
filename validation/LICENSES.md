# 同行库许可证结论

**本仓LICENSE是MIT**。MIT不与GPL/LGPL共存于同一个分发物——**GPL/LGPL类会传染**。
因此每个进`validation/`的库都必须在这里逐条结清：能不能装、能不能复制代码、能不能再分发。

WDS已有先例并已立边界（research/05第七节）：**GPL类来源只作对拍、不得复制代码，
可复制来源限MIT/Apache**。本页沿用同一条，并补一条更严的：
**本仓对同行库只调用、不移植——哪怕许可证允许**（轴7规则4：抄了实现，对拍就不独立了）。

## 一、逐库结论

| 库 | 版本 | 许可证 | 能装进`validation/`？ | 能复制其代码进本仓？ | 能随本仓再分发？ |
|---|---|---|---|---|---|
| python-fcl | 0.7.0.11 | BSD-3-Clause | ✅ | ⛔ **不复制**（许可证允许，纪律不允许） | ⛔ 不入库、不随wheel |
| FCL（`libfcl.0.7.0`，随轮） | 0.7.0 | BSD-3-Clause | ✅（随python-fcl的wheel带入） | ⛔ 同上 | ⛔ |
| libccd（`libccd.2.0`，随轮） | 2.0 | BSD-3-Clause | ✅（随轮） | ⛔ 同上 | ⛔ |
| OctoMap（`liboctomap`/`liboctomath` 1.9.8，随轮） | 1.9.8 | BSD-3-Clause（**仅core**；其可视化组件octovis是GPL-2.0，**未随轮**） | ✅（随轮） | ⛔ 同上 | ⛔ |
| numpy | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | ✅ | ⛔（本仓核心零依赖，与0016的加速档是两回事） | ⛔ |
| Cython | 3.2.9 | Apache-2.0 | ✅ | ⛔ | ⛔ |

**没有一个GPL/LGPL依赖进来**——本次对比实验不触发传染风险。
唯一需要盯的是OctoMap：它的core是BSD而octovis是GPL-2.0，两者同仓不同许可证。
本次装入的wheel**只含core的两个二进制**（已列名核对），octovis不在其中。

## 二、本地核验记录（2026-08-05，可复跑）

1. `python_fcl-0.7.0.11.dist-info/licenses/LICENSE`首行为`BSD 3-Clause License`，
   `Copyright (c) 2017, Matthew Matl`；METADATA的`License: BSD`与
   `Classifier: License :: OSI Approved :: BSD License`一致。
2. wheel内容清单（`zipfile.ZipFile(...).namelist()`）确认随轮二进制只有四个：
   `libfcl.0.7.0.dylib`、`libccd.2.0.dylib`、`liboctomap.1.9.8.dylib`、`liboctomath.1.9.8.dylib`。
3. 三个非python-fcl的dylib各自`strings | grep -ci gpl`结果均为**0**。
   这是**弱证据**（二进制里本来就不一定嵌许可证文本），只作反面排除，不作正面认定。
4. **诚实记一条缺口**：该wheel**没有为随轮的三个第三方二进制附各自的许可证文本**
   （`dist-info/licenses/`下只有python-fcl自己那一份）。因此上表中FCL/libccd/OctoMap
   三行的许可证结论来自**上游项目的公开声明**，不是从本地文件里读出来的。
   这不影响本仓——我们只在本机装、不再分发；但若将来要把同行库随任何产物分发，
   这三行必须先从上游取回许可证原文并入库。

## 三、入库素材的规矩

本次对比实验**没有向仓库复制任何同行库的素材**（无代码、无数据、无二进制）。
仓内只留：调用它的脚本、案例页、判据声明、以及manifest里的溯源字段。

将来若确需入库素材，一律附**来源URL + 版本 + SHA-256**，且来源许可证必须是MIT/Apache/BSD。
本次的锚点（虽未入库，仍在此备案，与`cases/peer_fcl_distance/criteria.json`一致）：

- 来源：https://pypi.org/simple （上游 https://github.com/berkeleyautomation/python-fcl ）
- 制品：`python_fcl-0.7.0.11-cp313-cp313-macosx_11_0_arm64.whl`
- SHA-256：`fb10b3037d82ef754ba236a94a7f12da1864e13d4bc2f0d1ae9078c9a814915f`
- 安装后二进制的逐个SHA-256写进每次对拍产物的manifest（`peer_library.bundled_binaries_sha256`）
  ——wheel名相同而编译产物不同是真实可能，只钉版本号钉不住。
