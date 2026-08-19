# 运行入口一页表

**这一页回答一个问题：想验证某件事，跑哪条命令、期望看到什么。**

立它的理由是2026-08-18实测出来的：本仓的入口散在`tools/`、`validation/`、
`cases/`与四个环境变量里，而**没有任何一个地方能让人一眼看完**。
后果不是不方便——是**有三条通道从来没有被整体跑过一次**
（见第三节），而`accept full`只会报skipped，**skip在回执里看起来和pass一样绿**。

> 本页只列**怎么跑**与**期望什么**。为什么这么设计在各自的决策记录里，逐条给了链接。

## 一、日常两条

| 想干什么 | 命令 | 期望 |
|---|---|---|
| **改完代码的内循环**（秒级） | `.venv/bin/python -m pytest tests/test_你在动的.py -q` | 绿 |
| **本机快档**（约32秒） | `.venv/bin/python -m pytest tests -q -m "not batch and not serverclass"` | 全绿，5条skip各带理由 |

## 二、验收：**墙钟以master为准，不以本机为准**

| 想干什么 | 命令 | 期望 |
|---|---|---|
| **全档验收（权威）** | `bash tools/master/run_accept_on_master.sh full` | `overall=PASS functional=PASS timing=PASS perf=EVALUABLE resource=QUALIFIED`，墙钟约110秒/180 |
| **本机全档**（只作参考） | `.venv/bin/python tools/accept.py full` | 四轴绿，但**这台Mac整天负载5—20，本机墙钟不进台账** |
| **在master跑任意一条命令** | `bash tools/master/run_on_master.sh '<命令>'` | 回执带节点、负载、核数 |

**为什么墙钟只认master**：plans/16的M8立的口径。同一功能面在本机实测到过
93.7／121.5／188.8／225四个数，而在master上三次跑离散度<1%。

## 三、选择进入档：**真实资产永不进仓（0073），所以要指给它**

```bash
python tools/verify_optin.py --search-root <你的lab根>
```
**期望**：三条通道逐条报"跑了还是跳了、以及为什么"，末尾一次pytest全绿。
一条都没解析到时**返回码2**——空跑不是通过。

| 变量 | 指向什么 | 谁在用 |
|---|---|---|
| `PE_REAL_CENTERLINE_CSV` | GCW导出的`centerline.csv`**或它们的一个目录**（同目录要有`centerline.meta.json`） | 真实工件的中心线不变量、模型工具 |
| `PE_REPLAY_CASE_RUNS` | 消费方已发布的run目录树 | 机械复读与篡改拒收 |
| `PE_REPLAY_OUTPUT_TREE` | 消费方的output树 | run package的装配与manifest绑定 |

## 四、重依赖档：**三个独立venv，`src/`一个都不import**

| 档 | 建环境 | 跑一次 | 它是什么身份 |
|---|---|---|---|
| **同行库对拍**（python-fcl） | `bash validation/setup_peer_env.sh` | `validation/.venv/bin/python validation/peer_fcl/run_comparison.py --out-dir work/peer_fcl --name run-001` | **验证期的证人，不是依赖**（0025） |
| **SDF烘焙**（point-cloud-utils） | `bash tools/model/sdf_bake/setup_sdf_bake_env.sh` | `PYTHONPATH=$PWD/src tools/model/sdf_bake/.venv/bin/python tools/model/sdf_bake/bake_sphere_probe.py --out work/sdf_bake/sphere_probe.json` | 离线烘焙，**内核只吃数字**（0074第二节） |
| **查看器**（rerun） | `bash tools/view/setup_view_env.sh` | 产：`PYTHONPATH=$PWD/src .venv/bin/python tools/view/trace_from_closed_loop.py --band nominal_band --stride 200 --out work/view/x.json`<br>画：`tools/view/.venv/bin/python tools/view/replay.py work/view/x.json --out work/view/x.rrd` | **两个环境互不认识对方的依赖，中间只有一份JSON**（0076） |

**同行库/烘焙档缺席时对应的门skip而不是fail**——它们是证人不是依赖，
**但skip必须说清是哪一步缺**，否则"永远绿"和"永远skip"在回执里长得一样。

## 五、度量与画像

| 想干什么 | 命令 | 备注 |
|---|---|---|
| 性能预算报告 | `.venv/bin/python tools/bench.py` | 产`work/bench/latest.json` |
| 热点画像 | `bash tools/master/run_on_master.sh 'python tools/profile_hotspots.py suite --marker batch'` | **在master跑**——本机负载会让占比失真 |
| 收敛阶扫描 | `bash tools/master/run_on_master.sh 'python tools/convergence_order.py run --problem all --levels 6 --scale full'` | 先`verify`自验判据 |

## 六、门与清册（`accept full`会跑它们，也可以单跑）

| 命令 | 它守什么 |
|---|---|
| `python3 tools/rtime-project-check.py . --strict` | 跨设备可移植性。**在`.claude/worktrees/`里跑等于空跑**（路径含`.claude`被跳过） |
| `.venv/bin/python tools/check_capability_ledger.py` | 能力位分子从清单算，**并对账README与案例索引里手写的那几个分数** |
| `.venv/bin/python tools/check_case_pages.py` | 案例页六字段、索引表、分层表三处一致 |
| `.venv/bin/python tools/check_gap_register.py` | 每份登记了欠账的决策记录都进了缺口清册 |
| `.venv/bin/python tools/check_conflict_markers.py` | 冲突标记 |
| `.venv/bin/python -m pytest tests/test_case_generators_reproduce.py -m batch` | **40个案例生成器重跑得动、且落在各自声明的容差内** |

## 七、发版（**只有干净仓＋accept绿＋CHANGELOG有条目才准发**）

```bash
.venv/bin/python tools/release.py
```
**版本不可覆盖**；wheel入`~/wheelhouse`并自动镜像GitHub。
**不经用户明确指示不跑它**（AGENTS.md）。
