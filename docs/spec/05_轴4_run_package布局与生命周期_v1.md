# 轴4：run package布局与生命周期（v1.0，2026-08-04冻结）

状态：**已冻结**（决策记录[0005](../decisions/0005_冻结轴3至5_20260804.md)）。
版本语义按[轴1](02_轴1_分面版本冻结制_v1.md)。

证据：WDS `storage/case_runs.py`（发布）、`storage/case_run_reader.py`（复读）、
`contracts/case_runtime.py:188-257`（状态机）；FTS `contracts/io.py:42-51,182-236`
（六文件闭包）、`io/run_package.py:731-805`（写序）、`contracts/state.py:644-731`
（manifest自指封顶）。

## 一、五条规则

1. **唯一运行目录，身份进目录名**。目录路径含案例与运行身份
   （WDS `<case_id>/<run_id>/`双段绑定），复读时目录名与manifest身份比对，
   改名即拒。排他创建：目录已存在即失败（`mkdir(exist_ok=False)`/preflight+
   no-replace），不抢占、不覆盖。
2. **文件闭包精确声明**。manifest列出全部载荷文件（路径、大小或哈希）；
   复读时目录内直接常规文件集合必须**精确等于**声明集合——缺一、多一、
   子目录、符号链接、设备文件全部拒收（两仓同：WDS
   `initial_names != expected_names`拒；FTS `set(names) != PACKAGE_ROOT_FILES`拒）。
3. **manifest最后写，是终态化动作**。manifest的出现即声称"集合完整"，
   因此必须在全部载荷落盘之后写入；manifest不必列自身，但必须能封顶其余
   文件（哈希表或自指哈希）。非终态生命周期不得落盘为已发布运行
   （WDS发布器只接受completed/failed/cancelled）。
4. **生命周期失败关闭**。最低状态集：排队/运行中/完成/失败/取消。
   排队与运行中**不得声称任何结果**；完成态必须持有可验证结果**或**求解失败
   诊断——**数值失败是一等结果**，出诊断、禁结果，二者互斥；失败态分阶段
   （构建/求解/发布/取消），各阶段对求解状态的约束显式声明（WDS
   `_lifecycle_is_fail_closed`是参考语义）。时间戳UTC。
5. **路径纪律**。包内路径单段basename；禁绝对路径、`..`、`.`、反斜杠、
   多段、Windows保留设备名；大小写折叠后不得与manifest撞名。

## 二、验收判据（门必须红过）

复读侧各造一例必须拒：缺文件、多文件、子目录、符号链接、目录改名、
非终态manifest、"运行中却带结果"的manifest（契约校验拒）、
converged却无result、numerical_failure却带result。

## 三、两个创始消费方的已知形制差（均合规）

| 项 | WDS | FTS |
|---|---|---|
| manifest封顶方式 | manifest列artifact哈希表；目录闭包由reader按layout推导 | manifest不列自身，`files`表五文件+两级自指哈希封顶 |
| 结果/诊断 | 互斥两文件，按场景layout声明 | 单一六文件固定集合，assessment内含claims_boundary |
| 状态机 | 五状态+四失败阶段，单验证器强制 | 无独立状态机（包只有终态），非终态由"写不完整→加载必拒"承担 |

FTS无显式非终态状态机是其单进程写路径的合理省略；若其将来出现异步/排队运行，
须补状态机或在采纳声明中声明等效机制。
