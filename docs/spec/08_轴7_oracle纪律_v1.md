# 轴7：oracle纪律（v1.0，2026-08-04冻结）

状态：**已冻结**（决策记录[0006](../decisions/0006_冻结轴6至7_20260804.md)）。
版本语义按[轴1](02_轴1_分面版本冻结制_v1.md)。

证据：FTS `oracles/m0-candidate/manifest.json`（8条oracle、逐条expected+
tolerances、数组双哈希、生成器algorithm_id）、`tests/oracles/test_m1_oracle_conformance.py`
（生产内核对拍）、其AGENTS"M0-M3 use only repository-owned synthetic scenes
and golden data"与"Do not weaken tolerances, delete tests, or regenerate golden
answers to make a change pass"；WDS AGENTS（M2四解析基准是任何模型改动的
回归门禁、每个能量/接触项必须带有限差分一致性测试、PyElastica对拍独立性——
"抄了实现那个对拍就不独立"）。

## 一、六条规则

1. **金标来自仓内解析oracle**。闭式解或直算f64参考实现，仓库自有、
   随仓版本化；**实测数据不作金标**（实测属标定与验收，是另一层）。
2. **oracle清单是一个面**。最低字段：生成器身份（algorithm_id+路径）、
   逐条expected与**逐条tolerances**、数组哈希（语义级+raw级）、清单自指哈希。
   清单按轴1登记、按轴3寻址。
3. **conformance用生产内核对拍**。测试调被验的生产代码算答案、与冻结oracle
   按清单里的容差比对；**容差从清单读取，不得在测试里私改**；
   不得在测试里复述oracle公式（那验证的是复述不是内核）。
4. **独立性**。oracle生成实现与被验内核不得共享代码路径；外部库作对拍源时
   不得抄其实现进内核，否则对拍失去独立性。
5. **金标冻结**。不得为让改动通过而放宽容差、删除测试或重生成金标；
   重生成金标必须走决策记录+版本跳变（轴1规则6的联动在此处最锋利）。
6. **物理改动的最低门**。任何物理内核改动至少过：解析oracle对拍+
   （有梯度/Hessian的）有限差分一致性+既有金标回归。每道物理门要有
   "它必须红"的输入——故意调错一个物理常数，门必须红过一次。

## 二、验收判据（门必须红过）

- 故意错的物理（错符号/错常数）→oracle门必须红（每仓至少验证过一例并留记录）。
- 无决策记录的金标重生成→轴1联动门红（依赖各仓补齐轴1规则6后生效）。
- 容差在测试内被覆写→评审判据（conformance测试的容差来源必须指向清单）。

## 三、两个创始消费方的已知缺口

| 仓 | 已有 | 缺口 |
|---|---|---|
| FTS | 规则1—6满配（oracle清单形制是规则2的样板；G3精确字节门+跨平台portable分离见其ADR 0028） | 无 |
| WDS | 规则1、3、4、6实践在位（M2四解析基准、有限差分门、PyElastica独立对拍、罚接触侵入门首跑即红的先例） | **规则2**：解析基准散在tests与benchmarks，无统一oracle清单面（expected/tolerances未集中成带哈希的清单）——采纳时建清单或声明偏差；规则5的"重生成须决策记录"依赖轴1联动门落地 |
