# tension_measuring_roll_resultant T-M0：测力轮两侧张力的矢量合力

判据正本为同目录`oracle.json`，生成器为`generate_oracle.py`。本案例验证
`physics_engine.tension_measurement`的第一层物理，不调用LTS电气链。
清单身份：`case/tension_measuring_roll_resultant`；负载级：`interactive`。

## 一、物理/几何设定

入口与出口切向都按带材行进方向定义。入口方向固定`(1,0,0)`，出口分别转0°、60°、
90°和180°；两侧张力均为17N，所有向量在xy平面，tare为0，单支承份额为1。

| 量 | 值/单位 |
|---|---|
| 入口张力 | 17N |
| 出口张力 | 17N |
| 包角 | 0°、60°、90°、180° |
| 敏感轴 | 本案例只判轮上合力，不接传感器 |
| 重力/tare | 0N |

## 二、参考解出处

静力平衡：入口带段对轮的力为`-T_in*t_in`，出口带段为`+T_out*t_out`。
等张力时，合力模长退化为：

```text
F = 2*T*sin(beta/2)
```

Maxcess《TLC Thin Load Cells》选型页与ABB Pressductor PFTL 101手册的测力矢量图均使用
同一关系；综述与完整著录见`docs/research/19`参考文献[18-19]。生成器只写静力式，
不import`physics_engine.tension_measurement`。

## 三、判据表

| 量 | rel/abs | 理由 |
|---|---|---|
| 合力x/y分量 | abs 4e-15N | 一次乘加约2 ulp；符号反了会差几十牛顿 |
| 合力z分量 | 零容差 | 平面输入按构造不产生z力 |
| 合力模长 | rel 4e-16/abs 4e-15N | 对独立`2*T*sin(beta/2)`闭式；同时覆盖零包角 |
| 0°与180°端点 | 零/2T | 两端定性相反，能抓把包角写成补角的实现 |

## 四、已知失效清单

1. 不含导轮自重、敏感轴、支承和电气链；这些由T-M1/T-M2负责。
2. 不含摩擦；两侧张力被显式给定，不从绞盘或输运求出。
3. 不含动态；只验证一个时刻的测力轮静力。
4. 不对应现场LTS安装，证据永久`hypothesis_only`。
5. 无skip。

## 五、档位与负载级

**A档/交互级（`interactive`）**。只有四个常数构型，必须在靶向内循环中完成。

## 六、本案例不是什么

- 不是张力传感器标定；
- 不是T0—T4观测器；
- 不是放线—收线动态；
- 不是现场LTS读数已经正确；
- 不是WII/GCW模型接入。
