# tension_measuring_roll_installation T-M1：敏感轴、tare与支承分配

判据正本为同目录`oracle.json`。本案例接在T-M0之后，验证同一轮上合力如何被安装方向、
导轮静载和支承结构映射成传感器轴向力。
清单身份：`case/tension_measuring_roll_installation`；负载级：`interactive`。

## 一、物理/几何设定

带材入口沿+x、出口沿+y，两侧各10N，因此web合力为`(-10,10,0)N`。

| 构型 | 敏感轴 | tare | 支承份额 |
|---|---|---|---|
| 单支承对准 | 对准web合力 | 0N | 1.0 |
| 双支承对准 | 对准web合力 | 沿合力5N | 0.5/0.5 |
| 非对称支承+轴偏30° | 合力方向向横向旋30° | 合力向5N+横向2N | 0.3/0.7，显式输入 |

本案例中的非对称份额是**已知输入**，不是由引擎从“两个支承”猜出来的。

## 二、参考解出处

每个构型只做三步独立刚体静力：

```text
gross_force = web_force + tare_force
axis_force = dot(force, sensor_axis)
support_force[i] = axis_force * declared_share[i]
```

ABB PFTL 101手册明确传感器只测量其敏感方向的力分量；Maxcess选型页把web合力、
导轮重量和安装角分开进入载荷计算。完整著录见`docs/research/19`参考文献[18-19]。
生成器不import被验模块。

## 三、判据表

| 量 | rel/abs | 理由 |
|---|---|---|
| gross轴向力 | rel 4e-16/abs 8e-15N | 矢量和后投影，第三构型非轴对齐 |
| tare轴向力 | rel 4e-16/abs 4e-15N | 必须独立计算，不能把gross与tare共同错掉 |
| net轴向力 | rel 4e-16/abs 8e-15N | 电子去皮后的模型值 |
| 每支承gross/tare/net | 同上 | 份额显式；各支承之和必须回到总量 |
| 支承份额 | 零容差和为1 | 分配不能凭空增减总载荷 |

## 四、已知失效清单

1. 支承份额是输入，不计算导轮梁挠曲、轴承位置或带材轴向落点。
2. tare只代表静态已知载荷；温漂、轴承摩擦和动态惯性未建模。
3. 没有LTS桥路和ADC；电气链由T-M2负责。
4. 未确认现场是单传感器还是双传感器，永久`hypothesis_only`。
5. 无skip。

## 五、档位与负载级

**A档/交互级（`interactive`）**。三个常数静力构型。

## 六、本案例不是什么

- 不是现场支承结构识别；
- 不是导轮有限元；
- 不是LTS五点标定；
- 不是ATC600显示值；
- 不是多跨张力观测器。
