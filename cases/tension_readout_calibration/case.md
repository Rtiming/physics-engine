# tension_readout_calibration T-M2：桥路、tare、ADC与五级正反程标定

判据正本为同目录`oracle.json`，生成器为`generate_oracle.py`。本案例接在T-M1之后，
回答测力轮敏感轴上的gross/tare/net力怎样变成mV、ADC和显示跨段张力。
清单身份：`case/tension_readout_calibration`；负载级：`interactive`。

## 一、物理/几何设定

传感器取100N满量程、20mV输出、12位ADC。标定使用0、10、20、30、40N五个载荷等级，
每一级都有正程和回程；合成传感器关系为`F_sensor = 1.5*T_span + 0.3N`，正反程各加
`±0.03N`。25N留出点单独评估，不进入拟合。

电气链同时跑两种显式候选：

1. `analog_pre_adc`：先扣tare，再由ADC量化net力；
2. `digital_post_adc`：gross与tare分别量化，再在数字端相减。

两者不是现场结论。实际ATC600清零位置未确认前，案例永久`hypothesis_only`。

## 二、参考解出处

生成器独立实现普通最小二乘、正反程差、满量程线性mV和就近ADC量化，不import
`physics_engine.tension_readout`。测力轮受力关系的同行/厂商出处见`docs/research/19`
参考文献[18-20]；本案例只验证其后的电气与标定层。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---:|---:|---|
| 斜率、截距、RMS、最大残差 | 8e-15 | 8e-15 | 对独立普通最小二乘，能分辨holdout污染 |
| 回程误差 | 8e-15 | 8e-15N | 每个等级正反程差换算到跨段张力 |
| holdout | 8e-15 | 8e-15N | 只预测，不得进入`fit_point_ids` |
| raw/tare/zeroed mV | 2e-15 | 2e-14mV | 对手算线性比例 |
| 两种tare模式的ADC值 | 2e-15 | 2e-14N | 对独立就近量化，半台阶错误必红 |
| gross过载 | 0 | 0 | 110N gross、20N tare时gross仍过载；net 90N不许掩盖它 |

## 四、已知失效清单

1. 标定点是合成数据，不是现场砝码或参考张力计。
2. 只拟合线性关系，不建温漂、蠕变、频响、激励电压漂移和放大器噪声。
3. 两种tare位置都是候选；现场结构未知。
4. 采样周期和时延由单元门验证，现场数值仍未知。
5. 不复现ATC600或VR451内部寄存器换算。
6. 无skip。

## 五、档位与负载级

**A档/交互级（`interactive`）**。全部是十个拟合点和三个常数电气构型。

## 六、本案例不是什么

- 不是现场五点砝码标定；
- 不是ATC600电路复现；
- 不是温漂、蠕变或动态频响模型；
- 不是T0—T4多跨张力观测器；
- 不是现场张力准确度验收。
