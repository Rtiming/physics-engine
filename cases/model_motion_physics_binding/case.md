# model_motion_physics_binding P3-M0：模型、运动与虚拟物理所有权

判据正本为同目录`oracle.json`，生成器为`generate_oracle.py`。本案例验证physics-engine自己的
通用输入合同，不读取WII或GCW代码。清单身份：`case/model_motion_physics_binding`；
负载级：`interactive`。

## 一、物理/几何设定

模型快照含三个并列组件：张力机、工件和机器人显示模型。张力机与工件各有独立collision
资产；机器人只有visual资产。虚拟物理关系把张力机绑定为static体、工件绑定为kinematic体，
机器人显示模型和它的运动track都明确排除于虚拟物理，但运动仍保留在计划中。process frame
由另一条运动track驱动，但它只是虚拟工艺frame，
不是刚体。

时间计划有0s和1s两个样点：工件从x=0mm移动到2mm并绕z轴转180°；process frame从
x=10mm移动到12mm；源状态保留A1=5°、E1=360°，累计送带由0增至100mm。

## 二、参考解出处

生成器独立写出组件所有权和中点解析值，不import`model_snapshot`、`planned_motion`或
`model_physics`。平移中点是1mm；单位四元数短弧SLERP中点是绕z轴90°，即z/w分量均为
`sqrt(1/2)`。身份、行为和排除列表按字节精确比较。

## 三、判据表

| 量 | rel | abs | 理由 |
|---|---:|---:|---|
| 物理体、行为、排除组件/运动、虚拟frame | 0 | 0 | 所有权必须逐字，不能静默忽略 |
| 工件中点平移 | 2e-15 | 2e-15mm | 独立线性插值 |
| 工件中点旋转 | 2e-15 | 2e-15 | 独立90°四元数 |
| 源状态与累计送带 | 0 | 0 | 上游状态只保留，不被接口重解释 |

## 四、已知失效清单

1. 全部资产是假SHA与合成路径，不是WII正式模型包。
2. 不读取网格字节，不证明碰撞几何正确。
3. 不计算动力学、接触力或张力，只验证输入所有权与运动边界。
4. process frame只有位姿，没有现场工艺误差带。
5. 无时间计划只在单元门验证，本案例走`time_s`。
6. 无skip。

## 五、档位与负载级

**A档/交互级（`interactive`）**。三个组件、两个track、两个样点，必须秒级完成。

## 六、本案例不是什么

- 不是WII适配器；
- 不是GCW输入链；
- 不是official WII时间线回放；
- 不是模型网格接触验收；
- 不是场景⑥端到端完成。
