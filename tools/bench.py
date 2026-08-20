#!/usr/bin/env python3
"""受控本机性能测量——**本脚本不是回退测试**。

照Drake `multibody/benchmarking/README.md`的形制明写这句：墙钟基准不作为
性能回退门。理由是实测的（research/05第四节）：共享runner的墙钟变异系数
CV=2.66%，配2%阈值假阳率45%，压到1%假阳需要约7%的门——等于测不到小回退。
MuJoCo的CI干脆不跑benchmark，Drake在README里明写其基准不承担回退检测。

**本仓的分工**：墙钟走本脚本，产报告进``work/bench/``（untracked）供人看；
真正进门的是``tests/perf/test_budgets.py``里的**确定性量**（字节数、模块数），
它们跨平台逐位稳定，不受负载影响——这正好绕开0014法则2禁止的那件事
（功能路径永不因宿主负载改变结果）。

**物理路径测量组**（决策0026）：上面三条测的是进程级墙钟（import与CLI），
它们量的是冷启动，量不到"引擎自己算物理有多快"。本脚本另有三组进程内测量：

* ``energy_assembly``：``EnergyRegistry.total``在2/8/32/128/512节点的链上，
  分``need_gradient=False``/``True``/``need_hessian=True``三档；
* ``scaling``：对上面的中位数拟合``t ∝ N^p``，报标度指数``p``——
  这个数是判断"瓶颈是解释器派发还是算术本身"的指纹（spec/12第8.1节，
  WDS在≤321节点上测到0.59）；
* ``integration``：``integrate``在``PurePythonOps``与``NumpyOps``下各跑固定步数，
  并且**分物理加速度回调与常加速度回调两种**——两者之差正是加速档边界的位置。
* ``tension_measurement_static``：1000个T-M1静力样点，覆盖矢量合力、敏感轴、
  tare与双支承，不接电气量化；这是plans/19要求新公开操作出生即带的预算。
* ``tension_readout_sampled``：1000个plant步，完整经过T-M2测力轮、标定、16位ADC、
  10步抽取、5样点时延和零阶保持；它回答采样通道是不是值得为GPU另开后端。
* ``model_motion_input_load``：100份P3-M0严格输入包复读，覆盖四层内容哈希、模型组件、
  规划track和虚拟物理所有权闭包；WII适配以后不会改变这条内核成本口径。
* ``model_scene_assembly``：100份P3.1两体场景装配，资源已预加载，量位姿组合、
  Scene finalize、MotionSource、虚拟frame和显式候选装配的进程内成本。

这些物理组也**不进门**，理由与墙钟同（决策0018第一节未被本页改动）。它们的用途是
spec/13第一节义务1的那个前提：**没有热点数据的优化提案不受理**。

进程内计时不关GC（``timeit``默认会关）——关掉GC测出来的是实验室数，
用户拿到的是带GC的数，本脚本报后者。

用法：``.venv/bin/python tools/bench.py [--repeat N] [--no-physics] [--no-profile]``
"""

from __future__ import annotations

import argparse
import cProfile
import math
import os
import pstats
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from physics_engine.canonical import FTS_PROFILE, WDS_PROFILE, canonical_file_bytes
from physics_engine.drives import TensionSensor
from physics_engine.energies import (
    AxialStretch,
    EnergyContext,
    EnergyRegistry,
    LinearBending,
    UniformGravity,
    clamped_chain_bending_stencils,
)
from physics_engine.engine_facets import PERF_BASELINE_FACET, PERF_BASELINE_VERSION
from physics_engine.integrate import (
    VELOCITY_VERLET,
    NumpyOps,
    PurePythonOps,
    integrate,
)
from physics_engine.materials import EvidenceRef
from physics_engine.model_physics import (
    BodyBehavior,
    GeometrySource,
    ModelPhysicsRelation,
    PhysicsBodyBinding,
    PhysicsModelMotionInput,
    load_physics_model_motion_input,
)
from physics_engine.model_scene import assemble_model_physics_scene
from physics_engine.model_snapshot import (
    AssetRole,
    ModelAssetRef,
    ModelComponent,
    ModelSnapshot,
)
from physics_engine.motion import InterpolationSemantics, Pose
from physics_engine.planned_motion import (
    MotionParameterization,
    MotionSourceArtifact,
    MotionTrack,
    PlannedMotion,
    PlannedMotionSample,
    TrackPose,
)
from physics_engine.state import State, StateField, StateLayout
from physics_engine.tension_measurement import MeasuringRoll
from physics_engine.tension_readout import (
    CalibrationDirection,
    CalibrationPurpose,
    LinearSpanCalibration,
    TareMode,
    TensionCalibrationPoint,
    TensionReadout,
    TensionReadoutChannel,
)
from tools.bench_model_scene import build_model_scene_benchmark_fixture

EXAMPLE_SCENE = ROOT / "examples/collision_preview_cell.scene.json"

#: 装配测量的节点规模。上界512是有意选的：spec/12第8.1节记的WDS标度指数
#: 转折点在321—641节点之间，取到512才跨得过那个区间。
PHYSICS_NODE_COUNTS: tuple[int, ...] = (2, 8, 32, 128, 512)

#: 积分测量的(节点数, 物理回调步数, 常回调步数)。步数按规模反比取，
#: 让每次测量落在几十毫秒量级——太短量的是计时器噪声，太长白等。
INTEGRATION_CASES: tuple[tuple[int, int, int], ...] = ((8, 200, 2000), (128, 40, 400), (512, 10, 100))

#: 热点分析的规模与重复次数。cProfile是确定性插桩，调用次数一次就准；
#: 重复只是让累计时间稳一点。
PROFILE_NODES = 512
PROFILE_INTEGRATION_STEPS = 10
PROFILE_TOP = 15
TENSION_MEASUREMENT_BATCH_SIZE = 1000
TENSION_READOUT_BATCH_SIZE = 1000
MODEL_MOTION_INPUT_BATCH_SIZE = 100
MODEL_SCENE_ASSEMBLY_BATCH_SIZE = 100


class BenchmarkError(RuntimeError):
    """测量链失效——这种失败不得被包装成一份正常的性能报告。"""


def _time_subprocess(
    argv: list[str], repeat: int, *, accepted_returncodes: tuple[int, ...] = (0,)
) -> dict[str, float]:
    """冷进程计时：每次都是新解释器，取中位数与最小值。

    报最小值是有意的——最小值最接近"无干扰时的真实成本"，中位数受负载影响。
    两个都报，读的人自己判断。
    """

    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
        if completed.returncode not in accepted_returncodes:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            detail = f"; stderr={stderr!r}" if stderr else ""
            raise BenchmarkError(
                f"timed subprocess failed with exit code {completed.returncode}: "
                f"{argv!r}{detail}"
            )
        samples.append(time.perf_counter() - started)
    return {
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
        "repeat": repeat,
    }


def _source_bytes(package_dir: Path) -> int:
    """确定性源码体积：递归覆盖所有Python子包。

    这必须与``tests/perf/test_budgets.py``和基线里的scope逐字同口径。
    新物理域按spec/15放在子包里；只扫顶层会让增长恰好在最需要看见的地方隐身。
    """

    return sum(path.stat().st_size for path in sorted(package_dir.rglob("*.py")))


def _current_wheel_record(dist_dir: Path, package_dir: Path) -> dict[str, object]:
    """只把与当前源码树逐文件逐字节相同的wheel认作当前构建。

    ``dist``是持久目录，文件名排序的“最后一个”既不能证明版本最新，也不能证明
    wheel由当前HEAD构建。这里比较wheel内的Python文件集合与每个文件的原始字节；
    任一处不同都只报告候选存在，不给它填``wheel_bytes``，避免旧产物冒充当前体积。
    """

    candidates = sorted(dist_dir.glob("*.whl")) if dist_dir.is_dir() else []
    candidate_names = [path.name for path in candidates]
    if not candidates:
        return {
            "wheel_bytes": None,
            "wheel_name": None,
            "wheel_status": "missing",
            "wheel_candidates": candidate_names,
        }

    prefix = f"{package_dir.name}/"
    source = {
        f"{prefix}{path.relative_to(package_dir).as_posix()}": path.read_bytes()
        for path in sorted(package_dir.rglob("*.py"))
    }
    #: 最后修改时间只决定“先检查谁”，**不参与当前性判据**；当前性只由上面的
    #: 文件集合与字节对拍决定。名字用作相同mtime时的稳定tie-break。
    newest_first = sorted(
        candidates, key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True
    )
    for wheel in newest_first:
        try:
            with ZipFile(wheel) as archive:
                wheel_names = {
                    name for name in archive.namelist()
                    if name.startswith(prefix) and name.endswith(".py")
                }
                if wheel_names != set(source):
                    continue
                if any(archive.read(name) != payload for name, payload in source.items()):
                    continue
        except (BadZipFile, KeyError, OSError):
            continue
        return {
            "wheel_bytes": wheel.stat().st_size,
            "wheel_name": wheel.name,
            "wheel_status": "current",
            "wheel_candidates": candidate_names,
        }

    return {
        "wheel_bytes": None,
        "wheel_name": None,
        "wheel_status": "stale_or_invalid",
        "wheel_candidates": candidate_names,
    }


def _load_average() -> float | None:
    try:
        return round(os.getloadavg()[0], 3)
    except (OSError, AttributeError):
        return None


def _time_in_process(work: Callable[[], object], repeat: int) -> dict:
    """进程内计时：先热身一次（把import、首次分配、分支预测的一次性成本挤掉），
    再取``repeat``次的中位数与最小值，并记下**当时**的1分钟负载。

    负载逐组记而不是只记开头一次——一次跑几秒，中途负载会变，
    只记开头等于把"这组数是在什么环境下测的"记错。
    """

    work()  # warm-up，不计入
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        work()
        samples.append(time.perf_counter() - started)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "repeat": repeat,
        "load_average_1m": _load_average(),
    }


def _measure_tension_measurement(repeat: int) -> dict:
    """1000个静力测量样点；输入固定，便于跨版本同口径复测。"""

    root_half = math.sqrt(0.5)
    roll = MeasuringRoll(
        measurement_id="measurement/bench-tension-roll",
        sensor_axis_xyz=(-root_half, root_half, 0.0),
        tare_force_n_xyz=(-5.0 * root_half, 5.0 * root_half, 0.0),
        support_shares=(0.5, 0.5),
        evidence=EvidenceRef(
            grade="estimated",
            evidence_id="evidence/bench-tension-roll-synthetic",
            method="Synthetic fixed input for performance measurement only.",
        ),
    )

    def batch() -> None:
        for _ in range(TENSION_MEASUREMENT_BATCH_SIZE):
            roll.measure(
                incoming_tension_n=10.0,
                outgoing_tension_n=10.0,
                incoming_tangent_xyz=(1.0, 0.0, 0.0),
                outgoing_tangent_xyz=(0.0, 1.0, 0.0),
            )

    result = _time_in_process(batch, repeat)
    result["batch_size"] = TENSION_MEASUREMENT_BATCH_SIZE
    result["median_per_sample_s"] = result["median_s"] / TENSION_MEASUREMENT_BATCH_SIZE
    result["min_per_sample_s"] = result["min_s"] / TENSION_MEASUREMENT_BATCH_SIZE
    return result


def _measure_tension_readout(repeat: int) -> dict:
    """1000个plant步的采样读出；标定和队列在热身前构造，不混入计时。"""

    evidence = EvidenceRef(
        grade="estimated",
        evidence_id="evidence/bench-tension-readout-synthetic",
        method="Synthetic fixed input for sampled readout performance only.",
    )
    gain = math.sqrt(2.0)
    points = tuple(
        TensionCalibrationPoint(
            point_id=f"calibration-point/bench-{index}-{direction.value}",
            sensor_force_n=gain * tension,
            reference_span_tension_n=tension,
            direction=direction,
            purpose=CalibrationPurpose.FIT,
        )
        for index, tension in enumerate((0.0, 10.0, 20.0, 30.0, 40.0))
        for direction in (
            CalibrationDirection.INCREASING,
            CalibrationDirection.DECREASING,
        )
    )
    calibration = LinearSpanCalibration.fit(
        calibration_id="calibration/bench-tension-readout",
        points=points,
        evidence=evidence,
        uncertainty_n=None,
    )
    root_half = math.sqrt(0.5)
    roll = MeasuringRoll(
        measurement_id="measurement/bench-tension-readout",
        sensor_axis_xyz=(-root_half, root_half, 0.0),
        tare_force_n_xyz=(-5.0 * root_half, 5.0 * root_half, 0.0),
        support_shares=(0.5, 0.5),
        evidence=evidence,
    )
    readout = TensionReadout(
        readout_id="readout/bench-tension-readout",
        transducer=TensionSensor(100.0, 20.0, 16),
        calibration=calibration,
        tare_mode=TareMode.ANALOG_PRE_ADC,
        evidence=evidence,
    )

    def batch() -> None:
        channel = TensionReadoutChannel.at_steady_state(
            channel_id="measurement-channel/bench-tension-readout",
            roll=roll,
            incoming_tangent_xyz=(1.0, 0.0, 0.0),
            outgoing_tangent_xyz=(0.0, 1.0, 0.0),
            readout=readout,
            plant_dt_s=1.0e-5,
            sample_decimation=10,
            delay_samples=5,
            span_tension_n=20.0,
        )
        for index in range(TENSION_READOUT_BATCH_SIZE):
            channel, _ = channel.advance(span_tension_n=20.0 + 0.001 * (index % 17))

    result = _time_in_process(batch, repeat)
    result["batch_size"] = TENSION_READOUT_BATCH_SIZE
    result["sample_decimation"] = 10
    result["delay_samples"] = 5
    result["median_per_plant_step_s"] = result["median_s"] / TENSION_READOUT_BATCH_SIZE
    result["min_per_plant_step_s"] = result["min_s"] / TENSION_READOUT_BATCH_SIZE
    return result


def _measure_model_motion_input(repeat: int) -> dict:
    """100份最小模型—运动—物理包严格复读；构造与序列化不混入计时。"""

    evidence = EvidenceRef(
        grade="estimated",
        evidence_id="evidence/bench-model-motion-synthetic",
        method="Synthetic fixed model-motion input for performance measurement only.",
    )
    identity = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    model = ModelSnapshot.create(
        model_id="model/bench-model-motion",
        root_frame_id="frame/bench-root",
        producer_id="producer/bench",
        source_manifest_sha256="b" * 64,
        components=(
            ModelComponent(
                component_id="model-component/bench-workpiece",
                frame_id="frame/bench-workpiece",
                semantic_role="workpiece",
                parent_component_id=None,
                parent_from_component=identity,
                visual_asset=None,
                collision_asset=ModelAssetRef(
                    asset_id="asset/bench-workpiece-collision",
                    role=AssetRole.COLLISION,
                    path_relative="assets/workpiece.collision.stl",
                    sha256="a" * 64,
                    format="stl",
                    units="mm",
                    frame_id="frame/bench-root",
                    component_from_asset=identity,
                ),
            ),
        ),
    )
    semantics = InterpolationSemantics(
        "linear", "slerp", "shortest", "hold_interval_start", "reject"
    )
    motion = PlannedMotion.create(
        motion_id="motion/bench-model-motion",
        producer_id="producer/bench",
        root_frame_id="frame/bench-root",
        parameterization=MotionParameterization.TIME_S,
        coordinate_unit="s",
        source_artifacts=(
            MotionSourceArtifact(
                "selected_plan", "artifact/bench-motion-plan", "c" * 64
            ),
        ),
        tracks=(
            MotionTrack(
                "motion-track/bench-workpiece",
                "model-component/bench-workpiece",
                "frame/bench-workpiece",
                semantics,
            ),
        ),
        state_coordinates=(),
        samples=(
            PlannedMotionSample(
                "motion-sample/bench-0", 0.0, 0.0,
                (TrackPose("motion-track/bench-workpiece", identity),),
                (), 0.0, "laydown",
            ),
            PlannedMotionSample(
                "motion-sample/bench-1", 1.0, 1.0,
                (
                    TrackPose(
                        "motion-track/bench-workpiece",
                        Pose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                    ),
                ),
                (), 1.0, "laydown",
            ),
        ),
    )
    relation = ModelPhysicsRelation.create(
        relation_id="model-physics/bench",
        model_snapshot_sha256=model.content_sha256,
        motion_plan_sha256=motion.content_sha256,
        body_bindings=(
            PhysicsBodyBinding(
                "body/bench-workpiece",
                "model-component/bench-workpiece",
                BodyBehavior.KINEMATIC,
                GeometrySource.COLLISION_ASSET,
                "motion-track/bench-workpiece",
                None,
                None,
                None,
            ),
        ),
        virtual_frame_bindings=(),
        excluded_component_ids=(),
        excluded_motion_track_ids=(),
    )
    package = PhysicsModelMotionInput.create(
        input_id="physics-input/bench-model-motion",
        model=model,
        motion=motion,
        relation=relation,
        evidence=evidence,
    )
    payload = canonical_file_bytes(package.to_document(), WDS_PROFILE)

    def batch() -> None:
        for _ in range(MODEL_MOTION_INPUT_BATCH_SIZE):
            load_physics_model_motion_input(payload)

    result = _time_in_process(batch, repeat)
    result["batch_size"] = MODEL_MOTION_INPUT_BATCH_SIZE
    result["payload_bytes"] = len(payload)
    result["median_per_document_s"] = result["median_s"] / MODEL_MOTION_INPUT_BATCH_SIZE
    result["min_per_document_s"] = result["min_s"] / MODEL_MOTION_INPUT_BATCH_SIZE
    return result


def _measure_model_scene_assembly(repeat: int) -> dict:
    """100份预加载资源的P3.1场景装配；fixture构造不混入计时。"""

    package, resources, interactions = build_model_scene_benchmark_fixture()

    def batch() -> None:
        for _ in range(MODEL_SCENE_ASSEMBLY_BATCH_SIZE):
            assemble_model_physics_scene(package, resources, interactions)

    result = _time_in_process(batch, repeat)
    result["batch_size"] = MODEL_SCENE_ASSEMBLY_BATCH_SIZE
    result["physical_body_count"] = len(package.relation.body_bindings)
    result["motion_track_count"] = len(package.motion.tracks)
    result["median_per_scene_s"] = result["median_s"] / MODEL_SCENE_ASSEMBLY_BATCH_SIZE
    result["min_per_scene_s"] = result["min_s"] / MODEL_SCENE_ASSEMBLY_BATCH_SIZE
    return result


def build_chain(
    nodes: int, *, with_bending: bool = False
) -> tuple[EnergyRegistry, State, EnergyContext]:
    """一根``nodes``节点的直链：重力 + 逐段轴向拉伸（可选 + 线性弯曲）。

    **为什么是链而不是随便一堆点**：轴向拉伸是逐单元的，链让边数恰好是``N−1``，
    于是"每节点成本"这个量有定义，可以直接对上spec/12第8.1节那条
    "每节点每次装配约60微秒"。初始位置带0.25mm的预拉伸与横向抖动——
    静止长度上伸长量为零会让梯度恰好为零，那测的就不是一般情形的分支了。

    ``with_bending``给的是**求解器实际会遇到的那个注册表**：`solve_equilibrium`
    每次牛顿迭代调一次``total(need_gradient=True, need_hessian=True)``，
    三项一起装配。两项的画像不代表那条路的成本，所以两个都测。
    """

    layout = StateLayout(
        layout_id=f"layout/bench_chain_n{nodes}",
        fields=tuple(
            field
            for index in range(nodes)
            for field in (
                StateField(f"node{index}_x_mm", 1),
                StateField(f"node{index}_y_mm", 1),
                StateField(f"node{index}_z_mm", 1),
            )
        ),
    )
    vector: list[float] = []
    for index in range(nodes):
        vector.extend((10.0 * index + 0.25, 0.5 * ((index % 3) - 1), 0.0))
    state = State(layout=layout, vector=tuple(vector))
    context = EnergyContext(
        context_id="context/bench_chain",
        node_masses_kg=tuple(0.4 + 0.001 * (index % 7) for index in range(nodes)),
        gravity_mm_s2=(0.0, -9806.65, 0.0),
    )
    edges = tuple((index, index + 1, 10.0, 2000.0) for index in range(nodes - 1))
    terms = [UniformGravity(), AxialStretch(edges=edges)]
    if with_bending:
        terms.append(
            LinearBending(stencils=clamped_chain_bending_stencils(nodes, 10.0, 2.0e4))
        )
    return EnergyRegistry(terms=tuple(terms)), state, context


#: 三档的名字与``total``的关键字。次序即报告次序。
ASSEMBLY_TIERS: tuple[tuple[str, dict], ...] = (
    ("energy_only", {}),
    ("with_gradient", {"need_gradient": True}),
    ("with_hessian", {"need_hessian": True}),
)


def _measure_energy_assembly(repeat: int) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {name: [] for name, _ in ASSEMBLY_TIERS}
    for nodes in PHYSICS_NODE_COUNTS:
        registry, state, context = build_chain(nodes)
        for name, kwargs in ASSEMBLY_TIERS:
            def work(registry=registry, state=state, context=context, kwargs=kwargs):
                return registry.total(state, context, **kwargs)

            record = _time_in_process(work, repeat)
            record["nodes"] = nodes
            record["dof"] = len(state.vector)
            record["per_node_us"] = record["median_s"] / nodes * 1e6
            results[name].append(record)
    return results


def _fit_power_law(points: list[tuple[int, float]]) -> dict | None:
    """对``t = a·N^p``做对数-对数最小二乘，返回``p``与``R²``。

    **报R²不是装饰**：``p``只有在数据真的服从幂律时才有意义。
    小规模上固定开销主导会把点压平，那时R²会掉下来，看的人必须能看见这件事。
    """

    usable = [(nodes, seconds) for nodes, seconds in points if nodes > 0 and seconds > 0.0]
    if len(usable) < 2:
        return None
    xs = [math.log(nodes) for nodes, _ in usable]
    ys = [math.log(seconds) for _, seconds in usable]
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True)
    )
    return {
        "p": slope,
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None,
        "node_range": [usable[0][0], usable[-1][0]],
        "point_count": count,
    }


#: 拟合窗口。三个窗口不是冗余：
#: ``le_321``是**对齐WDS口径**的那个（spec/12第8.1节的0.59测在≤321节点上）；
#: ``all``给全景；``upper``给"固定开销不再主导之后"的渐近值。
FIT_WINDOWS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("all", lambda nodes: True),
    ("le_321_wds_comparable", lambda nodes: nodes <= 321),
    ("upper_32_to_512", lambda nodes: nodes >= 32),
)


def _fit_scaling(assembly: dict[str, list[dict]]) -> dict[str, dict]:
    """两套拟合：按中位数与按最小值。

    **按最小值那套是主的。** 理由是本轮实测的：一次干扰只要打中最小的那个规模，
    中位数拟合出来的``p``就会整体偏掉（实测见过0.72与1.02两个值来自同一份代码）。
    最小值是各规模上"无干扰成本"的最好估计，而幂律要拟合的正是无干扰成本；
    中位数那套留着，因为两套差得远本身就是"这次测量不干净"的信号。
    """

    scaling: dict[str, dict] = {}
    for tier, records in assembly.items():
        fits: dict[str, dict] = {}
        for window_name, predicate in FIT_WINDOWS:
            chosen = [record for record in records if predicate(record["nodes"])]
            fits[window_name] = {
                "by_min": _fit_power_law([(r["nodes"], r["min_s"]) for r in chosen]),
                "by_median": _fit_power_law([(r["nodes"], r["median_s"]) for r in chosen]),
            }
        scaling[tier] = fits
    return scaling


def _constant_acceleration(dof: int):
    """常加速度回调：**用来把积分器自己的算术从物理里剥出来**。

    它不是一个物理场景，它是一把尺子。物理回调那一组才是用户会遇到的形状；
    两组一起报，才能回答"加速档到底加速了什么"。
    """

    constant = tuple(0.0 if index % 3 else -9806.65 for index in range(dof))

    def acceleration_of(x, v, t):
        return constant

    return acceleration_of


def _measure_integration(repeat: int) -> list[dict]:
    backends = (PurePythonOps(), NumpyOps())
    results: list[dict] = []
    for nodes, physical_steps, constant_steps in INTEGRATION_CASES:
        registry, state, context = build_chain(nodes)
        x0 = state.vector
        v0 = tuple(0.0 for _ in x0)
        callbacks = (
            ("physical_acceleration", registry.acceleration(context, state.layout),
             physical_steps),
            ("constant_acceleration", _constant_acceleration(len(x0)), constant_steps),
        )
        for flavour, acceleration, steps in callbacks:
            timings: dict[str, dict] = {}
            for ops in backends:
                def work(acceleration=acceleration, steps=steps, ops=ops, x0=x0, v0=v0):
                    return integrate(
                        VELOCITY_VERLET, x0=x0, v0=v0, dt_s=1e-5, steps=steps,
                        acceleration=acceleration, ops=ops,
                    )

                timings[ops.name] = _time_in_process(work, repeat)
            pure = timings["pure_python"]["median_s"]
            fast = timings["numpy"]["median_s"]
            results.append({
                "nodes": nodes,
                "dof": len(x0),
                "integrator": VELOCITY_VERLET.declaration.name,
                "acceleration_flavour": flavour,
                "steps": steps,
                "backends": timings,
                "pure_python_over_numpy": (pure / fast) if fast > 0.0 else None,
            })
    return results


def _format_function(func: tuple[str, int, str]) -> str:
    filename, lineno, name = func
    try:
        shown = str(Path(filename).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        shown = Path(filename).name
    return f"{shown}:{lineno}({name})"


PROFILE_NOTE = (
    "cProfile插桩本身有开销，绝对秒数比未插桩墙钟大；看占比，不看绝对值。"
    "累计占比之和会大于1（嵌套调用被上层重复计入），且顶层函数偶尔略超100%"
    "（total_tt是自身时间之和，与顶层累计时间的计量口径差一点）——"
    "**可加的是self_share那一列**。"
)


def _hotspot_rows(profiler: cProfile.Profile) -> tuple[list[dict], float]:
    stats = pstats.Stats(profiler)
    total_s = stats.total_tt
    rows = []
    for func, (_, calls, tottime, cumtime, _callers) in stats.stats.items():
        rows.append({
            "function": _format_function(func),
            "funcname": func[2],
            "calls": calls,
            "cumulative_s": cumtime,
            "self_s": tottime,
            "cumulative_share": (cumtime / total_s) if total_s > 0.0 else None,
            "self_share": (tottime / total_s) if total_s > 0.0 else None,
        })
    rows.sort(key=lambda row: (-row["cumulative_s"], row["function"]))
    return rows, total_s


def _measure_solver_path(repeat: int) -> dict:
    """求解器每次牛顿迭代那一次装配：三项、``need_gradient``与``need_hessian``同时要。

    `solve_equilibrium`就是这么调的（`solve.py:112`）。它是本仓今天**最贵的一次
    公开调用**，所以单列一组——两项的画像不代表它。
    """

    registry, state, context = build_chain(PROFILE_NODES, with_bending=True)
    kwargs = {"need_gradient": True, "need_hessian": True}

    def work():
        return registry.total(state, context, **kwargs)

    record = _time_in_process(work, repeat)
    record["nodes"] = PROFILE_NODES
    record["dof"] = len(state.vector)
    record["terms"] = list(registry.order)
    record["caller"] = "solve_equilibrium每次牛顿迭代一次"
    return record


def _profile_assembly(
    nodes: int, kwargs: dict, repeat: int, top: int, *, with_bending: bool = False
) -> dict:
    """cProfile跑最大规模装配，取累计时间前``top``个。"""

    registry, state, context = build_chain(nodes, with_bending=with_bending)
    registry.total(state, context, **kwargs)  # warm-up

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeat):
        registry.total(state, context, **kwargs)
    profiler.disable()

    rows, total_s = _hotspot_rows(profiler)
    return {
        "nodes": nodes,
        "dof": nodes * 3,
        "repeat": repeat,
        "terms": list(registry.order),
        "profiled_total_s": total_s,
        "note": PROFILE_NOTE,
        "top": rows[:top],
    }


def _profile_integration(nodes: int, steps: int, top: int) -> dict:
    """积分路径的热点——**这一块是为了回答"加速档该做在哪"而存在的**。

    报一个额外的数：加速度回调占整条积分路径的累计比。它直接给出加速档的
    Amdahl上限——回调是调用方代码，两个后端都用元组调它（见`integrate.py`的
    `Acceleration`注释），所以后端再快也快不动那一段。
    """

    registry, state, context = build_chain(nodes)
    acceleration = registry.acceleration(context, state.layout)
    x0 = state.vector
    v0 = tuple(0.0 for _ in x0)
    kwargs = {"x0": x0, "v0": v0, "dt_s": 1e-5, "acceleration": acceleration,
              "ops": PurePythonOps()}
    integrate(VELOCITY_VERLET, steps=2, **kwargs)  # warm-up

    profiler = cProfile.Profile()
    profiler.enable()
    integrate(VELOCITY_VERLET, steps=steps, **kwargs)
    profiler.disable()

    rows, total_s = _hotspot_rows(profiler)
    callback = next(
        (row for row in rows if row["funcname"] == "acceleration_of"), None
    )
    share = callback["cumulative_share"] if callback else None
    return {
        "nodes": nodes,
        "dof": nodes * 3,
        "steps": steps,
        "backend": "pure_python",
        "integrator": VELOCITY_VERLET.declaration.name,
        "profiled_total_s": total_s,
        "acceleration_callback_share": share,
        "backend_speedup_ceiling": (
            (1.0 / share) if share is not None and share > 0.0 else None
        ),
        "ceiling_note": (
            "上限=1/回调占比（Amdahl）。回调在加速档边界之外——两个后端都用元组调它，"
            "见integrate.py的`Acceleration`注释——所以后端把**其余全部**变成零耗时，"
            "剩下的仍是那一段回调，总加速最多这个倍数。"
        ),
        "note": PROFILE_NOTE,
        "top": rows[:top],
    }


def measure_physics(repeat: int, *, with_profile: bool) -> dict:
    assembly = _measure_energy_assembly(repeat)
    report = {
        "case": {
            "name": "chain",
            "description": "N节点直链，能量项次序=(uniform_gravity, axial_stretch)，边数N−1",
            "terms": ["uniform_gravity", "axial_stretch"],
            "node_counts": list(PHYSICS_NODE_COUNTS),
        },
        "energy_assembly": assembly,
        "scaling": _fit_scaling(assembly),
        "integration": _measure_integration(repeat),
        "solver_path_assembly": _measure_solver_path(repeat),
        "tension_measurement_static": _measure_tension_measurement(repeat),
        "tension_readout_sampled": _measure_tension_readout(repeat),
        "model_motion_input_load": _measure_model_motion_input(repeat),
        "model_scene_assembly": _measure_model_scene_assembly(repeat),
    }
    if with_profile:
        report["hotspots"] = {
            "assembly_with_gradient": _profile_assembly(
                PROFILE_NODES, {"need_gradient": True}, max(repeat * 4, 20), PROFILE_TOP
            ),
            "assembly_with_hessian": _profile_assembly(
                PROFILE_NODES, {"need_hessian": True}, 1, PROFILE_TOP
            ),
            "solver_path_assembly": _profile_assembly(
                PROFILE_NODES, {"need_gradient": True, "need_hessian": True}, 1,
                PROFILE_TOP, with_bending=True,
            ),
            "integration_physical": _profile_integration(
                PROFILE_NODES, PROFILE_INTEGRATION_STEPS, PROFILE_TOP
            ),
        }
    return report


def measure(repeat: int, *, with_physics: bool = True, with_profile: bool = True) -> dict:
    python = str(ROOT / ".venv/bin/python")
    package_dir = ROOT / "src/physics_engine"
    deterministic = {
        "source_bytes": _source_bytes(package_dir),
        **_current_wheel_record(ROOT / "dist", package_dir),
    }
    return {
        "facet": PERF_BASELINE_FACET,
        "facet_version": PERF_BASELINE_VERSION,
        "kind": "measurement_report",
        "load_average_1m_at_start": _load_average(),
        "physics": measure_physics(repeat, with_profile=with_profile) if with_physics else None,
        "wall_clock": {
            "import_physics_engine": _time_subprocess(
                [python, "-c", "import physics_engine"], repeat
            ),
            "pe_scene_validate": _time_subprocess(
                [python, "-m", "physics_engine.cli", "validate", str(EXAMPLE_SCENE)], repeat
            ),
            "pe_scene_check_collisions": _time_subprocess(
                [python, "-m", "physics_engine.cli", "check-collisions", str(EXAMPLE_SCENE)],
                repeat,
                #: CLI契约：0=无候选、1=有候选，二者都是一次成功查询；2才是非法输入。
                accepted_returncodes=(0, 1),
            ),
        },
        "deterministic": deterministic,
    }


def _print_physics(physics: dict) -> None:
    print("\n物理路径（进程内，决策0026）：")
    print("  能量装配 中位数ms / 每节点µs")
    for tier, records in physics["energy_assembly"].items():
        cells = " ".join(
            f"N={r['nodes']}:{r['median_s'] * 1e3:.3f}/{r['per_node_us']:.2f}"
            for r in records
        )
        print(f"    {tier:14s} {cells}")
    print("  标度指数 p（t ∝ N^p）——主口径按最小值，括号里是按中位数")
    for tier, windows in physics["scaling"].items():
        cells = " ".join(
            f"{name}:p={pair['by_min']['p']:.3f}"
            f"(R²={pair['by_min']['r_squared']:.3f}|中位{pair['by_median']['p']:.3f})"
            for name, pair in windows.items()
            if pair["by_min"] is not None and pair["by_min"]["r_squared"] is not None
        )
        print(f"    {tier:14s} {cells}")
    solver = physics["solver_path_assembly"]
    print(
        f"  求解器那一次装配（{'+'.join(solver['terms'])}，N={solver['nodes']}，梯度＋Hessian）"
        f" median={solver['median_s'] * 1e3:.1f}ms min={solver['min_s'] * 1e3:.1f}ms"
    )
    measurement = physics["tension_measurement_static"]
    print(
        f"  张力测量静力 {measurement['batch_size']}样点 "
        f"median={measurement['median_s'] * 1e3:.3f}ms "
        f"per_sample={measurement['median_per_sample_s'] * 1e6:.3f}us"
    )
    readout = physics["tension_readout_sampled"]
    print(
        f"  张力采样读出 {readout['batch_size']} plant步 "
        f"median={readout['median_s'] * 1e3:.3f}ms "
        f"per_step={readout['median_per_plant_step_s'] * 1e6:.3f}us "
        f"decimation={readout['sample_decimation']} delay_samples={readout['delay_samples']}"
    )
    model_motion = physics["model_motion_input_load"]
    print(
        f"  模型运动输入严格复读 {model_motion['batch_size']}份 "
        f"payload={model_motion['payload_bytes']}B "
        f"median={model_motion['median_s'] * 1e3:.3f}ms "
        f"per_document={model_motion['median_per_document_s'] * 1e6:.3f}us"
    )
    model_scene = physics["model_scene_assembly"]
    print(
        f"  模型场景装配 {model_scene['batch_size']}份 "
        f"bodies={model_scene['physical_body_count']} "
        f"tracks={model_scene['motion_track_count']} "
        f"median={model_scene['median_s'] * 1e3:.3f}ms "
        f"per_scene={model_scene['median_per_scene_s'] * 1e6:.3f}us"
    )
    print("  积分推进 pure_python/numpy 耗时比")
    for record in physics["integration"]:
        pure = record["backends"]["pure_python"]["median_s"]
        fast = record["backends"]["numpy"]["median_s"]
        print(
            f"    N={record['nodes']:<4d} {record['acceleration_flavour']:22s} "
            f"steps={record['steps']:<5d} pure={pure * 1e3:.2f}ms numpy={fast * 1e3:.2f}ms "
            f"ratio={record['pure_python_over_numpy']:.3f}"
        )
    hotspots = physics.get("hotspots")
    if hotspots:
        for name, block in hotspots.items():
            print(f"  热点 {name}（N={block['nodes']}，前5，cum%/self%）")
            for row in block["top"][:5]:
                print(
                    f"    cum={row['cumulative_share'] * 100:6.2f}% "
                    f"self={row['self_share'] * 100:6.2f}% calls={row['calls']:<9d} "
                    f"{row['function']}"
                )
            if block.get("acceleration_callback_share") is not None:
                print(
                    f"    加速度回调占比={block['acceleration_callback_share'] * 100:.2f}%"
                    f" → 后端加速上限={block['backend_speedup_ceiling']:.3f}×"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--no-physics", action="store_true", help="只测进程级墙钟，跳过物理路径测量组"
    )
    parser.add_argument(
        "--no-profile", action="store_true", help="测物理路径但跳过cProfile热点分析"
    )
    args = parser.parse_args(argv)

    report = measure(
        args.repeat, with_physics=not args.no_physics, with_profile=not args.no_profile
    )
    out_dir = ROOT / "work/bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latest.json"
    out_path.write_bytes(canonical_file_bytes(report, FTS_PROFILE))

    print("本报告不是回退测试（Drake形制）——墙钟只供人看，进门的是确定性量。")
    for name, stats in report["wall_clock"].items():
        print(f"  {name}: median={stats['median_s']}s min={stats['min_s']}s")
    determ = report["deterministic"]
    print(
        f"  source_bytes={determ['source_bytes']} wheel_bytes={determ['wheel_bytes']} "
        f"wheel_status={determ['wheel_status']}"
    )
    print(f"  load_average_1m={report['load_average_1m_at_start']}")
    if report["physics"] is not None:
        _print_physics(report["physics"])
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
