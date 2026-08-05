"""`case/ballistic_free_flight`的conformance门（轴7规则3）。

**本文件里没有一个判据数**：期望与容差全部从清单读，测试只把输入喂给
`integrate`、把算出的量交给清单比对。

本案例最要紧的一条是**误差必须带符号**：显式与半隐式Euler的常加速度误差
同幅反号，按`abs()`写的判据分不开这两个积分器。清单里`error_over_predicted`
对半隐式是`+1`、对显式是`−1`，容差`abs=1e-8`——写成绝对值就立刻假通过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_engine.integrate import INTEGRATORS, NumpyOps, PurePythonOps, integrate
from physics_engine.oracles import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_manifest(ROOT / "cases/ballistic_free_flight/oracle.json", root=ROOT)


def _fly(entry, ops=None):
    inputs = entry.inputs
    acceleration_mm_s2 = inputs["acceleration_mm_s2"]
    horizon_s, dt_s = inputs["horizon_s"], inputs["dt_s"]
    steps = int(round(horizon_s / dt_s))

    def acceleration(x, v, t):
        return (acceleration_mm_s2,)

    position, _, _ = integrate(
        INTEGRATORS[inputs["integrator"]],
        x0=(inputs["initial_position_mm"],),
        v0=(inputs["initial_velocity_mm_s"],),
        dt_s=dt_s,
        steps=steps,
        acceleration=acceleration,
        ops=ops,
    )
    return position[0]


@pytest.mark.parametrize("entry", MANIFEST.oracles, ids=lambda e: e.id)
def test_constant_acceleration_error_constants(entry):
    final_mm = _fly(entry)
    exact_mm = entry.expected["exact_position_mm"]
    measured = {"exact_position_mm": exact_mm}
    if "error_over_predicted" in entry.expected:
        # 带符号——这一行是本案例的承重条款。
        measured["error_over_predicted"] = (final_mm - exact_mm) / entry.expected[
            "predicted_error_mm"
        ]
        measured["predicted_error_mm"] = entry.expected["predicted_error_mm"]
    else:
        measured["relative_error"] = abs(final_mm - exact_mm) / abs(exact_mm)
    entry.check_all(measured)


def test_explicit_and_symplectic_errors_are_equal_and_opposite():
    """把"同幅反号"直接立成门：只比绝对值的判据分不开这两个积分器。"""

    pairs = {}
    for entry in MANIFEST.oracles:
        if "error_over_predicted" not in entry.expected:
            continue
        pairs.setdefault(entry.inputs["dt_s"], {})[entry.inputs["integrator"]] = (
            _fly(entry) - entry.expected["exact_position_mm"]
        )
    assert pairs, "清单里没有带符号的误差条目"
    for dt_s, by_integrator in pairs.items():
        symplectic = by_integrator["symplectic_euler"]
        explicit = by_integrator["explicit_euler"]
        assert symplectic > 0.0 > explicit or explicit > 0.0 > symplectic, (
            f"dt={dt_s}: 两者应当异号，实测 {symplectic!r} 与 {explicit!r}"
        )
        assert abs(abs(symplectic) - abs(explicit)) <= 1e-6 * abs(symplectic), (
            f"dt={dt_s}: 两者幅值应当相等，实测 {symplectic!r} 与 {explicit!r}"
        )


def test_pure_python_and_accel_backends_are_bitwise_identical():
    """0016甲案的进仓门。**逐字节，不是容差**（spec/12第5.3节：能逐字节就必须逐字节）。

    对拍规模申报（spec/12第5.4节）：本案例3个积分器×3档步长，
    单档最多4000步、每步一次逐元素运算。它足以判定的理由是——两个后端执行的是
    **同一串运算、同一个次序**（一份公式源按`VectorOps`求值），
    逐元素IEEE 754 float64加乘无归约、无次序重排，所以逐位相同是构造保证的，
    规模再大也不改变这个论证。真正需要更大规模对拍的是能量装配，随内核搬迁时再谈。
    """

    numpy = pytest.importorskip(
        "numpy", reason="加速档未安装（pip install -e '.[accel]'）——核心零依赖，这是可选档"
    )
    assert numpy is not None
    pure, accel = PurePythonOps(), NumpyOps()
    for entry in MANIFEST.oracles:
        assert _fly(entry, ops=pure) == _fly(entry, ops=accel), (
            f"{entry.id}: 两个后端结果不逐位相同——加速档换了数学，不只是换了求值方式"
        )
