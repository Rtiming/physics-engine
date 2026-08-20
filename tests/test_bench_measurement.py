"""性能测量脚本自己的门。

墙钟数不进回退门，不等于测量链可以吞掉失败或把旧wheel冒充成当前产物。
这里专门守测量的身份与失败关闭；物理性能预算仍由``tests/perf``负责。
"""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools import bench


def test_a_failed_timed_subprocess_fails_the_measurement_closed():
    """必须红：被计时命令退出7时，报告不得照样产出一个“很快”的数。"""

    with pytest.raises(bench.BenchmarkError, match="exit code 7"):
        bench._time_subprocess(
            [sys.executable, "-c", "raise SystemExit(7)"], repeat=1
        )


def test_a_documented_nonzero_success_code_must_be_opted_in_explicitly():
    """碰撞CLI用1表示“有候选”；只有调用点显式声明后才是成功。"""

    result = bench._time_subprocess(
        [sys.executable, "-c", "raise SystemExit(1)"],
        repeat=1,
        accepted_returncodes=(0, 1),
    )
    assert result["repeat"] == 1


def test_tension_measurement_benchmark_reports_batch_and_per_sample_costs():
    result = bench._measure_tension_measurement(repeat=2)
    assert result["batch_size"] == bench.TENSION_MEASUREMENT_BATCH_SIZE == 1000
    assert result["median_s"] > 0.0
    assert result["min_s"] > 0.0
    assert result["median_per_sample_s"] == pytest.approx(
        result["median_s"] / result["batch_size"], rel=0.0, abs=0.0
    )


def test_sampled_tension_readout_benchmark_reports_plant_step_costs():
    result = bench._measure_tension_readout(repeat=2)
    assert result["batch_size"] == bench.TENSION_READOUT_BATCH_SIZE == 1000
    assert result["sample_decimation"] == 10
    assert result["delay_samples"] == 5
    assert result["median_s"] > 0.0
    assert result["median_per_plant_step_s"] == pytest.approx(
        result["median_s"] / result["batch_size"], rel=0.0, abs=0.0
    )


def test_model_motion_input_benchmark_reports_strict_document_costs():
    result = bench._measure_model_motion_input(repeat=2)
    assert result["batch_size"] == bench.MODEL_MOTION_INPUT_BATCH_SIZE == 100
    assert result["payload_bytes"] > 0
    assert result["median_s"] > 0.0
    assert result["median_per_document_s"] == pytest.approx(
        result["median_s"] / result["batch_size"], rel=0.0, abs=0.0
    )


def test_model_scene_assembly_benchmark_reports_preloaded_resource_costs():
    result = bench._measure_model_scene_assembly(repeat=2)
    assert result["batch_size"] == bench.MODEL_SCENE_ASSEMBLY_BATCH_SIZE == 100
    assert result["physical_body_count"] == 2
    assert result["motion_track_count"] == 2
    assert result["median_s"] > 0.0
    assert result["median_per_scene_s"] == pytest.approx(
        result["median_s"] / result["batch_size"], rel=0.0, abs=0.0
    )


def test_source_bytes_include_python_subpackages(tmp_path: Path):
    """必须红：只用``glob('*.py')``会让新增物理域整个隐身。"""

    package = tmp_path / "physics_engine"
    nested = package / "optics"
    nested.mkdir(parents=True)
    (package / "root.py").write_bytes(b"12345")
    (nested / "nested.py").write_bytes(b"1234567")
    (nested / "not_python.txt").write_bytes(b"ignored")

    assert bench._source_bytes(package) == 12


def _write_wheel(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def test_a_stale_wheel_is_not_reported_as_the_current_build(tmp_path: Path):
    """必须红：``dist``里“最后一个文件”不是当前执行树的构建证据。"""

    package = tmp_path / "src" / "physics_engine"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"current\n")
    wheel = tmp_path / "dist" / "physics_engine-0.5.0-py3-none-any.whl"
    _write_wheel(wheel, {"physics_engine/__init__.py": b"stale\n"})

    record = bench._current_wheel_record(wheel.parent, package)

    assert record["wheel_status"] == "stale_or_invalid"
    assert record["wheel_bytes"] is None
    assert record["wheel_name"] is None
    assert record["wheel_candidates"] == [wheel.name]


def test_a_byte_identical_wheel_is_accepted_as_current(tmp_path: Path):
    """反向门：源码集合与字节全相同的wheel不能被误判为陈旧。"""

    package = tmp_path / "src" / "physics_engine"
    subpackage = package / "electromagnetics"
    subpackage.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"version = 'x'\n")
    (subpackage / "field.py").write_bytes(b"VALUE = 1\n")
    wheel = tmp_path / "dist" / "physics_engine-x-py3-none-any.whl"
    _write_wheel(
        wheel,
        {
            "physics_engine/__init__.py": b"version = 'x'\n",
            "physics_engine/electromagnetics/field.py": b"VALUE = 1\n",
            "physics_engine-x.dist-info/METADATA": b"Name: physics-engine\n",
        },
    )

    record = bench._current_wheel_record(wheel.parent, package)

    assert record == {
        "wheel_bytes": wheel.stat().st_size,
        "wheel_name": wheel.name,
        "wheel_status": "current",
        "wheel_candidates": [wheel.name],
    }
