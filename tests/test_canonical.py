"""轴3规则2参考实现的门：声明式规范化+读侧对称防御。"""

import math

import pytest

from physics_engine.canonical import (
    CanonicalError,
    CanonicalProfile,
    FTS_PROFILE,
    WDS_PROFILE,
    canonical_bytes,
    canonical_file_bytes,
    canonical_sha256,
    strict_loads,
)


def test_canonical_bytes_are_sorted_compact_and_deterministic():
    a = canonical_bytes({"b": 1, "a": [1.5, "x"]}, FTS_PROFILE)
    b = canonical_bytes({"a": [1.5, "x"], "b": 1}, FTS_PROFILE)
    assert a == b == b'{"a":[1.5,"x"],"b":1}'


def test_ascii_policy_is_the_declared_difference():
    doc = {"名": "值"}
    assert "\\u" in canonical_bytes(doc, FTS_PROFILE).decode("ascii")
    assert canonical_bytes(doc, WDS_PROFILE) == '{"名":"值"}'.encode("utf-8")


def test_file_bytes_carry_the_declared_trailing_newline():
    doc = {"a": 1}
    assert canonical_file_bytes(doc, FTS_PROFILE).endswith(b"}\n")
    bare = CanonicalProfile(ensure_ascii=True, file_trailing_newline=False)
    assert canonical_file_bytes(doc, bare).endswith(b"}")


def test_fingerprint_bytes_never_carry_the_newline():
    doc = {"a": 1}
    assert not canonical_bytes(doc, FTS_PROFILE).endswith(b"\n")
    assert canonical_sha256(doc, FTS_PROFILE) == canonical_sha256(doc, FTS_PROFILE)


def test_nan_is_rejected_at_write_time():
    with pytest.raises(CanonicalError, match="not canonicalizable"):
        canonical_bytes({"x": math.nan}, FTS_PROFILE)
    with pytest.raises(CanonicalError, match="not canonicalizable"):
        canonical_bytes({"x": math.inf}, WDS_PROFILE)


def test_duplicate_keys_are_rejected_at_read_time():
    with pytest.raises(CanonicalError, match="duplicate object key"):
        strict_loads(b'{"a":1,"a":2}')


def test_nonfinite_literals_are_rejected_at_read_time():
    with pytest.raises(CanonicalError, match="nonfinite JSON constant"):
        strict_loads(b'{"x":NaN}')
    with pytest.raises(CanonicalError, match="nonfinite JSON constant"):
        strict_loads(b'{"x":Infinity}')


def test_invalid_json_is_rejected():
    with pytest.raises(CanonicalError, match="not valid JSON"):
        strict_loads(b"{nope")


def test_roundtrip_through_strict_loads():
    doc = {"a": [1, 2.5], "b": {"c": "文"}}
    assert strict_loads(canonical_bytes(doc, WDS_PROFILE)) == doc
    assert strict_loads(canonical_file_bytes(doc, FTS_PROFILE).rstrip(b"\n")) == doc
