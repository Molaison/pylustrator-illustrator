"""Lossless masked-array equality and generated-command literals."""

from __future__ import annotations

import numpy as np

from pylustrator.commands import semantic_equal
from pylustrator.replay import replay_literal


def _evaluate(value, *, preserve_ndarray: bool = False):
    return eval(
        replay_literal(value, preserve_ndarray=preserve_ndarray),
        {"np": np},
    )


def test_semantic_equal_distinguishes_masked_and_unmasked_values() -> None:
    masked = np.ma.array([1.0, 2.0], mask=[False, True])

    assert not semantic_equal(masked, np.asarray(masked.data))
    assert not semantic_equal(np.asarray(masked.data), masked)
    assert semantic_equal(np.ma.masked, np.ma.masked)
    assert not semantic_equal(np.ma.masked, np.ma.array(0.0, mask=True))


def test_semantic_equal_compares_all_masked_array_state() -> None:
    data = np.array([0.25, -123.5, np.nan, np.inf, -np.inf])
    mask = np.array([False, True, True, True, True])
    original = np.ma.array(
        data,
        mask=mask,
        fill_value=-7.25,
        hard_mask=True,
    )

    assert semantic_equal(original, original.copy())

    within_tolerance = original.copy()
    within_tolerance.data[1] += 1e-13
    assert semantic_equal(original, within_tolerance)

    different_hidden_data = original.copy()
    different_hidden_data.data[1] = 456.0
    assert not semantic_equal(original, different_hidden_data)

    different_mask = original.copy()
    different_mask.mask[0] = True
    assert not semantic_equal(original, different_mask)

    different_fill = np.ma.array(
        data.copy(),
        mask=mask.copy(),
        fill_value=-8.25,
        hard_mask=True,
    )
    assert not semantic_equal(original, different_fill)

    different_hardmask = original.copy()
    different_hardmask.soften_mask()
    assert not semantic_equal(original, different_hardmask)

    different_dtype = np.ma.array(
        data.astype(np.float32),
        mask=mask,
        fill_value=np.float32(-7.25),
        hard_mask=True,
    )
    assert not semantic_equal(original, different_dtype)


def test_semantic_equal_distinguishes_nomask_from_explicit_false_mask() -> None:
    data = np.array([[1, 2], [3, 4]], dtype=np.int16)
    nomask = np.ma.array(data, mask=np.ma.nomask)
    explicit = np.ma.array(data, mask=np.zeros(data.shape, dtype=bool))

    assert nomask.mask is np.ma.nomask
    assert explicit.mask is not np.ma.nomask
    assert not semantic_equal(nomask, explicit)


def test_masked_replay_preserves_hidden_nonfinite_data_and_metadata() -> None:
    data = np.array(
        [[1.25, np.nan, np.inf], [-np.inf, -42.5, 0.0]],
        dtype=np.float32,
    )
    mask = np.array(
        [[False, True, True], [True, True, False]],
        dtype=bool,
    )
    original = np.ma.array(
        data,
        mask=mask,
        fill_value=np.float32(-7.5),
        hard_mask=True,
    )

    literal = replay_literal(original)
    replayed = eval(literal, {"np": np})

    assert literal.startswith("np.ma.array(")
    assert "mask=np.array(" in literal
    assert "fill_value=" in literal
    assert "hard_mask=True" in literal
    assert "dtype=np.dtype(" in literal
    assert semantic_equal(original, replayed, atol=0, rtol=0)
    np.testing.assert_equal(replayed.data, original.data)


def test_masked_replay_preserves_nomask_and_explicit_false_mask() -> None:
    data = np.arange(6, dtype=np.int16).reshape(2, 3)
    nomask = np.ma.array(data, mask=np.ma.nomask, fill_value=-123)
    explicit = np.ma.array(
        data,
        mask=np.zeros(data.shape, dtype=bool),
        fill_value=-123,
    )

    replayed_nomask = _evaluate(nomask)
    replayed_explicit = _evaluate(explicit)

    assert replayed_nomask.mask is np.ma.nomask
    assert replayed_explicit.mask is not np.ma.nomask
    assert replayed_explicit.mask.shape == data.shape
    assert semantic_equal(nomask, replayed_nomask, atol=0, rtol=0)
    assert semantic_equal(explicit, replayed_explicit, atol=0, rtol=0)
    assert not semantic_equal(replayed_nomask, replayed_explicit, atol=0, rtol=0)


def test_masked_replay_handles_masked_constant_and_scalar_array() -> None:
    assert replay_literal(np.ma.masked) == "np.ma.masked"
    assert _evaluate(np.ma.masked) is np.ma.masked

    scalar = np.ma.array(
        np.array(-np.inf, dtype=np.float64),
        mask=np.array(True),
        fill_value=-11.0,
        hard_mask=True,
    )
    replayed = _evaluate(scalar)

    assert replayed.shape == ()
    assert replayed.mask.shape == ()
    assert semantic_equal(scalar, replayed, atol=0, rtol=0)


def test_masked_replay_round_trips_inside_nested_containers() -> None:
    masked = np.ma.array(
        [1.0, -99.0],
        mask=[False, True],
        fill_value=-3.0,
    )
    value = {
        "payload": (masked, [np.ma.masked]),
        "nomask": np.ma.array([2, 3], mask=np.ma.nomask),
    }

    replayed = _evaluate(value)

    assert semantic_equal(value, replayed, atol=0, rtol=0)


def test_optional_ndarray_replay_preserves_shape_and_dtype() -> None:
    value = np.empty((0, 3), dtype=np.dtype(">f4"))

    assert replay_literal(value) == "[]"
    replayed = _evaluate(value, preserve_ndarray=True)

    assert isinstance(replayed, np.ndarray)
    assert replayed.shape == value.shape
    assert replayed.dtype == value.dtype

    nested = _evaluate(
        {"array": np.arange(6, dtype=np.uint16).reshape(2, 3)},
        preserve_ndarray=True,
    )
    assert nested["array"].shape == (2, 3)
    assert nested["array"].dtype == np.dtype(np.uint16)


def test_optional_ndarray_replay_preserves_structured_dtype() -> None:
    dtype = np.dtype([("count", ">i2"), ("score", "<f4")])
    value = np.array([[(2, 0.25), (7, np.inf)]], dtype=dtype)

    replayed = _evaluate(value, preserve_ndarray=True)

    assert replayed.shape == value.shape
    assert replayed.dtype == value.dtype
    np.testing.assert_equal(replayed, value)


def test_datetime_and_timedelta_literals_do_not_depend_on_numpy_repr() -> None:
    values = (
        np.datetime64("2024-01-02T03:04:05.000000006", "ns"),
        np.array("NaT", dtype="datetime64[2D]")[()],
        np.array(7, dtype="timedelta64[2h]")[()],
        np.timedelta64("NaT", "ns"),
    )

    for value in values:
        literal = replay_literal(value)
        replayed = eval(literal, {"np": np})
        assert literal.startswith(f"np.{type(value).__name__}(")
        assert replayed.dtype == value.dtype
        assert int(replayed.view(np.int64)) == int(value.view(np.int64))
