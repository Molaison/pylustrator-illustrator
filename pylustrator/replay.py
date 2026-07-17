"""Round-trip-safe literals for generated Python commands."""

from __future__ import annotations

import math

import numpy as np


_MASKED_CONSTANT_TYPE = type(np.ma.masked)


def replay_literal(value, *, preserve_ndarray: bool = False) -> str:
    """Return a Python literal suitable for a replay namespace containing ``np``.

    Python spells non-finite float representations as bare ``nan``/``inf``
    names, and fixed decimal formatting destroys small or large coordinates.
    Generated commands retain Python's shortest exact round-trip finite repr;
    source stability across undo is a transaction/bookkeeping responsibility,
    not a reason to quantize persistent geometry.

    Masked arrays always retain their raw data (including values hidden by the
    mask), exact mask representation, dtype, fill value, and hard-mask state.
    Ordinary ndarrays keep the compact, backwards-compatible list output by
    default.  Callers that need an ndarray's shape and dtype can opt in with
    ``preserve_ndarray=True``.
    """

    return _replay_literal(value, preserve_ndarray)


def _replay_literal(value, preserve_ndarray: bool) -> str:
    if isinstance(value, np.ndarray):
        if isinstance(value, _MASKED_CONSTANT_TYPE):
            return "np.ma.masked"
        if isinstance(value, np.ma.MaskedArray):
            return _masked_array_literal(value)
        if preserve_ndarray:
            return _ndarray_literal(value)
        value = value.tolist()
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return _datetime_like_literal(value)
    if isinstance(value, np.longdouble):
        return f"np.longdouble({_longdouble_text(value)!r})"
    if isinstance(value, np.clongdouble):
        real = _longdouble_text(value.real)
        imag = _longdouble_text(abs(value.imag))
        sign = "-" if np.signbit(value.imag) else "+"
        return f"np.clongdouble({f'{real}{sign}{imag}j'!r})"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        if math.isnan(value):
            return "np.nan"
        if value > 0:
            return "np.inf"
        return "-np.inf"
    if isinstance(value, complex):
        real = float(value.real)
        imag = float(value.imag)
        if math.isfinite(real) and math.isfinite(imag):
            return repr(value)
        return (
            f"complex({_replay_literal(real, False)}, {_replay_literal(imag, False)})"
        )
    if isinstance(value, list):
        return (
            "["
            + ", ".join(_replay_literal(item, preserve_ndarray) for item in value)
            + "]"
        )
    if isinstance(value, tuple):
        items = ", ".join(_replay_literal(item, preserve_ndarray) for item in value)
        if len(value) == 1:
            items += ","
        return f"({items})"
    if isinstance(value, dict):
        items = (
            f"{_replay_literal(key, preserve_ndarray)}: "
            f"{_replay_literal(item, preserve_ndarray)}"
            for key, item in value.items()
        )
        return "{" + ", ".join(items) + "}"
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return repr(value)
    raise TypeError(f"Unsupported generated-command value: {type(value).__name__}")


def _dtype_literal(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    descriptor = dtype.descr if dtype.fields is not None else dtype.str
    return f"np.dtype({_replay_literal(descriptor, False)})"


def _longdouble_text(value: np.longdouble) -> str:
    return np.format_float_scientific(value, unique=True, trim="k")


def _datetime_like_literal(value: np.datetime64 | np.timedelta64) -> str:
    """Serialize NumPy time scalars without relying on version-specific repr."""

    raw = int(value.view(np.int64))
    unit, count = np.datetime_data(value.dtype)
    constructor = (
        "np.datetime64" if isinstance(value, np.datetime64) else "np.timedelta64"
    )
    if unit == "generic":
        # A generic datetime can only represent NaT; generic timedeltas also
        # accept raw integers.  Keep these spellings explicit and evaluable in
        # every supported NumPy rather than emitting ``numpy.*`` on 1.23 and
        # ``np.*`` on newer releases.
        if np.isnat(value):
            return f"{constructor}('NaT')"
        return f"{constructor}({raw})"
    unit_literal = unit if count == 1 else f"{count}{unit}"
    return f"{constructor}({raw}, {unit_literal!r})"


def _ndarray_literal(value: np.ndarray) -> str:
    """Serialize an ndarray from a flat view without a large ``tolist`` copy."""

    flat = value.reshape(-1)
    items = ", ".join(_replay_literal(item, False) for item in flat)
    literal = f"np.array([{items}], dtype={_dtype_literal(value.dtype)})"
    if value.ndim != 1:
        literal += f".reshape({value.shape!r})"
    return literal


def _masked_array_literal(value: np.ma.MaskedArray) -> str:
    data = _ndarray_literal(np.asarray(value.data))
    if value.mask is np.ma.nomask:
        mask = "np.ma.nomask"
    else:
        mask = _ndarray_literal(np.asarray(value.mask))
    return (
        f"np.ma.array({data}, mask={mask}, "
        f"fill_value={_replay_literal(value.fill_value, False)}, "
        f"hard_mask={bool(value.hardmask)!r}, "
        f"dtype={_dtype_literal(value.dtype)})"
    )


__all__ = ["replay_literal"]
