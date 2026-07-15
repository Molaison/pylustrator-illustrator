"""Round-trip-safe literals for generated Python commands."""

from __future__ import annotations

import numpy as np


def replay_literal(value) -> str:
    """Return a Python literal suitable for a replay namespace containing ``np``.

    Python spells non-finite float representations as bare ``nan``/``inf``
    names, and fixed decimal formatting destroys small or large coordinates.
    Generated commands retain Python's shortest exact round-trip finite repr;
    source stability across undo is a transaction/bookkeeping responsibility,
    not a reason to quantize persistent geometry.
    """

    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "np.nan"
        if np.isposinf(value):
            return "np.inf"
        if np.isneginf(value):
            return "-np.inf"
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(replay_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        items = ", ".join(replay_literal(item) for item in value)
        if len(value) == 1:
            items += ","
        return f"({items})"
    if isinstance(value, dict):
        items = (
            f"{replay_literal(key)}: {replay_literal(item)}"
            for key, item in value.items()
        )
        return "{" + ", ".join(items) + "}"
    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    raise TypeError(f"Unsupported generated-command value: {type(value).__name__}")


__all__ = ["replay_literal"]
