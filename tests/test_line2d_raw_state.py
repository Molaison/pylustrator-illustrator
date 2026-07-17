from __future__ import annotations

from copy import deepcopy
from types import MethodType

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform

from pylustrator.artist_adapters import (
    UnsupportedArtistError,
    get_artist_adapter,
)
from pylustrator.commands import semantic_equal
from pylustrator.operations import TransformOperation
from pylustrator.snap import TargetWrapper


class RecordingChangeTracker:
    def __init__(self) -> None:
        self.calls = []

    def addChange(self, target, command) -> None:
        self.calls.append((target, command))


def _identity_line(xdata, ydata, **kwargs):
    fig = plt.figure(figsize=(5, 4), dpi=100)
    fig.change_tracker = RecordingChangeTracker()
    target = Line2D(
        xdata,
        ydata,
        transform=IdentityTransform(),
        clip_on=False,
        **kwargs,
    )
    fig.add_artist(target)
    fig.canvas.draw()
    return fig, target, get_artist_adapter(target)


def _assert_masked_exact(actual, expected) -> None:
    assert isinstance(actual, np.ma.MaskedArray)
    assert isinstance(expected, np.ma.MaskedArray)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.data, expected.data)
    assert (actual.mask is np.ma.nomask) == (expected.mask is np.ma.nomask)
    if actual.mask is not np.ma.nomask:
        np.testing.assert_array_equal(actual.mask, expected.mask)
    assert actual.fill_value == expected.fill_value
    assert actual.hardmask is expected.hardmask


def test_masked_raw_snapshot_translate_restore_preserves_every_payload() -> None:
    xdata = np.ma.array(
        [50.0, 111.0, np.nan, 200.0, np.inf, 275.0],
        mask=[False, True, False, False, True, False],
        fill_value=1234.5,
        hard_mask=True,
    )
    ydata = np.ma.array(
        [60.0, 222.0, 333.0, 210.0, 140.0, -np.inf],
        mask=[False, False, True, False, False, True],
        fill_value=-987.5,
        hard_mask=False,
    )
    fig, target, adapter = _identity_line(xdata, ydata, marker="o")
    before_x = deepcopy(target.get_xdata(orig=True))
    before_y = deepcopy(target.get_ydata(orig=True))
    state = adapter.snapshot()

    try:
        assert not np.shares_memory(state["xdata"].data, before_x.data)
        assert not np.shares_memory(state["ydata"].data, before_y.data)

        adapter.translate((7.0, -3.0))
        after_x = target.get_xdata(orig=True)
        after_y = target.get_ydata(orig=True)
        eligible = np.array([True, False, False, True, False, False])
        np.testing.assert_array_equal(
            after_x.data[eligible], before_x.data[eligible] + 7.0
        )
        np.testing.assert_array_equal(
            after_y.data[eligible], before_y.data[eligible] - 3.0
        )
        np.testing.assert_array_equal(after_x.data[~eligible], before_x.data[~eligible])
        np.testing.assert_array_equal(after_y.data[~eligible], before_y.data[~eligible])
        np.testing.assert_array_equal(after_x.mask, before_x.mask)
        np.testing.assert_array_equal(after_y.mask, before_y.mask)
        assert after_x.fill_value == before_x.fill_value
        assert after_y.fill_value == before_y.fill_value
        assert after_x.hardmask and not after_y.hardmask
        assert len(fig.change_tracker.calls) == 1
        assert "np.ma.array" in fig.change_tracker.calls[0][1]
        moved_state = adapter.snapshot()

        adapter.restore(state)
        _assert_masked_exact(target.get_xdata(orig=True), before_x)
        _assert_masked_exact(target.get_ydata(orig=True), before_y)

        adapter.restore(moved_state)
        _assert_masked_exact(target.get_xdata(orig=True), after_x)
        _assert_masked_exact(target.get_ydata(orig=True), after_y)
    finally:
        plt.close(fig)


def test_q_rotation_preserves_independently_invalid_raw_rows() -> None:
    xdata = np.ma.array(
        [100.0, 777.0, 200.0, 300.0],
        mask=[False, True, False, False],
        fill_value=81.0,
        hard_mask=True,
    )
    ydata = np.ma.array(
        [100.0, 150.0, 888.0, 200.0],
        mask=[False, False, True, False],
        fill_value=91.0,
    )
    fig, target, adapter = _identity_line(xdata, ydata, marker="o")
    before_x = deepcopy(target.get_xdata(orig=True))
    before_y = deepcopy(target.get_ydata(orig=True))

    try:
        plan = adapter.plan_rigid_rotation(30.0, (200.0, 150.0))
        np.testing.assert_array_equal(
            plan.native_array()[1:3],
            np.array([[np.nan, 150.0], [200.0, np.nan]]),
        )
        adapter.apply_rigid_rotation_plan(plan)
        after_x = target.get_xdata(orig=True)
        after_y = target.get_ydata(orig=True)
        np.testing.assert_array_equal(after_x.data[1:3], before_x.data[1:3])
        np.testing.assert_array_equal(after_y.data[1:3], before_y.data[1:3])
        np.testing.assert_array_equal(after_x.mask, before_x.mask)
        np.testing.assert_array_equal(after_y.mask, before_y.mask)
        assert after_x.fill_value == before_x.fill_value
        assert after_y.fill_value == before_y.fill_value
    finally:
        plt.close(fig)


def test_nomask_and_explicit_all_false_mask_remain_distinct() -> None:
    xdata = np.ma.array(
        [50.0, 100.0, 150.0], mask=np.ma.nomask, fill_value=19.0
    )
    ydata = np.ma.array(
        [60.0, 110.0, 160.0],
        mask=np.zeros(3, dtype=bool),
        fill_value=29.0,
    )
    fig, target, adapter = _identity_line(xdata, ydata)

    try:
        adapter.translate((4.0, 5.0))
        assert target.get_xdata(orig=True).mask is np.ma.nomask
        assert target.get_ydata(orig=True).mask is not np.ma.nomask
        np.testing.assert_array_equal(
            target.get_ydata(orig=True).mask, np.zeros(3, dtype=bool)
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.array([5.0]), np.array([5.0, 5.0, 5.0])),
        (np.arange(4.0).reshape(2, 2), np.arange(4.0)),
        (np.array([5.0, 6.0], dtype=np.float32), np.array([5.0, 6.0])),
        ([5.0, 6.0], np.array([5.0, 6.0])),
    ],
    ids=["broadcast-length", "shape", "dtype", "container-kind"],
)
def test_snapshot_metadata_exposes_raw_storage_only_changes(left, right) -> None:
    left_y = np.arange(np.size(left), dtype=float) + 50.0
    if np.size(left) == 1:
        left_y = np.array([50.0, 60.0, 70.0])
    right_y = np.arange(np.size(right), dtype=float) + 50.0
    fig, target, adapter = _identity_line(left, left_y)
    left_state = adapter.snapshot()

    try:
        target.set_data(right, right_y)
        right_state = adapter.snapshot()
        assert not semantic_equal(left_state, right_state)
        assert left_state["xdata_metadata"] != right_state["xdata_metadata"]
    finally:
        plt.close(fig)


def test_masked_raw_serialization_replays_without_filling_masked_values() -> None:
    xdata = np.ma.array(
        [[50.0, np.nan], [777.0, 200.0]],
        mask=[[False, False], [True, False]],
        fill_value=4321.0,
        hard_mask=True,
        dtype=np.float32,
    )
    ydata = np.ma.array(
        [[60.0, 888.0], [150.0, 210.0]],
        mask=[[False, True], [False, False]],
        fill_value=-1234.0,
        dtype=np.float32,
    )
    fig, target, adapter = _identity_line(xdata, ydata)

    try:
        command = adapter.serialize_changes()[0].command
        replay_target = Line2D([0.0], [0.0])
        exec(f"replay_target{command}", {"replay_target": replay_target, "np": np})
        _assert_masked_exact(replay_target.get_xdata(orig=True), xdata)
        _assert_masked_exact(replay_target.get_ydata(orig=True), ydata)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("xdata", "ydata", "expected_x_type", "expected_y_type"),
    [
        (
            np.array([[50.0, 100.0], [150.0, 200.0]]),
            np.array([[60.0, 110.0], [160.0, 210.0]]),
            np.ndarray,
            np.ndarray,
        ),
        (
            [[50.0, 100.0], [150.0, 200.0]],
            ((60.0, 110.0), (160.0, 210.0)),
            list,
            tuple,
        ),
    ],
    ids=["ndarray-2d", "nested-sequences"],
)
def test_ravel_storage_preserves_shape_and_container_type(
    xdata, ydata, expected_x_type, expected_y_type
) -> None:
    fig, target, adapter = _identity_line(xdata, ydata)

    try:
        adapter.translate((5.0, -2.0))
        actual_x = target.get_xdata(orig=True)
        actual_y = target.get_ydata(orig=True)
        assert type(actual_x) is expected_x_type
        assert type(actual_y) is expected_y_type
        assert np.shape(actual_x) == (2, 2)
        assert np.shape(actual_y) == (2, 2)
        np.testing.assert_allclose(np.asarray(actual_x), np.asarray(xdata) + 5.0)
        np.testing.assert_allclose(np.asarray(actual_y), np.asarray(ydata) - 2.0)
    finally:
        plt.close(fig)


def test_length_one_broadcast_supports_compressible_moves_and_rejects_q() -> None:
    fig, target, adapter = _identity_line(
        np.array([100.0]), np.array([50.0, 100.0, 150.0])
    )

    try:
        adapter.translate((5.0, 7.0))
        np.testing.assert_array_equal(target.get_xdata(orig=True), [105.0])
        np.testing.assert_array_equal(
            target.get_ydata(orig=True), [57.0, 107.0, 157.0]
        )
        before_x = target.get_xdata(orig=True).copy()
        before_y = target.get_ydata(orig=True).copy()
        calls_before = list(fig.change_tracker.calls)
        with pytest.raises(UnsupportedArtistError, match="length-one x broadcast"):
            adapter.plan_rigid_rotation(20.0, (100.0, 100.0))
        np.testing.assert_array_equal(target.get_xdata(orig=True), before_x)
        np.testing.assert_array_equal(target.get_ydata(orig=True), before_y)
        assert fig.change_tracker.calls == calls_before
    finally:
        plt.close(fig)


def test_broadcast_shared_with_invalid_row_cannot_change_hidden_logical_row() -> None:
    fig, target, adapter = _identity_line(
        np.array([100.0]), np.array([50.0, np.nan, 150.0])
    )
    before_x = target.get_xdata(orig=True).copy()
    before_y = target.get_ydata(orig=True).copy()

    try:
        with pytest.raises(UnsupportedArtistError, match="shared with non-finite rows"):
            adapter.translate((5.0, 0.0))
        np.testing.assert_array_equal(target.get_xdata(orig=True), before_x)
        np.testing.assert_array_equal(target.get_ydata(orig=True), before_y)
        assert not fig.change_tracker.calls
    finally:
        plt.close(fig)


def test_integer_and_float32_storage_never_silently_truncate() -> None:
    fig, target, adapter = _identity_line(
        np.array([50, 100], dtype=np.int32),
        np.array([60, 110], dtype=np.int32),
    )

    try:
        adapter.translate((0.8, 0.0))
        assert target.get_xdata(orig=True).dtype == np.dtype("int32")
        np.testing.assert_array_equal(target.get_xdata(orig=True), [51, 101])
    finally:
        plt.close(fig)


def test_integer_promotion_keeps_unrepresentable_masked_payload_exact() -> None:
    hidden = 2**60 + 1
    fill = 2**61 + 3
    xdata = np.ma.array(
        [50, hidden, 100],
        mask=[False, True, False],
        fill_value=fill,
        hard_mask=True,
        dtype=np.int64,
    )
    fig, target, adapter = _identity_line(xdata, [60.0, 80.0, 110.0])

    try:
        adapter.translate((0.6, 0.0))
        actual = target.get_xdata(orig=True)
        assert actual.dtype == np.dtype(object)
        assert actual.data[1] == hidden
        assert actual.fill_value == fill
        assert actual.hardmask
        np.testing.assert_array_equal(actual.mask, xdata.mask)
        np.testing.assert_allclose(
            np.asarray(actual.data[[0, 2]], dtype=float), [50.6, 100.6]
        )
    finally:
        plt.close(fig)

    fig, target, adapter = _identity_line(
        np.array([50, 100], dtype=np.int32),
        np.array([60, 110], dtype=np.int32),
    )
    try:
        adapter.translate((0.6, 0.0))
        assert target.get_xdata(orig=True).dtype == np.dtype(float)
        np.testing.assert_allclose(target.get_xdata(orig=True), [50.6, 100.6])
    finally:
        plt.close(fig)

    fig, target, adapter = _identity_line(
        np.array([1.0e8, 1.0e8 + 32], dtype=np.float32),
        np.array([60.0, 110.0], dtype=np.float32),
    )
    try:
        adapter.translate((1.0, 0.0))
        assert target.get_xdata(orig=True).dtype == np.dtype(float)
        np.testing.assert_allclose(
            target.get_xdata(orig=True), [1.0e8 + 1.0, 1.0e8 + 33.0]
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["datetime", "category", "custom-unit"])
def test_non_numeric_domains_are_typed_denials_but_snapshot_restore(kind) -> None:
    if kind == "datetime":
        fig = plt.figure(figsize=(5, 4), dpi=100)
        xdata = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
        target = Line2D(xdata, [1.0, 2.0], transform=IdentityTransform())
        fig.add_artist(target)
    elif kind == "category":
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        xdata = ["alpha", "beta"]
        target = ax.plot(xdata, [1.0, 2.0])[0]
    else:
        class UnitValue:
            def __init__(self, value) -> None:
                self.value = value

            def __float__(self) -> float:
                return float(self.value)

        fig = plt.figure(figsize=(5, 4), dpi=100)
        xdata = [UnitValue(1.0), UnitValue(2.0)]
        target = Line2D(xdata, [1.0, 2.0], transform=IdentityTransform())
        fig.add_artist(target)
    fig.change_tracker = RecordingChangeTracker()
    fig.canvas.draw()
    adapter = get_artist_adapter(target)
    state = adapter.snapshot()

    try:
        assert adapter.capabilities.can_snapshot
        assert adapter.capabilities.can_select
        assert TargetWrapper.supports_target(target)
        assert TargetWrapper(target).supported
        assert not adapter.operation_support(
            TransformOperation.TRANSLATE
        ).supported
        assert adapter.operation_support(TransformOperation.SERIALIZE).supported
        assert adapter.operation_support(
            TransformOperation.SCALE_APPEARANCE
        ).supported
        with pytest.raises(UnsupportedArtistError, match="categorical|datetime|custom"):
            adapter.translate((5.0, 0.0))
        if kind == "custom-unit":
            with pytest.raises(UnsupportedArtistError, match="cannot serialize"):
                adapter.record_changes()
            assert not fig.change_tracker.calls
        else:
            adapter.record_changes()
            assert len(fig.change_tracker.calls) == 1
            replay_target = Line2D([0.0], [0.0])
            exec(
                f"replay_target{fig.change_tracker.calls[-1][1]}",
                {"replay_target": replay_target, "np": np},
            )
            np.testing.assert_array_equal(
                replay_target.get_xdata(orig=True), state["xdata"]
            )

        target.set_data(deepcopy(xdata), [3.0, 4.0])
        adapter.restore(state)
        np.testing.assert_array_equal(
            target.get_ydata(orig=True), state["ydata"]
        )
        if kind == "datetime":
            np.testing.assert_array_equal(target.get_xdata(orig=True), state["xdata"])
        elif kind == "category":
            np.testing.assert_array_equal(target.get_xdata(orig=True), state["xdata"])
        else:
            assert [value.value for value in target.get_xdata(orig=True)] == [1.0, 2.0]
        expected_calls = 0 if kind == "custom-unit" else 2
        assert len(fig.change_tracker.calls) == expected_calls
    finally:
        plt.close(fig)


def test_date_axis_native_float_data_remains_writable() -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    numeric_dates = np.array([19723.0, 19724.0])
    target = ax.plot(numeric_dates, [1.0, 2.0])[0]
    set_converter = getattr(ax.xaxis, "set_converter", None)
    if callable(set_converter):
        set_converter(mdates.DateConverter())
    else:  # Matplotlib 3.8 compatibility
        ax.xaxis.converter = mdates.DateConverter()
    fig.change_tracker = RecordingChangeTracker()
    fig.canvas.draw()
    adapter = get_artist_adapter(target)

    try:
        assert target.get_xdata(orig=True).dtype.kind == "f"
        assert adapter.capabilities.can_translate
        adapter.translate((3.0, 0.0))
        assert len(fig.change_tracker.calls) == 1
    finally:
        plt.close(fig)


def test_custom_coordinate_domain_does_not_block_appearance_editing() -> None:
    class UnitValue:
        def __init__(self, value) -> None:
            self.value = value

        def __float__(self) -> float:
            return float(self.value)

    fig, target, adapter = _identity_line(
        [UnitValue(80.0), UnitValue(220.0)],
        [70.0, 120.0],
        marker="o",
        linewidth=2.0,
        markersize=6.0,
    )

    try:
        plan = adapter.plan_appearance_scale(1.5)
        adapter.apply_appearance_scale_plan(plan)
        assert target.get_linewidth() == pytest.approx(3.0)
        assert target.get_markersize() == pytest.approx(9.0)
        assert len(fig.change_tracker.calls) == 3
        assert all("set_data" not in command for _target, command in fig.change_tracker.calls)
    finally:
        plt.close(fig)


def test_unrepresentable_raw_shape_is_denied_before_mutation() -> None:
    xdata = np.arange(8.0).reshape(2, 2, 2) + 50.0
    ydata = np.arange(8.0) + 60.0
    fig, target, adapter = _identity_line(xdata, ydata)
    before_x = target.get_xdata(orig=True).copy()
    before_y = target.get_ydata(orig=True).copy()

    try:
        assert adapter.capabilities.can_select
        assert TargetWrapper(target).supported
        assert not adapter.operation_support(
            TransformOperation.TRANSLATE
        ).supported
        with pytest.raises(UnsupportedArtistError, match="one- or two-dimensional"):
            adapter.translate((5.0, 0.0))
        np.testing.assert_array_equal(target.get_xdata(orig=True), before_x)
        np.testing.assert_array_equal(target.get_ydata(orig=True), before_y)
        assert not fig.change_tracker.calls
    finally:
        plt.close(fig)


def test_stale_rigid_plan_cannot_overwrite_new_raw_data() -> None:
    fig, target, adapter = _identity_line(
        np.array([80.0, 140.0, 220.0]),
        np.array([70.0, 180.0, 120.0]),
        marker="o",
    )
    plan = adapter.plan_rigid_rotation(25.0, (150.0, 120.0))
    replacement_x = np.array([90.0, 150.0, 230.0])
    replacement_y = np.array([75.0, 185.0, 125.0])
    target.set_data(replacement_x, replacement_y)

    try:
        with pytest.raises(UnsupportedArtistError, match="stale plan"):
            adapter.apply_rigid_rotation_plan(plan)
        np.testing.assert_array_equal(target.get_xdata(orig=True), replacement_x)
        np.testing.assert_array_equal(target.get_ydata(orig=True), replacement_y)
        assert not fig.change_tracker.calls
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "mutation",
    ["finite-data", "hidden-data", "mask", "fill-value", "hard-mask"],
)
def test_stale_rigid_plan_detects_in_place_raw_mutation(mutation) -> None:
    xdata = np.ma.array(
        [80.0, 987.0, 220.0],
        mask=[False, True, False],
        fill_value=4321.0,
        hard_mask=True,
    )
    fig, target, adapter = _identity_line(
        xdata,
        np.array([70.0, 180.0, 120.0]),
        marker="o",
    )
    plan = adapter.plan_rigid_rotation(25.0, (150.0, 120.0))
    raw = target.get_xdata(orig=True)
    assert isinstance(plan.source_fingerprint, tuple)
    assert len(plan.source_fingerprint) == 3
    assert isinstance(plan.source_fingerprint[2], bytes)
    assert len(plan.source_fingerprint[2]) == 16

    if mutation == "finite-data":
        raw.data[0] += 1.0
    elif mutation == "hidden-data":
        raw.data[1] += 1.0
    elif mutation == "mask":
        raw.mask[1] = False
    elif mutation == "fill-value":
        raw.fill_value = 12345.0
    else:
        raw.soften_mask()
    mutated = deepcopy(raw)

    try:
        with pytest.raises(UnsupportedArtistError, match="stale plan"):
            adapter.apply_rigid_rotation_plan(plan)
        _assert_masked_exact(target.get_xdata(orig=True), mutated)
        assert not fig.change_tracker.calls
    finally:
        plt.close(fig)


def test_numeric_first_heterogeneous_list_is_typed_at_preflight() -> None:
    class UnitValue:
        def __init__(self, value) -> None:
            self.value = value

        def __float__(self) -> float:
            return float(self.value)

    xdata = [80.0, UnitValue(140.0), 220.0]
    fig, target, adapter = _identity_line(
        xdata,
        [70.0, 180.0, 120.0],
        marker="o",
    )
    raw = target.get_xdata(orig=True)

    try:
        assert adapter.operation_support(TransformOperation.TRANSLATE).supported
        with pytest.raises(UnsupportedArtistError, match="object coordinates"):
            adapter.translate((5.0, 0.0))
        assert target.get_xdata(orig=True) is raw
        assert raw[0] == 80.0
        assert raw[1].value == 140.0
        assert not fig.change_tracker.calls
    finally:
        plt.close(fig)


def test_set_data_half_write_failure_rolls_back_raw_and_processed_state() -> None:
    fig, target, adapter = _identity_line(
        np.array([80.0, 140.0, 220.0]),
        np.array([70.0, 180.0, 120.0]),
    )
    raw_x = target.get_xdata(orig=True)
    raw_y = target.get_ydata(orig=True)
    processed = target.get_xydata()
    path = target.get_path()
    invalid_flags = (target._invalidx, target._invalidy)
    tracker_before = list(fig.change_tracker.calls)
    original_set_ydata = target.set_ydata

    def fail_ydata(_self, _values) -> None:
        raise RuntimeError("injected y half-write failure")

    target.set_ydata = MethodType(fail_ydata, target)
    try:
        with pytest.raises(RuntimeError, match="half-write"):
            adapter.translate((5.0, 7.0))
        assert target.get_xdata(orig=True) is raw_x
        assert target.get_ydata(orig=True) is raw_y
        assert target.get_xydata() is processed
        assert target.get_path() is path
        assert (target._invalidx, target._invalidy) == invalid_flags
        assert fig.change_tracker.calls == tracker_before
    finally:
        target.set_ydata = original_set_ydata
        plt.close(fig)
