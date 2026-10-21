"""Scientific data invariants: immutable input, deterministic output, strict schema."""

import copy

import pytest

from pylustrator.web.datasets import (
    ValidationError, apply_recipe, parse_dataset, preview_table, table_from_rows,
)


def test_csv_inference_missing_and_bounded_preview():
    table = parse_dataset("sample,value,active\nA,1,true\nB,,false\nC,2.5,true\n")
    assert table["columns"] == [
        {"name": "sample", "type": "string", "missing": 0},
        {"name": "value", "type": "number", "missing": 1},
        {"name": "active", "type": "boolean", "missing": 0},
    ]
    preview = preview_table(table, 1, 1)
    assert preview["rowCount"] == 3
    assert preview["rows"] == [{"sample": "B", "value": None, "active": False}]
    assert preview["hash"] == table["hash"]
    preview["rows"][0]["sample"] = "changed"
    assert table["rows"][1]["sample"] == "B"


@pytest.mark.parametrize("content,format", [
    ("x,x\n1,2", "csv"),
    ("x,y\n1", "csv"),
    ('x\n"unterminated', "csv"),
    ("x\nInfinity", "csv"),
    ('[{"x": NaN}]', "json"),
    ('[{"x": {"code": "bad"}}]', "json"),
    ('[{"x":1},{"x":2,"other":3}]', "json"),
    ('{"columns":["x"],"rows":[[1,2]]}', "json"),
    ("x\n1", "python"),
])
def test_malformed_sources_are_rejected(content, format):
    with pytest.raises(ValidationError):
        parse_dataset(content, format)


def test_zero_row_table_retains_schema():
    table = parse_dataset("x,y\n")
    assert table["rowCount"] == 0
    assert [c["name"] for c in table["columns"]] == ["x", "y"]


def test_tsv_and_json_equivalent_content_have_same_hash():
    tsv = parse_dataset("x\ty\n1\t2\n3\t4\n", "tsv")
    records = parse_dataset('[{"x":1,"y":2},{"x":3,"y":4}]', "json")
    assert tsv["hash"] == records["hash"]


def test_recipe_is_reproducible_and_original_never_changes():
    original = parse_dataset("group,x\nA,2\nB,\nA,4\nB,6\n")
    before = copy.deepcopy(original)
    recipe = [
        {"op": "fill_missing", "columns": ["x"], "strategy": "mean"},
        {"op": "derive", "column": "x", "target": "normalized", "method": "normalize"},
        {"op": "sort", "column": "normalized", "descending": True},
        {"op": "rename", "mapping": {"group": "condition"}},
        {"op": "select", "columns": ["condition", "normalized"]},
    ]
    result = apply_recipe(original, recipe)
    assert result["rows"] == [
        {"condition": "B", "normalized": 1.0},
        {"condition": "B", "normalized": 0.5},
        {"condition": "A", "normalized": 0.5},
        {"condition": "A", "normalized": 0.0},
    ]
    assert original == before
    assert apply_recipe(original, recipe)["hash"] == result["hash"]
    assert result["inputHash"] == original["hash"]


def test_aggregate_handles_missing_values_and_typed_groups():
    source = table_from_rows([{"g": True, "x": 2}, {"g": 1, "x": 4}, {"g": 1, "x": None}])
    result = apply_recipe(source, [{"op": "aggregate", "groupBy": ["g"], "column": "x", "function": "mean"}])
    assert result["rowCount"] == 2
    assert result["rows"] == [{"g": True, "x_mean": 2}, {"g": 1, "x_mean": 4}]


def test_missing_last_sort_and_filter():
    source = parse_dataset("x,name\n2,B\n,A\n1,C\n")
    result = apply_recipe(source, [{"op": "sort", "column": "x", "descending": True}])
    assert [r["x"] for r in result["rows"]] == [2, 1, None]
    result = apply_recipe(source, [{"op": "filter", "column": "x", "operator": "gte", "value": 2}])
    assert result["rows"] == [{"x": 2, "name": "B"}]


@pytest.mark.parametrize("steps", [
    [{"op": "exec", "code": "import os"}],
    [{"op": []}],
    [{"op": "drop_missing", "columns": "x"}],
    [{"op": "drop_missing", "columns": None}],
    [{"op": "filter", "column": "unknown", "operator": "eq", "value": 1}],
    [{"op": "derive", "column": "x", "target": "x", "method": "normalize"}],
    [{"op": "derive", "column": "x", "target": "y", "method": "eval"}],
    [{"op": "derive", "column": "x", "target": "y", "method": "log"}],
    [{"op": "rename", "mapping": {"x": "name"}}],
    [{"op": "sort", "column": "x", "descending": "false"}],
    [{"op": "filter", "column": "x", "operator": "gt", "value": "apple"}],
])
def test_invalid_recipe_has_no_side_effects(steps):
    source = parse_dataset("x,name\n-1,A\n2,B\n")
    before = copy.deepcopy(source)
    with pytest.raises(ValidationError):
        apply_recipe(source, steps)
    assert source == before


def test_constant_column_normalization_and_standardization_are_finite():
    source = parse_dataset("x\n2\n2\n")
    for method in ("normalize", "zscore"):
        result = apply_recipe(source, [{"op": "derive", "column": "x", "target": "y", "method": method}])
        assert [r["y"] for r in result["rows"]] == [0, 0]


def test_preview_rejects_unbounded_or_negative_requests():
    source = parse_dataset("x\n1")
    for offset, limit in ((-1, 100), (0, 1000), (0, 0), (0, True)):
        with pytest.raises(ValidationError):
            preview_table(source, offset, limit)
