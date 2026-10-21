"""Bounded tabular imports and deterministic, non-executable data recipes.

Rows are ordinary JSON scalar records. Every recipe returns a new table, keeping
the source immutable and making its content hash suitable for figure provenance.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from copy import deepcopy
from typing import Any

MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 256
MAX_STEPS = 50
MAX_CELL_CHARS = 100_000
MAX_PREVIEW_ROWS = 500


class ValidationError(ValueError):
    """Input cannot be safely represented by the research-data contract."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValidationError("Data must contain finite JSON values.") from exc


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (float, int)):
        try:
            if not math.isfinite(value):
                raise ValidationError("Non-finite numbers are not accepted; use null for missing data.")
        except OverflowError as exc:
            raise ValidationError("Number is outside the supported finite range.") from exc
        return value
    if isinstance(value, str):
        if len(value) > MAX_CELL_CHARS:
            raise ValidationError(f"A data cell exceeds {MAX_CELL_CHARS} characters.")
        return value
    raise ValidationError("Table cells must be strings, finite numbers, booleans, or null.")


def _names(columns: Any) -> list[str]:
    if not isinstance(columns, list) or not columns or len(columns) > MAX_COLUMNS:
        raise ValidationError(f"Provide between 1 and {MAX_COLUMNS} columns.")
    if any(not isinstance(c, str) or not c.strip() or len(c) > 256 for c in columns):
        raise ValidationError("Column names must be nonempty strings of at most 256 characters.")
    if len(set(columns)) != len(columns):
        raise ValidationError("Duplicate column names are not allowed.")
    return columns


def table_from_rows(rows: Any, columns: list[str] | None = None) -> dict:
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise ValidationError(f"A dataset may contain at most {MAX_ROWS} rows.")
    if columns is None:
        if not rows or not isinstance(rows[0], dict):
            raise ValidationError("An empty table needs explicit column names.")
        columns = list(rows[0])
    columns = _names(columns)
    allowed = set(columns)
    cleaned = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValidationError(f"Row {index + 1} has invalid or unexpected columns.")
        cleaned.append({c: _scalar(row.get(c)) for c in columns})
    schema = []
    for column in columns:
        values = [r[column] for r in cleaned if r[column] is not None]
        if not values:
            kind = "unknown"
        elif all(isinstance(v, bool) for v in values):
            kind = "boolean"
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            kind = "number"
        elif all(isinstance(v, str) for v in values):
            kind = "string"
        else:
            kind = "mixed"
        schema.append({"name": column, "type": kind, "missing": len(cleaned) - len(values)})
    payload = {"columns": columns, "rows": cleaned}
    return {"columns": schema, "rows": cleaned, "rowCount": len(cleaned),
            "hash": content_hash(payload)}


def parse_dataset(content: str | bytes, format: str = "csv") -> dict:
    if isinstance(content, bytes):
        if len(content) > MAX_SOURCE_BYTES:
            raise ValidationError("Dataset exceeds the 16 MiB import limit.")
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValidationError("Upload UTF-8 encoded data.") from exc
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValidationError("Dataset must be UTF-8 text of at most 16 MiB.")
    content = content.lstrip("\ufeff")
    if not isinstance(format, str) or format not in {"csv", "tsv", "json"}:
        raise ValidationError("Supported data formats are csv, tsv, and json.")
    if format == "json":
        try:
            data = json.loads(content, parse_constant=lambda x: _reject_constant(x))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValidationError("Invalid dataset JSON.") from exc
        if isinstance(data, list):
            return table_from_rows(data)
        if isinstance(data, dict) and set(data).issubset({"columns", "rows"}):
            names = _names(data.get("columns"))
            rows = data.get("rows")
            if isinstance(rows, list) and rows and isinstance(rows[0], list):
                if any(not isinstance(r, list) or len(r) != len(names) for r in rows):
                    raise ValidationError("JSON rows must match the column count.")
                rows = [dict(zip(names, r)) for r in rows]
            return table_from_rows(rows, names)
        raise ValidationError("JSON data must be records or {columns, rows}.")
    try:
        reader = csv.reader(io.StringIO(content, newline=""), delimiter="\t" if format == "tsv" else ",",
                            strict=True)
        names = _names(next(reader, []))
        rows = []
        for number, cells in enumerate(reader, 2):
            if not cells:
                continue
            if len(rows) >= MAX_ROWS:
                raise ValidationError(f"A dataset may contain at most {MAX_ROWS} rows.")
            if len(cells) != len(names):
                raise ValidationError(f"Row {number} has {len(cells)} cells; expected {len(names)}.")
            rows.append({c: (v if v.strip() else None) for c, v in zip(names, cells)})
    except csv.Error as exc:
        raise ValidationError(f"Invalid delimited data: {exc}") from exc
    # Infer a whole column, never convert just a convenient subset of its values.
    for name in names:
        present = [r[name] for r in rows if r[name] is not None]
        if not present:
            continue
        converted = []
        numeric = True
        for value in present:
            try:
                number = float(value)
            except ValueError:
                numeric = False
                break
            if not math.isfinite(number):
                raise ValidationError(f"Column '{name}' contains a non-finite number; use an empty cell.")
            converted.append(int(value) if value.strip().lstrip("+-").isdigit() else number)
        if numeric:
            values = iter(converted)
            for row in rows:
                if row[name] is not None:
                    row[name] = next(values)
        elif all(value.strip().lower() in {"true", "false"} for value in present):
            for row in rows:
                if row[name] is not None:
                    row[name] = row[name].strip().lower() == "true"
    return table_from_rows(rows, names)


def _reject_constant(value: str):
    raise ValidationError(f"Non-finite JSON constant {value} is not accepted.")


def preview_table(table: dict, offset: int = 0, limit: int = 100) -> dict:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValidationError("Preview offset must be a nonnegative integer.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PREVIEW_ROWS:
        raise ValidationError(f"Preview limit must be between 1 and {MAX_PREVIEW_ROWS}.")
    result = {key: deepcopy(value) for key, value in table.items() if key != "rows"}
    result.update(rows=deepcopy(table["rows"][offset:offset + limit]), offset=offset, limit=limit)
    return result


OPERATION_SCHEMAS = {
    "drop_missing": {"required": [], "optional": ["columns"], "description": "Remove rows with missing values."},
    "fill_missing": {"required": ["columns", "strategy"], "optional": ["value"], "description": "Fill missing values with a constant, mean, or median."},
    "filter": {"required": ["column", "operator", "value"], "optional": [], "description": "Keep rows matching a comparison."},
    "sort": {"required": ["column"], "optional": ["descending"], "description": "Stable sort, with missing values last."},
    "select": {"required": ["columns"], "optional": [], "description": "Keep selected columns in the supplied order."},
    "rename": {"required": ["mapping"], "optional": [], "description": "Rename columns."},
    "derive": {"required": ["column", "target", "method"], "optional": ["factor"], "description": "Add a numeric column using normalize, zscore, log, log10, or scale."},
    "aggregate": {"required": ["column", "function"], "optional": ["groupBy", "target"], "description": "Group rows and summarize a column."},
}


def operations() -> list[dict]:
    return [{"op": key, **deepcopy(value)} for key, value in OPERATION_SCHEMAS.items()]


def validate_recipe(steps: Any) -> list[dict]:
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise ValidationError(f"A recipe must be a list of at most {MAX_STEPS} steps.")
    if len(canonical_json(steps).encode("utf-8")) > 128 * 1024:
        raise ValidationError("Recipe is too large.")
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("op"), str) or step["op"] not in OPERATION_SCHEMAS:
            raise ValidationError(f"Step {index + 1} has an unknown operation.")
        spec = OPERATION_SCHEMAS[step["op"]]
        if not set(spec["required"]).issubset(step):
            raise ValidationError(f"Step {index + 1} is missing required parameters: {spec['required']}.")
        if set(step) - {"op", *spec["required"], *spec["optional"]}:
            raise ValidationError(f"Step {index + 1} contains unsupported parameters.")
    return deepcopy(steps)


def _numeric(values: list, column: str) -> list[float]:
    present = [v for v in values if v is not None]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in present):
        raise ValidationError(f"Column '{column}' must contain numbers for this operation.")
    return present


def _column(value: Any, columns: list[str]) -> str:
    if not isinstance(value, str) or value not in columns:
        raise ValidationError(f"Unknown column: {value!r}.")
    return value


def _columns(value: Any, columns: list[str], *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty) or len(set(str(v) for v in value)) != len(value):
        raise ValidationError("Columns must be a list of unique column names.")
    return [_column(v, columns) for v in value]


def apply_recipe(table: dict, steps: list[dict]) -> dict:
    steps = validate_recipe(steps)
    columns = [c["name"] if isinstance(c, dict) else c for c in table["columns"]]
    rows = deepcopy(table["rows"])
    for number, step in enumerate(steps, 1):
        op = step["op"]
        try:
            if op == "drop_missing":
                selected = _columns(step.get("columns", columns), columns, allow_empty=True) or columns
                rows = [r for r in rows if all(r[c] is not None for c in selected)]
            elif op == "fill_missing":
                selected = _columns(step["columns"], columns)
                strategy = step["strategy"]
                if strategy not in {"value", "mean", "median"}:
                    raise ValidationError("Fill strategy must be value, mean, or median.")
                for c in selected:
                    if strategy == "value":
                        if "value" not in step:
                            raise ValidationError("Constant fill requires a value.")
                        value = _scalar(step["value"])
                    else:
                        values = _numeric([r[c] for r in rows], c)
                        if not values:
                            raise ValidationError(f"Column '{c}' has no values from which to compute {strategy}.")
                        value = statistics.mean(values) if strategy == "mean" else statistics.median(values)
                    for row in rows:
                        if row[c] is None:
                            row[c] = value
            elif op == "filter":
                c = _column(step["column"], columns)
                operator, value = step["operator"], step["value"]
                if operator not in {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"}:
                    raise ValidationError("Unsupported filter operator.")
                if operator == "in":
                    if not isinstance(value, list) or len(value) > 1000:
                        raise ValidationError("The in filter requires a list of at most 1000 values.")
                    value = [_scalar(v) for v in value]
                else:
                    value = _scalar(value)
                def matches(row):
                    left = row[c]
                    if operator == "eq":
                        return left == value
                    if operator == "ne":
                        return left != value
                    if operator == "in":
                        # Recipe validation guarantees a list; keep the invariant
                        # explicit inside the closure for type checking as well.
                        return isinstance(value, list) and left in value
                    if left is None:
                        return False
                    if operator == "contains":
                        return str(value) in str(left)
                    if value is None:
                        raise ValidationError("Ordered comparisons cannot use a null value.")
                    if operator == "gt":
                        return left > value
                    if operator == "gte":
                        return left >= value
                    if operator == "lt":
                        return left < value
                    return left <= value
                rows = [r for r in rows if matches(r)]
            elif op == "sort":
                c = _column(step["column"], columns)
                descending = step.get("descending", False)
                if not isinstance(descending, bool):
                    raise ValidationError("descending must be true or false.")
                present = [r for r in rows if r[c] is not None]
                absent = [r for r in rows if r[c] is None]
                rows = sorted(present, key=lambda r: r[c], reverse=descending) + absent
            elif op == "select":
                columns = _columns(step["columns"], columns)
                rows = [{c: r[c] for c in columns} for r in rows]
            elif op == "rename":
                mapping = step["mapping"]
                if not isinstance(mapping, dict) or not mapping:
                    raise ValidationError("Rename requires a nonempty mapping.")
                for c in mapping:
                    _column(c, columns)
                new_columns = _names([mapping.get(c, c) for c in columns])
                rows = [{mapping.get(c, c): r[c] for c in columns} for r in rows]
                columns = new_columns
            elif op == "derive":
                c = _column(step["column"], columns)
                target, method = step["target"], step["method"]
                if not isinstance(target, str) or not target.strip() or target in columns:
                    raise ValidationError("Derived column needs a new, nonempty name.")
                if method not in {"normalize", "zscore", "log", "log10", "scale"}:
                    raise ValidationError("Unsupported numeric derivation.")
                values = _numeric([r[c] for r in rows], c)
                if not values:
                    raise ValidationError(f"Column '{c}' contains no numeric values.")
                if method == "normalize":
                    lo, hi = min(values), max(values)
                    def calculate(v):
                        return (v - lo) / (hi - lo) if hi != lo else 0.0
                elif method == "zscore":
                    mean, sigma = statistics.mean(values), statistics.pstdev(values)
                    def calculate(v):
                        return (v - mean) / sigma if sigma else 0.0
                elif method in {"log", "log10"}:
                    if any(v <= 0 for v in values):
                        raise ValidationError("Logarithms require strictly positive values.")
                    calculate = math.log if method == "log" else math.log10
                else:
                    factor = step.get("factor", 1.0)
                    if isinstance(factor, bool) or not isinstance(factor, (float, int)) or not math.isfinite(factor):
                        raise ValidationError("Scale factor must be a finite number.")
                    def calculate(v):
                        return v * factor
                for row in rows:
                    row[target] = None if row[c] is None else _scalar(calculate(row[c]))
                columns = _names([*columns, target])
            elif op == "aggregate":
                c = _column(step["column"], columns)
                grouping = _columns(step.get("groupBy", []), columns, allow_empty=True)
                function = step["function"]
                if function not in {"mean", "sum", "count", "min", "max", "median"}:
                    raise ValidationError("Unsupported aggregate function.")
                target = step.get("target", f"{c}_{function}")
                new_columns = _names([*grouping, target])
                groups: dict[str, tuple[tuple, list]] = {}
                for row in rows:
                    key = tuple(row[g] for g in grouping)
                    groups.setdefault(canonical_json(key), (key, []))[1].append(row[c])
                result = []
                for key, members in groups.values():
                    values = [v for v in members if v is not None]
                    if function == "count":
                        value = len(values)
                    elif not values:
                        value = None
                    else:
                        values = _numeric(values, c)
                        value = {"mean": statistics.mean, "sum": sum, "min": min,
                                 "max": max, "median": statistics.median}[function](values)
                    result.append({**dict(zip(grouping, key)), target: _scalar(value)})
                rows, columns = result, new_columns
        except ValidationError as exc:
            raise ValidationError(f"Step {number} ({op}): {exc}") from exc
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ValidationError(f"Step {number} ({op}) is incompatible with these column values.") from exc
    result = table_from_rows(rows, columns)
    result["recipe"] = steps
    result["inputHash"] = table["hash"]
    return result
