"""Pure, non-executing rewrites for Pylustrator-generated Python source."""

from __future__ import annotations

import ast
import io
import re
import tokenize
import textwrap
from dataclasses import dataclass


GENERATED_STATE_VERSION = 2
START_MARKER = "#% start: automatic generated code from pylustrator"
END_MARKER = "#% end: automatic generated code from pylustrator"
_MARKER_LIKE = re.compile(
    r"#\s*%\s*(start|end):\s*automatic generated code from pylustrator.*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceRewrite:
    """One concrete, position-preserving source rewrite."""

    kind: str
    start: tuple[int, int]
    end: tuple[int, int]
    replacement: str


class UnsafeSourceMigration(ValueError):
    """A legacy pattern was found but cannot be rewritten losslessly."""

    def __init__(self, message: str, position: tuple[int, int]):
        super().__init__(message)
        self.position = position


def classify_marker_comment(comment: str) -> str | None:
    """Classify an exact or suspiciously noncanonical marker comment."""

    if comment == START_MARKER:
        return "start"
    if comment == END_MARKER:
        return "end"
    match = _MARKER_LIKE.fullmatch(comment)
    if match:
        return f"near-{match.group(1).lower()}"
    return None


def _ast_column_to_character(line: str, byte_column: int) -> int:
    encoded = line.encode("utf-8")
    return len(encoded[:byte_column].decode("utf-8"))


def _original_position(
    source: str,
    dedented: str,
    line_number: int,
    byte_column: int,
) -> tuple[int, int]:
    original_line = source.splitlines()[line_number - 1]
    dedented_line = dedented.splitlines()[line_number - 1]
    removed = len(original_line) - len(original_line.lstrip(" \t"))
    dedented_indent = len(dedented_line) - len(dedented_line.lstrip(" \t"))
    removed -= dedented_indent
    return (
        line_number,
        removed + _ast_column_to_character(dedented_line, byte_column),
    )


def _integer_constant(node: ast.AST, expected: int | None = None) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return expected is None or value == expected


_SCOPE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def _scope_for(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    current = node
    while not isinstance(current, _SCOPE_TYPES):
        current = parents[current]
    return current


def _legend_proxy_rewrites(source: str) -> list[SourceRewrite]:
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    offsets = _line_offsets(source)
    rewrites: list[SourceRewrite] = []
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.Subscript) or not _integer_constant(outer.slice):
            continue
        inner = outer.value
        if not isinstance(inner, ast.Subscript) or not _integer_constant(inner.slice, 0):
            continue
        call = inner.value
        if not isinstance(call, ast.Call) or call.args or call.keywords:
            continue
        function = call.func
        if (
            not isinstance(function, ast.Attribute)
            or function.attr != "get_legend_handles_labels"
        ):
            continue
        receiver = parents.get(outer)
        if (
            not isinstance(receiver, ast.Attribute)
            or receiver.value is not outer
            or not (receiver.attr == "set" or receiver.attr.startswith("set_"))
        ):
            continue
        if (
            function.end_lineno is None
            or call.end_lineno is None
            or inner.end_lineno is None
        ):
            continue
        function_end = _original_position(
            source,
            dedented,
            function.end_lineno,
            function.end_col_offset,
        )
        method_token = next(
            (
                item
                for item in tokens
                if item.type == tokenize.NAME
                and item.string == "get_legend_handles_labels"
                and item.end == function_end
            ),
            None,
        )
        if method_token is None:
            raise UnsafeSourceMigration(
                "the legacy Legend method token cannot be located exactly",
                function_end,
            )
        call_end = _original_position(
            source,
            dedented,
            call.end_lineno,
            call.end_col_offset,
        )
        inner_end = _original_position(
            source,
            dedented,
            inner.end_lineno,
            inner.end_col_offset,
        )
        bracket_start = _absolute_offset(offsets, call_end)
        bracket_end = _absolute_offset(offsets, inner_end)
        if "#" in source[bracket_start:bracket_end]:
            raise UnsafeSourceMigration(
                "a comment inside the legacy Legend tuple index cannot be preserved",
                call_end,
            )
        rewrites.extend(
            (
                SourceRewrite(
                    "legacy-legend-proxy",
                    method_token.start,
                    method_token.end,
                    "get_legend",
                ),
                SourceRewrite(
                    "legacy-legend-proxy-index",
                    call_end,
                    inner_end,
                    ".legend_handles",
                ),
            )
        )
    return rewrites


def _nonfinite_load_rewrites(source: str) -> list[SourceRewrite]:
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    ambiguous: dict[str, set[ast.AST]] = {"nan": set(), "inf": set()}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in ambiguous
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            ambiguous[node.id].add(_scope_for(node, parents))
        elif isinstance(node, ast.arg) and node.arg in ambiguous:
            ambiguous[node.arg].add(_scope_for(node, parents))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in ambiguous:
                ambiguous[node.name].add(_scope_for(parents[node], parents))
        elif isinstance(node, ast.alias):
            bound = node.asname or node.name.split(".")[0]
            if bound in ambiguous:
                ambiguous[bound].add(_scope_for(node, parents))
        elif isinstance(node, ast.ExceptHandler) and node.name in ambiguous:
            ambiguous[node.name].add(_scope_for(node, parents))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                if name in ambiguous:
                    ambiguous[name].add(_scope_for(node, parents))
    rewrites: list[SourceRewrite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if (
            node.id not in {"nan", "inf"}
            or node.end_lineno is None
        ):
            continue
        if _scope_for(node, parents) in ambiguous[node.id]:
            raise UnsafeSourceMigration(
                f"'{node.id}' is bound ambiguously in the generated block",
                _original_position(source, dedented, node.lineno, node.col_offset),
            )
        start = _original_position(source, dedented, node.lineno, node.col_offset)
        end = _original_position(
            source,
            dedented,
            node.end_lineno,
            node.end_col_offset,
        )
        rewrites.append(
            SourceRewrite(
                "bare-nonfinite",
                start,
                end,
                f'__import__("numpy").{node.id}',
            )
        )
    return rewrites


def find_unbound_numpy_rewrites(source: str) -> tuple[SourceRewrite, ...]:
    """Replace unsafe ``np`` attribute loads without introducing an alias."""

    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    uses: list[ast.Name] = []
    for node in ast.walk(tree):
        parent = parents.get(node)
        if (
            isinstance(node, ast.Name)
            and node.id == "np"
            and isinstance(node.ctx, ast.Load)
            and isinstance(parent, ast.Attribute)
            and parent.value is node
            and node.end_lineno is not None
        ):
            uses.append(node)
    if not uses:
        return ()
    imports_by_scope: dict[ast.AST, list[ast.Import]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "numpy" and alias.asname == "np" for alias in node.names
        ) and parents.get(node) is _scope_for(node, parents):
            imports_by_scope.setdefault(_scope_for(node, parents), []).append(node)
    ambiguous_scopes: set[ast.AST] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == "np"
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            ambiguous_scopes.add(_scope_for(node, parents))
        elif isinstance(node, ast.arg) and node.arg == "np":
            ambiguous_scopes.add(_scope_for(node, parents))
        elif isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == "np":
            if not (node.name == "numpy" and node.asname == "np"):
                ambiguous_scopes.add(_scope_for(node, parents))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "np":
                ambiguous_scopes.add(_scope_for(parents[node], parents))
        elif isinstance(node, ast.ExceptHandler):
            if node.name == "np":
                ambiguous_scopes.add(_scope_for(node, parents))
    ambiguous_use = next(
        (node for node in uses if _scope_for(node, parents) in ambiguous_scopes), None
    )
    if ambiguous_use is not None:
        raise UnsafeSourceMigration(
            "'np' is bound to an ambiguous value in the generated block",
            _original_position(
                source,
                dedented,
                ambiguous_use.lineno,
                ambiguous_use.col_offset,
            ),
        )
    unsafe_uses = [
        node
        for node in uses
        if not any(
            (import_node.lineno, import_node.col_offset)
            < (node.lineno, node.col_offset)
            for import_node in imports_by_scope.get(_scope_for(node, parents), ())
        )
    ]
    if not unsafe_uses:
        return ()
    return tuple(
        SourceRewrite(
            "unbound-numpy-alias",
            _original_position(source, dedented, node.lineno, node.col_offset),
            _original_position(
                source,
                dedented,
                node.end_lineno,
                node.end_col_offset,
            ),
            '__import__("numpy")',
        )
        for node in unsafe_uses
    )


def find_source_rewrites(
    source: str,
    *,
    from_version: int = 0,
    qualify_nonfinite: bool = True,
) -> tuple[SourceRewrite, ...]:
    """Find AST-confirmed migrations without evaluating *source*."""

    rewrites: list[SourceRewrite] = []
    if from_version < 2:
        rewrites.extend(_legend_proxy_rewrites(source))
    if qualify_nonfinite:
        rewrites.extend(_nonfinite_load_rewrites(source))
    return tuple(rewrites)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    if not source or source.endswith(("\n", "\r")):
        return offsets
    if offsets[-1] != len(source):
        offsets.append(len(source))
    return offsets


def _absolute_offset(offsets: list[int], position: tuple[int, int]) -> int:
    row, column = position
    if row < 1 or row > len(offsets):
        raise ValueError(f"invalid source position: {position!r}")
    return offsets[row - 1] + column


def apply_source_rewrites(
    source: str, rewrites: tuple[SourceRewrite, ...] | list[SourceRewrite]
) -> str:
    """Apply non-overlapping rewrites while preserving all untouched bytes."""

    offsets = _line_offsets(source)
    spans = sorted(
        (
            _absolute_offset(offsets, rewrite.start),
            _absolute_offset(offsets, rewrite.end),
            rewrite.replacement,
        )
        for rewrite in rewrites
    )
    previous_end = -1
    for start, end, _replacement in spans:
        if start < previous_end:
            raise ValueError("overlapping source rewrites")
        previous_end = end
    migrated = source
    for start, end, replacement in reversed(spans):
        migrated = migrated[:start] + replacement + migrated[end:]
    return migrated


def _generated_block_spans(source: str) -> tuple[tuple[int, int], ...] | None:
    """Return complete block spans, or ``None`` for malformed marker structure."""

    spans: list[tuple[int, int]] = []
    active: int | None = None
    offsets = _line_offsets(source)
    malformed = False
    saw_marker = False
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comments = [item for item in tokens if item.type == tokenize.COMMENT]
    except (
        IndentationError,
        SyntaxError,
        tokenize.TokenError,
        UnsafeSourceMigration,
    ):
        return None
    for item in comments:
        kind = classify_marker_comment(item.string)
        if kind is not None and kind.startswith("near-"):
            malformed = True
            kind = None
        if kind is not None and item.line[: item.start[1]].strip():
            malformed = True
            kind = None
        if kind is not None:
            saw_marker = True
        line_start = offsets[item.start[0] - 1]
        line_end = offsets[item.start[0]]
        if kind == "start":
            if active is not None:
                malformed = True
            else:
                active = line_start
        elif kind == "end":
            if active is None:
                malformed = True
            else:
                spans.append((active, line_end))
                active = None
    if active is not None:
        malformed = True
    if malformed:
        return None
    if not saw_marker:
        return ()
    return tuple(spans)


def migrate_generated_command(command: str, from_version: int = 0) -> str:
    """Upgrade one generated command without executing or reformatting it."""

    try:
        rewrites = find_source_rewrites(
            command,
            from_version=from_version,
            qualify_nonfinite=False,
        )
    except (
        IndentationError,
        SyntaxError,
        tokenize.TokenError,
        UnsafeSourceMigration,
    ):
        return command
    migrated = apply_source_rewrites(command, rewrites)
    try:
        ast.parse(textwrap.dedent(migrated))
    except (IndentationError, SyntaxError):
        return command
    return migrated


def _block_version(block: str) -> tuple[bool, int]:
    tree = ast.parse(textwrap.dedent(block))
    versions: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Attribute)
            and target.attr == "_pylustrator_generated_version"
            for target in targets
        ):
            continue
        value = node.value
        if (
            not isinstance(value, ast.Constant)
            or not isinstance(value.value, int)
            or isinstance(value.value, bool)
        ):
            return False, 0
        versions.append(value.value)
    if len(versions) > 1:
        return False, 0
    return True, versions[0] if versions else 0


def migrate_generated_source(
    source: str, from_version: int | None = None
) -> str:
    """Upgrade generated blocks while leaving surrounding user source untouched.

    Marker-free input and malformed block markers fail closed. Use
    :func:`migrate_generated_command` for an already-isolated command string.
    """

    spans = _generated_block_spans(source)
    if spans is None:
        return source
    if not spans:
        return source
    try:
        ast.parse(source)
    except (IndentationError, SyntaxError):
        return source
    migrated = source
    for start, end in reversed(spans):
        block = migrated[start:end]
        valid_version, detected_version = _block_version(block)
        if not valid_version or detected_version > GENERATED_STATE_VERSION:
            return source
        block_version = detected_version if from_version is None else from_version
        migrated_block = migrate_generated_command(block, block_version)
        migrated = migrated[:start] + migrated_block + migrated[end:]
    try:
        ast.parse(migrated)
    except (IndentationError, SyntaxError):
        return source
    return migrated


__all__ = [
    "END_MARKER",
    "GENERATED_STATE_VERSION",
    "START_MARKER",
    "SourceRewrite",
    "UnsafeSourceMigration",
    "apply_source_rewrites",
    "classify_marker_comment",
    "find_source_rewrites",
    "find_unbound_numpy_rewrites",
    "migrate_generated_command",
    "migrate_generated_source",
]
