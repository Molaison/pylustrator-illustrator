"""Offline diagnostics and atomic migration for generated source blocks."""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import json
import os
import stat
import sys
import tempfile
import token
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .source_migration import (
    END_MARKER,
    GENERATED_STATE_VERSION,
    START_MARKER,
    SourceRewrite,
    UnsafeSourceMigration,
    apply_source_rewrites,
    classify_marker_comment,
    find_source_rewrites,
    find_unbound_numpy_rewrites,
)


REPORT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SourceDiagnostic:
    """One issue found without importing or executing the inspected script."""

    code: str
    message: str
    line: int
    column: int
    severity: str
    fixable: bool
    block: int | None = None


@dataclass(frozen=True)
class SourceDoctorReport:
    """Diagnostics and a candidate migrated source string."""

    filename: str
    original_source: str
    migrated_source: str
    block_count: int
    diagnostics: tuple[SourceDiagnostic, ...]

    @property
    def changed(self) -> bool:
        return self.original_source != self.migrated_source

    @property
    def has_errors(self) -> bool:
        return any(item.severity == "error" for item in self.diagnostics)

    @property
    def needs_attention(self) -> bool:
        return bool(self.diagnostics)

    def to_data(self) -> dict:
        """Return a JSON-serializable report without duplicating source text."""

        return {
            "filename": self.filename,
            "blocks": self.block_count,
            "changed": self.changed,
            "has_errors": self.has_errors,
            "diagnostics": [asdict(item) for item in self.diagnostics],
        }


@dataclass(frozen=True)
class _GeneratedBlock:
    start: int
    end: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _VersionAssignment:
    value: int | None
    value_start: tuple[int, int]
    value_end: tuple[int, int]
    attribute_start: tuple[int, int]


_IGNORED_TOKEN_TYPES = {
    tokenize.COMMENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}
_SKIPPED_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def _find_blocks(source: str) -> tuple[list[_GeneratedBlock], list[SourceDiagnostic]]:
    blocks: list[_GeneratedBlock] = []
    diagnostics: list[SourceDiagnostic] = []
    active_offset: int | None = None
    active_line: int | None = None
    active_column: int | None = None
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comments = [item for item in tokens if item.type == tokenize.COMMENT]
    except (IndentationError, SyntaxError, tokenize.TokenError) as error:
        if START_MARKER not in source and END_MARKER not in source:
            return blocks, diagnostics
        relative_line = getattr(error, "lineno", None) or 1
        if isinstance(error, tokenize.TokenError) and len(error.args) > 1:
            position = error.args[1]
            if isinstance(position, tuple) and position:
                relative_line = int(position[0])
        diagnostics.append(
            SourceDiagnostic(
                "PYL004",
                f"generated-block markers cannot be located safely: {error}",
                relative_line,
                1,
                "error",
                False,
            )
        )
        return blocks, diagnostics
    for item in comments:
        kind = classify_marker_comment(item.string)
        if kind is not None and kind.startswith("near-"):
            diagnostics.append(
                SourceDiagnostic(
                    "PYL004",
                    "generated-block marker must match exactly",
                    item.start[0],
                    item.start[1] + 1,
                    "error",
                    False,
                )
            )
            continue
        if kind is not None and item.line[: item.start[1]].strip():
            diagnostics.append(
                SourceDiagnostic(
                    "PYL004",
                    "generated-block marker must be the only statement on its line",
                    item.start[0],
                    item.start[1] + 1,
                    "error",
                    False,
                )
            )
            continue
        line_number = item.start[0]
        column = item.start[1] + 1
        if kind == "start":
            if active_offset is not None:
                diagnostics.append(
                    SourceDiagnostic(
                        "PYL002",
                        "nested generated-block start marker",
                        line_number,
                        column,
                        "error",
                        False,
                    )
                )
            else:
                active_offset = offsets[line_number - 1]
                active_line = line_number
                active_column = item.start[1]
        elif kind == "end":
            if active_offset is None or active_line is None:
                diagnostics.append(
                    SourceDiagnostic(
                        "PYL001",
                        "generated-block end marker has no matching start",
                        line_number,
                        column,
                        "error",
                        False,
                    )
                )
            else:
                if active_column != item.start[1]:
                    diagnostics.append(
                        SourceDiagnostic(
                            "PYL007",
                            "generated-block start and end markers must use the same indentation",
                            line_number,
                            column,
                            "error",
                            False,
                        )
                    )
                blocks.append(
                    _GeneratedBlock(
                        active_offset,
                        offsets[line_number],
                        active_line,
                        line_number,
                    )
                )
                active_offset = None
                active_line = None
                active_column = None
    if active_offset is not None and active_line is not None:
        diagnostics.append(
            SourceDiagnostic(
                "PYL003",
                "generated block is not closed",
                active_line,
                1,
                "error",
                False,
            )
        )
    return blocks, diagnostics


def _significant_tokens(source: str) -> list[tokenize.TokenInfo]:
    items = list(tokenize.generate_tokens(io.StringIO(source).readline))
    significant: list[tokenize.TokenInfo] = []
    fstring_depth = 0
    fstring_start = getattr(token, "FSTRING_START", -1)
    fstring_end = getattr(token, "FSTRING_END", -1)
    for item in items:
        if item.type == fstring_start:
            fstring_depth += 1
            continue
        if item.type == fstring_end:
            fstring_depth = max(0, fstring_depth - 1)
            continue
        if fstring_depth or item.type in _IGNORED_TOKEN_TYPES:
            continue
        significant.append(item)
    return significant


def _literal_integer(value: str) -> int | None:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, int) and not isinstance(parsed, bool):
        return parsed
    return None


def _version_assignments(
    significant: list[tokenize.TokenInfo],
) -> list[_VersionAssignment]:
    assignments: list[_VersionAssignment] = []
    for index, item in enumerate(significant):
        if item.type != tokenize.NAME or item.string != "_pylustrator_generated_version":
            continue
        if index == 0 or significant[index - 1].string != ".":
            continue
        if index + 2 >= len(significant) or significant[index + 1].string != "=":
            continue
        value_token = significant[index + 2]
        value = (
            _literal_integer(value_token.string)
            if value_token.type == tokenize.NUMBER
            else None
        )
        assignments.append(
            _VersionAssignment(
                value,
                value_token.start,
                value_token.end,
                item.start,
            )
        )
    return assignments


def _line_figure_references(line: str) -> list[str]:
    stripped = line.lstrip(" \t")
    try:
        tree = ast.parse(stripped)
    except (IndentationError, SyntaxError):
        return []
    references: list[str] = []
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in {
                    "ax_dict",
                    "_pylustrator_generated_version",
                }:
                    references.append(ast.unparse(target.value))
        for call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "getattr"
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant)
                and call.args[1].value == "_pylustrator_init"
            ):
                references.append(ast.unparse(call.args[0]))
    return references


def _assigns_attribute(line: str, attribute: str) -> bool:
    stripped = line.lstrip(" \t")
    try:
        tree = ast.parse(stripped)
    except (IndentationError, SyntaxError):
        return False
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(
            isinstance(target, ast.Attribute) and target.attr == attribute
            for target in targets
        ):
            return True
    return False


def _figure_references(block: str) -> list[tuple[str, int]]:
    references: list[tuple[str, int]] = []
    for line_number, line in enumerate(block.splitlines(), start=1):
        references.extend(
            (reference, line_number)
            for reference in _line_figure_references(line)
        )
    return references


def _binds_name(source: str, name: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            return True
        if isinstance(node, ast.arg) and node.arg == name:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        if isinstance(node, ast.alias) and (
            node.asname or node.name.split(".")[0]
        ) == name:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
    return False


def _newline_for(source: str) -> str:
    for line in source.splitlines(keepends=True):
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
        if line.endswith("\r"):
            return "\r"
    return "\n"


def _block_indent(block: str) -> str:
    first = block.splitlines()[0] if block.splitlines() else ""
    return first[: len(first) - len(first.lstrip(" \t"))]


def _insert_version_line(block: str, statement: str) -> str:
    lines = block.splitlines(keepends=True)
    if not lines:
        return block
    newline = _newline_for(block)
    line = f"{_block_indent(block)}{statement}{newline}"
    end_index = max(1, len(lines) - 1)
    anchor = 0
    for index in range(1, end_index):
        stripped = lines[index].strip()
        if (
            "_pylustrator_init" in stripped
            or stripped.startswith("import ")
            or _assigns_attribute(lines[index], "ax_dict")
        ):
            anchor = index
    lines.insert(anchor + 1, line)
    return "".join(lines)


def _diagnostic_for_rewrite(
    rewrite: SourceRewrite, block: _GeneratedBlock, block_index: int
) -> SourceDiagnostic:
    line, column = rewrite.start
    if rewrite.kind == "legacy-legend-proxy":
        return SourceDiagnostic(
            "PYL201",
            "legacy Legend proxy locator must use the persisted Legend handles",
            block.start_line + line - 1,
            column + 1,
            "warning",
            True,
            block_index,
        )
    if rewrite.kind == "unbound-numpy-alias":
        return SourceDiagnostic(
            "PYL203",
            "numpy alias is not imported before use; use an alias-free NumPy lookup",
            block.start_line + line - 1,
            column + 1,
            "warning",
            True,
            block_index,
        )
    return SourceDiagnostic(
        "PYL202",
        "bare non-finite literal can fail before runtime migration; use an alias-free NumPy value",
        block.start_line + line - 1,
        column + 1,
        "warning",
        True,
        block_index,
    )


def _analyse_block(
    block_source: str,
    block: _GeneratedBlock,
    block_index: int,
) -> tuple[list[SourceDiagnostic], str]:
    diagnostics: list[SourceDiagnostic] = []
    try:
        significant = _significant_tokens(block_source)
    except (IndentationError, SyntaxError, tokenize.TokenError) as error:
        relative_line = getattr(error, "lineno", None) or 1
        if isinstance(error, tokenize.TokenError) and len(error.args) > 1:
            position = error.args[1]
            if isinstance(position, tuple) and position:
                relative_line = int(position[0])
        diagnostics.append(
            SourceDiagnostic(
                "PYL301",
                f"generated block cannot be tokenized safely: {error}",
                block.start_line + relative_line - 1,
                1,
                "error",
                False,
                block_index,
            )
        )
        return diagnostics, block_source

    versions = _version_assignments(significant)
    references = _figure_references(block_source)
    unique_references = list(dict.fromkeys(reference for reference, _line in references))
    reference = unique_references[0] if len(unique_references) == 1 else None
    if len(unique_references) > 1:
        first_reference = unique_references[0]
        conflicting_reference, conflicting_line = next(
            item for item in references if item[0] != first_reference
        )
        diagnostics.append(
            SourceDiagnostic(
                "PYL107",
                "generated block mixes Figure references "
                f"{first_reference!r} and {conflicting_reference!r}",
                block.start_line + conflicting_line - 1,
                1,
                "error",
                False,
                block_index,
            )
        )
    add_version = False
    effective_version = 0
    rewrites: list[SourceRewrite] = []
    if len(versions) > 1:
        marker = versions[1]
        diagnostics.append(
            SourceDiagnostic(
                "PYL104",
                "generated block has multiple schema-version assignments",
                block.start_line + marker.attribute_start[0] - 1,
                marker.attribute_start[1] + 1,
                "error",
                False,
                block_index,
            )
        )
    elif versions and versions[0].value is None:
        marker = versions[0]
        diagnostics.append(
            SourceDiagnostic(
                "PYL105",
                "generated schema version must be an integer literal",
                block.start_line + marker.value_start[0] - 1,
                marker.value_start[1] + 1,
                "error",
                False,
                block_index,
            )
        )
    elif versions and versions[0].value is not None:
        marker = versions[0]
        version = marker.value
        assert version is not None
        effective_version = version
        if version > GENERATED_STATE_VERSION:
            diagnostics.append(
                SourceDiagnostic(
                    "PYL103",
                    f"generated schema {version} is newer than supported schema {GENERATED_STATE_VERSION}",
                    block.start_line + marker.value_start[0] - 1,
                    marker.value_start[1] + 1,
                    "error",
                    False,
                    block_index,
                )
            )
        elif version < GENERATED_STATE_VERSION:
            diagnostics.append(
                SourceDiagnostic(
                    "PYL102",
                    f"generated schema {version} should migrate to {GENERATED_STATE_VERSION}",
                    block.start_line + marker.value_start[0] - 1,
                    marker.value_start[1] + 1,
                    "warning",
                    True,
                    block_index,
                )
            )
            rewrites.append(
                SourceRewrite(
                    "schema-version",
                    marker.value_start,
                    marker.value_end,
                    str(GENERATED_STATE_VERSION),
                )
            )
    else:
        fixable = reference is not None
        diagnostics.append(
            SourceDiagnostic(
                "PYL101" if fixable else "PYL106",
                (
                    f"generated block has no schema version; add schema {GENERATED_STATE_VERSION}"
                    if fixable
                    else "generated block has no schema version and its Figure reference cannot be inferred"
                ),
                block.start_line,
                1,
                "warning" if fixable else "error",
                fixable,
                block_index,
            )
        )
        add_version = fixable

    if any(item.severity == "error" for item in diagnostics):
        return diagnostics, block_source

    try:
        rewrites.extend(
            find_source_rewrites(
                block_source,
                from_version=effective_version,
            )
        )
        rewrites.extend(find_unbound_numpy_rewrites(block_source))
    except UnsafeSourceMigration as error:
        diagnostics.append(
            SourceDiagnostic(
                "PYL302",
                f"legacy source cannot be migrated losslessly: {error}",
                block.start_line + error.position[0] - 1,
                error.position[1] + 1,
                "error",
                False,
                block_index,
            )
        )
        return diagnostics, block_source
    except (IndentationError, SyntaxError, tokenize.TokenError) as error:
        relative_line = getattr(error, "lineno", None) or 1
        diagnostics.append(
            SourceDiagnostic(
                "PYL301",
                f"generated block cannot be parsed safely: {error}",
                block.start_line + relative_line - 1,
                1,
                "error",
                False,
                block_index,
            )
        )
        return diagnostics, block_source

    seen_rewrite_kinds: set[str] = set()
    for rewrite in rewrites:
        if rewrite.kind in {"legacy-legend-proxy-index", "schema-version"}:
            continue
        if rewrite.kind == "unbound-numpy-alias" and rewrite.kind in seen_rewrite_kinds:
            continue
        diagnostics.append(_diagnostic_for_rewrite(rewrite, block, block_index))
        seen_rewrite_kinds.add(rewrite.kind)

    migrated = apply_source_rewrites(block_source, rewrites)
    if add_version and reference is not None:
        migrated = _insert_version_line(
            migrated,
            f"{reference}._pylustrator_generated_version = {GENERATED_STATE_VERSION}",
        )
    return diagnostics, migrated


def diagnose_generated_source(
    source: str, *, filename: str = "<memory>"
) -> SourceDoctorReport:
    """Inspect and plan safe migrations without executing *source*."""

    blocks, diagnostics = _find_blocks(source)
    replacements: list[tuple[int, int, str]] = []
    if blocks and not diagnostics:
        try:
            ast.parse(source, filename=filename)
        except (IndentationError, SyntaxError) as error:
            diagnostics.append(
                SourceDiagnostic(
                    "PYL005",
                    f"source is not valid Python before migration: {error.msg}",
                    error.lineno or 1,
                    error.offset or 1,
                    "error",
                    False,
                )
            )
    if not diagnostics:
        for index, block in enumerate(blocks, start=1):
            block_source = source[block.start : block.end]
            block_diagnostics, migrated = _analyse_block(block_source, block, index)
            diagnostics.extend(block_diagnostics)
            replacements.append((block.start, block.end, migrated))
        alias_free = next(
            (item for item in diagnostics if item.code in {"PYL202", "PYL203"}), None
        )
        if alias_free is not None and _binds_name(source, "__import__"):
            diagnostics.append(
                SourceDiagnostic(
                    "PYL304",
                    "source binds '__import__'; alias-free NumPy migration is unsafe",
                    alias_free.line,
                    alias_free.column,
                    "error",
                    False,
                    alias_free.block,
                )
            )
    migrated_source = source
    if not any(item.severity == "error" for item in diagnostics):
        for start, end, migrated in reversed(replacements):
            migrated_source = migrated_source[:start] + migrated + migrated_source[end:]
        if migrated_source != source:
            try:
                ast.parse(migrated_source, filename=filename)
            except (IndentationError, SyntaxError) as error:
                diagnostics.append(
                    SourceDiagnostic(
                        "PYL006",
                        f"candidate migration is not valid Python: {error.msg}",
                        error.lineno or 1,
                        error.offset or 1,
                        "error",
                        False,
                    )
                )
                migrated_source = source
    diagnostics.sort(key=lambda item: (item.line, item.column, item.code))
    return SourceDoctorReport(
        filename,
        source,
        migrated_source,
        len(blocks),
        tuple(diagnostics),
    )


def _iter_source_paths(inputs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in inputs or ["."]:
        candidate = Path(value)
        if candidate.is_dir():
            discovered = (
                path
                for path in candidate.rglob("*.py")
                if not any(part in _SKIPPED_DIRECTORIES for part in path.parts)
            )
        else:
            discovered = (candidate,)
        for path in discovered:
            key = path.absolute()
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return sorted(paths, key=lambda path: str(path))


def _read_source(path: Path) -> tuple[bytes, str, str]:
    raw = path.read_bytes()
    encoding, _lines = tokenize.detect_encoding(io.BytesIO(raw).readline)
    return raw, raw.decode(encoding), encoding


def _atomic_write(path: Path, original: bytes, replacement: bytes) -> None:
    if path.is_symlink():
        raise OSError("refusing to replace a symbolic link")
    metadata = path.stat()
    if metadata.st_nlink > 1:
        raise OSError("refusing to break a multiply linked source file")
    mode = stat.S_IMODE(metadata.st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if path.read_bytes() != original:
            raise OSError("source changed while migration was being prepared")
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _unified_diff(report: SourceDoctorReport) -> str:
    return "".join(
        difflib.unified_diff(
            report.original_source.splitlines(keepends=True),
            report.migrated_source.splitlines(keepends=True),
            fromfile=report.filename,
            tofile=report.filename,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pylustrator-source",
        description=(
            "Inspect Pylustrator-generated blocks without executing the source; "
            "use --write to apply only safe, atomic migrations."
        ),
    )
    parser.add_argument("paths", nargs="*", help="Python files or directories (default: .)")
    parser.add_argument("--write", action="store_true", help="atomically apply safe migrations")
    parser.add_argument("--diff", action="store_true", help="print a unified migration diff")
    parser.add_argument("--json", action="store_true", help="emit machine-readable reports")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s schema {GENERATED_STATE_VERSION}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline doctor. Return 0 clean, 1 diagnosed, or 2 operational error."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.json and arguments.diff:
        parser.error("--json and --diff cannot be combined")
    paths = _iter_source_paths(arguments.paths)
    reports: list[dict] = []
    block_count = 0
    issue_count = 0
    changed_count = 0
    written_count = 0
    operational_errors = 0
    unresolved = False
    for path in paths:
        try:
            original, source, encoding = _read_source(path)
        except (OSError, SyntaxError, UnicodeError) as error:
            operational_errors += 1
            if arguments.json:
                reports.append({"filename": str(path), "error": str(error)})
            else:
                print(f"{path}: error: {error}", file=sys.stderr)
            continue
        report = diagnose_generated_source(source, filename=str(path))
        block_count += report.block_count
        issue_count += len(report.diagnostics)
        changed_count += int(report.changed)
        written = False
        write_error: str | None = None
        if arguments.diff and report.changed:
            print(_unified_diff(report), end="")
        if arguments.write:
            unresolved = unresolved or report.has_errors
            if report.changed and not report.has_errors:
                post_report = diagnose_generated_source(
                    report.migrated_source, filename=str(path)
                )
                if post_report.needs_attention:
                    operational_errors += 1
                    write_error = "candidate migration did not produce a clean source block"
                    if not arguments.json:
                        print(f"{path}: error: {write_error}", file=sys.stderr)
                else:
                    try:
                        _atomic_write(
                            path,
                            original,
                            report.migrated_source.encode(encoding),
                        )
                    except (OSError, UnicodeError) as error:
                        operational_errors += 1
                        write_error = str(error)
                        if not arguments.json:
                            print(f"{path}: error: {error}", file=sys.stderr)
                    else:
                        written = True
                        written_count += 1
                        if not arguments.json:
                            print(f"{path}: migrated")
            elif report.needs_attention and not report.has_errors:
                unresolved = True
        if arguments.json:
            data = report.to_data()
            data.update({"written": written, "write_error": write_error})
            reports.append(data)
        else:
            for diagnostic in report.diagnostics:
                suffix = " [fixable]" if diagnostic.fixable else ""
                print(
                    f"{path}:{diagnostic.line}:{diagnostic.column}: "
                    f"{diagnostic.code} {diagnostic.severity}: "
                    f"{diagnostic.message}{suffix}"
                )
    if arguments.json:
        print(
            json.dumps(
                {
                    "format_version": REPORT_FORMAT_VERSION,
                    "generated_schema": GENERATED_STATE_VERSION,
                    "files": reports,
                    "summary": {
                        "files": len(paths),
                        "blocks": block_count,
                        "issues": issue_count,
                        "changed": changed_count,
                        "written": written_count,
                        "operational_errors": operational_errors,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            f"Scanned {len(paths)} file(s), {block_count} generated block(s); "
            f"{issue_count} issue(s), {changed_count} file(s) need migration, "
            f"{written_count} migrated."
        )
    if operational_errors:
        return 2
    if arguments.write:
        return 1 if unresolved else 0
    return 1 if issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SourceDiagnostic",
    "SourceDoctorReport",
    "REPORT_FORMAT_VERSION",
    "diagnose_generated_source",
    "main",
]
