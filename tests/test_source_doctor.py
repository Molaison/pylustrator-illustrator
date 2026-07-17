from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from pylustrator.commands import migrate_generated_command, migrate_generated_source
from pylustrator.source_doctor import diagnose_generated_source, main


START = "#% start: automatic generated code from pylustrator"
END = "#% end: automatic generated code from pylustrator"


def generated_block(*lines: str, newline: str = "\n", indent: str = "") -> str:
    return newline.join(
        [f"{indent}{START}", *(f"{indent}{line}" for line in lines), f"{indent}{END}"]
    ) + newline


def test_doctor_migrates_legacy_block_without_executing_or_touching_user_source() -> None:
    source = (
        'outside = ".get_legend_handles_labels()[0] and nan"\n'
        "raise RuntimeError('must never execute')\n"
        + generated_block(
            "plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}",
            "import matplotlib as mpl",
            "getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()",
            "plt.figure(1).axes[0].get_legend_handles_labels()[0][1].set_alpha(0.5)",
            "plt.figure(1).axes[0].set_xlim([nan, -inf])",
            'message = "nan and .get_legend_handles_labels()[0]"',
            "# inf and .get_legend_handles_labels()[0]",
        )
        + "after = nan\n"
    )

    report = diagnose_generated_source(source, filename="figure.py")

    assert report.block_count == 1
    assert not report.has_errors
    assert report.changed
    assert {item.code for item in report.diagnostics} == {
        "PYL101",
        "PYL201",
        "PYL202",
    }
    assert report.original_source.startswith(
        'outside = ".get_legend_handles_labels()[0] and nan"'
    )
    migrated = report.migrated_source
    assert 'outside = ".get_legend_handles_labels()[0] and nan"' in migrated
    assert "after = nan" in migrated
    assert ".get_legend().legend_handles[1].set_alpha(0.5)" in migrated
    assert '.set_xlim([__import__("numpy").nan, -__import__("numpy").inf])' in migrated
    assert "._pylustrator_generated_version = 2" in migrated
    assert 'message = "nan and .get_legend_handles_labels()[0]"' in migrated
    assert "# inf and .get_legend_handles_labels()[0]" in migrated

    second = diagnose_generated_source(migrated, filename="figure.py")
    assert second.diagnostics == ()
    assert not second.changed


def test_current_generated_block_is_clean() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}",
        "import matplotlib as mpl",
        "import numpy as np",
        "getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()",
        "plt.figure(1)._pylustrator_generated_version = 2",
        "plt.figure(1).axes[0].set_xlim([np.nan, np.inf])",
    )

    report = diagnose_generated_source(source)

    assert report.diagnostics == ()
    assert report.migrated_source == source


def test_public_source_migration_limits_rewrites_to_generated_blocks() -> None:
    outside = "value = nan  # .get_legend_handles_labels()[0]\n"
    block = generated_block(
        "plt.figure(1).axes[0].get_legend_handles_labels()[0][0].set_alpha(nan)",
        'text = "nan .get_legend_handles_labels()[0]"',
    )
    migrated = migrate_generated_source(outside + block + outside)

    assert migrated.startswith(outside)
    assert migrated.endswith(outside)
    assert ".get_legend().legend_handles[0].set_alpha(nan)" in migrated
    assert 'text = "nan .get_legend_handles_labels()[0]"' in migrated


def test_command_migration_is_token_safe_and_version_aware() -> None:
    command = (
        "target.set(value=nan, limit=-inf, exact=np.nan, other=math.inf, inf=1); "
        'text = "nan"  # inf\n'
        "target.get_legend_handles_labels()[0][0].set_alpha(0.5)"
    )

    migrated = migrate_generated_command(command)
    current = migrate_generated_command(command, from_version=2)

    assert "value=nan" in migrated
    assert "limit=-inf" in migrated
    assert "exact=np.nan" in migrated
    assert "other=math.inf" in migrated
    assert "inf=1" in migrated
    assert 'text = "nan"  # inf' in migrated
    assert ".get_legend().legend_handles[0]" in migrated
    assert ".get_legend_handles_labels()[0]" in current
    assert "value=nan" in current
    assert migrate_generated_source(command) == command


def test_doctor_fails_closed_for_bound_nonfinite_names() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "import numpy as np",
        "plt.figure(1)._pylustrator_generated_version = 2",
        "nan, value = (1, 2)",
        "def helper(inf):",
        "    return inf",
        "result = nan",
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL302" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_legend_handle_list_used_to_build_a_legend_is_not_a_proxy_locator() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 2",
        "plt.figure(1).axes[0].legend(handles=plt.figure(1).axes[0].get_legend_handles_labels()[0])",
    )

    report = diagnose_generated_source(source)

    assert report.diagnostics == ()
    assert report.migrated_source == source


def test_indexed_axes_handle_used_to_build_legend_is_not_a_proxy_locator() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "plt.figure(1).axes[0].legend(handles=[plt.figure(1).axes[0].get_legend_handles_labels()[0][0]])",
    )

    report = diagnose_generated_source(source)

    assert {item.code for item in report.diagnostics} == {"PYL102"}
    assert "get_legend_handles_labels" in report.migrated_source


def test_legend_migration_preserves_comment_inside_call() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "plt.figure(1).axes[0].get_legend_handles_labels(  # preserve me",
        ")[0][1].set_alpha(0.5)",
    )

    report = diagnose_generated_source(source)

    assert not report.has_errors
    assert "# preserve me" in report.migrated_source
    assert "get_legend(" in report.migrated_source
    assert ").legend_handles[1].set_alpha" in report.migrated_source
    assert diagnose_generated_source(report.migrated_source).diagnostics == ()


def test_comment_inside_legacy_tuple_index_fails_closed() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "plt.figure(1).axes[0].get_legend_handles_labels()[  # cannot move",
        "0][1].set_alpha(0.5)",
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL302" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_logical_newlines_cannot_form_a_cross_statement_legend_locator() -> None:
    command = (
        "target.get_legend_handles_labels()\n"
        "[0]\n"
        "[1].set_alpha(0.5)\n"
    )
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        *command.splitlines(),
    )

    report = diagnose_generated_source(source)

    assert "PYL201" not in {item.code for item in report.diagnostics}
    assert "get_legend_handles_labels" in report.migrated_source
    assert migrate_generated_command(command) == command


def test_schema_two_indexed_handle_chain_is_not_reinterpreted_as_legacy() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 2",
        "plt.figure(1).axes[0].get_legend_handles_labels()[0][0].set_alpha(0.5)",
    )

    report = diagnose_generated_source(source)

    assert report.diagnostics == ()
    assert report.migrated_source == source
    assert migrate_generated_source(source) == source


def test_each_block_uses_its_own_schema_version() -> None:
    legacy = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "plt.figure(1).axes[0].get_legend_handles_labels()[0][0].set_alpha(0.5)",
    )
    current = generated_block(
        "plt.figure(2).ax_dict = {}",
        "plt.figure(2)._pylustrator_generated_version = 2",
        "plt.figure(2).axes[0].get_legend_handles_labels()[0][0].set_alpha(0.5)",
    )

    report = diagnose_generated_source(legacy + current)

    assert {item.code for item in report.diagnostics} == {"PYL102", "PYL201"}
    assert report.migrated_source.count("get_legend().legend_handles") == 1
    assert report.migrated_source.count("get_legend_handles_labels") == 1
    assert diagnose_generated_source(report.migrated_source).diagnostics == ()


def test_inconsistent_header_figure_references_fail_closed() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()",
        "plt.figure(2)._pylustrator_generated_version = 1",
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL107" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_init_only_legacy_block_can_infer_figure_reference() -> None:
    source = generated_block(
        "getattr(plt.figure(3), '_pylustrator_init', lambda: ...)()",
        "plt.figure(3).set_size_inches(4, 3)",
    )

    report = diagnose_generated_source(source)

    assert {item.code for item in report.diagnostics} == {"PYL101"}
    assert "plt.figure(3)._pylustrator_generated_version = 2" in report.migrated_source
    assert diagnose_generated_source(report.migrated_source).diagnostics == ()


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (f"{END}\n", "PYL001"),
        (f"{START}\n{START}\n{END}\n{END}\n", "PYL002"),
        (f"{START}\nvalue = nan\n", "PYL003"),
    ],
)
def test_malformed_marker_structure_fails_closed(source: str, code: str) -> None:
    report = diagnose_generated_source(source)

    assert report.has_errors
    assert code in {item.code for item in report.diagnostics}
    assert report.migrated_source == source
    assert migrate_generated_source(source) == source


def test_future_schema_fails_closed_even_when_repairs_are_visible() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 99",
        "plt.figure(1).axes[0].set_xlim([nan, inf])",
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL103" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_duplicate_or_noninteger_versions_fail_closed() -> None:
    duplicate = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "plt.figure(1)._pylustrator_generated_version = 2",
    )
    malformed = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = current_version",
    )

    duplicate_report = diagnose_generated_source(duplicate)
    malformed_report = diagnose_generated_source(malformed)

    assert duplicate_report.has_errors
    assert "PYL104" in {item.code for item in duplicate_report.diagnostics}
    assert duplicate_report.migrated_source == duplicate
    assert malformed_report.has_errors
    assert "PYL105" in {item.code for item in malformed_report.diagnostics}
    assert malformed_report.migrated_source == malformed


def test_missing_version_without_inferable_figure_is_not_guessed() -> None:
    source = generated_block("custom_target.set_alpha(0.5)")

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert report.diagnostics[0].code == "PYL106"
    assert not report.diagnostics[0].fixable
    assert report.migrated_source == source


def test_old_version_and_unbound_numpy_alias_are_repaired() -> None:
    source = generated_block(
        "plt.figure(7).ax_dict = {}",
        "plt.figure(7)._pylustrator_generated_version = 1",
        "plt.figure(7).axes[0].set_xlim([np.nan, 1])",
    )

    report = diagnose_generated_source(source)

    assert {item.code for item in report.diagnostics} == {"PYL102", "PYL203"}
    assert "._pylustrator_generated_version = 2" in report.migrated_source
    assert '__import__("numpy").nan' in report.migrated_source
    assert diagnose_generated_source(report.migrated_source).diagnostics == ()


def test_indented_nonfinite_migration_does_not_create_local_np_scope() -> None:
    source = (
        "import numpy as np\n"
        "def configure():\n"
        "    before = np.arange(3)\n"
        + generated_block(
            "plt.figure(1).ax_dict = {}",
            "plt.figure(1).axes[0].set_xlim([nan, 1])",
            indent="    ",
        )
        + "    return before\n"
    )

    report = diagnose_generated_source(source)

    assert not report.has_errors
    assert '__import__("numpy").nan' in report.migrated_source
    assert report.migrated_source.count("import numpy as np") == 1
    assert diagnose_generated_source(report.migrated_source).diagnostics == ()


def test_indented_crlf_blocks_preserve_newlines_and_migrate_independently() -> None:
    first = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1).axes[0].set_xlim([nan, 1])",
        newline="\r\n",
        indent="    ",
    )
    second = generated_block(
        "plt.figure(2).ax_dict = {}",
        "plt.figure(2)._pylustrator_generated_version = 1",
        newline="\r\n",
        indent="    ",
    )
    source = "def configure():\r\n" + first + second + "    return None\r\n"

    report = diagnose_generated_source(source)

    assert report.block_count == 2
    assert report.changed
    assert "\n" not in report.migrated_source.replace("\r\n", "")
    assert '__import__("numpy").nan' in report.migrated_source
    assert report.migrated_source.count("._pylustrator_generated_version = 2") == 2
    assert diagnose_generated_source(report.migrated_source).diagnostics == ()


def test_cli_is_dry_run_by_default_then_atomically_writes(tmp_path: Path, capsys) -> None:
    path = tmp_path / "figure.py"
    original = (
        "raise RuntimeError('offline doctor must not execute this')\n"
        + generated_block(
            "plt.figure(1).ax_dict = {}",
            "plt.figure(1).axes[0].set_xlim([nan, 1])",
        )
    )
    path.write_text(original)
    path.chmod(0o744)

    assert main([str(path)]) == 1
    assert path.read_text() == original
    assert "PYL202 warning" in capsys.readouterr().out

    assert main(["--diff", str(path)]) == 1
    assert '+plt.figure(1).axes[0].set_xlim([__import__("numpy").nan, 1])' in capsys.readouterr().out

    assert main(["--write", str(path)]) == 0
    output = capsys.readouterr().out
    assert f"{path}: migrated" in output
    assert stat.S_IMODE(path.stat().st_mode) == 0o744
    assert '__import__("numpy").nan' in path.read_text()
    assert diagnose_generated_source(path.read_text()).diagnostics == ()

    assert main(["--json", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format_version"] == 1
    assert payload["generated_schema"] == 2
    assert payload["files"][0]["diagnostics"] == []


def test_cli_preserves_declared_source_encoding(tmp_path: Path, capsys) -> None:
    path = tmp_path / "latin1.py"
    source = (
        "# -*- coding: latin-1 -*-\n"
        "title = 'caf\xe9'\n"
        + generated_block(
            "plt.figure(1).ax_dict = {}",
            "plt.figure(1).axes[0].set_xlim([nan, 1])",
        )
    )
    path.write_bytes(source.encode("latin-1"))

    assert main(["--write", str(path)]) == 0
    capsys.readouterr()
    migrated = path.read_bytes().decode("iso-8859-1")
    assert "title = 'caf\xe9'" in migrated
    assert '__import__("numpy").nan' in migrated


def test_cli_refuses_to_replace_symlink(tmp_path: Path, capsys) -> None:
    target = tmp_path / "target.py"
    link = tmp_path / "link.py"
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1).axes[0].set_xlim([nan, 1])",
    )
    target.write_text(source)
    link.symlink_to(target)

    assert main(["--write", str(link)]) == 2
    assert target.read_text() == source
    assert "refusing to replace a symbolic link" in capsys.readouterr().err


def test_cli_refuses_to_break_hardlink(tmp_path: Path, capsys) -> None:
    target = tmp_path / "target.py"
    link = tmp_path / "hardlink.py"
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1).axes[0].set_xlim([nan, 1])",
    )
    target.write_text(source)
    try:
        os.link(target, link)
    except OSError as error:
        pytest.skip(f"hardlinks unavailable: {error}")

    assert main(["--write", str(link)]) == 2
    assert target.read_text() == source
    assert link.read_text() == source
    assert "refusing to break a multiply linked source file" in capsys.readouterr().err


def test_directory_scan_skips_virtual_environments(tmp_path: Path, capsys) -> None:
    figure = tmp_path / "figure.py"
    figure.write_text(
        generated_block(
            "plt.figure(1).ax_dict = {}",
            "plt.figure(1).axes[0].set_xlim([nan, 1])",
        )
    )
    hidden = tmp_path / ".venv" / "broken.py"
    hidden.parent.mkdir()
    hidden.write_text(f"{START}\n")

    assert main(["--write", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Scanned 1 file(s)" in output
    assert hidden.read_text() == f"{START}\n"
    assert '__import__("numpy").nan' in figure.read_text()


def test_no_generated_blocks_is_clean() -> None:
    source = "value = nan\n# ordinary user source\n"

    report = diagnose_generated_source(source)

    assert report.block_count == 0
    assert report.diagnostics == ()
    assert report.migrated_source == source


def test_invalid_python_is_reported_without_partial_rewrite() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "value =",
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL005" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_marker_text_inside_string_is_not_a_generated_block() -> None:
    source = f'''payload = """\n{START}\nvalue = nan\n{END}\n"""\n'''

    report = diagnose_generated_source(source)

    assert report.block_count == 0
    assert report.diagnostics == ()
    assert report.migrated_source == source
    assert migrate_generated_source(source) == source


def test_marker_suffix_is_rejected() -> None:
    source = f"{START} unexpected\nvalue = nan\n{END}\n"

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL004" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


@pytest.mark.parametrize(
    "start",
    [
        "# % start: automatic generated code from pylustrator",
        f"{START}X",
    ],
)
def test_noncanonical_marker_variants_are_reported(start: str) -> None:
    source = f"{start}\nvalue = nan\n# % end: automatic generated code from pylustrator\n"

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL004" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_marker_indentation_must_match() -> None:
    source = (
        "def configure():\n"
        f"    {START}\n"
        "    plt.figure(1).ax_dict = {}\n"
        "        # ordinary nested comment\n"
        f"        {END}\n"
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL007" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_alias_free_migration_rejects_shadowed_import_builtin() -> None:
    source = (
        "__import__ = lambda name: None\n"
        + generated_block(
            "plt.figure(1).ax_dict = {}",
            "plt.figure(1).axes[0].set_xlim([nan, 1])",
        )
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL304" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_ambiguous_numpy_alias_binding_fails_closed() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "np = custom_namespace",
        "plt.figure(1).axes[0].set_xlim([np.nan, 1])",
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL302" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_numpy_import_followed_by_rebinding_fails_closed() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "import numpy as np",
        "np = object()",
        "plt.figure(1).axes[0].set_xlim([np.nan, 1])",
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL302" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_conditional_numpy_import_does_not_prove_alias_availability() -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "if False:",
        "    import numpy as np",
        "plt.figure(1).axes[0].set_xlim([np.nan, 1])",
    )

    report = diagnose_generated_source(source)

    assert not report.has_errors
    assert "PYL203" in {item.code for item in report.diagnostics}
    assert '__import__("numpy").nan' in report.migrated_source
    assert diagnose_generated_source(report.migrated_source).diagnostics == ()


@pytest.mark.parametrize(
    "binding",
    [
        "nan = 0",
        "if False:\n    nan = 0",
    ],
)
def test_nonfinite_binding_does_not_hide_unsafe_prior_load(binding: str) -> None:
    source = generated_block(
        "plt.figure(1).ax_dict = {}",
        "plt.figure(1)._pylustrator_generated_version = 1",
        "plt.figure(1).axes[0].set_xlim([nan, 1])",
        *binding.splitlines(),
    )

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL302" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source


def test_candidate_is_parsed_before_it_can_be_written(monkeypatch) -> None:
    import pylustrator.source_doctor as source_doctor_module

    source = generated_block("plt.figure(1).ax_dict = {}")
    original_insert = source_doctor_module._insert_version_line

    def invalid_insert(block: str, statement: str) -> str:
        migrated = original_insert(block, statement)
        return migrated.replace(END, f"value =\n{END}")

    monkeypatch.setattr(source_doctor_module, "_insert_version_line", invalid_insert)

    report = diagnose_generated_source(source)

    assert report.has_errors
    assert "PYL006" in {item.code for item in report.diagnostics}
    assert report.migrated_source == source
