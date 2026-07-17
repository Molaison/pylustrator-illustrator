Offline generated-source doctor
===============================

Why it is offline
-----------------

Pylustrator-generated blocks are ordinary Python and normally run before the
editor opens. Some historical source defects therefore cannot be repaired by
runtime compatibility code. A bare ``nan`` or ``inf`` name, for example, can
raise ``NameError`` before Pylustrator can inspect the block.

``pylustrator-source`` reads the file as data. It does not import the file,
construct a figure, or evaluate a generated command.

Usage
-----

.. code-block:: console

   # Read-only diagnosis (the default).
   pylustrator-source figure.py
   pylustrator-source figures/

   # Preview or apply the candidate migration.
   pylustrator-source --diff figure.py
   pylustrator-source --write figure.py

   # Stable machine-readable output for CI and batch audits.
   pylustrator-source --json figures/

Directories are scanned recursively for ``*.py`` files. Virtual environments,
version-control metadata, build output, caches, and ``node_modules`` are
excluded.

JSON output carries an independent ``format_version`` as well as the
``generated_schema`` it can migrate, so automation does not conflate the report
shape with the generated-source schema.

Safety contract
---------------

The doctor:

* recognizes only an exact start/end marker represented by a Python comment,
  so marker text in a string is not a block;
* rewrites only complete generated blocks and leaves surrounding user source
  untouched;
* understands the generated schema version and refuses unknown future,
  duplicate, non-integer, nested, orphaned, or unclosed block state;
* rewrites only an indexed legacy Legend proxy locator, never the semantic
  ``get_legend_handles_labels()[0]`` list used to construct a new Legend;
* qualifies bare non-finite values only when the Python AST identifies them as
  unbound read expressions, using an alias-free NumPy lookup that cannot change
  ``np`` scoping in an enclosing function;
* produces an idempotent candidate and validates that candidate before writing;
* writes a temporary file in the same directory, flushes it, preserves the
  original mode and encoding, checks for concurrent source changes, and then
  replaces the original atomically; and
* refuses to replace symbolic links, break hard-linked source files, or
  partially repair a file with a non-fixable diagnostic.

The default command is always read-only. ``--write`` is the explicit mutation
boundary.

Exit status
-----------

``0``
   Every inspected source is clean, or ``--write`` completed all safe
   migrations.

``1``
   Source diagnostics or unsupported schema state remain.

``2``
   Usage, decoding, I/O, race-detection, or atomic-write failure.

Library API
-----------

``pylustrator.diagnose_generated_source`` returns a ``SourceDoctorReport`` with
structured diagnostics and the candidate ``migrated_source``. It obeys the
same no-execution and fail-closed rules as the command-line tool.
