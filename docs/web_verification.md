# Web migration verification — 2026-10-21

This report separates implementation evidence from unverified deployment claims.
The branch starts at PR #14, `369f5d1a714df77e8713f28560946741df1d70c8`, which
contains main `27b2c94a71961d6d5ae03eee476446e6fc19c826`.

## Executed checks

Local runtime: Python 3.12.14, Matplotlib 3.10.8, NumPy 2.5.1, the committed lock
file, Qt offscreen for legacy desktop checks, and Playwright 1.56 / Chromium 141
for browser checks. This is local evidence; the PR's GitHub Actions checks are
the source for the remote Python/dependency matrix.

| Check | Observed result |
|---|---|
| Complete Python suite after desktop/headless isolation repair | **1,515 passed, 178 skipped, 3 failed** in 183.06 s; all three failures are existing `<4 ms` timing assertions, detailed below |
| Independent GitHub runners, Python 3.12 and 3.13 locked stacks | **1,518 passed, 178 skipped** on each; all three unchanged strict timing assertions passed |
| All Web tests in a separate installation without PyQt5 or qtpy | **99 passed** in 19.47 s, including native downloaded-code replay |
| Browser editor state tests | **7 passed**: gesture history/cancel, locking, true line geometry, groups/IDs, saved-state concurrency, rotation bounds, direct points |
| Real Chromium complete workflow | **Passed**, including native keyboard history and Python download, data processing, plotted output, 1CRN WebGL, provenance snapshot, checkpoint, export/reload and two-tab conflict |
| Independent design review A | Full scientific workflow plus targeted project-context, geometry, keyboard, colors and responsive repairs confirmed |
| Independent design review B | Metadata race, error recovery, zero-row processing, keyboard checkpoint, mobile nonoverlap and two-model molecular disclosure confirmed |
| Ruff, JavaScript syntax and `git diff --check` | Passed |
| Ty | Exit 0, no type errors; 28 existing unused-ignore warnings remain |
| Wheel/source build and isolated wheel smoke | Built; HTTP assets, project creation and packaged 1CRN parsing passed from an extracted wheel in the Qt-free environment |
| Vendor resources | 3Dmol.js 2.5.5 SHA-256 matches its manifest; both license notices and the real structure's attribution are included |

The 178 skips belong to the existing supported/denied artist capability matrix;
they are not a claim that the skipped operation is supported. The local full run
was not all green. Independent remote runs were green; no timing assertion was
removed or loosened.

## Local timing variability and independent remote confirmation

The exact three failures were rerun once in isolation, then against the untouched
archived baseline with the same dependencies. Values below are measured p95 in ms.

| Existing test | Full branch suite | Isolated branch | Isolated original baseline |
|---|---:|---:|---:|
| Annotation warm preview | 4.006, fail | 18.414, fail | 4.970, fail |
| Mixed-coordinate annotation direct-point preview | 8.543, fail | 67.478, fail | Pass |
| Warm figure hit testing | 4.639, fail | 6.723, fail | 4.684, fail |

Every limit remains `<4 ms`. The baseline reproduces two failures; it does not
reproduce the direct-point failure in this single comparison. The measurements
vary substantially on this shared runtime; that local direct-point result must
not be described as baseline-confirmed.

Subsequent independent GitHub runners passed the **complete 1,518-test suite**
on both locked Python versions, including those exact timing assertions:
[Python 3.12](https://github.com/Molaison/pylustrator-illustrator/actions/runs/39561007480/job/119091982178)
and [Python 3.13](https://github.com/Molaison/pylustrator-illustrator/actions/runs/39561007480/job/119091982272).
This confirms the performance contract on those runners while retaining the
local measurements as evidence of runtime sensitivity. It is not a guarantee
of a 4 ms deadline on every machine.

The initial remote run also exposed two CI setup defects: the unversioned lint
job selected Python 3.15, outside the tested matrix, and minimum-stack pytest
collection imported optional Playwright from the browser script. The follow-up
commit pins lint to Python 3.12 and delays that optional import until the browser
script is actually invoked. Neither change weakens an assertion or adds a GUI
dependency to the headless install.

The interop defect uncovered earlier was functional and was fixed separately:
desktop process-wide text/save wrappers now track only live pyplot-managed
Figures, identified through the registry. Bare Agg Figures retain ordinary
Matplotlib behavior. Same-process desktop → Web render/native-edit regressions
cover this boundary. Two existing timing tests now close their Figures in
`finally`, preventing a failed assertion from corrupting later desktop tests.

## Independent design reviews and resolved findings

The initial architecture/design audit and two fresh implementation reviewers read
the requested source skills, rather than relying on their names:

- [Trystan design system](https://github.com/Trystan-SA/claude-design-system-prompt),
  revision `3c3ddb07d7aa3fef051d83608596470c95cfd8fe`: accessibility, interaction
  states, hierarchy/rhythm and polish.
- [base-impeccable](https://github.com/tool-source/base-impeccable), revision
  `d272b9bd5dcfcb52d32482d192d06045ca31c503`: Operate mode, audit, harden and polish.

Confirmed fixes include:

- A delayed dataset response no longer places project A's table inside project B.
- Metadata saves, restore and project transitions preserve identity and lock the
  relevant navigation until their response can be reconciled.
- Image imports, structure loads and snapshots reject stale project contexts.
- Numerical line resizing changes actual points; direct-point edits have keyboard
  inputs and correct visible bounds.
- CSS color inputs round-trip as server-supported sRGB hex.
- Project details, notes, bundle actions and Help remain reachable at 390 px.
- Returning to a previously hidden Figure viewport correctly fits the artboard.
- Mobile tables and pagination no longer overlap processing controls. Review B
  measured pagination ending at y=882 and controls starting at y=940.5.
- Multi-model structures show the model count and first-model notice; blank chain
  IDs survive saved-view restoration.
- Input borders and missing-value text use verified contrast tokens.
- Native Undo/Redo and save shortcuts target the live Figure; save downloads its
  Python changes, with a matching help description.

Both reviewers inspected real desktop/mobile screenshots and nonempty molecular
renders, with zero page JavaScript errors in their exercised flows. A real 1CRN
source produced 327 atoms and 46 residues. The second reviewer also used a valid
two-model PDB input to verify disclosure and rendering.

Committed screenshots from the final full browser workflow:
[molecular workspace](assets/web-research-structures.png) and
[narrow data workspace](assets/web-research-data-mobile.png).

The final Impeccable source detector was run once over first-party markup, CSS and
JavaScript. It found one initially missing `src` on the dynamically populated native
Figure image. That initial markup now points to the actual attached session's SVG
endpoint, so there is no sourceless image while the session metadata loads. The
detector was not repeatedly rerun to optimize a score.

## Reproduce the checks

```bash
uv sync --locked --all-extras --dev
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ty check
node tests/web/state.test.mjs
uv run --no-sync python -m playwright install chromium
uv run --no-sync python scripts/test_web_browser.py --artifacts ./browser-results
uv build
```

The separate `headless-web` CI job installs only the Web/test extras, asserts that
both PyQt5 and qtpy are absent, and runs all Web tests. The browser job uses a fresh
temporary workspace, a real native Figure and the packaged molecular source.

## Explicit coverage limits

- External provider transport, proposal validation and failures were tested with
  mock HTTP responses. No paid/live provider request was made with user credentials.
- This is not a full Illustrator, arbitrary Bézier or arbitrary Matplotlib
  serialization parity claim. Native capability reasons remain authoritative.
- Native sessions require their original Python process/script. Durable project
  checkpoints refer to composition projects, not pickled live Figures.
- Rotated plot panels and molecular snapshots are raster images in the composed
  figure. Unrotated scientific plot panels retain vector output.
- The browser host lacked a CJK font. Chinese data/name round trips were checked,
  but rendered Chinese glyph coverage was not certified on that host.
- The reviews are bounded workflow/accessibility checks, not a whole-site WCAG or
  multi-user deployment certification.

Development and research records are in
[Linear project P-MOL-14](https://linear.app/molaison/project/pylustrator-web-research-studio-3476632f1788),
including MOL-251–MOL-261. When the workspace's free issue limit was reached,
second-review findings were appended to MOL-259 instead of leaving them untracked.
