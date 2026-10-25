# Dependency and CI checks

P0 completed 2026-10-25. PyYAML is a base dependency because annotation curation
is a core CLI command. Its absence caused two curation tests to fail in a base
installation; those tests now run without optional ML packages. No locked package
versions changed.

Fresh base testing also exposed a preparation test missing its NumPy dependency
guard. That test now follows the existing optional-pose test convention; the full
CI job still requires it to execute. Curation tests are not skipped.

`.github/workflows/checks.yml` runs separate locked base and pose/train installs.
The full job uses `pytest --require-no-skips`, which fails for collection or runtime
skips. Both jobs check Ruff and CLI help. Tests require no dataset or model download.

Local verification in two new Python 3.11.17 environments:

- Base: **108 passed, 68 skipped** (optional pose/ML dependencies absent).
- Pose/train: **289 passed, no skips**.
- Ruff and base CLI/curation help: passed.
- Negative control: base recording tests with `--require-no-skips` returned exit 1
  for two skips, as intended.

These are local macOS results, not a claim that remote Ubuntu CI has run. Future
service tests belong in the base suite and must not require training dependencies.

Reproduce with `uv sync --locked` for base, or
`uv sync --locked --extra pose --extra train` for full. Use the corresponding
`uv run --locked [--extra pose --extra train] pytest -ra` command and add
`--require-no-skips` for full coverage. See the README for copyable commands.
