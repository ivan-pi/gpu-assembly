# Developer notes

## Formatting and linting

The C++ and CUDA are formatted with [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
in the style of `.clang-format` (Google's, with a 4-space indent and
100 columns) and linted with [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
per `.clang-tidy`; the Python under `data/` with
[Black](https://black.readthedocs.io/) and [Ruff](https://docs.astral.sh/ruff/),
with the settings in `data/pyproject.toml`. All of it runs through
[pre-commit](https://pre-commit.com/), whose `.pre-commit-config.yaml`
pins the tool versions that CI checks with (`.github/workflows/style.yml`):

```
pip install pre-commit
pre-commit install                 # format on every commit from now on
pre-commit run --all-files         # or by hand, over the whole tree
```

clang-tidy wants the compile database of a configured build tree and
is a manual stage:

```
cmake -B build
pre-commit run --hook-stage manual clang-tidy --all-files
```

A block the formatter must leave alone, such as a hand-aligned table,
sits between `// clang-format off` and `// clang-format on`.

The reformatting commit that introduced the style is listed in
`.git-blame-ignore-revs`; `git blame` skips it after

```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Claude Code hooks

`.claude/settings.json` wires up two hooks:

- `.claude/hooks/session-start.sh` provisions a fresh remote session:
  the compilers CI builds with, the `pointclouds` package with its
  `tools` extra, and pre-commit with its hook environments. It does
  nothing on a developer's own machine.
- `.claude/hooks/format.sh` runs after every `Edit` or `Write`: the
  pre-commit hooks on that one file, so Claude's edits come out the
  way a commit would. Formatters fix silently; a finding they cannot
  fix, such as an undefined name, is fed back to Claude to correct.
