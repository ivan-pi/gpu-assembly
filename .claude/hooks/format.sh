#!/usr/bin/env bash
#
# PostToolUse hook: format and lint the file Claude just wrote.
#
# Runs the hooks of .pre-commit-config.yaml on that one file, so that
# Claude's edits come out in the same shape as a commit would, with the
# same pinned tool versions. Formatters (clang-format, Black, Ruff's
# --fix) rewrite the file in place and the hook is silent about it; a
# complaint the tools cannot fix themselves (an undefined name, an
# unused import Ruff will not drop) is fed back to Claude, which then
# fixes it.
#
# Without pre-commit, the tools are tried directly with whatever version
# is installed; without those, the hook says so once and steps aside.
# Either way the edit itself has already happened and is never undone.

set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

file=$(python3 -c 'import json, sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))' 2>/dev/null)
[ -n "$file" ] && [ -f "$file" ] || exit 0

# Only files pre-commit would look at: tracked or inside the work tree,
# not the vendored code and not the case files (see the exclude list in
# .pre-commit-config.yaml, which pre-commit itself applies).
case "$file" in
    "$PWD"/*) rel=${file#"$PWD"/} ;;
    /*) exit 0 ;;
    *) rel=$file ;;
esac

if command -v pre-commit >/dev/null 2>&1 && [ -f .pre-commit-config.yaml ]; then
    # First pass: the formatters rewrite the file (exit 1 when they do).
    # Second pass: only what they could not fix is left, and that is
    # what Claude should hear about.
    if ! out=$(pre-commit run --files "$rel" 2>&1); then
        if ! out=$(pre-commit run --files "$rel" 2>&1); then
            printf '%s\n' "$out" | grep -vE '^\S.*(Passed|Skipped)$' >&2
            echo "format hook: pre-commit still fails on $rel; fix the findings above" >&2
            exit 2
        fi
    fi
    exit 0
fi

# Fallback without pre-commit: the bare tools, if installed.
status=0
case "$rel" in
    third_party/*) exit 0 ;;
    *.py)
        if command -v ruff >/dev/null 2>&1; then
            ruff check --fix --quiet "$rel" || status=2
        fi
        if command -v black >/dev/null 2>&1; then
            black --quiet "$rel"
        else
            echo "format hook: black not installed; 'pip install pre-commit' and run 'pre-commit install-hooks'" >&2
        fi
        ;;
    *.cpp|*.cxx|*.cc|*.h|*.hpp|*.cu|*.cuh)
        if command -v clang-format >/dev/null 2>&1; then
            clang-format -i "$rel"
        else
            echo "format hook: clang-format not installed; 'pip install pre-commit' and run 'pre-commit install-hooks'" >&2
        fi
        ;;
esac
exit "$status"
