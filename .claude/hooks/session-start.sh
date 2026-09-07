#!/usr/bin/env bash
#
# SessionStart hook: make sure the host toolchains CI builds with are
# present before the session starts.
#
# The project declares `LANGUAGES CXX Fortran`, so `cmake -B build` fails
# during configure if no Fortran compiler is installed. The base image
# ships gcc/g++ and clang but never gfortran or flang, so every remote
# session would otherwise begin by re-diagnosing the same failure.
#
# The toolchains mirror .github/workflows/ci.yml:
#
#   g++ / gfortran         host-tests, ubuntu-latest
#   clang++-20 / flang-20  older-llvm, ubuntu-24.04
#   clang++-22 / flang-22  host-tests, ubuntu-26.04  -- see below
#
# clang/flang 22 are only packaged for Ubuntu 26.04; on this 24.04 image
# they are not installable, so that CI job cannot be reproduced locally.
# 20 is the LLVM release the Ubuntu 24.04 archive carries and is the one
# to reach for when reproducing an LLVM-specific failure.
#
# The CUDA parts (-DGPU_ASSEMBLY_ENABLE_CUDA=ON) need NVHPC and MathDx,
# neither of which is an apt package; they stay out of scope here and the
# option remains OFF by default.

set -euo pipefail

# Developers' own machines are their own business: only provision the
# ephemeral containers used by Claude Code on the web.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

# Tools this project needs, each with the package that provides it.
# Probed by executable name so a re-run on a warm (cached) container
# installs nothing.
REQUIRED_TOOLS=(
    "g++:g++"
    "gfortran:gfortran"
    "clang++-20:clang-20"
    "flang-20:flang-20"
    "cmake:cmake"
    "ninja:ninja-build"
)

# Packages with no executable of their own to probe; asked of dpkg instead.
# libomp-20-dev is what gives clang++-20 and flang-20 their OpenMP runtime,
# which CMakeLists.txt looks for via find_package(OpenMP).
REQUIRED_PACKAGES=(
    libomp-20-dev
)

missing=()

for entry in "${REQUIRED_TOOLS[@]}"; do
    tool=${entry%%:*}
    package=${entry##*:}
    if ! command -v "$tool" >/dev/null 2>&1; then
        missing+=("$package")
    fi
done

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -s "$package" >/dev/null 2>&1; then
        missing+=("$package")
    fi
done

if [ ${#missing[@]} -eq 0 ]; then
    echo "session-start: host toolchains already present, nothing to install"
else
    echo "session-start: installing ${missing[*]}"

    sudo=""
    if [ "$(id -u)" -ne 0 ]; then
        sudo="sudo"
    fi

    export DEBIAN_FRONTEND=noninteractive

    # Third-party sources in the base image (docker, a couple of PPAs) are
    # unreachable through the session proxy and make `update` exit non-zero.
    # The Ubuntu archive itself refreshes fine, and that is where all of
    # these packages come from, so a partial update is not fatal here --
    # the install below is what actually has to succeed.
    $sudo apt-get update -qq || echo "session-start: apt-get update reported errors, continuing"

    $sudo apt-get install -y --no-install-recommends "${missing[@]}"
fi

# Report what the session ended up with, and fail loudly rather than
# leaving Claude to rediscover a missing compiler mid-build.
status=0
for entry in "${REQUIRED_TOOLS[@]}"; do
    tool=${entry%%:*}
    if command -v "$tool" >/dev/null 2>&1; then
        printf '  %-12s %s\n' "$tool" "$("$tool" --version 2>&1 | head -1)"
    else
        printf '  %-12s NOT FOUND\n' "$tool"
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo "session-start: a required toolchain is missing; see above" >&2
    exit 1
fi

# The Python side: the pointclouds package under data/ with the
# libraries its tools need (numpy, scipy, matplotlib; the `tools` extra
# in data/pyproject.toml), and the formatting and linting tools that
# .claude/hooks/format.sh runs after every edit. pre-commit builds one
# environment per hook of .pre-commit-config.yaml on first use, which
# takes a minute; done here so the first edit does not pay for it.
#
# The `reorder` extra (pymetis, scikit-sparse) compiles against
# SuiteSparse and is left to whoever needs it.
#
# None of this is fatal: the compilers above are what a session cannot
# do without, the Python it can ask for by hand.
echo "session-start: installing the Python tools (pointclouds[tools], pre-commit, black, ruff)"
if python3 -m pip install --quiet -e "data[tools]" pre-commit black ruff; then
    if pre-commit install-hooks >/dev/null 2>&1; then
        echo "session-start: pre-commit hook environments ready"
    else
        echo "session-start: pre-commit install-hooks failed; format.sh will fall back to black/ruff/clang-format"
    fi
else
    echo "session-start: pip install failed; the data/ scripts and the format hook will complain"
fi

echo "session-start: build with 'cmake -B build && cmake --build build && ctest --test-dir build'"
echo "session-start: for the LLVM toolchain, 'CXX=clang++-20 FC=flang-20 cmake -B build-llvm'"
echo "session-start: 'pre-commit run --all-files' checks the style; see README.md, \"Formatting and linting\""
