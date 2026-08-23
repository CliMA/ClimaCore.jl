# AGENTS.md

Read these documents in the order listed when starting work on this repository.

## Before you act: agent autonomy

Before making changes that are externally visible or scientifically consequential (`git push`, version bumps, reproducibility-test edits, CI config changes, public API renames), check [docs/dev-guides/workflow/agent_autonomy.md](docs/dev-guides/workflow/agent_autonomy.md). The boundaries listed there require explicit user approval.

## Shared guides (via DeveloperGuides subtree)

The shared engineering guidelines are vendored as a Git subtree at `docs/dev-guides/`. Julia's `Pkg` does not resolve git submodules, so subtree is the standard mechanism across the CliMA ecosystem. The full index lives at [docs/dev-guides/AGENTS.md](docs/dev-guides/AGENTS.md). Start there for:
- Architecture, design patterns, and cross-repo contracts
- GPU performance, type stability, and AD compatibility
- Code style, testing, and PR review

Edits to shared guidelines belong in [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides), not in the vendored copy here. A scheduled workflow (`.github/workflows/update_dev_guides.yml`) syncs the subtree monthly.

## Formatting and pre-commit hooks

- Format code before committing. CI checks formatting via the `prek` hook in [.github/workflows/run-prek.yml](.github/workflows/run-prek.yml), which runs JuliaFormatter from the version-pinned [.dev/format/Project.toml](.dev/format/Project.toml) environment (currently `=2.10.1`). Match CI with either `prek run julia-formatter --all-files` or the pinned env directly: `julia --startup-file=no --project=.dev/format -e 'using Pkg; Pkg.instantiate(io=devnull); using JuliaFormatter; format(ARGS)' .`. Avoid `julia -e 'using JuliaFormatter; format(".")'` from your global environment — a mismatched version produces a different diff.
- Optional but recommended: install the pre-commit hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) (`uv tool install prek && prek install`) to auto-format and trim trailing whitespace on commit. See [docs/src/Contributing.md](docs/src/Contributing.md) ("Pre-commit hooks").

## Repo-specific guide

- [docs/clima_core_specific.md](docs/clima_core_specific.md) — directory layout, key abstractions, test groups, and conventions specific to this package.

## Self-correction

If this file is discovered to be stale or missing a section, update it.
