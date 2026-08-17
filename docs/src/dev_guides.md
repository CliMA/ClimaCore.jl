# Developer Guides

ClimaCore.jl follows the shared [CliMA Developer Guides](https://github.com/CliMA/DeveloperGuides) — engineering standards, architectural patterns, and development guidelines for human and AI developers across the [CliMA](https://clima.caltech.edu/) ecosystem.

## Where to find them

The guides are vendored as a Git subtree in this repository at `docs/dev-guides/`. The canonical source is [CliMA/DeveloperGuides on GitHub](https://github.com/CliMA/DeveloperGuides). A [scheduled workflow](https://github.com/CliMA/ClimaCore.jl/blob/main/.github/workflows/update_dev_guides.yml) syncs the local copy monthly.

## What they cover

- **Architecture** — repo structure conventions, ecosystem-wide design patterns, cross-repo contracts, dependency management
- **Performance** — GPU kernel rules, broadcast patterns, allocation avoidance, type stability
- **Workflow** — agent autonomy boundaries, code style, testing, and PR review guidelines

## Repo-specific guide

For conventions and patterns specific to ClimaCore.jl (directory layout, key abstractions, test groups), see the [repo-specific guide](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/clima_core_specific.md).

## Editing the guides

Edits to the shared guidelines belong in the canonical [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides) repository, not in the vendored `docs/dev-guides/` copy. Changes made to the subtree copy will be overwritten on the next sync.
