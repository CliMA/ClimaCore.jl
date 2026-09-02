# Contributing

ClimaCore.jl is developed in the open, and contributions of every size are
welcome: bug reports, documentation fixes, examples, and features. This page
says how to report a problem, how to set up a development environment, what
the pre-commit hooks and CI check, and how the documentation is built.

!!! note "Developer guides"

    ClimaCore follows the shared [CliMA Developer Guides](https://github.com/CliMA/DeveloperGuides),
    vendored at [`docs/dev-guides/`](https://github.com/CliMA/ClimaCore.jl/tree/main/docs/dev-guides)
    and synced monthly. They hold the code style, comment and documentation
    policy, GPU and type-stability rules, and the review checklist; the
    repository-specific guide is
    [`docs/clima_core_specific.md`](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/clima_core_specific.md).
    Edits to the shared guides belong upstream, not in the vendored copy.

## Ways to contribute

  - Run ClimaCore, and open an issue for anything that is wrong or hard to use.
  - Take an existing [issue](https://github.com/CliMA/ClimaCore.jl/issues);
    those labeled [good first issue](https://github.com/CliMA/ClimaCore.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    need no prior knowledge of the code base.
  - Improve the documentation, or write an example or tutorial.
  - Implement a feature.

Before starting on something larger than a fix, say so on the issue (or open
one), so that the work is not duplicated and design questions are settled
early. For questions rather than problems, use the repository's
[discussions](https://github.com/CliMA/ClimaCore.jl/discussions).

## Reporting a bug

Search the [issues](https://github.com/CliMA/ClimaCore.jl/issues) first. If
the problem is new, open an issue with the bug-report template and include

  - a minimal code snippet that reproduces it on the latest release, or on
    `main` if the release is not affected;
  - the complete error message and stack trace, however long;
  - the output of `versioninfo()` and `] status`, and the hardware (in
    particular the GPU, if any).

## Development setup

 1. Fork the repository on GitHub and clone your fork:

    ```
    git clone https://github.com/<your-user-name>/ClimaCore.jl.git
    cd ClimaCore.jl
    git remote add upstream https://github.com/CliMA/ClimaCore.jl.git
    ```

 2. Instantiate the project and run the tests:

    ```
    julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
    ```

    The full suite takes a long time; [`test/runtests.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/test/runtests.jl)
    shows how to run a subset.

 3. Install the pre-commit hooks (next section), so that formatting is checked
    before each commit.

## Pre-commit hooks

[`.pre-commit-config.yaml`](https://github.com/CliMA/ClimaCore.jl/blob/main/.pre-commit-config.yaml)
defines the checks CI runs on every pull request. Installing the hooks is
optional but saves a round trip:

```bash
uv tool install prek   # or: pipx install prek
prek install           # installs the git hook in this clone
```

[`prek`](https://prek.j178.dev) is a drop-in replacement for
[`pre-commit`](https://pre-commit.com); `pip install pre-commit && pre-commit install`
reads the same configuration. The hooks then run on the staged files at every
`git commit`; `prek run --all-files` sweeps the repository, and
`prek run julia-formatter --all-files` runs one hook. The hooks are:

  - The standard [`pre-commit-hooks`](https://github.com/pre-commit/pre-commit-hooks):
    trailing whitespace, end-of-file newline, mixed line endings, TOML and YAML
    syntax, merge-conflict markers, large files, case conflicts, broken symlinks.
  - `julia-formatter`, which runs JuliaFormatter from the version-pinned
    [`.dev/format/`](https://github.com/CliMA/ClimaCore.jl/blob/main/.dev/format/Project.toml)
    environment with the rules in
    [`.JuliaFormatter.toml`](https://github.com/CliMA/ClimaCore.jl/blob/main/.JuliaFormatter.toml).
    Use the pinned environment: another JuliaFormatter version produces a
    different diff from CI. `format_docstrings` and `format_markdown` are on, so
    docstring bodies and `.md` files are formatted too, and Julia code inside a
    fenced block is reformatted; tag REPL transcripts and program output as
    ```julia-repl`` or ```text``, not `````julia``. The
    vendored `docs/dev-guides/`, `.github/`, and the hand-laid-out `README.md`,
    `NEWS.md`, and `AGENTS.md` are excluded.
  - `markdown-link-ambiguity`, which flags a docs-build failure that is easy to
    introduce and slow to diagnose: Documenter parses `[text](target)` as a link
    even across whitespace or a line break, so bracketed units followed by a
    parenthetical, `[m/s] (at the surface)`, become a link with an unresolvable
    target. Because `docs/make.jl` uses `checkdocs = :exports`, such a link in
    an unrendered docstring stays latent until the symbol is added to a page.
    Separate the bracket and the parenthesis with punctuation, or put the units
    in backticks.

CI runs the same hooks through
[`.github/workflows/format.yml`](https://github.com/CliMA/ClimaCore.jl/blob/main/.github/workflows/format.yml).
Repository-wide formatting commits are listed in
[`.git-blame-ignore-revs`](https://github.com/CliMA/ClimaCore.jl/blob/main/.git-blame-ignore-revs);
`git config blame.ignoreRevsFile .git-blame-ignore-revs` keeps them out of
`git blame`.

## Pull requests

The project follows the [ColPrac](https://github.com/SciML/ColPrac) guide for
collaborative practices. Pull requests go against `main` from a branch of your
fork (collaborators may branch in the main repository). Keep each pull request
to one logical change, leave unrelated files alone, and write commit messages
in the style of [Chris Beams's guide](https://chris.beams.io/posts/git-commit/).
A pull request is merged once a collaborator has reviewed and approved it; an
author with write access merges their own, otherwise a collaborator merges
with the author's consent. Review takes time in proportion to the size of the
change, so small pull requests move faster. Contributors who have opened and
merged a pull request may ask for collaborator status, which adds the ability
to review.

If this is your first pull request, GitHub's
[guide to forks](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks)
and the [Open Source Guides](https://opensource.guide/how-to-contribute/)
explain the workflow; for a one-line fix, the GitHub web editor forks and
opens the pull request for you.

## What CI checks

| Check         | What it does                                                                                                                                                                                                                                                                                                                                                                                |
|:------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prek checks   | The pre-commit hooks above, over the whole repository.                                                                                                                                                                                                                                                                                                                                      |
| Documentation | Builds the documentation, runs its examples and doctests, and checks cross-references; a preview is deployed for the pull request.                                                                                                                                                                                                                                                          |
| Unit tests    | `Pkg.test()` on the latest commit of the pull request, on Linux; stale jobs are cancelled on push. Documentation-only changes skip it.                                                                                                                                                                                                                                                      |
| OS unit tests | The unit tests on Linux, macOS, and Windows.                                                                                                                                                                                                                                                                                                                                                |
| ClimaCore CI  | The expensive tests, the GPU tests, and the examples, as Slurm jobs on an HPC cluster through [Buildkite](https://buildkite.com/clima/climacore-ci), defined in `.buildkite/pipeline.yml`. [`docs/dev-guides/workflow/ci_triage.md`](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/workflow/ci_triage.md) is the checklist for a failure that does not reproduce locally. |

CI does not measure GPU kernel launch latency, because absolute latencies vary
between nodes and with load. A change to the CUDA extension
(`ext/ClimaCoreCUDAExt.jl`, `ext/cuda/`) should be checked by hand with the
scripts in [`perf/`](https://github.com/CliMA/ClimaCore.jl/tree/main/perf) and
[`benchmarks/`](https://github.com/CliMA/ClimaCore.jl/tree/main/benchmarks) or
by profiling.

## Documentation

The documentation is organized by [Diátaxis](https://diataxis.fr/) mode:
tutorials (Literate.jl scripts in `docs/tutorials/`), how-to guides
(`docs/src/howto/`), explanation (`docs/src/explanation/`), and reference
(`docs/src/reference/`, curated `@docs` blocks). A new page goes in the
directory of its mode and in the `pages` tree of `docs/make.jl`; a how-to
title is a task ("Run on a GPU"), and every code block a reader might paste is
an `@example` block or part of a tutorial, so that the build runs it.
[`docs/dev-guides/code-quality/documentation_policy.md`](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/code-quality/documentation_policy.md)
is the policy for pages and docstrings.

Tutorials are [Literate.jl](https://fredrikekre.github.io/Literate.jl/stable/)
scripts: comments become text, code runs and its output is shown, and a
trailing `;` suppresses the output of a line. Plots use CairoMakie with the
`ClimaCore.Visualize` recipes. Write the comments as an article that states its goal
first and shows a result at every step.

To build locally:

```
julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The output is in `docs/build/`; `LiveServer.servedocs()` rebuilds on save.
The build checks external links (`LINKCHECK_STRICT=1` makes a broken link
fatal) and, with `STRICT_DOCSTRING_REFS=1`, fails on a docstring `@ref` to a
symbol that no page renders
([`docs/check_docstring_refs.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/check_docstring_refs.jl)).

## Credits

This guide derives from the [ClimateMachine.jl](https://clima.github.io/ClimateMachine.jl/latest/Contributing/)
and [ClimaAtmos.jl](https://clima.github.io/ClimaAtmos.jl/dev/contributor_guide/)
contributor guides, which in turn derive from those of
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl/blob/main/CONTRIBUTING.md)
and [MetPy](https://github.com/Unidata/MetPy/blob/master/CONTRIBUTING.md).
