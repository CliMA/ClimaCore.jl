# DeveloperGuides

Shared engineering standards, architectural patterns, and development guidelines for human and AI developers across the [CliMA](https://clima.caltech.edu) ecosystem.

## Guides

Every guide applies across the CliMA ecosystem unless it says otherwise.

### Architecture

- [repo_structure.md](architecture/repo_structure.md): how to navigate any CliMA Julia package.
- [ecosystem_conventions.md](architecture/ecosystem_conventions.md): module aliases, `Y`/`Yₜ`/`p` state layout, `ᶜ`/`ᶠ` notation, CI structure, reproducibility, diagnostics.
- [architectural_boundaries.md](architecture/architectural_boundaries.md): layered architecture and boundary rules.
- [cross_repo_contracts.md](architecture/cross_repo_contracts.md): call-site conventions for ecosystem packages.
- [dependency_management.md](architecture/dependency_management.md): runtime vs dev dependencies, compat bounds.

### Performance

- [gpu_performance.md](performance/gpu_performance.md): GPU kernel rules, broadcast patterns, allocation avoidance.
- [branchless_code.md](performance/branchless_code.md): avoiding warp divergence with `ifelse`, evaluate-both-cases splits, and fixed-iteration solvers chosen by offline tests.
- [type_stability.md](performance/type_stability.md): Float32 compatibility, inference checks, struct field rules.
- [numerical_robustness.md](performance/numerical_robustness.md): denominator regularization, clamping, NaN/Inf avoidance.
- [ad_compatibility.md](performance/ad_compatibility.md): AD-safe patterns for ForwardDiff and Enzyme.
- [allocation_debugging.md](performance/allocation_debugging.md): locating heap allocations with `Profile.Allocs`, JET, `@code_warntype`, flame graphs.

### Code Quality

- [getting_started.md](code-quality/getting_started.md): orienting newcomers to writing pointwise code compatible with ClimaCore `Field`s and broadcasting.
- [code_style.md](code-quality/code_style.md): formatting, variable locality, Git workflow, feature removal, naming conventions.
- [documentation_policy.md](code-quality/documentation_policy.md): docstrings, repository-level docs, minimally viable documentation.
- [changelogs_and_versions.md](code-quality/changelogs_and_versions.md): `NEWS.md` format, SemVer rules, and the release/tagging flow.
- [variable_list.md](code-quality/variable_list.md): standardized CliMA variable naming conventions.
- [glossary.md](code-quality/glossary.md): general CliMA software and simulation terminology.
- [software_design_patterns.md](code-quality/software_design_patterns.md): numbered SDPs for branchless logic, functors, parameter extraction, and more.

### Infrastructure

- [testing_and_validation.md](infrastructure/testing_and_validation.md): type-stability checks, Aqua.jl, allocation regression, AD tests.
- [clima_comms.md](infrastructure/clima_comms.md): device-agnostic and MPI-distributed code patterns.

### Workflow

- [onboarding.md](workflow/onboarding.md): install Julia, clone a CliMA repo, set up Revise/Infiltrator/JuliaFormatter, first PR loop.
- [running_on_gpu.md](workflow/running_on_gpu.md): run a model on GPU — install Julia, add `CUDA.jl`, CUDA runtime compatibility, `CLIMACOMMS_DEVICE`, verify the device.
- [agent_autonomy.md](workflow/agent_autonomy.md): actions that require explicit user approval.
- [debugging.md](workflow/debugging.md): interactive debugging recipes for numerical instabilities, dispatch, and `Field` plotting.
- [review.md](workflow/review.md): PR review instructions and checklist.
- [ci_triage.md](workflow/ci_triage.md): checklist for "passes locally, fails on CI" failure modes.
- [cross_repo_issue_pr_search.md](workflow/cross_repo_issue_pr_search.md): org-scoped GitHub search to find and filter issues/PRs across CliMA.

---

## About DeveloperGuides

These guides are maintained in [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides) and vendored into consumer repos as a Git subtree at the standardized path `docs/dev-guides/`. The material below is for maintaining and consuming that subtree; readers looking for engineering guidance want the [Guides](#guides) overview above.

### Using the guides in a consumer repo

A consumer repo keeps its own `AGENTS.md` at the root, which references `docs/dev-guides/AGENTS.md` (the agent entry point) plus a repo-specific guide (e.g. `docs/clima_atmos_specific.md`). See [`templates/`](templates/) for ready-to-copy starter files: a root `AGENTS.md`, a repo-specific guide skeleton, and the monthly sync workflow.

```bash
# Add the subtree to a new consumer repo
git subtree add --prefix docs/dev-guides \
    https://github.com/CliMA/DeveloperGuides.git main --squash

# Pull the latest guides manually (most repos automate this monthly via update_dev_guides.yml)
git subtree pull --prefix docs/dev-guides \
    https://github.com/CliMA/DeveloperGuides.git main --squash \
    -m "chore: sync dev guides from central repo"
```

> [!NOTE]
> **Subtree pitfalls.**
>
> - DeveloperGuides ships its own `AGENTS.md`, `LICENSE`, and `README.md` at the repo root, which conflict with the consumer's own root files during `git subtree add`. Resolve by keeping the consumer's versions: `git checkout --ours AGENTS.md LICENSE README.md && git add … && git rebase --continue`.
> - `git subtree pull` exits with an error when there are no new commits upstream. In an automated workflow, append `|| true` so the step does not fail on months with no DeveloperGuides changes.

### Contributing

Edits to shared guidelines belong in [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides), not in the vendored copy inside a consumer repo. Open PRs there; once merged, the next subtree pull propagates them to every consumer.

- Each guide has a **Self-correction** section: if you discover a guide is stale or missing a pattern, update it directly.
- New guides go in the appropriate category directory and are added to this overview and to [AGENTS.md](AGENTS.md).
- Cross-references between guides use relative paths (e.g. `../performance/gpu_performance.md`).

### Repository layout

```text
├── AGENTS.md                  # Agent entry point: autonomy gate + guide index
├── README.md                  # This file: guide overview + repo info
├── architecture/              # System design, layering, contracts
├── performance/               # GPU, type stability, numerics, AD
├── code-quality/              # Style, docstrings, changelogs, naming
├── infrastructure/            # Testing, device abstraction
├── workflow/                  # Onboarding, debugging, review, CI triage
└── templates/                 # Starter files for consumer repos
```

### CliMA ecosystem

These guides are the central source of engineering standards across [CliMA](https://github.com/CliMA), including:

- [ClimaAtmos](https://github.com/CliMA/ClimaAtmos.jl)
- [ClimaCore](https://github.com/CliMA/ClimaCore.jl)
- [ClimaLand](https://github.com/CliMA/ClimaLand.jl)
- [ClimaOcean](https://github.com/CliMA/ClimaOcean.jl)
- [ClimaCoupler](https://github.com/CliMA/ClimaCoupler.jl)
- [Thermodynamics](https://github.com/CliMA/Thermodynamics.jl)
- [CloudMicrophysics](https://github.com/CliMA/CloudMicrophysics.jl)
- [SurfaceFluxes](https://github.com/CliMA/SurfaceFluxes.jl)
- [ClimaTimeSteppers](https://github.com/CliMA/ClimaTimeSteppers.jl)

### License

[![license][license-img]][license-url] Apache 2.0; see [LICENSE](LICENSE).

[license-img]: https://img.shields.io/github/license/CliMA/DeveloperGuides
[license-url]: https://github.com/CliMA/DeveloperGuides/blob/main/LICENSE

### Getting help

For questions or suggestions, open an issue on [GitHub](https://github.com/CliMA/DeveloperGuides/issues).
