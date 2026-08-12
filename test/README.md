# ClimaCore test suite

This directory holds the ClimaCore test suite. Tests are driven by a small
custom harness (`tabulated_tests.jl`) from `runtests.jl`, and are organized
along two orthogonal axes:

 - a **subsystem** (which module of the package the test exercises), and
 - a **tier** (what *kind* of test it is, and hence how expensive it is).

Our goals are:

 - high source-code coverage in the fast (`:unit`) tier,
 - a fast default run (the `:unit` tier should complete in a few minutes), and
 - correctness tests (convergence, conservation, analytical checks) that we can
   run selectively without slowing down the default run.

## The harness

`runtests.jl` declares a vector of `UnitTest` entries, each of which names a
test, points at a file (relative to `test/`), and tags it with a `tier` and a
`subsystem`:

```julia
UnitTest("FD ops - column", "Operators/finitedifference/unit_column.jl";
         tier = :unit, subsystem = :operators)
```

`tabulated_tests.jl` provides the machinery:

 - `validate_tests` — fails fast on duplicate or non-existent files (so a
   mistyped filename errors immediately, not at the end of a long run).
 - `run_unit_tests!` — runs each file in its own module (`prevent_leaky_tests`)
   so variables and imports do not leak between files. This is our in-house
   equivalent of `SafeTestsets`.
 - `filter_tests` — selects a subset by tier, subsystem, or tag (see below).
 - `tabulate_tests` — prints a timing table at the end of the run.

## Tiers

Every test carries a `tier`. The tier controls what runs in the default suite
and what is opt-in.

| tier         | prefix        | what it is                                                                 |
|--------------|---------------|----------------------------------------------------------------------------|
| `:unit`      | `unit_`       | Fast, high-coverage correctness tests (`Float64`, and `Float32` where cheap). The default run. |
| `:inference` | `inference_`  | Type-stability / `@inferred` / JET regression gates. Kept separate because inference checks are slower. |
| `:allocs`    | `allocs_`     | Runtime-allocation regression tests (warm-up + `@allocated == 0`). Requires running code twice, so separate from `:unit`. |
| `:conv`      | `conv_`       | Convergence tests. Run the code at several resolutions to verify a theoretical convergence rate. |
| `:opt`       | `opt_`        | Optimization / JET `@test_opt` / flop-counting checks. Run with default bounds checking in CI (JET's optimized-IR checks depend on the bounds flag). |
| `:smoke`     | `smoke_`      | Short end-to-end integrations (a few steps of a real driver) that assert conservation / error bounds. |
| `:gpu`       | `gpu_`        | GPU-only tests (CPU-vs-GPU comparison, CUDA kernels). Skipped unless a CUDA device is present. Files under `test/gpu/` need no prefix. |
| `:misc`      | —             | Quality gates that don't fit above (Aqua, deprecations).                   |

The `tier =` tag in `runtests.jl` is authoritative; the filename prefix
should match it for new files. Exceptions: `:misc` files, files under
`test/gpu/`, distributed files (whose names/folders convey the class, e.g.
`ddss2.jl`, `dtopo4.jl`), GPU-tier files with a `*_cuda.jl` suffix instead of
a `gpu_` prefix, and a handful of legacy/reproducer files (e.g.
`Operators/integrals.jl`, `MatrixFields/multiple_field_solve_reproducer_1.jl`,
the `Limiters/vertical_mass_borrowing_limiter*` pair).

Two additional non-tier file classes live under `test/` but are **not** run by
`runtests.jl`:

| prefix       | what it is                                                                  |
|--------------|-----------------------------------------------------------------------------|
| `benchmark_` | Benchmarks. These vary with hardware/noise, so we log results over time rather than asserting on them. Run only in CI perf jobs, at target resolution. |
| `utils_`     | Non-test helpers. These define methods only (no top-level `@test`), and should be written to work for both `Float32` and `Float64`. |

The `meta` tag filters tests by device / launch mode (see the meta-filter
block in `runtests.jl`):

 - `meta = :gpu_only` — dropped unless `ClimaComms.device()` is a `CUDADevice`;
 - `meta = :cpu_only` — dropped when running on a `CUDADevice`;
 - `meta = :distributed` — dropped on single-rank runs (see below).

### Distributed tests

Distributed (MPI) and multi-GPU tests — distributed DSS, ghost-exchange
topology, distributed limiters/remapping, and multi-rank sphere geometry —
are registered in `runtests.jl` with `meta = :distributed` for taxonomy
completeness, but they do not run in-process under a single-rank harness run
(the harness filters them out when `ClimaComms.nprocs(...) == 1`). Most live
under `test/**/distributed/` (CPU) and `test/Spaces/distributed_cuda/` (GPU);
a few sit next to their serial siblings (`Topologies/dtopo4.jl`,
`Operators/spectralelement/sphere_geometry_distributed.jl`,
`Fields/gpu_reduction_distributed.jl`). They are driven from CI (see
`.buildkite/pipeline.yml`) via `srun`/`mpiexec -n <ranks> julia ...` at their
required rank counts. Keep these files self-contained so a CI job can run a
single file.

## Subsystems

The `subsystem` tag mirrors the package's module structure and the folder
layout under `test/`:

`:datalayouts`, `:geometry`, `:domains`, `:meshes`, `:topologies`,
`:quadratures`, `:spaces`, `:fields`, `:operators`, `:matrixfields`,
`:hypsography`, `:limiters`, `:io`, `:remapping`, `:integration`, `:gpu`,
`:quality`, `:utilities`.

Note: `:gpu` is a device axis rather than a module — GPU tests live in the
folder of the subsystem they exercise (e.g. `test/Operators/hybrid/gpu_ops.jl`)
but carry `subsystem = :gpu` so that a single `TEST_TIER=gpu` /
`TEST_SUBSYSTEM=gpu` selection runs them all on a GPU agent.

Tests are first organized in folders by the package module they exercise
(`ClimaCore.Spaces` → `test/Spaces/`), then named by tier prefix within that
folder.

## Naming conventions

 - Folder = subsystem: `test/<Subsystem>/`.
 - Filename prefix = tier: `unit_`, `inference_`, `allocs_`, `conv_`, `opt_`,
   `smoke_`, `gpu_`, `benchmark_`, or `utils_` for shared helpers.
 - Split argument construction from the test itself. A `utils_*.jl` file (or a
   shared constructor in `TestUtilities`/`CommonSpaces`/`CommonGrids`) should
   build the spaces/fields; the `unit_`/`conv_`/... file should consume them.
   This keeps files short, REPL-friendly, and lets the different tiers share
   setup instead of copy-pasting it.

Shared setup lives in:

 - `TestUtilities/TestUtilities.jl` — space constructors (`all_spaces`,
   `SphereSpectralElementSpace`, ...), the `@test_precisions` dual-precision
   sweep macro, and allocation helpers.
 - `CommonSpaces/`, `CommonGrids/` — convenience constructors for common
   spaces and grids.

Prefer these over hand-rolling `Domain → Mesh → Topology → Space` in each file.

## Running the suite

The suite runs under the package's own project plus the `test` target extras
(declared in the top-level `Project.toml`'s `[targets]`), so it is driven by
`Pkg.test`, not a separate `test/Project.toml`:

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

Or from the package REPL:

```
ClimaCore> test
```

### Filtering

The run can be narrowed via environment variables (see the `filter_tests`
block in `runtests.jl`). Set them in the shell (or via `ENV`) before invoking:

| variable                | effect                                                        |
|-------------------------|---------------------------------------------------------------|
| `TEST_TIER`             | run only a given tier, e.g. `TEST_TIER=conv`                  |
| `TEST_EXCLUDE_TIER`     | drop tiers (comma-separated), e.g. `TEST_EXCLUDE_TIER=conv,opt` |
| `TEST_SUBSYSTEM`        | run only a given subsystem, e.g. `TEST_SUBSYSTEM=operators`   |
| `TEST_TAG`              | case-insensitive substring match on test name or filename, e.g. `TEST_TAG=dss` |
| `TEST_FAST=true`        | run only the `:unit` tier (drops conv/opt/smoke/...)          |
| `TEST_FAIL_FAST=false`  | run all tests and summarize failures at the end (default stops at the first failing file) |
| `CLIMACORE_TEST_OPT=true` | also run the CI-only opt checks (JET + allocation gates) embedded in the MatrixFields broadcasting tests. These default to on only under Buildkite (`ENV["BUILDKITE"]`), so a plain local run skips them — set this before trusting a local green run to predict CI. |

```bash
TEST_SUBSYSTEM=operators TEST_TAG=dg julia --project -e 'using Pkg; Pkg.test()'
```

If the filters match zero tests (e.g. a misspelled subsystem), the run errors
with a message naming the filter values — it never exits green having run
nothing.

To iterate on a single file in the REPL, `include` it directly under a project
that has the test extras available (e.g. after a `Pkg.activate(".")` + `Pkg.instantiate()`
with the test target), as the `#=` header at the top of most test files shows.

## Adding a test

1. Put the file under the subsystem folder, prefixed by its tier
   (`test/Operators/unit_my_thing.jl`).
2. Build spaces/fields via `TestUtilities`/`CommonSpaces` where possible.
3. Where cheap, run scientific tests for both `Float32` and `Float64` — the
   prevailing idiom is `@testset "... [$FT]" for FT in (Float32, Float64)`
   (per-precision testset labels); `TestUtilities.@test_precisions` is an
   equivalent loop without the labels. Express tolerances as multiples of
   `eps(FT)`; use `rtol` for scale-invariant comparisons and `atol` only
   where a value can be exactly zero.
4. Register it with a new `UnitTest(...)` entry in `runtests.jl`, with the
   correct `tier` and `subsystem`.
5. Every test file must assert something: a file that only checks "this code
   runs without throwing" (e.g. a crash reproducer) should still end with a
   `@test` on the computed values — silent wrong results are otherwise
   invisible.
6. Prefer measuring allocations from a dedicated `@noinline` helper, rather
   than inline in the test:

   ```julia
   @noinline my_op_allocs(dest, x) = @allocated my_op!(dest, x)
   @test my_op_allocs(dest, x) == 0        # preferred
   @test (@allocated my_op!(dest, x)) == 0 # can be brittle: the result depends
                                           # on how much of the caller's
                                           # inlining budget the surrounding
                                           # code consumed
   ```

   Inline measurements have failed CI with spurious 16–80-byte "allocations"
   after unrelated edits changed the enclosing function's size. The risk grows
   with the size of the enclosing function, so the helper matters most inside
   long test functions and shared helpers; a short `@testset` that measures a
   small call is usually fine as-is. Plenty of existing tests still measure
   inline — convert them when one turns brittle, not on sight.

   Either way, keep the thresholds hardware-independent (allocation counts, not
   times); anything time-based belongs in a `benchmark_` file or a soft-fail
   perf job.

See `docs/dev-guides/infrastructure/testing_and_validation.md` for the
scientific-test categories (limits, round-trip, analytical, consistency) that
new physics/operator tests should draw from.
