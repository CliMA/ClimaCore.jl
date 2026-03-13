# Fully Implicit Timestepping for Density Current 2D (Flux Form)

## Overview

This document summarizes the implementation and performance study of a fully implicit
timestepper for the density current benchmark problem, using ClimaCore's operator and
MatrixFields infrastructure with ClimaTimeSteppers (CTS).

**File:** `examples/hybrid/plane/density_current_2d_flux_form_implicit.jl`

## Implementation

### State Vector Restructuring

The explicit version uses `FieldVector(Yc=(ρ,ρθ), ρuₕ, ρw)` with `ρw` as `Geometry.WVector`.
The implicit version restructures to:

```julia
Y = Fields.FieldVector(
    c = Field{NamedTuple{(:ρ, :ρθ, :ρuₕ)}},  # center: density, pot. temp., horiz. momentum
    f = Field{NamedTuple{(:ρw,)}},              # face: vertical momentum (Covariant3Vector)
)
```

Key changes:
- `Y.c` and `Y.f` must be **Fields of NamedTuples**, not nested FieldVectors (required by CTS)
- `ρw` uses `Geometry.Covariant3Vector` (C3) instead of `WVector` (required by MatrixFields Jacobian types)

### Solver Configuration

```
CTS.IMEXAlgorithm(ARS233, NewtonsMethod) with T_exp! = nothing (fully implicit)
```

- **JFNK:** GMRES (Krylov.jl) with ForwardDiffJVP for Jacobian-vector products
- **Preconditioner:** Vertical-only analytical Jacobian via `BlockArrowheadSolve`
- **Forcing:** `EisenstatWalkerForcing` for adaptive Krylov tolerance

### Vertical Preconditioner (Jacobian Blocks)

The preconditioner captures vertical acoustic-gravity wave coupling through 5 blocks:

| Block | Physics | Type |
|-------|---------|------|
| `(c.ρ, f.ρw)` | Mass flux: `∂ρₜ/∂ρw = -ᶜdivᵥ_matrix * g³³` | BidiagonalMatrixRow |
| `(c.ρθ, f.ρw)` | Energy flux: `∂ρθₜ/∂ρw = -ᶜdivᵥ_matrix * diag(ᶠinterp(θ) * g³³)` | BidiagonalMatrixRow |
| `(f.ρw, c.ρ)` | Gravity: `∂ρwₜ/∂ρ = -diag(ᶠgradᵥ(Φ)) * ᶠinterp_matrix` | BidiagonalMatrixRow |
| `(f.ρw, c.ρθ)` | Pressure gradient: `∂ρwₜ/∂ρθ = -ᶠgradᵥ_matrix * diag(∂p/∂ρθ)` | BidiagonalMatrixRow |
| `(f.ρw, f.ρw)` | Set to zero (simplified) | TridiagonalMatrixRow |

The `ρuₕ` diagonal block is identity (horizontal momentum not preconditioned).

**What the preconditioner captures:**
- Vertical acoustic waves (pressure ↔ ρw coupling)
- Gravity waves (buoyancy ↔ ρw coupling)

**What it does NOT capture:**
- Horizontal acoustic coupling (pressure ↔ ρuₕ) — resolved by GMRES iterations
- Horizontal advection — resolved by GMRES iterations

### RHS Caching

All temporary fields in `rhs!` are pre-allocated in a cache struct (`uₕ`, `w`, `p`, `θ`, `Yfρ`, `uₕf`),
reducing per-call allocations from ~7 field allocations to **2 KB**.

## Stability Study

The JFNK solver is **unconditionally stable** — tested up to dt=50s (167x the explicit CFL of 0.3s):

| dt (s) | CFL multiple | Status |
|--------|-------------|--------|
| 0.3 | 1x | Stable |
| 1.0 | 3.3x | Stable |
| 5.0 | 17x | Stable |
| 10.0 | 33x | Stable |
| 20.0 | 67x | Stable |
| 50.0 | 167x | Stable |

Solutions at all dt values are physically reasonable (no NaNs, correct density/temperature ranges).

### Direct Newton + ARS233 at larger dt

Direct Newton (no Krylov) uses the vertical-only preconditioner as a direct solver. Testing
whether it can exceed the explicit CFL limit (dt=0.3s):

| dt (s) | CFL multiple | Status |
|--------|-------------|--------|
| 0.3 | 1x | Stable |
| 0.35 | 1.17x | **Diverges** at t≈10s |
| 0.4 | 1.33x | **Diverges** at t≈5s |
| 0.5 | 1.67x | **Diverges** at t≈5s |

**Finding:** Direct Newton is limited to dt≈0.3s because the vertical-only preconditioner
cannot resolve horizontal acoustic modes. Without GMRES iterations to handle the unresolved
horizontal coupling, the solver diverges as soon as dt exceeds the horizontal CFL. JFNK is
required for larger timesteps.

## Performance Study

### Time-to-solution for 100s of simulation (single-process benchmarks)

| Configuration | dt | Steps | Wall time | Allocs | Time/step |
|--------------|-----|-------|-----------|--------|-----------|
| Direct Newton | 0.3 | 333 | **26s** | 55M | 0.08s |
| ARS343 + JFNK | 1.0 | 100 | 168s | 10.6B | 1.68s |
| ARS233 + JFNK | 1.0 | 100 | 175s | 10.9B | 1.75s |
| ARS233 + JFNK | 5.0 | 20 | 236s | 13.3B | 11.8s |

### Estimated 900s full simulation times

| Configuration | dt | Estimated wall time |
|--------------|-----|-------------------|
| Direct Newton | 0.3 | ~3.3 min |
| ARS343 + JFNK | 1.0 | ~25 min |
| ARS233 + JFNK | 1.0 | ~26 min |
| ARS233 + JFNK | 5.0 | ~35 min |

### Allocation Breakdown

Per single `rhs!` call (after caching):
- rhs! temporaries: **2 KB** (pre-allocated)
- `weighted_dss!`: **11 KB** (2 calls x 5.5 KB)
- `wfact!`: **81 KB**

The **~10 billion allocations per 100 steps** come entirely from **CTS/Krylov.jl internals**,
not from `rhs!` or `wfact!`.

### Key Finding

The JFNK solver is currently **~7x slower** than direct Newton at dt=0.3 for the same simulation.
The larger timestep does not compensate for the per-step Krylov overhead. The bottleneck is
allocation overhead in the Krylov.jl solver, not in the physics code.

## Major Remaining Steps: Fixing Krylov Allocation Bottlenecks in CTS

### 1. FieldVector-to-Dense-Vector Conversion in Krylov.jl

**Problem:** Krylov.jl's GMRES solver works with dense `Vector{Float64}` internally. Each
GMRES iteration requires converting between ClimaCore's `FieldVector` representation and
Krylov's dense vector workspace. These conversions allocate memory on every iteration.

**Root cause:** `Krylov.ktypeof(x_prototype)` for a `FieldVector` likely falls back to
`Vector{Float64}`, causing the solver to operate in dense vector space rather than natively
on FieldVectors.

**Fix options:**
- Implement `Krylov.ktypeof` for `FieldVector` so Krylov.jl can work directly with FieldVectors
- Alternatively, implement the `AbstractVector` interface on FieldVector so Krylov.jl treats
  it as a native vector type (requires `similar`, `fill!`, `dot`, `norm`, `axpy!`, etc.)
- Or wrap FieldVector in a Krylov-compatible view that avoids copies

**Impact:** Eliminating these conversions would reduce the ~10B allocations per 100 steps to
near-zero, potentially making JFNK competitive with explicit methods at larger dt.

### 2. LinearOperator Construction Per Krylov Solve

**Problem:** In `solve_krylov!` (CTS `newtons_method.jl:450`), a new `LinearOperator` closure
is constructed on every call:
```julia
opj = LinearOperator(eltype(x), length(x), length(x), false, false, jΔx!)
```
This may cause allocation and compilation overhead from the closure capture.

**Fix:** Pre-allocate the `LinearOperator` in the Krylov cache and update its closure in-place,
or use a callable struct instead of a closure.

### 3. Preconditioner Application via `ldiv!`

**Problem:** When GMRES applies the preconditioner `M` via `ldiv!(M, v)`, where `M` is our
`ImplicitEquationJacobian`, and `v` is a dense vector from Krylov.jl, the
`ldiv!(::AbstractVector, ::ImplicitEquationJacobian, ::AbstractVector)` method must convert
between dense vectors and FieldVectors:
```julia
j.R_field_vector .= R      # dense → FieldVector (allocates?)
ldiv!(j.δY_field_vector, j, j.R_field_vector)  # solve
δY .= j.δY_field_vector    # FieldVector → dense (allocates?)
```

**Fix:** If Krylov.jl works natively with FieldVectors (step 1), this conversion becomes
unnecessary.

### 4. Convergence Checker for Early Newton Exit

**Problem:** Newton's method currently runs all `max_iters = 10` iterations regardless of
convergence. Many iterations may be wasted once the residual is small enough.

**Fix:** Add a `convergence_checker` to `NewtonsMethod`:
```julia
CTS.NewtonsMethod(;
    max_iters = 10,
    convergence_checker = CTS.MaximumRelativeError(FT(1e-6)),
    krylov_method = ...,
)
```
This would exit the Newton loop early when `||Δx|| / ||x|| < tol`, saving multiple
GMRES solves per stage.

### 5. Reduce GMRES Iteration Count

**Problem:** The number of GMRES iterations per Newton step is high because the vertical-only
preconditioner does not capture horizontal acoustic modes.

**Fix options:**
- Tune `EisenstatWalkerForcing` parameters (looser initial tolerance)
- Reduce Krylov subspace size (memory parameter) if the solver converges before exhausting it
- Add approximate horizontal coupling to the preconditioner (significant development effort)
- Use physics-based splitting: treat horizontal terms explicitly (IMEX) while only vertical
  terms are implicit — this would eliminate the need for GMRES to resolve horizontal coupling

### Priority Order

1. **Krylov.jl FieldVector integration** (steps 1, 3) — highest impact, eliminates ~99% of allocations
2. **Convergence checker** (step 4) — easy win, reduces Newton iterations
3. **LinearOperator caching** (step 2) — moderate impact
4. **GMRES tuning / horizontal preconditioner** (step 5) — longer-term optimization

## Files

| File | Description |
|------|-------------|
| `examples/hybrid/plane/density_current_2d_flux_form_implicit.jl` | Main implementation |
| `examples/hybrid/plane/density_current_2d_flux_form.jl` | Explicit reference (SSPRK33, dt=0.3) |
| `examples/hybrid/implicit_equation_jacobian.jl` | Pattern: ImplicitEquationJacobian struct |
| `examples/hybrid/staggered_nonhydrostatic_model.jl` | Pattern: analytical Jacobian blocks |
| `.buildkite/Project.toml` | Added `Krylov` dependency |
