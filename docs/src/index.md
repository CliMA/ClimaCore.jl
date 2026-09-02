# ClimaCore.jl

ClimaCore.jl is the spatial discretization library of the
[CliMA](https://clima.caltech.edu/) Earth system model. It provides the
grids, fields, and differential operators on which
[ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) and
[ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl) build their governing
equations, and it runs the same model code on CPUs and on NVIDIA GPUs.

## What ClimaCore provides

  - **Spectral elements in the horizontal, staggered finite differences in the
    vertical.** Horizontal grids are continuous (CG) or discontinuous (DG)
    Galerkin spectral elements on Gauss–Lobatto–Legendre nodes. The vertical grid
    is a staggered finite-difference grid with the vertical velocity on cell
    faces and all other variables on cell centers. An unstaggered vertical grid
    always carries computational modes; the staggering suppresses them, which
    matters most at the large aspect ratios of atmospheric grids, where the
    horizontal spacing exceeds the vertical spacing by orders of magnitude
    [Thuburn05n](@cite). See [Spectral elements: CG and DG](explanation/discretizations.md).
  - **One tendency for CG and DG.** The discretization is a property of the
    space. `Operators.tendency_completion` completes an element-local tendency
    by direct stiffness summation on CG spaces and by interface numerical fluxes
    on DG spaces, so model code is written once
    ([tutorial](tutorials/cg_dg_switch.md)).
  - **Portable performance.** Broadcast expressions over fields compile to CPU
    loops or CUDA kernels from one source. The CliMA atmosphere built on
    ClimaCore runs at more than one simulated year per day at 25–50 km horizontal
    resolution on a few dozen GPUs, with weak-scaling efficiency above 92% on
    GPUs and above 98% on CPUs, on supercomputers and on cloud GPU instances
    [Yatunin2026](@cite).
  - **Differentiable in forward mode.** ForwardDiff dual numbers propagate
    through fields, broadcasts, and operators, and grid metric terms are
    computed by automatic differentiation of the mesh coordinates. Reverse-mode
    differentiation is planned.
  - **Geometries from a single column to the cubed sphere.** Columns, x–z slices,
    Cartesian boxes and planes, and the equiangular cubed sphere, with
    terrain-following vertical coordinates. The same operators serve
    large-eddy-simulation boxes and global climate simulations
    [Sridhar22a, Yatunin2026](@cite).
  - **Time stepping by [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl).**
    `Field`s and `FieldVector`s are the state vectors of its explicit and
    implicit-explicit Runge–Kutta methods.

## Where to start

| You want to                                                | Go to                                                                                        |
|:---------------------------------------------------------- |:-------------------------------------------------------------------------------------------- |
| Learn the vocabulary: domain, mesh, space, field, operator | [Concepts and design](getting_started/concepts.md)                                           |
| Learn the library step by step                             | [Tutorial: Fields and operators](tutorials/fields_and_operators.md)                          |
| Write a model that runs on CG or DG                        | [Tutorial: CG and DG with one tendency](tutorials/cg_dg_switch.md)                           |
| Get a specific task done                                   | The how-to guides, e.g. [Run the examples](howto/run_examples.md)                            |
| Understand the numerics                                    | The explanation pages, starting with [Mathematical framework](explanation/math_framework.md) |
| Look up a function or type                                 | [API overview](reference/index.md)                                                           |

## Quick example

A one-dimensional column, a field on it, and a vertical derivative.

```@example quickstart
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Domains, Meshes, Spaces, Fields, Geometry, Operators

FT = Float64

# Build a 1D column: interval domain -> mesh -> finite-difference space
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(2π),
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain; nelems = 128)
space = Spaces.CenterFiniteDifferenceSpace(ClimaComms.device(), mesh)

# Define a field over the space and differentiate it with a composed operator
z = Fields.coordinate_field(space).z
θ = sin.(z)
grad = Operators.GradientC2F(
    bottom = Operators.SetValue(FT(0)),
    top = Operators.SetValue(FT(0)),
)
∂θ = @. Geometry.WVector(grad(θ))   # face-valued vertical gradient (≈ cos(z))
```

## Related packages

  - [ClimaComms.jl](https://clima.github.io/ClimaComms.jl/stable/) selects the
    compute device (CPU or GPU) and the communication context (single process or
    MPI) from environment variables.
  - [ClimaTimeSteppers.jl](https://clima.github.io/ClimaTimeSteppers.jl/stable/)
    advances ClimaCore fields in time.
  - [ClimaAtmos.jl](https://clima.github.io/ClimaAtmos.jl/stable/) documents the
    atmosphere model built on these operators, including its governing equations
    and their semi-discrete form.
  - The shared [CliMA developer guides](https://github.com/CliMA/DeveloperGuides),
    vendored at [`docs/dev-guides/`](https://github.com/CliMA/ClimaCore.jl/tree/main/docs/dev-guides), hold the engineering standards contributors
    follow.
