# # Tutorial: CG and DG with one tendency
#
# ClimaCore's spectral-element grids come in two variants, continuous Galerkin
# (CG) and discontinuous Galerkin (DG). They share the element-local operators
# and differ in how neighboring elements are coupled: CG by direct stiffness
# summation (DSS), DG by interface numerical fluxes
# ([Yatunin2026](@cite), [Souza2023](@cite)). This tutorial writes one
# shallow-water tendency and evaluates it on both.
#
# It covers:
# 1. constructing a CG and a DG space over the same mesh;
# 2. writing an element-local weak-form tendency;
# 3. completing it across element interfaces with `Operators.tendency_completion`.

using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
using ClimaCore:
    Domains, Meshes, Topologies, Quadratures, Grids, Spaces, Fields, Operators
import ClimaCore.Geometry
import ClimaCore.Geometry: ⊗

# ## 1. The mesh, and a CG and a DG space on it
#
# A doubly periodic square, 8 × 8 elements, fourth-order polynomials on
# Gauss–Lobatto–Legendre nodes.

FT = Float64
domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(-FT(2π)),
        Geometry.XPoint(FT(2π));
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(-FT(2π)),
        Geometry.YPoint(FT(2π));
        periodic = true,
    ),
)
mesh = Meshes.RectilinearMesh(domain, 8, 8)
context = ClimaComms.context()
topology = Topologies.Topology2D(context, mesh)
quad = Quadratures.GLL{4}()

# The discretization is a keyword of the space constructor and a property of
# the resulting space. `Grids.CG()` is the default.

cg_space = Spaces.SpectralElementSpace2D(topology, quad; discretization = Grids.CG())
dg_space = Spaces.SpectralElementSpace2D(topology, quad; discretization = Grids.DG())
Spaces.discretization(cg_space), Spaces.discretization(dg_space)

# ## 2. A shallow-water state and its physical flux
#
# The state is a `NamedTuple` field of depth `ρ`, momentum `ρu`, and a tracer
# `ρθ`; the initial condition is a Bickley jet with a vortical perturbation.

params = (; ϵ = FT(0.1), l = FT(0.5), k = FT(0.5), ρ₀ = FT(1), c = FT(2), g = FT(10))

function bickley_jet(coord, p)
    x, y = coord.x, coord.y
    U₁ = p.c / cosh(y)^2
    gaussian = exp(-(y + p.l / 10)^2 / (2 * p.l^2))
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)
    u = Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    return (; ρ = p.ρ₀, ρu = p.ρ₀ * u, ρθ = p.ρ₀ * sin(p.k * y))
end

# The physical flux `F(y)` of the conservation law `∂ₜy + ∇⋅F(y) = 0`, in the
# local orthonormal (U, V) basis, and an upper bound on its signal speed. The
# same `sw_flux` feeds the weak volume term on both spaces and the interface
# flux on the DG space.

function sw_flux(y, p)
    u = y.ρu / y.ρ
    return (;
        ρ = y.ρu,
        ρu = (y.ρu ⊗ u) + (p.g * y.ρ^2 / 2) * LinearAlgebra.I,
        ρθ = y.ρθ * u,
    )
end
sw_wavespeed(y, p) = sqrt(p.g * y.ρ) + norm(y.ρu / y.ρ)

# ## 3. One tendency, completed by the space
#
# The tendency is the element-local weak divergence of the flux, completed
# across element interfaces by the completion object. `tendency_completion`
# returns a `DSSCompletion` on the CG space and a `NumericalFluxCompletion`
# on the DG space; the `numflux` keyword is required on DG and ignored on CG,
# so the model passes it unconditionally.

function shallow_water_rhs!(dydt, y, (p, completion), t)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    rp = Ref(p)
    @. dydt = -wdiv(sw_flux(y, rp))
    Operators.complete_tendency!(completion, dydt, y, p)
    return dydt
end

numflux = Operators.RusanovNumericalFlux(sw_flux, sw_wavespeed)

function tendency(space)
    y = bickley_jet.(Fields.coordinate_field(space), Ref(params))
    dydt = similar(y)
    completion = Operators.tendency_completion(dydt; numflux)
    shallow_water_rhs!(dydt, y, (params, completion), FT(0))
    return completion, dydt
end

cg_completion, cg_dydt = tendency(cg_space)
dg_completion, dg_dydt = tendency(dg_space)
typeof(cg_completion).name.name, typeof(dg_completion).name.name

# Both completions conserve mass: the weak divergence telescopes over the
# periodic domain after DSS, and the antisymmetric interface flux does the same
# on the DG grid, so the integral of the depth tendency vanishes to round-off.

sum(cg_dydt.ρ), sum(dg_dydt.ρ)

# The two tendencies agree where the field is smooth and differ near element
# boundaries, where the Rusanov penalty acts on the inter-element jump. The
# difference is small for this well-resolved initial state:

maximum(abs, parent(cg_dydt.ρθ) .- parent(dg_dydt.ρθ)) /
maximum(abs, parent(cg_dydt.ρθ))

# From here, a model advances `y` with ClimaTimeSteppers; the example
# [`examples/plane/bickleyjet.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/plane/bickleyjet.jl) and the integration test
# [`test/Integration/smoke_bickley_jet_cg_dg.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/test/Integration/smoke_bickley_jet_cg_dg.jl) do so on both spaces.
