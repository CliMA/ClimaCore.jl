# To be launched with
# julia --threads=8 --project=test test/Operators/finitedifference/threading_allocs.jl
# or interactively with
# julia --threads=8 --project=test
# using Revise; include("test/Operators/finitedifference/threading_allocs.jl")

using Test
import ClimaCore

# ClimaCore.enable_threading() = true  # this has a big impact, false is faster
ClimaCore.enable_threading() = false # this has a big impact, false is faster

using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Operators
import ClimaCore.Utilities: half
import LinearAlgebra: norm_sqr

function get_vars(::Type{FT}) where {FT}
    radius = FT(1e7)
    zmax = FT(1e4)
    helem = npoly = 2
    velem = 4

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(hmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)

    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(pi);
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems = 10)
    z_topology = Topologies.IntervalTopology(z_mesh)
    z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    cs = Spaces.ExtrudedFiniteDifferenceSpace(hspace, z_space)
    fs = Spaces.FaceExtrudedFiniteDifferenceSpace(cs)

    ᶜρe = ones(cs)
    ᶜρ = ones(cs)
    ᶜp = ones(cs)
    ᶠw = Geometry.Covariant3Vector.(ones(fs))
    BDT = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
    QDT = Operators.StencilCoefs{-(1 + half), 1 + half, NTuple{4, FT}}
    # this is just for testing the optimizations, so we can just make everything 1s
    ϕ = similar(ᶜρ, BDT)
    ψ = similar(ᶜρ, QDT)
    return (; ϕ, ψ, ᶜρe, ᶜρ, ᶜp, ᶠw)
end

# Allow one() to be called on vectors.
Base.one(::T) where {T <: Geometry.AxisTensor} = one(T)
Base.one(::Type{T}) where {T′, A, S, T <: Geometry.AxisTensor{T′, 1, A, S}} =
    T(axes(T), S(one(T′)))

function threading_allocs_test(ψ, ϕ, ᶜρe, ᶜρ, ᶜp, ᶠw, obj)
    FT = eltype(ᶜρ)
    compose = Operators.ComposeStencils()
    ᶜdivᵥ = Operators.DivergenceF2C()
    ᶜdivᵥ_stencil = Operators.Operator2Stencil(ᶜdivᵥ)
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶠinterp_stencil = Operators.Operator2Stencil(ᶠinterp)
    R_d = FT(1)
    cv_d = FT(1)
    cspace = axes(ᶜρe)
    Fields.bycolumn(cspace) do colidx
        @. ψ[colidx] =
            -(ᶜdivᵥ_stencil(
                ᶠinterp(ᶜρe[colidx] + ᶜp[colidx]) * one(ᶠw[colidx]),
            )) - compose(
                ᶜdivᵥ_stencil(ᶠw[colidx]),
                compose(
                    ᶠinterp_stencil(one(ᶜp[colidx])),
                    -(ᶜρ[colidx] * R_d / cv_d) * ϕ[colidx],
                ),
            ) + myfun(obj)
        # nothing # try this
    end

    return nothing
end
Base.@kwdef struct Foo{FT}
    a::FT
    b::FT
end
myfun(f::Foo) = f.a + f.b
Base.broadcastable(a::Foo) = Ref(a)
FT = Float64
obj = Foo{FT}(; a = 1, b = 2)
(; ϕ, ψ, ᶜρe, ᶜρ, ᶜp, ᶠw) = get_vars(FT)
@time threading_allocs_test(ψ, ϕ, ᶜρe, ᶜρ, ᶜp, ᶠw, obj)
@time threading_allocs_test(ψ, ϕ, ᶜρe, ᶜρ, ᶜp, ᶠw, obj)

@testset "Test allocations" begin
    @time threading_allocs_test(ψ, ϕ, ᶜρe, ᶜρ, ᶜp, ᶠw, obj)
end
