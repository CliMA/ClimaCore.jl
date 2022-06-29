using Test

using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Fields, Operators

import ClimaCore.Utilities: half
import LinearAlgebra: norm_sqr

FT = Float32
radius = FT(1e7)
zmax = FT(1e4)
helem = npoly = 2
velem = 4

hdomain = Domains.SphereDomain(radius)
hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
htopology = Topologies.Topology2D(hmesh)
quad = Spaces.Quadratures.GLL{npoly + 1}()
hspace = Spaces.SpectralElementSpace2D(htopology, quad)

vdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(zero(FT)),
    Geometry.ZPoint{FT}(zmax);
    boundary_tags = (:bottom, :top),
)
vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
center_space = Spaces.CenterFiniteDifferenceSpace(vmesh)

#=
# TODO: Replace this with a space that includes topography.
center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
center_coords = Fields.coordinate_field(center_space)
face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
=#
face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

ᶜρe = ones(center_space)
ᶜρ = ones(center_space)
ᶜp = ones(center_space)
ᶠw = Geometry.Covariant3Vector.(ones(face_space))
BDT = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
QDT = Operators.StencilCoefs{-(1 + half), 1 + half, NTuple{4, FT}}

# this is just for testing the optimizations, so we can just make everything 1s
∂ᶜK∂ᶠw_data = similar(ᶜρ, BDT)
∂ᶜ𝔼ₜ∂ᶠ𝕄 = similar(ᶜρ, QDT)


# Allow one() to be called on vectors.
Base.one(::T) where {T <: Geometry.AxisTensor} = one(T)
Base.one(::Type{T}) where {T′, A, S, T <: Geometry.AxisTensor{T′, 1, A, S}} =
    T(axes(T), S(one(T′)))

function Base.show(io::IO, ::Type{T}) where {T <: Fields.Field}
    values_type(::Type{T}) where {V, T <: Fields.Field{V}} = V
    V = values_type(T)

    _apply!(f, ::T, match_list) where {T} = nothing # sometimes we need this...
    function _apply!(f, ::Type{T}, match_list) where {T}
        if f(T)
            push!(match_list, T)
        end
        for p in T.parameters
            _apply!(f, p, match_list)
        end
    end
    #=
        apply(::T) where {T <: Any}
    Recursively traverse type `T` and apply
    `f` to the types (and type parameters).
    Returns a list of matches where `f(T)` is true.
    =#
    apply(f, ::T) where {T} = apply(f, T)
    function apply(f, ::Type{T}) where {T}
        match_list = []
        _apply!(f, T, match_list)
        return match_list
    end

    nts = apply(x -> x <: NamedTuple, eltype(V))
    syms = unique(map(nt -> fieldnames(nt), nts))
    s = join(syms, ",")
    print(io, "Field{$s} (trunc disp)")
end

function wfact_test(∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶜK∂ᶠw_data, ᶜρe, ᶜρ, ᶜp, ᶠw)

    FT = eltype(ᶜρ)

    compose = Operators.ComposeStencils()

    ᶜdivᵥ = Operators.DivergenceF2C()
    ᶜdivᵥ_stencil = Operators.Operator2Stencil(ᶜdivᵥ)
    ᶜinterp = Operators.InterpolateF2C()
    ᶜinterp_stencil = Operators.Operator2Stencil(ᶜinterp)


    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶠinterp_stencil = Operators.Operator2Stencil(ᶠinterp)

    R_d = FT(1)
    cv_d = FT(1)

    # If we let ᶠw_data = ᶠw.components.data.:1 and ᶠw_unit = one.(ᶠw), then
    # ᶠw == ᶠw_data .* ᶠw_unit. The Jacobian blocks involve ᶠw_data, not ᶠw.
    ᶠw_data = ᶠw.components.data.:1

    # ᶜρeₜ = -ᶜdivᵥ(ᶠinterp(ᶜρe + ᶜp) * ᶠw)
    # ∂(ᶜρeₜ)/∂(ᶠw_data) =
    #     -ᶜdivᵥ_stencil(ᶠinterp(ᶜρe + ᶜp) * ᶠw_unit) -
    #     ᶜdivᵥ_stencil(ᶠw) * ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶠw_data)
    # ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶠw_data) =
    #     ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶜp) * ∂(ᶜp)/∂(ᶠw_data)
    # ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶜp) = ᶠinterp_stencil(1)
    # ∂(ᶜp)/∂(ᶠw_data) = ∂(ᶜp)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw_data)
    # ∂(ᶜp)/∂(ᶜK) = -ᶜρ * R_d / cv_d
    @. ∂ᶜK∂ᶠw_data =
        ᶜinterp(ᶠw_data) *
        norm_sqr(one(ᶜinterp(ᶠw))) *
        ᶜinterp_stencil(one(ᶠw_data))

    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
        -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρe + ᶜp) * one(ᶠw))) - compose(
            ᶜdivᵥ_stencil(ᶠw),
            compose(
                ᶠinterp_stencil(one(ᶜp)),
                -(ᶜρ * R_d / cv_d) * ∂ᶜK∂ᶠw_data,
            ),
        )

    return nothing
end

@time wfact_test(∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶜK∂ᶠw_data, ᶜρe, ᶜρ, ᶜp, ᶠw)
@time wfact_test(∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶜK∂ᶠw_data, ᶜρe, ᶜρ, ᶜp, ᶠw)

using JET
@test_opt wfact_test(∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶜK∂ᶠw_data, ᶜρe, ᶜρ, ᶜp, ᶠw)
