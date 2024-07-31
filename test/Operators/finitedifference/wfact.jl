using Test
using ClimaComms

import ClimaCore
# To avoid JET failures in the error message
ClimaCore.Operators.allow_mismatched_fd_spaces() = true

using ClimaCore:
    Geometry,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Fields,
    Operators,
    Quadratures

import ClimaCore.Utilities: half
import LinearAlgebra: norm_sqr

FT = Float32
radius = FT(1e7)
zmax = FT(1e4)
helem = npoly = 2
velem = 4

hdomain = Domains.SphereDomain(radius)
hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
htopology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), hmesh)
quad = Quadratures.GLL{npoly + 1}()
hspace = Spaces.SpectralElementSpace2D(htopology, quad)

vdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(zero(FT)),
    Geometry.ZPoint{FT}(zmax);
    boundary_names = (:bottom, :top),
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
            compose(ᶠinterp_stencil(one(ᶜp)), -(ᶜρ * R_d / cv_d) * ∂ᶜK∂ᶠw_data),
        )

    return nothing
end

@time wfact_test(∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶜK∂ᶠw_data, ᶜρe, ᶜρ, ᶜp, ᶠw)
@time wfact_test(∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶜK∂ᶠw_data, ᶜρe, ᶜρ, ᶜp, ᶠw)

using JET
@testset "JET test for `compose` in wfact! kernel" begin
    @test_opt wfact_test(∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶜK∂ᶠw_data, ᶜρe, ᶜρ, ᶜp, ᶠw)
end

ClimaCore.Operators.allow_mismatched_fd_spaces() = false
