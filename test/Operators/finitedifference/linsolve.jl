using Test
using ClimaComms

import ClimaCore
# To avoid JET failures in the error message
ClimaCore.Operators.allow_mismatched_fd_spaces() = true

using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Fields

FT = Float32
radius = FT(1e7)
zmax = FT(1e4)
helem = npoly = 2
velem = 4

hdomain = Domains.SphereDomain(radius)
hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
htopology =
    Topologies.DistributedTopology2D(ClimaComms.SingletonCommsContext(), hmesh)
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

function _linsolve!(x, A, b, update_matrix = false; kwargs...)

    FT = Spaces.undertype(axes(x.c))

    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ) = A

    is_momentum_var(symbol) = symbol in (:uₕ, :ρuₕ, :w, :ρw)

    # Compute Schur complement
    # Compute xᶠ𝕄
    xᶜρ = x.c.ρ
    bᶜρ = b.c.ρ
    ᶜ𝕄_name = Base.filter(is_momentum_var, propertynames(x.c))[1]
    xᶜ𝕄 = getproperty(x.c, ᶜ𝕄_name)
    bᶜ𝕄 = getproperty(b.c, ᶜ𝕄_name)
    ᶠ𝕄_name = Base.filter(is_momentum_var, propertynames(x.f))[1]
    xᶠ𝕄 = getproperty(x.f, ᶠ𝕄_name).components.data.:1
    bᶠ𝕄 = getproperty(b.f, ᶠ𝕄_name).components.data.:1

    @. xᶠ𝕄 = bᶠ𝕄 + (apply(∂ᶠ𝕄ₜ∂ᶜρ, bᶜρ))

    # Compute remaining components of x
    @. xᶜρ = -bᶜρ + apply(∂ᶜρₜ∂ᶠ𝕄, xᶠ𝕄)
    return nothing
end

import ClimaCore
include(
    joinpath(pkgdir(ClimaCore), "examples", "hybrid", "schur_complement_W.jl"),
)
jacobi_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact);
use_transform = false;

Y = Fields.FieldVector(
    c = map(
        coord -> (
            ρ = Float32(0),
            ρe = Float32(0),
            uₕ = Geometry.Covariant12Vector(Float32(0), Float32(0)),
        ),
        Fields.coordinate_field(center_space),
    ),
    f = map(
        _ -> (; w = Geometry.Covariant3Vector(Float32(0))),
        Fields.coordinate_field(face_space),
    ),
)

b = similar(Y)
W = SchurComplementW(Y, use_transform, jacobi_flags)

using JET
using Test
@time _linsolve!(Y, W, b)
@time _linsolve!(Y, W, b)

@testset "JET test for `apply` in linsolve! kernel" begin
    @test_opt _linsolve!(Y, W, b)
end

ClimaCore.Operators.allow_mismatched_fd_spaces() = false
