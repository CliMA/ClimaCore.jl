#=
julia --project=.buildkite
using Revise; include("test/Operators/unit_reinstantiate_bc.jl")
=#

# TODO: make this unit test more low-level
using ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.CommonSpaces
using ClimaCore: Spaces, Fields, Geometry, ClimaCore, Operators
using LazyBroadcast: lazy
using Test
using Base.Broadcast: materialize

const divₕ = Operators.Divergence()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

using ClimaCore.CommonSpaces

function foo_tendency_uₕ(ᶜuₕ, zmax)
    return lazy.(
        @. (
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                Geometry.Covariant12Axis(),
                wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
            )
        )
    )
end

@testset "Reinstantiation of SpectralBroadcasted" begin
    FT = Float64
    ᶜspace = ExtrudedCubedSphereSpace(
        FT;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )
    ᶠspace = Spaces.face_space(ᶜspace)
    ᶠz = Fields.coordinate_field(ᶠspace).z
    ᶜz = Fields.coordinate_field(ᶜspace).z
    ᶜuₕ = map(z -> zero(Geometry.Covariant12Vector{eltype(z)}), ᶜz)
    zmax = Spaces.z_max(axes(ᶠz))
    vst_uₕ = foo_tendency_uₕ(ᶜuₕ, zmax)
    ᶜuₕₜ = zero(ᶜuₕ)
    @. ᶜuₕₜ += vst_uₕ
end
