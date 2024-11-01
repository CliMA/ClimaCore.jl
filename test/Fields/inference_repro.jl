# Initially from ClimaLand.jl soiltest.jl phase change source term test
# To reproduce, run this script on GPU
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Fields, Domains, Geometry, Meshes, Spaces

struct LandParameters{FT}
    ρ_cloud_ice::FT
end
Base.broadcastable(x::LandParameters) = tuple(x)

struct vanGenuchten{FT}
    α::FT
end
Base.broadcastable(x::vanGenuchten) = tuple(x)

function phase_change_source(
    θ_l::FT,
    hydrology_cm::C,
    earth_param_set::EP,
) where {FT, EP, C}
    return nothing
end

function make_space(
    ::Type{FT};
    zlim = (FT(-1), FT(0)),
    nelements = 200,
) where {FT}
    boundary_names = (:bottom, :top)
    column = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = boundary_names,
    )
    mesh = Meshes.IntervalMesh(column; nelems = nelements)
    subsurface_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    return subsurface_space
end

function call_func(θ_l, hydrology_cm, earth_param_set)
    # function call_func(hydrology_cm, earth_param_set)
    # This fails with dynamic function invocation when `LandParameters`
    # and `vanGenuchten` both use `tuple` for broadcasting,
    # This passes when `Ref` is used for either `LandParameters` or `vanGenuchten` broadcasting
    @. phase_change_source(θ_l, hydrology_cm, earth_param_set)

    # These don't fail on GPU
    # @. phase_change_source(hydrology_cm, earth_param_set)
    # @. phase_change_source(θ_l, earth_param_set)
    # @. phase_change_source(θ_l, hydrology_cm)
    return nothing
end
function main(::Type{FT}) where {FT}
    earth_param_set = LandParameters{FT}(FT(0))
    hydrology_cm = vanGenuchten{FT}(FT(2.6))

    space_3d = make_space(FT; zlim = (FT(-1), FT(0)), nelements = 200)
    θ_l = Fields.ones(space_3d)

    call_func(θ_l, hydrology_cm, earth_param_set)
    return nothing
end

using Test
@testset "GPU inference failure" begin
    main(Float64)
end
