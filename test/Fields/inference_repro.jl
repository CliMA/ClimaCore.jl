# Initially from ClimaLand.jl soiltest.jl phase change source term test
# To reproduce, run this script on GPU
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Fields, Domains, Geometry, Meshes, Spaces

Base.@kwdef struct LandParameters{FT}
    ρ_cloud_ice::FT
    ρ_cloud_liq::FT
    LH_f0::FT
    T_freeze::FT
    grav::FT
end
Base.broadcastable(x::LandParameters) = tuple(x)

ρ_cloud_ice(x) = x.ρ_cloud_ice
ρ_cloud_liq(x) = x.ρ_cloud_liq
LH_f0(x) = x.LH_f0
T_freeze(x) = x.T_freeze
grav(x) = x.grav

struct vanGenuchten{FT}
    "The inverse of the air entry potential (1/m)"
    α::FT
    "The van Genuchten pore-size distribution index (unitless)"
    n::FT
    "The van Genuchten parameter m = 1 - 1/n (unitless)"
    m::FT
    "A derived parameter: the critical saturation at which capillary flow no longer replenishes the surface"
    S_c::FT
    function vanGenuchten{FT}(; α::FT, n::FT) where {FT}
        m = 1 - 1 / n
        S_c = (1 + ((n - 1) / n)^(1 - 2 * n))^(-m)
        return new{FT}(α, n, m, S_c)
    end
end
Base.broadcastable(x::vanGenuchten) = tuple(x)

function matric_potential(cm::vanGenuchten{FT}, S::FT) where {FT}
    (; α, m, n) = cm
    ψ = -((S^(-FT(1) / m) - FT(1)) * α^(-n))^(FT(1) / n)
    return ψ
end

function phase_change_source(
    θ_l::FT,
    # θ_i::FT,
    # T::FT,
    # τ::FT,
    # ν::FT,
    # θ_r::FT,
    hydrology_cm::C,
    earth_param_set::EP,
) where {FT, EP, C}
    _ρ_i = FT(ρ_cloud_ice(earth_param_set))
    # ψw0 = matric_potential(hydrology_cm, _ρ_i)
    return (θ_l - _ρ_i)
    # return (θ_l - _ρ_i) / τ
end

function make_space(::Type{FT}; zlim = (FT(-1), FT(0)), nelements = 200) where {FT}
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

function call_func(θ_l, θ_i, T, tau, ν, θ_r, hydrology_cm, earth_param_set)
    # This fails with dynamic function invocation when `LandParameters`
    # and `vanGenuchten` both use `tuple` for broadcasting, and it
    # passes when `Ref` is used for either `LandParameters` or `vanGenuchten` broadcasting
    # @. -phase_change_source(θ_l, θ_i, T, tau, ν, θ_r, hydrology_cm, earth_param_set)
    # @. -phase_change_source(θ_l, tau, hydrology_cm, earth_param_set)
    @. -phase_change_source(θ_l, hydrology_cm, earth_param_set)
    return nothing
end
function main(::Type{FT}) where {FT}
    Np = length(fieldtypes(LandParameters))
    earth_param_set = LandParameters{FT}(zeros(Np)...)
    ν = FT(0.495)
    θ_r = FT(0.1)
    hydrology_cm = vanGenuchten{FT}(; α = FT(2.6), n = FT(2.0))

    space_3d = make_space(FT; zlim = (FT(-1), FT(0)), nelements = 200)

    θ_l = Fields.ones(space_3d)
    θ_i = Fields.ones(space_3d)
    T = Fields.ones(space_3d)
    κ = Fields.ones(space_3d)
    tau = Fields.ones(space_3d)

    call_func(θ_l, θ_i, T, tau, ν, θ_r, hydrology_cm, earth_param_set)

    return nothing
end

using Test
@testset "GPU inference failure" begin
    if ClimaComms.device() isa ClimaComms.CUDADevice
        @test_broken try
            main(Float64)
            true
        catch e
            @assert occursin("GPUCompiler.InvalidIRError", string(e))
            @assert occursin("dynamic function invocation", e.errors[1][1])
            false
        end
    else
        main(Float64)
    end
end
