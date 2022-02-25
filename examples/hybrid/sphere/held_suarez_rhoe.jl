using ClimaCorePlots, Plots
using DiffEqCallbacks
using JLD2

include("baroclinic_wave_utilities.jl")

const sponge = false

space = ExtrudedSpace(;
    zmax = FT(30e3),
    zelem = 10,
    hspace = CubedSphere(; radius = R, pelem = 4, npoly = 4),
)
tend = FT(60 * 60 * 24 * 10)
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂p∂K, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

center_initial_condition(local_geometry) =
    center_initial_condition(local_geometry, Val(:ρe), false)
additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
    held_suarez_cache(ᶜlocal_geometry),
)
function additional_remaining_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
end

function postprocessing(sol, p, path)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(path, "v.mp4"), fps = 5)
end
