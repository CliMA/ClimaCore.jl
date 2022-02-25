using Test
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
    center_initial_condition(local_geometry, Val(:ρe), true)
additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
)
function additional_remaining_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
end

function postprocessing(sol, p, path)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"

    ᶜu_end = Geometry.UVVector.(sol.u[end].c.uₕ).components.data.:1
    Plots.png(Plots.plot(ᶜu_end, level = 3), joinpath(path, "u_end.png"))

    ᶜw_end = Geometry.WVector.(sol.u[end].f.w).components.data.:1
    Plots.png(
        Plots.plot(ᶜw_end, level = 3 + half, clim = (-4, 4)),
        joinpath(path, "w_end.png"),
    )

    ᶜu_start = Geometry.UVVector.(sol.u[1].c.uₕ).components.data.:1
    Plots.png(
        Plots.plot(ᶜu_end .- ᶜu_start, level = 3, clim = (-1, 1)),
        joinpath(path, "Δu_end.png"),
    )

    @test sol.u[end].c.ρ ≈ sol.u[1].c.ρ rtol = 5e-2
    @test sol.u[end].c.ρe ≈ sol.u[1].c.ρe rtol = 5e-2
    @test sol.u[end].c.uₕ ≈ sol.u[1].c.uₕ rtol = 5e-2
end
