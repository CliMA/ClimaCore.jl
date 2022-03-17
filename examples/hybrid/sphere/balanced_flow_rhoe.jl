using Test
using ClimaCorePlots, Plots

include("baroclinic_wave_utilities.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
space = ExtrudedSpace(;
    zmax = FT(30e3),
    zelem = 10,
    hspace = CubedSphere(; radius = R, helem = 4, npoly = 4),
)
t_end = FT(60 * 60)
dt = FT(5)
dt_save_to_sol = FT(50)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.SSPRK33

initial_condition(local_geometry) =
    initial_condition_ρe(local_geometry; is_balanced_flow = true)
initial_condition_velocity(local_geometry) =
    initial_condition_velocity(local_geometry; is_balanced_flow = true)

remaining_cache_values(Y, dt) = merge(
    baroclinic_wave_cache_values(Y, dt),
    final_adjustments_cache_values(Y, dt; use_rayleigh_sponge = sponge),
)

function remaining_tendency!(dY, Y, p, t)
    dY .= zero(eltype(dY))
    baroclinic_wave_ρe_remaining_tendency!(dY, Y, p, t; κ₄ = 2.0e17)
    final_adjustments!(
        dY,
        Y,
        p,
        t;
        use_flux_correction = false,
        use_rayleigh_sponge = sponge,
    )
    return dY
end

function postprocessing(sol, p, output_dir)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρe))"

    u_end = Geometry.UVVector.(sol.u[end].uₕ).components.data.:1
    Plots.png(Plots.plot(u_end, level = 3), joinpath(output_dir, "u_end.png"))

    w_end = Geometry.WVector.(sol.u[end].w).components.data.:1
    Plots.png(
        Plots.plot(w_end, level = 3 + half, clim = (-4, 4)),
        joinpath(output_dir, "w_end.png"),
    )

    Δu_end = Geometry.UVVector.(sol.u[end].uₕ .- sol.u[1].uₕ).components.data.:1
    Plots.png(
        Plots.plot(Δu_end, level = 3, clim = (-1, 1)),
        joinpath(output_dir, "Δu_end.png"),
    )

    @test sol.u[end].Yc.ρ ≈ sol.u[1].Yc.ρ rtol = 5e-2
    @test sol.u[end].Yc.ρe ≈ sol.u[1].Yc.ρe rtol = 5e-2
    @test sol.u[end].uₕ ≈ sol.u[1].uₕ rtol = 5e-2
end
