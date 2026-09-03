# Balanced flow on the 3D sphere: the baroclinic-wave initial state with its
# perturbation switched off, so the flow is in hydrostatic and gradient-wind
# balance and should not evolve. The run asserts that density, total energy, and
# horizontal velocity are unchanged after an hour, which makes this a check on
# the balance of the discretization itself. Run through `driver.jl` with
# `TEST_NAME=sphere/balanced_flow_rhoe`.
#
# The state is balanced in the continuous equations, not in the discrete ones,
# so the residual imbalance launches gravity waves. They are the diagnostic: a
# discretization that holds the balance leaves them bounded and small, one that
# does not lets them grow.
using Test
import ClimaTimeSteppers as CTS
using Plots
using ClimaCore.DataLayouts

include("baroclinic_wave_utils.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = 4)
npoly = 4
z_max = FT(30e3)
z_elem = 10
t_end = FT(60 * 60)
dt = FT(5)
dt_save_to_sol = FT(50)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = CTS.SSP33ShuOsher
# We use :exact for the energy-momentum Jacobian block because balanced flow
# requires high accuracy in the implicit solve to maintain stationarity,
# unlike the baroclinic wave which uses the cheaper :no_∂ᶜp∂ᶜK approximation.
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry) =
    sphere_center_initial_condition(local_geometry; is_balanced_flow = true)

function postprocessing(sol, output_dir)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"

    ᶜu_end = Geometry.UVVector.(sol.u[end].c.uₕ).components.data.:1
    Plots.png(Plots.plot(ᶜu_end, level = 3), joinpath(output_dir, "u_end.png"))

    physical_w(Y) = Geometry.WVector.(Y.f.w).components.data.:1
    ᶠw_end = physical_w(sol.u[end])
    Plots.png(
        Plots.plot(ᶠw_end, level = 3 + half, clim = (-10, 10)),
        joinpath(output_dir, "w_end.png"),
    )

    ᶜu_start = Geometry.UVVector.(sol.u[1].c.uₕ).components.data.:1
    Plots.png(
        Plots.plot(ᶜu_end .- ᶜu_start, level = 3, clim = (-1, 1)),
        joinpath(output_dir, "Δu_end.png"),
    )

    Y_start, Y_end = sol.u[1], sol.u[end]

    # Mass and total energy must be conserved (measured drift: 5e-7 and 4e-7,
    # which is roundoff at Float32 for a sum over this many points).
    @test abs(sum(Y_end.c.ρ) - sum(Y_start.c.ρ)) / sum(Y_start.c.ρ) < 1e-5
    @test abs(sum(Y_end.c.ρe) - sum(Y_start.c.ρe)) / abs(sum(Y_start.c.ρe)) < 1e-5

    # Nothing may evolve much: the state after an hour is the state it started
    # from (measured relative L₂ drift: 0.006, 0.016 and 0.017).
    @test Y_end.c.ρ ≈ Y_start.c.ρ rtol = 2e-2
    @test Y_end.c.ρe ≈ Y_start.c.ρe rtol = 4e-2
    @test Y_end.c.uₕ ≈ Y_start.c.uₕ rtol = 4e-2

    # The state is balanced analytically but not on the discrete grid, and with
    # no sponge the leftover imbalance radiates gravity waves. They oscillate at
    # a bounded amplitude instead of growing, which is the signature that the
    # balance itself holds: the zonal jet is untouched to under 1% of its 28 m/s
    # peak (measured max|Δu| = 0.21 m/s), the meridional wind stays two orders
    # of magnitude below it (measured max|v| = 0.50 m/s, from zero), and the
    # vertical velocity saturates in the stratosphere rather than running away
    # (measured max|w| = 8.7 m/s, reached within the first 400 s and flat
    # after).
    ᶜv_end = Geometry.UVVector.(Y_end.c.uₕ).components.data.:2
    @test maximum(abs, ᶜu_end .- ᶜu_start) < 0.01 * maximum(abs, ᶜu_start)
    @test maximum(abs, ᶜv_end) < 2
    @test maximum(maximum(abs, physical_w(Y)) for Y in sol.u) < 15
end
