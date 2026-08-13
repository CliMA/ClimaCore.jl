# Baroclinic wave on the 3D sphere (Ullrich et al., 2014): a balanced jet given a
# small perturbation, which grows over ~10 days into the familiar breaking wave.
# The standard benchmark for a dry dynamical core. Total energy is the prognostic
# thermodynamic variable, and the vertical acoustic terms are treated implicitly
# (`SSP333`). Run through `driver.jl` with `TEST_NAME=sphere/baroclinic_wave_rhoe`.
using ClimaCorePlots, Plots
using ClimaCore.DataLayouts

include("baroclinic_wave_utils.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = 4)
npoly = 4
z_max = FT(30e3)
z_elem = 10
t_end = FT(60 * 60 * 24 * 10)
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = CTS.SSP333
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry) =
    sphere_center_initial_condition(local_geometry)
function postprocessing(sol, output_dir)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)
end
