using ClimaCorePlots, Plots
import ClimaCore: level

const FT = Float64
include("hs_forcing_utilities.jl")

const sponge = false
const domain_width = FT(1.92e7)

temp(x, y, z) = 
    T_init + lapse_rate*z + 0.1*(sin(120*pi*x/domain_width)+2*sin(121*pi*y/domain_width)) * (z < 5000)

# Variables required for driver.jl (modify as needed)
space =
    ExtrudedSpace(;
    zmax = FT(30e3),
    zelem = 10,
    hspace = PeriodicRectangle(; xmax=domain_width, ymax=domain_width, xelem=8, yelem=8, npoly=4)
)

t_end = FT(60 * 60 * 24 * 1200)
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
#dt_save_to_disk = FT(60 * 60 * 24 * 10) # 0 means don't save to disk
dt_save_to_disk = FT(0) # 0 means don't save to disk
#ode_algorithm = OrdinaryDiffEq.Rosenbrock23
ode_algorithm = OrdinaryDiffEq.SSPRK33
if flux_form
  jacobian_flags = (;)
else
  jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)
end

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry, ᶠlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
    held_suarez_cache(ᶜlocal_geometry),
)
function additional_tendency!(Yₜ, Y, p, t)
  if flux_form
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_conservative_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
  else
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
  end
end

center_initial_condition(local_geometry) =
    center_initial_condition(local_geometry, Val(:ρθ))

function postprocessing(sol, p, output_dir)
    @info "L₂ norm of ρθ at t = $(sol.t[1]): $(norm(sol.u[1].c.ρθ))"
    @info "L₂ norm of ρθ at t = $(sol.t[end]): $(norm(sol.u[end].c.ρθ))"

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(level(ᶜv, 3), clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)
end
