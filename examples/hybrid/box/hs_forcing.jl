using ClimaCorePlots, Plots
import ClimaCore: level

const FT = Float64
include("hs_forcing_utilities.jl")

const sponge = false

#temp(ϕ, z) = T_init + rand(FT) * FT(0.1) * (z < 5000)
#pres(ϕ, z) = p_0 * exp(-z / scale_height)

# Variables required for driver.jl (modify as needed)
space =
    ExtrudedSpace(;
    zmax = FT(30e3),
    zelem = 10,
    hspace = PeriodicRectangle(; xmax=2.4e7, ymax=2.4e7, xelem=10, yelem=4, npoly=4)
)
t_end = FT(60 * 60 * 24)
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
dt_save_to_disk = FT(60 * 60 * 24 * 10) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry, ᶠlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
    held_suarez_cache(ᶜlocal_geometry),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
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
