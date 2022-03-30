using ClimaCorePlots, Plots

const FT = Float64
include("baroclinic_wave_utilities.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
space = ExtrudedSpace(;
    zmax = FT(30e3),
    zelem = 10,
    hspace = CubedSphere(; radius = R, helem = 4, npoly = 4),
)
t_end = FT(60 * 60 * 24 * 10)
dt = FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.Rosenbrock23
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    hyperdiffusion_cache(ᶜlocal_geometry, ᶠlocal_geometry; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry) =
    center_initial_condition(local_geometry, Val(:ρθ); is_balanced_flow = true)

function postprocessing(sol, p, output_dir)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρθ))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρθ))"

    anim = Plots.@animate for Y in sol.u
        ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
        Plots.plot(ᶜu, level = 3, clim = (-25, 25))
    end
    Plots.mp4(anim, joinpath(output_dir, "u.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-3, 3))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)

    anim = Plots.@animate for Y in sol.u
        Plots.plot(Y.c.ρθ ./ Y.c.ρ, level = 3, clim = (225, 255))
    end
    Plots.mp4(anim, joinpath(output_dir, "theta.mp4"), fps = 5)
end
