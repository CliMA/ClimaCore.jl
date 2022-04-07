using ClimaCorePlots, Plots
using ClimaCore.DataLayouts

include("baroclinic_wave_utilities.jl")

sponge = false

setups = [
    HybridDriverSetup(;
        additional_cache = make_additional_cache(sponge; κ₄ = FT(2e17)),
        additional_tendency! = make_additional_tendency(sponge),
        center_initial_condition = make_center_initial_condition(:ρθ),
        face_initial_condition = make_face_initial_condition(),
        horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = 4),
        npoly = 4,
        z_max = FT(30e3),
        z_elem = 10,
        t_end = FT(60 * 60 * 24 * 10),
        dt = FT(400),
        dt_save_to_sol = FT(60 * 60 * 24),
        dt_save_to_disk = FT(0), # 0 means don't save to disk
        ode_algorithm = Rosenbrock23,
        jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact),
    ),
]

function postprocessing(sols, output_dir)
    sol = sols[1]
    @info "L₂ norm of ρθ at t = $(sol.t[1]): $(norm(sol.u[1].c.ρθ))"
    @info "L₂ norm of ρθ at t = $(sol.t[end]): $(norm(sol.u[end].c.ρθ))"

    anim = Plots.@animate for Y in sol.u
        ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)
end
