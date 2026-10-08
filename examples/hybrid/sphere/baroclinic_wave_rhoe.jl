# Baroclinic wave on the 3D sphere (Ullrich et al., 2014), with total energy as
# the thermodynamic variable and implicit vertical acoustics (`SSP333`). Run
# through `driver.jl` with `TEST_NAME=sphere/baroclinic_wave_rhoe`.
#
# `DISCRETIZATION=DG` uses the flux form of `dg_tendency.jl` (`DG_FLUX` selects
# the fluxes); adding `MOMENTUM_FORM=vector_invariant` uses
# `dg_vector_invariant_tendency.jl` instead. `T_END`, `DT` and `ODE_ALGORITHM`
# override the run length, timestep and scheme (an explicit scheme such as
# `SSP33ShuOsher` needs dt ≈ 5 s).
using Test
using Plots
using ClimaCore.DataLayouts

include("baroclinic_wave_utils.jl")

const sponge = false

# Variables required for driver.jl (modify as needed)
h_elem = parse(Int, get(ENV, "H_ELEM", "4"))
horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = h_elem)
npoly = 4
z_max = FT(30e3)
z_elem = parse(Int, get(ENV, "Z_ELEM", "10"))
# DG: dt = 100 (Float32 at this resolution; dt = 400 diverges within two days).
# A spurious mode grows in both hemispheres and over the poles from about day 5
# (the CG run stays clean through day 10), so the DG run stops at two days.
t_end = discretization isa Grids.DG ? FT(60 * 60 * 24 * 2) : FT(60 * 60 * 24 * 10)
dt = discretization isa Grids.DG ? FT(100) : FT(400)
t_end = parse(FT, get(ENV, "T_END", string(t_end)))
dt = parse(FT, get(ENV, "DT", string(dt)))
dt_save_to_sol = min(FT(60 * 60 * 24), t_end)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = getproperty(CTS, Symbol(get(ENV, "ODE_ALGORITHM", "SSP333")))
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

# Hyperdiffusion is CG-only: `vector_laplacian` has no DG method.
use_hyperdiffusion(space) = Spaces.is_continuous(space)

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    use_hyperdiffusion(axes(ᶜlocal_geometry)) ?
    hyperdiffusion_cache(ᶜlocal_geometry; κ₄ = FT(2e17)) : (;),
    sponge ? rayleigh_sponge_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) : (;),
)
function additional_tendency!(Yₜ, Y, p, t)
    use_hyperdiffusion(axes(Y.c)) && hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry) =
    sphere_center_initial_condition(local_geometry)
function postprocessing(sol, output_dir)
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"

    # Conservation and growth of the perturbation (Float32: drift ≲ 5e-6;
    # max|v| 0.76 → 6.6 for CG over ten days, → 7-10 for DG over two).
    @test abs(sum(sol.u[end].c.ρ) - sum(sol.u[1].c.ρ)) / sum(sol.u[1].c.ρ) < 1e-4
    @test abs(sum(sol.u[end].c.ρe) - sum(sol.u[1].c.ρe)) / sum(sol.u[1].c.ρe) < 1e-4
    v_init = maximum(abs, center_velocity(sol.u[1].c).components.data.:2)
    v_end = maximum(abs, center_velocity(sol.u[end].c).components.data.:2)
    @test v_end > 4 * v_init > 0

    anim = Plots.@animate for Y in sol.u
        ᶜv = center_velocity(Y.c).components.data.:2
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)
    temperature_animation(sol, output_dir, center_temperature)
end
