# Baroclinic wave on the 3D sphere (Ullrich et al., 2014): a balanced jet given
# a small perturbation, which grows over ~10 days into the familiar breaking
# wave. The standard benchmark for a dry dynamical core. Total energy is the
# prognostic thermodynamic variable, and the vertical acoustic terms are treated
# implicitly (`SSP333`). Run through `driver.jl` with
# `TEST_NAME=sphere/baroclinic_wave_rhoe`.
#
# `DISCRETIZATION=DG` runs the same case on a discontinuous horizontal space,
# where the momentum equation takes the flux form of `dg_tendency.jl` and the
# element coupling is an interface numerical flux instead of a DSS. `DG_FLUX`
# selects that assembly; the default pairs the Kennedy-Gruber two-point volume
# flux with a Roe interface flux.
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
# The DG form runs a shorter case at a smaller timestep, for two separate
# reasons.
#
# The timestep: its explicit horizontal terms are stiffer, because the
# interface penalty acts through the small quadrature weight of an
# element-edge node. Measured at this resolution, the run diverges within the
# hour at dt = 200, survives dt = 150 in Float64, and needs dt = 100 in
# Float32.
#
# The horizon: with hyperdiffusion off (below) nothing damps a grid-scale
# momentum mode inside an element — the interface flux only damps jumps
# between elements. Under `DG_FLUX=rusanov`, whose weak-form volume term
# de-aliases nothing, one grows at the cubed-sphere panel corners: max|w| runs
# 0.011 m/s at 12 h, 0.022 at 24 h, 0.048 at 36 h and 0.108 at 48 h, doubling
# every ~12 h against the CG run's steady 0.01-0.03, and the run diverges on
# day 5. Flux differencing slows that a long way — under the default `kg-roe`
# the same sequence is 0.008, 0.022, 0.035, 0.090, and by 72 h 0.111 against
# `rusanov`'s 0.466, peaking over the equator rather than a panel corner —
# which is the de-aliasing `Operators.SplitDivergence` gives the CG form. It
# does not remove the mode: `kg-roe` reaches day 9 rather than day 5, but by
# day 5 max|v| is already 20 m/s against the CG run's 6.5 at day 10. Two days
# is inside the clean window for either, and long enough for the wave to grow
# (the check below wants a factor of 4; the DG run reaches 12). The full ten
# days needs a momentum closure on DG spaces — see the comment on
# hyperdiffusion below.
t_end = discretization isa Grids.DG ? FT(60 * 60 * 24 * 2) : FT(60 * 60 * 24 * 10)
dt = discretization isa Grids.DG ? FT(100) : FT(400)
dt_save_to_sol = FT(60 * 60 * 24)
dt_save_to_disk = FT(0) # 0 means don't save to disk
ode_algorithm = CTS.SSP333
jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

# Hyperdiffusion is the grid-scale closure of the CG form, which has no other
# dissipation. The DG form runs without it, because
# `Operators.vector_laplacian` has no DG method (it would need grad-div and
# curl-curl face lifting; ClimaCore.jl#2605 §4). Its scalar counterpart does
# run on DG, so only the momentum half is missing — but that is the half this
# case needs, which is why the DG horizon above is two days.
#
# Composing two DG `scalar_laplacian` passes into a ∇⁴ also needs its own κ₄:
# the two forms damp a planetary-scale mode at the same rate (-9.57e-10 s⁻¹
# against -9.55e-10 at κ₄ = 2e17), but the interior penalty, squared by the
# two passes, makes the DG operator's spectral radius 2700x the CG one
# (9.4 s⁻¹ against 3.5e-3), so the CG value is far outside the explicit
# stability limit there.
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

    # Conservation, and baroclinic growth: the initial 1 m/s perturbation
    # must amplify into a wave with meridional winds of several m/s. Measured
    # drift and growth over the CG run's ten days: 2e-6 in both, 0.76 → 6.5.
    # Over the DG run's two days: 0 in ρ (its momentum equation is in flux
    # form, so mass conservation is a property of the assembly) and 2.0e-7 in
    # ρe, with 0.76 → 9.4.
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
end
