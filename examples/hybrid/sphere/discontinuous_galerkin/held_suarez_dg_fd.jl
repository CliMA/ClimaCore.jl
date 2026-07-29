#=
Held–Suarez (1994) forced dry dynamical core on a cubed-sphere shell with DG
horizontal spectral elements (no DSS) and FD vertical staggering, fully
explicit SSP-RK3 by default (STEPPER=hevi for IMEX ARK with implicit
vertical acoustics). Same dynamics as `baroclinic_wave_dg_fd.jl` (see
`sphere_dg_fd_model.jl`) plus Rayleigh low-level drag and Newtonian
temperature relaxation; starts from the balanced zonal jet (no perturbation).

Defaults: t_end = 2 days (climate-length runs are impractical with the
explicit acoustic dt; override with T_END). κ₄ = min(2e17, explicit SIPG
cap) m⁴/s, no κ₂.

Run:
  julia --project=.buildkite examples/hybrid/sphere/held_suarez_dg_fd.jl

Environment: HELEM, NPOLY, ZELEM, ZMAX, DT, T_END, DT_SAVE, KAPPA4, FILTER,
STEPPER
=#

const FT = Float64
const apply_held_suarez = true
const is_balanced_flow = true
const t_end_default = 172800.0

include("sphere_dg_fd_model.jl")

import LinearAlgebra: norm

Y = initial_state(
    Fields.local_geometry_field(hv_center_space),
    Fields.local_geometry_field(hv_face_space),
)
const mass_0 = sum(Y.Yc.ρ)
const energy_0 = sum(Y.Yc.ρe)

# Smoke-check the RHS before committing to the run
dY = similar(Y)
rhs!(dY, Y, nothing, 0.0)
@info "Initial RHS" max_dρ = maximum(abs, parent(dY.Yc.ρ)) max_dρe =
    maximum(abs, parent(dY.Yc.ρe)) max_duₕ = maximum(abs, parent(dY.uₕ)) max_dw =
    maximum(abs, parent(dY.w))

const dt_save = parse(FT, get(ENV, "DT_SAVE", string(min(t_end, 21600.0))))
sol = run_simulation(Y; dt_save)

@info "Conservation (energy is forced, mass should be conserved)" mass_rel =
    (sum(sol.u[end].Yc.ρ) - mass_0) / mass_0 energy_rel =
    (sum(sol.u[end].Yc.ρe) - energy_0) / energy_0
@info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρe))"
@info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρe))"

import CairoMakie, ClimaCoreMakie
output_dir = joinpath(@__DIR__, "output", "held_suarez_dg_fd")
mkpath(output_dir)

# Move results to the CPU first (move the plain prognostic field, then extract
# components on the CPU). ClimaCoreMakie.fieldheatmap plots a 2D field, so slice
# level 3 out of the extruded field — its coordinates are LatLongPoints, so this
# renders a long–lat map.
getu(Yi) = Geometry.UVVector.(ClimaCore.to_cpu(Yi.uₕ)).components.data.:1
let fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "long [deg]", ylabel = "lat [deg]")
    plt = ClimaCoreMakie.fieldheatmap!(ax, Fields.level(getu(sol.u[end]), 3))
    CairoMakie.Colorbar(fig[1, 2], plt)
    CairoMakie.save(joinpath(output_dir, "u_end.png"), fig)
end
if length(sol.u) > 2
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "long [deg]", ylabel = "lat [deg]")
    frame = CairoMakie.Observable(Fields.level(getu(sol.u[1]), 3))
    plt = ClimaCoreMakie.fieldheatmap!(ax, frame)
    CairoMakie.Colorbar(fig[1, 2], plt)
    CairoMakie.record(
        fig,
        joinpath(output_dir, "u.mp4"),
        sol.u;
        framerate = 5,
    ) do Yi
        frame[] = Fields.level(getu(Yi), 3)
    end
end
@info "Output written to $output_dir"
