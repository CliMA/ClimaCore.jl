#=
Baroclinic wave (Ullrich et al. 2014 dry variant, as in the CG example
`baroclinic_wave_rhoe.jl`) on a cubed-sphere shell with DG horizontal
spectral elements (no DSS) and FD vertical staggering. Fully explicit
SSP-RK3 by default; STEPPER=hevi selects IMEX ARK with implicit vertical
acoustics (much larger Δt). See `sphere_dg_fd_model.jl` for the
discretization.

Defaults: h_elem = 4, npoly = 4, z_elem = 10, z_top = 30 km, dt = 4 s,
t_end = 1 day (the explicit acoustic dt makes 10 days ≈ 216k steps; override
with T_END=864000 for the full run). κ₄ = min(2e17, explicit SIPG cap ≈ 6e14
at default resolution) m⁴/s, no κ₂.

Run:
  julia --project=.buildkite examples/hybrid/sphere/baroclinic_wave_dg_fd.jl
Balanced-state check (perturbation off, 1 h):
  PERTURB=0 T_END=3600 julia --project=.buildkite examples/hybrid/sphere/baroclinic_wave_dg_fd.jl

Environment: HELEM, NPOLY, ZELEM, ZMAX, DT, T_END, DT_SAVE, KAPPA4, FILTER,
PERTURB, STEPPER
=#

const FT = get(ENV, "FLOAT_TYPE", "Float64") == "Float32" ? Float32 : Float64
const apply_held_suarez = false
const is_balanced_flow = get(ENV, "PERTURB", "1") == "0"
const t_end_default = 86400.0

include("sphere_dg_fd_model.jl")

import LinearAlgebra: norm

Y = initial_state(
    Fields.local_geometry_field(hv_center_space),
    Fields.local_geometry_field(hv_face_space),
)
uₕ_init = copy(Y.uₕ)
const mass_0 = sum(Y.Yc.ρ)
const energy_0 = sum(Y.Yc.ρe)

# Smoke-check the RHS before committing to the run
dY = similar(Y)
rhs!(dY, Y, nothing, 0.0)
@info "Initial RHS" max_dρ = maximum(abs, parent(dY.Yc.ρ)) max_dρe =
    maximum(abs, parent(dY.Yc.ρe)) max_duₕ = maximum(abs, parent(dY.uₕ)) max_dw =
    maximum(abs, parent(dY.w))

# min against t_end: a DT_SAVE larger than t_end would collapse saveat to
# [0] and silently discard the entire solution (same guard as FDDG driver)
const dt_save =
    min(t_end, parse(FT, get(ENV, "DT_SAVE", string(min(t_end, 21600.0)))))
sol = run_simulation(Y; dt_save)

@info "Conservation" mass_rel = (sum(sol.u[end].Yc.ρ) - mass_0) / mass_0 energy_rel =
    (sum(sol.u[end].Yc.ρe) - energy_0) / energy_0
@info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].Yc.ρe))"
@info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].Yc.ρe))"

if is_balanced_flow
    # The balanced zonal jet should be (approximately) steady: report drift.
    uv_end = Geometry.UVVector.(sol.u[end].uₕ)
    uv_init = Geometry.UVVector.(uₕ_init)
    max_v = maximum(abs, parent(uv_end.components.data.:2))
    du = @. uv_end - uv_init
    @info "Balanced-flow drift" max_v max_du =
        maximum(abs, parent(du)) max_w = maximum(abs, parent(sol.u[end].w)) max_u₀ =
        maximum(abs, parent(uv_init.components.data.:1))
end

import CairoMakie, ClimaCoreMakie
output_dir = joinpath(@__DIR__, "output", "baroclinic_wave_dg_fd")
mkpath(output_dir)

# Move results to the CPU first (move the plain prognostic field, then extract
# components on the CPU). ClimaCoreMakie.fieldheatmap plots a 2D field, so slice
# level 3 out of the extruded field — its coordinates are LatLongPoints, so this
# renders a long–lat map.
getv(Yi) = Geometry.UVVector.(ClimaCore.to_cpu(Yi.uₕ)).components.data.:2
let fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "long [deg]", ylabel = "lat [deg]")
    plt = ClimaCoreMakie.fieldheatmap!(
        ax,
        Fields.level(getv(sol.u[end]), 3);
        colorrange = (-6, 6),
    )
    CairoMakie.Colorbar(fig[1, 2], plt)
    CairoMakie.save(joinpath(output_dir, "v_end.png"), fig)
end
if length(sol.u) > 2
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "long [deg]", ylabel = "lat [deg]")
    frame = CairoMakie.Observable(Fields.level(getv(sol.u[1]), 3))
    plt = ClimaCoreMakie.fieldheatmap!(ax, frame; colorrange = (-6, 6))
    CairoMakie.Colorbar(fig[1, 2], plt)
    CairoMakie.record(
        fig,
        joinpath(output_dir, "v.mp4"),
        sol.u;
        framerate = 5,
    ) do Yi
        frame[] = Fields.level(getv(Yi), 3)
    end
end
@info "Output written to $output_dir"
