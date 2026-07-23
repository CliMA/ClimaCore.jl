#=
Baroclinic wave (Ullrich et al. 2014 dry variant, as in the CG example
`baroclinic_wave_rhoe.jl`) on a cubed-sphere shell with DG horizontal
spectral elements (no DSS) and FD vertical staggering, fully explicit SSP-RK3.
See `sphere_dg_fd_model.jl` for the discretization.

Defaults: h_elem = 4, npoly = 4, z_elem = 10, z_top = 30 km, dt = 4 s,
t_end = 1 day (the explicit acoustic dt makes 10 days ≈ 216k steps; override
with T_END=864000 for the full run). κ₄ = min(2e17, explicit SIPG cap ≈ 6e14
at default resolution) m⁴/s, no κ₂.

Run:
  julia --project=.buildkite examples/hybrid/sphere/baroclinic_wave_dg_fd.jl
Balanced-state check (perturbation off, 1 h):
  PERTURB=0 T_END=3600 julia --project=.buildkite examples/hybrid/sphere/baroclinic_wave_dg_fd.jl

Environment: HELEM, NPOLY, ZELEM, ZMAX, DT, T_END, DT_SAVE, KAPPA4, FILTER, PERTURB
=#

const FT = Float64
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

const dt_save = parse(FT, get(ENV, "DT_SAVE", string(min(t_end, 21600.0))))
prob = ODEProblem(rhs!, Y, (FT(0), t_end))
sol = solve(prob, SSPRK33(), dt = Δt, saveat = dt_save)

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

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
output_dir = joinpath(@__DIR__, "output", "baroclinic_wave_dg_fd")
mkpath(output_dir)

# Plot recipes index into field data, so move results to the CPU first.
ᶜv_end = ClimaCore.to_cpu(Geometry.UVVector.(sol.u[end].uₕ).components.data.:2)
Plots.png(
    Plots.plot(ᶜv_end, level = 3, clim = (-6, 6)),
    joinpath(output_dir, "v_end.png"),
)
if length(sol.u) > 2
    anim = Plots.@animate for Yi in sol.u
        ᶜv = ClimaCore.to_cpu(Geometry.UVVector.(Yi.uₕ).components.data.:2)
        Plots.plot(ᶜv, level = 3, clim = (-6, 6))
    end
    Plots.mp4(anim, joinpath(output_dir, "v.mp4"), fps = 5)
end
@info "Output written to $output_dir"
