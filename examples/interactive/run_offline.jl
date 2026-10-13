# Headless driver for the terrain-flow model: runs a preset terrain for a
# fixed physical time, reports cost per step and conservation, and writes an
# animation. Used to validate and tune the model without the GUI.
#
#     julia --project=examples/interactive examples/interactive/run_offline.jl [preset] [hours]
#
# `preset` is one of the keys of `TerrainFlow.TERRAIN_PRESETS` (default
# `schar`), `hours` the physical run time (default 1).

import TerrainFlow
import ClimaCore
using CairoMakie
import ClimaCore.Visualize: fieldheatmap, fieldheatmap!

preset = Symbol(get(ARGS, 1, "schar"))
hours = parse(Float64, get(ARGS, 2, "1"))

cfg = TerrainFlow.GridConfig()
p = TerrainFlow.default_params(cfg)
terrain = TerrainFlow.TERRAIN_PRESETS[preset]()
@info "Grid" Δx = TerrainFlow.Δx(cfg) Δz = TerrainFlow.Δz(cfg) dt =
    TerrainFlow.time_step(cfg)
@info "Parameters" p.u₀ p.N p.κ₂ p.κ₄

model = TerrainFlow.Model(cfg, p, terrain)
mass_0 = TerrainFlow.total_mass(model)
energy_0 = TerrainFlow.total_energy(model)

# Warm up (compile), then time.
TerrainFlow.step!(model, 2)
n_time = 50
t_wall = @elapsed TerrainFlow.step!(model, n_time)
ms_per_step = 1e3 * t_wall / n_time
@info "Cost" ms_per_step model_seconds_per_wall_second = model.dt / (t_wall / n_time)

# Run, saving frames every `save_every` seconds of model time.
t_end = 3600 * hours
save_every = 60.0
steps_per_frame = max(1, round(Int, save_every / model.dt))
frames = Any[]
times = Float64[]
w_max = Float64[]
while TerrainFlow.current_time(model) < t_end
    TerrainFlow.step!(model, steps_per_frame)
    w = TerrainFlow.diagnose(model, :w)
    push!(times, TerrainFlow.current_time(model))
    push!(w_max, maximum(abs, parent(w)))
    push!(
        frames,
        (
            w = copy(w),
            θ = TerrainFlow.diagnose(model, :θ),
            q = TerrainFlow.diagnose(model, :q),
        ),
    )
    if length(frames) % 10 == 0
        @info "t = $(round(times[end] / 60; digits = 1)) min, max|w| = $(round(w_max[end]; digits = 3)) m/s"
    end
end

mass_drift = abs(TerrainFlow.total_mass(model) - mass_0) / mass_0
energy_drift = abs(TerrainFlow.total_energy(model) - energy_0) / energy_0
@info "Conservation" mass_drift energy_drift
@info "Peak |w| over the run: $(maximum(w_max)) m/s; final: $(w_max[end]) m/s"

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)

CairoMakie.activate!(type = "png")
wlim = max(0.5, 0.8 * maximum(w_max))
fig = Figure(size = (1000, 800))
ax_w = Axis(fig[1, 1]; title = "w (m/s)", xlabel = "x (m)", ylabel = "z (m)")
ax_q = Axis(fig[2, 1]; title = "dye", xlabel = "x (m)", ylabel = "z (m)")
frame = Observable(frames[1])
w_obs = @lift $frame.w
q_obs = @lift $frame.q
hm = fieldheatmap!(ax_w, w_obs; colormap = :balance, colorrange = (-wlim, wlim))
Colorbar(fig[1, 2], hm)
fieldheatmap!(ax_q, q_obs; colormap = :grays, colorrange = (0, 1))
record(fig, joinpath(outdir, "$(preset).mp4"), eachindex(frames); framerate = 10) do i
    frame[] = frames[i]
    ax_w.title = "w (m/s), t = $(round(times[i] / 60; digits = 1)) min"
end
frame[] = frames[end]
ax_w.title = "w (m/s), t = $(round(times[end] / 60; digits = 1)) min"
save(joinpath(outdir, "$(preset)_final.png"), fig)
@info "Wrote $(joinpath(outdir, "$(preset).mp4")) and $(preset)_final.png"

fig2 = Figure(size = (600, 300))
ax = Axis(fig2[1, 1]; xlabel = "t (min)", ylabel = "max |w| (m/s)")
lines!(ax, times ./ 60, w_max)
save(joinpath(outdir, "$(preset)_wmax.png"), fig2)
