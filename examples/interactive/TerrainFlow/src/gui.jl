# Makie front end for the terrain-flow model in `model.jl`.
#
# Layout:
#
#   ┌ controls ┐ ┌──────────── flow: heatmap of the chosen diagnostic ─────────────┐
#   │ run/pause│ │                                                                 │
#   │ reset    │ │                                                                 │
#   │ sliders  │ │                      (terrain drawn in black)                   │
#   │ menus    │ └─────────────────────────────────────────────────────────────────┘
#   │          │ ┌──── terrain profile: click-drag to draw, released = applied ────┐
#   └──────────┘ └─────────────────────────────────────────────────────────────────┘
#
# This file only needs `Makie`; the caller activates a backend (`run.jl` uses
# GLMakie, `run_offline.jl` CairoMakie). `build_demo` assembles the
# figure and the callbacks; `run_demo` displays it and steps the model from
# the main task until the window closes, while the render loop runs
# concurrently. GUI callbacks only mutate parameters or set request flags that
# the main loop acts on, so the model is never rebuilt while it is stepped.

using Makie
import ClimaCore.Visualize: fieldheatmap!

const H_MAX = 3000.0  # tallest terrain the profile axis allows (m)
const ANOMALIES = (:w, :u′, :θ′, :p′)

"""
Colour range for a diagnostic: symmetric about zero for anomalies.
"""
function color_range(name::Symbol, field, contrast)
    data = filter(isfinite, vec(parent(field)))
    isempty(data) && return (-1.0, 1.0)
    if name in ANOMALIES
        m = max(maximum(abs, data), eps())
        return (-m * contrast, m * contrast)
    else
        lo, hi = extrema(data)
        hi > lo || (hi = lo + eps())
        mid = (lo + hi) / 2
        half = (hi - lo) / 2 * contrast
        return (mid - half, mid + half)
    end
end

color_map(name::Symbol) =
    name in ANOMALIES ? :balance :
    name === :q ? :grays : name === :θ ? :thermal : :viridis

km_ticks(values) = [string(round(v / 1e3; digits = 1)) for v in values]

"""
    build_demo(model) -> (; fig, refresh!, rebuild!, step!, running, model)

Assemble the figure and its callbacks around `model`. `refresh!()` redraws the
current state; `rebuild!()` applies any pending terrain, reset or resolution
change (and is what the main loop calls when `needs_rebuild()` is true);
`step!()` advances the model by the current steps-per-frame and redraws.
`model()` returns the current model, which `rebuild!` may replace.
"""
function build_demo(model::Model)
    cfg = model.cfg
    p = model.p

    # ------------------------------------------------------------------ state
    running = Observable(false)
    # Why the run stopped for good (blow-up or error), until the next rebuild.
    halted = Ref{Union{Nothing, String}}(nothing)
    quantity = Observable(:w)
    contrast = Observable(1.0)
    steps_per_frame = Observable(10)
    reset_requested = Ref(false)
    rebuild_cfg = Ref{Union{Nothing, GridConfig}}(nothing)
    pending_terrain = Ref{Any}(nothing)
    current = Ref{Model}(model)  # a resolution change alters the Model type

    # Untyped because a resolution change alters the field type (the number of
    # levels is a type parameter); the heatmap is then replaced, see `rebuild!`.
    field = Observable{Any}(TerrainFlow.diagnose(model, quantity[]))
    crange = Observable(color_range(quantity[], field[], contrast[]))
    cmap = Observable(color_map(quantity[]))
    status = Observable("")

    # Terrain profile the user edits, sampled on a fixed grid (m).
    xs = range(-cfg.Lx / 2, cfg.Lx / 2; length = 4 * cfg.nx * cfg.np)
    profile = Observable(Float64.(model.terrain.(xs)))

    # ------------------------------------------------------------------ figure
    fig = Figure(size = (1500, 900), fontsize = 14)
    controls = GridLayout(fig[1:2, 1]; tellheight = false, width = 260)
    ax = Axis(
        fig[1, 2];
        title = "Stratified flow over terrain (ClimaCore)",
        xlabel = "x (km)",
        ylabel = "z (km)",
        xtickformat = km_ticks,
        ytickformat = km_ticks,
        aspect = DataAspect(),
    )
    ax_profile = Axis(
        fig[2, 2];
        xlabel = "x (km)",
        ylabel = "terrain (km)",
        xtickformat = km_ticks,
        ytickformat = km_ticks,
        height = 160,
    )
    linkxaxes!(ax, ax_profile)

    heatmap!() = fieldheatmap!(ax, field; colormap = cmap, colorrange = crange)
    hm = Ref{Any}(heatmap!())
    Colorbar(
        fig[1, 3];
        colormap = cmap,
        limits = crange,
        label = @lift(DIAGNOSTICS[$quantity]),
    )
    xs_vec = collect(xs)
    # The grid uses the slope-limited terrain, which can differ from the drawn
    # profile; the band in the flow axis shows what the model actually sees.
    model_profile = Observable(Float64.(model.terrain.(xs)))
    terrain_band = band!(ax, xs_vec, zero(xs_vec), model_profile; color = :black)
    translate!(terrain_band, 0, 0, 10)
    limits!(ax, -cfg.Lx / 2, cfg.Lx / 2, 0, cfg.Lz)

    # Wind remapped to a regular grid once per frame, shared by the tracer
    # particles and the flowlines.
    wind = Ref{WindGrid}(WindGrid(model))
    TerrainFlow.remap!(wind[], model)

    # Flowlines: instantaneous streamlines traced from the upstream edge.
    show_flowlines = Observable(true)
    flowline_points = Observable(Point2f[])
    function update_flowlines!()
        x, z = TerrainFlow.streamlines(wind[], current[])
        flowline_points[] = Point2f.(x, z)
        return nothing
    end
    update_flowlines!()
    flowline_plot = lines!(
        ax,
        flowline_points;
        color = (:black, 0.6),
        linewidth = 1.2,
        visible = show_flowlines,
    )
    translate!(flowline_plot, 0, 0, 15)

    # Tracer particles, advected once per frame.
    tracers = Ref(Tracers(model))
    tracer_points = Observable(Point2f.(tracers[].x, tracers[].z))
    show_tracers = Observable(true)
    tracer_plot = scatter!(
        ax,
        tracer_points;
        color = :black,
        markersize = 5,
        visible = show_tracers,
    )
    translate!(tracer_plot, 0, 0, 20)

    band!(ax_profile, xs_vec, zero(xs_vec), profile; color = (:black, 0.6))
    lines!(ax_profile, xs_vec, profile; color = :black)
    limits!(ax_profile, -cfg.Lx / 2, cfg.Lx / 2, 0, H_MAX)
    Label(
        fig[2, 3],
        "draw here\n(release\nto apply)";
        justification = :left,
        tellheight = false,
    )

    # ------------------------------------------------------------------ controls
    row = Ref(1)
    nextrow() = (r = row[]; row[] += 1; r)

    Label(
        controls[nextrow(), 1],
        status;
        tellwidth = false,
        justification = :left,
        font = :bold,
    )

    run_button = Button(controls[nextrow(), 1]; label = "Run", tellwidth = false)
    on(run_button.clicks) do _
        # A halted run cannot continue; Reset clears it.
        running[] = !running[] && isnothing(halted[])
    end
    on(running) do r
        run_button.label = r ? "Pause" : "Run"
    end
    reset_button = Button(controls[nextrow(), 1]; label = "Reset", tellwidth = false)
    on(reset_button.clicks) do _
        reset_requested[] = true
    end

    Label(controls[nextrow(), 1], "terrain preset"; tellwidth = false)
    preset_menu = Menu(
        controls[nextrow(), 1];
        options = [(string(k), k) for k in keys(TERRAIN_PRESETS)],
        default = nothing,
        tellwidth = false,
    )
    on(preset_menu.selection) do sel
        isnothing(sel) && return
        h = TERRAIN_PRESETS[sel]()
        profile[] = Float64.(h.(xs))
        pending_terrain[] = SampledTerrain(xs, copy(profile[]))
    end

    Label(controls[nextrow(), 1], "plot"; tellwidth = false)
    quantity_menu = Menu(
        controls[nextrow(), 1];
        options = [(string(k), k) for k in keys(DIAGNOSTICS)],
        default = "w",
        tellwidth = false,
    )
    on(quantity_menu.selection) do sel
        isnothing(sel) && return
        quantity[] = sel
        isnothing(halted[]) || return  # shown after the next reset
        cmap[] = color_map(sel)
        field[] = TerrainFlow.diagnose(current[], sel)
        crange[] = color_range(sel, field[], contrast[])
    end

    sg = SliderGrid(
        controls[nextrow(), 1],
        (label = "wind U (m/s)", range = 2:0.5:25, startvalue = p.u₀),
        (label = "N (1/s)", range = 0.005:0.001:0.02, startvalue = p.N),
        (label = "diffusion ×", range = 0.1:0.1:10, startvalue = 1.0),
        (label = "steps/frame", range = 1:1:60, startvalue = steps_per_frame[]),
        (label = "contrast", range = 0.05:0.05:1, startvalue = 1.0);
        tellwidth = false,
    )
    u_slider, N_slider, diff_slider, steps_slider, contrast_slider = sg.sliders
    diffusivity_base = Ref((p.κ₂, p.κ₄))
    on(u_slider.value) do v
        current[].p.u₀ = v  # takes effect on reset (initial state and sponge target)
    end
    on(N_slider.value) do v
        current[].p.N = v   # takes effect on reset
    end
    on(diff_slider.value) do v
        κ₂, κ₄ = diffusivity_base[]
        current[].p.κ₂ = κ₂ * v
        current[].p.κ₄ = κ₄ * v
    end
    on(steps_slider.value) do v
        steps_per_frame[] = v
    end
    on(contrast_slider.value) do v
        contrast[] = v
        crange[] = color_range(quantity[], field[], v)
    end
    Label(
        controls[nextrow(), 1],
        "U and N apply on reset.\nRegime: N h / U < 0.5 waves,\n≈ 1 breaking, > 1.5 blocking.";
        tellwidth = false,
        justification = :left,
        fontsize = 12,
    )

    toggle_row = GridLayout(controls[nextrow(), 1]; tellwidth = false)
    flowline_toggle = Toggle(toggle_row[1, 1]; active = show_flowlines[])
    Label(toggle_row[1, 2], "flowlines"; tellwidth = false, halign = :left)
    tracer_toggle = Toggle(toggle_row[1, 3]; active = show_tracers[])
    Label(toggle_row[1, 4], "tracers"; tellwidth = false, halign = :left)
    on(flowline_toggle.active) do active
        show_flowlines[] = active
        # The wind grid is only refreshed while something uses it.
        if active
            TerrainFlow.remap!(wind[], current[])
            update_flowlines!()
        end
    end
    on(tracer_toggle.active) do active
        show_tracers[] = active
    end
    reseed_button =
        Button(controls[nextrow(), 1]; label = "Reseed tracers", tellwidth = false)
    on(reseed_button.clicks) do _
        tracers[] = Tracers(current[])
        tracer_points[] = Point2f.(tracers[].x, tracers[].z)
    end

    Label(controls[nextrow(), 1], "resolution"; tellwidth = false)
    grid_menu = Menu(
        controls[nextrow(), 1];
        options = [(string(k), k) for k in keys(GRID_PRESETS)],
        default = nothing,
        tellwidth = false,
    )
    on(grid_menu.selection) do sel
        isnothing(sel) && return
        rebuild_cfg[] = GRID_PRESETS[sel]
    end

    # ------------------------------------------------------------------ drawing
    # Drag in the profile axis to draw terrain; the model is rebuilt on release.
    foreach(
        k -> deregister_interaction!(ax_profile, k),
        (:rectanglezoom, :dragpan, :scrollzoom, :limitreset),
    )
    drawing = Ref(false)
    last_point = Ref{Union{Nothing, Tuple{Float64, Float64}}}(nothing)
    index_of(x) = clamp(round(Int, (x - first(xs)) / step(xs)) + 1, 1, length(xs))
    function paint!(x, h)
        h = clamp(h, 0.0, H_MAX)
        hs = profile[]
        if isnothing(last_point[])
            hs[index_of(x)] = h
        else
            # Fill between the previous and current mouse positions so fast
            # drags leave no gaps.
            x₀, h₀ = last_point[]
            i₀, i₁ = index_of(x₀), index_of(x)
            for i in min(i₀, i₁):max(i₀, i₁)
                α = i₁ == i₀ ? 1.0 : (i - i₀) / (i₁ - i₀)
                hs[i] = (1 - α) * h₀ + α * h
            end
        end
        last_point[] = (x, h)
        notify(profile)
    end
    on(events(ax_profile).mousebutton; priority = 10) do ev
        if ev.button == Mouse.left
            if ev.action == Mouse.press && is_mouseinside(ax_profile)
                drawing[] = true
                last_point[] = nothing
                paint!(mouseposition(ax_profile)...)
                return Consume(true)
            elseif ev.action == Mouse.release && drawing[]
                drawing[] = false
                pending_terrain[] = SampledTerrain(xs, copy(profile[]))
                return Consume(true)
            end
        end
        return Consume(false)
    end
    on(events(ax_profile).mouseposition; priority = 10) do _
        if drawing[]
            paint!(mouseposition(ax_profile)...)
            return Consume(true)
        end
        return Consume(false)
    end

    # ------------------------------------------------------------------ actions
    # Once halted, the state is not plotted again (it may hold NaNs or a
    # negative temperature); the last good frame stays on screen.
    function refresh!()
        m = current[]
        t = string("t = ", round(TerrainFlow.current_time(m) / 60; digits = 1), " min")
        if !isnothing(halted[])
            status[] = string(t, "\n", halted[])
            return nothing
        end
        field[] = TerrainFlow.diagnose(m, quantity[])
        crange[] = color_range(quantity[], field[], contrast[])
        h_max = maximum(m.terrain.hs)
        w_max = maximum(abs, parent(TerrainFlow.diagnose(m, :w)))
        status[] = string(
            t, "\n",
            "N h / U = ", round(TerrainFlow.froude_parameter(m.p, h_max); digits = 2),
            "\n",
            "max |w| = ", round(w_max; digits = 2), " m/s",
        )
        return nothing
    end

    """
    Stop the run until the next reset or rebuild, showing `reason`.
    """
    function halt!(reason)
        halted[] = reason
        running[] = false
        refresh!()
        return nothing
    end

    needs_rebuild() =
        reset_requested[] || !isnothing(pending_terrain[]) || !isnothing(rebuild_cfg[])

    function rebuild!()
        was_running = running[]
        running[] = false
        # Take the requests first, so a rebuild that throws is not retried on
        # every pass of the main loop.
        new_cfg, terrain = rebuild_cfg[], pending_terrain[]
        rebuild_cfg[] = pending_terrain[] = nothing
        reset_requested[] = false
        m = current[]
        if !isnothing(new_cfg)
            new_p = TerrainFlow.default_params(new_cfg; u₀ = m.p.u₀, N = m.p.N)
            diffusivity_base[] = (new_p.κ₂, new_p.κ₄)
            new_p.κ₂ *= diff_slider.value[]
            new_p.κ₄ *= diff_slider.value[]
            current[] = Model(new_cfg, new_p, SampledTerrain(xs, copy(profile[])))
            # The plot's inputs are typed by the first field it was given.
            delete!(ax, hm[])
            field.val = TerrainFlow.diagnose(current[], quantity[])
            hm[] = heatmap!()
        else
            TerrainFlow.reset!(m; terrain = something(terrain, m.terrain))
        end
        cmap[] = color_map(quantity[])  # in case the menu changed while halted
        halted[] = nothing
        model_profile[] = Float64.(current[].terrain.(xs))
        wind[] = WindGrid(current[])
        TerrainFlow.remap!(wind[], current[])
        update_flowlines!()
        tracers[] = Tracers(current[])
        tracer_points[] = Point2f.(tracers[].x, tracers[].z)
        refresh!()
        running[] = was_running
        return nothing
    end

    # With `interruptible = true` the substeps yield to the render loop, which
    # is where GLMakie polls mouse and keyboard events, and stop early when
    # the run is paused or a rebuild is requested. Without this a click waits
    # for the whole frame (all substeps plus the redraw) to finish.
    #
    # The state is checked after every substep, so a blow-up stops the run
    # before anything (diagnostics, tracers, flowlines) reads a NaN state.
    function step!(; interruptible = false)
        m = current[]
        isnothing(halted[]) || return nothing
        n = 0
        while n < steps_per_frame[]
            TerrainFlow.step!(m, 1)
            n += 1
            if !TerrainFlow.is_healthy(m)
                halt!("blew up: lower the terrain,\nraise diffusion, then Reset")
                return nothing
            end
            if interruptible
                yield()
                (running[] && !needs_rebuild()) || break
            end
        end
        if show_tracers[] || show_flowlines[]
            TerrainFlow.remap!(wind[], m)
        end
        if show_tracers[]
            TerrainFlow.advance!(tracers[], wind[], m, n * m.dt)
            tracer_points[] = Point2f.(tracers[].x, tracers[].z)
        end
        show_flowlines[] && update_flowlines!()
        refresh!()
        return nothing
    end

    refresh!()
    return (;
        fig,
        refresh!,
        rebuild!,
        needs_rebuild,
        step!,
        halt!,
        running,
        model = () -> current[],
    )
end

"""
    run_demo(; grid = :coarse, terrain = :agnesi)

Open the interactive window and block until it is closed. Returns the final
model.
"""
function run_demo(; grid::Symbol = :coarse, terrain::Symbol = :agnesi)
    cfg = GRID_PRESETS[grid]
    @info "Building the model and compiling the time step (first run: ~30 s)"
    model = Model(cfg, TerrainFlow.default_params(cfg), TERRAIN_PRESETS[terrain]())
    # Compile the time step now rather than on the first click of Run, which
    # would otherwise freeze the window for several seconds.
    TerrainFlow.step!(model, 1)
    TerrainFlow.reset!(model)
    demo = build_demo(model)
    screen = display(demo.fig)
    while isopen(screen)
        run_frame!(demo)
    end
    return demo.model()
end

"""
    run_frame!(demo)

One pass of the main loop: apply pending rebuilds, then step and redraw if
running. An error (from the model, a diagnostic, or a rebuild on an odd
terrain) is logged and halts the run until Reset instead of closing the
window.
"""
function run_frame!(demo)
    try
        demo.needs_rebuild() && demo.rebuild!()
        if demo.running[]
            demo.step!(; interruptible = true)
            yield()
        else
            sleep(0.05)
        end
    catch err
        err isa InterruptException && rethrow()
        @error "Demo step failed; paused until Reset" exception =
            (err, catch_backtrace())
        demo.halt!("error: $(nameof(typeof(err)))\n(see terminal), press Reset")
    end
    return nothing
end
