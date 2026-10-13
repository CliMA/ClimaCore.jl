# Precompile workload: run everything the demo does once, so that the compiled
# code is stored in the package image instead of being compiled on every
# launch. Model types depend on the grid (the number of levels and of nodes
# per element are type parameters), so each grid preset is exercised. The
# figure is assembled with Makie alone; the backend (GLMakie) compiles its own
# drawing code at the first display.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        for cfg in values(GRID_PRESETS)
            model = Model(cfg, default_params(cfg), TERRAIN_PRESETS.agnesi())
            step!(model, 2)
            is_healthy(model)
            foreach(name -> diagnose(model, name), keys(DIAGNOSTICS))
            wind = WindGrid(model)
            remap!(wind, model)
            advance!(Tracers(model), wind, model, model.dt)
            streamlines(wind, model)
            reset!(model; terrain = SampledTerrain(cfg, TERRAIN_PRESETS.twin()))
        end

        cfg = GRID_PRESETS.coarse
        demo = build_demo(Model(cfg, default_params(cfg), flat_terrain()))
        demo.running[] = true
        demo.step!()
        demo.running[] = false
        demo.step!(; interruptible = true)
        demo.halt!("precompiling")  # cleared by the rebuilds below
        for menu in demo.fig.content
            menu isa Menu || continue
            # Pick every quantity and preset, and the `fine` grid, through the
            # menus' own callbacks, then apply them as the main loop would.
            for i in eachindex(menu.options[])
                menu.i_selected[] = i
                demo.needs_rebuild() && demo.rebuild!()
            end
        end
    end
end
