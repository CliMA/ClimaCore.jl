# Entry point for the interactive terrain-flow demo.
#
#     julia --project=examples/interactive examples/interactive/run.jl [grid] [terrain]
#
# `grid` is one of `coarse` (default), `fine`, `finest`; `terrain` one of the
# presets in `TerrainFlow.TERRAIN_PRESETS` (default `agnesi`).

using GLMakie
import TerrainFlow

grid = Symbol(get(ARGS, 1, "coarse"))
terrain = Symbol(get(ARGS, 2, "agnesi"))
TerrainFlow.run_demo(; grid, terrain)
