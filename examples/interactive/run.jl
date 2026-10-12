# Entry point for the interactive terrain-flow demo.
#
#     julia --project=examples/interactive examples/interactive/run.jl [grid] [terrain]
#
# `grid` is one of `coarse` (default), `fine`, `finest`; `terrain` one of the
# presets in `TerrainFlow.TERRAIN_PRESETS` (default `agnesi`).

using GLMakie
include(joinpath(@__DIR__, "gui.jl"))

grid = Symbol(get(ARGS, 1, "coarse"))
terrain = Symbol(get(ARGS, 2, "agnesi"))
run_demo(; grid, terrain)
