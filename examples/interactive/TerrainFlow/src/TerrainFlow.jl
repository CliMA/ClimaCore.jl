"""
    TerrainFlow

Stratified flow over drawn terrain on ClimaCore: the model (`model.jl`) and
its Makie front end (`gui.jl`). A package rather than an `include`d script so
that the compiled code is cached between sessions, see `precompile.jl`.
"""
module TerrainFlow

include("model.jl")
include("gui.jl")
include("precompile.jl")

end # module
