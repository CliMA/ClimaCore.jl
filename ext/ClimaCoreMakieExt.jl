module ClimaCoreMakieExt

import Makie: Makie, @recipe, GLTriangleFace
import ClimaCore
import ClimaCore.Visualize:
    fieldcontourf,
    fieldcontourf!,
    fieldheatmap,
    fieldheatmap!,
    fieldline,
    fieldline!

include(joinpath("makie", "utils.jl"))
include(joinpath("makie", "fieldline.jl"))
include(joinpath("makie", "fieldheatmap.jl"))
include(joinpath("makie", "fieldcontourf.jl"))

end # module
