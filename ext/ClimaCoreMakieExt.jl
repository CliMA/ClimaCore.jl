module ClimaCoreMakieExt

import Makie: Makie, @recipe, lift, GLTriangleFace, Point3f, Observable
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
