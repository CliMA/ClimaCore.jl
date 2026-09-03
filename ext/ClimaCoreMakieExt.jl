module ClimaCoreMakieExt

import Makie: Makie, @recipe, GLTriangleFace, Point3f
import ClimaCore
import ClimaCore.Visualize:
    fieldline, fieldline!, fieldheatmap, fieldheatmap!, fieldcontourf,
    fieldcontourf!

include("makie/utils.jl")
include("makie/fieldline.jl")
include("makie/fieldheatmap.jl")
include("makie/fieldcontourf.jl")

end
