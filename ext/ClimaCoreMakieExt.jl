module ClimaCoreMakieExt

import Makie: Makie, @recipe, lift, GLTriangleFace, Point3f, Observable
import ClimaCore
import ClimaCore.Visualize:
    fieldline, fieldline!, fieldheatmap, fieldheatmap!, fieldcontourf,
    fieldcontourf!

include("makie/utils.jl")
include("makie/fieldline.jl")
include("makie/fieldheatmap.jl")
include("makie/fieldcontourf.jl")

end
