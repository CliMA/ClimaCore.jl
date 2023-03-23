module ClimaCoreMakie

export fieldcontourf, fieldcontourf!, fieldheatmap, fieldheatmap!

import Makie: Makie, @recipe, lift, GLTriangleFace, Point3f, Observable
import ClimaCore

include("utils.jl")
include("fieldline.jl")
include("fieldheatmap.jl")
include("fieldcontourf.jl")

end # module
