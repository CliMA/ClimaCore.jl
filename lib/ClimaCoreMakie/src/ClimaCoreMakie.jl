module ClimaCoreMakie

export fieldcontourf,
    fieldcontourf!,
    fieldheatmap,
    fieldheatmap!,
    plot_fieldmatrix_sign!,
    plot_fieldmatrix!

import Makie: Makie, @recipe, lift, GLTriangleFace, Point3f, Observable
import ClimaCore

include("utils.jl")
include("fieldline.jl")
include("fieldheatmap.jl")
include("fieldcontourf.jl")
include("matrix_field.jl")

end # module
