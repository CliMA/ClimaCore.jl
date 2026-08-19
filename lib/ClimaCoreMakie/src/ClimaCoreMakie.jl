"""
    ClimaCoreMakie

Deprecated compatibility shim: the Makie recipes now live in the
`ClimaCoreMakieExt` extension of ClimaCore, which loads automatically when
both `ClimaCore` and `Makie` are loaded. Use `ClimaCore.Visualize` directly;
this package only re-exports its plotting functions.
"""
module ClimaCoreMakie

import Makie, ClimaCore
import ClimaCore.Visualize:
    fieldline,
    fieldline!,
    fieldheatmap,
    fieldheatmap!,
    fieldcontourf,
    fieldcontourf!

export fieldcontourf, fieldcontourf!, fieldheatmap, fieldheatmap!

const FieldLine = Makie.Plot{fieldline}
const FieldHeatmap = Makie.Plot{fieldheatmap}
const FieldContourf = Makie.Plot{fieldcontourf}

end # module
