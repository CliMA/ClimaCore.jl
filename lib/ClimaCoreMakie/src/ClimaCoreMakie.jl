module ClimaCoreMakie

export fieldcontourf, fieldcontourf!, fieldheatmap, fieldheatmap!

import Makie
import ClimaCore
import ClimaCore.Visualize:
    fieldcontourf,
    fieldcontourf!,
    fieldheatmap,
    fieldheatmap!,
    fieldline,
    fieldline!

const _ext = Base.get_extension(ClimaCore, :ClimaCoreMakieExt)
const FieldContourf = _ext.FieldContourf
const FieldHeatmap = _ext.FieldHeatmap
const FieldLine = _ext.FieldLine

end # module
