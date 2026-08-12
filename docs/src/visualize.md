# Visualization

```@meta
CurrentModule = ClimaCore
```

ClimaCore provides [Makie.jl](https://github.com/MakieOrg/Makie.jl) recipes for
plotting fields through the `ClimaCoreMakieExt` package extension. The
extension loads automatically when ClimaCore and a Makie backend are both
loaded, and the plotting functions live in the `ClimaCore.Visualize` module:

```julia
using ClimaCore, CairoMakie

ClimaCore.Visualize.fieldheatmap(field)
```

Fields with a registered plot type can also be plotted with plain
`Makie.plot(field)`.

The recipe plot types (`FieldHeatmap`, `FieldContourf`, `FieldLine`) are
defined in the extension module. For theming, refer to them by symbol, e.g.
`Makie.Theme(FieldHeatmap = (colormap = :plasma,))`, or obtain the type with
`Makie.symbol_to_plot(Val(:FieldHeatmap))`.

```@docs
Visualize.fieldheatmap
Visualize.fieldheatmap!
Visualize.fieldcontourf
Visualize.fieldcontourf!
Visualize.fieldline
Visualize.fieldline!
```
