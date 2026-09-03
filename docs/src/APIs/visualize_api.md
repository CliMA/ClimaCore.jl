# Visualize

```@meta
CurrentModule = ClimaCore.Visualize
```

`ClimaCore.Visualize` provides functionality for plotting ClimaCore fields
extending the [Makie.jl](https://github.com/MakieOrg/Makie.jl) package. To use
them, load any of the `Makie` backends (e.g. via `CairoMakie` or `GLMakie`).

```@docs
fieldheatmap
fieldheatmap!
fieldcontourf
fieldcontourf!
fieldline
fieldline!
```

```@meta
CurrentModule = Base.get_extension(ClimaCore, :ClimaCoreMakieExt)
```

```@docs
FieldHeatmap
FieldContourf
FieldLine
```
