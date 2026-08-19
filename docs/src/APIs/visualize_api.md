# Visualize

```@meta
CurrentModule = ClimaCore.Visualize
```

`ClimaCore.Visualize` provides functionality for plotting ClimaCore fields
extending the [Makie.jl](https://github.com/MakieOrg/Makie.jl) package. The
plotting methods are implemented in the `ClimaCoreMakieExt` package extension,
which loads automatically when both `ClimaCore` and `Makie` (e.g. via
`CairoMakie` or `GLMakie`) are loaded.

```@docs
fieldheatmap
fieldheatmap!
fieldcontourf
fieldcontourf!
fieldline
fieldline!
```
