# Visualize

```@meta
CurrentModule = ClimaCore.Visualize
```

`ClimaCore.Visualize` plots ClimaCore fields through
[Makie.jl](https://github.com/MakieOrg/Makie.jl). The functions are empty until a
Makie backend is loaded: `import CairoMakie` (or another backend) activates the
`ClimaCoreMakieExt` extension that defines them. [Plot fields](../howto/plotting.md)
shows them in use.

```@docs
fieldheatmap
fieldheatmap!
fieldcontourf
fieldcontourf!
fieldline
fieldline!
```

The recipe types are defined in the Makie extension. Their attribute lists are
inherited from Makie and link to the Makie documentation.

```@meta
CurrentModule = Base.get_extension(ClimaCore, :ClimaCoreMakieExt)
```

```@docs
FieldHeatmap
FieldContourf
FieldLine
```
