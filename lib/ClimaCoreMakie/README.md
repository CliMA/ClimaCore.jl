## ClimaCoreMakie

**This package is now a compatibility shim.** The Makie plotting recipes for
ClimaCore fields live in ClimaCore itself, as the `ClimaCoreMakieExt` package
extension, with entry points in the `ClimaCore.Visualize` module. New code
should not depend on ClimaCoreMakie:

    julia> using ClimaCore, CairoMakie  # or GLMakie, WGLMakie, ...

    julia> ClimaCore.Visualize.fieldheatmap(field)

Loading ClimaCoreMakie continues to work: it re-exports the same functions
(`fieldheatmap(!)`, `fieldcontourf(!)`) and re-points the recipe types
(`FieldHeatmap`, `FieldContourf`, `FieldLine`) at the extension, so existing
code and downstream packages are unaffected.
