# Companion packages

Two packages under [`lib/`](https://github.com/CliMA/ClimaCore.jl/tree/main/lib) extend ClimaCore for output and analysis. Each has
its own `Project.toml` and is installed separately. Field plotting, by contrast,
is built into ClimaCore through the `ClimaCore.Visualize` module
([Visualize](../reference/visualize.md)), whose functions the `ClimaCoreMakieExt`
extension defines once a Makie backend is loaded.

  - **ClimaCoreTempestRemap** writes cubed-sphere meshes and fields in the
    formats [TempestRemap](https://github.com/ClimateGlobalChange/tempestremap)
    reads, and runs it to compute and apply conservative remapping weights to a
    latitude–longitude grid. Its interface is on the
    [ClimaCoreTempestRemap](ClimaCoreTempestRemap.md) page.
  - **ClimaCoreSpectra** computes spherical-harmonic power spectra of fields on
    latitude–longitude grids. Its interface is on the
    [ClimaCoreSpectra](ClimaCoreSpectra.md) page.
