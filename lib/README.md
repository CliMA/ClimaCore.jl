## Packages that extend ClimaCore functionality

* [ClimaCoreTempestRemap](https://github.com/CliMA/ClimaCore.jl/tree/main/lib/ClimaCoreTempestRemap): Interface for using [TempestRemap](https://github.com/ClimateGlobalChange/tempestremap/) with ClimaCore.
* [ClimaCoreSpectra](https://github.com/CliMA/ClimaCore.jl/tree/main/lib/ClimaCoreSpectra): ClimaCore for spherical harmonic spectra

Field plotting is built into ClimaCore itself: the [Makie](https://makie.juliaplots.org/stable) recipes live in the `ClimaCoreMakieExt` extension and the [Plots](https://docs.juliaplots.org/latest/) recipes in the `ClimaCoreRecipesBaseExt` extension, both of which load automatically alongside their plotting package.
