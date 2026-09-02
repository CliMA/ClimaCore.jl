# ClimaCoreSpectra.jl

```@meta
CurrentModule = ClimaCoreSpectra
```

ClimaCoreSpectra.jl provides functionality for calculating kinetic energy spectra using spherical harmonics.

## Interface

```@docs
ClimaCoreSpectra.SpectralSphericalMesh
ClimaCoreSpectra.power_spectrum_1d
ClimaCoreSpectra.power_spectrum_2d
ClimaCoreSpectra.compute_gaussian!
ClimaCoreSpectra.compute_legendre!
ClimaCoreSpectra.trans_grid_to_spherical!
ClimaCoreSpectra.compute_wave_numbers!
```

## Examples

`lib/ClimaCoreSpectra/test/gcm_visual_test.jl` computes the one- and
two-dimensional spectra of test fields on a latitude–longitude grid, transforms
them back, and plots the input, the spectra, and the reconstruction error. Run
it from the repository root with `BUILD_DOCS=true` set to write the figures:

```julia
import ClimaCore
fn = joinpath(pkgdir(ClimaCore), "lib", "ClimaCoreSpectra", "test", "gcm_visual_test.jl")
ENV["BUILD_DOCS"] = true
include(fn)
```
