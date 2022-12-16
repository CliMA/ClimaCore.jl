# ClimaCoreSpectra.jl

```@meta
CurrentModule = ClimaCoreSpectra
```

ClimaCoreSpectra.jl provides functionality for calculating kinetic energy spectra using spherical harmonics.

# Interface

```@docs
ClimaCoreSpectra.SpectralSphericalMesh
ClimaCoreSpectra.power_spectrum_1d
ClimaCoreSpectra.power_spectrum_2d
ClimaCoreSpectra.compute_gaussian!
ClimaCoreSpectra.compute_legendre!
ClimaCoreSpectra.trans_grid_to_spherical!
ClimaCoreSpectra.compute_wave_numbers!
```

# Examples

```@example
import ClimaCore
fn = joinpath(pkgdir(ClimaCore), "lib", "ClimaCoreSpectra", "test", "gcm_visual_test.jl")
@show fn
ENV["BUILD_DOCS"]=true
include(fn)
```

## 1D Spectrum Test
### Input wave frequency
![A 1D wave.](1D_spectrum_vs_wave_numbers_plot.png)

### Raw data on rll grid
![The 1D raw data to be transformed.](1D_raw_data_plot.png)
### 1D Spectrum
![The 1D spectrum calculated from the data.](1D_spectrum.png)
## 2D Spectrum Test

### Raw data on rll grid
![The 2D raw data to be transformed.](2d_raw_data_plot.png)
## Reconstruct onto spectral space and transform back to original
![The 2D transformed data.](2d_transformed.png)
## Error
![The error between the transformed and initial data.](error.png)
## 2D Spectra
![The 2D spectrum calculated from the data.](2d_spectra.png)
