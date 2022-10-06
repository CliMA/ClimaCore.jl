# ClimaCoreSpectra.jl

```@docs
ClimaCoreSpectra.power_spectrum_1d
ClimaCoreSpectra.compute_gaussian!
ClimaCoreSpectra.SpectralSphericalMesh
ClimaCoreSpectra.compute_legendre!
ClimaCoreSpectra.power_spectrum_2d
ClimaCoreSpectra.trans_grid_to_spherical!
ClimaCoreSpectra.compute_wave_numbers!
```

```@example
import ClimaCore
fn = joinpath(pkgdir(ClimaCore), "lib", "ClimaCoreSpectra", "test", "gcm_visual_test.jl")
@show fn
include(fn)
```

## 1D Spectrum Test
### Input wave frequency
![](1D_spectrum_vs_wave_numbers_plot.png)

### Raw data on rll grid
![](1D_raw_data_plot.png)
### 1D Spectrum
![](1D_spectrum.png)
## 2D Spectrum Test

### Raw data on rll grid
![](2d_raw_data_plot.png)
## Reconstruct onto spectral space and transform back to original
![](2d_transformed.png)
## Error
![](error.png)
## 2D Spectra
![](2d_spectra.png)
