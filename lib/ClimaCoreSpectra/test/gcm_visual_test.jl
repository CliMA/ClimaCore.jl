# Standalone test file that tests spectra visually.
# Taken from https://github.com/CliMA/ClimateMachine.jl/blob/master/test/Common/Spectra/gcm_standalone_visual_test.jl

import Plots

OUTPUT_DIR =
    haskey(ENV, "BUILD_DOCS") ? "" :
    mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

import ClimaCoreSpectra: compute_gaussian!, power_spectrum_1d, power_spectrum_2d
using FFTW

# Additional helper function for spherical harmonic spectrum tests.
# Adapted from: https://github.com/CliMA/ClimateMachine.jl/blob/master/test/Common/Spectra/spherical_helper_test.jl
include(joinpath(@__DIR__, "spherical_helper.jl"))

FT = Float64
# -- TEST 1: power_spectrum_1d
n_gauss_lats = 32

# Setup grid
sinθ, wts = compute_gaussian!(FT, n_gauss_lats)
yarray = asin.(sinθ) .* FT(180) / π
xarray =
    FT(180.0) ./ n_gauss_lats * collect(FT, 1:1:(2n_gauss_lats))[:] .- FT(180.0)
z = 1

# Setup variables
mass_weight = ones(FT, length(z));
rll_grid_variable =
    FT(1.0) * reshape(
        sin.(xarray / xarray[end] * FT(5.0) * 2π) .*
        (yarray .* FT(0.0) .+ FT(1.0))',
        length(xarray),
        length(yarray),
        1,
    ) +
    FT(1.0) * reshape(
        sin.(xarray / xarray[end] * FT(10.0) * 2π) .*
        (yarray .* FT(0.0) .+ FT(1.0))',
        length(xarray),
        length(yarray),
        1,
    )
nm_spectrum, wave_numbers =
    power_spectrum_1d(FT, rll_grid_variable, z, yarray, xarray, mass_weight);


# Check visually
Plots.plot(
    wave_numbers[:, 16, 1],
    nm_spectrum[:, 16, 1],
    xlims = (0, 20),
    xaxis = "wavenumber",
    yaxis = "nm spectrum",
)
Plots.savefig(joinpath(OUTPUT_DIR, "1D_spectrum_vs_wave_numbers_plot.png"))
Plots.contourf(rll_grid_variable[:, :, 1], c = :roma)
Plots.savefig(joinpath(OUTPUT_DIR, "1D_raw_data_plot.png"))
Plots.contourf(nm_spectrum[2:20, :, 1], c = :roma)
Plots.savefig(joinpath(OUTPUT_DIR, "1D_spectrum.png"))

# -- TEST 2: power_spectrum_2d
# Setup grid
sinθ, wts = compute_gaussian!(FT, n_gauss_lats)
yarray = asin.(sinθ) .* FT(180) / π
xarray =
    FT(180.0) ./ n_gauss_lats * collect(FT, 1:1:(2n_gauss_lats))[:] .- FT(180.0)
z = 1 # vertical levels: only one for sphere surface

# Setup variable: use an example analytical P_nm function
P_32 = sqrt(FT(105 / 8)) * (sinθ .- sinθ .^ 3) # degree 3, order 2 associated legendre polynomial
rll_grid_variable =
    FT(1.0) * reshape(
        sin.(xarray / xarray[end] * FT(3.0) * π) .* P_32',
        length(xarray),
        length(yarray),
        1,
    )

mass_weight = ones(FT, z);
spectrum, wave_numbers, spherical, mesh_info =
    power_spectrum_2d(FT, rll_grid_variable, mass_weight)

# Grid to spherical to grid reconstruction
reconstruction = trans_spherical_to_grid!(mesh_info, spherical, FT)

# Check visually
Plots.contourf(rll_grid_variable[:, :, 1], c = :roma)
Plots.savefig(joinpath(OUTPUT_DIR, "2d_raw_data_plot.png"))
Plots.contourf(reconstruction[:, :, 1], c = :roma)
Plots.savefig(joinpath(OUTPUT_DIR, "2d_transformed.png"))
Plots.contourf(rll_grid_variable[:, :, 1] .- reconstruction[:, :, 1], c = :roma)
Plots.savefig(joinpath(OUTPUT_DIR, "error.png"))

# Spectrum
Plots.contourf(
    collect(0:1:(mesh_info.num_fourier))[:],
    collect(0:1:(mesh_info.num_spherical - 1))[:],
    (spectrum[:, 2:end, 1])',
    xlabel = "m",
    ylabel = "n",
    clim = (0, 0.25),
    c = :roma, # this palette was tested for color-blindness safety using the online simulator https://www.color-blindness.com/coblis-color-blindness-simulator/
)

Plots.savefig(joinpath(OUTPUT_DIR, "2d_spectra.png"))

# Check magnitude
println(FT(0.5) .* sum(spectrum))

dθ = π / length(wts)
cosθ = sqrt.(FT(1) .- sinθ .^ 2)
area_factor = reshape(cosθ .* dθ .^ 2 / 4π, (1, length(cosθ)))

println(sum(0.5 .* rll_grid_variable[:, :, 1] .^ 2 .* area_factor))
