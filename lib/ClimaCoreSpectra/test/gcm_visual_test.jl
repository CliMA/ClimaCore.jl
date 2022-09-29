# Standalone test file that tests spectra visually
# Taken from https://github.com/CliMA/ClimateMachine.jl/blob/master/test/Common/Spectra/gcm_standalone_visual_test.jl
using Plots

using ClimaCoreSpectra:
    compute_gaussian!,
    compute_legendre!,
    SpectralSphericalMesh,
    trans_grid_to_spherical!,
    power_spectrum_1d,
    power_spectrum_2d,
    compute_wave_numbers
using FFTW


include("spherical_helper.jl")

FT = Float64
# -- TEST 1: power_spectrum_1d
n_gauss_lats = 32

# Setup grid
sinθ, wts = compute_gaussian!(n_gauss_lats)
yarray = asin.(sinθ) .* FT(180) / π
xarray =
    FT(180.0) ./ n_gauss_lats * collect(FT, 1:1:(2n_gauss_lats))[:] .- FT(180.0)
z = 1

# Setup variable
mass_weight = ones(FT, length(z));
var_grid =
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
    power_spectrum_1d(var_grid, z, yarray, xarray, mass_weight);


# Check visually
plot(wave_numbers[:, 16, 1], nm_spectrum[:, 16, 1], xlims = (0, 20))
contourf(var_grid[:, :, 1])
contourf(nm_spectrum[2:20, :, 1])

# -- TEST 2: power_spectrum_2d
# Setup grid
sinθ, wts = compute_gaussian!(n_gauss_lats)
yarray = asin.(sinθ) .* FT(180) / π
xarray =
    FT(180.0) ./ n_gauss_lats * collect(FT, 1:1:(2n_gauss_lats))[:] .- FT(180.0)
z = 1 # vertical levels: only one for sphere surface

# Setup variable: use an example analytical P_nm function
P_32 = sqrt(FT(105 / 8)) * (sinθ .- sinθ .^ 3)
var_grid =
    FT(1.0) * reshape(
        sin.(xarray / xarray[end] * FT(3.0) * π) .* P_32',
        length(xarray),
        length(yarray),
        1,
    )

mass_weight = ones(FT, z);
spectrum, wave_numbers, spherical, mesh =
    power_spectrum_2d(var_grid, mass_weight)

# Grid to spherical to grid reconstruction
reconstruction = trans_spherical_to_grid!(mesh, spherical)

# Check visually
contourf(var_grid[:, :, 1])
contourf(reconstruction[:, :, 1])
contourf(var_grid[:, :, 1] .- reconstruction[:, :, 1])

# Spectrum
contourf(
    collect(0:1:(mesh.num_fourier - 1))[:],
    collect(0:1:(mesh.num_spherical - 1))[:],
    (spectrum[:, :, 1])',
    xlabel = "m",
    ylabel = "n",
)

# Check magnitude
println(FT(0.5) .* sum(spectrum))

dθ = π / length(wts)
cosθ = sqrt.(FT(1) .- sinθ .^ 2)
area_factor = reshape(cosθ .* dθ .^ 2 / 4π, (1, length(cosθ)))

println(sum(0.5 .* var_grid[:, :, 1] .^ 2 .* area_factor))
