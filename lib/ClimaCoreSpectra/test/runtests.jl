# Taken from: https://github.com/CliMA/ClimateMachine.jl/blob/master/test/Common/Spectra/runtests.jl

using Test, FFTW

using ClimaCoreSpectra:
    compute_gaussian!,
    compute_legendre!,
    power_spectrum_1d,
    power_spectrum_2d,
    trans_grid_to_spherical!,
    compute_wave_numbers!

# additional helper function for spherical harmonic spectrum tests
# Adapted from: https://github.com/CliMA/ClimateMachine.jl/blob/master/test/Common/Spectra/spherical_helper_test.jl

include(joinpath(@__DIR__, "spherical_helper.jl"))

@testset "power_spectrum_1d (GCM)" begin
    FT = Float64
    # -- TEST 1: power_spectrum_1d

    n_gauss_lats = 32

    # Setup grid
    sinθ, wts = compute_gaussian!(FT, n_gauss_lats)
    yarray = asin.(sinθ) .* FT(180) / π
    xarray =
        FT(180.0) ./ n_gauss_lats * collect(FT, 1:(2n_gauss_lats))[:] .-
        FT(180.0)
    z = 1 # vertical levels, only 1 for sphere surface

    # Setup variable
    mass_weight = ones(FT, length(z))
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
        power_spectrum_1d(FT, rll_grid_variable, z, yarray, xarray, mass_weight)

    nm_spectrum_ = nm_spectrum[:, 10, 1]
    var_grid_ = rll_grid_variable[:, 10, 1]
    sum_spec = sum(nm_spectrum_)
    sum_grid = sum(var_grid_ .^ 2) / length(var_grid_)

    sum_res = (sum_spec - sum_grid) / sum_grid

    @test abs(sum_res) ≈ 0 atol = eps(FT)
end

@testset "power_spectrum_2d (GCM)" begin
    # -- TEST 2: power_spectrum_2d
    # Setup grid
    FT = Float64
    n_gauss_lats = 32
    sinθ, wts = compute_gaussian!(FT, n_gauss_lats)
    cosθ = sqrt.(1 .- sinθ .^ 2)
    yarray = asin.(sinθ) .* FT(180) / π
    xarray =
        FT(180.0) ./ n_gauss_lats * collect(FT, 1:(2n_gauss_lats))[:] .-
        FT(180.0)
    z = 1 # vertical levels, only 1 for sphere surface

    # Setup variable: use an example analytical P_nm function
    P_32 = sqrt(105 / 8) * (sinθ .- sinθ .^ 3)
    rll_grid_variable =
        FT(1.0) * reshape(
            sin.(xarray / xarray[end] * FT(3.0) * π) .* P_32',
            length(xarray),
            length(yarray),
            1,
        )

    mass_weight = ones(FT, z)
    spectrum, wave_numbers, spherical, mesh_info =
        power_spectrum_2d(FT, rll_grid_variable, mass_weight)

    # Grid to spherical to grid reconstruction
    reconstruction = trans_spherical_to_grid!(mesh_info, spherical, FT)

    sum_spec = sum((0.5 * spectrum))
    dθ = π / length(wts)
    area_factor = reshape(cosθ .* dθ .^ 2 / 4π, (1, length(cosθ)))

    sum_grid = sum(0.5 .* rll_grid_variable[:, :, 1] .^ 2 .* area_factor) # scaled to average over Earth's area (units: m2/s2)
    sum_reco = sum(0.5 .* reconstruction[:, :, 1] .^ 2 .* area_factor)

    sum_res_1 = (sum_spec - sum_grid) / sum_grid
    sum_res_2 = (sum_reco - sum_grid) / sum_grid

    @test abs(sum_res_1) ≈ 0 atol = 2e-2
    @test abs(sum_res_2) ≈ 0 atol = 6e-6
end
