# Taken from: https://github.com/CliMA/ClimateMachine.jl/blob/master/test/Common/Spectra/runtests.jl

using Test, FFTW

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields

using ClimaCoreSpectra:
    compute_gaussian!,
    compute_legendre!,
    power_spectrum_1d,
    power_spectrum_2d,
    trans_grid_to_spherical!,
    compute_wave_numbers

# additional helper function for spherical harmonic spectrum tests
# Adapted from: https://github.com/CliMA/ClimateMachine.jl/blob/master/test/Common/Spectra/spherical_helper_test.jl

"""
    TransSphericalToGrid!(mesh, snm, FT)

Transforms a variable expressed in spherical harmonics (var_spherical[num_fourier+1, num_spherical+1]) onto a Gaussian grid (pfield[nλ, nθ])

[THIS IS USED FOR TESTING ONLY]

    With F_{m,n} = (-1)^m F_{-m,n}*
    P_{m,n} = (-1)^m P_{-m,n}

    F(λ, η) = ∑_{m= -N}^{N} ∑_{n=|m|}^{N} F_{m,n} P_{m,n}(η) e^{imλ}
    = ∑_{m= 0}^{N} ∑_{n=m}^{N} F_{m,n} P_{m,n} e^{imλ} + ∑_{m= 1}^{N} ∑_{n=m}^{N} F_{-m,n} P_{-m,n} e^{-imλ}

    Here η = sinθ, N = num_fourier, and denote
    ! extra coeffients in snm n > N are not used.

    ∑_{n=m}^{N} F_{m,n} P_{m,n}     = g_{m}(η) m = 1, ... N
    ∑_{n=m}^{N} F_{m,n} P_{m,n}/2.0 = g_{m}(η) m = 0

    We have

    F(λ, η) = ∑_{m= 0}^{N} g_{m}(η) e^{imλ} + ∑_{m= 0}^{N} g_{m}(η)* e^{-imλ}
    = 2real{ ∑_{m= 0}^{N} g_{m}(η) e^{imλ} }

    snm = F_{m,n}         # Complex{Float64} [num_fourier+1, num_spherical+1]
    qnm = P_{m,n,η}         # Float64[num_fourier+1, num_spherical+1, nθ]
    fourier_g = g_{m, η} # Complex{Float64} nλ×nθ with padded 0s fourier_g[num_fourier+2, :] == 0.0
    pfiled = F(λ, η)      # Float64[nλ, nθ]

    ! use all spherical harmonic modes


# Arguments
- mesh: struct with mesh information
- snm: spherical variable
- FT: FloatType

# References
- Ehrendorfer, M., Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics, 2011
"""
function trans_spherical_to_grid!(mesh, snm, FT)
    num_fourier, num_spherical = mesh.num_fourier, mesh.num_spherical
    nλ, nθ, nd = mesh.nλ, mesh.nθ, mesh.nd

    qnm = mesh.qnm

    fourier_g = mesh.var_fourier .* FT(0.0)
    fourier_s = mesh.var_fourier .* FT(0.0)

    @assert(nθ % 2 == 0)
    nθ_half = div(nθ, 2)
    for m in 1:(num_fourier + 1)
        for n in m:num_spherical
            snm_t = transpose(snm[m, n, :, 1:nθ_half]) #snm[m,n, :] is complex number
            if (n - m) % 2 == 0
                fourier_s[m, 1:nθ_half, :] .+=
                    qnm[m, n, 1:nθ_half] .* sum(snm_t, dims = 1)   #even function part
            else
                fourier_s[m, (nθ_half + 1):nθ, :] .+=
                    qnm[m, n, 1:nθ_half] .* sum(snm_t, dims = 1)   #odd function part
            end
        end
    end
    fourier_g[:, 1:nθ_half, :] .=
        fourier_s[:, 1:nθ_half, :] .+ fourier_s[:, (nθ_half + 1):nθ, :]
    fourier_g[:, nθ:-1:(nθ_half + 1), :] .=
        fourier_s[:, 1:nθ_half, :] .- fourier_s[:, (nθ_half + 1):nθ, :] # this got ignored...

    fourier_g[1, :, :] ./= FT(2.0)
    pfield = zeros(Float64, nλ, nθ, nd)
    for j in 1:nθ
        pfield[:, j, :] .= FT(2.0) * nλ * real.(ifft(fourier_g[:, j, :], 1)) #fourier for the first dimension
    end
    return pfield
end

# Set up function space
function sphere_3D(
    R = 6.37122e6,
    zlim = (0, 12.0e3),
    helem = 4,
    zelem = 30,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(
        vertdomain,
        Meshes.ExponentialStretching(FT(7e3));
        nelems = zelem,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (horzspace, hv_face_space)
end

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
        power_spectrum_1d(FT, var_grid, z, yarray, xarray, mass_weight)

    nm_spectrum_ = nm_spectrum[:, 10, 1]
    var_grid_ = var_grid[:, 10, 1]
    sum_spec = sum(nm_spectrum_)
    sum_grid = sum(var_grid_ .^ 2) / length(var_grid_)

    sum_res = (sum_spec - sum_grid) / sum_grid

    @test sum_res < FT(0.1)
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
    var_grid =
        FT(1.0) * reshape(
            sin.(xarray / xarray[end] * FT(3.0) * π) .* P_32',
            length(xarray),
            length(yarray),
            1,
        )

    mass_weight = ones(FT, z)
    spectrum, wave_numbers, spherical, mesh =
        power_spectrum_2d(FT, var_grid, mass_weight)

    # Grid to spherical to grid reconstruction
    reconstruction = trans_spherical_to_grid!(mesh, spherical, FT)

    sum_spec = sum((0.5 * spectrum))
    dθ = π / length(wts)
    area_factor = reshape(cosθ .* dθ .^ 2 / 4π, (1, length(cosθ)))

    sum_grid = sum(0.5 .* var_grid[:, :, 1] .^ 2 .* area_factor) # scaled to average over Earth's area (units: m2/s2)
    sum_reco = sum(0.5 .* reconstruction[:, :, 1] .^ 2 .* area_factor)

    sum_res_1 = (sum_spec - sum_grid) / sum_grid
    sum_res_2 = (sum_reco - sum_grid) / sum_grid

    @test abs(sum_res_1) < FT(0.1)
    @test abs(sum_res_2) < FT(0.1)
end
