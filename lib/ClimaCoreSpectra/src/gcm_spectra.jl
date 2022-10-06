# Main implentation based on: https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Common/Spectra/power_spectrum_gcm.jl

#=
Cleanup items:

 - Do we need this to be allocation-free?
 - Can we use external packages (e.g., RootSolvers, AssociatedLegendrePolynomials)?
 - What CC API functions are needed?
    - Can we generalize the interface to use CC Fields to spare the user from having to do the remapping boilerplate?
=#

import FFTW

"""
    AbstractSpectralSphericalMesh

An abstract spherical mesh data structure for calculating spectra.
"""
abstract type AbstractSpectralSphericalMesh{FT, ArrF3, ArrI2, ArrC3, ArrC4} end

"""
    SpectralSphericalMesh

Spherical mesh data structure for calculating spectra. The mesh represents a regular lat-long grid.
"""
mutable struct SpectralSphericalMesh{FT, ArrF3, ArrI2, ArrC3, ArrC4} <:
               AbstractSpectralSphericalMesh{FT, ArrF3, ArrI2, ArrC3, ArrC4}
    # grid info
    num_fourier::Int
    num_spherical::Int
    nλ::Int
    nθ::Int
    nd::Int
    Δλ::FT
    qwg::ArrF3
    qnm::ArrF3   # n,m coordinates
    wave_numbers::ArrI2

    # variables
    var_grid::ArrF3
    var_fourier::ArrC3
    var_spherical::ArrC4
    var_spectrum::ArrF3
end

SpectralSphericalMesh{FT}(nθ::Int, nd::Int) where {FT} =
    SpectralSphericalMesh(nθ, nd, Array{FT}, Array{Complex{FT}}, Array{Int})

function SpectralSphericalMesh(
    nθ::Int,
    nd::Int,
    ::Type{ArrType},
    ::Type{ComplexType},
    ::Type{IntArrType},
) where {ArrType, ComplexType, IntArrType}
    nλ = 2nθ
    Δλ = 2π / nλ

    num_fourier = Int(floor((2 * nθ - 1) / 3)) # number of truncated zonal wavenumbers (m): minimum truncation given nθ - e.g.: nlat = 32 -> T21 (can change manually for more a severe truncation)
    num_spherical = Int(num_fourier + 1) # number of total wavenumbers (n)

    wave_numbers = IntArrType(undef, num_fourier + 1, num_spherical + 1)
    fill!(wave_numbers, 0)
    compute_wave_numbers!(wave_numbers, num_fourier, num_spherical)

    qwg = ArrType(undef, num_fourier + 1, num_spherical + 1, nθ)
    fill!(qwg, 0)
    qnm = ArrType(undef, num_fourier + 1, num_spherical + 2, nθ)
    fill!(qnm, 0)

    var_fourier = ComplexType(undef, nλ, nθ, nd)
    fill!(var_fourier, 0)
    var_grid = ArrType(undef, nλ, nθ, nd)
    fill!(var_grid, 0)
    nθ_half = div(nθ, 2)
    var_spherical =
        ComplexType(undef, num_fourier + 1, num_spherical + 1, nd, nθ_half)
    fill!(var_spherical, 0)
    var_spectrum = ArrType(undef, num_fourier + 1, num_spherical + 1, nd)
    fill!(var_spectrum, 0)

    SpectralSphericalMesh(
        num_fourier,
        num_spherical,
        nλ,
        nθ,
        nd,
        Δλ,
        qwg,
        qnm,
        wave_numbers,
        var_grid,
        var_fourier,
        var_spherical,
        var_spectrum,
    )
end

# Helper functions

"""
    compute_legendre!(FT, num_fourier, num_spherical, sinθ, nθ)

Normalized associated Legendre polynomials, P_{m,l} = qnm.

# Arguments:
- FT: FloatType
- num_fourier
- num_spherical
- sinθ
- nθ

# References:
- Ehrendorfer, M. (2011) Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics
- Winch, D. (2007) Spherical harmonics, in Encyclopedia of Geomagnetism and Paleomagnetism, Eds Gubbins D. and Herrero-Bervera, E., Springer

# Details (using notation and Eq. references from Ehrendorfer, 2011):
    l=0,1...∞    and m = -l, -l+1, ... l-1, l
    P_{0,0} = 1, such that 1/4π ∫∫YYdS = δ (where Y = spherical harmonics, S = domain surface area)
    P_{m,m} = sqrt((2m+1)/2m) cosθ P_{m-1m-1}
    P_{m+1,m} = sqrt(2m+3) sinθ P_{m m}
    sqrt((l^2-m^2)/(4l^2-1))P_{l,m} = P_{l-1, m} -  sqrt(((l-1)^2-m^2)/(4(l-1)^2 - 1))P_{l-2,m}
    THe normalization assures that 1/2 ∫_{-1}^1 P_{l,m}(sinθ) P_{n,m}(sinθ) dsinθ = δ_{n,l}
    Julia index starts with 1, so qnm[m+1,l+1] = P_l^m

TODO:
 - Can we unify the interface with an external package that does this?
"""
function compute_legendre!(FT, num_fourier, num_spherical, sinθ, nθ)
    qnm = zeros(FT, num_fourier + 1, num_spherical + 2, nθ)

    cosθ = sqrt.(1 .- sinθ .^ 2)
    ε = zeros(FT, num_fourier + 1, num_spherical + 2)

    qnm[1, 1, :] .= 1
    for m in 1:num_fourier
        qnm[m + 1, m + 1, :] = -sqrt((2m + 1) / (2m)) .* cosθ .* qnm[m, m, :] # Eq. B.20
        qnm[m, m + 1, :] = sqrt(2m + 1) * sinθ .* qnm[m, m, :] # Eq. B.22
    end
    qnm[num_fourier + 1, num_fourier + 2, :] =
        sqrt(2 * (num_fourier + 2)) * sinθ .*
        qnm[num_fourier + 1, num_fourier + 1, :]

    for m in 0:num_fourier
        for l in (m + 2):(num_spherical + 1)
            ε1 = sqrt(((l - 1)^2 - m^2) ./ (4 * (l - 1)^2 - 1))
            ε2 = sqrt((l^2 - m^2) ./ (4 * l^2 - 1))
            qnm[m + 1, l + 1, :] =
                (sinθ .* qnm[m + 1, l, :] - ε1 * qnm[m + 1, l - 1, :]) / ε2 # Eq. B.18
        end
    end

    return qnm[:, 1:(num_spherical + 1), :]
end

"""
    compute_gaussian!(FT, n)

Compute sin(latitude) and the weight factors for Gaussian integration.

# Arguments
- FT: FloatType
- n: number of Gaussian latitudes

# References
- Ehrendorfer, M., Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics, 2011
# Details (following notation from Ehrendorfer, 2011):
    Pn(x) is an odd function
    solve half of the n roots and weightes of Pn(x) # n = 2n_half
    P_{-1}(x) = 0
    P_0(x) = 1
    P_1(x) = x
    nP_n(x) = (2n-1)xP_{n-1}(x) - (n-1)P_{n-2}(x)
    P'_n(x) = n/(x^2-1)(xP_{n}(x) - P_{n-1}(x))
    x -= P_n(x)/P'_{n}()
    Initial guess xi^{0} = cos(π(i-0.25)/(n+0.5))
    wi = 2/(1-xi^2)/P_n'(xi)^2
"""
function compute_gaussian!(FT, n)
    itermax = 10000
    tol = 1.0e-15

    sinθ = zeros(FT, n)
    wts = zeros(FT, n)

    n_half = Int(n / 2)
    n_plus_half = FT(n + 0.5)
    for i in 1:n_half
        dp = 0.0
        z = cos(pi * (i - 0.25) / n_plus_half)
        for iter in 1:itermax
            p2 = 0.0
            p1 = 1.0

            for j in 1:n
                p3 = p2 # Pj-2
                p2 = p1 # Pj-1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j  #Pj
            end
            # P'_n
            dp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / dp
            if (abs(z - z1) <= tol)
                break
            end
            if iter == itermax
                @error("Compute_Gaussian! does not converge!")
            end
        end

        sinθ[i], sinθ[n - i + 1], = -z, z
        wts[i] = wts[n - i + 1] = 2.0 / ((1.0 - z * z) * dp * dp)
    end

    return sinθ, wts
end

"""
    trans_grid_to_spherical!(mesh::SpectralSphericalMesh, pfield::Arr{FT,2})

Transforms a variable on a Gaussian grid (pfield[nλ, nθ]) into the spherical harmonics domain (var_spherical2d[num_fourier+1, num_spherical+1])
Here λ = longitude, θ = latitude, η = sinθ, m = zonal wavenumber, n = total wavenumber:
var_spherical2d = F_{m,n}    # Output variable in spectral space (Complex{FT}[num_fourier+1, num_spherical+1])
qwg = P_{m,n}(η)w(η)         # Weighted Legendre polynomials (FT[num_fourier+1, num_spherical+1, nθ])
var_fourier2d = g_{m, θ}     # Untruncated Fourier transformation (Complex{FT} [nλ, nθ])
pfield = F(λ, η)             # Input variable on Gaussian grid FT[nλ, nθ]

# Arguments
- mesh: struct with mesh information
- pfield: variable on Gaussian grid to be transformed

# References
- Ehrendorfer, M., Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics, 2011
- [Wiin1967](@cite)
"""
function trans_grid_to_spherical!(
    mesh::SpectralSphericalMesh{FT},
    pfield::AbstractArray,
) where {FT}

    num_fourier, num_spherical = mesh.num_fourier, mesh.num_spherical
    var_fourier2d, var_spherical2d =
        mesh.var_fourier[:, :, 1] * 0, mesh.var_spherical[:, :, 1, :] * 0
    nλ, nθ, nd = mesh.nλ, mesh.nθ, mesh.nd

    # Retrieve weighted Legendre polynomials
    qwg = mesh.qwg # qwg[m,n,nθ]

    # Fourier transformation
    for j in 1:nθ
        var_fourier2d[:, j] = FFTW.fft(pfield[:, j], 1) / nλ
    end

    # Complete spherical harmonic transformation
    @assert(nθ % 2 == 0)
    nθ_half = div(nθ, 2)
    for m in 1:(num_fourier + 1)
        for n in m:num_spherical
            var_fourier2d_t = transpose(var_fourier2d[m, :])  # truncates var_fourier(nlon, nhlat) to (nfourier,nlat)
            if (n - m) % 2 == 0
                var_spherical2d[m, n, :] .=
                    (
                        var_fourier2d_t[1:nθ_half] .+
                        var_fourier2d_t[nθ:-1:(nθ_half + 1)]
                    ) .* qwg[m, n, 1:nθ_half] ./ 2
            else
                var_spherical2d[m, n, :] .=
                    (
                        var_fourier2d_t[1:nθ_half] .-
                        var_fourier2d_t[nθ:-1:(nθ_half + 1)]
                    ) .* qwg[m, n, 1:nθ_half] ./ 2
            end
        end
    end

    return var_spherical2d
end

"""
    compute_wave_numbers!(wave_numbers, num_fourier::Int, num_spherical::Int)

Set wave_numers[i,j] saves the wave number of this basis
"""
function compute_wave_numbers!(
    wave_numbers,
    num_fourier::Int,
    num_spherical::Int,
)

    for m in 0:num_fourier
        for n in m:num_spherical
            wave_numbers[m + 1, n + 1] = n
        end
    end

end

# Power spectrum 1D

"""
    power_spectrum_1d(FT, var_grid, z, lat, lon, weight)

For a variable `var_grid` on a (lon,lat,z) grid, given an array of
`weight`s, compute the zonal (1D) power spectrum using a Fourier
transform at each latitude, from a velocity field. The input velocities
must be intepolated to a Gaussian grid.

# Arguments
- FT: FloatType
- var_grid: variable (typically u or v) on a Gaussian (lon, lat, z) grid to be transformed
- z: Array with uniform z levels
- lat: Array with uniform lats
- lon: Array with uniform longs
- weight: Array with weights for mass-weighted calculations

"""
function power_spectrum_1d(FT, var_grid, z, lat, lon, weight)
    num_lev = length(z)
    num_lat = length(lat)
    num_lon = length(lon)
    num_fourier = Int(num_lon)

    # get number of positive Fourier coefficients incl. 0
    if mod(num_lon, 2) == 0 # even
        num_pfourier = div(num_lon, 2)
    else # odd
        num_pfourier = div(num_lon, 2) + 1
    end

    zon_spectrum = zeros(FT, num_pfourier, num_lat, num_lev)
    freqs = zeros(FT, num_pfourier, num_lat, num_lev)

    for k in 1:num_lev
        for j in 1:num_lat
            # compute fft frequencies for each latitude
            x = lon ./ 180 .* π
            dx = (lon[2] - lon[1]) ./ 180 .* π

            freqs_ = FFTW.fftfreq(num_fourier, 1.0 / dx) # 0,+ve freq,-ve freqs (lowest to highest)
            freqs[:, j, k] = freqs_[1:num_pfourier] .* 2.0 .* π

            # compute the fourier coefficients for all latitudes
            fourier = FFTW.fft(var_grid[:, j, k]) # e.g. vcos_grid, ucos_grid
            fourier = (fourier / num_fourier)

            # convert to energy spectra
            zon_spectrum[1, j, k] =
                zon_spectrum[1, j, k] +
                weight[k] * fourier[1] .* conj(fourier[1])

            for m in 2:num_pfourier
                zon_spectrum[m, j, k] =
                    zon_spectrum[m, j, k] +
                    2 * weight[k] * fourier[m] * conj(fourier[m]) # factor 2 for neg freq contribution
            end
        end
    end
    return zon_spectrum, freqs
end

# Power spectrum 2D

"""
    power_spectrum_2d(FT, var_grid, mass_weight)

- transform variable on grid to the 2d spectral space using fft on latitude circles
(as for the 1D spectrum) and Legendre polynomials for meridians, and calculate spectra.

# Arguments
- FT: FloatType
- var_grid: variable (typically u or v) on a Gaussian (lon, lat, z) grid to be transformed
- mass_weight: Array with weights for mass-weighted calculations
# References
- [Baer1972](@cite)

TODO:
 - Can we define `power_spectrum_2d(field::ClimaCore.Field, mass_weight::ClimaCore.Field)`
 - Call ClimaCoreTempestRemap to export lat-lon grid
 - ClimaCoreSpectra can then take this output and compute the spectra
"""
function power_spectrum_2d(FT, var_grid, mass_weight)
    #  initialize spherical mesh variables
    nθ, nd = (size(var_grid, 2), size(var_grid, 3))
    mesh = SpectralSphericalMesh{FT}(nθ, nd)
    var_spectrum = mesh.var_spectrum
    var_spherical = mesh.var_spherical

    sinθ, wts = compute_gaussian!(FT, mesh.nθ) # latitude weights using Gaussian quadrature, to orthogonalize Legendre polynomials upon summation
    mesh.qnm = compute_legendre!(
        FT,
        mesh.num_fourier,
        mesh.num_spherical,
        sinθ,
        mesh.nθ,
    ) #  normalized associated Legendre polynomials

    for k in 1:(mesh.nd)
        # apply Gaussian quadrature weights
        for i in 1:(mesh.nθ)
            mesh.qwg[:, :, i] .= mesh.qnm[:, :, i] * wts[i] * mass_weight[k]
        end

        # Transform variable using spherical harmonics
        var_spherical[:, :, k, :] =
            trans_grid_to_spherical!(mesh, var_grid[:, :, k]) # var_spherical[m,n,k,sinθ]

        # Calculate energy spectra
        var_spectrum[:, :, k] =
            2 .* sum(var_spherical[:, :, k, :], dims = 3) .*
            conj(sum(var_spherical[:, :, k, :], dims = 3))  # var_spectrum[m,n,k] # factor 2 to account for negative Fourier frequencies
        var_spectrum[1, :, k] = var_spectrum[1, :, k] ./ 2 # m=0
    end
    return var_spectrum, mesh.wave_numbers, var_spherical, mesh
end
