# Main implentation based on: https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Common/Spectra/power_spectrum_gcm.jl

#=
Cleanup items:
 - Can we use external packages (e.g., RootSolvers, AssociatedLegendrePolynomials)?
 - What CC API functions are needed?
    - Can we generalize the interface to use CC Fields to spare the user from having to do the remapping boilerplate?
=#

import FFTW

"""
    AbstractSpectralSphericalMesh{FT, ArrF3, ArrI2, ArrC3, ArrC4}

Supertype for spherical mesh data structures used to compute spectra. The only subtype
is [`SpectralSphericalMesh`](@ref).
"""
abstract type AbstractSpectralSphericalMesh{FT, ArrF3, ArrI2, ArrC3, ArrC4} end

"""
    SpectralSphericalMesh{FT}(nθ, nd)
    SpectralSphericalMesh(nθ, nd, ArrType, ComplexType, IntArrType)

Spherical mesh data structure for computing spectra on a regular latitude-longitude grid
with `nθ` latitudes, `nλ = 2nθ` longitudes, and `nd` vertical levels. The triangular
truncation is `num_fourier = floor((2nθ - 1) / 3)` (e.g. `nθ = 32` gives T21).

`ArrType`, `ComplexType`, and `IntArrType` are the array types used for the real,
complex, and integer work arrays; the first form uses `Array`s of `FT`.

# Fields

  - `num_fourier`: number of truncated zonal wavenumbers `m`.
  - `num_spherical`: number of total wavenumbers `n` (`num_fourier + 1`).
  - `nλ`, `nθ`, `nd`: numbers of longitudes, latitudes, and vertical levels.
  - `Δλ`: longitude spacing [rad].
  - `qwg`: Gaussian-weighted associated Legendre polynomials, indexed `[m, n, θ]`.
  - `qnm`: normalized associated Legendre polynomials, indexed `[m, n, θ]`.
  - `wave_numbers`: total wavenumber `n` for each `[m, n]` entry.
  - `var_grid`, `var_fourier`, `var_spherical`, `var_spectrum`: work arrays for the
    variable on the grid, after the Fourier transform, in spherical-harmonic space, and
    its power spectrum.
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

Compute the normalized associated Legendre polynomials ``P_{l,m}`` at the Gaussian
latitudes and return them as an array `qnm` of shape
`(num_fourier + 1, num_spherical + 1, nθ)`, with `qnm[m + 1, l + 1, :]` holding
``P_{l,m}``. The function allocates its result and mutates none of its arguments.

# Arguments

  - `FT`: float type.
  - `num_fourier`: number of truncated zonal wavenumbers `m`.
  - `num_spherical`: number of total wavenumbers `n`.
  - `sinθ`: array of `sin(latitude)` at the Gaussian latitudes.
  - `nθ`: number of Gaussian latitudes.

# Notes

Following the notation and equation numbers of Ehrendorfer (2011), Appendix B, with
`l = 0, 1, …` and `m = -l, …, l`:

    P_{0,0} = 1
    P_{m,m} = sqrt((2m+1)/2m) cosθ P_{m-1,m-1}
    P_{m+1,m} = sqrt(2m+3) sinθ P_{m,m}
    sqrt((l²-m²)/(4l²-1)) P_{l,m} = sinθ P_{l-1,m} - sqrt(((l-1)²-m²)/(4(l-1)²-1)) P_{l-2,m}

The normalization gives ``\\frac{1}{2} \\int_{-1}^1 P_{l,m}(x) P_{n,m}(x)\\, dx = δ_{n,l}``
with ``x = \\sin θ``.

References: Ehrendorfer, M. (2011), Spectral Numerical Weather Prediction Models,
Appendix B, SIAM; Winch, D. (2007), Spherical harmonics, in Encyclopedia of
Geomagnetism and Paleomagnetism, Springer.
"""
function compute_legendre!(FT, num_fourier, num_spherical, sinθ, nθ)

    # TODO:
    #  - Can we unify the interface with an external package that does this?
    qnm = zeros(FT, num_fourier + 1, num_spherical + 2, nθ)

    cosθ = sqrt.(1 .- sinθ .^ 2)
    ε = zeros(FT, num_fourier + 1, num_spherical + 2)

    qnm[1, 1, :] .= 1 # P_{0,0}
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

Compute `sin(latitude)` at the `n` Gaussian latitudes and the corresponding weights for
Gaussian integration, returned as the tuple `(sinθ, wts)` of arrays with element type
`FT`. `n` must be even. The function allocates its results and mutates none of its
arguments.

# Notes

The roots of the Legendre polynomial ``P_n`` are found by Newton iteration from the
initial guess ``x_i = \\cos(π(i - 1/4)/(n + 1/2))``, using the recurrences
``n P_n(x) = (2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)`` and
``P'_n(x) = \\frac{n}{x^2 - 1}(x P_n(x) - P_{n-1}(x))``; since ``P_n`` is odd, only half
of the roots are computed. The weights are ``w_i = 2 / ((1 - x_i^2) P'_n(x_i)^2)``. An
error is logged if the iteration does not converge. See Ehrendorfer, M. (2011), Spectral
Numerical Weather Prediction Models, Appendix B, SIAM.
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
    trans_grid_to_spherical!(mesh::SpectralSphericalMesh, pfield::AbstractArray)

Transform the variable `pfield` of shape `(nλ, nθ)` on a Gaussian grid into
spherical-harmonic space and return the complex coefficient array of shape
`(num_fourier + 1, num_spherical + 1, nθ ÷ 2)`, split by latitude hemisphere pairs.

The transform is a Fourier transform along each latitude circle followed by a Legendre
transform using the weighted polynomials `mesh.qwg`. `mesh` is read but not mutated;
`nθ` must be even.

# Arguments

  - `mesh`: mesh information and weighted Legendre polynomials.
  - `pfield`: variable on the Gaussian grid to be transformed.

# Notes

With λ the longitude, θ the latitude, η = sin θ, `m` the zonal wavenumber, and `n` the
total wavenumber:

    var_spherical2d = F_{m,n}    # output in spectral space
    qwg = P_{m,n}(η) w(η)        # weighted Legendre polynomials
    var_fourier2d = g_{m,θ}      # untruncated Fourier transform
    pfield = F(λ, η)             # input on the Gaussian grid

See Ehrendorfer, M. (2011), Spectral Numerical Weather Prediction Models, Appendix B,
SIAM, and [Wiin1967](@cite).
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

Store the total wavenumber `n` for each `(m, n)` entry of the triangular truncation in
the matrix `wave_numbers`, which is mutated in place. Entries with `n < m` are left
unchanged. Returns `nothing`.

# Arguments

  - `wave_numbers`: integer matrix of shape `(num_fourier + 1, num_spherical + 1)`.
  - `num_fourier`: number of truncated zonal wavenumbers `m`.
  - `num_spherical`: number of total wavenumbers `n`.
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

Compute the zonal (1D) power spectrum of the variable `var_grid` on a `(lon, lat, z)`
grid with a Fourier transform along each latitude circle, weighting each level by
`weight`. The input field must first be interpolated to a regular latitude-longitude
grid.

# Arguments

  - `FT`: float type.
  - `var_grid`: variable on the `(lon, lat, z)` grid to be transformed.
  - `z`: array of vertical levels.
  - `lat`: array of latitudes [degrees].
  - `lon`: array of uniformly spaced longitudes [degrees].
  - `weight`: array with one weight per level, e.g. for mass weighting.

# Returns

The tuple `(zon_spectrum, freqs)` of arrays of shape `(num_pfourier, nlat, nlev)`, where
`num_pfourier` is the number of non-negative Fourier frequencies: the power at each
frequency (with the negative-frequency contribution folded in) and the corresponding
angular wavenumbers.
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

Transform the variable `var_grid` on a regular latitude-longitude grid into
spherical-harmonic space, using an FFT along latitude circles (as for the 1D spectrum)
and a Legendre transform along meridians, and compute its 2D power spectrum.

# Arguments

  - `FT`: float type.
  - `var_grid`: variable on the `(lon, lat, z)` grid to be transformed, with `nlon = 2 nlat`.
  - `mass_weight`: array with one weight per level, e.g. for mass weighting.

# Returns

The tuple `(var_spectrum, wave_numbers, var_spherical, mesh)`: the power spectrum indexed
`[m, n, k]`, the total wavenumber of each `[m, n]` entry, the spherical-harmonic
coefficients indexed `[m, n, k, θ]`, and the [`SpectralSphericalMesh`](@ref) used.

See [Baer1972](@cite).
"""
function power_spectrum_2d(FT, var_grid, mass_weight)

    # TODO:
    #  - Can we define
    #    `power_spectrum_2d(field::ClimaCore.Field, mass_weight::ClimaCore.Field)`
    #  - Call ClimaCoreTempestRemap internally to export lat-lon grid
    #  - ClimaCoreSpectra can then take this output and compute the spectra

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
