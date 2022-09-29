module ClimaCoreSpectra

export compute_gaussian!,
    compute_legendre!,
    SpectralSphericalMesh,
    trans_grid_to_spherical!,
    power_spectrum_1d,
    power_spectrum_2d,
    compute_wave_numbers

import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields

using AssociatedLegendrePolynomials, FFTW

include("gcm_spectra.jl")

end # module
