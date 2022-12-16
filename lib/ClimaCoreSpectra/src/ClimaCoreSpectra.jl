module ClimaCoreSpectra

export compute_gaussian!, power_spectrum_1d, power_spectrum_2d

import ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields

include("gcm_spectra.jl")

end # module
