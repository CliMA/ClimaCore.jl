module ClimaCoreTempestRemap

export write_exodus, rll_mesh, overlap_mesh, remap_weights, apply_remap
export def_time_coord, def_space_coord

# Keep in sync with definition in DataLayouts.
@inline slab_index(i, j) = CartesianIndex(i, j, 1, 1, 1)
@inline slab_index(i) = CartesianIndex(i, 1, 1, 1, 1)
@inline vindex(v) = CartesianIndex(1, 1, 1, v, 1)

using ClimaComms
import ClimaCore
using ClimaCore:
    Geometry, Meshes, Domains, Topologies, Spaces, Fields, Quadratures

using NCDatasets, Dates, PkgVersion, LinearAlgebra, TempestRemap_jll

include("netcdf.jl")
include("exodus.jl")
include("wrappers.jl")
include("onlineremap.jl")

end # module
