module ClimaCoreTempestRemap

export write_exodus, rll_mesh, overlap_mesh, remap_weights, apply_remap
export def_time_coord, def_space_coord

# Keep in sync with definition in DataLayouts.
@inline slab_index(i::T, j::T) where {T} =
    CartesianIndex(i, j, T(1), T(1), T(1))
@inline slab_index(i::T) where {T} = CartesianIndex(i, T(1), T(1), T(1), T(1))
@inline vindex(v::T) where {T} = CartesianIndex(T(1), T(1), T(1), v, T(1))

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
