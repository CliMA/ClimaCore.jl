module ClimaCoreTempestRemap

export write_exodus, rll_mesh, overlap_mesh, remap_weights, apply_remap
export def_time_coord, def_space_coord


using ClimaComms
import ClimaCore
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields

using NCDatasets, Dates, PkgVersion, LinearAlgebra, TempestRemap_jll

include("netcdf.jl")
include("exodus.jl")
include("wrappers.jl")
include("onlineremap.jl")

end # module
