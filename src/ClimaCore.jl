module ClimaCore

using PkgVersion
const VERSION = PkgVersion.@Version

include("interface.jl")
include("Utilities/Utilities.jl")
include("RecursiveApply/RecursiveApply.jl")
include("DataLayouts/DataLayouts.jl")
include("Geometry/Geometry.jl")
include("Domains/Domains.jl")
include("Meshes/Meshes.jl")
include("Topologies/Topologies.jl")
include("Spaces/Spaces.jl")
include("Fields/Fields.jl")
include("Operators/Operators.jl")
include("Hypsography/Hypsography.jl")
include("Limiters/Limiters.jl")
include("InputOutput/InputOutput.jl")

include("deprecated.jl")

end # module
