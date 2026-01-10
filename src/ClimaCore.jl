module ClimaCore

using PkgVersion
const VERSION = PkgVersion.@Version
import ClimaComms

include("DebugOnly/DebugOnly.jl")
include("interface.jl")
include("devices.jl")
include("Utilities/Utilities.jl")
include("DataLayouts/DataLayouts.jl")
include("Geometry/Geometry.jl")
include("Domains/Domains.jl")
include("Meshes/Meshes.jl")
include("Topologies/Topologies.jl")
include("Quadratures/Quadratures.jl")
include("Grids/Grids.jl")
include("Spaces/Spaces.jl")
include("Fields/Fields.jl")
include("Operators/Operators.jl")
include("MatrixFields/MatrixFields.jl")
include("Hypsography/Hypsography.jl")
include("Limiters/Limiters.jl")
include("InputOutput/InputOutput.jl")
include("CommonGrids/CommonGrids.jl")
include("CommonSpaces/CommonSpaces.jl")
include("Remapping/Remapping.jl")

include("deprecated.jl")
include("to_device.jl")

end # module
