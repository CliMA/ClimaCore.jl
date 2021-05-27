module ClimateMachineCore

include("interface.jl")
include("RecursiveApply/RecursiveApply.jl")
include("DataLayouts/DataLayouts.jl")
include("Geometry/Geometry.jl")
include("Domains/Domains.jl")
include("Meshes/Meshes.jl")
include("Topologies/Topologies.jl")
include("Spaces/Spaces.jl")
include("Fields/Fields.jl")
include("Operators/Operators.jl")
include("Plots/plots.jl")

end # module
