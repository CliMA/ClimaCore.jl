using Base: operator_associativity


# Order of tests is intended to reflect dependency order of functionality

include("recursive.jl")

include("DataLayouts/data1d.jl")
include("DataLayouts/data2d.jl")
include("DataLayouts/data1dx.jl")
include("DataLayouts/data2dx.jl")

include("Geometry/geometry.jl")
include("Geometry/axistensors.jl")

include("Meshes/rectangle.jl")
include("Meshes/cubedsphere.jl")

include("Topologies/rectangle.jl")
include("Topologies/cubedsphere.jl")

include("quadrature.jl")
include("spaces.jl")
include("field.jl")
include("spectraloperators.jl")
include("spectralspaces_opt.jl")
include("diffusion2d.jl")
include("fdspaces.jl")
include("fdspaces_opt.jl")
include("fielddiffeq.jl")
include("hybrid2d.jl")
include("hybrid3d.jl")
include("remapping.jl")
#spheres.jl")
include("sphere_geometry.jl")
include("sphere_metric_terms.jl")
include("sphere_gradient.jl")
include("sphere_divergence.jl")
include("sphere_curl.jl")
include("sphere_diffusion.jl")
include("sphere_hyperdiffusion.jl")

if "CUDA" in ARGS
    include("gpu/cuda.jl")
    include("gpu/data.jl")
end
