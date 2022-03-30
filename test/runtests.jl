using Base: operator_associativity


# Order of tests is intended to reflect dependency order of functionality

include("recursive.jl")

include("DataLayouts/data1d.jl")
include("DataLayouts/data2d.jl")
include("DataLayouts/data1dx.jl")
include("DataLayouts/data2dx.jl")

include("Geometry/geometry.jl")
include("Geometry/axistensors.jl")

include("Meshes/interval.jl")
include("Meshes/rectangle.jl")
include("Meshes/cubedsphere.jl")

include("Topologies/rectangle.jl")
include("Topologies/cubedsphere.jl")
include("Topologies/distributed.jl")

include("Spaces/quadrature.jl")
include("Spaces/spaces.jl")
include("Spaces/sphere.jl")
include("Spaces/terrain_warp.jl")
include("Spaces/distributed.jl")

include("Fields/field.jl")
include("Fields/fielddiffeq.jl")

include("Operators/spectralelement/rectilinear.jl")
include("Operators/spectralelement/opt.jl")
include("Operators/spectralelement/diffusion2d.jl")
include("Operators/spectralelement/sphere_geometry.jl")
include("Operators/spectralelement/sphere_gradient.jl")
include("Operators/spectralelement/sphere_divergence.jl")
include("Operators/spectralelement/sphere_curl.jl")
include("Operators/spectralelement/sphere_diffusion.jl")
include("Operators/spectralelement/sphere_diffusion_vec.jl")
include("Operators/spectralelement/sphere_hyperdiffusion.jl")
include("Operators/spectralelement/sphere_hyperdiffusion_vec.jl")

include("Operators/finitedifference/column.jl")
include("Operators/finitedifference/opt.jl")
include("Operators/finitedifference/opt_examples.jl")
include("Operators/finitedifference/implicit_stencils.jl")
include("Operators/finitedifference/opt_implicit_stencils.jl")

include("Operators/hybrid/2d.jl")
include("Operators/hybrid/3d.jl")
include("Operators/hybrid/opt.jl")

include("Operators/remapping.jl")

include("Limiters/limiter.jl")

if "CUDA" in ARGS
    include("gpu/cuda.jl")
    include("gpu/data.jl")
end
