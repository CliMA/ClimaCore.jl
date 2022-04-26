using Base: operator_associativity


# Order of tests is intended to reflect dependency order of functionality

@time include("recursive.jl")

@time include("DataLayouts/data0d.jl")
@time include("DataLayouts/data1d.jl")
@time include("DataLayouts/data2d.jl")
@time include("DataLayouts/data1dx.jl")
@time include("DataLayouts/data2dx.jl")

@time include("Geometry/geometry.jl")
@time include("Geometry/axistensors.jl")

@time include("Meshes/interval.jl")
@time include("Meshes/rectangle.jl")
@time include("Meshes/cubedsphere.jl")

@time include("Topologies/rectangle.jl")
@time include("Topologies/cubedsphere.jl")
@time include("Topologies/distributed.jl")

@time include("Spaces/quadrature.jl")
@time include("Spaces/spaces.jl")
@time include("Spaces/sphere.jl")
@time include("Spaces/terrain_warp.jl")
@time include("Spaces/distributed.jl")

@time include("Fields/field.jl")
@time include("Fields/fielddiffeq.jl")

@time include("Operators/spectralelement/rectilinear.jl")
@time include("Operators/spectralelement/opt.jl")
@time include("Operators/spectralelement/diffusion2d.jl")
@time include("Operators/spectralelement/sphere_geometry.jl")
@time include("Operators/spectralelement/sphere_gradient.jl")
@time include("Operators/spectralelement/sphere_divergence.jl")
@time include("Operators/spectralelement/sphere_curl.jl")
@time include("Operators/spectralelement/sphere_diffusion.jl")
@time include("Operators/spectralelement/sphere_diffusion_vec.jl")
@time include("Operators/spectralelement/sphere_hyperdiffusion.jl")
@time include("Operators/spectralelement/sphere_hyperdiffusion_vec.jl")

@time include("Operators/finitedifference/column.jl")
@time include("Operators/finitedifference/opt.jl")
@time include("Operators/finitedifference/opt_examples.jl")
@time include("Operators/finitedifference/implicit_stencils.jl")
@time include("Operators/finitedifference/opt_implicit_stencils.jl")

@time include("Operators/hybrid/2d.jl")
@time include("Operators/hybrid/3d.jl")
@time include("Operators/hybrid/opt.jl")

@time include("Operators/remapping.jl")

@time include("Limiters/limiter.jl")

if "CUDA" in ARGS
    @time include("gpu/cuda.jl")
    @time include("gpu/data.jl")
end
