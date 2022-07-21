using SafeTestsets
using Base: operator_associativity

#! format: off
# Order of tests is intended to reflect dependency order of functionality

@safetestset "Recursive" begin @time include("recursive.jl") end
@safetestset "PlusHalf" begin @time include("Utilities/plushalf.jl") end

@safetestset "DataLayouts 0D" begin @time include("DataLayouts/data0d.jl") end
@safetestset "DataLayouts 1D" begin @time include("DataLayouts/data1d.jl") end
@safetestset "DataLayouts 2D" begin @time include("DataLayouts/data2d.jl") end
@safetestset "DataLayouts 1dx" begin @time include("DataLayouts/data1dx.jl") end
@safetestset "DataLayouts 2dx" begin @time include("DataLayouts/data2dx.jl") end

@safetestset "Geometry" begin @time include("Geometry/geometry.jl") end
@safetestset "AxisTensors" begin @time include("Geometry/axistensors.jl") end

@safetestset "Interval mesh" begin @time include("Meshes/interval.jl") end
@safetestset "Rectangle mesh" begin @time include("Meshes/rectangle.jl") end
@safetestset "Cubedsphere mesh" begin @time include("Meshes/cubedsphere.jl") end

@safetestset "Rectangle topology" begin @time include("Topologies/rectangle.jl") end
@safetestset "Rectangle surface topology" begin @time include("Topologies/rectangle_sfc.jl") end
@safetestset "Cubedsphere topology" begin @time include("Topologies/cubedsphere.jl") end
@safetestset "Cubedsphere surface topology" begin @time include("Topologies/cubedsphere_sfc.jl") end
@safetestset "Distributed topology" begin @time include("Topologies/distributed.jl") end

@safetestset "Quadrature" begin @time include("Spaces/quadrature.jl") end
@safetestset "Spaces" begin @time include("Spaces/spaces.jl") end
@safetestset "Sphere spaces" begin @time include("Spaces/sphere.jl") end
@safetestset "Terrain warp" begin @time include("Spaces/terrain_warp.jl") end
@safetestset "Distributed spaces" begin @time include("Spaces/distributed.jl") end

@safetestset "Fields" begin @time include("Fields/field.jl") end
@safetestset "Fields diffeq" begin @time include("Fields/fielddiffeq.jl") end

@safetestset "Spectral elem - rectilinear" begin @time include("Operators/spectralelement/rectilinear.jl") end
@safetestset "Spectral elem - opt" begin @time include("Operators/spectralelement/opt.jl") end
@safetestset "Spectral elem - Diffusion 2d" begin @time include("Operators/spectralelement/diffusion2d.jl") end
@safetestset "Spectral elem - sphere geometry" begin @time include("Operators/spectralelement/sphere_geometry.jl") end
@safetestset "Spectral elem - sphere gradient" begin @time include("Operators/spectralelement/sphere_gradient.jl") end
@safetestset "Spectral elem - sphere divergence" begin @time include("Operators/spectralelement/sphere_divergence.jl") end
@safetestset "Spectral elem - sphere curl" begin @time include("Operators/spectralelement/sphere_curl.jl") end
@safetestset "Spectral elem - sphere diffusion" begin @time include("Operators/spectralelement/sphere_diffusion.jl") end
@safetestset "Spectral elem - sphere diffusion vec" begin @time include("Operators/spectralelement/sphere_diffusion_vec.jl") end
@safetestset "Spectral elem - sphere hyperdiffusion" begin @time include("Operators/spectralelement/sphere_hyperdiffusion.jl") end
@safetestset "Spectral elem - sphere hyperdiffusion vec" begin @time include("Operators/spectralelement/sphere_hyperdiffusion_vec.jl") end

@safetestset "FD ops - column" begin @time include("Operators/finitedifference/column.jl") end
@safetestset "FD ops - opt" begin @time include("Operators/finitedifference/opt.jl") end
@safetestset "FD ops - wfact" begin @time include("Operators/finitedifference/wfact.jl") end
@safetestset "FD ops - linsolve" begin @time include("Operators/finitedifference/linsolve.jl") end
@safetestset "FD ops - examples" begin @time include("Operators/finitedifference/opt_examples.jl") end
# now part of buildkite
# @time include("Operators/finitedifference/implicit_stencils.jl")
# @time include("Operators/finitedifference/opt_implicit_stencils.jl")

@safetestset "Hybrid - 2D" begin @time include("Operators/hybrid/2d.jl") end
@safetestset "Hybrid - 3D" begin @time include("Operators/hybrid/3d.jl") end
@safetestset "Hybrid - dss opt" begin @time include("Operators/hybrid/dss_opt.jl") end
@safetestset "Hybrid - opt" begin @time include("Operators/hybrid/opt.jl") end

@safetestset "Remapping" begin @time include("Operators/remapping.jl") end

@safetestset "Limiter" begin @time include("Limiters/limiter.jl") end
@safetestset "Distributed limiters" begin @time include("Limiters/distributed.jl") end

# Code quality checks
@safetestset "Aqua" begin @time include("aqua.jl") end

@safetestset "InputOutput" begin include("InputOutput/runtests_inputoutput.jl") end

if "CUDA" in ARGS
    @safetestset "GPU - cuda" begin @time include("gpu/cuda.jl") end
    @safetestset "GPU - data" begin @time include("gpu/data.jl") end
end

#! format: on
