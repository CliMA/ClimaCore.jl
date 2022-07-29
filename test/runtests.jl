using SafeTestsets

include("utils.jl")

(s, parsed_args) = parse_commandline()
#= Test ID, for load balancing the test suite=#
tid(x) = parsed_args["bin_id"] == x || isinteractive()
import OrderedCollections
times = OrderedCollections.OrderedDict() # CI times

#! format: off
# Order of tests is intended to reflect dependency order of functionality

#= TODO: add windows test back in. Currently getting
ReadOnlyMemoryError()
ERROR: Package ClimaCore errored during testing (exit code: 541541187)
Stacktrace:
 [1] pkgerror(msg::String)
=#
if !Sys.iswindows()

tid(1) && @safetimedtestset times "Recursive" begin @time include("recursive.jl") end
tid(1) && @safetimedtestset times "PlusHalf" begin @time include("Utilities/plushalf.jl") end

tid(1) && @safetimedtestset times "DataLayouts 0D" begin @time include("DataLayouts/data0d.jl") end
tid(1) && @safetimedtestset times "DataLayouts 1D" begin @time include("DataLayouts/data1d.jl") end
tid(1) && @safetimedtestset times "DataLayouts 2D" begin @time include("DataLayouts/data2d.jl") end
tid(1) && @safetimedtestset times "DataLayouts 1dx" begin @time include("DataLayouts/data1dx.jl") end
tid(1) && @safetimedtestset times "DataLayouts 2dx" begin @time include("DataLayouts/data2dx.jl") end

tid(1) && @safetimedtestset times "Geometry" begin @time include("Geometry/geometry.jl") end
tid(1) && @safetimedtestset times "AxisTensors" begin @time include("Geometry/axistensors.jl") end

tid(1) && @safetimedtestset times "Interval mesh" begin @time include("Meshes/interval.jl") end
tid(1) && @safetimedtestset times "Rectangle mesh" begin @time include("Meshes/rectangle.jl") end
tid(1) && @safetimedtestset times "Cubedsphere mesh" begin @time include("Meshes/cubedsphere.jl") end

tid(1) && @safetimedtestset times "Rectangle topology" begin @time include("Topologies/rectangle.jl") end
tid(1) && @safetimedtestset times "Rectangle surface topology" begin @time include("Topologies/rectangle_sfc.jl") end
tid(1) && @safetimedtestset times "Cubedsphere topology" begin @time include("Topologies/cubedsphere.jl") end
tid(1) && @safetimedtestset times "Cubedsphere surface topology" begin @time include("Topologies/cubedsphere_sfc.jl") end
tid(1) && @safetimedtestset times "Distributed topology" begin @time include("Topologies/distributed.jl") end

tid(1) && @safetimedtestset times "Quadrature" begin @time include("Spaces/quadrature.jl") end
tid(1) && @safetimedtestset times "Spaces" begin @time include("Spaces/spaces.jl") end
tid(1) && @safetimedtestset times "Sphere spaces" begin @time include("Spaces/sphere.jl") end
tid(1) && @safetimedtestset times "Terrain warp" begin @time include("Spaces/terrain_warp.jl") end
tid(3) && @safetimedtestset times "Distributed spaces" begin @time include("Spaces/distributed.jl") end

tid(1) && @safetimedtestset times "Fields" begin @time include("Fields/field.jl") end
tid(2) && @safetimedtestset times "Fields diffeq" begin @time include("Fields/fielddiffeq.jl") end

tid(1) && @safetimedtestset times "Spectral elem - rectilinear" begin @time include("Operators/spectralelement/rectilinear.jl") end
tid(1) && @safetimedtestset times "Spectral elem - opt" begin @time include("Operators/spectralelement/opt.jl") end
tid(1) && @safetimedtestset times "Spectral elem - Diffusion 2d" begin @time include("Operators/spectralelement/diffusion2d.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere geometry" begin @time include("Operators/spectralelement/sphere_geometry.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere gradient" begin @time include("Operators/spectralelement/sphere_gradient.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere divergence" begin @time include("Operators/spectralelement/sphere_divergence.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere curl" begin @time include("Operators/spectralelement/sphere_curl.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere diffusion" begin @time include("Operators/spectralelement/sphere_diffusion.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere diffusion vec" begin @time include("Operators/spectralelement/sphere_diffusion_vec.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere hyperdiffusion" begin @time include("Operators/spectralelement/sphere_hyperdiffusion.jl") end
tid(1) && @safetimedtestset times "Spectral elem - sphere hyperdiffusion vec" begin @time include("Operators/spectralelement/sphere_hyperdiffusion_vec.jl") end

tid(1) && @safetimedtestset times "FD ops - column" begin @time include("Operators/finitedifference/column.jl") end
tid(1) && @safetimedtestset times "FD ops - opt" begin @time include("Operators/finitedifference/opt.jl") end
tid(1) && @safetimedtestset times "FD ops - wfact" begin @time include("Operators/finitedifference/wfact.jl") end
tid(1) && @safetimedtestset times "FD ops - linsolve" begin @time include("Operators/finitedifference/linsolve.jl") end
tid(1) && @safetimedtestset times "FD ops - examples" begin @time include("Operators/finitedifference/opt_examples.jl") end
# now part of buildkite
# @time include("Operators/finitedifference/implicit_stencils.jl")
# @time include("Operators/finitedifference/opt_implicit_stencils.jl")

tid(1) && @safetimedtestset times "Hybrid - 2D" begin @time include("Operators/hybrid/2d.jl") end
tid(1) && @safetimedtestset times "Hybrid - 3D" begin @time include("Operators/hybrid/3d.jl") end
tid(1) && @safetimedtestset times "Hybrid - dss opt" begin @time include("Operators/hybrid/dss_opt.jl") end
tid(1) && @safetimedtestset times "Hybrid - opt" begin @time include("Operators/hybrid/opt.jl") end

tid(1) && @safetimedtestset times "Remapping" begin @time include("Operators/remapping.jl") end

tid(1) && @safetimedtestset times "Limiter" begin @time include("Limiters/limiter.jl") end
tid(1) && @safetimedtestset times "Distributed limiters" begin @time include("Limiters/distributed.jl") end

# Code quality checks
tid(1) && @safetimedtestset times "Aqua" begin @time include("aqua.jl") end

tid(1) && @safetimedtestset times "InputOutput" begin include("InputOutput/runtests_inputoutput.jl") end
end

if "CUDA" in ARGS
    tid(1) && @safetimedtestset times "GPU - cuda" begin @time include("gpu/cuda.jl") end
    tid(1) && @safetimedtestset times "GPU - data" begin @time include("gpu/data.jl") end
end

#! format: on
