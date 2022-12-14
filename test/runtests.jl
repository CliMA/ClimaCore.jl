using Test
using Base: operator_associativity

#! format: off
# Order of tests is intended to reflect dependency order of functionality

#= TODO: add windows test back in. Currently getting
ReadOnlyMemoryError()
ERROR: Package ClimaCore errored during testing (exit code: 541541187)
Stacktrace:
 [1] pkgerror(msg::String)
=#
memusage() = @info "Memory usage" "allocated (MB)"=Base.gc_live_bytes() / 2^20
if !Sys.iswindows()
    @testset "Recursive" let @time include("recursive.jl") end
    memusage()
    @testset "PlusHalf" let @time include("Utilities/plushalf.jl") end
    memusage()
    @testset "DataLayouts 0D" let @time include("DataLayouts/data0d.jl") end
    memusage()
    @testset "DataLayouts 1D" let @time include("DataLayouts/data1d.jl") end
    memusage()
    @testset "DataLayouts 2D" let @time include("DataLayouts/data2d.jl") end
    memusage()
    @testset "DataLayouts 1dx" let @time include("DataLayouts/data1dx.jl") end
    memusage()
    @testset "DataLayouts 2dx" let @time include("DataLayouts/data2dx.jl") end
    memusage()

    @testset "Geometry" let @time include("Geometry/geometry.jl") end
    memusage()
    @testset "AxisTensors" let @time include("Geometry/axistensors.jl") end
    memusage()

    @testset "Interval mesh" let @time include("Meshes/interval.jl") end
    memusage()
    @testset "Rectangle mesh" let @time include("Meshes/rectangle.jl") end
    memusage()
    # now part of buildkite
    # @testset "Cubedsphere mesh" let @time include("Meshes/cubedsphere.jl") end

    @testset "Rectangle topology" let @time include("Topologies/rectangle.jl") end
    memusage()
    @testset "Rectangle surface topology" let @time include("Topologies/rectangle_sfc.jl") end
    memusage()
    @testset "Cubedsphere topology" let @time include("Topologies/cubedsphere.jl") end
    memusage()
    @testset "Cubedsphere surface topology" let @time include("Topologies/cubedsphere_sfc.jl") end
    memusage()
    # now part of buildkite
    # @testset "Distributed topology" let @time include("Topologies/distributed.jl") end

    @testset "Quadrature" let @time include("Spaces/quadrature.jl") end
    memusage()
    @testset "Spaces" let @time include("Spaces/spaces.jl") end
    memusage()
    # now part of buildkite
    # @testset "Sphere spaces" let @time include("Spaces/sphere.jl") end
    # @testset "Terrain warp" let @time include("Spaces/terrain_warp.jl") end
    # now part of buildkite
    # @testset "Distributed spaces" let @time include("Spaces/distributed.jl") end

    # now part of buildkite
    # @testset "Fields" let @time include("Fields/field.jl") end
    @testset "Fields diffeq" let @time include("Fields/fielddiffeq.jl") end
    memusage()

    @testset "Spectral elem - rectilinear" let @time include("Operators/spectralelement/rectilinear.jl") end
    memusage()
    @testset "Spectral elem - opt" let @time include("Operators/spectralelement/opt.jl") end
    memusage()
    @testset "Spectral elem - Diffusion 2d" let @time include("Operators/spectralelement/diffusion2d.jl") end
    memusage()
    @testset "Spectral elem - sphere geometry" let @time include("Operators/spectralelement/sphere_geometry.jl") end
    memusage()
    @testset "Spectral elem - sphere gradient" let @time include("Operators/spectralelement/sphere_gradient.jl") end
    memusage()
    @testset "Spectral elem - sphere divergence" let @time include("Operators/spectralelement/sphere_divergence.jl") end
    memusage()
    @testset "Spectral elem - sphere curl" let @time include("Operators/spectralelement/sphere_curl.jl") end
    memusage()
    @testset "Spectral elem - sphere diffusion" let @time include("Operators/spectralelement/sphere_diffusion.jl") end
    memusage()
    @testset "Spectral elem - sphere diffusion vec" let @time include("Operators/spectralelement/sphere_diffusion_vec.jl") end
    memusage()
    @testset "Spectral elem - sphere hyperdiffusion" let @time include("Operators/spectralelement/sphere_hyperdiffusion.jl") end
    memusage()
    @testset "Spectral elem - sphere hyperdiffusion vec" let @time include("Operators/spectralelement/sphere_hyperdiffusion_vec.jl") end
    memusage()

    @testset "FD ops - column" let @time include("Operators/finitedifference/column.jl") end
    memusage()
    @testset "FD ops - opt" let @time include("Operators/finitedifference/opt.jl") end
    memusage()
    @testset "FD ops - wfact" let @time include("Operators/finitedifference/wfact.jl") end
    memusage()
    @testset "FD ops - linsolve" let @time include("Operators/finitedifference/linsolve.jl") end
    memusage()
    @testset "FD ops - examples" let @time include("Operators/finitedifference/opt_examples.jl") end
    memusage()
    # now part of buildkite
    # @time include("Operators/finitedifference/implicit_stencils.jl")
    # @time include("Operators/finitedifference/opt_implicit_stencils.jl")

    @testset "Hybrid - 2D" let @time include("Operators/hybrid/2d.jl") end
    memusage()
    @testset "Hybrid - 3D" let @time include("Operators/hybrid/3d.jl") end
    memusage()
    @testset "Hybrid - dss opt" let @time include("Operators/hybrid/dss_opt.jl") end
    memusage()
    @testset "Hybrid - opt" let @time include("Operators/hybrid/opt.jl") end
    memusage()

    @testset "Hypsography - 2d" let @time include("Hypsography/2d.jl") end
    memusage()
    @testset "Hypsography - 3d sphere" let @time include("Hypsography/3dsphere.jl") end
    memusage()

    @testset "Remapping" let @time include("Operators/remapping.jl") end
    memusage()

    # now part of buildkite
    # @testset "Limiter" let @time include("Limiters/limiter.jl") end
    # @testset "Distributed limiters" let @time include("Limiters/distributed.jl") end

    @testset "InputOutput - spectralelement2d" let @time include("InputOutput/spectralelement2d.jl") end
    memusage()
    @testset "InputOutput - hybrid2dbox" let @time include("InputOutput/hybrid2dbox.jl") end
    memusage()
    @testset "InputOutput - hybrid2dbox_topography" let @time include("InputOutput/hybrid2dbox_topography.jl") end
    memusage()
    @testset "InputOutput - hybrid2dbox_stretched" let @time include("InputOutput/hybrid2dbox_stretched.jl") end
    memusage()
    @testset "InputOutput - hybrid3dbox" let @time include("InputOutput/hybrid3dbox.jl") end
    memusage()
    @testset "InputOutput - hybrid3dcubedsphere" let @time include("InputOutput/hybrid3dcubedsphere.jl") end
    memusage()
    @testset "InputOutput - hybrid3dcubedsphere_topography" let @time include("InputOutput/hybrid3dcubedsphere_topography.jl") end
    memusage()

    # Code quality checks
    @testset "Aqua" let @time include("aqua.jl") end
    memusage()
end
if "CUDA" in ARGS
    @testset "GPU - cuda" let @time include("gpu/cuda.jl") end
    @testset "GPU - data" let @time include("gpu/data.jl") end
end

#! format: on
