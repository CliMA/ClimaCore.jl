using SafeTestsets
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
    @safetestset "Recursive" begin @time include("recursive.jl") end
    memusage()
    @safetestset "PlusHalf" begin @time include("Utilities/plushalf.jl") end
    memusage()
    @safetestset "DataLayouts 0D" begin @time include("DataLayouts/data0d.jl") end
    memusage()
    @safetestset "DataLayouts 1D" begin @time include("DataLayouts/data1d.jl") end
    memusage()
    @safetestset "DataLayouts 2D" begin @time include("DataLayouts/data2d.jl") end
    memusage()
    @safetestset "DataLayouts 1dx" begin @time include("DataLayouts/data1dx.jl") end
    memusage()
    @safetestset "DataLayouts 2dx" begin @time include("DataLayouts/data2dx.jl") end
    memusage()

    @safetestset "Geometry" begin @time include("Geometry/geometry.jl") end
    memusage()
    @safetestset "AxisTensors" begin @time include("Geometry/axistensors.jl") end
    memusage()

    @safetestset "Interval mesh" begin @time include("Meshes/interval.jl") end
    memusage()
    @safetestset "Rectangle mesh" begin @time include("Meshes/rectangle.jl") end
    memusage()
    # now part of buildkite
    # @safetestset "Cubedsphere mesh" begin @time include("Meshes/cubedsphere.jl") end

    @safetestset "Rectangle topology" begin @time include("Topologies/rectangle.jl") end
    memusage()
    @safetestset "Rectangle surface topology" begin @time include("Topologies/rectangle_sfc.jl") end
    memusage()
    @safetestset "Cubedsphere topology" begin @time include("Topologies/cubedsphere.jl") end
    memusage()
    @safetestset "Cubedsphere surface topology" begin @time include("Topologies/cubedsphere_sfc.jl") end
    memusage()
    # now part of buildkite
    # @safetestset "Distributed topology" begin @time include("Topologies/distributed.jl") end

    @safetestset "Quadrature" begin @time include("Spaces/quadrature.jl") end
    memusage()
    @safetestset "Spaces" begin @time include("Spaces/spaces.jl") end
    memusage()
    # now part of buildkite
    # @safetestset "Sphere spaces" begin @time include("Spaces/sphere.jl") end
    # @safetestset "Terrain warp" begin @time include("Spaces/terrain_warp.jl") end
    # now part of buildkite
    # @safetestset "Distributed spaces" begin @time include("Spaces/distributed.jl") end

    # now part of buildkite
    # @safetestset "Fields" begin @time include("Fields/field.jl") end
    @safetestset "Fields diffeq" begin @time include("Fields/fielddiffeq.jl") end
    memusage()

    @safetestset "Spectral elem - rectilinear" begin @time include("Operators/spectralelement/rectilinear.jl") end
    memusage()
    @safetestset "Spectral elem - opt" begin @time include("Operators/spectralelement/opt.jl") end
    memusage()
    @safetestset "Spectral elem - Diffusion 2d" begin @time include("Operators/spectralelement/diffusion2d.jl") end
    memusage()
    @safetestset "Spectral elem - sphere geometry" begin @time include("Operators/spectralelement/sphere_geometry.jl") end
    memusage()
    @safetestset "Spectral elem - sphere gradient" begin @time include("Operators/spectralelement/sphere_gradient.jl") end
    memusage()
    @safetestset "Spectral elem - sphere divergence" begin @time include("Operators/spectralelement/sphere_divergence.jl") end
    memusage()
    @safetestset "Spectral elem - sphere curl" begin @time include("Operators/spectralelement/sphere_curl.jl") end
    memusage()
    @safetestset "Spectral elem - sphere diffusion" begin @time include("Operators/spectralelement/sphere_diffusion.jl") end
    memusage()
    @safetestset "Spectral elem - sphere diffusion vec" begin @time include("Operators/spectralelement/sphere_diffusion_vec.jl") end
    memusage()
    @safetestset "Spectral elem - sphere hyperdiffusion" begin @time include("Operators/spectralelement/sphere_hyperdiffusion.jl") end
    memusage()
    @safetestset "Spectral elem - sphere hyperdiffusion vec" begin @time include("Operators/spectralelement/sphere_hyperdiffusion_vec.jl") end
    memusage()

    @safetestset "FD ops - column" begin @time include("Operators/finitedifference/column.jl") end
    memusage()
    @safetestset "FD ops - opt" begin @time include("Operators/finitedifference/opt.jl") end
    memusage()
    @safetestset "FD ops - wfact" begin @time include("Operators/finitedifference/wfact.jl") end
    memusage()
    @safetestset "FD ops - linsolve" begin @time include("Operators/finitedifference/linsolve.jl") end
    memusage()
    @safetestset "FD ops - examples" begin @time include("Operators/finitedifference/opt_examples.jl") end
    memusage()
    # now part of buildkite
    # @time include("Operators/finitedifference/implicit_stencils.jl")
    # @time include("Operators/finitedifference/opt_implicit_stencils.jl")

    @safetestset "Hybrid - 2D" begin @time include("Operators/hybrid/2d.jl") end
    memusage()
    @safetestset "Hybrid - 3D" begin @time include("Operators/hybrid/3d.jl") end
    memusage()
    @safetestset "Hybrid - dss opt" begin @time include("Operators/hybrid/dss_opt.jl") end
    memusage()
    @safetestset "Hybrid - opt" begin @time include("Operators/hybrid/opt.jl") end
    memusage()

    @safetestset "Hypsography - 2d" begin @time include("Hypsography/2d.jl") end
    memusage()
    @safetestset "Hypsography - 3d sphere" begin @time include("Hypsography/3dsphere.jl") end
    memusage()

    @safetestset "Remapping" begin @time include("Operators/remapping.jl") end
    memusage()

    # now part of buildkite
    # @safetestset "Limiter" begin @time include("Limiters/limiter.jl") end
    # @safetestset "Distributed limiters" begin @time include("Limiters/distributed.jl") end

    @safetestset "InputOutput - spectralelement2d" begin @time include("InputOutput/spectralelement2d.jl") end
    memusage()
    @safetestset "InputOutput - hybrid2dbox" begin @time include("InputOutput/hybrid2dbox.jl") end
    memusage()
    @safetestset "InputOutput - hybrid2dbox_topography" begin @time include("InputOutput/hybrid2dbox_topography.jl") end
    memusage()
    @safetestset "InputOutput - hybrid2dbox_stretched" begin @time include("InputOutput/hybrid2dbox_stretched.jl") end
    memusage()
    @safetestset "InputOutput - hybrid3dbox" begin @time include("InputOutput/hybrid3dbox.jl") end
    memusage()
    @safetestset "InputOutput - hybrid3dcubedsphere" begin @time include("InputOutput/hybrid3dcubedsphere.jl") end
    memusage()
    @safetestset "InputOutput - hybrid3dcubedsphere_topography" begin @time include("InputOutput/hybrid3dcubedsphere_topography.jl") end
    memusage()

    # Code quality checks
    @safetestset "Aqua" begin @time include("aqua.jl") end
    memusage()
end
if "CUDA" in ARGS
    @safetestset "GPU - cuda" begin @time include("gpu/cuda.jl") end
    @safetestset "GPU - data" begin @time include("gpu/data.jl") end
end

#! format: on
