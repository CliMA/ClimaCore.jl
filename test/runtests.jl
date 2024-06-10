using Base: operator_associativity

#! format: off
# Order of tests is intended to reflect dependency order of functionality

@time "Unit tests" begin
    @time "Recursive" include("RecursiveApply/unit_recursive_apply.jl")
    @time "PlusHalf" include("Utilities/unit_plushalf.jl")
    @time "Geometry" include("Geometry/geometry.jl")
    @time "AxisTensors" include("Geometry/axistensors.jl")
    @time "DataLayouts fill" include("DataLayouts/unit_fill.jl")
    @time "DataLayouts ndims" include("DataLayouts/unit_ndims.jl")
    @time "DataLayouts 0D" include("DataLayouts/data0d.jl")
    @time "DataLayouts 1D" include("DataLayouts/data1d.jl")
    @time "DataLayouts 2D" include("DataLayouts/data2d.jl")
    @time "DataLayouts 1dx" include("DataLayouts/data1dx.jl")
    @time "DataLayouts 2dx" include("DataLayouts/data2dx.jl")
    @time "Interval mesh" include("Meshes/interval.jl")
    @time "Rectangle mesh" include("Meshes/rectangle.jl")
    @time "Cubedsphere mesh" include("Meshes/cubedsphere.jl")
    @time "Interval topology" include("Topologies/interval.jl")
    @time "Rectangle topology" include("Topologies/rectangle.jl")
    @time "Rectangle surface topology" include("Topologies/rectangle_sfc.jl")
    @time "Cubedsphere topology" include("Topologies/cubedsphere.jl")
    @time "Cubedsphere surface topology" include("Topologies/cubedsphere_sfc.jl")
    @time "Quadratures" include("Quadratures/Quadratures.jl")
    @time "Spaces" include("Spaces/spaces.jl")
    @time "Spaces DSS" include("Spaces/ddss1.jl")
    @time "Sphere spaces" include("Spaces/sphere.jl")
    @time "Terrain warp" include("Spaces/terrain_warp.jl")
    @time "Distributed spaces" include("Spaces/distributed.jl")
    @time "Fields" include("Fields/field.jl")
end
    @safetestset "Spectral elem - rectilinear" begin @time include("Operators/spectralelement/rectilinear.jl") end
    # @safetestset "Spectral elem - opt" begin @time include("Operators/spectralelement/opt.jl") end
    @safetestset "Spectral elem - Diffusion 2d" begin @time include("Operators/spectralelement/unit_diffusion2d.jl") end
    @safetestset "Spctral elem - sphere geometry" begin @time include("Operators/spectralelement/sphere_geometry.jl") end
    @safetestset "Spectral elem - sphere gradient" begin @time include("Operators/spectralelement/sphere_gradient.jl") end
    @safetestset "Spectral elem - sphere divergence" begin @time include("Operators/spectralelement/sphere_divergence.jl") end
    @safetestset "Spectral elem - sphere curl" begin @time include("Operators/spectralelement/sphere_curl.jl") end
    @safetestset "Spectral elem - sphere diffusion" begin @time include("Operators/spectralelement/sphere_diffusion.jl") end
    @safetestset "Spectral elem - sphere diffusion vec" begin @time include("Operators/spectralelement/sphere_diffusion_vec.jl") end
    @safetestset "Spectral elem - sphere hyperdiffusion" begin @time include("Operators/spectralelement/unit_sphere_hyperdiffusion.jl") end
    @safetestset "Spectral elem - sphere hyperdiffusion vec" begin @time include("Operators/spectralelement/unit_sphere_hyperdiffusion_vec.jl") end
    
    @safetestset "FD ops - column" begin @time include("Operators/finitedifference/column.jl") end
    @safetestset "FD ops - opt" begin @time include("Operators/finitedifference/opt.jl") end
    @safetestset "FD ops - wfact" begin @time include("Operators/finitedifference/wfact.jl") end
    @safetestset "FD ops - linsolve" begin @time include("Operators/finitedifference/linsolve.jl") end
    @safetestset "FD ops - examples" begin @time include("Operators/finitedifference/opt_examples.jl") end
    # now part of buildkite
    # @time include("Operators/finitedifference/implicit_stencils.jl")
    # @time include("Operators/finitedifference/opt_implicit_stencils.jl")
    @safetestset "Hybrid - 2D" begin @time include("Operators/hybrid/unit_2d.jl") end
    @safetestset "Hybrid - 3D" begin @time include("Operators/hybrid/unit_3d.jl") end
    @safetestset "Hybrid - dss opt" begin @time include("Operators/hybrid/dss_opt.jl") end
    @safetestset "Hybrid - opt" begin @time include("Operators/hybrid/opt.jl") end

#     @safetestset "MatrixFields - BandMatrixRow" begin @time include("MatrixFields/band_matrix_row.jl") end
#     @safetestset "MatrixFields - rmul_with_projection" begin @time include("MatrixFields/rmul_with_projection.jl") end
#     @safetestset "MatrixFields - field2arrays" begin @time include("MatrixFields/field2arrays.jl") end
#     @safetestset "MatrixFields - matrix multiplication at boundaries" begin @time include("MatrixFields/matrix_multiplication_at_boundaries.jl") end
#     @safetestset "MatrixFields - field names" begin @time include("MatrixFields/field_names.jl") end
#     # now part of buildkite
#     # @safetestset "MatrixFields - matrix field broadcasting" begin @time include("MatrixFields/matrix_field_broadcasting.jl") end
#     # @safetestset "MatrixFields - operator matrices" begin @time include("MatrixFields/operator_matrices.jl") end
#     # @safetestset "MatrixFields - field matrix solvers" begin @time include("MatrixFields/field_matrix_solvers.jl") end

#     @safetestset "Hypsography - 2d" begin @time include("Hypsography/2d.jl") end
#     @safetestset "Hypsography - 3d sphere" begin @time include("Hypsography/3dsphere.jl") end

#     @safetestset "Remapping" begin @time include("Operators/remapping.jl") end

#     # now part of buildkite
#     # @safetestset "Limiter" begin @time include("Limiters/limiter.jl") end
#     # @safetestset "Distributed limiters" begin @time include("Limiters/distributed.jl") end

#     @safetestset "InputOutput - hdf5" begin @time include("InputOutput/hdf5.jl") end
#     @safetestset "InputOutput - spectralelement2d" begin @time include("InputOutput/spectralelement2d.jl") end
#     @safetestset "InputOutput - hybrid2dbox" begin @time include("InputOutput/hybrid2dbox.jl") end
#     @safetestset "InputOutput - hybrid2dbox_topography" begin @time include("InputOutput/hybrid2dbox_topography.jl") end
#     @safetestset "InputOutput - hybrid2dbox_stretched" begin @time include("InputOutput/hybrid2dbox_stretched.jl") end
#     @safetestset "InputOutput - hybrid3dbox" begin @time include("InputOutput/hybrid3dbox.jl") end
#     @safetestset "InputOutput - hybrid3dcubedsphere" begin @time include("InputOutput/hybrid3dcubedsphere.jl") end
#     @safetestset "InputOutput - hybrid3dcubedsphere_topography" begin @time include("InputOutput/hybrid3dcubedsphere_topography.jl") end

#     @safetestset "Array interpolation" begin @time include("Remapping/interpolate_array.jl") end
#     @safetestset "Array interpolation" begin @time include("Remapping/distributed_remapping.jl") end

    # Code quality checks
    @safetestset "Aqua" begin @time include("aqua.jl") end
end
import ClimaComms
if ClimaComms.device() isa ClimaComms.CUDADevice
    @safetestset "GPU - cuda" begin @time include("gpu/cuda.jl") end
    @safetestset "GPU - data" begin @time include("DataLayouts/cuda.jl") end
    @safetestset "GPU - spaces" begin @time include("Spaces/spaces.jl") end
    @safetestset "Spaces - serial CUDA DSS" begin @time include("Spaces/ddss1.jl") end
    @safetestset "Spaces - serial CUDA DSS on CubedSphere" begin @time include("Spaces/ddss1_cs.jl") end
    @safetestset "Operators - spectral element CUDA" begin @time include("Operators/spectralelement/rectilinear_cuda.jl") end
    @safetestset "Operators - finite difference CUDA" begin @time include("Operators/hybrid/unit_cuda.jl") end
    @safetestset "Operators - extruded sphere space operators CUDA" begin @time include("Operators/hybrid/extruded_sphere_cuda.jl") end
    @safetestset "Operators - extruded 3dbox space operators CUDA" begin @time include("Operators/hybrid/extruded_3dbox_cuda.jl") end
    @safetestset "Fields - CUDA mapreduce" begin @time include("Fields/reduction_cuda.jl") end
end

# #! format: on
