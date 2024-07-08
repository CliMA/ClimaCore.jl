#=
julia --project
using Revise; include(joinpath("test", "runtests.jl"))
=#
using Test
include("tabulated_tests.jl")

#! format: off
unit_tests = [
UnitTest("DataLayouts fill"                        ,"DataLayouts/unit_fill.jl"),
UnitTest("DataLayouts ndims"                       ,"DataLayouts/unit_ndims.jl"),
UnitTest("Recursive"                               ,"RecursiveApply/unit_recursive_apply.jl"),
UnitTest("PlusHalf"                                ,"Utilities/unit_plushalf.jl"),
UnitTest("DataLayouts 0D"                          ,"DataLayouts/data0d.jl"),
UnitTest("DataLayouts 1D"                          ,"DataLayouts/data1d.jl"),
UnitTest("DataLayouts 2D"                          ,"DataLayouts/data2d.jl"),
UnitTest("DataLayouts 1dx"                         ,"DataLayouts/data1dx.jl"),
UnitTest("DataLayouts 2dx"                         ,"DataLayouts/data2dx.jl"),
UnitTest("DataLayouts mapreduce"                   ,"DataLayouts/unit_mapreduce.jl"),
UnitTest("Geometry"                                ,"Geometry/geometry.jl"),
UnitTest("rmul_with_projection"                    ,"Geometry/rmul_with_projection.jl"),
UnitTest("AxisTensors"                             ,"Geometry/axistensors.jl"),
UnitTest("Interval mesh"                           ,"Meshes/interval.jl"),
UnitTest("Rectangle mesh"                          ,"Meshes/rectangle.jl"),
UnitTest("Cubedsphere mesh"                        ,"Meshes/cubedsphere.jl"),
UnitTest("Interval topology"                       ,"Topologies/interval.jl"),
UnitTest("Rectangle topology"                      ,"Topologies/rectangle.jl"),
UnitTest("Rectangle surface topology"              ,"Topologies/rectangle_sfc.jl"),
UnitTest("Cubedsphere topology"                    ,"Topologies/cubedsphere.jl"),
UnitTest("Cubedsphere surface topology"            ,"Topologies/cubedsphere_sfc.jl"),
UnitTest("Quadratures"                             ,"Quadratures/Quadratures.jl"),
UnitTest("Spaces"                                  ,"Spaces/unit_spaces.jl"),
UnitTest("Spaces - serial CPU DSS"                 ,"Spaces/ddss1.jl"),
UnitTest("Sphere spaces"                           ,"Spaces/sphere.jl"),
# UnitTest("Terrain warp"                            ,"Spaces/terrain_warp.jl"), # appears to hang on GHA
UnitTest("Fields"                                  ,"Fields/unit_field.jl"), # has benchmarks
UnitTest("Spectral elem - rectilinear"             ,"Operators/spectralelement/rectilinear.jl"),
UnitTest("Spectral elem - opt"                     ,"Operators/spectralelement/opt.jl"),
UnitTest("Spectral elem - Diffusion 2d"            ,"Operators/spectralelement/unit_diffusion2d.jl"),
UnitTest("Spectral elem - sphere geometry"         ,"Operators/spectralelement/sphere_geometry.jl"),
UnitTest("Spectral elem - sphere gradient"         ,"Operators/spectralelement/sphere_gradient.jl"),
UnitTest("Spectral elem - sphere divergence"       ,"Operators/spectralelement/sphere_divergence.jl"),
UnitTest("Spectral elem - sphere curl"             ,"Operators/spectralelement/sphere_curl.jl"),
UnitTest("Spectral elem - sphere diffusion"        ,"Operators/spectralelement/sphere_diffusion.jl"),
UnitTest("Spectral elem - sphere diffusion vec"    ,"Operators/spectralelement/sphere_diffusion_vec.jl"),
UnitTest("Spectral elem - sphere hyperdiff"        ,"Operators/spectralelement/unit_sphere_hyperdiffusion.jl"),
UnitTest("Spectral elem - sphere hyperdiff vec"    ,"Operators/spectralelement/unit_sphere_hyperdiffusion_vec.jl"),
# UnitTest("Spectral elem - sphere hyperdiff vec"    ,"Operators/spectralelement/sphere_geometry_distributed.jl"), # MPI-only
UnitTest("FD ops - column"                         ,"Operators/finitedifference/unit_column.jl"),
UnitTest("FD ops - opt"                            ,"Operators/finitedifference/opt.jl"),
UnitTest("FD ops - wfact"                          ,"Operators/finitedifference/wfact.jl"),
UnitTest("FD ops - linsolve"                       ,"Operators/finitedifference/linsolve.jl"),
# UnitTest("FD ops - examples"                       ,"Operators/finitedifference/opt_examples.jl"), # only opt tests? (check coverage)
UnitTest("Hybrid - 2D"                             ,"Operators/hybrid/unit_2d.jl"),
UnitTest("Hybrid - 3D"                             ,"Operators/hybrid/unit_3d.jl"),
UnitTest("Hybrid - dss opt"                        ,"Operators/hybrid/dss_opt.jl"),
UnitTest("Hybrid - opt"                            ,"Operators/hybrid/opt.jl"),
UnitTest("Thomas Algorithm"                        ,"Operators/unit_thomas_algorithm.jl"),
UnitTest("MatrixFields - BandMatrixRow"            ,"MatrixFields/band_matrix_row.jl"),
UnitTest("MatrixFields - field2arrays"             ,"MatrixFields/field2arrays.jl"),
UnitTest("MatrixFields - mat mul at boundaries"    ,"MatrixFields/matrix_multiplication_at_boundaries.jl"),
UnitTest("MatrixFields - field names"              ,"MatrixFields/field_names.jl"),
UnitTest("MatrixFields - broadcasting (1)"         ,"MatrixFields/matrix_fields_broadcasting/test_scalar_1.jl"),
# UnitTest("MatrixFields - matrix field broadcast"   ,"MatrixFields/matrix_field_broadcasting.jl"), # too long
# UnitTest("MatrixFields - operator matrices"        ,"MatrixFields/operator_matrices.jl"), # too long
# UnitTest("MatrixFields - field matrix solvers"     ,"MatrixFields/field_matrix_solvers.jl"), # too long
UnitTest("Hypsography - 2d"                        ,"Hypsography/2d.jl"),
UnitTest("Hypsography - 3d sphere"                 ,"Hypsography/3dsphere.jl"),
UnitTest("Remapping"                               ,"Operators/remapping.jl"),
UnitTest("Limiter"                                 ,"Limiters/limiter.jl"),
# UnitTest("Limiter"                                 ,"Limiters/distributed/dlimiter.jl"), # requires MPI
UnitTest("InputOutput - hdf5"                      ,"InputOutput/hdf5.jl"),
UnitTest("InputOutput - spectralelement2d"         ,"InputOutput/spectralelement2d.jl"),
UnitTest("InputOutput - hybrid2dbox"               ,"InputOutput/hybrid2dbox.jl"),
UnitTest("InputOutput - hybrid2dbox_topography"    ,"InputOutput/hybrid2dbox_topography.jl"),
UnitTest("InputOutput - hybrid2dbox_stretched"     ,"InputOutput/hybrid2dbox_stretched.jl"),
UnitTest("InputOutput - hybrid3dbox"               ,"InputOutput/hybrid3dbox.jl"),
UnitTest("InputOutput - hybrid3dcubedsphere"       ,"InputOutput/hybrid3dcubedsphere.jl"),
UnitTest("InputOutput - hybrid3dcubedsphere_topo"  ,"InputOutput/hybrid3dcubedsphere_topography.jl"),
UnitTest("Array interpolation"                     ,"Remapping/interpolate_array.jl"),
UnitTest("Array interpolation"                     ,"Remapping/distributed_remapping.jl"),
UnitTest("Aqua"                                    ,"aqua.jl"),
UnitTest("Deprecations"                            ,"deprecations.jl"),
UnitTest("GPU - cuda"                              ,"gpu/cuda.jl";meta=:gpu_only),
UnitTest("GPU - data"                              ,"DataLayouts/cuda.jl";meta=:gpu_only),
UnitTest("Spaces - serial CUDA DSS on CubedSphere" ,"Spaces/ddss1_cs.jl";meta=:gpu_only),
UnitTest("Operators - spectral element CUDA"       ,"Operators/spectralelement/rectilinear_cuda.jl";meta=:gpu_only),
UnitTest("Operators - finite difference CUDA"      ,"Operators/hybrid/unit_cuda.jl";meta=:gpu_only),
UnitTest("Operators - extruded sphere space ops"   ,"Operators/hybrid/extruded_sphere_cuda.jl";meta=:gpu_only),
UnitTest("Operators - extruded 3dbox space ops"    ,"Operators/hybrid/extruded_3dbox_cuda.jl";meta=:gpu_only),
UnitTest("Fields - CUDA mapreduce"                 ,"Fields/reduction_cuda.jl";meta=:gpu_only),
]
#! format: on

# `validate_tests` returns one of (`:duplicate_file`, `:non_existent_file`, `:pass`)
err = validate_tests(unit_tests; test_path = @__DIR__)

import ClimaComms
ClimaComms.@import_required_backends

filter!(
    test -> !(
        test.meta == :gpu_only &&
        !(ClimaComms.device() isa ClimaComms.CUDADevice)
    ),
    unit_tests,
)
# Note: for `fail_fast = false`, the tests are all wrapped in `@testset "Unit tests"`
#       so output is suppressed until all tests are complete.
fail_fast = true

# Use prevent_leaky_tests = !isnothing(get(ENV, "CI", nothing))
# once https://github.com/CliMA/ClimaCore.jl/issues/1826 is fixed:
# prevent_leaky_tests = !isnothing(get(ENV, "CI", nothing)) # prevent leaky tests on CI)
prevent_leaky_tests = true
# tabulate_tests(unit_tests; include_timings = false) # uncomment for preview

# If `fail_fast` is `true`, then let's error on invalid tests before starting them.
if fail_fast
    err == :duplicate_file && error("Please remove the duplicate file.")
    err == :non_existent_file && error("Please remove the non-existent file.")
end

run_unit_tests!(unit_tests; fail_fast, prevent_leaky_tests)
tabulate_tests(unit_tests)

# Early warning of duplicate and missing files, but lazy error:
err == :duplicate_file && error("Please remove the duplicate file.")
err == :non_existent_file && error("Please remove the non-existent file.")

nothing
