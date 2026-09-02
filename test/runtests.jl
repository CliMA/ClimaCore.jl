using Test
include("tabulated_tests.jl")

#! format: off
unit_tests = [
    # DataLayouts
    UnitTest("DataLayouts get_struct"                   ,"DataLayouts/unit_struct.jl"; tier = :unit, subsystem = :datalayouts),
    UnitTest("DataLayouts slice indexing & setindex!"   ,"DataLayouts/unit_indexing.jl"; tier = :unit, subsystem = :datalayouts),
    UnitTest("DataLayouts scopes"                       ,"DataLayouts/unit_scopes.jl"; tier = :unit, subsystem = :datalayouts),
    UnitTest("DataLayouts loops"                        ,"DataLayouts/unit_loops.jl"; tier = :unit, subsystem = :datalayouts),
    UnitTest("DataLayouts mapreduce"                    ,"DataLayouts/unit_mapreduce.jl"; tier = :unit, subsystem = :datalayouts),
    UnitTest("DataLayouts layout args"                  ,"DataLayouts/unit_layout_args.jl"; tier = :unit, subsystem = :datalayouts),
    UnitTest("DataLayouts fill and copyto"              ,"DataLayouts/unit_fill_and_copyto.jl"; tier = :unit, subsystem = :datalayouts, slow = true), # 1.5 min: ~200 concrete layout types to specialize
    UnitTest("DataLayouts - similar inference"          ,"DataLayouts/inference_similar.jl"; tier = :inference, subsystem = :datalayouts),

    # Utilities
    UnitTest("PlusHalf"                                 ,"Utilities/unit_plushalf.jl"; tier = :unit, subsystem = :utilities),
    UnitTest("Stable views"                             ,"Utilities/unit_stable_view.jl"; tier = :unit, subsystem = :utilities),
    UnitTest("AutoBroadcaster"                          ,"Utilities/unit_auto_broadcaster.jl"; tier = :unit, subsystem = :utilities),
    UnitTest("DebugOnly"                                ,"Utilities/unit_debug_only.jl"; tier = :unit, subsystem = :utilities),

    # Domains
    UnitTest("Domains"                                  ,"Domains/unit_domains.jl"; tier = :unit, subsystem = :domains),

    # Geometry
    UnitTest("Geometry"                                 ,"Geometry/unit_geometry.jl"; tier = :unit, subsystem = :geometry),
    UnitTest("mul_with_projection"                      ,"Geometry/unit_mul_with_projection.jl"; tier = :unit, subsystem = :geometry),
    UnitTest("Tensors"                                  ,"Geometry/unit_tensors.jl"; tier = :unit, subsystem = :geometry),

    # Meshes
    UnitTest("Interval mesh"                            ,"Meshes/unit_interval.jl"; tier = :unit, subsystem = :meshes),
    UnitTest("Rectangle mesh"                           ,"Meshes/unit_rectangle.jl"; tier = :unit, subsystem = :meshes),
    UnitTest("Cubedsphere mesh"                         ,"Meshes/unit_cubedsphere.jl"; tier = :unit, subsystem = :meshes),
    UnitTest("Meshes - inference"                       ,"Meshes/inference_meshes.jl"; tier = :inference, subsystem = :meshes),

    # Topologies
    UnitTest("Interval topology"                        ,"Topologies/unit_interval.jl"; tier = :unit, subsystem = :topologies),
    UnitTest("Rectangle topology"                       ,"Topologies/unit_rectangle.jl"; tier = :unit, subsystem = :topologies),
    UnitTest("Rectangle surface topology"               ,"Topologies/unit_rectangle_sfc.jl"; tier = :unit, subsystem = :topologies),
    UnitTest("Cubedsphere topology"                     ,"Topologies/unit_cubedsphere.jl"; tier = :unit, subsystem = :topologies),
    UnitTest("Cubedsphere surface topology"             ,"Topologies/unit_cubedsphere_sfc.jl"; tier = :unit, subsystem = :topologies),
    UnitTest("dss_transform"                            ,"Topologies/unit_dss_transform.jl"; tier = :unit, subsystem = :topologies),

    # Quadratures & CommonSpaces
    UnitTest("Quadratures"                              ,"Quadratures/unit_quadratures.jl"; tier = :unit, subsystem = :quadratures),
    UnitTest("CommonGrids"                              ,"CommonGrids/unit_common_grids.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("CommonSpaces"                             ,"CommonSpaces/unit_common_spaces.jl"; tier = :unit, subsystem = :spaces),

    # Spaces
    UnitTest("Spaces"                                   ,"Spaces/unit_spaces.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("dss"                                      ,"Spaces/unit_dss.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("Spaces - exact 2x2 DSS"                   ,"Spaces/unit_dss_exact.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("Spaces - DSS vs grouped reference"        ,"Spaces/unit_dss_reference.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("Spaces - serial CPU DSS"                  ,"Spaces/unit_serial_cpu_dss.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("Spaces - discontinuous (DG) spaces"       ,"Spaces/unit_discontinuous_spaces.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("Sphere spaces"                            ,"Spaces/unit_sphere_spaces.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("Spaces - high resolution"                 ,"Spaces/unit_high_resolution_space.jl"; tier = :unit, subsystem = :spaces),
    UnitTest("Terrain warp"                             ,"Spaces/unit_terrain_warp.jl"; tier = :unit, subsystem = :spaces, slow = true), # 5.9 min: sweeps npoly/helem up to 10 in 3D
    UnitTest("Spaces - inference"                       ,"Spaces/inference_spaces.jl"; tier = :inference, subsystem = :spaces),

    # Fields
    UnitTest("Fields"                                   ,"Fields/unit_field.jl"; tier = :unit, subsystem = :fields),
    UnitTest("Fields - fused slice loops"              ,"Fields/unit_fusion.jl"; tier = :unit, subsystem = :fields, slow = true), # compiles the spectral stack for every space; excluded from the one-process GHA jobs
    UnitTest("Fields - inference regression"            ,"Fields/inference_repro.jl"; tier = :inference, subsystem = :fields),
    UnitTest("Fields - zero-allocation broadcasts"      ,"Fields/allocs_field.jl"; meta = :cpu_only, tier = :allocs, subsystem = :fields),
    UnitTest("Fields - convergence integrals"           ,"Fields/conv_field_integrals.jl"; tier = :conv, subsystem = :fields),
    UnitTest("Fields - multi broadcast fusion"          ,"Fields/unit_field_multi_broadcast_fusion.jl"; tier = :unit, subsystem = :fields),
    UnitTest("Fields - FieldVector flattening"          ,"Fields/unit_fieldvector_flatten.jl"; tier = :unit, subsystem = :fields),
    UnitTest("Fields - inference"                       ,"Fields/inference_fields.jl"; meta = :cpu_only, tier = :inference, subsystem = :fields),
    UnitTest("Placeholder Fields"                       ,"Operators/unit_common.jl"; tier = :unit, subsystem = :operators),

    # Spectral Element Operators
    UnitTest("Spectral elem - vector identities"        ,"Operators/spectralelement/unit_vector_identities.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - rectilinear"              ,"Operators/spectralelement/unit_rectilinear.jl"; tier = :unit, subsystem = :operators, slow = true), # 28 min under the GHA coverage job; Buildkite runs it uninstrumented
    UnitTest("Spectral elem - plane"                    ,"Operators/spectralelement/unit_plane.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - inference"                ,"Operators/spectralelement/inference_spectralelement.jl"; tier = :inference, subsystem = :operators),
    UnitTest("Spectral elem - gradient tensor"          ,"Operators/spectralelement/unit_covar_deriv_ops.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - Diffusion 2d"             ,"Operators/spectralelement/unit_diffusion2d.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - sphere geometry"          ,"Operators/spectralelement/unit_sphere_geometry.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - sphere gradient"          ,"Operators/spectralelement/conv_sphere_gradient.jl"; tier = :conv, subsystem = :operators),
    UnitTest("Spectral elem - sphere divergence"        ,"Operators/spectralelement/conv_sphere_divergence.jl"; tier = :conv, subsystem = :operators),
    # :cpu_only: the analytic λ_lm evaluation in utils_vsh_divergence.jl is
    # host-only; the numerics they check are device-independent.
    UnitTest("Spectral elem - VSH divergence"           ,"Operators/spectralelement/unit_sphere_vsh_divergence.jl"; meta = :cpu_only, tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - divergence jump conv"     ,"Operators/spectralelement/conv_sphere_divergence_jump.jl"; meta = :cpu_only, tier = :conv, subsystem = :operators),
    UnitTest("Spectral elem - sphere curl"              ,"Operators/spectralelement/conv_sphere_curl.jl"; tier = :conv, subsystem = :operators),
    UnitTest("Spectral elem - sphere diffusion"         ,"Operators/spectralelement/unit_sphere_diffusion.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - sphere diffusion vec"     ,"Operators/spectralelement/conv_sphere_diffusion_vec.jl"; tier = :conv, subsystem = :operators),
    UnitTest("Spectral elem - split divergence"         ,"Operators/spectralelement/unit_split_divergence.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - sphere hyperdiff"         ,"Operators/spectralelement/unit_sphere_hyperdiffusion.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - sphere hyperdiff vec"     ,"Operators/spectralelement/unit_sphere_hyperdiffusion_vec.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - sphere hyperdiff vec conv" ,"Operators/spectralelement/conv_sphere_hyperdiffusion_vec.jl"; tier = :conv, subsystem = :operators),
    UnitTest("Spectral elem - over-integration"         ,"Operators/spectralelement/unit_overintegration.jl"; meta = :cpu_only, tier = :unit, subsystem = :operators),
    # `add_numerical_flux_interior!`/`_boundary!` walk faces in a host loop and
    # scalar-index the data, so tests built on them either wrap the calls in a
    # scoped `allowscalar` (the two below) or are :cpu_only (the three after).
    UnitTest("Spectral elem - DG two-point fluxes"      ,"Operators/spectralelement/unit_two_point_fluxes.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - DG sphere fluxes"         ,"Operators/spectralelement/unit_sphere_dg_fluxes.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - DG stability properties"  ,"Operators/spectralelement/unit_dg_stability.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - DG boundary fluxes"       ,"Operators/spectralelement/unit_dg_boundary_fluxes.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - Laplacian atoms"          ,"Operators/spectralelement/unit_laplacians.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - DG extruded sphere"       ,"Operators/spectralelement/unit_extruded_sphere_dg.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - DG extruded plane"        ,"Operators/spectralelement/unit_extruded_plane_dg.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - tendency completion"      ,"Operators/spectralelement/unit_tendency_completion.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Spectral elem - DG divergence conv"       ,"Operators/spectralelement/conv_dg_divergence.jl"; meta = :cpu_only, tier = :conv, subsystem = :operators),
    UnitTest("Operators - broadcast inference"          ,"Operators/inference_operators.jl"; tier = :inference, subsystem = :operators),
    UnitTest("FD ops - zero-allocation stencils"        ,"Operators/finitedifference/allocs_fd_ops.jl"; meta = :cpu_only, tier = :allocs, subsystem = :operators),
    UnitTest("SEM ops - zero-allocation broadcasts"     ,"Operators/spectralelement/allocs_spectral_ops.jl"; meta = :cpu_only, tier = :allocs, subsystem = :operators),

    # Finite Difference & Hybrid Operators
    UnitTest("FD ops - column"                          ,"Operators/finitedifference/unit_column.jl"; tier = :unit, subsystem = :operators),
    UnitTest("FD ops - tensor"                          ,"Operators/finitedifference/unit_tensor.jl"; tier = :unit, subsystem = :operators, slow = true), # largest instrumented compile peak; the coverage job cannot afford it
    UnitTest("FD ops - boundary symmetry"               ,"Operators/finitedifference/unit_boundary_symmetry.jl"; tier = :unit, subsystem = :operators),
    UnitTest("FD ops - upwind schemes"                  ,"Operators/finitedifference/unit_upwind_schemes.jl"; tier = :unit, subsystem = :operators),
    UnitTest("FD ops - inference"                       ,"Operators/finitedifference/inference_finitedifference.jl"; meta = :cpu_only, tier = :inference, subsystem = :operators),
    UnitTest("FD ops - inference examples"              ,"Operators/finitedifference/inference_examples.jl"; tier = :inference, subsystem = :operators),
    UnitTest("FD ops - column conv"                     ,"Operators/finitedifference/conv_column.jl"; tier = :conv, subsystem = :operators),
    UnitTest("FD ops - advection-diffusion conv"        ,"Operators/finitedifference/conv_advection_diffusion1d.jl"; tier = :conv, subsystem = :operators),
    UnitTest("Hybrid - 2D"                              ,"Operators/hybrid/unit_2d.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Hybrid - 3D"                              ,"Operators/hybrid/unit_3d.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Hybrid - 3D simulation"                   ,"Operators/hybrid/simulation_3d.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Hybrid - 2D convergence"                  ,"Operators/hybrid/conv_2d.jl"; tier = :conv, subsystem = :operators),
    UnitTest("Hybrid - 3D convergence"                  ,"Operators/hybrid/conv_3d.jl"; tier = :conv, subsystem = :operators),
    UnitTest("Operators - remapping"                    ,"Operators/unit_remapping.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Operators - div/grad adjoint identity"    ,"Operators/unit_adjoint_identity.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Operators - levels & extruded examples"   ,"Operators/unit_operators_examples.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Operators - integrals"                    ,"Operators/integrals.jl"; tier = :unit, subsystem = :operators),
    UnitTest("Hybrid - dss inference"                   ,"Operators/hybrid/inference_dss.jl"; tier = :inference, subsystem = :operators),
    # :cpu_only for the same reason as the other opt-tier tests: JET reports the
    # runtime dispatch in CUDA's kernel-launch path, which is not ours to fix.
    UnitTest("Hybrid - inference"                       ,"Operators/hybrid/inference_hybrid.jl"; meta = :cpu_only, tier = :inference, subsystem = :operators),

    # MatrixFields
    UnitTest("MatrixFields - BandMatrixRow"             ,"MatrixFields/unit_band_matrix_row.jl"; tier = :unit, subsystem = :matrixfields),
    UnitTest("MatrixFields - field2arrays"              ,"MatrixFields/unit_field2arrays.jl"; meta = :cpu_only, tier = :unit, subsystem = :matrixfields),
    UnitTest("MatrixFields - mat mul at boundaries"     ,"MatrixFields/unit_matrix_multiplication_at_boundaries.jl"; tier = :unit, subsystem = :matrixfields),
    UnitTest("MatrixFields - field names"               ,"MatrixFields/unit_field_names.jl"; tier = :unit, subsystem = :matrixfields),
    UnitTest("MatrixFields - solvers"                   ,"MatrixFields/unit_field_matrix_solvers.jl"; tier = :unit, subsystem = :matrixfields, slow = true), # 5.3 min
    # Umbrella over matrix_fields_broadcasting/; see the README there. :cpu_only
    # because on GPU several cases take minutes to compile apiece and need a
    # process each; Buildkite runs them as a matrix, one case per job.
    UnitTest("MatrixFields - broadcasting"              ,"MatrixFields/unit_matrix_field_broadcasting.jl"; meta = :cpu_only, tier = :unit, subsystem = :matrixfields, slow = true), # 3.1 min for all 22 cases
    UnitTest("MatrixFields - multiple field solve"      ,"MatrixFields/multiple_field_solve_reproducer_1.jl"; tier = :unit, subsystem = :matrixfields),
    UnitTest("MatrixFields - flat spaces"               ,"MatrixFields/unit_flat_spaces.jl"; tier = :unit, subsystem = :matrixfields),
    UnitTest("MatrixFields - indexing"                  ,"MatrixFields/unit_field_matrix_indexing.jl"; tier = :unit, subsystem = :matrixfields, slow = true), # +1.5 GiB instrumented peak on the coverage job
    UnitTest("MatrixFields - operator matrices"         ,"MatrixFields/unit_operator_matrices.jl"; tier = :unit, subsystem = :matrixfields, slow = true), # 4.3 min
    UnitTest("MatrixFields - mat mul recursion"         ,"MatrixFields/unit_matrix_multiplication_recursion.jl"; tier = :unit, subsystem = :matrixfields),

    # Hypsography
    UnitTest("Hypsography - 2d"                         ,"Hypsography/unit_hypsography_2d.jl"; tier = :unit, subsystem = :hypsography),
    UnitTest("Hypsography - 3d sphere"                  ,"Hypsography/unit_hypsography_3dsphere.jl"; tier = :unit, subsystem = :hypsography, slow = true), # +3 GiB instrumented peak; the coverage job's repeated OOM site

    # Limiters, IO, Remapping
    UnitTest("Limiter"                                  ,"Limiters/unit_limiter.jl"; tier = :unit, subsystem = :limiters),
    UnitTest("Limiters - vertical mass borrowing"       ,"Limiters/vertical_mass_borrowing_limiter.jl"; tier = :unit, subsystem = :limiters),
    UnitTest("Limiters - vertical mass borrowing adv"   ,"Limiters/vertical_mass_borrowing_limiter_advection.jl"; tier = :unit, subsystem = :limiters, slow = true), # 2.3 min
    UnitTest("InputOutput - hdf5"                       ,"InputOutput/unit_hdf5.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - all-spaces round-trip"      ,"InputOutput/unit_allspaces_roundtrip.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - parse_type"                 ,"InputOutput/unit_read_type.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - spectralelement2d"          ,"InputOutput/unit_spectralelement2d.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - hybrid2dbox"                ,"InputOutput/unit_hybrid2dbox.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - hybrid2dbox_topography"     ,"InputOutput/unit_hybrid2dbox_topography.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - hybrid2dbox_stretched"      ,"InputOutput/unit_hybrid2dbox_stretched.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - hybrid3dbox"                ,"InputOutput/unit_hybrid3dbox.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - hybrid3dcubedsphere"        ,"InputOutput/unit_hybrid3dcubedsphere.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - hybrid3dcubedsphere_topo"   ,"InputOutput/unit_hybrid3dcubedsphere_topography.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - finitedifferences"          ,"InputOutput/unit_finitedifference.jl"; tier = :unit, subsystem = :io),
    UnitTest("InputOutput - pointspaces"                ,"InputOutput/unit_point.jl"; meta = :cpu_only, tier = :unit, subsystem = :io),
    UnitTest("Array interpolation"                      ,"Remapping/unit_interpolate_array.jl"; tier = :unit, subsystem = :remapping),
    UnitTest("Distributed remapping"                    ,"Remapping/unit_distributed_remapping.jl"; tier = :unit, subsystem = :remapping),
    UnitTest("Vertical interpolation"                   ,"Remapping/unit_interpolate_pressure.jl"; tier = :unit, subsystem = :remapping),

    # Integration
    UnitTest("Integration - Bickley Jet (CG & DG)"      ,"Integration/smoke_bickley_jet_cg_dg.jl"; tier = :smoke, subsystem = :integration),
    UnitTest("Integration - 3D Baroclinic wave"         ,"Integration/smoke_baroclinic_wave.jl"; tier = :smoke, subsystem = :integration),
    UnitTest("Integration - Solid-body rotation 3D"     ,"Integration/smoke_solid_body_rotation.jl"; meta = :cpu_only, tier = :smoke, subsystem = :integration),
    UnitTest("Integration - column FCT/van Leer advection", "Integration/smoke_column_advection.jl"; meta = :cpu_only, tier = :smoke, subsystem = :limiters),

    # Quality & Deprecations
    UnitTest("Aqua"                                     ,"aqua.jl"; tier = :misc, subsystem = :quality),
    UnitTest("Deprecations"                             ,"deprecations.jl"; tier = :misc, subsystem = :quality),
    UnitTest("Precompile workload"                      ,"precompile_workload.jl"; tier = :misc, subsystem = :quality, slow = true), # ~1 min: recompiles ClimaCore in a subprocess

    # GPU Only
    UnitTest("GPU - cuda"                               ,"gpu/cuda.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("GPU - compiler stress regression"         ,"gpu/compiler_stress_regression.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("GPU - kernel renaming"                    ,"gpu/kernel_renaming.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("GPU - data"                               ,"DataLayouts/gpu_cuda.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("DataLayouts - CUDA threadblocks"          ,"DataLayouts/gpu_cuda_threadblocks.jl"; meta = :gpu_only, tier = :gpu, subsystem = :datalayouts),
    UnitTest("Spaces - CUDA extruded spaces"            ,"Spaces/extruded_cuda.jl"; meta = :gpu_only, tier = :gpu, subsystem = :spaces),
    UnitTest("Spaces - CUDA point spaces"               ,"Spaces/point_cuda.jl"; meta = :gpu_only, tier = :gpu, subsystem = :spaces),
    UnitTest("Operators - spectral element CUDA"        ,"Operators/spectralelement/gpu_rectilinear.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("Operators - finite difference CUDA"       ,"Operators/hybrid/gpu_ops.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("Operators - FD broadcasting edge cases"   ,"Operators/finitedifference/broadcasting_edge_cases.jl"; meta = :gpu_only, tier = :gpu, subsystem = :operators),
    UnitTest("Operators - edge-case meshes"             ,"Operators/gpu_edge_cases.jl"; meta = :gpu_only, tier = :gpu, subsystem = :operators),
    UnitTest("Operators - extruded sphere space ops"    ,"Operators/hybrid/gpu_extruded_sphere.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("Operators - extruded 3dbox space ops"     ,"Operators/hybrid/gpu_extruded_3dbox.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("Operators - hybrid simulation CUDA"       ,"Operators/hybrid/simulation_cuda.jl"; meta = :gpu_only, tier = :gpu, subsystem = :operators),
    UnitTest("Fields - CUDA mapreduce"                  ,"Fields/gpu_reduction.jl"; meta = :gpu_only, tier = :gpu, subsystem = :gpu),
    UnitTest("MatrixFields - GPU compat bidiag row"     ,"MatrixFields/gpu_compat_bidiag_matrix_row.jl"; meta = :gpu_only, tier = :gpu, subsystem = :matrixfields),
    # `MatrixFields/gpu_matrix_field_broadcasting.jl` is absent because its
    # cases need a process apiece, which this harness cannot give, since every
    # test runs in one process. The `gpu_matrix_field_broadcasting` Buildkite
    # matrix drives them, one job per case.

    # Distributed (MPI) — registered for taxonomy completeness; filtered out on
    # single-process runs and driven by CI at fixed rank counts via srun/mpiexec
    # (see .buildkite/pipeline.yml).
    UnitTest("Distributed - topology (4 ranks)"         ,"Topologies/dtopo4.jl"; meta = :distributed, tier = :unit, subsystem = :topologies),
    UnitTest("Distributed - DSS (2 ranks)"              ,"Spaces/distributed/ddss2.jl"; meta = :distributed, tier = :unit, subsystem = :spaces),
    UnitTest("Distributed - DG operators (2 ranks)"     ,"Operators/spectralelement/distributed/ddg2.jl"; meta = :distributed, tier = :unit, subsystem = :operators),
    UnitTest("Distributed - DG operators (3 ranks)"     ,"Operators/spectralelement/distributed/ddg3.jl"; meta = :distributed, tier = :unit, subsystem = :operators),
    UnitTest("Distributed - DSS (3 ranks)"              ,"Spaces/distributed/ddss3.jl"; meta = :distributed, tier = :unit, subsystem = :spaces),
    UnitTest("Distributed - DSS (4 ranks)"              ,"Spaces/distributed/ddss4.jl"; meta = :distributed, tier = :unit, subsystem = :spaces),
    UnitTest("Distributed - gather (4 ranks)"           ,"Spaces/distributed/gather4.jl"; meta = :distributed, tier = :unit, subsystem = :spaces),
    UnitTest("Distributed - limiter (3 ranks)"          ,"Limiters/distributed/dlimiter.jl"; meta = :distributed, tier = :unit, subsystem = :limiters),
    UnitTest("Distributed - SEM sphere geometry"        ,"Operators/spectralelement/sphere_geometry_distributed.jl"; meta = :distributed, tier = :unit, subsystem = :operators),
    UnitTest("Distributed - GPU space construction"     ,"Spaces/distributed_cuda/space_construction.jl"; meta = :distributed, tier = :gpu, subsystem = :gpu),
    UnitTest("Distributed - GPU DSS (2 ranks)"          ,"Spaces/distributed_cuda/ddss2.jl"; meta = :distributed, tier = :gpu, subsystem = :gpu),
    UnitTest("Distributed - GPU DSS (3 ranks)"          ,"Spaces/distributed_cuda/ddss3.jl"; meta = :distributed, tier = :gpu, subsystem = :gpu),
    UnitTest("Distributed - GPU DSS (4 ranks)"          ,"Spaces/distributed_cuda/ddss4.jl"; meta = :distributed, tier = :gpu, subsystem = :gpu),
    UnitTest("Distributed - GPU DSS ne32 cubed sphere"  ,"Spaces/distributed_cuda/ddss_ne32_cs.jl"; meta = :distributed, tier = :gpu, subsystem = :gpu),
    UnitTest("Distributed - GPU reduction (2 ranks)"    ,"Fields/gpu_reduction_distributed.jl"; meta = :distributed, tier = :gpu, subsystem = :gpu),
]
#! format: on

# `validate_tests` returns one of (`:duplicate_file`, `:non_existent_file`, `:pass`)
err = validate_tests(unit_tests; test_path = @__DIR__)

import ClimaComms
ClimaComms.@import_required_backends

# Device / launch-mode filtering on `meta`:
#  - :gpu_only     requires a CUDA device;
#  - :cpu_only     is skipped when running on a CUDA device;
#  - :distributed  requires a multi-rank (mpiexec/srun) launch — CI drives
#    these directly at fixed rank counts (see `.buildkite/pipeline.yml`).
let
    on_cuda = ClimaComms.device() isa ClimaComms.CUDADevice
    multirank = ClimaComms.nprocs(ClimaComms.context()) > 1
    skip(test) =
        (test.meta == :gpu_only && !on_cuda) ||
        (test.meta == :cpu_only && on_cuda) ||
        (test.meta == :distributed && !multirank)
    filter!(!skip, unit_tests)
end

# Optional CLI / Environment-based test filtering:
# e.g. TEST_TIER=unit, TEST_EXCLUDE_TIER=conv,inference, TEST_SUBSYSTEM=operators,
# TEST_EXCLUDE_SUBSYSTEM=operators, TEST_TAG=dg, TEST_FAST=true,
# TEST_EXCLUDE_SLOW=true
let
    tier_filter = get(ENV, "TEST_TIER", nothing)
    exclude_tier_filter = get(ENV, "TEST_EXCLUDE_TIER", nothing)
    subsystem_filter = get(ENV, "TEST_SUBSYSTEM", nothing)
    exclude_subsystem_filter = get(ENV, "TEST_EXCLUDE_SUBSYSTEM", nothing)
    tag_filter = get(ENV, "TEST_TAG", nothing)
    fast_filter = get(ENV, "TEST_FAST", "false") == "true"
    exclude_slow_filter = get(ENV, "TEST_EXCLUDE_SLOW", "false") == "true"
    if !isnothing(tier_filter) || !isnothing(exclude_tier_filter) ||
       !isnothing(subsystem_filter) || !isnothing(exclude_subsystem_filter) ||
       !isnothing(tag_filter) || fast_filter || exclude_slow_filter
        filtered = filter_tests(
            unit_tests;
            tier = tier_filter,
            exclude_tier = exclude_tier_filter,
            subsystem = subsystem_filter,
            exclude_subsystem = exclude_subsystem_filter,
            tag = tag_filter,
            fast = fast_filter,
            exclude_slow = exclude_slow_filter,
        )
        empty!(unit_tests)
        append!(unit_tests, filtered)
    end
end

# The :inference tier pins JET report counts to the versions in the current
# Manifest. Downgrade forces the oldest compatible versions, under which those
# counts drift for reasons outside this package, so it skips the tier.
# test/aqua.jl consults the same variable.
if get(ENV, "CLIMACORE_DOWNGRADE_TESTS", "false") == "true"
    filter!(test -> test.tier != :inference, unit_tests)
end

# Fail if the filters selected nothing: a typo in
# TEST_TIER/TEST_SUBSYSTEM/TEST_TAG must not produce a green zero-test run,
# nor an opaque MethodError from summing over an empty list.
if isempty(unit_tests)
    filter_env = filter(
        !isnothing,
        [
            (k => get(ENV, k, nothing)) for k in (
                "TEST_TIER",
                "TEST_EXCLUDE_TIER",
                "TEST_SUBSYSTEM",
                "TEST_EXCLUDE_SUBSYSTEM",
                "TEST_TAG",
                "TEST_FAST",
                "TEST_EXCLUDE_SLOW",
            ) if haskey(ENV, k)
        ],
    )
    error(
        "No tests matched the requested filters ($(filter_env)). " *
        "Check the tier/subsystem/tag spelling against test/runtests.jl. " *
        "Note: GPU-only tests are also dropped when no CUDA device is available.",
    )
end

# `TEST_FAIL_FAST=false` runs all tests and summarizes failures at the end
# (coarser output, finer failure coverage); default is to stop at the first
# failing file.
fail_fast = get(ENV, "TEST_FAIL_FAST", "true") != "false"
prevent_leaky_tests = true

function error_on_invalid_tests(err)
    err == :duplicate_file && error("Please remove the duplicate file.")
    err == :non_existent_file && error("Please remove the non-existent file.")
    return nothing
end

# With `fail_fast`, error on invalid tests before starting them; otherwise
# defer the error until after all tests have run.
fail_fast && error_on_invalid_tests(err)

run_unit_tests!(unit_tests; fail_fast, prevent_leaky_tests)
tabulate_tests(unit_tests)

error_on_invalid_tests(err)

nothing
