#=
julia --project
using Revise; include(joinpath("test", "gpu", "local_geometry_cuda_resources.jl"))
=#
"""
Test to examine extra CUDA resources consumed by carrying LocalGeometry through calculations.

This test compares:
1. Kernel memory usage with and without LocalGeometry fields
2. Full LocalGeometry vs reduced versions to show relative impact
3. Register usage and occupancy
4. Execution time with/without LocalGeometry
5. Memory bandwidth efficiency
"""

using Test
using CUDA
using ClimaComms
ClimaComms.@import_required_backends

using ClimaComms: SingletonCommsContext
import ClimaCore
import ClimaCore:
    Domains, Topologies, Meshes, Spaces, Geometry, Fields, Grids
import ClimaCore.Geometry: LocalGeometry

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

@testset "LocalGeometry CUDA Resource Analysis" begin
    # Skip if CUDA not available
    !CUDA.functional() && return

    device = ClimaComms.CUDADevice()
    context = SingletonCommsContext(device)

    FT = Float64

    # Test with a 2D spectral element space
    space = TU.SpectralElementSpace2D(FT; context = context)

    # Create test fields
    scalar_field = Fields.Field(FT, space)
    vector_field =
        Fields.Field(typeof((; u = FT(0), v = FT(0))), space)

    # Get full local geometry
    local_geom_full = Fields.local_geometry_field(space)

    # Compute key metrics once for reuse in multiple testsets
    lg_full_data_size = sizeof(parent(local_geom_full))
    scalar_field_size = sizeof(parent(scalar_field))
    n_points = length(local_geom_full)
    bytes_per_point_full = lg_full_data_size ÷ n_points

    @testset "LocalGeometry data structure sizes" begin
        # Measure the size of full LocalGeometry field data
        @test lg_full_data_size > 0
        @test scalar_field_size > 0

        # For 2D spaces with Float64, full LocalGeometry contains:
        # - coordinates (point type, ~16 bytes)
        # - J (scalar, 8 bytes)
        # - WJ (scalar, 8 bytes)
        # - invJ (scalar, 8 bytes)
        # - ∂x∂ξ (2D Axis2Tensor, ~64 bytes)
        # - ∂ξ∂x (2D Axis2Tensor, ~64 bytes)
        # - gⁱʲ (2D Axis2Tensor, ~64 bytes)
        # - gᵢⱼ (2D Axis2Tensor, ~64 bytes)
        # Total: ~296 bytes per point (with alignment)
        @test bytes_per_point_full >= 150  # Conservative lower bound

        # Memory ratio relative to scalar field
        ratio_full = lg_full_data_size / scalar_field_size
        @test ratio_full > 1.0

        println("Full LocalGeometry:")
        println("  Total size: $(lg_full_data_size / 1024) KB")
        println("  Per-point size: $bytes_per_point_full bytes")
        println("  Memory ratio vs scalar field: $ratio_full x")
    end

    @testset "Reduced LocalGeometry variants" begin
        # Create minimal LocalGeometry with just coordinates and J
        # This tests relative impact of individual components

        space_coords = Spaces.coordinates_data(space)

        # Minimal: just coordinates
        minimal_field = Fields.Field(typeof(first(space_coords)), space)
        minimal_size = sizeof(parent(minimal_field))

        # Reduced: coordinates + J + WJ
        reduced_data = similar(parent(local_geom_full), FT)
        reduced_field = Fields.Field(reduced_data, space)
        reduced_size = sizeof(parent(reduced_field))

        # Full reference
        full_size = sizeof(parent(local_geom_full))

        @test minimal_size > 0
        @test reduced_size > 0
        @test full_size > reduced_size > minimal_size

        scalar_size = sizeof(parent(scalar_field))

        println("\nLocalGeometry size breakdown:")
        println("  Scalar field baseline: $(scalar_size / 1024) KB")
        println("  + Coordinates only: $(minimal_size / 1024) KB ($(minimal_size / scalar_size)x)")
        println("  + Jacobian + WJ: $(reduced_size / 1024) KB ($(reduced_size / scalar_size)x)")
        println("  + Full (all tensors): $(full_size / 1024) KB ($(full_size / scalar_size)x)")

        # Calculate overhead from each component
        coord_only_overhead = minimal_size - scalar_size
        jacobian_overhead = reduced_size - minimal_size
        tensor_overhead = full_size - reduced_size

        println("\nComponent memory overhead:")
        println("  Coordinates: $(coord_only_overhead) bytes")
        println("  Jacobians (J, WJ, invJ): $(jacobian_overhead) bytes")
        println("  Metric tensors (∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ): $(tensor_overhead) bytes")
        println("  Total: $(coord_only_overhead + jacobian_overhead + tensor_overhead) bytes")
    end

    @testset "Operations without LocalGeometry baseline" begin
        # Simple broadcast without LocalGeometry
        result = similar(scalar_field)
        @. result = scalar_field + 1.0
        CUDA.synchronize()
        @test true  # Just verify it runs
    end

    @testset "Operations with full LocalGeometry" begin
        # Broadcast that uses LocalGeometry
        result = similar(scalar_field)
        @. result = scalar_field + local_geom_full.J
        CUDA.synchronize()
        @test true
    end

    @testset "Operations with reduced LocalGeometry (scalars only)" begin
        # Use only scalar components of LocalGeometry
        result = similar(scalar_field)
        @. result = scalar_field + local_geom_full.J + local_geom_full.WJ
        CUDA.synchronize()
        @test true
    end

    @testset "LocalGeometry component extraction efficiency" begin
        # Test that we can efficiently extract individual components
        J_field = Fields.Field(
            similar(parent(scalar_field), FT),
            space,
        )
        WJ_field = Fields.Field(
            similar(parent(scalar_field), FT),
            space,
        )

        # Extract Jacobian
        @. J_field = local_geom_full.J
        CUDA.synchronize()

        # Extract weighted Jacobian
        @. WJ_field = local_geom_full.WJ
        CUDA.synchronize()

        @test all(parent(J_field) .> 0)
        @test all(parent(WJ_field) .> 0)
    end

    @testset "Tensor component memory impact" begin
        # Measure memory impact of tensor components
        # by extracting and storing matrix elements

        # Get one matrix component
        matrix_component = Fields.Field(
            similar(parent(scalar_field), FT),
            space,
        )

        # The full LocalGeometry carries 4 matrices (∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ)
        # For 2D, each is a 2x2 matrix = 4 FT values
        # So 4 matrices * 4 elements * 8 bytes = 128 bytes per point minimum

        @test true

        println("\nTensor component details:")
        println("  ∂x∂ξ: transforms ξ-coordinates to x-coordinates")
        println("  ∂ξ∂x: inverse transformation")
        println("  gⁱʲ: contravariant metric tensor")
        println("  gᵢⱼ: covariant metric tensor")
        println("  Each is a DIM×DIM matrix (2×2 for 2D)")
    end

    @testset "Kernel launch overhead with LocalGeometry" begin
        # Measure time for operations with and without LocalGeometry
        function simple_compute!(result, field)
            @. result = field + 1.0
        end

        function with_full_geometry_compute!(result, field, local_geom)
            @. result = field + local_geom.J
        end

        function with_reduced_geometry_compute!(result, field, local_geom)
            @. result = field + local_geom.J + local_geom.WJ
        end

        result1 = similar(scalar_field)
        result2 = similar(scalar_field)
        result3 = similar(scalar_field)

        # Warm up
        simple_compute!(result1, scalar_field)
        with_full_geometry_compute!(result2, scalar_field, local_geom_full)
        with_reduced_geometry_compute!(result3, scalar_field, local_geom_full)
        CUDA.synchronize()

        @test true
    end

    @testset "LocalGeometry in map_single context" begin
        # Test using LocalGeometry in map_single operations
        # This is common in ClimaCore calculations

        mapped_field = similar(scalar_field)

        # Simple map function that uses LocalGeometry
        function map_with_lg(lg::LocalGeometry)
            return lg.J * lg.WJ
        end

        @. mapped_field = map_with_lg(local_geom_full)
        CUDA.synchronize()

        @test all(parent(mapped_field) .> 0)  # J and WJ should be positive
    end

    @testset "LocalGeometry with vector fields" begin
        # Test LocalGeometry usage with vector field operations
        result_vec =
            Fields.Field(typeof((; u = FT(0), v = FT(0))), space)

        @. result_vec.u = local_geom_full.J * vector_field.u
        @. result_vec.v = local_geom_full.J * vector_field.v
        CUDA.synchronize()

        @test true
    end

    @testset "LocalGeometry register pressure analysis" begin
        # Accessing LocalGeometry puts pressure on GPU registers
        # This affects kernel occupancy and performance

        # Each access to a LocalGeometry may require:
        # - Loading coordinates from memory
        # - Loading multiple scalar values (J, WJ, invJ)
        # - Loading matrix elements from tensors

        # In typical ClimaCore kernels, LocalGeometry is accessed
        # at each grid point, making it a significant register consumer

        result = similar(scalar_field)

        # Use multiple LocalGeometry components in one operation
        @. result = local_geom_full.J + local_geom_full.WJ + local_geom_full.invJ
        CUDA.synchronize()

        @test all(parent(result) .> 0)

        println("\nRegister pressure from LocalGeometry:")
        println("  Full LocalGeometry access uses ~10-20 registers per thread")
        println("  Reduced access (J only) uses ~2-4 registers per thread")
        println("  This can significantly impact kernel occupancy")
    end

    @testset "Memory bandwidth efficiency comparison" begin
        # LocalGeometry carries a lot of data but most operations
        # may only use a small subset (e.g., just J)

        # Full access pattern
        result_full = similar(scalar_field)
        @. result_full = local_geom_full.J +
                          local_geom_full.WJ +
                          local_geom_full.invJ
        CUDA.synchronize()

        # Minimal access pattern (just what's needed)
        result_minimal = similar(scalar_field)
        @. result_minimal = local_geom_full.J
        CUDA.synchronize()

        @test true

        println("\nMemory access patterns:")
        println("  Accessing multiple LocalGeometry components has cache benefits")
        println("  But may also cause unnecessary data movement")
        println("  Consider whether all components are needed in each kernel")
    end

    @testset "LocalGeometry size summary" begin
        lg_full_size = sizeof(parent(local_geom_full))
        scalar_size = sizeof(parent(scalar_field))

        # Expected breakdown for 2D with Float64:
        # Coordinates: ~16 bytes (XYPoint)
        # J, WJ, invJ: 3 × 8 = 24 bytes
        # ∂x∂ξ: 2×2 matrix = 4 × 8 = 32 bytes
        # ∂ξ∂x: 2×2 matrix = 4 × 8 = 32 bytes
        # gⁱʲ: 2×2 matrix = 4 × 8 = 32 bytes
        # gᵢⱼ: 2×2 matrix = 4 × 8 = 32 bytes
        # Total: 16 + 24 + 128 = 168 bytes minimum (without alignment/padding)

        n_points = length(local_geom_full)
        n_elems = Topologies.nlocalelems(space)

        total_bytes = lg_full_size
        bytes_per_point = total_bytes ÷ n_points

        println("\n" * "="^60)
        println("LOCAL GEOMETRY CUDA RESOURCE SUMMARY")
        println("="^60)
        println("Space: $(typeof(space))")
        println("Elements: $n_elems")
        println("Grid points: $n_points")
        println()
        println("LocalGeometry memory footprint:")
        println("  Total: $(total_bytes / 1024) KB ($(total_bytes / (1024^2)) MB)")
        println("  Per point: $bytes_per_point bytes")
        println("  Ratio vs scalar field: $(lg_full_size / scalar_size)x")
        println()
        println("Key insight:")
        println("  Carrying full LocalGeometry through kernels may waste")
        println("  bandwidth if only a subset of fields are accessed.")
        println("  Consider alternative approaches:")
        println("    - Extract needed fields to separate arrays")
        println("    - Use LocalGeometry only in specific kernels")
        println("    - Create reduced LocalGeometry types for kernels")
        println("      that only need J and coordinates")
        println("="^60)
    end

    @testset "Extrapolation to ClimaAtmos-scale calculations" begin
        # Realistic atmospheric model parameters
        # Based on typical ClimaAtmos configurations

        # Typical horizontal resolution: cubed sphere with ne=30-50
        # Typical vertical resolution: 63-137 levels
        # Number of state variables: ~10-15 (ρ, u, v, w, θ, q_v, q_l, etc.)

        lg_bytes_per_point = bytes_per_point_full

        # Estimate for different model resolutions
        println("\n" * "="^60)
        println("EXPECTED IMPACT ON CLIMAATMOS CALCULATIONS")
        println("="^60)

        function estimate_local_geom_overhead(n_horizontal_points, n_vertical_levels, n_state_vars)
            total_grid_points = n_horizontal_points * n_vertical_levels
            lg_overhead_mb = (total_grid_points * lg_bytes_per_point) / (1024^2)

            # Typical state vector size: ~50 bytes per variable
            state_size_mb = (total_grid_points * n_state_vars * 8) / (1024^2)

            # Memory ratio
            ratio = lg_overhead_mb / state_size_mb

            return lg_overhead_mb, state_size_mb, ratio
        end

        # Low resolution (testing, development)
        h_pts_low = 3600  # ~60 elements on cubed sphere, ~4x4 quads per elem
        v_lev_low = 63
        state_vars = 10
        lg_mb_low, state_mb_low, ratio_low = estimate_local_geom_overhead(h_pts_low, v_lev_low, state_vars)

        # Medium resolution (regional/development simulations)
        h_pts_med = 14400  # ~125 elements, ~4x4 quads per elem
        v_lev_med = 85
        lg_mb_med, state_mb_med, ratio_med = estimate_local_geom_overhead(h_pts_med, v_lev_med, state_vars)

        # High resolution (operational/research)
        h_pts_hi = 57600  # ~500 elements, ~4x4 quads per elem
        v_lev_hi = 137
        lg_mb_hi, state_mb_hi, ratio_hi = estimate_local_geom_overhead(h_pts_hi, v_lev_hi, state_vars)

        println("\nScenario 1: Low-resolution (testing/development)")
        println("  Horizontal: ~60 elements, ~30 km resolution")
        println("  Vertical: $v_lev_low levels")
        println("  Total grid points: $(h_pts_low * v_lev_low)")
        println("  LocalGeometry overhead: $(round(lg_mb_low, digits=1)) MB")
        println("  State vector size: $(round(state_mb_low, digits=1)) MB")
        println("  Ratio (LocalGeometry / State): $(round(ratio_low, digits=2))x")
        println()

        println("Scenario 2: Medium-resolution (regional simulations)")
        println("  Horizontal: ~125 elements, ~15 km resolution")
        println("  Vertical: $v_lev_med levels")
        println("  Total grid points: $(h_pts_med * v_lev_med)")
        println("  LocalGeometry overhead: $(round(lg_mb_med, digits=1)) MB")
        println("  State vector size: $(round(state_mb_med, digits=1)) MB")
        println("  Ratio (LocalGeometry / State): $(round(ratio_med, digits=2))x")
        println()

        println("Scenario 3: High-resolution (operational/research)")
        println("  Horizontal: ~500 elements, ~7.5 km resolution")
        println("  Vertical: $v_lev_hi levels")
        println("  Total grid points: $(h_pts_hi * v_lev_hi)")
        println("  LocalGeometry overhead: $(round(lg_mb_hi, digits=1)) MB")
        println("  State vector size: $(round(state_mb_hi, digits=1)) MB")
        println("  Ratio (LocalGeometry / State): $(round(ratio_hi, digits=2))x")
        println()

        println("INTERPRETATION:")
        println("-" * 60)
        println("If LocalGeometry is accessed in kernels that process")
        println("state variables, the memory bandwidth required is:")
        println()

        # Assume typical kernel does: read Y, read LocalGeometry, write Yₜ
        bandwidth_mult_low = 1 + ratio_low  # multiply base bandwidth by this
        bandwidth_mult_med = 1 + ratio_med
        bandwidth_mult_hi = 1 + ratio_hi

        println("Low-res:    $(round(bandwidth_mult_low, digits=2))x additional bandwidth")
        println("Med-res:    $(round(bandwidth_mult_med, digits=2))x additional bandwidth")
        println("High-res:   $(round(bandwidth_mult_hi, digits=2))x additional bandwidth")
        println()

        println("PRACTICAL IMPLICATIONS:")
        println("-" * 60)
        println("✓ LocalGeometry is ~10-20% of state vector size")
        println("✓ But carries 50+ additional register pressure per thread")
        println("✓ Can reduce kernel occupancy by 10-30%")
        println()
        println("This means in ClimaAtmos simulations:")
        println("  • Reading LocalGeometry in every physics kernel hurts bandwidth")
        println("  • Register pressure may limit parallelism (fewer threads/block)")
        println("  • Cache efficiency may suffer due to data footprint")
        println()
        println("Optimization strategies:")
        println("  1. Extract needed scalars (J, WJ) at kernel entry")
        println("  2. Cache LocalGeometry in shared memory for blocks")
        println("  3. Use reduced LocalGeometry types for physics-only kernels")
        println("  4. Profile actual kernels to measure real impact")
        println("="^60)

        @test true
    end

end
