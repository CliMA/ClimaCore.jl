"""
Stress test for the ClimaCore compiler on pointwise/broadcast operations.

This script tests when inlining fails, CUDA performance degrades, or the
compiler fails altogether. The test suite covers:

1. **Arithmetic operations** - varying nesting depth and argument counts
   When do we fail to inline deeply chained operations?

2. **Projection operations** - axis tensor coordinate transformations
   How does projection complexity affect compilation?

3. **Multiple arguments** - varying numbers of nonlocal field arguments
   When does broadcasting with many arguments become problematic?

4. **Function composition** - log, sqrt, and other transcendental functions
   How do special functions interact with inlining?

Each test is run in a subprocess to avoid compilation state leakage.
The Julia project is automatically instantiated before running tests.
Uses the `.buildkite` project environment for reproducibility.

USAGE EXAMPLES:
  # Run all tests on CPU (default)
  julia perf/stress_test_compiler.jl

  # Run tests matching a filter on CPU
  julia perf/stress_test_compiler.jl arithmetic

  # Run on CUDA with GPU reservation
  CLIMACOMMS_DEVICE=CUDA srun --gpus=1 julia perf/stress_test_compiler.jl

  # Run a single specific test
  julia perf/stress_test_compiler.jl arithmetic_depth_5

NOTE: The script automatically detects and uses the `.buildkite` project directory,
so it should be run from the ClimaCore.jl root or from the perf/ directory.
"""

using Pkg
using BenchmarkTools
using Printf
using Dates
using Logging
using Statistics

# Set up logging to suppress info messages during testing
disable_logging(Logging.Info)

# Determine project root and buildkite directory
const PROJECT_ROOT = dirname(dirname(abspath(PROGRAM_FILE)))
const PROJECT_DIR = joinpath(PROJECT_ROOT, ".buildkite")

# Default to CUDA; can still be overridden explicitly by environment
const DEVICE = get(ENV, "CLIMACOMMS_DEVICE", "CUDA")

"""
    initialize_project()

Initialize the Julia project before running tests.
"""
function initialize_project()
    # Change to project directory
    cd(PROJECT_DIR)

    # Instantiate project if needed
    if !isfile(joinpath(PROJECT_DIR, "Manifest.toml"))
        @info "Instantiating project at $PROJECT_DIR"
        Pkg.instantiate()
    else
        @info "Project manifest found, skipping instantiation"
    end
end

"""
    has_cuda_env()

Check if CUDA device is requested via environment variable.
"""
has_cuda_env() = DEVICE == "CUDA"

"""
    run_test_subprocess(test_code::String, test_name::String) -> (success::Bool, output::String, error::String)

Run a test in a subprocess to avoid compilation state leakage.
Returns a tuple of (success, stdout, stderr).
"""
function run_test_subprocess(test_code::String, test_name::String)
    tmp_file = tempname() * ".jl"
    try
        write(tmp_file, test_code)

        # Always use srun with one GPU for subprocess isolation on SLURM systems.
        cmd = `srun --mpi=none --gpus=1 $(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_DIR) $tmp_file`

        try
            # Use withenv to set environment variables
            output = withenv("CLIMACOMMS_DEVICE" => DEVICE) do
                read(cmd, String)
            end
            return (true, output, "")
        catch e
            return (false, "", sprint(showerror, e))
        end
    finally
        rm(tmp_file, force=true)
    end
end

"""
    parse_timings_from_output(output::String) -> Dict{String, Float64}

Parse timing information from test output (expected format: "TIMING: name = value_seconds").
"""
function parse_timings_from_output(output::String)
    timings = Dict{String, Float64}()
    for line in split(output, '\n')
        if startswith(line, "TIMING:")
            parts = split(line, '=')
            if length(parts) == 2
                name = strip(split(parts[1], ':')[2])
                value_str = strip(parts[2])
                if endswith(value_str, "s")
                    value_str = value_str[1:end-1]
                end
                try
                    timings[name] = parse(Float64, value_str)
                catch
                end
            end
        end
    end
    return timings
end

"""
    parse_cuda_profile_from_output(output::String) -> Vector{String}

Parse CUDA profile lines from subprocess output.
Expected format: "CUDA_PROFILE: ...".
"""
function parse_cuda_profile_from_output(output::String)
    profiles = String[]
    for line in split(output, '\n')
        if startswith(line, "CUDA_PROFILE:")
            push!(profiles, strip(line))
        end
    end
    return profiles
end

# ============================================================================
# TEST CASE GENERATION
# ============================================================================

"""
    generate_field_test_code(test_name::String, test_impl::String) -> String

Generate a complete test code block that can be run in a subprocess.
Includes ClimaCore setup and proper device handling.
"""
function generate_field_test_code(test_name::String, test_impl::String)
    device_init = if has_cuda_env()
        """
        using CUDA
        CUDA.allowscalar(false)
        """
    else
        ""
    end

    return """
    import Pkg
    using Printf
    using BenchmarkTools
    import ClimaComms
    ClimaComms.@import_required_backends

    using ClimaCore
    using ClimaCore.Fields
    using ClimaCore.Spaces
    using ClimaCore.Domains
    using ClimaCore.Meshes
    using ClimaCore.Geometry
    using ClimaCore.Topologies
    using ClimaCore.Quadratures

    $device_init

    # Suppress informational logging
    using Logging
    disable_logging(Logging.Info)

    try
        $test_impl
    catch e
        @error "Test failed: \$e"
        rethrow()
    end
    """
end

"""
    create_spectral_space()

Helper function to create a reusable spectral element space setup code string.
"""
function create_spectral_space()
    return """
    FT = Float64
    context = ClimaComms.context()
    if context isa ClimaComms.MPICommsContext
        ClimaComms.init(context)
    end

    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(Geometry.XPoint(-1.0), Geometry.XPoint(1.0); periodic=true),
        Domains.IntervalDomain(Geometry.YPoint(-1.0), Geometry.YPoint(1.0); periodic=true),
    )
    mesh = Meshes.RectilinearMesh(domain, 1, 1)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{4}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    """
end

"""
    arithmetic_test(depth::Int) -> String

Generate test code for arithmetic operations with given nesting depth.
"""
function arithmetic_test(depth::Int)
    # Build expression with `depth` levels of nesting
    expr = "x"
    ops = ["+", "*", "/", "-"]
    for i in 1:depth
        op = ops[mod(i, length(ops)) + 1]
        val = i
        expr = "($expr $op $val.0)"
    end

    kernel_expr = replace(expr, "x" => "v")

    cuda_profile = has_cuda_env() ? """
    if ClimaComms.device(space) isa ClimaComms.CUDADevice
        x_d = CUDA.fill(1.5, 4096)
        y_d = similar(x_d)

        function profile_kernel!(y, x, reps)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i <= length(x)
                @inbounds v = x[i]
                @inbounds for _ in 1:reps
                    v = $kernel_expr
                end
                @inbounds y[i] = v
            end
            return
        end

        reps = $(max(depth, 1))
        k = @cuda launch=false profile_kernel!(y_d, x_d, reps)
        cfg = CUDA.launch_configuration(k.fun)
        threads = min(length(x_d), cfg.threads)
        blocks = cld(length(x_d), threads)
        attrs = CUDA.attributes(k.fun)
        regs_per_thread = attrs[CUDA.CU_FUNC_ATTRIBUTE_NUM_REGS]
        static_smem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
        local_mem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
        max_threads_per_block = attrs[CUDA.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
        warp = CUDA.warpsize(CUDA.device())

        CUDA.@sync @cuda threads=threads blocks=blocks profile_kernel!(y_d, x_d, reps)

        @printf "CUDA_PROFILE: test=arithmetic_depth_$(depth) regs_per_thread=%d static_smem_bytes=%d local_mem_bytes=%d max_threads_per_block=%d launch_threads=%d launch_blocks=%d warp_size=%d\\n" regs_per_thread static_smem_bytes local_mem_bytes max_threads_per_block threads blocks warp
    end
    """ : ""

    test_impl = create_spectral_space() * """

    f = Fields.Field(FT, space)
    fill!(Fields.field_values(f), 1.5)

    op(x) = $expr

    # Compile and benchmark
    _ = op(1.5)
    trial = @benchmark \$op.(\$f) samples=10 evals=1

    $cuda_profile

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: arithmetic_depth_$(depth) = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("arithmetic_depth_$(depth)", test_impl)
end

"""
    multiarg_test(nargs::Int) -> String

Generate test code for operations with multiple arguments (fields).
"""
function multiarg_test(nargs::Int)
    # Build argument list
    args_decl = join(["f$i = Fields.Field(FT, space);\n    fill!(Fields.field_values(f$i), $(Float64(i)))" for i in 1:nargs], "\n    ")
    args_list = join(["f$i" for i in 1:nargs], ", ")

    # Build operation: (f1 + f2 + ...) / (f_last + 1)
    sum_expr = join(["f$i" for i in 1:(nargs-1)], " + ")
    op_expr = "($sum_expr) / (f$nargs + 1.0)"

    cuda_profile = has_cuda_env() ? """
    if ClimaComms.device(space) isa ClimaComms.CUDADevice
        x_d = CUDA.fill(1.5, 4096)
        y_d = similar(x_d)

        function profile_kernel!(y, x)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i <= length(x)
                @inbounds v = x[i]
                @inbounds for j in 1:$(nargs)
                    v = (v + j) / (v + 1.0)
                end
                @inbounds y[i] = v
            end
            return
        end

        k = @cuda launch=false profile_kernel!(y_d, x_d)
        cfg = CUDA.launch_configuration(k.fun)
        threads = min(length(x_d), cfg.threads)
        blocks = cld(length(x_d), threads)
        attrs = CUDA.attributes(k.fun)
        regs_per_thread = attrs[CUDA.CU_FUNC_ATTRIBUTE_NUM_REGS]
        static_smem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
        local_mem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
        max_threads_per_block = attrs[CUDA.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
        warp = CUDA.warpsize(CUDA.device())

        CUDA.@sync @cuda threads=threads blocks=blocks profile_kernel!(y_d, x_d)

        @printf "CUDA_PROFILE: test=multiarg_$(nargs)_args regs_per_thread=%d static_smem_bytes=%d local_mem_bytes=%d max_threads_per_block=%d launch_threads=%d launch_blocks=%d warp_size=%d\\n" regs_per_thread static_smem_bytes local_mem_bytes max_threads_per_block threads blocks warp
    end
    """ : ""

    test_impl = create_spectral_space() * """

    $args_decl

    op($args_list) = $op_expr

    # Compile and benchmark
    _ = op($(join(["$(Float64(i))" for i in 1:nargs], ", ")))
    trial = @benchmark \$op.(\$$args_list) samples=10 evals=1

    $cuda_profile

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: multiarg_$(nargs)_args = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("multiarg_$(nargs)_args", test_impl)
end

"""
    functions_test(funcs::Vector{String}, depth::Int) -> String

Generate test code for composed mathematical functions.
"""
function functions_test(funcs::Vector{String}, depth::Int)
    # Build nested function composition
    if depth == 1
        expr = "$(funcs[1])(x + 0.5)"
    else
        expr = funcs[1] * "("
        for i in 2:depth
            expr *= funcs[i] * "("
        end
        expr *= "x + 0.5"
        for _ in 2:depth
            expr *= ")"
        end
        expr *= ")"
    end

    kernel_expr = replace(expr, "x" => "v")

    cuda_profile = has_cuda_env() ? """
    if ClimaComms.device(space) isa ClimaComms.CUDADevice
        x_d = CUDA.fill(1.5, 4096)
        y_d = similar(x_d)

        function profile_kernel!(y, x)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i <= length(x)
                @inbounds v = x[i]
                @inbounds v = $kernel_expr
                @inbounds y[i] = v
            end
            return
        end

        k = @cuda launch=false profile_kernel!(y_d, x_d)
        cfg = CUDA.launch_configuration(k.fun)
        threads = min(length(x_d), cfg.threads)
        blocks = cld(length(x_d), threads)
        attrs = CUDA.attributes(k.fun)
        regs_per_thread = attrs[CUDA.CU_FUNC_ATTRIBUTE_NUM_REGS]
        static_smem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
        local_mem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
        max_threads_per_block = attrs[CUDA.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
        warp = CUDA.warpsize(CUDA.device())

        CUDA.@sync @cuda threads=threads blocks=blocks profile_kernel!(y_d, x_d)

        @printf "CUDA_PROFILE: test=functions_$(join(funcs, "_"))_depth_$(depth) regs_per_thread=%d static_smem_bytes=%d local_mem_bytes=%d max_threads_per_block=%d launch_threads=%d launch_blocks=%d warp_size=%d\\n" regs_per_thread static_smem_bytes local_mem_bytes max_threads_per_block threads blocks warp
    end
    """ : ""

    test_impl = create_spectral_space() * """

    f = Fields.Field(FT, space)
    fill!(Fields.field_values(f), 1.5)

    op(x) = $expr

    # Compile and benchmark
    _ = op(1.5)
    trial = @benchmark \$op.(\$f) samples=10 evals=1

    $cuda_profile

    time_μs = minimum(trial.times) / 1000.0
    func_desc = join($(repr(funcs)), "_")
    @printf "TIMING: functions_\$(func_desc)_depth_$(depth) = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("functions_$(join(funcs, "_"))_depth_$(depth)", test_impl)
end

"""
    projection_test(complexity::Int) -> String

Generate test code for projection operations on geometric objects.
"""
function projection_test(complexity::Int)
    # Build multiple chained projections
    proj_chain = join(["Geometry.project(Geometry.Covariant12Axis(), v)" for _ in 1:complexity], " + ")

    cuda_profile = has_cuda_env() ? """
    if ClimaComms.device(space) isa ClimaComms.CUDADevice
        x_d = CUDA.fill(1.5, 4096)
        y_d = similar(x_d)

        function profile_kernel!(y, x)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i <= length(x)
                @inbounds v = x[i]
                @inbounds for _ in 1:$(complexity)
                    v = v * 1.1 + 0.5
                end
                @inbounds y[i] = v
            end
            return
        end

        k = @cuda launch=false profile_kernel!(y_d, x_d)
        cfg = CUDA.launch_configuration(k.fun)
        threads = min(length(x_d), cfg.threads)
        blocks = cld(length(x_d), threads)
        attrs = CUDA.attributes(k.fun)
        regs_per_thread = attrs[CUDA.CU_FUNC_ATTRIBUTE_NUM_REGS]
        static_smem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
        local_mem_bytes = attrs[CUDA.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
        max_threads_per_block = attrs[CUDA.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]
        warp = CUDA.warpsize(CUDA.device())

        CUDA.@sync @cuda threads=threads blocks=blocks profile_kernel!(y_d, x_d)

        @printf "CUDA_PROFILE: test=projection_$(complexity)x regs_per_thread=%d static_smem_bytes=%d local_mem_bytes=%d max_threads_per_block=%d launch_threads=%d launch_blocks=%d warp_size=%d\\n" regs_per_thread static_smem_bytes local_mem_bytes max_threads_per_block threads blocks warp
    end
    """ : ""

    test_impl = create_spectral_space() * """

    v_type = typeof(Fields.field_values(Fields.Field(Geometry.Contravariant12Vector{FT}, space))[1])

    v = Fields.Field(Geometry.Contravariant12Vector{FT}, space)
    fill!(Fields.field_values(v), Geometry.Contravariant12Vector(1.0, 2.0))

    op(v) = $proj_chain

    # Compile and benchmark
    _ = op(v)
    trial = @benchmark \$op.(\$v) samples=10 evals=1

    $cuda_profile

    time_μs = minimum(trial.times) / 1000.0
    @printf "TIMING: projection_$(complexity)x = %.6f s\\n" time_μs / 1e6
    """

    return generate_field_test_code("projection_$(complexity)x", test_impl)
end

# ============================================================================
# TEST CATALOG: Define all test cases
# ============================================================================

"""
    struct TestDef

Definition of a single test case for generation and execution.
"""
struct TestDef
    name::String
    description::String
    operation_type::String    # "arithmetic", "projection", "multiarg", "functions"
    complexity::Int           # nesting depth or argument count
    num_args::Int
    uses_geometry::Bool
    code_generator::Function  # Function that generates test code
end

# Create test definitions
const ALL_TESTS = [
    # Arithmetic with varying depth
    [TestDef("arithmetic_depth_$i", "Arithmetic operations with depth $i",
             "arithmetic", i, 1, false,
             () -> arithmetic_test(i)) for i in [1, 3, 5, 7, 10]]

    # Multiple arguments
    [TestDef("multiarg_$(i)_args", "Operations with $i field arguments",
             "multiarg", 1, i, false,
             () -> multiarg_test(i)) for i in [2, 3, 4, 6, 8]]

    # Function compositions
    [TestDef("functions_log_depth_$i", "Log function composed $i times",
             "functions", i, 1, false,
             () -> functions_test(["log"], i)) for i in [1, 2, 3]]

    [TestDef("functions_sqrt_depth_$i", "Sqrt function composed $i times",
             "functions", i, 1, false,
             () -> functions_test(["sqrt"], i)) for i in [1, 2, 3]]

    [TestDef("functions_mixed_depth_$i", "Mixed functions (log, sqrt, abs) depth $i",
             "functions", i, 1, false,
             () -> functions_test(["log", "sqrt", "abs"], i)) for i in [1, 2]]

    # Projection operations
    [TestDef("projection_$(i)x_chained", "Chained projection operations x$i",
             "projection", i, 1, true,
             () -> projection_test(i)) for i in [1, 2, 3, 5]]
] |> vec

# ============================================================================
# EXECUTION AND REPORTING
# ============================================================================

"""
    mutable struct TestResult

Stores the result of a single test execution.
"""
mutable struct TestResult
    test_def::TestDef
    success::Bool
    time_seconds::Union{Float64, Nothing}
    error_msg::String
    cuda_profiles::Vector{String}

    TestResult(test_def) = new(test_def, false, nothing, "", String[])
end

"""
    run_test(test_def::TestDef) -> TestResult

Run a single test case in a subprocess and collect results.
"""
function run_test(test_def::TestDef)
    result = TestResult(test_def)

    # Generate test code
    test_code = test_def.code_generator()

    # Run in subprocess
    success, output, error = run_test_subprocess(test_code, test_def.name)

    if success
        result.cuda_profiles = parse_cuda_profile_from_output(output)
        # Parse timing from output
        timings = parse_timings_from_output(output)
        if haskey(timings, test_def.name)
            result.time_seconds = timings[test_def.name]
            result.success = true
        else
            result.error_msg = "Failed to parse timing from output"
        end
    else
        result.error_msg = error
    end

    return result
end

"""
    print_result(result::TestResult)

Pretty-print a test result.
"""
function print_result(result::TestResult)
    test = result.test_def
    if result.success
        time_μs = result.time_seconds * 1e6
        @printf "  %-45s %10.3f μs" test.name time_μs
        @printf " (depth=%d, args=%d)" test.complexity test.num_args
        if test.uses_geometry
            print(" [geometry]")
        end
        println()
        for line in result.cuda_profiles
            println("    " * line)
        end
    else
        @printf "  %-45s ERROR: %s\n" test.name result.error_msg
    end
end

"""
    main(; test_filter::Union{String, Nothing}=nothing)

Run all tests and produce a report.
"""
function main(; test_filter::Union{String, Nothing}=nothing)
    println("=" ^ 90)
    println("ClimaCore Compiler Stress Test Suite - Pointwise/Broadcast Operations")
    println("Device: $(DEVICE)")
    has_cuda_env() && println("CUDA warnings disabled to catch only actual failures")
    println("=" ^ 90)
    println()

    # Filter tests if requested
    tests = if isnothing(test_filter)
        ALL_TESTS
    else
        filter(t -> contains(t.name, test_filter), ALL_TESTS)
    end

    if isempty(tests)
        println("No tests matching filter: $test_filter")
        println("\nAvailable test categories:")
        for op_type in sort(unique(t.operation_type for t in ALL_TESTS))
            matching = filter(t -> t.operation_type == op_type, ALL_TESTS)
            println("  $op_type: $(length(matching)) tests")
            for test in matching[1:min(3, length(matching))]
                println("    - $(test.name)")
            end
            length(matching) > 3 && println("    ... and $(length(matching) - 3) more")
        end
        return
    end

    println("Running $(length(tests)) test case(s):")
    println()

    results = TestResult[]

    # Run all tests
    for (i, test) in enumerate(tests)
        @printf "[%2d/%2d] %-45s ... " i length(tests) test.name
        flush(stdout)

        result = run_test(test)
        push!(results, result)

        if result.success
            println("✓")
        else
            println("✗")
        end
    end

    println()
    println("=" ^ 90)
    println("Results")
    println("=" ^ 90)
    println()

    # Group by operation type
    by_type = Dict{String, Vector{TestResult}}()
    for result in results
        op_type = result.test_def.operation_type
        if !haskey(by_type, op_type)
            by_type[op_type] = []
        end
        push!(by_type[op_type], result)
    end

    # Print results by type
    for op_type in sort(collect(keys(by_type)))
        type_results = by_type[op_type]
        successful = count(r -> r.success, type_results)

        println("$op_type operations ($successful/$(length(type_results)) successful):")

        for result in sort(type_results, by=r -> r.test_def.complexity)
            print_result(result)
        end

        println()
    end

    # Summary statistics
    successful = filter(r -> r.success, results)
    if !isempty(successful)
        println("=" ^ 90)
        println("Performance Summary")
        println("=" ^ 90)

        times = [r.time_seconds * 1e6 for r in successful]  # convert to microseconds

        println("Execution times (microseconds):")
        @printf "  Minimum:     %.3f μs\n" minimum(times)
        @printf "  Maximum:     %.3f μs\n" maximum(times)
        @printf "  Mean:        %.3f μs\n" mean(times)
        if length(times) >= 2
            @printf "  Median:      %.3f μs\n" median(times)
        end
    end

    println()
    num_successful = length(successful)
    num_failed = length(results) - num_successful

    if num_failed == 0
        println("✓ All $(num_successful) tests passed!")
    else
        println("✗ $(num_failed) test(s) failed out of $(length(results)) total")
    end
end

# ============================================================================
# Entry point
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Initialize project first
    initialize_project()

    # Parse command-line arguments
    if length(ARGS) > 0
        test_filter = ARGS[1]
        main(; test_filter)
    else
        main()
    end
end
