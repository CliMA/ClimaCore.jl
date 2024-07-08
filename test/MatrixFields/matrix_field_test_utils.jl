using Test
using JET
import Random: seed!

import ClimaComms
import BenchmarkTools as BT
ClimaComms.@import_required_backends
import ClimaCore:
    Geometry,
    Domains,
    Meshes,
    Topologies,
    Hypsography,
    Spaces,
    Fields,
    Quadratures
using ClimaCore.MatrixFields

# Test that an expression is true and that it is also type-stable.
macro test_all(expression)
    return quote
        local test_func() = $(esc(expression))
        @test test_func()                   # correctness
        @test (@allocated test_func()) == 0 # allocations
        @test_opt test_func()               # type instabilities
    end
end

# Compute the minimum time (in seconds) required to run an expression after it 
# has been compiled. This macro is used instead of @benchmark from
# BenchmarkTools.jl because the latter is extremely slow (it appears to keep
# triggering recompilations and allocating a lot of memory in the process).
macro benchmark(expression)
    return quote
        $(esc(expression)) # Compile the expression first. Use esc for hygiene.
        best_time = Inf
        start_time = time_ns()
        while time_ns() - start_time < 1e8 # Benchmark for 0.1 s (1e8 ns).
            best_time = min(best_time, @elapsed $(esc(expression)))
        end
        best_time
    end
end

const comms_device = ClimaComms.device()
# comms_device = ClimaComms.CPUSingleThreaded()
@show comms_device
const using_cuda = comms_device isa ClimaComms.CUDADevice
cuda_module(ext) = using_cuda ? ext.CUDA : ext
const cuda_mod = cuda_module(Base.get_extension(ClimaComms, :ClimaCommsCUDAExt))
const cuda_frames = using_cuda ? (AnyFrameModule(cuda_mod),) : ()
const cublas_frames = using_cuda ? (AnyFrameModule(cuda_mod.CUBLAS),) : ()
const invalid_ir_error = using_cuda ? cuda_mod.InvalidIRError : ErrorException

# Test the allocating and non-allocating versions of a field broadcast against
# a reference non-allocating implementation. Ensure that they are performant,
# correct, and type-stable, and print some useful information. If a reference
# implementation is not available, the performance and correctness checks are
# skipped.
function test_field_broadcast(;
    test_name,
    get_result::F1,
    set_result!::F2,
    ref_set_result!::F3 = nothing,
    time_ratio_limit = 10,
    max_eps_error_limit = 10,
    test_broken_with_cuda = false,
) where {F1, F2, F3}
    @testset "$test_name" begin
        if test_broken_with_cuda && using_cuda
            @test_throws invalid_ir_error get_result()
            @warn "$test_name:\n\tCUDA.InvalidIRError"
            return
        end

        result = get_result()
        result_copy = copy(result)
        time = @benchmark set_result!(result)
        time_rounded = round(time; sigdigits = 2)

        # Test that set_result! sets the same value as get_result.
        @test result == result_copy

        if isnothing(ref_set_result!)
            @info "$test_name:\n\tTime = $time_rounded s (reference \
                   implementation unavailable)"
        else
            ref_result = similar(result)
            ref_time = @benchmark ref_set_result!(ref_result)
            ref_time_rounded = round(ref_time; sigdigits = 2)
            time_ratio = time / ref_time
            time_ratio_rounded = round(time_ratio; sigdigits = 2)
            max_error = mapreduce(
                (a, b) -> (abs(a - b)),
                max,
                parent(result),
                parent(ref_result),
            )
            max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))

            @info "$test_name:\n\tTime Ratio = $time_ratio_rounded \
                   ($time_rounded s vs. $ref_time_rounded s for reference) \
                   \n\tMaximum Error = $max_eps_error eps"

            # Test that set_result! is performant and correct when compared
            # against ref_set_result!.
            @test time / ref_time <= time_ratio_limit
            @test max_eps_error <= max_eps_error_limit
        end

        # Test get_result and set_result! for type instabilities, and test
        # set_result! for allocations. Ignore the type instabilities in CUDA and
        # the allocations they incur.
        @test_opt ignored_modules = cuda_frames get_result()
        @test_opt ignored_modules = cuda_frames set_result!(result)
        using_cuda || @test (@allocated set_result!(result)) == 0

        if !isnothing(ref_set_result!)
            # Test ref_set_result! for type instabilities and allocations to
            # ensure that the performance comparison is fair.
            @test_opt ignored_modules = cuda_frames ref_set_result!(ref_result)
            using_cuda || @test (@allocated ref_set_result!(ref_result)) == 0
        end
    end
end

# Test the allocating and non-allocating versions of a field broadcast against
# a reference array-based non-allocating implementation. Ensure that they are
# performant, correct, and type-stable, and print some useful information. In
# order for the input arrays and temporary scratch arrays used by the reference
# implementation to be generated automatically, the corresponding fields must be
# passed to this function.
function test_field_broadcast_against_array_reference(;
    test_name,
    get_result::F1,
    set_result!::F2,
    input_fields,
    get_temp_value_fields = () -> (),
    ref_set_result!::F3,
    time_ratio_limit = 10,
    max_eps_error_limit = 10,
    test_broken_with_cuda = false,
) where {F1, F2, F3}
    @testset "$test_name" begin
        if test_broken_with_cuda && using_cuda
            @test_throws invalid_ir_error get_result()
            @warn "$test_name:\n\tCUDA.InvalidIRError"
            return
        end

        result = get_result()
        result_copy = copy(result)
        time = @benchmark set_result!(result)
        time_rounded = round(time; sigdigits = 2)

        # Test that set_result! sets the same value as get_result.
        @test result == result_copy

        ref_result = similar(result)
        temp_value_fields = map(similar, get_temp_value_fields())

        result_arrays = MatrixFields.field2arrays(result)
        ref_result_arrays = MatrixFields.field2arrays(ref_result)
        inputs_arrays = map(MatrixFields.field2arrays, input_fields)
        temp_values_arrays = map(MatrixFields.field2arrays, temp_value_fields)

        function call_ref_set_result!()
            for arrays in
                zip(ref_result_arrays, inputs_arrays..., temp_values_arrays...)
                ref_set_result!(arrays...)
            end
        end

        ref_time = @benchmark call_ref_set_result!()
        ref_time_rounded = round(ref_time; sigdigits = 2)
        time_ratio = time / ref_time
        time_ratio_rounded = round(time_ratio; sigdigits = 2)
        max_error = compute_max_error(result_arrays, ref_result_arrays)
        max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))

        @info "$test_name:\n\tTime Ratio = $time_ratio_rounded ($time_rounded \
               s vs. $ref_time_rounded s for reference)\n\tMaximum Error = \
               $max_eps_error eps"

        # Test that set_result! is performant and correct when compared against
        # ref_set_result!.
        @test time / ref_time <= time_ratio_limit
        @test max_eps_error <= max_eps_error_limit

        # Test get_result and set_result! for type instabilities, and test
        # set_result! for allocations. Ignore the type instabilities in CUDA and
        # the allocations they incur.
        @test_opt ignored_modules = cuda_frames get_result()
        @test_opt ignored_modules = cuda_frames set_result!(result)
        using_cuda || @test (@allocated set_result!(result)) == 0

        # Test ref_set_result! for type instabilities and allocations to ensure
        # that the performance comparison is fair.
        @test_opt ignored_modules = cuda_frames call_ref_set_result!()
        using_cuda || @test (@allocated call_ref_set_result!()) == 0
    end
end

# Generate extruded finite difference spaces for testing. Include topography
# when possible.
function test_spaces(::Type{FT}) where {FT}
    velem = 20 # This should be big enough to test high-bandwidth matrices.
    helem = npoly = 1 # These should be small enough for the tests to be fast.

    comms_ctx = ClimaComms.SingletonCommsContext(comms_device)
    hdomain = Domains.SphereDomain(FT(10))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(comms_ctx, hmesh)
    quad = Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(10));
        boundary_names = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vtopology = Topologies.IntervalTopology(comms_ctx, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    sfc_coord = Fields.coordinate_field(hspace)
    hypsography =
        using_cuda ? Hypsography.Flat() :
        Hypsography.LinearAdaption(
            Geometry.ZPoint.(@. cosd(sfc_coord.lat) + cosd(sfc_coord.long) + 1),
        ) # TODO: FD operators don't currently work with hypsography on GPUs.
    center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace, hypsography)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

# Generate a random field with elements of type T.
function random_field(::Type{T}, space) where {T}
    FT = Spaces.undertype(space)
    field = Fields.Field(T, space)
    parent(field) .= rand.(FT)
    return field
end

# Construct a highly nested type for testing integration with RecursiveApply.
nested_type(value) = nested_type(value, value, value)
nested_type(value1, value2, value3) =
    (; a = (), b = value1, c = (value2, (; d = (value3,)), (;)))

# A shorthand for typeof(nested_type(::FT)).
const NestedType{FT} = NamedTuple{
    (:a, :b, :c),
    Tuple{
        Tuple{},
        FT,
        Tuple{FT, NamedTuple{(:d,), Tuple{Tuple{FT}}}, NamedTuple{(), Tuple{}}},
    },
}

import Base.Broadcast: materialize, materialize!
import LazyBroadcast: @lazy
import BenchmarkTools as BT

function call_ref_set_result!(
    ref_set_result!::F,
    ref_result_arrays,
    inputs_arrays,
    temp_values_arrays,
) where {F}
    for arrays in
        zip(ref_result_arrays, inputs_arrays..., temp_values_arrays...)
        ref_set_result!(arrays...)
    end
    return nothing
end

function print_time_comparison(; time, ref_time)
    time_rounded = round(time; sigdigits = 2)
    ref_time_rounded = round(ref_time; sigdigits = 2)
    time_ratio = time / ref_time
    time_ratio_rounded = round(time_ratio; sigdigits = 2)
    @info "Times (ClimaCore,Array,ClimaCore/Array): = ($time_rounded, $ref_time_rounded, $time_ratio_rounded)."
    return nothing
end

function compute_max_error(result_arrays, ref_result_arrays)
    return mapreduce(max, result_arrays, ref_result_arrays) do array, ref_array
        mapreduce((a, b) -> (abs(a - b)), max, array, ref_array)
    end
end

set_result!(result, bc) = (materialize!(result, bc); nothing)

#####
##### Scalar test utils
#####

function unit_test_field_broadcast_vs_array_reference(
    result,
    bc;
    input_fields,
    temp_value_fields = (),
    using_cuda,
    ref_set_result! = mul!,
    allowed_max_eps_error = 0,
)
    inputs_arrays = map(MatrixFields.field2arrays, input_fields)
    temp_values_arrays = map(MatrixFields.field2arrays, temp_value_fields)
    result_arrays = MatrixFields.field2arrays(result)
    ref_result_arrays = MatrixFields.field2arrays(similar(result))
    result = materialize(bc)
    result₀ = copy(result)
    set_result!(result, bc)
    @test result == result₀
    call_ref_set_result!(
        ref_set_result!,
        ref_result_arrays,
        inputs_arrays,
        temp_values_arrays,
    )
    max_error = compute_max_error(result_arrays, ref_result_arrays)
    max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))
    @test max_eps_error ≤ allowed_max_eps_error
    return nothing
end

function opt_test_field_broadcast_against_array_reference(
    result,
    bc;
    input_fields,
    temp_value_fields = (),
    ref_set_result!::F = mul!,
    using_cuda,
) where {F}
    temp_values_arrays = map(MatrixFields.field2arrays, temp_value_fields)
    inputs_arrays = map(MatrixFields.field2arrays, input_fields)
    ref_result_arrays = MatrixFields.field2arrays(similar(result))
    ref_time = BT.@belapsed call_ref_set_result!(
        $ref_set_result!,
        $ref_result_arrays,
        $inputs_arrays,
        $temp_values_arrays,
    )
    time = BT.@belapsed set_result!($result, $bc)
    print_time_comparison(; time, ref_time)

    # Test get_result and set_result! for type instabilities, and test
    # set_result! for allocations. Ignore the type instabilities in CUDA and
    # the allocations they incur.
    @test_opt ignored_modules = cuda_frames materialize(bc)
    @test_opt ignored_modules = cuda_frames set_result!(result, bc)
    using_cuda || @test (@allocated set_result!(result, bc)) == 0

    # Test ref_set_result! for type instabilities and allocations to ensure
    # that the performance comparison is fair.
    @test_opt ignored_modules = cuda_frames call_ref_set_result!(
        ref_set_result!,
        ref_result_arrays,
        inputs_arrays,
        temp_values_arrays,
    )
    using_cuda || @test (@allocated call_ref_set_result!(
        ref_set_result!,
        ref_result_arrays,
        inputs_arrays,
        temp_values_arrays,
    )) == 0
    return nothing
end

#####
##### Non-scalar test utils
#####

function unit_test_field_broadcast(
    result,
    bc;
    ref_set_result!,
    allowed_max_eps_error = 10,
)
    result_copy = copy(result)
    set_result!(result, bc)
    # Test that set_result! sets the same value as get_result.
    @test result == result_copy

    ref_result = similar(result)
    ref_set_result!(ref_result)
    max_error = mapreduce(
        (a, b) -> (abs(a - b)),
        max,
        parent(result),
        parent(ref_result),
    )
    max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))

    # Test that set_result! is performant and correct when compared
    # against ref_set_result!.
    @test max_eps_error <= allowed_max_eps_error
    return nothing
end

function opt_test_field_broadcast(result, bc; ref_set_result!)
    time = @benchmark set_result!(result, bc)
    ref_result = similar(result)
    ref_time = @benchmark ref_set_result!(ref_result)
    print_time_comparison(; time, ref_time)

    # Test get_result and set_result! for type instabilities, and test
    # set_result! for allocations. Ignore the type instabilities in CUDA and
    # the allocations they incur.
    @test_opt ignored_modules = cuda_frames materialize(bc)
    @test_opt ignored_modules = cuda_frames set_result!(result, bc)
    using_cuda || @test (@allocated set_result!(result, bc)) == 0

    # Test ref_set_result! for type instabilities and allocations to
    # ensure that the performance comparison is fair.
    @test_opt ignored_modules = cuda_frames ref_set_result!(ref_result)
    using_cuda || @test (@allocated ref_set_result!(ref_result)) == 0
    return nothing
end
