using Test
using JET
import CUDA
import BandedMatrices: band
import LinearAlgebra: I, mul!
import Random: seed!

using ClimaCore.MatrixFields
import ClimaCore:
    Geometry, Domains, Meshes, Topologies, Hypsography, Spaces, Fields
import ClimaComms

# Using @benchmark from BenchmarkTools is extremely slow; it appears to keep
# triggering recompilations and allocating a lot of memory in the process.
# This macro returns the minimum time (in seconds) required to run the
# expression after it has been compiled.
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

# This function is used for benchmarking ref_set_result!.
function call_array_func(
    ref_set_result!::F,
    ref_result_arrays,
    inputs_arrays,
    temp_values_arrays,
) where {F}
    for arrays in
        zip(ref_result_arrays, inputs_arrays..., temp_values_arrays...)
        ref_set_result!(arrays...)
    end
end

function test_matrix_broadcast_against_array_reference(;
    test_name,
    inputs,
    get_result::F1,
    set_result!::F2,
    get_temp_values::F3 = (_...) -> (),
    ref_set_result!::F4,
    time_ratio_limit = 1,
    max_eps_error_limit = 7,
    test_broken_with_cuda = false,
) where {F1, F2, F3, F4}
    @testset "$test_name" begin
        is_using_cuda = ClimaComms.device(inputs[1]) isa ClimaComms.CUDADevice
        ignore_cuda = (AnyFrameModule(CUDA),)

        if test_broken_with_cuda && is_using_cuda
            @test_throws CUDA.InvalidIRError get_result(inputs...)
            @warn "$test_name:\n\tCUDA.InvalidIRError"
            return
        end

        result = get_result(inputs...)
        temp_values = get_temp_values(inputs...)

        # Fill all output fields with NaNs for testing correctness.
        result .*= NaN
        for temp_value in temp_values
            temp_value .*= NaN
        end

        ref_result_arrays = MatrixFields.field2arrays(result)
        inputs_arrays = map(MatrixFields.field2arrays, inputs)
        temp_values_arrays = map(MatrixFields.field2arrays, temp_values)

        best_time = @benchmark set_result!(result, inputs...)
        best_ref_time = @benchmark call_array_func(
            ref_set_result!,
            ref_result_arrays,
            inputs_arrays,
            temp_values_arrays,
        )

        # Compute the maximum error as an integer multiple of machine epsilon.
        result_arrays = MatrixFields.field2arrays(result)
        max_error =
            maximum(zip(result_arrays, ref_result_arrays)) do (array, ref_array)
                maximum(abs.(array .- ref_array))
            end
        max_eps_error = ceil(Int, max_error / eps(typeof(max_error)))

        @info "$test_name:\n\tBest Time = $best_time s\n\tBest Reference Time \
               = $best_ref_time s\n\tMaximum Error = $max_eps_error eps"

        # Test that set_result! is performant compared to ref_set_result!.
        @test best_time / best_ref_time < time_ratio_limit

        # Test set_result! for correctness, allocations, and type instabilities.
        # Ignore the type instabilities in CUDA and the allocations they incur.
        @test max_eps_error < max_eps_error_limit
        @test is_using_cuda || (@allocated set_result!(result, inputs...)) == 0
        @test_opt ignored_modules = ignore_cuda set_result!(result, inputs...)

        # Test ref_set_result! for allocations and type instabilities. This is
        # helpful for ensuring that the performance comparison is fair.
        @test (@allocated call_array_func(
            ref_set_result!,
            ref_result_arrays,
            inputs_arrays,
            temp_values_arrays,
        )) == 0
        @test_opt call_array_func(
            ref_set_result!,
            ref_result_arrays,
            inputs_arrays,
            temp_values_arrays,
        )

        # Test get_result (the allocating version of set_result!) for type
        # instabilities. Ignore the type instabilities in CUDA.
        @test_opt ignored_modules = ignore_cuda get_result(inputs...)
    end
end

function test_matrix_broadcast_against_reference(;
    test_name,
    inputs,
    get_result::F1,
    set_result!::F2,
    ref_inputs,
    ref_set_result!::F3,
) where {F1, F2, F3}
    @testset "$test_name" begin
        is_using_cuda = ClimaComms.device(inputs[1]) isa ClimaComms.CUDADevice
        ignore_cuda = (AnyFrameModule(CUDA),)

        result = get_result(inputs...)

        # Fill the output field with NaNs for testing correctness.
        result .*= NaN

        ref_result = copy(result)

        best_time = @benchmark set_result!(result, inputs...)
        best_ref_time = @benchmark ref_set_result!(ref_result, ref_inputs...)

        @info "$test_name:\n\tBest Time = $best_time s\n\tBest Reference Time \
               = $best_ref_time s"

        # Test that set_result! is performant compared to ref_set_result!.
        @test best_time < best_ref_time

        # Test set_result! for correctness, allocations, and type instabilities.
        # Ignore the type instabilities in CUDA and the allocations they incur.
        # Account for the roundoff errors that CUDA introduces.
        @test if is_using_cuda
            max_error = maximum(abs.(parent(result) .- parent(ref_result)))
            max_error < eps(typeof(max_error))
        else
            result == ref_result
        end
        @test is_using_cuda || (@allocated set_result!(result, inputs...)) == 0
        @test_opt ignored_modules = ignore_cuda set_result!(result, inputs...)

        # Test ref_set_result! for allocations and type instabilities. This is
        # helpful for ensuring that the performance comparison is fair. Ignore
        # the type instabilities in CUDA and the allocations they incur.
        @test is_using_cuda ||
              (@allocated ref_set_result!(ref_result, ref_inputs...)) == 0
        @test_opt ignored_modules = ignore_cuda ref_set_result!(
            ref_result,
            ref_inputs...,
        )

        # Test get_result (the allocating version of set_result!) for type
        # instabilities. Ignore the type instabilities in CUDA.
        @test_opt ignored_modules = ignore_cuda get_result(inputs...)
    end
end

function random_test_fields(::Type{FT}) where {FT}
    velem = 20 # This should be big enough to test high-bandwidth matrices.
    helem = npoly = 1 # These should be small enough for the tests to be fast.

    comms_ctx = ClimaComms.SingletonCommsContext()
    hdomain = Domains.SphereDomain(FT(10))
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(comms_ctx, hmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(10));
        boundary_tags = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vtopology = Topologies.IntervalTopology(comms_ctx, vmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)
    sfc_coord = Fields.coordinate_field(hspace)
    hypsography =
        comms_ctx.device isa ClimaComms.CUDADevice ? Hypsography.Flat() :
        Hypsography.LinearAdaption(
            @. cosd(sfc_coord.lat) + cosd(sfc_coord.long) + 1
        ) # TODO: FD operators don't currently work with hypsography on GPUs.
    center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace, hypsography)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    ᶜcoord = Fields.coordinate_field(center_space)
    ᶠcoord = Fields.coordinate_field(face_space)

    seed!(1) # ensures reproducibility
    ᶜᶜmat = map(c -> DiagonalMatrixRow(ntuple(_ -> rand(FT), 1)...), ᶜcoord)
    ᶜᶠmat = map(c -> BidiagonalMatrixRow(ntuple(_ -> rand(FT), 2)...), ᶜcoord)
    ᶠᶠmat = map(c -> TridiagonalMatrixRow(ntuple(_ -> rand(FT), 3)...), ᶠcoord)
    ᶠᶜmat = map(c -> QuaddiagonalMatrixRow(ntuple(_ -> rand(FT), 4)...), ᶠcoord)
    ᶜvec = map(c -> rand(FT), ᶜcoord)
    ᶠvec = map(c -> rand(FT), ᶠcoord)

    return ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec
end

@testset "Scalar Matrix Field Broadcasting" begin
    ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec = random_test_fields(Float64)

    test_matrix_broadcast_against_array_reference(;
        test_name = "diagonal matrix times vector",
        inputs = (ᶜᶜmat, ᶜvec),
        get_result = (ᶜᶜmat, ᶜvec) -> (@. ᶜᶜmat ⋅ ᶜvec),
        set_result! = (result, ᶜᶜmat, ᶜvec) -> (@. result = ᶜᶜmat ⋅ ᶜvec),
        ref_set_result! = (_result, _ᶜᶜmat, _ᶜvec) ->
            mul!(_result, _ᶜᶜmat, _ᶜvec),
        time_ratio_limit = 2, # This case's ref function is fast on Buildkite.
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "tri-diagonal matrix times vector",
        inputs = (ᶠᶠmat, ᶠvec),
        get_result = (ᶠᶠmat, ᶠvec) -> (@. ᶠᶠmat ⋅ ᶠvec),
        set_result! = (result, ᶠᶠmat, ᶠvec) -> (@. result = ᶠᶠmat ⋅ ᶠvec),
        ref_set_result! = (_result, _ᶠᶠmat, _ᶠvec) ->
            mul!(_result, _ᶠᶠmat, _ᶠvec),
        time_ratio_limit = 4, # This case's ref function is fast on Buildkite.
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "quad-diagonal matrix times vector",
        inputs = (ᶠᶜmat, ᶜvec),
        get_result = (ᶠᶜmat, ᶜvec) -> (@. ᶠᶜmat ⋅ ᶜvec),
        set_result! = (result, ᶠᶜmat, ᶜvec) -> (@. result = ᶠᶜmat ⋅ ᶜvec),
        ref_set_result! = (_result, _ᶠᶜmat, _ᶜvec) ->
            mul!(_result, _ᶠᶜmat, _ᶜvec),
        time_ratio_limit = 5, # This case's ref function is fast on Buildkite.
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix",
        inputs = (ᶜᶜmat, ᶜᶠmat),
        get_result = (ᶜᶜmat, ᶜᶠmat) -> (@. ᶜᶜmat ⋅ ᶜᶠmat),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat) -> (@. result = ᶜᶜmat ⋅ ᶜᶠmat),
        ref_set_result! = (_result, _ᶜᶜmat, _ᶜᶠmat) ->
            mul!(_result, _ᶜᶜmat, _ᶜᶠmat),
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "tri-diagonal matrix times tri-diagonal matrix",
        inputs = (ᶠᶠmat,),
        get_result = (ᶠᶠmat,) -> (@. ᶠᶠmat ⋅ ᶠᶠmat),
        set_result! = (result, ᶠᶠmat) -> (@. result = ᶠᶠmat ⋅ ᶠᶠmat),
        ref_set_result! = (_result, _ᶠᶠmat) -> mul!(_result, _ᶠᶠmat, _ᶠᶠmat),
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "quad-diagonal matrix times diagonal matrix",
        inputs = (ᶠᶜmat, ᶜᶜmat),
        get_result = (ᶠᶜmat, ᶜᶜmat) -> (@. ᶠᶜmat ⋅ ᶜᶜmat),
        set_result! = (result, ᶠᶜmat, ᶜᶜmat) -> (@. result = ᶠᶜmat ⋅ ᶜᶜmat),
        ref_set_result! = (_result, _ᶠᶜmat, _ᶜᶜmat) ->
            mul!(_result, _ᶠᶜmat, _ᶜᶜmat),
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. result = ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            ((@. ᶜᶜmat ⋅ ᶜᶠmat), (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat)),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
        ) -> begin
            mul!(_temp1, _ᶜᶜmat, _ᶜᶠmat)
            mul!(_temp2, _temp1, _ᶠᶠmat)
            mul!(_result, _temp2, _ᶠᶜmat)
        end,
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix, but with \
                     forced right-associativity",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. result = ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            ((@. ᶠᶠmat ⋅ ᶠᶜmat), (@. ᶜᶠmat ⋅ (ᶠᶠmat ⋅ ᶠᶜmat))),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
        ) -> begin
            mul!(_temp1, _ᶠᶠmat, _ᶠᶜmat)
            mul!(_temp2, _ᶜᶠmat, _temp1)
            mul!(_result, _ᶜᶜmat, _temp2)
        end,
        test_broken_with_cuda = true, # TODO: Fix this.
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix times \
                     vector",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec) ->
            (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec) ->
            (@. result = ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec) -> (
            (@. ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat),
            (@. ᶜᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶠmat ⋅ ᶠᶜmat),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _ᶜvec,
            _temp1,
            _temp2,
            _temp3,
        ) -> begin
            mul!(_temp1, _ᶜᶜmat, _ᶜᶠmat)
            mul!(_temp2, _temp1, _ᶠᶠmat)
            mul!(_temp3, _temp2, _ᶠᶜmat)
            mul!(_result, _temp3, _ᶜvec)
        end,
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "diagonal matrix times bi-diagonal matrix times \
                     tri-diagonal matrix times quad-diagonal matrix times \
                     vector, but with forced right-associativity",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec) ->
            (@. ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)))),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec) ->
            (@. result = ᶜᶜmat ⋅ (ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)))),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec) -> (
            (@. ᶠᶜmat ⋅ ᶜvec),
            (@. ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec)),
            (@. ᶜᶠmat ⋅ (ᶠᶠmat ⋅ (ᶠᶜmat ⋅ ᶜvec))),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _ᶜvec,
            _temp1,
            _temp2,
            _temp3,
        ) -> begin
            mul!(_temp1, _ᶠᶜmat, _ᶜvec)
            mul!(_temp2, _ᶠᶠmat, _temp1)
            mul!(_temp3, _ᶜᶠmat, _temp2)
            mul!(_result, _ᶜᶜmat, _temp3)
        end,
        time_ratio_limit = 15, # This case's ref function is fast on Buildkite.
        test_broken_with_cuda = true, # TODO: Fix this.
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "linear combination of matrix products and LinearAlgebra.I",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. result = 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (
            (@. 2 * ᶠᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat ⋅ ᶠᶠmat),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_result, 4I) # We can't directly use I in array broadcasts.
            @. _result = _temp3 + _temp4 / 3 - _result
        end,
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "another linear combination of matrix products and \
                     LinearAlgebra.I",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (@. result =
            ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (
            (@. ᶠᶜmat ⋅ ᶜᶜmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat / 3),
            (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
        ) -> begin
            mul!(_temp1, _ᶠᶜmat, _ᶜᶜmat)
            mul!(_temp2, _temp1, _ᶜᶠmat)
            @. _temp3 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
            mul!(_temp4, _temp3, _ᶠᶠmat)
            copyto!(_result, 4I) # We can't directly use I in array broadcasts.
            @. _result = _temp2 * 2 - _temp4 + _result
        end,
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "matrix times linear combination",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (@. ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,))),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (@. result =
            ᶜᶠmat ⋅ (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,))),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (
            (@. 2 * ᶠᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat ⋅ ᶠᶠmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
            _temp5,
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_temp5, 4I) # We can't directly use I in array broadcasts.
            @. _temp5 = _temp3 + _temp4 / 3 - _temp5
            mul!(_result, _ᶜᶠmat, _temp5)
        end,
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "linear combination times another linear combination",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) ->
            (@. (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
                (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (@. result =
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (
            (@. 2 * ᶠᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat ⋅ ᶠᶠmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
            (@. ᶠᶜmat ⋅ ᶜᶜmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat / 3),
            (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
            _temp5,
            _temp6,
            _temp7,
            _temp8,
            _temp9,
            _temp10,
        ) -> begin
            @. _temp1 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp2, _temp1, _ᶜᶜmat)
            mul!(_temp3, _temp2, _ᶜᶠmat)
            mul!(_temp4, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_temp5, 4I) # We can't directly use I in array broadcasts.
            @. _temp5 = _temp3 + _temp4 / 3 - _temp5
            mul!(_temp6, _ᶠᶜmat, _ᶜᶜmat)
            mul!(_temp7, _temp6, _ᶜᶠmat)
            @. _temp8 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
            mul!(_temp9, _temp8, _ᶠᶠmat)
            copyto!(_temp10, 4I) # We can't directly use I in array broadcasts.
            @. _temp10 = _temp7 * 2 - _temp9 + _temp10
            mul!(_result, _temp5, _temp10)
        end,
        max_eps_error_limit = 30, # This case's roundoff error is large on GPUs.
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "matrix times matrix times linear combination times matrix \
                     times another linear combination times matrix",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)) ⋅
            ᶠᶠmat),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (@. result =
            ᶠᶜmat ⋅ ᶜᶠmat ⋅
            (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
            ᶠᶠmat ⋅
            (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)) ⋅
            ᶠᶠmat),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat) -> (
            (@. ᶠᶜmat ⋅ ᶜᶠmat),
            (@. 2 * ᶠᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat ⋅ ᶠᶠmat),
            (@. 2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)),
            (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
                (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,))),
            (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
                (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
                ᶠᶠmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat),
            (@. ᶠᶠmat / 3),
            (@. (ᶠᶠmat / 3) ⋅ ᶠᶠmat),
            (@. ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,)),
            (@. ᶠᶜmat ⋅ ᶜᶠmat ⋅
                (2 * ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat + ᶠᶠmat ⋅ ᶠᶠmat / 3 - (4I,)) ⋅
                ᶠᶠmat ⋅
                (ᶠᶜmat ⋅ ᶜᶜmat ⋅ ᶜᶠmat * 2 - (ᶠᶠmat / 3) ⋅ ᶠᶠmat + (4I,))),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
            _temp5,
            _temp6,
            _temp7,
            _temp8,
            _temp9,
            _temp10,
            _temp11,
            _temp12,
            _temp13,
            _temp14,
        ) -> begin
            mul!(_temp1, _ᶠᶜmat, _ᶜᶠmat)
            @. _temp2 = 0 + 2 * _ᶠᶜmat # This allocates without the `0 + `.
            mul!(_temp3, _temp2, _ᶜᶜmat)
            mul!(_temp4, _temp3, _ᶜᶠmat)
            mul!(_temp5, _ᶠᶠmat, _ᶠᶠmat)
            copyto!(_temp6, 4I) # We can't directly use I in array broadcasts.
            @. _temp6 = _temp4 + _temp5 / 3 - _temp6
            mul!(_temp7, _temp1, _temp6)
            mul!(_temp8, _temp7, _ᶠᶠmat)
            mul!(_temp9, _ᶠᶜmat, _ᶜᶜmat)
            mul!(_temp10, _temp9, _ᶜᶠmat)
            @. _temp11 = 0 + _ᶠᶠmat / 3 # This allocates without the `0 + `.
            mul!(_temp12, _temp11, _ᶠᶠmat)
            copyto!(_temp13, 4I) # We can't directly use I in array broadcasts.
            @. _temp13 = _temp10 * 2 - _temp12 + _temp13
            mul!(_temp14, _temp8, _temp13)
            mul!(_result, _temp14, _ᶠᶠmat)
        end,
        time_ratio_limit = 2, # This case's ref function is fast on Buildkite.
        max_eps_error_limit = 70, # This case's roundoff error is large on GPUs.
    )

    test_matrix_broadcast_against_array_reference(;
        test_name = "matrix constructions and multiplications",
        inputs = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec),
        get_result = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec) ->
            (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
                TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat ⋅
                DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)),
        set_result! = (result, ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec) ->
            (@. result =
                BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
                TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat ⋅
                DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)),
        get_temp_values = (ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec) -> (
            (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec)),
            (@. TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1)),
            (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
                TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1)),
            (@. BidiagonalMatrixRow(ᶜᶠmat ⋅ ᶠvec, ᶜᶜmat ⋅ ᶜvec) ⋅
                TridiagonalMatrixRow(ᶠvec, ᶠᶜmat ⋅ ᶜvec, 1) ⋅ ᶠᶠmat),
            (@. DiagonalMatrixRow(ᶠvec)),
            (@. DiagonalMatrixRow(DiagonalMatrixRow(ᶠvec) ⋅ ᶠvec)),
        ),
        ref_set_result! = (
            _result,
            _ᶜᶜmat,
            _ᶜᶠmat,
            _ᶠᶠmat,
            _ᶠᶜmat,
            _ᶜvec,
            _ᶠvec,
            _temp1,
            _temp2,
            _temp3,
            _temp4,
            _temp5,
            _temp6,
        ) -> begin
            mul!(view(_temp1, band(0)), _ᶜᶠmat, _ᶠvec)
            mul!(view(_temp1, band(1)), _ᶜᶜmat, _ᶜvec)
            copyto!(view(_temp2, band(-1)), 1, _ᶠvec, 2)
            mul!(view(_temp2, band(0)), _ᶠᶜmat, _ᶜvec)
            fill!(view(_temp2, band(1)), 1)
            mul!(_temp3, _temp1, _temp2)
            mul!(_temp4, _temp3, _ᶠᶠmat)
            copyto!(view(_temp5, band(0)), 1, _ᶠvec, 1)
            mul!(view(_temp6, band(0)), _temp5, _ᶠvec)
            mul!(_result, _temp4, _temp6)
        end,
    )
end

@testset "Non-scalar Matrix Field Broadcasting" begin
    ᶜᶜmat, ᶜᶠmat, ᶠᶠmat, ᶠᶜmat, ᶜvec, ᶠvec = random_test_fields(Float64)

    ᶜlg = Fields.local_geometry_field(ᶜvec)
    ᶠlg = Fields.local_geometry_field(ᶠvec)

    ᶜᶠmat2 = map(row -> map(sin, row), ᶜᶠmat)
    ᶜᶠmat3 = map(row -> map(cos, row), ᶜᶠmat)
    ᶠᶜmat2 = map(row -> map(sin, row), ᶠᶜmat)
    ᶠᶜmat3 = map(row -> map(cos, row), ᶠᶜmat)

    ᶜᶠmat_AC1 = map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶜᶠmat)
    ᶜᶠmat_C12 = map(
        (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
        ᶜᶠmat2,
        ᶜᶠmat3,
    )
    ᶠᶜmat_AC1 = map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶠᶜmat)
    ᶠᶜmat_C12 = map(
        (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
        ᶠᶜmat2,
        ᶠᶜmat3,
    )

    test_matrix_broadcast_against_reference(;
        test_name = "matrix of covectors times matrix of vectors",
        inputs = (ᶜᶠmat_AC1, ᶠᶜmat_C12),
        get_result = (ᶜᶠmat_AC1, ᶠᶜmat_C12) -> (@. ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12),
        set_result! = (result, ᶜᶠmat_AC1, ᶠᶜmat_C12) ->
            (@. result = ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12),
        ref_inputs = (ᶜᶠmat, ᶠᶜmat2, ᶠᶜmat3, ᶠlg),
        ref_set_result! = (result, ᶜᶠmat, ᶠᶜmat2, ᶠᶜmat3, ᶠlg) -> (@. result =
            ᶜᶠmat ⋅ (
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) ⋅ ᶠᶜmat2 +
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) ⋅ ᶠᶜmat3
            )),
    )

    test_matrix_broadcast_against_reference(;
        test_name = "matrix of covectors times matrix of vectors times matrix \
                     of numbers times matrix of covectors times matrix of \
                     vectors",
        inputs = (ᶜᶠmat_AC1, ᶠᶜmat_C12, ᶜᶠmat, ᶠᶜmat_AC1, ᶜᶠmat_C12),
        get_result = (ᶜᶠmat_AC1, ᶠᶜmat_C12, ᶜᶠmat, ᶠᶜmat_AC1, ᶜᶠmat_C12) ->
            (@. ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12 ⋅ ᶜᶠmat ⋅ ᶠᶜmat_AC1 ⋅ ᶜᶠmat_C12),
        set_result! = (
            result,
            ᶜᶠmat_AC1,
            ᶠᶜmat_C12,
            ᶜᶠmat,
            ᶠᶜmat_AC1,
            ᶜᶠmat_C12,
        ) ->
            (@. result = ᶜᶠmat_AC1 ⋅ ᶠᶜmat_C12 ⋅ ᶜᶠmat ⋅ ᶠᶜmat_AC1 ⋅ ᶜᶠmat_C12),
        ref_inputs = (ᶜᶠmat, ᶜᶠmat2, ᶜᶠmat3, ᶠᶜmat, ᶠᶜmat2, ᶠᶜmat3, ᶜlg, ᶠlg),
        ref_set_result! = (
            result,
            ᶜᶠmat,
            ᶜᶠmat2,
            ᶜᶠmat3,
            ᶠᶜmat,
            ᶠᶜmat2,
            ᶠᶜmat3,
            ᶜlg,
            ᶠlg,
        ) -> (@. result =
            ᶜᶠmat ⋅ (
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) ⋅ ᶠᶜmat2 +
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) ⋅ ᶠᶜmat3
            ) ⋅ ᶜᶠmat ⋅ ᶠᶜmat ⋅ (
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:1) ⋅ ᶜᶠmat2 +
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:2) ⋅ ᶜᶠmat3
            )),
    )

    ᶜᶠmat_AC1_num =
        map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat_AC1, ᶜᶠmat)
    ᶜᶠmat_num_C12 =
        map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat, ᶜᶠmat_C12)
    ᶠᶜmat_C12_AC1 =
        map((row1, row2) -> map(tuple, row1, row2), ᶠᶜmat_C12, ᶠᶜmat_AC1)

    test_matrix_broadcast_against_reference(;
        test_name = "matrix of covectors and numbers times matrix of vectors \
                     and covectors times matrix of numbers and vectors times \
                     vector of numbers",
        inputs = (ᶜᶠmat_AC1_num, ᶠᶜmat_C12_AC1, ᶜᶠmat_num_C12, ᶠvec),
        get_result = (ᶜᶠmat_AC1_num, ᶠᶜmat_C12_AC1, ᶜᶠmat_num_C12, ᶠvec) ->
            (@. ᶜᶠmat_AC1_num ⋅ ᶠᶜmat_C12_AC1 ⋅ ᶜᶠmat_num_C12 ⋅ ᶠvec),
        set_result! = (
            result,
            ᶜᶠmat_AC1_num,
            ᶠᶜmat_C12_AC1,
            ᶜᶠmat_num_C12,
            ᶠvec,
        ) -> (@. result = ᶜᶠmat_AC1_num ⋅ ᶠᶜmat_C12_AC1 ⋅ ᶜᶠmat_num_C12 ⋅ ᶠvec),
        ref_inputs = (
            ᶜᶠmat,
            ᶜᶠmat2,
            ᶜᶠmat3,
            ᶠᶜmat,
            ᶠᶜmat2,
            ᶠᶜmat3,
            ᶠvec,
            ᶜlg,
            ᶠlg,
        ),
        ref_set_result! = (
            result,
            ᶜᶠmat,
            ᶜᶠmat2,
            ᶜᶠmat3,
            ᶠᶜmat,
            ᶠᶜmat2,
            ᶠᶜmat3,
            ᶠvec,
            ᶜlg,
            ᶠlg,
        ) -> (@. result = tuple(
            ᶜᶠmat ⋅ (
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:1) ⋅ ᶠᶜmat2 +
                DiagonalMatrixRow(ᶠlg.gⁱʲ.components.data.:2) ⋅ ᶠᶜmat3
            ) ⋅ ᶜᶠmat ⋅ ᶠvec,
            ᶜᶠmat ⋅ ᶠᶜmat ⋅ (
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:1) ⋅ ᶜᶠmat2 +
                DiagonalMatrixRow(ᶜlg.gⁱʲ.components.data.:2) ⋅ ᶜᶠmat3
            ) ⋅ ᶠvec,
        )),
    )

    T(value1, value2, value3) =
        (; a = (), b = value1, c = (value2, (; d = (value3,)), (;)))
    ᶜᶠmat_T = map((rows...) -> map(T, rows...), ᶜᶠmat, ᶜᶠmat2, ᶜᶠmat3)
    ᶠᶜmat_T = map((rows...) -> map(T, rows...), ᶠᶜmat, ᶠᶜmat2, ᶠᶜmat3)
    ᶜvec_T = @. T(ᶜvec, ᶜvec, ᶜvec)

    test_matrix_broadcast_against_reference(;
        test_name = "matrix of nested values times matrix of nested values \
                     times matrix of numbers times matrix of numbers times \
                     times vector of nested values",
        inputs = (ᶜᶠmat_T, ᶠᶜmat, ᶜᶠmat, ᶠᶜmat_T, ᶜvec_T),
        get_result = (ᶜᶠmat_T, ᶠᶜmat, ᶜᶠmat, ᶠᶜmat_T, ᶜvec_T) ->
            (@. ᶜᶠmat_T ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat_T ⋅ ᶜvec_T),
        set_result! = (result, ᶜᶠmat_T, ᶠᶜmat, ᶜᶠmat, ᶠᶜmat_T, ᶜvec_T) ->
            (@. result = ᶜᶠmat_T ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat_T ⋅ ᶜvec_T),
        ref_inputs = (ᶜᶠmat, ᶜᶠmat2, ᶜᶠmat3, ᶠᶜmat, ᶠᶜmat2, ᶠᶜmat3, ᶜvec),
        ref_set_result! = (
            result,
            ᶜᶠmat,
            ᶜᶠmat2,
            ᶜᶠmat3,
            ᶠᶜmat,
            ᶠᶜmat2,
            ᶠᶜmat3,
            ᶜvec,
        ) -> (@. result = T(
            ᶜᶠmat ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat ⋅ ᶜvec,
            ᶜᶠmat2 ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat2 ⋅ ᶜvec,
            ᶜᶠmat3 ⋅ ᶠᶜmat ⋅ ᶜᶠmat ⋅ ᶠᶜmat3 ⋅ ᶜvec,
        )),
    )
end
