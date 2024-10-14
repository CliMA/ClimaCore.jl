import BandedMatrices: band
import LinearAlgebra: I, mul!

include(joinpath("..", "matrix_field_test_utils.jl"))

if !(@isdefined(unit_test_field_broadcast_vs_array_reference))
    const FT = Float64
    const center_space, face_space = test_spaces(FT)

    seed!(1) # ensures reproducibility
    const ᶜvec = random_field(FT, center_space)
    const ᶠvec = random_field(FT, face_space)
    const ᶜᶜmat = random_field(DiagonalMatrixRow{FT}, center_space)
    const ᶜᶠmat = random_field(BidiagonalMatrixRow{FT}, center_space)
    const ᶠᶠmat = random_field(TridiagonalMatrixRow{FT}, face_space)
    const ᶠᶜmat = random_field(QuaddiagonalMatrixRow{FT}, face_space)
end

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
