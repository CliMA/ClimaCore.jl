import BandedMatrices: band
import LinearAlgebra: I, mul!

include(joinpath("..", "matrix_field_test_utils.jl"))

if !(@isdefined(unit_test_field_broadcast))
    const FT = Float64
    const center_space, face_space = test_spaces(FT)

    const ᶜlg = Fields.local_geometry_field(center_space)
    const ᶠlg = Fields.local_geometry_field(face_space)

    seed!(1) # ensures reproducibility
    const ᶜvec = random_field(FT, center_space)
    const ᶠvec = random_field(FT, face_space)
    const ᶜᶠmat = random_field(BidiagonalMatrixRow{FT}, center_space)
    const ᶜᶠmat2 = random_field(BidiagonalMatrixRow{FT}, center_space)
    const ᶜᶠmat3 = random_field(BidiagonalMatrixRow{FT}, center_space)
    const ᶠᶜmat = random_field(QuaddiagonalMatrixRow{FT}, face_space)
    const ᶠᶜmat2 = random_field(QuaddiagonalMatrixRow{FT}, face_space)
    const ᶠᶜmat3 = random_field(QuaddiagonalMatrixRow{FT}, face_space)

    const ᶜᶠmat_AC1 =
        map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶜᶠmat)
    const ᶜᶠmat_C12 = map(
        (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
        ᶜᶠmat2,
        ᶜᶠmat3,
    )
    const ᶠᶜmat_AC1 =
        map(row -> map(adjoint ∘ Geometry.Covariant1Vector, row), ᶠᶜmat)
    const ᶠᶜmat_C12 = map(
        (row1, row2) -> map(Geometry.Covariant12Vector, row1, row2),
        ᶠᶜmat2,
        ᶠᶜmat3,
    )

    const ᶜᶠmat_AC1_num =
        map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat_AC1, ᶜᶠmat)
    const ᶜᶠmat_num_C12 =
        map((row1, row2) -> map(tuple, row1, row2), ᶜᶠmat, ᶜᶠmat_C12)
    const ᶠᶜmat_C12_AC1 =
        map((row1, row2) -> map(tuple, row1, row2), ᶠᶜmat_C12, ᶠᶜmat_AC1)

    const ᶜvec_NT = @. nested_type(ᶜvec, ᶜvec, ᶜvec)
    const ᶜᶠmat_NT =
        map((rows...) -> map(nested_type, rows...), ᶜᶠmat, ᶜᶠmat2, ᶜᶠmat3)
    const ᶠᶜmat_NT =
        map((rows...) -> map(nested_type, rows...), ᶠᶜmat, ᶠᶜmat2, ᶠᶜmat3)
end

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
