include("matrix_field_test_utils.jl")

# Replace all entries in matrix_field that are outside the matrix with NaNs.
function nan_outside_entries!(matrix_field)
    @assert !any(isnan, parent(matrix_field)) # Check that there are no NaNs.
    nan_mask_field = copy(matrix_field)
    for nan_mask_array_view in MatrixFields.field2arrays_view(nan_mask_field)
        nan_mask_array_view .*= NaN
    end
    flip_nan_mask(nan_mask_entry, matrix_entry) =
        isnan(nan_mask_entry) ? matrix_entry : NaN
    @. matrix_field = map(flip_nan_mask, nan_mask_field, matrix_field)
    @assert any(isnan, parent(matrix_field)) # Check that there are now NaNs.
    return matrix_field
end

@testset "Matrix Multiplication At Boundaries" begin
    FT = Float64
    center_space, face_space = test_spaces(FT)

    seed!(1) # ensures reproducibility
    ᶜᶜmatrix_with_outside_entries =
        random_field(TridiagonalMatrixRow{FT}, center_space)
    ᶜᶠmatrix_with_outside_entries =
        random_field(QuaddiagonalMatrixRow{FT}, center_space)
    ᶠᶠmatrix_with_outside_entries =
        random_field(TridiagonalMatrixRow{FT}, face_space)
    ᶠᶜmatrix_with_outside_entries =
        random_field(QuaddiagonalMatrixRow{FT}, face_space)
    ᶜᶜmatrix_without_outside_entries =
        random_field(DiagonalMatrixRow{FT}, center_space)
    ᶜᶠmatrix_without_outside_entries =
        random_field(BidiagonalMatrixRow{FT}, center_space)
    ᶠᶠmatrix_without_outside_entries =
        random_field(DiagonalMatrixRow{FT}, face_space)
    # We can't have ᶠᶜmatrix_without_outside_entries because a CenterToFace
    # matrix field will always store entries that are outside the matrix.

    nan_outside_entries!(ᶜᶜmatrix_with_outside_entries)
    nan_outside_entries!(ᶜᶠmatrix_with_outside_entries)
    nan_outside_entries!(ᶠᶠmatrix_with_outside_entries)
    nan_outside_entries!(ᶠᶜmatrix_with_outside_entries)

    # Test all possible products of matrix fields that include outside entries.
    # Ensure that matrix multiplication never makes use of the outside entries.
    for (matrix_field1, matrix_field2) in (
        (ᶜᶜmatrix_with_outside_entries, ᶜᶜmatrix_with_outside_entries),
        (ᶜᶜmatrix_with_outside_entries, ᶜᶠmatrix_with_outside_entries),
        (ᶜᶠmatrix_with_outside_entries, ᶠᶠmatrix_with_outside_entries),
        (ᶜᶠmatrix_with_outside_entries, ᶠᶜmatrix_with_outside_entries),
        (ᶠᶠmatrix_with_outside_entries, ᶠᶠmatrix_with_outside_entries),
        (ᶠᶠmatrix_with_outside_entries, ᶠᶜmatrix_with_outside_entries),
        (ᶠᶜmatrix_with_outside_entries, ᶜᶜmatrix_with_outside_entries),
        (ᶠᶜmatrix_with_outside_entries, ᶜᶠmatrix_with_outside_entries),
        (ᶜᶜmatrix_with_outside_entries, ᶜᶜmatrix_without_outside_entries),
        (ᶜᶜmatrix_with_outside_entries, ᶜᶠmatrix_without_outside_entries),
        (ᶜᶠmatrix_with_outside_entries, ᶠᶠmatrix_without_outside_entries),
        (ᶠᶠmatrix_with_outside_entries, ᶠᶠmatrix_without_outside_entries),
        (ᶠᶜmatrix_with_outside_entries, ᶜᶜmatrix_without_outside_entries),
        (ᶠᶜmatrix_with_outside_entries, ᶜᶠmatrix_without_outside_entries),
        (ᶜᶜmatrix_without_outside_entries, ᶜᶜmatrix_with_outside_entries),
        (ᶜᶜmatrix_without_outside_entries, ᶜᶠmatrix_with_outside_entries),
        (ᶜᶠmatrix_without_outside_entries, ᶠᶠmatrix_with_outside_entries),
        (ᶜᶠmatrix_without_outside_entries, ᶠᶜmatrix_with_outside_entries),
        (ᶠᶠmatrix_without_outside_entries, ᶠᶠmatrix_with_outside_entries),
        (ᶠᶠmatrix_without_outside_entries, ᶠᶜmatrix_with_outside_entries),
    )
        @test !any(isnan, parent(@. matrix_field1 * matrix_field2))
    end
end
