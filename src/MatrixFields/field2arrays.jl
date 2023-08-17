# Computes the number of columns in a matrix with the given shape.
band_matrix_n_cols(n_rows, ::Square) = n_rows
band_matrix_n_cols(n_rows, ::FaceToCenter) = n_rows + 1
band_matrix_n_cols(n_rows, ::CenterToFace) = n_rows - 1

# Converts the diagonal index of a matrix field, which can be either an Int or a
# PlusHalf{Int}, to the corresponding diagonal index of a matrix (represented by
# a BandedMatrix), which must be an Int.
band_matrix_d(field_d, ::Square) = field_d
band_matrix_d(field_d, ::FaceToCenter) = field_d + half
band_matrix_d(field_d, ::CenterToFace) = field_d - half

function band_matrix_info(field)
    field_ld, field_ud = outer_diagonals(eltype(field))
    n_rows = Spaces.nlevels(axes(field))
    n_cols = band_matrix_n_cols(n_rows, matrix_shape(field))
    matrix_ld = band_matrix_d(field_ld, matrix_shape(field))
    matrix_ud = band_matrix_d(field_ud, matrix_shape(field))
    matrix_ld <= 0 && matrix_ud >= 0 ||
        error("BandedMatrices.jl does not yet support matrices that have \
               diagonals with indices in the range $matrix_ld:$matrix_ud")
    return n_rows, n_cols, matrix_ld, matrix_ud
end

"""
    column_field2array(field)

Converts a field defined on a `FiniteDifferenceSpace` into either a `Vector` or
a `BandedMatrix`, depending on whether the elements of the field are single
values or `BandMatrixRow`s. This involves copying the data stored in the field.
Because `BandedMatrix` does not currently support operations with `CuArray`s,
all GPU data is copied to the CPU.
"""
function column_field2array(field::Fields.FiniteDifferenceField)
    if eltype(field) <: BandMatrixRow # field represents a matrix
        n_rows, n_cols, matrix_ld, matrix_ud = band_matrix_info(field)
        matrix = BandedMatrix{eltype(eltype(field))}(
            undef,
            (n_rows, n_cols),
            (-matrix_ld, matrix_ud),
        )
        for (index_of_field_entry, matrix_d) in enumerate(matrix_ld:matrix_ud)
            matrix_diagonal = view(matrix, band(matrix_d))
            diagonal_field = field.entries.:($index_of_field_entry)
            diagonal_data =
                vec(reinterpret(eltype(eltype(field)), parent(diagonal_field)'))

            # Find the rows for which diagonal_data[row] is in the matrix.
            # Note: The matrix index (1, 1) corresponds to the diagonal index 0,
            # and the matrix index (n_rows, n_cols) corresponds to the diagonal
            # index n_cols - n_rows.
            first_row = matrix_d < 0 ? 1 - matrix_d : 1
            last_row = matrix_d < n_cols - n_rows ? n_rows : n_cols - matrix_d

            diagonal_data_view = view(diagonal_data, first_row:last_row)
            CUDA.@allowscalar copyto!(matrix_diagonal, diagonal_data_view)
        end
        return matrix
    else # field represents a vector
        return CUDA.@allowscalar Array(column_field2array_view(field))
    end
end

"""
    column_field2array_view(field)

Similar to `column_field2array(field)`, except that this version avoids copying
the data stored in the field.
"""
function column_field2array_view(field::Fields.FiniteDifferenceField)
    if eltype(field) <: BandMatrixRow # field represents a matrix
        _, n_cols, matrix_ld, matrix_ud = band_matrix_info(field)
        field_data_transpose =
            reinterpret(eltype(eltype(field)), parent(field)')
        matrix_transpose =
            _BandedMatrix(field_data_transpose, n_cols, matrix_ud, -matrix_ld)
        return permutedims(matrix_transpose)
        # TODO: Despite not copying any data, this function still allocates a
        # small amount of memory because of _BandedMatrix and permutedims.
    else # field represents a vector
        return vec(reinterpret(eltype(field), parent(field)'))
    end
end

all_columns(::Fields.ColumnField) = (((1, 1), 1),)
all_columns(field) = all_columns(axes(field))
all_columns(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    Spaces.all_nodes(Spaces.horizontal_space(space))

# TODO: Unify FiniteDifferenceField and ColumnField so that we can use this
# version instead.
# all_columns(::Fields.FiniteDifferenceField) = (((1, 1), 1),)
# all_columns(field::Fields.ExtrudedFiniteDifferenceField) =
#     Spaces.all_nodes(Spaces.horizontal_space(axes(field)))

column_map(f::F, field) where {F} =
    Iterators.map(all_columns(field)) do ((i, j), h)
        f(Spaces.column(field, i, j, h))
    end

"""
    field2arrays(field)

Converts a field defined on a `FiniteDifferenceSpace` or on an
`ExtrudedFiniteDifferenceSpace` into a collection of arrays, each of which
corresponds to a column of the field. This is done by calling
`column_field2array` on each of the field's columns.
"""
field2arrays(field) = collect(column_map(column_field2array, field))

"""
    field2arrays_view(field)

Similar to `field2arrays(field)`, except that this version calls
`column_field2array_view` instead of `column_field2array`.
"""
field2arrays_view(field) = column_map(column_field2array_view, field)
