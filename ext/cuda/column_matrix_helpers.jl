# Helpers for multiplying a single row of a banded matrix with a banded matrix or vector
# stored in shared memory. These are specialized for columns with CUDA.blockDim().x face
# levels, computed by threads `v = 1:CUDA.blockDim().x` (one column per block).
#
# `periodic` (a compile-time-constant `Bool` derived from the vertical topology) selects
# how the ends of a column are handled. In a NON-periodic column an operand slot past the
# top/bottom of the column contributes nothing, so it is zero-padded, and product-row
# entries that fall outside the product matrix are truncated to zero. In a PERIODIC column
# the neighbor wraps around to the opposite end, so the operand slot is wrapped with
# `mod1(slot, CUDA.blockDim().x)`, always contributes, and no truncation is applied
# (periodic columns span all `CUDA.blockDim().x` levels, so that is the wrap modulus).
#

# Number of valid slots in the column space of a matrix with the given shape.
@inline n_column_slots(::Union{FaceToCenter, FaceToFace}) = CUDA.blockDim().x
@inline n_column_slots(::Union{CenterToFace, CenterToCenter}) =
    CUDA.blockDim().x - 1i32

# Shape of `matrix1 * matrix2`: its rows match `matrix1`'s rows and its columns match
# `matrix2`'s columns.
@inline product_shape(
    ::Union{FaceToCenter, CenterToCenter},
    ::Union{CenterToFace, CenterToCenter},
) = CenterToCenter()
@inline product_shape(
    ::Union{FaceToCenter, CenterToCenter},
    ::Union{FaceToCenter, FaceToFace},
) = FaceToCenter()
@inline product_shape(
    ::Union{CenterToFace, FaceToFace},
    ::Union{CenterToFace, CenterToCenter},
) = CenterToFace()
@inline product_shape(
    ::Union{CenterToFace, FaceToFace},
    ::Union{FaceToCenter, FaceToFace},
) = FaceToFace()

# row_mul_mat! handles banded matrix * banded matrix. Entries of the product row whose
# column index lies outside the product matrix are set to zero, matching
# `multiply_matrix_at_index` in `src/MatrixFields/matrix_multiplication.jl`.
Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    shape1::MatrixFields.AbstractMatrixShape,
    shape2::MatrixFields.AbstractMatrixShape,
    periodic,
) where {P}
    v = threadIdx().x
    n = CUDA.blockDim().x
    ld1, ud1 = MatrixFields.outer_diagonals(typeof(mat1_row))
    ld2, ud2 = MatrixFields.outer_diagonals(eltype(matrix2))
    pd1, pd2 = MatrixFields.outer_diagonals(P)
    prod_shape = product_shape(shape1, shape2)
    zero_entry = zero(eltype(P))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        prod_slot = band_matrix_d(v + pd, prod_shape)
        if periodic || 0i32 < prod_slot <= n_column_slots(prod_shape)
            UnrolledUtilities.unrolled_mapreduce(+, (ld1:ud1...,)) do mat1_row_d
                mat2_slot = band_matrix_d(v + mat1_row_d, shape1)
                if !(ld2 <= pd - mat1_row_d <= ud2)
                    zero_entry
                elseif periodic
                    @inbounds mat1_row[mat1_row_d] *
                              matrix2[mod1(mat2_slot, n)][pd - mat1_row_d]
                elseif 0i32 < mat2_slot <= n_column_slots(shape1)
                    @inbounds mat1_row[mat1_row_d] *
                              matrix2[mat2_slot][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        else
            zero_entry
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

# row_mul_vec! handles banded matrix * vector.
Base.@propagate_inbounds function row_mul_vec!(
    ::Type{P},
    mat1_row,
    vector2,
    shape1::MatrixFields.AbstractMatrixShape,
    periodic,
) where {P}
    v = threadIdx().x
    n = CUDA.blockDim().x
    ld1, ud1 = MatrixFields.outer_diagonals(typeof(mat1_row))
    zero_entry = zero(P)
    return UnrolledUtilities.unrolled_mapreduce(
        +,
        ld1:ud1;
        init = zero_entry,
    ) do mat1_row_d
        vec2_slot = band_matrix_d(v + mat1_row_d, shape1)
        if periodic
            @inbounds mat1_row[mat1_row_d] *
                      vector2[mod1(vec2_slot, n)]
        elseif 0i32 < vec2_slot <= n_column_slots(shape1)
            @inbounds mat1_row[mat1_row_d] * vector2[vec2_slot]
        else
            zero_entry
        end
    end
end
