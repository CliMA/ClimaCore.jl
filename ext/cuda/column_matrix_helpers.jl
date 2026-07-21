# Helpers for multiplying a single row of a banded matrix with a banded matrix or vector
# stored in shared memory. These are specialized for columns with CUDA.blockDim().x face
# levels, computed by threads `v = 1:CUDA.blockDim().x` (one column per `threadIdx().y`).
#
# Shared memory holds one value per thread: values on cell centers are stored at slot
# `v = center index` (so the last slot is unused), and values on cell faces at slot
# `v = face index + half`.

# `column_slot(v, d, shape)` is the shared-memory slot of the value that entry `d` of the
# matrix row computed by thread `v` multiplies, for a matrix with the given `shape`: the row
# index of thread `v` is `v` for center rows and `v - half` for face rows, the column index
# is `row index + d`, and the slot of a column index is `index + half` for face columns and
# `index` for center columns.
Base.@propagate_inbounds column_slot(v, d, ::Union{CenterToCenter, FaceToFace}) = v + d
Base.@propagate_inbounds column_slot(v, d, ::FaceToCenter) = v + d + half
Base.@propagate_inbounds column_slot(v, d, ::CenterToFace) = v + d - half

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
) where {P}
    v = threadIdx().x
    block_col_idx = threadIdx().y
    ld1, ud1 = MatrixFields.outer_diagonals(typeof(mat1_row))
    ld2, ud2 = MatrixFields.outer_diagonals(eltype(matrix2))
    pd1, pd2 = MatrixFields.outer_diagonals(P)
    prod_shape = product_shape(shape1, shape2)
    mat2_offset = (block_col_idx - 1i32) * CUDA.blockDim().x
    zero_entry = zero(eltype(P))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        prod_slot = column_slot(v, pd, prod_shape)
        if prod_slot < 1i32 || prod_slot > n_column_slots(prod_shape)
            zero_entry # This entry is outside the product matrix.
        else
            UnrolledUtilities.unrolled_mapreduce(+, (ld1:ud1...,)) do mat1_row_d
                mat2_slot = column_slot(v, mat1_row_d, shape1)
                if ld2 <= pd - mat1_row_d <= ud2 &&
                   0i32 < mat2_slot <= n_column_slots(shape1)
                    @inbounds mat1_row[mat1_row_d] *
                              matrix2[mat2_slot + mat2_offset][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
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
) where {P}
    v = threadIdx().x
    block_col_idx = threadIdx().y
    ld1, ud1 = MatrixFields.outer_diagonals(typeof(mat1_row))
    vec2_offset = (block_col_idx - 1i32) * CUDA.blockDim().x
    zero_entry = zero(P)
    return UnrolledUtilities.unrolled_mapreduce(
        +,
        ld1:ud1;
        init = zero_entry,
    ) do mat1_row_d
        vec2_slot = column_slot(v, mat1_row_d, shape1)
        if 0i32 < vec2_slot <= n_column_slots(shape1)
            @inbounds outer_or_mul(
                mat1_row[mat1_row_d],
                vector2[vec2_slot + vec2_offset],
            )
        else
            zero_entry
        end
    end
end

# Handles multiplication in row_mul_vec!.
# Basically rmul, but some operators matrices require special handling
# general case
Base.@propagate_inbounds outer_or_mul(x::T1, y::T2) where {T1, T2} = x * y
# case for grad of a vec
Base.@propagate_inbounds outer_or_mul(x::T1, y::T2) where {T1 <: AbstractVector, T2} = x ⊗ y
# case for divgrad of a vec
Base.@propagate_inbounds outer_or_mul(
    x::T1,
    y::T2,
) where {T1 <: Geometry.AbstractCovector, T2 <: Geometry.AbstractTensor{2}} = (x * y)'
