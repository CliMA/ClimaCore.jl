# row_mul_mat! handles banded matrix * banded matrix. There are 8 methods, but they all have the
# same structure, so we they could be written as a single method.
# The others can be obtained by copy-pasting and changing the indices appropriately.
# Note that these are all specialized for CUDA.blockDim().x faces , so the indices are hardcoded.
Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::FaceToCenter,
    ::CenterToFace,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
    li = 1i32
    ri = CUDA.blockDim().x - 1i32
    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd < li || v + pd > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 &&
                   (0i32 < v + mat1_row_d + half <= CUDA.blockDim().x)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d + half + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::CenterToFace,
    ::FaceToCenter,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
    li = 1i32
    ri = CUDA.blockDim().x
    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd < li || v + pd > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 &&
                   (0i32 < v + mat1_row_d - half < CUDA.blockDim().x)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d - half + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::CenterToCenter,
    ::CenterToCenter,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
    li = 1i32
    ri = CUDA.blockDim().x - 1i32
    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd < li || v + pd > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 &&
                   (0i32 < v + mat1_row_d <= CUDA.blockDim().x - 1i32)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::FaceToFace,
    ::FaceToFace,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)

    li = 1i32
    ri = CUDA.blockDim().x

    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd < li || v + pd > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 && (0i32 < v + mat1_row_d <= CUDA.blockDim().x)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::FaceToCenter,
    ::FaceToFace,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
    li = 1i32
    ri = CUDA.blockDim().x
    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd + half < li || v + pd + half > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 &&
                   (0i32 < v + mat1_row_d + half <= CUDA.blockDim().x)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d + half + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::CenterToFace,
    ::CenterToCenter,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
    li = 1i32
    ri = CUDA.blockDim().x
    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd + half < li || v + pd + half > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 &&
                   (0i32 < v + mat1_row_d - half < CUDA.blockDim().x)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d - half + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::FaceToFace,
    ::CenterToFace,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
    li = 1i32
    ri = CUDA.blockDim().x
    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd + half < li || v + pd + half > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 && (0i32 < v + mat1_row_d <= CUDA.blockDim().x)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

Base.@propagate_inbounds function row_mul_mat!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::CenterToCenter,
    ::FaceToCenter,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    ld2, ud2 = MatrixFields.outer_diagonals(mat2_eltype)
    pd1, pd2 = MatrixFields.outer_diagonals(prod_eltype)
    li = 1i32
    ri = CUDA.blockDim().x
    zero_entry = rzero(eltype(prod_eltype))
    prod_entries = UnrolledUtilities.unrolled_map((pd1:pd2...,)) do pd
        if v + pd + half < li || v + pd + half > ri
            zero_entry
        else
            UnrolledUtilities.unrolled_mapreduce(⊞, (ld1:ud1...,)) do mat1_row_d
                if ld2 <= pd - mat1_row_d <= ud2 && (0i32 < v + mat1_row_d < CUDA.blockDim().x)
                    @inbounds mat1_row[mat1_row_d] ⊠
                              matrix2[v + mat1_row_d + (i - 1i32) * CUDA.blockDim().x][pd - mat1_row_d]
                else
                    zero_entry
                end
            end
        end
    end
    return BandMatrixRow{pd1}(prod_entries...)
end

# row_mul_vec! handles banded matrix * vector. There are four methods, but they all have the
# same structure, so we they could be written as a single method.
# The others can be obtained by copy-pasting and changing the indices appropriately.
# Note that these are all specialized for CUDA.blockDim().x faces , so the indices are hardcoded.
Base.@propagate_inbounds function row_mul_vec!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::FaceToCenter,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    li = 1i32
    ri = CUDA.blockDim().x - 1i32
    zero_entry = rzero(prod_eltype)
    return UnrolledUtilities.unrolled_mapreduce(
        ⊞,
        ld1:ud1;
        init = zero_entry,
    ) do mat1_row_d
        if (0i32 < v + mat1_row_d + half <= CUDA.blockDim().x)
            @inbounds outer_or_mul(
                mat1_row[mat1_row_d],
                matrix2[v + mat1_row_d + half + (i - 1i32) * CUDA.blockDim().x],
            )
        else
            zero_entry
        end
    end
end

Base.@propagate_inbounds function row_mul_vec!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::CenterToFace,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    li = 1i32
    ri = CUDA.blockDim().x
    zero_entry = rzero(prod_eltype)
    return UnrolledUtilities.unrolled_mapreduce(
        ⊞,
        ld1:ud1;
        init = zero_entry,
    ) do mat1_row_d
        if (0i32 < v + mat1_row_d - half < CUDA.blockDim().x)
            @inbounds outer_or_mul(
                mat1_row[mat1_row_d],
                matrix2[v + mat1_row_d - half + (i - 1i32) * CUDA.blockDim().x],
            )
        else
            zero_entry
        end
    end
end

Base.@propagate_inbounds function row_mul_vec!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::CenterToCenter,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    li = 1i32
    ri = CUDA.blockDim().x - 1i32
    zero_entry = rzero(prod_eltype)
    return UnrolledUtilities.unrolled_mapreduce(
        ⊞,
        ld1:ud1;
        init = zero_entry,
    ) do mat1_row_d
        if (0i32 < v + mat1_row_d <= CUDA.blockDim().x - 1i32)
            @inbounds outer_or_mul(
                mat1_row[mat1_row_d],
                matrix2[v + mat1_row_d + (i - 1i32) * CUDA.blockDim().x],
            )
        else
            zero_entry
        end
    end
end

Base.@propagate_inbounds function row_mul_vec!(
    ::Type{P},
    mat1_row,
    matrix2,
    ::FaceToFace,
) where {P}
    prod_eltype = P
    v = threadIdx().x
    i = threadIdx().y
    mat1_eltype = typeof(mat1_row)
    mat2_eltype = eltype(matrix2)
    ld1, ud1 = MatrixFields.outer_diagonals(mat1_eltype)
    li = 1i32
    ri = CUDA.blockDim().x
    zero_entry = rzero(prod_eltype)
    return UnrolledUtilities.unrolled_mapreduce(
        ⊞,
        ld1:ud1;
        init = zero_entry,
    ) do mat1_row_d
        if (0i32 < v + mat1_row_d <= CUDA.blockDim().x)
            @inbounds outer_or_mul(
                mat1_row[mat1_row_d],
                matrix2[v + mat1_row_d + (i - 1i32) * CUDA.blockDim().x],
            )
        else
            zero_entry
        end
    end
end

# Handles multiplication in row_mul_vec!.
# Basically rmul, but some operators matrices require special handling
# general case
Base.@propagate_inbounds outer_or_mul(x::T1, y::T2) where {T1, T2} = x ⊠ y
# case for grad of a vec
Base.@propagate_inbounds outer_or_mul(x::T1, y::T2) where {T1 <: AbstractVector, T2} = x ⊗ y
Base.@propagate_inbounds outer_or_mul(
    x::T1,
    y::T2,
) where {T1, T2 <: Union{Tuple, NamedTuple}} =
    RecursiveApply.rmap(Base.Fix1(outer_or_mul, x), y)
Base.@propagate_inbounds outer_or_mul(
    x::T1,
    y::T2,
) where {T1 <: Union{Tuple, NamedTuple}, T2 <: Union{Tuple, NamedTuple}} = x ⊠ y
Base.@propagate_inbounds outer_or_mul(
    x::T1,
    y::T2,
) where {T1 <: AbstractVector, T2 <: Union{Tuple, NamedTuple}} =
    RecursiveApply.rmap(Base.Fix1(outer_or_mul, x), y)
# case for divgrad of a vec
Base.@propagate_inbounds outer_or_mul(
    x::T1,
    y::T2,
) where {T1 <: Geometry.AdjointAxisVector, T2 <: Geometry.Axis2Tensor} = (x * y)'
