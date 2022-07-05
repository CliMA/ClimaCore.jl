#=
This file implements operations on banded matrices for ClimaCore, where matrices
are assumed to be stored as `Field`s of `StencilCoefs` objects, and where
vectors are assumed to be stored as regular `Field`s. This format for storing
banded matrices is only a placeholder until we have a more general and
conceptually clearer data structure.

The first comment below defines how mathematical vectors and matrices correspond
to their `Field` representations.

The second comment defines how to find the values of a `Field` that corresponds
to a matrix-vector product. It also defines the "Interior", "Left Boundary", and
"Right Boundary" for an `Operator` that implements such a matrix-vector product.

The third comment does the same for a matrix-matrix product.

The fourth comment proves that the bandwidths of a matrix-matrix product are the
sums of the bandwidths of the input matrices.

Note that the implementation of matrix-matrix products is particularly
complicated because `Field`s were not designed for storing matrices. This, and
the difficulty with matrix broadcasting outlined in `stencilcoefs.jl`, is why we
should develop a better data structure for storing matrices.

################################################################################

  - left_idx_vector = left_idx(axes(vector_field))
  - right_idx_vector = right_idx(axes(vector_field))
  - left_idx_matrix = left_idx(axes(matrix_field))
  - right_idx_matrix = right_idx(axes(matrix_field))

  - rows(vector) = left_idx_vector:right_idx_vector
  - rows(matrix) = left_idx_matrix:right_idx_matrix

  - vector[row] = vector_field[row]
  - matrix[row, col] = matrix_field[row][a * row + b * col + c] for some a, b, c

  - matrix[row, row + lbw] = matrix_field[row][1],
    matrix[row + 1, row + lbw + 1] = matrix_field[row + 1][1], and
    matrix[row, row + ubw] = matrix_field[row][ubw - lbw + 1]                ==>
  - a * row + b * (row + lbw) + c = 1,
    a * (row + 1) + b * (row + lbw + 1) + c = 1, and
    a * row + b * (row + ubw) + c = ubw - lbw + 1                            ==>
  - a = -1, b = 1, and c = 1 - lbw                                           ==>
  - matrix[row, col] = matrix_field[row][col - row - lbw + 1]

################################################################################

Matrix-Vector Product

  - output_vector[row] =
        sum(col -> matrix[row, col] * vector[col], rows(vector))             ==>
  - output_field[idx] =
        output_vector[idx] =
        sum(col -> matrix[idx, col] * vector[col], rows(vector)) =
        sum(
            col -> matrix_field[idx][col - idx - lbw + 1] * vector_field[col],
            left_idx_vector:right_idx_vector
        ) =
        sum(
            i -> matrix_field[idx][i - lbw + 1] * vector_field[idx + i],
            (left_idx_vector - idx):(right_idx_vector - idx)
        )

  - matrix[row, row + i] = 0 ∀ i ∉ lbw:ubw                                   ==>
  - matrix[idx, idx + i] = 0 ∀ i ∉ lbw:ubw                                   ==>
  - matrix_field[idx][i - lbw + 1] = 0 ∀ i ∉ lbw:ubw                         ==>
  - output_field[idx] =
        sum(
            i -> matrix_field[idx][i - lbw + 1] * vector_field[idx + i],
            max(lbw, left_idx_vector - idx):min(ubw, right_idx_vector - idx)
        )

Interior:
  - max(lbw, left_idx_vector - idx) = lbw and
    min(ubw, right_idx_vector - idx) = ubw                                   ==>
  - output_field[idx] =
        sum(
            i -> matrix_field[idx][i - lbw + 1] * vector_field[idx + i],
            lbw:ubw
        )

Left Boundary:
  - max(lbw, left_idx_vector - idx) = left_idx_vector - idx and
    min(ubw, right_idx_vector - idx) = ubw                                   ==>
  - output_field[idx] =
        sum(
            i -> matrix_field[idx][i - lbw + 1] * vector_field[idx + i],
            (left_idx_vector - idx):ubw
        )

  - idx >= left_idx_matrix and
    max(lbw, left_idx_vector - idx) = left_idx_vector - idx                  ==>
  - left_idx_matrix <= idx and left_idx_vector - idx > lbw                   ==>
  - left_idx_matrix <= idx < left_idx_vector - lbw                           ==>
  - number of values of idx at the left boundary =
        max((left_idx_vector - lbw) - left_idx_matrix, 0)
    The number of values of idx at the left boundary is 0 when
        left_idx_matrix >= left_idx_vector - lbw.

Right Boundary:
  - max(lbw, left_idx_vector - idx) = lbw and
    min(ubw, right_idx_vector - idx) = right_idx_vector - idx                ==>
  - output_field[idx] =
        sum(
            i -> matrix_field[idx][i - lbw + 1] * vector_field[idx + i],
            lbw:(right_idx_vector - idx)
        )

  - min(ubw, right_idx_vector - idx) = right_idx_vector - idx and
    idx <= right_idx_matrix                                                  ==>
  - right_idx_vector - idx < ubw and idx <= right_idx_matrix                 ==>
  - right_idx_vector - ubw < idx <= right_idx_matrix                         ==>
  - number of values of idx at the right boundary =
        max(right_idx_matrix - (right_idx_vector - ubw), 0)
    The number of values of idx at the right boundary is 0 when
        right_idx_vector - ubw >= right_idx_matrix.

################################################################################

Matrix-Matrix Product

  - output_matrix[row, col] =
        sum(col′ -> matrix1[row, col′] * matrix2[col′, col], rows(matrix2)) and
    lbw of output_matrix = lbw1 + lbw2                                       ==>
  - output_field[idx][j] =
        output_matrix[idx, j + idx + lbw1 + lbw2 - 1] =
        sum(
            col′ -> matrix1[idx, col′] *
                matrix2[col′, j + idx + lbw1 + lbw2 - 1],
            rows(matrix2)
        ) =
        sum(
            col′ -> matrix1_field[idx][col′ - idx - lbw1 + 1] *
                matrix2_field[col′][j - col′ + idx + lbw1],
            left_idx_matrix2:right_idx_matrix2
        ) =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            (left_idx_matrix2 - idx):(right_idx_matrix2 - idx)
        )

  - matrix1[row, row + i] = 0 ∀ i ∉ lbw1:ubw1                                ==>
  - matrix1[idx, idx + i] = 0 ∀ i ∉ lbw1:ubw1                                ==>
  - matrix1_field[idx][i - lbw1 + 1] = 0 ∀ i ∉ lbw1:ubw1                     ==>
  - output_field[idx][j] =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            max(lbw1, left_idx_matrix2 - idx):min(ubw1, right_idx_matrix2 - idx)
        )

  - matrix2[row, row + k] = 0 ∀ k ∉ lbw2:ubw2                                ==>
  - matrix2[idx + i, (idx + i) + (j - i + lbw1 + lbw2 - 1)] = 0 ∀
        (j - i + lbw1 + lbw2 - 1) ∉ lbw2:ubw2                                ==>
  - matrix2_field[idx + i][j - i + lbw1] = 0 ∀
        i ∉ (j + lbw1 - ubw2 + lbw2 - 1):(j + lbw1 - 1)                      ==>
  - output_field[idx][j] =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            max(lbw1, j + lbw1 - ubw2 + lbw2 - 1, left_idx_matrix2 - idx):
                min(ubw1, j + lbw1 - 1, right_idx_matrix2 - idx)
        ) =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            max(lbw1 + max(0, j - ubw2 + lbw2 - 1), left_idx_matrix2 - idx):
                min(ubw1 + min(0, j - ubw1 + lbw1 - 1), right_idx_matrix2 - idx)
        )

  - ubw1 - lbw1 + 1 = bw1 and ubw2 - lbw2 + 1 = bw2                          ==>
  - output_field[idx][j] =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            max(lbw1 + max(0, j - bw2), left_idx_matrix2 - idx):
                min(ubw1 + min(0, j - bw1), right_idx_matrix2 - idx)
        )

Interior:
  - max(lbw1 + max(0, j - bw2), left_idx_matrix2 - idx) =
        lbw1 + max(0, j - bw2) and
    min(ubw1 + min(0, j - bw1), right_idx_matrix2 - idx) =
        ubw1 + min(0, j - bw1)                                              ==>
  - output_field[idx][j] =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            (lbw1 + max(0, j - bw2)):(ubw1 + min(0, j - bw1))
        )

Left Boundary:
  - min(ubw1 + min(0, j - bw1), right_idx_matrix2 - idx) =
        ubw1 + min(0, j - bw1)                                              ==>
  - output_field[idx][j] =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            max(lbw1 + max(0, j - bw2), left_idx_matrix2 - idx):
                (ubw1 + min(0, j - bw1))
        )

  - idx >= left_idx_matrix1 and
    max(lbw1 + max(0, j - bw2), left_idx_matrix2 - idx) =
        left_idx_matrix2 - idx for some j                                    ==>
  - left_idx_matrix1 <= idx,
    left_idx_matrix2 - idx > lbw1 + max(0, j - bw2), and
    lbw1 + max(0, j - bw2) >= lbw1 + 0 = lbw1  ==>
  - left_idx_matrix1 <= idx and left_idx_matrix2 - idx > lbw1                ==>
  - left_idx_matrix1 <= idx < left_idx_matrix2 - lbw1                        ==>
  - number of values of idx at the left boundary =
        max((left_idx_matrix2 - lbw1) - left_idx_matrix1, 0)
    The number of values of idx at the left boundary is 0 when
        left_idx_matrix1 >= left_idx_matrix2 - lbw1.

Right Boundary:
  - max(lbw1 + max(0, j - bw2), left_idx_matrix2 - idx) =
        lbw1 + max(0, j - bw2)                                               ==>
  - output_field[idx][j] =
        sum(
            i -> matrix1_field[idx][i - lbw1 + 1] *
                matrix2_field[idx + i][j - i + lbw1],
            (lbw1 + max(0, j - bw2)):
                min(ubw1 + min(0, j - bw1), right_idx_matrix2 - idx)
        )

  - min(ubw1 + min(0, j - bw1), right_idx_matrix2 - idx) =
        right_idx_matrix2 - idx for some j and
    idx <= right_idx_matrix1                                                 ==>
  - right_idx_matrix2 - idx < ubw1 + min(0, j - bw1),
    ubw1 + min(0, j - bw1) <= ubw1 + 0 = ubw1, and
    idx <= right_idx_matrix1                                                 ==>
  - right_idx_matrix2 - idx < ubw1 and idx <= right_idx_matrix1              ==>
  - right_idx_matrix2 - ubw1 < idx <= right_idx_matrix1                      ==>
  - number of values of idx at the right boundary =
        max(right_idx_matrix1 - (right_idx_matrix2 - ubw), 0)
    The number of values of idx at the right boundary is 0 when
        right_idx_matrix2 - ubw >= right_idx_matrix1.

################################################################################

Bandwidths of Matrix-Matrix Product

  - output_matrix[row, row + i] =
        sum(
            col′ -> matrix1[row, col′] * matrix2[col′, row + i],
            rows(matrix2),
        ) =
        sum(
            col′ -> ifelse(
                col′ - row ∉ lbw1:ubw1 or row + i - col′ ∉ lbw2:ubw2,
                0,
                matrix1[row, col′] * matrix2[col′, row + i]
            ),
            rows(matrix2),
        ) =
        sum(
            col′ -> ifelse(
                col′ - row ∉ lbw1:ubw1 or col′ - row ∉ (i - ubw2):(i - lbw2),
                0,
                matrix1[row, col′] * matrix2[col′, row + i]
            ),
            rows(matrix2),
        ) =
        sum(
            col′ -> ifelse(
                col′ - row ∉ max(lbw1, i - ubw2):min(ubw1, i - lbw2),
                0,
                matrix1[row, col′] * matrix2[col′, row + i]
            ),
            rows(matrix2),
        )

  - i ∉ (lbw1 + lbw2):(ubw1 + ubw2)                                          ==>
  - i > ubw1 + ubw2 or i < lbw1 + lbw2                                       ==>
  - ubw1 < i - ubw2 or i - lbw2 < lbw1                                       ==>
  - min(ubw1, i - lbw2) < max(lbw1, i - ubw2)                                ==>
  - col′ - row ∉ max(lbw1, i - ubw2):min(ubw1, i - lbw2) ∀
        col′ ∈ rows(matrix2)                                                 ==>
  - output_matrix[row, row + i] = 0                                          ==>
  - bandwidths of output_matrix = (lbw1 + lbw2, ubw1 + ubw2)
=#

abstract type PointwiseStencilOperator <: FiniteDifferenceOperator end

struct LeftStencilBoundary <: BoundaryCondition end
struct RightStencilBoundary <: BoundaryCondition end

abstract type AbstractIndexRangeType end
struct IndexRangeInteriorType <: AbstractIndexRangeType end
struct IndexRangeLeftType <: AbstractIndexRangeType end
struct IndexRangeRightType <: AbstractIndexRangeType end

has_boundary(
    ::PointwiseStencilOperator,
    ::LeftBoundaryWindow{name},
) where {name} = true
has_boundary(
    ::PointwiseStencilOperator,
    ::RightBoundaryWindow{name},
) where {name} = true

get_boundary(
    ::PointwiseStencilOperator,
    ::LeftBoundaryWindow{name},
) where {name} = LeftStencilBoundary()
get_boundary(
    ::PointwiseStencilOperator,
    ::RightBoundaryWindow{name},
) where {name} = RightStencilBoundary()

stencil_interior_width(::PointwiseStencilOperator, stencil, arg) =
    ((0, 0), bandwidths(eltype(stencil)))

function boundary_width(
    ::PointwiseStencilOperator,
    ::LeftStencilBoundary,
    stencil,
    arg,
)
    lbw = bandwidths(eltype(stencil))[1]
    return max((left_idx(axes(arg)) - lbw) - left_idx(axes(stencil)), 0)
end
function boundary_width(
    ::PointwiseStencilOperator,
    ::RightStencilBoundary,
    stencil,
    arg,
)
    ubw = bandwidths(eltype(stencil))[2]
    return max(right_idx(axes(stencil)) - (right_idx(axes(arg)) - ubw), 0)
end

##
## ApplyStencil
##

# Matrix-Vector Product
struct ApplyStencil <: PointwiseStencilOperator end

# TODO: This is not correct for the same reason as the other 2-arg FD operators.
return_eltype(::ApplyStencil, stencil, arg) = eltype(eltype(stencil))

return_space(::ApplyStencil, stencil_space, arg_space) = stencil_space

function apply_stencil_at_idx(i_vals, stencil, arg, loc, idx, hidx)
    coefs = getidx(stencil, loc, idx, hidx)
    lbw = bandwidths(eltype(stencil))[1]
    i_func = i -> coefs[i - lbw + 1] ⊠ getidx(arg, loc, idx + i, hidx)
    return length(i_vals) == 0 ? zero(eltype(eltype(stencil))) :
           mapreduce(i_func, ⊞, i_vals)
end

function stencil_interior(::ApplyStencil, loc, idx, hidx, stencil, arg)
    lbw, ubw = bandwidths(eltype(stencil))
    i_vals = lbw:ubw
    return apply_stencil_at_idx(i_vals, stencil, arg, loc, idx, hidx)
end

function stencil_left_boundary(
    ::ApplyStencil,
    ::LeftStencilBoundary,
    loc,
    idx,
    hidx,
    stencil,
    arg,
)
    ubw = bandwidths(eltype(stencil))[2]
    i_vals = (left_idx(axes(arg)) - idx):ubw
    return apply_stencil_at_idx(i_vals, stencil, arg, loc, idx, hidx)
end

function stencil_right_boundary(
    ::ApplyStencil,
    ::RightStencilBoundary,
    loc,
    idx,
    hidx,
    stencil,
    arg,
)
    lbw = bandwidths(eltype(stencil))[1]
    i_vals = lbw:(right_idx(axes(arg)) - idx)
    return apply_stencil_at_idx(i_vals, stencil, arg, loc, idx, hidx)
end


##
## ComposeStencils
##

# Matrix-Matrix Product
struct ComposeStencils <: PointwiseStencilOperator end

function composed_bandwidths(stencil1, stencil2)
    lbw1, ubw1 = bandwidths(eltype(stencil1))
    lbw2, ubw2 = bandwidths(eltype(stencil2))
    return (lbw1 + lbw2, ubw1 + ubw2)
end

# TODO: This is not correct for the same reason as the other 2-arg FD operators.
function return_eltype(::ComposeStencils, stencil1, stencil2)
    lbw, ubw = composed_bandwidths(stencil1, stencil2)
    T = eltype(eltype(stencil1))
    return StencilCoefs{lbw, ubw, NTuple{ubw - lbw + 1, T}}
end

return_space(::ComposeStencils, stencil1_space, stencil2_space) = stencil1_space

function bandwidth_info(stencil1, stencil2)
    lbw1, ubw1 = bandwidths(eltype(stencil1))
    bw1 = bandwidth(eltype(stencil1))
    bw2 = bandwidth(eltype(stencil2))
    return lbw1, ubw1, bw1, bw2
end

is_non_zero(::Type{IndexRangeInteriorType}, a, b, space2, idx, k)::Bool = true

function is_non_zero(::Type{IndexRangeLeftType}, a, b, space2, idx, k)::Bool
    min_i = left_idx(space2) - idx
    a_lim = max(a, min_i)
    return a_lim ≤ k ≤ b
end

function is_non_zero(::Type{IndexRangeRightType}, a, b, space2, idx, k)::Bool
    max_i = right_idx(space2) - idx
    b_lim = min(b, max_i)
    return a ≤ k ≤ b_lim
end

function compose_stencils_at_idx(
    ::Type{ir_type},
    stencil1,
    stencil2,
    loc,
    idx,
    hidx,
) where {ir_type <: AbstractIndexRangeType}

    coefs1 = getidx(stencil1, loc, idx, hidx)
    lbw, ubw = composed_bandwidths(stencil1, stencil2)
    lbw1, ubw1, bw1, bw2 = bandwidth_info(stencil1, stencil2)
    space2 = axes(stencil2)
    n = (ubw - lbw + 1)::Int
    ntup = ntuple(Val(n)) do j
        Base.@_inline_meta
        a = (lbw1 + max(0, j - bw2))::typeof(lbw1)
        b = (ubw1 + min(0, j - bw1))::typeof(lbw1)
        N = (b - a + 1)::Int
        inner_ntup = ntuple(Val(N)) do ki
            Base.@_inline_meta
            k = (a + ki - 1)::typeof(lbw1)
            (k, is_non_zero(ir_type, a, b, space2, idx, k))
        end
        mapreduce(⊞, inner_ntup) do tup
            Base.@_inline_meta
            i = first(tup)::typeof(lbw1)
            is_non_zero_value = last(tup)::Bool
            if is_non_zero_value
                coefs1[i - lbw1 + 1] ⊠
                getidx(stencil2, loc, idx + i, hidx)[j - i + lbw1]
            else
                zero(eltype(eltype(stencil1)))
            end
        end
    end
    return StencilCoefs{lbw, ubw}(ntup)
end

function stencil_interior(::ComposeStencils, loc, idx, hidx, stencil1, stencil2)
    return compose_stencils_at_idx(
        IndexRangeInteriorType,
        stencil1,
        stencil2,
        loc,
        idx,
        hidx,
    )
end

function stencil_left_boundary(
    ::ComposeStencils,
    ::LeftStencilBoundary,
    loc,
    idx,
    hidx,
    stencil1,
    stencil2,
)
    return compose_stencils_at_idx(
        IndexRangeLeftType,
        stencil1,
        stencil2,
        loc,
        idx,
        hidx,
    )
end

function stencil_right_boundary(
    ::ComposeStencils,
    ::RightStencilBoundary,
    loc,
    idx,
    hidx,
    stencil1,
    stencil2,
)
    return compose_stencils_at_idx(
        IndexRangeRightType,
        stencil1,
        stencil2,
        loc,
        idx,
        hidx,
    )
end


##
## stencil_solve!
##

# TODO: Banded matrix inversion (solve Ax = b, where A is a banded matrix)
