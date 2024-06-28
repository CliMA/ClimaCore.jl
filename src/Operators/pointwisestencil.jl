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

struct LeftStencilBoundary <: AbstractBoundaryCondition end
struct RightStencilBoundary <: AbstractBoundaryCondition end

abstract type AbstractIndexRangeType end
struct IndexRangeInteriorType <: AbstractIndexRangeType end
struct IndexRangeLeftType <: AbstractIndexRangeType end
struct IndexRangeRightType <: AbstractIndexRangeType end

function get_range(
    ::Type{IndexRangeInteriorType},
    space,
    stencil1,
    stencil2,
    idx,
    j,
)
    lbw1, ubw1, bw1, bw2 = bandwidth_info(stencil1, stencil2)
    a = lbw1 + max(0, j - bw2)
    b = ubw1 + min(0, j - bw1)
    return a:b
end

function get_range(
    ::Type{IndexRangeLeftType},
    space,
    stencil1,
    stencil2,
    idx,
    j,
)
    lbw1, ubw1, bw1, bw2 = bandwidth_info(stencil1, stencil2)
    stencil2_space = reconstruct_placeholder_space(axes(stencil2), space)
    min_i = left_idx(stencil2_space) - idx
    a = max(lbw1 + max(0, j - bw2), min_i)
    b = (ubw1 + min(0, j - bw1))
    return a:b
end

function get_range(
    ::Type{IndexRangeRightType},
    space,
    stencil1,
    stencil2,
    idx,
    j,
)
    lbw1, ubw1, bw1, bw2 = bandwidth_info(stencil1, stencil2)
    stencil2_space = reconstruct_placeholder_space(axes(stencil2), space)
    a = lbw1 + max(0, j - bw2)
    max_i = right_idx(stencil2_space) - idx
    b = min(ubw1 + min(0, j - bw1), max_i)
    return a:b
end

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

# given a stencil of left-bandwidth lbw, what is the index of the leftmost interior point?
left_interior_idx_bandwidth(space::AbstractSpace, lbw::Integer) =
    left_idx(space) - lbw

left_interior_idx_bandwidth(
    space::Union{
        Spaces.FaceFiniteDifferenceSpace,
        Spaces.FaceExtrudedFiniteDifferenceSpace,
    },
    lbw::PlusHalf,
) = left_idx(space) - lbw + half

left_interior_idx_bandwidth(
    space::Union{
        Spaces.CenterFiniteDifferenceSpace,
        Spaces.CenterExtrudedFiniteDifferenceSpace,
    },
    lbw::PlusHalf,
) = left_idx(space) - lbw - half


function left_interior_idx(
    space::AbstractSpace,
    op::PointwiseStencilOperator,
    bc::LeftStencilBoundary,
    stencil,
    arg,
)
    lbw = bandwidths(eltype(stencil))[1]
    left_interior_idx_bandwidth(space, lbw)
end

right_interior_idx_bandwidth(space::AbstractSpace, rbw::Integer) =
    right_idx(space) - rbw

right_interior_idx_bandwidth(
    space::Union{
        Spaces.FaceFiniteDifferenceSpace,
        Spaces.FaceExtrudedFiniteDifferenceSpace,
    },
    rbw::PlusHalf,
) = right_idx(space) - rbw - half

right_interior_idx_bandwidth(
    space::Union{
        Spaces.CenterFiniteDifferenceSpace,
        Spaces.CenterExtrudedFiniteDifferenceSpace,
    },
    rbw::PlusHalf,
) = right_idx(space) - rbw + half


@inline function right_interior_idx(
    space::AbstractSpace,
    op::PointwiseStencilOperator,
    bc::RightStencilBoundary,
    stencil,
    arg,
)
    rbw = bandwidths(eltype(stencil))[2]
    right_interior_idx_bandwidth(space, rbw)
end

##
## ApplyStencil
##

# Matrix-Vector Product
struct ApplyStencil <: PointwiseStencilOperator end

# TODO: This is not correct for the same reason as the other 2-arg FD operators.
return_eltype(::ApplyStencil, stencil, arg) = eltype(eltype(stencil))

return_space(::ApplyStencil, stencil_space, arg_space) = stencil_space

Base.@propagate_inbounds function apply_stencil_at_idx(
    i_vals,
    stencil,
    arg,
    loc,
    space,
    idx,
    hidx,
)
    coefs = getidx(space, stencil, loc, idx, hidx)
    lbw = bandwidths(eltype(stencil))[1]
    val_type = eltype(eltype(stencil))
    val = zero(val_type)::val_type
    @inbounds for j in 1:length(i_vals)
        i = i_vals[j]
        val = val ⊞ coefs[i - lbw + 1] ⊠ getidx(space, arg, loc, idx + i, hidx)
    end
    return val
end

Base.@propagate_inbounds function stencil_interior(
    ::ApplyStencil,
    loc,
    space,
    idx,
    hidx,
    stencil,
    arg,
)
    lbw, ubw = bandwidths(eltype(stencil))
    i_vals = lbw:ubw
    return apply_stencil_at_idx(i_vals, stencil, arg, loc, space, idx, hidx)
end

Base.@propagate_inbounds function stencil_left_boundary(
    ::ApplyStencil,
    ::LeftStencilBoundary,
    loc,
    space,
    idx,
    hidx,
    stencil,
    arg,
)
    ubw = bandwidths(eltype(stencil))[2]
    arg_space = reconstruct_placeholder_space(axes(arg), space)
    i_vals = (left_idx(arg_space) - idx):ubw
    return apply_stencil_at_idx(i_vals, stencil, arg, loc, space, idx, hidx)
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::ApplyStencil,
    ::RightStencilBoundary,
    loc,
    space,
    idx,
    hidx,
    stencil,
    arg,
)
    lbw = bandwidths(eltype(stencil))[1]
    arg_space = reconstruct_placeholder_space(axes(arg), space)
    i_vals = lbw:(right_idx(arg_space) - idx)
    apply_stencil_at_idx(i_vals, stencil, arg, loc, space, idx, hidx)
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

# TODO: find out why using Base.@propagate_inbounds hangs
function compose_stencils_at_idx(
    ::Type{ir_type},
    stencil1,
    stencil2,
    loc,
    space,
    idx,
    hidx,
) where {ir_type <: AbstractIndexRangeType}

    coefs1 = getidx(space, stencil1, loc, idx, hidx)
    lbw1 = bandwidths(eltype(stencil1))[1]
    lbw, ubw = composed_bandwidths(stencil1, stencil2)
    n = (ubw - lbw + 1)::Int
    zeroT = eltype(eltype(stencil1))
    ntup = (
        ntuple(Val(n)) do j
            Base.@_inline_meta
            val = zero(zeroT)::zeroT
            for i in get_range(ir_type, space, stencil1, stencil2, idx, j)
                val =
                    val ⊞
                    coefs1[i - lbw1 + 1] ⊠
                    getidx(space, stencil2, loc, idx + i, hidx)[j - i + lbw1]
            end
            val
        end
    )::NTuple{n, zeroT}
    return StencilCoefs{lbw, ubw}(ntup)
end

# TODO: find out why using Base.@propagate_inbounds hangs
function stencil_interior(
    ::ComposeStencils,
    loc,
    space,
    idx,
    hidx,
    stencil1,
    stencil2,
)
    return compose_stencils_at_idx(
        IndexRangeInteriorType,
        stencil1,
        stencil2,
        loc,
        space,
        idx,
        hidx,
    )
end


# TODO: find out why using Base.@propagate_inbounds hangs
function stencil_left_boundary(
    ::ComposeStencils,
    ::LeftStencilBoundary,
    loc,
    space,
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
        space,
        idx,
        hidx,
    )
end

# TODO: find out why using Base.@propagate_inbounds hangs
function stencil_right_boundary(
    ::ComposeStencils,
    ::RightStencilBoundary,
    loc,
    space,
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
        space,
        idx,
        hidx,
    )
end


##
## stencil_solve!
##

# TODO: Banded matrix inversion (solve Ax = b, where A is a banded matrix)
