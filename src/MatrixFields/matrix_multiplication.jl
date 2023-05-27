"""
    MultiplyColumnwiseBandMatrixField()

An operator that multiplies a `ColumnwiseBandMatrixField` by another `Field`,
i.e., matrix-vector or matrix-matrix multiplication. The `⋅` symbol is an alias
for `MultiplyColumnwiseBandMatrixField()`.

What follows is a derivation of the algorithm used by this operator with
single-column `Field`s. For `Field`s on multiple columns, the same computation
is done for each column.
    
In this derivation, we will use ``M_1`` and ``M_2`` to denote two
`ColumnwiseBandMatrixField`s, and we will use ``V`` to denote a regular (vector)
`Field`. For both ``M_1`` and ``M_2``, we will use the array-like index notation
``M[row, col]`` to denote ``M[row][col-row]``, i.e., the the entry in the
`BandMatrixRow` ``M[row]`` located on the diagonal with index ``col - row``.

# 1. Matrix-Vector Multiplication

From the definition of matrix-vector multiplication,
```math
(M_1 ⋅ V)[i] = \\sum_k M_1[i, k] * V[k].
```
To establish bounds on the values of ``k``, let us define the following values:
- ``li ={}```left_idx```(```axes```(V))``
- ``ri ={}```right_idx```(```axes```(V))``
- ``ld_1, ud_1 ={}```outer_diagonals```(```eltype```(M_1))``
Note that, if `axes```(V))`` is unavailable, we can obtain ``li`` and ``ri``
from `axes```(M_1))`` by getting `left_idx```(```axes```(M_1))`` and
`right_idx```(```axes```(M_1))``, and then shifting them by either `0`, `half`,
or `-half`, depending on whether ``M_1`` is a square matrix, a face-to-center
matrix, or a center-to-face matrix.

The values ``M_1[i, k]`` and ``V[k]`` are only well-defined if ``k`` is a valid
row index for ``V`` and ``k - i`` is a valid diagonal index for ``M_1``, which
means that
```math
li \\leq k \\leq ri \\quad \\text{and} \\quad ld_1 \\leq k - i \\leq ud_1.
```
Combining these into a single inequality gives us
```math
\\text{max}(li, i + ld_1) \\leq k \\leq \\text{min}(ri, i + ud_1).
```
So, we can rewrite the expression for ``(M_1 ⋅ V)[i]`` as
```math
(M_1 ⋅ V)[i] =
    \\sum_{k\\ =\\ \\text{max}(li, i + ld_1)}^{\\text{min}(ri, i + ud_1)}
    M_1[i, k] * V[k].
```
If we replace the variable ``k`` with ``d = k - i`` and switch from array-like
indexing to `Field` indexing, we find that
```math
(M_1 ⋅ V)[i] =
    \\sum_{d\\ =\\ \\text{max}(li - i, ld_1)}^{\\text{min}(ri - i, ud_1)}
    M_1[i][d] * V[i + d].
```
If ``li - ld_1 \\leq i \\leq ri - ud_1``, then
``\\text{max}(li - i, ld_1) = ld_1`` and ``\\text{min}(ri - i, ud_1) = ud_1``,
so we can simplify this to
```math
(M_1 ⋅ V)[i] = \\sum_{d = ld_1}^{ud_1} M_1[i][d] * V[i + d].
```
The values of ``i`` in this range are considered to be in the "interior" of the
operator, while those not in this range (for which we cannot make the above
simplification) are considered to be on the "boundary".

# 2. Matrix-Matrix Multiplication

From the definition of matrix-matrix multiplication,
```math
(M_1 ⋅ M_2)[i, j] = \\sum_k M_1[i, k] * M_2[k, j].
```
To establish bounds on the values of ``k``, let us define the following values:
- ``li ={}```left_idx```(```axes```(M_2))``
- ``ri ={}```right_idx```(```axes```(M_2))``
- ``ld_1, ud_1 ={}```outer_diagonals```(```eltype```(M_1))``
- ``ld_2, ud_2 ={}```outer_diagonals```(```eltype```(M_2))``
Note that, if `axes```(M_2))`` is unavailable, we can obtain ``li`` and ``ri``
from `axes```(M_1))`` by getting `left_idx```(```axes```(M_1))`` and
`right_idx```(```axes```(M_1))``, and then shifting them by either `0`, `half`,
or `-half`, depending on whether ``M_1`` is a square matrix, a face-to-center
matrix, or a center-to-face matrix.

The values ``M_1[i, k]`` and ``M_2[k, j]`` are only well-defined if ``k`` is a
valid row index for ``M_2``, ``k - i`` is a valid diagonal index for ``M_1``,
and ``j - k`` is a valid diagonal index for ``M_2``, which means that
```math
li \\leq k \\leq ri, \\qquad ld_1 \\leq k - i \\leq ud_1,
\\quad \\text{and} \\quad ld_2 \\leq j - k \\leq ud_2.
```
Combining these into a single inequality gives us
```math
\\text{max}(li, i + ld_1, j - ud_2) \\leq k \\leq
\\text{min}(ri, i + ud_1, j - ld_2).
```
So, we can rewrite the expression for ``(M_1 ⋅ M_2)[i, j]`` as
```math
(M_1 ⋅ M_2)[i, j] =
    \\sum_{
        k\\ =\\ \\text{max}(li, i + ld_1, j - ud_2)
    }^{\\text{min}(ri, i + ud_1, j - ld_2)}
    M_1[i, k] * M_2[k, j].
```
If we replace the variable ``k`` with ``d = k - i``, replace the variable ``j``
with ``d_{prod} = j - i``, and switch from array-like indexing to `Field`
indexing, we find that
```math
(M_1 ⋅ M_2)[i][d_{prod}] =
    \\sum_{
        d\\ =\\ \\text{max}(li - i, ld_1, d_{prod} - ud_2)
    }^{\\text{min}(ri - i, ud_1, d_{prod} - ld_2)}
    M_1[i][d] * M_2[i + d][d_{prod} - d].
```
If ``li - ld_1 \\leq i \\leq ri - ud_1``, then
``\\text{max}(li - i, ld_1) = ld_1`` and ``\\text{min}(ri - i, ud_1) = ud_1``,
so we can simplify this to
```math
(M_1 ⋅ M_2)[i][d_{prod}] =
    \\sum_{
        d\\ =\\ \\text{max}(ld_1, d_{prod} - ud_2)
    }^{\\text{min}(ud_1, d_{prod} - ld_2)}
    M_1[i][d] * M_2[i + d][d_{prod} - d].
```
The values of ``i`` in this range are considered to be in the "interior" of the
operator, while those not in this range (for which we cannot make the above
simplification) are considered to be on the "boundary".

We only need to compute ``(M_1 ⋅ M_2)[i][d_{prod}]`` for values of ``d_{prod}``
that correspond to a nonempty sum in the interior, i.e, those for which
```math
\\text{max}(ld_1, d_{prod} - ud_2) \\leq \\text{min}(ud_1, d_{prod} - ld_2).
```
This corresponds to the four inequalities
```math
ld_1 \\leq ud_1, \\qquad ld_1 \\leq d_{prod} - ld_2, \\qquad
d_{prod} - ud_2 \\leq ud_1, \\quad \\text{and} \\quad
d_{prod} - ud_2 \\leq d_{prod} - ld_2.
```
By definition, ``ld_1 \\leq ud_1`` and ``ld_2 \\leq ud_2``, so the first and
last inequality are always true. Rearranging the remaining two inequalities
tells us that
```math
ld_1 + ld_2 \\leq d_{prod} \\leq ud_1 + ud_2.
```
In other words, the outer diagonal indices of ``M_1 ⋅ M_2`` are ``ld_1 + ld_2``
and ``ud_1 + ud_2``.
"""
struct MultiplyColumnwiseBandMatrixField <: Operators.FiniteDifferenceOperator end
const ⋅ = MultiplyColumnwiseBandMatrixField()

struct TopLeftMatrixCorner <: Operators.AbstractBoundaryCondition end
struct BottomRightMatrixCorner <: Operators.AbstractBoundaryCondition end

Operators.has_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::Operators.LeftBoundaryWindow{name},
) where {name} = true
Operators.has_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::Operators.RightBoundaryWindow{name},
) where {name} = true

Operators.get_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::Operators.LeftBoundaryWindow{name},
) where {name} = TopLeftMatrixCorner()
Operators.get_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::Operators.RightBoundaryWindow{name},
) where {name} = BottomRightMatrixCorner()

Operators.stencil_interior_width(
    ::MultiplyColumnwiseBandMatrixField,
    matrix1,
    arg,
) = ((0, 0), outer_diagonals(eltype(matrix1)))

# Obtain left_idx(arg_space) from matrix1_space.
arg_left_idx(matrix1_space, matrix1) = arg_left_idx(
    Operators.left_idx(matrix1_space),
    matrix_shape(matrix1, matrix1_space),
)
arg_left_idx(matrix1_left_idx, ::Square) = matrix1_left_idx
arg_left_idx(matrix1_left_idx, ::FaceToCenter) = matrix1_left_idx - half
arg_left_idx(matrix1_left_idx, ::CenterToFace) = matrix1_left_idx + half

# Obtain right_idx(arg_space) from matrix1_space.
arg_right_idx(matrix1_space, matrix1) = arg_right_idx(
    Operators.right_idx(matrix1_space),
    matrix_shape(matrix1, matrix1_space),
)
arg_right_idx(matrix1_right_idx, ::Square) = matrix1_right_idx
arg_right_idx(matrix1_right_idx, ::FaceToCenter) = matrix1_right_idx + half
arg_right_idx(matrix1_right_idx, ::CenterToFace) = matrix1_right_idx - half

Operators.left_interior_idx(
    space::Spaces.AbstractSpace,
    ::MultiplyColumnwiseBandMatrixField,
    ::TopLeftMatrixCorner,
    matrix1,
    arg,
) = arg_left_idx(space, matrix1) - outer_diagonals(eltype(matrix1))[1]
Operators.right_interior_idx(
    space::Spaces.AbstractSpace,
    ::MultiplyColumnwiseBandMatrixField,
    ::BottomRightMatrixCorner,
    matrix1,
    arg,
) = arg_right_idx(space, matrix1) - outer_diagonals(eltype(matrix1))[2]

function Operators.return_eltype(
    ::MultiplyColumnwiseBandMatrixField,
    matrix1,
    arg,
)
    eltype(matrix1) <: BandMatrixRow || error(
        "The first argument of ⋅ must have elements of type BandMatrixRow, but \
         the given argument has elements of type $(eltype(matrix1))",
    )
    if eltype(arg) <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = arg
        ld1, ud1 = outer_diagonals(eltype(matrix1))
        ld2, ud2 = outer_diagonals(eltype(matrix2))
        prod_ld, prod_ud = ld1 + ld2, ud1 + ud2
        prod_value_type =
            rmul_return_type(eltype(eltype(matrix1)), eltype(eltype(matrix2)))
        return band_matrix_row_type(prod_ld, prod_ud, prod_value_type)
    else # matrix-vector multiplication
        vector = arg
        return rmul_return_type(eltype(eltype(matrix1)), eltype(vector))
    end
end

Operators.return_space(::MultiplyColumnwiseBandMatrixField, space1, space2) =
    space1

# TODO: Use @propagate_inbounds here, and remove @inbounds from this function.
# As of Julia 1.8, doing this increases compilation time by more than an order
# of magnitude, and it also makes type inference fail for some complicated
# matrix field broadcast expressions (in particular, those that involve products
# of linear combinations of matrix fields). Not using @propagate_inbounds causes
# matrix field broadcast expressions to take roughly 3 or 4 times longer to
# evaluate, but this is less significant than the decrease in compilation time.
function multiply_matrix_at_index(
    loc,
    space,
    idx,
    hidx,
    matrix1,
    arg,
    ::Val{is_interior},
) where {is_interior}
    ld1, ud1 = outer_diagonals(eltype(matrix1))
    arg_space = Operators.reconstruct_placeholder_space(axes(arg), space)
    ld1_or_boundary_ld1 =
        is_interior ? ld1 : max(Operators.left_idx(arg_space) - idx, ld1)
    ud1_or_boundary_ud1 =
        is_interior ? ud1 : min(Operators.right_idx(arg_space) - idx, ud1)
    prod_type = Operators.return_eltype(⋅, matrix1, arg)
    matrix1_row = @inbounds Operators.getidx(space, matrix1, loc, idx, hidx)
    if eltype(arg) <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = arg
        matrix2_rows = map((ld1:ud1...,)) do d
            # TODO: Use @propagate_inbounds_meta instead of @inline_meta.
            Base.@_inline_meta
            if is_interior || ld1_or_boundary_ld1 <= d <= ud1_or_boundary_ud1
                @inbounds Operators.getidx(space, matrix2, loc, idx + d, hidx)
            else
                rzero(eltype(matrix2)) # This value is never used.
            end
        end # The rows are precomputed to avoid recomputing them multiple times.
        matrix2_rows_wrapper = BandMatrixRow{ld1}(matrix2_rows...)
        ld2, ud2 = outer_diagonals(eltype(matrix2))
        prod_ld, prod_ud = outer_diagonals(prod_type)
        zero_value = rzero(eltype(prod_type))
        prod_entries = map((prod_ld:prod_ud...,)) do prod_d
            # TODO: Use @propagate_inbounds_meta instead of @inline_meta.
            Base.@_inline_meta
            min_d = max(ld1_or_boundary_ld1, prod_d - ud2)
            max_d = min(ud1_or_boundary_ud1, prod_d - ld2)
            # Note: If min_d:max_d is an empty range, then the current entry
            # lies outside of the product matrix, so it should never be used in
            # any computations. By initializing prod_entry to zero_value, we are
            # implicitly setting all such entries to 0. We could alternatively
            # set all such entries to NaN (in order to more easily catch user
            # errors that involve accidentally using these entires), but that
            # would not generalize to non-floating-point types like Int or Bool.
            prod_entry = zero_value
            @inbounds for d in min_d:max_d
                value1 = matrix1_row[d]
                value2 = matrix2_rows_wrapper[d][prod_d - d]
                value2_lg = Geometry.LocalGeometry(space, idx + d, hidx)
                prod_entry = radd(
                    prod_entry,
                    rmul_with_projection(value1, value2, value2_lg),
                )
            end # Using this for-loop is currently faster than using mapreduce.
            prod_entry
        end
        return BandMatrixRow{prod_ld}(prod_entries...)
    else # matrix-vector multiplication
        vector = arg
        prod_value = rzero(prod_type)
        @inbounds for d in ld1_or_boundary_ld1:ud1_or_boundary_ud1
            value1 = matrix1_row[d]
            value2 = Operators.getidx(space, vector, loc, idx + d, hidx)
            value2_lg = Geometry.LocalGeometry(space, idx + d, hidx)
            prod_value = radd(
                prod_value,
                rmul_with_projection(value1, value2, value2_lg),
            )
        end # Using this for-loop is currently faster than using mapreduce.
        return prod_value
    end
end

Base.@propagate_inbounds Operators.stencil_interior(
    ::MultiplyColumnwiseBandMatrixField,
    loc,
    space,
    idx,
    hidx,
    matrix1,
    arg,
) = multiply_matrix_at_index(loc, space, idx, hidx, matrix1, arg, Val(true))

Base.@propagate_inbounds Operators.stencil_left_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::TopLeftMatrixCorner,
    loc,
    space,
    idx,
    hidx,
    matrix1,
    arg,
) = multiply_matrix_at_index(loc, space, idx, hidx, matrix1, arg, Val(false))

Base.@propagate_inbounds Operators.stencil_right_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    ::BottomRightMatrixCorner,
    loc,
    space,
    idx,
    hidx,
    matrix1,
    arg,
) = multiply_matrix_at_index(loc, space, idx, hidx, matrix1, arg, Val(false))

# For matrix field broadcast expressions involving 4 or more matrices, we
# sometimes hit a recursion limit and de-optimize.
# We know that the recursion will terminate due to the fact that broadcast
# expressions are not self-referential.
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(multiply_matrix_at_index)
        m.recursion_relation = dont_limit
    end
end
