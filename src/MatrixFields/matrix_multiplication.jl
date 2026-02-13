"""
    MultiplyColumnwiseBandMatrixField()

An operator that multiplies a `ColumnwiseBandMatrixField` by another `Field`,
i.e., matrix-vector or matrix-matrix multiplication.

What follows is a derivation of the algorithm used by this operator with
single-column `Field`s. For `Field`s on multiple columns, the same computation
is done for each column.

In this derivation, we will use ``M_1`` and ``M_2`` to denote two
`ColumnwiseBandMatrixField`s, and we will use ``V`` to denote a regular
(vector-like) `Field`. For both ``M_1`` and ``M_2``, we will use the array-like
index notation ``M[row, col]`` to denote ``M[row][col-row]``, i.e., the entry in
the `BandMatrixRow` ``M[row]`` located on the diagonal with index ``col - row``.
We will also use `outer_indices```(```space```)`` to denote the tuple
``(```left_idx```(```space```), ```right_idx```(```space```))``.

# 1. Matrix-Vector Multiplication

From the definition of matrix-vector multiplication,
```math
(M_1 * V)[i] = \\sum_k M_1[i, k] * V[k].
```
To establish bounds on the values of ``k``, let us define the following values:
- ``li_1, ri_1 ={}```outer_indices```(```column_axes```(M_1))``
- ``ld_1, ud_1 ={}```outer_diagonals```(```eltype```(M_1))``

Since ``M_1[i, k]`` is only well-defined if ``k`` is a valid column index and
``k - i`` is a valid diagonal index, we know that
```math
li_1 \\leq k \\leq ri_1 \\quad \\text{and} \\quad ld_1 \\leq k - i \\leq ud_1.
```
Combining these into a single inequality gives us
```math
\\text{max}(li_1, i + ld_1) \\leq k \\leq \\text{min}(ri_1, i + ud_1).
```
So, we can rewrite the expression for ``(M_1 * V)[i]`` as
```math
(M_1 * V)[i] =
    \\sum_{k\\ =\\ \\text{max}(li_1, i + ld_1)}^{\\text{min}(ri_1, i + ud_1)}
    M_1[i, k] * V[k].
```
If we replace the variable ``k`` with ``d = k - i`` and switch from array-like
indexing to `Field` indexing, we find that
```math
(M_1 * V)[i] =
    \\sum_{d\\ =\\ \\text{max}(li_1 - i, ld_1)}^{\\text{min}(ri_1 - i, ud_1)}
    M_1[i][d] * V[i + d].
```

## 1.1 Interior vs. Boundary Indices

Now, suppose that the row index ``i`` is such that
```math
li_1 - ld_1 \\leq i \\leq ri_1 - ud_1.
```
If this is the case, then the bounds on ``d`` can be simplified to
```math
\\text{max}(li_1 - i, ld_1) = ld_1 \\quad \\text{and} \\quad
\\text{min}(ri_1 - i, ud_1) = ud_1.
```
The expression for ``(M_1 * V)[i]`` then becomes
```math
(M_1 * V)[i] = \\sum_{d = ld_1}^{ud_1} M_1[i][d] * V[i + d].
```
The values of ``i`` in this range are considered to be in the "interior" of the
operator, while those not in this range (for which we cannot make the above
simplification) are considered to be on the "boundary".

# 2. Matrix-Matrix Multiplication

From the definition of matrix-matrix multiplication,
```math
(M_1 * M_2)[i, j] = \\sum_k M_1[i, k] * M_2[k, j].
```
To establish bounds on the values of ``j`` and ``k``, let us define the
following values:
- ``li_1, ri_1 ={}```outer_indices```(```column_axes```(M_1))``
- ``ld_1, ud_1 ={}```outer_diagonals```(```eltype```(M_1))``
- ``li_2, ri_2 ={}```outer_indices```(```column_axes```(M_2))``
- ``ld_2, ud_2 ={}```outer_diagonals```(```eltype```(M_2))``

In addition, let ``ld_{prod}`` and ``ud_{prod}`` denote the outer diagonal
indices of the product matrix ``M_1 * M_2``. We will derive the values of
``ld_{prod}`` and ``ud_{prod}`` in the last section.

Since ``M_1[i, k]`` is only well-defined if ``k`` is a valid column index and
``k - i`` is a valid diagonal index, we know that
```math
li_1 \\leq k \\leq ri_1 \\quad \\text{and} \\quad ld_1 \\leq k - i \\leq ud_1.
```
Since ``M_2[k, j]`` is only well-defined if ``j`` is a valid column index and
``j - k`` is a valid diagonal index, we also know that
```math
li_2 \\leq j \\leq ri_2 \\quad \\text{and} \\quad ld_2 \\leq j - k \\leq ud_2.
```
Finally, ``(M_1 * M_2)[i, j]`` is only well-defined if ``j - i`` is a valid
diagonal index, so
```math
ld_{prod} \\leq j - i \\leq ud_{prod}.
```
These inequalities can be combined to obtain
```math
\\begin{gather*}
\\text{max}(li_2, i + ld_{prod}) \\leq j \\leq
\\text{min}(ri_2, i + ud_{prod}) \\\\
\\text{and} \\\\
\\text{max}(li_1, i + ld_1, j - ud_2) \\leq k \\leq
\\text{min}(ri_1, i + ud_1, j - ld_2).
\\end{gather*}
```
So, we can rewrite the expression for ``(M_1 * M_2)[i, j]`` as
```math
\\begin{gather*}
(M_1 * M_2)[i, j] =
    \\sum_{
        k\\ =\\ \\text{max}(li_1, i + ld_1, j - ud_2)
    }^{\\text{min}(ri_1, i + ud_1, j - ld_2)}
    M_1[i, k] * M_2[k, j], \\text{ where} \\\\[0.5em]
\\text{max}(li_2, i + ld_{prod}) \\leq j \\leq \\text{min}(ri_2, i + ud_{prod}).
\\end{gather*}
```
If we replace the variable ``k`` with ``d = k - i``, replace the variable ``j``
with ``d_{prod} = j - i``, and switch from array-like indexing to `Field`
indexing, we find that
```math
\\begin{gather*}
(M_1 * M_2)[i][d_{prod}] =
    \\sum_{
        d\\ =\\ \\text{max}(li_1 - i, ld_1, d_{prod} - ud_2)
    }^{\\text{min}(ri_1 - i, ud_1, d_{prod} - ld_2)}
    M_1[i][d] * M_2[i + d][d_{prod} - d], \\text{ where} \\\\[0.5em]
\\text{max}(li_2 - i, ld_{prod}) \\leq d_{prod} \\leq
    \\text{min}(ri_2 - i, ud_{prod}).
\\end{gather*}
```

## 2.1 Interior vs. Boundary Indices

Now, suppose that the row index ``i`` is such that
```math
\\text{max}(li_1 - ld_1, li_2 - ld_{prod}) \\leq i \\leq
    \\text{min}(ri_1 - ud_1, ri_2 - ud_{prod}).
```
If this is the case, then the bounds on ``d_{prod}`` can be simplified to
```math
\\text{max}(li_2 - i, ld_{prod}) = ld_{prod} \\quad \\text{and} \\quad
\\text{min}(ri_2 - i, ud_{prod}) = ud_{prod}.
```
Similarly, the bounds on ``d`` can be simplified using the fact that
```math
\\text{max}(li_1 - i, ld_1) = ld_1 \\quad \\text{and} \\quad
\\text{min}(ri_1 - i, ud_1) = ud_1.
```
The expression for ``(M_1 * M_2)[i][d_{prod}]`` then becomes
```math
\\begin{gather*}
(M_1 * M_2)[i][d_{prod}] =
    \\sum_{
        d\\ =\\ \\text{max}(ld_1, d_{prod} - ud_2)
    }^{\\text{min}(ud_1, d_{prod} - ld_2)}
    M_1[i][d] * M_2[i + d][d_{prod} - d], \\text{ where} \\\\[0.5em]
ld_{prod} \\leq d_{prod} \\leq ud_{prod}.
\\end{gather*}
```
The values of ``i`` in this range are considered to be in the "interior" of the
operator, while those not in this range (for which we cannot make these
simplifications) are considered to be on the "boundary".

## 2.2 ``ld_{prod}`` and ``ud_{prod}``

We only need to compute ``(M_1 * M_2)[i][d_{prod}]`` for values of ``d_{prod}``
that correspond to a nonempty sum in the interior, i.e, those for which
```math
\\text{max}(ld_1, d_{prod} - ud_2) \\leq \\text{min}(ud_1, d_{prod} - ld_2).
```
This can be broken down into the four inequalities
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
In other words, the outer diagonal indices of ``M_1 * M_2`` are
```math
ld_{prod} = ld_1 + ld_2 \\quad \\text{and} \\quad ud_{prod} = ud_1 + ud_2.
```
This means that we can express the bounds on the interior values of ``i`` as
```math
\\text{max}(li_1, li_2 - ld_2) - ld_1 \\leq i \\leq
    \\text{min}(ri_1, ri_2 - ud_2) - ud_1.
```
"""
struct MultiplyColumnwiseBandMatrixField <: Operators.FiniteDifferenceOperator end

# TODO: Remove this in the next major release of ClimaCore.
const ⋅ = MultiplyColumnwiseBandMatrixField()

Operators.strip_space(op::MultiplyColumnwiseBandMatrixField, _) = op

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

function Operators.left_interior_idx(
    space::Spaces.AbstractSpace,
    ::MultiplyColumnwiseBandMatrixField,
    ::TopLeftMatrixCorner,
    matrix1,
    arg,
)
    column_space1 = column_axes(matrix1, space)
    li1 = Operators.left_idx(column_space1)
    ld1 = outer_diagonals(eltype(matrix1))[1]
    if eltype(arg) <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = arg
        column_space2 = column_axes(matrix2, column_space1)
        li2 = Operators.left_idx(column_space2)
        ld2 = outer_diagonals(eltype(matrix2))[1]
        return max(li1, li2 - ld2) - ld1
    else # matrix-vector multiplication
        return li1 - ld1
    end
end

function Operators.right_interior_idx(
    space::Spaces.AbstractSpace,
    ::MultiplyColumnwiseBandMatrixField,
    ::BottomRightMatrixCorner,
    matrix1,
    arg,
)
    column_space1 = column_axes(matrix1, space)
    ri1 = Operators.right_idx(column_space1)
    ud1 = outer_diagonals(eltype(matrix1))[2]
    if eltype(arg) <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = arg
        column_space2 = column_axes(matrix2, column_space1)
        ri2 = Operators.right_idx(column_space2)
        ud2 = outer_diagonals(eltype(matrix2))[2]
        return min(ri1, ri2 - ud2) - ud1
    else # matrix-vector multiplication
        return ri1 - ud1
    end
end

function Operators.return_eltype(
    ::MultiplyColumnwiseBandMatrixField,
    matrix1,
    arg,
)
    if (matrix1 isa Operators.StencilBroadcasted && matrix1.op isa FDOperatorMatrix && !(eltype(arg) <: BandMatrixRow))
       return Operators.return_eltype(matrix1.op.op, arg)
    end
    # if (matrix1 isa Base.Broadcast.Broadcasted  && matrix1.f isa FDOperatorMatrix)
    #    return eltype(matrix1)
    # end
    et_mat1 = eltype(matrix1)
    et_arg = eltype(arg)
    et_mat1 <: BandMatrixRow || error(
        "The first argument of MultiplyColumnwiseBandMatrixField must have
         elements of type BandMatrixRow, but the given argument has $et_mat1",
    )
    if et_arg <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = arg
        ld1, ud1 = outer_diagonals(et_mat1)
        ld2, ud2 = outer_diagonals(et_arg)
        prod_ld, prod_ud = ld1 + ld2, ud1 + ud2
        prod_value_type = rmul_return_type(eltype(et_mat1), eltype(et_arg))
        return band_matrix_row_type(prod_ld, prod_ud, prod_value_type)
    else # matrix-vector multiplication
        vector = arg
        return rmul_return_type(eltype(et_mat1), et_arg)
    end
end

function Operators.return_eltype(
    ::MultiplyColumnwiseBandMatrixField,
    matrix1,
    arg,
    ::Type{LG},
) where {LG}
    et_mat1 = eltype(matrix1)
    et_arg = eltype(arg)
    et_mat1 <: BandMatrixRow || error(
        "The first argument of MultiplyColumnwiseBandMatrixField must have
         elements of type BandMatrixRow, but the given argument has $et_mat1",
    )
    if et_arg <: BandMatrixRow # matrix-matrix multiplication
        matrix2 = arg
        ld1, ud1 = outer_diagonals(et_mat1)
        ld2, ud2 = outer_diagonals(et_arg)
        prod_ld, prod_ud = ld1 + ld2, ud1 + ud2
        prod_value_type = Base.promote_op(
            rmul_with_projection,
            eltype(et_mat1),
            eltype(et_arg),
            LG,
        )
        return band_matrix_row_type(prod_ld, prod_ud, prod_value_type)
    else # matrix-vector multiplication
        vector = arg
        prod_value_type =
            Base.promote_op(rmul_with_projection, eltype(et_mat1), et_arg, LG)
    end
end

Operators.return_space(::MultiplyColumnwiseBandMatrixField, space1, space2) =
    space1

# Compute max(li - i, ld).
boundary_modified_ld(_, ld, column_space, i) = ld
boundary_modified_ld(::TopLeftMatrixCorner, ld, column_space, i) =
    max(Operators.left_idx(column_space) - i, ld)

# Compute min(ri - i, ud).
boundary_modified_ud(_, ud, column_space, i) = ud
boundary_modified_ud(::BottomRightMatrixCorner, ud, column_space, i) =
    min(Operators.right_idx(column_space) - i, ud)

# TODO: Use @propagate_inbounds here, and remove @inbounds from this function.
# As of Julia 1.8, doing this increases compilation time by more than an order
# of magnitude, and it also makes type inference fail for some complicated
# matrix field broadcast expressions (in particular, those that involve products
# of linear combinations of matrix fields). Not using @propagate_inbounds causes
# matrix field broadcast expressions to take roughly 3 or 4 times longer to
# evaluate, but this is less significant than the decrease in compilation time.
# matrix-matrix multiplication
function multiply_matrix_at_index(
    space,
    idx,
    hidx,
    matrix1,
    arg,
    bc,
    ::Type{T},
) where {T <: BandMatrixRow}
    # T = eltype(arg)
    lg = Geometry.LocalGeometry(space, idx, hidx)
    prod_type = Operators.return_eltype(
        MultiplyColumnwiseBandMatrixField(),
        matrix1,
        arg,
        typeof(lg),
    )

    column_space1 = column_axes(matrix1, space)
    ld1, ud1 = outer_diagonals(eltype(matrix1))
    boundary_modified_ld1 = boundary_modified_ld(bc, ld1, column_space1, idx)
    boundary_modified_ud1 = boundary_modified_ud(bc, ud1, column_space1, idx)

    # Precompute the row that is needed from matrix1 so that it does not get
    # recomputed multiple times.
    matrix1_row = @inbounds Operators.getidx(space, matrix1, idx, hidx)

    matrix2 = arg
    column_space2 = column_axes(matrix2, column_space1)
    ld2, ud2 = outer_diagonals(eltype(matrix2))
    prod_ld, prod_ud = outer_diagonals(prod_type)
    boundary_modified_prod_ld =
        boundary_modified_ld(bc, prod_ld, column_space2, idx)
    boundary_modified_prod_ud =
        boundary_modified_ud(bc, prod_ud, column_space2, idx)

    # Precompute the rows that are needed from matrix2 so that they do not
    # get recomputed multiple times. To avoid inference issues at boundary
    # points, this is implemented as a padded map from ld1 to ud1, instead
    # of as a map from boundary_modified_ld1 to boundary_modified_ud1. For
    # simplicity, use zero padding for rows that are outside the matrix.
    # Wrap the rows in a BandMatrixRow so that they can be easily indexed.
    matrix2_rows = unrolled_map((ld1:ud1...,)) do d
        # TODO: Use @propagate_inbounds_meta instead of @inline_meta.
        Base.@_inline_meta
        if isnothing(bc) || boundary_modified_ld1 <= d <= boundary_modified_ud1
            @inbounds Operators.getidx(space, matrix2, idx + d, hidx)
        else
            zero(eltype(matrix2)) # This row is outside the matrix.
        end
    end
    matrix2_rows_wrapper = BandMatrixRow{ld1}(matrix2_rows...)

    # Precompute the zero value to avoid inference issues caused by passing
    # prod_type into the function closure below.
    zero_value = rzero(eltype(prod_type))

    # Compute the entries of the product matrix row. To avoid inference
    # issues at boundary points, this is implemented as a padded map from
    # prod_ld to prod_ud, instead of as a map from boundary_modified_prod_ld
    # to boundary_modified_prod_ud. For simplicity, use zero padding for
    # entries that are outside the matrix. Wrap the entries in a
    # BandMatrixRow before returning them.
    prod_entries = map((prod_ld:prod_ud...,)) do prod_d
        # TODO: Use @propagate_inbounds_meta instead of @inline_meta.
        Base.@_inline_meta
        if isnothing(bc) ||
           boundary_modified_prod_ld <= prod_d <= boundary_modified_prod_ud
            prod_entry = zero_value
            min_d = max(boundary_modified_ld1, prod_d - ud2)
            max_d = min(boundary_modified_ud1, prod_d - ld2)
            @inbounds for d in min_d:max_d
                value1 = matrix1_row[d]
                value2 = matrix2_rows_wrapper[d][prod_d - d]
                value2_lg = Geometry.LocalGeometry(space, idx + d, hidx)
                prod_entry = radd(
                    prod_entry,
                    rmul_with_projection(value1, value2, value2_lg),
                )
            end # Using a for-loop is currently faster than using mapreduce.
            prod_entry
        else
            zero_value # This entry is outside the matrix.
        end
    end
    return BandMatrixRow{prod_ld}(prod_entries...)
end
# matrix-vector multiplication
function multiply_matrix_at_index(
    space,
    idx,
    hidx,
    matrix1,
    arg,
    bc,
    ::Type{T},
) where {T}
    # T = eltype(arg)
    lg = Geometry.LocalGeometry(space, idx, hidx)
    prod_type = Operators.return_eltype(
        MultiplyColumnwiseBandMatrixField(),
        matrix1,
        arg,
        typeof(lg),
    )

    column_space1 = column_axes(matrix1, space)
    ld1, ud1 = outer_diagonals(eltype(matrix1))
    boundary_modified_ld1 = boundary_modified_ld(bc, ld1, column_space1, idx)
    boundary_modified_ud1 = boundary_modified_ud(bc, ud1, column_space1, idx)

    # Precompute the row that is needed from matrix1 so that it does not get
    # recomputed multiple times.
    matrix1_row = @inbounds Operators.getidx(space, matrix1, idx, hidx)

    vector = arg
    prod_value = rzero(prod_type)
    @inbounds for d in boundary_modified_ld1:boundary_modified_ud1
        value1 = matrix1_row[d]
        value2 = Operators.getidx(space, vector, idx + d, hidx)
        value2_lg = Geometry.LocalGeometry(space, idx + d, hidx)
        prod_value =
            radd(prod_value, rmul_with_projection(value1, value2, value2_lg))
    end # Using a for-loop is currently faster than using mapreduce.
    return prod_value
end

Base.@propagate_inbounds Operators.stencil_interior(
    ::MultiplyColumnwiseBandMatrixField,
    space,
    idx,
    hidx,
    matrix1,
    arg,
) = multiply_matrix_at_index(
    space,
    idx,
    hidx,
    matrix1,
    arg,
    nothing,
    eltype(arg),
)

Base.@propagate_inbounds Operators.stencil_left_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    bc::TopLeftMatrixCorner,
    space,
    idx,
    hidx,
    matrix1,
    arg,
) = multiply_matrix_at_index(space, idx, hidx, matrix1, arg, bc, eltype(arg))

Base.@propagate_inbounds Operators.stencil_right_boundary(
    ::MultiplyColumnwiseBandMatrixField,
    bc::BottomRightMatrixCorner,
    space,
    idx,
    hidx,
    matrix1,
    arg,
) = multiply_matrix_at_index(space, idx, hidx, matrix1, arg, bc, eltype(arg))

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
