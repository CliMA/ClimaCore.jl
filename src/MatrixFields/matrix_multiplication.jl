"""
    MultiplyColumnwiseBandMatrixField()

Operator that multiplies a `ColumnwiseBandMatrixField` by another `Field`, i.e.,
matrix-vector or matrix-matrix multiplication.

What follows is a derivation of the algorithm used by this operator with
single-column `Field`s. For `Field`s on multiple columns, the same computation
is done for each column.

In this derivation, we will use ``M_1`` and ``M_2`` to denote two
`ColumnwiseBandMatrixField`s, and we will use ``V`` to denote a regular
(vector-like) `Field`. For both ``M_1`` and ``M_2``, we will use the array-like
index notation ``M[row, col]`` to denote ``M[row][col-row]``, i.e., the entry in
the `BandMatrixRow` ``M[row]`` located on the diagonal with index ``col - row``.
We will also use `outer_indices`(`````space```)`` to denote the tuple ``(```left_idx`````(`space`), `right_idx`(`space```))``.

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

function Operators.return_eltype(::MultiplyColumnwiseBandMatrixField, matrix1, arg)
    et_mat1 = eltype(matrix1)
    et_arg = eltype(arg)
    # eltype may be the inference-failure sentinel Union{} when this is called
    # while probing an expression with unsafe_eltype; propagate it instead of
    # treating it as a BandMatrixRow (Union{} is a subtype of everything).
    (et_mat1 == Union{} || et_arg == Union{}) && return Union{}
    et_mat1 <: BandMatrixRow || invalid_matrix_eltype_error(et_mat1)
    # The entries of arg are already projected (see projected_operand).
    if et_arg <: BandMatrixRow # matrix-matrix multiplication
        ld1, ud1 = outer_diagonals(et_mat1)
        ld2, ud2 = outer_diagonals(et_arg)
        prod_ld, prod_ud = ld1 + ld2, ud1 + ud2
        prod_value_type = return_type(*, Tuple{eltype(et_mat1), eltype(et_arg)})
        return band_matrix_row_type(prod_ld, prod_ud, prod_value_type)
    else # matrix-vector multiplication
        return return_type(*, Tuple{eltype(et_mat1), et_arg})
    end
end

# The message is built at generation time, since GPU kernels cannot build strings.
@generated invalid_matrix_eltype_error(::Type{T}) where {T} =
    :(error($("The first argument of MultiplyColumnwiseBandMatrixField must have \
               elements of type BandMatrixRow, but the given argument has $T")))

# A product is on the staggering of matrix1, over the larger of the spaces of its arguments
# (e.g., a matrix on one column multiplies every column of an extruded field). A matrix band
# with half-integer offsets maps values from the other staggering.
function Operators.return_space(::MultiplyColumnwiseBandMatrixField, matrix1, arg)
    space1 = Fields.shared_space(matrix1)
    staggering = Spaces.staggering(space1)
    isnothing(staggering) && return space1
    staggers = first(outer_diagonals(eltype(matrix1))) isa PlusHalf
    arg_staggering =
        staggers == (staggering isa Spaces.CellCenter) ? Spaces.CellFace : Spaces.CellCenter
    Operators.check_staggering(arg, arg_staggering)
    space2 = Fields.staggered_like(Fields.shared_space(arg), space1)
    return Spaces.maybe_issubspace(space1, space2) ? space2 : space1
end

"""
    projected_operand(matrix1, arg)

Return the second argument of the product of `matrix1` and `arg`, which projects the values
of `arg` with `Geometry.project_for_mul` (once per evaluation) when the entries of `matrix1`
need it (see `Geometry._dual_axes_for_projection`), reading only the needed metric. This
also covers values that a `SetBoundaryOperator` in `arg` imposes, whose types can differ
from its eltype (see `Operators.imposed_values_metric`).
"""
function projected_operand(matrix1, arg)
    et_mat1 = eltype(matrix1)
    (et_mat1 == Union{} || !(et_mat1 <: BandMatrixRow)) && return arg
    axes = Geometry._dual_axes_for_projection(eltype(et_mat1))
    isnothing(axes) && return arg
    et_arg = eltype(Base.Broadcast.broadcastable(arg))
    et_arg == Union{} && return arg
    value_type = et_arg <: BandMatrixRow ? eltype(et_arg) : et_arg
    lg_field = Fields.local_geometry_field(column_axes(matrix1))
    has_boundary_values = Operators.has_set_boundary_operator(arg)
    value_metric = Geometry.projection_metric(axes, value_type, lg_field)
    metric = Geometry.combine_projected_metrics(
        isnothing(value_metric) ? nothing : Some(value_metric),
        Operators.imposed_values_metric(axes, arg, lg_field),
        lg_field,
    )
    metric = isnothing(metric) ? nothing : something(metric)
    projected_arg =
        isnothing(metric) ?
        Base.Broadcast.broadcasted(Base.Fix2(ProjectForMul(axes), nothing), arg) :
        Base.Broadcast.broadcasted(ProjectForMul(axes), arg, metric)
    # Skip the projection if it changes no value's type (see Operators.imposes_only), unless
    # a cached arg imposes values, which it then imposes once instead of at every read.
    return eltype(projected_arg) == et_arg && (
        !has_boundary_values ||
        Operators.imposes_only(arg, value_type) && !Operators.is_cached_arg(arg)
    ) ? arg : projected_arg
end

# Projection of every value of the second argument of a product (mapped over row entries).
struct ProjectForMul{A}
    axes::A
end
@inline (f::ProjectForMul)(value, lg) = Geometry.project_for_mul(f.axes, value, lg)
@inline (f::ProjectForMul)(row::BandMatrixRow, lg) = map(value -> f(value, lg), row)

# A projection that reads the local geometry is not recomputed (the Fix2 form above can be).
Operators.recomputable_node(::ProjectForMul) = false

# Rows read only their own row of matrix1, and the second argument at fixed offsets.
Operators.reads_neighbors(::MultiplyColumnwiseBandMatrixField, _, _) =
    (Val(false), Val(true))
Operators.reads_without_branching(::MultiplyColumnwiseBandMatrixField) = true

# Whether every row band of a matrix in space lies in column_space (folds at compile time).
@inline band_in_column(space, column_space, ld, ud) =
    Operators.is_valid_column_index(column_space, Operators.left_idx(space) + ld) &&
    Operators.is_valid_column_index(column_space, Operators.right_idx(space) + ud)

# Every row of a product spans the full bands of its factors, with zeros outside of the
# matrices, so rows need no boundary stencils; the second argument is read at every offset,
# at an index clamped into the column (a branch would serialize GPU reads), and zeroed
# afterwards, keeping reads branch-free.
Base.@propagate_inbounds function Operators.stencil_value(
    ::MultiplyColumnwiseBandMatrixField,
    space,
    idx,
    matrix1,
    arg,
)
    is_matrix_product = eltype(arg) <: BandMatrixRow
    (idx, matrix1) =
        is_matrix_product ? (idx, matrix1) : replicated_product_row(space, idx, matrix1)
    prod_type = Operators.return_eltype(MultiplyColumnwiseBandMatrixField(), matrix1, arg)
    column_space1 = column_axes(matrix1, space)
    (ld1, ud1) = outer_diagonals(eltype(matrix1))
    all_in_column1 = band_in_column(space, column_space1, ld1, ud1)
    matrix1_row = Operators.column_value(matrix1, space, idx)
    # Each entry's row of arg and whether it is in the column, or its product for vectors.
    band_values = unrolled_map((ld1:ud1...,)) do d
        Base.@_propagate_inbounds_meta
        in_column1 =
            all_in_column1 || Operators.is_valid_column_index(column_space1, idx + d)
        read_idx =
            all_in_column1 ? idx + d :
            Operators.clamped_column_index(column_space1, idx + d)
        value2 = Operators.column_value(arg, space, read_idx)
        # Multiply before zeroing entries outside of the column (matrix1 can have NaNs).
        value2 isa BandMatrixRow ? (value2, in_column1) :
        ifelse(in_column1, matrix1_row[d] * value2, zero(prod_type))
    end
    is_matrix_product || return unrolled_reduce(+, band_values; init = zero(prod_type))
    column_space2 = column_axes(arg, column_space1)
    (ld2, ud2) = outer_diagonals(eltype(arg))
    (prod_ld, prod_ud) = outer_diagonals(prod_type)

    # Read each row into registers, with zeros for entries outside the matrices (every entry
    # of the rows of matrix2 contributes to some entry of the product).
    matrix1_values = unrolled_map(band_values, (ld1:ud1...,)) do (_, in_column1), d
        ifelse(in_column1, matrix1_row[d], zero(eltype(eltype(matrix1))))
    end
    matrix2_rows = unrolled_map(band_values) do (row, in_column1)
        ifelse(in_column1, row, zero(eltype(arg)))
    end

    # Set out-of-bounds values to zero after computing them, keeping the product branch-free.
    zero_value = zero(eltype(prod_type))
    all_in_column2 = band_in_column(space, column_space2, prod_ld, prod_ud)
    prod_entries = unrolled_map((prod_ld:prod_ud...,)) do prod_d
        Base.@_propagate_inbounds_meta
        prod_entry = zero_value
        for d in max(ld1, prod_d - ud2):min(ud1, prod_d - ld2)
            value1 = matrix1_values[d - ld1 + 1]
            value2 = matrix2_rows[d - ld1 + 1][prod_d - d]
            prod_entry += value1 * value2
        end
        in_column2 =
            all_in_column2 || Operators.is_valid_column_index(column_space2, idx + prod_d)
        ifelse(in_column2, prod_entry, zero_value)
    end
    return BandMatrixRow{prod_ld}(prod_entries...)
end
