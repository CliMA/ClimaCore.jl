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

function Operators.return_eltype(
    ::MultiplyColumnwiseBandMatrixField,
    matrix1,
    arg,
)
    et_mat1 = eltype(matrix1)
    et_arg = eltype(arg)
    # eltype may be the inference-failure sentinel Union{} when this is called
    # while probing an expression with unsafe_eltype; propagate it instead of
    # treating it as a BandMatrixRow (Union{} is a subtype of everything).
    (et_mat1 == Union{} || et_arg == Union{}) && return Union{}
    et_mat1 <: BandMatrixRow || invalid_matrix_eltype_error(et_mat1)
    # The entries of arg are already projected for the multiplication (see
    # projected_operand).
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

# The message is built when the method is generated, since GPU kernels cannot
# build strings at run time.
@generated invalid_matrix_eltype_error(::Type{T}) where {T} = :(error(
    $("The first argument of MultiplyColumnwiseBandMatrixField must have \
       elements of type BandMatrixRow, but the given argument has $T"),
))

Operators.return_space(::MultiplyColumnwiseBandMatrixField, matrix1, _) =
    axes(matrix1)

"""
    projected_operand(matrix1, arg)

Return the second argument of the product of `matrix1` and `arg`. When the entries of
`matrix1` need the values of `arg` to be projected (see
`Geometry._dual_axes_for_projection`), this is a pointwise broadcast expression that
projects them with `Geometry.project_for_mul`, reading only the metric that the
projection needs from the local geometry of `matrix1`'s column space (see
`Geometry.projection_metric`). Otherwise, it is just `arg`.

The rows of a product multiply entries with `*`, and each value of `arg` they read is
projected wherever it is evaluated. Since the projection is part of the argument, a
stencil that evaluates the argument once per point also projects it once per point,
instead of once per read.

The values that a `SetBoundaryOperator` imposes can have other types than its argument
(e.g., a `Contravariant3Vector` imposed on a `Covariant3Vector` flux), so the eltype of
an `arg` that contains one does not determine the metrics its values need. Such an `arg`
is always projected, with the metric that covers its eltype and every imposed value type
(see `Operators.imposed_values_metric`): the one metric that some of these types need
when the others need none, or all of the local geometry when they need different ones.
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
    # The projection is skipped when it does not change the type of any value
    # of arg, i.e., when it does not change the type of its eltype and every
    # imposed value also has that type (see Operators.imposes_only). An arg with
    # imposed values that is cached (see Operators.is_cached_arg) keeps the
    # projection, so that its values are imposed once, when it is cached, rather
    # than at every read.
    return eltype(projected_arg) == et_arg && (
        !has_boundary_values ||
        Operators.imposes_only(arg, value_type) && !Operators.is_cached_arg(arg)
    ) ? arg : projected_arg
end

# Projection of every value of the second argument of a product, which is mapped
# over the entries of each row in matrix-matrix products.
struct ProjectForMul{A}
    axes::A
end
@inline (f::ProjectForMul)(value, lg) = Geometry.project_for_mul(f.axes, value, lg)
@inline (f::ProjectForMul)(row::BandMatrixRow, lg) = map(value -> f(value, lg), row)

# A projection that reads the local geometry is computed once per point, rather
# than at every point that reads it (see Operators.recomputable); a projection
# without a metric (the Fix2 form above) only changes axes and can be recomputed.
Operators.recomputable_node(::ProjectForMul) = false

# Each row of a product only reads the same row of matrix1, and it reads the
# second argument at the same offsets from every row (see multiply_matrix_row).
Operators.reads_neighbors(::MultiplyColumnwiseBandMatrixField, _, _) =
    (Val(false), Val(true))
Operators.reads_in_lockstep(::MultiplyColumnwiseBandMatrixField) = true

# Whether the band ld:ud of every row of a matrix with rows in space lies inside
# column_space, which holds when the bands of the first and last rows do (a row
# is only ever evaluated at an index of its space). This is the case for the
# face-to-center operator matrices (their bands reach at most one face past each
# center), but never for center-to-face ones, whose first and last rows reach a
# center outside of the column. Unlike Operators.in_column, this does not depend
# on the row index, so it folds at compile time, which lets a row read its whole
# band without branching on each entry (and lets the loads of the entries
# overlap).
@inline band_in_column(space, column_space, ld, ud) =
    Operators.in_column(column_space, Operators.left_idx(space) + ld) &&
    Operators.in_column(column_space, Operators.right_idx(space) + ud)

# Every row of a product is computed over the full bands of its factors, with
# zeros in place of entries that lie outside of the matrices, so that the rows
# near the boundaries need no stencils of their own. When the second argument
# requires lockstep reads (see Operators.requires_lockstep), it is read at every
# offset in the band, at the closest index in the column, so that every row reads
# it in lockstep; otherwise, the second argument of a matrix-vector product is
# only read inside the column. The rows of the second matrix of a matrix-matrix
# product are always read at the closest index in the column, and only replaced
# by zeros after they are read, so that the reads of a row's band need no
# branches (a branch around each read serializes the reads on GPUs).
Base.@propagate_inbounds Operators.stencil_value(
    ::MultiplyColumnwiseBandMatrixField,
    space,
    idx,
    matrix1,
    arg,
) = multiply_matrix_row(space, idx, matrix1, arg, eltype(arg))

# Row with index idx of a matrix-matrix product. Every row is inlined into the
# point loop that reads it, including the rows of products with an operand that
# contains another product (three or more band matrix factors per row): a row
# behind a function barrier costs a call per point, which also recomputes the
# column offsets of every factor, and makes the products of three factors 2.5
# to 5.5 times slower on CPUs.
Base.@propagate_inbounds multiply_matrix_row(
    space,
    idx,
    matrix1,
    matrix2,
    ::Type{<:BandMatrixRow},
) = matrix_matrix_row(space, idx, matrix1, matrix2)

# The value of arg (a vector or a matrix of band rows) at band entry d of the
# row at idx, read at an index clamped into the column, and whether the entry
# lies in the column. The read is clamped rather than skipped so that it is
# never moved into a branch (see Operators.requires_lockstep); the caller
# replaces the values outside of the column by zeros.
Base.@propagate_inbounds function clamped_band_entry(
    arg,
    space,
    column_space,
    idx,
    d,
    all_in_column,
)
    in_column = all_in_column || Operators.in_column(column_space, idx + d)
    read_idx = all_in_column ? idx + d : Operators.column_index(column_space, idx + d)
    return (Operators.column_value(arg, space, read_idx), in_column)
end

Base.@propagate_inbounds function matrix_matrix_row(space, idx, matrix1, matrix2)
    prod_type = Operators.return_eltype(
        MultiplyColumnwiseBandMatrixField(),
        matrix1,
        matrix2,
    )
    column_space1 = column_axes(matrix1, space)
    column_space2 = column_axes(matrix2, column_space1)
    (ld1, ud1) = outer_diagonals(eltype(matrix1))
    (ld2, ud2) = outer_diagonals(eltype(matrix2))
    (prod_ld, prod_ud) = outer_diagonals(prod_type)
    all_in_column1 = band_in_column(space, column_space1, ld1, ud1)

    # Read each row of matrix1 and matrix2 once, using zeros for the entries of
    # matrix1 and the rows of matrix2 that lie outside of the matrix (the entries
    # outside of matrix1 can be NaNs, which would not vanish when multiplied by
    # zero rows of matrix2).
    matrix1_row = Operators.column_value(matrix1, space, idx)
    zero_entry1 = zero(eltype(eltype(matrix1)))
    matrix1_entries = unrolled_map((ld1:ud1...,)) do d
        all_in_column1 || Operators.in_column(column_space1, idx + d) ?
        matrix1_row[d] : zero_entry1
    end
    matrix1_row_wrapper = BandMatrixRow{ld1}(matrix1_entries...)
    matrix2_rows = unrolled_map((ld1:ud1...,)) do d
        Base.@_propagate_inbounds_meta
        (row, in_column1) =
            clamped_band_entry(matrix2, space, column_space1, idx, d, all_in_column1)
        ifelse(in_column1, row, zero(eltype(matrix2)))
    end
    matrix2_rows_wrapper = BandMatrixRow{ld1}(matrix2_rows...)

    # Precompute the zero value to avoid inference issues caused by passing
    # prod_type into the closure below.
    zero_value = zero(eltype(prod_type))
    all_in_column2 = band_in_column(space, column_space2, prod_ld, prod_ud)
    # Every entry is computed before the entries outside of the matrix are
    # replaced by zeros, so that no read of matrix2_rows is moved into a branch.
    prod_entries = map((prod_ld:prod_ud...,)) do prod_d
        Base.@_propagate_inbounds_meta
        prod_entry = zero_value
        for d in max(ld1, prod_d - ud2):min(ud1, prod_d - ld2)
            value1 = matrix1_row_wrapper[d]
            value2 = matrix2_rows_wrapper[d][prod_d - d]
            prod_entry += value1 * value2
        end # Using a for-loop is currently faster than using mapreduce.
        in_column2 =
            all_in_column2 || Operators.in_column(column_space2, idx + prod_d)
        ifelse(in_column2, prod_entry, zero_value)
    end
    return BandMatrixRow{prod_ld}(prod_entries...)
end

# Row with index idx of a matrix-vector product.
Base.@propagate_inbounds function multiply_matrix_row(
    space,
    idx,
    matrix1,
    vector,
    ::Type,
)
    (idx, matrix1) = replicated_product_row(space, idx, matrix1)
    prod_type = Operators.return_eltype(
        MultiplyColumnwiseBandMatrixField(),
        matrix1,
        vector,
    )
    column_space1 = column_axes(matrix1, space)
    (ld1, ud1) = outer_diagonals(eltype(matrix1))
    all_in_column = band_in_column(space, column_space1, ld1, ud1)
    matrix1_row = Operators.column_value(matrix1, space, idx)
    zero_value = zero(prod_type)
    prod_terms = unrolled_map((ld1:ud1...,)) do d
        Base.@_propagate_inbounds_meta
        (value2, in_column) =
            clamped_band_entry(vector, space, column_space1, idx, d, all_in_column)
        # The product is formed before the entries outside of the column are
        # replaced by zeros (the entries of matrix1 there can be NaNs).
        ifelse(in_column, matrix1_row[d] * value2, zero_value)
    end
    return unrolled_reduce(+, prod_terms; init = zero_value)
end
