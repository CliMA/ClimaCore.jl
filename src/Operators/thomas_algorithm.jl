"""
    column_thomas_solve!(A, b)

Solves the linear system `A * x = b`, where `A` is a tri-diagonal matrix
(represented by a `Field` of tri-diagonal matrix rows), and where `b` is a
vector (represented by a `Field` of numbers). The data in `b` is overwritten
with the solution `x`, and the upper diagonal of `A` is also overwritten with
intermediate values used to compute `x`. The algorithm is described here:
https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm.
"""
column_thomas_solve!(A, b) =
    column_thomas_solve!(ClimaComms.device(axes(A)), A, b)

column_thomas_solve!(::ClimaComms.AbstractCPUDevice, A, b) =
    thomas_algorithm!(A, b)

thomas_algorithm_kernel!(
    A::Fields.FiniteDifferenceField,
    b::Fields.FiniteDifferenceField,
    us::DataLayouts.UniversalSize,
) = thomas_algorithm!(A, b)

function thomas_algorithm!(
    A::Fields.ExtrudedFiniteDifferenceField,
    b::Fields.ExtrudedFiniteDifferenceField,
)
    Fields.bycolumn(axes(A)) do colidx
        thomas_algorithm!(A[colidx], b[colidx])
    end
end

function thomas_algorithm!(
    A::Fields.FiniteDifferenceField,
    b::Fields.FiniteDifferenceField,
)
    nrows = Spaces.nlevels(axes(A))
    lower_diag = A.coefs.:1
    main_diag = A.coefs.:2
    upper_diag = A.coefs.:3

    # first row
    denominator = _getindex(main_diag, 1)
    _setindex!(upper_diag, 1, _getindex(upper_diag, 1) / denominator)
    _setindex!(b, 1, _getindex(b, 1) / denominator)

    # interior rows
    for row in 2:(nrows - 1)
        numerator =
            _getindex(b, row) -
            _getindex(lower_diag, row) * _getindex(b, row - 1)
        denominator =
            _getindex(main_diag, row) -
            _getindex(lower_diag, row) * _getindex(upper_diag, row - 1)
        _setindex!(upper_diag, row, _getindex(upper_diag, row) / denominator)
        _setindex!(b, row, numerator / denominator)
    end

    # last row
    numerator =
        _getindex(b, nrows) -
        _getindex(lower_diag, nrows) * _getindex(b, nrows - 1)
    denominator =
        _getindex(main_diag, nrows) -
        _getindex(lower_diag, nrows) * _getindex(upper_diag, nrows - 1)
    _setindex!(b, nrows, numerator / denominator)

    # back substitution
    # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
    # for row in (nrows - 1):-1:1; # results in StepRange
    row = (nrows - 1)
    while row ≥ 1
        value =
            _getindex(b, row) -
            _getindex(upper_diag, row) * _getindex(b, row + 1)
        _setindex!(b, row, value)
        row -= 1
    end
end

# This is the same as @inbounds Fields.field_values(column_field)[index]
_getindex(column_field, index) = @inbounds Operators.getidx(
    axes(column_field),
    column_field,
    Operators.Interior(),
    index - 1 + Operators.left_idx(axes(column_field)),
)

# This is the same as @inbounds Fields.field_values(column_field)[index] = value
_setindex!(column_field, index, value) = @inbounds Operators.setidx!(
    axes(column_field),
    column_field,
    index - 1 + Operators.left_idx(axes(column_field)),
    (1, 1, 1),
    value,
)
