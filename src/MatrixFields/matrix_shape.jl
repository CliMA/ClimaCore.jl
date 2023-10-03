abstract type AbstractMatrixShape end
struct Square <: AbstractMatrixShape end
struct FaceToCenter <: AbstractMatrixShape end
struct CenterToFace <: AbstractMatrixShape end

"""
    matrix_shape(matrix_field, [matrix_space])

Returns either `Square()`, `FaceToCenter()`, or `CenterToFace()`, depending on
whether the diagonal indices of `matrix_field` are `Int`s or `PlusHalf`s and
whether `matrix_space` is on cell centers or cell faces. By default,
`matrix_space` is set to `axes(matrix_field)`.
"""
matrix_shape(matrix_field, matrix_space = axes(matrix_field)) = _matrix_shape(
    eltype(outer_diagonals(eltype(matrix_field))),
    matrix_space.staggering,
)

_matrix_shape(::Type{Int}, _) = Square()
_matrix_shape(::Type{PlusHalf{Int}}, ::Spaces.CellCenter) = FaceToCenter()
_matrix_shape(::Type{PlusHalf{Int}}, ::Spaces.CellFace) = CenterToFace()

"""
    column_axes(matrix_field, [matrix_space])

Returns the space that corresponds to the columns of `matrix_field`, i.e., the
`axes` of the `Field`s by which `matrix_field` can be multiplied. The
`matrix_space`, on the other hand, is the space that corresponds to the rows of
`matrix_field`. By default, `matrix_space` is set to `axes(matrix_field)`.
"""
column_axes(matrix_field, matrix_space = axes(matrix_field)) =
    _column_axes(matrix_shape(matrix_field, matrix_space), matrix_space)

_column_axes(::Square, space) = space
_column_axes(::FaceToCenter, space) = Operators.reconstruct_placeholder_space(
    Operators.FacePlaceholderSpace(),
    space,
)
_column_axes(::CenterToFace, space) = Operators.reconstruct_placeholder_space(
    Operators.CenterPlaceholderSpace(),
    space,
)
