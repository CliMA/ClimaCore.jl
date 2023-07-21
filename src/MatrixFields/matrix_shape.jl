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
